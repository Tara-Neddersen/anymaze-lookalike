from __future__ import annotations

import dataclasses
import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .database import JobDatabase
from .engines import EngineSpec, run_engine
from .metrics import compute_all_metrics, to_per_frame_csv, to_summary_csv
from .models import AnimalMeta, JobStatus


@dataclass
class JobRecord:
    status: JobStatus
    video_path: str
    result_path: str
    csv_perframe_path: str
    csv_summary_path: str
    engine: EngineSpec
    meta: AnimalMeta = field(default_factory=AnimalMeta)


class JobStore:
    """
    In-process job store with one-thread-per-job execution and live progress.
    Results persist to SQLite so previous sessions survive server restarts.
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.uploads_dir = self.data_dir / "uploads"
        self.jobs_dir = self.data_dir / "jobs"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._db = JobDatabase(str(self.data_dir / "neurotrack.db"))

        # Reload completed jobs from DB on startup
        self._reload_from_db()

    # ------------------------------------------------------------------
    # DB reload on startup
    # ------------------------------------------------------------------

    def _reload_from_db(self) -> None:
        """Load all previously completed jobs back into memory from SQLite."""
        valid_fields = {f.name for f in dataclasses.fields(EngineSpec)}
        for row in self._db.all_jobs():
            job_id = row["id"]
            db_status = row["status"]
            # Mark any interrupted running jobs as error
            if db_status == "running":
                self._db.update(job_id, "error", 1.0, "Server restarted while running")
                db_status = "error"
            # Only load done jobs (error/queued jobs are not actionable on restart)
            if db_status != "done":
                continue
            # Skip if result file is gone
            if not Path(row["result_path"]).exists():
                continue
            try:
                engine_data = json.loads(row["engine_json"])
                # Filter to only known EngineSpec fields (handles schema migrations)
                safe_engine = {k: v for k, v in engine_data.items() if k in valid_fields}
                spec = EngineSpec(**safe_engine)
                meta_data = json.loads(row["meta_json"])
                meta = AnimalMeta(**{k: v for k, v in meta_data.items()
                                     if k in AnimalMeta.model_fields})
                status = JobStatus(id=job_id, status="done", progress=1.0)
                rec = JobRecord(
                    status=status,
                    video_path=row["video_path"],
                    result_path=row["result_path"],
                    csv_perframe_path=row["csv_perframe_path"],
                    csv_summary_path=row["csv_summary_path"],
                    engine=spec,
                    meta=meta,
                )
                with self._lock:
                    self._jobs[job_id] = rec
            except Exception:
                pass  # Skip corrupted records

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(self, upload_path: str, engine: EngineSpec, meta: AnimalMeta | None = None) -> JobStatus:
        job_id = uuid.uuid4().hex
        job_folder = self.jobs_dir / job_id
        job_folder.mkdir(parents=True, exist_ok=True)

        effective_meta = meta or AnimalMeta()
        status = JobStatus(id=job_id, status="queued", message=None, progress=0.0)
        rec = JobRecord(
            status=status,
            video_path=upload_path,
            result_path=str(job_folder / "result.json"),
            csv_perframe_path=str(job_folder / "perframe.csv"),
            csv_summary_path=str(job_folder / "summary.csv"),
            engine=engine,
            meta=effective_meta,
        )
        with self._lock:
            self._jobs[job_id] = rec

        # Persist to SQLite
        self._db.insert(
            job_id=job_id,
            video_path=upload_path,
            result_path=rec.result_path,
            csv_perframe=rec.csv_perframe_path,
            csv_summary=rec.csv_summary_path,
            engine_json=json.dumps(dataclasses.asdict(engine)),
            meta_json=effective_meta.model_dump_json(),
            video_filename=Path(upload_path).name,
        )

        t = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        t.start()
        return status

    def all_jobs_summary(self) -> list[dict[str, Any]]:
        """Return lightweight list of all jobs for the frontend session manager."""
        db_rows = self._db.all_jobs()
        out = []
        for row in db_rows:
            job_id = row["id"]
            with self._lock:
                mem_rec = self._jobs.get(job_id)
            status = mem_rec.status.status if mem_rec else row["status"]
            progress = mem_rec.status.progress if mem_rec else float(row.get("progress") or 0)
            message = mem_rec.status.message if mem_rec else row.get("message")
            try:
                meta = json.loads(row.get("meta_json") or "{}")
            except Exception:
                meta = {}
            out.append({
                "job_id": job_id,
                "status": status,
                "progress": progress,
                "message": message,
                "animal_id": meta.get("animal_id", ""),
                "treatment": meta.get("treatment", ""),
                "trial": meta.get("trial", ""),
                "session": meta.get("session", ""),
                "experiment_id": meta.get("experiment_id", ""),
                "experimenter": meta.get("experimenter", ""),
                "video_filename": row.get("video_filename", ""),
                "created_at": row.get("created_at", ""),
            })
        return out

    def delete_job(self, job_id: str) -> bool:
        with self._lock:
            rec = self._jobs.pop(job_id, None)
        if rec is None:
            return False
        self._db.delete(job_id)
        for path in [rec.result_path, rec.csv_perframe_path, rec.csv_summary_path]:
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass
        return True

    def get_status(self, job_id: str) -> JobStatus | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.status if rec else None

    def get_result_path(self, job_id: str) -> str | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.result_path if rec else None

    def get_csv_path(self, job_id: str, csv_type: str) -> str | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return None
            return rec.csv_perframe_path if csv_type == "perframe" else rec.csv_summary_path

    def get_video_path(self, job_id: str) -> str | None:
        with self._lock:
            rec = self._jobs.get(job_id)
            return rec.video_path if rec else None

    # ------------------------------------------------------------------
    # Internal progress / status helpers
    # ------------------------------------------------------------------

    def _set_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec:
                rec.status = JobStatus(
                    id=rec.status.id,
                    status=rec.status.status,
                    message=rec.status.message,
                    progress=max(0.0, min(1.0, progress)),
                )
        self._db.update(job_id, "running", progress, None)

    def _update(self, job_id: str, *, status: str, message: str | None = None, progress: float = 1.0) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec:
                rec.status = JobStatus(
                    id=rec.status.id,
                    status=status,  # type: ignore[arg-type]
                    message=message,
                    progress=progress,
                )
        self._db.update(job_id, status, progress, message)

    def _run_job(self, job_id: str) -> None:
        self._update(job_id, status="running", progress=0.01)
        try:
            rec = self._jobs[job_id]
            spec = rec.engine

            def progress_cb(frac: float) -> None:
                self._set_progress(job_id, frac)

            payload = run_engine(rec.video_path, str(self.jobs_dir / job_id), spec, progress_cb)

            # Pull the enriched raw_frames out (not serialized to disk)
            raw_frames: list[dict[str, Any]] = payload.pop("_raw_frames", [])

            # Compute full metrics
            arena_poly = spec.arena_poly
            poly_tuples = [(float(p[0]), float(p[1])) for p in arena_poly] if arena_poly else None

            metrics = compute_all_metrics(
                raw_frames=raw_frames,
                fps=float(payload.get("summary", {}).get("fps", 0.0) or 30.0),
                px_per_cm=spec.px_per_cm,
                zones=spec.zones,
                arena_poly=poly_tuples,
                freeze_threshold_cm_s=spec.freeze_threshold_cm_s,
            )
            payload["metrics"] = metrics.model_dump()

            # Bake all enriched fields into serialized frames for frontend use
            frame_extras: dict[int, dict[str, Any]] = {}
            for rf in raw_frames:
                fi = rf.get("frame_index")
                if fi is not None:
                    # Serialize per-animal centroids for multi-animal overlay
                    animals = rf.get("animals", [])
                    animal_centroids = []
                    for a in animals:
                        if a.get("ok") and a.get("centroid"):
                            animal_centroids.append({
                                "animal_id": a["animal_id"],
                                "x": a["centroid"]["x"],
                                "y": a["centroid"]["y"],
                                "rearing": a.get("rearing", False),
                                "heading_deg": a.get("heading_deg"),
                            })
                    frame_extras[fi] = {
                        "zone_id": rf.get("zone_id"),
                        "speed_px_s": rf.get("speed_px_s"),
                        "speed_cm_s": rf.get("speed_cm_s"),
                        "heading_deg": rf.get("heading_deg"),
                        "heading_delta_deg": rf.get("heading_delta_deg"),
                        "angular_velocity_deg_s": rf.get("angular_velocity_deg_s"),
                        "rearing": rf.get("rearing", False),
                        "grooming": rf.get("grooming", False),
                        "quality": rf.get("quality", "ok"),
                        "body_length_px": rf.get("body_length_px"),
                        "ear_span_px": rf.get("ear_span_px"),
                        "head_angle_deg": rf.get("head_angle_deg"),
                        "spine_curvature": rf.get("spine_curvature"),
                        "head_body_angle_deg": rf.get("head_body_angle_deg"),
                        "keypoints": rf.get("keypoints"),
                        "canonical_kps": rf.get("canonical_kps"),
                        "animal_centroids": animal_centroids if len(animal_centroids) > 1 else None,
                    }
            for f in payload.get("frames", []):
                fi = f.get("frame_index")
                if fi in frame_extras:
                    f.update(frame_extras[fi])

            # Bake animal metadata into result
            payload["animal_meta"] = rec.meta.model_dump()

            # Write result JSON
            Path(rec.result_path).write_text(json.dumps(payload))

            # Write CSVs
            Path(rec.csv_perframe_path).write_text(
                to_per_frame_csv(raw_frames, spec.px_per_cm)
            )
            zone_names = {z["id"]: z.get("name", z["id"]) for z in spec.zones}
            Path(rec.csv_summary_path).write_text(
                to_summary_csv(metrics, zone_names)
            )

            self._update(job_id, status="done", progress=1.0)
        except Exception as e:  # noqa: BLE001
            self._update(job_id, status="error", message=str(e), progress=1.0)
