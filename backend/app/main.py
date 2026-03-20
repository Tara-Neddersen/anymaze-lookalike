from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse

from .engines import EngineSpec
from .jobs import JobStore
from .metrics import compute_all_metrics, to_per_frame_csv, to_summary_csv
from .models import AnimalMeta, JobStatus
from .pose_qc import load_pose_qc, load_pose_valid_mask, write_pose_qc_artifacts
from .pose_qc_config import CANONICAL_SCHEMA_VERSION
from .tracker import extract_first_frame_jpeg


DATA_DIR = str(Path(__file__).resolve().parent.parent / "data")
store = JobStore(DATA_DIR)

# In-process task store for long-running cohort pipeline steps
_cohort_task_store: dict[str, dict] = {}
_cohort_task_lock = threading.Lock()


def _spawn_cohort_task(fn) -> str:
    """Run *fn* in a background thread. Returns a task_id immediately."""
    task_id = str(uuid.uuid4())
    with _cohort_task_lock:
        _cohort_task_store[task_id] = {"status": "running", "result": None, "error": None}

    def _wrapper():
        try:
            result = fn()
            with _cohort_task_lock:
                _cohort_task_store[task_id]["status"] = "done"
                _cohort_task_store[task_id]["result"] = result
        except Exception as exc:
            with _cohort_task_lock:
                _cohort_task_store[task_id]["status"] = "error"
                _cohort_task_store[task_id]["error"] = str(exc)

    threading.Thread(target=_wrapper, daemon=True).start()
    return task_id

# Per-video upload cap (raise if you have RAM/disk for very large files; reverse proxies may need matching limits)
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB


def _upload_limit_detail() -> str:
    """Human-readable limit for 413 responses (kept in sync with MAX_UPLOAD_BYTES)."""
    g = MAX_UPLOAD_BYTES / (1024**3)
    if g >= 1:
        return f"Video exceeds upload limit ({g:.0f} GB max per file)"
    mb = MAX_UPLOAD_BYTES / (1024**2)
    return f"Video exceeds upload limit ({mb:.0f} MB max per file)"


app = FastAPI(title="NeuroTrack Rodent Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


# ---------------------------------------------------------------------------
# Arena auto-detection
# ---------------------------------------------------------------------------

@app.post("/api/arena/detect")
async def arena_detect(image: UploadFile = File(...)) -> dict:
    """
    Accepts a JPEG/PNG image (e.g. first video frame) and returns a detected
    arena polygon plus a confidence score.
    """
    data = await image.read()
    from .arena_detect import detect_arena_from_bytes
    result = detect_arena_from_bytes(data)
    return result


# ---------------------------------------------------------------------------
# Jobs (single video)
# ---------------------------------------------------------------------------

@app.get("/api/jobs")
def list_jobs() -> dict:
    """Return all stored jobs (lightweight, for the Session Manager panel)."""
    return {"jobs": store.all_jobs_summary()}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """Delete a job and its result files."""
    ok = store.delete_job(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"ok": True}


@app.post("/api/import/csv")
async def import_tracking_csv(
    csv_file: UploadFile = File(...),
    fps: float = Form(default=25.0),
    px_per_cm: float = Form(default=0.0),
    arena_w: int = Form(default=0),
    arena_h: int = Form(default=0),
    zones_json: str | None = Form(default=None),
    animal_id: str = Form(default=""),
    treatment: str = Form(default=""),
    trial: str = Form(default=""),
    session: str = Form(default=""),
) -> dict:
    """
    Import pre-tracked x/y data from AnyMaze or any other tracker as CSV.

    Accepted column names (case-insensitive):
      - Time / t / timestamp → time in seconds (or frame index)
      - X / x / x_cm / x_px / centre x / center x
      - Y / y / y_cm / y_px / centre y / center y
      - Zone / zone_id  (optional)

    Returns a full result JSON identical to what tracking produces.
    """
    import csv as csvmod
    import uuid, dataclasses

    raw_text = (await csv_file.read()).decode("utf-8", errors="replace")
    reader = csvmod.DictReader(raw_text.splitlines())

    # Map column names
    def find_col(headers: list[str], *candidates: str) -> str | None:
        hl = [h.lower().strip() for h in headers]
        for c in candidates:
            if c in hl:
                return headers[hl.index(c)]
        # Partial match
        for c in candidates:
            for h, hl_h in zip(headers, hl):
                if c in hl_h:
                    return h
        return None

    headers = reader.fieldnames or []
    t_col = find_col(headers, "time", "t", "timestamp", "time(s)", "time_s")
    x_col = find_col(headers, "x", "x_cm", "x_px", "centre x", "center x", "pos x", "x_pos")
    y_col = find_col(headers, "y", "y_cm", "y_px", "centre y", "center y", "pos y", "y_pos")
    zone_col = find_col(headers, "zone", "zone_id", "zone id", "current zone")

    if not x_col or not y_col:
        raise HTTPException(status_code=422,
            detail=f"Cannot find X/Y columns. Found columns: {headers}")

    rows_raw = list(reader)
    if not rows_raw:
        raise HTTPException(status_code=422, detail="CSV file is empty")

    # Parse rows
    raw_frames: list[dict[str, Any]] = []
    max_x, max_y = 0.0, 0.0
    for i, row in enumerate(rows_raw):
        try:
            xv_raw = float(row[x_col]) if row.get(x_col, "").strip() else None
            yv_raw = float(row[y_col]) if row.get(y_col, "").strip() else None
        except (ValueError, TypeError):
            xv_raw, yv_raw = None, None

        if t_col and row.get(t_col, "").strip():
            try:
                t_sec = float(row[t_col])
                # If time looks like frame index (integers only, tiny values), treat as frame idx
                if t_sec < 1000 and i > 0 and t_sec < 2:
                    t_sec = i / fps
            except ValueError:
                t_sec = i / fps
        else:
            t_sec = i / fps

        ok = xv_raw is not None and yv_raw is not None
        centroid = {"x": xv_raw, "y": yv_raw} if ok else None
        if ok:
            max_x = max(max_x, xv_raw or 0)
            max_y = max(max_y, yv_raw or 0)

        zone_id = (row.get(zone_col) or "").strip() or None if zone_col else None

        raw_frames.append({
            "frame_index": i,
            "t_sec": t_sec,
            "ok": ok,
            "centroid": centroid,
            "zone_id": zone_id,
            "speed_px_s": None,
            "speed_cm_s": None,
            "heading_deg": None,
            "heading_delta_deg": None,
            "angular_velocity_deg_s": None,
            "rearing": False,
            "quality": "ok" if ok else "no_detection",
        })

    # Compute speeds from positions
    cal = float(px_per_cm) if px_per_cm > 0 else 0.0
    for i in range(1, len(raw_frames)):
        a, b = raw_frames[i - 1], raw_frames[i]
        if not a["ok"] or not b["ok"]:
            continue
        dt = b["t_sec"] - a["t_sec"]
        if dt <= 0:
            continue
        dx = b["centroid"]["x"] - a["centroid"]["x"]
        dy = b["centroid"]["y"] - a["centroid"]["y"]
        dist_px = math.hypot(dx, dy)
        spx = dist_px / dt
        b["speed_px_s"] = round(spx, 3)
        if cal > 0:
            b["speed_cm_s"] = round(spx / cal, 3)
        # Heading
        if dist_px > 0.5:
            b["heading_deg"] = round(math.degrees(math.atan2(dy, dx)), 2)

    # Arena size
    aw = int(arena_w) if arena_w > 0 else max(int(max_x) + 20, 640)
    ah = int(arena_h) if arena_h > 0 else max(int(max_y) + 20, 480)
    effective_fps = fps if fps > 0 else 25.0

    # Parse zones
    zones: list[dict[str, Any]] = []
    if zones_json:
        try:
            parsed = json.loads(zones_json)
            if isinstance(parsed, list):
                zones = parsed
        except Exception:
            pass

    # Zone assignment (if no zone in CSV)
    has_zone_data = any(f.get("zone_id") for f in raw_frames)
    if zones and not has_zone_data:
        from .arena import assign_zones, Poly
        zone_polys = {}
        for z in zones:
            pts = z.get("points", [])
            if len(pts) >= 3:
                zone_polys[z["id"]] = Poly([(float(p[0]), float(p[1])) for p in pts])
        for f in raw_frames:
            if f.get("ok") and f.get("centroid"):
                cx, cy = f["centroid"]["x"], f["centroid"]["y"]
                f["zone_id"] = assign_zones(cx, cy, zone_polys)

    # Compute metrics
    duration_s = raw_frames[-1]["t_sec"] if raw_frames else 0.0
    metrics = compute_all_metrics(
        raw_frames=raw_frames,
        fps=effective_fps,
        px_per_cm=cal,
        zones=zones,
        arena_poly=None,
    )

    # Build result payload identical to tracking output
    job_id = str(uuid.uuid4())
    frames_out = []
    for f in raw_frames:
        frames_out.append({
            "frame_index": f["frame_index"],
            "t_sec": f["t_sec"],
            "ok": f["ok"],
            "centroid": f.get("centroid"),
            "zone_id": f.get("zone_id"),
            "speed_px_s": f.get("speed_px_s"),
            "speed_cm_s": f.get("speed_cm_s"),
            "heading_deg": f.get("heading_deg"),
            "heading_delta_deg": f.get("heading_delta_deg"),
            "angular_velocity_deg_s": f.get("angular_velocity_deg_s"),
            "rearing": f.get("rearing", False),
            "quality": f.get("quality", "ok"),
        })

    payload = {
        "summary": {
            "fps": effective_fps,
            "frame_count": len(raw_frames),
            "ok_frames": sum(1 for f in raw_frames if f.get("ok")),
            "arena_size_px": [aw, ah],
        },
        "frames": frames_out,
        "metrics": metrics.model_dump(),
        "animal_meta": {
            "animal_id": animal_id, "treatment": treatment,
            "trial": trial, "session": session,
        },
        "_imported_csv": True,
    }

    # Persist like a real job so it shows in sessions
    import_dir = Path(store.jobs_dir) / job_id
    import_dir.mkdir(parents=True, exist_ok=True)
    result_path = str(import_dir / "result.json")
    Path(result_path).write_text(json.dumps(payload))

    from .engines import EngineSpec
    import dataclasses as dc
    spec = EngineSpec(name="opencv_mog2_centroid", px_per_cm=cal, zones=zones)
    meta_obj = AnimalMeta(animal_id=animal_id, treatment=treatment, trial=trial, session=session)

    store._db.insert(
        job_id=job_id,
        video_path="",
        result_path=result_path,
        csv_perframe="",
        csv_summary="",
        engine_json=json.dumps(dc.asdict(spec)),
        meta_json=meta_obj.model_dump_json(),
        video_filename=csv_file.filename or "imported.csv",
    )
    store._db.update(job_id, "done", 1.0, None)

    # Reload into memory
    from .jobs import JobRecord
    from .models import JobStatus as JS
    with store._lock:
        store._jobs[job_id] = JobRecord(
            job_id=job_id,
            video_path="",
            result_path=result_path,
            csv_perframe_path="",
            csv_summary_path="",
            engine=spec,
            meta=meta_obj,
            status=JS(id=job_id, status="done", progress=1.0),
        )

    return {"job_id": job_id, "status": "done", "frame_count": len(raw_frames),
            "ok_frames": payload["summary"]["ok_frames"]}


@app.post("/api/jobs", response_model=JobStatus)
async def create_job(
    video: UploadFile = File(...),
    pose_file: UploadFile | None = File(default=None),
    arena_json: str | None = Form(default=None),
    zones_json: str | None = Form(default=None),
    px_per_cm: float = Form(default=0.0),
    n_animals: int = Form(default=1),
    engine: str = Form(default="opencv_mog2_centroid"),
    animal_id: str = Form(default=""),
    treatment: str = Form(default=""),
    trial: str = Form(default=""),
    notes: str = Form(default=""),
    session: str = Form(default=""),
    experiment_id: str = Form(default=""),
    experimenter: str = Form(default=""),
    trim_start_s: float = Form(default=0.0),
    trim_end_s: float = Form(default=0.0),
    freeze_threshold_cm_s: float = Form(default=0.0),  # 0 = use default
) -> JobStatus:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Missing filename")

    # Validate engine name
    valid_engines = {"opencv_mog2_centroid", "dlc_csv", "sleap_slp"}
    if engine not in valid_engines:
        raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")

    # Save video
    stem = Path(video.filename).stem
    suffix = Path(video.filename).suffix
    dest = store.uploads_dir / f"{stem}-{time.time_ns()}{suffix}"
    total = 0
    with dest.open("wb") as f:
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=_upload_limit_detail())
            f.write(chunk)

    # Save optional pose file (DLC CSV or SLEAP .slp)
    pose_file_path: str | None = None
    if pose_file and pose_file.filename:
        pstem = Path(pose_file.filename).stem
        psuffix = Path(pose_file.filename).suffix
        pdest = store.uploads_dir / f"{pstem}-{time.time_ns()}{psuffix}"
        pose_data = await pose_file.read()
        pdest.write_bytes(pose_data)
        pose_file_path = str(pdest)

    arena_poly: list[list[float]] | None = None
    if arena_json:
        try:
            parsed = json.loads(arena_json)
            if isinstance(parsed, list) and len(parsed) >= 3:
                arena_poly = [[float(p[0]), float(p[1])] for p in parsed]
        except Exception:
            pass

    zones: list[dict[str, Any]] = []
    if zones_json:
        try:
            parsed_zones = json.loads(zones_json)
            if isinstance(parsed_zones, list):
                zones = parsed_zones
        except Exception:
            pass

    spec = EngineSpec(
        name=engine,  # type: ignore[arg-type]
        px_per_cm=float(px_per_cm),
        arena_poly=arena_poly,
        zones=zones,
        n_animals=max(1, int(n_animals)),
        pose_file_path=pose_file_path,
        trim_start_s=max(0.0, float(trim_start_s)),
        trim_end_s=max(0.0, float(trim_end_s)),
        freeze_threshold_cm_s=max(0.0, float(freeze_threshold_cm_s)),
    )
    meta = AnimalMeta(
        animal_id=animal_id, treatment=treatment, trial=trial, notes=notes,
        n_animals=max(1, int(n_animals)), engine=engine,
        session=session, experiment_id=experiment_id, experimenter=experimenter,
    )
    return store.create_job(str(dest), spec, meta)


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return st


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status == "error":
        raise HTTPException(status_code=500, detail=st.message or "Tracking failed")
    if st.status != "done":
        raise HTTPException(status_code=409, detail=f"Job not done (status={st.status})")
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=500, detail="Result missing")
    raw = json.loads(Path(p).read_text())
    _merge_pose_provenance(raw, job_id)
    return raw


def _merge_pose_provenance(result: dict[str, Any], job_id: str) -> None:
    """Attach pose_provenance for API consumers (also persisted after QC compute)."""
    p = store.get_result_path(job_id)
    if not p:
        return
    job_dir = Path(p).parent
    qc = load_pose_qc(job_dir)
    rec = store._jobs.get(job_id)
    engine = rec.engine if rec else None
    eng_name = engine.name if engine else (result.get("meta") or {}).get("engine", "unknown")
    pfn = None
    if engine and engine.pose_file_path:
        pfn = Path(engine.pose_file_path).name
    result["pose_provenance"] = {
        "pose_engine": eng_name,
        "pose_file_name": pfn,
        "pose_model_note": None,
        "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
        "qc_report_path": "pose_qc.json" if qc else None,
        "qc_decision": (qc or {}).get("decision", {}).get("label") if qc else None,
        "qc_generated_at": (qc or {}).get("computed_at") if qc else None,
    }


@app.post("/api/jobs/{job_id}/pose_qc")
def compute_job_pose_qc(
    job_id: str,
    confidence_threshold: float | None = Query(default=None),
) -> dict[str, Any]:
    """Compute pose QC metrics, write pose_qc.json + pose_valid_mask.npy, update result.json."""
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done")
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=500, detail="Result missing")
    result = json.loads(Path(p).read_text())
    frames = result.get("frames", [])
    has_pose = any((f.get("canonical_kps") or {}) for f in frames)
    if not has_pose:
        raise HTTPException(
            status_code=422,
            detail="No pose keypoints in this job. Use DeepLabCut or SLEAP engine with pose file.",
        )
    job_dir = Path(p).parent
    report = write_pose_qc_artifacts(
        job_dir, frames, confidence_threshold=confidence_threshold
    )
    _merge_pose_provenance(result, job_id)
    Path(p).write_text(json.dumps(result))
    return report


@app.get("/api/jobs/{job_id}/pose_qc")
def get_job_pose_qc(job_id: str) -> dict[str, Any]:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done")
    p = store.get_result_path(job_id)
    qc = load_pose_qc(Path(p).parent) if p else None
    if qc is None:
        raise HTTPException(status_code=404, detail="Pose QC not computed. POST /api/jobs/{id}/pose_qc first.")
    return qc


@app.get("/api/jobs/{job_id}/pose_qc/mask")
def get_job_pose_mask(job_id: str) -> FileResponse:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done")
    p = store.get_result_path(job_id)
    mask_path = Path(p).parent / "pose_valid_mask.npy"
    if not mask_path.exists():
        raise HTTPException(status_code=404, detail="Mask not found. Run pose QC first.")
    return FileResponse(str(mask_path), filename="pose_valid_mask.npy", media_type="application/octet-stream")


@app.get("/api/jobs/{job_id}/result/csv")
def get_job_csv(job_id: str, type: str = Query(default="perframe")) -> Response:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done")
    p = store.get_csv_path(job_id, type)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="CSV not found")
    fname = f"{type}.csv"
    return Response(
        content=Path(p).read_bytes(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


@app.patch("/api/jobs/{job_id}/frames/{frame_index}")
async def correct_frame(
    job_id: str,
    frame_index: int,
    x: float | None = None,
    y: float | None = None,
) -> dict:
    """
    Override the centroid for a single frame (manual correction).
    Pass x=null / y=null to mark the frame as not-ok.
    Recomputes metrics after the correction.
    """
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done yet")
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=500, detail="Result missing")

    result = json.loads(Path(p).read_text())
    frames = result.get("frames", [])

    # Find and update the frame
    updated = False
    for f in frames:
        if f.get("frame_index") == frame_index:
            if x is not None and y is not None:
                f["centroid"] = {"x": x, "y": y}
                f["ok"] = True
                f["quality"] = "manual"
            else:
                f["centroid"] = None
                f["ok"] = False
                f["quality"] = "manual_delete"
            updated = True
            break

    if not updated:
        raise HTTPException(status_code=404, detail="Frame not found")

    # Recompute metrics from updated frames
    spec = store._jobs[job_id].engine
    raw_frames = [dict(f) for f in frames]
    fps = float(result.get("summary", {}).get("fps", 30.0))
    poly = [(float(p2[0]), float(p2[1])) for p2 in spec.arena_poly] if spec.arena_poly else None
    metrics = compute_all_metrics(
        raw_frames=raw_frames, fps=fps, px_per_cm=spec.px_per_cm,
        zones=spec.zones, arena_poly=poly,
    )
    result["metrics"] = metrics.model_dump()
    Path(p).write_text(json.dumps(result))
    return {"ok": True, "frame_index": frame_index}


@app.post("/api/stats")
async def compute_stats(body: dict) -> dict:
    """
    Run t-test or ANOVA on provided groups.
    Body: {groups: {name: [values]}, metric: str}
    """
    from .stats import run_group_stats
    groups: dict[str, list[float]] = body.get("groups", {})
    if not groups:
        raise HTTPException(status_code=400, detail="No groups provided")
    return run_group_stats(groups)


@app.post("/api/jobs/{job_id}/reanalyze")
async def reanalyze(
    job_id: str,
    zones_json: str = Form(default="[]"),
    arena_json: str | None = Form(default=None),
    px_per_cm: float = Form(default=0.0),
) -> dict:
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.status != "done":
        raise HTTPException(status_code=409, detail="Job not done yet")
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=500, detail="Result missing")

    result = json.loads(Path(p).read_text())

    new_zones: list[dict[str, Any]] = []
    try:
        parsed = json.loads(zones_json)
        if isinstance(parsed, list):
            new_zones = parsed
    except Exception:
        pass

    new_arena_poly = None
    if arena_json:
        try:
            parsed = json.loads(arena_json)
            if isinstance(parsed, list) and len(parsed) >= 3:
                new_arena_poly = [(float(p[0]), float(p[1])) for p in parsed]
        except Exception:
            pass

    raw_frames = [dict(f) for f in result.get("frames", [])]
    fps = float(result.get("summary", {}).get("fps", 30.0))
    cal = float(px_per_cm) if px_per_cm > 0 else float(result.get("meta", {}).get("px_per_cm", 0.0))

    metrics = compute_all_metrics(
        raw_frames=raw_frames,
        fps=fps,
        px_per_cm=cal,
        zones=new_zones,
        arena_poly=new_arena_poly,
    )
    result["metrics"] = metrics.model_dump()

    frame_zone_map: dict[int, str | None] = {
        f.get("frame_index", i): f.get("zone_id") for i, f in enumerate(raw_frames)
    }
    for f in result.get("frames", []):
        fi = f.get("frame_index")
        if fi in frame_zone_map:
            f["zone_id"] = frame_zone_map[fi]

    Path(p).write_text(json.dumps(result))

    zone_names = {z["id"]: z.get("name", z["id"]) for z in new_zones}
    csv_path = store.get_csv_path(job_id, "summary")
    if csv_path:
        Path(csv_path).write_text(to_summary_csv(metrics, zone_names))
    pfcsv = store.get_csv_path(job_id, "perframe")
    if pfcsv:
        Path(pfcsv).write_text(to_per_frame_csv(raw_frames, cal))

    return result


@app.get("/api/jobs/{job_id}/video")
def stream_video(job_id: str) -> FileResponse:
    """Stream the original video file back so the frontend can play it."""
    video_path = store.get_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Job not found")
    p = Path(video_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Video file no longer on disk")
    suffix = p.suffix.lower()
    mime = {
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".mov": "video/quicktime", ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }.get(suffix, "video/mp4")
    return FileResponse(str(p), media_type=mime, filename=p.name)


@app.patch("/api/jobs/{job_id}/notes")
async def update_notes(job_id: str, body: dict) -> dict:
    """Update post-analysis notes stored in the result JSON."""
    st = store.get_status(job_id)
    if st is None:
        raise HTTPException(status_code=404, detail="Job not found")
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="Result not found")
    result = json.loads(Path(p).read_text())
    result["post_notes"] = str(body.get("notes", ""))
    Path(p).write_text(json.dumps(result))
    return {"ok": True}


@app.get("/api/jobs/{job_id}/report.pdf")
def get_pdf_report(job_id: str) -> StreamingResponse:
    """Generate a structured PDF report with metrics table + trajectory chart."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, HRFlowable,
    )
    from reportlab.graphics.shapes import Drawing, Circle, Line, PolyLine, String as RLString
    from reportlab.graphics.charts.lineplots import LinePlot
    import io as _io
    import datetime
    import numpy as _np

    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="Result not found")
    result = json.loads(Path(p).read_text())
    m = result.get("metrics", {})
    meta = result.get("animal_meta", {})
    summary = result.get("summary", {})
    fps = float(summary.get("fps", 25))

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    W, _ = A4
    content_w = W - 4*cm

    accent = colors.HexColor("#00DDB4")
    bg_dark = colors.HexColor("#1A1C1F")
    txt_main = colors.HexColor("#E8ECF0")
    txt_sub = colors.HexColor("#8A909A")

    title_style = ParagraphStyle("Title", parent=styles["Normal"],
        fontSize=18, textColor=accent, spaceAfter=4, fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("Sub", parent=styles["Normal"],
        fontSize=9, textColor=txt_sub)
    section_style = ParagraphStyle("Section", parent=styles["Normal"],
        fontSize=11, textColor=accent, spaceAfter=6, fontName="Helvetica-Bold",
        spaceBefore=12)
    note_style = ParagraphStyle("Note", parent=styles["Normal"],
        fontSize=9, textColor=txt_main, leading=14)

    def stat_table(rows: list[tuple[str, str]], title: str) -> list:
        """Build a 2-column statistics table flowable."""
        data = [["Metric", "Value"]] + list(rows)
        col_w = [content_w * 0.68, content_w * 0.32]
        t = Table(data, colWidths=col_w)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, 0), accent),
            ("TEXTCOLOR",     (0, 0), (-1, 0), bg_dark),
            ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",      (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#23262B"), colors.HexColor("#1E2124")]),
            ("TEXTCOLOR",     (0, 1), (-1, -1), txt_main),
            ("ALIGN",         (1, 0), (1, -1), "RIGHT"),
            ("GRID",          (0, 0), (-1, -1), 0.3, colors.HexColor("#30353D")),
            ("ROWPADDING",    (0, 0), (-1, -1), 4),
        ]))
        return [Paragraph(title, section_style), t]

    def fmt(v: Any, unit: str = "", decimals: int = 2) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{decimals}f}{(' ' + unit) if unit else ''}"
        except Exception:
            return str(v)

    story: list[Any] = []

    # --- Header ---
    story.append(Paragraph("Behavioral Analysis Report", title_style))
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    animal_info = " · ".join(filter(None, [
        meta.get("animal_id"), meta.get("treatment"),
        meta.get("trial"), meta.get("session"), meta.get("experiment_id"),
    ]))
    story.append(Paragraph(f"{animal_info or 'Session'} &nbsp;·&nbsp; {ts}", sub_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=accent))
    story.append(Spacer(1, 0.3*cm))

    # --- Trajectory image (drawn via reportlab shapes) ---
    frames = result.get("frames", [])
    valid_pts = [(f["centroid"]["x"], f["centroid"]["y"])
                 for f in frames if f.get("ok") and f.get("centroid")]

    if valid_pts:
        aw, ah = summary.get("arena_size_px", [640, 480])
        scale = min((content_w - 1*cm) / aw, 5*cm / ah)
        dw, dh = aw * scale, ah * scale

        drawing = Drawing(dw, dh)
        # Background
        from reportlab.graphics.shapes import Rect, PolyLine as RPL
        drawing.add(Rect(0, 0, dw, dh, fillColor=colors.HexColor("#1E2124"),
                         strokeColor=colors.HexColor("#30353D"), strokeWidth=0.5))
        # Trail
        xs = [x * scale for x, _ in valid_pts]
        ys = [dh - y * scale for _, y in valid_pts]
        step = max(1, len(xs) // 800)  # downsample for PDF size
        pts_flat = []
        for i in range(0, len(xs) - 1, step):
            pts_flat += [xs[i], ys[i], xs[i+1], ys[i+1]]
        if len(pts_flat) >= 4:
            for i in range(0, len(pts_flat) - 3, 4):
                prog = i / len(pts_flat)
                c = colors.Color(0.0, 0.5 + 0.5 * prog, 0.6 + 0.4 * prog, 0.8)
                drawing.add(Line(pts_flat[i], pts_flat[i+1],
                                 pts_flat[i+2], pts_flat[i+3],
                                 strokeColor=c, strokeWidth=0.6))
        # Start/end markers
        drawing.add(Circle(xs[0], ys[0], 4, fillColor=colors.green, strokeWidth=0))
        drawing.add(Circle(xs[-1], ys[-1], 4, fillColor=colors.red, strokeWidth=0))

        story.append(Paragraph("Trajectory", section_style))
        story.append(drawing)
        story.append(Spacer(1, 0.2*cm))

    # --- Core locomotion table ---
    loco_rows: list[tuple[str, str]] = [
        ("Total distance (cm)", fmt(m.get("total_distance_cm"), "cm")),
        ("Total distance (px)", fmt(m.get("total_distance_px"), "px", 0)),
        ("Mean speed (cm/s)",   fmt(m.get("mean_speed_cm_s"), "cm/s")),
        ("Max speed (cm/s)",    fmt(m.get("max_speed_cm_s"), "cm/s")),
        ("Duration (s)",        fmt(m.get("duration_s"), "s", 1)),
        ("Valid frames",        fmt(m.get("valid_fraction", 0) * 100, "%", 1)),
        ("Path efficiency",     fmt(m.get("path_efficiency"), "", 3)),
        ("CW rotations",        fmt(m.get("clockwise_rotations"), "°", 0)),
        ("CCW rotations",       fmt(m.get("anticlockwise_rotations"), "°", 0)),
    ]
    story += stat_table(loco_rows, "Locomotion")

    # --- Behavioral state table ---
    state_rows: list[tuple[str, str]] = [
        ("Mobile time (s)",   fmt(m.get("total_time_mobile_s"), "s", 1)),
        ("Immobile time (s)", fmt(m.get("total_time_immobile_s"), "s", 1)),
        ("Freezing time (s)", fmt(m.get("total_time_freezing_s"), "s", 1)),
        ("Freeze episodes",   str(m.get("freezing_episodes", 0))),
        ("Rearing time (s)",  fmt(m.get("total_time_rearing_s"), "s", 1)),
        ("Rearing episodes",  str(m.get("rearing_episodes", 0))),
        ("Thigmotaxis",       fmt(m.get("thigmotaxis_fraction"), "", 3) if m.get("thigmotaxis_fraction") is not None else "—"),
    ]
    story += stat_table(state_rows, "Behavioral States")

    # --- Zone table ---
    zone_data = m.get("zones", [])
    if zone_data:
        zone_rows = [(z["zone_name"],
                      f"{fmt(z['time_in_s'], 's', 1)}  ·  {z['entries']} entries  ·  "
                      f"latency {fmt(z.get('latency_first_entry_s'), 's', 1)}")
                     for z in zone_data]
        story += stat_table(zone_rows, "Zone Metrics")

    # --- Paradigm-specific tables ---
    if m.get("epm"):
        e = m["epm"]
        story += stat_table([
            ("Open arm time (%)", fmt(e["open_arm_time_pct"], "%", 1)),
            ("Closed arm time (s)", fmt(e["closed_arm_time_s"], "s", 1)),
            ("Open arm entries (%)", fmt(e["open_arm_entries_pct"], "%", 1)),
        ], "Elevated Plus Maze")

    if m.get("light_dark"):
        ld = m["light_dark"]
        story += stat_table([
            ("Light time (%)",   fmt(ld["light_time_pct"], "%", 1)),
            ("Dark time (s)",    fmt(ld["dark_time_s"], "s", 1)),
            ("Transitions",      str(ld["transitions"])),
            ("Latency to light", fmt(ld.get("latency_to_light_s"), "s", 1)),
        ], "Light/Dark Box")

    if m.get("ymaze"):
        ym = m["ymaze"]
        story += stat_table([
            ("Spontaneous alternation (%)", fmt(ym["spontaneous_alternation_pct"], "%", 1)),
            ("Arm entries",                str(ym["arm_entries"])),
            ("Alternations",               str(ym["alternations"])),
        ], "Y-Maze")

    if m.get("open_field"):
        of = m["open_field"]
        story += stat_table([
            ("Center time (%)",  fmt(of["center_time_pct"], "%", 1)),
            ("Center time (s)",  fmt(of["center_time_s"], "s", 1)),
            ("Center entries",   str(of["center_entries"])),
        ], "Open Field Center/Periphery")

    if m.get("fear_cond"):
        fc = m["fear_cond"]
        fc_rows = [(e["label"],
                    f"{fmt(e['freezing_pct'], '%', 1)} freezing  ·  {fmt(e['duration_s'], 's', 1)}")
                   for e in fc["epochs"]]
        if fc.get("baseline_freezing_pct") is not None:
            fc_rows.insert(0, ("Baseline freezing", fmt(fc["baseline_freezing_pct"], "%", 1)))
        if fc.get("mean_cs_freezing_pct") is not None:
            fc_rows.append(("Mean CS freezing", fmt(fc["mean_cs_freezing_pct"], "%", 1)))
        story += stat_table(fc_rows, "Fear Conditioning")

    # --- Post notes ---
    notes = result.get("post_notes", "")
    if notes and notes.strip():
        story.append(Paragraph("Notes", section_style))
        story.append(Paragraph(notes.replace("\n", "<br/>"), note_style))

    # --- Footer ---
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=0.3, color=txt_sub))
    story.append(Paragraph("Generated by NeuroTrack — open-source rodent behaviour analysis",
                            sub_style))

    doc.build(story)
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id[:8]}.pdf"'},
    )


@app.get("/api/jobs/{job_id}/result.json")
def get_result_json(job_id: str) -> Response:
    """Return the complete result JSON for programmatic use (R / Python pipelines)."""
    p = store.get_result_path(job_id)
    if not p or not Path(p).exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return Response(
        content=Path(p).read_bytes(),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="result_{job_id[:8]}.json"'},
    )


@app.get("/api/jobs/{job_id}/annotated_video")
def get_annotated_video(job_id: str, speed: bool = Query(default=True), zone: bool = Query(default=True)) -> StreamingResponse:
    """
    Render the original video with burned-in centroid dot, trajectory trail,
    speed text, zone label, and elapsed timer. Returns MP4.
    """
    import cv2 as _cv2
    import numpy as _np
    import io as _io
    import tempfile as _tmp

    video_path = store.get_video_path(job_id)
    p = store.get_result_path(job_id)
    if not video_path or not p or not Path(video_path).exists():
        raise HTTPException(status_code=404, detail="Video or result not found")

    result = json.loads(Path(p).read_text())
    frames_data = {f["frame_index"]: f for f in result.get("frames", [])}
    fps = float(result.get("summary", {}).get("fps", 25.0))

    # Parse zone polygons for overlay
    zones_def = result.get("engine", {}) or {}
    # zones may be embedded in result or in engine spec stored in job
    zone_list: list[dict] = []
    job_rec = store._jobs.get(job_id)
    if job_rec:
        zone_list = job_rec.engine.zones or []

    # Pre-build zone polygon arrays for cv2
    zone_polys_cv2: list[tuple[str, str, Any]] = []  # (zone_id, zone_name, np_pts)
    for z in zone_list:
        pts = z.get("points", [])
        if len(pts) >= 3:
            np_pts = _np.array([[int(pp[0]), int(pp[1])] for pp in pts], dtype=_np.int32)
            zone_polys_cv2.append((z.get("id", ""), z.get("name", z.get("id", "")), np_pts))

    # Zone colours (cycle through palette)
    ZONE_COLORS = [(0,200,255),(255,180,50),(180,255,80),(255,80,200),(80,200,255)]

    cap = _cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Cannot open video")

    W = int(cap.get(_cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(_cv2.CAP_PROP_FRAME_HEIGHT))
    tmp = _tmp.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    vw = _cv2.VideoWriter(tmp.name, _cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    trail: list[tuple[int, int]] = []
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fd = frames_data.get(fi, {})
        c = fd.get("centroid")
        t = fd.get("t_sec", fi / fps)

        # Draw zone polygons (semi-transparent fill + border)
        if zone and zone_polys_cv2:
            overlay = frame.copy()
            for zi, (zid, zname, np_pts) in enumerate(zone_polys_cv2):
                col = ZONE_COLORS[zi % len(ZONE_COLORS)]
                _cv2.fillPoly(overlay, [np_pts], col)
                _cv2.polylines(frame, [np_pts], True, col, 2)
                # Zone name at centroid
                cx_z = int(np_pts[:, 0].mean())
                cy_z = int(np_pts[:, 1].mean())
                _cv2.putText(frame, zname, (cx_z - 20, cy_z),
                    _cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            _cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        if c:
            cx_a, cy_a = int(c["x"]), int(c["y"])
            trail.append((cx_a, cy_a))
            if len(trail) > int(fps * 5):
                trail.pop(0)

            # Draw trail
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                col_t = (int(60*alpha), int(220*alpha), int(200*alpha))
                _cv2.line(frame, trail[i-1], trail[i], col_t, 2)

            # Centroid dot
            _cv2.circle(frame, (cx_a, cy_a), 6, (0, 220, 200), -1)
            _cv2.circle(frame, (cx_a, cy_a), 6, (255, 255, 255), 1)

            # Speed label
            if speed and fd.get("speed_cm_s") is not None:
                _cv2.putText(frame, f"{fd['speed_cm_s']:.1f} cm/s",
                    (cx_a + 10, cy_a - 8), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Zone label near animal
            if zone and fd.get("zone_id"):
                _cv2.putText(frame, str(fd["zone_id"]),
                    (cx_a + 10, cy_a + 14), _cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 220, 255), 1)

        # Timer overlay
        mm, s = divmod(t, 60)
        timer_str = f"{int(mm):02d}:{s:05.2f}"
        _cv2.putText(frame, timer_str, (10, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        _cv2.putText(frame, timer_str, (10, 22), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        vw.write(frame)
        fi += 1

    cap.release()
    vw.release()

    data = Path(tmp.name).read_bytes()
    Path(tmp.name).unlink(missing_ok=True)
    return StreamingResponse(
        _io.BytesIO(data), media_type="video/mp4",
        headers={"Content-Disposition": f'attachment; filename="annotated_{job_id[:8]}.mp4"'},
    )


@app.get("/api/jobs/{job_id}/first_frame")
def get_first_frame(job_id: str) -> Response:
    video_path = store.get_video_path(job_id)
    if not video_path:
        raise HTTPException(status_code=404, detail="Job not found")
    try:
        jpeg = extract_first_frame_jpeg(video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=jpeg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

@app.post("/api/batch")
async def create_batch(
    videos: list[UploadFile] = File(...),
    arena_json: str | None = Form(default=None),
    zones_json: str | None = Form(default=None),
    px_per_cm: float = Form(default=0.0),
    n_animals: int = Form(default=1),
    group_name: str = Form(default=""),
) -> dict:
    """
    Submit multiple videos for batch tracking. Returns a list of job IDs.
    Each video is queued as an independent job; use GET /api/jobs/{id} to poll.
    """
    if not videos:
        raise HTTPException(status_code=400, detail="No videos uploaded")

    arena_poly: list[list[float]] | None = None
    if arena_json:
        try:
            parsed = json.loads(arena_json)
            if isinstance(parsed, list) and len(parsed) >= 3:
                arena_poly = [[float(p[0]), float(p[1])] for p in parsed]
        except Exception:
            pass

    zones: list[dict[str, Any]] = []
    if zones_json:
        try:
            parsed_zones = json.loads(zones_json)
            if isinstance(parsed_zones, list):
                zones = parsed_zones
        except Exception:
            pass

    spec = EngineSpec(
        name="opencv_mog2_centroid",
        px_per_cm=float(px_per_cm),
        arena_poly=arena_poly,
        zones=zones,
        n_animals=max(1, int(n_animals)),
    )

    job_ids = []
    for video in videos:
        if not video.filename:
            continue
        stem = Path(video.filename).stem
        suffix = Path(video.filename).suffix
        dest = store.uploads_dir / f"{stem}-{time.time_ns()}{suffix}"
        total = 0
        with dest.open("wb") as f:
            while True:
                chunk = await video.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    dest.unlink(missing_ok=True)
                    break
                f.write(chunk)
        if not dest.exists():
            continue

        animal_id = Path(video.filename).stem
        meta = AnimalMeta(animal_id=animal_id, treatment=group_name)
        st = store.create_job(str(dest), spec, meta)
        job_ids.append({"job_id": st.id, "filename": video.filename})

    return {"jobs": job_ids, "count": len(job_ids)}


@app.get("/api/batch/csv")
def batch_combined_csv(job_ids: str = Query(...)) -> Response:
    """
    Download a combined summary CSV for a comma-separated list of job IDs.
    One row per animal (job).
    """
    import csv
    import io

    ids = [j.strip() for j in job_ids.split(",") if j.strip()]
    buf = io.StringIO()
    w = csv.writer(buf)

    # First pass: collect all zone names across all jobs for consistent columns
    all_zone_names: list[str] = []
    results_cache: dict[str, Any] = {}
    for jid in ids:
        p = store.get_result_path(jid)
        if not p or not Path(p).exists():
            continue
        result = json.loads(Path(p).read_text())
        results_cache[jid] = result
        for z in result.get("metrics", {}).get("zones", []):
            zn = z.get("zone_name", z.get("zone_id", ""))
            if zn and zn not in all_zone_names:
                all_zone_names.append(zn)

    header_written = False
    for jid in ids:
        result = results_cache.get(jid)
        if not result:
            continue
        m = result.get("metrics", {})
        am = result.get("animal_meta", {})
        row: dict[str, Any] = {
            "job_id": jid,
            "animal_id": am.get("animal_id", ""),
            "treatment": am.get("treatment", ""),
            "trial": am.get("trial", ""),
            "session": am.get("session", ""),
            "experiment_id": am.get("experiment_id", ""),
            "experimenter": am.get("experimenter", ""),
            "duration_s": m.get("duration_s", ""),
            "total_distance_cm": m.get("total_distance_cm", ""),
            "total_distance_px": m.get("total_distance_px", ""),
            "mean_speed_cm_s": m.get("mean_speed_cm_s", ""),
            "mean_speed_px_s": m.get("mean_speed_px_s", ""),
            "max_speed_cm_s": m.get("max_speed_cm_s", ""),
            "total_time_mobile_s": m.get("total_time_mobile_s", ""),
            "total_time_immobile_s": m.get("total_time_immobile_s", ""),
            "total_time_freezing_s": m.get("total_time_freezing_s", ""),
            "freezing_episodes": m.get("freezing_episodes", ""),
            "thigmotaxis_fraction": m.get("thigmotaxis_fraction", ""),
            "path_efficiency": m.get("path_efficiency", ""),
            "clockwise_rotations": m.get("clockwise_rotations", ""),
            "anticlockwise_rotations": m.get("anticlockwise_rotations", ""),
            "total_time_rearing_s": m.get("total_time_rearing_s", ""),
            "rearing_episodes": m.get("rearing_episodes", ""),
            "valid_fraction": m.get("valid_fraction", ""),
        }
        # Per-zone columns: time_in_s, entries, latency for each zone
        zone_map = {z.get("zone_name", z.get("zone_id", "")): z
                    for z in m.get("zones", [])}
        for zn in all_zone_names:
            z = zone_map.get(zn, {})
            row[f"zone_{zn}_time_s"] = z.get("time_in_s", "")
            row[f"zone_{zn}_entries"] = z.get("entries", "")
            row[f"zone_{zn}_latency_s"] = z.get("latency_first_entry_s", "")
        if not header_written:
            w.writerow(list(row.keys()))
            header_written = True
        w.writerow(list(row.values()))

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="batch_summary.csv"'},
    )


# ---------------------------------------------------------------------------
# Demo (SLEAP-derived, now with metrics)
# ---------------------------------------------------------------------------

@app.get("/api/demo/sleap")
def sleap_demo() -> dict:
    from .demo import build_demo_tracking_result
    payload = build_demo_tracking_result(DATA_DIR)

    # Compute metrics for demo frames (no calibration, no zones)
    raw_frames = [dict(f) for f in payload.get("frames", [])]
    fps = float(payload.get("summary", {}).get("fps", 30.0))
    try:
        metrics = compute_all_metrics(
            raw_frames=raw_frames,
            fps=fps,
            px_per_cm=0.0,
            zones=[],
            arena_poly=None,
        )
        payload["metrics"] = metrics.model_dump()
    except Exception:
        pass  # Demo metrics are best-effort

    return payload


@app.get("/api/demo/sleap/video")
def sleap_demo_video() -> FileResponse:
    from .demo import ensure_demo_files
    video_path, _ = ensure_demo_files(DATA_DIR)
    return FileResponse(str(video_path), media_type="video/mp4", filename="mice.mp4")


# ===========================================================================
# Cohort Management + Latent Phenotyping Pipeline
# ===========================================================================

from .cohort_store import (
    create_cohort, get_cohort, list_cohorts, add_animal, remove_animal,
    delete_cohort as _delete_cohort, update_pipeline_status, set_motif_library,
    save_npy, load_npy, save_json, load_json,
    CohortAnimal, job_pose_features_path, job_motif_labels_path, job_phenotype_path,
)
from pydantic import BaseModel as _BM


class _AddAnimalBody(_BM):
    job_id:      str
    animal_id:   str
    genotype:    str
    sex:         str | None = None
    age_weeks:   float | None = None
    treatment:   str | None = None


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

@app.post("/api/cohorts")
def api_create_cohort(name: str = Query(...)) -> dict:
    c = create_cohort(name)
    return {"cohort_id": c.cohort_id, "name": c.name, "created_at": c.created_at}


@app.get("/api/cohorts")
def api_list_cohorts() -> dict:
    cohorts = list_cohorts()
    return {"cohorts": [
        {
            "cohort_id": c.cohort_id,
            "name": c.name,
            "n_animals": len(c.animals),
            "created_at": c.created_at,
            "pipeline_status": c.pipeline_status,
        }
        for c in cohorts
    ]}


@app.get("/api/cohorts/{cohort_id}")
def api_get_cohort(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    import dataclasses
    return {
        "cohort_id": c.cohort_id,
        "name": c.name,
        "created_at": c.created_at,
        "pipeline_status": c.pipeline_status,
        "motif_library": c.motif_library,
        "animals": [dataclasses.asdict(a) for a in c.animals],
    }


@app.get("/api/cohorts/{cohort_id}/pose_qc_summary")
def api_cohort_pose_qc_summary(cohort_id: str) -> dict[str, Any]:
    from .pose_qc_stats import cohort_pose_qc_summary

    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    return cohort_pose_qc_summary(cohort_id, DATA_DIR)


@app.post("/api/cohorts/{cohort_id}/export_labeling_frames")
def api_export_labeling_frames(
    cohort_id: str,
    n_frames: int = Query(default=150, ge=20, le=500),
    write_png: bool = Query(default=False),
) -> dict[str, Any]:
    from .pose_label_sampler import export_labeling_manifest

    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    return export_labeling_manifest(
        cohort_id,
        DATA_DIR,
        n_frames=n_frames,
        write_png_crops=write_png,
        video_path_fn=lambda jid: store.get_video_path(jid),
    )


@app.post("/api/cohorts/{cohort_id}/animals")
def api_add_animal(cohort_id: str, body: _AddAnimalBody) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    animal = CohortAnimal(
        job_id=body.job_id,
        animal_id=body.animal_id,
        genotype=body.genotype,
        sex=body.sex,
        age_weeks=body.age_weeks,
        treatment=body.treatment,
    )
    c = add_animal(cohort_id, animal)
    return {"ok": True, "n_animals": len(c.animals)}


@app.delete("/api/cohorts/{cohort_id}/animals/{job_id}")
def api_remove_animal(cohort_id: str, job_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    remove_animal(cohort_id, job_id)
    return {"ok": True}


@app.delete("/api/cohorts/{cohort_id}")
def api_delete_cohort(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    _delete_cohort(cohort_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Background task status polling
# ---------------------------------------------------------------------------

@app.get("/api/cohorts/{cohort_id}/task_status/{task_id}")
def api_cohort_task_status(cohort_id: str, task_id: str) -> dict:
    with _cohort_task_lock:
        entry = _cohort_task_store.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "task_id": task_id,
        "cohort_id": cohort_id,
        "status": entry["status"],
        "result": entry.get("result"),
        "error": entry.get("error"),
    }


# ---------------------------------------------------------------------------
# Step 1: Compute pose features for all animals in cohort
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/compute_pose_features")
def api_compute_pose_features(
    cohort_id: str,
    use_pose_qc_mask: bool = Query(default=True),
) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .pose_features import (
        apply_pose_qc_mask,
        compute_pose_feature_matrix,
        impute_pose_matrix,
        valid_fraction,
    )
    import os
    import numpy as np

    update_pipeline_status(cohort_id, "pose_features", "running")
    results = []

    for animal in c.animals:
        job_dir = Path(DATA_DIR) / "jobs" / animal.job_id
        result_path = job_dir / "result.json"
        if not result_path.exists():
            results.append({"job_id": animal.job_id, "status": "no_result"})
            continue

        try:
            with open(result_path) as f:
                result = json.load(f)

            raw_frames = result.get("frames", [])
            fps = float(result.get("summary", {}).get("fps", 30.0))
            mat, vmask = compute_pose_feature_matrix(raw_frames, fps)
            mat, vmask = impute_pose_matrix(mat, vmask)
            qc_applied = False
            if use_pose_qc_mask:
                vmask, qc_applied = apply_pose_qc_mask(vmask, job_dir)
            vfrac = valid_fraction(vmask)

            out_path = job_pose_features_path(animal.job_id)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            np.save(out_path, mat)

            results.append({
                "job_id": animal.job_id,
                "status": "ok",
                "n_frames": len(mat),
                "valid_fraction": round(vfrac, 4),
                "pose_qc_mask_applied": qc_applied,
            })

            if vfrac < 0.20:
                results[-1]["warning"] = f"Low valid fraction ({vfrac:.1%}). Check pose data."

        except Exception as e:
            results.append({"job_id": animal.job_id, "status": "error", "error": str(e)})

    all_ok = all(r["status"] == "ok" for r in results)
    update_pipeline_status(cohort_id, "pose_features", "done" if all_ok else "error")
    return {"cohort_id": cohort_id, "results": results}


# ---------------------------------------------------------------------------
# Step 2: Discover motifs across cohort
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/discover_motifs")
def api_discover_motifs(cohort_id: str, k: int | None = Query(default=None)) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .window_features import pool_cohort_windows, window_labels_to_frame_labels, compute_window_features
    from .motif_discovery import discover_motifs
    import numpy as np
    import os

    # Preload data synchronously to catch obvious errors before spawning thread
    pose_matrices, valid_masks, loaded_animals, fps_list = [], [], [], []
    for animal in c.animals:
        path = job_pose_features_path(animal.job_id)
        if not os.path.exists(path):
            continue
        mat = np.load(path, allow_pickle=False)
        vmask = ~np.any(np.isnan(mat), axis=1)
        pose_matrices.append(mat)
        valid_masks.append(vmask)
        loaded_animals.append(animal)
        job_dir = Path(DATA_DIR) / "jobs" / animal.job_id / "result.json"
        fps = 30.0
        if job_dir.exists():
            try:
                with open(job_dir) as f:
                    fps = float(json.load(f).get("summary", {}).get("fps", 30.0))
            except Exception:
                pass
        fps_list.append(fps)

    if not pose_matrices:
        raise HTTPException(status_code=422, detail="No pose feature files found. Run compute_pose_features first.")

    update_pipeline_status(cohort_id, "motif_discovery", "running")

    def _run():
        avg_fps = float(np.mean(fps_list)) if fps_list else 30.0
        pooled_wf, pooled_valid, _animal_idx_arr, _n_per_animal = pool_cohort_windows(
            pose_matrices, valid_masks, avg_fps
        )
        try:
            library, all_labels = discover_motifs(pooled_wf, pooled_valid, k_override=k)
        except ValueError as exc:
            update_pipeline_status(cohort_id, "motif_discovery", "error")
            raise exc

        set_motif_library(cohort_id, library)
        save_json(cohort_id, "motif_library", library)

        offset = 0
        for animal, mat, vmask, fps in zip(loaded_animals, pose_matrices, valid_masks, fps_list):
            wf, wstarts, wvalid = compute_window_features(mat, vmask, fps)
            n_w = len(wf)
            animal_window_labels = all_labels[offset:offset + n_w]
            offset += n_w
            frame_labels = window_labels_to_frame_labels(
                animal_window_labels, wstarts, len(mat), library["k"], wvalid
            )
            np.save(job_motif_labels_path(animal.job_id), frame_labels)

        update_pipeline_status(cohort_id, "motif_discovery", "done")
        return {
            "cohort_id": cohort_id,
            "k": library["k"],
            "auto_labels": library["auto_labels"],
            "stability_score": library["stability_score"],
            "silhouette": library["silhouette"],
            "n_animals_with_features": len(loaded_animals),
        }

    task_id = _spawn_cohort_task(_run)
    return {"task_id": task_id, "status": "running", "cohort_id": cohort_id}


@app.get("/api/cohorts/{cohort_id}/motif_library")
def api_get_motif_library(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")
    lib = c.motif_library or load_json(cohort_id, "motif_library")
    if lib is None:
        raise HTTPException(status_code=404, detail="Motif library not found. Run discover_motifs first.")
    return lib


# ---------------------------------------------------------------------------
# Step 3: Compute sequence profiles
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/compute_sequences")
def api_compute_sequences(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .sequence_analysis import compute_sequence_profile
    import numpy as np
    import os

    update_pipeline_status(cohort_id, "sequence_analysis", "running")
    lib = c.motif_library or load_json(cohort_id, "motif_library")
    if lib is None:
        raise HTTPException(status_code=422, detail="Run discover_motifs first.")

    k = int(lib["k"])
    profiles = {}
    fps_map: dict[str, float] = {}

    for animal in c.animals:
        path = job_motif_labels_path(animal.job_id)
        if not os.path.exists(path):
            profiles[animal.job_id] = {"error": "no_motif_labels"}
            continue

        job_res = Path(DATA_DIR) / "jobs" / animal.job_id / "result.json"
        fps = 30.0
        if job_res.exists():
            try:
                with open(job_res) as f:
                    fps = float(json.load(f).get("summary", {}).get("fps", 30.0))
            except Exception:
                pass
        fps_map[animal.job_id] = fps

        labels = np.load(path, allow_pickle=False).astype(int)
        profile = compute_sequence_profile(labels, fps, k)
        profiles[animal.job_id] = profile

    save_json(cohort_id, "sequence_profiles", profiles)
    update_pipeline_status(cohort_id, "sequence_analysis", "done")
    return {"cohort_id": cohort_id, "n_profiles": len(profiles)}


# ---------------------------------------------------------------------------
# Step 4: UMAP embedding
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/compute_embedding")
def api_compute_embedding(
    cohort_id: str,
    max_frames: int = Query(default=50000),
) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .embedding import compute_perframe_embedding
    import numpy as np
    import os

    # Preload data synchronously so we fail fast if nothing is ready
    pose_matrices, valid_masks, animal_meta = [], [], []
    for animal in c.animals:
        path = job_pose_features_path(animal.job_id)
        if not os.path.exists(path):
            continue
        mat = np.load(path, allow_pickle=False)
        vmask = ~np.any(np.isnan(mat), axis=1)
        pose_matrices.append(mat)
        valid_masks.append(vmask)
        animal_meta.append({
            "job_id":    animal.job_id,
            "animal_id": animal.animal_id,
            "genotype":  animal.genotype,
        })

    if not pose_matrices:
        raise HTTPException(status_code=422, detail="No pose feature files found.")

    update_pipeline_status(cohort_id, "embedding", "running")

    def _run():
        result = compute_perframe_embedding(
            pose_matrices, valid_masks, animal_meta, max_frames=max_frames
        )
        save_json(cohort_id, "umap_perframe", result)
        update_pipeline_status(cohort_id, "embedding", "done")
        return {
            "cohort_id": cohort_id,
            "n_frames": result["n_valid"],
            "params": result["params"],
        }

    task_id = _spawn_cohort_task(_run)
    return {"task_id": task_id, "status": "running", "cohort_id": cohort_id}


# ---------------------------------------------------------------------------
# Step 5: Compute phenotype vectors
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/compute_phenotypes")
def api_compute_phenotypes(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .phenotype import build_phenotype_vector, zscore_phenotypes
    import os

    lib = c.motif_library or load_json(cohort_id, "motif_library")
    if lib is None:
        raise HTTPException(status_code=422, detail="Run discover_motifs first.")
    k = int(lib["k"])
    seq_profiles = load_json(cohort_id, "sequence_profiles") or {}

    # Collect all data synchronously so we fail fast
    phenotype_list: list = []
    animal_meta: list = []
    for animal in c.animals:
        result_path = Path(DATA_DIR) / "jobs" / animal.job_id / "result.json"
        if not result_path.exists():
            continue
        try:
            with open(result_path) as f:
                result = json.load(f)
            bm = result.get("metrics", {})
            sp = seq_profiles.get(animal.job_id, {})
            pv = build_phenotype_vector(sp, bm, k=k)
            phenotype_list.append(pv)
            animal_meta.append({
                "job_id": animal.job_id,
                "animal_id": animal.animal_id,
                "genotype": animal.genotype,
            })
        except Exception:
            pass

    update_pipeline_status(cohort_id, "phenotypes", "running")

    def _run():
        wt_indices = [i for i, m in enumerate(animal_meta) if m["genotype"].upper() == "WT"] or None
        if phenotype_list:
            zscored, ref_means, ref_stds = zscore_phenotypes(phenotype_list, wt_indices)
        else:
            zscored, ref_means, ref_stds = [], {}, {}

        phenotype_store = {
            "animals": animal_meta,
            "raw_vectors": phenotype_list,
            "zscored_vectors": zscored,
            "ref_means": ref_means,
            "ref_stds": ref_stds,
        }
        save_json(cohort_id, "phenotypes", phenotype_store)

        for meta, pv, zv in zip(animal_meta, phenotype_list, zscored):
            ph_path = job_phenotype_path(meta["job_id"])
            os.makedirs(os.path.dirname(ph_path), exist_ok=True)
            with open(ph_path, "w") as f:
                json.dump({"raw": pv, "zscored": zv}, f, indent=2)

        update_pipeline_status(cohort_id, "phenotypes", "done")
        return {"cohort_id": cohort_id, "n_animals": len(phenotype_list)}

    task_id = _spawn_cohort_task(_run)
    return {"task_id": task_id, "status": "running", "cohort_id": cohort_id}


# ---------------------------------------------------------------------------
# Step 6: Group comparison
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/run_group_comparison")
def api_run_group_comparison(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .group_analysis import compare_phenotype_groups, multivariate_comparison

    ph_store = load_json(cohort_id, "phenotypes")
    if not ph_store:
        raise HTTPException(status_code=422, detail="Run compute_phenotypes first.")

    update_pipeline_status(cohort_id, "group_comparison", "running")

    animals = ph_store["animals"]
    raw_vecs = ph_store["raw_vectors"]
    z_vecs   = ph_store["zscored_vectors"]

    # Group by genotype
    groups: dict[str, list] = {}
    for meta, rv in zip(animals, raw_vecs):
        g = meta["genotype"]
        groups.setdefault(g, []).append(rv)

    results: dict[str, Any] = {}

    group_names = list(groups.keys())
    if len(group_names) >= 2:
        # Pairwise comparisons
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                ga, gb = group_names[i], group_names[j]
                comp = compare_phenotype_groups(
                    groups[ga], groups[gb],
                    group_a_name=ga, group_b_name=gb,
                )
                results[f"{ga}_vs_{gb}"] = comp

        # MANOVA (if exactly 2 groups)
        if len(group_names) == 2:
            ga, gb = group_names[0], group_names[1]
            manova = multivariate_comparison(groups[ga], groups[gb])
            results["manova"] = manova

    # Radar data
    if len(group_names) >= 2:
        from .phenotype import radar_data as _radar_data
        ga, gb = group_names[0], group_names[1]
        # Get p-values for radar features from feature_results
        comp_key = f"{ga}_vs_{gb}"
        pvals: dict[str, float] = {}
        if comp_key in results:
            for fr in results[comp_key].get("feature_results", []):
                pvals[fr["feature"]] = fr.get("p_fdr", 1.0)

        radar = _radar_data(groups[ga], groups[gb], ga, gb, pvals)
        results["radar_data"] = radar

    save_json(cohort_id, "group_comparison", results)
    update_pipeline_status(cohort_id, "group_comparison", "done")
    return {"cohort_id": cohort_id, "n_comparisons": len(results)}


# ---------------------------------------------------------------------------
# Step 7: Classifier
# ---------------------------------------------------------------------------

@app.post("/api/cohorts/{cohort_id}/run_classifier")
def api_run_classifier(cohort_id: str) -> dict:
    c = get_cohort(cohort_id)
    if c is None:
        raise HTTPException(status_code=404, detail="Cohort not found")

    from .classifier import run_classifier

    ph_store = load_json(cohort_id, "phenotypes")
    if not ph_store:
        raise HTTPException(status_code=422, detail="Run compute_phenotypes first.")

    update_pipeline_status(cohort_id, "classifier", "running")

    animals  = ph_store["animals"]
    raw_vecs = ph_store["raw_vectors"]
    labels   = [m["genotype"] for m in animals]

    try:
        classifier_results = run_classifier(raw_vecs, labels)
    except Exception as e:
        update_pipeline_status(cohort_id, "classifier", "error")
        raise HTTPException(status_code=500, detail=str(e))

    save_json(cohort_id, "classifier_results", classifier_results)
    update_pipeline_status(cohort_id, "classifier", "done")
    return {"cohort_id": cohort_id, **{
        k: v for k, v in classifier_results.items()
        if k in ("logistic_regression", "random_forest", "unique_labels", "n_animals", "n_features")
    }}


# ---------------------------------------------------------------------------
# Results endpoints
# ---------------------------------------------------------------------------

@app.get("/api/cohorts/{cohort_id}/results/sequence_profiles")
def api_get_sequence_profiles(cohort_id: str) -> dict:
    data = load_json(cohort_id, "sequence_profiles")
    if data is None:
        raise HTTPException(status_code=404, detail="Sequence profiles not computed yet.")
    return data


@app.get("/api/cohorts/{cohort_id}/results/embedding")
def api_get_embedding(cohort_id: str) -> dict:
    data = load_json(cohort_id, "umap_perframe")
    if data is None:
        raise HTTPException(status_code=404, detail="Embedding not computed yet.")
    return data


@app.get("/api/cohorts/{cohort_id}/results/phenotypes")
def api_get_phenotypes(cohort_id: str) -> dict:
    data = load_json(cohort_id, "phenotypes")
    if data is None:
        raise HTTPException(status_code=404, detail="Phenotypes not computed yet.")
    return data


@app.get("/api/cohorts/{cohort_id}/results/statistics")
def api_get_statistics(cohort_id: str) -> dict:
    data = load_json(cohort_id, "group_comparison")
    if data is None:
        raise HTTPException(status_code=404, detail="Group comparison not computed yet.")
    return data


@app.get("/api/cohorts/{cohort_id}/results/classifier")
def api_get_classifier(cohort_id: str) -> dict:
    data = load_json(cohort_id, "classifier_results")
    if data is None:
        raise HTTPException(status_code=404, detail="Classifier not run yet.")
    return data


@app.get("/api/cohorts/{cohort_id}/results/export.csv")
def api_export_phenotype_csv(cohort_id: str) -> Response:
    import csv
    import io

    ph_store = load_json(cohort_id, "phenotypes")
    if not ph_store:
        raise HTTPException(status_code=404, detail="Phenotypes not computed yet.")

    animals  = ph_store["animals"]
    raw_vecs = ph_store["raw_vectors"]
    z_vecs   = ph_store.get("zscored_vectors", raw_vecs)

    if not raw_vecs:
        raise HTTPException(status_code=404, detail="No phenotype data.")

    feat_names = list(raw_vecs[0].keys())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["job_id", "animal_id", "genotype"] +
               feat_names +
               [f"z_{f}" for f in feat_names])

    for meta, rv, zv in zip(animals, raw_vecs, z_vecs):
        row = [meta["job_id"], meta["animal_id"], meta["genotype"]]
        row += [rv.get(f, "") for f in feat_names]
        row += [zv.get(f, "") for f in feat_names]
        w.writerow(row)

    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="cohort_{cohort_id}_phenotypes.csv"'},
    )
