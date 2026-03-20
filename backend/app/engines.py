from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .tracker import TrackConfig, track_centroid


EngineName = Literal["opencv_mog2_centroid", "dlc_csv", "sleap_slp"]


@dataclass(frozen=True)
class EngineSpec:
    name: EngineName = "opencv_mog2_centroid"
    px_per_cm: float = 0.0
    arena_poly: list[list[float]] | None = None
    zones: list[dict[str, Any]] = field(default_factory=list)
    n_animals: int = 1
    # Path to a pose file (DLC CSV or SLEAP .slp); only used for pose engines
    pose_file_path: str | None = None
    # Video trimming: analyse only this time window (0 = full video)
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0
    # Freezing threshold override (0 = use default 0.5 cm/s)
    freeze_threshold_cm_s: float = 0.0


def run_engine(
    video_path: str,
    out_dir: str,
    spec: EngineSpec,
    progress_cb: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """
    Run the chosen tracking engine and return a payload dict with:
      summary, frames (Pydantic-serialised), raw_frames (enriched dicts), meta.
    """
    if spec.name == "dlc_csv":
        return _run_dlc(video_path, spec, progress_cb)
    if spec.name == "sleap_slp":
        return _run_sleap(video_path, spec, progress_cb)

    # Default: OpenCV MOG2 centroid tracker
    return _run_opencv(video_path, spec, progress_cb)


# ---------------------------------------------------------------------------
# OpenCV engine
# ---------------------------------------------------------------------------

def _run_opencv(
    video_path: str,
    spec: EngineSpec,
    progress_cb: Callable[[float], None] | None,
) -> dict[str, Any]:
    cfg = TrackConfig(
        px_per_cm=spec.px_per_cm,
        arena_poly=spec.arena_poly,
        n_animals=max(1, spec.n_animals),
        trim_start_s=spec.trim_start_s,
        trim_end_s=spec.trim_end_s,
    )
    result = track_centroid(video_path, cfg, progress_cb=progress_cb)
    payload = result.model_dump()
    payload["_raw_frames"] = result._raw_frames
    payload["meta"] = {
        "engine": spec.name,
        "px_per_cm": spec.px_per_cm,
        "zones": spec.zones,
        "arena_poly": spec.arena_poly,
    }
    return payload


# ---------------------------------------------------------------------------
# DeepLabCut engine
# ---------------------------------------------------------------------------

def _run_dlc(
    video_path: str,
    spec: EngineSpec,
    progress_cb: Callable[[float], None] | None,
) -> dict[str, Any]:
    from .pose import parse_dlc_csv, pose_to_raw_frames, get_video_meta
    from .models import TrackingFrame, TrackingResult, TrackingSummary, Point

    if not spec.pose_file_path:
        raise ValueError("DLC engine requires pose_file_path pointing to the exported CSV")

    if progress_cb:
        progress_cb(0.05)

    pose_frames = parse_dlc_csv(spec.pose_file_path)
    if progress_cb:
        progress_cb(0.30)

    meta = get_video_meta(video_path)
    fps, w, h = meta["fps"], meta["width"], meta["height"]

    raw_frames = pose_to_raw_frames(
        pose_frames, fps=fps, px_per_cm=spec.px_per_cm, n_animals=spec.n_animals
    )
    if progress_cb:
        progress_cb(0.85)

    return _raw_frames_to_payload(raw_frames, fps, w, h, spec, "dlc_csv")


# ---------------------------------------------------------------------------
# SLEAP engine
# ---------------------------------------------------------------------------

def _run_sleap(
    video_path: str,
    spec: EngineSpec,
    progress_cb: Callable[[float], None] | None,
) -> dict[str, Any]:
    from .pose import parse_sleap_slp, pose_to_raw_frames, get_video_meta

    if not spec.pose_file_path:
        raise ValueError("SLEAP engine requires pose_file_path pointing to the .slp file")

    if progress_cb:
        progress_cb(0.05)

    pose_frames = parse_sleap_slp(spec.pose_file_path)
    if progress_cb:
        progress_cb(0.40)

    meta = get_video_meta(video_path)
    fps, w, h = meta["fps"], meta["width"], meta["height"]

    raw_frames = pose_to_raw_frames(
        pose_frames, fps=fps, px_per_cm=spec.px_per_cm, n_animals=spec.n_animals
    )
    if progress_cb:
        progress_cb(0.85)

    return _raw_frames_to_payload(raw_frames, fps, w, h, spec, "sleap_slp")


# ---------------------------------------------------------------------------
# Shared payload builder for pose engines
# ---------------------------------------------------------------------------

def _raw_frames_to_payload(
    raw_frames: list[dict[str, Any]],
    fps: float,
    w: int,
    h: int,
    spec: EngineSpec,
    engine_name: str,
) -> dict[str, Any]:
    from .models import TrackingFrame, TrackingResult, TrackingSummary, Point

    ok_frames = sum(1 for f in raw_frames if f.get("ok"))
    result = TrackingResult(
        summary=TrackingSummary(
            fps=fps,
            frame_count=len(raw_frames),
            ok_frames=ok_frames,
            arena_size_px=(w, h),
        ),
        frames=[
            TrackingFrame(
                frame_index=f["frame_index"],
                t_sec=f["t_sec"],
                centroid=Point(**f["centroid"]) if f.get("centroid") else None,
                ok=f["ok"],
                area_px=f.get("area_px"),
            )
            for f in raw_frames
        ],
    )
    result._raw_frames = raw_frames
    payload = result.model_dump()
    payload["_raw_frames"] = raw_frames
    payload["meta"] = {
        "engine": engine_name,
        "px_per_cm": spec.px_per_cm,
        "zones": spec.zones,
        "arena_poly": spec.arena_poly,
    }
    return payload
