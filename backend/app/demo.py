from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import requests
import sleap_io as sio


# IMPORTANT: video + predictions must match the same clip.
# This pair is linked from SLEAP datasets page under `mice_hc` "Example".
DEMO_VIDEO_URL = (
    "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/"
    "20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.mp4"
)
DEMO_PREDICTIONS_URL = (
    "https://storage.googleapis.com/sleap-data/datasets/eleni_mice/clips/"
    "20200111_USVpairs_court1_M1_F1_top-01112020145828-0000%400-2560.slp"
)


def download_to(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def ensure_demo_files(data_dir: str) -> tuple[Path, Path]:
    base = Path(data_dir) / "demo"
    video_path = base / "demo.mp4"
    slp_path = base / "predictions.slp"
    source_path = base / "source.json"

    desired = {"video_url": DEMO_VIDEO_URL, "predictions_url": DEMO_PREDICTIONS_URL}
    have = None
    if source_path.exists():
        try:
            have = json.loads(source_path.read_text())
        except Exception:
            have = None

    # If URLs changed (or cache corrupt), re-download and overwrite.
    if have != desired:
        if video_path.exists():
            video_path.unlink(missing_ok=True)
        if slp_path.exists():
            slp_path.unlink(missing_ok=True)

    if not video_path.exists():
        download_to(DEMO_VIDEO_URL, video_path)
    if not slp_path.exists():
        download_to(DEMO_PREDICTIONS_URL, slp_path)
    source_path.write_text(json.dumps(desired))
    return video_path, slp_path


def build_demo_tracking_result(data_dir: str) -> dict[str, Any]:
    """
    Convert a SLEAP predictions .slp into the app's simple centroid time series.
    This lets the frontend overlay a "pose-powered" trajectory out of the box.
    """
    base = Path(data_dir) / "demo"
    video_path, slp_path = ensure_demo_files(data_dir)
    out_json = base / "tracking_result.json"
    # Always regenerate if code changes; demo generation is cheap enough.
    # (Prevents stale cached paths after we adjust track selection logic.)
    if out_json.exists():
        out_json.unlink(missing_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    d = sio.load_file(str(slp_path)).to_dict()
    labeled_frames = d.get("labeled_frames", [])
    frame_count = int(max((lf.get("frame_idx", 0) for lf in labeled_frames), default=-1) + 1)

    frames: list[dict[str, Any]] = []
    ok_frames = 0
    for i in range(frame_count):
        frames.append({"frame_index": i, "t_sec": i / fps, "ok": False, "centroid": None, "area_px": None})

    # Choose a consistent track (this demo has 2 mice).
    # We pick the track_idx with the most frames present, then only use that track.
    track_counts: dict[int, int] = {}
    for lf in labeled_frames:
        for inst in lf.get("instances", []) or []:
            ti = inst.get("track_idx")
            if ti is None:
                continue
            ti = int(ti)
            track_counts[ti] = track_counts.get(ti, 0) + 1

    preferred_track: int | None = None
    if track_counts:
        preferred_track = max(track_counts.items(), key=lambda kv: kv[1])[0]

    # Use preferred track instance per frame; centroid = mean of visible points.
    for lf in labeled_frames:
        fi = int(lf.get("frame_idx", -1))
        if fi < 0 or fi >= frame_count:
            continue
        instances = lf.get("instances", [])
        if not instances:
            continue
        inst = None
        if preferred_track is not None:
            for cand in instances:
                if cand.get("track_idx") is not None and int(cand.get("track_idx")) == preferred_track:
                    inst = cand
                    break
        if inst is None:
            inst = instances[0]
        pts = inst.get("points") or []
        xs: list[float] = []
        ys: list[float] = []
        for p in pts:
            if p.get("visible") is True and p.get("x") is not None and p.get("y") is not None:
                xs.append(float(p["x"]))
                ys.append(float(p["y"]))
        if not xs:
            continue
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        frames[fi]["ok"] = True
        frames[fi]["centroid"] = {"x": cx, "y": cy}
        ok_frames += 1

    payload = {
        "summary": {"fps": fps, "frame_count": frame_count, "ok_frames": ok_frames, "arena_size_px": [w, h]},
        "frames": frames,
        "meta": {
            "engine": "sleap_demo_file_centroid",
            "video_path": str(video_path),
            "slp_path": str(slp_path),
        },
    }
    out_json.write_text(json.dumps(payload))
    return payload

