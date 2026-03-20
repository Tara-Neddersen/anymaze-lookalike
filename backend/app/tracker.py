from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import cv2
import numpy as np

from .assignment import hungarian_assign
from .models import Point, TrackingFrame, TrackingResult, TrackingSummary
from .smooth import kalman_smooth


@dataclass(frozen=True)
class TrackConfig:
    min_area_px: float = 80.0
    max_area_px: float = 25000.0
    history: int = 300
    var_threshold: float = 24.0
    detect_shadows: bool = True
    # Real-world calibration: pixels per cm. 0 means uncalibrated (report px only).
    px_per_cm: float = 0.0
    # Arena polygon as list of [x, y] pairs (image coordinates). None = full frame.
    arena_poly: list[list[float]] | None = None
    # Warm-up: run the background model for this many frames before accepting
    # detections. The animal is typically present from frame 0, which would
    # otherwise bake it into the background.
    warmup_frames: int = 90
    # Multi-animal: track up to n_animals blobs per frame.
    # 1 = single-animal (classic). 2+ enables multi-animal mode.
    n_animals: int = 1
    # Rearing detection: blob area drops below this fraction of baseline
    rearing_area_fraction: float = 0.40
    # Trim: only analyse frames within [trim_start_s, trim_end_s].
    # 0.0 for trim_end_s means "end of video".
    trim_start_s: float = 0.0
    trim_end_s: float = 0.0


def _build_arena_mask(h: int, w: int, arena_poly: list[list[float]] | None) -> np.ndarray | None:
    if not arena_poly or len(arena_poly) < 3:
        return None
    pts = np.array([[int(p[0]), int(p[1])] for p in arena_poly], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _top_n_blobs(
    contours: list,
    n: int,
    min_area: float,
    max_area: float,
) -> list[tuple[float, float, float]]:
    """Return up to n (cx, cy, area) tuples for the n largest valid blobs."""
    valid = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area or area > max_area:
            continue
        m = cv2.moments(c)
        if m["m00"] <= 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        valid.append((area, cx, cy))
    valid.sort(key=lambda t: t[0], reverse=True)
    return [(cx, cy, area) for area, cx, cy in valid[:n]]


def track_centroid(
    video_path: str,
    cfg: TrackConfig = TrackConfig(),
    progress_cb: Callable[[float], None] | None = None,
) -> TrackingResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Trim window in frames (inclusive start, exclusive end)
    trim_start_frame = int(cfg.trim_start_s * fps) if cfg.trim_start_s > 0 else 0
    trim_end_frame = int(cfg.trim_end_s * fps) if cfg.trim_end_s > 0 else (total_frames or 10**9)

    arena_mask = _build_arena_mask(h, w, cfg.arena_poly)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=cfg.history,
        varThreshold=cfg.var_threshold,
        detectShadows=cfg.detect_shadows,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # Per-animal state for speed, heading tracking
    n = max(1, cfg.n_animals)
    animal_ids = [f"animal_{i+1}" for i in range(n)]
    prev_ok_idx: list[int | None] = [None] * n
    # Baseline area per animal (exponential moving average)
    baseline_area: list[float | None] = [None] * n
    # Last known centroid per animal for Hungarian assignment
    prev_positions: list[tuple[float, float] | None] = [None] * n

    raw_frames: list[dict[str, Any]] = []
    frame_idx = 0
    warmup_done = False
    warmup_count = 0

    while True:
        ok, bgr = cap.read()
        if not ok:
            if not warmup_done:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                warmup_done = True
                frame_idx = 0
                continue
            break

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if arena_mask is not None:
            gray = cv2.bitwise_and(gray, gray, mask=arena_mask)

        fg = subtractor.apply(gray)

        if not warmup_done:
            warmup_count += 1
            if warmup_count >= cfg.warmup_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                warmup_done = True
                frame_idx = 0
            continue

        _, bw = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)
        if arena_mask is not None:
            bw = cv2.bitwise_and(bw, arena_mask)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Get ALL valid blobs (not just n) so Hungarian can pick the best assignment
        all_blobs = _top_n_blobs(contours, len(contours), cfg.min_area_px, cfg.max_area_px)
        curr_centroids = [(cx, cy) for cx, cy, _ in all_blobs]

        # Assign blobs to animal IDs using Hungarian algorithm
        if all(p is None for p in prev_positions):
            # First detections: assign n largest blobs in order
            assignments: list[int | None] = [
                i if i < len(curr_centroids) else None for i in range(n)
            ]
        else:
            assignments = hungarian_assign(prev_positions, curr_centroids)

        # Build per-animal frame entries
        animals: list[dict[str, Any]] = []
        for ai in range(n):
            aid = animal_ids[ai]
            blob_idx = assignments[ai] if ai < len(assignments) else None
            if blob_idx is not None and blob_idx < len(all_blobs):
                cx, cy, area = all_blobs[blob_idx]
                # Update rearing baseline (EMA)
                if baseline_area[ai] is None:
                    baseline_area[ai] = area
                else:
                    baseline_area[ai] = baseline_area[ai] * 0.97 + area * 0.03  # type: ignore[operator]
                rearing = (
                    baseline_area[ai] is not None
                    and area < cfg.rearing_area_fraction * baseline_area[ai]  # type: ignore[operator]
                )
                animals.append({
                    "animal_id": aid,
                    "ok": True,
                    "centroid": {"x": cx, "y": cy},
                    "area_px": area,
                    "rearing": rearing,
                    "quality": "ok",
                    "speed_px_s": None,
                    "speed_cm_s": None,
                    "heading_deg": None,
                    "heading_delta_deg": None,
                })
                prev_positions[ai] = (cx, cy)
            else:
                animals.append({
                    "animal_id": aid,
                    "ok": False,
                    "centroid": None,
                    "area_px": None,
                    "rearing": False,
                    "quality": "no_detection",
                    "speed_px_s": None,
                    "speed_cm_s": None,
                    "heading_deg": None,
                    "heading_delta_deg": None,
                })
                # Keep prev_positions[ai] as-is so recovery works

        # Outside trim window: still process background model, but mark as trimmed
        if frame_idx < trim_start_frame or frame_idx >= trim_end_frame:
            raw_frames.append({
                "frame_index": frame_idx,
                "t_sec": frame_idx / fps,
                "ok": False,
                "centroid": None,
                "area_px": None,
                "rearing": False,
                "quality": "trimmed",
                "zone_id": None,
                "speed_px_s": None,
                "speed_cm_s": None,
                "heading_deg": None,
                "heading_delta_deg": None,
                "angular_velocity_deg_s": None,
                "animals": [{**a, "ok": False, "centroid": None, "quality": "trimmed"} for a in animals],
            })
            frame_idx += 1
            if progress_cb and frame_idx % 50 == 0:
                pct = frame_idx / max(1, total_frames)
                progress_cb(min(0.98, pct * 0.85))
            continue

        raw_frames.append({
            "frame_index": frame_idx,
            "t_sec": frame_idx / fps,
            "ok": animals[0]["ok"],         # primary animal
            "centroid": animals[0]["centroid"],
            "area_px": animals[0]["area_px"],
            "rearing": animals[0]["rearing"],
            "quality": animals[0]["quality"],
            "zone_id": None,
            "speed_px_s": None,
            "speed_cm_s": None,
            "heading_deg": None,
            "heading_delta_deg": None,
            "angular_velocity_deg_s": None,
            # Multi-animal data lives here (not serialised to TrackingFrame model,
            # but available to metrics.py via raw_frames)
            "animals": animals,
        })
        frame_idx += 1

        if progress_cb and frame_idx % 50 == 0:
            pct = frame_idx / max(1, total_frames)
            progress_cb(min(0.98, pct * 0.85))

    cap.release()

    if progress_cb:
        progress_cb(0.87)

    # Smooth and compute speed/heading for every animal track
    for ai in range(n):
        # Build per-animal frame list for Kalman smoother
        animal_frames = [
            {
                **f["animals"][ai],
                "frame_index": f["frame_index"],
                "t_sec": f["t_sec"],
            }
            for f in raw_frames
        ]
        smoothed_a = kalman_smooth(animal_frames)

        prev_idx: int | None = None
        for i, sf in enumerate(smoothed_a):
            if sf.get("ok") and sf.get("centroid"):
                if prev_idx is not None:
                    pf = smoothed_a[prev_idx]
                    dx = sf["centroid"]["x"] - pf["centroid"]["x"]
                    dy = sf["centroid"]["y"] - pf["centroid"]["y"]
                    dt = sf["t_sec"] - pf["t_sec"]
                    dist = math.hypot(dx, dy)
                    if dt > 0:
                        spx = dist / dt
                        sf["speed_px_s"] = spx
                        if cfg.px_per_cm > 0:
                            sf["speed_cm_s"] = spx / cfg.px_per_cm
                    if dist > 1.0:
                        sf["heading_deg"] = math.degrees(math.atan2(dy, dx))
                        if pf.get("heading_deg") is not None:
                            diff = sf["heading_deg"] - pf["heading_deg"]
                            while diff > 180: diff -= 360
                            while diff <= -180: diff += 360
                            sf["heading_delta_deg"] = diff
                            if dt > 0:
                                sf["angular_velocity_deg_s"] = diff / dt
                prev_idx = i

            # Write back into raw_frames
            raw_frames[i]["animals"][ai] = sf
            if ai == 0:
                # Primary animal data propagates to top-level fields
                # Preserve the original "trimmed" quality — the Kalman smoother
                # does not know about the trim window and would overwrite it.
                orig_quality = raw_frames[i].get("quality")
                raw_frames[i]["ok"] = sf.get("ok", False)
                raw_frames[i]["centroid"] = sf.get("centroid")
                raw_frames[i]["area_px"] = sf.get("area_px")
                raw_frames[i]["rearing"] = sf.get("rearing", False)
                raw_frames[i]["quality"] = orig_quality if orig_quality == "trimmed" else sf.get("quality", "ok")
                raw_frames[i]["speed_px_s"] = sf.get("speed_px_s")
                raw_frames[i]["speed_cm_s"] = sf.get("speed_cm_s")
                raw_frames[i]["heading_deg"] = sf.get("heading_deg")
                raw_frames[i]["heading_delta_deg"] = sf.get("heading_delta_deg")
                raw_frames[i]["angular_velocity_deg_s"] = sf.get("angular_velocity_deg_s")

    # Inter-animal distance (when n >= 2)
    if n >= 2:
        for f in raw_frames:
            a0 = f["animals"][0]
            a1 = f["animals"][1]
            if a0.get("ok") and a1.get("ok") and a0.get("centroid") and a1.get("centroid"):
                d = math.hypot(
                    a0["centroid"]["x"] - a1["centroid"]["x"],
                    a0["centroid"]["y"] - a1["centroid"]["y"],
                )
                f["inter_animal_dist_px"] = d
                if cfg.px_per_cm > 0:
                    f["inter_animal_dist_cm"] = d / cfg.px_per_cm
            else:
                f["inter_animal_dist_px"] = None
                f["inter_animal_dist_cm"] = None

    ok_frames = sum(1 for f in raw_frames if f.get("ok"))

    if progress_cb:
        progress_cb(0.95)

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
    return result


def extract_first_frame_jpeg(video_path: str) -> bytes:
    """Return JPEG bytes of the first video frame."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read first frame")
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()
