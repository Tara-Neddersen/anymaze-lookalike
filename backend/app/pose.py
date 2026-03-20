"""
Pose-file parsers for DeepLabCut (CSV) and SLEAP (.slp / .h5 / .csv).

Both return a list of dicts in a *standardised keypoint format*:
  {
    frame_index: int,
    keypoints: {bodypart_name: {x: float, y: float, likelihood: float}},
    multi_animal: bool          # only True for maDLC / multi-instance SLEAP
    individuals: {ind_id: {bodypart: {x,y,likelihood}}}   # only when multi_animal
  }

The helper `pose_to_raw_frames()` converts this into the raw_frames list
expected by metrics.py.
"""
from __future__ import annotations

import csv
import math
from typing import Any


# ---------------------------------------------------------------------------
# Body-part name aliases  (case-insensitive)
# Standard 7-point rodent skeleton: nose, left_ear, right_ear, neck,
#   mid_spine, hips, tail_base — plus extras.
# ---------------------------------------------------------------------------
_CENTROID_ALIASES = ("mid_spine", "midspine", "spine2", "spine1", "center",
                     "body", "thorax", "mid", "midbody", "mid_body", "back")
_NOSE_ALIASES     = ("nose", "snout", "head", "rostrum")
_TAIL_ALIASES     = ("tail_base", "tailbase", "tail1", "tail_root",
                     "tailroot", "base_tail")
_EAR_L_ALIASES    = ("left_ear", "leftear", "lear", "ear_left",
                     "l_ear", "ear_l")
_EAR_R_ALIASES    = ("right_ear", "rightear", "rear", "ear_right",
                     "r_ear", "ear_r")
_NECK_ALIASES     = ("neck", "throat", "nape", "cervical")
_SPINE_ALIASES    = ("mid_spine", "midspine", "spine_mid", "mid_back",
                     "dorsum", "back_mid", "spine2")
_HIP_ALIASES      = ("hips", "hip", "rump", "sacrum", "pelvis",
                     "haunch", "tail_root_up")

# Canonical 7-keypoint names used in metrics and visualisation
CANONICAL_KPS = ["nose", "left_ear", "right_ear", "neck",
                 "mid_spine", "hips", "tail_base"]

# Skeleton connections for drawing
SKELETON_EDGES = [
    ("nose",      "left_ear"),
    ("nose",      "right_ear"),
    ("nose",      "neck"),
    ("left_ear",  "neck"),
    ("right_ear", "neck"),
    ("neck",      "mid_spine"),
    ("mid_spine", "hips"),
    ("hips",      "tail_base"),
]

# Colour per canonical keypoint (BGR for OpenCV, hex for canvas)
KP_COLORS_HEX = {
    "nose":       "#FF4C4C",
    "left_ear":   "#FFD93D",
    "right_ear":  "#FFD93D",
    "neck":       "#6BCB77",
    "mid_spine":  "#00DDB4",
    "hips":       "#4D96FF",
    "tail_base":  "#C77DFF",
}


def _best(kps: dict[str, dict], aliases: tuple[str, ...]) -> dict | None:
    """Find a keypoint by trying alias names (exact, then case-insensitive)."""
    for name in aliases:
        kp = kps.get(name)
        if kp:
            return kp
    kps_lo = {k.lower(): v for k, v in kps.items()}
    for name in aliases:
        kp = kps_lo.get(name.lower())
        if kp:
            return kp
    return None


def resolve_canonical(kps: dict[str, dict]) -> dict[str, dict | None]:
    """
    Map raw keypoint names to the 7 canonical landmarks.
    Returns {"nose": {...} | None, "left_ear": ..., ...}
    """
    return {
        "nose":      _best(kps, _NOSE_ALIASES),
        "left_ear":  _best(kps, _EAR_L_ALIASES),
        "right_ear": _best(kps, _EAR_R_ALIASES),
        "neck":      _best(kps, _NECK_ALIASES),
        "mid_spine": _best(kps, _SPINE_ALIASES),
        "hips":      _best(kps, _HIP_ALIASES),
        "tail_base": _best(kps, _TAIL_ALIASES),
    }


def _centroid_from_kps(kps: dict[str, dict]) -> dict | None:
    # Prefer mid-spine, then body-part averages
    kp = _best(kps, _CENTROID_ALIASES)
    if kp:
        return {"x": kp["x"], "y": kp["y"]}
    if not kps:
        return None
    xs = [v["x"] for v in kps.values() if "x" in v]
    ys = [v["y"] for v in kps.values() if "y" in v]
    if not xs:
        return None
    return {"x": sum(xs) / len(xs), "y": sum(ys) / len(ys)}


def _heading_from_kps(kps: dict[str, dict]) -> float | None:
    nose = _best(kps, _NOSE_ALIASES)
    tail = _best(kps, _TAIL_ALIASES)
    if nose and tail:
        dx = nose["x"] - tail["x"]
        dy = nose["y"] - tail["y"]
        if abs(dx) > 0.5 or abs(dy) > 0.5:
            return round(math.degrees(math.atan2(dy, dx)), 2)
    return None


def _body_length(kps: dict[str, dict]) -> float | None:
    """Nose → tail_base Euclidean distance."""
    nose = _best(kps, _NOSE_ALIASES)
    tail = _best(kps, _TAIL_ALIASES)
    if nose and tail:
        return round(math.hypot(nose["x"] - tail["x"], nose["y"] - tail["y"]), 2)
    return None


def _ear_span(kps: dict[str, dict]) -> float | None:
    """Left ear → right ear width (proxy for head size / grooming spread)."""
    le = _best(kps, _EAR_L_ALIASES)
    re = _best(kps, _EAR_R_ALIASES)
    if le and re:
        return round(math.hypot(le["x"] - re["x"], le["y"] - re["y"]), 2)
    return None


def _head_angle(kps: dict[str, dict]) -> float | None:
    """
    Head direction: angle of the nose relative to the neck (or ear midpoint).
    This differs from body heading when the animal turns its head.
    """
    nose = _best(kps, _NOSE_ALIASES)
    neck = _best(kps, _NECK_ALIASES)
    if nose and neck:
        dx, dy = nose["x"] - neck["x"], nose["y"] - neck["y"]
        if abs(dx) > 0.3 or abs(dy) > 0.3:
            return round(math.degrees(math.atan2(dy, dx)), 2)
    # Fallback: use ear midpoint as proxy for neck
    le = _best(kps, _EAR_L_ALIASES)
    re = _best(kps, _EAR_R_ALIASES)
    if nose and le and re:
        mx = (le["x"] + re["x"]) / 2
        my = (le["y"] + re["y"]) / 2
        dx, dy = nose["x"] - mx, nose["y"] - my
        if abs(dx) > 0.3 or abs(dy) > 0.3:
            return round(math.degrees(math.atan2(dy, dx)), 2)
    return None


def _spine_curvature(kps: dict[str, dict]) -> float | None:
    """
    Spine curvature: deviation of mid-spine from the nose→tail axis (0 = straight).
    Computed as the perpendicular distance of mid-spine from the nose–tailbase line,
    normalised by body length. High values indicate hunching / grooming posture.
    """
    nose  = _best(kps, _NOSE_ALIASES)
    mid   = _best(kps, _SPINE_ALIASES)
    tail  = _best(kps, _TAIL_ALIASES)
    if not (nose and mid and tail):
        return None
    # Line from nose to tail
    dx, dy = tail["x"] - nose["x"], tail["y"] - nose["y"]
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    # Perpendicular distance of mid-spine from that line
    px = mid["x"] - nose["x"]
    py = mid["y"] - nose["y"]
    perp = abs(px * dy - py * dx) / length  # cross product / |line|
    return round(perp / length, 4)           # normalised


def _head_body_angle(kps: dict[str, dict]) -> float | None:
    """
    Angle between head direction (nose→neck) and body axis (neck→hips).
    Large angles indicate head-turning or grooming.
    """
    nose  = _best(kps, _NOSE_ALIASES)
    neck  = _best(kps, _NECK_ALIASES)
    hips  = _best(kps, _HIP_ALIASES)
    if not (nose and neck and hips):
        return None
    # Vector: neck→nose (head direction)
    hx, hy = nose["x"] - neck["x"], nose["y"] - neck["y"]
    # Vector: neck→hips (body direction, reversed)
    bx, by = hips["x"] - neck["x"], hips["y"] - neck["y"]
    cos_a = (hx*bx + hy*by) / max(math.hypot(hx, hy) * math.hypot(bx, by), 1e-6)
    cos_a = max(-1.0, min(1.0, cos_a))
    return round(math.degrees(math.acos(cos_a)), 2)


def _is_grooming(kps: dict[str, dict]) -> bool:
    """
    Grooming heuristic: nose is close to the forepaw region (below neck in top view).
    Proxy: angle between nose and neck > 90° relative to body axis, AND ear span is
    large (arms spread). If neck not available, use spine curvature > 0.25.
    """
    curv = _spine_curvature(kps)
    if curv is not None:
        return curv > 0.25
    hba = _head_body_angle(kps)
    if hba is not None:
        return hba > 90.0
    return False


def _is_rearing(kps: dict[str, dict], baseline_body_len: float | None) -> bool:
    """
    Rearing heuristic for top-view cameras:
    Body collapses into a small footprint when the animal rears.
    Threshold: current body length < 55% of running EMA baseline.
    """
    bl = _body_length(kps)
    if bl is not None and baseline_body_len and baseline_body_len > 0:
        return bl < 0.55 * baseline_body_len
    return False


# ---------------------------------------------------------------------------
# DeepLabCut CSV parser
# ---------------------------------------------------------------------------

def parse_dlc_csv(path: str, min_likelihood: float = 0.6) -> list[dict[str, Any]]:
    """
    Parse a DeepLabCut output CSV.
    Handles both single-animal (3 header rows) and multi-animal maDLC
    (4 header rows that include an 'individuals' row).
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if len(rows) < 4:
        raise ValueError("DLC CSV must have at least 4 rows (3 headers + 1 data)")

    # Detect multi-animal layout: second row starts with 'individuals'
    if rows[1] and rows[1][0].strip().lower() == "individuals":
        return _parse_dlc_multi(rows, min_likelihood)
    return _parse_dlc_single(rows, min_likelihood)


def _parse_dlc_single(rows: list[list[str]], min_likelihood: float) -> list[dict[str, Any]]:
    """3-header-row single-animal DLC CSV."""
    bodyparts = rows[1][1:]
    coords    = rows[2][1:]

    col_map: dict[tuple[str, str], int] = {}
    for i, (bp, coord) in enumerate(zip(bodyparts, coords)):
        col_map[(bp.strip(), coord.strip())] = i + 1

    unique_bps = list(dict.fromkeys(b.strip() for b in bodyparts))
    frames: list[dict[str, Any]] = []

    for row in rows[3:]:
        if not row or not row[0].strip():
            continue
        try:
            fi = int(float(row[0]))
        except ValueError:
            continue
        kps: dict[str, dict[str, float]] = {}
        for bp in unique_bps:
            xi = col_map.get((bp, "x"))
            yi = col_map.get((bp, "y"))
            li = col_map.get((bp, "likelihood"))
            if xi is None or yi is None or xi >= len(row) or yi >= len(row):
                continue
            try:
                x = float(row[xi])
                y = float(row[yi])
                likelihood = float(row[li]) if (li is not None and li < len(row)) else 1.0
                if likelihood >= min_likelihood and not (math.isnan(x) or math.isnan(y)):
                    kps[bp] = {"x": x, "y": y, "likelihood": likelihood}
            except (ValueError, IndexError):
                continue
        frames.append({"frame_index": fi, "keypoints": kps, "multi_animal": False})
    return frames


def _parse_dlc_multi(rows: list[list[str]], min_likelihood: float) -> list[dict[str, Any]]:
    """4-header-row multi-animal maDLC CSV."""
    individuals = [s.strip() for s in rows[1][1:]]
    bodyparts   = [s.strip() for s in rows[2][1:]]
    coords      = [s.strip() for s in rows[3][1:]]

    col_map: dict[tuple[str, str, str], int] = {}
    for i, (ind, bp, coord) in enumerate(zip(individuals, bodyparts, coords)):
        col_map[(ind, bp, coord)] = i + 1

    unique_inds = list(dict.fromkeys(individuals))
    bps_per_ind: dict[str, list[str]] = {}
    for ind, bp in zip(individuals, bodyparts):
        bps_per_ind.setdefault(ind, [])
        if bp not in bps_per_ind[ind]:
            bps_per_ind[ind].append(bp)

    frames: list[dict[str, Any]] = []
    for row in rows[4:]:
        if not row or not row[0].strip():
            continue
        try:
            fi = int(float(row[0]))
        except ValueError:
            continue
        ind_kps: dict[str, dict[str, dict[str, float]]] = {}
        for ind in unique_inds:
            kps: dict[str, dict[str, float]] = {}
            for bp in bps_per_ind.get(ind, []):
                xi = col_map.get((ind, bp, "x"))
                yi = col_map.get((ind, bp, "y"))
                li = col_map.get((ind, bp, "likelihood"))
                if xi is None or yi is None or xi >= len(row) or yi >= len(row):
                    continue
                try:
                    x = float(row[xi])
                    y = float(row[yi])
                    likelihood = float(row[li]) if (li is not None and li < len(row)) else 1.0
                    if likelihood >= min_likelihood and not (math.isnan(x) or math.isnan(y)):
                        kps[bp] = {"x": x, "y": y, "likelihood": likelihood}
                except (ValueError, IndexError):
                    continue
            ind_kps[ind] = kps
        frames.append({
            "frame_index": fi,
            "keypoints": ind_kps,       # dict of ind → {bp: {x,y,l}}
            "multi_animal": True,
        })
    return frames


# ---------------------------------------------------------------------------
# SLEAP .slp / exported CSV parser
# ---------------------------------------------------------------------------

def parse_sleap_slp(path: str) -> list[dict[str, Any]]:
    """
    Parse a SLEAP .slp predictions file using sleap-io.
    Returns standardised keypoint frames.
    """
    try:
        import sleap_io as sio
    except ImportError:
        raise ImportError("sleap-io is required: pip install sleap-io>=0.6.5")

    labels = sio.load_file(path)
    skeleton_nodes = [n.name for n in labels.skeleton.nodes]
    frames: list[dict[str, Any]] = []

    for lf in labels.labeled_frames:
        fi = lf.frame_idx
        instances = list(lf.instances)
        if not instances:
            frames.append({"frame_index": fi, "keypoints": {}, "multi_animal": False})
            continue

        if len(instances) == 1:
            kps = _sleap_instance_kps(instances[0], skeleton_nodes)
            frames.append({"frame_index": fi, "keypoints": kps, "multi_animal": False})
        else:
            ind_kps: dict[str, dict] = {}
            for ai, inst in enumerate(instances):
                ind_id = f"ind{ai + 1}"
                ind_kps[ind_id] = _sleap_instance_kps(inst, skeleton_nodes)
            frames.append({"frame_index": fi, "keypoints": ind_kps, "multi_animal": True})

    frames.sort(key=lambda f: f["frame_index"])
    return frames


def _sleap_instance_kps(instance: Any, node_names: list[str]) -> dict[str, dict[str, float]]:
    kps: dict[str, dict[str, float]] = {}
    try:
        pts = instance.numpy()   # shape (n_nodes, 2)
    except Exception:
        return kps
    score = 1.0
    if hasattr(instance, "score") and instance.score is not None:
        try:
            score = float(instance.score)
        except Exception:
            pass
    for i, name in enumerate(node_names):
        if i >= len(pts):
            break
        x, y = float(pts[i, 0]), float(pts[i, 1])
        if not (math.isnan(x) or math.isnan(y)):
            kps[name] = {"x": x, "y": y, "likelihood": score}
    return kps


# ---------------------------------------------------------------------------
# Convert pose frames → raw_frames (metrics.py compatible)
# ---------------------------------------------------------------------------

def pose_to_raw_frames(
    pose_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    n_animals: int = 1,
) -> list[dict[str, Any]]:
    """
    Convert parsed pose data into the raw_frames list expected by metrics.py.
    Handles single-animal and multi-animal data.
    """
    is_multi = any(f.get("multi_animal") for f in pose_frames)
    if is_multi and n_animals > 1:
        return _multi_pose_to_raw(pose_frames, fps, px_per_cm, n_animals)
    return _single_pose_to_raw(pose_frames, fps, px_per_cm)


def _single_pose_to_raw(
    pose_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    ema_body_len: float | None = None
    prev_heading: float | None = None

    for pf in pose_frames:
        fi = pf["frame_index"]
        kps: dict[str, dict[str, float]] = pf.get("keypoints") or {}
        t_sec = fi / fps
        canonical = resolve_canonical(kps)
        centroid = _centroid_from_kps(kps) if kps else None
        ok = centroid is not None and len(kps) >= 1
        heading   = _heading_from_kps(kps) if kps else None
        body_len  = _body_length(kps) if kps else None
        ear_span  = _ear_span(kps) if kps else None
        head_ang  = _head_angle(kps) if kps else None
        spine_curv = _spine_curvature(kps) if kps else None
        head_body  = _head_body_angle(kps) if kps else None
        grooming   = _is_grooming(kps) if kps else False

        if body_len is not None:
            ema_body_len = (body_len if ema_body_len is None
                            else ema_body_len * 0.97 + body_len * 0.03)

        rearing = _is_rearing(kps, ema_body_len) if kps else False

        heading_delta: float | None = None
        if heading is not None and prev_heading is not None:
            delta = heading - prev_heading
            while delta > 180:  delta -= 360
            while delta <= -180: delta += 360
            heading_delta = round(delta, 2)
        if heading is not None:
            prev_heading = heading

        raw.append({
            "frame_index": fi,
            "t_sec": t_sec,
            "ok": ok,
            "centroid": centroid,
            "area_px": body_len,
            "rearing": rearing,
            "grooming": grooming,
            "quality": "ok" if ok else "no_detection",
            "zone_id": None,
            "speed_px_s": None,
            "speed_cm_s": None,
            "heading_deg": heading,
            "heading_delta_deg": heading_delta,
            "body_length_px": body_len,
            "ear_span_px": ear_span,
            "head_angle_deg": head_ang,
            "spine_curvature": spine_curv,
            "head_body_angle_deg": head_body,
            # All raw keypoints (full DLC/SLEAP output)
            "keypoints": kps,
            # 7-canonical resolved
            "canonical_kps": {k: v for k, v in canonical.items() if v is not None},
            "animals": [{
                "animal_id": "animal_1",
                "ok": ok,
                "centroid": centroid,
                "area_px": body_len,
                "rearing": rearing,
                "grooming": grooming,
                "quality": "ok" if ok else "no_detection",
                "speed_px_s": None,
                "speed_cm_s": None,
                "heading_deg": heading,
                "heading_delta_deg": heading_delta,
                "body_length_px": body_len,
                "keypoints": kps,
                "canonical_kps": {k: v for k, v in canonical.items() if v is not None},
            }],
        })

    _fill_speeds(raw, fps, px_per_cm)
    return raw


def _multi_pose_to_raw(
    pose_frames: list[dict[str, Any]],
    fps: float,
    px_per_cm: float,
    n_animals: int,
) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    ema_lens: dict[str, float] = {}
    prev_headings: dict[str, float] = {}

    for pf in pose_frames:
        fi = pf["frame_index"]
        t_sec = fi / fps
        ind_kps: dict[str, dict] = pf.get("keypoints") or {}
        animals: list[dict[str, Any]] = []

        for ai, (ind_id, kps) in enumerate(list(ind_kps.items())[:n_animals]):
            canonical = resolve_canonical(kps)
            centroid  = _centroid_from_kps(kps) if kps else None
            ok        = centroid is not None and len(kps) >= 1
            heading   = _heading_from_kps(kps) if kps else None
            body_len  = _body_length(kps) if kps else None
            ear_span  = _ear_span(kps) if kps else None
            head_ang  = _head_angle(kps) if kps else None
            spine_curv = _spine_curvature(kps) if kps else None
            head_body  = _head_body_angle(kps) if kps else None
            grooming   = _is_grooming(kps) if kps else False

            if body_len is not None:
                prev_bl = ema_lens.get(ind_id)
                ema_lens[ind_id] = (body_len if prev_bl is None
                                    else prev_bl * 0.97 + body_len * 0.03)

            rearing = _is_rearing(kps, ema_lens.get(ind_id)) if kps else False

            prev_h = prev_headings.get(ind_id)
            heading_delta: float | None = None
            if heading is not None and prev_h is not None:
                delta = heading - prev_h
                while delta > 180:  delta -= 360
                while delta <= -180: delta += 360
                heading_delta = round(delta, 2)
            if heading is not None:
                prev_headings[ind_id] = heading

            animals.append({
                "animal_id": ind_id,
                "ok": ok,
                "centroid": centroid,
                "area_px": body_len,
                "rearing": rearing,
                "grooming": grooming,
                "quality": "ok" if ok else "no_detection",
                "speed_px_s": None,
                "speed_cm_s": None,
                "heading_deg": heading,
                "heading_delta_deg": heading_delta,
                "body_length_px": body_len,
                "ear_span_px": ear_span,
                "head_angle_deg": head_ang,
                "spine_curvature": spine_curv,
                "head_body_angle_deg": head_body,
                "keypoints": kps,
                "canonical_kps": {k: v for k, v in canonical.items() if v is not None},
            })

        # Pad missing animals
        while len(animals) < n_animals:
            animals.append({
                "animal_id": f"ind{len(animals)+1}",
                "ok": False, "centroid": None, "area_px": None,
                "rearing": False, "quality": "no_detection",
                "speed_px_s": None, "speed_cm_s": None,
                "heading_deg": None, "heading_delta_deg": None,
                "body_length_px": None, "keypoints": {},
            })

        primary = animals[0]
        raw.append({
            "frame_index": fi,
            "t_sec": t_sec,
            "ok": primary["ok"],
            "centroid": primary["centroid"],
            "area_px": primary["area_px"],
            "rearing": primary["rearing"],
            "grooming": primary.get("grooming", False),
            "quality": primary["quality"],
            "zone_id": None,
            "speed_px_s": None,
            "speed_cm_s": None,
            "heading_deg": primary["heading_deg"],
            "heading_delta_deg": primary["heading_delta_deg"],
            "body_length_px": primary.get("body_length_px"),
            "ear_span_px": primary.get("ear_span_px"),
            "head_angle_deg": primary.get("head_angle_deg"),
            "spine_curvature": primary.get("spine_curvature"),
            "head_body_angle_deg": primary.get("head_body_angle_deg"),
            "keypoints": list(ind_kps.values())[0] if ind_kps else {},
            "canonical_kps": primary.get("canonical_kps", {}),
            "animals": animals,
        })

    # Inter-animal distances
    if n_animals >= 2:
        for f in raw:
            ans = f.get("animals", [])
            if (len(ans) >= 2 and ans[0].get("ok") and ans[1].get("ok")
                    and ans[0].get("centroid") and ans[1].get("centroid")):
                a0, a1 = ans[0]["centroid"], ans[1]["centroid"]
                d = math.hypot(a0["x"] - a1["x"], a0["y"] - a1["y"])
                f["inter_animal_dist_px"] = d
                f["inter_animal_dist_cm"] = d / px_per_cm if px_per_cm > 0 else None

    _fill_speeds(raw, fps, px_per_cm)
    return raw


def _fill_speeds(raw: list[dict[str, Any]], fps: float, px_per_cm: float) -> None:
    """Fill speed_px_s / speed_cm_s from consecutive centroid displacement."""
    prev: dict | None = None
    for f in raw:
        if f.get("ok") and f.get("centroid"):
            if prev and prev.get("centroid"):
                dx = f["centroid"]["x"] - prev["centroid"]["x"]
                dy = f["centroid"]["y"] - prev["centroid"]["y"]
                dt = f["t_sec"] - prev["t_sec"]
                if dt > 0:
                    spx = math.hypot(dx, dy) / dt
                    f["speed_px_s"] = spx
                    if px_per_cm > 0:
                        f["speed_cm_s"] = spx / px_per_cm
            prev = f
        else:
            prev = None


# ---------------------------------------------------------------------------
# Video metadata extraction for pose pipelines
# ---------------------------------------------------------------------------

def get_video_meta(video_path: str) -> dict[str, Any]:
    """Return fps, width, height from a video file (needs opencv)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"fps": fps, "width": w, "height": h}
