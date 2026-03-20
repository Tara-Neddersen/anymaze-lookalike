"""
pose_features.py — Translation/rotation/scale-invariant pose feature vectors.

For each frame that has a canonical_kps dict (produced by pose.py:resolve_canonical),
this module computes a 20-dimensional feature vector suitable for clustering,
embedding, and downstream machine learning.

Feature vector (20 dims):
  [0-11]  6 keypoint (x, y) pairs after normalization (mid_spine is excluded as
          it becomes the origin, so nose + left_ear + right_ear + neck + hips +
          tail_base = 6 kps × 2 = 12 dims)
  [12]    speed_norm         — speed / 95th-pct session speed
  [13]    angular_vel_norm   — |heading_delta_deg/s| / 95th-pct
  [14]    spine_curvature    — already normalised (0..~0.5)
  [15]    sin(head_body_angle)
  [16]    cos(head_body_angle)
  [17]    ear_span_norm      — ear_span_px / body_length_px
  [18]    grooming           — 0 or 1
  [19]    rearing            — 0 or 1
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

# Canonical KPs that appear in the output vector (mid_spine excluded — it's the origin)
_VECTOR_KPS = ["nose", "left_ear", "right_ear", "neck", "hips", "tail_base"]
FEATURE_DIM = 20

# Human-readable dimension labels
FEATURE_NAMES = [
    "nose_x", "nose_y",
    "left_ear_x", "left_ear_y",
    "right_ear_x", "right_ear_y",
    "neck_x", "neck_y",
    "hips_x", "hips_y",
    "tail_base_x", "tail_base_y",
    "speed_norm",
    "angular_vel_norm",
    "spine_curvature",
    "sin_head_body_angle",
    "cos_head_body_angle",
    "ear_span_norm",
    "grooming",
    "rearing",
]


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def normalize_pose_frame(canonical_kps: dict[str, dict]) -> np.ndarray | None:
    """
    Normalize a single frame's canonical keypoints into a 20-dim feature vector.

    Steps:
      1. Require nose + tail_base (minimum anchor pair).
      2. Translate origin to mid_spine (if present) else nose.
      3. Scale: divide by body_length (nose → tail_base).
      4. Rotate: align nose → tail_base with positive x-axis.

    Returns np.ndarray shape (20,) or None if insufficient keypoints.
    """
    nose = canonical_kps.get("nose")
    tail = canonical_kps.get("tail_base")
    if nose is None or tail is None:
        return None

    # Body axis vector
    body_vec = np.array([tail["x"] - nose["x"], tail["y"] - nose["y"]], dtype=float)
    body_len = float(np.linalg.norm(body_vec))
    if body_len < 1.0:
        return None

    # Choose origin: mid_spine if available, else nose
    origin_kp = canonical_kps.get("mid_spine") or nose
    origin = np.array([origin_kp["x"], origin_kp["y"]], dtype=float)

    # Rotation angle: align nose→tail with +x
    angle = -math.atan2(body_vec[1], body_vec[0])
    R = _rotation_matrix(angle)

    # Count valid keypoints (excluding mid_spine which becomes origin)
    valid_count = sum(
        1 for k in _VECTOR_KPS if canonical_kps.get(k) is not None
    )
    if valid_count < 4:
        return None  # too few landmarks

    # Build the 12 position features
    pos_features: list[float] = []
    for kp_name in _VECTOR_KPS:
        kp = canonical_kps.get(kp_name)
        if kp is not None:
            v = np.array([kp["x"], kp["y"]], dtype=float) - origin
            v_rot = R @ v / body_len
            pos_features.extend([float(v_rot[0]), float(v_rot[1])])
        else:
            pos_features.extend([float("nan"), float("nan")])

    return np.array(pos_features, dtype=np.float32)


def compute_pose_feature_matrix(
    raw_frames: list[dict[str, Any]],
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the (N_frames, 20) pose feature matrix for a single session.

    Args:
        raw_frames: list of frame dicts from jobs.py / pose.py
        fps: frames per second (for angular velocity normalisation)

    Returns:
        feature_matrix: np.ndarray shape (N, 20), NaN rows for invalid frames
        valid_mask:     np.ndarray bool shape (N,)
    """
    N = len(raw_frames)
    mat = np.full((N, FEATURE_DIM), np.nan, dtype=np.float32)

    # Pre-compute speeds and angular velocities
    speeds: list[float | None] = [f.get("speed_cm_s") for f in raw_frames]
    ang_vels: list[float | None] = []
    for f in raw_frames:
        hd = f.get("heading_delta_deg")
        ang_vels.append(abs(hd) * fps if hd is not None else None)

    # Compute 95th percentile norms (session-level normalisation)
    valid_speeds = [s for s in speeds if s is not None and not math.isnan(s)]
    valid_avels  = [a for a in ang_vels if a is not None and not math.isnan(a)]
    speed_95  = float(np.percentile(valid_speeds, 95)) if valid_speeds else 1.0
    avel_95   = float(np.percentile(valid_avels, 95))  if valid_avels  else 1.0
    speed_95  = max(speed_95, 1e-6)
    avel_95   = max(avel_95,  1e-6)

    for i, f in enumerate(raw_frames):
        ckps = f.get("canonical_kps") or {}
        if not ckps:
            continue

        pos_vec = normalize_pose_frame(ckps)
        if pos_vec is None:
            continue

        # Kinematic scalars
        spd = speeds[i]
        av  = ang_vels[i]
        sc  = f.get("spine_curvature")
        hba = f.get("head_body_angle_deg")
        bl  = f.get("body_length_px")
        es  = f.get("ear_span_px")

        speed_norm = (spd / speed_95) if spd is not None else float("nan")
        avel_norm  = (av  / avel_95)  if av  is not None else float("nan")
        spine_curv = float(sc)        if sc  is not None else float("nan")
        hba_val    = float(hba)       if hba is not None else float("nan")
        sin_hba    = math.sin(math.radians(hba_val)) if not math.isnan(hba_val) else float("nan")
        cos_hba    = math.cos(math.radians(hba_val)) if not math.isnan(hba_val) else float("nan")
        ear_norm   = (es / bl) if (es is not None and bl is not None and bl > 0) else float("nan")
        grooming   = 1.0 if f.get("grooming") else 0.0
        rearing    = 1.0 if f.get("rearing")  else 0.0

        row = np.array(
            list(pos_vec) + [
                speed_norm, avel_norm, spine_curv,
                sin_hba, cos_hba, ear_norm,
                grooming, rearing,
            ],
            dtype=np.float32,
        )
        mat[i] = row

    valid_mask = ~np.any(np.isnan(mat), axis=1)
    return mat, valid_mask


def impute_pose_matrix(
    mat: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation for gaps of ≤ max_gap consecutive invalid frames.
    Returns updated (mat, valid_mask).
    """
    mat = mat.copy()
    valid_mask = valid_mask.copy()
    N = len(mat)

    i = 0
    while i < N:
        if not valid_mask[i]:
            # Find end of invalid run
            j = i
            while j < N and not valid_mask[j]:
                j += 1
            gap = j - i
            if gap <= max_gap and i > 0 and j < N:
                # Interpolate between i-1 and j
                for k in range(gap):
                    alpha = (k + 1) / (gap + 1)
                    mat[i + k] = (1 - alpha) * mat[i - 1] + alpha * mat[j]
                    valid_mask[i + k] = True
            i = j
        else:
            i += 1

    return mat, valid_mask


def valid_fraction(valid_mask: np.ndarray) -> float:
    """Fraction of frames with a valid pose feature vector."""
    if len(valid_mask) == 0:
        return 0.0
    return float(np.sum(valid_mask)) / len(valid_mask)


def apply_pose_qc_mask(
    valid_mask: np.ndarray,
    job_dir: str | Path,
) -> tuple[np.ndarray, bool]:
    """
    AND the feature valid_mask with pose_valid_mask.npy if present and same length.

    Returns (updated_mask, applied).
    """
    job_dir = Path(job_dir)
    p = job_dir / "pose_valid_mask.npy"
    if not p.exists():
        return valid_mask, False
    qc_mask = np.load(p, allow_pickle=False).astype(bool)
    if len(qc_mask) != len(valid_mask):
        return valid_mask, False
    return valid_mask & qc_mask, True
