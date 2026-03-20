"""
Default thresholds for pose QC decision tiers (Acceptable / Borderline / Poor).

Reproducibility: all defaults are explicit; override via compute_pose_qc(..., confidence_threshold=...).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PoseQCThresholds:
    """τ and tier cutoffs."""

    confidence_tau: float = 0.6

    # valid_frame_fraction_all7
    valid_frac_acceptable: float = 0.75
    valid_frac_borderline_low: float = 0.50  # below this → poor (C) for this metric

    # min across keypoints of mean likelihood (when present)
    min_kp_likelihood_acceptable: float = 0.70
    min_kp_likelihood_borderline_low: float = 0.50

    # body length CV on frames with nose+tail
    body_cv_acceptable: float = 0.08
    body_cv_borderline_high: float = 0.15

    # max consecutive invalid gap (frames)
    max_gap_acceptable: int = 15
    max_gap_borderline_high: int = 45

    # jitter: median pairwise displacement (px), aggregated as mean of per-KP medians
    jitter_px_acceptable: float = 3.0
    jitter_px_borderline_high: float = 6.0

    # scale-free jitter alternative: median_disp / median_body_length
    jitter_norm_acceptable: float = 0.025
    jitter_norm_borderline_high: float = 0.05


DEFAULT_THRESHOLDS = PoseQCThresholds()

CANONICAL_SCHEMA_VERSION = "7pt_v1"
POSE_QC_SCHEMA_VERSION = "pose_qc_v1"
