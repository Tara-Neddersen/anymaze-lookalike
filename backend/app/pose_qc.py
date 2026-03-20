"""
Pose quality control metrics and decision tiers for top-down single-mouse DLC/SLEAP outputs.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .pose import CANONICAL_KPS
from .pose_qc_config import (
    DEFAULT_THRESHOLDS,
    POSE_QC_SCHEMA_VERSION,
    PoseQCThresholds,
)


def _likelihood(kp: dict[str, Any] | None) -> tuple[float, bool]:
    """Return (likelihood, assumed_default_if_missing)."""
    if kp is None:
        return 0.0, False
    if "likelihood" in kp:
        return float(kp["likelihood"]), False
    return 1.0, True


def frame_valid_all7(
    canonical_kps: dict[str, Any] | None,
    tau: float,
) -> bool:
    """True iff all 7 canonical keypoints exist with likelihood >= tau."""
    if not canonical_kps:
        return False
    for name in CANONICAL_KPS:
        kp = canonical_kps.get(name)
        if kp is None:
            return False
        lik, _ = _likelihood(kp if isinstance(kp, dict) else None)
        if lik < tau:
            return False
    return True


def compute_pose_qc_from_frames(
    frames: list[dict[str, Any]],
    *,
    confidence_threshold: float | None = None,
    thresholds: PoseQCThresholds | None = None,
) -> dict[str, Any]:
    """
    Compute metrics from result.json frames (must include canonical_kps where available).

    Returns a dict suitable for pose_qc.json (without file paths).
    """
    th = thresholds or DEFAULT_THRESHOLDS
    tau = float(confidence_threshold if confidence_threshold is not None else th.confidence_tau)

    N = len(frames)
    likelihood_assumed_any = False

    # Per-frame validity (all 7 + lik >= tau)
    valid_all7 = np.zeros(N, dtype=bool)
    # Per-keypoint missingness: absent OR lik < tau
    missing = {k: 0 for k in CANONICAL_KPS}
    lik_sum = {k: 0.0 for k in CANONICAL_KPS}
    lik_count = {k: 0 for k in CANONICAL_KPS}

    for i, f in enumerate(frames):
        ck = f.get("canonical_kps") or {}
        if not isinstance(ck, dict):
            ck = {}
        ok_all = True
        for name in CANONICAL_KPS:
            kp = ck.get(name)
            if kp is None or not isinstance(kp, dict):
                missing[name] += 1
                ok_all = False
                continue
            lik, assumed = _likelihood(kp)
            if assumed:
                likelihood_assumed_any = True
            if lik < tau:
                missing[name] += 1
                ok_all = False
            else:
                lik_sum[name] += lik
                lik_count[name] += 1
        valid_all7[i] = ok_all

    valid_frame_fraction_all7 = float(np.mean(valid_all7)) if N else 0.0

    per_keypoint: dict[str, dict[str, float]] = {}
    for name in CANONICAL_KPS:
        mf = missing[name] / N if N else 0.0
        mc = lik_count[name]
        mean_lik = (lik_sum[name] / mc) if mc else 0.0
        per_keypoint[name] = {
            "missing_fraction": round(mf, 5),
            "mean_likelihood_when_present": round(mean_lik, 5),
            "mean_likelihood": round(mean_lik * (mc / N) if N else 0.0, 5),
        }

    min_mean_likelihood_across_kps = min(
        (per_keypoint[k]["mean_likelihood_when_present"] for k in CANONICAL_KPS),
        default=0.0,
    )

    # Body length L_i = nose — tail_base
    L_list: list[float] = []
    for f in frames:
        ck = f.get("canonical_kps") or {}
        n = ck.get("nose")
        t = ck.get("tail_base")
        if not (n and t):
            continue
        if not isinstance(n, dict) or not isinstance(t, dict):
            continue
        L_list.append(math.hypot(n["x"] - t["x"], n["y"] - t["y"]))

    body_len_arr = np.array(L_list, dtype=float) if L_list else np.array([])
    if len(body_len_arr) > 2:
        med = float(np.median(body_len_arr))
        q1, q3 = np.percentile(body_len_arr, [25, 75])
        iqr = float(q3 - q1)
        cv = float(np.std(body_len_arr) / (np.mean(body_len_arr) + 1e-9))
        outlier_mask = (body_len_arr < (q1 - 1.5 * iqr)) | (body_len_arr > (q3 + 1.5 * iqr))
        body_outlier_frac = float(np.mean(outlier_mask))
    else:
        med = float(np.median(body_len_arr)) if len(body_len_arr) else 0.0
        q1, q3 = 0.0, 0.0
        iqr = 0.0
        cv = 0.0
        body_outlier_frac = 0.0

    # Jitter: per KP median pairwise displacement where both frames valid for that KP
    jitter_medians: list[float] = []
    per_kp_jitter: dict[str, float] = {}
    for name in CANONICAL_KPS:
        disp: list[float] = []
        for i in range(1, N):
            f0 = frames[i - 1].get("canonical_kps") or {}
            f1 = frames[i].get("canonical_kps") or {}
            k0 = f0.get(name) if isinstance(f0, dict) else None
            k1 = f1.get(name) if isinstance(f1, dict) else None
            if not k0 or not k1:
                continue
            l0, _ = _likelihood(k0)
            l1, _ = _likelihood(k1)
            if l0 < tau or l1 < tau:
                continue
            disp.append(math.hypot(k1["x"] - k0["x"], k1["y"] - k0["y"]))
        jm = float(np.median(disp)) if disp else 0.0
        per_kp_jitter[name] = round(jm, 4)
        jitter_medians.append(jm)

    mean_of_median_jitter = float(np.mean(jitter_medians)) if jitter_medians else 0.0
    jitter_norm = mean_of_median_jitter / (med + 1e-9) if med > 0 else 0.0

    # Gaps: invalid = not valid_all7
    invalid = ~valid_all7
    max_gap = 0
    cur = 0
    gaps_ge_5 = 0
    for i in range(N):
        if invalid[i]:
            cur += 1
        else:
            if cur > 0:
                max_gap = max(max_gap, cur)
                if cur >= 5:
                    gaps_ge_5 += 1
            cur = 0
    if cur > 0:
        max_gap = max(max_gap, cur)
        if cur >= 5:
            gaps_ge_5 += 1

    # Decision tier
    decision = _decision_tier(
        valid_frame_fraction_all7=valid_frame_fraction_all7,
        min_mean_likelihood= min_mean_likelihood_across_kps,
        body_cv=cv,
        max_gap_frames=max_gap,
        jitter_px=mean_of_median_jitter,
        jitter_norm=jitter_norm,
        thresholds=th,
    )

    mask = valid_all7.astype(np.uint8)

    report: dict[str, Any] = {
        "schema_version": POSE_QC_SCHEMA_VERSION,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "confidence_threshold": tau,
        "likelihood_assumed": likelihood_assumed_any,
        "per_keypoint": per_keypoint,
        "session": {
            "n_frames": N,
            "valid_frame_fraction_all7": round(valid_frame_fraction_all7, 5),
            "body_length_px": {
                "median": round(med, 4),
                "iqr": round(iqr, 4),
                "cv": round(cv, 5),
            },
            "body_length_outlier_fraction": round(body_outlier_frac, 5),
            "jitter_px": {
                "mean_of_median_per_kp": round(mean_of_median_jitter, 4),
                "per_keypoint": per_kp_jitter,
                "jitter_norm_vs_body_length": round(jitter_norm, 5),
            },
            "max_gap_frames": int(max_gap),
            "gaps_ge_5_frames": int(gaps_ge_5),
        },
        "decision": decision,
        "frame_mask": {"format": "uint8_numpy", "length": N},
    }
    return report, mask


def _decision_tier(
    *,
    valid_frame_fraction_all7: float,
    min_mean_likelihood: float,
    body_cv: float,
    max_gap_frames: int,
    jitter_px: float,
    jitter_norm: float,
    thresholds: PoseQCThresholds,
) -> dict[str, Any]:
    """
    Per metric: A = acceptable, B = borderline, C = poor (see pose_qc_config).
    Tier: C if any metric is C OR >=3 metrics are B; B if >=2 metrics are B; else A.
    """
    th = thresholds
    reasons: list[str] = []

    grades: list[str] = []

    # valid_fraction: higher better
    g = _metric_abc(
        valid_frame_fraction_all7,
        a_min=th.valid_frac_acceptable,
        b_low=th.valid_frac_borderline_low,
    )
    if valid_frame_fraction_all7 < th.valid_frac_borderline_low:
        g = "C"
    elif valid_frame_fraction_all7 < th.valid_frac_acceptable:
        g = "B"
    else:
        g = "A"
    grades.append(g)
    if g != "A":
        reasons.append(f"valid_frame_fraction_all7={valid_frame_fraction_all7:.3f} → {g}")

    # min likelihood across KPs
    if min_mean_likelihood < th.min_kp_likelihood_borderline_low:
        g = "C"
    elif min_mean_likelihood < th.min_kp_likelihood_acceptable:
        g = "B"
    else:
        g = "A"
    grades.append(g)

    # body CV: higher worse
    if body_cv > th.body_cv_borderline_high:
        g = "C"
    elif body_cv > th.body_cv_acceptable:
        g = "B"
    else:
        g = "A"
    grades.append(g)

    # max gap
    if max_gap_frames > th.max_gap_borderline_high:
        g = "C"
    elif max_gap_frames > th.max_gap_acceptable:
        g = "B"
    else:
        g = "A"
    grades.append(g)

    # jitter: bad if either px or norm exceeds C threshold
    jitter_bad_c = (
        jitter_px > th.jitter_px_borderline_high
        or jitter_norm > th.jitter_norm_borderline_high
    )
    jitter_bad_b = (
        jitter_px > th.jitter_px_acceptable
        or jitter_norm > th.jitter_norm_acceptable
    )
    if jitter_bad_c:
        g = "C"
    elif jitter_bad_b:
        g = "B"
    else:
        g = "A"
    grades.append(g)

    b_count = sum(1 for x in grades if x == "B")
    c_count = sum(1 for x in grades if x == "C")

    if c_count >= 1 or b_count >= 3:
        tier, label = "C", "poor"
        rec = "Pose quality is poor: consider retraining or relabeling, or use the labeling frame export."
    elif b_count >= 2:
        tier, label = "B", "borderline"
        rec = "Pose quality is borderline: fine-tuning or additional labeled frames may improve results."
    else:
        tier, label = "A", "acceptable"
        rec = "Pose quality is acceptable for downstream analysis."

    return {
        "tier": tier,
        "label": label,
        "recommendation": rec,
        "reasons": reasons[:12],
        "metric_grades": {
            "valid_frame_fraction": grades[0],
            "min_keypoint_likelihood": grades[1],
            "body_length_cv": grades[2],
            "max_gap_frames": grades[3],
            "jitter": grades[4],
        },
        "b_metric_count": b_count,
        "c_metric_count": c_count,
    }


def write_pose_qc_artifacts(
    job_dir: Path,
    frames: list[dict[str, Any]],
    *,
    confidence_threshold: float | None = None,
    thresholds: PoseQCThresholds | None = None,
) -> dict[str, Any]:
    """Compute QC, write pose_qc.json and pose_valid_mask.npy under job_dir."""
    report, mask = compute_pose_qc_from_frames(
        frames, confidence_threshold=confidence_threshold, thresholds=thresholds
    )
    qc_path = job_dir / "pose_qc.json"
    mask_path = job_dir / "pose_valid_mask.npy"
    if "computed_at" not in report:
        report["computed_at"] = datetime.now(timezone.utc).isoformat()
    report["frame_mask"]["path"] = "pose_valid_mask.npy"
    report["frame_mask"]["format"] = "npy_uint8"
    qc_path.write_text(json.dumps(report, indent=2))
    np.save(str(mask_path), mask)
    return report


def load_pose_qc(job_dir: Path) -> dict[str, Any] | None:
    p = job_dir / "pose_qc.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_pose_valid_mask(job_dir: Path) -> np.ndarray | None:
    p = job_dir / "pose_valid_mask.npy"
    if not p.exists():
        return None
    return np.load(p, allow_pickle=False)


def read_pose_qc_from_result_path(result_path: Path) -> dict[str, Any] | None:
    """Load pose_qc.json if present next to result.json."""
    job_dir = result_path.parent
    return load_pose_qc(job_dir)
