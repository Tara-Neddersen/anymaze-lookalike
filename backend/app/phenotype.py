"""
phenotype.py — Per-animal phenotype feature vector construction and Z-scoring.

Combines:
  - Motif usage fractions (k dims)
  - Sequence analysis metrics (entropy, rigidity, complexity, bout stats)
  - Classic scalar metrics from BehaviorMetrics
  - Pose kinematic summaries

Each feature is Z-scored against a WT (wildtype) baseline cohort.
Missing values are imputed with the cohort mean (0 after z-scoring).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_phenotype_vector(
    sequence_profile: dict[str, Any],
    behavior_metrics: dict[str, Any],
    k: int = 8,
) -> dict[str, float]:
    """
    Build a flat phenotype feature dict for one animal.

    Args:
        sequence_profile: output of sequence_analysis.compute_sequence_profile()
        behavior_metrics: BehaviorMetrics.model_dump() or equivalent dict
        k:                number of motifs

    Returns:
        dict of {feature_name: float}
    """
    vec: dict[str, float] = {}

    # --- Motif usage fractions (k features) ---
    usage = sequence_profile.get("motif_usage_pct", [0.0] * k)
    for i, u in enumerate(usage):
        vec[f"motif_{i}_usage_pct"] = float(u) if not _nan(u) else 0.0

    # --- Sequence features ---
    vec["behavioral_entropy"]    = _get(sequence_profile, "transition_entropy", 0.0)
    vec["entropy_normalized"]    = _get(sequence_profile, "entropy_normalized", 0.0)
    vec["behavioral_rigidity"]   = _get(sequence_profile, "self_transition_rate", 0.0)
    vec["behavioral_complexity"] = _shannon_entropy(usage)

    dwell_means = sequence_profile.get("motif_dwell_mean_s", [])
    dwell_cvs   = sequence_profile.get("motif_dwell_cv", [])
    valid_dm = [d for d in dwell_means if d > 0]
    valid_dc = [d for d in dwell_cvs   if not _nan(d)]
    vec["mean_bout_duration_s"] = float(np.mean(valid_dm)) if valid_dm else 0.0
    vec["bout_duration_cv"]     = float(np.mean(valid_dc)) if valid_dc else 0.0

    # --- Classic scalar metrics ---
    vec["distance_cm_norm"]  = _get(behavior_metrics, "total_distance_cm", 0.0)
    vec["freezing_frac"]     = _get(behavior_metrics, "freezing_pct",       0.0) / 100.0
    vec["thigmotaxis_frac"]  = _get(behavior_metrics, "thigmotaxis_pct",    0.0) / 100.0
    vec["rearing_frac"]      = _get(behavior_metrics, "rearing_pct",        0.0) / 100.0
    vec["grooming_frac"]     = _safe_div(
        _get(behavior_metrics, "total_time_grooming_s", 0.0),
        _get(behavior_metrics, "total_time_s", 1.0),
    )
    vec["immobility_frac"]   = _get(behavior_metrics, "immobility_pct",    0.0) / 100.0
    vec["mean_speed_cm_s"]   = _get(behavior_metrics, "mean_speed_cm_s",    0.0)
    vec["path_efficiency"]   = _get(behavior_metrics, "path_efficiency",    0.0)

    # --- Pose kinematics ---
    vec["mean_spine_curvature"]  = _get(behavior_metrics, "mean_spine_curvature",   0.0)
    vec["mean_head_body_angle"]  = _get(behavior_metrics, "mean_head_body_angle_deg", 0.0)
    vec["mean_body_length_cm"]   = _get(behavior_metrics, "mean_body_length_cm",     0.0)

    return vec


def _get(d: dict, key: str, default: float) -> float:
    v = d.get(key, default)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _nan(v: Any) -> bool:
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _shannon_entropy(probs: list[float]) -> float:
    H = 0.0
    for p in probs:
        if p is not None and not _nan(p) and p > 1e-12:
            H -= p * math.log(p)
    return H


# ---------------------------------------------------------------------------
# Z-scoring
# ---------------------------------------------------------------------------

def zscore_phenotypes(
    phenotype_vectors: list[dict[str, float]],
    reference_indices: list[int] | None = None,
) -> tuple[list[dict[str, float]], dict[str, float], dict[str, float]]:
    """
    Z-score phenotype vectors against a reference group (typically WT).

    Args:
        phenotype_vectors: list of phenotype dicts
        reference_indices: indices of the reference group (WT animals).
                           If None, use all animals.

    Returns:
        zscored_vectors:   list of z-scored phenotype dicts
        ref_means:         dict of feature → reference mean
        ref_stds:          dict of feature → reference std
    """
    if not phenotype_vectors:
        return [], {}, {}

    feat_names = list(phenotype_vectors[0].keys())
    ref_idx = reference_indices if reference_indices is not None else list(range(len(phenotype_vectors)))

    ref_means: dict[str, float] = {}
    ref_stds:  dict[str, float] = {}

    for feat in feat_names:
        vals = [phenotype_vectors[i].get(feat, 0.0) for i in ref_idx]
        vals_clean = [v for v in vals if not _nan(v)]
        ref_means[feat] = float(np.mean(vals_clean)) if vals_clean else 0.0
        ref_stds[feat]  = float(np.std(vals_clean))  if vals_clean else 1.0
        if ref_stds[feat] < 1e-8:
            ref_stds[feat] = 1.0  # avoid division by zero

    zscored: list[dict[str, float]] = []
    for pv in phenotype_vectors:
        z: dict[str, float] = {}
        for feat in feat_names:
            val = pv.get(feat, 0.0)
            if _nan(val):
                z[feat] = 0.0
            else:
                z[feat] = (val - ref_means[feat]) / ref_stds[feat]
        zscored.append(z)

    return zscored, ref_means, ref_stds


# ---------------------------------------------------------------------------
# Cohort-level phenotype matrix
# ---------------------------------------------------------------------------

def build_phenotype_matrix(
    phenotype_vectors: list[dict[str, float]],
) -> tuple[np.ndarray, list[str]]:
    """
    Convert list of phenotype dicts to a (N_animals, D) numpy matrix.
    Returns (matrix, feature_names).
    """
    if not phenotype_vectors:
        return np.empty((0, 0), dtype=np.float32), []

    feat_names = list(phenotype_vectors[0].keys())
    X = np.array(
        [[pv.get(f, 0.0) for f in feat_names] for pv in phenotype_vectors],
        dtype=np.float32,
    )
    X = np.nan_to_num(X, nan=0.0)
    return X, feat_names


# ---------------------------------------------------------------------------
# Radar chart data helper
# ---------------------------------------------------------------------------

_RADAR_FEATURES = [
    "entropy_normalized",
    "behavioral_rigidity",
    "freezing_frac",
    "thigmotaxis_frac",
    "rearing_frac",
    "grooming_frac",
    "mean_spine_curvature",
    "mean_speed_cm_s",
    "path_efficiency",
]

_RADAR_LABELS = [
    "Entropy",
    "Rigidity",
    "Freezing",
    "Thigmotaxis",
    "Rearing",
    "Grooming",
    "Spine Curv.",
    "Speed",
    "Path Eff.",
]


def radar_data(
    group_a_vectors: list[dict[str, float]],
    group_b_vectors: list[dict[str, float]],
    group_a_name: str = "WT",
    group_b_name: str = "BPAN",
    p_values: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Compute mean ± SEM for radar chart dimensions.
    """
    def group_stats(vecs: list[dict[str, float]]) -> tuple[list[float], list[float]]:
        means, sems = [], []
        for feat in _RADAR_FEATURES:
            vals = [v.get(feat, 0.0) for v in vecs]
            vals = [x for x in vals if not _nan(x)]
            if vals:
                m = float(np.mean(vals))
                s = float(np.std(vals) / math.sqrt(len(vals)))
            else:
                m, s = 0.0, 0.0
            means.append(m)
            sems.append(s)
        return means, sems

    a_means, a_sems = group_stats(group_a_vectors)
    b_means, b_sems = group_stats(group_b_vectors)

    sig = []
    for feat in _RADAR_FEATURES:
        p = (p_values or {}).get(feat, 1.0)
        if p < 0.001:
            sig.append("***")
        elif p < 0.01:
            sig.append("**")
        elif p < 0.05:
            sig.append("*")
        else:
            sig.append("")

    return {
        "features":    _RADAR_FEATURES,
        "labels":      _RADAR_LABELS,
        f"{group_a_name}_means": a_means,
        f"{group_a_name}_sems":  a_sems,
        f"{group_b_name}_means": b_means,
        f"{group_b_name}_sems":  b_sems,
        "significance": sig,
    }
