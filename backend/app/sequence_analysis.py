"""
sequence_analysis.py — Behavioral sequence analysis from motif label time series.

Computes per-animal sequence statistics:
  - Transition matrix (k×k row-normalised)
  - Stationary distribution
  - Transition entropy (H = -Σ T[i,j] log T[i,j])
  - Self-transition rate (mean diagonal of T)
  - Motif dwell times (mean and CV per motif)
  - Motif usage fractions
  - Top trigrams (3-motif sequences with frequency)
  - Markov order estimation (order 1 vs 2)
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def compute_sequence_profile(
    frame_labels: np.ndarray,
    fps: float,
    k: int,
) -> dict[str, Any]:
    """
    Compute the full sequence profile for one animal's motif label time series.

    Args:
        frame_labels: (N_frames,) int8 array, -1 = unassigned
        fps:          frames per second
        k:            number of motifs

    Returns:
        SequenceProfile dict (JSON-serialisable)
    """
    # Filter out unassigned (-1) frames
    valid = frame_labels[frame_labels >= 0].astype(int)

    if len(valid) < 4:
        return _empty_profile(k)

    # --- Transition matrix ---
    T = _transition_matrix(valid, k)

    # --- Stationary distribution ---
    stat_dist = _stationary_distribution(T)

    # --- Entropy ---
    entropy = _transition_entropy(T)

    # --- Self-transition rate ---
    self_rate = float(np.mean(np.diag(T)))

    # --- Dwell times ---
    dwell_mean, dwell_cv = _dwell_times(valid, k, fps)

    # --- Usage fractions ---
    usage = _usage_fractions(valid, k)

    # --- Trigrams ---
    trigrams = _top_trigrams(valid, n=5)

    # --- Markov order ---
    markov_order = _estimate_markov_order(valid, k)

    return {
        "transition_matrix":   T.tolist(),
        "stationary_distribution": stat_dist.tolist(),
        "transition_entropy":  round(entropy, 6),
        "max_entropy":         round(math.log(k * k) if k > 1 else 0.0, 6),
        "entropy_normalized":  round(entropy / math.log(k * k) if k > 1 else 0.0, 6),
        "self_transition_rate": round(self_rate, 6),
        "motif_dwell_mean_s":  [round(v, 4) for v in dwell_mean],
        "motif_dwell_cv":      [round(v, 4) for v in dwell_cv],
        "motif_usage_pct":     [round(v, 6) for v in usage],
        "top_trigrams":        trigrams,
        "markov_order":        markov_order,
        "n_valid_frames":      int(len(valid)),
        "n_transitions":       int(len(valid) - 1),
    }


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def _transition_matrix(labels: np.ndarray, k: int) -> np.ndarray:
    T = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(labels[:-1], labels[1:]):
        T[a, b] += 1.0
    # Row-normalise (avoid division by zero)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return T / row_sums


# ---------------------------------------------------------------------------
# Stationary distribution (dominant eigenvector of T^T)
# ---------------------------------------------------------------------------

def _stationary_distribution(T: np.ndarray) -> np.ndarray:
    try:
        eigenvalues, eigenvectors = np.linalg.eig(T.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stat = np.real(eigenvectors[:, idx])
        stat = np.abs(stat)
        s = stat.sum()
        if s > 0:
            return stat / s
    except Exception:
        pass
    # Fallback: uniform
    k = len(T)
    return np.full(k, 1.0 / k)


# ---------------------------------------------------------------------------
# Transition entropy H = -Σ_i p_i Σ_j T[i,j] log T[i,j]
# where p_i is the stationary distribution
# ---------------------------------------------------------------------------

def _transition_entropy(T: np.ndarray) -> float:
    stat = _stationary_distribution(T)
    H = 0.0
    for i in range(len(T)):
        for j in range(len(T)):
            t = T[i, j]
            if t > 1e-12:
                H -= stat[i] * t * math.log(t)
    return H


# ---------------------------------------------------------------------------
# Dwell times per motif
# ---------------------------------------------------------------------------

def _dwell_times(
    labels: np.ndarray,
    k: int,
    fps: float,
) -> tuple[list[float], list[float]]:
    """Compute mean and CV of dwell durations (in seconds) for each motif."""
    dwells: list[list[float]] = [[] for _ in range(k)]

    if len(labels) == 0:
        return [0.0] * k, [0.0] * k

    current = int(labels[0])
    run_len = 1
    for label in labels[1:]:
        if int(label) == current:
            run_len += 1
        else:
            dwells[current].append(run_len / fps)
            current = int(label)
            run_len = 1
    dwells[current].append(run_len / fps)

    means = []
    cvs   = []
    for d in dwells:
        if d:
            m = float(np.mean(d))
            s = float(np.std(d))
            means.append(m)
            cvs.append(s / m if m > 0 else 0.0)
        else:
            means.append(0.0)
            cvs.append(0.0)

    return means, cvs


# ---------------------------------------------------------------------------
# Usage fractions
# ---------------------------------------------------------------------------

def _usage_fractions(labels: np.ndarray, k: int) -> list[float]:
    counts = np.bincount(labels, minlength=k).astype(float)
    total  = counts.sum()
    if total == 0:
        return [0.0] * k
    return (counts / total).tolist()


# ---------------------------------------------------------------------------
# Top trigrams
# ---------------------------------------------------------------------------

def _top_trigrams(labels: np.ndarray, n: int = 5) -> list[tuple[int, int, int, float]]:
    if len(labels) < 3:
        return []
    trigrams = list(zip(labels[:-2], labels[1:-1], labels[2:]))
    counter = Counter(trigrams)
    total = sum(counter.values())
    top = counter.most_common(n)
    return [(int(a), int(b), int(c), round(cnt / total, 6)) for (a, b, c), cnt in top]


# ---------------------------------------------------------------------------
# Markov order estimation (order 1 vs 2)
# ---------------------------------------------------------------------------

def _estimate_markov_order(labels: np.ndarray, k: int, min_n: int = 100) -> int:
    """
    Test whether 2nd-order Markov model significantly improves log-likelihood.
    Returns 1 or 2.
    """
    if len(labels) < min_n or k < 2:
        return 1

    # 1st-order log-likelihood
    T1 = _transition_matrix(labels, k)
    ll1 = 0.0
    for a, b in zip(labels[:-1], labels[1:]):
        p = T1[a, b]
        if p > 1e-12:
            ll1 += math.log(p)

    # 2nd-order transition counts: T2[a, b] → count of transitions a→b→c
    T2: dict[tuple[int, int], Counter] = {}
    for a, b, c in zip(labels[:-2], labels[1:-1], labels[2:]):
        key = (int(a), int(b))
        if key not in T2:
            T2[key] = Counter()
        T2[key][int(c)] += 1

    # Normalise and compute LL
    ll2 = 0.0
    for a, b, c in zip(labels[:-2], labels[1:-1], labels[2:]):
        key = (int(a), int(b))
        cnts = T2.get(key)
        if cnts:
            total = sum(cnts.values())
            p = cnts.get(int(c), 0) / max(total, 1)
            if p > 1e-12:
                ll2 += math.log(p)

    # Likelihood ratio test (chi-squared approximation)
    # df = k * k * (k - 1)
    lr = 2 * (ll2 - ll1)
    df = k * k * (k - 1)
    # Conservative: only prefer 2nd order if improvement > 3.84 per df
    if df > 0 and lr / df > 3.84:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Empty profile
# ---------------------------------------------------------------------------

def _empty_profile(k: int) -> dict[str, Any]:
    return {
        "transition_matrix":     [[0.0] * k for _ in range(k)],
        "stationary_distribution": [1.0 / k] * k,
        "transition_entropy":    0.0,
        "max_entropy":           round(math.log(k * k) if k > 1 else 0.0, 6),
        "entropy_normalized":    0.0,
        "self_transition_rate":  0.0,
        "motif_dwell_mean_s":    [0.0] * k,
        "motif_dwell_cv":        [0.0] * k,
        "motif_usage_pct":       [0.0] * k,
        "top_trigrams":          [],
        "markov_order":          1,
        "n_valid_frames":        0,
        "n_transitions":         0,
    }


# ---------------------------------------------------------------------------
# Cohort-level sequence comparison
# ---------------------------------------------------------------------------

def compare_group_entropy(
    profiles_a: list[dict],
    profiles_b: list[dict],
    group_a_name: str = "WT",
    group_b_name: str = "BPAN",
) -> dict[str, Any]:
    """
    Compare transition entropy between two groups.
    Returns descriptive statistics and a simple non-parametric test.
    """
    ent_a = [p["transition_entropy"] for p in profiles_a]
    ent_b = [p["transition_entropy"] for p in profiles_b]

    from scipy.stats import mannwhitneyu
    if len(ent_a) >= 2 and len(ent_b) >= 2:
        stat, pval = mannwhitneyu(ent_a, ent_b, alternative="two-sided")
    else:
        stat, pval = 0.0, 1.0

    return {
        f"{group_a_name}_mean_entropy": float(np.mean(ent_a)) if ent_a else None,
        f"{group_b_name}_mean_entropy": float(np.mean(ent_b)) if ent_b else None,
        f"{group_a_name}_sem_entropy":  float(np.std(ent_a) / math.sqrt(len(ent_a))) if ent_a else None,
        f"{group_b_name}_sem_entropy":  float(np.std(ent_b) / math.sqrt(len(ent_b))) if ent_b else None,
        "mann_whitney_u":  float(stat),
        "p_value":         float(pval),
        "n_a": len(ent_a),
        "n_b": len(ent_b),
    }
