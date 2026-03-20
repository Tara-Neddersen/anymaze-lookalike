"""
motif_discovery.py — Unsupervised behavioral motif discovery via PCA + k-means.

Algorithm:
  1. PCA to 15 components (retains >95% variance, speeds up k-means)
  2. k-means sweep k = 2..20, record silhouette coefficient and inertia
  3. Select k by elbow + silhouette peak (default: k=8)
  4. Run final k-means with 20 random restarts
  5. Bootstrap stability validation (10 subsamples × 80%)
  6. Auto-label motifs by dominant kinematic signature
  7. Serialise to motif_library.json
"""
from __future__ import annotations

import json
import math
import warnings
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .window_features import WINDOW_FEATURE_DIM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K_MIN      = 2
K_MAX      = 20
K_DEFAULT  = 8
N_INIT     = 20     # k-means restarts for final model
N_BOOT     = 10     # bootstrap iterations
BOOT_FRAC  = 0.80   # fraction of windows per bootstrap
PCA_COMPS  = 15
STABILITY_WARN = 0.75   # warn if mean Jaccard < this


# ---------------------------------------------------------------------------
# Motif auto-labeling
# ---------------------------------------------------------------------------
# Feature-name indices in the 40-dim window feature vector:
# dims 0-11 = position means, 12=speed_mean, 13=angvel_mean, 14=spine_curv_mean,
# 15=sin_hba_mean, 16=cos_hba_mean, 17=ear_norm_mean, 18=groom_mean, 19=rear_mean
# dims 20-39 = same but std (offset by 20)
_IDX_SPEED_MEAN    = 12
_IDX_AVEL_MEAN     = 13
_IDX_SPINE_MEAN    = 14
_IDX_GROOM_MEAN    = 18
_IDX_REAR_MEAN     = 19
_IDX_SPEED_STD     = 32
_IDX_AVEL_STD      = 33


def _auto_label(centroid: np.ndarray) -> str:
    """Heuristic auto-label from a 40-dim window centroid vector."""
    speed  = float(centroid[_IDX_SPEED_MEAN])  if not math.isnan(centroid[_IDX_SPEED_MEAN])  else 0.0
    avel   = float(centroid[_IDX_AVEL_MEAN])   if not math.isnan(centroid[_IDX_AVEL_MEAN])   else 0.0
    spine  = float(centroid[_IDX_SPINE_MEAN])  if not math.isnan(centroid[_IDX_SPINE_MEAN])  else 0.0
    groom  = float(centroid[_IDX_GROOM_MEAN])  if not math.isnan(centroid[_IDX_GROOM_MEAN])  else 0.0
    rear   = float(centroid[_IDX_REAR_MEAN])   if not math.isnan(centroid[_IDX_REAR_MEAN])   else 0.0

    if rear > 0.3:
        return "rearing"
    if groom > 0.3:
        return "grooming"
    if speed < 0.1 and avel < 0.1:
        return "stillness"
    if spine > 0.25 and speed < 0.3:
        return "hunching"
    if avel > 0.6 and speed > 0.2:
        return "turning"
    if speed > 0.7:
        return "running"
    if speed > 0.3:
        return "walking"
    return "slow_movement"


# ---------------------------------------------------------------------------
# Elbow / silhouette heuristic k selection
# ---------------------------------------------------------------------------

def _select_k(
    inertias: list[float],
    silhouettes: list[float],
    k_values: list[int],
) -> int:
    """
    Select k by balancing silhouette peak and elbow.
    Strategy: take the k with the highest silhouette unless a clear elbow
    exists at a lower k (defined as inertia reduction < 10% of total range).
    """
    if not silhouettes:
        return K_DEFAULT

    sil_arr = np.array(silhouettes)
    best_sil_idx = int(np.argmax(sil_arr))

    # Elbow: look for the first k where marginal inertia gain drops below 10% of range
    iner_arr = np.array(inertias)
    iner_range = iner_arr[0] - iner_arr[-1]
    if iner_range > 0:
        for i in range(1, len(iner_arr) - 1):
            drop = iner_arr[i - 1] - iner_arr[i]
            if drop / iner_range < 0.05 and k_values[i] <= k_values[best_sil_idx]:
                return k_values[i]

    return k_values[best_sil_idx]


# ---------------------------------------------------------------------------
# Jaccard similarity for bootstrap stability
# ---------------------------------------------------------------------------

def _jaccard_similarity(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Compute Jaccard index between two label assignments on the same items.
    Uses the Hungarian matching between clusters to account for label permutation.
    """
    from scipy.optimize import linear_sum_assignment

    # Build co-occurrence / contingency matrix
    k_a = int(labels_a.max()) + 1
    k_b = int(labels_b.max()) + 1
    k = max(k_a, k_b)
    contingency = np.zeros((k, k), dtype=np.float64)
    for a, b in zip(labels_a, labels_b):
        if a >= 0 and b >= 0:
            contingency[a, b] += 1

    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    total   = len(labels_a)
    return float(matched / total) if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Main discovery function
# ---------------------------------------------------------------------------

def discover_motifs(
    window_features: np.ndarray,
    window_valid: np.ndarray,
    k_override: int | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Discover behavioral motifs from pooled window features.

    Args:
        window_features: (N_windows, 40) float32 array
        window_valid:    (N_windows,) bool — which windows to use
        k_override:      if set, skip sweep and use this k
        random_state:    reproducibility seed

    Returns:
        motif_library dict with keys:
          k, centroids (list of lists), pca_components, pca_mean, scaler_mean,
          scaler_scale, auto_labels, stability_score, silhouette, inertia,
          k_sweep_results, valid_fraction
    """
    valid_idx = np.where(window_valid)[0]
    if len(valid_idx) < K_MIN * 5:
        raise ValueError(
            f"Not enough valid windows ({len(valid_idx)}) to discover motifs. "
            "Need at least pose data from multiple animals."
        )

    X = window_features[valid_idx]

    # Replace any remaining NaNs with column median
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X_clean = X.copy()
    X_clean[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # PCA
    n_components = min(PCA_COMPS, X_scaled.shape[1], X_scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)

    # k-means sweep (skip if k_override)
    k_values   = list(range(K_MIN, min(K_MAX, len(valid_idx) // 5) + 1))
    inertias   = []
    silhouettes = []

    if k_override is not None:
        chosen_k = k_override
    else:
        for k in k_values:
            km = KMeans(n_clusters=k, n_init=5, random_state=random_state)
            labels = km.fit_predict(X_pca)
            inertias.append(float(km.inertia_))
            sil = silhouette_score(X_pca, labels, sample_size=min(5000, len(labels)))
            silhouettes.append(float(sil))
        chosen_k = _select_k(inertias, silhouettes, k_values)

    # Final k-means with full restarts
    km_final = KMeans(n_clusters=chosen_k, n_init=N_INIT, random_state=random_state)
    final_labels = km_final.fit_predict(X_pca)
    final_sil    = float(silhouette_score(X_pca, final_labels,
                                          sample_size=min(5000, len(final_labels))))

    # Bootstrap stability
    stability_scores = []
    rng = np.random.default_rng(random_state)
    for _ in range(N_BOOT):
        boot_idx = rng.choice(len(X_pca), size=int(len(X_pca) * BOOT_FRAC), replace=False)
        held_idx = np.setdiff1d(np.arange(len(X_pca)), boot_idx)
        if len(held_idx) < 2:
            continue
        km_boot = KMeans(n_clusters=chosen_k, n_init=5, random_state=random_state)
        km_boot.fit(X_pca[boot_idx])
        boot_labels_held = km_boot.predict(X_pca[held_idx])
        ref_labels_held  = final_labels[held_idx]
        j = _jaccard_similarity(ref_labels_held, boot_labels_held)
        stability_scores.append(j)

    stability_mean = float(np.mean(stability_scores)) if stability_scores else 0.0
    if stability_mean < STABILITY_WARN:
        warnings.warn(
            f"Motif clustering stability ({stability_mean:.2f}) is below threshold "
            f"({STABILITY_WARN}). Consider reducing k.",
            stacklevel=2,
        )

    # Centroids in ORIGINAL (pre-PCA, post-scaling) space for interpretability
    centroids_pca = km_final.cluster_centers_  # (k, n_pca_comps)
    centroids_scaled = pca.inverse_transform(centroids_pca)  # (k, 40)
    centroids_raw    = scaler.inverse_transform(centroids_scaled)  # (k, 40)

    auto_labels = [_auto_label(c) for c in centroids_raw]

    # Full per-window label assignment (for all valid windows)
    all_labels_valid = km_final.predict(X_pca)
    # Map back: valid_idx[i] → all_labels_valid[i]
    all_labels = np.full(len(window_features), -1, dtype=np.int8)
    all_labels[valid_idx] = all_labels_valid.astype(np.int8)

    k_sweep = {
        "k_values": k_values,
        "inertias": inertias,
        "silhouettes": silhouettes,
    }

    library: dict[str, Any] = {
        "k": chosen_k,
        "centroids": centroids_raw.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "n_pca_components": n_components,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "auto_labels": auto_labels,
        "stability_score": stability_mean,
        "silhouette": final_sil,
        "inertia": float(km_final.inertia_),
        "k_sweep_results": k_sweep,
        "valid_fraction": float(len(valid_idx) / len(window_features)),
    }

    return library, all_labels


def assign_labels_from_library(
    window_features: np.ndarray,
    window_valid: np.ndarray,
    library: dict[str, Any],
) -> np.ndarray:
    """
    Assign motif labels to new window features using an existing library.
    Returns label array of shape (N_windows,), -1 for invalid windows.
    """
    scaler_mean  = np.array(library["scaler_mean"],  dtype=np.float32)
    scaler_scale = np.array(library["scaler_scale"], dtype=np.float32)
    pca_comps    = np.array(library["pca_components"], dtype=np.float32)
    pca_mean     = np.array(library["pca_mean"],      dtype=np.float32)
    centroids    = np.array(library["centroids"],     dtype=np.float32)
    k            = library["k"]

    labels = np.full(len(window_features), -1, dtype=np.int8)

    valid_idx = np.where(window_valid)[0]
    if len(valid_idx) == 0:
        return labels

    X = window_features[valid_idx]
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    X_clean = X.copy()
    X_clean[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    # Apply saved scaler
    X_scaled = (X_clean - scaler_mean) / np.maximum(scaler_scale, 1e-8)

    # Apply saved PCA
    n_comps = library["n_pca_components"]
    X_pca   = (X_scaled - pca_mean) @ pca_comps[:n_comps].T

    # Centroid assignment via nearest centroid in PCA space
    # Project centroids to PCA space
    centroids_raw_arr = np.array(library["centroids"], dtype=np.float32)
    centroids_scaled  = (centroids_raw_arr - scaler_mean) / np.maximum(scaler_scale, 1e-8)
    centroids_pca     = (centroids_scaled - pca_mean) @ pca_comps[:n_comps].T

    # Nearest centroid
    dists    = np.linalg.norm(X_pca[:, np.newaxis, :] - centroids_pca[np.newaxis, :, :], axis=2)
    assigned = np.argmin(dists, axis=1).astype(np.int8)
    labels[valid_idx] = assigned

    return labels
