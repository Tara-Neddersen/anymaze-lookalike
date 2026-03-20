"""
embedding.py — UMAP embeddings for per-frame behavioral space and per-animal phenotypes.

Two embedding levels:
  1. Per-frame: UMAP on pooled cohort pose features → behavioral landscape figure
  2. Per-animal: UMAP on phenotype vectors → animal-level separation

The per-frame embedding is fit on ALL animals together (WT + BPAN pooled) so that
the resulting space captures the full behavioral landscape. Genotype coloring is
applied in the frontend.

UMAP parameters:
  Per-frame:  n_neighbors=30, min_dist=0.1, metric='euclidean'
  Per-animal: n_neighbors=5,  min_dist=0.3, metric='euclidean'
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

# UMAP import with graceful degradation
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ---------------------------------------------------------------------------
# Per-frame embedding
# ---------------------------------------------------------------------------

def compute_perframe_embedding(
    pose_matrices: list[np.ndarray],
    valid_masks: list[np.ndarray],
    animal_meta: list[dict[str, Any]],
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    random_state: int = 42,
    max_frames: int = 50_000,
) -> dict[str, Any]:
    """
    Compute a 2D UMAP embedding of per-frame pose features across all animals.

    Args:
        pose_matrices: list of (N_i, 20) arrays, one per animal
        valid_masks:   list of (N_i,) bool arrays
        animal_meta:   list of dicts with keys: job_id, genotype, animal_id
        n_neighbors:   UMAP parameter
        min_dist:      UMAP parameter
        random_state:  reproducibility seed
        max_frames:    maximum total frames to embed (subsampled if exceeded)

    Returns:
        dict with:
          coords:       list of [x, y] float pairs (N_valid_total, 2)
          frame_meta:   list of dicts with job_id, genotype, frame_idx, t_sec
          n_valid:      total valid frames embedded
          params:       dict of UMAP params used
    """
    if not UMAP_AVAILABLE:
        raise RuntimeError("umap-learn is not installed. Run: pip install umap-learn")

    # Pool valid frames
    all_X:     list[np.ndarray] = []
    all_meta:  list[dict] = []

    for ai, (mat, vmask, meta) in enumerate(zip(pose_matrices, valid_masks, animal_meta)):
        valid_idx = np.where(vmask)[0]
        if len(valid_idx) == 0:
            continue
        all_X.append(mat[valid_idx])
        for fi in valid_idx:
            all_meta.append({
                "job_id":    meta.get("job_id", f"animal_{ai}"),
                "genotype":  meta.get("genotype", "unknown"),
                "animal_id": meta.get("animal_id", f"animal_{ai}"),
                "frame_idx": int(fi),
            })

    if not all_X:
        return {"coords": [], "frame_meta": [], "n_valid": 0, "params": {}}

    X_pooled = np.concatenate(all_X, axis=0)

    # Impute any remaining NaN with column median
    col_meds = np.nanmedian(X_pooled, axis=0)
    for ci in range(X_pooled.shape[1]):
        mask = np.isnan(X_pooled[:, ci])
        X_pooled[mask, ci] = col_meds[ci]

    # Subsample if too large
    if len(X_pooled) > max_frames:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_pooled), size=max_frames, replace=False)
        idx.sort()
        X_pooled = X_pooled[idx]
        all_meta = [all_meta[i] for i in idx]

    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pooled)

    # UMAP
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = reducer.fit_transform(X_scaled)

    coords = embedding.tolist()
    params = {
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": "euclidean",
        "random_state": random_state,
        "n_frames": len(X_pooled),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    return {
        "coords":     coords,
        "frame_meta": all_meta,
        "n_valid":    len(X_pooled),
        "params":     params,
    }


# ---------------------------------------------------------------------------
# Per-animal embedding
# ---------------------------------------------------------------------------

def compute_animal_embedding(
    phenotype_vectors: list[dict[str, float]],
    animal_meta: list[dict[str, Any]],
    n_neighbors: int = 5,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compute a 2D UMAP embedding of per-animal phenotype vectors.

    Args:
        phenotype_vectors: list of dicts {feature_name: float} — one per animal
        animal_meta:       list of dicts with job_id, genotype, animal_id
        n_neighbors:       UMAP parameter
        min_dist:          UMAP parameter
        random_state:      reproducibility seed

    Returns:
        dict with coords (list of [x,y]), animal_meta, params
    """
    if not UMAP_AVAILABLE:
        raise RuntimeError("umap-learn is not installed.")

    if len(phenotype_vectors) < 2:
        return {"coords": [[0, 0]] * len(phenotype_vectors),
                "animal_meta": animal_meta, "params": {}}

    # Extract feature names from first vector
    feat_names = list(phenotype_vectors[0].keys())
    X = np.array([[v.get(f, 0.0) for f in feat_names] for v in phenotype_vectors],
                 dtype=np.float32)

    # Replace NaN with 0
    X = np.nan_to_num(X, nan=0.0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_nb = min(n_neighbors, len(X) - 1)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_nb,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
        verbose=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedding = reducer.fit_transform(X_scaled)

    return {
        "coords":      embedding.tolist(),
        "animal_meta": animal_meta,
        "feat_names":  feat_names,
        "params": {
            "n_neighbors": n_nb,
            "min_dist": min_dist,
            "n_animals": len(X),
        },
    }


# ---------------------------------------------------------------------------
# Density estimation (for contour overlay in frontend)
# ---------------------------------------------------------------------------

def kde_grid(
    coords: list[list[float]],
    grid_size: int = 50,
    bandwidth: float = 0.5,
) -> dict[str, Any]:
    """
    Compute a KDE density grid for UMAP coordinates.
    Returns grid metadata suitable for canvas rendering.
    """
    if not coords:
        return {"grid": [], "x_range": [0, 1], "y_range": [0, 1], "grid_size": grid_size}

    pts = np.array(coords, dtype=np.float64)
    x_min, y_min = pts.min(axis=0) - 1.0
    x_max, y_max = pts.max(axis=0) + 1.0

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

    # Simple Gaussian KDE
    h = bandwidth
    density = np.zeros(len(grid_pts))
    for pt in pts:
        d2 = np.sum((grid_pts - pt) ** 2, axis=1)
        density += np.exp(-0.5 * d2 / h ** 2)
    density /= (density.max() + 1e-8)

    return {
        "grid":    density.reshape(grid_size, grid_size).tolist(),
        "x_range": [float(x_min), float(x_max)],
        "y_range": [float(y_min), float(y_max)],
        "grid_size": grid_size,
    }
