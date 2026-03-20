"""
window_features.py — Sliding-window temporal features over the pose feature matrix.

Converts a continuous (N_frames, 20) pose feature matrix into a
(N_windows, 40) window feature matrix where each window is described
by the mean and std of 20 pose features over a ~0.67 s window.

Window parameters:
  length : 20 frames  (≈0.67 s @ 30 fps)
  step   :  5 frames  (50 ms temporal resolution)

Windows with >25% invalid (NaN) frames are flagged and excluded from clustering.
"""
from __future__ import annotations

from typing import Any

import numpy as np

WINDOW_LEN  = 20    # frames
WINDOW_STEP = 5     # frames
MAX_NAN_FRAC = 0.25 # windows above this threshold are invalid
WINDOW_FEATURE_DIM = 40  # mean(20) + std(20)


def compute_window_features(
    pose_matrix: np.ndarray,
    valid_mask: np.ndarray,
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sliding-window features from a pose feature matrix.

    Args:
        pose_matrix:  (N_frames, 20) float32 array (NaN for invalid frames)
        valid_mask:   (N_frames,) bool array
        fps:          frames per second (used to compute window_start_times_s)

    Returns:
        window_features:     (N_windows, 40) float32 — mean+std per window
        window_start_frames: (N_windows,) int — frame index of window start
        window_valid_mask:   (N_windows,) bool — True if window is usable
    """
    N = len(pose_matrix)
    feat_dim = pose_matrix.shape[1] if pose_matrix.ndim == 2 else 20

    starts = list(range(0, N - WINDOW_LEN + 1, WINDOW_STEP))
    n_windows = len(starts)

    wf = np.full((n_windows, feat_dim * 2), np.nan, dtype=np.float32)
    wf_valid = np.zeros(n_windows, dtype=bool)
    wf_starts = np.array(starts, dtype=np.int32)

    for wi, s in enumerate(starts):
        e = s + WINDOW_LEN
        window = pose_matrix[s:e]          # (WINDOW_LEN, 20)
        w_valid = valid_mask[s:e]

        nan_frac = 1.0 - float(np.mean(w_valid))
        if nan_frac > MAX_NAN_FRAC:
            continue  # leave as NaN, mark invalid

        # Use only valid rows for mean/std
        valid_rows = window[w_valid]
        if len(valid_rows) < 2:
            continue

        mu  = np.nanmean(valid_rows, axis=0)   # (20,)
        std = np.nanstd(valid_rows, axis=0)    # (20,)
        wf[wi] = np.concatenate([mu, std])
        wf_valid[wi] = True

    return wf, wf_starts, wf_valid


def pool_cohort_windows(
    matrices: list[np.ndarray],
    valid_masks: list[np.ndarray],
    fps: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """
    Pool window features from multiple animals into a single design matrix.

    Returns:
        pooled_features:   (N_total_windows, 40)
        pooled_valid:      (N_total_windows,) bool
        animal_indices:    (N_total_windows,) int — which animal each window came from
        n_windows_per_animal: list of window counts per animal
    """
    all_features:    list[np.ndarray] = []
    all_valid:       list[np.ndarray] = []
    animal_indices:  list[int]        = []
    n_per_animal:    list[int]        = []

    for ai, (mat, vmask) in enumerate(zip(matrices, valid_masks)):
        wf, _, wv = compute_window_features(mat, vmask, fps)
        all_features.append(wf)
        all_valid.append(wv)
        animal_indices.extend([ai] * len(wf))
        n_per_animal.append(len(wf))

    if not all_features:
        return (
            np.empty((0, WINDOW_FEATURE_DIM), dtype=np.float32),
            np.empty(0, dtype=bool),
            np.empty(0, dtype=np.int32),
            [],
        )

    pooled_features = np.concatenate(all_features, axis=0)
    pooled_valid    = np.concatenate(all_valid, axis=0)
    animal_idx_arr  = np.array(animal_indices, dtype=np.int32)

    return pooled_features, pooled_valid, animal_idx_arr, n_per_animal


def window_labels_to_frame_labels(
    window_labels: np.ndarray,
    window_starts: np.ndarray,
    n_frames: int,
    n_motifs: int,
    window_valid: np.ndarray,
) -> np.ndarray:
    """
    Map per-window motif labels back to per-frame labels.
    Each frame is assigned the label of the most recent valid window that contains it.
    Unassigned frames get label -1.
    """
    frame_labels = np.full(n_frames, -1, dtype=np.int8)

    for wi, (s, valid) in enumerate(zip(window_starts, window_valid)):
        if not valid:
            continue
        label = int(window_labels[wi])
        e = s + WINDOW_LEN
        frame_labels[s:e] = label  # later windows overwrite earlier ones (fine for step > 1)

    return frame_labels
