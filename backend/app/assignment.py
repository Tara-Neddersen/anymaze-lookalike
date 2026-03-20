"""
Hungarian (Munkres) algorithm for consistent multi-animal ID assignment
across video frames.  Prevents identity swaps when blobs cross or disappear.
"""
from __future__ import annotations

import math
from typing import Any


def hungarian_assign(
    prev_positions: list[tuple[float, float] | None],
    curr_centroids: list[tuple[float, float]],
    max_dist: float = 200.0,
) -> list[int | None]:
    """
    Assign each tracked animal (indexed 0..n_prev-1) to one of the detected
    centroids in curr_centroids using minimum total displacement.

    Returns a list the same length as prev_positions where each entry is
    the index into curr_centroids that was assigned (or None if no centroid
    was close enough).
    """
    n_prev = len(prev_positions)
    n_curr = len(curr_centroids)

    if n_curr == 0:
        return [None] * n_prev
    if n_prev == 0:
        return []

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        cost = np.full((n_prev, n_curr), max_dist * 2.0, dtype=float)
        for i, pc in enumerate(prev_positions):
            if pc is None:
                continue
            for j, cc in enumerate(curr_centroids):
                d = math.hypot(pc[0] - cc[0], pc[1] - cc[1])
                if d < max_dist:
                    cost[i, j] = d

        row_ind, col_ind = linear_sum_assignment(cost)
        result: list[int | None] = [None] * n_prev
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < max_dist:
                result[r] = int(c)
        return result

    except ImportError:
        return _greedy_assign(prev_positions, curr_centroids, max_dist)


def _greedy_assign(
    prev_positions: list[tuple[float, float] | None],
    curr_centroids: list[tuple[float, float]],
    max_dist: float,
) -> list[int | None]:
    """Nearest-neighbour greedy fallback (used when scipy is unavailable)."""
    used: set[int] = set()
    result: list[int | None] = []
    for pc in prev_positions:
        if pc is None:
            result.append(None)
            continue
        best_j: int | None = None
        best_d = max_dist
        for j, cc in enumerate(curr_centroids):
            if j in used:
                continue
            d = math.hypot(pc[0] - cc[0], pc[1] - cc[1])
            if d < best_d:
                best_d = d
                best_j = j
        result.append(best_j)
        if best_j is not None:
            used.add(best_j)
    return result
