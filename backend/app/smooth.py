from __future__ import annotations

import math
from typing import Any


def kalman_smooth(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    2D constant-velocity Kalman filter over centroid track.

    State vector: [x, y, vx, vy]
    Observation: [x, y] from raw centroid.

    Frames with no centroid are skipped during the update step (prediction only).
    Frames where the innovation magnitude exceeds 3 sigma are flagged quality='low'
    and the update is still applied (but with reduced weight via R inflation).

    Short gaps (missing centroids ≤3 consecutive frames) are filled by linear
    interpolation between the last and next valid smoothed positions.

    Returns a new list of frame dicts; does not mutate input.
    """
    n = len(frames)
    if n == 0:
        return list(frames)

    # Infer dt from fps (use median inter-frame time of valid frames)
    valid_times = [f["t_sec"] for f in frames if f.get("ok") and f.get("centroid")]
    if len(valid_times) >= 2:
        dts = [valid_times[i + 1] - valid_times[i] for i in range(len(valid_times) - 1)]
        dt = sorted(dts)[len(dts) // 2]
    else:
        dt = 1 / 30.0
    if dt <= 0:
        dt = 1 / 30.0

    # State transition matrix (constant velocity)
    # x_k = F * x_{k-1}
    F = [
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ]

    # Observation matrix: H * x = [x, y]
    H = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ]

    # Process noise covariance Q (tuned for ~30 px/s typical acceleration noise)
    q = (15.0 * dt) ** 2
    Q = [
        [q,  0,  0,  0],
        [0,  q,  0,  0],
        [0,  0,  q,  0],
        [0,  0,  0,  q],
    ]

    # Measurement noise covariance R (std ~5 px)
    R_base = 25.0  # 5^2
    R_inflate = 400.0  # Used when innovation is large (outlier likely)

    # Initial state from first valid centroid
    x0, y0 = 0.0, 0.0
    for f in frames:
        if f.get("ok") and f.get("centroid"):
            c = f["centroid"]
            x0, y0 = float(c["x"]), float(c["y"])
            break

    # State: 4×1 column vector as list
    state = [x0, y0, 0.0, 0.0]
    # Covariance: 4×4 as list of lists
    P = [
        [1000, 0, 0, 0],
        [0, 1000, 0, 0],
        [0, 0, 500, 0],
        [0, 0, 0, 500],
    ]

    # Small matrix helpers (4×4 only, no numpy dependency)
    def mv(A: list[list[float]], v: list[float]) -> list[float]:
        return [sum(A[i][j] * v[j] for j in range(len(v))) for i in range(len(A))]

    def mm(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        rA, rB, cB = len(A), len(B), len(B[0])
        return [[sum(A[i][k] * B[k][j] for k in range(rB)) for j in range(cB)] for i in range(rA)]

    def add_m(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def sub_m(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def transpose(A: list[list[float]]) -> list[list[float]]:
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    def inv2(A: list[list[float]]) -> list[list[float]]:
        """2×2 matrix inverse."""
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-12:
            det = 1e-12
        return [
            [A[1][1] / det, -A[0][1] / det],
            [-A[1][0] / det, A[0][0] / det],
        ]

    Ht = transpose(H)
    Ft = transpose(F)

    smoothed_xy: list[tuple[float, float] | None] = []
    quality_flags: list[str] = []

    for f in frames:
        # --- Predict ---
        state = mv(F, state)
        FP = mm(F, P)
        P = add_m(mm(FP, Ft), Q)

        if not f.get("ok") or not f.get("centroid"):
            smoothed_xy.append(None)
            quality_flags.append("no_detection")
            continue

        c = f["centroid"]
        z = [float(c["x"]), float(c["y"])]

        # --- Innovation ---
        z_pred = mv(H, state)
        innov = [z[0] - z_pred[0], z[1] - z_pred[1]]
        innov_mag = math.hypot(innov[0], innov[1])

        # Innovation covariance S = H*P*H' + R
        HP = mm(H, P)
        HPHt = mm(HP, Ht)
        R_val = R_inflate if innov_mag > 60 else R_base
        S = [[HPHt[i][j] + (R_val if i == j else 0) for j in range(2)] for i in range(2)]
        S_inv = inv2(S)

        # Mahalanobis distance for quality flag
        Si_innov = mv(S_inv, innov)
        mah2 = innov[0] * Si_innov[0] + innov[1] * Si_innov[1]
        quality = "low" if mah2 > 9.0 else "ok"  # 9 = 3^2 in chi2(2) approx

        # --- Update ---
        K = mm(mm(P, Ht), S_inv)  # 4×2
        correction = [K[i][0] * innov[0] + K[i][1] * innov[1] for i in range(4)]
        state = [state[i] + correction[i] for i in range(4)]

        KH = mm(K, H)
        I_KH = sub_m([[1 if i == j else 0 for j in range(4)] for i in range(4)], KH)
        P = mm(I_KH, P)

        smoothed_xy.append((state[0], state[1]))
        quality_flags.append(quality)

    # --- Build output frames with smoothed positions ---
    out = []
    for i, f in enumerate(frames):
        nf = dict(f)
        sxy = smoothed_xy[i]
        if sxy is not None:
            nf = dict(f)
            nf["centroid"] = {"x": sxy[0], "y": sxy[1]}
            nf["quality"] = quality_flags[i]
        else:
            nf["quality"] = quality_flags[i]
        out.append(nf)

    # --- Linear interpolation for short gaps (≤3 consecutive missing) ---
    GAP_FILL_MAX = 3
    i = 0
    while i < len(out):
        if out[i].get("quality") == "no_detection":
            # Find gap extent
            j = i
            while j < len(out) and out[j].get("quality") == "no_detection":
                j += 1
            gap_len = j - i
            if gap_len <= GAP_FILL_MAX:
                # Find prev valid
                prev_idx = i - 1
                while prev_idx >= 0 and out[prev_idx].get("quality") == "no_detection":
                    prev_idx -= 1
                next_idx = j
                while next_idx < len(out) and out[next_idx].get("quality") == "no_detection":
                    next_idx += 1

                if prev_idx >= 0 and next_idx < len(out):
                    p0 = out[prev_idx]["centroid"]
                    p1 = out[next_idx]["centroid"]
                    for k in range(i, j):
                        alpha = (k - prev_idx) / (next_idx - prev_idx)
                        out[k]["centroid"] = {
                            "x": p0["x"] + alpha * (p1["x"] - p0["x"]),
                            "y": p0["y"] + alpha * (p1["y"] - p0["y"]),
                        }
                        out[k]["ok"] = True
                        out[k]["quality"] = "interpolated"
            i = j
        else:
            i += 1

    return out
