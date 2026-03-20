from __future__ import annotations

import math
from typing import Any


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Poly = list[tuple[float, float]]  # list of (x, y) vertices


# ---------------------------------------------------------------------------
# Polygon geometry
# ---------------------------------------------------------------------------

def point_in_polygon(x: float, y: float, poly: Poly) -> bool:
    """
    Ray-casting test: returns True if (x, y) is inside the polygon.
    Works for non-convex polygons.
    """
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def polygon_perimeter(poly: Poly) -> float:
    """Total perimeter length of the polygon."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def distance_to_polygon_border(x: float, y: float, poly: Poly) -> float:
    """Minimum distance from point (x, y) to any edge of the polygon."""
    min_dist = float("inf")
    n = len(poly)
    for i in range(n):
        ax, ay = poly[i]
        bx, by = poly[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq < 1e-12:
            d = math.hypot(x - ax, y - ay)
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / seg_len_sq))
            px, py = ax + t * dx, ay + t * dy
            d = math.hypot(x - px, y - py)
        if d < min_dist:
            min_dist = d
    return min_dist


# ---------------------------------------------------------------------------
# Zone assignment
# ---------------------------------------------------------------------------

def assign_zones(
    frames: list[dict[str, Any]],
    zones: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    For each frame with a valid centroid, determine which named zone it belongs to.
    The first matching zone (in order) wins.  If none match, zone_id = None.

    Each zone in `zones` has:
        { "id": str, "name": str, "color": str, "points": [[x, y], ...] }

    Mutates frames in-place and returns them.
    """
    zone_polys: list[tuple[str, Poly]] = []
    for z in zones:
        pts = z.get("points", [])
        if len(pts) >= 3:
            poly: Poly = [(float(p[0]), float(p[1])) for p in pts]
            zone_polys.append((z["id"], poly))

    for f in frames:
        f["zone_id"] = None
        if not f.get("ok") or not f.get("centroid"):
            continue
        cx = float(f["centroid"]["x"])
        cy = float(f["centroid"]["y"])
        for zid, poly in zone_polys:
            if point_in_polygon(cx, cy, poly):
                f["zone_id"] = zid
                break

    return frames


# ---------------------------------------------------------------------------
# Thigmotaxis
# ---------------------------------------------------------------------------

def compute_thigmotaxis(
    frames: list[dict[str, Any]],
    arena_poly: Poly,
    margin_px: float = 50.0,
) -> float:
    """
    Fraction of tracked time the animal spends within `margin_px` of the arena
    wall (thigmotaxis = wall-hugging, a proxy for anxiety).

    Returns a value in [0, 1].
    """
    near_wall = 0
    total = 0
    for f in frames:
        if not f.get("ok") or not f.get("centroid"):
            continue
        total += 1
        cx = float(f["centroid"]["x"])
        cy = float(f["centroid"]["y"])
        if distance_to_polygon_border(cx, cy, arena_poly) <= margin_px:
            near_wall += 1
    return near_wall / total if total > 0 else 0.0
