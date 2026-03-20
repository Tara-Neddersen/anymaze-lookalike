"""
Automatic arena detection from a video first frame.

Two strategies:
  1. largest_rect   – finds the largest rectangular region (bright or dark arena)
  2. largest_circle – Hough circles for circular arenas (open field, EPM center)

Returns a list of [x, y] polygon points in image coordinates, or None if nothing
confident is found.
"""
from __future__ import annotations

import cv2
import numpy as np


def detect_arena_rect(frame: np.ndarray, margin: float = 0.03) -> list[list[float]] | None:
    """
    Detect the largest rectangular/polygon region that could be the arena.
    Works by finding the biggest contour after Canny edge detection.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold — handles both bright arenas on dark floor and vice versa
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 51, 5)
    edges = cv2.Canny(blurred, 30, 100)

    # Combine edges with threshold boundaries
    combined = cv2.bitwise_or(thresh, edges)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = frame.shape[:2]
    frame_area = w * h
    min_area = frame_area * 0.10  # must cover ≥ 10% of frame

    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) < min_area:
        return None

    # Approximate polygon with ~4-8 vertices
    epsilon = 0.02 * cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, epsilon, True)
    if len(approx) < 3:
        return None

    pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
    return pts


def detect_arena_circle(frame: np.ndarray) -> list[list[float]] | None:
    """
    Detect the largest circle using Hough Transform.
    Returns a polygon approximation of the circle.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    h, w = frame.shape[:2]
    min_r = int(min(w, h) * 0.20)
    max_r = int(min(w, h) * 0.52)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min(w, h) * 0.4,
        param1=50,
        param2=35,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return None

    cx, cy, r = circles[0][0]
    n = 36
    pts = [
        [float(cx + r * np.cos(2 * np.pi * i / n)),
         float(cy + r * np.sin(2 * np.pi * i / n))]
        for i in range(n)
    ]
    return pts


def detect_arena_from_bytes(jpeg_bytes: bytes) -> dict:
    """
    Run both detection strategies and return the best result.
    Returns {"polygon": [[x,y],...], "method": "rect"|"circle", "confidence": 0-1}
    or {"polygon": None, "method": None, "confidence": 0}
    """
    arr = np.frombuffer(jpeg_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"polygon": None, "method": None, "confidence": 0}

    h, w = frame.shape[:2]

    # Try circle first (more specific)
    circ = detect_arena_circle(frame)
    rect = detect_arena_rect(frame)

    def score_poly(poly: list[list[float]] | None) -> float:
        if poly is None or len(poly) < 3:
            return 0.0
        # Reward polygons that cover 20-90% of the frame
        pts = np.array(poly, dtype=np.int32)
        area = cv2.contourArea(pts)
        frac = area / (w * h)
        if frac < 0.08 or frac > 0.95:
            return 0.0
        return min(1.0, frac / 0.5)

    circ_score = score_poly(circ) * 1.1  # slight boost for circles (they're more distinctive)
    rect_score = score_poly(rect)

    if circ_score == 0 and rect_score == 0:
        return {"polygon": None, "method": None, "confidence": 0}

    if circ_score >= rect_score:
        return {"polygon": circ, "method": "circle", "confidence": round(circ_score, 2)}
    return {"polygon": rect, "method": "rect", "confidence": round(rect_score, 2)}
