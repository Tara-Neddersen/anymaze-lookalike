"""
Stratified frame sampling for DeepLabCut / SLEAP relabeling when pose QC is borderline/poor.

Outputs a CSV manifest (and optionally PNG frames). Avoids near-duplicates via dHash distance.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .cohort_store import get_cohort

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


def _dhash_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def export_labeling_manifest(
    cohort_id: str,
    data_dir: str,
    *,
    n_frames: int = 150,
    random_seed: int = 42,
    min_hamming_dist: int = 10,
    write_png_crops: bool = False,
    video_path_fn: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """
    Sample up to n_frames across genotype × session strata with diversity.

    `video_path_fn(job_id)` must return absolute path to video (from JobStore).
    """
    if cv2 is None:
        raise RuntimeError("OpenCV required for labeling frame export")

    c = get_cohort(cohort_id)
    if c is None:
        raise ValueError("cohort not found")

    rng = random.Random(random_seed)
    cohort_dir = Path(data_dir) / "cohorts" / cohort_id
    cohort_dir.mkdir(parents=True, exist_ok=True)
    img_dir = cohort_dir / "labeling_frames"
    if write_png_crops:
        img_dir.mkdir(parents=True, exist_ok=True)

    # Candidates: (job_id, frame_index, genotype, session_key)
    strata: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for animal in c.animals:
        job_dir = Path(data_dir) / "jobs" / animal.job_id
        rp = job_dir / "result.json"
        if not rp.exists():
            continue
        result = json.loads(rp.read_text())
        meta = result.get("animal_meta") or {}
        session_key = (meta.get("session") or meta.get("experiment_id") or "default")[:64]
        g = (animal.genotype or "unknown").upper()
        frames = result.get("frames", [])
        for i, f in enumerate(frames):
            if not (f.get("canonical_kps") or {}):
                continue
            fi = int(f.get("frame_index", i))
            key = (g, session_key)
            strata.setdefault(key, []).append((animal.job_id, fi))

    if not strata:
        raise ValueError("No jobs with pose data in cohort")

    n_strata = len(strata)
    per_stratum = max(1, n_frames // n_strata)
    remainder = n_frames - per_stratum * n_strata

    picked: list[dict[str, Any]] = []
    hashes: list[np.ndarray] = []

    def try_add(job_id: str, fi: int, g: str, sk: str, vp: str | None) -> bool:
        nonlocal picked, hashes
        if not vp or not Path(vp).exists():
            return False
        cap = cv2.VideoCapture(vp)
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            return False
        dh = _dhash_bgr(frame)
        if any(_hamming(dh, h) < min_hamming_dist for h in hashes):
            return False
        hashes.append(dh)
        row: dict[str, Any] = {
            "job_id": job_id,
            "frame_index": fi,
            "genotype": g,
            "session_key": sk,
            "video_path": vp,
        }
        if write_png_crops:
            fp = img_dir / f"{job_id}_f{fi}.png"
            cv2.imwrite(str(fp), frame)
            row["png_path"] = str(fp.relative_to(cohort_dir))
        picked.append(row)
        return True

    for si, (key, jobs) in enumerate(sorted(strata.items(), key=lambda x: x[0])):
        g, sk = key
        quota = per_stratum + (1 if si < remainder else 0)
        pool = list(jobs)
        rng.shuffle(pool)
        count = 0
        for job_id, fi in pool:
            if count >= quota or len(picked) >= n_frames:
                break
            vp = video_path_fn(job_id) if video_path_fn else None
            if try_add(job_id, fi, g, sk, vp):
                count += 1

    # Top up if needed
    all_pool: list[tuple[str, int, str, str]] = []
    for (g, sk), jobs in strata.items():
        for job_id, fi in jobs:
            all_pool.append((job_id, fi, g, sk))
    rng.shuffle(all_pool)
    for job_id, fi, g, sk in all_pool:
        if len(picked) >= n_frames:
            break
        if any(p["job_id"] == job_id and p["frame_index"] == fi for p in picked):
            continue
        vp = video_path_fn(job_id) if video_path_fn else None
        try_add(job_id, fi, g, sk, vp)

    manifest_path = cohort_dir / "labeling_manifest.csv"
    if picked:
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(picked[0].keys()))
            w.writeheader()
            w.writerows(picked)

    return {
        "cohort_id": cohort_id,
        "n_selected": len(picked),
        "recommended_first_round": "100–200 frames for initial DLC/SLEAP labeling",
        "manifest_path": str(manifest_path.relative_to(Path(data_dir))),
        "rows": picked,
    }
