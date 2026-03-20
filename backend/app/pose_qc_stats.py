"""
Group-level comparisons for pose QC metrics (e.g. WT vs BPAN tracking quality).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scipy import stats

from .cohort_store import get_cohort
from .pose import CANONICAL_KPS


def cohort_pose_qc_summary(cohort_id: str, data_dir: str) -> dict[str, Any]:
    """
    Aggregate pose_qc.json per animal in cohort; test whether QC metrics differ by genotype.

    Uses Mann-Whitney U on per-animal valid_frame_fraction_all7 when >=2 animals per group.
    """
    c = get_cohort(cohort_id)
    if c is None:
        return {"error": "cohort_not_found"}

    rows: list[dict[str, Any]] = []
    base = Path(data_dir) / "jobs"

    for animal in c.animals:
        qc_path = base / animal.job_id / "pose_qc.json"
        if not qc_path.exists():
            rows.append({
                "job_id": animal.job_id,
                "animal_id": animal.animal_id,
                "genotype": animal.genotype,
                "qc_available": False,
            })
            continue
        qc = json.loads(qc_path.read_text())
        dec = qc.get("decision", {})
        sess = qc.get("session", {})
        rows.append({
            "job_id": animal.job_id,
            "animal_id": animal.animal_id,
            "genotype": animal.genotype.upper(),
            "qc_available": True,
            "tier": dec.get("tier"),
            "label": dec.get("label"),
            "valid_frame_fraction_all7": sess.get("valid_frame_fraction_all7"),
            "min_keypoint_likelihood": min(
                (qc.get("per_keypoint", {}).get(k, {}).get("mean_likelihood_when_present", 0.0)
                 for k in CANONICAL_KPS),
                default=0.0,
            ),
            "body_length_cv": sess.get("body_length_px", {}).get("cv"),
            "max_gap_frames": sess.get("max_gap_frames"),
        })

    # Group-wise means for display
    genotypes: dict[str, list[float]] = {}
    for r in rows:
        if not r.get("qc_available") or r.get("valid_frame_fraction_all7") is None:
            continue
        g = r["genotype"] or "unknown"
        genotypes.setdefault(g, []).append(float(r["valid_frame_fraction_all7"]))

    group_means = {g: (sum(v) / len(v) if v else None) for g, v in genotypes.items()}

    # Mann-Whitney: first two genotypes with >=2 samples each (sorted)
    quality_differs = False
    p_value: float | None = None
    test_note = "insufficient_n_per_genotype"

    eligible = [(g, v) for g, v in genotypes.items() if len(v) >= 2]
    if len(eligible) >= 2:
        eligible.sort(key=lambda x: x[0])
        g1, a1 = eligible[0]
        g2, a2 = eligible[1]
        try:
            if len(a1) >= 2 and len(a2) >= 2:
                ures = stats.mannwhitneyu(a1, a2, alternative="two-sided")
                p_value = float(ures.pvalue)
                quality_differs = p_value < 0.05
                test_note = f"mannwhitney_{g1}_vs_{g2}"
            else:
                test_note = "need_n>=2_per_group"
        except Exception:
            test_note = "test_failed"

    return {
        "cohort_id": cohort_id,
        "per_animal": rows,
        "group_mean_valid_fraction": group_means,
        "tracking_quality_differs_by_genotype": quality_differs,
        "valid_fraction_mannwhitney_p": p_value,
        "test_note": test_note,
    }
