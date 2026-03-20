"""
Statistical testing for group comparison.
Supports independent-samples t-test (2 groups) and one-way ANOVA with
Tukey HSD post-hoc comparisons (3+ groups).
Requires scipy.
"""
from __future__ import annotations

import math
from typing import Any


def run_group_stats(
    groups: dict[str, list[float]],
) -> dict[str, Any]:
    """
    Run the appropriate test based on number of groups.
    Returns serializable dict with test name, p-value, and pairwise results.
    """
    groups = {k: [x for x in v if x is not None and not math.isnan(x)]
              for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if len(groups) < 2:
        return {"test": None, "p_value": None, "significant": False, "pairs": []}

    group_names = list(groups.keys())
    group_values = [groups[n] for n in group_names]

    try:
        from scipy import stats as scipy_stats

        if len(groups) == 2:
            t_stat, p = scipy_stats.ttest_ind(group_values[0], group_values[1],
                                               equal_var=False)  # Welch's t-test
            pairs = [{
                "group1": group_names[0],
                "group2": group_names[1],
                "p_value": round(float(p), 4),
                "significant": float(p) < 0.05,
                "stars": _stars(float(p)),
                "mean1": round(float(sum(group_values[0]) / len(group_values[0])), 3),
                "mean2": round(float(sum(group_values[1]) / len(group_values[1])), 3),
                "n1": len(group_values[0]),
                "n2": len(group_values[1]),
            }]
            return {
                "test": "Welch t-test",
                "p_value": round(float(p), 4),
                "significant": float(p) < 0.05,
                "stars": _stars(float(p)),
                "pairs": pairs,
            }

        else:
            # One-way ANOVA
            f_stat, p_anova = scipy_stats.f_oneway(*group_values)
            pairs = _tukey_hsd(group_names, group_values)
            return {
                "test": "One-way ANOVA",
                "p_value": round(float(p_anova), 4),
                "significant": float(p_anova) < 0.05,
                "stars": _stars(float(p_anova)),
                "pairs": pairs,
            }

    except ImportError:
        # scipy not available — return means and sample sizes only
        pairs = []
        for i in range(len(group_names)):
            for j in range(i + 1, len(group_names)):
                pairs.append({
                    "group1": group_names[i],
                    "group2": group_names[j],
                    "p_value": None, "significant": None, "stars": "n/a",
                    "mean1": round(float(sum(group_values[i]) / len(group_values[i])), 3),
                    "mean2": round(float(sum(group_values[j]) / len(group_values[j])), 3),
                    "n1": len(group_values[i]),
                    "n2": len(group_values[j]),
                })
        return {"test": "none (scipy missing)", "p_value": None,
                "significant": False, "stars": "n/a", "pairs": pairs}


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _tukey_hsd(
    names: list[str],
    groups: list[list[float]],
) -> list[dict[str, Any]]:
    """Honest Significant Difference (Tukey HSD) pairwise comparisons."""
    try:
        from scipy.stats import tukey_hsd as _tukey
        import numpy as np
        result = _tukey(*groups)
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p = float(result.pvalue[i, j])
                pairs.append({
                    "group1": names[i],
                    "group2": names[j],
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "stars": _stars(p),
                    "mean1": round(float(sum(groups[i]) / len(groups[i])), 3),
                    "mean2": round(float(sum(groups[j]) / len(groups[j])), 3),
                    "n1": len(groups[i]),
                    "n2": len(groups[j]),
                })
        return pairs
    except (ImportError, AttributeError):
        # Fallback: Bonferroni-corrected pairwise t-tests
        from scipy import stats as scipy_stats
        n_tests = len(names) * (len(names) - 1) // 2
        pairs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                _, p_raw = scipy_stats.ttest_ind(groups[i], groups[j], equal_var=False)
                p = min(1.0, float(p_raw) * n_tests)  # Bonferroni
                pairs.append({
                    "group1": names[i],
                    "group2": names[j],
                    "p_value": round(p, 4),
                    "significant": p < 0.05,
                    "stars": _stars(p),
                    "mean1": round(float(sum(groups[i]) / len(groups[i])), 3),
                    "mean2": round(float(sum(groups[j]) / len(groups[j])), 3),
                    "n1": len(groups[i]),
                    "n2": len(groups[j]),
                })
        return pairs
