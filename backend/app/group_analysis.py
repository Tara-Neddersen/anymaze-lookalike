"""
group_analysis.py — Enhanced group comparison statistics for behavioral phenotyping.

Extends stats.py with:
  - Benjamini–Hochberg FDR correction for multi-metric comparisons
  - Cohen's d effect size
  - Permutation test (non-parametric, preferred for small n < 10)
  - Mixed-effects model wrapper (statsmodels)
  - MANOVA wrapper (statsmodels)
  - Per-feature group comparison across full phenotype vectors

Backwards-compatible: re-exports run_group_stats() from stats.py.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import stats as scipy_stats

from .stats import run_group_stats, _stars  # re-export for backwards compatibility

__all__ = [
    "run_group_stats",
    "cohen_d",
    "permutation_test",
    "benjamini_hochberg_fdr",
    "compare_phenotype_groups",
    "mixed_effects_summary",
    "multivariate_comparison",
]

# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohen_d(group_a: list[float], group_b: list[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.
    Uses pooled standard deviation (equal-variance assumption is common for d).
    Returns 0.0 if insufficient data.
    """
    a = np.array([x for x in group_a if not math.isnan(x)], dtype=float)
    b = np.array([x for x in group_b if not math.isnan(x)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return 0.0
    n_a, n_b = len(a), len(b)
    var_a, var_b = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std < 1e-10:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_perm: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Two-sided permutation test of the difference in means.

    Returns:
        observed_diff: float
        p_value:       float
    """
    a = np.array([x for x in group_a if not math.isnan(x)], dtype=float)
    b = np.array([x for x in group_b if not math.isnan(x)], dtype=float)

    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0

    observed = float(np.mean(a) - np.mean(b))
    pooled   = np.concatenate([a, b])
    n_a      = len(a)

    rng = np.random.default_rng(random_state)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = np.mean(pooled[:n_a]) - np.mean(pooled[n_a:])
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)  # +1 for continuity correction
    return observed, float(p_value)


# ---------------------------------------------------------------------------
# FDR correction (Benjamini–Hochberg)
# ---------------------------------------------------------------------------

def benjamini_hochberg_fdr(p_values: list[float], alpha: float = 0.05) -> list[float]:
    """
    Benjamini–Hochberg FDR correction.

    Args:
        p_values: list of raw p-values
        alpha:    FDR threshold

    Returns:
        adjusted p-values (same length as input)
    """
    n = len(p_values)
    if n == 0:
        return []

    p_arr = np.array(p_values, dtype=float)
    order = np.argsort(p_arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)

    adjusted = p_arr * n / ranks
    # Enforce monotonicity (step-down)
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])

    return np.minimum(adjusted, 1.0).tolist()


# ---------------------------------------------------------------------------
# Per-feature phenotype group comparison
# ---------------------------------------------------------------------------

def compare_phenotype_groups(
    vectors_a: list[dict[str, float]],
    vectors_b: list[dict[str, float]],
    group_a_name: str = "WT",
    group_b_name: str = "BPAN",
    use_permutation_threshold: int = 10,
) -> dict[str, Any]:
    """
    Compare two groups across all features in their phenotype vectors.
    Automatically uses permutation test when n < use_permutation_threshold.
    Applies BH FDR correction to all p-values.

    Returns:
        {
          "feature_results": list of per-feature comparison dicts,
          "summary": overall comparison dict
        }
    """
    if not vectors_a or not vectors_b:
        return {"feature_results": [], "summary": {}}

    feat_names = list(vectors_a[0].keys())
    n_a, n_b   = len(vectors_a), len(vectors_b)
    use_perm   = (n_a < use_permutation_threshold or n_b < use_permutation_threshold)

    raw_results: list[dict[str, Any]] = []
    p_values_raw: list[float] = []

    for feat in feat_names:
        a_vals = [v.get(feat, 0.0) for v in vectors_a]
        b_vals = [v.get(feat, 0.0) for v in vectors_b]

        a_clean = [x for x in a_vals if not math.isnan(x)]
        b_clean = [x for x in b_vals if not math.isnan(x)]

        a_mean = float(np.mean(a_clean)) if a_clean else 0.0
        b_mean = float(np.mean(b_clean)) if b_clean else 0.0
        a_sem  = float(np.std(a_clean) / math.sqrt(len(a_clean))) if len(a_clean) > 1 else 0.0
        b_sem  = float(np.std(b_clean) / math.sqrt(len(b_clean))) if len(b_clean) > 1 else 0.0
        d      = cohen_d(a_clean, b_clean)

        if use_perm:
            _, p = permutation_test(a_clean, b_clean)
            test_name = "permutation"
        else:
            if len(a_clean) >= 2 and len(b_clean) >= 2:
                _, p = scipy_stats.ttest_ind(a_clean, b_clean, equal_var=False)
                p = float(p)
                test_name = "Welch t-test"
            else:
                p = 1.0
                test_name = "insufficient_n"

        p_values_raw.append(p)
        raw_results.append({
            "feature":          feat,
            f"{group_a_name}_mean": round(a_mean, 6),
            f"{group_b_name}_mean": round(b_mean, 6),
            f"{group_a_name}_sem":  round(a_sem, 6),
            f"{group_b_name}_sem":  round(b_sem, 6),
            "cohen_d":          round(d, 4),
            "p_raw":            round(p, 6),
            "test":             test_name,
            "n_a": len(a_clean),
            "n_b": len(b_clean),
        })

    # FDR correction
    p_adjusted = benjamini_hochberg_fdr(p_values_raw)
    for i, rec in enumerate(raw_results):
        rec["p_fdr"] = round(p_adjusted[i], 6)
        rec["significant_fdr"] = p_adjusted[i] < 0.05
        rec["stars"] = _stars(p_adjusted[i])

    # Sort by p_fdr ascending
    raw_results.sort(key=lambda r: r["p_fdr"])

    # Summary
    n_sig = sum(1 for r in raw_results if r["significant_fdr"])
    top_3 = [r["feature"] for r in raw_results[:3]]

    summary = {
        "group_a": group_a_name,
        "group_b": group_b_name,
        "n_a": n_a,
        "n_b": n_b,
        "n_features_tested": len(feat_names),
        "n_significant_fdr": n_sig,
        "top_discriminating_features": top_3,
        "test_method": "permutation" if use_perm else "Welch t-test",
        "fdr_correction": "Benjamini-Hochberg",
    }

    return {"feature_results": raw_results, "summary": summary}


# ---------------------------------------------------------------------------
# Mixed-effects model (longitudinal / repeated measures)
# ---------------------------------------------------------------------------

def mixed_effects_summary(
    data: list[dict[str, Any]],
    outcome_col: str,
    group_col: str,
    subject_col: str,
) -> dict[str, Any]:
    """
    Fit a linear mixed-effects model:
        outcome ~ group (fixed) + (1 | subject) (random intercept)

    Requires statsmodels. Returns parameter estimates and p-values.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        df = pd.DataFrame(data)
        df = df[[outcome_col, group_col, subject_col]].dropna()
        if len(df) < 4:
            return {"error": "Insufficient data for mixed-effects model"}

        formula = f"{outcome_col} ~ C({group_col})"
        model   = smf.mixedlm(formula, df, groups=df[subject_col])
        result  = model.fit(reml=False, disp=False)

        params = []
        for name, coef in result.params.items():
            pval = result.pvalues.get(name, 1.0)
            params.append({
                "parameter": name,
                "estimate":  round(float(coef), 6),
                "p_value":   round(float(pval), 6),
                "stars":     _stars(float(pval)),
            })

        return {
            "model":  "LinearMixedLM",
            "formula": formula,
            "params": params,
            "log_likelihood": round(float(result.llf), 4),
            "n_obs": int(result.nobs),
        }

    except ImportError:
        return {"error": "statsmodels is required for mixed-effects models"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# MANOVA (multivariate)
# ---------------------------------------------------------------------------

def multivariate_comparison(
    vectors_a: list[dict[str, float]],
    vectors_b: list[dict[str, float]],
) -> dict[str, Any]:
    """
    MANOVA test for multivariate group separation.
    Uses Pillai's trace via statsmodels.
    """
    try:
        import pandas as pd
        from statsmodels.multivariate.manova import MANOVA

        if not vectors_a or not vectors_b:
            return {"error": "Empty groups"}

        feat_names = list(vectors_a[0].keys())
        rows = []
        for v in vectors_a:
            rows.append({f: v.get(f, 0.0) for f in feat_names} | {"group": 0})
        for v in vectors_b:
            rows.append({f: v.get(f, 0.0) for f in feat_names} | {"group": 1})

        df = pd.DataFrame(rows).fillna(0.0)

        # Drop constant or near-constant columns
        df_feats = df[feat_names]
        stds = df_feats.std()
        valid_feats = stds[stds > 1e-6].index.tolist()
        if len(valid_feats) < 2:
            return {"error": "Insufficient non-constant features for MANOVA"}

        formula = " + ".join(valid_feats) + " ~ group"
        maov = MANOVA.from_formula(formula, data=df)
        res  = maov.mv_test()
        table = res.results["group"]["stat"]

        results = {}
        for test_name in ["Pillai's trace", "Wilks' lambda", "Hotelling-Lawley trace", "Roy's greatest root"]:
            row = table.loc[test_name] if test_name in table.index else None
            if row is not None:
                results[test_name] = {
                    "statistic": round(float(row.get("Value", 0)), 6),
                    "F":         round(float(row.get("F Value", 0)), 4),
                    "df_hyp":    float(row.get("Num DF", 0)),
                    "df_err":    float(row.get("Den DF", 0)),
                    "p_value":   round(float(row.get("Pr > F", 1.0)), 6),
                    "stars":     _stars(float(row.get("Pr > F", 1.0))),
                }

        return {
            "test": "MANOVA",
            "n_a": len(vectors_a),
            "n_b": len(vectors_b),
            "n_features": len(valid_feats),
            "results": results,
        }

    except ImportError:
        return {"error": "statsmodels is required for MANOVA"}
    except Exception as e:
        return {"error": str(e)}
