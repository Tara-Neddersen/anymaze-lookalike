"""
classifier.py — Genotype prediction from behavioral phenotype vectors.

Pipeline:
  1. Standardize features (z-score using training fold stats — no leakage)
  2. Leave-one-out cross-validation (appropriate for n=5-20)
  3. Classifiers: logistic regression (interpretable) + random forest (feature importances)
  4. Report: balanced accuracy, ROC-AUC, sensitivity, specificity, confusion matrix
  5. Feature importances → identifies which behavioral dimensions drive genotype prediction
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


# ---------------------------------------------------------------------------
# Main classifier function
# ---------------------------------------------------------------------------

def run_classifier(
    phenotype_vectors: list[dict[str, float]],
    labels: list[str],
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Run LOOCV classification on phenotype vectors.

    Args:
        phenotype_vectors: list of phenotype dicts (one per animal)
        labels:            list of genotype strings (e.g. ["WT", "BPAN", ...])
        random_state:      reproducibility seed

    Returns:
        Comprehensive classification report dict
    """
    if len(phenotype_vectors) < 4:
        return {"error": "Need at least 4 animals for classification"}

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return {"error": "Need at least 2 distinct genotype groups"}

    # Build feature matrix
    feat_names = list(phenotype_vectors[0].keys())
    X = np.array(
        [[pv.get(f, 0.0) for f in feat_names] for pv in phenotype_vectors],
        dtype=np.float32,
    )
    X = np.nan_to_num(X, nan=0.0)

    # Encode labels
    label_to_int = {lab: i for i, lab in enumerate(unique_labels)}
    y = np.array([label_to_int[lab] for lab in labels], dtype=int)

    # LOOCV
    loo = LeaveOneOut()

    results_lr  = _run_loocv(X, y, _make_lr(random_state), loo, len(unique_labels))
    results_rf  = _run_loocv(X, y, _make_rf(random_state), loo, len(unique_labels))

    # Feature importances (fit on full dataset)
    scaler_full = StandardScaler()
    X_scaled    = scaler_full.fit_transform(X)

    lr_full = _make_lr(random_state)
    lr_full.fit(X_scaled, y)
    rf_full = _make_rf(random_state)
    rf_full.fit(X_scaled, y)

    lr_importances = _lr_importances(lr_full, feat_names, unique_labels)
    rf_importances = _rf_importances(rf_full, feat_names)

    return {
        "unique_labels":   unique_labels,
        "label_to_int":    label_to_int,
        "n_animals":       len(labels),
        "n_features":      len(feat_names),
        "feature_names":   feat_names,
        "logistic_regression": {
            **results_lr,
            "feature_importances": lr_importances,
        },
        "random_forest": {
            **results_rf,
            "feature_importances": rf_importances,
        },
    }


# ---------------------------------------------------------------------------
# LOOCV loop
# ---------------------------------------------------------------------------

def _run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    clf,
    loo: LeaveOneOut,
    n_classes: int,
) -> dict[str, Any]:
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_prob_all: list[list[float]] = []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        try:
            clf.fit(X_train_s, y_train)
            pred = int(clf.predict(X_test_s)[0])
            if hasattr(clf, "predict_proba"):
                prob = clf.predict_proba(X_test_s)[0].tolist()
            else:
                prob = [0.0] * n_classes
                prob[pred] = 1.0
        except Exception:
            pred = int(y_train[0]) if len(y_train) > 0 else 0
            prob = [0.0] * n_classes

        y_true_all.append(int(y[test_idx[0]]))
        y_pred_all.append(pred)
        y_prob_all.append(prob)

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)
    y_prob = np.array(y_prob_all)

    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred).tolist()

    # AUC (binary: standard; multiclass: OvR)
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_prob[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
    except Exception:
        auc = float("nan")

    # Sensitivity / specificity (binary only)
    sensitivity, specificity = None, None
    if n_classes == 2 and len(cm) == 2:
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        sensitivity = round(tp / max(tp + fn, 1), 4)
        specificity = round(tn / max(tn + fp, 1), 4)

    return {
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc":           round(auc, 4) if not math.isnan(auc) else None,
        "sensitivity":       sensitivity,
        "specificity":       specificity,
        "confusion_matrix":  cm,
        "n_correct":         int(np.sum(y_true == y_pred)),
        "n_total":           len(y_true),
    }


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def _make_lr(random_state: int) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=random_state,
    )


def _make_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=random_state,
    )


# ---------------------------------------------------------------------------
# Feature importances
# ---------------------------------------------------------------------------

def _lr_importances(
    model: LogisticRegression,
    feat_names: list[str],
    unique_labels: list[str],
) -> list[dict[str, Any]]:
    """Logistic regression coefficients as feature importances."""
    coefs = model.coef_
    if coefs.shape[0] == 1:
        coef_arr = coefs[0]
    else:
        coef_arr = np.abs(coefs).mean(axis=0)

    ranked = sorted(
        zip(feat_names, coef_arr.tolist()),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in ranked[:20]]


def _rf_importances(
    model: RandomForestClassifier,
    feat_names: list[str],
) -> list[dict[str, Any]]:
    """Random forest feature importances (Gini impurity decrease)."""
    imp = model.feature_importances_
    ranked = sorted(
        zip(feat_names, imp.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in ranked[:20]]
