"""Metrics of the underlying model (plan Section 6.7)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


def ece_equal_mass(p, y, bins: int = 15) -> float:
    """Expected calibration error with equal-mass bins."""
    p = np.asarray(p, dtype=float); y = np.asarray(y, dtype=float)
    order = np.argsort(p, kind="stable")
    out = 0.0
    for part in np.array_split(order, bins):
        if len(part):
            out += len(part) / len(p) * abs(y[part].mean() - p[part].mean())
    return float(out)


def calibration_slope_intercept(p, y):
    """Logistic recalibration of y on logit(p): slope 1 and intercept 0 mean perfect calibration."""
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, np.asarray(y))
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def model_metrics(p, y, yhat) -> dict:
    p = np.asarray(p, dtype=float); y = np.asarray(y); yhat = np.asarray(yhat)
    slope, intercept = calibration_slope_intercept(p, y)
    return {"n": int(len(y)), "prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)), "brier": float(np.mean((p - y) ** 2)),
            "ece_15_equal_mass": ece_equal_mass(p, y), "calibration_slope": slope, "calibration_intercept": intercept,
            "sensitivity": float(yhat[y == 1].mean()), "specificity": float(1 - yhat[y == 0].mean()),
            "predicted_positive_share": float(yhat.mean())}
