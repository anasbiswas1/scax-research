"""Model training (libraries are used for training only; plan Algorithm 1).

XGBoost is trained with base_score = 0.5 so that the exported base margin is exactly zero.
The decision threshold tau is chosen on out-of-fold scores of the training data.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .predictor import native_scores

DEFAULTS = {
    "xgboost": {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.3, "tree_method": "hist", "max_bin": 256,
                "min_child_weight": 1.0, "gamma": 0.0},
    "lightgbm": {"n_estimators": 100, "num_leaves": 31, "max_depth": 6, "learning_rate": 0.1, "max_bin": 255},
    "sklearn_rf": {"n_estimators": 50, "max_depth": 6, "min_samples_leaf": 1},
}


def make_model(family: str, params: dict | None = None, seed: int = 0, n_jobs: int = -1):
    p = {**DEFAULTS[family], **(params or {})}
    if family == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(objective="binary:logistic", base_score=0.5, random_state=seed, n_jobs=n_jobs, **p)
    if family == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=n_jobs, verbose=-1, **p)
    if family == "sklearn_rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=seed, n_jobs=n_jobs, **p)
    raise ValueError(family)


def fit(family, X, y, params=None, seed=0):
    return make_model(family, params, seed).fit(np.asarray(X, dtype=np.float64), np.asarray(y))


def oof_scores(family, X, y, params=None, seed=0, folds=5) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y)
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = fit(family, X[tr], y[tr], params, seed)
        out[te] = native_scores(m, family, X[te])
    return out


def youden_threshold(scores, y) -> float:
    """Threshold maximising sensitivity + specificity - 1, placed midway between adjacent scores."""
    s = np.asarray(scores, dtype=np.float64); y = np.asarray(y)
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    P, N = tp[-1], fp[-1]
    distinct = np.r_[s[1:] != s[:-1], True]
    j = (tp / P - fp / N)[distinct]
    idx = np.flatnonzero(distinct)
    best = idx[int(np.argmax(j))]
    if best + 1 >= len(s):
        return float(s[best] - 1.0)
    return float((s[best] + s[best + 1]) / 2.0)
