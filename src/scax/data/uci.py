"""UCI Heart Disease, four sites, prepared for the site shift (plan Sections 6.5 and 6.6).

The predictor set is fixed before any model is fitted: the eight items recorded at every site. Left out:
serum cholesterol (stored as 0, that is missing, for Switzerland), fasting blood sugar (missing for most
Switzerland records), and slope, number of vessels and thalassemia (missing for most Hungary,
Switzerland and VA records). A resting blood pressure or cholesterol of 0 is a missing value. Records
missing any of the eight predictors are dropped (complete cases), and the counts are reported per site.
Outcome: angiographic disease, num > 0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SITES = ["cleveland", "hungarian", "switzerland", "va"]
SITE_NAMES = {"cleveland": "Cleveland", "hungarian": "Hungary", "switzerland": "Switzerland", "va": "VA Long Beach"}
PREDICTORS = ["age", "sex", "cp", "trestbps", "restecg", "thalach", "exang", "oldpeak"]
LEFT_OUT = {
    "chol": "stored as 0 (missing) for Switzerland",
    "fbs": "missing for most Switzerland records",
    "slope": "missing for most Hungary and VA records",
    "ca": "missing for most Hungary, Switzerland and VA records",
    "thal": "missing for most Hungary, Switzerland and VA records",
}
ZERO_IS_MISSING = ["trestbps", "chol"]
TARGET = "angiographic_disease"


def _with_zero_missing(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    for c in ZERO_IS_MISSING:
        if c in df:
            df.loc[df[c] == 0, c] = np.nan
    return df


def missingness(X: pd.DataFrame) -> pd.DataFrame:
    """Share of records missing each column, per site (a zero blood pressure or cholesterol counts as missing)."""
    df = _with_zero_missing(X)
    cols = [c for c in df.columns if c != "site"]
    return df.groupby("site", sort=False)[cols].apply(lambda g: g.isna().mean()).reindex(SITES)


def prepare(X: pd.DataFrame, y) -> tuple:
    """Complete cases on the eight predictors; returns the table and per-site counts."""
    df = _with_zero_missing(X)
    y = np.asarray(y).astype(int)
    keep = df[PREDICTORS].notna().all(axis=1).to_numpy()
    out = df.loc[keep, ["site"] + PREDICTORS].astype({c: float for c in PREDICTORS}).reset_index(drop=True)
    out[TARGET] = y[keep]
    counts = []
    for s in SITES:
        m = (df.site == s).to_numpy()
        counts.append({"site": s, "site_name": SITE_NAMES[s], "records": int(m.sum()), "complete_cases": int((m & keep).sum()),
                       "disease_share": float(y[m & keep].mean()) if (m & keep).any() else float("nan")})
    return out, pd.DataFrame(counts)
