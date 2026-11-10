"""Prediction-only baselines (plan Section 7.1, B9 to B11).

Each method decides which predictions to release from a confidence score; none gives an explanation.
They are run separately for predicted negatives (claim: no outcome) and predicted positives (claim:
outcome), with the same targets and confidence levels as SCAX's policy statements.

conf: confidence of the released claim (higher = more confident); err: 1 when the recorded outcome
differs from the claim. A threshold theta releases every record with conf >= theta.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import binom

from ..certify.clopper_pearson import cp_upper


def _sorted(conf, err):
    order = np.argsort(-np.asarray(conf, dtype=float), kind="stable")
    c = np.asarray(conf, dtype=float)[order]
    e = np.asarray(err, dtype=np.int64)[order]
    return c, np.cumsum(e)


def _accepted(c_desc, theta):
    """Number of records with conf >= theta (c_desc sorted in decreasing order)."""
    return int(np.searchsorted(-c_desc, -theta, side="right"))


def sgr(conf, err, target, delta):
    """Selection with guaranteed risk (Geifman and El-Yaniv 2017, Algorithm 1).

    Binary search over the sorted confidences for ceil(log2 m) steps; at each step the selective risk
    of the current threshold is bounded by the binomial tail at level delta / k. Returns the most
    permissive visited threshold whose bound is below the target (theta = +inf releases nothing).
    """
    c, cum = _sorted(conf, err)
    m = len(c)
    if m == 0:
        return {"theta": math.inf, "bound": 1.0, "accepted": 0, "steps": 0}
    k = max(1, math.ceil(math.log2(m)))
    asc = c[::-1]                                   # ascending confidences, as in the paper
    z_min, z_max = 1, m
    best = {"theta": math.inf, "bound": 1.0, "accepted": 0, "steps": k}
    for _ in range(k):
        z = math.ceil((z_min + z_max) / 2)
        theta = float(asc[z - 1])
        n = _accepted(c, theta)
        errors = int(cum[n - 1]) if n > 0 else 0
        b = float(cp_upper(errors, n, delta / k)) if n > 0 else 1.0
        if b < target:
            if theta < best["theta"]:
                best = {"theta": theta, "bound": b, "accepted": n, "steps": k}
            z_max = z
        else:
            z_min = z
        if z_max - z_min <= 0:
            break
    return best


def ltt(conf, err, target, delta, grid=None):
    """Learn-then-Test selective threshold with fixed-sequence testing (Angelopoulos et al. 2021).

    Candidate thresholds release the top q share of records by confidence, q on an increasing grid.
    Null hypothesis per threshold: selective risk > target; exact binomial p-value P(Bin(n, target) <= k).
    Thresholds are tested from the most conservative and testing stops at the first non-rejection.
    """
    c, cum = _sorted(conf, err)
    m = len(c)
    grid = np.linspace(0.01, 1.0, 100) if grid is None else np.asarray(grid)
    best = {"theta": math.inf, "accepted": 0, "p_value": 1.0, "share": 0.0}
    if m == 0:
        return best
    for q in grid:
        n_q = max(1, int(math.ceil(q * m)))
        theta = float(c[n_q - 1])
        n = _accepted(c, theta)
        errors = int(cum[n - 1])
        p = float(binom.cdf(errors, n, target))
        if p > delta:
            break
        best = {"theta": theta, "accepted": n, "p_value": p, "share": float(q)}
    return best


def conformal_quantile(scores, eps):
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    k = math.ceil((n + 1) * (1 - eps))
    return math.inf if k > n else float(s[k - 1])


def conformal_singletons(p_cal, y_cal, p_test, eps, mondrian=False):
    """Split conformal prediction with score 1 - p_hat(y); returns the singleton class per test record
    (0 or 1) or -1 when the prediction set is empty or contains both classes.

    mondrian=True uses one quantile per true class (class-conditional coverage).
    """
    p_cal = np.asarray(p_cal, dtype=float); y_cal = np.asarray(y_cal); p_test = np.asarray(p_test, dtype=float)
    s_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
    if mondrian:
        q0 = conformal_quantile(s_cal[y_cal == 0], eps)
        q1 = conformal_quantile(s_cal[y_cal == 1], eps)
    else:
        q0 = q1 = conformal_quantile(s_cal, eps)
    in0 = p_test <= q0                  # score of label 0 is p_hat(1)
    in1 = (1 - p_test) <= q1            # score of label 1 is 1 - p_hat(1)
    out = np.full(len(p_test), -1, dtype=np.int64)
    out[in0 & ~in1] = 0
    out[in1 & ~in0] = 1
    return out, {"q0": q0, "q1": q1}
