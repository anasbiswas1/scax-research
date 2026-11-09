"""Bound comparisons without inverting the Beta distribution.

The one-sided Clopper-Pearson upper limit U(k, n; gamma) is the p at which the binomial CDF
F(k; n, p) equals gamma, and F decreases in p. Hence, for k < n,

    U(k, n; gamma) <= a   if and only if   F(k; n, a) <= gamma,
    U(k, n; gamma) <  r   if and only if   F(k; n, r) <  gamma.

F(k; n, a) does not depend on gamma, so one evaluation per box serves every prior and every
delta_1. For k = n or n = 0 the bound is 1 and F = 1, which the same comparisons handle.
"""
from __future__ import annotations

import numpy as np
from scipy.special import bdtr


def binom_cdf(k, n, p) -> np.ndarray:
    k = np.asarray(k, dtype=np.int64); n = np.asarray(n, dtype=np.int64)
    p = np.broadcast_to(np.asarray(p, dtype=float), k.shape)
    out = np.ones(k.shape)
    ok = (n > 0) & (k < n)
    if ok.any():
        out[ok] = bdtr(k[ok], n[ok], p[ok])
    return out


LOG_TINY = np.log(np.finfo(float).tiny)


def passes(cdf_at_alpha, log_gamma) -> np.ndarray:
    """U <= alpha, given F(k; n, alpha) and log gamma (a level that underflows gives U = 1)."""
    with np.errstate(divide="ignore"):
        return (np.log(cdf_at_alpha) <= log_gamma) & (np.asarray(log_gamma) > LOG_TINY)


def violates(cdf_at_true_risk, log_gamma) -> np.ndarray:
    """U < R, given F(k; n, R) and log gamma (a level that underflows gives U = 1, never below R)."""
    with np.errstate(divide="ignore"):
        return (np.log(cdf_at_true_risk) < log_gamma) & (np.asarray(log_gamma) > LOG_TINY)
