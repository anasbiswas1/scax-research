"""Box certificates on calibration split 1 (plan Definition 7, Eq. 6, Algorithm 4).

For each box B with class y_B: n_B calibration points inside, k_B of them with recorded outcome
different from y_B, and the bound U(k_B, n_B; delta_1 * pi(B)). Levels are carried in log space;
a level below the smallest positive double gives the bound 1 (never released).
"""
from __future__ import annotations

import hashlib

import numpy as np
from scipy import sparse
from scipy.stats import beta

LOG_TINY = np.log(np.finfo(float).tiny)


def membership(grid, keys, C) -> sparse.csr_matrix:
    """Sparse boxes x rows matrix: 1 where row (cell indices C) lies inside the box."""
    rows, cols = [], []
    for i, key in enumerate(keys):
        idx = np.flatnonzero(grid.contains(key, C))
        rows.append(np.full(idx.size, i, dtype=np.int64)); cols.append(idx)
    r = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    c = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    return sparse.csr_matrix((np.ones(r.size, dtype=np.float64), (r, c)), shape=(len(keys), C.shape[0]))


def counts(mem: sparse.csr_matrix, box_class, y, weights=None):
    """n_B and k_B for labels y (one per row of the membership matrix), optional row weights."""
    w = np.ones(mem.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    y = np.asarray(y)
    n = mem @ w
    pos = mem @ (w * (y == 1))
    box_class = np.asarray(box_class)
    k = np.where(box_class == 1, n - pos, pos)
    return np.rint(n).astype(np.int64), np.rint(k).astype(np.int64)


def upper_bounds(k, n, log_gamma) -> np.ndarray:
    """U(k, n; gamma) with gamma = exp(log_gamma), vectorised; 1 when k == n, n == 0 or gamma underflows."""
    k = np.asarray(k, dtype=float); n = np.asarray(n, dtype=float)
    lg = np.broadcast_to(np.asarray(log_gamma, dtype=float), k.shape)
    out = np.ones(k.shape)
    ok = (n > 0) & (k < n) & (lg > LOG_TINY)
    if ok.any():
        out[ok] = beta.isf(np.exp(lg[ok]), k[ok] + 1.0, n[ok] - k[ok])
    return out


def certificate_hash(model_hash: str, prior_hash: str, split_hash: str) -> str:
    return hashlib.sha256(f"{model_hash}|{prior_hash}|{split_hash}".encode()).hexdigest()
