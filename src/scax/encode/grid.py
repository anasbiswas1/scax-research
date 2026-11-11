"""Threshold grid, cells and boxes (plan Definitions 2 and 3).

For feature j with sorted distinct thresholds theta_1 < ... < theta_K, a point's cell index c_j is
the number of thresholds on the far side of the split: with rule "lt" (x < theta goes left)
c_j = #{k: theta_k <= x}; with rule "le" (x <= theta goes left) c_j = #{k: theta_k < x}. Either way a
node splitting on theta_k sends the point left exactly when c_j < k, so the trees can be read on
cell indices alone.

A box fixes, for every feature, an inclusive range of cells [a_j, b_j] with 0 <= a_j <= b_j <= K_j;
a feature is free when [a_j, b_j] = [0, K_j]. A box is stored as a tuple of (a_j, b_j) pairs for all
features; its key lists only the constrained features.
"""
from __future__ import annotations

from math import comb, log10

import numpy as np

from ..models.export import LEAF, Ensemble


class Grid:
    def __init__(self, ens: Ensemble):
        self.ens = ens
        self.rule = ens.rule
        self.d = ens.n_features
        used = [set() for _ in range(self.d)]
        for t in ens.trees:
            for f, th in zip(t.feature, t.threshold):
                if f != LEAF:
                    used[f].add(th)
        self.thresholds = [np.array(sorted(s), dtype=np.float64) for s in used]
        self.K = np.array([len(s) for s in self.thresholds], dtype=np.int64)
        self.active = np.flatnonzero(self.K > 0)
        # cell-space trees: node -> (feature, k) with "left iff c_feature < k"
        self.trees = []
        for t in ens.trees:
            k = [0] * len(t.feature)
            for i, (f, th) in enumerate(zip(t.feature, t.threshold)):
                if f != LEAF:
                    k[i] = int(np.searchsorted(self.thresholds[f], th, side="left")) + 1
                    assert self.thresholds[f][k[i] - 1] == th
            self.trees.append((np.array(t.feature), np.array(k), np.array(t.left), np.array(t.right),
                               np.array(t.value, dtype=np.float64)))

    # ---------------------------------------------------------------- inputs and cells
    def cast(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != self.d:
            raise ValueError(f"expected {self.d} columns")
        if np.isnan(X).any():
            raise ValueError("the exported classifier declares no missing values")
        if self.ens.input_cast == "float32":
            X = X.astype(np.float32).astype(np.float64)
        return X

    def cells(self, X) -> np.ndarray:
        X = self.cast(X)
        side = "right" if self.rule == "lt" else "left"
        C = np.zeros(X.shape, dtype=np.int64)
        for j in self.active:
            C[:, j] = np.searchsorted(self.thresholds[j], X[:, j], side=side)
        if self.ens.family == "xgboost" and self.ens.input_cast == "float64":
            # XGBoost compares float32(x); a float64 export is exact only where that gives the same cell
            X32 = X.astype(np.float32).astype(np.float64)
            for j in self.active:
                if not np.array_equal(np.searchsorted(self.thresholds[j], X32[:, j], side=side), C[:, j]):
                    raise ValueError(f"feature {j}: an input lies within float32 rounding of a threshold, where "
                                     "XGBoost (which converts inputs to float32) and this float64 export differ; "
                                     "export with input_cast='float32'")
        return C

    def representative(self, j: int, c: int) -> float:
        """A real input value whose cell on feature j is c."""
        th = self.thresholds[j]
        K = len(th)
        if K == 0:
            return 0.0
        if self.rule == "lt":           # cell c is [theta_c, theta_{c+1})
            return float(th[0] - 1.0) if c == 0 else float(th[c - 1])
        return float(th[-1] + 1.0) if c == K else float(th[c])   # cell c is (theta_c, theta_{c+1}]

    # ---------------------------------------------------------------- boxes
    def full_box(self):
        return tuple((0, int(k)) for k in self.K)

    def point_box(self, cell_row):
        return tuple((int(c), int(c)) if k > 0 else (0, 0) for c, k in zip(cell_row, self.K))

    def constrained(self, box):
        return [j for j, (a, b) in enumerate(box) if (a, b) != (0, int(self.K[j]))]

    def size(self, box) -> int:
        return len(self.constrained(box))

    def key(self, box) -> tuple:
        return tuple((j, box[j][0], box[j][1]) for j in self.constrained(box))

    def from_key(self, key):
        box = list(self.full_box())
        for j, a, b in key:
            box[j] = (a, b)
        return tuple(box)

    def contains(self, box_or_key, C) -> np.ndarray:
        key = box_or_key if (len(box_or_key) == 0 or len(box_or_key[0]) == 3) else self.key(box_or_key)
        m = np.ones(C.shape[0], dtype=bool)
        for j, a, b in key:
            m &= (C[:, j] >= a) & (C[:, j] <= b)
        return m

    def describe(self, box, names=None) -> str:
        names = names or self.ens.feature_names
        parts = []
        for j in self.constrained(box):
            a, b = box[j]
            th = self.thresholds[j]
            lo = None if a == 0 else th[a - 1]
            hi = None if b == len(th) else th[b]
            if self.rule == "lt":
                s = ("" if lo is None else f"{lo:g} <= ") + names[j] + ("" if hi is None else f" < {hi:g}")
            else:
                s = ("" if lo is None else f"{lo:g} < ") + names[j] + ("" if hi is None else f" <= {hi:g}")
            parts.append(s)
        return " and ".join(parts) if parts else "(all patients)"

    # ---------------------------------------------------------------- family size, Eq. (1)
    def intervals_per_feature(self):
        return [comb(int(k) + 2, 2) for k in self.K]

    def family_size(self) -> int:
        out = 1
        for m in self.intervals_per_feature():
            out *= m
        return out

    def family_size_log10(self) -> float:
        return float(sum(log10(m) for m in self.intervals_per_feature()))
