"""SCAX's exported predictor: exact scores and decisions (plan Section 5.3).

Every leaf value is a float stored by the library, hence a dyadic rational. All leaf values and the
effective threshold tau' are scaled by one power of two 2**E so that they become integers, and the
score of a point is an exact integer sum. The decision is kappa(x) = 1[sum_t v_t(x) > tau'].
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..encode.grid import Grid
from .export import LEAF, Ensemble


def _exp2_den(fr: Fraction) -> int:
    d = fr.denominator
    if d & (d - 1):
        raise ValueError("value is not a dyadic rational")
    return d.bit_length() - 1


class ExactPredictor:
    def __init__(self, ens: Ensemble, grid: Grid | None = None):
        self.ens = ens
        self.grid = grid or Grid(ens)
        values = [Fraction(v) for t in ens.trees for f, v in zip(t.feature, t.value) if f == LEAF]
        tau_eff = ens.effective_threshold() if ens.tau is not None else Fraction(0)
        self.E = max([_exp2_den(v) for v in values] + [_exp2_den(tau_eff), 0])
        scale = 1 << self.E
        self.leaf_int = []   # per tree: python ints per node (0 at internal nodes)
        for t in ens.trees:
            self.leaf_int.append([int(Fraction(v) * scale) if f == LEAF else 0 for f, v in zip(t.feature, t.value)])
        self.tau_int = int(tau_eff * scale) if ens.tau is not None else None
        bound = sum(max(abs(v) for v in li) for li in self.leaf_int) + abs(self.tau_int or 0)
        self.use_int64 = bound < (1 << 62)
        self._leaf_arr = [np.array(li, dtype=np.int64 if self.use_int64 else object) for li in self.leaf_int]
        self.depth = max(self._depth(t) for t in self.grid.trees)

    @staticmethod
    def _depth(tree):
        feat, _, left, right, _ = tree
        best, stack = 0, [(0, 0)]
        while stack:
            i, dpt = stack.pop()
            if feat[i] == LEAF:
                best = max(best, dpt)
            else:
                stack += [(left[i], dpt + 1), (right[i], dpt + 1)]
        return best

    def leaves(self, C) -> np.ndarray:
        """Leaf node index reached in every tree, shape (n, T), from cell indices C."""
        n = C.shape[0]
        out = np.zeros((n, len(self.grid.trees)), dtype=np.int64)
        rows = np.arange(n)
        for t, (feat, k, left, right, _) in enumerate(self.grid.trees):
            node = np.zeros(n, dtype=np.int64)
            for _ in range(self.depth):
                f = feat[node]
                internal = f != LEAF
                if not internal.any():
                    break
                go_left = C[rows, np.where(internal, f, 0)] < k[node]
                node = np.where(internal, np.where(go_left, left[node], right[node]), node)
            out[:, t] = node
        return out

    def score_int(self, C) -> np.ndarray:
        """Exact scaled score sum_t v_t * 2**E for each row of cell indices."""
        L = self.leaves(C)
        s = np.zeros(C.shape[0], dtype=np.int64 if self.use_int64 else object)
        for t in range(L.shape[1]):
            s = s + self._leaf_arr[t][L[:, t]]
        return s

    def score(self, X) -> np.ndarray:
        """Score on the library's scale (margin, or mean probability), as float, for reporting only."""
        s = np.ldexp(np.asarray(self.score_int(self.grid.cells(X)), dtype=float), -self.E)
        if self.ens.score == "margin_sum":
            return s + self.ens.base
        return s / self.ens.n_trees

    def predict_cells(self, C) -> np.ndarray:
        if self.tau_int is None:
            raise ValueError("tau not set")
        return (self.score_int(C) > self.tau_int).astype(np.int8)

    def predict(self, X) -> np.ndarray:
        return self.predict_cells(self.grid.cells(X))

    def margin_to_tau(self, X) -> np.ndarray:
        """(score - tau) on the library's scale, for listing disagreements by distance from the threshold."""
        s_int = self.score_int(self.grid.cells(X))
        diff = np.array([Fraction(int(v) - self.tau_int, 1 << self.E) for v in s_int], dtype=object)
        if self.ens.score == "mean_probability":
            diff = diff / self.ens.n_trees
        return diff.astype(float)


def native_scores(model, family, X) -> np.ndarray:
    """Native library score on the same scale as the exported score."""
    X = np.asarray(X, dtype=np.float64)
    if family == "xgboost":
        return model.predict(X, output_margin=True).astype(np.float64)
    if family == "lightgbm":
        return model.predict(X, raw_score=True).astype(np.float64)
    if family == "sklearn_rf":
        return model.predict_proba(X)[:, 1].astype(np.float64)
    raise ValueError(family)
