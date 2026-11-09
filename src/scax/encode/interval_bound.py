"""Interval pre-check (plan Section 8.1): a sufficient condition for soundness without a solver.

For each tree, the smallest and largest leaf value reachable inside the box. If even the worst-case
total stays on the predicted side of tau', every point of the box receives the predicted class.
Leaf values are exact floats; only the sum over trees can round, so a float decision is accepted
only when it clears tau' by a wide margin, and the sum is otherwise recomputed exactly in integers.
"""
from __future__ import annotations

import numpy as np

from ..models.export import LEAF
from ..models.predictor import ExactPredictor

_REL_MARGIN = 1e-9


class IntervalBound:
    def __init__(self, pred: ExactPredictor):
        self.pred = pred
        g = pred.grid
        lo, hi, tree_of, node_of, val = [], [], [], [], []
        for t, (feat, k, left, right, values) in enumerate(g.trees):
            stack = [(0, np.zeros(g.d, dtype=np.int64), g.K.copy())]
            while stack:
                i, l, h = stack.pop()
                if feat[i] == LEAF:
                    lo.append(l); hi.append(h); tree_of.append(t); node_of.append(i); val.append(values[i])
                    continue
                f = feat[i]
                hl = h.copy(); hl[f] = min(hl[f], k[i] - 1)        # left: c_f < k
                lr = l.copy(); lr[f] = max(lr[f], k[i])            # right: c_f >= k
                stack.append((left[i], l, hl))
                stack.append((right[i], lr, h))
        order = np.lexsort((np.array(node_of), np.array(tree_of)))
        self.LO = np.array(lo)[order]
        self.HI = np.array(hi)[order]
        self.tree_of = np.array(tree_of)[order]
        self.node_of = np.array(node_of)[order]
        self.val = np.array(val, dtype=np.float64)[order]
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.tree_of) != 0])
        self.T = len(g.trees)
        self.abs_total = float(np.sum([np.max(np.abs(self.val[self.tree_of == t])) for t in range(self.T)]))
        self.tau_f = float(pred.tau_int) / 2.0 ** pred.E if pred.tau_int is not None else None

    def reachable(self, box) -> np.ndarray:
        A = np.array([a for a, _ in box]); B = np.array([b for _, b in box])
        return np.all((self.LO <= B) & (self.HI >= A), axis=1)

    def _extreme(self, reach, want_min):
        fill = np.inf if want_min else -np.inf
        v = np.where(reach, self.val, fill)
        red = np.minimum.reduceat(v, self.starts) if want_min else np.maximum.reduceat(v, self.starts)
        return red

    def _exact_sum(self, reach, want_min):
        total = 0
        for t, s in enumerate(self.starts):
            e = self.starts[t + 1] if t + 1 < len(self.starts) else len(self.val)
            idx = np.arange(s, e)[reach[s:e]]
            vals = [self.pred.leaf_int[t][self.node_of[i]] for i in idx]
            total += min(vals) if want_min else max(vals)
        return total

    def sound(self, box, y) -> bool:
        """True if the interval argument proves the box sound for class y; False means 'not proved'."""
        reach = self.reachable(box)
        want_min = (y == 1)                      # class 1 needs min score > tau'; class 0 needs max <= tau'
        ext = float(np.sum(self._extreme(reach, want_min)))
        gap = ext - self.tau_f
        if abs(gap) > _REL_MARGIN * (self.abs_total + abs(self.tau_f)) + 1e-300:
            return gap > 0 if y == 1 else gap <= 0
        s = self._exact_sum(reach, want_min)
        return s > self.pred.tau_int if y == 1 else s <= self.pred.tau_int

    def single_cell(self, box) -> bool:
        return all(a == b for a, b in box)
