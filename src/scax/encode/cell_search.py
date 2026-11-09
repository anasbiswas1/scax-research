"""Exact soundness decision by branch-and-bound over tree leaves.

To decide whether a box is sound for class y, the search looks for a counterexample: a point in the
box whose exact score lies on the other side of tau'. A search node is a sub-box. Its bound is the
sum over trees of the best leaf value reachable inside it (the interval argument); a node whose
bound cannot cross tau' is discarded. Otherwise one tree is fixed: the tree with the largest gap
between its best and second-best reachable leaf, and the node is split into one child per
reachable leaf of that tree (the sub-box intersected with the leaf's region). When every tree has
a single reachable leaf, the node's score is exact and is compared with tau'. The children cover
the sub-box, so the search is complete and its answer is exact. Leaf values are exact floats; any
float sum close to tau' is recomputed in integers. A node budget turns very large searches into
"unknown", which counts as not sound.
"""
from __future__ import annotations

import numpy as np

from .interval_bound import IntervalBound, _REL_MARGIN


class CellSearch:
    name = "leaf_branch_and_bound"

    def __init__(self, ib: IntervalBound, node_budget: int = 50000):
        self.ib = ib
        self.pred = ib.pred
        self.node_budget = node_budget
        self.vint = [ib.pred.leaf_int[ib.tree_of[i]][ib.node_of[i]] for i in range(len(ib.val))]
        self.ends = np.r_[ib.starts[1:], len(ib.val)]
        self.tau_int = ib.pred.tau_int
        self.tau_f = ib.tau_f
        self.margin = _REL_MARGIN * (ib.abs_total + abs(self.tau_f)) + 1e-300
        self.seg_len = self.ends - ib.starts
        self.calls = 0
        self.nodes = 0
        self.unknown = 0

    def _may_cross(self, v, reach, want_max) -> bool:
        """Exact test: can the per-tree best leaves (in signed values v) cross tau'?"""
        ib = self.ib
        best = np.maximum.reduceat(v, ib.starts)
        s = float(np.sum(best))
        thr = self.tau_f if want_max else -self.tau_f
        if s < thr - self.margin:
            return False
        if s > thr + self.margin:
            return True
        # exact: per-tree best leaf (any one attaining the maximum) in integers
        total = 0
        for t, (a, e) in enumerate(zip(ib.starts, self.ends)):
            seg = v[a:e]
            i = a + int(np.argmax(seg))
            total += self.vint[i]
        return total > self.tau_int if want_max else total <= self.tau_int

    def find_counterexample(self, A, B, want_max):
        """A sub-box (A, B) of single-leaf trees whose exact score crosses tau', None, or 'unknown'."""
        ib = self.ib
        LO, HI, starts, ends = ib.LO, ib.HI, ib.starts, self.ends
        sv = ib.val if want_max else -ib.val
        stack = [(np.asarray(A).copy(), np.asarray(B).copy())]
        nodes = 0
        while stack:
            A, B = stack.pop()
            nodes += 1
            if nodes > self.node_budget:
                self.nodes += nodes
                return "unknown"
            reach = np.all((LO <= B) & (HI >= A), axis=1)
            v = np.where(reach, sv, -np.inf)
            if not self._may_cross(v, reach, want_max):
                continue
            cnt = np.add.reduceat(reach.astype(np.int64), starts)
            multi = cnt > 1
            if not multi.any():
                s = sum(self.vint[i] for i in np.flatnonzero(reach))
                if (s > self.tau_int) if want_max else (s <= self.tau_int):
                    self.nodes += nodes
                    return (A, B)
                continue
            # gap between best and second-best reachable leaf, per tree
            best = np.maximum.reduceat(v, starts)
            tree_best = np.repeat(best, self.seg_len)
            is_top = v == tree_best
            n_top = np.add.reduceat(is_top.astype(np.int64), starts)
            second = np.maximum.reduceat(np.where(is_top, -np.inf, v), starts)
            gap = np.where(n_top > 1, 0.0, best - second)
            gap = np.where(multi, gap, -1.0)
            t = int(np.argmax(gap))
            idx = np.arange(starts[t], ends[t])[reach[starts[t]:ends[t]]]
            idx = idx[np.argsort(sv[idx], kind="stable")]       # best leaf pushed last, explored first
            for i in idx:
                nA = np.maximum(A, LO[i]); nB = np.minimum(B, HI[i])
                if np.all(nA <= nB):
                    stack.append((nA, nB))
        self.nodes += nodes
        return None

    def check(self, box, y):
        self.calls += 1
        A = np.array([a for a, _ in box], dtype=np.int64)
        B = np.array([b for _, b in box], dtype=np.int64)
        r = self.find_counterexample(A, B, want_max=(int(y) == 0))
        if isinstance(r, str):
            self.unknown += 1
            return "unknown"
        return "sound" if r is None else "unsound"

    def sound(self, box, y) -> bool:
        return self.check(box, y) == "sound"
