"""Second exact soundness check, on real thresholds and exact fractions.

This is an independent implementation of the leaf branch-and-bound that shares no data structure
with the primary check: it reads the exported trees directly, builds each leaf's region as real
intervals from the stored thresholds and the library's comparison rule ([lo, hi) for "lt",
(lo, hi] for "le"), works on real box bounds rather than cell indices, and sums leaf values as
exact fractions on the library's own scale (base margin, or mean probability). It runs on every
candidate box; a disagreement with the primary check stops the run. Z3 checks a random sample
of boxes and brute force checks small ensembles exhaustively.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from ..models.export import LEAF, Ensemble


class RealSearch:
    name = "real_threshold_search"

    def __init__(self, ens: Ensemble, thresholds, node_budget: int = 50000):
        self.ens = ens
        self.thresholds = thresholds
        self.node_budget = node_budget
        d = ens.n_features
        lo, hi, tree_of, vals = [], [], [], []
        for t, tr in enumerate(ens.trees):
            stack = [(0, np.full(d, -np.inf), np.full(d, np.inf))]
            while stack:
                i, l, h = stack.pop()
                if tr.feature[i] == LEAF:
                    lo.append(l); hi.append(h); tree_of.append(t); vals.append(tr.value[i])
                    continue
                j, th = tr.feature[i], tr.threshold[i]
                hl = h.copy(); hl[j] = min(hl[j], th)    # left: x < th (lt) or x <= th (le)
                lr = l.copy(); lr[j] = max(lr[j], th)    # right: x >= th (lt) or x > th (le)
                stack.append((tr.left[i], l, hl))
                stack.append((tr.right[i], lr, h))
        order = np.argsort(np.array(tree_of), kind="stable")
        self.LO = np.array(lo)[order]
        self.HI = np.array(hi)[order]
        self.tree_of = np.array(tree_of)[order]
        self.val = np.array(vals, dtype=np.float64)[order]
        self.frac = [Fraction(float(v)) for v in self.val]
        self.starts = np.flatnonzero(np.r_[True, np.diff(self.tree_of) != 0])
        self.ends = np.r_[self.starts[1:], len(self.val)]
        self.seg_len = self.ends - self.starts
        self.T = ens.n_trees
        # kappa(x) = 1[sum_t v_t > tau'] on the library's scale
        if ens.score == "margin_sum":
            self.tau_eff = Fraction(ens.tau) - Fraction(ens.base)
        else:
            self.tau_eff = Fraction(ens.tau) * self.T
        self.tau_f = float(self.tau_eff)
        abs_total = sum(float(np.max(np.abs(self.val[s:e]))) for s, e in zip(self.starts, self.ends))
        self.margin = 1e-9 * (abs_total + abs(self.tau_f)) + 1e-300
        self.calls = 0
        self.unknown = 0

    def box_bounds(self, box):
        L = np.full(self.ens.n_features, -np.inf); U = np.full(self.ens.n_features, np.inf)
        for j, (a, b) in enumerate(box):
            th = self.thresholds[j]
            if len(th) == 0:
                continue
            if a >= 1:
                L[j] = th[a - 1]
            if b + 1 <= len(th):
                U[j] = th[b]
        return L, U

    def _crosses(self, v, want_max):
        best = np.maximum.reduceat(v, self.starts)
        s = float(np.sum(best))
        thr = self.tau_f if want_max else -self.tau_f
        if s < thr - self.margin:
            return False
        if s > thr + self.margin:
            return True
        total = Fraction(0)
        for a, e in zip(self.starts, self.ends):
            total += self.frac[a + int(np.argmax(v[a:e]))]
        return total > self.tau_eff if want_max else -total >= -self.tau_eff

    def check(self, box, y):
        self.calls += 1
        want_max = int(y) == 0
        sv = self.val if want_max else -self.val
        L, U = self.box_bounds(box)
        stack = [(L, U)]
        nodes = 0
        while stack:
            L, U = stack.pop()
            nodes += 1
            if nodes > self.node_budget:
                self.unknown += 1
                return "unknown"
            reach = np.all(np.maximum(self.LO, L) < np.minimum(self.HI, U), axis=1)
            v = np.where(reach, sv, -np.inf)
            if not self._crosses(v, want_max):
                continue
            cnt = np.add.reduceat(reach.astype(np.int64), self.starts)
            multi = cnt > 1
            if not multi.any():
                s = sum((self.frac[i] for i in np.flatnonzero(reach)), Fraction(0))
                if (s > self.tau_eff) if want_max else (s <= self.tau_eff):
                    return "unsound"
                continue
            best = np.maximum.reduceat(v, self.starts)
            is_top = v == np.repeat(best, self.seg_len)
            n_top = np.add.reduceat(is_top.astype(np.int64), self.starts)
            second = np.maximum.reduceat(np.where(is_top, -np.inf, v), self.starts)
            gap = np.where(multi, np.where(n_top > 1, 0.0, best - second), -1.0)
            t = int(np.argmax(gap))
            a, e = self.starts[t], self.ends[t]
            idx = np.arange(a, e)[reach[a:e]]
            idx = idx[np.argsort(sv[idx], kind="stable")]
            for i in idx:
                nL = np.maximum(L, self.LO[i]); nU = np.minimum(U, self.HI[i])
                if np.all(nL < nU):
                    stack.append((nL, nU))
        return "sound"
