"""Independent exact soundness check with Z3 on real-valued inputs (plan Theorem 1, Algorithm 3 step 5).

It does not use cell indices, leaf ranges or path literals. Each input is a real variable, each tree
is a nested if-then-else over the library's own comparison (x < theta or x <= theta) with the stored
threshold as an exact rational, and the score is the exact rational sum of leaf values. For each
query the trees are first simplified with the box's real bounds (a branch whose comparison is
decided by the bounds is dropped), and Z3 is asked for a point inside the box whose score lies on
the other side of tau. UNSAT means sound; SAT means unsound; a timeout or UNKNOWN counts as not
sound. Agreement with the cell search therefore also checks the cell mapping and the comparison
rules.

PySAT is not used here: boosted leaf values are rationals, which a propositional encoding can hold
only through a pseudo-Boolean library that is not part of the stack.
"""
from __future__ import annotations

from fractions import Fraction

import z3

from ..models.export import LEAF, Ensemble


def _q(f: Fraction):
    return z3.Q(f.numerator, f.denominator)


class Z3RealEncoding:
    name = "z3_real_inputs"

    def __init__(self, ens: Ensemble, thresholds, timeout_ms: int = 60000):
        """thresholds: per-feature sorted threshold arrays, used only to turn a cell box into real bounds."""
        self.ens = ens
        self.thresholds = thresholds
        self.timeout_ms = timeout_ms
        self.lt = ens.rule == "lt"
        self.x = [z3.Real(f"x_{j}") for j in range(ens.n_features)]
        self.vals = [[Fraction(v) for v in tr.value] for tr in ens.trees]
        self.ths = [[Fraction(t) for t in tr.threshold] for tr in ens.trees]
        self.tau = Fraction(ens.tau)
        self.calls = 0
        self.unknown = 0

    def real_bounds(self, box):
        """Per feature (lower, upper) thresholds of the box as Fractions or None."""
        out = []
        for j, (a, b) in enumerate(box):
            th = self.thresholds[j]
            K = len(th)
            lo = Fraction(float(th[a - 1])) if (K and a >= 1) else None
            hi = Fraction(float(th[b])) if (K and b + 1 <= K) else None
            out.append((lo, hi))
        return out

    def _tree(self, t, bounds):
        tr = self.ens.trees[t]
        vals, ths = self.vals[t], self.ths[t]

        def walk(i):
            if tr.feature[i] == LEAF:
                return vals[i]
            j = tr.feature[i]
            lo, hi = bounds[j]
            th = ths[i]
            # lt: box is lo <= x < hi, node goes left iff x < th; le: box is lo < x <= hi, left iff x <= th.
            if hi is not None and hi <= th:
                return walk(tr.left[i])
            if lo is not None and lo >= th:
                return walk(tr.right[i])
            l, r = walk(tr.left[i]), walk(tr.right[i])
            if isinstance(l, Fraction) and isinstance(r, Fraction) and l == r:
                return l
            cond = (self.x[j] < _q(th)) if self.lt else (self.x[j] <= _q(th))
            return z3.If(cond, _q(l) if isinstance(l, Fraction) else l, _q(r) if isinstance(r, Fraction) else r)

        return walk(0)

    def check(self, box, y):
        self.calls += 1
        bounds = self.real_bounds(box)
        const = Fraction(0)
        terms = []
        for t in range(self.ens.n_trees):
            e = self._tree(t, bounds)
            if isinstance(e, Fraction):
                const += e
            else:
                terms.append(e)
        if self.ens.score == "margin_sum":
            const += Fraction(self.ens.base)
            scale = Fraction(1)
        else:
            scale = Fraction(1, self.ens.n_trees)
        if not terms:
            s = const * scale
            ok = (s > self.tau) if int(y) == 1 else (s <= self.tau)
            return "sound" if ok else "unsound"
        score = (_q(const) + z3.Sum(terms)) * _q(scale)
        solver = z3.SolverFor("QF_LRA")
        solver.set("timeout", int(self.timeout_ms))
        for j, (lo, hi) in enumerate(bounds):
            xj = self.x[j]
            if lo is not None:
                solver.add(xj >= _q(lo) if self.lt else xj > _q(lo))
            if hi is not None:
                solver.add(xj < _q(hi) if self.lt else xj <= _q(hi))
        solver.add(score <= _q(self.tau) if int(y) == 1 else score > _q(self.tau))
        r = solver.check()
        self.last_solver = solver
        if r == z3.unsat:
            return "sound"
        if r == z3.sat:
            return "unsound"
        self.unknown += 1
        return "unknown"

    def sound(self, box, y) -> bool:
        return self.check(box, y) == "sound"

    def counterexample(self, box, y):
        if self.check(box, y) != "unsound":
            return None
        m = self.last_solver.model()
        out = []
        for v in self.x:
            val = m.eval(v, model_completion=True)
            out.append(float(Fraction(val.numerator_as_long(), val.denominator_as_long())))
        return out
