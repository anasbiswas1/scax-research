"""Primary exact soundness encoding (plan Theorem 1 and Section 8.1).

Literals o[j][k] mean "the cell of feature j is below k", that is, the point goes left at every
node that splits feature j on its k-th threshold. Consecutive literals are ordered. Each leaf has a
path literal equal to the conjunction of the conditions on its path; the score is the exact
integer sum of the leaf values of the active paths. A box fixes ranges of the o literals, and the
query asks for a point in the box whose score lies on the other side of tau'. UNSAT means sound.
A timeout or UNKNOWN answer counts as not sound.
"""
from __future__ import annotations

import z3

from ..models.export import LEAF
from ..models.predictor import ExactPredictor


class Z3CellEncoding:
    name = "z3_cell_literals"

    def __init__(self, pred: ExactPredictor, timeout_ms: int = 20000):
        self.pred = pred
        g = pred.grid
        self.K = g.K
        self.timeout_ms = timeout_ms
        self.o = {j: [None] + [z3.Bool(f"o_{j}_{k}") for k in range(1, int(g.K[j]) + 1)] for j in g.active}
        s = z3.SolverFor("QF_LRA")
        s.set("timeout", int(timeout_ms))
        for j, lits in self.o.items():
            for k in range(1, len(lits) - 1):
                s.add(z3.Implies(lits[k], lits[k + 1]))
        terms = []
        for t, (feat, kk, left, right, _) in enumerate(g.trees):
            stack = [(0, [])]
            while stack:
                i, conds = stack.pop()
                if feat[i] == LEAF:
                    v = pred.leaf_int[t][i]
                    if v != 0:
                        path = z3.And(conds) if conds else z3.BoolVal(True)
                        terms.append(z3.If(path, z3.RealVal(v), z3.RealVal(0)))
                    continue
                lit = self.o[int(feat[i])][int(kk[i])]
                stack.append((int(left[i]), conds + [lit]))
                stack.append((int(right[i]), conds + [z3.Not(lit)]))
        score = z3.Sum(terms) if terms else z3.RealVal(0)
        self.refute = {1: z3.Bool("refute_class1"), 0: z3.Bool("refute_class0")}
        s.add(z3.Implies(self.refute[1], score <= z3.RealVal(pred.tau_int)))
        s.add(z3.Implies(self.refute[0], score > z3.RealVal(pred.tau_int)))
        self.solver = s
        self.calls = 0
        self.unknown = 0

    def box_literals(self, box):
        lits = []
        for j in self.o:
            a, b = box[j]
            K = int(self.K[j])
            if a >= 1:
                lits.append(z3.Not(self.o[j][a]))      # cell >= a
            if b + 1 <= K:
                lits.append(self.o[j][b + 1])          # cell <= b
        return lits

    def check(self, box, y):
        """'sound', 'unsound' or 'unknown'."""
        self.calls += 1
        r = self.solver.check(*(self.box_literals(box) + [self.refute[int(y)]]))
        if r == z3.unsat:
            return "sound"
        if r == z3.sat:
            return "unsound"
        self.unknown += 1
        return "unknown"

    def sound(self, box, y) -> bool:
        return self.check(box, y) == "sound"
