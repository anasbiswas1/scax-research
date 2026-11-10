"""Training rule dictionary (plan Section 7.1, B6).

Rules are mined on training data with RuleFit (imodels): rules generated from a tree ensemble and
selected by the lasso. Each rule is a conjunction of conditions lo < x_j <= hi. Every rule enters the
dictionary twice, once claiming no outcome and once claiming the outcome, and is certified like a
SCAX box (Eq. 6) with a prior fixed on the training portion: uniform over the dictionary.

RuleSpace maps records to cell indices on the rules' own thresholds, so rule counts use the same
bit-mask counting as boxes. outer_box gives the smallest box of the model's grid that contains a
rule; the rule is sound for the model exactly when that box is sound (soundness is over real inputs,
and every grid cell meeting the rule's region meets it in a set of positive length).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Rule:
    conds: dict          # feature index -> (lo, hi): lo < x <= hi; -inf / +inf when absent
    text: str

    @property
    def size(self) -> int:
        return len(self.conds)


def parse_conditions(items, names):
    """items: iterable of (feature name, operator, threshold) -> Rule (conditions on one feature merged)."""
    index = {n: i for i, n in enumerate(names)}
    conds = {}
    for name, op, thr in items:
        j = index[name]
        t = float(thr)
        lo, hi = conds.get(j, (-math.inf, math.inf))
        if op == "<=":
            hi = min(hi, t)
        elif op == "<":
            hi = min(hi, float(np.nextafter(t, -math.inf)))
        elif op == ">":
            lo = max(lo, t)
        elif op == ">=":
            lo = max(lo, float(np.nextafter(t, -math.inf)))
        else:
            raise ValueError(f"operator {op}")
        conds[j] = (lo, hi)
    if any(lo >= hi for lo, hi in conds.values()):
        return None                     # empty region
    parts = []
    for j in sorted(conds):
        lo, hi = conds[j]
        s = ("" if lo == -math.inf else f"{lo:g} < ") + names[j] + ("" if hi == math.inf else f" <= {hi:g}")
        parts.append(s)
    return Rule(conds, " and ".join(parts))


def fit_rulefit(X, y, names, max_rules=200, seed=0):
    """RuleFit rules with nonzero lasso coefficient, deduplicated (lasso path without cross-validation,
    rules only, no linear terms)."""
    import warnings

    from imodels import RuleFitClassifier
    m = RuleFitClassifier(max_rules=max_rules, random_state=seed, cv=False, include_linear=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")             # scikit-learn deprecation notices from inside imodels
        m.fit(np.asarray(X, dtype=float), np.asarray(y), feature_names=list(names))
    rules, seen = [], set()
    for r in m.rules_:
        coef = float(r.args[0]) if getattr(r, "args", None) else 1.0
        if coef == 0.0:
            continue
        items = [(name, op, thr) for (name, op), thr in r.agg_dict.items()]
        rule = parse_conditions(items, names)
        if rule is None or rule.size == 0:
            continue
        key = tuple(sorted(rule.conds.items()))
        if key not in seen:
            seen.add(key)
            rules.append(rule)
    return rules


class RuleSpace:
    """Cell indices on the dictionary's thresholds: r_j(x) = #{t in T_j : t < x}."""

    def __init__(self, rules, d):
        vals = [set() for _ in range(d)]
        for r in rules:
            for j, (lo, hi) in r.conds.items():
                if lo != -math.inf:
                    vals[j].add(lo)
                if hi != math.inf:
                    vals[j].add(hi)
        self.T = [np.array(sorted(v), dtype=float) for v in vals]
        self.K = np.array([len(t) for t in self.T], dtype=np.int64)

    def cells(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        C = np.zeros(X.shape, dtype=np.int64)
        for j in np.flatnonzero(self.K > 0):
            C[:, j] = np.searchsorted(self.T[j], X[:, j], side="left")
        return C

    def key(self, rule) -> tuple:
        out = []
        for j in sorted(rule.conds):
            lo, hi = rule.conds[j]
            a = 0 if lo == -math.inf else int(np.searchsorted(self.T[j], lo, side="left")) + 1
            b = int(self.K[j]) if hi == math.inf else int(np.searchsorted(self.T[j], hi, side="left"))
            out.append((j, a, b))
        return tuple(out)


def contains(rule, X) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    m = np.ones(len(X), dtype=bool)
    for j, (lo, hi) in rule.conds.items():
        m &= (X[:, j] > lo) & (X[:, j] <= hi)
    return m


def _f32_down(v):
    f = np.float32(v)
    return float(f) if float(f) <= v else float(np.nextafter(f, np.float32(-np.inf)))


def _f32_up(v):
    f = np.float32(v)
    return float(f) if float(f) >= v else float(np.nextafter(f, np.float32(np.inf)))


def outer_box(rule, grid):
    """Smallest box of the model's grid containing every real input of the rule's region.

    With float32 input casting the region is widened to the float32 values it can reach, so the box
    stays an outer box.
    """
    f32 = grid.ens.input_cast == "float32"
    box = list(grid.full_box())
    for j, (lo, hi) in rule.conds.items():
        K = int(grid.K[j])
        if K == 0:
            continue
        th = grid.thresholds[j]
        if lo == -math.inf:
            a = 0
        elif not f32:
            a = int(np.searchsorted(th, lo, side="right"))
        else:                           # x > lo may be cast down to the float32 value at or below lo
            a = int(np.searchsorted(th, _f32_down(lo), side="right" if grid.rule == "lt" else "left"))
        if hi == math.inf:
            b = K
        else:
            hi_e = _f32_up(hi) if f32 else hi
            b = int(np.searchsorted(th, hi_e, side="right" if grid.rule == "lt" else "left"))
        box[j] = (min(a, K), min(max(a, b), K))
    return tuple(box)
