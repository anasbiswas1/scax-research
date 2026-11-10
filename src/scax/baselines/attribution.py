"""Do attribution and anchor explanations form sufficient boxes? (plan Section 7.1, B12; descriptive).

SHAP and LIME rank features. The top-k box fixes the k highest-ranked features at the patient's own
cells and leaves the rest free; it is sufficient when it is sound for the predicted class. Soundness
is monotone in k (each extra fixed feature shrinks the box), so the smallest sufficient k is found by
binary search. Anchors are rules; an anchor is sufficient when the smallest grid box containing it is
sound. These methods were not designed for sufficiency, so the results are descriptive.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .rule_dictionary import Rule


def top_k_box(grid, cell_row, ranking, k):
    box = list(grid.full_box())
    for j in list(ranking)[:k]:
        if grid.K[j] > 0:
            box[j] = (int(cell_row[j]), int(cell_row[j]))
    return tuple(box)


def smallest_sufficient_k(oracle, grid, cell_row, y, ranking):
    """Smallest k whose top-k box is sound (None if even all ranked features are not enough)."""
    ranking = list(ranking)
    if not oracle.sound(top_k_box(grid, cell_row, ranking, len(ranking)), y):
        return None
    lo, hi = 0, len(ranking)
    while lo < hi:
        mid = (lo + hi) // 2
        if oracle.sound(top_k_box(grid, cell_row, ranking, mid), y):
            hi = mid
        else:
            lo = mid + 1
    return lo


_NUM = r"(-?\d+(?:\.\d+)?(?:e-?\d+)?)"
_RANGE = re.compile(rf"^\s*{_NUM}\s*<\s*(.+?)\s*<=\s*{_NUM}\s*$")
_LE = re.compile(rf"^\s*(.+?)\s*<=\s*{_NUM}\s*$")
_GT = re.compile(rf"^\s*(.+?)\s*>\s*{_NUM}\s*$")


def anchor_rule(names_list, feature_names, bins=None):
    """Rule from anchor predicate strings such as 'BMI <= 24.00' or '2.00 < Age <= 9.00'.

    Printed values are rounded; with bins (the discretizer's exact thresholds per feature) each value is
    replaced by the nearest exact threshold.
    """
    index = {n: i for i, n in enumerate(feature_names)}
    conds = {}

    def exact(j, v):
        if bins is None or j not in bins or len(bins[j]) == 0:
            return v
        b = np.asarray(bins[j], dtype=float)
        return float(b[np.argmin(np.abs(b - v))])

    for s in names_list:
        m = _RANGE.match(s)
        if m:
            j = index[m.group(2)]; lo, hi = exact(j, float(m.group(1))), exact(j, float(m.group(3)))
        else:
            m = _LE.match(s)
            if m:
                j = index[m.group(1)]; lo, hi = -math.inf, exact(j, float(m.group(2)))
            else:
                m = _GT.match(s)
                if not m:
                    raise ValueError(f"cannot parse anchor predicate: {s}")
                j = index[m.group(1)]; lo, hi = exact(j, float(m.group(2))), math.inf
        plo, phi = conds.get(j, (-math.inf, math.inf))
        conds[j] = (max(plo, lo), min(phi, hi))
    return Rule(conds, " and ".join(names_list))
