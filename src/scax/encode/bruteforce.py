"""Brute-force soundness by enumerating every cell of a box (small ensembles only).

Each cell is represented by a real input value inside it, and the score is computed by walking the
exported trees with the library's comparison on that real value and summing leaf values as exact
fractions. This path shares nothing with the cell-space predictor except the threshold list used
to pick representatives.
"""
from __future__ import annotations

import itertools
from fractions import Fraction

from ..encode.grid import Grid
from ..models.export import LEAF, Ensemble


def exact_score_point(ens: Ensemble, x) -> Fraction:
    lt = ens.rule == "lt"
    total = Fraction(0)
    for tr in ens.trees:
        i = 0
        while tr.feature[i] != LEAF:
            v = x[tr.feature[i]]
            go_left = (v < tr.threshold[i]) if lt else (v <= tr.threshold[i])
            i = tr.left[i] if go_left else tr.right[i]
        total += Fraction(tr.value[i])
    if ens.score == "margin_sum":
        return Fraction(ens.base) + total
    return total / ens.n_trees


def exact_class_point(ens: Ensemble, x) -> int:
    return int(exact_score_point(ens, x) > Fraction(ens.tau))


def n_cells(grid: Grid, box) -> int:
    out = 1
    for a, b in box:
        out *= (b - a + 1)
    return out


def brute_force_sound(grid: Grid, box, y, max_cells: int = 200000):
    """True / False, or None when the box has more cells than max_cells."""
    if n_cells(grid, box) > max_cells:
        return None
    ranges = [range(a, b + 1) for a, b in box]
    for cell in itertools.product(*ranges):
        x = [grid.representative(j, c) for j, c in enumerate(cell)]
        if exact_class_point(grid.ens, x) != y:
            return False
    return True
