"""Smallest sufficient top-k by binary search equals a linear scan; anchor predicates parse to rules."""
import math

import numpy as np

from scax.baselines.attribution import anchor_rule, smallest_sufficient_k, top_k_box
from scax.explain.candidates import SoundnessOracle
from small_models import small_classifier


def test_binary_search_matches_scan():
    _, ens, pred, X, _ = small_classifier("xgboost")
    g = pred.grid
    oracle = SoundnessOracle(pred)
    rng = np.random.default_rng(0)
    C = g.cells(X[:80]); yhat = pred.predict_cells(C)
    for i in range(len(C)):
        ranking = list(rng.permutation(g.d))
        k = smallest_sufficient_k(oracle, g, C[i], int(yhat[i]), ranking)
        scan = next(kk for kk in range(g.d + 1) if oracle.sound(top_k_box(g, C[i], ranking, kk), int(yhat[i])))
        assert k == scan


def test_anchor_parsing():
    names = ["HighBP", "Age", "BMI"]
    r = anchor_rule(["BMI <= 41.00", "HighBP <= 0.00", "4.00 < Age <= 7.00"], names,
                    bins={2: [23.0, 32.0, 41.0], 1: [4.0, 7.0, 10.0]})
    assert r.conds == {2: (-math.inf, 41.0), 0: (-math.inf, 0.0), 1: (4.0, 7.0)}
    r = anchor_rule(["BMI > 31.99"], names, bins={2: [23.0, 32.0, 41.0]})
    assert r.conds == {2: (32.0, math.inf)}
