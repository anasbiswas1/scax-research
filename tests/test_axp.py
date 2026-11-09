"""Candidate sets: every AXp is sound and subset-minimal; every inflated box is sound."""
import numpy as np

from scax.encode.bruteforce import brute_force_sound
from scax.explain.candidates import CandidateGenerator
from small_models import small_classifier


def test_axp_minimal_and_inflation_sound(tmp_path):
    m, ens, pred, X, y = small_classifier("xgboost")
    g = pred.grid
    gen = CandidateGenerator(pred, M=4, seed=0)
    C = g.cells(X[:60])
    for i in range(len(C)):
        yhat = int(pred.predict_cells(C[i:i + 1])[0])
        cands = gen.candidates(C[i])
        assert cands and all(c.y == yhat for c in cands)
        for c in cands:
            box = g.from_key(c.key)
            assert g.contains(c.key, C[i:i + 1])[0]
            assert brute_force_sound(g, box, yhat)
            if c.kind == "axp":
                for j in g.constrained(box):
                    freed = list(box); freed[j] = (0, int(g.K[j]))
                    assert not brute_force_sound(g, tuple(freed), yhat)
    assert gen.oracle.stats.search_unknown == 0
