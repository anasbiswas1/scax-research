"""Confidence-aware AXps keep the class and the confidence floor everywhere in the box, and are subset-minimal."""
import numpy as np

from scax.baselines.confidence_axp import ConfidenceAXp
from scax.encode.bruteforce import brute_force_sound
from small_models import small_classifier


def test_confidence_axp_sound_and_minimal():
    _, ens, pred, X, _ = small_classifier("xgboost")
    g = pred.grid
    caxp = ConfidenceAXp(ens, {0: 0.75, 1: 0.6})
    C = g.cells(X[:150]); yhat = pred.predict_cells(C).astype(int)
    elig = caxp.eligible(C, yhat)
    assert elig.any() and (~elig).any()
    for i in np.flatnonzero(elig)[:60]:
        y = int(yhat[i])
        box = caxp.explain(C[i], y)
        gy = caxp.gen[y].pred.grid
        assert brute_force_sound(gy, box, y)
        assert brute_force_sound(g, box, y)            # the original class is kept too
        for j in gy.constrained(box):
            freed = list(box); freed[j] = (0, int(gy.K[j]))
            assert not brute_force_sound(gy, tuple(freed), y)
