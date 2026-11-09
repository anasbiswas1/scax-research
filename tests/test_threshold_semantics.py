"""Comparison rules and threshold storage of each library match the exported classifier."""
import numpy as np
import pytest

from scax.encode.grid import Grid
from scax.models.predictor import native_scores
from small_models import small_classifier


@pytest.mark.parametrize("family,rule", [("xgboost", "lt"), ("lightgbm", "le"), ("sklearn_rf", "le")])
def test_rule_and_points_on_thresholds(family, rule):
    m, ens, pred, X, y = small_classifier(family)
    assert ens.rule == rule
    g = pred.grid
    # a point exactly on a threshold goes right under "lt" (x < theta is false) and left under "le"
    for j in g.active:
        for k, th in enumerate(g.thresholds[j], start=1):
            Z = np.zeros((1, g.d)); Z[0, j] = th
            c = g.cells(Z)[0, j]
            assert (c >= k) if rule == "lt" else (c < k)
    # on and around every threshold, the exported score equals the native score
    pts = []
    for j in g.active:
        for th in g.thresholds[j]:
            for dx in (-1e-3, 0.0, 1e-3):
                z = X[0].copy(); z[j] = th + dx; pts.append(z)
    P = np.array(pts)
    assert np.allclose(pred.score(P), native_scores(m, family, P), atol=1e-5, rtol=0)


def test_xgboost_thresholds_are_float32():
    m, ens, pred, X, y = small_classifier("xgboost")
    for t in ens.trees:
        for f, th in zip(t.feature, t.threshold):
            if f >= 0:
                assert float(np.float32(th)) == th


def test_missing_values_refused():
    m, ens, pred, X, y = small_classifier("xgboost")
    Z = X[:2].copy(); Z[0, 0] = np.nan
    with pytest.raises(ValueError):
        pred.grid.cells(Z)
