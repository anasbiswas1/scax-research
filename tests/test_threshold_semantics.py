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


def test_xgboost_float32_input_conversion():
    """XGBoost converts inputs to float32: a float32 export matches it everywhere, and a float64 export
    refuses inputs whose cell would differ after that conversion."""
    from scax.models.export import export_model
    from scax.models.predictor import ExactPredictor
    from scax.models.train import fit
    rng = np.random.default_rng(0)
    X = np.round(rng.uniform(0, 5, (3000, 2)), 1)
    y = (X[:, 0] + rng.normal(0, 1, 3000) > 2.5).astype(int)
    m = fit("xgboost", X, y, {"n_estimators": 5, "max_depth": 2}, 0)
    e64, e32 = export_model(m, "xgboost", ["a", "b"]), export_model(m, "xgboost", ["a", "b"], input_cast="float32")
    assert e64.input_cast == "float64" and e32.input_cast == "float32" and e64.hash() != e32.hash()
    for e in (e64, e32):
        e.tau = 0.0
    p64, p32 = ExactPredictor(e64), ExactPredictor(e32)
    j = int(p32.grid.active[0]); th = float(p32.grid.thresholds[j][0])
    z = X[:1].copy(); z[0, j] = np.nextafter(th, -np.inf)       # float32 rounds it up onto the threshold
    assert np.float32(z[0, j]) == np.float32(th)
    assert np.allclose(p32.score(z), native_scores(m, "xgboost", z), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        p64.grid.cells(z)
    assert np.allclose(p32.score(X), native_scores(m, "xgboost", X), atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        export_model(m, "lightgbm", ["a", "b"], input_cast="float32")
