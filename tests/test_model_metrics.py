import numpy as np

from scax.metrics.model_metrics import ece_equal_mass, model_metrics


def test_metrics_on_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 50000)
    y = (rng.random(p.size) < p).astype(int)
    m = model_metrics(p, y, (p > 0.5).astype(int))
    assert m["ece_15_equal_mass"] < 0.01
    assert abs(m["calibration_slope"] - 1) < 0.05 and abs(m["calibration_intercept"]) < 0.05
    assert 0.8 < m["auroc"] < 0.9
    assert ece_equal_mass(np.full(1000, 0.9), np.zeros(1000)) > 0.89
