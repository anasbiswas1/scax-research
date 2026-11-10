"""Released-claim metrics and the paired tests of notebook 11."""
import numpy as np
import pandas as pd

from scax.certify.clopper_pearson import cp_lower
from scax.experiments.baselines import paired_tests, released_metrics


def test_released_metrics_by_hand():
    rel = np.array([0, 0, 1, 2, -1, 2])
    yhat = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 1, 0, 1, 0, 0])
    size = np.array([2, 3, 4]); n_cal = np.array([100, 50, 80])
    n_test = np.array([1000, 10, 400]); k_test = np.array([200, 0, 100])
    alpha = {0: 0.05, 1: 0.70}
    m = released_metrics(rel, None, yhat, y, size, n_cal, n_test, k_test, alpha, len(rel))
    assert m[0]["accepted"] == 3 and m[0]["errors"] == 1 and m[1]["accepted"] == 2 and m[1]["errors"] == 1
    assert cp_lower(200, 1000, 0.05 / 2) > 0.05                     # box 0 refuted, box 1 not
    assert np.isclose(m[0]["refuted_share"], 2 / 3) and m[0]["refuted_boxes"] == 1
    assert m[1]["refuted_share"] == 0.0 and np.isclose(m[0]["mean_size"], (2 + 2 + 3) / 3)


def test_paired_tests_holm():
    rng = np.random.default_rng(0)
    rows = []
    for r in range(20):
        base = rng.normal(0.6, 0.01)
        for m, shift in (("SCAX", 0.0), ("A", -0.05), ("B", -0.0001), ("C", 0.0)):
            rows.append({"task": "t", "kind": 0, "redraw": r, "method": m, "coverage": base + shift + rng.normal(0, 0.001)})
    out = paired_tests(pd.DataFrame(rows), "coverage")
    out = out.set_index("baseline")
    assert out.loc["A", "p_holm"] < 1e-3 and out.loc["A", "median_difference"] > 0
    assert (out.p_holm >= out.p).all() and out.p_holm.max() <= 1.0
