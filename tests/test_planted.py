
import numpy as np
import pandas as pd

from scax.data.planted import (Box, draw_labels, draw_with_replacement, make_population, p4_intercept,
                               probability, rule_mask, true_box_risk, true_selective_risk)


def toy_population(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "HighBP": rng.integers(0, 2, n), "HighChol": rng.integers(0, 2, n), "BMI": rng.integers(15, 50, n),
        "Smoker": rng.integers(0, 2, n), "Age": rng.integers(1, 14, n), "GenHlth": rng.integers(1, 6, n),
        "DiffWalk": rng.integers(0, 2, n), "Stroke": rng.integers(0, 2, n), "Diabetes": rng.integers(0, 3, n),
        "Sex": rng.integers(0, 2, n)}).astype(float)


def test_rule_region_risk_equals_noise():
    X = toy_population()
    for eps in (0.0, 0.05, 0.2):
        p = probability("P1", X, eps)
        inside = Box({"HighBP": (1, 2), "Age": (10, np.inf)})
        assert abs(true_box_risk(inside, 1, X, p) - eps) < 1e-12
        assert np.array_equal(inside.contains(X), rule_mask("P1", X))


def test_whole_space_risk_is_prevalence():
    X = toy_population()
    p = probability("P2", X, 0.1)
    assert abs(true_box_risk(Box({}), 0, X, p) - p.mean()) < 1e-12
    assert abs(true_box_risk(Box({}), 1, X, p) - (1 - p.mean())) < 1e-12


def test_oracle_matches_simulation():
    X = toy_population()
    p = probability("P3", X, 0.05)
    box = Box({"GenHlth": (4, np.inf)})
    rng = np.random.default_rng(1)
    m = box.contains(X)
    sims = [np.mean(draw_labels(p[m], rng) != 1) for _ in range(400)]
    assert abs(np.mean(sims) - true_box_risk(box, 1, X, p)) < 0.01


def test_selective_risk_and_overlap_example():
    # two boxes with 5% error each whose union has 10/105 errors (plan Section 5.6)
    p = np.zeros(105)
    p[:10] = 1.0                      # ten points whose recorded outcome is 1
    pred = np.zeros(105, dtype=int)   # every released prediction is 0
    accepted = np.ones(105, dtype=bool)
    assert abs(true_selective_risk(accepted, pred, p) - 10 / 105) < 1e-12


def test_p4_intercept_hits_target():
    X = toy_population()
    b0 = p4_intercept(X, 0.10)
    assert abs(probability("P4", X, p4_b0=b0).mean() - 0.10) < 1e-6


def test_population_and_draws_are_reproducible():
    a = make_population(1000, 200, 42)
    b = make_population(1000, 200, 42)
    assert np.array_equal(a, b) and len(np.unique(a)) == 200
    r1 = draw_with_replacement(200, 50, np.random.default_rng(3))
    r2 = draw_with_replacement(200, 50, np.random.default_rng(3))
    assert np.array_equal(r1, r2)
