"""Other model families on the frozen design: settings, box table, grid rows and consistency check."""
import numpy as np
import pytest

from scax.experiments import families as fam
from scax.experiments import shift as sh
from scax.experiments.model_ablations import fit_variant
from small_models import small_data

FROZEN = {"gamma": 2, "max_bin": 32, "max_depth": 4, "min_child_weight": 50, "n_estimators": 100}
TARGETS = {"alpha_0": 0.3, "alpha_0_grid": [0.2, 0.3], "beta": 0.3, "beta_grid": [0.2, 0.3], "alpha_1": 0.7,
           "alpha_sel": {"negative_outcome": 0.4, "enrichment": 0.8}, "delta_1": 0.05, "delta_2": 0.05}


def test_family_settings():
    names = ["a", "b", "c", "d"]
    lgb = fam.family_params(FROZEN, names, "lightgbm")
    rf = fam.family_params(FROZEN, names, "sklearn_rf")
    assert lgb["n_estimators"] == 100 and lgb["max_depth"] == 4 and lgb["num_leaves"] <= 31
    assert rf["n_estimators"] == 50 and rf["max_depth"] == 4 and rf["min_samples_leaf"] == 50
    with pytest.raises(ValueError):
        fam.family_params(FROZEN, names, "catboost")


@pytest.mark.parametrize("family", ["lightgbm", "sklearn_rf"])
def test_family_run(family):
    X, y = small_data(5, n=5000)
    part = np.random.default_rng(105).integers(0, 4, len(y))        # 0 train, 1 calibration 1, 2 calibration 2, 3 test
    tr = np.flatnonzero(part == 0)
    params = {"lightgbm": {"n_estimators": 8, "num_leaves": 6, "max_depth": 3, "min_child_samples": 5},
              "sklearn_rf": {"n_estimators": 6, "max_depth": 3}}[family]
    _, _, pred = fit_variant(family, X[tr], y[tr], params, 0, ["a", "b", "c", "d"])
    sm = sh.ShiftModel("toy", pred, M=3, order_seed=0, node_budget=200000, lam=0.5, alpha={0: 0.3, 1: 0.7}, delta1=0.05)
    sm.set_prior(X[tr[:60]])
    sm.set_calibration(X[part == 1], y[part == 1])
    c2, te = np.flatnonzero(part == 2)[:120], np.flatnonzero(part == 3)[:150]
    run = fam.FamilyRun(sm, TARGETS, X[c2], y[c2], X[te], y[te])
    rows, main = run.grid_rows()
    assert len(rows) == 2 * 2 * 2 and sum(r["frozen_targets"] for r in rows) == 2
    assert main is not None and len(main) == len(te)
    for b in np.unique(main[main >= 0]):                        # released boxes meet their target and passed the check
        assert run.U[b] <= (0.7 if run.cls[b] == 1 else 0.3)
        assert sm.verdicts[(run.keys[b], int(run.cls[b]))] == "sound"
    looser = [r for r in rows if r["alpha_0"] == 0.3 and r["kind"] == "negative outcome"]
    stricter = [r for r in rows if r["alpha_0"] == 0.2 and r["kind"] == "negative outcome"]
    assert looser[0]["test_accepted"] >= stricter[0]["test_accepted"]
    cons = run.consistency(main, sm.g.cells(X[part == 3]), y[part == 3])
    assert len(cons) == len(np.unique(main[main >= 0]))
