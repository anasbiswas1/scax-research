"""Model-factor ablations: variant list, monotone signs, and one end-to-end run per model family."""
import numpy as np

from scax.experiments.model_ablations import ModelVariantRun, exported_vs_native, fit_variant, monotone_signs, variants
from small_models import SMALL_PARAMS, small_data

NAMES = ["HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke", "Diabetes", "PhysActivity", "Fruits", "Veggies",
         "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth", "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age",
         "Education", "Income"]


def test_monotone_signs_and_variant_list():
    s = dict(zip(NAMES, monotone_signs(NAMES)))
    assert s["Age"] == 1 and s["Income"] == -1 and s["Sex"] == 0 and s["Fruits"] == 0
    frozen = {"gamma": 2, "max_bin": 32, "max_depth": 4, "min_child_weight": 50, "n_estimators": 100}
    v = variants(frozen, NAMES)
    assert len(v) == 11
    levels = [lvl for _, lvl, _, _ in v]
    assert "depth 4" not in levels and "100 trees" not in levels and "max_bin 32" not in levels
    for factor, level, family, p in v:
        if family == "xgboost" and factor != "tree depth":
            assert p["max_depth"] == 4
        if factor == "monotone constraints":
            assert p["monotone_constraints"].count(",") == len(NAMES) - 1


def test_end_to_end_run_each_family():
    X, y = small_data(0, n=3000)
    rng = np.random.default_rng(0)
    code = rng.integers(0, 5, len(y))                 # 0 train, 1 dev_cal, 2 cal1, 3 cal2, 4 test
    tr = np.flatnonzero(code <= 1); te = np.flatnonzero(code == 4)
    for family in ("xgboost", "lightgbm", "sklearn_rf"):
        model, ens, pred = fit_variant(family, X[tr], y[tr], SMALL_PARAMS[family], 0, ["a", "b", "c", "d"])
        assert exported_vs_native(model, family, pred, X[te]) == 0
        run = ModelVariantRun(X, y, code, prior_rows=tr[:40], test_rows=te[:40], M=3, order_seed=0, node_budget=200000,
                              lam=0.5, alpha={0: 0.5, 1: 0.9}, delta1=0.05)
        rows, info, yhat = run.run(pred, log=lambda *_: None)
        assert [r["kind"] for r in rows] == [0, 1] and info["second_check_failures"] == 0
        assert sum(r["accepted"] for r in rows) > 0
