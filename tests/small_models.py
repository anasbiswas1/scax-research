"""Small ensembles on small integer grids, shared by the soundness tests."""
import numpy as np

from scax.models.export import export_model
from scax.models.predictor import ExactPredictor, native_scores
from scax.models.train import fit


def small_data(seed=0, n=4000):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 3, n), rng.integers(0, 5, n), rng.integers(0, 4, n)]).astype(float)
    logit = -1.0 + 1.2 * X[:, 0] + 0.6 * X[:, 1] - 0.5 * X[:, 2] + 0.7 * (X[:, 3] >= 2) + 0.8 * X[:, 0] * (X[:, 2] <= 1)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


SMALL_PARAMS = {
    "xgboost": {"n_estimators": 8, "max_depth": 3},
    "lightgbm": {"n_estimators": 8, "num_leaves": 6, "max_depth": 3, "min_child_samples": 5},
    "sklearn_rf": {"n_estimators": 6, "max_depth": 3},
}


def small_classifier(family, seed=0):
    X, y = small_data(seed)
    m = fit(family, X, y, SMALL_PARAMS[family], seed)
    ens = export_model(m, family, ["a", "b", "c", "d"])
    s = native_scores(m, family, X)
    ens.tau = float(np.median(s))
    return m, ens, ExactPredictor(ens), X, y


def random_boxes(grid, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        box = []
        for k in grid.K:
            k = int(k)
            a = int(rng.integers(0, k + 1)); b = int(rng.integers(a, k + 1))
            if rng.random() < 0.3:
                a, b = 0, k
            box.append((a, b))
        out.append(tuple(box))
    return out
