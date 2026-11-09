"""Exported classifier against the native library, and the exact point evaluator."""
import numpy as np
import pytest

from scax.encode.bruteforce import exact_class_point
from scax.models.export import Ensemble
from scax.models.predictor import ExactPredictor, native_scores
from small_models import small_classifier, small_data


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "sklearn_rf"])
def test_scores_and_decisions(family, tmp_path):
    m, ens, pred, X, y = small_classifier(family)
    Xt, _ = small_data(seed=5, n=3000)
    nat = native_scores(m, family, Xt)
    assert np.max(np.abs(pred.score(Xt) - nat)) < 1e-5
    far = np.abs(nat - ens.tau) > 1e-5
    assert np.array_equal(pred.predict(Xt)[far], (nat[far] > ens.tau).astype(np.int8))
    for i in range(50):
        assert exact_class_point(ens, list(Xt[i])) == pred.predict(Xt[i:i + 1])[0]
    path = tmp_path / "model.json"
    ens.save(path)
    again = Ensemble.load(path)
    assert again.hash() == ens.hash()
    assert np.array_equal(ExactPredictor(again).predict(Xt), pred.predict(Xt))


def test_hash_depends_on_tau():
    m, ens, pred, X, y = small_classifier("xgboost")
    h = ens.hash()
    ens.tau = ens.tau + 0.125
    assert ens.hash() != h and ens.hash(with_tau=False) == ens.hash(with_tau=False)
