import pytest

from scax.explain.candidates import CandidateGenerator
from scax.models.predictor import ExactPredictor
from small_models import small_classifier


def test_cache_refuses_other_classifier(tmp_path):
    m, ens, pred, X, y = small_classifier("xgboost")
    gen = CandidateGenerator(pred, M=2, seed=0)
    C = pred.grid.cells(X[:5])
    for i in range(5):
        gen.candidates(C[i])
    path = tmp_path / "cache.pkl"
    gen.save(path)
    same = CandidateGenerator(pred, M=2, seed=0)
    assert same.load(path) and len(same.cache) == len(gen.cache)
    ens.tau = ens.tau + 0.25
    other = CandidateGenerator(ExactPredictor(ens), M=2, seed=0)
    with pytest.raises(ValueError):
        other.load(path)
    with pytest.raises(ValueError):
        CandidateGenerator(pred, M=3, seed=0).load(path)
