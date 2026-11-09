import numpy as np

from scax.certify.prior import OccamPrior
from scax.experiments.development import log_prior, prior_counts, smallest_at_or_above
from scax.explain.candidates import Candidate


def test_prior_counts_respect_M_and_multiplicity():
    a = Candidate(((0, 1, 1),), 0, 1, 0, "axp"); b = Candidate(((1, 0, 0),), 0, 1, 3, "axp")
    counts = prior_counts([[a, b], [a], [b]], M=2)
    assert counts == {((0, 1, 1),): 2}
    counts = prior_counts([[a, b], [a], [b]], M=10)
    assert counts == {((0, 1, 1),): 2, ((1, 0, 0),): 2}


def test_log_prior_mixture_mass():
    occ = OccamPrior([1, 2])
    keys = [((0, 1, 1),), ((1, 0, 0),), ((0, 0, 0), (1, 2, 2))]
    lp = log_prior(keys, occ, {keys[0]: 3, keys[1]: 1}, 0.5)
    assert np.all(np.exp(lp) <= 1) and np.exp(lp).sum() <= 1 + 1e-12
    assert np.allclose(log_prior(keys, occ, {}, 1.0), [occ.log_pi(k) for k in keys])


def test_grid_choice():
    assert smallest_at_or_above(0.041, [0.05, 0.075, 0.10]) == 0.05
    assert smallest_at_or_above(0.08, [0.05, 0.075, 0.10]) == 0.10
    assert smallest_at_or_above(0.3, [0.05, 0.075, 0.10]) == 0.10
