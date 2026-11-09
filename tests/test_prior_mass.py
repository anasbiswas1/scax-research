"""pi_Occ sums to exactly 1 over the whole family when every feature has a threshold."""
import itertools
from math import comb

import numpy as np

from scax.certify.prior import MixturePrior, OccamPrior


def all_keys(K):
    per = []
    for j, k in enumerate(K):
        per.append([None] + [(j, a, b) for a in range(k + 1) for b in range(a, k + 1) if (a, b) != (0, k)])
    for combo in itertools.product(*per):
        yield tuple(c for c in combo if c is not None)


def test_occam_sums_to_one():
    K = [1, 2, 3]
    pr = OccamPrior(K)
    keys = list(all_keys(K))
    assert len(keys) == np.prod([comb(k + 2, 2) for k in K])
    assert abs(sum(pr.pi(k) for k in keys) - 1.0) < 1e-12


def test_mixture_at_most_one():
    K = [1, 2, 3]
    occ = OccamPrior(K)
    keys = list(all_keys(K))
    counts = {keys[3]: 5, keys[10]: 2}
    mix = MixturePrior(occ, 0.5, counts)
    assert sum(mix.pi(k) for k in keys) <= 1.0 + 1e-12


def test_worked_example_value():
    # d = 21 binary features, three-feature box: (1/22) * (1/1330) * (1/8)
    pr = OccamPrior([1] * 21)
    key = ((0, 1, 1), (1, 1, 1), (2, 0, 0))
    assert abs(pr.pi(key) - 1 / 22 / 1330 / 8) < 1e-18
