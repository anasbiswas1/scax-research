"""Simulated family-wise violation frequency stays below delta with the prior, not without it."""
import numpy as np

from scax.certify.certificate import upper_bounds
from scax.certify.fast import binom_cdf, passes, violates


def test_family_wise_rate():
    rng = np.random.default_rng(0)
    n_boxes, reps, delta = 200, 400, 0.05
    R = rng.uniform(0.02, 0.2, n_boxes)
    n = rng.integers(50, 400, n_boxes)
    log_pi = np.full(n_boxes, -np.log(n_boxes))
    v_prior = v_naive = 0
    for _ in range(reps):
        k = rng.binomial(n, R)
        v_prior += np.any(upper_bounds(k, n, np.log(delta) + log_pi) < R)
        v_naive += np.any(upper_bounds(k, n, np.full(n_boxes, np.log(delta))) < R)
    assert v_prior / reps <= delta
    assert v_naive / reps > 0.5


def test_fast_comparisons_match_bounds():
    rng = np.random.default_rng(1)
    n = rng.integers(1, 5000, 20000)
    k = np.minimum(rng.binomial(n, rng.uniform(0, 0.3, n.size)), n)
    lg = np.log(10.0 ** rng.uniform(-50, -1, n.size))
    a = rng.uniform(0.01, 0.5, n.size)
    U = upper_bounds(k, n, lg)
    assert np.array_equal(U <= a, passes(binom_cdf(k, n, a), lg))
    r = rng.uniform(0, 0.4, n.size)
    assert np.array_equal(U < r, violates(binom_cdf(k, n, r), lg))
    assert np.all(upper_bounds([3], [3], [-1.0]) == 1.0) and np.all(upper_bounds([0], [0], [-1.0]) == 1.0)
