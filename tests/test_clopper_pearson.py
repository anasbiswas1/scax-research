
import numpy as np
import pytest
from scipy.stats import beta, binom
from statsmodels.stats.proportion import proportion_confint

from scax.certify.clopper_pearson import cp_lower, cp_upper


@pytest.mark.parametrize("k,n,gamma", [(0, 50, 0.05), (3, 100, 0.025), (15, 500, 0.01), (150, 5000, 0.05)])
def test_upper_matches_statsmodels_two_sided(k, n, gamma):
    # a one-sided limit at level gamma equals the two-sided Clopper-Pearson limit at level 2 gamma
    _, hi = proportion_confint(k, n, alpha=2 * gamma, method="beta")
    assert abs(cp_upper(k, n, gamma) - hi) < 1e-10


def test_upper_definition():
    k, n, gamma = 7, 300, 0.01
    u = cp_upper(k, n, gamma)
    assert abs(binom.cdf(k, n, u) - gamma) < 1e-9


def test_tiny_level_does_not_round_to_one():
    gamma = 1e-18
    assert beta.ppf(1 - gamma, 1, 500) == 1.0          # the naive quantile breaks
    assert 0 < cp_upper(0, 500, gamma) < 1              # the inverse survival function does not


def test_boundaries():
    assert cp_upper(5, 5, 0.05) == 1.0
    assert cp_upper(0, 0, 0.05) == 1.0
    assert cp_lower(0, 10, 0.05) == 0.0
    with pytest.raises(ValueError):
        cp_upper(6, 5, 0.05)
    with pytest.raises(ValueError):
        cp_upper(1, 5, 0.0)


def test_plan_worked_example():
    # plan Section 5.8: three-feature binary box, d = 21, delta_1 = 0.025
    pi = 1 / 22 / 1330 / 8
    g = 0.025 * pi
    assert abs(cp_upper(15, 500, g) - 0.0891) < 5e-4
    assert abs(cp_upper(150, 5000, g) - 0.0444) < 5e-4
    assert cp_upper(0, 312, g) > 0.05 >= cp_upper(0, 313, g)


def test_vectorised():
    u = cp_upper(np.array([0, 1, 2]), np.array([10, 10, 10]), 0.05)
    assert u.shape == (3,) and np.all(np.diff(u) > 0)
