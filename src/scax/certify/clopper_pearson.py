
"""One-sided Clopper-Pearson limits (Eq. 5 of the plan).

The upper limit uses the inverse survival function of the Beta distribution, not the quantile at
1 - gamma: for gamma below about 1e-16, 1 - gamma rounds to exactly 1 and the quantile returns 1.
"""
import numpy as np
from scipy.stats import beta


def cp_upper(k, n, gamma):
    """Smallest p such that P(Bin(n, p) <= k) <= gamma; equals 1 when k == n or n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma <= 0) or np.any(gamma >= 1):
        raise ValueError("gamma must lie in (0, 1)")
    if np.any(k < 0) or np.any(k > n):
        raise ValueError("need 0 <= k <= n")
    k, n, gamma = np.broadcast_arrays(k, n, gamma)
    out = np.ones(k.shape, dtype=float)
    ok = (n > 0) & (k < n)
    if np.any(ok):
        out[ok] = beta.isf(gamma[ok], k[ok] + 1.0, n[ok] - k[ok])
    return out if out.ndim else float(out)


def cp_lower(k, n, gamma):
    """Largest p such that P(Bin(n, p) >= k) <= gamma; equals 0 when k == 0 or n == 0."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if np.any(gamma <= 0) or np.any(gamma >= 1):
        raise ValueError("gamma must lie in (0, 1)")
    k, n, gamma = np.broadcast_arrays(k, n, gamma)
    out = np.zeros(k.shape, dtype=float)
    ok = (n > 0) & (k > 0)
    if np.any(ok):
        out[ok] = beta.ppf(gamma[ok], k[ok], n[ok] - k[ok] + 1.0)
    return out if out.ndim else float(out)
