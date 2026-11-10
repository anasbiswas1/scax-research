import numpy as np

from scax.certify.counting import _popcount, _popcount_table, box_counts


def test_bitset_counts_match_row_by_row():
    rng = np.random.default_rng(0)
    C = np.column_stack([rng.integers(0, k + 1, 5003) for k in (1, 2, 5, 12, 3)])
    y = rng.integers(0, 2, 5003)
    keys, cls = [], []
    for _ in range(300):
        feats = sorted(rng.choice(5, size=int(rng.integers(0, 4)), replace=False))
        key = []
        for j in feats:
            K = (1, 2, 5, 12, 3)[j]
            a = int(rng.integers(0, K + 1)); b = int(rng.integers(a, K + 1))
            key.append((int(j), a, b))
        keys.append(tuple(key)); cls.append(int(rng.integers(0, 2)))
    n, k = box_counts(keys, cls, C, y, batch=7)
    for i, key in enumerate(keys):
        m = np.ones(len(C), dtype=bool)
        for j, a, b in key:
            m &= (C[:, j] >= a) & (C[:, j] <= b)
        assert n[i] == m.sum()
        assert k[i] == ((y[m] != cls[i]).sum())
    n2, k2 = box_counts(keys, cls, C)
    assert np.array_equal(n, n2) and k2 is None


def test_popcount_paths_agree():
    rng = np.random.default_rng(1)
    w = rng.integers(0, 2**63, size=(50, 37), dtype=np.uint64) | np.uint64(1 << 63)
    ref = np.array([sum(bin(int(v)).count("1") for v in row) for row in w])
    assert np.array_equal(_popcount_table(w), ref)
    assert np.array_equal(_popcount(w), ref)
