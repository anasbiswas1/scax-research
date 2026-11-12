"""Subgroup analysis: group coding, group counts and bounds, release with group bounds, intervals."""
import numpy as np
import pytest

from scax.certify.certificate import upper_bounds
from scax.certify.counting import box_counts
from scax.certify.decision import release
from scax.experiments import subgroups as sg


def test_income_cuts_on_brfss_shares():
    counts = {1: 9811, 2: 11783, 3: 15994, 4: 20135, 5: 25883, 6: 36470, 7: 43219, 8: 90385}
    v = np.concatenate([np.full(n, c) for c, n in counts.items()])
    cuts = sg.income_cuts(v)
    assert cuts == (5, 7)
    assert sg.income_labels(cuts) == ["Income under $35,000", "Income $35,000 to $74,999", "Income $75,000 or more"]


def test_group_coding():
    names = ["Sex", "Age", "Income"]
    X = np.array([[0, 1, 1], [1, 5, 6], [0, 6, 8], [1, 9, 5], [0, 10, 7], [1, 13, 8]], dtype=float)
    g = sg.groups_for("brfss2015_heart", X, names, cuts=(5, 7))
    assert g["sex"][0].tolist() == [0, 1, 0, 1, 0, 1]
    assert g["age"][0].tolist() == [0, 0, 1, 1, 2, 2]
    assert g["income"][0].tolist() == [0, 1, 2, 0, 1, 2]
    Xc = np.array([[40 * 365.25, 1], [50 * 365.25, 2], [64.9 * 365.25, 2]])
    gc = sg.groups_for("cardio70k", Xc, ["age", "gender"])
    assert gc["sex"][0].tolist() == [0, 1, 1] and gc["age"][0].tolist() == [0, 1, 1] and "income" not in gc
    kept = sg.drop_empty(gc, gc)
    assert kept["age"][1] == ["Age 18 to 44", "Age 45 to 64"]


def _boxes(rng, d=3, K=6, n_boxes=40):
    keys = []
    for _ in range(n_boxes):
        feats = sorted(rng.choice(d, size=int(rng.integers(1, d + 1)), replace=False))
        key = []
        for j in feats:
            a, b = sorted(rng.integers(0, K, size=2))
            key.append((int(j), int(a), int(b)))
        keys.append(tuple(key))
    return keys


def test_group_counts_partition_and_bounds():
    rng = np.random.default_rng(0)
    keys = _boxes(rng)
    cls = rng.integers(0, 2, len(keys))
    C = rng.integers(0, 6, size=(500, 3))
    y = rng.integers(0, 2, 500)
    codes = rng.integers(0, 3, 500)
    n, k = sg.group_counts(keys, cls, C, y, codes, 3)
    n_all, k_all = box_counts(keys, cls, C, y)
    assert np.array_equal(n.sum(axis=0), n_all) and np.array_equal(k.sum(axis=0), k_all)
    log_pi = np.log(np.full(len(keys), 1 / len(keys)))
    U1 = sg.group_bounds(n_all[None, :], k_all[None, :], log_pi, 0.05, 1.0)[0]
    assert np.allclose(U1, upper_bounds(k_all, n_all, np.log(0.05) + log_pi))
    U3 = sg.group_bounds(n, k, log_pi, 0.05, 1 / 3)
    assert U3.shape == (3, len(keys)) and np.all((U3 >= 0) & (U3 <= 1))


def test_release_pairs_matches_frozen_rule_and_group_rule():
    rng = np.random.default_rng(1)
    n_pat, n_box = 60, 25
    pp = np.repeat(np.arange(n_pat), 4)
    pb = rng.integers(0, n_box, len(pp))
    size = rng.integers(1, 5, n_box); cls = rng.integers(0, 2, n_box)
    n_B = rng.integers(10, 500, n_box); U = rng.random(n_box)
    alpha = {0: 0.3, 1: 0.6}
    assert np.array_equal(sg.release_pairs(pp, pb, U[pb], n_pat, size, cls, n_B, alpha),
                          release(pp, pb, n_pat, size, cls, n_B, U, alpha))
    U_g = rng.random((2, n_box)); codes = rng.integers(-1, 2, n_pat)
    rel = sg.release_pairs(pp, pb, sg.pair_bound(U_g, codes, pp, pb), n_pat, size, cls, n_B, alpha)
    for i in np.flatnonzero(rel >= 0):
        b = rel[i]
        assert codes[i] >= 0 and U_g[codes[i], b] <= alpha[int(cls[b])]
    assert np.all(rel[codes == -1] == -1)
    # strict: every group bound of the patient must meet the target
    U_h = rng.random((3, n_box)); codes_h = rng.integers(0, 3, n_pat)
    strict = np.maximum(sg.pair_bound(U_g, codes, pp, pb), sg.pair_bound(U_h, codes_h, pp, pb))
    rs = sg.release_pairs(pp, pb, strict, n_pat, size, cls, n_B, alpha)
    for i in np.flatnonzero(rs >= 0):
        b = rs[i]
        assert max(U_g[codes[i], b], U_h[codes_h[i], b]) <= alpha[int(cls[b])]


def test_needs_check_and_policy():
    cls = np.array([0, 1, 0, 1])
    U_g = np.array([[0.04, 0.9, 0.2, 0.75], [0.3, 0.6, 0.01, 0.95]])
    idx = sg.needs_check([U_g], cls, {0: 0.05, 1: 0.7}, np.array([True, False, False, False]))
    assert idx.tolist() == [1, 2]
    rel = np.array([0, 0, -1, 2, 2, 2]); yhat = np.array([0, 0, 0, 0, 0, 0]); y = np.array([0, 1, 0, 0, 0, 0])
    codes = np.array([0, 0, 0, 1, 1, 1])
    pol = sg.group_policy(rel, yhat, y, codes, 2, 0, 0.5, 0.05, 2)
    assert pol[0]["cal2_accepted"] == 2 and pol[0]["cal2_errors"] == 1 and pol[0]["policy_level"] == pytest.approx(0.0125)
    rows = sg.group_metrics(rel, yhat, y, codes, ["a", "b"], 0)
    assert rows[0]["group"] == "All" and rows[0]["accepted"] == 5 and rows[1]["errors"] == 1


def test_newcombe_published_example():
    d, lo, hi = sg.newcombe_difference(56, 70, 48, 80)          # Newcombe (1998), Table II, example (a)
    assert d == pytest.approx(0.2) and lo == pytest.approx(0.0524, abs=5e-4) and hi == pytest.approx(0.3339, abs=5e-4)
