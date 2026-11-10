"""Selection-rule variants, the re-draw split and candidate flags."""
import numpy as np

from scax.certify.decision import release
from scax.experiments.ablations import release_by
from scax.experiments.baselines import candidate_flags, redraw_split
from scax.explain.candidates import Candidate


def _random_problem(rng, n_pat=60, n_box=40, n_pairs=400):
    pp = rng.integers(0, n_pat, n_pairs); pb = rng.integers(0, n_box, n_pairs)
    size = rng.integers(1, 6, n_box); cls = rng.integers(0, 2, n_box)
    n_B = rng.integers(0, 50, n_box); U = rng.random(n_box)
    return pp, pb, n_pat, size, cls, n_B, U, {0: 0.5, 1: 0.7}


def test_succinct_rule_equals_frozen_release():
    rng = np.random.default_rng(0)
    for _ in range(50):
        args = _random_problem(rng)
        assert np.array_equal(release_by(*args, rule="succinct"), release(*args))


def test_support_and_bound_rules_by_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(30):
        pp, pb, n_pat, size, cls, n_B, U, alpha = _random_problem(rng)
        for rule, key in (("support", lambda b: (-n_B[b], size[b], U[b])), ("bound", lambda b: (U[b], size[b], -n_B[b]))):
            rel = release_by(pp, pb, n_pat, size, cls, n_B, U, alpha, rule)
            for i in range(n_pat):
                ok = [b for p, b in zip(pp, pb) if p == i and U[b] <= alpha[int(cls[b])]]
                if not ok:
                    assert rel[i] == -1
                else:
                    assert key(rel[i]) == min(key(b) for b in ok)


def test_redraw_split_keeps_explained_patients_in_split_2():
    keep = np.array([3, 10, 50, 77])
    a = redraw_split(100, 40, keep, 5); b = redraw_split(100, 40, keep, 5); c = redraw_split(100, 40, keep, 6)
    assert a.sum() == 40 and not a[keep].any() and np.array_equal(a, b) and not np.array_equal(a, c)


def test_candidate_flags():
    K = np.array([4, 4, 4])
    cands = [Candidate(((0, 1, 1), (1, 2, 2)), 1, 2, 0, "axp"), Candidate(((0, 0, 2), (1, 2, 2)), 1, 2, 0, "inflated"),
             Candidate(((0, 0, 3), (1, 1, 3)), 1, 2, 0, "inflated"), Candidate(((2, 1, 1),), 1, 1, 1, "axp")]
    flags = candidate_flags(cands, K)
    assert [f[0] for f in flags] == [0, 1, 1, 0]
    assert [f[1] for f in flags] == [False, False, True, True]
