"""Release rule (Eq. 8), policy certification (Eq. 9) and the overlap example."""
import numpy as np

from scax.certify.decision import certify_policy, release


def test_lexicographic_release():
    # patient 0: boxes 0 (size 2) and 1 (size 1, fails), 2 (size 1, passes) -> box 2
    # patient 1: boxes 3 and 4 both size 2 and pass; box 4 has more support -> box 4
    # patient 2: only a failing box -> abstain
    pair_patient = np.array([0, 0, 0, 1, 1, 2]); pair_box = np.array([0, 1, 2, 3, 4, 5])
    size = np.array([2, 1, 1, 2, 2, 1]); cls = np.array([0, 0, 0, 1, 1, 0])
    n_B = np.array([100, 500, 300, 200, 400, 50]); U = np.array([0.04, 0.2, 0.05, 0.6, 0.65, 0.3])
    out = release(pair_patient, pair_box, 3, size, cls, n_B, U, {0: 0.05, 1: 0.70})
    assert out.tolist() == [2, 4, -1]


def test_policy_certification():
    assert certify_policy(30, 2000, 0.05, 0.025)["certified"]
    assert not certify_policy(90, 2000, 0.05, 0.025)["certified"]
    assert not certify_policy(0, 0, 0.05, 0.025)["certified"]


def test_overlap_union_exceeds_each_box():
    # two boxes share 95 correct points and each adds 5 different errors: 5% each, union 10/105
    err = np.r_[np.zeros(95), np.ones(5), np.ones(5)]
    in1 = np.r_[np.ones(95), np.ones(5), np.zeros(5)].astype(bool)
    in2 = np.r_[np.ones(95), np.zeros(5), np.ones(5)].astype(bool)
    assert err[in1].mean() == 0.05 and err[in2].mean() == 0.05
    assert abs(err[in1 | in2].mean() - 10 / 105) < 1e-12
