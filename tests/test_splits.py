import numpy as np

from scax.data.splits import indices, make_splits, summary


def test_shares_disjoint_and_deterministic():
    code = make_splits(253680, 42)
    assert np.array_equal(code, make_splits(253680, 42))
    s = summary(code)
    assert s["dev_train"] + s["dev_cal"] == round(0.4 * 253680)
    assert s["cal1"] == round(0.3 * 253680) and s["cal2"] == round(0.15 * 253680)
    assert sum(s[k] for k in ("dev_train", "dev_cal", "cal1", "cal2", "test")) == 253680
    tr = indices(code, "train")
    assert len(np.intersect1d(tr, indices(code, "cal1"))) == 0
