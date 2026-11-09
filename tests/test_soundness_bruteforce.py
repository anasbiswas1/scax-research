"""Every soundness procedure against brute-force enumeration on small ensembles (gate G2)."""
import pytest

from scax.encode.bruteforce import brute_force_sound
from scax.encode.cell_search import CellSearch
from scax.encode.interval_bound import IntervalBound
from scax.encode.real_encoding import Z3RealEncoding
from scax.encode.real_search import RealSearch
from scax.encode.z3_encoding import Z3CellEncoding
from small_models import random_boxes, small_classifier


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "sklearn_rf"])
def test_all_procedures_agree_with_brute_force(family):
    m, ens, pred, X, y = small_classifier(family)
    g = pred.grid
    ib = IntervalBound(pred); cs = CellSearch(ib); rs = RealSearch(ens, g.thresholds)
    zr = Z3RealEncoding(ens, g.thresholds); zc = Z3CellEncoding(pred)
    boxes = random_boxes(g, 150, seed=1)
    n_sound = 0
    for i, box in enumerate(boxes):
        for yb in (0, 1):
            truth = brute_force_sound(g, box, yb)
            assert truth is not None
            n_sound += truth
            assert (cs.check(box, yb) == "sound") == truth
            assert (rs.check(box, yb) == "sound") == truth
            if ib.sound(box, yb):
                assert truth
            if i < 40:
                assert (zr.check(box, yb) == "sound") == truth
                assert (zc.check(box, yb) == "sound") == truth
    assert n_sound > 0
