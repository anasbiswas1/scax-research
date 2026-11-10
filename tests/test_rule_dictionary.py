"""Rule dictionary: rule-space counts equal direct counts; the outer grid box contains the rule and is tight."""
import math

import numpy as np

from scax.baselines import rule_dictionary as rd
from scax.certify.counting import box_counts
from small_models import small_classifier


def _random_rules(rng, X, n):
    rules = []
    for _ in range(n):
        items = []
        for j in rng.choice(X.shape[1], size=int(rng.integers(1, 4)), replace=False):
            v = np.unique(X[:, j])
            t = float(rng.choice(v)) + float(rng.choice([-0.5, 0.0, 0.5, 0.25]))
            items.append((f"f{j}", str(rng.choice(["<=", ">"])), t))
        r = rd.parse_conditions(items, [f"f{j}" for j in range(X.shape[1])])
        if r is not None:
            rules.append(r)
    return rules


def test_rule_space_counts_match_direct():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, k, 3000) for k in (2, 3, 5, 13)]).astype(float)
    y = rng.integers(0, 2, 3000)
    rules = _random_rules(rng, X, 120)
    space = rd.RuleSpace(rules, X.shape[1])
    keys = [space.key(r) for r in rules]
    cls = rng.integers(0, 2, len(rules))
    n, k = box_counts(keys, cls, space.cells(X), y)
    for i, r in enumerate(rules):
        m = rd.contains(r, X)
        assert n[i] == m.sum() and k[i] == (y[m] != cls[i]).sum()


def test_outer_box_contains_rule_and_is_tight():
    _, ens, pred, X, _ = small_classifier("xgboost")
    g = pred.grid
    rng = np.random.default_rng(1)
    names = list(ens.feature_names)
    for _ in range(200):
        items = []
        for j in rng.choice(g.d, size=int(rng.integers(1, 3)), replace=False):
            lo_hi = rng.uniform(-1, 6, 2)
            items.append((names[j], ">", float(min(lo_hi))))
            items.append((names[j], "<=", float(max(lo_hi))))
        r = rd.parse_conditions(items, names)
        if r is None:
            continue
        box = rd.outer_box(r, g)
        # every point of the region lies in the box
        Z = rng.uniform(-2, 7, size=(4000, g.d))
        inside = rd.contains(r, Z)
        assert g.contains(box, g.cells(Z[inside])).all()
        # both end cells of every constrained feature are reached by points of the region
        for j, (lo, hi) in r.conds.items():
            if g.K[j] == 0:
                continue
            a, b = box[j]
            ends = []
            for v in (np.nextafter(np.float32(lo), np.float32(np.inf)) if lo != -math.inf else -1e6,
                      hi if hi != math.inf else 1e6):
                z = np.zeros((1, g.d)); z[0, j] = float(v)
                ends.append(int(g.cells(z)[0, j]))
            assert ends[0] == a and ends[1] == b
