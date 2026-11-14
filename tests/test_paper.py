import math

import numpy as np
import pandas as pd
import pytest

from scax.experiments import paper


def test_categories():
    cat = paper.categories([0, 0, 1, 1, 0, 1], [True, True, True, True, False, False], [0, 1, 1, 0, 1, 0])
    assert cat.tolist() == [1, 2, 3, 4, 0, 0]


def test_counts_from_and_inconsistent():
    assert paper.counts_from(10, 4, 1, 3, 2).tolist() == [3, 3, 1, 1, 2]
    with pytest.raises(ValueError):
        paper.counts_from(5, 4, 1, 3, 2)


def test_patient_and_count_bootstrap_agree():
    rng = np.random.default_rng(0)
    n = 2000
    pred = rng.integers(0, 2, n)
    rel = rng.random(n) < 0.6
    y = (rng.random(n) < np.where(pred == 1, 0.7, 0.05)).astype(int)
    cat = paper.categories(pred, rel, y)
    obs = np.bincount(cat, minlength=5)
    B = 4000
    s1 = paper.summarise(paper.bootstrap_patients(cat, B, seed=1), obs)
    s2 = paper.summarise(paper.bootstrap_counts(obs, B, seed=2), obs)
    assert paper.same_distribution(s1, s2, B)
    for k in ("share_0", "error_1"):
        assert s1[f"{k}_lo"] < s1[k] < s1[f"{k}_hi"]
        assert abs(s1[f"{k}_lo"] - s2[f"{k}_lo"]) < 0.01


def test_same_distribution_detects_a_difference():
    a = paper.summarise(paper.bootstrap_counts([500, 200, 10, 100, 40], 3000, seed=1), np.array([500, 200, 10, 100, 40]))
    b = paper.summarise(paper.bootstrap_counts([500, 200, 40, 100, 40], 3000, seed=2), np.array([500, 200, 40, 100, 40]))
    assert not paper.same_distribution(a, b, 3000)


def test_no_release_gives_undefined_error():
    obs = np.array([50, 0, 0, 0, 0])
    s = paper.summarise(paper.bootstrap_counts(obs, 200, seed=0), obs)
    assert s["share_0"] == 0 and math.isnan(s["error_0"]) and math.isnan(s["error_0_lo"])
    assert s["error_0_undefined_share"] == 1.0


def test_strata_keep_their_sizes():
    C = paper.bootstrap_counts([np.array([10, 5, 1, 3, 1]), np.array([30, 0, 0, 8, 2])], 500, seed=3)
    assert (C.sum(axis=1) == 60).all()


def test_label_seed_is_stable():
    assert paper.label_seed("x") == paper.label_seed("x") != paper.label_seed("y")


def test_md_table_and_claims():
    t = paper.md_table(pd.DataFrame({"a": [0.5, float("nan")], "b": ["x|y", "z"], "c": [True, False]}), {"a": paper.pct})
    lines = t.splitlines()
    assert lines[0] == "| a | b | c |" and lines[2] == "| 50.0% | x/y | yes |" and lines[3] == "| n/a | z | no |"
    cl = paper.Claims()
    cl.add("A1", "Section", "text", ["f.csv"], True)
    with pytest.raises(ValueError):
        cl.add("A1", "Section", "text", ["f.csv"], True)
    assert "1 of 1 checks hold" in cl.markdown("T", "intro")
    assert paper.pct_ci(0.586, 0.573, 0.6) == "58.6% (57.3 to 60.0)"
