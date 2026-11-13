"""Runtime study: the timed generator gives the same candidates, clocks add up, model size and cache readers."""
import numpy as np
import pytest

from scax.experiments import runtime as rt
from scax.explain.candidates import CandidateGenerator
from scax.models.export import Ensemble, Tree
from small_models import small_classifier


def test_timed_generator_same_answers_and_clocks():
    m, ens, pred, X, y = small_classifier("xgboost")
    C = pred.grid.cells(X[:80]); yh = pred.predict_cells(C)
    plain = CandidateGenerator(pred, M=4, seed=0)
    tg = rt.TimedGenerator(pred, M=4, seed=0)
    rows = tg.time_cells(C, yh)
    distinct = {plain.cell_key(c) for c in C}
    assert len(rows) == len(distinct)
    for c in C:
        k = plain.cell_key(c)
        a = sorted((x.key, x.size, x.order, x.kind) for x in plain.candidates(c))
        b = sorted((x.key, x.size, x.order, x.kind) for x in tg.gen.cache[k])
        assert a == b
    for r in rows:
        assert r["precheck_seconds"] + r["search_seconds"] <= r["seconds"] + 1e-6
        assert r["deletion_seconds"] + r["inflation_seconds"] <= r["seconds"] + 1e-6
        assert r["deletion_precheck_seconds"] + r["deletion_search_seconds"] <= r["deletion_seconds"] + 1e-6
        assert r["checks"] >= r["memo_hits"] + r["precheck_sound"] and r["candidates"] >= 1
    s = rt.phase_summary(rows)
    assert 0 <= s["precheck_share"] <= 1 and 0 <= s["search_share"] <= 1 and s["search_unknown"] == 0
    assert s["precheck_share"] + s["search_share"] + s["other_share"] == pytest.approx(1, abs=1e-6)
    assert len(rt.TimedGenerator(pred, M=4, seed=0).time_cells(C, yh, max_cells=5)) == 5


def test_cell_cap_censors_and_leaves_answers_unchanged():
    m, ens, pred, X, y = small_classifier("xgboost")
    C = pred.grid.cells(X[:40]); yh = pred.predict_cells(C)
    tg = rt.TimedGenerator(pred, M=4, seed=0)
    rows = tg.time_cells(C, yh, cell_cap=1e-5)           # the timer fires inside every cell
    assert rows and all(r["censored"] and r["candidates"] == 0 for r in rows)
    assert not tg.gen.cache
    rows2 = tg.time_cells(C, yh)                         # no cap: the same generator still gives the frozen answers
    plain = CandidateGenerator(pred, M=4, seed=0)
    assert len(rows2) == len({plain.cell_key(c) for c in C}) and not any(r["censored"] for r in rows2)
    for c in C:
        a = sorted((x.key, x.order, x.kind) for x in plain.candidates(c))
        assert a == sorted((x.key, x.order, x.kind) for x in tg.gen.cache[plain.cell_key(c)])


def test_model_size_and_spread():
    leaf = Tree([-1], [0.0], [-1], [-1], [0.5])
    t2 = Tree([0, -1, 1, -1, -1], [0.5, 0.0, 1.5, 0.0, 0.0], [1, -1, 3, -1, -1], [2, -1, 4, -1, -1], [0.0, 0.1, 0.0, -0.2, 0.3])
    ens = Ensemble("xgboost", "lt", "float64", "margin_sum", ["a", "b"], [leaf, t2, leaf])
    s = rt.model_size(ens)
    assert s == {"trees": 3, "trees_with_a_split": 1, "internal_nodes": 2, "leaves": 5, "max_depth_reached": 2,
                 "mean_depth_reached": pytest.approx(2 / 3)}
    sp = rt.spread([1, 2, 3, 4, 100])
    assert sp["median_seconds"] == 3 and sp["max_seconds"] == 100 and sp["cells"] == 5 and sp["censored_cells"] == 0
    assert rt.spread([]) == {"cells": 0}
    sc = rt.spread([1, 9, 9, 9], [False, True, True, True])
    assert sc["censored_cells"] == 3 and sc["median_is_lower_bound"] and sc["p95_is_lower_bound"]


def test_cache_runtime(tmp_path):
    m, ens, pred, X, y = small_classifier("xgboost")
    gen = CandidateGenerator(pred, M=2, seed=0)
    C = pred.grid.cells(X[:30])
    for c in C:
        gen.candidates(c)
    gen.save(tmp_path / "c.pkl")
    r = rt.cache_runtime(tmp_path / "c.pkl")
    assert r["cells"] == len(gen.cache) == r["cells_in_cache"] and r["checks"] == gen.oracle.stats.checks
    keys = [gen.cell_key(c) for c in C[:5]]
    assert rt.cache_runtime(tmp_path / "c.pkl", keys)["cells"] == len(set(keys))
    assert set(rt.hardware()) == {"cpu", "logical_cpus", "python", "machine"}
