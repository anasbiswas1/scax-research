"""Shift experiments: regions, label-shift estimate, transport, re-certification and box diagnostics."""
import numpy as np
import pytest

from scax.experiments import shift as sh
from scax.experiments.model_ablations import fit_variant
from small_models import SMALL_PARAMS, small_data


def test_regions_cover_every_state_once():
    codes = [f for fs in sh.CENSUS_REGIONS.values() for f in fs]
    assert len(codes) == len(set(codes)) == 51 and set(codes) == set(sh.STATE_NAMES)
    r = sh.region_of([36, 6, 48, 17, 66, 72, np.nan, 3])
    assert list(r) == ["Northeast", "West", "South", "Midwest", None, None, None, None]


def test_largest_states_skip_territories_and_break_ties_by_code():
    state = [72] * 9 + [12] * 5 + [6] * 5 + [36] * 4 + [20] * 4 + [53] * 1 + [3] * 7
    assert sh.largest_states(state, 3) == [6, 12, 20]
    assert sh.largest_states(state, 10) == [6, 12, 20, 36, 53]


def test_bbse_recovers_pure_label_shift():
    rng = np.random.default_rng(0)

    def draw(n, prev):
        y = (rng.random(n) < prev).astype(int)
        yhat = np.where(y == 1, rng.random(n) < 0.8, rng.random(n) < 0.1).astype(int)
        return y, yhat

    y_s, yh_s = draw(200000, 0.2)
    _, yh_t = draw(200000, 0.4)
    assert abs(sh.bbse_prevalence(yh_s, y_s, yh_t) - 0.4) < 0.01
    assert abs(sh.bbse_prevalence(yh_s, y_s, yh_s) - y_s.mean()) < 1e-9


def _setup(n=4000, seed=0):
    X, y = small_data(seed, n=n)
    rng = np.random.default_rng(seed + 100)                 # a stream independent of the one that drew X
    part = rng.integers(0, 4, len(y))                       # 0 train, 1 source calibration, 2 and 3 target
    tr, cal = np.flatnonzero(part == 0), np.flatnonzero(part == 1)
    _, _, pred = fit_variant("xgboost", X[tr], y[tr], SMALL_PARAMS["xgboost"], 0, ["a", "b", "c", "d"])
    sm = sh.ShiftModel("toy", pred, M=3, order_seed=0, node_budget=200000, lam=0.5, alpha={0: 0.5, 1: 0.9}, delta1=0.05)
    sm.set_prior(X[tr[:60]])
    sm.set_calibration(X[cal], y[cal])
    return X, y, part, sm


def test_transport_and_recertification():
    X, y, part, sm = _setup()
    tgt = np.flatnonzero(part >= 2)
    exp, rest = tgt[:80], tgt[80:]
    te = sh.TargetEval(sm, X[exp], y[exp], X[rest], y[rest])
    met = te.transport()
    assert set(met) == {0, 1}
    assert sum(met[k]["accepted"] for k in (0, 1)) > 0
    for k in (0, 1):
        assert met[k]["predicted"] == int((te.yhat == k).sum())
        assert 0 <= met[k]["certificate_refuted_share"] <= 1
    rel = te.rel_transport
    for b in np.unique(rel[rel >= 0]):                    # every released box passed its target and the second check
        assert te.U_src[b] <= te.a_box[b]
        assert sm.verdicts[(te.keys[b], int(te.cls[b]))] == "sound"
    rows = te.recertify([100, 300, len(rest)], draws=3, seed=1, min_eval=50)
    assert {r["m"] for r in rows} == {100, 300}               # a size without enough records left over is skipped
    assert len(rows) == 2 * 3 * 2
    again = te.recertify([100], draws=3, seed=1, min_eval=50)
    assert [r["coverage"] for r in again] == [r["coverage"] for r in rows if r["m"] == 100]
    s = sh.summarise_recert(rows)
    assert len(s) == 4 and (s.coverage_q025 <= s.coverage + 1e-12).all() and (s.coverage <= s.coverage_q975 + 1e-12).all()


def test_fixed_model_hash_is_checked():
    X, y, part, sm = _setup()
    sm.pred.ens.tau = sm.pred.ens.tau + 1.0
    with pytest.raises(RuntimeError):
        sm.candidate_sets(X[:5])


def test_domain_classifier_and_box_diagnostics():
    X, y, part, sm = _setup(n=6000)
    tgt = np.flatnonzero(part >= 2)
    i_s, i_t, p_s, p_t, auc = sh.domain_scores(sm_X := X[part == 1], X[tgt], seed=0)
    assert abs(auc - 0.5) < 0.06                               # same distribution
    Xs = X[tgt].copy(); Xs[:, 2] = np.minimum(4, Xs[:, 2] + 2 * (np.random.default_rng(1).random(len(Xs)) < 0.7))
    _, _, _, _, auc_shift = sh.domain_scores(sm_X, Xs, seed=0)
    assert auc_shift > 0.6
    exp, rest = tgt[:80], tgt[80:]
    te = sh.TargetEval(sm, X[exp], y[exp], X[rest], y[rest])
    te.transport()
    i_s, i_t, p_s, p_t, _ = sh.domain_scores(sm_X, X[rest], seed=0)
    ids = np.unique(te.rel_transport[te.rel_transport >= 0])
    df = sh.box_diagnostics(te, ids, sm.g.cells(sm_X[i_s]), te.C_rest[i_t], p_s, p_t, min_records=20)
    assert len(df) == len(ids) and df.released_patients.sum() == int((te.rel_transport >= 0).sum())
    have = df.ba_estimate.notna()
    assert have.any() and (df.ba_estimate[have] >= 1).all()
    assert ((df.adjusted_bound[have] >= df.U_source[have] - 1e-12) | (df.adjusted_bound[have] == 1.0)).all()


def test_target_specific_alpha_and_small_sample_helpers():
    X, y, part, sm = _setup()
    tgt = np.flatnonzero(part >= 2)
    strict = sh.TargetEval(sm, X[tgt[:80]], y[tgt[:80]], X[tgt[80:]], y[tgt[80:]], alpha={0: 0.2, 1: 0.6})
    loose = sh.TargetEval(sm, X[tgt[:80]], y[tgt[:80]], X[tgt[80:]], y[tgt[80:]], alpha={0: 0.6, 1: 0.95})
    a, b = strict.transport(), loose.transport()
    assert sum(b[k]["accepted"] for k in (0, 1)) >= sum(a[k]["accepted"] for k in (0, 1))
    assert sm.alpha == {0: 0.5, 1: 0.9}
    yy = np.r_[np.zeros(200, int), np.ones(100, int)]
    code = sh.split_small(yy, 0)
    assert set(code) == {0, 1, 2} and abs((code == 0).mean() - 0.5) < 0.01 and abs((code == 2).mean() - 0.15) < 0.01
    for c in (0, 1, 2):
        assert abs(yy[code == c].mean() - 1 / 3) < 0.02            # stratified
    assert np.array_equal(code, sh.split_small(yy, 0))
    best, table = sh.choose_settings(X[part == 0], y[part == 0], [{"max_depth": 1, "n_estimators": 5},
                                                                  {"max_depth": 3, "n_estimators": 30}], seed=0)
    assert len(table) == 2 and best == [{"max_depth": 1, "n_estimators": 5},
                                         {"max_depth": 3, "n_estimators": 30}][int(table.oof_auroc.idxmax())]


def test_exported_vs_native_detail():
    from scax.experiments.model_ablations import exported_vs_native_detail
    X, y = small_data(3, n=2000)
    model, _, pred = fit_variant("xgboost", X, y, SMALL_PARAMS["xgboost"], 0, ["a", "b", "c", "d"])
    d = exported_vs_native_detail(model, "xgboost", pred, X)
    assert d["disagreements"] == d["within_rounding"] + d["beyond_rounding"] == 0
    assert 0 < d["rounding_bound"] < 1e-3
