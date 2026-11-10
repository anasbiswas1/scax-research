"""SGR, Learn-then-Test and conformal baselines keep their own guarantees on simulated data."""
import math

import numpy as np

from scax.baselines.selective import conformal_quantile, conformal_singletons, ltt, sgr


def _true_risk(theta):
    # conf ~ U(0, 1); P(error | conf) = 0.4 * (1 - conf); selective risk of conf >= theta in closed form
    if theta >= 1:
        return 0.0
    t = max(theta, 0.0)
    return 0.4 * (1 - t) / 2


def test_sgr_and_ltt_risk_control():
    rng = np.random.default_rng(0)
    target, delta, reps = 0.08, 0.1, 300
    viol = {"sgr": 0, "ltt": 0}
    for _ in range(reps):
        conf = rng.random(3000)
        err = (rng.random(3000) < 0.4 * (1 - conf)).astype(int)
        for name, fn in (("sgr", sgr), ("ltt", ltt)):
            res = fn(conf, err, target, delta)
            if math.isfinite(res["theta"]) and _true_risk(res["theta"]) > target:
                viol[name] += 1
    bound = delta + 3 * math.sqrt(delta * (1 - delta) / reps)
    assert viol["sgr"] / reps <= bound and viol["ltt"] / reps <= bound


def test_unreachable_target_releases_nothing():
    conf = np.linspace(0, 1, 500)
    err = np.ones(500, dtype=int)
    assert sgr(conf, err, 0.1, 0.05)["theta"] == math.inf
    assert ltt(conf, err, 0.1, 0.05)["theta"] == math.inf


def test_conformal_coverage():
    rng = np.random.default_rng(1)
    eps, reps, cover, cover1 = 0.1, 200, [], []
    for _ in range(reps):
        p = rng.random(2001)
        y = (rng.random(2001) < p).astype(int)
        p_cal, y_cal, p_te, y_te = p[:2000], y[:2000], p[2000:], y[2000:]
        s_cal = np.where(y_cal == 1, 1 - p_cal, p_cal)
        q = conformal_quantile(s_cal, eps)
        s_te = 1 - p_te[0] if y_te[0] == 1 else p_te[0]
        cover.append(s_te <= q)
        single, _ = conformal_singletons(p_cal, y_cal, p_te, eps, mondrian=True)
        if y_te[0] == 1:
            q1 = conformal_quantile(s_cal[y_cal == 1], eps)
            cover1.append(1 - p_te[0] <= q1)
    assert np.mean(cover) >= 1 - eps - 3 * math.sqrt(eps * (1 - eps) / reps)
    assert np.mean(cover1) >= 1 - eps - 4 * math.sqrt(eps * (1 - eps) / max(1, len(cover1)))


def test_singletons_are_consistent_with_sets():
    rng = np.random.default_rng(2)
    p_cal = rng.random(500); y_cal = (rng.random(500) < p_cal).astype(int)
    p_te = rng.random(300)
    single, q = conformal_singletons(p_cal, y_cal, p_te, 0.2)
    in0 = p_te <= q["q0"]; in1 = 1 - p_te <= q["q1"]
    assert np.array_equal(single == 0, in0 & ~in1) and np.array_equal(single == 1, in1 & ~in0)
