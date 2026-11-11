"""Model-factor ablations (plan Section 7.4): family, tree depth, number of trees, threshold bins and
monotone constraints, one factor at a time.

Each variant changes one setting of the frozen model and is trained on the same training portion
with the same seed; its threshold is chosen as for the frozen model (Youden on 5-fold out-of-fold
scores). Every variant, and the frozen model as reference, then runs the same pipeline on the same
patients: prior from the same training patients (mixture with the frozen lambda), candidate sets with
the frozen M, order seed and node budget for the same explained test patients, certificates on the
full calibration split 1, the second exact check on releasable boxes, and release at the frozen
targets. The policy test on calibration split 2 is not run for the variants: it would need candidate
sets for thousands more patients per variant.
"""
from __future__ import annotations

import math
import time

import numpy as np
from sklearn.metrics import roc_auc_score

from ..certify.certificate import upper_bounds
from ..certify.counting import box_counts
from ..certify.decision import release
from ..certify.prior import OccamPrior
from ..data import splits
from ..explain.candidates import CandidateGenerator
from ..explain.verify import verify_boxes
from ..models import train
from ..models.export import export_model
from ..models.predictor import ExactPredictor, native_scores
from .baselines import KINDS, released_metrics
from .development import log_prior, prior_counts

RISK_UP = {"HighBP", "HighChol", "BMI", "Smoker", "Stroke", "HeartDiseaseorAttack", "Diabetes", "GenHlth",
           "PhysHlth", "DiffWalk", "Age"}
RISK_DOWN = {"PhysActivity", "Income", "Education"}


def monotone_signs(names):
    """+1 where risk rises with the feature, -1 where it falls, 0 where no direction is imposed."""
    return tuple(1 if n in RISK_UP else -1 if n in RISK_DOWN else 0 for n in names)


def variants(frozen_params, names):
    """(factor, level, family, params) for every one-factor variant of the frozen XGBoost settings."""
    p = dict(frozen_params)
    d, t, b = int(p["max_depth"]), int(p["n_estimators"]), int(p["max_bin"])
    lgb = {"n_estimators": t, "max_depth": d, "num_leaves": 2 ** d - 1, "max_bin": b, "learning_rate": 0.3,
           "min_child_samples": int(p.get("min_child_weight", 20)), "min_split_gain": float(p.get("gamma", 0.0))}
    rf = {"n_estimators": t, "max_depth": d, "min_samples_leaf": int(p.get("min_child_weight", 1))}
    out = [("model family", "LightGBM", "lightgbm", lgb), ("model family", "random forest", "sklearn_rf", rf)]
    out += [("tree depth", f"depth {x}", "xgboost", {**p, "max_depth": x}) for x in (3, 4, 5, 6) if x != d]
    out += [("number of trees", f"{x} trees", "xgboost", {**p, "n_estimators": x}) for x in (25, 50, 100) if x != t]
    out += [("threshold bins", f"max_bin {x}", "xgboost", {**p, "max_bin": x}) for x in (16, 32, 64, 256) if x != b]
    signs = monotone_signs(names)
    out.append(("monotone constraints", "clinically signed", "xgboost",
                {**p, "monotone_constraints": "(" + ",".join(str(s) for s in signs) + ")"}))
    return out


def fit_variant(family, X, y, params, seed, names, input_cast=None):
    oof = train.oof_scores(family, X, y, params, seed)
    tau = train.youden_threshold(oof, y)
    model = train.fit(family, X, y, params, seed)
    ens = export_model(model, family, names, input_cast)
    ens.tau = tau
    return model, ens, ExactPredictor(ens)


def exported_vs_native(model, family, pred, X):
    """Number of rows whose exported class differs from the native model's class at the same threshold."""
    s = native_scores(model, family, X)
    return int((pred.predict(X) != (s > pred.ens.tau).astype(int)).sum())


def exported_vs_native_detail(model, family, pred, X) -> dict:
    """Disagreements between the exported class and the native class at the same threshold, split by the
    distance of the exact score from the threshold.

    XGBoost sums leaf values in float32, so a row whose exact score lies within that rounding error of
    the threshold can be classified either way by the library; the exported exact sum decides those rows.
    The bound is 2 (T + 1) eps32 max(1, S), with S the largest possible |score| plus |tau|. A disagreement
    beyond the bound is an export error and stops the pipeline (gate G2).
    """
    X = np.asarray(X, dtype=float)
    s = native_scores(model, family, X)
    d = np.flatnonzero(pred.predict(X) != (s > pred.ens.tau).astype(int))
    total = sum(max(abs(v) for v in li) for li in pred.leaf_int) / float(1 << pred.E) + abs(float(pred.ens.tau or 0))
    bound = 2.0 * (pred.ens.n_trees + 1) * float(np.finfo(np.float32).eps) * max(1.0, total)
    gap = np.abs(np.asarray(pred.margin_to_tau(X[d]), dtype=float)) if d.size else np.zeros(0)
    return {"disagreements": int(d.size), "within_rounding": int((gap <= bound).sum()),
            "beyond_rounding": int((gap > bound).sum()), "rounding_bound": float(bound),
            "max_gap": float(gap.max()) if d.size else 0.0}


class ModelVariantRun:
    def __init__(self, Xa, y, code, prior_rows, test_rows, M, order_seed, node_budget, lam, alpha, delta1):
        self.X, self.y = Xa, np.asarray(y)
        self.prior_rows, self.test_rows = np.asarray(prior_rows), np.asarray(test_rows)
        self.cal1 = splits.indices(code, "cal1")
        self.test_all = splits.indices(code, "test")
        self.M, self.order_seed, self.node_budget = M, order_seed, node_budget
        self.lam, self.alpha, self.delta1 = lam, dict(alpha), delta1

    def _candidates(self, gen, C, yhat, cache_path, save_every, log, what):
        out, t0 = [], time.time()
        for i in range(len(C)):
            out.append(gen.candidates(C[i], int(yhat[i])))
            if cache_path is not None and (i + 1) % save_every == 0:
                gen.save(cache_path)
                log(f"    {what}: {i + 1}/{len(C)} patients, {time.time() - t0:.0f}s")
        if cache_path is not None:
            gen.save(cache_path)
        return out

    def run(self, pred, cache_path=None, preload=None, save_every=200, log=print):
        g = pred.grid
        gen = CandidateGenerator(pred, M=self.M, seed=self.order_seed, node_budget=self.node_budget)
        if cache_path is not None and cache_path.exists():
            gen.load(cache_path)
        elif preload is not None and preload.exists():
            gen.load(preload)
        before = set(gen.cache)
        Cp = g.cells(self.X[self.prior_rows])
        prior_lists = self._candidates(gen, Cp, pred.predict_cells(Cp), cache_path, save_every, log, "prior patients")
        counts = prior_counts(prior_lists, self.M)
        CT = g.cells(self.X[self.test_rows])
        yhatT = pred.predict_cells(CT).astype(int)
        listsT = self._candidates(gen, CT, yhatT, cache_path, save_every, log, "test patients")
        boxes = sorted({(c.key, c.y) for cands in listsT for c in cands if c.order < self.M})
        index = {b: i for i, b in enumerate(boxes)}
        keys = [k for k, _ in boxes]
        cls = np.array([c for _, c in boxes], dtype=int)
        size = np.array([len(k) for k in keys])
        log_pi = log_prior(keys, OccamPrior(g.K), counts, self.lam)
        n, k = box_counts(keys, cls, g.cells(self.X[self.cal1]), self.y[self.cal1])
        U = np.where(np.isfinite(log_pi), upper_bounds(k, n, math.log(self.delta1) + np.where(np.isfinite(log_pi), log_pi, 0.0)), 1.0)
        a_box = np.where(cls == 1, self.alpha[1], self.alpha[0])
        elig = np.flatnonzero(U <= a_box)
        verdicts, _ = verify_boxes(g, [(keys[i], int(cls[i])) for i in elig])
        bad = [i for i in elig if verdicts[(keys[i], int(cls[i]))] != "sound"]
        U[bad] = 1.0
        pp, pb = [], []
        for i, cands in enumerate(listsT):
            for c in cands:
                if c.order < self.M:
                    pp.append(i); pb.append(index[(c.key, c.y)])
        rel = release(np.array(pp), np.array(pb), len(listsT), size, cls, n, U, self.alpha)
        n_test, k_test = box_counts(keys, cls, g.cells(self.X[self.test_all]), self.y[self.test_all])
        yT = self.y[self.test_rows]
        met = released_metrics(rel, cls, yhatT, yT, size, n, n_test, k_test, self.alpha, len(rel))
        new = [gen.seconds[c] for c in gen.seconds if c not in before]
        model = {"auroc_test": float(roc_auc_score(self.y[self.test_all], pred.score(self.X[self.test_all]))),
                 "thresholds": int(sum(len(t) for t in g.thresholds)), "active_features": int(len(g.active)),
                 "prior_boxes": len(counts), "test_boxes": len(keys), "releasable_boxes": int(len(elig)),
                 "second_check_failures": len(bad), "search_unknown": int(gen.oracle.stats.search_unknown),
                 "median_seconds_per_cell": float(np.median(new)) if new else np.nan,
                 "cells_computed": len(new)}
        return [{"kind": kk, **met[kk]} for kk in KINDS], model, yhatT
