"""Development and freezing on the training portion only (plan Algorithm 1, Sections 9.2 and 9.3).

Gate G1 compares each constrained model with a tuned unconstrained XGBoost on dev-cal (AUROC loss at
most 0.01). Gate G3 runs the whole SCAX pipeline inside development: the model is trained on
dev-train, the prior comes from candidate sets of a dev-train sample, certificates are computed on
one part of dev-cal and releases are evaluated on patients from the other part. Configurations are
tried in the plan's G3 order and the first that meets the utility target is frozen.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..certify.certificate import membership, upper_bounds
from ..certify.clopper_pearson import cp_upper
from ..certify.decision import release
from ..certify.prior import OccamPrior
from ..explain.candidates import CandidateGenerator
from ..explain.verify import drop_unverified, verify_boxes
from ..models.export import export_model
from ..models.predictor import ExactPredictor, native_scores
from ..models import train

TUNED_GRID = [{"max_depth": d, "n_estimators": n, "learning_rate": lr, "min_child_weight": 1.0, "gamma": 0.0, "max_bin": 256}
              for d in (4, 6, 8) for n in (200, 400) for lr in (0.05, 0.1)]

CASCADE = [   # plan G3 order after the prior weight: coarser thresholds, then a shallower model
    ("reg_depth4_bin32", {"n_estimators": 100, "max_depth": 4, "min_child_weight": 50, "gamma": 2, "max_bin": 32}),
    ("reg_depth4_bin16", {"n_estimators": 100, "max_depth": 4, "min_child_weight": 50, "gamma": 2, "max_bin": 16}),
    ("reg_depth3_bin16", {"n_estimators": 100, "max_depth": 3, "min_child_weight": 50, "gamma": 2, "max_bin": 16}),
]
LAMBDAS = (0.5, 0.25)


def tuned_reference(X_fit, y_fit, X_ev, y_ev, seed):
    rows = []
    for p in TUNED_GRID:
        m = train.fit("xgboost", X_fit, y_fit, p, seed)
        rows.append({**p, "auroc": roc_auc_score(y_ev, native_scores(m, "xgboost", X_ev))})
    table = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    return table


def fit_exported(X_fit, y_fit, params, seed, feature_names):
    """Train, choose tau on 5-fold out-of-fold scores, export, return (model, ensemble, predictor)."""
    oof = train.oof_scores("xgboost", X_fit, y_fit, params, seed)
    tau = train.youden_threshold(oof, y_fit)
    model = train.fit("xgboost", X_fit, y_fit, params, seed)
    ens = export_model(model, "xgboost", feature_names)
    ens.tau = tau
    return model, ens, ExactPredictor(ens)


def candidate_sets(gen, C, yhat, log=None, every=500):
    out = []
    t0 = time.time()
    for i in range(len(C)):
        out.append(gen.candidates(C[i], int(yhat[i])))
        if log is not None and (i + 1) % every == 0:
            log(f"    {i + 1}/{len(C)} patients, {time.time() - t0:.0f}s")
    return out


def prior_counts(cand_lists, M):
    """N_tr(B): training patients (with multiplicity) whose candidate set built with the first M orders contains B."""
    counts = {}
    for cands in cand_lists:
        for k in {c.key for c in cands if c.order < M}:
            counts[k] = counts.get(k, 0) + 1
    return counts


def log_prior(keys, occam, counts, lam):
    log_occ = np.array([occam.log_pi(k) for k in keys])
    if lam >= 1.0:
        return log_occ
    total = float(sum(counts.values()))
    n_tr = np.array([counts.get(k, 0) for k in keys], dtype=float)
    pi = lam * np.exp(log_occ) + (1 - lam) * (n_tr / total if total > 0 else 0.0)
    with np.errstate(divide="ignore"):
        return np.log(pi)


def evaluate_releases(eval_cands, y_eval, yhat_eval, grid, C_cert, y_cert, prior_lists, lambdas, M_list,
                      alpha0_list, beta_list, delta1, delta2_each):
    """Release rates and selective errors on development evaluation patients for every setting."""
    occam = OccamPrior(grid.K)
    box_list = sorted({(c.key, c.y) for cands in eval_cands for c in cands})
    index = {b: i for i, b in enumerate(box_list)}
    keys = [k for k, _ in box_list]
    cls = np.array([y for _, y in box_list])
    size = np.array([len(k) for k in keys])
    mem = membership(grid, keys, C_cert)
    n_B = np.rint(mem @ np.ones(C_cert.shape[0])).astype(np.int64)
    pos = np.rint(mem @ (y_cert == 1).astype(float)).astype(np.int64)
    k_B = np.where(cls == 1, n_B - pos, pos)
    pp, pb, po = [], [], []
    for i, cands in enumerate(eval_cands):
        for c in cands:
            pp.append(i); pb.append(index[(c.key, c.y)]); po.append(c.order)
    pp, pb, po = np.array(pp), np.array(pb), np.array(po)
    rows = []
    n_eval = len(eval_cands)
    for M in M_list:
        counts = prior_counts(prior_lists, M)
        sel = po < M
        for lam in lambdas:
            lg = np.log(delta1) + log_prior(keys, occam, counts, lam)
            U = upper_bounds(k_B, n_B, lg)
            for a0 in alpha0_list:
                for beta in beta_list:
                    a1 = round(1 - beta, 10)
                    rel = release(pp[sel], pb[sel], n_eval, size, cls, n_B, U, {0: a0, 1: a1})
                    acc = rel >= 0
                    row = {"M": M, "lambda": lam, "alpha_0": a0, "beta": beta, "alpha_1": a1,
                           "release_rate": float(acc.mean())}
                    for y in (0, 1):
                        ay = acc & (yhat_eval == y)
                        err = int((y_eval[ay] != y).sum()); a = int(ay.sum())
                        row[f"release_rate_class{y}"] = float(ay.mean())
                        row[f"accepted_class{y}"] = a
                        row[f"errors_class{y}"] = err
                        row[f"selective_error_class{y}"] = err / a if a else float("nan")
                        row[f"selective_upper_class{y}"] = float(cp_upper(err, a, delta2_each)) if a else 1.0
                        rb = rel[ay]
                        row[f"released_size_class{y}"] = float(size[rb].mean()) if a else float("nan")
                        row[f"released_support_class{y}"] = float(n_B[rb].mean()) if a else float("nan")
                    rows.append(row)
    return pd.DataFrame(rows), {"boxes": len(box_list), "pairs": int(len(pp))}


def evaluate_config(name, params, X, y, idx_fit, idx_cert, idx_eval, idx_prior, feature_names, seed, M, delta1,
                    delta2_each, alpha0_list, beta_list, lambdas, M_list, log=print):
    t0 = time.time()
    model, ens, pred = fit_exported(X[idx_fit], y[idx_fit], params, seed, feature_names)
    g = pred.grid
    dev_cal = np.r_[idx_cert, idx_eval]
    auroc = roc_auc_score(y[dev_cal], pred.score(X[dev_cal]))
    gen = CandidateGenerator(pred, M=M, seed=seed)
    C_prior = g.cells(X[idx_prior]); C_eval = g.cells(X[idx_eval]); C_cert = g.cells(X[idx_cert])
    log(f"  {name}: candidate sets for {len(idx_prior)} dev-train patients (prior)")
    prior_lists = candidate_sets(gen, C_prior, pred.predict_cells(C_prior), log)
    yhat_eval = pred.predict_cells(C_eval).astype(int)
    log(f"  {name}: candidate sets for {len(idx_eval)} evaluation patients")
    eval_cands = candidate_sets(gen, C_eval, yhat_eval, log)
    boxes = sorted({(c.key, c.y) for v in gen.cache.values() for c in v})
    verdicts, _ = verify_boxes(g, boxes)
    removed = drop_unverified(gen.cache, verdicts)
    prior_lists = [gen.cache[gen.cell_key(C_prior[i])] for i in range(len(C_prior))]
    eval_cands = [gen.cache[gen.cell_key(C_eval[i])] for i in range(len(C_eval))]
    table, info = evaluate_releases(eval_cands, y[idx_eval], yhat_eval, g, C_cert, y[idx_cert], prior_lists,
                                    lambdas, M_list, alpha0_list, beta_list, delta1, delta2_each)
    table.insert(0, "config", name)
    secs = gen.oracle.stats.seconds
    meta = {"config": name, "params": params, "auroc_dev_cal": auroc, "tau": ens.tau, "thresholds": int(g.K.sum()),
            "active_features": int(len(g.active)), "log10_family_size": g.family_size_log10(),
            "distinct_boxes": len(boxes), "unverified_removed": int(removed),
            "median_seconds_per_patient": float(np.median(secs)) if secs else None,
            "p95_seconds_per_patient": float(np.quantile(secs, 0.95)) if secs else None,
            "search_unknown": int(gen.oracle.stats.search_unknown), "seconds": time.time() - t0, **info}
    return table, meta


def smallest_at_or_above(value, grid):
    for g in grid:
        if value <= g:
            return g
    return grid[-1]
