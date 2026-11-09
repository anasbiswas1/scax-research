"""Validity study on the planted-rule population (plan Sections 6.4, 6.7, 6.8 and 7.3).

For one label setting (rule and noise) the model, the candidate sets of every population cell, the
true risk of every candidate box (Eq. 13) and the training prior are fixed. Each repetition then
re-draws calibration split 1 and split 2 with replacement from the population, with fresh labels,
and recomputes certificates, releases and the policy test by counting only. Per repetition:

  V1   some evaluated box has R_true(B) > U_hat(B)                      (frequency checks delta_1)
  V2   policy certified and R_sel_true > alpha_sel                      (frequency checks delta_2)
  claim violation: some released box has R_true(B*) > alpha_y
  overlap: every released box satisfies its claim but R_sel_true > alpha_sel
  the same family-wise and claim quantities for naive certification (level delta_1, no prior).

Bounds are compared through the binomial CDF (certify/fast.py), so one CDF evaluation per box
serves every prior and every delta_1; U_hat itself is computed only for boxes that pass, for the
tie-break of the release rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..certify.certificate import membership, upper_bounds
from ..certify.clopper_pearson import cp_upper
from ..certify.fast import binom_cdf, passes, violates
from ..data import planted


class Setting:
    """Everything fixed for one label setting before calibration data are drawn."""

    def __init__(self, grid, pred, XP, p, cache, cell_keys, cell_of_row, box_list, prior_logs, alpha, alpha_sel,
                 rule_features=None):
        self.grid = grid
        self.p = np.asarray(p, dtype=float)
        self.N = len(self.p)
        self.cell_of_row = np.asarray(cell_of_row)          # population row -> distinct cell index
        self.n_cells = len(cell_keys)
        self.box_list = box_list                            # list of (key, y)
        self.box_index = {b: i for i, b in enumerate(box_list)}
        self.box_size = np.array([len(k) for k, _ in box_list])
        self.box_class = np.array([y for _, y in box_list])
        self.alpha = alpha
        self.alpha_sel = alpha_sel
        self.box_alpha = np.where(self.box_class == 1, alpha[1], alpha[0])
        C_pop = grid.cells(XP.to_numpy())
        # cell-level aggregates of the population
        self.cell_count = np.bincount(self.cell_of_row, minlength=self.n_cells).astype(float)
        kappa = pred.predict_cells(C_pop).astype(int)
        self.kappa_cell = np.zeros(self.n_cells, dtype=int)
        self.kappa_cell[self.cell_of_row] = kappa
        err_prob = np.where(kappa == 1, 1 - self.p, self.p)          # P(Y != kappa(x) | x)
        self.cell_err = np.bincount(self.cell_of_row, weights=err_prob, minlength=self.n_cells)
        # membership of distinct cells in boxes
        rep_row = np.zeros(self.n_cells, dtype=np.int64)
        rep_row[self.cell_of_row] = np.arange(self.N)
        self.C_cells = C_pop[rep_row]
        self.mem = membership(grid, [k for k, _ in box_list], self.C_cells)
        # true risk per box, Eq. (13)
        pos_cell = np.bincount(self.cell_of_row, weights=self.p, minlength=self.n_cells)
        n_pop = self.mem @ self.cell_count
        pos_pop = self.mem @ pos_cell
        with np.errstate(invalid="ignore", divide="ignore"):
            self.R_true = np.where(self.box_class == 1, (n_pop - pos_pop) / n_pop, pos_pop / n_pop)
        if np.any(n_pop <= 0):
            raise ValueError("every candidate box contains its own population cell, so n_pop > 0")
        # candidate pairs (cell, box, first order)
        pc, pb, po = [], [], []
        for ci, key in enumerate(cell_keys):
            for c in cache[key]:
                pc.append(ci); pb.append(self.box_index[(c.key, c.y)]); po.append(c.order)
        order = np.lexsort((np.array(pb), np.array(pc)))
        self.pair_cell = np.array(pc)[order]; self.pair_box = np.array(pb)[order]; self.pair_order = np.array(po)[order]
        self.prior_logs = prior_logs                        # name -> callable(M) -> log pi per box
        self.rule_features = rule_features
        if rule_features is not None:
            names = grid.ens.feature_names
            prec, rec = [], []
            for key, _ in box_list:
                fs = {names[j] for j, _, _ in key}
                inter = len(fs & rule_features)
                prec.append(inter / len(fs) if fs else 0.0); rec.append(inter / len(rule_features))
            self.box_prec, self.box_rec = np.array(prec), np.array(rec)

    def set_priors(self, occam, lam: float, train_cell_count):
        """pi_Occ and pi_mix(lambda) from the training draw (cell counts of training patients)."""
        log_occ = np.array([occam.log_pi(k) for k, _ in self.box_list])
        tc = np.asarray(train_cell_count, dtype=float)

        def occ(M):
            return log_occ

        def mix(M):
            pm = self.pair_order < M
            n_tr = np.bincount(self.pair_box[pm], weights=tc[self.pair_cell[pm]], minlength=len(self.box_list))
            pi = lam * np.exp(log_occ) + (1 - lam) * n_tr / n_tr.sum()
            with np.errstate(divide="ignore"):
                return np.log(pi)

        self.prior_logs = {"occam": occ, f"mix{lam:g}": mix}
        return list(self.prior_logs)

    def boxes_up_to(self, M):
        """Boxes that appear in some candidate set built with the first M orders."""
        m = np.zeros(len(self.box_list), dtype=bool)
        m[self.pair_box[self.pair_order < M]] = True
        return m


def draw_counts(setting: Setting, n: int, rng):
    """Calibration draw with replacement: per-cell draw counts and per-cell counts of Y = 1."""
    idx = planted.draw_with_replacement(setting.N, n, rng)
    y = planted.draw_labels(setting.p[idx], rng)
    cells = setting.cell_of_row[idx]
    c = np.bincount(cells, minlength=setting.n_cells).astype(float)
    pos = np.bincount(cells, weights=y, minlength=setting.n_cells)
    return c, pos


def run_repetitions(setting: Setting, n1_list, M_list, delta1_list, prior_names, R, n2, delta2, seed, progress=None):
    rng = np.random.default_rng(seed)
    rows = []
    box_sets = {M: setting.boxes_up_to(M) for M in M_list}
    pair_sets = {M: setting.pair_order < M for M in M_list}
    log_pi = {(name, M): setting.prior_logs[name](M) for name in prior_names for M in M_list}
    has_pairs = np.bincount(setting.pair_cell, minlength=setting.n_cells) > 0
    cells_with = np.flatnonzero(has_pairs)
    seg = np.searchsorted(setting.pair_cell, cells_with)            # pairs are sorted by cell
    pair_pos = np.arange(len(setting.pair_cell))
    for n1 in n1_list:
        for r in range(R):
            c1, pos1 = draw_counts(setting, n1, rng)
            c2, pos2 = draw_counts(setting, n2, rng)
            n_B = np.rint(setting.mem @ c1).astype(np.int64)
            pos_B = np.rint(setting.mem @ pos1).astype(np.int64)
            k_B = np.where(setting.box_class == 1, n_B - pos_B, pos_B)
            F_alpha = binom_cdf(k_B, n_B, setting.box_alpha)
            F_true = binom_cdf(k_B, n_B, setting.R_true)
            err2 = np.where(setting.kappa_cell == 1, c2 - pos2, pos2)
            for d1 in delta1_list:
                variants = {name: None for name in prior_names}
                variants["naive"] = None
                for name in variants:
                    for M in M_list:
                        lg = np.full(len(n_B), np.log(d1)) if name == "naive" else log_pi[(name, M)] + np.log(d1)
                        bm, pm = box_sets[M], pair_sets[M]
                        V1 = bool(np.any(violates(F_true, lg) & bm))
                        ok_box = passes(F_alpha, lg) & bm
                        U_box = np.ones(len(n_B))
                        if ok_box.any():
                            U_box[ok_box] = upper_bounds(k_B[ok_box], n_B[ok_box], lg[ok_box])
                        # Eq. (8) as one key: fewest features, then most support, then tightest bound
                        key_box = setting.box_size * 1e7 - n_B + U_box / 2.0
                        kp = np.where(pm & ok_box[setting.pair_box], key_box[setting.pair_box], np.inf)
                        rel = np.full(setting.n_cells, -1, dtype=np.int64)
                        if np.isfinite(kp).any():
                            mn = np.minimum.reduceat(kp, seg)
                            acc_c = np.isfinite(mn)
                            target = np.full(setting.n_cells, np.inf); target[cells_with] = mn
                            hit = np.isfinite(kp) & (kp == target[setting.pair_cell])
                            hp = pair_pos[hit]
                            hc = setting.pair_cell[hp]
                            firsts = np.r_[True, hc[1:] != hc[:-1]]
                            rel[hc[firsts]] = setting.pair_box[hp[firsts]]
                        acc = rel >= 0
                        w = setting.cell_count[acc]
                        n_acc = float(w.sum())
                        R_sel = float(setting.cell_err[acc].sum() / n_acc) if n_acc > 0 else float("nan")
                        rb = rel[acc]
                        claim_ok = setting.R_true[rb] <= setting.box_alpha[rb]
                        claim_viol = bool(np.any(~claim_ok))
                        a = int(c2[acc].sum()); e = int(err2[acc].sum())
                        U_sel = float(cp_upper(e, a, delta2)) if a > 0 else 1.0
                        certified = U_sel <= setting.alpha_sel
                        sel_exceeds = (n_acc > 0) and (R_sel > setting.alpha_sel)
                        row = {"n1": n1, "rep": r, "delta1": d1, "M": M, "prior": name, "V1": V1,
                               "claim_violation": claim_viol, "cert_rate": n_acc / setting.N, "R_sel": R_sel,
                               "claim_precision": float((w * claim_ok).sum() / n_acc) if n_acc > 0 else float("nan"),
                               "policy_accepted": a, "policy_errors": e, "U_sel": U_sel,
                               "policy_certified": bool(certified), "V2": bool(certified and sel_exceeds),
                               "overlap": bool((n_acc > 0) and (not claim_viol) and sel_exceeds),
                               "boxes_evaluated": int(bm.sum()),
                               "released_size": float((w * setting.box_size[rb]).sum() / n_acc) if n_acc > 0 else float("nan"),
                               "released_support": float((w * n_B[rb]).sum() / n_acc) if n_acc > 0 else float("nan")}
                        if setting.rule_features is not None:
                            row["feature_precision"] = float((w * setting.box_prec[rb]).sum() / n_acc) if n_acc > 0 else float("nan")
                            row["feature_recall"] = float((w * setting.box_rec[rb]).sum() / n_acc) if n_acc > 0 else float("nan")
                        rows.append(row)
            if progress is not None:
                progress(n1, r)
    return pd.DataFrame(rows)


def build_setting(XP, rule, eps, p4_b0, model_params, n_train, M, seed, alpha, alpha_sel, lam=0.5,
                  cache_path=None, z3_n=50, log=print):
    """Train the model on a training draw, build candidate sets for every population cell, verify
    every candidate box with the second check (and a Z3 sample), and assemble the Setting."""
    import time
    from pathlib import Path

    from ..certify.prior import OccamPrior
    from ..explain.candidates import CandidateGenerator
    from ..explain.verify import drop_unverified, verify_boxes, z3_sample
    from ..models.export import export_model
    from ..models.predictor import ExactPredictor
    from ..models.train import fit

    X = XP.to_numpy(dtype=float)
    p = planted.probability(rule, XP, eps, p4_b0=p4_b0)
    rng = np.random.default_rng(seed)
    idx_train = planted.draw_with_replacement(len(XP), n_train, rng)
    y_train = planted.draw_labels(p[idx_train], rng)
    model = fit("xgboost", X[idx_train], y_train, model_params, seed)
    ens = export_model(model, "xgboost", list(XP.columns))
    ens.tau = 0.0                                   # predicted probability 0.5: the Bayes rule for P1 to P3
    pred = ExactPredictor(ens)
    grid = pred.grid
    C = grid.cells(X)
    keys_arr, inv = np.unique(C[:, grid.active], axis=0, return_inverse=True)
    inv = inv.ravel()
    cell_keys = [tuple(int(v) for v in row) for row in keys_arr]
    rep = np.zeros(len(cell_keys), dtype=np.int64); rep[inv] = np.arange(len(C))
    gen = CandidateGenerator(pred, M=M, seed=seed)
    if cache_path is not None and gen.load(cache_path):
        log(f"candidate cache loaded: {len(gen.cache)} cells")
    t0 = time.time()
    todo = [ci for ci, k in enumerate(cell_keys) if k not in gen.cache]
    for n_done, ci in enumerate(todo):
        gen.candidates(C[rep[ci]])
        if cache_path is not None and (n_done + 1) % 2000 == 0:
            gen.save(cache_path)
            log(f"  {n_done + 1}/{len(todo)} cells, {time.time() - t0:.0f}s")
    t_gen = time.time() - t0
    boxes = sorted({(c.key, c.y) for v in gen.cache.values() for c in v})
    t0 = time.time()
    verdicts, _ = verify_boxes(grid, boxes)
    t_ver = time.time() - t0
    removed = drop_unverified(gen.cache, verdicts)
    z3_res, _ = z3_sample(grid, [b for b in boxes if verdicts[b] == "sound"], z3_n, seed, timeout_ms=30000,
                           time_budget_s=180)
    if cache_path is not None:
        gen.save(cache_path)
    box_list = sorted({(c.key, c.y) for v in gen.cache.values() for c in v})
    rule_features = planted.RULE_FEATURES[rule]
    st = Setting(grid, pred, XP, p, gen.cache, cell_keys, inv, box_list, None, alpha, alpha_sel, rule_features)
    st.set_priors(OccamPrior(grid.K), lam, np.bincount(inv[idx_train], minlength=len(cell_keys)))
    bayes = (p > 0.5).astype(int)
    kappa = pred.predict_cells(C)
    diag = {"rule": rule, "eps": eps, "cells": len(cell_keys), "active_features": int(len(grid.active)),
            "thresholds": int(grid.K.sum()), "agreement_with_bayes_rule": float((kappa == bayes).mean()),
            "boxes": len(box_list), "pairs": int(len(st.pair_cell)), "unverified_removed": int(removed),
            "z3_sample": z3_res, "seconds_candidates": t_gen, "seconds_second_check": t_ver,
            "median_seconds_per_cell": float(np.median(gen.oracle.stats.seconds)) if gen.oracle.stats.seconds else None,
            "search_unknown": int(gen.oracle.stats.search_unknown), "classifier_sha256": ens.hash()}
    return st, diag
