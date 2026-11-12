"""Main results for the other model families (plan Section 7.2, E1) on the frozen design.

The frozen design of each task is kept: training portion and seed, the 2,000 prior training patients,
lambda, M, feature-order seed and search budget, the explained subsamples of calibration split 2 and
of the test split, calibration split 1 for the certificates, the target grids and alpha_sel. Only the
model changes: LightGBM and random forest with the settings closest to the frozen XGBoost (as in
notebook 12b; the random forest has the plan's 50 trees), each with a Youden threshold on 5-fold
out-of-fold scores of the training portion. Every box that could be released at the loosest grid
target is re-checked by the second exact check; release, the policy test and the test results follow
notebook 09.
"""
from __future__ import annotations

import numpy as np

from ..certify.counting import box_counts
from ..certify.clopper_pearson import cp_lower, cp_upper
from .main_results import kind_summary, pair_arrays, policy_test, release_patients
from .model_ablations import variants

FAMILY_NAMES = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "sklearn_rf": "random forest"}
KIND_NAMES = {0: "negative outcome", 1: "enrichment"}


def family_params(frozen_params, names, family):
    """Settings of one family on the frozen design (random forest: the plan's 50 trees)."""
    found = [p for _, _, f, p in variants(frozen_params, names) if f == family]
    if len(found) != 1:
        raise ValueError(family)
    p = dict(found[0])
    if family == "sklearn_rf":
        p["n_estimators"] = 50
    return p


class FamilyRun:
    """Box table, certificates and releases of one model on the frozen explained subsamples."""

    def __init__(self, sm, targets, X_cal2, y_cal2, X_test, y_test, log=None):
        self.sm, self.T = sm, targets
        _, yh2, L2 = sm.candidate_sets(X_cal2, log, what="calibration split 2 patients")
        _, yht, Lt = sm.candidate_sets(X_test, log, what="test patients")
        boxes = sorted({(c.key, c.y) for L in (L2, Lt) for cands in L for c in cands if c.order < sm.M})
        index = {b: i for i, b in enumerate(boxes)}
        self.keys = [k for k, _ in boxes]
        self.cls = np.array([c for _, c in boxes], dtype=int)
        self.size = np.array([len(k) for k in self.keys], dtype=int)
        self.n, self.k = box_counts(self.keys, self.cls, sm.C_cal, sm.y_cal)
        self.U = sm.bounds(self.k, self.n, sm.log_prior(self.keys))
        loosest = {0: max(targets["alpha_0_grid"]), 1: round(1 - min(targets["beta_grid"]), 10)}
        elig = np.flatnonzero(self.U <= np.where(self.cls == 1, loosest[1], loosest[0]))
        ok = sm.second_check(self.keys, self.cls, elig) if elig.size else np.zeros(0, dtype=bool)
        self.U[elig[~ok]] = 1.0                          # a box the second check cannot confirm is never released
        self.eligible, self.unverified = int(elig.size), int((~ok).sum())
        self.parts = {"cal2": (pair_arrays(L2, index, sm.M), yh2.astype(int), np.asarray(y_cal2).astype(int)),
                      "test": (pair_arrays(Lt, index, sm.M), yht.astype(int), np.asarray(y_test).astype(int))}

    def release(self, name, a0, a1):
        (pp, pb), yhat, _ = self.parts[name]
        return release_patients(pp, pb, len(yhat), self.size, self.cls, self.n, self.U, a0, a1)

    def grid_rows(self):
        """Rows in the layout of results/09_main.csv for every target pair on the frozen grids."""
        T = self.T
        alpha_sel = {0: T["alpha_sel"]["negative_outcome"], 1: T["alpha_sel"]["enrichment"]}
        rows, main_test = [], None
        for a0 in T["alpha_0_grid"]:
            for beta in T["beta_grid"]:
                a1 = round(1 - beta, 10)
                is_main = (a0 == T["alpha_0"]) and (beta == T["beta"])
                r2, rt = self.release("cal2", a0, a1), self.release("test", a0, a1)
                _, yh2, y2 = self.parts["cal2"]
                pol = policy_test(r2, yh2, y2, alpha_sel, T["delta_2"])
                _, yht, yt = self.parts["test"]
                ks = kind_summary(rt, yht, yt)
                if is_main:
                    main_test = rt
                for k in (0, 1):
                    a = (rt >= 0) & (yht == k)
                    rb = rt[a]
                    rows.append({"kind": KIND_NAMES[k], "alpha_0": a0, "beta": beta, "target": a0 if k == 0 else a1,
                                 "frozen_targets": is_main, "cal2_accepted": pol[k]["accepted"],
                                 "cal2_errors": pol[k]["errors"], "U_sel": pol[k]["U_sel"],
                                 "alpha_sel": pol[k]["alpha_sel"],
                                 "policy_certified": pol[k]["certified"] if is_main else None,
                                 "test_patients": len(yht), "test_certified_share": ks[k]["share_of_patients"],
                                 "test_certified_share_of_predicted_class": float(a.sum() / max(1, (yht == k).sum())),
                                 "test_accepted": ks[k]["accepted"], "test_errors": ks[k]["errors"],
                                 "test_error_rate": ks[k]["error_rate"], "test_error_lower_95": ks[k]["error_lower_95"],
                                 "test_error_upper_95": ks[k]["error_upper_95"],
                                 "mean_size": float(self.size[rb].mean()) if a.any() else None,
                                 "median_calibration_support": float(np.median(self.n[rb])) if a.any() else None})
        return rows, main_test

    def consistency(self, rel, C_test_all, y_test_all):
        """Test records and errors inside every box released at the frozen targets (notebook 09 check)."""
        ids = np.unique(rel[rel >= 0])
        if not ids.size:
            return []
        n_t, k_t = box_counts([self.keys[i] for i in ids], self.cls[ids], C_test_all, np.asarray(y_test_all).astype(int))
        K = len(ids)
        out = []
        for j, i in enumerate(ids):
            target = self.T["alpha_1"] if self.cls[i] == 1 else self.T["alpha_0"]
            low = float(cp_lower(int(k_t[j]), int(n_t[j]), 0.05 / K)) if n_t[j] else 0.0
            out.append({"box_class": int(self.cls[i]), "size": int(self.size[i]), "calibration_n": int(self.n[i]),
                        "calibration_k": int(self.k[i]), "U": float(self.U[i]), "target": target,
                        "test_n": int(n_t[j]), "test_k": int(k_t[j]),
                        "test_error": float(k_t[j] / n_t[j]) if n_t[j] else None,
                        "test_upper_95": float(cp_upper(int(k_t[j]), int(n_t[j]), 0.05)) if n_t[j] else 1.0,
                        "test_lower_corrected": low, "estimate_above_target": bool(low > target)})
        return out
