"""Shift experiments (plan Sections 5.10, 6.6 and 7.5; RQ4).

One fixed model per source. The model, its prior (candidate sets of source training patients) and
its box certificates (source calibration split 1) are built once, and every target domain is then
evaluated with the identical model, whose hash is checked before each evaluation.

Transport: the candidate sets of explained target patients are certified with the source
certificates and released by the frozen rule. Claims are checked on target labels: the explained
patients' own outcomes, and box risks on the other target records.

Re-certification: a labelled target sample of m records replaces the source calibration split; the
model, the candidate family and the prior stay fixed (Section 5.10). Each size is drawn repeatedly;
claims are checked on the target records outside the sample.

Diagnostics: a domain classifier (source calibration records against target records, cross-fitted)
gives a within-box AUROC, and quantiles of the estimated density ratio inside a box give an estimate
of b/a in Eq. (12). Black-box shift estimation (Lipton et al. 2018) gives the target prevalence that
pure label shift would imply.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..certify.certificate import upper_bounds
from ..certify.clopper_pearson import cp_lower, cp_upper
from ..certify.counting import box_counts
from ..certify.decision import release
from ..certify.prior import OccamPrior
from ..explain.candidates import CandidateGenerator
from ..explain.verify import verify_boxes
from ..models import train
from .baselines import KINDS, released_metrics
from .development import log_prior, prior_counts

# US Census Bureau regions by FIPS state code (50 states and the District of Columbia)
CENSUS_REGIONS = {
    "Northeast": (9, 23, 25, 33, 34, 36, 42, 44, 50),
    "Midwest": (17, 18, 19, 20, 26, 27, 29, 31, 38, 39, 46, 55),
    "South": (1, 5, 10, 11, 12, 13, 21, 22, 24, 28, 37, 40, 45, 47, 48, 51, 54),
    "West": (2, 4, 6, 8, 15, 16, 30, 32, 35, 41, 49, 53, 56),
}
STATE_NAMES = {
    1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California", 8: "Colorado", 9: "Connecticut",
    10: "Delaware", 11: "District of Columbia", 12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho",
    17: "Illinois", 18: "Indiana", 19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine",
    24: "Maryland", 25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi", 29: "Missouri",
    30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New Hampshire", 34: "New Jersey", 35: "New Mexico",
    36: "New York", 37: "North Carolina", 38: "North Dakota", 39: "Ohio", 40: "Oklahoma", 41: "Oregon",
    42: "Pennsylvania", 44: "Rhode Island", 45: "South Carolina", 46: "South Dakota", 47: "Tennessee",
    48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia", 53: "Washington", 54: "West Virginia",
    55: "Wisconsin", 56: "Wyoming",
}
_REGION_OF = {f: r for r, codes in CENSUS_REGIONS.items() for f in codes}


def region_of(state) -> np.ndarray:
    """Census region per record from the FIPS state code; None for territories and unknown codes."""
    s = np.asarray(state, dtype=float)
    return np.array([_REGION_OF.get(int(v)) if np.isfinite(v) else None for v in s], dtype=object)


def largest_states(state, k: int = 5) -> list:
    """The k states (50 states and DC) with the most records; ties go to the lower FIPS code."""
    s = np.asarray(state, dtype=float)
    s = s[np.isfinite(s)].astype(int)
    s = s[np.isin(s, list(STATE_NAMES))]
    codes, counts = np.unique(s, return_counts=True)
    order = np.lexsort((codes, -counts))
    return [int(codes[i]) for i in order[:k]]


def bbse_prevalence(yhat_src, y_src, yhat_tgt) -> float:
    """Target prevalence implied by pure label shift (black-box shift estimation, Lipton et al. 2018).

    Solves C w = mu, with C[i, j] = P_source(yhat = i, y = j) and mu[i] = P_target(yhat = i); the
    estimate is w_1 p_1 / (w_0 p_0 + w_1 p_1) with the source class shares p_j.
    """
    yhat_src, y_src, yhat_tgt = (np.asarray(a).astype(int) for a in (yhat_src, y_src, yhat_tgt))
    C = np.array([[np.mean((yhat_src == i) & (y_src == j)) for j in (0, 1)] for i in (0, 1)])
    mu = np.array([np.mean(yhat_tgt == i) for i in (0, 1)])
    w = np.clip(np.linalg.solve(C, mu), 0.0, None)
    q = w * C.sum(axis=0)
    return float(q[1] / q.sum()) if q.sum() > 0 else float("nan")


def model_summary(pred, X, y) -> dict:
    """Records, prevalence, AUROC, predicted-positive share, sensitivity and specificity of the fixed model."""
    y = np.asarray(y).astype(int)
    yhat = pred.predict(np.asarray(X, dtype=float)).astype(int)
    both = 0 < y.sum() < len(y)
    return {"records": int(len(y)), "prevalence": float(y.mean()),
            "auroc": float(roc_auc_score(y, pred.score(np.asarray(X, dtype=float)))) if both else float("nan"),
            "predicted_positive": float(yhat.mean()),
            "sensitivity": float(yhat[y == 1].mean()) if (y == 1).any() else float("nan"),
            "specificity": float(1 - yhat[y == 0].mean()) if (y == 0).any() else float("nan")}


class ShiftModel:
    """One fixed model with its prior and its source certificates; candidate sets cached on Drive."""

    def __init__(self, name, pred, M, order_seed, node_budget, lam, alpha, delta1, cache_path=None):
        self.name, self.pred, self.g = name, pred, pred.grid
        self.model_hash = pred.ens.hash()
        self.M, self.lam, self.delta1 = int(M), float(lam), float(delta1)
        self.alpha = {int(k): float(v) for k, v in dict(alpha).items()}
        self.gen = CandidateGenerator(pred, M=self.M, seed=order_seed, node_budget=node_budget)
        self.cache_path = None if cache_path is None else Path(cache_path)
        if self.cache_path is not None and self.cache_path.exists():
            self.gen.load(self.cache_path)
        self.occam = OccamPrior(self.g.K)
        self.counts = None
        self.verdicts = {}
        self.X_cal = self.C_cal = self.y_cal = self.yhat_cal = None

    def check_hash(self):
        if self.pred.ens.hash() != self.model_hash:
            raise RuntimeError(f"{self.name}: the fixed model has changed")

    def candidate_sets(self, X, log=None, every=500, what="patients"):
        """Cells, predicted classes and candidate sets; the cache is saved every `every` new cells."""
        self.check_hash()
        C = self.g.cells(np.asarray(X, dtype=float))
        yhat = self.pred.predict_cells(C).astype(int)
        out, t0, fresh, total_new = [], time.time(), 0, 0
        for i in range(len(C)):
            if self.gen.cell_key(C[i]) not in self.gen.cache:
                fresh += 1; total_new += 1
            out.append(self.gen.candidates(C[i], int(yhat[i])))
            if self.cache_path is not None and fresh >= every:
                self.gen.save(self.cache_path); fresh = 0
                if log is not None:
                    log(f"    {self.name}, {what}: {i + 1}/{len(C)}, {time.time() - t0:.0f}s")
        if self.cache_path is not None and fresh:
            self.gen.save(self.cache_path)
        if log is not None:
            log(f"    {self.name}, {what}: {len(C)} done ({total_new} new cells), {time.time() - t0:.0f}s")
        return C, yhat, out

    def set_prior(self, X_prior, log=None):
        _, _, lists = self.candidate_sets(X_prior, log, what="prior patients")
        self.counts = prior_counts(lists, self.M)

    def set_calibration(self, X_cal, y_cal):
        self.X_cal = np.asarray(X_cal, dtype=float)          # kept for the domain classifier
        self.C_cal = self.g.cells(self.X_cal)
        self.y_cal = np.asarray(y_cal).astype(int)
        self.yhat_cal = self.pred.predict_cells(self.C_cal).astype(int)

    def log_prior(self, keys):
        if self.counts is None:
            raise RuntimeError("set_prior first")
        return log_prior(keys, self.occam, self.counts, self.lam)

    def bounds(self, k, n, log_pi):
        fin = np.isfinite(log_pi)
        return np.where(fin, upper_bounds(k, n, math.log(self.delta1) + np.where(fin, log_pi, 0.0)), 1.0)

    def second_check(self, keys, cls, idx):
        """True where the box passed the second exact check (run once per box and class)."""
        todo = [(keys[i], int(cls[i])) for i in idx if (keys[i], int(cls[i])) not in self.verdicts]
        if todo:
            v, _ = verify_boxes(self.g, todo)
            self.verdicts.update(v)
        return np.array([self.verdicts[(keys[i], int(cls[i]))] == "sound" for i in idx], dtype=bool)


class TargetEval:
    """One target domain under one fixed model: explained patients, the other target records."""

    def __init__(self, sm: ShiftModel, X_exp, y_exp, X_rest, y_rest, log=None, alpha=None):
        if sm.C_cal is None:
            raise RuntimeError("set_calibration first")
        self.sm = sm
        _, self.yhat, lists = sm.candidate_sets(X_exp, log, what="explained patients")
        self.y = np.asarray(y_exp).astype(int)
        self.n_patients = len(lists)
        boxes = sorted({(c.key, c.y) for cands in lists for c in cands if c.order < sm.M})
        index = {b: i for i, b in enumerate(boxes)}
        self.keys = [k for k, _ in boxes]
        self.cls = np.array([c for _, c in boxes], dtype=int)
        self.size = np.array([len(k) for k in self.keys], dtype=int)
        pp, pb = [], []
        for i, cands in enumerate(lists):
            for c in cands:
                if c.order < sm.M:
                    pp.append(i); pb.append(index[(c.key, c.y)])
        self.pp, self.pb = np.array(pp, dtype=np.int64), np.array(pb, dtype=np.int64)
        self.log_pi = sm.log_prior(self.keys)
        self.n_src, self.k_src = box_counts(self.keys, self.cls, sm.C_cal, sm.y_cal)
        self.U_src = sm.bounds(self.k_src, self.n_src, self.log_pi)
        self.C_rest = sm.g.cells(np.asarray(X_rest, dtype=float))
        self.y_rest = np.asarray(y_rest).astype(int)
        self.n_rest, self.k_rest = box_counts(self.keys, self.cls, self.C_rest, self.y_rest)
        # targets: the model's own, or another target pair for the same candidate sets
        self.alpha = sm.alpha if alpha is None else {int(k): float(v) for k, v in dict(alpha).items()}
        self.a_box = np.where(self.cls == 1, self.alpha[1], self.alpha[0])
        self.rel_transport = None

    def _release(self, U, support):
        U = U.copy()
        elig = np.flatnonzero(U <= self.a_box)
        if elig.size:
            ok = self.sm.second_check(self.keys, self.cls, elig)
            U[elig[~ok]] = 1.0
        return release(self.pp, self.pb, self.n_patients, self.size, self.cls, support, U, self.alpha)

    def _metrics(self, rel, support, n_eval, k_eval):
        met = released_metrics(rel, self.cls, self.yhat, self.y, self.size, support, n_eval, k_eval, self.alpha,
                               self.n_patients)
        for k in KINDS:
            r = met[k]
            r["predicted"] = int((self.yhat == k).sum())
            r["error_upper_95"] = float(cp_upper(r["errors"], r["accepted"], 0.05)) if r["accepted"] else 1.0
        return met

    def transport(self):
        """Release with the source certificates; claims and certificates checked on target labels."""
        self.sm.check_hash()
        rel = self._release(self.U_src, self.n_src)
        met = self._metrics(rel, self.n_src, self.n_rest, self.k_rest)
        for k in KINDS:
            a = (rel >= 0) & (self.yhat == k)
            rb = rel[a]
            ids = np.unique(rb)
            refuted = set()
            if ids.size:
                ok = self.n_rest[ids] > 0
                low = np.zeros(ids.size)
                if ok.any():
                    low[ok] = cp_lower(self.k_rest[ids][ok], self.n_rest[ids][ok], 0.05 / ids.size)
                refuted = set(ids[low > self.U_src[ids]].tolist())
            met[k]["certificate_refuted_share"] = float(np.mean([b in refuted for b in rb])) if rb.size else 0.0
            met[k]["certificate_refuted_boxes"] = len(refuted)
        self.rel_transport = rel
        return met

    def recertify(self, sizes, draws, seed, min_eval):
        """Certificates from labelled target samples of each size (the rest of the target checks claims)."""
        self.sm.check_hash()
        rows, n = [], len(self.y_rest)
        for m in sizes:
            if m + min_eval > n:
                continue
            for r in range(draws):
                rng = np.random.default_rng([int(seed), int(m), int(r)])
                S = rng.choice(n, int(m), replace=False)
                n_q, k_q = box_counts(self.keys, self.cls, self.C_rest[S], self.y_rest[S])
                U_q = self.sm.bounds(k_q, n_q, self.log_pi)
                rel = self._release(U_q, n_q)
                met = self._metrics(rel, n_q, self.n_rest - n_q, self.k_rest - k_q)
                rows += [{"m": int(m), "draw": r, "kind": k, **met[k]} for k in KINDS]
        return rows


def summarise_recert(rows) -> pd.DataFrame:
    """Per size and kind: mean coverage with the 2.5% and 97.5% draw quantiles, errors and refuted claims."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = []
    for (m, k), g in df.groupby(["m", "kind"], sort=True):
        out.append({"m": int(m), "kind": int(k), "draws": int(len(g)), "coverage": float(g.coverage.mean()),
                    "coverage_q025": float(g.coverage.quantile(0.025)), "coverage_q975": float(g.coverage.quantile(0.975)),
                    "error_rate": float(np.nanmean(g.error_rate)) if g.error_rate.notna().any() else float("nan"),
                    "refuted_share": float(g.refuted_share.mean()),
                    "draws_with_refuted_claims": float((g.refuted_boxes > 0).mean()),
                    "above_target_share": float(g.above_target_share.mean()),
                    "mean_size": float(np.nanmean(g.mean_size)) if g.mean_size.notna().any() else float("nan")})
    return pd.DataFrame(out)


def domain_scores(X_src, X_tgt, seed, max_per_domain=50000, folds=2):
    """Cross-fitted probability that a record comes from the target, on subsamples of both domains.

    Returns the subsample indices of each domain, their out-of-fold scores, and the overall AUROC.
    """
    X_src, X_tgt = np.asarray(X_src, dtype=float), np.asarray(X_tgt, dtype=float)
    rng = np.random.default_rng(seed)
    i_s = np.sort(rng.choice(len(X_src), min(max_per_domain, len(X_src)), replace=False))
    i_t = np.sort(rng.choice(len(X_tgt), min(max_per_domain, len(X_tgt)), replace=False))
    X = np.vstack([X_src[i_s], X_tgt[i_t]])
    d = np.r_[np.zeros(len(i_s), dtype=int), np.ones(len(i_t), dtype=int)]
    p = np.zeros(len(d))
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, d):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        clf.fit(X[tr], d[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return i_s, i_t, p[: len(i_s)], p[len(i_s):], float(roc_auc_score(d, p))


def box_diagnostics(te: TargetEval, ids, C_src_sub, C_tgt_sub, p_src, p_tgt, min_records=50, q=0.05, clip=1e-3):
    """Per box: source and target risk, within-box domain AUROC and estimated b/a (Eq. 12).

    b/a is estimated as the ratio of the (1 - q) and q quantiles of the estimated density ratio
    w = p / (1 - p) (times a constant that cancels) over the subsampled records of both domains in the
    box. Target lower limits are Bonferroni-corrected over the boxes analysed.
    """
    g = te.sm.g
    released = np.bincount(te.rel_transport[te.rel_transport >= 0], minlength=len(te.keys)) \
        if te.rel_transport is not None else np.zeros(len(te.keys), dtype=int)
    rows = []
    for i in np.asarray(ids, dtype=int):
        ms, mt = g.contains(te.keys[i], C_src_sub), g.contains(te.keys[i], C_tgt_sub)
        ns, nt = int(ms.sum()), int(mt.sum())
        row = {"box": int(i), "box_class": int(te.cls[i]), "size": int(te.size[i]), "released_patients": int(released[i]),
               "n_source": int(te.n_src[i]), "k_source": int(te.k_src[i]), "n_target": int(te.n_rest[i]),
               "k_target": int(te.k_rest[i]), "U_source": float(te.U_src[i]), "alpha": float(te.a_box[i]),
               "domain_records_source": ns, "domain_records_target": nt, "domain_auroc": np.nan, "ba_estimate": np.nan}
        if ns >= min_records and nt >= min_records:
            s = np.r_[p_src[ms], p_tgt[mt]]
            row["domain_auroc"] = float(roc_auc_score(np.r_[np.zeros(ns), np.ones(nt)], s))
            pc = np.clip(s, clip, 1 - clip)
            lw = np.log(pc) - np.log1p(-pc)
            lo, hi = np.quantile(lw, [q, 1 - q])
            row["ba_estimate"] = float(np.exp(hi - lo))
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    with np.errstate(invalid="ignore", divide="ignore"):
        df["risk_source"] = np.where(df.n_source > 0, df.k_source / df.n_source.clip(lower=1), np.nan)
        df["risk_target"] = np.where(df.n_target > 0, df.k_target / df.n_target.clip(lower=1), np.nan)
    low = np.zeros(len(df))
    ok = df.n_target.to_numpy() > 0
    if ok.any():
        low[ok] = cp_lower(df.k_target.to_numpy()[ok], df.n_target.to_numpy()[ok], 0.05 / len(df))
    df["target_lower_95"] = low
    df["certificate_refuted"] = low > df.U_source.to_numpy()
    df["claim_refuted"] = low > df.alpha.to_numpy()
    df["above_certificate"] = df.risk_target > df.U_source
    df["adjusted_bound"] = np.minimum(1.0, df.ba_estimate * df.U_source)
    df["above_adjusted_bound"] = np.where(df.ba_estimate.notna(), df.risk_target > df.adjusted_bound, np.nan)
    return df


def split_small(y, seed, shares=(0.5, 0.35, 0.15)) -> np.ndarray:
    """Stratified random split codes 0, 1, 2 (training, calibration split 1, test) for a small dataset."""
    y = np.asarray(y).astype(int)
    rng = np.random.default_rng(seed)
    code = np.empty(len(y), dtype=int)
    cum = np.cumsum(shares)[:-1] / np.sum(shares)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for k, part in enumerate(np.split(idx, np.round(cum * len(idx)).astype(int))):
            code[part] = k
    return code


def choose_settings(X, y, grid, seed, family="xgboost", folds=5):
    """The grid point with the highest out-of-fold AUROC on the given (training) rows; ties go to the first."""
    y = np.asarray(y).astype(int)
    rows = []
    for p in grid:
        s = train.oof_scores(family, X, y, p, seed, folds=folds)
        rows.append({**p, "oof_auroc": float(roc_auc_score(y, s))})
    table = pd.DataFrame(rows)
    return dict(grid[int(np.argmax(table.oof_auroc.to_numpy()))]), table
