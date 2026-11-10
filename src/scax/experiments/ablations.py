"""Ablations on the frozen model, one factor at a time (plan Section 7.4).

Everything except the ablated factor stays as frozen: model, candidate sets of notebook 08,
calibration split 1 for certificates, the explained patients of split 2 for the policy test, targets
and confidence levels. Boxes that become releasable under a variant get the second exact check first.
These runs describe how the design choices behave; they do not change the frozen configuration.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..certify.certificate import upper_bounds
from ..certify.counting import box_counts
from ..explain.verify import verify_boxes
from .baselines import KINDS, RealComparison, released_metrics
from .development import log_prior

RULES = {"most succinct": "succinct", "largest support": "support", "tightest bound": "bound"}


def release_by(pair_patient, pair_box, n_patients, box_size, box_class, n_B, U, alpha, rule="succinct"):
    """Released box per patient (-1 = abstain) under a selection rule.

    succinct (frozen): fewest features, then most support, then tightest bound;
    support: most support, then fewest features, then tightest bound;
    bound: tightest bound, then fewest features, then most support.
    """
    pair_patient = np.asarray(pair_patient); pair_box = np.asarray(pair_box)
    a = np.where(np.asarray(box_class)[pair_box] == 1, alpha[1], alpha[0])
    ok = U[pair_box] <= a
    out = np.full(n_patients, -1, dtype=np.int64)
    if not ok.any():
        return out
    pp, pb = pair_patient[ok], pair_box[ok]
    size, n, u = np.asarray(box_size)[pb], n_B[pb], U[pb]
    keys = {"succinct": (u, -n, size, pp), "support": (u, size, -n, pp), "bound": (-n, size, u, pp)}[rule]
    order = np.lexsort(keys)
    pp, pb = pp[order], pb[order]
    first = np.r_[True, pp[1:] != pp[:-1]]
    out[pp[first]] = pb[first]
    return out


class FrozenAblation(RealComparison):
    def __init__(self, ft, box_df, gen, Xa, y, code, delta1, delta2):
        super().__init__(ft, box_df, gen, Xa, y, code, delta1, delta2)
        from ..data import splits
        cal1 = splits.indices(code, "cal1")
        self.C1, self.y1 = self.g.cells(Xa[cal1]), np.asarray(y)[cal1]
        self.n_full, self.k_full = box_df.n_B.to_numpy(), box_df.k_B.to_numpy()

    def prior(self, lam):
        return log_prior(self.keys, self.ft.occam, self.ft.counts, lam)

    def evaluate(self, n, k, log_pi, maskE=None, maskT=None, rule="succinct"):
        """Releases on the explained patients of split 2 (policy test) and of the test split (results)."""
        with np.errstate(invalid="ignore"):
            U = upper_bounds(k, n, math.log(self.delta1) + np.where(np.isfinite(log_pi), log_pi, 0.0))
        U = np.where(np.isfinite(log_pi), U, 1.0)
        a_box = np.where(self.cls == 1, self.alpha[1], self.alpha[0])
        need = np.flatnonzero((U <= a_box) & (self.status == 0))
        if len(need):
            verdicts, _ = verify_boxes(self.g, [(self.keys[i], int(self.cls[i])) for i in need])
            self.status[need] = [1 if verdicts[(self.keys[i], int(self.cls[i]))] == "sound" else -1 for i in need]
        U = np.where(self.status == 1, U, 1.0)
        mE = np.ones(len(self.ppE), bool) if maskE is None else maskE
        mT = np.ones(len(self.ppT), bool) if maskT is None else maskT
        relE = release_by(self.ppE[mE], self.pbE[mE], len(self.yhatE), self.size, self.cls, n, U, self.alpha, rule)
        relT = release_by(self.ppT[mT], self.pbT[mT], len(self.yhatT), self.size, self.cls, n, U, self.alpha, rule)
        met = released_metrics(relT, self.cls, self.yhatT, self.yT, self.size, n, self.n_test, self.k_test, self.alpha, len(relT))
        pol = self._policy(relE, self.yhatE, self.yE)
        rows = []
        for kk in KINDS:
            rows.append({"kind": kk, **met[kk], "cal2_accepted": pol[kk]["accepted"], "cal2_errors": pol[kk]["errors"],
                         "U_sel": pol[kk]["U_sel"], "alpha_sel": pol[kk]["alpha_sel"], "policy_certified": pol[kk]["certified"]})
        return rows, int(len(need))

    def run_factors(self, n1_sizes, n1_draws, seed, lambdas=(1.0, 0.75, 0.5, 0.25, 0.0), Ms=(1, 3, 5, 10), log=print):
        """All one-factor variants; returns a DataFrame with factor, level, frozen flag and draw."""
        out = []
        lam0 = float(self.ft.lam)
        base_pi = self.prior(lam0)

        def add(factor, level, frozen, rows, draw=0):
            for r in rows:
                out.append({"factor": factor, "level": level, "frozen": frozen, "draw": draw, **r})

        for lam in lambdas:
            name = {1.0: "Occam prior only", 0.0: "training support only"}.get(lam, f"mixture, lambda = {lam:g}")
            rows, nv = self.evaluate(self.n_full, self.k_full, base_pi if lam == lam0 else self.prior(lam))
            add("prior", name, lam == lam0, rows)
            log(f"  prior: {name} (newly checked boxes {nv})")
        constructions = {"AXps only": (self.pkE == 0, self.pkT == 0),
                         "widest inflated box of each order only": (self.pwE, self.pwT),
                         "every sound box on the inflation path": (None, None)}
        for name, (mE, mT) in constructions.items():
            rows, nv = self.evaluate(self.n_full, self.k_full, base_pi, mE, mT)
            add("explanation construction", name, mE is None, rows)
            log(f"  construction: {name} (newly checked boxes {nv})")
        for name, rule in RULES.items():
            rows, _ = self.evaluate(self.n_full, self.k_full, base_pi, rule=rule)
            add("selection rule", name, rule == "succinct", rows)
        for M in Ms:
            rows, nv = self.evaluate(self.n_full, self.k_full, base_pi, self.poE < M, self.poT < M)
            add("candidates M", f"M = {M}", M == self.M, rows)
        rng = np.random.default_rng(seed)
        for size in n1_sizes:
            for d in range(n1_draws):
                idx = rng.choice(len(self.y1), size, replace=False)
                n, k = box_counts(self.keys, self.cls, self.C1[idx], self.y1[idx])
                rows, nv = self.evaluate(n, k, base_pi)
                add("calibration size", f"{size:,} records", False, rows, d)
            log(f"  calibration split 1 subsample of {size:,}: {n1_draws} draws")
        rows, _ = self.evaluate(self.n_full, self.k_full, base_pi)
        add("calibration size", f"{len(self.y1):,} records", True, rows)
        return pd.DataFrame(out)
