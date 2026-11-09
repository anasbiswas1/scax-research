"""Priors over the finite box family (plan Definition 6, Eqs. 3 and 4).

pi_Occ splits mass equally over box sizes 0..d, then equally over feature subsets of that size,
then equally over the non-full intervals of each constrained feature: it sums to exactly 1 when
every feature has at least one threshold, and to less than 1 otherwise. pi_mix mixes it with the
share of training candidate sets that contain the box. Both are fixed from the training portion
only; the prior file and its hash are written before calibration split 1 is loaded.
"""
from __future__ import annotations

import hashlib
import json
from math import comb, lgamma, log

import numpy as np


def log_comb(n, k):
    return lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)


class OccamPrior:
    def __init__(self, K):
        self.K = np.asarray(K, dtype=np.int64)
        self.d = len(self.K)
        self.m = [comb(int(k) + 2, 2) - 1 for k in self.K]    # non-full intervals per feature
        self.log_m = [log(m) if m > 0 else float("inf") for m in self.m]

    def log_pi(self, key) -> float:
        s = len(key)
        out = -log(self.d + 1) - log_comb(self.d, s)
        for j, _, _ in key:
            out -= self.log_m[j]
        return out

    def pi(self, key) -> float:
        return float(np.exp(self.log_pi(key)))

    def describe(self):
        return {"type": "occam", "d": self.d, "K": self.K.tolist()}


class MixturePrior:
    """lambda * pi_Occ + (1 - lambda) * N_tr(B) / sum_B' N_tr(B')."""

    def __init__(self, occam: OccamPrior, lam: float, train_counts: dict):
        if not 0 <= lam <= 1:
            raise ValueError("lambda must lie in [0, 1]")
        self.occam = occam
        self.lam = lam
        self.counts = dict(train_counts)
        self.total = float(sum(self.counts.values()))

    def pi(self, key) -> float:
        p = self.lam * self.occam.pi(key) if self.lam > 0 else 0.0
        if self.lam < 1 and self.total > 0:
            p += (1 - self.lam) * self.counts.get(key, 0) / self.total
        return p

    def log_pi(self, key) -> float:
        p = self.pi(key)
        return float(np.log(p)) if p > 0 else float("-inf")

    def describe(self):
        return {"type": "mixture", "lambda": self.lam, "occam": self.occam.describe(),
                "train_boxes": len(self.counts), "train_total": self.total}


def train_counts(candidate_sets) -> dict:
    """N_tr(B): number of training cells (with multiplicity) whose candidate set contains B."""
    out = {}
    for cands in candidate_sets:
        for c in {c.key for c in cands}:
            out[c] = out.get(c, 0) + 1
    return out


def prior_hash(prior) -> str:
    d = prior.describe()
    if isinstance(prior, MixturePrior):
        d["counts"] = sorted((json.dumps(k), v) for k, v in prior.counts.items())
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
