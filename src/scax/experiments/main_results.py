"""Frozen-pipeline helpers for notebooks 08 and 09 (plan Algorithms 4 and 5, Theorem 4).

load_frozen_task checks every hash recorded at freezing time before anything is used. The release
helpers apply the frozen decision rule (Eq. 8) to the candidate sets of many patients at once.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..certify.clopper_pearson import cp_lower, cp_upper
from ..certify.decision import certify_policy_by_class, release
from ..certify.prior import OccamPrior
from ..frozen import load_frozen
from ..models.export import Ensemble
from ..models.predictor import ExactPredictor
from ..utils import sha256_array, sha256_file


class FrozenTask:
    def __init__(self, task, repo_dir, drive_root, frozen_sha):
        repo_dir, drive_root = Path(repo_dir), Path(drive_root)
        self.task = task
        self.cfg, self.cfg_sha = load_frozen(repo_dir / "configs" / "frozen" / f"{task}.json", frozen_sha)
        cfg = self.cfg
        model_file = drive_root / cfg["model"]["file"]
        self.ens = Ensemble.load(model_file)
        if self.ens.hash() != cfg["model"]["sha256"]:
            raise ValueError(f"{task}: model file differs from the frozen model")
        self.pred = ExactPredictor(self.ens)
        self.grid = self.pred.grid
        prior_file = drive_root / cfg["prior"]["file"]
        if sha256_file(prior_file) != cfg["prior"]["sha256"]:
            raise ValueError(f"{task}: prior file differs from the frozen prior")
        prior = json.load(open(prior_file))
        self.lam = float(prior["lambda"])
        self.occam = OccamPrior(prior["K"])
        if list(self.occam.K) != list(self.grid.K):
            raise ValueError(f"{task}: prior grid differs from the model grid")
        self.counts = {tuple(tuple(t) for t in k): v for k, v in prior["counts"]}
        self.M = int(cfg["candidates"]["M"])
        frozen_dir = model_file.parent
        self.test_rows = np.load(frozen_dir / f"{task}__test_explained.npy")
        self.cal2_rows = np.load(frozen_dir / f"{task}__cal2_explained.npy")
        if sha256_array(self.test_rows) != cfg["explained"]["test_sha256"] or \
                sha256_array(self.cal2_rows) != cfg["explained"]["cal2_sha256"]:
            raise ValueError(f"{task}: explained subsample files differ from the frozen ones")
        self.targets = cfg["targets"]

    def log_prior(self, keys):
        from .development import log_prior
        return log_prior(keys, self.occam, self.counts, self.lam)


def pair_arrays(cand_lists, box_index, M):
    pp, pb = [], []
    for i, cands in enumerate(cand_lists):
        for c in cands:
            if c.order < M and (c.key, c.y) in box_index:
                pp.append(i); pb.append(box_index[(c.key, c.y)])
    return np.array(pp, dtype=np.int64), np.array(pb, dtype=np.int64)


def release_patients(pp, pb, n_patients, box_size, box_class, n_B, U, alpha0, alpha1):
    return release(pp, pb, n_patients, box_size, box_class, n_B, U, {0: alpha0, 1: alpha1})


def kind_summary(rel, yhat, y=None):
    """Accepted counts per kind (class of the released box = predicted class), and errors if y is given."""
    out = {}
    acc = rel >= 0
    for k in (0, 1):
        a = acc & (yhat == k)
        out[k] = {"accepted": int(a.sum()), "share_of_patients": float(a.mean())}
        if y is not None:
            e = int((y[a] != k).sum())
            out[k]["errors"] = e
            out[k]["error_rate"] = e / int(a.sum()) if a.sum() else float("nan")
            out[k]["error_upper_95"] = float(cp_upper(e, int(a.sum()), 0.05)) if a.sum() else 1.0
            out[k]["error_lower_95"] = float(cp_lower(e, int(a.sum()), 0.05)) if a.sum() else 0.0
    return out


def policy_test(rel, yhat, y, alpha_sel, delta2):
    s = kind_summary(rel, yhat, y)
    return certify_policy_by_class({k: s[k]["errors"] for k in (0, 1)}, {k: s[k]["accepted"] for k in (0, 1)},
                                   alpha_sel, delta2)
