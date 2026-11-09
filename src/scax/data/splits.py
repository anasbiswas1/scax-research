"""Label-independent random splits (plan Section 6.3).

Rows are allocated by a seeded random permutation that never looks at the outcome: training
portion 40% (development: dev-train 75%, dev-cal 25% of it), calibration split 1 30%, calibration
split 2 15%, test 15%. The assignment is a random partition of the available rows. Only the
training portion may be loaded before notebook 05 writes the frozen configuration hash.
"""
from __future__ import annotations

import numpy as np

from ..utils import sha256_array

SHARES = {"train": 0.40, "cal1": 0.30, "cal2": 0.15, "test": 0.15}
DEV_TRAIN_SHARE = 0.75
SPLIT_CODES = {"dev_train": 0, "dev_cal": 1, "cal1": 2, "cal2": 3, "test": 4}


def make_splits(n_rows: int, seed: int) -> np.ndarray:
    """Split code per row (see SPLIT_CODES)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_rows)
    n_train = int(round(SHARES["train"] * n_rows))
    n_cal1 = int(round(SHARES["cal1"] * n_rows))
    n_cal2 = int(round(SHARES["cal2"] * n_rows))
    n_dev_train = int(round(DEV_TRAIN_SHARE * n_train))
    code = np.empty(n_rows, dtype=np.int8)
    bounds = [0, n_dev_train, n_train, n_train + n_cal1, n_train + n_cal1 + n_cal2, n_rows]
    for c in range(5):
        code[perm[bounds[c]:bounds[c + 1]]] = c
    return code


def indices(code: np.ndarray, name: str) -> np.ndarray:
    if name == "train":
        return np.flatnonzero(code <= 1)
    return np.flatnonzero(code == SPLIT_CODES[name])


def summary(code: np.ndarray) -> dict:
    out = {name: int((code == c).sum()) for name, c in SPLIT_CODES.items()}
    out["sha256"] = sha256_array(code)
    return out
