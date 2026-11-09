"""Verification of candidate boxes (plan Algorithm 3, step 5; gate G2).

Every distinct candidate box is re-checked by the second exact implementation (RealSearch, real
thresholds and exact fractions). A box it calls unsound is a disagreement between two exact
procedures and stops the run; a box on which it exhausts its budget is not verified and is removed
from the candidate sets, so it is never released. A random sample of boxes is also checked by Z3
on the real-input encoding, with a timeout; Z3 finding a counterexample in a candidate box stops
the run, and timeouts are counted.
"""
from __future__ import annotations

import time

import numpy as np

from ..encode.real_encoding import Z3RealEncoding
from ..encode.real_search import RealSearch


class DisagreementError(RuntimeError):
    pass


def verify_boxes(grid, boxes, node_budget: int = 200000):
    """boxes: iterable of (key, y). Returns {(key, y): 'sound' | 'unknown'} and per-box seconds."""
    rs = RealSearch(grid.ens, grid.thresholds, node_budget=node_budget)
    out, times = {}, []
    for key, y in boxes:
        if (key, y) in out:
            continue
        t0 = time.perf_counter()
        r = rs.check(grid.from_key(key), y)
        times.append(time.perf_counter() - t0)
        if r == "unsound":
            raise DisagreementError(f"second check finds a counterexample in a box the primary check proved sound: {key}, class {y}")
        out[(key, y)] = r
    return out, times


def z3_sample(grid, boxes, n: int, seed: int, timeout_ms: int = 30000, time_budget_s: float | None = None):
    """Z3 on a random sample of candidate boxes: counts of sound / unknown, and seconds.

    With a time budget, sampling stops once the budget is spent; 'checked' says how many were run.
    """
    boxes = list(dict.fromkeys(boxes))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(boxes), size=min(n, len(boxes)), replace=False) if boxes else []
    enc = Z3RealEncoding(grid.ens, grid.thresholds, timeout_ms=timeout_ms)
    res, times = {"sound": 0, "unknown": 0, "checked": 0}, []
    start = time.perf_counter()
    for i in pick:
        if time_budget_s is not None and time.perf_counter() - start > time_budget_s:
            break
        res["checked"] += 1
        key, y = boxes[int(i)]
        t0 = time.perf_counter()
        r = enc.check(grid.from_key(key), y)
        times.append(time.perf_counter() - t0)
        if r == "unsound":
            raise DisagreementError(f"Z3 finds a counterexample in a candidate box: {key}, class {y}")
        res[r] += 1
    return res, times


def drop_unverified(cache: dict, verdicts: dict) -> int:
    removed = 0
    for cell, cands in cache.items():
        keep = [c for c in cands if verdicts.get((c.key, c.y)) == "sound"]
        removed += len(cands) - len(keep)
        cache[cell] = keep
    return removed
