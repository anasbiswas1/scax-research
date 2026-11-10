"""Candidate explanations per grid cell (plan Section 5.5 and Algorithm 3).

For a cell with predicted class y, deletion-based AXps are built along M feature orders; each AXp
is then inflated feature by feature, each side of each interval by binary search over adjacent
cells (soundness is monotone under inclusion), and every sound box visited on the inflation path is
kept as a candidate. Soundness is decided exactly: the interval pre-check first, then the cell
branch-and-bound. Candidate sets are cached per cell, keyed by the classifier hash and the
generation settings, and every distinct candidate box is verified afterwards by the independent
real-input Z3 encoding (verify.py).
"""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..encode.cell_search import CellSearch
from ..encode.grid import Grid
from ..encode.interval_bound import IntervalBound
from ..models.predictor import ExactPredictor


@dataclass
class Candidate:
    key: tuple          # ((feature, a, b), ...) over constrained features
    y: int              # predicted class of the cell (the box's class)
    size: int
    order: int          # index of the first feature order that produced it (0-based)
    kind: str           # "axp" or "inflated"


@dataclass
class Stats:
    cells: int = 0
    checks: int = 0
    memo_hits: int = 0
    precheck_sound: int = 0
    search_calls: int = 0
    search_unknown: int = 0
    seconds: list = field(default_factory=list)


class SoundnessOracle:
    """Exact soundness with memoisation: interval pre-check, then branch-and-bound."""

    def __init__(self, pred: ExactPredictor, node_budget: int = 200000, memo_limit: int = 300000):
        self.pred = pred
        self.grid = pred.grid
        self.ib = IntervalBound(pred)
        self.cs = CellSearch(self.ib, node_budget=node_budget)
        self.memo = {}
        self.memo_limit = memo_limit          # the memo is a pure cache; clearing it never changes an answer
        self.stats = Stats()

    def sound(self, box, y) -> bool:
        self.stats.checks += 1
        k = (self.grid.key(box), int(y))
        if k in self.memo:
            self.stats.memo_hits += 1
            return self.memo[k]
        if self.ib.sound(box, y):
            self.stats.precheck_sound += 1
            r = True
        else:
            self.stats.search_calls += 1
            out = self.cs.check(box, y)
            if out == "unknown":
                self.stats.search_unknown += 1
            r = out == "sound"
        if len(self.memo) >= self.memo_limit:
            self.memo.clear()
        self.memo[k] = r
        return r


def feature_orders(grid: Grid, M: int, seed: int):
    """Order 0 tries the least-used features first (by number of splits); orders 1..M-1 are seeded permutations."""
    counts = np.zeros(grid.d, dtype=np.int64)
    for feat, *_ in grid.trees:
        for f in feat:
            if f >= 0:
                counts[f] += 1
    active = list(grid.active)
    first = sorted(active, key=lambda j: (counts[j], j))
    rng = np.random.default_rng(seed)
    orders = [first]
    while len(orders) < M:
        orders.append([active[i] for i in rng.permutation(len(active))])
    return orders


class CandidateGenerator:
    def __init__(self, pred: ExactPredictor, M: int = 10, seed: int = 0, node_budget: int = 200000):
        self.pred = pred
        self.grid = pred.grid
        self.M = M
        self.seed = seed
        self.oracle = SoundnessOracle(pred, node_budget)
        self.orders = feature_orders(self.grid, M, seed)
        self.cache = {}
        self.seconds = {}
        self.settings = {"classifier": pred.ens.hash(), "M": M, "seed": seed, "node_budget": node_budget,
                         "inflation": "binary search per side, every sound visited box kept"}

    def cell_key(self, cell_row) -> tuple:
        return tuple(int(cell_row[j]) for j in self.grid.active)

    def axp(self, cell_row, y, order):
        box = list(self.grid.point_box(cell_row))
        if not self.oracle.sound(tuple(box), y):
            raise RuntimeError("a single cell must be sound for its own predicted class")
        for j in order:
            trial = box.copy()
            trial[j] = (0, int(self.grid.K[j]))
            if self.oracle.sound(tuple(trial), y):
                box = trial
        return tuple(box)

    def inflate(self, box, y, order):
        """Every sound box visited while widening each side of each constrained interval."""
        box = list(box)
        visited = []
        for j in [f for f in order if f in set(self.grid.constrained(box))]:
            a, b = box[j]
            lo, hi = 0, a                       # smallest sound lower end in [0, a]
            while lo < hi:
                mid = (lo + hi) // 2
                trial = box.copy(); trial[j] = (mid, b)
                if self.oracle.sound(tuple(trial), y):
                    hi = mid
                    visited.append(tuple(trial))
                else:
                    lo = mid + 1
            box[j] = (hi, b)
            a = hi
            K = int(self.grid.K[j])
            lo, hi = b, K                       # largest sound upper end in [b, K]
            while lo < hi:
                mid = (lo + hi + 1) // 2
                trial = box.copy(); trial[j] = (a, mid)
                if self.oracle.sound(tuple(trial), y):
                    lo = mid
                    visited.append(tuple(trial))
                else:
                    hi = mid - 1
            box[j] = (a, lo)
        return visited

    def candidates(self, cell_row, y=None):
        key = self.cell_key(cell_row)
        if key in self.cache:
            return self.cache[key]
        t0 = time.perf_counter()
        if y is None:
            y = int(self.pred.predict_cells(np.asarray(cell_row)[None, :])[0])
        seen = {}
        for m, order in enumerate(self.orders):
            base = self.axp(cell_row, y, order)
            k = self.grid.key(base)
            if k not in seen:
                seen[k] = Candidate(k, y, len(k), m, "axp")
            for b in self.inflate(base, y, order):
                kb = self.grid.key(b)
                if kb not in seen:
                    seen[kb] = Candidate(kb, y, len(kb), m, "inflated")
        out = list(seen.values())
        self.cache[key] = out
        dt = time.perf_counter() - t0
        self.seconds[key] = dt
        self.oracle.stats.cells += 1
        self.oracle.stats.seconds.append(dt)
        return out

    # ------------------------------------------------------------ cache on Drive
    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"settings": self.settings, "cache": self.cache, "seconds": self.seconds,
                         "stats": vars(self.oracle.stats)}, f)

    def load(self, path) -> bool:
        p = Path(path)
        if not p.exists():
            return False
        with open(p, "rb") as f:
            d = pickle.load(f)
        if d["settings"] != self.settings:
            raise ValueError("cache was built for a different classifier or settings")
        self.cache = d["cache"]
        self.seconds = d.get("seconds", {})
        if "stats" in d:
            self.oracle.stats = Stats(**d["stats"])
        return True
