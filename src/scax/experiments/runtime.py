"""Runtime (plan Section 7.7, E6, RQ6).

Time per distinct grid cell, split by stage (deletion AXps and inflation) and by operation (interval
pre-check, exact search, and everything else: memo look-ups, box bookkeeping); scaling with the number
of trees and the depth; effective model size (trees with at least one split, internal nodes, depth
actually reached, distinct thresholds); cache hit rates.

The timed generator is the frozen CandidateGenerator: only its pre-check, exact search, deletion and
inflation methods are wrapped with clocks on the instance, so every answer is unchanged.
"""
from __future__ import annotations

import os
import pickle
import platform
import signal
import time
from collections import defaultdict

import numpy as np

from ..explain.candidates import CandidateGenerator


def tree_depth(tree):
    """Depth reached by one tree (0 for a single leaf)."""
    best, stack = 0, [(0, 0)]
    while stack:
        node, d = stack.pop()
        if tree.feature[node] < 0:
            best = max(best, d)
        else:
            stack.append((tree.left[node], d + 1))
            stack.append((tree.right[node], d + 1))
    return best


def model_size(ens, grid=None):
    """Requested and effective size of an exported ensemble."""
    internal = [sum(1 for f in t.feature if f >= 0) for t in ens.trees]
    depths = [tree_depth(t) for t in ens.trees]
    out = {"trees": ens.n_trees, "trees_with_a_split": int(sum(n > 0 for n in internal)),
           "internal_nodes": int(sum(internal)), "leaves": int(sum(len(t.feature) - n for t, n in zip(ens.trees, internal))),
           "max_depth_reached": int(max(depths)) if depths else 0,
           "mean_depth_reached": float(np.mean(depths)) if depths else 0.0}
    if grid is not None:
        out["distinct_thresholds"] = int(np.asarray(grid.K).sum())
        out["log10_family_size"] = float(grid.family_size_log10())
    return out


def spread(seconds, censored=None):
    """Median, quartiles, 95th percentile; with censored cells (stopped at a cap) the times are lower
    bounds, so a quantile is a lower bound once the censored share reaches its level."""
    s = np.asarray(seconds, dtype=float)
    if s.size == 0:
        return {"cells": 0}
    q = np.percentile(s, [25, 50, 75, 95])
    c = np.zeros(s.size, dtype=bool) if censored is None else np.asarray(censored, dtype=bool)
    share = float(c.mean())
    return {"cells": int(s.size), "median_seconds": float(q[1]), "q25_seconds": float(q[0]), "q75_seconds": float(q[2]),
            "p95_seconds": float(q[3]), "max_seconds": float(s.max()), "mean_seconds": float(s.mean()),
            "total_seconds": float(s.sum()), "censored_cells": int(c.sum()),
            "median_is_lower_bound": bool(share >= 0.5), "p95_is_lower_bound": bool(share >= 0.05)}


class CellTimeout(Exception):
    """A cell went over its time cap."""


def _alarm(signum, frame):
    raise CellTimeout()


class TimedGenerator:
    """The frozen candidate generator with clocks on its pre-check, exact search, deletion and inflation."""

    def __init__(self, pred, M=10, seed=42, node_budget=200000):
        self.gen = gen = CandidateGenerator(pred, M=M, seed=seed, node_budget=node_budget)
        self.clock = defaultdict(float)
        self.stage = "deletion"
        o = gen.oracle
        ib_sound, cs_check, axp, inflate = o.ib.sound, o.cs.check, gen.axp, gen.inflate

        def ib_timed(box, y):
            t = time.perf_counter()
            r = ib_sound(box, y)
            self.clock[(self.stage, "precheck")] += time.perf_counter() - t
            return r

        def cs_timed(box, y):
            t = time.perf_counter()
            r = cs_check(box, y)
            self.clock[(self.stage, "search")] += time.perf_counter() - t
            return r

        def axp_timed(*a):
            self.stage = "deletion"
            t = time.perf_counter()
            r = axp(*a)
            self.clock[("deletion", "stage")] += time.perf_counter() - t
            return r

        def inflate_timed(*a):
            self.stage = "inflation"
            t = time.perf_counter()
            r = inflate(*a)
            self.clock[("inflation", "stage")] += time.perf_counter() - t
            self.stage = "deletion"
            return r

        o.ib.sound, o.cs.check, gen.axp, gen.inflate = ib_timed, cs_timed, axp_timed, inflate_timed

    def time_cells(self, cells, yhat, time_budget=None, max_cells=None, cell_cap=None, log=None):
        """Candidate sets for distinct cells in the given order, one timing row per new cell.

        time_budget stops the run between cells; cell_cap (seconds, Unix timer) stops a single cell,
        which is then recorded as censored at the cap and not tried again."""
        gen, o = self.gen, self.gen.oracle
        rows, start, censored_keys = [], time.perf_counter(), set()
        old = signal.signal(signal.SIGALRM, _alarm) if cell_cap else None
        try:
            for i in range(len(cells)):
                if max_cells is not None and len(rows) >= max_cells:
                    break
                if time_budget is not None and time.perf_counter() - start > time_budget:
                    break
                key = gen.cell_key(cells[i])
                if key in gen.cache or key in censored_keys:
                    continue
                c0, s0 = dict(self.clock), dict(vars(o.stats))
                t = time.perf_counter()
                censored = False
                try:
                    try:
                        if cell_cap:
                            signal.setitimer(signal.ITIMER_REAL, cell_cap)
                        cands = gen.candidates(cells[i], int(yhat[i]))
                    finally:
                        if cell_cap:
                            signal.setitimer(signal.ITIMER_REAL, 0)
                except CellTimeout:                  # also caught if the timer fires while it is being cancelled
                    censored, cands = True, []
                    censored_keys.add(key)
                    self.stage = "deletion"
                total = time.perf_counter() - t
                rows.append(self._row(i, yhat, total, censored, cands, c0, s0))
                if log is not None and len(rows) % 50 == 0:
                    log(f"  {len(rows)} cells, {time.perf_counter() - start:.0f}s")
        finally:
            if cell_cap:
                signal.signal(signal.SIGALRM, old)
        return rows

    def _row(self, i, yhat, total, censored, cands, c0, s0):
        o = self.gen.oracle
        d = {k: self.clock[k] - c0.get(k, 0.0) for k in self.clock}
        st = {k: vars(o.stats)[k] - s0[k] for k in ("checks", "memo_hits", "precheck_sound", "search_calls", "search_unknown")}
        pre = d.get(("deletion", "precheck"), 0.0) + d.get(("inflation", "precheck"), 0.0)
        sea = d.get(("deletion", "search"), 0.0) + d.get(("inflation", "search"), 0.0)
        return {"position": i, "predicted": int(yhat[i]), "seconds": total, "censored": censored,
                "deletion_seconds": d.get(("deletion", "stage"), 0.0),
                "inflation_seconds": d.get(("inflation", "stage"), 0.0),
                "deletion_precheck_seconds": d.get(("deletion", "precheck"), 0.0),
                "deletion_search_seconds": d.get(("deletion", "search"), 0.0),
                "inflation_precheck_seconds": d.get(("inflation", "precheck"), 0.0),
                "inflation_search_seconds": d.get(("inflation", "search"), 0.0),
                "precheck_seconds": pre, "search_seconds": sea, "other_seconds": max(0.0, total - pre - sea),
                "candidates": len(cands), **st}


def phase_summary(rows):
    """Mean seconds per cell by stage and operation, and shares of the total."""
    if not rows:
        return {}
    keys = ["deletion_precheck_seconds", "deletion_search_seconds", "inflation_precheck_seconds",
            "inflation_search_seconds", "precheck_seconds", "search_seconds", "other_seconds",
            "deletion_seconds", "inflation_seconds", "seconds"]
    m = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    tot = m["seconds"]
    checks = sum(r["checks"] for r in rows)
    out = {f"mean_{k}": v for k, v in m.items()}
    out.update({"precheck_share": m["precheck_seconds"] / tot, "search_share": m["search_seconds"] / tot,
                "other_share": m["other_seconds"] / tot, "inflation_share": m["inflation_seconds"] / tot,
                "checks_per_cell": checks / len(rows),
                "memo_hit_rate": sum(r["memo_hits"] for r in rows) / max(1, checks),
                "precheck_decided_share": sum(r["precheck_sound"] for r in rows) / max(1, checks),
                "search_calls_per_cell": sum(r["search_calls"] for r in rows) / len(rows),
                "search_unknown": int(sum(r["search_unknown"] for r in rows)),
                "candidates_per_cell": float(np.mean([r["candidates"] for r in rows]))})
    return out


def cache_runtime(path, keys=None):
    """Seconds per cell and solver counts stored with a candidate cache (optionally only the given cells)."""
    with open(path, "rb") as f:
        d = pickle.load(f)
    sec = d.get("seconds", {})
    s = [sec[k] for k in dict.fromkeys(keys) if k in sec] if keys is not None else list(sec.values())   # each distinct cell once
    st = d.get("stats", {})
    checks = st.get("checks", 0)
    out = spread(s)
    out.update({"cells_in_cache": len(d.get("cache", {})), "checks": int(checks),
                "memo_hit_rate": st.get("memo_hits", 0) / checks if checks else None,
                "precheck_decided_share": st.get("precheck_sound", 0) / checks if checks else None,
                "search_unknown": int(st.get("search_unknown", 0))})
    return out


def hardware():
    cpu = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"cpu": cpu or platform.processor(), "logical_cpus": os.cpu_count(), "python": platform.python_version(),
            "machine": platform.machine()}
