"""Paper results (notebook 16): percentile bootstrap intervals for the main table, markdown tables and the
claims register.

The bootstrap resamples the explained test patients with replacement. Every main-table statistic (the share
of patients given a certified claim of each kind, and the error rate among them) depends on a patient only
through one of five categories: no release; a negative-outcome release with a correct or a wrong
prediction; an enrichment release with a correct or a wrong prediction. Resampling patients is therefore the
same as drawing the five category counts from a multinomial distribution with the observed shares, so the
bootstrap can run on the saved per-patient releases or, where only counts were saved, on the counts. Both
give the same distribution; notebook 16 checks this on the rows that have both.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

CATEGORIES = ("no release", "negative outcome, correct", "negative outcome, wrong", "enrichment, correct",
              "enrichment, wrong")
KIND_NAMES = {0: "negative outcome", 1: "enrichment"}


def label_seed(label, base=42):
    """A stable 32-bit seed for a row label."""
    return int.from_bytes(hashlib.sha256(f"{base}:{label}".encode()).digest()[:4], "little")


def categories(predicted, released, outcome):
    """Category (0 to 4, see CATEGORIES) of every patient."""
    p = np.asarray(predicted, dtype=int)
    r = np.asarray(released, dtype=bool)
    y = np.asarray(outcome, dtype=int)
    cat = np.zeros(len(p), dtype=np.int64)
    neg, enr = r & (p == 0), r & (p == 1)
    cat[neg] = np.where(y[neg] == 0, 1, 2)
    cat[enr] = np.where(y[enr] == 1, 3, 4)
    return cat


def counts_from(n, accepted_0, errors_0, accepted_1, errors_1):
    """The five category counts from saved counts."""
    c = np.array([n - accepted_0 - accepted_1, accepted_0 - errors_0, errors_0, accepted_1 - errors_1, errors_1],
                 dtype=np.int64)
    if (c < 0).any():
        raise ValueError(f"inconsistent counts: {c}")
    return c


def bootstrap_patients(cat, B=10000, seed=0, chunk=500):
    """Category counts of B resamples of the patients (rows: resamples)."""
    cat = np.asarray(cat, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n, out = len(cat), []
    for start in range(0, B, chunk):
        b = min(chunk, B - start)
        c = cat[rng.integers(0, n, size=(b, n))]
        out.append(np.stack([(c == k).sum(axis=1) for k in range(5)], axis=1))
    return np.vstack(out)


def bootstrap_counts(counts, B=10000, seed=0):
    """Category counts of B resamples drawn from saved counts. With a list of count vectors (one per
    stratum, for example per site), each stratum is resampled with its own size and the counts are added."""
    rng = np.random.default_rng(seed)
    strata = [np.asarray(counts, dtype=np.int64)] if np.ndim(counts) == 1 else [np.asarray(c, dtype=np.int64) for c in counts]
    total = np.zeros((B, 5), dtype=np.int64)
    for c in strata:
        n = int(c.sum())
        if n:
            total += rng.multinomial(n, c / n, size=B)
    return total


def statistics(C):
    """Shares and error rates for rows of category counts (NaN error rate when nothing is released)."""
    C = np.atleast_2d(np.asarray(C, dtype=float))
    n = C.sum(axis=1)
    acc0, acc1 = C[:, 1] + C[:, 2], C[:, 3] + C[:, 4]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"share_0": acc0 / n, "share_1": acc1 / n,
                "error_0": np.where(acc0 > 0, C[:, 2] / acc0, np.nan),
                "error_1": np.where(acc1 > 0, C[:, 4] / acc1, np.nan)}


def summarise(C, observed, level=0.95):
    """Observed value, percentile interval, bootstrap mean and standard deviation of each statistic.
    Resamples without any release of a kind have no error rate and are left out of that interval; their
    share is reported."""
    obs = {k: float(v[0]) for k, v in statistics(observed).items()}
    boot = statistics(C)
    a = (1 - level) / 2
    out = {}
    for k, s in boot.items():
        ok = ~np.isnan(s)
        out[k] = obs[k]
        if ok.any():
            out[f"{k}_lo"], out[f"{k}_hi"] = (float(v) for v in np.quantile(s[ok], [a, 1 - a]))
            out[f"{k}_boot_mean"], out[f"{k}_boot_sd"] = float(s[ok].mean()), float(s[ok].std())
        else:
            out[f"{k}_lo"] = out[f"{k}_hi"] = out[f"{k}_boot_mean"] = out[f"{k}_boot_sd"] = float("nan")
        out[f"{k}_undefined_share"] = float(1 - ok.mean())
    return out


def same_distribution(s1, s2, B):
    """True when two bootstrap summaries of the same statistics agree within Monte Carlo error: means within
    six standard errors and standard deviations within 10%."""
    for k in ("share_0", "share_1", "error_0", "error_1"):
        m1, m2, d1, d2 = s1[f"{k}_boot_mean"], s2[f"{k}_boot_mean"], s1[f"{k}_boot_sd"], s2[f"{k}_boot_sd"]
        if any(math.isnan(v) for v in (m1, m2, d1, d2)):
            if not (math.isnan(m1) and math.isnan(m2)):
                return False
            continue
        sd = max(d1, d2)
        if abs(m1 - m2) > 6 * sd / math.sqrt(B) + 1e-12 or abs(d1 - d2) > 0.1 * sd + 1e-12:
            return False
    return True


def pct(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{d}f}%"


def pct_ci(v, lo, hi, d=1):
    """'58.6% (57.3 to 60.0)'."""
    if v is None or math.isnan(v):
        return "n/a"
    if lo is None or math.isnan(lo):
        return pct(v, d)
    return f"{100 * v:.{d}f}% ({100 * lo:.{d}f} to {100 * hi:.{d}f})"


def md_table(df, formats=None, na="n/a"):
    """A markdown table; formats maps a column to a format string or a function."""
    formats = formats or {}
    cols = list(df.columns)

    def cell(c, v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return na
        f = formats.get(c)
        if callable(f):
            return f(v)
        if f:
            return format(v, f)
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return f"{v:.4g}"
        return str(v).replace("|", "/")

    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(cell(c, v) for c, v in zip(cols, row)) + " |")
    return "\n".join(lines)


class Claims:
    """The claims register: each claim's text is written from the results file it cites, and its check is
    computed from that file."""

    def __init__(self):
        self.rows = []

    def add(self, cid, section, text, sources, holds, note=""):
        if any(r["id"] == cid for r in self.rows):
            raise ValueError(f"duplicate claim id {cid}")
        self.rows.append({"id": cid, "section": section, "claim": text,
                          "sources": ", ".join(sources) if isinstance(sources, (list, tuple)) else sources,
                          "holds": bool(holds), "note": note})

    def frame(self):
        return pd.DataFrame(self.rows, columns=["id", "section", "claim", "sources", "holds", "note"])

    def markdown(self, title, intro):
        df = self.frame()
        parts = [f"# {title}", "", intro, "",
                 f"{int(df.holds.sum())} of {len(df)} checks hold.", ""]
        for section, g in df.groupby("section", sort=False):
            parts += [f"## {section}", "", md_table(g[["id", "claim", "sources", "holds", "note"]]), ""]
        return "\n".join(parts)
