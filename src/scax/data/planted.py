
"""Semi-synthetic planted-rule population (plan Section 6.4).

A fixed population of real BRFSS 2015 covariate rows; labels are generated from a known probability
p(x), so the true risk of any box (Eq. 13) and the true selective risk of any release policy (Eq. 14)
are computed exactly by averaging over the population, with no estimation error.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

RULE_IDS = ("P1", "P2", "P3", "P4")
NOISE_LEVELS = (0.0, 0.05, 0.10, 0.20)
RULE_FEATURES = {
    "P1": {"HighBP", "Age"},
    "P2": {"HighBP", "HighChol", "BMI", "Smoker"},
    "P3": {"GenHlth", "DiffWalk", "Stroke", "Age", "Diabetes", "HighBP", "Sex"},
    "P4": {"HighBP", "HighChol", "BMI", "Age"},
}
P4_TARGET_PREVALENCE = 0.10


def rule_mask(rule: str, X: pd.DataFrame) -> np.ndarray:
    """Positive region of rules P1 to P3 (Age is the BRFSS five-year group, 1 to 13)."""
    if rule == "P1":
        m = (X["HighBP"] == 1) & (X["Age"] >= 10)
    elif rule == "P2":
        m = (X["HighBP"] == 1) & (X["HighChol"] == 1) & ((X["BMI"] >= 30) | (X["Smoker"] == 1))
    elif rule == "P3":
        m = (((X["GenHlth"] >= 4) & (X["DiffWalk"] == 1))
             | ((X["Stroke"] == 1) & (X["Age"] >= 9))
             | ((X["Diabetes"] == 2) & (X["HighBP"] == 1) & (X["Sex"] == 1)))
    else:
        raise ValueError(f"{rule} has no exact rule")
    return np.asarray(m, dtype=bool)


def _p4_linear(X: pd.DataFrame) -> np.ndarray:
    return np.asarray(0.8 * X["HighBP"] + 0.6 * X["HighChol"] + 0.05 * (X["BMI"] - 28)
                      + 0.25 * (X["Age"] - 7) + 0.5 * X["HighBP"] * (X["Age"] >= 10), dtype=float)


def p4_intercept(X_pop: pd.DataFrame, target: float = P4_TARGET_PREVALENCE) -> float:
    """Intercept that makes the population mean of p equal to target (bisection)."""
    lin = _p4_linear(X_pop)
    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(mid + lin)))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def probability(rule: str, X: pd.DataFrame, eps: float = 0.0, p4_b0: float | None = None) -> np.ndarray:
    """True P(Y = 1 | x). Rules P1 to P3: eps + (1 - 2 eps) * 1[rule(x)]. P4: logistic, eps ignored."""
    if rule == "P4":
        if p4_b0 is None:
            raise ValueError("P4 needs the population intercept")
        return 1 / (1 + np.exp(-(p4_b0 + _p4_linear(X))))
    if not 0.0 <= eps < 0.5:
        raise ValueError("eps must lie in [0, 0.5)")
    return eps + (1 - 2 * eps) * rule_mask(rule, X).astype(float)


def make_population(n_rows: int, size: int, seed: int) -> np.ndarray:
    """Row indices of the fixed population, drawn once without replacement from the source table."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, size=size, replace=False))


def draw_with_replacement(pop_size: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices into the population for one split: independent draws from the uniform population law."""
    return rng.integers(0, pop_size, size=n)


def draw_labels(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(len(p)) < p).astype(int)


@dataclass(frozen=True)
class Box:
    """Set of points with lo <= x_j < hi for each constrained feature j; other features free."""
    bounds: dict = field(default_factory=dict)

    def contains(self, X: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(X), dtype=bool)
        for j, (lo, hi) in self.bounds.items():
            col = X[j].to_numpy(dtype=float)
            m &= (col >= lo) & (col < hi)
        return m

    @property
    def size(self) -> int:
        return len(self.bounds)


def true_box_risk(box: Box, y_box: int, X_pop: pd.DataFrame, p_pop: np.ndarray) -> float:
    """Eq. 13: share of the population inside the box whose label differs from y_box, in expectation."""
    m = box.contains(X_pop)
    if not m.any():
        return float("nan")
    r = (1 - p_pop[m]) if y_box == 1 else p_pop[m]
    return float(r.mean())


def true_selective_risk(accepted: np.ndarray, pred: np.ndarray, p_pop: np.ndarray) -> float:
    """Eq. 14: expected error rate over every accepted population point."""
    accepted = np.asarray(accepted, dtype=bool)
    if not accepted.any():
        return float("nan")
    err = np.where(np.asarray(pred)[accepted] == 1, 1 - p_pop[accepted], p_pop[accepted])
    return float(err.mean())
