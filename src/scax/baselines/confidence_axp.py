"""Confidence-aware abductive explanations (plan Section 7.1, B8), re-implemented following Chagas,
Cavalcante and Rocha (2026): an AXp that keeps the predicted class and a floor on the model's
confidence everywhere in the box.

For a confidence level c_y on class y, the box must satisfy score > t_1 = max(tau, logit c_1) for
y = 1, or score <= t_0 = min(tau, logit(1 - c_0)) for y = 0. Both are class-preservation conditions
for the same trees with a shifted threshold, so the exact SCAX soundness check and the deletion AXp
of order 0 are reused unchanged. A patient whose own confidence is below c_y gets no explanation.
The claim of such an explanation concerns model confidence; reading it as an outcome-risk claim
(error at most 1 - c_y) is what the comparison tests.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..explain.candidates import CandidateGenerator
from ..models.predictor import ExactPredictor


def _to_score(ens, prob):
    if ens.score == "margin_sum":
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return math.log(prob / (1 - prob))
    return prob


class ConfidenceAXp:
    def __init__(self, ens, conf, node_budget: int = 200000):
        """conf: {0: c_0, 1: c_1}, the confidence floors for negative and positive predictions."""
        self.conf = dict(conf)
        tau = float(ens.tau)
        self.t = {1: max(tau, _to_score(ens, conf[1])), 0: min(tau, _to_score(ens, 1 - conf[0]))}
        self.gen = {}
        for y in (0, 1):
            e = copy.copy(ens)
            e.tau = self.t[y]
            self.gen[y] = CandidateGenerator(ExactPredictor(e), M=1, seed=0, node_budget=node_budget)

    def eligible(self, C, yhat) -> np.ndarray:
        """Whether each patient's own cell meets the confidence floor of its predicted class."""
        out = np.zeros(len(C), dtype=bool)
        for y in (0, 1):
            m = np.asarray(yhat) == y
            if m.any():
                out[m] = self.gen[y].pred.predict_cells(C[m]) == y
        return out

    def explain(self, cell_row, y):
        g = self.gen[int(y)]
        return g.axp(cell_row, int(y), g.orders[0])
