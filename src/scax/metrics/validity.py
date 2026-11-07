
"""Three-way validity rule of plan Section 6.8.

Upper one-sided limit at or below delta: violation rate low enough. Lower one-sided limit above
delta: excess violations. Otherwise inconclusive. With K configurations, the excess test uses level
alpha / K so that one configuration slightly above delta by chance is not read as a failure.
"""
from ..certify.clopper_pearson import cp_lower, cp_upper


def validity_verdict(violations: int, repetitions: int, delta: float, alpha: float = 0.05, k_configs: int = 1) -> dict:
    upper = cp_upper(violations, repetitions, alpha)
    lower = cp_lower(violations, repetitions, alpha / k_configs)
    if upper <= delta:
        verdict = "low enough"
    elif lower > delta:
        verdict = "excess"
    else:
        verdict = "inconclusive"
    return {"violations": int(violations), "repetitions": int(repetitions), "rate": violations / repetitions,
            "upper": float(upper), "lower": float(lower), "delta": delta, "verdict": verdict}
