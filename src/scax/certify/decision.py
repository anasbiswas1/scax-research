"""Release rule (plan Definition 8, Eq. 8) and policy certification (Definition 10, Eq. 9).

A patient's candidate boxes that pass their class target, U_hat(B) <= alpha_y, form A(x). SCAX
releases the lexicographic minimum by (fewest constrained features, most calibration support,
tightest bound), and abstains when A(x) is empty. Candidate sets are given as flat pair arrays
(patient, box) so the rule runs vectorised over all patients.
"""
from __future__ import annotations

import numpy as np

from .clopper_pearson import cp_upper


def release(pair_patient, pair_box, n_patients, box_size, box_class, n_B, U, alpha):
    """Released box index per patient (-1 = abstain). alpha: dict class -> target."""
    pair_patient = np.asarray(pair_patient); pair_box = np.asarray(pair_box)
    a = np.where(np.asarray(box_class)[pair_box] == 1, alpha[1], alpha[0])
    ok = U[pair_box] <= a
    out = np.full(n_patients, -1, dtype=np.int64)
    if not ok.any():
        return out
    pp, pb = pair_patient[ok], pair_box[ok]
    order = np.lexsort((U[pb], -n_B[pb], np.asarray(box_size)[pb], pp))
    pp, pb = pp[order], pb[order]
    first = np.r_[True, pp[1:] != pp[:-1]]
    out[pp[first]] = pb[first]
    return out


def certify_policy(errors: int, accepted: int, alpha_sel: float, delta_2: float) -> dict:
    u = float(cp_upper(errors, accepted, delta_2)) if accepted > 0 else 1.0
    return {"accepted": int(accepted), "errors": int(errors), "U_sel": u, "alpha_sel": alpha_sel,
            "delta_2": delta_2, "certified": bool(u <= alpha_sel)}


def certify_policy_by_class(errors: dict, accepted: dict, alpha_sel: dict, delta_2: float) -> dict:
    """Separate policy tests for negative-outcome releases (class 0) and enrichment releases (class 1).

    One error rate over both kinds would mix a 5% claim with a 70% claim, so each kind is certified on
    its own at delta_2 / 2; by the union bound both statements hold together with probability at least
    1 - delta_2.
    """
    return {y: certify_policy(errors[y], accepted[y], alpha_sel[y], delta_2 / 2) for y in (0, 1)}
