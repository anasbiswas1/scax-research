"""Subgroups (plan Section 7.6, E5, RQ5; Definition 11, Eq. 11, Corollary 6).

One fixed model per task, the frozen one, with its candidate sets, box table, explained subsamples
and targets. Groups: sex; age bands 18 to 44, 45 to 64, 65 or older; income tertiles where the table
has income. Race and ethnicity are not in any of the three public tables.

Methods
- Marginal SCAX: the frozen release, bound U(B) from all of calibration split 1.
- Group-conditional SCAX, standard: for one attribute with groups G,
  U(B, g) = U(k_Bg, n_Bg; delta_1 pi(B) / |G|); a patient in group g receives the first box in the
  frozen order whose U(B, g) meets the class target.
- Group-conditional SCAX, strict: all attributes together, rho(g) = 1 / (number of groups over all
  attributes); a patient receives B only if U(B, g) meets the target for every one of the patient's
  groups.
- SGR (prediction only): one threshold per kind, and per kind and group.

The release order is the frozen one (fewest constrained features, most calibration support counted
on all of calibration split 1, tightest bound), with the group bound in place of the marginal bound.
A box can be released only if the second exact check has confirmed it.
"""
from __future__ import annotations

import math

import numpy as np

from ..certify.certificate import upper_bounds
from ..certify.clopper_pearson import cp_lower, cp_upper
from ..certify.counting import box_counts

KIND_NAMES = {0: "negative outcome", 1: "enrichment"}
ATTRIBUTE_NAMES = {"sex": "Sex", "age": "Age band", "income": "Income tertile"}
SEX_LABELS = ["Female", "Male"]
AGE_LABELS = ["Age 18 to 44", "Age 45 to 64", "Age 65 or older"]
BRFSS_INCOME = {1: "under $10,000", 2: "$10,000 to $14,999", 3: "$15,000 to $19,999", 4: "$20,000 to $24,999",
                5: "$25,000 to $34,999", 6: "$35,000 to $49,999", 7: "$50,000 to $74,999", 8: "$75,000 or more"}
BRFSS_INCOME_LOW = {1: 0, 2: 10000, 3: 15000, 4: 20000, 5: 25000, 6: 35000, 7: 50000, 8: 75000}


def income_cuts(income_codes):
    """Two cut codes c1 < c2 so that codes <= c1, c1 < codes <= c2 and codes > c2 come closest to thirds
    (income is coded in 8 bands, so exact tertiles are not possible)."""
    v = np.asarray(income_codes, dtype=float)
    v = v[~np.isnan(v)]
    codes = np.unique(v)
    share = np.array([(v <= c).mean() for c in codes])
    best, cuts = math.inf, None
    for i in range(len(codes)):
        for j in range(i + 1, len(codes) - 1):
            sizes = np.array([share[i], share[j] - share[i], 1 - share[j]])
            dev = float(np.abs(sizes - 1 / 3).max())
            if dev < best - 1e-12:
                best, cuts = dev, (int(codes[i]), int(codes[j]))
    if cuts is None:
        raise ValueError("income needs at least three codes")
    return cuts


def income_labels(cuts):
    c1, c2 = cuts
    lo2, lo3 = BRFSS_INCOME_LOW[c1 + 1], BRFSS_INCOME_LOW[c2 + 1]
    return [f"Income under ${lo2:,}", f"Income ${lo2:,} to ${lo3 - 1:,}", f"Income ${lo3:,} or more"]


def groups_for(task, X, names, cuts=None):
    """Group code per record (0, 1, ...; -1 outside every group) and labels, per attribute."""
    X = np.asarray(X, dtype=float)
    col = {n: X[:, i] for i, n in enumerate(names)}
    out = {}
    if task == "cardio70k":
        sex = np.where(col["gender"] == 1, 0, np.where(col["gender"] == 2, 1, -1))   # dataset card: 1 women, 2 men
        years = col["age"] / 365.25
        age = np.where(years < 18, -1, np.where(years < 45, 0, np.where(years < 65, 1, 2)))
        out["sex"] = (sex.astype(np.int64), SEX_LABELS)
        out["age"] = (age.astype(np.int64), AGE_LABELS)
        return out
    sex = np.where(col["Sex"] == 0, 0, np.where(col["Sex"] == 1, 1, -1))                 # BRFSS SEX recoded: 0 female, 1 male
    a = col["Age"]                                                                     # _AGEG5YR 1 (18-24) to 13 (80+)
    age = np.where((a >= 1) & (a <= 5), 0, np.where((a >= 6) & (a <= 9), 1, np.where((a >= 10) & (a <= 13), 2, -1)))
    out["sex"] = (sex.astype(np.int64), SEX_LABELS)
    out["age"] = (age.astype(np.int64), AGE_LABELS)
    if cuts is not None:
        inc = col["Income"]
        c1, c2 = cuts
        g = np.where(np.isnan(inc), -1, np.where(inc <= c1, 0, np.where(inc <= c2, 1, 2)))
        out["income"] = (g.astype(np.int64), income_labels(cuts))
    return out


def drop_empty(groups, present):
    """Keep only groups with at least one record in `present` (codes renumbered, labels kept in order)."""
    out = {}
    for attr, (codes, labels) in groups.items():
        pc = present[attr][0]
        keep = [g for g in range(len(labels)) if (pc == g).any()]
        remap = np.full(len(labels) + 1, -1, dtype=np.int64)
        for new, old in enumerate(keep):
            remap[old] = new
        out[attr] = (np.where(codes >= 0, remap[np.clip(codes, 0, None)], -1), [labels[g] for g in keep])
    return out


def group_counts(keys, cls, C, y, codes, n_groups):
    """n_Bg and k_Bg for every box and group (groups x boxes)."""
    n = np.zeros((n_groups, len(keys)), dtype=np.int64)
    k = np.zeros((n_groups, len(keys)), dtype=np.int64)
    for g in range(n_groups):
        m = np.asarray(codes) == g
        if m.any():
            n[g], k[g] = box_counts(keys, cls, np.asarray(C)[m], np.asarray(y)[m])
    return n, k


def group_bounds(n, k, log_pi, delta1, rho):
    """U(B, g) = U(k_Bg, n_Bg; delta_1 pi(B) rho) for every group (groups x boxes)."""
    lg = math.log(delta1) + np.asarray(log_pi, dtype=float) + math.log(rho)
    return np.vstack([upper_bounds(k[g], n[g], lg) for g in range(n.shape[0])])


def pair_bound(U_g, codes, pp, pb):
    """Bound of each (patient, box) pair under the patient's group; 1 for patients outside every group."""
    c = np.asarray(codes)[pp]
    return np.where(c >= 0, U_g[np.clip(c, 0, None), pb], 1.0)


def release_pairs(pp, pb, U_pair, n_patients, size, cls, n_B, alpha):
    """The frozen release rule with a bound per (patient, box) pair (Eq. 8 with U(B, g) in place of U(B))."""
    pp, pb, U_pair = np.asarray(pp), np.asarray(pb), np.asarray(U_pair, dtype=float)
    a = np.where(np.asarray(cls)[pb] == 1, alpha[1], alpha[0])
    ok = U_pair <= a
    out = np.full(n_patients, -1, dtype=np.int64)
    if not ok.any():
        return out
    p, b, u = pp[ok], pb[ok], U_pair[ok]
    order = np.lexsort((u, -np.asarray(n_B)[b], np.asarray(size)[b], p))
    p, b = p[order], b[order]
    first = np.r_[True, p[1:] != p[:-1]]
    out[p[first]] = b[first]
    return out


def needs_check(U_by_attr, cls, alpha, verified_sound):
    """Boxes that some group bound would release but the second check has not yet confirmed."""
    a = np.where(np.asarray(cls) == 1, alpha[1], alpha[0])
    hit = np.zeros(len(cls), dtype=bool)
    for U_g in U_by_attr:
        hit |= (U_g <= a[None, :]).any(axis=0)
    return np.flatnonzero(hit & ~np.asarray(verified_sound))


def group_metrics(rel, yhat, y, codes, labels, kind, size=None):
    """Per group (and 'All'): patients, accepted, certified share, errors, error rate with one-sided 95% bounds."""
    rel, yhat, y, codes = np.asarray(rel), np.asarray(yhat), np.asarray(y), np.asarray(codes)
    rows = []
    for g, lab in [(-2, "All")] + list(enumerate(labels)):
        m = np.ones(len(rel), dtype=bool) if g == -2 else codes == g
        a = m & (rel >= 0) & (yhat == kind)
        n_acc, e = int(a.sum()), int((y[a] != kind).sum())
        rows.append({"group": lab, "patients": int(m.sum()), "predicted_class": int((m & (yhat == kind)).sum()),
                     "accepted": n_acc, "certified_share": float(n_acc / m.sum()) if m.any() else float("nan"),
                     "certified_share_of_predicted_class": float(n_acc / max(1, (m & (yhat == kind)).sum())),
                     "errors": e, "error_rate": e / n_acc if n_acc else float("nan"),
                     "error_lower_95": float(cp_lower(e, n_acc, 0.05)) if n_acc else 0.0,
                     "error_upper_95": float(cp_upper(e, n_acc, 0.05)) if n_acc else 1.0,
                     "mean_size": float(np.mean(np.asarray(size)[rel[a]])) if (size is not None and n_acc) else None})
    return rows


def group_policy(rel, yhat, y, codes, n_groups, kind, alpha_sel, delta2, n_tests):
    """Policy test of one kind in every group at level delta_2 / (2 n_tests)."""
    rel, yhat, y, codes = np.asarray(rel), np.asarray(yhat), np.asarray(y), np.asarray(codes)
    level = delta2 / (2 * n_tests)
    out = []
    for g in range(n_groups):
        a = (codes == g) & (rel >= 0) & (yhat == kind)
        n_acc, e = int(a.sum()), int((y[a] != kind).sum())
        u = float(cp_upper(e, n_acc, level)) if n_acc else 1.0
        out.append({"cal2_accepted": n_acc, "cal2_errors": e, "U_sel_group": u, "policy_level": level,
                    "policy_certified_in_group": bool(n_acc > 0 and u <= alpha_sel)})
    return out


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe_difference(k1, n1, k2, n2):
    """Difference p1 - p2 with Newcombe's hybrid score 95% interval (method 10)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi
