"""Baseline comparisons (plan Sections 6.8 and 7.1, RQ3).

Real data. The frozen model, prior and candidate sets are kept; calibration splits 1 and 2 are pooled
and re-partitioned at random into new splits of the same sizes (re-draws). The explained patients of
the old split 2 that fall into the new split 2 are a uniform random subsample of it, because the
splits and the subsample were drawn at random without labels. Every method then uses the same
labelled records: the new split 1 and the explained patients of the new split 2. All methods are
evaluated on the same 5,000 explained test patients; box claims are checked against the full test
split, which no method sees.

Synthetic data. On the planted-rule population the true risk of every box and rule is known, so each
method's coverage can be set against its true claim precision (share of released patients whose
explanation's claim is true).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
from scipy import sparse

from ..baselines import rule_dictionary as rd
from ..baselines.selective import conformal_singletons, ltt, sgr
from ..certify.certificate import upper_bounds
from ..certify.clopper_pearson import cp_lower
from ..certify.counting import box_counts
from ..certify.decision import certify_policy_by_class, release
from ..data import planted, splits
from ..explain.verify import verify_boxes

KINDS = (0, 1)


def widest_order0(cands, K):
    """Order-0 candidate with the largest relative volume (the end of the inflation path)."""
    best, best_v = None, -math.inf
    for c in cands:
        if c.order != 0:
            continue
        v = sum(math.log((b - a + 1) / (K[j] + 1)) for j, a, b in c.key)
        if v > best_v:
            best, best_v = c, v
    return best


def axp_order0(cands):
    for c in cands:
        if c.order == 0 and c.kind == "axp":
            return c
    return None


def released_metrics(rel, kind_of_box, yhat, y, size, n_cal, n_test, k_test, alpha, n_patients):
    """Per kind: coverage, errors, size, support, and claims refuted on the full test split.

    A released box's claim (error at most alpha) is refuted when the exact lower 95% limit of its test
    error, Bonferroni-corrected over the distinct released boxes of that kind, is above alpha.
    """
    out = {}
    for k in KINDS:
        a = (rel >= 0) & (yhat == k)
        rb = rel[a]
        acc = int(a.sum())
        err = int((y[a] != k).sum())
        row = {"accepted": acc, "coverage": acc / n_patients, "errors": err,
               "error_rate": err / acc if acc else np.nan}
        if size is not None:
            ids = np.unique(rb)
            nt, kt = n_test[ids], k_test[ids]
            low = np.zeros(len(ids))
            ok = nt > 0
            if ok.any():
                low[ok] = cp_lower(kt[ok], nt[ok], 0.05 / max(1, len(ids)))
            refuted = set(ids[low > alpha[k]].tolist())
            with np.errstate(invalid="ignore", divide="ignore"):
                est = np.where(n_test[rb] > 0, k_test[rb] / np.maximum(n_test[rb], 1), np.nan)
            row.update({"mean_size": float(size[rb].mean()) if acc else np.nan,
                        "median_support": float(np.median(n_cal[rb])) if acc else np.nan,
                        "distinct_boxes": int(len(ids)),
                        "refuted_share": float(np.mean([b in refuted for b in rb])) if acc else 0.0,
                        "refuted_boxes": len(refuted),
                        "above_target_share": float(np.nanmean(est > alpha[k])) if acc else 0.0})
        out[k] = row
    return out


class RealComparison:
    """Everything fixed for one task before the re-draws."""

    def __init__(self, ft, box_df, gen, Xa, y, code, delta1, delta2):
        self.ft, self.g = ft, ft.grid
        self.delta1, self.delta2 = delta1, delta2
        self.alpha = {0: ft.targets["alpha_0"], 1: ft.targets["alpha_1"]}
        self.alpha_sel = {0: ft.targets["alpha_sel"]["negative_outcome"], 1: ft.targets["alpha_sel"]["enrichment"]}
        self.keys = [tuple(tuple(t) for t in json.loads(k)) for k in box_df.key]
        self.cls = box_df.box_class.to_numpy().astype(int)
        self.size = box_df["size"].to_numpy()
        self.log_pi = box_df.log_prior.to_numpy()
        v = box_df.verified.to_numpy()
        self.status = np.where(v == "sound", 1, np.where(v == "unknown", -1, 0)).astype(np.int8)
        box_index = {(k, int(c)): i for i, (k, c) in enumerate(zip(self.keys, self.cls))}
        cal1, cal2 = splits.indices(code, "cal1"), splits.indices(code, "cal2")
        self.pool = np.sort(np.r_[cal1, cal2])
        self.n_cal1 = len(cal1)
        self.X_pool = Xa[self.pool]
        self.C_pool = self.g.cells(self.X_pool)
        self.y_pool = np.asarray(y)[self.pool]
        self.margin_pool = ft.pred.score(self.X_pool)
        self.yhat_pool = ft.pred.predict_cells(self.C_pool).astype(int)
        E, T = ft.cal2_rows, ft.test_rows
        self.E_pos = np.searchsorted(self.pool, E)
        assert np.array_equal(self.pool[self.E_pos], E), "explained calibration patients must lie in the pool"
        CE, CT = self.g.cells(Xa[E]), self.g.cells(Xa[T])
        listsE = [gen.cache[gen.cell_key(c)] for c in CE]
        listsT = [gen.cache[gen.cell_key(c)] for c in CT]
        M = ft.M

        def pairs(lists):
            pp, pb = [], []
            for i, cands in enumerate(lists):
                for c in cands:
                    if c.order < M and (c.key, c.y) in box_index:
                        pp.append(i); pb.append(box_index[(c.key, c.y)])
            return np.array(pp, dtype=np.int64), np.array(pb, dtype=np.int64)

        self.ppE, self.pbE = pairs(listsE)
        self.ppT, self.pbT = pairs(listsT)
        self.yhatE = ft.pred.predict_cells(CE).astype(int)
        self.yhatT = ft.pred.predict_cells(CT).astype(int)
        self.yE, self.yT = np.asarray(y)[E], np.asarray(y)[T]
        self.XT, self.CT = Xa[T], CT
        self.marginT = ft.pred.score(Xa[T])
        K = self.g.K
        self.b1 = np.array([box_index[(c.key, c.y)] for c in (axp_order0(l) for l in listsT)])
        self.b2 = np.array([box_index[(c.key, c.y)] for c in (widest_order0(l, K) for l in listsT)])
        test_all = splits.indices(code, "test")
        self.X_test_all, self.y_test_all = Xa[test_all], np.asarray(y)[test_all]
        self.n_test, self.k_test = box_counts(self.keys, self.cls, self.g.cells(self.X_test_all), self.y_test_all)
        self.rules = None

    # ------------------------------------------------------------------ rule dictionary (B6)
    def set_rules(self, rules, oracle):
        self.rules = rules
        space = rd.RuleSpace(rules, self.g.d)
        self.r_keys = [space.key(r) for r in rules for _ in KINDS]
        self.r_cls = np.array([c for _ in rules for c in KINDS])
        self.r_size = np.array([r.size for r in rules for _ in KINDS])
        self.r_text = [r.text for r in rules for _ in KINDS]
        self.r_log_pi = np.full(len(self.r_keys), -math.log(len(self.r_keys)))
        self.r_sound = np.array([oracle.sound(rd.outer_box(r, self.g), c) for r in rules for c in KINDS])
        self.r_C_pool = space.cells(self.X_pool)

        def pairs(X, yhat):
            pp, pb = [], []
            for ri, r in enumerate(rules):
                idx = np.flatnonzero(rd.contains(r, X))
                pp.append(idx); pb.append(2 * ri + yhat[idx])
            return np.concatenate(pp).astype(np.int64), np.concatenate(pb).astype(np.int64)

        XE = self.X_pool[self.E_pos]
        self.r_ppE, self.r_pbE = pairs(XE, self.yhatE)
        self.r_ppT, self.r_pbT = pairs(self.XT, self.yhatT)
        self.r_n_test, self.r_k_test = box_counts(self.r_keys, self.r_cls, space.cells(self.X_test_all), self.y_test_all)

    # ------------------------------------------------------------------ one re-draw
    def split(self, seed):
        rng = np.random.default_rng(seed)
        in1 = np.zeros(len(self.pool), dtype=bool)
        in1[rng.permutation(len(self.pool))[: self.n_cal1]] = True
        return in1

    def _subset_pairs(self, pp, pb, keep_patients):
        idx = np.flatnonzero(keep_patients)
        remap = np.full(len(keep_patients), -1, dtype=np.int64); remap[idx] = np.arange(len(idx))
        m = keep_patients[pp]
        return remap[pp[m]], pb[m], idx

    def _policy(self, rel, yhat, y):
        err = {k: int(((rel >= 0) & (yhat == k) & (y != k)).sum()) for k in KINDS}
        acc = {k: int(((rel >= 0) & (yhat == k)).sum()) for k in KINDS}
        return certify_policy_by_class(err, acc, self.alpha_sel, self.delta2)

    def run(self, r, seed, log=print):
        in1 = self.split(seed)
        n, k = box_counts(self.keys, self.cls, self.C_pool[in1], self.y_pool[in1])
        U_scax = upper_bounds(k, n, math.log(self.delta1) + self.log_pi)
        a_box = np.where(self.cls == 1, self.alpha[1], self.alpha[0])
        need = np.flatnonzero((U_scax <= a_box) & (self.status == 0))
        if len(need):
            verdicts, _ = verify_boxes(self.g, [(self.keys[i], int(self.cls[i])) for i in need])
            self.status[need] = [1 if verdicts[(self.keys[i], int(self.cls[i]))] == "sound" else -1 for i in need]
        U = {"SCAX": np.where(self.status == 1, U_scax, 1.0),
             "Per-box bound without prior": upper_bounds(k, n, math.log(self.delta1)),
             "Empirical error rule": np.where(n > 0, k / np.maximum(n, 1), 1.0)}
        e_in2 = ~in1[self.E_pos]
        ppE, pbE, idxE = self._subset_pairs(self.ppE, self.pbE, e_in2)
        yhatE, yE = self.yhatE[idxE], self.yE[idxE]
        rows, rel_scax = [], None
        nT = len(self.yhatT)

        def add(method, met, pol=None, extra=None):
            for kk in KINDS:
                row = {"redraw": r, "method": method, "kind": kk, **met[kk]}
                if pol is not None:
                    row.update({"cal2_accepted": pol[kk]["accepted"], "cal2_errors": pol[kk]["errors"],
                                "U_sel": pol[kk]["U_sel"], "alpha_sel": pol[kk]["alpha_sel"],
                                "policy_certified": pol[kk]["certified"]})
                if extra:
                    row.update(extra[kk] if isinstance(extra, dict) and kk in extra else extra)
                rows.append(row)

        for name, Um in U.items():
            relE = release(ppE, pbE, len(idxE), self.size, self.cls, n, Um, self.alpha)
            relT = release(self.ppT, self.pbT, nT, self.size, self.cls, n, Um, self.alpha)
            met = released_metrics(relT, self.cls, self.yhatT, self.yT, self.size, n, self.n_test, self.k_test, self.alpha, nT)
            add(name, met, self._policy(relE, yhatE, yE))
            if name == "SCAX":
                rel_scax, met_scax = relT, met
        # uncertified explanations, full and matched coverage
        for name, boxes in (("Uncertified AXp", self.b1), ("Uncertified inflated box", self.b2)):
            met = released_metrics(boxes.copy(), self.cls, self.yhatT, self.yT, self.size, n, self.n_test, self.k_test, self.alpha, nT)
            add(f"{name}, all patients", met)
            relm = np.full(nT, -1, dtype=np.int64)
            for kk in KINDS:
                cand = np.flatnonzero(self.yhatT == kk)
                conf = self.marginT[cand] if kk == 1 else -self.marginT[cand]
                top = cand[np.argsort(-conf, kind="stable")[: met_scax[kk]["accepted"]]]
                relm[top] = boxes[top]
            met = released_metrics(relm, self.cls, self.yhatT, self.yT, self.size, n, self.n_test, self.k_test, self.alpha, nT)
            add(f"{name}, matched coverage", met)
        # rule dictionary (B6), certified with the same rule and level
        if self.rules is not None:
            rn, rk = box_counts(self.r_keys, self.r_cls, self.r_C_pool[in1], self.y_pool[in1])
            rU = upper_bounds(rk, rn, math.log(self.delta1) + self.r_log_pi)
            rppE, rpbE, ridx = self._subset_pairs(self.r_ppE, self.r_pbE, e_in2)
            for name, Ur in (("Rule dictionary", rU), ("Rule dictionary, sound rules only", np.where(self.r_sound, rU, 1.0))):
                relE = release(rppE, rpbE, len(ridx), self.r_size, self.r_cls, rn, Ur, self.alpha)
                relT = release(self.r_ppT, self.r_pbT, nT, self.r_size, self.r_cls, rn, Ur, self.alpha)
                met = released_metrics(relT, self.r_cls, self.yhatT, self.yT, self.r_size, rn, self.r_n_test, self.r_k_test, self.alpha, nT)
                sound = {kk: {"sound_share": float(self.r_sound[relT[(relT >= 0) & (self.yhatT == kk)]].mean())
                              if ((relT >= 0) & (self.yhatT == kk)).any() else np.nan} for kk in KINDS}
                add(name, met, self._policy(relE, self.yhatE[ridx], self.yE[ridx]), sound)
        # prediction-only methods on the same labelled records
        lab = in1.copy(); lab[self.E_pos[e_in2]] = True
        m_l, y_l, yh_l = self.margin_pool[lab], self.y_pool[lab], self.yhat_pool[lab]
        for name, fn in (("SGR", sgr), ("Learn-then-Test", ltt)):
            relp = np.full(nT, -1, dtype=np.int64)
            extra = {}
            for kk in KINDS:
                sel = yh_l == kk
                conf = m_l[sel] if kk == 1 else -m_l[sel]
                res = fn(conf, (y_l[sel] != kk).astype(int), self.alpha_sel[kk], self.delta2 / 2)
                tsel = self.yhatT == kk
                tconf = self.marginT if kk == 1 else -self.marginT
                relp[tsel & (tconf >= res["theta"])] = 0
                extra[kk] = {"threshold": res["theta"], "cal_accepted": res["accepted"]}
            add(name, released_metrics(relp, None, self.yhatT, self.yT, None, None, None, None, self.alpha, nT), extra=extra)
        p_l, pT = 1 / (1 + np.exp(-m_l)), 1 / (1 + np.exp(-self.marginT))
        for name, mond in (("Split conformal, singletons", False), ("Mondrian conformal, singletons", True)):
            single, q = conformal_singletons(p_l, y_l, pT, 0.05, mondrian=mond)
            relc = np.where(single >= 0, 0, -1)
            add(name, released_metrics(relc, None, single, self.yT, None, None, None, None, self.alpha, nT))
        return pd.DataFrame(rows), int(len(need))


def paired_tests(df, metric, reference="SCAX", higher_is_better=True):
    """Paired Wilcoxon signed-rank tests across re-draws, Holm correction across baselines."""
    from scipy.stats import wilcoxon
    out = []
    for (task, kind), g in df.groupby(["task", "kind"]):
        piv = g.pivot_table(index="redraw", columns="method", values=metric)
        if reference not in piv:
            continue
        rows = []
        for m in piv.columns:
            if m == reference:
                continue
            d = (piv[reference] - piv[m]).dropna()
            if len(d) == 0:
                continue
            p = 1.0 if np.allclose(d, 0) else float(wilcoxon(d, zero_method="wilcox").pvalue)
            rows.append({"task": task, "kind": kind, "metric": metric, "baseline": m, "units": len(d),
                         "median_difference": float(np.median(d)), "p": p,
                         "reference_mean": float(piv[reference].mean()), "baseline_mean": float(piv[m].mean())})
        rows.sort(key=lambda r: r["p"])
        k = len(rows)
        running = 0.0
        for i, r in enumerate(rows):
            running = max(running, min(1.0, (k - i) * r["p"]))
            r["p_holm"] = running
        out += rows
    return pd.DataFrame(out)


# ====================================================================== synthetic population
class SyntheticComparison:
    """Coverage against true claim precision on one planted-rule setting (Setting from validity.py)."""

    def __init__(self, st, M, delta1, prior="mix0.5", alpha=None):
        self.st, self.M, self.delta1 = st, M, delta1
        self.alpha = dict(st.alpha if alpha is None else alpha)
        self.pm = st.pair_order < M
        self.log_pi = st.prior_logs[prior](M)
        self.pos_cell = np.bincount(st.cell_of_row, weights=st.p, minlength=st.n_cells)
        self.kappa_row = st.kappa_cell[st.cell_of_row]
        self.err_row = np.where(self.kappa_row == 1, 1 - st.p, st.p)
        self.box_alpha = np.where(st.box_class == 1, self.alpha[1], self.alpha[0])
        K = st.grid.K
        b1 = [axp_order0(st.cache[k]) for k in st.cell_keys]
        b2 = [widest_order0(st.cache[k], K) for k in st.cell_keys]
        self.b1 = np.array([st.box_index[(c.key, c.y)] for c in b1])
        self.b2 = np.array([st.box_index[(c.key, c.y)] for c in b2])
        self.fixed = {}

    def _cells_summary(self, rel_cell, R_true, alpha_of):
        st = self.st
        acc = rel_cell >= 0
        w = st.cell_count[acc]
        n_acc = float(w.sum())
        if n_acc == 0:
            return {"coverage": 0.0, "claim_precision": np.nan, "claim_violation": False, "true_selective_risk": np.nan}
        ok = R_true[rel_cell[acc]] <= alpha_of[rel_cell[acc]] + 1e-12
        return {"coverage": n_acc / st.N, "claim_precision": float((w * ok).sum() / n_acc),
                "claim_violation": bool((~ok).any()), "true_selective_risk": float(st.cell_err[acc].sum() / n_acc)}

    def add_confidence_axp(self, caxp):
        st = self.st
        elig = caxp.eligible(st.C_cells, st.kappa_cell)
        keys, Rt, cls = [], [], []
        rel = np.full(st.n_cells, -1, dtype=np.int64)
        for ci in np.flatnonzero(elig):
            y = int(st.kappa_cell[ci])
            key = st.grid.key(caxp.explain(st.C_cells[ci], y))
            m = st.grid.contains(key, st.C_cells)
            n_pop, pos = st.cell_count[m].sum(), self.pos_cell[m].sum()
            rel[ci] = len(keys); keys.append(key); cls.append(y)
            Rt.append((n_pop - pos) / n_pop if y == 1 else pos / n_pop)
        Rt, cls = np.array(Rt), np.array(cls, dtype=int)
        alpha_of = np.where(cls == 1, self.alpha[1], self.alpha[0])
        self.fixed["Confidence-aware AXp"] = {**self._cells_summary(rel, Rt, alpha_of),
                                              "mean_size": float(np.average([len(keys[i]) for i in rel[rel >= 0]],
                                                                            weights=st.cell_count[rel >= 0])) if (rel >= 0).any() else np.nan}

    def add_rules(self, rules, rule_sound):
        """rules: list of Rule; rule_sound: bool per (rule, class) entry, ordered rule-major."""
        st = self.st
        X = st.XP.to_numpy(dtype=float)
        mem = np.array([rd.contains(r, X) for r in rules])               # rules x rows
        self.r_mem = sparse.csr_matrix(np.repeat(mem, 2, axis=0).astype(np.float64))
        self.r_cls = np.tile(np.array(KINDS), len(rules))
        self.r_size = np.repeat([r.size for r in rules], 2)
        n_pop = self.r_mem @ np.ones(st.N); pos = self.r_mem @ st.p
        self.r_R = np.where(self.r_cls == 1, (n_pop - pos) / n_pop, pos / n_pop)
        self.r_alpha = np.where(self.r_cls == 1, self.alpha[1], self.alpha[0])
        self.r_log_pi = np.full(len(self.r_cls), -math.log(len(self.r_cls)))
        self.r_sound = np.asarray(rule_sound, dtype=bool)
        rows, ents = [], []
        for ri in range(len(rules)):
            idx = np.flatnonzero(mem[ri])
            rows.append(idx); ents.append(2 * ri + self.kappa_row[idx])
        self.r_pp, self.r_pb = np.concatenate(rows), np.concatenate(ents)

    def _rows_summary(self, rel_row, R, alpha_of):
        acc = rel_row >= 0
        if not acc.any():
            return {"coverage": 0.0, "claim_precision": np.nan, "claim_violation": False, "true_selective_risk": np.nan}
        ok = R[rel_row[acc]] <= alpha_of[rel_row[acc]] + 1e-12
        return {"coverage": float(acc.mean()), "claim_precision": float(ok.mean()), "claim_violation": bool((~ok).any()),
                "true_selective_risk": float(self.err_row[acc].mean())}

    def run(self, R, n1, seed):
        st = self.st
        rng = np.random.default_rng(seed)
        rows = []
        for name, b in (("Uncertified AXp", self.b1), ("Uncertified inflated box", self.b2)):
            s = self._cells_summary(b, st.R_true, self.box_alpha)
            self.fixed[name] = {**s, "mean_size": float(np.average(st.box_size[b], weights=st.cell_count))}
        pc, pb = st.pair_cell[self.pm], st.pair_box[self.pm]
        for r in range(R):
            idx = planted.draw_with_replacement(st.N, n1, rng)
            yy = planted.draw_labels(st.p[idx], rng)
            c_row = np.bincount(idx, minlength=st.N).astype(float)
            pos_row = np.bincount(idx, weights=yy, minlength=st.N)
            c_cell = np.bincount(st.cell_of_row, weights=c_row, minlength=st.n_cells)
            p_cell = np.bincount(st.cell_of_row, weights=pos_row, minlength=st.n_cells)
            n_B = np.rint(st.mem @ c_cell).astype(np.int64); pos_B = np.rint(st.mem @ p_cell).astype(np.int64)
            k_B = np.where(st.box_class == 1, n_B - pos_B, pos_B)
            Us = {"SCAX": upper_bounds(k_B, n_B, math.log(self.delta1) + self.log_pi),
                  "Per-box bound without prior": upper_bounds(k_B, n_B, math.log(self.delta1)),
                  "Empirical error rule": np.where(n_B > 0, k_B / np.maximum(n_B, 1), 1.0)}
            for name, U in Us.items():
                rel = release(pc, pb, st.n_cells, st.box_size, st.box_class, n_B, U, self.alpha)
                s = self._cells_summary(rel, st.R_true, self.box_alpha)
                acc = rel >= 0
                s["mean_size"] = float(np.average(st.box_size[rel[acc]], weights=st.cell_count[acc])) if acc.any() else np.nan
                rows.append({"rep": r, "method": name, **s})
            if hasattr(self, "r_mem"):
                rn = np.rint(self.r_mem @ c_row).astype(np.int64); rp = np.rint(self.r_mem @ pos_row).astype(np.int64)
                rk = np.where(self.r_cls == 1, rn - rp, rp)
                rU = upper_bounds(rk, rn, math.log(self.delta1) + self.r_log_pi)
                for name, U in (("Rule dictionary", rU), ("Rule dictionary, sound rules only", np.where(self.r_sound, rU, 1.0))):
                    rel = release(self.r_pp, self.r_pb, st.N, self.r_size, self.r_cls, rn, U, self.alpha)
                    s = self._rows_summary(rel, self.r_R, self.r_alpha)
                    s["mean_size"] = float(self.r_size[rel[rel >= 0]].mean()) if (rel >= 0).any() else np.nan
                    rows.append({"rep": r, "method": name, **s})
        df = pd.DataFrame(rows)
        for name, s in self.fixed.items():
            df = pd.concat([df, pd.DataFrame([{"rep": -1, "method": name, **s}])], ignore_index=True)
        return df
