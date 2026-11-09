"""Export a trained tree ensemble to SCAX's own exact format (plan Section 5.3).

The export fixes, for every tree, each node's feature, the threshold exactly as the library stores
it, the comparison rule and the leaf values. Floats are written with float.hex(), so the file is an
exact record. The library is used for training only; the exported classifier is what SCAX
certifies and deploys.

Rules: XGBoost stores float32 thresholds and routes x < threshold to the left child ("lt").
LightGBM stores float64 thresholds and leaf values and routes x <= threshold left ("le").
scikit-learn stores float64 thresholds, casts inputs to float32 and routes x <= threshold left.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

LEAF = -1


@dataclass
class Tree:
    feature: list    # node feature index, LEAF (-1) at leaves
    threshold: list  # float (exact stored value) at internal nodes, 0.0 at leaves
    left: list
    right: list
    value: list      # leaf value (exact float), 0.0 at internal nodes

    def to_json(self):
        return {"feature": self.feature, "threshold": [float(t).hex() for t in self.threshold],
                "left": self.left, "right": self.right, "value": [float(v).hex() for v in self.value]}

    @staticmethod
    def from_json(d):
        return Tree(list(d["feature"]), [float.fromhex(t) for t in d["threshold"]], list(d["left"]),
                    list(d["right"]), [float.fromhex(v) for v in d["value"]])


@dataclass
class Ensemble:
    family: str          # xgboost | lightgbm | sklearn_rf
    rule: str            # lt | le
    input_cast: str      # float64 | float32
    score: str           # margin_sum (s = base + sum v) | mean_probability (s = sum v / T)
    feature_names: list
    trees: list
    base: float = 0.0
    tau: float | None = None   # classifier: kappa(x) = 1[s(x) > tau]
    meta: dict = field(default_factory=dict)

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def n_trees(self):
        return len(self.trees)

    def to_json(self):
        return {"family": self.family, "rule": self.rule, "input_cast": self.input_cast, "score": self.score,
                "feature_names": list(self.feature_names), "base": float(self.base).hex(),
                "tau": None if self.tau is None else float(self.tau).hex(),
                "trees": [t.to_json() for t in self.trees], "meta": self.meta}

    @staticmethod
    def from_json(d):
        return Ensemble(d["family"], d["rule"], d["input_cast"], d["score"], list(d["feature_names"]),
                        [Tree.from_json(t) for t in d["trees"]], float.fromhex(d["base"]),
                        None if d["tau"] is None else float.fromhex(d["tau"]), d.get("meta", {}))

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_json(), f)

    @staticmethod
    def load(path):
        with open(path) as f:
            return Ensemble.from_json(json.load(f))

    def hash(self, with_tau=True) -> str:
        d = self.to_json()
        d.pop("meta")
        if not with_tau:
            d.pop("tau")
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

    def effective_threshold(self) -> Fraction:
        """tau' such that kappa(x) = 1[sum_t v_t(x) > tau'] exactly."""
        if self.tau is None:
            raise ValueError("tau not set")
        if self.score == "margin_sum":
            return Fraction(self.tau) - Fraction(self.base)
        return Fraction(self.tau) * self.n_trees


def export_xgboost(model, feature_names) -> Ensemble:
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    cfg = json.loads(booster.save_raw("json"))
    lmp = cfg["learner"]["learner_model_param"]
    base_score = float(str(lmp["base_score"]).strip("[]"))
    if base_score != 0.5:
        raise ValueError("train XGBoost with base_score=0.5 so that the exported base margin is exactly 0")
    gb = cfg["learner"]["gradient_booster"]
    if gb.get("name") != "gbtree":
        raise ValueError("only gbtree boosters are supported")
    trees = []
    for t in gb["model"]["trees"]:
        if any(t.get("split_type", [])) or t.get("categories"):
            raise ValueError("categorical splits are not supported")
        left, right = t["left_children"], t["right_children"]
        cond, idx = t["split_conditions"], t["split_indices"]
        feat, thr, val = [], [], []
        for i in range(len(left)):
            if left[i] == -1:
                feat.append(LEAF); thr.append(0.0); val.append(float(np.float32(cond[i])))
            else:
                feat.append(int(idx[i])); thr.append(float(np.float32(cond[i]))); val.append(0.0)
        trees.append(Tree(feat, thr, [int(c) for c in left], [int(c) for c in right], val))
    return Ensemble("xgboost", "lt", "float64", "margin_sum", list(feature_names), trees, 0.0,
                    meta={"base_score": base_score})


def export_lightgbm(model, feature_names) -> Ensemble:
    booster = model.booster_ if hasattr(model, "booster_") else model
    d = booster.dump_model()
    if d.get("num_tree_per_iteration", 1) != 1:
        raise ValueError("binary models only")
    if d.get("average_output"):
        raise ValueError("random-forest mode LightGBM is not supported")
    trees = []
    for info in d["tree_info"]:
        feat, thr, left, right, val = [], [], [], [], []

        def walk(node):
            i = len(feat)
            feat.append(None); thr.append(0.0); left.append(-1); right.append(-1); val.append(0.0)
            if "leaf_value" in node:
                feat[i] = LEAF
                val[i] = float(node["leaf_value"])
                return i
            if node["decision_type"] != "<=":
                raise ValueError(f"unsupported decision type {node['decision_type']}")
            if node.get("missing_type") not in (None, "None"):
                raise ValueError("models trained with missing values are not supported")
            feat[i] = int(node["split_feature"])
            thr[i] = float(node["threshold"])
            left[i] = walk(node["left_child"])
            right[i] = walk(node["right_child"])
            return i

        walk(info["tree_structure"])
        trees.append(Tree(feat, thr, left, right, val))
    return Ensemble("lightgbm", "le", "float64", "margin_sum", list(feature_names), trees, 0.0)


def export_sklearn_rf(model, feature_names) -> Ensemble:
    trees = []
    for est in model.estimators_:
        t = est.tree_
        if t.n_outputs != 1 or t.value.shape[2] != 2:
            raise ValueError("binary single-output forests only")
        v = t.value[:, 0, :].astype(np.float64)
        p1 = v[:, 1] / v.sum(axis=1)   # same float64 operation as DecisionTreeClassifier.predict_proba
        feat, thr, val = [], [], []
        for i in range(t.node_count):
            if t.children_left[i] == -1:
                feat.append(LEAF); thr.append(0.0); val.append(float(p1[i]))
            else:
                feat.append(int(t.feature[i])); thr.append(float(t.threshold[i])); val.append(0.0)
        trees.append(Tree(feat, thr, [int(c) for c in t.children_left], [int(c) for c in t.children_right], val))
    return Ensemble("sklearn_rf", "le", "float32", "mean_probability", list(feature_names), trees, 0.0)


def export_model(model, family, feature_names) -> Ensemble:
    return {"xgboost": export_xgboost, "lightgbm": export_lightgbm, "sklearn_rf": export_sklearn_rf}[family](
        model, feature_names)
