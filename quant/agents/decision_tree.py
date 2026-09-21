"""
Decision-tree directional agent  (Quant_Analysis: "3. Decision Trees").

A single shallow tree on the shared feature panel. Role: DIRECTIONAL. Its
selling point is explainability, so the rationale exposes the actual
root-to-leaf rule that fired, in plain English.
"""
from __future__ import annotations

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ._ml_base import MLDirectionalAgent


class DecisionTreeAgent(MLDirectionalAgent):
    name = "decision_tree"
    weight = 0.8   # single tree is weaker than the ensembles; down-weighted

    def __init__(self, max_depth: int = 3, **kw):
        super().__init__(**kw)
        self.max_depth = max_depth

    def _build_model(self):
        return DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_leaf=20, random_state=0,
        )

    def _extra_rationale(self, model, latest_row: pd.DataFrame, X: pd.DataFrame) -> str:
        return "rule: " + _leaf_rule(model, latest_row)


def _leaf_rule(clf, row: pd.DataFrame) -> str:
    """Human-readable root-to-leaf path for the predicted row."""
    tree = clf.tree_
    feat_names = list(row.columns)
    node, parts = 0, []
    x = row.iloc[0].to_dict()
    while tree.children_left[node] != tree.children_right[node]:
        f = feat_names[tree.feature[node]]
        thr = tree.threshold[node]
        if x[f] <= thr:
            parts.append(f"{f}<={thr:.3f}")
            node = tree.children_left[node]
        else:
            parts.append(f"{f}>{thr:.3f}")
            node = tree.children_right[node]
    return " AND ".join(parts) if parts else "(root leaf)"
