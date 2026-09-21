"""
Random-forest directional agent  (Quant_Analysis: "1. Random Forests").

Ensemble of decorrelated trees on the shared feature panel. Role: DIRECTIONAL.
The double randomisation (bootstrap rows + random feature subset per split)
averages out variance. Rationale surfaces the top feature importances — the
screenshots' "importances sort forty signals before you commit to any".
"""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ._ml_base import MLDirectionalAgent


class RandomForestAgent(MLDirectionalAgent):
    name = "random_forest"
    weight = 1.0

    def __init__(self, n_estimators: int = 300, max_depth: int = 5, **kw):
        super().__init__(**kw)
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _build_model(self):
        return RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            min_samples_leaf=50, max_features="sqrt", n_jobs=-1, random_state=0,
        )

    def _extra_rationale(self, model, latest_row, X) -> str:
        imp = sorted(zip(X.columns, model.feature_importances_),
                     key=lambda t: -t[1])[:3]
        return "top: " + ", ".join(f"{k}={v:.2f}" for k, v in imp)
