"""
Gradient-boosting directional agent  (Quant_Analysis: "2. XGBoost").

Each tree fits the residual the ensemble has left over (F_m = F_{m-1} + eta*h_m).
Role: DIRECTIONAL. Uses sklearn HistGradientBoostingClassifier — same algorithm
family as XGBoost (histogram-binned boosting with early stopping), chosen
because real xgboost needs the OpenMP runtime absent on this machine. Swap to
`xgboost` later with `brew install libomp && pip install xgboost`.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier

from ._ml_base import MLDirectionalAgent


class GradientBoostAgent(MLDirectionalAgent):
    name = "gradient_boost"
    weight = 1.0

    def __init__(self, learning_rate: float = 0.03, max_depth: int = 4,
                 max_iter: int = 400, **kw):
        super().__init__(**kw)
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_iter = max_iter

    def _build_model(self):
        return HistGradientBoostingClassifier(
            learning_rate=self.learning_rate, max_depth=self.max_depth,
            max_iter=self.max_iter, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.2,
            n_iter_no_change=50, random_state=0,
        )

    def _extra_rationale(self, model, latest_row, X) -> str:
        n = getattr(model, "n_iter_", None)
        return f"boosted {n} rounds (early-stopped)" if n else ""
