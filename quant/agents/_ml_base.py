"""
Shared machinery for the supervised directional agents (decision tree, random
forest, gradient boosting). They differ only in the classifier; everything else
— feature panel, chronological split, out-of-sample accuracy, conviction
tempered by real skill — is identical and lives here.

Conviction rule (same for all): an agent is only as confident as its edge AND
its proven out-of-sample skill. A model that can't beat a coin flip out of
sample contributes ~0 conviction, so the committee ignores it rather than
trusting a curve-fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ..base import DirectionalAgent, DirectionalSignal, Direction
from ..data.loader import make_features, label_forward, temporal_split


class MLDirectionalAgent(DirectionalAgent):
    name = "ml_directional"
    weight = 1.0

    def __init__(self, horizon: int = 5, min_bars: int = 400, train_frac: float = 0.7):
        self.horizon = horizon
        self.min_bars = min_bars
        self.train_frac = train_frac

    # --- subclass hooks -------------------------------------------------
    def _build_model(self):
        raise NotImplementedError

    def _extra_rationale(self, model, latest_row: pd.DataFrame, X: pd.DataFrame) -> str:
        return ""

    def _features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return make_features(ohlcv)

    # --- shared flow ----------------------------------------------------
    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> DirectionalSignal:
        if ohlcv is None or len(ohlcv) < self.min_bars:
            return self._abstain(instrument, f"need >= {self.min_bars} bars")

        X = self._features(ohlcv)
        y = label_forward(ohlcv, self.horizon)
        Xtr, ytr, Xte, yte = temporal_split(X, y, self.train_frac)
        if len(Xtr) < 100 or ytr.nunique() < 2:
            return self._abstain(instrument, "insufficient / single-class training data")

        model = self._build_model()
        model.fit(Xtr, ytr)
        oos_acc = accuracy_score(yte, model.predict(Xte)) if len(Xte) else float("nan")

        latest = X.dropna().iloc[[-1]]
        if latest.empty:
            return self._abstain(instrument, "latest features incomplete")
        classes = list(model.classes_)
        proba_up = float(model.predict_proba(latest)[0][classes.index(1)]) if 1 in classes else 0.0

        edge = proba_up - 0.5
        skill = max(0.0, (oos_acc - 0.5) * 2) if np.isfinite(oos_acc) else 0.0
        conviction = min(1.0, abs(edge) * 2 * (0.5 + 0.5 * skill))
        direction = (Direction.BULL if edge > 0 else
                     Direction.BEAR if edge < 0 else Direction.NEUTRAL)

        extra = self._extra_rationale(model, latest, X)
        importances = {}
        if hasattr(model, "feature_importances_"):
            importances = dict(zip(X.columns, np.round(model.feature_importances_, 3)))
        return DirectionalSignal(
            self.name, instrument, direction, conviction,
            rationale=f"P(up)={proba_up:.2f}, OOS acc={oos_acc:.2f}. {extra}".strip(),
            diagnostics={"oos_accuracy": oos_acc, "proba_up": proba_up,
                         "feature_importances": importances},
        )
