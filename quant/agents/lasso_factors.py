"""
Lasso directional agent  (Quant_Analysis: "5. LASSO").

Linear regression with an L1 penalty on the wide candidate-signal panel. L1
drives most coefficients to exactly zero, so the model selects a handful of
signals and predicts forward return from them. Role: DIRECTIONAL — sign of the
predicted return is the vote; conviction scales with predicted magnitude and
out-of-sample R². Rationale lists the surviving (non-zero) signals, the whole
point of the method.

Uses LassoCV with TimeSeriesSplit — temporal folds, never shuffled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from ..base import DirectionalAgent, DirectionalSignal, Direction
from ..data.loader import make_features_wide, temporal_split


class LassoFactorAgent(DirectionalAgent):
    name = "lasso_factors"
    weight = 0.9

    def __init__(self, horizon: int = 5, min_bars: int = 500, train_frac: float = 0.7):
        self.horizon = horizon
        self.min_bars = min_bars
        self.train_frac = train_frac

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> DirectionalSignal:
        if ohlcv is None or len(ohlcv) < self.min_bars:
            return self._abstain(instrument, f"need >= {self.min_bars} bars")

        X = make_features_wide(ohlcv)
        # Target: forward return (regression, not classification).
        fwd_ret = ohlcv["close"].shift(-self.horizon) / ohlcv["close"] - 1.0
        Xtr, ytr, Xte, yte = temporal_split(X, fwd_ret.rename("y"), self.train_frac)
        if len(Xtr) < 150:
            return self._abstain(instrument, "insufficient training data")

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        try:
            model = LassoCV(cv=TimeSeriesSplit(5), max_iter=5000,
                            random_state=0).fit(Xtr_s, ytr)
        except Exception as e:
            return self._abstain(instrument, f"lasso fit failed: {e}")

        # Out-of-sample R² as the skill gate.
        Xte_s = scaler.transform(Xte)
        oos_r2 = model.score(Xte_s, yte) if len(Xte) else float("nan")

        survivors = [(c, float(w)) for c, w in zip(X.columns, model.coef_) if abs(w) > 1e-8]
        n_surv = len(survivors)
        if n_surv == 0:
            return DirectionalSignal(
                self.name, instrument, Direction.NEUTRAL, 0.0,
                rationale=f"Lasso zeroed all {X.shape[1]} signals — no linear edge",
                diagnostics={"n_signals": X.shape[1], "survivors": 0, "oos_r2": oos_r2})

        latest = X.dropna().iloc[[-1]]
        pred = float(model.predict(scaler.transform(latest))[0])

        # Conviction: gated by positive OOS R² (predictive out of sample).
        skill = max(0.0, oos_r2) if np.isfinite(oos_r2) else 0.0
        typical = float(np.abs(ytr).mean()) or 1e-6
        conviction = min(1.0, (abs(pred) / (3 * typical)) * (0.3 + 0.7 * min(1.0, skill * 10)))
        direction = (Direction.BULL if pred > 0 else
                     Direction.BEAR if pred < 0 else Direction.NEUTRAL)

        top = sorted(survivors, key=lambda t: -abs(t[1]))[:4]
        return DirectionalSignal(
            self.name, instrument, direction, conviction,
            rationale=(f"pred fwd-ret={pred*100:.2f}%, OOS R²={oos_r2:.3f}; "
                       f"{n_surv}/{X.shape[1]} signals survive: "
                       + ", ".join(f"{c}({w:+.3f})" for c, w in top)),
            diagnostics={"n_signals": X.shape[1], "survivors": n_surv,
                         "oos_r2": oos_r2, "predicted_return": pred},
        )
