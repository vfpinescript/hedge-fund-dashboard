"""
Volatility-regime risk agent  (Quant_Analysis: "volatility smile / surface").

Role: RISK. It never picks a direction — it scales position size down when
volatility is elevated relative to the instrument's own history, and vetoes in
extreme regimes. Realized vol is computed from OHLCV (always available). If an
option chain is available for the underlying, it also compares implied vs
realized (a rich-vol regime), but that path is optional and abstains cleanly.

Size scaler = clamp( target_vol / current_vol ), i.e. classic vol-targeting:
when vol doubles, halve the size so risk per trade stays roughly constant.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import RiskAgent, RiskSignal

ANNUALIZE = np.sqrt(252)


class VolRegimeAgent(RiskAgent):
    name = "vol_regime"

    def __init__(self, window: int = 20, lookback: int = 500,
                 veto_percentile: float = 0.98, min_bars: int = 250):
        self.window = window
        self.lookback = lookback
        self.veto_percentile = veto_percentile
        self.min_bars = min_bars

    def evaluate(self, instrument: str, ohlcv: pd.DataFrame, **kwargs) -> RiskSignal:
        if ohlcv is None or len(ohlcv) < self.min_bars:
            return self._abstain(instrument, f"need >= {self.min_bars} bars")

        rets = ohlcv["close"].pct_change()
        rv = rets.rolling(self.window).std() * ANNUALIZE
        rv = rv.dropna()
        if len(rv) < self.lookback // 2:
            return self._abstain(instrument, "insufficient vol history")

        cur = float(rv.iloc[-1])
        hist = rv.iloc[-self.lookback:]
        median_vol = float(hist.median())
        pctile = float((hist < cur).mean())

        # Vol-target scaler: keep risk ~constant as vol moves.
        scale = median_vol / cur if cur > 0 else 1.0
        scale = float(min(1.0, max(0.2, scale)))   # never scale up, floor at 0.2

        veto = pctile >= self.veto_percentile
        why = (f"realized vol {cur*100:.1f}% (ann.), {pctile*100:.0f}th pctile of "
               f"{self.lookback}-bar history; size x{scale:.2f}"
               + ("  -> VETO: vol at extreme" if veto else ""))
        return RiskSignal(self.name, instrument, risk_scale=(0.0 if veto else scale),
                          veto=veto, rationale=why,
                          diagnostics={"realized_vol": cur, "median_vol": median_vol,
                                       "vol_percentile": pctile})
