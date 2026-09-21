"""
FX-majors short-term reversal — mapped onto Alpaca-tradable CurrencyShares ETFs.

Universe (ETF -> FX major it tracks):
  FXE=EUR  FXB=GBP  FXY=JPY  FXF=CHF  FXA=AUD  FXC=CAD

Signal: 7-day short-term reversal (the DSR-robust edge), vol-targeted and
equal-risk across the six. Produces signed target dollar exposures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpaca.client import Alpaca

UNIVERSE = ["FXE", "FXB", "FXY", "FXF", "FXA", "FXC"]
REVERSAL_WINDOW = 7
VOL_WINDOW = 20
TARGET_GROSS_FRACTION = 0.50   # deploy 50% of equity gross (conservative; 4x BP available)


def _closes(a: Alpaca) -> pd.DataFrame:
    cols = {}
    for sym in UNIVERSE:
        bars = a.daily_bars(sym, days=200)
        if not bars:
            continue
        s = pd.Series({b["t"][:10]: b["c"] for b in bars})
        s.index = pd.to_datetime(s.index)
        cols[sym] = s
    return pd.DataFrame(cols).sort_index()


def compute_targets(a: Alpaca | None = None) -> dict:
    a = a or Alpaca()
    px = _closes(a)
    if px.empty:
        return {"error": "no price data"}
    ret = px.pct_change()
    mom = px / px.shift(REVERSAL_WINDOW) - 1.0          # past 7-day return
    vol = ret.rolling(VOL_WINDOW).std()
    raw = -np.sign(mom) / vol                            # reversal, vol-scaled
    last = raw.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    if last.abs().sum() == 0 or last.empty:
        return {"error": "flat signal"}
    weights = last / last.abs().sum()                    # equal-risk, sum|w|=1

    acct = a.account()
    equity = float(acct.get("equity", 0))
    gross = equity * TARGET_GROSS_FRACTION
    prices = px.iloc[-1]

    targets = {}
    for sym, w in weights.items():
        dollars = float(w) * gross
        price = float(prices[sym])
        targets[sym] = {
            "weight": round(float(w), 4),
            "target_dollars": round(dollars, 2),
            "price": round(price, 2),
            "target_shares": round(dollars / price, 3),
            "reversal_signal": "long" if w > 0 else "short",
            "past_7d_return_pct": round(float(mom.iloc[-1][sym]) * 100, 2),
        }
    return {
        "as_of": px.index[-1].strftime("%Y-%m-%d"),
        "equity": equity,
        "target_gross": round(gross, 2),
        "universe": UNIVERSE,
        "targets": targets,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(compute_targets(), indent=2))
