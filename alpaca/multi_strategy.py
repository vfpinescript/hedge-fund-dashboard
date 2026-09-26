"""
Combined target book for the Alpaca paper account: two sleeves.

  1. FX reversal sleeve  — the existing DSR-robust FX-ETF reversal book.
  2. Leverage rotation   — Gayed's rule: SPY above its 200d MA -> hold SSO (2x
                           S&P), else hold BIL (T-bills).

Each sleeve gets a fixed fraction of equity; the executor reconciles the whole
account to the union of both sleeves' targets. DRY_RUN prints the plan only.
"""
from __future__ import annotations
import os
import json

import numpy as np

from alpaca.client import Alpaca
from alpaca.strategy import UNIVERSE as FX_UNIVERSE, _closes, REVERSAL_WINDOW, VOL_WINDOW
import pandas as pd

FX_GROSS = float(os.environ.get("FX_GROSS", "0.40"))     # FX reversal sleeve, % equity
LEV_GROSS = float(os.environ.get("LEV_GROSS", "0.40"))   # leverage rotation sleeve, % equity
MA_WINDOW = 200
MIN_TRADE_DOLLARS = 50.0
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"


def _fx_targets(a, equity):
    px = _closes(a)
    if px.empty:
        return {}
    ret = px.pct_change()
    mom = px / px.shift(REVERSAL_WINDOW) - 1.0
    vol = ret.rolling(VOL_WINDOW).std()
    raw = -np.sign(mom) / vol
    last = raw.iloc[-1].replace([np.inf, -np.inf], np.nan).dropna()
    if last.empty or last.abs().sum() == 0:
        return {}
    w = last / last.abs().sum()
    gross = equity * FX_GROSS
    prices = px.iloc[-1]
    return {s: (float(w[s]) * gross) / float(prices[s]) for s in w.index}   # symbol -> shares


def _leverage_sleeve(a, equity):
    bars = a.daily_bars("SPY", days=260)
    closes = np.array([b["c"] for b in bars], dtype=float)
    above = closes[-1] > closes[-MA_WINDOW:].mean()
    sym = "SSO" if above else "BIL"
    price = a.latest_trade(sym) or a.daily_bars(sym, days=3)[-1]["c"]
    shares = (equity * LEV_GROSS) / float(price)
    return {sym: shares}, ("risk-on -> SSO (2x S&P)" if above else "risk-off -> BIL (T-bills)")


def compute_combined(a=None):
    a = a or Alpaca()
    equity = float(a.account().get("equity", 0))
    fx = _fx_targets(a, equity)
    lev, regime = _leverage_sleeve(a, equity)
    targets = dict(fx)
    for k, v in lev.items():                     # merge (no symbol overlap expected)
        targets[k] = targets.get(k, 0.0) + v
    return {"equity": equity, "regime": regime, "targets": targets,
            "fx_sleeve": fx, "lev_sleeve": lev}


def plan(a=None):
    a = a or Alpaca()
    t = compute_combined(a)
    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    prices = {}
    orders = []
    for sym, want in t["targets"].items():
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 3)
        px = prices.get(sym) or a.latest_trade(sym) or 0
        if px and abs(delta * px) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target": round(want, 3), "cur": cur})
    for sym, cur in positions.items():           # exit anything no longer targeted
        if sym not in t["targets"] and abs(cur) > 0:
            orders.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                           "qty": abs(cur), "target": 0.0, "cur": cur})
    return {"equity": t["equity"], "regime": t["regime"],
            "fx_sleeve_pct": FX_GROSS * 100, "lev_sleeve_pct": LEV_GROSS * 100,
            "n_orders": len(orders), "orders": orders}


def execute(a=None):
    a = a or Alpaca()
    p = plan(a)
    if DRY_RUN:
        p["dry_run"] = True
        return p
    if not a.clock().get("is_open"):
        p["skipped"] = "market_closed"
        return p
    submitted = []
    for o in p["orders"]:
        r = a.submit_order(o["symbol"], round(o["qty"], 3), o["side"])
        submitted.append({"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                          "status": r.get("status"), "error": r.get("message")})
    p["submitted"] = submitted
    p["dry_run"] = False
    return p


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
