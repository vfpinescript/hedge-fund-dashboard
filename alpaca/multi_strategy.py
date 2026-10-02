"""
Combined target book for the Alpaca paper account: three uncorrelated sleeves,
weighted toward the forward-validated edges rather than any single backtest.

  1. FX reversal     (35%) — the DSR-robust FX-ETF short-term reversal book.
  2. Leverage rotation (30%) — SPY above its 200d MA -> SSO (2x), else BIL.
  3. Crypto trend    (25%) — BTC/ETH above their 200d MA -> hold, else cash.
  (~10% cash buffer.)

Diversifying across three genuinely different return streams is the honest way
to improve a book's return/risk, not concentrating into the luckiest backtester.
DRY_RUN prints the plan only.
"""
from __future__ import annotations
import os
import json

import numpy as np
import pandas as pd

from alpaca.client import Alpaca
from alpaca.strategy import UNIVERSE as FX_UNIVERSE, _closes, REVERSAL_WINDOW, VOL_WINDOW

FX_GROSS = float(os.environ.get("FX_GROSS", "0.35"))
LEV_GROSS = float(os.environ.get("LEV_GROSS", "0.30"))
CRYPTO_GROSS = float(os.environ.get("CRYPTO_GROSS", "0.25"))
MA_WINDOW = 200
MIN_TRADE_DOLLARS = 50.0
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
CRYPTO_ORDER = {"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD"}   # position form -> order form


def _fx_targets(a, equity, prices):
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
    out = {}
    for s in w.index:
        prices[s] = float(px.iloc[-1][s])
        out[s] = (float(w[s]) * gross) / prices[s]
    return out


def _leverage_sleeve(a, equity, prices):
    bars = a.daily_bars("SPY", days=260)
    closes = np.array([b["c"] for b in bars], dtype=float)
    above = closes[-1] > closes[-MA_WINDOW:].mean()
    sym = "SSO" if above else "BIL"
    price = a.latest_trade(sym) or a.daily_bars(sym, days=3)[-1]["c"]
    prices[sym] = float(price)
    return {sym: (equity * LEV_GROSS) / float(price)}, ("risk-on -> SSO" if above else "risk-off -> BIL")


def _crypto_sleeve(a, equity, prices):
    bars = a.crypto_bars()
    tgt, held = {}, []
    per = equity * CRYPTO_GROSS / 2.0
    for order_sym, pos_sym in [("BTC/USD", "BTCUSD"), ("ETH/USD", "ETHUSD")]:
        b = bars.get(order_sym) or []
        if len(b) < MA_WINDOW + 1:
            continue
        c = np.array([x["c"] for x in b], dtype=float)
        prices[pos_sym] = float(c[-1])
        if c[-1] > c[-MA_WINDOW:].mean():       # above 200d MA -> hold
            tgt[pos_sym] = per / float(c[-1])
            held.append(pos_sym.replace("USD", ""))
    regime = ("hold " + "+".join(held)) if held else "all below MA -> cash"
    return tgt, regime


def compute_combined(a=None):
    a = a or Alpaca()
    equity = float(a.account().get("equity", 0))
    prices = {}
    fx = _fx_targets(a, equity, prices)
    lev, lev_regime = _leverage_sleeve(a, equity, prices)
    cry, cry_regime = _crypto_sleeve(a, equity, prices)
    targets = {}
    for d in (fx, lev, cry):
        for k, v in d.items():
            targets[k] = targets.get(k, 0.0) + v
    return {"equity": equity, "targets": targets, "prices": prices,
            "lev_regime": lev_regime, "crypto_regime": cry_regime,
            "sleeves": {"fx": fx, "leverage": lev, "crypto": cry}}


def plan(a=None):
    a = a or Alpaca()
    t = compute_combined(a)
    prices = t["prices"]
    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    orders = []
    for sym, want in t["targets"].items():
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 6)
        px = prices.get(sym, 0)
        if px and abs(delta * px) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target": round(want, 4), "cur": cur})
    for sym, cur in positions.items():
        if sym not in t["targets"] and abs(cur) > 0:
            orders.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                           "qty": abs(cur), "target": 0.0, "cur": cur})
    return {"equity": t["equity"], "lev_regime": t["lev_regime"],
            "crypto_regime": t["crypto_regime"],
            "weights": {"fx": FX_GROSS, "leverage": LEV_GROSS, "crypto": CRYPTO_GROSS},
            "n_orders": len(orders), "orders": orders}


def execute(a=None):
    a = a or Alpaca()
    p = plan(a)
    if DRY_RUN:
        p["dry_run"] = True
        return p
    eq_open = a.clock().get("is_open")
    submitted = []
    for o in p["orders"]:
        is_crypto = o["symbol"] in CRYPTO_ORDER
        if not is_crypto and not eq_open:
            submitted.append({"symbol": o["symbol"], "skipped": "market_closed"})
            continue
        order_sym = CRYPTO_ORDER.get(o["symbol"], o["symbol"])
        qty = round(o["qty"], 6 if is_crypto else 3)
        r = a.submit_order(order_sym, qty, o["side"])
        submitted.append({"symbol": o["symbol"], "side": o["side"], "qty": qty,
                          "status": r.get("status"), "error": r.get("message")})
    p["submitted"] = submitted
    p["dry_run"] = False
    return p


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
