"""
Two additional Alpaca paper desks, each running one strategy on a basket.

  Desk 2 ($50k)  "trend"    — MA Timing (hold above the 200d MA, else cash) on
                              SPY, QQQ, BTC, ETH (the best-performing assets).
  Desk 3 ($100k) "reversal" — 7-day short-term reversal, dollar-neutral long/short,
                              on liquid shortable equity ETFs (SPY, QQQ, IWM, DIA).

Each desk uses its own keys, reconciles its account to its targets, and runs on
the schedule. DRY_RUN prints the plan only.
"""
from __future__ import annotations
import os
import json

import numpy as np
import pandas as pd

from alpaca.client import Alpaca

DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
MIN_TRADE_DOLLARS = 50.0
CRYPTO_ORDER = {"BTCUSD": "BTC/USD", "ETHUSD": "ETH/USD", "SOLUSD": "SOL/USD"}

DESKS = {
    "trend": {
        "key_env": "ALPACA2_KEY_ID", "secret_env": "ALPACA2_SECRET",
        "strategy": "ma_timing", "gross": 0.90,
        "assets": ["SPY", "QQQ", "BTC/USD", "ETH/USD"],
    },
    "reversal": {
        "key_env": "ALPACA3_KEY_ID", "secret_env": "ALPACA3_SECRET",
        "strategy": "reversal", "gross": 0.60,
        "assets": ["SPY", "QQQ", "IWM", "DIA"],
    },
}


def _pos_sym(s):          # order form -> position form (crypto has no slash in positions)
    return s.replace("/", "") if "/" in s else s


def _order_sym(s):        # position form -> order form
    return CRYPTO_ORDER.get(s, s)


def _daily_closes(a, assets, days=400) -> pd.DataFrame:
    out = {}
    stocks = [s for s in assets if "/" not in s]
    cryptos = [s for s in assets if "/" in s]
    for s in stocks:
        bars = a.daily_bars(s, days=days)
        out[s] = pd.Series({b["t"][:10]: b["c"] for b in bars})
    if cryptos:
        cb = a.crypto_bars(",".join(cryptos), days=days)
        for s in cryptos:
            b = cb.get(s) or []
            out[_pos_sym(s)] = pd.Series({x["t"][:10]: x["c"] for x in b})
    df = pd.DataFrame(out)
    df.index = pd.to_datetime(df.index)
    return df.sort_index().ffill()


def ma_timing_targets(a, assets, gross, equity):
    df = _daily_closes(a, assets)
    last = df.iloc[-1]
    ma = df.rolling(200).mean().iloc[-1]
    held = [s for s in df.columns if last[s] > ma[s]]
    if not held:
        return {}, "all below 200d MA -> cash"
    per = equity * gross / len(held)
    return {s: per / float(last[s]) for s in held}, "hold " + "+".join(held)


def reversal_targets(a, assets, gross, equity, window=7, vol_win=20):
    df = _daily_closes(a, assets)
    ret = df.pct_change()
    mom = df / df.shift(window) - 1.0
    vol = ret.rolling(vol_win).std()
    raw = (-np.sign(mom.iloc[-1]) / vol.iloc[-1]).replace([np.inf, -np.inf], np.nan).dropna()
    if raw.abs().sum() == 0 or raw.empty:
        return {}, "flat"
    w = raw / raw.abs().sum()
    last = df.iloc[-1]
    longs = [s for s in w.index if w[s] > 0]
    shorts = [s for s in w.index if w[s] < 0]
    note = "long " + "+".join(longs) + (" / short " + "+".join(shorts) if shorts else "")
    return {s: (float(w[s]) * equity * gross) / float(last[s]) for s in w.index}, note


def plan_desk(name):
    cfg = DESKS[name]
    a = Alpaca(key_env=cfg["key_env"], secret_env=cfg["secret_env"])
    equity = float(a.account().get("equity", 0))
    df = _daily_closes(a, cfg["assets"])
    prices = {s: float(df[s].iloc[-1]) for s in df.columns}
    if cfg["strategy"] == "ma_timing":
        targets, note = ma_timing_targets(a, cfg["assets"], cfg["gross"], equity)
    else:
        targets, note = reversal_targets(a, cfg["assets"], cfg["gross"], equity)
    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    orders = []
    for sym, want in targets.items():
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 6)
        px = prices.get(sym, 0)
        if px and abs(delta * px) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target": round(want, 4), "cur": cur})
    for sym, cur in positions.items():
        if sym not in targets and abs(cur) > 0:
            orders.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                           "qty": abs(cur), "target": 0.0, "cur": cur})
    return a, {"desk": name, "equity": equity, "strategy": cfg["strategy"],
               "note": note, "n_orders": len(orders), "orders": orders}


def execute_desk(name):
    a, p = plan_desk(name)
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
        qty = round(o["qty"], 6 if is_crypto else 3)
        r = a.submit_order(_order_sym(o["symbol"]), qty, o["side"])
        submitted.append({"symbol": o["symbol"], "side": o["side"], "qty": qty,
                          "status": r.get("status"), "error": r.get("message")})
    p["submitted"] = submitted
    p["dry_run"] = False
    return p


def run_all():
    return {name: execute_desk(name) for name in DESKS}


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
