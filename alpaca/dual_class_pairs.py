"""
Dual-class statistical-arbitrage desk — market-neutral pairs on one paper account
(ALPACA3). Long the cheap share class, short the rich one of the SAME company
when their log-spread z-score stretches past +/-2, flatten near 0.

ONLY the pairs that are actually SHORTABLE on Alpaca are traded (the strategy is
market-neutral, so BOTH legs must be executable). Verified on Alpaca:
  - GOOGL/GOOG : liquid, easy-to-borrow  -> the clean anchor
  - GEF/GEF-B  : both shortable (small-cap, thin fills; smaller weight)
Dropped: BRK-B/BRK-A (BRK-A ~$477k/share, cannot size a leg), and MOG/CRD/BIO/LEN
(their B-shares are not shortable on Alpaca).

HONEST NOTES: this is the backtest's clean subset, so live Sharpe tracks ~1.0
(GOOGL/GOOG), not the diversified ~1.8 book. It has NO forward evidence yet
(in-sample backtest only). DRY_RUN previews; nothing trades until DRY_RUN=0.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpaca.client import Alpaca
from quant.data.loader import load_ohlcv

KEY_ENV, SECRET_ENV = "ALPACA3_KEY_ID", "ALPACA3_SECRET"
DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"

# (class A, class B, weight) — only Alpaca-shortable pairs. Weight favours the
# liquid anchor; GEF is thin so it gets a small sleeve.
LIVE_PAIRS = [("GOOGL", "GOOG", 0.80), ("GEF", "GEF-B", 0.20)]
INVESTED = float(os.environ.get("PAIRS_INVESTED", "0.90"))   # gross deployed
WINDOW, Z_ENTRY, Z_EXIT = 20, 2.0, 0.3
MIN_TRADE_DOLLARS = 25.0


def _alp(sym: str) -> str:
    return sym.replace("-", ".")           # Alpaca class-share format: GEF.B, BRK.B


def _signal(a: str, b: str):
    """Current pair position in {+1: long A/short B, -1: short A/long B, 0}."""
    try:
        pa = load_ohlcv(a, bars=400)["close"]
        pb = load_ohlcv(b, bars=400)["close"]
    except Exception:
        return 0
    df = pd.concat([pa, pb], axis=1).dropna()
    if len(df) < WINDOW + 5:
        return 0
    spr = np.log(df.iloc[:, 0]) - np.log(df.iloc[:, 1])
    z = (spr - spr.rolling(WINDOW).mean()) / spr.rolling(WINDOW).std()
    zc = float(z.iloc[-1])
    if np.isnan(zc):
        return 0
    if zc > Z_ENTRY:
        return -1                          # spread rich: short A, long B
    if zc < -Z_ENTRY:
        return 1                           # spread cheap: long A, short B
    return 0


def plan():
    a = Alpaca(key_env=KEY_ENV, secret_env=SECRET_ENV)
    equity = float(a.account().get("equity", 0))
    deployable = equity * INVESTED

    # build target dollar exposure per leg (market-neutral: +leg / -leg)
    targets: dict[str, float] = {}         # symbol -> target $ (signed)
    notes = []
    for A, B, w in LIVE_PAIRS:
        sig = _signal(A, B)
        leg = deployable * w / 2.0          # each leg half the pair's sleeve
        if sig == 1:
            targets[A] = targets.get(A, 0) + leg
            targets[B] = targets.get(B, 0) - leg
            notes.append(f"{A}/{B}: long {A} / short {B}")
        elif sig == -1:
            targets[A] = targets.get(A, 0) - leg
            targets[B] = targets.get(B, 0) + leg
            notes.append(f"{A}/{B}: short {A} / long {B}")
        else:
            notes.append(f"{A}/{B}: flat (within band)")

    # price and convert $ targets -> share targets (signed)
    prices, tgt_qty = {}, {}
    for sym, dollars in targets.items():
        px = a.latest_trade(_alp(sym))
        if not px or px <= 0:
            continue
        prices[sym] = px
        tgt_qty[sym] = dollars / px

    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    orders = []
    universe = set(tgt_qty) | set(positions)
    for sym in universe:
        want = tgt_qty.get(sym, 0.0)
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 6)
        px = prices.get(sym) or a.latest_trade(_alp(sym)) or 0
        if px and abs(delta * px) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": _alp(sym), "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target_shares": round(want, 4),
                       "cur": cur})
    return a, {"desk": "dual_class_pairs", "account": "ALPACA3",
               "equity": round(equity, 2), "signals": notes,
               "n_orders": len(orders), "orders": orders}


def execute():
    a, p = plan()
    if DRY_RUN:
        p["dry_run"] = True
        return p
    is_open = a.clock().get("is_open")
    submitted = []
    for o in p["orders"]:
        if not is_open:
            submitted.append({"symbol": o["symbol"], "skipped": "market_closed"})
            continue
        qty = round(o["qty"], 3)
        if qty <= 0:
            continue
        r = a.submit_order(o["symbol"], qty, o["side"])
        submitted.append({"symbol": o["symbol"], "side": o["side"], "qty": qty,
                          "status": r.get("status"), "error": r.get("message")})
    p["submitted"] = submitted
    p["dry_run"] = False
    return p


if __name__ == "__main__":
    print(json.dumps(execute(), indent=2))
