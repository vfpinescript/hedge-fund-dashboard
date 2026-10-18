"""
Congress-copy desk — one paper account (ALPACA3) that EXCLUSIVELY mirrors the
stock trades disclosed by members of the U.S. Congress.

It reads the latest Senate + House disclosures (quant/data/congress.py, via the
free FMP key), aggregates them into a net-buying signal per ticker, and
reconciles the account to hold what Congress is net-buying, equal-ish weighted by
disclosed dollar size. A ticker they are net-SELLING drops out of the target and
is closed. Same reconcile/execute pattern as the other desks; DRY_RUN previews.

HONEST LIMITATIONS (read before trusting it):
  - Disclosures are LAGGED. The STOCK Act allows up to ~45 days between a trade
    and its disclosure, so this follows delayed public filings, not live trades.
  - Only U.S.-listed equities Alpaca can trade are copied; options, bonds, funds,
    and non-U.S. assets in the filings are skipped.
  - Filers disclose dollar RANGES, not exact sizes or their portfolio weights, so
    position sizing here is an approximation (midpoint of the range), not a true
    replica of any member's book.
This is a public-record-following experiment, not investment advice.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpaca.client import Alpaca
from quant.data.congress import latest_trades, amount_mid

# dedicated account for this desk
KEY_ENV, SECRET_ENV = "ALPACA3_KEY_ID", "ALPACA3_SECRET"

DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
INVESTED = float(os.environ.get("CONGRESS_INVESTED", "0.95"))   # fraction deployed
MAX_POS_PCT = float(os.environ.get("CONGRESS_MAX_POS", "0.12")) # cap per name
MAX_NAMES = int(os.environ.get("CONGRESS_MAX_NAMES", "25"))     # breadth cap
MIN_TRADE_DOLLARS = float(os.environ.get("MIN_TRADE_DOLLARS", "20"))
PAGES = int(os.environ.get("CONGRESS_PAGES", "3"))              # disclosure depth

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")   # plain U.S. equity tickers only


def _is_purchase(t: str) -> bool:
    t = (t or "").lower()
    return "purchase" in t or t == "buy" or "bought" in t


def _is_sale(t: str) -> bool:
    t = (t or "").lower()
    return "sale" in t or "sold" in t or t == "sell"


def net_signal(pages: int = PAGES) -> dict:
    """Net disclosed dollar flow per ticker: sum(purchase midpoints) minus
    sum(sale midpoints) across the recent Senate+House disclosures. Positive =
    Congress is net-buying."""
    flows: dict[str, float] = {}
    for rec in latest_trades(pages=pages):
        tk = (rec.get("ticker") or "").strip().upper()
        if not _TICKER_RE.match(tk):
            continue                                  # skip options/bonds/blanks
        amt = amount_mid(rec.get("amount", ""))
        if _is_purchase(rec.get("type", "")):
            flows[tk] = flows.get(tk, 0.0) + amt
        elif _is_sale(rec.get("type", "")):
            flows[tk] = flows.get(tk, 0.0) - amt
    return flows


def target_weights(pages: int = PAGES) -> dict:
    """Equal-ish weights over the tickers Congress is net-BUYING, sized by net
    disclosed dollars, capped per name and limited to the top MAX_NAMES."""
    flows = net_signal(pages)
    buys = {k: v for k, v in flows.items() if v > 0}
    if not buys:
        return {}
    top = dict(sorted(buys.items(), key=lambda kv: -kv[1])[:MAX_NAMES])
    total = sum(top.values())
    weights = {k: min(MAX_POS_PCT, v / total) for k, v in top.items()}
    s = sum(weights.values())
    return {k: w / s for k, w in weights.items()} if s else {}


def plan():
    a = Alpaca(key_env=KEY_ENV, secret_env=SECRET_ENV)
    equity = float(a.account().get("equity", 0))
    weights = target_weights()
    deployable = equity * INVESTED

    # price each target and convert weight -> target share qty
    targets, priced = {}, {}
    for sym, w in weights.items():
        px = a.latest_trade(sym)
        if not px or px <= 0:
            continue                                  # not tradable / no quote
        priced[sym] = px
        targets[sym] = (w * deployable) / px

    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    orders = []
    for sym, want in targets.items():
        cur = positions.get(sym, 0.0)
        delta = round(want - cur, 6)
        if abs(delta * priced[sym]) < MIN_TRADE_DOLLARS:
            continue
        orders.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                       "qty": abs(delta), "target": round(want, 4), "cur": cur})
    # exit anything Congress is no longer net-buying
    for sym, cur in positions.items():
        if sym not in targets and abs(cur) > 0:
            orders.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                           "qty": abs(cur), "target": 0.0, "cur": cur})
    return a, {"desk": "congress", "account": "ALPACA3", "equity": round(equity, 2),
               "n_names": len(targets), "n_orders": len(orders),
               "top_targets": dict(sorted(weights.items(), key=lambda kv: -kv[1])[:10]),
               "orders": orders}


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
