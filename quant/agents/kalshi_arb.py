"""
Kalshi mispricing & arbitrage agent.

Prediction-market contracts are binary: a "Yes" contract pays $1 if the event
happens, "No" pays $1 if it doesn't. Two near-riskless arbitrages follow:

  1. Buy-both lock: buying Yes at yes_ask and No at no_ask always returns $1.
     If yes_ask + no_ask < 100¢ (minus fees), that's a guaranteed profit.
  2. Exhaustive-group lock: within one event where exactly one outcome occurs,
     sum of the yes_asks across all outcomes should be ≥ 100¢. If it's less,
     buy them all for a guaranteed $1.

This agent SCANS public Kalshi market data (no auth) and flags these locks net
of Kalshi's trading fee. It is genuinely uncorrelated with our price signals —
a different risk premium entirely. It NEVER places an order; it surfaces the
opportunity for a human to act on.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict

import ssl
import urllib.request
import urllib.parse
import json

import certifi

BASE = os.environ.get("KALSHI_API_BASE", "https://external-api.kalshi.com/trade-api/v2")
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())


def _get(path, params=None, timeout=20):
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "hedge-fund-v2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL_CTX) as r:
        return json.loads(r.read().decode())


# The open-markets firehose is dominated by zero-volume placeholder shards, so
# we scan liquid *series* directly. Discovered dynamically; this is the fallback.
FALLBACK_SERIES = [
    "KXBTCD", "KXBTC", "KXETHD", "KXETH", "KXSOLD",              # crypto
    "KXHIGHNY", "KXHIGHCHI", "KXHIGHLAX", "KXHIGHMIA", "KXHIGHAUS",  # weather
    "KXFED", "KXCPIYOY", "KXU3", "KXGDP",                        # macro
    "KXNASDAQ100", "KXSP500", "KXINXD", "KXNASDAQ100D",          # equities
]


def fetch_markets(max_markets=1500, page=200, series=None, max_series=25):
    """Fetch open markets from liquid series (real two-sided quotes).
    Defaults to the curated liquid set — fast and reliable. Scanning the full
    /series list is possible but slow, so it is capped by `max_series`."""
    series = (series or FALLBACK_SERIES)[:max_series]
    out = []
    for s in series:
        if len(out) >= max_markets:
            break
        try:
            d = _get("/markets", {"limit": page, "status": "open", "series_ticker": s})
        except Exception:
            continue
        out.extend(d.get("markets", []))
        time.sleep(0.1)
    return out


def kalshi_fee(price_dollars):
    """Kalshi trading fee per contract (USD): ~0.07 * P * (1-P), P in dollars."""
    p = float(price_dollars)
    return 0.07 * p * (1.0 - p)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class KalshiArbitrageAgent:
    name = "kalshi_arbitrage"

    def __init__(self, min_edge_cents=1.0, min_volume=0):
        self.min_edge_cents = min_edge_cents      # net profit threshold after fees
        self.min_volume = min_volume

    def scan(self, max_markets=1500) -> dict:
        markets = fetch_markets(max_markets)
        if not markets:
            return {"available": False, "reason": "no market data (network/API)",
                    "buy_both": [], "group_locks": []}

        buy_both = []
        events = set()
        priced = 0
        min_edge = self.min_edge_cents / 100.0            # threshold in dollars
        for m in markets:
            ya = _f(m.get("yes_ask_dollars"))
            na = _f(m.get("no_ask_dollars"))
            vol = _f(m.get("volume_fp")) or 0.0
            if m.get("event_ticker"):
                events.add(m["event_ticker"])
            # two-sided, tradeable, actually-traded market
            two_sided = ya is not None and na is not None and 0 < ya < 1 and 0 < na < 1
            if not two_sided or vol < max(1.0, self.min_volume):
                continue
            priced += 1
            cost = ya + na                                # dollars
            fee = kalshi_fee(ya) + kalshi_fee(na)
            net = (1.0 - cost) - fee                      # dollars profit per pair
            if net >= min_edge:
                buy_both.append({
                    "ticker": m.get("ticker"), "title": (m.get("title") or "")[:80],
                    "yes_ask": round(ya * 100, 1), "no_ask": round(na * 100, 1),
                    "cost_cents": round(cost * 100, 1),
                    "net_profit_cents": round(net * 100, 2), "volume": int(vol),
                })

        buy_both.sort(key=lambda x: -x["net_profit_cents"])
        return {
            "available": True, "scanned": len(markets), "priced": priced,
            "events_seen": len(events),
            "buy_both": buy_both[:25], "n_buy_both": len(buy_both),
            "note": ("Buy-both locks (yes_ask+no_ask<$1) are always valid and net of "
                     "Kalshi fees. Group/exhaustive locks require the COMPLETE event set "
                     "and mutually_exclusive=True — use verify_group_lock(event) for those "
                     "(a partial sum is meaningless). Execution is manual — verify orderbook depth."),
        }


def verify_group_lock(event_ticker, min_edge_cents=0.5):
    """Correctly check one event for an exhaustive-group lock: fetch ALL its
    markets, confirm mutually_exclusive, and sum the FULL set of yes_asks.
    Buying every outcome guarantees $1 iff exactly one resolves Yes."""
    try:
        ev = _get(f"/events/{event_ticker}").get("event", {})
        d = _get("/markets", {"event_ticker": event_ticker, "limit": 500, "status": "open"})
    except Exception as e:
        return {"event": event_ticker, "valid": False, "reason": f"fetch failed: {e}"}
    if not ev.get("mutually_exclusive"):
        return {"event": event_ticker, "valid": False, "reason": "event not mutually exclusive"}
    asks = [_f(m.get("yes_ask_dollars")) for m in d.get("markets", [])]
    asks = [a for a in asks if a is not None and 0 < a < 1]
    if len(asks) < 2:
        return {"event": event_ticker, "valid": False, "reason": "incomplete two-sided quotes"}
    total = sum(asks)
    fee = sum(kalshi_fee(a) for a in asks)
    net = (1.0 - total) - fee
    return {"event": event_ticker, "valid": net >= min_edge_cents / 100.0,
            "n_outcomes": len(asks), "sum_yes_asks_cents": round(total * 100, 1),
            "net_profit_cents": round(net * 100, 2), "mutually_exclusive": True}
