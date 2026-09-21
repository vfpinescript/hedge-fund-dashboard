"""
Snapshot the Alpaca paper account: account summary, open positions with P&L,
recent orders. Appends equity to equity_history.csv and emits JSON for the
dashboard's Live Paper tab.
"""
from __future__ import annotations

import json
import datetime as dt
from pathlib import Path

from alpaca.client import Alpaca

HERE = Path(__file__).resolve().parent
HIST = HERE / "equity_history.csv"
START_EQUITY = 50000.0


def snapshot() -> dict:
    a = Alpaca()
    acct = a.account()
    equity = float(acct.get("equity", 0))

    positions = []
    for p in a.positions():
        positions.append({
            "symbol": p["symbol"],
            "side": "long" if float(p["qty"]) >= 0 else "short",
            "qty": float(p["qty"]),
            "avgPrice": float(p["avg_entry_price"]),
            "lastPrice": float(p["current_price"]),
            "marketValue": float(p["market_value"]),
            "pl": float(p["unrealized_pl"]),
            "plPercent": float(p["unrealized_plpc"]) * 100,
        })

    orders = []
    for o in a.orders(status="all", limit=20):
        orders.append({"symbol": o.get("symbol"), "side": o.get("side"),
                       "qty": float(o.get("qty") or 0), "status": o.get("status"),
                       "price": o.get("filled_avg_price")})

    # append today's equity to history (one row per day)
    today = dt.date.today().isoformat()
    rows = []
    if HIST.exists():
        rows = [l for l in HIST.read_text().strip().splitlines() if l]
    if not rows or not rows[-1].startswith(today):
        HIST.write_text("\n".join(rows + [f"{today},{equity:.2f}"]) + "\n")
    else:
        rows[-1] = f"{today},{equity:.2f}"
        HIST.write_text("\n".join(rows) + "\n")

    return {
        "account": {
            "equity": equity,
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "unrealizedPnl": sum(p["pl"] for p in positions),
            "status": acct.get("status"),
        },
        "positions": positions,
        "orders": orders,
        "start_equity": START_EQUITY,
        "total_pl": equity - START_EQUITY,
        "return_pct": round((equity / START_EQUITY - 1) * 100, 3),
        "as_of": dt.datetime.utcnow().isoformat() + "Z",
    }


if __name__ == "__main__":
    print(json.dumps(snapshot(), indent=2))
