"""
Reconcile the live Alpaca paper account to the reversal strategy's targets.

DRY_RUN=1 (default) prints the plan and places NOTHING. DRY_RUN=0 submits
market orders. Reconciliation is delta-based: it trades only the difference
between current and target shares, so re-running is idempotent.
"""
from __future__ import annotations

import os
import json

from alpaca.client import Alpaca
from alpaca.strategy import compute_targets, UNIVERSE

DRY_RUN = os.environ.get("DRY_RUN", "1") != "0"
MIN_TRADE_DOLLARS = 50.0        # skip dust rebalances


def plan_and_execute() -> dict:
    a = Alpaca()
    tgt = compute_targets(a)
    if "error" in tgt:
        return {"error": tgt["error"]}

    clock = a.clock()
    positions = {p["symbol"]: float(p["qty"]) for p in a.positions()}
    plan = []

    # symbols we want to hold
    for sym, t in tgt["targets"].items():
        cur = positions.get(sym, 0.0)
        want = t["target_shares"]
        delta = want - cur
        price = t["price"]
        if abs(delta * price) < MIN_TRADE_DOLLARS:
            continue
        plan.append({"symbol": sym, "side": "buy" if delta > 0 else "sell",
                     "qty": round(abs(delta), 3), "cur": cur, "target": want,
                     "signal": t["reversal_signal"], "est_dollars": round(abs(delta) * price, 2)})

    # symbols held but no longer targeted -> close
    for sym, cur in positions.items():
        if sym not in tgt["targets"] and abs(cur) > 0:
            plan.append({"symbol": sym, "side": "sell" if cur > 0 else "buy",
                         "qty": round(abs(cur), 3), "cur": cur, "target": 0.0,
                         "signal": "exit", "est_dollars": None})

    results = []
    if not DRY_RUN:
        if not clock.get("is_open"):
            return {"dry_run": False, "skipped": "market_closed",
                    "note": "US market closed — orders would reject. Run during 09:30-16:00 ET.",
                    "plan": plan, "as_of": tgt["as_of"]}
        for o in plan:
            r = a.submit_order(o["symbol"], o["qty"], o["side"])
            results.append({"symbol": o["symbol"], "side": o["side"], "qty": o["qty"],
                            "order_id": r.get("id"), "status": r.get("status"),
                            "error": r.get("message")})

    return {
        "dry_run": DRY_RUN,
        "as_of": tgt["as_of"],
        "market_open": clock.get("is_open"),
        "equity": tgt["equity"],
        "target_gross": tgt["target_gross"],
        "n_orders": len(plan),
        "plan": plan,
        "submitted": results,
    }


if __name__ == "__main__":
    print(json.dumps(plan_and_execute(), indent=2))
