"""
Minimal Alpaca paper-trading client (stdlib + requests only).

Reads credentials from the project .env (ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY).
Paper endpoints only — this never touches a live-money account.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import requests

PAPER_BASE = "https://paper-api.alpaca.markets"
DATA_BASE = "https://data.alpaca.markets"


def _load_env() -> None:
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


class Alpaca:
    def __init__(self):
        _load_env()
        self.key = os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise RuntimeError("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set in .env")
        self.h = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    # ---- account / positions / orders ----
    def account(self) -> dict:
        return requests.get(f"{PAPER_BASE}/v2/account", headers=self.h, timeout=20).json()

    def clock(self) -> dict:
        return requests.get(f"{PAPER_BASE}/v2/clock", headers=self.h, timeout=20).json()

    def positions(self) -> list:
        return requests.get(f"{PAPER_BASE}/v2/positions", headers=self.h, timeout=20).json()

    def orders(self, status: str = "all", limit: int = 50) -> list:
        return requests.get(f"{PAPER_BASE}/v2/orders",
                            headers=self.h, params={"status": status, "limit": limit},
                            timeout=20).json()

    def submit_order(self, symbol: str, qty: float, side: str,
                     type_: str = "market", tif: str = "day") -> dict:
        """side: 'buy'|'sell'. qty in shares (fractional allowed for market/day)."""
        body = {"symbol": symbol, "qty": str(qty), "side": side,
                "type": type_, "time_in_force": tif}
        return requests.post(f"{PAPER_BASE}/v2/orders", headers=self.h, json=body, timeout=20).json()

    def close_position(self, symbol: str) -> dict:
        return requests.delete(f"{PAPER_BASE}/v2/positions/{symbol}", headers=self.h, timeout=20).json()

    def asset(self, symbol: str) -> dict:
        return requests.get(f"{PAPER_BASE}/v2/assets/{symbol}", headers=self.h, timeout=20).json()

    # ---- market data ----
    def latest_trade(self, symbol: str) -> Optional[float]:
        r = requests.get(f"{DATA_BASE}/v2/stocks/{symbol}/trades/latest",
                         headers=self.h, timeout=20).json()
        try:
            return float(r["trade"]["p"])
        except Exception:
            return None

    def daily_bars(self, symbol: str, days: int = 400) -> "list[dict]":
        """Daily bars, oldest→newest. Uses IEX feed (free tier)."""
        import datetime as dt
        start = (dt.datetime.utcnow() - dt.timedelta(days=days * 2)).strftime("%Y-%m-%d")
        out, page = [], None
        while True:
            params = {"timeframe": "1Day", "start": start, "limit": 10000, "feed": "iex",
                      "adjustment": "all"}
            if page:
                params["page_token"] = page
            r = requests.get(f"{DATA_BASE}/v2/stocks/{symbol}/bars",
                             headers=self.h, params=params, timeout=30).json()
            out.extend(r.get("bars") or [])
            page = r.get("next_page_token")
            if not page:
                break
        return out[-days:]


if __name__ == "__main__":
    a = Alpaca()
    acct = a.account()
    print("account:", acct.get("status"), "equity", acct.get("equity"), "cash", acct.get("cash"))
    print("clock:", a.clock())
