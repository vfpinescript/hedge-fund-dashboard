"""
Information-diffusion / news-sentiment agent.

Honest scope: the theory is that news arrivals cluster (self-excitation, like a
Hawkes process), so accelerating coverage signals an attention/volatility regime.
In practice, Firecrawl's search saturates its result cap, so a reliable
news-ARRIVAL INTENSITY isn't measurable from it — a true diffusion model needs a
dated-news feed with full counts (a paid news API). What IS reliable is the
SENTIMENT of recent coverage, so this agent delivers a news-sentiment directional
tilt (keyword-scored) with a coarse attention proxy in diagnostics. It is
deliberately low-weight (news is noisy) and ABSTAINS if the feed is unavailable —
it never fabricates coverage. Genuinely uncorrelated with price-based signals.

Data: Firecrawl search with a time window (needs FIRECRAWL_API_KEY).
"""
from __future__ import annotations

import os
import ssl
import json
import urllib.request
from pathlib import Path

import certifi

from ..base import DirectionalAgent, DirectionalSignal, Direction

_SSL = ssl.create_default_context(cafile=certifi.where())

# instrument -> news query
QUERY = {
    "BTCUSD": "Bitcoin", "ETHUSD": "Ethereum", "SOLUSD": "Solana",
    "XAUUSD": "gold price", "XAGUSD": "silver price",
    "USOIL": "crude oil price", "UKOIL": "Brent crude oil", "NATGAS": "natural gas price",
    "EURUSD": "euro dollar exchange rate", "GBPUSD": "British pound", "USDJPY": "Japanese yen",
    "ES1!": "S&P 500", "NQ1!": "Nasdaq 100", "SPX": "S&P 500", "NDX": "Nasdaq",
}
BULL = ("surge", "rally", "soar", "gain", "jump", "rise", "bull", "record high",
        "upgrade", "beat", "climb", "boost", "optimism", "high")
BEAR = ("plunge", "crash", "fall", "drop", "slump", "bear", "selloff", "sell-off",
        "downgrade", "miss", "warn", "tumble", "fear", "low", "concern")


def _firecrawl_key():
    k = os.environ.get("FIRECRAWL_API_KEY")
    if k:
        return k
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("FIRECRAWL_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _search_news(query, tbs, limit=20, key=None):
    """Firecrawl news search over a time window (tbs: 'qdr:d' day, 'qdr:w' week)."""
    key = key or _firecrawl_key()
    if not key:
        return None
    # This Firecrawl version supports `tbs` (time window) but not `sources`.
    body = json.dumps({"query": query, "limit": limit, "tbs": tbs}).encode()
    req = urllib.request.Request(
        "https://api.firecrawl.dev/v1/search", data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=45, context=_SSL).read())
    except Exception:
        return None
    data = r.get("data", [])
    if isinstance(data, dict):
        return data.get("news") or data.get("web") or []
    return data or []


def _sentiment(titles):
    text = " ".join(titles).lower()
    b = sum(text.count(w) for w in BULL)
    s = sum(text.count(w) for w in BEAR)
    tot = b + s
    return (b - s) / tot if tot else 0.0, b, s


class InfoDiffusionAgent(DirectionalAgent):
    name = "info_diffusion"
    weight = 0.4                       # news is noisy — low weight

    def __init__(self, min_articles: int = 3):
        self.min_articles = min_articles

    def evaluate(self, instrument: str, ohlcv=None, **kwargs) -> DirectionalSignal:
        key = _firecrawl_key()
        if not key:
            return self._abstain(instrument, "no Firecrawl key")
        q = QUERY.get(instrument.upper(), instrument)

        week = _search_news(q, "qdr:w", 40, key)
        if week is None:
            return self._abstain(instrument, "news feed unavailable")
        n = len(week)
        if n < self.min_articles:
            return self._abstain(instrument, "insufficient news coverage")

        titles = [(x.get("title") or "") for x in week]
        sent, nb, ns = _sentiment(titles)

        # Direction from news sentiment; conviction from its strength & agreement.
        agreement = abs(nb - ns) / max(nb + ns, 1)            # how one-sided
        conviction = min(1.0, abs(sent) * (0.5 + 0.5 * agreement))
        direction = (Direction.BULL if sent > 0.1 else
                     Direction.BEAR if sent < -0.1 else Direction.NEUTRAL)

        return DirectionalSignal(
            self.name, instrument, direction, conviction,
            rationale=(f"{n} recent articles, sentiment {sent:+.2f} "
                       f"({nb} bullish / {ns} bearish keywords)"),
            diagnostics={"articles": n, "sentiment": sent,
                         "bull_kw": nb, "bear_kw": ns, "attention": min(1.0, n / 40)},
        )
