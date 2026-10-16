"""
Congressional trading feed (free, via the existing FMP key).

Follows the most recent stock/bond transactions disclosed by members of the U.S.
Senate and House, from Financial Modeling Prep's senate-latest / house-latest
endpoints (which the fund's FMP key covers; per-symbol history needs a higher
tier). This is informational, a public record of what politicians disclosed, not
a trade signal and not wired into the systematic engine.

The paid QuiverQuant connector would add per-ticker history, insider trades,
lobbying, gov contracts etc., but it requires an active Quiver subscription.
"""
from __future__ import annotations

import os
import re
import time

import requests

BASE = "https://financialmodelingprep.com/stable"
_CACHE: dict = {}
_TTL = 2 * 3600


def amount_mid(a: str) -> float:
    """Midpoint of an FMP amount range like '$1,001 - $15,000'."""
    nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]{2,}", a or "")]
    if len(nums) >= 2:
        return (nums[0] + nums[1]) / 2.0
    return float(nums[0]) if nums else 0.0


def _norm(rec: dict, chamber: str) -> dict:
    name = f"{rec.get('firstName','').strip()} {rec.get('lastName','').strip()}".strip()
    return {
        "name": name or rec.get("office", ""),
        "chamber": chamber,
        "district": rec.get("district", ""),
        "ticker": (rec.get("symbol") or "").strip(),
        "asset": rec.get("assetDescription", ""),
        "assetType": rec.get("assetType", ""),
        "type": rec.get("type", ""),
        "amount": rec.get("amount", ""),
        "transactionDate": rec.get("transactionDate", ""),
        "disclosureDate": rec.get("disclosureDate", ""),
        "link": rec.get("link", ""),
    }


def latest_trades(pages: int = 2) -> list[dict]:
    """Combined recent Senate + House disclosures, newest first. Cached.
    (FMP's default page size is 100; a `limit` param is gated to <=25 on this
    tier, so we omit it and paginate instead.)"""
    key = f"latest:{pages}"
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    api = os.environ.get("FMP_API_KEY")
    out: list[dict] = []
    if api:
        for chamber, ep in (("senate", "senate-latest"), ("house", "house-latest")):
            for p in range(pages):
                try:
                    r = requests.get(f"{BASE}/{ep}",
                                     params={"page": p, "apikey": api},
                                     timeout=25)
                    if r.status_code != 200:
                        break
                    data = r.json()
                    if not isinstance(data, list) or not data:
                        break
                    out.extend(_norm(x, chamber) for x in data)
                except Exception:
                    break
    out.sort(key=lambda x: x.get("disclosureDate") or "", reverse=True)
    _CACHE[key] = (time.time(), out)
    return out


def latest_insider(pages: int = 1) -> list[dict]:
    """Recent corporate insider (exec/director) Form-4 transactions via FMP."""
    key = f"insider:{pages}"
    hit = _CACHE.get(key)
    if hit and time.time() - hit[0] < _TTL:
        return hit[1]
    api = os.environ.get("FMP_API_KEY")
    out: list[dict] = []
    if api:
        for p in range(pages):
            try:
                r = requests.get(f"{BASE}/insider-trading/latest",
                                 params={"page": p, "apikey": api}, timeout=25)
                if r.status_code != 200:
                    break
                data = r.json()
                if not isinstance(data, list) or not data:
                    break
                for rec in data:
                    ad = (rec.get("acquisitionOrDisposition") or "").upper()
                    shares = rec.get("securitiesTransacted") or 0
                    price = rec.get("price") or 0
                    out.append({
                        "ticker": (rec.get("symbol") or "").strip(),
                        "name": rec.get("reportingName", ""),
                        "role": rec.get("typeOfOwner", ""),
                        "side": "Buy" if ad == "A" else "Sell" if ad == "D" else ad,
                        "shares": shares, "price": price,
                        "value": round(float(shares) * float(price)) if shares and price else 0,
                        "date": rec.get("transactionDate", ""),
                        "filed": rec.get("filingDate", ""),
                        "formType": rec.get("transactionType", ""),
                    })
            except Exception:
                break
    out.sort(key=lambda x: x.get("filed") or "", reverse=True)
    _CACHE[key] = (time.time(), out)
    return out
