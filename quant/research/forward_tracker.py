"""
Composite forward-test tracker — a REAL out-of-sample record.

A backtest sliced to a recent window is still in-sample: the model was chosen
knowing that data. A genuine forward test freezes the strategy today and records
what it does from this moment on. This module does exactly that. From INCEPTION
it appends each new day's return per composite to an append-only store; once a
date is recorded it is NEVER overwritten, so the track stays immutable even if
the model is later changed. The accumulating record is the only honest evidence
of live skill, as opposed to a flattering backtest.

Run `python scripts/record_forward.py` once a day (wired into the rebalance
GitHub Action) to grow the record. `track_summary()` reports OOS metrics
computed only from the recorded returns.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .composites import COMPOSITES

ROOT = Path(__file__).resolve().parents[2]
TRACK_FILE = ROOT / "data_cache" / "composite_forward_track.json"
# The day composite forward tracking began. Everything before this is backtest
# (visible on the Backtest/Matrix tabs); everything from here is a live record.
INCEPTION = "2026-10-17"


def _load() -> dict:
    try:
        return json.loads(TRACK_FILE.read_text())
    except Exception:
        return {"inception": INCEPTION, "tracks": {}}


def _save(d: dict) -> None:
    TRACK_FILE.parent.mkdir(exist_ok=True)
    TRACK_FILE.write_text(json.dumps(d, separators=(",", ":")))


def update_track(as_of: str | None = None) -> dict:
    """Append any new (date, return) rows per composite since the last record.
    Append-only: a date already in the store is left untouched. Returns the
    count of new rows added per composite."""
    store = _load()
    tracks = store.setdefault("tracks", {})
    added = {}
    for cid, spec in COMPOSITES.items():
        try:
            strat, _ = spec["fn"]()
        except Exception:
            added[cid] = 0
            continue
        if strat is None or len(strat) == 0:
            added[cid] = 0
            continue
        strat = strat[strat.index >= pd.Timestamp(INCEPTION)]
        rec = tracks.setdefault(cid, {})
        before = len(rec)
        for d, r in strat.items():
            key = d.strftime("%Y-%m-%d")
            if key not in rec and np.isfinite(r):      # append-only
                rec[key] = round(float(r), 6)
        added[cid] = len(rec) - before
    store["inception"] = INCEPTION
    store["last_update"] = as_of or date.today().isoformat()
    _save(store)
    return added


def track_summary() -> dict:
    """Out-of-sample metrics per composite, computed ONLY from recorded returns.
    Ranked by realized return. Days=0 means the record is still accumulating."""
    store = _load()
    rows = []
    for cid, spec in COMPOSITES.items():
        rec = store.get("tracks", {}).get(cid, {})
        if not rec:
            rows.append({"id": cid, "name": spec["name"], "family": spec.get("family"),
                         "days": 0, "return": None, "sharpe": None,
                         "sortino": None, "max_dd": None})
            continue
        s = pd.Series(rec).sort_index()
        r = s.values.astype(float)
        eq = np.cumprod(1 + r)
        tot = float((eq[-1] - 1) * 100)
        sh = float(r.mean() / r.std() * np.sqrt(252)) if len(r) > 1 and r.std() > 0 else None
        downside = r[r < 0]
        so = (float(r.mean() / downside.std() * np.sqrt(252))
              if len(downside) > 1 and downside.std() > 0 else None)
        mdd = float((eq / np.maximum.accumulate(eq) - 1).min() * 100)
        rows.append({"id": cid, "name": spec["name"], "family": spec.get("family"),
                     "days": len(r), "return": round(tot, 2),
                     "sharpe": round(sh, 2) if sh is not None else None,
                     "sortino": round(so, 2) if so is not None else None,
                     "max_dd": round(mdd, 2)})
    rows.sort(key=lambda x: -(x["return"] if x["return"] is not None else -1e9))
    return {"inception": store.get("inception", INCEPTION),
            "last_update": store.get("last_update"),
            "as_of": pd.Timestamp.utcnow().strftime("%Y-%m-%d"),
            "rows": rows}


if __name__ == "__main__":
    print("added:", update_track())
    import json as _j
    print(_j.dumps(track_summary(), indent=2))
