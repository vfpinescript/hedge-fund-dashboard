#!/usr/bin/env python
"""
Precompute the heavy scan grids (Strategy Matrix + Forward Test) in a clean,
standalone process and write them to data_cache/ as JSON. The web server serves
those files directly, so it never runs the multi-second grid compute inside its
own worker (where it is pathologically slow and blocks requests). Run on a
schedule or spawned by the server when the cache is missing/stale.
"""
import os
import sys
import json
import math
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["DASHBOARD_PREWARM"] = "0"          # don't recurse into prewarm
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard import compute  # noqa: E402

CACHE = ROOT / "data_cache"
FILES = {"matrix": "_grid_matrix.json", "forward": "_grid_forward.json",
         "leaderboard": "_grid_leaderboard.json"}


def _json_safe(o):
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _write(key, data):
    CACHE.mkdir(exist_ok=True)
    dst = CACHE / FILES[key]
    fd, tmp = tempfile.mkstemp(dir=str(CACHE), suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(_json_safe(data), f)
    os.replace(tmp, dst)               # atomic
    print(f"wrote {dst.name} ({dst.stat().st_size} bytes)")


if __name__ == "__main__":
    _write("matrix", compute.strategy_matrix())
    _write("forward", compute.forward_test())
    _write("leaderboard", compute.leaderboard())   # reuses the two above from cache
