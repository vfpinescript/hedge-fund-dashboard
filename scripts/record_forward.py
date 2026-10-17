#!/usr/bin/env python
"""
Daily composite forward-test recorder — run once a day (cron / GitHub Action) to
grow the out-of-sample track record of the portfolio composites. Append-only:
each run adds only the new trading days since the last record. Prints a one-line
summary and appends it to data_cache/record.log.
"""
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.research.forward_tracker import update_track, track_summary  # noqa: E402

if __name__ == "__main__":
    added = update_track()
    summ = track_summary()
    n_days = max((r["days"] for r in summ["rows"]), default=0)
    line = (f"{datetime.now().isoformat(timespec='seconds')}  forward_track "
            f"added={json.dumps(added)} days={n_days} inception={summ['inception']}")
    log = Path(__file__).resolve().parents[1] / "data_cache" / "record.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(line + "\n")
    print(line)
