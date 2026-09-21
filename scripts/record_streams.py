#!/usr/bin/env python
"""
Daily alt-stream recorder — run once a day (cron) to grow the news-sentiment
history and log the Kalshi arbitrage snapshot. Prints a one-line summary and
appends a timestamped line to data_cache/record.log.
"""
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.research.alt_streams import record_today  # noqa: E402

if __name__ == "__main__":
    res = record_today()
    line = f"{datetime.now().isoformat(timespec='seconds')}  {json.dumps(res)}"
    log = Path(__file__).resolve().parents[1] / "data_cache" / "record.log"
    log.parent.mkdir(exist_ok=True)
    with open(log, "a") as f:
        f.write(line + "\n")
    print(line)
