#!/bin/bash
# Daily Alpaca reversal rebalance + snapshot. Safe to run any time — execute.py
# skips when the US market is closed. Set DRY_RUN=0 to trade live paper.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
# Use the local venv if present (Mac), else the environment's python (Railway/cloud).
if [ -x "./venv/bin/python" ]; then PY="./venv/bin/python"; else PY="python"; fi
export DRY_RUN="${DRY_RUN:-0}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] rebalance start (DRY_RUN=$DRY_RUN)"
$PY -m alpaca.execute
echo "[$(date '+%Y-%m-%d %H:%M:%S')] snapshot"
$PY -m alpaca.snapshot > alpaca/last_snapshot.json
echo "[$(date '+%Y-%m-%d %H:%M:%S')] done"
