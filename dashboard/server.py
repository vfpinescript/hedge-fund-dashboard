"""
Horizon-style hedge-fund dashboard — local Flask server.

Run:  ./venv/bin/python dashboard/server.py   (serves http://localhost:8010)

Serves a dark, institutional single-page UI plus JSON APIs computed live from
real market data via the quant engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import compute

HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")


@app.route("/")
def index():
    return send_from_directory(HERE / "static", "index.html")


@app.route("/api/strategies")
def api_strategies():
    return jsonify(compute.build_strategies())


@app.route("/api/signals")
def api_signals():
    inst = request.args.get("instrument", "XAUUSD")
    try:
        return jsonify(compute.live_signals(inst))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/surfaces")
def api_surfaces():
    inst = request.args.get("instrument", "XAUUSD")
    try:
        return jsonify(compute.surfaces(inst))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy_lab")
def api_lab():
    try:
        return jsonify(compute.strategy_lab())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/strategy_library")
def api_strategy_library():
    return jsonify(compute.strategy_library())


@app.route("/api/strategy_run")
def api_strategy_run():
    try:
        return jsonify(compute.strategy_run(request.args.get("strategy"),
                                            request.args.get("instrument")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/live")
def api_live():
    try:
        return jsonify(compute.live_paper())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/kalshi")
def api_kalshi():
    try:
        return jsonify(compute.kalshi_scan())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Warming strategy cache (first load computes backtests)...")
    try:
        compute.build_strategies()
        print("  ready.")
    except Exception as e:
        print("  warmup failed:", e)
    app.run(host="127.0.0.1", port=8010, debug=False, threaded=True)
