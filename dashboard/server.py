"""
Horizon-style hedge-fund dashboard — local Flask server.

Run:  ./venv/bin/python dashboard/server.py   (serves http://localhost:8010)

Serves a dark, institutional single-page UI plus JSON APIs computed live from
real market data via the quant engine.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask.json.provider import DefaultJSONProvider

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv(path):
    """Minimal .env loader (no dependency) so the app has its API keys no matter
    how it is launched (gunicorn, the IDE preview panel, a bare python call).
    Only fills keys that are not already set in the environment."""
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    except FileNotFoundError:
        pass


_load_dotenv(ROOT / ".env")

from dashboard import compute


def _json_safe(obj):
    """Recursively replace NaN/Infinity floats with None so the payload is
    valid JSON (JSON has no NaN token; browsers' JSON.parse rejects it)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


class SafeJSONProvider(DefaultJSONProvider):
    """Serializes non-finite floats (NaN/Inf) as null across every endpoint."""
    def dumps(self, obj, **kwargs):
        return super().dumps(_json_safe(obj), **kwargs)


HERE = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(HERE / "static"), static_url_path="/static")
app.json = SafeJSONProvider(app)


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


@app.route("/api/strategy_matrix")
def api_strategy_matrix():
    try:
        return jsonify(compute.strategy_matrix())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forward_test")
def api_forward_test():
    try:
        return jsonify(compute.forward_test())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forward_series")
def api_forward_series():
    try:
        return jsonify(compute.forward_series(request.args.get("strategy"),
                                              request.args.get("instrument")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analytics")
def api_analytics():
    try:
        return jsonify(compute.analytics(request.args.get("strategy"),
                                         request.args.get("instrument")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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


@app.route("/api/desks")
def api_desks():
    try:
        return jsonify(compute.desk_accounts())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/anomaly")
def api_anomaly():
    try:
        return jsonify(compute.anomaly_scan())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro")
def api_macro():
    try:
        return jsonify(compute.macro_monitor())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/macro_signals")
def api_macro_signals():
    try:
        return jsonify(compute.macro_signals())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/news")
def api_news():
    try:
        return jsonify(compute.news_view(request.args.get("ticker", "AAPL")))
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
