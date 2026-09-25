"""
Flask API bridging the Python model to the Babylon.js scene.

Trains the model ONCE at startup (not per-request -- retraining on
every slider move would be slow and pointless, the model doesn't
need to change, only the inputs fed into it do), then serves three
endpoints:

    GET  /api/scene-data        -> the same payload main.py exports to JSON
    POST /api/predict           -> { volume_norm, volatility, momentum } -> prediction
    POST /api/adversarial-attack -> { volume_norm, volatility, momentum } -> attack result

Run with:
    python api.py --csv data/raw/StockPriceDataset.csv --ticker AAPL

Then point the scene's fetch calls at http://localhost:5000/api/...
CORS is enabled so the scene (served separately via `python -m
http.server` from the scene/ folder, or any static host) can call it.
"""

import argparse

from flask import Flask, jsonify, request
from flask_cors import CORS

from src.data_loader import load_or_synthesize
from src.features import engineer_features
from src.model import (
    train_baseline_model,
    predict_latest,
    predict_with_overrides,
    feature_contributions,
)
from src.adversarial import run_adversarial_attack
from src.export_scene_data import build_scene_payload

app = Flask(__name__)
CORS(app)

# populated by main() at startup -- see bottom of file
STATE = {}


@app.route("/api/scene-data", methods=["GET"])
def scene_data():
    """Same payload as main.py's exported JSON -- lets the scene
    fetch fresh data from a live server instead of a static file."""
    return jsonify(STATE["payload"])


@app.route("/api/health", methods=["GET"])
def health():
    """Matches api/index.py's health route so scene/index.html's
    status badge works the same way against local dev and prod."""
    return jsonify({"status": "ok", "mae": round(STATE["trained"].mae, 2)})


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Body: { "volume_norm": 64, "volatility": 31, "momentum": 0.42 }
    Returns: { "prediction": 193.3, "inputs_used": {...} }
    This is what the VR sliders call every time the user drags one.
    """
    overrides = request.get_json()
    missing = [c for c in STATE["trained"].feature_columns if c not in overrides]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    result = predict_with_overrides(STATE["trained"], overrides, STATE["baseline_close"])
    result["contributions"] = feature_contributions(STATE["trained"], overrides, STATE["baseline_close"])
    return jsonify(result)


@app.route("/api/adversarial-attack", methods=["POST"])
def adversarial_attack():
    """
    Body: { "volume_norm": 64, "volatility": 31, "momentum": 0.42 }
        (the CURRENT slider values -- the attack searches for the
        worst nudge starting FROM here, not from the original baseline)
    Returns: original/attacked predictions, the shift, and which
    inputs the attack perturbed -- everything the "RUN ADVERSARIAL
    ATTACK" button needs to visualize the result.
    """
    current_inputs = request.get_json()
    missing = [c for c in STATE["trained"].feature_columns if c not in current_inputs]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    attack = run_adversarial_attack(
        STATE["trained"], current_inputs, STATE["baseline_close"]
    )
    return jsonify(
        {
            "original_prediction": attack.original_prediction,
            "attacked_prediction": attack.attacked_prediction,
            "shift_dollars": attack.shift_dollars,
            "shift_percent": attack.shift_percent,
            "perturbed_inputs": attack.perturbed_inputs,
            "original_inputs": attack.original_inputs,
        }
    )


def load_and_train(csv_path: str, ticker: str | None) -> None:
    print(f"Loading + training on startup (ticker={ticker or 'N/A'})...")
    raw_df = load_or_synthesize(csv_path, ticker)
    feat_df = engineer_features(raw_df)
    trained = train_baseline_model(feat_df)
    result = predict_latest(trained, feat_df)

    dataset_id = (ticker or "synthetic").lower()
    label = f"{ticker or 'Synthetic Stock'} — Historical Price Data"
    payload = build_scene_payload(dataset_id, label, feat_df, result, importances=trained.importances)

    STATE["trained"] = trained
    STATE["baseline_close"] = result["baseline_close"]
    STATE["payload"] = payload

    print(f"  Ready. Baseline close: {STATE['baseline_close']}, MAE: {trained.mae:.2f}, "
          f"return R^2: {trained.return_r2:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Analytics Lab API server")
    parser.add_argument("--csv", default="data/raw/StockPriceDataset.csv")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    load_and_train(args.csv, args.ticker)
    app.run(debug=True, port=args.port)