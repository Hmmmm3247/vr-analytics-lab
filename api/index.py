"""
Serverless Flask API for Vercel.

Self-contained on purpose: does NOT import from src/. Vercel packages
each Python function somewhat independently, and crossing the api/ <->
src/ boundary is a common source of import-path breakage on serverless
platforms, so prediction + adversarial-attack logic is duplicated here
in plain numpy rather than shared via package import. The model itself
is trained once, locally, by train_and_save.py -- this file only loads
the resulting api/model.pkl and serves predictions. It never trains.

Routes:
    GET  /api/health              -> { status: "ok" }
    POST /api/predict             -> { volume_norm, volatility, momentum } -> prediction
    POST /api/adversarial-attack  -> { volume_norm, volatility, momentum } -> attack result
"""

import os
import pickle

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(MODEL_PATH, "rb") as f:
    BUNDLE = pickle.load(f)

MODEL = BUNDLE["model"]
FEATURE_COLUMNS = BUNDLE["feature_columns"]
BASELINE_CLOSE = BUNDLE["baseline_close"]

INPUT_BOUNDS = {
    "volume_norm": (0, 100),
    "volatility": (0, 100),
    "momentum": (-1, 1),
}


def _predict(inputs: dict) -> float:
    row = np.array([[inputs[col] for col in FEATURE_COLUMNS]])
    predicted_return = float(MODEL.predict(row)[0])
    return round(BASELINE_CLOSE * (1 + predicted_return), 2)


def _missing_fields(body: dict) -> list[str]:
    return [c for c in FEATURE_COLUMNS if c not in body]


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "mae": round(BUNDLE["mae"], 2), "return_r2": round(BUNDLE["return_r2"], 3)})


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    missing = _missing_fields(body)
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    prediction = _predict(body)
    return jsonify({"prediction": prediction, "inputs_used": body})


@app.route("/api/adversarial-attack", methods=["POST"])
def adversarial_attack():
    """
    Simplified stand-in for a gradient-based attack: for each feature,
    nudge it +/- 15% of its allowed range from the CURRENT slider
    values and keep whichever single nudge shifts the prediction the
    most. Finite-difference search, not a true gradient (RandomForest
    has none) -- mirrors src/adversarial.py's approach.
    """
    baseline_inputs = request.get_json(silent=True) or {}
    missing = _missing_fields(baseline_inputs)
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    epsilon = 0.15
    original_prediction = _predict(baseline_inputs)

    worst_shift = 0.0
    worst_inputs = dict(baseline_inputs)

    for feature in FEATURE_COLUMNS:
        min_val, max_val = INPUT_BOUNDS[feature]
        nudge = (max_val - min_val) * epsilon

        for direction in (1, -1):
            candidate = dict(baseline_inputs)
            candidate[feature] = max(min_val, min(max_val, baseline_inputs[feature] + direction * nudge))

            shift = abs(_predict(candidate) - original_prediction)
            if shift > worst_shift:
                worst_shift = shift
                worst_inputs = candidate

    attacked_prediction = _predict(worst_inputs)
    shift_dollars = round(attacked_prediction - original_prediction, 2)
    shift_percent = round((shift_dollars / original_prediction) * 100, 2) if original_prediction else 0.0

    return jsonify(
        {
            "original_prediction": original_prediction,
            "attacked_prediction": attacked_prediction,
            "shift_dollars": shift_dollars,
            "shift_percent": shift_percent,
            "perturbed_inputs": worst_inputs,
            "original_inputs": baseline_inputs,
        }
    )
