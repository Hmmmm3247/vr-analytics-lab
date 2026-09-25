"""
Builds the JSON file the VR scene reads. This is the ONLY contract
between the Python/ML side and the 3D side -- the scene never
hardcodes 'volume' or 'sentiment', it just loops over `inputs`.

Keeping this shape stable means swapping datasets later is a new
JSON file, not a scene rebuild.
"""

import json
import os

import pandas as pd

from .features import FEATURE_COLUMNS

INPUT_METADATA = {
    "volume_norm": {
        "label": "Trading Volume",
        "min": 0,
        "max": 100,
        "source": "Volume, rolling 252d min-max scaled",
    },
    "volatility": {
        "label": "Volatility Index",
        "min": 0,
        "max": 100,
        "source": "(High-Low)/Close, rolling 14d mean",
    },
    "momentum": {
        "label": "Price Momentum",
        "min": -1,
        "max": 1,
        "source": "% change in Close, rolling 5d",
    },
}


def build_scene_payload(
    dataset_id: str,
    label: str,
    df: pd.DataFrame,
    prediction_result: dict,
    series_length: int = 12,
    importances: dict | None = None,
) -> dict:
    """
    df: the engineered DataFrame (must have FEATURE_COLUMNS + Close).
    prediction_result: output of model.predict_latest().
    series_length: how many recent closing prices to send for the
        floating bar-chart / line visual (kept short -- the VR scene
        renders this many bars, not the full history).
    """
    inputs = []
    latest = prediction_result["inputs_used"]
    for col in FEATURE_COLUMNS:
        meta = INPUT_METADATA[col]
        entry = {
            "key": col,
            "label": meta["label"],
            "min": meta["min"],
            "max": meta["max"],
            "default": round(float(latest[col]), 2),
        }
        if importances:
            entry["importance"] = round(importances[col], 4)
        inputs.append(entry)

    recent_series = (
        df["Close"].tail(series_length).round(2).tolist()
    )

    # real OHLC for candlestick rendering, not just closing price.
    # NOTE: High/Low/Open are the raw (non-split-adjusted) values --
    # only Close was swapped to the adjusted series in data_loader.
    # Fine for a short recent window (unlikely a split lands inside
    # the last `series_length` days), but flag this in the report if
    # you ever extend the window across a known split date.
    recent = df.tail(series_length)
    candles = [
        {
            "date": str(d.date()) if hasattr(d, "date") else str(d),
            "open": round(float(o), 2),
            "high": round(float(h), 2),
            "low": round(float(l), 2),
            "close": round(float(c), 2),
        }
        for d, o, h, l, c in zip(
            recent["Date"], recent["Open"], recent["High"], recent["Low"], recent["Close"]
        )
    ]

    return {
        "dataset_id": dataset_id,
        "label": label,
        "inputs": inputs,
        "prediction": {
            "label": "Next-Day Close",
            "unit": "$",
            "value": prediction_result["prediction"],
            "confidence": prediction_result["confidence"],
        },
        "series": recent_series,
        "candles": candles,
    }


def save_scene_payload(payload: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved scene data -> {output_path}")