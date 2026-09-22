"""
Baseline predictive model: RandomForestRegressor predicting next-day
RETURN (% change) from (volume_norm, volatility, momentum), with the
dollar price reconstructed afterwards.

Kept deliberately simple -- an intro-level model a first-year can
inspect and reason about, per the assignment brief. Swap in
XGBoost/LightGBM later only if you want the adversarial-attack demo
to have a sharper, more exploitable decision boundary.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from .features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass
class TrainedModel:
    model: RandomForestRegressor
    mae: float           # in dollars, on the RECONSTRUCTED price (not the raw return)
    r2: float             # in dollars, same reconstruction
    return_r2: float      # in RETURN terms -- the honest number, see note below
    feature_columns: list[str]


def train_baseline_model(df: pd.DataFrame, test_size: float = 0.2) -> TrainedModel:
    """
    Chronological train/test split (no shuffling -- this is a time
    series; shuffling would leak future data into training). The
    model itself is trained on next_return, but MAE/R^2 are reported
    on the reconstructed dollar price (today's Close * (1+predicted
    return)) since that's what's actually meaningful to look at.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]  # next_return

    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=6,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # reconstruct dollar predictions on the test set for reporting
    predicted_returns = model.predict(X_test)
    today_close_test = df["Close"].iloc[split_idx:].values
    predicted_price = today_close_test * (1 + predicted_returns)
    actual_price = df["next_close"].iloc[split_idx:].values

    mae = mean_absolute_error(actual_price, predicted_price)
    r2 = r2_score(actual_price, predicted_price)

    # ALSO score in return terms -- the honest number. Because prices
    # don't move much day-to-day, "tomorrow ~= today" alone gives a
    # deceptively high R^2 once reconstructed into dollars, even if
    # the model barely predicts the actual size/direction of the
    # move. return_r2 measures whether the model learned anything
    # beyond that baseline. Report BOTH in the writeup -- the price
    # R^2 for interpretability, the return R^2 for honesty.
    y_test = df[TARGET_COLUMN].iloc[split_idx:]
    return_r2 = r2_score(y_test, predicted_returns)

    return TrainedModel(
        model=model, mae=mae, r2=r2, return_r2=return_r2, feature_columns=FEATURE_COLUMNS
    )


def predict_latest(trained: TrainedModel, df: pd.DataFrame) -> dict:
    """
    Predicts next_return from the most recent row's features, then
    reconstructs the dollar price using the latest known Close.
    'confidence' is a simple, honest proxy -- the model's held-out
    R^2 (in dollar terms), clipped to [0, 1] -- NOT a statistical
    confidence interval. Good enough for a student-facing "how much
    should I trust this" cue; say so explicitly if asked in the
    report or by a marker.
    """
    latest_row = df.iloc[[-1]][trained.feature_columns]
    latest_close = float(df.iloc[-1]["Close"])

    predicted_return = float(trained.model.predict(latest_row)[0])
    prediction = latest_close * (1 + predicted_return)
    # confidence uses return_r2 (the honest number), NOT price r2 --
    # see the note on TrainedModel.return_r2. Displaying a price-R2-
    # based confidence would overstate how much the model actually
    # knows, which is precisely what Section 6.6 warns against.
    confidence = float(np.clip(trained.return_r2, 0, 1))

    return {
        "prediction": round(prediction, 2),
        "confidence": round(confidence, 2),
        "mae": round(trained.mae, 2),
        "inputs_used": latest_row.iloc[0].round(2).to_dict(),
        "baseline_close": round(latest_close, 2),
    }


def predict_with_overrides(trained: TrainedModel, overrides: dict, baseline_close: float) -> dict:
    """
    Predicts using manually supplied feature values instead of the
    latest row -- this is the hook the VR sliders (and the adversarial
    attack) call: pass {"volume_norm": 64, "volatility": 31, "momentum": 0.9}
    plus the current baseline price, and get back what the model
    predicts for THAT combination of inputs.
    """
    row = pd.DataFrame([{col: overrides[col] for col in trained.feature_columns}])
    predicted_return = float(trained.model.predict(row)[0])
    prediction = baseline_close * (1 + predicted_return)
    return {"prediction": round(prediction, 2), "inputs_used": overrides}