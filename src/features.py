"""
Feature engineering for the VR Predictive Analytics Lab.

Turns raw OHLCV (Open/High/Low/Close/Volume) columns into the three
model inputs the scene reads: volume, volatility, momentum.

Column names expected on the input DataFrame:
    Date, Open, High, Low, Close, Volume
(Adj Close is optional and unused here.)
"""

import pandas as pd
import numpy as np


def normalize_volume(df: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Rolling min-max scale of Volume into a 0-100 range.
    window=252 ~= one trading year, so 'high volume' is relative to
    the stock's own recent history, not an arbitrary fixed number.
    """
    roll_min = df["Volume"].rolling(window, min_periods=20).min()
    roll_max = df["Volume"].rolling(window, min_periods=20).max()
    scaled = (df["Volume"] - roll_min) / (roll_max - roll_min).replace(0, np.nan)
    return (scaled * 100).clip(0, 100)


def compute_volatility(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Rolling standard deviation of daily % returns on Close, scaled to
    0-100. Deliberately NOT using (High-Low)/Close here: High/Low are
    raw (not split-adjusted) while Close has been swapped to the
    split-adjusted price in data_loader -- mixing the two would create
    a fake volatility spike right at every stock-split date. Return
    volatility only needs Close, so it stays consistent.
    """
    daily_return = df["Close"].pct_change()
    rolling_std = daily_return.rolling(window, min_periods=5).std()
    # scale so typical single-digit-percent daily vol lands in a readable 0-100 band
    scaled = (rolling_std * 100 * 5).clip(0, 100)
    return scaled


def compute_momentum(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """
    Rolling % change in Close over `window` days, clipped to [-1, 1].
    Positive = price trending up recently, negative = trending down.
    """
    momentum = df["Close"].pct_change(periods=window)
    return momentum.clip(-1, 1)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds volume_norm, volatility, momentum columns and a next_close
    target column (tomorrow's Close, for supervised training).
    Drops rows with NaNs introduced by rolling windows / the shift.
    """
    df = df.sort_values("Date").reset_index(drop=True).copy()

    df["volume_norm"] = normalize_volume(df)
    df["volatility"] = compute_volatility(df)
    df["momentum"] = compute_momentum(df)

    # target: next day's % RETURN, not the raw price.
    # A RandomForest trained on bounded, scale-free features
    # (volume_norm/volatility/momentum) to predict an absolute price
    # can only ever output values it saw during training. Since
    # AAPL's price drifted from ~$20 (2014) to ~$190 (2023), the
    # model has no way to know "today" sits at the top of a decade-
    # long uptrend -- it just matches today's features to whichever
    # historical day looked similar and returns THAT day's price.
    # Predicting the return instead keeps the target roughly
    # stationary; the actual price is reconstructed afterwards as
    # today's_close * (1 + predicted_return).
    df["next_close"] = df["Close"].shift(-1)
    df["next_return"] = (df["next_close"] - df["Close"]) / df["Close"]

    feature_cols = ["volume_norm", "volatility", "momentum"]
    df = df.dropna(subset=feature_cols + ["next_return"]).reset_index(drop=True)

    return df


FEATURE_COLUMNS = ["volume_norm", "volatility", "momentum"]
TARGET_COLUMN = "next_return"