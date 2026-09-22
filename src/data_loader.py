"""
Loads the Kaggle OHLCV CSV and filters to one ticker.

Handles a bit of column-name variation since Kaggle stock datasets
aren't perfectly standardized (some use 'Name', some 'Ticker', some
'Symbol' for the ticker column; some use 'Adj Close', some don't).
"""

import os
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
TICKER_COLUMN_CANDIDATES = ["Name", "Ticker", "Symbol", "ticker", "symbol"]
ADJ_CLOSE_CANDIDATES = ["Adj Close", "Adj_Close", "AdjClose", "adj_close", "Adjusted Close"]


def _find_adj_close_column(df: pd.DataFrame) -> str | None:
    for candidate in ADJ_CLOSE_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _find_ticker_column(df: pd.DataFrame) -> str | None:
    for candidate in TICKER_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def load_ticker_data(csv_path: str, ticker: str | None = None) -> pd.DataFrame:
    """
    Loads csv_path, optionally filters to a single ticker, validates
    required columns exist, and returns a clean DataFrame sorted by Date.
    """
    df = pd.read_csv(csv_path)

    ticker_col = _find_ticker_column(df)
    if ticker and ticker_col:
        df = df[df[ticker_col].str.upper() == ticker.upper()].copy()
        if df.empty:
            available = sorted(df[ticker_col].unique()) if ticker_col in df else []
            raise ValueError(
                f"No rows found for ticker '{ticker}'. "
                f"Available tickers in this file: {available[:15]}"
            )

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Use Adj Close instead of raw Close as the actual working price.
    # Raw Close/Open/High/Low are NOT split-adjusted, so a stock like
    # AAPL (4-for-1 split Aug 2020, 7-for-1 split Jun 2014) shows a
    # fake ~75%/85% overnight "crash" in the raw series that has
    # nothing to do with real price movement. Adj Close corrects for
    # this and gives a continuous, learnable series.
    adj_col = _find_adj_close_column(df)
    if adj_col:
        df["Close"] = df[adj_col]
    else:
        print(
            "[warning] No 'Adj Close' column found -- using raw 'Close'. "
            "If this ticker had a stock split in the date range, expect "
            "a corrupted price series and bad model performance."
        )

    keep_cols = REQUIRED_COLUMNS + ([ticker_col] if ticker_col else [])
    return df[keep_cols]


def make_synthetic_data(n_days: int = 800, seed: int = 42) -> pd.DataFrame:
    """
    Generates a plausible-looking synthetic OHLCV series so the rest
    of the pipeline (features, model, export) can be built and tested
    BEFORE the real Kaggle CSV is downloaded and authenticated.

    Not for the actual report -- swap in the real dataset before
    training the model you'll submit.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_days, freq="B")

    returns = rng.normal(loc=0.0004, scale=0.015, size=n_days)
    close = 150 * np.exp(np.cumsum(returns))

    daily_range_pct = np.abs(rng.normal(loc=0.012, scale=0.006, size=n_days))
    high = close * (1 + daily_range_pct / 2)
    low = close * (1 - daily_range_pct / 2)
    open_ = low + (high - low) * rng.random(n_days)
    volume = rng.integers(2_000_000, 20_000_000, size=n_days)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def load_or_synthesize(csv_path: str, ticker: str | None = None) -> pd.DataFrame:
    """
    Convenience wrapper: uses the real CSV if it exists at csv_path,
    otherwise falls back to synthetic data with a warning printed.
    """
    if os.path.exists(csv_path):
        return load_ticker_data(csv_path, ticker)

    print(
        f"[warning] '{csv_path}' not found -- using SYNTHETIC data instead. "
        f"Run the Kaggle download step, then re-run with the real CSV path."
    )
    return make_synthetic_data()