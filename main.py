"""
VR Predictive Analytics Lab -- end-to-end pipeline.

Usage:
    python main.py --csv data/raw/historical_stock_prices.csv --ticker AAPL
    python main.py                      # runs on synthetic data (no CSV needed yet)

Produces: exports/<ticker>_scene_data.json  -- ready for the Babylon.js scene.
"""

import argparse

from src.data_loader import load_or_synthesize
from src.features import engineer_features
from src.model import train_baseline_model, predict_latest
from src.export_scene_data import build_scene_payload, save_scene_payload


def run(csv_path: str, ticker: str | None, output_dir: str) -> None:
    print(f"Loading data (ticker={ticker or 'N/A'})...")
    raw_df = load_or_synthesize(csv_path, ticker)
    print(f"  {len(raw_df)} rows loaded, {raw_df['Date'].min().date()} -> {raw_df['Date'].max().date()}")

    print("Engineering features (volume_norm, volatility, momentum)...")
    feat_df = engineer_features(raw_df)
    print(f"  {len(feat_df)} rows remain after feature engineering")

    print("Training baseline model...")
    trained = train_baseline_model(feat_df)
    print(f"  MAE: {trained.mae:.2f}  |  R^2 (price, reconstructed): {trained.r2:.3f}  |  R^2 (return, honest): {trained.return_r2:.3f}")

    print("Predicting from latest row...")
    result = predict_latest(trained, feat_df)
    print(f"  Prediction: {result['prediction']}  |  Confidence proxy: {result['confidence']}")

    dataset_id = (ticker or "synthetic").lower()
    label = f"{ticker or 'Synthetic Stock'} — Historical Price Data"

    payload = build_scene_payload(dataset_id, label, feat_df, result)

    output_path = f"{output_dir}/{dataset_id}_scene_data.json"
    save_scene_payload(payload, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VR Analytics Lab data pipeline")
    parser.add_argument(
        "--csv",
        default="data/raw/historical_stock_prices.csv",
        help="Path to the Kaggle CSV (falls back to synthetic data if not found)",
    )
    parser.add_argument("--ticker", default=None, help="Ticker symbol to filter to, e.g. AAPL")
    parser.add_argument("--output-dir", default="exports", help="Where to save the scene JSON")
    args = parser.parse_args()

    run(args.csv, args.ticker, args.output_dir)