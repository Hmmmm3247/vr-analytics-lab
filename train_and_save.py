"""
One-shot local pipeline: trains the model, then writes the two
artifacts the deployed app actually needs:

    api/model.pkl        -- pickled {model, feature_columns, baseline_close}
                             (loaded, not retrained, by api/index.py)
    scene/scene_data.json -- static payload the Babylon.js scene fetches

Run once locally before deploying:
    python train_and_save.py --csv data/raw/StockPriceDataset.csv --ticker AAPL

Not deployed itself -- src/, data/, and this script stay local-only.
"""

import argparse
import pickle

from src.data_loader import load_or_synthesize
from src.features import engineer_features
from src.model import train_baseline_model, predict_latest
from src.export_scene_data import build_scene_payload, save_scene_payload


def run(csv_path: str, ticker: str | None) -> None:
    print(f"Loading data (ticker={ticker or 'N/A'})...")
    raw_df = load_or_synthesize(csv_path, ticker)

    print("Engineering features...")
    feat_df = engineer_features(raw_df)

    print("Training baseline model...")
    trained = train_baseline_model(feat_df)
    print(f"  MAE: {trained.mae:.2f}  |  return R^2: {trained.return_r2:.3f}")

    result = predict_latest(trained, feat_df)

    dataset_id = (ticker or "synthetic").lower()
    label = f"{ticker or 'Synthetic Stock'} — Historical Price Data"
    payload = build_scene_payload(dataset_id, label, feat_df, result)
    save_scene_payload(payload, "scene/scene_data.json")

    bundle = {
        "model": trained.model,
        "feature_columns": trained.feature_columns,
        "baseline_close": result["baseline_close"],
        "mae": trained.mae,
        "return_r2": trained.return_r2,
    }
    with open("api/model.pkl", "wb") as f:
        pickle.dump(bundle, f)
    print("Saved model -> api/model.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model and export deployment artifacts")
    parser.add_argument("--csv", default="data/raw/StockPriceDataset.csv")
    parser.add_argument("--ticker", default=None)
    args = parser.parse_args()

    run(args.csv, args.ticker)
