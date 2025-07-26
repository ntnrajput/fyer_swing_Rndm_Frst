# predict.py

import pandas as pd
import joblib
import os
from datetime import datetime
from config import MODEL_FILE, FEATURE_COLUMNS, HISTORICAL_DATA_FILE, LATEST_DATA_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

def load_model():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)

def predict_today():
    logger.info(" Running prediction for today...")

    # Load latest feature data
    df = pd.read_parquet(LATEST_DATA_FILE)
    latest_date = df["date"].max()
    df_latest = df[df["date"] == latest_date].copy()

    print(df_latest)

    # Load model
    model = load_model()

    # Predict
    X = df_latest[FEATURE_COLUMNS]
    df_latest["predicted_swing"] = model.predict(X)
    df_latest["prediction_proba"] = model.predict_proba(X)[:, 1]  # probability of class 1 (target hit)

    # Filter probable swing trades
    swing_stocks = df_latest[df_latest["predicted_swing"] == 1].sort_values(by="prediction_proba", ascending=False)

    # Output
    logger.info(f"Found {len(swing_stocks)} swing trade opportunities for {latest_date}")
    output_path = f"output/predictions_{latest_date}.csv"
    os.makedirs("output", exist_ok=True)
    swing_stocks.to_csv(output_path, index=False)

    print(swing_stocks[["symbol", "close", "nearest_support", "nearest_resistance", "prediction_proba"]])
    logger.info(f" Predictions saved to {output_path}")

if __name__ == "__main__":
    predict_today()
