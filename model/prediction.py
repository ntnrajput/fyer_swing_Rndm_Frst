# refactored_fyers_swing/model/predict.py

import pandas as pd
import joblib
from config import MODEL_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

def load_model():
    """Load the trained model from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        return None

def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Predict the probability of target hit for each row in df.
    
    Parameters:
    - model: Trained ML model
    - df: DataFrame with features (including symbol, date)

    Returns:
    - DataFrame with predictions
    """
    try:
        features = df.drop(columns=["symbol", "date"], errors="ignore").copy()
        features = features.dropna()

        probs = model.predict_proba(features)[:, 1]
        preds = model.predict(features)

        result = df.loc[features.index, ["symbol", "date"]].copy()
        result["probability"] = probs
        result["prediction"] = preds

        logger.info("Prediction completed.")
        return result

    except Exception as e:
        logger.exception(f"Prediction failed: {e}")
        return pd.DataFrame()
