# refactored_fyers_swing/strategy/signal_generator.py

import pandas as pd
from config import EMA_PERIODS
from utils.logger import get_logger

logger = get_logger(__name__)

def generate_signals(pred_df: pd.DataFrame, features_df: pd.DataFrame, prob_threshold: float = 0.7) -> pd.DataFrame:
    """
    Filter predicted stocks based on probability and confirm with indicators.

    Parameters:
    - pred_df: DataFrame with ['symbol', 'date', 'probability', 'prediction']
    - features_df: DataFrame with technical indicators for those symbols
    - prob_threshold: Minimum probability to consider a signal

    Returns:
    - DataFrame of filtered bullish candidates
    """
    try:
        merged = pd.merge(pred_df, features_df, on=["symbol", "date"], how="left")

        # Signal rules
        def is_bullish(row):
            # Conditions: High probability, EMA 20 > EMA 50, RSI > 50, Close > EMA 20
            if row["probability"] < prob_threshold:
                return False
            if row.get("ema_20") and row.get("ema_50") and row["ema_20"] <= row["ema_50"]:
                return False
            if row.get("rsi") and row["rsi"] < 50:
                return False
            if row.get("close") and row.get("ema_20") and row["close"] < row["ema_20"]:
                return False
            return True

        merged["bullish"] = merged.apply(is_bullish, axis=1)
        result = merged[merged["bullish"] == True][["symbol", "date", "probability", "rsi", "ema_20", "ema_50", "close"]]

        logger.info(f"{len(result)} bullish signals generated.")
        return result.sort_values(by="probability", ascending=False)

    except Exception as e:
        logger.exception(f"Signal generation failed: {e}")
        return pd.DataFrame()
