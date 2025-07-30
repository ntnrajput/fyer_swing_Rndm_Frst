# refactored_fyers_swing/config.py

import os
from pathlib import Path

# === FYERS API ===
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "VE3CCLJZWA-100" )
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "QEGA69PVUL")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://www.google.com" )
FYERS_APP_ID_HASH = os.getenv("FYERS_APP_ID_HASH", "b209632623b60de416ea3bcbd2b780ef11ebdbb652b3f06f63ffdd34366faa18")

TOKEN_PATH = Path("outputs/fyers_access_token.txt")


#=====Good Stock Filter====
AVG_VOL = 3
AVG_PRICE = 100

# === DATA SETTINGS ===
SYMBOLS_FILE = Path("symbols.csv")
HISTORICAL_DATA_FILE = Path("outputs/all_symbols_history.parquet")
HISTORICAL_DATA_FILE_csv = Path("outputs/all_symbols_history.csv")
LATEST_DATA_FILE = Path("outputs/latest_full_data.parquet")
DAILY_DATA_FILE = Path("outputs/today_data.csv")


# === INDICATOR SETTINGS ===
EMA_PERIODS = [20, 50, 200]
RSI_PERIOD = 14
VOLUME_LOOKBACK = 20

# === MODEL ===
MODEL_FILE = Path("enhanced_model_pipeline.pkl")

# === LOGGING ===
LOG_FILE = Path("outputs/logs/system.log")

#====Feature Columns====
FEATURE_COLUMNS = [
    'rsi', 'atr_pct', 'bb_position',  # Technical indicators
    'support_distance_pct', 'resistance_distance_pct',  # Added comma here
    'is_bullish', 'is_bearish', 'body_to_range', 'upper_shadow_to_range', 'lower_shadow_to_range',
    'is_doji', 'is_hammer', 'is_shooting_star', 'vol_by_avg_vol', 'ema20_ema50', 'ema50_ema200',
    'price_change_pct', 'high_low_pct', 'close_position_in_range', 'gap_pct', 'fib_pivot_distance_pct',
    'fib_r1_distance_pct', 'fib_r2_distance_pct', 'fib_s1_distance_pct', 'fib_s2_distance_pct', 'ema20_price', 'ema50_price', 'ema200_price'
]


CONFIDENCE_THRESHOLD = 0.65

