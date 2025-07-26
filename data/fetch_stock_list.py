# refactored_fyers_swing/data/fetch_stock_list.py

import pandas as pd
from config import SYMBOLS_FILE
from utils.logger import get_logger

logger = get_logger(__name__)

def load_symbols() -> list:
    """Load symbol list from CSV file."""
    try:
        df = pd.read_csv(SYMBOLS_FILE)
        symbols = df['symbol'].dropna().tolist()
        logger.info(f"Loaded {len(symbols)} symbols from {SYMBOLS_FILE}")
        return symbols
    except Exception as e:
        logger.exception(f"Error loading symbols: {e}")
        return []
