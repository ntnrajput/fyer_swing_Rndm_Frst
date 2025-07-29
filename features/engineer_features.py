# features/engineer_features.py

import numpy as np
import pandas as pd
from numba import jit
import warnings
warnings.filterwarnings('ignore')

from utils.helper import calculate_ema, calculate_rsi, calculate_atr
from features.swing_utils import (
    calculate_bb_position,
    add_candle_features,
    get_support_resistance,
    add_nearest_sr,
    generate_swing_labels
)
from config import EMA_PERIODS, RSI_PERIOD, AVG_VOL,AVG_PRICE
from utils.logger import get_logger

logger = get_logger(__name__)

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators with optimized performance.
    """
    # Pre-allocate list for better memory management
    all_dfs = []
    
    # Group by symbol for vectorized operations
    grouped = df.groupby('symbol')
    
    for symbol, symbol_df in grouped:
        if symbol_df['volume'].iloc[-50:].mean() < AVG_VOL or symbol_df['close'].iloc[-50:].mean() < AVG_PRICE:
            continue

        symbol_df = symbol_df.copy()
        symbol_df.sort_values("date", inplace=True)
        
        # Reset index for consistent indexing
        symbol_df.reset_index(drop=True, inplace=True)
        
        # Extract OHLCV arrays once for reuse
        high = symbol_df['high'].values
        low = symbol_df['low'].values
        close = symbol_df['close'].values
        open_price = symbol_df['open'].values
        volume = symbol_df['volume'].values if 'volume' in symbol_df.columns else None
        
        # ➕ Basic Technical Indicators (vectorized)
        symbol_df = add_basic_indicators_vectorized(symbol_df, close, high, low, open_price,volume)
       
        # ➕ Candlestick Pattern Features
        symbol_df = add_candle_features(symbol_df)
        
        # ➕ Bollinger Band position
        symbol_df["bb_position"] = calculate_bb_position(symbol_df["close"], 20)
        
        # ➕ Support & Resistance
        support_levels, resistance_levels = get_support_resistance(symbol_df)
        
        symbol_df = add_nearest_sr(symbol_df, support_levels, resistance_levels)

        # ➕ Swing Labels
        symbol_df = generate_swing_labels(symbol_df)

        all_dfs.append(symbol_df)
        logger.info(f" Features added for {symbol}")

    return pd.concat(all_dfs, ignore_index=True)

def calculate_rolling_fib_pivots(df, window=10):
    """Calculate rolling Fibonacci pivot levels using a rolling window."""
    
    # Calculate rolling high, low, close over window periods
    rolling_high = df['high'].rolling(window=window).max()
    rolling_low = df['low'].rolling(window=window).min()
    rolling_close = df['close'].rolling(window=window).mean()
    
    # Calculate pivot point
    pivot = (rolling_high + rolling_low + rolling_close) / 3
    
    # Fibonacci levels
    fib_range = rolling_high - rolling_low
    fib_r1 = pivot + 0.382 * fib_range
    fib_r2 = pivot + 0.618 * fib_range
    fib_s1 = pivot - 0.382 * fib_range
    fib_s2 = pivot - 0.618 * fib_range
    
    # Calculate distances as percentages
    df['fib_pivot_distance_pct'] = (df['close'] - pivot) / pivot * 100
    df['fib_r1_distance_pct'] = (df['close'] - fib_r1) / fib_r1 * 100
    df['fib_r2_distance_pct'] = (df['close'] - fib_r2) / fib_r2 * 100
    df['fib_s1_distance_pct'] = (df['close'] - fib_s1) / fib_s1 * 100
    df['fib_s2_distance_pct'] = (df['close'] - fib_s2) / fib_s2 * 100
    
    return df

def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    fast_length = 6
    slow_length = 9
    signal_length = 6

    def compute_macd(group):
        close = group['close']
        ema_fast = close.ewm(span=fast_length, adjust=False).mean()
        ema_slow = close.ewm(span=slow_length, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal_length, adjust=False).mean()
        macd_hist = macd - macd_signal
        macd_crossover = macd > macd_signal
        macd_cross_signal = macd_crossover.ne(macd_crossover.shift(1))
        macd_signal_type = macd_crossover & macd_cross_signal
        macd_signal_type = macd_signal_type.map({True: 'buy'}).fillna(
            macd_cross_signal.map({True: 'sell'}).fillna(''))

        macd_ls_signal = ((macd > 0) & (macd_signal > 0) & (macd_crossover)).astype(int)
        group['macd_ls_signal'] = macd_ls_signal
        
        return group
    return df.groupby('symbol', group_keys=False).apply(compute_macd)

def add_basic_indicators_vectorized(df: pd.DataFrame, close: np.ndarray, high: np.ndarray, low: np.ndarray, open_price: np.ndarray, volume:np.ndarray) -> pd.DataFrame:
    """Add basic indicators using vectorized operations."""
    
    # EMA calculations (vectorized)
    for period in EMA_PERIODS:
        df[f"ema{period}"] = calculate_ema(df["close"], period)
    

    df['ema20_ema50'] = df['ema20']/df['ema50']
    df['ema50_ema200'] = df['ema50']/df['ema200']
    df['ema20_price'] =df['ema20']/df['close']
    df['ema50_price'] =df['ema50']/df['close']
    df['ema200_price'] =df['ema200']/df['close']
    # RSI and ATR
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["atr"] = calculate_atr(df)
    df['atr_pct'] =df['atr']/df['close']

    volume = np.array(volume, dtype=float)
    
    # Calculate rolling mean while skipping nan
    rolling_mean = pd.Series(volume).rolling(window= 50 , min_periods=1).mean()
    
    # Compute volume / rolling mean
    ratio = volume / rolling_mean.to_numpy()

    df['vol_by_avg_vol'] = ratio

    df['price_change_pct'] = (close - np.roll(close, 1)) / np.roll(close, 1) * 100
    df['high_low_pct'] = (high - low) / close * 100
    range_hl = high - low
    df['close_position_in_range'] = np.where(range_hl != 0, (close - low) / range_hl, 0.5)
    df['gap_pct'] = (open_price - np.roll(close, 1)) / np.roll(close, 1) * 100

    df = calculate_rolling_fib_pivots(df,window=5)
    df = add_macd(df)
    return df


    
    # Price acceleration
    df['price_acceleration'] = close_series.diff().diff()
    
    # Trend direction (simplified and vectorized)
    df['trend_direction'] = np.where(
        (df['ema20'] > df['ema50']) & (close_series > df['ema20']), 1,
        np.where((df['ema20'] < df['ema50']) & (close_series < df['ema20']), -1, 0)
    )
    
    # Structure breaks
    df['structure_break'] = (df['trend_direction'] != df['trend_direction'].shift(1)).astype(int)
    
    # Handle NaNs efficiently
    df = df.fillna(method='ffill').fillna(0)
    
    return df
