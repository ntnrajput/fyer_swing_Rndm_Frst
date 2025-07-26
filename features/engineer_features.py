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
        
        # ➕ Advanced Technical Indicators (optimized)
        # symbol_df = add_advanced_indicators_optimized(symbol_df, close, high, low, volume)
       
        # ➕ Candlestick Pattern Features
        symbol_df = add_candle_features(symbol_df)
        
        # ➕ Bollinger Band position
        symbol_df["bb_position"] = calculate_bb_position(symbol_df["close"], 20)
        
        
        # ➕ Support & Resistance
        support_levels, resistance_levels = get_support_resistance(symbol_df)
        
        symbol_df = add_nearest_sr(symbol_df, support_levels, resistance_levels)


        # ➕ All other features in one optimized pass
        # symbol_df = add_all_features_optimized(symbol_df, close, high, low, open_price, volume)
        
        # ➕ Swing Labels

        symbol_df = generate_swing_labels(symbol_df)

        print(symbol_df)

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

    return df


@jit(nopython=True)
def calculate_parabolic_sar_numba(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
    """Optimized Parabolic SAR calculation using Numba."""
    length = len(close)
    sar = np.zeros(length)
    trend = np.zeros(length)
    af = np.zeros(length)
    ep = np.zeros(length)
    
    # Initialize
    sar[0] = low[0]
    trend[0] = 1
    af[0] = af_start
    ep[0] = high[0]
    
    for i in range(1, length):
        if trend[i-1] == 1:  # Uptrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if high[i] > ep[i-1]:
                ep[i] = high[i]
                af[i] = min(af[i-1] + af_increment, af_max)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            
            if sar[i] > low[i]:
                trend[i] = -1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = low[i]
            else:
                trend[i] = 1
                
        else:  # Downtrend
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if low[i] < ep[i-1]:
                ep[i] = low[i]
                af[i] = min(af[i-1] + af_increment, af_max)
            else:
                ep[i] = ep[i-1]
                af[i] = af[i-1]
            
            if sar[i] < high[i]:
                trend[i] = 1
                sar[i] = ep[i-1]
                af[i] = af_start
                ep[i] = high[i]
            else:
                trend[i] = -1
    
    return sar


def add_advanced_indicators_optimized(df: pd.DataFrame, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Add advanced indicators with optimized calculations."""
    
    # Pre-calculate common values
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    # MACD (vectorized)
    ema12 = close_series.ewm(span=12, adjust=False).mean()
    ema26 = close_series.ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    # Stochastic (vectorized)
    low_min = low_series.rolling(window=14, min_periods=1).min()
    high_max = high_series.rolling(window=14, min_periods=1).max()
    df['stoch_k'] = 100 * (close_series - low_min) / (high_max - low_min)
    df['stoch_d'] = df['stoch_k'].rolling(window=3, min_periods=1).mean()
    
    # Williams %R (vectorized)
    df['williams_r'] = -100 * (high_max - close_series) / (high_max - low_min)
    
    # CCI (vectorized)
    typical_price = (high_series + low_series + close_series) / 3
    sma_tp = typical_price.rolling(window=20, min_periods=1).mean()
    mad = typical_price.rolling(window=20, min_periods=1).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
    
    # Money Flow Index (if volume exists)
    if volume is not None:
        volume_series = pd.Series(volume)
        money_flow = typical_price * volume_series
        positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
        
        pos_mf = pd.Series(positive_flow).rolling(window=14, min_periods=1).sum()
        neg_mf = pd.Series(negative_flow).rolling(window=14, min_periods=1).sum()
        df['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-10)))
    
    # Parabolic SAR (optimized with Numba)
    df['sar'] = calculate_parabolic_sar_numba(high, low, close)
    
    # Ichimoku (vectorized)
    df = add_ichimoku_vectorized(df, high_series, low_series, close_series)
    
    return df


def add_ichimoku_vectorized(df: pd.DataFrame, high_series: pd.Series, low_series: pd.Series, close_series: pd.Series) -> pd.DataFrame:
    """Add Ichimoku components with vectorized operations."""
    
    # Tenkan-sen (9-period)
    tenkan_high = high_series.rolling(window=9, min_periods=1).max()
    tenkan_low = low_series.rolling(window=9, min_periods=1).min()
    df['tenkan_sen'] = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (26-period)
    kijun_high = high_series.rolling(window=26, min_periods=1).max()
    kijun_low = low_series.rolling(window=26, min_periods=1).min()
    df['kijun_sen'] = (kijun_high + kijun_low) / 2
    
    # Senkou Span A
    df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
    
    # Senkou Span B
    senkou_high = high_series.rolling(window=52, min_periods=1).max()
    senkou_low = low_series.rolling(window=52, min_periods=1).min()
    df['senkou_span_b'] = ((senkou_high + senkou_low) / 2).shift(26)
    
    # Chikou Span
    df['chikou_span'] = close_series.shift(-26)
    
    # Cloud thickness
    df['cloud_thickness'] = abs(df['senkou_span_a'] - df['senkou_span_b'])
    
    return df


def add_all_features_optimized(df: pd.DataFrame, close: np.ndarray, high: np.ndarray, low: np.ndarray, open_price: np.ndarray, volume: np.ndarray) -> pd.DataFrame:
    """Add all remaining features in one optimized pass."""
    
    close_series = pd.Series(close)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    open_series = pd.Series(open_price)
    
    # Basic calculations
    df['daily_return'] = close_series.pct_change()
    df['range_pct'] = (high_series - low_series) / close_series
    
    # Pre-calculate body and shadows for candle patterns
    df['body'] = abs(close_series - open_series)
    df['upper_shadow'] = high_series - np.maximum(close_series, open_series)
    df['lower_shadow'] = np.minimum(close_series, open_series) - low_series
    
    # Vectorized rolling calculations
    periods = [5, 10, 20]
    for period in periods:
        # Volatility and returns
        df[f'volatility_{period}'] = df['daily_return'].rolling(period, min_periods=1).std()
        df[f'return_mean_{period}'] = df['daily_return'].rolling(period, min_periods=1).mean()
        df[f'return_skew_{period}'] = df['daily_return'].rolling(period, min_periods=1).skew()
        df[f'return_kurt_{period}'] = df['daily_return'].rolling(period, min_periods=1).kurt()
        
        # ROC and momentum
        df[f'roc_{period}'] = (close_series / close_series.shift(period) - 1) * 100
        df[f'momentum_{period}'] = close_series - close_series.shift(period)
        
        # ATR
        tr1 = high_series - low_series
        tr2 = abs(high_series - close_series.shift(1))
        tr3 = abs(low_series - close_series.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        df[f'atr_{period}'] = pd.Series(true_range).rolling(period, min_periods=1).mean()
        
        # Trend strength
        df[f'trend_strength_{period}'] = close_series.rolling(period, min_periods=period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0, 
            raw=True
        )
    
    # EMA relationships (vectorized)
    df['price_above_ema20'] = (close_series > df['ema20']).astype(int)
    df['price_above_ema50'] = (close_series > df['ema50']).astype(int)
    df['ema20_above_ema50'] = (df['ema20'] > df['ema50']).astype(int)
    
    # Distance features
    df['ema20_distance'] = (close_series - df['ema20']) / close_series
    df['ema50_distance'] = (close_series - df['ema50']) / close_series
    df['ema_spread'] = (df['ema20'] - df['ema50']) / close_series
    
    # Normalize support/resistance distances
    df['norm_dist_to_support'] = df['dist_to_support'] / close_series
    df['norm_dist_to_resistance'] = df['dist_to_resistance'] / close_series
    
    # Market structure (vectorized)
    df['higher_high'] = ((high_series > high_series.shift(1)) & 
                        (high_series.shift(1) > high_series.shift(2))).astype(int)
    df['lower_low'] = ((low_series < low_series.shift(1)) & 
                      (low_series.shift(1) < low_series.shift(2))).astype(int)
    
    # Price action patterns (vectorized)
    df['pin_bar_bull'] = ((df['lower_shadow'] > 2 * df['body']) & 
                         (df['upper_shadow'] < 0.5 * df['body']) & 
                         (close_series > open_series)).astype(int)
    
    df['pin_bar_bear'] = ((df['upper_shadow'] > 2 * df['body']) & 
                         (df['lower_shadow'] < 0.5 * df['body']) & 
                         (close_series < open_series)).astype(int)
    
    # Engulfing patterns
    df['bullish_engulfing'] = ((close_series > open_series) & 
                              (close_series.shift(1) < open_series.shift(1)) & 
                              (close_series > open_series.shift(1)) & 
                              (open_series < close_series.shift(1))).astype(int)
    
    df['bearish_engulfing'] = ((close_series < open_series) & 
                              (close_series.shift(1) > open_series.shift(1)) & 
                              (close_series < open_series.shift(1)) & 
                              (open_series > close_series.shift(1))).astype(int)
    
    # Inside/Outside bars
    df['inside_bar'] = ((high_series < high_series.shift(1)) & 
                       (low_series > low_series.shift(1))).astype(int)
    df['outside_bar'] = ((high_series > high_series.shift(1)) & 
                        (low_series < low_series.shift(1))).astype(int)
    
    # Gap analysis
    df['gap_up'] = (low_series > high_series.shift(1)).astype(int)
    df['gap_down'] = (high_series < low_series.shift(1)).astype(int)
    
    # Volume features (if available)
    if volume is not None:
        volume_series = pd.Series(volume)
        
        # Volume moving averages
        df['volume_sma_10'] = volume_series.rolling(10, min_periods=1).mean()
        df['volume_sma_20'] = volume_series.rolling(20, min_periods=1).mean()
        df['volume_sma_50'] = volume_series.rolling(50, min_periods=1).mean()
        
        # Volume ratios
        df['volume_ratio_10'] = volume_series / df['volume_sma_10']
        df['volume_ratio_20'] = volume_series / df['volume_sma_20']
        df['volume_ratio'] = volume_series / (df['volume_sma_20'] + 1)
        
        # OBV (vectorized)
        price_change = np.where(close_series > close_series.shift(1), 1, 
                               np.where(close_series < close_series.shift(1), -1, 0))
        df['obv'] = (volume_series * price_change).cumsum()
        
        # VPT
        df['vpt'] = (volume_series * close_series.pct_change()).cumsum()
        
        # A/D Line
        clv = ((close_series - low_series) - (high_series - close_series)) / (high_series - low_series + 1e-10)
        df['ad_line'] = (clv * volume_series).cumsum()
        
        # VWAP
        df['vwap'] = (close_series * volume_series).cumsum() / volume_series.cumsum()
        
        # Volume breakouts
        df['volume_breakout'] = (volume_series > df['volume_sma_20'] * 1.5).astype(int)
        
        df['price_volume'] = close_series * volume_series
    
    # Volatility features
    df['volatility_ratio'] = df['volatility_5'] / (df['volatility_20'] + 1e-10)
    df['volatility_breakout'] = (df['volatility_5'] > df['volatility_20'] * 1.5).astype(int)
    
    # Keltner Channels
    df['keltner_upper'] = df['ema20'] + (2 * df['atr_20'])
    df['keltner_lower'] = df['ema20'] - (2 * df['atr_20'])
    df['keltner_position'] = (close_series - df['keltner_lower']) / (df['keltner_upper'] - df['keltner_lower'] + 1e-10)
    
    # Regime indicators
    df['bull_regime'] = ((close_series > df['ema20']) & 
                        (df['ema20'] > df['ema50']) & 
                        (df['rsi'] > 50)).astype(int)
    
    df['bear_regime'] = ((close_series < df['ema20']) & 
                        (df['ema20'] < df['ema50']) & 
                        (df['rsi'] < 50)).astype(int)
    
    # Confluence scores
    df['bullish_confluence'] = (
        df['price_above_ema20'] + 
        df['ema20_above_ema50'] + 
        (df['rsi'] > 50).astype(int) + 
        (df['macd'] > df['macd_signal']).astype(int) + 
        df['bullish_engulfing'] + 
        df['pin_bar_bull']
    )
    
    df['bearish_confluence'] = (
        (1 - df['price_above_ema20']) + 
        (1 - df['ema20_above_ema50']) + 
        (df['rsi'] < 50).astype(int) + 
        (df['macd'] < df['macd_signal']).astype(int) + 
        df['bearish_engulfing'] + 
        df['pin_bar_bear']
    )
    
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
