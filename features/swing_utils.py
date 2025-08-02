# features/swing_utils.py

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, period=14):
    """Calculate RSI with improved smoothing."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Use Wilder's smoothing method
    gain_smooth = gain.ewm(alpha=1/period, adjust=False).mean()
    loss_smooth = loss.ewm(alpha=1/period, adjust=False).mean()
    
    rs = gain_smooth / loss_smooth
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bb_position(prices, period=20, std_dev=2):
    """Calculate Bollinger Band position with configurable parameters."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    # Calculate position within bands
    bb_position = (prices - lower_band) / (upper_band - lower_band)
    
    # Add squeeze detection
    bb_width = (upper_band - lower_band) / sma
    bb_squeeze = bb_width < bb_width.rolling(20).quantile(0.1)
    
    return bb_position.clip(0, 1)


def add_candle_features(df):
    """Enhanced candlestick pattern recognition."""
    df = df.copy()
    
    # Vectorized basic candle components
    df['body'] = np.abs(df['close'] - df['open'])
    df['range'] = df['high'] - df['low']
    
    # Vectorized max/min operations
    ohlc_max = np.maximum(df['open'], df['close'])
    ohlc_min = np.minimum(df['open'], df['close'])
    
    df['upper_shadow'] = df['high'] - ohlc_max 
    df['lower_shadow'] = ohlc_min - df['low']
    
    # Candle direction
    df['is_bullish'] = (df['close'] > df['open']).astype(np.int8)
    df['is_bearish'] = (df['open'] > df['close']).astype(np.int8)
    
    # Ratios with epsilon for numerical stability
    range_safe = df['range'] + 1e-8
    df['body_to_range'] = df['body'] / range_safe
    df['upper_shadow_to_range'] = df['upper_shadow'] / range_safe
    df['lower_shadow_to_range'] = df['lower_shadow'] / range_safe
    
    # Vectorized basic patterns
    df['is_doji'] = (df['body_to_range'] < 0.1).astype(np.int8)
    df['is_hammer'] = ((df['body_to_range'] < 0.3) & 
                       (df['lower_shadow_to_range'] > 0.6) & 
                       (df['upper_shadow_to_range'] < 0.1)).astype(np.int8)
    df['is_shooting_star'] = ((df['body_to_range'] < 0.3) & 
                             (df['upper_shadow_to_range'] > 0.6) & 
                             (df['lower_shadow_to_range'] < 0.1)).astype(np.int8)
    
    df['range_pct'] = df['range'] / df['close']
    df['close_below_1/2'] = df['close'] < (df['low']+(df['range'] * 0.4))
    df['strong_rejection'] = ((df['range_pct'] > 0.05)  | (df['close_below_1/2'])).astype(int)
    
    # Advanced patterns
    # df = add_advanced_candle_patterns(df)
    
    # Pattern strength
    # df['pattern_strength'] = calculate_pattern_strength(df)
    
    return df


def add_advanced_candle_patterns(df):
    """Add advanced candlestick patterns."""
    df = df.copy()
    
    # Vectorized spinning tops
    df['spinning_top'] = ((df['body_to_range'] < 0.3) & 
                         (df['upper_shadow_to_range'] > 0.3) & 
                         (df['lower_shadow_to_range'] > 0.3)).astype(np.int8)
    
    # Vectorized Marubozu
    df['marubozu_bull'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] > df['open'])).astype(np.int8)
    df['marubozu_bear'] = ((df['body_to_range'] > 0.95) & 
                          (df['close'] < df['open'])).astype(np.int8)
    
    # Pre-compute shifts for multi-candle patterns
    close_s1 = df['close'].shift(1)
    close_s2 = df['close'].shift(2)
    is_bullish_s1 = df['is_bullish'].shift(1)
    is_bullish_s2 = df['is_bullish'].shift(2)
    is_bearish_s1 = df['is_bearish'].shift(1)
    is_bearish_s2 = df['is_bearish'].shift(2)
    body_s1 = df['body'].shift(1)
    body_s2 = df['body'].shift(2)
    open_s2 = df['open'].shift(2)
    
    # Three white soldiers / Three black crows
    df['three_white_soldiers'] = (
        (df['is_bullish'] == 1) & 
        (is_bullish_s1 == 1) & 
        (is_bullish_s2 == 1) & 
        (df['close'] > close_s1) & 
        (close_s1 > close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    df['three_black_crows'] = (
        (df['is_bearish'] == 1) & 
        (is_bearish_s1 == 1) & 
        (is_bearish_s2 == 1) & 
        (df['close'] < close_s1) & 
        (close_s1 < close_s2) & 
        (df['body'] > body_s1 * 0.8) & 
        (body_s1 > body_s2 * 0.8)
    ).astype(np.int8)
    
    # Morning star / Evening star
    midpoint_s2 = (open_s2 + close_s2) / 2
    df['morning_star'] = (
        (is_bearish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bullish'] == 1) & 
        (df['close'] > midpoint_s2)
    ).astype(np.int8)
    
    df['evening_star'] = (
        (is_bullish_s2 == 1) & 
        (body_s1 < body_s2 * 0.3) & 
        (df['is_bearish'] == 1) & 
        (df['close'] < midpoint_s2)
    ).astype(np.int8)
    
    # Harami patterns
    df['bullish_harami'] = (
        (is_bearish_s1 == 1) & 
        (df['is_bullish'] == 1) & 
        (df['high'] < close_s1) & 
        (df['low'] > open_s2)
    ).astype(np.int8)
    
    df['bearish_harami'] = (
        (is_bullish_s1 == 1) & 
        (df['is_bearish'] == 1) & 
        (df['high'] < open_s2) & 
        (df['low'] > close_s1)
    ).astype(np.int8)
    
    return df


def calculate_pattern_strength(df):
    """Calculate overall pattern strength score."""
    pattern_columns = [
        'is_doji', 'is_hammer', 'is_shooting_star', 'spinning_top',
        'marubozu_bull', 'marubozu_bear', 'three_white_soldiers',
        'three_black_crows', 'morning_star', 'evening_star',
        'bullish_harami', 'bearish_harami'
    ]
    
    # Weight patterns by reliability
    weights = np.array([
        0.5, 2.0, 2.0, 0.5, 1.5, 1.5, 3.0, 3.0, 2.5, 2.5, 1.5, 1.5
    ])
    
    # Vectorized calculation
    pattern_strength = np.zeros(len(df))
    for i, col in enumerate(pattern_columns):
        if col in df.columns:
            pattern_strength += df[col].values * weights[i]
    
    return pattern_strength


def cluster_levels(levels, price_threshold=0.01):
    """
    Cluster support/resistance levels that are within a certain price threshold.
    Args:
        levels: List of (idx, price, touches) tuples.
        price_threshold: Fractional threshold (e.g., 0.01 for 1%) for clustering levels.
    Returns:
        List of clustered (idx, price, touches) tuples.
    """
    if not levels:
        return []
    # Sort by price
    levels = sorted(levels, key=lambda x: x[1])
    clustered = []
    cluster = [levels[0]]
    
    for lvl in levels[1:]:
        prev_price = cluster[-1][1]
        if abs(lvl[1] - prev_price) / prev_price < price_threshold:
            cluster.append(lvl)
        else:
            # Merge cluster
            idxs, prices, touches = zip(*cluster)
            avg_idx = int(np.mean(idxs))
            avg_price = np.mean(prices)
            total_touches = int(np.sum(touches))
            clustered.append((avg_idx, avg_price, total_touches))
            cluster = [lvl]
    
    # Merge last cluster
    if cluster:
        idxs, prices, touches = zip(*cluster)
        avg_idx = int(np.mean(idxs))
        avg_price = np.mean(prices)
        total_touches = int(np.sum(touches))
        clustered.append((avg_idx, avg_price, total_touches))
    
    return clustered

def get_support_resistance(df, window=7, min_strength=3, tolerance=0.015, cluster_threshold=0.015):
    """
    Enhanced support and resistance detection with proper touch counting:
    - Support levels: Only count low touches
    - Resistance levels: Only count high touches  
    - Flip zones: Must have both high_touches >= 1 and low_touches >= 1
    """
    df = df.copy()
    highs = df['high'].values
    lows = df['low'].values
    
    # Find traditional support levels (only count low touches)
    support_indices = argrelextrema(lows, np.less, order=window)[0]
    support_levels = []
    
    for idx in support_indices:
        if idx < len(df):
            level_price = df.iloc[idx]['low']
            low_touches = np.sum(np.abs(lows - level_price) / level_price < tolerance)
            
            if low_touches >= min_strength:
                support_levels.append((idx, level_price, low_touches))
    
    # Find traditional resistance levels (only count high touches)
    resistance_indices = argrelextrema(highs, np.greater, order=window)[0]
    resistance_levels = []
    
    for idx in resistance_indices:
        if idx < len(df):
            level_price = df.iloc[idx]['high']
            high_touches = np.sum(np.abs(highs - level_price) / level_price < tolerance)
            
            if high_touches >= min_strength:
                resistance_levels.append((idx, level_price, high_touches))
    
    # Find flip zones (levels with both support and resistance touches)
    # Only check levels from significant peaks/troughs to avoid noise
    significant_prices = []
    
    # Add resistance peak prices
    for idx in resistance_indices:
        if idx < len(df):
            significant_prices.append(df.iloc[idx]['high'])
    
    # Add support trough prices  
    for idx in support_indices:
        if idx < len(df):
            significant_prices.append(df.iloc[idx]['low'])
    
    # Remove duplicates and only check significant price levels
    significant_prices = list(set(significant_prices))
    
    for price in significant_prices:
        high_touches = np.sum(np.abs(highs - price) / price < tolerance)
        low_touches = np.sum(np.abs(lows - price) / price < tolerance)
        
        # Flip zone condition: at least 1 of each type and total >= min_strength
        if high_touches >= 1 and low_touches >= 1 and (high_touches + low_touches) >= (min_strength +1):
            # Find the most recent index for this level
            high_indices = np.where(np.abs(highs - price) / price < tolerance)[0]
            low_indices = np.where(np.abs(lows - price) / price < tolerance)[0]
            
            if len(high_indices) > 0 and len(low_indices) > 0:
                recent_idx = max(np.max(high_indices), np.max(low_indices))
                
                # Check if this flip zone is already captured by traditional levels
                is_duplicate_support = any(abs(level[1] - price) / price < tolerance for level in support_levels)
                is_duplicate_resistance = any(abs(level[1] - price) / price < tolerance for level in resistance_levels)
                
                # Add to support list with low_touches count
                if not is_duplicate_support:
                    support_levels.append((recent_idx, price, low_touches))
                
                # Add to resistance list with high_touches count  
                if not is_duplicate_resistance:
                    resistance_levels.append((recent_idx, price, high_touches))
    
    # Remove duplicates within each category (keep stronger levels)
    support_levels = remove_duplicates(support_levels, tolerance)
    resistance_levels = remove_duplicates(resistance_levels, tolerance)
    
    # Cluster levels
    support_levels = cluster_levels(support_levels, price_threshold=cluster_threshold)
    resistance_levels = cluster_levels(resistance_levels, price_threshold=cluster_threshold)
    
    return support_levels, resistance_levels

def remove_duplicates(levels, tolerance):
    """Remove duplicate levels that are too close to each other."""
    if not levels:
        return []
    
    levels = sorted(levels, key=lambda x: x[1])
    filtered = [levels[0]]
    
    for current in levels[1:]:
        prev_price = filtered[-1][1]
        current_price = current[1]
        
        # If levels are too close, keep the one with more touches
        if abs(current_price - prev_price) / prev_price < tolerance:
            if current[2] > filtered[-1][2]:
                filtered[-1] = current
        else:
            filtered.append(current)
    
    return filtered

def plot_support_resistance(df, support_levels, resistance_levels, n_bars=100):
    """
    Plot price with clustered support and resistance levels.
    Args:
        df: DataFrame with 'close' prices.
        support_levels: List of (idx, price, touches) tuples.
        resistance_levels: List of (idx, price, touches) tuples.
        n_bars: Number of bars to plot from the end.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(14, 7))
    plot_df = df.tail(n_bars).reset_index(drop=True)
    plt.plot(plot_df['close'], label='Close Price', color='black')
    for idx, price, touches in support_levels:
        if idx >= len(df) - n_bars:
            plt.axhline(price, color='green', linestyle='--', alpha=0.7, label='Support' if touches == support_levels[0][2] else None)
    for idx, price, touches in resistance_levels:
        if idx >= len(df) - n_bars:
            plt.axhline(price, color='red', linestyle='--', alpha=0.7, label='Resistance' if touches == resistance_levels[0][2] else None)
    plt.title('Support and Resistance Levels')
    plt.xlabel('Bar')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def get_dynamic_support_resistance(df, level_type='support'):
    """Get dynamic support/resistance from moving averages and trend lines."""
    dynamic_levels = []
    
    # EMAs as dynamic support/resistance
    ema_periods = [20, 50, 100, 200]
    current_close = df['close'].iloc[-1]
    
    for period in ema_periods:
        ema_col = f'ema{period}'
        if ema_col in df.columns:
            ema_values = df[ema_col].dropna()
            if len(ema_values) > 0:
                recent_ema = ema_values.iloc[-1]
                strength = period // 10
                
                if level_type == 'support' and recent_ema < current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
                elif level_type == 'resistance' and recent_ema > current_close:
                    dynamic_levels.append((len(df)-1, recent_ema, strength))
    
    # Trend line support/resistance
    trend_levels = calculate_trend_lines(df, level_type)
    dynamic_levels.extend(trend_levels)
    
    return dynamic_levels


def calculate_trend_lines(df, level_type='support', lookback=50):
    """Calculate trend lines as dynamic support/resistance."""
    if len(df) < lookback:
        return []
    
    trend_levels = []
    recent_df = df.tail(lookback)
    
    if level_type == 'support':
        # Find upward trend line through recent lows
        lows = recent_df['low'].values
        
        # Get bottom 20% of lows for trend line
        n_lows = max(2, len(lows) // 5)
        low_indices = np.argpartition(lows, n_lows)[:n_lows]
        
        if len(low_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(low_indices, lows[low_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(lows) - 1) + intercept
            
            # Only consider as support if trend is upward and correlation is good
            if slope > 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    else:  # resistance
        # Find downward trend line through recent highs
        highs = recent_df['high'].values
        
        # Get top 20% of highs for trend line
        n_highs = max(2, len(highs) // 5)
        high_indices = np.argpartition(highs, -n_highs)[-n_highs:]
        
        if len(high_indices) >= 2:
            slope, intercept, r_value, _, _ = linregress(high_indices, highs[high_indices])
            
            # Project trend line to current position
            current_trend_level = slope * (len(highs) - 1) + intercept
            
            # Only consider as resistance if trend is downward and correlation is good
            if slope < 0 and r_value > 0.5:
                strength = int(abs(r_value) * 5)
                trend_levels.append((len(df)-1, current_trend_level, strength))
    
    return trend_levels

def add_nearest_sr(df, support_levels, resistance_levels):
    """Simplified support/resistance distance calculation."""
    df = df.copy()
    
    nearest_support = []
    support_strength = []
    support_distance_pct = []

    nearest_resistance = []
    resistance_strength = []
    resistance_distance_pct = []

    for i in range(len(df)):
        close = df.iloc[i]['close']

        # Nearest Support
        valid_supports = [level for level in support_levels if level[1] <= close] if support_levels else []
        if valid_supports:
            best_support = max(valid_supports, key=lambda x: x[1])
            nearest_support.append(best_support[1])
            support_strength.append(best_support[2])
            support_distance_pct.append((close - best_support[1]) / close * 100)
        else:
            default_support = close * 0.95
            nearest_support.append(default_support)
            support_strength.append(1)
            support_distance_pct.append((close - default_support) / close * 100)

        # Nearest Resistance
        valid_resistances = [level for level in resistance_levels if level[1] >= close] if resistance_levels else []
        if valid_resistances:
            best_resistance = min(valid_resistances, key=lambda x: x[1])
            nearest_resistance.append(best_resistance[1])
            resistance_strength.append(best_resistance[2])
            resistance_distance_pct.append((best_resistance[1] - close) / close * 100)
        else:
            default_resistance = close * 1.05
            nearest_resistance.append(default_resistance)
            resistance_strength.append(1)
            resistance_distance_pct.append((default_resistance - close) / close * 100)

    df['nearest_support'] = nearest_support
    df['support_strength'] = support_strength
    df['support_distance_pct'] = support_distance_pct

    df['nearest_resistance'] = nearest_resistance
    df['resistance_strength'] = resistance_strength
    df['resistance_distance_pct'] = resistance_distance_pct

    return df


def generate_swing_labels(df, target_pct=0.15, window=21, stop_loss_pct=0.05):
    """
    Simplified swing label generation for single target.
    
    Parameters:
    - df: DataFrame with OHLC data
    - target_pct: Target return (default 7%)
    - window: Days to look ahead (default 10)
    - stop_loss_pct: Maximum acceptable loss (default 3%)
    
    Returns:
    - DataFrame with target_hit column (1 if target hit without stop loss, 0 otherwise)
    - Additional columns: max_return, min_return, risk_reward_ratio
    """
    df = df.copy()
    n_rows = len(df)
    
    # Pre-allocate arrays for results
    target_hit = np.full(n_rows, np.nan)
    max_return = np.full(n_rows, np.nan)
    min_return = np.full(n_rows, np.nan)
    risk_reward_ratio = np.full(n_rows, np.nan)
    
    # Get price arrays for vectorized operations
    close_prices = df['close'].values
    high_prices = df['high'].values
    low_prices = df['low'].values
    
    # Process each row (except last 'window' rows)
    for i in range(n_rows - window):
        entry_price = close_prices[i]
        
        # Get future prices for the next 'window' days
        future_highs = high_prices[i+1:i+1+window]
        future_lows = low_prices[i+1:i+1+window]
        
        if len(future_highs) == 0:
            continue
            
        # Calculate returns
        max_ret = (np.max(future_highs) - entry_price) / entry_price
        min_ret = (np.min(future_lows) - entry_price) / entry_price
        
        # Store risk metrics
        max_return[i] = max_ret
        min_return[i] = min_ret
        
        # Calculate risk-reward ratio
        if min_ret < 0:
            risk_reward_ratio[i] = max_ret / abs(min_ret)
        else:
            risk_reward_ratio[i] = max_ret / 0.01  # Use small denominator if no downside
        
        # Check if target hit without exceeding stop loss
        if max_ret >= target_pct and min_ret > -stop_loss_pct:
            target_hit[i] = 1
        else:
            target_hit[i] = 0
    
    # Add results to dataframe
    df['target_hit'] = target_hit
    df['max_return'] = max_return
    df['min_return'] = min_return
    df['risk_reward_ratio'] = risk_reward_ratio
    
    return df


def add_advanced_swing_labels(df, window=10):
    """Add advanced swing trading labels and features."""
    df = df.copy()
    
    # Trend following vs mean reversion classification
    n_rows = len(df)
    trend_trade = np.full(n_rows, np.nan)
    mean_reversion_trade = np.full(n_rows, np.nan)
    breakout_trade = np.full(n_rows, np.nan)
    
    # Vectorized processing where possible
    target_hit_mask = ~pd.isna(df['target_hit'])
    valid_indices = np.where(target_hit_mask)[0]
    valid_indices = valid_indices[valid_indices < n_rows - window]
    
    for i in valid_indices:
        # Get current market state
        current_trend = df['trend_direction'].iloc[i] if 'trend_direction' in df.columns else 0
        near_support = df['in_support_zone'].iloc[i] if 'in_support_zone' in df.columns else 0
        near_resistance = df['in_resistance_zone'].iloc[i] if 'in_resistance_zone' in df.columns else 0
        
        # Classify trade type
        if current_trend == 1 and near_support:
            trend_trade[i] = 1
            mean_reversion_trade[i] = 0
            breakout_trade[i] = 0
        elif current_trend == 0 and (near_support or near_resistance):
            mean_reversion_trade[i] = 1
            trend_trade[i] = 0
            breakout_trade[i] = 0
        else:
            breakout_trade[i] = 1
            trend_trade[i] = 0
            mean_reversion_trade[i] = 0
    
    df['trend_trade'] = trend_trade
    df['mean_reversion_trade'] = mean_reversion_trade
    df['breakout_trade'] = breakout_trade
    
    # Success rate by trade type
    df['trade_type_success'] = np.nan
    
    # Calculate win rate for different market conditions
    df = add_conditional_success_rates(df)
    
    return df


def add_conditional_success_rates(df):
    """Add success rates based on market conditions."""
    df = df.copy()
    
    # Define market conditions
    conditions = {
        'high_volume': 'volume_ratio > 1.5' if 'volume_ratio' in df.columns else None,
        'low_volatility': 'volatility_5 < volatility_20' if 'volatility_5' in df.columns else None,
        'bullish_regime': 'bull_regime == 1' if 'bull_regime' in df.columns else None,
        'bearish_regime': 'bear_regime == 1' if 'bear_regime' in df.columns else None,
        'high_rsi': 'rsi > 70' if 'rsi' in df.columns else None,
        'low_rsi': 'rsi < 30' if 'rsi' in df.columns else None,
    }
    
    # Calculate success rates for each condition
    for condition_name, condition_query in conditions.items():
        if condition_query is None:
            df[f'{condition_name}_success_rate'] = 0.5
            continue
            
        try:
            condition_mask = df.query(condition_query).index
            if len(condition_mask) > 0:
                success_rate = df.loc[condition_mask, 'target_hit'].mean()
                if pd.isna(success_rate):
                    success_rate = 0.5
            else:
                success_rate = 0.5
            df[f'{condition_name}_success_rate'] = success_rate
        except:
            df[f'{condition_name}_success_rate'] = 0.5
    
    return df


def calculate_swing_quality_score(df):
    """Calculate a comprehensive swing quality score."""
    df = df.copy()
    
    # Initialize score
    swing_score = np.zeros(len(df))
    
    # Technical alignment (30% weight)
    if 'bullish_confluence' in df.columns:
        max_confluence = df['bullish_confluence'].max()
        if max_confluence > 0:
            swing_score += (df['bullish_confluence'] / max_confluence) * 0.3
    
    # Support/Resistance context (25% weight)
    if 'in_support_zone' in df.columns:
        swing_score += df['in_support_zone'].values * 0.25
    
    # Pattern strength (20% weight)
    if 'pattern_strength' in df.columns:
        max_pattern = df['pattern_strength'].max()
        if max_pattern > 0:
            swing_score += (df['pattern_strength'] / max_pattern) * 0.2
    
    # Volume confirmation (15% weight)
    if 'volume_ratio' in df.columns:
        volume_score = np.clip(df['volume_ratio'].values - 1, 0, 1)
        swing_score += volume_score * 0.15
    
    # Risk-reward potential (10% weight)
    if 'risk_reward_ratio' in df.columns:
        rr_score = np.clip(df['risk_reward_ratio'].values / 3, 0, 1)
        swing_score += rr_score * 0.1
    
    df['swing_quality_score'] = swing_score
    
    return df
