# model/backtest_model.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import joblib
import warnings
from time import sleep
warnings.filterwarnings('ignore')

from utils.logger import get_logger
from config import MODEL_FILE, CONFIDENCE_THRESHOLD

logger = get_logger(__name__)

class SwingTradingBacktester:
    def __init__(self, initial_capital=100000, commission=0.001, slippage=0.0005):
        """
        Initialize backtester with trading parameters
        
        Args:
            initial_capital: Starting portfolio value
            commission: Transaction cost (0.001 = 0.1%)
            slippage: Price slippage (0.0005 = 0.05%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
        self.portfolio_history = []
        self.current_positions = {}
        self.cash = initial_capital
        self.total_portfolio_value = initial_capital
        
    def load_model(self, model_path):
        """Load trained model pipeline"""
        try:
            self.model_pipeline = joblib.load(MODEL_FILE)
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f" Failed to load model: {e}")
            return False
        
    def prepare_features(self, df):
        """Prepare features using the model pipeline"""
        try:
            full_feature_list = self.model_pipeline['all_features']
            
            missing = [f for f in full_feature_list if f not in df.columns]
            if missing:
                logger.warning(f"Missing features: {missing}")
                return None

            X = df[full_feature_list].copy()

            # Drop rows with any NaN values
            X = X.dropna()
            if X.empty:
                logger.warning("All rows dropped due to NaNs in features")
                return None

            df = df.loc[X.index]  # Align original df with cleaned X

            self.features_for_prediction = X  # Store features for use in prediction

            return df  # Cleaned df (not transformed!)

        except Exception as e:
            logger.error(f" Error preparing features: {e}")
            return None

    def generate_signals(self, df):
        """Generate trading signals using the trained model"""
        try:
            if not hasattr(self, 'features_for_prediction'):
                logger.error("Features not prepared properly")
                return None

            X = self.features_for_prediction  # Cleaned raw features

            pipeline = self.model_pipeline['pipeline']  # This includes scaler + selector + model
            predictions = pipeline.predict(X)
            probabilities = pipeline.predict_proba(X)[:, 1]

            # Add predictions to DataFrame
            df = df.copy()
            df['strong_signal'] = predictions
            df['signal_confidence'] = probabilities

            df.to_csv('backtest_pred.csv')

            return df

        except Exception as e:
            logger.error(f" Error generating signals: {e}")
            return df
    
    def calculate_position_size(self, current_price, confidence, volatility=None):
        """Calculate position size based on confidence and risk management"""
        # Base position size as percentage of portfolio
        base_position_pct = 0.05  # 5% of portfolio per trade
        
        # Adjust based on confidence (0.5 to 1.0 range)
        confidence_multiplier = max(0.5, min(2.0, confidence * 2))
        
        # Adjust based on volatility if available
        if volatility is not None:
            volatility_multiplier = max(0.5, min(2.0, 1 / (volatility + 0.01)))
        else:
            volatility_multiplier = 1.0
        
        # Calculate position size
        position_value = (self.total_portfolio_value * base_position_pct * 
                         confidence_multiplier * volatility_multiplier)
        
        # Number of shares (rounded down)
        shares = int(position_value / current_price)
        
        return max(1, shares)  # Minimum 1 share
    
    def execute_trade(self, symbol, action, price, shares, date, confidence=None):
        """Execute a trade and update portfolio"""
        # Calculate costs
        trade_value = shares * price
        commission_cost = trade_value * self.commission
        slippage_cost = trade_value * self.slippage
        total_cost = commission_cost + slippage_cost
        
        if action == 'BUY':
            # Check if we have enough cash
            total_required = trade_value + total_cost
            if self.cash >= total_required:
                self.cash -= total_required
                
                # Add to positions
                if symbol not in self.current_positions:
                    self.current_positions[symbol] = {
                        'shares': 0,
                        'avg_price': 0,
                        'entry_date': date
                    }
                
                # Update position
                current_shares = self.current_positions[symbol]['shares']
                current_value = current_shares * self.current_positions[symbol]['avg_price']
                new_avg_price = (current_value + trade_value) / (current_shares + shares)
                
                self.current_positions[symbol]['shares'] += shares
                self.current_positions[symbol]['avg_price'] = new_avg_price
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'value': trade_value,
                    'commission': commission_cost,
                    'slippage': slippage_cost,
                    'confidence': confidence
                })
                
                return True
            else:
                logger.warning(f"Insufficient cash for {symbol} purchase")
                return False
                
        elif action == 'SELL':
            # Check if we have the position
            if symbol in self.current_positions and self.current_positions[symbol]['shares'] >= shares:
                self.cash += trade_value - total_cost
                
                # Update position
                self.current_positions[symbol]['shares'] -= shares
                
                # Calculate P&L
                entry_price = self.current_positions[symbol]['avg_price']
                pnl = (price - entry_price) * shares - total_cost
                pnl_pct = (price - entry_price) / entry_price * 100
                
                # Record trade
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'shares': shares,
                    'price': price,
                    'value': trade_value,
                    'commission': commission_cost,
                    'slippage': slippage_cost,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'entry_price': entry_price,
                    'confidence': confidence
                })
                
                # Remove position if no shares left
                if self.current_positions[symbol]['shares'] == 0:
                    del self.current_positions[symbol]
                
                return True
            else:
                logger.warning(f"Insufficient shares for {symbol} sale")
                return False
    
    def update_portfolio_value(self, current_prices):
        """Update total portfolio value based on current prices"""
        position_value = 0
        for symbol, position in self.current_positions.items():
            if symbol in current_prices:
                position_value += position['shares'] * current_prices[symbol]
        
        self.total_portfolio_value = self.cash + position_value
        return self.total_portfolio_value
    
    def run_backtest(self, df, holding_period=10, stop_loss_pct=0.03, take_profit_pct=0.08):
        """
        Run complete backtest simulation
        
        Args:
            df: DataFrame with OHLCV data and signals
            holding_period: Days to hold position
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        """
        logger.info(" Starting backtest simulation...")
        df = self.prepare_features(df)
        if df is None:
            logger.error("Features not prepared properly")
            return None
            
        # Generate signals
        df = self.generate_signals(df)
        if df is None:
            logger.error("Signal generation failed")
            return None
        
        # Sort by date for proper time series processing
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # Track daily portfolio values
        daily_portfolio = {}
        
        for i, row in df.iterrows():
            date = row['date']
            symbol = row['symbol']
            current_price = row['close']
            
            # Update daily portfolio tracking
            if date not in daily_portfolio:
                daily_portfolio[date] = {}
            daily_portfolio[date][symbol] = current_price
            
            # Process buy signals
            if row['strong_signal'] == 1 and row['signal_confidence'] > CONFIDENCE_THRESHOLD and row['strong_rejection']!= 1 and symbol not in self.current_positions:
            # if row['strong_signal'] == 1 and row['signal_confidence'] > CONFIDENCE_THRESHOLD and row['strong_rejection']!= 1 :
                confidence = row['signal_confidence']
                volatility = row.get('volatility_20', None)
                
                shares = self.calculate_position_size(current_price, confidence, volatility)
                
                if self.execute_trade(symbol, 'BUY', current_price, shares, date, confidence):
                    logger.info(f"BUY {shares} shares of {symbol} at ${current_price:.2f} on {date}")
            
            # Process sell signals (stop loss, take profit, holding period)
            elif symbol in self.current_positions:
                position = self.current_positions[symbol]
                entry_price = position['avg_price']
                entry_date = position['entry_date']
                
                # Calculate current P&L
                current_pnl_pct = (current_price - entry_price) / entry_price
                
                # Calculate holding period
                holding_days = (pd.to_datetime(date) - pd.to_datetime(entry_date)).days
                
                # Sell conditions
                sell_reason = None
                if current_pnl_pct <= -stop_loss_pct:
                    sell_reason = "Stop Loss"
                elif current_pnl_pct >= take_profit_pct:
                    sell_reason = "Take Profit"
                elif holding_days >= holding_period:
                    sell_reason = "Holding Period"
                if sell_reason:
                    shares_to_sell = position['shares']
                    if self.execute_trade(symbol, 'SELL', current_price, shares_to_sell, date):
                        logger.info(f" SELL {shares_to_sell} shares of {symbol} at ${current_price:.2f} on {date} - {sell_reason}")
        
        # Calculate final portfolio value
        final_prices = {}
        for symbol in self.current_positions.keys():
            final_data = df[df['symbol'] == symbol].iloc[-1]
            final_prices[symbol] = final_data['close']
        
        final_portfolio_value = self.update_portfolio_value(final_prices)
        
        # Calculate daily portfolio history
        for date, prices in daily_portfolio.items():
            portfolio_value = self.update_portfolio_value(prices)
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })
        
        logger.info(f" Backtest completed. Final portfolio value: ${final_portfolio_value:,.2f}")
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            logger.warning("No trades executed during backtest")
            return {}
        
        # Convert trades to DataFrame
        trades_df = pd.DataFrame(self.trades)
        portfolio_df = pd.DataFrame(self.portfolio_history)
        
        # Basic metrics
        total_trades = len(trades_df)
        buy_trades = trades_df[trades_df['action'] == 'BUY']
        sell_trades = trades_df[trades_df['action'] == 'SELL']
        
        # P&L metrics
        if not sell_trades.empty:
            total_pnl = sell_trades['pnl'].sum()
            winning_trades = sell_trades[sell_trades['pnl'] > 0]
            losing_trades = sell_trades[sell_trades['pnl'] < 0]
            
            win_rate = len(winning_trades) / len(sell_trades) * 100
            avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
            
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if not losing_trades.empty else float('inf')
        else:
            total_pnl = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Portfolio metrics
        final_value = self.total_portfolio_value
        total_return = (final_value - self.initial_capital) / self.initial_capital * 100
        
        # Calculate drawdown
        if not portfolio_df.empty:
            portfolio_df['cumulative_max'] = portfolio_df['portfolio_value'].cummax()
            portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cumulative_max']) / portfolio_df['cumulative_max'] * 100
            max_drawdown = portfolio_df['drawdown'].min()
            
            # Calculate daily returns for Sharpe ratio
            portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
            daily_returns = portfolio_df['daily_return'].dropna()
            
            if len(daily_returns) > 1:
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)  # Annualized
                volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized %
            else:
                sharpe_ratio = 0
                volatility = 0
        else:
            max_drawdown = 0
            sharpe_ratio = 0
            volatility = 0
        
        # Compile results
        results = {
            'total_trades': total_trades,
            'completed_trades': len(sell_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_return': total_return,
            'final_portfolio_value': final_value,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'trades_df': trades_df,
            'portfolio_df': portfolio_df
        }
        
        return results
    
    def generate_report(self, results):
        """Generate comprehensive backtest report"""
        logger.info("\n" + "="*80)
        logger.info(" BACKTEST PERFORMANCE REPORT")
        logger.info("="*80)
        
        # Trading Statistics
        logger.info(" TRADING STATISTICS:")
        logger.info(f"   Total Trades: {results['total_trades']}")
        logger.info(f"   Completed Trades: {results['completed_trades']}")
        logger.info(f"   Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"   Profit Factor: {results['profit_factor']:.2f}")
        
        # P&L Analysis
        logger.info("\n P&L ANALYSIS:")
        logger.info(f"   Total P&L: ${results['total_pnl']:,.2f}")
        logger.info(f"   Average Win: ${results['avg_win']:,.2f}")
        logger.info(f"   Average Loss: ${results['avg_loss']:,.2f}")
        
        # Portfolio Performance
        logger.info("\n PORTFOLIO PERFORMANCE:")
        logger.info(f"   Initial Capital: ${self.initial_capital:,.2f}")
        logger.info(f"   Final Value: ${results['final_portfolio_value']:,.2f}")
        logger.info(f"   Total Return: {results['total_return']:.2f}%")
        logger.info(f"   Max Drawdown: {results['max_drawdown']:.2f}%")
        
        # Risk Metrics
        logger.info("\n RISK METRICS:")
        logger.info(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"   Volatility: {results['volatility']:.2f}%")
        
        # Performance Rating
        logger.info("\n PERFORMANCE RATING:")
        if results['win_rate'] >= 60 and results['sharpe_ratio'] >= 1.5:
            logger.info("    EXCELLENT - Strategy shows strong performance")
        elif results['win_rate'] >= 50 and results['sharpe_ratio'] >= 1.0:
            logger.info("    GOOD - Strategy shows decent performance")
        elif results['win_rate'] >= 45 and results['sharpe_ratio'] >= 0.5:
            logger.info("    MODERATE - Strategy needs improvement")
        else:
            logger.info("    POOR - Strategy requires significant optimization")
        
        logger.info("="*80)
        
        return results
    
    def plot_results(self, results, save_path=None):
        """Create visualization plots for backtest results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtest Results Dashboard', fontsize=16, fontweight='bold')
            
            portfolio_df = results['portfolio_df']
            trades_df = results['trades_df']
            trades_df.to_excel('backtest_df.xlsx')
            
            # 1. Portfolio Value Over Time
            if not portfolio_df.empty:
                axes[0, 0].plot(pd.to_datetime(portfolio_df['date']), 
                               portfolio_df['portfolio_value'], 
                               linewidth=2, color='blue')
                axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.7)
                axes[0, 0].set_title('Portfolio Value Over Time')
                axes[0, 0].set_ylabel('Portfolio Value ($)')
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Drawdown Chart
            if not portfolio_df.empty and 'drawdown' in portfolio_df.columns:
                axes[0, 1].fill_between(pd.to_datetime(portfolio_df['date']), 
                                       portfolio_df['drawdown'], 0, 
                                       alpha=0.7, color='red')
                axes[0, 1].set_title('Drawdown Analysis')
                axes[0, 1].set_ylabel('Drawdown (%)')
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Trade P&L Distribution
            if not trades_df.empty:
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                if not sell_trades.empty:
                    axes[1, 0].hist(sell_trades['pnl'], bins=20, alpha=0.7, 
                                   color='green', edgecolor='black')
                    axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
                    axes[1, 0].set_title('Trade P&L Distribution')
                    axes[1, 0].set_xlabel('P&L ($)')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Win Rate by Month
            if not trades_df.empty:
                sell_trades = trades_df[trades_df['action'] == 'SELL'].copy()
                if not sell_trades.empty:
                    sell_trades['month'] = pd.to_datetime(sell_trades['date']).dt.to_period('M')
                    monthly_stats = sell_trades.groupby('month').agg({
                        'pnl': lambda x: (x > 0).sum() / len(x) * 100
                    }).reset_index()
                    
                    if not monthly_stats.empty:
                        axes[1, 1].bar(range(len(monthly_stats)), monthly_stats['pnl'], 
                                      alpha=0.7, color='blue')
                        axes[1, 1].set_title('Monthly Win Rate')
                        axes[1, 1].set_ylabel('Win Rate (%)')
                        axes[1, 1].set_xlabel('Month')
                        axes[1, 1].set_xticks(range(len(monthly_stats)))
                        axes[1, 1].set_xticklabels([str(m) for m in monthly_stats['month']], rotation=45)
                        axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f" Charts saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f" Error creating plots: {e}")

def run_backtest(df, model_path=None, **kwargs):
    """
    Main function to run backtest
    
    Args:
        df: DataFrame with OHLCV data and features
        model_path: Path to trained model (defaults to MODEL_FILE)
        **kwargs: Additional parameters for backtesting
    """
    try:
        # Initialize backtester
        backtester = SwingTradingBacktester(
            initial_capital=kwargs.get('initial_capital', 100000),
            commission=kwargs.get('commission', 0.001),
            slippage=kwargs.get('slippage', 0.0005)
        )
        
        # Load model
        model_path = MODEL_FILE
        if not backtester.load_model(model_path):
            return None
        
        # Run backtest
        results = backtester.run_backtest(
            df,
            holding_period=kwargs.get('holding_period', 10),
            stop_loss_pct=kwargs.get('stop_loss_pct', 0.03),
            take_profit_pct=kwargs.get('take_profit_pct', 0.08)
        )

        # Generate report
        results = backtester.generate_report(results)
        
        # Create plots
        if kwargs.get('create_plots', True):
            backtester.plot_results(results, kwargs.get('save_path'))
        
        return results
        
    except Exception as e:
        logger.error(f" Backtest failed: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from config import HISTORICAL_DATA_FILE
    
    # Load data
    df = pd.read_parquet(HISTORICAL_DATA_FILE)
    
    # Run backtest
    results = run_backtest(df, create_plots=True)