# refactored_fyers_swing/main.py

import argparse
import pandas as pd
from config import HISTORICAL_DATA_FILE, LATEST_DATA_FILE, AVG_PRICE,AVG_VOL
from data.fetch_token import launch_browser_login
from data.fetch_stock_list import load_symbols
from data.fetch_historical_data import fetch_and_store_all
from features.engineer_features import add_technical_indicators
from model.train_model import train_model
from model.backtest_model import run_backtest
from strategy.screener import run_screener
from data.fetch_auth import generate_access_token
from utils.logger import get_logger
from time import sleep

logger = get_logger("Main")

def main():
    parser = argparse.ArgumentParser(description="Swing Trading AI System")
    parser.add_argument("--auth", action="store_true", help="Run authentication flow")
    parser.add_argument("--fetch-history", action="store_true", help="Fetch historical OHLCV data")
    parser.add_argument("--train", action="store_true", help="Train model")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on trained model")
    parser.add_argument("--screener", action="store_true", help="Run daily screener")
    parser.add_argument("--token", type=str, help="Generate access token using auth code")
    
    # Backtest parameters
    parser.add_argument("--capital", type=float, default=200000, help="Initial capital for backtest")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (0.001 = 0.1%)")
    parser.add_argument("--slippage", type=float, default=0.0005, help="Slippage rate (0.0005 = 0.05%)")
    parser.add_argument("--holding-period", type=int, default=21, help="Maximum holding period in days")
    parser.add_argument("--stop-loss", type=float, default=0.05, help="Stop loss percentage (0.05 = 5%)")
    parser.add_argument("--take-profit", type=float, default=0.15, help="Take profit percentage (0.08 = 8%)")
    parser.add_argument("--no-plots", action="store_true", help="Skip creating backtest plots")
    parser.add_argument("--save-charts", type=str, help="Path to save backtest charts")
    
    args = parser.parse_args()

    if args.auth:
        launch_browser_login()

    elif args.token:
        logger.info("Generating access token from auth code...")
        generate_access_token(args.token)

    elif args.fetch_history:
        symbols = load_symbols()
        fetch_and_store_all(symbols)

    elif args.train:
        logger.info(" Starting model training...")
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] < pd.to_datetime("2025-01-31")]
        df = add_technical_indicators(df)
        
        if 'target_hit' not in df.columns:
            logger.error("'target_hit' column not found. Cannot train model.")
        else:
            train_model(df)

        # Save latest data for screener
        df_latest = df.copy()
        latest_date = df_latest["date"].max()
        df_latest = df_latest[df_latest["date"] == latest_date].copy()
        logger.info(f" Saving latest data ({len(df_latest)} records) to {LATEST_DATA_FILE}")
        df_latest.to_parquet(LATEST_DATA_FILE, index=False)

    elif args.backtest:
        logger.info(" Starting backtest...")
        
        # Load historical data with features
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] > pd.to_datetime("2025-01-30")]
        
        filtered_symbols = []

        for symbol, group in df.groupby('symbol'):
            group = group.sort_values('date')  # sort by date to ensure correct order
            
            # Skip if less than 50 records available
            if len(group) < 50:
                continue

            avg_volume = group['volume'].tail(50).mean()
            latest_close = group.iloc[-1]['close']

            if avg_volume > AVG_VOL and latest_close > AVG_PRICE:
                filtered_symbols.append(symbol)

        # Filter original df to keep only those symbols
        df = df[df['symbol'].isin(filtered_symbols)]

        df = add_technical_indicators(df)
        
        # Prepare backtest parameters
        backtest_params = {
            'initial_capital': args.capital,
            'commission': args.commission,
            'slippage': args.slippage,
            'holding_period': args.holding_period,
            'stop_loss_pct': args.stop_loss,
            'take_profit_pct': args.take_profit,
            'create_plots': not args.no_plots,
            'save_path': args.save_charts
        }
        
        logger.info(" Backtest Parameters:")
        logger.info(f"   Initial Capital: ${backtest_params['initial_capital']:,.2f}")
        logger.info(f"   Commission: {backtest_params['commission']*100:.2f}%")
        logger.info(f"   Slippage: {backtest_params['slippage']*100:.3f}%")
        logger.info(f"   Holding Period: {backtest_params['holding_period']} days")
        logger.info(f"   Stop Loss: {backtest_params['stop_loss_pct']*100:.1f}%")
        logger.info(f"   Take Profit: {backtest_params['take_profit_pct']*100:.1f}%")
        
        # Run backtest
        results = run_backtest(df, **backtest_params)
        
        if results:
            logger.info("Backtest completed successfully!")
            
            # Save detailed results
            if args.save_charts:
                results_path = args.save_charts.replace('.png', '_results.csv')
                if 'trades_df' in results and not results['trades_df'].empty:
                    results['trades_df'].to_csv(results_path, index=False)
                    logger.info(f"📄 Detailed trade results saved to {results_path}")
        else:
            logger.error(" Backtest failed!")

    elif args.screener:
        logger.info(" Starting model training...")
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        df = add_technical_indicators(df)
        df_latest = df.copy()
        latest_date = df_latest["date"].max()
        df_latest = df_latest[df_latest["date"] == latest_date].copy()
        run_screener(df_latest)

    else:
        logger.info("No argument provided. Available options:")
        logger.info("  --auth: Run authentication flow")
        logger.info("  --fetch-history: Fetch historical data")
        logger.info("  --train: Train the model")
        logger.info("  --backtest: Run backtest simulation")
        logger.info("  --screener: Run daily screener")
        logger.info("  --help: Show all options")

if __name__ == "__main__":
    main()