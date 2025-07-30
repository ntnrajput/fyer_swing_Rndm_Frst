import os
import time
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from concurrent.futures import ThreadPoolExecutor, as_completed
from data.fetch_auth import get_saved_access_token
from config import HISTORICAL_DATA_FILE, HISTORICAL_DATA_FILE_csv
from utils.logger import get_logger

logger = get_logger(__name__)

API_SLEEP = 0.5     # Sleep between API calls to respect rate limits
MAX_WORKERS = 3    # Number of parallel workers

def get_fyers_client():
    token = get_saved_access_token()
    if not token:
        raise ValueError("Access token is missing. Please authenticate.")
    return fyersModel.FyersModel(token=token, is_async=False, client_id=None)


def fetch_ohlcv_data_range(fyers, symbol, from_date, to_date, resolution="1D"):
    logger.info(f"Fetching data for {symbol} from {from_date.date()} to {to_date.date()}")

    all_data = []
    current_start = from_date

    while current_start < to_date:
        current_end = min(current_start + timedelta(days=364), to_date)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")

        logger.info(f"Fetching chunk: {symbol} from {start_str} to {end_str}")
        try:
            data = fyers.history({
                "symbol": symbol.strip(),
                "resolution": resolution,
                "date_format": "1",
                "range_from": start_str,
                "range_to": end_str,
                "cont_flag": "1"
            })

            if data["s"] == "ok":
                df = pd.DataFrame(data["candles"], columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["volume"] /= 1e5
                df["symbol"] = symbol
                df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                all_data.append(df.drop(columns=["timestamp"]))
                logger.info(f"Fetched {len(df)} records for {symbol} from {start_str} to {end_str}")
            else:
                logger.warning(f"Failed fetching {symbol} for {start_str} to {end_str}: {data.get('message')}, full response: {data}")

        except Exception as e:
            logger.exception(f"Exception fetching {symbol} from {start_str} to {end_str}: {e}")

        current_start = current_end + timedelta(days=1)
        time.sleep(API_SLEEP)

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def fetch_and_store_all(symbols, years=3):
    
    fyers = get_fyers_client()
    today = pd.Timestamp.now().normalize()
    print(today)
    start_date_default = today - timedelta(days=365 * years)

    # Step 1: Check if initial file exists
    if os.path.exists(HISTORICAL_DATA_FILE):
        existing_data = pd.read_parquet(HISTORICAL_DATA_FILE)
        existing_data['date'] = pd.to_datetime(existing_data['date'])
        logger.info(f"Loaded existing historical data from {HISTORICAL_DATA_FILE}")
    else:
        existing_data = pd.DataFrame()
        logger.info("No existing data found. Creating new dataset for all symbols...")


    def process_symbol(symbol):
       
        logger.info(f"Checking updates for {symbol}")
        symbol_data = existing_data[existing_data["symbol"] == symbol] if not existing_data.empty else pd.DataFrame()

        if not symbol_data.empty:
            latest_date = symbol_data["date"].max()
            fetch_from = latest_date
            if fetch_from > today:
                logger.info(f"{symbol} is up-to-date. Skipping fetch.")
                return symbol_data
            logger.info(f"Latest data for {symbol} is till {latest_date}. Refetching from {latest_date}")
            symbol_data = symbol_data[symbol_data["date"] < latest_date]
        else:
            fetch_from = start_date_default

        # if fetch_from >= today:
        #     logger.info(f"{symbol} is up-to-date. Skipping fetch.")
        #     return symbol_data

        new_data = fetch_ohlcv_data_range(fyers, symbol, fetch_from, today)
        time.sleep(API_SLEEP)

        combined = pd.concat([symbol_data, new_data], ignore_index=True) if not new_data.empty else symbol_data
        return combined

    updated_data = []

    

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                data = future.result()
                if not data.empty:
                    logger.info(f"Fetched & processed data for {symbol}: {len(data)} records")
                else:
                    logger.info(f"No new data for {symbol}")
                updated_data.append(data)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    if updated_data:
        full_data = pd.concat(updated_data, ignore_index=True)
        full_data.drop_duplicates(subset=["symbol", "date"], inplace=True)
        full_data.sort_values(by=["symbol", "date"], inplace=True)

        full_data.to_parquet(HISTORICAL_DATA_FILE, index=False)
        # full_data.to_csv(HISTORICAL_DATA_FILE_csv,index=False)
        logger.info(f"Historical data saved to {HISTORICAL_DATA_FILE}")
    else:
        logger.warning("No data was fetched or updated.")
