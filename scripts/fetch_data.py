import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import time
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Get configuration from environment variables
API_TOKEN = os.getenv('OANDA_API_TOKEN')
ACCOUNT_ID = os.getenv('OANDA_ACCOUNT_ID')
API_URL = os.getenv('OANDA_API_URL', 'https://api-fxpractice.oanda.com')

if not API_TOKEN:
    raise ValueError("OANDA_API_TOKEN not found in .env file")
if not ACCOUNT_ID:
    raise ValueError("OANDA_ACCOUNT_ID not found in .env file")

# print(f"API URL: {API_URL}") 
# print(f"Account ID: {ACCOUNT_ID}")
# print("API Token: [HIDDEN]")

def list_instruments():
    """
    List all instruments available in the OANDA account.
    
    Returns:
        pandas.DataFrame: DataFrame containing instrument details
    """
    url = f"{API_URL}/v3/accounts/{ACCOUNT_ID}/instruments"
    
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        instruments = data.get('instruments', [])
        
        # Create a DataFrame with relevant information
        instruments_data = []
        for inst in instruments:
            instruments_data.append({
                'name': inst.get('name'),
                'type': inst.get('type'),
                'displayName': inst.get('displayName'),
                'pipLocation': inst.get('pipLocation'),
                'displayPrecision': inst.get('displayPrecision'),
                'tradeUnitsPrecision': inst.get('tradeUnitsPrecision'),
                'minimumTradeSize': inst.get('minimumTradeSize'),
                'maximumTrailingStopDistance': inst.get('maximumTrailingStopDistance'),
                'minimumTrailingStopDistance': inst.get('minimumTrailingStopDistance')
            })
        
        df = pd.DataFrame(instruments_data)
        print(f"Found {len(df)} instruments")
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching instruments: {e}")
        if hasattr(e.response, 'text'):
            print(f"Response: {e.response.text}")
        return None
    
def test_list_instruments():
    df = list_instruments()
    if df is not None:
        print(df.head())
    else:
        print("Failed to retrieve instruments.")

import cProfile
import pstats
from functools import wraps

def profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        print(f"\nProfiling results for {func.__name__}:")
        stats.print_stats(20)  # print top 20 lines
        return result
    return wrapper

# To profile a function, add @profile above its definition, e.g.:
# @profile
# def get_candles(...):
#     ...

OANDA_MAX_FREQ = 100 # Max 100 requests per second

def _ensure_utc(dt):
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

def _parquet_filename(instrument, year, granularity, save_dir=None):
    fname = f"{instrument}_{year}{'_' + granularity if granularity != 'D' else ''}.parquet"
    if save_dir:
        fname = os.path.join(save_dir, fname)
    return fname

def _log_oanda_request(instrument, granularity, price_type, from_time, to_time, status_code, url, params):
    import datetime
    log_path = "oanda_requests.log"
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    log_line = f"[{timestamp}] [{instrument}] [{granularity}] [{price_type}] [{from_time}] [{to_time}] [{status_code}] [{url}] [{params}]\n"
    with open(log_path, "a") as f:
        f.write(log_line)

def _oanda_candles_request(session, instrument, granularity, current_start, end_dt, price_type='M', count=5000):
    """Low-level single request wrapper. Returns (candles, status_code, error_message)."""
    current_start = current_start.isoformat().replace('+00:00', 'Z')
    end_dt = end_dt.isoformat().replace('+00:00', 'Z')
    url = f"{API_URL}/v3/instruments/{instrument}/candles"
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Content-Type': 'application/json'
    }
    params = {
        'Accept-Datetime-Format': 'RFC3339',
        'from': current_start,
        'granularity': granularity,
        'price': price_type,
        'count': count
    }
    if not count:
        params.pop('count')
        params['to'] = end_dt
    try:
        response = session.get(url, headers=headers, params=params)
        status_code = response.status_code
        _log_oanda_request(instrument, granularity, price_type, current_start, end_dt, status_code, url, params)
        response.raise_for_status()
        data = response.json()
        candles = data.get('candles', [])
        return candles, status_code, None
    except requests.exceptions.RequestException as e:
        status_code = getattr(e.response, 'status_code', 'ERR') if hasattr(e, 'response') and e.response is not None else 'ERR'
        _log_oanda_request(instrument, granularity, price_type, current_start, end_dt.isoformat(), status_code, url, params)
        error_message = str(e)
        if hasattr(e, 'response') and e.response is not None:
            error_message += f"\nResponse: {e.response.text}"
            print(f"Error fetching candles: {error_message}")
        return None, status_code, error_message

def _transform_candles(candles, price_type):
    rows = []
    for c in candles:
        base = {
            'time': c['time'], 
        }
        prefix, key = {'M': ('m', 'mid'), 'B': ('b', 'bid'), 'A': ('a', 'ask')}[price_type]
        base.update({ # 'm_o','m_h','m_l','m_c', etc.
            f'{prefix}_o': np.float32(c[key]['o']),
            f'{prefix}_h': np.float32(c[key]['h']),
            f'{prefix}_l': np.float32(c[key]['l']),
            f'{prefix}_c': np.float32(c[key]['c']),
            f'{prefix}_t': np.int32(c['volume'])
        })
        rows.append(base)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
    return df

def _fetch_price_type_paginated(instrument, start_dt, end_dt, granularity, price_type):
    """Generic pagination for a single price type (M,B,A)."""
    start_dt = _ensure_utc(start_dt); 
    end_dt = _ensure_utc(end_dt)
    all_parts = []
    current_start = start_dt
    with requests.Session() as session:
        while True:
            candles, status_code, error_message = _oanda_candles_request(session, instrument, granularity, current_start, end_dt, price_type=price_type, count=5000)
            if candles is None:
                if error_message:
                    print(f"Error fetching {price_type} candles for {instrument}: {error_message}")
                break
            if not candles:
                break
            part = _transform_candles(candles, price_type)
            if part is not None:
                all_parts.append(part)
            last_candle_time = max(c['time'] for c in candles)
            last_candle_dt = pd.to_datetime(last_candle_time, utc=True)
            if last_candle_dt > end_dt or len(candles) < 5000:
                break
            current_start = last_candle_time
            time.sleep(1.0 / OANDA_MAX_FREQ)
    if not all_parts:
        return None
    df = pd.concat(all_parts)
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df

def _fetch_price_type_single(instrument, start_dt, end_dt, granularity, price_type):
    """Single request for a single price type (M,B,A)."""
    start_dt = _ensure_utc(start_dt); 
    end_dt = _ensure_utc(end_dt)
    with requests.Session() as session:
        candles, status_code, error_message = _oanda_candles_request(session, instrument, granularity, start_dt, end_dt, price_type=price_type, count=None)
        if candles is None:
            if error_message:
                print(f"Error fetching {price_type} candles for {instrument}: {error_message}")
            return None
        df = _transform_candles(candles, price_type)
        if df is not None:
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        return df

def fetch_candles_and_merge_price_types(instrument, start_dt, end_dt, granularity, price_types=('M',), fetcher = _fetch_price_type_paginated):
    """High-level fetch for one or multiple price types; returns merged DataFrame."""
    frames = []
    for pt in price_types:
        f = fetcher(instrument, start_dt, end_dt, granularity, pt)
        if f is not None:
            frames.append(f)
    if not frames:
        return None
    merged = frames[0]
    for add in frames[1:]:
        merged = merged.join(add, how='outer')
    merged.sort_index(inplace=True)
    return merged

def get_candles_all_prices_during_interval(instrument, start_dt, end_dt, granularity='S5', save_path=None):
    # First, construct the list of datetimes to fetch, in chunks of 6 hours (less than max 5000 candles at S5):
    times = pd.date_range(start_dt, end_dt, freq='6h').append(pd.DatetimeIndex([end_dt])).unique()
    all_dfs = []
    for i in range(len(times)-1):
        chunk_start = times[i]
        chunk_end = times[i+1]
        print(f"Fetching all prices of {instrument} [{granularity}] from {chunk_start} to {chunk_end} ...", end=' ')
        df = fetch_candles_and_merge_price_types(instrument, chunk_start, chunk_end, granularity, price_types=('M','B','A'), fetcher=_fetch_price_type_single)
        if df is not None:
            all_dfs.append(df)
            print(f"got {len(df)} rows.")
        else:
            print("no candles found.")
    if not all_dfs:
        return None
    result = pd.concat(all_dfs)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_parquet(save_path)
    return result

def get_candles_all_prices_year_by_year(instrument, start_date='2005-01-01', end_date=None, granularity='S5', save_dir=None):
    start_dt = _ensure_utc(datetime.strptime(start_date, '%Y-%m-%d'))
    end_dt = _ensure_utc(datetime.strptime(end_date, '%Y-%m-%d')) if end_date else datetime.now(timezone.utc)
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    for year in range(start_dt.year, end_dt.year + 1):
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        fetch_start = max(start_dt, year_start)
        fetch_end = min(end_dt, year_end)
        df = fetch_candles_and_merge_price_types(instrument, fetch_start, fetch_end, granularity, price_types=('M','B','A'))
        if df is not None:
            fname = _parquet_filename(instrument, year, granularity + '_BA', save_dir)
            df.to_parquet(fname)
    return

def check_missing_data(input_dir):
    # For each time interval between the end of the current file and the beginning of the next file, try to download the missing data.
    parquet_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.parquet')])
    instrument = 'EUR_CHF'
    for i in range(len(parquet_files) - 1):
        current_file = parquet_files[i]
        next_file = parquet_files[i + 1]
        current_df = pd.read_parquet(os.path.join(input_dir, current_file))
        next_df = pd.read_parquet(os.path.join(input_dir, next_file))
        # End of the index:
        current_end_time = current_df.index.to_series().iloc[-1]
        next_start_time = next_df.index.to_series().iloc[0]
        if (next_start_time - current_end_time).total_seconds() > 5:
            print(f'Missing data between {current_end_time} and {next_start_time}. Attempting to fetch...')
            # Fetch the missing data:
            missing = get_candles_all_prices_during_interval(instrument, current_end_time, next_start_time)
            if missing is not None:
                print(f'Successfully fetched missing data for interval {current_end_time} to {next_start_time}. Saving to parquet.')
                # Format the times as "YYYYMMDDHHMMSS":
                t0 = current_end_time.strftime('%Y%m%d%H%M%S')
                t1 = next_start_time.strftime('%Y%m%d%H%M%S')
                missing.to_parquet(os.path.join(input_dir, f'missing_{t0}_{t1}.parquet'))