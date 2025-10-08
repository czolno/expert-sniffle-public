import os
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import time
import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # optional dependency
    tqdm = None

def _format_bytes(num: int) -> str:
    """Return human-readable memory size for a number of bytes."""
    for unit in ['B','KB','MB','GB','TB']:
        if num < 1024.0:
            return f"{num:3.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"

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

OANDA_MAX_FREQ_PER_SESSION = 100 # Max 100 requests per second

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
        time.sleep(1.0 / OANDA_MAX_FREQ_PER_SESSION)
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
        })
        if prefix == 'm':
            base['m_t'] = np.int32(c['volume'])  # ticks
        rows.append(base)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)
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
        return df


def fetch_interval_complete(instrument,
                            start_dt,
                            end_dt,
                            granularity='S5',
                            price_types=('M','B','A'),
                            chunk_hours=6,
                            save_path=None,
                            use_tqdm=True,
                            show_eta=True,
                            eta_smoothing=0.3):
    """
    Fetch candles for multiple price types over [start_dt, end_dt] and return a
    DataFrame whose index is the complete expected timeline for the OANDA
    granularity, with data filled in for available timestamps and (optionally)
    forward/other filled for missing ones.

    Strategy:
      1. Normalize times to UTC.
      2. Derive pandas frequency from OANDA granularity.
      3. Create the full DateTimeIndex covering [start_dt, end_dt].
      4. Iterate over consecutive chunk_hours (default 6h) intervals; each chunk
         is fetched with a SINGLE low-level request per price type
         (reuses existing `_fetch_price_type_single`).
         For S5 a 6h window => 4320 candles < 5000 limit => single request OK.
      5. Concatenate chunk results, drop duplicate indices, sort.
      6. Reindex onto the full index; fill missing values with 'ffill'.
      7. Optionally persist to parquet (save_path).

    Parameters
    ----------
    instrument : str
    start_dt, end_dt : datetime-like
    granularity : str (e.g. 'S5','M1','M5','M15','H1','D')
    price_types : iterable of {'M','B','A'}
    chunk_hours : int
    save_path : optional parquet output path
    use_tqdm : bool, default True
        If True and tqdm is installed, show a dynamic progress bar instead of static prints per chunk.
    show_eta : bool, default True
        If True, prints an ETA after each chunk using exponential moving average of chunk durations.
    eta_smoothing : float, default 0.3
        Smoothing factor (0<alpha<=1) for EMA of chunk durations when estimating ETA.

    Returns
    -------
    pandas.DataFrame
    """
    GRANULARITY_TO_FREQ = {
        'S5': '5s', 'S10': '10s', 'S15': '15s', 'S30': '30s',
        'M1': '1min', 'M2': '2min', 'M4': '4min', 'M5': '5min',
        'M10': '10min','M15': '15min','M30': '30min',
        'H1': '1H','H2':'2H','H3':'3H','H4':'4H','H6':'6H','H8':'8H','H12':'12H',
        'D': '1D', 'W': '1W'
    }
    if granularity not in GRANULARITY_TO_FREQ:
        raise ValueError(f"Unsupported granularity: {granularity}")

    start_dt = _ensure_utc(pd.to_datetime(start_dt))
    end_dt   = _ensure_utc(pd.to_datetime(end_dt))
    if end_dt < start_dt:
        raise ValueError("end_dt must be >= start_dt")

    freq = GRANULARITY_TO_FREQ[granularity]

    # Build the complete expected index (inclusive ends)
    full_index = pd.date_range(start=start_dt, end=end_dt, freq=freq, tz=timezone.utc)
    # Initialize result DataFrame with only index; columns will be added on demand
    result = pd.DataFrame(index=full_index)

    # Prepare chunk boundaries (6h default). We make the last edge exactly end_dt.
    chunk_edges = pd.date_range(start=start_dt, end=end_dt, freq=f'{chunk_hours}H', tz=timezone.utc)
    if chunk_edges[-1] != end_dt:
        chunk_edges = chunk_edges.append(pd.DatetimeIndex([end_dt]))

    total_chunks = len(chunk_edges) - 1
    ema_chunk_seconds = None
    overall_start_time = time.time()

    iterator = range(total_chunks)
    use_bar = use_tqdm and (tqdm is not None)
    if use_bar:
        iterator = tqdm(iterator, total=total_chunks, desc=f"{instrument} {granularity} chunks", unit="chunk")

    def _fmt(sec):
        if sec < 60:
            return f"{sec:0.1f}s"
        m, s = divmod(int(sec), 60)
        if m < 60:
            return f"{m}m{s:02d}s"
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m{s:02d}s"

    for i in iterator:
        chunk_start = chunk_edges[i]
        chunk_end   = chunk_edges[i + 1]
        iter_start_time = time.time()
        mem_bytes = result.memory_usage(deep=True).sum()
        mem_hr = _format_bytes(mem_bytes)
        if not use_bar:
            print(f"[fetch_interval_complete] Mem={mem_hr} | Chunk {i+1}/{total_chunks}: {instrument} {granularity} {chunk_start} -> {chunk_end}")
        for pt in price_types:
            df_pt = _fetch_price_type_single(instrument, chunk_start, chunk_end, granularity, pt)
            if df_pt is None or df_pt.empty:
                if not use_bar:
                    print(f"  - {pt}: no data")
                continue
            for col in df_pt.columns:
                if col not in result.columns:
                    result[col] = np.nan
            result.loc[df_pt.index, df_pt.columns] = df_pt
            if not use_bar:
                print(f"  - {pt}: {len(df_pt)} rows")
        # ETA
        if show_eta:
            elapsed_chunk = time.time() - iter_start_time
            if ema_chunk_seconds is None:
                ema_chunk_seconds = elapsed_chunk
            else:
                ema_chunk_seconds = eta_smoothing * elapsed_chunk + (1 - eta_smoothing) * ema_chunk_seconds
            chunks_left = total_chunks - (i + 1)
            eta_seconds = ema_chunk_seconds * chunks_left
            total_elapsed = time.time() - overall_start_time
            if use_bar:
                iterator.set_postfix({
                    'mem': mem_hr,
                    'last': f"{elapsed_chunk:0.2f}s",
                    'avg': f"{ema_chunk_seconds:0.2f}s",
                    'eta': _fmt(eta_seconds),
                    'elapsed': _fmt(total_elapsed)
                })
            else:
                pct = (i + 1) / total_chunks * 100
                print(f"    -> Progress: {pct:5.1f}% | Mem {mem_hr} | Last {elapsed_chunk:0.2f}s | Avg {ema_chunk_seconds:0.2f}s | ETA {_fmt(eta_seconds)} | Elapsed {_fmt(total_elapsed)}")

    result = result.ffill()

    # Trim leading rows with any NaNs using a faster heuristic: take the max of
    # each column's first valid index (earliest row where that column has data).
    # This avoids computing an all-columns row mask across the entire frame up front.
    if not result.empty and result.shape[1] > 0:
        first_valid = max([result[col].first_valid_index() for col in result.columns])
        result = result.loc[first_valid:]

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.to_parquet(save_path)
        print(f"[fetch_interval_complete] Saved to {save_path}")

    return result

if __name__ == "__main__":
    # Example usage:
    fetch_interval_complete('EUR_CHF', 
                            '2009-01-01', 
                            '2015-12-31', 
                            granularity='S5', 
                            price_types=('M','B','A'), 
                            save_path='data/examples/EUR_CHF_20090101_20151231_5S.parquet')

