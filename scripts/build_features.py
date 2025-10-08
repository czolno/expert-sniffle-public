import pandas as pd
import numpy as np
import os
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


def drop_unused_columns(df):
    # Drop columns that are not used in feature calculations
    df.drop(columns=[
        'a_o', 'a_h', 'a_l', 
        'b_o', 'b_h', 'b_l', 
        'm_o', 'm_h', 'm_l'
    ], inplace=True) 
    return df

def complete_to_uniform_time_index(df, freq='5s'):
    # Ensure the Data Frame has a uniform time index with specified frequency
    df = df.sort_index()
    full_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq)
    df = df[~df.index.duplicated(keep='first')]
    df = df.reindex(full_index)
    df[['a_c', 'b_c', 'm_c']] = df[['a_c', 'b_c', 'm_c']].ffill()  # Forward fill prices
    df['m_t'] = df['m_t'].fillna(0)  # Fill missing tick counts with 0
    return df

def since_last_nonzero_nonshifted(series):
    mask = series != 0
    # cumsum increases at each nonzero, so group by this
    group = mask.cumsum()
    # Where group==0, it's before the first nonzero, so set to nan
    result = series.groupby(group).cumcount() + 1
    result[group == 0] = np.nan
    return result.values

def add_since_last_nonzero(df, col, new_col):
    df[new_col] = since_last_nonzero_nonshifted(df[col])
    df[new_col] = df[new_col].shift()
    return df

def calculate_core_features(df):

    # Assumes df has a uniform datetime index and necessary columns
    df.sort_index(inplace=True)

    #
    # Spread-related features:
    #
    # - spread: a_c - b_c
    df['spread'] = df['a_c'] - df['b_c']
    # - spread rolling mean (window=1 min)
    df['spread_rolling_mean_1min'] = df['spread'].rolling(window='1min').mean()
    # - spread rolling mean (window=5 min)
    df['spread_rolling_mean_5min'] = df['spread'].rolling(window='5min').mean()
    # - spread rolling std (window=5 min)
    df['spread_rolling_std_5min'] = df['spread'].rolling(window='5min').std()
    # - spread percentile (window=5 min)
    df['spread_percentile_5min'] = df['spread'].rolling(window='5min').apply(lambda x: (x <= x.iloc[-1]).mean())
    #
    # Liquidity proxy features:
    #
    # - number of ticks in the last ...
    # df['ticks_5s'] = df['m_t'] # would be redundant
    df['ticks_30s'] = df['m_t'].rolling(window='30s').sum()
    df['ticks_1min'] = df['m_t'].rolling(window='1min').sum()
    #  - time since last update (in seconds; looking for previous non-zero tick count)
    df = add_since_last_nonzero(df, 'm_t', 'time_since_last_update')
    df['time_since_last_update'] = df['time_since_last_update'].fillna(0) * 5  # Assuming original freq is 5s
    # - inactivity flag (1 if > 5s since last update, else 0; over 75% of data points are from active periods)
    df['inactivity_flag_5s'] = (df['time_since_last_update'] > 5).astype(int)
    #
    # Price/volatility features:
    #
    # - log prices for current mid:
    df['m_log_c'] = np.log(df['m_c'])
    # - mid price returns (log returns)
    df['m_log_return'] = df['m_log_c'].diff().fillna(0)
    # - squared mid price returns
    df['m_log_return_sq'] = df['m_log_return'] ** 2
    # - realized volatility
    df['m_realized_vol_1min'] = df['m_log_return_sq'].rolling(window='1min').sum() ** 0.5
    df['m_realized_vol_5min'] = df['m_log_return_sq'].rolling(window='5min').sum() ** 0.5
    df['m_realized_vol_15min'] = df['m_log_return_sq'].rolling(window='15min').sum() ** 0.5
    # - rollling max abs return
    df['m_max_abs_return_1min'] = df['m_log_return'].abs().rolling(window='1min').max()
    # - volatility ratio
    df['m_volatility_ratio_1min_15min'] = df['m_realized_vol_1min'] / df['m_realized_vol_15min'].replace(0, np.nan)
    #
    # Temporal features:
    #
    # - hour of day (0-23)
    df['hour_of_day'] = df.index.hour
    # - day of week (0=Monday, 6=Sunday)
    df['day_of_week'] = df.index.dayofweek
    # - is high liquidity hour (e.g., 05:00-17:00 UTC)
    df['is_high_liquidity_hour'] = df['hour_of_day'].between(5, 17, inclusive='both').astype(int)
    #
    # Regime indicator features:
    #
    # - high spread flag (1 if spread > rolling mean + 2*std over 5 min, else 0)
    df['high_spread_flag'] = (df['spread'] > (df['spread_rolling_mean_5min'] + 2 * df['spread_rolling_std_5min'])).astype(int)
    # - vol regime flag (1 if 1-min realized vol > P90 30d distribution, else 0)
    df['vol_regime_flag'] = (df['m_realized_vol_1min'] > df['m_realized_vol_1min'].rolling(window='30d').quantile(0.9)).astype(int)
    return df

def add_future_shifted_feature(df, col, new_col, freq='5min'):
    df[new_col] = df[col].shift(freq=freq)
    return df

# def process_parquet_file(input_file, output_file):
#     df = pd.read_parquet(input_file)
#     df = drop_unused_columns(df)
#     df = complete_to_uniform_time_index(df, freq='5s') # Will make right the distributions of time-based rolling features 
#     df = calculate_core_features(df)
#     df = add_future_shifted_feature(df, 'm_realized_vol_5min', 'm_realized_vol_5min_future', freq='5min')
#     df.to_parquet(output_file)

# def add_core_features_to_parquet(input_dir, output_dir):
#     for filename in sorted(os.listdir(input_dir)):
#         if filename.endswith('.parquet'):
#             input_path = os.path.join(input_dir, filename)
#             output_path = os.path.join(output_dir, filename)
#             process_parquet_file(input_path, output_path)
#             print(f'Added core features to {filename} and saved to {output_path}')

def merge_raw_parquets(input_dir, merged_raw_path):
    """
    Merge all parquet files in input_dir into a single parquet (arrow) file.
    This uses pyarrow.dataset to avoid loading everything into RAM at once.
    """
    dataset = ds.dataset(input_dir, format="parquet")
    scanner = dataset.scanner()
    table = scanner.to_table()
    os.makedirs(os.path.dirname(merged_raw_path), exist_ok=True)
    pq.write_table(table, merged_raw_path)
    print(f"Merged raw parquet written to {merged_raw_path}")
    return merged_raw_path

def merge_and_process_all(input_dir='data/raw', output_dir='data/processed',
                          merged_raw_name='merged_raw.parquet', processed_name='merged_data.parquet'):
    """
    1) Merge all raw parquet files from input_dir into a single raw parquet
    2) Load the merged raw parquet into pandas (single DF), run the existing
       processing pipeline and write a single processed parquet to output_dir.
    """
    merged_raw_path = os.path.join(output_dir, merged_raw_name)
    processed_path  = os.path.join(output_dir, processed_name)
    os.makedirs(output_dir, exist_ok=True)

    # 1) Merge raw files into one arrow/parquet
    merge_raw_parquets(input_dir, merged_raw_path)

    # 2) Load merged raw into pandas and process (reuse existing pipeline)
    print(f"Loading merged raw parquet from {merged_raw_path} into pandas...")
    df = pd.read_parquet(merged_raw_path)

    # Ensure proper index name if necessary
    if 'time' in df.columns and df.index.name is None:
        df = df.set_index('time')

    df = drop_unused_columns(df)
    df = complete_to_uniform_time_index(df, freq='5s') # existing helper
    df = calculate_core_features(df)                   # existing helper
    # Add target column for forecasting:
    y_col = 'm_realized_vol_5min_future'
    df = add_future_shifted_feature(df, 'm_realized_vol_5min', y_col, freq='5min')
    # Drop rows with NaN in the target column (due to shifting)
    first_valid_index = df[y_col].first_valid_index()
    df = df.loc[first_valid_index:]

    # Save single processed parquet
    df.to_parquet(processed_path)
    print(f"Processed merged data written to {processed_path}")

if __name__ == "__main__":
    # When run directly, merge raw -> process -> produce single processed parquet
    merge_and_process_all(input_dir='data/raw', output_dir='data/processed')
