import pandas as pd
import numpy as np

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
    df['m_volatility_ratio_1min_15min'] = df['m_realized_vol_1min'] / df['m_realized_vol_15min']
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
    # shift in opposite direction (lead by one `freq` interval)
    df[new_col] = df[col].shift(periods=-1, freq=freq)
    # This makes the last N rows NaN, where N is number of periods in `freq`.
    # Let's drop those rows now: find the last valid index and truncate.
    last_valid_index = df[new_col].last_valid_index()
    df = df.loc[:last_valid_index]
    return df

def build_features(input_file, output_file):
    df = pd.read_parquet(input_file)
    # df = drop_unused_columns(df)
    df = calculate_core_features(df)
    df = add_future_shifted_feature(df, 'm_realized_vol_5min', 'm_realized_vol_5min_future', freq='5min')
    df.to_parquet(output_file)

