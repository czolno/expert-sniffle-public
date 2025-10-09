import os
from pandas import Timestamp

# Centralized configuration for the project.

def get_config():
    #
    # PATHS:
    #
    data_dir = '../data/raw'
    feature_dir = '../data/processed'
    split_dir = '../data/split_2012_2014_2015'
    model_dir = '../models'
    return {
        'data_dir': data_dir,
        'feature_dir': feature_dir,
        'split_dir': split_dir,
        'model_dir': model_dir,
        #
        # DATA:
        # ====================================================
        # Data parameters:
        'instrument': 'EUR_CHF',
        'timeframe': 'S5',  # 5-second candles
        #
        # Date time ranges:
        'start_date': '2009-01-01',
        'end_date': '2015-12-31',
        'train_start': Timestamp('2012-01-01', tz='UTC'),
        'train_cutoff': Timestamp('2014-01-01', tz='UTC'),
        'val_cutoff': Timestamp('2015-01-01', tz='UTC'),
        #
        # Data file paths:
        'raw_data_path': os.path.join(data_dir, f'EUR_CHF_20090101_20151231_5S.parquet'),
        'feature_data_path': os.path.join(feature_dir, f'EUR_CHF_20090101_20151231_5S_features.parquet'),
        #
        # Data targets:
        'vol_source_col_name': 'm_realized_vol_5min',
        'vol_target_col_name': 'm_realized_vol_5min_future',
        'vol_shift_freq': '5min',
        #
        # MODEL:
        # ====================================================
        # Model file paths:
        'vol_model_path': os.path.join(model_dir, 'xgb_model_rv5min_MAE_2012_2014_2015.json'),
        #
    }


