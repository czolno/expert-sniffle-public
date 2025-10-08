import os
from pandas import Timestamp

# Centralized configuration for the project.

def get_config():
    data_dir = '../data/raw'
    feature_dir = '../data/processed'
    return {
        'instrument': 'EUR_CHF',
        'timeframe': 'S5',  # 5-second candles
        'start_date': '2009-01-01',
        'end_date': '2015-12-31',
        'raw_data_path': os.path.join(data_dir, f'EUR_CHF_20090101_20151231_5S.parquet'),
        'feature_data_path': os.path.join(feature_dir, f'EUR_CHF_20090101_20151231_5S_features.parquet'),
        'train_cutoff': Timestamp('2014-01-01', tz='UTC'),
        'val_cutoff': Timestamp('2015-01-01', tz='UTC'),
        'data_dir': data_dir,
        'feature_dir': feature_dir,
        'split_dir': '../data/split',
    }


