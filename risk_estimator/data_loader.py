# Load the list of parquet files, and merge them into a single Data Frame.
import os
import pandas as pd

data_output_directory = '../data/processed'

vol_target = 'm_realized_vol_5min_future'

def get_split_paths():
    """
    If the dataset was split by [scripts/split_dataset.py](scripts/split_dataset.py),
    return the three file paths. Otherwise return None.
    """
    train_path = os.path.join(data_output_directory, 'train.parquet')
    val_path   = os.path.join(data_output_directory, 'val.parquet')
    test_path  = os.path.join(data_output_directory, 'test.parquet')
    if all(os.path.exists(p) for p in (train_path, val_path, test_path)):
        return {'train': train_path, 'val': val_path, 'test': test_path}
    return None

def clean_labels(X, y):
    # Ensure no NaNs in labels
    na = y.index[y.isna()]
    if (len(na) < 400):
        print("Warning: Found NaNs in y, interpolating.")
        y = y.interpolate()
        na = y.index[y.isna()]
        if len(na) > 0:
            X = X.drop(index=na)
            y = y.drop(index=na)
            print(f"Dropped {len(na)} rows with NaN after interpolation.")
    else:
        raise ValueError("Too many NaNs in y, please check the preprocessing.")
    return X, y

def get_data(part='train'):
    """
    Load the specified data split ('train', 'val', 'test') into a single DataFrame.
    """
    paths = get_split_paths()
    if paths is None:
        raise FileNotFoundError("Data splits not found. Please run the data preparation scripts first.")
    if part not in paths:
        raise ValueError(f"Invalid part specified: {part}. Choose from 'train', 'val', 'test'.")
    
    df = pd.read_parquet(paths[part])
    X = df.drop(columns=[vol_target])
    y = df[vol_target]

    X, y = clean_labels(X, y)

    return X, y
