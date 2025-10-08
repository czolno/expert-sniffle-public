# Load the list of parquet files, and merge them into a single Data Frame.
import os
import pandas as pd

def get_split_paths(data_output_directory):
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

def get_data(data_output_directory, part, target):
    """
    Load the specified data split ('train', 'val', 'test') into a single DataFrame.
    """
    paths = get_split_paths(data_output_directory)
    df = pd.read_parquet(paths[part])
    X = df.drop(columns=[target])
    y = df[target]

    return X, y
