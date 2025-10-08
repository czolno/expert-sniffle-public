# New file: memory-efficient splitter for processed merged parquet
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from datetime import datetime

PROCESSED_PATH = 'data/processed/merged_data.parquet'
OUT_DIR = 'data/processed'
TRAIN_PATH = os.path.join(OUT_DIR, 'train.parquet')
VAL_PATH   = os.path.join(OUT_DIR, 'val.parquet')
TEST_PATH  = os.path.join(OUT_DIR, 'test.parquet')

# Define split cutoffs (time-based)
TRAIN_CUTOFF = pd.Timestamp('2014-01-01', tz='UTC')
VAL_CUTOFF   = pd.Timestamp('2015-01-01', tz='UTC')  # val: [2014-01-01, 2015-01-01)

def _make_writer(path, schema):
    return pq.ParquetWriter(path, schema)

def _ensure_index(df):
    # If index was saved as column, restore it
    if 'index' in df.columns and df.index.name is None:
        df = df.set_index('index')
    elif 'time' in df.columns and df.index.name is None:
        df = df.set_index('time')
    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index, utc=True)
    return df

def drop_first_NAs(df):
    y_col = 'm_realized_vol_5min_future'
    # Drop rows with NaN in the first N rows
    first_valid_index = df[y_col].first_valid_index()
    df = df.loc[first_valid_index:]
    return df

def split_processed_parquet(processed_path=PROCESSED_PATH, train_path=TRAIN_PATH, val_path=VAL_PATH, test_path=TEST_PATH):
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    pf = pq.ParquetFile(processed_path)
    writers = {}
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            df = table.to_pandas()
            df = _ensure_index(df)

            # Partition by time windows
            df_train = df[df.index < TRAIN_CUTOFF]
            df_val   = df[(df.index >= TRAIN_CUTOFF) & (df.index < VAL_CUTOFF)]
            df_test  = df[df.index >= VAL_CUTOFF]

            # Drop initial NaNs in each subset
            df_train = drop_first_NAs(df_train)
            df_val   = drop_first_NAs(df_val)
            df_test  = drop_first_NAs(df_test)

            for subset_df, path in ((df_train, train_path), (df_val, val_path), (df_test, test_path)):
                if subset_df.empty:
                    continue
                table_out = pa.Table.from_pandas(subset_df, preserve_index=True)
                if path not in writers:
                    writers[path] = _make_writer(path, table_out.schema)
                writers[path].write_table(table_out)
                print(f"Appended {len(subset_df)} rows to {path} (row_group {rg})")
    finally:
        for w in writers.values():
            w.close()
    print("Splitting complete. Output files:", [train_path, val_path, test_path])

if __name__ == "__main__":
    split_processed_parquet()