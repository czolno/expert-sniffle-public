# Memory-efficient splitter for parquet datasets
import os
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd

def _make_writer(path, schema):
    return pq.ParquetWriter(path, schema)

def split_processed_parquet(processed_path, out_dir, 
                            train_start,
                            train_cutoff, 
                            val_cutoff):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    train_path = os.path.join(out_dir, 'train.parquet')
    val_path   = os.path.join(out_dir, 'val.parquet')
    test_path  = os.path.join(out_dir, 'test.parquet')

    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed file not found: {processed_path}")

    pf = pq.ParquetFile(processed_path)
    writers = {}
    try:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg)
            df = table.to_pandas()

            # Partition by time windows
            df_train = df[(df.index >= train_start) & (df.index < train_cutoff)]
            df_val   = df[(df.index >= train_cutoff) & (df.index < val_cutoff)]
            df_test  = df[df.index >= val_cutoff]

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
