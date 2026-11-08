from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "final_data"

file_path = DATA_DIR / "train" / "beeradvocate_reviews.parquet"

parquet_file = pq.ParquetFile(file_path)

beer_ids = set()
missing_count = 0

for batch in parquet_file.iter_batches(
    batch_size=50000,
    columns=["beer_id"]
):
    for beer_id in batch.column("beer_id").to_pylist():
        if beer_id is None or not str(beer_id).strip():
            missing_count += 1
        else:
            beer_ids.add(str(beer_id).strip())

print("=== Beer ID Verification ===")
print(f"Unique beer IDs: {len(beer_ids):,}")
print(f"Missing beer IDs: {missing_count:,}")
print(f"Total rows: {parquet_file.metadata.num_rows:,}")

df = pd.read_parquet(file_path)

missing_rows = df[
    df["beer_id"].isna()
    | df["beer_id"].astype("string").str.strip().eq("").fillna(False)
]
print("\n=== Missing Beer ID Rows ===")
print(missing_rows.to_string(index=False))

print("\nStep 23 Completed.")

