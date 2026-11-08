
from pathlib import Path
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

file_path = PROCESSED_DIR / "beeradvocate_quality.parquet"

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
            beer_ids.add(str(beer_id))

print("=== Beer ID Verification ===")
print(f"Unique beer IDs: {len(beer_ids):,}")
print(f"Missing beer IDs: {missing_count:,}")
print(f"Total rows: {parquet_file.metadata.num_rows:,}")
