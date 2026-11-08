from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATASETS = {
    "BeerAdvocate": {
        "input": PROCESSED_DIR / "beeradvocate_standardized.parquet",
        "output": PROCESSED_DIR / "beeradvocate_normalized.parquet"
    },
    "RateBeer": {
        "input": PROCESSED_DIR / "ratebeer_standardized.parquet",
        "output": PROCESSED_DIR / "ratebeer_normalized.parquet"
    }
}

BATCH_SIZE = 50000

def normalize_rating(value, source):
    # Return missing values without modification
    if value is None or pd.isna(value):
        return None

    try:
        if source == "BeerAdvocate":
            rating = float(value)

        elif source == "RateBeer":
            numerator, denominator = str(value).split("/")
            denominator = float(denominator)

            if denominator == 0:
                return None

            rating = float(numerator) / denominator * 5.0

        else:
            return None

        # Reject values outside the expected rating range
        if not 0 <= rating <= 5:
            return None

        return rating

    except (ValueError, TypeError):
        return None

def normalize_dataset(name, input_path, output_path):
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")

    parquet_file = pq.ParquetFile(input_path)

    # Preserve the existing schema and add a normalized rating
    output_schema = parquet_file.schema_arrow.append(
        pa.field("rating_5", pa.float64())
    )

    writer = None
    total = 0
    invalid_count = 0

    try:
        writer = pq.ParquetWriter(
            output_path,
            output_schema,
            compression="snappy"
        )

        # Process the dataset in batches
        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):
            df = batch.to_pandas()

            # Convert original ratings to a common 5-point scale
            df["rating_5"] = df["rating_raw"].apply(
                lambda value: normalize_rating(value, name)
            )

            invalid_count += df["rating_5"].isna().sum()
            total += len(df)

            # Write the normalized batch
            table = pa.Table.from_pandas(
                df,
                schema=output_schema,
                preserve_index=False
            )

            writer.write_table(table)

            if total % 500000 == 0:
                print(f"Processed {total:,} reviews...")

    finally:
        if writer is not None:
            writer.close()

    print("\n=== Summary ===")
    print(f"Total reviews: {total:,}")
    print(f"Missing or invalid ratings: {invalid_count:,}")
    print(f"Saved to: {output_path}")
    print(f"{name} completed.")

# Normalize both datasets
for dataset_name, paths in DATASETS.items():
    normalize_dataset(
        dataset_name,
        paths["input"],
        paths["output"]
    )

print("\nStep 7 Completed.")
