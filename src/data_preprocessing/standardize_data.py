import gzip
import json
import ast
import html
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Define input and output files
DATASETS = {
    "BeerAdvocate": {
        "input": RAW_DIR / "beeradvocate.json.gz",
        "output": PROCESSED_DIR / "beeradvocate_standardized.parquet"
    },
    "RateBeer": {
        "input": RAW_DIR / "ratebeer.json.gz",
        "output": PROCESSED_DIR / "ratebeer_standardized.parquet"
    }
}

# Map original column names to standardized names
COLUMN_MAPPING = {
    "beer/name": "beer_name",
    "beer/beerId": "beer_id",
    "beer/brewerId": "brewer_id",
    "beer/ABV": "abv",
    "beer/style": "beer_style",
    "review/profileName": "user_id",
    "review/overall": "rating_raw",
    "review/text": "review_text",
    "review/time": "review_time",
    "review/appearance": "appearance_raw",
    "review/aroma": "aroma_raw",
    "review/palate": "palate_raw",
    "review/taste": "taste_raw"
}

# Define a consistent output schema
SCHEMA = pa.schema([
    ("source", pa.string()),
    ("source_row_id", pa.int64()),
    ("beer_name", pa.string()),
    ("beer_id", pa.string()),
    ("brewer_id", pa.string()),
    ("abv", pa.float64()),
    ("beer_style", pa.string()),
    ("user_id", pa.string()),
    ("rating_raw", pa.string()),
    ("review_text", pa.string()),
    ("review_time", pa.int64()),
    ("appearance_raw", pa.string()),
    ("aroma_raw", pa.string()),
    ("palate_raw", pa.string()),
    ("taste_raw", pa.string())
])

CHUNK_SIZE = 50000

def parse_record(line):
    # Parse JSON or Python-literal formatted records
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)

def clean_string(value):
    # Convert empty strings to missing values
    if value is None:
        return None

    value = str(value).strip()
    return value if value else None

def convert_float(value):
    # Convert numeric strings to floating-point values
    value = clean_string(value)

    if value is None:
        return None

    try:
        return float(value)
    except ValueError:
        return None

def convert_integer(value):
    # Convert numeric strings to integers
    value = clean_string(value)

    if value is None:
        return None

    try:
        return int(value)
    except ValueError:
        return None

def standardize_record(record, source, row_id):
    # Standardize column names
    result = {
        new_name: record.get(old_name)
        for old_name, new_name in COLUMN_MAPPING.items()
    }

    # Add source information and original row number
    result["source"] = source
    result["source_row_id"] = row_id

    # Standardize string columns
    string_columns = [
        "beer_name", "beer_id", "brewer_id",
        "beer_style", "user_id", "rating_raw",
        "review_text", "appearance_raw",
        "aroma_raw", "palate_raw", "taste_raw"
    ]

    for column in string_columns:
        result[column] = clean_string(result[column])

    # Decode HTML entities in beer metadata
    for column in ["beer_name", "beer_style"]:
        if result[column] is not None:
            result[column] = html.unescape(result[column])

    # Convert numeric columns
    result["abv"] = convert_float(result["abv"])
    result["review_time"] = convert_integer(
        result["review_time"]
    )

    return result

def write_chunk(records, writer):
    # Convert a chunk to the predefined Parquet schema
    df = pd.DataFrame(records)
    table = pa.Table.from_pandas(
        df,
        schema=SCHEMA,
        preserve_index=False
    )

    writer.write_table(table)

def standardize_dataset(name, input_path, output_path):
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")

    records = []
    total = 0
    writer = None

    try:
        # Stream records from the compressed source file
        with gzip.open(
            input_path, "rt", encoding="utf-8"
        ) as f:
            for line in f:
                if not line.strip():
                    continue

                record = parse_record(line)
                total += 1

                standardized = standardize_record(
                    record, name, total
                )
                records.append(standardized)

                # Write records in fixed-size chunks
                if len(records) >= CHUNK_SIZE:
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_path,
                            SCHEMA,
                            compression="snappy"
                        )

                    write_chunk(records, writer)
                    records.clear()

                    print(f"Processed {total:,} reviews...")

        # Write any remaining records
        if records:
            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    SCHEMA,
                    compression="snappy"
                )

            write_chunk(records, writer)

    finally:
        if writer is not None:
            writer.close()

    print(f"\nTotal processed: {total:,}")
    print(f"Saved to: {output_path}")
    print(f"{name} completed.")

# Standardize both datasets
for dataset_name, paths in DATASETS.items():
    standardize_dataset(
        dataset_name,
        paths["input"],
        paths["output"]
    )

print("\nStep 6 Completed.")
