from pathlib import Path
import sqlite3
import tempfile
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATASETS = {
    "BeerAdvocate": {
        "input": PROCESSED_DIR / "beeradvocate_quality.parquet",
        "output": PROCESSED_DIR / "beeradvocate_beers.parquet",
    },
    "RateBeer": {
        "input": PROCESSED_DIR / "ratebeer_quality.parquet",
        "output": PROCESSED_DIR / "ratebeer_beers.parquet",
    },
}

BATCH_SIZE = 50000

def normalize_name(value):
    # Normalize Unicode, letter case, and whitespace
    if value is None or pd.isna(value):
        return None

    value = unicodedata.normalize("NFKC", str(value))
    value = " ".join(value.casefold().split())

    return value if value else None

def clean_value(value):
    # Preserve meaningful text and convert blanks to null
    if value is None or pd.isna(value):
        return None

    value = str(value).strip()
    return value if value else None

def create_database(conn):
    # Store occurrence counts for beer metadata values
    conn.execute("""
        CREATE TABLE metadata_counts (
            beer_id TEXT NOT NULL,
            field_name TEXT NOT NULL,
            field_value TEXT NOT NULL,
            occurrence_count INTEGER NOT NULL,
            PRIMARY KEY (beer_id, field_name, field_value)
        )
    """)

    # Store review counts for each beer
    conn.execute("""
        CREATE TABLE beer_counts (
            beer_id TEXT PRIMARY KEY,
            review_count INTEGER NOT NULL
        )
    """)

    conn.commit()

def build_metadata(input_path, conn):
    parquet_file = pq.ParquetFile(input_path)

    columns = [
        "beer_id",
        "beer_name",
        "abv",
        "beer_style",
        "brewer_id",
    ]

    total = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=columns
    ):
        df = batch.to_pandas()

        # Exclude records without a beer ID
        df = df[df["beer_id"].notna()].copy()
        df = df[df["beer_id"].astype("string").str.strip().ne("")]

        # Count reviews for each beer
        counts = df.groupby("beer_id").size()

        conn.executemany(
            """
            INSERT INTO beer_counts (beer_id, review_count)
            VALUES (?, ?)
            ON CONFLICT(beer_id)
            DO UPDATE SET
                review_count = review_count + excluded.review_count
            """,
            [
                (str(beer_id), int(count))
                for beer_id, count in counts.items()
            ]
        )

        # Count each observed metadata value
        for field in [
            "beer_name",
            "abv",
            "beer_style",
            "brewer_id",
        ]:
            values = df[["beer_id", field]].copy()

            values[field] = values[field].map(clean_value)
            values = values.dropna(subset=[field])

            field_counts = (
                values.groupby(["beer_id", field])
                .size()
            )

            rows = [
                (str(beer_id), field, str(value), int(count))
                for (beer_id, value), count
                in field_counts.items()
            ]

            conn.executemany(
                """
                INSERT INTO metadata_counts (
                    beer_id, field_name,
                    field_value, occurrence_count
                )
                VALUES (?, ?, ?, ?)
                ON CONFLICT(beer_id, field_name, field_value)
                DO UPDATE SET
                    occurrence_count =
                    occurrence_count + excluded.occurrence_count
                """,
                rows
            )

        conn.commit()
        total += len(df)

        if total % 500000 < BATCH_SIZE:
            print(f"Processed approximately {total:,} reviews...")

    return total

def export_beer_metadata(source, output_path, conn):
    # Read aggregated counts, not individual reviews
    counts_df = pd.read_sql_query(
        "SELECT beer_id, review_count FROM beer_counts",
        conn
    )

    metadata_df = pd.read_sql_query(
        """
        SELECT
            beer_id,
            field_name,
            field_value,
            occurrence_count
        FROM metadata_counts
        """,
        conn
    )

    # Select the most frequent value for each field
    metadata_df = metadata_df.sort_values(
        ["beer_id", "field_name",
         "occurrence_count", "field_value"],
        ascending=[True, True, False, True]
    )

    representative = metadata_df.drop_duplicates(
        subset=["beer_id", "field_name"]
    )

    wide_df = representative.pivot(
        index="beer_id",
        columns="field_name",
        values="field_value"
    ).reset_index()

    # Count distinct values per field
    distinct_counts = (
        metadata_df.groupby(["beer_id", "field_name"])
        .size()
        .unstack(fill_value=0)
        .add_suffix("_value_count")
        .reset_index()
    )

    result = counts_df.merge(
        wide_df, on="beer_id", how="left"
    ).merge(
        distinct_counts, on="beer_id", how="left"
    )

    result["source"] = source

    result = result.rename(columns={
        "brewer_id_value_count": "brewer_id_count"
    })

    # Convert ABV back to numeric format
    result["abv"] = pd.to_numeric(
        result["abv"], errors="coerce"
    )

    # Create conservative normalized beer names
    result["normalized_name"] = (
        result["beer_name"].map(normalize_name)
    )

    # Keep important columns first
    first_columns = [
        "source",
        "beer_id",
        "beer_name",
        "normalized_name",
        "abv",
        "beer_style",
        "review_count",
        "brewer_id_count",
    ]

    other_columns = [
        col for col in result.columns
        if col not in first_columns
    ]

    result = result[first_columns + other_columns]

    result.to_parquet(
        output_path,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("\n=== Metadata Summary ===")
    print(f"Unique beers: {len(result):,}")
    print(
        "Missing normalized names:",
        f"{result['normalized_name'].isna().sum():,}"
    )
    print(
        "Beers with multiple brewer IDs:",
        f"{(result['brewer_id_count'] > 1).sum():,}"
    )
    print(f"Saved to: {output_path}")

def process_dataset(source, paths):
    print(f"\n{'=' * 60}")
    print(f"Processing: {source}")
    print(f"{'=' * 60}")

    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "beer_metadata.sqlite"
        conn = sqlite3.connect(db_path)

        try:
            create_database(conn)

            build_metadata(
                paths["input"], conn
            )

            export_beer_metadata(
                source, paths["output"], conn
            )

        finally:
            conn.close()

for source, paths in DATASETS.items():
    process_dataset(source, paths)

print("\nStep 9 Completed.")
