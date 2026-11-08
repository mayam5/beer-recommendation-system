from pathlib import Path
import sqlite3
import tempfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DATASETS = {
    "BeerAdvocate": {
        "input": PROCESSED_DIR / "beeradvocate_normalized.parquet",
        "output": PROCESSED_DIR / "beeradvocate_quality.parquet",
    },
    "RateBeer": {
        "input": PROCESSED_DIR / "ratebeer_normalized.parquet",
        "output": PROCESSED_DIR / "ratebeer_quality.parquet",
    },
}

BATCH_SIZE = 50000

def has_value(series):
    # Check whether values are non-null and non-empty
    return series.notna() & series.astype("string").str.strip().ne("").fillna(False)

def create_quality_database(conn):
    # Store unique user-beer pairs and their occurrence counts
    conn.execute("""
        CREATE TABLE pair_counts (
            user_id TEXT NOT NULL,
            beer_id TEXT NOT NULL,
            review_count INTEGER NOT NULL,
            PRIMARY KEY (user_id, beer_id)
        )
    """)

    # Store distinct beer-brewer combinations
    conn.execute("""
        CREATE TABLE beer_brewers (
            beer_id TEXT NOT NULL,
            brewer_id TEXT NOT NULL,
            PRIMARY KEY (beer_id, brewer_id)
        )
    """)

    conn.commit()

def build_quality_statistics(input_path, conn):
    # Read only the columns required for duplicate analysis
    parquet_file = pq.ParquetFile(input_path)

    total = 0

    for batch in parquet_file.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["user_id", "beer_id", "brewer_id"]
    ):
        df = batch.to_pandas()

        # Exclude missing IDs from pair statistics
        valid_pairs = df[
            has_value(df["user_id"]) &
            has_value(df["beer_id"])
        ]

        pair_counts = (
            valid_pairs
            .groupby(["user_id", "beer_id"])
            .size()
            .reset_index(name="review_count")
        )

        conn.executemany(
            """
            INSERT INTO pair_counts (
                user_id, beer_id, review_count
            )
            VALUES (?, ?, ?)
            ON CONFLICT(user_id, beer_id)
            DO UPDATE SET
                review_count = review_count + excluded.review_count
            """,
            pair_counts.itertuples(index=False, name=None)
        )

        # Keep distinct beer-brewer combinations
        valid_brewers = df[
            has_value(df["beer_id"]) &
            has_value(df["brewer_id"])
        ][["beer_id", "brewer_id"]].drop_duplicates()

        conn.executemany(
            """
            INSERT OR IGNORE INTO beer_brewers (
                beer_id, brewer_id
            )
            VALUES (?, ?)
            """,
            valid_brewers.itertuples(index=False, name=None)
        )

        conn.commit()

        total += len(df)

        if total % 500000 == 0:
            print(f"Statistics: {total:,} reviews processed...")

    # Build compact lookup tables for repeated pairs
    conn.execute("""
        CREATE TABLE repeated_pairs AS
        SELECT user_id, beer_id
        FROM pair_counts
        WHERE review_count > 1
    """)

    conn.execute("""
        CREATE UNIQUE INDEX idx_repeated_pairs
        ON repeated_pairs(user_id, beer_id)
    """)

    # Identify beers associated with multiple brewer IDs
    conn.execute("""
        CREATE TABLE conflicting_beers AS
        SELECT beer_id
        FROM beer_brewers
        GROUP BY beer_id
        HAVING COUNT(*) > 1
    """)

    conn.execute("""
        CREATE UNIQUE INDEX idx_conflicting_beers
        ON conflicting_beers(beer_id)
    """)

    conn.commit()

def add_quality_flags(input_path, output_path, conn):
    parquet_file = pq.ParquetFile(input_path)

    flag_names = [
        "has_valid_user",
        "has_valid_beer",
        "has_valid_rating",
        "has_review_text",
        "is_repeated_pair",
        "has_brewer_conflict",
    ]

    output_schema = parquet_file.schema_arrow

    for flag in flag_names:
        output_schema = output_schema.append(
            pa.field(flag, pa.bool_())
        )

    # Use SQLite lookups without loading all review records
    repeated_pairs = set(
        conn.execute(
            "SELECT user_id, beer_id FROM repeated_pairs"
        ).fetchall()
    )

    conflicting_beers = set(
        row[0]
        for row in conn.execute(
            "SELECT beer_id FROM conflicting_beers"
        )
    )

    total = 0
    flag_counts = {flag: 0 for flag in flag_names}

    writer = None

    try:
        writer = pq.ParquetWriter(
            output_path,
            output_schema,
            compression="snappy"
        )

        for batch in parquet_file.iter_batches(
            batch_size=BATCH_SIZE
        ):
            df = batch.to_pandas()

            # Add missing-value flags
            df["has_valid_user"] = has_value(df["user_id"])
            df["has_valid_beer"] = has_value(df["beer_id"])
            df["has_valid_rating"] = (
                pd.to_numeric(
                    df["rating_5"],
                    errors="coerce"
                ).notna()
            )
            df["has_review_text"] = has_value(df["review_text"])

            # Mark all reviews belonging to repeated pairs
            pairs = zip(df["user_id"], df["beer_id"])

            df["is_repeated_pair"] = [
                (user_id, beer_id) in repeated_pairs
                if pd.notna(user_id) and pd.notna(beer_id)
                else False
                for user_id, beer_id in pairs
            ]

            # Mark beers associated with multiple brewer IDs
            df["has_brewer_conflict"] = (
                df["beer_id"]
                .isin(conflicting_beers)
            )

            for flag in flag_names:
                flag_counts[flag] += int(df[flag].sum())

            table = pa.Table.from_pandas(
                df,
                schema=output_schema,
                preserve_index=False
            )

            writer.write_table(table)

            total += len(df)

            if total % 500000 == 0:
                print(f"Writing: {total:,} reviews processed...")

    finally:
        if writer is not None:
            writer.close()

    print("\n=== Quality Summary ===")
    print(f"Total reviews: {total:,}")

    for flag, count in flag_counts.items():
        print(f"{flag}: {count:,}")

    print(f"Saved to: {output_path}")

def process_dataset(name, paths):
    print(f"\n{'=' * 60}")
    print(f"Processing: {name}")
    print(f"{'=' * 60}")

    # Automatically delete the temporary database afterward
    with tempfile.TemporaryDirectory() as temp_dir:
        db_path = Path(temp_dir) / "quality_stats.sqlite"

        conn = sqlite3.connect(db_path)

        try:
            create_quality_database(conn)

            print("\nBuilding quality statistics...")
            build_quality_statistics(
                paths["input"], conn
            )

            print("\nAdding quality flags...")
            add_quality_flags(
                paths["input"],
                paths["output"],
                conn
            )

        finally:
            conn.close()

    print(f"{name} completed.")

for dataset_name, paths in DATASETS.items():
    process_dataset(dataset_name, paths)

print("\nStep 8 Completed.")
