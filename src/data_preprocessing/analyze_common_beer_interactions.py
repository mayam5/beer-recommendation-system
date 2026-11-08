from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MAPPING_PATH = (
    PROCESSED_DIR / "provisional_beer_mapping.parquet"
)

DATASETS = {
    "BeerAdvocate": {
        "path": PROCESSED_DIR / "beeradvocate_quality.parquet",
        "mapping_column": "ba_beer_id"
    },
    "RateBeer": {
        "path": PROCESSED_DIR / "ratebeer_quality.parquet",
        "mapping_column": "rb_beer_id"
    }
}

def analyze_dataset(name, config, mapping):
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 60}")

    # Load only columns needed for interaction analysis
    df = pd.read_parquet(
        config["path"],
        columns=[
            "beer_id",
            "user_id",
            "rating_5"
        ]
    )

    # Prepare platform-specific mapping
    beer_mapping = mapping[
        [config["mapping_column"], "canonical_beer_id"]
    ].rename(
        columns={
            config["mapping_column"]: "beer_id"
        }
    )

    # Keep reviews belonging to provisionally matched beers
    common = df.merge(
        beer_mapping,
        on="beer_id",
        how="inner",
        validate="many_to_one"
    )

    # Count valid interactions
    valid = common.dropna(
        subset=["user_id", "rating_5"]
    ).copy()

    # Count distinct user-beer interactions
    unique_pairs = valid[
        ["user_id", "canonical_beer_id"]
    ].drop_duplicates()

    # Calculate user interaction counts
    user_counts = (
        unique_pairs.groupby("user_id")
        .size()
    )

    # Calculate beer interaction counts
    beer_counts = (
        unique_pairs.groupby("canonical_beer_id")
        .size()
    )

    print("\n=== Common Beer Interaction Summary ===")
    print(f"Total reviews in dataset: {len(df):,}")
    print(f"Reviews of common beers: {len(common):,}")
    print(f"Valid reviews of common beers: {len(valid):,}")

    print(
        "Unique users:",
        f"{valid['user_id'].nunique():,}"
    )

    print(
        "Common beers with valid reviews:",
        f"{valid['canonical_beer_id'].nunique():,}"
    )

    print(
        "Unique user-beer pairs:",
        f"{len(unique_pairs):,}"
    )

    print("\n=== Users by Minimum Distinct Beer Count ===")

    for threshold in [1, 2, 3, 5, 10, 20]:
        count = (user_counts >= threshold).sum()
        print(f"Users with >= {threshold} beers: {count:,}")

    print("\n=== User Interaction Statistics ===")

    if not user_counts.empty:
        print(user_counts.describe().to_string())

    print("\n=== Beer Interaction Statistics ===")

    if not beer_counts.empty:
        print(beer_counts.describe().to_string())

    print(f"\n{name} analysis completed.")

def main():
    # Load provisional cross-platform mapping
    mapping = pd.read_parquet(MAPPING_PATH)

    print("=== Provisional Mapping ===")
    print(f"Mapped beer pairs: {len(mapping):,}")

    for name, config in DATASETS.items():
        analyze_dataset(name, config, mapping)

if __name__ == "__main__":
    main()
    print("\nStep 18 Completed.")
