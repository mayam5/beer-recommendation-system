from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

MAPPING_PATH = (
    PROCESSED_DIR / "provisional_beer_mapping.parquet"
)

DATASETS = {
    "BeerAdvocate": (
        PROCESSED_DIR / "beeradvocate_quality.parquet",
        "ba_beer_id"
    ),
    "RateBeer": (
        PROCESSED_DIR / "ratebeer_quality.parquet",
        "rb_beer_id"
    )
}

def analyze_rating_distribution(name, path, mapping, id_column):
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 60}")

    # Load required review columns
    df = pd.read_parquet(
        path,
        columns=["beer_id", "user_id", "rating_5"]
    )

    # Prepare the cross-platform beer mapping
    beer_mapping = mapping[
        [id_column, "canonical_beer_id"]
    ].rename(columns={id_column: "beer_id"})

    # Select reviews of provisionally matched beers
    common = df.merge(
        beer_mapping,
        on="beer_id",
        how="inner",
        validate="many_to_one"
    )

    # Keep reviews with valid user IDs and ratings
    valid = common.dropna(
        subset=["user_id", "rating_5"]
    ).copy()

    print("\n=== Rating Statistics ===")
    print(valid["rating_5"].describe().to_string())

    # Count ratings above candidate relevance thresholds
    print("\n=== Positive Rating Thresholds ===")

    for threshold in [3.0, 3.5, 4.0, 4.5]:
        positive = valid["rating_5"] >= threshold

        print(
            f"Rating >= {threshold}: "
            f"{positive.sum():,} "
            f"({positive.mean() * 100:.2f}%)"
        )

    # Count users with enough positive distinct beers
    print("\n=== Users by Positive Beer Count ===")

    for threshold in [3.5, 4.0, 4.5]:
        positive_pairs = valid[
            valid["rating_5"] >= threshold
        ][["user_id", "canonical_beer_id"]].drop_duplicates()

        positive_counts = (
            positive_pairs.groupby("user_id")
            .size()
        )

        print(f"\nRating >= {threshold}")

        for minimum in [1, 2, 3, 5]:
            count = (positive_counts >= minimum).sum()
            print(
                f"Users with >= {minimum} positive beers: "
                f"{count:,}"
            )

def main():
    mapping = pd.read_parquet(MAPPING_PATH)

    for name, (path, id_column) in DATASETS.items():
        analyze_rating_distribution(
            name,
            path,
            mapping,
            id_column
        )

if __name__ == "__main__":
    main()
    print("\nStep 19 Completed.")
