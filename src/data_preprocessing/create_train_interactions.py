from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REVIEWS_PATH = (
    PROCESSED_DIR / "beeradvocate_quality.parquet"
)

MAPPING_PATH = (
    PROCESSED_DIR / "provisional_beer_mapping.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "beeradvocate_train_interactions.parquet"
)

def create_train_interactions():
    # Load BeerAdvocate reviews
    reviews = pd.read_parquet(
        REVIEWS_PATH,
        columns=[
            "user_id",
            "beer_id",
            "rating_5"
        ]
    )

    # Load provisional cross-platform beer mapping
    mapping = pd.read_parquet(
        MAPPING_PATH,
        columns=[
            "ba_beer_id",
            "canonical_beer_id"
        ]
    )

    # Rename BeerAdvocate beer ID for merging
    mapping = mapping.rename(
        columns={"ba_beer_id": "beer_id"}
    )

    # Keep reviews of provisionally matched beers
    common_reviews = reviews.merge(
        mapping,
        on="beer_id",
        how="inner",
        validate="many_to_one"
    )

    # Remove reviews without valid user IDs or ratings
    valid_reviews = common_reviews.dropna(
        subset=["user_id", "rating_5"]
    ).copy()

    # Aggregate repeated ratings for each user-beer pair
    interactions = (
        valid_reviews.groupby(
            ["user_id", "canonical_beer_id"],
            as_index=False
        )
        .agg(
            rating=("rating_5", "mean"),
            review_count=("rating_5", "size")
        )
    )

    # Check that each user-beer pair is unique
    duplicate_count = interactions.duplicated(
        subset=["user_id", "canonical_beer_id"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            "Duplicate user-beer pairs remain after aggregation."
        )

    # Sort rows for reproducible output
    interactions = interactions.sort_values(
        ["user_id", "canonical_beer_id"]
    ).reset_index(drop=True)

    # Save the training interaction table
    interactions.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("=== BeerAdvocate Train Interaction Summary ===")

    print(f"Original reviews: {len(reviews):,}")
    print(f"Common beer reviews: {len(common_reviews):,}")
    print(f"Valid reviews: {len(valid_reviews):,}")
    print(f"Training interactions: {len(interactions):,}")

    print(
        "Repeated review rows merged:",
        f"{len(valid_reviews) - len(interactions):,}"
    )

    print(
        "Unique users:",
        f"{interactions['user_id'].nunique():,}"
    )

    print(
        "Unique common beers:",
        f"{interactions['canonical_beer_id'].nunique():,}"
    )

    print("\n=== Rating Statistics ===")
    print(interactions["rating"].describe().to_string())

    print("\n=== Review Count per Interaction ===")
    print(
        interactions["review_count"]
        .value_counts()
        .sort_index()
        .head(10)
        .to_string()
    )

    print("\n=== Sample Interactions ===")
    print(interactions.head(10).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_train_interactions()
    print("\nStep 20 Completed.")
