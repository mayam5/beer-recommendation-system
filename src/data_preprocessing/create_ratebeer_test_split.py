from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

REVIEWS_PATH = PROCESSED_DIR / "ratebeer_quality.parquet"
MAPPING_PATH = PROCESSED_DIR / "provisional_beer_mapping.parquet"

INPUT_PATH = PROCESSED_DIR / "ratebeer_test_input.parquet"
HOLDOUT_PATH = PROCESSED_DIR / "ratebeer_test_holdout.parquet"
USERS_PATH = PROCESSED_DIR / "ratebeer_test_users.parquet"

MIN_BEERS = 10
INPUT_RATIO = 0.7
POSITIVE_THRESHOLD = 3.5
RANDOM_SEED = 42

def create_test_split():
    # Load RateBeer reviews and provisional beer mapping
    reviews = pd.read_parquet(
        REVIEWS_PATH,
        columns=["user_id", "beer_id", "rating_5"]
    )

    mapping = pd.read_parquet(
        MAPPING_PATH,
        columns=["rb_beer_id", "canonical_beer_id"]
    ).rename(columns={"rb_beer_id": "beer_id"})

    # Keep valid reviews of common beers
    common = reviews.merge(
        mapping,
        on="beer_id",
        how="inner",
        validate="many_to_one"
    ).dropna(subset=["user_id", "rating_5"])

    # Aggregate repeated ratings if any exist
    interactions = (
        common.groupby(
            ["user_id", "canonical_beer_id"],
            as_index=False
        )
        .agg(rating=("rating_5", "mean"))
    )

    # Select users with enough distinct rated beers
    counts = interactions.groupby("user_id").size()
    eligible_ids = counts[counts >= MIN_BEERS].index

    eligible = interactions[
        interactions["user_id"].isin(eligible_ids)
    ].copy()

    # Sort before random splitting for reproducibility
    eligible = eligible.sort_values(
        ["user_id", "canonical_beer_id"]
    ).reset_index(drop=True)

    rng = np.random.default_rng(RANDOM_SEED)

    input_parts = []
    holdout_parts = []

    # Split each user's interactions independently
    for _, group in eligible.groupby("user_id", sort=True):
        indices = rng.permutation(len(group))
        input_size = int(np.floor(len(group) * INPUT_RATIO))

        input_parts.append(group.iloc[indices[:input_size]])
        holdout_parts.append(group.iloc[indices[input_size:]])

    input_df = pd.concat(input_parts, ignore_index=True)
    holdout_df = pd.concat(holdout_parts, ignore_index=True)

    # Label positive held-out interactions
    holdout_df["is_positive"] = (
        holdout_df["rating"] >= POSITIVE_THRESHOLD
    )

    # Identify users with at least one positive held-out beer
    user_summary = (
        holdout_df.groupby("user_id")
        .agg(
            holdout_count=("rating", "size"),
            positive_holdout_count=("is_positive", "sum")
        )
        .reset_index()
    )

    user_summary["evaluable"] = (
        user_summary["positive_holdout_count"] >= 1
    )

    # Verify that input and holdout are disjoint
    input_pairs = pd.MultiIndex.from_frame(
        input_df[["user_id", "canonical_beer_id"]]
    )
    holdout_pairs = pd.MultiIndex.from_frame(
        holdout_df[["user_id", "canonical_beer_id"]]
    )

    if len(input_pairs.intersection(holdout_pairs)) != 0:
        raise ValueError("Input and holdout overlap.")

    if len(input_df) + len(holdout_df) != len(eligible):
        raise ValueError("Split interaction counts do not match.")

    # Save separate evaluation files
    input_df.to_parquet(INPUT_PATH, index=False)
    holdout_df.to_parquet(HOLDOUT_PATH, index=False)
    user_summary.to_parquet(USERS_PATH, index=False)

    print("=== RateBeer Test Split Summary ===")
    print(f"Users with >= {MIN_BEERS} beers: {len(eligible_ids):,}")
    print(f"Total interactions: {len(eligible):,}")
    print(f"Input interactions: {len(input_df):,}")
    print(f"Holdout interactions: {len(holdout_df):,}")

    print("\n=== Evaluation Eligibility ===")
    print(
        "Users with positive holdout:",
        f"{user_summary['evaluable'].sum():,}"
    )
    print(
        "Users without positive holdout:",
        f"{(~user_summary['evaluable']).sum():,}"
    )
    print(
        "Positive holdout interactions:",
        f"{holdout_df['is_positive'].sum():,}"
    )

    print("\n=== Split Validation ===")
    print("Overlapping user-beer pairs: 0")
    print("All eligible interactions preserved: True")

    print("\nSaved files:")
    print(INPUT_PATH)
    print(HOLDOUT_PATH)
    print(USERS_PATH)

if __name__ == "__main__":
    create_test_split()
    print("\nStep 21 Completed.")
    
