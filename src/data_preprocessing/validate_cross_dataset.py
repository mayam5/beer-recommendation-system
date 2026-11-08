from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

TRAIN_PATH = (
    PROCESSED_DIR / "beeradvocate_train_interactions.parquet"
)

INPUT_PATH = (
    PROCESSED_DIR / "ratebeer_test_input.parquet"
)

HOLDOUT_PATH = (
    PROCESSED_DIR / "ratebeer_test_holdout.parquet"
)

USERS_PATH = (
    PROCESSED_DIR / "ratebeer_test_users.parquet"
)

MAPPING_PATH = (
    PROCESSED_DIR / "provisional_beer_mapping.parquet"
)

def validate_cross_dataset():
    # Load training and evaluation data
    train = pd.read_parquet(TRAIN_PATH)
    test_input = pd.read_parquet(INPUT_PATH)
    test_holdout = pd.read_parquet(HOLDOUT_PATH)
    test_users = pd.read_parquet(USERS_PATH)
    mapping = pd.read_parquet(MAPPING_PATH)

    # Collect beer IDs from each dataset
    mapped_beers = set(mapping["canonical_beer_id"])
    train_beers = set(train["canonical_beer_id"])
    input_beers = set(test_input["canonical_beer_id"])
    holdout_beers = set(test_holdout["canonical_beer_id"])

    print("=== Dataset Summary ===")
    print(f"Mapped beers: {len(mapped_beers):,}")
    print(f"Train beers: {len(train_beers):,}")
    print(f"Test input beers: {len(input_beers):,}")
    print(f"Test holdout beers: {len(holdout_beers):,}")

    # Check beer coverage across datasets
    print("\n=== Cross-Dataset Beer Coverage ===")

    print(
        "Input beers found in train:",
        f"{len(input_beers & train_beers):,}",
        f"/ {len(input_beers):,}"
    )

    print(
        "Holdout beers found in train:",
        f"{len(holdout_beers & train_beers):,}",
        f"/ {len(holdout_beers):,}"
    )

    print(
        "Input beers missing from train:",
        f"{len(input_beers - train_beers):,}"
    )

    print(
        "Holdout beers missing from train:",
        f"{len(holdout_beers - train_beers):,}"
    )

    # Analyze training interaction counts per beer
    beer_counts = train.groupby("canonical_beer_id").size()

    print("\n=== Training Beer Sparsity ===")

    for threshold in [1, 2, 5, 10, 20, 50, 100]:
        count = (beer_counts >= threshold).sum()
        print(
            f"Beers with >= {threshold} interactions: "
            f"{count:,}"
        )

    print("\n=== Training Beer Statistics ===")
    print(beer_counts.describe().to_string())

    # Count recommendable beers for each user
    candidate_counts = []

    for user_id, group in test_input.groupby("user_id"):
        seen_beers = set(group["canonical_beer_id"])
        candidates = train_beers - seen_beers

        candidate_counts.append({
            "user_id": user_id,
            "candidate_count": len(candidates)
        })

    candidate_df = pd.DataFrame(candidate_counts)

    print("\n=== Recommendation Candidate Statistics ===")
    print(
        candidate_df["candidate_count"]
        .describe()
        .to_string()
    )

    # Check whether positive holdout beers are in train
    positive_holdout = test_holdout[
        test_holdout["is_positive"]
    ].copy()

    positive_holdout["in_train"] = (
        positive_holdout["canonical_beer_id"]
        .isin(train_beers)
    )

    print("\n=== Positive Holdout Coverage ===")

    print(
        "Total positive holdout interactions:",
        f"{len(positive_holdout):,}"
    )

    print(
        "Positive holdout interactions in train:",
        f"{positive_holdout['in_train'].sum():,}"
    )

    print(
        "Positive holdout interactions missing from train:",
        f"{(~positive_holdout['in_train']).sum():,}"
    )

    # Check evaluation user coverage
    evaluable_users = test_users[
        test_users["evaluable"]
    ]

    print("\n=== Evaluation User Coverage ===")

    print(
        "Evaluable users:",
        f"{len(evaluable_users):,}"
    )

    print(
        "Evaluable users with input:",
        f"{evaluable_users['user_id'].isin(test_input['user_id']).sum():,}"
    )

    print(
        "Evaluable users with holdout:",
        f"{evaluable_users['user_id'].isin(test_holdout['user_id']).sum():,}"
    )

if __name__ == "__main__":
    validate_cross_dataset()
    print("\nStep 22 Completed.")
