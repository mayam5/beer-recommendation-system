from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR / "brewer_consistency_candidates.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "provisional_beer_mapping.parquet"
)

def create_provisional_mapping():
    # Load all classified candidate pairs
    df = pd.read_parquet(INPUT_PATH)

    # Select candidates that passed current screening rules
    eligible = df[
        (df["match_status"] == "high_confidence_candidate") &
        (df["brewer_metadata_consistent"])
    ].copy()

    # Ensure one-to-one mapping between platforms
    if eligible["ba_beer_id"].duplicated().any():
        raise ValueError("Duplicate BeerAdvocate beer IDs found.")

    if eligible["rb_beer_id"].duplicated().any():
        raise ValueError("Duplicate RateBeer beer IDs found.")

    # Sort candidates to generate reproducible IDs
    eligible = eligible.sort_values(
        ["normalized_name", "ba_beer_id", "rb_beer_id"]
    ).reset_index(drop=True)

    # Assign provisional cross-platform beer IDs
    eligible["canonical_beer_id"] = [
        f"BEER_{i:06d}"
        for i in range(1, len(eligible) + 1)
    ]

    # Record the provisional verification status
    eligible["mapping_status"] = "provisional"

    # Save only mapping-related columns
    columns = [
        "canonical_beer_id",
        "ba_beer_id",
        "rb_beer_id",
        "normalized_name",
        "ba_beer_name",
        "rb_beer_name",
        "ba_abv",
        "rb_abv",
        "abv_difference",
        "ba_style",
        "rb_style",
        "style_relation",
        "mapping_status"
    ]

    mapping = eligible[columns].copy()

    mapping.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("=== Provisional Beer Mapping Summary ===")
    print(f"Mapped beer pairs: {len(mapping):,}")

    print(
        "Unique canonical IDs:",
        f"{mapping['canonical_beer_id'].nunique():,}"
    )

    print(
        "Unique BeerAdvocate IDs:",
        f"{mapping['ba_beer_id'].nunique():,}"
    )

    print(
        "Unique RateBeer IDs:",
        f"{mapping['rb_beer_id'].nunique():,}"
    )

    print("\n=== Style Relations ===")
    print(mapping["style_relation"].value_counts().to_string())

    print("\n=== Mapping Samples ===")

    print(
        mapping[
            [
                "canonical_beer_id",
                "normalized_name",
                "ba_beer_id",
                "rb_beer_id",
                "mapping_status"
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    create_provisional_mapping()

print("\nStep 17 Completed.")
