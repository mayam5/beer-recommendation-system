from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CANDIDATES_PATH = (
    PROCESSED_DIR / "classified_match_candidates.parquet"
)

BA_BEERS_PATH = (
    PROCESSED_DIR / "beeradvocate_beers.parquet"
)

RB_BEERS_PATH = (
    PROCESSED_DIR / "ratebeer_beers.parquet"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "brewer_consistency_candidates.parquet"
)

def check_brewer_consistency():
    # Load classified candidate pairs
    df = pd.read_parquet(CANDIDATES_PATH)

    # Brewer metadata is already included in the candidate file
    required_columns = [
        "ba_brewer_id_count",
        "rb_brewer_id_count"
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    # Detect missing or conflicting brewer IDs
    df["has_missing_brewer"] = (
        df["ba_brewer_id_count"].isna() |
        df["rb_brewer_id_count"].isna() |
        df["ba_brewer_id_count"].eq(0) |
        df["rb_brewer_id_count"].eq(0)
    )

    df["has_brewer_conflict"] = (
        df["ba_brewer_id_count"].gt(1) |
        df["rb_brewer_id_count"].gt(1)
    )

    # Flag candidates with internally consistent brewer IDs
    df["brewer_metadata_consistent"] = (
        ~df["has_missing_brewer"] &
        ~df["has_brewer_conflict"]
    )

    # Save results without removing any candidates
    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("=== Brewer Consistency Summary ===")
    print(f"Total candidates: {len(df):,}")

    for status in [
        "high_confidence_candidate",
        "review_required"
    ]:
        subset = df[df["match_status"] == status]

        print(f"\n=== {status} ===")
        print(f"Candidates: {len(subset):,}")
        print(
            "Consistent brewer metadata:",
            f"{subset['brewer_metadata_consistent'].sum():,}"
        )
        print(
            "Missing brewer metadata:",
            f"{subset['has_missing_brewer'].sum():,}"
        )
        print(
            "Brewer ID conflicts:",
            f"{subset['has_brewer_conflict'].sum():,}"
        )

    print("\n=== High-Confidence Candidates with Conflicts ===")

    conflicts = df[
        (df["match_status"] == "high_confidence_candidate") &
        (~df["brewer_metadata_consistent"])
    ]

    columns = [
        "normalized_name",
        "ba_beer_id",
        "rb_beer_id",
        "ba_brewer_id_count",
        "rb_brewer_id_count",
        "has_missing_brewer",
        "has_brewer_conflict"
    ]

    print(conflicts[columns].head(15).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    check_brewer_consistency()

print("\nStep 16 Completed.")
