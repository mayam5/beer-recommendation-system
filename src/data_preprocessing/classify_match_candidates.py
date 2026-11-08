from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CANDIDATES_PATH = (
    PROCESSED_DIR / "name_match_candidates.parquet"
)

STYLE_MAPPING_PATH = (
    PROCESSED_DIR / "style_mapping_reviewed.csv"
)

OUTPUT_PATH = (
    PROCESSED_DIR / "classified_match_candidates.parquet"
)

def classify_candidates():
    # Load candidate pairs and style mapping
    candidates = pd.read_parquet(CANDIDATES_PATH)
    style_mapping = pd.read_csv(STYLE_MAPPING_PATH)

    # Keep only the columns required for joining
    style_mapping = style_mapping[
        ["ba_style", "rb_style", "style_relation"]
    ]

    # Ensure each style pair has only one classification
    if style_mapping.duplicated(
        ["ba_style", "rb_style"]
    ).any():
        raise ValueError("Duplicate style mapping keys found.")

    # Attach style relations to candidate pairs
    df = candidates.merge(
        style_mapping,
        on=["ba_style", "rb_style"],
        how="left",
        validate="many_to_one"
    )

    # Keep unmatched style mappings as unreviewed
    df["style_relation"] = (
        df["style_relation"].fillna("unreviewed")
    )

    # Check whether the candidate has a close ABV
    df["abv_close"] = (
        df["abv_difference"].round(4) <= 0.3
    ).fillna(False)

    # Check whether styles have a reviewed relationship
    df["style_supported"] = df[
        "style_relation"
    ].isin(["equivalent", "compatible"])

    # Identify high-confidence candidates
    high_confidence = (
        (~df["is_ambiguous_name"]) &
        df["abv_close"] &
        df["style_supported"]
    )

    df["match_status"] = "review_required"

    df.loc[
        high_confidence,
        "match_status"
    ] = "high_confidence_candidate"

    # Ensure that no candidate pairs were lost
    if len(df) != len(candidates):
        raise ValueError("Candidate count changed during merging.")

    # Save classified candidates
    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("=== Match Classification Summary ===")
    print(f"Total candidates: {len(df):,}")

    print("\n=== Match Status ===")
    print(df["match_status"].value_counts().to_string())

    print("\n=== High-Confidence Candidates by Style ===")

    high = df[
        df["match_status"] == "high_confidence_candidate"
    ]

    print(
        high["style_relation"]
        .value_counts()
        .to_string()
    )

    print("\n=== Unique Beers in High-Confidence Candidates ===")
    print(
        "BeerAdvocate:",
        f"{high['ba_beer_id'].nunique():,}"
    )
    print(
        "RateBeer:",
        f"{high['rb_beer_id'].nunique():,}"
    )

    print("\n=== Sample High-Confidence Candidates ===")

    columns = [
        "normalized_name",
        "ba_beer_id",
        "rb_beer_id",
        "ba_abv",
        "rb_abv",
        "style_relation",
        "match_status"
    ]

    print(high[columns].head(10).to_string(index=False))

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    classify_candidates()

print("\nStep 15 Completed.")
