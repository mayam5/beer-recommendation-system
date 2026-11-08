from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "name_match_candidates.parquet"
OUTPUT_PATH = PROCESSED_DIR / "style_mapping_template.csv"

def create_style_mapping_template():
    # Load candidate pairs
    df = pd.read_parquet(
        INPUT_PATH,
        columns=[
            "ba_style",
            "rb_style",
            "is_ambiguous_name",
            "abv_difference"
        ]
    )

    # Count observed style combinations
    style_pairs = (
        df.groupby(
            ["ba_style", "rb_style"],
            dropna=False
        )
        .size()
        .reset_index(name="candidate_count")
    )

    # Count candidates with unambiguous names and close ABV
    strong = df[
        (~df["is_ambiguous_name"]) &
        (df["abv_difference"].round(4) <= 0.3)
    ]

    strong_counts = (
        strong.groupby(
            ["ba_style", "rb_style"],
            dropna=False
        )
        .size()
        .reset_index(name="strong_candidate_count")
    )

    # Combine frequency statistics
    result = style_pairs.merge(
        strong_counts,
        on=["ba_style", "rb_style"],
        how="left"
    )

    result["strong_candidate_count"] = (
        result["strong_candidate_count"]
        .fillna(0)
        .astype(int)
    )

    # Create a column for manually reviewed style relations
    result["style_relation"] = "unreviewed"

    # Preserve exact string matches as a separate category
    exact_match = (
        result["ba_style"].astype("string").str.casefold() ==
        result["rb_style"].astype("string").str.casefold()
    ).fillna(False)

    result.loc[exact_match, "style_relation"] = "equivalent"

    # Review frequent combinations first
    result = result.sort_values(
        ["strong_candidate_count", "candidate_count"],
        ascending=False
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=== Style Mapping Template Summary ===")
    print(f"Unique style combinations: {len(result):,}")
    print(
        "Exact style combinations:",
        f"{exact_match.sum():,}"
    )
    print(
        "Combinations requiring review:",
        f"{(~exact_match).sum():,}"
    )
    print(f"Saved to: {OUTPUT_PATH}")

    print("\n=== Top 15 Combinations ===")
    print(result.head(15).to_string(index=False))

if __name__ == "__main__":
    create_style_mapping_template()

print("\nStep 13 Completed.")
