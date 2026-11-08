from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "name_match_candidates.parquet"

def analyze_candidates():
    df = pd.read_parquet(INPUT_PATH)

    print("=== ABV Difference Distribution ===")

    difference = df["abv_difference"]

    categories = {
        "Exact match (0.0)": difference.eq(0),
        "0.0 < diff <= 0.3": difference.gt(0) & difference.le(0.3),
        "0.3 < diff <= 0.5": difference.gt(0.3) & difference.le(0.5),
        "0.5 < diff <= 1.0": difference.gt(0.5) & difference.le(1.0),
        "Diff > 1.0": difference.gt(1.0),
        "Missing ABV": difference.isna(),
    }

    for name, condition in categories.items():
        print(f"{name}: {int(condition.sum()):,}")

    print("\n=== Ambiguity and ABV ===")

    for label, subset in [
        ("Unambiguous", df[~df["is_ambiguous_name"]]),
        ("Ambiguous", df[df["is_ambiguous_name"]]),
    ]:
        print(f"\n{label} candidates: {len(subset):,}")
        print(
            "Both ABVs available:",
            f"{subset['abv_difference'].notna().sum():,}"
        )
        print(
            "ABV difference <= 0.3:",
            f"{subset['abv_difference'].le(0.3).sum():,}"
        )

    print("\n=== Largest ABV Differences ===")

    columns = [
        "normalized_name",
        "ba_beer_id",
        "rb_beer_id",
        "ba_abv",
        "rb_abv",
        "abv_difference",
        "is_ambiguous_name",
    ]

    print(
        df.dropna(subset=["abv_difference"])
        .sort_values("abv_difference", ascending=False)
        [columns]
        .head(15)
        .to_string(index=False)
    )

    print("\n=== ABV Differences in Sample Candidates ===")

    print(
        df[
            df["abv_difference"].gt(0.5) &
            ~df["is_ambiguous_name"]
        ][columns]
        .head(15)
        .to_string(index=False)
    )

if __name__ == "__main__":
    analyze_candidates()

print("\nStep 11 Completed.")
