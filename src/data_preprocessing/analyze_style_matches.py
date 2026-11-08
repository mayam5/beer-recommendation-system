from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "name_match_candidates.parquet"

def analyze_styles():
    df = pd.read_parquet(INPUT_PATH)

    print("=== Style Availability ===")
    print(f"Total candidates: {len(df):,}")

    both_styles = (
        df["ba_style"].notna() &
        df["rb_style"].notna()
    )

    print(f"Both styles available: {both_styles.sum():,}")
    print(
        "Exact style matches:",
        f"{df['style_exact_match'].sum():,}"
    )

    # Examine common style combinations
    style_pairs = (
        df.loc[both_styles]
        .groupby(["ba_style", "rb_style"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\n=== Top 30 Style Combinations ===")
    print(style_pairs.head(30).to_string(index=False))

    # Examine styles among strong name and ABV candidates
    strong_candidates = df[
        (~df["is_ambiguous_name"]) &
        (df["abv_difference"].round(4) <= 0.3)
    ].copy()

    strong_style_pairs = (
        strong_candidates
        .dropna(subset=["ba_style", "rb_style"])
        .groupby(["ba_style", "rb_style"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    print("\n=== Top 30 Style Combinations (Strong Candidates) ===")
    print(strong_style_pairs.head(30).to_string(index=False))

    print("\n=== Strong Candidate Summary ===")
    print(f"Strong candidates: {len(strong_candidates):,}")
    print(
        "Exact style matches:",
        f"{strong_candidates['style_exact_match'].sum():,}"
    )

if __name__ == "__main__":
    analyze_styles()

print("\nStep 12 Completed.")