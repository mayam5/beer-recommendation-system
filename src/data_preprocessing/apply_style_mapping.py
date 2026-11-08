from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_PATH = PROCESSED_DIR / "style_mapping_template.csv"
OUTPUT_PATH = PROCESSED_DIR / "style_mapping_reviewed.csv"

# Define directly corresponding style pairs
EQUIVALENT_PAIRS = [
    ("American IPA", "India Pale Ale (IPA)"),
    ("Hefeweizen", "German Hefeweizen"),
    ("American Pale Ale (APA)", "American Pale Ale"),
    ("German Pilsener", "Classic German Pilsener"),
    ("English Bitter", "Bitter"),
    ("Witbier", "Belgian White (Witbier)"),
    ("Märzen / Oktoberfest", "Oktoberfest/Mrzen"),
    ("American Double / Imperial IPA", "Imperial/Double IPA"),
    ("Doppelbock", "Doppelbock"),
    ("Saison / Farmhouse Ale", "Saison"),
    ("Tripel", "Abbey Tripel"),
    ("Altbier", "Altbier"),
    ("Dunkelweizen", "Dunkelweizen"),
]

# Define broadly compatible style pairs
COMPATIBLE_PAIRS = [
    ("American Adjunct Lager", "Pale Lager"),
    ("American Porter", "Porter"),
    ("Fruit / Vegetable Beer", "Fruit Beer"),
    ("Light Lager", "Pale Lager"),
    ("American Amber / Red Ale", "Amber Ale"),
    ("Euro Pale Lager", "Pale Lager"),
    ("German Pilsener", "Pilsener"),
    ("Bock", "Dunkler Bock"),
    ("American Blonde Ale", "Golden Ale/Blond Ale"),
    ("English Brown Ale", "Brown Ale"),
    ("American Stout", "Stout"),
    ("Maibock / Helles Bock", "Heller Bock"),
    ("American Pale Wheat Ale", "Wheat Ale"),
    ("Munich Dunkel Lager", "Dunkel"),
    ("Munich Helles Lager", "Dortmunder/Helles"),
    ("Dortmunder / Export Lager", "Dortmunder/Helles"),
]

def apply_style_mapping():
    # Load the original style mapping template
    df = pd.read_csv(INPUT_PATH)

    # Keep existing classifications from the template
    df["style_relation"] = df["style_relation"].fillna(
        "unreviewed"
    )

    equivalent_set = set(EQUIVALENT_PAIRS)
    compatible_set = set(COMPATIBLE_PAIRS)

    # Ensure that mapping rules do not conflict
    overlap = equivalent_set & compatible_set
    if overlap:
        raise ValueError(f"Conflicting mapping rules: {overlap}")

    # Apply manually defined style relationships
    for index, row in df.iterrows():
        pair = (row["ba_style"], row["rb_style"])

        if pair in equivalent_set:
            df.at[index, "style_relation"] = "equivalent"

        elif pair in compatible_set:
            df.at[index, "style_relation"] = "compatible"

    # Save the reviewed mapping table
    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("=== Style Mapping Results ===")

    print(f"Total style combinations: {len(df):,}")

    for relation, count in df["style_relation"].value_counts().items():
        print(f"{relation}: {count:,}")

    print("\n=== Candidate Coverage by Style Relation ===")

    coverage = (
        df.groupby("style_relation")[
            ["candidate_count", "strong_candidate_count"]
        ]
        .sum()
        .sort_values(
            "candidate_count",
            ascending=False
        )
    )

    print(coverage.to_string())

    print(f"\nSaved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    apply_style_mapping()

print("\nStep 14 Completed.")
