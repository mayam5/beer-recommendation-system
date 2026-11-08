from pathlib import Path
import pandas as pd

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

BA_PATH = PROCESSED_DIR / "beeradvocate_beers.parquet"
RB_PATH = PROCESSED_DIR / "ratebeer_beers.parquet"

OUTPUT_PATH = PROCESSED_DIR / "name_match_candidates.parquet"

def load_beer_metadata(path, prefix):
    # Load only columns required for candidate matching
    columns = [
        "beer_id",
        "beer_name",
        "normalized_name",
        "abv",
        "beer_style",
        "review_count",
        "brewer_id_count",
    ]

    df = pd.read_parquet(path, columns=columns)

    # Remove beers without usable normalized names
    df = df.dropna(subset=["normalized_name"]).copy()
    df = df[df["normalized_name"].str.strip().ne("")]

    # Add platform-specific prefixes to avoid ID collisions
    df = df.rename(
        columns={
            "beer_id": f"{prefix}_beer_id",
            "beer_name": f"{prefix}_beer_name",
            "abv": f"{prefix}_abv",
            "beer_style": f"{prefix}_style",
            "review_count": f"{prefix}_review_count",
            "brewer_id_count": f"{prefix}_brewer_id_count",
        }
    )

    return df

def find_candidates():
    print("Loading beer metadata...")

    ba = load_beer_metadata(BA_PATH, "ba")
    rb = load_beer_metadata(RB_PATH, "rb")

    print(f"BeerAdvocate beers: {len(ba):,}")
    print(f"RateBeer beers: {len(rb):,}")

    # Count how many beers share each normalized name
    ba_name_counts = (
        ba.groupby("normalized_name")
        .size()
        .rename("ba_name_count")
    )

    rb_name_counts = (
        rb.groupby("normalized_name")
        .size()
        .rename("rb_name_count")
    )

    # Match candidate beers using normalized names
    candidates = ba.merge(
        rb,
        on="normalized_name",
        how="inner",
        validate="many_to_many"
    )

    # Add name frequency information
    candidates = candidates.join(
        ba_name_counts,
        on="normalized_name"
    )

    candidates = candidates.join(
        rb_name_counts,
        on="normalized_name"
    )

    # Mark names that create ambiguous matches
    candidates["is_ambiguous_name"] = (
        (candidates["ba_name_count"] > 1) |
        (candidates["rb_name_count"] > 1)
    )

    # Calculate ABV differences for later verification
    candidates["abv_difference"] = (
        candidates["ba_abv"] - candidates["rb_abv"]
    ).abs()

    # Compare style strings without assuming equivalence
    ba_style = (
        candidates["ba_style"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    rb_style = (
        candidates["rb_style"]
        .astype("string")
        .str.strip()
        .str.casefold()
    )

    candidates["style_exact_match"] = (
        ba_style.notna() &
        rb_style.notna() &
        ba_style.eq(rb_style).fillna(False)
    )

    # Save all candidates without automatically confirming matches
    candidates.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy"
    )

    print("\n=== Name Matching Summary ===")

    print(f"Candidate pairs: {len(candidates):,}")

    print(
        "Unique matched names:",
        f"{candidates['normalized_name'].nunique():,}"
    )

    print(
        "BeerAdvocate beers with candidates:",
        f"{candidates['ba_beer_id'].nunique():,}"
    )

    print(
        "RateBeer beers with candidates:",
        f"{candidates['rb_beer_id'].nunique():,}"
    )

    print(
        "Unambiguous candidate pairs:",
        f"{(~candidates['is_ambiguous_name']).sum():,}"
    )

    print(
        "Ambiguous candidate pairs:",
        f"{candidates['is_ambiguous_name'].sum():,}"
    )

    print(
        "Candidates with both ABVs:",
        f"{candidates['abv_difference'].notna().sum():,}"
    )

    print(
        "Candidates with exact style match:",
        f"{candidates['style_exact_match'].sum():,}"
    )

    print(f"\nSaved to: {OUTPUT_PATH}")

    # Display a small sample of candidates
    print("\n=== Candidate Samples ===")

    sample_columns = [
        "normalized_name",
        "ba_beer_id",
        "rb_beer_id",
        "ba_abv",
        "rb_abv",
        "abv_difference",
        "is_ambiguous_name",
    ]

    print(
        candidates[sample_columns]
        .head(10)
        .to_string(index=False)
    )

if __name__ == "__main__":
    find_candidates()

print("\nStep 10 Completed.")
