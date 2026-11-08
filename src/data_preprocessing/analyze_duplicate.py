import gzip
import json
import ast
from pathlib import Path
from collections import Counter

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "BeerAdvocate": RAW_DIR / "beeradvocate.json.gz",
    "RateBeer": RAW_DIR / "ratebeer.json.gz"
}


def parse_record(line):
    # Parse JSON or Python-literal formatted records
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)

def analyze_duplicates(name, file_path):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    first_records = {}
    stats = Counter()

    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = parse_record(line)

            user = record.get("review/profileName")
            beer = record.get("beer/beerId")

            if not user or not beer:
                continue

            key = (user, beer)

            if key not in first_records:
                first_records[key] = record
                continue

            # Compare each repeated record with its first occurrence
            first = first_records[key]
            stats["extra_reviews"] += 1

            if first == record:
                stats["exact_same_record"] += 1

            if (
                first.get("review/text") == record.get("review/text")
                and first.get("review/time") == record.get("review/time")
                and first.get("review/overall") == record.get("review/overall")
            ):
                stats["same_text_time_rating"] += 1

            if first.get("review/time") != record.get("review/time"):
                stats["different_time"] += 1

            if first.get("review/overall") != record.get("review/overall"):
                stats["different_rating"] += 1

            if first.get("review/text") != record.get("review/text"):
                stats["different_text"] += 1

            if first.get("beer/brewerId") != record.get("beer/brewerId"):
                stats["different_brewer"] += 1

    print("\n=== Duplicate Type Summary ===")

    for key, label in [
        ("extra_reviews", "Extra reviews"),
        ("exact_same_record", "Exactly identical records"),
        ("same_text_time_rating", "Same text, time, and rating"),
        ("different_time", "Different review time"),
        ("different_rating", "Different overall rating"),
        ("different_text", "Different review text"),
        ("different_brewer", "Different brewer ID")
    ]:
        print(f"{label}: {stats[key]:,}")

    print(f"\n{name} analysis completed.")

# Analyze both datasets
for dataset_name, dataset_path in DATASETS.items():
    analyze_duplicates(dataset_name, dataset_path)

print("\nStep 5 Completed.")
