import gzip
import json
import ast
from pathlib import Path

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

def inspect_duplicates(name, file_path, sample_size=5):
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    # Store the first occurrence of each user-beer pair
    first_records = {}
    duplicate_examples = []

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

            if key in first_records:
                # Collect a limited number of duplicate examples
                if len(duplicate_examples) < sample_size:
                    duplicate_examples.append(
                        (first_records[key], record)
                    )

                if len(duplicate_examples) >= sample_size:
                    break
            else:
                first_records[key] = record

    # Display duplicate examples
    for i, (first, second) in enumerate(
        duplicate_examples, start=1
    ):
        print(f"\n--- Duplicate Example {i} ---")
        print("Beer:", first.get("beer/name"))

        print("First rating:", first.get("review/overall"))
        print("Second rating:", second.get("review/overall"))

        print("First time:", first.get("review/time"))
        print("Second time:", second.get("review/time"))

        print(
            "Same review text:",
            first.get("review/text") == second.get("review/text")
        )

        print("Exactly same record:", first == second)

        # Identify fields with different values
        different_fields = []

        for field in sorted(set(first) | set(second)):
            if first.get(field) != second.get(field):
                different_fields.append(field)

        print("Different fields:", different_fields)

        # Display differences without printing full review text
        for field in different_fields:
            if field == "review/text":
                continue

            print(f"  {field}:")
            print(f"    First: {first.get(field)}")
            print(f"    Second: {second.get(field)}")

    print(f"\nFound {len(duplicate_examples)} examples.")


# Inspect both datasets
for dataset_name, dataset_path in DATASETS.items():
    inspect_duplicates(dataset_name, dataset_path)

print("\nStep 4 Completed.")
