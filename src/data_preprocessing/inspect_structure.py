import gzip
import json
import ast
from pathlib import Path
from pprint import pprint

# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

# Define dataset file paths
DATASETS = {
    "BeerAdvocate": RAW_DIR / "beeradvocate.json.gz",
    "RateBeer": RAW_DIR / "ratebeer.json.gz"
}


def inspect_dataset(name, file_path, sample_size=3):
    # Display dataset information
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"{'=' * 60}")

    # Read only the first few records - to avoid loading large files
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for i in range(sample_size):
            line = f.readline()

            if not line:
                break

            # Parse a JSON record, with fallback for Python-literal format
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = ast.literal_eval(line)

            # Display field names and sample values
            print(f"\n--- Record {i + 1} ---")
            pprint(record)

            if i == 0:
                print("\nAvailable fields:")
                print(list(record.keys()))

# Inspect both datasets
for dataset_name, dataset_path in DATASETS.items():
    inspect_dataset(dataset_name, dataset_path)

print("\nStep 2 Completed.")
