import gzip
import json
import ast
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt


# Define project directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

DATASETS = {
    "BeerAdvocate": RAW_DIR / "beeradvocate.json.gz",
    "RateBeer": RAW_DIR / "ratebeer.json.gz"
}

FIELDS = [
    "beer/name",
    "beer/beerId",
    "beer/brewerId",
    "beer/ABV",
    "beer/style",
    "review/appearance",
    "review/aroma",
    "review/overall",
    "review/palate",
    "review/taste",
    "review/profileName",
    "review/text",
    "review/time"
]


def parse_record(line):
    # Parse JSON or Python-literal formatted records
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)

def plot_rating_distribution(name, rating_counts):
    # Sort original rating values numerically
    if name == "RateBeer":
        ratings = sorted(
            rating_counts.keys(),
            key=lambda x: int(x.split("/")[0])
        )
    else:
        ratings = sorted(
            rating_counts.keys(),
            key=lambda x: float(x)
        )

    counts = [rating_counts[r] for r in ratings]

    # Create bar chart
    plt.figure(figsize=(12, 5))

    plt.bar(
        ratings,
        counts,
        color="steelblue",
        edgecolor="black"
    )

    plt.title(f"{name} - Original Overall Rating Distribution")
    plt.xlabel("Original Overall Rating")
    plt.ylabel("Number of Reviews")

    plt.xticks(rotation=45)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # Save chart
    output_dir = BASE_DIR / "results" / "data_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name.lower()}_original_rating_distribution.png"

    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Rating distribution plot saved: {output_path}")

def analyze_dataset(name, file_path):
    # Initialize statistics
    total_reviews = 0
    missing_counts = Counter()
    rating_counts = Counter()

    users = set()
    beers = set()
    user_beer_counts = Counter()

    print(f"\n{'=' * 60}")
    print(f"Analyzing: {name}")
    print(f"{'=' * 60}")

    # Read records sequentially to reduce memory usage
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            record = parse_record(line)
            total_reviews += 1

            # Count missing or empty values
            for field in FIELDS:
                value = record.get(field)

                if value is None or str(value).strip() == "":
                    missing_counts[field] += 1

            # Collect unique users and beers
            user = record.get("review/profileName")
            beer = record.get("beer/beerId")

            if user and beer:
                users.add(user)
                beers.add(beer)
                user_beer_counts[(user, beer)] += 1

            # Count original rating values
            rating = record.get("review/overall")
            if rating is not None and str(rating).strip():
                rating_counts[str(rating)] += 1

            # Display progress
            if total_reviews % 500000 == 0:
                print(f"Processed {total_reviews:,} reviews...")

    # Count repeated user-beer combinations
    duplicate_pairs = sum(
        count > 1 for count in user_beer_counts.values()
    )

    extra_reviews = sum(
        count - 1 for count in user_beer_counts.values()
        if count > 1
    )

    # Display summary
    print("\n=== Dataset Summary ===")
    print(f"Total reviews: {total_reviews:,}")
    print(f"Unique users: {len(users):,}")
    print(f"Unique beers: {len(beers):,}")
    print(f"Unique user-beer pairs: {len(user_beer_counts):,}")

    print("\n=== Missing Values ===")
    for field in FIELDS:
        print(f"{field}: {missing_counts[field]:,}")

    print("\n=== Repeated User-Beer Pairs ===")
    print(f"Repeated pairs: {duplicate_pairs:,}")
    print(f"Extra reviews: {extra_reviews:,}")

    print("\n=== Overall Rating Distribution ===")
    for rating, count in sorted(rating_counts.items()):
        print(f"{rating}: {count:,}")
        
    # Visualize original rating distribution
    plot_rating_distribution(name, rating_counts)

    print(f"\n{name} analysis completed.")

# Analyze both datasets
for dataset_name, dataset_path in DATASETS.items():
    analyze_dataset(dataset_name, dataset_path)

print("\nStep 3 Completed.")
