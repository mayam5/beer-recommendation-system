from pathlib import Path
import pandas as pd

# =========================================================
# Configuration
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_DIR / "data" / "final_data"

FILES = {
    "mapping": DATA_DIR / "beer_mapping.parquet",
    "train": DATA_DIR / "train" / "beeradvocate_train_interactions.parquet",
    "beers": DATA_DIR / "train" / "beeradvocate_beers.parquet",
    "reviews": DATA_DIR / "train" / "beeradvocate_reviews.parquet",
    "input": DATA_DIR / "test" / "ratebeer_test_input.parquet",
    "holdout": DATA_DIR / "test" / "ratebeer_test_holdout.parquet",
    "users": DATA_DIR / "test" / "ratebeer_test_users.parquet",
}

REQUIRED_COLUMNS = {
    "mapping": [
        "canonical_beer_id", "ba_beer_id", "rb_beer_id"
    ],
    "train": [
        "user_id", "canonical_beer_id", "rating", "review_count"
    ],
    "beers": [
        "beer_id", "beer_name", "abv", "beer_style"
    ],
    "reviews": [
        "user_id", "beer_id", "rating_5",
        "has_valid_user", "has_valid_beer",
        "has_valid_rating", "has_review_text"
    ],
    "input": [
        "user_id", "canonical_beer_id", "rating"
    ],
    "holdout": [
        "user_id", "canonical_beer_id", "rating", "is_positive"
    ],
    "users": [
        "user_id", "holdout_count",
        "positive_holdout_count", "evaluable"
    ],
}

EXPECTED_ROWS = {
    "mapping": 2095,
    "train": 140665,
    "beers": 66055,
    "input": 150532,
    "holdout": 67124,
    "users": 4101,
}

data = {}
errors = []
warnings = []

def check(condition, message):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[FAIL] {message}")
        errors.append(message)

def warn(condition, message):
    if condition:
        print(f"[PASS] {message}")
    else:
        print(f"[WARN] {message}")
        warnings.append(message)

def check_unique(df, columns, name):
    count = int(df.duplicated(subset=columns).sum())
    check(count == 0, f"{name}: duplicate keys = {count}")

def check_rating(df, column, name):
    rating = pd.to_numeric(df[column], errors="coerce")
    invalid = int((~rating.between(0, 5)).sum())
    check(invalid == 0, f"{name}: invalid ratings = {invalid}")

# =========================================================
# 1. File loading
# =========================================================

print("\n========== 1. FILE LOADING ==========")
print("Data directory:", DATA_DIR)

for key, path in FILES.items():
    if not path.is_file():
        check(False, f"{key}: file not found - {path}")
        continue

    try:
        df = pd.read_parquet(path)
        data[key] = df

        print(f"\nFile: {path.relative_to(PROJECT_DIR)}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(
            f"Memory: "
            f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        check(True, f"{key}: readable")

        if key in EXPECTED_ROWS:
            warn(
                len(df) == EXPECTED_ROWS[key],
                f"{key}: expected {EXPECTED_ROWS[key]} rows, "
                f"actual {len(df)}"
            )

    except Exception as exc:
        check(False, f"{key}: read error - {exc}")

# =========================================================
# 2. Column validation
# =========================================================

print("\n========== 2. COLUMN VALIDATION ==========")

for key, df in data.items():
    missing = sorted(
        set(REQUIRED_COLUMNS[key]) - set(df.columns)
    )
    check(
        len(missing) == 0,
        f"{key}: missing required columns = {missing}"
    )

if len(data) != len(FILES) or errors:
    print("\n========== VALIDATION ERRORS ==========")

    for error in errors:
        print("[ERROR]", error)

    print("\nValidation stopped: file or column errors.")
    raise SystemExit(1)

mapping = data["mapping"]
train = data["train"]
beers = data["beers"]
reviews = data["reviews"]
test_input = data["input"]
holdout = data["holdout"]
users = data["users"]

# =========================================================
# 3. Missing values
# =========================================================

print("\n========== 3. MISSING VALUES ==========")

CRITICAL_COLUMNS = {
    "mapping": [
        "canonical_beer_id", "ba_beer_id", "rb_beer_id"
    ],
    "train": [
        "user_id", "canonical_beer_id", "rating"
    ],
    "beers": ["beer_id"],
    "reviews": [],
    "input": [
        "user_id", "canonical_beer_id", "rating"
    ],
    "holdout": [
        "user_id", "canonical_beer_id", "rating", "is_positive"
    ],
    "users": [
        "user_id", "holdout_count",
        "positive_holdout_count", "evaluable"
    ],
}

for key, df in data.items():
    print(f"\n[{key}]")

    missing_counts = df.isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if len(missing_counts) > 0:
        print(missing_counts.to_string())
    else:
        print("No missing values")

    critical = CRITICAL_COLUMNS[key]

    if critical:
        count = int(df[critical].isna().sum().sum())

        check(
            count == 0,
            f"{key}: critical missing values = {count}"
        )

# =========================================================
# 4. Duplicate validation
# =========================================================

print("\n========== 4. DUPLICATE VALIDATION ==========")

for column in [
    "canonical_beer_id", "ba_beer_id", "rb_beer_id"
]:
    check_unique(mapping, [column], f"mapping {column}")

check_unique(beers, ["beer_id"], "beer metadata")

check_unique(
    train,
    ["user_id", "canonical_beer_id"],
    "train"
)

check_unique(
    test_input,
    ["user_id", "canonical_beer_id"],
    "test input"
)

check_unique(
    holdout,
    ["user_id", "canonical_beer_id"],
    "test holdout"
)

check_unique(users, ["user_id"], "test users")

# Repeated reviews are allowed in the original review dataset.
repeated_reviews = int(
    reviews.duplicated(
        subset=["user_id", "beer_id"]
    ).sum()
)

print(
    f"[INFO] reviews: repeated user-beer rows = "
    f"{repeated_reviews}"
)

# =========================================================
# 5. Rating validation
# =========================================================

print("\n========== 5. RATING VALIDATION ==========")

for key in ["train", "input", "holdout"]:
    check_rating(data[key], "rating", key)

valid_review_ratings = reviews.loc[
    reviews["has_valid_rating"].eq(True), "rating_5"
]

check_rating(
    pd.DataFrame({"rating_5": valid_review_ratings}),
    "rating_5",
    "valid reviews"
)

expected_positive = (
    pd.to_numeric(
        holdout["rating"], errors="coerce"
    ) >= 3.5
)

check(
    holdout["is_positive"].eq(expected_positive).all(),
    "holdout: is_positive matches rating >= 3.5"
)

check(
    (
        pd.to_numeric(
            train["review_count"], errors="coerce"
        ) >= 1
    ).all(),
    "train: review_count >= 1"
)

# =========================================================
# 6. Mapping coverage
# =========================================================

print("\n========== 6. MAPPING COVERAGE ==========")

mapped_ids = set(
    mapping["canonical_beer_id"].dropna()
)

ba_ids = set(
    mapping["ba_beer_id"].dropna()
)

train_beers = set(
    train["canonical_beer_id"].dropna()
)

for key in ["train", "input", "holdout"]:
    ids = set(
        data[key]["canonical_beer_id"].dropna()
    )

    missing = ids - mapped_ids

    check(
        len(missing) == 0,
        f"{key}: beers missing from mapping = {len(missing)}"
    )

metadata_ids = set(
    beers["beer_id"].dropna()
)

check(
    len(ba_ids - metadata_ids) == 0,
    "mapping: all BA beer IDs exist in beer metadata"
)

check(
    len(
        set(holdout["canonical_beer_id"]) - train_beers
    ) == 0,
    "holdout: all beers exist in train"
)

check(
    len(
        set(test_input["canonical_beer_id"]) - train_beers
    ) == 0,
    "input: all beers exist in train"
)

# =========================================================
# 7. Review data validation
# =========================================================

print("\n========== 7. REVIEW DATA VALIDATION ==========")

common_reviews = reviews[
    reviews["beer_id"].isin(ba_ids)
].copy()

print("Total review rows:", len(reviews))
print("Common-beer review rows:", len(common_reviews))
print(
    "Common beers with reviews:",
    common_reviews["beer_id"].nunique()
)

valid_common_reviews = common_reviews[
    common_reviews["has_valid_user"].eq(True)
    & common_reviews["has_valid_beer"].eq(True)
    & common_reviews["has_valid_rating"].eq(True)
]

print(
    "Valid common-beer reviews:",
    len(valid_common_reviews)
)

text_reviews = common_reviews[
    common_reviews["has_review_text"].eq(True)
]

print(
    "Common-beer reviews with text:",
    len(text_reviews)
)

check(
    len(common_reviews) > 0,
    "reviews: common-beer reviews exist"
)

warn(
    common_reviews["beer_id"].nunique() == len(ba_ids),
    "reviews: every mapped BA beer has reviews"
)

warn(
    len(text_reviews) > 0,
    "reviews: common-beer text is available"
)

warn(
    len(valid_common_reviews) == 142296,
    "reviews: valid common reviews match previous result (142296)"
)

# =========================================================
# 8. Train/review consistency
# =========================================================

print("\n========== 8. TRAIN/REVIEW CONSISTENCY ==========")

# Recreate training interactions from valid common reviews.
id_mapping = mapping[
    ["ba_beer_id", "canonical_beer_id"]
]

reconstructed = valid_common_reviews.merge(
    id_mapping,
    left_on="beer_id",
    right_on="ba_beer_id",
    how="inner",
    validate="many_to_one"
)

reconstructed = (
    reconstructed.groupby(
        ["user_id", "canonical_beer_id"],
        as_index=False
    )
    .agg(
        expected_rating=("rating_5", "mean"),
        expected_review_count=("rating_5", "size")
    )
)

comparison = train.merge(
    reconstructed,
    on=["user_id", "canonical_beer_id"],
    how="outer",
    indicator=True
)

check(
    comparison["_merge"].eq("both").all(),
    "train: all user-beer pairs match valid source reviews"
)

matched = comparison[
    comparison["_merge"].eq("both")
]

rating_difference = (
    matched["rating"] - matched["expected_rating"]
).abs()

check(
    rating_difference.le(1e-6).all(),
    "train: average ratings match source reviews"
)

check(
    matched["review_count"].eq(
        matched["expected_review_count"]
    ).all(),
    "train: review counts match source reviews"
)

# =========================================================
# 9. Input/Holdout split validation
# =========================================================

print("\n========== 9. TEST SPLIT VALIDATION ==========")

input_pairs = pd.MultiIndex.from_frame(
    test_input[
        ["user_id", "canonical_beer_id"]
    ]
)

holdout_pairs = pd.MultiIndex.from_frame(
    holdout[
        ["user_id", "canonical_beer_id"]
    ]
)

overlap = input_pairs.intersection(
    holdout_pairs
)

check(
    len(overlap) == 0,
    f"input/holdout: overlapping pairs = {len(overlap)}"
)

registered_users = set(
    users["user_id"].dropna()
)

input_users = set(
    test_input["user_id"].dropna()
)

holdout_users = set(
    holdout["user_id"].dropna()
)

check(
    registered_users == input_users,
    "input users match registered test users"
)

check(
    registered_users == holdout_users,
    "holdout users match registered test users"
)

input_counts = (
    test_input.groupby("user_id").size()
)

holdout_counts = (
    holdout.groupby("user_id").size()
)

total_counts = input_counts.add(
    holdout_counts,
    fill_value=0
)

input_ratios = input_counts.div(
    total_counts
)

print("\nInput ratio statistics:")
print(input_ratios.describe().to_string())

warn(
    input_ratios.between(0.60, 0.75).all(),
    "all user input ratios are between 60% and 75%"
)

# =========================================================
# 10. Evaluation user validation
# =========================================================

print("\n========== 10. EVALUATION USER VALIDATION ==========")

actual_holdout_counts = (
    holdout.groupby("user_id").size()
)

actual_positive_counts = (
    holdout[
        holdout["is_positive"].eq(True)
    ]
    .groupby("user_id")
    .size()
)

expected = users.set_index("user_id")

actual_holdout_counts = (
    actual_holdout_counts.reindex(
        expected.index,
        fill_value=0
    )
)

actual_positive_counts = (
    actual_positive_counts.reindex(
        expected.index,
        fill_value=0
    )
)

check(
    expected["holdout_count"].eq(
        actual_holdout_counts
    ).all(),
    "users: holdout_count matches actual data"
)

check(
    expected["positive_holdout_count"].eq(
        actual_positive_counts
    ).all(),
    "users: positive_holdout_count matches actual data"
)

expected_evaluable = (
    actual_positive_counts >= 1
)

check(
    expected["evaluable"].eq(
        expected_evaluable
    ).all(),
    "users: evaluable matches positive holdout count"
)

evaluable_count = int(
    expected_evaluable.sum()
)

positive_count = int(
    actual_positive_counts.sum()
)

print("\nTotal test users:", len(users))
print("Evaluable users:", evaluable_count)
print(
    "Positive holdout interactions:",
    positive_count
)

warn(
    evaluable_count == 3858,
    "evaluable users match previous result (3858)"
)

warn(
    positive_count == 28709,
    "positive holdout count matches previous result (28709)"
)

# =========================================================
# 11. Final summary
# =========================================================

print("\n========== FINAL VALIDATION SUMMARY ==========")

print("Errors:", len(errors))
print("Warnings:", len(warnings))

if errors:
    print("\nFAILED CHECKS:")
    for message in errors:
        print("-", message)

if warnings:
    print("\nWARNINGS:")
    for message in warnings:
        print("-", message)

if not errors:
    print("\nRESULT: CORE DATA VALIDATION PASSED")
    print("Files are structurally ready for model development.")
else:
    print("\nRESULT: VALIDATION FAILED")
    print("Resolve errors before sharing the data.")
