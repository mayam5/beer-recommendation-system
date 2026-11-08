from pathlib import Path
import numpy as np
import pandas as pd

# =========================================================
# 1. Set project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "final_data"

TRAIN_PATH = (
    DATA_DIR / "train" / "beeradvocate_train_interactions.parquet"
)

MAPPING_PATH = DATA_DIR / "beer_mapping.parquet"

# =========================================================
# 2. Load training data
# =========================================================

train = pd.read_parquet(TRAIN_PATH)
mapping = pd.read_parquet(MAPPING_PATH)

print("========== TRAIN DATA ==========")
print("Shape:", train.shape)
print("Columns:", train.columns.tolist())
print(train.head())

# =========================================================
# 3. Create user and beer indices
# =========================================================

user_ids = sorted(train["user_id"].unique())
beer_ids = sorted(mapping["canonical_beer_id"].unique())

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

beer_to_idx = {
    beer_id: idx
    for idx, beer_id in enumerate(beer_ids)
}

train["user_idx"] = train["user_id"].map(user_to_idx)
train["beer_idx"] = train["canonical_beer_id"].map(beer_to_idx)

print("\n========== INDEX INFORMATION ==========")
print("Number of users:", len(user_ids))
print("Number of beers:", len(beer_ids))
print("Number of interactions:", len(train))

# =========================================================
# 4. Convert data to NumPy arrays
# =========================================================

user_indices = train["user_idx"].to_numpy(dtype=np.int32)
beer_indices = train["beer_idx"].to_numpy(dtype=np.int32)
ratings = train["rating"].to_numpy(dtype=np.float64)

print("\n========== NUMPY ARRAYS ==========")
print("User indices shape:", user_indices.shape)
print("Beer indices shape:", beer_indices.shape)
print("Ratings shape:", ratings.shape)

# =========================================================
# 5. Validate prepared data
# =========================================================

print("\n========== VALIDATION ==========")

assert not train["user_idx"].isna().any()
assert not train["beer_idx"].isna().any()

assert np.isfinite(ratings).all()
assert ((ratings >= 0) & (ratings <= 5)).all()

assert user_indices.min() >= 0
assert beer_indices.min() >= 0

assert user_indices.max() < len(user_ids)
assert beer_indices.max() < len(beer_ids)

print("[PASS] All indices are valid")
print("[PASS] All ratings are valid")

# =========================================================
# 6. Rating statistics
# =========================================================

print("\n========== RATING STATISTICS ==========")

print("Mean rating:", round(ratings.mean(), 4))
print("Standard deviation:", round(ratings.std(), 4))
print("Minimum rating:", ratings.min())
print("Maximum rating:", ratings.max())

# =========================================================
# 7. Calculate matrix sparsity
# =========================================================

num_users = len(user_ids)
num_beers = len(beer_ids)

total_possible = num_users * num_beers
observed = len(train)

sparsity = 1 - (observed / total_possible)

print("\n========== MATRIX INFORMATION ==========")
print("Matrix shape:", (num_users, num_beers))
print("Total possible ratings:", total_possible)
print("Observed ratings:", observed)
print("Sparsity:", f"{sparsity * 100:.4f}%")

print("\nMF-Step1 Completed.")
