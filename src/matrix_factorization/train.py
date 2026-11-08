from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from model import MatrixFactorization

# =========================================================
# 1. Set project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data" / "final_data"

TRAIN_PATH = (
    DATA_DIR / "train" / "beeradvocate_train_interactions.parquet"
)

MAPPING_PATH = DATA_DIR / "beer_mapping.parquet"

RESULTS_DIR = PROJECT_DIR / "results" / "matrix_factorization"
OUTPUT_DIR = RESULTS_DIR / "training"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Load data
# =========================================================

train = pd.read_parquet(TRAIN_PATH)
mapping = pd.read_parquet(MAPPING_PATH)

print("========== DATA LOADING ==========")
print("Interactions:", len(train))
print("Unique users:", train["user_id"].nunique())
print("Unique beers:", train["canonical_beer_id"].nunique())

# =========================================================
# 3. Create indices
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
train["item_idx"] = train["canonical_beer_id"].map(beer_to_idx)

assert not train[["user_idx", "item_idx"]].isna().any().any()

# =========================================================
# 4. Train/Validation split
# =========================================================

train_df, val_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Move validation interactions with unseen users/items
# into training to avoid cold-start during validation.

train_users = set(train_df["user_idx"])
train_items = set(train_df["item_idx"])

unseen_mask = (
    ~val_df["user_idx"].isin(train_users)
    | ~val_df["item_idx"].isin(train_items)
)

moved_rows = val_df.loc[unseen_mask]
val_df = val_df.loc[~unseen_mask]

train_df = pd.concat(
    [train_df, moved_rows],
    ignore_index=True
)

print("\n========== DATA SPLIT ==========")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(val_df))
print("Moved cold-start interactions:", len(moved_rows))

assert len(train_df) + len(val_df) == len(train)
assert len(val_df) > 0

assert set(val_df["user_idx"]).issubset(set(train_df["user_idx"]))
assert set(val_df["item_idx"]).issubset(set(train_df["item_idx"]))

# =========================================================
# 5. Convert to NumPy arrays
# =========================================================

train_users = train_df["user_idx"].to_numpy(dtype=np.int32)
train_items = train_df["item_idx"].to_numpy(dtype=np.int32)
train_ratings = train_df["rating"].to_numpy(dtype=np.float64)

val_users = val_df["user_idx"].to_numpy(dtype=np.int32)
val_items = val_df["item_idx"].to_numpy(dtype=np.int32)
val_ratings = val_df["rating"].to_numpy(dtype=np.float64)

# =========================================================
# 6. Initialize model
# =========================================================

model = MatrixFactorization(
    n_users=len(user_ids),
    n_items=len(beer_ids),
    n_factors=20,
    learning_rate=0.01,
    reg=0.05,
    n_epochs=20,
    random_state=42
)

# =========================================================
# 7. Train model
# =========================================================

model.fit(
    train_users,
    train_items,
    train_ratings,
    val_user_indices=val_users,
    val_item_indices=val_items,
    val_ratings=val_ratings
)

# =========================================================
# 8. Final validation results
# =========================================================

print("\n========== FINAL RESULTS ==========")

best_epoch = int(np.argmin(model.val_history)) + 1
best_val_rmse = float(np.min(model.val_history))

final_train_rmse = float(model.history[-1])
final_val_rmse = float(model.val_history[-1])

print("Best epoch:", best_epoch)
print("Best validation RMSE:", round(best_val_rmse, 4))
print("Final train RMSE:", round(final_train_rmse, 4))
print("Final validation RMSE:", round(final_val_rmse, 4))

# Save epoch-by-epoch training history
history_df = pd.DataFrame({
    "epoch": np.arange(1, len(model.history) + 1),
    "train_rmse": model.history,
    "validation_rmse": model.val_history
})

history_path = OUTPUT_DIR / "training_history.csv"
history_df.to_csv(history_path, index=False)

# Save final training summary
summary_df = pd.DataFrame([{
    "best_epoch": best_epoch,
    "best_validation_rmse": best_val_rmse,
    "final_train_rmse": final_train_rmse,
    "final_validation_rmse": final_val_rmse,
    "train_interactions": len(train_df),
    "validation_interactions": len(val_df),
    "moved_cold_start_interactions": len(moved_rows)
}])

summary_path = OUTPUT_DIR / "training_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n========== SAVED RESULTS ==========")
print("Training history:", history_path)
print("Training summary:", summary_path)

print("\nMF-Step3 Completed.")
