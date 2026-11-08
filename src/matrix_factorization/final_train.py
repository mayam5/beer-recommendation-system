from pathlib import Path
import json

import numpy as np
import pandas as pd

from model import MatrixFactorization

# =========================================================
# 1. Set project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "final_data"


RESULTS_DIR = PROJECT_DIR / "results" / "matrix_factorization"

TRAINING_DIR = RESULTS_DIR / "training"
MODEL_DIR = RESULTS_DIR / "final_model"

TRAIN_PATH = (
    DATA_DIR / "train" / "beeradvocate_train_interactions.parquet"
)

MAPPING_PATH = DATA_DIR / "beer_mapping.parquet"

PARAMS_PATH = TRAINING_DIR / "best_params.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Load best hyperparameters
# =========================================================

with open(PARAMS_PATH, "r", encoding="utf-8") as f:
    best_params = json.load(f)

print("========== BEST HYPERPARAMETERS ==========")

for key, value in best_params.items():
    print(f"{key}: {value}")

# =========================================================
# 3. Load full BeerAdvocate training data
# =========================================================

train = pd.read_parquet(TRAIN_PATH)
mapping = pd.read_parquet(MAPPING_PATH)

print("\n========== FULL TRAINING DATA ==========")
print("Interactions:", len(train))
print("Users:", train["user_id"].nunique())
print("Mapped beers:", mapping["canonical_beer_id"].nunique())

# =========================================================
# 4. Create user and item indices
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
# 5. Convert data to NumPy arrays
# =========================================================

user_indices = train["user_idx"].to_numpy(dtype=np.int32)
item_indices = train["item_idx"].to_numpy(dtype=np.int32)
ratings = train["rating"].to_numpy(dtype=np.float64)

assert np.isfinite(ratings).all()
assert ((ratings >= 0) & (ratings <= 5)).all()

print("\n========== MODEL INPUT ==========")
print("User indices:", user_indices.shape)
print("Item indices:", item_indices.shape)
print("Ratings:", ratings.shape)

# =========================================================
# 6. Initialize final MF model
# =========================================================

model = MatrixFactorization(
    n_users=len(user_ids),
    n_items=len(beer_ids),
    n_factors=int(best_params["n_factors"]),
    learning_rate=float(best_params["learning_rate"]),
    reg=float(best_params["reg"]),
    n_epochs=int(best_params["n_epochs"]),
    random_state=int(best_params["random_state"])
)

# =========================================================
# 7. Train on the full dataset
# =========================================================

model.fit(
    user_indices,
    item_indices,
    ratings
)

# =========================================================
# 8. Calculate final training RMSE
# =========================================================

final_train_rmse = model.compute_rmse(
    user_indices,
    item_indices,
    ratings
)

print("\n========== FINAL TRAINING RESULT ==========")
print("Final train RMSE:", round(final_train_rmse, 4))

final_history_df = pd.DataFrame({
    "epoch": np.arange(1, len(model.history) + 1),
    "train_rmse": model.history
})

final_history_df.to_csv(
    TRAINING_DIR / "final_training_history.csv",
    index=False
)

# =========================================================
# 9. Save trained model parameters
# =========================================================

np.savez_compressed(
    MODEL_DIR / "mf_model.npz",
    user_factors=model.user_factors,
    item_factors=model.item_factors,
    user_bias=model.user_bias,
    item_bias=model.item_bias,
    global_mean=np.array(model.global_mean)
)

print("\n[PASS] Model parameters saved")

# =========================================================
# 10. Save user and beer index mappings
# =========================================================

user_index_df = pd.DataFrame({
    "user_id": user_ids,
    "user_idx": np.arange(len(user_ids))
})

beer_index_df = pd.DataFrame({
    "canonical_beer_id": beer_ids,
    "item_idx": np.arange(len(beer_ids))
})

user_index_df.to_parquet(
    MODEL_DIR / "user_index.parquet",
    index=False
)

beer_index_df.to_parquet(
    MODEL_DIR / "beer_index.parquet",
    index=False
)

print("[PASS] User index saved")
print("[PASS] Beer index saved")

# =========================================================
# 11. Save training information
# =========================================================

training_info = {
    "dataset": "BeerAdvocate",
    "n_users": len(user_ids),
    "n_items": len(beer_ids),
    "n_interactions": len(train),
    "n_factors": int(best_params["n_factors"]),
    "learning_rate": float(best_params["learning_rate"]),
    "reg": float(best_params["reg"]),
    "n_epochs": int(best_params["n_epochs"]),
    "random_state": int(best_params["random_state"]),
    "global_mean": float(model.global_mean),
    "final_train_rmse": float(final_train_rmse)
}

with open(
    MODEL_DIR / "training_info.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        training_info,
        f,
        indent=4,
        ensure_ascii=False
    )

print("[PASS] Training information saved")

# =========================================================
# 12. Verify saved model
# =========================================================

with np.load(MODEL_DIR / "mf_model.npz") as saved:
    assert saved["user_factors"].shape == (
        len(user_ids),
        int(best_params["n_factors"])
    )

    assert saved["item_factors"].shape == (
        len(beer_ids),
        int(best_params["n_factors"])
    )

    assert saved["user_bias"].shape == (len(user_ids),)
    assert saved["item_bias"].shape == (len(beer_ids),)

    assert np.isfinite(saved["user_factors"]).all()
    assert np.isfinite(saved["item_factors"]).all()
    assert np.isfinite(saved["user_bias"]).all()
    assert np.isfinite(saved["item_bias"]).all()

print("[PASS] Saved model verified")

print("\nMF-Step5 Completed.")
