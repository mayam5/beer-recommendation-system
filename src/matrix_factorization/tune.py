from pathlib import Path
from itertools import product
import json

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

OUTPUT_DIR = (
    PROJECT_DIR / "results"
    / "matrix_factorization"
    / "training"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Load data and create indices
# =========================================================

train = pd.read_parquet(TRAIN_PATH)
mapping = pd.read_parquet(MAPPING_PATH)

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
# 3. Create fixed Train/Validation split
# =========================================================

train_df, val_df = train_test_split(
    train,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Move cold-start interactions into training
train_users_set = set(train_df["user_idx"])
train_items_set = set(train_df["item_idx"])

unseen_mask = (
    ~val_df["user_idx"].isin(train_users_set)
    | ~val_df["item_idx"].isin(train_items_set)
)

moved_rows = val_df.loc[unseen_mask]

val_df = val_df.loc[~unseen_mask].copy()

train_df = pd.concat(
    [train_df, moved_rows],
    ignore_index=True
)

assert len(val_df) > 0
assert len(train_df) + len(val_df) == len(train)

print("========== DATA SPLIT ==========")
print("Train interactions:", len(train_df))
print("Validation interactions:", len(val_df))
print("Moved cold-start interactions:", len(moved_rows))

# =========================================================
# 4. Convert data to NumPy arrays
# =========================================================

train_users = train_df["user_idx"].to_numpy(dtype=np.int32)
train_items = train_df["item_idx"].to_numpy(dtype=np.int32)
train_ratings = train_df["rating"].to_numpy(dtype=np.float64)

val_users = val_df["user_idx"].to_numpy(dtype=np.int32)
val_items = val_df["item_idx"].to_numpy(dtype=np.int32)
val_ratings = val_df["rating"].to_numpy(dtype=np.float64)

# =========================================================
# 5. Define hyperparameter grid
# =========================================================

param_grid = {
    "n_factors": [10, 20],
    "learning_rate": [0.005, 0.01],
    "reg": [0.02, 0.05],
}

MAX_EPOCHS = 30
RANDOM_STATE = 42

combinations = list(
    product(
        param_grid["n_factors"],
        param_grid["learning_rate"],
        param_grid["reg"]
    )
)

print("\n========== TUNING CONFIGURATION ==========")
print("Number of combinations:", len(combinations))
print("Maximum epochs:", MAX_EPOCHS)

# =========================================================
# 6. Run grid search
# =========================================================

results = []

for experiment_id, (n_factors, learning_rate, reg) in enumerate(
    combinations,
    start=1
):

    print("\n" + "=" * 60)
    print(f"Experiment {experiment_id}/{len(combinations)}")
    print("n_factors:", n_factors)
    print("learning_rate:", learning_rate)
    print("reg:", reg)
    print("=" * 60)

    model = MatrixFactorization(
        n_users=len(user_ids),
        n_items=len(beer_ids),
        n_factors=n_factors,
        learning_rate=learning_rate,
        reg=reg,
        n_epochs=MAX_EPOCHS,
        random_state=RANDOM_STATE
    )

    model.fit(
        train_users,
        train_items,
        train_ratings,
        val_user_indices=val_users,
        val_item_indices=val_items,
        val_ratings=val_ratings
    )

    best_epoch = int(np.argmin(model.val_history)) + 1
    best_val_rmse = float(np.min(model.val_history))
    train_rmse_at_best = float(model.history[best_epoch - 1])

    results.append({
        "experiment_id": experiment_id,
        "n_factors": n_factors,
        "learning_rate": learning_rate,
        "reg": reg,
        "best_epoch": best_epoch,
        "best_val_rmse": best_val_rmse,
        "train_rmse_at_best": train_rmse_at_best,
        "final_val_rmse": float(model.val_history[-1])
    })

    # Save progress after each experiment
    pd.DataFrame(results).to_csv(
        OUTPUT_DIR / "tuning_results.csv",
        index=False
    )

# =========================================================
# 7. Compare tuning results
# =========================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="best_val_rmse",
    ascending=True
).reset_index(drop=True)

results_df.to_csv(
    OUTPUT_DIR / "tuning_results.csv",
    index=False
)

print("\n========== TUNING RESULTS ==========")

print(
    results_df[
        [
            "experiment_id",
            "n_factors",
            "learning_rate",
            "reg",
            "best_epoch",
            "best_val_rmse"
        ]
    ].to_string(index=False)
)

# =========================================================
# 8. Save best hyperparameters
# =========================================================

best = results_df.iloc[0]

best_params = {
    "n_factors": int(best["n_factors"]),
    "learning_rate": float(best["learning_rate"]),
    "reg": float(best["reg"]),
    "n_epochs": int(best["best_epoch"]),
    "random_state": RANDOM_STATE,
    "validation_rmse": float(best["best_val_rmse"])
}

with open(
    OUTPUT_DIR / "best_params.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        best_params,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n========== BEST PARAMETERS ==========")

for key, value in best_params.items():
    print(f"{key}: {value}")

print("\nMF-Step4 Completed.")
