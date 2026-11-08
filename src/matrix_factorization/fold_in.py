from pathlib import Path
import json

import numpy as np
import pandas as pd

# =========================================================
# 1. Set project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "final_data"

RESULTS_DIR = PROJECT_DIR / "results" / "matrix_factorization"

MODEL_DIR = RESULTS_DIR / "final_model"

INPUT_PATH = (
    DATA_DIR / "test" / "ratebeer_test_input.parquet"
)

USERS_PATH = (
    DATA_DIR / "test" / "ratebeer_test_users.parquet"
)

MODEL_PATH = MODEL_DIR / "mf_model.npz"
BEER_INDEX_PATH = MODEL_DIR / "beer_index.parquet"

OUTPUT_PATH = MODEL_DIR / "ratebeer_user_factors.npz"
INFO_PATH = MODEL_DIR / "fold_in_info.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# 2. Load trained MF model
# =========================================================

with np.load(MODEL_PATH) as saved:
    item_factors = saved["item_factors"].copy()
    item_bias = saved["item_bias"].copy()
    global_mean = float(saved["global_mean"])

n_items, n_factors = item_factors.shape

print("========== TRAINED MF MODEL ==========")
print("Items:", n_items)
print("Latent factors:", n_factors)
print("Global mean:", round(global_mean, 4))

# =========================================================
# 3. Load RateBeer input data
# =========================================================

rb_input = pd.read_parquet(INPUT_PATH)
rb_users = pd.read_parquet(USERS_PATH)

beer_index = pd.read_parquet(BEER_INDEX_PATH)

print("\n========== RATEBEER INPUT ==========")
print("Input interactions:", len(rb_input))
print("Input users:", rb_input["user_id"].nunique())
print("Test users:", len(rb_users))

# =========================================================
# 4. Map beer IDs to MF item indices
# =========================================================

beer_to_idx = dict(
    zip(
        beer_index["canonical_beer_id"],
        beer_index["item_idx"]
    )
)

rb_input["item_idx"] = (
    rb_input["canonical_beer_id"].map(beer_to_idx)
)

assert not rb_input["item_idx"].isna().any()

rb_input["item_idx"] = rb_input["item_idx"].astype(int)

assert rb_input["item_idx"].between(
    0, n_items - 1
).all()

assert np.isfinite(rb_input["rating"]).all()

# =========================================================
# 5. Prepare user indices
# =========================================================

user_ids = sorted(rb_users["user_id"].unique())

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

n_users = len(user_ids)

user_factors = np.zeros(
    (n_users, n_factors),
    dtype=np.float64
)

user_bias = np.zeros(
    n_users,
    dtype=np.float64
)

# =========================================================
# 6. Configure Fold-in regularization
# =========================================================

# Fixed regularization values for the experiment.
FACTOR_REG = 1.0
BIAS_REG = 1.0

# =========================================================
# 7. Estimate new user factors
# =========================================================

print("\n========== FOLD-IN ==========")

grouped = rb_input.groupby("user_id")

for user_id, group in grouped:

    u = user_to_idx[user_id]

    item_indices = group["item_idx"].to_numpy(
        dtype=np.int32
    )

    ratings = group["rating"].to_numpy(
        dtype=np.float64
    )

    # Fixed BeerAdvocate item latent factors
    Q = item_factors[item_indices]

    # Residual ratings after global mean and item bias
    y = ratings - global_mean - item_bias[item_indices]

    # Design matrix:
    # First column = user bias
    # Remaining columns = user latent factors
    X = np.column_stack([
        np.ones(len(ratings)),
        Q
    ])

    # Regularization matrix
    regularization = np.diag(
        [BIAS_REG] + [FACTOR_REG] * n_factors
    )

    # Ridge regression solution
    A = X.T @ X + regularization
    b = X.T @ y

    solution = np.linalg.solve(A, b)

    # Save user bias and latent factors
    user_bias[u] = solution[0]
    user_factors[u] = solution[1:]

# =========================================================
# 8. Validate Fold-in results
# =========================================================

print("\n========== VALIDATION ==========")

assert np.isfinite(user_factors).all()
assert np.isfinite(user_bias).all()

assert user_factors.shape == (n_users, n_factors)
assert user_bias.shape == (n_users,)

assert set(user_ids) == set(rb_input["user_id"].unique())

print("[PASS] All RateBeer users processed")
print("[PASS] User factor dimensions valid")
print("[PASS] All user factors are finite")

# =========================================================
# 9. Save Fold-in user factors
# =========================================================

np.savez_compressed(
    OUTPUT_PATH,
    user_factors=user_factors,
    user_bias=user_bias,
    user_ids=np.array(user_ids, dtype=str)
)

print("\n[PASS] Fold-in user factors saved")

# =========================================================
# 10. Save Fold-in information
# =========================================================

fold_in_info = {
    "source_model": "BeerAdvocate MF",
    "target_dataset": "RateBeer",
    "method": "Ridge Regression Fold-in",
    "n_users": int(n_users),
    "n_items": int(n_items),
    "n_factors": int(n_factors),
    "factor_reg": FACTOR_REG,
    "bias_reg": BIAS_REG,
    "input_interactions": int(len(rb_input))
}

with open(
    INFO_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        fold_in_info,
        f,
        indent=4,
        ensure_ascii=False
    )

print("[PASS] Fold-in information saved")

# =========================================================
# 11. Display summary
# =========================================================

print("\n========== FOLD-IN SUMMARY ==========")
print("RateBeer users:", n_users)
print("Latent factors per user:", n_factors)
print("User factors shape:", user_factors.shape)
print("User bias shape:", user_bias.shape)

print("\nMF-Step6 Completed.")
