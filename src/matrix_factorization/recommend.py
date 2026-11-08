from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# 1. Set project paths
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_DIR / "data" / "final_data"

RESULTS_DIR = PROJECT_DIR / "results" / "matrix_factorization"

MODEL_DIR = RESULTS_DIR / "final_model"
OUTPUT_DIR = RESULTS_DIR / "recommendations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "mf_model.npz"
FOLD_IN_PATH = MODEL_DIR / "ratebeer_user_factors.npz"
BEER_INDEX_PATH = MODEL_DIR / "beer_index.parquet"

INPUT_PATH = (
    DATA_DIR / "test" / "ratebeer_test_input.parquet"
)

OUTPUT_PATH = (
    OUTPUT_DIR / "top10_recommendations.parquet"
)

TOP_K = 10

# =========================================================
# 2. Load trained MF model
# =========================================================

with np.load(MODEL_PATH) as saved:
    item_factors = saved["item_factors"].copy()
    item_bias = saved["item_bias"].copy()
    global_mean = float(saved["global_mean"])

print("========== TRAINED MF MODEL ==========")
print("Item factors shape:", item_factors.shape)
print("Item bias shape:", item_bias.shape)
print("Global mean:", round(global_mean, 4))

# =========================================================
# 3. Load RateBeer Fold-in results
# =========================================================

with np.load(FOLD_IN_PATH) as saved:
    user_factors = saved["user_factors"].copy()
    user_bias = saved["user_bias"].copy()
    user_ids = saved["user_ids"].astype(str)

print("\n========== RATEBEER USERS ==========")
print("User factors shape:", user_factors.shape)
print("User bias shape:", user_bias.shape)
print("Number of users:", len(user_ids))

# =========================================================
# 4. Load beer indices and RateBeer input
# =========================================================

beer_index = pd.read_parquet(BEER_INDEX_PATH)
rb_input = pd.read_parquet(INPUT_PATH)

beer_index = beer_index.sort_values(
    "item_idx"
).reset_index(drop=True)

beer_ids = beer_index["canonical_beer_id"].to_numpy()

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

# =========================================================
# 5. Validate model dimensions
# =========================================================

n_users, n_factors = user_factors.shape
n_items, item_factors_dim = item_factors.shape

assert n_factors == item_factors_dim
assert len(user_bias) == n_users
assert len(item_bias) == n_items
assert len(beer_ids) == n_items

assert np.array_equal(
    beer_index["item_idx"].to_numpy(),
    np.arange(n_items)
)

assert len(set(user_ids)) == n_users
assert set(user_ids) == set(rb_input["user_id"].unique())

print("\n[PASS] Model dimensions verified")

# =========================================================
# 6. Calculate predicted scores
# =========================================================

# Shape: (n_users, n_items)
scores = (
    global_mean
    + user_bias[:, np.newaxis]
    + item_bias[np.newaxis, :]
    + user_factors @ item_factors.T
)

assert scores.shape == (n_users, n_items)
assert np.isfinite(scores).all()

print("[PASS] Prediction scores calculated")
print("Score matrix shape:", scores.shape)

# =========================================================
# 7. Exclude already observed beers
# =========================================================

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

rb_input["user_idx"] = rb_input["user_id"].map(
    user_to_idx
)

assert not rb_input["user_idx"].isna().any()

observed_user_indices = (
    rb_input["user_idx"].to_numpy(dtype=np.int32)
)

observed_item_indices = (
    rb_input["item_idx"].to_numpy(dtype=np.int32)
)

scores[
    observed_user_indices,
    observed_item_indices
] = -np.inf

print("[PASS] Observed beers excluded")

# =========================================================
# 8. Generate Top-K recommendations
# =========================================================

recommendations = []

for user_idx, user_id in enumerate(user_ids):

    user_scores = scores[user_idx]

    # Sort by descending score.
    # Ties are resolved by ascending item index.
    ranked_indices = np.lexsort((
        np.arange(n_items),
        -user_scores
    ))

    top_indices = ranked_indices[:TOP_K]

    assert np.isfinite(
        user_scores[top_indices]
    ).all()

    for rank, item_idx in enumerate(
        top_indices,
        start=1
    ):
        recommendations.append({
            "user_id": user_id,
            "canonical_beer_id": beer_ids[item_idx],
            "score": float(user_scores[item_idx]),
            "rank": rank
        })

# =========================================================
# 9. Save recommendations
# =========================================================

recommendations_df = pd.DataFrame(
    recommendations
)

recommendations_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("\n[PASS] Recommendations saved")

# =========================================================
# 10. Validate recommendation results
# =========================================================

expected_rows = n_users * TOP_K

assert len(recommendations_df) == expected_rows

assert recommendations_df["user_id"].nunique() == n_users

assert (
    recommendations_df.groupby("user_id").size() == TOP_K
).all()

assert not recommendations_df.duplicated(
    ["user_id", "canonical_beer_id"]
).any()

assert recommendations_df["rank"].between(
    1, TOP_K
).all()

# Check that no input interaction is recommended.
overlap = recommendations_df.merge(
    rb_input[["user_id", "canonical_beer_id"]],
    on=["user_id", "canonical_beer_id"],
    how="inner"
)

assert len(overlap) == 0

print("[PASS] Recommendation count verified")
print("[PASS] No duplicate recommendations")
print("[PASS] No observed beers recommended")

# =========================================================
# 11. Display summary
# =========================================================

print("\n========== RECOMMENDATION SUMMARY ==========")

print("Users:", n_users)
print("Recommendations per user:", TOP_K)
print("Total recommendations:", len(recommendations_df))

print("\n========== SAMPLE RECOMMENDATIONS ==========")

sample_user = user_ids[0]

print(
    recommendations_df[
        recommendations_df["user_id"] == sample_user
    ].to_string(index=False)
)

print("\nMF-Step7 Completed.")


