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
RECOMMENDATION_DIR = RESULTS_DIR / "recommendations"
OUTPUT_DIR = RESULTS_DIR / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RECOMMENDATION_PATH = (
    RECOMMENDATION_DIR / "top10_recommendations.parquet"
)

HOLDOUT_PATH = (
    DATA_DIR / "test" / "ratebeer_test_holdout.parquet"
)

USERS_PATH = (
    DATA_DIR / "test" / "ratebeer_test_users.parquet"
)

MODEL_PATH = MODEL_DIR / "mf_model.npz"
FOLD_IN_PATH = MODEL_DIR / "ratebeer_user_factors.npz"
BEER_INDEX_PATH = MODEL_DIR / "beer_index.parquet"

RESULT_PATH = OUTPUT_DIR / "evaluation_results.csv"
USER_METRICS_PATH = OUTPUT_DIR / "user_metrics.parquet"

K_VALUES = [5, 10]

# =========================================================
# 2. Load evaluation data
# =========================================================

recommendations = pd.read_parquet(RECOMMENDATION_PATH)
holdout = pd.read_parquet(HOLDOUT_PATH)
test_users = pd.read_parquet(USERS_PATH)

print("========== EVALUATION DATA ==========")
print("Recommendations:", len(recommendations))
print("Holdout interactions:", len(holdout))
print("Test users:", len(test_users))

# =========================================================
# 3. Validate input data
# =========================================================

required_rec_columns = {
    "user_id", "canonical_beer_id", "score", "rank"
}

required_holdout_columns = {
    "user_id", "canonical_beer_id", "is_positive", "rating"
}

required_user_columns = {
    "user_id", "evaluable"
}

assert required_rec_columns.issubset(
    recommendations.columns
)

assert required_holdout_columns.issubset(
    holdout.columns
)

assert required_user_columns.issubset(
    test_users.columns
)

assert not recommendations.duplicated(
    ["user_id", "canonical_beer_id"]
).any()

assert not recommendations.duplicated(
    ["user_id", "rank"]
).any()

assert not holdout.duplicated(
    ["user_id", "canonical_beer_id"]
).any()

assert not test_users["user_id"].duplicated().any()

assert recommendations["rank"].between(1, 10).all()

assert recommendations["score"].notna().all()

assert (
    recommendations.groupby("user_id").size() == 10
).all()

print("[PASS] Evaluation input validated")

# =========================================================
# 4. Select evaluable users
# =========================================================

evaluable_users = test_users.loc[
    test_users["evaluable"] == True,
    "user_id"
].tolist()

evaluable_user_set = set(evaluable_users)

print("\n========== EVALUABLE USERS ==========")
print("Evaluable users:", len(evaluable_users))

assert len(evaluable_users) > 0

assert evaluable_user_set.issubset(
    set(recommendations["user_id"])
)

# =========================================================
# 5. Prepare positive holdout items
# =========================================================

positive_holdout = holdout[
    holdout["is_positive"] == True
].copy()

positive_items = (
    positive_holdout
    .groupby("user_id")["canonical_beer_id"]
    .apply(set)
    .to_dict()
)

print("Positive holdout interactions:", len(positive_holdout))

assert all(
    len(positive_items.get(user_id, set())) > 0
    for user_id in evaluable_users
)

print("[PASS] Positive holdout prepared")

# =========================================================
# 6. Prepare ranked recommendation lists
# =========================================================

recommendations = recommendations.sort_values(
    ["user_id", "rank"]
)

recommended_items = (
    recommendations
    .groupby("user_id")["canonical_beer_id"]
    .apply(list)
    .to_dict()
)

# =========================================================
# 7. Define evaluation metrics
# =========================================================

def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]

    hits = sum(
        item in relevant
        for item in top_k
    )

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0

    top_k = recommended[:k]

    relevance = np.array([
        1.0 if item in relevant else 0.0
        for item in top_k
    ])

    positions = np.arange(1, len(top_k) + 1)

    discounts = 1.0 / np.log2(positions + 1)

    dcg = np.sum(relevance * discounts)

    ideal_count = min(len(relevant), k)

    ideal_positions = np.arange(
        1, ideal_count + 1
    )

    idcg = np.sum(
        1.0 / np.log2(ideal_positions + 1)
    )

    return float(dcg / idcg)

# =========================================================
# 8. Evaluate each user
# =========================================================

print("\n========== EVALUATING USERS ==========")

user_results = []

for user_id in evaluable_users:

    recommended = recommended_items[user_id]
    relevant = positive_items[user_id]

    result = {
        "user_id": user_id,
        "n_positive_holdout": len(relevant)
    }

    for k in K_VALUES:
        result[f"recall@{k}"] = recall_at_k(
            recommended,
            relevant,
            k
        )

        result[f"ndcg@{k}"] = ndcg_at_k(
            recommended,
            relevant,
            k
        )

    user_results.append(result)

user_metrics = pd.DataFrame(user_results)

assert len(user_metrics) == len(evaluable_users)

metric_columns = [
    "recall@5",
    "recall@10",
    "ndcg@5",
    "ndcg@10"
]

assert user_metrics[metric_columns].notna().all().all()

assert (
    (user_metrics[metric_columns] >= 0)
    & (user_metrics[metric_columns] <= 1)
).all().all()

print("[PASS] User-level metrics calculated")

# =========================================================
# 9. Calculate RateBeer holdout rating metrics
# =========================================================

print("\n========== RATING PREDICTION EVALUATION ==========")

beer_index = pd.read_parquet(BEER_INDEX_PATH)

with np.load(MODEL_PATH) as saved:
    item_factors = saved["item_factors"].copy()
    item_bias = saved["item_bias"].copy()
    global_mean = float(saved["global_mean"])

with np.load(FOLD_IN_PATH) as saved:
    user_factors = saved["user_factors"].copy()
    user_bias = saved["user_bias"].copy()
    user_ids = saved["user_ids"].astype(str)

assert len(set(user_ids)) == len(user_ids)
assert not beer_index["canonical_beer_id"].duplicated().any()
assert not beer_index["item_idx"].duplicated().any()
assert len(beer_index) == len(item_factors) == len(item_bias)
assert len(user_ids) == len(user_factors) == len(user_bias)
assert user_factors.shape[1] == item_factors.shape[1]
assert beer_index["item_idx"].between(0, len(item_factors) - 1).all()

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(user_ids)
}

beer_to_idx = dict(zip(
    beer_index["canonical_beer_id"],
    beer_index["item_idx"]
))

holdout_user_indices = holdout["user_id"].astype(str).map(user_to_idx)
holdout_item_indices = holdout["canonical_beer_id"].map(beer_to_idx)

assert len(holdout) > 0
assert holdout_user_indices.notna().all()
assert holdout_item_indices.notna().all()
assert holdout["rating"].notna().all()

u = holdout_user_indices.to_numpy(dtype=np.int64)
i = holdout_item_indices.to_numpy(dtype=np.int64)
actual = holdout["rating"].to_numpy(dtype=np.float64)

assert np.isfinite(actual).all()

# Same un-clipped prediction formula as model.py.
predicted = (
    global_mean
    + user_bias[u]
    + item_bias[i]
    + np.sum(
        user_factors[u] * item_factors[i],
        axis=1
    )
)

assert np.isfinite(predicted).all()

errors = actual - predicted
mae = float(np.mean(np.abs(errors)))
mse = float(np.mean(errors ** 2))
rmse = float(np.sqrt(mse))

print("Test interactions:", len(holdout))
print(f"Test MAE: {mae:.6f}")
print(f"Test MSE: {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print("[PASS] Rating prediction metrics calculated")

# =========================================================
# 10. Calculate mean evaluation metrics
# =========================================================

mean_metrics = user_metrics[metric_columns].mean()

evaluation_results = pd.DataFrame([{
    "model": "MF_SGD_FoldIn",
    "n_evaluable_users": len(evaluable_users),
    "n_test_interactions": len(holdout),
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "recall@5": float(mean_metrics["recall@5"]),
    "recall@10": float(mean_metrics["recall@10"]),
    "ndcg@5": float(mean_metrics["ndcg@5"]),
    "ndcg@10": float(mean_metrics["ndcg@10"])
}])

# =========================================================
# 11. Save evaluation results
# =========================================================

evaluation_results.to_csv(
    RESULT_PATH,
    index=False
)

user_metrics.to_parquet(
    USER_METRICS_PATH,
    index=False
)

print("[PASS] Evaluation results saved")
print("[PASS] User-level metrics saved")

# =========================================================
# 12. Display final results
# =========================================================

print("\n========== FINAL MF EVALUATION ==========")

print("Model: MF_SGD_FoldIn")
print("Evaluable users:", len(evaluable_users))
print("Test interactions:", len(holdout))
print(f"MAE: {mae:.6f}")
print(f"MSE: {mse:.6f}")
print(f"RMSE: {rmse:.6f}")

for metric in metric_columns:
    print(
        f"{metric}: {mean_metrics[metric]:.6f}"
    )

print("\n========== OUTPUT FILES ==========")
print(RESULT_PATH)
print(USER_METRICS_PATH)

print("\nMF-Step8 Completed.")
