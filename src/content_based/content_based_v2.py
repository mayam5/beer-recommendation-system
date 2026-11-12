"""
Content-based Filtering for Beer Recommendation

Training (beer features): BeerAdvocate review text
Test: RateBeer user ratings (input 70% / holdout 30%)

Code structure (same order as [1]-[8] in the console output)
[1] Load data, clean reviews of common beers
[2] Build a TF-IDF vector for each beer
[3] Compute cosine similarity between beers
[4] Prepare test data + prediction/recommendation functions
    - Rating prediction: weighted average over the K most similar beers the user rated
    - Recommendation ranking: dot product of the user profile and beer vectors
[5] Select K and ranking method (part of the input data used for validation)
[6] Evaluate models (RMSE, MAE, Recall@N, NDCG@N)
[7] Compare review counts of holdout beers
[8] Recommendation example

---------------------------------------------------------------------
Libraries/functions not covered in class
- pyarrow.parquet.ParquetFile(path).iter_batches(batch_size): read a large parquet file in chunks (saves memory)
- TfidfVectorizer max_features: max vocabulary size / sublinear_tf: use 1+log(tf) instead of tf
- sklearn.preprocessing.normalize(X, norm="l2"): scale each row vector to length 1
- scipy.sparse.csr_matrix, hstack: store sparse matrices and concatenate them column-wise
- numpy.fill_diagonal: set the diagonal (self-similarity) to 0
- numpy.argpartition: find top-k positions without a full sort
- numpy.put_along_axis: put values at given positions in each row
- numpy.where, numpy.clip: choose values by condition / cut values outside a range to the boundary
- numpy.random.default_rng(seed).permutation: seeded shuffle (validation split)
- numpy.log2, arange: rank discounts for NDCG
- pandas.get_dummies: one-hot encoding of beer style
- pandas groupby.transform("median"): fill missing ABV with the median of each style
- pandas groupby.apply(" ".join), reindex: join reviews per beer, align beer order

References
- Lecture slides, Machine Learning Module 6 Recommendation, Module 9 (TF-IDF)
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
- https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
- https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
- https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html
- NDCG: Järvelin & Kekäläinen (2002), Cumulated gain-based evaluation of IR techniques
"""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# =====================================================================
# Settings
# Experiment parameters, paths
# =====================================================================
K_CANDIDATES = [3, 5, 10, 20, 30, 50]    # K candidates chosen by validation
TOP_N_LIST = [5, 10, 20]
VAL_RATIO = 0.2                          # fraction of input data used for validation
W_STYLE = 0.5                            # weights for the comparison model (style + ABV)
W_ABV = 0.2
SEED = 42

plt.rcParams["axes.unicode_minus"] = False

# Use the folder containing data/final_data as the project root
ROOT = Path(__file__).resolve().parent
while not (ROOT / "data" / "final_data").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
DATA = ROOT / "data" / "final_data"
OUT = ROOT / "results" / "content_based"
OUT.mkdir(parents=True, exist_ok=True)


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# =====================================================================
# [1] Load data
#   - Fix the index of the 2,095 common beers by beer_mapping order
#   - Use only BeerAdvocate reviews of common beers
#   - Remove empty reviews and identical reviews by the same user on the same beer
# =====================================================================
mapping = pd.read_parquet(DATA / "beer_mapping.parquet")
mapping = mapping.sort_values("canonical_beer_id").reset_index(drop=True)
beer_ids = mapping["canonical_beer_id"].tolist()
beer_index = {b: i for i, b in enumerate(beer_ids)}   # common beer id -> row index
n_items = len(beer_ids)
ba_to_canon = dict(zip(mapping["ba_beer_id"].astype(str), mapping["canonical_beer_id"]))

train_inter = pd.read_parquet(DATA / "train" / "beeradvocate_train_interactions.parquet")
test_users = pd.read_parquet(DATA / "test" / "ratebeer_test_users.parquet")
test_input = pd.read_parquet(DATA / "test" / "ratebeer_test_input.parquet")
test_holdout = pd.read_parquet(DATA / "test" / "ratebeer_test_holdout.parquet")

# If the common-beer review file does not exist, extract it from the full reviews (682MB) and save
common_path = DATA / "train" / "beeradvocate_reviews_common.parquet"
if not common_path.exists():
    pf = pq.ParquetFile(DATA / "train" / "beeradvocate_reviews.parquet")
    parts = []
    for batch in pf.iter_batches(batch_size=200_000):
        b = batch.to_pandas()
        parts.append(b[b["beer_id"].astype(str).isin(ba_to_canon)])
    pd.concat(parts, ignore_index=True).to_parquet(common_path, index=False)

reviews = pd.read_parquet(common_path, columns=["beer_id", "user_id", "review_text"])
reviews["canonical_beer_id"] = reviews["beer_id"].astype(str).map(ba_to_canon)
n_raw = len(reviews)

reviews = reviews[reviews["review_text"].fillna("").str.strip() != ""]
reviews = reviews.drop_duplicates(subset=["user_id", "canonical_beer_id", "review_text"])

review_count = reviews.groupby("canonical_beer_id").size().reindex(beer_ids, fill_value=0)
section("[1] Data")
print(f"- Common beer reviews: {n_raw:,} -> {len(reviews):,} (empty/duplicates removed)")
print(f"- Median reviews per beer: {review_count.median():.0f}")
print(f"- Share of beers with 3 or fewer reviews: {(review_count <= 3).mean():.1%}")


# =====================================================================
# [2] Beer feature vectors (TF-IDF)
#   - Join all reviews of a beer so that 1 beer = 1 document
#   - Build a (beers x words) matrix with TF-IDF
#   - Also build a TF-IDF + style (one-hot) + ABV matrix for the comparison model
# =====================================================================
docs = (reviews.groupby("canonical_beer_id")["review_text"]
        .apply(" ".join)
        .reindex(beer_ids, fill_value=""))

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    min_df=5,          # drop words that appear in fewer than 5 beers
    max_df=0.8,        # drop words that appear in 80%+ of beers (beer, taste, ...)
    sublinear_tf=True,
)
X_text = tfidf.fit_transform(docs)     # returned with each row L2-normalized
vocab = np.array(tfidf.get_feature_names_out())
section("[2] Beer features (TF-IDF)")
print(f"- TF-IDF matrix: {X_text.shape[0]:,} beers x {X_text.shape[1]:,} words")

# Check the top words of the 3 most-reviewed beers
print("\nTop words of the 3 most-reviewed beers")
for b in review_count.sort_values(ascending=False).index[:3]:
    i = beer_index[b]
    top_words = vocab[X_text[i].toarray().ravel().argsort()[::-1][:10]]
    print(f"- {mapping.loc[i, 'ba_beer_name']} ({mapping.loc[i, 'ba_style']})")
    print(f"    {', '.join(top_words)}")

# Comparison model: TF-IDF + style one-hot + ABV
style_onehot = sparse.csr_matrix(pd.get_dummies(mapping["ba_style"]).values.astype(float))
abv = mapping["ba_abv"].fillna(mapping.groupby("ba_style")["ba_abv"].transform("median"))
abv = abv.fillna(abv.median())                      # missing ABV filled with the style median
abv_z = ((abv - abv.mean()) / abv.std()).clip(-3, 3).values.reshape(-1, 1)
X_combo = sparse.hstack([X_text, W_STYLE * style_onehot, W_ABV * sparse.csr_matrix(abv_z)])
X_combo = normalize(X_combo.tocsr())


# =====================================================================
# [3] Beer-to-beer similarity
#   - Vectors have length 1, so dot product = cosine similarity
#   - Self-similarity (diagonal) is set to 0
# =====================================================================
S_text = (X_text @ X_text.T).toarray()
S_combo = (X_combo @ X_combo.T).toarray()
np.fill_diagonal(S_text, 0)
np.fill_diagonal(S_combo, 0)

section("[3] Similar beer example (cosine similarity)")
i = beer_index[review_count.idxmax()]               # the most-reviewed beer
print(f"Query: {mapping.loc[i, 'ba_beer_name']} ({mapping.loc[i, 'ba_style']} / most reviewed)")
for rank, j in enumerate(S_text[i].argsort()[::-1][:5], start=1):
    print(f"  {rank}. {S_text[i, j]:.3f}  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")


# =====================================================================
# [4] Test data + prediction/recommendation functions
#   - Group each user's data into (beer index, rating) arrays
#   - predict_ratings: rating prediction (similarity-weighted average)
#   - profile_scores: ranking score using both liked and disliked beers
#   - liked_profile_scores: ranking score using liked beers only
# =====================================================================
def to_user_dict(df, with_positive=False):
    """user_id -> [beer index array, rating array(, positive flag array)]"""
    df = df.assign(idx=df["canonical_beer_id"].map(beer_index))
    out = {}
    for u, g in df.groupby("user_id"):
        item = [g["idx"].values, g["rating"].values.astype(float)]
        if with_positive:
            item.append(g["is_positive"].values.astype(bool))
        out[u] = item
    return out


def predict_ratings(S, in_idx, in_r, k):
    """
    R(m,u) = sum_j S(m,j) R(j,u) / sum_j S(m,j)
    j: the k beers most similar to m among the beers the user rated
    If there is no similar beer, use the user's mean rating
    """
    sims = S[:, in_idx].copy()
    sims[sims < 0] = 0
    if sims.shape[1] > k:
        # keep only the top k per row, set the other similarities to 0
        drop = np.argpartition(sims, -k, axis=1)[:, :-k]
        np.put_along_axis(sims, drop, 0, axis=1)
    w_sum = sims.sum(axis=1)
    return np.where(w_sum > 0, (sims @ in_r) / np.maximum(w_sum, 1e-12), in_r.mean())


def profile_scores(S, in_idx, in_r):
    """
    user profile = sum_j (R(j,u) - user mean) * x_j
    score = x_m · profile = sum_j S(m,j) (R(j,u) - user mean)
    + if similar to beers rated above the mean, - if similar to beers rated below it
    """
    return S[:, in_idx] @ (in_r - in_r.mean())


def liked_profile_scores(S, in_idx, in_r):
    """
    User profile using liked beers only
    Beers rated above the mean are weighted by (rating - mean); beers at or below the mean get 0
    """
    return S[:, in_idx] @ np.clip(in_r - in_r.mean(), 0, None)


eval_users = test_users.loc[test_users["evaluable"], "user_id"].tolist()
inputs = to_user_dict(test_input)
holdout = to_user_dict(test_holdout, with_positive=True)
section("[4] Test data (RateBeer)")
print(f"- Users evaluated: {len(eval_users):,}")
print(f"- Mean input beers per user: {np.mean([len(inputs[u][0]) for u in eval_users]):.1f}")


# =====================================================================
# [5] Select K and ranking method
#   - Holdout is not used; 20% of the input data is split off for validation
#   - K: lowest validation RMSE
#   - Ranking method: highest validation Recall@10
# =====================================================================
rng = np.random.default_rng(SEED)
val_split = {}
for u in eval_users:
    in_idx, in_r = inputs[u]
    perm = rng.permutation(len(in_idx))
    n_val = max(1, int(round(len(in_idx) * VAL_RATIO)))
    val_split[u] = (perm[n_val:], perm[:n_val])   # (train positions, validation positions)

# Select K
val_rmse = {}
for k in K_CANDIDATES:
    se, cnt = 0.0, 0
    for u in eval_users:
        in_idx, in_r = inputs[u]
        tr, va = val_split[u]
        pred = predict_ratings(S_text, in_idx[tr], in_r[tr], k)
        se += ((pred[in_idx[va]] - in_r[va]) ** 2).sum()
        cnt += len(va)
    val_rmse[k] = np.sqrt(se / cnt)

BEST_K = min(val_rmse, key=val_rmse.get)


# Select ranking method (validation beers rated 3.5+ count as positives)
def val_recall_at10(rank_fn):
    recalls = []
    for u in eval_users:
        in_idx, in_r = inputs[u]
        tr, va = val_split[u]
        positives = in_idx[va][in_r[va] >= 3.5]
        if len(positives) == 0:
            continue
        score = rank_fn(S_text, in_idx[tr], in_r[tr]).astype(float)
        score[in_idx[tr]] = -np.inf
        top10 = np.argpartition(-score, 10)[:10]
        recalls.append(np.isin(top10, positives).sum() / len(positives))
    return np.mean(recalls)


rank_methods = {"like+dislike": profile_scores, "like only": liked_profile_scores}
val_rank = {name: val_recall_at10(fn) for name, fn in rank_methods.items()}
BEST_RANK = max(val_rank, key=val_rank.get)

section("[5] Select K and ranking method (validation data)")
print("K selection (RMSE, lower is better)")
for k, v in val_rmse.items():
    print(f"  K={k:>2}   {v:.4f}" + ("   <- selected" if k == BEST_K else ""))
print("\nRanking method selection (Recall@10, higher is better)")
for name, v in val_rank.items():
    print(f"  {v:.4f}   {name}" + ("   <- selected" if name == BEST_RANK else ""))

plt.figure(figsize=(6, 4))
plt.plot(list(val_rmse.keys()), list(val_rmse.values()), marker="o")
plt.axvline(BEST_K, color="gray", linestyle="--")
plt.xlabel("K (number of similar beers used)")
plt.ylabel("Validation RMSE")
plt.title("K selection (validation data)")
plt.tight_layout()
plt.savefig(OUT / "content_based_k_selection.png", dpi=150)
plt.close()


# =====================================================================
# [6] Model evaluation
#   - Predict/recommend from the input (70%) and score on the holdout (30%)
#   - Rating prediction: RMSE, MAE
#   - Top-N recommendation: Recall@N, NDCG@N (holdout beers rated 3.5+ count as hits)
#   - Beers in the input are excluded from recommendation candidates
# =====================================================================
popularity = (train_inter.groupby("canonical_beer_id").size()
              .reindex(beer_ids, fill_value=0).values.astype(float))


def evaluate(name, rating_fn, rank_fn):
    """rating_fn, rank_fn: (in_idx, in_r) -> predicted rating / ranking score per beer"""
    sq, ab, cnt = 0.0, 0.0, 0
    max_n = max(TOP_N_LIST)
    recall = {n: [] for n in TOP_N_LIST}
    ndcg = {n: [] for n in TOP_N_LIST}
    discount = 1 / np.log2(np.arange(2, max_n + 2))

    for u in eval_users:
        in_idx, in_r = inputs[u]
        ho_idx, ho_r, ho_pos = holdout[u]

        # Rating prediction error
        err = rating_fn(in_idx, in_r)[ho_idx] - ho_r
        sq += (err ** 2).sum()
        ab += np.abs(err).sum()
        cnt += len(err)

        # Top-N recommendation
        score = rank_fn(in_idx, in_r).astype(float)
        score[in_idx] = -np.inf
        top = np.argpartition(-score, max_n)[:max_n]
        top = top[np.argsort(-score[top])]
        positives = ho_idx[ho_pos]
        hit = np.isin(top, positives).astype(float)
        for n in TOP_N_LIST:
            recall[n].append(hit[:n].sum() / len(positives))
            idcg = discount[:min(n, len(positives))].sum()
            ndcg[n].append((hit[:n] * discount[:n]).sum() / idcg)

    row = {"model": name, "RMSE": np.sqrt(sq / cnt), "MAE": ab / cnt}
    for n in TOP_N_LIST:
        row[f"Recall@{n}"] = np.mean(recall[n])
        row[f"NDCG@{n}"] = np.mean(ndcg[n])
    return row


t0 = time.time()
results = pd.DataFrame([
    evaluate("Baseline",
             lambda i, r: np.full(n_items, r.mean()),
             lambda i, r: popularity.copy()),
    evaluate("CB-TFIDF",
             lambda i, r: predict_ratings(S_text, i, r, BEST_K),
             lambda i, r: profile_scores(S_text, i, r)),
    evaluate("CB-TFIDF(pred-rank)",
             lambda i, r: predict_ratings(S_text, i, r, BEST_K),
             lambda i, r: predict_ratings(S_text, i, r, BEST_K)),
    evaluate("CB-TFIDF+style+ABV",
             lambda i, r: predict_ratings(S_combo, i, r, BEST_K),
             lambda i, r: profile_scores(S_combo, i, r)),
    evaluate("CB-TFIDF(liked)",
             lambda i, r: predict_ratings(S_text, i, r, BEST_K),
             lambda i, r: liked_profile_scores(S_text, i, r)),
]).set_index("model")
eval_time = time.time() - t0

pd.set_option("display.width", 200)
shown = results.round(4).rename_axis(None)       # drop the index name (model) line when printing
section(f"[6] Evaluation results ({len(eval_users):,} RateBeer users)")
print("Rating prediction (lower is better)")
print(shown[["RMSE", "MAE"]].to_string())
print("\nTop-N recommendation - Recall (higher is better)")
print(shown[[f"Recall@{n}" for n in TOP_N_LIST]].to_string())
print("\nTop-N recommendation - NDCG (higher is better)")
print(shown[[f"NDCG@{n}" for n in TOP_N_LIST]].to_string())
# Mark the final model according to the ranking method selected in [5]
final_model = {"like+dislike": "CB-TFIDF", "like only": "CB-TFIDF(liked)"}[BEST_RANK]
model_desc = {
    "Baseline": "rating = user mean, ranking = most-rated beers first",
    "CB-TFIDF": "review TF-IDF, ranking uses both liked and disliked beers",
    "CB-TFIDF(pred-rank)": "same as CB-TFIDF, but ranked by predicted rating",
    "CB-TFIDF+style+ABV": "CB-TFIDF plus beer style + ABV",
    "CB-TFIDF(liked)": "review TF-IDF, ranking uses liked beers only",
}
print("\nModel descriptions")
for name, desc in model_desc.items():
    mark = "   <- final (selected by validation)" if name == final_model else ""
    print(f"- {name:<20}: {desc}{mark}")
print(f"(evaluation time: {eval_time:.1f}s)")
results.round(4).to_csv(OUT / "content_based_results.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
results[["RMSE", "MAE"]].plot.bar(ax=axes[0], rot=20, title="Rating prediction error (lower is better)")
results[[f"Recall@{n}" for n in TOP_N_LIST]].plot.bar(ax=axes[1], rot=20, title="Recall@N")
results[[f"NDCG@{n}" for n in TOP_N_LIST]].plot.bar(ax=axes[2], rot=20, title="NDCG@N")
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(OUT / "content_based_metrics.png", dpi=150)
plt.close()


# =====================================================================
# [7] Review counts of holdout beers
#   - Check whether the beers users actually liked are mostly well-reviewed (popular) ones
#   - Explains why the Baseline (most-rated first) scores high
# =====================================================================
pos_beers = test_holdout[test_holdout["user_id"].isin(eval_users) & test_holdout["is_positive"]]
pos_review_cnt = review_count.reindex(pos_beers["canonical_beer_id"])
section("[7] Review count comparison")
print(f"- Median review count, all common beers: {review_count.median():.0f}")
print(f"- Median review count, beers users actually liked (holdout, 3.5+): {pos_review_cnt.median():.0f}")


# =====================================================================
# [8] Recommendation example
#   - Among users with 40 or fewer input beers, the one with the most holdout beers
#   - Print 5 top-rated beers and the Top 10 from the final ranking method
# =====================================================================
u = max(eval_users, key=lambda x: len(holdout[x][0]) if len(inputs[x][0]) <= 40 else 0)
in_idx, in_r = inputs[u]
ho_idx, ho_r, ho_pos = holdout[u]

section(f"[8] Recommendation example (user: {u}, ranking method: {BEST_RANK})")
print("Top-rated beers")
for pos in np.argsort(-in_r)[:5]:
    j = in_idx[pos]
    print(f"  {in_r[pos]:.2f}  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")

pred = predict_ratings(S_text, in_idx, in_r, BEST_K)
score = rank_methods[BEST_RANK](S_text, in_idx, in_r)
score[in_idx] = -np.inf
positive_set = set(ho_idx[ho_pos])
print("\nTop 10 recommendations  (* = rated 3.5+ in holdout / pred = predicted rating)")
for rank, j in enumerate(np.argsort(-score)[:10], start=1):
    mark = "*" if j in positive_set else " "
    print(f" {rank:>2}. {mark} pred {pred[j]:.2f}  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")

print("\nResults saved to:", OUT)
