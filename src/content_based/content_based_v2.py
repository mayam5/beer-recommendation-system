"""
Content-based Filtering 맥주 추천

학습(맥주 특징): BeerAdvocate 리뷰 텍스트
테스트: RateBeer 사용자 평점 (입력 70% / 정답 30%)

코드 구성 (실행 화면의 [1]~[8]과 같은 순서)
[1] 데이터 불러오기, 공통 맥주 리뷰 정리
[2] 맥주별 TF-IDF 벡터 생성
[3] 맥주 간 코사인 유사도 계산
[4] 테스트 데이터 정리 + 예측/추천 함수
    - 평점 예측: 사용자가 평가한 맥주 중 유사한 K개의 가중평균
    - 추천 순위: 사용자 프로필과 맥주 벡터의 내적
[5] K, 추천 방식 선택 (입력 데이터 일부를 검증용으로 사용)
[6] 모델별 평가 (RMSE, MAE, Recall@N, NDCG@N)
[7] 정답 맥주 리뷰 수 비교
[8] 추천 예시

---------------------------------------------------------------------
수업에서 다루지 않은 라이브러리/함수
- pyarrow.parquet.ParquetFile(path).iter_batches(batch_size): 큰 parquet 파일을 나눠 읽음 (메모리 절약)
- TfidfVectorizer의 max_features: 사용할 최대 단어 수 / sublinear_tf: tf 대신 1+log(tf) 사용
- sklearn.preprocessing.normalize(X, norm="l2"): 각 행 벡터를 길이 1로 정규화
- scipy.sparse.csr_matrix, hstack: 희소행렬 저장 및 옆으로 이어붙이기
- numpy.fill_diagonal: 대각선(자기 자신과의 유사도)을 0으로
- numpy.argpartition: 전체 정렬 없이 상위 k개 위치 찾기
- numpy.put_along_axis: 행마다 지정한 위치에 값 넣기
- numpy.where, numpy.clip: 조건에 따라 값 선택 / 범위 밖 값을 경계값으로 자름
- numpy.random.default_rng(seed).permutation: 시드 고정 셔플 (검증셋 분할)
- numpy.log2, arange: NDCG 순위별 할인값 계산
- pandas.get_dummies: 스타일 원핫 인코딩
- pandas groupby.transform("median"): 스타일별 중앙값으로 ABV 결측 채우기
- pandas groupby.apply(" ".join), reindex: 맥주별 리뷰 합치기, 맥주 순서 맞추기

참고
- 강의자료 Machine Learning Module 6 Recommendation, Module 9 (TF-IDF)
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.normalize.html
- https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html
- https://numpy.org/doc/stable/reference/generated/numpy.argpartition.html
- https://numpy.org/doc/stable/reference/generated/numpy.put_along_axis.html
- NDCG: Järvelin & Kekäläinen (2002), Cumulated gain-based evaluation of IR techniques
"""
import platform
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
# 설정
#   실험 파라미터, 경로 설정
# =====================================================================
K_CANDIDATES = [3, 5, 10, 20, 30, 50]   # 검증으로 고를 K 후보
TOP_N_LIST = [5, 10, 20]
VAL_RATIO = 0.2                          # 입력 데이터 중 검증용 비율
W_STYLE = 0.5                            # 비교 모델(스타일+ABV) 가중치
W_ABV = 0.2
SEED = 42

if platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# data/final_data가 있는 폴더를 프로젝트 루트로 사용
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
# [1] 데이터 불러오기
#   - beer_mapping 순서로 공통 맥주 2,095개의 인덱스 고정
#   - BeerAdvocate 리뷰 중 공통 맥주 리뷰만 사용
#   - 빈 리뷰, 같은 사용자가 같은 맥주에 남긴 동일 리뷰 제거
# =====================================================================
mapping = pd.read_parquet(DATA / "beer_mapping.parquet")
mapping = mapping.sort_values("canonical_beer_id").reset_index(drop=True)
beer_ids = mapping["canonical_beer_id"].tolist()
beer_index = {b: i for i, b in enumerate(beer_ids)}   # 공통 맥주 id -> 행 번호
n_items = len(beer_ids)
ba_to_canon = dict(zip(mapping["ba_beer_id"].astype(str), mapping["canonical_beer_id"]))

train_inter = pd.read_parquet(DATA / "train" / "beeradvocate_train_interactions.parquet")
test_users = pd.read_parquet(DATA / "test" / "ratebeer_test_users.parquet")
test_input = pd.read_parquet(DATA / "test" / "ratebeer_test_input.parquet")
test_holdout = pd.read_parquet(DATA / "test" / "ratebeer_test_holdout.parquet")

# 공통 맥주 리뷰 파일이 없으면 전체 리뷰(682MB)에서 추출해서 저장
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
section("[1] 데이터")
print(f"- 공통 맥주 리뷰: {n_raw:,}건 -> {len(reviews):,}건 (빈 리뷰/중복 제거)")
print(f"- 맥주당 리뷰 수 중앙값: {review_count.median():.0f}개")
print(f"- 리뷰 3개 이하 맥주 비율: {(review_count <= 3).mean():.1%}")


# =====================================================================
# [2] 맥주 특징 벡터 (TF-IDF)
#   - 맥주별 리뷰를 하나로 합쳐서 맥주 1개 = 문서 1개로 만듦
#   - TF-IDF로 (맥주 수 x 단어 수) 행렬 생성
#   - 비교 모델용으로 TF-IDF + 스타일(원핫) + ABV 행렬도 생성
# =====================================================================
docs = (reviews.groupby("canonical_beer_id")["review_text"]
        .apply(" ".join)
        .reindex(beer_ids, fill_value=""))

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=20000,
    min_df=5,          # 5개 미만 맥주에서만 나온 단어 제외
    max_df=0.8,        # 80% 이상 맥주에서 나온 단어 제외 (beer, taste 등)
    sublinear_tf=True,
)
X_text = tfidf.fit_transform(docs)     # 행 단위 L2 정규화된 상태로 반환됨
vocab = np.array(tfidf.get_feature_names_out())
section("[2] 맥주 특징 (TF-IDF)")
print(f"- TF-IDF 행렬: 맥주 {X_text.shape[0]:,}개 x 단어 {X_text.shape[1]:,}개")

# 리뷰 많은 맥주 3개의 대표 단어 확인
print("\n리뷰가 가장 많은 맥주 3개의 상위 단어")
for b in review_count.sort_values(ascending=False).index[:3]:
    i = beer_index[b]
    top_words = vocab[X_text[i].toarray().ravel().argsort()[::-1][:10]]
    print(f"- {mapping.loc[i, 'ba_beer_name']} ({mapping.loc[i, 'ba_style']})")
    print(f"    {', '.join(top_words)}")

# 비교 모델: TF-IDF + 스타일 원핫 + ABV
style_onehot = sparse.csr_matrix(pd.get_dummies(mapping["ba_style"]).values.astype(float))
abv = mapping["ba_abv"].fillna(mapping.groupby("ba_style")["ba_abv"].transform("median"))
abv = abv.fillna(abv.median())                      # ABV 결측은 스타일 중앙값으로 대체
abv_z = ((abv - abv.mean()) / abv.std()).clip(-3, 3).values.reshape(-1, 1)
X_combo = sparse.hstack([X_text, W_STYLE * style_onehot, W_ABV * sparse.csr_matrix(abv_z)])
X_combo = normalize(X_combo.tocsr())


# =====================================================================
# [3] 맥주 간 유사도
#   - 벡터 길이가 1이라 내적 = 코사인 유사도
#   - 자기 자신과의 유사도(대각선)는 0으로 처리
# =====================================================================
S_text = (X_text @ X_text.T).toarray()
S_combo = (X_combo @ X_combo.T).toarray()
np.fill_diagonal(S_text, 0)
np.fill_diagonal(S_combo, 0)

section("[3] 유사한 맥주 예시 (코사인 유사도)")
i = beer_index[review_count.idxmax()]               # 리뷰 수 1위 맥주 기준
print(f"기준: {mapping.loc[i, 'ba_beer_name']} ({mapping.loc[i, 'ba_style']} / 리뷰 수 1위)")
for rank, j in enumerate(S_text[i].argsort()[::-1][:5], start=1):
    print(f"  {rank}. {S_text[i, j]:.3f}  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")


# =====================================================================
# [4] 테스트 데이터 정리 + 예측/추천 함수
#   - 사용자별로 (맥주 인덱스, 평점) 배열로 묶어둠
#   - predict_ratings: 평점 예측 (유사도 가중평균)
#   - profile_scores: 추천 점수, 좋아한/싫어한 맥주 모두 반영
#   - liked_profile_scores: 추천 점수, 좋아한 맥주만 반영
# =====================================================================
def to_user_dict(df, with_positive=False):
    """user_id -> [맥주 인덱스 배열, 평점 배열(, 긍정 여부 배열)]"""
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
    j: 사용자가 평가한 맥주 중 m과 가장 유사한 k개
    유사한 맥주가 없으면 사용자 평균 평점
    """
    sims = S[:, in_idx].copy()
    sims[sims < 0] = 0
    if sims.shape[1] > k:
        # 행마다 상위 k개만 남기고 나머지 유사도는 0
        drop = np.argpartition(sims, -k, axis=1)[:, :-k]
        np.put_along_axis(sims, drop, 0, axis=1)
    w_sum = sims.sum(axis=1)
    return np.where(w_sum > 0, (sims @ in_r) / np.maximum(w_sum, 1e-12), in_r.mean())


def profile_scores(S, in_idx, in_r):
    """
    사용자 프로필 = sum_j (R(j,u) - 사용자 평균) * x_j
    점수 = x_m · 프로필 = sum_j S(m,j) (R(j,u) - 사용자 평균)
    평균보다 높게 준 맥주와 비슷할수록 +, 낮게 준 맥주와 비슷할수록 -
    """
    return S[:, in_idx] @ (in_r - in_r.mean())


def liked_profile_scores(S, in_idx, in_r):
    """
    좋아한 맥주만 반영한 사용자 프로필
    평균보다 높게 준 맥주만 (평점 - 평균)만큼 가중치, 평균 이하로 준 맥주는 0
    """
    return S[:, in_idx] @ np.clip(in_r - in_r.mean(), 0, None)


eval_users = test_users.loc[test_users["evaluable"], "user_id"].tolist()
inputs = to_user_dict(test_input)
holdout = to_user_dict(test_holdout, with_positive=True)
section("[4] 테스트 데이터 (RateBeer)")
print(f"- 평가 대상 사용자: {len(eval_users):,}명")
print(f"- 사용자당 평균 입력 맥주: {np.mean([len(inputs[u][0]) for u in eval_users]):.1f}개")


# =====================================================================
# [5] K, 추천 방식 선택
#   - 정답(holdout)은 쓰지 않고, 입력 데이터의 20%를 검증용으로 분리
#   - K: 검증 RMSE가 가장 낮은 값
#   - 추천 방식: 검증 Recall@10이 가장 높은 방식
# =====================================================================
rng = np.random.default_rng(SEED)
val_split = {}
for u in eval_users:
    in_idx, in_r = inputs[u]
    perm = rng.permutation(len(in_idx))
    n_val = max(1, int(round(len(in_idx) * VAL_RATIO)))
    val_split[u] = (perm[n_val:], perm[:n_val])   # (학습용 위치, 검증용 위치)

# K 선택
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


# 추천 방식 선택 (검증용 맥주 중 3.5점 이상을 정답으로 봄)
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


rank_methods = {"좋아함+싫어함": profile_scores, "좋아함만": liked_profile_scores}
val_rank = {name: val_recall_at10(fn) for name, fn in rank_methods.items()}
BEST_RANK = max(val_rank, key=val_rank.get)

section("[5] K, 추천 방식 선택 (검증 데이터)")
print("K 선택 (RMSE, 낮을수록 좋음)")
for k, v in val_rmse.items():
    print(f"  K={k:>2}   {v:.4f}" + ("   <- 선택" if k == BEST_K else ""))
print("\n추천 방식 선택 (Recall@10, 높을수록 좋음)")
for name, v in val_rank.items():
    print(f"  {v:.4f}   {name}" + ("   <- 선택" if name == BEST_RANK else ""))

plt.figure(figsize=(6, 4))
plt.plot(list(val_rmse.keys()), list(val_rmse.values()), marker="o")
plt.axvline(BEST_K, color="gray", linestyle="--")
plt.xlabel("K (참고하는 유사 맥주 수)")
plt.ylabel("검증 RMSE")
plt.title("K 선택 (검증 데이터)")
plt.tight_layout()
plt.savefig(OUT / "content_based_k_selection.png", dpi=150)
plt.close()


# =====================================================================
# [6] 모델별 평가
#   - 입력(70%)으로 예측/추천하고 정답(30%)으로 채점
#   - 평점 예측: RMSE, MAE
#   - Top-N 추천: Recall@N, NDCG@N (정답 중 3.5점 이상을 맞힌 것으로 봄)
#   - 입력에 있던 맥주는 추천 후보에서 제외
# =====================================================================
popularity = (train_inter.groupby("canonical_beer_id").size()
              .reindex(beer_ids, fill_value=0).values.astype(float))


def evaluate(name, rating_fn, rank_fn):
    """rating_fn, rank_fn: (in_idx, in_r) -> 맥주별 예측 평점 / 추천 점수"""
    sq, ab, cnt = 0.0, 0.0, 0
    max_n = max(TOP_N_LIST)
    recall = {n: [] for n in TOP_N_LIST}
    ndcg = {n: [] for n in TOP_N_LIST}
    discount = 1 / np.log2(np.arange(2, max_n + 2))

    for u in eval_users:
        in_idx, in_r = inputs[u]
        ho_idx, ho_r, ho_pos = holdout[u]

        # 평점 예측 오차
        err = rating_fn(in_idx, in_r)[ho_idx] - ho_r
        sq += (err ** 2).sum()
        ab += np.abs(err).sum()
        cnt += len(err)

        # Top-N 추천
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
shown = results.round(4).rename_axis(None)       # 출력할 때 index 이름(model) 줄 제거
section(f"[6] 평가 결과 (RateBeer 사용자 {len(eval_users):,}명)")
print("평점 예측 (낮을수록 좋음)")
print(shown[["RMSE", "MAE"]].to_string())
print("\nTop-N 추천 - Recall (높을수록 좋음)")
print(shown[[f"Recall@{n}" for n in TOP_N_LIST]].to_string())
print("\nTop-N 추천 - NDCG (높을수록 좋음)")
print(shown[[f"NDCG@{n}" for n in TOP_N_LIST]].to_string())
# [5]에서 선택된 추천 방식에 맞춰 최종 모델 표시
final_model = {"좋아함+싫어함": "CB-TFIDF", "좋아함만": "CB-TFIDF(liked)"}[BEST_RANK]
model_desc = {
    "Baseline": "평점은 사용자 평균, 추천은 평가 수가 많은 맥주부터",
    "CB-TFIDF": "리뷰 TF-IDF, 추천은 좋아한+싫어한 맥주 모두 반영",
    "CB-TFIDF(pred-rank)": "CB-TFIDF와 같고, 추천만 예측 평점 순서로 정렬",
    "CB-TFIDF+style+ABV": "CB-TFIDF에 맥주 스타일 + 도수 추가",
    "CB-TFIDF(liked)": "리뷰 TF-IDF, 추천은 좋아한 맥주만 반영",
}
print("\n모델 설명")
for name, desc in model_desc.items():
    mark = "   <- 최종 (검증으로 선택)" if name == final_model else ""
    print(f"- {name:<20}: {desc}{mark}")
print(f"(평가 소요 시간: {eval_time:.1f}초)")
results.round(4).to_csv(OUT / "content_based_results.csv", encoding="utf-8-sig")

fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
results[["RMSE", "MAE"]].plot.bar(ax=axes[0], rot=20, title="평점 예측 오차 (낮을수록 좋음)")
results[[f"Recall@{n}" for n in TOP_N_LIST]].plot.bar(ax=axes[1], rot=20, title="Recall@N")
results[[f"NDCG@{n}" for n in TOP_N_LIST]].plot.bar(ax=axes[2], rot=20, title="NDCG@N")
for ax in axes:
    ax.set_xlabel("")
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
plt.savefig(OUT / "content_based_metrics.png", dpi=150)
plt.close()


# =====================================================================
# [7] 정답 맥주 리뷰 수 비교
#   - 사용자들이 실제로 좋아한 맥주가 리뷰 많은(유명한) 맥주 위주인지 확인
#   - Baseline(평가 수 많은 순) 점수가 높게 나오는 이유 설명용
# =====================================================================
pos_beers = test_holdout[test_holdout["user_id"].isin(eval_users) & test_holdout["is_positive"]]
pos_review_cnt = review_count.reindex(pos_beers["canonical_beer_id"])
section("[7] 맥주 리뷰 수 비교")
print(f"- 전체 공통 맥주의 리뷰 수 중앙값: {review_count.median():.0f}개")
print(f"- 사용자가 실제로 좋아한 맥주(정답 중 3.5점 이상)의 리뷰 수 중앙값: {pos_review_cnt.median():.0f}개")


# =====================================================================
# [8] 추천 예시
#   - 입력 맥주 40개 이하인 사용자 중 정답 맥주가 가장 많은 사용자 1명
#   - 높게 평가한 맥주 5개와 최종 추천 방식으로 뽑은 Top 10 출력
# =====================================================================
u = max(eval_users, key=lambda x: len(holdout[x][0]) if len(inputs[x][0]) <= 40 else 0)
in_idx, in_r = inputs[u]
ho_idx, ho_r, ho_pos = holdout[u]

section(f"[8] 추천 예시 (사용자: {u}, 추천 방식: {BEST_RANK})")
print("높게 평가한 맥주")
for pos in np.argsort(-in_r)[:5]:
    j = in_idx[pos]
    print(f"  {in_r[pos]:.2f}점  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")

pred = predict_ratings(S_text, in_idx, in_r, BEST_K)
score = rank_methods[BEST_RANK](S_text, in_idx, in_r)
score[in_idx] = -np.inf
positive_set = set(ho_idx[ho_pos])
print("\n추천 Top 10  (* = 정답에서 실제로 3.5점 이상 준 맥주 / 예측 = 예측 평점)")
for rank, j in enumerate(np.argsort(-score)[:10], start=1):
    mark = "*" if j in positive_set else " "
    print(f" {rank:>2}. {mark} 예측 {pred[j]:.2f}  {mapping.loc[j, 'ba_beer_name']} ({mapping.loc[j, 'ba_style']})")

print("\n결과 파일 저장 위치:", OUT)
