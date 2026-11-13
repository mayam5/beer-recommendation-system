"""Item-based CF converted from the supplied final notebook.

Evaluation: leave-one-out (the notebook does not include a 70:30 split or Recall/NDCG).
Selection: exact maximum clique with at least 5 common raters for every beer pair.
Raw / Adjusted-all / Adjusted-selected, K=2,3,5, using positive similarities only.
Run: python src/item_based/item_based.py
Override paths with --ba, --rb, --mapping, and --output-dir.
"""
from pathlib import Path
import argparse
import json
import platform
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / 'data' / 'final_data'
RESULTS_DIR = PROJECT_DIR / 'results' / 'item_based'
K_VALUES = (2, 3, 5)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ba', type=Path, default=DATA_DIR / 'train' / 'beeradvocate_clean.parquet')
    parser.add_argument('--rb', type=Path, default=DATA_DIR / 'test' / 'ratebeer_clean.parquet')
    parser.add_argument('--mapping', type=Path, default=DATA_DIR / 'beer_mapping.parquet')
    parser.add_argument('--output-dir', type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def make_similarity(frame):
    """Compute cosine similarity using only common raters, including pair-specific denominators.

    Vectorize the same calculation as the pairwise loop in the notebook.
    numerator[i,j] = sum of r_i * r_j over common raters
    square_norms[i,j] = sum of squared r_i over common raters
    """
    values = frame.to_numpy(dtype=float)
    observed = np.isfinite(values).astype(float)
    values = np.where(np.isfinite(values), values, 0.0)
    numerator = values.T @ values
    square_norms = (values * values).T @ observed
    denominator = np.sqrt(square_norms * square_norms.T)
    similarity = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    np.fill_diagonal(similarity, 0.0)
    assert np.allclose(similarity, similarity.T)
    # Check representative pairs against the original pairwise formula.
    for i in range(min(5, frame.shape[1])):
        for j in range(i + 1, min(5, frame.shape[1])):
            a, b = frame.iloc[:, i].to_numpy(), frame.iloc[:, j].to_numpy()
            mask = np.isfinite(a) & np.isfinite(b)
            a, b = a[mask], b[mask]
            den = np.linalg.norm(a) * np.linalg.norm(b)
            expected = float(np.dot(a, b) / den) if den > 0 else 0.0
            assert np.isclose(similarity[i, j], expected, atol=1e-12)
    return pd.DataFrame(similarity, index=frame.columns, columns=frame.columns)


def select_training_beers(ba, mapping):
    matched = ba.merge(mapping[['ba_beer_id', 'canonical_beer_id']],
                       left_on='beer_id', right_on='ba_beer_id', how='inner',
                       validate='many_to_one').rename(columns={'canonical_beer_id': 'common_beer_id'})
    matched = matched[matched.user_id.notna() & matched.user_id.str.strip().ne('')]
    train = matched.pivot_table(index='user_id', columns='common_beer_id', values='rating', aggfunc='mean').sort_index(axis=1)
    # Count common raters using sparse multiplication, equivalent to dense rated.T @ rated.
    observed = csr_matrix(train.notna().to_numpy(dtype=np.int32))
    counts = (observed.T @ observed).toarray()
    adjacency = counts >= 5
    np.fill_diagonal(adjacency, False)
    graph = nx.from_numpy_array(adjacency)
    print(f'Searching for the maximum clique among {train.shape[1]} common beers...', flush=True)
    indices = sorted(max(nx.find_cliques(graph), key=len))
    selected = train.iloc[:, indices].copy()
    pair_counts = counts[np.ix_(indices, indices)][np.triu_indices(len(indices), 1)]
    if not len(pair_counts) or pair_counts.min() < 5:
        raise ValueError('No beer pairs satisfy the minimum common-rater requirement.')
    print(f'Selected {len(indices)} beers / minimum common raters: {pair_counts.min()}', flush=True)
    return train, selected, int(pair_counts.min())


def evaluate_loo(test, models, ids):
    """Hide each target rating and predict it from the remaining user ratings, as in the notebook."""
    beer_position = {beer: i for i, beer in enumerate(ids)}
    predictions = []
    user_count = test.user_id.nunique()
    for number, (user_id, group) in enumerate(test.groupby('user_id'), 1):
        beers = group.common_beer_id.to_numpy()
        actual = group.rating.to_numpy(dtype=float)
        positions = np.array([beer_position[beer] for beer in beers])
        user_parts = []
        for model_name, matrix in models.items():
            sims = matrix.to_numpy()[np.ix_(positions, positions)]
            # Exclude the target rating. Positive neighbors retain the notebook ordering.
            weights = np.where(np.isfinite(sims) & (sims > 0), sims, 0.0)
            np.fill_diagonal(weights, 0.0)
            order = np.argsort(-weights, axis=1, kind='stable')
            for k in K_VALUES:
                neighbors = order[:, :k]
                w = np.take_along_axis(weights, neighbors, axis=1)
                r = actual[neighbors]
                denominator = w.sum(axis=1)
                predicted = np.full(len(actual), np.nan)
                np.divide((w * r).sum(axis=1), denominator, out=predicted, where=denominator > 0)
                predicted = np.clip(predicted, 1.0, 5.0)
                user_parts.append(pd.DataFrame({
                    'user_id': user_id, 'target_beer': beers, 'model': model_name,
                    'K': k, 'actual': actual, 'predicted': predicted,
                    'used_neighbors': (w > 0).sum(axis=1)
                }))
        predictions.append(pd.concat(user_parts, ignore_index=True))
        if number % 500 == 0:
            print(f'Evaluation progress: {number}/{user_count} users', flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    summary = []
    for (model, k), group in predictions.groupby(['model', 'K']):
        valid = group.dropna(subset=['predicted'])
        errors = valid.predicted - valid.actual
        mse = (errors ** 2).mean()
        summary.append({'model': model, 'K': k, 'MAE': errors.abs().mean(),
                        'RMSE': np.sqrt(mse), 'MSE': mse,
                        'Coverage': len(valid) / len(group),
                        'successful_predictions': len(valid), 'total_predictions': len(group)})
    return predictions, pd.DataFrame(summary).sort_values(['K', 'model']).reset_index(drop=True)


def plot_results(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ['MSE', 'RMSE']):
        for name, group in results.groupby('model'):
            group = group.sort_values('K')
            ax.plot(group.K, group[metric], marker='o', label=name)
            for k, value in zip(group.K, group[metric]):
                offset = -19 if name == 'Adjusted-233' else 10
                ax.annotate(f'{value:.4f}', (k, value), textcoords='offset points', xytext=(0, offset), ha='center', fontsize=9)
        ax.set(title=f'Top-K: {metric}', xlabel='K', ylabel=metric, xticks=list(K_VALUES))
        ax.margins(x=0.12, y=0.25)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    for path in [args.ba, args.rb, args.mapping]:
        if not path.is_file():
            raise FileNotFoundError(f'Data file not found: {path}')
    started = time.time()
    evaluation_dir = args.output_dir / 'evaluation'
    plots_dir = args.output_dir / 'plots'
    similarity_dir = args.output_dir / 'similarity'
    for directory in [evaluation_dir, plots_dir, similarity_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    ba, rb, mapping = [pd.read_parquet(path) for path in [args.ba, args.rb, args.mapping]]
    train, selected, minimum = select_training_beers(ba, mapping)
    n = selected.shape[1]
    models = {
        'Raw': make_similarity(selected),
        f'Adjusted-{train.shape[1]}': make_similarity(selected.sub(train.mean(axis=1), axis=0)),
        f'Adjusted-{n}': make_similarity(selected.sub(selected.mean(axis=1), axis=0))
    }
    for name, similarity in models.items():
        similarity.to_csv(similarity_dir / f'{name}.csv')
    selected_meta = mapping.set_index('canonical_beer_id').loc[selected.columns].rename_axis('canonical_beer_id').reset_index()
    selected_meta.to_csv(evaluation_dir / 'selected_beers.csv', index=False, encoding='utf-8-sig')
    matched = rb.merge(mapping[['rb_beer_id', 'canonical_beer_id']],
                       left_on='beer_id', right_on='rb_beer_id', how='inner',
                       validate='many_to_one').rename(columns={'canonical_beer_id': 'common_beer_id'})
    test = matched.loc[matched.common_beer_id.isin(selected.columns), ['user_id', 'common_beer_id', 'rating']].copy()
    counts = test.groupby('user_id').common_beer_id.nunique()
    test = test[test.user_id.isin(counts[counts >= 10].index)]
    if test.empty:
        raise ValueError('No users have rated at least 10 selected beers.')
    if test.duplicated(['user_id', 'common_beer_id']).any():
        raise ValueError('Duplicate user-beer pairs found in the test data.')
    print(f'Evaluation users: {test.user_id.nunique()} / ratings: {len(test)}', flush=True)
    predictions, results = evaluate_loo(test, models, selected.columns.tolist())
    results.to_csv(evaluation_dir / 'evaluation_results.csv', index=False, encoding='utf-8-sig')
    predictions.to_parquet(evaluation_dir / 'topk_predictions.parquet', index=False, compression='zstd')
    plot_results(results, plots_dir / 'mse_rmse_comparison.png')
    settings = {
        'evaluation': 'leave-one-out; no Recall/NDCG in supplied notebook',
        'common_beers': train.shape[1], 'selected_beers': n, 'minimum_common_raters': minimum,
        'training_users': train.shape[0], 'evaluation_users': int(test.user_id.nunique()),
        'test_ratings': len(test), 'K': list(K_VALUES), 'similarity': 'pairwise common finite raters only',
        'prediction': 'positive-similarity weighted raw-rating mean; clip [1,5]; no neighbors => NaN',
        'errors': 'computed on finite predictions; Coverage reported separately',
        'selection': 'exact max(nx.find_cliques(graph), key=len), same as supplied notebook',
        'seconds': round(time.time() - started, 2),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'networkx': nx.__version__}
    }
    (evaluation_dir / 'run_settings.json').write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(results.to_string(index=False), flush=True)
    print(f'Output directory: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
