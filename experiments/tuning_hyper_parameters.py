import pandas as pd
import scipy.sparse as sp
import numpy as np
from pathlib import Path
from itertools import product
from aiolli import AiolliSimilarity
from vsm import Similarity

# Define data paths
train_path = Path(r"output_balanced\dropped2\train\train.csv")
val_path = Path(r"output_balanced\dropped2\val\val.csv")
test_path = Path(r"output_balanced\dropped2\val\val.csv")
all_feature_paths = Path(r"joined_features\features.csv")
cbf_paths = [
    Path(r"filtered_output_features\key.tsv\key.csv"),
    Path(r"filtered_output_features\acousticness.tsv\acousticness.csv"),
    Path(r"filtered_output_features\danceability.tsv\danceability.csv"),
    Path(r"filtered_output_features\energy.tsv\energy.csv"),
    Path(r"filtered_output_features\instrumentalness.tsv\instrumentalness.csv"),
    Path(r"filtered_output_features\liveness.tsv\liveness.csv"),
    Path(r"filtered_output_features\loudness.tsv\loudness.csv"),
    Path(r"filtered_output_features\mode.tsv\mode.csv"),
    Path(r"filtered_output_features\speechiness.tsv\speechiness.csv"),
    Path(r"filtered_output_features\tempo.tsv\tempo.csv"),
    Path("filtered_output_features\valence.tsv\valence.csv"),
]
feature_names = [
    'key', 'acousticness', 'danceability', 'energy', 'instrumentalness',
    'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'
]

# Load data
train_df = pd.read_csv(train_path, header=None, sep='\t')
val_df = pd.read_csv(val_path, header=None, sep='\t')
test_df = pd.read_csv(test_path, header=None, sep='\t')

# Map original IDs to internal indices
all_users = pd.concat([train_df[0], val_df[0], test_df[0]]).unique()
all_items = pd.concat([train_df[1], val_df[1], test_df[1]]).unique()
user_id_map = {uid: idx for idx, uid in enumerate(all_users)}
user_id_map2 = {int(k): v for k, v in user_id_map.items()}
item_id_map = {iid: idx for idx, iid in enumerate(all_items)}

# Reverse maps
rev_user_id_map = {v: k for k, v in user_id_map2.items()}
rev_item_id_map = {v: k for k, v in item_id_map.items()}

# Assign column names
train_df.columns = ['user', 'item']
val_df.columns = ['user', 'item']
test_df.columns = ['user', 'item']

# Add dummy ratings (implicit feedback)
train_df['rating'] = 1

# Map user/item IDs to internal indices
train_df['user'] = train_df['user'].map(user_id_map)
train_df['item'] = train_df['item'].map(item_id_map)
val_df['user'] = val_df['user'].map(user_id_map)
val_df['item'] = val_df['item'].map(item_id_map)
test_df['user'] = test_df['user'].map(user_id_map)
test_df['item'] = test_df['item'].map(item_id_map)

# Drop rows with missing mappings
train_df = train_df.dropna(subset=['user', 'item'])
val_df = val_df.dropna(subset=['user', 'item'])
test_df = test_df.dropna(subset=['user', 'item'])

# Convert to integer
train_df['user'] = train_df['user'].astype(int)
train_df['item'] = train_df['item'].astype(int)
val_df['user'] = val_df['user'].astype(int)
val_df['item'] = val_df['item'].astype(int)
test_df['user'] = test_df['user'].astype(int)
test_df['item'] = test_df['item'].astype(int)

# Verify indices
print(f"Max user index in train_df: {train_df['user'].max()}, len(user_id_map): {len(user_id_map)}")
print(f"Max item index in train_df: {train_df['item'].max()}, len(item_id_map): {len(item_id_map)}")
print(f"Max user index in val_df: {val_df['user'].max()}, len(user_id_map): {len(user_id_map)}")
print(f"Max item index in val_df: {val_df['item'].max()}, len(item_id_map): {len(item_id_map)}")
print(f"Max user index in test_df: {test_df['user'].max()}, len(user_id_map): {len(user_id_map)}")
print(f"Max item index in test_df: {test_df['item'].max()}, len(item_id_map): {len(item_id_map)}")

assert train_df['user'].max() < len(user_id_map), "Train user index exceeds matrix dimension"
assert train_df['item'].max() < len(item_id_map), "Train item index exceeds matrix dimension"
assert val_df['user'].max() < len(user_id_map), "Val user index exceeds matrix dimension"
assert val_df['item'].max() < len(item_id_map), "Val item index exceeds matrix dimension"
assert test_df['user'].max() < len(user_id_map), "Test user index exceeds matrix dimension"
assert test_df['item'].max() < len(item_id_map), "Test item index exceeds matrix dimension"

# Build train matrix
train_matrix = sp.csr_matrix(
    (train_df['rating'], (train_df['user'], train_df['item'])),
    shape=(len(user_id_map), len(item_id_map))
)

# Build validation matrix
val_df['rating'] = 1
val_matrix = sp.csr_matrix(
    (np.ones(len(val_df)), (val_df['user'], val_df['item'])),
    shape=(len(user_id_map), len(item_id_map))
).toarray().astype(bool)

# Build test matrix
test_matrix = sp.csr_matrix(
    (np.ones(len(test_df)), (test_df['user'], test_df['item'])),
    shape=(len(user_id_map), len(item_id_map))
).toarray().astype(bool)

# Fix all_combinations to use zip instead of product
all_combinations = list(zip(feature_names, cbf_paths))
print("Feature-Path Combinations:", all_combinations)

# Load and process CBF data
train_matrices_cbf = []
for feature_name, cbf_path in all_combinations:
    print(f"Processing feature: {feature_name}")
    cbf_df = pd.read_csv(cbf_path, header=None, sep='\t')
    cbf_df.columns = ['track_id', 'feature_value']
    cbf_df['item'] = cbf_df['track_id'].map(item_id_map)
    cbf_df = cbf_df.dropna(subset=['item']).copy()
    cbf_df['item'] = cbf_df['item'].astype(int)

    # Map feature values to indices (starting from 1 to avoid zero)
    unique_values = cbf_df['feature_value'].astype(str).unique()
    value_map = {val: idx + 1 for idx, val in enumerate(unique_values)}
    cbf_df['feature_value'] = cbf_df['feature_value'].astype(str).map(value_map)

    # Merge with train_df
    train_with_features = pd.merge(train_df, cbf_df[['item', 'feature_value']], on='item', how='inner')

    # Build sparse matrix
    train_matrix_cbf = sp.csr_matrix(
        (train_with_features['feature_value'], (train_with_features['user'], train_with_features['item'])),
        shape=(len(user_id_map), len(item_id_map))
    )
    train_matrices_cbf.append(train_matrix_cbf)

# Define evaluation functions
def compute_hit_rate_at_k(recommender, train_matrix, test_matrix, test_df, k=10):
    test_users = test_df['user'].unique()
    hits = 0

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue
        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])
        recommended_items = {item_id_map[item_id] for item_id, _ in recs if item_id in item_id_map}
        if true_items & recommended_items:
            hits += 1

    return hits / len(test_users) if len(test_users) > 0 else 0.0

def compute_mrr_at_k(recommender, train_matrix, test_matrix, test_df, k=10):
    test_users = test_df['user'].unique()
    reciprocal_ranks = []

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue
        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])
        for rank, (item_id, _) in enumerate(recs[:k]):
            internal_item = item_id_map.get(item_id)
            if internal_item is not None and internal_item in true_items:
                reciprocal_ranks.append(1 / (rank + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(test_users) if len(test_users) > 0 else 0.0

def compute_precision_at_k(recommender, train_matrix, test_matrix, test_df, k=10):
    test_users = test_df['user'].unique()
    precisions = []

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue
        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])
        recommended_items = {item_id_map[item_id] for item_id, _ in recs if item_id in item_id_map}
        num_relevant = len(true_items & recommended_items)
        precisions.append(num_relevant / k)

    return sum(precisions) / len(test_users) if len(test_users) > 0 else 0.0

def compute_recall_at_k(recommender, train_matrix, test_matrix, test_df, k=10):
    test_users = test_df['user'].unique()
    recalls = []

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue
        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])
        recommended_items = {item_id_map[item_id] for item_id, _ in recs if item_id in item_id_map}
        num_relevant = len(true_items & recommended_items)
        recalls.append(num_relevant / len(true_items) if len(true_items) > 0 else 0.0)

    return sum(recalls) / len(test_users) if len(test_users) > 0 else 0.0

# DummyData class
class DummyData:
    def __init__(self, train_matrix, user_id_map, item_id_map, cbf_matrix=None):
        self.sp_i_train = train_matrix
        self.sp_i_train_ratings = train_matrix
        self.sp_i_train_cbf = cbf_matrix if cbf_matrix is not None else train_matrix
        self.private_users = {v: k for k, v in user_id_map.items()}
        self.public_users = user_id_map
        self.private_items = {v: k for k, v in item_id_map.items()}
        self.public_items = item_id_map
        self.num_users = train_matrix.shape[0]
        self.num_items = train_matrix.shape[1]

# Define hyperparameter grid
param_grid = {
    'maxk': [10],
    'shrink': [0, 10, 50],
    'similarity': ['cosine', 'adjusted', 'asymmetric', 'pearson', 'jaccard', 'dice', 'tversky', 'tanimoto'],
    'normalize': [True],
}

asymmetric_alpha_values = [0.5, 1.0]
tversky_alpha_values = [0.5, 1.0]
tversky_beta_values = [0.5, 1.0]

# Function to generate valid parameter combinations
def generate_param_combinations():
    base_keys, base_values = zip(*param_grid.items())
    base_combinations = [dict(zip(base_keys, v)) for v in product(*base_values)]
    valid_combinations = []

    for params in base_combinations:
        combination = params.copy()
        if params['similarity'] == 'asymmetric':
            for alpha in asymmetric_alpha_values:
                new_comb = combination.copy()
                new_comb['asymmetric_alpha'] = alpha
                new_comb['tversky_alpha'] = False
                new_comb['tversky_beta'] = False
                valid_combinations.append(new_comb)
        elif params['similarity'] == 'tversky':
            for t_alpha, t_beta in product(tversky_alpha_values, tversky_beta_values):
                new_comb = combination.copy()
                new_comb['asymmetric_alpha'] = False
                new_comb['tversky_alpha'] = t_alpha
                new_comb['tversky_beta'] = t_beta
                valid_combinations.append(new_comb)
        else:
            new_comb = combination.copy()
            new_comb['asymmetric_alpha'] = False
            new_comb['tversky_alpha'] = False
            new_comb['tversky_beta'] = False
            valid_combinations.append(new_comb)

    return valid_combinations

# Generate all valid parameter combinations
valid_combinations = generate_param_combinations()

# Grid search results
results = []
best_params_per_feature = {}

for idx, (cbf_matrix, feature_name) in enumerate(zip(train_matrices_cbf, feature_names)):
    print(f"\nGrid Search for feature: {feature_name}")
    data = DummyData(train_matrix, user_id_map2, item_id_map, cbf_matrix)
    best_hit_rate = -1
    best_params = None

    # Grid search over hyperparameters
    for params in valid_combinations:
        print(f"Testing parameters: {params}")
        recommender = AiolliSimilarity(
            data=data,
            maxk=params['maxk'],
            shrink=params['shrink'],
            similarity=params['similarity'],
            implicit=False,
            normalize=params['normalize'],
            asymmetric_alpha=params['asymmetric_alpha'],
            tversky_alpha=params['tversky_alpha'],
            tversky_beta=params['tversky_beta'],
            row_weights=None
        )
        recommender.initialize()
        hit_rate = compute_hit_rate_at_k(recommender, train_matrix, val_matrix, val_df, k=10)
        print(f"Validation Hit Rate@10: {hit_rate:.4f}")

        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            best_params = params.copy()

    print(f"\nBest parameters for {feature_name}: {best_params}")
    print(f"Best Validation Hit Rate@10: {best_hit_rate:.4f}")
    best_params_per_feature[feature_name] = best_params

    # Train recommender with best parameters and evaluate on test set
    recommender = AiolliSimilarity(
        data=data,
        maxk=best_params['maxk'],
        shrink=best_params['shrink'],
        similarity=best_params['similarity'],
        implicit=False,
        normalize=best_params['normalize'],
        asymmetric_alpha=best_params['asymmetric_alpha'],
        tversky_alpha=best_params['tversky_alpha'],
        tversky_beta=best_params['tversky_beta'],
        row_weights=None
    )
    recommender.initialize()

    # Compute metrics on test set
    hit_rate = compute_hit_rate_at_k(recommender, train_matrix, test_matrix, test_df, k=10)
    mrr = compute_mrr_at_k(recommender, train_matrix, test_matrix, test_df, k=10)
    precision = compute_precision_at_k(recommender, train_matrix, test_matrix, test_df, k=10)
    recall = compute_recall_at_k(recommender, train_matrix, test_matrix, test_df, k=10)

    results.append((feature_name, hit_rate, mrr, precision, recall))

    # Print per-feature metrics
    print(f"\nTest Metrics for feature: {feature_name}")
    print(f"Hit Rate@10:   {hit_rate:.4f}")
    print(f"MRR@10:        {mrr:.4f}")
    print(f"Precision@10:  {precision:.4f}")
    print(f"Recall@10:     {recall:.4f}")

# Summary of all results
print("\nSummary of Results:")
print(f"{'Feature':<15} {'HitRate@10':>12} {'MRR@10':>10} {'Precision@10':>15} {'Recall@10':>12}")
for feature_name, hit_rate, mrr, precision, recall in results:
    print(f"{feature_name:<15} {hit_rate:>12.4f} {mrr:>10.4f} {precision:>15.4f} {recall:>12.4f}")

# Print best parameters for each feature
print("\nBest Hyperparameters per Feature:")
for feature_name, params in best_params_per_feature.items():
    print(f"{feature_name}: {params}")

    # Best
    # Hyperparameters
    # per
    # Feature:
    # key: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #       'tversky_alpha': False, 'tversky_beta': False}
    # acousticness: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #                'tversky_alpha': False, 'tversky_beta': False}
    # danceability: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #                'tversky_alpha': False, 'tversky_beta': False}
    # energy: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #          'tversky_alpha': False, 'tversky_beta': False}
    # instrumentalness: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #                    'tversky_alpha': False, 'tversky_beta': False}
    # liveness: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #            'tversky_alpha': False, 'tversky_beta': False}
    # loudness: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #            'tversky_alpha': False, 'tversky_beta': False}
    # mode: {'maxk': 10, 'shrink': 10, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #        'tversky_alpha': False, 'tversky_beta': False}
    # speechiness: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #               'tversky_alpha': False, 'tversky_beta': False}
    # tempo: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #         'tversky_alpha': False, 'tversky_beta': False}
    # valence: {'maxk': 10, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'asymmetric_alpha': False,
    #           'tversky_alpha': False, 'tversky_beta': False}