import os

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
test_path = Path(r"output_balanced\dropped2\test_age16\16.csv")
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
features_df = pd.read_csv(all_feature_paths,header=None,sep='\t')
features_df.set_index(features_df.columns[0], inplace=True)
num_rows_with_nan = features_df.isna().any(axis=1).sum()
nan_song_ids = features_df[features_df.isna().any(axis=1)].index.tolist()

print("Song IDs with at least one NaN value:")
print(nan_song_ids)
print(f"Number of rows with at least one NaN: {num_rows_with_nan}")
# Map original IDs to internal indices
all_users = pd.concat([train_df[0], val_df[0], test_df[0]]).unique()
all_items = pd.concat([train_df[1], val_df[1], test_df[1]]).unique()
user_id_map = {uid: idx for idx, uid in enumerate(all_users)}
user_id_map2 = {int(k): v for k, v in user_id_map.items()}
item_id_map = {iid: idx for idx, iid in enumerate(all_items)}
import pandas as pd

# Load the CSV
genre_df = pd.read_csv(r"output\song_genres.csv")

# Convert to dictionary: track_id -> list of genres
item_to_subtopics = {
    int(row['track_id']): [g.strip().lower() for g in str(row['genre_annotations']).split(',')]
    for _, row in genre_df.iterrows()
}

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

    # Map track_id to item index used in train_df
    cbf_df['item'] = cbf_df['track_id'].map(item_id_map)
    cbf_df = cbf_df.dropna(subset=['item']).copy()
    cbf_df['item'] = cbf_df['item'].astype(int)

    # Map feature values to numeric values starting from 1
    unique_values = cbf_df['feature_value'].astype(str).unique()

    cbf_df['feature_value'] = cbf_df['feature_value'].astype(float)


    # Merge to get [user, item, feature_value]
    train_with_features = pd.merge(train_df[['user', 'item']], cbf_df[['item', 'feature_value']], on='item', how='inner')

    # If needed, also map back to original track_id and user_id
    train_with_features['track_id'] = train_with_features['item'].map(rev_item_id_map)
    train_with_features['user_id'] = train_with_features['user'].map(rev_user_id_map)

    # Reorder and show only desired columns
    triple_df = train_with_features[['user_id', 'track_id', 'feature_value']]

    # If still needed for modeling, build sparse matrix
    train_matrix_cbf = sp.csr_matrix(
        (train_with_features['feature_value'], (train_with_features['user'], train_with_features['item'])),
        shape=(len(user_id_map), len(item_id_map))
    )
    # print(train_matrix_cbf)
    train_matrices_cbf.append(train_matrix_cbf)


# Define evaluation functions
def compute_hit_rate_at_k(recommender, train_matrix, test_matrix, test_df, k=10, feature_name=None):
    test_users = test_df['user'].unique()
    hits = 0

    # Prepare output file path (once per call)
    if feature_name:
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
        output_dir = "hit_logs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"hit_results_{safe_feature_name}.csv")
        f = open(output_file, "w")
        f.write("user_id,hit\n")  # header
    else:
        f = None

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])
        recommended_items = {item_id_map[item_id] for item_id, _ in recs if item_id in item_id_map}

        is_hit = int(bool(true_items & recommended_items))
        if is_hit>0:
            hits+=1
        # hits += is_hit

        if f:
            f.write(f"{user},{is_hit}\n")

    if f:
        f.close()

    return hits / len(test_users) if len(test_users) > 0 else 0.0

def compute_mrr_at_k(recommender, train_matrix, test_matrix, test_df, k=10, feature_name=None):
    test_users = test_df['user'].unique()
    reciprocal_ranks = []

    # Prepare file for logging
    if feature_name:
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
        output_dir = "mrr_logs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"mrr_results_{safe_feature_name}.csv")
        f = open(output_file, "w")
        f.write("user_id,reciprocal_rank\n")
    else:
        f = None

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])

        rr = 0.0
        for rank, (item_id, _) in enumerate(recs[:k]):
            internal_item = item_id_map.get(item_id)
            if internal_item is not None and internal_item in true_items:
                rr = 1 / (rank + 1)
                break

        reciprocal_ranks.append(rr)
        if f:
            f.write(f"{user},{rr:.4f}\n")

    if f:
        f.close()

    return sum(reciprocal_ranks) / len(test_users) if len(test_users) > 0 else 0.0
import numpy as np

import os

def compute_coverage(recommender, train_matrix, test_df, all_item_ids, k=10, feature_name=None):
    test_users = test_df['user'].unique()
    recommended_items = set()

    for user in test_users:
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)

        for item_id, _ in recs[:k]:
            recommended_items.add(item_id)

    coverage = len(recommended_items) / len(all_item_ids) if all_item_ids else 0.0

    # Optionally log the result
    # if feature_name:
    #     safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
    #     output_dir = "coverage_logs"
    #     os.makedirs(output_dir, exist_ok=True)
    #     output_file = os.path.join(output_dir, f"coverage_results_{safe_feature_name}.csv")
    #     with open(output_file, "w") as f:
    #         f.write("feature_name,coverage\n")
    #         f.write(f"{feature_name},{coverage:.4f}\n")

    return coverage

def compute_ndcg_at_k(recommender, train_matrix, test_matrix, test_df, k=10, feature_name=None):
    test_users = test_df['user'].unique()
    ndcg_scores = []

    # Prepare file for logging
    if feature_name:
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
        output_dir = "ndcg_logs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"ndcg_results_{safe_feature_name}.csv")
        f = open(output_file, "w")
        f.write("user_id,ndcg\n")
    else:
        f = None

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)
        true_items = set(test_matrix[internal_user].nonzero()[0])

        if not true_items:
            ndcg = 0.0
        else:
            # Compute DCG
            dcg = 0.0
            for rank, (item_id, _) in enumerate(recs[:k]):
                internal_item = item_id_map.get(item_id)
                if internal_item is not None and internal_item in true_items:
                    dcg += 1 / np.log2(rank + 2)  # rank+2 because rank starts at 0

            # Compute ideal DCG
            ideal_dcg = sum(1 / np.log2(i + 2) for i in range(min(len(true_items), k)))
            ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0

        ndcg_scores.append(ndcg)
        if f:
            f.write(f"{user},{ndcg:.4f}\n")

    if f:
        f.close()

    return sum(ndcg_scores) / len(test_users) if len(test_users) > 0 else 0.0

def compute_srecall_at_k(recommender, train_matrix, test_df, item_to_subtopics, k=10, feature_name=None):
    """
    recommender: your trained recommendation model
    train_matrix: user-item matrix (used for masking)
    test_df: DataFrame with test interactions (must have 'user' column)
    item_to_subtopics: dict mapping item_id -> list/set of subtopics
    k: cutoff for top-k recommendations
    feature_name: optional label for logging results
    """
    test_users = test_df['user'].unique()
    srecalls = []

    # Logging setup
    if feature_name:
        safe_feature_name = feature_name.replace(" ", "_").replace("/", "_")
        output_dir = "srecall_logs"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"srecall_results_{safe_feature_name}.csv")
        f = open(output_file, "w")
        f.write("user_id,srecall\n")
    else:
        f = None

    for user in test_users:
        internal_user = user
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)

        recommended_items = [item_id for item_id, _ in recs[:k]]
        covered_subtopics = set()

        for item in recommended_items:

            covered_subtopics.update(item_to_subtopics.get(item, []))

        srecall = len(covered_subtopics) / 19.0  # replace 19.0 with actual nA if it varies
        srecalls.append(srecall)

        # if f:
            # f.write(f"{user},{srecall:.4f}\n")

    if f:
        f.close()

    return sum(srecalls) / len(srecalls) if srecalls else 0.0

from collections import Counter

import numpy as np
from collections import Counter
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np


def cosine_distance(u, v):
    # Compute cosine distance manually
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        # Avoid division by zero, return max distance
        return 1.0

    cosine_sim = dot_product / (norm_u * norm_v)
    cosine_dist = 1.0 - cosine_sim
    return cosine_dist


def recommender_diversity(recommender, train_matrix, test_matrix, test_df, k=10):
    test_users = test_df['user'].unique()
    diversity_scores = []

    for user in test_users:
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)

        # Get recommended items
        items = [item_id for item_id, _ in recs[:k]]
        valid_items = [int(item) for item in items if int(item) in features_df.index]

        # Ensure at least two items for pairwise distance
        if len(valid_items) < 2:
            continue

        # Extract feature vectors for recommended items
        filtered_features_df = features_df.loc[valid_items]
        item_features = filtered_features_df.values
        n = len(valid_items)

        # Manually compute sum of pairwise cosine similarities
        pairwise_sum = 0.0
        for i in range(n):
            for j in range(n):
                if i != j:
                    dot_product = np.dot(item_features[i], item_features[j])
                    norm_i = np.linalg.norm(item_features[i])
                    norm_j = np.linalg.norm(item_features[j])

                    if norm_i == 0 or norm_j == 0:
                        cosine_sim = 0.0
                    else:
                        cosine_sim = dot_product / (norm_i * norm_j)

                    pairwise_sum += cosine_sim

        # Intra-list similarity for this user
        intra_list_similarity = pairwise_sum / (n * (n - 1))

        # Append to diversity_scores
        diversity_scores.append(intra_list_similarity)

    # Return average intra-list similarity across users
    return np.mean(diversity_scores) if diversity_scores else 0.0

def compute_item_popularity_diversity(recommender, train_matrix, test_df, k=10):
    """
    Computes normalized Shannon entropy over the distribution of recommended items across all users.
    """
    test_users = test_df['user'].unique()
    item_counts = Counter()

    for user in test_users:
        user2 = rev_user_id_map.get(user)
        if user2 is None:
            continue

        mask = ~train_matrix[user].toarray().astype(bool).flatten()
        recs = recommender.get_user_recs(user2, mask, k=k)

        items = [item_id for item_id, _ in recs[:k]]
        item_counts.update(items)

    total = sum(item_counts.values())
    if total == 0:
        return 0.0

    probs = [count / total for count in item_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)

    max_entropy = np.log2(len(item_counts)) if len(item_counts) > 0 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return normalized_entropy


# DummyData class
class DummyData:
    def __init__(self, train_matrix, user_id_map, item_id_map, cbf_matrix=None):
        self.sp_i_train = train_matrix
        self.sp_i_train_ratings = train_matrix
        self.sp_i_train_cbf = cbf_matrix
        self.private_users = {v: k for k, v in user_id_map.items()}
        self.public_users = user_id_map
        self.private_items = {v: k for k, v in item_id_map.items()}
        self.public_items = item_id_map
        self.num_users = train_matrix.shape[0]
        self.num_items = train_matrix.shape[1]

# Fixed parameters
fixed_params = {
    'maxk': 10,
    'shrink': 0,
    'similarity': 'cosine',
    'normalize': True,
    'asymmetric_alpha': False,
    'tversky_alpha': False,
    'tversky_beta': False
}

# Test results
results = []

for idx, (cbf_matrix, feature_name) in enumerate(zip(train_matrices_cbf, feature_names)):
    print(f"\nTesting feature: {feature_name}")
    data = DummyData(train_matrix, user_id_map2, item_id_map, cbf_matrix)

    # Initialize recommender with fixed parameters
    recommender = AiolliSimilarity(
        data=data,
        maxk=fixed_params['maxk'],
        shrink=fixed_params['shrink'],
        similarity=fixed_params['similarity'],
        implicit=False,
        normalize=fixed_params['normalize'],
        asymmetric_alpha=fixed_params['asymmetric_alpha'],
        tversky_alpha=fixed_params['tversky_alpha'],
        tversky_beta=fixed_params['tversky_beta'],
        row_weights=None
    )
    recommender.initialize()

    # Compute metrics on test set
    hit_rate = compute_hit_rate_at_k(recommender, train_matrix, test_matrix, test_df, k=10,feature_name=feature_name)
    mrr = compute_mrr_at_k(recommender, train_matrix, test_matrix, test_df, k=10,feature_name=feature_name)
    ndcg = compute_ndcg_at_k(recommender, train_matrix, test_matrix, test_df, k=10,feature_name=feature_name)
    srecall = compute_srecall_at_k(recommender, train_matrix, test_df, item_to_subtopics, k=10,
                                   feature_name=feature_name)
    coverage = compute_coverage(recommender, train_matrix, test_df, item_to_subtopics, k=10)
    entropy = compute_item_popularity_diversity(recommender,train_matrix,test_df,k=10)
    diversity = recommender_diversity(recommender,train_matrix,test_matrix,test_df,k=10)
    results.append((feature_name, hit_rate, mrr,srecall,ndcg,coverage,entropy,diversity))
    # Print per-feature metrics
    print(f"\nTest Metrics for feature: {feature_name}")
    print(f"Hit Rate@10:   {hit_rate:.4f}")
    print(f"MRR@10:        {mrr:.4f}")
    print(f"S-Recall@10:   {srecall:.4f}")
    print(f"nDCG@10:   {ndcg:.4f}")
    print(f"Coverage@10:   {coverage:.4f}")
    print(f"Item Entropy@10:   {entropy:.4f}")
    print(f"Diversity@10:   {diversity:.4f}")
# Summary of all results
print("\nSummary of Results:")
print(f"{'Feature':<15} {'HitRate@10':>12} {'MRR@10':>10} {'Srecall@10':>10} {'NDCG@10':>10} {'Coverage@10':>10} {'Entropy@10':>10} {'Diversity@10':>10}")
for feature_name, hit_rate, mrr, srecall, ndcg,coverage,entropy,diversity in results:
    print(f"{feature_name:<15} {hit_rate:>12.4f} {mrr:>10.4f} {srecall:>10.4f} {ndcg:>10.4f} {coverage:>10.4f} {entropy:>10.4f} {diversity:>10.4f}")