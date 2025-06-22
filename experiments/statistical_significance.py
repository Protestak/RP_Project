import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
import os
import scipy.stats as stats
from scipy.stats import wilcoxon, ttest_rel

# Age groups
age_groups = [15, 16, 17, 18]

# Features to test
features = [
    "acousticness", "danceability", "energy", "instrumentalness", "key",
    "liveness", "loudness", "mode", "speechiness", "tempo", "valence"
]

# Base directory structure
root_path = "C:/Users/erkin/ResearchProject/traincbf/statistical_significance"

# McNemar's Test for HitRate
for age in age_groups:
    print(f"\n{'='*35}")
    print(f"Statistical Analysis for Age Group: {age}")
    print(f"{'='*35}")
    print("McNemar's Test for HitRate@10")
    print(f"{'-'*35}")

    age_dir = os.path.join(root_path, f"logs_{age}", "hit_logs")
    baseline_path = os.path.join(age_dir, "hit_results_baseline.csv")
    baseline_df = pd.read_csv(baseline_path)

    for feature in features:
        feature_path = os.path.join(age_dir, f"hit_results_{feature}.csv")
        feature_df = pd.read_csv(feature_path)

        # Merge on user_id
        merged = pd.merge(baseline_df, feature_df, on="user_id", suffixes=('_baseline', f'_{feature}'))

        if merged.empty:
            print(f"Warning: No overlapping users found for feature '{feature}' at age {age}. Skipping feature.")
            continue

        # Contingency table values
        a = ((merged.hit_baseline == 1) & (merged[f"hit_{feature}"] == 1)).sum()
        b = ((merged.hit_baseline == 1) & (merged[f"hit_{feature}"] == 0)).sum()
        c = ((merged.hit_baseline == 0) & (merged[f"hit_{feature}"] == 1)).sum()
        d = ((merged.hit_baseline == 0) & (merged[f"hit_{feature}"] == 0)).sum()

        table = [[a, b], [c, d]]

        print(f"\nComparison: Baseline vs. {feature} for Age {age}")
        result = mcnemar(table, exact=True)
        p_value = result.pvalue
        print(f"McNemar's Test Result: p-value = {p_value:.4f}")
        print(f"Conclusion: {'Statistically Significant' if p_value < 0.05 else 'Not Statistically Significant'} (α = 0.05)")

# Wilcoxon and Paired t-Test for MRR
for age in age_groups:
    print(f"\n{'='*35}")
    print(f"Statistical Analysis for Age Group: {age}")
    print(f"{'='*35}")
    print("Wilcoxon Signed-Rank and Paired t-Test for MRR@10")
    print(f"{'-'*35}")

    age_dir = os.path.join(root_path, f"logs_{age}", "mrr_logs")
    baseline_path = os.path.join(age_dir, "mrr_results_baseline.csv")
    baseline_df = pd.read_csv(baseline_path)
    baseline_df.rename(columns={'reciprocal_rank': 'mrr_baseline'}, inplace=True)

    for feature in features:
        feature_path = os.path.join(age_dir, f"mrr_results_{feature}.csv")
        feature_df = pd.read_csv(feature_path)
        feature_df.rename(columns={'reciprocal_rank': f'mrr_{feature}'}, inplace=True)

        # Merge on user_id
        merged = pd.merge(baseline_df, feature_df, on="user_id")

        if merged.empty:
            print(f"Warning: No overlapping users found for feature '{feature}' at age {age}. Skipping feature.")
            continue

        mrr_baseline = merged["mrr_baseline"]
        mrr_feature = merged[f"mrr_{feature}"]
        differences = mrr_baseline - mrr_feature

        print(f"\nComparison: Baseline vs. {feature} for Age {age}")
        print("Normality Check for Differences (Shapiro-Wilk Test)")
        shapiro_stat, shapiro_p = stats.shapiro(differences)
        print(f"Shapiro-Wilk Test Result: p-value = {shapiro_p:.4f}")
        print(f"Conclusion: {'Not Normally Distributed' if shapiro_p < 0.05 else 'Normally Distributed'} (α = 0.05)")

        # Wilcoxon Signed-Rank Test
        stat_w, p_value_w = wilcoxon(mrr_baseline, mrr_feature)
        print("\nWilcoxon Signed-Rank Test Result")
        print(f"p-value = {p_value_w:.4f}")
        print(f"Conclusion: {'Statistically Significant' if p_value_w < 0.05 else 'Not Statistically Significant'} (α = 0.05)")

        # Paired t-Test
        stat_t, p_value_t = ttest_rel(mrr_baseline, mrr_feature)
        print("\nPaired t-Test Result")
        print(f"p-value = {p_value_t:.4f}")
        print(f"Conclusion: {'Statistically Significant' if p_value_t < 0.05 else 'Not Statistically Significant'} (α = 0.05)")

# Wilcoxon and Paired t-Test for NDCG
for age in age_groups:
    print(f"\n{'='*35}")
    print(f"Statistical Analysis for Age Group: {age}")
    print(f"{'='*35}")
    print("Wilcoxon Signed-Rank and Paired t-Test for NDCG")
    print(f"{'-'*35}")

    age_dir = os.path.join(root_path, f"logs_{age}", "ndcg_logs")
    baseline_path = os.path.join(age_dir, "ndcg_results_baseline.csv")
    baseline_df = pd.read_csv(baseline_path).rename(columns={'ndcg': 'ndcg_baseline'})

    for feature in features:
        feature_path = os.path.join(age_dir, f"ndcg_results_{feature}.csv")
        feature_df = pd.read_csv(feature_path).rename(columns={'ndcg': f'ndcg_{feature}'})
        merged = pd.merge(baseline_df, feature_df, on="user_id")

        if merged.empty:
            print(f"Warning: No overlapping users found for feature '{feature}' at age {age}. Skipping feature.")
            continue

        ndcg_baseline = merged["ndcg_baseline"]
        ndcg_feature = merged[f"ndcg_{feature}"]
        differences = ndcg_baseline - ndcg_feature

        print(f"\nComparison: Baseline vs. {feature} for Age {age}")
        print("Normality Check for Differences (Shapiro-Wilk Test)")
        shapiro_stat, shapiro_p = stats.shapiro(differences)
        print(f"Shapiro-Wilk Test Result: p-value = {shapiro_p:.4f}")
        print(f"Conclusion: {'Not Normally Distributed' if shapiro_p < 0.05 else 'Normally Distributed'} (α = 0.05)")

        stat_w, p_value_w = wilcoxon(ndcg_baseline, ndcg_feature)
        print("\nWilcoxon Signed-Rank Test Result")
        print(f"p-value = {p_value_w:.4f}")
        print(f"Conclusion: {'Statistically Significant' if p_value_w < 0.05 else 'Not Statistically Significant'} (α = 0.05)")

        stat_t, p_value_t = ttest_rel(ndcg_baseline, ndcg_feature)
        print("\nPaired t-Test Result")
        print(f"p-value = {p_value_t:.4f}")
        print(f"Conclusion: {'Statistically Significant' if p_value_t < 0.05 else 'Not Statistically Significant'} (α = 0.05)")