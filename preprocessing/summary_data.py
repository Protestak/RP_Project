import pandas as pd


train_df = pd.read_csv('output_balanced/dropped2/train/train.csv', sep='\t', header=None, names=['user_id', 'track_id'])
val_df = pd.read_csv('output_balanced/dropped2/val/val.csv', sep='\t', header=None, names=['user_id', 'track_id'])


a
def summarize_no_age(df, name):
    print(f"\n--- {name.upper()} SET ---")
    total_interactions = len(df)
    unique_users = df['user_id'].nunique()
    avg_interactions_per_user = total_interactions / unique_users if unique_users else 0

    print(f"Total interactions: {total_interactions}")
    print(f"Unique users: {unique_users}")
    print(f"Average interactions per user: {avg_interactions_per_user:.2f}")


# Summarize train and validation sets
summarize_no_age(train_df, "train")
summarize_no_age(val_df, "validation")

# Test sets: Add age column manually
test_dfs = {}
for age in range(15, 19):
    df = pd.read_csv(f'output_balanced/dropped2/test_age{age}/{age}.csv', sep='\t', header=None,
                     names=['user_id', 'track_id'])
    df['age'] = age
    test_dfs[age] = df

# Combine test sets
test_all = pd.concat(test_dfs.values(), ignore_index=True)


print("\n--- TEST SET SUMMARY BY AGE ---")
interactions_per_age = test_all['age'].value_counts().sort_index()
print("Interactions per age group:")
print(interactions_per_age)

avg_interactions_per_user_by_age = test_all.groupby('age')['user_id'].value_counts().groupby(level=0).mean()
print("\nAverage interactions per user by age group:")
print(avg_interactions_per_user_by_age)


user_age_map = test_all[['user_id', 'age']].drop_duplicates()

train_with_age = train_df.merge(user_age_map, on='user_id', how='inner')
val_with_age = val_df.merge(user_age_map, on='user_id', how='inner')


def summarize_by_age(df, name):
    print(f"\n--- {name.upper()} SET (Users from test set with known age) ---")

    interactions_per_age = df['age'].value_counts().sort_index()
    print("Interactions per age group:")
    print(interactions_per_age)

    avg_interactions_per_user_by_age = df.groupby('age')['user_id'].value_counts().groupby(level=0).mean()
    print("\nAverage interactions per user by age group:")
    print(avg_interactions_per_user_by_age)


# Summarize train and validation sets by age
summarize_by_age(train_with_age, "train")
summarize_by_age(val_with_age, "validation")
