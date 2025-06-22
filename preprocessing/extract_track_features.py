from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("TrackFeatureJoin") \
    .getOrCreate()

# Paths
train_path = "output_balanced/filtered/train/train.csv"
val_path = "output_balanced/filtered/val/val.csv"
test_paths = [
    "output_balanced/filtered/test_age15/15.csv",
    "output_balanced/filtered/test_age16/16.csv",
    "output_balanced/filtered/test_age17/17.csv",
    "output_balanced/filtered/test_age18/18.csv"
]
acoustic_path = "LFM-BeyMS/creation/acoustic_features_lfm_id.tsv"

# Read all datasets (tab-separated, no header)
train_df = spark.read.option("header", False).option("sep", "\t").csv(train_path).select("_c1")
val_df = spark.read.option("header", False).option("sep", "\t").csv(val_path).select("_c1")
acoustic_df = spark.read.option("header", True).option("sep", "\t").csv(acoustic_path)

test_dfs = [
    spark.read.option("header", False).option("sep", "\t").csv(path).select("_c1")
    for path in test_paths
]
test_df = test_dfs[0]
for df in test_dfs[1:]:
    test_df = test_df.union(df)

# Combine all and rename column to 'track_id'
all_track_ids_df = train_df.union(val_df).union(test_df).distinct().withColumnRenamed("_c1", "track_id")
print("Unique track_ids:", all_track_ids_df.count())

# Join with acoustic features
joined_df = acoustic_df.join(all_track_ids_df, on="track_id")
joined_df.show(truncate=False)
print("Joined count:", joined_df.count())

# Get list of feature columns (excluding track_id)
feature_columns = [col for col in acoustic_df.columns if col != "track_id"]
joined_df.coalesce(1).write \
    .option("header", False) \
    .option("sep", "\t") \
    .mode("overwrite") \
    .csv("joined_features")
#
output_dir = "output_features"  # Directory to store output files
for feature in feature_columns:
    # Select track_id and the feature column
    feature_df = joined_df.select("track_id", feature)

    # Define output path for the feature
    output_path = f"{output_dir}/{feature}.tsv"

    # Write to a single tab-separated file without header
    feature_df.coalesce(1).write \
        .option("header", False) \
        .option("sep", "\t") \
        .mode("overwrite") \
        .csv(output_path)

    print(f"Written {feature} to {output_path}")

# Stop Spark session
spark.stop()