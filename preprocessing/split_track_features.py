from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SplitFeatures").getOrCreate()

joined_csv_path = "output3/track_features_joined/part-00000-ab7cc4f7-652a-42c9-ba1e-b506d417dab6-c000.csv"

# Read joined CSV with header
df = spark.read.option("header", True).option("sep", "\t").csv(joined_csv_path)


feature_columns = [c for c in df.columns if c != "track_id"]

output_base_path = "output3/track_features_separate/"

for feature in feature_columns:
    feature_df = df.select("track_id", feature)
    feature_df.coalesce(1).write.mode("overwrite").option("header", True).option("sep", "\t").csv(f"{output_base_path}{feature}.csv")
    print(f"Wrote feature file: {feature}.csv")
