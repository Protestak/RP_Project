from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, month, countDistinct, count, when, avg,
    collect_set, size, array_intersect, desc, lit, row_number
)
from pyspark.sql.window import Window


spark = SparkSession.builder.appName("TrackSplitAndFilter").getOrCreate()


df = spark.read.option("header", True).option("delimiter", "\t").csv(
    "training_set_2012_tsv/part-00000-abc2d2c5-a115-46d9-8217-2953186ac59a-c000.csv"
)


df = df.withColumn("date", to_date(col("timestamp"))) \
       .withColumn("month", month(col("date")))


df = df.filter(col("age_at_listen").between(15, 18))


train_df = df.filter(col("month").between(1, 5))
val_df = df.filter(col("month").between(6, 7))
test_df = df.filter(col("month").between(8, 10))


train_counts = train_df.select("age_at_listen", "user_id", "track_id").dropDuplicates() \
                       .groupBy("age_at_listen", "user_id") \
                       .agg(countDistinct("track_id").alias("train_count"))

val_counts = val_df.select("age_at_listen", "user_id", "track_id").dropDuplicates() \
                   .groupBy("age_at_listen", "user_id") \
                   .agg(countDistinct("track_id").alias("val_count"))

test_counts = test_df.select("age_at_listen", "user_id", "track_id").dropDuplicates() \
                     .groupBy("age_at_listen", "user_id") \
                     .agg(countDistinct("track_id").alias("test_count"))


qualified_users = train_counts.join(val_counts, ["age_at_listen", "user_id"]) \
                              .join(test_counts, ["age_at_listen", "user_id"]) \
                              .filter(
                                  (col("train_count") >= 25) &
                                  (col("val_count") >= 7) &
                                  (col("test_count") >= 7)
                              ) \
                              .select("age_at_listen", "user_id")


df_qualified = df.join(qualified_users, on=["age_at_listen", "user_id"], how="inner")


df_binarized = df_qualified.select("age_at_listen", "user_id", "track_id", "month") \
                           .dropDuplicates(["user_id", "track_id"])

train_bin = df_binarized.filter(col("month").between(1, 5))
all_bin = df_binarized  # all months: 1–10
val_bin = df_binarized.filter(col("month").between(6, 7))
test_bin = df_binarized.filter(col("month").between(8, 10))

val_track_counts = val_bin.groupBy("track_id") \
                          .agg(countDistinct("user_id").alias("val_user_count"))

test_track_counts = test_bin.groupBy("track_id") \
                            .agg(countDistinct("user_id").alias("test_user_count"))


train_track_counts = train_bin.groupBy("track_id") \
                              .agg(countDistinct("user_id").alias("train_user_count"))

all_track_counts = all_bin.groupBy("track_id") \
                          .agg(countDistinct("user_id").alias("total_user_count"))


track_counts = train_track_counts.join(all_track_counts, on="track_id")
track_counts = track_counts \
    .join(val_track_counts, on="track_id", how="left") \
    .join(test_track_counts, on="track_id", how="left") \
    .fillna(0, subset=["val_user_count", "test_user_count"])

qualified_tracks = track_counts.filter(
    (col("train_user_count") >= 5) &
    (col("total_user_count") >= 10) &
    (col("val_user_count") >= 1) &
    (col("test_user_count") >= 1)
).select("track_id")


df_final = df_binarized.join(qualified_tracks, on="track_id", how="inner")


# ---- AGE 15 + OVERLAP SELECTION SECTION ---- #

from pyspark.sql.functions import rand


user_counts_by_age = df_final.select("age_at_listen", "user_id").distinct() \
    .groupBy("age_at_listen").agg(countDistinct("user_id").alias("num_users"))

min_user_count = user_counts_by_age.agg({"num_users": "min"}).collect()[0][0]


balanced_users = []
for age in [15, 16, 17, 18]:
    users = df_final.filter(col("age_at_listen") == age).select("user_id").distinct()
    sampled = users.orderBy(rand()).limit(min_user_count)
    sampled = sampled.withColumn("age_at_listen", lit(age))
    balanced_users.append(sampled)

selected_users = balanced_users[0]
for other_age_df in balanced_users[1:]:
    selected_users = selected_users.unionByName(other_age_df)

df_balanced = df_final.join(selected_users, on=["age_at_listen", "user_id"], how="inner")


train_df = df_balanced.filter(col("month").between(1, 5))
val_df = df_balanced.filter(col("month").between(6, 7))
test_df = df_balanced.filter(col("month").between(8, 10))

test_users = test_df.select("user_id").distinct()
train_df = train_df.join(test_users, on="user_id", how="inner")
val_df = val_df.join(test_users, on="user_id", how="inner")

train_df.select("user_id", "track_id") \
    .coalesce(1) \
    .write.option("header", True).option("delimiter", "\t") \
    .csv("output_balanced/train_set.csv", mode="overwrite")

val_df.select("user_id", "track_id") \
    .coalesce(1) \
    .write.option("header", True).option("delimiter", "\t") \
    .csv("output_balanced/val_set.csv", mode="overwrite")

for age in [15, 16, 17, 18]:
    test_df.filter(col("age_at_listen") == age).select("user_id", "track_id") \
        .coalesce(1) \
        .write.option("header", True).option("delimiter", "\t") \
        .csv(f"output_balanced/test_set_age{age}.csv", mode="overwrite")

