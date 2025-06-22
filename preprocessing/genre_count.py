from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, count, sum as spark_sum, round

spark = SparkSession.builder.getOrCreate()

metadata_df = spark.read.option("header", True).option("sep", "\t") \
    .csv("training_set_2012_tsv/part-00000-abc2d2c5-a115-46d9-8217-2953186ac59a-c000.csv")


balanced_df_train = spark.read.option("header", False).option("sep", "\t") \
    .csv("output_balanced/drop2/train/train.csv") \
    .toDF("user_id", "track_id")

balanced_df_validation = spark.read.option("header", False).option("sep", "\t") \
    .csv("output_balanced/filtered/val/val.csv") \
    .toDF("user_id", "track_id")



artist_genres_df = spark.read.option("header", False).option("sep", "\t") \
    .csv("lfm/artists_valid 1.tsv") \
    .toDF("artist_id", "artist_name", "genre_annotations")


joined_df = balanced_df_train.join(
    metadata_df.select("user_id", "track_id", "artist_id"),
    on=["user_id", "track_id"],
    how="inner"
)


final_df = joined_df.join(
    artist_genres_df.select("artist_id", "genre_annotations"),
    on="artist_id",
    how="inner"
)


result_df = final_df.select("user_id", "track_id", "artist_id", "genre_annotations")
result_df_2 = result_df.select("track_id", "genre_annotations").distinct()
# Step 1: Split genre string by "," and explode into individual rows
genres_exploded = result_df_2.select(
    explode(split(col("genre_annotations"), ",")).alias("genre")
)
from pyspark.sql.functions import split, explode, trim, lower

cleaned_genres = genres_exploded.select(trim(lower(col("genre"))).alias("genre")).filter(col("genre") != "")


unique_genre_count = cleaned_genres.distinct().count()

print(f"Number of unique genres: {unique_genre_count}")
