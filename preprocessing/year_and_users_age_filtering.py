#
from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

spark = SparkSession.builder.appName("TeenListeningJoin").getOrCreate()

events_df = spark.read.option("header", True).option("sep", "\t").csv("lfm/listening-events.tsv")


teen_listens = events_df.filter((events_df.age_at_listen >= 12) & (events_df.age_at_listen <= 18))

teen_tracks = teen_listens.select("age_at_listen", "track_id").distinct()

teen_tracks = teen_tracks.repartition("age_at_listen").cache()

entry_count = teen_tracks.groupBy("age_at_listen") \
    .agg(countDistinct("track_id").alias("track_count"))

# Step 7: Show result
entry_count.show()

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os

# Step 1: Create Spark session with better memory config
spark = SparkSession.builder \
    .appName("TeenListeningAnalysis") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

events_df = spark.read.option("header", True).option("sep", "\t").csv("lfm/listening-events.tsv")

filtered_events_df = events_df.filter(
    (col("age_at_listen") >= 12) &
    (col("age_at_listen") <= 18) &
    (col("listen_year") == 2012)
)


acoustics_df = spark.read.option("header", True).option("sep", "\t").csv("LFM-BeyMS/creation/acoustic_features_lfm_id.tsv")
from pyspark.sql.functions import col


non_null_columns = [c for c in acoustics_df.columns if acoustics_df.filter(col(c).isNull()).count() == 0]


acoustics_df_clean = acoustics_df.select(non_null_columns)

joined_events_df = filtered_events_df.join(acoustics_df_clean, on="track_id", how="inner")

# Define output path
output_path = os.path.expanduser("./training_set_2012_tsv")


joined_events_df.coalesce(1).write \
    .option("header", True) \
    .option("sep", "\t") \
    .mode("overwrite") \
    .csv(output_path)

print("Data written to:", output_path)

