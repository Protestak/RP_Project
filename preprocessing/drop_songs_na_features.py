from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("FilterCommonUsers").getOrCreate()

train_df = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/train/train.csv")
val_df = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/val/val.csv")
test_df_15 = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/test_age15/15.csv")
test_df_16 = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/test_age16/16.csv")
test_df_17 = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/test_age17/17.csv")
test_df_18 = spark.read.option("sep", "\t").option("header", "false").csv("output_balanced/filtered/test_age18/18.csv")

ids_to_remove = [17744455, 5415016, 4776283, 28103240]
second_col = train_df.columns[1]

def filter_and_save(df, output_path):
    filtered_df = df.filter(~col(second_col).isin(ids_to_remove))
    filtered_df.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", "false") \
        .option("sep", "\t") \
        .csv(output_path)

filter_and_save(train_df, "output_balanced/dropped2/train_set.csv")
filter_and_save(val_df, "output_balanced/dropped2/val_set.csv")
filter_and_save(test_df_15, "output_balanced/dropped2/test_set_age15.csv")
filter_and_save(test_df_16, "output_balanced/dropped2/test_set_age16.csv")
filter_and_save(test_df_17, "output_balanced/dropped2/test_set_age17.csv")
filter_and_save(test_df_18, "output_balanced/dropped2/test_set_age18.csv")

spark.stop()
