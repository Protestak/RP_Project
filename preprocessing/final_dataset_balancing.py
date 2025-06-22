from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from functools import reduce
from pyspark.sql import DataFrame

spark = SparkSession.builder.appName("FilterCommonUsers").getOrCreate()


train_df = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/train_set.csv/train.csv")
val_df = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/val_set.csv/val.csv")
test_df_15 = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/test_set_age15.csv")
test_df_16 = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/test_set_age16.csv")
test_df_17 = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/test_set_age17.csv")
test_df_18 = spark.read.option("sep", "\t").option("header", "true").csv("output_balanced/test_set_age18.csv")


train_users = train_df.select("user_id").distinct()
val_users = val_df.select("user_id").distinct()

train_val_users = train_users.intersect(val_users)


test_users_15 = test_df_15.select("user_id").distinct().join(train_val_users, on="user_id")
test_users_16 = test_df_16.select("user_id").distinct().join(train_val_users, on="user_id")
test_users_17 = test_df_17.select("user_id").distinct().join(train_val_users, on="user_id")
test_users_18 = test_df_18.select("user_id").distinct().join(train_val_users, on="user_id")

# Get the minimum number of eligible users across all age groups
min_users = min(
    test_users_15.count(),
    test_users_16.count(),
    test_users_17.count(),
    test_users_18.count()
)


sampled_15 = test_users_15.limit(min_users)
sampled_16 = test_users_16.limit(min_users)
sampled_17 = test_users_17.limit(min_users)
sampled_18 = test_users_18.limit(min_users)


final_users_df = reduce(DataFrame.unionAll, [sampled_15, sampled_16, sampled_17, sampled_18]).distinct()


train_final = train_df.join(final_users_df, on="user_id", how="inner")
val_final = val_df.join(final_users_df, on="user_id", how="inner")
test_15_final = test_df_15.join(final_users_df, on="user_id", how="inner")
test_16_final = test_df_16.join(final_users_df, on="user_id", how="inner")
test_17_final = test_df_17.join(final_users_df, on="user_id", how="inner")
test_18_final = test_df_18.join(final_users_df, on="user_id", how="inner")


print(f"Sampled users per test set: {min_users}")
print(f"Total selected users: {final_users_df.count()}")
print(f"Train set user count: {train_final.select('user_id').distinct().count()}")
print(f"Validation set user count: {val_final.select('user_id').distinct().count()}")
print(f"Test age 15 user count: {test_15_final.select('user_id').distinct().count()}")
print(f"Test age 16 user count: {test_16_final.select('user_id').distinct().count()}")
print(f"Test age 17 user count: {test_17_final.select('user_id').distinct().count()}")
print(f"Test age 18 user count: {test_18_final.select('user_id').distinct().count()}")

output_path = "output_balanced/filtered"
train_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/train")
val_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/val")
test_15_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/test_age15")
test_16_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/test_age16")
test_17_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/test_age17")
test_18_final.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(f"{output_path}/test_age18")