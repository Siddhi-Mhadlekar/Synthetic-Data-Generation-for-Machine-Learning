import os
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit

# Initialize Spark session
spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()

# Load images from the data folder
image_dir = "data/sample_images/"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png")]

# Create a DataFrame
df = pd.DataFrame(image_files, columns=["image_path"])
df["label"] = "synthetic"

# Convert to Spark DataFrame
spark_df = spark.createDataFrame(df)

# Save as Parquet file
spark_df.write.parquet("data/processed/images.parquet")

print("Data preprocessing completed.")
