import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import kagglehub

# Download latest version
path = kagglehub.dataset_download("uom190346a/water-quality-and-potability")

print("Path to dataset files:", path)

for file_name in os.listdir(path):
    if file_name.endswith(".csv"):  # Check if the file is a CSV
        csv_path = os.path.join(path, file_name)
        print("Reading CSV file:", csv_path)
        data = pd.read_csv(csv_path)  # Read the CSV file into a pandas DataFrame

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

data_path = os.path.join("data", "raw")

os.makedirs(data_path)

train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)