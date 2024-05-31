import os
from src.data_preprocessing import preprocess_data

input_dir = "data/raw_data/"
output_dir = "data/processed_data/"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

preprocess_data(input_dir, output_dir)
