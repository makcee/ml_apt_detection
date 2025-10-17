import json
import pandas as pd
import numpy as np

# Load the sample features dictionary from the saved JSON file
with open('sample_features.json', 'r') as file:
    sample_features = json.load(file)

# Read the CSV file containing all labeled samples
df_labeled_based_on_score = pd.read_csv('labeled_based_on_score.csv')

# Extract the samples and their corresponding labels
samples = df_labeled_based_on_score['sample'].tolist()
labels = df_labeled_based_on_score['label'].tolist()

# Convert dictionary values to int
sample_features = {int(k): {int(k2): int(v2) for k2, v2 in v.items()} for k, v in sample_features.items()}

# Create a list to store the uncentered correlation coefficients
correlation_coefficients = []

max_feature_num = 0
for features in sample_features.values():
    max_feature_num = max(max_feature_num, max(features.keys()))
# 8173553

label_array = np.array(labels)
label_norm = np.linalg.norm(label_array)
calculated_row = 1
# Compute the uncentered correlation coefficient for each feature
for feature_num in range(1, max_feature_num + 1):
    feature_values = []
    for sample_num in samples:
        feature_values.append(sample_features.get(sample_num, {}).get(feature_num, 0))
    feature_array = np.array(feature_values)
    feature_norm = np.linalg.norm(feature_array)
    print(calculated_row)
    calculated_row += 1
    if feature_norm == 0:
        continue 
    cj = np.dot(feature_array, label_array) / (feature_norm * label_norm)
    correlation_coefficients.append((feature_num, abs(cj)))



# Save the list to a text file
with open('features.txt', 'w') as file:
    for feature_num, cj in correlation_coefficients:
        file.write(f"{feature_num}\t{cj}\n")


print("Top features saved successfully.")