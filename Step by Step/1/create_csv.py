#---------------------------------------- Step 1 ----------------------------------------#
import csv

# Function to categorize the score
def categorize_score(score):
    if score == 0:
        return -1
    elif score >= 0.3:
        return 1
    else:
        return 0  # Marked as 0 to indicate it's to be skipped

# Read pace_classification.txt and categorize scores
with open('pace_classification.txt', 'r') as file:
    scores = file.readlines()
    categorized_scores = [categorize_score(float(score.strip())) for score in scores]

# Create CSV file
with open('labeled_based_on_score.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sample', 'label'])  # Write header
    for i, score in enumerate(categorized_scores, start=1):
        writer.writerow([i, score])

print("CSV file created successfully.")



#---------------------------------------- Step 2 ----------------------------------------#
import pandas as pd
import json

# Read the feature matrix file
df_features = pd.read_csv('pace_feature_matrix.txt', sep='\t', header=None, names=['sample', 'feature', 'count'])

# Create a dictionary to store features for each sample
sample_features = {}
calculated_row = 0
# Iterate through the feature matrix and populate the dictionary
for index, row in df_features.iterrows():
    sample = row['sample'] + 1  # Adjusting sample number to start from 1
    feature = row['feature']
    count = row['count']
    if sample not in sample_features:
        sample_features[sample] = {}
    sample_features[sample][feature] = count
    print(calculated_row)
    calculated_row += 1

# Convert dictionary values to int
sample_features = {k: {int(k2): int(v2) for k2, v2 in v.items()} for k, v in sample_features.items()}

# Convert int64 keys to Python integers
sample_features = {int(k): v for k, v in sample_features.items()}

# Save the sample_features dictionary to a JSON file
with open('sample_features.json', 'w') as file:
    json.dump(sample_features, file, indent=4)

print("Sample features saved successfully.")



#---------------------------------------- Step 3 ----------------------------------------#
import json
import pandas as pd
from collections import Counter

# Load the sample features dictionary from the saved JSON file
with open('sample_features.json', 'r') as file:
    sample_features = json.load(file)

# Read the CSV file containing all labeled samples
df_labeled_based_on_score = pd.read_csv('labeled_based_on_score.csv')

# Extract the samples
samples = df_labeled_based_on_score['sample'].tolist()

# Convert dictionary values to int
sample_features = {int(k): {int(k2): int(v2) for k2, v2 in v.items()} for k, v in sample_features.items()}

# Count the occurrences of each feature across samples
feature_counts = Counter(feature for features in sample_features.values() for feature in features.keys())

# Select the top 100,000 most frequently occurring features
top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10000]

# Create a list to store rows
rows = []

calculated_row = 0
# Iterate through the samples and fill in the feature values
for sample_num in samples:
    row_data = [sample_num] + [sample_features[sample_num].get(feature_num, 0) for feature_num, _ in top_features]
    rows.append(row_data)
    print(calculated_row)
    calculated_row += 1

# Create DataFrame from rows
df_csv = pd.DataFrame(rows, columns=['sample'] + [f'feature{feat}' for feat, _ in top_features])

# Merge the labels with the top features DataFrame based on the sample numbers
df_merged = pd.merge(df_csv, df_labeled_based_on_score[['sample', 'label']], on='sample')

# Filter out rows with label 0
df_merged = df_merged[df_merged['label'] != 0]

# Save the merged DataFrame to a CSV file
df_merged.to_csv('top10000_freq.csv', index=False)

print("CSV file with top features created successfully.")
