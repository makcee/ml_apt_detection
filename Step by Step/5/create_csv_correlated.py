import json
import pandas as pd

# Read the top correlated features from the text file
top_correlated_features = []
with open('top_correlated_features.txt', 'r') as file:
    for line in file:
        feature_num, cj = line.strip().split('\t')
        top_correlated_features.append((int(feature_num), float(cj)))

# Load the sample features dictionary from the saved JSON file
with open('sample_features.json', 'r') as file:
    sample_features = json.load(file)

# Read the CSV file containing all labeled samples
df_labeled_based_on_score = pd.read_csv('labeled_based_on_score.csv')

# Extract the samples
samples = df_labeled_based_on_score['sample'].tolist()

# Convert dictionary values to int
sample_features = {int(k): {int(k2): int(v2) for k2, v2 in v.items()} for k, v in sample_features.items()}

# Select the top correlated features
top_49042_correlated_features = top_correlated_features[:49042]

# Create a list to store rows
rows = []



# Define chunk size
chunk_size = 237

print("reached the for loop")
# Iterate through the samples in chunks
for i in range(0, len(samples), chunk_size):
    chunk_samples = samples[i:i+chunk_size]
    rows = []

    calculated_row = 0
    # Iterate through the samples in the chunk and fill in the feature values
    for sample_num in chunk_samples:
        row_data = [sample_num] + [sample_features[sample_num].get(feature_num, 0) for feature_num, _ in top_49042_correlated_features]
        rows.append(row_data)
        print(calculated_row)
        calculated_row += 1

    # Create DataFrame from rows
    df_csv = pd.DataFrame(rows, columns=['sample'] + [f'feature{feat}' for feat, _ in top_49042_correlated_features])

    # Merge the labels with the top features DataFrame based on the sample numbers
    df_merged = pd.merge(df_csv, df_labeled_based_on_score[['sample', 'label']], on='sample')

    # Filter out rows with label 0
    df_merged = df_merged[df_merged['label'] != 0]

    # Save the merged DataFrame to a CSV file
    df_merged.to_csv(f'{i+1}to{i+chunk_size}_correlated.csv', index=False)

    print(f"CSV file with samples {i+1} to {i+chunk_size} created successfully.")




# calculated_row = 0
# # Iterate through the samples and fill in the feature values
# for sample_num in samples:
#     row_data = [sample_num] + [sample_features[sample_num].get(feature_num, 0) for feature_num, _ in top_49042_correlated_features]
#     rows.append(row_data)
#     print(calculated_row)
#     calculated_row += 1

# # Create DataFrame from rows
# df_csv = pd.DataFrame(rows, columns=['sample'] + [f'feature{feat}' for feat, _ in top_49042_correlated_features])

# # Merge the labels with the top features DataFrame based on the sample numbers
# df_merged = pd.merge(df_csv, df_labeled_based_on_score[['sample', 'label']], on='sample')

# # Filter out rows with label 0
# df_merged = df_merged[df_merged['label'] != 0]

# # Save the merged DataFrame to a CSV file
# df_merged.to_csv('top49042_correlated.csv', index=False)
# # df_merged.to_csv('ordered_correlated.csv', index=False)

# print("CSV file with top 49042 correlated features created successfully.")
# # print("CSV file with correlated features created successfully.")
