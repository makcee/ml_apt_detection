# Read the text file containing feature numbers and correlation coefficients
correlation_coefficients = []
with open('features.txt', 'r') as file:
    for line in file:
        feature_num, cj = line.strip().split('\t')
        correlation_coefficients.append((int(feature_num), float(cj)))

# Sort the correlation coefficients list in descending order based on the absolute value of cj
correlation_coefficients.sort(key=lambda x: abs(x[1]), reverse=True)

# Select the top 1,000,000 correlated features
top_features = correlation_coefficients[:1000000]

# Save the top_features list to a text file
with open('top_correlated_features.txt', 'w') as file:
    for feature_num, cj in top_features:
        file.write(f"{feature_num}\t{cj}\n")
