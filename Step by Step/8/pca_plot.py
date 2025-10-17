import matplotlib.pyplot as plt

# Read the file and extract the number of components and mean F1 score percentage
data = []
with open('XGBoost mean F1 scores.txt', 'r') as file:
    for line in file:
        if 'PCA with' in line:
            components = int(line.split()[2])
            f1_score = float(line.split()[-1][:-1])
            data.append((components, f1_score))

# Sort the data based on the number of components
sorted_data = sorted(data)

# Extract sorted components and mean F1 scores
sorted_components = [item[0] for item in sorted_data]
f1_scores = [item[1] for item in sorted_data]

# Plot the sorted data
plt.figure(figsize=(12, 6))
plt.plot(sorted_components, f1_scores, marker='o', linestyle='', color='b', markersize=3)
plt.title('Mean F1 Score vs Number of Components')
plt.xlabel('Number of Components')
plt.ylabel('Mean F1 Score (%)')
plt.grid(True)
# plt.xticks(range(min(sorted_components)-1, max(sorted_components)+2, 20))
plt.show()
