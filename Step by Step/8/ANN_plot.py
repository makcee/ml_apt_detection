import matplotlib.pyplot as plt

# Lists to store combinations and corresponding F1 scores
hidden_layers_list = []
neurons_list = []
f1_scores_list = []

with open('evaluation_results.txt', 'r') as file:
    lines = file.readlines()
    for i, line in enumerate(lines):
        if 'Hidden Layers' in line:
            hidden_layers = int(line.split()[0])
            # if hidden_layers == 5:
            #     break
        elif 'Neurons Each' in line:
            neurons = int(line.split()[0])
        elif 'Mean' in line:
            f1_score = float(line.split()[2]) / 100  # Convert from percentage to fraction
            # Append data to lists
            hidden_layers_list.append(hidden_layers)
            neurons_list.append(neurons)
            f1_scores_list.append(f1_score)

# Find the maximum F1 score and its index
max_f1_score = max(f1_scores_list)
max_f1_index = f1_scores_list.index(max_f1_score)

# # Filter F1 scores list for at least 20 neurons
# filtered_f1_scores = [f1_scores_list[i] for i in range(len(f1_scores_list)) if neurons_list[i] >= 20]

# # Find the maximum F1 score among filtered scores
# max_f1_score = max(filtered_f1_scores)

# # Find the index of the maximum F1 score in the original list
# max_f1_index = f1_scores_list.index(max_f1_score)

# Plot F1 scores for all combinations
plt.figure(figsize=(16, 6))
for i, hidden_layers in enumerate(set(hidden_layers_list)):
    plt.plot([neurons_list[j] for j in range(len(hidden_layers_list)) if hidden_layers_list[j] == hidden_layers], 
             [f1_scores_list[j] for j in range(len(hidden_layers_list)) if hidden_layers_list[j] == hidden_layers], 
             label=f'{hidden_layers} H Layers')

# Calculate midpoint of x-axis and y-axis ranges
x_mid = (min(neurons_list) + max(neurons_list)) / 2
y_mid = (min(f1_scores_list) + max(f1_scores_list)) / 2

# # Annotate the point with the highest F1 score
# plt.annotate(f'Max F1: {max_f1_score*100:.2f}%\n{hidden_layers_list[max_f1_index]} Hidden Layers, {neurons_list[max_f1_index]} Neurons Each', 
#              xy=(neurons_list[max_f1_index], max_f1_score), 
#              xytext=(x_mid, y_mid), 
#              arrowprops=dict(facecolor='black', arrowstyle='->'))

# Set x-axis ticks every 2 numbers
plt.xticks(range(min(neurons_list), max(neurons_list) + 1))

# Add labels and legend to the plot
plt.xlabel('Number of Neurons')
plt.ylabel('F1-Score')
plt.title('F1-Score vs. Number of Neurons for Different Hidden Layers')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))  # Move legend outside plot area
plt.grid(True)
plt.show()




# Create a dictionary to store the range of F1 scores for each hidden layer number
f1_score_ranges = {}

# Iterate over each unique hidden layer number
for hidden_layers in set(hidden_layers_list):
    # Filter the F1 scores corresponding to the current hidden layer number
    target_f1_scores = [f1_scores_list[i] for i in range(len(hidden_layers_list)) if hidden_layers_list[i] == hidden_layers]
    # Calculate the range of F1 scores
    f1_score_range = max(target_f1_scores) - min(target_f1_scores)
    # Store the range in the dictionary
    f1_score_ranges[hidden_layers] = f1_score_range

# Find the hidden layer number with the smallest overall range
min_range_hidden_layers = min(f1_score_ranges, key=f1_score_ranges.get)
min_range_value = f1_score_ranges[min_range_hidden_layers]

# Print the hidden layer number with the smallest overall range
print(f"The hidden layer number with the least variation in F1 scores is: {min_range_hidden_layers}")
print(f"The range of F1 scores for this hidden layer number is: {min_range_value}")




# Create a dictionary to store the range of F1 scores for each number of neurons
f1_score_ranges = {}

# Iterate over each unique number of neurons
for neurons in set(neurons_list):
    # Filter the F1 scores corresponding to the current number of neurons
    target_f1_scores = [f1_scores_list[i] for i in range(len(neurons_list)) if neurons_list[i] == neurons]
    # Calculate the range of F1 scores
    f1_score_range = max(target_f1_scores) - min(target_f1_scores)
    # Store the range in the dictionary
    f1_score_ranges[neurons] = f1_score_range

# Find the number of neurons with the smallest overall range
min_range_neurons = min(f1_score_ranges, key=f1_score_ranges.get)
min_range_value = f1_score_ranges[min_range_neurons]

# Find the corresponding minimum and maximum F1 scores
min_f1_score = min([f1_scores_list[i] for i in range(len(neurons_list)) if neurons_list[i] == min_range_neurons])
max_f1_score = max([f1_scores_list[i] for i in range(len(neurons_list)) if neurons_list[i] == min_range_neurons])

# Print the number of neurons with the smallest overall range
print(f"The number of neurons with the least variation in F1 scores is: {min_range_neurons}")
print(f"The range of F1 scores for this number of neurons is: {min_range_value}")
print(f"The minimum F1 score for this number of neurons is: {min_f1_score}")
print(f"The maximum F1 score for this number of neurons is: {max_f1_score}")


