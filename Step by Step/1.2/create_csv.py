#---------------------------------------- Step 1 ----------------------------------------#
import csv

# Function to categorize the score
def categorize_score(score):
    if score == 0:
        return -1
    elif score == 3:
        return 3
    elif score >= 0.3:
        return 1
    else:
        return 0  # Marked as 0 to indicate it's to be skipped

# Read pace_classification.txt and categorize scores
with open('pace_classification.txt', 'r') as file:
    scores = file.readlines()
    categorized_scores = [categorize_score(float(score.strip())) for score in scores]

# Create CSV file
with open('apt_labeled_based_on_score.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['sample', 'label'])  # Write header
    for i, score in enumerate(categorized_scores, start=1):
        writer.writerow([i, score])

print("CSV file created successfully.")



#---------------------------------------- Step 2 ----------------------------------------#
import csv

# Step 1: Read the existing CSV file and filter out rows with label 0
filtered_rows = []
sample_numbers_with_label_3 = []

with open('apt_labeled_based_on_score.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header
    for row in reader:
        sample_number, label = int(row[0]), int(row[1])
        if label != 0:  # Keep rows with labels 1, -1, and 3
            filtered_rows.append(row)
        if label == 3:
            sample_numbers_with_label_3.append(sample_number)

# Step 2: Write the filtered rows to a new CSV file
with open('apt_filtered.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # Write header
    writer.writerows(filtered_rows)

# Step 3: Write the sample numbers with label 3 to a text file
with open('samples_with_label_3.txt', 'w') as txtfile:
    for sample_number in sample_numbers_with_label_3:
        txtfile.write(f"{sample_number}\n")

print("Filtered CSV and text files created successfully.")

