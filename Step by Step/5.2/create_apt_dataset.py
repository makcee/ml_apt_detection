import csv

# Step 1: Read sample numbers from samples_with_label_3.txt
with open('samples_with_label_3.txt', 'r') as txtfile:
    sample_numbers_with_label_3 = set(int(line.strip()) for line in txtfile)

# Step 2: Read combined_data.csv and filter rows
filtered_rows = []
with open('combined_data.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)  # Read the header row
    for row in reader:
        sample_number = int(row[0])  # Assuming the sample number is in the first column
        label = int(row[-1])         # Assuming the label is in the last column
        if label == -1 or sample_number in sample_numbers_with_label_3:
            filtered_rows.append(row)
            print(f"Sample Number: {sample_number}, Label: {label}")

# Step 3: Write the filtered rows to a new CSV file
with open('filtered_combined_data.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)  # Write the header row
    writer.writerows(filtered_rows)

print("Filtered CSV file created successfully.")
