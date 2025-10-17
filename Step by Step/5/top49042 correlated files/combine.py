import os

# Current directory
directory = os.getcwd()

# Output file path
output_file = 'combined_data.csv'

# Counter to track the number of files processed
file_counter = 0

# Flag to track whether it's the first file
first_file = True

# Open the output file in write mode
with open(output_file, 'w') as output_csv:
    # Iterate through each file in the current directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv') and not filename.startswith("combined_data"):
            filepath = os.path.join(directory, filename)
            # Open the current CSV file
            with open(filepath, 'r') as input_csv:
                # Read each line from the current CSV file
                for i, line in enumerate(input_csv):
                    # Write the header from the first file
                    if first_file or i > 0:
                        output_csv.write(line)
            # After processing the first file, set the flag to False
            first_file = False
            
            # Increment the file counter
            file_counter += 1
            print(f'Processed {file_counter} out of 100 files.')

# Print a message indicating that all files have been processed
print('All files have been combined into one CSV file.')
