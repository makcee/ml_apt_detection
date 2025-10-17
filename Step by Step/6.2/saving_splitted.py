"""
# Saving splitted datasets
"""

"""## Importing the libraries"""

import pandas as pd

"""## Importing the dataset"""

data = pd.read_csv('filtered_combined_data.csv')
X = data.iloc[:, 1:49043].values
y = data.iloc[:, 49043].values

"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Convert the train and test sets to DataFrames
train_df = pd.DataFrame(data=X_train)
train_df['label'] = y_train
test_df = pd.DataFrame(data=X_test)
test_df['label'] = y_test

# Save the train and test DataFrames as CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

print("Train and test datasets saved as train_data.csv and test_data.csv, respectively.")
