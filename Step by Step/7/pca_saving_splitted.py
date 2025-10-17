"""
# Saving splitted PCA datasets
"""

"""## Importing the libraries"""
from datetime import datetime
current_time = datetime.now()
print(current_time)

import pandas as pd

# Load the train and test datasets from the CSV files
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

current_time = datetime.now()
print(current_time)

# Extract features (X) and labels (y) from the train and test datasets
X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values
X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

current_time = datetime.now()
print(current_time)

# Applying PCA
from sklearn.decomposition import PCA
for i in range (990 , 1010 + 1):
    pca = PCA(n_components = i)
    new_X_train = pca.fit_transform(X_train)
    new_X_test = pca.transform(X_test)

    # Convert the train and test sets to DataFrames
    train_df = pd.DataFrame(data=new_X_train)
    train_df['label'] = y_train
    test_df = pd.DataFrame(data=new_X_test)
    test_df['label'] = y_test

    # Save the train and test DataFrames as CSV files
    train_df.to_csv(f'pca{i}_train_data.csv', index=False)
    test_df.to_csv(f'pca{i}_test_data.csv', index=False)

    print(f"PCA train and test datasets for n_components={i} saved.")
    current_time = datetime.now()
    print(current_time)


