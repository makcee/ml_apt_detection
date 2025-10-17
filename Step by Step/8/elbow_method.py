import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# Load the train dataset from the CSV file
train_df = pd.read_csv('train_data.csv')

# Extract features (X) and labels (y) from the train dataset
X_train = train_df.drop(columns=['label']).values

# Calculate PCA with varying number of components
pca = PCA()
pca.fit(X_train)

# Plot the cumulative explained variance ratio
plt.plot(np.cumsum(pca.explained_variance_ratio_[:40]))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Elbow Method for Optimal Number of Components')
plt.grid(True)
plt.show()
