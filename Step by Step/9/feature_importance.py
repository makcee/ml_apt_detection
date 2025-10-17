from sklearn.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

from datetime import datetime

current_time = datetime.now()
print(current_time)
print("Loading original data (the 49,042 selected features) ...")

# Step 1: Load original data (the 49,042 selected features)
train_df = pd.read_csv('train_data.csv')
test_df = pd.read_csv('test_data.csv')

x_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values

current_time = datetime.now()
print(current_time)
print("Initializing PCA with 300 components (as per our original transformation) ...")

# Step 2: Initialize PCA with 300 components (as per our original transformation)
pca = PCA(n_components=300)

# Fit PCA to original data (recreate PCA transformation)
pca.fit(x_train)

# Get PCA loadings (components) - shape (300, 49042)
pca_loadings = pca.components_

current_time = datetime.now()
print(current_time)
print("Loading PCA-transformed train and test data ...")

# Step 3: Load PCA-transformed train and test data
pca_train_df = pd.read_csv('pca300_train_data.csv')
pca_test_df = pd.read_csv('pca300_test_data.csv')

current_time = datetime.now()
print(current_time)
print("Retraining the LR model ...")

# Separate features and labels
X_train = pca_train_df.drop(columns=['label']).values
y_train = pca_train_df['label'].values
X_test = pca_test_df.drop(columns=['label']).values
y_test = pca_test_df['label'].values

# Step 4: Standardize the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5: Initialize Logistic Regression classifier
classifier = LogisticRegression(random_state=0, max_iter=500)

# Fit the model on the training data
classifier.fit(X_train, y_train)

# Step 6: Predict on the test set
y_pred = classifier.predict(X_test)

# Step 7: Get the Logistic Regression coefficients after training (shape: (1, 300) for binary classification)
lr_coefficients = classifier.coef_

current_time = datetime.now()
print(current_time)
print("Multiplying logistic regression coefficients by PCA loadings to get original feature importance ...")

# Step 8: Multiply logistic regression coefficients by PCA loadings to get original feature importance
# This projects the importance of each PCA component back to the original features
feature_importance = np.dot(lr_coefficients, pca_loadings)  # Shape: (1, 49042)

# Step 9: Get indices of the most important features in descending order
important_features_idx = np.argsort(np.abs(feature_importance))[0][::-1]

# Step 10: Print the top N most important original features
top_n = 10
for idx in important_features_idx[:top_n]:
    print(f"Original feature {idx} has importance {feature_importance[0, idx]}")

# Step 11: Get the feature names of the original dataset (train_df), excluding the label
original_feature_names = train_df.columns[:-1]  # Exclude 'label' column

current_time = datetime.now()
print(current_time)
print("Saving the Feature Importance ...")

# Create a DataFrame for the top features
top_features = pd.DataFrame({
    'Feature': original_feature_names[important_features_idx],
    'Importance': feature_importance[0, important_features_idx]
})

# Step 12: Save the top 300 features and their importance scores to a CSV file
top_features.to_csv('top_feature_importance.csv', index=False)

current_time = datetime.now()
print(current_time)
print("Evaluating model performance and printing metrics ...")

# Step 13: Evaluate model performance and print metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")

current_time = datetime.now()
print(current_time)