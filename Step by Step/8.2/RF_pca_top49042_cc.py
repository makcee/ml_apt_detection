"""
# Random Forest Classification
"""

"""## Importing the libraries"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score

from datetime import datetime

# List to store mean F1 scores and their corresponding number of components
mean_f1_scores = []

# Open the text file in write mode
with open('RF300 PCA Correlation Results for top 49042.txt', 'w') as f:
    # Write the header
    f.write("top49042_correlated with PCA\n")
    f.write(f"The model used is Random Forest classifier\n")
    current_time = datetime.now()
    print(current_time)
    
    # Iterate over each value of n_components
    for i in range(300, 300 + 1):
        # Load the train and test datasets from the CSV files
        train_df = pd.read_csv(f'pca{i}_train_data.csv')
        test_df = pd.read_csv(f'pca{i}_test_data.csv')
        # Extract features (X) and labels (y) from the train and test datasets
        X_train = train_df.drop(columns=['label']).values
        y_train = train_df['label'].values
        X_test = test_df.drop(columns=['label']).values
        y_test = test_df['label'].values

        """## Feature Scaling"""
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        """## Training the Random Forest Classification model on the Training set"""
        classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
        classifier.fit(X_train, y_train)
        """## Predicting the Test set results"""
        y_pred = classifier.predict(X_test)

        # Write the results to the text file
        f.write(f"#####################################################################################\n")
        f.write(f"PCA with {i} components\n")
        f.write(f"The accuracy is {accuracy_score(y_test, y_pred)}\n")
        f.write(f"The precision is {precision_score(y_test, y_pred)}\n")
        f.write(f"The recall is {recall_score(y_test, y_pred)}\n")
        f.write(f"The F1-Score is {f1_score(y_test, y_pred)}\n")
        # f.write(f"The Matthews correlation coefficient is {matthews_corrcoef(y_test, y_pred)}\n")

        """## Applying k-Fold Cross Validation"""
        # Perform cross-validation
        accuracy_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring='accuracy')
        precision_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring='precision')
        recall_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring='recall')
        f1_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring='f1')
        
        mean_f1 = f1_scores.mean()
        mean_f1_scores.append((mean_f1, i))

        f.write("Cross Validation Scores\n")
        f.write(f"Accuracy: Mean = {accuracy_scores.mean()*100:.2f} % , Std = {accuracy_scores.std()*100:.2f} %\n")
        f.write(f"Precision: Mean = {precision_scores.mean()*100:.2f} % , Std = {precision_scores.std()*100:.2f} %\n")
        f.write(f"Recall: Mean = {recall_scores.mean()*100:.2f} % , Std = {recall_scores.std()*100:.2f} %\n")
        f.write(f"F1-Score: Mean = {f1_scores.mean()*100:.2f} % , Std = {f1_scores.std()*100:.2f}% \n")
        
        current_time = datetime.now()
        print(f"{current_time} ---> {i}")

	# Sort the mean F1 scores in descending order
    mean_f1_scores.sort(reverse=True)
		
	# Write the mean F1 scores in descending order
    f.write("#####################################################################################\n")
    f.write("Mean F1 Scores of Cross-Validation in Descending Order:\n")
    for mean_f1, i in mean_f1_scores:
        f.write(f"PCA with {i} components: Mean F1 Score = {mean_f1 * 100:.2f}%\n")

print("All Results saved.")




# # Load the train and test datasets from the CSV files
# n_components = 40
# train_df = pd.read_csv(f'pca{n_components}_train_data.csv')
# test_df = pd.read_csv(f'pca{n_components}_test_data.csv')

# # Extract features (X) and labels (y) from the train and test datasets
# X_train = train_df.drop(columns=['label']).values
# y_train = train_df['label'].values
# X_test = test_df.drop(columns=['label']).values
# y_test = test_df['label'].values

# # Convert labels back to original format if necessary (e.g., for decoding)
# le = LabelEncoder()
# y_train = le.inverse_transform(y_train)
# y_test = le.inverse_transform(y_test)


# """## Training the Random Forest Classification model on the Training set"""

# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

# """## Predicting the Test set results"""

# y_pred = classifier.predict(X_test)
# # print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# # Evaluating the classifier
# # printing every score of the classifier
# # scoring in anything
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.metrics import precision_score, recall_score
# from sklearn.metrics import f1_score, matthews_corrcoef
# from sklearn.metrics import confusion_matrix

# # n_outliers = len(fraud)
# print(f"PCA with {n_components} components")
# print("The model used is Random Forest classifier")

# acc = accuracy_score(y_test, y_pred)
# print(f"The accuracy is {acc}")

# prec = precision_score(y_test, y_pred)
# print("The precision is {}".format(prec))

# rec = recall_score(y_test, y_pred)
# print("The recall is {}".format(rec))

# f1 = f1_score(y_test, y_pred)
# print("The F1-Score is {}".format(f1))

# MCC = matthews_corrcoef(y_test, y_pred)
# print("The Matthews correlation coefficient is {}".format(MCC))


# """## Applying k-Fold Cross Validation"""

# from sklearn.model_selection import cross_val_score
# scoring = {'accuracy' : 'accuracy',
#            'precision': 'precision',
#            'recall': 'recall',
#            'f1_score' : 'f1'}

# accuracy_score = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, scoring=scoring['accuracy'])
# precision_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring=scoring['precision'])
# recall_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring=scoring['recall'])
# f1_scores = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, scoring=scoring['f1_score'])
# print("Accuracy: {:.2f} %".format(accuracy_score.mean()*100))
# print("Standard Deviation: {:.2f} %".format(accuracy_score.std()*100))
# print("Precision: {:.2f} %".format(precision_scores.mean() * 100))
# print("Standard Deviation of Precision: {:.2f} %".format(precision_scores.std() * 100))
# print("Recall: {:.2f} %".format(recall_scores.mean() * 100))
# print("Standard Deviation of Recall: {:.2f} %".format(recall_scores.std() * 100))
# print("F1: {:.2f} %".format(f1_scores.mean() * 100))
# print("Standard Deviation of F1: {:.2f} %".format(f1_scores.std() * 100))

# """## Making the Confusion Matrix"""

# # printing the confusion matrix
# LABELS = ['BENIGN', 'MALICIOUS']
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize =(6, 6))
# sns.heatmap(conf_matrix, xticklabels = LABELS,
# 			yticklabels = LABELS, annot = True, fmt ="d");
# plt.title("Confusion matrix\nFor Random Forest classifier")
# plt.ylabel('True class')
# plt.xlabel('Predicted class')
# plt.show()