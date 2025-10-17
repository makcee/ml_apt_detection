# Artificial Neural Network

# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from datetime import datetime

# Load the train and test datasets from the CSV files
train_df = pd.read_csv(f'pca300_train_data.csv')
test_df = pd.read_csv(f'pca300_test_data.csv')
# Extract features (X) and labels (y) from the train and test datasets
X_train = train_df.drop(columns=['label']).values
y_train = train_df['label'].values
X_test = test_df.drop(columns=['label']).values
y_test = test_df['label'].values

"""## Feature Scaling"""
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

# Define the number of folds for cross-validation
k_folds = 5

# Define different numbers of hidden layers and neurons
hidden_layers = range(1, 1 + 1)  # From 1 to 20 hidden layers
neurons = range(12, 12 + 1)  # From 1 to 40 neurons

# Open a file to write the results
with open('ANN300 PCA Correlation Results for top 49042.txt', 'w') as file:
    # Perform K-fold cross-validation for each configuration
    for layers in hidden_layers:
        for neuron_count in neurons:
            # Initialize lists to store scores for each fold
            accuracy_scores = []
            precision_scores = []
            recall_scores = []
            f1_scores = []

            # Perform K-fold cross-validation
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
            
            for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
                # Split the data into training and validation sets for this fold
                X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
                y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
                
                # Initialize the ANN
                ann = tf.keras.models.Sequential()
                for _ in range(layers):
                    ann.add(tf.keras.layers.Dense(units=neuron_count, activation='relu'))
                ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
                
                # Compile the ANN
                ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
                
                # Train the ANN on the training fold
                ann.fit(X_train_fold, y_train_fold, batch_size=32, epochs=40, verbose=0)
                
                # Predicting the validation set results
                y_pred = ann.predict(X_val_fold)
                y_pred = (y_pred >= 0.5)

                # # Calculate the F1 score for this fold
                # f1 = f1_score(y_val_fold, y_pred)
                # f1_scores.append(f1)

                # Calculate evaluation metrics
                accuracy = accuracy_score(y_val_fold, y_pred)
                precision = precision_score(y_val_fold, y_pred)
                recall = recall_score(y_val_fold, y_pred)
                f1 = f1_score(y_val_fold, y_pred)
                
                # Append evaluation metrics to lists
                accuracy_scores.append(accuracy)
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

                # # Write the F1 score for this fold to the file
                # file.write(f"Fold {fold_idx}\n")
                # file.write(f"F1-Score: {f1*100:.2f}\n")

            # Write the mean F1 score for this configuration to the file
            file.write(f"{layers} Hidden Layers\n")
            file.write(f"{neuron_count} Neurons Each\n")

            mean_accuracy_score = np.mean(accuracy_scores)
            file.write(f"Mean accuracy_score: {mean_accuracy_score*100:.2f}\n")
            mean_precision_score = np.mean(precision_scores)
            file.write(f"Mean precision_score: {mean_precision_score*100:.2f}\n")
            mean_recall_score = np.mean(recall_scores)
            file.write(f"Mean recall_score: {mean_recall_score*100:.2f}\n")
            mean_f1_score = np.mean(f1_scores)
            file.write(f"Mean F1-Score: {mean_f1_score*100:.2f}\n")

            if neuron_count!=40:
                file.write("----------------------------------------\n")
            current_time = datetime.now()
            print(current_time)
            print(f"{layers} Hidden Layers")
            print(f"{neuron_count} Neurons Each")
        file.write("################################################################################\n")




# # Define the number of folds for cross-validation
# k_folds = 5

# # Define different numbers of units in hidden layers
# hidden_units = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60]

# # Open a file to write the results
# with open('evaluation_results.txt', 'w') as file:
#     # Perform K-fold cross-validation for each number of units
#     for units in hidden_units:
#         current_time = datetime.now()
#         print(current_time)

#         # Initialize lists to store evaluation metrics for each fold
#         accuracy_scores = []
#         precision_scores = []
#         recall_scores = []
#         f1_scores = []
        
#         # Perform K-fold cross-validation
#         skf = StratifiedKFold(n_splits=k_folds, shuffle=True)
        
#         for fold_idx, (train_index, val_index) in enumerate(skf.split(X_train, y_train), 1):
#             # Split the data into training and validation sets for this fold
#             X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
#             y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
            
#             # Initialize the ANN
#             ann = tf.keras.models.Sequential([
#                 tf.keras.layers.Dense(units=units, activation='relu'),
#                 tf.keras.layers.Dense(units=units, activation='relu'),
#                 tf.keras.layers.Dense(units=units, activation='relu'),
#                 tf.keras.layers.Dense(units=1, activation='sigmoid')
#             ])
            
#             # Compile the ANN
#             ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
            
#             # Train the ANN on the training fold
#             ann.fit(X_train_fold, y_train_fold, batch_size=32, epochs=40, verbose=0)
            
#             # Predicting the Test set results
#              y_pred = ann.predict(X_val_fold)
#              y_pred = (y_pred >= 0.5)

#             # Calculate evaluation metrics
#             accuracy = accuracy_score(y_val_fold, y_pred)
#             precision = precision_score(y_val_fold, y_pred)
#             recall = recall_score(y_val_fold, y_pred)
#             f1 = f1_score(y_val_fold, y_pred)
            
#             # Append evaluation metrics to lists
#             accuracy_scores.append(accuracy)
#             precision_scores.append(precision)
#             recall_scores.append(recall)
#             f1_scores.append(f1)
        
#         # Write evaluation metrics to file for this configuration
#         file.write(f"Hidden Units: {units}\n")
#         file.write("Fold\tAccuracy\tPrecision\tRecall\tF1-Score\n")
#         for fold_idx in range(k_folds):
#             accuracy_str = "{:.2f}%".format(accuracy_scores[fold_idx] * 100)
#             precision_str = "{:.2f}%".format(precision_scores[fold_idx] * 100)
#             recall_str = "{:.2f}%".format(recall_scores[fold_idx] * 100)
#             f1_str = "{:.2f}%".format(f1_scores[fold_idx] * 100)
#             file.write(f"{fold_idx+1}\t\t{accuracy_str}\t\t{precision_str}\t\t{recall_str}\t{f1_str}\n")
#         mean_accuracy_str = "{:.2f}%".format(np.mean(accuracy_scores) * 100)
#         mean_precision_str = "{:.2f}%".format(np.mean(precision_scores) * 100)
#         mean_recall_str = "{:.2f}%".format(np.mean(recall_scores) * 100)
#         mean_f1_str = "{:.2f}%".format(np.mean(f1_scores) * 100)
#         file.write(f"Mean Accuracy:\t{mean_accuracy_str}\n")
#         file.write(f"Mean Precision:\t{mean_precision_str}\n")
#         file.write(f"Mean Recall:\t{mean_recall_str}\n")
#         file.write(f"Mean F1-Score:\t{mean_f1_str}\n\n")



# # Initializing the ANN
# ann = tf.keras.models.Sequential()
# # Adding the input layer and the first hidden layer
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# # Adding the second hidden layer
# ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

# best_f1_scores = []
# with open('ANN_results.txt', 'w') as file:
#     for hidden_layers in range (1, 100 + 1):
#         for neurons in range(2, 60 + 1, 2):
#             ann = tf.keras.models.Sequential()  # Initialize the model
#             for i in range (1, hidden_layers + 1):
#                 ann.add(tf.keras.layers.Dense(units=neurons, activation='relu'))
#             ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))  # Add output layer
#             ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
#             ann.fit(X_train, y_train, batch_size = 32, epochs = 40, verbose=0)
#             y_pred = ann.predict(X_test)
#             y_pred = (y_pred >= 0.5)
#             accuracy = accuracy_score(y_test, y_pred)
#             precision = precision_score(y_test, y_pred)
#             recall = recall_score(y_test, y_pred)
#             f1 = f1_score(y_test, y_pred)
#             best_f1_scores.append((f1, hidden_layers, neurons))
#             file.write(f"{hidden_layers} Hidden Layers\n")
#             file.write(f"{neurons} Neurons Each\n")
#             file.write(f"Accuracy: {accuracy*100:.2f} %\n")
#             file.write(f"Precision: {precision*100:.2f} %\n")
#             file.write(f"Recall: {recall*100:.2f} %\n")
#             file.write(f"F1-Score: {f1*100:.2f} %\n")
#             if neurons!=100:
#                 file.write("----------------------------------------\n")
#             current_time = datetime.now()
#             print(current_time)
#             print(f"{hidden_layers} Hidden Layers")
#             print(f"{neurons} Neurons Each")
#         file.write("################################################################################\n")

#     # Sort best F1 scores in descending order
#     best_f1_scores.sort(reverse=True)
#     # Write best F1 scores to the file
#     file.write("Best F1 Scores (Descending Order):\n")
#     for score, hidden_layers, neurons in best_f1_scores:
#         file.write(f"Hidden Layers: {hidden_layers}, Neurons Each: {neurons}, F1-Score: {score*100:.2f} %\n")
            


# for i in range (1, 15 + 1):
#     ann.add(tf.keras.layers.Dense(units=58, activation='relu'))
# # Adding the output layer
# ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# # Compiling the ANN
# ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
# # Training the ANN on the Training set
# ann.fit(X_train, y_train, batch_size = 32, epochs = 40)


# # Predicting the Test set results
# y_pred = ann.predict(X_test)
# y_pred = (y_pred >= 0.5)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# print(f"The accuracy is {accuracy_score(y_test, y_pred)})")
# print(f"The precision is {precision_score(y_test, y_pred)}")
# print(f"The recall is {recall_score(y_test, y_pred)}")
# print(f"The F1-Score is {f1_score(y_test, y_pred)}")


# # Evaluating the model on the Test set
# evaluation = ann.evaluate(X_test, y_test)
# print("Test Loss:", evaluation[0])
# print("Test Accuracy:", evaluation[1])
# print("Test Precision:", evaluation[2])
# print("Test Recall:", evaluation[3])
# # Calculate precision
# precision = evaluation[2]
# # Calculate recall
# recall = evaluation[3]
# # Calculate F1 score
# f1 = 2 * (precision * recall) / (precision + recall)
# print("Test F1 Score:", f1)