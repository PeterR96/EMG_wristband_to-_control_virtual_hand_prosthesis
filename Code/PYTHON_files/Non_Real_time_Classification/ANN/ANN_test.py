# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 14:02:56 2023

@author: Peter Rott
"""
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
SUM = False
#############################################################################
# Specify the base folder path
if SUM == True:
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data\Test'
else:  
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data\Test'

dfs = []
#############################################################################################################################
#Test all data at once
folder_path = f'{base_folder_path}'
# Retrieve the file paths of all CSV files in the current participant folder
file_paths = glob.glob(f'{folder_path}/*.csv')
# Loop through each file path and read the CSV file as a data frame
for file_path in file_paths:
      df = pd.read_csv(file_path)
      dfs.append(df)
# Concatenate the data frames into a single data frame
merged_df = pd.concat(dfs, ignore_index=True)
#############################################################################################################################
#Test the sequences one by one
# test_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data\Test'
# filename = './labeled_Feature_Data_test_data_10.csv'
# merged_df = pd.read_csv(test_folder_path+filename)
#############################################################################################################################
# Remove Label
X_data = merged_df.drop(["Label"], axis=1)
feature_names = X_data.columns.tolist()
y_data = merged_df["Label"]
X_data = X_data.values
y_data = y_data.values
def normalize(emg_features_final):
    scaler = StandardScaler()
    norm_features = []
    
    for features in emg_features_final:
        normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()
        norm_features.append(normalized)
    
    norm_features = np.array(norm_features)
    return norm_features

# normalize features
scaler = StandardScaler()
scaler.fit(X_data)
X_data = scaler.transform(X_data)
#############################################################################################################################
#confusion matrix and classification report
def print_confusion_matrix(y_true, y_pred, title, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    ax.set_title(title)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))

def accuracy_wrapper(estimator, X, y):
    y_pred = estimator.predict(X)
    
    return accuracy_score(y, np.argmax(y_pred, axis=1))

def f_importances(ANN_model, X_data, y_data, feature_names):
   # Calculate permutation importance
    perm_importance = permutation_importance(
        ANN_model, X_data, y_data, scoring=accuracy_wrapper, n_repeats=10, random_state=0
    )
    importances = perm_importance.importances_mean
    features = np.array(feature_names)
    # Sort feature importances
    sorted_idx = importances.argsort()
    feature_names_sorted = [feature_names[i] for i in sorted_idx]
    importances_sorted = importances[sorted_idx]
    
    # Print feature importances
    for feature, importance in zip(feature_names_sorted, importances_sorted):
        print(f"{feature}: {importance}")
    
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(features)))  # Increase figure height based on the number of features
    ax.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    ax.set_xlabel("Feature Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Visualizing Important Features ANN")
    ax.invert_yaxis()  # Invert the y-axis to display features from top to bottom
    plt.tight_layout()  # Adjust spacing between the bars and labels
    plt.show()
    plt.savefig('Feature_relevance_ann.png', dpi=300)
    
    # Print feature importances
    for feature, importance in zip(feature_names, importances):
        print(f"{feature}: {importance}")
    
    # Plot feature importances
    plt.barh(feature_names, importances)
    plt.xlabel("Feature Importance")
    plt.show()
    
if SUM == True:
    ANN_model =tf.keras.models.load_model('./Parameter_Study/Model/ANN_sumed.h5')
    y_pred = ANN_model.predict(X_data)
    accuracy = accuracy_score(y_data, np.argmax(y_pred, axis=1))
    print('Accuracy:', accuracy)
    merged_array = np.column_stack((y_data, np.argmax(y_pred, axis=1)))
    print_confusion_matrix(y_data, np.argmax(y_pred, axis=1), 'Confusion Matrix of ANN with testset of summed EMG data')
    predicted_probabilities = np.max(y_pred, axis=1)
   
else:
    ANN_model =tf.keras.models.load_model('./Parameter_Study/Model/ANN_individual.h5')
    y_pred = ANN_model.predict(X_data)
    accuracy = accuracy_score(y_data, np.argmax(y_pred, axis=1))
    print('Accuracy:', accuracy)
    merged_array = np.column_stack((y_data, np.argmax(y_pred, axis=1)))
    label_vector = pd.DataFrame(np.argmax(y_pred, axis=1))
    print_confusion_matrix(y_data, np.argmax(y_pred, axis=1), 'Confusion Matrix of ANN with testset of individual EMG data')
    #f_importances(ANN_model,X_data,y_data, feature_names)

#label_vector.to_csv(test_folder_path+'./Predicted_labels/predicted_labels_05.csv', index=False)