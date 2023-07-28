import pandas as pd
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from joblib import dump
import numpy as np
#########################################################################################################################
SUM = False #Set true to test avg model
# Initialize an empty list to store the data frames
dfs = []
#############################################################################################################################
 # Specify the base folder path
if SUM == True:  
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data'
else:
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data'
folder_name = 'data_250'
#############################################################################################################################
# Loop through each participant folder 
for participant_no in range(1, 6):
    # Construct the folder path for the current participant
    folder_path = f'{base_folder_path}\\{participant_no}'
    
    #Train with data of 250ms time window
    #folder_path = os.path.join(base_folder_path, f"{participant_no}\\{folder_name}")
    # Retrieve the file paths of all CSV files in the current participant folder
    file_paths = glob.glob(f'{folder_path}/*.csv')
    
    # Loop through each file path and read the CSV file as a data frame
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dfs.append(df)

# Concatenate the data frames into a single data frame
merged_df = pd.concat(dfs, ignore_index=True)
#############################################################################################################################

def normalize(emg_features_final):
    scaler = StandardScaler()
    norm_features = []
    
    for features in emg_features_final:
        normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()
        norm_features.append(normalized)
    
    norm_features = np.array(norm_features)
    return norm_features

#########################################################################################################################
# Load dataset
X_data = merged_df.drop(["Label"], axis=1)
y_data = merged_df["Label"]
X_data = X_data.values
y_data = y_data.values

# split into training and test datasets
x_trainval, x_test, y_trainval, y_test = train_test_split(X_data, y_data,
                                                          test_size=0.25,
                                                          stratify=y_data,
                                                          random_state=1004)
# normalize features
scaler = StandardScaler()
scaler.fit(x_trainval)
x_trainval = scaler.transform(x_trainval)
x_test = scaler.transform(x_test)

# #normalize for 250 window/ real time model
# x_trainval = normalize(x_trainval)
# x_test = normalize(x_test)

#########################################################################################################################

#Parameter Study
svm = SVC()

##### predefine candidate for finding best hyperparameter - Support Vector Machine
parameter = {'kernel' : ['linear','rbf'], 
              "C": [1, 10, 100, 1000], 
              'gamma': [1, 1e-1, 1e-2, 1e-3, 1e-4]}

grid_search = GridSearchCV(estimator= svm, param_grid=parameter, n_jobs=-1, cv=10)
grid_search.fit(x_trainval, y_trainval)

svm_test_score = grid_search.score(x_test, y_test)
print("SVM Test Score : {0}".format(svm_test_score))
print(grid_search.best_params_)
best_params = grid_search.best_params_
result = grid_search.cv_results_
df_result = pd.DataFrame(result)
svm_model = SVC(**best_params)
svm_model.fit(x_trainval, y_trainval)
y_pred = svm_model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

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
    plt.ylabel('redicted labels')
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))

if SUM == True:
    df_result.to_csv("./svm_grid_result_summed.csv", encoding="utf-8-sig", index=False)
    dump(svm_model, './Model/SVM_summed.joblib')
    print_confusion_matrix(y_test, y_pred, 'Confusion Matrix of SVM trained with summed EMG data')
else:
    df_result.to_csv("./svm_grid_result_individual.csv", encoding="utf-8-sig", index=False)
    dump(svm_model, './Model/SVM_individual.joblib')
    print_confusion_matrix(y_test, y_pred, 'Confusion Matrix of SVM trained with individual EMG data')


