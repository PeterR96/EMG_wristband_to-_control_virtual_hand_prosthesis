# import libraries
import os
import pandas as pd
import numpy as np
from datetime import datetime
import glob
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
#############################################################################################################################
SUM = True
num_labels = 5
# Initialize an empty list to store the data frames
dfs = []
folder_name = 'data_250'
#############################################################################################################################
 # Specify the base folder path
if SUM == True :  
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data'
    num_features = 7
else:
    base_folder_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data'
    num_features = 42
    
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
scaler.fit(x_trainval)
x_trainval = scaler.transform(x_trainval)
x_test = scaler.transform(x_test)


#normalize for 250 window/ real time model
# x_trainval = normalize(x_trainval)
# x_test = normalize(x_test)

#############################################################################################################################
# set params for gridesearch (tensorboard) parameter study
#HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([2, 3, 4]))
#HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300, 600, 1000]))
#HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.2, 0.3]))

HP_NUM_LAYERS = hp.HParam('num_layers', hp.Discrete([4]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([300]))
HP_DROPOUT = hp.HParam('dropout_rate', hp.Discrete([0.2]))

METRIC_ACCURACY = 'mean_accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning_results').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_LAYERS, HP_NUM_UNITS, HP_DROPOUT],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Mean_accuracy')],
    )
#############################################################################################################################
# build a model
def train_model(x_train, y_train, hparams, log_dir, x_valid=None, y_valid=None, learning_rate=0.001, batchnorm=True, dropout=True, final=False):

    tf.keras.backend.clear_session()

    #----- model structure
    inputs = tf.keras.layers.Input(shape=(num_features,))
    x = inputs

    for i in range(hparams[HP_NUM_LAYERS]):
        x = tf.keras.layers.Dense(hparams[HP_NUM_UNITS])(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
        if dropout:
            x = tf.keras.layers.Dropout(hparams[HP_DROPOUT])(x)

    outputs = tf.keras.layers.Dense(num_labels)(x)
    outputs = tf.keras.layers.Softmax()(outputs)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    #----- model compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    #----- model train
    ckpt_path = os.path.join(os.getcwd(), 'temp.h5')
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir)

    model.fit(x_train, y_train, validation_data=(x_valid, y_valid) if final is False else None,
              batch_size=1024,
              epochs=1000,
              callbacks=[ckpt, tensorboard_cb] if final is False else [tensorboard_cb],
              verbose=0)

    if not x_valid is None:
        model.load_weights(ckpt_path)

    return model
#############################################################################################################################
# build a stratifiedKFold process
def kfold_cv(x_trainval, y_trainval, hparams, run_dir, n_splits=2):
    skf = StratifiedKFold(n_splits=n_splits)
    accs = []
    j=1

    for train_index, valid_index in skf.split(x_trainval, y_trainval):
        print('CV Process: %d / %d ...' % (j, n_splits))
        x_train, x_valid = x_trainval[train_index], x_trainval[valid_index]
        y_train, y_valid = y_trainval[train_index], y_trainval[valid_index]

        trained_model = train_model(x_train, y_train, hparams,
                                    log_dir=os.path.join('logs/hparam_tuning/', run_dir, 'cv-%s' % j),
                                    x_valid=x_valid, y_valid=y_valid,
                                    )
        _, accuracy = trained_model.evaluate(x_valid, y_valid, verbose=0)
        accs.append(accuracy)
        with tf.summary.create_file_writer(os.path.join('logs/hparam_tuning/', run_dir, 'cv_summary')).as_default():
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=j)
        j += 1

    mean_accuracy = sum(accs) / len(accs)

    with tf.summary.create_file_writer(os.path.join('logs/hparam_tuning_results/', run_dir)).as_default():
        hp.hparams(hparams, trial_id=run_dir)
        tf.summary.scalar(METRIC_ACCURACY, mean_accuracy, step=1)

    return mean_accuracy
#############################################################################################################################
# grid search
s_results = {}

session_num = 0
start_time = datetime.now()
print("[%s] Start parameter search for the model" % start_time)

for num_layers in HP_NUM_LAYERS.domain.values:
    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in HP_DROPOUT.domain.values:
            hparams = {
                HP_NUM_LAYERS: num_layers,
                HP_NUM_UNITS: num_units,
                HP_DROPOUT: dropout_rate
                }
            run_name = "run-%d" % session_num

            print()
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})

            mean_accuracy = kfold_cv(x_trainval, y_trainval, hparams, run_name)

            s_results[run_name] = {'hparams': hparams,
                                    'mean_accuracy': mean_accuracy}
            session_num += 1

end_time = datetime.now()
duration_time = (end_time - start_time).seconds
print("[%s] Finish parameter search for the model (time: %d seconds)" % (end_time, duration_time))

# print best params
df_results = pd.DataFrame(s_results).T
best_param = df_results.sort_values(by='mean_accuracy', ascending=False).head(1)['hparams']
print('the best params:\n', {h.name: best_param.item()[h] for h in best_param.item()})
#############################################################################################################################
#Confusion Matrix
def print_confusion_matrix(y_true, y_pred, title, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)   
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)   
    fig, ax = plt.subplots(figsize=(8, 6),dpi=300)
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    ax.set_title(title)
    plt.xlabel('True labels')
    plt.ylabel('Predicted labels')
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))

#############################################################################################################################        
hparams = [4, 300, 0.2]
predictions_df = pd.DataFrame()
# model retrain with whole trainval datasets and best params & model save
if SUM == True :
    final_model = train_model(x_trainval, y_trainval, best_param.item(), log_dir='./logs/ANN_summed_model', final=True)
    final_model.save('./Model/ANN_sumed.h5')
    
    #load saved model and calcuate accuracy with test datasets
    saved_model = tf.keras.models.load_model('./Model/ANN_sumed.h5')
    predictions = saved_model.predict(x_test)
    predictions_df['Predictions'] = np.argmax(predictions, axis=-1)
    print_confusion_matrix(y_test, predictions_df['Predictions'], 'Confusion Matrix of ANN trained with summed EMG data')
    accuracy_model = accuracy_score(np.array(y_test), np.argmax(saved_model.predict(x_test), axis=-1))
    print(accuracy_model)
else:
    final_model = train_model(x_trainval, y_trainval, best_param.item(), log_dir='./logs/ANN_individual_model', final=True)
    final_model.save('./Model/ANN_individual.h5')
    
    #load saved model and calcuate accuracy with test datasets
    saved_model = tf.keras.models.load_model('./Model/ANN_individual.h5')
    predictions = saved_model.predict(x_test)
    predictions_df['Predictions'] = np.argmax(predictions, axis=-1)
    print_confusion_matrix(y_test, predictions_df['Predictions'], 'Confusion Matrix of ANN trained with individual EMG data')
    accuracy_model = accuracy_score(np.array(y_test), np.argmax(saved_model.predict(x_test), axis=-1))
    print(accuracy_model)



