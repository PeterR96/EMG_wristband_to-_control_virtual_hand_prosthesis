# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 15:04:02 2023

@author: Peter Rott
"""
import Feature_extraction_FCNs as FE_FCNs
import pandas as pd
frame = 250
step = 200
sfreq = 1000
channel_name = 'Right Masseter'
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 

#import labeld data
export_path_train = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Feature_Extraction_split_dataset\Training_data'
export_path_test = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Feature_Extraction_split_dataset\Test_data'
import_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Load_and_procsess_Data\Exported_data_for_labeling'

_fist = '\Summed_EMG_Enveleoped_Fist_02.csv'
_spred = '\Summed_EMG_Enveleoped_Spred_02.csv'
_pinch = '\Summed_EMG_Enveleoped_Pinch_02.csv'
_thump = '\Summed_EMG_Enveleoped_Thump_02.csv'

data_fist = pd.read_csv(import_path + _fist)
data_spred = pd.read_csv(import_path + _spred)
data_pinch = pd.read_csv(import_path + _pinch)
data_thump = pd.read_csv(import_path + _thump)

data_fist_raw = pd.read_csv('Fist_data_me.txt', delimiter=';', names=colum_names)
sums = data_fist_raw.iloc[:, :6].sum(axis=1)/6


data_total = pd.concat([data_fist, data_spred, data_pinch, data_thump], ignore_index=True)
#Feature extraction
#emg_features, features_names, labels_modified = FE_FCNs.features_estimation(data_fist.iloc[:, 0], frame, step)
#emg_features.to_csv('EMG_features.csv', index=False)

emg_fist = data_fist.iloc[:, 0]
# EMG Feature Extraction
emg_features, features_names = FE_FCNs.features_estimation(sums, channel_name,
                                                   sfreq, frame, step)
emg_features = emg_features.transpose()