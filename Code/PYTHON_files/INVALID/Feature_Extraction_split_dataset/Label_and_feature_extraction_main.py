# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:29:15 2023

@author: Peter Rott
"""

import pandas as pd
import Data_Labeling_FCNs as LE_FCNs
import Feature_extraction_FCNs as FE_FCNs

frame = 250
step = 200
sfreq = 1000
colum_name = ['EMG_data']
imoprt_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Load_and_procsess_Data\Exported_data_for_labeling'
export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Lableing_Feature_Extraction\Training_data'
filename = '\Summed_EMG_Enveleoped_Fist_01.csv'

# Read the CSV file
data = pd.read_csv(imoprt_path + filename,names=colum_name, skiprows=1)

#Data labeling
threshold = 0.013
label_name = "Fist"
data_labeled = LE_FCNs.label_data(data,'EMG_data',threshold, label_name)
labels =  data_labeled['Label']
data_ = data_labeled ['EMG_data']

#Feature extraction
emg_features, features_names, labels_modified = FE_FCNs.features_estimation(data_, labels, label_name, sfreq, frame, step)
emg_features.to_csv('EMG_features.csv', index=False)
        



