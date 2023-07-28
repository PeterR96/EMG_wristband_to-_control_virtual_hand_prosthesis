3# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:08:18 2023

@author: Peter Rott
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import EMG_Filter_FCNs as FCNs
import Feature_extraction_FCNs as FE_FCNs
import Data_Labeling_FCNs as LE_FCNs

channel_name = 'Right Masseter'
frame = 250
step = 200
sfreq = 1000
low_pass = 5
high_band = 20
low_band = 450
notch_freq = 50
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6']

#Load raw data and sum it up
raw_data = pd.read_csv('Fist_data_test_me.txt', delimiter=';', names=colum_names)
sums = raw_data.iloc[:, :6].sum(axis=1)/6
modified_sums = sums.drop(1000)

#Signal Processing
time = FCNs.time_calc(modified_sums) 
raw_data_plot = FCNs.raw_data(modified_sums,time)
emg_without_offset = FCNs.remove_mean (modified_sums,time)
emg_filterd, emg_enveloped = FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)

#Data labeling
threshold = 0.03
label_name = "Fist"
labels = LE_FCNs.label_emg_data(emg_enveloped,threshold)

# Assign labels to emg_data array
labeled_emg_data = list(zip(emg_filterd, labels))

# Extract sample values and labels into separate lists
sample_values = [x[0] for x in labeled_emg_data]
labels = [x[1] for x in labeled_emg_data]

# Plotting scatter plot
plt.scatter(time, sample_values, c=labels, cmap='viridis')

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Sample Values')
plt.title('Sample Values with Labels')

labels = np.array(labels)
#Feature extraction
emg_features, features_names, labels_modified = FE_FCNs.features_estimation(emg_filterd,labels, label_name, channel_name,
                                                   sfreq, frame, step)
emg_features.to_csv('EMG_features.csv', index=False)