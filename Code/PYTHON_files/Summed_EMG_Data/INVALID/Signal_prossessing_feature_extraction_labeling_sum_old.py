# -*- coding: utf-8 -*-
"""
Created on Sat Jul  1 20:11:59 2023

@author: Peter Rott
"""

import numpy as np
from scipy.signal import find_peaks
import pandas as pd
import matplotlib.pyplot as plt
import EMG_FCNs as FCNs

sfreq = 1000                                                        #sample Frequency
low_pass = 3                                                        #cut off frequency for enveloped signal
high_band = 20                                                      #cut off frequency for raw signal
low_band = 450                                                      #cut off frequency for raw signal
notch_freq = 50                                                     #Notch frequency 50/60Hz depens on grid
window_size=250                                                     #Datasegment for Feature extraction
overlap=200                                                         #Overlap of Segments: Window_size - overlap
wl_threshold= 200                                                #Threshold for labeling
label_name = 'Thumpup'
import_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Recorded_data\2'
export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data\2'
filename = '\Thump_data_me.txt'

# Import EMG data
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 
data_fist_raw = pd.read_csv(import_path+filename, delimiter=';', names=colum_names)

emg_data = data_fist_raw.iloc[:, :6].sum(axis=1)/6                                                              #build an avg of all sensor data

#Signal processing
time = FCNs.time_calc(data_fist_raw)                                                                            #calculates the recorded time with the sensor sampels
emg_data_df = emg_data.to_frame()                                                                               #transformation to data frame
emg_sum_without_offset = FCNs.remove_mean (emg_data_df,time)                                                    #removing the offset of the EMG signal
filtered_emg = FCNs.filteremg(time, emg_sum_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter

# Extract EMG features and label the data
emg_features = FCNs.extract_emg_features(filtered_emg, window_size, overlap)                                    #Feature extraction
emg_features_labeled, label = FCNs.label_emg_features(emg_features, wl_threshold, label_name)                          #Feature labeling
emg_features_replaced  = FCNs.replace_zero_labels(emg_features_labeled,label)                                         #replace wrong detected labels
emg_features_final = FCNs.replace_labels_zero(emg_features_replaced,label)                                         #checks for a label between no labels
FCNs.plot_features (emg_features_replaced)                                                                      #plot features and labels

#Data export
emg_features_final.to_csv(export_path+'\labeled_Feature_Data'+label_name+'.csv', index=False)                 #export labeld features as csv file

