# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:55:50 2023

@author: Peter Rott
"""

import EMG_FCNs as FCNs
import pandas as pd
import numpy as np
import json
#############################################################################################################################
sfreq = 1000                                                        #sample Frequency
low_pass = 3                                                        #cut off frequency for enveloped signal
high_band = 20                                                      #cut off frequency for raw signal
low_band = 450                                                      #cut off frequency for raw signal
notch_freq = 50                                                     #Notch frequency 50/60Hz depens on grid
window_size=250                                                     #Datasegment for Feature extraction
overlap=250
x = 0                                                             #Overlap of Segments: Window_size - overlap
label_name = 'test_data'
export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\HoloLens_lablled_data'
#test_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\HoloLens_lablled_data'
filename = './test_01.txt'

label_path = r'./HoloLens_lablled_data'
labelfile = './EMGTimeLabel_02.txt'

with open (labelfile, 'r') as file:
    label_data = json.load(file)
    # Extract the GestureID values from the label_data
    
gesture_ids = [item["GestureID"] for item in label_data]
# Create a DataFrame from the GestureID values
labels = pd.DataFrame({"GestureID": gesture_ids})
#Import EMG data
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 


raw_data = pd.read_csv(filename, delimiter=';', names=colum_names)
emg_sum_data = raw_data.iloc[:, :].sum(axis=1)/6                                                             #build an avg of all sensor data
#############################################################################################################################                                                           
    
#Signal processing for each sensor
time = FCNs.time_calc(raw_data)                                                                             #calculates the recorded time with the sensor sampels
emg_without_offset = FCNs.remove_mean(raw_data,time)                                           #removing the offset of the EMG signal
filtered_emg = FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter
#############################################################################################################################
    
#Signal processing for the avg sesnor values
emg_sum_data_df = emg_sum_data.to_frame()                                                                       #transformation to data frame
emg_sum_without_offset = FCNs.remove_mean (emg_sum_data_df,time)                                                #removing the offset of the EMG signal
filtered_emg_sum = FCNs.filteremg(time, emg_sum_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter
#############################################################################################################################
   
#Feature extraction
emg_features = FCNs.extract_emg_features(filtered_emg, window_size, overlap)
num_rows_to_remove = emg_features.shape[0] - labels.shape[0]

filtered_emg = filtered_emg.iloc[:num_rows_to_remove*4]
filtered_emg_sum = filtered_emg_sum[:num_rows_to_remove*4]
time = time[:num_rows_to_remove*4]
time = np.array(time)
# Remove the last num_rows_to_remove rows from the feature DataFrame
emg_features = emg_features.iloc[:num_rows_to_remove]

                                    #Feature extraction
emg_features_sum = FCNs.extract_emg_features(filtered_emg_sum, window_size, overlap)
WL_sum = FCNs.calculate_wave_length(filtered_emg_sum, window_size, overlap)
emg_features_WL_sum = pd.concat([emg_features, WL_sum], axis=1)

emg_featuers_labeld = pd.concat([emg_features, labels], axis=1)
plot_data = emg_featuers_labeld [["GestureID"]]
#############################################################################################################################
    
FCNs.plot_WL(emg_features_WL_sum, 'Wavelentgh feature of the gesture '+label_name)


#FCNs.plot_features (plot_data,filtered_emg.iloc[:, 1],time,'Labels predicted by the HoloLens')
FCNs.plot_labels(labels['GestureID'],filtered_emg_sum.iloc[:,0],time,'Labels predicted by the HoloLens')

#############################################################################################################################

#Data export
#emg_features_final.to_csv(export_path+'\labeled_Feature_Data_'+label_name+'_02.csv', index=False)                 #export labeld features as csv file
#emg_features_sum_final.to_csv(export_path_sum + '\labeled_Feature_Data_' + label_name + '.csv', index=False)  # export labeled features as CSV file