# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:55:50 2023

@author: Peter Rott
"""
import EMG_FCNs_250 as FCNs
import pandas as pd
#############################################################################################################################
sfreq = 1000                                                        #sample Frequency
low_pass = 3                                                        #cut off frequency for enveloped signal
high_band = 20                                                      #cut off frequency for raw signal
low_band = 450                                                      #cut off frequency for raw signal
notch_freq = 50                                                     #Notch frequency 50/60Hz depens on grid
window_size=250                                                    #Datasegment for Feature extraction
overlap=250
x = 0
run = 2                                            #Overlap of Segments: Window_size - overlap
label_name = 'test_data'
export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data\Test'
export_path_sum = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data\Test'
# Fist = 1 ; Pinch = 2; Thump = 3; Spred = 4
#label_list = pd.DataFrame({'Label': [1, 4, 3, 1, 4, 2, 2, 3]})

#label_list = [3, 2, 1, 1, 4, 1, 3, 4]
label_list = [3, 3, 1, 1, 1, 2, 1, 1]
#label_list = [2, 3, 2, 1, 1, 4, 3, 2]
#label_list = [1, 4, 3, 1, 4, 2, 2, 3]
#label_list = [3, 1, 1, 2, 4, 1, 2, 2]
#label_list = [2, 4, 4, 1, 1, 4, 4, 1]
#label_list = [3, 3, 3, 4, 4, 4, 4, 2]
#label_list = [4, 3, 4, 1, 3, 3, 2, 1]
#label_list = [2, 4, 3, 3, 4, 4, 4, 2]
#label_list = [4, 3, 1, 3, 1, 3, 1, 1]




test_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Recorded_data\Test_sequences'
filename = './test_sequence_0%s.txt'%run



#Import EMG data
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 
raw_data = pd.read_csv(test_path+filename, delimiter=';', names=colum_names)
emg_sum_data = raw_data.iloc[:, :6].sum(axis=1)/6                                                              #build an avg of all sensor data
#############################################################################################################################                                                           
    
#Signal processing for each sensor
time = FCNs.time_calc(raw_data)                                                                             #calculates the recorded time with the sensor sampels
emg_without_offset = FCNs.remove_mean(raw_data,time,250)                                           #removing the offset of the EMG signal
filtered_emg = FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter
filtered_emg = pd.DataFrame(filtered_emg)
#############################################################################################################################
    
#Signal processing for the avg sesnor values
emg_sum_data_df = emg_sum_data.to_frame()                                                                       #transformation to data frame
emg_sum_without_offset = FCNs.remove_mean (emg_sum_data_df,time,250)                                                #removing the offset of the EMG signal
filtered_emg_sum = FCNs.filteremg(time, emg_sum_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter
filtered_emg_sum =pd.DataFrame(filtered_emg_sum)
#############################################################################################################################
   
#Feature extraction
emg_features = FCNs.extract_emg_features(filtered_emg, window_size, overlap)                                    #Feature extraction
emg_features_sum = FCNs.extract_emg_features(filtered_emg_sum, window_size, overlap)
WL_sum = FCNs.calculate_wave_length(filtered_emg_sum, window_size, overlap)
emg_features_WL_sum = pd.concat([emg_features, WL_sum], axis=1)

#############################################################################################################################
    
#Data labelling
emg_features_labeled, label, wl_threshold, line_values = FCNs.label_emg_features(emg_features_WL_sum, label_name, x)                   #Feature labeling
emg_features_replaced  = FCNs.replace_zero_labels(emg_features_labeled, label) 
for i in range(2):
    emg_features_final = FCNs.replace_labels_zero(emg_features_replaced,label)  #replace wrong detected labels

emg_features_with_correct_label = FCNs.replace_labels(emg_features_replaced['Label'], label_list)
emg_features_replaced['Label'] = emg_features_with_correct_label

FCNs.plot_WL(emg_features_replaced, wl_threshold, 'Wavelentgh feature of the gesture '+label_name)
emg_features_final = emg_features_replaced.drop('WL_sum', axis=1)
emg_features_sum_final = emg_features_sum.assign(Label=emg_features_final['Label'])

plot_data = emg_features_sum_final [["Label"]]
FCNs.plot_features (plot_data,raw_data.iloc[:, 1],time,'')
#############################################################################################################################

#Data export
#emg_features_final.to_csv(export_path+'\data_250\labeled_Feature_Data_'+label_name+'0%s.csv'%run, index=False)                 #export labeld features as csv file
#emg_features_sum_final.to_csv(export_path_sum + '\labeled_Feature_Data_' + label_name + '.csv', index=False)  # export labeled features as CSV file

