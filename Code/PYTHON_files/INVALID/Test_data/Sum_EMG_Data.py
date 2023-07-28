# -*- coding: utf-8 -*-
"""
Created on Wed May 31 18:18:32 2023

@author: Peter Rott
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import EMG_Filter_FCNs as FCNs
import Feature_extraction_FCNs as FE_FCNs

channel_name = 'Right Masseter'
frame = 250
step = 125
sfreq = 1000
low_pass = 5
high_band = 20
low_band = 450
notch_freq = 50
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6', 't']

raw_data = pd.read_csv('EMG_Fist_data_01.txt', delimiter=';', names=colum_names)

# Extract the first six numbers from each row and calculate the sums
#sums = raw_data.iloc[:, :6].sum(axis=1)
sums = raw_data.iloc[:, [0, 1, 3, 4, 5]].sum(axis=1)

#Signal Processing
time = FCNs.time_calc(sums) 
raw_data_plot = FCNs.raw_data(sums,time)
emg_without_offset = FCNs.remove_mean (sums, time)
emg_filterd = FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)

#Feature extraction
emg_features, features_names = FE_FCNs.features_estimation(emg_filterd, channel_name,
                                                   sfreq, frame, step)