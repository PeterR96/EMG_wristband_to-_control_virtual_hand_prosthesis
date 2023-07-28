import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import EMG_FCNs as FCNs

sfreq = 1000
low_pass = 5
high_band = 20
low_band = 450
notch_freq = 50
colum_names = ['emg_1','t'] #'emg_2','emg_3','emg_4','emg_5','emg_6',

x = 1000

raw_data = pd.read_csv('EMG_Processing_Test_pyton.txt', delimiter='\t', names=colum_names)

raw_data = raw_data.drop('t',axis=1)
baseline = raw_data.iloc[:2000].mean()

raw_data = (raw_data - baseline)
raw_data = raw_data.multiply(x)

time = FCNs.time_calc(raw_data.emg_1) 
raw_data_plot = FCNs.raw_data(raw_data.emg_1,time)
emg_without_offset = FCNs.remove_mean (raw_data.emg_1, time)
FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)

# show what different low pass filter cut-offs do
#for i in [3, 10, 40]:
   # FCNs.filteremg(time, emg_filtered, low_pass=i)
    
