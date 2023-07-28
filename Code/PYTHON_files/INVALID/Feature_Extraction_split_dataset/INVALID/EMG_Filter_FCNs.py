# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:23:52 2023

@author: Peter Rott
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import signal
from scipy import fftpack

def time_calc(raw_data):
    time = []
    for i in range(0, len(raw_data), 1):
        i = i/1000
        time.append(i)
        np.array(time)        
    return time    

def raw_data (raw_data,time):
    fig = plt.figure(dpi=300)
    plt.plot(time,raw_data)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.title ('Raw EMG Signal')
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')
    fig_name = 'EMG_raw_data_test.png'
    fig.savefig(fig_name, dpi= 300)  

def remove_mean (emg_data, time):
    # process EMG signal: remove mean
    emg_correctmean = emg_data - np.mean(emg_data)
    
    # plot comparison of emg_data with offset vs mean-corrected values
    fig = plt.figure(dpi=300)
    plt.title('EMG Signal without offset')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')  
    fig.tight_layout()
    fig_name = 'EMG_without_offset.png'
    plt.show()
    fig.savefig(fig_name, dpi=300)
    
    return emg_correctmean

def filteremg(time, emg_data, low_pass, sfreq, high_band, low_band, notch_freq):
    """
    time: Time data
    emg: EMG data
    high: high-pass cut off frequency
    low: low-pass cut off frequency
    sfreq: sampling frequency
    """
    # normalise cut-off frequencies to sampling frequency
    high = high_band/(sfreq/2)
    low = low_band/(sfreq/2)
    b1, a1 = sp.signal.butter(4, [high,low], btype='bandpass')
    
    #create notch filter
    Q=30
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, sfreq)
    b1_notch, a1_notch = signal.iirnotch(notch_freq, Q, sfreq)
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b1, a1, emg_data)    
    emg_filtered = signal.lfilter(b1_notch, a1_notch, emg_filtered)
    
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    # create lowpass filter and apply to rectified signal to get EMG envelope
    low_pass = low_pass/(sfreq/2)
    b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
    emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)
    
    # create frequency spectrum
    N=emg_filtered.size
    tsteps = 1/sfreq
    
    t = np.linspace(0, (N-1)*tsteps, N)
    fsteps = sfreq/N
    f = np.linspace(0,(N-1)*fsteps, N)
    
    X = np.fft.fft(emg_filtered)
    X_mag = np.abs(X) 
    
    f_plot = f [0:int(N/2 +1)]
    X_mag_plot = 2* X_mag[0:int(N/2+1)]
    
    # plot comparison of emg_data with offset vs mean-corrected values
    fig = plt.figure(dpi=300)
    plt.title('Filtered and rectified EMG Signal')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')
    fig.tight_layout()
    fig_name = 'Rectified_EMG.png'
    plt.show()
    fig.savefig(fig_name, dpi=300)

    # plot comparison of emg_data with offset vs mean-corrected values
    fig = plt.figure(dpi=300)
    plt.title('Filtered EMG Signal')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')    
    fig.tight_layout()
    fig_name = 'Filtered_EMG.png'
    plt.show()
    fig.savefig(fig_name, dpi=300)
    
    # plot comparison of emg_data with offset vs mean-corrected values
    fig = plt.figure(dpi=300)
    plt.title('Enveloped rectified EMG Signal')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')    
    fig.tight_layout()
    fig_name = 'Enveloped_EMG.png'
    plt.show()
    fig.savefig(fig_name, dpi=300)
    
    return emg_rectified, emg_envelope