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
    #fig.set_size_inches(w=11,h=7)
    fig.savefig(fig_name, dpi= 300)
    return 
def remove_mean (emg_data, time):
    # process EMG signal: remove mean
    emg_correctmean = emg_data - np.mean(emg_data)
    
    # plot comparison of emg_data with offset vs mean-corrected values
    fig = plt.figure(dpi=300)
    plt.title('EMG Signal without offset')
    plt.plot(time, emg_correctmean)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    #plt.ylim(-1.5, 1.5)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')
    
    fig.tight_layout()
    fig_name = 'EMG_without_offset.png'
    fig.savefig(fig_name, dpi=300)
    return emg_correctmean
    
    
def emg_filter (emg_correctmean,time, high_band, low_band,notch_freq,sfreq):
    # create bandpass filter for EMG
    high = high_band/(sfreq/2)
    low = low_band/(sfreq/2)
    b, a = sp.signal.butter(4, [high,low], btype='bandpass')
    
    #create notch filter
    Q=30
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, sfreq)
    
    # process EMG signal: filter EMG
    emg_filtered = sp.signal.filtfilt(b, a, emg_correctmean)
    #emg_filtered = signal.lfilter(b_notch, a_notch, emg_filtered)
                                  
    
    return emg_filtered

def emg_rectified (emg_filtered, time):
    # process EMG signal: rectify
    emg_rectified = abs(emg_filtered)
    
    
    return emg_rectified

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
    #X_mag_plot[0] = X_mag_plot[0] / 2
    
   # PLOTS------------------------------------------------------------------------------------------------- 
    
    # plot comparison of unfiltered vs filtered mean-corrected EMG
    fig = plt.figure(dpi=300)
    plt.title('Bandpass filtered EMG Signal '+ str(int(high_band)) + '-' + str(int(low_band)) +' Hz')
    plt.plot(time, emg_filtered)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [V]')   
    fig.tight_layout()
    fig_name = 'EMG_filtered.png'
    fig.savefig(fig_name, dpi=300)
    plt.show()
    
    # plot comparison of unrectified vs rectified EMG
    fig = plt.figure(dpi= 300)
    plt.title('Rectified EMG Signal')
    plt.plot(time, emg_rectified)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.ylabel('Volt in [V]')
    plt.xlabel('Time [sec]')
    fig.tight_layout()
    fig_name = 'Rectified_EMG_data.png'
    fig.savefig(fig_name, dpi= 300)
    plt.show()
    
    # plot envelope
    fig = plt.figure(dpi= 300)
    plt.title('Filtered, rectified ' + '\n' + 'EMG envelope: ' + str(int(low_pass*sfreq)) + ' Hz')
    plt.plot(time, emg_envelope)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=10)
    plt.ylabel('Volt in [V]')
    plt.xlabel('Time [sec]')
    fig.tight_layout()
    fig_name = 'EMG_envelope' + str(int(low_pass*sfreq)) + '.png'
    fig.savefig(fig_name, dpi= 300)
    plt.show()
    
    # plot frequency spectrum
    fig = plt.figure(dpi= 300)
    plt.title('Filtered frequency spectrum of three muscle contractions')
    plt.plot(f_plot, X_mag_plot)
    plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Frequency in [Hz]')
    plt.ylabel('Counts')
    fig.tight_layout()
    fig_name = 'Frequency Spectrum_filter.png'
    fig.savefig(fig_name, dpi= 300)
    plt.show()
    
    return emg_filtered,  emg_envelope

def get_power(emg_data, sfreq):
    N=emg_data.size
    tsteps = 1/sfreq
    
    t = np.linspace(0, (N-1)*tsteps, N)
    fsteps = sfreq/N
    f = np.linspace(0,(N-1)*fsteps, N)
    
    X = np.fft.fft(emg_data)
    X_mag = np.abs(X) 
    
    f_plot = f [0:int(N/2 +1)]
    X_mag_plot = 2* X_mag[0:int(N/2+1)]
    #X_mag_plot[0] = X_mag_plot[0] / 2
    
    fig = plt.figure(dpi= 300)
    plt.title('Frequency spectrum of three muscle contractions')
    plt.plot(f_plot, X_mag_plot)
    plt.locator_params(axis='x', nbins=20)
    plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Frequency in [Hz]')
    plt.ylabel('Counts')
    fig.tight_layout()
    fig_name = 'Frequency Spectrum_no_filter.png'
    fig.savefig(fig_name, dpi= 300)
    plt.show()
    
    
    fig_1 =plt.figure(dpi=300)
    plt.title('Frequency spectrum Histogram for contracted muscle')
    plt.hist(X_mag_plot, bins=100)
    plt.locator_params(axis='x', nbins=10)
    plt.locator_params(axis='y', nbins=20)
    plt.xlabel('Frequencies in [Hz]')
    plt.ylabel('Counts')
    plt.show()
    
    return X_mag_plot, f_plot

def get_power_1(data, sfreq):
    sig_fft = fftpack.fft(data)
    
    # And the power (sig_fft is of complex dtype)
    power = np.abs(sig_fft)
    
    # The corresponding frequencies
    sample_freq1 = fftpack.fftfreq(data.size, d=1/sfreq)
    frequencies = sample_freq1[sample_freq1 > 0]
    power = power[sample_freq1 > 0]
    return power, frequencies