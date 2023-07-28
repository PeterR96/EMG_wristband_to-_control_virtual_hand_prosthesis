# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:23:52 2023

@author: Peter Rott
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.preprocessing import StandardScaler

def time_calc(raw_data):
    """
    raw_data: DataFrame with EMG signals
    """
    time = []
    num_rows = len(raw_data.iloc[:, 0])  # Get the length of the first column

    for i in range(num_rows):
        t = i / 1000
        time.append(t)

    return time

def plot_data(data, time, title, threshold,):
    """
    data: DataFrame with EMG signals (each column represents a signal)
    time: Time data
    """
    num_signals = len(data.columns)
    num_columns = data.shape[1]
    
    if num_columns > 1:
        fig, axes = plt.subplots(num_signals, 1, figsize=(8, 2 * num_signals), dpi=300, sharex=True)
        fig.suptitle(title)
    
        for i, column in enumerate(data.columns):
            ax = axes[i]
            ax.plot(time, data[column])
            ax.set_ylabel(f'EMG {i+1}')
                
        axes[-1].set_xlabel('Time [sec]')
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
    
        #plt.savefig(f'{title}.png')
        plt.show()
        
    else:
        fig = plt.figure(dpi=300)
        plt.plot(time,data[0])
        plt.locator_params(axis='x', nbins=10)
        plt.locator_params(axis='y', nbins=10)
        plt.title (title)
        plt.xlabel('Time in [sec]')
        plt.ylabel('Volt in [V]')
        plt.axhline(threshold, color='r', linestyle='--')

# def remove_mean(raw_data, time):
#     """
#     raw_data: DataFrame with EMG signals (each column represents a signal)
#     time: Time data
#     """
#     emg_correctmean = raw_data.copy()
#     num_columns = emg_correctmean.shape[1]
    
#     if num_columns > 1:  
#         for column in raw_data.columns:
#             emg_correctmean[column] -= np.mean(raw_data[column])  # Remove mean for each signal
def remove_mean(raw_data, time, window_size):
    num_samples = raw_data.shape[0]
    num_columns = raw_data.shape[1]
    num_windows = num_samples // window_size
    emg_correctmean = raw_data.copy()

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = (window_idx + 1) * window_size

        for column_idx in range(num_columns):
            window_data = emg_correctmean.iloc[start_idx:end_idx, column_idx]
            mean = np.mean(window_data)
            emg_correctmean.iloc[start_idx:end_idx, column_idx] -= mean

    return emg_correctmean



def filteremg(time, emg_data, low_pass, sfreq, high_band, low_band, notch_freq):
    """
    time: Time data
    emg_data: DataFrame with EMG signals (each column represents a signal)
    high_band: high-pass cutoff frequency
    low_band: low-pass cutoff frequency
    sfreq: sampling frequency
    notch_freq: notch filter frequency
    """

    num_samples = emg_data.shape[0]
    num_columns = emg_data.shape[1]
    window_size = 250
    num_windows = num_samples // window_size

    # Create empty arrays to store rectified and enveloped signals
    emg_rectified = np.empty(emg_data.shape)
    emg_envelope = np.empty(emg_data.shape)
    emg_filtered = np.empty(emg_data.shape)

    # Normalize cut-off frequencies to sampling frequency
    high = high_band / (sfreq / 2)
    low = low_band / (sfreq / 2)
    b1, a1 = signal.butter(4, [high, low], btype='bandpass')

    # Create notch filter
    Q = 30
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, sfreq)
    b1_notch, a1_notch = signal.iirnotch(notch_freq, Q, sfreq)

    for window_idx in range(num_windows):
        start_idx = window_idx * window_size
        end_idx = (window_idx + 1) * window_size

        # Process each EMG signal separately within the window
        for column_idx in range(num_columns):
            signal_data = emg_data.iloc[start_idx:end_idx, column_idx].values

            # Filter EMG
            filtered_signal = signal.filtfilt(b1, a1, signal_data)
            filtered_signal = signal.lfilter(b1_notch, a1_notch, filtered_signal)

            # Rectify
            rectified_signal = np.abs(filtered_signal)

            # Create low-pass filter and apply to rectified signal to get EMG envelope
            low_pass_normalized = low_pass / (sfreq / 2)
            b2, a2 = signal.butter(4, low_pass_normalized, btype='lowpass')
            envelope_signal = signal.filtfilt(b2, a2, rectified_signal)

            # Store the filtered, rectified, and enveloped signals in the respective arrays
            emg_rectified[start_idx:end_idx, column_idx] = rectified_signal
            emg_envelope[start_idx:end_idx, column_idx] = envelope_signal
            emg_filtered[start_idx:end_idx, column_idx] = filtered_signal
        
    return emg_filtered


def extract_emg_features(emg_data, window_size, overlap):
    feature_names = ['MAV', 'ZC', 'SSC', 'WL', 'VAR', 'IEMG', 'RMS']
    num_sensors = emg_data.shape[1]
    features = []
    window_start = 0
    window_end = window_size

    while window_end <= len(emg_data):
        emg_window = emg_data.iloc[window_start:window_end, :]

        sensor_features = []
        for sensor in range(num_sensors):
            mav = np.mean(np.abs(emg_window.iloc[:, sensor]))
            zc = np.sum(np.diff(np.sign(emg_window.iloc[:, sensor])) != 0)
            ssc = np.sum(np.diff(np.sign(np.diff(emg_window.iloc[:, sensor]))) != 0)
            wl = np.sum(np.abs(np.diff(emg_window.iloc[:, sensor])))
            variance = np.var(emg_window.iloc[:, sensor])
            integrated_emg = np.sum(np.abs(emg_window.iloc[:, sensor]))
            rms = np.sqrt(np.mean(emg_window.iloc[:, sensor]**2))

            sensor_features.extend([mav, zc, ssc, wl, variance, integrated_emg, rms])

        features.append(sensor_features)

        window_start += overlap
        window_end += overlap
    if num_sensors == 1:
        columns = [f"{feature}" for sensor in range(num_sensors) for feature in feature_names ]
        features_df = pd.DataFrame(features, columns=columns)    
    else:
        columns = [f"{feature}_{sensor+1}" for sensor in range(num_sensors) for feature in feature_names ]
        features_df = pd.DataFrame(features, columns=columns)
        
    # # Modify column names to include sensor number
    # sensor_numbers = list(range(1, num_sensors+1))
    # column_names_modified = [f"{feature}{sensor}" for feature in feature_names for sensor in sensor_numbers]
    # features_df.columns = column_names_modified   
    return features_df


def calculate_wave_length(emg_data, window_size, overlap):
    feature_name = ['WL_sum']
    WL_sum = []
    window_start = 0
    window_end = window_size

    while window_end <= len(emg_data):
        emg_window = emg_data.iloc[window_start:window_end,0]

        wl = np.sum(np.abs(np.diff(emg_window)))

        WL_sum.append([wl])

        window_start += overlap
        window_end += overlap

    WL_sum_df = pd.DataFrame(WL_sum, columns=feature_name)
    return WL_sum_df
        

def plot_features (emg_features,raw_data,time,title):
    # Plotting the features
    fig, axs = plt.subplots(nrows=len(emg_features.columns)+1, ncols=1, figsize=(8, 12))
    fig.suptitle(title)
    
    if raw_data is not None:
        axs[0].plot(time,raw_data)
        axs[0].set_ylabel('Volt in [mV]')
        axs[0].set_xlabel('Time')
        axs[0].set_title('Filtered Data')
        
    
    for i, col in enumerate(emg_features.columns, start=1 if raw_data is not None else 0):
        axs[i].plot(emg_features[col])
        axs[i].set_ylabel('Amplitude')
        axs[i].set_xlabel('Window')
        axs[i].set_title(col )  # Modified plot title ' Feature'
            
    plt.tight_layout()
    plt.show()

# def plot_features(emg_features, raw_data, time, title):
#     if raw_data is not None:
#         fig1, ax1 = plt.subplots(figsize=(8, 6))
#         ax1.plot(time, raw_data)
#         ax1.set_ylabel('Volt in [mV]')
#         ax1.set_xlabel('Time')
#         ax1.set_title('Test EMG data sequence')

#     for col in emg_features.columns:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         fig.suptitle(title + '')
#         ax.plot(emg_features[col])
#         ax.set_ylabel('Amplitude')
#         ax.set_xlabel('Window')
#         ax.set_title(col)

#     plt.tight_layout()
#     plt.show()


def plot_WL(emg_features, threshold, title):
    """
    emg_features: DataFrame with EMG features
    threshold: DataFrame with threshold values
    labels: List of label values
    title: Title for the plot
    """
    labels = emg_features['Label'] 
    
    # Plotting the features
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(emg_features['WL_sum'])
    ax.set_ylabel('WL_sum')
    ax.set_xlabel('Window')
    ax.set_title(title)
    plt.locator_params(axis='y', nbins=14)
    
    # Plotting the thresholds
    ax.axhline(threshold.loc[0, 0], color='r', linestyle='--', label='Thr 1')
    ax.axhline(threshold.loc[0, 1], color='g', linestyle='--', label='Thr 2')
    ax.legend()

    previous_label = labels[0]
    for j, label in enumerate(labels[1:], start=1):
        if label != previous_label:
            ax.axvline(j, color='b', linestyle='-', alpha=0.5)
        previous_label = label
    
    plt.tight_layout()
    plt.show()

def label_emg_features(emg_features, label_name, x):
    wl_threshold = pd.DataFrame()

    if label_name == 'Fist':
        label = 1
    elif label_name == 'Pinch':
        label = 2
    elif label_name == 'Thump':
        label = 31
    elif label_name == 'Spred':
        label = 4
    elif label_name == 'Wavein':
        label = 5
    elif label_name == 'Waveout':
        label = 6
    elif label_name == 'test_data':
        label = 7
    else:
        label = 0

    wl_threshold.loc[0, 0] = emg_features['WL_sum'].head(8).max() * 1.8 - x
    wl_threshold.loc[0, 1] = emg_features['WL_sum'].head(8).max() * 1.65 - x
    
    #labels = emg_features['WL_1'].apply(lambda x: label if x > wl_threshold else 0)
    #emg_features['Label'] = labels
    #  
    line_values = pd.DataFrame(columns=['X_Value'])      
    labels = []
    active = False
    for i,value in enumerate(emg_features['WL_sum']):
        if not active and value > wl_threshold.loc[0, 0]:
            active = True
            labels.append(label)
            line_values = line_values.append({'X_Value': i}, ignore_index=True)  # Append the x-value to line_values DataFrame
        elif active and value < wl_threshold.loc[0, 1]:
            active = False
            labels.append(0)
            line_values = line_values.append({'X_Value': i}, ignore_index=True)  # Append the x-value to line_values DataFrame
        else:
            labels.append(labels[-1] if labels else 0)

    emg_features['Label'] = labels

    return emg_features, label, wl_threshold, line_values

def replace_zero_labels(emg_features, label):
    label_col = emg_features['Label']
    label_changes = np.diff(label_col)
    zero_indices = np.where(label_changes == -label)[0]
    for idx in zero_indices:
        if label_col[idx + 2] == label:
            label_col[idx + 1] = label
    emg_features['Label'] = label_col
    return emg_features

def replace_labels_zero(emg_features, label):
    label_col = emg_features['Label']
    label_changes = np.diff(label_col)
    zero_indices = np.where(label_changes == -label)[0]
    for idx in zero_indices:
        if idx - 2 >= 0 and idx + 2 < len(label_col) and label_col[idx - 2] == 0 and label_col[idx + 2] == 0:
            label_col[idx] = 0
    emg_features['Label'] = label_col
    return emg_features


def replace_labels(label_vector, label_list):
    final_label_vector = label_vector.copy()
    label_idx = 0
    prev_label = 0
    
    for i, label in enumerate(label_vector):
        if label == 0 and prev_label!=0:
                label_idx = (label_idx + 1) % len(label_list)
        if label == 7:
            final_label_vector[i] = label_list[label_idx]
            
        prev_label = label
    return final_label_vector

def normalize(emg_features_final):
    scaler = StandardScaler()
    norm_features = []
    
    for features in emg_features_final:
        normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()
        norm_features.append(normalized)
    
    norm_features = np.array(norm_features)
    return norm_features

def plot_labels(labels, raw_data, time, title, additional_labels):
    numbers = []
    for i in range(len(labels)):
        numbers.append(i + 1)
    plt.title(title)
    plt.xlabel('Time in [sec]')
    plt.ylabel('Volt in [mV]')
    plt.plot(time, raw_data,label = 'Filtered EMG Signal')  # against 1st x, 1st y
    plt.axis([0, len(time) / 1000, -1700, 1700])
    
    plt.twinx()
    plt.ylabel('Labels')
    plt.twiny()
    plt.xlabel('Window in [250ms/Window]')
    plt.plot(numbers, labels, 'b', label = 'Original Labels')  # against 2nd x, 2nd y
    
    # Plot the additional label vector
    plt.plot(numbers, additional_labels,color=(1.0, 0.5, 0.5),label = 'Predicted Labels')  # against 2nd x, additional y
    
    plt.axis([0, len(labels), -4.5, 4.5])
    plt.tight_layout()
    plt.legend()
    plt.show()
