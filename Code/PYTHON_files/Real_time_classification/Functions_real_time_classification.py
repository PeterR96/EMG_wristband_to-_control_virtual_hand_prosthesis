import numpy as np
import scipy as sp
from scipy import signal
from sklearn.preprocessing import StandardScaler

def process_data(data, model):
    num_sensors = data.shape[1]
    features = []
    feature_names = ['MAV', 'ZC', 'SSC', 'WL', 'VAR', 'IEMG', 'RMS']
    emg_correctmean = data.copy()
    high_band = 20  # cut-off frequency for raw signal
    low_band = 450  # cut-off frequency for raw signal
    notch_freq = 50
    Q = 30
    sfreq = 1000
    emg_filtered = np.empty_like(data)
    
    # Normalize cut-off frequencies to sampling frequency
    high = high_band / (sfreq / 2)
    low = low_band / (sfreq / 2)
    b1, a1 = sp.signal.butter(4, [high, low], btype='bandpass')

    # Create notch filter
    Q = 30
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, sfreq)
    b1_notch, a1_notch = signal.iirnotch(notch_freq, Q, sfreq)
    
    for column in range(num_sensors):
         
        emg_correctmean[ :, column] -= np.mean(data[ :, column])
        
        # Filter EMG
        filtered_signal = sp.signal.filtfilt(b1, a1, emg_correctmean[ :, column])
        filtered_signal = signal.lfilter(b1_notch, a1_notch, filtered_signal)
        emg_filtered[:, column] = filtered_signal
        
        
        sensor_features = []
        for sensor in range(num_sensors):
            mav = np.mean(np.abs(emg_filtered[:, sensor]))
            zc = np.sum(np.diff(np.sign(emg_filtered[:, sensor])) != 0)
            ssc = np.sum(np.diff(np.sign(np.diff(emg_filtered[:, sensor]))) != 0)
            wl = np.sum(np.abs(np.diff(emg_filtered[:, sensor])))
            variance = np.var(emg_filtered[:, sensor])
            integrated_emg = np.sum(np.abs(emg_filtered[:, sensor]))
            rms = np.sqrt(np.mean(emg_filtered[:, sensor] ** 2))
            
            sensor_features.extend([mav, zc, ssc, wl, variance, integrated_emg, rms])
    
    features.append(sensor_features)   
    features_array = np.array(features)
    
    return emg_filtered,features_array

def normalize(emg_features_final):
    scaler = StandardScaler()
    norm_features = []
    
    for features in emg_features_final:
        normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()
        norm_features.append(normalized)
    
    norm_features = np.array(norm_features)
    return norm_features

def label(y_pred):
    
    if y_pred == 1:
        label = 'Fist'
    elif y_pred == 2:
        label = 'Pinch'
    elif y_pred == 3:
        label = 'Thumb'
    elif y_pred == 4:
        label = 'Spred'
    else:
        label ='Rest'
    return label