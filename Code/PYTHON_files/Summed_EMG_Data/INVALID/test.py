
import pandas as pd
import EMG_FCNs as FCNs

sfreq = 1000                                                        #sample Frequency
low_pass = 3                                                        #cut off frequency for enveloped signal
high_band = 20                                                      #cut off frequency for raw signal
low_band = 450                                                      #cut off frequency for raw signal
notch_freq = 50                                                     #Notch frequency 50/60Hz depens on grid
window_size=250                                                     #Datasegment for Feature extraction
overlap=200                                                         #Overlap of Segments: Window_size - overlap
x = 0     

label_names = ['Fist', 'Pinch', 'Thump', 'Spred']
participant_nos = ['3']#, '2', '3', '4', '5', '6', '7', '8', '9', '10']

for participant_no in participant_nos:
    import_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Recorded_data\%s' % participant_no
    export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data\%s' % participant_no

    for label_name in label_names:
        filename = '\%s_data_0%s.txt' % (label_name, participant_no)
        
        if label_name == 'Fist':
            x=0
        elif label_name == 'Pinch':
            x=50
        elif label_name == 'Thump':
            x=0
        elif label_name == 'Spred':
            x=0
        else:
            x=0
            
        # Import EMG data
        column_names = ['emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6']
        data_first_raw = pd.read_csv(import_path + filename, delimiter=';', names=column_names)
        emg_data = data_first_raw.iloc[:, :6].sum(axis=1) / 6  # build an avg of all sensor data

        # Signal processing
        time = FCNs.time_calc(data_first_raw)  # calculates the recorded time with the sensor samples
        emg_data_df = emg_data.to_frame()  # transformation to DataFrame
        emg_sum_without_offset = FCNs.remove_mean(emg_data_df, time, label_name)  # removing the offset of the EMG signal
        filtered_emg = FCNs.filteremg(time, emg_sum_without_offset, low_pass, sfreq, high_band, low_band,
                                      notch_freq)  # filters the EMG signal with bandpass and notch filter

        # Extract EMG features and label the data
        emg_features = FCNs.extract_emg_features(filtered_emg, window_size, overlap)  # Feature extraction
        # emg_features = emg_features.iloc[20:]
        emg_features_labeled, label, wl_threshold = FCNs.label_emg_features(emg_features, label_name,
                                                                             x)  # Feature labeling
        emg_features_replaced = FCNs.replace_zero_labels(emg_features_labeled, label)  # replace wrong detected labels
        for i in range(2):
            emg_features_final = FCNs.replace_labels_zero(emg_features_replaced,
                                                          label)  # checks for a label between no labels

        # Plot features and activation regions
        FCNs.plot_features(emg_features_final, emg_data_df, time, wl_threshold, 'EMG Features')  # plot features and labels
        FCNs.plot_WL(emg_features_final, wl_threshold,
                     'Wavelength feature to detect the beginning and end of the gesture')

        # Data export
        emg_features_final.to_csv(export_path + '\labeled_Feature_Data_' + label_name + '.csv', index=False)  # export labeled features as CSV file
