import EMG_FCNs_250 as FCNs
import pandas as pd
import numpy as np
#############################################################################################################################
sfreq = 1000                                                        #sample Frequency
low_pass = 3                                                        #cut off frequency for enveloped signal
high_band = 20                                                      #cut off frequency for raw signal
low_band = 450                                                      #cut off frequency for raw signal
notch_freq = 50                                                     #Notch frequency 50/60Hz depens on grid
window_size=250                                                    #Datasegment for Feature extraction
overlap=250
x = 0                                                #Overlap of Segments: Window_size - overlap
label_names = ['Fist', 'Pinch', 'Thump', 'Spred']
participant_nos = ['1']#, '2', '3', '4', '5', '6', '7', '8', '9', '10']
############################################################################################################################
for participant_no in participant_nos:
    import_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Recorded_data\%s' % participant_no
    export_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Individual_EMG_data\Labeld_Feature_Data\%s\data_250' % participant_no
    export_path_sum = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Summed_EMG_Data\Labeld_Feature_Data\%s\data_250' % participant_no
   
       
    for label_name in label_names:
        filename = '\%s_data_0%s.txt' % (label_name, participant_no)
        #treshhold adjustments for better labeling
        if label_name == 'Fist':                    
            x=0
        elif label_name == 'Pinch':
            x=300
        elif label_name == 'Thump':
            x=0
        elif label_name == 'Spred':
            x=0
        else:
            x=0
#############################################################################################################################
        # Import EMG data
        colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 
        raw_data = pd.read_csv(import_path+filename, delimiter=';',header=None, names=colum_names)
        emg_sum_data = raw_data.iloc[:, :6].sum(axis=1)/6                                                              #build an avg of all sensor data
#############################################################################################################################                           #####################################################################                                                           
    
        #Signal processing for each sensor
        time = FCNs.time_calc(raw_data)                                                                             #calculates the recorded time with the sensor sampels
        emg_without_offset = FCNs.remove_mean(raw_data,time,window_size)                                           #removing the offset of the EMG signal
        filtered_emg = FCNs.filteremg(time, emg_without_offset, low_pass, sfreq, high_band, low_band,notch_freq)    #filters the EMG signal with bandbass and notch filter
        filtered_emg = pd.DataFrame(filtered_emg)
#############################################################################################################################
    
        #Signal processing for the avg sesnor values
        emg_sum_data_df = emg_sum_data.to_frame()                                                                       #transformation to data frame
        emg_sum_without_offset = FCNs.remove_mean (emg_sum_data_df,time,window_size)                                                #removing the offset of the EMG signal
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
            emg_features_final = FCNs.replace_labels_zero(emg_features_replaced,label)  
                                        #replace wrong detected labels
       
        FCNs.plot_WL(emg_features_replaced, wl_threshold, 'Wavelentgh feature of the gesture '+label_name)
        emg_features_final = emg_features_replaced.drop('WL_sum', axis=1)
        emg_features_sum_final = emg_features_sum.assign(Label=emg_features_final['Label'])
        
        emg_features_final_array = np.array(emg_features_final)
        emg_features_normalized = FCNs.normalize(emg_features_final_array)
        
        plot_data = emg_features_sum_final [["Label"]]
        #FCNs.plot_features (plot_data,raw_data.iloc[:, 1],time,'')
#############################################################################################################################

        #Data export
        #emg_features_final.to_csv(export_path+'\labeled_Feature_Data_'+label_name+'.csv', index=False)                 #export labeld features as csv file
        #emg_features_sum_final.to_csv(export_path_sum + '\250_data\labeled_Feature_Data_' + label_name + '.csv', index=False)  # export labeled features as CSV file