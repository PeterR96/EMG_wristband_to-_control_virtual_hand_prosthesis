import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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
x = 0                       #Overlap of Segments: Window_size - overlap
label_name = 'test_data'
run = 3
# Fist = 1 ; Pinch = 2; Thump = 3; Spred = 4

label_list_1 = [2, 4, 4, 3, 2, 1, 3, 1, 3, 1]
label_list_2 = [3, 2, 1, 2, 3, 4, 3, 2, 1, 4]
label_list_3 = [3, 2, 4, 1, 2, 4, 2, 3, 2, 1]
label_list_4 = [1, 1, 3, 3, 1, 4, 4, 2, 3, 3]
label_list_5 = [2, 4, 1, 3, 2, 3, 3, 4, 3, 2]
label_list_6 = [3, 4, 1, 1, 4, 4, 2, 3, 2, 4]
label_list_7 = [3, 3, 4, 4, 1, 3, 1, 4, 2, 2]
label_list_8 = [1, 1, 4, 3, 2, 1, 2, 3, 1, 3]
label_list_9 = [4, 4, 2, 3, 1, 1, 3, 4, 2, 1]
label_list_10 =[2, 3, 4, 4, 3, 1, 4, 2, 1, 4]

label_list = [
    label_list_1,
    label_list_2,
    label_list_3,
    label_list_4,
    label_list_5,
    label_list_6,
    label_list_7,
    label_list_8,
    label_list_9,
    label_list_10
]

def get_element_at_index(label_list, index):
    if index < 0 or index >= len(label_list):
        return None
    return label_list[index]


label_list = get_element_at_index(label_list, run-1)

test_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Real_time_classification\Test_sequences'
filename = r'./Data/Data_0%s.txt'%run
label_file = r'./Label/Label_0%s.txt'%run

#Import EMG data
colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6'] 
raw_data = pd.read_csv(test_path+filename, delimiter=';', names=colum_names)
label_data = pd.read_csv(test_path + label_file, names= ['Label'])
emg_sum_data = raw_data.iloc[:, :6].sum(axis=1)/6           

#new_row = [0]
#label_data = label_data.append(pd.Series(new_row, index=label_data.columns), ignore_index=True)                                                   #build an avg of all sensor data
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
FCNs.plot_labels(plot_data,filtered_emg.iloc[:, 1],time,'Correct labels and the predicted Labels',label_data)
#############################################################################################################################

def print_confusion_matrix(y_true, y_pred, title, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    ax.set_title(title)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_true, y_pred))

y_true = emg_features_final.loc[:, "Label"]
print_confusion_matrix(y_true, label_data, 'Confusion Matrix for ANN real time classification')