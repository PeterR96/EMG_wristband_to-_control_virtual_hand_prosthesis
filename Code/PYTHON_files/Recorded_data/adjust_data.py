import pandas as pd

label_names = ['Fist', 'Pinch', 'Thump', 'Spred']
participant_nos =  ['1']#['2', '3', '4']#, '5', '6', '7', '8', '9', '10']
y = 1000
raw_data = {}
adjusted_data = {}
for participant_no in participant_nos:
    import_path = r'C:\Users\Peter Rott\Desktop\Masterthesis\Coding\PYTHON_files\Recorded_data\%s' % participant_no

    for label_name in label_names:
        filename = '\%s_data_0%s.txt' % (label_name, participant_no)
                   
        # Import EMG data
        #column_names = ['emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6']
        data_raw = pd.read_csv(import_path + filename, delimiter=';', header=None, names=None)
        
        # Multiply the data with the adjustment value
        
        data_adjusted = data_raw * y
        # Round the values to two decimal places
        data_adjusted = data_adjusted.applymap('{:.2f}'.format)
        
        # Store the raw data and adjusted data in the dictionaries with the label name as the key
        raw_data[label_name] = data_raw
        adjusted_data[label_name] = data_adjusted
        
        # Save the adjusted data to a text file
        data_adjusted.to_csv('./%s/Adjusted/%s_data_0%s.txt' % (participant_no,label_name, participant_no), sep=';', index=False, header=False)