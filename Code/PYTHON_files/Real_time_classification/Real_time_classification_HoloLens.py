import serial
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import Functions_real_time_classification as FCNs
import socket

sfreq = 1000
high_band = 20
low_band = 450
notch_freq = 50
colum_names = ['EMG_1', 'EMG_2', 'EMG_', 'EMG_4', 'EMG_5', 'EMG_6']
feature_names = ['MAV', 'ZC', 'SSC', 'WL', 'VAR', 'IEMG', 'RMS']
ML_model =tf.keras.models.load_model('./ANN_RT/Parameter_Study/Model/ANN_individual_realtime.h5')
ser = serial.Serial('COM10', 500000)
filtered_data_array = np.empty((0, 6))
mean_remove_data_array = np.empty((0, 6))
features_array = np.empty((0, 42))
label_pred_array = np.empty((0, 1))
data_array = np.empty((0,6))
delay_time = np.empty((0,1))


num_sensors = 6  # Number of sensors connected to the Arduino
buffer_size = 250  # Number of lines to collect before processing

# Initialize an empty list to store the arrays
sensor_arrays = []
execution_times = []
#hostname = "pcclassification.local"
#socket.sethostname(hostname)
# Server IP address and port
# Define the hostname used in the DNS server
server_hostname = 'pcclassification'

# Get the IP address of the server using the defined hostname
server_ip = socket.gethostbyname(server_hostname)
#server_ip = '192.168.4.2'  # Replace with your PC's IP address
server_port = 9000
#host = socket.getaddrinfo(hostname, None)[0][-1][0]
# Create a TCP server socket
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)

server_socket.bind((server_ip,server_port))


server_socket.listen(1)  # Number of simultaneous connections

print('Server listening for incoming connections...')

try:
    # Accept a client connection
    client_socket, client_address = server_socket.accept()
    print('Client connected:', client_address)
    
except KeyboardInterrupt:
    # User pressed "Ctrl+C" to interrupt the program
    print('Server shutdown by user.')

try:
    start_time = time.time()
    print(f"Start time: {start_time} ms")
    while True:
        start_time_loop = time.time()
        line_count = 0  # Counter for the number of lines accumulated
        start_time_i = time.time()
        sensor_values = []  # Temporarily store the sensor values for each iteration
        while line_count < buffer_size:
            if ser.in_waiting > 0:
                line = ser.readline().decode('latin-1', errors='replace').strip()
                values = line.split(",")

                if len(values) == num_sensors:
                    try:
                        # Split the elements and convert them to float
                        split_list = [float(value) for value in values]
                        sensor_values.append(split_list)
                        line_count += 1
                        #print(line)  # Optional: Print the line received
                    except ValueError:
                        print("Skipped line:", line)

        # Append the sensor_values to the sensor_arrays list
        sensor_arrays.append(sensor_values)
        
        if len(sensor_values) == buffer_size:
            emg_array = np.array (sensor_values)
            data_array = np.concatenate((data_array, emg_array))
            filtered_data, features = FCNs.process_data(emg_array, ML_model)
            features_norm = FCNs.normalize(features)
           
            y_pred = ML_model.predict(features_norm)
            y_pred = np.argmax(y_pred, axis=1)
            label = FCNs.label(y_pred)
            
            print(f"Label: {label}")
                 
            # Append the calculated values to the respective arrays
            filtered_data_array = np.vstack((filtered_data_array, filtered_data))
            features_array = np.vstack((features_array, features))
            label_pred_array = np.vstack((label_pred_array, y_pred))
            end_time_loop = time.time()
            time_loop = (end_time_loop - start_time_loop) * 1000
            delay_time = np.vstack((delay_time, time_loop))
            client_socket.sendall(str(y_pred).encode())
            
except KeyboardInterrupt:
    end_time = time.time()
    #Clean up or perform any necessary actions before exiting
    ser.close()
    print("Script interrupted. Exiting...")
    # Calculate the elapsed time in milliseconds
    delay_mean = np.mean(delay_time)-250
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Mean delay time: {delay_mean}")
    print(f"Elapsed time: {elapsed_time_ms} ms")
    filtered_data_array = pd.DataFrame(filtered_data_array,columns=colum_names)
    
    
    columns = [f"{feature}_{sensor+1}" for sensor in range(num_sensors) for feature in feature_names ]
    features_array = pd.DataFrame(features_array, columns=columns)
    data_array = pd.DataFrame (data_array, columns = colum_names)
    label_pred_array = pd.DataFrame(label_pred_array, columns = ["Label"])
    
    #data_array.to_csv("./Test_sequences/Data/Data_0%s.txt"%run, encoding="utf-8-sig", index=False, header=False, sep=';')
    #label_pred_array.to_csv("./Test_sequences/Label/Label_0%s.txt"%run, encoding="utf-8-sig", index=False, header=False)        

    client_socket.close()
    server_socket.close()
    