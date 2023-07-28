import serial
import time
import pandas as pd
import tensorflow as tf
import numpy as np
import Functions_real_time_classification as FCNs
from joblib import load
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sfreq = 1000
high_band = 20
low_band = 450
notch_freq = 50
column_names = ['EMG_1', 'EMG_2', 'EMG_3', 'EMG_4', 'EMG_5', 'EMG_6']
feature_names = ['MAV', 'ZC', 'SSC', 'WL', 'VAR', 'IEMG', 'RMS']
ML_model = tf.keras.models.load_model('./Model/ANN_individual_realtime.h5')
ser = serial.Serial('COM10', 500000)
filtered_data_array = np.empty((0, 6))
mean_removed_data_array = np.empty((0, 6))
features_array = np.empty((0, 42))
label_pred_array = np.empty((0, 1))
data_array = np.empty((0, 6))
delay_time = np.empty((0, 1))
run = 5

num_sensors = 6  # Number of sensors connected to the Arduino
buffer_size = 250  # Number of lines to collect before processing

# Initialize an empty list to store the arrays
sensor_arrays = []
# filtered_data_df = []
execution_times = []

# Create a figure and axis for the plot
fig, ax = plt.subplots()

# Initialize an empty line
p, = ax.plot([], [])
# Turn on interactive mode
plt.ion()
# Show the initial plot
plt.show()


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
                        # print(line)  # Optional: Print the line received
                    except ValueError:
                        print("Skipped line:", line)

        # Append the sensor_values to the sensor_arrays list
        sensor_arrays.append(sensor_values)

        if len(sensor_values) == buffer_size:
            emg_array = np.array(sensor_values)
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

            # Update the x and y values for the plot
            x_values = np.arange(len(filtered_data_array))
            y_values = data_array[:, 0]  # Assuming you want to plot the first column of data_array

            # Update the plot data
            p.set_data(x_values, y_values)
            
            # Adjust the plot view limits
            ax.relim()
            ax.autoscale_view()
            
           # plt.show()
            
            # Refresh the plot
            fig.canvas.draw()
            plt.pause(0.001)

except KeyboardInterrupt:
    end_time = time.time()
    # Clean up or perform any necessary actions before exiting
    ser.close()
    print("Script interrupted. Exiting...")
    # Calculate the elapsed time in milliseconds
    delay_mean = np.mean(delay_time) - 250
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"Mean delay time: {delay_mean}")
    print(f"Elapsed time: {elapsed_time_ms} ms")
    filtered_data_array = pd.DataFrame(filtered_data_array, columns=column_names)

    columns = [f"{feature}_{sensor + 1}" for sensor in range(num_sensors) for feature in feature_names]
    features_array = pd.DataFrame(features_array, columns=columns)
    data_array = pd.DataFrame(data_array, columns=column_names)
    label_pred_array = pd.DataFrame(label_pred_array, columns=["Label"])

    data_array.to_csv("./Test_sequences/Data/Data_0%s.txt" % run, encoding="utf-8-sig", index=False, header=False,
                      sep=';')
    label_pred_array.to_csv("./Test_sequences/Label/Label_0%s.txt" % run, encoding="utf-8-sig", index=False,
                            header=False)

# Close the plot window when the loop is interrupted
plt.close()
