import serial
import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# Set up serial connection
port = 'COM4'  # Modify this based on your ESP32 port
baud_rate = 500000
ser = serial.Serial(port, baud_rate)

# Global variables
record_data = False
data = []

# Create the main GUI window
root = tk.Tk()
root.title('EMG Sensor Data Recorder')

# Set up real-time plot
fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot([], [])
ax.set_xlabel('Time')
ax.set_ylabel('Sensor Data')
ax.set_title('Real-time Sensor Data')

# Embed the plot into the GUI window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

# Function to start data recording
def start_recording():
    global record_data, data
    record_data = True
    data = []
    messagebox.showinfo('Data Recording', 'Data recording started.')

# Function to stop data recording
def stop_recording():
    global record_data, data
    record_data = False
    save_data()

# Function to save recorded data to a text file
def save_data():
    global data
    file_path = filedialog.asksaveasfilename(defaultextension='.txt')
    if file_path:
        with open(file_path, 'w') as f:
            for line in data:
                f.write(','.join(map(str, line)) + '\n')
        messagebox.showinfo('Save Data', 'Data saved successfully.')

# Update real-time plot
def update_plot():
    global record_data, data
    if record_data:
        line_data = ser.readline().decode().strip()
        sensor_values = list(map(int, line_data.split(',')))
        data.append(sensor_values)
        x = np.arange(len(sensor_values))
        y = np.array(sensor_values)
        line.set_data(x, y)
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)
    root.after(1, update_plot)

# Start button
start_button = tk.Button(root, text='Start Recording', command=start_recording)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

# Stop button
stop_button = tk.Button(root, text='Stop Recording', command=stop_recording)
stop_button.pack(side=tk.LEFT, padx=10, pady=10)

# Save button
def save_data_with_filename():
    global data
    file_path = filedialog.asksaveasfilename(defaultextension='.txt')
    if file_path:
        with open(file_path, 'w') as f:
            for line in data:
                f.write(','.join(map(str, line)) + '\n')
        messagebox.showinfo('Save Data', 'Data saved successfully.')

save_button = tk.Button(root, text='Save Data', command=save_data_with_filename)
save_button.pack(side=tk.LEFT, padx=10, pady=10)

# Start updating the plot
update_plot()

# Run the GUI main loop
root.mainloop()

# Close the serial connection
ser.close()
