# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 19:08:02 2023

@author: Peter Rott
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import EMG_FCNs as FCNs

colum_names = ['emg_1','emg_2','emg_3','emg_4','emg_5','emg_6']
# Read the CSV file
filename = 'Rosi_Pinch.txt'
data = pd.read_csv(filename, delimiter=';', names=colum_names)
sums = data.iloc[:, :6].sum(axis=1)/6

time_sec = FCNs.time_calc(sums) 


# Create a subplot with 6 windows
fig, axs = plt.subplots(7, 1, figsize=(8, 12), sharex=True)

# Iterate over the sensors and plot the data in each window
for i, column in enumerate(data.columns):
    axs[i].plot(time_sec, data[column])
    axs[i].set_ylabel('Sensor ' + str(i+1))
    axs[i].set_title('EMG Sensor ' + str(i+1))

axs[6].plot(time_sec, sums)
axs[6].set_ylabel('Sum ')
axs[6].set_title('EMG Sensor Sum ')
# Set the x-axis label for the last subplot
axs[-1].set_xlabel('Time (seconds)')

# Get the file name without extension
title = os.path.splitext(filename)[0]

# Add a title for the entire plot
plt.suptitle(title, fontsize=16)

# Adjust the spacing between subplots
plt.tight_layout()

plt.savefig(f'{title}.png')

# Display the plot
plt.show()







