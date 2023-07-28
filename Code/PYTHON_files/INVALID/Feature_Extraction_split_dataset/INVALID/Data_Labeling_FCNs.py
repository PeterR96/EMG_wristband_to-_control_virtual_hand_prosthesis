# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 19:13:22 2023

@author: Peter Rott
"""


def label_data(data, column_name, threshold, label_name):
    
    if label_name == 'Fist':
        label = 1
        
    if label_name == 'Pinch':
        label = 2
        
    if label_name == 'Thumpup':
        label = 3
    
    if label_name == 'Spred':
        label = 4
    
    if label_name == 'Wavein':
        label = 5
    
    if label_name == 'Waveout':
        label = 6
        
    data['Label'] = data[column_name].apply(lambda x: label if x > threshold else 0)
    return data