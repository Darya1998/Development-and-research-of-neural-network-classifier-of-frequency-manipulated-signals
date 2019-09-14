#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:04:06 2019

@author: darya
"""

import numpy as np
import os.path as fs
from three_dim_noise import create_data_set

N = 128
f1 = 10
f2 = 20
train_dim = 10000
test_dim = 1000
dispersion = np.arange(0,51,5)
amount_ex = 30
   
str1_noise = fs.join('/home/dashat/redone_programm/noise.hdf5')
str1_quantile = fs.join('/home/dashat/redone_programm/quantile.hdf5')

#str1_noise = fs.join('/home/darya/Documents/coursework/redone_programm/noise.hdf5')
#str1_quantile = fs.join( '/home/darya/Documents/coursework/redone_programm/quantile.hdf5')

#Создаем новый DATASET
#create_data_set(f1, f2, N, train_dim, test_dim, dispersion, amount_ex, str1_noise)







