#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:52:10 2019

@author: darya
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import pickle
import os.path as fs
import pandas as pd

from  classifiers import log_fit, log_predict
from three_dim_noise import create_data_set, get_data_set

N = 128
f1 = 10
f2 = 20
train_dim = 10000
test_dim = 1000
dispersion = np.arange(0,51,5)
amount_ex = 30
   
str1_noise = fs.join('/home/dashat/redone_programm/noise.hdf5')
str1_quantile = fs.join('/home/dashat/redone_programm/quantile.hdf5')
str1_f1 = fs.join('/home/dashat/redone_programm/f1.hdf5')
path1 = fs.join("/home/dashat/redone_programm/information")


#str1_noise = fs.join('/home/darya/Documents/coursework/redone_programm/noise.hdf5')
#str1_quantile = fs.join( '/home/darya/Documents/coursework/redone_programm/quantile.hdf5')
#str1_f1 = fs.join('/home/darya/Documents/coursework/redone_programm/f1.hdf5')
#path1= fs.join("/home/darya/Documents/coursework/redone_programm/information")



#Получаем DATASET
[data_train, data_test, truly, truly_test, D] = get_data_set(str1_noise)



#КВАРТИЛИ ДЛЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ
f1_log = np.zeros((len(dispersion), amount_ex))
for d in range(len(dispersion)):
    for i in range(0, amount_ex):
        file_name = 'log_d%02i_ex%02i'%(d*5, i+1)
        print('\n Start train '+file_name)
        model = log_fit(data_train[d, i], truly)#обучили
        with open(fs.join(path1,file_name+'.pickle'), 'wb') as file:  
            pickle.dump(model, file)
        f1_log[d, i]= log_predict(model, truly_test, data_test[d, i])
      
f1_quartile_log = np.percentile(f1_log, [25, 50, 75], axis=1)
f1_quartile_log = np.transpose(f1_quartile_log)
with h5py.File(str1_quantile, 'w') as f:
    f.create_dataset("f1_quartile_log", data=f1_quartile_log)

with h5py.File(str1_f1, 'w') as f:
    f.create_dataset("f1_log", data=f1_log)
