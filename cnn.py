#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 18:52:17 2019

@author: darya
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import pickle
import os.path as fs
import pandas as pd

from  classifiers import cnn_fit, cnn_predict
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


#КВАРТИЛИ ДЛЯ CNN
f1_cnn = np.zeros((len(D), amount_ex))
for d in range(len(D)):
    for i in range(amount_ex):
        file_name = 'cnn_d%02i_ex%02i'%(d*5, i+1)
        print('\n Start train '+file_name)
        [model, history] = cnn_fit(data_train[d, i], truly, N, data_test[d, i], truly_test)
        model.save(fs.join(path1, file_name+'.hdf5'))#сохраняем модель
        with open(fs.join(path1, 'history_'+file_name +'.json'), 'w') as f:#сохраняем историю
            json.dump(history.history, f)
        f1_cnn[d][i] = cnn_predict(data_test[d, i], truly_test, model)
        
f1_quartile_cnn = np.percentile(f1_cnn, [25, 50, 75], axis=1)
f1_quartile_cnn = np.transpose(f1_quartile_cnn)

#Записываем в файл значения квантилей
with h5py.File(str1_quantile, 'a') as f:
    f.create_dataset("f1_quartile_cnn", data=f1_quartile_cnn)

with h5py.File(str1_f1, 'a') as f:
    f.create_dataset("f1_cnn", data=f1_cnn)
