#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:57:01 2019

@author: darya
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing 


#нормирование
def norm(X):
    X_min = X.min(axis=2)
    X_max = X.max(axis=2)
    X_std = (X - X_min.reshape(*X.shape[:2],1)) / (X_max.reshape(*X.shape[:2],1) - X_min.reshape(*X.shape[:2],1))
    return X_std

#Проверка
def check_mean(arr_t):
    means = np.zeros((arr_t.shape[0], arr_t.shape[1], arr_t.shape[2]))
    for d in range(arr_t.shape[0]): 
        for i in range(arr_t.shape[1]):
            means[d, i] = np.mean(arr_t[d, i], axis = 1)
    
    return np.percentile(means.reshape(-1), [25, 50, 75])

#Проверка
def check_variance(arr_t):
    varias = np.zeros((arr_t.shape[0], arr_t.shape[1], arr_t.shape[2]))
    for d in range(arr_t.shape[0]): 
        for i in range(arr_t.shape[1]):
            varias[d, i] = np.var(arr_t[d, i], axis = 1)
    
   
    return np.percentile(varias.reshape(-1), [25, 50, 75])

def centering(arr_t):
    for d in range(arr_t.shape[0]): 
        for i in range(arr_t.shape[1]):
            t_mean = np.mean(arr_t[d, i], axis = 1)
            t_mean = t_mean.reshape(-1, 1)
            arr_t[d, i] = arr_t[d, i]-t_mean
    return arr_t
   
def rescale(arr_t):
    arr_t_new = np.zeros_like(arr_t)
    for d in range(arr_t.shape[0]): 
        for i in range(arr_t.shape[1]):
            arr_t_new[d, i] = preprocessing.scale(arr_t[d, i], axis = 1)
    return arr_t_new
  
def create_data_set(f1, f2, N, train_dim, test_dim, D, amount_ex, str_noise):
    
    #DataSet
    x = np.arange(0, 0.4 * np.pi, 0.4*np.pi/ 128)
    y1 = np.sin(f1 * x)
    y2 = np.sin(f2 * x)
 
    train = np.zeros((train_dim, N))
    test = np.zeros((test_dim, N))
    
    truly = np.zeros(train_dim)
    truly_test = np.zeros(test_dim)
    
    #тренировочные данные
    for i in range(train_dim):
        if (i % 2 == 0):
            train[i] = y1
            truly[i] = 0
        else:
            train[i] = y2
            truly[i] = 1
    
    #перемешаем
    mix = np.random.permutation(train_dim)
    train = train[mix]
    truly = truly[mix]
    
    #тестовые данные
    for i in range(test_dim):
        if (i % 2 == 0):
            test[i] = y1
            truly_test[i] = 0
        else:
            test[i] = y2
            truly_test[i] = 1
            
    #4D array of noise(train, test)
    arr_train = np.zeros(shape=(len(D), amount_ex, train_dim, N), dtype=float)
    arr_test = np.zeros(shape=(len(D), amount_ex, test_dim, N), dtype=float)
    
    for d in range(len(D)):
        arr_train[d] = np.random.uniform(-D[d]/2, D[d]/2, (amount_ex, train_dim, N))
        arr_test[d] = np.random.uniform(-D[d]/2, D[d]/2, (amount_ex, test_dim, N))

    #Центрируем шум
    arr_train = centering(arr_train)
    arr_test = centering(arr_test)
    
    #Прибавляем синус к шуму
    arr_train = arr_train+train.reshape(1, 1, *train.shape)
    arr_test =  arr_test+test.reshape(1, 1, *test.shape)
    
    #Стандартизируем  тренировочные и тестовые данные
    arr_train = rescale(arr_train)
    arr_test = rescale(arr_test)
    
    #Проверим, что данные центрировались
    print('Трениовочные данные, мат.ожидание:', check_mean(arr_train))
    print('Тестовые данные, мат.ожидание:', check_mean(arr_test))


         
    #Проверим, что данные нормировались
    print('Трениовочные данные, дисперсия:', check_variance(arr_train))
    print('Тестовые данные, дисперсия:', check_variance(arr_test))
 
    #Записываем в файл 
    with h5py.File(str_noise, 'w') as f:
        f.create_dataset("train", data=arr_train)
        f.create_dataset("test", data=arr_test)
        f.create_dataset("train_truly", data=truly)
        f.create_dataset("test_truly", data=truly_test)
        f.create_dataset("D", data=D)

    
    

def get_data_set(path):
    with h5py.File(path, 'r') as f:
        data_train = f['train'][...]
        data_test = f['test'][...]
        data_train_truly = f['train_truly'][...]
        data_test_truly = f['test_truly'][...]
        data_D = f['D'][...]
    return data_train.astype('float32'), data_test.astype('float32'), data_train_truly, data_test_truly, data_D

def print_plots(f1, f2, N, train_dim, test_dim, D, amount_ex):
    x = np.arange(0, 0.4 * np.pi, 0.4*np.pi/ 128)
    y1 = np.sin(f1 * x)
    y2 = np.sin(f2 * x)    
    plt.figure(figsize=(10.0, 7.0))
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.title("y(x)", fontsize=19)
    plt.plot(x, y1)
    plt.savefig('sin_f0.png', fontsize=19)
    plt.show()
    
    
    plt.figure(figsize=(10.0, 7.0))
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.title("y(x)", fontsize=19)
    plt.plot(x, y2)
    plt.savefig('sin_f1.png', fontsize=19)
    plt.show()
    
    for d in D:
        z = y1 + np.random.uniform(-d/2, d/2, y1.shape[0])
        plt.figure(figsize=(10.0, 7.0))
        plt.tick_params(axis="x", labelsize=18)
        plt.tick_params(axis="y", labelsize=18)
        plt.title("y(x), f0 = 10, D = %s"% d, fontsize=19)
        plt.plot(x, z)
        plt.savefig('noise_f0_%s.png'%d)
        plt.show()
        
    for d in D:
        print(d)
        z = y2 + np.random.uniform(-d/2, d/2, y1.shape[0])
        plt.figure(figsize=(10.0, 7.0))
        plt.tick_params(axis="x", labelsize=18)
        plt.tick_params(axis="y", labelsize=18)
        plt.title("y(x), f1 = 20, D = %s"% d, fontsize=19)
        plt.plot(x, z)
        plt.savefig('noise_f1_%s.png'%d)
        plt.show()
    
if __name__ == "__main__":
    N = 128
    f1 = 10
    f2 = 20
    train_dim = 10000
    test_dim = 1000
    D = np.arange(0,51,5)
    amount_ex = 30
    start = 0
    finish = 50
    step = 5
    
    create_data_set(f1, f2, N, train_dim, test_dim, D, amount_ex,'/home/darya/Documents/coursework/new_programm/noise.hdf5')
    print_plots(f1, f2, N, train_dim, test_dim, D, amount_ex)
    
