#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 00:40:25 2019

@author: darya
"""

import numpy as np
import math

import sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn. metrics import accuracy_score
from matplotlib.colors import ListedColormap
from scipy.stats.mstats import gmean
from sklearn import preprocessing 
from scipy.stats.mstats import gmean as gmean
from sklearn import metrics
from sklearn.model_selection import cross_val_score
#from imblearn.metrics import geometric_mean_score

def centering(arr_t):
    for i in range(len(arr_t)):
        t_mean = np.mean(arr_t)
        arr_t = arr_t-t_mean
    return arr_t
   
def rescale(arr_t):
    arr_t_new = np.zeros_like(arr_t)
    arr_t_new = preprocessing.scale(arr_t)
    return arr_t_new

f1=10
f2=20
x = np.arange(0, 0.4 * np.pi, 0.4*np.pi/ 128)
y1 = np.sin(f1 * x)
y2 = np.sin(f2 * x)
D = np.arange(0,51,5)


for d in D:
    z = rescale(y1 + centering(np.random.uniform(-d/2, d/2, y1.shape[0])))
    plt.figure(figsize=(10.0, 7.0))
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.title("y(x), f0 = 10, D = %s"% d, fontsize=19)
    plt.plot(x, z)
    plt.savefig('noise_f0_%s.png'%d)
    plt.show()
        
for d in D:
    print(d)
    z = rescale(y2 + centering(np.random.uniform(-d/2, d/2, y1.shape[0])))
    plt.figure(figsize=(10.0, 7.0))
    plt.tick_params(axis="x", labelsize=18)
    plt.tick_params(axis="y", labelsize=18)
    plt.title("y(x), f1 = 20, D = %s"% d, fontsize=19)
    plt.plot(x, z)
    plt.savefig('noise_f1_%s.png'%d)
    plt.show()