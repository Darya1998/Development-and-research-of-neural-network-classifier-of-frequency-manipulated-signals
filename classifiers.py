#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 20:01:49 2019

@author: darya
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K
from keras.layers import Conv1D, MaxPooling1D
from scipy.stats.mstats import gmean as gmean
from sklearn.metrics import classification_report
import tensorflow as tf

def callback_list(PathModel, NameModel, UpdatePath, UpdateFileNameMask, model):
    checkpoint = keras.callbacks.ModelCheckpoint(fs.join(PathModel, NameModel), verbose = 0,
                                                save_best_only = False, save_weights_only = False, mode = 'auto', period = 1)
    get_update = ku.getUpdateLogger_grads(UpdatePath, UpdateFileNameMask, model)
    return [checkpoint, get_update]


#_________________________ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ______________________________________
def log_fit(train,truly):
    lr = LogisticRegression(C=1000.0, random_state=0)
    return lr.fit(train, truly)
    
    
def log_predict(model,truly_test, test):
    y_pred = model.predict(test)
    return gmean(f1_score(truly_test, y_pred, average = None))


#_____________________________________NN____________________________________________-
def F1(y_pred, y_true):
        
    def recall(y_true, y_pred):
   
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def nn_sin_fit(train,truly, N, test, truly_test):
    batch_size = 64
    epochs = 100
    
    truly_ = truly.reshape(train.shape[0], 1)
    truly_test_ = truly_test.reshape(test.shape[0], 1)
     
    model = Sequential()
    model.add(Dense(30, activation='tanh', input_shape=(128,)))
    model.add(Dropout(0.25))
    model.add(Dense(16, activation='tanh'))
    model.add(Dropout(0.35))
    model.add(Dense(1, activation='sigmoid'))
    
#    print(model.summary())
    
    
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=[F1])
    
    history = model.fit(train,
                        truly_,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(test, truly_test_),
                        verbose=2)

    
    return model, history

    
def nn_sin_predict(test, truly_test, model):
    predict_test = model.predict(test)
    predict_test = np.heaviside(predict_test-0.5, 0).astype(np.int)  
    return gmean(f1_score(truly_test, predict_test, average = None))



#____________________________________СNN____________________________________________-
def cnn_fit(train, truly, N, test, truly_test):
    batch_size = 64
    epochs = 10
    
    PathModel = '/home/darya/Documents/coursework/ready_programm/redone_programm(new)/ModUpdatesMnist/model'
    NameModel = 'model.{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5'
    UpdatePath = '/home/darya/Documents/coursework/ready_programm/redone_programm(new)/ModUpdatesMnist/update'
    UpdateFileNameMask = 'batch_updates_%s.{epoch:03d}.hdf5'%K.backend()

    
    train_ = train.reshape(train.shape[0], train.shape[1], 1)
    truly_ = truly.reshape(train.shape[0], 1)
    test_ = test.reshape(test.shape[0], test.shape[1], 1)
    truly_test_ = truly_test.reshape(test.shape[0], 1)
        
    model = Sequential()
    model.add(Conv1D(64,  kernel_size = 3,
                     activation='tanh',
                     input_shape=(N, 1)))
    model.add(Conv1D(64, 3, activation='tanh'))
    model.add(MaxPooling1D(pool_size=(3)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics=[F1])
    
    callback_list = callback_list(PathModel, NameModel, UpdatePath, UpdateFileNameMask, model)
    
    
    model = load_model()
    history = model.fit(train_, truly_,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(test_, truly_test_),
              verbose=2, callback = callback_list)
    return model, history

def cnn_predict(test, truly_test, model):
    test_ = test.astype('float32')
    test_ = test_.reshape(test_.shape[0], test_.shape[1], 1)
#    truly_test_ = truly_test.reshape(test.shape[0], 1)
    predict_test = model.predict(test_)
    predict_test = np.heaviside(predict_test-0.5, 0).astype(np.int)
    return gmean(f1_score(truly_test, predict_test, average = None))
