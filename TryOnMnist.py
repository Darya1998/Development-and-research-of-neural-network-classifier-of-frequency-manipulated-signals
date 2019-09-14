#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:06:48 2019

@author: vsevolod
"""
import numpy as np
import h5py as h5

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, LeakyReLU
import keras
from keras import backend as K

import os.path as fs 
import keras_utils as ku

def callback_list(PathModel, NameModel, UpdatePath, UpdateFileNameMask, model):
    checkpoint = keras.callbacks.ModelCheckpoint(fs.join(PathModel, NameModel), verbose = 0,
                                                save_best_only = False, save_weights_only = False, mode = 'auto', period = 1)
    get_update = ku.getUpdateLogger_grads(UpdatePath, UpdateFileNameMask, model)
    return [checkpoint, get_update]

PathModel = '/home/vsevolod/Desktop/ModUpdatesMnist/model'
NameModel = 'model.{epoch:03d}-{loss:.4f}-{val_loss:.4f}.hdf5'
UpdatePath = '/home/vsevolod/Desktop/ModUpdatesMnist/update'
UpdateFileNameMask = 'batch_updates_%s.{epoch:03d}.hdf5'%K.backend()

batch_size = 256
num_classes = 10
epochs = 1
opt = ku.Nadam_grads()

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

callback_list = callback_list(PathModel, NameModel, UpdatePath, UpdateFileNameMask, model)

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 2, validation_data = (x_test, y_test), callbacks = callback_list)


#Демонстрация вычисления градиентов по updates на примере SGD
update_read = ku.read_update(UpdatePath, 'batch_updates_%s.001.hdf5'%K.backend(), ['new_p', 'velocity', 'grads'])
layers_name = ku.preparation_layers_name(UpdatePath)

all_updates, layers_bound = ku.preparation_updated(update_read)
del update_read

#velocity = all_updates[1, :, :]
grads = ku.computation_grads_SGD(all_updates[1, :, :], lr_init=0.03, momentum=0.1, decay=1e-6)

#Средняя ошибка вычисления градиентов, grads (исходные) = all_updates[2, :, :]
print('Средняя ошибка вычисления градиентов')
print(np.sum(np.abs(grads - all_updates[2, :, :]))/(grads.shape[0]*grads.shape[1]))


# Построение и сохранения картинок
# Для сохранения картинок, в папке со скриптом и keras_utils создайте папку picture
ku.analysis_update(UpdatePath, 'batch_updates_%s.001.hdf5'%K.backend(), ['new_p', 'velocity', 'grads'])


#Построение и сохранение картинок, если нет градиентов в updates
# Для сохранения картинок, в папке со скриптом и keras_utils создайте папку picture
#ku.analysis_update_without_grads(UpdatePath, 'batch_updates_%s.001.hdf5'%K.backend(), ['new_p'], grads)