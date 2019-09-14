# -*- coding: utf-8 -*-
"""
Created on 2018.04.17

@author: poruss
"""

import numpy as np
import h5py as h5
import os.path as fs
import matplotlib.pyplot as plt
import bottleneck as bn

import keras
from keras import backend as K
import keras.optimizers as opt
from keras.callbacks import Callback

from keras.layers import Layer, InputSpec
from keras.initializers import Initializer
from keras.backend.tensorflow_backend import tf, _regular_normalize_batch_in_training
from keras import initializers, regularizers, constraints

from numba import njit, prange, float32
#%%
epsilon = K.epsilon()
def get_custom_crossentropy(k1=1.0, k2=1.0, k3=1.0, k4=1.0):
  def custom_crossentropy(target, output):
    output = K.clip(output, epsilon, 1 - epsilon)
    output = -k1*target*K.log(output) - k2*(1-target)*K.log(1-output) - k3*K.log(1-K.relu(target-output)) - k4*K.log(1-K.relu(output-target))
    return output
  return custom_crossentropy

#%%
class LearningRateLogger(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end=' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')
    if not hasattr(self.model.optimizer, 'decay'):
      print('Optimizer don\'t have a "decay" attribute.')
    if not hasattr(self.model.optimizer, 'iterations'):
      print('Optimizer don\'t have a "iterations" attribute.')
  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    decay = self.model.optimizer.decay
    iterations = self.model.optimizer.iterations
    lr_with_decay = lr / (1. + decay * K.cast(iterations, K.dtype(decay)))
    self.lr_logs.append(K.eval(lr_with_decay))
    print(' - curent_lr: %.6f'%K.eval(lr_with_decay))

class LearningRateLoggerForNadam(keras.callbacks.Callback):
  def on_train_begin(self, logs={}):
    print('Optimizer:', self.model.optimizer.__class__.__name__, end=' - ')
    print(self.model.optimizer.get_config())
    self.lr_logs = []
    if not hasattr(self.model.optimizer, 'lr'):
      print('Optimizer don\'t have a "lr" attribute.')
  def on_epoch_end(self, epoch, logs={}):
    lr = self.model.optimizer.lr
    self.lr_logs.append(K.eval(lr))
    print(' - curent_lr: %.6f'%K.eval(lr))

#%%
def get_lr_logger(model):
  if model.optimizer.__class__.__name__ == 'Nadam':
    lr_logger = LearningRateLoggerForNadam()
  else:
    lr_logger = LearningRateLogger()
  return lr_logger

#%%
def get_lr_scheduler(lr_sched, batch_sched):
  if len(lr_sched) != len(batch_sched):
    print('len(lr_sched) != len(batch_sched)')
    return None
  lr_schedule = np.ones(np.sum(batch_sched), dtype=np.float32)
  lr_schedule[:batch_sched[0]] = np.full(batch_sched[0], lr_sched[0], dtype=np.float32)
  for i in range(1, len(batch_sched)):
    lr_schedule[np.sum(batch_sched[:i]):np.sum(batch_sched[:i+1])] = \
      np.full(batch_sched[i], lr_sched[i], dtype=np.float32)
  lr_schedule = lr_schedule.tolist()

  def scheduler(epoch):
    # print('\nScheduler set lr: %.6f'%lr_schedule[epoch])
    return lr_schedule[epoch]

  LRS = keras.callbacks.LearningRateScheduler(scheduler)
  return LRS

#%%
def get_model_checkpointer(ModelFileNameMask):
  MCheckpointer = keras.callbacks.ModelCheckpoint(ModelFileNameMask, monitor='val_loss',
     verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
  return MCheckpointer
#%%
"""
Gettings update without gradients
"""

#save layers name in model
def save_layer_name(PathFile, model):
  f = h5.File(fs.join(PathFile, 'layers_name.hdf5'), 'w')
  param_names = []
  for la in model.layers:
    p_names = la.weights
    for i in range(len(p_names)):
      param_names.append(p_names[i].name+" "+str(p_names[i].shape))
  for i in range(len(param_names)):
    f.create_dataset('weights_%d'%i, data = param_names[i])
  f.close()
#tensorfow backend
class UpdateLoggerTF(Callback):
  def __init__(self, filepath, model):
    super(UpdateLoggerTF, self).__init__()
    self.filepath = filepath
    self.k = 1  # start of weights from updates
    self.update_names = []  # different for optimizers
    self.updates_RP = []
    self.loss_RP = []

    model._make_train_function()
    i = 0
    for update in model.optimizer.updates:
      if isinstance(update, tuple):  # special for Nadam
        for t in range(len(update)):  # m_schedule
          model.metrics_names.append("update_%d"%i)
          model.train_function.outputs.append(update[t])
          i += 1
      else:
        model.train_function.outputs.append(update)
        model.metrics_names.append("update_%d"%i)
        i += 1
    self.len_updates = i

  def on_train_begin(self, logs=None):
    print(self.model.optimizer.__class__.__name__, self.model.optimizer.get_config())
    if len(self.model.metrics_names) < len(self.model.optimizer.updates):
      raise ValueError(
        "metrics_names: %s, bat len(updates) = %d"%(self.model.metrics_names, len(self.model.optimizer.updates)))
    print(self.model.metrics_names)

    if self.model.optimizer.__class__.__name__ == "SGD":
      self.update_names = ["iterations", "velocity", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "RMSprop":
      self.update_names = ["iterations", "accumulators", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adagrad":
      self.update_names = ["iterations", "accumulators", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adadelta":
      self.update_names = ["iterations", "accumulators", "new_p", "delta_accumulators"]
    elif self.model.optimizer.__class__.__name__ == "Adam":
      if self.model.optimizer.amsgrad:
        self.update_names = ["iterations", "vhat", "m", "v", "new_p"]
      else:
        self.update_names = ["iterations", "m", "v", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adamax":
      self.update_names = ["iterations", "m", "u", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Nadam":
      self.update_names = ["iterations", "m_schedule", "m", "v", "new_p"]
      self.k = 2
    for _ in self.update_names:
      self.updates_RP.append([])
    print(self.update_names)

  def reshape_updates(self, updates):
    self.updates_RP[0].append(updates.pop(0))  # iterations

    if self.model.optimizer.__class__.__name__ == "Nadam":
      self.updates_RP[1].append([updates.pop(0), updates.pop(0)])  # m_schedule

    n = len(self.update_names)-self.k
    for i in range(n):
      temp = []
      n_t = len(updates)//n
      for t in range(n_t):
        temp.append(updates[i+t*n])
      self.updates_RP[i+self.k].append(np.asarray(temp))

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    self.loss_RP.append(logs["loss"])
    batch_updates = []
    for i in range(self.len_updates):
      batch_updates.append(logs["update_%d"%i])
    self.reshape_updates(batch_updates)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    filepath = self.filepath.format(epoch=epoch + 1, **logs)
    f = h5.File(filepath, 'w')
    f.create_dataset("len_iterations", data=len(self.updates_RP[0]), dtype=np.int)
    f.create_dataset("len_weights", data=len(self.model.get_weights()), dtype=np.int)
    dt = h5.special_dtype(vlen=str)
    data_text = [name.encode('ascii', 'ignore') for name in self.update_names]
    f.create_dataset("update_names", data=data_text, dtype=dt)
    f.create_dataset("loss", data=self.loss_RP, dtype=np.float32)

    for i in range(self.k):
      weights = np.asarray(self.updates_RP[i])
      f.create_dataset(self.update_names[i], data=weights, dtype=np.float32)

    n = len(self.update_names)-self.k
    for i in range(n):
      weights = np.asarray(self.updates_RP[i+self.k])
      for w in range(weights.shape[1]):
        temp = np.zeros((len(weights), *weights[0][w].shape), dtype=np.float32)
        for t in range(weights.shape[0]):
          temp[t] = weights[t][w]
        f.create_dataset("%s/weights_%d"%(self.update_names[i+self.k], w), data = temp,
                         dtype=np.float32)
    f.close()

    self.loss_RP = []
    self.updates_RP = []
    for _ in self.update_names:
      self.updates_RP.append([])
###############################################################################
#cntk backend
class UpdateLoggerCNTK(Callback):
  def __init__(self, filepath):
    super(UpdateLoggerCNTK, self).__init__()
    self.filepath = filepath
    self.k = 1 # start of weights from updates
    self.update_names = [] # different for optimizers
    self.updates_RP = []
    self.loss_RP = []

  def on_train_begin(self, logs = None):
    print(self.model.optimizer.__class__.__name__, self.model.optimizer.get_config())
    if self.model.optimizer.__class__.__name__ == 'SGD':
      self.update_names = ['iterations', 'velocity', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'RMSprop':
      self.update_names = ['iterations', 'accumulators', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adagrad':
      self.update_names = ['iterations', 'accumulators', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adadelta':
      self.update_names = ['iterations', 'accumulators', 'new_p', 'delta_accumulators']
    elif self.model.optimizer.__class__.__name__ == 'Adam':
      if self.model.optimizer.amsgrad:
        self.update_names = ['iterations', 'vhat', 'm', 'v', 'new_p']
      else:
        self.update_names = ['iterations', 'm', 'v', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adamax':
      self.update_names = ['iterations', 'm', 'u', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Nadam':
      self.update_names = ['iterations', 'm_schedule', 'm', 'v', 'new_p']
      self.k = 2
    for _ in self.update_names:
      self.updates_RP.append([])
    print(self.update_names)

  def reshape_updates(self, updates):
    self.updates_RP[0].append(updates[0]) # iterations

    if self.model.optimizer.__class__.__name__ == 'Nadam':
      self.updates_RP[1].append(updates[1]) # m_schedule

    n = len(self.update_names)-self.k
    for i in range(n):
      temp = []
      n_t = (len(updates)-self.k)//n
      for t in range(n_t):
        temp.append(updates[i+self.k+t*n])
      self.updates_RP[i+self.k].append(np.asarray(temp))

  def on_batch_end(self, batch, logs=None):
    logs = logs or {}
    self.loss_RP.append(logs["loss"])
    batch_updates = []
    if self.model.optimizer.__class__.__name__ == 'Nadam':
      batch_updates.append(K.get_value(self.model.optimizer.updates[0])) # iterations
      batch_updates.append(K.batch_get_value(self.model.optimizer.updates[1])) # m_schedule
      for i in range(2, len(self.model.optimizer.updates)):
        batch_updates.append(K.get_value(self.model.optimizer.updates[i]))
    else:
      batch_updates = K.batch_get_value(self.model.optimizer.updates)
    self.reshape_updates(batch_updates)
    #print(batch_updates[0])

  def on_epoch_end(self, epoch, logs = None):
    logs = logs or {}
    #updates = self.updates_RP
    filepath = self.filepath.format(epoch = epoch + 1, **logs)
    f = h5.File(filepath, 'w')
    f.create_dataset('batch_size', data = len(self.updates_RP[0]), dtype = np.int)
    f.create_dataset('weights_shape', data = len(self.model.get_weights()), dtype = np.int)
    dt = h5.special_dtype(vlen = str)
    data_text = [name.encode('ascii', 'ignore') for name in self.update_names]
    f.create_dataset('update_names', data = data_text, dtype = dt)
    f.create_dataset("loss", data = self.loss_RP, dtype = np.float32)

    for i in range(self.k):
      weights = np.asarray(self.updates_RP[i])
      f.create_dataset('%s'%self.update_names[i], data = weights, dtype = np.float32)

    n = len(self.update_names)-self.k
    for i in range(n):
      weights = np.asarray(self.updates_RP[i+self.k])
      for w in range(weights.shape[1]):
        temp = np.zeros((len(weights), *weights[0][w].shape), dtype = np.float32)
        for t in range(weights.shape[0]):
          temp[t] = weights[t][w]
        f.create_dataset('%s/weights_%d'%(self.update_names[i+self.k], w), data = temp, dtype = np.float32)
    f.close()

    self.updates_RP = []
    self.loss_RP = []
    for _ in self.update_names:
      self.updates_RP.append([])

###############################################################################
#save weight
class WeightLogger(Callback):
  def __init__(self, filepath):
    super(WeightLogger, self).__init__()
    self.filepath = filepath
    self.weights_RP = []

  def on_train_begin(self, logs = None):
    print(self.model.optimizer.__class__.__name__, self.model.optimizer.get_config())

  def on_batch_end(self, batch, logs = None):
    logs = logs or {}
    ws = self.model.get_weights()
    self.weights_RP.append(np.asarray(ws))

  def on_epoch_end(self, epoch, logs = None):
    logs = logs or {}
    weights = np.asarray(self.weights_RP)
    filepath = self.filepath.format(epoch = epoch + 1, **logs)
    f = h5.File(filepath, 'w')
    f.create_dataset('batch_size', data = len(weights), dtype = np.int)
    f.create_dataset('weights_shape', data = weights.shape[1], dtype = np.int)
    for w in range(weights.shape[1]):
      temp = np.zeros((len(weights), *weights[0][w].shape), dtype = np.float32)
      for i in range(weights.shape[0]):
        temp[i] = weights[i][w]
      f.create_dataset('weights_%d'%(w), data = temp, dtype = np.float32)
    f.close()
    self.weights_RP = []


def getUpdateLogger(filepath, maskfile, model):
  save_layer_name(filepath, model)
  if K.backend() == 'cntk':
    return UpdateLoggerCNTK(fs.join(filepath, maskfile))
  if K.backend() == 'tensorflow':
    return UpdateLoggerTF(fs.join(filepath, maskfile), model)
#%%
"""
Custom optimizers
"""
class SGD_grads(opt.SGD):
     def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            v = self.momentum * m - lr * g  # velocity
            self.updates.append(K.update(m, v))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

###############################################################################
class RMSprop_grads(opt.RMSprop):
     def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

###############################################################################
class Adagrad_grads(opt.Adagrad):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a in zip(params, grads, accumulators):
            new_a = a + K.square(g)  # update accumulator
            self.updates.append(K.update(a, new_a))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            new_p = p - lr * g / (K.sqrt(new_a) + self.epsilon)

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

###############################################################################
class Adadelta_grads(opt.Adadelta):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        shapes = [K.int_shape(p) for p in params]
        accumulators = [K.zeros(shape) for shape in shapes]
        delta_accumulators = [K.zeros(shape) for shape in shapes]
        self.weights = accumulators + delta_accumulators
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        for p, g, a, d_a in zip(params, grads, accumulators, delta_accumulators):
            # update accumulator
            new_a = self.rho * a + (1. - self.rho) * K.square(g)
            self.updates.append(K.update(a, new_a))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            # use the new accumulator and the *old* delta_accumulator
            update = g * K.sqrt(d_a + self.epsilon) / K.sqrt(new_a + self.epsilon)
            new_p = p - lr * update

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))

            # update delta_accumulator
            new_d_a = self.rho * d_a + (1 - self.rho) * K.square(update)
            self.updates.append(K.update(d_a, new_d_a))
        return self.updates

###############################################################################
class Adam_grads(opt.Adam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

###############################################################################
class Adamax_grads(opt.Adamax):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr / (1. - K.pow(self.beta_1, t))

        shapes = [K.int_shape(p) for p in params]
        # zero init of 1st moment
        ms = [K.zeros(shape) for shape in shapes]
        # zero init of exponentially weighted infinity norm
        us = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + ms + us

        for p, g, m, u in zip(params, grads, ms, us):

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            u_t = K.maximum(self.beta_2 * u, K.abs(g))
            p_t = p - lr_t * m_t / (u_t + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(u, u_t))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

###############################################################################
class Nadam_grads(opt.Nadam):
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            if K.backend() == 'cntk':
                self.updates.append(K.update(g, g))

            if K.backend() == 'tensorflow':
                self.updates.append(K.cast(g, K.floatx()))

            p_t = p - self.lr * m_t_bar / (K.sqrt(v_t_prime) + self.epsilon)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
#%%
class UpdateLoggerTF_grads(UpdateLoggerTF):
  def on_train_begin(self, logs = None):
    print(self.model.optimizer.__class__.__name__, self.model.optimizer.get_config())
    if len(self.model.metrics_names) < len(self.model.optimizer.updates):
      raise ValueError(
        "metrics_names: %s, bat len(updates) = %d"%(self.model.metrics_names, len(self.model.optimizer.updates)))
    print(self.model.metrics_names)

    if self.model.optimizer.__class__.__name__ == "SGD_grads":
      self.update_names = ["iterations", "velocity", "grads", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "RMSprop_grads":
      self.update_names = ["iterations", "accumulators", "grads", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adagrad_grads":
      self.update_names = ["iterations", "accumulators", "grads", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adadelta_grads":
      self.update_names = ["iterations", "accumulators", "grads", "new_p", "delta_accumulators"]
    elif self.model.optimizer.__class__.__name__ == "Adam_grads":
      if self.model.optimizer.amsgrad:
        self.update_names = ["iterations", "vhat", "m", "v", "grads", "new_p"]
      else:
        self.update_names = ["iterations", "m", "v", "grads", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Adamax_grads":
      self.update_names = ["iterations", "m", "u", "grads", "new_p"]
    elif self.model.optimizer.__class__.__name__ == "Nadam_grads":
      self.update_names = ["iterations", "m_schedule", "m", "v", "grads", "new_p"]
      self.k = 2
    for _ in self.update_names:
      self.updates_RP.append([])
    print(self.update_names)

  def reshape_updates(self, updates):
    self.updates_RP[0].append(updates.pop(0))  # iterations

    if self.model.optimizer.__class__.__name__ == "Nadam_grads":
      self.updates_RP[1].append([updates.pop(0), updates.pop(0)])  # m_schedule

    n = len(self.update_names)-self.k
    for i in range(n):
      temp = []
      n_t = len(updates)//n
      for t in range(n_t):
        temp.append(updates[i+t*n])
      self.updates_RP[i+self.k].append(np.asarray(temp))


class UpdateLoggerCNTK_grads(UpdateLoggerCNTK):
  def on_train_begin(self, logs = None):
    print(self.model.optimizer.__class__.__name__, self.model.optimizer.get_config())
    if self.model.optimizer.__class__.__name__ == 'SGD_grads':
      self.update_names = ['iterations', 'velocity', 'grads', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'RMSprop_grads':
      self.update_names = ['iterations', 'accumulators', 'grads', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adagrad_grads':
      self.update_names = ['iterations', 'accumulators', 'grads', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adadelta_grads':
      self.update_names = ['iterations', 'accumulators', 'grads', 'new_p', 'delta_accumulators']
    elif self.model.optimizer.__class__.__name__ == 'Adam_grads':
      if self.model.optimizer.amsgrad:
        self.update_names = ['iterations', 'vhat', 'm', 'v', 'grads', 'new_p']
      else:
        self.update_names = ['iterations', 'm', 'v', 'grads', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Adamax_grads':
      self.update_names = ['iterations', 'm', 'u', 'grads', 'new_p']
    elif self.model.optimizer.__class__.__name__ == 'Nadam_grads':
      self.update_names = ['iterations', 'm_schedule', 'm', 'v', 'grads', 'new_p']
      self.k = 2
    for _ in self.update_names:
      self.updates_RP.append([])
    print(self.update_names)

  def reshape_updates(self, updates):
    self.updates_RP[0].append(updates[0]) # iterations

    if self.model.optimizer.__class__.__name__ == 'Nadam_grads':
      self.updates_RP[1].append(updates[1]) # m_schedule

    n = len(self.update_names)-self.k
    for i in range(n):
      temp = []
      n_t = (len(updates)-self.k)//n
      for t in range(n_t):
        temp.append(updates[i+self.k+t*n])
      self.updates_RP[i+self.k].append(np.asarray(temp))

  def on_batch_end(self, batch, logs = None):
    logs = logs or {}
    #print(logs)
    batch_updates = []
    if self.model.optimizer.__class__.__name__ == 'Nadam_grads':
      batch_updates.append(K.get_value(self.model.optimizer.updates[0])) # iterations
      batch_updates.append(K.batch_get_value(self.model.optimizer.updates[1])) # m_schedule
      for i in range(2, len(self.model.optimizer.updates)):
        batch_updates.append(K.get_value(self.model.optimizer.updates[i]))
    else:
      batch_updates = K.batch_get_value(self.model.optimizer.updates)
    self.reshape_updates(batch_updates)
    #print(batch_updates[0])
###############################################################################
def getUpdateLogger_grads(filepath, maskfile, model):
  save_layer_name(filepath, model)
  if K.backend() == 'cntk':
    return UpdateLoggerCNTK_grads(fs.join(filepath, maskfile))
  if K.backend() == 'tensorflow':
    return UpdateLoggerTF_grads(fs.join(filepath, maskfile), model)
#%%
def analysis_update(PathUpdates, NamesUpdates, update_list):
    update_read = read_update(PathUpdates, NamesUpdates, update_list)
    layers_name = preparation_layers_name(PathUpdates)

    all_updates, layers_bound = preparation_updated(update_read)
    del update_read
    plot_analysis_update(PathUpdates, NamesUpdates, update_list, all_updates, layers_bound, layers_name)
    pass

def analysis_update_without_grads(PathUpdates, NamesUpdates, update_list, grads):
    update_read = read_update(PathUpdates, NamesUpdates, update_list)
    layers_name = preparation_layers_name(PathUpdates)

    all_updates, layers_bound = preparation_updated(update_read)
    del update_read

    update_list = update_list + ['grads']
    grads = np.reshape(grads, (1, grads.shape[0], grads.shape[1]))
    all_updates = np.concatenate((all_updates, grads), axis = 0)

    plot_analysis_update(PathUpdates, NamesUpdates, update_list, all_updates, layers_bound, layers_name)
    pass

def plot_analysis_update(PathUpdates, NamesUpdates, update_list, all_updates, layers_bound, layers_name):
    q_list = [0.5, 5.0, 25.0, 50.0, 75.0, 95.0, 99.5]

    loss_every_batch = preparation_loss(PathUpdates, NamesUpdates)

    for i in range(len(update_list)):
        plot_weights_series(all_updates[i, -1, :], update_list[i], bins = 750)

    for i in range(len(update_list)):
        perecentile_plot(all_updates[i, -1, :], layers_bound, q_list, update_list[i])

    percentile_layers_batch(np.abs(all_updates), layers_bound, update_list, q_list, layers_name)
    pass


#%%
"""
Считывание updates
"""
def read_update(PathUpDatesFile, NameUpDatesFile, update_list):
    f = h5.File(fs.join(PathUpDatesFile, NameUpDatesFile), 'r')
    updates_data_list = []
    for name_group in update_list:
        updates_data_list.append(reading_by_layers(f, name_group))
    f.close()
    return updates_data_list

"""
Послойное считываение updates сети
"""
def reading_by_layers(f, name_group):
    group = f[name_group]
    weight_list = []
    n = len(list(group.keys()))
    for i in range(n):
        weight_list.append(f[name_group]['weights_%d'%i][...])
    return weight_list

"""
Нахождение границ слоев в развернутом массиве
"""
def find_layer_boundaries(update_one_batch):
    layers_bound = [0]
    len_weight = 0
    n = len(update_one_batch) #Количество слоев
    for i in range(n):
        array = update_one_batch[i][0].flatten()
        len_weight = len_weight + len(array)
        layers_bound.append(len_weight)
    return layers_bound

"""
Преобразование массивов в удобный вид
"""
def preparation_updated(updates_data_list):
    num_updates_name = len(updates_data_list)
    num_layers = len(updates_data_list[0])
    num_batches = len(updates_data_list[0][0])

    layers_bound = find_layer_boundaries(updates_data_list[0])

    all_updates = np.empty((num_updates_name, num_batches, layers_bound[-1]), dtype = np.float32)
    for upd_name in range(num_updates_name):
        for batch in range(num_batches):
            for layer in range(num_layers):
                all_updates[upd_name, batch, layers_bound[layer]:layers_bound[layer + 1]] = updates_data_list[upd_name][layer][batch].flatten()
    del updates_data_list
    return all_updates, layers_bound

# вычисление перцентилей ряда
def percentile(series, q):
  return np.percentile(series, q, axis = 0)

def preparation_layers_name(PathUpDates):
    layers_name = read_name_layer(PathUpDates)
    layers_name = parsing_name(layers_name)
    return layers_name

def read_name_layer(PathUpDates):
    f = h5.File(fs.join(PathUpDates, 'layers_name.hdf5'), 'r')
    layers_name = []
    n = len(list(f.keys()))
    for i in range(n):
        layers_name.append(f['weights_%d'%i][...])
    f.close()
    return layers_name

def parsing_name(layers_name):
    n = len(layers_name)
    layers_name_pars = []
    for i in range(n):
        a = str(layers_name[i]).find('\'')
        b = str(layers_name[i]).rfind('\'')
        layers_name_pars.append(str(layers_name[i])[a + 1:b])
    return layers_name_pars

def preparation_loss(PathUpDates, NameUpDates, figsize = (15, 10), window = 15):
    loss_every_batch = read_loss(PathUpDates, NameUpDates)
    plt_loss_one_epoch(loss_every_batch, figsize, window)
    return loss_every_batch

#считывание значение loss функци
def read_loss(PathUpDates, NameUpDates, size = 0):
    f = h5.File(fs.join(PathUpDates, NameUpDates), 'r')
    if size == 0:
        loss_every_batch = f['loss'][...]
    else:
        loss_every_batch = f['loss'][:size]
    f.close()
    return loss_every_batch

#построение loss функции (используется скользящее среднее, window - размер окна скольжения)
def plt_loss_one_epoch(batch_loss, figsize = (15, 10), window = 15):
    loss_every_batch_mm = bn.move_mean(batch_loss, window = window, min_count=1)
    plt.figure(figsize = figsize)
    plt.plot(np.arange(len(batch_loss)), batch_loss, np.arange(len(loss_every_batch_mm)), loss_every_batch_mm)
    plt.yscale('log')
    plt.title('loss epoch')
    plt.xlabel('batchs')
    plt.ylabel('loss')
    plt.savefig(fs.join('pictures', 'loss.png'))
    pass

#построение гистограммы распределения
def plot_weights_series(series, title_name, bins = 100):
    plt.figure(figsize=(10, 10))
    plt.title('Гистограмма распределения %s'%title_name)
    plt.hist(series, bins = bins)
    plt.savefig(fs.join('pictures','Гистограмма распределения %s'%title_name))
    pass

#построение перцентильного распределения
def perecentile_plot(series, layers_bound, q_list, title_name):
    n = len(layers_bound)
    plt.figure(figsize=(10, 10))
    lst_abs = ['abs = False', 'abs = True']
    for j in range(2):

        for q in q_list:
            p_series = []
            for i in range(n - 1):
                if j == 0:
                    p_series.append(percentile(series[layers_bound[i]:layers_bound[i + 1]], q))
                else:
                    p_series.append(percentile(np.abs(series[layers_bound[i]:layers_bound[i + 1]]), q))

            plt.subplot(2, 1, j + 1)
            if q == 50.0:
                plt.plot(np.arange(len(p_series)), p_series, lw = 3)
            else :
                plt.plot(np.arange(len(p_series)), p_series)

        plt.title('Квантили распределения %s (по слоям); %s'%(title_name, lst_abs[j]))
        plt.xlabel('слой')
        plt.ylabel('значение')
        plt.grid(True)

        plt.legend(('0.5', '5.0', '25.0', '50.0', '75.0', '95.0', '99.5'))
    plt.savefig(fs.join('pictures','Квантили распределения %s (по слоям)'%title_name))
    pass

"""
Построение на всех батчах послойно
"""
def percentile_layers_batch(all_updates, layers_bound, update_list, q_list, layers_name):
    num_layers = len(layers_bound) - 1
    num_batch = len(all_updates[0])

    for layer in range(num_layers):
        plt.figure(figsize=(10*len(all_updates), 10*len(all_updates)))
        perc_distribution = np.empty((4, len(q_list), num_batch))
        for num_updates in range(len(all_updates)):
            for q in range(len(q_list)):
                for batch in range(num_batch):
                    perc_distribution[num_updates, q, batch] = percentile(all_updates[num_updates, batch,
                                     layers_bound[layer]:layers_bound[layer + 1]], q_list[q])

        for num_updates in range(len(all_updates)):
            plt.subplot(len(all_updates), 1, num_updates + 1)
            for q in range(len(q_list)):
                if q_list[q] == 50.0:
                    plt.plot(np.arange(len(perc_distribution[num_updates, q, :])), perc_distribution[num_updates, q, :], lw = 3)
                else:
                    plt.plot(np.arange(len(perc_distribution[num_updates, q, :])), perc_distribution[num_updates, q, :])

            plt.title('Распределение %s. layer %s; layer_num %d; num_params = %d'%(update_list[num_updates],
                      layers_name[layer], layer, (layers_bound[layer + 1] - layers_bound[layer])))
            plt.xlabel('батч')
            plt.ylabel('значение')
            plt.grid(True)
            plt.legend(('0.5', '5.0', '25.0', '50.0', '75.0', '95.0', '99.5'))
        plt.savefig(fs.join('pictures','Распределение layer_num %d.png'%layer))
    pass
#%%
"""
Обратная задача, по параметрам оптимизатора и весам (updates)
найти градиенты
"""
#SGD Optimizer
@njit([float32[:, :](float32[:, :], float32, float32, float32)], parallel = True)
def computation_grads_SGD(velocity, lr_init, momentum, decay):
    grads = np.empty(velocity.shape, dtype = np.float32)
    v = np.concatenate((np.zeros((1, velocity.shape[1]), dtype = np.float32), velocity), axis = 0)
    for i in prange(grads.shape[0]):
        if decay > 0:
            lr = lr_init * (1. / (1. + decay * (i)))
        grads[i, :] = (momentum*v[i, :] - v[i + 1, :])/lr
    return grads

#RMSprop Optimizer
@njit([float32[:, :](float32[:, :], float32[:, :], float32, float32, float32)], parallel = True)
def computation_grads_RMSprop(new_p, accumulators, lr_init, rho, decay):
    grads = np.empty(accumulators.shape, dtype = np.float32)
    new_p = np.concatenate((np.zeros((1, new_p.shape[1]), dtype = np.float32), new_p), axis = 0)
    for i in prange(grads.shape[0]):
        if decay > 0:
            lr = lr_init * (1. / (1. + decay * (i)))
        grads[i, :] = ((new_p[i, :] - new_p[i + 1, :])*(np.sqrt(accumulators[i, :] + 1e-07))/lr)
    return grads

#Adagrad Optimizer
@njit([float32[:, :](float32[:, :], float32[:, :], float32, float32)], parallel = True)
def computation_grads_Adagrad(new_p, accumulators, lr_init, decay):
    grads = np.empty(accumulators.shape, dtype = np.float32)
    new_p = np.concatenate((np.zeros((1, new_p.shape[1]), dtype = np.float32), new_p), axis = 0)
    for i in prange(grads.shape[0]):
        if decay > 0:
            lr = lr_init * (1. / (1. + decay * (i)))
        grads[i, :] = ((new_p[i, :] - new_p[i + 1, :])*(np.sqrt(accumulators[i, :] + 1e-07))/lr)
    return grads

#Adadelta Optimizer
@njit([float32[:, :](float32[:, :], float32[:, :], float32[:, :], float32, float32)], parallel = True)
def computation_grads_Adadelta(new_p, accumulators, delta_accumulators, lr_init, decay):
    grads = np.empty(accumulators.shape, dtype = np.float32)
    new_p = np.concatenate((np.zeros((1, new_p.shape[1]), dtype = np.float32), new_p), axis = 0)
    delta_accumulators = np.concatenate((np.zeros((1, new_p.shape[1]), dtype = np.float32), delta_accumulators), axis = 0)
    for i in prange(grads.shape[0]):
        if decay > 0:
            lr = lr_init * (1. / (1. + decay * (i)))
        grads[i, :] = ((new_p[i, :] - new_p[i + 1, :])*(np.sqrt(accumulators[i, :] + 1e-07))/(lr*np.sqrt(delta_accumulators[i, :] + 1e-07)))
    return grads

#Adam Optimizer
@njit([float32[:, :](float32[:, :], float32)], parallel = True)
def computation_grads_Adam(m, beta_1):
    grads = np.empty(m.shape, dtype = np.float32)
    m = np.concatenate((np.zeros((1, m.shape[1]), dtype = np.float32), m), axis = 0)
    for i in prange(grads.shape[0]):
        grads[i, :] = (m[i + 1, :] - beta_1*m[i, :])/(1 - beta_1)
    return grads

#Adamax Optimizer
@njit([float32[:, :](float32[:, :], float32)], parallel = True)
def computation_grads_Adamax(m, beta_1):
    grads = np.empty(m.shape, dtype = np.float32)
    m = np.concatenate((np.zeros((1, m.shape[1]), dtype = np.float32), m), axis = 0)
    for i in prange(grads.shape[0]):
        grads[i, :] = (m[i + 1, :] - beta_1*m[i, :])/(1 - beta_1)
    return grads

#Nadam Optimizer
@njit([float32[:, :](float32[:, :], float32)], parallel = True)
def computation_grads_Nadam(m, beta_1):
    grads = np.empty(m.shape, dtype = np.float32)
    m = np.concatenate((np.zeros((1, m.shape[1]), dtype = np.float32), m), axis = 0)
    for i in prange(grads.shape[0]):
        grads[i, :] = (m[i + 1, :] - beta_1*m[i, :])/(1 - beta_1)
    return grads
#%%
"""
Краткая инструкция по вычислению на float16:

1) Вызываем функцию preparation_f16, параметр eps по умолчанию равен 1e-4,
    но он может вариороваться, например 1e-3. Если поставить маленькое значение,
    например 1e-7, то возможен вылет в NaN loss функции.
2) Заменяем слой BatchNormalization в нейросети на BatchNormalizationF16, работа с
    ним ничем не меняется

3) Переводим все данные в float16 через .astype(np.float16)

Замечания: смена типа позволяет увеличить размер батча вдвое
"""

def preparation_f16(eps = 1e-4):
    K.set_floatx('float16')
    K.set_epsilon(eps)

###############################################################################
#custom initializers to force float32
class Ones32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(1, shape=shape, dtype='float32')

class Zeros32(Initializer):
    def __call__(self, shape, dtype=None):
        return K.constant(0, shape=shape, dtype='float32')

class BatchNormalizationF16(Layer):

    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(BatchNormalizationF16, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = (
            initializers.get(moving_variance_initializer))
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)
            else:
                return tf.nn.batch_normalization(#K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    #axis=self.axis,
                    self.epsilon)#epsilon=self.epsilon)

        # If the learning phase is *static* and set to inference:
        if training in {0, False}:
            return normalize_inference()

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = _regular_normalize_batch_in_training(#K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer':
                initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer':
                initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalizationF16, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
#%%
"""
CNN Plot
"""
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def generate_pattern(model, layer_name, filter_index, size, num_chanel = 1):
    # Build a loss function that maximizes the activation
    # of the nth filter of the layer considered.
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # Compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, model.input)[0]

    # Normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # This function returns the loss and grads given the input picture
    iterate = K.function([model.input], [loss, grads])

    # We start from a gray image with some noise
    input_img_data = np.random.random((1, size[0], size[1], num_chanel)) * 20 + 128.

    # Run gradient ascent for 40 steps
    step = 1.
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    img = input_img_data[0]
    return deprocess_image(img)