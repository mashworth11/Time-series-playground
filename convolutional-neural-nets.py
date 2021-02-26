#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we want to implement convolutional neural net architectures to develop our
intuition for how these critters work on time-seres data 
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.optimizers import Adam

#%% Generate sequence data. We want to generate in such a way that we have 3 initial
# steps being used to make the prediction from time t=0 to t+1. 
def generate_time_series(batch_size, n_steps):
    # batch_size = number of time series
    freq1, freq2, offset1, offset2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5*np.sin((time - offset1) * (freq1 * 10 + 10))
    series += 0.2*np.sin((time - offset2) * (freq2 * 20 + 20))
    series += 0.1*(np.random.rand(batch_size, n_steps) - 0.5)
    return series[..., np.newaxis]
    
# generate train, validation and test sets
n_steps = 50
series = generate_time_series(10000, n_steps + 3)
X_train = series[:7000, :n_steps + 2]
X_valid = series[7000:9000, :n_steps + 2]
X_test = series[9000:, :n_steps + 2]

Y = np.empty((10000, n_steps, 1))
for step_ahead in range(3, 3 + 1):
    Y[:, :, step_ahead - 3] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]

# visualise
fig, axs = plt.subplots(1, 3, sharey = True)
for i in range(3):
    axs[i].plot(np.linspace(0,51,52), X_train[i].ravel(), 'x-', label = 'input')
    axs[i].plot(np.linspace(3,52,50), Y_train[i].ravel(), 'o', label = 'target')
    axs[i].set_xlabel('t')
    axs[i].legend()
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')


#%% Define iterator
def msa_CNN_iterator(X, Y, n_steps, model):
    """
    Multistep-ahead iterator for the convolutional neural network. 
    
    Arguments:
        X - inputs
        Y - outputs
        n_steps - number of timesteps
        model - model, in this case a CNN
             
    Returns:
        Y - updated output array
    """
    X = X.copy()
    Y = Y.copy()
    for i in range(n_steps):
        prediction = model.predict(X)
        Y[:, i, np.newaxis] = prediction[:, -1, np.newaxis]
        X = np.concatenate((X[:,1:3], prediction), axis = 1) 
    return Y


#%% Basic CNN seq2seq model 
model_CNN = Sequential([Conv1D(filters=20, kernel_size=3, activation='relu', 
                               input_shape=[None, 1]),
                        TimeDistributed(Dense(1, activation = 'linear'))])
model_CNN.compile(loss = 'mse', optimizer =  Adam(lr = 0.1))
model_CNN.fit(X_train, Y_train, epochs = 10, batch_size = 32)

# prediction on training and validation sequences - treated as a single-step ahead prediction
y_train_cnn = model_CNN.predict(X_train)
y_train_cnn = y_train_cnn[:,:,0]
RMSE_tr_cnn = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_cnn.ravel()))
print(f'Training score across all timesteps as an SSA: {RMSE_tr_cnn}')

y_valid_cnn = model_CNN.predict(X_valid)
y_valid_cnn = y_valid_cnn[:,:,0]
RMSE_va_cnn = np.sqrt(mean_squared_error(Y_valid.ravel(), y_valid_cnn.ravel()))
print(f'Validation score across all timesteps as an SSA: {RMSE_va_cnn}')

# visualise
fig, axs = plt.subplots(1, 3, sharey = True)
for i in range(3):
    axs[i].plot(np.linspace(0,51,52), X_train[i].ravel(), 'x-', label = 'input')
    axs[i].plot(np.linspace(3,52,50), Y_train[i].ravel(), 'o:', label = 'target')
    axs[i].plot(np.linspace(3,52,50), y_train_cnn[i].ravel(), 'd', label = 'ssa prediction')
    axs[i].set_xlabel('t')
    axs[i].legend()
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')


#%% multi-step ahead prediction
y_tr_msa_cnn = msa_CNN_iterator(X_train[:, 0:3], np.zeros(Y_train.shape), n_steps, model_CNN)
RMSE_tr_msa_cnn = np.sqrt(mean_squared_error(Y_train.ravel(), y_tr_msa_cnn.ravel()))
print(f'Training score across all timesteps as an MSA: {RMSE_tr_msa_cnn}')

y_val_msa_cnn = msa_CNN_iterator(X_valid[:, 0:3], np.zeros(Y_valid.shape), n_steps, model_CNN)
RMSE_val_msa_cnn = np.sqrt(mean_squared_error(Y_valid.ravel(), y_val_msa_cnn.ravel()))
print(f'Validation score across all timesteps as an MSA: {RMSE_val_msa_cnn}')

# visualise
fig, axs = plt.subplots(1, 3, sharey = True)
for i in range(3):
    axs[i].plot(np.linspace(0,51,52), X_train[i].ravel(), 'x-', label = 'input')
    axs[i].plot(np.linspace(3,52,50), Y_train[i].ravel(), 'o:', label = 'target')
    axs[i].plot(np.linspace(3,52,50), y_tr_msa_cnn[i].ravel(), '<', label = 'msa prediction')
    axs[i].set_xlabel('t')
    axs[i].legend()
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')


#%% Dilating CNN seq2seq model (HOML pgs. 521-523)
X_train_ = X_train[:,2:,:]
X_valid_ = X_valid[:,2:,:]

model_DCNN = Sequential()
model_DCNN.add(keras.layers.InputLayer(input_shape=[None,1]))
for rate in (1, 2, 4, 8)*2:
    model_DCNN.add(Conv1D(filters=20, kernel_size=2, padding='causal', activation='relu', 
                          dilation_rate=rate))
#model_DCNN.add(Conv1D(filters=1, kernel_size=1))
model_DCNN.add(TimeDistributed(Dense(1, activation = 'linear')))
model_DCNN.compile(loss='mse', optimizer=Adam(lr=0.01))
model_DCNN.fit(X_train_, Y_train, epochs = 20, batch_size = 32)

# prediction on training and validation sequences - treated as a single-step ahead prediction
y_train_dcnn = model_DCNN.predict(X_train_)
y_train_dcnn = y_train_dcnn[:,:,0]
RMSE_tr_dcnn = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_dcnn.ravel()))
print(f'Training score across all timesteps as an SSA: {RMSE_tr_dcnn}')

y_valid_dcnn = model_DCNN.predict(X_valid_)
y_valid_dcnn = y_valid_dcnn[:,:,0]
RMSE_va_dcnn = np.sqrt(mean_squared_error(Y_valid.ravel(), y_valid_dcnn.ravel()))
print(f'Validation score across all timesteps as an SSA: {RMSE_va_dcnn}')













 