#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python script to explore different approaches to time-series modelling
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


#%% Generate sequence data a la Hands on Machine Learning pg. 504
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
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1:, 0]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1:, 0]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1:, 0]

# visualise
fig, axs = plt.subplots(1, 3, sharey = True)
for i in range(3):
    axs[i].plot(np.linspace(0,50,51), series[i].ravel())
    axs[i].set_xlabel('t')
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')


#%% Pt. I: Simple RNN - seq2vec
# Use 50 steps to predict 1 step into the future, trained and used as a seq2vec
model = Sequential([
        SimpleRNN(20, return_sequences = True, input_shape=[None, 1]),
        SimpleRNN(20),
        Dense(1)])
model.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
model.fit(X_train, y_train, epochs = 10)
y_val_s2v = model.predict(X_valid)
RMSE_s2v = np.sqrt(mean_squared_error(y_valid, y_val_s2v)) # mse across batches for the next value in the sequence
print(f'Validation score: {RMSE_s2v}')


#%% Generate time series data we can compare on for a seq2seq case
new_series = generate_time_series(1, n_steps + 1)
X_new = new_series[:, :n_steps]
Y_new = np.empty((1, n_steps, 1))
for step_ahead in range(1, 1 + 1):
    Y_new[:, :, step_ahead - 1] = new_series[:, step_ahead:step_ahead + n_steps, 0]

# Create seq2seq time series data
Y = np.empty((10000, n_steps, 1))
for step_ahead in range(1, 1 + 1):
    Y[:, :, step_ahead - 1] = series[:, step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]


#%% Pt. II: Simple RNN - seq2seq
# Predict at each step, trained and used as a seq2seq
model_SRRN = Sequential([
             SimpleRNN(20, return_sequences = True, input_shape=[None, 1]),
             SimpleRNN(20, return_sequences = True), # return sequence states for every timestep in the sequence
             TimeDistributed(Dense(1))]) # apply dense to every timestep in the sequence
model_SRRN.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
model_SRRN.fit(X_train, Y_train, epochs = 10)
y_val_s2s = model_SRRN.predict(X_valid)
y_val_s2s = y_val_s2s[:,:,0]
RMSE_s2s = np.sqrt(mean_squared_error(y_valid, y_val_s2s[:,-1]))
print(f'Validation score for last timestep: {RMSE_s2s}')
RMSE_s2s = np.sqrt(mean_squared_error(Y_valid.ravel(), y_val_s2s.ravel()))
print(f'Validation score across all timesteps: {RMSE_s2s}')

# viz
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), model_SRRN.predict(X_new).ravel(), '*-', label = 'prediction')
axs[i].set_xlabel('t')
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')
ax.legend()


#%% Pt. III: LSTM - seq2seq
# Predict at each step, trained and used as a seq2seq. Teacher forcing is implicit.
model_LSTM = Sequential([
             LSTM(20, return_sequences = True, input_shape=[None, 1]),
             LSTM(20, return_sequences = True),
             TimeDistributed(Dense(1, activation = 'linear'))])
model_LSTM.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
model_LSTM.fit(X_train, Y_train, epochs = 10)

# use ground truths from previous timesteps as inputs
y_train_lstm_s2s = model_LSTM.predict(X_train)
y_train_lstm_s2s = y_train_lstm_s2s[:,:,0]
RMSE_lstm_s2s = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_lstm_s2s.ravel()))
print(f'Training score across all timesteps: {RMSE_lstm_s2s}')
y_val_lstm_s2s = model_LSTM.predict(X_valid)
y_val_lstm_s2s = y_val_lstm_s2s[:,:,0]
RMSE_lstm_s2s = np.sqrt(mean_squared_error(Y_valid.ravel(), y_val_lstm_s2s.ravel()))
print(f'Validation score across all timesteps: {RMSE_lstm_s2s}')

# viz
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), model_LSTM.predict(X_new).ravel(), '*-', label = 'prediction')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()
ax.set_title('Predictions with groundtruth inputs')


#%% Pt. IV: LSTM - Iterative seq., i.e. using predictions from previous timesteps as inputs
# Essentially 2 ways to do this according to
# https://stackoverflow.com/questions/38714959/understanding-keras-lstms/50235563#50235563
# i.e. Repeating the input vector or iteratively

# (1) Repeating the input vector
X_valid_msa = X_valid[:, 0, np.newaxis]
X_valid_msa = np.repeat(X_valid_msa, 50, axis = 1)
y_msa_rep = model_LSTM.predict(X_valid_msa)
RMSE_msa1 = np.sqrt(mean_squared_error(Y_valid.ravel(), y_msa_rep.ravel()))
print(f'Validation score: {RMSE_msa1_s2s}')

# Viz
X_new_msa = X_new[:, 0, np.newaxis]
X_new_msa = np.repeat(X_new_msa, 50, axis = 1)
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), model_LSTM.predict(X_new_msa).ravel(), '*-', label = 'prediction')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()

# DOESN'T SEEM TO WORK TOO WELL (unless I'm doing something wrong, which could also be a possibility)


# (2) Iterative approaches: stateless vs stateful
# Looper
X_new_msa = X_new[:, 0, np.newaxis]
def msa_iterator(X, Y, n_steps, model, option):
    """
    Multistep-ahead iterator. 
    
    Arguments:
        X - inputs
        Y - outputs
        model - model e.g. LSTM
        option - 'continuous' means sequential without replacement, whilst 'sequential' 
                 means continuous with replacement. 
             
    Returns:
        Y - updated output array
    """
    if option == 'continuous':
        for i in range(n_steps):
            prediction = model.predict(X)
            Y[:, i] = prediction[:, -1, np.newaxis]
            X = np.concatenate([X, Y[:, i, np.newaxis]], axis = 1)
    elif option == 'sequential':
        for i in range(n_steps):
            prediction = model.predict(X)
            Y[:, i] = prediction[:,-1, np.newaxis]
            X = Y[:, i, np.newaxis]
    return Y


### Stateless
y_msa_sless = np.zeros((1, n_steps, 1))
y_msa_sless = msa_iterator(X_new_msa, y_msa_sless, n_steps, model_LSTM, 'continuous')

# stateless viz
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), y_msa_sless.ravel(), '*-', label = 'prediction')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()


### Stateful 
stateless_weights = model_LSTM.get_weights()
model_LSTM_II = Sequential([
                LSTM(20, return_sequences = True, stateful = True, batch_input_shape=[1, None, 1]),
                LSTM(20, return_sequences = True, stateful = True,),
                TimeDistributed(Dense(1, activation = 'linear'))])
model_LSTM_II.set_weights(stateless_weights)
y_msa_sful = np.zeros((1, n_steps, 1))
y_msa_sful = msa_iterator(X_new_msa, y_msa_sful, n_steps, model_LSTM_II, 'sequential')

# stateful viz
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), y_msa_sless.ravel(), 'd-', label = 'prediction - stateless continuous')
ax.plot(np.linspace(1,50,50), y_msa_sful.ravel(), '*-', label = 'prediction - stateful')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()


# FINDINGS - so we found that continuous LSTM is equivalent to a stateful sequential LSTM! This is useful!
# A good next experiment would be to see if we could get better performance by using three terms for an
# initial prediction instead of one. That is exactly what we will do next.







