#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we want to build on what we learnt in recurrent-neural-nets.py and use 
longer input sequences i.e. starting with x0 = t-2,t-1,t0 terms, instead of 
x0 = t0 as we did before. This may require an encoder-decoder network me reckons...
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


#%% Generate sequence data. We want to generate it in such a way that we have 3 initial
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


#%% Encoder-decoder seq2seq wrapper ((pg. 544-545 in HOML))
# also use info. from https://www.angioi.com/time-series-encoder-decoder-tensorflow/, 
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
encoder_inputs = keras.layers.Input(shape = [None, 1])
decoder_inputs = keras.layers.Input(shape = [None, 1])

# encoder 
encoder = LSTM(units=20, return_state = True)
encoder_ouputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder
decoder = LSTM(units=20, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(inputs = decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(units=1, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# define and train encoder-decoder model
model_ED = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)
model_ED.compile(loss='mse', optimizer=Adam(lr=0.01))
encoder_train = X_train[:,0:3]
decoder_train = np.concatenate((np.zeros((X_train.shape[0],1,1)), X_train[:,3:]), axis = 1)
model_ED.fit([encoder_train, decoder_train], Y_train, epochs = 10, batch_size = 32)

# prediction on training sequence - treated as a single-step ahead prediction
y_train_ed = model_ED.predict([encoder_train, decoder_train])
y_train_ed = y_train_ed[:,:,0]
RMSE_tr_ed = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_ed.ravel()))
print(f'Training score across all timesteps as an SSA: {RMSE_tr_ed}')

# visualise
fig, axs = plt.subplots(1, 3, sharey = True)
for i in range(3):
    axs[i].plot(np.linspace(0,51,52), X_train[i].ravel(), 'x-', label = 'input')
    axs[i].plot(np.linspace(3,52,50), Y_train[i].ravel(), 'o:', label = 'target')
    axs[i].plot(np.linspace(3,52,50), y_train_ed[i].ravel(), 'd', label = 'ssa prediction')
    axs[i].set_xlabel('t')
    axs[i].legend()
axs[0].set_ylim([-1, 1])
axs[0].set_ylabel('x(t)')


#%% Setup multistep-ahead prediction ED model and iterator function
# setup encoder model to output states given input sequence using trained encoder above
encoder_model = keras.Model(encoder_inputs, encoder_states) 

# create Tensorflow decoder objects for multi-step ahead prediction based on trained decoder above 
decoder_state_input_h = keras.layers.Input(shape=(20,))
decoder_state_input_c = keras.layers.Input(shape=(20,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c] 
decoder_outputs = decoder_dense(decoder_outputs)

# create decoder model from decoder objects
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs] + decoder_states)


def msa_ED_iterator(X, Y, num_steps, encoder_model, decoder_model):
    """
    Multi-step ahead encoder-decoder iterator. 
    
    Arguments:
        X - inputs
        Y - outputs
        n_steps - number of timesteps
        encoder_model - encoder
        decoder_model - decoder
             
    Returns:
        Y - updated output array
    """
    X = X.copy()
    Y = Y.copy()
    states = encoder_model.predict(X)
    decoder_input = np.zeros((X.shape[0], 1, 1))
    for i in range(num_steps):
        output, h, c = decoder_model.predict([decoder_input] + states)
        Y[:, i, np.newaxis] = output[:, -1, np.newaxis]
        decoder_input = Y[:, i, np.newaxis]
        states = [h, c]   
    return Y
        
    
#%% Perform msa prediction on abritrary time series using ED net
# create series and predict
new_series = generate_time_series(1, n_steps + 3)
X_new = new_series[:, :n_steps + 2]
Y_new = np.empty((1, n_steps, 1))
for step_ahead in range(3, 3 + 1):
    Y_new[:, :, step_ahead - 3] = new_series[:, step_ahead:step_ahead + n_steps, 0]
encoder_newin = X_new[:,0:3]

# train and validation scores across all timesteps with the iterator
y_tr_msa_ed = msa_ED_iterator(X_train[:, 0:3], np.zeros(Y_train.shape), 
                              n_steps, encoder_model, decoder_model)
RMSE_tr_msa_ed = np.sqrt(mean_squared_error(Y_train.ravel(), y_tr_msa_ed.ravel()))
print(f'Training score across all timesteps using encoder-decoder LSTM: {RMSE_tr_msa_ed}')
y_val_msa_ed = msa_ED_iterator(X_valid[:, 0:3], np.zeros(Y_valid.shape), 
                               n_steps, encoder_model, decoder_model)
y_val_msa_ed = y_val_msa_ed[:,:,0]
RMSE_val_msa_ed = np.sqrt(mean_squared_error(Y_valid.ravel(), y_val_msa_ed.ravel()))
print(f'Validation score across all timesteps using encoder-decoder LSTM: {RMSE_val_msa_ed}')
    
# viz
y_new_msa_ed = msa_ED_iterator(encoder_newin, np.zeros((1, n_steps, 1)), n_steps, encoder_model, decoder_model)
y_new_msa_stl = msa_iterator(encoder_newin[:, -1, np.newaxis], np.zeros((1, n_steps, 1)), n_steps, model_LSTM, 'continuous') # taken from rnn script
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), y_new_msa_ed.ravel(), 'd-', label = 'prediction: ED')
ax.plot(np.linspace(1,50,50), y_new_msa_stl.ravel(), '*-', label = 'prediction: stateless LSTM')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()  
    
# FINDINGS - Generally, the encoder-decoder, gives better early time performance (qualitatively) 
# compared to the iterative LSTM. This seems reasonable given that we are encoding with three terms
# whereas the iterative LSTM starts from just one term. s


