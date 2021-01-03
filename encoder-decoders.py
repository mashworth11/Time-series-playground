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
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
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


#%% Encoder-decoder seq2seq wrapper (Hands on Machine Learning pg. 545)
# also use info. from https://www.angioi.com/time-series-encoder-decoder-tensorflow/, 
# https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html
encoder_inputs = keras.layers.Input(shape = [None, 1])
decoder_inputs = keras.layers.Input(shape = [None, 1])

# encoder 
encoder = LSTM(20, return_state = True)
encoder_ouputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

# decoder
decoder = LSTM(20, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder(inputs = decoder_inputs, initial_state = encoder_states)
decoder_dense = Dense(1, activation='linear')
decoder_outputs = decoder_dense(decoder_outputs)

# define and train encoder-decoder model
model_ED = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = decoder_outputs)
model_ED.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
encoder_train = X_train[:,0:3]
decoder_train = np.concatenate((np.zeros((X_train.shape[0],1,1)), X_train[:,3:]), axis = 1)
model_ED.fit([encoder_train, decoder_train], Y_train, epochs = 10)

# prediction on training sequence - treated as a singlestep-ahead prediction
y_train_ed_s2s = model_ED.predict([encoder_train, decoder_train])
y_train_ed_s2s = y_train_ed_s2s[:,:,0]
RMSE_ed_s2s = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_ed_s2s.ravel()))
print(f'Training score across all timesteps: {RMSE_ed_s2s}')


#%% Setup multistep-ahead prediction ED model and iterator function
# setup encoder model to output states given input sequence using trained encoder above
encoder_model = keras.Model(encoder_inputs, encoder_states) 

# create Tensorflow decoder objects for multistep-ahead prediction based on trained decoder above 
decoder_state_input_h = keras.layers.Input(shape=(20,))
decoder_state_input_c = keras.layers.Input(shape=(20,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder(decoder_inputs, initial_state = decoder_states_inputs)
decoder_states = [state_h, state_c] 
decoder_outputs = decoder_dense(decoder_outputs)

# create decoder model from decoder objects
decoder_model = keras.Model([decoder_inputs] + decoder_states_inputs,
                            [decoder_outputs] + decoder_states)


X_new_msa = X_new[:, 0, np.newaxis]
def msa_ED_iterator(X, Y, num_steps, encoder_model, decoder_model):
    """
    Multistep-ahead encoder-decoder iterator. 
    
    Arguments:
        X - inputs
        Y - outputs
        encoder_model - encoder
        decoder_model - decoder
             
    Returns:
        Y - updated output array
    """
    states = encoder_model.predict(X)
    decoder_input = np.zeros((X.shape[0], 1, 1))
    for i in range(num_steps):
        output, h, c = decoder_model.predict([decoder_input] + states)
        Y[:, i] = output[:, -1, np.newaxis]
        decoder_input = Y[:, i]
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

y_msa_ED = np.zeros((1, n_steps, 1))
y_msa_ED = msa_ED_iterator(encoder_newin, y_msa_ED, n_steps, encoder_model, decoder_model)
y_msa_LSTM = np.zeros((1, n_steps, 1))
y_msa_LSTM = msa_iterator(encoder_newin[:, -1, np.newaxis], y_msa_LSTM, n_steps, model_LSTM, 'continuous') # taken from rnn script
    
# viz
fig, ax = plt.subplots()
ax.plot(np.linspace(1,50,50), Y_new.ravel(), 'o-', label = 'target')
ax.plot(np.linspace(1,50,50), y_msa_ED.ravel(), 'd-', label = 'prediction-ED')
ax.plot(np.linspace(1,50,50), y_msa_LSTM.ravel(), '*-', label = 'prediction-LSTM')
ax.set_xlabel('t')
ax.set_ylim([-1, 1])
ax.set_ylabel('x(t)')
ax.legend()  
    



