#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we want to build on what we learnt in time-series-modelling.py and use 
longer input sequences i.e. starting with x0 = t-2,t-1,t0 terms, instead of 
x0 = t0 as we did before. This may require an encoder-decoder network me reckons...
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import LSTMCell
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa


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
# also use info. from https://www.angioi.com/time-series-encoder-decoder-tensorflow/
encoder_inputs = keras.layers.Input(shape = [None, 1])
decoder_inputs = keras.layers.Input(shape = [None, 1])

encoder = LSTM(20, return_state = True)
encoder_ouputs, state_h, state_c = encoder(encoder_inputs)
encoder_state = [state_h, state_c]

sampler = tfa.seq2seq.sampler.TrainingSampler()
decoder_cell = LSTMCell(20) # contains the calculation logic for one-time step (as oppose to the LSTM objcet, which contains the 
                            # logic for the recurrent calculations)
output_layer = Dense(1, activation = 'linear')
decoder = tfa.seq2seq.basic_decoder.BasicDecoder(decoder_cell, sampler, output_layer = output_layer)
final_outputs, _, _ = decoder(inputs = decoder_inputs, initial_state = encoder_state)
output = final_outputs.rnn_output

model_ED = keras.Model(inputs = [encoder_inputs, decoder_inputs], outputs = output)
model_ED.compile(loss = 'mse', optimizer = Adam(lr = 0.01))
encoder_train = X_train[:,0:3]
decoder_train = np.concatenate((np.zeros((X_train.shape[0],1,1)), X_train[:,3:]), axis = 1)
model_ED.fit([encoder_train, decoder_train], Y_train, epochs = 10)


#%% Inference on training sequence using ground truths from previous timesteps as inputs to the decoder
y_train_ed_s2s = model_ED.predict([encoder_train, decoder_train])
y_train_ed_s2s = y_train_ed_s2s[:,:,0]
RMSE_ed_s2s = np.sqrt(mean_squared_error(Y_train.ravel(), y_train_ed_s2s.ravel()))
print(f'Training score across all timesteps: {RMSE_ed_s2s}')

# Inference on training sequence using predictions from previous timesteps as inputs to the decoder







