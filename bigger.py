# Larger LSTM Network to Generate Text for Alice in Wonderland
import os
import sys

import pickle
import numpy
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.utils import np_utils

# load ascii text and covert to lowercase
from sample_model import SampleModelCallbac
from dataset import sample_data

filename = "politikk.txt"
raw_text = open(filename).read()
raw_text = raw_text.lower()
# create mapping of unique chars to integers
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 40


# dataX = []
# dataY = []


def myGenerator(filename):
    raw_text = open(filename).read()
    raw_text = raw_text.lower()
    while 1:
        for i in range(0, n_chars - seq_length, 1):
            seq_in = raw_text[i:i + seq_length]
            seq_out = raw_text[i + seq_length]
            dataX = [char_to_int[char] for char in seq_in]
            dataY = [char_to_int[seq_out]]
            n_patterns = len(dataX)
            #print("Total Patterns: ", n_patterns)
            # reshape X to be [samples, time steps, features]
            #X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
            X = numpy.reshape(dataX, (1, 1, seq_length))
            # normalize
            X = X / float(n_vocab)
            # one hot encode the output variable
            y = np_utils.to_categorical(dataY, n_vocab)
            yield X, y


# define the LSTM model
model = Sequential()
#model.add(LSTM(512, input_shape=(len(chars), 512), return_sequences=True))
#model.add(LSTM(len(chars), 512, return_sequences=True))
model.add(LSTM(512, input_dim=(seq_length), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256, return_sequences=False))

model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

model_filename = "weights.hdf5"
if os.path.isfile(model_filename):
    model.load_weights(model_filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

sample_model = SampleModelCallbac(int_to_char, model, sample_data(), n_vocab)
callbacks_list = [checkpoint, sample_model]
# fit the model
# history = model.fit(X, y, nb_epoch=40, batch_size=64, callbacks=callbacks_list)
model.fit_generator(myGenerator(filename), samples_per_epoch=60000, nb_epoch=100, verbose=2, callbacks=callbacks_list, nb_worker=1)


