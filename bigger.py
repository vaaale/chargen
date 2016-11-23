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
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
    seq_in = raw_text[i:i + seq_length]
    seq_out = raw_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# define the LSTM model
model = Sequential()
model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(256, return_sequences=False))

model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
filename = "weights.hdf5"
if os.path.isfile(filename):
    model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


class SampleModelCallbac(Callback):
    def __init__(self, int_to_char, model):
        super().__init__()
        self.int_to_char = int_to_char
        self.model = model

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        start = numpy.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
        print("Seed:")
        print("\"", ''.join([self.int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(300):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = self.int_to_char[index]
            seq_in = [self.int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

sample_model = SampleModelCallbac(int_to_char, model)
callbacks_list = [checkpoint, sample_model]
# fit the model
history = model.fit(X, y, nb_epoch=40, batch_size=64, callbacks=callbacks_list)

pickle.dump(open('hist.pkl', 'wb'), history)


