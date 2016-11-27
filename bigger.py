# Larger LSTM Network to Generate Text for Alice in Wonderland
import os
import pickle
from keras.callbacks import ModelCheckpoint, RemoteMonitor
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
import dataset as ds
from sample_model import SampleModelCallback


batches_per_epoch = 1500
batch_size = 64
samples_per_epoch = batches_per_epoch * batch_size
seq_length = 100
vocab_length = ds.vocab_length()
generator = ds.dataset(batch_size, seq_length)


def large_model():
    # define the LSTM model
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(128, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(vocab_length, activation='softmax'))
    return model


def small_model():
    model = Sequential()
    model.add(LSTM(256, input_shape=(seq_length, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_length, activation='softmax'))
    return model

model = large_model()

filename = "weights.hdf5"
if os.path.isfile(filename):
    print('Resuming training')
    model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
# define the checkpoint
filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
sample = SampleModelCallback(seq_length)
remote = RemoteMonitor(root='http://localhost:9000')

callbacks_list = [checkpoint, sample, remote]
# fit the model
history = model.fit_generator(generator=generator, nb_epoch=200, samples_per_epoch=samples_per_epoch, callbacks=callbacks_list)



