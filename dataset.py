import os
import pickle

import numpy as numpy
from keras.utils import np_utils
import re

# load ascii text and covert to lowercase
filename = "allcode.txt"
meta_file = "meta.pkl"
if not os.path.isfile(meta_file):
    _raw_text = open(filename, encoding='UTF-8').read()
    #_raw_text = re.sub('[^\sa-zA-Z0-9æøå,.:?–]+', '', _raw_text)
    _raw_text = _raw_text.lower()
    # create mapping of unique chars to integers
    _chars = sorted(list(set(_raw_text)))
    _char_to_int = dict((c, i) for i, c in enumerate(_chars))
    _int_to_char = dict((i, c) for i, c in enumerate(_chars))
    # summarize the loaded data
    _meta = {
        'size': len(_raw_text),
        'vocab_size': len(_chars),
        'char_to_int': _char_to_int,
        'int_to_char': _int_to_char,
        'vocab': _chars
    }
    with open('clean' + filename, 'w', encoding='UTF-8') as out:
        out.write(_raw_text)
    pickle.dump(_meta, open(meta_file, 'wb'))
else:
    _meta = pickle.load(open(meta_file, 'rb'))

filename = 'clean' + filename

n_chars = _meta['size']
n_vocab = _meta['vocab_size']
_char_to_int = _meta['char_to_int']
_int_to_char = _meta['int_to_char']
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)
print(_meta['vocab'])


def vocab_length():
    return n_vocab


def meta():
    return _meta


def dataset(batchsize=64, seq_length=40):
    chunk_size = seq_length + batchsize
    num_chunks = int(_meta['size'] / chunk_size)
    # re.sub('[^\sa-zA-Z0-9æøå,.:?\–]+', '', ' '.join(segs[5:])
    while True:
        with open(filename, encoding='UTF-8') as f:
            for chunk in range(0, num_chunks, 1):
                # Read data for one batch
                raw_text = f.read(chunk_size)
                raw_text = raw_text.lower()

                # prepare the dataset of input to output pairs encoded as integers
                dataX = []
                dataY = []
                for i in range(0, chunk_size - seq_length, 1):
                    seq_in = raw_text[i:i + seq_length]
                    seq_out = raw_text[i + seq_length]
                    dataX.append([_char_to_int[char] for char in seq_in])
                    dataY.append(_char_to_int[seq_out])

                n_patterns = len(dataX)
                # reshape X to be [samples, time steps, features]
                X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
                # normalize
                X = X / float(n_vocab)
                # one hot encode the output variable
                y = np_utils.to_categorical(dataY, nb_classes=n_vocab)

                yield (X, y)


def sample(batchsize=64, seq_length=40):
    chunk_size = seq_length + batchsize
    num_chunks = int(_meta['size'] / chunk_size)

    while True:
        with open(filename, encoding='UTF-8') as f:
            for chunk in range(0, num_chunks, 1):
                # Read data for one batch
                raw_text = f.read(chunk_size)
                raw_text = raw_text.lower()

                # prepare the dataset of input to output pairs encoded as integers
                dataX = []
                dataY = []
                for i in range(0, chunk_size - seq_length, 1):
                    seq_in = raw_text[i:i + seq_length]
                    seq_out = raw_text[i + seq_length]
                    dataX.append([_char_to_int[char] for char in seq_in])
                    dataY.append(_char_to_int[seq_out])

                yield (dataX, dataY)




if __name__ == '__main__':
    for x, y in dataset(batchsize=3):
        print(x)


