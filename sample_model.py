from keras.callbacks import Callback
import numpy as numpy
import sys

class SampleModelCallbac(Callback):

    def __init__(self, int_to_char, model, dataX, n_vocab):
        super().__init__()
        self.int_to_char = int_to_char
        self.model = model
        self.dataX = dataX
        self.n_vocab = n_vocab

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        start = numpy.random.randint(0, len(self.dataX) - 1)
        pattern = self.dataX[start]
        print("Seed:")
        print("\"", ''.join([self.int_to_char[value] for value in pattern]), "\"")
        # generate characters
        for i in range(300):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = self.int_to_char[index]
            seq_in = [self.int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

