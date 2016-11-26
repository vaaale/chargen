from keras.callbacks import Callback
import numpy as numpy
import sys
import dataset as ds


class SampleModelCallback(Callback):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.int_to_char = ds.meta()['int_to_char']
        self.n_vocab = ds.meta()['vocab_size']
        self.generator = ds.sample(100, 40)

    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, epoch, logs={}):
        dataX, _ = next(self.generator)
        start = numpy.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
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

