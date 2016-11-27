from keras.callbacks import Callback
import numpy as numpy
import dataset as ds


class SampleModelCallback(Callback):

    def __init__(self, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.int_to_char = ds.meta()['int_to_char']
        self.n_vocab = ds.meta()['vocab_size']
        self.generator = ds.sample(100, seq_length)

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        dataX, _ = next(self.generator)
        start = numpy.random.randint(0, len(dataX) - 1)
        pattern = dataX[start]
        print("Seed:")
        print("\"", ''.join([self.int_to_char[value] for value in pattern]), "\"")
        # generate characters
        output = "START: "
        for i in range(300):
            x = numpy.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = numpy.argmax(prediction)
            result = self.int_to_char[index]
            output = output + result
            seq_in = [self.int_to_char[value] for value in pattern]
            #sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]

        print(output+' :END')
