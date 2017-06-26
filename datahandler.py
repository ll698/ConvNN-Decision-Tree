from __future__ import print_function
import pickle
import numpy as np
import cifar
import keras

class DataHandler:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def load_data_set(self, filepath,
                      unpickle=True, normalize=True, scale=255):
        """DOCSTRING HERE"""

        x_train = np.zeros((50000, 32, 32, 3), dtype='uint8')
        y_train = np.zeros((50000,), dtype='uint8')

        for i in range(1, 6):
            fpath = filepath+ str(i)
            data, labels = cifar.load_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = 'data/cifar10/test_batch'
        x_test, y_test = cifar.load_batch(fpath)


        self.x_train = x_train
        self.x_test = x_test

        y_test = np.reshape(y_test, (len(y_test), 1))
        self.y_test = keras.utils.to_categorical(y_test, 10)

        y_train = np.reshape(y_train, (len(y_train), 1))
        self.y_train = keras.utils.to_categorical(y_train, 10)

        if normalize:
            self.x_train = x_train.astype('float32')
            self.x_test = x_test.astype('float32')
            self.x_train /= 255
            self.x_test /= 255




       # self.x_train = np.reshape(self.x_train, list(np.shape(self.x_train)) + [1, 1, 1])
        #self.x_test = np.reshape(self.x_test, list(np.shape(self.x_test)) + [1])

        #print(self.x_train.shape)
        #self.x_train = self.x_train[::32, 32, 3]
        #self.x_test = self.x_test[::32, 32, 3]
        print(self.x_train.shape)

    def sort_data_by_label(self, x_data, label):
        """DOCSTRING HERE"""
        return NotImplementedError


    def concatenate_training_data(self, x_data, y_data=None):
        """DOCSTRING HERE"""

        if isinstance(x_data) is DataHandler:
            x_data = x_data.x_train
            y_data = x_data.y_train




        assert self.x_train.shape[1] == x_data.shape[1], "data shapes do not match: %r vs %r" % (self.x_train.shape, x_data.shape)
        assert self.y_train.shape[1] == y_data.shape[1], "label shapes do not match: %r vs %r" % (self.y_train.shape, y_data.shape)
        self.x_train = np.concatenate(self.x_train, x_data)
        self.y_train = np.concatenate(self.y_train, y_data)
