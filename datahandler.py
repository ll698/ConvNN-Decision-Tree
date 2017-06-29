
import numpy as np
import cifar
import keras

class DataHandler:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def load_cifar_data_set(self, filepath, scale=255):
        """DOCSTRING HERE"""

        x_train = np.zeros((50000, 3, 32, 32), dtype='uint8')
        y_train = np.zeros((50000,), dtype='uint8')

        for i in range(1, 6):
            fpath = filepath+ str(i)
            data, labels = cifar.load_batch(fpath)
            x_train[(i - 1) * 10000: i * 10000, :, :, :] = data
            y_train[(i - 1) * 10000: i * 10000] = labels

        fpath = 'data/cifar10/test_batch'
        x_test, y_test = cifar.load_batch(fpath)

        y_test = np.reshape(y_test, (len(y_test), 1))
        self.y_test = keras.utils.to_categorical(y_test, 10)

        y_train = np.reshape(y_train, (len(y_train), 1))
        self.y_train = keras.utils.to_categorical(y_train, 10)

        self.x_train = np.transpose(x_train, (0, 2, 3, 1))
        self.x_test = np.transpose(x_test, (0, 2, 3, 1))

    def normalize(self):
        """"DOCSTRING HERE"""
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= 255
        self.x_test /= 255


    def sort_data_by_label(self, x_data, label):
        """DOCSTRING HERE"""
        return NotImplementedError
