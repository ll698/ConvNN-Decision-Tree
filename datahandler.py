
import numpy as np
import keras
from keras.datasets import cifar10

class DataHandler:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def load_cifar_data_set(self, num_classes):
        """DOCSTRING HERE"""

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        #self.normalize(255)

    def normalize(self, val):
        """"DOCSTRING HERE"""
        self.x_train = self.x_train.astype('float32')
        self.x_test = self.x_test.astype('float32')
        self.x_train /= val
        self.x_test /= val


    def sort_data_by_label(self, x_data, label):
        """DOCSTRING HERE"""
        return NotImplementedError
