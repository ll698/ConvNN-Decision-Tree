
import numpy as np
import keras
from keras.datasets import cifar10

class DataHandler:

    def __init__(self, x_train=None, y_train=None, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def load_cifar_data_set(self, num_classes, tag = ""):
        """DOCSTRING HERE"""

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        print('x_train shape:', x_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')
        x, y, z, a, self.y_train = self.sort_data_by_label(x_train, y_train, tag)
        x, y, z, a, self.y_test = self.sort_data_by_label(x_test, y_test, tag)
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


    def sort_data_by_label(self, data, label, tag):
        """DOCSTRING HERE"""
        ind1 = np.where(np.logical_and(labels > 1, labels < 8))[0] 
        ind2 = np.where(np.logical_or(labels <= 1, labels >= 8))[0] 
        animal_labels = labels[ind1]
        animal_data = data[ind1]

        vehicle_labels = labels[ind2]
        vehicle_data = data[ind2]

        np.put(labels,ind1, 0)
        np.put(labels,ind2, 1)
        binary_labels = labels

        temp = (vehicle_labels > 7).astype(int) * 7
        vehicle_labels = vehicle_labels - temp

        if (tag == "binary"):
            return data, binary_labels

        else if (tag) == "animal"):
            return animal_data, animal_labels

        else if (tag == "vehicle"):
            return vehicle_data, vehicle_labels

        else:
            return data, label

