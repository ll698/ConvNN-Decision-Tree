from __future__ import print_function
import pickle
import numpy as np

class DataHandler:

    def __init__(self, x_train, y_train, x_test=None, y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def load_data_set(self, train_filepath, test_filepath=False,
                      unpickle=True, normalize=True, scale=255):
        """DOCSTRING HERE"""
        if unpickle:
            with open(train_filepath, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            self.x_train = data_dict['data']
            self.y_train = data_dict['labels']
            if test_filepath:
                with open(test_filepath, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                self.x_test = data_dict['data']
                self.y_test = data_dict['labels']
        else:
            return AssertionError("unpickled data loading not yet implemented")

        if self.x_train != NotImplemented and normalize:
            self.x_train = self.x_train.astype("float32")
            self.x_test = self.x_test.astype("float32")
            self.x_train /= scale
            self.x_test /= scale


    def sort_data_by_label(self, x_data, label):
        """DOCSTRING HERE"""
        return NotImplementedError

