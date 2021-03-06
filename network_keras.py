from __future__ import print_function
import keras
import pickle
import datahandler
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



class Network:
    batch_size = 32
    classes = 10
    epochs = 200
    data_augmentation = True
    data = datahandler.DataHandler
    preprocess = False
    model = Sequential()

    def __init__(self, batch_size, num_classes, epochs, data_augmentation, data):
        self.batch_size = batch_size
        self.classes = num_classes
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.data = data

    def define_model(self, model):
        assert isinstance(model) == Sequential()
        self.model = model


    def preprocess(self, custom = False):
        """DOCSTRING HERE"""

        self.preprocess = True
        if custom:
            assert type(custom) is ImageDataGenerator, "custom is not of type ImageDataGenerator"
        else:
            self.datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images




