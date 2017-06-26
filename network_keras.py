from __future__ import print_function
import keras
import pickle
import loaddata
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



class Network:
    batch_size = 32
    classes = 10
    epochs = 200
    data_augmentation = True
    data = NotImplemented

    def __init__(self, batch_size, num_classes, epochs, data_augmentation, output):
        self.batch_size = batch_size
        self.classes = num_classes
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.model = Sequential()

    def define_network(self, is_conv=True, conv_layer_size = 32, activation_type = "relu", output_activation = "softmax", dropout_prob = .25):
        assert(activation_type in ['relu', 'softmxax', 'sigmoid', 'tanh'])
        assert(output_activation in ['relu', 'softmax', 'sigmoid', 'tanh'])

        if self.x_train == NotImplemented:
            return AssertionError("Error: data_set has not been loaded yet")

        self.model.add(Conv2D(conv_layer_size, (3, 3),
        padding='same', input_shape=x_train.shape[1:]))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(conv_layer_size, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classes))
        self.model.add(Activation('softmax'))


