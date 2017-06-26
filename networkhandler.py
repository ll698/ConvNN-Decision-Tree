from __future__ import print_function
import pickle
import datahandler
import numpy as np
import keras
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
    prepr = False
    model = Sequential()
    datagen = None
    opt = None

    def __init__(self, batch_size, num_classes, epochs, data_augmentation, data):
        self.batch_size = batch_size
        self.classes = num_classes
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.data = data

    def define_model(self, model):
        assert isinstance(model, Sequential) 
        self.model = model

    def preprocess(self, custom = False):
        """DOCSTRING HERE"""

        self.prepr = True
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

    def optimizer(self):
        self.opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
    
    
    def compile(self, loss = 'categorical_crossentropy', metrics =['accuracy']):
        self.model.compile(loss=loss,
              optimizer=self.opt,
              metrics=['accuracy'])


    def train(self):
        if self.datagen != None:
            self.datagen.fit(self.data.x_train)
            self.model.fit_generator(self.datagen.flow(self.data.x_train, self.data.y_train,
                                                       batch_size=self.batch_size),
                                     steps_per_epoch=self.data.x_train.shape[0] // self.batch_size,
                                     epochs=self.epochs,
                                     validation_data=(self.data.x_test, self.data.y_test))