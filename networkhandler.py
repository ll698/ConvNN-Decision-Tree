from __future__ import print_function
import pickle
import datahandler
import numpy as np
import json

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D



class Network:
    classes = 10
    data_augmentation = True
    data = datahandler.DataHandler
    prepr = False
    model = Sequential()
    datagen = None
    opt = None
    left = None
    right = None
    name = "placeholder"


    def __init__(self, name, batch_size, num_classes, epochs, data_augmentation, data):
        self.name = name
        self.batch_size = batch_size
        self.classes = num_classes
        self.epochs = epochs
        self.data_augmentation = data_augmentation
        self.data = data

    def define_model(self, model):
        assert isinstance(model, Sequential) 
        self.model = model

    def preprocess(self, custom=False):
        """DOCSTRING HERE"""

        self.prepr = True
        if custom:
            assert isinstance(custom, ImageDataGenerator), "custom is not type ImageDataGenerator"
        else:
            self.datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally
                height_shift_range=0.1,  # randomly shift images vertically
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

    def optimizer(self, lr, decay):
        self.opt = keras.optimizers.rmsprop(lr, decay)

    # Let's train the model using RMSprop
    def compile(self, loss = 'categorical_crossentropy', metrics =['accuracy']):
        self.model.compile(loss = loss, optimizer = self.opt, metrics = metrics)





    #Trains network on x_dataset
    def train(self, batch_size, epochs):
        if self.datagen != None:
            self.datagen.fit(self.data.x_train)
            self.model.fit_generator(self.datagen.flow(self.data.x_train, self.data.y_train,
                                                       batch_size=batch_size),
                                     steps_per_epoch=self.data.x_train.shape[0] // batch_size,
                                     epochs=epochs,
                                     validation_data=(self.data.x_test, self.data.y_test))



    #Model and Architecture save and store operations
    def save_model(self):
        self.model.save("models/" + self.name + ".h5")

    def save_architecture(self):
        newfile = open("architectures/" + self.name + ".json", 'w')
        json_string = self.model.to_json()
        json.dump(json_string, newfile)

    def load_model(self, filepath):
        self.model = load_model(filepath)

    def load_architecture(self, filepath):
        openfile =  open(filepath, 'r')
        data = json.load(openfile)
        self.model = model_from_json(data)
