from __future__ import print_function

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.layers import Input, Dense
from keras.models import Model
# Untrained architecture for root node


def newModel(x_train, num_classes):
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    #return model"""


    image = Input(shape=x_train.shape[1:])
    inc1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    drop1 = Dropout(.2)(inc1)
    inc2 = Conv2D(32, (6, 6), activation='relu', padding='same')(drop1)
    drop2 = Dropout(.2)(inc1)
    inc3 = Conv2D(32, (12, 12), activation='relu', padding='same')(drop2)
    inc_pooling = MaxPooling2D((2, 2))(inc3)

    conv1 = Conv2D(32, (12, 12), activation='relu', padding='same')(image)
    drop3 = Dropout(.2)(conv1)
    conv2 = Conv2D(32, (6, 6), activation='relu', padding='same')(drop3)
    drop4 = Dropout(.2)(conv2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(drop4)
    dec_pooling = MaxPooling2D((2, 2))(conv3)

    conc = keras.layers.concatenate([inc_pooling, dec_pooling], axis=1)
    out = Flatten()(conc)
    dense1 = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(out)
    drop5 = Dropout(.2)(dense1)
    dense2 = Dense(512, activation='relu', kernel_constraint=maxnorm(3))(drop5)
    drop6 = Dropout(.2)(dense2)
    dense3 = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(drop6)
    drop7 = Dropout(.2)(dense3)
    output = Dense(10, activation='softmax')(drop7)
    return Model(inputs=image, outputs=output)