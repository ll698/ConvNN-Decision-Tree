from __future__ import print_function

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
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
    model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model
    """
    image = Input(shape=x_train.shape[1:])
    conv1 = Conv2D(96, (3, 3), activation='relu', padding='same')(image)
    conv2 = Conv2D(96, (6, 3), activation='relu', padding='same')(conv1)
    conv3 = Conv2D(96, (3, 3), activation='relu', padding='same', strides = 2)(conv2)
    conv4 = Conv2D(192, (3, 3), activation='relu', padding='same')(conv3)
    conv5 = Conv2D(192, (6, 6), activation='relu', padding='same')(conv4)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)
    maxpool = MaxPooling2D(pool_size=(2, 2), strides = 2)(conv6)
    conv7 = Conv2D(192, (3, 3), activation='relu', padding='same')(maxpool)
    conv8 = Conv2D(192, (1, 1), activation='relu', padding='same')(conv7)
    conv9 = Conv2D(10, (1, 1), activation='relu', padding='same')(conv8)
    avepool = GlobalAveragePooling2D()(conv6)
    output = Dense(num_classes, activation='softmax')
    return Model(inputs=image, outputs=output)
