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


    image = Input(shape=(3, 32, 32))
    inc_conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(image)
    inc_conv2 = Conv2D(64, (6, 6), activation='relu', padding='same')(inc_conv1)
    inc_conv3 = Conv2D(64, (12, 12), activation='relu', padding='same')(inc_conv2)

    dec_conv1 = Conv2D(32, (12, 12), activation='relu', padding='same')(image)
    dec_conv2 = Conv2D(64, (6, 6), activation='relu', padding='same')(dec_conv1)
    dec_conv3 = Conv2D(64, (3, 3), activation='relu', padding='same')(dec_conv2)

    conc = keras.layers.concatenate([inc_conv3, dec_conv3], axis=1)

    dense1 = Dense(512, activation='relu')(conc)
    dense2 = Dense(512, activation='relu')(dense1)
    dense3 = Dense(256, activation='relu')(dense2)
    output = Dense(10, activation='softmax')(dense3)
    return Model(inputs=image, outputs=output)