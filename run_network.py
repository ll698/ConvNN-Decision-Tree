from __future__ import print_function
import datahandler
import networkhandler
import rootmodel
import keras
from keras.datasets import cifar10



#init and load cifar dataset
dataset = datahandler.DataHandler()
#dataset.load_cifar_data_set("data/cifar10/data_batch_")
#dataset.normalize()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
dataset.x_train = x_train
dataset.y_train = y_train
dataset.x_test = x_test
dataset.y_test = y_test

#init network
root_network = networkhandler.Network("root", 32, 10, 200, True, dataset)

#define root network model
root_network.define_model(rootmodel.newModel(dataset.x_train))
#root_network.load_model("models/root.h5")
root_network.preprocess()
root_network.optimizer(0.0001, 1e-6)
root_network.compile()
root_network.save_model()
root_network.train(32, 200)



