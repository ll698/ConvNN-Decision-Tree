from __future__ import print_function
import datahandler
import networkhandler
import rootmodel
import keras
from keras.datasets import cifar10
#import cifar

num_classes = 10
batch_size = 32
epochs = 200

#init and load cifar dataset
dataset = datahandler.DataHandler()
dataset.load_cifar_data_set(num_classes)
#dataset.normalize()

#init network
root_network = networkhandler.Network("root", batch_size, num_classes, epochs, True, dataset)

#define root network model
root_network.define_model(rootmodel.newModel(dataset.x_train, num_classes))
#root_network.load_model("models/root.h5")
root_network.preprocess()
root_network.optimizer(0.0001, 1e-6)
root_network.compile()
#root_network.save_model()
root_network.train(batch_size, epoch)



