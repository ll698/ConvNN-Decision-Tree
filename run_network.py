from __future__ import print_function
import datahandler
import networkhandler
import rootmodel
import keras
from keras.datasets import cifar10
#import cifar

batch_size = 32
epochs = 200
lrate = 0.01
decay = lrate/epochs

#init and load cifar dataset
bin_data = datahandler.DataHandler()
bin_data.load_cifar_data_set(2, "binary")
bin_data.normalize(255)

#dataset.sort_data_by_label(dataset.x_train, dataset.y_train)
#init network
root_network = networkhandler.Network("binary", batch_size, 2, epochs, True, bin_data)

#define root network model
root_network.define_model(rootmodel.newModel(bin_data.x_train, 2))
root_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
root_network.compile(opt)
print(root_network.model.summary())

root_network.train(batch_size, epochs)
root_network.save_model()
scores = root_network.model.evaluate(root_network.data.x_test, root_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#Vehicle classifying network---------------------------------------


#init and load cifar dataset
bin_data = datahandler.DataHandler()
bin_data.load_cifar_data_set(4, "vehicle")
bin_data.normalize(255)

#dataset.sort_data_by_label(dataset.x_train, dataset.y_train)
#init network
root_network = networkhandler.Network("vehicle", batch_size, 4, epochs, True, bin_data)

#define root network model
root_network.define_model(rootmodel.newModel(bin_data.x_train, 4))
root_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
root_network.compile(opt)
print(root_network.model.summary())

root_network.train(batch_size, epochs)
root_network.save_model()
scores = root_network.model.evaluate(root_network.data.x_test, root_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#Animal classifying network---------------------------------------


#init and load cifar dataset
bin_data = datahandler.DataHandler()
bin_data.load_cifar_data_set(6, "animal")
bin_data.normalize(255)

#dataset.sort_data_by_label(dataset.x_train, dataset.y_train)
#init network
root_network = networkhandler.Network("animal", batch_size, 6, epochs, True, bin_data)

#define root network model
root_network.define_model(rootmodel.newModel(bin_data.x_train, 6))
root_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
root_network.compile(opt)
print(root_network.model.summary())

root_network.train(batch_size, epochs)
root_network.save_model()
scores = root_network.model.evaluate(root_network.data.x_test, root_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

