from __future__ import print_function
import datahandler
import networkhandler
import rootmodel
import keras
from keras.datasets import cifar10
from keras.layers import concatenate
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
vehicle_network = networkhandler.Network("vehicle", batch_size, 4, epochs, True, bin_data)

#define root network model
vehicle_network.define_model(rootmodel.newModel(bin_data.x_train, 4))
vehicle_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
vehicle_network.compile(opt)
print(vehicle_network.model.summary())

vehicle_network.train(batch_size, epochs)
vehicle_network.save_model()
scores = vehicle_network.model.evaluate(vehicle_network.data.x_test, vehicle_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#Animal classifying network---------------------------------------


#init and load cifar dataset
bin_data = datahandler.DataHandler()
bin_data.load_cifar_data_set(6, "animal")
bin_data.normalize(255)

#dataset.sort_data_by_label(dataset.x_train, dataset.y_train)
#init network
animal_network = networkhandler.Network("animal", batch_size, 6, epochs, True, bin_data)

#define root network model
animal_network.define_model(rootmodel.newModel(bin_data.x_train, 6))
animal_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
animal_network.compile(opt)
print(animal_network.model.summary())

animal_network.train(batch_size, epochs)
animal_network.save_model()
scores = animal_network.model.evaluate(animal_network.data.x_test, animal_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


#ensemble network-------------------------------------------------

ensemble_data = datahandler.DataHandler()
ensemble_data.load_cifar_data_set(10)


image = Input(shape=bin_data.x_train.shape[1:])
binary = root_network(input)
vehicle = vehicle_network(input) * binary[0]
animal = animal_network(input) * binary[1]
output = concatenate([vehicle[0:2],animal, vehicle[2:4]])
ensemble_network = Model(inputs=image, outputs=output)
scores = ensemble_network.model.evaluate(ensemble_data.x_test, ensemble_data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))






