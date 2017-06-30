from __future__ import print_function
import datahandler
import networkhandler
import rootmodel
import keras
from keras.datasets import cifar10
#import cifar

num_classes = 2
batch_size = 32
epochs = 200
lrate = 0.01
decay = lrate/epochs

#init and load cifar dataset
dataset = datahandler.DataHandler()
dataset.load_cifar_data_set(num_classes)
dataset.normalize(255)

#dataset.sort_data_by_label(dataset.x_train, dataset.y_train)
#init network
root_network = networkhandler.Network("binary", batch_size, num_classes, epochs, True, dataset)

#define root network model
root_network.define_model(rootmodel.newModel(dataset.x_train, num_classes))
#root_network.load_model("models/root.h5")
root_network.preprocess()
opt = keras.optimizers.SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
root_network.compile(opt)
print(root_network.model.summary())
#root_network.data.normalize()
root_network.save_model()
root_network.train(batch_size, epochs)
scores = root_network.model.evaluate(root_network.data.x_test, root_network.data.y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


