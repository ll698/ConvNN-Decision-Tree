import pickle
from keras.preprocessing.image import ImageDataGenerator

class data:





    def __init__(self):
        self.x_train = NotImplemented
        self.y_train = NotImplemented
        self.x_test = NotImplemented
        self.y_test = NotImplemented



    def load_data_set(self, train_filepath, test_filepath=False,
                      unpickle=True, normalize=True, scale=255):
        """DOCSTRING HERE"""
        if unpickle:
            with open(train_filepath, 'rb') as fo:
                data_dict = pickle.load(fo, encoding='bytes')
            self.x_train = data_dict['data']
            self.y_train = data_dict['labels']
            if test_filepath:
                with open(test_filepath, 'rb') as fo:
                    data_dict = pickle.load(fo, encoding='bytes')
                self.x_test = data_dict['data']
                self.y_test = data_dict['labels']

        if self.x_train != NotImplemented and normalize:
            self.x_train = self.x_train.astype("float32")
            self.x_test = self.x_test.astype("float32")
            self.x_train /= scale
            self.x_test /= scale
        return data


    def sort_data_by_label(self, data, label):
        return NotImplementedError



    def augment_data(self):
        assert self.x_train != NotImplemented

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

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).






self.datagen.fit(self.x_train)