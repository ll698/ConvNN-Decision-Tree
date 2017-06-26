import pickle




def load_data_set(self, filepath, unpickle = True):
    print("loading data")
    if unpickle:
        with open(filepath, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        self.x_train = dict['data']
    


