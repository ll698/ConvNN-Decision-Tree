
from __future__ import absolute_import
import pickle


#CIFAR10 decode helper function
def load_batch(filepath):
    with open(filepath, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
        d_decoded = {}
        for k, v in d.items():
            d_decoded[k.decode('utf8')] = v
        d = d_decoded
        fo.close()
        data = d['data']
        labels = d['labels']

        data = data.reshape(data.shape[0], 3, 32, 32)
        return data, labels
