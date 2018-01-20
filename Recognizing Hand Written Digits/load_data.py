import pickle
import gzip
import numpy as np

def load_data():
    with gzip.open('mnist.pkl.gz','rb') as file :
        u = pickle._Unpickler( file )
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load()
    file.close()
    return (training_data,validation_data, test_data)


def vectorize( out ) :
    x = np.zeros((10,1))
    x[out] = 1.0
    return x

def organize_data() :
    train , valid, test = load_data()
    training_inputs = [np.reshape(data, (784,1)) for data in train[0]]
    training_outputs= [vectorize(out) for out in train[1]]
    validation_inputs = [np.reshape(data, (784,1)) for data in valid[0]]
    test_inputs = [np.reshape(data, (784,1)) for data in test[0]]
    training_data = list(zip(training_inputs, training_outputs))
    validation_data = list(zip(validation_inputs, valid[1]))
    test_data = list(zip(test_inputs, test[1]))
    return (training_data, validation_data, test_data)
