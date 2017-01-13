"""
Code is based on:
[1] http://deeplearning.net/tutorial/logreg.html
[2] http://deeplearning.net/tutorial/mlp.html
[3] http://deeplearning.net/tutorial/lenet.html
[4] Materials of Deep Learning and Neural Networks by prof. Aurel Lazar, Columbai University
"""

import os
import sys
import numpy
import scipy.io

import theano
import theano.tensor as T
from numpy import loadtxt
import cPickle

def shared_dataset(data_xy, borrow=True):
    """ 
    Function for transfarring data to GPU
    """
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=True)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=True)
    return shared_x, T.cast(shared_y, 'int32')

def load_faceScrub(theano_shared=True):

    def convert_data_format(data):
        X = data[:,:-1]
        y = data[:,-1].flatten()
        return (X,y)

    # Load the dataset
    data = []
    RED_IMAGE_SIZE= 100
    with open('../../faceScrub/image_data_new.csv', 'rw') as file:
        for idx,line in enumerate(file):
            z = line.split(',')
            if len(z) == (RED_IMAGE_SIZE*RED_IMAGE_SIZE + 1):
                data.append(numpy.array([int (elem) for elem in z]))
    data = numpy.array(data)
    (num_images, len_img_plus_one) = data.shape


    # reshaping the data to desired form
    data_set = convert_data_format(data)
    # Downsample the training dataset if specified
    train_set_len = len(data_set[1])
    numpy.random.seed(1990)
    l = numpy.arange(train_set_len)
    numpy.random.shuffle(l)
    mul_idx = len(l)
    train_idx = l[:mul_idx/2]
    val_idx = l[mul_idx/2: 3*mul_idx/4]
    test_idx = l[3*mul_idx/4:]

    valid_set = [x[val_idx] for x in data_set]
    train_set = [x[train_idx] for x in data_set]
    test_set = [x[test_idx] for x in data_set]
    

    print ('Data loaded')

    # Load data onto GPU if selected
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        train_set_x, train_set_y = shared_dataset(train_set)

        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    else:
        rval = [train_set, valid_set, test_set]

    return rval

def load_test(theano_shared=False):

    # Load testset
    X_test = numpy.loadtxt('data/X_test.txt', delimiter=',', dtype=int) 

    test_set = [X_test, numpy.zeros((X_test.shape[0],1))]
    
    print ('Data loaded')
    # Load data in GPU if selected
    if theano_shared:
        test_set_x, test_set_y = shared_dataset(test_set)
        rval = [(test_set_x, test_set_y)]
    else:
        rval = [test_set]

    return rval
