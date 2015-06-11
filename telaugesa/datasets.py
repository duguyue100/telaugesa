"""Datasets

This module implements the interface that loads datasets

+ MNIST
"""

import gzip;
import cPickle as pickle;

import numpy as np;

import theano;
import theano.tensor as T;

def shared_dataset(data_xy, borrow=True):
    """Create shared data from dataset
    
    Parameters
    ----------
    data_xy : list
        list of data and its label (data, label)
    borrow : bool
        borrow property
        
    Returns
    -------
    shared_x : shared matrix
    shared_y : shared vector
    """
        
    data_x, data_y = data_xy;
    shared_x = theano.shared(np.asarray(data_x,
                                        dtype='float32'),
                             borrow=borrow);
                             
    shared_y = theano.shared(np.asarray(data_y,
                                        dtype='float32'),
                             borrow=borrow);
        
    return shared_x, T.cast(shared_y, 'int32');

def load_mnist(dataset):
    """Load MNIST dataset
    
    Parameters
    ----------
    dataset : string
        address of MNIST dataset
    
    Returns
    -------
    rval : list
        training, valid and testing dataset files
    """
    
    # Load the dataset
    f = gzip.open(dataset, 'rb');
    train_set, valid_set, test_set = pickle.load(f);
    f.close();
  
    #mean_image=get_mean_image(train_set[0]);

    test_set_x, test_set_y = shared_dataset(test_set);
    valid_set_x, valid_set_y = shared_dataset(valid_set);
    train_set_x, train_set_y = shared_dataset(train_set);

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)];
    return rval;