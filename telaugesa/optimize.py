"""Optimization method

Supported method:

+ Stochastic gradient descent
"""

from collections import OrderedDict;

import theano.tensor as T;

def gd_updates(cost,
               params,
               updates=None,
               max_norm=5.0,
               learning_rate=0.1,
               eps=1e-6,
               rho=0.95,
               method="sgd"):
    """Gradient Descent based optimization
    
    Parameters
    ----------
    cost : scalar
        total cost of the cost function.
    params : list
        parameter list
    method : string
        optimization method: "sgd", "adagrad", "adadelta"
        
    Returns
    -------
    updates : OrderedDict
        dictionary of updates
    """
    
    if updates is None:
        updates=OrderedDict();
    
    gparams=T.grad(cost, params);
    
    for gparam, param in zip(gparams, params):
        if method=="sgd":
            updates[param]=param-learning_rate*gparam;
            
    return updates;

theano_rng=T.shared_randomstreams.RandomStreams(1234);

def dropout(shape, prob=0.):
    """generate dropout mask
    
    Parameters
    ----------
    shape : tuple
        shape of the dropout mask
    prob : double
        probability of each sample
        
    Returns
    -------
    mask : tensor
        dropout mask
    """
    
    mask=theano_rng.binominal(n=1, p=1-prob, size=shape);
    return T.cast(x=mask, dtype="float32");

def multi_dropout(shapes, prob=0.):
    """generate a list of dropout mask
    
    Parameters
    ----------
    shapes : tuple of tuples
        list of shapes of dropout masks
    prob : double
        probability of each sample
    
    Returns
    -------
    masks : tuple of tensors
        list of dropout masks
    """
    return [dropout(shape, dropout) for shape in shapes];

def apply_dropout(X, mask=None):
    """apply dropout operation
    
    Parameters
    ----------
    X : tensor
        data to be masked
    mask : dropout mask
    
    Returns
    -------
    masked_X : tensor
        dropout masked data
    """
    
    if mask is not None:
        return X*mask;
    else:
        return X;
    
def corrupt_input(X, corruption_level=0.):
    """Add noise on data
    
    Parameters
    ----------
    X : tensor
        data to be corrupted
    corruption_level : double
        probability of the corruption level
    Returns
    -------
    corrupted_out : tensor
        corrupted output 
    """
    
    return apply_dropout(X, dropout(X.shape, corruption_level));