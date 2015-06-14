"""Cost

This module implements several general cost functions

+ Mean-square cost
+ Categorical Cross Entropy cost
+ Binary Cross Entropy cost
+ L1 regularization
+ L2 regularization
"""

import theano.tensor as T;

def mean_square_cost(Y_hat, Y_star):
    """Mean Square Reconstruction Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of mean square reconstruction cost
    """
    
    return T.sum((Y_hat-Y_star)**2, axis=1).mean();

def binary_cross_entropy_cost(Y_hat, Y_star):
    """Binary Cross Entropy Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of binary cross entropy cost
    """
    
    return T.nnet.binary_crossentropy(Y_hat, Y_star).mean();

def categorical_cross_entropy_cost(Y_hat, Y_star):
    """Categorical Cross Entropy Cost
    
    Parameters
    ----------
    Y_hat : tensor
        predicted output of neural network
    Y_star : tensor
        optimal output of neural network
        
    Returns
    -------
    costs : scalar
        cost of Categorical Cross Entropy Cost
    """
    
    return T.nnet.categorical_crossentropy(Y_hat, Y_star).mean();

def L1_regularization(params, L1_rate=0.):
    """L1 Regularization
    
    Parameters
    ----------
    params : tuple
        list of params
    L1_rate : double
        decay rate of L1 regularization
        
    Returns
    -------
    cost : scalar
        L1 regularization decay
    """
    
    cost=0;
    for param in params:
        cost+=T.sum(T.abs_(param));
        
    return L1_rate*cost;

def L2_regularization(params, L2_rate=0.):
    """L2 Regularization
    
    Parameters
    ----------
    params : tuple
        list of params
    L2_rate : double
        decay rate of L2 regularization
        
    Returns
    -------
    cost : scalar
        L2 regularization decay
    """
    
    cost=0;
    for param in params:
        cost+=T.sum(param**2);
    
    return L2_rate*cost;