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
            updates[param]=param-gparam*learning_rate;
            
    return updates;