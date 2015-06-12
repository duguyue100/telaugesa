"""Training container

This module implements a training container
"""

import theano;
import theano.tensor as T;

class Train(object):
    """Training model container"""
    
    def __init__(self,
                 model,
                 cost,
                 algorithm):
        """Init a training container
        
        Parameters
        ----------
        model : object
            A neural network model that is trained
        algorithm : function
            A training algorithm that feedback the updates
        cost : scalar
            A cost function that provides the costs
        """
        
        self.model=model;
        self.cost=cost;
        self.algorithm=algorithm;
        
    def do(self,
           X,
           y,
           train_set_x,
           train_set_y,
           batch_size,
           num_epochs):
        """Perform training
        
        Parameters
        ----------
        X : tensor
            tensor for batch of data
        y : tensor
            tensor for batch of data
        train_set_x : shared matrix
            entire training data
        train_set_y : shared matrix
            entire training target
        batch_size : int
            number of cases in the batch
        num_epochs : int
            total number of training epochs
        """
        
        updates=self.algorithm(self.cost, self.model.params);
        
        idx=T.lscalar("index");
        train_model=theano.function(inputs=[idx],
                                    outputs=self.cost,
                                    updates=updates,
                                    givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                                            y: train_set_y[idx * batch_size: (idx + 1) * batch_size]})
        
        epoch=0;
        while (epoch < num_epochs):
            epoch+=1;
            
        ### Plans
        ### 1. complete training
        ### 2. also add testing and validation set to monitor
        ### 3. logging
        ### 4. monitor the training process
        ### 5. parameters to __init__