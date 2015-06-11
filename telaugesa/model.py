"""Models

This module documented some training models

+ Feedforward Neural Network (including ConvNets)
"""

class FeedForward(object):
    """Feedforward Neural Network model"""
    
    def __init__(self,
                 layers=None):
        """Initialize feedforward model
        
        Parameters
        ----------
        in_dim : int
            number of input size
        layers : list
            list of layers
        """
        self.layers=layers;
        
    def fprop(self,
              X):
        """Forward propagation
        
        Parameters
        ----------
        X : matrix
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        out : list
            output list from each layer
        """
        
        out=[];
        layer_input=X;
        level_out=layer_input;
        for layer in enumerate(self.layers):
            level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;
    
    @property
    def params(self):
        return [param for layer in self.layers for param in layer.params];