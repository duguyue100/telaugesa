"""Models

This module documented some training models

+ Feedforward Neural Network (including ConvNets)
"""

class FeedForward(object):
    """Feedforward Neural Network model"""
    
    def __init__(self,
                 num_in,
                 layers=None):
        """Initialize feedforward model
        
        Parameters
        ----------
        num_in : int
            number of input size
        layers : list
            list of layers
        """
        
        self.num_in=num_in;
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
        for k, layer in enumerate(self.layers):
            level_out=layer_input;
            
            if layer.is_recursive:
                pass
            else:
                level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;