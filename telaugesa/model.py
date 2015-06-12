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
        X : matrix or 4D tensor
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        out : list
            output list from each layer
        """
        
        out=[];
        level_out=X;
        for k, layer in enumerate(self.layers):
            
            level_out=layer.apply(level_out);
            
            out.append(level_out);
            
        return out;
    
    @property
    def params(self):
        return [param for layer in self.layers if hasattr(layer, 'params') for param in layer.params];
    
class AutoEncoder(object):
    """AutoEncoder model for MLP layers
    
    This model only checking the condition of auto-encoders,
    the training is done by FeedForward model
    """
    
    def __init__(self, layers=None):
        """Initialize AutoEncoder
        
        Parameters
        ----------
        layers : tuple
            list of MLP layers
        """
        
        self.layers=layers;
        self.check();
        
    def check(self):
        """Check the validity of an AutoEncoder
        """
        
        assert self.layers[0].in_dim==self.layers[-1].out_dim, \
            "Input dimension is not match to output dimension";
           
        for layer in self.layers:
            assert hasattr(layer, 'params'), \
                "Layer doesn't have necessary parameters";