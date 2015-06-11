"""Recurrent Neural Network

This module implements base RNN class.
"""

import telaugesa.util as util;
from telaugesa.layer import Layer;

class RNNLayer(Layer):
    """Base RNN class"""
    
    def __init__(self, *args, **kwargs):
        super(RNNLayer, self).__init__(*args, **kwargs);
        
        self.is_recursive=True;
        
    def initialize(self, weight_type="none"):
        """Initialize weights for RNN
        """ 
        
        Layer.initialize(weight_type);
        self.hidden=util.init_weights("Hidden", self.out_dim, weight_type=weight_type);
        
    def apply_lin(self, X, pre_h):
        """apply transformation in RNN
        
        Parameters
        ----------
        X : matrix
            input samples, the size is (number of cases, in_dim)
        pre_h : previous hidden state
        """
        
        pass