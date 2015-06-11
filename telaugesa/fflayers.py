"""Feed-forward Layers (not includeing ConvNet Layer)

This module contains feedforward layers for 

+ Identity layer
+ Tanh layer
+ Sigmoid layer
+ ReLU layer
+ Softmax layer
"""

import numpy as np;

import theano;
import theano.tensor as T;

import telaugesa.nnfuns as nnfuns;
from telaugesa.layer import Layer;

class IdentityLayer(Layer):
    """Identity Layer
    """
    
    def __init__(self, **kwargs):
        super(IdentityLayer, self).__init__(**kwargs);
    
    def apply(self, X):
        return self.apply_lin(X);
        
class TanhLayer(Layer):
    """Tanh Layer
    """
    
    def __init__(self, **kwargs):
        super(TanhLayer, self).__init__(**kwargs);
        
        self.initialize("tanh");
        
    def apply(self, X):
        return nnfuns.tanh(self.apply_lin(X));
    
class SigmoidLayer(Layer):
    """Sigmoid Layer"""
    
    def __init__(self, **kwargs):
        super(SigmoidLayer, self).__init__(**kwargs);
        
        self.initialize("sigmoid");
        
    def apply(self, X):
        return nnfuns.sigmoid(self.apply_lin(X));

class ReLULayer(Layer):
    """ReLU Layer"""
    
    def __init__(self, **kwargs):
        super(ReLULayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.relu(self.apply_lin(X));
    
class SoftmaxLayer(Layer):
    """Softmax Layer"""
    def __init__(self, **kwargs):
        super(SoftmaxLayer, self).__init__(**kwargs);
        
    def apply(self, X):
        return nnfuns.softmax(self.apply_lin(X));
    