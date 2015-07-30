"""Highway Networks

This is a implementation of Highway Networks which originally proposed in:

[Highway Networks](http://arxiv.org/abs/1505.00387) by Rupesh Kumar Srivastava, Klaus Greff, Jurgen Schmidhuber.

+ Highway Identity Layer
+ Highway Tanh Layer
+ Highway Sigmoid Layer
+ Highway ReLU Layer
"""

import theano.tensor as T;
import telaugesa.nnfuns as nnfuns;
import telaugesa.util as util;

class HighwayLayerBase(object):
    """Base layer of Highway Network"""
    
    def __init__(self,
                 in_dim,
                 layer_name="Highway Layer",
                 W_h=None,
                 W_t=None,
                 bias_h=None,
                 bias_t=None,
                 gate_bias=-5,
                 use_bias=True,
                 **kwargs):
        """Base initialization of highway networks layer
        
        Parameters
        ----------
        in_dim : int
            input dimension of the layer
        W_h : matrix
            hidden weight matrix for the layer, the size should be (in_dim, out_dim),
            if it is None, then the class will create one
        W_t : matrix
            transform weight matrix for the layer, the size should be (in_dim, out_dim,
            if it is None, then the class will create one
        bias_h : vector
            bias vector for the layer, the size should be (out_dim),
            if it is None, then the class will create one
        bias_t : vector
            bias vector for the transform gate, the size should be (out_dim),
            if it is None, then the class will create one
        """
        
        self.in_dim=in_dim;
        self.out_dim=in_dim;
        self.W_h=W_h;
        self.W_t=W_t;
        self.bias_h=bias_h;
        self.bias_t=bias_t;
        self.gate_bias=gate_bias;
        self.use_bias=use_bias;
        
        self.initialize();
        
    def initialize(self, weight_type="none"):
        """initialize weights
        
        Parameters
        ----------
        weight_type : string
            type of weights: "none", "tanh", "sigmoid"
        """
        
        if self.W_h is None:
            self.W_h=util.init_weights("W_h", self.out_dim, self.in_dim, weight_type=weight_type);
        if self.W_t is None:
            self.W_t=util.init_weights("W_t", self.out_dim, self.in_dim, weight_type=weight_type);
            
        if self.bias_h is None:
            self.bias_h=util.init_weights("bias_h", self.out_dim, weight_type=weight_type);
        if self.bias_t is None:
            self.bias_t=util.shared_floatx_ones((self.out_dim,), value=self.gate_bias, name="bias_t");
            
    def apply_lin(self, X):
        """Apply linear transformation for highway networks
        
        Parameters
        ----------
        X : matrix
            input samples, the size is (number of cases, in_dim)
            
        Returns
        -------
        h : matrix
            output results, the size is (number of cases, out_dim);
        t : matrix
            transform gate output, the size of (number of cases, out_dim);
        """
        
        h=T.dot(X, self.W_h);
        t=T.dot(X, self.W_t);
        
        if self.use_bias==True:
            h+=self.bias_h;
            t+=self.bias_t;
            
        return h, t;
    
    def get_dim(self, name):
        """Get dimension
        
        Parameters
        ----------
        name : string
            "input" or "output"
        
        Returns
        -------
        dimension : int
            input or output dimension
        """
        
        if name=="input":
            return self.in_dim;
        elif name=="output":
            return self.out_dim;
    
    @property    
    def params(self):
        return (self.W_h, self.W_t, self.bias_h, self.bias_t);
    
    @params.setter
    def params(self, param_list):
        self.W_h.set_value(param_list[0].get_value());
        self.W_t.set_value(param_list[1].get_value());
        self.bias_h.set_value(param_list[2].get_value());
        self.bias_t.set_value(param_list[3].get_value());
        
####################################
# Highway Layers
####################################

class HighwayIdentityLayer(HighwayLayerBase):
    """Highway Identity Layer """
    
    def __init__(self, **kwargs):
        super(HighwayIdentityLayer, self).__init__(**kwargs);
    
    def apply(self, X):
        h, t=self.apply_lin(X)
        return h*t+X*(1-t);
    
class HighwayTanhLayer(HighwayLayerBase):
    def __init__(self, **kwargs):
        super(HighwayTanhLayer, self).__init__(**kwargs);
    
    def apply(self, X):
        h, t=self.apply_lin(X);
        h=nnfuns.tanh(h);
        t=nnfuns.tanh(t);
    
        return h*t+X*(1-t);

class HighwaySigmoidLayer(HighwayLayerBase):
    def __init__(self, **kwargs):
        super(HighwaySigmoidLayer, self).__init__(**kwargs);
    
    def apply(self, X):
        h, t=self.apply_lin(X);
        h=nnfuns.sigmoid(h);
        t=nnfuns.sigmoid(t);
        
        return h*t+X*(1-t);
    
class HighwayReLULayer(HighwayLayerBase):
    def __init__(self, **kwargs):
        super(HighwayReLULayer, self).__init__(**kwargs);
        
    def apply(self, X):
        h, t=self.apply_lin(X);
        h=nnfuns.relu(h);
        t=nnfuns.relu(t);
        
        return h*t+X*(1-t);