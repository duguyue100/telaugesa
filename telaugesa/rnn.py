"""Recurrent Neural Network

This module implements base RNN class.
"""

import theano.tensor as T;

import telaugesa.util as util;
from telaugesa.layer import Layer
from telaugesa.fflayers import IdentityLayer;
from telaugesa.nnfuns import tanh, sigmoid;

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
    
class LSTM(object):
    """LSTM implementation"""
    
    def __init__(self,
                 in_dim,
                 out_dim,
                 act_fun=None):
        """Long-short Term Memory
        
        Parameters
        ----------
        in_dim : int
            input dimension
        out_dim : int
            hidden dimension
        act_fun : function
            activation function for input
        """
        
        self.in_dim=in_dim;
        self.out_dim=out_dim;
        
        self.in_layer=IdentityLayer(in_dim=self.in_dim,
                                    out_dim=self.out_dim*4,
                                    layer_name="LSTM in layer");
        
        if act_fun is None:
            self.act_fun=tanh;
        else:
            self.act_fun=act_fun;
            
        self.init_weights();
        
    def init_weights(self):
        
        self.W_state=util.init_weights("W_state", self.out_dim*4, self.in_dim, weight_type="sigmoid");
        self.W_cell_to_in=util.shared_floatx_nans((self.out_dim,), name="W cell to in");
        self.W_cell_to_forget=util.shared_floatx_nans((self.out_dim,), name="W cell to forget");
        self.W_cell_to_out=util.shared_floatx_nans((self.out_dim,), name="W cell to out");
        
        self.init_state=util.shared_floatx_zeros((self.out_dim, ), name="initial states");
        self.init_cell=util.shared_floatx_zeros((self.out_dim, ), name="initial cell");
    
    def apply(self, X, states, cells, mask=None):
        """Apply LSTM activation
        
        Parameters
        ----------
        X : 2D tensor
            input samples in (batch_size, input_dim)
        states : 2D tensor 
            features in (batch_size, feature_dim)
        cells : 2D tensor
            cell states in (batch_size, cell_dim)
            
        Returns
        -------
        states : 2D tensor
            next state of network
        cells : 2D tensor
            next cell activation
        """
        
        def slice_st(x, no):
            return x[:, no*self.dim:(no+1)*self.dim];
        
        activation=T.dot(states, self.W_state)+self.in_layer.apply(X);
        in_gate=sigmoid(slice_st(activation, 0)+cells*self.W_cell_to_in);
        forget_gate=sigmoid(slice_st(activation, 1)+cells*self.W_cell_to_forget);
        next_cells=(forget_gate*cells+in_gate*self.act_fun(slice_st(activation, 2)));
        out_gate=sigmoid(slice_st(activation,3)+next_cells*self.W_cell_to_out)
        next_states=out_gate*self.act_fun(next_cells);
        
        if mask:
            next_states=(mask[:, None]*next_states+(1-mask[:,None]*states));
            next_cells=(mask[:, None]*next_cells+(1-mask[:,None]*cells));
            
        return next_states, next_cells;
        
    @property
    def params(self):
        return (self.in_layer.params, self.W_state, self.W_cell_to_in,
                self.W_cell_to_forget, self.W_cell_to_out, self.init_state, self.init_cell);