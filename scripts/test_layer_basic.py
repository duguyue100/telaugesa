"""Basic test of layer
"""

import numpy as np;

import theano;
import theano.tensor as T;

from telaugesa.fflayers import SigmoidLayer;

# Create simple matrix

idx=T.lscalar("index")
X=T.matrix("data");

data_in=np.array([[1,2,3],
                  [4,5,6],
                  [7,8,9]], dtype="float32");
                  
data_in=theano.shared(data_in,
                      borrow=True);
                                            
layer=SigmoidLayer(in_dim=3,
                   out_dim=3);
                
output=theano.function([idx], layer.apply_lin(X),
                       givens={X: data_in[idx:idx+1]});

print output(int(0));