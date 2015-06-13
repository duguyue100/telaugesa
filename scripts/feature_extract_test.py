"""Feature extraction test"""

import numpy as np;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.fflayers import ReLULayer;
from telaugesa.fflayers import SoftmaxLayer;
from telaugesa.convnet import ReLUConvLayer;
from telaugesa.convnet import MaxPooling;
from telaugesa.convnet import Flattener;
from telaugesa.model import FeedForward;
from telaugesa.optimize import gd_updates;
from telaugesa.cost import categorical_cross_entropy_cost;
from telaugesa.cost import L2_regularization;

n_epochs=100;
batch_size=100;

datasets=ds.load_mnist("../data/mnist.pkl.gz");
train_set_x, train_set_y = datasets[0];
valid_set_x, valid_set_y = datasets[1];
test_set_x, test_set_y = datasets[2];

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size;
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;

print "[MESSAGE] The data is loaded"

X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();

images=X.reshape((batch_size, 1, 28, 28))

layer_0=ReLUConvLayer(filter_size=(7,7),
                      num_filters=50,
                      num_channels=1,
                      fm_size=(28,28),
                      batch_size=batch_size);
                      
extract=theano.function(inputs=[idx],
                        outputs=layer_0.apply(images),
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                        
print extract(1).shape