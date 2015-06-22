"""Noise Adaption"""

import numpy as np;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.optimize import gd_updates;
from telaugesa.optimize import corrupt_input;
from telaugesa.cost import binary_cross_entropy_cost;

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
idx=T.lscalar();
noise=theano.shared(np.asarray(np.random.normal(scale=0.1, size=(batch_size, 784)),
                               dtype="float32"),
                    borrow=True);
                    
#corrupted=corrupt_input(X, corruption_level=noise, noise_type="gaussian");
corrupted=X+noise;

cost=binary_cross_entropy_cost(corrupted, X);

updates=gd_updates(cost, [noise], method="sgd", learning_rate=0.001);

train=theano.function(inputs=[idx],
                      outputs=[cost],
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                      
epoch = 0;
while (epoch < n_epochs):
    epoch = epoch + 1;
    c = [];
    
    for batch_index in xrange(n_train_batches):
        train_cost=train(batch_index);
        c.append(train_cost);
        #co.append(curr_corr);

    print 'Training epoch %d, cost ' % epoch, np.mean(c);
