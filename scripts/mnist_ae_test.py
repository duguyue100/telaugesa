"""MNIST AutoEncoder Test"""

import numpy as np;
import matplotlib.pyplot as plt;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.fflayers import ReLULayer;
from telaugesa.fflayers import SigmoidLayer;
from telaugesa.model import AutoEncoder;
from telaugesa.optimize import gd_updates;
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
y=T.ivector("label");
idx=T.lscalar();
ep_idx=T.lscalar();
corruption_level=T.fscalar();

encode_layer=SigmoidLayer(in_dim=784,
                          out_dim=500);
                       
decode_layer=SigmoidLayer(in_dim=500,
                          out_dim=784);
                          
model=AutoEncoder(layers=[encode_layer, decode_layer]);

#out=model.fprop(X, corruption_level=corruption_level, noise_type="gaussian");
out=model.fprop(X, corruption_level=corruption_level);
cost=binary_cross_entropy_cost(out[-1], X);

updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.1);

train=theano.function(inputs=[idx, corruption_level],
                      outputs=[cost],
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                      
print "[MESSAGE] The model is built"

epoch = 0;
min_cost=None;
corr=np.random.uniform(low=0.2, high=0.3, size=1).astype("float32");
corr_best=corr[0]
while (epoch < n_epochs):
    epoch = epoch + 1;
    c = []
    #co= [];
    
    #corr=np.random.rand(1).astype("float32");
    for batch_index in xrange(n_train_batches):
        train_cost=train(batch_index, corr_best)
        c.append(train_cost);
        #co.append(curr_corr);
        
    if min_cost==None:
        min_cost=np.mean(c);
    else:
        if (np.mean(c)<min_cost):
            min_cost=np.mean(c);
            corr_best=corr[0]
            corr=np.random.uniform(low=corr_best, high=corr_best+0.15, size=1).astype("float32");
        else:
            corr=np.random.uniform(low=corr_best, high=corr_best+0.15, size=1).astype("float32");

    print 'Training epoch %d, cost ' % epoch, np.mean(c), corr_best;
    
filters=model.layers[0].W.get_value(borrow=True);

for i in xrange(500):
    plt.subplot(25, 20, i);
    plt.imshow(np.reshape(filters[:,i], (28, 28)), cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off')
plt.show();