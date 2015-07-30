"""MLP with MNIST test
"""

import numpy as np;
import matplotlib.pyplot as plt;
import cPickle as pickle;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.fflayers import ReLULayer;
from telaugesa.fflayers import SoftmaxLayer;
from telaugesa.model import FeedForward;
from telaugesa.optimize import gd_updates;
from telaugesa.optimize import multi_dropout;
from telaugesa.cost import categorical_cross_entropy_cost;

n_epochs=100;
batch_size=100;

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10("../data/CIFAR10");

Xtr=np.mean(Xtr, 3);
Xte=np.mean(Xte, 3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;

train_set_x, train_set_y=ds.shared_dataset((Xtrain, Ytr));
test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte));

n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;

print "[MESSAGE] The data is loaded"

X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();
dropout_rate=T.fscalar();

layer_0=ReLULayer(in_dim=1024,
                  out_dim=500);
layer_1=ReLULayer(in_dim=500,
                  out_dim=200);
layer_2=SoftmaxLayer(in_dim=200,
                     out_dim=10);
                     
dropout=multi_dropout([(batch_size, 1024), (batch_size, 500), (batch_size, 200)], dropout_rate);
                                          
model=FeedForward(layers=[layer_0, layer_1, layer_2], dropout=dropout);
model_test=FeedForward(layers=[layer_0, layer_1, layer_2]);
#model=FeedForward(layers=[layer_0, layer_1, layer_2]);
                  
out=model.fprop(X);
out_test=model_test.fprop(X);
cost=categorical_cross_entropy_cost(out[-1], y);

# Configurations that work
updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.01, momentum=0.9);

train=theano.function(inputs=[idx, dropout_rate],
                      outputs=cost,
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                              y: train_set_y[idx * batch_size: (idx + 1) * batch_size]});

test=theano.function(inputs=[idx],
                     outputs=model_test.layers[-1].error(out_test[-1], y),
                     givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                             y: test_set_y[idx * batch_size: (idx + 1) * batch_size]});
                              
print "[MESSAGE] The model is built"

test_record=np.zeros((n_epochs, 1));
epoch = 0;
while (epoch < n_epochs):
    dr=np.random.uniform(low=0., high=0.5, size=1).astype("float32");
    drs=dr[0];
    epoch+=1;
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train(minibatch_index, drs);
        
        iteration = (epoch - 1) * n_train_batches + minibatch_index;
        
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL';
            test_losses = [test(i) for i in xrange(n_test_batches)];
            test_record[epoch-1] = np.mean(test_losses);
            
            print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.));
                  
pickle.dump(test_record, open("../data/cifar10_dropout_rand.pkl", "w"));

filters=model.layers[0].W.get_value(borrow=True);

plt.figure(1);
for i in xrange(100):
    plt.subplot(10, 10, i);
    plt.subplots_adjust(hspace = .001)
    plt.subplots_adjust(wspace = .001)
    plt.imshow(np.reshape(filters[:,i], (32, 32)), cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off')

plt.savefig("../data/cifar10_dropout_filters_rand.eps", bbox_inches='tight', pad_inches=0);
plt.show();