"""
This file contains a example of regression using highway neural networks

Author: Yuhuang Hu
Email: duguyue100@gmail.com
"""

import numpy as np;
import numpy.linalg as LA;
from mpl_toolkits.mplot3d import Axes3D;
from matplotlib import cm;
import matplotlib;
matplotlib.use('tkagg');
import matplotlib.pyplot as plt;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.fflayers import ReLULayer, SigmoidLayer, TanhLayer, IdentityLayer;
from telaugesa.highway import HighwayReLULayer, HighwayTanhLayer;
from telaugesa.model import FeedForward;
from telaugesa.optimize import gd_updates;
from telaugesa.optimize import multi_dropout;
from telaugesa.cost import mean_squared_cost, L2_regularization, binary_cross_entropy_cost;

### General Parameters

n_epochs=300;
batch_size=10;

### Load and process data

X_data, y_data=ds.load_ccs_data("../data/Concrete_Data.csv");

X_data=X_data-np.mean(X_data, axis=0);
X_data=X_data/np.std(X_data,axis=1).reshape((X_data.shape[0],1));
#### PCA
X_cov=X_data.T.dot(X_data)/X_data.shape[0];
U, S, _ = LA.svd(X_cov);
X_data=U.T.dot(X_data.T).T;
X_data=X_data[:, :6];
#### PCA

X_train=X_data[:700, :];
y_train=y_data[:700].reshape((700,1));

X_test=X_data[700:, :];
y_test=y_data[700:].reshape((330,1));

# plt.figure();
# for i in xrange(X_data.shape[1]):
#   plt.subplot(3,3,i+1);
#   plt.plot(X_data[:,i], y_data, '.');
# plt.show()

X_train=theano.shared(X_train, borrow=True);
y_train=theano.shared(y_train, borrow=True);
X_test=theano.shared(X_test, borrow=True);
y_test=theano.shared(y_test, borrow=True);

n_train_batches=X_train.get_value(borrow=True).shape[0]/batch_size;
n_test_batches=X_test.get_value(borrow=True).shape[0]/batch_size;

print "[MESSAGE] Data is loaded and processed"

X=T.matrix("data");
y=T.matrix("target");
idx=T.lscalar();

layers=[ReLULayer(in_dim=6, out_dim=30)];

for i in xrange(20):
    layers.append(HighwayReLULayer(in_dim=30));
    
layers.append(ReLULayer(in_dim=30, out_dim=1));

model=FeedForward(layers=layers);
out=model.fprop(X);
cost=mean_squared_cost(out[-1], y)+L2_regularization(model.params, 0.01);
#cost=binary_cross_entropy_cost(out[-1], y)+L2_regularization(model.params, 0.01);

#updates=gd_updates(cost=cost, params=model.params, method="adagrad", learning_rate=0.01);
updates=gd_updates(cost=cost, params=model.params, method="rmsprop", learning_rate=0.001);

train=theano.function(inputs=[idx],
                      outputs=cost,
                      updates=updates,
                      givens={X: X_train[idx * batch_size: (idx + 1) * batch_size],
                              y: y_train[idx * batch_size: (idx + 1) * batch_size]});

test=theano.function(inputs=[idx],
                     outputs=cost,
                     givens={X: X_test[idx * batch_size: (idx + 1) * batch_size],
                             y: y_test[idx * batch_size: (idx + 1) * batch_size]});

print "[MESSAGE] The model is built"

test_record=np.zeros((n_epochs, 1));
epoch = 0;
while (epoch < n_epochs):
    epoch+=1;
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train(minibatch_index);
        
        iteration = (epoch - 1) * n_train_batches + minibatch_index;
        
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL %f' % mlp_minibatch_avg_cost ;
            test_losses = [test(i) for i in xrange(n_test_batches)];
            test_record[epoch-1] = np.mean(test_losses);
            
            print(('     epoch %i, minibatch %i/%i, test error %f') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1]));
                  
test_out=theano.function(inputs=[X],
                         outputs=out[-1]);
                         
predict_out=test_out(X_train.get_value(borrow=True));
real_out=y_train.get_value(borrow=True);

x_bar=np.arange(100);
width=0.35;
fig, ax = plt.subplots();

rects1 = ax.bar(x_bar, real_out[:100], width, color='r');
rects2 = ax.bar(x_bar + width, predict_out[:100], width, color='y')

plt.show();