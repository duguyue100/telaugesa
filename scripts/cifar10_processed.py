"""Load CIFAR-10 processed image"""

import sys;
sys.path.append("..");

import numpy as np;
import matplotlib.pyplot as plt;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.convnet import ReLUConvLayer;
from telaugesa.convnet import IdentityConvLayer;
from telaugesa.convnet import MaxPoolingSameSize;
from telaugesa.model import ConvAutoEncoder;
from telaugesa.optimize import gd_updates;
from telaugesa.cost import mean_square_cost;

n_epochs=200;
batch_size=200;
nkerns=64;

Xtr, Ytr, Xte, Yte=ds.load_CIFAR10_Processed("../data/CIFAR10/train.npy",
                                             "../data/CIFAR10/train.pkl",
                                             "../data/CIFAR10/test.npy",
                                             "../data/CIFAR10/test.pkl");
Xtr=Xtr.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).mean(3);
Xte=Xte.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).mean(3);
Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])
Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])

train_set_x, train_set_y=ds.shared_dataset((Xtrain, Ytr));
test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte));
 
n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;
 
print "[MESSAGE] The data is loaded"
 
X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();
 
images=X.reshape((batch_size, 1, 32, 32))
layer_0=ReLUConvLayer(filter_size=(5,5),
                      num_filters=nkerns,
                      num_channels=1,
                      fm_size=(32,32),
                      batch_size=batch_size,
                      border_mode="same");
 
layer_1=ReLUConvLayer(filter_size=(5,5),
                      num_filters=nkerns,
                      num_channels=nkerns,
                      fm_size=(32,32),
                      batch_size=batch_size,
                      border_mode="same");
  
layer_2=ReLUConvLayer(filter_size=(5,5),
                      num_filters=nkerns,
                      num_channels=nkerns,
                      fm_size=(32,32),
                      batch_size=batch_size,
                      border_mode="same");
                                                   
layer_3=IdentityConvLayer(filter_size=(7, 7),
                          num_filters=1,
                          num_channels=nkerns,
                          fm_size=(32,32),
                          batch_size=batch_size,
                          border_mode="same");
                          
model=ConvAutoEncoder(layers=[layer_0, layer_1, layer_2, MaxPoolingSameSize((32, 32)), layer_3]);
#model=ConvAutoEncoder(layers=[layer_0, layer_3]);
 
out=model.fprop(images, corruption_level=None);
cost=mean_square_cost(out[-1], images);#+L2_regularization(model.params, 0.005);
 
updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.001, momentum=0.9);
 
train=theano.function(inputs=[idx],
                      outputs=[cost],
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                       
print "[MESSAGE] The model is built"
 
epoch = 0;
while (epoch < n_epochs):
    epoch = epoch + 1;
    c = []
    for batch_index in xrange(n_train_batches):
        train_cost=train(batch_index)
        c.append(train_cost);
         
    print 'Training epoch %d, cost ' % epoch, np.mean(c);
     
filters=model.layers[-1].filters.get_value(borrow=True);
 
for i in xrange(nkerns):    
    image_adr="../data/dConvAE_CIFAR10_0_1_fixed/dConvAE_CIFAR10_0_1_fixed_%d.eps" % (i);
    plt.imshow(filters[0, i, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off');
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);