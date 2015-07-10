"""Conv AE test on CIFAR-10"""

import sys;
sys.path.append("..");

import numpy as np;
import matplotlib.pyplot as plt;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.convnet import ReLUConvLayer;
from telaugesa.convnet import SigmoidConvLayer;
from telaugesa.convnet import IdentityConvLayer;
from telaugesa.convnet import MaxPoolingSameSize;
from telaugesa.model import ConvAutoEncoder;
from telaugesa.optimize import gd_updates;
from telaugesa.cost import mean_square_cost;
from telaugesa.cost import L2_regularization;

n_epochs=200;
batch_size=200;
nkerns=64;

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
layer_0=ReLUConvLayer(filter_size=(5,5),
                      num_filters=nkerns,
                      num_channels=1,
                      fm_size=(28,28),
                      batch_size=batch_size,
                      border_mode="same");
                                                                      
layer_3=IdentityConvLayer(filter_size=(11, 11),
                          num_filters=1,
                          num_channels=nkerns,
                          fm_size=(28,28),
                          batch_size=batch_size,
                          border_mode="same");
                         
model=ConvAutoEncoder(layers=[layer_0, MaxPoolingSameSize((28, 28)), layer_3]);

out=model.fprop(images, corruption_level=0.8);
cost=mean_square_cost(out[-1], images);#+L2_regularization(model.params, 0.005);

updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.001, momentum=0.975);

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
#     plt.subplot(10, 10, i);
    image_adr="../data/dConvAE_0_8_fixed/dConvAE_0_8_fixed_%d.eps" % (i);
    plt.imshow(filters[0, i, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off');
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);
# plt.show();