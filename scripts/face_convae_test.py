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

Xtr, ytr=ds.load_fer_2013("../data/fer2013/fer2013.csv");

Xtr/=255.0;

train_set_x, _=ds.shared_dataset((Xtr, ytr));
n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;

print "[MESSAGE] The data is loaded"
 
X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();
 
images=X.reshape((batch_size, 1, 48, 48))
layer_0=ReLUConvLayer(filter_size=(7, 7),
                      num_filters=nkerns,
                      num_channels=1,
                      fm_size=(48,48),
                      batch_size=batch_size,
                      border_mode="same");
                       
layer_1=ReLUConvLayer(filter_size=(7, 7),
                      num_filters=nkerns,
                      num_channels=nkerns,
                      fm_size=(48,48),
                      batch_size=batch_size,
                      border_mode="same");
                       
layer_2=ReLUConvLayer(filter_size=(7, 7),
                      num_filters=nkerns,
                      num_channels=nkerns,
                      fm_size=(48,48),
                      batch_size=batch_size,
                      border_mode="same");
                                                                       
layer_3=IdentityConvLayer(filter_size=(15, 15),
                          num_filters=1,
                          num_channels=nkerns,
                          fm_size=(48,48),
                          batch_size=batch_size,
                          border_mode="same");
                          
model=ConvAutoEncoder(layers=[layer_0, layer_1, layer_2, MaxPoolingSameSize((48, 48)), layer_3]);
 
out=model.fprop(images, corruption_level=None);
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
    image_adr="../data/face_dConvAE_fixed/face_dConvAE_fixed_%d.eps" % (i);
    plt.imshow(filters[0, i, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off');
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);
# plt.show();