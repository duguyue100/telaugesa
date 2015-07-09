"""Stacked fixed noise dCOnvAE test"""

import sys;
sys.path.append("..");

import numpy as np;
import matplotlib.pyplot as plt;
import cPickle as pickle;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;
from telaugesa.fflayers import ReLULayer;
from telaugesa.fflayers import SoftmaxLayer;
from telaugesa.convnet import ReLUConvLayer;
from telaugesa.convnet import SigmoidConvLayer;
from telaugesa.model import ConvAutoEncoder;
from telaugesa.convnet import MaxPooling;
from telaugesa.convnet import Flattener;
from telaugesa.model import FeedForward;
from telaugesa.optimize import gd_updates;
from telaugesa.cost import mean_square_cost;
from telaugesa.cost import categorical_cross_entropy_cost;
from telaugesa.cost import L2_regularization;

n_epochs=50;
batch_size=100;
nkerns=100;

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

################################## FIRST LAYER #######################################

X=T.matrix("data");
y=T.ivector("label");
idx=T.lscalar();
corruption_level=T.fscalar();

images=X.reshape((batch_size, 1, 32, 32))
layer_0_en=ReLUConvLayer(filter_size=(7,7),
                         num_filters=100,
                         num_channels=1,
                         fm_size=(32,32),
                         batch_size=batch_size);
                                                  
layer_0_de=SigmoidConvLayer(filter_size=(7,7),
                            num_filters=1,
                            num_channels=100,
                            fm_size=(26,26),
                            batch_size=batch_size,
                            border_mode="full");
                         
model=ConvAutoEncoder(layers=[layer_0_en, layer_0_de]);

out=model.fprop(images, corruption_level=corruption_level);
cost=mean_square_cost(out[-1], images)+L2_regularization(model.params, 0.005);

updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.1);

train=theano.function(inputs=[idx, corruption_level],
                      outputs=[cost],
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                      
print "[MESSAGE] The Layer 0 model is built"

epoch = 0;
corr=np.random.uniform(low=0.1, high=0.2, size=1).astype("float32");
min_cost=None;
corr_best=corr[0];
max_iter=0;
while (epoch < n_epochs):
    epoch = epoch + 1;
    c = []
    for batch_index in xrange(n_train_batches):
        train_cost=train(batch_index, corr_best)
        c.append(train_cost);
        
    if min_cost==None:
        min_cost=np.mean(c);
    else:
        if (np.mean(c)<min_cost*0.5) or (max_iter>=20):
            min_cost=np.mean(c);
            corr_best=corr[0]
            corr=np.random.uniform(low=corr_best, high=corr_best+0.1, size=1).astype("float32");
            max_iter=0;
        else:
            max_iter+=1;
            
    print 'Training epoch %d, cost ' % epoch, np.mean(c), corr_best, min_cost, max_iter;
    
print "[MESSAGE] The Lyer 0 model is trained"

################################## SECOND LAYER #######################################

## append a max-pooling layer

model_trans=FeedForward(layers=[layer_0_en, MaxPooling(pool_size=(2,2))]);
out_trans=model_trans.fprop(images);

## construct new dConvAE

layer_1_en=ReLUConvLayer(filter_size=(4,4),
                         num_filters=50,
                         num_channels=100,
                         fm_size=(13,13),
                         batch_size=batch_size);
                                                   
layer_1_de=SigmoidConvLayer(filter_size=(4,4),
                            num_filters=100,
                            num_channels=50,
                            fm_size=(10,10),
                            batch_size=batch_size,
                            border_mode="full");
                             
model_1=ConvAutoEncoder(layers=[layer_1_en, layer_1_de]);

out_1=model_1.fprop(out_trans[-1], corruption_level=corruption_level);
cost_1=mean_square_cost(out_1[-1], out_trans[-1])+L2_regularization(model_1.params, 0.005);

updates=gd_updates(cost=cost_1, params=model_1.params, method="sgd", learning_rate=0.1);

train_1=theano.function(inputs=[idx, corruption_level],
                        outputs=[cost_1],
                        updates=updates,
                        givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                      
print "[MESSAGE] The Layer 1 model is built"

epoch = 0;
min_cost=None;
corr=np.random.uniform(low=0.1, high=0.2, size=1).astype("float32");
corr_best=corr[0];
max_iter=0;
while (epoch < n_epochs):
    epoch = epoch + 1;
    c = []
    for batch_index in xrange(n_train_batches):
        train_cost=train_1(batch_index, corr_best)
        c.append(train_cost);
        
    if min_cost==None:
        min_cost=np.mean(c);
    else:
        if (np.mean(c)<min_cost*0.5) or (max_iter>=20):
            min_cost=np.mean(c);
            corr_best=corr[0]
            corr=np.random.uniform(low=corr_best, high=corr_best+0.1, size=1).astype("float32");
            max_iter=0;
        else:
            max_iter+=1;
            
    print 'Training epoch %d, cost ' % epoch, np.mean(c), corr_best, min_cost, max_iter;
    
print "[MESSAGE] The Lyer 1 model is trained"

################################## BUILD SUPERVISED MODEL #######################################
                     
pool_0=MaxPooling(pool_size=(2,2));
pool_1=MaxPooling(pool_size=(2,2));
flattener=Flattener();
layer_2=ReLULayer(in_dim=50*25,
                  out_dim=800);
layer_3=SoftmaxLayer(in_dim=800,
                     out_dim=10);
model_sup=FeedForward(layers=[layer_0_en, pool_0, layer_1_en, pool_1, flattener, layer_2, layer_3]);
 
out_sup=model_sup.fprop(images);
cost_sup=categorical_cross_entropy_cost(out_sup[-1], y);
updates=gd_updates(cost=cost_sup, params=model_sup.params, method="sgd", learning_rate=0.1);
 
train_sup=theano.function(inputs=[idx],
                          outputs=cost_sup,
                          updates=updates,
                          givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size],
                                  y: train_set_y[idx * batch_size: (idx + 1) * batch_size]});
 
test_sup=theano.function(inputs=[idx],
                         outputs=model_sup.layers[-1].error(out_sup[-1], y),
                         givens={X: test_set_x[idx * batch_size: (idx + 1) * batch_size],
                                 y: test_set_y[idx * batch_size: (idx + 1) * batch_size]});
                              
print "[MESSAGE] The supervised model is built"

n_epochs=100;
test_record=np.zeros((n_epochs, 1));
epoch = 0;
while (epoch < n_epochs):
    epoch+=1;
    for minibatch_index in xrange(n_train_batches):
        mlp_minibatch_avg_cost = train_sup(minibatch_index);
         
        iteration = (epoch - 1) * n_train_batches + minibatch_index;
         
        if (iteration + 1) % n_train_batches == 0:
            print 'MLP MODEL';
            test_losses = [test_sup(i) for i in xrange(n_test_batches)];
            test_record[epoch-1] = np.mean(test_losses);
             
            print(('     epoch %i, minibatch %i/%i, test error %f %%') %
                  (epoch, minibatch_index + 1, n_train_batches, test_record[epoch-1] * 100.));
     
filters=model.layers[0].filters.get_value(borrow=True);
 
pickle.dump(test_record, open("../data/stacked_improve_dconvae.pkl", "w"));
 
for i in xrange(100):
    image_adr="../data/stacked_improve_dconvae/stacked_improve_dconvae_filter_%d.eps" % (i);
    plt.imshow(filters[i, 0, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
    plt.axis('off');
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);