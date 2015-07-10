import sys;
sys.path.append("..");

import numpy as np;
import matplotlib.pyplot as plt;
from sacred import Experiment;

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

exp=Experiment("dConvAEs")

@exp.config
def dconvae_config():    
    nkerns=0;
    n_epochs=0;
    batch_size=0;
    start_noise=0;
    noise_step=0;
    
@exp.automain
def dconvae_experiment(nkerns,
                       n_epochs,
                       batch_size,
                       start_noise,
                       noise_step):
    """ Setup dConvAE experiment
    
    This experiment is setting up for testing
    multi level noises on dConvAE using CIFAR-10
    
    Parameters
    ----------
    nkerns : int
        number of filters
    n_epochs : int
        number of training epochs
    batch_size : int
        size of each training batch
    start_noise : float
        start level of noise
    noise_step : float
        incremental steps of noises
    """
    
#     Xtr, Ytr, _, _=ds.load_CIFAR10("../data/CIFAR10");
# 
#     Xtr=np.mean(Xtr, 3);
#     # Xte=np.mean(Xte, 3);
#     Xtrain=Xtr.reshape(Xtr.shape[0], Xtr.shape[1]*Xtr.shape[2])/255.0;
#     # Xtest=Xte.reshape(Xte.shape[0], Xte.shape[1]*Xte.shape[2])/255.0;
# 
#     train_set_x, _=ds.shared_dataset((Xtrain, Ytr));
#     # test_set_x, test_set_y=ds.shared_dataset((Xtest, Yte));
# 
#     n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
#     # n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;

    datasets=ds.load_mnist("../data/mnist.pkl.gz");
    train_set_x, train_set_y = datasets[0];
    valid_set_x, valid_set_y = datasets[1];
    test_set_x, test_set_y = datasets[2];

    n_train_batches=train_set_x.get_value(borrow=True).shape[0]/batch_size;
    n_valid_batches=valid_set_x.get_value(borrow=True).shape[0]/batch_size;
    n_test_batches=test_set_x.get_value(borrow=True).shape[0]/batch_size;

    print "[MESSAGE] The data is loaded"
    
    X=T.matrix("data");
    # y=T.ivector("label");
    idx=T.lscalar();
    corruption_level=T.fscalar();
    
    images=X.reshape((batch_size, 1, 28, 28))
    layer_0=ReLUConvLayer(filter_size=(5,5),
                          num_filters=nkerns,
                          num_channels=1,
                          fm_size=(28,28),
                          batch_size=batch_size,
                          border_mode="same");
                                                  
    layer_1=IdentityConvLayer(filter_size=(11,11),
                              num_filters=1,
                              num_channels=nkerns,
                              fm_size=(28,28),
                              batch_size=batch_size,
                              border_mode="same");
                         
    model=ConvAutoEncoder(layers=[layer_0, MaxPoolingSameSize((28,28)), layer_1]);

    out=model.fprop(images, corruption_level=corruption_level);
    cost=mean_square_cost(out[-1], images)#+L2_regularization(model.params, 0.005);
    
    updates=gd_updates(cost=cost, params=model.params, method="sgd", learning_rate=0.001, momentum=0.975);
    
    train=theano.function(inputs=[idx, corruption_level],
                      outputs=[cost],
                      updates=updates,
                      givens={X: train_set_x[idx * batch_size: (idx + 1) * batch_size]});
                      
    print "[MESSAGE] The model is built"

    epoch = 0;
    min_cost=None;
    corr=np.random.uniform(low=start_noise, high=start_noise+0.1, size=1).astype("float32");
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
            if (np.mean(c)<min_cost*0.5) or (max_iter>=40):
                min_cost=np.mean(c);
                corr_best=corr[0]
                corr=np.random.uniform(low=corr_best, high=corr_best+noise_step, size=1).astype("float32");
                max_iter=0;
            else:
                max_iter+=1;
            
        print 'Training epoch %d, cost %f, min cost %f, curr best %f, curr iter %d' % (epoch, np.mean(c), min_cost, corr_best, max_iter);
    
    filters=model.layers[-1].filters.get_value(borrow=True);

    for i in xrange(nkerns):
        image_adr="../data/dConvAE_multi_level/dConvAE_multi_level_%d.eps" % (i);
        plt.imshow(filters[0, i, :, :], cmap = plt.get_cmap('gray'), interpolation='nearest');
        plt.axis('off');
        plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);