"""
Principle Components Analysis (PCA) example for Concrete Compressive Strength Data

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

import telaugesa.datasets as ds;

### General Parameters

n_epochs=1000;
batch_size=10;

### Load and process data

X_data, y_data=ds.load_ccs_data("../data/Concrete_Data.csv");

X_data=X_data-np.mean(X_data, axis=0);
X_data=X_data/np.std(X_data,axis=1).reshape((X_data.shape[0],1));

X_cov=X_data.T.dot(X_data)/X_data.shape[0];

U, S, _ = LA.svd(X_cov);

X_data=U.T.dot(X_data.T).T;

X_data=X_data[:, :6];

print X_data.shape

print S;

print S[0:6]

print np.sum(S[0:6])/np.sum(S);

