"""Image Caption Generation Test"""

import sys;
import numpy as np;
from collections import defaultdict;

import theano;
import theano.tensor as T;

import telaugesa.datasets as ds;

data=ds.load_mat("../data/flickr8k/vgg_feats.mat").T;
des=ds.load_json("../data/flickr8k/dataset.json");

split=defaultdict(list);

for img in des['images']:
    split[img['split']].append(img);
    
print split['train'][0]['sentids']; 

