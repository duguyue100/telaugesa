"""Image Caption Generation Test"""

import sys;
import numpy as np;
from collections import defaultdict;

import theano;
import theano.tensor as T;

from telaugesa.im2text import Im2TextDataProvider;

data=Im2TextDataProvider(feature_name="../data/coco/vgg_feats.mat",
                         description_name="../data/coco/dataset.json");
                         
print len(data.desc)