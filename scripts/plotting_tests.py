"""Some Plotting tests"""

import numpy as np;
import matplotlib.pyplot as plt;

A=np.random.rand(10, 10, 10);

prefix="image"
for i in xrange(10):
    image_adr="image-%d.eps" % (i);
    
    plt.imshow(A[i,:,:]);
    plt.axis('off');
    plt.savefig(image_adr , bbox_inches='tight', pad_inches=0);
    
    print image_adr;