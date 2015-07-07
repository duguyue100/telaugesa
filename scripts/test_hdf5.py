"""testing with hdf5"""

import h5py

f=h5py.File("../data/angry.h5", "r");

print 'loaded'

data=f['/angry'][...]

print data