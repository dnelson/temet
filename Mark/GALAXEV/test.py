from mpi4py import MPI
import numpy as np
import tables
import sys
import readhaloHDF5
import readsnapHDF5 as rs
import conversions as co
from scipy import interpolate
from scipy.ndimage import zoom
import readsubfHDF5
import time


base="/n/hernquistfs1/Illustris/Runs/Illustris-3/output/"
subnum=10
num=135
cat=readsubfHDF5.subfind_catalog(base, num, long_ids=True, keysel=["SubhaloStellarPhotometrics","SubhaloLenType"])

# "SubhaloStellarPhotometrics":["FLOAT",8]}  #band luminosities: U, B, V, K, g, r, i, z

fname = "test3.hdf5"           #postbase + "/galprop/galprop_" + str(num).zfill(3) + ".hdf5"
f=tables.openFile(fname, mode = "r")
magsub_g = f.root.magsub_g[:]
f.close()

gmag = cat.SubhaloStellarPhotometrics[:,4]

idx=magsub_g>0
 gmag
print magsub_g

