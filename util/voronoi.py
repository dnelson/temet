"""
util/voronoi.py
  Algorithms and methods related to construction and use of the Voronoi mesh.
"""
import numpy as np
import h5py
from os.path import isfile

from cosmo.load import _haloOrSubhaloSubset

def readVoronoiConnectivityVPPP():
    """ Read the Voronoi mesh data from Chris Byrohl using his vppp (voro++ parallel) approach. """
    file = "/freya/ptmp/mpa/cbyrohl/public/vppp_dataset/IllustrisTNG_z1.0_posdata.bin.nb"
    ngb_ind_file = "/freya/ptmp/mpa/cbyrohl/public/vppp_dataset/IllustrisTNG_z1.0_posdata.bin.nb2"

    dtype_nb = np.dtype([
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("gidx", np.int64), # snapshot index (1-indexed !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)
        ("noffset", np.int64), #offset in neighbor list
        ("ncount", np.int32), #neighborcount
    ])

    # (A) load all (gidx,noffset,ncount)
    if 0:
        with open(file, "rb") as f:
            # get npart
            f.seek(0)
            npart = np.fromfile(f,dtype=np.int64,count=1)[0]

            # chunked load
            chunksize = 10000000

            nloaded = 0
            byte_offset = 8 # skip npart

            while nloaded < npart:
                print('loaded [%9d] of [%9d]' % (nloaded,npart))
                f.seek(byte_offset)
                data = np.fromfile(f, dtype=dtype_nb, count=chunksize)

                # save something from data
                res = np.histogram(data['ncount'],bins=201,range=[-0.5,200.5])

                # continue
                nloaded += data.size
                byte_offset += data.size * dtype_nb.itemsize

    # (B) for a given entry, read its neighbor (index) list
    cell_index = 123456

    with open(file, "rb") as f:
        f.seek(0)
        tot_num_cells = np.fromfile(f, dtype=np.int64, count=1)[0]
        
        byte_offset = 8 + cell_index * dtype_nb.itemsize
        f.seek(byte_offset)

        entry = np.fromfile(f, dtype=dtype_nb, count=1)[0]

    with open(ngb_ind_file,"rb") as f:
        f.seek(0)
        tot_num_entries = np.fromfile(f, dtype=np.int64, count=1)[0]

        byte_offset = 8 + entry['noffset'] * 8
        f.seek(byte_offset)

        ngb_inds = np.fromfile(f, dtype=np.int64, count=entry['ncount'])

    import pdb; pdb.set_trace()

    print('done.')

def loadSingleHaloVPPP(sP, haloID):
    """ Load Voronoi connectivity information for a single FoF halo. """

    subset = _haloOrSubhaloSubset(sP, haloID=haloID)

    indStart = subset['offsetType'][sP.ptNum('gas')]
    indStop  = indStart + subset['lenType'][sP.ptNum('gas')]

    # read neighbor list for all gas cells of this halo

    # construct concatenated neighbor list, and offsets (per cell) into this list

    # verify all neighbors are halo-scope (if any go beyond fof, set them to -1)

    # return (n_ngb[ncells], ngb_list[n_ngb.sum()], ngb_offset[ncells])

def _localVoronoiMaxima(connectivity, property):
    """ For a given set of gas cells with connectivity, identify all of those which correspond to 
    local maximum of the given property vector, defined as all natural neighbors have a smaller value. """
    pass

def voronoiThresholdSegmentation(sP, haloID, propertyName, propertyThreshold):
    """ For all the gas cells of a given halo, identify collections which are spatially connected in the 
    sense that they are Voronoi natural neighbors, and which satisfy a threshold criterion on a particular 
    gas property, e.g. log(T) < 4.5. """
    pass

def voronoiWatershed(sP, haloID, propertyName):
    """ For all the gas cells of a given halo, identify collections which are spatially connected (in the 
    sense that they are Voronoi natural neighbors) using a watershed algorithm. That is, start at all the 
    local maxima simultaneously, and flood outwards until encountering an already flooded cell, the 
    interface between the two then becoming the segmentation line. """
    pass

def benchmark():
    """ Test above routines. """
    from util.simParams import simParams
    import time

    # config
    sP = simParams(run='tng50-1', redshift=0.5)
    haloID = 8
    propName = 'Mg II numdens'
    propThresh = 1e-7

    # run
    start_time = time.time()

    result = voronoiThresholdSegmentation(sP, haloID=haloID, propertyName=propName, propertyThreshold=propThresh)

    print("Took: [%g sec]" % (time.time()-start_time))
