"""
util/voronoi.py
  Algorithms and methods related to construction and use of the Voronoi mesh.
"""
import numpy as np
import h5py
from os.path import isfile

from cosmo.load import _haloOrSubhaloSubset

def loadSingleHaloVPPP(sP, haloID):
    """ Load Voronoi connectivity information for a single FoF halo. """
    subset = _haloOrSubhaloSubset(sP, haloID=haloID)

    indStart = subset['offsetType'][sP.ptNum('gas')]
    indStop  = indStart + subset['lenType'][sP.ptNum('gas')]

    # read neighbor list for all gas cells of this halo
    filename = sP.derivPath + 'voronoi/mesh_%02d.hdf5' % sP.snap

    with h5py.File(filename,'r') as f:
        num_ngb = f['num_ngb'][indStart:indStop]
        offset_ngb = f['offset_ngb'][indStart:indStop]

        tot_ngb = num_ngb.sum()
        assert offset_ngb[0] + tot_ngb == offset_ngb[-1] + num_ngb[-1]

        ngb_inds = f['ngb_inds'][offset_ngb[0] : offset_ngb[0]+tot_ngb]

    # make mesh indices and offsets halo-local
    ngb_inds -= offset_ngb[0]
    offset_ngb -= offset_ngb[0]

    # flag any mesh neighbors which are beyond halo-scope (outside fof) as -1
    w = np.where( (ngb_inds < 0) or (ngb_inds >= tot_ngb) )

    import pdb; pdb.set_trace()
    ngb_inds[w] = -1

    # return (n_ngb[ncells], ngb_list[n_ngb.sum()], ngb_offset[ncells])
    return num_ngb, ngb_inds, offset_ngb

def _localVoronoiMaxima(connectivity, property):
    """ For a given set of gas cells with connectivity, identify all of those which correspond to 
    local maximum of the given property vector, defined as all natural neighbors have a smaller value. """
    pass

def _contiguousVoronoiCells(num_ngb, offset_ngb, thresh_mask, identity):
    """ Identify contiguous (naturally connected) subsets of the input Voronoi mesh cells.
    Only those with thresh_mask == True are assigned. Identity output. """
    ncells = num_ngb.size

    count = 0

    # loop over all cells
    for i in range(ncells):
        # skip cells which do not satisfy thershold
        if mode == 0:
            if prop_val[i] < propThresh:
                continue

        if mode == 1:
            if prop_val[i] >= propThresh:
                continue

        # loop over all natural neighbors of this cell
        for j in range(num_ngb[i]):
            # index of this voronoi neighbor
            ngb_index = offset_ngb[i] + j

            # if the neighbor does not satisfy threshold, skip
            if not thresh_mask[ngb_index]:
                continue

            # if the neighbor belongs to an existing object? then assign to this cell, and exit
            if identity[ngb_index] >= 0:
                identity[i] = identity[ngb_index]
                break

        # no neighbors already assigned, so start a new object now
        if identity[i] < 0:
            identity[i] = count
            count += 1

    return count

def voronoiThresholdSegmentation(sP, haloID, propName, propThresh, propThreshComp):
    """ For all the gas cells of a given halo, identify collections which are spatially connected in the 
    sense that they are Voronoi natural neighbors, and which satisfy a threshold criterion on a particular 
    gas property, e.g. log(T) < 4.5. """

    # load
    num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID)
    prop_val = sP.snapshotSubsetP('gas', propName, haloID=haloID)

    ncells = num_ngb.size

    assert prop_val.shape[0] == num_ngb.size

    # sub-select
    thresh_mask = np.zeros( ncells, dtype='bool')

    if propThreshComp == 'gt':
        w_thresh = np.where(prop_val >= propThresh)
        mode = 0
    if propThreshComp == 'lt':
        w_thresh = np.where(prop_val < propThresh)
        mode = 1

    thresh_mask[w_thresh] = 1

    # run
    identity = np.zeros( ncells, dtype='int32' ) - 1

    count = _contiguousVoronoiCells(num_ngb, offset_ngb, thresh_mask, identity)

    # all cells satisfying threshold should have been assigned
    assert identity[w_thresh] >= 0

    # create list of cells (and number of cells) per object
    sizes = np.bincount(identity[w_thresh])

    obj_indices = np.zeros( sizes.sum(), dtype='int32' )
    # TODO inverse histogram identity with np.digitize()

    import pdb; pdb.set_trace()

    # return (n_objs, obj_lens[n_objs], obj_indices[obj_lens.sum()], obj_ids_per_cell[ncells])
    return count, sizes, obj_indices, identity

def voronoiWatershed(sP, haloID, propName):
    """ For all the gas cells of a given halo, identify collections which are spatially connected (in the 
    sense that they are Voronoi natural neighbors) using a watershed algorithm. That is, start at all the 
    local maxima simultaneously, and flood outwards until encountering an already flooded cell, the 
    interface between the two then becoming the segmentation line. """

    # load
    num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID)
    prop_val = sP.snapshotSubsetP('gas', propName, haloID=haloID)

    ncells = num_ngb.size

    assert prop_val.shape[0] == num_ngb.size

    

    # sort all cells which satisfy threshold by decreasing/increasing value

def benchmark():
    """ Test above routines. """
    from util.simParams import simParams
    import time

    # config
    sP = simParams(run='tng50-1', redshift=0.5)
    haloID = 8
    propName = 'Mg II numdens'
    propThresh = 1e-7
    propThreshComp = 'gt'

    # run
    start_time = time.time()

    result = voronoiThresholdSegmentation(sP, haloID=haloID, propertyName=propName, propertyThreshold=propThresh)

    print("Took: [%g sec]" % (time.time()-start_time))
