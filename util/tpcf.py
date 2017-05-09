"""
util/tpcf.py
  Two-point correlation functions (pairwise distances).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import threading
from numba import jit
from util.helper import pSplit, pSplitRange
from util.sphMap import _NEAREST
from scipy.special import gamma

@jit(nopython=True, nogil=True, cache=True)
def _calcTPCFBinned(pos, rad_bins_sq, boxSizeSim, xi_int, start_ind, stop_ind):
    """ Core routine for tpcf(), see below. """
    numPart = pos.shape[0]
    boxHalf = boxSizeSim / 2.0

    # loop over all particles
    for i in range(start_ind,stop_ind):
        #print(i)
        pi_0 = pos[i,0]
        pi_1 = pos[i,1]
        pi_2 = pos[i,2]

        for j in range(0,numPart):
            if i == j:
                continue

            pj_0 = pos[j,0]
            pj_1 = pos[j,1]
            pj_2 = pos[j,2]

            # calculate 3d periodic squared distance
            dx = _NEAREST(pi_0-pj_0,boxHalf,boxSizeSim)
            dy = _NEAREST(pi_1-pj_1,boxHalf,boxSizeSim)
            dz = _NEAREST(pi_2-pj_2,boxHalf,boxSizeSim)

            r2 = dx*dx + dy*dy + dz*dz

            # find histogram bin and accumulate
            k = 1
            while r2 > rad_bins_sq[k]:
                k += 1

            xi_int[k-1] += 1

    # void return

@jit(nopython=True, nogil=True, cache=True)
def _reduceQuantsInRad(pos_search, pos_target, rad_search, quants, reduced_quants, 
                       reduce_type, boxSizeSim, start_ind, stop_ind):
    """ Core routine for quantReductionInRad(). """
    numQuants = quants.shape[1]
    numTarget = pos_target.shape[0]
    boxHalf = boxSizeSim / 2.0

    rad_search_sq = rad_search * rad_search

    for i in range(start_ind,stop_ind):
        pi_0 = pos_search[i,0]
        pi_1 = pos_search[i,1]
        pi_2 = pos_search[i,2]

        i_save = i - start_ind

        for j in range(0,numTarget):
            if i == j:
                continue

            pj_0 = pos_target[j,0]
            pj_1 = pos_target[j,1]
            pj_2 = pos_target[j,2]

            # calculate 3d periodic squared distance
            dx = _NEAREST(pi_0-pj_0,boxHalf,boxSizeSim)
            if dx > rad_search: continue
            dy = _NEAREST(pi_1-pj_1,boxHalf,boxSizeSim)
            if dy > rad_search: continue
            dz = _NEAREST(pi_2-pj_2,boxHalf,boxSizeSim)
            if dz > rad_search: continue

            r2 = dx*dx + dy*dy + dz*dz

            # within radial search aperture?
            if r2 > rad_search_sq:
                continue

            # MAX
            if reduce_type == 0:
                for k in range(0,numQuants):
                    if quants[j,k] > reduced_quants[i_save,k]:
                        reduced_quants[i_save,k] = quants[j,k]

            # MIN
            if reduce_type == 1:
                for k in range(0,numQuants):
                    if quants[j,k] < reduced_quants[i_save,k]:
                        reduced_quants[i_save,k] = quants[j,k]

            # SUM
            if reduce_type == 2:
                for k in range(0,numQuants):
                    reduced_quants[i_save,k] += quants[j,k]

    # void return

def tpcf(pos, radialBins, boxSizeSim, nThreads=16):
    """ Calculate and simultaneously histogram the results of a two-point auto correlation function, 
    by computing all the pairwise (periodic) distances in pos. 3D only. 

      pos[N,3]      : array of 3-coordinates for the galaxies/points
      radialBins[M] : array of (inner) bin edges in radial distance (code units)
      boxSizeSim[1] : the physical size of the simulation box for periodic wrapping (0=non periodic)

      return is xi(r) of size M-1, where xi[i] is the (DD/RR-1) value computed between radialBins[i:i+1]
    """
    # input sanity checks
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] <= 1:
        raise Exception('Strange dimensions of pos.')
    if radialBins.ndim != 1 or radialBins.size < 2:
        raise Exception('Strange dimensions of radialBins.')
    if pos.dtype != np.float32 and pos.dtype != np.float64:
        raise Exception('pos not in float32/64')

    # square radial bins
    nPts = pos.shape[0]
    rad_bins_sq = np.copy(radialBins)**2
    cutFirst = False

    # add a inner bin edge at zero, and an outer bin edge at np.inf if not already present
    if rad_bins_sq[0] != 0.0:
        rad_bins_sq = np.insert(rad_bins_sq, 0, 0.0)
        cutFirst = True
    if rad_bins_sq[-1] != np.inf:
        rad_bins_sq = np.append(rad_bins_sq, np.inf)

    def _analytic_estimator(xi_int):
        # calculate RR expectation for periodic cube
        vol_enclosed = 4.0 / 3 * np.pi * np.sqrt(rad_bins_sq[:-1])**3.0 # spheres
        bin_volume = np.diff(vol_enclosed) # spherical shells

        mean_num_dens = float(nPts)**2 / boxSizeSim**3.0 # N^2 / V_box
        RR_counts = bin_volume * mean_num_dens

        # xi = DD / RR - 1.0
        xi = xi_int[:-1].astype('float32') / RR_counts - 1.0

        # user did not start with a 0.0 inner bin edge, so throw this first bin out
        if cutFirst:
            xi = xi[1:]

        return xi

    # allocate return
    xi_int = np.zeros( rad_bins_sq.size - 1, dtype='int64' )

    # single threaded?
    # ----------------
    if nThreads == 1:
        # call JIT compiled kernel
        start_ind = 0
        stop_ind = nPts

        _calcTPCFBinned(pos, rad_bins_sq, boxSizeSim, xi_int, start_ind, stop_ind)

        # transform integer counts into the correlation function with an analytic estimate for RR
        xi = _analytic_estimator(xi_int)

        return xi

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """ Subclass Thread() to provide local storage (xi_int) which can be retrieved after 
            this thread terminates and added to the global return. """
        def __init__(self, threadNum, nThreads):
            super(mapThread, self).__init__()

            # allocate local returns as attributes of the function
            self.xi_int = np.zeros( rad_bins_sq.size - 1, dtype='int64' )

            # determine local slice (these are views not copies, even better)
            self.threadNum = threadNum
            self.nThreads  = nThreads

            self.start_ind, self.stop_ind = pSplitRange( [0,nPts], nThreads, threadNum )

            # make local view of pos (non-self inputs to JITed function appears to prevent GIL release)
            self.pos         = pos
            self.rad_bins_sq = rad_bins_sq
            self.boxSizeSim  = boxSizeSim

        def run(self):
            # call JIT compiled kernel
            _calcTPCFBinned(self.pos, self.rad_bins_sq, self.boxSizeSim, 
                            self.xi_int, self.start_ind, self.stop_ind)

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        xi_int += thread.xi_int

    # transform integer counts into the correlation function with an analytic estimate for RR
    xi = _analytic_estimator(xi_int)

    return xi

def quantReductionInRad(pos_search, pos_target, rad_search, quants, reduce_op, boxSizeSim, nThreads=16):
    """ Calculate a reduction operation on one or more quantities for all target points falling within a 3D 
        periodic search radius of each search point.

      pos_search[N,3] : array of 3-coordinates for the galaxies/points to search from
      pos_target[M,3] : array of the 3-coordiantes of the galaxies/points to search over
      rad_search[1]   : single float, 3D search radius (code units)
      quants[M]/[M,P] : 1d or P-d array of quantities, one per pos_target, to process
      reduce_op[str]  : one of 'min', 'max', 'sum'
      boxSizeSim[1]   : the physical size of the simulation box for periodic wrapping (0=non periodic)

      return is reduced_quants[N]/[N,P]
    """
    # input sanity checks
    if pos_search.ndim != 2 or pos_search.shape[1] != 3 or pos_search.shape[0] <= 1:
        raise Exception('Strange dimensions of pos_search.')
    if pos_target.ndim != 2 or pos_target.shape[1] != 3 or pos_target.shape[0] <= 1:
        raise Exception('Strange dimensions of pos_target.')
    if type(rad_search) != type(1.0):
        if rad_search.dtype not in [np.float32,np.float64]:
            raise Exception('Strange type of rad_search.')
    if pos_search.dtype != np.float32 and pos_search.dtype != np.float64:
        raise Exception('pos_search not in float32/64')
    if pos_target.dtype != np.float32 and pos_target.dtype != np.float64:
        raise Exception('pos_target not in float32/64')
    if quants.ndim not in [1,2] or (quants.ndim == 2 and quants.shape[0] != pos_target.shape[0]):
        raise Exception('Strange dimensions of quants.')

    # prepare
    reduce_op = reduce_op.lower()
    reduceOps = {'max':0, 'min':1, 'sum':2}
    assert reduce_op in reduceOps.keys()
    reduce_type = reduceOps[reduce_op]

    nSearch = pos_search.shape[0]
    nTarget = pos_target.shape[0]
    nQuants = quants.shape[1] if quants.ndim == 2 else 1

    # allocate return
    reduced_quants = np.zeros( (nSearch,nQuants), dtype=quants.dtype )
    if reduce_op == 'max': reduced_quants.fill(-np.inf)
    if reduce_op == 'min': reduced_quants.fill(np.inf)

    # single threaded?
    # ----------------
    if nThreads == 1:
        # call JIT compiled kernel
        start_ind = 0
        stop_ind = nSearch

        _reduceQuantsInRad(pos_search, pos_target, rad_search, quants, reduced_quants, 
                           reduce_type, boxSizeSim, start_ind, stop_ind)

        return reduced_quants

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """ Subclass Thread() to provide local storage (reduced_quants) which can be retrieved after 
            this thread terminates and added to the global return. """
        def __init__(self, threadNum, nThreads):
            super(mapThread, self).__init__()

            # determine local slice
            self.threadNum = threadNum
            self.nThreads  = nThreads

            self.start_ind, self.stop_ind = pSplitRange( [0,nSearch], nThreads, threadNum )

            # allocate local returns as attributes of the function
            nSearchLocal = self.stop_ind - self.start_ind
            self.reduced_quants = np.zeros( (nSearchLocal,nQuants), dtype=quants.dtype )

            if reduce_op == 'max': self.reduced_quants.fill(-np.inf)
            if reduce_op == 'min': self.reduced_quants.fill(np.inf)

            # make local view of pos (non-self inputs to JITed function appears to prevent GIL release)
            self.pos_search  = pos_search
            self.pos_target  = pos_target
            self.rad_search  = rad_search
            self.quants      = quants
            self.reduce_type = reduce_type
            self.boxSizeSim  = boxSizeSim

        def run(self):
            # call JIT compiled kernel
            _reduceQuantsInRad(self.pos_search, self.pos_target, self.rad_search, self.quants, 
                               self.reduced_quants, self.reduce_type, self.boxSizeSim, 
                               self.start_ind, self.stop_ind)

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        reduced_quants[thread.start_ind : thread.stop_ind] = thread.reduced_quants

    return reduced_quants

def benchmark():
    """ Benchmark performance of tpcf(). 
    Single thread: 600sec for 100k points, perfect O(N^2) scaling, so 16.7 hours for 1M points. """
    np.random.seed(424242)
    from cosmo.load import groupCat
    from util.simParams import simParams
    import matplotlib.pyplot as plt
    import time

    # config
    nThreads = 16
    rMin = 10.0 # kpc/h
    numRadBins = 40

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    ax.set_xlabel('r [kpc/h]')
    ax.set_ylabel('$\\xi(r)$')
    ax.set_xscale('log')

    #for nThreads in [8,16,32]:
    #for nPts in [1000,8000,16000]:

    # config data
    if 0:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 1500
        pos = np.random.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype('float32')

    if 1:
        # load some gas in a box
        sP = simParams(res=512, run='tng', redshift=0.0, variant=0000)
        pos = groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']
        nPts = pos.shape[0]
        ax.set_yscale('log')

    # make radial bin edges
    rMax = sP.boxSize/2 # boxSize/2
    radialBins = np.logspace( np.log10(rMin), np.log10(rMax), numRadBins)

    rrBinSizeLog = (np.log10(rMax) - np.log10(rMin)) / numRadBins
    rr = 10.0**(np.log10(radialBins) + rrBinSizeLog/2)[:-1]

    # calculate and time
    start_time = time.time()
    nLoops = 1

    for i in np.arange(nLoops):
        xi = tpcf(pos, radialBins, sP.boxSize, nThreads=nThreads)

    sec_per = (time.time()-start_time)/nLoops
    print('2 iterations took [%.3f] sec, nPts = %d nThreads = %d' % (sec_per,nPts,nThreads))

    ax.plot(rr, xi, '-', label='N = %d' % nPts)

    ax.legend()
    fig.tight_layout()
    fig.savefig('benchmark.pdf')
    plt.close(fig)

def benchmark2():
    """ Benchmark performance of quantReductionInRad(). 100k points, 16 threads in 23 sec."""
    np.random.seed(424242)
    import time

    # config
    radSearch  = 100.0 # code units
    nSearch    = 50000
    boxSizeSim = 50000.0
    nTarget    = nSearch
    nQuants    = 4
    reduce_op  = 'max'

    # generate random testing data
    pos_search = np.random.uniform(low=0.0, high=boxSizeSim, size=(nSearch,3)).astype('float32')
    pos_target = np.random.uniform(low=0.0, high=boxSizeSim, size=(nTarget,3)).astype('float32')
    quants     = np.random.uniform(low=0.1, high=1.0, size=(nTarget,nQuants)).astype('float32')

    rsave = None
    for nThreads in [16,8,4,2,1]:
        # calculate and time
        start_time = time.time()
        nLoops = 1

        for i in np.arange(nLoops):
            r = quantReductionInRad(pos_search, pos_target, radSearch, quants, reduce_op, 
                                    boxSizeSim, nThreads=nThreads)

        sec_per = (time.time()-start_time)/nLoops
        print('2 iterations took [%.3f] sec, nSearch = %d nThreads = %d' % (sec_per,nSearch,nThreads))

        if rsave is None: rsave = r
        assert np.array_equal(rsave,r)