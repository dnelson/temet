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

    # loop over all particles (Note: np.arange() seems to have a huge penalty here instead of range())
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
        """ Subclass Thread() to provide local storage (rDens,rQuant) which can be retrieved after 
            this thread terminates and added to the global return. Note (on Ody2): This technique with this 
            algorithm has ~94 percent scaling efficiency to 16 threads, drops to ~70 percent at 32. """
        def __init__(self, threadNum, nThreads):
            super(mapThread, self).__init__()

            # allocate local returns as attributes of the function
            self.xi_int = np.zeros( rad_bins_sq.size - 1, dtype='int64' )

            # determine local slice (these are views not copies, even better)
            self.threadNum = threadNum
            self.nThreads  = nThreads

            self.start_ind, self.stop_ind = pSplitRange( [0,nPts], nThreads, threadNum )

            # make local view of pos (non-self inputs to _calc() appears to prevent GIL release)
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

    # map and time
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
