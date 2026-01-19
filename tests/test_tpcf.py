""" Test and benchmark util.tpcf. """
import time

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.tpcf import quantReductionInRad, tpcf

def _benchmark_tpcf(synthetic_data=True):
    """ Benchmark performance of two-point correlation function calculation.

    Single thread: 600sec for 100k points, perfect O(N^2) scaling, so 16.7 hours for 1M points.
    """
    rng = np.random.default_rng(424242)

    # config
    rMin = 10.0 # kpc/h
    numRadBins = 40
    nThreadsTest = [1,2,4] #[8,16,32,64]
    nPts = 1000 # for synthetic tests

    # config data
    if synthetic_data:
        # generate random testing data
        class sP:
            boxSize = 100.0

        pos = rng.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype('float32')
        weights = rng.uniform(low=0.0, high=1.0, size=(nPts)).astype('float32')
        #sP.boxSize = 20.0 # reduce max bin to leverage early termination
    else:
        # load some galaxies in a box
        sP = sim(res=128, run='tng', redshift=0.0, variant='0000')
        pos = sP.groupCat(fieldsSubhalos=['SubhaloPos'])
        nPts = pos.shape[0]
        weights = None

    # verify threaded output is the same as serial
    xi_serial = None

    for nThreads in nThreadsTest:
        # make radial bin edges
        rMax = sP.boxSize/2
        radialBins = np.logspace(np.log10(rMin), np.log10(rMax), numRadBins)

        #rrBinSizeLog = (np.log10(rMax) - np.log10(rMin)) / numRadBins
        #rr = 10.0**(np.log10(radialBins) + rrBinSizeLog/2)[:-1]

        # calculate and time
        start_time = time.time()

        xi, DD, RR = tpcf(pos, radialBins, sP.boxSize, weights=weights, nThreads=nThreads)

        if nThreads == 1:
            xi_serial = xi
        else:
            assert np.allclose(xi, xi_serial), "Threaded tpcf result does not match serial result!"

        sec_per = (time.time()-start_time)
        print('iterations took [%.3f] sec, nPts = %d nThreads = %d' % (sec_per,nPts,nThreads))

        assert xi.size == radialBins.size - 1

def test_tpcf_synthetic(benchmark):
    """ Wrapper for pytest. """
    benchmark(_benchmark_tpcf)

@pytest.mark.requires_data
def test_tpcf_sim():
    """ Wrapper for pytest. """
    _benchmark_tpcf(synthetic_data=False)

def test_benchmark_quantreduction(benchmark):
    """ Benchmark performance of quantReductionInRad(). Typically: 100k points, 16 threads in 23 sec. """
    rng = np.random.default_rng(424242)

    # config
    rMin = 10.0 # kpc/h
    rMax = 25000.0 # kpc/h
    numRadBins = 40

    nSearch    = 10000
    boxSizeSim = 50000.0
    nTarget    = nSearch // 10
    nThreadsTest = [1,2,4] #[16,8,4,2,1]
    nQuants    = 4
    reduce_op  = 'max'

    # generate random testing data
    pos_search = rng.uniform(low=0.0, high=boxSizeSim, size=(nSearch,3)).astype('float32')
    pos_target = rng.uniform(low=0.0, high=boxSizeSim, size=(nTarget,3)).astype('float32')
    quants     = rng.uniform(low=0.1, high=1.0, size=(nTarget,nQuants)).astype('float32')

    # closure for compute kernel: loop over requested path lengths
    @benchmark
    def _run():
        # verify threaded output is the same as serial
        r_serial = None

        for nThreads in nThreadsTest:
            # calculate and time
            start_time = time.time()

            radialBins = np.logspace(np.log10(rMin), np.log10(rMax), numRadBins)

            r = quantReductionInRad(pos_search, pos_target, radialBins, quants, reduce_op,
                                    boxSizeSim, nThreads=nThreads)

            sec_per = (time.time()-start_time)
            print('took [%.3f] sec, nSearch = %d nThreads = %d' % (sec_per,nSearch,nThreads))

            if nThreads == 1:
                r_serial = r
            else:
                assert np.allclose(r, r_serial), "Threaded result does not match serial result!"
