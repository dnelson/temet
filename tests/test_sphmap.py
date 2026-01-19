""" Test and benchmark util.sphMap. """
import time

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.sphMap import sphMapWholeBox

def _benchmark_sphmap(synthetic_data=True):
    """ Benchmark performance of sphMap(). """
    # config data
    if synthetic_data:
        # generate random testing data
        class sP:
            boxSize = 100.0
            isSubbox = False

        nPts = 200000
        posDtype = 'float32'
        hsmlMinMax = [1.0,10.0]
        massMinMax = [1e-5,1e-4]

        rng = np.random.default_rng(424242)

        pos   = rng.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)
        hsml  = rng.uniform(low=hsmlMinMax[0], high=hsmlMinMax[1], size=nPts).astype('float32')
        mass  = rng.uniform(low=massMinMax[0], high=massMinMax[1], size=nPts).astype('float32')
        quant = np.zeros(nPts, dtype='float32') + 10.0

    else:
        # load some gas in a box
        sP = sim(res=128, run='tng', redshift=0.0, variant='0000')
        pos   = sP.snapshotSubset('gas', 'pos')
        hsml  = sP.snapshotSubset('gas', 'cellrad')
        mass  = sP.snapshotSubset('gas', 'mass')
        quant = sP.snapshotSubset('gas', 'temp')

    # config imaging
    nPixels    = 100
    axes       = [0,1]

    # map mass and quantity
    densMap  = sphMapWholeBox(pos, hsml, mass, None, axes, nPixels, sP)
    quantMap = sphMapWholeBox(pos, hsml, mass, quant, axes, nPixels, sP)

    assert densMap.shape == quantMap.shape
    assert densMap.sum() == pytest.approx(mass.sum())

def test_sphmap_synthetic(benchmark):
    """ Wrapper for pytest. """
    benchmark(_benchmark_sphmap)

@pytest.mark.requires_data
def test_sphmap_sim():
    """ Wrapper for pytest. """
    _benchmark_sphmap(synthetic_data=False)

def test_sphmap_postarget(benchmark):
    """ Benchmark performance of sphMap with posTargets. """
    rng = np.random.default_rng(424242)

    # config imaging
    nPixels    = 20
    axes       = [0,1]

    # generate random testing data
    class sP:
        boxSize = 100.0
        isSubbox = False

    nPts = 2000
    nTarg = 100
    posDtype = 'float32'
    hsmlMinMax = [1.0,10.0]
    massMinMax = [1e-3,1e-2]

    # map
    pos   = rng.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)
    hsml  = rng.uniform(low=hsmlMinMax[0], high=hsmlMinMax[1], size=nPts).astype('float32')
    mass  = rng.uniform(low=massMinMax[0], high=massMinMax[1], size=nPts).astype('float32')

    posTarget = rng.uniform(low=0.0, high=sP.boxSize, size=(nTarg,3)).astype(posDtype)

    # closure for compute kernel: loop over requested path lengths
    @benchmark
    def _run():
        densMap = sphMapWholeBox(pos, hsml, mass, None, axes, nPixels, sP, nThreads=1, posTarget=posTarget)

        assert densMap.shape == (nTarg,)
