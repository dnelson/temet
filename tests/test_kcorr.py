""" Test and benchmark cosmo.kCorr. """
import numpy as np
import pytest

from temet.cosmo.kCorr import kCorrections

def test_benchmark_kcorr(benchmark):
    """ Benchmark. """
    rng = np.random.default_rng(424242)
    filter_name = 'u'
    color_name = 'ui'

    N = 100000
    redshifts    = rng.uniform(0.0, 0.1, size=N)
    color_values = rng.uniform(0.5, 2.5, size=N)

    # closure for compute kernel: loop over requested path lengths
    @benchmark
    def _run():
        k = kCorrections(filter_name, redshifts, color_name, color_values)

        # verify
        assert k.shape == redshifts.shape
        assert k[0] == pytest.approx(0.12561865)
