"""Test and benchmark cosmo.lightcone and util.boxRemap."""

import numpy as np
import pytest

from temet.util import simParams as sim
from temet.util.boxRemap import remapPositions


@pytest.mark.requires_data
def test_boxremap():
    # load some sim data
    sP = sim(res=128, run="tng", redshift=0.0, variant="0000")
    pos = sP.snapshotSubset("gas", "pos")

    # two configurations
    for i in [0, 1]:
        # configs
        if i == 0:
            nPixels = [2000, 2000]
            remapRatio = [2.44, 2.44, 0.168]

        if i == 1:
            nPixels = [1920, 1080] * 2
            remapRatio = [5.0, 2.8125, 0.0711]

        # note: remapRatio must satisfy L1*L2*L3 == 1 constraint, last entry gives fractional z-width
        pos_new, newBoxSize = remapPositions(sP, pos, remapRatio, nPixels)

        # verify
        assert pos_new.shape == pos.shape
        assert newBoxSize.shape == (3,)
        assert newBoxSize == pytest.approx(sP.boxSize * np.array(remapRatio), rel=1e-2)
