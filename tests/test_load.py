""" Test loading from snapshots, group catalogs, merger trees, etc. """
import numpy as np
import pytest

from temet.util import simParams as sim


@pytest.mark.requires_data
def test_snap_header():
    x = sim('TNG100-3', redshift=1.0)

    num_dm = x.snapshotHeader()['NumPart'][x.ptNum('dm')]
    assert num_dm == 455**3

@pytest.mark.requires_data
def test_snap_values():
    x = sim('TNG100-3', redshift=0.5)

    mass = x.snapshotSubset('gas', 'mass')
    assert mass.size == 89749115
    assert mass.sum() == pytest.approx(559386.1)

@pytest.mark.requires_data
def test_snap_custom_values():
    x = sim('TNG100-3', redshift=0.5)

    age = x.snapshotSubset('stars', 'star_age')
    assert age.size == 1933642
    assert np.count_nonzero(np.isnan(age)) == 6150
    assert np.nanmean(age) == 4.491116 # Gyr
