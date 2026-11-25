"""Test Cloudy line luminosities."""

from types import SimpleNamespace

import numpy as np
import pytest

from temet.cosmo.cloudy import cloudyEmission, cloudyIon


def test_cloudy_line_luminosity_oxygen_abundance():
    line = "O  6 1031.91A"
    emission = cloudyEmission(None)
    emission.grid = {
        "redshift": np.array([0.0]),
        "dens": np.array([0.0]),
        "metal": np.array([0.0, 1.0]),
        "temp": np.array([5.5]),
    }
    emission.range = {name: (grid[0], grid[-1]) for name, grid in emission.grid.items()}
    # The synthetic table's emissivity is proportional to metallicity.
    emission.data = {line: emission.grid["metal"][None, None, :, None]}
    emission.redshiftInterp = True
    emission.order = 1

    solar_oxygen = cloudyIon(None)._solarMetalAbundanceMassRatio("O")
    fields = {
        "hdens_log": np.zeros(3),
        "temp_log": np.full(3, 5.5),
        "metal": np.array([2.5, 2.5, 20.0]),
        "metals_O": solar_oxygen * np.array([2.5, 3.0, 15.0]),
        "vol": np.ones(3),
    }
    snapshot = SimpleNamespace(
        eEOS=False,
        metals=["O"],
        redshift=0.0,
        snapshotSubset=lambda part_type, field, indRange: fields[field].copy(),
        snapHasField=lambda part_type, field: field == "GFM_Metals",
        units=SimpleNamespace(
            metallicityInSolar=lambda metal, log: np.log10(metal), codeVolumeToCm3=lambda volume: volume
        ),
    )

    luminosity = emission.calcGasLineLuminosity(snapshot, line)
    assert luminosity == pytest.approx([2.5, 3.0, 15.0])

    scaled_solar = emission.calcGasLineLuminosity(snapshot, line, solarAbunds=True)
    assert scaled_solar == pytest.approx([2.5, 2.5, 10.0])
