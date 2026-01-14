""" Test simParams and other sim-level functionality. """

from temet.util import simParams as sim

def test_sim_path():
    # instantiation with explicit path
    x = sim('/virgotng/universe/IllustrisTNG/TNG50-1/')
    assert x.omega_m == 0.3089

def test_sim_name():
    # instantiation with name
    x = sim('Illustris-1')
    assert x.boxSize == 75000.0

def test_sim_redshift():
    # instantiation with redshift (generates cached mapping file)
    x = sim('TNG50-4', redshift=0.0)
    assert x.snap == 99
