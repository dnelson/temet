"""temet."""

__all__ = ["ICs", "catalog", "cosmo", "load", "obs", "plot", "spectra", "tracer", "util", "vis"]
# currently exclude "ML" to avoid torch* dependencies
# exclude "projects" as this is example/user code
from temet import ICs, catalog, cosmo, load, obs, plot, spectra, tracer, util, vis

from .util.simParams import simParams as sim


# check for data tables/ download
util.extern.check_data_tables()
