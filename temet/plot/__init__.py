"""Plotting routines."""

import pathlib

import matplotlib
import matplotlib.pyplot as plt

from . import clustering, cosmoMisc, drivers, driversObs, driversSizes, meta, perf, quantities, snapshot, subhalos, util


matplotlib.use("Agg")

style_path = pathlib.Path(__file__).parent.resolve()
plt.style.use(str(style_path / "mpl.style"))
