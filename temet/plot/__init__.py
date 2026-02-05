"""Plotting routines."""

import logging
import pathlib
from importlib import resources

import matplotlib
import matplotlib.pyplot as plt

from . import (
    clustering,
    cosmoMisc,
    drivers,
    driversObs,
    driversSizes,
    meta,
    perf,
    quantities,
    snapshot,
    subhalos,
    subhalos_evo,
    util,
)


# set default (non-interactive) backend
matplotlib.use("Agg")

# set default style
style_path = pathlib.Path(__file__).parent.resolve()
plt.style.use(str(style_path / "mpl.style"))

# add fonts
try:
    font_path = resources.files("temet") / "tables" / "fonts"
    for font_file in font_path.iterdir():
        matplotlib.font_manager.fontManager.addfont(font_file)
except FileNotFoundError:
    pass

# disable fontTools timestamp warnings
logging.getLogger("fontTools.ttLib.tables").setLevel(logging.ERROR)
