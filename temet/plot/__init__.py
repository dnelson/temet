"""Plotting routines."""

import logging
import pathlib
import sys
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


# check if we are in a Jupyter notebook
def in_notebook():
    """Determine if we are inside a Jupyter notebook (or similar), or not."""
    if "IPython" in sys.modules:
        IPython = sys.modules["IPython"]
        ipython = IPython.core.getipython.get_ipython()

        try:
            shell = ipython.__class__.__name__
            if shell == "ZMQInteractiveShell":
                return True  # Jupyter notebook or qtconsole
            elif shell == "TerminalInteractiveShell":
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False  # Probably standard Python interpreter

    return False  # no ipython


in_notebook = in_notebook()

# set default (non-interactive) backend
if not in_notebook:
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
