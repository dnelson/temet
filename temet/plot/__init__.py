""" Plotting routines. """
from . import clustering
from . import cosmoMisc
from . import drivers
from . import driversObs
from . import driversSizes
from . import meta
from . import perf
from . import quantities
from . import snapshot
from . import subhalos
from .config import *

import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

style_path = pathlib.Path(__file__).parent.resolve()
plt.style.use(str(style_path / 'mpl.style'))
