""" Plotting routines. """
#from . import clustering
from . import cosmoGeneral
from . import cosmoMisc
from . import general
from . import globalComp
from . import meta
from . import perf
from . import quantities
from . import sizes
from .config import *

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.style.use('temet/plot/temet.mplstyle')
