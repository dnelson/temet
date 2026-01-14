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

import pathlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

style_path = pathlib.Path(__file__).parent.resolve()
plt.style.use(str(style_path / 'mpl.style'))
