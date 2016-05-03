"""
tracerEvo.py
  Analysis and plotting of evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib as mpl
#import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.tracerMC import subhaloTracersTimeEvo
from util import simParams

def zoomDataDriver():
    """ Run and save data files for tracer evolution in several quantities of interest. """
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    subhaloID = sP.zoomSubhaloID

    trFields     = ['tracer_maxtemp'] 
    parFields    = ['rad_rvir','vrad','entr','temp','sfr']

    subhaloTracersTimeEvo(sP,subhaloID,trFields,parFields)

    