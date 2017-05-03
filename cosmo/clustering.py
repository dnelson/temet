"""
clustering.py
  Calculations for TNG clustering paper.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import time
#from os.path import isfile, isdir, expanduser
#from os import mkdir
#from glob import glob

from cosmo.load import groupCat, groupCatHeader, auxCat
#from cosmo.util import correctPeriodicDistVecs, cenSatSubhaloIndices, snapNumToRedshift

def twoPointAutoCorrelationPeriodicCube():
    """ Calculate the two-point auto-correlation function in a periodic cube geometry. """
    pass
