# add our local path on prior to system path
# otherwise, even if we add ~/.local/ to PYTHONPATH first,
# np is e.g. always loaded from the system-wide version
import sys
import platform
from os.path import expanduser

# depreciated, can be removed: (move to having site-packages inside .local/envs/dylan3 for python3.x)
#if 'haswell-login' in platform.node():
#    print('on haswell')
#    sys.path.insert(0,expanduser("~")+'/.local_haswell/lib/python2.7/site-packages/')
#else:
#    sys.path.insert(0,expanduser("~")+'/.local/lib/python2.7/site-packages/')

# libraries
import numpy as np
#import matplotlib.pyplot as plt
import h5py

# general
import illustris_python as il

# dnelson
import ArepoVTK
import ICs
import cosmo
import plot
import tracer
import util
import obs
import vis
import projects
from util import simParams
