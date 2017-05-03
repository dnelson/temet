"""
clustering.py
  Plots for TNG clustering.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy.signal import savgol_filter

#from util import simParams
from util.helper import running_median, logZeroNaN
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, groupCatSingle, groupCatHeader
from plot.general import simSubhaloQuantity, getWhiteBlackColors
from vis.common import setAxisColors
from plot.config import *

def galaxyTwoPoints(sPs, pdf):
    """ Plot a single galaxy two-point correlation function. """

    # visual config
    rMinMax = [-2.0, 2.0] # log Mpc
    yMinMax = [1, 100]

    rLabel = 'r [ Mpc ]'
    yLabel = '$\xi(r)$ [ galaxy two-point 3D real-space autocorrelation ]'

    # start plot
    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    sizefac = sfclean if clean else 1.0

    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
    ax = fig.add_subplot(111, axisbg=color1)
    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xlabel(rLabel)
    ax.set_ylabel(yLabel)

    # plot


    # finish plot
    ax.legend()
    fig.tight_layout()
    pdf.savefig(facecolor=fig.get_facecolor())
    plt.close(fig)

def paperPlots():
    """ Construct all the final plots for the paper. """
    import plot.config
    plot.config.clean = True

    L75   = simParams(res=1820,run='tng',redshift=0.0)
    L205  = simParams(res=2500,run='tng',redshift=0.0)
    L75FP = simParams(res=1820,run='illustris',redshift=0.0)

    # figure 1
    if 1:
        sPs = [L75]

        pdf = PdfPages('figure1_tpcf_%s.pdf' % ('_'.join([sP.simName for sP in sPs])))
        galaxyTwoPoint(sPs, pdf)
        pdf.close()