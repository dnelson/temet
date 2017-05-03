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
from cosmo.clustering import twoPointAutoCorrelationPeriodicCube
from plot.general import simSubhaloQuantity, getWhiteBlackColors
from vis.common import setAxisColors
from plot.config import *

def galaxyTwoPoint(sPs, pdf, cenSatSelects=['all','cen','sat']):
    """ Plot the galaxy two-point correlation function for a run or multiple runs. """

    # config
    colorBounds  = None
    cType        = None #[['g','r'], defSimColorModel]
    mstarBounds  = None # [10.5, 15.0]
    mType        = None #'mstar2_log'

    # visual config
    rMinMax = [0.01, 100.0] # log Mpc
    yMinMax = [1, 100]

    rLabel = 'r [ Mpc ]'
    yLabel = '$\\xi(r \pm \Delta r)$  [ real space two-point autocorr ]'

    # load/calculate
    cfs = []
    for sP in sPs:
        for cenSatSelect in cenSatSelects:
            rad, xi = twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect=cenSatSelect, 
                        colorBounds=colorBounds, cType=cType, mstarBounds=mstarBounds, mType=mType)

            cfs.append({'rad':rad, 'xi':xi, 'css':cenSatSelect, 'sP':sP})

    # start plot
    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    sizefac = sfclean if clean else 1.0

    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
    ax = fig.add_subplot(111, axisbg=color1)
    setAxisColors(ax, color2)

    ax.set_xlim(rMinMax)
    #ax.set_ylim(yMinMax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(rLabel)
    ax.set_ylabel(yLabel)

    # plot
    for i, cf in enumerate(cfs):
        #c = ax._get_lines.prop_cycler.next()['color']
        xx = cf['sP'].units.codeLengthToMpc(cf['rad'])

        if len(cenSatSelects) == 1:
            label = cf['sP'].simName
        else:
            label = cf['sP'].simName + ' ' + cssLabels[cf['css']]

        l, = ax.plot(xx, cf['xi'], '-', label=label)
        #ax.fill_between(xx, yy*0.8, yy*1.1, color=l.get_color(), interpolate=True, alpha=0.2)

    # finish plot
    ax.legend()
    fig.tight_layout()
    pdf.savefig(facecolor=fig.get_facecolor())
    plt.close(fig)

def paperPlots():
    """ Construct all the final plots for the paper. """
    from util import simParams
    import plot.config
    plot.config.clean = True

    L75   = simParams(res=1820,run='tng',redshift=0.0)
    L205  = simParams(res=2500,run='tng',redshift=0.0)
    L75FP = simParams(res=1820,run='illustris',redshift=0.0)

    # figure 1
    if 1:
        #sPs = [L75]
        sPs = [simParams(res=910,run='tng',redshift=0.0)]

        pdf = PdfPages('figure1_tpcf_%s.pdf' % ('_'.join([sP.simName for sP in sPs])))
        galaxyTwoPoint(sPs, pdf)
        pdf.close()