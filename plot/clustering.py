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
from collections import OrderedDict
#from scipy.signal import savgol_filter

#from util import simParams
from util.helper import running_median, logZeroNaN, sampleColorTable
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, groupCatSingle, groupCatHeader
from cosmo.clustering import twoPointAutoCorrelationPeriodicCube
from plot.general import simSubhaloQuantity, getWhiteBlackColors
from vis.common import setAxisColors
from plot.config import *

def galaxyTwoPoint(sPs, saveBase='', cenSatSelects=['all','cen','sat'], 
                   colorBin=None, cType=None, mstarBin=None, mType=None):
    """ Plot the galaxy two-point correlation function for a run or multiple runs. """

    # visual config
    rMinMax = [0.01, 100.0] # log Mpc
    yMinMax = [1e-2, 5e4]
    lw = 2.5

    rLabel = 'r [ Mpc ]'
    yLabel = '$\\xi(r \pm \Delta r)$  [ real space two-point autocorr ]'

    # load/calculate
    cfs = []
    for sP in sPs:
        for cenSatSelect in cenSatSelects:
            rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect=cenSatSelect, 
                                   colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType)

            cfs.append({'rad':rad, 'xi':xi, 'xi_err':xi_err, 'css':cenSatSelect, 'sP':sP})

    # start plot
    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    sizefac = sfclean if clean else 1.0

    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
    ax = fig.add_subplot(111, axisbg=color1)
    setAxisColors(ax, color2)

    ax.set_xlim(rMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(rLabel)
    ax.set_ylabel(yLabel)

    # plot
    for i, cf in enumerate(cfs):
        xx = cf['sP'].units.codeLengthToMpc(cf['rad'])
        ww = np.where( cf['xi'] > 0.0 )

        label = cf['sP'].simName
        if len(cenSatSelects) > 1: label += ' ' + cssLabels[cf['css']]
        if cf['sP'].redshift > 0.0: label += ' z = %.1f' % cf['sP'].redshift

        l, = ax.plot(xx, cf['xi'], '-', lw=lw, label=label)

        yy0 = yFac*(cf['xi'][ww] - cf['xi_err'][ww]/2)
        yy1 = yFac*(cf['xi'][ww] + cf['xi_err'][ww]/2)

        ax.fill_between(xx[ww], yy0, yy1, color=l.get_color(), interpolate=True, alpha=0.15)

    # finish plot
    ax.legend()
    fig.tight_layout()
    fig.savefig('%stpcf_%s.pdf' % (saveBase,'_'.join([sP.simName for sP in sPs])), 
        facecolor=fig.get_facecolor())
    plt.close(fig)

def galaxyTwoPointQuantBounds(sPs, saveBase='', cenSatSelect='all', 
                              colorBins=None, cType=None, mstarBins=None, mType=None, redshiftBins=None):
    """ Plot the galaxy two-point correlation function for a run or multiple runs, showing a range 
    of bins in either color, stellar mass, or redshift (choose one). """
    if colorBins is not None and mstarBins is not None:
        assert len(colorBins) == 1 or len(mstarBins) == 1 # only one of the two can vary
    assert redshiftBins is None # not implemented yet

    # visual config
    rMinMax = [0.01, 100.0] # log Mpc
    yMinMaxes = [[1e-2, 1e5], [5e-1, 5e3], [1e0, 1e3]]
    ctName = 'jet'
    lw = 2.5

    rLabel = 'r [ Mpc ]'
    yLabel = '$\\xi(r)$  [ real space two-point autocorr ]'

    # load/calculate
    cfs = OrderedDict()

    loadByColor = False
    loadByMass = False
    if colorBins is not None and len(colorBins) > 1:
        loadByColor = True
    if not loadByColor: loadByMass
    
    if loadByColor:
        # can specify no mstarBin, or a single mstarBin, within which these color bins are applied
        mstarBin = mstarBins[0] if mstarBins is not None else mstarBins

        for colorBin in colorBins:
            label = '%.1f < (%s-%s) < %.1f' % (colorBin[0], cType[0][0], cType[0][1], colorBin[1])
            cfs[label] = []

            for sP in sPs:
                rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect=cenSatSelect, 
                                       colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType)

                cfs[label].append({'rad':rad, 'xi':xi, 'xi_err':xi, 'sP':sP})

    if loadByMass:
        # can specify no colorBin, or a single colorBin, within which these mass bins are applied
        colorBin = colorBins[0] if colorBins is not None else colorBins

        for mstarBin in mstarBins:
            label = '%4.1f < log($M_\star$/M$_{\\rm sun}$) < %4.1f' % (mstarBin[0], mstarBin[1])
            cfs[label] = []

            for sP in sPs:
                rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect=cenSatSelect, 
                                       colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType)

                cfs[label].append({'rad':rad, 'xi':xi, 'xi_err':xi, 'sP':sP})

    if redshiftBins is not None:
        assert 0

    #cm = sampleColorTable( ctName, np.sum([len(cfs[k]) for k in cfs.keys()]) )

    # iterate over y-axes: xi(r), r*xi(r), r^2*xi(r)
    for iterNum in [0,1,2]:
        # start plot
        color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
        sizefac = sfclean if clean else 1.0

        fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac),facecolor=color1)
        ax = fig.add_subplot(111, axisbg=color1)
        setAxisColors(ax, color2)

        ax.set_xlim(rMinMax)
        ax.set_ylim(yMinMaxes[iterNum])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(rLabel)
        ax.set_ylabel(['','r','r$^2$'][iterNum] + yLabel)

        # plot: loop over each bin
        for k, cfBoundSet in enumerate(cfs.keys()):
            c = ax._get_lines.prop_cycler.next()['color']

            # loop over each run for this bin
            for i, cf in enumerate(cfs[cfBoundSet]):
                xx = cf['sP'].units.codeLengthToMpc(cf['rad'])
                ww = np.where( cf['xi'] > 0.0 )

                label = cfBoundSet if i == 0 else ''

                # y-axis multiplier
                yFac = 1.0
                if iterNum == 1: yFac = xx[ww]
                if iterNum == 2: yFac = xx[ww]**2

                l, = ax.plot(xx[ww], yFac*cf['xi'][ww], lw=lw, linestyle=linestyles[i], label=label, color=c)

                yy0 = yFac*(cf['xi'][ww] - cf['xi_err'][ww]/2)
                yy1 = yFac*(cf['xi'][ww] + cf['xi_err'][ww]/2)

                if i == 0:
                    ax.fill_between(xx[ww], yy0, yy1, color=l.get_color(), interpolate=True, alpha=0.15)

        # legends
        loc = 'lower left' if iterNum < 2 else 'upper left'
        legend1 = ax.legend(loc=loc)
        ax.add_artist(legend1)

        if len(sPs) > 0:
            handles, labels = [], []
            for i, sP in enumerate(sPs):
                handles.append(plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[i]) )
                labels.append(sP.simName)

            legend2 = ax.legend(handles, labels, loc='upper right')
            ax.add_artist(legend2)

        # finish plot
        fig.tight_layout()
        fig.savefig('%stpcf_%s_%s.pdf' % (saveBase,['xi','rxi','r2xi'][iterNum],
            '_'.join([sP.simName for sP in sPs])), facecolor=fig.get_facecolor())
        plt.close(fig)

def paperPlots():
    """ Construct all the final plots for the paper. """
    from util import simParams
    import plot.config
    plot.config.clean = True

    L75   = simParams(res=1820,run='tng',redshift=0.0)
    L205  = simParams(res=2500,run='tng',redshift=0.0)
    L75FP = simParams(res=1820,run='illustris',redshift=0.0)

    # unless we are exploring in mass bins, we always apply the following (resolution) cut:
    mstarBinDef = [9.0, 13.0]
    mTypeDef = 'mstar2_log'

    # figure 1: TNG100 split by cen/sat/all
    if 0:
        sPs = [L75]
        sPs = [simParams(res=455,run='tng',redshift=0.0)]
        saveBase = 'figure1_'

        galaxyTwoPoint(sPs, saveBase=saveBase, cenSatSelects=['all','cen','sat'], 
          colorBin=None, cType=None, mstarBin=mstarBinDef, mType=mTypeDef)

    # figure 2: TNG300 in stellar mass bins at z=0
    if 0:
        sPs = [L205]
        saveBase = 'figure2_massbins_'

        mstarBins = [[9.5,10.0], [10.5,11.0], [11.5,12.0], [12.0,12.5]]
        mType = 'mstar2_log'

        galaxyTwoPointQuantBounds(sPs, saveBase=saveBase, cenSatSelect='all', 
          colorBins=None, cType=None, mstarBins=mstarBins, mType=mType)

    # figure 3: color bins at z=0
    if 1:
        sPs = [L205] # L75

        colorBins = [[0.2,0.3], [0.3, 0.4], [0.7,1.0]]
        cType = [['g','r'], defSimColorModel]
        saveBase = 'figure3_color_%s_%s_' % (''.join(cType[0]),cType[1])

        galaxyTwoPointQuantBounds(sPs, saveBase=saveBase, cenSatSelect='all', 
          colorBins=colorBins, cType=cType, mstarBins=[mstarBinDef], mType=mTypeDef)

    # figure 4: redshift evolution
    if 0:
        sPs = []
        for redshift in [0.0, 0.5, 1.0, 2.0]:
            sPs.append( simParams(res=1820, run='tng', redshift=redshift) )

        saveBase = 'figure4_'

        galaxyTwoPoint(sPs, saveBase=saveBase, cenSatSelects=['all'], 
          colorBin=None, cType=None, mstarBin=mstarBinDef, mType=mTypeDef)