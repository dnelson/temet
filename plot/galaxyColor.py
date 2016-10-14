"""
galaxyColor.py
  Plots for TNG flagship paper: galaxy colors, color bimodality.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from util import simParams
from util.helper import loadColorTable, running_median, contourf, logZeroNaN
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.util import cenSatSubhaloIndices
from cosmo.stellarPop import loadSimGalColors

# global configuration (remove duplication with globalComp.py)
sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines
figsize = (16,9) # (8,6)
clean   = False  # make visually clean plots with less information

def _bandMagRange(bands):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    return mag_range

def simSubhaloQuantity(sP, quant):
    """ Return a 1D vector of size Nsubhalos, one quantity per subhalo as specified by the string 
    cQuant, wrapping any special loading or processing. Also return an appropriate label and range. """
    label = None

    if quant == 'mstar':
        # stellar mass (within 2r1/2stars) used for x-axis, so already returned in log
        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
        vals = sP.units.codeMassToLogMsun( gc['subhalos'][:,4] )

        label = 'M$_{\\rm \star}(<2r_{\star,1/2})$ [ log M$_{\\rm sun}$ ]'
        if clean: label = 'M$_{\\rm \star}$ [ log M$_{\\rm sun}$ ]'
        minMax = [9.0, 12.0] # log Mstar

    if quant == 'ssfr':
        # specific star formation rate (SFR and Mstar both within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
        mstar = sP.units.codeMassToMsun( gc['subhalos']['SubhaloMassInRadType'][:,4] )

        # fix mstar=0 values such that vals_raw is zero, which is then masked
        w = np.where(mstar == 0.0)[0]
        if len(w):
            mstar[w] = 1.0
            gc['subhalos']['SubhaloSFRinRad'][w] = 0.0

        vals = gc['subhalos']['SubhaloSFRinRad'] / mstar
        #vals[vals == 0.0] = vals[vals > 0.0].min() * 1.0 # set SFR=0 values

        label = 'log sSFR [ M$_\odot$ / yr ]'
        minMax = [-9.0,-12.0]

    if quant == 'Z_stars':
        # mass-weighted mean stellar metallicity (within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloStarMetallicity'])
        vals = sP.units.metallicityInSolar(gc['subhalos'])

        label = 'log ( Z$_{\\rm stars}$ / Z$_{\odot}$ )'
        minMax = [-2.0,1.0]

    if quant == 'Z_gas':
        # mass-weighted mean gas metallicity  (within 2r1/2stars)
        gc = groupCat(sP, fieldsSubhalos=['SubhaloGasMetallicity'])
        vals = sP.units.metallicityInSolar(gc['subhalos'])

        label = 'log ( Z$_{\\rm gas}$ / Z$_{\odot}$ )'
        minMax = [-2.0,1.0]



    assert label is not None
    return vals, label, minMax

def histo2D(sP, pdf, bands, xQuant='mstar', cenSatSelect='cen', cQuant=None, cStatistic=None, minCount=None):
    """ Make a 2D histogram of subhalos with some color on the y-axis, a property on the x-axis, 
    and optionally a third property as the colormap per bin. minCount specifies the minimum number of 
    points a bin must contain to show it as non-white. """
    assert cenSatSelect in ['all', 'cen', 'sat']
    assert cStatistic in [None,'mean','median','count','sum'] # or any user function

    # hard-coded config
    nBins          = 80
    simColorsModel = 'p07c_bc00dust' # snap, p07c_nodust, p07c_bc00dust

    cmap = loadColorTable('plasma')
    flag_color = '#000000' # for non-empty colorized bins which have e.g. median(roperty)==0

    if bands == ['u','i']: mag_range = [0.5,4.0]
    if bands == ['g','r']: mag_range = [0.15,0.85]
    if bands == ['r','i']: mag_range = [0.0,0.6]
    if bands == ['i','z']: mag_range = [0.0,0.4]

    # load fullbox galaxy properties and set plot options for x-axis
    sim_xvals, xlabel, xMinMax = simSubhaloQuantity(sP, xQuant)

    # load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )
        assert cStatistic in [None,'count']

        # overrides for density distribution
        cStatistic = 'count'
        cmap = loadColorTable('gray_r')

        clabel = 'log N$_{\\rm gal}$'
        cMinMax = [0.0,2.0]
    else:
        sim_cvals, clabel, cMinMax = simSubhaloQuantity(sP, cQuant)

    # load/calculate simulation colors for y-axis
    sim_colors = loadSimGalColors(sP, simColorsModel, bands=bands)

    # central/satellite selection?
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_colors = sim_colors[wSelect]

    # reduce to the subset with non-NaN galaxy colors (i.e. minimum 1 star particle)
    wFiniteColor = np.isfinite(sim_colors)
    sim_colors = sim_colors[wFiniteColor]
    sim_cvals  = sim_cvals[wFiniteColor]
    sim_xvals  = sim_xvals[wFiniteColor]

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('(%s-%s) color [ mag ]' % (bands[0],bands[1]))

    if not clean:
        print(' ','-'.join(bands),xQuant,cenSatSelect,cQuant,cStatistic,minCount)
        ax.set_title('stat=%s model=%s select=%s mincount=%s' % \
            (cStatistic,simColorsModel,cenSatSelect,minCount))

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, cStatistic, 
                                                 bins=nBins2D, range=[xMinMax,mag_range])

    # only show bins with a minimum number of points?
    if minCount is not None:
        nn, _, _, _ = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, 'count', 
                                          bins=nBins2D, range=[xMinMax,mag_range])

        cc[nn < minCount] = np.nan

    # for now: log on density and all color quantities
    cc2d = logZeroNaN(cc.T)

    # sSFR: mask bins with median==0 and map to special color 
    if cQuant == 'ssfr':
        # FINISH
        cmap.set_over(flag_color, 1.0) # set over values to gray (use over instead of bad)
        #cmap.set_under('r', 1.0) # set over values to gray (use over instead of bad)
        #cc2d = np.ma.masked_where(cc.T == 0.0, cc2d) # empty bins/bad stay at NaN which is transparent

    # method (A)
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    plt.imshow(cc2d, extent=extent, origin='lower', interpolation='nearest', aspect='auto', 
               cmap=cmap, norm=norm)

    # method (B) unused
    #reduceMap = {'mean':np.mean, 'median':np.median, 'count':np.size, 'sum':np.sum}
    #reduceFunc = reduceMap[cStatistic] if cStatistic in reduceMap else cStatistic
    #plt.hexbin(sim_xvals, sim_colors, C=None, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False)
    #plt.hexbin(sim_xvals, sim_colors, C=sim_cvals, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False, reduce_C_function=reduceFunc)

    # median line?
    if cQuant is None:
        binSizeMed = (xMinMax[1]-xMinMax[0]) / nBins * 2
        colorMed = 'orange'
        lwMed = 2.0

        xm, ym, sm, pm = running_median(sim_xvals,sim_colors,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        ym2 = savgol_filter(ym,sKn,sKo)
        sm2 = savgol_filter(sm,sKn,sKo)
        pm2 = savgol_filter(pm,sKn,sKo,axis=1)

        ax.plot(xm[:-1], ym2[:-1], '-', color=colorMed, lw=lwMed, label='median')

        ax.plot(xm[:-1], pm2[1,:-1], ':', color=colorMed, lw=lwMed, label='P[10,90]')
        ax.plot(xm[:-1], pm2[-2,:-1], ':', color=colorMed, lw=lwMed)

        if not clean:
            #ax.plot(xm[:-1], pm2[0,:-1], ':', color='red', lw=lwMed, label='P[5,95]')
            #ax.plot(xm[:-1], pm2[-1,:-1], ':', color='red', lw=lwMed)
            #ax.plot(xm[:-1], pm2[2,:-1], ':', color='green', lw=lwMed, label='IQR[25,75]')
            #ax.plot(xm[:-1], pm2[-3,:-1], ':', color='green', lw=lwMed)
            #ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='yellow', lw=lwMed, label='$\pm 1\sigma$')
            #ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='yellow', lw=lwMed)
            ax.legend(loc='lower right')

    # colorbar
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)

    # finish plot
    #fig.save('histo2d_%s_%s_%s_%s.pdf' % ('-'.join(bands),xQuant,cQuant,cenSatSelect))
    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def plots():
    """ Driver. """
    pdf = PdfPages('galaxyColor_2dhistos.pdf')
    sP = simParams(res=1820, run='tng', redshift=0.0)

    # plots
    #for cs in ['median','mean']:
    #    for css in ['cen','sat','all']:
    bands = ['g','r']
    xQuant = 'mstar'
    cs = 'median'
    css = 'cen'

    #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant=None)
    histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='ssfr', cStatistic=cs)
    #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_stars', cStatistic=cs)
    #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_gas', cStatistic=cs)

    pdf.close()
