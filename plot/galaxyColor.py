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
from matplotlib.colors import Normalize, LogNorm, colorConverter
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from util import simParams
from util.helper import loadColorTable, running_median, contourf, logZeroSafe, logZeroNaN
from cosmo.util import cenSatSubhaloIndices
from cosmo.stellarPop import loadSimGalColors
from plot.general import simSubhaloQuantity

# global configuration (remove duplication with globalComp.py)
sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines
figsize = (14,10) # (8,6)
clean   = False  # make visually clean plots with less information

def _bandMagRange(bands):
    """ Hard-code some band dependent magnitude ranges. """
    if bands[0] == 'u' and bands[1] == 'i': mag_range = [0.5,4.0]
    if bands[0] == 'g' and bands[1] == 'r': mag_range = [0.0,1.0]
    if bands[0] == 'r' and bands[1] == 'i': mag_range = [0.0,0.6]
    if bands[0] == 'i' and bands[1] == 'z': mag_range = [0.0,0.4]
    return mag_range

def histo2D(sP, pdf, bands, xQuant='mstar2_log', cenSatSelect='cen', cQuant=None, cStatistic=None, 
            minCount=None, simColorsModel='p07c_bc00dust'):
    """ Make a 2D histogram of subhalos with some color on the y-axis, a property on the x-axis, 
    and optionally a third property as the colormap per bin. minCount specifies the minimum number of 
    points a bin must contain to show it as non-white. """
    assert cenSatSelect in ['all', 'cen', 'sat']
    assert cStatistic in [None,'mean','median','count','sum'] # or any user function
    assert simColorsModel in ['snap','p07c_nodust','p07c_bc00dust']

    # hard-coded config
    nBins = 80

    cmap = loadColorTable('viridis') # plasma

    if bands == ['u','i']: mag_range = [0.5,4.0]
    if bands == ['g','r']: mag_range = [0.15,0.85]
    if bands == ['r','i']: mag_range = [0.0,0.6]
    if bands == ['i','z']: mag_range = [0.0,0.4]

    # x-axis: load fullbox galaxy properties and set plot options
    sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant, clean)
    if xMinMax[0] > xMinMax[1]: xMinMax = xMinMax[::-1] # reverse

    # y-axis: load/calculate simulation colors
    sim_colors = loadSimGalColors(sP, simColorsModel, bands=bands)

    ylabel = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
    if not clean: ylabel += ' %s' % simColorsModel

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )
        assert cStatistic in [None,'count']

        # overrides for density distribution
        cStatistic = 'count'
        cmap = loadColorTable('gray_r')

        clabel = 'log N$_{\\rm gal}$'
        cMinMax = [0.0,2.0]
    else:
        sim_cvals, clabel, cMinMax, cLog = simSubhaloQuantity(sP, cQuant, clean)

    # central/satellite selection?
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_colors = sim_colors[wSelect]

    # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
    wFiniteColor = np.isfinite(sim_colors) #& np.isfinite(sim_xvals)
    sim_colors = sim_colors[wFiniteColor]
    sim_cvals  = sim_cvals[wFiniteColor]
    sim_xvals  = sim_xvals[wFiniteColor]

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if not clean:
        print(' ','-'.join(bands),simColorsModel,xQuant,cenSatSelect,cQuant,cStatistic,minCount)
        ax.set_title('stat=%s select=%s mincount=%s' % (cStatistic,cenSatSelect,minCount))

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],mag_range[0],mag_range[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    if '_log' not in xQuant: sim_xvals = logZeroSafe(sim_xvals) # xMinMax always corresponds to log values

    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, cStatistic, 
                                                 bins=nBins2D, range=[xMinMax,mag_range])

    cc = cc.T # imshow convention

    # only show bins with a minimum number of points?
    if minCount is not None:
        nn, _, _, _ = binned_statistic_2d(sim_xvals, sim_colors, sim_cvals, 'count', 
                                          bins=nBins2D, range=[xMinMax,mag_range])

        cc[nn.T < minCount] = np.nan

    # for now: log on density and all color quantities
    cc2d = cc
    if cQuant is None or cLog is True:
        cc2d = logZeroNaN(cc)

    # normalize and color map
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap(norm(cc2d))

    # mask bins with median==0 and map to special color, which right now have been set to log10(0)=NaN
    if cQuant is not None:
        cc2d_rgb[cc == 0.0,:] = colorConverter.to_rgba('#dddddd')

    # mask bins with median==NaN (empty) to white
    cc2d_rgb[~np.isfinite(cc),:] = colorConverter.to_rgba('#ffffff')

    plt.imshow(cc2d_rgb, extent=extent, origin='lower', interpolation='nearest', aspect='auto', 
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

    # finish plot and save
    fig.tight_layout()
    if pdf is not None:
        pdf.savefig()
    else:
        fig.save('histo2d_%s_%s_%s_%s_%s_%s_%s.pdf' % \
            ('-'.join(bands),simColorsModel,xQuant,cQuant,cStatistic,cenSatSelect,minCount))
    plt.close(fig)

def plots():
    """ Driver. """
    sP = simParams(res=1820, run='tng', redshift=0.0)

    # debug:
    #pdf = PdfPages('galaxyColor_2dhistos.pdf')
    #bands = ['g','r']
    #xQuant = 'mstar2_log'
    #cs = 'median'
    #css = 'cen'
    #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='ssfr', cStatistic=cs)
    #pdf.close()
    #import pdb; pdb.set_trace()

    # plots
    #for cs in ['median','mean']:
    for css in ['cen','sat','all']:
        bands = ['g','r']
        xQuant = 'mstar2_log'
        cs = 'median'
        #css = 'cen'

        pdf = PdfPages('galaxyColor_2dhistos_%s_%s_%s_%s.pdf' % (''.join(bands),xQuant,cs,css))

        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant=None)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='ssfr', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_stars', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='Z_gas', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='size_stars', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='size_gas', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fgas1', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fgas2', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='mgas2', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='mgas1', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='stellarage', cStatistic=cs)

        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_mm5', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_ma5', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='zform_poly7', cStatistic=cs)

        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_all_eps07o', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_all_eps07m', cStatistic=cs)
        #histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_10re_eps07o', cStatistic=cs)
        histo2D(sP, pdf, bands, xQuant=xQuant, cenSatSelect=css, cQuant='fcirc_10re_eps07m', cStatistic=cs)

        #histo2D(sP, pdf, bands, xQuant='ssfr', cenSatSelect=css, cQuant=None)
        #histo2D(sP, pdf, bands, xQuant='ssfr', cenSatSelect=css, cQuant='fgas2', cStatistic=cs)

        pdf.close()

def viewingAngleVariation():
    """ Demo of Nside>>1 results for variation of (one or two) galaxy colors as a function of 
    viewing angle. 1D Histogram. """

    # config
    nBins = 100

    sP = simParams(res=1820, run='tng', redshift=0.0)

    ac_nodust = 'p07c_cf00dust'
    ac_demo   = 'p07c_ns8_demo'

    bands = ['g','r']

    # load
    ac = cosmo.load.auxCat(sP, fields=['Subhalo_StellarPhot_'+ac_demo])
    demo_ids = ac['Subhalo_StellarPhot_'+ac_demo+'_attrs']['subhaloIDs']

    nodust_colors = loadSimGalColors(sP, ac_nodust, bands=bands)
    demo_colors   = loadSimGalColors(sP, ac_demo, bands=bands)

    import pdb; pdb.set_trace()

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    mag_range = _bandMagRange(bands)
    ax.set_xlim(mag_range)
    ax.set_ylim([0,1])
    ax.set_xlabel('(%s-%s) color [ mag ]' % (bands[0],bands[1]))
    ax.set_ylabel('PDF $\int=1$')

    # histogram demo color distribution
    #for i in range(demo_colors.shape[0]):
    yy, xx = np.histogram(demo_colors, bins=nBins, range=mag_range, density=True)
    xx = xx[:-1] + 0.5*(mag_range[1]-mag_range[0])/nBins
    l, = ax.plot(xx, yy, label='', lw=2.5)

    # overplot nodust color
    #ax.plot()

    # finish plot and save
    fig.tight_layout()
    fig.save('viewing_angle_variation.pdf')
    plt.close(fig)
    