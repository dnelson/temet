"""
cosmoGeneral.py
  Fully generalized plots and general plot helpers related to cosmological boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm, colorConverter
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, expanduser
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from util import simParams
from util.helper import running_median, running_median_sub, logZeroNaN, loadColorTable, getWhiteBlackColors, sampleColorTable
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, groupCatSingle, snapshotSubset
from cosmo.color import loadSimGalColors, calcMstarColor2dKDE
from vis.common import setAxisColors
from plot.quantities import quantList, simSubhaloQuantity, simParticleQuantity
from plot.config import *

def addRedshiftAxis(ax, sP, zVals=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,10.0]):
    """ Add a redshift axis as a second x-axis on top (assuming bottom axis is Age of Universe [Gyr]). """
    axTop = ax.twiny()
    axTickVals = sP.units.redshiftToAgeFlat( np.array(zVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(zVals)
    axTop.set_xlabel("Redshift")

def addUniverseAgeAxis(ax, sP, ageVals=[0.7,1.0,1.5,2.0,3.0,4.0,6.0,9.0]):
    """ Add a age of the universe [Gyr] axis as a second x-axis on top (assuming bottom is redshift). """
    axTop = ax.twiny()

    ageVals.append( sP.units.redshiftToAgeFlat([0.0]).round(2) )
    axTickVals = sP.units.ageFlatToRedshift( np.array(ageVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(ageVals)
    axTop.set_xlabel("Age of the Universe [Gyr]")

def addRedshiftAgeAxes(ax, sP, xrange=[-1e-4,8.0], xlog=True):
    """ Add bottom vs. redshift (and top vs. universe age) axis for standard X vs. redshift plots. """
    ax.set_xlim(xrange)
    ax.set_xlabel('Redshift')

    if xlog:
        ax.set_xscale('symlog')
        zVals = [0,0.5,1,1.5,2,3,4,5,6,7,8] # [10]
    else:
        ax.set_xscale('linear')
        zVals = [0,1,2,3,4,5,6,7,8]

    ax.set_xticks(zVals)
    ax.set_xticklabels(zVals)

    addUniverseAgeAxis(ax, sP)

def tngModel_chi(M_BH):
    """ Return chi(M_BH) for the fiducial TNG model parameters. M_BH in Msun. """
    chi0 = 0.002
    beta = 2.0
    chi_max = 0.1

    chi_bh = np.clip( chi0 * (M_BH / 1e8)**beta, 0.0, chi_max )

    return chi_bh

# ------------------------------------------------------------------------------------------------------

def quantHisto2D(sP, pdf, yQuant, xQuant='mstar2_log', cenSatSelect='cen', cQuant=None, 
                 cStatistic=None, minCount=None, fig_subplot=[None,None], pStyle='white'):
    """ Make a 2D histogram of subhalos with one quantity on the y-axis, another property on the x-axis, 
    and optionally a third property as the colormap per bin. minCount specifies the minimum number of 
    points a bin must contain to show it as non-white. If '_nan' is not in cStatistic, then by default, 
    empty bins are white, and bins whose cStatistic is NaN (e.g. any NaNs in bin) are gray. Or, if 
    '_nan' is in cStatistic, then empty bins remain white, while the cStatistic for bins with any 
    non-NaN values is computed ignoring NaNs (e.g. np.nanmean() instead of np.mean()), and bins 
    which are non-empty but contain only NaN values are gray. """
    assert cenSatSelect in ['all', 'cen', 'sat']
    assert cStatistic in [None,'mean','median','count','sum','median_nan'] # or any user function

    # hard-coded config
    nBins    = 80
    cmap     = loadColorTable('viridis') # plasma
    colorMed = 'black'
    lwMed    = 2.0

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    medianLine = False
    colorContours = False

    if cenSatSelect == 'cen':
        medianLine = True
    if cQuant is None:
        colorMed = 'orange'

    #if cenSatSelect == 'cen' and 'color_' in yQuant: # tng_colors paper, not tng_oxygen paper
    #    colorContours = True

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant, clean)
    if xMinMax[0] > xMinMax[1]: xMinMax = xMinMax[::-1] # reverse

    # y-axis: load/calculate simulation colors, cached in sP.data
    sim_yvals, ylabel, yMinMax, yLog = simSubhaloQuantity(sP, yQuant, clean, tight=True)
    if yLog is True: sim_yvals = logZeroNaN(sim_yvals)

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )

        # overrides for density distribution
        cStatistic = 'count'
        ctName = 'gray_r' if pStyle == 'white' else 'gray'
        cmap = loadColorTable(ctName)

        clabel = 'log N$_{\\rm gal}$'
        cMinMax = [0.0,2.0]
        if sP.boxSize > 100000: cMinMax = [0.0,2.5]
    else:
        sim_cvals, clabel, cMinMax, cLog = simSubhaloQuantity(sP, cQuant, clean, tight=False)
        if yQuant == 'color_C_gr': print('Warning: to reproduce TNG colors paper, set tight=True maybe.')

    if sim_cvals is None:
        return # property is not calculated for this run (e.g. expensive auxCat)

    # central/satellite selection?
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_yvals = sim_yvals[wSelect]

    # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
    wFiniteColor = np.isfinite(sim_yvals) #& np.isfinite(sim_xvals)
    sim_yvals = sim_yvals[wFiniteColor]
    sim_cvals = sim_cvals[wFiniteColor]
    sim_xvals = sim_xvals[wFiniteColor]

    # _nan cStatistic? separate points into two sets
    nanFlag = False
    if '_nan' in cStatistic:
        nanFlag = True

        wFiniteCval = np.isfinite(sim_cvals)
        wNaNCval = np.isnan(sim_cvals)
        wInfCval = np.isinf(sim_cvals)

        if np.count_nonzero(wInfCval) > 0: # unusual
            print(' warning: [%d] infinite color values [%s].' % (np.count_nonzero(wInfCval),cQuant))

        assert np.count_nonzero(wFiniteCval) + np.count_nonzero(wNaNCval) + \
               np.count_nonzero(wInfCval) == sim_cvals.size

        # save points with NaN cvals
        sim_yvals_nan = sim_yvals[wNaNCval]
        sim_cvals_nan = sim_cvals[wNaNCval]
        sim_xvals_nan = sim_xvals[wNaNCval]

        # override default binning to only points with finite cvals
        sim_yvals = sim_yvals[wFiniteCval]
        sim_cvals = sim_cvals[wFiniteCval]
        sim_xvals = sim_xvals[wFiniteCval]

        # replace cStatistic string
        cStatistic = cStatistic.split("_nan")[0]

    # start plot
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize,facecolor=color1)
        ax = fig.add_subplot(111, axisbg=color1)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1], axisbg=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    print(' ',cQuant,sP.simName,ylabel,xQuant,cenSatSelect,cStatistic,minCount)
    if not clean:
        pass
        #ax.set_title('stat=%s select=%s mincount=%s' % (cStatistic,cenSatSelect,minCount))
    else:
        if cQuant is None:
            cssStrings = {'all':'all galaxies', 'cen':'centrals only', 'sat':'satellites'}
            ax.set_title(sP.simName + ': ' + cssStrings[cenSatSelect])

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    if '_log' not in xQuant: sim_xvals = logZeroNaN(sim_xvals) # xMinMax always corresponds to log values

    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_yvals, sim_cvals, cStatistic, 
                                                 bins=nBins2D, range=[xMinMax,yMinMax])

    cc = cc.T # imshow convention

    # only show bins with a minimum number of points?
    nn, _, _, _ = binned_statistic_2d(sim_xvals, sim_yvals, sim_cvals, 'count', 
                                      bins=nBins2D, range=[xMinMax,yMinMax])
    nn = nn.T

    if minCount is not None:
        cc[nn < minCount] = np.nan

    # for now: log on density and all color quantities
    cc2d = cc
    if cQuant is None or cLog is True:
        cc2d = logZeroNaN(cc)

    # normalize and color map
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap(norm(cc2d))

    # mask bins with median==0 and map to special color, which right now have been set to log10(0)=NaN
    if cQuant is not None:
        cc2d_rgb[cc == 0.0,:] = colorConverter.to_rgba(color4)

    if nanFlag:
        # bin NaN point set counts
        nn_nan, _, _, _ = binned_statistic_2d(sim_xvals_nan, sim_yvals_nan, sim_cvals_nan, 'count', 
                                              bins=nBins2D, range=[xMinMax,yMinMax])
        nn_nan = nn_nan.T        

        # flag bins with nn_nan>0 and nn==0 (only NaNs in bin) as second gray color
        cc2d_rgb[ ((nn_nan > 0) & (nn == 0)), :] = colorConverter.to_rgba(color3)

        nn += nn_nan # accumulate total counts
    else:
        # mask bins with median==NaN (nonzero number of NaNs in bin) to gray
        cc2d_rgb[~np.isfinite(cc),:] = colorConverter.to_rgba(color3)

    # mask empty bins to white
    cc2d_rgb[(nn == 0),:] = colorConverter.to_rgba(color1)

    # plot
    plt.imshow(cc2d_rgb, extent=extent, origin='lower', interpolation='nearest', aspect='auto', 
               cmap=cmap, norm=norm)

    # method (B) unused
    #reduceMap = {'mean':np.mean, 'median':np.median, 'count':np.size, 'sum':np.sum}
    #reduceFunc = reduceMap[cStatistic] if cStatistic in reduceMap else cStatistic
    #plt.hexbin(sim_xvals, sim_yvals, C=None, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False)
    #plt.hexbin(sim_xvals, sim_yvals, C=sim_cvals, gridsize=nBins, extent=extent, bins='log', 
    #          mincnt=minCount, cmap=cmap, marginals=False, reduce_C_function=reduceFunc)

    # median line?
    if medianLine:
        binSizeMed = (xMinMax[1]-xMinMax[0]) / nBins * 2

        xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)
            sm = savgol_filter(sm,sKn,sKo)
            pm = savgol_filter(pm,sKn,sKo,axis=1)

        ax.plot(xm[:-1], ym[:-1], '-', color=colorMed, lw=lwMed, label='median')

        ax.plot(xm[:-1], pm[1,:-1], ':', color=colorMed, lw=lwMed, label='P[10,90]')
        ax.plot(xm[:-1], pm[-2,:-1], ':', color=colorMed, lw=lwMed)

        if not clean:
            l = ax.legend(loc='lower right')
            for text in l.get_texts(): text.set_color(color2)

    # contours?
    if colorContours:        
        extent = [xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]]
        cLevels = [0.75,0.95]
        cAlphas = [0.5,0.8]

        # determine which color model/bands are requested
        _, model, bands = yQuant.split("_")
        simColorsModel = colorModelNames[model]
        bands = [bands[0],bands[1]]

        xx, yy, kde_sim = calcMstarColor2dKDE(bands, sim_xvals, sim_yvals, xMinMax, yMinMax, 
                                              sP=sP, simColorsModel=simColorsModel)

        for k in range(kde_sim.shape[0]):
            kde_sim[k,:] /= kde_sim[k,:].max() # by column normalization

        for k, cLevel in enumerate(cLevels):
            ax.contour(xx, yy, kde_sim, [cLevel], colors=[color2], 
                       alpha=cAlphas[k], linewidths=lwMed, extent=extent)

    # special behaviors
    if yQuant == 'size_gas':
        # add virial radius median line
        aux_yvals, _, _, _ = simSubhaloQuantity(sP, 'rhalo_200_log', clean)
        aux_yvals = aux_yvals[wSelect][wFiniteColor]
        if nanFlag: aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,aux_yvals,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed)
        ax.text(9.2, 2.06, 'Halo $R_{\\rm 200,crit}$', color=color, size=15)

    if yQuant in ['temp_halo','temp_halo_volwt']:
        # add virial temperature median line
        aux_yvals = groupCat(sP, fieldsSubhalos=['tvir_log'])
        aux_yvals = aux_yvals[wSelect][wFiniteColor]
        if nanFlag: aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,aux_yvals,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed)
        ax.text(9.35, 5.55, 'Halo $T_{\\rm vir}$', color=color, size=15)

    if yQuant == 'fgas_r200':
        # add constant f_b line
        f_b = np.log10(sP.units.f_b)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xMinMax, [f_b,f_b], '--', color=color, lw=lwMed)
        ax.text(11.4, f_b+0.05, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', color=color, size=17)

    if yQuant in ['BH_CumEgy_low','BH_CumEgy_high']:
        # add approximate halo binding energy line = (3/5)*GM^2/R
        G = sP.units.G / 1e10 # kpc (km/s)**2 / msun
        r_halo, _, _, _ = simSubhaloQuantity(sP, 'rhalo_200', clean) # pkpc
        m_halo, _, _, _ = simSubhaloQuantity(sP, 'mhalo_200', clean) # msun
        e_b = (3.0/5.0) * G * m_halo**2 * sP.units.f_b / r_halo # (km/s)**2 * msun
        e_b = np.array(e_b, dtype='float64') * 1e10 * sP.units.Msun_in_g # cm^2/s^2 * g
        e_b = logZeroNaN(e_b).astype('float32') # log(cm^2/s^2 * g)
        e_b = e_b[wSelect][wFiniteColor]
        if nanFlag: e_b = e_b[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,e_b,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed)
        ax.text(9.2, 57.5, 'Halo $E_{\\rm B}$', color=color, size=17)

    # colorbar
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel, color=color2)
    cb.outline.set_edgecolor(color2)
    cb.ax.yaxis.set_tick_params(color=color2)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=color2)

    # finish plot and save
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig(facecolor=fig.get_facecolor())
        else:
            fig.savefig('histo2d_%s_%s_%s_%s_%s_%s.pdf' % \
                (ylabel,xQuant,cQuant,cStatistic,cenSatSelect,minCount), 
                facecolor=fig.get_facecolor())
        plt.close(fig)

def quantSlice1D(sPs, pdf, xQuant, yQuants, sQuant, sRange, cenSatSelect='cen', fig_subplot=[None,None]):
    """ Make a 1D slice through the 2D histogram by restricting to some range sRange of some quantity
    sQuant which is typically Mstar (e.g. 10.4<log_Mstar<10.6 to slice in the middle of the bimodality).
    For all subhalos in this slice, optically restricted by cenSatSelect, load a set of quantities 
    yQuants (could be just one) and plot this (y-axis) against xQuant, with any additional configuration 
    provided by xQuantSpec. Supports multiple sPs which are overplotted. Multiple yQuants results in a grid. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    if len(yQuants) == 0: return
    nRows = np.floor(np.sqrt(len(yQuants)))
    nCols = np.ceil(len(yQuants) / nRows)

    # hard-coded config
    lw = 2.5
    ptPlotThresh = 2000

    sizefac = 1.0 if not clean else sfclean
    if nCols > 4: sizefac *= 0.8 # enlarge text for big panel grids

    # start plot
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=[figsize[0]*nCols*sizefac, figsize[1]*nRows*sizefac])
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]

    # loop over each yQuantity (panel)
    for i, yQuant in enumerate(yQuants):
        if fig_subplot[0] is None:
            ax = fig.add_subplot(nRows,nCols,i+1)
        else:
            ax = fig.add_subplot(fig_subplot[1])

        for sP in sPs:
            # loop over each run and add to the same plot
            print(' ',yQuant,sP.simName,xQuant,cenSatSelect,sQuant,sRange)

            # y-axis: load galaxy properties (in histo2D were the color mappings)
            sim_yvals, ylabel, yMinMax, yLog = simSubhaloQuantity(sP, yQuant, clean, tight=True)

            if sim_yvals is None:
                print('   skip')
                continue # property is not calculated for this run (e.g. expensive auxCat)
            if yLog is True: sim_yvals = logZeroNaN(sim_yvals)

            # slice values: load fullbox galaxy property to slice on (e.g. Mstar or Mhalo)
            sim_svals, _, _, _ = simSubhaloQuantity(sP, sQuant, clean)

            if sim_svals is None:
                print('   skip')
                continue

            # x-axis: load/calculate x-axis quantity (e.g. simulation colors), cached in sP.data
            sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)

            if sim_xvals is None:
                print('   skip')
                continue
            if xLog is True: sim_xvals = logZeroNaN(sim_xvals)

            # central/satellite selection?
            wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

            sim_xvals = sim_xvals[wSelect]
            sim_yvals = sim_yvals[wSelect]
            sim_svals = sim_svals[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.isfinite(sim_xvals) & np.isfinite(sim_yvals)
            sim_xvals = sim_xvals[wFinite]
            sim_yvals = sim_yvals[wFinite]
            sim_svals = sim_svals[wFinite]

            # make slice selection
            wSlice = np.where( (sim_svals >= sRange[0]) & (sim_svals < sRange[1]) )
            sim_xvals = sim_xvals[wSlice]
            sim_yvals = sim_yvals[wSlice]
            sim_svals = sim_svals[wSlice]

            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # plot points
            c = ax._get_lines.prop_cycler.next()['color']

            if sim_xvals.size < ptPlotThresh:
                ax.plot(sim_xvals, sim_yvals, 'o', color=c, alpha=0.3)

            # median and 10/90th percentile lines
            nBins = 30
            if sim_xvals.size >= ptPlotThresh:
                nBins *= 2

            binSize = (xMinMax[1]-xMinMax[0]) / nBins

            xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
            if xm.size > sKn:
                ym = savgol_filter(ym,sKn,sKo)
                sm = savgol_filter(sm,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1)

            ax.plot(xm, ym, linestyles[0], lw=lw, color=c, label=sP.simName)

            # percentile band:
            if sim_xvals.size >= ptPlotThresh:
                ax.fill_between(xm, pm[1,:], pm[-2,:], facecolor=c, alpha=0.1, interpolate=True)

        # legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            handlesO = []
            labelsO = []

            if not clean:
                handlesO = [plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[0])]
                labelsO  = ['median, P[10,90]']

            legend = ax.legend(handles+handlesO, labels+labelsO, loc='best')

    # finish plot and save
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig()
        else:
            fig.savefig('slice1d_%s_%s_%s_%s_%s.pdf' % (xlabel,cQuant,cStatistic,cenSatSelect,minCount))
        plt.close(fig)

def quantMedianVsSecondQuant(sPs, pdf, yQuants, xQuant, cenSatSelect='cen', 
                             sQuant=None, sLowerPercs=None, sUpperPercs=None, 
                             scatterPoints=False, markSubhaloIDs=None, fig_subplot=[None,None]):
    """ Make a running median of some quantity (e.g. SFR) vs another on the x-axis (e.g. Mstar).
    For all subhalos, optically restricted by cenSatSelect, load a set of quantities 
    yQuants (could be just one) and plot this (y-axis) against the xQuant. Supports multiple sPs 
    which are overplotted. Multiple yQuants results in a grid. 
    If sQuant is not None, then in addition to the median, load this third quantity and split the 
    subhalos on it according to sLowerPercs, sUpperPercs (above/below the given percentiles), for 
    each split plotting the sub-sample yQuant again versus xQuant.
    If scatterPoints, include all raw points with a scatterplot. 
    If markSubhaloIDs, highlight these subhalos especially on the plot. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    nRows = np.floor(np.sqrt(len(yQuants)))
    nCols = np.ceil(len(yQuants) / nRows)

    # hard-coded config
    lw = 2.5
    ptPlotThresh = 2000
    markersize = 4.0
    nBins = 60
    legendLoc = 'best'

    sizefac = 1.0 if not clean else sfclean
    if nCols > 4: sizefac *= 0.8 # enlarge text for big panel grids

    # start plot
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=[figsize[0]*nCols*sizefac, figsize[1]*nRows*sizefac])
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]

    # loop over each yQuantity (panel)
    for i, yQuant in enumerate(yQuants):
        if fig_subplot[0] is None:
            ax = fig.add_subplot(nRows,nCols,i+1)
        else:
            ax = fig.add_subplot(fig_subplot[1])

        for sP in sPs:
            # loop over each run and add to the same plot
            print(' ',yQuant,xQuant,sP.simName,cenSatSelect)

            # y-axis: load fullbox galaxy properties
            sim_yvals, ylabel, yMinMax, yLog = simSubhaloQuantity(sP, yQuant, clean, tight=True)

            if sim_yvals is None:
                print('   skip')
                continue # property is not calculated for this run (e.g. expensive auxCat)
            if yLog: sim_yvals = logZeroNaN(sim_yvals)

            # x-axis: load fullbox galaxy properties
            sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)
            if xLog: sim_xvals = logZeroNaN(sim_xvals)

            # splitting on third quantity? load now
            if sQuant is not None:
                sim_svals, slabel, _, sLog = simSubhaloQuantity(sP, sQuant, clean, tight=True)
                if sim_svals is None:
                    print('   skip')
                    continue
                if sLog: sim_svals = logZeroNaN(sim_svals)

            # central/satellite selection?
            wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

            sim_yvals_orig = np.array(sim_yvals)
            sim_xvals_orig = np.array(sim_xvals)

            sim_yvals = sim_yvals[wSelect]
            sim_xvals = sim_xvals[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.isfinite(sim_xvals) & np.isfinite(sim_yvals)
            sim_yvals  = sim_yvals[wFinite]
            sim_xvals  = sim_xvals[wFinite]

            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # plot points (todo: update for medianQuant)
            c = ax._get_lines.prop_cycler.next()['color']

            if sim_xvals.size < ptPlotThresh:
                ax.plot(sim_xvals, sim_yvals, 'o', color=c, alpha=0.3)

            # median and 10/90th percentile lines
            binSize = (xMinMax[1]-xMinMax[0]) / nBins
            if sP.boxSize < 205000.0: binSize *= 2.0

            xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
            if xm.size > sKn:
                ym = savgol_filter(ym,sKn,sKo)
                sm = savgol_filter(sm,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1)

            ax.plot(xm, ym, linestyles[0], lw=lw, color=c, label=sP.simName)

            # percentile:
            if sim_xvals.size >= ptPlotThresh:
                ax.fill_between(xm, pm[1,:], pm[-2,:], facecolor=c, alpha=0.1, interpolate=True)

            # slice value?
            if sQuant is not None:
                svals_loc = sim_svals[wSelect][wFinite]
                binSizeS = binSize*2

                if 1 or len(sPs) == 1:
                    # if only one run, use new colors for above and below slices (currently always do this)
                    c = ax._get_lines.prop_cycler.next()['color']

                xm, yma, ymb, pma, pmb = running_median_sub(sim_xvals,sim_yvals,svals_loc,binSize=binSizeS,
                                                    sPercs=sLowerPercs)

                for j, sLowerPerc in enumerate(sLowerPercs):
                    label = '%s < P[%d]' % (slabel,sLowerPerc)
                    ax.plot(xm, ymb[j], linestyles[1+j], lw=lw, color=c, label=label)

                lsOffset = len(sLowerPercs)
                if 1 or len(sPs) == 1:
                    c = ax._get_lines.prop_cycler.next()['color']
                    lsOffset = 0

                xm, yma, ymb, pma, pmb = running_median_sub(sim_xvals,sim_yvals,svals_loc,binSize=binSizeS,
                                                    sPercs=sUpperPercs)

                for j, sUpperPerc in enumerate(sUpperPercs):
                    label = '%s > P[%d]' % (slabel,sUpperPerc)
                    ax.plot(xm, yma[j], linestyles[1+j+lsOffset], lw=lw, color=c, label=label)

            # contours (optionally conditional, i.e. independently normalized for each x-axis value
            # todo

            # scatter all points?
            if scatterPoints:
                w = np.where( (sim_xvals >= xMinMax[0]) & (sim_xvals <= xMinMax[1]) ) # reduce PDF weight
                ax.scatter(sim_xvals[w], sim_yvals[w], s=markersize, marker='o', color=c, edgecolors='none', alpha=0.2)

            # highlight/overplot a single subhalo or a few subhalos?
            if markSubhaloIDs is not None:
                for subID in markSubhaloIDs:
                    label = 'Subhalo #%d' % subID if len(markSubhaloIDs) <= 3 else ''
                    ax.plot(sim_xvals_orig[subID], sim_yvals_orig[subID], 'o', markersize=markersize*2, 
                        color=sampleColorTable('tableau10','red'), alpha=1.0, label=label)

        # special case: BH_CumEgy_ratio add theory curve on top from BH model
        if clean and yQuant in ['BH_CumEgy_ratio','BH_CumEgy_ratioInv']:
            # make a second y-axis on the right
            color2 = '#999999'

            #ax.set_ylim([0.0,6.0])
            ax2 = ax.twinx()
            #ax2.set_ylim([-0.002, 0.103])
            ax2.set_ylim([8e-5, 0.12])
            ax2.set_yscale('log')

            #ax2.set_ylabel('BH Low State Transition Threshold ($\chi$)', color=color2)
            ax2.set_ylabel('Blackhole Accretion Rate / Eddington Rate', color=color2)
            ax2.tick_params('y', which='both', colors=color2)

            # need median M_BH as a function of x-axis (e.g. M_star)
            for bhIterNum, bhRedshift in enumerate([0.0]):
                # more than 1 redshift
                sP_loc = copy.copy(sP)
                sP_loc.setRedshift(bhRedshift)

                sim_x_loc, _, _, take_log2 = simSubhaloQuantity(sP_loc, xQuant, clean)
                if take_log2: sim_x_loc = logZeroNaN(sim_x_loc) # match

                # same filters as above
                wSelect = cenSatSubhaloIndices(sP_loc, cenSatSelect=cenSatSelect)
                sim_x_loc = sim_x_loc[wSelect]

                for bhPropNum, bhPropName in enumerate(['M_BH_actual','Mdot_BH_edd']):
                    sim_m_bh, _, _, take_log2 = simSubhaloQuantity(sP_loc, bhPropName, clean)
                    if not take_log2: sim_m_bh = 10.0**sim_m_bh # undo log then

                    # same filters as above
                    sim_m_bh = sim_m_bh[wSelect]

                    wFinite = np.isfinite(sim_x_loc) & np.isfinite(sim_m_bh)
                    sim_x_loc2 = sim_x_loc[wFinite]
                    sim_m_bh = sim_m_bh[wFinite]

                    xm_bh, ym_bh, _ = running_median(sim_x_loc2,sim_m_bh,binSize=binSize*2,skipZeros=True)
                    ym_bh = savgol_filter(ym_bh,sKn,sKo)
                    w = np.where( (ym_bh > 0.0) ) #& (xm_bh > xMinMax[0]) & (xm_bh < xMinMax[1]))
                    xm_bh = xm_bh[w]
                    ym_bh = ym_bh[w]

                    # derive eddington ratio transition as a function of x-axis (e.g. M_star)
                    linestyle = '-' if (bhIterNum == 0 and bhPropNum == 0) else ':'
                    if bhPropName == 'M_BH_actual':
                        ym_bh = tngModel_chi(ym_bh)

                    ax2.plot( xm_bh, ym_bh, linestyle=linestyle, lw=lw, color=color2)

            ax.set_xlim(xMinMax) # fix
            legendLoc = 'lower right'

        # legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            handlesO = []
            labelsO = []

            if not clean:
                handlesO = [plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[0])]
                labelsO  = ['median, P[10,90]']

            legend = ax.legend(handles+handlesO, labels+labelsO, loc=legendLoc)

    # finish plot and save
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        if pdf is not None:
            pdf.savefig()
        else:
            fig.savefig('medianQuants_%s_%s_%s_%s.pdf' % \
                ('-'.join([sP.simName for sP in sPs],'-'.join([xQ for xQ in xQuant]),yQuant,cenSatSelect)))
        plt.close(fig)

# ------------------------------------------------------------------------------------------------------

def plots():
    """ Driver (exploration 2D histograms). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = 'M_BH_actual' # 'color_C_gr'
    xQuant = 'mstar_30pkpc_log'
    cs     = 'median_nan'
    cenSatSelects = ['cen'] #['cen','sat','all']
    pStyle = 'white'

    quants = quantList(wTr=True, wMasses=True)
    quants = ['fgas2']

    for sP in sPs:
        for css in cenSatSelects:

            pdf = PdfPages('galaxy_2dhistos_%s_%s_%s_%s_%s.pdf' % (sP.simName,yQuant,xQuant,cs,css))

            for cQuant in quants:
                quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, 
                             cenSatSelect=css, cQuant=cQuant, cStatistic=cs, pStyle=pStyle)

            pdf.close()

def plots2():
    """ Driver (exploration 1D slices). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    xQuant = 'color_C_gr'
    sQuant = 'mstar2_log'
    sRange = [10.4,10.6]
    cenSatSelects = ['cen']

    quants = quantList(wCounts=False,wTr=False)
    quantsTr = quantList(wCounts=False,onlyTr=True)

    for css in cenSatSelects:
        pdf = PdfPages('galaxyColor_1Dslices_%s_%s_%s-%.1f-%.1f_%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))

        # all quantities on one multi-panel page:
        #quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=quants, sQuant=sQuant, 
        #             sRange=sRange, cenSatSelect=css, )
        #quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=quantsTr, sQuant=sQuant, 
        #             sRange=sRange, cenSatSelect=css)

        # one page per quantity:
        for yQuant in quants + quantsTr:
            quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=[yQuant], sQuant=sQuant, 
                         sRange=sRange, cenSatSelect=css)

        pdf.close()

def plots3():
    """ Driver (exploration median trends). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    #sPs.append( simParams(res=1820, run='illustris', redshift=0.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    xQuant = 'mstar_30pkpc' #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    cenSatSelects = ['cen']

    sQuant = 'color_C_gr' #'mstar_out_100kpc_frac_r200'
    sLowerPercs = [10,50]
    sUpperPercs = [90,50]

    yQuants = quantList(wCounts=False, wTr=True, wMasses=True)
    yQuants = ['size_gas']

    # make plots
    for css in cenSatSelects:
        pdf = PdfPages('medianQuants_%s_x=%s_%s_slice=%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,css,sQuant))

        # all quantities on one multi-panel page:
        #quantMedianVsSecondQuant(sPs, pdf, yQuants=yQuants, xQuant=xQuant, cenSatSelect=css)

        # individual plot per y-quantity:
        #for yQuant in yQuants:
        #    quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css,
        #                             sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)

        # individual plot per s-quantity:
        #for sQuant in sQuants:
        #    quantMedianVsSecondQuant(sPs, pdf, yQuants=yQuant, xQuant=xQuant, cenSatSelect=css,
        #                             sQuant=[sQuant], sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)

        pdf.close()

def plots4():
    """ Driver (single median trend). """
    sP = simParams(res=1820, run='tng', redshift=0.0)

    xQuant = 'mstar_30pkpc' #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    yQuant = 'mhi_30pkpc'
    cenSatSelect = 'cen'

    sQuant = None #'color_C_gr','mstar_out_100kpc_frac_r200'
    sLowerPercs = None #[10,50]
    sUpperPercs = None #[90,50]

    pdf = PdfPages('median_x=%s_y=%s_%s_slice=%s_%s_z%.1f.pdf' % \
        (xQuant,yQuant,cenSatSelect,sQuant,sP.simName,sP.redshift))

    # one quantity
    quantMedianVsSecondQuant([sP], pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             #sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, 
                             scatterPoints=True, markSubhaloIDs=[252245])

    pdf.close()
