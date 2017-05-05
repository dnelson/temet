"""
cosmoGeneral.py
  Fully generalized plots and general plot helpers related to cosmological boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize, LogNorm, colorConverter
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, expanduser
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from util import simParams
from util.helper import running_median, logZeroNaN, loadColorTable
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, snapshotSubset
from cosmo.color import loadSimGalColors, calcMstarColor2dKDE
from vis.common import setAxisColors
from plot.general import simSubhaloQuantity, getWhiteBlackColors, bandMagRange
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

# ------------------------------------------------------------------------------------------------------

def _loadColorOrQuant(sP,xQuant,xSpec):
    """ Load either a simulation quantity (xSpec==None) or a color (xSpec=[bands,colorModel]).
    Return: tuple of (sim_vals,label,saveLabel,minMax). """
    k = 'sim_' + xQuant + '_'

    if k+'vals' in sP.data:
        # data already exists in sP cache?
        vals, label, saveLabel, minMax = \
          sP.data[k+'vals'], sP.data[k+'label'], sP.data[k+'saveLabel'], sP.data[k+'minMax']

        return vals, label, saveLabel, minMax

    # load
    if xQuant == 'color':
        bands, simColorsModel = xSpec[0], xSpec[1]
        vals, _ = loadSimGalColors(sP, simColorsModel, bands=bands)

        label = '(%s-%s) color [ mag ]' % (bands[0],bands[1])
        minMax = bandMagRange(bands, tight=True)
        if not clean: label += ' %s' % simColorsModel
        saveLabel = '%s_%s_%s' % (xQuant,'-'.join(bands),simColorsModel)                    
    else:
        vals, label, minMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)
        #if sQuant == xQuant: minMax = sRange # compress x range in this case
        if xLog: vals = logZeroNaN(vals)
        #minMax = [np.nanmin(vals), np.nanmax(vals)] # auto
        saveLabel = '%s' % (xQuant)

    # save into sP.data
    sP.data[k+'vals'], sP.data[k+'label'], sP.data[k+'saveLabel'], sP.data[k+'minMax'] = \
        vals, label, saveLabel, minMax
          
    return vals, label, saveLabel, minMax

def quantHisto2D(sP, pdf, yQuant, ySpec, xQuant='mstar2_log', cenSatSelect='cen', cQuant=None, 
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
    colorMed = 'orange'
    lwMed    = 2.0

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    medianLine = False
    colorContours = False

    if cQuant is None and cenSatSelect == 'cen':
        medianLine = True
    if cenSatSelect == 'cen' and yQuant == 'color': # fig_subplot[0] is not None and cQuant is not None:
        colorContours = True

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    if 'sim_xvals' in sP.data:
        sim_xvals, xlabel, xMinMax = sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax']
    else:
        sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant, clean)
        if xMinMax[0] > xMinMax[1]: xMinMax = xMinMax[::-1] # reverse
        sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax'] = sim_xvals, xlabel, xMinMax

    # y-axis: load/calculate simulation colors, cached in sP.data
    sim_yvals, ylabel, ySaveLabel, yMinMax = _loadColorOrQuant(sP,yQuant,ySpec)

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )

        # overrides for density distribution
        cStatistic = 'count'
        ctName = 'gray_r' if pStyle == 'white' else 'gray'
        cmap = loadColorTable(ctName)

        clabel = 'log N$_{\\rm gal}$'
        cMinMax = [0.0,2.0]
        if sP.boxSize > 100000: cMinMax = [1.0,2.5]
    else:
        sim_cvals, clabel, cMinMax, cLog = simSubhaloQuantity(sP, cQuant, clean)

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

    print(' ',cQuant,sP.simName,ySaveLabel,xQuant,cenSatSelect,cStatistic,minCount)
    if not clean:
        ax.set_title('stat=%s select=%s mincount=%s' % (cStatistic,cenSatSelect,minCount))
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
        ym2 = savgol_filter(ym,sKn,sKo)
        sm2 = savgol_filter(sm,sKn,sKo)
        pm2 = savgol_filter(pm,sKn,sKo,axis=1)

        ax.plot(xm[:-1], ym2[:-1], '-', color=colorMed, lw=lwMed, label='median')

        ax.plot(xm[:-1], pm2[1,:-1], ':', color=colorMed, lw=lwMed, label='P[10,90]')
        ax.plot(xm[:-1], pm2[-2,:-1], ':', color=colorMed, lw=lwMed)

        l = ax.legend(loc='lower right')
        for text in l.get_texts(): text.set_color(color2)

    # contours?
    if colorContours:        
        extent = [xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]]
        cLevels = [0.75,0.95]
        cAlphas = [0.5,0.8]

        xx, yy, kde_sim = calcMstarColor2dKDE(ySpec[0], sim_xvals, sim_yvals, xMinMax, yMinMax, 
                                              sP=sP, simColorsModel=ySpec[1])

        for k in range(kde_sim.shape[0]):
            kde_sim[k,:] /= kde_sim[k,:].max() # by column normalization

        for k, cLevel in enumerate(cLevels):
            ax.contour(xx, yy, kde_sim, [cLevel], colors=[color2], 
                       alpha=cAlphas[k], linewidths=lwMed, extent=extent)

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
                (ySaveLabel,xQuant,cQuant,cStatistic,cenSatSelect,minCount), 
                facecolor=fig.get_facecolor())
        plt.close(fig)

def quantSlice1D(sPs, pdf, xQuant, yQuants, sQuant, sRange, cenSatSelect='cen', 
                 xSpec=None, fig_subplot=[None,None]):
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

            # slice values: load fullbox galaxy properties (almost always Mstar)
            if 'histoSlice_svals' in sP.data:
                sim_svals = sP.data['histoSlice_svals']
            else:
                sim_svals, _, _, _ = simSubhaloQuantity(sP, sQuant, clean)
                sP.data['histoSlice_svals'] = sim_svals

            # x-axis: load/calculate x-axis quantity (usually simulation colors), cached in sP.data
            sim_xvals, xlabel, xSaveLabel, xMinMax = _loadColorOrQuant(sP,xQuant,xSpec)

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
            xm = xm[1:-1]
            ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
            sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
            pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

            ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

            # percentile band:
            if sim_xvals.size >= ptPlotThresh:
                ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

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
            fig.savefig('slice1d_%s_%s_%s_%s_%s.pdf' % (xSaveLabel,cQuant,cStatistic,cenSatSelect,minCount))
        plt.close(fig)

def quantMedianVsSecondQuant(sPs, pdf, yQuants, xQuant, cenSatSelect='cen', fig_subplot=[None,None]):
    """ Make a running median of some quantity (e.g. SFR) vs another on the x-axis (e.g. Mstar).
    For all subhalos, optically restricted by cenSatSelect, load a set of quantities 
    yQuants (could be just one) and plot this (y-axis) against the xQuant. Supports multiple sPs 
    which are overplotted. Multiple yQuants results in a grid. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    nRows = np.floor(np.sqrt(len(yQuants)))
    nCols = np.ceil(len(yQuants) / nRows)

    # hard-coded config
    lw = 2.5
    ptPlotThresh = 2000
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

            if yLog is True:
                sim_yvals = logZeroNaN(sim_yvals)

            # x-axis: load fullbox galaxy properties
            if 'quantMedian_xvals' in sP.data:
                sim_xvals, xlabel, xMinMax = sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax']                
            else:
                sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)
                if xLog:
                    sim_xvals = logZeroNaN(sim_xvals)
                sP.data['sim_xvals'], sP.data['xlabel'], sP.data['xMinMax'] = sim_xvals, xlabel, xMinMax

            # central/satellite selection?
            wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

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

            xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
            xm = xm[1:-1]
            ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
            sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
            pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

            ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

            # percentile:
            if sim_xvals.size >= ptPlotThresh:
                ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

        # special case: BH_CumEgy_ratio add theory curve on top from BH model
        if clean and yQuant == 'BH_CumEgy_ratio':
            # make a second y-axis on the right
            color2 = '#999999'
            chi0 = 0.002 # TNG fiducial model
            beta = 2.0 # TNG fiducial model
            chi_max = 0.1 # TNG fiducial model

            ax.set_ylim([0.0,6.0])
            ax2 = ax.twinx()
            ax2.set_ylim([-0.002, 0.11])
            #ax2.set_yscale('log')
            ax2.set_ylabel('BH Low State Transition Threshold ($\chi$)', color=color2)
            ax2.tick_params('y', which='both', colors=color2)

            # need median M_BH as a function of x-axis (e.g. M_star)
            sim_m_bh, _, _, take_log = simSubhaloQuantity(sP, 'M_BH_actual', clean)
            sim_m_bh = sim_m_bh[wSelect][wFinite]
            if not take_log: assert 0 # undo log then

            xm_bh, ym_bh, _ = running_median(sim_xvals,sim_m_bh,binSize=binSize,skipZeros=True)
            w = np.where( (ym_bh > 0.0) ) #& (xm_bh > xMinMax[0]) & (xm_bh < xMinMax[1]))
            xm_bh = xm_bh[w]
            ym_bh = ym_bh[w]

            # derive eddington ratio transition as a function of x-axis (e.g. M_star)
            chi_bh = np.clip( chi0 * (ym_bh / 1e8)**beta, 0.0, chi_max )
            ax2.plot( xm_bh, chi_bh, '-', lw=lw, color=color2)

            ax.set_xlim(xMinMax) # fix
            legendLoc = 'lower left'

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
    sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = 'color'
    ySpec  = [ ['g','r'], defSimColorModel ] # bands, simColorModel
    xQuant = 'mstar2_log' # mstar2_log, ssfr, M_BH_actual
    cs     = 'median_nan'
    cenSatSelects = ['cen'] #['cen','sat','all']

    quants = quantList()

    for sP in sPs:
        for css in cenSatSelects:

            pdf = PdfPages('galaxyColor_2dhistos_%s_%s_%s_%s_%s.pdf' % (sP.simName,yQuant,xQuant,cs,css))

            for cQuant in quants:
                quantHisto2D(sP, pdf, yQuant=yQuant, ySpec=ySpec, xQuant=xQuant, 
                             cenSatSelect=css, cQuant=cQuant, cStatistic=cs)

            pdf.close()

def plots2():
    """ Driver (exploration 1D slices). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    xQuant = 'color'
    xSpec  = [ ['g','r'], defSimColorModel ] # bands, simColorModel
    sQuant = 'mstar2_log'
    sRange = [10.4,10.6]
    cenSatSelects = ['cen']

    quants = quantList(wCounts=False,wTr=False)
    quantsTr = quantList(wCounts=False,onlyTr=True)

    for css in cenSatSelects:
        pdf = PdfPages('galaxyColor_1Dslices_%s_%s_%s-%.1f-%.1f_%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))

        # all quantities on one multi-panel page:
        quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=quants, sQuant=sQuant, 
                     sRange=sRange, cenSatSelect=css, )
        quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=quantsTr, sQuant=sQuant, 
                     sRange=sRange, cenSatSelect=css)

        # one page per quantity:
        for yQuant in quants + quantsTr:
            quantSlice1D(sPs, pdf, xQuant=xQuant, xSpec=xSpec, yQuants=[yQuant], sQuant=sQuant, 
                         sRange=sRange, cenSatSelect=css)

        pdf.close()

def plots3():
    """ Driver (exploration median trends). """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    #sPs.append( simParams(res=1820, run='illustris', redshift=0.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    bands = ['g','r']
    xQuant = 'mstar1_log'
    cenSatSelects = ['cen']

    quants = quantList(onlyBH=True)

    # make plots
    for css in cenSatSelects:
        pdf = PdfPages('medianQuants_%s_%s_%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,css))

        # all quantities on one multi-panel page:
        quantMedianVsSecondQuant(sPs, pdf, yQuants=quants, xQuant=xQuant, cenSatSelect=css)

        # one page per quantity:
        for yQuant in quants:
            quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css)

        pdf.close()
