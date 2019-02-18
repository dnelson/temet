"""
cosmoGeneral.py
  Fully generalized plots and general plot helpers related to group catalog objects of cosmological boxes.
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
from os.path import isfile
from getpass import getuser
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from util import simParams
from util.helper import running_median, running_median_sub, logZeroNaN, loadColorTable, getWhiteBlackColors, sampleColorTable
from cosmo.color import loadSimGalColors, calcMstarColor2dKDE
from vis.common import setAxisColors, setColorbarColors
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

# ------------------------------------------------------------------------------------------------------

def quantHisto2D(sP, pdf, yQuant, xQuant='mstar2_log', cenSatSelect='cen', cQuant=None, xlim=None, ylim=None, clim=None, 
                 cStatistic=None, cNaNZeroToMin=False, minCount=None, cRel=None, cFrac=None, nBins=None, qRestrictions=None, 
                 filterFlag=False, medianLine=True, sizeFac=1.0, 
                 fig_subplot=[None,None], pStyle='white', ctName=None, saveFilename=None, output_fmt=None):
    """ Make a 2D histogram of subhalos with one quantity on the y-axis, another property on the x-axis, 
    and optionally a third property as the colormap per bin. minCount specifies the minimum number of 
    points a bin must contain to show it as non-white. If '_nan' is not in cStatistic, then by default, 
    empty bins are white, and bins whose cStatistic is NaN (e.g. any NaNs in bin) are gray. Or, if 
    '_nan' is in cStatistic, then empty bins remain white, while the cStatistic for bins with any 
    non-NaN values is computed ignoring NaNs (e.g. np.nanmean() instead of np.mean()), and bins 
    which are non-empty but contain only NaN values are gray. If cRel is not None, then should be a 
    3-tuple of [relMin,relMax,takeLog] in which case the colors are not of the physical cQuant itself, 
    but rather the value of that quantity relative to the median at that value of the x-axis (e.g. mass).
    If cFrac is not None, then a 4-tuple of [fracMin,fracMax,takeLog,label] specifying a criterion on the values 
    of cQuant such that the colors are not of the physical cQuant itself, but rather represent the fraction of 
    subhalos in each pixel satisfying (fracMin <= cQuant < fracMax), where +/-np.inf is allowed for one-sided, 
    takeLog should be True or False, and label is either a string or None for automatic.
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If filterFlag, exclude SubhaloFlag==0 (non-cosmological) objects.
    If xlim, ylim, or clim are not None, then override the respective axes ranges with these [min,max] bounds. 
    If cNanZeroToMin, then change the color of the NaN-only bins from the usual gray to the colormap minimum. """
    assert cenSatSelect in ['all', 'cen', 'sat']
    assert cStatistic in [None,'mean','median','count','sum','median_nan'] # or any user function
    assert np.count_nonzero([cRel,cFrac]) <= 1 # at most one

    # hard-coded config
    if nBins is None: 
        nBins = 80

    cmap     = loadColorTable(ctName if ctName is not None else 'viridis')
    colorMed = 'black'
    lwMed    = 2.0

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    colorContours = False
    if cQuant is None:
        colorMed = 'orange'

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean)
    if xMinMax[0] > xMinMax[1]: xMinMax = xMinMax[::-1] # reverse
    if xLog is True: sim_xvals = logZeroNaN(sim_xvals)
    if xlim is not None: xMinMax = xlim

    # y-axis: load/calculate simulation colors, cached in sP.data
    sim_yvals, ylabel, yMinMax, yLog = simSubhaloQuantity(sP, yQuant, clean, tight=True)
    if yLog is True: sim_yvals = logZeroNaN(sim_yvals)
    if ylim is not None: yMinMax = ylim

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros( sim_xvals.size, dtype='float32' )

        # overrides for density distribution
        cStatistic = 'count'
        if ctName is None:
            ctName = 'gray_r' if pStyle == 'white' else 'gray'
        cmap = loadColorTable(ctName)

        clabel = 'log N$_{\\rm gal}$+1'
        cMinMax = [0.0,2.0] if clim is None else clim
        if sP.boxSize > 100000: cMinMax = [0.0,2.5]
    else:
        if cStatistic is None: cStatistic = 'median_nan' # default if not specified with cQuant
        sim_cvals, clabel, cMinMax, cLog = simSubhaloQuantity(sP, cQuant, clean, tight=False)
        if clim is not None: cMinMax = clim
        if yQuant == 'color_C_gr': print('Warning: to reproduce TNG colors paper, set tight=True maybe.')

    if sim_cvals is None:
        return # property is not calculated for this run (e.g. expensive auxCat)

    # flagging?
    sim_flag = np.ones(sim_xvals.shape).astype('bool')
    if filterFlag and sP.groupCatHasField('Subhalo','SubhaloFlag'):
        # load SubhaloFlag and override sim_flag (0=bad, 1=good)
        sim_flag = sP.groupCat(fieldsSubhalos=['SubhaloFlag'])

    # central/satellite selection?
    wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_yvals = sim_yvals[wSelect]
    sim_flag  = sim_flag[wSelect]

    # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
    wFinite = np.isfinite(sim_yvals) #& np.isfinite(sim_xvals)

    # reduce to the good-flagged subset
    wFinite &= (sim_flag)

    sim_yvals = sim_yvals[wFinite]
    sim_cvals = sim_cvals[wFinite]
    sim_xvals = sim_xvals[wFinite]

    # arbitrary property restriction(s)?
    if qRestrictions is not None:
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # load and restrict
            vals, _, _, _ = sP.simSubhaloQuantity(rFieldName)
            vals = vals[wSelect][wFinite]

            wRestrict = np.where( (vals>=rFieldMin) & (vals<rFieldMax) )

            sim_yvals = sim_yvals[wRestrict]
            sim_xvals = sim_xvals[wRestrict]
            sim_cvals = sim_cvals[wRestrict]

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
        fig = plt.figure(figsize=(figsize[0]*sizeFac,figsize[1]*sizeFac),facecolor=color1)
        ax = fig.add_subplot(111, facecolor=color1)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1], facecolor=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if getuser() == 'dnelson':
        print(' ',xQuant,yQuant,cQuant,sP.simName,cenSatSelect)

    if not clean:
        pass
        #ax.set_title('stat=%s select=%s mincount=%s' % (cStatistic,cenSatSelect,minCount))
    else:
        cssStrings = {'all':'all galaxies', 'cen':'centrals only', 'sat':'satellites'}
        if getuser() != 'dnelson':
            ax.set_title(sP.simName + ': ' + cssStrings[cenSatSelect])

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])
    extent = [xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_yvals, sim_cvals, cStatistic, 
                                                 bins=nBins2D, range=[xMinMax,yMinMax])

    cc = cc.T # imshow convention

    # only show bins with a minimum number of points?
    nn, _, _, _ = binned_statistic_2d(sim_xvals, sim_yvals, sim_cvals, 'count', 
                                      bins=nBins2D, range=[xMinMax,yMinMax])
    nn = nn.T

    # relative coloring as a function of the x-axis?
    if cRel is not None:
        # override min,max of color and whether or not to log
        cMinMax[0], cMinMax[1], cLog = cRel

        # normalize each column by median, ignore empty pixels
        if cStatistic == 'count':
            w = np.where(cc == 0.0)
            cc[w] = np.nan

        with np.errstate(invalid='ignore'):
            medVals = np.nanmedian(cc, axis=0)
            cc /= medVals[np.newaxis, :]

        cmap   = loadColorTable('coolwarm') # diverging
        clabel = 'Relative ' + clabel.split('[')[0] + ('[ log ]' if cLog else '')

    # color based on fraction of systems in a pixel which satisfy some criterion?
    if cFrac is not None:
        # override min,max of color and whether or not to log
        fracMin, fracMax, cLog, fracLabel = cFrac
        if clim is None:
            cMinMax = [0.0, 1.0] if not cLog else [-1.5,0.0]

        # select sim values which satisfy criterion, and re-count
        w = np.where( (sim_cvals >= fracMin) & (sim_cvals < fracMax) )

        nn_sat, _, _, _ = binned_statistic_2d(sim_xvals[w], sim_yvals[w], sim_cvals[w], 'count', 
                                          bins=nBins2D, range=[xMinMax,yMinMax])
        nn_sat = nn_sat.T

        with np.errstate(invalid='ignore'):
            # set each pixel value to the fraction (= N_sat / N_tot)
            cc = nn_sat / nn

            # set absolute zeros to a small finite value, to avoid special (gray) coloring
            w = np.where(cc == 0.0)
        cc[w] = 1e-10

        # modify colortable and label
        cmap = loadColorTable('matter_r') # haline, thermal, solar, deep_r, dense_r, speed_r, amp_r, matter_r

        qStr = clabel.split('[')[0] # everything to the left of the units
        #if '$' in clabel:
        #    qStr = '$%s$' % clabel.split('$')[1] # just the label symbol, if one is present
        qUnitStr = ''
        if '[' in clabel:
            qUnitStr = ' ' + clabel.split('[')[1].split(']')[0].strip()
            if qUnitStr == ' log': qUnitStr = ''

        # 1 digit after the decimal point if the bounds numbers are not roundable to ints, else just integers
        qDigits = 1 if ((np.isfinite(fracMin) & ~float(fracMin).is_integer()) | (np.isfinite(fracMax) & ~float(fracMax).is_integer())) else 0

        clabel = 'Fraction ('
        if np.isinf(fracMin):
            clabel += '%s < %.*f%s)' % (qStr,qDigits,fracMax,qUnitStr)
        elif np.isinf(fracMax):
            clabel += '%s > %.*f%s)' % (qStr,qDigits,fracMin,qUnitStr)
        else:
            clabel += '%.*f < %s [%s] < %.*f)' % (qDigits,fracMin,qStr,qUnitStr,qDigits,fracMax)
        clabel += (' [ log ]' if cLog else '')

        # manually specified label?
        if fracLabel is not None:
            clabel = fracLabel

    # for now: log on density and all color quantities
    cc2d = cc
    if cQuant is None: cc2d += 1.0 # add 1 to count
    if cQuant is None or cLog is True:
        cc2d = logZeroNaN(cc)

    # normalize and color map
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap(norm(cc2d))

    # mask bins with median==0 and map to special color, which right now have been set to log10(0)=NaN
    color3 = colorConverter.to_rgba(color3)
    color4 = colorConverter.to_rgba(color4)

    if cNaNZeroToMin:
        color3 = cmap(0.0)
        color4 = cmap(0.0)

    if cQuant is not None:
        cc2d_rgb[cc == 0.0,:] = color4

    if nanFlag:
        # bin NaN point set counts
        nn_nan, _, _, _ = binned_statistic_2d(sim_xvals_nan, sim_yvals_nan, sim_cvals_nan, 'count', 
                                              bins=nBins2D, range=[xMinMax,yMinMax])
        nn_nan = nn_nan.T        

        # flag bins with nn_nan>0 and nn==0 (only NaNs in bin) as second gray color
        cc2d_rgb[ ((nn_nan > 0) & (nn == 0)), :] = color3

        nn += nn_nan # accumulate total counts
    else:
        # mask bins with median==NaN (nonzero number of NaNs in bin) to gray
        cc2d_rgb[~np.isfinite(cc),:] = color3

    # mask empty bins to white
    cc2d_rgb[(nn == 0),:] = colorConverter.to_rgba(color1)

    if minCount is not None:
        cc2d_rgb[nn < minCount] = colorConverter.to_rgba(color1)

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
    if np.count_nonzero(np.isnan(sim_xvals)) == sim_xvals.size:
        warnStr = 'Warning! All x-axis values are NaN, so nothing to plot (for example, mhalo_200 is NaN for satellites).'
        ax.text( np.mean(ax.get_xlim()), np.mean(ax.get_ylim()), warnStr, ha='center', va='center', color='black', fontsize=11)
        medianLine = False # all x-axis values are nan (i.e. mhalo_200 for cenSatSelect=='sat')

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

        if 0: #not clean:
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
        aux_yvals = aux_yvals[wSelect][wFinite]
        if nanFlag: aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,aux_yvals,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed, label='Halo $R_{\\rm 200,crit}$')
        ax.legend(loc='upper left')

    if yQuant in ['temp_halo','temp_halo_volwt']:
        # add virial temperature median line
        aux_yvals = sP.groupCat(fieldsSubhalos=['tvir_log'])
        aux_yvals = aux_yvals[wSelect][wFinite]
        if nanFlag: aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,aux_yvals,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed, label='Halo $T_{\\rm vir}$')
        ax.legend(loc='upper left')

    if yQuant == 'fgas_r200':
        # add constant f_b line
        f_b = np.log10(sP.units.f_b)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xMinMax, [f_b,f_b], '--', color=color, lw=lwMed)
        ax.text(np.mean(ax.get_xlim()), f_b+0.05, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', color=color, size=17)

    if yQuant in ['BH_CumEgy_low','BH_CumEgy_high']:
        # add approximate halo binding energy line = (3/5)*GM^2/R
        G = sP.units.G / 1e10 # kpc (km/s)**2 / msun
        r_halo, _, _, _ = simSubhaloQuantity(sP, 'rhalo_200', clean) # pkpc
        m_halo, _, _, _ = simSubhaloQuantity(sP, 'mhalo_200', clean) # msun
        e_b = (3.0/5.0) * G * m_halo**2 * sP.units.f_b / r_halo # (km/s)**2 * msun
        e_b = np.array(e_b, dtype='float64') * 1e10 * sP.units.Msun_in_g # cm^2/s^2 * g
        e_b = logZeroNaN(e_b).astype('float32') # log(cm^2/s^2 * g)
        e_b = e_b[wSelect][wFinite]
        if nanFlag: e_b = e_b[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals,e_b,binSize=binSizeMed,percs=[5,10,25,75,90,95])
        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)

        color = sampleColorTable('tableau10','purple')
        ax.plot(xm[:-1], ym[:-1], '--', color=color, lw=lwMed, label='Halo $E_{\\rm B}$')
        ax.legend(loc='upper left')

    if xQuant in ['color_nodust_VJ','color_C-30kpc-z_VJ'] and yQuant in ['color_nodust_UV','color_C-30kpc-z_UV']:
        # UVJ color-color diagram, add Tomczak+2014 separation of passive and SFing galaxies
        xx = [0.0,0.7,1.4,1.4]
        yy = [1.4,1.4,2.0,2.45]
        ax.plot(xx, yy, ':', lw=lw, color='red', label='Tomczak+14')

        # Muzzin+2013b separation line (Equations 1-3)
        if sP.redshift <= 1.0: off = 0.69
        if sP.redshift > 1.0: off = 0.59
        xx = [0.0,(1.3-off)/0.88,1.5,1.5]
        yy = [1.3,1.3,1.5*0.88+off,2.45]
        ax.plot(xx, yy, ':', lw=lw, color='orange', label='Muzzin+13b')
        ax.legend(loc='upper left')

    # colorbar
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)
    if len(clabel) > 45:
        newsize = 23 - (len(clabel)-45)/5
        cb.ax.set_ylabel(clabel, size=newsize) # default: 24.192 (14 * x-large)
    setColorbarColors(cb, color2)

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
            # note: saveFilename could be an in-memory buffer
            if saveFilename is None:
                saveFilename = 'histo2d_%s_%s_%s_%s_%s_%s.pdf' % (ylabel,xQuant,cQuant,cStatistic,cenSatSelect,minCount)
            fig.savefig(saveFilename, format=output_fmt, facecolor=fig.get_facecolor())
        plt.close(fig)

    return True

def quantSlice1D(sPs, pdf, xQuant, yQuants, sQuant, sRange, cenSatSelect='cen', yRel=None, xlim=None, ylim=None, 
                 filterFlag=False, sizefac=None, fig_subplot=[None,None]):
    """ Make a 1D slice through the 2D histogram by restricting to some range sRange of some quantity
    sQuant which is typically Mstar (e.g. 10.4<log_Mstar<10.6 to slice in the middle of the bimodality).
    For all subhalos in this slice, optically restricted by cenSatSelect, load a set of quantities 
    yQuants (could be just one) and plot this (y-axis) against xQuant, with any additional configuration 
    provided by xQuantSpec. Supports multiple sPs which are overplotted. Multiple yQuants results in a grid. 
    If xlim or ylim are not None, then override the respective axes ranges with these [min,max] bounds. 
    If sRange is a list of lists, then overplot multiple different slice ranges. If yRel is not None, then should be a 
    3-tuple of [relMin,relMax,takeLog] or 4-tuple of [relMin,relMax,takeLog,yLabel] in which case the y-axis is not 
    of the physical yQuants themselves, but rather the value of the quantity relative to the median in the slice (e.g. mass).
    If filterFlag, exclude SubhaloFlag==0 (non-cosmological) objects. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    if len(yQuants) == 0: return
    nRows = np.floor(np.sqrt(len(yQuants)))
    nCols = np.ceil(len(yQuants) / nRows)

    # just a single sRange? wrap in an outer list
    sRanges = sRange
    if not isinstance(sRange[0],list):
        sRanges = [sRange]

    # hard-coded config
    lw = 2.5
    ptPlotThresh = 2000

    if sizefac is None:
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
            if ylim is not None: yMinMax = ylim

            if sim_yvals is None:
                print('   skip')
                continue # property is not calculated for this run (e.g. expensive auxCat)
            if yLog is True: sim_yvals = logZeroNaN(sim_yvals)

            # slice values: load fullbox galaxy property to slice on (e.g. Mstar or Mhalo)
            sim_svals, slabel, _, _ = simSubhaloQuantity(sP, sQuant, clean)

            if sim_svals is None:
                print('   skip')
                continue

            # x-axis: load/calculate x-axis quantity (e.g. simulation colors), cached in sP.data
            sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)
            if xlim is not None: xMinMax = xlim

            if sim_xvals is None:
                print('   skip')
                continue
            if xLog is True: sim_xvals = logZeroNaN(sim_xvals)

            # relative coloring relative to the median in the slice?
            if yRel is not None:
                # override min,max of y-axis and whether or not to log
                assert yLog is False # otherwise handle in a general way (and maybe undo?)
                if len(yRel) == 3:
                    yMinMax[0], yMinMax[1], yLog = yRel
                    ylabel = 'Relative ' + ylabel.split('[')[0] + ('[ log ]' if yLog else '')
                if len(yRel) == 4:
                    yMinMax[0], yMinMax[1], yLog, ylabel = yRel
                
            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # flagging?
            sim_flag = np.ones(sim_xvals.shape).astype('bool')
            if filterFlag and sP.groupCatHasField('Subhalo','SubhaloFlag'):
                # load SubhaloFlag and override sim_flag (0=bad, 1=good)
                sim_flag = sP.groupCat(fieldsSubhalos=['SubhaloFlag'])

            # central/satellite selection?
            wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

            sim_xvals = sim_xvals[wSelect]
            sim_yvals = sim_yvals[wSelect]
            sim_svals = sim_svals[wSelect]
            sim_flag  = sim_flag[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.isfinite(sim_xvals) & np.isfinite(sim_yvals)

            # reduce to the good-flagged subset
            wFinite &= (sim_flag)

            sim_xvals = sim_xvals[wFinite]
            sim_yvals = sim_yvals[wFinite]
            sim_svals = sim_svals[wFinite]

            # loop over slice ranges
            for sRange in sRanges:
                # make slice selection
                wSlice = np.where( (sim_svals >= sRange[0]) & (sim_svals < sRange[1]) )
                xx = sim_xvals[wSlice]
                yy = sim_yvals[wSlice]

                # relative coloring relative to the median in the slice?
                if yRel is not None:
                    yy /= np.nanmedian(yy)

                # plot points
                c = next(ax._get_lines.prop_cycler)['color']

                if xx.size < ptPlotThresh:
                    ax.plot(xx, yy, 'o', color=c, alpha=0.3)

                # median and 10/90th percentile lines
                nBins = 30
                if xx.size >= ptPlotThresh:
                    nBins *= 2

                binSize = (xMinMax[1]-xMinMax[0]) / nBins

                xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,percs=[5,10,25,75,90,95])
                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                sName = slabel.split('[')[0].rstrip() # shortened version (remove units) of split quant name for legend
                label = sP.simName if len(sRanges) == 1 else '%.1f < %s < %.1f' % (sRange[0],sName,sRange[1])
                ax.plot(xm, ym, linestyles[0], lw=lw, color=c, label=label)

                # percentile band:
                if xx.size >= ptPlotThresh:
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
                             scatterPoints=False, markSubhaloIDs=None, mark1to1=False,
                             xlim=None, ylim=None, filterFlag=False, fig_subplot=[None,None]):
    """ Make a running median of some quantity (e.g. SFR) vs another on the x-axis (e.g. Mstar).
    For all subhalos, optically restricted by cenSatSelect, load a set of quantities 
    yQuants (could be just one) and plot this (y-axis) against the xQuant. Supports multiple sPs 
    which are overplotted. Multiple yQuants results in a grid. 
    If sQuant is not None, then in addition to the median, load this third quantity and split the 
    subhalos on it according to sLowerPercs, sUpperPercs (above/below the given percentiles), for 
    each split plotting the sub-sample yQuant again versus xQuant.
    If scatterPoints, include all raw points with a scatterplot. 
    If markSubhaloIDs, highlight these subhalos especially on the plot. 
    If mark1to1, show a 1-to-1 line (i.e. assuming x and y axes could be closely related). 
    If filterFlag, exclude SubhaloFlag==0 (non-cosmological) objects. """
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
            if ylim is not None: yMinMax = ylim

            if sim_yvals is None:
                print('   skip')
                continue # property is not calculated for this run (e.g. expensive auxCat)
            if yLog: sim_yvals = logZeroNaN(sim_yvals)

            # x-axis: load fullbox galaxy properties
            sim_xvals, xlabel, xMinMax, xLog = simSubhaloQuantity(sP, xQuant, clean, tight=True)
            if xLog: sim_xvals = logZeroNaN(sim_xvals)
            if xlim is not None: xMinMax = xlim

            # splitting on third quantity? load now
            if sQuant is not None:
                sim_svals, slabel, _, sLog = simSubhaloQuantity(sP, sQuant, clean, tight=True)
                if sim_svals is None:
                    print('   skip')
                    continue
                if sLog: sim_svals = logZeroNaN(sim_svals)

            # flagging?
            sim_flag = np.ones(sim_xvals.shape).astype('bool')
            if filterFlag and sP.groupCatHasField('Subhalo','SubhaloFlag'):
                # load SubhaloFlag and override sim_flag (0=bad, 1=good)
                sim_flag = sP.groupCat(fieldsSubhalos=['SubhaloFlag'])

            # central/satellite selection?
            wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

            sim_yvals_orig = np.array(sim_yvals)
            sim_xvals_orig = np.array(sim_xvals)

            sim_yvals = sim_yvals[wSelect]
            sim_xvals = sim_xvals[wSelect]
            sim_flag  = sim_flag[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.isfinite(sim_xvals) & np.isfinite(sim_yvals)

            # reduce to the good-flagged subset
            wFinite &= (sim_flag)

            sim_yvals  = sim_yvals[wFinite]
            sim_xvals  = sim_xvals[wFinite]

            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # plot points (todo: update for medianQuant)
            c = next(ax._get_lines.prop_cycler)['color']

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

            label = sP.simName + ' z=%.1f' % sP.redshift
            ax.plot(xm, ym, linestyles[0], lw=lw, color=c, label=label)

            # percentile:
            if sim_xvals.size >= ptPlotThresh:
                ax.fill_between(xm, pm[1,:], pm[-2,:], facecolor=c, alpha=0.1, interpolate=True)

            # slice value?
            if sQuant is not None:
                svals_loc = sim_svals[wSelect][wFinite]
                binSizeS = binSize*2

                if 1 or len(sPs) == 1:
                    # if only one run, use new colors for above and below slices (currently always do this)
                    c = next(ax._get_lines.prop_cycler)['color']

                xm, yma, ymb, pma, pmb = running_median_sub(sim_xvals,sim_yvals,svals_loc,binSize=binSizeS,
                                                    sPercs=sLowerPercs)

                for j, sLowerPerc in enumerate(sLowerPercs):
                    label = '%s < P[%d]' % (slabel,sLowerPerc)
                    ax.plot(xm, ymb[j], linestyles[1+j], lw=lw, color=c, label=label)

                lsOffset = len(sLowerPercs)
                if 1 or len(sPs) == 1:
                    c = next(ax._get_lines.prop_cycler)['color']
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

            # 1-to-1 line?
            if mark1to1:
                x0 = np.min( [ax.get_xlim()[0], ax.get_ylim()[0]] )
                x1 = np.max( [ax.get_xlim()[1], ax.get_ylim()[1]] )
                ax.plot( [x0,x1], [x0,x1], ':', lw=lw, color='black', alpha=0.9, label='1-to-1')

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
                wSelect = sP_loc.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
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
                        ym_bh = sP.units.BH_chi(ym_bh)

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
    """ Driver (exploration 2D histograms, vary over all known quantities as cQuant). """
    sPs = []
    sPs.append( simParams(res=2160, run='tng', redshift=1.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = 'ssfr'
    xQuant = 'mstar_30pkpc_log'
    cenSatSelects = ['cen'] #['cen','sat','all']

    quants = [None,'sfr2','ssfr','delta_sfms'] #quantList(wTr=True, wMasses=True)
    clim = None
    medianLine = True
    minCount = 0

    for sP in sPs:
        for css in cenSatSelects:

            pdf = PdfPages('galaxy_2dhistos_%s_%d_%s_%s_%s_min=%d.pdf' % (sP.simName,sP.snap,yQuant,xQuant,css,minCount))

            for cQuant in quants:
                quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, clim=clim, minCount=minCount, 
                             medianLine=medianLine, cenSatSelect=css, cQuant=cQuant)

            pdf.close()

def plots_explore(sP):
    """ Driver (exploration 2D histograms, vary over all known quantities as y-axis). """
    cQuants = ['slit_vsigma_halpha','slit_vrot_halpha','slit_voversigma_halpha',
               'slit_vsigma_starlight','slit_vrot_starlight','slit_voversigma_starlight']

    css = 'cen' #['cen','sat','all']

    yQuants = quantList(wCounts=False, wTr=False, wMasses=True)
    #yQuants = yQuants[0:22] # temporary

    xQuant = 'mstar_30pkpc_log'
    xlim   = [8.7, 11.2]

    for cQuant in cQuants:
        pdf = PdfPages('2dhistos_%s_%d_x=%s_y=all_c=%s_%s.pdf' % (sP.simName,sP.snap,xQuant,cQuant,css))

        for yQuant in yQuants:
            quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, xlim=xlim, cenSatSelect=css, cQuant=cQuant)

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
    cenSatSelects = ['all']

    sQuant = 'color_C_gr' #'mstar_out_100kpc_frac_r200'
    sLowerPercs = [10,50]
    sUpperPercs = [90,50]

    yQuants = quantList(wCounts=False, wTr=True, wMasses=True)
    yQuants = ['size_stars']

    # make plots
    for css in cenSatSelects:
        pdf = PdfPages('medianQuants_%s_x=%s_%s_slice=%s.pdf' % \
            ('-'.join([sP.simName for sP in sPs]),xQuant,css,sQuant))

        # all quantities on one multi-panel page:
        quantMedianVsSecondQuant(sPs, pdf, yQuants=yQuants, xQuant=xQuant, cenSatSelect=css,
                                 sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)

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
    yQuant = 'size_stars'
    cenSatSelect = 'all'
    filterFlag = True

    xlim = [7.5,11.5]
    ylim = [-1.0,1.2]

    sQuant = None #'color_C_gr'
    sLowerPercs = None #[10,50]
    sUpperPercs = None #[90,50]

    pdf = PdfPages('median_x=%s_y=%s_%s_slice=%s_%s_z%.1f.pdf' % \
        (xQuant,yQuant,cenSatSelect,sQuant,sP.simName,sP.redshift))

    # one quantity
    quantMedianVsSecondQuant([sP], pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             #sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, 
                             xlim=xlim, ylim=ylim, scatterPoints=True, markSubhaloIDs=None, filterFlag=filterFlag)

    pdf.close()

def plots_uvj():
    """ Driver. Explore UVJ color-color diagram. """
    sPs = []
    sPs.append( simParams(res=1820, run='tng', redshift=0.0) )
    #sPs.append( simParams(res=2500, run='tng', redshift=2.0) )

    yQuant = 'color_C-30kpc-z_UV' #'color_nodust_UV' #
    xQuant = 'color_C-30kpc-z_VJ' #'color_nodust_VJ' #
    cenSatSelects = ['all'] #['cen','sat','all']
    pStyle = 'white'

    cNaNZeroToMin = True # False
    medianLine = False # True

    if 0:
        # color-coded by SFR
        cs       = 'median_nan'
        quants   = ['ssfr']
        clim     = [-2.5,0.0] # None
        minCount = 10
        qRestrictions = None
        xlim     = None
        ylim     = None
    if 1:
        cs       = 'count'
        quants   = [None]
        clim     = [0.0, 2.5] # log N_gal
        minCount = 0
        qRestrictions = [ ['mstar_30pkpc_log',10.0,np.inf] ] # LEGA-C mass cut
        xlim     = [0.0,1.7]
        ylim     = [0.4,2.6]

    for sP in sPs:
        for css in cenSatSelects:

            pdf = PdfPages('galaxy_2dhistos_%s_%d_%s_%s_%s_%s_min=%d.pdf' % (sP.simName,sP.snap,yQuant,xQuant,cs,css,minCount))

            for cQuant in quants:
                quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, xlim=xlim, ylim=ylim, clim=clim, cNaNZeroToMin=cNaNZeroToMin, 
                             minCount=minCount, medianLine=medianLine, cenSatSelect=css, cQuant=cQuant, cStatistic=cs, 
                             qRestrictions=qRestrictions, pStyle=pStyle)

            pdf.close()

def plots_tng50_structural(rel=False, sP=None):
    """ Driver (exploration 2D histograms). """
    if sP is None:
        sP = simParams(res=2160, run='tng', redshift=1.0)

    xQuant  = 'mstar_30pkpc_log'
    xlim    = [8.7,11.2]
    yQuants = ['slit_vsigma_halpha','slit_vrot_halpha','slit_voversigma_halpha',
               'slit_vsigma_starlight','slit_vrot_starlight','slit_voversigma_starlight']

    quants_gas = [None,'sfr2','ssfr','delta_sfms','fgas2_alt','etaM_100myr_20kpc_0kms','vout_90_20kpc',
                  'size2d_halpha','diskheight2d_halpha','diskheightnorm2d_halpha','shape_s_sfrgas','shape_ratio_sfrgas']
    quants_stars = ['size2d_starlight','diskheight2d_starlight','diskheightnorm2d_starlight','shape_s_stars','shape_ratio_stars']

    css   = 'cen'
    clim  = None
    nBins = 60

    if rel:
        cRel = [0.5,1.5,False] # [cMin,cMax,cLog] #None
    else:
        cRel = None

    for yQuant in yQuants:

        pdf = PdfPages('2dhisto_%s_%d_x=%s_y=%s_rel=%s_%s.pdf' % (sP.simName,sP.snap,xQuant,yQuant,rel,css))

        for cQuant in quants_gas + quants_stars + yQuants:
            if cQuant == yQuant: continue

            quantHisto2D(sP, pdf, yQuant=yQuant, xQuant=xQuant, xlim=xlim, clim=clim, 
                         cenSatSelect=css, cQuant=cQuant, nBins=nBins, cRel=cRel)
        pdf.close()

    # return with all cached data, can be passed back in for rapid re-plotting
    return sP
