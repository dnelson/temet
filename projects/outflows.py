"""
projects/outflows.py
  Plots: Outflows paper (TNG50 presentation).
  in prep.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.interpolate import griddata, interp1d
from functools import partial

from util import simParams
from util.helper import running_median, logZeroNaN, nUnique, loadColorTable, sgolay2d, sampleColorTable
from plot.config import *
from plot.general import plotHistogram1D, plotPhaseSpace2D
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from projects.outflows_analysis import halo_selection, loadRadialMassFluxes
from projects.outflows_vis import subboxSingleVelocityFrame, galaxyMosaic_topN
from tracer.tracerMC import match3

labels = {'rad'     : 'Radius [ pkpc ]',
          'vrad'    : 'Radial Velocity [ km/s ]',
          'vcut'    : 'Minimum Outflow Velocity Cut [ km/s ]',
          'temp'    : 'Gas Temperature [ log K ]',
          'z_solar' : 'Gas Metallicity [ log Z$_{\\rm sun}$ ]',
          'numdens' : 'Gas Density [ log cm$^{-3}$ ]',
          'theta'   : 'Galactocentric Angle [ 0, $\pm\pi$ = major axis ]'}

def explore_vrad_halos(sP, haloIDs):
    """ Exploration: a variety of plots looking at halo-centric gas/wind radial velocities, for individual halos. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    # plot: booklets of 1D profiles / 2D phase diagrams, one per halo
    for haloID in haloIDs:

        pdf = PdfPages('halo_%s_%d_halo-%d.pdf' % (sP.simName,sP.snap,haloID))

        plotHistogram1D([sP], haloIDs=[haloID], ptType='gas', ptProperty='vrad', 
            sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, 
            yQuant='vrelmag', ylim=[0,3000], nBins=nBins, clim=clim)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloID=haloID, pdf=pdf, 
            yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, sfreq0=True, **commonOpts)

        plotPhaseSpace2D(sP, partType='wind_real', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)

        plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['rad'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
            meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)

        plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
            meancolors=['vrad'], weights=None, haloID=haloID, clim=vrad_lim, pdf=pdf)

        pdf.close()

def sample_comparison_z2_sins_ao(sP):
    """ Compare available galaxies vs. the SINS-AO sample of ~35 systems. """
    from util.loadExtern import foersterSchreiber2018

    # config
    xlim = [9.0, 12.0]
    ylim = [-2.5, 4.0]

    msize = 4.0 # marker size for simulated points
    binSize = 0.2 # in M* for median line
    fullSubhaloSFR = True # use total SFR in subhalo, otherwise within 2rhalf

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
    ax = fig.add_subplot(111)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel('Star Formation Rate [ log M$_{\\rm sun}$ / yr ]')
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')

    # load simulation points
    sfrField = 'SubhaloSFR' if fullSubhaloSFR else 'SubhaloSFRinRad'
    fieldsSubhalos = ['SubhaloMassInRadType',sfrField,'central_flag']

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)

    xx_code = gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')]
    xx = sP.units.codeMassToLogMsun( xx_code )

    yy = gc['subhalos'][sfrField]

    # centrals only above some mass limit
    with np.errstate(invalid='ignore'):
        ww = np.where( (xx > xlim[0]+0.2) & gc['subhalos']['central_flag'] )

    w_nonzero = np.where(yy[ww] > 0.0)
    w_zero = np.where(yy[ww] == 0.0)

    l, = ax.plot(xx[ww][w_nonzero], np.log10(yy[ww][w_nonzero]), 'o', markersize=msize, label=sP.simName)
    ax.plot(xx[ww][w_zero], np.zeros(len(w_zero[0])) + ylim[0]+0.1, 'D', markersize=msize, color=l.get_color(), alpha=0.5)

    # median line and 1sigma band
    xm, ym, sm = running_median(xx[ww][w_nonzero],np.log10(yy[ww][w_nonzero]),binSize=binSize,skipZeros=True)
    l, = ax.plot(xm[:-1], ym[:-1], '-', lw=2.0, alpha=0.4, color=l.get_color())

    y_down = np.array(ym[:-1]) - sm[:-1]
    y_up   = np.array(ym[:-1]) + sm[:-1]
    ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

    # observational points (put on top at the end)
    fs = foersterSchreiber2018()
    l1, = ax.plot(fs['Mstar'], np.log10(fs['SFR']), 's', color='#444444', label=fs['label'])

    # second legend
    legend2 = ax.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('sample_comparison_%s_sfrFullSub=%s.pdf' % (sP.simName,fullSubhaloSFR))
    plt.close(fig)

def gasOutflowRatesVsQuant(sP, ptType, xQuant='mstar_30pkpc', eta=False, config=None, massField='Masses'):
    """ Explore radial mass flux data, aggregating into a single Msun/yr value for each galaxy, and plotting 
    trends as a function of stellar mass or any other galaxy/halo property. """

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptTypes = ['Gas','Wind','total']
    assert ptType in ptTypes
    if eta and massField == 'Masses': assert ptType == 'total' # to avoid ambiguity, since massLoadingsSN() is always total
    if massField != 'Masses': assert ptType == 'Gas' # to avoid ambiguity, since other massField's only exist for Gas

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == 'mstar_30pkpc':
        xlim = [7.5, 11.25]
        xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'

    if config is not None and 'xlim' in config: xlim = config['xlim']

    # plot config (y)
    if eta:
        saveBase = 'massLoading'
        pStr1 = ''
        pStr2 = 'w' # 'wind'
        ylim = [-1.15, 1.65] # mass loadings default
        if massField != 'Masses':
            pStr1 = '_{\\rm %s}' % massField
            pStr2 = massField
            ylim = [-10.5, -2.0]
            saveBase += massField
        ylabel = 'Mass Loading $\eta%s = \dot{M}_{\\rm %s} / \dot{M}_\star$ [ log ]' % (pStr1,pStr2)

    else:
        saveBase = 'outflowRate'
        pStr = '%s ' % ptType if ptType != 'total' else ''
        if massField != 'Masses': pStr = '%s ' % massField
        ylabel = '%sOutflow Rate [ log M$_{\\rm sun}$ / yr ]' % pStr
        ylim = [-2.8, 2.5] # outflow rates default
    
    ptStr = '_%s' % ptType
    binSize = 0.2 # in M*
    markersize = 0.0 # 4.0, or 0.0 to disable
    malpha = 0.2
    linestyles = ['-','--',':','-.']

    def _plotHelper(vcutIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,stat='median',skipZeroVals=False,addModelTNG=False):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        labels_sec = []
        colors = []

        if addModelTNG:
            # load mass loading (of TNG model at injection) analysis
            GFM_etaM_mean, _, _, _ = sP.simSubhaloQuantity('wind_etaM')

            # load x-axis property
            GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
            if takeLog: GFM_xquant = logZeroNaN(GFM_xquant)

            if xQuant == 'mstar_30pkpc':
                fit_xlim = [7.5, 11.5] # override

            # plot points
            #with np.errstate(invalid='ignore'):
            #    w = np.where(GFM_etaM_mean > 0)
            #ax.plot(GFM_xquant[w], logZeroNaN(GFM_etaM_mean[w]), 'o', color='red', alpha=0.2)

            # fit
            with np.errstate(invalid='ignore'):
                w_fit = np.where( (GFM_etaM_mean > 0) & (GFM_xquant > fit_xlim[0]) &  (GFM_xquant < fit_xlim[1]) )
            x_fit = GFM_xquant[w_fit]
            y_fit = logZeroNaN(GFM_etaM_mean[w_fit])

            result, resids, rank, singv, rcond = np.polyfit(x_fit,y_fit,2,full=True,cov=False)
            xx = np.linspace(fit_xlim[0],fit_xlim[1],30)
            yy = np.polyval(result,xx)

            # plot fit
            ax.fill_between(xx, yy-0.1, yy+0.1, color='black', interpolate=True, alpha=0.1)
            ax.plot(xx, yy, '-', lw=lw, color='black', alpha=0.6)
            if len(radIndsPlot) == 1:
                ax.text(10.5, 0.03, 'TNG Model (at Injection)', color='black', alpha=0.6, rotation=-43.0)
            else:
                ax.text(10.78, -0.24, 'TNG Model', color='black', alpha=0.6, rotation=-43.0)
                ax.text(10.69, -0.32, '(at Injection)', color='black', alpha=0.6, rotation=-43.0)

        # loop over radii and/or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):

            for j, vcut_ind in enumerate(vcutIndsPlot):
                # local data
                yy = np.squeeze( vals[:,rad_ind,vcut_ind] ).copy() # zero flux -> nan, skipped in median

                # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                if skipZeroVals:
                    w_zero = np.where(yy == 0.0)
                    yy[w_zero] = np.nan
                    # note: currently does nothing given the logZeroNaN() below, which effectively skips zeros regardless

                    yy = logZeroNaN(yy) # zero flux -> nan (skipped in median)

                # label and color
                if rad_ind < binConfig['rad'].size - 1:
                    radMidPoint = '%3d kpc' % (0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1]))
                else:
                    radMidPoint = 'all'

                if len(vcutIndsPlot) == 1:
                    label = 'r = %s' % radMidPoint
                    labelFixed = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                if len(radIndsPlot) == 1:
                    label = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                    labelFixed = 'r = %s' % radMidPoint
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    label = 'r = %s' % radMidPoint # primary label radius, by color
                    labels_sec.append( 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind] ) # second legend: vcut by ls
                    if j > 0: label = ''
                
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    # one color per v_rad, if cycling over both
                    if i == 0:
                        colors.append( ax._get_lines.prop_cycler.next()['color'] )
                    c = colors[j]

                if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
                    c = ax._get_lines.prop_cycler.next()['color']

                # symbols for each system
                if markersize > 0:# or (i==1 and j==1): # hard-coded option
                    size = markersize if markersize > 0 else 4.0
                    yy_mark = yy if skipZeroVals else logZeroNaN(yy)
                    ax.plot(xvals, yy_mark, 's', color=c, markersize=size, alpha=malpha, rasterized=True)

                    # mark those at absolute zero just above the bottom of the y-axis
                    off = 0.2
                    w_zero = np.where(np.isnan(yy_mark))
                    yy_zero = np.random.uniform( size=len(w_zero[0]), low=ylim[0]+off/2, high=ylim[0]+off )
                    ax.plot(xvals[w_zero], yy_zero, 's', alpha=malpha/2, markersize=size, color=c)

                # median line and 1sigma band
                xm, ym, sm, pm = running_median(xvals,yy,binSize=binSize,percs=[16,84],mean=(stat == 'mean'))

                if not skipZeroVals:
                    # take log after running mean/median, instead of before, allows skipZeros=False to have an impact
                    ym = logZeroNaN(ym)
                    sm = logZeroNaN(sm)
                    pm = logZeroNaN(pm)

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                lsInd = i if len(vcutIndsPlot) < 4 else j
                if markersize > 0: lsInd = 0
                l, = ax.plot(xm, ym, linestyles[lsInd], lw=lw, alpha=1.0, color=c, label=label)

                # shade percentile band?
                if i == j or markersize > 0:
                    if 0:
                        nSkip = 1
                        y_down = pm[0,:-nSkip] #np.array(ym[:-1]) - sm[:-1]
                        y_up   = pm[-1,:-nSkip] #np.array(ym[:-1]) + sm[:-1]

                        # repairs
                        w = np.where(np.isnan(y_up))[0]
                        if len(w) and len(w) < len(y_up):
                            lastGoodInd = np.max(w) + 1
                            lastGoodVal = y_up[lastGoodInd] - ym[:-nSkip][lastGoodInd]
                            y_up[w] = ym[:-nSkip][w] + lastGoodVal

                        w = np.where(np.isnan(y_down) & np.isfinite(ym[:-nSkip]))
                        y_down[w] = ylimLoc[0] # off bottom

                        # plot bottom
                        ax.fill_between(xm[:-nSkip], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)
                    if 1:
                        y_down = pm[0,:] #np.array(ym[:-1]) - sm[:-1]
                        y_up   = pm[-1,:] #np.array(ym[:-1]) + sm[:-1]

                        # repairs
                        w = np.where(np.isnan(y_up))[0]
                        if len(w) and len(w) < len(y_up):
                            lastGoodInd = np.max(w) + 1
                            lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                            y_up[w] = ym[:][w] + lastGoodVal

                        w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                        y_down[w] = ylimLoc[0] # off bottom

                        # plot bottom
                        ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc='lower right' if len(radIndsPlot) > 1 else 'lower left', 
                                handlelength=-0.5, frameon=1, framealpha=0.5, borderpad=0.2, fancybox=False)
            #for text in legend2.get_texts(): text.set_color('white')
            frame = legend2.get_frame()
            frame.set_facecolor('white')
            ax.add_artist(legend2)

            legend1 = ax.legend(loc='upper right' if eta else 'upper left')

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color=colors[j], marker='', lw=lw, linestyle='-') for j in range(len(vcutIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc='upper right')
            ax.add_artist(legend2)

            legend1 = ax.legend(loc='lower right' if eta else 'upper left')
            for handle in legend1.legendHandles: handle.set_color('black')

        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    mdot = {}
    mdot['Gas'], mstar_30pkpc_log, sub_ids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, 'Gas', massField=massField)

    if massField == 'Masses':
        mdot['Wind'], _, sub_ids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, 'Wind', massField=massField)
        mdot['total'] = mdot['Gas'] + mdot['Wind']
    else:
        mdot['total'] = mdot['Gas']

    # load mass loadings (total)
    acField = 'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr'
    if massField != 'Masses':
        acField = 'Subhalo_MassLoadingSN_%s_SubfindWithFuzz_SFR-100myr' % massField

    etaM = sP.auxCat(acField)[acField]

    if eta:
        vals = etaM
    else:
        vals = mdot[ptType]

    # restrict Mdot/etaM values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
    if config is not None and 'minMstar' in config:
        w = np.where(mstar_30pkpc_log < config['minMstar'])
        vals[w] = np.nan

    # load x-axis values, stellar mass or other?
    xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
    xvals = xvals[sub_ids]
    if takeLog: xvals = logZeroNaN(xvals)

    if xlabel is None: xlabel = xlabel2 # use suggestions if not hard-coded above
    if xlim is None: xlim = xlim2

    # one specific plot requested? make now and exit
    if config is not None:
        saveName = '%s%s_%s_%s_%d_v%dr%d_%s_skipzeros-%s.pdf' % \
          (saveBase,ptStr,xQuant,sP.simName,sP.snap,len(config['vcutInds']),len(config['radInds']),config['stat'],config['skipZeros'])
        if 'saveName' in config: saveName = config['saveName']
        if 'markersize' in config: markersize = config['markersize']
        if 'addModelTNG' not in config: config['addModelTNG'] = False
        if 'ylim' not in config: config['ylim'] = None

        _plotHelper(vcutIndsPlot=config['vcutInds'],radIndsPlot=config['radInds'],saveName=saveName,
                    stat=config['stat'],skipZeroVals=config['skipZeros'],ylimLoc=config['ylim'],addModelTNG=config['addModelTNG'])
        return

    # plot
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:
            print(ptType,stat,'zeros:',skipZeros,'eta:',eta)
            # (A) plot for a given vcut, at many radii
            radInds = [1,3,4,5,6,7]

            pdf = PdfPages('%s%s_%s_A_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros))
            for vcut_ind in range(vcut_vals.size):
                _plotHelper(vcutIndsPlot=[vcut_ind],radIndsPlot=radInds,pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (B) plot for a given radii, at many vcuts
            vcutInds = [0,1,2,3,4]

            pdf = PdfPages('%s%s_%s_B_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros))
            for rad_ind in range(numBins['rad']):
                _plotHelper(vcutIndsPlot=vcutInds,radIndsPlot=[rad_ind],pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (C) single-panel combination of both radial and vcut variations
            if ptType in ['Gas','total']:
                vcutIndsPlot = [0,2,3]
                radIndsPlot = [1,2,5]
                ylimLoc = [-2.5,2.0] if not eta else ylim

            if ptType == 'Wind':
                vcutIndsPlot = [0,2,4]
                radIndsPlot = [1,2,5]
                ylimLoc = [-3.0,1.0]

            saveName = '%s%s_%s_C_%s_%d_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,sP.snap,stat,skipZeros)
            _plotHelper(vcutIndsPlot,radIndsPlot,saveName,ylimLoc=ylimLoc,stat=stat,skipZeroVals=skipZeros)

def gasOutflowVelocityVsQuant(sP_in, xQuant='mstar_30pkpc', ylog=False, redshifts=[None], config=None, massField='Masses'):
    """ Explore outflow velocity, aggregating into a single vout [km/s] value for each galaxy, and plotting 
    trends as a function of stellar mass or any other galaxy/halo property. """
    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'

    mdotThreshVcutInd = 0 # vrad>0 km/s
    mdotThreshValue   = 0.0 # msun/yr

    # plot config (y)
    ylim = [0, 1200]

    if massField == 'Masses':
        ylabel = 'Outflow Velocity [ km/s ]'
        saveBase = 'outflowVelocity'
        ptStr = '_total'
    else:
        ylabel = '%s Outflow Velocity [ km/s ]' % massField
        saveBase = 'outflowVelocity%s' % massField
        ptStr = '_Gas'

    if ylog:
        ylabel = ylabel.replace('km/s','log km/s')
        ylim = [1.4, 3.2]

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == 'mstar_30pkpc':
        xlim = [7.5, 11.0]
        xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
    if 'etaM' in xQuant:
        xlim = [0.0, 2.7] # explore


    binSize = 0.2 # in M*
    markersize = 0.0 # 4.0, or 0.0 to disable
    malpha = 0.2
    percs = [16,84]
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])

    def _plotHelper(percIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,stat='median',skipZeroVals=False,addModelTNG=False):
        """ Plot a radii series, vcut series, or both. """
        if len(redshifts) > 1: assert len(radIndsPlot) == 1 # otherwise needs generalization

        # plot setup
        fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if config is not None and 'xlabel' in config: ax.set_xlabel(config['xlabel'])
        if config is not None and 'ylabel' in config: ax.set_ylabel(config['ylabel'])

        labels_sec = []

        # TNG minimum velocity band
        if 'mstar' in xQuant or 'mhalo' in xQuant:
            minVel = 350.0 if not ylog else np.log10(350.0)
            minVelTextY = 370.0 if not ylog else 2.58
            ax.fill_between(xlim, [0,0], [minVel,minVel], color='#cccccc', alpha=0.05)
            ax.plot(xlim, [minVel,minVel], '-', lw=lw, color='#cccccc', alpha=0.5)
            ax.text(xlim[0] + (xlim[1]-xlim[0])*0.04, minVelTextY, 'TNG v$_{\\rm wind,min}$ = 350 km/s', color='black', alpha=0.6)

        if addModelTNG:
            # loop over multiple-redshifts if requested
            redshift0 = sP.redshift
            redshiftsToDo = redshifts if redshifts[0] is not None else [sP.redshift]
            for k, redshift in enumerate(redshiftsToDo):
                sP.setRedshift(redshift)

                # load velocity (of TNG model at injection) analysis
                GFM_windvel_mean, _, _, _ = sP.simSubhaloQuantity('wind_vel')
                if ylog: GFM_windvel_mean = logZeroNaN(GFM_windvel_mean)

                # load x-axis property
                GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
                if takeLog: GFM_xquant = logZeroNaN(GFM_xquant)

                if xQuant == 'mstar_30pkpc':
                    fit_xlim = [8.5, 11.5] #[8.7, 11.5] # override

                assert GFM_windvel_mean.shape == GFM_xquant.shape

                # plot individual points
                #w = np.where(GFM_windvel_mean > 0)
                #ax.plot(GFM_xquant[w], GFM_windvel_mean[w], 'o', color='black', alpha=0.2)

                # fit
                if 0:
                    with np.errstate(invalid='ignore'):
                        w_fit = np.where( (GFM_windvel_mean > 0) & (GFM_xquant > fit_xlim[0]) &  (GFM_xquant < fit_xlim[1]) )
                    x_fit = GFM_xquant[w_fit]
                    y_fit = GFM_windvel_mean[w_fit]

                    result, resids, rank, singv, rcond = np.polyfit(x_fit,y_fit,6,full=True,cov=False)
                    xx = np.linspace(fit_xlim[0],fit_xlim[1],20)
                    yy = np.polyval(result,xx)
                    xx = np.insert(xx, 0, ax.get_xlim()[0])
                    yy = np.insert(yy, 0, yy[0])

                    # plot
                    ax.fill_between(xx, yy-25, yy+25, color='black', interpolate=True, alpha=0.1)
                    ax.plot(xx, yy, '-', lw=lw, color='black', alpha=0.6, label='TNG Model (at Injection)')

                # median
                with np.errstate(invalid='ignore'):
                    w_fit = np.where( (GFM_windvel_mean > 0) )
                x_fit = GFM_xquant[w_fit]
                y_fit = GFM_windvel_mean[w_fit]

                xm, ym, sm, pm = running_median(x_fit,y_fit,binSize=0.2,percs=[16,84])

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                alpha = 0.1 if len(redshifts) == 1 else 0.05
                ax.fill_between(xm, pm[0,:], pm[-1,:], color='black', interpolate=True, alpha=alpha)
                xm2 = np.linspace(xm.min(), xm.max(), 100)
                ym2 = interp1d(xm, ym, kind='cubic', fill_value='extrapolate')(xm2)

                label = 'TNG Model (at Injection)' if k == 0 else ''
                #if len(redshifts) > 1: label = 'TNG model (z=%.1f)' % redshift
                if len(redshifts) > 1: # special case labeling
                    ax.text(9.66, 720.0, '$z$ = 6', color='#888888', rotation=70.0)
                    ax.text(9.88, 730.0, '1 < $z$ < 4', color='#888888', rotation=65.0)

                alpha = 0.6 if len(redshifts) == 1 else 0.5
                ax.plot(xm2, ym2, '-', lw=lw, linestyle=linestyles[k], color='black', alpha=alpha, label=label)

        # loop over redshifts
        for k, redshift in enumerate(redshifts):
            # get local data
            mdot, xx, binConfig, numBins, vals, percs = data[k]

            # loop over radii or vcut selections
            for i, rad_ind in enumerate(radIndsPlot):

                if (len(percIndsPlot) > 1 and len(radIndsPlot) > 1) or len(redshifts) > 1:
                    c = ax._get_lines.prop_cycler.next()['color'] # one color per rad, if cycling over both

                # local data (outflow rates in this radial bin)
                if rad_ind < mdot.shape[1]:
                    mdot_local = mdot[:,rad_ind]
                else:
                    mdot_local = np.sum(mdot, axis=1) # 'all'

                for j, perc_ind in enumerate(percIndsPlot):
                    # local data (velocities in this radial/perc bin)
                    yy = np.squeeze( vals[:,rad_ind,perc_ind] ).copy() # zero flux -> nan, skipped in median

                    # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                    if skipZeroVals:
                        w_zero = np.where(yy == 0.0)
                        yy[w_zero] = np.nan

                    # mdot < threshold: exclude
                    w_below = np.where(mdot_local < mdotThreshValue)
                    yy[w_below] = np.nan

                    if ylog:
                        yy = logZeroNaN(yy)

                    # label and color
                    labelFixed = None
                    if rad_ind == numBins['rad']:
                        radMidPoint = 'all'
                    else:
                        radMidPoint = '%3d kpc' % (0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1]))

                    if len(percIndsPlot) == 1:
                        label = 'r = %s' % radMidPoint
                        labelFixed = 'v$_{\\rm out,%d}$' % percs[perc_ind]
                    if len(radIndsPlot) == 1:
                        label = 'v$_{\\rm out,%d}$' % percs[perc_ind]
                        labelFixed = 'r = %s' % radMidPoint
                    if len(percIndsPlot) > 1 and (len(radIndsPlot) > 1 or len(redshifts) > 1):
                        label = 'r = %s' % radMidPoint # primary label radius, by color
                        labels_sec.append( 'v$_{\\rm out,%d}$' % percs[perc_ind] ) # second legend: vcut by ls
                        if j > 0: label = ''
                    if len(redshifts) == 1:
                        if labelFixed is None or 'mstar' not in xQuant:
                            labelFixed = 'z = %.1f' % sP.redshift
                        else:
                            labelFixed += ', z = %.1f' % sP.redshift

                    if len(redshifts) > 1:
                        label = 'z = %.1f' % redshift
                    if len(redshifts) > 1 and j > 0:
                        label = '' # move percs to separate labels

                    if (len(percIndsPlot) == 1 or len(radIndsPlot) == 1) and len(redshifts) == 1:
                        c = ax._get_lines.prop_cycler.next()['color']

                    # symbols for each system
                    if markersize > 0:
                        ax.plot(xx, yy, 's', color=c, markersize=markersize, alpha=malpha, rasterized=True)

                        # mark those at absolute zero just above the bottom of the y-axis
                        off = 10
                        w_zero = np.where(np.isnan(yy))
                        yy_zero = np.random.uniform( size=len(w_zero[0]), low=ylim[0]+off/2, high=ylim[0]+off )
                        ax.plot(xx[w_zero], yy_zero, 's', alpha=malpha/2, markersize=markersize, color=c)

                    # median line and 1sigma band
                    minNum = 5 if 'mstar' in xQuant else 2 # for xQuants = etaM, SFR, ...
                    xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,percs=percs,mean=(stat == 'mean'),minNumPerBin=minNum)

                    if xm.size > sKn:
                        ym = savgol_filter(ym,sKn,sKo)
                        sm = savgol_filter(sm,sKn,sKo)
                        pm = savgol_filter(pm,sKn,sKo,axis=1)

                    lsInd = j if len(percIndsPlot) < 4 else i
                    if markersize > 0: lsInd = 0

                    #xm2 = np.linspace(xm.min(), xm.max(), 100)
                    #ym2 = interp1d(xm, ym, kind='cubic', fill_value='extrapolate')(xm2)

                    l, = ax.plot(xm, ym, linestyles[lsInd], lw=lw, alpha=1.0, color=c, label=label)

                    # shade percentile band?
                    if i == j or markersize > 0 or 'mstar' not in xQuant:
                        y_down = pm[0,:] #np.array(ym[:-1]) - sm[:-1]
                        y_up   = pm[-1,:] #np.array(ym[:-1]) + sm[:-1]

                        # repairs
                        w = np.where(np.isnan(y_up))[0]
                        if len(w) and len(w) < len(y_up) and w.max()+1 < y_up.size:
                            lastGoodInd = np.max(w) + 1
                            lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                            y_up[w] = ym[:][w] + lastGoodVal

                        w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                        y_down[w] = ylimLoc[0] # off bottom

                        # plot bottom
                        ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # special behavior (including observational data sets)
        if 'etaM_' in xQuant:
            xx = np.array([0.4, 1.39])
            yy = 900 * (10.0**xx)**(-1.0) # 'momentum driven', 1000 is arbitrary, this is proportionality only
            if ylog: yy = np.log10(yy)
            ax.plot(xx, yy, '--', lw=lw, color='black', alpha=0.6)
            ax.text(0.37, 2.46, '$\eta_{\\rm M} \propto v_{\\rm out}^{-1}$', color='#555555', rotation=-50.0)

            xx = np.array([0.4, 2.0])
            yy = 1000 * (10.0**xx)**(-0.5) # 'energy driven'
            if ylog: yy = np.log10(yy)
            ax.plot(xx, yy, '--', lw=lw, color='black', alpha=0.6)
            ax.text(0.4, 2.705, '$\eta_{\\rm M} \propto v_{\\rm out}^{-2}$', color='#555555', rotation=-31.0)

        if 'sfr_' in xQuant:
            # obs data: v_out vs SFR
            from util.loadExtern import chen10, rubin14, robertsborsani18

            color = '#555555'
            labels = []

            obs = chen10()
            for i, ref in enumerate(obs['labels']):
                w = np.where(obs['ref'] == ref)
                ax.plot( obs['sfr'][w], obs['vout'][w], markers[i], color=color)
                labels.append( ref )

            obs = robertsborsani18()
            ax.plot( obs['sfr'], obs['vout'], markers[i+1], color=color)
            labels.append( obs['label'] )

            obs = rubin14()
            ax.plot( obs['sfr'], obs['vout'], markers[i+2], color=color)
            labels.append( obs['label'] )

            # legend
            handles = [ plt.Line2D( (0,1), (0,0), color=color, marker=markers[i], lw=lw, linestyle='') for i in range(len(labels)) ]
            legend3 = ax.legend(handles, labels, ncol=2, columnspacing=1.0, fontsize=18.0, loc='upper left')
            ax.add_artist(legend3)

        # legends and finish plot
        loc1 = 'lower right' if (config is None or 'loc1' not in config) else config['loc1']
        if len(percIndsPlot) == 1 or len(radIndsPlot) == 1:
            line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc=loc1)
            ax.add_artist(legend2)

        if len(percIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[j]) for j in range(len(percIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc=loc1)
            ax.add_artist(legend2)

        handles, labels = ax.get_legend_handles_labels()
        if len(redshifts) > 1 and len(percIndsPlot) > 1:
            handles += [ plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[j]) for j in range(len(percIndsPlot)) ]
            labels += labels_sec
        loc2 = 'upper right' if (config is None or 'loc2' not in config) else config['loc2']
        legend1 = ax.legend(handles, labels, loc=loc2)

        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates and outflow velocities (total)
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)

        mdot, mstar_30pkpc_log, sub_ids, binConfig, numBins, _ = loadRadialMassFluxes(sP, scope, 'Gas', massField=massField)
        mdot = mdot[:,:,mdotThreshVcutInd]

        acField = 'Subhalo_OutflowVelocity_%s' % scope
        if massField != 'Masses':
            acField = 'Subhalo_OutflowVelocity_%s_%s' % (massField,scope)

        ac = sP.auxCat(acField)

        vals  = ac[acField]
        percs = ac[acField + '_attrs']['percs']

        # restrict included v_out values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
        if config is not None and 'minMstar' in config:
            w = np.where(mstar_30pkpc_log < config['minMstar'])
            vals[w] = np.nan

        # load x-axis values, stellar mass or other?
        xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
        if takeLog: xvals = logZeroNaN(xvals[sub_ids])

        if xlabel is None: xlabel = xlabel2 # use suggestions if not hard-coded above
        if xlim is None: xlim = xlim2

        # save one data-list per redshift
        data.append( [mdot,xvals,binConfig,numBins,vals,percs] )

    allRadInd = vals.shape[1] - 1 # last bin is not a radial bin, but all radii combined

    # one specific plot requested? make now and exit
    if config is not None:
        saveName = '%s%s_%s_%s_%s_nr%d_np%d_%s_skipzeros-%s.pdf' % \
          (saveBase,ptStr,xQuant,sP.simName,zStr,len(config['radInds']),len(config['percInds']),config['stat'],config['skipZeros'])
        if 'saveName' in config: saveName = config['saveName']
        if 'ylim' not in config: config['ylim'] = None
        if 'xlim' in config: xlim = config['xlim']
        if 'addModelTNG' not in config: config['addModelTNG'] = False
        if 'markersize' in config: markersize = config['markersize']
        if 'binSize' in config: binSize = config['binSize']
        if 'percs' in config: percs = config['percs']

        _plotHelper(percIndsPlot=config['percInds'],radIndsPlot=config['radInds'],saveName=saveName,
                    ylimLoc=config['ylim'],stat=config['stat'],skipZeroVals=config['skipZeros'],
                    addModelTNG=config['addModelTNG'])
        return

    # plot
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:
            # (A) plot for a given perc, at many radii
            radInds = [1,3,4,5,6,7,allRadInd]

            pdf = PdfPages('%s%s_%s_A_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros))
            for perc_ind in range(percs.size):
                _plotHelper(percIndsPlot=[perc_ind],radIndsPlot=radInds,pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (B) plot for a given radii, at many percs
            percInds = [0,1,2,3,4,5]

            pdf = PdfPages('%s%s_%s_B_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros))
            for rad_ind in range(numBins['rad']+1): # last one is 'all'
                _plotHelper(percIndsPlot=percInds,radIndsPlot=[rad_ind],pdf=pdf,stat=stat,skipZeroVals=skipZeros)
            pdf.close()

            # (C) single-panel combination of both radial and perc variations
            percIndsPlot = [1,2,4]
            radIndsPlot = [1,2,13]
            ylimLoc = [0,800] if not ylog else [1.5,3.0]

            saveName = '%s%s_%s_C_%s_%s_%s_skipzeros-%s.pdf' % (saveBase,ptStr,xQuant,sP.simName,zStr,stat,skipZeros)
            _plotHelper(percIndsPlot,radIndsPlot,saveName,ylimLoc=ylimLoc,stat=stat,skipZeroVals=skipZeros)

def gasOutflowRatesVsQuantStackedInMstar(sP_in, quant, mStarBins, redshifts=[None], config=None):
    """ Explore radial mass flux data, as a function of one of the histogrammed quantities (x-axis), for single 
    galaxies or stacked in bins of stellar mass. Optionally at multiple redshifts. """
    import warnings

    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptType = 'Gas'

    # plot config
    ylim = [-3.0,2.0] if (config is None or 'ylim' not in config) else config['ylim']
    vcuts = [0,1,2,3,4] if quant != 'vrad' else [None]
    linestyles = ['-','--',':','-.']
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])

    limits = {'temp'    : [2.9,8.1],
              'z_solar' : [-3.0,1.0],
              'numdens' : [-5.0,2.0],
              'vrad'    : [0, 3000],
              'theta'   : [-np.pi,np.pi]}

    if len(redshifts) > 1:
        # multi-z plots restricted to smaller M* bins, modify limits
        limits = {'temp'    : [3.0,8.0],
                  'z_solar' : [-2.0,1.0],
                  'numdens' : [-5.0,2.0],
                  'vrad'    : [0, 1800],
                  'theta'   : [-np.pi,np.pi]}

    def _plotHelper(vcut_ind,rad_ind,quant,mStarBins=None,stat='mean',skipZeroFluxes=False,saveName=None,pdf=None):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
        ax = fig.add_subplot(111)
        
        ax.set_xlim(limits[quant])
        ax.set_ylim(ylim)

        ax.set_xlabel(labels[quant])
        ax.set_ylabel('%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType)

        if quant == 'theta':
            # special x-axis labels for angle theta
            ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
            ax.set_xticklabels(['$-\pi$','$-\pi/2$ (minor axis)','$0$','$+\pi/2$ (minor axis)','$+\pi$'])
            ax.plot([-np.pi/2,-np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)
            ax.plot([+np.pi/2,+np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)

        # loop over redshifts
        colors = []

        for j, redshift in enumerate(redshifts):
            # get local data
            mdot, mstar, subids, binConfig, numBins, vcut_vals = data[j]

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):

                if j == 0:
                    c = ax._get_lines.prop_cycler.next()['color'] # one color per bin (fixed across redshift)
                    colors.append(c)
                else:
                    c = colors[i]

                # local data
                w = np.where( (mstar > mStarBin[0]) & (mstar <= mStarBin[1]) )
                #print(mStarBin, ' number of galaxies: ',len(w[0]))

                if vcut_ind is not None:
                    mdot_local = np.squeeze( mdot[w,rad_ind,vcut_ind,:] ).copy()
                else:
                    mdot_local = np.squeeze( mdot[w,rad_ind,:] ).copy() # plot: mdot vs. vrad directly

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if skipZeroFluxes:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # avoid RuntimeWarning: Mean of empty slice (single galaxies with only zero values)
                    if stat == 'median':
                        yy = np.nanmedian(mdot_local, axis=0) # median on subhalo axis
                    if stat == 'mean':
                        yy = np.nanmean(mdot_local, axis=0) # mean

                    # median line and 1sigma band
                    pm = np.nanpercentile(mdot_local, [16,84], axis=0, interpolation='linear')
                    pm = logZeroNaN(pm)
                    sm = np.nanstd(logZeroNaN(mdot_local), axis=0)

                yy = logZeroNaN(yy) # zero flux -> nan

                # label and color
                xx = 0.5*(binConfig[quant][:-1] + binConfig[quant][1:])

                radMidPoint   = 0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1])
                mStarMidPoint = 0.5*(mStarBin[0] + mStarBin[1])

                labelFixed = 'r = %3d kpc' % radMidPoint
                if vcut_ind is not None:
                    labelFixed += ', v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                if len(redshifts) == 1:
                    labelFixed += ', z = %.1f' % sP.redshift

                label = 'M$^\star$ = %.1f' % mStarMidPoint if j == 0 else '' # label M* only once

                yy = savgol_filter(yy,sKn,sKo)
                if pm.ndim > 1:
                    pm = savgol_filter(pm,sKn,sKo,axis=1)
                sm = savgol_filter(sm,sKn,sKo)

                #l, = ax.plot(xm[:-1], ym[:-1], linestyles[i], lw=lw, alpha=1.0, color=c, label=label)
                l, = ax.plot(xx, yy, linestyles[0], linestyle=linestyles[j], lw=lw, alpha=1.0, color=c, label=label)

                if j == 0 and (i == 0 or i == len(mStarBins)-1):
                    #w = np.where( np.isfinite(pm[0,:]) & np.isfinite(pm[-1,:]) )[0]
                    #ax.fill_between(xx[w], pm[0,w], pm[-1,w], color=l.get_color(), interpolate=True, alpha=0.05)
                    w = np.where( np.isfinite(yy) )
                    ax.fill_between(xx[w], yy[w]-sm[w], yy[w]+sm[w], color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        if len(redshifts) > 1:
            sExtra = []
            lExtra = []

            for j, redshift in enumerate(redshifts):
                sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[j],marker='')]
                lExtra += ['z = %.1f' % redshift]

            legend2 = ax.legend(sExtra, lExtra, loc='lower right')
            ax.add_artist(legend2)

        line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
        loc = 'upper left' if (len(redshifts) > 1 or quant == 'vrad' or quant == 'numdens') else 'lower right'
        if quant == 'temp': loc = 'upper right'
        legend3 = ax.legend([line], [labelFixed], handlelength=-0.5, loc=loc)
        ax.add_artist(legend3) # "r = X kpc" or "r = X kpc, z = Y"

        legend1 = ax.legend(loc='upper right' if quant != 'temp' else 'upper left') # M* bins

        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)
        data.append( loadRadialMassFluxes(sP, scope, ptType, thirdQuant=quant) )

    if config is not None:
        saveName = 'outflowRate_%s_%s_mstar_%s_%s_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,zStr,config['stat'],config['skipZeros'])
        if 'vcutInd' not in config: config['vcutInd'] = None # quant == vrad
        _plotHelper(config['vcutInd'],config['radInd'],quant,mStarBins,config['stat'],skipZeroFluxes=config['skipZeros'],saveName=saveName)
        return

    for stat in ['mean']: #['mean','median']:
        for skipZeros in [False]: #[True,False]:
            print(quant,stat,skipZeros)

            # (A) vs quant, booklet across rad and vcut variations
            pdf = PdfPages('outflowRate_A_%s_%s_mstar_%s_%s_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,zStr,stat,skipZeros))

            for radInd in [1,3,4,5,6,7]:
                for vcutInd in vcuts:
                    _plotHelper(vcutInd,radInd,quant,mStarBins,stat,skipZeroFluxes=skipZeros,pdf=pdf)

            pdf.close()

def gasOutflowRates2DStackedInMstar(sP_in, xAxis, yAxis, mStarBins, redshifts=[None], 
      clims=[[-3.0,2.0]], config=None, eta=False, discreteColors=False, contours=None):
    """ Explore radial mass flux data, 2D panels where color indicates Mdot_out. 
    Give clims as a list, one per mStarBin, or if just one element, use the same for all bins.
    If config is None, generate many exploration plots, otherwise just create the single desired plot. 
    If eta is True, plot always mass-loadings instead of mass-outflow rates. 
    If discreteColors is True, split the otherwise continuous colormap into discrete segments. """

    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant) # copy

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptType = 'Gas'
    cStr = '_contour' if contours is not None else ''

    if eta:
        cbarlabel = '%s Mass Loading $\eta = \dot{M}_{\\rm w} / \dot{M}_\star$ [ log ]' % ptType
        cbarlabel2 = '%s Mass Loading (Inflow) [ log ]' % ptType
        saveBase = 'massLoading2D'
        contourlabel = 'log $\eta$'
    else:
        cbarlabel = '%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType
        cbarlabel2 = '%s Inflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType
        saveBase = 'outflowRate2D'
        contourlabel = 'log $\dot{M}_{\\rm out}$'

    if len(clims) == 1: clims = [clims[0]] * len(mStarBins) # one for each
    assert yAxis != 'rad' # keep on x-axis if wanted
    assert xAxis != 'vcut' # keep on y-axis if wanted

    # plot config
    limits = {'rad'     : None, # discrete labels (small number of bins)
              'vrad'    : False, # fill from binConfig
              'vcut'    : None, # discrete labels (small number of bins)
              'temp'    : False, # fill from binConfig
              'z_solar' : False, # fill from binConfig
              'numdens' : False, # fill from binConfig
              'theta'   : [-np.pi,np.pi]} # always fixed

    def _plotHelper(xAxis,yAxis,mStarBins=None,stat='mean',skipZeroFluxes=False,saveName=None,vcut_ind=None,rad_ind=None,pdf=None):
        """ Plot a number of 2D histogram panels. mdot_2d should have 3 dimensions: [subhalo_ids, xAxis_quant, yAxis_quant]. """

        # replicate vcut/rad indices into lists, one per mass bin, if not already
        if not isinstance(vcut_ind,list):
            vcut_ind = [vcut_ind] * len(mStarBins)
        if not isinstance(rad_ind,list):
            rad_ind = [rad_ind] * len(mStarBins)

        if contours is not None:
            # only 1 panel for all M*/redshift variations, make panel now
            fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])

            ax = fig.add_subplot(111)
            lines = []
            labels1 = []
            colors = sampleColorTable('tab10',len(contours))
        else:
            # non-contour plot setup: multi-panel
            nRows = int(np.floor(np.sqrt(len(mStarBins))))
            nCols = int(np.ceil(len(mStarBins) / nRows))
            nRows, nCols = nCols, nRows

            fig = plt.figure(figsize=[figsize[0]*sfclean*nCols, figsize[1]*sfclean*nRows])

        # loop over redshifts
        for j, redshift in enumerate(redshifts):
            # get local data
            mdot_in, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed = data[j]

            # axes are always (rad,vcut), i.e. we never actually slice mdot_in
            if all(ind is None for ind in rad_ind+vcut_ind):
                mdot_2d = mdot_in.copy()

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):

                # create local mdot, resize if needed (final dimensions are [subhalos, xaxis_quant, yaxis_quant])
                if vcut_ind[i] is not None and rad_ind[i] is None:
                    mdot_2d = np.squeeze( mdot_in[:,:,vcut_ind[i],:] ).copy()
                if rad_ind[i] is not None and vcut_ind[i] is None:
                    mdot_2d = np.squeeze( mdot_in[:,rad_ind[i],:,:] ).copy()
                    if yAxis == 'vcut':
                        mdot_2d = np.swapaxes(mdot_2d, 1, 2) # put vcut as last (y) axis
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    mdot_2d = np.squeeze( mdot_in[:,rad_ind[i],vcut_ind[i],:,:] ).copy()

                # start panel
                if contours is None:
                    ax = fig.add_subplot(nRows,nCols,i+1)
                
                xlim = limits[xAxis] if limits[xAxis] is not None else [0,mdot_2d.shape[1]]
                ylim = limits[yAxis] if limits[yAxis] is not None else [0,mdot_2d.shape[2]]
                ax.set_xlim(xlim) 
                ax.set_ylim(ylim)

                assert mdot_2d.shape[1] == binConfig[xAxis].size - 1
                assert mdot_2d.shape[2] == binConfig[yAxis].size - 1 if yAxis != 'vcut' else binConfig[yAxis].size

                ax.set_xlabel(labels[xAxis])
                ax.set_ylabel(labels[yAxis])

                # local data
                w = np.where( (mstar > mStarBin[0]) & (mstar <= mStarBin[1]) )
                mdot_local = np.squeeze( mdot_2d[w,:,:] ).copy()
                print('z=',redshift,' M* bin:',mStarBin, ' number of galaxies: ',len(w[0]))

                if eta:
                    # normalize each included system by its smoothed SFR (expand dimensionality for broadcasting)
                    sfr_norm = sfr_smoothed[w[0],None,None]
                    w = np.where(sfr_norm > 0.0)
                    mdot_local[w,:,:] /= sfr_norm[w,:,:]

                    w = np.where(sfr_norm == 0.0)
                    mdot_local[w,:,:] = np.nan

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if skipZeroFluxes:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                if stat == 'median':
                    h2d = np.nanmedian(mdot_local, axis=0) # median on subhalo axis
                if stat == 'mean':
                    h2d = np.nanmean(mdot_local, axis=0) # mean

                # handle negative values (inflow) so they exist post-log, by separating the matrix into positive and negative components
                h2d_pos = h2d.copy()
                h2d_neg = h2d.copy()
                h2d_pos[np.where(h2d < 0.0)] = np.nan
                h2d_neg[np.where(h2d >= 0.0)] = np.nan

                h2d = logZeroNaN(h2d) # zero flux -> nan
                h2d_pos = logZeroNaN(h2d_pos)
                h2d_neg = logZeroNaN(-h2d_neg)

                #h2d = sgolay2d(h2d,sKn,sKo) # smoothing

                # set NaN/blank to minimum color
                with np.errstate(invalid='ignore'):
                    w_neg_clip = np.where(h2d_neg < (clims[i][0] + (clims[i][1]-clims[i][0])*0.1) )
                    h2d_neg[w_neg_clip] = np.nan # 10% clip near bottom (black) edge of colormap, let these pixels stay as pos background color

                w = np.where(np.isnan(h2d_pos) & np.isnan(h2d_neg))
                h2d_pos[w] = clims[i][0]
                #h2d_neg[w] = clims[i][0] # let h2d_pos assign 'background' color for all nan pixels

                # set special x/y axis labels? on a small, discrete number of bins
                if limits[xAxis] is None:
                    xx = list( 0.5*(binConfig[xAxis][:-1] + binConfig[xAxis][1:]) )
                    if np.isinf(xx[-1]):
                        # last bin was some [finite,np.inf] range
                        xx[-1] = '>%s' % binConfig[xAxis][-2]

                    if np.isinf(xx[0]) or binConfig[xAxis][0] == 0.0:
                        # first bin was some [-np.inf,finite] or [0,finite] range (midpoint means little)
                        xx[0] = '<%s' % binConfig[xAxis][1]

                    xticklabels = []
                    for xval in xx:
                        xticklabels.append( xval if isinstance(xval,basestring) else '%d' % xval )

                    ax.set_xticks(np.arange(mdot_2d.shape[1]) + 0.5)
                    ax.set_xticklabels(xticklabels)
                    assert len(xx) == mdot_2d.shape[1]

                if limits[yAxis] is None:
                    yy = list( 0.5*(binConfig[yAxis][:-1] + binConfig[yAxis][1:]) ) if yAxis != 'vcut' else list(binConfig[yAxis])

                    yticklabels = []
                    for yval in yy:
                        yticklabels.append( yval if isinstance(yval,basestring) else '%d' % yval )

                    ax.set_yticks(np.arange(mdot_2d.shape[2]) + 0.5)
                    ax.set_yticklabels(yticklabels)
                    assert len(yy) == mdot_2d.shape[2]

                # label
                mStarMidPoint = 0.5*(mStarBin[0] + mStarBin[1])
                label1 = 'M$^\star$ = %.1f' % mStarMidPoint
                label2 = None

                if len(mStarBins) == 1 and len(redshifts) > 1:
                    label1 = 'z = %.1f' % redshift
                
                if j > 0: label1 = '' # only label on first redshift

                if vcut_ind[i] is not None:
                    label2 = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind[i]]
                if rad_ind[i] is not None:
                    radMidPoint = 0.5*(binConfig['rad'][rad_ind[i]] + binConfig['rad'][rad_ind[i]+1])
                    label2 = 'r = %3d kpc' % radMidPoint
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    radMidPoint = 0.5*(binConfig['rad'][rad_ind[i]] + binConfig['rad'][rad_ind[i]+1])
                    label2 = 'r = %3d kpc, v$_{\\rm rad}$ > %3d km/s' % (radMidPoint,vcut_vals[vcut_ind[i]])

                # plot: positive and negative components separately
                norm = Normalize(vmin=clims[i][0], vmax=clims[i][1])

                numColors = None # continuous cmap
                if discreteColors:
                    numColors = (clims[i][1] - clims[i][0]) * 2 # discrete for each 0.5 interval
                cmap_pos = loadColorTable('viridis', numColors=numColors)
                cmap_neg = loadColorTable('inferno', numColors=numColors)

                imOpts = {'extent':[xlim[0],xlim[1],ylim[0],ylim[1]], 'origin':'lower', 'interpolation':'nearest', 'aspect':'auto'}

                if contours is None:
                    # 2D histogram image
                    im_neg = plt.imshow(h2d_neg.T, cmap=cmap_neg, norm=norm, **imOpts)
                    im_pos = plt.imshow(h2d_pos.T, cmap=cmap_pos, norm=norm, **imOpts)

                    ax.set_facecolor(cmap_pos(0.0)) # set background color inside plot axes to lowest value instead of white, to prevent boundary artifacts

                else:
                    # 2D contour: first resample, increasing resolution
                    XX = np.linspace(xlim[0], xlim[1], mdot_2d.shape[1])
                    YY = np.linspace(ylim[0], ylim[1], mdot_2d.shape[2])

                    # origin space
                    grid_x, grid_y = np.meshgrid(XX, YY, indexing='ij')
                    grid_xy = np.zeros( (grid_x.size,2), dtype=grid_x.dtype )
                    grid_xy[:,0] = grid_x.reshape( grid_x.shape[0]*grid_x.shape[1] ) # flatten
                    grid_xy[:,1] = grid_y.reshape( grid_y.shape[0]*grid_y.shape[1] ) # flatten

                    grid_z = h2d_pos.copy().reshape( h2d_pos.shape[0]*h2d_pos.shape[1] ) # flatten

                    # target space
                    nn = 50

                    # only above some minimum size (always true for now)
                    if h2d_pos.shape[0] < nn or h2d_pos.shape[1] < nn:
                        # remove any NaN's (for 2d sg)
                        w = np.where(np.isnan(grid_z))
                        grid_z[w] = clims[i][0]

                        # only if 2D histogram is actually small(er) than this
                        XX_out = np.linspace(xlim[0], xlim[1], nn)
                        YY_out = np.linspace(ylim[0], ylim[1], nn)
                        grid_out_x, grid_out_y = np.meshgrid(XX_out, YY_out, indexing='ij')

                        grid_out = np.zeros( (grid_out_x.size,2), dtype=grid_out_x.dtype )
                        grid_out[:,0] = grid_out_x.reshape( nn*nn ) # flatten
                        grid_out[:,1] = grid_out_y.reshape( nn*nn ) # flatten

                        # resample and smooth
                        grid_z_out = griddata(grid_xy, grid_z, grid_out, method='cubic').reshape(nn,nn)

                        if yAxis == 'vrad':
                            # in this case, vrad crosses the zero boundary separating inflow/outflow, do not smooth across
                            min_pos_ind = np.where(YY_out > 0.0)[0].min()
                            grid_z_out[:,min_pos_ind:] = sgolay2d(grid_z_out[:,min_pos_ind:],sKn*3,sKo)
                        else:
                            grid_z_out = sgolay2d(grid_z_out,sKn*3,sKo)

                    # render contour (different linestyles for different contour values)
                    color = cmap_pos( float(i) / len(mStarBins) )
                    if j == 0: lines.append(plt.Line2D( (0,1), (0,0), color=color, marker='', lw=lw))
                    c_ls = linestyles if len(redshifts) == 1 else linestyles[j] # linestyle per redshift
                    im_pos = ax.contour(XX_out, YY_out, grid_z_out.T, contours, linestyles=c_ls, linewidths=lw, colors=[color])
                    #im_neg

                # special x-axis labels for angle theta
                if yAxis == 'theta':                
                    ax.set_yticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
                    ax.set_yticklabels(['$-\pi$','$-\pi/2$','$0$','$+\pi/2$','$+\pi$'])
                    ax.plot(xlim,[-np.pi/2,-np.pi/2],'-',color='#aaaaaa',alpha=0.3)
                    ax.plot(xlim,[+np.pi/2,+np.pi/2],'-',color='#aaaaaa',alpha=0.3)
                if xAxis == 'theta':
                    ax.set_xticks([-np.pi, -np.pi/2, 0.0, np.pi/2, np.pi])
                    ax.set_xticklabels(['$-\pi$','$-\pi/2$','$0$','$+\pi/2$','$+\pi$'])
                    ax.plot([-np.pi/2,-np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)
                    ax.plot([+np.pi/2,+np.pi/2],ylim,'-',color='#aaaaaa',alpha=0.3)

                # legend
                if contours is not None:
                    if j == 0:
                        labels1.append(label1)
                    continue # no colorbars, and no 'per panel' legends (add one at the end)

                line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
                if label2 is not None:
                    legend2 = ax.legend([line,line], [label1,label2], handlelength=0.0, loc='upper right')
                else:
                    legend2 = ax.legend([line], [label1], handlelength=0.0, loc='upper right')
                ax.add_artist(legend2)
                plt.setp(legend2.get_texts(), color='white')

                # colorbar(s)
                if len(mStarBins) > 1 or yAxis != 'vrad':
                    # some panels with each colorbar
                    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
                    if yAxis == 'vrad' and i % 2 == 1:
                        cb = plt.colorbar(im_neg, cax=cax)
                        cb.ax.set_ylabel(cbarlabel2)
                    else:
                        cb = plt.colorbar(im_pos, cax=cax)
                        cb.ax.set_ylabel(cbarlabel)
                else:
                    # both colorbars on the same, single panel
                    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.7)
                    cb = plt.colorbar(im_pos, cax=cax)
                    cb.ax.set_ylabel(cbarlabel.replace('Outflow','Inflow/Outflow'))

                    bbox_ax = cax.get_position()
                    cax2 = fig.add_axes( [0.814, bbox_ax.y0+0.004, 0.038, bbox_ax.height+0.0785]) # manual tweaks
                    cb2 = plt.colorbar(im_neg, cax=cax2)
                    cb2.ax.set_yticklabels('')

                # for massBins done
            # for redshifts done

        # single legend?
        if contours is not None:
            # labels for M* bins
            leg_lines = lines
            leg_labels = labels1
            leg_lines2 = []
            leg_labels2 = []

            if len(contours) > 1: # labels for contour levels
                for i, contour in enumerate(contours):
                    leg_lines2.append( plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[i]) )
                    leg_labels2.append('%s = %.1f' % (contourlabel,contour))

            if len(redshifts) > 1: # labels for redshifts
                for j, redshift in enumerate(redshifts):
                    leg_lines2.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[j],marker='') )
                    leg_labels2.append('z = %.1f' % redshift)

            legend3 = ax.legend(leg_lines, leg_labels, loc='lower left')
            ax.add_artist(legend3)

            legend4 = ax.legend(leg_lines2, leg_labels2, loc='upper right')
            ax.add_artist(legend4)

            # label for r,vcut?
            if label2 is not None:
                if len(contours) == 1: label2 += ', %s = %.1f' % (contourlabel,contours[0]) # not enumerated, so show the 1 choice
                line = plt.Line2D((0,1), (0,0), color='white', marker='', lw=lw)
                legend4 = ax.legend([line], [label2], handlelength=0.0, loc='upper left')
                ax.add_artist(legend4)

        # finish plot
        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    thirdQuant = None if (xAxis == 'rad' and yAxis == 'vcut') else xAxis # use to load x-axis quantity
    if thirdQuant == 'rad': thirdQuant = yAxis # if x-axis is rad, then use to load y-axis quantity (which is something other than vcut)

    fourthQuant = None if (xAxis == 'rad' or yAxis == 'vcut') else yAxis # use to load y-axis quantity (which is neither rad nor vcut)

    # load
    data = []

    for redshift in redshifts:

        if redshift is not None:
            sP.setRedshift(redshift)

        if xAxis == 'vrad' or yAxis == 'vrad':
            # non-standard dataset, i.e. not rad.vrad.*
            secondQuant = xAxis
            thirdQuant = yAxis
            fourthQuant = None

            mdot, mstar, subids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, ptType, secondQuant=secondQuant, thirdQuant=thirdQuant)
        else:
            # default behavior
            mdot, mstar, subids, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, ptType, thirdQuant=thirdQuant, fourthQuant=fourthQuant)

        binConfig['vcut'] = vcut_vals
        numBins['vcut'] = vcut_vals.size

        # update bounds based on the loaded dataset
        for quant in [xAxis,yAxis]:
            assert quant in binConfig
            if limits[quant] is not False: continue # hard-coded

            limits[quant] = [ binConfig[quant].min(), binConfig[quant].max() ] # always linear spacing

            if np.any(np.isinf(limits[quant])):
                limits[quant] = None # discrete labels (small number of bins)

        # load smoothed star formation rates, and crossmatch to subhalos with mdot
        sfr_smoothed = None

        if eta:
            sfr_timescale = 100.0 # Myr
            sfr_smoothed,_,_,_ = sP.simSubhaloQuantity('sfr_30pkpc_%dmyr' % sfr_timescale) # msun/yr

            gcIDs = np.arange(0, sP.numSubhalos)
            assert sP.numSubhalos == sfr_smoothed.size
            gc_inds, ac_inds = match3(gcIDs, subids)

            sfr_smoothed = sfr_smoothed[gc_inds]

        # append to data list
        data.append( (mdot, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed) )

    # single plot: if config passed in
    zStr = str(sP.snap) if len(redshifts) == 1 else 'z='+'-'.join(['%.1f'%z for z in redshifts])

    if config is not None:
        saveName = '%s_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,config['stat'],config['skipZeros'],cStr)
        if 'saveName' in config: saveName = config['saveName']
        if 'vcutInd' not in config: config['vcutInd'] = None
        if 'radInd' not in config: config['radInd'] = None

        _plotHelper(xAxis,yAxis,mStarBins,config['stat'],skipZeroFluxes=config['skipZeros'],
                    vcut_ind=config['vcutInd'],rad_ind=config['radInd'],saveName=saveName)
        return

    # plots: explore all
    for stat in ['mean']:#['mean','median']:
        for skipZeros in [False]:#[True,False]:

            print(xAxis,yAxis,stat,'zeros:',skipZeros,'eta:',eta)
            # (A) 2D histogram, where axes consider all (rad,vcut) or (rad,vrad) values
            if xAxis == 'rad' and yAxis in ['vcut','vrad']:
                saveName = '%s_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr)
                _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,saveName=saveName)

                continue

            # (B) 2D histogram, where xAxis is still rad, so make separate plots for each (vcut) value
            if xAxis == 'rad':
                pdf = PdfPages('%s_B_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr))

                for vcutInd in range(numBins['vcut']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,vcut_ind=vcutInd,pdf=pdf)

                pdf.close()
                continue

            # (C) 2D histogram, where yAxis is (vcut) values, so make separate plots for each (rad) value
            if yAxis in ['vcut','vrad']:
                pdf = PdfPages('%s_C_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr))

                for radInd in range(numBins['rad']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,rad_ind=radInd,pdf=pdf)

                pdf.close()
                continue

            # (D) 2D histogram, where neither xAxis nor yAxis cover any (rad,vcut) values, so need to iterate over both
            pdf = PdfPages('%s_D_%s_%s-%s_mstar_%s_%s_%s_skipzeros-%s%s.pdf' % (saveBase,ptType,xAxis,yAxis,sP.simName,zStr,stat,skipZeros,cStr))

            for vcutInd in range(numBins['vcut']):
                for radInd in range(numBins['rad']):
                    _plotHelper(xAxis,yAxis,mStarBins,stat,skipZeroFluxes=skipZeros,vcut_ind=vcutInd,rad_ind=radInd,pdf=pdf)

            pdf.close()

def _haloSizeScalesHelper(ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c):
    """ Helper to draw lines at given fixed or adaptive sizes, i.e. rvir fractions, in radial profile plots. """
    textOpts = {'va':'bottom', 'ha':'right', 'fontsize':16.0, 'alpha':0.1, 'rotation':90}
    lim = ax.get_ylim()
    y1 = np.array([ lim[1], lim[1] - (lim[1]-lim[0])*0.1]) - (lim[1]-lim[0])/40
    y2 = np.array( [lim[0], lim[0] + (lim[1]-lim[0])*0.1]) + (lim[1]-lim[0])/40
    xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) / 150

    if xaxis in ['log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
        y1[1] -= (lim[1]-lim[0]) * 0.02 * (len(massBins)-k) # lengthen

        if 're' in xaxis: divisor = avg_re_code
        if 'rvir' in xaxis: divisor = avg_rvir_code
        if 'rhalf' in xaxis: divisor = avg_rhalf_code

        # 50 kpc at the top
        num_kpc = 20 if 'rvir' in xaxis else 10
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)

        ax.plot(xrvir, y1, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == len(massBins)-1 and i == 0: ax.text(xrvir[0]-xoff, y1[1], '%d kpc' % num_kpc, color=c, **textOpts)

        # 10 kpc at the bottom
        num_kpc = 5
        rvir_Npkpc_ratio = sP.units.physicalKpcToCodeLength(num_kpc) / divisor
        xrvir = [rvir_Npkpc_ratio, rvir_Npkpc_ratio]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)

        ax.plot(xrvir, y2, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
        if k == 0 and i == 0: ax.text(xrvir[0]-xoff, y2[0], '%d kpc' % num_kpc, color=c, **textOpts)

    elif xaxis in ['log_pkpc','pkpc']:
        y1[1] -= (lim[1]-lim[0]) * 0.02 * k # lengthen

        # Rvir at the top
        rVirFac = 10 if 'log' in xaxis else 5
        xrvir = [avg_rvir_code/rVirFac, avg_rvir_code/rVirFac]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)
        textStr = 'R$_{\\rm vir}$/%d' % rVirFac

        if 1: #i == 0 or i == len(sPs)-1: # only at first/last redshift, since largely overlapping
            ax.plot(xrvir, y1, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == 0 and i == 0: ax.text(xrvir[0]-xoff, y1[1], textStr, color=c, **textOpts)

        # Rhalf at the bottom
        rHalfFac = 2 if 'log' in xaxis else 10
        targetK = len(massBins)-1 # largest
        if field == 'SFR' and 'log' in xaxis: # special case
            rHalfFac = 1
            targetK = 0

        xrvir = [rHalfFac*avg_rhalf_code, rHalfFac*avg_rhalf_code]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)
        textStr = '%dr$_{1/2,\star}$' % rHalfFac if rHalfFac != 1 else 'r$_{1/2,\star}$'

        if 1: #i == 0 or i == len(sPs)-1:
            ax.plot(xrvir, y2, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == targetK and i == 0: ax.text(xrvir[0]-xoff, y2[0], textStr, color=c, **textOpts)

def stackedRadialProfiles(sPs, field, cenSatSelect='cen', projDim='3D', xaxis='log_pkpc', reBand='jwst_f115w',
                          haloMassBins=None, mStarBins=None, ylabel='', ylim=None, colorOff=0, saveName=None, pdf=None):
    """ Plot average/stacked radial profiles for a series of stellar mass bins and/or runs (sPs) i.e. at different 
    redshifts. """
    from projects.oxygen import _resolutionLineHelper
    assert xaxis in ['log_pkpc','log_rvir','log_rhalf','log_re','pkpc','rvir','rhalf','re']

    percs = [16,84]

    # plot setup
    fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
    ax = fig.add_subplot(111)
    
    radStr = 'Radius' if '3D' in projDim else 'Projected Distance'

    if xaxis == 'log_rvir':
        ax.set_xlim([-2.5, 0.0])
        ax.set_xlabel('%s / Virial Radius [ log ]' % radStr)
    elif xaxis == 'rvir':
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel('%s / Virial Radius' % radStr)
    elif xaxis == 'log_rhalf':
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel('%s / Stellar Half-mass Radius [ log ]' % radStr)
    elif xaxis == 'rhalf':
        ax.set_xlim([0, 10])
        ax.set_xlabel('%s / Stellar Half-mass Radius' % radStr)
    elif xaxis == 'log_pkpc':
        ax.set_xlim([-0.5, 2.0])
        ax.set_xlabel('%s [ log pkpc ]' % radStr)
    elif xaxis == 'pkpc':
        ax.set_xlim([0, 100])
        ax.set_xlabel('%s [ pkpc ]' % radStr)
    elif xaxis == 'log_re':
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel('%s / Stellar R$_{\\rm e}$ (JWST f115w) [ log ]' % radStr)
    elif xaxis == 're':
        ax.set_xlim([0, 10])
        ax.set_xlabel('%s / Stellar R$_{\\rm e}$ (JWST f115w)' % radStr)

    ylabels_3d = {'SFR'                   : '$\dot{\\rho}_\star$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-3}$ ]',
                  'Gas_Mass'              : '$\\rho_{\\rm gas}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Stars_Mass'            : '$\\rho_{\\rm stars}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\\rho_{\\rm gas}$ / $\\rho_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\\rho_{\\rm metals}$ [ log M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Metallicity'       : 'Gas Metallicity (unweighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Metallicity_sfrWt' : 'Gas Metallicity (SFR weighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Bmag'              : 'Gas Magnetic Field Strength [ log Gauss ]'}
    ylims_3d   = {'SFR'                   : [-10.0, 0.0],
                  'Gas_Mass'              : [0.0, 9.0],
                  'Stars_Mass'            : [0.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [-4.0,  8.0],
                  'Gas_Metallicity'       : [-2.0, 1.0],
                  'Gas_Metallicity_sfrWt' : [-1.5, 1.0],
                  'Gas_Bmag'              : [-9.0, -4.0]}

    ylabels_2d = {'SFR'                   : '$\dot{\Sigma}_\star$ [ log M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ]',
                  'Gas_Mass'              : '$\Sigma_{\\rm gas}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Stars_Mass'            : '$\Sigma_{\\rm stars}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\Sigma_{\\rm gas}$ / $\Sigma_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\Sigma_{\\rm metals}$ [ log M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Metallicity'       : ylabels_3d['Gas_Metallicity'],
                  'Gas_Metallicity_sfrWt' : ylabels_3d['Gas_Metallicity_sfrWt'],
                  'Gas_Bmag'              : ylabels_3d['Gas_Bmag'],
                  'Gas_LOSVel_sfrWt'      : 'Gas Velocity v$_{\\rm LOS}$ (SFR weighted) [ km/s ]',
                  'Gas_LOSVelSigma'       : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D}$ [ km/s ]',
                  'Gas_LOSVelSigma_sfrWt' : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D,SFRw}$ [ km/s ]'}
    ylims_2d   = {'SFR'                   : [-6.0, 0.0],
                  'Gas_Mass'              : [3.0, 9.0],
                  'Stars_Mass'            : [3.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [0.0, 8.0],
                  'Gas_Metallicity'       : ylims_3d['Gas_Metallicity'],
                  'Gas_Metallicity_sfrWt' : ylims_3d['Gas_Metallicity_sfrWt'],
                  'Gas_Bmag'              : ylims_3d['Gas_Bmag'],
                  'Gas_LOSVel_sfrWt'      : [0, 350],
                  'Gas_LOSVelSigma'       : [0, 350],
                  'Gas_LOSVelSigma_sfrWt' : [0, 350]}

    # only these fields are treated as total sums, and normalized/unit converted appropriately, otherwise we 
    # assume the auxCat() profiles are already e.g. mean or medians in the desired units
    totSumFields = ['SFR','Gas_Mass','Gas_Metal_Mass','Stars_Mass']

    if len(sPs) > 1:
        # multi-redshift, adjust bounds
        ylims_3d['SFR'] = [-10.0, 2.0]
        ylims_2d['SFR'] = [-7.0, 2.0]
        ylims_3d['Gas_Mass'] = [1.0, 10.0]
        ylims_2d['Gas_Mass'] = [4.0, 10.0]
        ylims_3d['Gas_Metallicity'] = [-2.5, 0.5]
        ylims_2d['Gas_Metallicity'] = [-2.5, 0.5]
        ylims_3d['Gas_Metallicity_sfrWt'] = [-2.0, 1.0]
        ylims_2d['Gas_Metallicity_sfrWt'] = [-2.0, 1.0]

    fieldName = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, field)

    if field == 'Gas_Fraction':
        # handle stellar mass auxCat load and normalization below
        fieldName = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, 'Gas_Mass')
        fieldName2 = 'Subhalo_RadProfile%s_FoF_%s' % (projDim, 'Stars_Mass')

    if '3D' in projDim:
        ax.set_ylabel(ylabels_3d[field])
        ax.set_ylim(ylims_3d[field])
    else:
        ax.set_ylabel(ylabels_2d[field])
        ax.set_ylim(ylims_2d[field])

    # init
    colors = []
    rvirs  = []
    rhalfs = []
    res    = []

    if haloMassBins is not None:
        massField = 'mhalo_200_log'
        massBins = haloMassBins
    else:
        massField = 'mstar_30pkpc_log'
        massBins = mStarBins

    labelNames = True if nUnique([sP.simName for sP in sPs]) > 1 else False
    labelRedshifts = True if nUnique([sP.redshift for sP in sPs]) > 1 else False

    # loop over each fullbox run
    txt = []

    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii, tsellar half mass radii, and (optionally) effective optical radii
        rad_rvir  = sP.groupCat(fieldsSubhalos=['rhalo_200_code']) 
        rad_rhalf = sP.groupCat(fieldsSubhalos=['rhalf_stars_code'])
        rad_rvir  = rad_rvir[cssInds]
        rad_rhalf = rad_rhalf[cssInds]

        rad_re = np.zeros( rad_rvir.size, dtype='float32' ) # unused by default
        if xaxis in ['log_re','re']:
            fieldNameRe = 'Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc'
            ac_re = sP.auxCat(fieldNameRe)
            bandInd = list(ac_re[fieldNameRe + '_attrs']['bands']).index(reBand)
            rad_re = ac_re[fieldNameRe][:,bandInd] # code units

        print('[%s]: %s (z=%.1f)' % (field,sP.simName,sP.redshift))

        # load and apply CSS
        ac = sP.auxCat(fields=[fieldName])

        assert ac[fieldName].ndim == 2 # self-halo term only

        # special cases requiring multiple auxCat datasets
        if field == 'Gas_Fraction':
            # gas fraction = (M_gas)/(M_gas+M_stars)
            ac2 = sP.auxCat(fields=[fieldName2])
            assert ac2[fieldName2].ndim == 2
            assert np.array_equal( ac['subhaloIDs'], ac2['subhaloIDs'] )
            assert np.array_equal( ac[fieldName+'_attrs']['rad_bins_code'], ac2[fieldName2+'_attrs']['rad_bins_code'] )

            ac[fieldName] = ac[fieldName] / (ac[fieldName] + ac2[fieldName2])

        # crossmatch 'subhaloIDs' to cssInds
        ac_inds, css_inds = match3( ac['subhaloIDs'], cssInds )
        ac[fieldName] = ac[fieldName][ac_inds,:]

        masses    = masses[css_inds]
        rad_rvir  = rad_rvir[css_inds]
        rad_rhalf = rad_rhalf[css_inds]
        rad_re    = rad_re[css_inds]

        yy = ac[fieldName]

        # loop over mass bins
        for k, massBin in enumerate(massBins):
            txt_mb = {}

            # select
            with np.errstate(invalid='ignore'):
                w = np.where( (masses >= massBin[0]) & (masses < massBin[1]) )

            print(' %s %s [%d] %4.1f - %4.1f : %d' % (field,projDim,k,massBin[0],massBin[1],len(w[0])))
            if len(w[0]) == 0:
                continue

            # radial bins: normalize to rvir, rhalf, or re if requested
            avg_rvir_code  = np.nanmedian( rad_rvir[w] )
            avg_rhalf_code = np.nanmedian( rad_rhalf[w] )
            avg_re_code    = np.nanmedian( rad_re[w] )

            if (i == 0 and len(massBins)>1) or (k == 0 and len(sPs)>1):
                rvirs.append( avg_rvir_code )
                rhalfs.append( avg_rhalf_code )
                res.append( avg_re_code )

            # sum and calculate percentiles in each radial bin
            yy_local = np.squeeze( yy[w,:] )

            if xaxis in ['log_rvir','rvir']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rvir_code
            elif xaxis in ['log_rhalf','rhalf']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rhalf_code
            elif xaxis in ['log_re','re']:
                rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_re_code
            elif xaxis in ['log_pkpc','pkpc']:
                rr = ac[fieldName+'_attrs']['rad_bins_pkpc']

            # unit conversions: sum per bin to (sum 3D spatial density) or (sum 2D surface density)
            if '3D' in projDim:
                normField = 'bin_volumes_code'
                unitConversionFunc = partial(sP.units.codeDensToPhys, totKpc3=True)
            else:
                normField = 'bin_areas_code' # 2D
                unitConversionFunc = partial(sP.units.codeColDensToPhys, totKpc2=True)

            if field in totSumFields:
                yy_local /= ac[fieldName+'_attrs'][normField] # sum -> (sum/volume) or (sum/area), in code units

            if '_Mass' in field:
                # convert the numerator, e.g. code masses -> msun (so msun/kpc^3)
                yy_local = sP.units.codeMassToMsun(yy_local)

            # resample, integral preserving, to combine poor statistics bins at large distances
            if 1:
                # construct new versions of yy, rr, and normalizations
                shape = np.array(yy_local.shape)
                start_ind = int(shape[1] * 0.4)
                yy_local_new = np.zeros( shape, dtype=yy.dtype )
                rr_new = np.zeros( shape[1], dtype=rr.dtype )
                norm_new = np.zeros( shape[1], dtype=rr.dtype )

                cur_ind = 0
                read_ind = 0
                accum_size = 1

                if field in totSumFields:
                    yy_local *= ac[fieldName+'_attrs'][normField]

                while read_ind < shape[1]:
                    #print('[%d] avg [%d - %d]' % (cur_ind,read_ind,read_ind+accum_size))
                    # copy or average
                    if field in totSumFields:
                        yy_local_new[:,cur_ind] = np.nansum( yy_local[:,read_ind:read_ind+accum_size], axis=1 )
                    else:
                        yy_local_new[:,cur_ind] = np.nanmedian( yy_local[:,read_ind:read_ind+accum_size], axis=1 )

                    rr_new[cur_ind] = np.nanmean( rr[read_ind:read_ind+accum_size])
                    norm_new[cur_ind] = np.nansum( ac[fieldName+'_attrs'][normField][read_ind:read_ind+accum_size] )

                    # update window
                    cur_ind += 1
                    read_ind += accum_size

                    # enlarge averaging region only at large distances
                    if cur_ind >= start_ind:
                        if cur_ind % 10 == 0: accum_size += 1

                # re-do normalization and reduce to new size
                yy_local = yy_local_new[:,0:cur_ind]
                if field in totSumFields:
                    yy_local /= norm_new[0:cur_ind]
                rr = rr_new[0:cur_ind]
                #print('  Note: Resampled yy,rr from [%d] to [%d] total radial bins!' % (shape[1],rr.size))

            if field in totSumFields:
                yy_local = unitConversionFunc(yy_local) # convert area or volume in code units to pkpc^2 or pkpc^3

            # replace zeros by nan so they are not included in percentiles
            # note: we don't want the median to be dragged to zero due to bins with zero particles in individual subhalos
            # rather, want to accumulate across subhalos and then normalize (i.e. yy_mean), so if we set zero bins to nan 
            # here the resulting yy_med (and yp) are similar
            yy_local[yy_local == 0.0] = np.nan

            # calculate totsum profile and scatter
            yy_mean = np.nansum( yy_local, axis=0 ) / len(w[0])
            yy_med  = np.nanmedian( yy_local, axis=0 )
            yp = np.nanpercentile( yy_local, percs, axis=0 )

            # log both axes and smooth
            if '_LOSVel' not in field and '_Fraction' not in field:
                yy_mean = logZeroNaN(yy_mean)
                yy_med = logZeroNaN(yy_med)
                yp = logZeroNaN(yp)

            if 'log_' in xaxis:
                rr = np.log10(rr)

            if rr.size > sKn:
                yy_mean = savgol_filter(yy_mean,sKn+4,sKo)
                yy_med = savgol_filter(yy_med,sKn+4,sKo)
                yp = savgol_filter(yp,sKn+4,sKo,axis=1)

            #if 'Metallicity' in field:
            #    # test: remove noisy last point which is non-monotonic
            #    w = np.where(np.isfinite(yy_med))
            #    if yy_med[w][-1] > yy_med[w][-2]:
            #        yy_med[w[0][-1]] = np.nan

            # extend line to right-edge of x-axis?
            w = np.where( np.isfinite(yy_med) )
            xmax = ax.get_xlim()[1]

            if rr[w][-1] < xmax:
                new_ind = w[0].max() + 1
                rr[new_ind] = xmax

                yy_mean[new_ind] = interp1d(rr[w][-3:], yy_mean[w][-3:], kind='linear', fill_value='extrapolate')(xmax)
                yy_med[new_ind] = interp1d(rr[w][-3:], yy_med[w][-3:], kind='linear', fill_value='extrapolate')(xmax)
                for j in range(yp.shape[0]):
                    yp[j,new_ind] = interp1d(rr[w][-3:], yp[j,w][:,-3:], kind='linear', fill_value='extrapolate')(xmax)

            # determine color
            if i == 0:
                for _ in range(colorOff+1):
                    c = ax._get_lines.prop_cycler.next()['color']
                colors.append(c)
            else:
                c = colors[k]

            # plot totsum and/or median line
            if haloMassBins is not None:
                label = '$M_{\\rm halo}$ = %.1f' % (0.5*(massBin[0]+massBin[1])) if (i == 0) else ''
            else:
                label = 'M$^\star$ = %.1f' % (0.5*(massBin[0]+massBin[1])) if (i == 0) else ''

            ax.plot(rr, yy_med, lw=lw, color=c, linestyle=linestyles[i], label=label)
            #ax.plot(rr, yy_mean, lw=lw, color=c, linestyle=':', alpha=0.5)

            txt_mb['bin'] = massBin
            txt_mb['rr'] = rr
            txt_mb['yy'] = yy_med
            txt_mb['yy_0'] = yp[0,:]
            txt_mb['yy_1'] = yp[-1,:]

            # draw rvir lines (or 100pkpc lines if x-axis is already relative to rvir)
            _haloSizeScalesHelper(ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c)

            # show percentile scatter only for first run
            if i == 0:
                # show percentile scatter only for first/last massbin
                if (k == 0 or k == len(massBins)-1) or (field == 'Gas_LOSVelSigma' in field and k == int(len(massBins)/2)):
                    ax.fill_between(rr, yp[0,:], yp[-1,:], color=c, interpolate=False, alpha=0.2)

            txt.append(txt_mb)

    # gray resolution band at small radius
    if xaxis in ['log_rvir','log_pkpc']:
        _resolutionLineHelper(ax, sPs, xaxis=='log_rvir', rvirs=rvirs)

    # print
    #for k in range(len(txt)): # loop over mass bins (separate file for each)
    #    filename = 'figX_%s_%sdens_rad%s_m-%.2f.txt' % \
    #      (field,projDim, 'rvir' if radRelToVirRad else 'kpc', np.mean(txt[k]['bin']))
    #    out = '# Nelson+ (in prep) http://arxiv.org/...\n'
    #    out += '# Figure X n_OVI [log cm^-3] (%s z=%.1f)\n' % (sP.simName, sP.redshift)
    #    out += '# Halo Mass Bin [%.1f - %.1f]\n' % (txt[k]['bin'][0], txt[k]['bin'][1])
    #    out += '# rad_logpkpc val val_err0 val_err1\n'
    #    for i in range(1,txt[k]['rr'].size): # loop over radial bins
    #        out += '%8.4f  %8.4f %8.4f %8.4f\n' % (txt[k]['rr'][i], txt[k]['yy'][i], txt[k]['yy_0'][i], txt[k]['yy_1'][i])
    #    with open(filename, 'w') as f:
    #        f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            label = ''
            if labelNames: label = sP.simName
            if labelRedshifts: label += ' z=%.1f' % sP.redshift
            lExtra += [label.strip()]

    handles, labels = ax.get_legend_handles_labels()
    legendLoc = 'upper right'
    if '_Fraction' in field: legendLoc = 'lower right' # typically rising not falling with radius
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc=legendLoc)

    fig.tight_layout()
    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(saveName)
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def paperPlots(sPs=None):
    """ Construct all the final plots for the paper. """
    #redshift = 0.73 # snapshot 58, where intermediate trees were constructed
    redshift = 2.0

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)
    TNG50_4 = simParams(res=270,run='tng',redshift=redshift)

    TNG50_z1 = simParams(res=2160,run='tng',redshift=1.0)

    mStarBins    = [ [7.9,8.1],[8.9,9.1],[9.4,9.6],[9.9,10.1],[10.3,10.7],[10.8,11.2],[11.2,11.8] ]
    mStarBinsSm  = [ [8.7,9.3],[9.9,10.1],[10.3,10.7],[10.8,11.2] ] # less
    mStarBinsSm2 = [ [7.9,8.1],[9.4,9.6],[10.3,10.7],[10.8,11.2] ]
    redshifts    = [1.0, 2.0, 4.0]

    radProfileFields = ['SFR','Gas_Mass','Gas_Metal_Mass', 'Stars_Mass', 'Gas_Fraction',
                        'Gas_Metallicity','Gas_Metallicity_sfrWt',
                        'Gas_LOSVelSigma','Gas_LOSVelSigma_sfrWt']

    quants1 = ['ssfr','Z_gas','fgas2','size_gas','temp_halo_volwt','mass_z']
    quants2 = ['surfdens1_stars','Z_stars','color_B_gr','size_stars','vout_75_all','etaM_100myr_10kpc_0kms']
    quants3 = ['nh_halo_volwt','fgas_r200','pratio_halo_volwt','Krot_oriented_stars2','Krot_oriented_gas2','_dummy_']
    quants4 = ['BH_BolLum','BH_BolLum_basic','BH_EddRatio','BH_dEdt','BH_CumEgy_low','M_BH_actual']
    quantSets = [quants1, quants2, quants3, quants4]

    # --------------------------------------------------------------------------------------------------------------------------------------

    # TODO (future, moved outside the scope of this paper):
    #  * add vlos,r2d as quantities calculated in instantaneousMassFluxes(), can then simply bin (for a reasonable 2d aperture)
    #    vlos instead of vrad, getting quite close to slit/fiber/down the barrel spectra
    #  *  add r/rvir, v/vir as quantities in instantaneousMassFluxes(), so we can also bin in these
    #  * play with 'total mass absorption spectra' (or even e.g. MgII, CIV), down the barrel, R_e aperture, need Voigt profile or maybe not

    # TODO (now, for the paper):
    #  [.] obs data points for vout and eta, etc
    #  [ ] z>=6 teaser plot: ?
    #  [.] galaxy property correlations (BHs)
    #  [ ] check: T_eEOS->constant value in mass flux binning, propagate through
    #  [.] eta_E (energy), eta_p (momentum), eta_Z (metal)
    #  [ ] plot with time on the x-axis (SFR, BH_Mdot, v_out different rad, eta_out different rad) (all from a subbox?)
    #  [ ] vis: SN vs BH driven outflows schematic (streamlines?)

    if 0:
        # fig 1: resolution/volume metadata of TNG50 vs other sims
        pass

    if 0:
        # fig 2: large-scale visualization of the gas density field
        subboxSingleVelocityFrame()

    if 0:
        # fig 3: large mosaic of many galaxies (stellar light + gas density)
        galaxyMosaic_topN(numHalosInd=2, panelNum=1)
        galaxyMosaic_topN(numHalosInd=2, panelNum=1)

    if 0:
        # fig 4: mass loading as a function of M* at one redshift, three v_rad values with individual markers
        config = {'vcutInds':[0,2,3], 'radInds':[1], 'stat':'mean', 'ylim':[-0.55,2.05], 'skipZeros':False, 'markersize':4.0, 'addModelTNG':True}
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config)

        # mass loading as a function of M* at one redshift, few variations in both (radius,vcut)
        config = {'vcutInds':[1,2,4], 'radInds':[1,2,5], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}

        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config)
        #gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='Gas', eta=True, config=config, massField='MgII') # testing MgII

        # mass loading 2D contours in (radius,vcut) plane: dependence on eta thresholds, at fixed redshift
        #contours = [-0.5, 0.0, 0.5]
        #gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=contours, eta=True)

        # mass loading 2D contours in (radius,vcut) plane: redshift evolution
        contours = [-1.5]
        gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=contours, redshifts=redshifts, eta=True)

    if 0:
        # fig 5: net outflow rates (2D): dependence on radius and vcut, for one/four stellar mass bins
        mStarBins = [ [9.9,10.1] ] #[ [8.7,9.3],[9.9,10.1],[10.3,10.7],[10.8,11.2] ]
        clims     = [ [-3.0,2.0] ]
        config    = {'stat':'mean', 'skipZeros':False}

        gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBins, clims=clims, config=config)

    if 0:
        # fig 6: outflow velocity as a function of M* at one redshift, two v_perc values with individual markers
        config = {'percInds':[2,4], 'radInds':[1], 'ylim':[0,800], 'stat':'mean', 'skipZeros':False, 'markersize':4.0, 'addModelTNG':True}
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config)
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config, massField='MgII') # testing MgII

        # outflow velocity as a function of M* at one redshift, variations in (radius,v_perc) values
        config = {'percInds':[1,2,4], 'radInds':[1,2,13], 'ylim':[0,800], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', config=config)

        # outflow velocity: redshift evo
        config = {'percInds':[3], 'radInds':[1], 'ylim':[0,800], 'stat':'mean', 'skipZeros':False, 'addModelTNG':True}
        redshifts_loc = [1.0, 2.0, 3.0, 4.0, 6.0]
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts_loc, config=config)

    if 0:
        # fig 7: distribution of radial velocities
        config = {'radInd':2, 'stat':'mean', 'ylim':[-3.0, 1.0], 'skipZeros':False}

        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='vrad', mStarBins=mStarBins, config=config)
        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='vrad', mStarBins=mStarBinsSm2, config=config, redshifts=redshifts)

    if 0:
        # fig 8: vrad plots, single halo
        subInd = 389836 # first one in subbox0 intersecting with >11.5 selection
        haloInd = TNG50.groupCatSingle(subhaloID=subInd)['SubhaloGrNr']
        explore_vrad_halos(TNG50, haloIndsPlot=[haloInd])

    if 0:
        # fig 9: outflow rates vs a single quantity (marginalized over all others), one redshift, stacked in M* bins
        config = {'radInd':2, 'vcutInd':2, 'stat':'mean', 'ylim':[-3.0, 2.0], 'skipZeros':False}

        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='temp', mStarBins=mStarBins, config=config)
        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='numdens', mStarBins=mStarBins, config=config)
        #gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='z_solar', mStarBins=mStarBins, config=config)
        config['ylim'] = [-3.0, 1.0]
        gasOutflowRatesVsQuantStackedInMstar(TNG50, quant='z_solar', mStarBins=mStarBinsSm2[0:3], redshifts=redshifts, config=config)

    if 0:
        # fig 7,10: radial velocities of outflows (2D): dependence on radius, and dependence on temperature, for one m* bin
        mStarBins = [ [10.7,11.3] ]

        config    = {'stat':'mean', 'skipZeros':False, 'radInd':[3]}
        gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='vrad', mStarBins=mStarBins, clims=[[-3.5,0.0]], config=config)

        config    = {'stat':'mean', 'skipZeros':False}
        gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vrad', mStarBins=mStarBins, clims=[[-3.0,0.5]], config=config)

    if 0:
        # fig 11: angular dependence: 2D histogram of outflow rate vs theta, for 2 demonstrative stellar mass bins
        mStarBins = [[9.8,10.2],[10.8,11.2]]
        clims     = [[-2.5,-1.0],[-1.5,0.0]]
        config    = {'stat':'mean', 'skipZeros':False, 'vcutInd':[3,5]}

        gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='theta', mStarBins=mStarBins, clims=clims, config=config)

    if 0:
        # fig 12: (relative) outflow velocity vs. Delta_SFMS, as a function of M*
        sP = TNG50_z1

        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuant = 'mstar_30pkpc_log'
        cQuant = 'vout_50_all'
        yQuant = 'delta_sfms'
        cRel   = [0.7,1.3,False] # [cMin,cMax,cLog] #None
        ylim   = [-0.75, 1.25]
        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant, 'ylim':ylim, 'cRel':cRel}

        pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, pdf, yQuant=yQuant, fig_subplot=[fig,111], **params)
        pdf.close()

        # inset: trend of relative vout with delta_MS for two M* slices
        xQuant  = 'delta_sfms'
        sQuant  = 'mstar2_log'
        sRange  = [[9.4,9.6],[10.4,10.8]]
        xlim    = [-1.0, 0.5]
        yRel    = [0.65,1.25,False,'$v_{\\rm out}$ / $v_{\\rm out,median}$'] # [cMin,cMax,cLog] #None
        sizefac = 0.4
        css     = 'cen'
        yQuant  = 'vout_50_all'

        pdf = PdfPages('slice_%s_%d_x=%s_y=%s_s=%s_%s.pdf' % (sP.simName,sP.snap,xQuant,yQuant,sQuant,css))
        quantSlice1D([sP], pdf, xQuant=xQuant, yQuants=[yQuant], sQuant=sQuant, 
                     sRange=sRange, xlim=xlim, yRel=yRel, sizefac=sizefac, cenSatSelect=css)
        pdf.close()

    if 0:
        # fig 13: fraction of 'fast outflow' galaxies, in the Delta_SFMS vs M* plane
        sP = TNG50_z1

        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuant = 'mstar_30pkpc_log'
        nBins  = 50
        yQuant = 'delta_sfms'

        cQuant = 'vout_75_all'
        cFrac  = [200, np.inf, False, 'Fraction w/ Fast Outflows ($v_{\\rm out}$ > 200 km/s)']

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'cFrac':cFrac, 'nBins':nBins}

        pdf = PdfPages('histofrac2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, pdf, xQuant=xQuant, yQuant=yQuant, fig_subplot=[fig,111], **params)
        pdf.close()

    if 0:
        # fig 14: stacked radial profiles of SFR surface density
        sP = TNG50_z1
        cenSatSelect = 'cen'
        field   = 'SFR'
        projDim = '2Dz'
        xaxis   = 'log_pkpc'

        pdf = PdfPages('radprofiles_%s-%s-%s_%s_%d_%s.pdf' % (field,projDim,xaxis,sP.simName,sP.snap,cenSatSelect))
        stackedRadialProfiles([sP], field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                              projDim=projDim, mStarBins=mStarBins, pdf=pdf)
        pdf.close()

    if 0:
        # fig 15: in progress (observational comparisons, many panels)

        # vout vs. etaM
        config = {'percInds':[1,2,4,5], 'radInds':[1], 'ylim':[1.5,3.5], 'stat':'mean', 'skipZeros':False, 
                  'binSize':0.25, 'loc2':'upper left', 'markersize':0.0, 'percs':[5,95], 'minMstar':7.5,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]', 'xlabel':'Mass Loading $\eta_{\\rm M}$ [ log ]'}
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_10kpc_0kms', ylog=True, config=config)

        # vout vs SFR
        config = {'percInds':[0,1,2,4,5], 'radInds':[1], 'xlim': [-1.0, 2.6], 'ylim':[1.45,3.55], 'stat':'mean', 'skipZeros':False, 
                  'binSize':0.25, 'loc1':'upper right', 'loc2':'lower right', 'markersize':0.0, 'percs':[5,95], 'minMstar':7.5,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]', 'xlabel':'Star Formation Rate [ log M$_{\\rm sun}$ yr$^{-1}$ ]'}
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='sfr_30pkpc_100myr', ylog=True, config=config)

    # --------------------------------------------------------------------------------------------------------------------------------------

    if 1:
        # TESTING (fig 15)
        config = {'percInds':[0,1,2,4,5], 'radInds':[1], 'ylim':[1.45,3.55], 'stat':'mean', 'skipZeros':False, 
                  'binSize':0.25, 'markersize':0.0, 'percs':[5,95], 'minMstar':7.5,
                  'ylabel':'Outflow Velocity $v_{\\rm out}$ [ log km/s ]', }
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='BH_EddRatio', ylog=True, config=config)

    if 1:
        # TESTING: (fig 15)
        config = {'vcutInds':[0,1,2], 'radInds':[1], 'stat':'mean', 'ylim':[0.0, 2.0], 'skipZeros':False, 
                  'binSize':0.25, 'markersize':0.0, 'percs':[5,95], 'minMstar':7.5}
        gasOutflowRatesVsQuant(TNG50_z1, ptType='total', xQuant='BH_BolLum', eta=True, config=config)

    if 0:
        # TEST: outlier check in above
        sP = TNG50_z1
        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuant = 'etaM_100myr_10kpc_0kms'
        nBins = 50
        yQuant = 'vout_95_10kpc_log'
        cRel   = None #[0.7,1.3,False] # [cMin,cMax,cLog] #None

        cQuant = 'mstar_30pkpc_log'
        cFrac  = None #[20.0, np.inf, False, None]

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'cRel':cRel, 'cFrac':cFrac, 'nBins':nBins}

        pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
        fig = plt.figure(figsize=figsize_loc)
        quantHisto2D(sP, pdf, xQuant=xQuant, yQuant=yQuant, fig_subplot=[fig,111], **params)
        pdf.close()

    # --------------------------------------------------------------------------------------------------------------------------------------

    if 0:
        # explore: sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG50.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG50)

    if 0:
        # explore: outflow velocity as a function of M* at one redshift
        gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', ylog=True)
        #gasOutflowVelocityVsQuant(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts)

        # explore: outflow velocity as a function of etaM at one redshift
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_10kpc_0kms', ylog=True)
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_20kpc_0kms', ylog=True)
        gasOutflowVelocityVsQuant(TNG50_z1, xQuant='etaM_100myr_all_0kms', ylog=True)

    if 0:
        # explore: net outflow rates (and mass loading factors), fully marginalized, as a function of stellar mass
        for ptType in ['Gas','Wind','total']:
            gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType=ptType, eta=False)
        gasOutflowRatesVsQuant(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True)

    if 0:
        # explore: net outflow rate distributions, vs a single quantity (marginalized over all others), stacked in M* bins
        for quant in ['temp','numdens','z_solar','theta','vrad']:
            gasOutflowRatesVsQuantStackedInMstar(TNG50, quant=quant, mStarBins=mStarBins)

            # explore: redshift dependence        
            gasOutflowRatesVsQuantStackedInMstar(TNG50, quant=quant, mStarBins=mStarBinsSm2, redshifts=redshifts)

    if 0:
        # explore: net 2D outflow rates/mass loadings, for several M* bins in three different configurations:
        #  (i) multi-panel, one per M* bin, at one redshift
        #  (ii) single-panel, many contours for each M* bin, at one redshift
        #  (ii) single-panel, one contour for each M* bin, at multiple redshifts
        # 2D here render binConfigs: 2,3,4, 5,6, 7,8,9,10,11 (haven't actually used #1...)
        for eta in [True,False]:
            # special case:
            z_contours = [-0.5-1*eta]

            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, clims=[[-3.0,2.0-1*eta]], eta=eta)
            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=[-0.5, 0.0, 0.5], eta=eta)
            gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=z_contours, redshifts=redshifts, eta=eta)

            # set contour levels
            if eta:
                contours = [-2.5, -2.0]
            else:
                contours = [-1.0, -0.5, 0.0] # msun/yr

            optsSets = [ {'mStarBins':mStarBinsSm, 'contours':contours, 'eta':eta}, # multi-contour, single redshift
                         {'mStarBins':mStarBinsSm, 'contours':z_contours, 'redshifts':redshifts, 'eta':eta},  # single-contour, multi-redshift
                         {'mStarBins':mStarBinsSm, 'clims':[[-3.0,0.5-1*eta]], 'eta':eta} ] # no contours, instead many panels

            for opts in optsSets:

                for quant in ['temp','numdens','z_solar','theta']:
                    gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis=quant, **opts)
                    gasOutflowRates2DStackedInMstar(TNG50, xAxis=quant, yAxis='vcut', **opts)

                gasOutflowRates2DStackedInMstar(TNG50, xAxis='numdens', yAxis='temp', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='z_solar', yAxis='temp', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='theta', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='z_solar', yAxis='theta', **opts)

                gasOutflowRates2DStackedInMstar(TNG50, xAxis='temp', yAxis='vrad', **opts)
                gasOutflowRates2DStackedInMstar(TNG50, xAxis='rad', yAxis='vrad', **opts)

    if 0:
        # explore: radial profiles: stellar mass stacks, at one redshift
        TNG50.setRedshift(1.0)
        if sPs is None: sPs = [TNG50]
        cenSatSelect = 'cen'

        for field in radProfileFields:
            pdf = PdfPages('radprofiles_%s_%s_%d_%s.pdf' % (field,sPs[0].simName,sPs[0].snap,cenSatSelect))

            for projDim in ['2Dz','3D','2Dfaceon','2Dedgeon']:
                if projDim == '3D' and '_LOSVel' in field: continue # undefined
                if projDim in ['2Dfaceon','2Dedgeon'] and 'LOSVel' not in field: continue # just 2Dz,3D for most

                for xaxis in ['log_pkpc','pkpc','log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
                    stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                                          projDim=projDim, mStarBins=mStarBins, pdf=pdf)
            pdf.close()

        return sPs

    if 0:
        # explore: radial profiles: vs redshift, separate plot for each mstar bin
        redshifts = [1.0, 2.0, 4.0, 6.0]
        cenSatSelect = 'cen'

        if sPs is None:
            sPs = []
            for redshift in redshifts:
                sP = simParams(res=2160, run='tng', redshift=redshift)
                sPs.append(sP)

        for field in radProfileFields:
            pdf = PdfPages('radprofiles_%s_%s_zevo_%s.pdf' % (field,sPs[0].simName,cenSatSelect))

            for projDim in ['2Dz','3D','2Dfaceon','2Dedgeon']:
                if projDim == '3D' and '_LOSVel' in field: continue # undefined
                if projDim in ['2Dfaceon','2Dedgeon'] and 'LOSVel' not in field: continue # just 2Dz,3D for most

                for xaxis in ['log_pkpc','log_rvir','log_rhalf','pkpc']:
                    for i, mStarBin in enumerate(mStarBins):
                        stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect=cenSatSelect, 
                            projDim=projDim, mStarBins=[mStarBin], colorOff=i, pdf=pdf)

            pdf.close()

        return sPs

    # exploration: eta_M vs stellar/halo mass, split by everything else
    if 0:
        sPs = [TNG50]

        css = 'cen'
        #quants = quantList(wCounts=False, wTr=False, wMasses=True)
        quants = ['ssfr']
        priQuant = 'vout_99_all' #'etaM_100myr_10kpc_0kms'
        sLowerPercs = [10,50]
        sUpperPercs = [90,50]

        for xQuant in ['mstar_30pkpc','mhalo_200_log']:
            # individual plot per y-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_slice=%s.pdf' % (sPs[0].simName,xQuant,css,priQuant))
            for yQuant in quants:
                quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=priQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)
            pdf.close()

            # individual plot per s-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_y=%s.pdf' % (sPs[0].simName,xQuant,css,priQuant))
            for sQuant in quants:
                quantMedianVsSecondQuant(sPs, pdf, yQuants=[priQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)

            pdf.close()

    # exporation: 2d histos of new quantities (delta_sfms) vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160,run='tng',redshift=2.0)
        figsize_loc = [figsize[0]*0.7, figsize[1]*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50
        yQuant = 'delta_sfms'
        cRel   = None #[0.7,1.3,False] # [cMin,cMax,cLog] #None

        #cQuant = 'vout_75_all'
        #cFrac  = [250, np.inf, False, 'Fast Outflow Fraction ($v_{\\rm out}$ > 250 km/s)'] #[200, np.inf, False]

        cQuant = 'etaM_100myr_10kpc_0kms'
        cFrac  = [20.0, np.inf, False, None]

        params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'cRel':cRel, 'cFrac':cFrac, 'nBins':nBins}

        for xQuant in xQuants:
            pdf = PdfPages('histo2d_x=%s_y=%s_c=%s_%s_%d.pdf' % (xQuant,yQuant,cQuant,sP.simName,sP.snap))
            fig = plt.figure(figsize=figsize_loc)
            quantHisto2D(sP, pdf, xQuant=xQuant, yQuant=yQuant, fig_subplot=[fig,111], **params)
            pdf.close()

    # exploration: 2d histos of everything vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160,run='tng',redshift=1.0)
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50

        cQuants = ['vout_75_all']#,'etaM_100myr_10kpc_0kms']
        cRel   = [0.7,1.3,False] # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == '_dummy_': quants3[-1] = xQuants[1-i] # include the other

            for cQuant in cQuants:

                # for each (x) quant, make a number of 6-panel figures, different y-axis (same coloring) for every panel
                for j, yQuants in enumerate([quants2]): #quantSets):
                    params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant, 'cRel':cRel, 'nBins':nBins}

                    pdf = PdfPages('histo2d_x=%s_c=%s_set-%d_%s_%d%s.pdf' % (xQuant,cQuant,j,sP.simName,sP.snap,'_rel' if cRel is not None else ''))
                    fig = plt.figure(figsize=figsize_loc)
                    quantHisto2D(sP, pdf, yQuant=yQuants[0], fig_subplot=[fig,321], **params)
                    quantHisto2D(sP, pdf, yQuant=yQuants[1], fig_subplot=[fig,322], **params)
                    quantHisto2D(sP, pdf, yQuant=yQuants[2], fig_subplot=[fig,323], **params)
                    quantHisto2D(sP, pdf, yQuant=yQuants[3], fig_subplot=[fig,324], **params)
                    quantHisto2D(sP, pdf, yQuant=yQuants[4], fig_subplot=[fig,325], **params)
                    quantHisto2D(sP, pdf, yQuant=yQuants[5], fig_subplot=[fig,326], **params)
                    pdf.close()

    # exploration: 2d histos of new quantities (vout,eta,BH_BolLum,etc) vs M*, colored by everything else
    if 0:
        sP = simParams(res=2160,run='tng',redshift=1.0)
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        nBins = 50

        yQuants = ['vout_90_all','etaM_100myr_10kpc_0kms','delta_sfms']
        cRel = None #[0.7,1.3,False] # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == '_dummy_': quants3[-1] = xQuants[1-i].replace('_log','') # include the other

            for yQuant in yQuants:

                # for each (x,y) quant set, make a number of 6-panel figures, different coloring for every panel
                for j, cQuants in enumerate(quantSets):
                    params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'yQuant':yQuant, 'xQuant':xQuant, 'cRel':cRel, 'nBins':nBins}

                    pdf = PdfPages('histo2d_x=%s_y=%s_set-%d_%s_%d%s.pdf' % (xQuant,yQuant,j,sP.simName,sP.snap,'_rel' if cRel is not None else ''))
                    fig = plt.figure(figsize=figsize_loc)
                    quantHisto2D(sP, pdf, cQuant=cQuants[0], fig_subplot=[fig,321], **params)
                    quantHisto2D(sP, pdf, cQuant=cQuants[1], fig_subplot=[fig,322], **params)
                    quantHisto2D(sP, pdf, cQuant=cQuants[2], fig_subplot=[fig,323], **params)
                    quantHisto2D(sP, pdf, cQuant=cQuants[3], fig_subplot=[fig,324], **params)
                    quantHisto2D(sP, pdf, cQuant=cQuants[4], fig_subplot=[fig,325], **params)
                    quantHisto2D(sP, pdf, cQuant=cQuants[5], fig_subplot=[fig,326], **params)
                    pdf.close()
