"""
general.py
  General exploratory/diagnostic plots of single halos or entire boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import binned_statistic, binned_statistic_2d

from util import simParams
from util.helper import loadColorTable, getWhiteBlackColors, running_median, logZeroNaN, iterable
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.util import periodicDists
from plot.quantities import quantList, simSubhaloQuantity, simParticleQuantity
from plot.config import *

def plotHistogram1D(sPs, ptType='gas', ptProperty='temp_linear', ptWeight=None, subhaloIDs=None, haloIDs=None, 
      ylog=True, ylim=None, xlim=None, sfreq0=False, pdf=None):
    """ Simple 1D histogram/PDF of some quantity ptProperty of ptType, either of the whole box (subhaloIDs and haloIDs 
    both None), or of one or more halos/subhalos, where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    Alternatively, subhaloIDs/haloIDs can be a list of lists, each sub-list of IDs of objects for that run, which are overplotted.
    If ptWeight is None, uniform weighting, otherwise weight by this quantity.
    If sfreq0 is True, include only non-starforming cells. 
    If ylim is None, then hard-coded typical limits for PDFs. If 'auto', then autoscale. Otherwise, 2-tuple to use as limits. """

    # config
    if ylog:
        ylabel = 'PDF [ log ]'
        if ylim is None:
            ylim = [-3.0, 1.0]
        else:
            if ylim == 'auto': ylim = None
    else:
        ylim = [0.0, 1.0]
        ylabel = 'PDF'

    nBins = 400
    lw = 2.0

    # special behavior (not yet generalized)
    coldDenseCGM = False # zooms2.josh
    radWithin10pkpc = False

    # inputs
    oneObjPerRun = False

    assert np.sum(e is not None for e in [haloIDs,subhaloIDs]) in [0,1] # pick one, or neither
    if subhaloIDs is not None:
        assert (len(subhaloIDs) == len(sPs)) or len(sPs) == 1 # one subhalo ID per sP, or one sP
        assert isinstance(subhaloIDs, (list,np.ndarray))
        if not isinstance(subhaloIDs[0], (list,np.ndarray)):
            assert len(subhaloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = subhaloIDs

    if haloIDs is not None:
        assert (len(haloIDs) == len(sPs)) or len(sPs) == 1 # one subhalo ID per sP, or one sP
        assert isinstance(haloIDs, (list,np.ndarray))
        if not isinstance(haloIDs[0], (list,np.ndarray)):
            assert len(haloIDs) == len(sPs)
            oneObjPerRun = True
        objIDs = haloIDs

    # load
    haloLims = (subhaloIDs is not None or haloIDs is not None)
    xlabel, xlim_quant, xlog = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean, haloLims=haloLims)
    if xlim is None: xlim = xlim_quant

    # start plot
    fig = plt.figure(figsize=(14,10)) #(11.2,8.0)
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # loop over halo/subhalo IDs
        if subhaloIDs is not None or haloIDs is not None:
            if not oneObjPerRun:
                sP_objIDs = objIDs[i] # list
            else:
                sP_objIDs = [objIDs[i]]

        for objID in sP_objIDs:
            # get corresponding halo or subhalo ID for this object
            #if subhaloIDs is not None:
            #    haloID = groupCatSingle(sP, subhaloID=objID)['SubhaloGrNr']
            #if haloIDs is not None:
            #    subhaloID = groupCatSingle(sP, haloID=objID)['GroupFirstSub']

            # load
            load_haloID = objID if haloIDs is not None else None
            load_subID = objID if subhaloIDs is not None else None

            vals = snapshotSubset(sP, ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)
            if xlog: vals = np.log10(vals)

            print(sP.simName,', min: ', np.nanmin(vals), ' max: ', np.nanmax(vals))

            # weights
            if ptWeight is None:
                weights = np.zeros( vals.size, dtype='float32' ) + 1.0
            else:
                weights = snapshotSubset(sP, ptType, ptWeight, haloID=load_haloID, subhaloID=load_subID)

            if sfreq0:
                # restrict to non eEOS cells
                sfr = snapshotSubset(sP, ptType, 'sfr', haloID=load_haloID, subhaloID=load_subID)
                w_sfr = np.where(sfr == 0.0)
                vals = vals[w_sfr]
                weights = weights[w_sfr]

            if coldDenseCGM:
                print('Restricting to cold-dense phase of the CGM.')
                temp = snapshotSubset(sP, ptType, 'temp', haloID=load_haloID, subhaloID=load_subID)
                hdens = snapshotSubset(sP, ptType, 'hdens', haloID=load_haloID, subhaloID=load_subID)
                if sfreq0:
                    temp = temp[w_sfr]
                    hdens = hdens[w_sfr]
                w0 = np.where( (temp < 5.0) & (hdens > 1e-3) )
                vals = vals[w0]
                weights = weights[w0]

            if radWithin10pkpc:
                print('Restricting to rad > 10 pkpc.')
                rad = snapshotSubset(sP, ptType, 'rad_kpc', haloID=load_haloID, subhaloID=load_subID)
                if sfreq0: rad = rad[w_sfr]
                if coldDenseCGM: rad = rad[w0]

                wr = np.where( rad > 10.0 )
                vals = vals[wr]
                weights = weights[wr]

            # histogram (all equivalent methods)
            bins = np.linspace( xlim[0], xlim[1], nBins+1 )
            xx = bins[:-1] + (bins[1]-bins[0])/2
            binsize = bins[1]-bins[0]

            #yy4 = np.zeros( nBins, dtype='float32' )
            #for j in range(nBins):
            #    w = np.where( (vals >= bins[j]) & (vals < bins[j+1]) )
            #    yy4[j] = np.sum(weights[w])
            #yy4 /= (np.sum(weights)*binsize)
            
            #yy2, xx2, _ = binned_statistic(vals, weights, statistic='sum', bins=bins)
            #yy2 /= (vals.size*binsize)
            #yy3, xx3 = np.histogram(vals, bins=bins, weights=weights, density=False)
            #yy3 /= (vals.size*binsize)
            yy, xx4 = np.histogram(vals, bins=bins, weights=weights, density=True)

            if ylog: yy = logZeroNaN(yy)
            #if xx.size > sKn:
            #    yy = savgol_filter(yy,sKn,sKo)

            label = '%s [%d]' % (sP.simName,objID)
            ls = ':' if sP.simName == 'L11 (Primordial Only)' else '-'
            l, = ax.plot(xx, yy, linestyle=ls, lw=lw, label=label)

    # finish plot
    fig.tight_layout()
    ax.legend(loc='upper left')

    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    hStr = 'global'
    if haloIDs is not None: hStr = 'haloIDs-n%d' % len(haloIDs)
    elif subhaloIDs is not None: hStr = 'subhIDs-n%d' % len(subhaloIDs)

    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('histo1D_%s_%s_%s_wt-%s_%s.pdf' % (sPstr,ptType,ptProperty,ptWeight,hStr))
    plt.close(fig)

def plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', weights=['mass'], meancolors=None, haloID=None, pdf=None,
                     contours=None, xlim=None, ylim=None, clim=None, hideBelow=False, smoothSigma=0.0, nBins=None, sfreq0=False):
    """ Plot a 2D phase space plot (arbitrary values on x/y axes), for a single halo or for an entire box 
    (if haloID is None). weights is a list of the gas properties to weight the 2D histogram by, 
    if more than one, a horizontal multi-panel plot will be made with a single colorbar. Or, if meancolors is 
    not None, then show the mean value per pixel of these quantities, instead of weighted histograms.
    If xlim,ylim,clim specified, then use these bounds, otherwise use default/automatic bounds.
    If contours is not None, draw solid contours at these levels on top of the 2D histogram image. 
    If smoothSigma is not zero, gaussian smooth contours at this level. 
    If hideBelow, then pixel values below clim[0] are left pure white. 
    If sfreq0 is True, include only non-starforming cells. """

    # config
    if nBins is None:
        nBinsX = 800
        nBinsY = 400
        if sP.isZoom:
            nBinsX = 250
            nBinsY = 250
    else:
        nBinsX = nBins
        nBinsY = nBins

    sizefac = 0.9
    clim_default = [-6.0,0.0]

    # binned_statistic_2d instead of histogram2d?
    binnedStat = False
    if meancolors is not None:
        assert weights is None # one or the other
        binnedStat = True
        weights = iterable(meancolors) # loop over these instead
    if weights is None:
        # one or the other
        assert meancolors is not None

    ctNameHisto = 'viridis'
    contoursColor = 'k' # black

    # load: x-axis
    xlabel, xlim_quant, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
    if xlim is None: xlim = xlim_quant
    xvals = snapshotSubset(sP, partType, xQuant, haloID=haloID)

    if xlog: xvals = np.log10(xvals)

    # load: y-axis
    ylabel, ylim_quant, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
    if ylim is None: ylim = ylim_quant
    yvals = snapshotSubset(sP, partType, yQuant, haloID=haloID)

    if ylog: yvals = np.log10(yvals)

    if sfreq0:
        # restrict to non eEOS cells
        sfr = snapshotSubset(sP, partType, 'sfr', haloID=haloID)
        w_sfr = np.where(sfr == 0.0)
        xvals = xvals[w_sfr]
        yvals = yvals[w_sfr]

    # start figure
    fig = plt.figure(figsize=[figsize[0]*sizefac*(len(weights)*0.9), figsize[1]*sizefac])

    # loop over each weight requested
    for i, wtProp in enumerate(weights):
        # load: weights
        weight = snapshotSubset(sP, partType, wtProp, haloID=haloID)

        if sfreq0:
            weight = weight[w_sfr]

        # add panel
        ax = fig.add_subplot(1,len(weights),i+1)

        if len(weights) == 1: # title
            hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
            wtStr = partType.capitalize() + ' ' + wtProp.capitalize()
            ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # oxygen paper manual fix: remove interpolation wiggles near sharp dropoff
        if xQuant == 'hdens' and yQuant == 'temp' and len(weights) == 3:
            if wtProp == 'O VI mass':
                w = np.where( ((xvals > -3.7) & (yvals < 5.0)) | ((xvals > -3.1) & (yvals < 5.15)) )
                yvals[w] = 0.0
            if wtProp == 'O VII mass':
                w = np.where( ((xvals > -4.0) & (yvals < 5.0)) | ((xvals > -3.5) & (yvals < 5.15)) )
                yvals[w] = 0.0
            if wtProp == 'O VIII mass':
                w = np.where( ((xvals > -4.8) & (yvals < 5.1)) | ((xvals > -4.4) & (yvals < 5.3)) )
                yvals[w] = 0.0

        if binnedStat:
            # plot 2D image, each pixel colored by the mean value of a third quantity
            clabel, clim_quant, clog = simParticleQuantity(sP, partType, wtProp, clean=clean, haloLims=(haloID is not None))
            wtStr = 'Mean ' + clabel
            zz, _, _, _ = binned_statistic_2d(xvals, yvals, weight, 'mean', # median unfortunately too slow
                                                         bins=[nBinsX, nBinsY], range=[xlim,ylim])
            zz = zz.T
            if clog: zz = logZeroNaN(zz)

            if clim is None:
                clim = clim_quant # colorbar limits
        else:
            # plot 2D histogram image, optionally weighted
            zz, _, _ = np.histogram2d(xvals, yvals, bins=[nBinsX, nBinsY], range=[xlim,ylim], 
                                        normed=True, weights=weight)
            zz = logZeroNaN(zz.T)

        if clim is None:
            clim = clim_default

        if hideBelow:
            w = np.where(zz < clim[0])
            zz[w] = np.nan

        cmap = loadColorTable(ctNameHisto)
        norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

        # plot contours?
        if contours is not None:
            if binnedStat:
                zz, xc, yc, _ = binned_statistic_2d(xvals, yvals, weight, 'mean',
                                                             bins=[nBinsX/4, nBinsY/4], range=[xlim,ylim])
                if clog: zz = logZeroNaN(zz)
            else:
                zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBinsX/4, nBinsY/4], range=[xlim,ylim], 
                                            normed=True, weights=weight)
                zz = logZeroNaN(zz)

            XX, YY = np.meshgrid(xc[:-1], yc[:-1], indexing='ij')

            # smooth, ignoring NaNs
            if smoothSigma > 0:
                zz1 = zz.copy()
                zz1[np.isnan(zz)] = 0.0
                zz1 = gaussian_filter(zz1, smoothSigma)
                zz2 = 0 * zz.copy() + 1.0
                zz2[np.isnan(zz)] = 0.0
                zz2 = gaussian_filter(zz2, smoothSigma)
                zz = zz1/zz2

            c = plt.contour(XX, YY, zz, contours, colors=contoursColor, linestyles='solid')

        if len(weights) > 1: # text label inside panel
            wtStr = 'Gas Oxygen Ion Mass'
            labelText = wtProp.replace(" mass","").replace(" ","")
            ax.text(xlim[0]+0.3, yMinMax[-1]-0.3, labelText, 
                va='top', ha='left', color='black', fontsize='40')

    # colorbar and save
    fig.tight_layout()
    
    if xQuant == 'hdens' and yQuant == 'temp' and len(weights) == 3:
        # TNG colors paper
        fig.subplots_adjust(right=0.93)
        cbar_ax = fig.add_axes([0.94, 0.131, 0.02, 0.831]) 
    else:
        # more general
        fig.subplots_adjust(right=0.89)
        cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)

    cb = plt.colorbar(im, cax=cbar_ax)
    if not binnedStat: wtStr = 'Relative ' + wtStr + ' [ log ]'
    cb.ax.set_ylabel(wtStr)
    
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('phase_%s_z%.1f_%s_x-%s_y-%s_wt-%s_h-%s.pdf' % \
            (sP.simName,sP.redshift,partType,xQuant,yQuant,"-".join([w.replace(" ","") for w in weights]),haloID))
    plt.close(fig)

def plotParticleMedianVsSecondQuant(sPs, partType='gas', xQuant='hdens', yQuant='Si_H_numratio', 
                                   haloID=None, radMinKpc=None, radMaxKpc=None):
    """ Plot the (median) relation between two particle properties for a single halo (if haloID is not None), 
    or the whole box (if haloID is None). If a halo is specified, optionally restrict to a given 
    [radMinKpc,radMaxKpc] radial range, specified in physical kpc."""

    # config
    nBins = 50
    lw = 3.0

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    sStr = '%s z=%.1f' % (sPs[0].simName, sPs[0].redshift) if len(sPs) == 1 else ''
    ax.set_title('%s %s' % (sStr,hStr))

    for i, sP in enumerate(sPs):
        # load
        xlabel, xlim, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
        sim_xvals = snapshotSubset(sP, partType, xQuant, haloID=haloID)
        if xlog: sim_xvals = logZeroNaN(sim_xvals)

        ylabel, ylim, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
        sim_yvals = snapshotSubset(sP, partType, yQuant, haloID=haloID)
        if ylog: sim_yvals = logZeroNaN(sim_yvals)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # radial restriction
        if radMaxKpc is not None or radMinKpc is not None:
            assert haloID is not None
            rad = snapshotSubset(sP, partType, 'rad_kpc', haloID=haloID)
            
            if radMinKpc is None:
                w = np.where( (rad <= radMaxKpc) )
            elif radMaxKpc is None:
                w = np.where( (rad > radMinKpc) )
            else:
                w = np.where( (rad > radMinKpc) & (rad <= radMaxKpc) )

            sim_xvals = sim_xvals[w]
            sim_yvals = sim_yvals[w]

            if radMinKpc is not None:
                hStr += '_rad_gt_%.1fkpc' % radMinKpc
            if radMaxKpc is not None:
                hStr += '_rad_lt_%.1fkpc' % radMaxKpc

        # median and 10/90th percentile lines
        binSize = (xlim[1]-xlim[0]) / nBins

        xm, ym, sm, pm = running_median(sim_xvals,sim_yvals,binSize=binSize,percs=[5,10,25,75,90,95])
        xm = xm[1:-1]
        ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
        sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
        pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

        c = ax._get_lines.prop_cycler.next()['color']
        ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

        # percentile:
        if len(sPs) <= 3 or (len(sPs) > 3 and i == 0):
            ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    ax.legend(loc='best')

    # finish plot
    sStr = '%s_z-%.1f' % (sPs[0].simName,sPs[0].redshift) if len(sPs) == 1 else 'sPn%d' % len(sPs)
    fig.savefig('particleMedian_%s_%s-vs-%s_%s_%s.pdf' % (partType,xQuant,yQuant,sStr,hStr))
    plt.close(fig)

def plotStackedRadialProfiles1D(sPs, subhalo=None, ptType='gas', ptProperty='temp_linear', weighting=None, halo=None):
    """ Radial profile(s) of some quantity ptProperty of ptType vs. radius from halo centers 
    (parent FoF particle restricted, using non-caching auxCat functionality). 
    subhalo is a list, one entry per sPs entry. For each entry of subhalo:
    If subhalo[i] is a single subhalo ID number, then one halo only. If a list, then median stack.
    If a dict, then k:v pairs where keys are a string description, and values are subhaloID lists, which 
    are then overplotted. sPs supports one or multiple runs to be overplotted. 
    If halo is not None, then use these FoF IDs as inputs instead of Subfind IDs. """
    from cosmo.auxcatalog import subhaloRadialProfile
    from tracer.tracerMC import match3

    # config
    xlim = [0.0,3.0] # for plot only [loc pkpc]
    percs = [10,90]
    lw = 2.0
    scope = 'fof' # fof, subfind
    ptRestriction = 'sfreq0' # None
    op = 'mean' # mean, sum, min, max

    assert subhalo is not None or halo is not None # pick one
    if subhalo is None: subhalo = halo # use halo ids
    if isinstance(subhalo,(int,long)) and len(sPs) == 1: subhalo = [subhalo] # single number to list (one sP case)
    assert (len(subhalo) == len(sPs)) # one subhalo ID list per sP

    ylabel, ylim, ylog = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean)

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    ax.set_xlabel('radius [ log pkpc ]')
    ax.set_ylabel(ylabel)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        subhaloIDs = subhalo[i] # for this run

        # subhalo is a single number or dict? make a concatenated list
        if isinstance(subhaloIDs,(int,long)):
            subhaloIDs = [subhaloIDs]
        if isinstance(subhaloIDs,dict):
            subhaloIDs = np.hstack( [subhaloIDs[key] for key in subhaloIDs.keys()])

        if halo is not None:
            # transform fof ids to subhalo ids
            firstsub = groupCat(sP, fieldsHalos=['GroupFirstSub'], sq=True)
            subhaloIDs = firstsub[subhaloIDs]

        # load
        data, attrs = subhaloRadialProfile(sP, pSplit=None, ptType=ptType, ptProperty=ptProperty, op=op, 
                                          scope=scope, weighting=weighting, subhaloIDsTodo=subhaloIDs, 
                                          ptRestriction=ptRestriction)
        assert data.shape[0] == len(subhaloIDs)

        nSamples = 1 if not isinstance(subhalo[i],dict) else len(subhalo[i].keys())

        for j in range(nSamples):
            # crossmatch attrs['subhaloIDs'] with subhalo[key] sub-list if needed
            subIDsLoc = subhalo[i][subhalo[i].keys()[j]] if isinstance(subhalo[i],dict) else subhaloIDs
            w, _ = match3( attrs['subhaloIDs'], subIDsLoc )
            assert len(w) == len(subIDsLoc)

            # calculate median radial profile and scatter
            yy_mean = np.nanmean( data[w,:], axis=0 )
            yy_median = np.nanmedian( data[w,:], axis=0 )
            yp = np.nanpercentile( data[w,:], percs, axis=0 )

            if ylog:
                yy_median = logZeroNaN(yy_median)
                yy_mean = logZeroNaN(yy_mean)
                yp = logZeroNaN(yp)
            rr = logZeroNaN(attrs['rad_bins_pkpc'])

            if rr.size > sKn:
                yy_mean = savgol_filter(yy_mean,sKn,sKo)
                yy_median = savgol_filter(yy_median,sKn,sKo)
                yp = savgol_filter(yp,sKn,sKo,axis=1) # P[10,90]

            sampleDesc = '' if nSamples == 1 else subhalo[i].keys()[j]
            l, = ax.plot(rr, yy_median, '-', lw=lw, label='%s %s' % (sP.simName,sampleDesc))
            ax.plot(rr, yy_mean, ':', lw=lw, color=l.get_color())
            if len(sPs) == 1 and subhaloIDs.size > 1:
                ax.fill_between(rr, yp[0,:], yp[-1,:], color=l.get_color(), interpolate=True, alpha=0.2)

    # finish plot
    fig.tight_layout()
    ax.legend(loc='best')
    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    wtStr = '_wt-'+weighting if weighting is not None else ''
    fig.savefig('radProfilesStack_%s_%s_%s_Ns-%d_Nh-%d_scope-%s%s.pdf' % \
        (sPstr,ptType,ptProperty,nSamples,len(subhaloIDs),scope,wtStr))
    plt.close(fig)

def plotSingleRadialProfile(sPs, ptType='gas', ptProperty='temp_linear', subhaloIDs=None, haloIDs=None, 
    xlog=True, xlim=None, sfreq0=False, colorOffs=None):
    """ Radial profile of some quantity ptProperty of ptType vs. radius from halo center,
    where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs. """

    # config
    if xlog:
        if xlim is not None: xlim = [-0.5,3.0]
        xlabel = 'radius [ log pkpc ]'
    else:
        if xlim is not None: xlim = [0.0,500.0]
        xlabel = 'radius [ pkpc ]'

    percs = [10,25,75,90]
    nRadBins = 40
    lw = 2.0
    scope = 'fof' # global, fof, subfind

    assert np.sum(e is not None for e in [haloIDs,subhaloIDs]) == 1 # pick one
    if subhaloIDs is not None: assert (len(subhaloIDs) == len(sPs)) # one subhalo ID per sP
    if haloIDs is not None: assert (len(haloIDs) == len(sPs)) # one subhalo ID per sP

    ylabel, ylim, ylog = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean, haloLims=True)

    # start plot
    fig = plt.figure(figsize=(11.2,8.0)) #(14,10)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Galactocentric Radius [ log pkpc ]')
    ax.set_ylabel(ylabel)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)

    # loop over simulations
    for i, sP in enumerate(sPs):
        # get halo and subhalo IDs
        if subhaloIDs is not None:
            subhaloID = subhaloIDs[i]
            haloID = groupCatSingle(sP, subhaloID=subhaloID)['SubhaloGrNr']
        else:
            haloID = haloIDs[i]
            subhaloID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub']

        # load
        load_haloID = haloID if scope == 'fof' else None
        load_subID = subhaloID if scope == 'subfind' else None
        if load_haloID is None and load_subID is None: assert scope == 'global'

        rad  = snapshotSubset(sP, ptType, 'rad_kpc', haloID=load_haloID, subhaloID=load_subID)
        vals = snapshotSubset(sP, ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)

        if sfreq0:
            # restrict to non eEOS cells
            sfr = snapshotSubset(sP, ptType, 'sfr', haloID=load_haloID, subhaloID=load_subID)
            w = np.where(sfr == 0.0)
            rad = rad[w]
            vals = vals[w]

        # radial bin
        rad_bins = np.linspace( xlim[0], xlim[1], nRadBins+1 )
        rr = rad_bins[:-1] + (rad_bins[1]-rad_bins[0])/2
        if xlog: rad = np.log10(rad)

        yy_mean = np.zeros( nRadBins, dtype='float32' )
        yy_median = np.zeros( nRadBins, dtype='float32' )
        yy_perc = np.zeros( (len(percs),nRadBins), dtype='float32' )

        for j in range(nRadBins):
            # calculate median radial profile and scatter
            w = np.where( (rad >= rad_bins[j]) & (rad < rad_bins[j+1]) )
            if len(w[0]) == 0:
                continue

            yy_mean[j] = np.nanmean( vals[w] )
            yy_median[j] = np.nanmedian( vals[w] )
            yy_perc[:,j] = np.nanpercentile( vals[w], percs )

        if ylog:
            yy_mean = logZeroNaN(yy_mean)
            yy_median = logZeroNaN(yy_median)
            yy_perc = logZeroNaN(yy_perc)

        if rr.size > sKn:
            yy_mean = savgol_filter(yy_mean,sKn,sKo)
            yy_median = savgol_filter(yy_median,sKn,sKo)
            yy_perc = savgol_filter(yy_perc,sKn,sKo,axis=1) # P[10,90]

        # plot lines
        if colorOffs is not None:
            for _ in range(colorOffs[i]):
                _ = ax._get_lines.prop_cycler.next()['color']

        label = '%s haloID=%d [%s]' % (sP.simName,haloID,scope) if not clean else sP.simName
        l, = ax.plot(rr, yy_median, '-', lw=lw, label=label)
        ax.plot(rr, yy_mean, '--', lw=lw, color=l.get_color())

        if len(sPs) <= 2:
            for j in range(yy_perc.shape[0]/2):
                ax.fill_between(rr, yy_perc[0+j,:], yy_perc[-(j+1),:], color=l.get_color(), interpolate=True, alpha=0.15*(j+1))

    # special behavior
    if 0:
        rad_cgm_zoom = {'r$_{\\rm CGM,min}$':10,'r$_{\\rm CGM,max}$':300,'r$_{\\rm IGM}$':500} # pkpc
        ylim_p = [ylim[0] + (ylim[1]-ylim[0])/15, ylim[1] - (ylim[1]-ylim[0])/15]
        alpha = 1.0 if ptProperty == 'mass_msun' else 0.2
        for label,rad in rad_cgm_zoom.items():
            off = {10:0.0, 300:-0.04, 500:+0.04}[rad]
            if xlog: rad = np.log10(rad)
            ax.plot( [rad,rad], ylim_p, '-', lw=lw, color='black', alpha=0.1 )
            ax.text( rad+off, ylim_p[0], label, color='black', fontsize=20, alpha=alpha, verticalalignment='top', horizontalalignment='center')

    if ptProperty == 'cellsize_kpc':
        xlim_p = [xlim[0] + (xlim[1]-xlim[0])/40, xlim[1] - (xlim[1]-xlim[0])/40]
        notable_sizes = {'100pc':-1.0, '1kpc':0.0}
        for label,val in notable_sizes.items():
            ax.plot( xlim_p, [val,val], ':', lw=lw, color='black', alpha=0.05 )
            ax.text( xlim_p[0]+0.1, val+0.03, label, color='black', fontsize=20, alpha=0.1, verticalalignment='bottom', horizontalalignment='center' )

    # finish plot
    fig.tight_layout()
    ax.legend(loc='best')

    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    if haloIDs is not None:
        hStr = 'haloID-%d' % haloIDs[0] if len(haloIDs) == 1 else 'nH-%d' % len(haloIDs)
    else:
        hStr = 'subhID-%d' % subhaloIDs[0] if len(subhaloIDs) == 1 else 'nSH-%d' % len(subhaloIDs)

    fig.savefig('radProfile_%s_%s_%s_%s_scope-%s.pdf' % (sPstr,ptType,ptProperty,hStr,scope))
    plt.close(fig)

# -------------------------------------------------------------------------------------------------

def compareRuns_PhaseDiagram():
    """ Driver. Compare a series of runs in a PDF booklet of phase diagrams. """
    import glob
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    yAxis = 'temp'
    xAxis = 'numdens'

    # get list of all 512 method runs via filesystem search
    sP = simParams(res=512,run='tng',redshift=0.0,variant='0000')
    dirs = glob.glob(sP.arepoPath + '../L25n512_*')
    variants = sorted([d.rsplit("_",1)[1] for d in dirs])

    # start PDF, add one page per run
    pdf = PdfPages('compareRunsPhaseDiagram.pdf')

    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=0.0,variant=variant)
        if sP.simName == 'DM only': continue
        print(variant,sP.simName)
        plotPhaseSpace2D(sP, yAxis, xAxis=xAxis, haloID=None, pdf=pdf)

    pdf.close()

def compareRuns_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two different 
    types of halos. One run. """
    from projects.oxygen import variantsMain as variants

    sPs = []
    subhalos = []

    for variant in variants:
        sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

        mhalo = groupCat(sPs[-1], fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w = np.where( (mhalo > 11.5) & (mhalo < 12.5) )

        subhalos.append( w[0] )

    for field in ['metaldens']: #,'dens','temp_linear','P_gas_linear','z_solar']:
        plotStackedRadialProfiles1D(sPs, subhalo=subhalos, ptType='gas', ptProperty=field, weighting='O VI mass')

def compareHaloSets_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two different 
    types of halos. One run. """
    sPs = []
    sPs.append( simParams(res=1820,run='tng',redshift=0.0) )

    mhalo = groupCat(sPs[0], fieldsSubhalos=['mhalo_200_log'])
    gr,_,_,_ = simSubhaloQuantity(sPs[0], 'color_B_gr')

    with np.errstate(invalid='ignore'):
        w1 = np.where( (mhalo > 11.8) & (mhalo < 12.2) & (gr < 0.35) )
        w2 = np.where( (mhalo > 11.8) & (mhalo < 12.2) & (gr > 0.65) )

    print( len(w1[0]), len(w2[0]) )

    subhalos = [{'11.8 < M$_{\\rm halo}$ < 12.2, (g-r) < 0.35':w1[0], 
                 '11.8 < M$_{\\rm halo}$ < 12.2, (g-r) > 0.65':w2[0]}]

    for field in ['tcool']: #['metaldens','dens','temp_linear','P_gas_linear','z_solar']:
        plotStackedRadialProfiles1D(sPs, subhalo=subhalos, ptType='gas', ptProperty=field, weighting='O VI mass')

def compareHaloSets_1DHists():
    """ Driver. Compare 1D histograms of a quantity, overplotting several halos. One run. """
    sPs = []

    sPs.append( simParams(res=910,run='tng',redshift=0.0) )
    mhalo = groupCat(sPs[-1], fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w1 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
    subhaloIDs = [w1[0][0:5]]

    if 0:
        # add a second run
        sPs.append( simParams(res=455,run='tng',redshift=0.0) )
        mhalo = groupCat(sPs[-1], fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w2 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
        subhaloIDs.append( w2[0][0:5] )

    for field in ['temp']: #['tcool','vrad']:
        plotHistogram1D(sPs, subhaloIDs=subhaloIDs, ptType='gas', ptProperty=field)

def singleHaloProperties():
    """ Driver. Several phase/radial profile plots for a single halo. """
    if 1:
        sPs = []
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='PO',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='MO',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='FP',redshift=2.25) )
        sPs.append( simParams(res=9,run='zooms2',hInd=2,redshift=2.25) )
        haloIDs = np.zeros( len(sPs), dtype='int32' )

        #plotSingleRadialProfile(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='mass_msun', colorOffs=[0,2], xlim=[2.0,4.7])
        #plotSingleRadialProfile(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='cellsize_kpc', sfreq0=True, colorOffs=[0,2])
        plotHistogram1D(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='mass_msun', sfreq0=True)
        #plotHistogram1D(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='cellsize_kpc', sfreq0=True)

        #for prop in ['hdens','temp_linear','cellsize_kpc','radvel','temp']:
        #    plotStackedRadialProfiles1D([sP], halo=haloID, ptType='gas', ptProperty=prop)
        #    plotPhaseSpace2D(sP, partType='gas', xQuant='hdens', yQuant=prop, haloID=haloID)

    if 0:
        sP = simParams(res=455,run='tng',redshift=0.0)

        # pick a MW
        gc = groupCat(sP, fieldsHalos=['Group_M_Crit200','GroupPos'])
        haloMasses = sP.units.codeMassToLogMsun(gc['halos']['Group_M_Crit200'])

        haloIDs = np.where( (haloMasses > 12.02) & (haloMasses < 12.03) )[0]
        haloID = haloIDs[6] # random: 3, 4, 5, 6

        plotParticleMedianVsSecondQuant([sP], partType='gas', xQuant='hdens', yQuant='Si_H_ratio', haloID=haloID, radMinKpc=6.0, radMaxKpc=9.0)

def compareRuns_particleQuant():
    """ Driver. Compare a series of runs in a single panel plot of a particle median quantity vs another. """

    # config
    yQuant = 'tcool'
    xQuant = 'dens_critb'
    ptType = 'gas'
    variants = ['0000'] #,'2102','2202','2302']

    # start PDF, add one page per run
    sPs = []
    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=0.0,variant=variant)
        if sP.simName == 'DM only': continue
        sPs.append(sP)

    plotParticleMedianVsSecondQuant(sPs, partType=ptType, xQuant=xQuant, yQuant=yQuant)
