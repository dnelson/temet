"""
general.py
  General exploratory/diagnostic plots of particle-level data of single halos or entire boxes.
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
from util.helper import loadColorTable, getWhiteBlackColors, running_median, logZeroNaN, iterable, evenlySample
from plot.quantities import quantList, simSubhaloQuantity, simParticleQuantity
from plot.config import *

def plotHistogram1D(sPs, ptType='gas', ptProperty='temp_linear', ptWeight=None, subhaloIDs=None, haloIDs=None, 
      ylog=True, ylim=None, xlim=None, qRestrictions=None, nBins=400, medianPDF=False, pdf=None):
    """ Simple 1D histogram/PDF of some quantity ptProperty of ptType, either of the whole box (subhaloIDs and haloIDs 
    both None), or of one or more halos/subhalos, where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    Alternatively, subhaloIDs/haloIDs can be a list of lists, each sub-list of IDs of objects for that run, which are overplotted.
    If ptWeight is None, uniform weighting, otherwise weight by this quantity.
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If ylim is None, then hard-coded typical limits for PDFs. If 'auto', then autoscale. Otherwise, 2-tuple to use as limits. 
    If medianPDF is True, then add this mean (per sP) on top. If meanPDF is 'only', then skip the individual objects. """

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
    fig = plt.figure(figsize=(11.2,8.0)) # (14,10)
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

        yy_save = np.zeros( (nBins,len(sP_objIDs)), dtype='float32' )

        for j, objID in enumerate(sP_objIDs):
            # load
            load_haloID = objID if haloIDs is not None else None
            load_subID = objID if subhaloIDs is not None else None

            vals = sP.snapshotSubset(ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)
            if xlog: vals = np.log10(vals)

            # weights
            if ptWeight is None:
                weights = np.zeros( vals.size, dtype='float32' ) + 1.0
            else:
                weights = sP.snapshotSubset(ptType, ptWeight, haloID=load_haloID, subhaloID=load_subID)

            # arbitrary property restriction(s)?
            if qRestrictions is not None:
                mask = np.zeros( vals.size, dtype='int16' )
                for rFieldName, rFieldMin, rFieldMax in qRestrictions:
                    # load and update mask
                    r_vals = sP.snapshotSubset(ptType, rFieldName, haloID=load_haloID, subhaloID=load_subID)

                    wRestrict = np.where( (r_vals < rFieldMin) | (r_vals > rFieldMax) )
                    mask[wRestrict] = 1
                    print('[%d] restrict [%s] eliminated [%d] of [%d] = %.2f%%' % \
                        (objID,rFieldName,len(wRestrict[0]),mask.size,len(wRestrict[0])/mask.size*100))

                # apply mask
                wRestrict = np.where(mask == 0)
                vals = vals[wRestrict]
                weights = weights[wRestrict]

            #print(sP.simName,', min: ', np.nanmin(vals), ' max: ', np.nanmax(vals), ' median: ', np.nanmedian(vals))
            #print(sP.simName, ', 5,10,90,95 percentiles: ',np.nanpercentile(vals, [5,10,90,95]))

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

            yy_save[:,j] = yy

            # plot
            if xx.size > sKn:
                yy = savgol_filter(yy,sKn,sKo)

            label = '%s [%d]' % (sP.simName,objID) if len(sPs) > 1 else str(objID)
            ls = ':' if sP.simName == 'L11 (Primordial Only)' else '-'
            if medianPDF != 'only':
                l, = ax.plot(xx, yy, linestyle=ls, lw=lw-1, label=label)

        # add mean?
        if len(sP_objIDs) > 1 and medianPDF:
            yy1 = np.nanmean(yy_save, axis=1)
            yy2 = np.nanmedian(yy_save, axis=1)

            if xx.size > sKn:
                yy1 = savgol_filter(yy1,sKn,sKo)
                yy2 = savgol_filter(yy2,sKn,sKo)

            ax.plot(xx, yy1, linestyle='-', color='black', lw=lw+0.5, alpha=0.7, label='mean')
            ax.plot(xx, yy2, linestyle='-', color='black', lw=lw+0.5, alpha=1.0, label='median')

    # finish plot
    fig.tight_layout()
    ax.legend(loc='best', ncol=3)

    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    hStr = 'global'
    if haloIDs is not None: hStr = 'haloIDs-n%d' % len(haloIDs)
    elif subhaloIDs is not None: hStr = 'subhIDs-n%d' % len(subhaloIDs)

    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('histo1D_%s_%s_%s_wt-%s_%s.pdf' % (sPstr,ptType,ptProperty,ptWeight,hStr))
    plt.close(fig)

def plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', weights=['mass'], meancolors=None, haloID=None, 
                     xlim=None, ylim=None, clim=None, contours=None, normColMax=False, hideBelow=False, smoothSigma=0.0, 
                     nBins=None, qRestrictions=None, pdf=None):
    """ Plot a 2D phase space plot (arbitrary values on x/y axes), for a single halo or for an entire box 
    (if haloID is None). weights is a list of the gas properties to weight the 2D histogram by, 
    if more than one, a horizontal multi-panel plot will be made with a single colorbar. Or, if meancolors is 
    not None, then show the mean value per pixel of these quantities, instead of weighted histograms.
    If xlim,ylim,clim specified, then use these bounds, otherwise use default/automatic bounds.
    If contours is not None, draw solid contours at these levels on top of the 2D histogram image. 
    if normColMax, then normalize every column to its maximum (i.e. conditional 2D PDF).
    If hideBelow, then pixel values below clim[0] are left pure white. 
    If smoothSigma is not zero, gaussian smooth contours at this level. 
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by. """

    # config
    nBins2D = None

    if nBins is None:
        # automatic (2d binning set below based on aspect ratio)
        nBins = 400
        if sP.isZoom:
            nBins = 250
    else:
        if isinstance(nBins,(list,np.ndarray)):
            # fully specified
            nBins2D = nBins
        else:
            # one-dim specified (2d binning set below based on aspect ratio)
            nBins = nBins

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

    if meancolors is not None and '_ratio' in meancolors[0]: ctNameHisto = 'RdYlGn_r' # symmetric about clim centerpoint

    # load: x-axis
    xlabel, xlim_quant, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
    if xlim is None: xlim = xlim_quant
    xvals = sP.snapshotSubset(partType, xQuant, haloID=haloID)

    if xlog: xvals = np.log10(xvals)

    # load: y-axis
    ylabel, ylim_quant, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
    if ylim is None: ylim = ylim_quant
    yvals = sP.snapshotSubset(partType, yQuant, haloID=haloID)

    if ylog: yvals = np.log10(yvals)

    # arbitrary property restriction(s)?
    if qRestrictions is not None:
        mask = np.zeros( xvals.size, dtype='int16' )
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # load and update mask
            r_vals = sP.snapshotSubset(partType, rFieldName, haloID=haloID)

            wRestrict = np.where( (r_vals < rFieldMin) | (r_vals > rFieldMax) )
            mask[wRestrict] = 1
            print('[%d] restrict [%s] eliminated [%d] of [%d] = %.2f%%' % \
                (haloID,rFieldName,len(wRestrict[0]),mask.size,len(wRestrict[0])/mask.size*100))

        # apply mask
        wRestrict = np.where(mask == 0)
        xvals = xvals[wRestrict]
        yvals = yvals[wRestrict]

    # start figure
    fig = plt.figure(figsize=[figsize[0]*sizefac*len(weights), figsize[1]*sizefac])

    # loop over each weight requested
    for i, wtProp in enumerate(weights):
        # load: weights
        weight = sP.snapshotSubset(partType, wtProp, haloID=haloID)

        if qRestrictions is not None:
            weight = weight[wRestrict]

        # add panel
        ax = fig.add_subplot(1,len(weights),i+1)

        if len(weights) == 1: # title
            hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
            wtStr = partType.capitalize() + ' ' + wtProp.capitalize()
            ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if nBins2D is None:
            bbox = ax.get_window_extent()
            nBins2D = np.array([nBins, int(nBins*(bbox.height/bbox.width))])

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
            if 1:
                # remove NaN weight points prior to binning (default op is mean, not nanmean)
                w = np.where(np.isfinite(weight))
                xvals = xvals[w]
                yvals = yvals[w]
                weight = weight[w]
                
            clabel, clim_quant, clog = simParticleQuantity(sP, partType, wtProp, clean=clean, haloLims=(haloID is not None))
            wtStr = 'Mean ' + clabel
            zz, _, _, _ = binned_statistic_2d(xvals, yvals, weight, 'mean', # median unfortunately too slow
                                                         bins=nBins2D, range=[xlim,ylim])
            zz = zz.T
            if clog: zz = logZeroNaN(zz)

            if clim is None:
                clim = clim_quant # colorbar limits
        else:
            # plot 2D histogram image, optionally weighted
            zz, _, _ = np.histogram2d(xvals, yvals, bins=nBins2D, range=[xlim,ylim], 
                                      normed=True, weights=weight)
            zz = zz.T

            if normColMax:
                colMax = np.nanmax(zz, axis=0)
                w = np.where(colMax == 0)
                colMax[w] = 1.0 # entire column is zero, will be log->nan anyways then not shown
                zz /= colMax[np.newaxis,:]

            zz = logZeroNaN(zz)

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
                                                    bins=[nBins2D[0]/4, nBins2D[1]/4], range=[xlim,ylim])
                if clog: zz = logZeroNaN(zz)
            else:
                zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBins2D[0]/4, nBins2D[1]/4], range=[xlim,ylim], 
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

        if 0:
            # debugging EOS correction
            xx = np.linspace(-6.0, -4.0, 10)
            yy = (0.45*(xx+5.0)+3.45)
            ax.plot(xx,yy,'o-',color='black')
            ax.plot([-4,-1],[3.9,3.9],'-',color='#444444')

        # mark virial radius?
        if haloID is not None and xQuant in ['rad','rad_kpc','rad_kpc_linear']:
            textOpts = {'rotation':90.0, 'horizontalalignment':'left', 'verticalalignment':'top', 'fontsize':18, 'color':'#cccccc'}
            rvir = sP.groupCatSingle(haloID=haloID)['Group_R_Crit200']
            if '_kpc' in xQuant: rvir = sP.units.codeLengthToKpc(rvir)

            for fac in [1,2,4,10,100]:
                xx = rvir/fac if '_linear' in xQuant else np.log10(rvir/fac)
                yy = [ax.get_ylim()[1]*0.80, ax.get_ylim()[1]*0.98]
                xoff = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.01
                if xx >= ax.get_xlim()[1]: continue

                ax.plot( [xx,xx], yy, '-', lw=lw, color=textOpts['color'])
                ax.text( xx + xoff, yy[1], '$r_{\\rm vir}$/%d'%fac if fac != 1 else '$r_{\\rm vir}$', **textOpts)

        # special behaviors
        if xQuant in ['rad_kpc','rad_kpc_linear'] and yQuant == 'vrad':
            if 0:
                # escape velocity curve, direct from enclosed mass profile                
                ptTypes = ['stars','gas','dm']
                haloLen = sP.groupCatSingle(haloID=haloID)['GroupLenType']
                totSize = np.sum( [haloLen[sP.ptNum(ptType)] for ptType in ptTypes] )

                offset = 0
                mass = np.zeros( totSize, dtype='float32' )
                rad  = np.zeros( totSize, dtype='float32' )

                for ptType in ptTypes:
                    mass[offset:offset+haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, 'mass', haloID=haloID)
                    rad[offset:offset+haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, xQuant, haloID=haloID)
                    offset += haloLen[sP.ptNum(ptType)]

                sort_inds = np.argsort(rad)
                mass = mass[sort_inds]
                rad = rad[sort_inds]
                cum_mass = np.cumsum(mass)

                # sample down to N radial points
                rad_code = evenlySample(rad[1:], 100)
                tot_mass_enc = evenlySample(cum_mass[1:], 100)

                if '_kpc' in xQuant: rad_code = sP.units.physicalKpcToCodeLength(rad_code) # pkpc -> code
                vesc = np.sqrt(2 * sP.units.G * tot_mass_enc / rad_code) # code velocity units = [km/s]
            if 1:
                # escape velocity curve, directly from potential of particles
                vesc = sP.snapshotSubset('dm', 'vesc', haloID=haloID)
                rad = sP.snapshotSubset('dm', xQuant, haloID=haloID)

                sort_inds = np.argsort(rad)
                vesc = vesc[sort_inds]
                rad = rad[sort_inds]

                rad_code = evenlySample(rad, 100, logSpace=True)
                vesc = evenlySample(vesc, 100, logSpace=True)

            if xlog: rad_code = np.log10(rad_code)
            ax.plot(rad_code[1:], vesc[1:], '-', lw=lw, color='#000000', alpha=0.5)
            ax.text(rad_code[-17], vesc[-17]*1.02, '$v_{\\rm esc}(r)$', color='#000000', alpha=0.5, fontsize=18.0, va='bottom', rotation=-4.0)

        if xQuant == 'density' and yQuant == 'temp':
            # add Torrey+12 'ISM cut' line
            xx = np.array(ax.get_xlim())
            yy = 6.0 + 0.25*xx
            ax.plot(xx, yy, '-', lw=lw, color='#000000', alpha=0.7, label='Torrey+12 ISM cut')

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
        fig.savefig('phase2d_%s_%d_%s_x-%s_y-%s_wt-%s_h-%s.pdf' % \
            (sP.simName,sP.snap,partType,xQuant,yQuant,"-".join([w.replace(" ","") for w in weights]),haloID))
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
    fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    sStr = '%s z=%.1f' % (sPs[0].simName, sPs[0].redshift) if len(sPs) == 1 else ''
    ax.set_title('%s %s' % (sStr,hStr))

    for i, sP in enumerate(sPs):
        # load
        xlabel, xlim, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
        sim_xvals = sP.snapshotSubset(partType, xQuant, haloID=haloID)
        if xlog: sim_xvals = logZeroNaN(sim_xvals)

        ylabel, ylim, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
        sim_yvals = sP.snapshotSubset(partType, yQuant, haloID=haloID)
        if ylog: sim_yvals = logZeroNaN(sim_yvals)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # radial restriction
        if radMaxKpc is not None or radMinKpc is not None:
            assert haloID is not None
            rad = sP.snapshotSubset(partType, 'rad_kpc', haloID=haloID)
            
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

        c = next(ax._get_lines.prop_cycler)['color']
        ax.plot(xm, ym2, linestyles[0], lw=lw, color=c, label=sP.simName)

        # percentile:
        if len(sPs) <= 3 or (len(sPs) > 3 and i == 0):
            ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    ax.legend(loc='best')
    fig.tight_layout()

    # finish plot
    sStr = '%s_z-%.1f' % (sPs[0].simName,sPs[0].redshift) if len(sPs) == 1 else 'sPn%d' % len(sPs)
    fig.savefig('particleMedian_%s_%s-vs-%s_%s_%s.pdf' % (partType,xQuant,yQuant,sStr,hStr))
    plt.close(fig)

def plotStackedRadialProfiles1D(sPs, subhalo=None, ptType='gas', ptProperty='temp_linear', op='mean', weighting=None, halo=None):
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
    #op = 'mean' # mean, sum, min, max

    assert subhalo is not None or halo is not None # pick one
    if subhalo is None: subhalo = halo # use halo ids
    if isinstance(subhalo,int) and len(sPs) == 1: subhalo = [subhalo] # single number to list (one sP case)
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
        if isinstance(subhaloIDs,int):
            subhaloIDs = [subhaloIDs]
        if isinstance(subhaloIDs,dict):
            subhaloIDs = np.hstack( [subhaloIDs[key] for key in subhaloIDs.keys()])

        if halo is not None:
            # transform fof ids to subhalo ids
            firstsub = sP.groupCat(fieldsHalos=['GroupFirstSub'])
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
    xlog=True, xlim=None, ylog=None, ylim=None, sfreq0=False, colorOffs=None, scope='fof'):
    """ Radial profile of some quantity ptProperty of ptType vs. radius from halo center,
    where subhaloIDs (or haloIDs) is an ID list with one entry per sPs entry. 
    If haloIDs is not None, then use these FoF IDs as inputs instead of Subfind IDs. 
    Scope can be: global, fof, subfind. """

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

    assert np.sum(e is not None for e in [haloIDs,subhaloIDs]) == 1 # pick one
    if subhaloIDs is not None: assert (len(subhaloIDs) == len(sPs)) # one subhalo ID per sP
    if haloIDs is not None: assert (len(haloIDs) == len(sPs)) # one subhalo ID per sP

    ylabel, ylim_q, ylog_q = simParticleQuantity(sPs[0], ptType, ptProperty, clean=clean, haloLims=True)
    if ylim is None: ylim = ylim_q
    if ylog is None: ylog = ylog_q

    # start plot
    fig = plt.figure(figsize=(11.2,8.0)) #(11.2,8.0) #(14,10)
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
            haloID = sP.groupCatSingle(subhaloID=subhaloID)['SubhaloGrNr']
        else:
            haloID = haloIDs[i]
            subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

        # load
        load_haloID = haloID if scope == 'fof' else None
        load_subID = subhaloID if scope == 'subfind' else None
        if load_haloID is None and load_subID is None: assert scope == 'global'

        rad  = sP.snapshotSubset(ptType, 'rad_kpc', haloID=load_haloID, subhaloID=load_subID)
        vals = sP.snapshotSubset(ptType, ptProperty, haloID=load_haloID, subhaloID=load_subID)

        if sfreq0:
            # restrict to non eEOS cells
            sfr = sP.snapshotSubset(ptType, 'sfr', haloID=load_haloID, subhaloID=load_subID)
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
                _ = next(ax._get_lines.prop_cycler)['color']

        label = '%s haloID=%d [%s]' % (sP.simName,haloID,scope) if not clean else sP.simName
        l, = ax.plot(rr, yy_median, '-', lw=lw, label=label)
        ax.plot(rr, yy_mean, '--', lw=lw, color=l.get_color())

        if len(sPs) <= 2:
            for j in range(yy_perc.shape[0]/2):
                ax.fill_between(rr, yy_perc[0+j,:], yy_perc[-(j+1),:], color=l.get_color(), interpolate=True, alpha=0.15*(j+1))

    # special behavior
    if 'L11_12' in sPs[0].simName:
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
    redshift = 0.0
    yQuant = 'temp'
    xQuant = 'numdens'

    # get list of all 512 method runs via filesystem search
    sP = simParams(res=512,run='tng',redshift=redshift,variant='0000')
    dirs = glob.glob(sP.arepoPath + '../L25n512_*')
    variants = sorted([d.rsplit("_",1)[1] for d in dirs])
    variants = ['0000','1006']

    # start PDF, add one page per run
    pdf = PdfPages('compareRunsPhaseDiagram.pdf')

    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=redshift,variant=variant)
        if sP.simName == 'DM only': continue
        print(variant,sP.simName)
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, haloID=None, pdf=pdf)
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, meancolors=['coolrate_ratio'], weights=None, haloID=None, pdf=pdf)

    pdf.close()

def compareVariants_NO_OH_stellar():
    """ Driver. Igor. """
    import glob
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    partType = 'stars'
    yQuant = 'N_O_massratio'
    ylim   = [-3.5,0.0]
    xQuant = 'O_H_massratio'
    xlim   = [-5.5,-0.5]
    redshift = 0.0

    # variants
    sP = simParams(res=512,run='tng',redshift=redshift,variant='0000')
    dirs = glob.glob(sP.arepoPath + '../L25n512_*')
    variants = sorted([d.rsplit("_",1)[1] for d in dirs])

    # start PDF, add one page per run
    pdf = PdfPages('compareRuns_x=%s_y=%s_%s.pdf' % (xQuant,yQuant,partType))

    for variant in variants:
        sP = simParams(res=512,run='tng',redshift=redshift,variant=variant)
        if sP.simName in ['L25n512_0020','L25n512_0030']: continue
        print(variant,sP.simName)
        plotPhaseSpace2D(sP, partType=partType, xQuant=xQuant, yQuant=yQuant, xlim=xlim, ylim=ylim, haloID=None, pdf=pdf)

    pdf.close()

def oneRun_PhaseDiagram(snaps=None):
    """ Driver. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    sP = simParams(res=1820,run='tng')
    yQuant = 'temp'
    xQuant = 'nh'

    zoom = False

    if zoom:
        xlim = [-6.0, -0.5]
        ylim = [2.0, 4.5]
    else:
        xlim = [-9.0, 2.0]
        ylim = [2.0, 8.5]
    clim   = [-6.0,-0.2]
    
    if snaps is None:
        #snaps = sP.validSnapList()[::2] # [99]
        snaps = [0,10,17,18,19,20,30,40,50,60,70,80,90,99]

    # start PDF, add one page per snapshot
    for snap in snaps:
        sP.setSnap(snap)
        print(snap)

        pdf = PdfPages('phaseDiagram_%s_%s%s_%d.pdf' % (yQuant,sP.simName,'_zoom' if zoom else '',snap))

        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, #meancolors=[cQuant], weights=weights, 
            xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, haloID=None, pdf=pdf)

        pdf.close()

def compareRuns_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two runs. """
    #from projects.oxygen import variantsMain as variants
    variants = ['0000','0010']

    sPs = []
    subhalos = []

    for variant in variants:
        sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

        mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w = np.where( (mhalo > 11.5) & (mhalo < 12.5) )

        subhalos.append( w[0] )

    for field in ['temp_linear']: #,'dens','temp_linear','P_gas_linear','z_solar']:
        plotStackedRadialProfiles1D(sPs, subhalo=subhalos, ptType='gas', ptProperty=field, weighting='O VI mass')

def compareHaloSets_RadProfiles():
    """ Driver. Compare median radial profile of a quantity, differentiating between two different 
    types of halos. One run. """
    sPs = []
    sPs.append( simParams(res=1820,run='tng',redshift=0.0) )

    mhalo = sPs[0].groupCat(fieldsSubhalos=['mhalo_200_log'])
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
    mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
    with np.errstate(invalid='ignore'):
        w1 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
    subhaloIDs = [w1[0][0:5]]

    if 0:
        # add a second run
        sPs.append( simParams(res=455,run='tng',redshift=0.0) )
        mhalo = sPs[-1].groupCat(fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w2 = np.where( (mhalo > 11.8) & (mhalo < 12.2)  )
        subhaloIDs.append( w2[0][0:5] )

    for field in ['temp']: #['tcool','vrad']:
        plotHistogram1D(sPs, subhaloIDs=subhaloIDs, ptType='gas', ptProperty=field)

def singleHaloProperties():
    """ Driver. Several phase/radial profile plots for a single halo. """
    sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')

    partType = 'gas'
    xQuant = 'coolrate'
    yQuant = 'coolrate_ratio'

    # pick a MW
    #gc = sP.groupCat(fieldsHalos=['Group_M_Crit200','GroupPos'])
    #haloMasses = sP.units.codeMassToLogMsun(gc['Group_M_Crit200'])

    #haloIDs = np.where( (haloMasses > 12.02) & (haloMasses < 12.03) )[0]
    #haloID = haloIDs[6] # random: 3, 4, 5, 6

    haloID = None
    rMin = None
    rMax = None

    plotParticleMedianVsSecondQuant([sP], partType=partType, xQuant=xQuant, yQuant=yQuant, haloID=haloID, 
                                   radMinKpc=rMin, radMaxKpc=rMax)

    #for prop in ['hdens','temp_linear','cellsize_kpc','radvel','temp']:
    #    plotStackedRadialProfiles1D([sP], halo=haloID, ptType='gas', ptProperty=prop)
    #    plotPhaseSpace2D(sP, partType='gas', xQuant='hdens', yQuant=prop, haloID=haloID)

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

def coolingPhase():
    """ Driver. """
    from matplotlib.backends.backend_pdf import PdfPages

    # config
    yQuant = 'temp'
    xQuant = 'numdens'
    cQuants = ['coolrate','heatrate','coolrate_powell','coolrate_ratio']
    xlim   = [-9.0, 2.0]
    ylim   = [2.0, 8.0]

    #sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')
    sP = simParams(res=1820,run='tng',redshift=0.0)

    # start PDF, add one page per run
    pdf = PdfPages('phaseDiagram_B_%s_%d.pdf' % (sP.simName,sP.snap))

    for cQuant in cQuants:
        plotPhaseSpace2D(sP, xQuant=xQuant, yQuant=yQuant, meancolors=[cQuant], xlim=xlim, ylim=ylim, 
                         weights=None, hideBelow=False, haloID=None, pdf=pdf)

    pdf.close()
