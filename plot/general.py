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

from util import simParams
from util.helper import loadColorTable, getWhiteBlackColors, running_median, logZeroNaN
from cosmo.load import groupCat, auxCat, snapshotSubset
from cosmo.util import periodicDists
from plot.quantities import quantList, simSubhaloQuantity, simParticleQuantity
from plot.config import *

def plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', weights=['mass'], haloID=None, pdf=None,
                     xMinMaxForce=None, yMinMaxForce=None, contours=None, 
                     massFracMinMax=[-10.0,0.0], hideBelow=False, smoothSigma=0.0):
    """ Plot a 2D phase space plot (arbitrary values on x/y axes), for a single halo or for an entire box 
    (if haloID is None). weights is a list of the gas properties to weight the 2D histogram by, 
    if more than one, a horizontal multi-panel plot will be made with a single colorbar. If 
    x[y]MinMaxForce, use these range limits. If contours is not None, draw solid contours at 
    these levels on top of the 2D histogram image. If smoothSigma is not zero, gaussian smooth 
    contours at this level. If hideBelow, then pixel values below massFracMinMax[0] are left pure white. """

    # config
    nBinsX = 800
    nBinsY = 400
    sizefac = 0.7

    ctNameHisto = 'viridis'
    contoursColor = 'k' # black

    # load: x-axis
    xlabel, xlim, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean)
    xvals = snapshotSubset(sP, partType, xQuant, haloID=haloID)

    if xlog: xvals = np.log10(xvals)

    # load: y-axis
    ylabel, ylim, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean)
    yvals = snapshotSubset(sP, partType, yQuant, haloID=haloID)

    if ylog: yvals = np.log10(yvals)

    # overrides to default ranges?
    if xMinMaxForce is not None: xlim = xMinMaxForce
    if yMinMaxForce is not None: ylim = yMinMaxForce

    # start figure
    fig = plt.figure(figsize=[figsize[0]*sizefac*(len(weights)*0.9), figsize[1]*sizefac])

    # loop over each weight requested
    for i, wtProp in enumerate(weights):
        # load: weights
        weight = snapshotSubset(sP, partType, wtProp, haloID=haloID)

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

        # plot 2D histogram image
        zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBinsX, nBinsY], range=[xlim,ylim], 
                                    normed=True, weights=weight)
        zz = logZeroNaN(zz.T)

        if hideBelow:
            w = np.where(zz < massFracMinMax[0])
            zz[w] = np.nan

        cmap = loadColorTable(ctNameHisto)
        norm = Normalize(vmin=massFracMinMax[0], vmax=massFracMinMax[1], clip=False)
        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

        # plot contours?
        if contours is not None:
            zz, xc, yc = np.histogram2d(xvals, yvals, bins=[nBinsX/4, nBinsY/4], range=[xlim,ylim], 
                                        normed=True, weights=weight)
            XX, YY = np.meshgrid(xc[:-1], yc[:-1], indexing='ij')
            zz = logZeroNaN(zz)

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
    fig.subplots_adjust(right=0.93)
    cbar_ax = fig.add_axes([0.94, 0.131, 0.02, 0.831]) # 0.821
    #cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.ax.set_ylabel('Relative %s [ log ]' % wtStr)
    
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig('phase_%s_z=%.1f_%s_x-%s_y-%s_wt-%s_h-%s.pdf' % \
            (sP.simName,sP.redshift,partType,xQuant,yQuant,"-".join([w.replace(" ","") for w in weights]),haloID))
    plt.close(fig)

def plotParticleMedianVsSecondQuant(sP, partType='gas', xQuant='hdens', yQuant='Si_H_numratio', haloID=0):
    """ Plot the (median) relation between two particle properties for a single halo (if haloID is specified), 
    or the whole box (if haloID is None). """

    # config
    nBins = 50
    lw = 3.0

    radMinKpc = 6.0
    radMaxKpc = 9.0 # physical kpc, or None for none

    # load
    xlabel, xlim, xlog = simParticleQuantity(sP, partType, xQuant, clean=clean, haloLims=(haloID is not None))
    sim_xvals = snapshotSubset(sP, partType, xQuant, haloID=haloID)

    ylabel, ylim, ylog = simParticleQuantity(sP, partType, yQuant, clean=clean, haloLims=(haloID is not None))
    sim_yvals = snapshotSubset(sP, partType, yQuant, haloID=haloID)
    
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

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))
    ax.set_xlabel(xlabel)

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
    ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    # finish plot
    fig.savefig('particleMedian_%s_%s-vs-%s_%s_z=%.1f_%s.pdf' % (partType,xQuant,yQuant,sP.simName,sP.redshift,hStr))
    plt.close(fig)

def plotRadialProfile1D(sPs, subhalo=None, ptType='gas', ptProperty='temp', halo=None):
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
                                          scope=scope, weighting=None, subhaloIDsTodo=subhaloIDs, 
                                          ptRestriction=ptRestriction)
        assert data.shape[0] == len(subhaloIDs)

        nSamples = 1 if not isinstance(subhalo[i],dict) else len(subhalo[i].keys())

        for j in range(nSamples):
            # crossmatch attrs['subhaloIDs'] with subhalo[key] sub-list if needed
            subIDsLoc = subhalo[i][subhalo[i].keys()[j]] if isinstance(subhalo[i],dict) else subhaloIDs
            w, _ = match3( attrs['subhaloIDs'], subIDsLoc )
            assert len(w) == len(subIDsLoc)

            # calculate median radial profile and scatter
            #yy_mean = np.nansum( data[w,:], axis=0 ) / len(w)
            yy_mean = np.nanmedian( data[w,:], axis=0 )
            yp = np.nanpercentile( data[w,:], percs, axis=0 )

            if ylog: yy_mean = logZeroNaN(yy_mean)
            if ylog: yp = logZeroNaN(yp)
            rr = logZeroNaN(attrs['rad_bins_pkpc'])

            if rr.size > sKn:
                yy_mean = savgol_filter(yy_mean,sKn,sKo)
                yp = savgol_filter(yp,sKn,sKo,axis=1) # P[10,90]

            sampleDesc = '' if nSamples == 1 else subhalo[i].keys()[j]
            l, = ax.plot(rr, yy_mean, lw=lw, label='%s %s' % (sP.simName,sampleDesc))
            if len(sPs) == 1 and subhaloIDs.size > 1:
                ax.fill_between(rr, yp[0,:], yp[-1,:], color=l.get_color(), interpolate=True, alpha=0.2)

    # finish plot
    fig.tight_layout()
    ax.legend(loc='best')
    sPstr = sP.simName if len(sPs) == 1 else 'nSp-%d' % len(sPs)
    fig.savefig('radProfile_%s_%s_%s_Ns-%d_Nh-%d_scope=%s.pdf' % \
        (sPstr,ptType,ptProperty,nSamples,len(subhaloIDs),scope))
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
    from plot.oxygen import variantsMain as variants

    sPs = []
    subhalos = []

    for variant in variants:
        sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

        mhalo = groupCat(sPs[-1], fieldsSubhalos=['mhalo_200_log'])
        with np.errstate(invalid='ignore'):
            w = np.where( (mhalo > 11.5) & (mhalo < 12.5) )

        subhalos.append( w[0] )

    for field in ['metaldens']: #,'dens','temp_linear','P_gas_linear','z_solar']:
        plotRadialProfile1D(sPs, subhalo=subhalos, ptType='gas', ptProperty=field)

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

    for field in ['metaldens','dens','temp_linear','P_gas_linear','z_solar']:
        plotRadialProfile1D(sPs, subhalo=subhalos, ptType='gas', ptProperty=field)

def singleHaloProperties():
    """ Driver. Several phase/radial profile plots for a single halo. """
    if 1:
        sP = simParams(res=11,run='zooms2_josh',hInd=2,variant='PO',redshift=2.25)
        haloID = 0

        #plotRadialProfile1D([sP], halo=haloID, ptType='gas', ptProperty='hdens')
        #plotRadialProfile1D([sP], halo=haloID, ptType='gas', ptProperty='temp_linear')
        #plotRadialProfile1D([sP], halo=haloID, ptType='gas', ptProperty='cellsize_kpc')
        plotRadialProfile1D([sP], halo=haloID, ptType='gas', ptProperty='radvel')
        #plotPhaseSpace2D(sP, partType='gas', xQuant='hdens', yQuant='temp_linear', haloID=haloID)

    if 0:
        sP = simParams(res=455,run='tng',redshift=0.0)

        # pick a MW
        gc = groupCat(sP, fieldsHalos=['Group_M_Crit200','GroupPos'])
        haloMasses = sP.units.codeMassToLogMsun(gc['halos']['Group_M_Crit200'])

        haloIDs = np.where( (haloMasses > 12.02) & (haloMasses < 12.03) )[0]
        haloID = haloIDs[6] # random: 3, 4, 5, 6

        plotParticleMedianVsSecondQuant(sP, partType='gas', xQuant='hdens', yQuant='Si_H_ratio', haloID=haloID)