"""
projects/outflows_vis.py
  Plots: Outflows paper (TNG50 presentation), related vis and movies.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import multiprocessing as mp
from functools import partial

from util import simParams
from util.helper import loadColorTable, logZeroNaN, closest, getWhiteBlackColors
from plot.config import *
from cosmo.mergertree import mpbPositionComplete
from vis.common import gridBox, setAxisColors, setColorbarColors
from vis.halo import renderSingleHalo, selectHalosFromMassBin
from vis.box import renderBox
from projects.outflows_analysis import halo_selection, selection_subbox_overlap, haloTimeEvoDataSubbox, haloTimeEvoDataFullbox

def galaxyMosaics(conf=1, rotation='face-on'):
    """ Mosaic, top N most massive. 
      todo: multi-redshift
      todo: pick systems by hand instead of top N
      todo: more than 1 mass bin?"""
    res        = 2160
    redshift   = 4.0
    run        = 'tng'
    rVirFracs  = None
    method     = 'sphMap' #'histo'
    nPixels    = [960,960]
    sizeType   = 'codeUnits'
    axes       = [0,1]
    #rotation   = 'face-on'
    labelHalo  = 'Mstar'

    numGals    = 12
    massBin    = None # if None, then top N
    #iterNum    = 0

    # subhaloIDs of twelve z=2 systems, 3 per 'mass bin', to show as a 3x4 mosaic
    z2_inds = []
    z1_evo_inds = []

    class plotConfig:
        plotStyle = 'edged'
        rasterPx  = 960
        nRows     = numGals/3 # 4x3 panels
        colorbars = True

    # combined plot of centrals in mass bins
    sP = simParams(res=res, run=run, redshift=redshift)

    if massBin is None:
        cen_flag = sP.groupCat(fieldsSubhalos=['central_flag'])
        cen_inds = np.where(cen_flag)[0]
        hIDs = cen_inds[0:0+numGals]
        #hIDs = [0, 25239, 41129, 55632, 70415, 79612, 87125, 94859, 101009, 106001, 110862, 114772]
    else:
        hIDs, _ = selectHalosFromMassBin(sP, [massBin], numPerBin=np.inf, massBinInd=0)
        import pdb; pdb.set_trace()

    # configure panels
    panels = []

    for i, hID in enumerate(hIDs):
        # set semi-adaptive size (code units)
        loc_size = 80.0

        # either stars or gas, face-on
        if conf == 1:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'sfr_msunyrkpc2', 'valMinMax':[-3.0,1.0]} )
        if conf == 2: 
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[7.0,8.5]} )
        if conf == 3:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-0.5,0.5]} )
        if conf == 4:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'vel_los_sfrwt', 'valMinMax':[-300,300]} )
            if rotation is None or rotation == 'edge-on': panels[-1]['valMinMax'] = [-400,400]
        if conf == 5:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'velsigma_los_sfrwt', 'valMinMax':[0,400]} )
        if conf == 6:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'velsigma_los', 'valMinMax':[0,400]} )
        if conf == 7:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'HI_segmented', 'valMinMax':[13.5,21.5]} )

        if i == 0: # upper left
            panels[-1]['labelScale'] = 'physical'
        if i == 2: # upper right
            panels[-1]['labelZ'] = True

    hStr = '_HISTO' if method == 'histo' else ''
    plotConfig.saveFilename = 'renderHalos_%s-%d_%s_%s%s.pdf' % (sP.simName,sP.snap,panels[0]['partField'],rotation,hStr)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def galaxyMosaic_topN(numHalosInd,panelNum=1):
    """ Mosaic, top N most massive. """
    res        = 2160
    redshift   = 2.0 # 1.0
    run        = 'tng'
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [960,960]
    sizeType   = 'codeUnits'
    axes       = [0,1]
    rotation   = 'face-on'

    class plotConfig:
        plotStyle = 'edged'
        rasterPx  = 240
        nRows     = 1 # overriden below
        colorbars = True

    # configure panels
    panels = []

    starsMM = [6.5,10.0] # coldens_msunkpc2
    gasMM   = [7.0,8.5]

    # combined plot of centrals in mass bins
    sP = simParams(res=res, run=run, redshift=redshift)

    mhalo = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])
    cen_inds = np.where(np.isfinite(mhalo))[0]

    if numHalosInd == 0:
        hIDs = cen_inds[0:0+8] # eight total
        plotConfig.nRows = 4 # 4x2
    if numHalosInd == 1:
        hIDs = cen_inds[8:8+40] # 40 total, skipping the first eight
        plotConfig.nRows = 8 # 8x5
    if numHalosInd == 2:
        hIDs = cen_inds[0:29*26] # 754 total, starting again at the first
        plotConfig.nRows = 29 # 29x26 = aspect ratio about half of 16:9

    for hID in hIDs:
        # set semi-adaptive size (code units)
        loc_size = 80.0
        if mhalo[hID] <= 12.5:
            loc_size = 60.0
        if mhalo[hID] <= 12.1:
            loc_size = 40.0

        # either stars or gas, face-on
        if panelNum == 1:
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'stars', 'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'} )
        if panelNum == 2: 
            panels.append( {'hInd':hID, 'size':loc_size, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )

    plotConfig.saveFilename = savePathDefault + 'renderHalos_%s-%d_n%d_%s.pdf' % \
                              (sP.simName,sP.snap,plotConfig.nRows,panels[0]['partType'])

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def _renderSingleImage(snap, sP, subhaloPos):
    """ Multipricessing pool target. """
    # constant config
    method     = 'sphMap'
    axes       = [0,1]
    nPixels    = [600,600]
    hsmlFac    = 2.5
    rotMatrix  = None
    rotCenter  = None
    projType   = 'ortho'
    projParams = {}
    boxSizes   = [20.0,200.0,2000.0] # code units
    partType   = 'gas'
    partField  = 'coldens_msunkpc2'

    # snapshot parallel for subboxes, so set snap now, while for fullboxes we are looping this with a pre-caching
    if sP.snap != snap:
        sP.setSnap(snap)

    # configure render at this snapshot
    subPos = subhaloPos[snap,:]
    boxCenter = subPos[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering

    # loop over more than one boxsize if desired
    for boxSize in boxSizes:
        boxSizeImg = boxSize * np.array([1.0, 1.0, 1.0]) # same width, height, and depth

        # call gridBox
        gridBox(sP, method, partType, partField, nPixels, axes, projType, projParams,
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

def preRenderSubboxImages(sP, sbNum, selInd, minM200=11.5):
    """ Pre-render a number of images for selInd in sP(sbNum), through all snapshots. """
    sel = halo_selection(sP, minM200=minM200)
    _, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel)

    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    # loop over all snapshots of relevance
    snaps = range(minSBsnap.min(),maxSBsnap.max()+1)
    #snaps = range(2100,maxSBsnap.max()-1)
    #print('REMOVE THIS OVERRIDE')

    # thread parallelize by snapshot
    nThreads = 1
    pool = mp.Pool(processes=nThreads)
    func = partial(_renderSingleImage, sP=sP_sub, subhaloPos=subhaloPos[selInd,:,:])

    if nThreads > 1:
        pool.map(func, snaps)
    else:
        for snap in snaps:
            func(snap)

def preRenderFullboxImages(sP, haloInds, snaps=None):
    """ Pre-render a number of images for haloInds at sP, through all snapshots or some 
    subset as specified. """
    posSets = []
    partType = 'gas'

    for haloInd in haloInds:
        halo = sP.groupCatSingle(haloID=haloInd)
        snaps_loc, snapTimes, haloPos = mpbPositionComplete(sP, halo['GroupFirstSub'])
        posSets.append(haloPos)

    if snaps is None: snaps = snaps_loc

    # snapshot loop
    for snap in snaps:
        sP.setSnap(snap)

        # global pre-cache of selected fields into memory
        for field in ['Coordinates','Masses']:
            cache_key = 'snap%d_%s_%s' % (sP.snap,partType,field)
            print('[%s] Caching [%s] now...' % (snap,field))
            sP.data[cache_key] = sP.snapshotSubset(partType, field)
        print('All caching done.')

        # render in serial loop, using pre-cached particle level data
        for pos, ind in zip(posSets,haloInds):
            _renderSingleImage(snap, sP, pos)

def visHaloTimeEvo(sP, data, haloPos, snapTimes, haloInd, extended=False, pStyle='white'):
    """ Visualize halo evolution data. 3x2 panel image sequence, or 5x3 if extended == True. """

    def _histo2d_helper(gs,i,pt,xaxis,yaxis,color,clim):
        """ Add one panel of a 2D phase diagram. """
        ax = plt.subplot(gs[i], facecolor=color1)
        setAxisColors(ax, color2)

        xlim   = data['limits'][xaxis]
        ylim   = data['limits'][yaxis]
        xlabel = labels[xaxis]
        ylabel = labels[yaxis]
        clabel = labels[color]

        colNorm = False
        if color == 'massfracnorm':
            colNorm = True
            color = 'massfrac'

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # plot 2d histogram
        key = 'histo2d_%s_%s_%s' % (xaxis,yaxis,color)
        zz = data[pt][key][snap,:,:]

        if colNorm:
            zz = 10.0**zz
            for i in range(zz.shape[1]):
                if np.count_nonzero(np.isfinite(zz[:,i])) == 0: continue
                colMax = np.nanmax(zz[:,i])
                if np.isfinite(colMax): zz[:,i] /= colMax
            zz = logZeroNaN(zz)

        # render
        norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False)
        im = plt.imshow(zz, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], 
                   cmap=cmap, norm=norm, origin='lower', interpolation='nearest', aspect='auto')

        # colorbar
        cbar_ax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(im, cax=cbar_ax)
        cb.ax.set_ylabel(clabel)
        setColorbarColors(cb, color2)

        # legend (lower right)
        y0 = 0.04
        yd = 0.04

        x0 = 0.85 if extended else 0.89
        if xaxis in ['rad','radlog'] and yaxis == 'vrel':
            # lower right is occupied, move to upper right
            y0 = 1 - y0
            yd = -yd

        legend_labels = ['snap = %6d' % snap, 'zred  = %6.3f' % redshifts[snap], 't/gyr = %6.3f' % tage[snap]]
        textOpts = {'fontsize':fontsizeTime, 'color':color2, 
                    'horizontalalignment':'center', 'verticalalignment':'center'}

        for j, label in enumerate(legend_labels):
            ax.text(x0, y0+yd*j, label, transform=ax.transAxes, **textOpts)

        return ax

    def _histo1d_helper(gs,i,pt,xaxis,yaxis,ylim):
        """ Add one panel of a 1D profile, all the available apertures by default. """
        ax = plt.subplot(gs[i], facecolor=color1)
        setAxisColors(ax, color2)
        
        key    = 'histo1d_%s_%s' % (xaxis,yaxis)
        xlim   = data['limits'][xaxis]
        xlabel = labels[xaxis]
        ylabel = labels[yaxis]

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # reconstruct bin midpoints
        histoNbins = data[pt][key].shape[2]
        xx = np.linspace(xlim[0], xlim[1], histoNbins+1)[:-1] + (xlim[1]-xlim[0])/histoNbins/2

        # plot
        for j, aperture in enumerate(data['apertures']['histo1d']):
            if xaxis in ['rad','radlog'] and j < len(data['apertures']['histo1d'])-1:
                continue # no need for radially restricted radial profiles

            yy = data[pt][key][snap,j,:]
            if yaxis == 'count': yy = logZeroNaN(yy)

            label = 'r < %04d ckpc/h' % aperture if xaxis not in ['rad','radlog'] else ''
            ax.plot(xx, yy, '-', lw=lw, alpha=1.0, label=label)

        # for radial profiles, overplot a few at selected moments in time as they are passed
        if xaxis in ['rad','radlog']:
            cur_redshift = redshifts[snap]
            freeze_redshifts = [6.0, 2.0, 1.0]
            aperture_ind = len(data['apertures']['histo1d']) - 1 # full profile

            for z in freeze_redshifts:
                if cur_redshift <= z:
                    _, z_ind = closest(redshifts, z)
                    yy = data[pt][key][z_ind,aperture_ind,:]
                    ax.plot(xx, yy, '-', lw=lw, alpha=0.3, label='z = %d' % z)

        # legend
        l = ax.legend(loc='upper right', prop={'size':fontsizeLegend})
        if l is not None:
            for text in l.get_texts(): text.set_color(color2)
        return ax

    def _image_helper(gs, i, pt, field, axes, boxSize):
        """ Add one panel of a gridBox() rendered image projection. """
        ax = plt.subplot(gs[i], facecolor=color1)
        setAxisColors(ax, color2)

        # render config
        method     = 'sphMap'
        nPixels    = [600,600]
        hsmlFac    = 2.5
        rotMatrix  = None
        rotCenter  = None
        projType   = 'ortho'
        projParams = {}
        boxSizeImg = np.array([boxSize,boxSize,boxSize])
        boxCenter  = haloPos[snap,:]

        # load/render
        sPr = simParams(res=sP.res, run=sP.run, variant=sP.variant, snap=snap)

        boxCenter  = boxCenter[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering
        grid, config = gridBox(sPr, method, pt, field, nPixels, axes, projType, projParams, 
                               boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter)

        # plot config
        valMinMax = [5.5, 8.0] # todo generalize
        if boxSize >= 500.0: valMinMax = [4.5, 8.0]
        if boxSize <= 50.0:
            valMinMax = [5.2,8.2]
            if 'sb' in haloInd: valMinMax = [5.7, 9.2]
        cmap = loadColorTable(config['ctName'], valMinMax=valMinMax)

        extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
                   boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]
        extent[0:2] -= boxCenter[0] # make coordinates relative
        extent[2:4] -= boxCenter[1]

        ax.set_xlabel( ['x','y','z'][axes[0]] + ' [ckpc/h]')
        ax.set_ylabel( ['x','y','z'][axes[1]] + ' [ckpc/h]')

        # plot
        plt.imshow(grid, extent=extent, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])
        ax.autoscale(False)
        plt.clim(valMinMax)

        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.15)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel(config['label'])
        setColorbarColors(cb, color2)

    def _lineplot_helper(gs,i,conf=1):
        """ Add one panel of a composite line plot of various component masses and BH energies. """
        ax = plt.subplot(gs[i], facecolor=color1)
        setAxisColors(ax, color2)
        ax.set_xlabel('Redshift')

        if conf == 1:
            # various mass components on y-axis 1, BH energetics on y-axis 2
            ax.set_ylabel('Masses [log M$_{\\rm sun}$]')
            ax.set_ylim([5.0, 8.0])
            yy = sP.units.codeMassToLogMsun( data['bhs']['BH_Mass'][:,bh_apInd] )
            if np.nanmax(yy) > 8.1:
                ax.set_ylim([5.0, 8.5]) # massive halo
            if np.nanmax(yy) > 8.9:
                ax.set_ylim([5.0, 9.5]) # massive halo
            yy_label = 'BH Mass'
        if conf == 2:
            # BH mdot and SFR on y-axis 1
            ax.set_ylabel('SFR or $\dot{M}_{\\rm BH}$ [log M$_{\\rm sun}$ / yr]')
            yy = bh_mdot
            yy_label = 'BH Mdot'

        # y-axis 1: first property
        l, = ax.plot(redshifts[w], yy[w], '-', lw=lw, alpha=0.5, label=yy_label)
        ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=l.get_color())
        ax.set_xlim(ax.get_xlim()[::-1]) # time increasing to the right
        for t in ax.get_yticklabels(): t.set_color(l.get_color())

        # y-axis 1: additional properties
        if conf == 1:
            for key in ['gas','stars']:
                c = ax._get_lines.prop_cycler.next()['color']
                for j, aperture in enumerate(data['apertures']['scalar']):
                    yy = sP.units.codeMassToLogMsun( data[key]['mass'][:,j])
                    yy -= 3.0 # 3 dex offset to get within range of M_BH
                    label = key.capitalize() + 'Mass/$10^3$' if j == 0 else ''
                    l, = ax.plot(redshifts[w], yy[w], linestyles[j], lw=lw, alpha=0.5, label=label, color=c)
                    ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=c)
        if conf == 2:
            c = ax._get_lines.prop_cycler.next()['color']
            for j, aperture in enumerate(data['apertures']['scalar']):
                yy = logZeroNaN( data['gas']['StarFormationRate'][:,j] )
                label = 'Gas_SFR' if j == 0 else ''
                l, = ax.plot(redshifts[w], yy[w], linestyles[j], lw=lw, alpha=0.5, label=label, color=c)
                ax.plot(redshifts[snap], yy[snap], 'o', markersize=14.0, alpha=0.7, color=c)

        # y-axis 2
        if conf == 1:
            # blackhole energetics
            ax2 = ax.twinx()
            setAxisColors(ax2, color2)
            ax2.set_ylabel('BH $\Delta$ E$_{\\rm low}$ (dotted), E$_{\\rm high}$ (solid) [ log erg ]')
            c = ax._get_lines.prop_cycler.next()['color']
            ax2.plot(redshifts[w], dy_high[w], '-', lw=lw, alpha=0.7, color=c)
            ax2.plot(redshifts[w], dy_low[w], ':', lw=lw, alpha=0.7, color=c)
            ax2.plot(redshifts[snap], dy_high[snap], 'o', markersize=14.0, color=c, alpha=0.6)
            ax2.plot(redshifts[snap], dy_low[snap], 'o', markersize=14.0, color=c, alpha=0.6)
            for t in ax2.get_yticklabels(): t.set_color(c)

            ax2.set_ylim([54,60])
            if np.nanmax(dy_high[w]) > 60.1:
                ax2.set_ylim([54,61]) # massive halo

        # legend
        handles, labels = ax.get_legend_handles_labels()
        sExtra = [plt.Line2D((0,1), (0,0), color=color2, marker='', lw=lw, linestyle=ls) for ls in linestyles]
        apertures = ['%dckpc/h' % aperture for aperture in data['apertures']['scalar']]
        l = ax.legend(handles+sExtra, labels+apertures, loc='lower right', prop={'size':fontsizeLegend})
        for text in l.get_texts(): text.set_color(color2)

    # config
    fontsizeTime   = 22.0
    fontsizeLegend = 16.0
    cmap = loadColorTable('viridis')

    labels = {'radlog'       : 'Radius [log ckpc/h]',
              'rad'          : 'Radius [ckpc/h]',
              'vrad'         : 'Radial Velocity [km/s]',
              'vrel'         : 'Halo-Rel Velocity Mag [km/s]',
              'numdens'      : 'Gas Density [log cm$^{-3}$]',
              'temp'         : 'Gas Temperature [log K]',
              'templinear'   : 'Gas Temperature [log K]',
              'massfrac'     : 'Relative Mass Fraction [log]',
              'massfracnorm' : 'Conditional Mass Fraction [log]',
              'count'        : 'PDF'}

    lim1 = [-4.0, -1.5] # massfrac
    lim2 = [-2.5, 0.0] # massfracnorm
    lim3 = [-5.0, -1.5] # histo1d
    lim4 = [4.0, 6.5] # temp
    lim5 = [-100, 300] # vrad
    lim6 = [-6.0, 1.0] # numdens profile
    lim7 = [3.5, 6.0] # temp profile

    if 'sb2' in haloInd:
        # massive
        lim3 = [-6.0, -2.0]
        lim4 = [4.0, 7.5]
        lim5 = [-100, 500]
        lim6 = [-5.0, 2.0]
        lim7 = [3.5, 7.5]

    linestyles = ['-','--',':'] # for the three apertures

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    # derive blackhole differential energetics (needed for _lineplot_helpers)
    redshifts = 1.0 / snapTimes - 1.0
    tage = sP.units.redshiftToAgeFlat(redshifts)
    dtyr = np.diff(tage * 1e9, n=1)
    dtyr = np.append(dtyr, dtyr[-1])

    bh_apInd = 0 # aperture to use for BH derived lines (all going to be the same)
    yy_high = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_QM'][:,bh_apInd], log=True)
    yy_low  = sP.units.codeEnergyToErg(data['bhs']['BH_CumEgyInjection_RM'][:,bh_apInd], log=True)
    dy_high = np.diff(10.0**yy_high, n=1) # dy[i] is delta_E between snap=i and snap=i+1
    dy_low  = np.diff(10.0**yy_low, n=1)
    dy_high = logZeroNaN( np.append(dy_high, dy_high[-1]) ) # restore to original size, then log
    dy_low  = logZeroNaN( np.append(dy_low, dy_low[-1]) )
    bh_mdot = np.diff(sP.units.codeMassToMsun(data['bhs']['BH_Mass'][:,bh_apInd]) / dtyr[::-1], n=1)
    bh_mdot = logZeroNaN( np.append(bh_mdot, bh_mdot[-1]) )
    wmax = np.where(bh_mdot == np.nanmax(bh_mdot)) # spurious spike due to seed event
    bh_mdot[wmax] = np.nan

    # make plot booklets / movies
    w = np.where(data['global']['mask'] == 1)

    for snap in w[0]: #[2000]:

        # typical configuration (3x2 panels, 1.5 aspect ratio)
        if not extended:
            # plot setup
            fig = plt.figure(figsize=[38.4, 21.6]) # produce 3840x2160px image at dpi==100
            gs = gridspec.GridSpec(2,3)

            # upper left: histo2d [radlog,vrad] massfrac
            _histo2d_helper(gs,i=0,pt='gas',xaxis='radlog',yaxis='vrad',color='massfracnorm',clim=lim2)

            # upper center: histo2d [radlog,vrad] mean temperature
            _histo2d_helper(gs,i=1,pt='gas',xaxis='radlog',yaxis='vrad',color='templinear',clim=lim4)

            # upper right: [rad,vrelmag] mass frac
            _histo2d_helper(gs,i=2,pt='gas',xaxis='rad',yaxis='vrel',color='massfracnorm',clim=lim2)

            # lower left: 2dhisto [dens,temp] mean vrad
            _histo2d_helper(gs,i=3,pt='gas',xaxis='numdens',yaxis='temp',color='vrad',clim=lim5)

            # lower center: gas dens image, 200 kpc, xy
            _image_helper(gs,i=4,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=200.0)

            # lower right: (top) masses/blackhole energetics, (bottom) vrad histograms
            gs_local = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[5], height_ratios=[2,1], hspace=0.20)
            _lineplot_helper(gs_local,i=0,conf=1)
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='vrad',yaxis='count',ylim=lim3)

        # extended configuration (5x3 panels, 1.67 aspect ratio)
        if extended:
            # plot setup
            fig = plt.figure(figsize=[38.4/0.75, 19.8/0.75]) # produce 3840x1920px image at dpi==75
            gs = gridspec.GridSpec(3,5)

            # upper col0: histo2d [rad,vrad] massfrac
            _histo2d_helper(gs,i=0,pt='gas',xaxis='radlog',yaxis='vrad',color='massfracnorm',clim=lim2)

            # upper col1: histo2d [rad,vrad] mean temperature
            _histo2d_helper(gs,i=1,pt='gas',xaxis='radlog',yaxis='vrad',color='templinear',clim=lim4)

            # upper col2: histo2d [rad,vrelmag] mass frac
            _histo2d_helper(gs,i=2,pt='gas',xaxis='rad',yaxis='vrel',color='massfracnorm',clim=lim2)

            # upper col3: gas radial density profile, vrad vs. temp relation
            gs_local = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[3], hspace=0.25)
            _histo1d_helper(gs_local,i=0,pt='gas',xaxis='radlog',yaxis='numdens',ylim=lim6)
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='radlog',yaxis='temp',ylim=lim7)

            # upper col4: SFR and BH Mdot
            _lineplot_helper(gs,i=4,conf=2)

            # center col0: 2dhisto [dens,temp] mass frac
            _histo2d_helper(gs,i=5,pt='gas',xaxis='numdens',yaxis='temp',color='massfrac',clim=[-4.0,-0.5])

            # center col1: 2dhisto [rad,vrad] temp
            _histo2d_helper(gs,i=6,pt='gas',xaxis='rad',yaxis='vrad',color='templinear',clim=lim4)

            # center col2: 2dhisto [rad,vrad] mass frac (stars)
            _histo2d_helper(gs,i=7,pt='stars',xaxis='rad',yaxis='vrad',color='massfracnorm',clim=lim2)

            # center col3: gas vrad 1d histograms
            _histo1d_helper(gs,i=8,pt='gas',xaxis='vrad',yaxis='count',ylim=lim3)
            
            # center col4: gas vrel/temp 1d histograms
            gs_local = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[9], hspace=0.25)
            _histo1d_helper(gs_local,i=0,pt='gas',xaxis='vrel',yaxis='count',ylim=lim3)
            _histo1d_helper(gs_local,i=1,pt='gas',xaxis='temp',yaxis='count',ylim=[-2.0, 0.5])

            # bottom col0: 2dhisto [dens,temp] mean vrad
            _histo2d_helper(gs,i=10,pt='gas',xaxis='numdens',yaxis='temp',color='vrad',clim=lim5)

            # bottom col1: gas dens image, 200 kpc, xz (or 20kpc, xy)
            #_image_helper(gs,i=11,pt='gas',field='coldens_msunkpc2',axes=[0,2],boxSize=200.0)
            _image_helper(gs,i=11,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=20.0)

            # bottom col2: gas dens image, 200 kpc, xy
            _image_helper(gs,i=12,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=200.0)

            # bottom col3: gas dens image, 2000 kpc, xy
            _image_helper(gs,i=13,pt='gas',field='coldens_msunkpc2',axes=[0,1],boxSize=2000.0)

            # bottom col4: line plot, masses and blackhole diagnostics
            _lineplot_helper(gs,i=14,conf=1)

        # finish
        fig.tight_layout()
        fig.savefig('vis_%s%s_h%s_%04d.png' % (sP.simName,'_extended' if extended else '',haloInd,snap), 
            dpi=(75 if extended else 100),facecolor=color1)
        plt.close(fig)

def visHaloTimeEvoSubbox(sP, sbNum, selInd, minM200=11.5, extended=False, pStyle='black'):
    """ Visualize halo time evolution as a series of complex multi-panel images, for subbox-based tracking. """
    sel = halo_selection(sP, minM200=minM200)
    _, subboxInds, _, _, subhaloPos, subboxScaleFac, _ = selection_subbox_overlap(sP, sbNum, sel)

    # slice out information for only the selected halo
    subhaloPos = np.squeeze( subhaloPos[selInd,:,:] )

    # load data and call render
    data = haloTimeEvoDataSubbox(sP, sbNum, selInd, minM200=minM200)
    sP_sub = simParams(res=sP.res, run=sP.run, variant='subbox%d' % sbNum)

    haloInd = 'sb%dh%d' % (sbNum,subboxInds[selInd])

    visHaloTimeEvo(sP_sub, data, subhaloPos, subboxScaleFac, haloInd, extended=extended, pStyle=pStyle)

def visHaloTimeEvoFullbox(sP, haloInd, extended=False, pStyle='white'):
    """ As above, but for full-box based tracking. """
    halo = sP.groupCatSingle(haloID=haloInd)
    snaps, snapTimes, haloPos = mpbPositionComplete(sP, halo['GroupFirstSub'])

    # load data and call render
    data = haloTimeEvoDataFullbox(sP, [haloInd])

    visHaloTimeEvo(sP, data, haloPos, snapTimes, haloInd, extended=extended, pStyle=pStyle)

def singleHaloDemonstrationImage():
    """ Projections of a big halo showing outflow characteristics. """
    panels = []

    run        = 'tng'
    res        = 1080
    redshift   = 1.6
    rVirFracs  = [0.5, 1.0] # None
    method     = 'sphMap'
    nPixels    = [1920,1080]
    axes       = [0,1]
    labelZ     = True
    labelScale = True
    labelSim   = True
    labelHalo  = True
    relCoords  = True
    rotation   = None

    size = 1000.0
    sizeType = 'pkpc'

    haloID = 4

    sP = simParams(res=res, run=run, redshift=redshift)
    
    hInd = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

    #panels.append( {'partType':'gas', 'partField':'shocks_dedt', 'valMinMax':[33, 39.5]} )
    panels.append( {'partType':'gas', 'partField':'velmag', 'valMinMax':[0, 900]} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0] #1200 #nPixels[0]
        colorbars    = True
        saveFilename = './oneHaloSingleField_%s_%d_z%.1f_haloID-%d.pdf' % (run,res,redshift,haloID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def subboxSingleVelocityFrame():
    """ Grab one of the velocity frames from SB2. """
    panels = []

    run     = 'tng'
    method  = 'sphMap'
    nPixels = [3840,2160]
    axes    = [0,1] # x,y

    labelScale = 'physical'
    labelZ     = True
    plotHalos  = False

    res      = 2160
    redshift = 1.68
    variant = 'subbox2'

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    panels.append( {'partType':'gas',   'partField':'velmag', 'valMinMax':[50,1100]} )
    #panels.append( {'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )
    #panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.4]} )
    #panels.append( {'partType':'gas', 'partField':'Z_solar', 'valMinMax':[-2.0,0.0]} )
    #panels.append( {'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.3]} )

    class plotConfig:
        saveFilename = 'vis_%s_%d_%s_%s.pdf' % (sP.simName,sP.snap,panels[0]['partType'],panels[0]['partField'])
        #savePath = '/u/dnelson/data/frames/%s%s/' % (res,variant)
        plotStyle = 'edged'
        rasterPx  = nPixels
        colorbars = True

        # movie config
        #minZ      = 0.0
        #maxZ      = 50.0 # tng subboxes start at a=0.02
        #maxNSnaps = None #2400 #4500 # 2.5 min at 30 fps (1820sb0 render)

    renderBox(panels, plotConfig, locals())
    #renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def run():
    """ Run. """
    redshift = 0.73 # snapshot 58, where intermediate trees were constructed

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)

    if 0:
        # subbox: vis image sequence
        #preRenderSubboxImages(TNG50, sbNum=2, selInd=0)
        visHaloTimeEvoSubbox(TNG50, sbNum=2, selInd=0, extended=True, minM200=12.0)

    if 0:
        # fullbox: vis image sequence
        sel = halo_selection(TNG50, minM200=12.0)
        #preRenderFullboxImages(TNG50, haloInds=sel['haloInds'][0:20])
        visHaloTimeEvoFullbox(TNG50, haloInd=sel['haloInds'][0], extended=True)

