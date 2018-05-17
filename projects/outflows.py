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
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, colorConverter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

from collections import OrderedDict
import multiprocessing as mp
from functools import partial

from util import simParams
from util.helper import loadColorTable, running_median, logZeroNaN, iterable, closest, getWhiteBlackColors, nUnique
from plot.config import *
from plot.general import plotHistogram1D, plotPhaseSpace2D
from plot.cosmoGeneral import quantMedianVsSecondQuant
from cosmo.mergertree import mpbPositionComplete
from vis.common import gridBox, setAxisColors, setColorbarColors
from vis.halo import renderSingleHalo, selectHalosFromMassBin
from projects.outflows_analysis import halo_selection, selection_subbox_overlap, haloTimeEvoDataSubbox, haloTimeEvoDataFullbox, \
    loadRadialMassFluxes
from tracer.tracerMC import match3

def galaxyMosaics(conf=1, rotation='face-on'):
    """ Mosaic, top N most massive. 
      todo: multi-redshift
      todo: pick systems by hand instead of top N
      todo: more than 1 mass bin?"""
    res        = 2160
    redshift   = 1.0
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

        if i == 0: # upper left
            panels[-1]['labelScale'] = 'physical'
        if i == 2: # upper right
            panels[-1]['labelZ'] = True

    hStr = '_HISTO' if method == 'histo' else ''
    plotConfig.saveFilename = 'renderHalos_%s-%d_%s_%s%s.pdf' % (sP.simName,sP.snap,panels[0]['partField'],rotation,hStr)

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

def explore_vrad_selection(sP):
    """ Testing. A variety of plots looking at halo-centric gas/wind radial velocities. For entire selection. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    if sP.snap == 58: # z=0.73
        sel = halo_selection(sP, minM200=12.0)
        haloIndsPlot = sel['haloInds']
    else:
        haloIndsPlot = [22] #[20,21,22,23,24,25,26,27,28]
        print('At snap=%d hard-coded haloIndsPlot = ' % sP.snap, haloIndsPlot)

    # plot: booklet of 1D vrad profiles
    if 0:
        numPerPage = 5
        numPages = haloIndsPlot.size / numPerPage
        pdf = PdfPages('histo1d_vrad.pdf')

        for i in range(numPages):
            haloIDs = [haloIndsPlot[(i+0)*numPerPage : (i+1)*numPerPage]] # fof scope
            plotHistogram1D([sP], haloIDs=haloIDs, ptType='gas', ptProperty='vrad', 
                sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

        pdf.close()

    # plot: booklets of 2D phase diagrams
    if 0:
        pdf = PdfPages('phase2d_vrad_numdens.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrelmag_rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, 
                yQuant='vrelmag', ylim=[0,3000], nBins=nBins, clim=clim)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_rad_kpc_linear.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad_kpc_linear', haloID=haloID, pdf=pdf, 
                yQuant='vrad', ylim=vrad_lim, nBins=nBins, clim=[-4.5,-7.0])
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_rad_sfreq0.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', haloID=haloID, pdf=pdf, sfreq0=True, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_rad_wind.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='wind_real', xQuant='rad', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', haloID=haloID, pdf=pdf, **commonOpts)
        pdf.close()
    if 0:
        pdf = PdfPages('phase2d_vrad_temp_c=rad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='temp', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['rad'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_vrad_numdens_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 1:
        pdf = PdfPages('phase2d_vrad_rad_c=temp.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['temp'], weights=None, haloID=haloID, pdf=pdf)
            plotPhaseSpace2D(sP, partType='gas', xQuant='rad', yQuant='vrad', ylim=vrad_lim, nBins=nBins, 
                meancolors=['coolrate_ratio'], weights=None, haloID=haloID, pdf=pdf)
        pdf.close()

    if 0:
        pdf = PdfPages('phase2d_dens_temp_c=vrad.pdf')
        for haloID in haloIndsPlot:
            plotPhaseSpace2D(sP, partType='gas', xQuant='numdens', yQuant='temp', nBins=nBins, 
                meancolors=['vrad'], weights=None, haloID=haloID, clim=vrad_lim, pdf=pdf)
        pdf.close()


def explore_vrad_halos(sP, haloIndsPlot):
    """ Testing. A variety of plots looking at halo-centric gas/wind radial velocities. For input halos. """

    # general config
    nBins = 200
    vrad_lim = [-1000.0, 2000.0]
    clim = [-2.0, -6.0]
    commonOpts = {'yQuant':'vrad', 'ylim':vrad_lim, 'nBins':nBins, 'clim':clim}

    pdf = PdfPages('halos_%s.pdf' % ('-'.join([str(i) for i in haloIndsPlot])))

    # plot: booklet of 1D vrad profiles
    numPerPage = 5
    numPages = len(haloIndsPlot) / numPerPage

    for i in range(numPages):
        haloIDs = [haloIndsPlot[(i+0)*numPerPage : (i+1)*numPerPage]] # fof scope
        plotHistogram1D([sP], haloIDs=haloIDs, ptType='gas', ptProperty='vrad', 
            sfreq0=False, ylim=[-6.0,-2.0], xlim=vrad_lim, pdf=pdf)

    # plot: booklets of 2D phase diagrams
    for haloID in haloIndsPlot:
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

def sfms_smoothing_comparison(sP):
    """ Compare instantaneous vs. timescale smoothed galaxy SFRs, vs stellar mass. """

    xQuant = 'mstar_30pkpc' #'mhalo_200_log',mstar1_log','mstar_30pkpc'
    xlim = [8.0, 12.0]
    yQuants = ['sfr_30pkpc_instant','sfr_30pkpc_10myr','sfr_30pkpc_50myr','sfr_30pkpc_100myr',
               'ssfr_30pkpc_instant','ssfr_30pkpc_10myr','ssfr_30pkpc_50myr','ssfr_30pkpc_100myr']
    cenSatSelect = 'cen'

    sQuant = None #'color_C_gr','mstar_out_100kpc_frac_r200'
    sLowerPercs = None #[10,50]
    sUpperPercs = None #[90,50]

    pdf = PdfPages('sfms_smoothing_comparison.pdf')
    for yQuant in yQuants:
        quantMedianVsSecondQuant([sP], pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                                 #sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs, 
                                 xlim=xlim, scatterPoints=True)
    pdf.close()

def gasOutflowRatesVsMstar(sP, ptType):
    """ Explore radial mass flux data, aggregating into a single Msun/yr value for each galaxy, and plotting 
    trends as a function of stellar mass. """

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    assert ptType in ['Gas','Wind']

    # plot config
    xlim = [7.5, 11.0]
    ylim = [-2.8, 2.5] # outflow rates default
    ylimEta = [-2.0, 2.0] # mass loadings default

    binSize = 0.2 # in M*
    markersize = 0.0 # 4.0, or 0.0 to disable
    malpha = 0.4
    linestyles = ['-','--',':','-.']

    def _plotHelper(vcutIndsPlot,radIndsPlot,saveName=None,pdf=None,ylimLoc=None,massLoading=False):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
        ax = fig.add_subplot(111)
        
        if ylimLoc is None: ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)

        ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ]')

        if massLoading:
            ax.set_ylabel('Mass Loading $\eta = \dot{M}_{\\rm w} / \dot{M}_\star$ [ log ]')
        else:
            ax.set_ylabel('%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType)

        labels_sec = []

        # loop over radii or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):

            if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                c = ax._get_lines.prop_cycler.next()['color'] # one color per rad, if cycling over both

            for j, vcut_ind in enumerate(vcutIndsPlot):
                # local data
                if massLoading:
                    yy = logZeroNaN( eta[:,rad_ind,vcut_ind] ) # zero flux -> nan, skipped in median
                else:
                    yy = logZeroNaN( mdot[:,rad_ind,vcut_ind] ) # zero flux -> nan, skipped in median

                # label and color
                radMidPoint = 0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1])
                if len(vcutIndsPlot) == 1:
                    label = 'r = %3d kpc' % radMidPoint
                    labelFixed = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                if len(radIndsPlot) == 1:
                    label = 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind]
                    labelFixed = 'r = %3d kpc' % radMidPoint
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    label = 'r = %3d kpc' % radMidPoint # primary label radius, by color
                    labels_sec.append( 'v$_{\\rm rad}$ > %3d km/s' % vcut_vals[vcut_ind] ) # second legend: vcut by ls
                    if j > 0: label = ''

                if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
                    c = ax._get_lines.prop_cycler.next()['color']

                # symbols for each system
                if markersize > 0:
                    ax.plot(mstar, yy, 'o', color=c, markersize=markersize, alpha=malpha)                    

                # mark those at absolute zero just above the bottom of the y-axis
                if markersize > 0:
                    off = 0.4
                    w_zero = np.where(np.isnan(yy))
                    yy_zero = np.random.uniform( size=len(w_zero[0]), low=ylim[0]+off/2, high=ylim[0]+off )
                    ax.plot(mstar[w_zero], yy_zero, 'o', alpha=malpha/2, markersize=markersize, color=c)

                # median line and 1sigma band
                xm, ym, sm, pm = running_median(mstar,yy,binSize=binSize,percs=[16,84])

                if xm.size > sKn:
                    ym = savgol_filter(ym,sKn,sKo)
                    sm = savgol_filter(sm,sKn,sKo)
                    pm = savgol_filter(pm,sKn,sKo,axis=1)

                lsInd = j if len(vcutIndsPlot) < 4 else i
                l, = ax.plot(xm[:-1], ym[:-1], linestyles[lsInd], lw=lw, alpha=1.0, color=c, label=label)

                if i == 0 or j == 0:
                    y_down = pm[0,:-1] #np.array(ym[:-1]) - sm[:-1]
                    y_up   = pm[-1,:-1] #np.array(ym[:-1]) + sm[:-1]
                    ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc='lower right')
            ax.add_artist(legend2)

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [ plt.Line2D( (0,1), (0,0), color='black', marker='', lw=lw, linestyle=linestyles[j]) for j in range(len(vcutIndsPlot)) ]
            legend2 = ax.legend(lines, labels_sec, loc='lower right')
            ax.add_artist(legend2)

        legend1 = ax.legend(loc='upper right' if massLoading else 'upper left')

        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    mdot, mstar, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, ptType)

    # load mass loadings
    acField = 'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr_Outflow-instantaneous'
    ac = sP.auxCat(acField)
    eta = ac[acField]

    # (A) plot for a given vcut, at many radii
    radInds = [1,3,4,5,6,7]

    pdf = PdfPages('outflowRate_%s_mstar_A_%s_%d.pdf' % (ptType,sP.simName,sP.snap))
    for vcut_ind in range(vcut_vals.size):
        _plotHelper(vcutIndsPlot=[vcut_ind],radIndsPlot=radInds,pdf=pdf)
    pdf.close()

    pdf = PdfPages('massLoading_mstar_A_%s_%d.pdf' % (sP.simName,sP.snap))
    for vcut_ind in range(vcut_vals.size):
        _plotHelper(vcutIndsPlot=[vcut_ind],radIndsPlot=radInds,ylimLoc=ylimEta,pdf=pdf,massLoading=True)
    pdf.close()

    # (B) plot for a given radii, at many vcuts
    vcutInds = [0,1,2,3,4]

    pdf = PdfPages('outflowRate_%s_mstar_B_%s_%d.pdf' % (ptType,sP.simName,sP.snap))
    for rad_ind in range(numBins['rad']):
        _plotHelper(vcutIndsPlot=vcutInds,radIndsPlot=[rad_ind],pdf=pdf)
    pdf.close()

    pdf = PdfPages('massLoading_mstar_B_%s_%d.pdf' % (sP.simName,sP.snap))
    for rad_ind in range(numBins['rad']):
        _plotHelper(vcutIndsPlot=vcutInds,radIndsPlot=[rad_ind],ylimLoc=ylimEta,pdf=pdf,massLoading=True)
    pdf.close()

    # (C) single-panel combination of both radial and vcut variations
    if ptType == 'Gas':
        vcutIndsPlot = [0,2,3]
        radIndsPlot = [1,2,5]
        ylimLoc = [-2.5,2.0]

    if ptType == 'Wind':
        vcutIndsPlot = [0,2,4]
        radIndsPlot = [1,2,5]
        ylimLoc = [-3.0,1.0]

    saveName = 'outflowRate_%s_mstar_C_%s_%d.pdf' % (ptType,sP.simName,sP.snap)
    _plotHelper(vcutIndsPlot,radIndsPlot,saveName,ylimLoc=ylimLoc)

    saveName = 'massLoading_mstar_C_%s_%d.pdf' % (sP.simName,sP.snap)
    _plotHelper(vcutIndsPlot,radIndsPlot,saveName,ylimLoc=ylimEta,massLoading=True)


    print('TODO: consider mean=True in running_median, or otherwise how to better consider zero outflow points?')

def gasOutflowRatesVsQuantStackedInMstar(sP, quant, mStarBins):
    """ Explore radial mass flux data, as a function of one of the histogrammed quantities (x-axis), for single 
    galaxies or stacked in bins of stellar mass. """

    # config
    scope = 'SubfindWithFuzz' # or 'Global'
    ptType = 'Gas'

    # plot config
    ylim = [-3.0,1.0]
    linestyles = ['-','--',':','-.']

    labels = {'temp'    : 'Gas Temperature [ log K ]',
              'z_solar' : 'Gas Metallicity [ log Z$_{\\rm sun}$ ]',
              'numdens' : 'Gas Density [ log cm$^{-3}$ ]'}
    limits = {'temp'    : [3.0,9.0],
              'z_solar' : [-3.0,1.0],
              'numdens' : [-6.0,4.0]}

    def _plotHelper(vcut_ind,rad_ind,quant,mStarBins=None,indivInds=None,stat='mean',skipZeroFluxes=False,saveName=None,pdf=None):
        """ Plot a radii series, vcut series, or both. """
        # plot setup
        fig = plt.figure(figsize=[figsize[0]*sfclean, figsize[1]*sfclean])
        ax = fig.add_subplot(111)
        
        ax.set_xlim(limits[quant])
        ax.set_ylim(ylim)

        ax.set_xlabel(labels[quant])
        ax.set_ylabel('%s Outflow Rate [ log M$_{\\rm sun}$ / yr ]' % ptType)

        labels_sec = []

        # loop over stellar mass bins and stack
        for i, mStarBin in enumerate(mStarBins):

            c = ax._get_lines.prop_cycler.next()['color'] # one color per bin

            # local data
            w = np.where( (mstar > mStarBin[0]) & (mstar <= mStarBin[1]) )
            mdot_local = np.squeeze( mdot[w,rad_ind,vcut_ind,:] )
            #print(mStarBin, ' number of galaxies: ',len(w[0]))

            # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
            if skipZeroFluxes:
                w_zero = np.where(mdot_local == 0.0)
                mdot_local[w_zero] = np.nan

            if stat == 'median':
                yy = np.nanmedian(mdot_local, axis=0) # median on subhalo axis
            if stat == 'mean':
                yy = np.nanmean(mdot_local, axis=0) # mean

            yy = logZeroNaN(yy) # zero flux -> nan

            # label and color
            xx = 0.5*(binConfig[quant][:-1] + binConfig[quant][1:])

            radMidPoint   = 0.5*(binConfig['rad'][rad_ind] + binConfig['rad'][rad_ind+1])
            mStarMidPoint = 0.5*(mStarBin[0] + mStarBin[1])
            labelFixed = 'r = %3d kpc, v$_{\\rm rad}$ > %3d km/s' % (radMidPoint,vcut_vals[vcut_ind])
            label = 'M$^\star$ = %.1f' % mStarMidPoint

            # median line and 1sigma band
            pm = np.nanpercentile(mdot_local, [16,84], axis=0, interpolation='linear')
            pm = logZeroNaN(pm)

            yy  = savgol_filter(yy,sKn,sKo)
            pm = savgol_filter(pm,sKn,sKo,axis=1)

            #l, = ax.plot(xm[:-1], ym[:-1], linestyles[i], lw=lw, alpha=1.0, color=c, label=label)
            l, = ax.plot(xx, yy, linestyles[0], lw=lw, alpha=1.0, color=c, label=label)

            if i == 0 or i == len(mStarBins)-1:
                w = np.where( np.isfinite(pm[0,:]) & np.isfinite(pm[-1,:]) )[0]
                ax.fill_between(xx[w], pm[0,w], pm[-1,w], color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        line = plt.Line2D( (0,1), (0,0), color='white', marker='', lw=0.0)
        legend2 = ax.legend([line], [labelFixed], handlelength=0.0, loc='upper left')
        ax.add_artist(legend2)

        legend1 = ax.legend(loc='upper right')

        fig.tight_layout()
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    mdot, mstar, binConfig, numBins, vcut_vals = loadRadialMassFluxes(sP, scope, ptType, thirdQuant=quant)

    for stat in ['mean','median']:
        for skipZeros in [True,False]:
            print(quant,stat,skipZeros)
            # (A) vs temperature, composite at fixed rad/vcut
            radInd  = 1
            vcutInd = 0
            saveName = 'outflowRate_%s_%s_mstar_%s_%d_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,sP.snap,stat,skipZeros)
            _plotHelper(vcutInd,radInd,quant,mStarBins,stat,skipZeroFluxes=skipZeros,saveName=saveName)

            # (B) vs temperature, booklet across rad and vcut variations
            pdf = PdfPages('outflowRate_B_%s_%s_mstar_%s_%d_%s_skipzeros-%s.pdf' % (ptType,quant,sP.simName,sP.snap,stat,skipZeros))

            for radInd in [1,3,4,5,6,7]:
                for vcutInd in [0,1,2,3,4]:
                    _plotHelper(vcutInd,radInd,quant,mStarBins,stat,skipZeroFluxes=skipZeros,pdf=pdf)

            pdf.close()

def _haloSizeScalesHelper(ax, sP, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c):
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
        xrvir = [rHalfFac*avg_rhalf_code, rHalfFac*avg_rhalf_code]
        if 'log_' in xaxis: xrvir = np.log10(xrvir)
        textStr = '%dr$_{\star}$' % rHalfFac

        if 1: #i == 0 or i == len(sPs)-1:
            ax.plot(xrvir, y2, lw=lw*1.5, ls=linestyles[i], color=c, alpha=0.1)
            if k == len(massBins)-1 and i == 0: ax.text(xrvir[0]-xoff, y2[0], textStr, color=c, **textOpts)

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
        ax.set_xlim([-0.5, 2.5])
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

    ylabels_3d = {'SFR'                   : '$\dot{\\rho}_\star$ [ M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-3}$ ]',
                  'Gas_Mass'              : '$\\rho_{\\rm gas}$ [ M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Stars_Mass'            : '$\\rho_{\\rm stars}$ [ M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\\rho_{\\rm gas}$ / $\\rho_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\\rho_{\\rm metals}$ [ M$_{\\rm sun}$ kpc$^{-3}$ ]',
                  'Gas_Metallicity'       : 'Gas Metallicity (unweighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Metallicity_sfrWt' : 'Gas Metallicity (SFR weighted) [ log Z$_{\\rm sun}$ ]'}
    ylims_3d   = {'SFR'                   : [-10.0, 0.0],
                  'Gas_Mass'              : [0.0, 9.0],
                  'Stars_Mass'            : [0.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [-4.0,  8.0],
                  'Gas_Metallicity'       : [-2.0, 1.0],
                  'Gas_Metallicity_sfrWt' : [-1.5, 1.0]}

    ylabels_2d = {'SFR'                   : '$\dot{\Sigma}_\star$ [ M$_{\\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ]',
                  'Gas_Mass'              : '$\Sigma_{\\rm gas}$ [ M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Stars_Mass'            : '$\Sigma_{\\rm stars}$ [ M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Fraction'          : 'f$_{\\rm gas}$ = $\Sigma_{\\rm gas}$ / $\Sigma_{\\rm b}$',
                  'Gas_Metal_Mass'        : '$\Sigma_{\\rm metals}$ [ M$_{\\rm sun}$ kpc$^{-2}$ ]',
                  'Gas_Metallicity'       : 'Gas Metallicity (unweighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_Metallicity_sfrWt' : 'Gas Metallicity (SFR weighted) [ log Z$_{\\rm sun}$ ]',
                  'Gas_LOSVel_sfrWt'      : 'Gas Velocity v$_{\\rm LOS}$ (SFR weighted) [ km/s ]',
                  'Gas_LOSVelSigma'       : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D}$ [ km/s ]',
                  'Gas_LOSVelSigma_sfrWt' : 'Gas Velocity Dispersion $\sigma_{\\rm LOS,1D,SFRw}$ [ km/s ]'}
    ylims_2d   = {'SFR'                   : [-7.0, 0.0],
                  'Gas_Mass'              : [3.0, 9.0],
                  'Stars_Mass'            : [3.0, 11.0],
                  'Gas_Fraction'          : [0.0, 1.0],
                  'Gas_Metal_Mass'        : [0.0, 8.0],
                  'Gas_Metallicity'       : [-2.0, 1.0],
                  'Gas_Metallicity_sfrWt' : [-1.5, 1.0],
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
                yy_mean = savgol_filter(yy_mean,sKn,sKo)
                yy_med = savgol_filter(yy_med,sKn,sKo)
                yp = savgol_filter(yp,sKn,sKo,axis=1)

            #if 'Metallicity' in field:
            #    # test: remove noisy last point which is non-monotonic
            #    w = np.where(np.isfinite(yy_med))
            #    if yy_med[w][-1] > yy_med[w][-2]:
            #        yy_med[w[0][-1]] = np.nan

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
            _haloSizeScalesHelper(ax, sP, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, c)

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
    redshift = 0.73 # snapshot 58, where intermediate trees were constructed

    TNG50   = simParams(res=2160,run='tng',redshift=redshift)
    TNG100  = simParams(res=1820,run='tng',redshift=redshift)
    TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)
    TNG50_3 = simParams(res=540,run='tng',redshift=redshift)
    TNG50_4 = simParams(res=270,run='tng',redshift=redshift)

    mStarBins = [ [7.9,8.1],[8.9,9.1],[9.4,9.6],[9.9,10.1],[10.3,10.7],[10.8,11.2],[11.3,11.7] ]

    radProfileFields = ['SFR','Gas_Mass','Gas_Metal_Mass', 'Stars_Mass', 'Gas_Fraction',
                        'Gas_Metallicity','Gas_Metallicity_sfrWt',
                        'Gas_LOSVelSigma','Gas_LOSVelSigma_sfrWt']

    if 0:
        # vrad plots, entire selection
        explore_vrad_selection(TNG50)

    if 0:
        # vrad plots, single halo
        subInd = 389836 # first one in subbox0 intersecting with >11.5 selection
        haloInd = TNG50.groupCatSingle(subhaloID=subInd)['SubhaloGrNr']
        explore_vrad_halos(TNG50, haloIndsPlot=[haloInd])

    if 0:
        # subbox: vis image sequence
        #preRenderSubboxImages(TNG50, sbNum=2, selInd=0)
        visHaloTimeEvoSubbox(TNG50, sbNum=2, selInd=0, extended=True, minM200=12.0)

    if 0:
        # fullbox: vis image sequence
        sel = halo_selection(TNG50, minM200=12.0)
        #preRenderFullboxImages(TNG50, haloInds=sel['haloInds'][0:20])
        visHaloTimeEvoFullbox(TNG50, haloInd=sel['haloInds'][0], extended=True)

    if 0:
        # sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG50.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG50)

    if 0:
        sfms_smoothing_comparison(TNG50)

    if 1:
        # net outflow rates, fully marginalized, as a function of stellar mass
        for ptType in ['Gas','Wind']:
            gasOutflowRatesVsMstar(TNG50, ptType=ptType)

    if 0:
        # net outflow rate distributions, vs a single quantity (marginalized over all others), stacked in M* bins
        for quant in ['temp','numdens','z_solar']:
            gasOutflowRatesVsQuantStackedInMstar(TNG50, quant=quant, mStarBins=mStarBins)

    if 0:
        # radial profiles: stellar mass stacks, at one redshift
        if sPs is None: sPs = [TNG50]
        cenSatSelect = 'cen'

        for field in radProfileFields:
            pdf = PdfPages('radprofiles_%s_%s_%d_%s.pdf' % (field,sPs[0].simName,sPs[0].snap,cenSatSelect))

            for projDim in ['2Dz','3D']:
                if projDim == '3D' and '_LOSVel' in field: continue

                for xaxis in ['log_pkpc','pkpc','log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
                    stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect='cen', 
                                          projDim=projDim, mStarBins=mStarBins, pdf=pdf)
            pdf.close()

        return sPs

    if 0:
        # TEST radial profiles, look at edgeon LOSVel, faceon LOSVelSigma
        if sPs is None: sPs = [TNG50]
        cenSatSelect = 'cen'

        for field in ['Gas_LOSVelSigma']:
            pdf = PdfPages('radprofiles_%s_%s_%d_%s.pdf' % (field,sPs[0].simName,sPs[0].snap,cenSatSelect))

            for xaxis in ['log_pkpc','pkpc','log_rvir','rvir','log_rhalf','rhalf','log_re','re']:
                for projDim in ['2Dz','2Dedgeon','3D']:
                    if projDim == '3D' and '_LOSVel' in field: continue

                    stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect='cen', 
                                          projDim=projDim, mStarBins=mStarBins, pdf=pdf)
            pdf.close()

        return sPs



    if 0:
        # radial profiles: vs redshift, separate plot for each mstar bin
        redshifts = [1.0, 2.0, 4.0, 6.0]
        cenSatSelect = 'cen'

        if sPs is None:
            sPs = []
            for redshift in redshifts:
                sP = simParams(res=2160, run='tng', redshift=redshift)
                sPs.append(sP)

        for field in radProfileFields:

            pdf = PdfPages('radprofiles_%s_%s_zevo_%s.pdf' % (field,sPs[0].simName,cenSatSelect))

            for projDim in ['2Dz','3D']:
                if projDim == '3D' and '_LOSVel' in field: continue

                for xaxis in ['log_pkpc','log_rvir','log_rhalf','pkpc']:
                    for i, mStarBin in enumerate(mStarBins):
                        stackedRadialProfiles(sPs, field, xaxis=xaxis, cenSatSelect='cen', 
                            projDim=projDim, mStarBins=[mStarBin], colorOff=i, pdf=pdf)

            pdf.close()

        return sPs