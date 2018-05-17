"""
projects/zooming2.py
  Plots for "Zooming in on accretion" paper series (II) - Suresh et al.
  in prep.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt

from util.simParams import simParams
from cosmo.load import groupCatSingle
from plot.general import plotPhaseSpace2D, plotHistogram1D, plotSingleRadialProfile
from vis.halo import renderSingleHalo

def visualize_halo(conf=1, quadrant=False, snap=None):
    """ Visualize single final halo of h2_L11_12_FP (boosted, sims.zooms2) at z=2.25. """
    panels = []

    run        = 'zooms2_josh' # 'zooms2'
    res        = 11
    hInd       = 2
    variant    = 'FP' # MO, PO, FP, FP1/FP2/FP3 None

    redshift = 2.25 if snap is None else None

    rVirFracs  = [1.0]
    method     = 'sphMap_global'
    nPixels    = [3840,3840] # 960 or 3840
    axes       = [1,0]
    labelZ     = True
    labelScale = True
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    rotation   = None

    size       = 180.0 #400.0
    sizeType   = 'pkpc'
    axesUnits  = 'kpc'
    depthFac   = 1.0

    sP = simParams(res=res, run=run, redshift=redshift, snap=snap, variant=variant, hInd=hInd)
    if redshift is None: redshift = sP.redshift

    if quadrant:
        # zoom in to upper right quadrant
        halo = groupCatSingle(sP, haloID=sP.zoomSubhaloID)
        cenShift = [halo['Group_R_Crit200']*(0.25+0.05),halo['Group_R_Crit200']*(0.25+0.05),0]
        size = sP.units.codeLengthToKpc(halo['Group_R_Crit200']*0.4)
        labelHalo = False

    if conf == 0:
        # stellar mass column density
        panels.append( {'partType':'stars',  'partField':'coldens_msunkpc2', 'valMinMax':[4.5,8.0]} )
    if conf == 1:
        # gas column density
        panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0, 8.5]} )
    if conf == 2:
        # gas OVI column
        panels.append( {'partType':'gas', 'partField':'O VI', 'valMinMax':[12.0, 17.0]} )
    if conf == 3:
        # gas MgII column
        vMM = [11.5, 16.5] if quadrant else [10.0, 16.5]
        panels.append( {'partType':'gas', 'partField':'Mg II', 'valMinMax':vMM} )
    if conf == 4:
        # temperature
        panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.2,5.9]} )
    if conf == 5:
        # radial velocity
        panels.append( {'partType':'gas', 'partField':'radvel', 'valMinMax':[-260,260]} )
    if conf == 6:
        # magnitude of specific angular momentum
        panels.append( {'partType':'gas', 'partField':'specj_mag', 'valMinMax':[2.0,4.2]} )
    if conf == 7:
        # gas metallicity
        panels.append( {'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.0]} )

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = int(nPixels[0]*1.0)
        colorbars    = True
        saveFilename = './%s_%s_%s_%d_%d_z%.2f_%d%s.pdf' % (sP.simName,panels[0]['partType'],panels[0]['partField'],res,sP.snap,redshift,size,sizeType)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def phase_diagram_ovi():
    # OVI mass phase diagram
    sP = simParams(res=11,run='zooms2_josh',variant='FP',hInd=2,redshift=2.25)

    ptType = 'gas'
    xQuant = 'hdens'
    yQuant = 'temp'
    weights = ['O VI mass'] #,'O VII mass','O VIII mass']
    xMinMax = [-6.0,0.0]
    yMinMax = [3.5,7.0]
    contours = [-3.0, -2.0, -1.0]
    massFracMinMax = [-4.0, -1.0] #[-10.0, 0.0]
    hideBelow = True
    smoothSigma = 1.5
    haloID = 0 # None for fullbox

    plotPhaseSpace2D(sP, ptType, xQuant, yQuant, weights=weights, haloID=haloID, 
                     massFracMinMax=massFracMinMax,xMinMaxForce=xMinMax, yMinMaxForce=yMinMax, 
                     contours=contours, smoothSigma=smoothSigma, hideBelow=True)

def phase_diagram_coolingtime():
    # cooling time phase diagram
    sP = simParams(res=11,run='zooms2_josh',variant='FP',hInd=2,redshift=2.25)

    ptType = 'gas'
    xQuant = 'hdens'
    yQuant = 'temp'
    weights = None
    meancolors = ['cooltime']
    xMinMax = [-6.0,0.0]
    yMinMax = [3.5,7.0]
    contours = [-2.0, -1.0, 0.0]
    massFracMinMax = [-3.0, 1.0] # 1 Myr to 10 Gyr
    hideBelow = False
    smoothSigma = 1.5

    plotPhaseSpace2D(sP, ptType, xQuant, yQuant, weights=weights, meancolors=meancolors, haloID=0, 
                     massFracMinMax=massFracMinMax,xMinMaxForce=xMinMax, yMinMaxForce=yMinMax, 
                     contours=contours, smoothSigma=smoothSigma, hideBelow=True)

def phase_diagram_ovi_tng50_comparison():
    # TNG50 analog for comparison
    print('Need to disable OVI cache for now...')
    sP = simParams(res=2160,run='tng',redshift=2.25)
    haloID = 100

    ptType = 'gas'
    xQuant = 'hdens'
    yQuant = 'temp'
    weights = ['O VI mass'] #,'O VII mass','O VIII mass']
    xMinMax = [-6.0,0.0]
    yMinMax = [3.5,7.0]
    contours = [-3.0, -2.0, -1.0]
    massFracMinMax = [-4.0, 0.0] #[-10.0, 0.0]
    hideBelow = True
    smoothSigma = 1.0

    plotPhaseSpace2D(sP, ptType, xQuant, yQuant, weights=weights, haloID=haloID, 
                     massFracMinMax=massFracMinMax,xMinMaxForce=xMinMax, yMinMaxForce=yMinMax, 
                     contours=contours, smoothSigma=smoothSigma, hideBelow=True)

def figure1_res_statistics(conf=0):
    """ Figure 1: resolution statistics in mass/size for gas cells, comparing runs. """
    sPs = []

    if conf in [0,3]:
        # 4 run comparison
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='PO',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='MO',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='FP',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2',hInd=2,redshift=2.25) )
    if conf in [1,2]:
        # just compare L11 primordial vs. L11_12 primordial
        sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='PO',redshift=2.25) )
        sPs.append( simParams(res=11,run='zooms2',hInd=2,redshift=2.25) )

    haloIDs = np.zeros( len(sPs), dtype='int32' )

    if conf == 0:
        # Figure 1, lower left panel
        plotHistogram1D(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='mass_msun', sfreq0=True, xlim=[2.0,4.7])
    if conf == 3:
        # unused (cellsize histograms)
        plotHistogram1D(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='cellsize_kpc', sfreq0=True)
    if conf == 1:
        # Figure 1, lower right panel
        plotSingleRadialProfile(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='cellsize_kpc', 
            sfreq0=True, colorOffs=[0,2], xlim=[-0.5,3.0], scope='global')
    if conf == 2:
        # Figure 1, upper panel
        plotSingleRadialProfile(sPs, haloIDs=haloIDs, ptType='gas', ptProperty='mass_msun', 
            colorOffs=[0,2], xlim=[2.0,4.7], scope='global')

def tracer_ambient_hot_halo():
    """ Check the existence of an ambient/pre-existing hot halo at r<0.25rvir, as opposed to the possibility that all 
    hot gas is arising from wind. """
    from tracer.tracerMC import match3

    sP = simParams(res=11,run='zooms2_josh',redshift=2.25,variant='FP',hInd=2)

    temp_bins = [ [4.0,4.5], [4.5, 4.8], [4.8,5.0], [5.0,5.2], [5.2,5.4], [5.4, 5.6], [5.6, 5.8], [5.8, 6.0], [6.0, 6.2], [6.2, 6.5]]
    rad_min = 0.4
    rad_max = 0.5

    # load ParentIDs of tracer catalog
    with h5py.File(sP.postPath + 'tracer_tracks/tr_all_groups_%d_meta.hdf5' % sP.snap) as f:
        ParentIDs = f['ParentIDs'][()]

    # load radius, sfr, make selection
    rad = sP.snapshotSubset('gas', 'rad_rvir', subhaloID=0)
    sfr = sP.snapshotSubset('gas', 'sfr', subhaloID=0)
    ids = sP.snapshotSubset('gas', 'ids', subhaloID=0)

    ww = np.where( (sfr == 0.0) & (rad > rad_min) & (rad < rad_max) )

    print('Selected [%d] of [%d] gas cells.' % (len(ww[0]),sfr.size))

    # load temperature histories
    with h5py.File(sP.postPath + 'tracer_tracks/tr_all_groups_%d_temp.hdf5' % sP.snap) as f:
        redshifts = f['redshifts'][()]
        temp = f['temp'][()]

    print('Loaded temperatures.')

    # crossmatch and take selection
    ind_cat, ind_snap = match3(ParentIDs, ids[ww])

    print('Crossmatched.')

    rad = rad[ind_snap]
    temp = temp[:,ind_cat]

    # in a number of temp bins, find mean temperature shift as a function of time backwards
    temp_prev = np.zeros( (len(temp_bins), redshifts.size), dtype='float32' )
    frac_prev = np.zeros( (len(temp_bins), redshifts.size), dtype='float32' )

    for i, temp_bin in enumerate(temp_bins):
        # locate at z_final
        w = np.where( (temp[0,:] > temp_bin[0]) & (temp[0,:] <= temp_bin[1]) )[0]
        assert len(w) > 0

        loc_temps = temp[:,w]

        temp_prev[i,:] = np.nanmean(loc_temps, axis=1)
        frac_prev[i,:] = np.sum(loc_temps >= temp_bin[0], axis=1) / float(len(w))
        print(i, len(w))

    # plot
    fig = plt.figure(figsize=[14.0, 10.0])
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('redshift')
    ax.set_ylabel('temperature [ log K ]')

    ax.set_ylim([3.8,6.5])
    ax.set_xlim([2.2,2.6])

    for i, temp_bin in enumerate(temp_bins):
        label = 'T$_{\\rm zf}$ $\in$ [%.1f, %.1f]' % (temp_bin[0],temp_bin[1])
        ax.plot( redshifts, temp_prev[i,:], '-', lw=2.0, label=label)

    ax.legend()
    fig.tight_layout()
    fig.savefig('temp_evo_rvir=%.2f-%.2f.pdf' % (rad_min,rad_max))
    plt.close(fig)

    # plot 2
    fig = plt.figure(figsize=[14.0, 10.0])
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel('redshift')
    ax.set_ylabel('fraction of original bin still above bin min temp')

    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([2.2,2.6])

    for i, temp_bin in enumerate(temp_bins):
        label = 'T$_{\\rm zf}$ $\in$ [%.1f, %.1f]' % (temp_bin[0],temp_bin[1])
        ax.plot( redshifts, frac_prev[i,:], '-', lw=2.0, label=label)

    ax.legend()
    fig.tight_layout()
    fig.savefig('tempfrac_evo_rvir=%.2f-%.2f.pdf' % (rad_min,rad_max))
    plt.close(fig)
