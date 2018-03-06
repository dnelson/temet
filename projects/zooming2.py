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

def visualize_halo(conf=1, quadrant=False):
    """ Visualize single final halo of h2_L11_12_FP (boosted, sims.zooms2) at z=2.25. """
    panels = []

    run        = 'zooms2_josh' # 'zooms2'
    res        = 11
    redshift   = 2.25
    hInd       = 2
    variant    = 'FP' # MO, PO, FP, None

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

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant, hInd=hInd)

    if quadrant:
        # zoom in to upper right quadrant
        halo = groupCatSingle(sP, haloID=sP.zoomSubhaloID)
        cenShift = [halo['Group_R_Crit200']*0.25,halo['Group_R_Crit200']*0.25,0]
        size = sP.units.codeLengthToKpc(halo['Group_R_Crit200']*0.5)
        labelHalo = False
        relCoords = False

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
        panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,6.3]} )
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
        saveFilename = './%s_%s_%s_%d_z%.2f_%d%s.pdf' % (sP.simName,panels[0]['partType'],panels[0]['partField'],res,redshift,size,sizeType)

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
