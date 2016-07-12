"""
haloDrivers.py
  Render specific halo visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import saveBasePath
from vis.halo import renderSingleHalo, renderSingleHaloFrames, selectHalosFromMassBin
from util.helper import pSplit
from cosmo.load import groupCat, groupCatSingle
from cosmo.util import snapNumToRedshift, redshiftToSnapNum
from util import simParams

def oneHaloPressureCompAndRatios(fofHaloID=0):
    """ In 3x1 panels compare gas and magnetic pressures and their ratio. """
    panels = []

    panels.append( {'hInd':fofHaloID, 'partField':'P_B', 'valMinMax':[1.0,9.0]} )
    panels.append( {'hInd':fofHaloID, 'partField':'P_gas', 'valMinMax':[1.0,9.0]} )
    panels.append( {'hInd':fofHaloID, 'partField':'pressure_ratio', 'valMinMax':[-4.0,0.0]} )

    run        = 'tng'
    res        = 2160
    redshift   = 6.0
    partType   = 'gas'
    rVirFracs  = [1.0]
    method     = 'sphMap'
    nPixels    = [3840,3840]
    sizeFac    = 2.5
    hsmlFac    = 2.5
    axes       = [1,0]
    labelZ     = False
    labelScale = False
    labelSim   = False
    labelHalo  = False
    relCoords  = True
    rotation   = None
    mpb        = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = saveBasePath + 'presComp_%s_%d_z%.1f_halo-%d.pdf' % (run,res,redshift,fofHaloID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def multiHalosPagedOneQuantity(curPageNum, numPages=7):
    """ Split over several pages, plot many panels, one per halo, showing a single quantity. """
    panels = []

    # subhalo ID list (Guinevere sample)
    sP = simParams(res=1820, run='illustris', redshift=0.0)
    subhaloIDs = np.loadtxt(sP.derivPath + 'guinevere.list.subs.txt', dtype='int32')

    # split by page, append one panel per subhalo on this page
    subhaloIDs_loc = pSplit(subhaloIDs, numPages, curPageNum)

    for subhaloID in subhaloIDs_loc:
        panels.append( {'hInd':subhaloID} )

    #panels.append( {'partField':'HI', 'valMinMax':[14.0,21.0]} )
    #panels.append( {'partField':'velmag', 'valMinMax':[400,900]} )
    #panels.append( {'partField':'metal_solar', 'valMinMax':[-1.0,0.5]} )
    #panels.append( {'partField':'Si III', 'valMinMax':[14.0,21.0]} )

    run        = sP.run
    res        = sP.res
    redshift   = sP.redshift
    partType   = 'gas'
    partField  = 'HI_segmented'
    valMinMax  = [13.5,21.5]
    rVirFracs  = [1.0]
    method     = 'sphMap' # sphMap_global
    nPixels    = [960,960]
    sizeFac    = -140.0 # sizeFac = 10^2.8 * 0.7 * 2 (match to pm2.0 for Guinevere)
    hsmlFac    = 2.5
    axes       = [1,0]
    rotation   = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = saveBasePath + 'sample_%s_page-%d-of-%d_%s_%d.pdf' % \
                       (partField,curPageNum,numPages,run,res)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def loopInputSerial():
    """ Call another driver function several times, looping over a possible input. """
    numPages = 7
    for i in range(numPages):
        multiHalosPagedOneQuantity(i)

def boxHalo_MultiQuant():
    """ Diagnostic plot, a few quantities of a halo from a periodic box. """
    panels = []

    #panels.append( {'hsmlFac':1.0, 'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,7.5]} )
    #panels.append( {'hsmlFac':1.0, 'partType':'dm', 'partField':'coldens2_msunkpc2', 'valMinMax':[12.0,14.0]} )
    #panels.append( {'partType':'gas', 'partField':'HI_segmented', 'valMinMax':[13.5,21.5]} )

    panels.append( {'nPixels':[500,500],'partType':'gas', 'partField':'O VI', 'valMinMax':[14.0,17.0]} )
    panels.append( {'nPixels':[960,960],'partType':'gas', 'partField':'O VI', 'valMinMax':[14.0,17.0]} )
    panels.append( {'nPixels':[1920,1920],'partType':'gas', 'partField':'O VI', 'valMinMax':[14.0,17.0]} )

    hInd       = 362540
    run        = 'illustris'
    res        = 1820
    redshift   = 0.0
    rVirFracs  = [1.0]
    method     = 'sphMap'
    #nPixels    = [1920,1920]
    sizeFac    = 2.5
    hsmlFac    = 2.5
    axes       = [1,0]
    rotation   = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = saveBasePath + 'box_%s_%d_z%.1f_shID-%d_multiQuant.pdf' % (run,res,redshift,hInd)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def zoomHalo_z2_MultiQuant():
    """ For a single zooms/zooms2 halo at z=2, plot several panels comparing different quantities. """
    panels = []

    panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,9.5]} )
    panels.append( {'partType':'gas', 'partField':'entr',             'valMinMax':[6.0,9.0]} )
    panels.append( {'partType':'gas', 'partField':'velmag',           'valMinMax':[100,300]} )
    panels.append( {'partType':'gas', 'partField':'O VI',             'valMinMax':[11.0,17.0]} )
    panels.append( {'partType':'gas', 'partField':'C IV',             'valMinMax':[11.0,17.0]} )
    panels.append( {'partType':'gas', 'partField':'HI',               'valMinMax':[16.0,22.0]} )

    hInd       = 2
    run        = 'zooms2'
    res        = 10
    redshift   = 2.0
    rVirFracs  = [1.0]
    method     = 'sphMap' # sphMap_global
    nPixels    = [960,960]
    sizeFac    = 2.5
    hsmlFac    = 2.5
    axes       = [1,0]
    rotation   = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = saveBasePath + '%s_h%dL%d_z%.1f_multiQuant.pdf' % (run,hInd,res,redshift)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def tngCluster_center_timeSeriesPanels(conf=0):
    """ Plot a time series of panels from subsequent snapshots in the center of fof0. """
    panels = []

    zStart     = 0.3 # start plotting at this snapshot
    nSnapsBack = 12   # one panel per snapshot, back in time

    hInd       = 0
    run        = 'illustris'
    res        = 1820
    rVirFracs  = None #[0.05]
    method     = 'sphMap'
    nPixels    = [960,960]
    sizeFac    = -100.0
    hsmlFac    = 2.5
    labelZ     = True
    axes       = [1,0]
    rotation   = None

    if conf == 0:
        partType   = 'gas'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [6.5,9.0]
    if conf == 1:
        partType   = 'stars'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [6.5,10.0]
    if conf == 2:
        partType   = 'gas'
        partField  = 'pressure_ratio'
        valMinMax  = [-2.0,1.0]
    if conf == 3:
        partType   = 'gas'
        partField  = 'metal_solar'
        valMinMax  = [-0.5,0.5]
    if conf == 4:
        partType   = 'dm'
        partField  = 'coldens2_msunkpc2'
        valMinMax  = [12.0,15.0]

    # configure panels
    sP = simParams(res=res, run=run, redshift=zStart)
    for i in range(nSnapsBack):
        halo = groupCatSingle(sP, subhaloID=hInd)
        print(sP.snap, sP.redshift, halo['SubhaloPos'])

        panels.append( {'redshift':snapNumToRedshift(sP)} )
        sP.setSnap(sP.snap-1)

    panels[0]['labelScale'] = True
    panels[-1]['labelHalo'] = True

    class plotConfig:
        plotStyle    = 'edged_black'
        colorbars    = True
        rasterPx     = 960
        saveFilename = saveBasePath + 'timePanels_%s_hInd-%d_%s-%s_z%.1f_n%d.pdf' % \
                       (sP.simName,hInd,partType,partField,zStart,nSnapsBack)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def loopTimeSeries():
    """ Helper: Loop the above over configs. """
    for i in range(5):
        tngCluster_center_timeSeriesPanels(conf=i)

def massBinsSample_3x2_EdgeOnFaceOn(conf,haloOrMassBinNum=None):
    """ For a series of mass bins (log Mhalo), take a uniform number of halos from each and 
    make a 3x2 plot with the top row face-on and the bottom row edge-on. """
    massBins  = [[13.5,13.8],[13.0,13.1],[12.5,12.6],[12.0,12.1],[11.5,11.6]]
    numPerBin = 30

    assert haloNum is not None and haloNum < numPerBin*len(massBins)

    panels = []

    res        = 1820
    redshift   = 0.5
    run        = 'tng'
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [960,960]
    sizeFac    = -120.0 # 120 ckpc/h
    hsmlFac    = 2.5
    axes       = [1,0]

    class plotConfig:
        plotStyle = 'open_black'
        rasterPx  = 1400
        colorbars = True

    # configure panels
    starsMM = [6.5,10.0] # coldens_msunkpc2
    gasMM   = [6.5,8.0]

    if conf == 'single_halos':
        # loop over centrals in mass bins, one figure each
        sP = simParams(res=res, run=run, redshift=redshift)

        hID, binInd = selectHalosFromMassBin(sP, massBins, numPerBin, haloNum=haloNum)

        if hID is None:
            print('Task past bin size, quitting.')
            return

        if binInd >= 4: sizeFac = -60.0

        panels.append( {'hInd':hID, 'hsmlFac':1.0, 'rotation':'face-on', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':starsMM, 'labelHalo':True} )
        panels.append( {'hInd':hID, 'rotation':'face-on', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
        panels.append( {'hInd':hID, 'rotation':'face-on', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-0.3,0.3]} )
        panels.append( {'hInd':hID, 'hsmlFac':1.0, 'rotation':'edge-on', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':starsMM, 'labelScale':True} )
        panels.append( {'hInd':hID, 'rotation':'edge-on', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
        panels.append( {'hInd':hID, 'rotation':'edge-on', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-0.3,0.3]} )

        plotConfig.saveFilename = saveBasePath + 'renderHalo_%s-%d_bin%d_halo%d_hID-%d_shID-%d.pdf' % \
                                  (sP.simName,sP.snap,binInd,haloNum,haloInd,hID)

    if conf == 'halos_combined':
        # combined plot of centrals in mass bins
        sP = simParams(res=res, run=run, redshift=redshift)

        hIDs, binInd = selectHalosFromMassBin(sP, massBins, numPerBin, massBinInd=haloNum)

        if binInd >= 4: sizeFac = -60.0

        for hID in hIDs:
            if panelNum == 0:
                panels.append( {'hInd':hID, 'hsmlFac':1.0, 'rotation':'face-on', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':starsMM} )
            if panelNum == 1:
                panels.append( {'hInd':hID, 'rotation':'face-on', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
            if panelNum == 2: 
                panels.append( {'hInd':hID, 'rotation':'face-on', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-0.3,0.3]} )
            if panelNum == 3: 
                panels.append( {'hInd':hID, 'hsmlFac':1.0, 'rotation':'edge-on', 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':starsMM} )
            if panelNum == 4: 
                panels.append( {'hInd':hID, 'rotation':'edge-on', 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
            if panelNum == 5: 
                panels.append( {'hInd':hID, 'rotation':'edge-on', 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-0.3,0.3]} )
            if panelNum == 6: 
                panels.append( {'hInd':hID, 'hsmlFac':1.0, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':starsMM} )
            if panelNum == 7: 
                panels.append( {'hInd':hID, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )

        plotConfig.saveFilename = saveBasePath + 'renderHalo_%s-%d_bin%d_%s-%s.pdf' % \
                                  (sP.simName,sP.snap,binInd,panels[0]['partType'],panels[0]['partField'])

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def loopMassBins(run=None):
    """ Call another driver function several times, looping over a possible input. """
    for i in range(30*5):
        massBinsSample_3x2_EdgeOnFaceOn('single_halos', haloOrMassBinNum=i)
    for i in range(5):
        massBinsSample_3x2_EdgeOnFaceOn('halos_combined', haloOrMassBinNum=i, panelNum=run)

def zoomEvoMovies(conf):
    """ Configurations to render movies of the sims.zooms2 runs (at ~400 total snapshots). """
    panels = []

    if conf == 'oneRes_DensTempEntr':
        panels.append( {'res':11, 'partField':'coldens', 'valMinMax':[19.0,23.0], 'labelScale':True} )
        panels.append( {'res':11, 'partField':'temp',    'valMinMax':[4.0, 6.5]} )
        panels.append( {'res':11, 'partField':'entr',    'valMinMax':[6.0,9.0], 'labelHalo':True} )

    if conf == 'threeRes_DensTemp':
        panels.append( {'res':9,  'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'res':10, 'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'res':11, 'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'res':9,  'partField':'temp',    'valMinMax':[4.0,6.5]} )
        panels.append( {'res':10, 'partField':'temp',    'valMinMax':[4.0,6.5]} )
        panels.append( {'res':11, 'partField':'temp',    'valMinMax':[4.0,6.5]} )

    hInd       = 2
    run        = 'zooms2'
    partType   = 'gas'
    rVirFracs  = [0.15,0.5,1.0]
    method     = 'sphMap'
    nPixels    = [1920,1920]
    sizeFac    = 3.5
    hsmlFac    = 2.5
    axes       = [1,0]
    labelSim   = False
    relCoords  = True
    rotation   = None

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = 1200
        colorbars    = True
        saveFileBase = '%s_evo_h%d_%s' % (run,hInd,conf)

        # movie config
        treeRedshift = 2.0
        minRedshift  = 2.0
        maxRedshift  = 100.0

    renderSingleHaloFrames(panels, plotConfig, localVars)
