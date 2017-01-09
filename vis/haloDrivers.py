"""
haloDrivers.py
  Render specific halo visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import savePathDefault
from vis.halo import renderSingleHalo, renderSingleHaloFrames, selectHalosFromMassBin
from util.helper import pSplit
from cosmo.load import groupCat, groupCatSingle
from cosmo.util import snapNumToRedshift, redshiftToSnapNum
from util import simParams

def oneHaloPressureCompAndRatios(shID=0):
    """ In 3x1 panels compare gas and magnetic pressures and their ratio. """
    panels = []

    panels.append( {'hInd':shID, 'partField':'P_B', 'valMinMax':[1.0,9.0]} )
    panels.append( {'hInd':shID, 'partField':'P_gas', 'valMinMax':[1.0,9.0]} )
    panels.append( {'hInd':shID, 'partField':'pressure_ratio', 'valMinMax':[-4.0,0.0]} )

    run        = 'tng'
    res        = 1820
    redshift   = 0.0
    partType   = 'gas'
    rVirFracs  = [1.0]
    method     = 'sphMap'
    nPixels    = [960,960]
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
        plotStyle    = 'open'
        rasterPx     = 960
        colorbars    = True
        saveFilename = './gasPressureComp_%s_%d_z%.1f_subhalo-%d.pdf' % (run,res,redshift,shID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def oneHaloGaussProposal():
    """ Render single halo with B field streamlines for Gauss proposal (MHD figure). """
    shID = 4
    panels = []

    #panels.append( {'hInd':shID, 'partField':'bmag_uG', 'valMinMax':[-2.0,1.0]} )
    panels.append( {'hInd':shID, 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,7.2]} )
    #panels.append( {'hInd':shID, 'partField':'bfield_x', 'valMinMax':[-1e-2,1e-2]} )
    #panels.append( {'hInd':shID, 'partField':'bfield_y', 'valMinMax':[-1e-2,1e-2]} )

    run        = 'tng'
    res        = 1820
    redshift   = 0.0
    partType   = 'gas'
    rVirFracs  = [1.0]
    method     = 'sphMap'
    nPixels    = [960,960]
    sizeFac    = 0.3 # central object
    vecOverlay = True # experimental B field streamlines
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
        plotStyle    = 'open'
        rasterPx     = 960
        colorbars    = True
        saveFilename = './gasDens_%s_%d_z%.1f_sh-%d.pdf' % (run,res,redshift,shID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def resSeriesGaussProposal(fofInputID=12, resInput=256):
    """ Render a 3x2 panel resolution series for Gauss proposal (MW/structural fig). """
    panels = []

    run = 'tng'
    variant = 'L12.5'
    fofHaloIDs = [fofInputID]
    resLevels  = [resInput]

    #run = 'illustris'
    #fofHaloIDs = [196] # subhalo ID 283832
    #resLevels = [1820]

    valMinMaxG  = [6.0,8.2]
    valMinMaxS  = [6.0,9.0]

    redshift   = 0.0
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [960,960]
    sizeFac    = -80.0 #0.3 # central object
    axes       = [0,1]
    labelZ     = False
    labelScale = True
    labelSim   = False
    labelHalo  = False
    relCoords  = True
    mpb        = None

    for fofHaloID, resLevel in zip(fofHaloIDs,resLevels):
        # get subhalo ID
        sP = simParams(res=resLevel, run=run, redshift=redshift, variant=variant)
        h = groupCatSingle(sP, haloID=fofHaloID)
        shID = h['GroupFirstSub']
        print('subhalo ID: ',shID)

        # append some panels
        pF = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'

        #panels.append( {'partType':'stars', 'hsmlFac':0.5, \
        #                'partField':pF, 'res':resLevel, 'hInd':shID, sizeFac:-50.0} )
        panels.append( {'partType':'stars', 'hsmlFac':0.5, 'rotation':'face-on-I', \
                        'partField':pF, 'res':resLevel, 'hInd':shID, 'sizeFac':-50.0} )
        panels.append( {'partType':'gas', 'hsmlFac':2.5, 'partField':'coldens_msunkpc2', \
                        'rotation':'face-on-I', 'res':resLevel, 'hInd':shID, 'valMinMax':valMinMaxG} )

        panels.append( {'partType':'stars', 'hsmlFac':0.5, 'rotation':'edge-on-I', 'nPixels':[960,320], \
                        'partField':pF, 'res':resLevel, 'hInd':shID, 'sizeFac':-50.0} )
        panels.append( {'partType':'gas', 'hsmlFac':2.5, 'partField':'coldens_msunkpc2', 'nPixels':[960,320], \
                        'rotation':'edge-on-I', 'res':resLevel, 'hInd':shID, 'valMinMax':valMinMaxG} )
        #panels.append( {'partType':'gas', 'hsmlFac':2.5, 'partField':'coldens_msunkpc2', \
        #                'res':resLevel, 'hInd':shID, 'valMinMax':valMinMaxG} )

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = 960
        colorbars    = True
        haloStr      = '-'.join([str(r) for r in fofHaloIDs])
        resStr       = '-'.join([str(r) for r in resLevels])
        saveFilename = './resSeriesGauss_%s_%s_%s.pdf' % (run,haloStr,resStr)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def helperLoop():
    for i in range(20):
        resSeriesGaussProposal(i,resInput=256)
    for i in range(20):
        resSeriesGaussProposal(i,resInput=512)

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
    partField  = 'Si II' #'O VI' #'HI_segmented'
    valMinMax  = [14.0,17.0] #[13.5, 15.5] #[13.5,21.5]
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
        saveFilename = savePathDefault + 'sample_%s_page-%d-of-%d_%s_%d.pdf' % \
                       (partField,curPageNum,numPages,run,res)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def loopInputSerial():
    """ Call another driver function several times, looping over a possible input. """
    numPages = 7
    for i in range(numPages):
        multiHalosPagedOneQuantity(i)

def boxHalo_HI():
    """ Single halo HI plots (col dens, line of sight velocity) with smoothing. """
    panels = []

    vmm_col = [13.5,21.5] # 1/cm^2
    vmm_vel = [-300,300] # km/s

    # smoothing
    #panels.append( {'smoothFWHM':None, 'partField':'HI_segmented', 'valMinMax':vmm_col, 'labelScale':True} )
    #panels.append( {'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    #panels.append( {'smoothFWHM':6.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    #panels.append( {'smoothFWHM':None, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #panels.append( {'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #panels.append( {'smoothFWHM':6.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #axes = [0,1]

    # rotations
    #panels.append( {'rotation':None, 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col, 'labelScale':True} )
    #panels.append( {'rotation':'edge-on', 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    #panels.append( {'rotation':'face-on', 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    #panels.append( {'rotation':None, 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #panels.append( {'rotation':'edge-on', 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #panels.append( {'rotation':'face-on', 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #sizeFac = 2.5

    # proposed
    panels.append( {'rotation':None, 'sizeFac':2.5, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'edge-on', 'sizeFac':-120.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'face-on', 'sizeFac':-120.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':None, 'sizeFac':2.5, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'edge-on', 'sizeFac':-120.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    panels.append( {'rotation':'face-on', 'sizeFac':-120.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    #smoothFWHM = 2.0
    labelScale = True

    hInd       = 362540
    run        = 'illustris'
    partType   = 'gas'
    res        = 1820
    redshift   = 0.0
    rVirFracs  = [1.0] # None
    method     = 'sphMap'
    nPixels    = [960,960]
    hsmlFac    = 3.0
    rotation   = None

    class plotConfig:
        plotStyle    = 'open'
        colorbars    = True
        rasterPx     = 960
        saveFilename = savePathDefault + 'fig5b_%s_%d_z%.1f_shID-%d.pdf' % \
                       (run,res,redshift,hInd)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def boxHalo_MultiQuant():
    """ Diagnostic plot, a few quantities of a halo from a periodic box. """
    panels = []

    #panels.append( {'hsmlFac':1.0, 'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,7.5]} )
    #panels.append( {'hsmlFac':1.0, 'partType':'dm', 'partField':'coldens2_msunkpc2', 'valMinMax':[12.0,14.0]} 

    panels.append( {'rotation':'edge-on','hsmlFac':2.5, 'nPixels':[960,960],  'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.5,8.4]} )
    panels.append( {'rotation':'edge-on','hsmlFac':1.0, 'nPixels':[960,960],  'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[7.5,9.0]} )
    panels.append( {'rotation':'edge-on','hsmlFac':1.0, 'nPixels':[1920,1920],'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[7.5,9.0]} )
    panels.append( {'rotation':'face-on','hsmlFac':2.5, 'nPixels':[960,960],  'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.5,8.4]} )
    panels.append( {'rotation':'face-on','hsmlFac':1.0, 'nPixels':[960,960],  'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[7.5,9.0]} )
    panels.append( {'rotation':'face-on','hsmlFac':1.0, 'nPixels':[1920,1920],'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[7.5,9.0]} )

    #panels.append( {'hsmlFac':2.5, 'nPixels':[960,960],  'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,8.4]} )
    #panels.append( {'hsmlFac':1.0, 'nPixels':[960,960],  'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} )
    #panels.append( {'hsmlFac':1.0, 'nPixels':[1920,1920],'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} )

    hInd       = 362540
    run        = 'illustris'
    res        = 1820
    redshift   = 0.0
    rVirFracs  = [1.0] # None
    method     = 'sphMap'
    #nPixels    = [1920,1920]
    sizeFac    = 2.5 #-50.0
    #hsmlFac    = 2.5
    #axes       = [1,2]
    #rotation   = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = savePathDefault + 'box_%s_%d_z%.1f_shID-%d_multiQuant_sf-%.1f.pdf' % (run,res,redshift,hInd,sizeFac)

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
        saveFilename = savePathDefault + '%s_h%dL%d_z%.1f_multiQuant.pdf' % (run,hInd,res,redshift)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def tngDwarf_firstNhalos(conf=0):
    """ Plot gas/stellar densities of centers of N most massive L35n2160 halos (at some redshift).
    All separate (fullpage) plots. """
    run      = 'tng'
    res      = 2160
    redshift = 6.0
    nHalos   = 10

    rVirFracs  = [0.1]
    method     = 'sphMap'
    nPixels    = [1000,1000]
    sizeFac    = -100.0 # 100 ckpc
    hsmlFac    = 2.5
    labelZ     = True
    labelScale = True
    labelHalo  = True
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
        #hsmlFac    = 1.0 # when real CalcHsml() is working again

    sP = simParams(res=res, run=run, redshift=redshift)

    # render one plot per halo
    for i in range(nHalos):
        halo = groupCatSingle(sP, haloID=i)
        print(sP.simName, sP.snap, sP.redshift, i, halo['GroupFirstSub'], halo['GroupPos'])

        panels = [ {'hInd':halo['GroupFirstSub']} ]

        class plotConfig:
            plotStyle    = 'open_black'
            colorbars    = True
            rasterPx     = 1000
            saveFilename = savePathDefault + '%s_haloInd-%d_%s-%s_z%.1f.pdf' % \
                           (sP.simName,i,partType,partField,redshift)

        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def tngCluster_center_timeSeriesPanels(conf=0):
    """ Plot a time series of panels from subsequent snapshots in the center of fof0. """
    panels = []

    zStart     = 0.3 # start plotting at this snapshot
    nSnapsBack = 12   # one panel per snapshot, back in time

    #hInd       = 0
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
        hsmlFac    = 1.0
    if conf == 2:
        partType   = 'gas'
        partField  = 'metal_solar'
        valMinMax  = [-0.5,0.5]
    if conf == 3:
        partType   = 'dm'
        partField  = 'coldens2_msunkpc2'
        valMinMax  = [15.0,16.0]
        hsmlFac    = 1.0
    if conf == 4:
        partType   = 'gas'
        partField  = 'pressure_ratio'
        valMinMax  = [-2.0,1.0]

    # configure panels
    sP = simParams(res=res, run=run, redshift=zStart)
    for i in range(nSnapsBack):
        hIndLoc = 0
        if run == 'tng' and i < 2: hIndLoc = 1

        halo = groupCatSingle(sP, haloID=hIndLoc)
        print(sP.snap, sP.redshift, hIndLoc, halo['GroupFirstSub'], halo['GroupPos'])

        panels.append( {'hInd':halo['GroupFirstSub'], 'redshift':snapNumToRedshift(sP)} )
        sP.setSnap(sP.snap-1)

    panels[0]['labelScale'] = True
    panels[-1]['labelHalo'] = True

    class plotConfig:
        plotStyle    = 'edged_black'
        colorbars    = True
        rasterPx     = 960
        saveFilename = savePathDefault + 'timePanels_%s_hInd-0_%s-%s_z%.1f_n%d.pdf' % \
                       (sP.simName,partType,partField,zStart,nSnapsBack)

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

        plotConfig.saveFilename = savePathDefault + 'renderHalo_%s-%d_bin%d_halo%d_hID-%d_shID-%d.pdf' % \
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

        plotConfig.saveFilename = savePathDefault + 'renderHalo_%s-%d_bin%d_%s-%s.pdf' % \
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
