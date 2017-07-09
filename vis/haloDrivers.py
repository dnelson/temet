"""
haloDrivers.py
  Render specific halo visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from datetime import datetime

from vis.common import savePathDefault
from vis.halo import renderSingleHalo, renderSingleHaloFrames, selectHalosFromMassBin
from plot.general import simSubhaloQuantity
from util.helper import pSplit
from cosmo.load import groupCat, groupCatSingle, auxCat
from cosmo.mergertree import loadMPB
from cosmo.util import snapNumToRedshift, redshiftToSnapNum, crossMatchSubhalosBetweenRuns
from util import simParams

def oneHaloSingleField(conf=0, shID=0):
    """ In a single (or a few) panel(s) centered on a halo, show one field from the box. """
    panels = []

    if conf == 0:
        # magnetic pressure, gas pressure, and their ratio
        panels.append( {'partType':'gas', 'hInd':shID, 'partField':'P_B', 'valMinMax':[1.0,9.0]} )
        panels.append( {'partType':'gas', 'hInd':shID, 'partField':'P_gas', 'valMinMax':[1.0,9.0]} )
        panels.append( {'partType':'gas', 'hInd':shID, 'partField':'pressure_ratio', 'valMinMax':[-4.0,0.0]} )
        hsmlFac = 2.5
    if conf == 1:
        # stellar mass column density
        panels.append( {'partType':'stars',  'partField':'coldens_msunkpc2', 'valMinMax':[4.8,7.8]} )
        hsmlFac = 0.5

    run        = 'tng'
    res        = 1820
    redshift   = 0.0
    rVirFracs  = [1.0]
    method     = 'sphMap'
    nPixels    = [3840,3840]
    axes       = [1,0]
    labelZ     = False
    labelScale = False
    labelSim   = False
    labelHalo  = False
    relCoords  = True
    rotation   = None
    mpb        = None

    if 0:
        size = 2.5
        sizeType = 'rVirial'
    if 1:
        size = 3000.0
        sizeType = 'pkpc'

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = nPixels[0]
        colorbars    = False
        saveFilename = './oneHaloSingleField_%d_%s_%d_z%.1f_subhalo-%d.png' % (conf,run,res,redshift,shID)

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
    size       = 0.3 # central object
    sizeType   = 'rVirial'
    hsmlFac    = 2.5
    axes       = [1,0]
    labelZ     = False
    labelScale = False
    labelSim   = False
    labelHalo  = False
    relCoords  = True
    rotation   = None
    mpb        = None

    vecOverlay = 'bfield' # experimental B field streamlines
    vecMethod  = 'E' # colored streamlines, uniform thickness
    vecColorPT = 'gas'
    vecColorPF = 'bmag'

    class plotConfig:
        plotStyle    = 'open'
        rasterPx     = 960
        colorbars    = True
        saveFilename = './gasDens_%s_%d_z%.1f_sh-%d.pdf' % (run,res,redshift,shID)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)

def oneGalaxyThreeRotations(conf=0):
    """ Plot stellar stamps of N random massive L25n512_0000 galaxies. 
    If curPage,numPages both specified, do a paged exploration instead. """
    run        = 'illustris' 
    res        = 1820
    redshift   = 0.0
    hInd       = 283832 # subhalo ID
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [700,700]
    axes       = [0,1]
    labelZ     = False
    labelSim   = False
    labelHalo  = True
    relCoords  = True
    mpb        = None
    rotations  = [None, 'face-on', 'edge-on']

    #size       = 70.0 # 140 ckpc/h size length
    #sizeType   = 'codeUnits'
    size       = 5.0
    sizeType   = 'rHalfMassStars'

    if conf == 0:
        partType  = 'stars'
        partField = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'
        hsmlFac   = 0.5

    if conf == 1:
        partType  = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [6.0, 8.2]
        hsmlFac   = 2.5

    # create panels, one per view
    sPloc = simParams(res=res, run=run, redshift=redshift)
    sub   = groupCatSingle(sPloc, subhaloID=hInd)
    panels = []

    for i, rot in enumerate(rotations):
        labelScaleLoc = True if i == 0 else False
        panels.append( {'rotation':rot, 'labelScale':labelScaleLoc} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 700
        colorbars    = False
        saveFilename = './fig_galaxy_%s_%d_shID=%d_%s.pdf' % \
          (sPloc.simName,sPloc.snap,hInd,partType)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

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
    size       = 80.0 #0.3 # central object
    sizeType   = 'codeUnits'
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

        # which stellar composite to use
        pF_stars = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'

        # append panels
        panels.append( {'partType':'stars', 'hsmlFac':0.5, 'partField':pF_stars, 'rotation':'face-on', \
                        'res':resLevel, 'hInd':shID, 'size':50.0, 'sizeType':'codeUnits'} )
        panels.append( {'partType':'gas', 'hsmlFac':2.5, 'partField':'coldens_msunkpc2', \
                        'rotation':'face-on', 'res':resLevel, 'hInd':shID, 'valMinMax':valMinMaxG} )

        #'nPixels':[960,320], \ # reduce vertical size of edge-on panels
        panels.append( {'partType':'stars', 'hsmlFac':0.5, 'partField':pF_stars, 'rotation':'edge-on', \
                        'res':resLevel, 'hInd':shID, 'size':50.0, 'sizeType':'codeUnits'} )
        panels.append( {'partType':'gas', 'hsmlFac':2.5, 'partField':'coldens_msunkpc2', \
                        'rotation':'edge-on', 'res':resLevel, 'hInd':shID, 'valMinMax':valMinMaxG} )

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
        resSeriesGaussProposal(i,resInput=256) #fof=12
    for i in range(20):
        resSeriesGaussProposal(i,resInput=512) #fof=10

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
    size       = 140.0 # size = 10^2.8 * 0.7 * 2 (match to pm2.0 for Guinevere)
    sizeType   = 'codeUnits'
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
    #size = 2.5

    # proposed
    panels.append( {'rotation':None, 'size':2.5, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'edge-on', 'size':120.0, 'sizeType':'codeUnits', 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'face-on', 'size':120.0, 'sizeType':'codeUnits', 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':None, 'size':2.5, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    panels.append( {'rotation':'edge-on', 'size':120.0, 'sizeType':'codeUnits', 'partField':'vel_los', 'valMinMax':vmm_vel} )
    panels.append( {'rotation':'face-on', 'size':120.0, 'sizeType':'codeUnits', 'partField':'vel_los', 'valMinMax':vmm_vel} )
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
    size       = 2.5 #-50.0
    sizeType   = 'rVirial'
    #hsmlFac    = 2.5
    #axes       = [1,2]
    #rotation   = None

    class plotConfig:
        plotStyle    = 'open_black'
        colorbars    = True
        saveFilename = savePathDefault + 'box_%s_%d_z%.1f_shID-%d_multiQuant_sf-%.1f.pdf' % (run,res,redshift,hInd,size)

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
    size       = 2.5
    sizeType   = 'rVirial'
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
    size       = 100.0
    sizeType   = 'codeUnits'
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

def tngMethods2_stamps(conf=0, curPage=None, numPages=None, rotation=None, 
                       variant=None, matchedToVariant=None):
    """ Plot stellar stamps of N random massive L25n512_0000 galaxies. 
    If curPage,numPages both specified, do a paged exploration instead. 
    If matchedToVariant is not None, then run the halo selection on this variant instead (e.g. always 0000) 
    and then use the SubhaloMatching catalog to pick the matched halos in this run. """
    run       = 'tng'
    res       = 1024
    redshift  = 3.0
    #variant   = 4503 #0010 #0000
    massBin   = [12.0, 14.0]
    nGalaxies = 15
    selType   = 'random'

    # stellar composite, 50 kpc/h on a side, include M* label per panel, and scale bar once
    # select random in [12.0,inf] halo mass range, do each of: face-on, edge-on, random(z)
    # possible panel configurations: 4x2 = 8, 4x3 = 12, 5x3 = 15, 6x3 = 18
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [700,700]
    axes       = [0,1]
    labelZ     = False
    labelSim   = False
    labelHalo  = 'Mstar'
    relCoords  = True
    mpb        = None

    size      = 50.0 # 25 ckpc/h each direction
    sizeType  = 'codeUnits'
    #size       = 6.0
    #sizeType   = 'rHalfMassStars'

    if conf == 0:
        partType = 'stars'
        partField = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'
        #partField = 'stellarComp-snap_K-snap_B-snap_U'
        #partField = 'stellarComp-sdss_i-sdss_r-sdss_g' #irg # zirgu (red -> blue)
        hsmlFac = 0.5

    if conf == 1:
        partType = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [6.8, 8.8] # for z=3 1024 # [6.0, 8.2] for z=2 512
        hsmlFac = 2.5

    # load halos of this bin, from this run or from the matchedToVariant source run
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    sP_from = sP
    mvStr = ''

    if matchedToVariant is not None:
        sP_from = simParams(res=res, run=run, redshift=redshift, variant=matchedToVariant)
        mvStr = '_matched-to-%s' % sP_from.simName
    
    if curPage is None:
        # non-paged, load requested number
        shIDs, _ = selectHalosFromMassBin(sP_from, [massBin], nGalaxies, massBinInd=0, selType=selType)
        saveFilename2 = './methods2_stamps_%s_%s_rot=%s%s.pdf' % \
          (sP.simName,partType,rotation,mvStr)
    else:
        # paged, load all and sub-divide
        if sP.res == 512: massBin = [11.8,14.0] # methods2 paper
        if sP.res == 1024: massBin = [11.3, 14.0]
        shIDsAll, _ = selectHalosFromMassBin(sP_from, [massBin], 200, massBinInd=0, selType='linear')
        shIDs = pSplit(shIDsAll, numPages, curPage)[0:nGalaxies]
        assert shIDs.size == nGalaxies # make sure we have nGalaxies per render
        saveFilename2 = './methods2_pages_%s_%s_rot=%s%s_page-%dof%d.pdf' % \
          (sP.simName,partType,rotation,mvStr,curPage,numPages)

    # if we loaded the subhalo list to plot from another run, match them to the current run
    if matchedToVariant is not None:
        shIDs = crossMatchSubhalosBetweenRuns(sP_from, sP, shIDs)
        assert shIDs.min() >= 0 # if any matches failed, we should make a blank panel

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        labelScaleLoc = True if (i == 0 or sizeType == 'rHalfMassStars') else False
        panels.append( {'hInd':shID, 'labelScale':labelScaleLoc} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 500
        colorbars    = False
        saveFilename = saveFilename2

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def loop_stamps():
    """ Helper. """
    numPages = 10

    # plot some other run choosing its subhalos matched to the 0000 selection
    variant = 4503 # 0010 for methods2
    matchedToVariant = 0000

    for conf in [0,1]:
        for rotation in [None,'edge-on','face-on']:
            for curPage in range(numPages):
                print(conf,rotation,curPage,numPages)
                tngMethods2_stamps(conf=conf, curPage=curPage, numPages=numPages, 
                                   rotation=rotation, variant=variant, matchedToVariant=matchedToVariant)

    # plot the 0000 selection itself
    variant = 0000
    matchedToVariant = None

    for conf in [0,1]:
        for rotation in [None,'edge-on','face-on']:
            for curPage in range(numPages):
                print(conf,rotation,curPage,numPages)
                tngMethods2_stamps(conf=conf, curPage=curPage, numPages=numPages, 
                                   rotation=rotation, variant=variant, matchedToVariant=matchedToVariant)

def tngMethods2_windPatterns(conf=1, pageNum=0):
    """ Plot gas streamlines (galaxy wind patterns), 4x2, top four from L25n512_0000 and bottom four 
    from L25n512_0010 (Illustris model), matched. """
    # change to: barAreaHeight = np.max([0.035,0.14 / nRows]) if conf.colorbars else 0.0
    # change to: if sP.isPartType(partType,'gas'):   config['ctName'] = 'perula' #'magma'
    run       = 'tng'
    res       = 512
    variant   = 0000 # TNG fiducial
    matchedToVariant = 0010 # Illustris fiducial

    # stellar composite, 50 kpc/h on a side, include M* label per panel, and scale bar once
    redshift   = 2.0
    rVirFracs  = None
    method     = 'sphMap'
    nPixels    = [700,700]
    axes       = [0,1]
    labelZ     = False
    labelSim   = False
    labelHalo  = 'Mstar'
    relCoords  = True
    mpb        = None
    rotation   = 'edge-on'

    vecOverlay  = 'gas_vel' # experimental gas (x,y) velocity streamlines
    vecMinMax   = [0,450] # range for streamlines color scaling and colorbar
    vecColorbar = True
    vecMethod   = 'E' # colored streamlines, uniform thickness
    vecColorPT  = 'gas'
    vecColorPF  = 'vmag'

    size      = 50.0 # [50,80,120] --> 25,40,60 ckpc/h each direction
    sizeType  = 'codeUnits'

    if conf == 0:
        partType = 'stars'
        partField = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'
        hsmlFac = 0.5

    if conf == 1:
        partType = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [7.2, 8.6]
        hsmlFac = 2.5

    # pick halos from this run and crossmatch
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)
    sP2 = simParams(res=res, run=run, redshift=redshift, variant=matchedToVariant)
    
    # z2_11.5_page (z=2)
    #selectHalosFromMassBin(): In massBin [11.5 11.7] have 85 halos total.
    #pages = [[3498, 3833, 3861, 4097], [4250, 4481, 4511, 4578], [4601, 4656, 4720, 4763], 
    #         [4882, 4898, 4913, 4928], [4952, 4985, 5004, 5023], [5037, 5049, 5062, 5077],
    #         [5133, 5154, 5173, 5186], [5202, 5220, 5231, 5240], [5278, 5295, 5310, 5323],
    #         [5396, 5441, 5459, 5469], [5482, 5491, 5508, 5520], [5533, 5546, 5558, 5584]]

    # z2 selections from above
    pages = [[3498, 4250, 5396, 5173], [4481, 4656, 5482, 5546]]

    shIDs = pages[pageNum]

    # todo: change to manual ID selection below
    #nGalaxies = 4
    #shIDs, _ = selectHalosFromMassBin(sP, [[11.5,11.7]], nGalaxies, massBinInd=0, selType='linear')
    #import pdb; pdb.set_trace()

    # crossmatch to other run
    shIDs2 = crossMatchSubhalosBetweenRuns(sP, sP2, shIDs)
    assert shIDs2.min() >= 0 # if any matches failed, we should make a blank panel

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        labelScaleLoc = True if i == 0 else False
        panels.append( {'hInd':shID, 'labelScale':labelScaleLoc, 'variant':variant} )
    for i, shID in enumerate(shIDs2):
        panels.append( {'hInd':shID, 'labelScale':labelScaleLoc, 'variant':matchedToVariant} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 700
        colorbars    = True
        saveFilename = './methods2_gasflows_z2_final_page-%d_%s-%s_%s-%s_%s_%dckpch.pdf' % (pageNum,sP.simName,matchedToVariant,partType,partField,rotation,size)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def loop_patterns():
    for i in range(2):
        print(i)
        tngMethods2_windPatterns(conf=1, pageNum=i)

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
    size       = 100.0
    sizeType   = 'codeUnits'
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
    size       = 120.0
    sizeType   = 'codeUnits'
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

        if binInd >= 4: size = 60.0

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

        if binInd >= 4: size = 60.0

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
    size       = 3.5
    sizeType   = 'rVirial'
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

def tngFlagship_galaxyStellarRedBlue(blueSample=False, redSample=False, greenSample=False, 
                                     evo=False, curPage=None, conf=0):
    """ Plot stellar stamps red/blue galaxies around 10^10.5 Msun.
    If evo==True, then tracked back in time from z=0 to z=2.0 in M steps using the merger tree.
    If evo==False, then show full NxM panel sample at z=0.
    In either case, choose blueSample, redSample, or greenSample.
    If curPage specified, do a paged exploration instead. """
    from cosmo.color import loadSimGalColors
    from plot.config import defSimColorModel
    from cosmo.util import redshiftToSnapNum

    # we have chosen by hand for L75n1820TNG z=0 from the massBin = [12.0,12.2] below these two sets
    # we define blue as (g-r)<0.6, red as (g-r)>0.7 and green as 0.5<(g-r)<0.7
    blue_z0 = [438297,448732,446548,452577,455335,
               463062,463649,464576,460692,468590,
               466436,469315,470617,472457,471740,
               473004,473349,473898,474813,476487,
               477179,477677,479174,479117,480230,
               480441,481126,481300,482169,483370,
               482869,484771,485441,486590,487244, # 0-35
               486525,486966,488415,486781,488123,
               488500,488722,489054,489100,489593,
               490280,490806,491282,493034,493751,
               493491,493517,493964,494771,494273,
               497032,496990,495515,497646,498576,
               499130,499996,499223,499463,500867,
               500494,501761,502312,502648,502919] # 35-70
    red_z0  = [441141,443914,453835,421956,440477,
               497926,491801,460076,496436,475490,
               498958,451243,478160,479314,479917,
               502881,461038,467519,469589,473125,
               481347,482257,505333,483868,482533,
               484113,484257,480645,485365,486052,
               487152,489314,488841,508985,510751, # 0-35
               490986,469102,492392,492614,493230,
               494009,495442,471857,497800,499025,
               499522,500448,502168,502956,480194,
               502461,503393,504142,482714,507070,
               508985,515318,517881,511005,490577,
               454963,463139,477518,469930,480550,
               480879,480750,481254,483900,485233] # 35-70

    green_z0 = [415972, 418076, 429041, 429758, 438297, 438453, 445185, 452357,
               454963, 455058, 459394, 459517, 460008, 460526, 461283, 462010,
               462141, 463139, 463958, 464669, 466801, 467127, 467445, 467519,
               468318, 468450, 468508, 468590, 468695, 469102, 469740, 469930,
               470679, 471109, 471591, 471857, 472418, 472747, 473349, 474271,
               474437, 474860, 475542, 475722, 476487, 476892, 477031, 477474,
               477518, 477624, 478452, 478661, 478927, 479411, 479798, 479839,
               480194, 480550, 480750, 480803, 480879, 481254, 481804, 482006,
               482054, 482169, 482196, 482714, 482757, 482869, 482987, 483179,
               483421, 483484, 483900, 484207, 484307, 484715, 484944, 485233,
               485283, 485326, 485485, 486124, 486186, 486281, 486875, 487042,
               487152, 487698, 487913, 487965, 488228, 489273, 489356, 489666,
               489812, 489953, 490109, 490172, 490195, 490238, 490475, 490533,
               490617, 490661, 490711, 491066, 491282, 491872, 491901, 492092,
               492125, 492178, 492223, 492947, 493183, 493437, 493575, 493658,
               493751, 494326, 495442, 495772, 496019, 496244, 496292, 496326,
               496369, 496757, 496805, 497088, 497176, 497351, 497585, 497964,
               498031, 498297, 499025, 499113, 499463, 499636, 500239, 500268,
               500294, 500448, 500725, 501430, 501526, 501604, 501658, 502347,
               502427, 502595, 502767, 502956, 503393, 503636, 503778, 503951,
               504000, 504030, 504065, 505793, 505892, 505928, 506063, 506313,
               506943, 507299, 507361, 507587, 507701, 508156, 509018, 510085,
               511576, 512330, 514988] # 179

    # config
    run           = 'tng'
    res           = 1820 
    rVirFracs     = None
    method        = 'sphMap'
    nPixels       = [300,300]
    axes          = [0,1]
    labelZ        = False
    labelSim      = False
    labelHalo     = 'Mstar'
    relCoords     = True
    mpb           = None
    rotation      = 'face-on'
    sizeType      = 'pkpc'

    if conf == 0:
        size          = 60.0 # 30 kpc in each direction from center
        partType      = 'stars'
        partField     = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'
        hsmlFac       = 0.5
    if conf == 1:
        size          = 800.0 # 400 kpc in each direction from center
        partType      = 'gas'
        partField     = 'O VI'
        hsmlFac       = 3.5
        valMinMax     = [12.0,15.0]
        rVirFracs     = [1.0]

    evo_redshifts = [0.0, 0.2, 0.4, 0.7, 1.0] # [0.0, 0.2, 0.5, 1.0, 2.0]
    redshift_init = 0.0
    nGalaxies     = 45 # 35 (note: set to 45 for talk image)
    nRowsFig      = 9 # 7 (note: set to 9 for talk image)

    # load halos of this bin, from this run
    sP = simParams(res=res, run=run, redshift=redshift_init)

    if curPage is None:
        if evo is False:
            # z=0 samples
            if redSample: shIDs = red_z0[(redSample-1)*nGalaxies:(redSample)*nGalaxies]
            if blueSample: shIDs = blue_z0[(blueSample-1)*nGalaxies:(blueSample)*nGalaxies]
            if greenSample: shIDs = green_z0[(greenSample-1)*nGalaxies:(greenSample)*nGalaxies]

            redshifts = np.zeros(len(shIDs), dtype='float32') + redshift_init
        else:
            # take nRows galaxies from z0 samples, then load their MPBs and get IDs at earlier redshifts
            assert redSample > 0 or blueSample > 0 or greenSample > 0
            if redSample: shIDs_z0 = red_z0[(redSample-1)*nRowsFig:(redSample)*nRowsFig]
            if blueSample: shIDs_z0 = blue_z0[(blueSample-1)*nRowsFig:(blueSample)*nRowsFig]
            if greenSample: shIDs_z0 = green_z0[(greenSample-1)*nRowsFig:(greenSample)*nRowsFig]

            evo_snapshots = redshiftToSnapNum(evo_redshifts, sP)
            shIDs = []
            redshifts = []

            for shID_z0 in shIDs_z0:
                # load main progenitor branch
                mpbLocal = loadMPB(sP, id=shID_z0)

                # append to shIDs and redshifts the 5 evolution steps for this subhalo
                for evo_snapshot, evo_redshift in zip(evo_snapshots, evo_redshifts):
                    treeIndex = tuple(mpbLocal['SnapNum']).index(evo_snapshot)
                    shIDs.append(mpbLocal['SubfindID'][treeIndex])
                    redshifts.append(evo_redshift)

        saveFilename2 = './figure14_stamps_%s_evo-%d_red-%d_blue-%d_rot-%s.pdf' % \
          (sP.simName,evo,redSample,blueSample,rotation)
    else:
        # paged exploration for picking interesting galaxies, load all and sub-divide
        if 1:
            baseStr = 'figure12_pages'
            numPages = 20
            massBin  = [12.0, 12.2]
        elif 0:
            baseStr = 'figure12_pagesHighMass'
            numPages = 31
            massBin  = [12.2, 13.5]

        shIDsAll, _ = selectHalosFromMassBin(sP, [massBin], 2000, massBinInd=0, selType='linear')
        shIDs = pSplit(shIDsAll, numPages, curPage)[0:nGalaxies]
        redshifts = np.zeros(len(shIDs), dtype='float32') + redshift_init

        assert shIDs.size == nGalaxies # make sure we have nGalaxies per render
        saveFilename2 = './%s_%s_%s_rot=%s_page-%dof%d_hsml%.1f-age1-3-2_size%.1f.pdf' % \
          (baseStr,sP.simName,partType,rotation,curPage,numPages,hsmlFac,size)

    # load colors and morphs for custom labeling
    if not evo:
        gr_colors_z0, _ = loadSimGalColors(sP, defSimColorModel, bands=['g','r'], projs='random')
        kappa_stars, _, _, _ = simSubhaloQuantity(sP, 'Krot_oriented_stars2')

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        lCust = ['ID %d z = %.1f' % (shID, redshifts[i])]
        lScale = False #True if i == 0 else False

        if not evo:
            detailsStr = '(g-r) = %.2f $\kappa_\star$ = %.2f' % (gr_colors_z0[shID],kappa_stars[shID])
            lCust.append(detailsStr) 
                           
        panels.append( {'hInd':shID, 'redshift':redshifts[i], 'labelCustom':lCust, 'labelScale':lScale} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 1200
        colorbars    = False
        nRows        = nRowsFig
        saveFilename = saveFilename2

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def vogelsberger_redBlue42(run='illustris', sample='blue'):
    """ Recreate the 'Vogelsberger+ (2014) sample of 41 red/blue galaxies', either for 
    Illustris-1 or cross-matched in TNG100-1. """
    if sample == 'red':
        illustris_ids = [123773,178998,215567,234535,251933,267605,
                         299439,310062,326049,344358,359184,378583,
                         385020,393669,135289,192506,217716,239606,
                         262030,272551,305959,310951,331451,350144,
                         372778,382424,386640,393722,154948,195486,
                         219708,242071,264095,293759,309593,318565,
                         342383,354524,377255,384031,392429,395125]
    if sample == 'blue':
        illustris_ids = [242959,288927,310273,319371,331996,336439,
                         342103,345367,358785,366034,374531,384734,
                         386479,391881,261085,300120,312287,324323,
                         332327,336790,343669,351433,362540,366317,
                         375386,385795,386720,394285,283832,303990,
                         313541,326247,332891,339311,344821,352713,
                         365316,374140,376363,386304,390653,394942]

    # config
    res         = 1820 
    redshift    = 0.0
    rVirFracs   = None
    method      = 'sphMap'
    nPixels     = [300,300]
    axes        = [0,1]
    labelZ      = False
    labelSim    = False
    labelHalo   = 'Mstar'
    relCoords   = True
    mpb         = None
    labelScale  = False
    rotation    = 'face-on'
    size        = 60.0 # 30 kpc in each direction from center
    sizeType    = 'pkpc'
    partType    = 'stars'
    partField   = 'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'
    hsmlFac     = 0.5
    nRowsFig    = 7 # 6 columns, 7 rows

    # which subhalos?
    sP = simParams(res=res, run=run, redshift=redshift)

    if run == 'illustris':
        subhalo_ids = illustris_ids
    else:
        # cross-match and get TNG subhalo IDs
        sP_illustris = simParams(res=res, run='illustris', redshift=redshift)
        subhalo_ids = crossMatchSubhalosBetweenRuns(sP_illustris, sP, illustris_ids, method='Positional')

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(subhalo_ids):                           
        panels.append( {'hInd':shID, 'labelCustom':['ID %d' % shID]} )

    class plotConfig:
        plotStyle    = 'edged'
        rasterPx     = 1200
        colorbars    = False
        nRows        = nRowsFig
        saveFilename = './sample42_%s_%s.pdf' % (sample,run)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)
