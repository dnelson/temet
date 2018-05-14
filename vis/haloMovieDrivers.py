"""
haloMovieDrivers.py
  Render specific halo movie/time-series visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py

from vis.common import savePathDefault
from vis.halo import renderSingleHalo, renderSingleHaloFrames, selectHalosFromMassBin
from cosmo.mergertree import loadMPB
from util import simParams

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
        partField  = 'metal_solar'
        valMinMax  = [-0.5,0.5]
    if conf == 3:
        partType   = 'dm'
        partField  = 'coldens2_msunkpc2'
        valMinMax  = [15.0,16.0]
    if conf == 4:
        partType   = 'gas'
        partField  = 'pressure_ratio'
        valMinMax  = [-2.0,1.0]

    # configure panels
    sP = simParams(res=res, run=run, redshift=zStart)
    for i in range(nSnapsBack):
        hIndLoc = 0
        if run == 'tng' and i < 2: hIndLoc = 1

        halo = sP.groupCatSingle(haloID=hIndLoc)
        print(sP.snap, sP.redshift, hIndLoc, halo['GroupFirstSub'], halo['GroupPos'])

        panels.append( {'hInd':halo['GroupFirstSub'], 'redshift':sP.redshift} )
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