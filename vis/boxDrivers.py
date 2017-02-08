"""
haloDrivers.py
  Render specific halo visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import savePathDefault
from vis.box import renderBox, renderBoxFrames
from util.helper import pSplit
from util import simParams
from cosmo.load import groupCatSingle

def realizations(conf=1):
    """ Render a whole box frame of one TNG run at one redshift, comparing gas and magnetic pressure. """
    panels = []

    for i in range(1,11):
        variant = 'r%03d' % i
        panels.append( {'variant':variant} )

    if conf == 1:
        partType   = 'dm'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [5.0,8.5]

    if conf == 2:
        partType   = 'gas'
        partField  = 'coldens_msunkpc2'
        valMinMax  = [4.2,7.2]

    run        = 'tng'
    res        = 256
    redshift   = 0.0
    hsmlFac    = 0.5
    nPixels    = 960
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = True
    plotHalos  = 10

    # render config (global)
    class plotConfig:
        plotStyle  = 'open_black'
        rasterPx   = 960
        colorbars  = True

        sP = simParams(res=res, run=run, redshift=redshift)
        saveFilename = savePathDefault + 'realizations_gas_%s_z%.1f.pdf' % (sP.simName, redshift)

    renderBox(panels, plotConfig, locals())

def Illustris_1_subbox0_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing several quantities of a single subbox. """
    panels = []

    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True} )
    panels.append( {'hsmlFac':0.5, 'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.5]} )
    panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[3.2,8.2]} )
    #panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'stellar_age', 'valMinMax':[2.0,13.0]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'bmag_uG', 'valMinMax':[-3.0,1.0]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'velmag', 'valMinMax':[100,1400]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'O VI', 'valMinMax':[10,16], 'labelZ':True} )

    run     = 'tng' #'illustris'
    variant = 'subbox0'
    res     = 1820
    method  = 'sphMap'
    nPixels = 960
    axes    = [0,1] # x,y

    class plotConfig:
        savePath = '/home/extdylan/data/frames/%s_sb0/' % run
        plotStyle = 'edged_black'
        rasterPx  = 960
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_vs_TNG_subbox0_2x1_onequant_movie(curTask=0, numTasks=1, conf=1):
    """ Render a movie comparing Illustris-1 and L75n1820TNG subbox0, one quantity side by side. """
    panels = []

    # subbox0:
    #panels.append( {'run':'illustris', 'variant':'subbox0', 'zoomFac':0.99, 'labelScale':True} )
    #panels.append( {'run':'tng',       'variant':'subbox0', 'zoomFac':0.99, 'labelZ':True} )
    # subbox1:
    panels.append( {'run':'illustris', 'variant':'subbox2', 'zoomFac':0.99, 'labelScale':True} )
    panels.append( {'run':'tng',       'variant':'subbox1', 'zoomFac':0.99*(5.0/7.5), 'labelZ':True} )

    if conf == 1:
        hsmlFac = 2.5
        partType = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [4.2,7.2]

    res      = 1820
    method   = 'sphMap'
    nPixels  = 1920
    labelSim = True
    axes     = [0,1] # x,y

    class plotConfig:
        savePath  = '/home/extdylan/data/frames/comp_gasdens_sb1/'
        plotStyle = 'edged_black'
        rasterPx  = 1920
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = None

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_vs_TNG_subbox0_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing Illustris-1 (top) and L75n1820TNG subbox0 (bottom), 4 quantities per row. """
    panels = []

    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True, 'labelSim':True} )
    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'illustris', 'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[3.2,8.2]} )

    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelSim':True} )
    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'tng', 'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[3.2,8.2], 'labelZ':True} )

    variant = 'subbox0'
    res     = 1820
    method  = 'sphMap'
    nPixels = 960
    axes    = [0,1] # x,y

    class plotConfig:
        savePath  = '/home/extdylan/data/frames/comp_4x2_sb0/'
        plotStyle = 'edged_black'
        rasterPx  = 960
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = None

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_1_4subboxes_gasdens_movie(curTask=0, numTasks=1):
    """ Render a movie of a single quantity from multiple subboxes. """
    panels = []

    panels.append( {'variant':'subbox0', 'labelSim':True, 'labelScale':True} ) # upper left
    panels.append( {'variant':'subbox1', 'labelSim':True} )                    # upper right
    panels.append( {'variant':'subbox2', 'labelSim':True} )                    # lower left
    panels.append( {'variant':'subbox3', 'labelSim':True, 'labelZ':True} )     # lower right

    run       = 'illustris'
    partType  = 'gas'
    partField = 'density'
    valMinMax = [-5.5, -2.0]
    res       = 1820
    nPixels   = 960
    axes      = [0,1] # x,y
    redshift  = 0.0

    class plotConfig:
        plotStyle    = 'edged_black'
        rasterPx     = 960
        colorbars    = True
        saveFileBase = 'Illustris-1-4sb-gasDens'
        saveFilename = 'out.png'

        # movie config
        minZ      = 0.0
        maxZ      = 4.0
        maxNSnaps = 30

    renderBox(panels, plotConfig, locals())
    #renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def TNG_mainImages(res, conf=0):
    """ Create the FoF[0/1]-centered slices to be used for main presentation of the box. """
    panels = []

    dmMM  = [5.0, 8.5]
    gasMM = [4.3,7.3]

    if res in [455,910,1820]:
        # L75
        centerHaloID = 1 # fof
        nSlicesTot   = 3 # slice depth equal to a third, 25 Mpc/h = 37 Mpc
        curSlice     = 0 # offset slice along projection direction?
    if res in [625,1250,2500]:
        # L205
        centerHaloID = 0 # fof
        nSlicesTot   = 3 # slice depth equal to a fifth, 41 Mpc/h = 60 Mpc
        curSlice     = 0 # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 0.5 
        gasMM[0] += 0.7
    if res in [270,540,1080,2160]:
        # L35
        centerHaloID = 0 # fof
        nSlicesTot   = 1 # slice depth equal to a fifth, 35 Mpc/h = 52 Mpc
        curSlice     = 0 # offset slice along projection direction?

    if conf == 0:  panels.append( {'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':gasMM} )
    if conf == 1:  panels.append( {'partType':'dm',  'partField':'coldens_msunkpc2', 'valMinMax':dmMM} )
    if conf == 2:  panels.append( {'partType':'stars',  'partField':'coldens_msunkpc2'} )
    if conf == 3:  panels.append( {'partType':'stars',  'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'} )
    if conf == 4:  panels.append( {'partType':'gas', 'partField':'pressure_ratio', 'valMinMax':[-8,1], 'cmapCenVal':-3.0} )
    if conf == 5:  panels.append( {'partType':'gas', 'partField':'bmag_uG',   'valMinMax':[-3.5,1.0]} )
    if conf == 6:  panels.append( {'partType':'gas', 'partField':'Z_solar', 'valMinMax':[-2.0,-0.2]} )
    if conf == 7:  panels.append( {'partType':'gas', 'partField':'temp', 'valMinMax':[4.3,7.2]} )
    if conf == 8:  panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_Fe', 'valMinMax':[0.0,2.6]} )
    if conf == 9:  panels.append( {'partType':'gas', 'partField':'SN_IaII_ratio_metals', 'valMinMax':[-1.0,2.5]} )
    if conf == 10: panels.append( {'partType':'gas', 'partField':'SN_Ia_AGB_ratio_metals', 'valMinMax':[-0.48,0.06]} )
    if conf == 11: panels.append( {'partType':'gas', 'partField':'xray_lum', 'valMinMax':[29, 37.5]} )

    run        = 'tng'
    redshift   = 0.0
    nPixels    = 2000
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelSim   = False
    plotHalos  = False
    hsmlFac    = 2.5 # use for all: gas, dm, stars (for whole box)
    variant    = None

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # slice centering
    sliceFac  = (1.0/nSlicesTot)
    relCenPos = None

    #for curSlice in range(nSlicesTot):
    absCenPos = groupCatSingle(sP, haloID=centerHaloID)['GroupPos']
    absCenPos[3-axes[0]-axes[1]] += curSlice * sliceFac * sP.boxSize

    # render config (global)
    class plotConfig:
        plotStyle  = 'open'
        rasterPx   = 2000
        colorbars  = True

        saveFilename = './boxImage_%s_%s-%s_fof-%d_axes%d%d_%dof%d.png' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],centerHaloID,
           axes[0],axes[1],curSlice,nSlicesTot)

    renderBox(panels, plotConfig, locals())

def zoom_gasColDens_3res_or_3quant():
    """ Diagnostic plot of gas column dens in the entire zoom box at z=2 (e.g. 3 res or 3 quant comp). """
    panels = []

    #panels.append( {'res':10, 'partField':'coldens'} )
    #panels.append( {'res':10, 'partField':'coldens_msunkpc2'} )
    #panels.append( {'res':10, 'partField':'density'} )

    panels.append( {'res':9,  'partField':'coldens_msunkpc2'} )
    panels.append( {'res':10, 'partField':'coldens_msunkpc2'} )
    panels.append( {'res':11, 'partField':'coldens_msunkpc2'} )

    hInd       = 7
    run        = 'zooms'
    redshift   = 2.0
    partType   = 'gas'
    hsmlFac    = 2.0
    nPixels    = 1400
    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    plotHalos  = 20

    # render config (global)
    class plotConfig:
        plotStyle  = 'open_black'
        rasterPx   = 1400
        colorbars  = True

        Lstr = '-'.join([str(p['res']) for p in panels])
        saveFilename = savePathDefault + '%s_FullBoxGasColDens_h%dL%s.pdf' % (run,hInd,Lstr)

    renderBox(panels, plotConfig, locals())
