"""
haloDrivers.py
  Render specific halo visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import saveBasePath
from vis.box import renderBox, renderBoxFrames
from util.helper import pSplit
from util import simParams

def Illustris_1_subbox0_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing several quantities of a single subbox. """
    panels = []

    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.0,7.2], 'labelScale':True} )
    panels.append( {'hsmlFac':0.5, 'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[4.5,8.5]} )
    panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[3.4,8.2]} )
    panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'stellar_age', 'valMinMax':[2.0,13.0]} )
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
        plotStyle = 'edged_black'
        rasterPx  = 960
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 128.0
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

def tng_boxPressureComp():
    """ Render a whole box frame of one TNG run at one redshift, comparing gas and magnetic pressure. """
    panels = []

    panels.append( {'partField':'P_B', 'valMinMax':[-5,7]} )
    panels.append( {'partField':'P_gas', 'valMinMax':[-5,7]} )
    panels.append( {'partField':'pressure_ratio'} )

    run        = 'tng'
    res        = 1820
    redshift   = 0.5
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

        sP = simParams(res=res, run=run, redshift=redshift)
        saveFilename = saveBasePath + 'boxPressureComp_%s_z%.1f.pdf' % (sP.simName, redshift)

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
        saveFilename = saveBasePath + '%s_FullBoxGasColDens_h%dL%s.pdf' % (run,hInd,Lstr)

    renderBox(panels, plotConfig, locals())
