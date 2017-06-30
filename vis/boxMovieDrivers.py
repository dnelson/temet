"""
boxMovieDrivers.py
  Render specific fullbox (movie frame) visualizations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime

from vis.common import savePathDefault
from vis.box import renderBox, renderBoxFrames
from util import simParams

def subbox_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing several quantities of a single subbox (4x2 panels, 4K). """
    panels = []

    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True} )
    panels.append( {'hsmlFac':0.5, 'partType':'dm',    'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.5]} )
    panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2]} )
    #panels.append( {'hsmlFac':0.5, 'partType':'stars', 'partField':'stellar_age', 'valMinMax':[2.0,13.0]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'bmag_uG', 'valMinMax':[-3.0,1.0]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'hsmlFac':2.5, 'partType':'gas',   'partField':'velmag', 'valMinMax':[100,1000]} )
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
    panels.append( {'run':'illustris', 'variant':'subbox0', 'zoomFac':0.99, 'labelScale':True} )
    panels.append( {'run':'tng',       'variant':'subbox0', 'zoomFac':0.99, 'labelZ':True} )
    # subbox1:
    #panels.append( {'run':'illustris', 'variant':'subbox2', 'zoomFac':0.99, 'labelScale':True} )
    #panels.append( {'run':'tng',       'variant':'subbox1', 'zoomFac':0.99*(5.0/7.5), 'labelZ':True} )

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
        savePath  = '/home/extdylan/data/frames/comp_gasdens_sb0/'
        plotStyle = 'edged_black'
        rasterPx  = 1920
        colorbars = True

        # movie config
        minZ      = 0.0
        maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)

def Illustris_vs_TNG_subbox0_4x2_movie(curTask=0, numTasks=1):
    """ Render a movie comparing Illustris-1 (top) and L75n1820TNG subbox0 (bottom), 4 quantities per row. """
    panels = []

    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelScale':True, 'labelSim':True} )
    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'illustris', 'hsmlFac':2.5, 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'illustris', 'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2]} )

    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.2], 'labelSim':True} )
    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'temp', 'valMinMax':[4.4,7.6]} )
    panels.append( {'run':'tng', 'hsmlFac':2.5, 'partType':'gas', 'partField':'metal_solar', 'valMinMax':[-2.0,0.4]} )
    panels.append( {'run':'tng', 'hsmlFac':0.5, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.2], 'labelZ':True} )

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
        maxNSnaps = 2700 # 90 seconds at 30 fps

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
