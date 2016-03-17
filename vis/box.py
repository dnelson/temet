"""
box.py
  Visualizations for whole (cosmological) boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from vis.common import renderMultiPanel

def boxImgSpecs(sP, zoomFac, relCenPos, axes, **kwargs):
    """ Factor out some box/image related calculations common to all whole box plots. """
    boxSizeImg = [sP.boxSize, sP.boxSize, sP.boxSize]
    boxCenter  = [relCenPos[0],relCenPos[1],0.5] * np.array(boxSizeImg)

    boxSizeImg[0] *= zoomFac
    boxSizeImg[1] *= zoomFac

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0],
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    return boxSizeImg, boxCenter, extent

def renderBox():
    """ Driver: render views of a full/fraction of a cosmological box, variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    #panels.append( {'hsmlFac':1.0} )
    #panels.append( {'hsmlFac':2.0} )
    #panels.append( {'hsmlFac':3.0} )

    panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )

    # plot config (common, applied to all panels)
    #run       = 'tracer'   # run name
    #res       = 128        # run resolution
    #redshift  = 2.0        # run redshift
    #partType  = 'gas'      # which particle type to project
    #partField = 'coldens'  # which quantity/field to project for that particle type
    method    = 'sphMap'    # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = 600         # number of pixels per dimension of images when projecting
    zoomFac   = 0.5           # [0,1], only in axes, not along projection direction
    hsmlFac   = 2.5         # multiplier on smoothing lengths for sphMap
    relCenPos = [0.75,0.75]   # [0-1,0-1] relative coordinates of where to center image, only in axes
    axes      = [2,1]       # e.g. [0,1] is x,y

    # render config (global)
    plotStyle    = 'open'  # open, edged
    rasterPx     = 1200     # each panel will have this number of pixels if making a raster (png) output
                            # but note also it controls the relative size balance of raster/vector (e.g. fonts)
    saveFilename = 'renderBox_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in locals().iteritems():
            if cName in ['panels','simParams','plotStyle','rasterPx','saveFilename','p']:
                continue
            p[cName] = cVal

        # add simParams() and whole box image config
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'])

        p['boxSizeImg'], p['boxCenter'], p['extent'] = boxImgSpecs(**p)

        p['nPixels'] = [p['nPixels'],p['nPixels']]

    # request render and save
    renderMultiPanel(panels, plotStyle, rasterPx, saveFilename)
