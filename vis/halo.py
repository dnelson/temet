"""
halo.py
  Visualizations for individual halos/subhalos from cosmological runs.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from vis.common import renderMultiPanel
from cosmo.load import groupCatSingle

def haloImgSpecs(sP, sizeFac, nPixels, axes, **kwargs):
    """ Factor out some box/image related calculations common to all halo plots. """
    # load halo position and virial radius (of the central zoom halo, or a given halo in a periodic box)
    if sP.isZoom:
        shID = sP.zoomSubhaloID
    else:
        #shID = sP.matchedSubhaloID()
        shID = sP.hInd # assume direct input

    sh = groupCatSingle(sP, subhaloID=shID)
    gr = groupCatSingle(sP, haloID=sh['SubhaloGrNr'])

    if gr['GroupFirstSub'] != shID:
        raise Exception('Probably not intended to render a non-central subhalo.')

    haloVirRad = gr['Group_R_Crit200']
    boxSizeImg = sizeFac * np.array([haloVirRad, haloVirRad*(nPixels[1]/nPixels[0]), haloVirRad])
    boxCenter  = sh['SubhaloPos'][ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    return boxSizeImg, boxCenter, extent

def renderSingleHalo():
    """ Driver: render view(s) of a single halo in one plot, with a variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
    panels.append( {'partType':'gas', 'partField':'temp', 'axes':[0,1]} )
    panels.append( {'partType':'gas', 'partField':'temp', 'axes':[1,0]} )
    #panels.append( {'partType':'gas', 'partField':'temp', 'axes':[1,2]} )

    # plot config (common)
    hInd       = 0             # zoom halo index
    run       = 'feedback'      # run name
    res       = 128           # run resolution
    redshift  = 2.0           # run redshift
    #partType  = 'gas'         # which particle type to project
    #partField = 'coldens'     # which quantity/field to project for that particle type
    rVirCircs = [0.15,0.5,1.0] # draw circles at these fractions of a virial radius
    method    = 'sphMap'       # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = [600,600]      # number of pixels for each dimension of images when projecting
    sizeFac   = 3.0            # size of imaging box around halo center in units of its virial radius
    hsmlFac   = 2.5            # multiplier on smoothing lengths for sphMap
    #axes      = [0,1]          # e.g. [0,1] is x,y

    # render config (global)
    plotStyle    = 'open'  # open, edged
    rasterPx     = 1200     # each panel will have this number of pixels if making a raster (png) output
                            # but note also it controls the relative size balance of raster/vector (e.g. fonts)
    saveFilename = 'renderHalo_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in locals().iteritems():
            if cName in ['panels','simParams','plotStyle','rasterPx','saveFilename','p']:
                continue
            p[cName] = cVal

        # add simParams() and halo image config
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=p['hInd'])

        p['boxSizeImg'], p['boxCenter'], p['extent'] = haloImgSpecs(**p)

    # request render and save
    renderMultiPanel(panels, plotStyle, rasterPx, saveFilename)
