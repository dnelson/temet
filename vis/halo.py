"""
halo.py
  Visualizations for individual halos/subhalos from cosmological runs.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from vis.common import renderMultiPanel, meanAngMomVector, rotationMatrixFromVec
from cosmo.load import groupCatSingle

def haloImgSpecs(sP, sizeFac, nPixels, axes, relCoords, rotation, **kwargs):
    """ Factor out some box/image related calculations common to all halo plots. """
    # load halo position and virial radius (of the central zoom halo, or a given halo in a periodic box)
    if sP.isZoom:
        shID = sP.zoomSubhaloID
    else:
        #shID = sP.matchedSubhaloID()
        shID = sP.hInd # assume direct input of subhalo ID

    sh = groupCatSingle(sP, subhaloID=shID)
    gr = groupCatSingle(sP, haloID=sh['SubhaloGrNr'])

    if gr['GroupFirstSub'] != shID:
        raise Exception('Probably not intended to render a non-central subhalo.')

    haloVirRad = gr['Group_R_Crit200']
    boxCenter  = sh['SubhaloPos'][ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering

    if sizeFac > 0:
        # multiplier on the virial radius
        boxSizeImg = sizeFac * np.array([haloVirRad, haloVirRad*(nPixels[1]/nPixels[0]), haloVirRad])
    else:
        # absolute size (in -1* code length units)
        boxSizeImg = -sizeFac * np.array([1.0, 1.0*(nPixels[1]/nPixels[0]), 1.0])

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    if relCoords:
        extent[0:2] -= boxCenter[0]
        extent[2:4] -= boxCenter[1]

    # rotation?
    rotMatrix = None

    if rotation:
        # calculate 'mean angular momentum' vector of the galaxy (method choices herein)
        jVec = meanAngMomVector(sP, subhaloID=shID)

        target_vec = np.zeros( 3, dtype='float32' )

        # face-on: rotate the galaxy j vector onto the unit axis vector we are projecting along
        if str(rotation) == 'face-on': target_vec[ 3-axes[0]-axes[1] ] = 1.0

        # edge-on: rotate the galaxy j vector to be aligned with the 2nd (e.g. y) requested axis
        if str(rotation) == 'edge-on': target_vec[ axes[1] ] = 1.0

        if target_vec.sum() == 0.0:
            raise Exception('Not implemented.')

        rotMatrix = rotationMatrixFromVec(jVec, target_vec)

        # debugging
        #vec1 = np.array( (1,0,0), dtype='float32' )
        #vec2 = np.array( (0,1,0), dtype='float32' )
        #rotMatrix = rotationMatrixFromVec(vec1, vec2) # should take x to y, e.g. rotate 90 deg CCW

    return boxSizeImg, boxCenter, extent, haloVirRad, rotMatrix

def renderSingleHalo():
    """ Driver: render view(s) of a single halo in one plot, with a variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
 
    #panels.append( {'run':'zooms', 'hInd':7,'res':11,'redshift':2.5} )
    #panels.append( {'run':'zooms2','hInd':2,'res':11,'redshift':2.5} )

    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'O VI',  'valMinMax':[10.0,15.0]} )
    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'HI',    'valMinMax':[10.0,24.0]} )
    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'C IV',  'valMinMax':[10.0,17.0]} )

    # sizeFac = 10^2.8 * 0.7 * 2 (match to pm2.0 for Guinevere)
    panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'temp'} )
    panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'coldens_msunkpc2'} )
    panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'HI', 'valMinMax':[14.0,22.0]} )
    panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,9.5]} )

    # plot config (common)
    #hInd       = 7             # zoom halo index
    run       = 'illustris'     # run name
    res       = 1820            # run resolution
    redshift  = 0.0             # run redshift
    #partType  = 'gas'          # which particle type to project
    #partField = 'temp'         # which quantity/field to project for that particle type
    #valMinMax = [4.2, 6.5]     # stretch colortable between minimum and maximum field values
    rVirFracs = [0.15,0.5]      # draw circles at these fractions of a virial radius
    method    = 'sphMap'        # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = [600,600]       # number of pixels for each dimension of images when projecting
    #sizeFac   = 3.0            # size of imaging box around halo center in units of its virial radius
    hsmlFac   = 2.5             # multiplier on smoothing lengths for sphMap
    axes      = [1,0]           # e.g. [0,1] is x,y
    relCoords = True            # if plotting x,y,z coordinate labels, make them relative to box/halo center
    rotation  = 'edge-on'       # 'face-on', 'edge-on', or False

    # render config (global)
    plotStyle    = 'open'  # open, edged
    rasterPx     = 1200     # each panel will have this number of pixels if making a raster (png) output
                            # but note also it controls the relative size balance of raster/vector elements
    saveFilename = 'renderHalo_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in locals().iteritems():
            if cName in ['panels','simParams','plotStyle','rasterPx','saveFilename','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue

            p[cName] = cVal

        # add simParams info
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=p['hInd'])

        # add imaging config for single halo view
        p['boxSizeImg'], p['boxCenter'], p['extent'], p['haloVirRad'], p['rotMatrix'] = haloImgSpecs(**p)

    # request render and save
    renderMultiPanel(panels, plotStyle, rasterPx, saveFilename)
