"""
box.py
  Visualizations for whole (cosmological) boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from vis.common import renderMultiPanel
from cosmo.util import multiRunMatchedSnapList

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

    #panels.append( {'run':'zooms', 'res':10,  'partField':'coldens', 'hInd':7} )
    #panels.append( {'run':'zooms', 'res':10, 'partField':'density_cgs', 'hInd':7} )
    #panels.append( {'run':'zooms', 'res':10, 'partField':'density', 'hInd':7} )

    #panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'tracer', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'feedback', 'res':128, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )

    panels.append( {'res':455, 'axes':[0,1] } )
    panels.append( {'res':455, 'axes':[0,2] } )

    # plot config (common, applied to all panels)
    run       = 'tng'       # run name
    #res       = 128        # run resolution
    redshift  = 0.0         # run redshift
    partType  = 'dm'        # which particle type to project
    partField = 'coldens'   # which quantity/field to project for that particle type
    method    = 'sphMap'    # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = 900         # number of pixels per dimension of images when projecting
    zoomFac   = 1.0         # [0,1], only in axes, not along projection direction
    hsmlFac   = 0.5         # multiplier on smoothing lengths for sphMap
    relCenPos = [0.5,0.5]   # [0-1,0-1] relative coordinates of where to center image, only in axes
    #axes      = [0,1]       # e.g. [0,1] is x,y
    plotHalos = 20          # plot virial circles for the N most massive halos in the box
    rotMatrix = None        # rotation matrix
    rotCenter = None        # rotation center

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
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue

            p[cName] = cVal

        # add simParams info
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=None)

        # add imaging config for [square render of] whole box
        p['boxSizeImg'], p['boxCenter'], p['extent'] = boxImgSpecs(**p)
        p['nPixels'] = [p['nPixels'],p['nPixels']]

    # request render and save
    renderMultiPanel(panels, plotStyle, rasterPx, saveFilename)

def renderBoxFrames():
    """ Driver: render views of a full/fraction of a cosmological box, variable number of panels, and 
    repeat this frame across snapshots in order to make a movie. """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    panels.append( {'run':'tng', 'res':128, 'hsmlFac':0.5, 'partType':'gas', 'partField':'density' } )
    panels.append( {'run':'tng', 'res':128, 'hsmlFac':0.5, 'partType':'dm',  'partField':'density' } )
    panels.append( {'run':'tng', 'res':128, 'variant':'L12.5', 'hsmlFac':0.5, 'partType':'dm', 'partField':'density' } ) 
    #panels.append( {'run':'tracer', 'res':128, 'hsmlFac':0.5, 'partType':'gas', 'partField':'velmag' } )
    #panels.append( {'run':'feedback', 'res':128, 'hsmlFac':0.5, 'partType':'dm',  'partField':'velmag' } )

    # plot config (common, applied to all panels)
    #run       = 'tng'      # run name
    #res       = 128        # run resolution
    #partType  = 'dm'       # which particle type to project
    #partField = 'coldens'  # which quantity/field to project for that particle type
    method    = 'sphMap'    # sphMap, voronoi_const, voronoi_grads, ...
    nPixels   = 400         # number of pixels per dimension of images when projecting
    zoomFac   = 1.0         # [0,1], only in axes, not along projection direction
    #hsmlFac   = 0.5        # multiplier on smoothing lengths for sphMap
    relCenPos = [0.5,0.5]   # [0-1,0-1] relative coordinates of where to center image, only in axes
    axes      = [0,1]       # e.g. [0,1] is x,y
    plotHalos = 20          # plot virial circles for the N most massive halos in the box
    rotMatrix = None        # rotation matrix
    rotCenter = None        # rotation center

    # movie config
    minZ      = 0.0         # ending redshift of frame sequence (we go forward in time)
    maxZ      = 200.0       # starting redshift of frame sequence (we go forward in time)
    maxNSnaps = None        # make at most this many evenly spaced frames, or None for all
    matchUse  = 'condense'  # 'expand' or 'condense' to determine matching snaps between runs

    # render config (global)
    plotStyle    = 'open'   # open, edged
    rasterPx     = 1200     # each panel will have this number of pixels if making a raster (png) output
                            # but note also it controls the relative size balance of raster/vector (e.g. fonts)

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

        v = p['variant'] if 'variant' in p else None
        p['sP'] = simParams(res=p['res'], run=p['run'], variant=v)

        # add imaging config for [square render of] whole box
        p['nPixels'] = [p['nPixels'],p['nPixels']]

    # determine frame sequence
    snapNumLists = multiRunMatchedSnapList(panels, matchUse, maxNum=maxNSnaps, 
                                           minRedshift=minZ, maxRedshift=maxZ)
    numFrames = snapNumLists[0].size

    # render sequence
    for frameNum in range(numFrames):
        snapNumsStr = ' '.join([str(s) for s in [snapList[frameNum] for snapList in snapNumLists]])
        print('Frame [%d of %d]: using snapshots [%s]' % (frameNum,numFrames,snapNumsStr))

        # finalize panels list (all properties not set here are invariant in time)
        for i, p in enumerate(panels):
            # override simParams info at this snapshot
            snapNum = snapNumLists[i][frameNum]
            p['sP'] = simParams(res=p['sP'].res, run=p['sP'].run, variant=p['sP'].variant, snap=snapNum)

            # setup currenty constant in time, could here give a rotation/zoom/etc with time
            p['boxSizeImg'], p['boxCenter'], p['extent'] = boxImgSpecs(**p)
            
        # request render and save
        saveFilename = 'renderBoxFrame_%03d.png' % (frameNum)
        renderMultiPanel(panels, plotStyle, rasterPx, saveFilename)
