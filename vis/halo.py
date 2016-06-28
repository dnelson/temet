"""
halo.py
  Visualizations for individual halos/subhalos from cosmological runs.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from os.path import isfile

from vis.common import renderMultiPanel, meanAngMomVector, rotationMatrixFromVec, saveBasePath
from cosmo.load import groupCatSingle
from cosmo.util import validSnapList
from cosmo.mergertree import mpbSmoothedProperties

def haloImgSpecs(sP, sizeFac, nPixels, axes, relCoords, rotation, mpb, **kwargs):
    """ Factor out some box/image related calculations common to all halo plots. """
    if mpb is None:
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
    else:
        # use the smoothed MPB properties to get halo properties at this snapshot
        if sP.snap < mpb['SnapNum'].min():
            # for very early times, linearly interpolate properties at start of tree back to t=0
            if rotation is not None:
                raise Exception('Cannot use rotation (or any group-ordered load) prior to mpb start.')

            fitSize = np.max( [np.int(mpb['SnapNum'].size * 0.02), 3] )
            fitN = 1 # polynomial order, 1=linear, 2=quadratic

            fitX = mpb['SnapNum'][-fitSize:]

            haloVirRad = np.poly1d( np.polyfit( fitX, mpb['sm']['rvir'][-fitSize:], fitN ) )(sP.snap)

            boxCenter = np.zeros( 3, dtype='float32' )
            for i in range(3):
                boxCenter[i] = np.poly1d( np.polyfit( fitX, mpb['sm']['pos'][-fitSize:,i], fitN ) )(sP.snap)
        else:
            # for times within actual MPB, use smoothed properties directly
            ind = np.where( mpb['SnapNum'] == sP.snap )[0]
            assert len(ind)

            shID = mpb['SubfindID'][ind[0]]
            haloVirRad = mpb['sm']['rvir'][ind[0]]
            boxCenter = mpb['sm']['pos'][ind[0],:]
            boxCenter = boxCenter[ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering

    if sizeFac > 0:
        # multiplier on the virial radius
        boxSizeImg = sizeFac * np.array([haloVirRad, haloVirRad*(nPixels[1]/nPixels[0]), haloVirRad])
    else:
        # absolute size (in -1* code length units)
        boxSizeImg = -sizeFac * np.array([1.0, 1.0*(nPixels[1]/nPixels[0]), 1.0])

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0], 
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    # make coordinates relative
    if relCoords:
        extent[0:2] -= boxCenter[0]
        extent[2:4] -= boxCenter[1]

    # derive appropriate rotation matrix if requested
    rotMatrix = None
    rotCenter = None

    if rotation is not None:
        # calculate 'mean angular momentum' vector of the galaxy (method choices herein)
        if mpb is None:
            jVec = meanAngMomVector(sP, subhaloID=shID)
        else:
            shPos = mpb['sm']['pos'][ind[0],:]
            shVel = mpb['sm']['vel'][ind[0],:]

            jVec = meanAngMomVector(sP, subhaloID=shID, shPos=shPos, shVel=shVel)
            rotCenter = shPos

        target_vec = np.zeros( 3, dtype='float32' )

        # face-on: rotate the galaxy j vector onto the unit axis vector we are projecting along
        if str(rotation) == 'face-on': target_vec[ 3-axes[0]-axes[1] ] = 1.0

        # edge-on: rotate the galaxy j vector to be aligned with the 2nd (e.g. y) requested axis
        if str(rotation) == 'edge-on': target_vec[ axes[1] ] = 1.0

        if target_vec.sum() == 0.0:
            raise Exception('Not implemented.')

        rotMatrix = rotationMatrixFromVec(jVec, target_vec)

    return boxSizeImg, boxCenter, extent, haloVirRad, rotMatrix, rotCenter

def renderSingleHalo(confNum):
    """ Driver: render view(s) of a single halo in one plot, with a variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'temp'} )
    #panels.append( {'run':'zooms', 'res':9, 'redshift':2.0, 'partType':'gas', 'partField':'coldens'} )
 
    #panels.append( {'run':'zooms', 'hInd':7,'res':11,'redshift':2.5} )
    #panels.append( {'run':'zooms2','hInd':2,'res':11,'redshift':2.5,'partField':'dens'} )
    #panels.append( {'run':'zooms2','hInd':2,'res':11,'redshift':2.5,'partField':'entr','valMinMax':[6.0,9.0]} )
    #panels.append( {'run':'zooms2','hInd':2,'res':11,'redshift':2.5,'partField':'velmag', 'valMinMax':[100,300]} )

    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'O VI',  'valMinMax':[10.0,15.0]} )
    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'HI',    'valMinMax':[10.0,24.0]} )
    #panels.append( {'run':'zooms2','hInd':2,'res':10, 'partField':'C IV',  'valMinMax':[10.0,17.0]} )

    # sizeFac = 10^2.8 * 0.7 * 2 (match to pm2.0 for Guinevere)
    #panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'temp'} )
    #panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'coldens_msunkpc2'} )
    #panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'gas',   'partField':'HI', 'valMinMax':[14.0,22.0]} )
    #panels.append( {'hInd':350671, 'sizeFac':-140.0, 'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,9.5]} )

    if confNum == 0:
        panels.append( {'hInd':0, 'partType':'gas', 'partField':'TimeStep', 'valMinMax':[-3.6,-5.2]} )
    if confNum == 1:
        panels.append( {'hInd':0, 'partType':'gas', 'partField':'TimebinHydro', 'valMinMax':[30,45]} )
    if confNum == 2:
        panels.append( {'hInd':0, 'partType':'dm', 'partField':'TimeStep', 'valMinMax':[-3.6,-5.2]} )
    if confNum == 3:
        panels.append( {'hInd':0, 'partType':'stars', 'partField':'TimeStep', 'valMinMax':[-3.6,-5.2]} )

    # plot config (common)
    #hInd      = 7             # zoom halo index
    run       = 'tng'          # run name
    res       = 2160           # run resolution
    redshift  = 6.0            # run redshift
    #partType   = 'gas'        # which particle type to project
    #partField = 'temp'        # which quantity/field to project for that particle type
    #valMinMax = [4.2, 6.5]    # stretch colortable between minimum and maximum field values
    rVirFracs  = [0.15,1.0]    # draw circles at these fractions of a virial radius
    method     = 'sphMap'      # sphMap, voronoi_const, voronoi_grads, ...
    nPixels    = [1400,1400]   # number of pixels for each dimension of images when projecting
    sizeFac    = 2.5           # size of imaging box around halo center in units of its virial radius
    hsmlFac    = 2.5           # multiplier on smoothing lengths for sphMap
    axes       = [1,0]         # e.g. [0,1] is x,y
    labelZ     = False         # label redshift inside (upper right corner) of panel
    labelScale = False         # label spatial scale with scalebar (upper left of panel)
    labelSim   = False         # label simulation name (lower right corner) of panel
    relCoords  = True          # if plotting x,y,z coordinate labels, make them relative to box/halo center
    rotation   = None          # 'face-on', 'edge-on', or None
    mpb        = None          # use None for non-movie/single frame

    # render config (global)
    class plotConfig:
        plotStyle  = 'open_black'  # open, edged, open_black, edged_black
        rasterPx   = 1400    # each panel will have this number of pixels if making a raster (png) output
                             # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True     # include colorbars
        #saveFilename = saveBasePath + 'renderHalo_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))
        saveFilename = saveBasePath + 'renderHalo_%s-%s_%s.pdf' % \
          (panels[0]['partType'],panels[0]['partField'],datetime.now().strftime('%d-%m-%Y'))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in locals().iteritems():
            if cName in ['panels','plotConfig','simParams','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue

            p[cName] = cVal

        # add simParams info
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=p['hInd'])

        # add imaging config for single halo view
        p['boxSizeImg'], p['boxCenter'], p['extent'], \
        p['haloVirRad'], p['rotMatrix'], p['rotCenter'] = haloImgSpecs(**p)

    # request render and save
    renderMultiPanel(panels, plotConfig)

def renderSingleHaloFrames(confName):
    """ Driver: render view(s) of a single halo in one plot, and repeat this frame across all snapshots 
    using the smoothed MPB properties. """
    from util import simParams

    # plot config (non-common, each entry adds one panel)
    panels = []

    if confName == 'movie1':
        panels.append( {'hInd':2, 'res':11, 'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'hInd':2, 'res':11, 'partField':'temp',    'valMinMax':[4.0, 6.5]} )
        panels.append( {'hInd':2, 'res':11, 'partField':'entr',    'valMinMax':[6.0,9.0]} )

    if confName == 'movie2':
        panels.append( {'hInd':2, 'res':9,  'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'hInd':2, 'res':10, 'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'hInd':2, 'res':11, 'partField':'coldens', 'valMinMax':[19.0,23.0]} )
        panels.append( {'hInd':2, 'res':9,  'partField':'temp',    'valMinMax':[4.0,6.5]} )
        panels.append( {'hInd':2, 'res':10, 'partField':'temp',    'valMinMax':[4.0,6.5]} )
        panels.append( {'hInd':2, 'res':11, 'partField':'temp',    'valMinMax':[4.0,6.5]} )

    # plot config (common)
    #hInd      = 2               # zoom halo index
    run        = 'zooms2'        # run name
    #res       = 9               # run resolution
    partType   = 'gas'           # which particle type to project
    #partField = 'temp'          # which quantity/field to project for that particle type
    #valMinMax = [4.2, 6.5]      # stretch colortable between minimum and maximum field values
    rVirFracs  = [0.15,0.5,1.0]  # draw circles at these fractions of a virial radius
    method     = 'sphMap'        # sphMap, voronoi_const, voronoi_grads, ...
    nPixels    = [1000,1000]     # number of pixels for each dimension of images when projecting
    sizeFac    = 3.5             # size of imaging box around halo center in units of its virial radius
    hsmlFac    = 2.5             # multiplier on smoothing lengths for sphMap
    axes       = [1,0]           # e.g. [0,1] is x,y
    labelZ     = False           # label redshift inside (upper right corner) of panel
    labelScale = False           # label spatial scale with scalebar (upper left of panel)
    labelSim   = False           # label simulation name (lower right corner) of panel
    relCoords  = True            # if plotting x,y,z coordinate labels, make them relative to box/halo center
    rotation   = None            # 'face-on', 'edge-on', or None

    # movie config
    treeRedshift = 2.0          # at what redshift does the tree/MPB start (for periodic box, snap of hInd)
    minRedshift  = 2.0          # ending redshift of frame sequence (we go forward in time)
    maxRedshift  = 100.0        # starting redshift of frame sequence (we go forward in time)
    maxNumSnaps  = None         # make at most this many evenly spaced frames, or None for all

    # render config (global)
    class plotConfig:
        plotStyle  = 'open' # open, edged, open_black, edged_black
        rasterPx   = 1200   # each panel will have this number of pixels if making a raster (png) output
                            # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True    # include colorbars

    # load MPB properties for each panel, could be e.g. different runs (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in locals().iteritems():
            if cName in ['panels','plotConfig','simParams','sP','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue

            p[cName] = cVal

        # load MPB once per panel
        sP = simParams(res=p['res'], run=p['run'], hInd=p['hInd'], redshift=treeRedshift)

        p['shID'] = sP.zoomSubhaloID if sP.isZoom else sP.hInd # direct input of subhalo ID for periodic box
        p['mpb'] = mpbSmoothedProperties(sP, p['shID'])

    # determine frame sequence (as the last sP in panels is used somewhat at random, we are here 
    # currently assuming that all runs in panels have the same snapshot configuration)
    snapNums = validSnapList(sP, maxNum=maxNumSnaps, minRedshift=minRedshift, maxRedshift=maxRedshift)
    frameNum = 0

    for snapNum in snapNums:
        print('Frame [%d of %d] at snap %d:' % (frameNum,snapNums.size,snapNum))
        # finalize panels list (all properties not set here are invariant in time)
        for p in panels:
            # override simParams info at this snapshot
            p['sP'] = simParams(res=p['res'], run=p['run'], snap=snapNum, hInd=p['hInd'])

            # add imaging config for single halo view using MPB
            p['boxSizeImg'], p['boxCenter'], p['extent'], \
            p['haloVirRad'], p['rotMatrix'], p['rotCenter'] = haloImgSpecs(**p)

        # request render and save
        plotConfig.saveFilename = saveBasePath + 'renderHaloFrame_%03d.png' % (frameNum)
        frameNum += 1

        if isfile(plotConfig.saveFilename):
            print('SKIP: %s' % plotConfig.saveFilename)
            continue

        renderMultiPanel(panels, plotConfig)
