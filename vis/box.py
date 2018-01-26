"""
box.py
  Visualizations for whole (cosmological) boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from os.path import isfile

from vis.common import renderMultiPanel, savePathDefault, defaultHsmlFac
from cosmo.util import multiRunMatchedSnapList
from util.helper import iterable, pSplit
from util import simParams

def boxImgSpecs(sP, zoomFac, sliceFac, relCenPos, absCenPos, axes, **kwargs):
    """ Factor out some box/image related calculations common to all whole box plots. 
    Image zoomFac fraction of entire fullbox/subbox, zooming around relCenPos 
    ([0.5,0.5] being box center point). """
    assert relCenPos is None or absCenPos is None
    
    if sP.subbox is None:
        boxSizeImg = np.array([sP.boxSize, sP.boxSize, sP.boxSize])

        if absCenPos is None:
            boxCenter = [relCenPos[0],relCenPos[1],0.5] * np.array(boxSizeImg)
        else:
            boxCenter = absCenPos
    else:
        boxSizeImg = sP.subboxSize[sP.subbox] * np.array([1,1,1])

        boxCenter0 = relCenPos[0] * sP.subboxCen[sP.subbox][axes[0]]-sP.subboxSize[sP.subbox]*0.5 + \
               (1.0-relCenPos[0]) * sP.subboxCen[sP.subbox][axes[0]]+sP.subboxSize[sP.subbox]*0.5

        boxCenter1 = relCenPos[1] * sP.subboxCen[sP.subbox][axes[1]]-sP.subboxSize[sP.subbox]*0.5 + \
               (1.0-relCenPos[1]) * sP.subboxCen[sP.subbox][axes[1]]+sP.subboxSize[sP.subbox]*0.5

        boxCenter2 = sP.subboxCen[sP.subbox][3-axes[0]-axes[1]]
        boxCenter  = np.array([boxCenter0, boxCenter1, boxCenter2])

    boxSizeImg[0] *= zoomFac
    boxSizeImg[1] *= zoomFac
    boxSizeImg[2] *= sliceFac

    extent = [ boxCenter[0] - 0.5*boxSizeImg[0], boxCenter[0] + 0.5*boxSizeImg[0],
               boxCenter[1] - 0.5*boxSizeImg[1], boxCenter[1] + 0.5*boxSizeImg[1]]

    return boxSizeImg, boxCenter, extent

def renderBox(panels, plotConfig, localVars, skipExisting=True, retInfo=False):
    """ Driver: render views of a full/fraction of a cosmological box, variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """

    # defaults (all panel fields that can be specified)

    run         = 'tng'       # run name
    res         = 1820        # run resolution
    redshift    = 0.0         # run redshift
    partType    = 'dm'        # which particle type to project
    partField   = 'coldens'   # which quantity/field to project for that particle type
    #valMinMax  = [min,max]   # stretch colortable between minimum and maximum field values
    method      = 'sphMap'    # sphMap, sphMap_global, sphMap_minIP, sphMap_maxIP, voronoi_*, ...
    nPixels     = 1400        # number of pixels per dimension of images when projecting (960 1400)
    zoomFac     = 1.0         # [0,1], only in axes, not along projection direction
    #hsmlFac     = 1.0        # multiplier on smoothing lengths for sphMap (dm 0.2) (gas 2.5)
    relCenPos   = [0.5,0.5]   # [0-1,0-1] relative coordinates of where to center image, only in axes
    absCenPos   = None        # [x,y,z] in simulation coordinates to place at center of image
    sliceFac    = 1.0         # [0,1], only along projection direction, relative depth wrt boxsize
    axes        = [0,1]       # e.g. [0,1] is x,y
    axesUnits   = 'code'      # code [ckpc/h], mpc, deg, arcmin
    labelZ      = False       # label redshift inside (upper right corner) of panel
    labelScale  = False       # label spatial scale with scalebar (upper left of panel)
    labelSim    = False       # label simulation name (lower right corner) of panel
    labelCustom = False       # custom label string to include
    plotHalos   = 20          # plot virial circles for the N most massive halos in the box
    rotMatrix   = None        # rotation matrix
    rotCenter   = None        # rotation center

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle = 'open'   # open, edged, open_black, edged_black
        rasterPx  = 1400     # each panel will have this number of pixels if making a raster (png) output
                             # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True     # include colorbars

        saveFilename = savePathDefault + 'renderBox_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig,var):
            setattr(plotConfig,var,getattr(plotConfigDefaults,var))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in localVars.iteritems():
            if cName in ['panels','plotConfig','plotConfigDefaults','simParams','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue
            p[cName] = cVal

        for cName,cVal in locals().iteritems():
            if cName in p or cName in ['panels','plotConfig','plotConfigDefaults','simParams','p']:
                continue
            p[cName] = cVal

        if 'hsmlFac' not in p: p['hsmlFac'] = defaultHsmlFac(p['partType'], p['partField'])

        # add simParams info
        v = p['variant'] if 'variant' in p else None
        h = p['hInd'] if 'hInd' in p else None
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=h, variant=v)

        # add imaging config for [square render of] whole box
        p['boxSizeImg'], p['boxCenter'], p['extent'] = boxImgSpecs(**p)
        if not isinstance(p['nPixels'],list): p['nPixels'] = [p['nPixels'],p['nPixels']]

    # request render and save
    if retInfo: return panels
   
    # skip if final output render file already exists?
    if skipExisting and isfile(plotConfig.saveFilename):
        print('SKIP: %s' % plotConfig.saveFilename)
        return
 
    renderMultiPanel(panels, plotConfig)

def renderBoxFrames(panels, plotConfig, localVars, curTask=0, numTasks=1, skipExisting=True):
    """ Driver: render views of a full/fraction of a cosmological box, variable number of panels, and 
    repeat this frame across snapshots in order to make a movie. """
    
    # defaults (all panel fields that can be specified)

    run         = 'illustris' # run name
    res         = 1820        # run resolution
    partType    = 'dm'        # which particle type to project
    partField   = 'coldens'   # which quantity/field to project for that particle type
    #valMinMax  = [min,max]   # stretch colortable between minimum and maximum field values
    method      = 'sphMap'    # sphMap, sphMap_global, sphMap_minIP, sphMap_maxIP, voronoi_*, ...
    nPixels     = 960         # number of pixels per dimension of images when projecting
    zoomFac     = 1.0         # [0,1], only in axes, not along projection direction
    #hsmlFac     = 2.5        # multiplier on smoothing lengths for sphMap
    relCenPos   = [0.5,0.5]   # [0-1,0-1] relative coordinates of where to center image, only in axes
    absCenPos   = None        # [x,y,z] in simulation coordinates to place at center of image
    sliceFac    = 1.0         # [0,1], only along projection direction, relative depth wrt boxsize
    axes        = [0,1]       # e.g. [0,1] is x,y
    axesUnits   = 'code'      # code [ckpc/h], Mpc, deg, arcmin
    labelZ      = False       # label redshift inside (upper right corner) of panel
    labelScale  = False       # label spatial scale with scalebar (upper left of panel)
    labelSim    = False       # label simulation name (lower right corner) of panel
    labelCustom = False       # custom label string to include
    plotHalos   = 0           # plot virial circles for the N most massive halos in the box
    rotMatrix   = None        # rotation matrix
    rotCenter   = None        # rotation center

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle = 'open'     # open, edged, open_black, edged_black
        rasterPx  = 960        # each panel will have this number of pixels if making a raster (png) output
                               # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True       # include colorbars

        savePath     = savePathDefault
        saveFileBase = 'renderBoxFrame' # filename base upon which frame numbers are appended

        # movie config
        minZ      = 0.0        # ending redshift of frame sequence (we go forward in time)
        maxZ      = 128.0      # starting redshift of frame sequence (we go forward in time)
        maxNSnaps = None       # make at most this many evenly spaced frames, or None for all
        matchUse  = 'condense' # 'expand' or 'condense' to determine matching snaps between runs

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig,var):
            setattr(plotConfig,var,getattr(plotConfigDefaults,var))

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in localVars.iteritems():
            if cName in ['panels','plotConfig','plotConfigDefaults','simParams','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue
            p[cName] = cVal

        for cName,cVal in locals().iteritems():
            if cName in p or cName in ['panels','plotConfig','plotConfigDefaults','simParams','p']:
                continue
            p[cName] = cVal

        if 'hsmlFac' not in p: p['hsmlFac'] = defaultHsmlFac(p['partType'], p['partField'])

        v = p['variant'] if 'variant' in p else None
        h = p['hInd'] if 'hInd' in p else None
        p['sP'] = simParams(res=p['res'], run=p['run'], hInd=h, variant=v)

        # add imaging config for [square render of] whole box
        if not isinstance(p['nPixels'],list): p['nPixels'] = [p['nPixels'],p['nPixels']]

    # determine frame sequence
    snapNumLists = multiRunMatchedSnapList(panels, plotConfig.matchUse, maxNum=plotConfig.maxNSnaps, 
                                           minRedshift=plotConfig.minZ, maxRedshift=plotConfig.maxZ)
    numFramesTot = snapNumLists[0].size

    # optionally parallelize over multiple tasks
    fNumsThisTask = pSplit(range(numFramesTot), numTasks, curTask)

    print('Task [%d of %d] rendering [%d] frames of [%d] total (from %d to %d)...' % \
        (curTask,numTasks,len(fNumsThisTask),numFramesTot,np.min(fNumsThisTask),np.max(fNumsThisTask)))

    # render sequence
    for frameNum in fNumsThisTask:
        snapNumsStr = ' '.join([str(s) for s in [iterable(snapList)[frameNum] for snapList in snapNumLists]])
        print('\nFrame [%d of %d]: using snapshots [%s]' % (frameNum,numFramesTot-1,snapNumsStr))

        # finalize panels list (all properties not set here are invariant in time)
        for i, p in enumerate(panels):
            # override simParams info at this snapshot
            snapNum = iterable(snapNumLists[i])[frameNum]
            p['sP'] = simParams(res=p['sP'].res, run=p['sP'].run, variant=p['sP'].variant, snap=snapNum)

            # setup currenty constant in time, could here give a rotation/zoom/etc with time
            p['boxSizeImg'], p['boxCenter'], p['extent'] = boxImgSpecs(**p)

            # e.g. update the upper bound of 'stellar_age' valMinMax, if set, to the current tAge [in Gyr]
            #if p['partField'] == 'stellar_age' and 'valMinMax' in p:
            #    p['valMinMax'][1] = np.max( [p['sP'].units.redshiftToAgeFlat(p['sP'].redshift), 3.0] )
            
        # request render and save
        plotConfig.saveFilename = plotConfig.savePath + plotConfig.saveFileBase + '_%03d.png' % (frameNum)

        if skipExisting and isfile(plotConfig.saveFilename):
            print('SKIP: ' + plotConfig.saveFilename)
            continue

        renderMultiPanel(panels, plotConfig)
