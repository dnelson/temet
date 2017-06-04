"""
halo.py
  Visualizations for individual halos/subhalos from cosmological runs.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from datetime import datetime
from os.path import isfile

from vis.common import renderMultiPanel, meanAngMomVector, rotationMatrixFromVec, savePathDefault, \
                       momentOfInertiaTensor, rotationMatricesFromInertiaTensor
from cosmo.load import groupCat, groupCatSingle, snapshotSubset
from cosmo.util import validSnapList
from cosmo.mergertree import mpbSmoothedProperties
from util import simParams

def haloImgSpecs(sP, size, sizeType, nPixels, axes, relCoords, rotation, mpb, **kwargs):
    """ Factor out some box/image related calculations common to all halo plots. """
    assert sizeType in ['rVirial','rHalfMass','rHalfMassStars','codeUnits','pkpc']

    if mpb is None:
        # load halo position and virial radius (of the central zoom halo, or a given halo in a periodic box)
        if sP.isZoom:
            shID = sP.zoomSubhaloID
        else:
            #shID = sP.matchedSubhaloID()
            shID = sP.hInd # assume direct input of subhalo ID

        if shID == -1 or shID is None: # e.g. a blank panel
            return None, None, None, None, None, None

        sh = groupCatSingle(sP, subhaloID=shID)
        gr = groupCatSingle(sP, haloID=sh['SubhaloGrNr'])

        if gr['GroupFirstSub'] != shID:
            print('WARNING! Rendering a non-central subhalo [id %d z = %.2f]...' % (shID,sP.redshift))

        haloVirRad = gr['Group_R_Crit200']
        galHalfMassRad = sh['SubhaloHalfmassRad']
        galHalfMassRadStars = sh['SubhaloHalfmassRadType'][sP.ptNum('stars')]
        boxCenter  = sh['SubhaloPos'][ axes + [3-axes[0]-axes[1]] ] # permute into axes ordering
    else:
        # use the smoothed MPB properties to get halo properties at this snapshot
        assert sizeType not in ['rHalfMass','rHalfMassStars'] # not implemented

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

    # convert size into code units
    if sizeType == 'rVirial':
        boxSizeImg = size * haloVirRad
    if sizeType == 'rHalfMass':
        boxSizeImg = size * galHalfMassRad
    if sizeType == 'rHalfMassStars':
        boxSizeImg = size * galHalfMassRadStars
        if boxSizeImg == 0.0:
            print(' WARNING! Subhalo HalfmassRadStars==0, switching to HalfmassRad/5.')
            boxSizeImg = size * galHalfMassRad / 5
    if sizeType == 'codeUnits':
        boxSizeImg = size
    if sizeType == 'pkpc':
        boxSizeImg = sP.units.physicalKpcToCodeLength(size)

    boxSizeImg = boxSizeImg * np.array([1.0, 1.0, 1.0]) # same width, height, and depth
    boxSizeImg[1] *= (nPixels[1]/nPixels[0]) # account for aspect ratio

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
        if str(rotation) in ['face-on-j','edge-on-j']:
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
            if str(rotation) == 'face-on-j': target_vec[ 3-axes[0]-axes[1] ] = 1.0

            # edge-on: rotate the galaxy j vector to be aligned with the 2nd (e.g. y) requested axis
            if str(rotation) == 'edge-on-j': target_vec[ axes[1] ] = 1.0

            if target_vec.sum() == 0.0:
                raise Exception('Not implemented.')

            rotMatrix = rotationMatrixFromVec(jVec, target_vec)

        if str(rotation) in ['face-on','edge-on','edge-on-smallest','edge-on-random']:
            # calculate moment of inertia tensor
            I = momentOfInertiaTensor(sP, subhaloID=shID)

            # hardcoded such that face-on must be projecting along z-axis (think more if we want to relax)
            assert 3-axes[0]-axes[1] == 2
            assert axes[0] == 0 and axes[1] == 1 # e.g. if flipped, then edge-on is vertical not horizontal

            # calculate rotation matrix
            rotMatrices = rotationMatricesFromInertiaTensor(I)
            rotMatrix = rotMatrices[rotation]

    return boxSizeImg, boxCenter, extent, haloVirRad, rotMatrix, rotCenter

def renderSingleHalo(panels, plotConfig, localVars, skipExisting=True):
    """ Render view(s) of a single halo in one plot, with a variable number of panels, comparing 
        any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...). """

    # defaults (all panel fields that can be specified)

    hInd        = 0              # zoom halo index
    run         = 'illustris'    # run name
    res         = 1820           # run resolution
    redshift    = 0.0            # run redshift
    partType    = 'gas'         # which particle type to project
    partField   = 'temp'         # which quantity/field to project for that particle type
    #valMinMax  = [min,max]     # stretch colortable between minimum and maximum field values
    rVirFracs   = [1.0]         # draw circles at these fractions of a virial radius
    method      = 'sphMap'      # sphMap, sphMap_global, sphMap_minIP, sphMap_maxIP, voronoi_*, ...
    nPixels     = [1920,1920]   # [1400,1400] number of pixels for each dimension of images when projecting
    size        = 3.0           # side-length specification of imaging box around halo/galaxy center
    sizeType    = 'rVirial'     # size is multiplying [rVirial,rHalfMass,rHalfMassStars] or in [codeUnits,pkpc]
    hsmlFac     = 2.5           # multiplier on smoothing lengths for sphMap
    axes        = [1,0]         # e.g. [0,1] is x,y
    vecOverlay  = False         # add vector field quiver/streamlines on top? then name of field [bfield,vel]
    vecMethod   = 'E'           # method to use for vector vis: A, B, C, D, E, F (see common.py)
    vecMinMax   = None          # stretch vector field visualizaton between these bounds (None=automatic)
    vecColorPT  = None          # partType to use for vector field vis coloring (if None, =partType)
    vecColorPF  = None          # partField to use for vector field vis coloring (if None, =partField)
    vecColorbar = False         # add additional colorbar for the vector field coloring
    vecColormap = 'afmhot'      # default colormap to use when showing quivers or streamlines
    labelZ      = False         # label redshift inside (upper right corner) of panel
    labelScale  = False         # label spatial scale with scalebar (upper left of panel)
    labelSim    = False         # label simulation name (lower right corner) of panel
    labelHalo   = False         # label halo total mass and stellar mass
    labelCustom = False         # custom label string to include
    relCoords   = True          # if plotting x,y,z coordinate labels, make them relative to box/halo center
    rotation    = None          # 'face-on', 'edge-on', or None
    mpb         = None          # use None for non-movie/single frame

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle  = 'open'  # open, edged, open_black, edged_black
        rasterPx   = 1400    # each panel will have this number of pixels if making a raster (png) output
                             # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars  = True    # include colorbars

        saveFilename = savePathDefault + 'renderHalo_N%d_%s.pdf' % (len(panels),datetime.now().strftime('%d-%m-%Y'))

    # skip if final output render file already exists?
    if skipExisting and isfile(plotConfig.saveFilename):
        print('SKIP: %s' % plotConfig.saveFilename)
        return

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig,var):
            setattr(plotConfig,var,getattr(plotConfigDefaults,var))

    # finalize panels list (insert defaults as necessary)
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

        # add simParams info
        v = p['variant'] if 'variant' in p else None
        p['sP'] = simParams(res=p['res'], run=p['run'], redshift=p['redshift'], hInd=p['hInd'], variant=v)

        # add imaging config for single halo view
        p['boxSizeImg'], p['boxCenter'], p['extent'], \
        p['haloVirRad'], p['rotMatrix'], p['rotCenter'] = haloImgSpecs(**p)

    # request render and save
    renderMultiPanel(panels, plotConfig)

def renderSingleHaloFrames(panels, plotConfig, localVars, skipExisting=True):
    """ Render view(s) of a single halo in one plot, and repeat this frame across all snapshots 
    using the smoothed MPB properties. """

    # defaults (all panel fields that can be specified)

    hInd        = 2               # zoom halo index
    run         = 'zooms2'        # run name
    res         = 9               # run resolution
    redshift    = 2.0             # run redshift
    partType    = 'gas'           # which particle type to project
    partField   = 'temp'          # which quantity/field to project for that particle type
    #valMinMax  = [min,max]       # stretch colortable between minimum and maximum field values
    rVirFracs   = [0.15,0.5,1.0]  # draw circles at these fractions of a virial radius
    method      = 'sphMap'        # sphMap, sphMap_global, sphMap_minIP, sphMap_maxIP, voronoi_*, ...
    nPixels     = [1400,1400]     # number of pixels for each dimension of images when projecting
    size        = 3.0             # side-length specification of imaging box around halo/galaxy center
    sizeType    = 'rVirial'       # size is multiplying [rVirial,rHalfMass,rHalfMassStars] or in [codeUnits,pkpc]
    hsmlFac     = 2.5             # multiplier on smoothing lengths for sphMap
    axes        = [1,0]           # e.g. [0,1] is x,y
    vecOverlay  = False           # add vector field quiver/streamlines on top? then name of field [bfield,vel]
    vecMethod   = 'E'             # method to use for vector vis: A, B, C, D, E, F (see common.py)
    vecMinMax   = None            # stretch vector field visualizaton between these bounds (None=automatic)
    vecColorPT  = None            # partType to use for vector field vis coloring (if None, =partType)
    vecColorPF  = None            # partField to use for vector field vis coloring (if None, =partField)
    vecColorbar = False           # add additional colorbar for the vector field coloring
    vecColormap = 'afmhot'        # default colormap to use when showing quivers or streamlines
    labelZ      = False           # label redshift inside (upper right corner) of panel
    labelScale  = False           # label spatial scale with scalebar (upper left of panel)
    labelSim    = False           # label simulation name (lower right corner) of panel
    labelHalo   = False           # label halo total mass and stellar mass
    labelCustom = False         # custom label string to include
    relCoords   = True            # if plotting x,y,z coordinate labels, make them relative to box/halo center
    rotation    = None            # 'face-on', 'edge-on', or None

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle  = 'open' # open, edged, open_black, edged_black
        rasterPx   = 1200   # each panel will have this number of pixels if making a raster (png) output
                            # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True    # include colorbars

        savePath = savePathDefault
        saveFileBase = 'renderHaloFrame' # filename base upon which frame numbers are appended

        # movie config
        treeRedshift = 2.0       # at what redshift does the tree/MPB start (for periodic box, snap of hInd)
        minRedshift  = 2.0       # ending redshift of frame sequence (we go forward in time)
        maxRedshift  = 10.0      # starting redshift of frame sequence (we go forward in time)
        maxNumSnaps  = None      # make at most this many evenly spaced frames, or None for all

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig,var):
            setattr(plotConfig,var,getattr(plotConfigDefaults,var))

    # load MPB properties for each panel, could be e.g. different runs (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName,cVal in localVars.iteritems():
            if cName in ['panels','plotConfig','plotConfigDefaults','simParams','sP','p']:
                continue
            if cName in p:
                print('Warning: Letting panel specification ['+cName+'] override common value.')
                continue
            p[cName] = cVal

        for cName,cVal in locals().iteritems():
            if cName in p or cName in ['panels','plotConfig','plotConfigDefaults','simParams','p']:
                continue
            p[cName] = cVal

        # load MPB once per panel
        v = p['variant'] if 'variant' in p else None
        sP = simParams(res=p['res'], run=p['run'], hInd=p['hInd'], 
                       redshift=plotConfig.treeRedshift, variant=v)

        p['shID'] = sP.zoomSubhaloID if sP.isZoom else sP.hInd # direct input of subhalo ID for periodic box
        p['mpb'] = mpbSmoothedProperties(sP, p['shID'])

    # determine frame sequence (as the last sP in panels is used somewhat at random, we are here 
    # currently assuming that all runs in panels have the same snapshot configuration)
    snapNums = validSnapList(sP, maxNum=plotConfig.maxNumSnaps, 
                                 minRedshift=plotConfig.minRedshift, maxRedshift=plotConfig.maxRedshift)
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
        plotConfig.saveFilename = plotConfig.savePath + plotConfig.saveFileBase + '_%03d.png' % (frameNum)
        frameNum += 1

        if skipExisting and isfile(plotConfig.saveFilename):
            print('SKIP: %s' % plotConfig.saveFilename)
            continue

        renderMultiPanel(panels, plotConfig)

def selectHalosFromMassBin(sP, massBins, numPerBin, haloNum=None, massBinInd=None, selType='linear'):
    """ Select one or more subhalo indices from an input set of massBins (log Mhalo), a requested 
    number of halos per bin, and either (i) an index haloNum which should iterate from 0 to the total 
    number of halos requested across all bins, in which case the return is a single subhalo ID 
    as appropriate for a multi-quantity single system comparison figure, or (ii) an index massBinInd 
    which should iterate from 0 to the number of bins, in which case all subhalo IDs in that bin 
    are returned (limited to numPerBin), as appropriate for a multi-system single-quantity figure. """
    assert selType in ['linear','even','random']
    from util.helper import evenlySample

    gc = groupCat(sP, fieldsHalos=['Group_M_Crit200','GroupFirstSub'])
    haloMasses = sP.units.codeMassToLogMsun(gc['halos']['Group_M_Crit200'])

    # locate # of halos in mass bins (informational only)
    for massBin in massBins:
        w = np.where((haloMasses >= massBin[0]) & (haloMasses < massBin[1]))[0]
        print('selectHalosFromMassBin(): In massBin [%.1f %.1f] have %d halos total.' % \
            (massBin[0],massBin[1],len(w)))

    # choose mass bin
    if haloNum is not None:
        myMassBinInd = int(np.floor(float(haloNum)/numPerBin))
    else:
        myMassBinInd = massBinInd

    myMassBin = massBins[ myMassBinInd ]
    wMassBinAll = np.where((haloMasses >= myMassBin[0]) & (haloMasses < myMassBin[1]))[0]

    #print(gc['halos']['GroupFirstSub'][wMassBinAll])

    # what algorithm to sub-select within mass bin
    if selType == 'linear':
        wMassBin = wMassBinAll[0:numPerBin]
    if selType == 'even':
        wMassBin = evenlySample(wMassBinAll, numPerBin)
    if selType == 'random':
        np.random.seed(seed=424242+sP.snap+sP.res+int(myMassBin[0]*100)+int(myMassBin[1]*100))
        wMassBin = sorted(np.random.choice(wMassBinAll, size=numPerBin, replace=False))

    if haloNum is not None:
        haloInd = haloNum - myMassBinInd*numPerBin

        # job past requested range, tell to skip
        if haloInd >= len(wMassBin):
            return None, None

        # single halo ID return
        shIDs = gc['halos']['GroupFirstSub'][wMassBin[haloInd]]

        print('[%d] Render halo [%d] subhalo [%d] from massBin [%.1f %.1f] ind [%d of %d]...' % \
            (haloNum,wMassBin[haloInd],shIDs,myMassBin[0],myMassBin[1],haloInd,len(wMassBin)))
    else:
        # return full set in this mass bin
        shIDs = gc['halos']['GroupFirstSub'][wMassBin]

        #for i, shID in enumerate(shIDs):
        #    print('[%d] Render halo [%d] subhalo [%d] from massBin [%.1f %.1f]...' % \
        #        (i,wMassBin[i],shID,myMassBin[0],myMassBin[1]))

    return shIDs, myMassBinInd
