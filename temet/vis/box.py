"""
Visualizations for whole (cosmological) boxes.
"""

from copy import deepcopy
from os import makedirs
from os.path import isdir, isfile

import numpy as np

from ..cosmo.util import multiRunMatchedSnapList
from ..util.boxRemap import findCuboidRemapInds
from ..util.helper import iterable, pSplit
from ..util.rotation import rotationMatrixFromAngleDirection
from ..util.simParams import simParams
from ..vis.common import renderMultiPanel
from ..vis.render import defaultHsmlFac, gridBox


def boxImgSpecs(sP, zoomFac, sliceFac, relCenPos, absCenPos, axes, nPixels, boxOffset, remapRatio, **kwargs):
    """Factor out some box/image related calculations common to all whole box plots.

    Image zoomFac fraction of entire fullbox/subbox, zooming around relCenPos ([0.5,0.5] being box center point).
    """
    assert relCenPos is None or absCenPos is None
    if remapRatio is not None:
        assert sP.subbox is None

    if sP.subbox is None:
        if remapRatio is None:
            # standard periodic full-box
            boxSizeImg = np.array([sP.boxSize, sP.boxSize, sP.boxSize])
        else:
            # periodic -> cuboid remapping
            remapMatrix, newBoxSize = findCuboidRemapInds(remapRatio, nPixels)
            boxSizeImg = np.array(newBoxSize) * sP.boxSize

        if absCenPos is None:
            boxCenter = [relCenPos[0], relCenPos[1], 0.5] * np.array(boxSizeImg)
        else:
            boxCenter = absCenPos
    else:
        boxSizeImg = sP.subboxSize[sP.subbox] * np.array([1, 1, 1])

        boxCenter0 = (
            relCenPos[0] * sP.subboxCen[sP.subbox][axes[0]]
            - sP.subboxSize[sP.subbox] * 0.5
            + (1.0 - relCenPos[0]) * sP.subboxCen[sP.subbox][axes[0]]
            + sP.subboxSize[sP.subbox] * 0.5
        )

        boxCenter1 = (
            relCenPos[1] * sP.subboxCen[sP.subbox][axes[1]]
            - sP.subboxSize[sP.subbox] * 0.5
            + (1.0 - relCenPos[1]) * sP.subboxCen[sP.subbox][axes[1]]
            + sP.subboxSize[sP.subbox] * 0.5
        )

        boxCenter2 = sP.subboxCen[sP.subbox][3 - axes[0] - axes[1]]
        boxCenter = np.array([boxCenter0, boxCenter1, boxCenter2])

    # non-square aspect ratio
    if isinstance(nPixels, (list, np.ndarray)) and remapRatio is None:
        aspect = float(nPixels[0]) / nPixels[1]
        boxSizeImg[1] /= aspect  # e.g. 16/9 = 1.778 decreases vertical height to 56.25% of original

    boxSizeImg[0] *= zoomFac
    boxSizeImg[1] *= zoomFac
    boxSizeImg[2] *= sliceFac

    for i in range(3):
        boxCenter[i] += boxOffset[i]

    extent = [
        boxCenter[0] - 0.5 * boxSizeImg[0],
        boxCenter[0] + 0.5 * boxSizeImg[0],
        boxCenter[1] - 0.5 * boxSizeImg[1],
        boxCenter[1] + 0.5 * boxSizeImg[1],
    ]

    return boxSizeImg, boxCenter, extent


def renderBox(panels_in, plotConfig=None, localVars=None, skipExisting=False, retInfo=False, returnData=False):
    """Render views of a full cosmological box (or a zoomed subset), with a variable number of image panels.

    These can compare any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...).
    """
    panels = deepcopy(panels_in)

    # defaults (all panel fields that can be specified)
    # run         = 'tng'       # run name
    # res         = 1820        # run resolution
    # redshift    = 0.0         # run redshift
    partType = "dm"  # which particle type to project
    partField = "coldens"  # which quantity/field to project for that particle type
    valMinMax = None  # if not None (auto), then stretch colortable between 2-tuple [min,max] field values
    method = "sphMap"  # sphMap[_subhalo,_global], sphMap_{min/max}IP, histo, voronoi_slice/proj[_subhalo,_global]
    nPixels = 1400  # number of pixels per dimension of images when projecting (960 1400)
    zoomFac = 1.0  # [0,1], only in axes, not along projection direction
    # hsmlFac     = 1.0        # multiplier on smoothing lengths for sphMap (dm 0.2) (gas 2.5)
    ptRestrictions = None  # dictionary of particle-level restrictions to apply
    relCenPos = [0.5, 0.5]  # [0-1,0-1] relative coordinates of where to center image, only in axes
    absCenPos = None  # [x,y,z] in simulation coordinates to place at center of image
    sliceFac = 1.0  # [0,1], only along projection direction, relative depth wrt boxsize
    axes = [0, 1]  # e.g. [0,1] is x,y
    boxOffset = [0, 0, 0]  # offset in x,y,z directions (code units) from fiducial center
    axesUnits = "code"  # code [ckpc/h], kpc, mpc, deg, arcmin, arcsec
    labelZ = False  # label redshift inside (upper right corner) of panel (True or 'tage')
    labelScale = False  # label spatial scale with scalebar (upper left of panel) (True, 'physical', or 'lightyears')
    labelSim = False  # label simulation name (lower right corner) of panel
    labelCustom = False  # custom label string to include
    ctName = None  # if not None (automatic based on field), specify colormap name
    plotHalos = 20  # plot virial circles for the N most massive halos in the box
    labelHalos = False  # label halo virial circles with values like M*, Mhalo, SFR
    projType = "ortho"  # projection type, 'ortho', 'equirectangular', 'mollweide'
    projParams = {}  # dictionary of parameters associated to this projection type
    rotMatrix = None  # rotation matrix
    rotCenter = None  # rotation center
    remapRatio = None  # [x,y,z] periodic->cuboid remapping ratios, or None

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle = "open"  # open, edged, open_black, edged_black
        rasterPx = [1000, 1000]  # each panel will have this number of pixels if making a raster (png) output
        # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True  # include colorbars
        colorbarOverlay = False  # overlay on top of image
        title = True  # include title (only for open* styles)
        outputFmt = None  # if not None (automatic), then a format string for the matplotlib backend

        _sim_str = ""
        _field_str = ""
        if all("sP" in p for p in panels) and len(panels) <= 2:
            _sim_str = "_" + "-".join([p["sP"].simName for p in panels])
        if all("partType" in p for p in panels) and all("partField" in p for p in panels) and len(panels) <= 2:
            _field_str = "_" + "_".join(["%s-%s" % (p["partType"], p["partField"]) for p in panels])
        saveFilename = "renderBox_N%d%s%s.jpg" % (len(panels), _sim_str, _field_str)

    if plotConfig is None:
        plotConfig = plotConfigDefaults()
    if isinstance(plotConfig, dict):
        # todo: remove this backward compatibility hack (plotConfig should just be a dict in the future)
        config = plotConfigDefaults()
        for k, v in plotConfig.items():
            setattr(config, k, v)
        plotConfig = config
    if localVars is None:
        localVars = {}

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig, var):
            setattr(plotConfig, var, getattr(plotConfigDefaults, var))

    if not isinstance(plotConfig.rasterPx, list):
        plotConfig.rasterPx = [plotConfig.rasterPx, plotConfig.rasterPx]

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName, cVal in localVars.items():
            if cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            if cName in p:
                print("Warning: Letting panel specification [" + cName + "] override common value.")
                continue
            p[cName] = cVal

        for cName, cVal in locals().items():
            if cName in p or cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            p[cName] = cVal

        if "hsmlFac" not in p:
            p["hsmlFac"] = defaultHsmlFac(p["partType"])

        # add simParams info if not directly input
        if ("run" in p) or ("sP" not in p):
            if "run" in p and "sP" in p:
                print("Warning: Both sP and run specified in panel, too ambiguous.")

            v = p["variant"] if "variant" in p else None
            h = p["hInd"] if "hInd" in p else None
            s = p["snap"] if "snap" in p else None
            z = p["redshift"] if "redshift" in p and s is None else None  # skip if snap specified
            rp = p["refPos"] if "refPos" in p else None
            rv = p["refVel"] if "refVel" in p else None

            p["sP"] = simParams(res=p["res"], run=p["run"], redshift=z, snap=s, hInd=h, variant=v)
            p["sP"].refPos = rp
            p["sP"].refVel = rv

        # allow modifications of sP to vary e.g. resolution, redshift, variant
        if "redshift" in p:
            sP_loc = p["sP"].copy()
            sP_loc.setRedshift(p["redshift"])
            if sP_loc.snap != p["sP"].snap:
                print("Overriding sP snap to match redshift for panel.")
            p["sP"] = sP_loc

        # add imaging config for render of the whole box, if not directly specified
        boxSizeImg_loc, boxCenter_loc, extent_loc = boxImgSpecs(**p)
        if "boxSizeImg" not in p:
            p["boxSizeImg"] = boxSizeImg_loc
        if "boxCenter" not in p:
            p["boxCenter"] = boxCenter_loc
        if "extent" not in p:
            p["extent"] = extent_loc

        if not isinstance(p["nPixels"], list):
            p["nPixels"] = [p["nPixels"], p["nPixels"]]

    # attach any cached data to sP (testing)
    if "dataCache" in localVars:
        for key in localVars["dataCache"]:
            for p in panels:
                p["sP"].data[key] = localVars["dataCache"][key]

    # request render and save
    if retInfo:
        return panels

    # request raw data grid and return?
    if returnData:
        assert len(panels) == 1  # otherwise could return a list of grids
        _, config, data_grid = gridBox(**panels[0])
        return data_grid, config

    # skip if final output render file already exists?
    if skipExisting and isfile(plotConfig.saveFilename):
        print("SKIP: %s" % plotConfig.saveFilename)
        return

    renderMultiPanel(panels, plotConfig)


def renderBoxFrames(panels_in, plotConfig=None, localVars=None, curTask=0, numTasks=1, skipExisting=True):
    """Render views of a cosmological box, with a variable number of panels, for many snapshots to make a movie."""
    panels = deepcopy(panels_in)

    # defaults (all panel fields that can be specified)
    run = "illustris"  # run name
    res = 1820  # run resolution
    partType = "dm"  # which particle type to project
    partField = "coldens"  # which quantity/field to project for that particle type
    valMinMax = None  # if not None (auto), then stretch colortable between 2-tuple [min,max] field values
    method = "sphMap"  # sphMap[_subhalo,_global], sphMap_{min/max}IP, histo, voronoi_slice/proj[_subhalo,_global]
    nPixels = 960  # number of pixels per dimension of images when projecting
    zoomFac = 1.0  # [0,1], only in axes, not along projection direction
    # hsmlFac     = 2.5        # multiplier on smoothing lengths for sphMap
    ptRestrictions = None  # dictionary of particle-level restrictions to apply
    relCenPos = [0.5, 0.5]  # [0-1,0-1] relative coordinates of where to center image, only in axes
    absCenPos = None  # [x,y,z] in simulation coordinates to place at center of image
    sliceFac = 1.0  # [0,1], only along projection direction, relative depth wrt boxsize
    axes = [0, 1]  # e.g. [0,1] is x,y
    boxOffset = [0, 0, 0]  # offset in x,y,z directions (code units) from fiducial center
    axesUnits = "code"  # code [ckpc/h], kpc, mpc, deg, arcmin, arcsec
    labelZ = False  # label redshift inside (upper right corner) of panel
    labelScale = False  # label spatial scale with scalebar (upper left of panel) (True or 'physical')
    labelSim = False  # label simulation name (lower right corner) of panel
    labelCustom = False  # custom label string to include
    ctName = None  # if not None (automatic based on field), specify colormap name
    plotHalos = 0  # plot virial circles for the N most massive halos in the box
    labelHalos = False  # label halo virial circles with values like M*, Mhalo, SFR
    projType = "ortho"  # projection type, 'ortho', 'equirectangular', 'mollweide'
    projParams = {}  # dictionary of parameters associated to this projection type
    rotMatrix = None  # rotation matrix
    rotCenter = None  # rotation center
    rotSequence = None  # rotation sequence [numFramesPerRot, dirVec]
    remapRatio = None  # [x,y,z] periodic->cuboid remapping ratios, or None

    # defaults (global plot configuration options)
    class plotConfigDefaults:
        plotStyle = "open"  # open, edged, open_black, edged_black
        rasterPx = [1000, 1000]  # each panel will have this number of pixels if making a raster (png) output
        # but it also controls the relative size balance of raster/vector (e.g. fonts)
        colorbars = True  # include colorbars
        colorbarOverlay = False  # overlay on top of image
        title = True  # include title (only for open* styles)
        outputFmt = None  # if not None (automatic), then a format string for the matplotlib backend

        savePath = ""  # savePathDefault
        saveFileBase = "renderBoxFrame"  # filename base upon which frame numbers are appended

        # movie config
        minZ = 0.0  # ending redshift of frame sequence (we go forward in time)
        maxZ = 128.0  # starting redshift of frame sequence (we go forward in time)
        maxNSnaps = None  # make at most this many evenly spaced frames, or None for all
        matchUse = "condense"  # 'expand' or 'condense' to determine matching snaps between runs

    if plotConfig is None:
        plotConfig = plotConfigDefaults()
    if isinstance(plotConfig, dict):
        # todo: remove this backward compatibility hack (plotConfig should just be a dict in the future)
        config = plotConfigDefaults()
        for k, v in plotConfig.items():
            setattr(config, k, v)
        plotConfig = config
    if localVars is None:
        localVars = {}

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig, var):
            setattr(plotConfig, var, getattr(plotConfigDefaults, var))

    if not isinstance(plotConfig.rasterPx, list):
        plotConfig.rasterPx = [plotConfig.rasterPx, plotConfig.rasterPx]

    if not isdir(plotConfig.savePath):
        print(f"Note: save path [{plotConfig.savePath}] does not exist, creating it.")
        makedirs(plotConfig.savePath)

    # finalize panels list (do not modify below)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName, cVal in localVars.items():
            if cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            if cName in p:
                print("Warning: Letting panel specification [" + cName + "] override common value.")
                continue
            p[cName] = cVal

        for cName, cVal in locals().items():
            if cName in p or cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            p[cName] = cVal

        if "hsmlFac" not in p:
            p["hsmlFac"] = defaultHsmlFac(p["partType"])

        # add simParams info
        v = p["variant"] if "variant" in p else None
        h = p["hInd"] if "hInd" in p else None
        s = p["snap"] if "snap" in p else None
        z = p["redshift"] if "redshift" in p and s is None else None  # skip if snap specified
        rp = p["refPos"] if "refPos" in p else None
        rv = p["refVel"] if "refVel" in p else None

        p["sP"] = simParams(res=p["res"], run=p["run"], redshift=z, snap=s, hInd=h, variant=v)
        p["sP"].refPos = rp
        p["sP"].refVel = rv

        # add imaging config for [square render of] whole box
        if not isinstance(p["nPixels"], list):
            p["nPixels"] = [p["nPixels"], p["nPixels"]]

    # determine frame sequence
    snapNumLists = multiRunMatchedSnapList(
        panels,
        plotConfig.matchUse,
        maxNum=plotConfig.maxNSnaps,
        minRedshift=plotConfig.minZ,
        maxRedshift=plotConfig.maxZ,
    )

    numFramesTot = snapNumLists[0].size

    # optionally parallelize over multiple tasks
    fNumsThisTask = pSplit(range(numFramesTot), numTasks, curTask)

    print(
        "Task [%d of %d] rendering [%d] frames of [%d] total (from %d to %d)..."
        % (curTask, numTasks, len(fNumsThisTask), numFramesTot, np.min(fNumsThisTask), np.max(fNumsThisTask))
    )

    # render sequence
    for frameNum in fNumsThisTask:
        snapNumsStr = " ".join([str(s) for s in [iterable(snapList)[frameNum] for snapList in snapNumLists]])
        print("\nFrame [%d of %d]: using snapshots [%s]" % (frameNum, numFramesTot - 1, snapNumsStr))

        # finalize panels list (all properties not set here are invariant in time)
        for i, p in enumerate(panels):
            # override simParams info at this snapshot
            snapNum = iterable(snapNumLists[i])[frameNum]
            p["sP"] = simParams(res=p["sP"].res, run=p["sP"].run, variant=p["sP"].variant, snap=snapNum)

            # setup currenty constant in time, could here give a rotation/zoom/etc with time
            p["boxSizeImg"], p["boxCenter"], p["extent"] = boxImgSpecs(**p)

            # e.g. update the upper bound of 'stellar_age' valMinMax, if set, to the current tAge [in Gyr]
            # if p['partField'] == 'stellar_age' and p['valMinMax'] is not None:
            #    p['valMinMax'][1] = np.max( [p['sP'].units.redshiftToAgeFlat(p['sP'].redshift), 3.0] )

            # update rotation matrix if we are rotating in time
            if p["rotSequence"] is not None:
                numFramesPerRot, dirVec = p["rotSequence"]
                rotAngleDeg = 360.0 * (frameNum / numFramesPerRot)
                p["rotCenter"] = p["boxCenter"]
                p["rotMatrix"] = rotationMatrixFromAngleDirection(rotAngleDeg, dirVec)

        # request render and save
        plotConfig.saveFilename = plotConfig.savePath + plotConfig.saveFileBase + "_%04d.png" % (frameNum)

        if skipExisting and isfile(plotConfig.saveFilename):
            print("SKIP: " + plotConfig.saveFilename)
            continue

        renderMultiPanel(panels, plotConfig)
