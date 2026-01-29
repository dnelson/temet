"""
Visualizations for individual halos/subhalos from ..cosmological runs.
"""

from copy import deepcopy
from getpass import getuser
from os.path import isfile

import numpy as np

from ..util.helper import evenlySample
from ..util.rotation import (
    meanAngMomVector,
    momentOfInertiaTensor,
    rotationMatricesFromInertiaTensor,
    rotationMatrixFromAngleDirection,
    rotationMatrixFromVec,
)
from ..util.simParams import simParams
from ..vis.common import renderMultiPanel
from ..vis.render import defaultHsmlFac, gridBox


def haloImgSpecs(
    sP,
    size,
    sizeType,
    nPixels,
    axes,
    relCoords,
    rotation,
    inclination,
    mpb,
    cenShift,
    depthFac,
    depth,
    depthType,
    **kwargs,
):
    """Factor out some box/image related calculations common to all halo plots."""
    assert sizeType in ["rVirial", "r500", "rHalfMass", "rHalfMassStars", "codeUnits", "kpc", "arcsec", "arcmin"]

    if mpb is None:
        # load halo position and virial radius (of the central zoom halo, or a given halo in a periodic box)
        if sP.subhaloInd == -1 or sP.subhaloInd is None:  # e.g. a blank panel
            return None, None, None, None, None, None, None, None

        sh = sP.groupCatSingle(subhaloID=sP.subhaloInd)
        gr = sP.groupCatSingle(haloID=sh["SubhaloGrNr"])

        if gr["GroupFirstSub"] != sP.subhaloInd and kwargs["fracsType"] == "rVirial" and getuser() != "wwwrun":
            print("WARNING! Rendering a non-central subhalo [id %d z = %.2f]..." % (sP.subhaloInd, sP.redshift))

        sP.refPos = sh["SubhaloPos"]
        sP.refVel = sh["SubhaloVel"]
        sP.refSubhalo = sh

        haloVirRad = gr["Group_R_Crit200"]
        haloR500 = gr["Group_R_Crit500"]
        galHalfMassRad = sh["SubhaloHalfmassRad"]
        galHalfMassRadStars = sh["SubhaloHalfmassRadType"][sP.ptNum("stars")]
        boxCenter = sh["SubhaloPos"][axes + [3 - axes[0] - axes[1]]]  # permute into axes ordering
    else:
        # use the smoothed MPB properties to get halo properties at this snapshot
        assert sizeType not in ["rHalfMass", "r500", "rHalfMassStars"]  # not implemented
        assert (sP.refPos is None) and (sP.refPos is None)  # will overwrite

        if sP.snap < mpb["SnapNum"].min():
            # for very early times, linearly interpolate properties at start of tree back to t=0
            if rotation is not None:
                raise Exception("Cannot use rotation (or any group-ordered load) prior to mpb start.")

            fitSize = np.max([int(mpb["SnapNum"].size * 0.02), 3])
            fitN = 1  # polynomial order, 1=linear, 2=quadratic

            fitX = mpb["SnapNum"][-fitSize:]

            sP.subhaloInd = 0
            haloVirRad = np.poly1d(np.polyfit(fitX, mpb["Group_R_Crit200"][-fitSize:], fitN))(sP.snap)
            galHalfMassRad = np.poly1d(np.polyfit(fitX, mpb["SubhaloHalfmassRad"][-fitSize:], fitN))(sP.snap)
            galHalfMassRadStars = np.poly1d(
                np.polyfit(fitX, mpb["SubhaloHalfmassRadType"][-fitSize:, sP.ptNum("stars")], fitN)
            )(sP.snap)

            boxCenter = np.zeros(3, dtype="float32")
            galVel = np.zeros(3, dtype="float32")

            for i in range(3):
                boxCenter[i] = np.poly1d(np.polyfit(fitX, mpb["SubhaloPos"][-fitSize:, i], fitN))(sP.snap)
                galVel[i] = np.poly1d(np.polyfit(fitX, mpb["SubhaloVel"][-fitSize:, i], fitN))(sP.snap)

        else:
            # for times within actual MPB, use smoothed properties directly
            ind = np.where(mpb["SnapNum"] == sP.snap)[0]
            assert len(ind)

            sP.subhaloInd = mpb["SubfindID"][ind[0]]
            haloVirRad = mpb["Group_R_Crit200"][ind[0]]
            boxCenter = mpb["SubhaloPos"][ind[0], :]
            boxCenter = boxCenter[axes + [3 - axes[0] - axes[1]]]  # permute into axes ordering
            galHalfMassRad = mpb["SubhaloHalfmassRad"][ind[0]]
            galHalfMassRadStars = mpb["SubhaloHalfmassRadType"][ind[0], sP.ptNum("stars")]
            galVel = mpb["SubhaloVel"][ind[0], :]

        # set refPos and refVel, used e.g. for halo-centric quantities
        sP.refPos = boxCenter.copy()
        sP.refVel = galVel

    boxCenter += np.array(cenShift)

    # convert size into code units
    def _convert_size(s, s_type):
        """Helper. Convert a numeric size [s] given a string type [s_type]."""
        if s_type == "rVirial":
            s_img = s * haloVirRad
        if s_type == "r500":
            s_img = s * haloR500
        if s_type == "rHalfMass":
            s_img = s * galHalfMassRad
        if s_type == "rHalfMassStars":
            s_img = s * galHalfMassRadStars
            if s_img == 0.0:
                s_img = s * galHalfMassRad / 5
        if s_type == "codeUnits":
            s_img = s
        if s_type == "kpc":
            s_img = sP.units.physicalKpcToCodeLength(s)
        if s_type == "arcsec":
            s_pkpc = sP.units.arcsecToAngSizeKpcAtRedshift(s, sP.redshift)
            s_img = sP.units.physicalKpcToCodeLength(s_pkpc)
        if s_type == "arcmin":
            s_pkpc = sP.units.arcsecToAngSizeKpcAtRedshift(s * 60, sP.redshift)
            s_img = sP.units.physicalKpcToCodeLength(s_pkpc)
        if s_type == "deg":
            s_pkpc = sP.units.arcsecToAngSizeKpcAtRedshift(s * 60 * 60, sP.redshift)
            s_img = sP.units.physicalKpcToCodeLength(s_pkpc)

        return s_img

    boxSizeImg = _convert_size(size, sizeType)

    boxSizeImg = boxSizeImg * np.array([1.0, 1.0, 1.0])  # same width, height, and depth
    boxSizeImg[1] *= nPixels[1] / nPixels[0]  # account for aspect ratio

    extent = [
        boxCenter[0] - 0.5 * boxSizeImg[0],
        boxCenter[0] + 0.5 * boxSizeImg[0],
        boxCenter[1] - 0.5 * boxSizeImg[1],
        boxCenter[1] + 0.5 * boxSizeImg[1],
    ]

    # modify depth?
    if depth is None:
        # depthFac modifies size interpreted as sizeType
        boxSizeImg[2] *= depthFac
    else:
        # depthFac modifies depth interpreted as depthType
        boxSizeImg[2] = _convert_size(depth, depthType) * depthFac

    # make coordinates relative
    if relCoords:
        extent[0:2] -= boxCenter[0]
        extent[2:4] -= boxCenter[1]

    # derive appropriate rotation matrix if requested
    rotMatrix = None
    rotCenter = None

    if rotation is not None:
        if str(rotation) in ["face-on-j", "edge-on-j"]:
            # calculate 'mean angular momentum' vector of the galaxy (method choices herein)
            if mpb is None:
                jVec = meanAngMomVector(sP, subhaloID=sP.subhaloInd)
            else:
                shPos = mpb["sm"]["pos"][ind[0], :]
                shVel = mpb["sm"]["vel"][ind[0], :]

                jVec = meanAngMomVector(sP, subhaloID=sP.subhaloInd, shPos=shPos, shVel=shVel)
                rotCenter = shPos

            target_vec = np.zeros(3, dtype="float32")

            # face-on: rotate the galaxy j vector onto the unit axis vector we are projecting along
            if str(rotation) == "face-on-j":
                target_vec[3 - axes[0] - axes[1]] = 1.0

            # edge-on: rotate the galaxy j vector to be aligned with the 2nd (e.g. y) requested axis
            if str(rotation) == "edge-on-j":
                target_vec[axes[1]] = 1.0

            if target_vec.sum() == 0.0:
                raise Exception("Not implemented.")

            rotMatrix = rotationMatrixFromVec(jVec, target_vec)

        if str(rotation) in ["face-on", "edge-on", "edge-on-smallest", "edge-on-random", "edge-on-stars"]:
            # calculate moment of inertia tensor
            onlyStars = False
            rotName = rotation

            if rotation == "edge-on-stars":
                onlyStars = True
                rotName = rotation.replace("-stars", "")

            I = momentOfInertiaTensor(sP, subhaloID=sP.subhaloInd, onlyStars=onlyStars)

            # hardcoded such that face-on must be projecting along z-axis (think more if we want to relax)
            assert 3 - axes[0] - axes[1] == 2
            assert axes[0] == 0 and axes[1] == 1  # e.g. if flipped, then edge-on is vertical not horizontal

            # calculate rotation matrix
            rotMatrices = rotationMatricesFromInertiaTensor(I)
            rotMatrix = rotMatrices[rotName]

    if inclination is not None:
        # derive additional rotation matrix for inclination angle request
        incRotMatrix = rotationMatrixFromAngleDirection(inclination, [1, 0, 0])

        # if rotMatrix already exists, multiply our inclination rotation matrix in
        if rotMatrix is not None:
            rotMatrix = np.dot(incRotMatrix, rotMatrix)
        else:
            rotMatrix = incRotMatrix

    return boxSizeImg, boxCenter, extent, haloVirRad, galHalfMassRad, galHalfMassRadStars, rotMatrix, rotCenter


def renderSingleHalo(panels_in, plotConfig=None, localVars=None, skipExisting=False, returnData=False):
    """Render view(s) of a single halo in one plot, with a variable number of panels.

    Compare any combination of parameters (res, run, redshift, vis field, vis type, vis direction, ...).
    """
    panels = deepcopy(panels_in)

    # defaults (all panel fields that can be specified)
    # fmt: off
    #subhaloInd  = 0            # subhalo (subfind) index to visualize
    hInd        = None          # halo index for zoom run
    #run         = 'illustris'  # run name
    #res         = 1820         # run resolution
    #redshift    = 0.0          # run redshift
    partType    = 'gas'         # which particle type to project
    partField   = 'temp'        # which quantity/field to project for that particle type
    valMinMax   = None          # if not None (auto), then stretch colortable between 2-tuple [min,max] field values
    rVirFracs   = [1.0]         # draw circles at these fractions of a virial radius
    fracsType   = 'rVirial'     # if not rVirial, draw circles at fractions of another quant, same as sizeType
    method      = 'sphMap'      # sphMap[_subhalo,_global], sphMap_{min/max}IP, histo, voronoi_slice/proj[_subhalo]
    nPixels     = [1920,1920]   # [1400,1400] number of pixels for each dimension of images when projecting
    cenShift    = [0,0,0]       # [x,y,z] coordinates to shift default box center location by
    size        = 3.0           # side-length specification of imaging box around halo/galaxy center
    depthFac    = 1.0           # projection depth, relative to size (1.0=same depth as width and height)
    sizeType    = 'rVirial'     # size units [rVirial,r500,rHalfMass,rHalfMassStars,codeUnits,kpc,arcsec,arcmin,deg]
    depth       = None          # if None, depth is taken as size*depthFac, otherwise depth is provided here
    depthType   = 'rVirial'     # as sizeType except for depth, if depth is not None
    #hsmlFac     = 2.5          # multiplier on smoothing lengths for sphMap
    ptRestrictions = None       # dictionary of particle-level restrictions to apply
    axes        = [0,1]         # e.g. [0,1] is x,y
    axesUnits   = 'code'        # code [ckpc/h], kpc, mpc, deg, arcmin, arcsec
    vecOverlay  = False         # add vector field quiver/streamlines on top? then name of field [bfield,vel]
    vecMethod   = 'E'           # method to use for vector vis: A, B, C, D, E, F (see common.py)
    vecMinMax   = None          # stretch vector field visualizaton between these bounds (None=automatic)
    vecColorPT  = None          # partType to use for vector field vis coloring (if None, =partType)
    vecColorPF  = None          # partField to use for vector field vis coloring (if None, =partField)
    vecColorbar = False         # add additional colorbar for the vector field coloring
    vecColormap = 'afmhot'      # default colormap to use when showing quivers or streamlines
    labelZ      = False         # label redshift inside (upper right corner) of panel {True, tage}
    labelScale  = False         # label spatial scale with scalebar (upper left of panel) {True, physical, lightyears}
    labelSim    = False         # label simulation name (lower right corner) of panel
    labelHalo   = False         # label halo total mass and stellar mass
    labelCustom = False         # custom label string to include
    ctName      = None          # if not None (automatic based on field), specify colormap name
    plotSubhalos = False        # plot halfmass circles for the N most massive subhalos in this (sub)halo
    plotBHs     = False         # plot markers for the N most massive SMBHs in this (sub)halo
    relCoords   = True          # if plotting x,y,z coordinate labels, make them relative to box/halo center
    projType    = 'ortho'       # projection type, 'ortho', 'equirectangular', 'mollweide'
    projParams  = {}            # dictionary of parameters associated to this projection type
    rotation    = None          # 'face-on', 'edge-on', 'edge-on-stars', or None
    inclination = None          # inclination angle (degrees, about the x-axis) (0=unchanged)
    rotMatrix   = None          # rotation matrix, i.e. manually specify if rotation is None
    rotCenter   = None          # rotation center, i.e. manually specify if rotation is None
    mpb         = None          # use None for non-movie/single frame
    remapRatio  = None          # [x,y,z] periodic->cuboid remapping ratios, always None for single halos
    # fmt: on

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
        saveFilename = "renderHalo_N%d%s%s.jpg" % (len(panels), _sim_str, _field_str)

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

    # skip if final output render file already exists?
    if skipExisting and hasattr(plotConfig, "saveFilename") and isfile(plotConfig.saveFilename) and not returnData:
        print("SKIP: %s" % plotConfig.saveFilename)
        return

    # add plotConfig defaults
    for var in [v for v in vars(plotConfigDefaults) if not v.startswith("__")]:
        if not hasattr(plotConfig, var):
            setattr(plotConfig, var, getattr(plotConfigDefaults, var))

    if not isinstance(plotConfig.rasterPx, list):
        plotConfig.rasterPx = [plotConfig.rasterPx, plotConfig.rasterPx]

    # finalize panels list (insert defaults as necessary)
    for p in panels:
        # add all local variables to each (assumed to be common for all panels)
        for cName, cVal in localVars.items():
            if cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            if cName in p:
                print(f"Warning: Letting panel value [{cName} = {p[cName]}] override common value [{cVal}].")
                continue
            p[cName] = cVal

        for cName, cVal in locals().items():
            if cName in p or cName in ["panels", "plotConfig", "plotConfigDefaults", "simParams", "p"]:
                continue
            p[cName] = cVal

        if "hsmlFac" not in p:
            p["hsmlFac"] = defaultHsmlFac(p["partType"])

        # add simParams info if not directly input
        if "run" in p:
            v = p["variant"] if "variant" in p else None
            s = p["snap"] if "snap" in p else None
            r = p["res"] if "res" in p else None
            z = p["redshift"] if "redshift" in p and s is None else None  # skip if snap specified

            if "sP" in p:
                print("Warning: Overriding common sP with specified run,snap,redshift.")

            p["sP"] = simParams(run=p["run"], res=r, redshift=z, snap=s, hInd=p["hInd"], variant=v)

        if "subhaloInd" in p and p["sP"].subhaloInd is None:
            p["sP"] = p["sP"].copy()
            p["sP"].subhaloInd = p["subhaloInd"]

        if "subhaloInd" not in p and p["sP"].subhaloInd is None and p["sP"].isZoom:
            p["sP"].subhaloInd = p["sP"].zoomSubhaloID
            print("Note: Using sP.zoomSubhaloID = %d as subhaloInd for vis." % p["sP"].zoomSubhaloID)

        assert "subhaloInd" in p or p["sP"].subhaloInd is not None, "subhaloInd unspecified!"

        # add imaging config for single halo view
        if not isinstance(p["nPixels"], list):
            p["nPixels"] = [p["nPixels"], p["nPixels"]]

        (
            p["boxSizeImg"],
            p["boxCenter"],
            p["extent"],
            p["haloVirRad"],
            p["galHalfMass"],
            p["galHalfMassStars"],
            haloRotMatrix,
            haloRotCenter,
        ) = haloImgSpecs(**p)

        if p["rotMatrix"] is None:
            p["rotMatrix"], p["rotCenter"] = haloRotMatrix, haloRotCenter

    # attach any cached data to sP (testing)
    if "dataCache" in localVars:
        for key in localVars["dataCache"]:
            for p in panels:
                p["sP"].data[key] = localVars["dataCache"][key]

    # request raw data grid and return?
    if returnData:
        assert len(panels) == 1  # otherwise could return a list of grids
        _, config, data_grid = gridBox(**panels[0])
        return data_grid, config

    # request render and save
    renderMultiPanel(panels, plotConfig)


def renderSingleHaloFrames(panels_in, plotConfig=None, localVars=None, skipExisting=True):
    """Render view(s) of a single halo, repeating across all snapshots using the smoothed MPB properties."""
    panels = deepcopy(panels_in)

    # defaults (all panel fields that can be specified)
    # fmt: off
    #subhaloInd = 0               # subhalo (subfind) index to visualize
    hInd        = None            # halo index for zoom run
    #run        = 'tng'           # run name
    #res        = 1820            # run resolution
    #redshift   = 2.0             # run redshift
    partType    = 'gas'           # which particle type to project
    partField   = 'temp'          # which quantity/field to project for that particle type
    valMinMax   = None            # if not None (auto), then stretch colortable between 2-tuple [min,max] field values
    rVirFracs   = [0.15,0.5,1.0]  # draw circles at these fractions of a virial radius
    fracsType   = 'rVirial'       # if not rVirial, draw circles at fractions of another quant, same as sizeType
    method      = 'sphMap'        # sphMap[_subhalo,_global], sphMap_{min/max}IP, histo, voronoi_slice/proj[_subhalo]
    nPixels     = [1400,1400]     # number of pixels for each dimension of images when projecting
    cenShift    = [0,0,0]         # [x,y,z] coordinates to shift default box center location by
    size        = 3.0             # side-length specification of imaging box around halo/galaxy center
    depthFac    = 1.0             # projection depth, relative to size (1.0=same depth as width and height)
    sizeType    = 'rVirial'       # size units [rVirial,r500,rHalfMass,rHalfMassStars,codeUnits,kpc,arcsec,arcmin,deg]
    depth       = None            # if None, depth is taken as size*depthFac, otherwise depth is provided here
    depthType   = 'rVirial'       # as sizeType except for depth, if depth is not None
    #hsmlFac     = 2.5            # multiplier on smoothing lengths for sphMap
    ptRestrictions = None         # dictionary of particle-level restrictions to apply
    axes        = [0,1]           # e.g. [0,1] is x,y
    axesUnits   = 'code'          # code [ckpc/h], mpc, deg, arcmin, arcsec
    vecOverlay  = False           # add vector field quiver/streamlines on top? then name of field [bfield,vel]
    vecMethod   = 'E'             # method to use for vector vis: A, B, C, D, E, F (see common.py)
    vecMinMax   = None            # stretch vector field visualizaton between these bounds (None=automatic)
    vecColorPT  = None            # partType to use for vector field vis coloring (if None, =partType)
    vecColorPF  = None            # partField to use for vector field vis coloring (if None, =partField)
    vecColorbar = False           # add additional colorbar for the vector field coloring
    vecColormap = 'afmhot'        # default colormap to use when showing quivers or streamlines
    labelZ      = False           # label redshift inside (upper right corner) of panel
    labelScale  = False           # label spatial scale with scalebar (upper left of panel) (True or 'physical')
    labelSim    = False           # label simulation name (lower right corner) of panel
    labelHalo   = False           # label halo total mass and stellar mass
    labelCustom = False           # custom label string to include
    ctName      = None            # if not None (automatic based on field), specify colormap name
    plotSubhalos = False          # plot halfmass circles for the N most massive subhalos in this (sub)halo
    plotBHs     = False           # plot markers for the N most massive SMBHs in this (sub)halo
    relCoords   = True            # if plotting x,y,z coordinate labels, make them relative to box/halo center
    projType    = 'ortho'         # projection type, 'ortho', 'equirectangular', 'mollweide'
    projParams  = {}              # dictionary of parameters associated to this projection type
    rotation    = None            # 'face-on', 'edge-on', or None
    inclination = None            # inclination angle (degrees, about the x-axis) (0=unchanged)
    remapRatio  = None            # [x,y,z] periodic->cuboid remapping ratios, always None for single halos
    # fmt: on

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
        saveFileBase = "renderHaloFrame"  # filename base upon which frame numbers are appended

        # movie config
        minRedshift = 0.0  # ending redshift of frame sequence (we go forward in time)
        maxRedshift = 100.0  # starting redshift of frame sequence (we go forward in time)
        maxNumSnaps = None  # make at most this many evenly spaced frames, or None for all

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

    # load MPB properties for each panel, could be e.g. different runs (do not modify below)
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
        if "run" in p:
            v = p["variant"] if "variant" in p else None
            s = p["snap"] if "snap" in p else None
            r = p["res"] if "res" in p else None
            z = p["redshift"] if "redshift" in p and s is None else None  # skip if snap specified

            if "sP" in p:
                print("Warning: Overriding common sP with specified run,snap,redshift.")

            p["sP"] = simParams(run=p["run"], res=r, redshift=z, snap=s, hInd=p["hInd"], variant=v)

        if "subhaloInd" in p and p["sP"].subhaloInd is None:
            p["sP"] = p["sP"].copy()
            p["sP"].subhaloInd = p["subhaloInd"]

        # load MPB once per panel
        quants = [
            "SubfindID",
            "SnapNum",
            "Group_R_Crit200",
            "SubhaloPos",
            "SubhaloVel",
            "SubhaloHalfmassRad",
            "SubhaloHalfmassRadType",
        ]
        p["mpb"] = p["sP"].quantMPB(p["sP"].subhaloInd, quants=quants, add_ghosts=True, smooth=True)

        if not isinstance(p["nPixels"], list):
            p["nPixels"] = [p["nPixels"], p["nPixels"]]

    # determine frame sequence (as the last sP in panels is used somewhat at random, we are here
    # currently assuming that all runs in panels have the same snapshot configuration)
    snapNums = p["sP"].validSnapList(
        maxNum=plotConfig.maxNumSnaps, minRedshift=plotConfig.minRedshift, maxRedshift=plotConfig.maxRedshift
    )
    frameNum = 0

    for snapNum in snapNums:
        print("Frame [%d of %d] at snap %d:" % (frameNum, snapNums.size, snapNum))
        # finalize panels list (all properties not set here are invariant in time)
        for p in panels:
            # override simParams info at this snapshot
            p["sP"] = p["sP"].copy()
            p["sP"].setSnap(snapNum)

            # add imaging config for single halo view using MPB
            (
                p["boxSizeImg"],
                p["boxCenter"],
                p["extent"],
                p["haloVirRad"],
                p["galHalfMass"],
                p["galHalfMassStars"],
                p["rotMatrix"],
                p["rotCenter"],
            ) = haloImgSpecs(**p)

        # request render and save
        plotConfig.saveFilename = plotConfig.savePath + plotConfig.saveFileBase + "_%03d.png" % (frameNum)
        frameNum += 1

        if skipExisting and isfile(plotConfig.saveFilename):
            print("SKIP: %s" % plotConfig.saveFilename)
            continue

        renderMultiPanel(panels, plotConfig)


def selectHalosFromMassBin(sP, massBins, numPerBin, haloNum=None, massBinInd=None, selType="linear"):
    """Select subhalos IDs from a set of halo mass bins, using different sampling methods.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      massBins (list[tuple,2]): list of [min,max] 2-tuples of halo mass bins [log Msun].
      numPerBin (int): requested number of halos per bin.
      haloNum (int or None): an index haloNum which should iterate from 0 to the total number of halos requested
        across all bins, in which case each return is a single subhalo ID, as appropriate for a multi-quantity single
        system comparison figure. Specify either haloNum or massBinInd, not both.
      massBinInd (int or None): an index ranging from 0 to the number of bins, in which case all subhalo IDs in that
        bin are returned (limited to numPerBin), as appropriate for a multi-system single-quantity figure.
      selType (str): selection type within mass bin, one of "linear", "even", "random".
    """
    assert selType in ["linear", "even", "random"]

    gc = sP.groupCat(fieldsHalos=["Group_M_Crit200", "GroupFirstSub"])
    haloMasses = sP.units.codeMassToLogMsun(gc["halos"]["Group_M_Crit200"])

    # locate # of halos in mass bins (informational only)
    # for massBin in massBins:
    #    with np.errstate(invalid='ignore'):
    #        w = np.where((haloMasses >= massBin[0]) & (haloMasses < massBin[1]))[0]
    #    print('selectHalosFromMassBin(): In massBin [%.1f %.1f] have %d halos total.' % \
    #        (massBin[0],massBin[1],len(w)))

    # choose mass bin
    if haloNum is not None:
        assert massBinInd is None, "Specify either haloNum or massBinInd, not both."
        massBinInd = int(np.floor(float(haloNum) / numPerBin))

    massBin = massBins[massBinInd]

    with np.errstate(invalid="ignore"):
        wMassBinAll = np.where((haloMasses >= massBin[0]) & (haloMasses < massBin[1]))[0]

    # what algorithm to sub-select within mass bin
    if selType == "linear":
        wMassBin = wMassBinAll[0:numPerBin]
    if selType == "even":
        wMassBin = evenlySample(wMassBinAll, numPerBin)
    if selType == "random":
        np.random.seed(seed=424242 + sP.snap + sP.res + int(massBin[0] * 100) + int(massBin[1] * 100))
        num = np.clip(numPerBin, 1, wMassBinAll.size)
        wMassBin = sorted(np.random.choice(wMassBinAll, size=num, replace=False))

    if haloNum is not None:
        haloInd = haloNum - massBinInd * numPerBin

        # job past requested range, tell to skip
        if haloInd >= len(wMassBin):
            return None, None

        # single halo ID return
        shIDs = gc["GroupFirstSub"][wMassBin[haloInd]]

        # print('[%d] Render halo [%d] subhalo [%d] from massBin [%.1f %.1f] ind [%d of %d]...' % \
        #    (haloNum,wMassBin[haloInd],shIDs,massBin[0],massBin[1],haloInd,len(wMassBin)))
    else:
        # return full set in this mass bin
        shIDs = gc["GroupFirstSub"][wMassBin]

    return shIDs, massBinInd


def selectHalosFromMassBins(sP, massBins, numPerBin, selType="linear"):
    """Select one or more halo IDs from a set of halo mass bins, using different sampling methods.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      massBins (list[tuple,2]): list of [min,max] 2-tuples of halo mass bins [log Msun].
      numPerBin (int): requested number of halos per bin.
      selType (str): selection type within mass bin, one of "linear", "even", "random".
    """
    assert selType in ["linear", "even", "random"]

    gc = sP.groupCat(fieldsHalos=["Group_M_Crit200"])
    haloMasses = sP.units.codeMassToLogMsun(gc)

    inds = []

    for massBin in massBins:
        # locate all halos in bin
        with np.errstate(invalid="ignore"):
            wMassBinAll = np.where((haloMasses >= massBin[0]) & (haloMasses < massBin[1]))[0]

        print(
            "selectHalosFromMassBin(): In massBin [%.1f %.1f] have %d halos total."
            % (massBin[0], massBin[1], len(wMassBinAll))
        )

        if wMassBinAll.size == 0:
            inds.append([])
            continue

        # what algorithm to sub-select within mass bin
        if selType == "linear":
            wMassBin = wMassBinAll[0:numPerBin]
        if selType == "even":
            wMassBin = evenlySample(wMassBinAll, numPerBin)
        if selType == "random":
            np.random.seed(seed=424242 + sP.snap + sP.res + int(massBin[0] * 100) + int(massBin[1] * 100))
            num = np.clip(numPerBin, 1, wMassBinAll.size)
            wMassBin = sorted(np.random.choice(wMassBinAll, size=num, replace=False))

        inds.append(wMassBin)

    return inds
