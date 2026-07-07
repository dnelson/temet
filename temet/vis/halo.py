"""
Visualizations for individual halos/subhalos from ..cosmological runs.
"""

import gc
from copy import deepcopy
from getpass import getuser
from os.path import isfile

import numpy as np
from numba import set_num_threads
from scipy.interpolate import interp1d, make_interp_spline
from scipy.signal import savgol_filter

from ..util.helper import (
    evenlySample,
    hermite_interp,
    hermite_interp_2deriv,
    hermite_interp_3point,
    linear_interp,
    num_cpus,
    pSplit,
)
from ..util.match import match
from ..util.rotation import (
    meanAngMomVector,
    momentOfInertiaTensor,
    rotationMatricesFromInertiaTensor,
    rotationMatrixFromAngleDirection,
    rotationMatrixFromVec,
)
from ..util.simParams import simParams
from ..vis.common import easeQuant, renderMultiPanel
from ..vis.render import _grid_filename, defaultHsmlFac, getHsmlForPartType, gridBox


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
    assert sizeType in ["rVirial", "r500", "rHalfMass", "rHalfMassStars", "codeUnits", "kpc", "pc", "arcsec", "arcmin"]

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

        # time interpolation between snapshots?
        if "interpTime" in kwargs and ind[0] > 0:
            assert sP.snap >= mpb["SnapNum"].min()  # implement for interp before start of MPB

            mpb_times = sP.units.redshiftToAgeFlat(mpb["redshift"])
            inds = [ind[0], ind[0] - 1]  # mpb is ordered in increasing snapnum i.e. decreasing time
            assert inds[1] >= 0, "Avoid this case, do not interp on last snap in general."

            haloVirRad = np.interp(kwargs["interpTime"], mpb_times[inds], mpb["Group_R_Crit200"][inds])
            boxCenter = np.array(
                [np.interp(kwargs["interpTime"], mpb_times[inds], mpb["SubhaloPos"][inds, i]) for i in range(3)]
            )
            boxCenter = boxCenter[axes + [3 - axes[0] - axes[1]]]  # permute into axes ordering
            galHalfMassRad = np.interp(kwargs["interpTime"], mpb_times[inds], mpb["SubhaloHalfmassRad"][inds])
            galHalfMassRadStars = np.interp(
                kwargs["interpTime"], mpb_times[inds], mpb["SubhaloHalfmassRadType"][inds, sP.ptNum("stars")]
            )
            galVel = np.array(
                [np.interp(kwargs["interpTime"], mpb_times[inds], mpb["SubhaloVel"][inds, i]) for i in range(3)]
            )

        # set refPos and refVel, used e.g. for halo-centric quantities
        sP.refPos = boxCenter.copy()
        sP.refVel = galVel.copy()

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
        if s_type == "pc":
            s_img = sP.units.physicalKpcToCodeLength(s / 1000)
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
    boxCenter   = None          # [x,y,z] absolute coordinates of box center (overrides cenShift)
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
    plotStars   = False         # plot markers for individual massive stars
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
        saveFilename = "renderHalo_N%d%s%s.png" % (len(panels), _sim_str, _field_str)

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
            boxCenter_halo,
            p["extent"],
            p["haloVirRad"],
            p["galHalfMass"],
            p["galHalfMassStars"],
            haloRotMatrix,
            haloRotCenter,
        ) = haloImgSpecs(**p)

        if p["boxCenter"] is None:
            p["boxCenter"] = boxCenter_halo

        if p["rotMatrix"] is None:
            p["rotMatrix"], p["rotCenter"] = haloRotMatrix, haloRotCenter

    # check for aspect ratio mismatch
    raster_aspect = plotConfig.rasterPx[0] / plotConfig.rasterPx[1]
    panel0_aspect = panels[0]["nPixels"][0] / panels[0]["nPixels"][1]

    if len(panels) == 1 and abs(raster_aspect - panel0_aspect) > 1e-6:
        print(f"WARNING: Fixing raster aspect ratio [{raster_aspect:.2f}] mismatch with panel [{panel0_aspect:.2f}].")
        plotConfig.rasterPx[1] = int(plotConfig.rasterPx[0] / panel0_aspect)

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


def _smooth_mpb_pos(sP, mpb):
    """Additional smoothing on a MPB position, to avoid jitters and edge effects."""
    SubhaloPos = mpb["SubhaloPos"].copy()

    for i in range(3):
        sg_size = 20
        pos_1d = SubhaloPos[:, i].copy()  # time backwards
        # to avoid edge effects, linearly interpolate forward in time first
        mpb_times = sP.units.redshiftToAgeFlat(mpb["redshift"])[::-1]  # time forwards
        N_add = sg_size // 2 + 1
        dt = mpb_times[-1] - mpb_times[-2]
        mpb_times_new = mpb_times[-1] + np.linspace(dt, dt * N_add, num=N_add)
        f_pos = interp1d(mpb_times, pos_1d[::-1], kind="linear", fill_value="extrapolate")
        new_pos_1d = f_pos(mpb_times_new)
        # savgol smooth
        pos_1d_b = np.hstack((new_pos_1d[::-1], pos_1d))  # time backwards
        pos_1d_b_sm = savgol_filter(pos_1d_b, sg_size, 1)[N_add:]
        SubhaloPos[:, i] = pos_1d_b_sm

    return SubhaloPos


def renderSingleHaloFrames(
    panels_in, plotConfig=None, localVars=None, curTask=0, numTasks=1, skipExisting=True, getStats=False
):
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
    plotStars   = False         # plot markers for individual massive stars
    relCoords   = True            # if plotting x,y,z coordinate labels, make them relative to box/halo center
    projType    = 'ortho'         # projection type, 'ortho', 'equirectangular', 'mollweide'
    projParams  = {}              # dictionary of parameters associated to this projection type
    rotation    = None            # 'face-on', 'edge-on', or None
    inclination = None            # inclination angle (degrees, about the x-axis) (0=unchanged)
    remapRatio  = None            # [x,y,z] periodic->cuboid remapping ratios, always None for single halos
    numFramesPerRot = None        # if not None, then add a camera rotation that takes this many frames per full 360
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
        interpFac = None  # interpolate in time between snapshots, adding N frames per original snapshot
        interpDt = None  # interpolate in time between snapshots, adding frames with this constant time step (Myr)
        keyframeDt = None  # list of 2-tuples of [redshift,dt] to keyframe variable frame time spacing
        keyframeCamera = None  # list of 2-tuples of [redshift, camera_z] to keyframe camera configuration changes
        vmmEvoScalefac = None  # shift valMinMax by this factor times the scalefactor at each frame

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

    # preserve original valMinMax for later possible modification in movie frames
    for p in panels:
        if plotConfig.vmmEvoScalefac is not None and p["valMinMax"] is not None:
            p["valMinMaxOrig"] = p["valMinMax"].copy()

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
        if "mpb" not in p:
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
            p["mpb_unsmoothed"] = p["sP"].quantMPB(p["sP"].subhaloInd, quants=quants, add_ghosts=True)  # for make_rel

        # additional smoothing to remove short time-scale positional oscillations
        p["mpb"]["SubhaloPos"] = _smooth_mpb_pos(p["sP"], p["mpb"])

        if not isinstance(p["nPixels"], list):
            p["nPixels"] = [p["nPixels"], p["nPixels"]]

    # determine frame sequence (as the last sP in panels is used somewhat at random, we are here
    # currently assuming that all runs in panels have the same snapshot configuration)
    snapNums = p["sP"].validSnapList(
        maxNum=plotConfig.maxNumSnaps, minRedshift=plotConfig.minRedshift, maxRedshift=plotConfig.maxRedshift
    )
    frameNums = np.arange(snapNums.size)

    # time interpolation? (constant number of frames per snap)
    if plotConfig.interpFac is not None:
        # new frame number <-> snapshot number mapping
        frameNums = np.arange(0, (snapNums.size - 1) * plotConfig.interpFac + 1)

        print(f"Time [interpFac = {plotConfig.interpFac}] gives {frameNums.size} frames from {snapNums.size} snaps.")

        snapNums = np.repeat(snapNums, plotConfig.interpFac)[: -plotConfig.interpFac + 1]  # single frame on last snap

        # set interpolation times
        snapRedshift = p["sP"].snapNumToRedshift(all=True)
        snapAges = p["sP"].units.redshiftToAgeFlat(snapRedshift)  # Gyr

        frameTimesGyr = np.zeros(frameNums.size)
        frameRedshifts = np.zeros(frameNums.size)

        for i, frameNum in enumerate(frameNums):
            snapNum0 = snapNums[i]
            snapNum1 = snapNums[i + 1] if i < frameNums.size - 1 else snapNums[i]

            t0 = snapAges[snapNum0]
            t1 = snapAges[snapNum1]
            z0 = snapRedshift[snapNum0]
            z1 = snapRedshift[snapNum1]

            frameTimesGyr[i] = (t1 - t0) / plotConfig.interpFac * (frameNum % plotConfig.interpFac) + t0
            frameRedshifts[i] = (z1 - z0) / plotConfig.interpFac * (frameNum % plotConfig.interpFac) + z0

        curInterpSnap0 = None

    # time interpolation? (constant time step, independent of snapshot spacing)
    if plotConfig.interpDt is not None:
        # new frame number <-> snapshot number mapping
        snapRedshift = p["sP"].snapNumToRedshift(snapNums)  # already restricted to [minRedshift, maxRedshift]
        snapTimesMyr = p["sP"].units.redshiftToAgeFlat(snapRedshift) * 1000  # Myr

        # constant timestep
        frameTimesMyr = np.arange(snapTimesMyr[0], snapTimesMyr[-1], plotConfig.interpDt)

        # variable timestep as defined by a series of keyframes
        if plotConfig.keyframeDt is not None:
            # create interpolating function that defines dt as a function of redshift
            z_vals = [snapRedshift[0]] + [kf[0] for kf in plotConfig.keyframeDt] + [snapRedshift[-1]]
            t_vals = p["sP"].units.redshiftToAgeFlat(z_vals) * 1000  # Myr
            dt_vals = [plotConfig.interpDt] + [kf[1] for kf in plotConfig.keyframeDt] + [plotConfig.interpDt]

            f_dt = make_interp_spline(t_vals, dt_vals, k=1)  # hangs on k>1

            # step through
            frameTimesMyr = []
            cur_time = snapTimesMyr[0]

            while cur_time + f_dt(cur_time) < snapTimesMyr[-1]:
                # get the interpolation time step
                cur_dt = f_dt(cur_time)

                # add the time step to the list
                frameTimesMyr.append(cur_time)
                cur_time += cur_dt

            frameTimesMyr.append(snapTimesMyr[-1])
            frameTimesMyr = np.array(frameTimesMyr)

        frameNums = np.arange(frameTimesMyr.size)

        print(f"Time interp dt config gives [{frameNums.size}] frames from [{snapNums.size}] snaps.")

        # find closest snap preceeding each frame time
        snapNums = np.zeros(frameNums.size, dtype=np.int32) - 1
        for i, t in enumerate(frameTimesMyr):
            snapNums[i] = np.searchsorted(snapTimesMyr, t, side="right") - 1

        assert snapNums.min() >= 0 and snapNums.max() < snapTimesMyr.size, "interpDt out of snap time bounds!"

        curInterpSnap0 = None
        match_i0_snap = None

    # return frame sequence info if requested
    if getStats:
        return frameNums, snapNums

    # optionally parallelize over multiple tasks
    numFramesTot = frameNums.size
    frameCount = 0
    frameNums = pSplit(frameNums, numTasks, curTask)
    snapNums = pSplit(snapNums, numTasks, curTask)

    print(
        "Task [%d of %d] rendering [%d] frames of [%d] total (from %d to %d)..."
        % (curTask, numTasks, len(frameNums), numFramesTot, np.min(frameNums), np.max(frameNums))
    )

    # helper functions
    def _load_interp_data(panel_loc, snap_num):
        partType = panel_loc["partType"]
        sim = panel_loc["sP"].copy()
        sim.setSnap(snap_num)

        # load common fields, needed for all part types and vis fields
        ids = sim.snapshotSubsetP(partType, "id")
        pos = sim.snapshotSubsetP(partType, "pos")

        r = {"ids": ids, "pos": pos, "time": sim.tage, "redshift": sim.redshift}

        # load hsml?
        loadHsml = False

        if not sim.isPartType(partType, "stars") or sim.star not in [2, 3]:
            # hsml needed for both gas and dm (unless tetra method), and also stars (unless point-like)
            loadHsml = True

        if sim.isPartType(partType, "dm") and "tetra" in panel_loc["method"]:
            loadHsml = False

        if loadHsml:
            hsml = getHsmlForPartType(sim, partType)
            r["hsml"] = hsml

        # load velocities?
        loadVel = True

        if sim.isPartType(partType, "stars") and star_interp_method == "linear":
            loadVel = False

        if loadVel:
            r["vel"] = sim.snapshotSubsetP(partType, "vel_ckpch_gyr")  # units consistent with pos/Gyr

        # stars
        if sim.isPartType(partType, "stars"):
            r["sftime"] = sim.stars("sftime")
            r["metallicity"] = sim.stars("metallicity")
            r["initialmass"] = sim.stars("initialmass")
            if star_interp_method == "hermite_accel":
                r["acc"] = sim.stars("accel_ckpch_gyr2")  # units consistent with pos/Gyr^2

        # non-dm
        if not sim.isPartType(partType, "dm"):
            mass = sim.snapshotSubsetP(partType, "mass")
            r["mass"] = mass

        # gas
        if sim.isPartType(partType, "gas"):
            dens = sim.snapshotSubsetP(partType, "density")
            r["dens"] = dens

        # additional fields
        if panel_loc["ptRestrictions"] is not None:
            # mappings for 'requested field' to 'field that needs to be available in data cache to load'
            remaps = {"highres_massfrac": "HighResGasMass"}
            for field in panel_loc["ptRestrictions"]:
                fieldLoad = remaps[field] if field in remaps else field
                r[fieldLoad] = sim.snapshotSubsetP(partType, fieldLoad)

        # get subhalo frame of reference at this snapshot
        if make_rel:
            mpb_ind = np.where(p["mpb_unsmoothed"]["SnapNum"] == snap_num)[0]
            assert len(mpb_ind)

            r["sub_pos"] = p["mpb_unsmoothed"]["SubhaloPos"][mpb_ind[0], :]
            r["sub_vel"] = p["mpb_unsmoothed"]["SubhaloVel"][mpb_ind[0], :]
            r["sub_vel"] *= panel_loc["sP"].HubbleParam * panel_loc["sP"].units.kmS_in_kpcGyr / sim.scalefac
            r["sub_rel"] = False  # whether sub_pos and sub_vel are relative to subhalo

        return r

    def _cart_to_sph(pos, vel):
        r2_xy = pos[:, 0] ** 2 + pos[:, 1] ** 2
        r = np.sqrt(r2_xy + pos[:, 2] ** 2)
        theta = np.arccos(pos[:, 2] / r)
        phi = np.arctan2(pos[:, 1], pos[:, 0])

        # Suppress runtime warnings for division by zero at the origin/z-axis
        # with np.errstate(divide='ignore', invalid='ignore'):
        r_dot = (pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1] + pos[:, 2] * vel[:, 2]) / r
        theta_dot = ((pos[:, 0] * vel[:, 0] + pos[:, 1] * vel[:, 1]) * pos[:, 2] - r2_xy * vel[:, 2]) / (
            (r**2) * np.sqrt(r2_xy)
        )
        phi_dot = (pos[:, 0] * vel[:, 1] - pos[:, 1] * vel[:, 0]) / r2_xy

        pos_sph = np.column_stack((r, theta, phi))
        vel_sph = np.column_stack((r_dot, theta_dot, phi_dot))
        return pos_sph, vel_sph

    def _sph_to_cart(pos_sph):
        """Convert spherical coordinates back to cartesian coordinates."""
        r, theta, phi = pos_sph[:, 0], pos_sph[:, 1], pos_sph[:, 2]

        sin_t, cos_t = np.sin(theta), np.cos(theta)
        sin_p, cos_p = np.sin(phi), np.cos(phi)

        # Inverse Position
        x = r * sin_t * cos_p
        y = r * sin_t * sin_p
        z = r * cos_t

        pos = np.column_stack((x, y, z))
        return pos

    for snapNum, frameNum in zip(snapNums, frameNums):
        frameCount += 1
        print("Frame [#%d] [%d of %d] at snap %d:" % (frameNum, frameCount, snapNums.size, snapNum))

        # request render and save
        plotConfig.saveFilename = plotConfig.savePath + plotConfig.saveFileBase + "_%03d.png" % (frameNum)

        if skipExisting and isfile(plotConfig.saveFilename):
            print("SKIP: %s" % plotConfig.saveFilename)
            continue

        # set redshift for all panels (overrides simParams and e.g. zeros simParams.data, so must be before interp)
        for p in panels:
            p["sP"] = p["sP"].copy()
            p["sP"].setSnap(snapNum)

        # time interpolation?
        if plotConfig.interpFac is not None or plotConfig.interpDt is not None:
            # identify time (Gyr) and redshift for this frame
            if plotConfig.interpFac is not None:
                # time_interp = dt / plotConfig.interpFac * (frameNum % plotConfig.interpFac) + t0  # Gyr
                time_interp = frameTimesGyr[frameNum]
                time_interp_next = frameTimesGyr[frameNum + 1] if frameNum < numFramesTot - 1 else time_interp
                z_interp = frameRedshifts[frameNum]

            if plotConfig.interpDt is not None:
                time_interp = frameTimesMyr[frameNum] / 1000  # Gyr
                time_interp_next = frameTimesMyr[frameNum + 1] / 1000 if frameNum < numFramesTot - 1 else time_interp
                z_interp = p["sP"].units.ageFlatToRedshift(time_interp)

            p["interpTime"] = time_interp  # generates unique grid cache filename, triggers interp in vis overplots

            sP_next = p["sP"].copy()
            sP_next2 = p["sP"].copy()  # for stars 3-point interp
            if frameNum < numFramesTot - 1:
                sP_next.setSnap(sP_next.snap + 1)
                sP_next2.setSnap(sP_next.snap)
            else:
                # set to final snap
                sP_next.setSnap(sP_next.numSnaps - 1)
            if p["sP"].snap + 2 < p["sP"].numSnaps:
                sP_next2.setSnap(sP_next2.snap + 1)

            t0 = p["sP"].tage
            t1 = sP_next.tage
            t2 = sP_next2.tage
            dt = t1 - t0  # Gyr

            z0 = p["sP"].redshift
            z1 = sP_next.redshift
            dz = z1 - z0

            a0 = p["sP"].scalefac
            a1 = sP_next.scalefac

            if frameNum == numFramesTot - 1:
                time_interp = t1
                dt = time_interp - t0  # set tau = 1

            assert time_interp >= t0 - 1e-6 and time_interp <= t1, "interpDt frame time out of current snap interval."

            tau = (time_interp - t0) / dt  # [0,1]

            print(f" {t0 = :.4f}, {t1 = :.4f}, {time_interp = :.4f}, tau = {tau:.2f}, z = {z_interp:.2f}")

            p["interpSnapTimes"] = [t0, t1, a0, a1, dt, tau]  # used in vis.common() for overplotting interp

            # do stars in a relative frame of reference? 2-snap or 3-snap? in spherical coords? with accelerations?
            star_interp_method = "linear"  # hermite_2pt, hermite_3pt, hermite_accel
            star_3snaps = True
            stars_randomize_tau = False
            make_rel = False
            sph_coordinates = False

            # override sim.scalefac (e.g. physicalKpcToCodeLength)
            for p in panels:
                p["sP"].redshift = z_interp
                p["sP"].units.scalefac = 1 / (1 + z_interp)

        # keyframes?
        if plotConfig.keyframeCamera is not None:
            for panel in panels:
                assert panel["projType"] == "perspective"

                # current redshift
                cur_redshift = panel["sP"].redshift
                if plotConfig.interpFac is not None or plotConfig.interpDt is not None:
                    cur_redshift = z_interp

                # derive current size
                n_keyframes = len(plotConfig.keyframeCamera)
                assert n_keyframes >= 2, "Need at least 2 keyframes for keyframeCamera config."
                assert plotConfig.keyframeCamera[1][0] < plotConfig.keyframeCamera[0][0], "decreasing redshift order."

                # loop over keyframe pairs and find which one we are in
                for kf_i in range(n_keyframes - 1):
                    kf_z0, _ = plotConfig.keyframeCamera[kf_i]
                    kf_z1, _ = plotConfig.keyframeCamera[kf_i + 1]

                    if cur_redshift >= kf_z1:
                        break

                kf_z0, kf_size0 = plotConfig.keyframeCamera[kf_i]
                kf_z1, kf_size1 = plotConfig.keyframeCamera[kf_i + 1]
                # (kf_z0, kf_size0), (kf_z1, kf_size1) = plotConfig.keyframeCamera

                # current: float, start: float, end: float, target_start: float, target_end: float
                cur_size = easeQuant(cur_redshift, kf_z0, kf_z1, kf_size0, kf_size1)

                panel["size"] = cur_size

                print(f" Keyframe camera config at redshift {cur_redshift:.2f}, setting size to {cur_size:.2f}.")

                # set adaptive projection parameters
                panel["projParams"]["n"] = 10.0 * (cur_size / 2)  # effectively sets zoom?
                panel["projParams"]["f"] = 15.0 * (cur_size / 2)

        # finalize panels list (all properties not set here are invariant in time)
        allGridsExist = True

        for p in panels:
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

            if p["numFramesPerRot"] is not None:
                rotDirVec = [0.0, 1.0, 0.0]  # horizontal seeming spin

                p["rotCenter"] = p["boxCenter"]

                rotAngleDeg = 360.0 * (frameNum / p["numFramesPerRot"])
                locRotMatrix = rotationMatrixFromAngleDirection(rotAngleDeg, rotDirVec)

                # if rotMatrix already exists, multiply ours in
                if p["rotMatrix"] is not None:
                    p["rotMatrix"] = np.dot(locRotMatrix, p["rotMatrix"])
                else:
                    p["rotMatrix"] = locRotMatrix

            # does cached grid file already exist? then no need to load snapshot data for possible interp
            gridFilename = _grid_filename(**p)
            if not isfile(gridFilename):
                allGridsExist = False

            if 0 and make_rel:
                # also need the boxCenter of the next frame

                # for times within actual MPB, use smoothed properties directly
                ind = np.where(p["mpb"]["SnapNum"] == snapNums[frameCount])[0]
                assert len(ind)
                assert snapNums[frameCount] >= p["mpb"]["SnapNum"].min()  # implement for interp before start of MPB

                mpb_times = p["sP"].units.redshiftToAgeFlat(p["mpb"]["redshift"])
                inds = [ind[0], ind[0] - 1]  # mpb is ordered in increasing snapnum i.e. decreasing time
                assert inds[1] >= 0, "Avoid this case, do not interp on last snap in general."

                p["boxCenterNext"] = np.array(
                    [np.interp(time_interp_next, mpb_times[inds], p["mpb"]["SubhaloPos"][inds, i]) for i in range(3)]
                )
                p["boxCenterNext"] = p["boxCenterNext"][p["axes"] + [3 - p["axes"][0] - p["axes"][1]]]  # axes ordering

                # calculate derivative (velocity) of box motion
                frame_dt = time_interp_next - time_interp
                p["boxVelocity"] = (p["boxCenterNext"] - p["boxCenter"]) / frame_dt  # (ckpc/h)/Gyr

            # scale valMinMax (colorbar range) by redshift?
            if plotConfig.vmmEvoScalefac is not None and p["valMinMax"] is not None:
                adjust_fac = p["sP"].scalefac * plotConfig.vmmEvoScalefac * p["sP"].units.scalefac
                # print(f"Scaling valMinMax by factor {adjust_fac:.3f} at redshift {p['sP'].redshift:.2f}.")
                p["valMinMax"] = [v + adjust_fac for v in p["valMinMaxOrig"]]

        # time interpolation? load bracketing snapshots, cross-match, and interpolate positions and properties
        if plotConfig.interpFac is not None or plotConfig.interpDt is not None and not allGridsExist:
            assert len(panels) == 1  # otherwise generalize to multiple panels with potentially different pt/fields/runs
            p = panels[0]

            if curInterpSnap0 is None:
                # init: load data for first snapshot(s)
                next_interp_data = _load_interp_data(p, snapNum)

                # for stars, load a further next snapshot for 3 point interp
                if star_3snaps and p["sP"].isPartType(p["partType"], "stars") and p["sP"].snap + 2 <= p["sP"].numSnaps:
                    next2_interp_data = _load_interp_data(p, snapNum + 1)

            loadNextData = frameNum != numFramesTot - 1
            if p["sP"].isPartType(p["partType"], "stars"):
                loadNextData &= snapNum + 2 < p["sP"].numSnaps

            if curInterpSnap0 != snapNum:
                # we have moved to the next snapshot, move "next" data to "current" data, and load next snapshot
                cur_interp_data = next_interp_data

                # for stars, load a further next snapshot for 3 point interp
                if star_3snaps and p["sP"].isPartType(p["partType"], "stars"):
                    next_interp_data = next2_interp_data

                    if loadNextData:
                        next2_interp_data = _load_interp_data(p, snapNum + 2)
                else:
                    if loadNextData:
                        next_interp_data = _load_interp_data(p, snapNum + 1)

                # reset match indices for new snapshot pair(s)
                match_i0_snap, match_i0_global = None, None
                match_i1_snap, match_i1_global = None, None
                match_i2_snap, match_i2_global = None, None

                curInterpSnap0 = snapNum

            gc.collect()  # python will not automatically gc when reaching a MemoryError, only occasionally

            if isinstance(cur_interp_data, dict) and isinstance(cur_interp_data["ids"], dict):
                # no particles of this type (yet), dict['count'] == 0
                renderMultiPanel(panels, plotConfig)  # empty
                continue

            # cross-match by ids
            if match_i0_snap is None:
                # all particles will match? (DM) (simple logic)
                if p["sP"].isPartType(p["partType"], "dm"):
                    # cross-match
                    match_i0_snap, match_i1_snap = match(cur_interp_data["ids"], next_interp_data["ids"])

                    # re-shuffle 'first' snapshot data to be in same order
                    for key in cur_interp_data.keys():
                        # skip metadata
                        if key in ["time", "redshift", "sub_pos", "sub_vel", "sub_rel"]:
                            continue

                        cur_interp_data[key] = cur_interp_data[key][match_i0_snap]
                elif star_3snaps and p["sP"].isPartType(p["partType"], "stars"):
                    # stars, 3-snap interp
                    new_data_cur = {}
                    new_data_next = {}
                    new_data_next2 = {}
                    for key in ["time", "redshift", "sub_pos", "sub_vel", "sub_rel"]:
                        if key in cur_interp_data:
                            new_data_cur[key] = cur_interp_data[key]
                            new_data_next[key] = next_interp_data[key]
                            new_data_next2[key] = next2_interp_data[key]

                    # define unique (total) id list, and then cross-match each snap to this master list
                    ids_unique = np.hstack([cur_interp_data["ids"], next_interp_data["ids"], next2_interp_data["ids"]])
                    ids_unique = np.unique(ids_unique)

                    match_i0_snap, match_i0_global = match(cur_interp_data["ids"], ids_unique)
                    match_i1_snap, match_i1_global = match(next_interp_data["ids"], ids_unique)
                    match_i2_snap, match_i2_global = match(next2_interp_data["ids"], ids_unique)

                    # statistics
                    n0 = cur_interp_data["ids"].size
                    n1 = next_interp_data["ids"].size
                    n2 = next2_interp_data["ids"].size
                    n_match0 = match_i0_snap.size
                    n_match1 = match_i1_snap.size
                    n_match2 = match_i2_snap.size
                    n_unique = ids_unique.size

                    # print(f"  Tot: {n0} (cur), {n1} (next), {n2} (next2) snap.")
                    # print(f"  Unique: {n_unique} total unique particles across the three snapshots.")

                    # locate un-matched
                    mask_cur = np.zeros(n_unique, dtype=bool)
                    mask_next = np.zeros(n_unique, dtype=bool)
                    mask_next2 = np.zeros(n_unique, dtype=bool)
                    mask_cur[match_i0_global] = True
                    mask_next[match_i1_global] = True
                    mask_next2[match_i2_global] = True
                    n_unmatched_cur = np.count_nonzero(~mask_cur)
                    n_unmatched_next = np.count_nonzero(~mask_next)
                    n_unmatched_next2 = np.count_nonzero(~mask_next2)
                    # print(f"  Unmatched: {n_unmatched_cur} (cur) snap to global.")
                    # print(f"  Unmatched: {n_unmatched_next} (next) snap to global.")
                    # print(f"  Unmatched: {n_unmatched_next2} (next2) snap to global.")

                    w_disappearing_cur = np.where(mask_cur & ~mask_next)[0]
                    w_appearing_next = np.where(mask_next & ~mask_cur)[0]
                    w_disappearing_next = np.where(mask_next & ~mask_next2)[0]
                    w_appearing_next2 = np.where(mask_next2 & ~mask_next)[0]
                    # print(f"  Disappearing: {w_disappearing_cur.size} (cur), {w_disappearing_next.size} (next) snap.")
                    # print(f"  Appearing: {w_appearing_next.size} (next), {w_appearing_next2.size} (next2) snap.")

                    for key in cur_interp_data:
                        if key in ["time", "redshift", "sub_pos", "sub_vel", "sub_rel"]:
                            continue

                        # create empty array of full size [n_unique], ordered as ids_unique
                        shape = (n_unique, 3) if cur_interp_data[key].ndim == 2 else (n_unique,)
                        new_data_cur[key] = np.zeros(shape, dtype=cur_interp_data[key].dtype)
                        new_data_next[key] = np.zeros(shape, dtype=next_interp_data[key].dtype)
                        new_data_next2[key] = np.zeros(shape, dtype=next2_interp_data[key].dtype)

                        # stamp matches for current snap
                        new_data_cur[key][match_i0_global] = cur_interp_data[key][match_i0_snap]
                        new_data_next[key][match_i1_global] = next_interp_data[key][match_i1_snap]
                        new_data_next2[key][match_i2_global] = next2_interp_data[key][match_i2_snap]

                        # if key in ["mass", "initialmass"]:
                        #    # leave all masses at zero for unmatched subset (too many visual artifacts) (NO)
                        #    continue

                        # current: fill in missing values with constant values from the later snaps
                        inds_from_next = np.where((new_data_cur[key] == 0) & (new_data_next[key] != 0))
                        new_data_cur[key][inds_from_next] = new_data_next[key][inds_from_next]
                        inds_from_next2 = np.where((new_data_cur[key] == 0) & (new_data_next2[key] != 0))
                        new_data_cur[key][inds_from_next2] = new_data_next2[key][inds_from_next2]

                        assert new_data_cur[key].min() != 0

                        # next: fill in missing values
                        inds_from_next2 = np.where((new_data_next[key] == 0) & (new_data_next2[key] != 0))
                        new_data_next[key][inds_from_next2] = new_data_next2[key][inds_from_next2]
                        inds_from_cur = np.where((new_data_next[key] == 0) & (new_data_cur[key] != 0))
                        new_data_next[key][inds_from_cur] = new_data_cur[key][inds_from_cur]

                        assert new_data_next[key].min() != 0

                        # next2: fill in missing values
                        inds_from_next = np.where((new_data_next2[key] == 0) & (new_data_next[key] != 0))
                        new_data_next2[key][inds_from_next] = new_data_next[key][inds_from_next]
                        inds_from_cur = np.where((new_data_next2[key] == 0) & (new_data_cur[key] != 0))
                        new_data_next2[key][inds_from_cur] = new_data_cur[key][inds_from_cur]

                        assert new_data_next2[key].min() != 0

                    # set leftmost pos for unmatched particles using extrapolation from subsequent snapshot interval
                    if w_appearing_next.size > 0:
                        tau_extrap = (time_interp - t1) / (t2 - t1)  # note: < 0

                        new_pos = np.zeros((w_appearing_next.size, 3), dtype="float32")
                        for i in range(3):
                            pos1 = new_data_next["pos"][w_appearing_next, i]
                            pos2 = new_data_next2["pos"][w_appearing_next, i]
                            new_data_cur["pos"][w_appearing_next, i] = linear_interp(pos1, pos2, tau_extrap)

                    # set data arrays to our new 'full' matching arrays
                    cur_interp_data = new_data_cur
                    next_interp_data = new_data_next
                    next2_interp_data = new_data_next2

                else:
                    # gas, stars (old method)

                    # cross-match gas, (de)refinement means both appearences and disappearances
                    match_i0_snap, match_i1_snap = match(cur_interp_data["ids"], next_interp_data["ids"])

                    # create new arrays
                    new_data_cur = {}
                    new_data_next = {}
                    for key in ["time", "redshift", "sub_pos", "sub_vel", "sub_rel"]:
                        if key in cur_interp_data:
                            new_data_cur[key] = cur_interp_data[key]
                            new_data_next[key] = next_interp_data[key]

                    # total unique
                    n_match = match_i0_snap.size
                    n_unique = np.unique(np.concatenate([cur_interp_data["ids"], next_interp_data["ids"]])).size
                    # print(f"  Tot: {cur_interp_data['ids'].size} (cur), {next_interp_data['ids'].size} (next) snap.")
                    # print(f"  Unique: {n_unique} total unique particles across the two snapshots.")

                    # locate un-matched
                    mask_cur = np.zeros(cur_interp_data["ids"].size, dtype=bool)
                    mask_next = np.zeros(next_interp_data["ids"].size, dtype=bool)
                    mask_cur[match_i0_snap] = True
                    mask_next[match_i1_snap] = True
                    w_unmatched_cur = np.where(~mask_cur)[0]
                    w_unmatched_next = np.where(~mask_next)[0]
                    # print(f"  Unmatched: {w_unmatched_cur.size} (cur), {w_unmatched_next.size} (next) snap.")

                    # print(f"WARNING: {w_unmatched_cur.size} unmatched stars in current snap {snapNum} disappear.")

                    # need to preserve mass and initial mass of unmatched stars and gas for logic below
                    if p["sP"].isPartType(p["partType"], "stars"):
                        next_unmatched_data = {}
                        next_unmatched_data["mass"] = next_interp_data["mass"][w_unmatched_next]
                        next_unmatched_data["initialmass"] = next_interp_data["initialmass"][w_unmatched_next]

                    if p["sP"].isPartType(p["partType"], "gas"):
                        cur_unmatched_mass = cur_interp_data["mass"][w_unmatched_cur]
                        next_unmatched_mass = next_interp_data["mass"][w_unmatched_next]

                    for key in cur_interp_data:
                        if key in ["time", "redshift", "sub_pos", "sub_vel", "sub_rel"]:
                            continue

                        # create empty array of full size, structured as:
                        # [0:n_matched] matched particles in same order
                        # [n_matched:n_matched + w_unmatched_cur.size] current particles with no match in next
                        # [n_matched + w_unmatched_cur.size:] next particles with no match in current
                        shape = (n_unique, 3) if cur_interp_data[key].ndim == 2 else (n_unique,)
                        new_data_cur[key] = np.zeros(shape, dtype=cur_interp_data[key].dtype)
                        new_data_next[key] = np.zeros(shape, dtype=next_interp_data[key].dtype)

                        # stamp matches (beginning of arrays)
                        new_data_cur[key][0:n_match] = cur_interp_data[key][match_i0_snap]
                        new_data_next[key][0:n_match] = next_interp_data[key][match_i1_snap]

                        if key in ["mass", "initialmass"]:
                            # leave all masses at zero for unmatched subset (too many visual artifacts)
                            continue

                        # copy values for missing matches (unchanged for now, modified below)
                        new_data_cur[key][n_match : n_match + w_unmatched_cur.size] = cur_interp_data[key][
                            w_unmatched_cur
                        ]
                        new_data_cur[key][n_match + w_unmatched_cur.size :] = next_interp_data[key][w_unmatched_next]

                        new_data_next[key][n_match : n_match + w_unmatched_cur.size] = cur_interp_data[key][
                            w_unmatched_cur
                        ]
                        new_data_next[key][n_match + w_unmatched_cur.size :] = next_interp_data[key][w_unmatched_next]

                    # set next_interp_data['pos'] for unmatched particles using constant vel assumption
                    # leads to 'pulsed collapse' visual artifacts for star clusters? lin interp of ~radial orbits?
                    #  - would be better extrapolating hermite from an adjacent snapshot pair?
                    dpos_cur = cur_interp_data["vel"][w_unmatched_cur] * dt
                    new_data_next["pos"][n_match : n_match + w_unmatched_cur.size] += dpos_cur  # ckpc/h

                    dpos_next = next_interp_data["vel"][w_unmatched_next] * dt
                    new_data_cur["pos"][n_match + w_unmatched_cur.size :] -= dpos_next  # ckpc/h

                    # set mass to zero for unmatched particles, on other side of the interval, so they fade away/in (no)
                    # new_data_next["mass"][n_match : n_match + w_unmatched_cur.size] = 1e-10
                    # new_data_cur["mass"][n_match + w_unmatched_cur.size :] = 1e-10

                    # set data arrays to our new 'full' matching arrays
                    cur_interp_data = new_data_cur
                    next_interp_data = new_data_next

                assert np.array_equal(cur_interp_data["ids"], next_interp_data["ids"]), "ID mismatch after sorting!"

            # at this particular interpolation time, use actual birth times of stars, leading to instantaneous
            # appearence (by setting non-zero mass and initial mass) after formation
            if (not star_3snaps) and p["sP"].isPartType(p["partType"], "stars"):
                if w_unmatched_cur.size > 0:
                    cur_interp_data["initialmass"][n_match : n_match + w_unmatched_cur.size] = 1e-20
                    next_interp_data["initialmass"][n_match : n_match + w_unmatched_cur.size] = 1e-20

                if w_unmatched_next.size > 0:
                    # first, set vanishing small initial mass (zero causes error in sps for mags)
                    cur_interp_data["initialmass"][n_match + w_unmatched_cur.size :] = 1e-20
                    next_interp_data["initialmass"][n_match + w_unmatched_cur.size :] = 1e-20

                    # convert scale factor to age of the universe in Gyr
                    sftime_next = next_interp_data["sftime"][n_match + w_unmatched_cur.size :]
                    sfz_next = 1 / sftime_next - 1

                    sfage_next = np.atleast_1d(p["sP"].units.redshiftToAgeFlat(sfz_next))

                    # which stars have already formed by this interpolation time?
                    w_formed = np.where(sfage_next <= time_interp)

                    for key in ["mass", "initialmass"]:
                        mass_val = next_unmatched_data[key][w_formed]
                        cur_interp_data[key][n_match + w_unmatched_cur.size :][w_formed] = mass_val
                        next_interp_data[key][n_match + w_unmatched_cur.size :][w_formed] = mass_val

                    # print(f"  Stars: {w_formed[0].size} formed (of {w_unmatched_next.size}), appear at interp time.")

            if star_3snaps and p["sP"].isPartType(p["partType"], "stars"):
                if w_disappearing_cur.size > 0:
                    # print(f"WARNING: {w_disappearing_cur.size} stars in current snap {snapNum} disappear.")
                    cur_interp_data["initialmass"][w_disappearing_cur] = 1e-20
                    next_interp_data["initialmass"][w_disappearing_cur] = 1e-20
                    next2_interp_data["initialmass"][w_disappearing_cur] = 1e-20  # unused

                if w_disappearing_next.size > 0:
                    # print(f"WARNING: {w_disappearing_next.size} stars in next snap {snapNum + 1} disappear.")
                    cur_interp_data["initialmass"][w_disappearing_next] = 1e-20
                    next_interp_data["initialmass"][w_disappearing_next] = 1e-20
                    next2_interp_data["initialmass"][w_disappearing_next] = 1e-20  # unused

                if w_appearing_next.size > 0:
                    # convert scale factor to age of the universe in Gyr
                    sftime_next = next_interp_data["sftime"][w_appearing_next]
                    sfz_next = 1 / sftime_next - 1
                    sfage_next = np.atleast_1d(p["sP"].units.redshiftToAgeFlat(sfz_next))

                    # which stars have -not yet- formed by this interpolation time?
                    w_unformed = np.where(sfage_next > time_interp)

                    for key in ["mass", "initialmass"]:
                        cur_interp_data[key][w_appearing_next][w_unformed] = 1e-20
                        next_interp_data[key][w_appearing_next][w_unformed] = 1e-20

                    # note: we don't care about w_appearing_next2 since we only render from cur to next, and then shift
                    # print(f"  Stars: {w_unformed[0].size} unformed (of {w_appearing_next.size}) at interp time.")

                # although the following negates some of the logic above, leads to the best visual result, with
                # neither streak-ins due to bulk SubhaloVel through the box, nor blue implosions of star clusters
                for key in ["mass", "initialmass"]:
                    cur_interp_data[key][w_appearing_next] = 1e-20
                    cur_interp_data[key][w_appearing_next2] = 1e-20
                    cur_interp_data[key][w_disappearing_cur] = 1e-20
                    cur_interp_data[key][w_disappearing_next] = 1e-20

                    next_interp_data[key][w_appearing_next] = 1e-20
                    next_interp_data[key][w_appearing_next2] = 1e-20
                    next_interp_data[key][w_disappearing_cur] = 1e-20
                    next_interp_data[key][w_disappearing_next] = 1e-20

            if p["sP"].isPartType(p["partType"], "stars"):
                assert cur_interp_data["initialmass"].min() > 0, "Should not occur."
                assert next_interp_data["initialmass"].min() > 0, "Should not occur."
                if star_3snaps:
                    assert next2_interp_data["initialmass"].min() > 0, "Should not occur."

            # for gas, randomly assign refinement (next cells with no match in current) and derefinement
            # (current cells with no match in next) times within the interval, and set mass to zero on the
            # other side of these times, so that they instantaneously appear across the interval,
            # rather than fading in/out, which avoids pulsational visual artifacts and is closer to the
            # actual numerical process of cell (de)refinement
            if p["sP"].isPartType(p["partType"], "gas"):
                # new random number generator with seed
                rng = np.random.default_rng(seed=frameNum + n_match)

                gas_tform_next = rng.uniform(low=t0, high=t1, size=w_unmatched_next.size)
                gas_tderef_cur = rng.uniform(low=t0, high=t1, size=w_unmatched_cur.size)

                # which gas cells (that are disappearing/being derefined) are still present?
                w_cur = np.where(gas_tderef_cur > time_interp)

                mass_val = cur_unmatched_mass[w_cur]
                cur_interp_data["mass"][n_match : n_match + w_unmatched_cur.size] = 0.0
                next_interp_data["mass"][n_match : n_match + w_unmatched_cur.size] = 0.0
                cur_interp_data["mass"][n_match : n_match + w_unmatched_cur.size][w_cur] = mass_val
                next_interp_data["mass"][n_match : n_match + w_unmatched_cur.size][w_cur] = mass_val

                # print(f"  Gas: {w_cur[0].size} derefining (of {w_unmatched_cur.size}), still visible at interp time.")

                # which gas cells (that are appearing due to refinement) are already present?
                w_next = np.where(gas_tform_next < time_interp)

                mass_val = next_unmatched_mass[w_next]
                cur_interp_data["mass"][n_match + w_unmatched_cur.size :] = 0.0
                next_interp_data["mass"][n_match + w_unmatched_cur.size :] = 0.0
                cur_interp_data["mass"][n_match + w_unmatched_cur.size :][w_next] = mass_val
                next_interp_data["mass"][n_match + w_unmatched_cur.size :][w_next] = mass_val

                # print(f"  Gas: {w_next[0].size} refining (of {w_unmatched_next.size}), now visible at interp time.")

            # cubic hermite interpolation in time
            pos0 = cur_interp_data["pos"]
            pos1 = next_interp_data["pos"]
            if "vel" in cur_interp_data:
                vel0 = cur_interp_data["vel"]
                vel1 = next_interp_data["vel"]

            nThreads = np.clip(num_cpus() // 2, 1, 36)  # determine threading automatically
            set_num_threads(nThreads)

            # frame of reference? make copies not views! (potentially reused for next frame)
            if make_rel:
                # pos0 = pos0 - p["boxCenter"]
                # vel0 = vel0 - p["boxVelocity"]
                # pos1 = pos1 - p["boxCenter"]  # NEXT??
                # vel1 = vel1 - p["boxVelocity"]  # assume constant box velocity across interval

                pos0 = pos0 - cur_interp_data["sub_pos"]  # shift to subhalo frame of reference
                vel0 = vel0 - cur_interp_data["sub_vel"]
                if "vel" in cur_interp_data:
                    pos1 = pos1 - next_interp_data["sub_pos"]  # NEXT??
                    vel1 = vel1 - next_interp_data["sub_vel"]  # assume constant box velocity across interval

            if sph_coordinates:
                assert make_rel

                # cartesian -> spherical coordinates
                pos0_sph, vel0_sph = _cart_to_sph(pos0, vel0)
                pos1_sph, vel1_sph = _cart_to_sph(pos1, vel1)

                # handle periodic wrapping
                dist_theta = pos1_sph[:, 1] - pos0_sph[:, 1]

                w = np.where(dist_theta > np.pi / 2)[0]
                pos1_sph[w, 1] -= np.pi
                w = np.where(dist_theta < -np.pi / 2)[0]
                pos1_sph[w, 1] += np.pi

                pos0_sph[:, 2] += np.pi
                pos1_sph[:, 2] += np.pi
                dist_phi = pos1_sph[:, 2] - pos0_sph[:, 2]
                w = np.where(dist_phi > np.pi)[0]
                pos1_sph[w, 2] -= 2 * np.pi
                w = np.where(dist_phi < -np.pi)[0]
                pos1_sph[w, 2] += 2 * np.pi
                pos0_sph[:, 2] -= np.pi
                pos1_sph[:, 2] -= np.pi

                pos0 = pos0_sph
                vel0 = vel0_sph
                pos1 = pos1_sph
                vel1 = vel1_sph

            # test:
            if 0 and sph_coordinates:
                # interpolate 1d radius
                r0 = np.linalg.norm(pos0, axis=1)
                r1 = np.linalg.norm(pos1, axis=1)

                r0 = np.clip(r0, 1e-10, None)  # eps
                r1 = np.clip(r1, 1e-10, None)

                dr0_dt = np.sum(pos0 * vel0, axis=1) / r0
                dr1_dt = np.sum(pos1 * vel1, axis=1) / r1

                rad_t_sph = hermite_interp(r0, r1, dr0_dt, dr1_dt, tau, dt)

                # 'direction vector interpolation' i.e. hermite on unit vectors
                dir0 = pos0 / r0[:, np.newaxis]
                dir1 = pos1 / r1[:, np.newaxis]

                # calculate angular velocity vectors (omega = r x v / r^2)
                omega0 = np.cross(pos0, vel0) / r0[:, np.newaxis] ** 2
                omega1 = np.cross(pos1, vel1) / r1[:, np.newaxis] ** 2

                # calculate direction derivatives (ddir/dt = omega x dir)
                ddir0_dt = np.cross(omega0, dir0)
                ddir1_dt = np.cross(omega1, dir1)

                # interpolate the unit direction vectors
                dir_t = hermite_interp(dir0, dir1, ddir0_dt, ddir1_dt, tau, dt)

                # re-normalize direction vectors
                dir_t /= np.linalg.norm(dir_t, axis=1)[:, np.newaxis]

                pos_t_sph = dir_t * rad_t_sph[:, np.newaxis]

                # spherical -> cartesian coordinates
                pos_t = _sph_to_cart(pos_t_sph)

            if p["sP"].isPartType(p["partType"], "stars"):
                # stars: add noise to tau to reduce correlated radial oscillation visual artifact
                if stars_randomize_tau:
                    rng = np.random.default_rng(seed=frameNum + pos0.size + pos1.size)

                    tau_indiv = tau + rng.uniform(low=-0.02, high=0.02, size=pos0.shape[0])
                    assert star_interp_method == "linear", "Need to change hermite interp funcs to vartau versions."

                # different interpolation methods for stars
                if star_interp_method == "hermite_2pt":
                    # 2-points with first derivatives (velocities) (cubic)
                    pos_t = hermite_interp(pos0, pos1, vel0, vel1, tau, dt)
                elif star_interp_method == "linear":
                    # linear in each of x,y,z
                    pos_t = np.zeros_like(pos0)
                    for i in range(3):
                        pos_t[:, i] = linear_interp(pos0[:, i], pos1[:, i], tau)
                        # pos_t[:, i] = linear_interp_vartau(pos0[:, i], pos1[:, i], tau_indiv)
                elif star_interp_method == "hermite_accel":
                    # 2-points also with second derivatives (accelerations) (quintic)
                    acc0 = cur_interp_data["acc"] * p["sP"].scalefac ** 4
                    acc1 = next_interp_data["acc"] * p["sP"].scalefac ** 4
                    pos_t = hermite_interp_2deriv(pos0, pos1, vel0, vel1, acc0, acc1, tau, dt)
                elif star_interp_method == "hermite_3pt":
                    # 3-point with first derivatives (quintic)
                    assert star_3snaps
                    pos2 = next2_interp_data["pos"]
                    vel2 = next2_interp_data["vel"]
                    pos_t = hermite_interp_3point(pos0, pos1, pos2, vel0, vel1, vel2, time_interp, [t0, t1, t2])
            else:
                # normal 2-pt hermite for gas and dm
                pos_t = hermite_interp(pos0, pos1, vel0, vel1, tau, dt)

            if sph_coordinates:
                # spherical -> cartesian coordinates
                pos_t = _sph_to_cart(pos_t)
                assert np.count_nonzero(np.isnan(pos_t)) == 0, "NaN in interpolated positions!"

            if make_rel:
                # undo frame of reference
                # pos_t += p["boxCenter"]
                pos_t += cur_interp_data["sub_pos"]

            # save interpolated positions into dataCache (intercepted in snapshot loads during vis)
            cache_key_prefix = "snap%d_%s_" % (snapNum, p["partType"])
            p["sP"].data[cache_key_prefix + "Coordinates"] = pos_t

            # linear interp other properties, and override data in panel with interpolated data
            if not p["sP"].isPartType(p["partType"], "dm"):
                # only for DM do we skip mass, which is a (constant) scalar
                mass_t = linear_interp(cur_interp_data["mass"], next_interp_data["mass"], tau)
                p["sP"].data[cache_key_prefix + "Masses"] = mass_t

            if p["sP"].isPartType(p["partType"], "gas"):
                dens_t = linear_interp(cur_interp_data["dens"], next_interp_data["dens"], tau)
                p["sP"].data[cache_key_prefix + "Density"] = dens_t

            if p["sP"].isPartType(p["partType"], "dm") and "tetra" in p["method"]:
                # for phase-space tetrahedral rendering of DM, need also the shuffled ids
                p["sP"].data[cache_key_prefix + "IDs"] = cur_interp_data["ids"]

            # any additional fields for linear interp, if present
            extraFields = ["hsml", "HighResGasMass"]
            for field in extraFields:
                if field in cur_interp_data:
                    hr_gas_mass_t = linear_interp(cur_interp_data[field], next_interp_data[field], tau)
                    p["sP"].data[cache_key_prefix + field] = hr_gas_mass_t

            # three fields needed for stellar sps (light) are all constant after birth
            if p["sP"].isPartType(p["partType"], "stars"):
                p["sP"].data[cache_key_prefix + "StellarFormationTime"] = cur_interp_data["sftime"]  # constant
                p["sP"].data[cache_key_prefix + "Metallicity"] = cur_interp_data["metallicity"]
                p["sP"].data[cache_key_prefix + "InitialMass"] = cur_interp_data["initialmass"]

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
