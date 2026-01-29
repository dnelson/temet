"""
Render specific halo movie/time-series visualizations.
"""

from os.path import isfile

import h5py
import numpy as np

from ..util.rotation import rotationMatrixFromAngleDirection
from ..util.simParams import simParams
from ..vis.common import savePathBase, savePathDefault
from ..vis.halo import renderSingleHalo, renderSingleHaloFrames


def tngCluster_center_timeSeriesPanels(conf=0):
    """Plot a time series of panels from subsequent snapshots in the center of fof0."""
    panels = []

    zStart = 0.3  # start plotting at this snapshot
    nSnapsBack = 12  # one panel per snapshot, back in time

    run = "illustris"
    res = 1820
    rVirFracs = None  # [0.05]
    method = "sphMap"
    nPixels = [960, 960]
    size = 100.0
    sizeType = "codeUnits"
    labelZ = True
    axes = [1, 0]
    rotation = None

    if conf == 0:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [6.5, 9.0]
    if conf == 1:
        partType = "stars"
        partField = "coldens_msunkpc2"
        valMinMax = [6.5, 10.0]
    if conf == 2:
        partType = "gas"
        partField = "metal_solar"
        valMinMax = [-0.5, 0.5]
    if conf == 3:
        partType = "dm"
        partField = "coldens2_msunkpc2"
        valMinMax = [15.0, 16.0]
    if conf == 4:
        partType = "gas"
        partField = "pressure_ratio"
        valMinMax = [-2.0, 1.0]

    # configure panels
    sP = simParams(res=res, run=run, redshift=zStart)
    for i in range(nSnapsBack):
        haloID_loc = 0
        if run == "tng" and i < 2:
            haloID_loc = 1

        halo = sP.groupCatSingle(haloID=haloID_loc)
        print(sP.snap, sP.redshift, haloID_loc, halo["GroupFirstSub"], halo["GroupPos"])

        panels.append({"subhaloInd": halo["GroupFirstSub"], "redshift": sP.redshift})
        sP.setSnap(sP.snap - 1)

    panels[0]["labelScale"] = True
    panels[-1]["labelHalo"] = True

    class plotConfig:
        plotStyle = "edged_black"
        colorbars = True
        rasterPx = 960
        saveFilename = savePathDefault + "timePanels_%s_shID-0_%s-%s_z%.1f_n%d.pdf" % (
            sP.simName,
            partType,
            partField,
            zStart,
            nSnapsBack,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def loopTimeSeries():
    """Helper: Loop the above over configs."""
    for i in range(5):
        tngCluster_center_timeSeriesPanels(conf=i)


def zoomEvoMovies(conf):
    """Configurations to render movies of the sims.zooms2 runs (at ~400 total snapshots)."""
    panels = []

    if conf == "oneRes_DensTempEntr":
        panels.append({"res": 11, "partField": "coldens", "valMinMax": [19.0, 23.0], "labelScale": True})
        panels.append({"res": 11, "partField": "temp", "valMinMax": [4.0, 6.5]})
        panels.append({"res": 11, "partField": "entr", "valMinMax": [6.0, 9.0], "labelHalo": True})

    if conf == "threeRes_DensTemp":
        panels.append({"res": 9, "partField": "coldens", "valMinMax": [19.0, 23.0]})
        panels.append({"res": 10, "partField": "coldens", "valMinMax": [19.0, 23.0]})
        panels.append({"res": 11, "partField": "coldens", "valMinMax": [19.0, 23.0]})
        panels.append({"res": 9, "partField": "temp", "valMinMax": [4.0, 6.5]})
        panels.append({"res": 10, "partField": "temp", "valMinMax": [4.0, 6.5]})
        panels.append({"res": 11, "partField": "temp", "valMinMax": [4.0, 6.5]})

    hInd = 2
    subhaloInd = 0
    run = "zooms2"
    partType = "gas"
    rVirFracs = [0.15, 0.5, 1.0]
    method = "sphMap"
    nPixels = [1920, 1920]
    size = 3.5
    sizeType = "rVirial"
    axes = [1, 0]
    labelSim = False
    relCoords = True
    rotation = None

    class plotConfig:
        plotStyle = "open"
        rasterPx = 1200
        colorbars = True
        saveFileBase = "%s_evo_h%d_%s" % (run, hInd, conf)

        # movie config
        minRedshift = 2.0
        maxRedshift = 100.0

    renderSingleHaloFrames(panels, plotConfig, locals())


def singleEvoFrames_3x2(frame=0, subhaloID=402572, justStars=False):
    """Plot frames for a time-evolution movie (using merger tree) of a single halo/galaxy.

    3x2 panels: stars, DM, gas, galaxy-scale and halo-scale. Or 2x1 panels: just one component.
    """
    panels = []

    zStart = 0.0

    run = "illustris"
    res = 1820
    rVirFracs = [0.5]
    method = "sphMap"
    nPixels = [800, 800]
    axes = [1, 0]
    rotation = None
    relCoords = True

    # load MPB from z=0 and get subhalo ID at this snapshot
    sP = simParams(res=res, run=run, redshift=zStart)

    tree_mpb = sP.loadMPB(subhaloID)
    assert frame < len(tree_mpb["SnapNum"])

    subhaloInd = tree_mpb["SubfindID"][frame]
    redshift = sP.snapNumToRedshift(tree_mpb["SnapNum"][frame])
    print(
        "[%d of %d] render subhaloInd = %d at snap = %d (z = %.3f)"
        % (frame, len(tree_mpb["SnapNum"]), subhaloInd, tree_mpb["SnapNum"][frame], redshift)
    )

    # galaxy-scale
    gal_size = 0.2
    halo_size = 1.5

    sizeType = "rVirial"

    if not justStars:
        panels.append({"size": gal_size, "partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.5, 9.0]})
        panels.append({"size": gal_size, "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]})
    panels.append({"size": gal_size, "partType": "stars", "partField": "stellarComp"})

    # halo-scale
    if not justStars:
        panels.append({"size": halo_size, "partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.5, 9.0]})
        panels.append({"size": halo_size, "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.5, 8.0]})
    panels.append({"size": halo_size, "partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [4.5, 8.0]})

    if justStars:
        panels[0]["labelScale"] = "physical"
        panels[0]["labelZ"] = True
    else:
        panels[2]["labelZ"] = True  # upper right
    panels[-1]["labelScale"] = "physical"
    panels[-1]["labelHalo"] = "Mstar"

    class plotConfig:
        plotStyle = "open"
        rasterPx = 1080
        nRows = 2
        colorbars = True
        title = False

        saveFilename = savePathDefault + "timePanels_%s_subhaloID-%d_%02d.pdf" % (sP.simName, subhaloID, frame)

    renderSingleHalo(panels, plotConfig, locals())


def _create_global_subset_saves(doInds=False):
    """Pre-create the global subset saves used in static_halo_rotation_fullbox().

    If doInds, then Coordinates are loaded float32 for memory savings.
    """
    sP = simParams(res=2160, run="tng", redshift=0.5)

    fieldsToCache = ["Masses", "Density"]  # ['Coordinates']
    shIDs = [54051, 77279, 92816, 105136, 115152, 124403, 131715, 160812, 166602, 173133, 178497]
    shIDs += [190042, 194783, 199396, 202856, 211991, 216072]

    size = 2.2
    depthFac = 2.0

    # one field at a time
    for field in fieldsToCache:
        data = {}

        # load snapshot
        print("Loading [%s] now..." % field, flush=True)
        data[field] = sP.snapshotSubset("gas", field, float32=doInds)
        print("All loading done.", flush=True)

        for shID in shIDs:
            saveFilename = sP.cachePath + "vis_static_halo_rotation_fullbox_sh%d_s%d.hdf5" % (shID, sP.snap)

            # load subhalo info
            subhalo = sP.groupCatSingle(subhaloID=shID)
            halo = sP.groupCatSingle(haloID=subhalo["SubhaloGrNr"])

            # reduce to reasonable spatial subset
            print("[%d] Subsetting..." % shID, flush=True)
            boxSizeMax = halo["Group_R_Crit200"] * (size / 2) * depthFac

            mask = np.zeros(data[field].shape[0], dtype="int16")

            data_loc = {}

            # save already exists? get inds ('Coordinates' not required in fieldsToCache)
            if isfile(saveFilename):
                with h5py.File(saveFilename, "r") as f:
                    data_loc["inds"] = f["inds"][()]
                print("Loaded inds from [%s]" % saveFilename)
            else:
                # generate now ('Coordinates' required in fieldsToCache)
                for i in range(3):
                    dx = (data["Coordinates"][:, i] - subhalo["SubhaloPos"][i]).astype("float32")
                    sP.correctPeriodicDistVecs(dx)
                    w = np.where(np.abs(dx) > boxSizeMax)
                    mask[w] = 1
                    print(" %d" % i, flush=True)

                # create subset
                w = np.where(mask == 0)[0]
                data_loc["inds"] = w

            print(" [%d] of [%d] in spatial subset" % (len(data_loc["inds"]), data[field].shape[0]))
            data_loc[field] = data[field][data_loc["inds"], ...]

            with h5py.File(saveFilename) as f:
                for key in data_loc:
                    if key in f:
                        f[key][...] = data_loc[key]
                    else:
                        f[key] = data_loc[key]

            print(" Saved: [%s]" % saveFilename, flush=True)


def static_halo_rotation_fullbox(objInd=10, conf="one"):
    """Create movie frames for a rotation about a single halo in a fullbox (static in time)."""
    res = 2160
    redshift = 0.5
    run = "tng"
    method = "sphMap_global"

    axes = [0, 1]  # x,y
    labelScale = "physical"
    labelZ = True
    labelHalo = "mstar,mhalo,id"
    plotHalos = False
    nPixels = [3840, 2160]

    numFramesPerRot = 600  # 20 sec rotation, 0.6 deg per frame

    size = 2.2
    depthFac = 2.0
    sizeType = "rVirial"

    # objects (mhalo_200_log > 13.0, excluding sh==0)
    shIDs = [54051, 77279, 92816, 105136, 115152, 124403, 131715, 160812, 166602, 173133, 178497]
    shIDs += [190042, 194783, 199396, 202856, 211991, 216072]

    subhaloInd = shIDs[objInd]

    sP = simParams(res=res, run=run, redshift=redshift)
    subhalo = sP.groupCatSingle(subhaloID=subhaloInd)
    halo = sP.groupCatSingle(haloID=subhalo["SubhaloGrNr"])

    # define panels
    fieldsToCache = ["Coordinates", "Masses", "Density"]

    if conf == "one":
        panel = {"partType": "gas", "partField": "coldens_msunkpc2", "ctName": "magma", "valMinMax": [5.0, 8.0]}
    if conf == "two":
        panel = {"partType": "gas", "partField": "metal_solar", "valMinMax": [-1.5, 0.2]}
        fieldsToCache.append("GFM_Metallicity")
    if conf == "three":
        panel = {"partType": "gas", "partField": "velmag", "valMinMax": [0, 1000]}
        fieldsToCache.append("Velocities")
    if conf == "four":
        panel = {"partType": "gas", "partField": "radvel", "valMinMax": [-400, 400], "ctName": "BdRd_r_black"}
        fieldsToCache.append("Velocities")
    if conf == "five":
        panel = {"partType": "gas", "partField": "bmag_uG", "valMinMax": [-1.0, 1.6]}

    # global pre-cache of selected fields into memory, as we do global renders
    dataCache = {}

    saveFilename = sP.cachePath + "vis_static_halo_rotation_fullbox_sh%d_s%d.hdf5" % (subhaloInd, sP.snap)

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                cache_key = "snap%s_%s_%s" % (sP.snap, panel["partType"], key)
                dataCache[cache_key] = f[key][()]
        print("Loaded: [%s]" % saveFilename)
    else:
        print("No cache exists!")
        import pdb

        pdb.set_trace()  # will work to just continue, but better to cache first

    # render
    class plotConfig:
        saveFilename = ""
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = True
        colorbarOverlay = True

    # loop over frames
    for frame in range(numFramesPerRot):
        # redefine panels for each frame (avoid caching e.g. rotMatrix)
        panels = [panel.copy()]

        # derive rotation
        print(" [%s] subhalo ID = %d, frame = %3d" % (conf, subhaloInd, frame), flush=True)

        rotAngleDeg = 360.0 * (frame / numFramesPerRot)
        dirVec = [0.1, 1.0, 0.4]  # full non-axis aligned tumble

        rotCenter = subhalo["SubhaloPos"]
        rotMatrix = rotationMatrixFromAngleDirection(rotAngleDeg, dirVec)

        plotConfig.saveFilename = savePathBase + "%s_s%d_sh%d/frame_%s_%d.png" % (
            sP.simName,
            sP.snap,
            subhaloInd,
            conf,
            frame,
        )

        renderSingleHalo(panels, plotConfig, locals())
