"""
Render specific halo visualizations.
"""

import numpy as np

from ..cosmo.util import crossMatchSubhalosBetweenRuns
from ..util.helper import pSplit
from ..util.simParams import simParams
from ..vis.common import savePathDefault
from ..vis.halo import renderSingleHalo, selectHalosFromMassBin


def oneHaloSingleField(conf=0, haloID=None, subhaloInd=None, redshift=0.0):
    """In a single panel(s) centered on a halo, show one field from the box."""
    panels = []

    sP = simParams(run="tng100-1", redshift=redshift, variant=None)

    rVirFracs = [0.5, 1.0]  # None
    method = "sphMap"  # 'voronoi_slice', 'voronoi_proj'
    nPixels = [1200, 1200]  # [800,800] #[1920,1920]
    axes = [0, 1]
    labelZ = True
    labelScale = True
    labelSim = True
    labelHalo = True
    relCoords = True
    rotation = None
    mpb = None

    if haloID is not None:
        # periodic box, FoF/Halo ID
        subhaloInd = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

    if conf == 0:
        # magnetic pressure, gas pressure, and their ratio
        panels.append({"partType": "gas", "partField": "P_B", "valMinMax": [1.0, 9.0]})
        panels.append({"partType": "gas", "partField": "P_gas", "valMinMax": [1.0, 9.0]})
        panels.append({"partType": "gas", "partField": "pressure_ratio", "valMinMax": [-4.0, 0.0]})
    if conf == 1:
        # stellar mass column density
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [4.8, 7.8]})
    if conf == 2:
        # dm column density
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.8]})
    if conf == 3:
        # shock mach number
        panels.append(
            {
                "partType": "gas",
                "partField": "shocks_dedt",
                "valMinMax": [35, 39.5],
                "method": "histo",
                "nPixels": [4000, 4000],
                "smoothFWHM": 0.5,
            }
        )
    if conf == 4:
        # gas column density
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.0, 7.0]})
    if conf == 5:
        # magnetic field strength
        # panels.append( {'partType':'gas', 'partField':'bmag_uG',   'valMinMax':[-9.0,0.5]} )
        # panels.append( {'partType':'gas', 'partField':'bmag_uG',   'valMinMax':[-3.0,3.5]} )
        # panels.append( {'partType':'gas', 'partField':'xray_lum_05-2kev',  'valMinMax':[34,38]} )
        panels.append({"partType": "gas", "partField": "sz_yparam", "valMinMax": [-8, -3]})
        # panels.append( {'partType':'gas', 'partField':'EW_MgII 2803',  'valMinMax':[-1.0,1.0]} )
    if conf == 6:
        # stellar composite
        panels.append({"partType": "stars", "partField": "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"})
    if conf == 7:
        # gas temp
        panels.append({"partType": "gas", "partField": "temp_sfcold", "valMinMax": [4.0, 6.8]})

    if 1:
        size = 2.5
        sizeType = "rVirial"
    if 0:
        size = 6000.0
        sizeType = "kpc"
    if 0:
        size = 40.0
        sizeType = "rHalfMassStars"
        rVirFracs = [2.0, 10.0]
        fracsType = "rHalfMassStars"
    if 0:
        plotSubhalos = 100
        size = 3500.0
        sizeType = "codeUnits"

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1200
        colorbars = True
        saveFilename = "./oneHaloSingleField_%d_%s_z%.1f_ID-%d_%s.png" % (
            conf,
            sP.simName,
            redshift,
            subhaloInd if subhaloInd is not None else haloID,
            method,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def oneHaloGaussProposal():
    """Render single halo with B field streamlines for Gauss proposal (MHD figure)."""
    subhaloInd = 4
    panels = []

    # panels.append( {'partField':'bmag_uG', 'valMinMax':[-2.0,1.0]} )
    panels.append({"partField": "coldens_msunkpc2", "valMinMax": [6.0, 7.2]})
    # panels.append( {'partField':'bfield_x', 'valMinMax':[-1e-2,1e-2]} )
    # panels.append( {partField':'bfield_y', 'valMinMax':[-1e-2,1e-2]} )

    run = "tng"
    res = 1820
    redshift = 0.0
    partType = "gas"
    rVirFracs = [1.0]
    method = "sphMap"
    nPixels = [960, 960]
    size = 0.3  # central object
    sizeType = "rVirial"
    axes = [1, 0]
    labelZ = False
    labelScale = False
    labelSim = False
    labelHalo = False
    relCoords = True
    rotation = None
    mpb = None

    vecOverlay = "bfield"  # experimental B field streamlines
    vecMethod = "E"  # colored streamlines, uniform thickness
    vecColorPT = "gas"
    vecColorPF = "bmag"

    class plotConfig:
        plotStyle = "open"
        rasterPx = 960
        colorbars = True
        saveFilename = "./gasDens_%s_%d_z%.1f_sh-%d.pdf" % (run, res, redshift, subhaloInd)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def oneGalaxyThreeRotations(conf=0):
    """Plot stellar and gas images of a single galaxy, with three different rotations."""
    run = "illustris"
    res = 1820
    redshift = 0.0
    subhaloInd = 283832
    rVirFracs = None
    method = "sphMap"
    nPixels = [700, 700]
    axes = [0, 1]
    labelZ = False
    labelSim = False
    labelHalo = True
    relCoords = True
    mpb = None
    rotations = [None, "face-on", "edge-on"]

    # size       = 70.0 # 140 ckpc/h size length
    # sizeType   = 'codeUnits'
    size = 5.0
    sizeType = "rHalfMassStars"

    if conf == 0:
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"

    if conf == 1:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [6.0, 8.2]

    # create panels, one per view
    sPloc = simParams(res=res, run=run, redshift=redshift)
    sub = sPloc.groupCatSingle(subhaloID=subhaloInd)
    panels = []

    for i, rot in enumerate(rotations):
        labelScaleLoc = True if i == 0 else False
        panels.append({"rotation": rot, "labelScale": labelScaleLoc})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 700
        colorbars = False
        saveFilename = "./fig_galaxy_%s_%d_shID=%d_%s.pdf" % (sPloc.simName, sPloc.snap, subhaloInd, partType)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def resSeriesGaussProposal(fofInputID=12, resInput=256):
    """Render a 3x2 panel resolution series for Gauss proposal (MW/structural fig)."""
    panels = []

    run = "tng"
    variant = "L12.5"
    fofHaloIDs = [fofInputID]
    resLevels = [resInput]

    # run = 'illustris'
    # fofHaloIDs = [196] # subhalo ID 283832
    # resLevels = [1820]

    valMinMaxG = [6.0, 8.2]
    valMinMaxS = [6.0, 9.0]

    redshift = 0.0
    rVirFracs = None
    method = "sphMap"
    nPixels = [960, 960]
    size = 80.0  # central object
    sizeType = "codeUnits"
    axes = [0, 1]
    labelZ = False
    labelScale = True
    labelSim = False
    labelHalo = False
    relCoords = True
    mpb = None

    for fofHaloID, res in zip(fofHaloIDs, resLevels):
        # get subhalo ID
        sP = simParams(res=res, run=run, redshift=redshift, variant=variant)
        h = sP.groupCatSingle(haloID=fofHaloID)
        subhaloInd = h["GroupFirstSub"]
        print("subhalo ID: ", subhaloInd)

        # which stellar composite to use
        pF_stars = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"

        # append panels
        panels.append(
            {"partType": "stars", "partField": pF_stars, "rotation": "face-on", "size": 50.0, "sizeType": "codeUnits"}
        )
        panels.append(
            {"partType": "gas", "partField": "coldens_msunkpc2", "rotation": "face-on", "valMinMax": valMinMaxG}
        )

        #'nPixels':[960,320], \ # reduce vertical size of edge-on panels
        panels.append(
            {"partType": "stars", "partField": pF_stars, "rotation": "edge-on", "size": 50.0, "sizeType": "codeUnits"}
        )
        panels.append(
            {"partType": "gas", "partField": "coldens_msunkpc2", "rotation": "edge-on", "valMinMax": valMinMaxG}
        )

    class plotConfig:
        plotStyle = "open"
        rasterPx = 960
        colorbars = True
        haloStr = "-".join([str(r) for r in fofHaloIDs])
        resStr = "-".join([str(r) for r in resLevels])
        saveFilename = "./resSeriesGauss_%s_%s_%s.pdf" % (run, haloStr, resStr)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def multiHalosPagedOneQuantity(curPageNum, numPages=7):
    """Split over several pages, plot many panels, one per halo, showing a single quantity."""
    panels = []

    # subhalo ID list (Guinevere sample)
    sP = simParams(res=1820, run="illustris", redshift=0.0)
    subhaloIDs = np.loadtxt(sP.derivPath + "guinevere.list.subs.txt", dtype="int32")

    # split by page, append one panel per subhalo on this page
    subhaloIDs_loc = pSplit(subhaloIDs, numPages, curPageNum)

    for shID in subhaloIDs_loc:
        panels.append({"subhaloInd": shID})

    # panels.append( {'partField':'HI', 'valMinMax':[14.0,21.0]} )
    # panels.append( {'partField':'velmag', 'valMinMax':[400,900]} )
    # panels.append( {'partField':'metal_solar', 'valMinMax':[-1.0,0.5]} )
    # panels.append( {'partField':'Si III', 'valMinMax':[14.0,21.0]} )

    run = sP.run
    res = sP.res
    redshift = sP.redshift
    partType = "gas"
    partField = "Si II"  #'O VI' #'HI_segmented'
    valMinMax = [14.0, 17.0]  # [13.5, 15.5] #[13.5,21.5]
    rVirFracs = [1.0]
    method = "sphMap"  # sphMap_global
    nPixels = [960, 960]
    size = 140.0  # size = 10^2.8 * 0.7 * 2 (match to pm2.0 for Guinevere)
    sizeType = "codeUnits"
    axes = [1, 0]
    rotation = None

    class plotConfig:
        plotStyle = "open_black"
        colorbars = True
        saveFilename = savePathDefault + "sample_%s_page-%d-of-%d_%s_%d.pdf" % (
            partField,
            curPageNum,
            numPages,
            run,
            res,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def loopInputSerial():
    """Call another driver function several times, looping over a possible input."""
    numPages = 7
    for i in range(numPages):
        multiHalosPagedOneQuantity(i)


def boxHalo_HI():
    """Single halo HI plots (col dens, line of sight velocity) with smoothing."""
    panels = []

    vmm_col = [13.5, 21.5]  # 1/cm^2
    vmm_vel = [-300, 300]  # km/s

    # smoothing
    # panels.append( {'smoothFWHM':None, 'partField':'HI_segmented', 'valMinMax':vmm_col, 'labelScale':True} )
    # panels.append( {'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    # panels.append( {'smoothFWHM':6.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    # panels.append( {'smoothFWHM':None, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # panels.append( {'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # panels.append( {'smoothFWHM':6.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # axes = [0,1]

    # rotations
    # panels.append( {'rotation':None, 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    # panels.append( {'rotation':'edge-on', 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    # panels.append( {'rotation':'face-on', 'smoothFWHM':2.0, 'partField':'HI_segmented', 'valMinMax':vmm_col} )
    # panels.append( {'rotation':None, 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # panels.append( {'rotation':'edge-on', 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # panels.append( {'rotation':'face-on', 'smoothFWHM':2.0, 'partField':'vel_los', 'valMinMax':vmm_vel} )
    # size = 2.5

    # proposed
    panels.append({"rotation": None, "size": 2.5, "partField": "HI_segmented", "valMinMax": vmm_col})
    panels.append(
        {
            "rotation": "edge-on",
            "size": 120.0,
            "sizeType": "codeUnits",
            "partField": "HI_segmented",
            "valMinMax": vmm_col,
        }
    )
    panels.append(
        {
            "rotation": "face-on",
            "size": 120.0,
            "sizeType": "codeUnits",
            "partField": "HI_segmented",
            "valMinMax": vmm_col,
        }
    )
    panels.append({"rotation": None, "size": 2.5, "partField": "HI_segmented", "valMinMax": vmm_col})
    panels.append(
        {"rotation": "edge-on", "size": 120.0, "sizeType": "codeUnits", "partField": "vel_los", "valMinMax": vmm_vel}
    )
    panels.append(
        {"rotation": "face-on", "size": 120.0, "sizeType": "codeUnits", "partField": "vel_los", "valMinMax": vmm_vel}
    )
    # smoothFWHM = 2.0
    labelScale = True

    subhaloInd = 362540
    run = "illustris"
    partType = "gas"
    res = 1820
    redshift = 0.0
    rVirFracs = [1.0]  # None
    method = "sphMap"
    nPixels = [960, 960]
    rotation = None

    class plotConfig:
        plotStyle = "open"
        colorbars = True
        rasterPx = 960
        saveFilename = savePathDefault + "fig5b_%s_%d_z%.1f_shID-%d.pdf" % (run, res, redshift, subhaloInd)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def boxHalo_MultiQuant():
    """Diagnostic plot, a few quantities of a halo from a periodic box."""
    panels = []

    # panels.append( {'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,7.5]} )
    # panels.append( {'partType':'dm', 'partField':'coldens2_msunkpc2', 'valMinMax':[12.0,14.0]}

    partField = "coldens_msunkpc2"

    panels.append({"rotation": "edge-on", "nPixels": [960, 960], "partType": "gas", "valMinMax": [6.5, 8.4]})
    panels.append({"rotation": "edge-on", "nPixels": [960, 960], "partType": "stars", "valMinMax": [7.5, 9.0]})
    panels.append({"rotation": "edge-on", "nPixels": [1920, 1920], "partType": "stars", "valMinMax": [7.5, 9.0]})
    panels.append({"rotation": "face-on", "nPixels": [960, 960], "partType": "gas", "valMinMax": [6.5, 8.4]})
    panels.append({"rotation": "face-on", "nPixels": [960, 960], "partType": "stars", "valMinMax": [7.5, 9.0]})
    panels.append({"rotation": "face-on", "nPixels": [1920, 1920], "partType": "stars", "valMinMax": [7.5, 9.0]})

    # panels.append( {'nPixels':[960,960],  'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[4.5,8.4]} )
    # panels.append( {'nPixels':[960,960],  'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} )
    # panels.append( {'nPixels':[1920,1920],'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} )

    subhaloInd = 362540
    run = "illustris"
    res = 1820
    redshift = 0.0
    rVirFracs = [1.0]  # None
    method = "sphMap"
    # nPixels    = [1920,1920]
    size = 2.5  # -50.0
    sizeType = "rVirial"
    # axes       = [1,2]
    # rotation   = None

    class plotConfig:
        plotStyle = "open_black"
        colorbars = True
        saveFilename = savePathDefault + "box_%s_%d_z%.1f_shID-%d_multiQuant_sf-%.1f.pdf" % (
            run,
            res,
            redshift,
            subhaloInd,
            size,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def zoomHalo_z2_MultiQuant():
    """For a single zooms/zooms2 halo at z=2, plot several panels comparing different quantities."""
    panels = []

    panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 9.5]})
    panels.append({"partType": "gas", "partField": "entr", "valMinMax": [6.0, 9.0]})
    panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [100, 300]})
    panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [11.0, 17.0]})
    panels.append({"partType": "gas", "partField": "C IV", "valMinMax": [11.0, 17.0]})
    panels.append({"partType": "gas", "partField": "HI", "valMinMax": [16.0, 22.0]})

    hInd = 2
    run = "zooms2"
    res = 10
    redshift = 2.0
    subhaloInd = 0

    rVirFracs = [1.0]
    method = "sphMap"  # sphMap_global
    nPixels = [960, 960]
    size = 2.5
    sizeType = "rVirial"
    axes = [1, 0]
    rotation = None

    class plotConfig:
        plotStyle = "open_black"
        colorbars = True
        saveFilename = savePathDefault + "%s_h%dL%d_z%.1f_multiQuant.pdf" % (run, hInd, res, redshift)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def tng50_firstNhalos(conf=0):
    """Plot gas/stellar densities of the N most massive halos (at some redshift). All separate (fullpage) plots."""
    run = "tng"
    res = 2160
    redshift = 6.0
    nHalos = 10

    rVirFracs = [0.1]
    method = "sphMap"
    nPixels = [1000, 1000]
    size = 100.0
    sizeType = "codeUnits"
    labelZ = True
    labelScale = True
    labelHalo = True
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

    sP = simParams(res=res, run=run, redshift=redshift)

    # render one plot per halo
    for i in range(nHalos):
        halo = sP.groupCatSingle(haloID=i)
        print(sP.simName, sP.snap, sP.redshift, i, halo["GroupFirstSub"], halo["GroupPos"])

        panels = [{"subhaloInd": halo["GroupFirstSub"]}]

        class plotConfig:
            plotStyle = "open_black"
            colorbars = True
            rasterPx = 1000
            saveFilename = savePathDefault + "%s_haloInd-%d_%s-%s_z%.1f.pdf" % (
                sP.simName,
                i,
                partType,
                partField,
                redshift,
            )

        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def tngMethods2_stamps(conf=0, curPage=None, numPages=None, rotation=None, variant=None, matchedToVariant=None):
    """Plot stellar stamps of N random massive L25n512_0000 galaxies.

    If curPage,numPages both specified, do a paged exploration instead.
    If matchedToVariant is not None, then run the halo selection on this variant instead (e.g. always 0000)
    and then use the SubhaloMatching catalog to pick the matched halos in this run.
    https://arxiv.org/abs/1703.02970 (Figure 3).
    """
    run = "tng"
    res = 1024
    redshift = 0.0
    # variant   = 4503 #0010 #0000
    massBin = [11.8, 14.0]
    nGalaxies = 15
    selType = "random"

    # stellar composite, 50 kpc/h on a side, include M* label per panel, and scale bar once
    # select random in [12.0,inf] halo mass range, do each of: face-on, edge-on, random(z)
    # possible panel configurations: 4x2 = 8, 4x3 = 12, 5x3 = 15, 6x3 = 18
    rVirFracs = None
    method = "sphMap"
    nPixels = [700, 700]
    axes = [0, 1]
    labelZ = False
    labelSim = False
    labelHalo = "Mstar"
    relCoords = True
    mpb = None

    size = 50.0  # 25 ckpc/h each direction
    sizeType = "codeUnits"
    # size       = 6.0
    # sizeType   = 'rHalfMassStars'

    if conf == 0:
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
        # partField = 'stellarComp-snap_K-snap_B-snap_U'
        # partField = 'stellarComp-sdss_i-sdss_r-sdss_g' #irg # zirgu (red -> blue)

    if conf == 1:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [6.8, 8.8]  # for z=3 1024 # [6.0, 8.2] for z=2 512

    # load halos of this bin, from this run or from the matchedToVariant source run
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    sP_from = sP
    mvStr = ""

    if matchedToVariant is not None:
        sP_from = simParams(res=res, run=run, redshift=redshift, variant=matchedToVariant)
        mvStr = "_matched-to-%s" % sP_from.simName

    if curPage is None:
        # non-paged, load requested number
        shIDs, _ = selectHalosFromMassBin(sP_from, [massBin], nGalaxies, massBinInd=0, selType=selType)
        saveFilename2 = "./methods2_stamps_%s_%s_rot=%s%s.pdf" % (sP.simName, partType, rotation, mvStr)
    else:
        # paged, load all and sub-divide
        if sP.res == 512:
            massBin = [11.8, 14.0]  # methods2 paper
        if sP.res == 1024:
            massBin = [11.3, 14.0]
        shIDsAll, _ = selectHalosFromMassBin(sP_from, [massBin], 200, massBinInd=0, selType="linear")
        shIDs = pSplit(shIDsAll, numPages, curPage)[0:nGalaxies]
        assert shIDs.size == nGalaxies  # make sure we have nGalaxies per render
        saveFilename2 = "./methods2_pages_%s_%s_rot=%s%s_page-%dof%d.pdf" % (
            sP.simName,
            partType,
            rotation,
            mvStr,
            curPage,
            numPages,
        )

    # if we loaded the subhalo list to plot from another run, match them to the current run
    if matchedToVariant is not None:
        shIDs = crossMatchSubhalosBetweenRuns(sP_from, sP, shIDs)
        assert shIDs.min() >= 0  # if any matches failed, we should make a blank panel

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        labelScaleLoc = True if (i == 0 or sizeType == "rHalfMassStars") else False
        panels.append({"subhaloInd": shID, "labelScale": labelScaleLoc})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 500
        colorbars = False
        saveFilename = saveFilename2

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def loop_stamps():
    """Helper."""
    numPages = 10

    # plot some other run choosing its subhalos matched to the 0000 selection
    variant = 4503  # 0010 for methods2
    matchedToVariant = 0000

    for conf in [0, 1]:
        for rotation in [None, "edge-on", "face-on"]:
            for curPage in range(numPages):
                print(conf, rotation, curPage, numPages)
                tngMethods2_stamps(
                    conf=conf,
                    curPage=curPage,
                    numPages=numPages,
                    rotation=rotation,
                    variant=variant,
                    matchedToVariant=matchedToVariant,
                )

    # plot the 0000 selection itself
    variant = 0000
    matchedToVariant = None

    for conf in [0, 1]:
        for rotation in [None, "edge-on", "face-on"]:
            for curPage in range(numPages):
                print(conf, rotation, curPage, numPages)
                tngMethods2_stamps(
                    conf=conf,
                    curPage=curPage,
                    numPages=numPages,
                    rotation=rotation,
                    variant=variant,
                    matchedToVariant=matchedToVariant,
                )


def massBinsSample_3x2_EdgeOnFaceOn(res, conf, haloOrMassBinNum=None, panelNum=None):
    """For a series of mass bins, take a uniform number of halos from each and make different plots.

    Either (i) a 3x2 plot with the top row face-on and the bottom row edge-on, one plot per galaxy, or
    (ii) montage pages showing many galaxies at once, one per mass bin and quantify.
    """
    massBins = [[12.7, 13.0], [12.3, 12.5], [12.0, 12.1], [11.5, 11.6]]
    numPerBin = 20

    assert haloOrMassBinNum is not None and haloOrMassBinNum < numPerBin * len(massBins)

    panels = []

    # res        = 2160
    redshift = 1.0
    run = "tng"
    rVirFracs = None
    method = "sphMap"
    nPixels = [960, 960]
    size = 80.0
    sizeType = "codeUnits"
    axes = [0, 1]

    class plotConfig:
        plotStyle = "open"
        rasterPx = 1400
        colorbars = True

    # configure panels
    starsMM = [6.5, 10.0]  # coldens_msunkpc2
    gasMM = [6.5, 8.2]
    gasMMedge = [6.5, 8.8]

    if conf == "single_halos":
        # loop over centrals in mass bins, one figure each
        sP = simParams(res=res, run=run, redshift=redshift)

        subhaloInd, binInd = selectHalosFromMassBin(sP, massBins, numPerBin, haloNum=haloOrMassBinNum)

        if subhaloInd is None:
            print("Task past bin size, quitting.")
            return

        haloInd = sP.groupCatSingle(subhaloID=subhaloInd)["SubhaloGrNr"]
        if binInd >= 2:
            size = 40.0

        panels.append(
            {
                "rotation": "face-on",
                "partType": "stars",
                "partField": "coldens_msunkpc2",
                "valMinMax": starsMM,
                "labelHalo": True,
            }
        )
        panels.append({"rotation": "face-on", "partType": "stars", "partField": "stellarComp"})
        panels.append({"rotation": "face-on", "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMM})
        panels.append(
            {"rotation": "edge-on", "partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": starsMM}
        )
        panels.append({"rotation": "edge-on", "partType": "stars", "partField": "stellarComp"})
        panels.append(
            {"rotation": "edge-on", "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMMedge}
        )

        plotConfig.saveFilename = savePathDefault + "renderHalo_%s-%d_bin%d_halo%d_hID-%d_shID-%d.pdf" % (
            sP.simName,
            sP.snap,
            binInd,
            haloOrMassBinNum,
            haloInd,
            subhaloInd,
        )

    if conf == "halos_combined":
        # combined plot of centrals in mass bins
        sP = simParams(res=res, run=run, redshift=redshift)

        shIDs, binInd = selectHalosFromMassBin(sP, massBins, numPerBin, massBinInd=haloOrMassBinNum)

        if binInd >= 2:
            size = 40.0

        for shID in shIDs:
            if panelNum == 0:
                panels.append(
                    {
                        "subhaloInd": shID,
                        "rotation": "face-on",
                        "partType": "stars",
                        "partField": "coldens_msunkpc2",
                        "valMinMax": starsMM,
                    }
                )
            if panelNum == 1:
                panels.append(
                    {"subhaloInd": shID, "rotation": "face-on", "partType": "stars", "partField": "stellarComp"}
                )
            if panelNum == 2:
                panels.append(
                    {
                        "subhaloInd": shID,
                        "rotation": "face-on",
                        "partType": "gas",
                        "partField": "coldens_msunkpc2",
                        "valMinMax": gasMM,
                    }
                )
            if panelNum == 3:
                panels.append(
                    {
                        "subhaloInd": shID,
                        "rotation": "edge-on",
                        "partType": "stars",
                        "partField": "coldens_msunkpc2",
                        "valMinMax": starsMM,
                    }
                )
            if panelNum == 4:
                panels.append(
                    {"subhaloInd": shID, "rotation": "edge-on", "partType": "stars", "partField": "stellarComp"}
                )
            if panelNum == 5:
                panels.append(
                    {
                        "subhaloInd": shID,
                        "rotation": "edge-on",
                        "partType": "gas",
                        "partField": "coldens_msunkpc2",
                        "valMinMax": gasMMedge,
                    }
                )
            if panelNum == 6:
                panels.append(
                    {"subhaloInd": shID, "partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": starsMM}
                )
            if panelNum == 7:
                panels.append(
                    {"subhaloInd": shID, "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMM}
                )

        rotStr = "_" + panels[0]["rotation"] if "rotation" in panels[0] else ""
        plotConfig.saveFilename = savePathDefault + "renderHalo_%s-%d_bin%d_%s-%s%s.pdf" % (
            sP.simName,
            sP.snap,
            binInd,
            panels[0]["partType"],
            panels[0]["partField"],
            rotStr,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def loopMassBins():
    """Call another driver function several times, looping over a possible input."""
    nMassBins = 4
    for res in [1080]:
        for i in range(20 * nMassBins):
            massBinsSample_3x2_EdgeOnFaceOn(res, "single_halos", haloOrMassBinNum=i)
        for i in range(nMassBins):
            for j in range(8):
                massBinsSample_3x2_EdgeOnFaceOn(res, "halos_combined", haloOrMassBinNum=i, panelNum=j)


def tngFlagship_galaxyRedBlue(blueSample=False, redSample=False, greenSample=False, evo=False, curPage=None, conf=0):
    """Plot stellar stamps red/blue galaxies around 10^10.5 Msun.

    If evo==True, then tracked back in time from z=0 to z=2.0 in M steps using the merger tree.
    If evo==False, then show full NxM panel sample at z=0.
    In either case, choose blueSample, redSample, or greenSample.
    If curPage specified, do a paged exploration instead.
    """
    # we have chosen by hand for L75n1820TNG z=0 from the massBin = [12.0,12.2] below these two sets
    # we define blue as (g-r)<0.6, red as (g-r)>0.7 and green as 0.5<(g-r)<0.7
    blue_z0 = [438297, 448732, 446548, 452577, 455335, 463062, 463649, 464576, 460692, 468590, 466436, 469315, 470617,
               472457, 471740, 473004, 473349, 473898, 474813, 476487, 477179, 477677, 479174, 479117, 480230, 480441,
               481126, 481300, 482169, 483370, 482869, 484771, 485441, 486590, 487244,  # 0-35
               486525, 486966, 488415, 486781, 488123, 488500, 488722, 489054, 489100, 489593, 490280, 490806, 491282,
               493034, 493751, 493491, 493517, 493964, 494771, 494273, 497032, 496990, 495515, 497646, 498576, 499130,
               499996, 499223, 499463, 500867, 500494, 501761, 502312, 502648, 502919]  # 35-70 # fmt: skip
    red_z0 = [441141, 443914, 453835, 421956, 440477, 497926, 491801, 460076, 496436, 475490, 498958, 451243, 478160,
              479314, 479917, 502881, 461038, 467519, 469589, 473125, 481347, 482257, 505333, 483868, 482533, 484113,
              484257, 480645, 485365, 486052, 487152, 489314, 488841, 508985, 510751,  # 0-35
              490986, 469102, 492392, 492614, 493230, 494009, 495442, 471857, 497800, 499025, 499522, 500448, 502168,
              502956, 480194, 502461, 503393, 504142, 482714, 507070, 508985, 515318, 517881, 511005, 490577, 454963,
              463139, 477518, 469930, 480550, 480879, 480750, 481254, 483900, 485233]  # 35-70 # fmt: skip

    green_z0 = [415972, 418076, 429041, 429758, 438297, 438453, 445185, 452357, 454963, 455058, 459394, 459517, 460008,
                460526, 461283, 462010, 462141, 463139, 463958, 464669, 466801, 467127, 467445, 467519, 468318, 468450,
                468508, 468590, 468695, 469102, 469740, 469930, 470679, 471109, 471591, 471857, 472418, 472747, 473349,
                474271, 474437, 474860, 475542, 475722, 476487, 476892, 477031, 477474, 477518, 477624, 478452, 478661,
                478927, 479411, 479798, 479839, 480194, 480550, 480750, 480803, 480879, 481254, 481804, 482006, 482054,
                482169, 482196, 482714, 482757, 482869, 482987, 483179, 483421, 483484, 483900, 484207, 484307, 484715,
                484944, 485233, 485283, 485326, 485485, 486124, 486186, 486281, 486875, 487042, 487152, 487698, 487913,
                487965, 488228, 489273, 489356, 489666, 489812, 489953, 490109, 490172, 490195, 490238, 490475, 490533,
                490617, 490661, 490711, 491066, 491282, 491872, 491901, 492092, 492125, 492178, 492223, 492947, 493183,
                493437, 493575, 493658, 493751, 494326, 495442, 495772, 496019, 496244, 496292, 496326, 496369, 496757,
                496805, 497088, 497176, 497351, 497585, 497964, 498031, 498297, 499025, 499113, 499463, 499636, 500239,
                500268, 500294, 500448, 500725, 501430, 501526, 501604, 501658, 502347, 502427, 502595, 502767, 502956,
                503393, 503636, 503778, 503951, 504000, 504030, 504065, 505793, 505892, 505928, 506063, 506313, 506943,
                507299, 507361, 507587, 507701, 508156, 509018, 510085, 511576, 512330, 514988]  # 179 # fmt: skip

    # config
    run = "tng"
    res = 1820
    rVirFracs = None
    method = "sphMap"
    nPixels = [300, 300]
    axes = [0, 1]
    labelZ = False
    labelSim = False
    labelHalo = None  #'Mstar'
    relCoords = True
    mpb = None
    rotation = "face-on"
    sizeType = "kpc"

    # init
    evo_redshifts = [0.0, 0.2, 0.4, 0.7, 1.0]  # [0.0, 0.2, 0.5, 1.0, 2.0]
    redshift_init = 0.0

    sP = simParams(res=res, run=run, redshift=redshift_init)

    # which plot configuration?
    if conf == 0:
        size = 60.0  # 30 kpc in each direction from center
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
        colorBar = False
        nGalaxies = 35  # 35 (note: set to 45 for talk image)
        nRowsFig = 7  # 7 (note: set to 9 for talk image)
    if conf in [1, 2]:
        size = 800.0  # 400 kpc in each direction from center
        partType = "gas"
        partField = "O VI"
        hsmlFac = 3.5
        valMinMax = [12.0, 15.0]
        rVirFracs = [1.0]
        colorBar = True
        nGalaxies = 30
        nRowsFig = 6
        method = "sphMap"  # sphMap_global'
    if conf == 2:
        partField = "O VI fracmass"
        valMinMax = [-5.0, -3.0]
    if conf in [3]:
        size = 500.0  # 500: 250 kpc in each direction from the center; 800: 400 kpc in each direction from center
        partType = "gas"
        partField = "O VII"
        ctName = "magma"
        hsmlFac = 3.5
        valMinMax = [13.0, 16.5]
        rVirFracs = [1.0]
        colorBar = True
        nGalaxies = 30
        nRowsFig = 6
        method = "sphMap"  # sphMap_global'
    if conf in [4]:
        size = 500.0  # 500: 250 kpc in each direction from the center; 800: 400 kpc in each direction from center
        partType = "gas"
        partField = "O VIII"
        ctName = "magma"
        hsmlFac = 3.5
        valMinMax = [13.0, 16.5]
        rVirFracs = [1.0]
        colorBar = True
        nGalaxies = 30
        nRowsFig = 6
        method = "sphMap"  # sphMap_global'
    if conf in [5]:
        size = 500.0  # 500: 250 kpc in each direction from the center; 800: 400 kpc in each direction from center
        partType = "gas"
        partField = "Fe XVII"
        ctName = "magma"
        hsmlFac = 3.5
        valMinMax = [13.0, 16.5]
        rVirFracs = [1.0]
        colorBar = True
        nGalaxies = 30
        nRowsFig = 6
        method = "sphMap"  # sphMap_global'

        # global pre-cache of selected fields into memory
        if 0:
            fieldsToCache = ["Coordinates", "Masses", "O VI mass"]
            dataCache = {}
            for field in fieldsToCache:
                cache_key = "snap%d_%s_%s" % (sP.snap, partType, field.replace(" ", "_"))
                print("Caching [%s] now..." % field)
                dataCache[cache_key] = sP.snapshotSubsetP(partType, field)
            print("All caching done.")

    # load halos of this bin, from this run
    if curPage is None:
        if evo is False:
            # z=0 samples
            if redSample:
                shIDs = red_z0[(redSample - 1) * nGalaxies : (redSample) * nGalaxies]
            if blueSample:
                shIDs = blue_z0[(blueSample - 1) * nGalaxies : (blueSample) * nGalaxies]
            if greenSample:
                shIDs = green_z0[(greenSample - 1) * nGalaxies : (greenSample) * nGalaxies]

            redshifts = np.zeros(len(shIDs), dtype="float32") + redshift_init
        else:
            # take nRows galaxies from z0 samples, then load their MPBs and get IDs at earlier redshifts
            assert redSample > 0 or blueSample > 0 or greenSample > 0
            if redSample:
                shIDs_z0 = red_z0[(redSample - 1) * nRowsFig : (redSample) * nRowsFig]
            if blueSample:
                shIDs_z0 = blue_z0[(blueSample - 1) * nRowsFig : (blueSample) * nRowsFig]
            if greenSample:
                shIDs_z0 = green_z0[(greenSample - 1) * nRowsFig : (greenSample) * nRowsFig]

            evo_snapshots = sP.redshiftToSnapNum(evo_redshifts)
            shIDs = []
            redshifts = []

            for shID_z0 in shIDs_z0:
                # load main progenitor branch
                mpbLocal = sP.loadMPB(id=shID_z0)

                # append to shIDs and redshifts the 5 evolution steps for this subhalo
                for evo_snapshot, evo_redshift in zip(evo_snapshots, evo_redshifts):
                    treeIndex = tuple(mpbLocal["SnapNum"]).index(evo_snapshot)
                    shIDs.append(mpbLocal["SubfindID"][treeIndex])
                    redshifts.append(evo_redshift)

        saveFilename2 = "./stamps_%s_evo-%d_red-%d_blue-%d_rot-%s.pdf" % (
            sP.simName,
            evo,
            redSample,
            blueSample,
            rotation,
        )
    else:
        # paged exploration for picking interesting galaxies, load all and sub-divide
        if 1:
            baseStr = "pages"
            numPages = 20
            massBin = [12.0, 12.2]
        elif 0:
            baseStr = "pagesHighMass"
            numPages = 31
            massBin = [12.2, 13.5]

        shIDsAll, _ = selectHalosFromMassBin(sP, [massBin], 2000, massBinInd=0, selType="linear")
        shIDs = pSplit(shIDsAll, numPages, curPage)[0:nGalaxies]
        redshifts = np.zeros(len(shIDs), dtype="float32") + redshift_init

        assert shIDs.size == nGalaxies  # make sure we have nGalaxies per render
        saveFilename2 = "./%s_%s_%s_rot=%s_page-%dof%d_hsml%.1f-age1-3-2_size%.1f.pdf" % (
            baseStr,
            sP.simName,
            partType,
            rotation,
            curPage,
            numPages,
            hsmlFac,
            size,
        )

    # load colors and morphs for custom labeling
    # if not evo:
    # gr_colors_z0, _ = loadSimGalColors(sP, defSimColorModel, bands=['g','r'], projs='random')
    # kappa_stars, _, _, _ = sP.simSubhaloQuantity('Krot_oriented_stars2')
    # mass_ovi, _, _, _ = sP.simSubhaloQuantity('mass_ovi')
    # mass_ovi = logZeroNaN(mass_ovi)

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        lCust = ["%s, ID %d, z = %.1f" % (sP.simName, shID, redshifts[i])]
        lScale = True if i == 0 else False

        # if not evo:
        #    if conf == 0:
        #        detailsStr = '(g-r) = %.2f $\kappa_\star$ = %.2f' % (gr_colors_z0[shID],kappa_stars[shID])
        #    if conf in [1,2]:
        #        detailsStr = '(g-r) = %.2f $M_{\\rm OVI}$ = %.1f' % (gr_colors_z0[shID],mass_ovi[shID])
        #    lCust.append(detailsStr)

        panels.append({"subhaloInd": shID, "redshift": redshifts[i], "labelCustom": lCust, "labelScale": lScale})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1200
        colorbars = colorBar
        nRows = nRowsFig
        saveFilename = saveFilename2

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vogelsberger_redBlue42(run="illustris", sample="blue"):
    """Recreate the 'Vogelsberger+ (2014) sample of 42 red/blue galaxies', either for Illustris-1 or TNG100-1.

    Or, if sample=='guinevere', same for the Milky Way mass sample of Kauffmann+ (2016).
    """
    if sample == "red":
        illustris_ids = [123773, 178998, 215567, 234535, 251933, 267605, 299439, 310062, 326049, 344358, 359184,
                         378583, 385020, 393669, 135289, 192506, 217716, 239606, 262030, 272551, 305959, 310951,
                         331451, 350144, 372778, 382424, 386640, 393722, 154948, 195486, 219708, 242071, 264095,
                         293759, 309593, 318565, 342383, 354524, 377255, 384031, 392429, 395125]  # fmt: skip
    if sample == "blue":
        illustris_ids = [242959, 288927, 310273, 319371, 331996, 336439, 342103, 345367, 358785, 366034, 374531,
                         384734, 386479, 391881, 261085, 300120, 312287, 324323, 332327, 336790, 343669, 351433,
                         362540, 366317, 375386, 385795, 386720, 394285, 283832, 303990, 313541, 326247, 332891,
                         339311, 344821, 352713, 365316, 374140, 376363, 386304, 390653, 394942]  # fmt: skip
    if sample == "guinevere":
        illustris_ids = [127229, 129772, 140596, 175439, 185232, 203460, 245224, 248848, 287571, 293191, 315318,
                         344090, 345729, 348898, 349346, 349485, 353174, 355725, 357194, 358170, 359796, 364332,
                         368113, 371865, 371954, 372133, 408152, 73669, 86191, 140594, 152865, 175438, 198180,
                         200655, 214814, 224437, 257809, 291078, 302606, 308748, 330182, 335588, 339047, 340308,
                         340843, 346523, 351433, 354318, 354955, 356087, 356527, 360773, 361188, 362406, 366108,
                         368309, 376968, 383095, 392952, 412138, 358362, 359699, 360972, 362079, 363972, 366624,
                         369080, 369185, 371429, 373348, 373841, 373936, 374803, 377918, 380950, 383237, 384490,
                         389428, 389986, 390393, 390653, 391958, 392904, 393815, 395537, 397847, 399779, 403411,
                         408163, 409443, 413363, 227855, 255101, 267947, 268841, 317208, 326247, 348189, 356427,
                         360070, 362540, 364905, 368528, 368955, 374140, 375386, 375705, 377719, 379309, 380301,
                         380894, 384734, 384785, 385878, 385916, 389570, 390300, 394067, 394757, 398436, 398470,
                         330939, 348302, 357902, 359699, 395537, 408877, 409564, 413363, 413821, 418247, 424081,
                         424496, 430099, 432733, 458329]  # fmt: skip

    # config
    res = 1820
    redshift = 0.0
    rVirFracs = None
    method = "sphMap"
    nPixels = [300, 300]
    axes = [0, 1]
    labelZ = False
    labelSim = False
    labelHalo = "mstar"
    relCoords = True
    mpb = None
    labelScale = False
    rotation = "face-on"
    size = 60.0  # 30 kpc in each direction from center
    sizeType = "kpc"
    partType = "stars"
    partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
    nRowsFig = 7  # 6 columns, 7 rows
    matchMethod = "PositionalAll"  # Lagrange

    # which subhalos?
    sP = simParams(res=res, run=run, redshift=redshift)
    sP_illustris = simParams(res=res, run="illustris", redshift=redshift)

    if run == "illustris":
        subhalo_ids = illustris_ids
    else:
        # cross-match and get TNG subhalo IDs
        subhalo_ids = crossMatchSubhalosBetweenRuns(sP_illustris, sP, illustris_ids, method=matchMethod)

    if sample == "guinevere":
        # verify which are centrals
        nRowsFig = 8
        cen_flags = sP.groupCat(fieldsSubhalos=["cen_flag"])

        if run != "illustris":
            # write out text-file with matches
            header = "# Illustris-1 z=0 subhalo_id, %s z=0 subhalo_id, is_central_in_illustris" % run
            np.savetxt(
                "out.txt", np.vstack([illustris_ids, subhalo_ids, cen_flags]).T, header=header, fmt="%d", delimiter=", "
            )

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(subhalo_ids):
        cenSatStr = ""
        if sample == "guinevere" and run == "illustris":
            cenSatStr = " (SAT)" if cen_flags[i] == 0 else ""

        panels.append({"subhaloInd": shID, "labelCustom": ["ID %d%s" % (shID, cenSatStr)]})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 600
        colorbars = False
        nRows = nRowsFig
        saveFilename = "./sampleMatched_%s_%s_%s.pdf" % (sample, run, matchMethod)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)
