"""
Render specific fullbox visualizations.
"""

import numpy as np

from ..util.rotation import rotationMatrixFromVec
from ..util.simParams import simParams
from ..vis.box import renderBox
from ..vis.common import savePathDefault
from ..vis.halo import renderSingleHalo


def realizations(conf=1):
    """Render a whole box image at one redshift, of one field, comparing multiple runs."""
    panels = []

    # runs
    panels.append({"variant": "0000"})
    panels.append({"variant": "2302"})

    if conf == 1:
        partType = "dm"
        partField = "coldens_msunkpc2"
        valMinMax = [5.0, 8.5]

    if conf == 2:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [4.2, 7.2]

    if conf == 3:
        partType = "gas"
        partField = "O VI"
        valMinMax = [11, 15]

    run = "tng"
    res = 512
    redshift = 0.0
    nPixels = 800
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = 20

    # render config (global)
    class plotConfig:
        plotStyle = "open"
        rasterPx = 600
        colorbars = True

        saveFilename = savePathDefault + "realizations_%s_%s_nSp%d_z%.1f.pdf" % (
            partType,
            partField,
            len(panels),
            redshift,
        )

    renderBox(panels, plotConfig, locals())


def _TNGboxSliceConfig(res):
    """Get main slice config for presentation: slice depth, and center position."""
    # L75 configs
    dmMM = [5.0, 8.5]
    gasMM = [4.3, 7.3]
    starsMM = [1.0, 7.0]

    gasFullMM = [12.0, 16.0]  # equirectangular/full depth

    if res in [455, 910, 1820, 1024, 1504]:
        # L75
        centerHaloID = 1  # fof
        nSlicesTot = 3  # slice depth equal to a third, 25 Mpc/h = 37 Mpc
        curSlice = 0  # offset slice along projection direction?
    if res in [625, 1250, 2500]:
        # L205
        centerHaloID = 0  # fof
        nSlicesTot = 3  # slice depth equal to a third, ~68.333 Mpc/h ~ 100.875 Mpc
        curSlice = 0  # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 0.5
        gasMM[0] += 0.7
        gasFullMM[0] += 1.5
        gasFullMM[1] += 1.5
    if res in [2048]:
        # L680
        centerHaloID = 0  # fof
        nSlicesTot = 4  # slice depth equal to a third, ~170 Mpc/h ~ 251 Mpc
        curSlice = 0  # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 1.2
        dmMM[1] -= 0.2
        gasMM[0] += 1.6
    if res in [270, 540, 1080, 2160]:
        # L35
        centerHaloID = 0  # fof
        nSlicesTot = 1  # slice depth equal to a fifth, 35 Mpc/h = 52 Mpc
        curSlice = 0  # offset slice along projection direction?

        # adjust for deeper slice
        dmMM[0] += 0.3
        gasMM[0] += 0.5
    if res in [128, 256, 512]:
        # L25 variants
        centerHaloID = None
        nSlicesTot = None
        curSlice = None
    if res in [None, 1, 2]:
        # original Millennium simulations
        dmMM[0] += 2.0
        dmMM[1] += 1.0

        centerHaloID = None
        nSlicesTot = None
        curSlice = None

    return dmMM, gasMM, starsMM, gasFullMM, centerHaloID, nSlicesTot, curSlice


def _TNGboxFieldConfig(res, conf, thinSlice, remap=False):
    panels = []

    dmMM, gasMM, starsMM, gasFullMM, centerHaloID, nSlicesTot, curSlice = _TNGboxSliceConfig(res)

    if conf == 0:
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMM})
    if conf == 1:
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": dmMM})
    if conf == 2:
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [2.8, 6.5]})  # [2.0,6.4]
    if conf == 3:
        panels.append({"partType": "stars", "partField": "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"})
    if conf == 4:
        panels.append({"partType": "gas", "partField": "pressure_ratio", "valMinMax": [-8, 1], "cmapCenVal": -3.0})
    if conf == 5:
        panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-9.0, 0.5]})
    if conf == 6:
        panels.append({"partType": "gas", "partField": "Z_solar", "valMinMax": [-2.5, -0.4]})  # [-2.0,-0.2]
    if conf == 7:
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.3, 7.2]})
    if conf == 8:
        panels.append({"partType": "gas", "partField": "SN_IaII_ratio_Fe", "valMinMax": [0.0, 2.6]})
    if conf == 9:
        panels.append({"partType": "gas", "partField": "SN_IaII_ratio_metals", "valMinMax": [-1.0, 2.5]})
    if conf == 10:
        panels.append({"partType": "gas", "partField": "SN_Ia_AGB_ratio_metals", "valMinMax": [-0.48, 0.06]})
    if conf == 11:
        panels.append({"partType": "gas", "partField": "xray_lum", "valMinMax": [29, 37.5]})
    if conf == 12:
        panels.append({"partType": "gas", "partField": "shocks_machnum", "valMinMax": [0.0, 1.5]})
    if conf == 13:
        panels.append({"partType": "gas", "partField": "shocks_dedt", "valMinMax": [33, 38.5]})
    if conf == 14:
        panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [100, 1000]})
    if conf == 15:
        panels.append({"partType": "dm", "partField": "velmag", "valMinMax": [0, 1200]})
    if conf == 16:
        panels.append({"partType": "gas", "partField": "HI_segmented", "valMinMax": [13.5, 21.5]})

    # testing mip:
    if conf == 17:
        panels.append(
            {"partType": "gas", "partField": "shocks_machnum", "valMinMax": [0, 150], "method": "sphMap_maxIP"}
        )
    if conf == 18:
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.3, 8.0], "method": "sphMap_maxIP"})
    if conf == 19:
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [2.5, 4.5], "method": "sphMap_minIP"})
    if conf == 20:
        panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [200, 1000], "method": "sphMap_maxIP"})

    # more fields:
    if conf == 21:
        panels.append({"partType": "gas", "partField": "potential", "valMinMax": [-6.5, 6.5], "cmapCenVal": 0.0})
    if conf == 22:
        panels.append({"partType": "dm", "partField": "id"})
    if conf == 23:
        panels.append({"partType": "dm", "partField": "coldens_sq_msunkpc2", "valMinMax": [-4.0, 4.5]})
    if conf == 24:
        panels.append({"partType": "gas", "partField": "p_sync_ska", "valMinMax": [-5.0, -2.7]})
    if conf == 25:
        panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [11, 15]})
    if conf == 26:
        panels.append({"partType": "gas", "partField": "O VII", "valMinMax": [11, 16]})
    if conf == 27:
        panels.append({"partType": "gas", "partField": "O VIII", "valMinMax": [11, 16]})

    if conf == 28:
        panels.append({"partType": "gas", "partField": "sb_H-alpha", "valMinMax": [-13.0, -8.0]})
    if conf == 29:
        panels.append({"partType": "gas", "partField": "sb_OVIII", "valMinMax": [-15.0, -10.0]})
    if conf == 30:
        panels.append({"partType": "gas", "partField": "sb_O--8-16.0067A", "valMinMax": [-15.0, -10.0]})
    if conf == 31:
        panels.append({"partType": "gas", "partField": "sb_Lyman-alpha", "valMinMax": [-13.0, -8.0]})
    if conf == 32:
        panels.append({"partType": "gas", "partField": "sb_O--6-1031.91A_ster", "valMinMax": [-4.0, 2.0]})
    if conf == 33:
        panels.append({"partType": "gas", "partField": "ionmassratio_O6_O8", "valMinMax": [-2.0, 3.0]})
    if conf == 34:
        panels.append({"partType": "gas", "partField": "cellsize_kpc", "valMinMax": [-0.5, 1.5], "ctName": "magma"})
    if conf == 35:
        panels.append({"partType": "gas", "partField": "sz_yparam", "valMinMax": [-11.3, -6.5]})

    # testing equirectangular projections:
    if conf == 36:
        panels.append({"partType": "gas", "partField": "coldens_msun_ster", "valMinMax": gasFullMM})

    # thin slices may need different optimal bounds:
    if thinSlice:
        if conf == 0:
            panels[0]["valMinMax"] = [2.0, 5.0]  # gas coldens_msunkpc2
        if conf == 1:
            panels[0]["valMinMax"] = [2.6, 6.6]  # dm coldens_msunkpc2
        if conf == 5:
            panels[0]["valMinMax"] = [-9.0, 0.0]
            panels[0]["plawScale"] = 0.6  # gas bmag_uG
        if conf == 7:
            panels[0]["valMinMax"] = [3.3, 7.3]
            panels[0]["plawScale"] = 1.8  # gas temp
        if conf == 11:
            panels[0]["valMinMax"] = [28.5, 37.0]  # gas xray_lum
        if conf == 12:
            panels[0]["valMinMax"] = [0, 8]
            panels[0]["plawScale"] = 1.6  # gas shocks_machnum

    # 16:9 remappings may need different optimal bounds: (i.e. even thinner slices)
    if remap:
        if conf == 0:
            panels[0]["valMinMax"] = [3.3, 6.3]  # gas coldens_msunkpc2
        if conf == 1:
            panels[0]["valMinMax"] = [4.1, 8.0]  # dm coldens_msunkpc2
        if conf == 2:
            panels[0]["valMinMax"] = [0.0, 6.4]  # stars coldens_msunkpc2
        if conf == 5:
            panels[0]["valMinMax"] = [-7.0, -1.0]  # gas bmag_ug
        if conf == 6:
            panels[0]["valMinMax"] = [-4.0, -0.2]  # gas Z_solar
        if conf == 7:
            panels[0]["valMinMax"] = [3.5, 6.7]  # gas temp
        if conf == 11:
            panels[0]["valMinMax"] = [26, 36]  # gas xray_lum
        if conf == 13:
            panels[0]["valMinMax"] = [32, 36.5]  # gas shocks_dedt
        if conf == 14:
            panels[0]["valMinMax"] = [0, 600]  # [0, 700] # gas velmag

    return panels, centerHaloID, nSlicesTot, curSlice


def TNG_mainImages(res, conf=0, variant=None, thinSlice=False):
    """Create the FoF[0/1]-centered slices to be used for main presentation of the box."""
    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice)

    run = "tng"  #'eagle' #'millennium'
    redshift = 0.0
    nPixels = 2000  # 800, 2000, 8000
    axes = [0, 1]  # x,y
    labelZ = False  # True
    labelScale = False  # True
    labelSim = False  # True
    plotHalos = False  # 100
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # slice centering
    sliceStr = ""

    if centerHaloID is not None:
        relCenPos = None
        sliceFac = 1.0 / nSlicesTot
        sliceStr = "_fof-%d_%dof%d" % (centerHaloID, curSlice, nSlicesTot)

        # for curSlice in range(nSlicesTot):
        absCenPos = sP.groupCatSingle(haloID=centerHaloID)["GroupPos"]
        absCenPos[3 - axes[0] - axes[1]] += curSlice * sliceFac * sP.boxSize

    if thinSlice:
        # do very thin 100 kpc 'slice' instead
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize
        sliceStr = "_thinSlice"

    # render config (global)
    mStr = "" if method == "sphMap" else "_" + method
    mStr = mStr if "method" not in panels[0] else "_" + panels[0]["method"]

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels if isinstance(nPixels, list) else [nPixels, nPixels]
        colorbars = False

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d%s%s.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            sliceStr,
            mStr,
        )

    renderBox(panels, plotConfig, locals())


def singleFullBox(conf=0):
    """Create a full box view."""
    sP = simParams(run="tng100-3", redshift=0.0)

    panels, _, _, _ = _TNGboxFieldConfig(res=None, conf=conf, thinSlice=False)

    nPixels = 200
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = True
    labelSim = False
    plotHalos = False
    method = "voronoi_proj_global"  # sphMap, sphMap_minIP, sphMap_maxIP
    axesUnits = "mpc"
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)
    sliceFac = 0.5

    class plotConfig:
        plotStyle = "open"  # open, edged
        colorbars = True

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
        )

    renderBox(panels, plotConfig, locals())


def TNG_remapImages(res, redshift=0.0, conf=0, variant=None, phase=None, colorbars=False):
    """Create the full-box (full volume) remapped images."""
    panels, _, _, _ = _TNGboxFieldConfig(res, conf, thinSlice=False, remap=True)

    run = "tng"
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False  # True
    labelSim = False
    plotHalos = False  # 100 # False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)

    if 0:
        nPixels = [7680, 4280]  # 1080p x4
        remapRatio = [5.0, 2.7857, 0.0718]  # about 16:9 aspect, 7% depth
    if 0:
        nPixels = [2000, 2000]
        remapRatio = [2.44, 2.44, 0.168]  # square, 17% depth
        # remapRatio = [5.0, 5.0, 0.04] # square, 4% depth
    if 0:
        nPixels = [40167, 2160]  # for 4K and 5120x1440 movies
        if phase is not None:
            splitphase = [phase, 2]  # cannot render >30k side-length PNGs, split them
        # nPixels = [22310, 1200] # ~300 dpi for printing
        remapRatio = [10.488, 0.564, 0.169]  # 20x longer than tall
    if 1:
        nPixels = [28153, 4320]
        remapRatio = [6.1644, 0.9459, 0.1715]
    if colorbars:
        # render pdf for colorbars
        nPixels = [1200, 1200]
        res = 270

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # render config (global)
    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = False

    if not colorbars:
        plotConfig.saveFilename = "./boxImage_%s_z%.1f_%s-%s_axes%d%d_ratio-%g-%g-%g.png" % (
            sP.simName,
            redshift,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            remapRatio[0],
            remapRatio[1],
            remapRatio[2],
        )

    if colorbars:
        plotConfig.plotStyle = "edged_black"
        plotConfig.colorbars = True
        plotConfig.saveFilename = "./colorbar_%s_%s.pdf" % (panels[0]["partType"], panels[0]["partField"])

    renderBox(panels, plotConfig, locals())


def fullBox360(res, conf=34, variant=None, snap=None):
    """Create full box 180 or 360 degree images in various projections."""
    panels, _, _, _ = _TNGboxFieldConfig(res, conf, thinSlice=False)

    run = "tng"
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)

    if 0:
        projType = "equirectangular"  # 'azimuthalequidistant'
        projParams = {"fov": 360.0}
        nPixels = [4000, 2000]
    if 1:
        projType = "azimuthalequidistant"
        projParams = {"fov": 180.0}
        nPixels = [2000, 2000]

    axesUnits = "rad_pi"

    sP = simParams(res=res, run=run, snap=snap, variant=variant)
    redshift = 0.0 if snap is None else sP.redshift

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels if isinstance(nPixels, list) else [nPixels, nPixels]
        colorbars = False

        saveFilename = "./boxImage_%s_%s-%s_%s_%s.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            projType,
            sP.snap,
        )

    renderBox(panels, plotConfig, locals())


def smuggleMWAndromedaMergerDome(snap=100):
    """Create 180 degree dome renders for the smuggle Andromeda-Milky Way merger sim."""
    sim_path = "/u/dnelson/data/sims.other/local_group_smuggle/1e3_100IGM/"

    sP = simParams(sim_path, snap=snap)

    # config
    axes = [0, 1]  # x,y
    labelZ = False
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)

    # projection
    projType = "azimuthalequidistant"
    projParams = {"fov": 180.0}
    nPixels = [2000, 2000]
    axesUnits = "rad_pi"

    # view direction is hard-coded, do a 90 deg rotation to center system
    relCenPos = None
    absCenPos = [0.55 * sP.boxSize, 0.45 * sP.boxSize, 0.45 * sP.boxSize]
    rotMatrix = rotationMatrixFromVec([1, 0, 0])
    rotCenter = [sP.boxSize / 2, sP.boxSize / 2, sP.boxSize / 2]

    # panel
    # panels = [{'partType':'gas','partField':'coldens_msun_ster','valMinMax':[5.0,15.0]}]
    panels = [{"partType": "gas", "partField": "coldens_msun_ster", "valMinMax": [1.0, 6.0]}]
    # panels = [{'partType':'gas','partField':'temp','valMinMax':[4.5,6.0]}]
    # panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,10.0]}] # normal box render

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels if isinstance(nPixels, list) else [nPixels, nPixels]
        colorbars = False

        saveFilename = "./boxImage_%s_%s-%s_%s.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            sP.snap,
        )

    renderBox(panels, plotConfig, locals())


def dragonflyHalpha(res=1820):
    """Create the FoF[0/1]-centered slices to be used for main presentation of the box."""
    conf = 32  # H-alpha SB (sf0 TEMP)
    # conf = 31 # Lyman alpha/OVIII test

    if res in [128, 256, 512]:  # testing boxes
        variant = "0000"
        redshift = 0.0
        nPixels = 2000
    else:
        variant = None
        redshift = 0.1
        nPixels = 8000

    run = "tng"
    axes = [0, 1]  # x,y
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP
    labelZ = False
    labelSim = False
    axesUnits = "deg"
    labelScale = True

    # get field config, and adjust colormap bounds for z=0 SB
    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice=False)
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # do very thin 3nm wavelength 'slice' instead (E=hc/lambda) (nu*lambda=c) (E=h*nu)
    thinStr = ""
    if 0:
        deltaNm = 3.0
        restNm = 653.3  # h-alpha
        dEfrac = restNm / (restNm - deltaNm) - 1.0
        dL = dEfrac * sP.units.c_km_s * (1 + sP.redshift) / sP.units.H_z  # kpc
        sliceWidth = sP.units.physicalKpcToCodeLength(dL)
        print("Using [thin] slice width: %.1f code units" % sliceWidth)
        sliceFac = sliceWidth / sP.boxSize
        thinStr = "thin_"

    # info:
    pxSizeCode = sP.boxSize / nPixels
    pxSizeArcsec = sP.units.codeLengthToAngularSize(pxSizeCode, arcsec=True)
    print("pixel size: %f arcsec" % pxSizeArcsec)

    class plotConfig:
        plotStyle = "open"  # open, edged
        rasterPx = 1500
        colorbars = True
        saveFilename = "./boxImage_%s_z%.1f_%s%s-%s_axes%d%d.png" % (
            sP.simName,
            sP.redshift,
            thinStr,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
        )

    # class plotConfig:
    #    plotStyle  = 'edged' # open, edged
    #    rasterPx   = nPixels
    #    colorbars  = False
    #    saveFilename = './boxImage_%s_z%.1f_%sfull_%s-%s_axes%d%d.png' % \
    #      (sP.simName,sP.redshift,thinStr,panels[0]['partType'],panels[0]['partField'],axes[0],axes[1])

    renderBox(panels, plotConfig, locals())


def TNG_colorFlagshipBoxImage(part=0):
    """Create the parts of the fullbox demonstration image for the galaxy colors L75/L205 flagship paper."""
    panels = []

    run = "tng"
    redshift = 0.0
    nPixels = 2000
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = True
    labelSim = False
    plotHalos = False

    # parts 0,1,2 = L205, parts 3,4 = L75
    if part in [0, 1, 2]:
        res = 2500
    if part in [3, 4]:
        res = 1820

    sP = simParams(res=res, run=run, redshift=redshift)

    dmMM, gasMM, starsMM, gasFullMM, centerHaloID, nSlicesTot, curSlice = _TNGboxSliceConfig(res)
    sliceFac = 1.0 / nSlicesTot

    if part == 0:  # part 0: L205 gas dens
        plotHalos = 50
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMM})

    if part == 1:  # part 1: L205 Bmag thinSlice
        panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-9.0, -1.0]})  # [-9.0,0.5]
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize

    if part == 2:  # part 2: L205 gas temp
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.3, 7.2]})

    if part == 3:  # part 3: L75 gas dens
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gasMM})

    if part == 4:  # part 4: L75 dm dens
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": dmMM})

    # slice centering
    relCenPos = None

    sliceStr = "_fof-%d_%dof%d" % (centerHaloID, curSlice, nSlicesTot)
    absCenPos = sP.groupCatSingle(haloID=centerHaloID)["GroupPos"]
    absCenPos[3 - axes[0] - axes[1]] += curSlice * sliceFac * sP.boxSize

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = 2000
        colorbars = True

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d%s.pdf" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            sliceStr,
        )

    renderBox(panels, plotConfig, locals())


def TNG_oxygenPaperImages(part=0):
    """Create the parts of the fullbox demonstration image for the TNG oxygen paper."""
    panels = []

    run = "tng"
    redshift = 0.0
    res = 1820
    nPixels = 6000  # 2000
    axes = [0, 1]  # x,y
    labelZ = True
    labelScale = True
    labelSim = False
    plotHalos = False
    axesUnits = "mpc"

    sP = simParams(res=res, run=run, redshift=redshift)

    if part in [0, 3, 4]:
        # part 0: TNG100 full box OVII (15 Mpc depth)
        _, _, _, _, centerHaloID, _, _ = _TNGboxSliceConfig(res)
        sliceFac = sP.units.physicalKpcToCodeLength(15000.0) / sP.boxSize
        curSlice = 0
        nSlicesTot = 7

        # slice centering
        relCenPos = None

        saveStr = "_fof-%d_%dof%d" % (centerHaloID, curSlice, nSlicesTot)
        absCenPos = sP.groupCatSingle(haloID=centerHaloID)["GroupPos"]
        absCenPos[3 - axes[0] - axes[1]] += curSlice * sliceFac * sP.boxSize

        plotHalos = 100
        if part == 0:
            panels.append({"partType": "gas", "partField": "O VII", "valMinMax": [11, 16]})  # O VII
        if part == 3:
            panels.append({"partType": "gas", "partField": "ionmassratio_O6_O8", "valMinMax": [-2.0, 2.0]})
        if part == 4:
            panels.append({"partType": "gas", "partField": "sb_OVIII", "valMinMax": [-22, -10], "ctName": "magma_gray"})

    if part == 1:
        # part 1: cluster halo scale OVIII (halo #22)
        haloID = 22
        shID = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

        rVirFracs = [1.0]
        size = 3.5
        sizeType = "rVirial"
        nPixels = [nPixels, nPixels]  # [int(nPixels/2),int(nPixels/2)]
        relCoords = True
        axes = [1, 2]
        plotSubhalos = 50
        method = "sphMap_global"

        saveStr = "_fof-%d_shid-%d_size-%.1f_%s" % (haloID, shID, size, sizeType)

        panels.append({"partType": "gas", "partField": "O VIII", "subhaloInd": shID, "valMinMax": [13.5, 15.8]})

    if part == 2:
        # part 2: galaxy halo scale OVI
        haloID = 22
        halo = sP.groupCatSingle(haloID=haloID)

        nPixels = [int(nPixels / 2), int(nPixels / 2)]
        relCoords = True
        axes = [1, 2]
        method = "sphMap_global"

        # (A) pick a satellite of this cluster
        haloSubInds = np.arange(halo["GroupFirstSub"], halo["GroupFirstSub"] + halo["GroupNsubs"])
        ##subMstar = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])[haloSubInds]
        ##subRadRvir = sP.groupCat(fieldsSubhalos=['rdist_rvir'])[haloSubInds]
        # subMstar:   [ 11.852, 10.90, 10.69, 10.31, 10.41, 10.74, 10.65, 10.41, 9.75, 10.34]
        # subRadRvir: [ 0.0,    1.53,  2.16,  1.91,  0.73,  0.17,  0.67,  1.23,  1.67,  0.79]
        # shID = haloSubInds[1]

        # rVirFracs = [1.0,2.0,5.0]
        # fracsType = 'rHalfMass'
        # size      = 10.0
        # sizeType  = 'rHalfMass'

        # (B) pick a central (optionally nearby to this cluster?)
        # subPos = sP.groupCat(fieldsSubhalos=['SubhaloPos'])
        # subHaloMass = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])
        # dist = sP.periodicDists( subPos[halo['GroupFirstSub'],:], subPos )
        # w = np.where( (subHaloMass > 11.9) & (subHaloMass < 12.6) & (dist < (size+1.0)*halo['Group_R_Crit200']) )
        # w = np.where( (subHaloMass > 12.4) & (subHaloMass < 12.5) )
        shID = [376544, 378745, 380751, 383723, 389704, 389917, 393336, 394241, 395882, 396851, 397568, 400694][9]

        sh = sP.groupCatSingle(subhaloID=shID)
        halo = sP.groupCatSingle(haloID=sh["SubhaloGrNr"])
        mstar = sP.units.codeMassToLogMsun(sh["SubhaloMassType"][4])
        mhalo = sP.units.codeMassToLogMsun(halo["Group_M_Crit200"])
        print(mstar, mhalo)

        rVirFracs = [1.0]
        size = 3.0
        sizeType = "rVirial"
        axes = [0, 2]

        saveStr = "_shid-%d_size-%.1f_%s" % (shID, size, sizeType)

        panels.append({"partType": "gas", "partField": "O VI", "subhaloInd": shID, "valMinMax": [12.8, 15.4]})

    class plotConfig:
        plotStyle = "edged"  # open
        rasterPx = 1800 if part in [0, 3, 4] else 900
        # rasterPx   = 6000
        colorbars = True

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d%s.pdf" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            saveStr,
        )

    if part in [0, 3, 4]:
        renderBox(panels, plotConfig, locals())
    else:
        renderSingleHalo(panels, plotConfig, locals())


def TNG_explorerImageSegments(conf=0, taskNum=0, retInfo=False):
    """Construct image segments which are then split into the pyramids for the TNG explorer 2d."""
    res = 2500
    nPixels = 16384  # 2048x4 = 8k (testing), 16384x8 = 131072 (target final size)
    nPanels = 64  # 8x8
    hsmlFac = 2.5  # use for all: gas, dm, stars

    run = "tng"
    redshift = 0.0
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP

    # field
    sP = simParams(res=res, run=run, redshift=redshift)

    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice=False)

    # slice positioning
    relCenPos = None
    sliceFac = 1.0 / nSlicesTot

    absCenPos = sP.groupCatSingle(haloID=centerHaloID)["GroupPos"]
    absCenPos[3 - axes[0] - axes[1]] += curSlice * sliceFac * sP.boxSize

    # panel positioning
    zoomFac = 1.0 / np.sqrt(nPanels)

    panelSize = sP.boxSize / np.sqrt(nPanels)
    panelRow = int(np.floor(taskNum / np.sqrt(nPanels)))
    panelCol = int(taskNum % np.sqrt(nPanels))

    absCenPos[axes[0]] = absCenPos[axes[0]] - sP.boxSize / 2 + panelSize / 2 + panelSize * panelCol
    absCenPos[axes[1]] = absCenPos[axes[1]] - sP.boxSize / 2 + panelSize / 2 + panelSize * panelRow

    print(taskNum, panelRow, panelCol, absCenPos[0], absCenPos[1])

    # render config (global)
    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = False

        saveFilename = "./boxImageExplorer_%s_%s-%s_%d.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            taskNum,
        )

    if retInfo:
        return renderBox(panels, plotConfig, locals(), retInfo=retInfo)

    renderBox(panels, plotConfig, locals())


def oneBox_LIC(res, conf=0, variant=None, thinSlice=False):
    """Testing whole-box LIC."""
    panels, centerHaloID, nSlicesTot, curSlice = _TNGboxFieldConfig(res, conf, thinSlice)

    run = "tng"
    redshift = 0.0
    nPixels = 1000  # 800, 2000, 8000
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP

    # LIC return is [0,1], now inherit colormap from original conf field
    # panels[0]['valMinMax'] = [0.0, 1.0]
    ##panels[0]['partField'] = 'vel_x'
    ##panels[0]['valMinMax'] = [-200,200]

    # panels[0]['valMinMax'][0] += 1.0 # account for thinner slice
    licMethod = 2  # None, 1, 2
    licSliceDepth = 5000.0
    sliceFac = 0.2  # to match to licSliceDepth
    licPartType = "gas"
    licPartField = "bfield"
    licPixelFrac = 0.2

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    # slice centering
    sliceStr = ""

    if thinSlice:
        # do very thin 100 kpc 'slice' instead
        sliceWidth = sP.units.physicalKpcToCodeLength(100.0)
        sliceFac = sliceWidth / sP.boxSize
        sliceStr = "_thinSlice"

    # render config (global)
    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels
        colorbars = False

        saveFilename = "./boxImageLIC_%s_%s-%s_axes%d%d%s.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            sliceStr,
        )

    renderBox(panels, plotConfig, locals())


def oneBox_multiQuantCollage(variant=0000):
    """Make a collage for a single run, of every quantity we can.

    (now 15=5x3 panels, 1.67 aspect ratio vs 1.78 for 1920x1080 or 1.6 for 1920x1200).
    """
    panels = []
    panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.3, 7.3]})
    panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})
    panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 6.0]})
    panels.append({"partType": "gas", "partField": "HI_segmented", "valMinMax": [13.5, 21.5]})
    panels.append({"partType": "gas", "partField": "pressure_ratio", "valMinMax": [-8, 1], "cmapCenVal": -3.0})
    panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-3.5, 1.0]})
    panels.append({"partType": "gas", "partField": "Z_solar", "valMinMax": [-2.0, -0.2]})
    panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.3, 7.2]})
    panels.append({"partType": "gas", "partField": "SN_IaII_ratio_Fe", "valMinMax": [0.0, 2.6]})
    panels.append({"partType": "gas", "partField": "SN_IaII_ratio_metals", "valMinMax": [-1.0, 2.5]})
    panels.append({"partType": "gas", "partField": "SN_Ia_AGB_ratio_metals", "valMinMax": [-0.48, 0.06]})
    panels.append({"partType": "gas", "partField": "xray_lum", "valMinMax": [29, 37.5]})
    panels.append({"partType": "gas", "partField": "shocks_machnum", "valMinMax": [0, 4]})
    panels.append({"partType": "gas", "partField": "shocks_dedt", "valMinMax": [32, 38]})
    panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [100, 500]})

    panels[4]["labelScale"] = True
    panels[-1]["labelSim"] = True

    run = "tng"
    redshift = 2.0
    res = 1024
    # variant    = 0000

    nPixels = 800
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False

    # render config (global)
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 800
        colorbars = False

        saveFilename = "./boxCollage_%s_z=%.1f_%dpanels_axes%d%d.png" % (
            sP.simName,
            sP.redshift,
            len(panels),
            axes[0],
            axes[1],
        )

    renderBox(panels, plotConfig, locals())


def multiBoxComparison():
    """Compare two boxes, side-by-side, one quantity."""
    redshift = 0.0
    sPs = []

    sPs.append(simParams(res=512, run="tng", redshift=redshift, variant="0000"))
    sPs.append(simParams(res=512, run="tng", redshift=redshift, variant="0010"))

    partType = "gas"
    nPixels = 1000
    axes = [0, 1]
    plotHalos = False
    labelSim = True
    labelScale = "physical"

    if 0:
        partField = "coldens_msunkpc2"
        valMinMax = [4.5, 7.0]
    if 1:
        partField = "temp"
        valMinMax = [5.0, 6.8]  # [4.5, 6.5]
    if 0:
        partField = "metal_solar"
        valMinMax = [-2.0, -0.5]

    # zoom in to FoF 0? show out to 2rvir
    if 0:
        halo0 = simParams(run="tng50-1", redshift=redshift).halo(0)
        relCenPos = None
        absCenPos = halo0["GroupPos"]
        zoomFac = (halo0["Group_R_Crit200"] * 4) / sPs[0].boxSize
        sliceFac = (halo0["Group_R_Crit200"] * 4) / sPs[0].boxSize
        valMinMax = [5.0, 7.8]

    # make panels
    panels = []
    for sP in sPs:
        panels.append({"run": sP.run, "redshift": sP.redshift, "res": sP.res, "variant": sP.variant})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = True

        saveFilename = "./boxComparison_%s.pdf" % partField

    renderBox(panels, plotConfig, locals())


def oneBox_redshiftEvoPanels(redshifts=(6.0, 2.0, 0.8)):
    """Create a linear series of redshift evolution panels for one simulation + one quantity."""
    run = "tng"
    res = 2160
    nPixels = 800
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = True
    labelSim = False
    plotHalos = False
    method = "sphMap"

    partType = "gas"
    partField = "coldens_msunkpc2"
    valMinMax = [4.3, 7.3]

    panels = []
    for redshift in redshifts:
        panels.append({"redshift": redshift})

    sP = simParams(res=res, run=run, redshift=redshift)

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels
        colorbars = True
        nRows = 1  # single row

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d_z=%s.pdf" % (
            sP.simName,
            partType,
            partField,
            axes[0],
            axes[1],
            "_".join(["%.1f" % z for z in redshifts]),
        )

    renderBox(panels, plotConfig, locals())


def zoom_gasColDens_3res_or_3quant():
    """Diagnostic plot of gas column dens in the entire zoom box at z=2 (e.g. 3 res or 3 quant comp)."""
    panels = []

    # panels.append( {'res':10, 'partField':'coldens'} )
    # panels.append( {'res':10, 'partField':'coldens_msunkpc2'} )
    # panels.append( {'res':10, 'partField':'density'} )

    panels.append({"res": 9, "partField": "coldens_msunkpc2"})
    panels.append({"res": 10, "partField": "coldens_msunkpc2"})
    panels.append({"res": 11, "partField": "coldens_msunkpc2"})

    hInd = 7
    run = "zooms"
    redshift = 2.0
    partType = "gas"
    nPixels = 1400
    axes = [0, 1]  # x,y
    labelZ = True
    labelScale = True
    labelSim = True
    plotHalos = 20

    # render config (global)
    class plotConfig:
        plotStyle = "open_black"
        rasterPx = 1400
        colorbars = True

        Lstr = "-".join([str(p["res"]) for p in panels])
        saveFilename = savePathDefault + "%s_FullBoxGasColDens_h%dL%s.pdf" % (run, hInd, Lstr)

    renderBox(panels, plotConfig, locals())


def box_slices(curSlice=7, conf=0):
    """Create a series of slices moving through a box (no halo centering)."""
    run = "tng"
    res = 2500
    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False
    method = "sphMap"  # sphMap, sphMap_minIP, sphMap_maxIP

    if conf == 0:
        redshift = 9.0
        nPixels = 1200
    if conf == 1:
        redshift = 0.4
        nPixels = int(7.14 * 1200)

    panels, _, _, _ = _TNGboxFieldConfig(res, conf=1, thinSlice=False)

    sP = simParams(res=res, run=run, redshift=redshift)
    ###panels[0]['valMinMax'] += np.array([0.4,0.0])

    # slicing config
    nSlicesTot = 10  # ~30 mpc deep each
    sliceFac = 1.0 / nSlicesTot
    sliceStr = "slice_%d_of_%d" % (curSlice, nSlicesTot)

    relCenPos = None
    absCenPos = np.array([0.5, 0.5, 0.5]) * sP.boxSize
    absCenPos[1] = 125000.0  # center choice
    absCenPos[0] = 75000.0  # center choice
    absCenPos[3 - axes[0] - axes[1]] += curSlice * sliceFac * sP.boxSize

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels
        colorbars = False

        saveFilename = "./boxImage_%s_%s-%s_axes%d%d_%s.png" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            axes[0],
            axes[1],
            sliceStr,
        )

    renderBox(panels, plotConfig, locals())


def fire2_publicdata(fire_run="core_FIRE-2_runs/m12i_res7100/"):
    """Test full-box visualization of one of the publicly released FIRE-2 Milky Way runs."""
    basePath = "/virgotng/mpia/FIRE-2/"

    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = True
    labelSim = False
    plotHalos = False
    method = "sphMap"
    nPixels = 1200

    partType = "gas"
    partField = "coldens_msunkpc2"
    valMinMax = [3.2, 6.8]

    # note: only high-res gas exists (in the center of the 60 Mpc/h box)
    zoomFac = 0.1

    # one panel
    panels = [{}]

    # redshift
    redshift = 0.0

    sP = simParams(basePath + fire_run, redshift=redshift)

    class plotConfig:
        plotStyle = "open"  # open, edged
        rasterPx = nPixels
        colorbars = True

        saveFilename = "./boxImage_%s_z%0.1f_%s-%s.png" % (sP.simName, sP.redshift, partType, partField)

    renderBox(panels, plotConfig, locals())
