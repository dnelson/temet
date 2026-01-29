"""
Render specific fullbox (movie frame) visualizations.
"""

import numpy as np

from ..cosmo.time_evo import subhalo_subbox_overlap
from ..cosmo.util import subboxSubhaloCat
from ..plot.util import setAxisColors
from ..util.helper import logZeroNaN
from ..util.rotation import rotationMatrixFromAngleDirection
from ..util.simParams import simParams
from ..vis.box import renderBox, renderBoxFrames
from ..vis.common import savePathBase


def subbox_4x2_movie(curTask=0, numTasks=1):
    """Render a movie comparing several quantities of a single subbox (4x2 panels, 4K)."""
    panels = []

    panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.2, 7.2], "labelScale": True})
    panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})
    panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [2.8, 8.2]})
    panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-3.0, 1.0]})
    panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.4, 7.6]})
    panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-2.0, 0.4]})
    panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [100, 1000]})
    panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [10, 16], "labelZ": True})

    run = "tng"  #'illustris'
    variant = "subbox0"
    res = 1820
    method = "sphMap"
    nPixels = 960
    axes = [0, 1]  # x,y

    class plotConfig:
        savePath = savePathBase + "%s_sb0/" % run
        plotStyle = "edged_black"
        rasterPx = 960
        colorbars = True

        # movie config
        minZ = 0.0
        maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700  # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_2x1_movie(curTask=0, numTasks=1):
    """Render a movie comparing two quantities of a single subbox (2x1 panels, 4K)."""
    panels = []

    # panels.append( {'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.2,7.5], 'labelScale':True} )
    # panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[2.8,8.4], 'labelZ':True} )

    panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.7, 8.0]})  # 5.8,7.4
    panels.append({"partType": "stars", "partField": "stellarComp", "labelZ": False, "labelScale": "physical"})  # True

    run = "tng"
    variant = "subbox0"  #'subbox0'
    res = 2500  # 1820
    method = "sphMap"
    nPixels = [1920, 1080]
    axes = [1, 2]  # x,y

    class plotConfig:
        savePath = savePathBase + "%s_%s/" % (run, variant)
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False  # True

        # movie config
        minZ = 0.0
        maxZ = 10.0  # tng subboxes start at a=0.02
        maxNSnaps = 2100

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_movie_tng300fof0(curTask=0, numTasks=1):
    """Render a movie of the TNG300 most massive cluster (1 field, 4K)."""
    # panels = [{'partType':'gas',   'partField':'coldens_msunkpc2', 'valMinMax':[4.7,8.0]}]
    panels = [{"partType": "dm", "partField": "coldens_msunckpc2", "valMinMax": [5.8, 7.8]}]
    # panels = [{'partType':'stars', 'partField':'coldens_msunckpc2', 'valMinMax':[2.8,7.6]}]
    # panels = [{'partType':'gas',   'partField':'metal_solar', 'valMinMax':[-3.0,-0.4]}]

    run = "tng"
    variant = "subbox0"
    res = 2500
    method = "sphMap"
    nPixels = [3840, 2160]
    axes = [0, 2]  # [1,2] for original TNG300, [0,2] for TNG-Cluster
    labelZ = True
    labelScale = "physical"

    class plotConfig:
        savePath = savePathBase + "%s_%s_%s_%s/" % (run, variant, panels[0]["partType"], panels[0]["partField"])
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False  # True

        # movie config
        # note: dloga doubles for (final) frames 555-583 inclusive, fix in post with 0.5x speed
        minZ = 0.0
        maxZ = 30.0  # 10 for TNG300, 20 for TNG-C # tng subboxes start at a=0.02
        maxNSnaps = 2000

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_movie_tng300fof0_persrot(curTask=0, numTasks=1):
    """Render a movie of the TNG300 most massive cluster (1 field, 4K). Perspective camera with rotation."""
    panels = [{"partType": "gas", "partField": "coldens_msunckpc2", "valMinMax": [4.7, 8.0]}]
    # panels = [{'partType':'gas', 'partField':'bmag_uG', 'valMinMax':[-3.0,1.2], 'ctName':'afmhot'}]
    # panels = [{'partType':'dm',    'partField':'coldens_msunckpc2', 'valMinMax':[5.8,7.8]}]
    # panels = [{'partType':'gas', 'partField':'xray_lum_0.5-2.0kev', 'valMinMax':[34.0,39.0]}]

    run = "tng"
    variant = "subbox0"
    res = 2500
    method = "sphMap"
    nPixels = [3840, 2160]
    axes = [0, 1]  # [1,2] for original TNG300, [0,2] for TNG-Cluster
    labelZ = True
    labelAge = True
    labelScale = "physical"  # semi-accurate in perspective

    # rotation
    rotSequence = [360 * 4, [0.1, 1.0, 0.4]]

    # perspective projection config
    projType = "perspective"
    projParams = {}
    projParams["n"] = 11000.0  # sets fov = 110 deg
    projParams["f"] = 15000.0
    projParams["l"] = -7500.0 * (16 / 9)
    projParams["r"] = 7500.0 * (16 / 9)
    projParams["b"] = -7500.0
    projParams["t"] = 7500.0

    class plotConfig:
        savePath = savePathBase + "%s_%s_persrot_%s_%s/" % (run, variant, panels[0]["partType"], panels[0]["partField"])
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False

        # movie config
        minZ = 0.0
        maxZ = 30.0  # 10 for TNG300, 20 for TNG-C # tng subboxes start at a=0.02
        maxNSnaps = 2000

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_movie_tng300fof0_tracking(conf=0, clean=False):
    """Use the subbox tracking catalog to create a movie highlighting the evolution of the TNG300 FoF0."""
    # selection
    sP = simParams(run="tng300-1", variant="subbox0", redshift=0.0)

    subhaloID = 0  # halo 0, snap 99

    method = "sphMap"
    axes = [0, 1]  # x,y
    labelScale = "physical"  # semi-accurate in perspective
    labelZ = True
    plotHalos = False
    nPixels = [3840, 2160]

    class plotConfig:
        saveFilename = ""  # set later
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False  # True
        colorbarOverlay = True

    # render config
    if conf == 0:
        panels = [{"partType": "gas", "partField": "coldens_msunckpc2", "valMinMax": [5.0, 8.0]}]  # time constant
    if conf == 1:
        panels = [{"partType": "gas", "partField": "temp_sfcold", "valMinMax": [6.0, 8.2]}]  # time constant
    if conf == 2:
        panels = [{"partType": "gas", "partField": "coldens_msunckpc2", "valMinMax": [0.0, 0.0]}]  # time variable
    if conf == 3:
        panels = [{"partType": "gas", "partField": "temp_sfcold", "valMinMax": [0.0, 0.0]}]  # time variable
    if conf == 4:
        panels = [
            {"partType": "stars", "partField": "coldens_msunckpc2", "valMinMax": [4.8, 8.0]}
        ]  # time constant (z=0 rot)
        clean = True
    if conf == 5:
        panels = [{"partType": "gas", "partField": "shocks_dedt", "valMinMax": [35.0, 40.0]}]  # time constant (z=0 rot)
        weightField = "dens"
        method = "histo"  # or e.g. hsmlFac = 0.01 for point-like SPH
        clean = True

    # (time evolving) box size config
    boxSizeLg_zi = 3000  # ckpc/h at z=30
    boxSizeLg_z0 = 12000  # ckpc/h at z=0
    aspect = float(nPixels[0]) / nPixels[1]

    # (time evolving) rotation config
    numFramesPerRot = 360 * 4
    rotDirVec = [0.0, 1.0, 0.0]  # horizontal seeming spin

    # frame config
    maxNSnaps = 2000

    sbSnapNums = sP.validSnapList(maxNum=maxNSnaps, minRedshift=0.0, maxRedshift=30.0)

    # pre-load subbox cat, get time evolving positions
    sP_par = simParams(run=sP.run, res=sP.res, redshift=sP.redshift)
    cat = subboxSubhaloCat(sP_par, sbNum=sP.subbox)

    assert cat["EverInSubboxFlag"][subhaloID]

    w = np.where(cat["SubhaloIDs"] == subhaloID)[0]
    assert len(w) == 1

    subhalo_pos = cat["SubhaloPos"][w[0], :, :]  # [nSubboxSnaps,3]
    snap_start = cat["SubhaloMinSBSnap"][w[0]]
    snap_stop = cat["SubhaloMaxSBSnap"][w[0]]

    assert sbSnapNums.min() >= snap_start and sbSnapNums.max() <= snap_stop

    # add frames for a final (time fixed) rotation at z=0
    sbSnapNum_final = sbSnapNums.max()

    sbSnapNums = np.hstack((sbSnapNums, np.arange(sbSnapNum_final + 1, sbSnapNum_final + 360 * 3)))

    # extra property calculations
    cat_z = 1 / cat["SubboxScaleFac"] - 1
    mpb = sP_par.loadMPB(subhaloID)
    mpb_z = sP_par.snapNumToRedshift(mpb["SnapNum"])

    haloMass = np.interp(cat_z, mpb_z, mpb["Group_M_Crit200"])
    haloR200 = np.interp(cat_z, mpb_z, mpb["Group_R_Crit200"])
    haloR500 = np.interp(cat_z, mpb_z, mpb["Group_R_Crit500"])

    # normal render
    for frameNum, snap in enumerate(sbSnapNums):
        if conf == 0 and snap < sbSnapNum_final - 300:
            continue  # only used for ending
        if conf in [4, 5] and snap < sbSnapNum_final:
            continue  # only used for z=0 rotations

        # set snapshot for render
        if snap > sbSnapNum_final:
            snap = sbSnapNum_final  # rotation sequence at fixed z=0
        sP.setSnap(snap)

        # set image size and center at this time
        boxSize_z = np.interp(snap, [sbSnapNums.min(), sbSnapNums.max()], [boxSizeLg_zi, boxSizeLg_z0])
        boxSizeImg = [int(boxSize_z * aspect), boxSize_z, boxSize_z]

        boxCenter = subhalo_pos[snap, :]

        # set color bounds at this time
        if conf == 2:
            dens_min = 3.0 + 4.5 * sP.scalefac  # reaches 5.0 at z=2
            dens_max = 6.0 + 4.5 * sP.scalefac  # reaches 7.0 at z=2
            panels[0]["valMinMax"] = [dens_min, dens_max]
        if conf == 3:
            temp_min = 4.4 + 1.3 * sP.scalefac  # 5.0 at z=2, 5.7 by z=0
            temp_max = 7.1 + 1.3 * sP.scalefac  # 7.5 at z=2, 8.2 by z=0
            panels[0]["valMinMax"] = [temp_min, temp_max]

        # perspective projection config
        projType = "perspective"
        projParams = {}
        projParams["n"] = 5000.0  # 3000.0
        projParams["f"] = 15000.0
        projParams["l"] = -boxSize_z * aspect
        projParams["r"] = boxSize_z * aspect
        projParams["b"] = -boxSize_z
        projParams["t"] = boxSize_z

        # rotation
        rotAngleDeg = 360.0 * (frameNum / numFramesPerRot)
        rotCenter = boxCenter
        rotMatrix = rotationMatrixFromAngleDirection(rotAngleDeg, rotDirVec)

        # add custom label of subbox time resolution galaxy properties
        if not clean:
            aperture_num = 0  # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h
            stellarMass = sP_par.units.codeMassToLogMsun(np.squeeze(cat["SubhaloStars_Mass"][w[0], aperture_num, snap]))
            SFR = np.squeeze(cat["SubhaloGas_SFR"][w[0], aperture_num, snap])
            M200c = sP_par.units.codeMassToLogMsun(haloMass[snap])
            R200c = sP.units.codeLengthToMpc(haloR200[snap])  # pMpc
            labelCustom = []
            labelCustom.append(r"$\rm{t_{age}}$ = %5.2f Gyr" % sP.tage)
            labelCustom.append(r"log $\rm{M_{200c}} = %.2f$ ($\rm{R_{200c} = %.2f \,Mpc}$)" % (M200c, R200c))
            labelCustom.append(
                r"log $\rm{M_{\star}} = %.2f$ (SFR = %.1f $\rm{M_\odot \,yr^{-1}}$)" % (stellarMass, SFR)
            )

            aperture_num = 0  # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h
            bhMass = sP.units.codeMassToLogMsun(np.squeeze(cat["SubhaloBH_Mass"][w[0], aperture_num, snap]))
            bhMdot = np.log10(
                sP.units.codeMassOverTimeToMsunPerYear(np.squeeze(cat["SubhaloBH_Mdot"][w[0], aperture_num, snap]))
            )
            labelCustom.append(
                r"log $\rm{M_{BH}} = %.2f$ (log $\rm{\dot{M}_{BH}}$ = %.1f $\rm{M_\odot \,yr^{-1}}$)" % (bhMass, bhMdot)
            )

        def func_post(parent_ax, panel, conf):
            """Custom post-render hook to draw (i) virial radii circles, and (ii) BH Mdot vs time plot."""
            import matplotlib.pyplot as plt
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes

            # config
            snap = panel["sP"].snapNum
            boxCenter = conf["boxCenter"]
            aperture_num = 0  # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h

            # draw BH Mdot vs time plot
            ax = inset_axes(parent_ax, width="20%", height="20%", loc=4, borderpad=4.5)  # lower right corner

            ax.set_xlabel("Redshift", size=30)
            ax.set_ylabel(r"$\rm{\dot{M}_{BH} \,[log \,M_\odot/yr}]$", size=30)
            ax.set_xlim([6.0, 0.0])
            ax.set_ylim([-4.2, 0.5])
            for tick in ax.get_xticklabels() + ax.get_yticklabels():
                tick.set_fontsize(24)

            ax.set_facecolor("None")
            setAxisColors(ax, "white")

            xx = 1 / cat["SubboxScaleFac"] - 1
            yy = logZeroNaN(np.squeeze(cat["SubhaloBH_Mdot"][w[0], aperture_num, :]))

            ax.plot(xx, yy, "-", lw=1.5, color="white", alpha=0.5)
            ax.plot(xx[snap], yy[snap], "o", ms=14, color="white")

            # draw r200c and r500c circles
            xy_cen = [boxCenter[0], boxCenter[1]]

            c200 = plt.Circle(xy_cen, haloR200[snap], color="#fff", linewidth=1.5, fill=False, alpha=0.4)
            c500 = plt.Circle(xy_cen, haloR500[snap], color="#fff", linewidth=1.5, fill=False, alpha=0.4)
            parent_ax.add_artist(c200)
            parent_ax.add_artist(c500)

        panels[0]["f_post"] = func_post if not clean else None

        extent = [
            boxCenter[0] - 0.5 * boxSizeImg[0],
            boxCenter[0] + 0.5 * boxSizeImg[0],
            boxCenter[1] - 0.5 * boxSizeImg[1],
            boxCenter[1] + 0.5 * boxSizeImg[1],
        ]

        # render
        plotConfig.saveFilename = savePathBase + "tng300_conf%s_tracking%s/frame_%04d.png" % (
            conf,
            "_clean" if clean else "",
            frameNum,
        )

        renderBox(panels, plotConfig, locals())


def subbox_movie_tng300fof0_4x2(curTask=0, numTasks=1):
    """Render a movie comparing several quantities of a single subbox (4x2 panels, 4K)."""
    panels = []

    # first row
    panels.append({"partType": "dm", "partField": "coldens_msunckpc2", "valMinMax": [5.0, 8.5], "labelScale": True})
    panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-3.0, 1.0]})
    panels.append({"partType": "gas", "partField": "temp", "valMinMax": [5.0, 7.8]})
    panels.append({"partType": "gas", "partField": "xray_lum_0.5-2.0kev", "valMinMax": [35.0, 39.0], "labelZ": True})

    # second row
    panels.append({"partType": "gas", "partField": "coldens_msunckpc2", "valMinMax": [4.2, 7.2]})
    panels.append({"partType": "stars", "partField": "coldens_msunckpc2", "valMinMax": [2.8, 8.0]})
    panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-3.0, -0.4]})
    panels.append({"partType": "gas", "partField": "p_sync_ska", "valMinMax": [14.0, 20.0]})

    run = "tng"
    variant = "subbox0"
    res = 2500
    method = "sphMap"
    nPixels = 960
    axes = [0, 2]

    class plotConfig:
        savePath = savePathBase + "%s_4x2/" % run
        plotStyle = "edged_black"
        rasterPx = 960
        colorbars = True
        colorbarsmall = True  # place field names inside colorbar
        fontsize = 24

        # movie config
        minZ = 0.0
        maxZ = 30.0
        maxNSnaps = 2000

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_movie_tng50(curTask=0, numTasks=1, conf="one", render8k=False):
    """Render a 4K movie of a single field from one subbox."""
    panels = []

    run = "tng"
    method = "sphMap"
    nPixels = [3840, 2160]
    axes = [0, 1]  # x,y
    res = 2160
    variant = "subbox2"

    labelScale = "physical"
    labelZ = True

    if conf == "one":
        # TNG50_sb2_gasvel_stars movie: gasvel
        saveStr = "gas_velmag"
        if res == 2160:
            mm = [50, 1100]
        if res == 2500:
            mm = [100, 2200]
        panels.append({"partType": "gas", "partField": "velmag", "valMinMax": mm})

    if conf == "two":
        # TNG50_sb2_gasvel_stars movie: temp (unused)
        saveStr = "gas_temp"
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.4, 7.6]})

    if conf == "three":
        # TNG50_sb2_gasvel_stars movie: stars
        saveStr = "stars"
        if res == 2160:
            mm = [2.8, 8.4]
        if res == 2500:
            mm = [2.6, 7.6]
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": mm})

    if conf == "four":
        # x-ray emission (0.5-2.0 keV SB [erg/s/kpc^2] based on APEC redshift-dependent tables)
        saveStr = "xray"
        panels.append({"partType": "gas", "partField": "xray_lum_0.5-2.0kev", "valMinMax": [30, 37]})

    if conf == "five":
        # baryon fraction (ayromlou+22 movie)
        saveStr = "fb"
        panels.append({"partType": "gas", "partField": "baryon_frac", "valMinMax": [0.0, 2.0]})
        # saveStr = 'fb_gridmethod'
        # panels.append( {'partType':'gas', 'partField':'coldens', 'valMinMax':[0.0,2.0], 'ctName':'seismic'} )

    if conf == "bmag":
        saveStr = "bmag"
        panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-3.5, 0.0]})

    if render8k:
        nPixels = [7680, 7680]
        labelScale = False
        labelZ = False
        plotHalos = False
        saveStr += "_8k"

    class plotConfig:
        savePath = savePathBase + "%s%s_%s/" % (res, variant, saveStr)
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False
        saveFilename = "out.png"

        # movie config
        minZ = 0.0
        maxZ = 50.0  # tng subboxes start at a=0.02
        maxNSnaps = 3168  # there are 867 snaps with excessively small spacing between a=0.33 and a=0.47 (1308-2344)
        # as a final config, filter out half: take Nsb_final-867/2 (currently: 3600-433+eps = 3168)

    # for TNG100 set 2.5 min max (150 sec * 30 fps), for TNG300 use all subboxes (only ~2500)
    if res in [1820, 2500]:
        plotConfig.maxNSnaps = 4500
        plotConfig.colorbars = True
        plotConfig.colorbarOverlay = True

    if 0:
        # render single z=0.0 frame for testing
        # sliceFac = 0.005 # thin slice (only for conf=='bmag' for DFG-calendar2022)
        redshift = 0.0
        renderBox(panels, plotConfig, locals())
        return

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_movie_tng_galaxyevo_frame(
    sbSnapNum=2687, gal="two", conf="one", frameNum=None, rotSeqFrameNum=None, rotSeqFrameNum2=None, cat=None
):
    """Use the subbox tracking catalog to create a movie highlighting the evolution of a single galaxy.

    If frameNum is not None, then use this for save filename instead of sbSnapNum.
    If rotSeqFrameNum is not None, then proceed to render rotation squence (at fixed time iff sbSnapNum is kept fixed).
    """
    if 0:
        # helper to make subhalo selection
        sP = simParams(res=2160, run="tng", snap=90)
        sbNum = 2
        gc = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log", "is_central", "SubhaloGrNr"])
        w = np.where(gc["mstar_30pkpc_log"] >= 10.2)  # & (gc['is_central']) )
        subInds = w[0]
        subbox_overlap = subhalo_subbox_overlap(sP, sbNum, subInds, verbose=True)
        assert 0  # todo: finish

    # set selection subhaloID at sP.snap
    if gal == "one":
        # first movie, Andromeda (sbSnaps 51 - 3600)
        sP_par = simParams(res=2160, run="tng", snap=99)
        sbNum = 0
        # subhaloID = 389836 # halo 296, snap 58
        # subhaloID = 440389 # re-located at snap 90 (halo 227)
        subhaloID = 537941  # re-locate at snap 99
        refVel = np.array([-45.357, 2.279, 82.549])  # SubhaloVel at main snap=40 (z=1.5)

    if gal in ["one", "three"]:
        mm1 = [5.2, 8.2]
        mm2 = [6.8, 8.6]
        mm4 = [5.0, 8.4]
        mm5 = [4.5, 7.2]
        mm6 = [-2.0, 0.2]
        mm7 = [0, 400]
        mm8 = [-200, 200]
        mm9 = [-170, 170]
        mm10 = [37.0, 40.7]
        mm11 = [4.8, 6.8]

    if gal == "two":
        # second movie, massive elliptical (sbSnaps 0 - 3600)
        sP_par = simParams(res=2160, run="tng", snap=99)
        sbNum = 2
        subhaloID = 0  # halo 0, snap 58, also snap 99
        refVel = np.array([-195.3, -52.9, -157.0])  # avg of stars within 30 pkpc of subhalo_pos at sbSnapNum=1762

        mm1 = [5.7, 8.8]
        mm2 = [6.8, 9.2]
        mm4 = [5.6, 9.2]
        mm5 = [5.4, 8.4]
        mm6 = [-1.5, 0.2]
        mm7 = [0, 1000]
        mm8 = [-400, 400]
        mm9 = [-500, 500]
        mm10 = [37.5, 41.0]
        mm11 = [5.3, 7.4]

    if gal == "three":
        # third movie, Milky Way (sbSnaps 0 - ...)
        sP_par = simParams(res=2160, run="tng", snap=90)
        sbNum = 0
        subhaloID = 481167  # halo 359, snap 90
        refVel = np.array([-10.29, -13.75, 74.17])  # snap 40, z=1.5

        mm7 = [50, 300]

    if gal == "mwbubbles1":
        # annalisa TNG50 MW bubbles paper: object one
        sP_par = simParams(run="tng50-1", redshift=0.0)
        sbNum = 2
        subhaloID = 543114  # snaps 3211 - 3599

    if gal == "mwbubbles2":
        sP_par = simParams(run="tng50-1", redshift=0.0)
        sbNum = 2
        subhaloID = 565089  # snaps 3030 - 3599

    if gal in ["mwbubbles1", "mwbubbles2"]:
        # add custom label for time elapsed since z=0.25 (movie start) in Myr
        sP_sub = simParams(run="tng50-1", variant="subbox%d" % sbNum, redshift=0.25)
        age_start = sP_sub.tage
        sP_sub.setSnap(sbSnapNum)
        labelCustom = [r"$\Delta t$ = %6.1f Myr" % ((sP_sub.tage - age_start) * 1000)]

    # parse subbox catalog, get time-evolving positions
    if cat is None:
        cat = subboxSubhaloCat(sP_par, sbNum)
    assert cat["EverInSubboxFlag"][subhaloID]

    w = np.where(cat["SubhaloIDs"] == subhaloID)[0]
    assert len(w) == 1

    subhalo_pos = cat["SubhaloPos"][w[0], :, :]  # [nSubboxSnaps,3]
    snap_start = cat["SubhaloMinSBSnap"][w[0]]
    snap_stop = cat["SubhaloMaxSBSnap"][w[0]]

    assert sbSnapNum >= snap_start and sbSnapNum <= snap_stop

    # rotation?
    rotStr = ""

    if rotSeqFrameNum is not None:
        # (first) intermediate-z rotation
        sbSnapNum = 1762  # z=1.5
        frameNum = 1569
        numFramesPerRot = 360  # 12 sec rotation, 1 deg per frame
        rotStr = "_rot_%d" % rotSeqFrameNum

        # global pre-cache of selected fields into memory
        if 0:
            sPsb = simParams(res=2160, run="tng", snap=sbSnapNum, variant="subbox%d" % sbNum)
            fieldsToCache = ["pos", "mass"]
            partType = "gas"

            dataCache = {}
            for field in fieldsToCache:
                cache_key = "snap%d_%s_%s" % (sPsb.snap, partType, field.replace(" ", "_"))
                print(" caching [%s] now..." % field)
                dataCache[cache_key] = sPsb.snapshotSubset(partType, field)
            print("All caching done.")

    if rotSeqFrameNum2 is not None:
        # (second) low-z rotation
        sbSnapNum = 2667  # z=0.74
        frameNum = 2235
        numFramesPerRot = 360  # 12 sec rotation, 1 deg per frame
        rotStr = "_rot2_%d" % rotSeqFrameNum2
        rotSeqFrameNum = rotSeqFrameNum2

        if conf in ["eight", "nine"]:
            raise Exception("Add refVel at this time, for each galaxy.")

    if rotSeqFrameNum is not None or rotSeqFrameNum2 is not None:
        # calculate rotation matrix
        print("rot frame: ", rotSeqFrameNum, " (overriding sbSnapNum)")

        rotAngleDeg = 360.0 * (rotSeqFrameNum / numFramesPerRot)
        dirVec = [0.1, 1.0, 0.4]  # full non-axis aligned tumble

        rotCenter = subhalo_pos[sbSnapNum, :]  # == boxCenter
        rotMatrix = rotationMatrixFromAngleDirection(rotAngleDeg, dirVec)

    # render configuration
    panels = []

    res = 2160
    run = "tng"
    method = "sphMap"
    variant = "subbox%d" % sbNum
    snap = sbSnapNum

    axes = [0, 1]  # x,y

    labelScale = "physical"  #'lightyears'
    labelZ = True  #'tage'
    plotHalos = False

    nPixels = [1920, 1080]  # [3840,2160]
    nPixelsSq = [540, 540]
    nPixelsSm = [960, 540]

    boxSizeLg = 300  # ckpc/h, main 16/9
    boxSizeSq = 30  # ckpc/h, galaxy square
    boxSizeSm = 900  # ckpc/h ,large-scale 16/9

    aspect = float(nPixels[0]) / nPixels[1]

    # set image center at this time
    boxCenter = subhalo_pos[sbSnapNum, :]

    # panel config
    if conf in ["one", "six", "seven", "eight", "nine", "ten", "eleven", "fifteen", "nineteen"]:
        # main panel: gas density on intermediate scales
        boxSizeImg = [int(boxSizeLg * aspect), boxSizeLg, boxSizeLg]
        loc = [0.003, 0.26]

        if conf in ["one", "fifteen"]:
            panels.append(
                {
                    "partType": "gas",
                    "partField": "coldens_msunkpc2",
                    "ctName": "magma",
                    "valMinMax": mm1,
                    "legendLoc": loc,
                }
            )
        if conf in ["nineteen"]:
            panels.append(
                {
                    "partType": "gas",
                    "partField": "coldens_msunkpc2",
                    "ctName": "cubehelix",
                    "valMinMax": mm11,
                    "legendLoc": loc,
                }
            )
        if conf == "six":
            panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": mm6, "legendLoc": loc})
        if conf == "seven":
            panels.append({"partType": "gas", "partField": "velmag", "valMinMax": mm7, "legendLoc": loc})
        if conf == "eight":
            refPos = subhalo_pos[sbSnapNum, :]
            panels.append(
                {
                    "partType": "gas",
                    "partField": "radvel",
                    "valMinMax": mm8,
                    "legendLoc": loc,
                    "refPos": refPos,
                    "refVel": refVel,
                    "ctName": "BdRd_r_black",
                }
            )
        if conf == "nine":
            projParams = {"noclip": True}
            panels.append(
                {
                    "partType": "gas",
                    "partField": "vel_los",
                    "valMinMax": mm9,
                    "legendLoc": loc,
                    "refVel": refVel,
                    "projParams": projParams,
                    "ctName": "BdRd_r_black2",
                }
            )

        if conf == "ten":
            panels.append({"partType": "gas", "partField": "sfr_halpha", "valMinMax": mm10, "legendLoc": loc})
        if conf == "eleven":
            panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-1.0, 1.6], "legendLoc": loc})

        # add custom label of subbox time resolution galaxy properties if extended info is available
        if conf == "one" and "SubhaloStars_Mass" in cat and 0:  # disabled
            import locale

            x = locale.setlocale(locale.LC_ALL, "de_DE.utf-8")
            aperture_num = 0  # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h
            stellarMass = np.squeeze(cat["SubhaloStars_Mass"][w[0], aperture_num, sbSnapNum])
            SFR = np.squeeze(cat["SubhaloGas_SFR"][w[0], aperture_num, sbSnapNum])
            labelCustom = [
                r"log M$_{\star}$ = %.2f" % sP_par.units.codeMassToLogMsun(stellarMass),
                r"SFR = %.1f M$_\odot$ yr$^{-1}$" % SFR,
            ]
            # labelCustom = ["galaxy stellar mass = %s million suns" % \
            #                locale.format_string('%d', sP_par.units.codeMassToMsun(stellarMass)/1e6,'1')]

    if conf == "two":
        # galaxy zoom panel: gas
        nPixels = nPixelsSq
        labelScale = "physical"
        labelZ = False

        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "ctName": "magma", "valMinMax": mm2})
        boxSizeImg = [boxSizeSq, boxSizeSq, boxSizeSq]

    if conf == "three":
        # galaxy zoom panel: stars
        nPixels = nPixels  # nPixelsSq
        labelScale = False
        labelZ = False

        panels.append({"partType": "stars", "partField": "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"})
        boxSizeImg = [int(boxSizeSq * aspect), boxSizeSq, boxSizeSq]
        # boxSizeImg = [boxSizeSq, boxSizeSq, boxSizeSq]

    if conf == "four":
        # large-scale structure: zoom out, DM
        nPixels = nPixelsSm
        labelScale = "physical"
        labelZ = False

        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": mm4})
        boxSizeImg = [int(boxSizeSm * aspect), boxSizeSm, boxSizeSm]

    if conf == "five":
        # large-scale structure: zoom out, gas
        nPixels = nPixelsSm
        labelScale = "physical"
        labelZ = False

        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "ctName": "thermal", "valMinMax": mm5})
        boxSizeImg = [int(boxSizeSm * aspect), boxSizeSm, boxSizeSm]

    if conf == "twelve":
        # SWR: medium zoom stars
        panels.append({"partType": "stars", "partField": "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"})
        boxSizeImg = [boxSizeSq * 2, boxSizeSq * 2, boxSizeSq * 2]

    if conf == "thirteen":
        # SWR: large zoom DM
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [6.8, 9.0]})
        boxSizeImg = [boxSizeLg, boxSizeLg, boxSizeLg]
        hsmlFac = 1.0

    if conf == "fourteen":
        # SWR: large zoom stars
        panels.append({"partType": "stars", "partField": "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"})
        boxSizeImg = [boxSizeLg, boxSizeLg, boxSizeLg]

    if conf == "sixteen":
        # bubbles, xray/temp
        panels.append({"partType": "gas", "partField": "xray", "valMinMax": [33, 38]})
        boxSizeImg = [int(boxSizeSq * aspect * 4), boxSizeSq * 4, boxSizeSq * 4]
        labelZ = True
        labelScale = True
    if conf == "seventeen":
        # bubbles, xray/temp
        panels.append({"partType": "gas", "partField": "P_gas", "valMinMax": [1.5, 5]})
        boxSizeImg = [int(boxSizeSq * aspect * 4), boxSizeSq * 4, boxSizeSq * 4]
        labelZ = True
        labelScale = True
    if conf == "eighteen":
        # xray TNG50 Fof0 cluster center thin slice
        panels.append({"partType": "gas", "partField": "xray_lum_0.5-2.0kev", "valMinMax": [36.3, 38.3]})

        boxSizeImg = [int(200.0 * aspect), 200.0, 10.0]
        labelZ = True
        labelScale = True

        sP_sub = simParams(run="tng50-1", variant="subbox%d" % sbNum, redshift=0.0)
        age_z0 = sP_sub.tage
        sP_sub.setSnap(sbSnapNum)
        labelCustom = [r"$\Delta t$ = %6.1f Myr" % ((age_z0 - sP_sub.tage) * 1000)]
        if "SubhaloBH_Mass" in cat:
            aperture_num = 0  # 0= 30 pkpc, 1= 30 ckpc/h, 2= 50 ckpc/h
            bhMass = np.squeeze(cat["SubhaloBH_Mass"][w[0], aperture_num, sbSnapNum])
            bhMdot = np.squeeze(cat["SubhaloBH_Mdot"][w[0], aperture_num, sbSnapNum])
            bhMass = sP_sub.units.codeMassToLogMsun(bhMass)
            bhMdot = np.log10(sP_sub.units.codeMassOverTimeToMsunPerYear(bhMdot))
            labelCustom.append(r"$\rm{M_{BH} = %.1f \,M_{sun}}$" % bhMass)
            labelCustom.append(r"$\rm{log\,\dot{M}_{BH} = %.1f \,M_{sun}/yr}$" % bhMdot)

            def func_post(parent_ax):
                """Custom post-render hook to draw plot on top."""
                from mpl_toolkits.axes_grid1.inset_locator import inset_axes

                ax = inset_axes(parent_ax, width="20%", height="20%", loc=4, borderpad=2.5)  # lower right corner

                ax.set_xlabel("Redshift", size=12)
                ax.set_ylabel(r"$\rm{\dot{M}_{BH} \,[log \,M_{sun}/yr}]$", size=16)
                ax.set_xlim([0.5, 0.0])
                ax.set_ylim([-4.5, -1.0])
                for tick in ax.get_xticklabels() + ax.get_yticklabels():
                    tick.set_fontsize(12)

                ax.set_facecolor("None")
                setAxisColors(ax, "white")

                xx = 1 / cat["SubboxScaleFac"] - 1
                yy = logZeroNaN(np.squeeze(cat["SubhaloBH_Mdot"][w[0], aperture_num, :]))

                ax.plot(xx, yy, "-", lw=1.5, color="white", alpha=0.5)
                ax.plot(xx[sbSnapNum], yy[sbSnapNum], "o", color="white")

            panels[0]["f_post"] = func_post

    if conf == "nineteen":
        ptRestrictions = {"temp_log": ["gt", 5.5]}

    if 0:
        # SWR
        nPixels = [1200, 1200]  # square
        labelScale = False
        labelZ = False
        labelCustom = None
        if conf == "one":
            boxSizeImg = [boxSizeLg, boxSizeLg, boxSizeLg]

    extent = [
        boxCenter[0] - 0.5 * boxSizeImg[0],
        boxCenter[0] + 0.5 * boxSizeImg[0],
        boxCenter[1] - 0.5 * boxSizeImg[1],
        boxCenter[1] + 0.5 * boxSizeImg[1],
    ]

    # render
    frameSaveNum = sbSnapNum if frameNum is None else frameNum

    class plotConfig:
        saveFilename = savePathBase + "frame_%s_%04d%s.png" % (conf, frameSaveNum, rotStr)
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False

    if conf in ["two", "three", "four", "five"]:
        plotConfig.fontsize = 13

    renderBox(panels, plotConfig, locals())


def subbox_movie_tng_galaxyevo(gal="one", conf="one"):
    """Control creation of individual frames using the above function."""
    # movie config
    minZ = 0.0

    if gal == "one":
        maxZ = 12.7  # tng subboxes start at a=0.02, but gal=='one' starts at sbSnapNum==51
    elif gal in ["mwbubbles1", "mwbubbles2"]:
        maxZ = 0.25  # short study
    else:
        maxZ = 50.0

    maxNSnaps = 2968  # there are 867 snaps with excessively small spacing between a=0.33 and a=0.47 (1308-2344)
    # as a final config, filter out half: take Nsb_final-867/2 (currently: 3400-433+eps = 2968)

    if conf == "eighteen":
        # annalisa custom movie TNG50 Fof0
        maxZ = 0.5
        maxNSnaps = None

    # get snapshot list
    sP = simParams(res=2160, run="tng", snap=90, variant="subbox0")

    sbSnapNums = sP.validSnapList(maxNum=maxNSnaps, minRedshift=minZ, maxRedshift=maxZ)

    # pre-load subbox cat (optional, must be correct for gal!)
    cat = None
    if 1:
        assert gal == "one"  #'two'
        cat = subboxSubhaloCat(simParams(run="tng50-1", redshift=0.0), sbNum=0)  # 2)

    # normal render
    for i, sbSnapNum in enumerate(sbSnapNums):
        subbox_movie_tng_galaxyevo_frame(sbSnapNum=sbSnapNum, gal=gal, conf=conf, frameNum=i, cat=cat)


def Illustris_vs_TNG_subbox0_2x1_onequant_movie(curTask=0, numTasks=1, conf=1):
    """Render a movie comparing Illustris-1 and L75n1820TNG subbox0, one quantity side by side."""
    panels = []

    # subbox0:
    panels.append({"run": "illustris", "variant": "subbox0", "zoomFac": 0.99, "labelScale": True})
    panels.append({"run": "tng", "variant": "subbox0", "zoomFac": 0.99, "labelZ": True})
    # subbox1:
    # panels.append( {'run':'illustris', 'variant':'subbox2', 'zoomFac':0.99, 'labelScale':True} )
    # panels.append( {'run':'tng',       'variant':'subbox1', 'zoomFac':0.99*(5.0/7.5), 'labelZ':True} )

    if conf == 1:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [4.2, 7.2]

    res = 1820
    method = "sphMap"
    nPixels = 1920
    labelSim = True
    axes = [0, 1]  # x,y

    class plotConfig:
        # savePath  = savePathBase + 'comp_gasdens_sb0/'
        savePath = savePathBase + "1820subbox0_highz_gasdens/"
        plotStyle = "edged_black"
        rasterPx = 1920
        colorbars = True

        # movie config
        minZ = 5.0
        maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700  # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def subbox_highz_gasdens(curTask=0, numTasks=1):
    """Render a movie of the high-z evolution (down to ~1 Gyr, z=5) of a subbox."""
    panels = []

    panels.append({"run": "tng", "res": 1820, "variant": "subbox0", "zoomFac": 0.99})

    partType = "gas"
    partField = "coldens_msunkpc2"
    valMinMax = [5.0, 8.1]
    ctName = "magma"

    method = "sphMap"
    nPixels = [1920, 1080]
    labelSim = False
    labelAge = True
    labelZ = True
    labelScale = "physical"
    axes = [0, 1]  # x,y
    textcolor = "black"

    class plotConfig:
        savePath = savePathBase + "1820subbox0_highz_gasdens/"
        plotStyle = "edged_black"
        rasterPx = nPixels
        colorbars = False
        # colorbarOverlay = True

        # movie config
        minZ = 5.0
        maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = None  # 2700 # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def Illustris_vs_TNG_subbox0_4x2_movie(curTask=0, numTasks=1):
    """Render a movie comparing Illustris-1 (top) and L75n1820TNG subbox0 (bottom), 4 quantities per row."""
    panels = []

    panels.append(
        {
            "run": "illustris",
            "partType": "gas",
            "partField": "coldens_msunkpc2",
            "valMinMax": [4.2, 7.2],
            "labelScale": True,
            "labelSim": True,
        }
    )
    panels.append({"run": "illustris", "partType": "gas", "partField": "temp", "valMinMax": [4.4, 7.6]})
    panels.append({"run": "illustris", "partType": "gas", "partField": "metal_solar", "valMinMax": [-2.0, 0.4]})
    panels.append({"run": "illustris", "partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [2.8, 8.2]})

    panels.append(
        {"run": "tng", "partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.2, 7.2], "labelSim": True}
    )
    panels.append({"run": "tng", "partType": "gas", "partField": "temp", "valMinMax": [4.4, 7.6]})
    panels.append({"run": "tng", "partType": "gas", "partField": "metal_solar", "valMinMax": [-2.0, 0.4]})
    panels.append(
        {"run": "tng", "partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [2.8, 8.2], "labelZ": True}
    )

    variant = "subbox0"
    res = 1820
    method = "sphMap"
    nPixels = 960
    axes = [0, 1]  # x,y

    class plotConfig:
        savePath = savePathBase + "comp_4x2_sb0/"
        plotStyle = "edged_black"
        rasterPx = 960
        colorbars = True

        # movie config
        minZ = 0.0
        maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700  # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def Illustris_1_4subboxes_gasdens_movie(curTask=0, numTasks=1):
    """Render a movie of a single quantity from multiple subboxes."""
    panels = []

    panels.append({"variant": "subbox0", "labelSim": True, "labelScale": True})  # upper left
    panels.append({"variant": "subbox1", "labelSim": True})  # upper right
    panels.append({"variant": "subbox2", "labelSim": True})  # lower left
    panels.append({"variant": "subbox3", "labelSim": True, "labelZ": True})  # lower right

    run = "illustris"
    partType = "gas"
    partField = "density"
    valMinMax = [-5.5, -2.0]
    res = 1820
    nPixels = 960
    axes = [0, 1]  # x,y
    redshift = 0.0

    class plotConfig:
        plotStyle = "edged_black"
        rasterPx = 960
        colorbars = True
        saveFileBase = "Illustris-1-4sb-gasDens"
        saveFilename = "out.png"

        # movie config
        minZ = 0.0
        maxZ = 4.0
        maxNSnaps = 30

    renderBox(panels, plotConfig, locals())
    # renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)


def planetarium_TychoBrahe_frames(curTask=0, numTasks=1, conf=0):
    """Render a movie comparing Illustris-1 and L75n1820TNG subbox0, one quantity side by side."""
    panels = []

    run = "tng"  # 'illustris'
    variant = "subbox0"
    zoomFac = 0.99
    res = 1820
    method = "sphMap"
    nPixels = 1920
    labelSim = True
    axes = [0, 1]  # x,y
    labelScale = False
    labelZ = False
    labelSim = False
    ctName = "gray"  # all grayscale

    if conf == 0:
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.2, 7.2]})
    if conf == 1:
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})
    if conf == 2:
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [2.8, 8.2]})
    if conf == 3:
        panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-3.0, 1.0]})
    if conf == 4:
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.4, 7.6]})
    if conf == 5:
        panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-2.0, 0.4]})
    if conf == 6:
        panels.append({"partType": "gas", "partField": "velmag", "valMinMax": [100, 1000]})
    if conf == 7:
        panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [10, 16], "labelZ": True})

    class plotConfig:
        savePath = savePathBase + "tycho/"
        plotStyle = "edged_black"
        rasterPx = 1920
        colorbars = False

        # movie config
        minZ = 0.0
        maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
        maxNSnaps = 2700  # 90 seconds at 30 fps

    renderBoxFrames(panels, plotConfig, locals(), curTask, numTasks)
