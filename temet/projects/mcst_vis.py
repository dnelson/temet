"""
MCST: visualizations / intro paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

from os.path import isfile

import h5py
import numpy as np
from scipy.signal import savgol_filter

from temet.util.simParams import simParams
from temet.vis.box import renderBox
from temet.vis.halo import _smooth_mpb_pos, renderSingleHalo, renderSingleHaloFrames

from ..util.helper import pSplit as pSplitArr
from ..util.rotation import rotationMatrixFromAngleDirection
from ..vis.common import easeQuant


def vis_single_galaxy(sP, conf=0, size=None, noSats=False):
    """Visualization: single image (or gallery of fields) of a single galaxy, with both face-on and edge-on views.

    Note: cannot use for a movie since the face-on/edge-on rotations have random orientations each frame.
    """
    rVirFracs = [2.0, 5.0, 10.0]
    fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
    nPixels = [1400, 1400]  # [960, 960]  # face-on panels
    nPixels_e = [1400, 350]  # [960, 240]  # edge-on panels

    if size is None:
        if sP.hInd > 30000:
            size = 1.0
        elif sP.hInd > 20000:
            size = 2.0
        else:
            size = 5.0

    sizeType = "kpc"
    labelSim = False  # True
    labelHalo = False  # "mhalo,mstar,haloid"
    labelZ = True
    labelScale = "physical"
    plotBHs = "all"
    plotSubhalos = False  #'all'
    relCoords = True
    axes = [0, 1]

    # observational resolution?
    if 0:
        # note: 1.0 should be 2.0, since FWHM is 2x pixel scale
        nircam_fwhm = 1.0 * 0.031  # arcsec at 0.6-2.3 um (is 2x worse at 2.4-5 um)
        smoothFWHM = sP.units.arcsecToCodeLength(nircam_fwhm)

    subhaloInd = sP.halo(sP.haloInd)["GroupFirstSub"]

    # remove all particle/cells in satellite subhalos
    if noSats:
        # ptRestrictions = {'subhalo_id':['eq',subhaloInd]}
        ptRestrictions = {"sat_member": ["eq", 0]}
        plotSubhalos = False

    # redshift-dependent vis
    zfac = 0.0
    if sP.redshift >= 9.9:
        zfac = 1.0
        # size = 0.05  # 0.1 # z=10, 11, 12 tests of L16

    # panels (can vary hInd, variant, res)
    panels = []

    if conf == 0:
        gas_field = "coldens_msunkpc2"
        stars_field = "stellarCompObsFrame"  # stellarComp'

        gas_mm = [5.0 + zfac, 8.5 + zfac]  # [20.0+zfac,22.5+zfac]
        if sP.hInd > 30000:
            gas_mm[0] += 0.5
        dm_mm = [7.0 + zfac, 11.0 + zfac]
        panels.append(
            {"partType": "gas", "partField": gas_field, "valMinMax": gas_mm, "rotation": "face-on", "labelZ": False}
        )
        # panels.append( {'partType':'dm', 'partField':gas_field, 'valMinMax':dm_mm, 'rotation':'face-on'} )
        # panels.append( {'partType':'stars', 'nPixels':[480,480], 'method':'histo', 'partField':stars_field,
        #                 'rotation':'face-on'} )
        panels.append({"partType": "stars", "partField": stars_field, "rotation": "face-on"})

        # add skinny edge-on panels below:
        panels.append(
            {
                "partType": "gas",
                "partField": gas_field,
                "nPixels": nPixels_e,
                "valMinMax": gas_mm,
                "labelScale": False,
                "labelSim": True,
                "labelHalo": False,
                "labelZ": False,
                "rotation": "edge-on",
            }
        )
        # panels.append( {'partType':'dm', 'partField':gas_field, 'nPixels':nPixels_e, 'valMinMax':dm_mm,
        #                'labelScale':False, 'labelSim':True, 'labelHalo':False, 'labelZ':False, 'rotation':'edge-on'} )
        # panels.append( {'partType':'stars', 'method':'histo', 'partField':stars_field, 'nPixels':[480,120],
        #                'labelScale':False, 'labelSim':True, 'labelHalo':False, 'labelZ':False, 'rotation':'edge-on'} )
        panels.append(
            {
                "partType": "stars",
                "partField": stars_field,
                "nPixels": nPixels_e,
                "labelScale": False,
                "labelSim": False,
                "labelHalo": False,
                "labelZ": False,
                "rotation": "edge-on",
            }
        )

    if conf == 1:
        # comparison (first set): N_gas, N_HI, N_H2, stars
        partType = "gas"
        edge_opts = {"nPixels": nPixels_e, "labelScale": False, "labelHalo": False, "labelZ": False}

        # face-on
        panels.append({"partType": "stars", "partField": "stellarCompObsFrame", "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "coldens", "valMinMax": [19.0, 23.0], "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "HI", "valMinMax": [18.0, 22.5], "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "coldens_H2", "valMinMax": [18.0, 22.5], "rotation": "face-on", "labelZ": False})

        # edge-on
        panels.append(
            {"partType": "stars", "partField": "stellarCompObsFrame", "rotation": "edge-on", "labelSim": True}
            | edge_opts
        )
        panels.append({"partField": "coldens", "valMinMax": [19.0, 23.0], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "HI", "valMinMax": [18.0, 22.5], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "coldens_H2", "valMinMax": [18.0, 22.5], "rotation": "edge-on"} | edge_opts)

    if conf == 2:
        # comparison (second set): dm, N_dust, temp, vmag
        partType = "gas"
        edge_opts = {"nPixels": nPixels_e, "labelScale": False, "labelHalo": False, "labelZ": False}

        # face-on
        panels.append(
            {
                "partType": "dm",
                "partField": "coldens_msunkpc2",
                "valMinMax": [7.2, 10.0],
                "rotation": "face-on",
                "method": "tetra_proj",
                "labelZ": False,
            }
        )
        panels.append({"partField": "coldens_dust", "valMinMax": [17.0, 20.0], "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "temp", "valMinMax": [3.5, 5.5], "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "vmag", "valMinMax": [70, 350], "rotation": "face-on", "labelZ": False})

        # edge-on
        panels.append(
            {
                "partType": "dm",
                "partField": "coldens_msunkpc2",
                "valMinMax": [7.2, 10.0],
                "rotation": "edge-on",
                "method": "tetra_proj",
                "labelSim": True,
            }
            | edge_opts
        )
        panels.append({"partField": "coldens_dust", "valMinMax": [17.0, 20.0], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "temp", "valMinMax": [3.5, 5.5], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "vmag", "valMinMax": [70, 350], "rotation": "edge-on"} | edge_opts)

    if conf == 3:
        # comparison (second set): rad_fuv, pres, LyA, [N/O]
        partType = "gas"
        edge_opts = {"nPixels": nPixels_e, "labelScale": False, "labelHalo": False, "labelZ": False}

        # face-on
        panels.append({"partField": "rad_FUV", "valMinMax": [-12, -9], "rotation": "face-on", "labelZ": False})
        panels.append({"partField": "P_gas", "valMinMax": [3, 6], "rotation": "face-on", "labelZ": False})
        panels.append(
            {"partField": "sb_Lyman-alpha_ergs", "valMinMax": [-20, -17], "rotation": "face-on", "labelZ": False}
        )
        # panels.append(
        #    {"partField": "sb_[OII]3729_ergs", "valMinMax": [-22, -19], "rotation": "face-on", "labelZ": False}
        # )
        panels.append({"partField": "massratio_N_O", "valMinMax": [-1.6, -1.0], "rotation": "face-on", "labelZ": False})

        # edge-on
        panels.append(
            {"partField": "rad_FUV", "valMinMax": [-12, -9], "rotation": "edge-on", "labelSim": "True"} | edge_opts
        )
        panels.append({"partField": "P_gas", "valMinMax": [3, 6], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "sb_Lyman-alpha_ergs", "valMinMax": [-20, -17], "rotation": "edge-on"} | edge_opts)
        # panels.append({"partField": "sb_[OII]3729_ergs", "valMinMax": [-22, -19], "rotation": "edge-on"} | edge_opts)
        panels.append({"partField": "massratio_N_O", "valMinMax": [-1.6, -1.0], "rotation": "edge-on"} | edge_opts)

    class plotConfig:
        plotStyle = "edged"
        colorbars = False if conf == 0 else True
        fontsize = 34  # 28  # 24
        saveFilename = "galaxy_%s_%d_h%d_conf%d%s.pdf" % (
            sP.simName,
            sP.snap,
            sP.haloInd,
            conf,
            "_nosats" if noSats else "",
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vis_single_large(sP, conf=1, size=None, noSats=False):
    """Visualization: single image (or gallery of fields) of a single galaxy, with both face-on and edge-on views.

    Note: cannot use for a movie since the face-on/edge-on rotations have random orientations each frame.
    """
    rVirFracs = [2.0, 5.0, 10.0]
    fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
    nPixels = [1400, 1400]  # face-on panels

    if size is None:
        if sP.hInd > 30000:
            size = 2.0  # 1.0
        elif sP.hInd > 20000:
            size = 4.0  # 2.0

    sizeType = "kpc"
    labelSim = True
    labelHalo = "mhalo,mstar,haloid"
    labelZ = True
    labelScale = "physical"
    plotBHs = "all"
    plotSubhalos = False  #'all'
    relCoords = True
    rotation = "face-on"
    axes = [0, 1]

    # method = "voronoi_slice"

    subhaloInd = sP.halo(sP.haloInd)["GroupFirstSub"]

    # remove all particle/cells in satellite subhalos
    if noSats:
        # ptRestrictions = {'subhalo_id':['eq',subhaloInd]}
        ptRestrictions = {"sat_member": ["eq", 0]}
        plotSubhalos = False

    # panels
    partType = "gas"

    if conf == 1:
        panels = [{"partType": "stars", "partField": "stellarCompObsFrame"}]
    if conf == 2:
        panels = [{"partField": "coldens", "valMinMax": [19.0, 23.0]}]
    if conf == 3:
        panels = [{"partField": "HI", "valMinMax": [18.0, 22.5]}]
    if conf == 4:
        panels = [{"partField": "coldens_H2", "valMinMax": [18.0, 22.5]}]
    if conf == 5:
        panels = [{"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [7.0, 9.5]}]
    if conf == 6:
        panels = [{"partField": "coldens_dust", "valMinMax": [17.0, 20.0]}]
    if conf == 7:
        panels = [{"partField": "temp", "valMinMax": [3.5, 5.5]}]
    if conf == 8:
        panels = [{"partField": "vmag", "valMinMax": [70, 350]}]
    if conf == 9:
        panels = [{"partField": "rad_FUV", "valMinMax": [-12, -9]}]
    if conf == 10:
        panels = [{"partField": "P_gas", "valMinMax": [3, 6]}]
    if conf == 11:
        panels = [{"partField": "sb_Lyman-alpha_ergs", "valMinMax": [-20, -17]}]
    if conf == 12:
        panels = [{"partField": "sb_[OII]3729_ergs", "valMinMax": [-22, -19]}]
    if conf == 13:
        panels = [{"partField": "massratio_N_O", "valMinMax": [-1.6, -1.0]}]

    class plotConfig:
        plotStyle = "edged"
        colorbars = True  # False
        saveFilename = "galaxy_lg_%s_%d_h%d_conf%d%s.pdf" % (
            sP.simName,
            sP.snap,
            sP.haloInd,
            conf,
            "_nosats" if noSats else "",
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vis_gallery_galaxy(sims, conf=0):
    """Visualization: gallery of images of galaxies (one per run)."""
    rVirFracs = [1.0]
    fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
    nPixels = [960, 960]
    sizeType = "kpc"
    labelSim = True
    labelHalo = "mhalo,mstar"
    labelZ = True
    labelScale = "physical"
    plotBHs = "all"
    method = "sphMap"
    relCoords = True
    axes = [0, 1]

    # panels (can vary hInd, variant, res)
    if conf == 0:
        partType = "gas"
        partField = "coldens_msunkpc2"  # 'HI'
        valMinMax = [6.0, 8.5]

    if conf == 1:
        partType = "stars"
        partField = "stellarCompObsFrame"
        valMinMax = None

    panels = []

    for sim in sims:
        # face-on + edge-on pairs
        sub_ind = sim.halo(sim.haloInd)["GroupFirstSub"]
        size_loc = 1.0 if sim.hInd < 300000 else 0.5

        panels.append({"sP": sim, "subhaloInd": sub_ind, "rotation": "face-on", "size": size_loc})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1000
        colorbars = True
        fontsize = 32
        saveFilename = "gallery_galaxy_conf%d_%d.pdf" % (conf, len(sims))

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vis_gallery_clusters(sims):
    """Visualization: gallery of images of individual star clusters."""
    rVirFracs = False
    fracsType = "rHalfMassStars"
    nPixels = [480, 480]  # 240, 360
    labelSim = True
    labelHalo = "id,mstar,rhalfstars"
    labelZ = True
    labelScale = "physical"
    method = "sphMap"  # fof-scope
    axes = [0, 1]

    # render setup
    size = 2.0
    sizeType = "pc"

    partType = "stars"
    partField = "stellarCompObsFrame"
    valMinMax = None
    autoLimits = [20, 80]  # percs for auto-scaling of stellar images

    # create one panel per cluster (simulations and redshifts vary)
    panels = []

    for sim in sims:
        panels.append({"sP": sim, "subhaloInd": sim.subhaloInd})

    class plotConfig:
        plotStyle = "edged"
        colorbars = False
        saveFilename = "gallery_clusters_%d.pdf" % len(sims)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vis_single_halo(sP, haloID=0, movie=False, galscale=False):
    """Visualization: single halo, multiple fields."""
    nPixels = [960, 960]
    labelSim = False  # True
    labelHalo = False  # 'mhalo'
    labelZ = False
    labelScale = False
    plotBHs = "all"
    relCoords = True
    axes = [0, 1]
    # rotation   = 'edge-on' #'face-on'

    if galscale:
        rVirFracs = [1.0]
        fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
        size = 1.0 if sP.hInd > 20000 else 5.0
        sizeType = "kpc"
    else:
        rVirFracs = [1.0]
        fracsType = "rVirial"
        size = 3.5  # 2.5
        sizeType = "rVirial"

    method = "voronoi_slice"  # 'sphMap'

    subhaloInd = sP.halo(haloID)["GroupFirstSub"]

    # panels: top row
    panels = []

    if method == "voronoi_slice":
        panels.append({"partType": "gas", "partField": "dens", "valMinMax": [-4.0, -1.0]})
        # panels.append( {'partType':'dm', 'partField':'dmdens', 'valMinMax':[-3.0, 0.0]} ) # not available in mini-snap
        panels.append({"partType": "dm", "method": "sphMap", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})
    else:
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.5, 7.0]})
        panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})

    panels.append({"partType": "gas", "partField": "temp", "valMinMax": [3.5, 5.0]})
    panels.append({"partType": "gas", "partField": "machnum", "valMinMax": [0, 3]})

    # bottom row
    panels.append({"partType": "gas", "partField": "rad_FUV", "valMinMax": [-15.0, -13.0], "labelScale": "physical"})
    # panels.append( {'partType':'gas', 'partField':'rad_FUV_UVB_ratio', 'valMinMax':[-0.5,0.5]} )
    # panels.append( {'partType':'gas', 'partField':'rad_LW'} )
    # panels.append( {'partType':'gas', 'partField':'rad_FUV_LW_ratio', 'valMinMax':[0.0,0.5]} )
    panels.append({"partType": "stars", "method": "sphMap", "partField": "stellarCompObsFrame", "autoLimits": False})
    panels.append({"partType": "gas", "partField": "Z_solar", "valMinMax": [-2.5, 0.0]})
    panels.append({"partType": "gas", "partField": "vrad", "valMinMax": [-60, 60], "labelZ": True})

    class plotConfig:
        plotStyle = "edged_black"
        colorbars = True
        colorbarOverlay = True
        fontsize = 28  # 24
        saveFilename = "%s_%s_%d_h%d.pdf" % ("galaxy" if galscale else "halo", sP.simName, sP.snap, haloID)

    if movie:
        plotConfig.savePath = ""
        plotConfig.saveFileBase = "%s_%sevo" % (sP.simName, "galaxy" if galscale else "halo")
        renderSingleHaloFrames(panels, plotConfig, locals())
    else:
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


# -------------------------------------------------------------------------------------------------


def vis_movie(sP, haloID=0, frame=None):
    """Visualization: movie of a single halo. Use minimal SubLink MPB tracking.

    Note: cannot use rotation for face-on/edge-on since it has random orientations each frame.
    """
    rVirFracs = [1.0]
    fracsType = "rHalfMassStars"
    nPixels = [960, 960]
    size = 2.0 if sP.hInd > 20000 else 5.0
    sizeType = "kpc"
    labelSim = True
    labelHalo = "mhalo,mstar"
    labelZ = True
    labelScale = "physical"
    plotBHs = "all"
    method = "sphMap_global"
    relCoords = True
    # axes = [0,1]

    subhaloInd = sP.halo(haloID)["GroupFirstSub"]

    # panels
    panels = []

    gas_mm = [6.0, 8.0]
    if sP.hInd >= 10000:
        gas_mm = [5.5, 7.5]
    if sP.hInd >= 1e5:
        gas_mm = [5.0, 7.0]

    if "ST" in sP.variant:
        gas_mm[0] += 1.0
        gas_mm[1] += 1.5

    panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": gas_mm})
    # panels.append( {'partType':'gas', 'partField':'HI', 'valMinMax':[20.0,22.5]} )

    # if sP.star == 1: # normal SSPs
    #    panels.append( {'partType':'stars', 'partField':'stellarComp'} )
    # if sP.star > 1: # single/solo stars
    #    panels.append( {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[gas_mm[0]-1,gas_mm[1]-1]} )

    panels.append({"partType": "stars", "partField": "stellarComp", "autoLimits": False})

    class plotConfig:
        plotStyle = "edged_black"
        colorbars = True
        fontsize = 28

    snapList = sP.validSnapList()[::-1]

    # use tree-based tracking?
    filename = sP.postPath + "/trees/SubLink/tree.hdf5"

    if isfile(filename):
        # use tree.hdf5 file for manual MPB
        print(f"Using [{filename}] for tree-based tracking.")

        with h5py.File(filename, "r") as f:
            tree = f["Tree"][()]

        # what subhalo do we search for?
        sP.setSnap(snapList[0])  # at largest snapshot number from validSnapList()
        halo = sP.halo(haloID)
        SubfindID_starting = halo["GroupFirstSub"]

        ind = np.where((tree["SnapNum"] == snapList[0]) & (tree["SubfindID"] == SubfindID_starting))[0]
        assert len(ind) == 1
        ind = ind[0]

        # get MPB
        SubhaloID = tree["SubhaloID"][ind]
        MainLeafProgID = tree["MainLeafProgenitorID"][ind]

        if MainLeafProgID == SubhaloID:
            # did not find MPB, i.e. subhalo has no tree, search one snapshot prior
            ind = np.where((tree["SnapNum"] == snapList[0] - 1) & (tree["SubfindID"] == SubfindID_starting))[0]
            assert len(ind) == 1
            ind = ind[0]

            SubhaloID = tree["SubhaloID"][ind]
            MainLeafProgID = tree["MainLeafProgenitorID"][ind]

        ind_stop = ind + (MainLeafProgID - SubhaloID)

        assert ind_stop > ind

        snaps = tree["SnapNum"][ind:ind_stop]
        subids = tree["SubfindID"][ind:ind_stop]

    if frame is not None:
        snapList = [frame]

    for snap in snapList:
        sP.setSnap(snap)

        halo = sP.halo(haloID)

        if isfile(filename):
            # use MPB tree from above
            w = np.where(snaps == snap)[0]
            if len(w) == 0:
                subhaloInd = halo["GroupFirstSub"]
            else:
                subhaloInd = subids[w[0]]
            print(f" snap [{snap:3d}] using subid = {subhaloInd:5d}")

        plotConfig.saveFilename = "%s_%03d.png" % (sP.simName, sP.snap)
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def vis_movie_mpbsm(sims, haloID=0, conf=2):
    """Render movie of a -single- zoom run using the final merger tree and MPB-smoothed halo tracking."""
    panels = []

    # panel selection
    rVirFracs = [1.0]
    fracsType = "rHalfMassStars"
    method = "sphMap_global"
    nPixels = [960, 960]
    size = 2.0 if np.max([s.hInd for s in sims]) > 20000 else 5.0
    sizeType = "kpc"
    axes = [0, 1]
    plotBHs = "all"
    labelSim = False
    relCoords = True
    rotation = None
    autoLimits = False  # disable auto-scaling of stellar band images across frames

    dmMM = [6.0, 8.5]
    gasMM = [5.0, 7.5]

    if conf == 1:
        pt1 = "gas"
        pf1 = "coldens_msunkpc2"
        vmm1 = gasMM

        pt2 = "stars"
        pf2 = "stellarComp"  # ObsFrame
        vmm2 = None

    if conf == 2:
        pt1 = "dm"
        pf1 = "coldens_msunkpc2"
        vmm1 = dmMM

    # one run: gas and stars side-by-side
    sub_ind = sims[0].halo(haloID)["GroupFirstSub"]

    panels.append(
        {
            "sP": sims[0],
            "subhaloInd": sub_ind,
            "partType": pt1,
            "partField": pf1,
            "valMinMax": vmm1,
            "labelScale": "physical",
            "labelSim": True,
        }
    )
    panels.append(
        {
            "sP": sims[0],
            "subhaloInd": sub_ind,
            "partType": pt2,
            "partField": pf2,
            "valMinMax": vmm2,
            "labelScale": "physical",
            "labelHalo": True,
            "labelZ": True,
        }
    )

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels[0]
        colorbars = True
        fontsize = 26
        savePath = ""
        saveFileBase = "%s_evo_%s" % ("-".join([s.simName for s in sims]), conf)

    renderSingleHaloFrames(panels, plotConfig, locals())


def vis_movie_mpbsm_interp(sim, haloID=0, conf="gas", pSplit=None):
    """Render movie of a -single- zoom run using the final merger tree and MPB-smoothed halo tracking (interp test)."""
    if pSplit is None:
        pSplit = [0, 1]

    panels = []

    # panel selection
    rVirFracs = []  # disabled
    fracsType = "rHalfMassStars"
    method = "sphMap_global"
    nPixels = [1920, 1080]
    # nPixels = [5120, 1440] # ultra-wide test
    size = 2.0 if sim.hInd > 20000 else 5.0
    sizeType = "kpc"
    axes = [0, 1]
    plotBHs = "all" if "_ clean" not in conf else False
    labelZ = "z_tage" if "_clean" not in conf else False
    labelScale = "physical" if "_clean" not in conf else False
    labelSim = True if "_clean" not in conf else False
    labelHalo = "mhalo,mstar" if "_clean" not in conf else False
    relCoords = True
    rotation = None
    autoLimits = False  # disable auto-scaling of stellar band images across frames

    # gradually slow down to final time
    keyf = [[5.6, 0.3], [5.5, 0.01]]

    if conf.startswith("gas"):
        pt1 = "gas"
        pf1 = "coldens_msunkpc2"
        vmm1 = [5.1, 7.6]
        vmmEvo = 3.0

        if sim.hInd == 219612 and sim.res == 16:
            # slow-down first starburst at z ~ 11.8 - 11.2
            keyf = [[12.0, 0.3], [11.8, 0.05], [11.2, 0.05], [11.0, 0.3], [5.6, 0.3], [5.5, 0.01]]
        # if sim.hInd == 311384 and sim.res == 16:
        #    # slow-down one starburst (of many) at z ~ 11.8 - 11.2
        #    keyf = [[7.0, 0.3], [6.8, 0.05], [6.5, 0.05], [6.4, 0.3]]

    if conf.startswith("dm"):
        pt1 = "dm"
        pf1 = "coldens_msunkpc2"
        method = "tetra_proj"
        vmm1 = [6.0, 9.0]  # [6.1, 8.6]
        plotSubhalos = 20 if "_clean" not in conf else False
        labelSubhalos = "msubhalo"
        vmmEvo = 6.0

    if conf.startswith("stars"):
        pt1 = "stars"
        pf1 = "stellarComp"  # ObsFrame
        vmm1 = None
        vmmEvo = None

        if sim.hInd == 219612 and sim.res == 16 and "_perspective" in conf:
            # we will make a compositive movie with gas, stay consistent in timing
            keyf = [[12.0, 0.3], [11.8, 0.05], [11.2, 0.05], [11.0, 0.3], [5.6, 0.3], [5.5, 0.01]]

    # def _custom_sfr_label(panel):
    #    sP = panel["sP"]
    #    sfr = sP.halo(haloID)["GroupSFR"]  # Msun/yr
    #    return r"SFR = %.3f $\rm{M_\odot\,yr^{-1}}$" % sfr
    # labelCustom = [_custom_sfr_label]

    # target subhalo
    sub_ind = sim.halo(haloID)["GroupFirstSub"]

    panels.append({"sP": sim, "subhaloInd": sub_ind, "partType": pt1, "partField": pf1, "valMinMax": vmm1})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = False  # True
        colorbarOverlay = True
        fontsize = 24
        savePath = ""
        saveFileBase = f"{sim.simName}_evo_interp"

        # movie config
        # interpFac = 10
        interpDt = 0.3  # Myr
        vmmEvoScalefac = vmmEvo  # shift valMinMax by this factor times the scalefactor at each frame
        keyframeDt = keyf

    # perspective rendering with keyframed camera position that moves in at z~14.5
    if "_perspective" in conf:
        # size = 30.0  # note: overwritten by keyframe
        projType = "perspective"
        projParams = {}
        if 1:
            # adaptive (e.g. large to small zoom)
            # note: overwritten by keyframe
            projParams["n"] = 10.0 * (size / 2)  # effectively sets zoom?
            projParams["f"] = 15.0 * (size / 2)
        else:
            # normal config for size = 2
            projParams["n"] = 10.0  # effectively sets zoom? 10 ~ normal, 20 ~ zoomed in, <5 distorted wide angle
            projParams["f"] = 15.0

        projParams["fov"] = 90.0  # no impact
        projParams["camera_z"] = 0.0  # camera pos offset [code units] in los direction (>1 pulls out, <1 moves in)

        if conf.startswith("gas") or conf.startswith("dm"):
            vmm1[0] += 0.5
            vmm1[1] += 0.1

        # add a slow rotation with time
        numFramesPerRot = 360 * 4

        if "_parent" in conf:
            # size keyframes: parent box (TBD)
            # plotConfig.keyframeCamera = [[17.0, 500], [16.5, 500], [15.5, 500], [15.0, 30], [14.5, 30.0], [14.0, 2.0]]
            print("TODO")
            plotConfig.keyframeDt = [[12.0, 0.3], [11.8, 0.05], [11.2, 0.05], [11.0, 0.3]]  # same as above for testing
        else:
            # size keyframes: rapid zoom-in near z ~ 14.5
            plotConfig.keyframeCamera = [[14.5, 30.0], [14.0, 2.0]]  # [z0, size0], [z1, size1]

            # for such a zoomed-out view, do not show low-res and buffer gas
            if conf.startswith("gas"):
                ptRestrictions = {"highres_massfrac": ["gt", 0.7]}

    mpb_quants = [
        "SubfindID",
        "SnapNum",
        "Group_R_Crit200",
        "SubhaloPos",
        "SubhaloVel",
        "SubhaloHalfmassRad",
        "SubhaloHalfmassRadType",
    ]

    # parent box or zoom?
    if "_parent" in conf:
        # load MPB of zoom simulation target subhalo (for consistent boxCenter)
        panels[0]["mpb"] = sim.quantMPB(sub_ind, quants=mpb_quants, add_ghosts=True, smooth=True)

        # restore global position of zoom halo
        panels[0]["mpb"]["SubhaloPos"] -= panels[0]["mpb"]["SubhaloPos"][0]  # position at z=5.5 in zoom
        panels[0]["mpb"]["SubhaloPos"] += sim.sP_parent.halo(sim.hInd)["GroupPos"]  # position at z=5.5 in TNG50

        # match redshift bounds
        plotConfig.minRedshift = sim.sP_parent.redshift

        # change target simulation from zoom to parent box
        panels[0]["subhaloInd"] = sim.hInd
        sim = simParams("tng50-2", redshift=sim.sP_parent.redshift)
        print("TODO: switch to TNG50-1")
        panels[0]["sP"] = sim
        labelHalo = False

        # testing:
        # size = 2000  # 30  # 500 # 2000
        # assert size < sim.units.codeLengthToKpc(sim.boxSize)

        # large sizes need bounds adjustment for column density
        if size >= 500:
            vmm1[0] += 1.5
            vmm1[1] += 1.0

    # render time evolution
    if "_endrot" not in conf:
        renderSingleHaloFrames(panels, plotConfig, locals(), curTask=pSplit[0], numTasks=pSplit[1])
    else:
        # render rotation (frozen in time) at final snapshot
        numFramesPerRot = 360 * 4
        rotDirVec = [0.0, 1.0, 0.0]  # horizontal seeming spin

        snap = sim.numSnaps - 1
        panels[0]["sP"].setSnap(snap)

        # get center position consistent with interpolated frames
        mpb_loc = sim.quantMPB(sub_ind, quants=["SnapNum", "SubhaloPos"], add_ghosts=True, smooth=True)
        SubhaloPos = _smooth_mpb_pos(sim, mpb_loc)
        mpb_ind = np.where(mpb_loc["SnapNum"] == snap)[0][0]
        rotCenter = SubhaloPos[mpb_ind, :]

        panels[0]["boxCenter"] = rotCenter

        frameNums = np.arange(numFramesPerRot)
        frameNums = pSplitArr(frameNums, pSplit[1], pSplit[0])
        print(f"Rendering rotation frames [{frameNums[0]} - {frameNums[-1]}] of {numFramesPerRot}")

        for frameNum in frameNums:
            # rotation amount
            rotAngleDeg = 360.0 * (frameNum / numFramesPerRot)

            # slow zoom-in/out during final rotation (only if perspective)
            if "_perspective" in conf:
                kf_frame0 = 30 * 5  # start after 5 seconds
                kf_frame1 = 30 * 5 + int(numFramesPerRot / 2)  # slow zoom for half the remaining time (~20 sec)
                kf_size0 = size
                kf_size1 = size
                if sim.hInd <= 219612:
                    # zoom-out from size=2 to size=8
                    kf_size1 = size * 4.0
                if sim.hInd in [311384, 446076]:
                    # zoom-in from size=2 to size=0.5
                    kf_size1 = size * 0.25

                cur_size = easeQuant(frameNum, kf_frame0, kf_frame1, kf_size0, kf_size1)

                panels[0]["size"] = cur_size

                # set adaptive projection parameters
                projParams["n"] = 10.0 * (cur_size / 2)
                projParams["f"] = 15.0 * (cur_size / 2)

                # add 'initial' rotation at the end of the time-evolving sequence
                evo_frameNums, _ = renderSingleHaloFrames(panels, plotConfig, locals(), getStats=True)
                frameNumTotal = evo_frameNums.size

                rotAngle0 = 360.0 * (frameNumTotal / numFramesPerRot)
                rotAngleDeg += rotAngle0

            # rotation matrix
            panels[0]["rotMatrix"] = rotationMatrixFromAngleDirection(rotAngleDeg, rotDirVec)

            # render
            plotConfig.saveFilename = f"h{sim.hInd}_L{sim.res}_{sim.variant}_{conf}_{frameNum:03d}.png"

            renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def vis_movie_mpbsm_multi(sims, conf=1):
    """Render movie of -many- zoom runs using the final merger tree and MPB-smoothed halo tracking."""
    panels = []

    # panel selection
    rVirFracs = [1.0]
    fracsType = "rHalfMassStars"
    method = "sphMap_global"
    nPixels = [960, 960]
    size = 2.0 if np.max([s.hInd for s in sims]) > 20000 else 5.0
    sizeType = "kpc"
    axes = [0, 1]
    plotBHs = "all"
    labelSim = True
    labelHalo = "mstar,mhalo"
    labelScale = "physical"
    relCoords = True
    rotation = None
    autoLimits = False  # disable auto-scaling of stellar band images across frames

    if conf == 1:
        pt = "gas"
        pf = "coldens_msunkpc2"
        vmm = [5.0, 7.5]

    if conf == 2:
        pt = "stars"
        pf = "stellarComp"  # ObsFrame
        vmm = None

    # multiple runs: one panel each
    for sim in sims:
        sub_ind = sim.halo(sim.haloInd)["GroupFirstSub"]

        panels.append({"sP": sim, "subhaloInd": sub_ind, "partType": pt, "partField": pf, "valMinMax": vmm})

    panels[-1]["labelZ"] = True  # lower right corner

    class plotConfig:
        plotStyle = "edged"
        rasterPx = int(nPixels[0] / 2)  # 1920, or 3840
        colorbars = False  # True
        fontsize = 14  # 28
        savePath = ""
        saveFileBase = "evo_n%d_conf%s" % (len(sims), conf)

    renderSingleHaloFrames(panels, plotConfig, locals())


# -------------------------------------------------------------------------------------------------


def vis_highres_region(sP, partType="dm"):
    """Visualize large-scale region that bounds all high-res DM."""
    nPixels = 1000
    axes = [0, 2]  # x,z
    labelZ = True
    labelScale = True
    labelSim = True
    plotHalos = 100
    labelHalos = "mhalo"
    relCenPos = None  # specified in absCenPos
    method = "sphMap"
    plotBHs = "all"

    # determine center and bounding box (always use high-res DM or high-res gas)
    if partType == "dm":
        pos = sP.dm("pos")
    else:
        pos = sP.gas("pos")
        highresfrac = sP.gas("highres_massfrac")

        w = np.where(highresfrac >= 0.5)[0]
        pos = pos[w, :]

    boxsize = 0.0
    absCenPos = [0, 0, 0]

    for i in range(3):
        absCenPos[i] = np.mean(pos[:, i])

    for i in range(2):
        min_v = absCenPos[axes[i]] - pos[:, axes[i]].min()
        max_v = pos[:, axes[i]].max() - absCenPos[axes[i]]

        boxsize = np.max([boxsize, min_v, max_v])

    zoomFac = 1.8 * boxsize / sP.boxSize  # fraction of box-size
    sliceFac = zoomFac  # same projection depth as zoom

    absCenPos = [absCenPos[axes[0]], absCenPos[axes[1]], absCenPos[3 - axes[0] - axes[1]]]

    if partType == "dm":
        panels = [{"partField": "coldens_msunkpc2", "valMinMax": [5.5, 8.5]}]

    if partType == "gas":
        # only high-res, no buffer
        # ptRestrictions = {'Masses':['lt',sP.targetGasMass * 3]} # approximate
        ptRestrictions = {"highres_massfrac": ["gt", 0.5]}  # need ST15+ for mini snaps
        panels = [{"partField": "coldens_msunkpc2", "valMinMax": [4.8, 7.5]}]

    class plotConfig:
        plotStyle = "edged_black"
        # colorbars  = False
        colorbarOverlay = True
        saveFilename = "./boxImage_%s_%s-%s_%03d.png" % (sP.simName, partType, panels[0]["partField"], sP.snap)

    renderBox(panels, plotConfig, locals(), skipExisting=True)


def vis_parent_box(sP, partType="dm"):
    """Visualize large-scale region that bounds all high-res DM."""
    nPixels = 2000
    axes = [0, 2]  # x,y
    labelZ = False
    labelScale = True
    labelSim = True
    plotHalos = 100  # TODO: label the specific zoom targets (only) (at z=6)
    labelHalos = "mhalo"
    method = "sphMap"
    plotBHs = "all"

    sP.setRedshift(6.0)  # z=5.5 is not a full snap, do not have SubfindHsml for DM, headache

    panels = [{"partField": "coldens_msunkpc2"}]

    if partType == "dm":
        panels[0]["valMinMax"] = [7.6, 8.8]

    if partType == "gas":
        panels[0]["valMinMax"] = [4.8, 7.5]

    class plotConfig:
        plotStyle = "edged_black"
        # colorbars  = False
        colorbarOverlay = True
        saveFilename = "./boxImage_%s_%s-%s.pdf" % (sP.simName, partType, panels[0]["partField"])

    renderBox(panels, plotConfig, locals(), skipExisting=False)
