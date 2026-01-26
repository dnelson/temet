"""
MCST: visualizations / intro paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

from os.path import isfile

import h5py
import numpy as np

from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo, renderSingleHaloFrames


def vis_single_galaxy(sP, haloID=0, noSats=False):
    """Visualization: single image of a galaxy.

    Note: cannot use for a movie since the face-on/edge-on rotations have random orientations each frame.
    """
    rVirFracs = [1.0]
    fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
    nPixels = [960, 960]  # face-on panels
    nPixels_e = [960, 240]  # edge-on panels
    size = 1.0 if sP.hInd > 20000 else 5.0
    sizeType = "kpc"
    labelSim = False  # True
    labelHalo = "mhalo,mstar,haloid"
    labelZ = True
    labelScale = "physical"
    plotBHs = "all"
    plotSubhalos = False  #'all'
    relCoords = True
    if 1:
        axes = [0, 1]
        # rotation   = 'edge-on' #'face-on'

    # observational resolution?
    if 0:
        # note: 1.0 should be 2.0, since FWHM is 2x pixel scale
        nircam_fwhm = 1.0 * 0.031  # arcsec at 0.6-2.3 um (is 2x worse at 2.4-5 um)
        smoothFWHM = sP.units.arcsecToCodeLength(nircam_fwhm)

    subhaloInd = sP.halo(haloID)["GroupFirstSub"]

    # remove all particle/cells in satellite subhalos
    if noSats:
        # ptRestrictions = {'subhalo_id':['eq',subhaloInd]}
        ptRestrictions = {"sat_member": ["eq", 0]}
        plotSubhalos = False

    # redshift-dependent vis (h31619 L16 tests)
    zfac = 0.0
    if sP.redshift >= 9.9:
        zfac = 1.0
        size = 0.05  # 0.1 # z=10, 11, 12 tests of L16

    # panels (can vary hInd, variant, res)
    panels = []

    if 1:
        gas_field = "coldens_msunkpc2"  # 'HI'
        stars_field = "stellarCompObsFrame"  # stellarComp'

        gas_mm = [4.0 + zfac, 8.5 + zfac]  # [20.0+zfac,22.5+zfac]
        dm_mm = [7.0 + zfac, 11.0 + zfac]
        panels.append({"partType": "gas", "partField": gas_field, "valMinMax": gas_mm, "rotation": "face-on"})
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
                "labelSim": True,
                "labelHalo": False,
                "labelZ": False,
                "rotation": "edge-on",
            }
        )

    class plotConfig:
        plotStyle = "edged"
        colorbars = True
        fontsize = 28  # 24
        saveFilename = "galaxy_%s_%d_h%d%s.pdf" % (sP.simName, sP.snap, haloID, "_nosats" if noSats else "")

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def vis_gallery_galaxy(sims, conf=0):
    """Visualization: gallery of images of galaxies (one per run)."""
    rVirFracs = [1.0]
    fracsType = "rhalf_stars_fof"  #'rHalfMassStars'
    nPixels = [960, 960]
    sizeType = "kpc"
    # labelSim   = True
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
        label_sim = sim.simName
        if sim.haloInd > 0:
            label_sim += "b"

        panels.append(
            {"sP": sim, "subhaloInd": sub_ind, "labelSim": label_sim, "rotation": "face-on", "size": size_loc}
        )

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1000
        colorbars = True
        fontsize = 32
        saveFilename = "gallery_galaxy_conf%d_%d.pdf" % (conf, len(sims))

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


def vis_movie_mpbsm(sims, haloID=0, conf=1):
    """Render movie of a zoom run using the final merger tree and MPB-smoothed halo tracking."""
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

    if len(sims) == 1:
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
    else:
        # multiple runs: one panel each
        for sim in sims:
            sub_ind = sim.halo(haloID)["GroupFirstSub"]

            panels.append(
                {
                    "sP": sim,
                    "subhaloInd": sub_ind,
                    "partType": pt1,
                    "partField": pf1,
                    "valMinMax": vmm1,
                    "labelScale": "physical",
                    "labelSim": True,
                    "labelHalo": True,
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
