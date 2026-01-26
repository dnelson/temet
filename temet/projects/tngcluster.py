"""
TNG-Cluster: introduction paper.

https://arxiv.org/abs/2311.06338
"""

import warnings
from functools import partial
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from temet.cosmo.zooms import _halo_ids_run, contamination_mindist
from temet.plot import subhalos
from temet.plot.config import colors, figsize, lw, markers, sKn, sKo
from temet.plot.cosmoMisc import simClustersComparison
from temet.plot.util import loadColorTable
from temet.util import simParams
from temet.util.helper import logZeroNaN, running_median
from temet.util.match import match
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def vis_fullbox_virtual(sP, conf=0):
    """Visualize the entire virtual reconstructed box."""
    axes = [0, 1]  # x,y
    labelZ = True
    labelScale = True
    labelSim = True
    nPixels = 2000

    # halo plotting
    plotHalos = False

    if conf in [0, 1, 2, 3, 4, 5]:
        pri = sP.groups("GroupPrimaryZoomTarget")
        plotHaloIDs = np.where(pri == 1)[0]

    # panel config
    if conf == 0:
        method = "sphMap_globalZoom"
        panels = [{"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [6.5, 7.1]}]

    if conf == 1:
        method = "sphMap"  # is global, overlapping coarse cells
        panels = [{"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [9.1, 9.6]}]

    if conf == 2:
        method = "sphMap_globalZoom"
        panels = [{"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]}]

    if conf == 3:
        method = "sphMap"  # is global
        panels = [{"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]}]

    if conf in [4, 5]:
        method = "sphMap"  # is global

        if conf == 4:
            panels = [{"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.8, 7.2]}]
            numBufferLevels = 3  # 2 or 3, free parameter
        if conf == 5:
            panels = [{"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [6.2, 7.6]}]
            numBufferLevels = 4

        maxGasCellMass = sP.targetGasMass
        if numBufferLevels >= 1:
            # first buffer level is 27x mass (TODO CHECK STILL?), then 8x mass for each subsequent level
            # maxGasCellMass *= 27 * np.power(8,numBufferLevels-1)
            # TEST:
            maxGasCellMass *= np.power(8, numBufferLevels)
            # add padding for x2 Gaussian distribution
            maxGasCellMass *= 3

        ptRestrictions = {"Masses": ["lt", maxGasCellMass]}

    if conf == 6:
        sP = simParams(run="tng_dm", res=2048, redshift=0.0)  # parent box
        method = "sphMap"
        panels = [{"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [7.0, 8.4]}]

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = [1080, 1080]
        colorbars = True
        fontsize = 22

        saveFilename = f"./boxImage_{sP.simName}_{panels[0]['partType']}-{panels[0]['partField']}_{sP.snap}_{conf}.pdf"

    renderBox(panels, plotConfig, locals(), skipExisting=False)


def vis_gallery(sP, conf=0, num=20):
    """Visualize a single halo in multiple fields, or a gallery of multiple halos."""
    rVirFracs = [1.0]
    axes = [0, 1]  # x,y
    labelZ = True if conf == 7 and num == 1 else False
    labelScale = "physical" if conf in [1, 7] and num == 1 else False
    labelHalo = "mhalo,haloidorig" if conf in [1, 7] and num == 1 else False
    rotation = None  # random
    nPixels = 600
    size = 3.0
    sizeType = "rVirial"

    if num == 1:
        # for single halo showcase image
        nPixels = [1920, 1080] if conf in [1, 7] else [960, 540]
        size = 4.0

    method = "sphMap_globalZoomOrig"  # all particles of original zoom run only

    # panel
    partType = "gas"
    valMinMax = None  # auto

    if conf == 0:
        partField = "coldens_msunkpc2"
        valMinMax = [5.8, 7.8]
    if conf == 1:
        # main panel
        partField = "xray_lum_0.5-2.0kev"  #'xray'
        valMinMax = [34.0, 39.5]
    if conf == 2:
        partType = "dm"
        partField = "coldens_msunkpc2"
        valMinMax = [5.5, 9.3]
    if conf == 3:
        partType = "stars"
        partField = "coldens_msunkpc2"
        valMinMax = [4.0, 7.0]
    if conf == 4:
        partField = "sz_yparam"
    if conf == 5:
        partField = "bmag_uG"
    if conf == 6:
        partField = "Z_solar"
    if conf == 7:
        partField = "HI_segmented"
        valMinMax = [12.0, 21.2]  # [12.0, 21.0]
        plotSubhalos = 100
    if conf == 8:
        partField = "temp"
        valMinMax = [7.0, 7.8]
    if conf == 9:
        partField = "vrad"
        valMinMax = [-1200, 1200]
    if conf == 10:
        partField = "velsigma_los"
    if conf == 11:
        # inset
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
        method = "sphMap"
        nPixels = [600, 600]
        size = 400.0
        sizeType = "kpc"
        labelScale = True
    if conf == 12:
        partField = "sz_yparam"
        valMinMax = [-8.0, -4.5]
        contourLevels = [-7.0, 7.0]
        contourSmooth = 2
        size = 6.0  # times rvir
        nPixels = 1000  # square aspect ratio

    # targets
    pri_target = sP.groups("GroupPrimaryZoomTarget")
    subIDs = sP.groups("GroupFirstSub")[np.where(pri_target == 1)]

    if num == 1:
        subIDs = [subIDs[2]]
    else:
        subIDs = subIDs[0:num]
        labelHalo = "haloidorig"

    # panels
    panels = []

    for subID in subIDs:
        panels.append({"subhaloInd": subID})

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels
        colorbars = True if (num == 1 and conf != 11) else False
        fontsize = 24 if conf == 1 else 26
        nCols = int(np.floor(np.sqrt(num)))
        nRows = int(np.ceil(num / nCols))

        saveFilename = "./gallery_%s_%d_%s-%s_n%d.pdf" % (sP.simName, sP.snap, partType, partField, num)

    if 0 and conf == 12:
        # tSZ and kSZ side-by-side
        ksz_vmm = [-9.0, 9.0]  # [-5e-6,5e-6] # linear vs +/-log
        panels.append({"partField": "ksz_yparam", "subhaloInd": subID, "valMinMax": ksz_vmm})

        plotConfig.nCols = 2

        panels[0]["contour"] = ["gas", "sz_yparam"]
        panels[1]["contour"] = ["gas", "ksz_yparam"]

        panels[0]["labelHalo"] = "mhalo,id"
        panels[0]["labelSim"] = True
        panels[1]["labelScale"] = True
        panels[1]["labelZ"] = True

        # plotConfig.plotStyle = 'open'

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    if conf == 12:
        ret = renderSingleHalo(panels, plotConfig, locals(), skipExisting=False, returnData=True)

        with h5py.File("%s.hdf5" % partField, "w") as f:
            f["grid"] = ret[0]
            for key in ret[1]:
                f["grid"].attrs[key] = ret[1][key]


def mass_function(secondaries=False):
    """Plot halo mass function from the parent box (TNG300) and the zoom sample.

    Args:
      secondaries (bool): if True, then also include non-targeted halos with no/little contamination.
    """
    mass_range = [14.0, 15.5]
    binSize = 0.1
    redshift = 0.0

    sP_tng300 = simParams(res=2500, run="tng", redshift=redshift)
    sP_tngc = simParams(res=2048, run="tng_dm", redshift=redshift)  # TNG-Cluster-Dark!

    # load halos
    halo_inds = _halo_ids_run(onlyDone=True)

    # start figure
    fig, ax = plt.subplots()

    nBins = int((mass_range[1] - mass_range[0]) / binSize)

    ax.set_xlim(mass_range)
    ax.set_xticks(np.arange(mass_range[0], mass_range[1], 0.1))
    ax.set_xlabel(r"Halo Mass M$_{\rm 200c}$ [ log M$_{\rm sun}$ ]")
    ax.set_ylabel(r"Number of Halos [%.1f dex$^{-1}$]" % binSize)
    ax.set_yscale("log")
    ax.yaxis.set_ticks_position("both")

    hh = []
    labels = []

    for sP in [sP_tng300, sP_tngc]:
        if sP == sP_tng300:
            # tng300
            gc = sP_tng300.halos("Group_M_Crit200")
            masses = sP_tng300.units.codeMassToLogMsun(gc)
            label = "TNG300-1"
        elif sP == sP_tngc:
            # tng-cluster
            gc = sP_tngc.halos("Group_M_Crit200")
            masses = sP_tngc.units.codeMassToLogMsun(gc[halo_inds])
            label = "TNG-Cluster"

        w = np.where(~np.isnan(masses))
        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)

        hh.append(masses[w])
        labels.append(label)

    # 'bonus': halos above 14.0 in the high-res regions of more massive zoom targets
    if secondaries:
        # note: halo_inds is for TNG-Cluster_Dark, do not use for TNG-Cluster!
        sP = simParams("tng-cluster", redshift=redshift)
        masses = sP.units.codeMassToLogMsun(sP.halos("Group_M_Crit200"))
        pri_target = sP.halos("GroupPrimaryZoomTarget")  # exclude targeted halos

        # zero contamination
        f_contam = sP.halos("GroupContaminationFracByMass")
        w = np.where((masses > ax.get_xlim()[0]) & (f_contam == 0) & (pri_target == 0))

        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)

        hh.append(masses[w])
        labels.append("TNG-Cluster Bonus (no contamination)")

        # small contamination
        contam_thresh = 1e-2
        w = np.where((masses > ax.get_xlim()[0]) & (f_contam < contam_thresh) & (f_contam != 0) & (pri_target == 0))

        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)

        hh.append(masses[w])
        labels.append("TNG-Cluster Bonus ($f_{\\rm contam} < %.1e$)" % contam_thresh)

    # plot
    ax.hist(hh, bins=nBins, range=mass_range, label=labels, histtype="bar", alpha=0.9, stacked=True)

    ax.set_ylim([0.8, 100])
    ax.legend(loc="upper right" if not secondaries else "lower left")

    fig.savefig("mass_functions.pdf")
    plt.close(fig)

    # plot histogram of contamination fraction
    if secondaries:
        bad_val = -6.0
        w = np.where(f_contam == 0)
        f_contam = logZeroNaN(f_contam)
        f_contam[w] = bad_val

        fig, ax = plt.subplots()
        ax.hist(f_contam, bins=100, range=[-7.0, 0.0])
        ax.text(bad_val + 0.1, 1e7, "Zero", ha="left", va="bottom")
        ax.set_xlabel("Contamination Fraction [by mass]")
        ax.set_ylabel("Number of Halos")
        ax.set_yscale("log")
        ax.set_xlim([bad_val - 0.2, 0.0])

        fig.savefig("contamination_fraction.pdf")
        plt.close(fig)


def sample_halomasses_vs_redshift(sPs):
    """Compare simulation vs observed cluster samples as a function of (redshift,mass)."""
    from temet.load.data import (
        adami18xxl,
        arnaud21chexmate,
        bleem20spt,
        hilton21act,
        piffaretti11rosat,
        pintoscastro19,
        rossetti17planck,
    )

    zrange = [0.0, 0.8]
    alpha = 1.0  # for data
    msize = 30  # scatter() default is 20

    # start plot
    fig = plt.figure(figsize=(figsize[0] * 1.2, figsize[1] * 1.2))
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Halo Mass M$_{\\rm 500c}$ [log M$_{\\rm sun}$]")

    ax.set_xlim(zrange)
    ax.set_ylim([13.97, 15.5])

    # load simulations and plot
    for i, sP in enumerate(sPs):
        # cache file
        cache_file = sP.cachePath + "clusters_m500_evo.hdf5"

        if isfile(cache_file):
            print("Loading [%s]" % cache_file)
            with h5py.File(cache_file, "r") as f:
                z = f["z"][()]
                m500 = f["m500"][()]
                subid = f["subid"][()]
        else:
            # TNG300: all halos with M200c > 14.0, TNG-Cluster: all primary zoom targets
            subhaloIDs = sP.cenSatSubhaloIndices(cenSatSelect="cen")
            m200c = sP.subhalos("mhalo_200_log")[subhaloIDs]

            if sP.simName != "TNG-Cluster":
                # include low-mass progenitors at high redshift
                w = np.where(m200c > 14.0)[0]
                subhaloIDs = subhaloIDs[w]

            # allocate
            snaps = sP.validSnapList()
            nsnaps = len(snaps)

            z = np.zeros(nsnaps, dtype="float32")
            subid = np.zeros((subhaloIDs.size, nsnaps), dtype="int32")
            m500 = np.zeros((subhaloIDs.size, nsnaps), dtype="float32")

            subid.fill(-1)
            m500.fill(np.nan)

            z = sP.snapNumToRedshift(snaps)

            # loop over each cluster
            for j in range(len(subhaloIDs)):
                # load the MPB
                mpb = sP.loadMPB(subhaloIDs[j], fields=["SubfindID", "SnapNum"])

                # match to master snapshot list
                inds, _ = match(snaps, mpb["SnapNum"])
                subid[j, inds] = mpb["SubfindID"]

            # loop over each snapshot
            for j, snap in enumerate(snaps):
                # load m500 values
                sP.setSnap(snap)
                m500_loc = sP.subhalos("m500_log")  # log msun

                # index to our subset of interest and stamp
                w = np.where(subid[:, j] >= 0)[0]
                m500[w, j] = m500_loc[subid[w, j]]

            # save
            with h5py.File(cache_file, "w") as f:
                f["z"] = z
                f["m500"] = m500
                f["subid"] = subid

        # draw evolution lines
        for j in range(subid.shape[0]):
            label = sP.simName if j == 0 else ""

            # interpolate to fill any nan values for display
            m500_loc = m500[j, :]

            w_nan = np.where(np.isnan(m500_loc))[0]
            if len(w_nan) > 0:
                w_finite = np.where(~np.isnan(m500_loc))[0]
                new_m500_values = np.interp(z[w_nan], z[w_finite][::-1], m500_loc[w_finite][::-1])
                m500_loc[w_nan] = new_m500_values

            # decide an alpha which decreases with decreasing z=0 mass
            if sP.simName == "TNG-Cluster":
                alpha_loc = 0.2 + 1.0 * (m500_loc[-1] - 14.2)
                zorder = -2
            else:
                alpha_loc = 0.2 + 0.5 * (m500_loc[-1] - 13.4)
                zorder = -3
            alpha_loc = np.clip(alpha_loc, 0.2, 0.9)

            ax.plot(z, m500_loc, c=colors[i], alpha=alpha_loc, zorder=zorder, label=label)
            # ax.plot(z[-1], m500_loc[-1], 'o', c=colors[i], alpha=alpha_loc, zorder=zorder, ms=4)

    # first legend
    legend1 = ax.legend(loc="upper left")
    ax.add_artist(legend1)

    # plot obs samples
    r17 = rossetti17planck()
    pc19 = pintoscastro19(sPs[0])
    h21 = hilton21act()
    a18 = adami18xxl()
    b20 = bleem20spt(sPs[0])
    p11 = piffaretti11rosat()
    a21 = arnaud21chexmate()

    opts = {"alpha": alpha, "zorder": 0}
    ax.scatter(r17["z"], r17["m500"], s=msize + 8, c="#000000", marker="s", label=r17["label"], **opts)
    ax.scatter(pc19["z"], pc19["m500"], s=msize + 8, c="#222222", marker="*", label=pc19["label"], **opts)
    ax.scatter(h21["z"], h21["m500"], s=msize - 9, c="#222222", marker="p", label=h21["label"], **opts)
    ax.scatter(a18["z"], a18["m500"], s=msize + 8, c="#222222", marker="D", label=a18["label"], **opts)
    ax.scatter(b20["z"], b20["m500"], s=msize + 8, c="#222222", marker="X", label=b20["label"], **opts)
    ax.scatter(p11["z"], p11["m500"], s=msize - 4, c="#222222", marker="h", label=p11["label"], **opts)
    ax.scatter(a21["z"], a21["m500"], s=msize + 10, c="#222222", marker="x", label=a21["label"], **opts)

    # add first legend
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles[len(sPs) :], labels[len(sPs) :], loc="upper right", frameon=True, fontsize=15)
    legend2.get_frame().set_edgecolor("#bbbbbb")
    legend2.get_frame().set_linewidth(1.0)
    ax.add_artist(legend2)

    # plot coma cluster
    def _plot_single_cluster(m500_msun, m500_err_up, m500_err_down, redshift, name):
        """Helper. Input in linear msun."""
        m500 = np.log10([m500_msun, m500_err_up, m500_err_down])

        error_lower = m500[0] - m500[2]
        error_upper = m500[1] - m500[0]
        yerr = np.reshape([error_lower, error_upper], (2, 1))

        color = "#ffffff" if name not in ["Bullet", "El Gordo"] else "#000000"
        zoff = 0.01 if name != "El Gordo" else -0.01
        ha = "left" if name != "El Gordo" else "right"
        ax.errorbar(redshift, m500[0], yerr=yerr, color=color, marker="H")
        t = ax.text(redshift + zoff, m500[0], name, fontsize=14, va="center", ha=ha, color="#ffffff")
        t.set_bbox({"facecolor": "#000000", "alpha": 0.2 if color == "#ffffff" else 0.4, "linewidth": 0})

    # plot coma cluster (Okabe+2014 Table 8, g+ profile)
    coma_z = 0.0231
    coma_m500 = np.array([3.89, 3.89 + 1.04, 3.89 - 0.76]) * 1e14 / sP.HubbleParam
    _plot_single_cluster(coma_m500[0], coma_m500[1], coma_m500[2], coma_z, "Coma")

    # plot pheonix cluster
    pheonix_z = 0.597  # currently off edge of plot
    pheonix_m500 = 2.34e15  # msun, Tozzi+15 (Section 3)
    m500_err = 0.71e15  # msun
    _plot_single_cluster(pheonix_m500, pheonix_m500 + m500_err, pheonix_m500 - m500_err, pheonix_z, "Pheonix")

    # plot el gordo cluster
    elg_z = 0.795  # true z = 0.870, moved for visibility
    elg_m500 = 8.8e14  # msun (Botteon+16)
    elgo_m500_err = 1.2e14  # msun
    _plot_single_cluster(elg_m500, elg_m500 + elgo_m500_err, elg_m500 - elgo_m500_err, elg_z, "El Gordo")

    # plot bullet cluster
    bullet_z = 0.296
    bullet_m500 = 1.1e15  # msun, Clowe+2006
    bullet_m500_err = 0.2e15  # msun
    _plot_single_cluster(bullet_m500, bullet_m500 + bullet_m500_err, bullet_m500 - bullet_m500_err, bullet_z, "Bullet")

    # plot perseus cluster
    perseus_z = 0.0183
    perseus_m500 = sP.units.m200_to_m500(6.65e14)  # Simionescu+2011
    perseus_m500_errup = sP.units.m200_to_m500(6.65e14 + 0.43e14)
    perseus_m500_errdown = sP.units.m200_to_m500(6.65e14 - 0.46e14)
    _plot_single_cluster(perseus_m500, perseus_m500_errup, perseus_m500_errdown, perseus_z, "Perseus")

    # plot virgo cluster (note: fornax m500<1e14)
    virgo_z = 0.01
    virgo_m500 = 1.01e14  # 0.8e14 msun is true value (Simionescu+2017), moved up for visibility
    virgo_m500_err = 0.05e14  # msun
    _plot_single_cluster(virgo_m500, virgo_m500 + virgo_m500_err, virgo_m500 - virgo_m500_err, virgo_z, "Virgo")

    # plot eROSITA completeness goal
    # fmt: off
    erosita_minhalo = [0.20,0.32,0.47,0.65,0.86,1.12,1.44,1.87,2.33,2.91,3.46,4.19,4.86,5.80,6.68,7.33,7.79]
    erosita_z = [0.05,0.08,0.11,0.14,0.17,0.21,0.25,0.32,0.38,0.47,0.56,0.69,0.82,1.03,1.30,1.60,1.92]
    # fmt: on

    erosita_minhalo = np.log10(sP.units.m200_to_m500(np.array(erosita_minhalo) * 1e14))  # log msun

    (l,) = ax.plot(erosita_z, erosita_minhalo, "-", alpha=alpha, color="#ffffff")
    # ax.arrow(erosita_z[6], erosita_minhalo[6]+0.02, 0.0, 0.1, head_length=0.008, color=l.get_color())
    t = ax.text(erosita_z[8] - 0.04, 14.22, "eROSITA All-Sky Complete", color=l.get_color(), fontsize=14, rotation=21)
    t.set_bbox({"facecolor": "#000000", "alpha": 0.3, "linewidth": 0})

    fig.savefig("sample_halomass_vs_redshift.pdf")
    plt.close(fig)


def bfield_strength_vs_halomass(sPs, redshifts):
    """Driver for subhalos.median."""
    sPs_in = []
    for redshift in redshifts:
        for sP in sPs:
            sPloc = sP.copy()
            sPloc.setRedshift(redshift)
            sPs_in.append(sPloc)

    xQuant = "mhalo_200_log"
    yQuant = "bmag_halfr500_volwt"
    scatterColor = "redshift"
    cenSatSelect = "cen"

    xlim = [14.0, 15.4]
    ylim = [-0.65, 0.85]
    clim = [0.0, 2.0]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    def _draw_data(ax):
        """Draw data constraints on figure."""
        # Di Gennaro+2020 (https://arxiv.org/abs/2011.01628)
        #  -- measurements based on integrated flux within <~ 0.5*r500
        bmin = np.log10(1.0)  # uG
        bmax = np.log10(3.0)  # uG
        mass_range = sPs[0].units.m500_to_m200(np.array([5e14, 9e14]))  # msun

        ax.fill_between(
            np.log10(mass_range),
            y1=[bmin, bmin],
            y2=[bmax, bmax],
            edgecolor="#cccccc",
            facecolor="#cccccc",
            alpha=0.7,
            label=r"Di Gennaro+20 ($z \sim 0.8$)",
            zorder=-1,
        )

        # Boehringer+2016 (https://arxiv.org/abs/1610.02887)
        # -- about ~90 measurements have mean r/r500 = 0.32, median r/r500 = 0.25
        bmin = np.log10(2.0)  # uG
        bmax = np.log10(6.0)  # uG
        mass_range = [2e14, 4e14]  # m200 msun

        ax.fill_between(
            np.log10(mass_range),
            y1=[bmin, bmin],
            y2=[bmax, bmax],
            edgecolor="#eeeeee",
            facecolor="#eeeeee",
            label=r"B$\rm\"{o}$hringer+16 ($z \sim 0.1$)",
            zorder=-1,
        )

    def _draw_data2(ax):
        """Draw additional data constraints on figure, individual halos."""
        b = np.log10(2.0)  # uG, Bonafede+10 https://arxiv.org/abs/1002.0594
        yerr = np.reshape([0.34, 0.34], (2, 1))  # 1-4.5 uG (center vs 1 Mpc), Bonafede+10
        m200 = 14.88  # Okabe+14 m500->m200
        xerr = np.reshape([0.1, 0.1], (2, 1))  # Okabe+14

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color="#000000", marker="D", label="Bonafede+10 (Coma)")

        b = np.log10((1.5 + 0.3) / 2)  # average of <B0>=1.5 uG and ~0.3 uG (volume average within 1 Mpc, ~1r500)
        yerr = np.reshape([0.47, 0.22], (2, 1))
        m200 = np.log10(1e14 / 0.6774)  # Govoni+17 Sec 4.2
        xerr = np.reshape([0.1, 0.1], (2, 1))  # assumed, e.g. minimum of ~30% uncertainty

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color="#000000", marker="H", label="Govoni+17 (Abell 194)")

        # TODO: Stuardi, C.+2021, Abell 2345
        # |B| = 2.8 +/- 0.1 uG (within 200 kpc)
        # M_500,SZ = 5.91e14 Msun

        # TODO: Mernier+2022 https://arxiv.org/abs/2207.10092
        # |B| = 1.9 +/- 0.3 uG (volume-averaged) (z=0.1) (aperture? mass?)

        # second legend
        handles = [plt.Line2D([0], [0], color="black", lw=0, marker=["o", "s"][i]) for i in range(len(sPs))]
        legend2 = ax.legend(handles, [sP.simName for sP in sPs], borderpad=0.4, loc="upper right")
        ax.add_artist(legend2)

    subhalos.median(
        sPs_in,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data,
        f_post=_draw_data2,
        legendLoc="lower right",
        labelSims=False,
        pdf=None,
    )


def stellar_mass_vs_halomass(sPs, conf=0):
    """Plot various stellar mass quantities vs halo mass."""
    from temet.load.data import behrooziSMHM, behrooziUM, chiu18, kravtsovSMHM, mosterSMHM

    xQuant = "mhalo_500_log"
    cenSatSelect = "cen"

    xlim = [13.8, 15.4]
    clim = [-0.4, 0.0]  # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    if conf == 0:
        yQuant = "mstar_30pkpc"
        ylabel = "BCG Stellar Mass [ log M$_{\\rm sun}$ ]"
        ylim = [10.7, 12.8]
        scatterColor = None

        def _draw_data(ax):
            # empirical SMHM relations
            b13 = behrooziSMHM(sPs[0], redshift=0.0)
            m13 = mosterSMHM(sPs[0], redshift=0.0)
            k14 = kravtsovSMHM(sPs[0])
            b19 = behrooziUM(sPs[0])

            w = np.where(b13["m500c"] < 14.8)  # for visual clarity
            l1 = ax.plot(b13["m500c"][w], b13["mstar_mid"][w], "--", lw=lw * 1.2, color="#bbbbbb")
            # ax.fill_between(b13['m500c'], b13['mstar_low'], b13['mstar_high'], color='#bbbbbb', alpha=0.3)

            w = np.where(m13["m500c"] < 14.7)  # for visual clarity
            l2 = ax.plot(m13["m500c"][w], m13["mstar_mid"][w], "-.", lw=lw * 1.2, color="#bbbbbb")
            # ax.fill_between(m13['m500c'], m13['mstar_low'], m13['mstar_high'], color='#bbbbbb', alpha=0.3)

            w = np.where(k14["m500c"] < 14.9)  # for visual clarity
            l3 = ax.plot(k14["m500c"][w], k14["mstar_mid"][w], lw=lw * 1.2, color="#888888")

            # UniverseMachine
            l4 = ax.plot(b19["m500c"], b19["mstar_mid"], lw=lw * 1.2, color="#333333")
            # ax.fill_between(b19['m500c'], b19['mstar_low'], b19['mstar_high'], color='#333333', alpha=0.3)
            ax.fill_between(b19["m500c"], b19["mstar_mid"] - 0.15, b19["mstar_mid"] + 0.15, color="#333333", alpha=0.3)

            # direct points from Fig 10 (centrals only, good, but z=0.1?, not good)
            # b19_mhalo = sPs[0].units.m200_to_m500(np.array([7.94e14, 6e13, 4e12]))
            # b19_ratio = np.array([0.0012,0.0047,0.018])
            # b19_ratio_down = np.array([0.0009,0.004,0.016])
            # b19_ratio_up = np.array([0.00167,0.0053,0.0186])
            # b19_mstar = np.log10(b19_ratio * b19_mhalo) # log msun
            # b19_mstar_up = np.log10(b19_ratio_up * b19_mhalo) # log msun
            # b19_mstar_down = np.log10(b19_ratio_down * b19_mhalo) # log msun
            # b19_mhalo = np.log10(b19_mhalo)

            # EMERGE
            m19_label = "Moster+ (2020)"
            m19_mhalo = np.array([12.0, 13.0, 14.0, 15.0])
            m19_mhalo = np.log10(sPs[0].units.m200_to_m500(10.0**m19_mhalo))  # note: actually virial, not m200
            m19_mstar = np.array([10.51, 11.04, 11.53, 11.995])  # log msun

            l5 = ax.plot(m19_mhalo, m19_mstar, color="#000000", lw=lw * 1.2, alpha=0.5)

            # Kravtsov+ 2018 (Table 1 for M500crit + Table 4 for M*<30kpc) - removed first point (in legend)
            k18_label = "Kravtsov+ (2018)"
            m500c = np.log10(np.array([10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]) * 1e14)  # 15.60,
            mstar_30pkpc = np.log10(np.array([10.44, 7.12, 3.85, 3.67, 4.35, 4.71, 4.59, 6.76]) * 1e11)  # 5.18,

            l6 = ax.scatter(m500c, mstar_30pkpc, s=markersize + 20, c="#222222", marker="D")

            # fmt: off
            # Akino+ 2022 (HSC-XXL) - Fig 4, only complete above M500>14.0
            a22_label = "Akino+ (2022)"
            a22_mhalo = [6.878e14,5.817e14,6.028e14,4.833e14,3.916e14,3.673e14,2.535e14,2.675e14,2.684e14,1.677e14,
                         1.459e14,1.507e14,1.794e14,1.671e14,1.731e14,1.961e14,1.695e14,1.750e14,1.545e14,1.306e14,
                         1.051e14,1.491e14,2.517e14,2.781e14,2.732e14,2.198e14,2.077e14,2.344e14,1.927e14,1.256e14,
                         1.161e14,8.890e13,8.488e13,1.011e14,1.121e14,1.221e14,7.654e13,7.308e13,7.334e13,6.614e13]
            a22_mstar = [1.356e12,9.435e11,5.955e11,6.704e11,8.469e11,1.100e12,9.272e11,8.588e11,7.953e11,8.528e11,
                         8.093e11,7.496e11,6.918e11,6.081e11,5.477e11,4.649e11,4.489e11,3.542e11,3.212e11,3.604e11,
                         3.157e11,2.482e11,3.960e11,3.201e11,2.975e11,2.456e11,9.884e10,1.451e11,1.721e11,1.627e11,
                         1.965e11,4.814e11,5.439e11,5.439e11,5.852e11,1.135e12,8.150e11,6.543e11,5.126e11,5.252e11]

            a22_mhalo = np.log10(a22_mhalo)
            a22_mstar = np.log10(a22_mstar)
            l7 = ax.scatter(a22_mhalo, a22_mstar, s=markersize + 20, c="#333333", marker="o")

            # Dimaio+ 2020
            d20_label = "DeMaio+ (2020)"
            d20_m500_lowz = [1.089e14,2.601e14,1.138e14,1.093e14,1.668e14,2.772e14,2.731e14,3.978e14,4.115e14,
                             2.606e14,3.724e14,5.930e14]
            d20_mstar_lowz = [6.173e11,6.270e11,7.662e11,8.417e11,9.541e11,8.261e11,8.906e11,9.046e11,9.661e11,
                              9.798e11,1.160e12,1.021e12]
            d20_m500_midz = [3.804e14,2.085e14,2.042e14,2.299e14,2.754e14,2.629e14,3.149e14,3.669e14,3.399e14,
                             2.256e14,6.096e14,7.058e14,9.383e14,5.447e14,3.600e14,7.950e13,6.006e13,6.539e13,
                             4.940e13,3.307e13,3.321e13,3.630e13]
            d20_mstar_midz = [7.107e11,8.617e11,8.822e11,8.976e11,1.037e12,1.118e12,1.162e12,1.257e12,1.299e12,
                              1.498e12,1.928e12,1.816e12,1.186e12,1.070e12,9.526e11,6.770e11,6.739e11,5.463e11,
                              3.716e11,3.474e11,5.753e11,6.369e11]

            d20_m500_lowz = np.log10(d20_m500_lowz)
            d20_mstar_lowz = np.log10(d20_mstar_lowz)
            d20_m500_midz = np.log10(d20_m500_midz)
            d20_mstar_midz = np.log10(d20_mstar_midz)

            l8 = ax.scatter(d20_m500_lowz, d20_mstar_lowz, s=markersize + 15, c="#333333", marker="s")
            ax.scatter(d20_m500_midz, d20_mstar_midz, s=markersize + 15, c="#333333", marker="s")

            # fmt: on
            # second legend
            handles = [l1[0], l2[0], l3[0], l4[0], l5[0], l6, l7, l8]
            labels = [
                b13["label"],
                m13["label"],
                k14["label"],
                b19["label"],
                m19_label,
                k18_label,
                a22_label,
                d20_label,
            ]

            legend = ax.legend(handles, labels, loc="lower right")
            ax.add_artist(legend)

    if conf == 1:
        yQuant = "mstar_r500"
        ylabel = r"Total Halo Stellar Mass [ log M$_{\rm sun}$ ]"  # BCG+ICL+SAT (e.g. fof-scope <r500c)
        ylim = [11.75, 13.4]
        scatterColor = None

        def _draw_data(ax):
            # Kravtsov+ 2018 (Figure 7 for M*tot(r<r500c), Figure 8 for M*sat(r<r500c))
            k18_label = "Kravtsov+ (2018)"
            m500c = np.log10([5.31e13, 5.68e13, 1.29e14, 1.79e14, 2.02e14, 5.40e14, 5.87e14, 8.59e14, 1.19e15])
            mstar_r500c = np.log10([1.47e12, 1.45e12, 2.28e12, 2.80e12, 2.42e12, 4.36e12, 6.58e12, 1.01e13, 1.33e13])
            # mstar_sats  = np.log10([7.97e11,3.89e11,1.45e12,1.78e12,1.83e12,3.27e12,4.39e12,6.61e12,1.07e13])

            l1 = ax.scatter(m500c, mstar_r500c, s=markersize + 20, c="#000000", marker="s")

            # fmt: off
            # Gonzalez+13 (Figure 7, mstar is <r500c, and msats is satellites within r500c)
            g13_label = "Gonzalez+ (2013)"
            m500c = [9.55e13,9.84e13,9.54e13,1.45e14,3.66e14,3.52e14,3.23e14,5.35e14,2.28e14,2.44e14,2.42e14,2.26e14]
            mstar = [2.82e12,3.21e12,4.18e12,3.06e12,4.99e12,6.07e12,7.53e12,7.04e12,5.95e12,5.95e12,5.56e12,5.50e12]
            # msats = [1.96e12,1.75e12,1.55e12,1.51e12,2.65e12,4.60e12,4.61e12,4.94e12,3.48e12,3.56e12,3.65e12,3.87e12]

            l2 = ax.scatter(np.log10(m500c), np.log10(mstar), s=markersize + 20, c="#000000", marker="D")

            # Leauthaud+12 (obtained from Kravtsov+18 Fig 7)
            l12_label = "Leauthaud+ (2012)"
            m500c = [3e13, 4.26e14]
            mstar_r500c = [5.8e11, 5.6e12]

            l3 = ax.plot(np.log10(m500c), np.log10(mstar_r500c), "-", color="#000000")

            # Akino+ (2022) HSC-XXL (Fig 3) - only complete above M500>14.0
            a22_label = "Akino+ (2022)"
            a22_m500 = [4.831e14,6.901e14,5.815e14,6.070e14,4.028e14,3.334e14,2.701e14,2.740e14,2.663e14,2.205e14,
                        2.533e14,2.524e14,3.928e14,3.671e14,1.516e14,1.675e14,1.687e14,1.468e14,1.220e14,1.057e14,
                        1.124e14,1.259e14,1.013e14,1.160e14,1.549e14,1.495e14,1.736e14,1.805e14,1.960e14,1.705e14,
                        2.082e14,2.359e14,2.780e14,1.939e14,1.761e14,1.310e14,8.911e13,8.538e13,8.297e13]
            a22_mstar = [1.099e13,6.380e12,6.447e12,3.408e12,3.349e12,2.746e12,3.157e12,3.591e12,4.551e12,4.551e12,
                         6.077e12,7.568e12,7.386e12,6.606e12,6.447e12,4.663e12,4.099e12,3.904e12,4.395e12,3.361e12,
                         2.356e12,2.389e12,1.932e12,1.802e12,2.236e12,2.708e12,2.699e12,2.873e12,1.972e12,1.771e12,
                         1.600e12,1.407e12,1.524e12,1.039e12,1.254e12,1.076e12,2.028e12,2.267e12,2.824e12]

            l4 = ax.scatter(np.log10(a22_m500), np.log10(a22_mstar), s=markersize + 20, c="#333333", marker="o")

            # Chiu+18
            c18 = chiu18()

            l5 = ax.errorbar(
                c18["M500"],
                c18["M_star"],
                xerr=c18["M500_err"],
                yerr=c18["M_star_err"],
                color="#000000",
                fmt="s",
                alpha=0.4,
            )

            # Bahe+17 (Hydrangea sims, Fig 4 left) (arXiv:1703.10610)
            b17_label = "Bahe+ (2017)"
            m500c = [13.83,13.92,13.88,13.97,14.04,14.07,14.29,14.31,14.35,14.40,14.42,14.48,14.55,14.58,14.64,14.79,
                     14.81,14.84,14.90,14.90,15.04,15.07,]#14.69,
            mstar = [12.02,12.14,12.21,12.25,12.32,12.29,12.47,12.53,12.49,12.60,12.66,12.69,12.71,12.76,12.81,13.00,
                     12.97,12.99,13.05,13.08,13.17,13.23,]#12.38,
            # fmt: on

            l6 = ax.scatter(m500c, mstar, s=markersize + 20, c="#000000", marker="*")

            # second legend
            handles = [l1, l2, l3[0], l4, l5, l6]
            labels = [k18_label, g13_label, l12_label, a22_label, c18["label"], b17_label]

            legend = ax.legend(handles, labels, loc="lower right")
            ax.add_artist(legend)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_post=_draw_data,
        ylabel=ylabel,
        legendLoc="upper left",
        pdf=None,
    )


def gas_fraction_vs_halomass(sPs):
    """Plot f_gas vs halo mass."""
    from temet.load.data import giodini2009, gonzalez2013, lovisari2015, lovisari2020

    xQuant = "mhalo_500_log"
    cenSatSelect = "cen"

    xlim = [14.0, 15.3]
    clim = [-0.4, 0.0]  # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "fgas_r500"
    ylim = [0.03, 0.18]
    scatterColor = None  #'massfrac_exsitu2'

    def _draw_data(ax):
        # observational points
        g09 = giodini2009(sPs[0])
        l15 = lovisari2015(sPs[0])
        l20 = lovisari2020()
        g13 = gonzalez2013()

        l1 = ax.errorbar(g09["m500"], g09["fGas500"], yerr=g09["fGas500Err"], color="#555555", fmt=markers[1])
        l2 = ax.errorbar(
            l15["m500"],
            l15["fgas"],
            yerr=l15["fgas_err"],
            xerr=[l15["m500_err1"], l15["m500_err2"]],
            color="#555555",
            fmt=markers[2],
        )
        l3 = ax.errorbar(
            l20["m500"],
            l20["fgas"],
            xerr=[l20["m500_err1"], l20["m500_err2"]],
            yerr=l20["fgas_err"],
            color="#888888",
            zorder=-2,
            fmt=markers[3],
        )
        l4 = ax.errorbar(
            g13["m500"], g13["fgas"], xerr=g13["m500_err"], yerr=g13["fgas_err"], color="#333333", fmt=markers[5]
        )

        # Tanimura+2020 (https://arxiv.org/abs/2007.02952) (xerr assumed)
        t20_label = "Tanimura+ (2020)"
        t20_m500 = np.log10(0.9e14 / 0.6774)
        t20_fgas = 0.13
        l5 = ax.errorbar(t20_m500, t20_fgas, xerr=0.1, yerr=0.03, marker=markers[4], alpha=0.9, color="#333333")

        # Akino+2022 (HSC-XXL) - Fig 5
        # a22_label = 'Akino+ (2022)'
        # m500       = [1.0e13, 4.4e13, 1.48e14, 4.46e14, 1.0e15]
        # fgas       = [0.039, 0.059, 0.081, 0.109, 0.139]
        # fgas_lower = [0.028, 0.051, 0.073, 0.079, 0.070]
        # fgas_upper = [0.050, 0.066, 0.089, 0.138, 0.198]

        # l6 = ax.plot(np.log10(m500), fgas, '-', color='#333')
        # ax.fill_between(np.log10(m500), fgas_lower, fgas_upper, color='#333', alpha=0.1)

        # FLAMINGO sim
        xx = [1.11e13, 1.95e13, 3.15e13, 5.48e13, 8.95e13, 1.41e14, 2.38e14, 4.57e14, 9.37e14, 1.40e15, 1.81e15]
        yy = [0.0252, 0.0366, 0.0512, 0.0702, 0.0866, 0.0988, 0.109, 0.118, 0.125, 0.126, 0.130]
        xx = np.log10(xx)

        opts = {"alpha": 0.8, "fontsize": 14, "rotation": 15}
        ax.plot(xx, yy, "-", color="#332288", alpha=0.8)
        ax.text(xx[6] - 0.022, yy[6] - 0.017, "FLAMINGO", color="#332288", ha="right", **opts)

        # MTNG sim
        xx = [1.14e13, 2.21e13, 4.01e13, 7.22e13, 1.01e14, 1.47e14, 2.20e14, 3.42e14, 5.32e14]
        yy = [0.0444, 0.0624, 0.0843, 0.106, 0.116, 0.124, 0.131, 0.134, 0.138]
        xx = np.log10(xx)

        ax.plot(xx, yy, "-", color="#3fa716", alpha=0.8)
        ax.text(xx[5] + 0.05, yy[5] - 0.013, "MTNG", color="#3fa716", ha="center", **opts)

        # TNG-Cluster fit line (sent to Elena Rasia)
        # bin_cen = [14.03, 14.08, 14.16, 14.22, 14.30, 14.36, 14.43, 14.50, 14.57, 14.63, 14.70,
        #           14.76, 14.82, 14.90, 14.97, 15.04, 15.10, 15.18]
        # median_f500 = [0.124, 0.127, 0.130, 0.132, 0.134, 0.135, 0.138, 0.139, 0.140, 0.140, 0.140,
        #               0.140, 0.140, 0.141, 0.141, 0.142, 0.143, 0.145]
        # ax.plot(bin_cen, median_f500, '-', color='#130268', lw=lw*2, alpha=0.8)

        # https://arxiv.org/pdf/2110.02228.pdf (Fig 2)
        # https://arxiv.org/abs/2206.08591 (Figure 7)
        # HSC-XXL: https://arxiv.org/abs/2111.10080 (fgas,f*,fb vs M500)
        # Eckert+19 (X-COP)
        # also: Gonzalez+13, Sun+08, Arnaud+07, Vikhlinin+06
        # https://ui.adsabs.harvard.edu/abs/2022MNRAS.510..131M/abstract (gas fractions in a shell at r2500...)

        # universal baryon fraction line
        OmegaU = sPs[0].omega_b / sPs[0].omega_m
        ax.plot(xlim, [OmegaU, OmegaU], "--", lw=1.0, color="#444444", alpha=0.3)
        ax.text(xlim[1] - 0.13, OmegaU + 0.003, r"$\Omega_{\rm b} / \Omega_{\rm m}$", size="large", alpha=0.3)

        # second legend
        handles = [l1, l2, l3, l4, l5]  # ,l6[0]]
        labels = [g09["label"], l15["label"], l20["label"], g13["label"], t20_label]  # ,a22_label]

        legend = ax.legend(handles, labels, loc="lower right")
        ax.add_artist(legend)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_post=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def sfr_vs_halomass(sPs):
    """Plot star formation rate vs halo mass."""
    # from temet.load.data import giodini2009, lovisari2015
    xQuant = "mhalo_200_log"
    cenSatSelect = "cen"

    xlim = [14.0, 15.4]
    clim = [-0.4, 0.0]  # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "sfr_30pkpc"
    ylim = [-3.5, 4.0]
    scatterColor = None

    def _draw_data(ax):
        # observational points

        # 'quenched' indicators i.e. threshold line
        mhalo = sPs[0].subhalos(xQuant)
        mstar = sPs[0].subhalos("mstar_30pkpc_log")
        inds = sPs[0].cenSatSubhaloIndices(cenSatSelect="cen")
        mhalo = mhalo[inds]
        mstar = mstar[inds]

        xx, yy, _ = running_median(mhalo, mstar, binSize=0.1)  # determine mstar/mhalo relation

        xx_mhalo = xlim  # np.linspace(xlim[0], xlim[1], 10) # mhalo
        f_interp = interp1d(xx, yy, kind="linear", fill_value="extrapolate")  # interpolate mstar to mhalo values
        xx_mstar = f_interp(xx_mhalo)

        ssfr_thresh = 1e-11  # 1/yr
        sfr = np.log10(10.0**xx_mstar * ssfr_thresh)  # log(1/yr)
        label = "Quiescent\n(sSFR < %g yr$^{-1}$)" % ssfr_thresh
        ax.plot(xx_mhalo, sfr, "-", color="#000", alpha=0.5)
        ax.fill_between(xx_mhalo, ylim[0], sfr, color="#000", alpha=0.1)
        ax.text(xlim[1] - 0.05, ylim[0] + 2.0, label, color="#000", alpha=0.5, fontsize=20, ha="right")

    def _draw_cc(ax):
        # highlight cool-cores (TNG-Cluster only)
        xx = sPs[1].subhalos(xQuant)
        yy = np.log10(sPs[1].subhalos(yQuant))
        cc_flag = sPs[1].subhalos("coolcore_flag")

        w_cc = np.where(cc_flag == 0)

        ax.plot(xx[w_cc], yy[w_cc], lw=0, marker="x", ms=8, mfc="none", mec="#000", mew=1, alpha=0.5)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data,
        f_post=_draw_cc,
        legendLoc="upper left",
        pdf=None,
    )


def mhi_vs_halomass(sPs):
    """Plot cold gas mass (M_HI) vs halo mass."""
    from temet.load.data import obuljen2019

    xQuant = "mhalo_200_log"
    cenSatSelect = "cen"

    xlim = [14.0, 15.4]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    # TODO: add 2d projection within r200 for TNG-C (from projections) (to match ALFALFA beam)
    yQuant = "mhi_halo"
    ylim = [9.9, 12.0]
    scatterColor = None  #'xray_peak_offset_2d_r500'
    clim = [-2.0, -1.0]  # log fraction

    def _draw_data(ax):
        # observational points
        o18 = obuljen2019()

        # l1, = ax.plot(o18['Mhalo'], o18['mHI'], color=color)
        # o18_down = savgol_filter(o18['mHI_low'],sKn,sKo)
        # o18_up = savgol_filter(o18['mHI_high'],sKn,sKo)
        # ax.fill_between(o18['Mhalo'], o18_down, o18_up, color='#222222', alpha=0.2)

        l1 = ax.errorbar(
            o18["pts_M_halo"],
            o18["pts_MHI"],
            yerr=[o18["pts_MHI_errdown"], o18["pts_MHI_errup"]],
            fmt="D",
            ms=10,
            zorder=2,
            color="#000000",
            alpha=1.0,
        )

        # TODO: cold gas mass (https://arxiv.org/abs/2305.12750 Fig 8)

        # TODO: add virgo
        # https://arxiv.org/pdf/2209.07691.pdf (Fig 8)

        # second legend
        handles = [l1]
        labels = [o18["label"]]

        legend = ax.legend(handles, labels, loc="lower right")
        ax.add_artist(legend)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def szy_vs_halomass(sPs):
    """Plot SZ y-parameter vs halo mass."""
    from temet.load.data import nagarajan19, planck13xx  # bleem15spt, hilton21act

    xQuant = "mhalo_500_log"
    cenSatSelect = "cen"

    xlim = [14.0, 15.3]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "szy_r500c_3d"  # 2D: 'szy_r500c_2d' (only for TNG-Cluster)
    ylim = [-5.6, -3.4]
    scatterColor = None

    def _draw_data_pre(ax):
        pass

    def _draw_data(ax):
        # create second ghost axis (to hold a second legend) (note: zorder only works within an axis)
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_axis_off()

        # (1) observational shaded bands

        # could add: https://arxiv.org/abs/2402.04006

        # lines from Jimeno+18 (Fig 7) for Planck (red)
        # xx = np.log10([8.012e13, 3.0e15]) # 1.655e15
        # y_lower = np.log10([7.818e-7, 0.00052325]) # 1.798e-4
        # y_upper = np.log10([1.578e-6, 0.00106515]) # 3.655e-4

        # ax2.fill_between(xx, y_lower, y_upper, color='#000000', alpha=0.2, zorder=-4, label='Planck XX (2013)')

        # lines from Jimeno+18 (Fig 7) for that work (blue)
        # xx = np.log10([8.012e13, 3.0e15])
        # y_lower = np.log10([8.244e-7, 0.000380666])
        # y_upper = np.log10([1.447e-6, 0.000658573])

        # Jimeno+ (2018) https://arxiv.org/abs/1706.00395 (Eqn. 30)
        # M500 = 10.0**np.array(ax.get_xlim()) # msun
        # mass_bias = 0.82
        # Y500 = 10.0**(-0.35) * (mass_bias*M500 / 6e14)**(1.70)
        # Y500 = np.log10(Y500 * 1e-4) # convert to pMpc^2
        # ax.plot(np.log10(M500), Y500, '-', color='#000000', alpha=0.7, label='Jimeno+ (2018) Planck')

        # ax2.fill_between(xx, y_lower, y_upper, color='#000000', alpha=0.3, zorder=-3, label='Jimeno+ (2018) Planck')

        # Hill+ 2018
        M500 = np.array([1.0e12, 1.0e13, 6.63e13, 2.44e14, 5.59e14, 1.0e15])
        Y500 = np.array([2.845e-9, 5.687e-7, 5.381e-5, 5.179e-4, 2.218e-3, 6.074e-3])  # arcmin^2 at z=0.15
        Y500 *= 3600  # arcsec^2
        Y500 *= sPs[0].units.arcsecToAngSizeKpcAtRedshift(1.0, 0.15) ** 2 / 1e6  # convert to pMpc^2

        ax2.plot(np.log10(M500), np.log10(Y500), "--", color="#000000", alpha=0.7, label="Hill+ (2018) CB")

        # self-similar slope
        M500 = np.array([1.5e14, 5e14])
        Y500 = np.log10(2e-5 * (M500 / 5e14) ** (5 / 3))

        ax.plot(np.log10(M500), Y500, ":", color="#000000", alpha=0.7, label=r"$\rm{Y_{500} \propto M_{500}^{5/3}}$")

        # (2) observational pointsets
        # b15 = bleem15spt(sPs[0])
        # h21 = hilton21act()
        p13 = planck13xx()
        n19 = nagarajan19()

        # ax.plot(h21['m500'], h21['sz_y'], 'p', color='#000000', ms=6, alpha=0.7)
        ax2.plot(p13["M500"], p13["Y500"], "*", color="#000000", ms=10, alpha=0.7, label="Planck XX (2013)")
        # ax.plot(b15['M500'], b15['Y'], 's', color='#000000', ms=6, alpha=0.7)

        xerr = np.vstack((n19["M500_errdown"], n19["M500_errup"]))
        yerr = np.vstack((n19["Y_errup"], n19["Y_errdown"]))
        ax2.errorbar(
            n19["M500"],
            n19["Y"],
            xerr=xerr,
            yerr=yerr,
            fmt="D",
            zorder=-2,
            color="#555555",
            ms=6,
            alpha=0.7,
            label=n19["label"],
        )

        # Adam+2023 XXL Survey - Fig 8 right panel (NIKA2 points, direct NFW mass modeling, also from other authors)
        a22_label = "Adam+ (2023)"
        a23_m500 = np.log10([1.16e14, 1.96e14, 2.47e14, 3.79e14, 5.79e14, 6.12e14, 7.45e14, 1.24e15])
        a23_m500_err1 = np.log10([1.34e14, 2.22e14, 3.16e14, 4.36e14, 6.02e14, 6.30e14, 8.69e14, 1.38e15])
        a23_m500_err2 = np.log10([9.79e13, 1.71e14, 1.77e14, 3.22e14, 5.58e14, 5.94e14, 6.49e14, 1.09e15])
        a23_y500 = np.log10(np.array([7.84, 14.0, 12.3, 27.2, 73.4, 66.8, 92.7, 169]) / 1e6)  # kpc^2 -> Mpc^2
        a23_y500_err1 = np.log10(np.array([9.39, 16.1, 15.6, 30.0, 82.3, 116, 104, 179]) / 1e6)
        a23_y500_err2 = np.log10(np.array([6.32, 11.9, 9.06, 24.4, 65.5, 43.5, 83.8, 157]) / 1e6)

        xerr = np.vstack((a23_m500_err1 - a23_m500, a23_m500 - a23_m500_err2))
        yerr = np.vstack((a23_y500_err1 - a23_y500, a23_y500 - a23_y500_err2))
        ax2.errorbar(
            a23_m500,
            a23_y500,
            xerr=xerr,
            yerr=yerr,
            fmt="s",
            zorder=-2,
            color="#444444",
            ms=10,
            alpha=0.7,
            label=a22_label,
        )

        # Planck+16 as compiled in McCarthy+17 (for <5r500)!
        # m17_xx = [14.18, 14.26, 14.36, 14.45, 14.56, 14.64, 14.75, 14.85, 14.94]
        # m17_yy = [-4.87, -4.66, -4.48, -4.39, -4.17, -4.06, -3.89, -3.73, -3.56]
        # m17_yy_lower = [-5.04, -4.87, -4.76, -4.53, -4.35, -4.19, -4.03, -3.87, -3.70]
        # m17_yy_upper = [-4.60, -4.47, -4.27, -4.17, -4.00, -3.89, -3.75, -3.62, -3.43]

        # second legend
        ax2.legend(loc="lower right")

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data_pre,
        f_post=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def XrayLum_vs_halomass(sPs):
    """Plot X-ray luminosity vs halo mass."""
    from temet.load.data import bulbul19, lovisari2020, mantz16, pratt09, vikhlinin09

    xQuant = "mhalo_500_log"
    cenSatSelect = "cen"

    xlim = [14.0, 15.3]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "xray_05_2kev_r500_halo"  # 2D: 'xraylum_r500c_2d' (only for TNG-Cluster)
    ylim = [42.8, 46.0]
    scatterColor = None

    def _draw_data(ax):
        # observational points
        p09 = pratt09()
        v09 = vikhlinin09()
        m16 = mantz16()
        b19 = bulbul19()
        l20 = lovisari2020()

        markers = ["p", "D", "*", "s", "H"]

        ax.plot(p09["M500"], p09["L05_2"], markers[0], color="#000000", ms=6, alpha=0.7)
        ax.plot(v09["M500_Y"], v09["LX"], markers[1], color="#000000", ms=6, alpha=0.7)
        ax.plot(m16["M500"], m16["LX"], markers[2], color="#000000", ms=9, alpha=0.7)
        ax.plot(b19["M500"], b19["LX"], markers[3], color="#000000", ms=6, alpha=0.7)
        ax.plot(l20["m500"], l20["LX"], markers[4], color="#000000", ms=6, alpha=0.7)

        labels = [p09["label"], v09["label"], m16["label"], b19["label"], l20["label"]]

        # second legend
        handles = [plt.Line2D([0], [0], color="black", lw=0, marker=m) for m in markers]
        legend2 = ax.legend(handles, labels, borderpad=0.4, loc="lower right")
        ax.add_artist(legend2)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_post=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def smbh_mass_vs_veldisp(sPs):
    """Plot SMBH mass versus stellar velocity dispersion."""
    from temet.load.data import bogdan2018, mcconnellMa2013

    xQuant = "veldisp1d_10pkpc"  # veldisp1d_4pkpc2d, veldisp1d (1re), veldisp1d_05re
    cenSatSelect = "cen"

    xlim = [2.2, 3.0]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "mass_smbh"  # largest in subhalo
    ylim = [8.4, 11.0]  # 8.9
    scatterColor = None

    def _draw_data(ax):
        # observational relations
        xx = np.array([2.0, 2.6])  # km/s

        # Kormendy+ (2013) Figure 17 / Eqn 3
        sigma0 = 200.0
        alpha = -0.510 + 9.0
        beta = 4.377

        log_mbh = alpha + beta * np.log10(10.0**xx / sigma0)

        l1 = ax.plot(xx, log_mbh, "--", color="#000000", alpha=0.7)

        # McConnell+ (2013) - Table 2 - Early-type
        sigma0 = 200.0
        alpha = 8.39
        beta = 5.05

        log_mbh = alpha + beta * np.log10(10.0**xx / sigma0)

        l2 = ax.plot(xx, log_mbh, "-", color="#000000", alpha=0.7)

        # McConnell+ (2013) individual points
        m13 = mcconnellMa2013()

        xerr = [m13["pts"]["sigma_down"], m13["pts"]["sigma_up"]]
        yerr = [m13["pts"]["M_BH_down"], m13["pts"]["M_BH_up"]]
        l2b = ax.errorbar(
            m13["pts"]["sigma"], m13["pts"]["M_BH"], xerr=xerr, yerr=yerr, color="#000", alpha=0.5, fmt="D"
        )

        # Bogdan+ (2018) individual points
        b18 = bogdan2018()

        xerr = [b18["sigma_errdown"], b18["sigma_errup"]]
        yerr = [b18["mbh_errdown"], b18["mbh_errup"]]
        l3 = ax.errorbar(b18["sigma"], b18["mbh"], xerr=xerr, yerr=yerr, color="#000", alpha=0.6, fmt="s")

        # Caglar+ (2023) https://arxiv.org/abs/2308.01800
        sigma0 = 200.0  # km/s
        alpha = 8.04  # Table 3 (AV corrected)
        alpha_err = 0.07
        beta = 3.09
        beta_err = 0.39

        log_mbh = alpha + beta * np.log10(10.0**xx / sigma0)

        # assume: totally uncorrelated uncertainties
        # log_mbh_up = (alpha + alpha_err) + (beta + beta_err) * np.log10(10.0**xx / sigma0)
        # log_mbh_down = (alpha - alpha_err) + (beta - beta_err) * np.log10(10.0**xx / sigma0)

        # randomly sample on parameter uncertainties and compute 1sigma of resulting values
        rng = np.random.default_rng(424242)
        alpha_rnd = rng.normal(alpha, alpha_err, size=1000)
        beta_rnd = rng.normal(beta, beta_err, size=1000)
        log_mbh_std0 = np.std(alpha_rnd + beta_rnd * np.log10(10.0 ** xx[0] / sigma0))
        log_mbh_std1 = np.std(alpha_rnd + beta_rnd * np.log10(10.0 ** xx[1] / sigma0))

        log_mbh_up2 = [log_mbh[0] + log_mbh_std0, log_mbh[1] + log_mbh_std1]
        log_mbh_down2 = [log_mbh[0] - log_mbh_std0, log_mbh[1] - log_mbh_std1]

        l4 = ax.plot(xx, log_mbh, ":", color="#666666", alpha=0.7)
        # ax.fill_between(xx, y1=log_mbh+alpha_err, y2=log_mbh-alpha_err, color='#000000', alpha=0.4)
        # ax.fill_between(xx, y1=log_mbh_down, y2=log_mbh_up, color='#000000', alpha=0.2)
        ax.fill_between(xx, y1=log_mbh_down2, y2=log_mbh_up2, color="#666666", alpha=0.2)

        # second legend
        handles = [l1[0], l2[0], l2b[0], l3[0], l4[0]]
        labels = ["Kormendy+ (2013)", "McConnell+ (2013)", "McConnell+ (2013)", "Bogdan+ (2018)", "Caglar+ (2023)"]
        legend = ax.legend(handles, labels, loc="lower right")
        ax.add_artist(legend)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def smbh_mass_vs_halomass(sPs):
    """Plot SMBH mass versus halo mass (m500c)."""
    from temet.load.data import bogdan2018

    xQuant = "m500c"
    cenSatSelect = "cen"

    xlim = [14.0, 15.3]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    yQuant = "mass_smbh"  # largest in subhalo
    ylim = [8.9, 10.8]  # 8.9
    scatterColor = None

    def _draw_data(ax):
        # observational points: Bogdan+18
        b18 = bogdan2018()

        xerr = [b18["m500_errdown"], b18["m500_errup"]]
        yerr = [b18["mbh_errdown"], b18["mbh_errup"]]
        l1 = ax.errorbar(b18["m500"], b18["mbh"], xerr=xerr, yerr=yerr, color="#000", fmt="D")

        # Gaspari+19, Figure 8 (two 9.44e13 values -> 1.05e14 for visual clarity)
        g19_m500 = np.log10([7.52e14, 2.03e14, 1.05e14, 1.02e14, 1.05e14])
        g19_m500_low = np.log10([5.59e14, 1.55e14, 7.14e13, 7.74e13, 7.18e13])
        g19_m500_high = np.log10([1.01e15, 2.64e14, 1.25e14, 1.32e14, 1.25e14])
        g19_mbh = np.log10([2.09e10, 9.01e9, 6.33e9, 7.16e9, 2.50e9])
        g19_mbh_low = np.log10([7.55e9, 6.55e9, 5.59e9, 5.16e9, 2.19e9])
        g19_mbh_high = np.log10([5.78e10, 1.26e10, 7.29e9, 1.00e10, 2.88e9])

        xerr = [g19_m500 - g19_m500_low, g19_m500_high - g19_m500]
        yerr = [g19_mbh - g19_mbh_low, g19_mbh_high - g19_mbh]
        l2 = ax.errorbar(g19_m500, g19_mbh, xerr=xerr, yerr=yerr, color="#000", fmt="s")

        # Perseus
        perseus_m500c = 6.1e14  # msun (Giacintucci+19 Table 6)
        perseus_m500c_err = 0.6e14  # msun

        perseus_mbh = 3.5e9  # msun (van den Emsellem+13)
        perseus_mbh_err = 1.5e9  # msun
        perseus_mbh2 = 1.7e10  # msun (van den Bosch+12)
        perseus_mbh2_err = 0.3e10  # msun

        x = np.log10(perseus_m500c)
        y1 = np.log10(perseus_mbh)
        y2 = np.log10(perseus_mbh2)

        xerr1 = np.log10(perseus_m500c) - np.log10(perseus_m500c - perseus_m500c_err)
        xerr2 = np.log10(perseus_m500c + perseus_m500c_err) - np.log10(perseus_m500c)
        xerr = np.reshape([xerr1, xerr2], (2, 1))

        yerr1 = [y1 - np.log10(perseus_mbh - perseus_mbh_err), np.log10(perseus_mbh + perseus_mbh_err) - y1]
        yerr2 = [y2 - np.log10(perseus_mbh2 - perseus_mbh2_err), np.log10(perseus_mbh2 + perseus_mbh2_err) - y2]
        yerr1 = np.reshape(yerr1, (2, 1))
        yerr2 = np.reshape(yerr2, (2, 1))

        l3 = ax.errorbar(x, y1, xerr=xerr, yerr=yerr1, color="#000", fmt="H")
        ax.errorbar(x, y2, xerr=xerr, yerr=yerr2, color="#000", fmt="H")
        ax.plot([x, x], [y1, y2], ":", color="#000", alpha=0.3)

        # Bassini+19 simulations (Table 2, X=M500, Y=MBH)
        a = 0.45
        b = 0.76
        log_m500 = np.array(ax.get_xlim())
        log_mbh = 9.0 + a + b * np.log10(10.0**log_m500 / 1e14)

        l4 = ax.plot(log_m500, log_mbh, "--", color="#000", alpha=0.6)

        # second legend
        handles = [l1, l2, l3, l4[0]]
        labels = ["Bogdan+ (2018)", "Gaspari+ (2019)", "Perseus (NGC1277)", "Bassini+ (2019)"]
        legend = ax.legend(handles, labels, loc="lower right")
        ax.add_artist(legend)

    subhalos.median(
        sPs,
        yQuants=[yQuant],
        xQuant=xQuant,
        cenSatSelect=cenSatSelect,
        xlim=xlim,
        ylim=ylim,
        drawMedian=drawMedian,
        markersize=markersize,
        scatterPoints=scatterPoints,
        scatterColor=scatterColor,
        sizefac=sizefac,
        f_pre=_draw_data,
        legendLoc="upper left",
        pdf=None,
    )


def cluster_radial_profiles(sim, quant="Metallicity", weight=""):
    """Plot radial profiles for various quantities.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      quant (str): quantity to plot, one of: ['Metallicity', 'Temp', 'ne', 'Entropy'].
      weight (str): if '' then mass-weighted. Otherwise '_XrayWt' is available.
    """
    from matplotlib.legend_handler import HandlerTuple
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # load
    acField = "Subhalo_RadProfile3D_Global_Gas_%s%s" % (quant, weight)
    ac = sim.auxCat(acField)

    data = ac[acField]
    attrs = ac[acField + "_attrs"]
    subhaloIDs = ac["subhaloIDs"]

    assert attrs["radRvirUnits"]  # otherwise generalize
    assert attrs["radBinsLog"]  # otherwise generalize

    rad_bin_edges = attrs["rad_bin_edges"]
    rad_bin_cen = (rad_bin_edges[1:] + rad_bin_edges[:-1]) / 2
    rad_bin_cen[0] = rad_bin_edges[0]
    rad_bin_cen[-1] = rad_bin_edges[-1]

    # masses and color values
    m200 = sim.subhalos("m200c_log")[subhaloIDs]

    clabel = r"Halo Mass M$_{\rm 200c}$ [ log M$_\odot$ ]"
    color_minmax = [14.4, 15.4]

    # compute mass binned profiles
    mass_bins = [[14.4, 14.6], [14.6, 14.8], [14.8, 15.0], [15.0, 15.2], [15.2, 15.4]]

    data_median = np.zeros((len(mass_bins), data.shape[1]), dtype="float32")
    data_mean = np.zeros((len(mass_bins), data.shape[1]), dtype="float32")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i, mass_bin in enumerate(mass_bins):
            w = np.where((m200 >= mass_bin[0]) & (m200 < mass_bin[1]))[0]
            data_median[i, :] = np.nanmedian(data[w, :], axis=0)
            data_mean[i, :] = np.nanmean(data[w, :], axis=0)

    # plot config
    ctName = "thermal"
    xlim = [0.01, 1.5]

    ylabel, _, ylog = sim.simParticleQuantity(attrs["ptType"], attrs["ptProperty"])
    if quant == "Metallicity":
        ylabel = ylabel.replace("log ", "")

    ylims = {
        "Metallicity": [0.04, 1.12],
        "Temp": [7.1, 8.2],
        "ne": [-3.6, -0.5],
        "Entropy": [8.1, 11.0],
        "Bmag": [-6.8, -4.5],
    }
    ylog = {"Metallicity": False, "Temp": True, "ne": True, "Entropy": True, "Bmag": True}

    cmap = loadColorTable(ctName)  # , fracSubset=[0.2,0.9])
    cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=color_minmax[0], vmax=color_minmax[1]), cmap=cmap)

    if ylog[quant]:
        data = logZeroNaN(data)
        data_median = logZeroNaN(data_median)
        data_mean = logZeroNaN(data_mean)

    # start plot
    xfac = 1.1 if quant == "Metallicity" else 0.8
    fig = plt.figure(figsize=(figsize[0] * xfac, figsize[1] * 0.8))
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    ax.set_xlabel(r"Radius [R$_{\rm 200c}$]")
    ax.set_ylabel(ylabel)

    ax.set_xlim(xlim)
    ax.set_xscale("log")
    # ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    ax.set_ylim(ylims[quant])

    # plot
    for i in range(data.shape[0]):
        color = cmap.to_rgba(m200[i])
        # defaults
        ls = "-"
        alpha = 0.4
        lw = 1.0

        # temp: is value at 0<r<0.25 less than 0.75<r<1.0
        # if quant == 'Temp':
        #    w1 = np.where((rad_bin_cen >= 0.0) & (rad_bin_cen < 0.25))[0]
        #    w2 = np.where((rad_bin_cen >= 0.75) & (rad_bin_cen < 1.0))[0]
        #   if np.nanmedian(data[i,w1]) < np.nanmedian(data[i,w2]):
        #        ls = '-'
        #        alpha = 1.0
        #        lw = 1.0

        ax.plot(rad_bin_cen, data[i, :], ls=ls, color=color, lw=lw, alpha=alpha, zorder=0)

    for i, mass_bin in enumerate(mass_bins):
        color = cmap.to_rgba(np.mean(mass_bin))
        ax.plot(rad_bin_cen, data_median[i, :], "-", color=color, lw=6, alpha=1.0, zorder=1)
        if quant == "Metallicity":
            ax.plot(rad_bin_cen, data_mean[i, :], ":", color=color, lw=6, alpha=1.0, zorder=1)

    # metallicity: obs data
    if quant == "Metallicity":
        # Lovisari+19 (Table 1)
        r_r500 = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.35, 0.45, 0.6, 0.85]
        data_obs_rel = [0.818, 0.589, 0.458, 0.400, 0.349, 0.313, 0.287, 0.261, 0.252, 0.278]
        data_err_rel = [0.236, 0.158, 0.110, 0.103, 0.087, 0.093, 0.095, 0.120, 0.122, 0.093]
        data_obs_dis = [0.486, 0.447, 0.417, 0.393, 0.361, 0.348, 0.322, 0.285, 0.266, 0.286]
        data_err_dis = [0.220, 0.147, 0.132, 0.117, 0.119, 0.125, 0.139, 0.132, 0.158, 0.275]

        # convert from Asplund+09 Z_solar
        Z_solar_A09 = 0.0134
        data_obs_rel = np.array(data_obs_rel) * (Z_solar_A09 / sim.units.Z_solar)
        data_obs_dis = np.array(data_obs_dis) * (Z_solar_A09 / sim.units.Z_solar)
        data_err_rel = np.array(data_err_rel) * (Z_solar_A09 / sim.units.Z_solar)
        data_err_dis = np.array(data_err_dis) * (Z_solar_A09 / sim.units.Z_solar)

        # convert from r500 -> r200
        r200 = sim.subhalos("r200")[subhaloIDs]
        r500 = sim.subhalos("r500")[subhaloIDs]

        r500_to_r200 = np.median(r500 / r200)  # 0.65

        r_r200 = np.array(r_r500) * r500_to_r200

        # plot
        opts = {"color": "#000000", "ms": 10, "alpha": 0.8}
        p1 = ax.errorbar(r_r200, data_obs_rel, yerr=data_err_rel, fmt="o", label="Lovisari+19 (rel)", **opts)
        p2 = ax.errorbar(r_r200, data_obs_dis, yerr=data_err_dis, fmt="s", label="Lovisari+19 (dis)", **opts)

        # Leccardi+08 (from Lovisari+19 Fig 6)
        xx = [0.032, 0.098, 0.180, 0.240, 0.345, 0.450, 0.623]  # r500c
        y_upper = [0.821, 0.658, 0.623, 0.609, 0.592, 0.713, 0.689]
        y_lower = [0.561, 0.323, 0.239, 0.178, 0.081, 0.015, 0.0]

        xx = np.array(xx) * r500_to_r200

        # convert from Anders+89 Z_solar (note: 0.6 converts to Asplund+05 which is Z_solar=0.0122)
        # since we take these curves from Lovisari+19, we assume they have already converted to Asplund+09
        Z_solar_AG89 = 0.01941

        y_upper = np.array(y_upper) * (Z_solar_A09 / sim.units.Z_solar)
        y_lower = np.array(y_lower) * (Z_solar_A09 / sim.units.Z_solar)

        p3 = ax.fill_between(xx, y_lower, y_upper, color="#000", alpha=0.2, label="Leccardi+08")

        # Molendi+16 (from Lovisari+19 Fig 6)
        # note: this is hardly a result from data, more of an assumption
        xx = [0.5, 2.0]  # arbitrary, 'near r180c' e.g. see Mernier+17 Fig 13
        y_upper = 0.366 * (Z_solar_A09 / sim.units.Z_solar)

        opts = {"facecolor": "#999", "alpha": 0.2, "lw": 2, "edgecolor": "#000", "ls": "--"}
        p4 = ax.fill_between(xx, [0.0, 0.0], [y_upper, y_upper], label="Molendi+16", **opts)
        # ax.plot([xx[0],xx[0]], [0.0,y_upper], '--', color='#000', alpha=0.4)
        # ax.plot(xx, [y_upper,y_upper], '--', color='#000', alpha=0.4)

        # Ghizzardi+21 XCOP - Table 2
        r_r500 = [0.0, 0.025, 0.050, 0.075, 0.150, 0.225, 0.300, 0.375, 0.450, 0.525, 0.675, 0.875, 1.12]
        r_r500 = (np.array(r_r500)[1:] + np.array(r_r500)[:-1]) / 2  # midpoints
        Z_mean = [0.578, 0.432, 0.371, 0.317, 0.276, 0.243, 0.236, 0.245, 0.252, 0.250, 0.240, 0.200]
        Z_mean_err = [0.193, 0.103, 0.066, 0.052, 0.042, 0.046, 0.042, 0.054, 0.064, 0.053, 0.047, 0.076]
        Z_median = [0.440, 0.363, 0.293, 0.299, 0.275, 0.241, 0.249, 0.274, 0.263, 0.268, 0.240, 0.174]

        r_r200 = np.array(r_r500) * r500_to_r200
        Z_mean = np.array(Z_mean) * (Z_solar_AG89 / sim.units.Z_solar)
        Z_mean_err = np.array(Z_mean_err) * (Z_solar_AG89 / sim.units.Z_solar)
        Z_median = np.array(Z_median) * (Z_solar_AG89 / sim.units.Z_solar)

        # ax.errorbar(r_r200, Z_mean, yerr=Z_mean_err, fmt='H', color='#000000', ms=10, alpha=0.8, label='Ghizzardi+21')
        (p5,) = ax.plot(r_r200, Z_mean, "-", color="#000", alpha=1.0, label="Ghizzardi+21")
        opts = {"color": (0, 0, 0, 0.2), "lw": 3, "edgecolor": (0, 0, 0, 1.0), "ls": ":"}
        p6 = ax.fill_between(r_r200, Z_mean - Z_mean_err, Z_mean + Z_mean_err, label="Ghizzardi+21", **opts)

        ax.legend(
            [(p1, p2), p3, p4, (p5, p6)],
            ["Lovisari+19", p3.get_label(), p4.get_label(), p6.get_label()],
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
        )

    # finish plot
    cax = make_axes_locatable(ax).append_axes("right", size="3%" if quant == "Metallicity" else "4%", pad=0.2)
    plt.colorbar(cmap, label=clabel, cax=cax)

    fig.savefig("rad_profiles_%s_%s%s_%d.pdf" % (sim.name, quant, weight, sim.snap))
    plt.close(fig)


def galaxy_number_profile(sim, criterion="Mr_lt205_2D"):
    """Plot radial profiles of satellite/galaxy numbers."""

    # load
    def _load_profile(sim, criterion):
        """Helper."""
        acField = "Subhalo_CountProfile_%s" % (criterion)
        ac = sim.auxCat(acField)

        data = ac[acField]
        attrs = ac[acField + "_attrs"]
        subhaloIDs = ac["subhaloIDs"]

        # rad_bin_edges = attrs['rad_bin_edges']
        rad_bins_mpc = attrs["rad_bins_pkpc"] / 1000

        m200 = sim.subhalos("m200c_log")[subhaloIDs]

        # compute mass binned profiles
        mass_bins = [[14.4, 14.6], [14.6, 14.8], [14.8, 15.0], [15.0, 15.2], [15.2, 15.4]]

        prof_binned = np.zeros((len(mass_bins), data.shape[1]), dtype="float32")
        prof_binned.fill(np.nan)

        for i, mass_bin in enumerate(mass_bins):
            w = np.where((m200 >= mass_bin[0]) & (m200 < mass_bin[1]))[0]
            prof_binned[i, :] = np.nanmean(data[w, :], axis=0)

            # patch any zeros
            w1 = np.where(prof_binned[i, :] == 0)[0]
            w2 = np.where(prof_binned[i, :] > 0)[0]
            if len(w1):
                prof_binned[i, w1] = np.interp(rad_bins_mpc[w1], rad_bins_mpc[w2], prof_binned[i, w2])

        # normalize by volume or area
        if "_2D" in criterion:
            bin_norm = sim.units.codeAreaToMpc2(attrs["bin_areas_code"])
        else:
            bin_norm = sim.units.codeVolumeToMpc3(attrs["bin_volumes_code"])

        prof_binned /= bin_norm

        return prof_binned, mass_bins, rad_bins_mpc

    # load data
    data_mean1z, mass_bins, rad_bins_mpc = _load_profile(sim, criterion)
    data_mean1x, mass_bins, rad_bins_mpc = _load_profile(sim, criterion + "x")
    data_mean1y, mass_bins, rad_bins_mpc = _load_profile(sim, criterion + "x")
    data_mean2, _, _ = _load_profile(sim, "Mstar_Gt105_2D")
    data_mean3, _, _ = _load_profile(sim, "Mstar_Gt9_2D")
    data_mean4, _, _ = _load_profile(sim, "Mstar_Gt115_2D")
    data_mean5, _, _ = _load_profile(sim, "Mr_lt205_2D_nodust")

    data_mean5z, _, _ = _load_profile(sim, "Mr_lt205_2D_nodust")
    data_mean5x, _, _ = _load_profile(sim, "Mr_lt205_2Dx_nodust")
    data_mean5y, _, _ = _load_profile(sim, "Mr_lt205_2Dy_nodust")

    data_mean1 = np.nanmean(np.array([data_mean1z, data_mean1x, data_mean1y]), axis=0)
    data_mean5 = np.nanmean(np.array([data_mean5z, data_mean5x, data_mean5y]), axis=0)

    # plot config
    ctName = "thermal"
    xlim = [0.01, 7]  # pMpc

    color_minmax = [14.4, 15.4]
    cmap = loadColorTable(ctName)
    cmap = plt.cm.ScalarMappable(norm=Normalize(vmin=color_minmax[0], vmax=color_minmax[1]), cmap=cmap)

    ylim = [0.08, 1000]

    if "_2D" in criterion:
        ylabel = r"Galaxy Surface Density $\Sigma_{\rm g}$ [ Mpc$^{-2}$ ]"
    else:
        ylabel = r"Galaxy Number Density [ Mpc$^{-3}$ ]"

    # start plot
    fig = plt.figure(figsize=(figsize[0] * 0.8, figsize[1] * 0.8))
    ax = fig.add_subplot(111)

    ax.set_xlabel("Radius [ pMpc ]")
    ax.set_ylabel(ylabel)
    ax.set_ylim(ylim)
    ax.set_xlim(xlim)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # plot
    for i, mass_bin in enumerate(mass_bins):
        color = cmap.to_rgba(np.mean(mass_bin))
        label = "%.1f < M < %.1f" % (mass_bin[0], mass_bin[1])
        ax.plot(rad_bins_mpc, savgol_filter(data_mean1[i, :], sKn, sKo), "-", color=color, alpha=1.0, label=label)

        ax.plot(rad_bins_mpc, savgol_filter(data_mean2[i, :], sKn, sKo), "--", color=color, alpha=0.4)
        ax.plot(rad_bins_mpc, savgol_filter(data_mean3[i, :], sKn, sKo), "--", color=color, alpha=0.4)
        ax.plot(rad_bins_mpc, savgol_filter(data_mean4[i, :], sKn, sKo), "--", color=color, alpha=0.4)
        ax.plot(rad_bins_mpc, savgol_filter(data_mean5[i, :], sKn, sKo), ":", color=color, alpha=1.0)

    opts = {"ha": "center", "va": "center", "color": "#000", "alpha": 0.7, "fontsize": 13}
    ax.text(0.12, 0.5, r"$M_\star > 10^{11.5} \,\rm{M}_\odot$", **opts)
    ax.text(0.12, 7.5, r"$M_\star > 10^{10.5} \,\rm{M}_\odot$", **opts)
    ax.text(0.12, 370, r"$M_\star > 10^{9.0} \,\rm{M}_\odot$", **opts)

    # obs data - Budzynski+12 (Figure 5, SDSS)
    ax.fill_between([0.01, 0.03], y1=[ylim[0], ylim[0]], y2=[ylim[1], ylim[1]], color="#444", alpha=0.2)
    ax.text(0.018, 2.0, "BCG\nObscuration", ha="center", va="center", alpha=0.6, fontsize=12)

    # fmt: off
    xx = np.array([0.041,0.055,0.073,0.100,0.138,0.191,0.258,0.353,0.476,0.610,0.896,1.221,1.688,2.256,3.103,4.247])
    y1 = np.array([109.31,106.75,84.19,70.46,56.91,49.06,39.16,29.10,24.21,17.90,11.19,5.83,3.18,1.98,1.09,0.75])
    y2 = np.array([89.87,79.34,62.95,50.24,38.92,29.98,22.95,18.21,14.11,10.55,6.60,3.40,1.98,1.20,0.80,0.55])
    y3 = np.array([68.00,55.90,42.30,31.81,24.79,18.10,14.10,10.55,7.98,6.22,3.85,2.11,1.21,0.78,0.54,0.36])
    yerr = np.array([0.23,0.23,0.24,0.24,0.24,0.23,0.22,0.22,0.23,0.24,0.27,0.28,0.29,0.30,0.31,0.31]) # relative
    # fmt: on

    y1_err = (yerr - 0.2) * (y1 / 0.2)  # y1 for 14.7-15.0
    y2_err = (yerr - 0.2) * (y2 / 0.2)  # y2 for 14.4-14.7
    y3_err = (yerr - 0.2) * (y3 / 0.2)

    p1 = ax.errorbar(xx, y1, yerr=y1_err, marker="o", ls="--", color="#000000", ms=6, alpha=0.8)
    ax.errorbar(xx, y2, yerr=y2_err, marker="o", ls="--", color="#000000", ms=6, alpha=0.8)
    ax.errorbar(xx, y3, yerr=y3_err, marker="o", ls="--", color="#000000", ms=6, alpha=0.8)
    # p1 = ax.fill_between(xx, y1=y1-y1_err, y2=y1+y1_err, color='#000', alpha=0.4)
    # ax.fill_between(xx, y1=y2-y2_err, y2=y2+y2_err, color='#000', alpha=0.4)
    # ax.fill_between(xx, y1=y3-y3_err, y2=y3+y3_err, color='#000', alpha=0.4)

    # obs data - Riggs+22 (GAMA)
    xx = [0.0129, 0.0205, 0.0324, 0.0512, 0.0815, 0.128, 0.204, 0.322, 0.509, 0.811, 1.28, 2.04, 3.24]
    yy = [66.6, 98.3, 75.2, 62.7, 43.2, 29.0, 19.8, 11.6, 5.52, 1.37, 0.123, 0.0074, 0.0014]
    yy_err = [20.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0035, 0.001]

    xx = np.array(xx) / sim.HubbleParam
    yy = np.array(yy) * sim.HubbleParam**2

    p2 = ax.errorbar(xx, yy, yerr=yy_err, marker="s", ls="-.", color="#000000", ms=6, alpha=0.8)

    # obs data - van der Burg+15 (Figure 4)
    # fmt: off
    r200_avg = 1.7  # Mpc

    xx = np.array([0.0151,0.0184,0.022,0.027,0.033,0.039,0.048,0.058,0.07,0.085,0.103,0.125,0.151,0.183,0.221,0.268,
                   0.325,0.393,0.478,0.578,0.700,0.848,1.03,1.24,1.51,1.82])
    yy1 = [np.nan,np.nan,973,1200,1060,834,789,629,614,483,446,406,361,308,264,205,177,148,122,95.0,76.3,62.1,45.9,
           31.1,22.0,16.2,]
    yy1_err = [np.nan, np.nan, 147, 180, 140, 100, 77, 78, 46, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.2, 1.3]

    yy2 = np.array([1500,1240,752,688,519,625,446,367,320,317,265,204,171,162,131,104,86.0,69.3,56.6,45.6,33.2,25.0,
                    21.6,14.8,12.2,10.2])
    yy2_up = np.array([1820,1480,934,792,614,700,497,407,345,339,276,219,179,174,137,109,89.6,71.4,59.0,47.2,34.4,
                       25.8,22.2,15.2,12.9,11.1,])
    yy2_err = yy2_up - yy2
    # fmt: on

    xx_edges = np.hstack((0.019, (xx[1:] + xx[:-1]) / 2, 1.95))
    bin_areas_r200 = np.pi * ((xx_edges[1:] + xx_edges[:-1]) / 2) ** 2
    bin_areas_Mpc2 = np.pi * ((xx_edges[1:] + xx_edges[:-1]) / 2 * r200_avg) ** 2

    yy1 = np.array(yy1) * bin_areas_r200 / bin_areas_Mpc2  # ratio is constant = 0.346
    yy1_err = np.array(yy1_err) * bin_areas_r200 / bin_areas_Mpc2
    xx_mpc = np.array(xx) * r200_avg  # r200 -> Mpc

    yy2 = np.array(yy2) * bin_areas_r200 / bin_areas_Mpc2
    yy2_err = np.array(yy2_err) * bin_areas_r200 / bin_areas_Mpc2

    p3 = ax.errorbar(xx_mpc, yy1, yerr=yy1_err, marker="D", ls=":", color="#000000", ms=6, alpha=0.8)
    ax.errorbar(xx_mpc, yy2, yerr=yy2_err, marker="D", ls=":", color="#000000", ms=6, alpha=0.8)

    # https://ui.adsabs.harvard.edu/abs/2016MNRAS.462..830Z/abstract (Fig 2)

    # finish plot
    legend1 = ax.legend([p2, p1, p3], ["Riggs+22", "Budzynski+12", "van der Burg+15"], loc="lower left")
    ax.add_artist(legend1)

    ax.legend(loc="upper right")

    fig.savefig("rad_profiles_%s_%s_%d.pdf" % (sim.name, criterion, sim.snap))
    plt.close(fig)


def halo_properties_table(sim):
    """Write out a table of primary target halo properties (in several formats).

    To then overwrite publicly available data:
    rsync -tr --progress TNG-Cluster_Catalog.* freyator:/var/www/html/files/
    """
    import pyarrow
    import pyarrow.parquet
    import zarr
    from astropy.table import Table

    cc_str = {0: "CC", 1: "WCC", 2: "NCC"}

    # TNG-Cluster: all primary zoom targets
    subhaloIDs = sim.cenSatSubhaloIndices(cenSatSelect="cen")
    haloIDs = sim.subhalos("SubhaloGrNr")[subhaloIDs]

    # load
    subhalos = sim.subhalos(
        [
            "mhalo_200_log",
            "mhalo_500_log",
            "r200",
            "r500",
            "mstar_30kpc_log",
            "mstar_100kpc_log",
            "mhi_halo_log",
            "mass_smbh_log",
            "szy_r500c_3d_log",
            "zform",
            "coolcore_flag",
            "coolcore_tcool",
            "coolcore_entropy",
            "coolcore_ne",
            "coolcore_ne_slope",
            "coolcore_c_phys",
            "coolcore_c_scaled",
            "peakoffset_xray_x",
            "peakoffset_xray_y",
            "peakoffset_xray_z",
            "peakoffset_sz_x",
            "peakoffset_sz_y",
            "peakoffset_sz_z",
        ]
    )

    for field in ["fgas_r500", "sfr_30pkpc_log", "xray_05_2kev_r500_halo_log"]:
        quant, _, _, _ = sim.simSubhaloQuantity(field)
        subhalos[field] = quant

    halos = {"origID": sim.halos("GroupOrigHaloID")}

    # compute a richness
    sub_haloIDs = sim.subhalos("SubhaloGrNr")

    for mstar_limit in [9.5, 10.0, 10.5, 11.0]:
        k = f"richness_{mstar_limit:.1f}"
        halos[k] = np.zeros(sim.numHalos, dtype="int32")
        mstar_mask = subhalos["mstar_30kpc_log"] > mstar_limit
        for haloID in haloIDs:
            w = np.where(sub_haloIDs == haloID)[0]
            halos[k][haloID] = np.sum(mstar_mask[w])

    # auxcats
    acFields = ["Subhalo_Bmag_uG_10kpc_hot_massWt", "Subhalo_ne_10kpc_hot_massWt", "Subhalo_temp_10kpc_hot_massWt"]

    for acField in acFields:
        data = sim.auxCat(acField)
        # assert np.array_equal(data['subhaloIDs'],subhaloIDs) # else take subset
        subhalos[acField] = data[acField]

    # custom files
    assert sim.snap == 99 and sim.simName == "TNG-Cluster"
    with h5py.File(sim.postPath + "released/PerseusLike_Flags_099.hdf5", "r") as f:
        halos["perseus_like_flag"] = f["FoF_Flags"][()]
        assert halos["perseus_like_flag"].size == sim.numHalos

    # take subset
    for key in subhalos:
        subhalos[key] = subhalos[key][subhaloIDs]
    for key in halos:
        halos[key] = halos[key][haloIDs]

    # list of fields: keys are name, then 3-tuple of [data, description, format string for txt output]
    # note: all data arrays have size 352 and are indexed from [0, ..., 351]
    fields = {
        "origID": [
            halos["origID"],  # [1]
            "Original Parent Halo ID (for reference only)",
            "%4d",
        ],
        "haloID": [
            haloIDs,  # [2]
            "TNG-Cluster Halo ID (in the group catalog)",
            "%8d",
        ],
        "mhalo_200c": [
            subhalos["mhalo_200_log"],  # [3]
            r"Halo mass within r200c [log M$_\odot$]",
            "%5.2f",
        ],
        "mhalo_500c": [
            subhalos["mhalo_500_log"],  # [4]
            r"Halo mass within r500c [log M$_\odot$]",
            "%5.2f",
        ],
        "r200c": [
            subhalos["r200"] / 1000,  # [5]
            "Halo radius r200c [Mpc]",
            "%5.3f",
        ],
        "r500c": [
            subhalos["r500"] / 1000,  # [6]
            "Halo radius r500c [Mpc]",
            "%5.3f",
        ],
        "mstar_30kpc": [
            subhalos["mstar_30kpc_log"],  # [7]
            r"Stellar mass within 30 kpc [log(M$_\star$ / M$_\odot$]",
            "%5.2f",
        ],
        "mstar_100kpc": [
            subhalos["mstar_100kpc_log"],  # [8]
            r"Stellar mass within 100 kpc [log(M$_\star$ / M$_\odot$]",
            "%5.2f",
        ],
        "mhi_halo": [
            subhalos["mhi_halo_log"],  # [9]
            r"Neutral HI mass within the (FoF) halo [log M$_\odot$]",
            "%5.2f",
        ],
        "mass_smbh": [
            subhalos["mass_smbh_log"],  # [10]
            r"Mass of central SMBH [log M$_\odot$]",
            "%5.2f",
        ],
        "fgas_r500": [
            subhalos["fgas_r500"],  # [11]
            "Gas fraction within r500c [unitless]",
            "%5.3f",
        ],
        "sfr_30pkpc": [
            subhalos["sfr_30pkpc_log"],  # [12]
            r"Star formation rate within 30 pkpc (nan indicates an upper limit of -3.0) [log M$_\odot$ / yr]",
            "%5.2f",
        ],
        "xray_05_2kev": [
            subhalos["xray_05_2kev_r500_halo_log"],  # [13]
            "X-ray luminosity 0.5-2 keV soft-band, within r500c in 3D [log erg/s]",
            "%5.2f",
        ],
        "szy_r500c": [
            subhalos["szy_r500c_3d_log"],  # [14]
            "Integrated Y_SZ parameter within r500c in 3D [log Mpc$^2$]",
            "%5.2f",
        ],
        "Bmag_10kpc": [
            subhalos["Subhalo_Bmag_uG_10kpc_hot_massWt"],  # [15]
            "Magnetic field magnitude within 10 kpc, mass-weighted mean of log(T)>5.5 gas [micro Gauss]",
            "%6.2f",
        ],
        "ne_10kpc": [
            np.log10(subhalos["Subhalo_ne_10kpc_hot_massWt"]),  # [16]
            "Electron density within 10 kpc, mass-weighted mean of log(T)>5.5 gas [log cm$^{-3}$]",
            "%5.2f",
        ],
        "temp_10kpc": [
            np.log10(subhalos["Subhalo_temp_10kpc_hot_massWt"]),  # [17]
            "Temperature within 10 kpc, mass-weighted mean of log(T)>5.5 gas [log K]",
            "%5.2f",
        ],
        "zform": [
            subhalos["zform"],  # [18]
            "Formation redshift [unitless]",
            "%4.2f",
        ],
        "richness_9.5": [
            halos["richness_9.5"],  # [19]
            "Number of satellite galaxies with log(M*) (30pkpc) > 9.5 Msun [unitless]",
            "%3d",
        ],
        "richness_10.0": [
            halos["richness_10.0"],  # [20]
            "Number of satellite galaxies with log(M*) (30pkpc) > 10.0 Msun [unitless]",
            "%3d",
        ],
        "richness_10.5": [
            halos["richness_10.5"],  # [21]
            "Number of satellite galaxies with log(M*) (30pkpc) > 10.5 Msun [unitless]",
            "%3d",
        ],
        "richness_11.0": [
            halos["richness_11.0"],  # [22]
            "Number of satellite galaxies with log(M*) (30pkpc) > 11.0 Msun [unitless]",
            "%3d",
        ],
        "coolcore_flag": [
            subhalos["coolcore_flag"].astype("int32"),  # [23]
            "Cool core status, based on central cooling time (0=CC, 1=WCC, or 2=NCC) {Source: Lehle+2024}",
            "%3s",
        ],
        "coolcore_tcool": [
            subhalos["coolcore_tcool"],  # [24]
            "Central cooling time, within a 3D aperture of 0.012*r500c {Source: Lehle+2024} [Gyr]",
            "%5.2f",
        ],
        "coolcore_entropy": [
            subhalos["coolcore_entropy"],  # [25]
            "Central entropy, within a 3D aperture of 0.012*r500c {Source: Lehle+2024} [keV cm$^2$]",
            "%6.2f",
        ],
        "coolcore_ne": [
            np.log10(subhalos["coolcore_ne"]),  # [26]
            "Central electron density, within a 3D aperture of 0.012*r500c {Source: Lehle+2024} [log cm$^{-3}$]",
            "%6.2f",
        ],
        "coolcore_ne_slope": [
            subhalos["coolcore_ne_slope"],  # [27]
            "Central electron density slope (alpha), at 0.04*r500c {Source: Lehle+2024} [unitless]",
            "%5.2f",
        ],
        "coolcore_C_phys": [
            subhalos["coolcore_c_phys"],  # [28]
            "X-ray concentration parameter C_phys, at 40kpc versus 400kpc. {Source: Lehle+2024} [unitless]",
            "%4.2f",
        ],
        "coolcore_C_scaled": [
            subhalos["coolcore_c_scaled"],  # [29]
            "X-ray concentration parameter C_scaled, at 0.15*r500c versus r500c. {Source: Lehle+2024} [unitless]",
            "%4.2f",
        ],
        "perseus_like_flag": [
            halos["perseus_like_flag"],  # [30]
            "Perseus-like flag (0=no, 1=yes) {Source: Truong+2024} [unitless]",
            "%1d",
        ],
        "peakoffset_xray_x": [
            np.log10(subhalos["peakoffset_xray_x"]),  # [31]
            "X-ray peak offset in x-direction [log pkpc]",
            "%5.2f",
        ],
        "peakoffset_xray_y": [
            np.log10(subhalos["peakoffset_xray_y"]),  # [32]
            "X-ray peak offset in y-direction [log pkpc]",
            "%5.2f",
        ],
        "peakoffset_xray_z": [
            np.log10(subhalos["peakoffset_xray_z"]),  # [33]
            "X-ray peak offset in z-direction [log pkpc]",
            "%5.2f",
        ],
        "peakoffset_sz_x": [
            np.log10(subhalos["peakoffset_sz_x"]),  # [34]
            "SZ peak offset in x-direction [log pkpc]",
            "%5.2f",
        ],
        "peakoffset_sz_y": [
            np.log10(subhalos["peakoffset_sz_y"]),  # [35]
            "SZ peak offset in y-direction [log pkpc]",
            "%5.2f",
        ],
        "peakoffset_sz_z": [
            np.log10(subhalos["peakoffset_sz_z"]),  # [36]
            "SZ peak offset in z-direction [log pkpc]",
            "%5.2f",
        ],
    }

    # write (text file export)
    filename = "TNG-Cluster_Catalog.txt"

    f = open(filename, "w")
    f.write("# TNG-Cluster Catalog")
    f.write(" (see: https://www.tng-project.org/data/cluster/ and http://arxiv.org/abs/2311.06338)\n")
    f.write("# Note: this table is dynamic, and new columns are frequently added. Requests for additions welcome.\n")
    f.write("# If you use quantities in this table, please also cite the relevant paper, when indicated in a column.\n")
    f.write(f"# [{len(haloIDs)}] halos, all properties at [z=0] unless specified otherwise.\n")
    f.write("#\n")
    f.write("# columns: \n")
    for i, key in enumerate(fields):
        f.write(f"# [{i + 1}] {key}: {fields[key][1]}\n")

    f.write("#\n")

    for i in range(len(haloIDs)):
        line = ""
        for key in fields:
            val = fields[key][0][i] if key != "coolcore_flag" else cc_str[fields[key][0][i]]
            line += fields[key][2] % val + " "
        f.write(line + "\n")
    f.close()
    print(f"Wrote: [{filename}].")

    # data (hdf5 file export)
    filename = "TNG-Cluster_Catalog.hdf5"

    f = h5py.File(filename, "w")
    header = f.create_group("Header")
    header.attrs["Description"] = (
        "TNG-Cluster Catalog (see: https://www.tng-project.org/data/cluster/ and http://arxiv.org/abs/2311.06338)"
    )
    header.attrs["Note1"] = (
        "This table is dynamic, and new columns are frequently added. Requests for additions welcome."
    )
    header.attrs["Note2"] = (
        "If you use quantities in this table, please also cite the relevant paper, when indicated in a column."
    )
    header.attrs["Note3"] = f"[{len(haloIDs)}] halos, all properties at [z=0] unless specified otherwise."

    for key in fields:
        f[key] = fields[key][0]
        f[key].attrs["Description"] = fields[key][1]

    f.close()
    print(f"Wrote: [{filename}].")

    # write (zarr file export)
    filename = "TNG-Cluster_Catalog.zarr"

    z = zarr.group(filename)

    for key in fields:
        z[key] = np.array(fields[key][0])
        z[key].attrs["Description"] = fields[key][1]

    print(f"Wrote: [{filename}].")

    # write (FITS)
    filename = "TNG-Cluster_Catalog.fits"

    cols = [val[0] for _, val in fields.items()]
    descriptions = [val[1] for _, val in fields.items()]
    names = [key for key, _ in fields.items()]

    t = Table(cols, names=names, descriptions=descriptions)
    t.write(filename, format="fits", overwrite=True)

    print(f"Wrote: [{filename}].")

    # write (parquet)
    filename = "TNG-Cluster_Catalog.parquet"

    metadata = {key: val[1] for key, val in fields.items()}  # name:description pairs
    t = pyarrow.Table.from_arrays(cols, names=names, metadata=metadata)

    pyarrow.parquet.write_table(t, filename)

    print(f"Wrote: [{filename}].")

    # write (original latex table for intro paper)
    filename = "TNG-Cluster_Catalog.tex"
    f = open(filename, "w")
    for i in range(len(haloIDs)):
        line = "    %4d & %8d & " % (halos["origID"][i], haloIDs[i])
        line += "%5.2f & %5.2f & %5.3f & %5.3f & %5.2f & %5.2f & " % (
            subhalos["mhalo_200_log"][i],
            subhalos["mhalo_500_log"][i],
            subhalos["r200"][i] / 1000,
            subhalos["r500"][i] / 1000,
            subhalos["mstar_30kpc_log"][i],
            subhalos["mhi_halo_log"][i],
        )
        line += "%5.3f & %5.2f & %5.2f & " % (
            subhalos["fgas_r500"][i],
            subhalos["sfr_30pkpc_log"][i],
            subhalos["mass_smbh_log"][i],
        )
        line += "%5.2f & %5.2f & %4.2f & %3d & %3s" % (
            subhalos["xray_05_2kev_r500_halo_log"][i],
            subhalos["szy_r500c_3d_log"][i],
            subhalos["zform"][i],
            halos["richness_10.5"][i],
            cc_str[int(subhalos["coolcore_flag"][i])],
        )
        line += " \\\\\n"

        line = line.replace(" nan ", "  -- ")  # zero SFRs

        f.write(line)
    f.close()

    print(f"Wrote: [{filename}].")


def paperPlots():
    """Plots for TNG-Cluster intro paper."""
    # all analysis at z=0 unless changed below
    TNG300 = simParams(run="tng300-1", redshift=0.0)
    TNG_C = simParams(run="tng-cluster", redshift=0.0)

    sPs = [TNG300, TNG_C]

    # generate projections (done by hand)
    # from temet.catalog.maps import projections
    # for sP in sPs:
    #   projections(sP, partType='gas', partField='coldens_msunkpc2', conf=0, cenSatSelect='cen', m200_min=14.0)

    # figure 1 - mass function
    if 0:
        mass_function()
        # mass_function(secondaries=True)

    # fig 2 - virtual full box vis
    if 0:
        for conf in [0, 1, 2, 3, 4, 5, 6]:
            vis_fullbox_virtual(TNG_C, conf=conf)
        # TNG_C.setRedshift(7.0)
        # vis_fullbox_virtual(TNG_C, conf=5)

    # figure 3 - samples
    if 1:
        sample_halomasses_vs_redshift(sPs)

    # figure 4 - simulation meta-comparison
    if 0:
        simClustersComparison()

    # figures 5,6 - individual halo/gallery vis (x-ray)
    if 0:
        for conf in range(11):
            vis_gallery(TNG_C, conf=conf, num=1)  # single
        vis_gallery(TNG_C, conf=1, num=72)  # gallery
        # vis_gallery(TNG_C, conf=12, num=1) # AtLAST White Paper for Aurora/Luca
        # vis_gallery(TNG_C, conf=7, num=1) # Staffel+25 Fig 3

    # figure 7 - gas fractions
    if 0:
        gas_fraction_vs_halomass(sPs)

    # figure 8 - magnetic fields
    if 0:
        redshifts = [0.0, 1.0, 2.0]
        bfield_strength_vs_halomass(sPs, redshifts)

    # figure 9 - halo synchrotron power
    if 0:
        from temet.plot.driversObs import haloSynchrotronPower

        haloSynchrotronPower(sPs, xlim=[14.0, 15.3], ylim=[21.5, 28.5])

    # figure 10 - SZ-y and X-ray vs mass scaling relations
    if 0:
        szy_vs_halomass(sPs)
        XrayLum_vs_halomass(sPs)

    # figure 11 - X-ray scaling relations
    if 0:
        from temet.plot.driversObs import haloXrayLum

        sPs_loc = [sP.copy() for sP in sPs]
        for sP in sPs_loc:
            sP.setRedshift(0.3)  # median of data at high-mass end
        haloXrayLum(sPs_loc, xlim=[11.4, 12.7], ylim=[41.6, 45.7])

    # figure 12 - radial profiles
    if 0:
        # TNG_C.setRedshift(0.2)
        cluster_radial_profiles(TNG_C, quant="Temp")
        cluster_radial_profiles(TNG_C, quant="Metallicity")
        cluster_radial_profiles(TNG_C, quant="Metallicity", weight="_XrayWt")
        cluster_radial_profiles(TNG_C, quant="Metallicity", weight="_XrayWt_2D")
        cluster_radial_profiles(TNG_C, quant="Entropy")
        ##cluster_radial_profiles(TNG_C, quant='ne')
        ##cluster_radial_profiles(TNG_C, quant='Bmag', weight='_VolWt')

    # figure 13 - black hole mass scaling relation
    if 0:
        # from temet.plot.driversObs import blackholeVsStellarMass
        # pdf = PdfPages('blackhole_masses_vs_mstar_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        # blackholeVsStellarMass(sPs, pdf, twiceR=True, xlim=[11,13.0], ylim=[7.5,11], sizefac=0.8)
        # pdf.close()
        smbh_mass_vs_veldisp(sPs)
        smbh_mass_vs_halomass(sPs)

    # figure 14 - sfr/cold gas mass
    if 0:
        sfr_vs_halomass(sPs)
        mhi_vs_halomass(sPs)

    # figure 15 - stellar mass contents
    if 0:
        # sPs_loc = [sP.copy() for sP in sPs]
        # for sP in sPs_loc: sP.setRedshift(0.3) # median of data at high-mass end
        stellar_mass_vs_halomass(sPs, conf=0)
        stellar_mass_vs_halomass(sPs, conf=1)

    # figure 16 - satellite number profile
    if 0:
        galaxy_number_profile(TNG_C)

    # appendix - contamination
    if 0:
        contamination_mindist()

    # appendix - halo properties table
    if 0:
        halo_properties_table(TNG_C)

    # BCG SFR(z) - https://arxiv.org/pdf/2302.10943.pdf (Fig 4?)
    # and https://arxiv.org/abs/2311.04867
    # in general: redshift evolution/buildup of some of the properties?
    # satellite property profiles (radial color trends?)


# add auxcats
from temet.catalog.maps import summarize_projection_2d
from temet.load.auxcat_fields import def_fields as ac


ac["Subhalo_XrayLum_0.5-2.0kev_R500c_2D_d=r200"] = partial(
    summarize_projection_2d, quantity="xray_lum_0.5-2.0kev", projConf="2r200_d=r200", aperture="r500"
)
ac["Subhalo_XrayOffset_2D"] = partial(
    summarize_projection_2d, quantity="xray_lum_0.5-2.0kev", projConf="0.5r500_d=3r200", op="peak_offset"
)
ac["Subhalo_SZOffset_2D"] = partial(
    summarize_projection_2d, quantity="sz_yparam", projConf="0.5r500_d=3r200", op="peak_offset"
)

ac["Subhalo_SZY_R500c_2D_d=r200"] = partial(
    summarize_projection_2d, quantity="sz_yparam", projConf="2r200_d=r200", aperture="r500"
)
ac["Subhalo_SZY_R500c_2D_d=3r200"] = partial(
    summarize_projection_2d, quantity="sz_yparam", projConf="2r200_d=3r200", aperture="r500"
)
ac["Subhalo_SZY_R500c_2D"] = partial(
    summarize_projection_2d, quantity="sz_yparam", projConf="r500_d=r500", aperture="r500"
)
ac["Subhalo_SZY_R200c_2D"] = partial(
    summarize_projection_2d, quantity="sz_yparam", projConf="2r200_d=r200", aperture="r200"
)
