"""
Resonant scattering of x-ray line emission (e.g. OVII) for LEM.

https://arxiv.org/abs/2306.05453
"""

import glob
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from temet.plot.config import figsize, figsize_sm, linestyles, lw, percs, sKn, sKo
from temet.plot.util import loadColorTable
from temet.util import simParams
from temet.util.helper import dist_theta_grid, logZeroNaN, running_median
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def lemIGM():
    """Create plots for LEM proposal/STM."""
    # redshift = 0.1 # z=0.08 or z=0.035 good
    redshift = 0.07

    # Q: what can we learn from imaging the lines of the WHIM/IGM?
    # Will not measure continuum (--> cannot constrain density).
    # Measure multiple lines at the same time.
    # Options:
    #  (i) 32x32 arcmin FoV with 2ev resolution
    #  (ii) 16x16 arcmin FoV with 0.9ev resolution

    sP = simParams(run="tng300-1", redshift=redshift)

    # config
    nPixels = 2000
    axes = [0, 1]  # x,y
    labelZ = True
    labelScale = "physical"
    labelSim = True
    plotHalos = 50
    method = "sphMap"
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)
    drawFOV = 32 * 60  # arcsec

    sliceFac = 0.15

    partType = "gas"
    # panels = [{'partField':'sb_OVIII', 'valMinMax':[-18,-10]}]
    # panels = [{'partField':'sb_OVII', 'valMinMax':[-18,-10]}]
    # panels = [{'partField':'O VII', 'valMinMax':[11,16]}]
    # panels = [{'partField':'sb_CVI', 'valMinMax':[-18,-10]}]
    # panels = [{'partField':'sb_NVII', 'valMinMax':[-18,-10]}]
    # panels = [{'partField':'sb_Ne10 12.1375A', 'valMinMax':[-18,-10]}] # also: Fe XVII (neither in elInfo)
    panels = [{"partField": "coldens_msunkpc2", "valMinMax": [5, 8]}]

    class plotConfig:
        plotStyle = "open"  # open, edged
        rasterPx = nPixels  # if isinstance(nPixels,list) else [nPixels,nPixels]
        colorbars = True

        saveFilename = "./boxImage_%s-%s_z%.1f.pdf" % (sP.simName, panels[0]["partField"], redshift)

    renderBox(panels, plotConfig, locals())


def _photons_projected(sim, photons, attrs, halo):
    """Project photons along a line-of-sight and return projected (x,y) positions and luminosities.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      photons (dict): the contents of a 'photons_input' or 'photons_peeling' group from an ILTIS output file.
      attrs (dict): the attributes of the 'config' group from an ILTIS output file.
      halo (dict): the group catalog properties for this halo, from sim.halo(haloID).
    """
    # line of sight (is specified by the RT run)
    los = np.array([float(i) for i in attrs["line_of_sight"].split(" ")])
    assert np.count_nonzero(los) == 1  # xyz aligned
    peeling_index = np.where(los == 1)[0][0]  # 0=x, 1=y, 2=z

    imageplane_i1 = (1 + peeling_index) % 3
    imageplane_i2 = (2 + peeling_index) % 3

    # photon packet weights (luminosities)
    lum = photons["weight"] if "weight" in photons else photons["weight_peeling"]  # 1e42 erg/s
    lum = lum.astype("float64") * 1e42  # erg/s

    # periodic distances
    xyz = np.zeros((lum.size, 3), dtype="float32")

    if photons["shifted"]:
        # halo center moved to middle of box
        xyz[:, 0] = (photons["lspx"] - 0.5) * sim.boxSize
        xyz[:, 1] = (photons["lspy"] - 0.5) * sim.boxSize
        xyz[:, 2] = (photons["lspz"] - 0.5) * sim.boxSize
    else:
        # halo center in original global position
        xyz[:, 0] = photons["lspx"] * sim.boxSize - halo["GroupPos"][0]
        xyz[:, 1] = photons["lspy"] * sim.boxSize - halo["GroupPos"][1]
        xyz[:, 2] = photons["lspz"] * sim.boxSize - halo["GroupPos"][2]

        sim.correctPeriodicDistVecs(xyz)

    # print(f'{los = }, {peeling_index = }, {imageplane_i1 = }, {imageplane_i2 = }')

    return xyz[:, imageplane_i1], xyz[:, imageplane_i2], lum


def _sb_profile(sim, photons, attrs, halo):
    """Compute surface brightness radial profile for a given photon set.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      photons (dict): the contents of a 'photons_input' or 'photons_peeling' group from an ILTIS output file.
      attrs (dict): the attributes of the 'config' group from an ILTIS output file.
      halo (dict): the group catalog properties for this halo, from sim.halo(haloID).
    """
    # binning config
    nrad_bins = 50
    rad_minmax = [0, 500]  # pkpc

    # project photons
    x, y, lum = _photons_projected(sim, photons, attrs, halo)

    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2))  # pkpc

    # calc radial surface brightness profile
    yy = np.zeros(nrad_bins, dtype="float64")

    bin_edges = np.linspace(rad_minmax[0], rad_minmax[1], nrad_bins + 1)
    bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2  # pkpc
    bin_areas = np.pi * (bin_edges[1:] ** 2 - bin_edges[:-1] ** 2)  # pkpc^2

    for i in range(nrad_bins):
        w = np.where((dist_2d >= bin_edges[i]) & (dist_2d < bin_edges[i + 1]))
        yy[i] = lum[w].sum()  # erg/s

    sb = yy / bin_areas  # [erg/s/pkpc^2]

    return bin_mid, sb


def _sb_image(sim, photons, attrs, halo, size=250, nbins=200):
    """Compute a surface brightness image for a given photon set.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      photons (dict): the contents of a 'photons_input' or 'photons_peeling' group from an ILTIS output file.
      attrs (dict): the attributes of the 'config' group from an ILTIS output file.
      halo (dict): the group catalog properties for this halo, from sim.halo(haloID).
      size (float): half the box side-length [pkpc].
      nbins (int): total number of pixels in each dimension.
    """
    extent = [[-size, size], [-size, size]]

    # project photons
    x, y, lum = _photons_projected(sim, photons, attrs, halo)

    x = sim.units.codeLengthToKpc(x)
    y = sim.units.codeLengthToKpc(y)

    # histogram
    im, _, _ = np.histogram2d(x, y, weights=lum, bins=nbins, range=extent)

    # normalize by pixel area
    px_area = (extent[0][1] - extent[0][0]) / nbins * (extent[1][1] - extent[1][0]) / nbins  # pkpc^2

    im = im.astype("float64") / px_area  # erg/s/pkpc^2
    im = logZeroNaN(im)

    return im


def _load_data(sim, haloID, b, ver="v4", line="O--7-21.6020A"):
    """Load VoroILTIS data file for a specific halo cutout.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component to load.
      ver (str): the file prefix i.e. run set version to load.
      line (str): emission line name.
    """
    # config
    bStr = "" if b is None else "_b%s" % b
    lineStr = "" if line is None else "_%s" % line
    if b is None:
        assert ver in ["v1", "v2"]  # deprecated, b must be specified for v3 onwards
    if line is None:
        assert ver in ["v1", "v2", "v3"]  # deprecated, line must be specified for v4 onwards

    path = "/vera/ptmp/gc/byrohlc/public/OVII_RT/"
    run = "%s_cutout_%s_%d_halo%d_size2%s%s" % (ver, sim.name, sim.snap, haloID, bStr, lineStr)
    file = "data.hdf5"

    photons_input = {"shifted": False}
    photons_peeling = {"shifted": False}

    filepath = "%s%s/%s" % (path, run, file)
    filepath2 = filepath.replace(f"{ver}_", f"{ver}shifted_")
    if isfile(filepath2):
        filepath = filepath2  # use v4shifted instead of v4 if it exists

        photons_input["shifted"] = True
        photons_peeling["shifted"] = True

    if "O--8" in line:
        filepath = filepath.replace("v4_cutout", "v4-O8_cutout")

    print(filepath.replace(path, ""))

    with h5py.File(filepath, "r") as f:
        # load
        for key in f["photons_input"]:
            photons_input[key] = f["photons_input"][key][()]
        for key in f["photons_peeling_los0"]:
            photons_peeling[key] = f["photons_peeling_los0"][key][()]
        attrs = dict(f["config"].attrs)

    return photons_input, photons_peeling, attrs


def radialProfile(sim, haloID, b, line="O--7-21.6020A"):
    """RT-scattered photon datasets from VoroILTIS: surface brightness radial profile.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter(s) of the hot ISM component to load.
      line (str): emission line name.
    """
    ylim = [2e31, 1e37]

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b, line=line)

    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo["Group_R_Crit200"])
    halo_r500 = sim.units.codeLengthToKpc(halo["Group_R_Crit500"])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo["GroupFirstSub"])["SubhaloMassInRadType"][4])

    # start plot
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8, 0.2], figsize=(10.4, 8.8))

    title = r"O VII 21.6020$\rm{\AA}$"
    if line == "O--8-18.9709A":
        title = r"O VIII 18.9709$\rm{\AA}$"

    ax.set_title(r"%s (%s $\cdot$ h%d $\cdot\, \rm{M_\star = 10^{%.1f} \,M_\odot}$)" % (title, sim, haloID, mstar))
    ax.set_xlabel("Projected Distance [pkpc]")
    ax.set_yscale("log")
    ax.set_ylabel("Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")
    ax.set_xlim([-5, halo_r200 * 1.5])
    ax.set_ylim(ylim)

    ax.xaxis.set_tick_params(labelbottom=True)

    # radial profiles of intrinsic (input) versus scattered (peeling)
    rr1, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
    ax.plot(rr1, yy_intrinsic, label="Intrinsic (no RT)")

    rr2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)
    ax.plot(rr2, yy_scattered, label="Scattered (w/ RT)")

    ax.plot([halo_r200, halo_r200], ax.get_ylim(), ":", color="#aaa", label="Halo R$_{200}$", zorder=-1)
    ax.plot([halo_r500, halo_r500], ax.get_ylim(), "--", color="#aaa", label="Halo R$_{500}$", zorder=-1)

    # sub-axis: ratio
    subax.set_ylim([0.5, 400])  # 20
    if line == "O--8-18.9709A":
        subax.set_ylim([0.9, 1.5])
    subax.set_xlabel("Projected Distance [pkpc]")
    subax.set_ylabel("Ratio")
    subax.set_yscale("log")

    ratio = yy_scattered / yy_intrinsic
    assert np.array_equal(rr1, rr2)
    subax.plot(rr1, ratio, "-", color="black")

    subax.plot([halo_r200, halo_r200], subax.get_ylim(), ":", color="#aaa", zorder=-1)
    subax.plot([halo_r500, halo_r500], subax.get_ylim(), "--", color="#aaa", zorder=-1)
    for ratio in [1, 5, 10]:
        subax.plot([0, rr1.max()], [ratio, ratio], "-", lw=1.0, color="#ccc", zorder=-1)

    # finish and save plot
    ax.legend(fontsize=24, loc="upper right")
    lineStr = "_O8" if "O--8" in line else ""
    fig.savefig("sb_profile_%s_%d_h%d_b%s%s.pdf" % (sim.name, sim.snap, haloID, b, lineStr))
    plt.close(fig)


def radialProfiles(sim, haloID, b):
    """RT-scattered photon datasets from VoroILTIS: surface brightness radial profile comparison.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (list[float]): the boost parameters of the hot ISM component to load.
    """
    ylim = [2e31, 1e38]

    # load
    data = {}
    for bval in b:
        # old v3 results
        data[bval] = _load_data(sim, haloID, bval, ver="v3", line=None)

    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo["Group_R_Crit200"])
    halo_r500 = sim.units.codeLengthToKpc(halo["Group_R_Crit500"])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo["GroupFirstSub"])["SubhaloMassInRadType"][4])

    # start plot
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8, 0.3], figsize=(10.4, 9.8))

    line = r"O VII 21.6020$\rm{\AA}$"
    ax.set_title(r"%s (%s $\cdot$ h%d $\cdot\, \rm{M_\star = 10^{%.1f} \,M_\odot}$)" % (line, sim, haloID, mstar))
    ax.set_xlabel("Projected Distance [pkpc]")
    ax.set_yscale("log")
    ax.set_ylabel("Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")
    ax.set_xlim([-5, halo_r200 * 1.5])
    ax.set_ylim(ylim)

    ax.xaxis.set_tick_params(labelbottom=True)

    # radial profiles of intrinsic (input) versus scattered (peeling)
    colors = []

    for bval in b:
        photons_input, photons_peeling, attrs = data[bval]

        rr1, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
        (l,) = ax.plot(rr1, yy_intrinsic, ":", label="")
        colors.append(l.get_color())

        rr2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)
        ax.plot(rr2, yy_scattered, "-", color=l.get_color(), label="b = %g" % bval)

    ax.plot([halo_r200, halo_r200], ax.get_ylim(), ":", color="#aaa", zorder=-1)
    ax.plot([halo_r500, halo_r500], ax.get_ylim(), "--", color="#aaa", zorder=-1)

    opts = {"rotation": 90, "ha": "right", "va": "center", "fontsize": 16, "color": "#aaa"}
    ax.text(halo_r200 - 2, ylim[1] / 5, "Halo R$_{200}$", **opts)
    ax.text(halo_r500 - 2, ylim[1] / 5, "Halo R$_{500}$", **opts)

    # sub-axis: ratio
    subax.set_ylim([0.5, 100])
    subax.set_xlabel("Projected Distance [pkpc]")
    subax.set_ylabel("Ratio")
    subax.set_yscale("log")

    for i, bval in enumerate(b):
        photons_input, photons_peeling, attrs = data[bval]

        rr1, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
        rr2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)

        ratio = yy_scattered / yy_intrinsic
        assert np.array_equal(rr1, rr2)
        subax.plot(rr1, ratio, "-", color=colors[i])

    subax.plot([halo_r200, halo_r200], subax.get_ylim(), ":", color="#aaa", zorder=-1)
    subax.plot([halo_r500, halo_r500], subax.get_ylim(), "--", color="#aaa", zorder=-1)
    for ratio in [1, 5, 10]:
        subax.plot([0, rr1.max()], [ratio, ratio], "-", lw=1.0, color="#ccc", zorder=-1)

    # finish and save plot
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([0], [0], color="black", marker="", linestyle=":"),
        plt.Line2D([0], [0], color="black", marker="", linestyle="-"),
    ]
    labels += ["Intrinsic", "Scattered"]
    ax.legend(handles, labels, fontsize=22, loc="upper right")

    fig.savefig("sb_profile_%s_%d_h%d_b%s.pdf" % (sim.name, sim.snap, haloID, len(b)))
    plt.close(fig)


def stackedRadialProfiles(sim, haloIDs, b, addObsThresholds=True):
    """RT-scattered photon datasets from VoroILTIS: stacked surface brightness radial profiles.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of the halo indices to load.
      b (float): the boost parameter of the hot ISM component to load.
      addObsThresholds (bool): add observational detection thresholds to the plot.
    """
    mstarBins = [[10.0, 10.2], [10.2, 10.4], [10.4, 10.6], [10.6, 10.8], [10.8, 11.0]]
    xlim = [0, 250]  # pkpc
    ylim = [2e31, 2e36]  # erg/s/kpc^2

    # cache
    cacheFile = sim.cachePath + "iltis_profiles_%s-%d_nh%d_b%s.hdf5" % (sim.simName, sim.snap, len(haloIDs), b)

    if isfile(cacheFile):
        with h5py.File(cacheFile, "r") as f:
            rad_mid = f["rad_mid"][()]
            profiles_intr = f["profiles_intr"][()]
            profiles_scat = f["profiles_scat"][()]
            mstar = f["mstar"][()]
            assert np.array_equal(haloIDs, f["haloIDs"][()])
        print("Loaded: [%s]" % cacheFile)
    else:
        # load
        profiles_intr = []
        profiles_scat = []

        for haloID in haloIDs:
            # iltis photons
            photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

            # radial profiles
            halo = sim.halo(haloID)
            rad_mid, yy_intrinsic = _sb_profile(sim, photons_input, attrs, halo)
            rad_mid2, yy_scattered = _sb_profile(sim, photons_peeling, attrs, halo)

            assert np.array_equal(rad_mid, rad_mid2)

            profiles_intr.append(yy_intrinsic)
            profiles_scat.append(yy_scattered)

        profiles_intr = np.vstack(profiles_intr)
        profiles_scat = np.vstack(profiles_scat)

        # stellar masses
        mstar = sim.subhalos("mstar_30pkpc_log")[sim.halos("GroupFirstSub")[haloIDs]]

        # save cache
        with h5py.File(cacheFile, "w") as f:
            f["rad_mid"] = rad_mid
            f["profiles_intr"] = profiles_intr
            f["profiles_scat"] = profiles_scat
            f["haloIDs"] = haloIDs
            f["mstar"] = mstar

        print(f"Saved: [{cacheFile}]")

    # create stacks
    intr_stack = np.zeros((len(mstarBins), len(percs), profiles_intr.shape[1]), dtype="float64")
    scat_stack = np.zeros((len(mstarBins), len(percs), profiles_intr.shape[1]), dtype="float64")
    counts = np.zeros(len(mstarBins), dtype="int32")

    intr_stack.fill(np.nan)
    scat_stack.fill(np.nan)

    for i, mstarBin in enumerate(mstarBins):
        w = np.where((mstar >= mstarBin[0]) & (mstar < mstarBin[1]))[0]
        print(mstarBin, len(w))

        if len(w) == 0:
            continue

        intr_stack[i, :] = np.percentile(profiles_intr[w, :], percs, axis=0)
        scat_stack[i, :] = np.percentile(profiles_scat[w, :], percs, axis=0)
        counts[i] = len(w)

    # start plot
    figsize_loc = (figsize[0], figsize[1] * 1.3)
    fig, (ax, subax) = plt.subplots(ncols=1, nrows=2, sharex=True, height_ratios=[0.8, 0.3], figsize=figsize_loc)

    ax.set_xlabel("Projected Distance [pkpc]")
    ax.set_yscale("log")
    ax.set_ylabel("O VII(r) Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.xaxis.set_tick_params(labelbottom=True)

    # obs thresholds (see minSBtable.dat) (100 km/s linewidth, 5 sigma, wabs 3e18 galactic absorption)
    if addObsThresholds:
        # note: these are "single pointings" at the given exp time, i.e. we assume that all the pixels
        # in the entire FoV are binned together to try to make a detection. probably should change to
        # a constant size (angular or kpc) for the binning, otherwise we completely neglect the
        # spatial mapping capabilities. however, at the 1 arcmin^2 level (~20 pkpc at z=0.035), the
        # exposure times for XRISM and XIFU become ridiculous e.g. 10Ms to see anything.
        c = "#aaa"

        # XRISM-Resolve 1Ms
        xmax = 50.0
        sb_thresh = 1.1e35
        ax.plot([xlim[0], xmax], [sb_thresh, sb_thresh], lw=lw * 2, color=c)
        ax.text(xmax + 5, sb_thresh, "XRISM-Resolve (1Ms)", ha="left", va="center", color=c, fontsize=16)

        # Athena-XIFU 100ks
        xmax = 80.0
        sb_thresh = 1.3e34
        ax.plot([xlim[0], xmax], [sb_thresh, sb_thresh], lw=lw * 2, color=c)
        ax.text(xmax + 5, sb_thresh, "Athena-XIFU (100ks)", ha="left", va="center", color=c, fontsize=16)

        # Athena-XIFU 1Ms
        xmax = 150.0
        sb_thresh = 3.2e33
        ax.plot([xlim[0], xmax], [sb_thresh, sb_thresh], lw=lw * 2, color=c)
        ax.text(xmax + 5, sb_thresh, "Athena-XIFU (1Ms)", ha="left", va="center", color=c, fontsize=16)

        # LEM 1Ms
        xmax = 200.0
        sb_thresh = 1.17e33  # Gerrit Schellenberger (see emails)
        ax.plot([xlim[0], xmax], [sb_thresh, sb_thresh], lw=lw * 2, color=c)
        ax.text(xmax + 5, sb_thresh, "LEM (1Ms)", ha="left", va="center", color="#555", fontsize=16)

    # loop over each stellar mass bin
    colors = []

    for i, mstarBin in enumerate(mstarBins):
        # plot median radial profiles of intrinsic (input) versus scattered (peeling)
        label = r"%.1f < log($M_\star / \rm{M}_{\odot}$) < %.1f" % (mstarBin[0], mstarBin[1])

        (l,) = ax.plot(rad_mid, intr_stack[i, 1, :], linestyle=":", label="")
        ax.plot(rad_mid, scat_stack[i, 1, :], linestyle="-", label=label, color=l.get_color())

        colors.append(l.get_color())

        # plot percentile bands
        if i in [0, int(np.floor(len(mstarBins) / 2)), len(mstarBins) - 1]:
            ax.fill_between(rad_mid, scat_stack[i, 0, :], scat_stack[i, 2, :], color=l.get_color(), alpha=0.2)

    # sub-axis: ratio
    subax.set_ylim([0.5, 300])
    subax.set_xlabel("Projected Distance [pkpc]")
    subax.set_ylabel("Enhancement Factor")
    subax.set_yscale("log")

    # loop over each stellar mass bin
    for i in len(mstarBins):
        # plot median profile ratio
        with np.errstate(invalid="ignore"):
            ratio = scat_stack[i, 1, :] / intr_stack[i, 1, :]
        subax.plot(rad_mid, savgol_filter(ratio, sKn, sKo), "-", color=colors[i])

    for ratio in [1, 2, 5, 10, 20, 50, 100]:
        subax.plot([0, rad_mid.max()], [ratio, ratio], "-", lw=1.0, color="#ccc", zorder=-1)
        if ratio != 1:
            subax.text(5, ratio * 1.05, f"{ratio}x", ha="left", va="bottom", color="#ccc", zorder=-1)

    # main panel legend
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        plt.Line2D([0], [0], color="black", marker="", ls=":"),
        plt.Line2D([0], [0], color="black", marker="", ls="-"),
    ]
    labels += ["Intrinsic", "Scattered"]
    ax.legend(handles, labels, loc="upper right")

    # finish and save plot
    bStr = b[0] if isinstance(b, list) else b
    fig.savefig("sb_stacked_profiles_%s_%d_nh%d_b%s.pdf" % (sim.name, sim.snap, len(haloIDs), bStr))
    plt.close(fig)


def radialProfilesInput(sim, haloID):
    """Debug plot: input SB profiles of emission.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to plot. If None, all halos (i.e. all input files) are shown.
    """
    nrad_bins = 100
    rad_minmax = [0, 300]

    if haloID is None:
        path = "/u/dnelson/data/public/OVII_iltis/cutout_TNG50-1_99_halo*.hdf5"
    else:
        path = "/u/dnelson/data/public/OVII_iltis/cutout_TNG50-1_99_halo%d_*.hdf5" % haloID

    files = glob.glob(path)

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlabel("Distance [pkpc]")
    ax.set_yscale("log")
    ax.set_ylabel("Emissivity [ erg s$^{-1}$ kpc$^{-3}$ ]")

    # loop over each input file found
    for file in files:
        # load
        print(file)

        with h5py.File(file, "r") as f:
            x = f["CoordinateX"][()] * sim.boxSize  # code units
            y = f["CoordinateY"][()] * sim.boxSize  # code units
            z = f["CoordinateZ"][()] * sim.boxSize  # code units
            emis = f["Emissivity"][()].astype("float64") * 1e42  # erg/s

        rad = np.sqrt((x - x.mean()) ** 2 + (y - y.mean()) ** 2 + (z - z.mean()) ** 2)
        rad = sim.units.codeLengthToKpc(rad)

        # calc radial 3D surface brightness profile
        yy = np.zeros(nrad_bins, dtype="float64")

        bin_edges = np.linspace(rad_minmax[0], rad_minmax[1], nrad_bins + 1)
        bin_mid = (bin_edges[1:] + bin_edges[:-1]) / 2  # pkpc
        bin_vol = 4 / 3 * np.pi * (bin_edges[1:] ** 3 - bin_edges[:-1] ** 3)  # pkpc^3

        for i in range(nrad_bins):
            w = np.where((rad >= bin_edges[i]) & (rad < bin_edges[i + 1]))
            yy[i] = emis[w].sum()  # erg/s

        sb = yy / bin_vol  # erg/s/pkpc^3

        # plot
        ax.plot(bin_mid, sb, "-", label=file.rsplit("/", 1)[1])

    # finish plot
    ax.legend(loc="upper right")
    fig.savefig("input_profiles_%s_%d.pdf" % (sim.name, sim.snap))
    plt.close(fig)


def imageSBcomp(sim, haloID, b, line="O--7-21.6020A"):
    """RT-scattered photon datasets from VoroILTIS: surface brightness image, intrinsic vs scattered.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
      line (str): emission line name.
    """
    # config
    size = 250  # pkpc

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b, line=line)

    halo = sim.halo(haloID)

    halo_r200 = sim.units.codeLengthToKpc(halo["Group_R_Crit200"])
    halo_r500 = sim.units.codeLengthToKpc(halo["Group_R_Crit500"])
    mstar = sim.units.codeMassToLogMsun(sim.subhalo(halo["GroupFirstSub"])["SubhaloMassInRadType"][4])

    circOpts = {"color": "#fff", "alpha": 0.2, "linewidth": 2.0, "fill": False}
    vmm = [27, 37]  # log(erg/s/kpc^2)
    extent = [-size, size, -size, size]

    # start plot
    fig, (ax_left, ax_mid, ax_right) = plt.subplots(ncols=3, nrows=1, figsize=(figsize[0] * 2.0, figsize[1] * 0.85))

    # left: intrinsic
    ax_left.set_title("Intrinsic (no RT)")
    ax_left.set_xlabel(r"$\rm{\Delta\,x}$ [pkpc]")
    ax_left.set_ylabel(r"$\rm{\Delta\,y}$ [pkpc]")

    im_intrinsic = _sb_image(sim, photons_input, attrs, halo, size=size)
    im_left = ax_left.imshow(im_intrinsic, cmap="inferno", extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

    ax_left.add_artist(plt.Circle((0, 0), halo_r200, **circOpts))
    ax_left.add_artist(plt.Circle((0, 0), halo_r500, **circOpts))

    title = r"O VII 21.6020$\rm{\AA}$"
    if line == "O--8-18.9709A":
        title = r"O VIII 18.9709$\rm{\AA}$"

    s = r"%s\n%s\nHaloID %d\n$\rm{M_\star = 10^{%.1f} \,M_\odot}$" % (title, sim, haloID, mstar)
    ax_left.text(0.03, 0.03, s, ha="left", va="bottom", color="#fff", alpha=0.5, transform=ax_left.transAxes)

    cax = make_axes_locatable(ax_left).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(im_left, cax=cax)
    cb.ax.set_ylabel("Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")

    # middle: scattered
    ax_mid.set_title("Scattered (w/ RT)")
    ax_mid.set_xlabel(r"$\rm{\Delta\,x}$ [pkpc]")
    ax_mid.set_ylabel(r"$\rm{\Delta\,y}$ [pkpc]")

    im_scattered = _sb_image(sim, photons_peeling, attrs, halo, size=size)
    im_mid = ax_mid.imshow(im_scattered, cmap="inferno", extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

    ax_mid.add_artist(plt.Circle((0, 0), halo_r200, **circOpts))
    ax_mid.add_artist(plt.Circle((0, 0), halo_r500, **circOpts))

    cax = make_axes_locatable(ax_mid).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(im_mid, cax=cax)
    cb.ax.set_ylabel("Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")

    # right: ratio
    ax_right.set_title("Ratio (Scattered / Intrinsic)")
    ax_right.set_xlabel(r"$\rm{\Delta\,x}$ [pkpc]")
    ax_right.set_ylabel(r"$\rm{\Delta\,y}$ [pkpc]")

    vmm = [-1.0, 1.0]
    if line == "O--8-18.9709A":
        vmm = [-0.1, 0.1]

    im_ratio = np.log10(10.0**im_scattered / 10.0**im_intrinsic)
    im_right = ax_right.imshow(im_ratio, cmap="coolwarm", extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

    circOpts["color"] = "#000"
    ax_right.add_artist(plt.Circle((0, 0), halo_r200, **circOpts))
    ax_right.add_artist(plt.Circle((0, 0), halo_r500, **circOpts))

    cax = make_axes_locatable(ax_right).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(im_right, cax=cax)
    cb.ax.set_ylabel("Surface Brightness Ratio [ log ]")

    # finish and save plot
    lineStr = "_O8" if "O--8" in line else ""
    fig.savefig("sb_image_%s_%d_h%d_b%s%s.pdf" % (sim.name, sim.snap, haloID, b, lineStr))
    plt.close(fig)


def imageSBgallery(sim, haloIDs, b):
    """RT-scattered photon datasets from VoroILTIS: gallery of surface brightness images.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): the halo indices to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    size = 250  # pkpc
    scalebar_size = 100  # pkpc

    circOpts = {"color": "#fff", "alpha": 0.2, "linewidth": 2.0, "fill": False}
    vmm = [31, 36]  # log(erg/s/kpc^2)

    ncols = 3
    cbar_height = 0.07  # fraction

    extent = [-size, size, -size, size]

    # start plot
    nrows = int(np.ceil(len(haloIDs) / ncols))
    fig = plt.figure(figsize=(18, 18 * (nrows + cbar_height + 0.1) / ncols), layout="constrained")

    # fig, axes = plt.subplots(ncols=ncols, nrows=nrows, gridspec_kw={'bottom':0.1}, figsize=(20, 20*nrows/ncols*1.1))
    # axes = [ax for subaxes_list in axes for ax in subaxes_list] # flatten
    gs = fig.add_gridspec(nrows + 1, ncols, hspace=0.01, wspace=0.01, height_ratios=[1] * nrows + [cbar_height])
    axes = [fig.add_subplot(gs[i]) for i in range(len(haloIDs))]

    # loop over each requested halo
    for haloID, ax in zip(haloIDs, axes):
        # load metadata
        halo = sim.halo(haloID)
        subhalo = sim.subhalo(halo["GroupFirstSub"])

        halo_r200 = sim.units.codeLengthToKpc(halo["Group_R_Crit200"])
        halo_r500 = sim.units.codeLengthToKpc(halo["Group_R_Crit500"])
        mstar = sim.units.codeMassToLogMsun(subhalo["SubhaloMassInRadType"][4])
        sfr = subhalo["SubhaloSFRinRad"]

        # cache
        cacheFile = sim.cachePath + "iltis_sbimage_%s-%d_%d_b%s_s%d.hdf5" % (sim.simName, sim.snap, haloID, b, size)

        if isfile(cacheFile):
            with h5py.File(cacheFile, "r") as f:
                im_scattered = f["im_scattered"][()]
        else:
            # load and create SB image
            _, photons_peeling, attrs = _load_data(sim, haloID, b)
            im_scattered = _sb_image(sim, photons_peeling, attrs, halo, size=size)

            with h5py.File(cacheFile, "w") as f:
                f["im_scattered"] = im_scattered

        # show SB image
        im_left = ax.imshow(im_scattered, cmap="inferno", extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

        # add r200 and r500 circles
        ax.add_artist(plt.Circle((0, 0), halo_r200, **circOpts))
        ax.add_artist(plt.Circle((0, 0), halo_r500, **circOpts))

        # add text
        s = r"%s HaloID %d $\rm{M_\star = 10^{%.1f} \,M_\odot}$" % (sim, haloID, mstar)
        s += r" SFR = %.1f $\rm{M_\odot yr^{-1}}$" % sfr
        ax.text(0.03, 0.03, s, ha="left", va="bottom", color="#fff", alpha=0.5, transform=ax.transAxes)

        # disable ticks and add scalebar
        ax.set_aspect("equal")
        ax.set_yticks([])
        ax.set_xticks([])

        ax.plot([-size + 10, -size + scalebar_size + 10], [size - 10, size - 10], "-", color="#fff", alpha=0.7)
        opts = {"ha": "center", "va": "top", "color": "#fff", "alpha": 0.7}
        ax.text(-size + 10 + scalebar_size / 2, size - 20, "%d kpc" % scalebar_size, **opts)

    # cax = fig.add_axes([0.3,0.05,0.4,0.03]) # xmin, ymin, width, height
    cax = fig.add_subplot(gs[-1, 1:-1])
    cb = plt.colorbar(im_left, orientation="horizontal", cax=cax)
    cb.ax.set_xlabel("OVII(r) Scattered Surface Brightness [ erg s$^{-1}$ kpc$^{-2}$ ]")

    # finish and save plot
    fig.savefig("sb_gallery_%s_%d_nh%d_b%s.pdf" % (sim.name, sim.snap, len(haloIDs), b))
    plt.close(fig)


def spectrum(sim, haloID, b):
    """RT-scattered photon datasets from VoroILTIS: line emission spectrum.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    radbin = [30, 50]  # pkpc
    nspecbins = 50

    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)

    # start plot
    fig, ax = plt.subplots()

    line = r"O VII 21.6020$\rm{\AA}$"
    ax.set_title(r"%s (%s $\cdot$ HaloID %d) (%d < R/kpc < %d)" % (line, sim, haloID, radbin[0], radbin[1]))
    ax.set_xlabel(r"Offset from Line Center $\rm{\Delta \lambda} \ [ \AA ]}$")
    # ax.set_xlabel('Offset from Line Center $\\rm{\Delta E} \ [ keV ]}$')
    ax.set_ylabel(r"Spectrum [ erg s$^{-1}$ $\rm{\AA}^{-1}$ ]")
    # ax.set_yscale('log')

    # loop for intrinsic vs. scattered
    for photons, label in zip([photons_input, photons_peeling], ["Intrinsic", "Scattered"]):
        # project photons
        x, y, lum = _photons_projected(sim, photons, attrs, halo)

        dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2))  # pkpc

        # restrict to radial projected aperture, and compute 'total' spectrum
        w_rad = np.where((dist_2d >= radbin[0]) & (dist_2d < radbin[1]))

        spec, spec_bins = np.histogram(photons["lambda"][w_rad], weights=lum[w_rad], bins=nspecbins)

        spec_mid = (spec_bins[1:] + spec_bins[:-1]) / 2
        spec_dwave = spec_bins[1] - spec_bins[0]
        spec /= spec_dwave  # erg/s -> erg/s/Ang

        # convert dAng to dKev for the x-axis
        # spec_dKeV = sim.units.hc_kev_ang / (wave0 + spec_mid)

        # plot
        ax.plot(spec_mid, spec, "-", label=label)

    # finish and save plot
    ax.legend(loc="upper right")
    fig.savefig("spec_%s_%d_h%d_b%s.pdf" % (sim.name, sim.snap, haloID, b))
    plt.close(fig)


def galaxyLum(sim, haloID, b, aperture_kpc=10.0):
    """Compute (total) luminosity (within some aperture) for scattered photon datasets.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloID (int): the halo index to load.
      b (float): the boost parameter of the hot ISM component.
      aperture_kpc (float): the radial aperture within which to sum luminosity [pkpc].
    """
    # load
    photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

    halo = sim.halo(haloID)
    # subhalo = sim.subhalo(halo['GroupFirstSub'])
    # mstar = sim.units.codeMassToLogMsun(subhalo['SubhaloMassInRadType'][4])[0]
    # sfr = subhalo['SubhaloSFRinRad']

    # intrinsic: project and 2d distances, sum for lum
    x, y, lum = _photons_projected(sim, photons_input, attrs, halo)
    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2))  # pkpc
    w = np.where(dist_2d <= aperture_kpc)

    tot_lum_int = lum[w].sum()

    # scattered
    x, y, lum = _photons_projected(sim, photons_peeling, attrs, halo)
    dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2))  # pkpc
    w = np.where(dist_2d <= aperture_kpc)

    tot_lum_scat = lum[w].sum()

    # note: https://academic.oup.com/mnras/article/356/2/727/1159998 for NGC 7213
    # (a S0 at D=23 Mpc, w/ an AGN Lbol=1.7e43 erg/s, strong outflow, MBH ~ 1e8 Msun, lambda_edd ~ 1e-3)
    # (SFR = 1.0 +/- 0.1 Msun/yr from Gruppioni+2016)
    # MBH vs M* scaling gives M* between 10-11 and median at 10.5 Msun
    # flux = 25e-6 phot/cm^2/s gives a OVIIr 21.6A luminosity of 1.5e+39 erg/s

    # note: https://www.aanda.org/articles/aa/abs/2007/21/aa6340-06/aa6340-06.html for NGC 253
    # (a SAB starburst, like M82, at a D=3.94 Mpc, z=0.000864) (M* ~ 4e10 Msun, Bailin+11)
    # summing up fluxes across all 4 spatial regions (Table 2), flux = 5.3e-6 cm^-2 s^-1 (assume phot cm^-2 s^-1)
    # times 9.2e-10 erg/phot (for OVIIr) = 4.9e-15 erg/s/cm^2
    # gives a OVIIr 21.6A luminosity of 9.0e36 erg/s (this is a hot superwind outflow within <~= 5 kpc)

    # note: https://ui.adsabs.harvard.edu/abs/2012MNRAS.420.3389L/abstract sample of 9 nearby star-forming galaxies
    # names = ['NGC253A','M51','M94','M83','NGC2903','M61','NGC4631','Antennae','NGC253B','M82A','M82B','M82C']
    # fluxes_o7r = [0.9, 1.1, 1.8, 1.3, 1.3, 1.1, 0.8, 0.5, 0.4, 1.5, 1.1, 2.1] * 1e-5 photons/s/cm^2
    #            = [8.28e-15, 1.01e-14, 1.66e-14, 1.20e-14, 1.20e-14, 1.01e-14, 7.36e-15, 4.60e-15, 3.68e-15,
    #               1.38e-14, 1.01e-14, 1.93e-14] erg/s/cm^2
    # distances = [3.2, 8.0, 5.0, 4.7, 9.4, 12.1, 6.7, 21.6, 3.2, 3.9, 3.9, 3.9] Mpc
    # gives OVIIr 21.6A luminosities = [1.0e37, 7.8e37, 5.0e37, 3.2e37, 1.3e38, 1.8e38, 4.0e37, 2.6e38, 4.5e36,
    #                                   2.5e37, 1.8e37, 3.5e37] erg/s

    # print(f'{haloID = } {mstar = :.1f} {sfr = :.1f} {b = } {tot_lum_int = :g} [erg/s], {tot_lum_scat = :g} [erg/s]')
    return tot_lum_int, tot_lum_scat


def _get_subhalo_sample(sim, mstar_minmax=(10.0, 11.0), verbose=False):
    """Define the main sample for ILTIS processing."""
    from temet.cosmo.util import subsampleRandomSubhalos

    # load
    mstar = sim.subhalos("mstar_30pkpc_log")
    cen_flag = sim.subhalos("cen_flag")

    grnr = sim.subhalos("SubhaloGrNr")

    # select
    w_iltis = np.where((mstar > mstar_minmax[0]) & (mstar <= mstar_minmax[1]) & cen_flag)

    # sub-sample for at most N systems per 0.1 dex stellar mass bin
    inds_iltis, _ = subsampleRandomSubhalos(sim, maxPerDex=300, mstarMinMax=mstar_minmax, mstar=mstar[w_iltis])
    subinds_iltis = w_iltis[0][inds_iltis]

    grnr_iltis = grnr[subinds_iltis]

    # sample selection in mstar for ILTIS runs
    if verbose:
        hID_min, hID_max = grnr[w_iltis].min(), grnr[w_iltis].max()
        shID_min, shID_max = grnr[subinds_iltis].min(), grnr[subinds_iltis].max()
        print(f"found [{len(w_iltis[0])}] galaxies in mass range, haloIDs spanning [{hID_min} - {hID_max}]")
        print(f"sub-selected to [{len(subinds_iltis)}] galaxies, subhaloIDs spanning [{shID_min} - {shID_max}]")

        # print(repr(grnr_iltis))

    return mstar, cen_flag, w_iltis, subinds_iltis, grnr_iltis


def galaxyLumVsSFR(sim, b=1, addDiffuse=True, correctLineToBandFluxRatio=False):
    """Test the hot ISM emission model by comparing to observational scaling relations.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      b (float): the boost parameter of the hot ISM component.
      addDiffuse (bool): in addition to the hot ISM component, add the non-starforming (i.e. diffuse) emission.
      correctLineToBandFluxRatio (bool): correct the Mineo+12 0.5-2 keV downwards to account for the fact
        that OVII(r) is only a (uncertain) fraction of this broadband luminosity
    """
    from matplotlib.patches import FancyArrowPatch

    # load catalog of OVII luminosities, per subhalo, star-forming gas only (<10 kpc or <30 kpc)
    acField = "Subhalo_OVIIr_GalaxyLum_1rstars"  # 1rstars, 10pkpc, 30pkpc
    ac = sim.auxCat(acField)
    lum = ac[acField].astype("float64") * 1e30  # unit conversion

    # apply boost factor (this auxCat has only SFR>0 gas, so we directly, and only, modify the lum2phase modeled gas)
    lum *= b

    if addDiffuse:
        # add contributions from all non-starforming gas (<10 kpc or <30 kpc)
        print("Adding diffuse contribution within 10pkpc to star-forming lum2phase.")
        acField = "Subhalo_OVIIr_DiffuseLum_1rstars"  # 1rstars, 10pkpc, 30pkpc
        ac = sim.auxCat(acField)
        lum_diffuse = ac[acField].astype("float64") * 1e30  # unit conversion

        lum += lum_diffuse

    # sample selection in mstar and SFR for plot
    sfr_min = 1e-2
    mstar_min = 10.0
    mstar_max = 11.0

    mstar, cen_flag, w_iltis, subinds_iltis, _ = _get_subhalo_sample(sim, mstar_min, mstar_max, verbose=True)

    sfr = sim.subhalos("sfr2")

    w = np.where((mstar > mstar_min) & (mstar <= mstar_max) & (sfr >= sfr_min) & cen_flag)

    print(f"{sim}: found [{len(w[0])}] galaxies with ({mstar_min:.1f} < M* < {mstar_max:.1f}) and (SFR > {sfr_min:g})")

    # start plot
    fig, ax = plt.subplots(figsize=figsize_sm)

    ax.set_xlabel("Galaxy SFR [ $\\rm{M_{sun}}$ yr$^{-1}$ ]")
    ax.set_ylabel("Galaxy OVII(r) Luminosity [ erg s$^{-1}$ ]")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim([1e36, 1e41])

    cmap = loadColorTable("viridis", fracSubset=[0.15, 0.95])
    s = ax.scatter(sfr[w], lum[w], c=mstar[w], cmap=cmap, vmin=mstar_min, vmax=mstar_max)

    # Mineo+ (2012) observed relation for L_{0.5-2keV} vs SFR, i.e. an upper limit on L_OVIIr vs SFR
    xx = np.linspace(0.1, 20, 50)
    yy = 8.3e38 * xx
    yy2 = 5.2e38 * xx

    # Mineo+ (2012) data points (Figure 6 left panel)
    # fmt: off
    mineo_sfr = np.array([0.08, 0.09, 0.17, 0.18, 0.29, 0.29, 0.38, 0.44, 1.83, 1.84, 3.05, 3.76, 4.09, 4.60, 5.29,
                          11.46, 14.65, 5.99, 5.36, 7.08, 16.74])
    mineo_lum = np.array([ 1.06, 0.67, 0.30, 0.38, 0.59, 1.14, 2.28, 3.11, 13.14, 16.42, 11.62, 44.10, 38.69, 33.31,
                          53.65, 41.95, 59.29, 90.14, 148.58, 212.43, 254.49])
    # fmt: on

    if correctLineToBandFluxRatio:
        # take approximate fraction of OVII(r) luminosity to total 0.5-2.0 keV luminosity as a correction
        # factor, to convert our L_OVII(r) output into a L_0.5-2.0keV, for comparison with the data
        # Q: what fraction of the 0.5-2KeV lum comes from OVIIr?
        # --> actually a lot! depends on density and temp.
        # --> for n=-2.0 and denser, ~25% (at 5.8 < logT[K] < 6.25), dropping to 5% at 10^5.6K and 10^6.5K
        # note: the median/mean density of star-forming gas is about 5-20x the threshold density in TNG50-1 MW halos
        # --> eEOS hot-phase temperature is actually within the range of peak OVII(r) fraction
        line_ratio_fac = 0.1

        print(f"Correcting Mineo+ 0.5-2 keV data to OVII(r) line luminosity with {line_ratio_fac = }")

        yy *= line_ratio_fac
        yy2 *= line_ratio_fac
        mineo_lum *= line_ratio_fac

    ax.plot(xx, yy, "--", color="#000", label="Mineo+12 $\\rm{L_{0.5-2keV}}$-derived relation")
    ax.plot(xx, yy2, "--", color="#555", label="Mineo+12 mekal best-fit relation")

    ax.plot(mineo_sfr, np.array(mineo_lum) * 1e38, "D", color="#000", label="Mineo+12 galaxies")

    # Salvestrini+2020 NGC 7213 L_OVIIr = 1.9e+39 erg/s and SFR = 1.0 +/- 0.1 Msun/yr (Gruppioni+2016)
    ax.errorbar(1.0, 1.9e39, xerr=0.2, yerr=1.0e39, marker="o", markersize=10.0, color="black", label="NGC 7213")

    # add arrows to indicate possible rescalings
    color = "#ccc"
    arrowstyle = "simple, head_width=8, head_length=8, tail_width=2"
    textOpts = {"color": color, "rotation": 90.0, "fontsize": 15, "ha": "center", "va": "top"}

    if correctLineToBandFluxRatio:
        # 1% arrow is downwards, 25% arrow is upwards
        y = 1.5e39
        x = 22
        f = 2.5  # 2.5x higher than fiducial i.e. 25%
        p1 = FancyArrowPatch(posA=[x, y], posB=[x, y * f], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p1)
        ax.text(x, y * f * 2.4, "0.25", **textOpts)

        x = 22
        f = 0.1  # 10% lower than fiducial i.e. 1%
        p2 = FancyArrowPatch(posA=[x, y], posB=[x, y * f], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p2)
        ax.text(x, y * f * 0.9, "0.01", **textOpts)
    else:
        # all arrows are downwards
        y = 3.5e39
        x = 12
        f = 0.25
        p1 = FancyArrowPatch(posA=[x, y], posB=[x, y * f], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p1)
        ax.text(x, y * f * 0.9, "%4.2f" % f, **textOpts)

        x = 16.5
        f = 0.10
        p2 = FancyArrowPatch(posA=[x, y], posB=[x, y * f], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p2)
        ax.text(x, y * f * 0.9, "%4.2f" % f, **textOpts)

        x = 22
        f = 0.01
        p2 = FancyArrowPatch(posA=[x, y], posB=[x, y * f], arrowstyle=arrowstyle, alpha=1.0, color=color)
        ax.add_artist(p2)
        ax.text(x, y * f * 0.9, "%4.2f" % f, **textOpts)

    # colobar and save plot
    ax.legend(loc="upper left")
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel(r"Galaxy Stellar Mass [ $\rm{M_{sun}}$ ]")

    fig.savefig("galaxy_OVIIr_lum_vs_SFR_%s_%d_b%s.pdf" % (sim.name, sim.snap, b))
    plt.close(fig)

    # iltis sample plot
    fig, ax = plt.subplots()

    ax.set_xlabel(r"Galaxy Stellar Mass [ $\rm{M_{sun}}$ ]")
    ax.set_ylabel("Galaxy OVII(r) Luminosity [ erg s$^{-1}$ ]")
    ax.set_yscale("log")
    ax.set_xlim([mstar_min - 0.02, mstar_max + 0.02])

    s = ax.scatter(mstar[w_iltis], lum[w_iltis], label="Parent Sample")

    ax.plot([mstar_min, mstar_min], ax.get_ylim(), ":", alpha=0.6, color="black")
    ax.plot([mstar_max, mstar_max], ax.get_ylim(), ":", alpha=0.6, color="black")

    ax.plot(
        mstar[subinds_iltis],
        lum[subinds_iltis],
        linestyle="none",
        color="black",
        marker="o",
        markersize=4.0,
        label="ILTIS sample",
    )

    # finish plot
    ax.legend(loc="upper left")
    fig.savefig("galaxy_OVIIr_lum_vs_mstar_%s_%d_b%s.pdf" % (sim.name, sim.snap, b))
    plt.close(fig)


def enhancementVsMass(sim, haloIDs, b, rad=4, color_quant="sfr", median=False, pxRatios=True):
    """Derive and plot SB enhancement factor as a function of mass and radial ranges, coloring by other quantities.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of the halo indices to load.
      b (float): the boost parameter of the hot ISM component.
      rad (int): which of the five radial ranges to use.
      color_quant (str): which galaxy property to color points by, 'sfr', 'LOVII', 'Lbol', or 'm200'.
      median (bool): plot the median (across pixels) SB ratio, instead of the mean (default).
      pxRatios (bool): if True, define enhancement factor as the mean or median of the ratio of
        scattered to intrinsic pixel SB values, for a given radial range. If False, instead define
        as the ratio of (a) the mean or median scattered SB values for all pixels in a given radial
        range, to (b) the mean or median intrinsic SB values for the same pixels.
    """
    # config
    xlim = [10.0, 11.0]  # log msun

    # config - must recompute cache
    size = 250.0  # pkpc for imaging
    lum_aperture_kpc = 10.0  # pkpc for L_OVIIr calculation

    # calculate enhancement factor
    cacheFile = sim.cachePath + "iltis_enhancefac_%s-%d_nh%d_b%s.hdf5" % (sim.simName, sim.snap, len(haloIDs), b)

    subhaloIDs = sim.halos("GroupFirstSub")[haloIDs]

    if isfile(cacheFile):
        with h5py.File(cacheFile, "r") as f:
            assert np.array_equal(haloIDs, f["haloIDs"][()])
            # enhancement factors
            enhancement_mean = f["enhancement_mean"][()]
            enhancement_percs = f["enhancement_percs"][()]
            enhancement2_mean = f["enhancement2_mean"][()]
            enhancement2_percs = f["enhancement2_percs"][()]
            # sb values
            sb_scattered_mean = f["sb_scattered_mean"][()]
            sb_scattered_percs = f["sb_scattered_percs"][()]
            # galaxy propreties
            tot_lum_scattered = f["tot_lum_scattered"][()]
            mstar = f["mstar"][()]
            sfr = f["sfr"][()]
            lbol = f["lbol"][()]
            mbh = f["mbh"][()]
            m200 = f["m200"][()]

        print("Loaded: [%s]" % cacheFile)
    else:
        # loop over all halos
        n_radranges = 5  # actual definitions hard-coded below

        # enhancements as mean or median of all ratio pixels in a given radial range
        enhancement_mean = np.zeros((len(haloIDs), n_radranges), dtype="float32")
        enhancement_percs = np.zeros((len(haloIDs), n_radranges, len(percs)), dtype="float32")

        # enhancements as ratio of the mean or median of all pixels in a given radial range
        enhancement2_mean = np.zeros((len(haloIDs), n_radranges), dtype="float32")
        enhancement2_percs = np.zeros((len(haloIDs), n_radranges, len(percs)), dtype="float32")

        tot_lum_scattered = np.zeros((len(haloIDs)), dtype="float64")
        sb_scattered_mean = np.zeros((len(haloIDs), n_radranges), dtype="float64")
        sb_scattered_percs = np.zeros((len(haloIDs), n_radranges, len(percs)), dtype="float64")

        for i, haloID in enumerate(haloIDs):
            # iltis photons
            photons_input, photons_peeling, attrs = _load_data(sim, haloID, b)

            # SB images
            halo = sim.halo(haloID)
            r200_kpc = sim.units.codeLengthToKpc(halo["Group_R_Crit200"])
            r500_kpc = sim.units.codeLengthToKpc(halo["Group_R_Crit500"])

            # has some dependence on pixel size, adopt LEM resolution at z=0.01
            lem_res_arcsec = 15.0
            lem_res_arcsec *= 4  # to minimize MC shot noise, bring medians up to reasonable values
            lem_res_kpc_at_z0p01 = sim.units.arcsecToAngSizeKpcAtRedshift(lem_res_arcsec, z=0.01)
            nbins = int(2 * size / lem_res_kpc_at_z0p01)

            im_int = 10.0 ** (_sb_image(sim, photons_input, attrs, halo, size=size, nbins=nbins))
            im_scat = 10.0 ** (_sb_image(sim, photons_peeling, attrs, halo, size=size, nbins=nbins))
            im_ratio = im_scat / im_int  # linear ratio

            dist, theta = dist_theta_grid(2 * size, [nbins, nbins])  # 2*size for convention

            for j in range(5):
                # radial range selection
                if j == 0:
                    # radial range #1 = whole halo (r=0 to r=r200)
                    w_px = np.where((dist > 0) & (dist < r200_kpc))

                if j == 1:
                    # radial range #2 = CGM (r=20 kpc to r=200 kpc)
                    w_px = np.where((dist >= 20) & (dist < 200))

                if j == 2:
                    # radial range #3 = outer CGM (r=50 kpc to r=200 kpc)
                    w_px = np.where((dist >= 50) & (dist < 200))

                if j == 3:
                    # radial range #4 = at r500 (0.95 < r/r500 < 1.05)
                    w_px = np.where((dist >= 0.95 * r200_kpc) & (dist < 1.05 * r200_kpc))

                if j == 4:
                    # radial range #5 = at r200 (0.9 < r/r200 < 1.1)
                    w_px = np.where((dist >= 0.9 * r500_kpc) & (dist < 1.1 * r500_kpc))

                # compute ratio (sky area-weighted, SB ratio, of scattered to intrinsic)
                enhancement_mean[i, j] = np.mean(im_ratio[w_px])
                enhancement_percs[i, j, :] = np.percentile(im_ratio[w_px], percs)

                # compute ratio
                enhancement2_mean[i, j] = np.mean(im_scat[w_px]) / np.mean(im_int[w_px])
                enhancement2_percs[i, j, :] = np.percentile(im_scat[w_px], percs) / np.percentile(im_int[w_px], percs)

                # save actual SB values
                sb_scattered_mean[i, j] = np.mean(im_scat[w_px])
                sb_scattered_percs[i, j, :] = np.percentile(im_scat[w_px], percs)

            # compute galaxy luminosity in this line
            x, y, lum = _photons_projected(sim, photons_peeling, attrs, halo)
            dist_2d = sim.units.codeLengthToKpc(np.sqrt(x**2 + y**2))  # pkpc
            w = np.where(dist_2d <= lum_aperture_kpc)

            tot_lum_scattered[i] = lum[w].sum()

        # load galaxy properties
        mstar = sim.subhalos("mstar_30pkpc_log")[subhaloIDs]
        sfr = sim.subhalos("sfr_30pkpc_log")[subhaloIDs]
        lbol = sim.auxCat("Subhalo_BH_BolLum_largest")["Subhalo_BH_BolLum_largest"][subhaloIDs]
        mbh = sim.auxCat("Subhalo_BH_Mass_largest")["Subhalo_BH_Mass_largest"][subhaloIDs]
        m200 = sim.subhalos("m200_log")[subhaloIDs]

        # save cache
        with h5py.File(cacheFile, "w") as f:
            f["haloIDs"] = haloIDs
            # enhancement factors
            f["enhancement_mean"] = enhancement_mean
            f["enhancement_percs"] = enhancement_percs
            f["enhancement2_mean"] = enhancement2_mean
            f["enhancement2_percs"] = enhancement2_percs
            # sb values
            f["sb_scattered_mean"] = sb_scattered_mean
            f["sb_scattered_percs"] = sb_scattered_percs
            # galaxy properties
            f["tot_lum_scattered"] = tot_lum_scattered
            f["sfr"] = sfr
            f["mstar"] = mstar
            f["lbol"] = lbol
            f["mbh"] = mbh
            f["m200"] = m200

        print(f"Saved: [{cacheFile}]")

    # start plot
    figsize_loc = figsize if color_quant == "sfr" else [6.7, 4.8]  # paper Fig 5 setup
    fig, ax = plt.subplots(figsize=figsize_loc)

    labels = [
        r"$\rm{R < R_{200}}$",
        "20 < R [kpc] < 200",
        "50 < R [kpc] < 200",
        r"$\rm{R = R_{500}}$",
        r"$\rm{R = R_{200}}$",
    ]

    ax.set_xlabel(r"Galaxy Stellar Mass [ log $\rm{M_{sun}}$ ]")
    ax.set_ylabel("%s SB Enhancement (%s)" % ("Median" if median else "Mean", labels[rad]))
    ax.set_yscale("log")
    ax.set_xlim(xlim)

    if color_quant != "sfr":
        ax.set_ylabel("%s Enhancement" % ("Median" if median else "Mean"))

    # select color quantity
    cmap = loadColorTable("plasma", fracSubset=[0.1, 0.9])

    assert color_quant in ["sfr", "LOVII", "Lbol", "mbh", "m200", "meansb_r200"]
    if color_quant == "sfr":
        cvals = sfr
        cminmax = [-1.0, 1.0]  # log msun/yr
        clabel = r"Star Formation Rate [ log $\rm{M_{sun}}$ yr$^{-1}$ ]"
    if color_quant == "LOVII":
        cvals = np.log10(tot_lum_scattered)
        cminmax = [37, 40]
        clabel = r"Galaxy $\rm{L_{OVII(r)}}$ [ log erg s$^{-1}$ ]"
    if color_quant == "Lbol":
        cvals = np.log10(lbol)
        cminmax = [37, 43]
        clabel = r"SMBH $\rm{L_{bol}}$ [ log erg s$^{-1}$ ]"
    if color_quant == "mbh":
        cvals = sim.units.codeMassToLogMsun(mbh)
        cminmax = [7.5, 8.5]
        clabel = r"SMBH Mass [ log $\rm{M_\odot}$ ]"
    if color_quant == "m200":
        cvals = m200
        cminmax = [11.4, 12.6]
        clabel = r"Halo Mass $\rm{M_{200c}}$ [ log $\rm{M_\odot}$ ]"
    if color_quant == "meansb_r200":
        cvals = np.log10(sb_scattered_mean[:, 4])
        cminmax = [30.5, 33]
        clabel = r"$<\rm{SB}>$ [ log erg s$^{-1}$ kpc$^{-2}$ ]"

    # select which radial range, and statistic
    if pxRatios:
        facs_mean = enhancement_mean
        facs_med = enhancement_percs
    else:
        facs_mean = enhancement2_mean
        facs_med = enhancement2_percs

    if median:
        fac = facs_med[:, rad, 1]  # percs = [16, 50, 85]
    else:
        fac = facs_mean[:, rad]

    ax.set_ylim([0.8, np.nanmax(fac) * 1.3])

    # plot individual galaxies
    markersize = 12**2 if color_quant == "sfr" else 8**2
    s = ax.scatter(mstar, fac, markersize, c=cvals, cmap=cmap, vmin=cminmax[0], vmax=cminmax[1])

    if median:
        # draw individual percentiles as vertical errorbars
        for i in range(len(haloIDs)):
            xx = mstar[i]
            yy_low = facs_med[i, rad, 0]
            yy_high = facs_med[i, rad, -1]

            ax.plot([xx, xx], [yy_low, yy_high], "-", color="#eee", alpha=0.4, zorder=-1)

    for ratio in [1, 10, 100, 1000]:
        if ratio > ax.get_ylim()[1]:
            continue
        ax.plot(xlim, [ratio, ratio], "-", lw=1.0, color="#ccc", zorder=-1)
        ax.text(xlim[0] + 0.01, ratio * 1.05, f"{ratio}x", ha="left", va="bottom", color="#ccc", zorder=-1)

    # colobar and save plot
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel(clabel)

    fig.savefig(
        "sb_enhancement_vs_mass_rad%d_%s_px%d_%s_%s_%d_nh%d_b%s.pdf"
        % (rad, "median" if median else "mean", pxRatios, color_quant, sim.name, sim.snap, len(haloIDs), b)
    )
    plt.close(fig)


def enhancementTrendVsMass(sim, haloIDs, b):
    """Plot mean/median trends of SB enhancement factor for different radial ranges.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of the halo indices to load.
      b (float): the boost parameter of the hot ISM component.
    """
    # config
    xlim = [10.0, 11.0]  # log msun
    ylim = [0.8, 1e3]  # enhancement factor
    binsize = 0.1  # 0.17 # log mstar

    cacheFile = sim.cachePath + "iltis_enhancefac_%s-%d_nh%d_b%s.hdf5" % (sim.simName, sim.snap, len(haloIDs), b)

    if isfile(cacheFile):
        with h5py.File(cacheFile, "r") as f:
            assert np.array_equal(haloIDs, f["haloIDs"][()])
            # enhancement factors
            enhancement_mean = f["enhancement_mean"][()]
            enhancement_percs = f["enhancement_percs"][()]
            enhancement2_mean = f["enhancement2_mean"][()]
            enhancement2_percs = f["enhancement2_percs"][()]
            # sb values
            # sb_scattered_mean = f['sb_scattered_mean'][()]
            # sb_scattered_percs = f['sb_scattered_percs'][()]
            # galaxy propreties
            # tot_lum_scattered = f['tot_lum_scattered'][()]
            mstar = f["mstar"][()]
            # sfr = f['sfr'][()]
            # lbol = f['lbol'][()]
            # mbh = f['mbh'][()]
            # m200 = f['m200'][()]

        print("Loaded: [%s]" % cacheFile)
    else:
        raise Exception("Run enhancementVsMass() first.")

    # start plot
    fig, ax = plt.subplots(figsize=figsize_sm)

    labels = [
        r"$\rm{R < R_{200}}$",
        r"20 < R/kpc < 200",
        r"50 < R/kpc < 200",
        r"$\rm{R = R_{500}}$",
        r"$\rm{R = R_{200}}$",
    ]

    ax.set_xlabel(r"Galaxy Stellar Mass [ log $\rm{M_{sun}}$ ]")
    ax.set_ylabel("Surface Brightness Enhancement")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # radial range and mean vs median
    colors = ["skip_rad0"]

    for rad in [1, 2, 3, 4]:
        for i in range(2):
            # pull out values
            if i == 0:
                # mean and median of ratio pixels values
                fac_median = enhancement_percs[:, rad, 1]  # percs = [16, 50, 85]
                fac_medlower = enhancement_percs[:, rad, 0]
                fac_medupper = enhancement_percs[:, rad, -1]
                fac_mean = enhancement_mean[:, rad]
            if i == 1:
                # ratio of mean or median pixel values
                fac_median = enhancement2_percs[:, rad, 1]  # percs = [16, 50, 85]
                fac_mean = enhancement2_mean[:, rad]

            # compute running trend vs mass
            xm_mean, ym_mean, _, pm_mean = running_median(mstar, fac_mean, binSize=binsize, percs=percs)
            xm_med, ym_med, _, pm_med = running_median(mstar, fac_median, binSize=binsize, percs=percs)

            if 0:
                # percentiles across the population
                ym_medlower = pm_med[0, :]
                ym_medupper = pm_med[-1, :]
            else:
                # percentiles from each halo separately, then take median trend of this envelope across the population
                xm_test1, ym_medlower, _, _ = running_median(mstar, fac_medlower, binSize=binsize, percs=percs)
                xm_test2, ym_medupper, _, _ = running_median(mstar, fac_medupper, binSize=binsize, percs=percs)
                assert np.array_equal(xm_test1, xm_med)
                assert np.array_equal(xm_test2, xm_med)

            # extend to right-edge of plot
            xm_mean = np.append(xm_mean, xlim[1])
            xm_med = np.append(xm_med, xlim[1])

            yy_mean_extrap = 10.0 ** interp1d(xm_mean[:-1], np.log10(ym_mean), fill_value="extrapolate")(xlim[1])
            yy_med_extrap = 10.0 ** interp1d(xm_med[:-1], np.log10(ym_med), fill_value="extrapolate")(xlim[1])
            ym_mean = np.append(ym_mean, yy_mean_extrap)
            ym_med = np.append(ym_med, yy_med_extrap)

            yy_lower_extrap = 10.0 ** interp1d(xm_med[:-1], np.log10(ym_medlower), fill_value="extrapolate")(xlim[1])
            yy_upper_extrap = 10.0 ** interp1d(xm_med[:-1], np.log10(ym_medupper), fill_value="extrapolate")(xlim[1])
            ym_medlower = np.append(ym_medlower, yy_lower_extrap)
            ym_medupper = np.append(ym_medupper, yy_upper_extrap)

            # extend to left-edge of plot
            ym_mean_extrap = 10.0 ** interp1d(xm_mean, np.log10(ym_mean), fill_value="extrapolate")(xlim[0])
            ym_med_extrap = 10.0 ** interp1d(xm_med, np.log10(ym_med), fill_value="extrapolate")(xlim[0])
            ym_mean = np.insert(ym_mean, 0, ym_mean_extrap)
            ym_med = np.insert(ym_med, 0, ym_med_extrap)

            ym_lower_extrap = 10.0 ** interp1d(xm_med, np.log10(ym_medlower), fill_value="extrapolate")(xlim[0])
            ym_upper_extrap = 10.0 ** interp1d(xm_med, np.log10(ym_medupper), fill_value="extrapolate")(xlim[0])
            ym_medlower = np.insert(ym_medlower, 0, ym_lower_extrap)
            ym_medupper = np.insert(ym_medupper, 0, ym_upper_extrap)

            ym_mean = savgol_filter(ym_mean, sKn, sKo)
            ym_med = savgol_filter(ym_med, sKn, sKo)
            ym_medlower = savgol_filter(ym_medlower, sKn, sKo)
            ym_medupper = savgol_filter(ym_medupper, sKn, sKo)

            xm_mean = np.insert(xm_mean, 0, xlim[0])
            xm_med = np.insert(xm_med, 0, xlim[0])

            # plot
            if i == 0:
                (l,) = ax.plot(xm_mean, ym_mean, "-", label=labels[rad])
                colors.append(l.get_color())
                ax.plot(xm_med, ym_med, ":", color=colors[rad])
                if rad in [3, 4]:
                    ax.fill_between(xm_med, ym_medlower, ym_medupper, color=colors[rad], alpha=0.1)
            if i == 1:
                # similar for large distances, much smaller for small distances
                # ax.plot(xm_mean, ym_mean, '-.', color=colors[rad])
                ax.plot(xm_med, ym_med, "--", color=colors[rad])

    # finish plot
    type_labels = [
        r"<$L_{\rm scattered}$/L$_{\rm intrinsic}$> Mean",
        r"<$L_{\rm scattered}$/$L_{\rm intrinsic}$> Median",
        r"<$L_{\rm scattered}$>/<$L_{\rm intrinsic}$> Median",
    ]

    handles = [plt.Line2D([0], [0], color="black", linestyle=linestyles[i]) for i in range(len(type_labels))]
    legend2 = ax.legend(handles, type_labels, borderpad=1.0, loc="lower left")
    ax.add_artist(legend2)

    ax.legend(loc="upper right")

    for ratio in [1, 10, 100]:
        if ratio > ax.get_ylim()[1]:
            continue
        ax.plot(xlim, [ratio, ratio], "-", lw=1.0, color="#ccc", zorder=-1)
        ax.text(xlim[0] + 0.01, ratio * 1.05, f"{ratio}x", ha="left", va="bottom", color="#ccc", zorder=-1)

    fig.savefig("sb_enhancement_vs_mass_%s_%d_nh%d_b%s.pdf" % (sim.name, sim.snap, len(haloIDs), b))
    plt.close(fig)


def singleHaloImageO7(sP, subhaloInd):
    """OVII optical depth projection."""
    method = "sphMap"  # note: fof-scope
    axes = [0, 1]
    labelZ = True
    labelScale = "physical"
    labelSim = False
    labelHalo = True
    relCoords = True
    size = 100
    sizeType = "kpc"
    rotation = "edge-on"

    panels = []

    # equirectangular - O7 optical depth map
    panels.append({"partType": "gas", "partField": "tau0_OVIIr", "valMinMax": [-0.5, 1.5]})

    sP.createCloudyCache = False
    projType = "equirectangular"  #'mollweide'
    projParams = {"fov": 360.0}
    nPixels = [1200, 600]
    axesUnits = "rad_pi"

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = True
        fontsize = 16
        saveFilename = "%s_%d_%d_%s.pdf" % (sP.simName, sP.snap, subhaloInd, "-".join([p["partField"] for p in panels]))

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def paperPlots():
    """Generate all plots of the paper."""
    # config
    sim = simParams("tng50-1", redshift=0.0)

    _, _, _, _, haloIDs = _get_subhalo_sample(sim)
    haloID_demo = 204  # used for single-halo plots
    b_fiducial = 0.0001

    if 0:
        # figs 1 and 2: single halo demonstration, fiducial model
        radialProfile(sim, haloID=haloID_demo, b=b_fiducial)
        imageSBcomp(sim, haloID=haloID_demo, b=b_fiducial)

    if 0:
        # fig 3: gallery of scattered images
        haloIDs = [82, 103, 124, 128, 160, 163, 201, 203, 204, 237, 281, 287]
        imageSBgallery(sim, haloIDs=haloIDs, b=b_fiducial)

    if 0:
        # fig 4: stacked SB radial profiles across mstar bins
        stackedRadialProfiles(sim, haloIDs, b=b_fiducial)

    if 0:
        # fig 5: enhancement factor (at r200) vs mstar, colored by galaxy properties
        for cquant in ["sfr", "LOVII", "Lbol", "meansb_r200"]:
            enhancementVsMass(sim, haloIDs, b=b_fiducial, rad=4, color_quant=cquant, median=False, pxRatios=True)

    if 0:
        # fig 6: enhancement factor vs mass, for different radii and mean vs median
        enhancementTrendVsMass(sim, haloIDs, b=b_fiducial)

    if 0:
        # fig 7: check galaxy OVIIr luminosity vs observational constraints
        galaxyLumVsSFR(sim, b=b_fiducial, addDiffuse=True, correctLineToBandFluxRatio=True)

    if 0:
        # fig 8: impact of central source/boost factor
        radialProfiles(sim, haloID=haloID_demo, b=[0, 0.0001, 0.001, 0.01])

    if 0:
        # fig X: OVIII case studies
        haloIDs = [201, 202, 203, 204]
        line = "O--8-18.9709A"

        for haloID in haloIDs:
            radialProfile(sim, haloID=haloID, b=b_fiducial, line=line)
            imageSBcomp(sim, haloID=haloID, b=b_fiducial, line=line)

    if 0:
        # fig X: explore enhancements factor vs mass for (i) different radial ranges, and
        # (ii) as a function of galaxy properties: SFR, L_OVIIr, L_AGN, M_BH, etc
        for px in [True, False]:
            for i in range(5):
                enhancementVsMass(sim, haloIDs, b=b_fiducial, rad=i, median=False, pxRatios=px)
                enhancementVsMass(sim, haloIDs, b=b_fiducial, rad=i, median=True, pxRatios=px)
            for cquant in ["sfr", "LOVII", "Lbol", "mbh", "m200", "meansb_r200"]:
                enhancementVsMass(sim, haloIDs, b=b_fiducial, rad=4, color_quant=cquant, median=False, pxRatios=px)

    if 0:
        # fig X: make all individual halo plots, for all halos
        for haloID in haloIDs:
            galaxyLum(sim, haloID=haloID, b=0)  # check intrinsic vs. scattered galaxy lum
            radialProfile(sim, haloID=haloID, b=0)
            imageSBcomp(sim, haloID=haloID, b=0)
            spectrum(sim, haloID=haloID, b=0)

    if 0:
        # fig X: check impact of velocities (v4novel cases)
        _load_data.__defaults__ = ("v4novel", _load_data.__defaults__[1])  # hack
        for haloID in [201, 202, 203, 204, 530, 531, 532, 535]:
            radialProfile(sim, haloID=haloID, b=b_fiducial)
            imageSBcomp(sim, haloID=haloID, b=b_fiducial)

    if 0:
        # fig X: check input luminosity profiles
        radialProfilesInput(sim, haloID=None)

    if 0:
        # fig X: boost factor explorations
        radialProfilesInput(sim, haloID=haloID_demo)  # check boost models
        for b in [0, 0.0001, 0.001, 0.01]:
            radialProfile(sim, haloID=haloID_demo, b=b)
            imageSBcomp(sim, haloID=haloID_demo, b=b)
            spectrum(sim, haloID=haloID_demo, b=b)

    if 0:
        # fig X: check galaxy OVIIr luminosity vs observational constraints
        # decision: b=0.001 is the bright case, b=0 is the dim case, and they likely bracket the truth
        # (adopt b=0.0001 as the fiducial case)
        for b in [0, 0.0001, 0.001, 0.01]:
            galaxyLumVsSFR(sim, b=b, addDiffuse=True, correctLineToBandFluxRatio=False)
            # galaxyLum(sim, haloID=haloID_demo, b=b)

    if 0:
        # fig X: optical depth maps
        singleHaloImageO7(sim, sim.halo(haloID_demo)["GroupFirstSub"])

    if 0:
        # (possibilities for additional content)
        # - spectral profiles, e.g. intrinsic vs scattered, few different radii
        # - scattering diagnostics e.g. number of scatterings per photon, optical depths
        # - a dens-temp phase diagram, and/or with rad, for intrinsic vs scattered (using cell index info)
        # - a dens-temp phase diagram, and/or with rad, showing intrisic/scattered ratio (using cell index info)
        pass
