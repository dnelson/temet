"""
Analysis and helpers specifically for zoom resimulations in cosmological volumes.
"""

from glob import glob
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy.signal import savgol_filter

from ..load.simtxt import getCpuTxtLastTimestep
from ..plot.config import colors, linestyles, lw, sKn, sKo
from ..util.helper import logZeroNaN, running_median
from ..util.simParams import simParams
from ..vis.box import renderBox
from ..vis.halo import renderSingleHalo, selectHalosFromMassBins


def pick_halos():
    """Testing."""
    sP = simParams(res=2048, run="tng_dm", redshift=0.0)
    # sP = simParams(res=2500, run='tng', redshift=0.0)

    # config
    bins = [[x + 0.0, x + 0.1] for x in np.linspace(14.0, 15.4, 15)]
    numPerBin = 30

    hInds = selectHalosFromMassBins(sP, bins, numPerBin, "random")

    for i, bin in enumerate(bins):
        print(bin, hInds[i])

    # note: skipped h305 (IC gen failures, replaced with 443 in its mass bin)
    # note: skipped h1096 (IC gen failure, spans box edge, replaced with 799)
    # note: skipped h604 (corrupt GroupNsubs != Nsubgroups_Total in snap==53, replaced with 616)
    return hInds


def _halo_ids_run(res=14, onlyDone=False):
    """Parse runs.txt and return the list of (all) halo IDs."""
    path = "/virgotng/mpia/TNG-Cluster/individual/"
    path2 = "/virgotng/mpia/TNG-Cluster/inprogress/"

    # runs.txt file no longer relevant, use directories which exist
    dirs = glob(path + "L680n2048TNG_h*_L%d_sf3" % res)
    halo_inds = sorted([int(folder.split("_")[-3][1:]) for folder in dirs])

    if onlyDone:
        return halo_inds

    dirs_inprogress = glob(path2 + "L680n2048TNG_h*_L%d_sf3" % res)
    halo_inds2 = sorted([int(folder.split("_")[-3][1:]) for folder in dirs_inprogress])

    return sorted(halo_inds + halo_inds2)


def calculate_contamination(sPzoom, rVirFacs=(1, 2, 3, 4, 5, 10), verbose=False):
    """Calculate statistics of low-resolution dark matter particles in/near the halo.

    Compute: number of low-res DM within each rVirFac*rVir distance, as well as the minimum distance to
    any low-res DM particle, and a radial profile of contaminating particles.
    """
    cacheFile = sPzoom.derivPath + "contamination_stats.hdf5"

    # check for existence of cache
    if isfile(cacheFile):
        r = {}
        with h5py.File(cacheFile, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # load and calculate now
    halo = sPzoom.groupCatSingle(haloID=0)
    r200 = halo["Group_R_Crit200"]

    h = sPzoom.snapshotHeader()

    pos_hr = sPzoom.snapshotSubset("dm", "pos")
    pos_lr = sPzoom.snapshotSubset(2, "pos")

    mass_lr = sPzoom.snapshotSubset(2, "mass")
    mass_hr = h["MassTable"][sPzoom.ptNum("dm")]

    dists_lr = sPzoom.periodicDists(halo["GroupPos"], pos_lr)
    dists_hr = sPzoom.periodicDists(halo["GroupPos"], pos_hr)

    min_dist_lr = dists_lr.min()  # code units
    min_dist_lr = sPzoom.units.codeLengthToMpc(min_dist_lr)  # pMpc
    if verbose:
        print("min dists from halo to closest low-res DM [pMpc]: ", min_dist_lr)

    # allocate
    counts = np.zeros(len(rVirFacs), dtype="int32")  # number low-res DM
    fracs = np.zeros(len(rVirFacs), dtype="float32")  # of low-res DM to HR DM
    massfracs = np.zeros(len(rVirFacs), dtype="float32")  # of low-res DM mass to HR DM mass

    # calculate counts
    for i, rVirFac in enumerate(rVirFacs):
        w_lr = np.where(dists_lr < rVirFac * r200)
        w_hr = np.where(dists_hr < rVirFac * r200)

        counts[i] = len(w_lr[0])
        fracs[i] = counts[i] / float(len(w_hr[0]))

        totmass_lr = mass_lr[w_lr].sum()
        totmass_hr = len(w_hr[0]) * mass_hr

        massfracs[i] = totmass_lr / totmass_hr

        if verbose:
            print(
                "num within %2d rvir (%6.1f) = %6d (%5.2f%% of HR num) (%5.2f%% of HR mass)"
                % (rVirFac, rVirFac * r200, counts[i], fracs[i] * 100, massfracs[i] * 100)
            )

    # calculate radial profiles
    rlim = [0.0, np.max(rVirFacs) * r200]
    nbins = 50

    r_count_hr, rr = np.histogram(dists_hr, bins=nbins, range=rlim)
    r_count_lr, _ = np.histogram(dists_lr, bins=nbins, range=rlim)

    r_mass_lr, _ = np.histogram(dists_lr, bins=nbins, range=rlim, weights=mass_lr)
    r_mass_hr = r_count_hr * mass_hr

    r_frac = r_count_lr / r_count_hr
    r_massfrac = r_mass_lr / r_mass_hr

    r_frac_cum = np.cumsum(r_count_lr) / np.cumsum(r_count_hr)
    r_massfrac_cum = np.cumsum(r_mass_lr) / np.cumsum(r_mass_hr)

    rr = rr[:-1] + (rlim[1] - rlim[0]) / nbins  # bin midpoints

    # save cache
    r = {
        "min_dist_lr": min_dist_lr,
        "rVirFacs": rVirFacs,
        "counts": counts,
        "fracs": fracs,
        "massfracs": massfracs,
        "rr": rr,
        "r_frac": r_frac,
        "r_massfrac": r_massfrac,
        "r_massfrac_cum": r_massfrac_cum,
        "r_frac_cum": r_frac_cum,
    }

    with h5py.File(cacheFile, "w") as f:
        for key in r:
            f[key] = r[key]
    print("Saved: [%s]" % cacheFile)

    return r


def contamination_profile():
    """Check level of low-resolution contamination (DM particles) in zoom run. Plot radial profile."""
    # config
    hInd = 0  # 31619 # 10677

    # load zoom: group catalog
    # sPz = simParams(res=zoomRes, run=zoomRun, hInd=hInd, redshift=redshift, variant=variant)
    sPz = simParams(run="tng50_zoom", hInd=11, res=11, variant="sf8", redshift=6.0)

    halo_zoom = sPz.groupCatSingle(haloID=0)

    # load parent box
    sP = sPz.sP_parent
    halo = sP.groupCatSingle(haloID=hInd)

    print("parent halo pos: ", halo["GroupPos"])
    print("zoom halo cenrelpos: ", halo_zoom["GroupPos"] - sP.boxSize / 2)
    print("parent halo mass: ", sP.units.codeMassToLogMsun([halo["Group_M_Crit200"], halo["GroupMass"]]))
    print("zoom halo mass: ", sP.units.codeMassToLogMsun([halo_zoom["Group_M_Crit200"], halo_zoom["GroupMass"]]))

    # print/load contamination statistics
    contam = calculate_contamination(sPz, verbose=True)
    min_dist_lr = contam["min_dist_lr"] * sPz.HubbleParam

    # plot contamination profiles
    fig, ax = plt.subplots()
    ylim = [-5.0, 0.0]

    ax.set_xlabel("Distance [%s]" % sPz.units.UnitLength_str)
    ax.set_ylabel("Low-res DM Contamination Fraction [log]")
    ax.xaxis.set_minor_locator(MultipleLocator(500))
    ax.set_xlim([0.0, contam["rr"].max()])
    ax.set_ylim(ylim)

    ax.plot([0, contam["rr"][-1]], [-1.0, -1.0], "-", color="#888888", alpha=0.5, label="10%")
    ax.plot([0, contam["rr"][-1]], [-2.0, -2.0], "-", color="#bbbbbb", alpha=0.2, label="1%")
    ax.plot(contam["rr"], logZeroNaN(contam["r_frac"]), "-", label="by number")
    ax.plot(contam["rr"], logZeroNaN(contam["r_massfrac"]), "-", label="by mass")

    ax.plot([min_dist_lr, min_dist_lr], ylim, ":", color="#555555", alpha=0.5, label="closest LR")

    ax2 = ax.twiny()
    ax2.set_xlabel(r"Distance [$R_{\rm 200,crit}$]")
    ax2.set_xlim([0.0, contam["rr"].max() / halo_zoom["Group_R_Crit200"]])
    ax2.xaxis.set_minor_locator(MultipleLocator(1))

    ax.legend(loc="lower right")
    fig.savefig("contamination_profile_%s_%d.pdf" % (sPz.simName, sPz.snap))
    plt.close(fig)


def contamination_compare_profiles():
    """Compare contamination radial profiles between runs."""
    zoomRes = 11
    hInds = [11]
    redshift = 6.0
    variants = ["sf8", "sf32"]
    run = "tng50_zoom"

    # start plot
    fig, ax = plt.subplots()
    ylim = [-5.0, 0.0]

    ax.set_xlabel(r"Distance [$R_{\rm 200,crit}$]")
    ax.set_ylabel("Low-res DM Contamination Fraction [log]")
    ax.set_xlim([0.0, 10.0])
    ax.set_ylim(ylim)

    # load: loop over hInd/variant combination
    for i, hInd in enumerate(hInds):
        for j, variant in enumerate(variants):
            # load zoom: group catalog
            sPz = simParams(res=zoomRes, run=run, hInd=hInd, redshift=redshift, variant=variant)

            halo_zoom = sPz.groupCatSingle(haloID=0)

            # load contamination statistics and plot
            contam = calculate_contamination(sPz, verbose=True)
            rr = contam["rr"] / halo_zoom["Group_R_Crit200"]
            halo_r200_pMpc = sPz.units.codeLengthToMpc(halo_zoom["Group_R_Crit200"])
            min_dist_lr = contam["min_dist_lr"] / halo_r200_pMpc

            (l,) = ax.plot(
                rr, logZeroNaN(contam["r_frac"]), linestyles[j], color=colors[i], label="h%d_%s" % (hInd, variant)
            )
            # l, = ax.plot(rr, logZeroNaN(contam['r_massfrac']), '--', color=colors[i])

            ax.plot([min_dist_lr, min_dist_lr], [ylim[1] - 0.3, ylim[1]], linestyles[j], color=l.get_color(), alpha=0.5)
            print(hInd, variant, min_dist_lr, halo_r200_pMpc)

    ax.plot([0, rr[-1]], [-1.0, -1.0], "-", color="#888888", alpha=0.4, label="10%")
    ax.plot([0, rr[-1]], [-2.0, -2.0], "-", color="#bbbbbb", alpha=0.4, label="1%")
    ax.plot([1.0, 1.0], ylim, "-", color="#bbbbbb", alpha=0.2)
    ax.plot([2.0, 2.0], ylim, "-", color="#bbbbbb", alpha=0.2)

    ax.legend(loc="upper left")
    fig.savefig("contamination_profiles_L%d_hN%d_%s.pdf" % (zoomRes, len(hInds), "-".join(variants)))
    plt.close(fig)


def contamination_mindist():
    """Plot distribution of contamination minimum distances, and trend with halo mass."""
    # config
    zoomRes = 14
    hInds = _halo_ids_run(onlyDone=True)
    variant = "sf3"
    redshift = 0.0
    run = "tng_zoom"

    frac_thresh = 1e-3

    # load data
    halo_mass = np.zeros(len(hInds), dtype="float32")
    min_dists = np.zeros(len(hInds), dtype="float32")
    min_dists_thresh = np.zeros(len(hInds), dtype="float32")

    for i, hInd in enumerate(hInds):
        if i % len(hInds) // 10 == 0:
            print(f"{i * 10:2d}% ")

        sPz = simParams(res=zoomRes, run=run, hInd=hInd, redshift=redshift, variant=variant)
        halo_zoom = sPz.groupCatSingle(haloID=0)
        halo_mass[i] = sPz.units.codeMassToLogMsun(halo_zoom["Group_M_Crit200"])

        contam = calculate_contamination(sPz)

        # minimum distance to first LR particle
        min_dist_lr = contam["min_dist_lr"] / sPz.units.codeLengthToMpc(halo_zoom["Group_R_Crit200"])
        min_dists[i] = min_dist_lr

        # distance at which cumulative fraction of LR/HR particles exceeds a threshold (linear interp)
        # min_ind = np.where(contam['r_frac_cum'] > frac_thresh)[0].min()
        # min_dists_thresh[i] = contam['rr'][min_ind] / halo_zoom['Group_R_Crit200']
        min_dists_thresh[i] = np.interp(frac_thresh, contam["r_frac_cum"], contam["rr"]) / halo_zoom["Group_R_Crit200"]

    # plot distribution
    xlim = [0.0, 10.0]
    nbins = 60

    fig, ax = plt.subplots()

    ax.set_xlabel("Minimum Contamination Distance [$R_{\\rm 200,crit}$]")
    ax.set_ylabel("Number of Halos")
    ax.set_xlim(xlim)

    label1 = "Single Closest LR Particle"
    label2 = "Low-Resolution Fraction $f_{\\rm LR} > 10^{%d}$" % np.log10(frac_thresh)
    ax.hist(min_dists, bins=nbins, range=xlim, alpha=0.7, label=label1)
    ax.hist(min_dists_thresh, bins=nbins, range=xlim, alpha=0.7, label=label2)
    ax.legend(loc="upper right")

    ax.plot([1, 1], ax.get_ylim(), "-", color="#bbbbbb", alpha=0.2)

    fig.savefig("contamination_mindist_L%d_hN%d_%s.pdf" % (zoomRes, len(hInds), variant))
    plt.close(fig)

    # plot min dist vs mass trend
    fig, ax = plt.subplots()

    ax.set_xlabel("Halo Mass M$_{\\rm 200c}$ [ log $M_{\\odot}$ ]")
    ax.set_ylabel("Minimum Contamination Distance [ $R_{\\rm 200,crit}$ ]")
    ax.set_xlim([14.25, 15.4])
    ax.set_ylim([0.0, 10.0])

    for rr in np.arange(1, 7):
        ax.plot(ax.get_xlim(), [rr, rr], "-", color="#bbbbbb", alpha=0.2)
    ax.plot(halo_mass, min_dists, "o", label=label1)
    ax.plot(halo_mass, min_dists_thresh, "o", label=label2)

    xm, ym, _ = running_median(halo_mass, min_dists, binSize=0.1)
    xm2, ym2, _ = running_median(halo_mass, min_dists_thresh, binSize=0.1)
    ym = savgol_filter(ym, sKn, sKo)
    ax.plot(xm, ym, "--", lw=lw * 2, color="black", alpha=0.7, label="Median (Closest LR particle)")
    ax.plot(xm2, ym2, "-", lw=lw * 2, color="black", label="Median ($f_{\\rm LR} > 10^{%d}$)" % np.log10(frac_thresh))

    ax.legend(loc="upper left")
    fig.savefig("contamination_mindist_vs_mass_L%d_hN%d_%s.pdf" % (zoomRes, len(hInds), variant))
    plt.close(fig)


def contamination_mindist2():
    """For a virtual box, plot contamination min dist histograms and trends."""
    # config
    sim = simParams(run="tng-cluster", redshift=0.0)  # 7.0

    frac_thresh = 1e-3

    acField_lr = "Subhalo_RadProfile3D_Global_LowResDM_Count"
    acField_hr = "Subhalo_RadProfile3D_Global_HighResDM_Count"

    # load data
    ac_lr = sim.auxCat(acField_lr)
    ac_hr = sim.auxCat(acField_hr)

    # gather
    profiles_lr = ac_lr[acField_lr]
    profiles_hr = ac_hr[acField_hr]

    subhaloIDs = ac_lr["subhaloIDs"]
    rad_bin_edges = ac_lr[acField_lr + "_attrs"]["rad_bin_edges"]
    rad_bins = (rad_bin_edges[:-1] + rad_bin_edges[1:]) / 2

    assert np.array_equal(ac_lr["subhaloIDs"], ac_hr["subhaloIDs"])
    assert np.array_equal(ac_lr[acField_lr + "_attrs"]["rad_bin_edges"], ac_hr[acField_hr + "_attrs"]["rad_bin_edges"])

    halo_mass = sim.subhalos("mhalo_200_log")[subhaloIDs]
    min_dists = np.zeros(len(subhaloIDs), dtype="float32")
    min_dists_thresh = np.zeros(len(subhaloIDs), dtype="float32")

    for i in range(len(subhaloIDs)):
        # minimum distance to first LR particle
        min_dist_ind = np.where(profiles_lr[i, :] > 0)[0]
        if len(min_dist_ind) == 0:
            min_dists[i] = rad_bins.max()
        else:
            min_dists[i] = rad_bins[min_dist_ind.min()]

        # distance at which cumulative fraction of LR/HR particles exceeds a threshold (linear interp)
        r_frac_cum = np.cumsum(profiles_lr[i, :]) / np.cumsum(profiles_hr[i, :])
        min_dists_thresh[i] = np.interp(frac_thresh, r_frac_cum, rad_bins)

    # plot distribution
    xlim = [0.0, 10.0]
    nbins = 60

    fig, ax = plt.subplots()

    ax.set_xlabel(r"Minimum Contamination Distance [$R_{\rm 200,crit}$]")
    ax.set_ylabel("Number of Halos")
    ax.set_xlim(xlim)

    label1 = "Single Closest LR Particle"
    label2 = "Low-Resolution Fraction $f_{\\rm LR} > 10^{%d}$" % np.log10(frac_thresh)
    ax.hist(min_dists, bins=nbins, range=xlim, alpha=0.7, label=label1)
    ax.hist(min_dists_thresh, bins=nbins, range=xlim, alpha=0.7, label=label2)
    ax.legend(loc="upper right")

    ax.plot([1, 1], ax.get_ylim(), "-", color="#bbbbbb", alpha=0.2)

    fig.savefig("contamination_mindist_%s_%d.pdf" % (sim.simName, sim.snap))
    plt.close(fig)

    # plot min dist vs mass trend
    fig, ax = plt.subplots()

    ax.set_xlabel(r"Halo Mass M$_{\\rm 200c}$ [ log $M_{\odot}$ ]")
    ax.set_ylabel(r"Minimum Contamination Distance [ $R_{\rm 200,crit}$ ]")
    ax.set_xlim([14.25, 15.4])
    ax.set_ylim([0.0, 10.0])

    for rr in np.arange(1, 7):
        ax.plot(ax.get_xlim(), [rr, rr], "-", color="#bbbbbb", alpha=0.2)
    ax.plot(halo_mass, min_dists, "o", label=label1)
    ax.plot(halo_mass, min_dists_thresh, "o", label=label2)

    xm, ym, _ = running_median(halo_mass, min_dists, binSize=0.1)
    xm2, ym2, _ = running_median(halo_mass, min_dists_thresh, binSize=0.1)
    ym = savgol_filter(ym, sKn, sKo)
    ax.plot(xm, ym, "--", lw=lw * 2, color="black", alpha=0.7, label="Median (Closest LR particle)")
    ax.plot(xm2, ym2, "-", lw=lw * 2, color="black", label="Median ($f_{\\rm LR} > 10^{%d}$)" % np.log10(frac_thresh))

    ax.legend(loc="upper left")
    fig.savefig("contamination_mindist_vs_mass_%s_%d.pdf" % (sim.simName, sim.snap))
    plt.close(fig)


def sizefacComparison():
    """Compare SizeFac 2,3,4 runs (contamination and CPU times) in the testing set."""
    # config
    zoomRes = 14
    redshift = 0.0

    if 0:
        # testing contam/CPU time scalings with size fac
        hInds = [8, 50, 51, 90]
        variants = ["sf2", "sf3", "sf4"]
        run = "tng_zoom_dm"

    if 0:
        # testing CPU time scaling with core count and unit systems
        hInds = [50]
        variants = ["sf2_n160s", "sf2_n160s_mpc", "sf2_n320s", "sf2_n640s", "sf3"]

    if 1:
        # main TNG-Cluster sample
        hInds = _halo_ids_run(onlyDone=True)
        variants = ["sf3"]
        run = "tng_zoom"

    # load
    results = []

    for hInd in hInds:
        for variant in variants:
            sP = simParams(run=run, res=zoomRes, hInd=hInd, redshift=redshift, variant=variant)

            _, _, _, cpuHours = getCpuTxtLastTimestep(sP.simPath + "/txt-files/cpu.txt")

            contam = calculate_contamination(sP, verbose=True)

            halo = sP.groupCatSingle(haloID=0)
            haloMass = sP.units.codeMassToLogMsun(halo["Group_M_Crit200"])
            haloRvir = sP.units.codeLengthToMpc(halo["Group_R_Crit200"])

            print("Load hInd=%4d variant=%s minDist=%5.2f" % (hInd, variant, contam["min_dist_lr"]))

            r = {
                "hInd": hInd,
                "variant": variant,
                "cpuHours": cpuHours,
                "haloMass": haloMass,
                "haloRvir": haloRvir,
                "contam_min": contam["min_dist_lr"],
                "contam_rvirfacs": contam["rVirFacs"],
                "contam_counts": contam["counts"],
            }
            results.append(r)

    # print some stats
    print("Median contam [pMpc]: ", np.median([result["contam_min"] for result in results]))
    print("Median contam [rVir]: ", np.median([result["contam_min"] / result["haloRvir"] for result in results]))
    print("Mean CPU hours: ", np.mean([result["cpuHours"] for result in results]))

    num_lowres = []
    for result in results:
        contam_rel = result["contam_min"] * sP.HubbleParam / result["haloRvir"]
        num = result["contam_counts"][0]  # 0=1rvir, 1=2rvir

        if contam_rel > 1.0:
            continue

        num_lowres.append(num)
        print(" [h = %4d] min contamination = %.2f rvir, num inside rvir = %d" % (result["hInd"], contam_rel, num))

    print(" mean median number of low-res dm (<1rvir): ", len(num_lowres), np.mean(num_lowres), np.median(num_lowres))

    # set up unique coloring by variant/sizeFac
    colors = {variant: colors[i] for i, variant in enumerate(variants)}

    # start plot
    fig = plt.figure(figsize=(22, 12))

    for rowNum in [0, 1]:
        xlabel = "Halo ID" if rowNum == 0 else r"Halo Mass [log M$_{\rm sun}$]"
        ax = fig.add_subplot(2, 3, rowNum * 3 + 1)

        handles = [plt.Line2D([0], [0], color=colors[sf], marker="o") for sf in colors.keys()]

        # (A) contamination dist kpc
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Min LR Dist [pMpc]")

        for result in results:
            xx = result["hInd"] if rowNum == 0 else result["haloMass"]
            color = colors[result["variant"]]
            ax.plot(xx, result["contam_min"], "o", color=color, label="")

        ax.legend(handles, ["%s" % variant for variant in colors.keys()], loc="best")

        # (B) contamination rvir
        ax = fig.add_subplot(2, 3, rowNum * 3 + 2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"Min LR Dist [r$_{\rm vir}$]")

        for result in results:
            xx = result["hInd"] if rowNum == 0 else result["haloMass"]
            color = colors[result["variant"]]
            ax.plot(xx, result["contam_min"] * sP.HubbleParam / result["haloRvir"], "o", color=color, label="")

        xlim = ax.get_xlim()
        for rVirFac in [5, 2, 1]:
            ax.plot(xlim, [rVirFac, rVirFac], "-", color="#bbbbbb", alpha=0.4)

        ax.legend(handles, ["%s" % variant for variant in colors.keys()], loc="best")

        # (C) cpu hours
        ax = fig.add_subplot(2, 3, rowNum * 3 + 3)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CPU Time [log kHours]")

        for result in results:
            xx = result["hInd"] if rowNum == 0 else result["haloMass"]
            color = colors[result["variant"]]
            ax.plot(xx, np.log10(result["cpuHours"] / 1e3), "o", color=color, label="")

        ax.legend(handles, ["%s" % variant for variant in colors.keys()], loc="best")

    # finish
    fig.savefig("sizefac_comparison.pdf")
    plt.close(fig)


def parentBoxVisualComparison(haloID, conf=0):
    """Make a visual comparison between halos in the parent box and their zoom realizations.

    Args:
      haloID (int): the zoom halo ID, at the final redshift (z=0).
      variant (str): the zoom variant.
      conf (int): the plotting configuration.
      snap (int): if not the final snapshot, plot at some redshift other than z=0.
    """
    sPz = simParams(run="tng50_zoom", res=11, hInd=haloID, redshift=6.0, variant="sf8")

    # render config
    rVirFracs = [1.0]  # [0.5, 1.0] # None
    method = "sphMap"  # sphMap
    nPixels = [800, 800]  # [1920,1920]
    axes = [0, 1]
    labelZ = True
    labelScale = True
    labelSim = True
    labelHalo = True
    relCoords = True

    # size       = 500.0
    # sizeType   = 'kpc'
    size = 4.0
    sizeType = "rVirial"

    # setup panels
    if conf == 0:
        # dm column density
        p = {"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.5, 9.5]}
    if conf == 1:
        # gas column density
        p = {"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.5, 8.0]}
    if conf == 2:
        # stellar density
        p = {"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]}

    panel_zoom = p.copy()
    panel_parent = p.copy()

    # sPz at a different redshift than the parent volume?
    if np.abs(sPz.redshift - sPz.sP_parent.redshift) > 0.1:
        # load MPB of this halo
        haloMPB = sPz.sP_parent.loadMPB(sPz.sP_parent.groupCatSingle(haloID=haloID)["GroupFirstSub"])
        assert sPz.snap in haloMPB["SnapNum"]

        # locate subhaloID at requested snapshot (could be z=0 or z>0)
        parSubID = haloMPB["SubfindID"][list(haloMPB["SnapNum"]).index(sPz.snap)]
    else:
        # same redshift
        parSubID = sPz.sP_parent.halo(haloID)["GroupFirstSub"]

    panel_zoom.update({"sP": sPz})
    panel_parent.update({"sP": sPz.sP_parent, "subhaloInd": parSubID})

    panels = [panel_zoom, panel_parent]

    class plotConfig:
        plotStyle = "open"
        rasterPx = nPixels[0]
        colorbars = True
        saveFilename = "./zoomParentVisComp_%s_z%.1f_%s_snap%d.pdf" % (
            sPz.simName,
            sPz.redshift,
            p["partType"],
            sPz.snap,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def zoomBoxVis(sPz=None, conf=0):
    """Make a visualization of a zoom simulation, without using/requiring group catalog information."""
    if sPz is None:
        sPz = simParams(res=11, run="tng100_zoom", redshift=0.0, hInd=5405, variant="sf4")

    # render config
    method = "sphMap_global"
    nPixels = [1000, 1000]  # [1920,1920]
    axes = [0, 1]
    labelZ = True
    labelScale = True
    labelSim = True
    plotHalos = 20

    # size?
    region_size_cMpc = 1.0  # show 5 cMpc size region around location

    zoomFac = region_size_cMpc * 1000 * sPz.HubbleParam / sPz.boxSize
    sliceFac = region_size_cMpc * 1000 * sPz.HubbleParam / sPz.boxSize

    if 1:
        # center on zoom halo
        absCenPos = sPz.subhalo(sPz.zoomSubhaloID)["SubhaloPos"]
        relCenPos = None

    # setup panels
    if conf == 0:
        # dm column density
        p = {"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 9.0]}
    if conf == 1:
        # gas column density
        p = {"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [4.5, 7.5]}
    if conf == 2:
        # gas temp
        p = {"partType": "gas", "partField": "temp", "valMinMax": [3.5, 5.0]}
    if conf == 3:
        # stars
        p = {"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [4.0, 8.0]}

    panel_zoom = p.copy()
    # panel_parent = p.copy()

    panel_zoom.update(
        {"run": sPz.run, "res": sPz.res, "redshift": sPz.redshift, "variant": sPz.variant, "hInd": sPz.hInd}
    )
    # panel_parent.update( {'run':sPz.sP_parent.run, 'res':sPz.sP_parent.res, 'redshift':sPz.sP_parent.redshift,
    #                       'hInd':parSubID})

    panels = [panel_zoom]  # [panel_zoom, panel_parent]

    class plotConfig:
        plotStyle = "open"
        rasterPx = nPixels[0]
        colorbars = True
        saveFilename = "./zoomBoxVis_%s_%s-%s_snap%d.png" % (sPz.simName, p["partType"], p["partField"], sPz.snap)

    renderBox(panels, plotConfig, locals(), skipExisting=False)


def plot_timeevo():
    """Diagnostic plots: group catalog properties for one halo vs time (no merger trees)."""
    # config simulations and field
    sims = []
    sims.append(simParams(run="tng_zoom", res=14, hInd=1335, variant="sf3"))
    sims.append(simParams(run="tng_zoom", res=14, hInd=1335, variant="sf3_s"))
    sims.append(simParams(run="tng_zoom", res=14, hInd=1335, variant="sf3_kpc"))
    sims.append(simParams(run="tng_zoom", res=14, hInd=1919, variant="sf3"))
    sims.append(simParams(run="tng_zoom", res=14, hInd=1919, variant="sf3_s"))
    sims.append(simParams(run="tng_zoom", res=14, hInd=1919, variant="sf3_kpc"))

    # SubhaloSFR, SubhaloBHMass, SubhaloMassInRadType, SubhaloHalfmassRadType
    # SubhaloGasMetallicity, SubhaloVelDisp, SubaloVmaxRad
    field = "SubhaloMassInRadType"  # VmaxRad'
    fieldIndex = 4  # -1 # -1 for scalar fields, otherwise >=0 index to use
    subhaloID = 0

    # quick check
    for sim in sims:
        sim.setSnap(33)
        subhalo = sim.subhalo(subhaloID)
        bhmass = sim.units.codeMassToLogMsun(subhalo["SubhaloBHMass"])[0]
        mstar = sim.units.codeMassToLogMsun(subhalo["SubhaloMassInRadType"][4])[0]

        print(f"{sim = } {bhmass = :.3f} {mstar = :.3f}")
    # return

    # load
    data = []

    for sim in sims:
        cache_file = "cache_%s_%s_%d.hdf5" % (sim.simName, field, fieldIndex)

        if isfile(cache_file):
            with h5py.File(cache_file, "r") as f:
                data.append({"sim": sim, "result": f["result"][()], "z": f["z"][()]})
            print("Loaded: [%s]" % cache_file)
            continue

        # load
        snaps = sim.validSnapList()
        z = sim.snapNumToRedshift(snaps)

        result = np.zeros(snaps.size, dtype="float32")
        result.fill(np.nan)

        for i, snap in enumerate(snaps):
            print(sim.simName, snap)
            # set snap and load single subhalo from group catalog
            sim.setSnap(snap)
            subhalo = sim.subhalo(subhaloID)

            if field not in subhalo:
                print(" skip")
                continue

            # store result
            if subhalo[field].size > 1:
                assert fieldIndex >= 0
            if subhalo[field].size == 1:
                assert fieldIndex == -1

            result[i] = subhalo[field] if subhalo[field].size == 1 else subhalo[field][fieldIndex]

        assert len(result) == z.size

        # save cache
        with h5py.File(cache_file, "w") as f:
            f["result"] = result
            f["z"] = z
        print("Saved: [%s]" % cache_file)

        data.append({"sim": sim, "result": result, "z": z})

    # plot
    fig, ax = plt.subplots()

    ylabel = field if fieldIndex == -1 else field + "-%d" % fieldIndex
    if "Mass" in field:
        ylabel += r" [ log M$_{\rm sun}$ ]"
    if "SFR" in field:
        ylabel += r" [ M$_{\rm sun}$/yr ]"
    if "HalfmassRad" in field:
        ylabel += r" [ log ckpc/h ]"

    ax.set_xscale("symlog")
    ax.set_xlabel("Redshift")
    ax.set_xticks([0, 1, 2, 4, 6, 10])
    ax.set_ylabel(ylabel)
    # ax.set_yscale('log')

    # loop over runs
    for i, sim in enumerate(sims):
        # load
        x = data[i]["z"]
        y = data[i]["result"]

        # unit conversions?
        if "Mass" in field:
            y = sim.units.codeMassToLogMsun(y)
        if "HalfmassRad" in field:
            y = np.log10(y)

        ax.plot(x, y, label=sim.simName)

    # finish plot
    ax.legend(loc="best")
    fig.savefig("time_evo_sh%d_%s_%d.pdf" % (subhaloID, field, fieldIndex))
    plt.close(fig)
