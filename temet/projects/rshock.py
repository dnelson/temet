"""
Virial shock radius: methods based on spherical healpix shell samplings of gas properties.
"""

import time
from functools import partial
from os import mkdir
from os.path import isdir, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

from temet.catalog.profile import healpix_shells_points
from temet.plot import subhalos
from temet.plot.config import colors, linestyles, lw, sKn, sKo
from temet.plot.util import loadColorTable
from temet.util import simParams
from temet.util.helper import last_nonzero, logZeroNaN, reportMemory, running_median
from temet.util.treeSearch import buildFullTree, calcHsml, calcParticleIndices, calcQuantReduction
from temet.vis.halo import renderSingleHalo


def plotHealpixShells(rad, data, label, rads=None, clim=None, ctName="viridis", saveFilename="plot.pdf"):
    """Plot a series of healpix shell samplings."""
    fig, ax = plt.subplots(figsize=(9, 16))

    xlim = [rad.min(), rad.max()]
    ylim = [0, 12]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("r / r$_{\\rm vir}$")
    ax.set_ylabel("Angular Direction ($\\theta, \\phi$)")

    ax.set_yticks(np.arange(ylim[0] + 1, ylim[1] + 1))
    # ax.set_xticks(list(ax.get_xticks()) + xlim)

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    norm = Normalize(vmin=clim[0], vmax=clim[1], clip=False) if clim is not None else None
    cmap = loadColorTable(ctName)

    plt.imshow(data, extent=extent, cmap=cmap, norm=norm, aspect="auto")  # origin='lower'

    if rads is not None:
        for i, rad in enumerate(rads):
            ax.plot([rad, rad], ylim, "-", label="#%d" % i)
        l = ax.legend(loc="upper left")
        for text in l.get_texts():
            text.set_color("white")

    # colorbar
    cax = make_axes_locatable(ax).append_axes("top", size="3%", pad=0.2)
    plt.colorbar(cax=cax, orientation="horizontal")
    cax.set_title(label)
    cax.xaxis.set_ticks_position("top")

    # finish
    fig.savefig(saveFilename)
    plt.close(fig)


def _thresholded_radius(radPts, h2d, thresh_perc, inequality, saveBase=None):
    """Derive a radius (rshock) by a thresholded histogramming/voting of a healpix sampling input.

    If saveBase is not None, then also dump debug plots.
    """
    assert inequality in ["<", ">"]

    # config
    thresh_minrad = 0.5  # r/rvir
    windowsize = 0.5  # +/- in units of rvir
    combineBins = 4  # radial smoothing (in histogramming)

    # copy input, derive threshold value
    q2d = h2d.copy()

    w_rad = np.where(radPts < thresh_minrad)
    q2d[:, w_rad] = np.nan
    w_zero = np.where(q2d == 0)  # e.g. shocks_dedt, shocks_mach for unflagged cells
    q2d[w_zero] = np.nan

    thresh = np.nanpercentile(q2d, thresh_perc)

    # select pixels on value>threshold or value<threshold
    if inequality == "<":
        q2d[np.isnan(q2d)] = thresh + 1  # do not select nan
        w_thresh = np.where(q2d < thresh)
    if inequality == ">":
        q2d[np.isnan(q2d)] = thresh - 1
        w_thresh = np.where(q2d > thresh)

    # collect all radii of these satisfying pixels, restricted to beyond minimum distance
    rad_thresh = radPts[w_thresh[1]]  # second dimension only
    rad_thresh = rad_thresh[np.where(rad_thresh >= thresh_minrad)]

    # make 2d mask
    result_mask = np.zeros(h2d.shape, "int32")
    result_mask[w_thresh] = 1

    w_rad = np.where(radPts < thresh_minrad)
    result_mask[:, w_rad] = 0

    # rshock: answer 1, localize histogram peak, take median rad of nearby thresholded pixels
    hist1d, xx = np.histogram(rad_thresh, bins=radPts[::combineBins])

    radPtsHist = xx[:-1] + (xx[1] - xx[0]) * combineBins / 2

    ind = np.argmax(hist1d)
    w = np.where((rad_thresh > xx[ind] - windowsize) & (rad_thresh <= xx[ind] + windowsize))
    rshock1 = np.nanmedian(rad_thresh[w])

    # rshock: answer 2, median of all thresholded pixels (no peak localization)
    rshock2 = np.nanmedian(rad_thresh)

    # go 'outside in', from large radii towards r->0, and find the first (farthest) pixel satisfying thresh
    # each ray gets one 'vote' for a radius
    inds = last_nonzero(result_mask, axis=1)
    w = np.where(inds >= 0)
    rad_thresh_ray = radPts[inds[w]]

    # rshock: answer 3, localize around ray-based max, but again median all thresholded pixels
    hist1dray, xx = np.histogram(rad_thresh_ray, bins=radPts[::combineBins])

    ind_raymax = np.argmax(hist1dray)
    w = np.where((rad_thresh > xx[ind_raymax] - windowsize) & (rad_thresh <= xx[ind_raymax] + windowsize))
    rshock3 = np.nanmedian(rad_thresh[w])

    # rshock: answer 4, localize around ray-based max, and use only ray-satisfying pixels
    w = np.where((rad_thresh_ray > xx[ind_raymax] - windowsize) & (rad_thresh_ray <= xx[ind_raymax] + windowsize))
    rshock4 = np.nanmedian(rad_thresh_ray[w])

    # rshock: answer 5, median of all ray-satisfying pixels
    rshock5 = np.nanmedian(rad_thresh_ray)

    rshock_vals = [rshock1, rshock2, rshock3, rshock4, rshock5]

    if saveBase is not None:
        # verbose print
        print(
            " thresh: [%4.1f = %6.3f] rshock [%.2f %.2f] ray: [%.2f %.2f %.2f peak=%.2f]"
            % (thresh_perc, thresh, rshock1, rshock2, rshock3, rshock4, rshock5, xx[ind_raymax])
        )

        # plot 2d mask
        label = "%s Thresh [P$_{\\rm %d}$ = %.2f]" % (inequality, thresh_perc, thresh)
        label2d = "Quantity [%s] %s" % (saveBase.split("_")[-2], label)
        plotHealpixShells(
            radPts,
            result_mask,
            label=label2d,
            clim=[0, 1],
            ctName="gray",
            rads=rshock_vals,
            saveFilename=saveBase + "_mask2d.pdf",
        )

        # plot 1d histo
        fig, ax = plt.subplots()
        ax.minorticks_on()
        ax.xaxis.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.3, color="black")
        ax.xaxis.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.05, color="black")

        ax.set_xlim([0, radPts.max()])
        ax.set_xlabel("r / r$_{\\rm vir}$")
        ax.set_ylabel("Pixel Fraction %s" % label)

        yy = hist1d / hist1d.sum()
        (l,) = ax.plot(radPtsHist, yy, "-", color="black", lw=lw + 1, drawstyle="steps-mid")

        for i, rshock in enumerate(rshock_vals):
            ax.plot([rshock, rshock], [0, yy.max() * 1.15], "-", lw=lw, label="#%d" % i)

        ax.legend(loc="upper right")
        fig.savefig(saveBase + "_1d.pdf")
        plt.close(fig)

        # plot 1d ray histo
        fig, ax = plt.subplots()
        ax.minorticks_on()
        ax.xaxis.grid(which="major", linestyle="-", linewidth=1.0, alpha=0.3, color="black")
        ax.xaxis.grid(which="minor", linestyle="-", linewidth=0.5, alpha=0.05, color="black")

        ax.set_xlim([0, radPts.max()])
        ax.set_xlabel("r / r$_{\\rm vir}$")
        ax.set_ylabel("Ray Fraction %s" % label)

        yy = hist1dray / hist1dray.sum()
        (l,) = ax.plot(radPtsHist, yy, "-", color="black", lw=lw + 1, drawstyle="steps-mid")

        for i, rshock in enumerate(rshock_vals):
            ax.plot([rshock, rshock], [0, yy.max() * 1.15], "-", lw=lw, label="#%d" % i)

        ax.legend(loc="upper right")
        fig.savefig(saveBase + "_ray1d.pdf")
        plt.close(fig)

    return rshock_vals


def _get_threshold_config(quantName):
    """Helper to hold config used by more than one analysis function."""
    percs_low = [0.1, 0.5, 1, 2, 5, 10]
    percs_high = [90, 95, 98, 99, 99.5, 99.9]

    if quantName == "Temp":
        percs = percs_low
        ineq = "<"
        log = True
        useDeriv = True

    if quantName == "Entropy":
        percs = percs_low
        ineq = "<"
        log = True
        useDeriv = False

    if quantName == "RadVel":
        percs = percs_low
        ineq = "<"
        log = False
        useDeriv = False

    if quantName == "ShocksEnergyDiss":
        percs = percs_high
        ineq = ">"
        log = True
        useDeriv = False

    if quantName == "ShocksMachNum":
        percs = percs_high
        ineq = ">"
        log = False
        useDeriv = False

    return useDeriv, percs, ineq, log


def healpixThresholdedRadius(sP, pSplit=None, ptType="Gas", quant="Temp", radMax=5, radNumBins=400, Nside=16):
    """Derive virial shock radius for all subhalos using a given algorithm."""
    assert pSplit is None  # not supported
    assert quant in ["Temp", "Entropy", "ShocksMachNum", "ShocksEnergyDiss", "RadVel"]

    # load
    acField = "Subhalo_SphericalSamples_Global_%s_%s_%drvir_%drad_%dns" % (ptType, quant, radMax, radNumBins, Nside)
    ac = sP.auxCat(acField)

    attrs = ac[acField + "_attrs"]
    radPts = attrs["rad_bins_rvir"]
    radBinSize = radPts[1] - radPts[0]  # r/rvir
    subhaloIDs = ac["subhaloIDs"]

    subhalo_r200 = sP.subhalos("rhalo_200_code")

    # config and unit conversions
    useDeriv, percs, ineq, log = _get_threshold_config(quant)

    # allocate
    dummy_results = _thresholded_radius(radPts, ac[acField][0, :, :], 50, "<")

    r = np.zeros((subhaloIDs.size, len(dummy_results), len(percs)), dtype="float32")
    r.fill(np.nan)

    # loop over all subhalos
    printFac = 100.0 if (sP.res > 512) else 10.0

    for i, subhaloID in enumerate(subhaloIDs):
        if i % np.max([1, int(subhaloIDs.size / printFac)]) == 0 and i <= subhaloIDs.size:
            print("   %4.1f%%" % (float(i + 1) * 100.0 / subhaloIDs.size), flush=True)

        # prepare
        vals = ac[acField][i, :, :]
        if log:
            vals = logZeroNaN(vals)

        # partial derivative of quantity with respect to radius
        # sign: negative if quantity decreases moving outwards (from r=0)
        if useDeriv:
            binsize_pkpc = sP.units.codeLengthToKpc(radBinSize * subhalo_r200[subhaloID])  # pkpc
            vals = np.gradient(vals, binsize_pkpc, axis=1)

        # calculate
        for j, perc in enumerate(percs):
            rshock_vals = _thresholded_radius(radPts, vals, perc, ineq)
            r[i, :, j] = rshock_vals

    # return
    desc = "Virial shock (or splashback) radius identified by [%s,%s] algorithm." % (ptType, quant)
    attrs["Description"] = desc.encode("ascii")
    attrs["subhaloIDs"] = subhaloIDs
    attrs["percs"] = percs

    return r, attrs


def local_gas_subset(sP, haloID=0, maxRadR200=5.2, useTree=True):
    """Obtain and cache a set of gas cells in the vicinity of a halo. Debugging only, independent of the auxCats."""
    gas_local = {}

    # cache
    cacheFilename = sP.cachePath + "rshock_subset_%d_h%d_r%.1f.hdf5" % (sP.snap, haloID, maxRadR200)

    if isfile(cacheFilename):
        print("Loading: [%s]" % cacheFilename)
        with h5py.File(cacheFilename, "r") as f:
            for key in f:
                gas_local[key] = f[key][()]

        return gas_local

    # metadata
    halo = sP.halo(haloID)
    print("Creating new local gas subset, halo mass: ", sP.units.codeMassToLogMsun(halo["Group_M_Crit200"]))

    haloPos = halo["GroupPos"]
    maxRad = maxRadR200 * halo["Group_R_Crit200"]

    # global load
    pos = sP.snapshotSubsetP("gas", "pos", float32=True)

    if useTree:
        savePath = sP.derivPath + "tree/"
        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + "tree32_%s_%d_gas.hdf5" % (sP.simName, sP.snap)

        if not isfile(saveFilename):
            # construct tree
            print("Start build of global oct-tree...", reportMemory(), flush=True)
            start_time = time.time()
            tree = buildFullTree(pos, boxSizeSim=sP.boxSize, treePrec="float32", verbose=True)
            print("Tree finished [%.1f sec]." % (time.time() - start_time))

            with h5py.File(saveFilename, "w") as f:
                for i, item in enumerate(tree):
                    f["item_%d" % i] = item
            print("Saved: [%s]" % saveFilename)
        else:
            # load previously saved tree
            print("Loading: [%s]" % saveFilename)
            with h5py.File(saveFilename, "r") as f:
                tree = []
                for item in f:
                    tree.append(f[item][()])

        # tree search
        print("Start tree search...")
        start_time = time.time()
        loc_inds = calcParticleIndices(pos, haloPos, maxRad, boxSizeSim=sP.boxSize, tree=tree)
        print("Tree search finished [%.1f sec]." % (time.time() - start_time))

        if 0:
            # brute-force verify
            print("Start brute-force verify...")
            start_time = time.time()
            dists = sP.periodicDistsN(haloPos, pos)
            loc_inds2 = np.where(dists <= maxRad)[0]

            zz = np.argsort(loc_inds)
            zz = loc_inds[zz]
            assert np.array_equal(zz, loc_inds2)
            print("Verify finished [%.1f sec]." % (time.time() - start_time))
    else:
        # avoid tree, simply do brute-force distance search
        print("Start brute-force search...")
        start_time = time.time()

        dists = sP.periodicDistsN(haloPos, pos)
        loc_inds = np.where(dists <= maxRad)[0]

        print("Brute-force search finished [%.1f sec]." % (time.time() - start_time))

    # take local subset
    gas_local["inds"] = loc_inds
    gas_local["pos"] = pos[loc_inds]

    gas_local["rad"] = sP.periodicDists(haloPos, gas_local["pos"])
    gas_local["rad"] /= halo["Group_R_Crit200"]

    # shocks_dedt, temp, entr
    gas_local["ShocksEnergyDiss"] = sP.snapshotSubsetC("gas", "shocks_dedt", inds=loc_inds)
    gas_local["ShocksEnergyDiss"] = sP.units.codeEnergyRateToErgPerSec(gas_local["ShocksEnergyDiss"])
    gas_local["ShocksMachNum"] = sP.snapshotSubsetC("gas", "shocks_machnum", inds=loc_inds)

    gas_local["Temp"] = sP.snapshotSubsetC("gas", "temp", inds=loc_inds)
    gas_local["Entropy"] = sP.snapshotSubsetC("gas", "entr", inds=loc_inds)

    # vrad
    sub = sP.subhalo(halo["GroupFirstSub"])
    sP.refPos = sub["SubhaloPos"]
    sP.refVel = sub["SubhaloVel"]

    gas_local["RadVel"] = sP.snapshotSubsetC("gas", "vrad", inds=loc_inds)

    # save cache
    with h5py.File(cacheFilename, "w") as f:
        for key in gas_local:
            f[key] = gas_local[key]
    print("Saved: [%s]" % cacheFilename)

    # run algorithm
    return gas_local


def virialShockRadiusSingle(sP, haloID, useExistingAuxCat=True):
    """Exploration: given gas information around a halo, use Healpix sampling to determine the virial shock radius."""
    # config
    nRad = 400
    Nside = 16

    clim_percs = [5, 99]  # colorbar range

    halo = sP.halo(haloID)

    if not useExistingAuxCat:
        # get sample points, shift to box coords, halo centered (handle periodic boundaries)
        pts, nProj, radPts, radBinSize = healpix_shells_points(nRad=nRad, Nside=Nside)

        pts *= halo["Group_R_Crit200"]
        pts += halo["GroupPos"][np.newaxis, :]

        sP.correctPeriodicPosVecs(pts)

        # load a particle subset
        p = local_gas_subset(sP, haloID=haloID)

        # construct tree
        print("build tree...")
        tree = buildFullTree(p["pos"], boxSizeSim=sP.boxSize, treePrec="float32")

        # derive hsml (one per sample point)
        nNGB = 20
        nNGBDev = 1

        print("calc hsml...")
        hsml = calcHsml(p["pos"], sP.boxSize, posSearch=pts, nNGB=nNGB, nNGBDev=nNGBDev, tree=tree)

    # sample and plot different quantiites
    for field in ["Temp", "Entropy", "RadVel", "ShocksMachNum", "ShocksEnergyDiss"]:
        saveBase = "healpix_%s_z%d_h%d_ns%d_nr%d_%s" % (sP.simName, sP.redshift, haloID, Nside, nRad, field)

        if field == "ShocksEnergyDiss":
            label = "Shock Energy Dissipation [ log $10^{30}$ erg/s ]"
            label2 = r"log( $\dot{E}_{\rm shock}$ / <$\dot{E}_{\rm shock}$> )"
            label3 = r"$\partial$$\dot{E}_{\rm shock}$/$\partial$r [ log $10^{30}$ erg/s kpc$^{-1}$ ]"

        if field == "ShocksMachNum":
            label = "Shock Mach Number [ linear ]"
            label2 = r"log( $\mathcal{M}_{\rm shock}$ / <$\mathcal{M}_{\rm shock}$> )"
            label3 = r"$\partial$$\mathcal{M}_{\rm shock}$/$\partial$r [ linear kpc$^{-1}$ ]"

        if field == "Temp":
            label = "Temperature [ log K ]"
            label2 = r"log( $\delta$T / <T> )"
            label3 = r"$\partial$T/$\partial$r [ log K kpc$^{-1}$ ]"

        if field == "Entropy":
            label = "Entropy [ log K cm$^2$ ]"
            label2 = r"log( $\delta$S / <S> )"
            label3 = r"$\partial$S/$\partial$r [ log K cm$^2$ kpc$^{-1}$ ]"

        if field == "RadVel":
            label = "Radial Velocity [ km/s ]"
            label2 = r"log( v$_{\rm rad}$ / <v$_{\rm rad}$> )"
            label3 = r"$\partial$v$_{\rm rad}$/$\partial$r [ km/s kpc$^{-1}$ ]"

        # load existing or calculate now?
        if useExistingAuxCat:
            ptType = "Gas"
            acField = "Subhalo_SphericalSamples_Global_%s_%s_5rvir_%drad_%dns" % (ptType, field, nRad, Nside)
            print("Loading existing [%s]..." % acField)

            # load
            ac = sP.auxCat(acField, subhaloIDs=halo["GroupFirstSub"])
            result = ac[acField]

            radPts = ac[acField + "_attrs"]["rad_bins_rvir"]
            radBinSize = radPts[1] - radPts[0]  # r/rvir
        else:
            op = "kernel_mean"  # mean
            print("Total number of samples [%d], running [%s]..." % (pts.shape[0], field))

            quant = p[field]
            if field in ["ShocksEnergyDiss"]:
                quant /= 1e30  # avoid float32 overflow

            result = calcQuantReduction(p["pos"], quant, hsml, op, sP.boxSize, posSearch=pts, tree=tree)
            result = np.reshape(result, (nProj, nRad))

        # unit conversions
        useDeriv, percs, ineq, log = _get_threshold_config(field)

        if log:
            # result = logZeroSafe(result, zeroVal=result[result>0].min()) # for shocks_dedt
            result = logZeroNaN(result)

        # plot quantity
        clim = np.nanpercentile(result, clim_percs)
        clim = np.round(clim * 10) / 10

        plotHealpixShells(radPts, result, label=label, clim=clim, saveFilename=saveBase + ".pdf")

        # plot quantity relative to its average at that radius (subtract out radial profile)
        if field in ["ShocksEnergyDiss", "Temp", "Entropy"]:
            if np.isfinite(result[:, 0]).sum() == 0:
                result[:, 0] = 1.0  # avoid all nan slice
            rad_mean = np.nanmean(10.0**result, axis=0)
            rad_mean[np.where(rad_mean == 0)] = 1.0  # avoid division by zero (first, r=0 bin)
            result_norm = logZeroNaN(10.0**result / rad_mean)

        if field in ["RadVel", "ShocksMachNum"]:
            rad_mean = np.nanmean(result, axis=0)
            rad_mean[np.where(rad_mean == 0)] = 1.0  # avoid division by zero (first, r=0 bin)
            result_norm = logZeroNaN(result / rad_mean)

        clim_val = np.abs(np.nanpercentile(result_norm, clim_percs)).min()
        clim_val = np.clip(clim_val, 0.001, np.inf)
        clim2 = np.array([-clim_val, clim_val])
        if clim2[0] < 10.0:
            clim2 = np.round(clim2 * 10) / 10

        plotHealpixShells(radPts, result_norm, label=label2, clim=clim2, saveFilename=saveBase + "_norm.pdf")

        # plot partial derivative of quantity with respect to radius
        radBinSize = sP.units.codeLengthToKpc(radBinSize * halo["Group_R_Crit200"])  # pkpc

        # sign: negative if quantity decreases moving outwards (from r=0)
        result_deriv = np.gradient(result, radBinSize, axis=1)

        clim_val = np.abs(np.nanpercentile(result_deriv, clim_percs)).min()
        clim_val = np.clip(clim_val, 0.001, np.inf)
        clim3 = np.array([-clim_val, clim_val])
        if clim3[0] < 10.0:
            clim3 = np.round(clim3 * 100) / 100

        plotHealpixShells(
            radPts, result_deriv, label=label3, clim=clim3, ctName="curl", saveFilename=saveBase + "_deriv.pdf"
        )

        # flag dquant/dr pixels below a threshold, and plot 1d histogram of their radii
        # vals = result if not useDeriv else result_deriv

        # for perc in percs:
        #    # derive rshock and save debug plots
        #    base = saveBase + "_thresh%g" % perc

        #    rshock_vals = _thresholded_radius(radPts, vals, perc, ineq, saveBase=base)
        # todo: finish


def plotRshockVsMass(sPs, quants=("Temp_400rad_16ns",), vsHaloMass=True, kpc=False, percInds=None, methodInds=None):
    """Plot a particular virial shock radius measurement vs halo/stellar mass."""
    binSize = 0.1  # log mass

    if percInds is None and methodInds is None:
        assert len(sPs) == 1 and len(quants) == 1
    if percInds is None or methodInds is None:
        assert len(sPs) == 1 or len(quants) == 1

    # plot setup
    fig, ax = plt.subplots()

    mHaloLabel = r"Halo Mass [ log M$_{\rm sun}$ ]"
    mHaloField = "mhalo_200_log"
    mStarLabel = r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ]"
    mStarField = "mstar_30pkpc_log"

    if vsHaloMass:
        ax.set_xlim([10.5, 15.1])
        ax.set_xlabel(mHaloLabel)
        massField = mHaloField
    else:
        ax.set_xlim([7.5, 12.0])
        ax.set_xlabel(mStarLabel)
        massField = mStarField

    if kpc:
        ax.set_ylim([1.5, 3.5])
        ax.set_ylabel("Virial Shock Radius [ log pkpc ]")
        ax.plot(ax.get_xlim(), [2.0, 2.0], "-", color="black", alpha=0.1)
        ax.plot(ax.get_xlim(), [3.0, 3.0], "-", color="black", alpha=0.1)
    else:
        ax.set_ylim([0.5, 5.0])
        ax.set_ylabel(r"R$_{\rm shock}$ / R$_{\rm vir}$")
        ax.plot(ax.get_xlim(), [1.0, 1.0], "-", color="black", alpha=0.1)

    # helper for quant labels
    p1 = [quant.split("_")[0] for quant in quants]
    p2 = [quant.split("_")[1] for quant in quants]
    p3 = [quant.split("_")[2] for quant in quants]

    allQuantsSame = p1.count(p1[0]) == len(p1)
    allnRadSame = p2.count(p2[0]) == len(p2)
    allNsideSame = p3.count(p3[0]) == len(p3)

    # loop over each fullbox run
    for i, sP in enumerate(sPs):
        # load halo/stellar masses, rshock auxcat
        gc = sP.subhalos([massField, "rhalo_200"])

        # loop over auxCats (e.g. 'Temp_400rad_16ns' can vary quant, nRad, and/or Nside)
        for j, quant in enumerate(quants):
            print("[%s]: %s" % (sP.simName, quant))

            # load auxCat
            acField = "Subhalo_VirShockRad_" + quant
            ac = sP.auxCat(acField)

            xx = gc[massField][ac["subhaloIDs"]]

            # unit conversions
            yy_kpc = ac[acField] * gc["rhalo_200"][ac["subhaloIDs"], np.newaxis, np.newaxis]  # [rvir] -> [pkpc]
            yy_kpc = logZeroNaN(yy_kpc)  # log pkpc

            yy_rvir = ac[acField]  # [rvir]

            # determine percs and methods to show (specified in input, or all)
            if methodInds is None:
                methodInds = np.arange(ac[acField].shape[1])  # all

            if percInds is None:
                percInds = np.arange(ac[acField].shape[2])  # all

            # loop over all combinations of {perc,method}
            for k, percInd in enumerate(percInds):
                for l, methodInd in enumerate(methodInds):
                    print(" p = %d m = %d" % (percInd, methodInd))
                    perc = ac[acField + "_attrs"]["percs"][percInd]

                    yy_loc_kpc = np.squeeze(yy_kpc[:, methodInd, percInd])
                    yy_loc_rvir = np.squeeze(yy_rvir[:, methodInd, percInd])

                    # calculate median and smooth
                    percs = [10, 50, 90]

                    xm, _, _, pm_kpc = running_median(xx, yy_loc_kpc, binSize=binSize, skipZeros=True, percs=percs)
                    xm2, _, _, pm_rvir = running_median(xx, yy_loc_rvir, binSize=binSize, skipZeros=True, percs=percs)
                    assert np.array_equal(xm, xm2)

                    if xm.size > sKn:
                        pm_kpc = savgol_filter(pm_kpc, sKn, sKo, axis=1)
                        pm_rvir = savgol_filter(pm_rvir, sKn, sKo, axis=1)

                    # select
                    pm = pm_kpc if kpc else pm_rvir

                    # determine quantName for label
                    quantName = quant
                    if quant.count("_") == 2:
                        quantName, nRad, Nside = quant.split("_")  # e.g. 'Temp_400rad_16ns'
                    elif quant.count("_") == 3:
                        quantName, rvirFac, nRad, Nside = quant.split("_")  # e.g. 'Temp_10rvir_800rad_16ns'
                        rvirFac = "($%dr_{\\rm vir}$)" % int(rvirFac.replace("rvir", ""))
                    nRad = "$N_{\\rm rad} = %d$" % int(nRad.replace("rad", ""))
                    Nside = "$N_{\\rm side} = %d$" % int(Nside.replace("ns", ""))
                    quantLabel = ""

                    if not allQuantsSame:
                        quantLabel += quantName
                    if not allnRadSame:
                        quantLabel += " " + nRad
                    if not allNsideSame:
                        quantLabel += " " + Nside
                    if quant.count("_") == 3:
                        quantLabel += " " + rvirFac

                    # determine color, linestyle, and label
                    label = None

                    if len(sPs) == 1:
                        # one run, color varies by quant or perc
                        ind = j
                        if len(percInds) > 1:
                            ind = k
                        c = colors[ind]

                        # linestyle varies by quant or perc or method
                        ls = linestyles[j + l]

                        label = quantLabel
                        if len(percInds) > 1 and l == 0:
                            label += " p=%s" % perc
                    else:
                        # multiple runs/redshifts, color varies by run
                        c = colors[i]

                        # linestyle varies by quant or perc
                        if len(percInds) > 1:
                            ls = linestyles[j + k]
                        if len(percInds) == 1:
                            ls = linestyles[j + l]

                        if k == 0 and l == 0:
                            label = sP.simName
                        if sPs[1].redshift != sPs[0].redshift:
                            label += " (z=%.1f)" % sP.redshift
                        if len(quants) > 1:
                            label += " %s" % quantLabel
                        if len(percInds) > 1 and len(sPs) == 1:
                            label += " p=%s" % perc
                        if len(methodInds) > 1 and len(sPs) == 1:
                            label += " m=%d" % methodInd

                    # plot median line
                    ax.plot(xm, pm[1, :], color=c, linestyle=ls, label=label)

                    if k == 0 and l == 0:
                        # show percentile scatter (for all runs/quants)
                        ax.fill_between(xm, pm[0, :], pm[-1, :], color=c, interpolate=True, alpha=0.2)

    # legends
    sExtra = []
    lExtra = []

    if len(percInds) > 1 and len(methodInds) == 1 and len(quants) == 1:
        for i, percInd in enumerate(percInds):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            lExtra += ["perc=%s" % ac[acField + "_attrs"]["percs"][percInd]]

    if len(methodInds) > 1:
        for i, methodInd in enumerate(methodInds):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            lExtra += ["method #%d" % methodInd]

    if len(quants) == 1 and len(percInds) == 1 and len(methodInds) == 1:
        if len(sPs) > 1 and sPs[1].redshift == sPs[0].redshift:
            sExtra += [plt.Line2D([0], [0], color="black", lw=0, linestyle="-", marker="")]
            lExtra += ["z = %.1f" % sPs[0].redshift]

    legend1 = ax.legend(sExtra, lExtra, loc="upper right")
    ax.add_artist(legend1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")

    qStr = quants[0] if len(quants) == 1 else "%dquants" % len(quants)
    sStr = "-".join([sP.simName for sP in sPs])
    mStr = "m%d" % methodInds[0] if len(methodInds) == 1 else "m-all"
    pStr = "p%d" % percInds[0] if len(percInds) == 1 else "p-all"
    massAxis = "mhalo" if vsHaloMass else "mstar"
    fig.savefig(
        "rshock_vs_%s_%s_z%.1f_%s_%s_%s_%s.pdf"
        % (massAxis, sStr, sPs[0].redshift, "kpc" if kpc else "rvir", qStr, mStr, pStr)
    )
    plt.close(fig)


def visualizeHaloVirialShock(sP, haloID, conf=0, depthFac=1.0, dataCache=None):
    """Driver for a single halo vis example, highlighting the virial shock structure."""
    run = sP.run
    res = sP.res
    redshift = sP.redshift
    subhaloInd = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

    rVirFracs = [1.0, 2.0, 3.0, 4.0]
    method = "sphMap_global"
    nPixels = [1000, 1000]
    axes = [0, 2]  # TODO: change back [0,1]

    labelZ = False
    labelScale = "physical"
    labelSim = False
    labelHalo = True

    size = 9.0
    sizeType = "rVirial"

    # global pre-cache of selected fields into memory
    if 1 and dataCache is None:
        dataCache = {}
        cacheKeys = ["Coordinates", "Masses"] + ["EnergyDissipation"]
        for key in cacheKeys:
            print("Caching [%s] now..." % key, flush=True)
            dataCache["snap%d_gas_%s" % (sP.snap, key)] = sP.snapshotSubsetP("gas", key, float32=True)
        print("All caching done.", flush=True)

    # panel
    partType = "gas"

    if conf == 0:
        partField = "shocks_dedt"
        valMinMax = [34.0, 38.0]
    if conf == 1:
        partField = "shocks_machnum"
        valMinMax = [0.0, 5.0]
    if conf == 2:
        partField = "temp"
        valMinMax = [4.0, 7.2]
    if conf == 3:
        partField = "entropy"
        valMinMax = [3.0, 6.0]
    if conf == 4:
        partField = "vrad"
        valMinMax = [-350, 350]
        depthFac = 0.1  # todo

    if conf == 5:
        # widescreen image
        nPixels = [1920, 1080]
        partField = "xray"
        valMinMax = [30.0, 39.0]
        rVirFracs = [1.0]
        size = 4.5

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels
        colorbars = True
        saveFilename = "./vis_virshock_%s_%d_%s_h%d.pdf" % (sP.simName, sP.snap, partField, haloID)

    # render
    renderSingleHalo([{}], plotConfig, locals(), skipExisting=False)

    return dataCache


def paperPlots():
    """Generate all plots for virial shock radius paper."""
    # TNG50_z0 = simParams(run='tng50-1', redshift=0.0)
    # TNG50_2_z0 = simParams(run='tng50-2', redshift=0.0)

    # figure 1: single-halo visualization of the virial shock
    if 0:
        sP = simParams(run="tng50-1", redshift=2.0)

        for conf in [0]:
            for haloID in [0, 10, 20, 110, 120, 200, 300]:
                for depthFac in [1.0, 0.5, 0.22]:
                    visualizeHaloVirialShock(sP, haloID=haloID, conf=conf, depthFac=depthFac)

    # figure 2: rshock vs mass
    if 0:
        quants = ["ShocksMachNum_400rad_16ns"]
        percInds = [2]
        methodInds = [2]
        kpc = False
        redshift = 0.0
        runs = ["tng50-1", "tng100-1", "tng300-1"]

        sPs = [simParams(run=run, redshift=redshift) for run in runs]

        for vsHaloMass in [True, False]:
            plotRshockVsMass(
                sPs, quants=quants, vsHaloMass=vsHaloMass, kpc=kpc, percInds=percInds, methodInds=methodInds
            )

    # figure 3: 2d plot explore
    if 0:
        sP = simParams(run="tng300-1", redshift=2.0)

        xQuant = "mhalo_200_log"
        cQuant = "fgas2"
        yQuant = "rshock_ShocksMachNum_m2p2"
        xlim = [11.8, 14.0]
        ylim = [0.0, 5.0]
        clim = [-2.0, 0.0]
        cRel = None  # [0.65,1.35,False] # [cMin,cMax,cLog] #None
        params = {
            "cenSatSelect": "cen",
            "cStatistic": "median_nan",
            "cQuant": cQuant,
            "xQuant": xQuant,
            "xlim": xlim,
            "ylim": ylim,
            "clim": clim,
            "cRel": cRel,
        }

        subhalos.histogram2d(sP, yQuant=yQuant, **params)

    # figure X: explore 5rvir vs 10rvir max radius
    if 0:
        quants = ["ShocksMachNum_400rad_16ns", "ShocksMachNum_10rvir_800rad_16ns"]
        percInds = [2]  # None
        methodInds = [2]  # None

        for redshift in [0.0, 1.0, 2.0]:
            sP = simParams(run="tng50-2", redshift=redshift)

            plotRshockVsMass([sP], quants=quants, vsHaloMass=True, kpc=False, percInds=percInds, methodInds=methodInds)

    # figure X: explore rshock vs mass (percs, methods, runs, ...)
    if 0:
        quants = ["ShocksMachNum_400rad_16ns"]
        percInds = None  # [3]
        methodInds = None  # [3]

        TNG100 = simParams(run="tng100-1", redshift=2.0)
        # TNG50 = simParams(run='tng50-2', redshift=1.0)

        for vsHaloMass in [True]:
            for kpc in [True, False]:
                plotRshockVsMass(
                    [TNG100], quants=quants, vsHaloMass=vsHaloMass, kpc=kpc, percInds=percInds, methodInds=methodInds
                )

    # figure X: explore rshock vs mass (different quants)
    if 0:
        quantNames = ["Temp", "Entropy", "ShocksMachNum", "ShocksEnergyDiss", "RadVel"]
        quants = ["%s_400rad_16ns" % q for q in quantNames]

        percInds = [2]
        methodInds = [2]
        vsHaloMass = True

        sP = simParams(run="tng100-1", redshift=2.0)

        for kpc in [True, False]:
            plotRshockVsMass(
                [sP], quants=quants, vsHaloMass=vsHaloMass, kpc=kpc, percInds=percInds, methodInds=methodInds
            )

    # figure appendix: testing rshock detection method, associated plots
    if 0:
        haloID = 20  # 0,10,20,110,120,200,300
        sP = simParams(run="tng50-2", redshift=2.0)

        virialShockRadiusSingle(sP, haloID, useExistingAuxCat=True)

    # figure test: why methods start to fail towards z=0
    if 1:
        haloID = 100
        sP = simParams(run="tng50-1", redshift=0.0)
        print("halo mass: ", sP.units.codeMassToLogMsun(sP.halo(haloID)["Group_M_Crit200"]))

        virialShockRadiusSingle(sP, haloID, useExistingAuxCat=True)

    # TODO: check on L75n910FP_NR
    # TODO: make time evolution plot with time along x-axis, distance along y-axis, color=quant
    #  -- see if our method may need to be explicitly time aware


# add auxcats
from temet.load.auxcat_fields import def_fields as ac


ac["Subhalo_VirShockRad_Temp_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="Temp", radNumBins=400, Nside=16
)
ac["Subhalo_VirShockRad_Temp_400rad_8ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="Temp", radNumBins=400, Nside=8
)
ac["Subhalo_VirShockRad_Entropy_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="Entropy", radNumBins=400, Nside=16
)
ac["Subhalo_VirShockRad_ShocksMachNum_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="ShocksMachNum", radNumBins=400, Nside=16
)
ac["Subhalo_VirShockRad_ShocksMachNum_10rvir_800rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="ShocksMachNum", radNumBins=800, radMax=10, Nside=16
)
ac["Subhalo_VirShockRad_ShocksEnergyDiss_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="ShocksEnergyDiss", radNumBins=400, Nside=16
)
ac["Subhalo_VirShockRad_RadVel_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Gas", quant="RadVel", radNumBins=400, Nside=16
)
ac["Subhalo_SplashbackRad_DM_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="DM", quant="RadVel", radNumBins=400, Nside=16
)
ac["Subhalo_SplashbackRad_Stars_400rad_16ns"] = partial(
    healpixThresholdedRadius, ptType="Stars", quant="RadVel", radNumBins=400, Nside=16
)
