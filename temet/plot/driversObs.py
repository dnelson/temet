"""
Summary plots comparing one or more cosmological simulations to observational constraints.

Includes the usual comparisons such as stellar to halo mass ratio (SMHM), the stellar mass function (SMF), and so on.
"""

from datetime import datetime
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter

from ..load.simtxt import sfrTxt
from ..plot import snapshot, subhalos
from ..plot.config import binSize, colors, figsize, linestyles, lw, sKn, sKo
from ..plot.driversSizes import galaxyHISizeMass, galaxySizes
from ..util.helper import iterable, logZeroNaN, running_histogram, running_median
from ..util.simParams import simParams
from .subhalos import addRedshiftAgeAxes


def stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=False, use30kpc=False, simRedshift=0.0, dataRedshift=0.0):
    """Stellar mass vs. halo mass relation."""
    from ..load.data import behrooziSMHM, kravtsovSMHM, mosterSMHM

    # plot setup
    xrange = [10.0, 15.0]
    yrange = [0.0, 0.30]
    nIndivPoints = 10
    if dataRedshift is not None and dataRedshift > 0.0:
        yrange[1] = 0.25

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    if ylog:
        ax.set_yscale("log")
        ax.set_ylim([1e-3, 1e0])

    ax.set_xlabel(r"M$_{\rm halo}$ [ log M$_{\rm sun}$ ] [ M$_{\rm 200c}$ ]")
    ax.set_ylabel(r"M$_\star$ / M$_{\rm halo}$ $(\Omega_{\rm b} / \Omega_{\rm m})^{-1}$ [ only centrals ]")

    # observational data: abundance matching constraints
    b = behrooziSMHM(sPs[0], redshift=dataRedshift)
    m = mosterSMHM(sPs[0], redshift=dataRedshift)
    k = kravtsovSMHM(sPs[0])

    ax.plot(b["haloMass_i"], b["y_mid_i"], color="#333333", label="Behroozi+ (2013) z=%d" % dataRedshift)
    ax.fill_between(b["haloMass_i"], b["y_low_i"], b["y_high_i"], color="#333333", interpolate=True, alpha=0.3)

    ax.plot(m["haloMass"], m["y_mid"], color="#dddddd", label="Moster+ (2013) z=%d" % dataRedshift)
    ax.fill_between(m["haloMass"], m["y_low"], m["y_high"], color="#dddddd", interpolate=True, alpha=0.3)

    if dataRedshift == 0.0:
        ax.plot(k["haloMass"], k["y_mid"], color="#888888", label="Kravtsov+ (2014) z=0")

    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc="upper right")
    plt.gca().add_artist(legend1)

    # loop over each run
    lines = []

    for i, sP in enumerate(sPs):
        sP.setRedshift(simRedshift)
        print("SMHM (z=%d): %s (z=%d)" % (dataRedshift, sP.simName, sP.redshift))

        if sP.isZoom:
            gc = sP.groupCatSingle(subhaloID=sP.zoomSubhaloID)
            gh = sP.groupCatSingle(haloID=gc["SubhaloGrNr"])

            # halo mass definition
            xx_code = gh["Group_M_Crit200"]  # gc['SubhaloMass']
            xx = sP.units.codeMassToLogMsun(xx_code)

            # stellar mass definition(s)
            yy = gc["SubhaloMassType"][4] / xx_code / (sP.omega_b / sP.omega_m)
            ax.plot(xx, yy, sP.marker, color=colors[0])

            yy = gc["SubhaloMassInRadType"][4] / xx_code / (sP.omega_b / sP.omega_m)
            ax.plot(xx, yy, sP.marker, color=colors[1])

            yy = gc["SubhaloMassInHalfRadType"][4] / xx_code / (sP.omega_b / sP.omega_m)
            ax.plot(xx, yy, sP.marker, color=colors[2])

        else:
            # fullbox:
            gc = sP.groupCat(
                fieldsHalos=["GroupFirstSub", "Group_M_Crit200"],
                fieldsSubhalos=["SubhaloMass", "SubhaloMassType", "SubhaloMassInRadType", "SubhaloMassInHalfRadType"],
            )

            label = sP.simName + " z=%d" % sP.redshift

            # centrals only
            wHalo = np.where((gc["halos"]["GroupFirstSub"] >= 0) & (gc["halos"]["Group_M_Crit200"] > 0))
            w = gc["halos"]["GroupFirstSub"][wHalo]

            # halo mass definition
            xx_code = gc["halos"]["Group_M_Crit200"][wHalo]
            xx = sP.units.codeMassToLogMsun(xx_code)

            # stellar mass definition(s)
            if use30kpc:
                # load auxcat
                field = "Subhalo_Mass_30pkpc_Stars"
                ac = sP.auxCat(fields=[field])

                yy = ac[field][w] / xx_code / (sP.omega_b / sP.omega_m)
                xm, ym, sm = running_median(xx, yy, binSize=binSize)
                ym2 = savgol_filter(ym, sKn, sKo)
                (l,) = ax.plot(xm[:-1], ym2[:-1], linestyles[0], c=colors[i], lw=lw, label=label)
                lines.append(l)

            if allMassTypes:
                yy = gc["subhalos"]["SubhaloMassType"][w, 4] / xx_code / (sP.omega_b / sP.omega_m)
                xm, ym, sm = running_median(xx, yy, binSize=binSize)
                ym2 = savgol_filter(ym, sKn, sKo)
                (l,) = ax.plot(xm[:-1], ym2[:-1], linestyles[1], c=colors[i], lw=lw)

            if not use30kpc or allMassTypes:
                # primary (in 2rhalf_stars)
                yy = gc["subhalos"]["SubhaloMassInRadType"][w, 4] / xx_code / (sP.omega_b / sP.omega_m)
                xm, ym, sm = running_median(xx, yy, binSize=binSize)
                ym2 = savgol_filter(ym, sKn, sKo)
                (l,) = ax.plot(xm[:-1], ym2[:-1], linestyles[0], c=colors[i], lw=lw, label=label)
                lines.append(l)

            if allMassTypes:
                yy = gc["subhalos"]["SubhaloMassInHalfRadType"][w, 4] / xx_code / (sP.omega_b / sP.omega_m)
                xm, ym, sm = running_median(xx, yy, binSize=binSize)
                ym2 = savgol_filter(ym, sKn, sKo)
                (l,) = ax.plot(xm[:-1], ym2[:-1], linestyles[2], c=colors[i], lw=lw)

            # individual N most massive points
            ax.plot(xx[0:nIndivPoints], yy[0:nIndivPoints], "o", color=l.get_color(), alpha=0.9)

    # second legend
    markers = []
    sExtra = []
    lExtra = []

    for handle in lines:
        sExtra.append(handle)
        lExtra.append(handle.get_label())

    if allMassTypes:
        sExtra += [
            plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[1]),
            plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[0]),
            plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[2]),
        ]
        lExtra += [r"$M_\star^{\rm tot}$", r"$M_\star^{< 2r_{1/2}}$", r"$M_\star^{< r_{1/2}}$"]
    if use30kpc:
        sExtra += [plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[0])]
        lExtra += [r"$M_\star$ (< 30 pkpc)"]

    for sP in sPs:
        if not sP.isZoom or sP.marker in markers:
            continue
        sExtra.append(plt.Line2D([0], [0], color="black", marker=sP.marker, linestyle="", label="test"))
        lExtra.append(sP.simName + " z=%d" % sP.redshift)
        markers.append(sP.marker)

    ax.legend(sExtra, lExtra, loc="upper left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def sfrAvgVsRedshift(sPs, pdf):
    """Average SFRs in some halo mass bins vs. redshift vs. abundance matching lines."""
    from ..load.data import behrooziSFRAvgs

    # config
    plotMassBins = [10.6, 11.2, 11.8]
    massBinColors = ["#333333", "#666666", "#999999"]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_ylim([8e-3, 5e2])
    addRedshiftAgeAxes(ax, sPs[0])
    ax.set_ylabel(r"<SFR> [ M$_{\rm sun}$ / yr ] [ < 2r$_{1/2}$ ] [ only centrals ]")
    ax.set_yscale("log")

    # calculate and cache from simulations function
    def _loadSfrAvg(sP, haloMassBins, haloBinSize, maxNumSnaps=60):
        """Helper function to calculate average SFR in halo mass bins across snapshots."""
        snaps = sP.validSnapList(maxNum=maxNumSnaps)

        saveFilename = sP.derivPath + "sfr_avgs_%d-%d_%d.hdf5" % (snaps.min(), snaps.max(), len(snaps))

        if isfile(saveFilename):
            print(" Loaded: [%s]" % saveFilename.split(sP.derivPath)[1])
            r = {}
            with h5py.File(saveFilename, "r") as f:
                for key in f:
                    r[key] = f[key][()]
            return r

        # allocate
        sfrFields = ["SubhaloSFR", "SubhaloSFRinRad", "SubhaloSFRinHalfRad"]

        r = {}

        r["haloMassBins"] = haloMassBins
        r["haloBinSize"] = haloBinSize
        r["redshifts"] = np.zeros(len(snaps))

        r["sfrs_med"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")
        r["sfrs_mean"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")
        r["sfrs_std"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")
        r["sfrs_med_noZero"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")
        r["sfrs_mean_noZero"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")
        r["sfrs_std_noZero"] = np.zeros((len(snaps), len(haloMassBins), 3), dtype="float32")

        r["sfrs_med"].fill(np.nan)
        r["sfrs_mean"].fill(np.nan)
        r["sfrs_std"].fill(np.nan)
        r["sfrs_med_noZero"].fill(np.nan)
        r["sfrs_mean_noZero"].fill(np.nan)
        r["sfrs_std_noZero"].fill(np.nan)

        # loop over all snapshots
        for j, snap in enumerate(snaps):
            print(" snap %d [%d of %d]" % (snap, j, len(snaps)))
            sP.setSnap(snap)

            gc = sP.groupCat(fieldsHalos=["GroupFirstSub", "Group_M_Crit200"], fieldsSubhalos=sfrFields)

            if not gc["halos"]["count"]:
                continue  # high redshift

            r["redshifts"][j] = sP.redshift

            # centrals only, given halo mass definition, in this halo mass bin
            for k, haloMassBin in enumerate(haloMassBins):
                haloMassesLogMsun = sP.units.codeMassToLogMsun(gc["halos"]["Group_M_Crit200"])

                wHalo = np.where(
                    (gc["halos"]["GroupFirstSub"] >= 0)
                    & (haloMassesLogMsun > haloMassBin - 0.5 * haloBinSize)
                    & (haloMassesLogMsun <= haloMassBin + 0.5 * haloBinSize)
                )

                if len(wHalo[0]) == 0:
                    continue

                w = gc["halos"]["GroupFirstSub"][wHalo]

                # sfr definition(s)
                for m, sfrField in enumerate(sfrFields):
                    r["sfrs_med"][j, k, m] = np.median(gc["subhalos"][sfrField][w])
                    r["sfrs_mean"][j, k, m] = np.mean(gc["subhalos"][sfrField][w])
                    r["sfrs_std"][j, k, m] = np.std(gc["subhalos"][sfrField][w])

                    # repeat but exclude all SFR==0 entries
                    loc_w = np.where(gc["subhalos"][sfrField][w] > 0.0)

                    if len(loc_w[0]) == 0:
                        continue

                    r["sfrs_med_noZero"][j, k, m] = np.median(gc["subhalos"][sfrField][w][loc_w])
                    r["sfrs_mean_noZero"][j, k, m] = np.mean(gc["subhalos"][sfrField][w][loc_w])
                    r["sfrs_std_noZero"][j, k, m] = np.std(gc["subhalos"][sfrField][w][loc_w])

        # save
        with h5py.File(saveFilename, "w") as f:
            for key in r:
                f[key] = r[key]
        print(" Saved: [%s]" % saveFilename.split(sP.derivPath)[1])
        return r

    # load observational data
    b = behrooziSFRAvgs()

    for i, massBin in enumerate(plotMassBins):
        xx = b[str(massBin)]["redshift"]
        yy = b[str(massBin)]["sfr"]
        yyDown = b[str(massBin)]["sfr"] - b[str(massBin)]["errorDown"]
        yyUp = b[str(massBin)]["sfr"] + b[str(massBin)]["errorUp"]

        label = r"Behroozi+ (2013) $10^{" + str(massBin) + r"}$ M$_{\rm sun}$ Halos"
        (l,) = ax.plot(xx, yy, label=label, color=massBinColors[i])
        ax.fill_between(xx, yyDown, yyUp, color=l.get_color(), interpolate=True, alpha=0.3)

    # loop over each fullbox run
    for i, sP in enumerate(sPs):
        if sP.isZoom:
            continue

        print("SFRavg: " + sP.simName)

        # load saved simulation data
        simData = _loadSfrAvg(sP, b["haloMassBins"], b["haloBinSize"])
        xx = simData["redshifts"]

        # plot line for each halo mass bin
        for haloMassBin in plotMassBins:
            # locate this mass bin in saved data
            k = np.where(simData["haloMassBins"] == haloMassBin)[0]
            assert len(k) == 1

            # different sfr definitions
            for j in [1]:  # <2r1/2
                label = sP.simName if (haloMassBin == plotMassBins[0] and j == 1) else ""

                # ax.plot(xx, simData['sfrs_med'][:,k,j], ':', color=c, lw=lw, label=label)
                ax.plot(xx, simData["sfrs_med_noZero"][:, k, j], "-", color=colors[i], lw=lw, label=label)

                # if sP == sPs[0] and j == 1:
                #    yy_down = simData['sfrs_med_noZero'][:,k,j] - simData['sfrs_std_noZero'][:,k,j]
                #    yy_up = simData['sfrs_med_noZero'][:,k,j] + simData['sfrs_std_noZero'][:,k,j]
                #    ax.fill_between(xx, np.squeeze(yy_down), np.squeeze(yy_up),
                #                    color=c, interpolate=True, alpha=0.2)

    # legend
    ax.legend(loc="upper left")

    pdf.savefig()
    plt.close(fig)


def sfrdVsRedshift(sPs, pdf, xlog=True, addSubhalosOnly=False):
    """Star formation rate density of the universe, vs redshift, vs observational points."""
    from ..load.data import behrooziObsSFRD, bouwensSFRD2014, eniaSFRD2022

    # plot setup
    fig, ax = plt.subplots()

    ax.set_ylim([5e-4, 5e-1])
    addRedshiftAgeAxes(ax, sPs[0], xlog=xlog)
    ax.set_ylabel(r"SFRD [ M$_{\rm sun}$  yr$^{-1}$  Mpc$^{-3}$]")
    ax.set_yscale("log")

    # observational points
    be = behrooziObsSFRD()

    l1, _, _ = ax.errorbar(
        be["redshift"],
        be["sfrd"],
        yerr=[be["errorDown"], be["errorUp"]],
        color="#999999",
        ecolor="#999999",
        alpha=0.9,
        capsize=0.0,
        fmt="o",
    )

    bo = bouwensSFRD2014()

    l2, _, _ = ax.errorbar(
        bo["redshift"],
        bo["sfrd"],
        xerr=bo["redshiftErr"],
        yerr=[bo["errorDown"], bo["errorUp"]],
        color="#333333",
        ecolor="#333333",
        alpha=0.9,
        capsize=0.0,
        fmt="D",
    )

    en = eniaSFRD2022()

    l3, _, _ = ax.errorbar(
        en["redshift"],
        en["sfrd"],
        xerr=[en["redshift_errLeft"], en["redshift_errRight"]],
        yerr=[en["sfrd_errDown"], en["sfrd_errUp"]],
        color="#666666",
        ecolor="#666666",
        alpha=0.9,
        capsize=0.0,
        fmt="s",
    )

    # todo: https://arxiv.org/abs/2311.08975

    legend1 = ax.legend([l1, l2, l3], [be["label"], bo["label"], en["label"]], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        # load sfr.txt file
        print("SFRD: " + sP.simName)
        s = sfrTxt(sP)

        ax.plot(s["redshift"], s["sfrd"], "-", lw=lw, label=sP.simName)

        # load subhalo-based derivation of SFRD (optional)
        if addSubhalosOnly:
            saveFilename = sP.derivPath + "sfrd_sub.hdf5"

            if isfile(saveFilename):
                # read existing
                print("Read: [%s]" % saveFilename)
                with h5py.File(saveFilename, "r") as f:
                    z = f["z"][()]
                    sfrd = f["sfrd"][()]
            else:
                # calculate new
                print("Calculating new: [%s]" % saveFilename)
                sP.setRedshift(0.0)

                z = np.zeros(sP.snap, dtype="float32")
                sfrd = np.zeros(sP.snap, dtype="float32")
                sfrd.fill(np.nan)

                for snap in range(0, sP.snap):
                    print(snap)
                    sP.setSnap(snap)
                    gc = sP.groupCat(fieldsSubhalos=["SubhaloSFRinRad"])
                    sfrd[snap] = gc.sum() / sP.boxSizeCubicComovingMpc  # msun/yr/mpc^3
                    z[snap] = sP.redshift

                # save
                with h5py.File(saveFilename, "w") as f:
                    f["z"] = z
                    f["sfrd"] = sfrd
                print("Wrote: [%s]" % saveFilename)

            ax.plot(z, sfrd, "-", lw=lw, label=sP.simName + " (sub)")

    # second legend
    ax.legend(loc="lower left")

    pdf.savefig()
    plt.close(fig)


def blackholeVsStellarMass(
    sPs,
    pdf,
    twiceR=False,
    vsHaloMass=False,
    vsBulgeMass=False,
    actualBHMasses=False,
    actualLargestBHMasses=True,
    simRedshift=0.0,
    sizefac=1.0,
    xlim=None,
    ylim=None,
):
    """Black hole mass vs. stellar (bulge) mass relation at z=0."""
    from ..load.data import kormendyHo2013, mcconnellMa2013

    assert twiceR or vsHaloMass or vsBulgeMass

    # plot setup
    fig = plt.figure(figsize=(figsize[0] * sizefac, figsize[1] * sizefac))
    ax = fig.add_subplot(111)

    xlim_def = [8.5, 13.0]
    ylim_def = [5.5, 11.0]

    ax.set_xlim(xlim_def if xlim is None else xlim)
    ax.set_ylim(ylim_def if ylim is None else ylim)

    ylabel = r"Black Hole Mass [ log M$_{\rm sun}$ ]"

    ax.set_ylabel(ylabel)

    ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ] [ < 1r$_{1/2}$ ]")
    if twiceR:
        ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ] [ < 2r$_{1/2}$ ]")
    if vsHaloMass:
        ax.set_xlabel(r"M$_{\rm halo}$ [ log M$_{\rm sun}$ ] [ M$_{\rm 200c}$ ]")
        if xlim is None:
            ax.set_xlim([9, 14.5])
        if ylim is None:
            ax.set_ylim([5.0, 11.0])
    if vsBulgeMass:
        ax.set_xlabel(r"M$_{\rm bulge,\star}$ [ log M$_{\rm sun}$ ] [ 2*counter-rotating < 1r$_{1/2}$ ]")

    # observational points
    if not vsHaloMass:
        k = kormendyHo2013()
        m = mcconnellMa2013()

        l3, _, _ = ax.errorbar(
            m["pts"]["M_bulge"],
            m["pts"]["M_BH"],
            yerr=[m["pts"]["M_BH_down"], m["pts"]["M_BH_up"]],
            color="#bbbbbb",
            ecolor="#dddddd",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
        )

        (l2,) = ax.plot(m["M_bulge"], m["M_BH"], "-", color="#999999")
        ax.fill_between(m["M_bulge"], m["errorDown"], m["errorUp"], color="#999999", interpolate=True, alpha=0.3)

        (l1,) = ax.plot(k["M_bulge"], k["M_BH"], "-", color="#333333")
        ax.fill_between(k["M_bulge"], k["errorDown"], k["errorUp"], color="#333333", interpolate=True, alpha=0.3)

        legend1 = ax.legend([l1, l2, l3], [k["label"], m["label"], m["pts"]["label"]], loc="lower right")
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        sP.setRedshift(simRedshift)

        numBHs = sP.snapshotHeader()["NumPart"][sP.ptNum("bhs")]
        if not sP.groupCatHasField("Subhalo", "SubhaloBHMass") or numBHs == 0:
            print("BHMass: %s [SKIP: sim has no BHs]" % sP.simName)
            continue

        print("BHMass: " + sP.simName)

        fieldsSubhalos = ["SubhaloBHMass", "SubhaloMassType", "SubhaloGrNr"]
        if vsBulgeMass:
            fieldsSubhalos.append("SubhaloMassInHalfRadType")
        if twiceR:
            fieldsSubhalos.append("SubhaloMassInRadType")

        gc = sP.groupCat(fieldsHalos=["GroupFirstSub", "Group_M_Crit200"], fieldsSubhalos=fieldsSubhalos)

        # centrals only
        w = sP.cenSatSubhaloIndices(cenSatSelect="cen")
        wHalo = gc["subhalos"]["SubhaloGrNr"][w]

        # stellar mass definition: would want to mimic bulge mass measurements
        if not twiceR and not vsHaloMass:
            xx_code = gc["subhalos"]["SubhaloMassInHalfRadType"][w, sP.ptNum("stars")]
        if twiceR:
            xx_code = gc["subhalos"]["SubhaloMassInRadType"][w, sP.ptNum("stars")]
        if vsHaloMass:
            xx_code = gc["halos"]["Group_M_Crit200"][wHalo]
        if vsBulgeMass:
            # load auxCat and compute bulge-mass
            acField = "Subhalo_StellarRotation_1rhalfstars"
            acIndex = 2

            ac = sP.auxCat(fields=[acField])
            ac = np.squeeze(ac[acField][w, acIndex])  # counter-rotating mass fraction relative to total
            ac[np.where(np.isnan(ac))] = 0.0  # set NaN to zero (consistent with groupcat)

            # multiply 2 x (massfrac) x (stellar mass)
            mass_1rhalf = gc["subhalos"]["SubhaloMassInHalfRadType"][w, sP.ptNum("stars")]
            xx_code = 2.0 * ac * np.squeeze(mass_1rhalf)

        xx = sP.units.codeMassToLogMsun(xx_code)

        # 'total' black hole mass in this subhalo, exclude those with no BHs
        # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here
        # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
        if actualBHMasses:
            # "actual" BH masses, excluding gas reservoir
            yy = gc["subhalos"]["SubhaloBHMass"][w]
        else:
            # dynamical (particle masses)
            yy = gc["subhalos"]["SubhaloMassType"][w, sP.ptNum("bhs")]
        if actualLargestBHMasses:
            # load auxCat (the most massive BH in each subhalo)
            yy = sP.subhalos("mass_smbh")[w]

        yy = sP.units.codeMassToLogMsun(yy)
        ww = np.where(yy > 0.0)

        minPerBin = 1 if sP.simName == "TNG-Cluster" else 10
        xm, ym, sm = running_median(xx[ww], yy[ww], binSize=binSize, skipZeros=True, minNumPerBin=minPerBin)
        ym2 = savgol_filter(ym, sKn, sKo) if ym.size > sKn else ym
        sm2 = savgol_filter(sm, sKn, sKo) if sm.size > sKn else sm

        if any(sP.simName == "TNG-Cluster" for sP in sPs):
            # TNG-Cluster (+TNG300), use individual markers
            ww = np.where((xx > ax.get_xlim()[0]) & (yy > 0))
            ax.plot(xx[ww], yy[ww], "o", label=sP.simName)
        else:
            # normal: show running medians with percentile bands
            (l,) = ax.plot(xm[:-1], ym2[:-1], "-", lw=lw, label=sP.simName)

            if (len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2:
                y_down = np.array(ym2[:-1]) - sm2[:-1]
                y_up = np.array(ym2[:-1]) + sm2[:-1]
                ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.2)

    # second legend
    ax.legend(loc="upper left")

    pdf.savefig()
    plt.close(fig)


def stellarMassFunction(
    sPs,
    pdf,
    highMassEnd=False,
    centralsOnly=False,
    use30kpc=False,
    use30H=False,
    useP10=False,
    haloMasses=False,
    s850fluxes=False,
    simRedshift=0.0,
    dataRedshift=0.0,
):
    """Stellar mass function (number density of galaxies) at redshift zero, or above."""
    from ..load.data import (
        baldry2008SMF,
        baldry2012SMF,
        bernardi2013SMF,
        caputi2015SMF,
        davidzon2017SMF,
        dsouza2015SMF,
        grazian2015SMF,
        song2015SMF,
    )

    # config
    mts = ["SubhaloMassInRadType", "SubhaloMassInHalfRadType", "SubhaloMassType"]

    # plot setup
    fig, ax = plt.subplots()

    # ax.set_ylim([5e-6,2e-1])
    ax.set_ylim([1e-5, 3e-1])
    ax.set_xlim([6.5, 12.5])
    if dataRedshift is not None and dataRedshift >= 3.0:
        ax.set_ylim([5e-7, 6e-2])

    if highMassEnd:
        # ax.set_ylim([1e-7,2e-2])
        # ax.set_xlim([10.0,12.5])
        ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < various ]")
    else:
        # ax.set_ylim([5e-4,2e-1])
        # ax.set_xlim([7,11.5])
        ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < 2r$_{\star,1/2}$ ]")
    ax.set_ylabel("Stellar Mass Function [ Mpc$^{-3}$ dex$^{-1}$ ]")
    ax.set_yscale("log")

    if use30kpc:
        ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < 30 pkpc ]")
    if use30H:
        ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < min(2r$_{\star,1/2}$,30 pkpc) ]")
    if useP10:
        ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < puchwein2010 r$_{\rm cut}$ ]")

    # alternative x-axes
    if haloMasses:
        ax.set_xlim([9.0, 14.5])
        ax.set_ylabel(r"Mass Functioon $\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ]")
        ax.set_xlabel(r"Halo Mass [ log M$_{\rm sun}$ ] [ 200crit ]")

    if s850fluxes:
        ax.set_xlim([-1.5, 1.0])
        ax.set_ylim([5e-8, 1e-2])
        ax.set_ylabel(r"S850$\mu$m Functioon $\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ]")
        ax.set_xlabel(r"Submilliter Flux S850$\mu$m [ log mJy ]")

    # observational points
    data = []
    lines = []

    if dataRedshift == 0.0:
        data.append(baldry2008SMF())
        data.append(baldry2012SMF())
        data.append(bernardi2013SMF()["SerExp"])
        data.append(dsouza2015SMF())
    if dataRedshift == 1.0:
        raise Exception("todo")
    if dataRedshift == 2.0:
        raise Exception("todo")
    if dataRedshift == 3.0:
        data.append(davidzon2017SMF(redshift=2.5))
        data.append(davidzon2017SMF(redshift=3.0))
        data.append(caputi2015SMF(redshift=3))
    if dataRedshift == 4.0:
        data.append(davidzon2017SMF(redshift=3.5))
        data.append(song2015SMF(redshift=4))
        data.append(caputi2015SMF(redshift=3))
        data.append(caputi2015SMF(redshift=4))
        data.append(grazian2015SMF(redshift=3.5))
    if dataRedshift == 5.0:
        data.append(davidzon2017SMF(redshift=4.5))
        data.append(song2015SMF(redshift=5))
        data.append(caputi2015SMF(redshift=4))
        data.append(grazian2015SMF(redshift=4.5))

    symbols = ["D", "o", "p", "s", "x"]
    colors_obs = ["#bbbbbb", "#888888", "#555555", "#222222", "#000000"]

    for i, d in enumerate(data):
        ll = d["lowerLimits"] if "lowerLimits" in d else False
        if "errorUp" in d:
            l, _, _ = ax.errorbar(
                d["stellarMass"],
                d["numDens"],
                yerr=[d["errorDown"], d["errorUp"]],
                color=colors_obs[i],
                ecolor=colors_obs[i],
                alpha=0.9,
                capsize=0.0,
                fmt=symbols[i],
                lolims=ll,
            )
        if "error" in d:
            l, _, _ = ax.errorbar(
                d["stellarMass"],
                d["numDens"],
                yerr=d["error"],
                color=colors_obs[i],
                ecolor=colors_obs[i],
                alpha=0.9,
                capsize=0.0,
                fmt=symbols[i],
                lolims=ll,
            )

        lines.append(l)

    legend1 = ax.legend(lines, [d["label"] for d in data], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        for redshift in iterable(simRedshift):
            # loop over redshifts if more than 1
            print("SMF: " + sP.simName + " (z=%.1f)" % redshift)
            sP.setRedshift(redshift)

            gc = sP.groupCat(fieldsHalos=["GroupFirstSub", "Group_M_Crit200"], fieldsSubhalos=mts)

            # centrals only
            if centralsOnly:
                wHalo = np.where(gc["halos"]["GroupFirstSub"] >= 0)
                w = gc["halos"]["GroupFirstSub"][wHalo]
            else:
                w = np.arange(gc["subhalos"]["count"], dtype="int32")

            # for each of the three stellar mass definitions, calculate SMF and plot
            count = 0

            for mt in mts:
                if not highMassEnd and mt != "SubhaloMassInRadType":
                    continue

                # temporary Mstar selection
                if use30kpc:
                    field = "Subhalo_Mass_30pkpc_Stars"
                    ac = sP.auxCat(fields=[field])
                    xx = sP.units.codeMassToLogMsun(ac[field][w])
                if use30H:
                    field = "Subhalo_Mass_min_30pkpc_2rhalf_Stars"
                    ac = sP.auxCat(fields=[field])
                    xx = sP.units.codeMassToLogMsun(ac[field][w])
                if useP10:
                    field = "Subhalo_Mass_puchwein10_Stars"
                    ac = sP.auxCat(fields=[field])
                    xx = sP.units.codeMassToLogMsun(ac[field][w])
                if not use30kpc and not useP10 and not use30H:
                    xx = gc["subhalos"][mt][w, sP.ptNum("stars")]
                    xx = sP.units.codeMassToLogMsun(xx)

                if haloMasses:
                    # halo mass instead: no w index, always all
                    xx = sP.units.codeMassToLogMsun(gc["halos"]["Group_M_Crit200"])
                    print("using halo masses instead")
                if s850fluxes:
                    # S850 fluxes instead
                    acKey = "Subhalo_S850um"  # _25pkpc'
                    xx = logZeroNaN(sP.auxCat(acKey)[acKey][w])
                    print("using s850 fluxes instead")

                normFac = sP.boxSizeCubicComovingMpc * binSize
                xm, ym = running_histogram(xx, binSize=binSize, normFac=normFac, skipZeros=True)
                ym = savgol_filter(ym, sKn, sKo)

                label = sP.simName + " z=%.1f" % sP.redshift if count == 0 else ""
                color = l.get_color() if count > 0 else None
                (l,) = ax.plot(xm[3:], ym[3:], linestyles[count], color=color, lw=lw, label=label)

                if s850fluxes:
                    # update previous label
                    l.set_label("%s [cen+sat]" % sP.simName)
                    # add centrals only
                    wHalo = np.where(gc["halos"]["GroupFirstSub"] >= 0)
                    w_cen = gc["halos"]["GroupFirstSub"][wHalo]

                    xm, ym = running_histogram(xx[w_cen], binSize=binSize, normFac=normFac, skipZeros=True)
                    ym = savgol_filter(ym, sKn, sKo)
                    (l,) = ax.plot(xm[3:], ym[3:], linestyles[count], lw=lw, label="%s [cen]" % sP.simName)

                count += 1

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    if highMassEnd:
        sExtra = [
            plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[2]),
            plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[0]),
            plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[1]),
        ]
        lExtra = [r"$M_\star^{\rm tot}$", r"$M_\star^{< 2r_{\star,1/2}}$", r"$M_\star^{< r_{\star,1/2}}$"]
    else:
        sExtra = []
        lExtra = []

    ax.legend(handles + sExtra, labels + lExtra, loc="lower left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def HIMassFunction(sPs, pdf, centralsOnly=True, simRedshift=0.0):
    """HI mass function (number density of HI masses) at redshift zero."""
    acFields = ["Subhalo_Mass_100pkpc_HI", "Subhalo_Mass_30pkpc_HI", "Subhalo_Mass_HI"]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_ylim([1e-6, 4e-1])
    ax.set_xlim([6.5, 11.0])

    ax.set_xlabel(r"Galaxy HI Mass [ log M$_{\rm sun}$ ] [ < various ]")
    ax.set_ylabel(r"HI Mass Function [ Mpc$^{-3}$ dex$^{-1}$ ]")
    ax.set_yscale("log")

    # observational points (Jones, M.G.+ 2018)
    data = [
        {"phi_star": 0.0045, "m_star": 9.94, "alpha": -1.25, "label": "ALFALFA $\\alpha$.100"},
        {"phi_star": 0.0049, "m_star": 9.94, "alpha": -1.29, "label": "ALFALFA $\\alpha$.100S"},
        {"phi_star": 0.0043, "m_star": 9.92, "alpha": -1.15, "label": "ALFALFA $\\alpha$.100F"},
        {"phi_star": 0.0062, "m_star": 9.76, "alpha": -1.22, "label": "ALFALFA $\\alpha$.100N"},
    ]
    lines = []

    for i, d in enumerate(data):
        xx = np.linspace(6.5, 11.0, 100)
        x_ratio = 10.0**xx / 10.0 ** d["m_star"]
        yy = np.log(10) * d["phi_star"] * x_ratio ** (d["alpha"] + 1.0) * np.exp(-x_ratio)
        (l,) = ax.plot(xx, yy, linestyle=linestyles[i], color="black", lw=lw, alpha=0.6 if i == 0 else 0.3)
        lines.append(l)

    legend1 = ax.legend(lines, [d["label"] for d in data], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print("HI MF: " + sP.simName + " (z=%.1f)" % simRedshift)
        sP.setRedshift(simRedshift)

        gc = sP.groupCat(fieldsSubhalos=["central_flag"])

        # centrals only?
        w = np.arange(gc.size)
        if centralsOnly:
            w = np.where(gc == 1)

        # for each of the three stellar mass definitions, calculate HIMF and plot
        for i, acField in enumerate(acFields):
            # load HI masses under this definition
            ac = sP.auxCat(fields=[acField])[acField]
            xx = sP.units.codeMassToLogMsun(ac[w])

            # calculate mass function
            normFac = sP.boxSizeCubicComovingMpc * binSize
            xm, ym_i = running_histogram(xx, binSize=binSize, normFac=normFac, skipZeros=True)
            ym = savgol_filter(ym_i, sKn, sKo)

            label = sP.simName + " z=%.1f" % sP.redshift if i == 0 else ""
            color = l.get_color() if i > 0 else None
            (l,) = ax.plot(xm[3:], ym[3:], linestyles[i], color=color, lw=lw, label=label)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []
    for i, acField in enumerate(acFields):
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[i]))
        lExtra.append(acField)

    ax.legend(handles + sExtra, labels + lExtra, loc="lower left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def HIMassFraction(sPs, pdf, centralsOnly=True, simRedshift=0.0):
    """HI mass fraction (M_HI/M*) vs M* at redshift zero."""
    from ..load.data import catinella2018

    acFields = ["Subhalo_Mass_100pkpc_HI", "Subhalo_Mass_30pkpc_HI", "Subhalo_Mass_HI"]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_ylim([-3.0, 1.0])
    ax.set_xlim([8.5, 12.0])

    ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ] [ < 30pkpc ]")
    ax.set_ylabel(r"M$_{\rm HI}$ / M$_\star$ [ log ]")

    # observational points (xCOLDGASS)
    c18 = catinella2018()

    symbols = ["D", "o"]
    color = "#222222"
    l1, _, _ = ax.errorbar(
        c18["mStar"],
        c18["HI_frac_mean"],
        yerr=c18["HI_frac_meanErr"],
        color=color,
        ecolor=color,
        alpha=0.9,
        capsize=0.0,
        fmt=symbols[0],
    )
    (l2,) = ax.plot(c18["mStar"], c18["HI_frac_median"], symbols[1], color=color)

    legend1 = ax.legend([l1, l2], [c18["label"] + " mean", c18["label"] + " median"], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        if sP.isZoom:
            continue

        print("HI FRAC: " + sP.simName + " (z=%.1f)" % simRedshift)
        sP.setRedshift(simRedshift)

        gc = sP.groupCat(fieldsSubhalos=["central_flag"])

        # centrals only?
        w = np.arange(gc.size)
        if centralsOnly:
            w = np.where(gc == 1)

        # load galaxy stellar masses, using the given definition
        massField = "Subhalo_Mass_30pkpc_Stars"
        ac = sP.auxCat(fields=[massField])[massField]
        xx = sP.units.codeMassToLogMsun(ac[w])

        # add obs. scatter
        if 1:
            stellarMassErrorDex = 0.1
            np.random.seed(424242)
            mass_errors_dex = np.random.normal(loc=0.0, scale=stellarMassErrorDex, size=xx.size)
            xx += mass_errors_dex

        # for each of the HI mass definitions, calculate HI mass fractions and plot
        for i, acField in enumerate(acFields):
            # load HI masses under this definition
            ac = sP.auxCat(fields=[acField])[acField]
            m_HI = sP.units.codeMassToLogMsun(ac[w])

            # calculate ratio, apply detection threshold treatment of upper limits of xGASS
            yy = 10.0**m_HI / 10.0**xx

            if 1:
                # note: neither of these actually modify the running_median() result
                # above M* = 10^9.7, detection limit equal to MHI/M*=0.02
                w_mstar_above = np.where(xx > 9.7)
                w_undetected = np.where(yy[w_mstar_above] < 0.02)
                yy[w_mstar_above][w_undetected] = 0.02
                # print('%d of %d above M* = 10^9.7 set to 2%% limit' % (len(w_undetected[0]),len(w_mstar_above[0])))

                # below M* = 10^9.7, detection limit equal to MHI=10^8 msun
                w_mstar_below = np.where(xx <= 9.7)
                w_undetected = np.where(m_HI[w_mstar_below] < 8.0)
                yy[w_mstar_below][w_undetected] = 10.0**8.0 / 10.0 ** xx[w_mstar_below][w_undetected]
                # print('%d of %d below M* = 10^9.7 set to 8.0 limit' % (len(w_undetected[0]),len(w_mstar_below[0])))

            yy = np.log10(yy)  # obs. points are average/median of log of gas fractions

            # calculate median
            xm, ym_i, sm_i, pm_i = running_median(
                xx, yy, binSize=binSize, skipZeros=True, percs=[10, 25, 75, 90], mean=True
            )

            if xm.size > sKn:
                ym = savgol_filter(ym_i, sKn, sKo)
                # sm = savgol_filter(sm_i, sKn, sKo)
                pm = savgol_filter(pm_i, sKn, sKo, axis=1)

            label = sP.simName + " z=%.1f" % sP.redshift if i == 0 else ""
            ax.plot(xm, ym, linestyles[i], color=colors[j], lw=lw, label=label)

            if i == 0:
                ax.fill_between(xm, pm[0, :], pm[-1, :], color=colors[j], interpolate=True, alpha=0.25)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []
    for i, acField in enumerate(acFields):
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[i]))
        lExtra.append(acField)

    ax.legend(handles + sExtra, labels + lExtra, loc="lower left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def HIvsHaloMass(sPs, pdf, simRedshift=0.0):
    """HI mass (M_HI) vs M_halo at redshift zero."""
    from ..load.data import obuljen2019
    # todo: https://arxiv.org/abs/2502.00110

    acFields = ["Subhalo_Mass_FoF_HI", "Subhalo_Mass_HI"]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_ylim([6.0, 11.5])
    ax.set_xlim([9.5, 14.0])

    ax.set_xlabel(r"Halo Mass [ log M$_{\rm sun}$ ]")
    ax.set_ylabel(r"M$_{\rm HI}$ [ log M$_{\rm sun}$ ]")

    # observational points (Obuljen+ 2018)
    o18 = obuljen2019()

    color = "#222222"
    (l1,) = ax.plot(o18["Mhalo"], o18["mHI"], color=color)
    ax.fill_between(
        o18["Mhalo"],
        savgol_filter(o18["mHI_low"], sKn, sKo),
        savgol_filter(o18["mHI_high"], sKn, sKo),
        color=color,
        alpha=0.2,
    )

    legend1 = ax.legend([l1], [o18["label"]], loc="lower right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        if sP.isZoom:
            continue

        print("HI/MHALO: " + sP.simName + " (z=%.1f)" % simRedshift)
        sP.setRedshift(simRedshift)

        # load halo masses, restrict to centrals only
        gc = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])
        w = np.where(np.isfinite(gc))
        xx = gc[w]

        # for each of the HI mass definitions, calculate HI masses and plot
        for i, acField in enumerate(acFields):
            # load HI masses under this definition
            ac = sP.auxCat(fields=[acField], expandPartial=True)[acField]
            yy = sP.units.codeMassToLogMsun(ac[w])

            # calculate median
            xm, ym_i, sm_i, pm_i = running_median(
                xx, yy, binSize=binSize, skipZeros=True, percs=[10, 25, 75, 90], mean=True
            )

            if xm.size > sKn:
                ym = savgol_filter(ym_i, sKn, sKo)
                # sm = savgol_filter(sm_i, sKn, sKo)
                pm = savgol_filter(pm_i, sKn, sKo, axis=1)

            label = sP.simName + " z=%.1f" % sP.redshift if i == 0 else ""
            ax.plot(xm, ym, linestyles[i], color=colors[j], lw=lw, label=label)

            if i == 0:
                ax.fill_between(xm, pm[0, :], pm[-1, :], color=colors[j], interpolate=True, alpha=0.25)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []
    for i, acField in enumerate(acFields):
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", lw=lw, linestyle=linestyles[i]))
        lExtra.append(acField)

    ax.legend(handles + sExtra, labels + lExtra, loc="upper left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def massMetallicityStars(sPs, pdf, simRedshift=0.0, sdssFiberFits=False):
    """Stellar mass-metallicity relation at z=0."""
    from ..load.data import gallazzi2005, kirby2013, loadSDSSFits, woo2008

    # config
    acMetalFields = [
        "Subhalo_StellarZ_SDSSFiber4pkpc_rBandLumWt",
        "Subhalo_StellarZ_SDSSFiber_rBandLumWt",
        "Subhalo_StellarZ_4pkpc_rBandLumWt",
    ]
    metalFields = ["SubhaloStarMetallicityHalfRad", "SubhaloStarMetallicity", "SubhaloStarMetallicityMaxRad"]

    minNumStars = 1  # log(Mstar) ~= 8.2 (1820) or 9.1 (2500)

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([8.0, 12.5])
    ax.set_ylim([-1.5, 1.0])

    xlabel = r"Stellar Mass [ log M$_{\rm sun}$ ]"
    ylabel = r"Z$_{\rm stars}$ [ log Z$_{\rm sun}$ ]"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # observational points
    g = gallazzi2005(sPs[0])
    w = woo2008(sPs[0])
    k = kirby2013()

    (l1,) = ax.plot(g["stellarMass"], g["Zstars"], "-", color="#333333", lw=2.0, alpha=0.7)
    ax.fill_between(g["stellarMass"], g["ZstarsDown"], g["ZstarsUp"], color="#333333", interpolate=True, alpha=0.3)

    l2, _, _ = ax.errorbar(
        w["stellarMass"],
        w["Zstars"],
        yerr=w["ZstarsErr"],
        color="#999999",
        ecolor="#999999",
        alpha=0.9,
        capsize=0.0,
        fmt="D",
    )

    l3, _, _ = ax.errorbar(
        k["stellarMass"],
        k["Zstars"],
        xerr=[k["stellarMassErr"], k["stellarMassErr"]],
        yerr=[k["ZstarsErr"], k["ZstarsErr"]],
        color="#666666",
        ecolor="#666666",
        alpha=0.9,
        capsize=0.0,
        fmt="o",
    )

    lines = [l1, l2, l3]
    labels = [g["label"], w["label"], k["label"]]

    if sdssFiberFits:
        # load MCMC fits to z<0.1 SDSS sample
        sdss = loadSDSSFits()

        (l4,) = ax.plot(sdss["logzsol"]["xm"], sdss["logzsol"]["ym"], "-", color="red", lw=2.0, alpha=0.7)
        ax.fill_between(
            sdss["logzsol"]["xm"],
            sdss["logzsol"]["pm"][1, :],
            sdss["logzsol"]["pm"][3, :],
            color="red",
            interpolate=True,
            alpha=0.1,
        )

        lines.append(l4)
        labels.append(sdss["label"])

    legend1 = ax.legend(lines, labels, loc="lower right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        print("MMStars: " + sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        # load
        gc = sP.groupCat(
            fieldsHalos=["GroupFirstSub", "Group_M_Crit200"],
            fieldsSubhalos=["SubhaloMass", "SubhaloMassInRadType"] + metalFields,
        )
        ac = sP.auxCat(fields=acMetalFields)

        # include: centrals + satellites (no noticeable difference vs. centrals only)
        # stellar mass definition, enforce resolution limit
        xx_code = gc["subhalos"]["SubhaloMassInRadType"][:, sP.ptNum("stars")]

        w = np.where(xx_code >= minNumStars * sP.targetGasMass)
        xx = sP.units.codeMassToLogMsun(xx_code[w])

        # metallicities based on auxCat calculation?
        for i, acMetalField in enumerate(acMetalFields):
            iters = [0, 1]  # add Guidi corrections

            for i_num in iters:
                yy = logZeroNaN(ac[acMetalField][w] / sP.units.Z_solar)

                if i_num == 1:
                    # apply and plot Guidi+ (2016) correction from Z(Lum-W_fibre) to Z(OBS)
                    yy = (yy - 0.23) / 0.69

                # only include subhalos with non-nan entries (e.g. at least 1 real star within radial cut)
                ww = np.where(np.isfinite(yy))
                yy_loc = yy[ww]
                xx_loc = xx[ww]

                xm, ym, sm, pm = running_median(xx_loc, yy_loc, binSize=binSize, skipZeros=True, percs=[10, 25, 75, 90])

                if xm.size >= sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)  # P[10,90]

                if i_num == 1:
                    # only show Guidi correction for [restricted] applicable mass range
                    ww = np.where((xm >= 10.0) & (xm <= 11.1))
                    xm = xm[ww]
                    ym = ym[ww]

                label = sP.simName if (i == 0 and i_num == 0) else ""
                ax.plot(xm[:-1], ym[:-1], linestyles[i + i_num * 2], color=colors[j], lw=lw, label=label)

                if i_num == 0:
                    ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color=colors[j], interpolate=True, alpha=0.25)

        # metallicities from groupcat, measured within what radius?
        for i, metalField in enumerate(metalFields):
            # note: Vogelsberger+ (2014a) scales the simulation values by Z_solar=0.02 instead of
            # correcting the observational Gallazzi/... points, resulting in the vertical shift
            # with respect to this plot (sim,Gal,Woo all shift up, but I think Kirby is good as is)
            yy = logZeroNaN(gc["subhalos"][metalField][w] / sP.units.Z_solar)

            xm, ym, sm, pm = running_median(xx, yy, binSize=binSize, skipZeros=True, percs=[10, 25, 75, 90])
            ym2 = savgol_filter(ym, sKn, sKo)
            # pm2 = savgol_filter(pm, sKn, sKo, axis=1)  # P[10,90]

            ax.plot(xm[1:-1], ym2[1:-1], linestyles[i + len(acMetalFields)], color=colors[j], lw=lw)

        # testing
        if sdssFiberFits and sP.simName == "TNG100-1":
            sP.setRedshift(0.1)
            yy = sP.subhalos("fiber_logzsol")
            yy = yy[w]

            # only include subhalos with non-nan age entries (e.g. at least 1 real star within radial cut)
            ww = np.where(np.isfinite(yy))
            yy_loc = yy[ww]
            xx_loc = xx[ww]

            xm, ym_i, sm_i, pm_i = running_median(xx_loc, yy_loc, binSize=binSize, percs=[10, 25, 75, 90])

            ym = savgol_filter(ym_i, sKn, sKo)
            sm = savgol_filter(sm_i, sKn, sKo)
            pm = savgol_filter(pm_i, sKn, sKo, axis=1)

            label = sP.simName + " fiber"
            ax.plot(xm[:-1], ym[:-1], linestyles[0], color="green", lw=lw, label=label)
            ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color="green", interpolate=True, alpha=0.25)

    # second legend
    handles, labels = ax.get_legend_handles_labels()

    sExtra = [
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[0]),
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[1]),
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[2]),
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[3]),
    ]
    lExtra = [
        r"Z$_{\rm stars}$ (r < 4pkpc, rBand-LumWt)",
        r"Z$_{\rm stars}$ (r < 1r$_{1/2})$",
        r"Z$_{\rm stars}$ (r < 2r$_{1/2})$",
        r"Z$_{\rm stars}$ (r < r$_{\rm max})$",
    ]

    ax.legend(handles + sExtra, labels + lExtra, loc="upper left")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def massMetallicityGas(sPs, pdf, simRedshift=0.0):
    """Gas mass-metallicity relation at z=0."""
    from ..load.data import guo2016, tremonti2004, zahid2012, zahid2014

    # config
    metalFields = ["SubhaloGasMetallicitySfrWeighted", "SubhaloGasMetallicitySfr"]

    # plot setup
    fig, ax = plt.subplots()

    if simRedshift == 0.0:
        ax.set_xlim([8.0, 11.5])
        ax.set_ylim([-1.2, 1.0])
    if simRedshift == 0.7:
        ax.set_xlim([7.5, 11.5])
        ax.set_ylim([-1.25, 0.75])

    ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ] [ < 2r$_{1/2}$ ]")
    ax.set_ylabel(r"Z$_{\rm gas}$ [ log Z$_{\rm sun}$ ] [ centrals and satellites ]")

    # observational points
    if simRedshift == 0.0:
        z12a = zahid2012(pp04=False, redshift=0)
        z12b = zahid2012(pp04=True, redshift=0)
        z14a = zahid2014(pp04=False, redshift=0.08)
        z14b = zahid2014(pp04=True, redshift=0.08)
        t04 = tremonti2004()

        l1, _, _ = ax.errorbar(
            z12a["stellarMass"],
            z12a["Zgas"],
            yerr=z12a["Zgas_err"],
            color="#666666",
            ecolor="#666666",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
        )

        l2, _, _ = ax.errorbar(
            z12b["stellarMass"],
            z12b["Zgas"],
            yerr=z12b["Zgas_err"],
            color="#666666",
            ecolor="#666666",
            alpha=0.9,
            capsize=0.0,
            fmt="s",
        )

        # l3,_,_ = ax.errorbar(t04['stellarMass'], t04['Zgas'], yerr=[t04['Zgas_errDown'],t04['Zgas_errUp']],
        #                     color='#bbbbbb', ecolor='#bbbbbb', alpha=0.9, capsize=0.0, fmt='o')
        (l3,) = ax.plot(t04["stellarMass"], t04["Zgas"], ":", color="#bbbbbb", alpha=0.9)
        ax.fill_between(
            t04["stellarMass"], t04["Zgas_Down"], t04["Zgas_Up"], color="#bbbbbb", interpolate=True, alpha=0.2
        )

        (l4,) = ax.plot(z14a["stellarMass"], z14a["Zgas"], "-", color="#999999", lw=2.0, alpha=0.9)
        (l5,) = ax.plot(z14a["stellarMass"], z14b["Zgas"], "--", color="#999999", lw=2.0, alpha=0.9)

        labels = [z12a["label"], z12b["label"], t04["label"], z14a["label"], z14b["label"]]
        legend1 = ax.legend([l1, l2, l3, l4, l5], labels, loc="lower right")
        ax.add_artist(legend1)

    if simRedshift == 0.7:
        g16a = guo2016(O3O2=False)
        g16b = guo2016(O3O2=True)
        z12a = zahid2012(pp04=False, redshift=1)
        z12b = zahid2012(pp04=True, redshift=1)
        z14a = zahid2014(pp04=False, redshift=0.78)
        z14b = zahid2014(pp04=True, redshift=0.78)

        (l1,) = ax.plot(g16a["stellarMass"], g16a["Zgas"], "-", color="#666666", lw=2.0, alpha=0.9)
        (l2,) = ax.plot(g16b["stellarMass"], g16b["Zgas"], "--", color="#666666", lw=2.0, alpha=0.9)

        opts = {"color": "#999999", "ecolor": "#999999", "alpha": 0.9, "capsize": 0.0}
        l3, _, _ = ax.errorbar(z12a["stellarMass"], z12a["Zgas"], yerr=z12a["Zgas_err"], fmt="D", **opts)
        l4, _, _ = ax.errorbar(z12b["stellarMass"], z12b["Zgas"], yerr=z12b["Zgas_err"], fmt="s", **opts)

        (l5,) = ax.plot(z14a["stellarMass"], z14a["Zgas"], "-", color="#bbbbbb", lw=2.0, alpha=0.9)
        (l6,) = ax.plot(z14a["stellarMass"], z14b["Zgas"], "--", color="#bbbbbb", lw=2.0, alpha=0.9)

        labels = [g16a["label"], g16b["label"], z12a["label"], z12b["label"], z14a["label"], z14b["label"]]
        legend1 = ax.legend([l1, l2, l3, l4, l5, l6], labels, loc="lower right")
        ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        if sP.isZoom:
            continue

        print("MMGas (z=%3.1f): %s" % (simRedshift, sP.simName))
        sP.setRedshift(simRedshift)

        gc = sP.groupCat(
            fieldsHalos=["GroupFirstSub", "Group_M_Crit200"], fieldsSubhalos=["SubhaloMassInRadType"] + metalFields
        )

        # include: centrals + satellites (no noticeable difference vs. centrals only)
        w = np.arange(gc["subhalos"]["count"], dtype="int32")
        # w = np.where(gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')] > 0.0)[0]

        # stellar mass definition
        xx_code = gc["subhalos"]["SubhaloMassInRadType"][w, sP.ptNum("stars")]
        xx = sP.units.codeMassToLogMsun(xx_code)

        # metallicity measured how/within what radius?
        for i, metalField in enumerate(metalFields):
            # only subhalos with nonzero metalField (some star-forming gas)
            wNz = np.where(gc["subhalos"][metalField][w] > 0.0)

            # log (Z_gas/Z_solar)
            yy = logZeroNaN(gc["subhalos"][metalField][w][wNz] / sP.units.Z_solar)

            xm, ym, sm = running_median(xx[wNz], yy, binSize=binSize)
            ym2 = savgol_filter(ym, sKn, sKo)
            sm2 = savgol_filter(sm, sKn, sKo)

            label = sP.simName + " z=%3.1f" % simRedshift if i == 0 else ""
            ax.plot(xm[:-1], ym2[:-1], linestyles[i], color=colors[j], lw=lw, label=label)

            if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0:
                ax.fill_between(xm[:-1], ym2[:-1] - sm2[:-1], ym2[:-1] + sm2[:-1], color=colors[j], alpha=0.3)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[0]),
        plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[1]),
    ]
    lExtra = [r"Z$_{\rm gas}$ (sfr>0 sfr-weighted)", r"Z$_{\rm gas}$ (sfr>0 mass-weighted)"]

    ax.legend(handles + sExtra, labels + lExtra, loc="upper left")

    pdf.savefig()
    plt.close(fig)


def baryonicFractionsR500Crit(sPs, pdf, simRedshift=0.0):
    """Gas, star, and total baryonic fractions within r500_crit (for massive systems)."""
    from ..load.data import giodini2009, lovisari2015

    # config
    markers = ["o", "D", "s"]  # gas, stars, baryons
    fracTypes = ["gas", "stars", "baryons"]

    acField = "Group_Mass_Crit500_Type"

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([11.0, 15.0])
    ax.set_ylim([0, 0.25])
    ax.set_xlabel(r"Halo Mass [ log M$_{\rm sun}$ ] [ < r$_{\rm 500c}$ ]")
    ax.set_ylabel(r"Gas/Star/Baryon Fraction [ M / M$_{\rm 500c}$ ]")

    # observational points
    g = giodini2009(sPs[0])
    l = lovisari2015(sPs[0])

    l1, _, _ = ax.errorbar(
        g["m500"],
        g["fGas500"],
        yerr=g["fGas500Err"],
        color="#999999",
        ecolor="#999999",
        alpha=0.9,
        capsize=0.0,
        fmt=markers[0] + linestyles[0],
    )
    l2, _, _ = ax.errorbar(
        g["m500"],
        g["fStars500"],
        yerr=g["fStars500Err"],
        color="#999999",
        ecolor="#999999",
        alpha=0.9,
        capsize=0.0,
        fmt=markers[1] + linestyles[1],
    )
    l3, _, _ = ax.errorbar(
        g["m500"],
        g["fBaryon500"],
        yerr=g["fBaryon500Err"],
        color="#999999",
        ecolor="#999999",
        alpha=0.9,
        capsize=0.0,
        fmt=markers[2] + linestyles[2],
    )
    l4, _, _ = ax.errorbar(
        l["m500"],
        l["fgas500"],
        yerr=l["fgas500_err"],
        color="#555555",
        ecolor="#555555",
        alpha=0.9,
        capsize=0.0,
        marker=markers[0],
        linestyle="",
    )

    labels = [
        g["label"] + r" f$_{\rm gas}$",
        g["label"] + r" f$_{\rm stars}$",
        g["label"] + r" f$_{\rm baryons}$",
        l["label"] + r" f$_{\rm gas}$",
    ]
    legend1 = ax.legend([l1, l2, l3, l4], labels, loc="upper left")
    ax.add_artist(legend1)

    # universal baryon fraction line
    OmegaU = sPs[0].omega_b / sPs[0].omega_m
    ax.plot([11.0, 15.0], [OmegaU, OmegaU], ":", lw=1.0, color="#444444", alpha=0.2)
    ax.text(12.5, OmegaU + 0.003, r"$\Omega_{\rm b} / \Omega_{\rm m}$", size="large", alpha=0.2)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        print("Fracs500Crit: " + sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            data = sP.auxCat(fields=[acField])[acField][sP.zoomSubhaloID, :]

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum(data)
            xx = sP.units.codeMassToLogMsun(xx_code)

            for i, fracType in enumerate(fracTypes):
                if fracType == "gas":
                    val = data[1]
                if fracType == "stars":
                    val = data[2]
                if fracType == "baryons":
                    val = data[1] + data[2]

                yy = val / xx_code  # fraction with respect to total
                ax.plot(xx, yy, markers[i], color=colors[0])
        else:
            data = sP.auxCat(fields=[acField])[acField]

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum(data, axis=1)

            # handle NaNs
            ww = np.isnan(xx_code)
            xx_code[ww] = 1e-10
            xx_code[xx_code == 0.0] = 1e-10
            data[ww, 0] = 1e-10
            data[ww, 1:2] = 0.0

            xx = sP.units.codeMassToLogMsun(xx_code)

            # loop over fraction types
            for i, fracType in enumerate(fracTypes):
                if fracType == "gas":
                    val = data[:, 1]
                if fracType == "stars":
                    val = data[:, 2]
                if fracType == "baryons":
                    val = data[:, 1] + data[:, 2]

                yy = val / xx_code  # fraction with respect to total

                xm, ym, sm = running_median(xx, yy, binSize=binSize)
                ym2 = savgol_filter(ym, sKn, sKo)

                label = sP.simName if i == 0 else ""
                ax.plot(xm[:], ym2[:], linestyles[i], color=colors[j], lw=lw, label=label)

                # if fracType == 'gas':
                #    ax.fill_between(xm[:-1], ym2[:-1]-sm[:-1], ym2[:-1]+sm[:-1],
                #                    color=colors[j], interpolate=True, alpha=0.3)

    # f_labels legend
    sExtra = [plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=ls) for ls in linestyles]
    lExtra = [r"f$_{\rm " + t + "}$" for t in fracTypes]

    legend3 = ax.legend(sExtra, lExtra, loc="lower right")
    ax.add_artist(legend3)

    # sim legend
    sExtra = [plt.Line2D([0], [0], color="black", lw=lw, alpha=0.0, marker="")]
    lExtra = ["[ sims z=%3.1f ]" % simRedshift]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="upper right")

    pdf.savefig()
    plt.close(fig)


def nHIcddf(sPs, pdf, moment=0, simRedshift=3.0, molecular=False):
    """CDDF (column density distribution function) of neutral (atomic) hydrogen, in the whole box."""
    from ..load.data import kim2013cddf, noterdaeme2009, noterdaeme2012, prochaska10cddf, zafar2013

    # config
    if molecular:
        # H2
        speciesList = ["nH2_BR_depth10", "nH2_GK_depth10", "nH2_KMT_depth10"]
        # speciesList = ['nH2_GK_depth10','nH2_GK_depth10_allSFRgt0','nH2_GK_depth10_onlySFRgt0']
        # speciesList = ['nH2_GK_depth10','nH2_GD14_depth10','nH2_GK11_depth10','nH2_K13_depth10','nH2_S14_depth10']
        # speciesList = ['nH2_GK_depth10_cell3','nH2_GK_depth10','nH2_GK_depth10_cell1']
        # speciesList = ['nH2_GK_depth5','nH2_GK_depth10','nH2_GK_depth20','nH2_GK','nH2_GK_depth1']
        sStr = "H_2"
        ylim0 = [-31, -17]
        ylim1 = [-5, 0]
        xlim = [17.8, 24.2]
    else:
        # HI, show with and without H2 corrections
        speciesList = ["nHI_noH2", "nHI"]  # ,'nHI2','nHI3']
        sStr = "HI"
        ylim0 = [-27, -18]
        ylim1 = [-4, 0]
        xlim = [17, 23]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_xlabel(r"log N$_{\rm %s}$ [ cm$^{-2}$ ]" % sStr)

    if moment == 0:
        ax.set_ylim(ylim0)
        ax.set_ylabel(r"CDDF (O$^{\rm th}$ moment):  log f(N$_{\rm %s}$)  [ cm$^{2}$ ]" % sStr)
    if moment == 1:
        ax.set_ylim(ylim1)
        ax.set_ylabel(r"CDDF (1$^{\rm st}$ moment):  log N$_{\rm %s}$ f(N$_{\rm %s}$)" % (sStr, sStr))

    # observational points
    if sStr == "HI":
        z13 = zafar2013()
        n12 = noterdaeme2012()
        n09 = noterdaeme2009()
        k13 = kim2013cddf()
        p10 = prochaska10cddf()

        if moment == 1:
            z13["log_fHI"] = np.log10(10.0 ** z13["log_fHI"] * 10.0 ** z13["log_NHI"])
            n12["log_fHI"] = np.log10(10.0 ** n12["log_fHI"] * 10.0 ** n12["log_NHI"])
            n09["log_fHI"] = np.log10(10.0 ** n09["log_fHI"] * 10.0 ** n09["log_NHI"])
            k13["log_fHI"] = np.log10(10.0 ** k13["log_fHI"] * 10.0 ** k13["log_NHI"])
            p10["log_fHI_lower"] = np.log10(10.0 ** p10["log_fHI_lower"] * 10.0 ** p10["log_NHI"])
            p10["log_fHI_upper"] = np.log10(10.0 ** p10["log_fHI_upper"] * 10.0 ** p10["log_NHI"])

        l1, _, _ = ax.errorbar(
            z13["log_NHI"],
            z13["log_fHI"],
            yerr=[z13["log_fHI_errDown"], z13["log_fHI_errUp"]],
            xerr=z13["log_NHI_xerr"],
            color="#999999",
            ecolor="#999999",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
        )
        l2, _, _ = ax.errorbar(
            n12["log_NHI"],
            n12["log_fHI"],
            yerr=n12["log_fHI_err"],
            xerr=n12["log_NHI_xerr"],
            color="#666666",
            ecolor="#666666",
            alpha=0.9,
            capsize=0.0,
            fmt="s",
        )
        l3, _, _ = ax.errorbar(
            n09["log_NHI"],
            n09["log_fHI"],
            yerr=n09["log_fHI_err"],
            xerr=n12["log_NHI_xerr"],
            color="#cccccc",
            ecolor="#cccccc",
            alpha=0.9,
            capsize=0.0,
            fmt="o",
        )
        l4, _, _ = ax.errorbar(
            k13["log_NHI"],
            k13["log_fHI"],
            yerr=[k13["log_fHI_errDown"], k13["log_fHI_errUp"]],
            color="#444444",
            ecolor="#444444",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
        )

        l5 = ax.fill_between(p10["log_NHI"], p10["log_fHI_lower"], p10["log_fHI_upper"], color="#dddddd", alpha=0.3)

        labels = [z13["label"], n12["label"], n09["label"], k13["label"], p10["label"]]
        legend1 = ax.legend([l1, l2, l3, l4, l5], labels, loc="lower left")
        ax.add_artist(legend1)

        # colDens definitions, plot vertical dotted lines [cm^-2] at dividing points
        limitDLA = 20.3
        ax.plot([limitDLA, limitDLA], ax.get_ylim(), "--", color="#dddddd", alpha=0.5)

    if sStr == "H_2":
        # Zwaan, Prochaska+ 2006
        xx = np.linspace(21.0, 24.0, 100)
        f_star = 1.1e-25
        sigma = 0.65
        mu = 20.6
        yy = f_star * np.exp(-(((xx - mu) / sigma) ** 2) / 2)
        yy = np.log10(yy)

        (l1,) = ax.plot(xx, yy, "-", color="#555555", lw=lw)

        # Peroux+ (in prep)
        xx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # log N_H2 [1/cm^2]
        yy = [-16.87, -20.11, -21.73, -23, -24, -25, -26, -27, -28, -29]  # f(N), all upper limits

        (l2,) = ax.plot(xx, yy, "v", color="#222222")

        # legend
        labels = ["Zwaan & Prochaska (2006) z=0", "Peroux+ (in prep)"]
        legend1 = ax.legend([l1, l2], labels, loc="lower left")
        ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        if sP.isZoom:
            continue

        print("CDDF %s: %s" % (sStr, sP.simName))
        if simRedshift is not None:
            sP.setRedshift(simRedshift)

        # once including H2 modeling, once without
        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            ac = sP.auxCat(fields=["Box_CDDF_" + species])

            n_species = ac["Box_CDDF_" + species][0, :]
            fN_species = ac["Box_CDDF_" + species][1, :]

            # plot
            xx = np.log10(n_species)

            if moment == 0:
                yy = logZeroNaN(fN_species)
            if moment == 1:
                yy = logZeroNaN(fN_species * n_species)

            label = "%s z=%.1f" % (sP.simName, sP.redshift) if i == 0 else ""
            ax.plot(xx, yy, lw=lw, linestyle=linestyles[i], color=colors[j], label=label)

    # legend
    # sExtra = [plt.Line2D([0],[0],color='black',lw=0.0,alpha=0.0,marker='')]
    # lExtra = ['[ sims z=%3.1f ]' % simRedshift]
    sExtra = [plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=ls) for ls in linestyles]
    lExtra = [str(s.replace("nH2_", "")) for s in speciesList]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="upper right")

    pdf.savefig()
    plt.close(fig)


def dlaMetallicityPDF(sPs, pdf, simRedshift=3.0):
    """Metallicity distribution of Damped Lyman-alpha Absorbers (DLAs; N_HI>20.3), using whole box grids."""
    from ..load.data import rafelski2012

    # config
    speciesList = ["nHI_noH2", "nHI"]
    log_nHI_limitDLA = 20.3
    log_Z_nBins = 50
    log_Z_range = [-3.0, 0.0]

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim(log_Z_range)
    ax.set_xlabel(r"log ( Z / Z$_{\rm solar}$ )")
    ax.set_ylim([0.0, 1.2])
    ax.set_ylabel("PDF of DLA Metallicities")

    # observational points
    sPs[0].setRedshift(simRedshift)
    r12 = rafelski2012(sPs[0])

    l1, _, _ = ax.errorbar(
        r12["log_Z"],
        r12["pdf"],
        yerr=r12["pdf_err"],
        xerr=r12["log_Z_err"],
        color="#666666",
        ecolor="#666666",
        alpha=0.9,
        capsize=0.0,
        fmt="D",
    )

    legend1 = ax.legend([l1], [r12["label"]], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        if sP.isZoom:
            continue
        else:
            print("DLA Z PDF: " + sP.simName)
            sP.setRedshift(simRedshift)

            # once including H2 modeling, once without
            for i, species in enumerate(speciesList):
                # load pre-computed Z PDF
                ac = sP.auxCat(fields=["Box_Grid_" + species, "Box_Grid_Z"])
                ww = np.where(ac["Box_Grid_" + species] > log_nHI_limitDLA)

                # ac = sP.auxCat(fields=['Box_CDDF_'+species])
                # n_HI  = ac['Box_CDDF_'+species][0,:]
                # fN_HI = ac['Box_CDDF_'+species][1,:]

                # plot (xx in log(Z/Zsolar) already)
                yy, xx = np.histogram(ac["Box_Grid_Z"][ww], bins=log_Z_nBins, range=log_Z_range, density=True)

                xx = xx[:-1] + 0.5 * (log_Z_range[1] - log_Z_range[0]) / log_Z_nBins

                label = sP.simName + " z=%3.1f" % sP.redshift if i == 0 else ""
                ax.plot(xx, yy, lw=lw, linestyle=linestyles[i], color=colors[j], label=label)

    # second legend
    sExtra = [plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=ls) for ls in linestyles]
    lExtra = [str(s) for s in speciesList]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="upper left")

    pdf.savefig()
    plt.close(fig)


def velocityFunction(sPs, pdf, centralsOnly=True, simRedshift=0.0):
    """Velocity function (galaxy counts as a function of v_circ/v_max)."""
    from ..load.data import bekeraite16VF

    binSizeLogKms = 0.03

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([50, 350])
    ax.set_ylim([1e-3, 3e-1])
    ax.set_xlabel(r"v$_{\rm circ}$ [ km/s ] [ sim = v$_{\rm max}$ ]")

    if centralsOnly:
        ax.set_ylabel(r"$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ only centrals ]")
    else:
        ax.set_ylabel(r"$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ centrals & satellites ]")
    ax.set_yscale("log")

    # observational points
    b16 = bekeraite16VF()

    l1, _, _ = ax.errorbar(
        b16["v_circ"],
        b16["numDens"],
        yerr=b16["numDens_err"],
        color="#666666",
        ecolor="#666666",
        alpha=0.9,
        capsize=0.0,
        fmt="s",
    )

    legend1 = ax.legend([l1], [b16["label"]], loc="upper right")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print("VF: " + sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        gc = sP.groupCat(fieldsHalos=["GroupFirstSub"], fieldsSubhalos=["SubhaloVmax"])

        # centrals only?
        if centralsOnly:
            wHalo = np.where(gc["halos"] >= 0)
            w = gc["halos"][wHalo]
        else:
            w = np.arange(gc["subhalos"].size)

        # histogram in log(v) and plot in linear(v)
        xx = np.log10(gc["subhalos"][w])
        normFac = sP.boxSizeCubicPhysicalMpc * binSizeLogKms

        xm_i, ym_i = running_histogram(xx, binSize=binSizeLogKms, normFac=normFac, skipZeros=True)
        xm = 10.0**xm_i
        ym = savgol_filter(ym_i, sKn, sKo)

        (l,) = ax.plot(xm[1:-1], ym[1:-1], linestyles[0], lw=lw, label=sP.simName)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="lower left")

    pdf.savefig()
    plt.close(fig)


def stellarAges(sPs, pdf, centralsOnly=False, simRedshift=0.0, sdssFiberFits=False):
    """Luminosity or mass weighted stellar ages, as a function of Mstar (Vog 14b Fig 25)."""
    from ..load.data import bernardi10, gallazzi2005, loadSDSSFits

    ageTypes = [
        "Subhalo_StellarAge_SDSSFiber4pkpc_rBandLumWt",
        "Subhalo_StellarAge_4pkpc_rBandLumWt",
        "Subhalo_StellarAge_NoRadCut_MassWt",
        "Subhalo_StellarAge_NoRadCut_rBandLumWt",
    ]

    minNumStars = 1  # log(Mstar) ~= 8.2 (1820) or 9.1 (2500)

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([8.0, 12.5])
    ax.set_ylim([0, 14])

    ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ]")
    ax.set_ylabel("Stellar Age [ Gyr ]")

    # observational points
    g05 = gallazzi2005(sPs[0])
    b10 = bernardi10()

    (l1,) = ax.plot(g05["stellarMass"], g05["ageStars"], "-", color="#333333", lw=2.0, alpha=0.7)
    ax.fill_between(
        g05["stellarMass"], g05["ageStarsDown"], g05["ageStarsUp"], color="#333333", interpolate=True, alpha=0.2
    )

    (l2,) = ax.plot(b10["stellarMass"], b10["ageStars"], "-", color="#777777", lw=2.0, alpha=0.7)
    ax.fill_between(
        b10["stellarMass"], b10["ageStarsDown"], b10["ageStarsUp"], color="#333333", interpolate=True, alpha=0.1
    )

    lines = [l1, l2]
    labels = [g05["label"], b10["label"]]

    if sdssFiberFits:
        # load MCMC fits to z<0.1 SDSS sample
        sdss = loadSDSSFits()

        (l3,) = ax.plot(sdss["tage"]["xm"], sdss["tage"]["ym"], "-", color="red", lw=2.0, alpha=0.7)
        ax.fill_between(
            sdss["tage"]["xm"],
            sdss["tage"]["pm"][1, :],
            sdss["tage"]["pm"][3, :],
            color="red",
            interpolate=True,
            alpha=0.1,
        )

        lines.append(l3)
        labels.append(sdss["label"])

    legend1 = ax.legend(lines, labels, loc="upper left")
    ax.add_artist(legend1)

    # loop over each fullbox run
    for j, sP in enumerate(sPs):
        print("AGES: " + sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        # load
        gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
        ac = sP.auxCat(fields=ageTypes)

        # include: centrals + satellites, or centrals only
        if centralsOnly:
            GroupFirstSub = sP.groupCat(fieldsHalos=["GroupFirstSub"])
            wHalo = np.where(GroupFirstSub >= 0)
            w = GroupFirstSub[wHalo]
        else:
            w = np.arange(gc.shape[0])

        # stellar mass definition, and enforce resolution limit
        xx_code = gc[:, sP.ptNum("stars")]

        wResLimit = np.where(xx_code >= minNumStars * sP.targetGasMass)[0]
        w = np.intersect1d(w, wResLimit)

        xx = sP.units.codeMassToLogMsun(xx_code[w])

        # loop through ages measured through different techniques
        for i, ageType in enumerate(ageTypes):
            iters = [0]
            # iters = [0,1] # show Guidi correction

            for i_num in iters:
                yy = ac[ageType][w]

                if i_num == 1:  # apply Guidi+ (2016) correction from Age(Lum-W_fibre) to Age(OBS)
                    yy = (yy - 2.37) / 0.97

                # only include subhalos with non-nan age entries (e.g. at least 1 real star within radial cut)
                ww = np.where(np.isfinite(yy))
                yy_loc = yy[ww]
                xx_loc = xx[ww]

                xm, ym, sm, pm = running_median(xx_loc, yy_loc, binSize=binSize, percs=[10, 25, 75, 90])

                if xm.size >= sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)

                if i_num == 1:
                    # only show Guidi correction for [restricted] applicable mass range
                    ww = np.where((xm >= 10.0) & (xm <= 11.0))
                    xm = xm[ww]
                    ym = ym[ww]

                label = sP.simName if (i == 0 and i_num == 0) else ""
                ax.plot(xm[:-1], ym[:-1], linestyles[i + 2 * i_num], color=colors[j], lw=lw, label=label)

                if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0 and i_num == 0:  # P[10,90]
                    ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color=colors[j], alpha=0.25)

        # testing
        if sdssFiberFits and sP.simName == "TNG100-1":
            sP.setRedshift(0.1)
            yy = sP.subhalos("fiber_tage")
            yy = yy[w]

            # only include subhalos with non-nan age entries (e.g. at least 1 real star within radial cut)
            ww = np.where(np.isfinite(yy))
            yy_loc = yy[ww]
            xx_loc = xx[ww]

            xm, ym_i, sm_i, pm_i = running_median(xx_loc, yy_loc, binSize=binSize, percs=[10, 25, 75, 90])

            ym = savgol_filter(ym_i, sKn, sKo)
            sm = savgol_filter(sm_i, sKn, sKo)
            pm = savgol_filter(pm_i, sKn, sKo, axis=1)

            label = sP.simName + " fiber"
            ax.plot(xm[:-1], ym[:-1], linestyles[0], color="green", lw=lw, label=label)
            ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color="green", alpha=0.25)

    # legend
    handles, labels = ax.get_legend_handles_labels()

    sExtra = [plt.Line2D([0], [0], color="black", lw=lw, fmt=linestyles[i]) for i, ageType in enumerate(ageTypes)]
    lExtra = [", ".join(ageType.split("_")[2:]) for ageType in ageTypes]
    # sExtra += [plt.Line2D([0], [0], color='black', lw=lw, marker='', linestyle=linestyles[2])]
    # lExtra += ['Guidi+ (2016) Correction']

    ax.legend(handles + sExtra, labels + lExtra, loc="lower right")

    # finish figure
    pdf.savefig()
    plt.close(fig)


def haloXrayLum(sPs, pdf=None, xlim=(10, 12), ylim=(38, 45), bolometric=False):
    """X-ray luminosity scaling relation vs stellar mass. Bolometric or 0.5-2 keV soft band."""
    from ..load.data import anderson2015

    # config
    xQuant = "mstar_30pkpc"
    yQuant = "xray_05_2kev_r500_halo" if not bolometric else "xray_r500"

    cenSatSelect = "cen"
    scatterPoints = any(sP.simName == "TNG-Cluster" for sP in sPs)  # TNG-Cluster (+TNG300)
    drawMedian = not scatterPoints
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    def _draw_data(ax):
        # observational points (only at z~0)
        for sP in sPs:
            if sP.redshift > 0.3:
                return

        a15 = anderson2015(sPs[0])

        key = "LumSoft" if "kev" in yQuant else "LumBol"  # either 0.5-2 kev or bolometric (via correction)
        l1, _, _ = ax.errorbar(
            a15["stellarMass"],
            a15[f"xray_{key}"],
            xerr=a15["stellarMass_err"],
            yerr=[a15[f"xray_{key}_errDown"], a15[f"xray_{key}_errUp"]],
            color="#000000",
            ecolor="#000000",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
            zorder=10 if scatterPoints else 1,
        )

        legend1 = ax.legend([l1], [a15["label"]], loc="lower right")
        ax.add_artist(legend1)

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
        sizefac=sizefac,
        f_post=_draw_data,
        legendLoc="upper left",
        pdf=pdf,
    )


def haloSynchrotronPower(sPs, pdf=None, xlim=(12.5, 15.5), ylim=(19, 26)):
    """Halo total synchrotron power at 1.4 Ghz (VLA configuration) vs M500."""
    from ..load.data import cassano13

    # config
    xQuant = "mhalo_500"
    yQuant = "p_sync_vla"

    cenSatSelect = "cen"
    scatterPoints = any(sP.simName == "TNG-Cluster" for sP in sPs)  # TNG-Cluster (+TNG300)
    drawMedian = not scatterPoints
    markersize = 40.0
    sizefac = 0.8  # for single column figure

    def _draw_data(ax):
        # observational points (only at z~0)
        c13 = cassano13()

        w_det = np.where(c13["p14_err"] > 0)
        l1, _, _ = ax.errorbar(
            c13["m500"][w_det],
            c13["p14"][w_det],
            xerr=c13["m500_err"][w_det],
            yerr=c13["p14_err"][w_det],
            color="#444444",
            ecolor="#444444",
            alpha=0.9,
            capsize=0.0,
            fmt="D",
        )
        w_upperlim = np.where(c13["p14_err"] < 0)
        ax.plot(c13["m500"][w_upperlim], c13["p14"][w_upperlim] - 0.15, "v", markersize=7, color="#888888", alpha=0.9)
        ax.plot(
            [c13["m500"][w_upperlim], c13["m500"][w_upperlim]],
            [c13["p14"][w_upperlim], c13["p14"][w_upperlim] - 0.15],
            "-",
            color="#888888",
            alpha=0.9,
        )

        legend1 = ax.legend([l1], [c13["label"]], loc="lower right")
        ax.add_artist(legend1)

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
        sizefac=sizefac,
        f_post=_draw_data,
        legendLoc="upper left",
        pdf=pdf,
    )


def plots():
    """Plot portfolio of global population comparisons between runs."""
    # run config
    sPs = []

    # sPs.append( simParams(run='tng100-1', redshift=0.0) )
    # sPs.append( simParams(run='tng100-1', redshift=1.0) )
    # sPs.append( simParams(run='tng100-1', redshift=2.0) )
    # sPs.append( simParams(run='tng100-1', redshift=4.0) )

    # for variant in ['0000','5018','5200']: #'5010','5014','5015','5017']:
    #    sPs.append( simParams(res=512, run='tng', variant=variant) )
    for variant in ["", "0000", "4503"]:
        sPs.append(simParams(res=625, run="tng", variant=variant))

    # sPs.append( simParams(res=2160, run='tng') )
    # sPs.append( simParams(res=1080, run='tng') )
    # sPs.append( simParams(res=540, run='tng') )
    # sPs.append( simParams(res=270, run='tng') )

    # sPs.append( simParams(res=1820, run='illustris', redshift=0.0) )
    # sPs.append( simParams(run='eagle', redshift=0.0) )
    # sPs.append( simParams(run='simba50', redshift=0.0) )
    # sPs.append( simParams(run='simba100', redshift=0.0) )
    # sPs.append( simParams(run='simba25', redshift=0.0) )
    # sPs.append( simParams(run='tng-cluster') )

    # change to plot simulations at z>0 against z=0 observational data
    zZero = 0.0

    if 0:
        # single plot and quit
        pdf = PdfPages("comptest_%s.pdf" % (datetime.now().strftime("%d-%m-%Y")))
        stellarMassHaloMass(sPs, pdf, ylog=False, use30kpc=True, simRedshift=zZero)
        pdf.close()
        return

    # make multipage PDF
    pdf = PdfPages("globalComps_%s.pdf" % (datetime.now().strftime("%d-%m-%Y")))

    stellarMassHaloMass(sPs, pdf, ylog=False, use30kpc=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=True, use30kpc=True, simRedshift=zZero)

    for redshift in [1, 2, 3, 4]:  # 6
        stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True, simRedshift=redshift)
        stellarMassHaloMass(sPs, pdf, ylog=True, use30kpc=True, simRedshift=redshift)

    sfrAvgVsRedshift(sPs, pdf)
    sfrdVsRedshift(sPs, pdf, xlog=True)
    sfrdVsRedshift(sPs, pdf, xlog=False)

    blackholeVsStellarMass(sPs, pdf, vsBulgeMass=True, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, twiceR=True, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, vsHaloMass=True, simRedshift=zZero)
    galaxySizes(sPs, vsHaloMass=False, simRedshift=zZero, addHalfLightRad=None, pdf=pdf)
    galaxySizes(sPs, vsHaloMass=True, simRedshift=zZero, addHalfLightRad=None, pdf=pdf)
    stellarMassFunction(
        sPs, pdf, highMassEnd=False, use30kpc=False, simRedshift=zZero, dataRedshift=None, haloMasses=True
    )
    stellarMassFunction(sPs, pdf, highMassEnd=False, use30kpc=True, simRedshift=zZero)
    stellarMassFunction(sPs, pdf, highMassEnd=True, simRedshift=zZero)

    for redshift in [1, 2, 3, 4]:
        stellarMassFunction(sPs, pdf, use30kpc=True, highMassEnd=False, simRedshift=redshift)

    massMetallicityStars(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=0.7)
    baryonicFractionsR500Crit(sPs, pdf, simRedshift=zZero)

    if 0:
        nHIcddf(sPs, pdf)  # z=3
        nHIcddf(sPs, pdf, moment=1)
        # todo: generalize nOVIcddf(), move out of projects/
        # nOVIcddf(sPs, pdf)  # z=0.2
        # nOVIcddf(sPs, pdf, moment=1)
        dlaMetallicityPDF(sPs, pdf)  # z=3

    # todo: generalize galaxyColorPDF(), move out of projects/
    # cheapDustModel = "p07c_cf00dust_rad30pkpc"  #'p07c_cf00dust_res_conv_ns1_rad30pkpc' is very expensive to run
    # galaxyColorPDF(sPs, pdf, bands=["u", "i"], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
    # galaxyColorPDF(sPs, pdf, bands=["g", "r"], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
    # galaxyColorPDF(sPs, pdf, bands=["r", "i"], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
    # galaxyColorPDF(sPs, pdf, bands=["i", "z"], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
    # galaxyColor2DPDFs(sPs, pdf, simRedshift=zZero, simColorsModel=cheapDustModel)

    velocityFunction(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=True, simRedshift=zZero)
    haloXrayLum(sPs, pdf)
    haloSynchrotronPower(sPs, pdf)
    HIMassFunction(sPs, pdf, simRedshift=zZero)
    HIMassFraction(sPs, pdf, simRedshift=zZero)
    HIvsHaloMass(sPs, pdf, simRedshift=zZero)
    galaxyHISizeMass(sPs, pdf, simRedshift=zZero)

    for sP in sPs:
        snapshot.phaseSpace2d(
            sP, xQuant="numdens", yQuant="temp", pdf=pdf
        )  # xlim=xlim, ylim=ylim, clim=clim, hideBelow=False, haloID=None,

    # todo: Vmax vs Mstar (tully-fisher) (Torrey Fig 9) (Vog 14b Fig 23) (Schaye Fig 12)
    # todo: Mbaryon vs Mstar (baryonic tully-fisher) (Vog 14b Fig 23)
    # todo: SFR main sequence (Schaye Fig 11)
    # todo: active/passive fraction vs Mstar (Schaye Fig 11) (or red/blue Vog Fig ?)
    # todo: SFRD decomposed into contribution by halo mass bin (Genel Fig ?)
    # todo: other metal CDDFs (e.g. Schaye Fig 17) (Bird 2016 Fig 6 Carbon) (HI z=0.1 Gurvich2016)
    # todo: Omega_X(z) (e.g. Bird? Fig ?)
    # todo: B/T distributions in Mstar bins, early/late fraction vs Mstar (kinematic)

    pdf.close()
