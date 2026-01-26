"""
Outflows paper (TNG50 presentation): analysis.

https://arxiv.org/abs/1902.05554
"""

import hashlib
import multiprocessing as mp
from functools import partial
from os import mkdir
from os.path import expanduser, isdir, isfile

import h5py
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

from ..cosmo.mergertree import loadMPBs, mpbPositionComplete
from ..plot.quantities import simSubhaloQuantity
from ..util import simParams
from ..util.helper import iterable, logZeroNaN
from ..util.match import match


def fit_vout():
    """For text discussion and fit equations relating to outflow velocities."""
    import pickle

    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq

    from ..plot.config import figsize, lw

    # load
    filename = "vout_75"
    with open("%s_TNG50-1.pickle" % filename, "rb") as f:
        data_z = pickle.load(f)

    # gather data of vout(M*,z)
    nskip = 2  # exclude last N datapoints from each vout(M*) line, due to poor statistics
    tot_len = np.sum([dset["xm"].size - nskip for dset in data_z])

    mstar = np.zeros(tot_len, dtype="float32")
    vout = np.zeros(tot_len, dtype="float32")
    redshift = np.zeros(tot_len, dtype="float32")

    offset = 0

    for dset in data_z:
        count = dset["xm"].size - nskip
        mstar[offset : offset + count] = dset["xm"][:-nskip]
        vout[offset : offset + count] = dset["ym"][:-nskip]
        redshift[offset : offset + count] = dset["redshift"]  # constant
        offset += count

    if 0:
        # least-squares fit
        def _error_function(params, mstar, z, vout):
            """Define error function to minimize."""
            (a, b, c, d, e, f) = params  # v = a + (M*/b)^c + (1+z)^d
            # vout_fit = a * (1+z)**e + (mstar/b)**(c) * (1+z)**d
            # vout_fit = a * (1+z)**e + (mstar/9)**(c) * (1+z)**d
            vout_fit = a * (1 + f * z**2) + b * (1 + z * c) * mstar + (mstar / d) ** e
            # vout_fit = a*np.sqrt(1+z) + b*(1+z) * (mstar/10) + c*(1+z) * (mstar/10)**2
            # vout_fit = np.log10(vout_fit)
            return vout_fit - vout

        params_init = [100.0, 8.0, 1.0, 1.0, 1.0, 1.0]
        # args = (mstar,redshift,np.log10(vout))
        args = (mstar, redshift, vout)

        params_best, params_cov, info, errmsg, retcode = leastsq(
            _error_function, params_init, args=args, full_output=True
        )

        print("params best:", params_best)

        # (A) vs. redshift plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlim([0.8 + 1, 6.2 + 1])
        ax.set_ylim([0, 1200])
        # ax.set_ylim([1,3]) # log
        ax.set_xlabel("(1+z)")
        ax.set_ylabel(filename + " [km/s]")

        for mass_ind in [3, 6, 10, 14, 15, 16, 17]:
            # make (x,y) datapoints
            xx = [1 + dset["redshift"] for dset in data_z]
            yy = []
            for dset in data_z:
                if mass_ind < dset["ym"].size:
                    yy.append(dset["ym"][mass_ind])
                else:
                    yy.append(np.nan)

            w = np.where(np.isfinite(yy))
            xx = np.array(xx)[w]
            yy = np.array(yy)[w]

            # plot
            x_mstar = data_z[0]["xm"][mass_ind]  # log msun
            label = r"M$_\star$ = %.2f" % x_mstar
            (l,) = ax.plot(xx, yy, "-", lw=lw, label=label)  # np.log10(yy)

            # plot fit
            x_redshift = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            vout_fit = _error_function(params_best, x_mstar, x_redshift, 0.0)
            ax.plot(1 + x_redshift, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig("%s_vs_redshift.pdf" % filename)
        plt.close(fig)

        # (B) vs M* plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlim([7.0, 12.0])
        ax.set_ylim([0, 1200])
        ax.set_xlabel("Stellar Mass [ log Msun ]")
        ax.set_ylabel(filename + " [km/s]")

        for dset in data_z:
            # plot
            x_mstar = dset["xm"]  # log msun
            y_vout = dset["ym"]

            (l,) = ax.plot(x_mstar, y_vout, "-", lw=lw, label="z = %.1f" % dset["redshift"])

            # plot fit
            x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            vout_fit = _error_function(params_best, x_mstar, dset["redshift"], 0.0)
            ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig("%s_vs_mstar.pdf" % filename)
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    param_z_degree = 1

    def _error_function_loc(params, mstar, vout):
        """Define error function to minimize. Form of vout(M*) at one redshift."""
        (a, b, c) = params
        vout_fit = a + b * (mstar / 10) ** c
        return vout_fit - vout

    def _error_function_loc2(params, mstar, z, vout):
        """Define error function to minimize. Same but with polynomial dependence of (1+z) on every parameter."""
        (a_coeffs, b_coeffs, c_coeffs) = params.reshape(3, param_z_degree + 1)
        a = np.poly1d(a_coeffs)(1 + z)
        b = np.poly1d(b_coeffs)(1 + z)
        c = np.poly1d(c_coeffs)(1 + z)
        vout_fit = a + b * (mstar / 10) ** c
        return vout_fit - vout

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0, 1200])
    ax.set_xlabel("Stellar Mass [ log Msun ]")
    ax.set_ylabel(filename + " [km/s]")

    params_z = []

    # let's fit just each redshift alone
    redshift_targets = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    scalefac = 1 / (1 + redshift_targets)
    H_z_H0 = np.zeros(scalefac.size, dtype="float64")  # if float32, fails mysteriously in fitting
    for i, z in enumerate(redshift_targets):
        sP = simParams(res=1820, run="tng", redshift=z)
        H_z_H0[i] = sP.units.H_z
    H_z_H0 /= simParams(res=1820, run="tng", redshift=0.0).units.H_z

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc = vout[w]

        params_init = [100.0, 10.0, 1.0]
        args = (mstar_loc, vout_loc)

        params_best, _, _, _, _ = leastsq(_error_function_loc, params_init, args=args, full_output=True)
        params_z.append(params_best)

        print("best fit at [z=%.1f]" % redshift_target, params_best)

        # plot data and fit
        (l,) = ax.plot(mstar_loc, vout_loc, "-", lw=lw, label="z = %.1f" % redshift_target)

        x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
        vout_fit = _error_function_loc(params_best, x_mstar, 0.0)
        ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig("%s_vs_mstar_z_indiv.pdf" % filename)
    plt.close(fig)

    if 1:
        # what is the scaling of vout with (1+z), a, and H(z)?
        def _error_function_plaw(params, xx, vout):
            """Define error function to minimize."""
            (a, b) = params
            vout_fit = a * (xx) ** b
            return vout_fit - vout

        for x_mstar in [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]:
            vout_fit = np.zeros(len(redshift_targets))

            for i in range(len(redshift_targets)):
                vout_fit[i] = _error_function_loc(params_z[i], x_mstar, 0.0)

            for iterNum in [0, 2]:
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111)

                ax.set_ylim([100, 400])
                ax.set_ylabel("fit vout")

                if iterNum == 0:
                    ax.set_xlim([1.5, 7.5])
                    ax.set_xlabel("(1 + Redshift)")
                    xx = 1 + np.array(redshift_targets)
                if iterNum == 1:
                    ax.set_xlim([0.0, 0.6])
                    ax.set_xlabel("Scale factor")
                    xx = scalefac
                if iterNum == 2:
                    ax.set_xlim([0.0, 14.0])
                    ax.set_xlabel("H(z) / H0")
                    xx = H_z_H0

                yy = vout_fit
                ax.plot(xx, yy, "-", marker="o", lw=lw)

                # fit line
                line_fit = np.polyfit(xx, yy, deg=1)
                ax.plot(xx, np.poly1d(line_fit)(xx), ":", lw=lw, marker="s")

                # fit powerlaw
                params_init = [100.0, 1.0]
                args = (xx, yy)

                params_best, _, _, _, _ = leastsq(_error_function_plaw, params_init, args=args, full_output=True)
                print(x_mstar, iterNum, " exponent = %.2f" % params_best[1])
                ax.plot(xx, _error_function_plaw(params_best, xx, 0.0), ":", lw=lw, marker="d")

                # finish
                fig.savefig("%s_vs_time_%d.pdf" % (filename, iterNum))
                plt.close(fig)

    # plot each parameter vs. redshift
    line_fits = []

    for i in range(len(params_z[0])):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_xlim([1.5, 7.5])
        # ax.set_ylim([0,1200])
        ax.set_xlabel("(1 + Redshift)")
        ax.set_ylabel("parameter [%d]" % i)

        xx = 1 + np.array(redshift_targets)
        yy = [pset[i] for pset in params_z]

        ax.plot(xx, yy, "-", marker="o", lw=lw)

        line_fit = np.polyfit(xx, yy, deg=param_z_degree)
        print(" param [%d] line: " % i, line_fit)
        ax.plot(xx, np.poly1d(line_fit)(xx), ":", lw=lw, marker="s")
        line_fits.append(line_fit)

        fig.savefig("%s_param_%d_vs_z.pdf" % (filename, i))
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    # re-fit with the given (1+z) dependence allowed for every parameter
    params_init = line_fits
    args = (mstar, vout, redshift)

    if 0:
        print("LOADING ACTUAL GALAXIES TEST")
        mstar_z = []
        vout_z = []
        z_z = []

        for redshift in [1.0, 2.0, 4.0, 6.0]:
            sP = simParams(res=2160, run="tng", redshift=redshift)
            # load mstar and count
            xvals = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])

            acField = "Subhalo_OutflowVelocity_SubfindWithFuzz"
            ac = sP.auxCat(acField)

            mstar = xvals[ac["subhaloIDs"]]
            radInd = 1  # 10kpc
            percInd = 3  # 90, percs = [25,50,75,90,95,99]
            vout = ac[acField][:, radInd, percInd]

            w = np.where(mstar >= 9.0)
            mstar_z.append(mstar[w])
            vout_z.append(vout[w])
            z = np.ones(len(w[0])) + redshift
            z_z.append(z)

        # condense
        mstar2 = np.hstack(mstar_z)
        vout2 = np.hstack(vout_z)
        redshift2 = np.hstack(z_z)

        args = (mstar2, vout2, redshift2)

    params_best, _, _, _, _ = leastsq(_error_function_loc2, params_init, args=args, full_output=True)

    (a_coeffs, b_coeffs, c_coeffs) = params_best.reshape(3, param_z_degree + 1)
    print(filename, "best z-evo fit: ", params_best)
    print(
        "%s = [%.1f + %.1f(1+z)] + [%.1f + %.1f(1+z)] * (log Mstar/10)^{%.1f + %.1f(1+z)}"
        % (filename, a_coeffs[0], a_coeffs[1], b_coeffs[0], b_coeffs[1], c_coeffs[0], c_coeffs[1])
    )

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0, 1200])
    ax.set_xlabel("Stellar Mass [ log Msun ]")
    ax.set_ylabel(filename + " [km/s]")

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc = vout[w]

        # plot data and fit
        (l,) = ax.plot(mstar_loc, vout_loc, "-", lw=lw, label="z = %.1f" % redshift_target)

        x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
        vout_fit = _error_function_loc2(params_best, x_mstar, redshift_target, 0.0)
        ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig("%s_vs_mstar_z_result.pdf" % filename)
    plt.close(fig)


def halo_selection(sP, minM200=11.5):
    """Make a quick halo selection above some mass limit.

    Sorted based on energy injection in the low BH state between this snapshot and the previous.
    """
    snap = sP.snap
    tage = sP.tage

    r = {}

    # quick caching
    saveFilename = expanduser("~") + "/temp_haloselect_%d_%.1f.hdf5" % (sP.snap, minM200)
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # halo selection: all centrals above 10^12 Mhalo
    m200 = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])
    with np.errstate(invalid="ignore"):
        w = np.where(m200 >= minM200)
    subInds = w[0]

    print("Halo selection [%d] objects (m200 >= %.1f)." % (len(subInds), minM200))

    # load mergertree for mapping the subhalos between adjacent snapshots
    mpbs = loadMPBs(sP, subInds, fields=["SnapNum", "SubfindID"])

    prevInds = np.zeros(subInds.size, dtype="int32") - 1

    for i in range(subInds.size):
        if subInds[i] not in mpbs:
            continue
        mpb = mpbs[subInds[i]]
        w = np.where(mpb["SnapNum"] == sP.snap - 1)
        if len(w[0]) == 0:
            continue  # skipped sP.snap-1 in the MPB
        prevInds[i] = mpb["SubfindID"][w]

    # restrict to valid matches
    w = np.where(prevInds >= 0)
    print("Using [%d] of [%d] snapshot adjacent matches through the MPBs." % (len(w[0]), prevInds.size))

    prevInds = prevInds[w]
    subInds = subInds[w]

    # compute a delta(BH_CumEgyInjection_RM) between this snapshot and the last
    bh_egyLow_cur, _, _, _ = simSubhaloQuantity(sP, "BH_CumEgy_low")

    sP.setSnap(snap - 1)
    bh_egyLow_prev, _, _, _ = simSubhaloQuantity(sP, "BH_CumEgy_low")  # erg

    dt_myr = (tage - sP.tage) * 1000
    sP.setSnap(snap)

    bh_dedt_low = (bh_egyLow_cur[subInds] - bh_egyLow_prev[prevInds]) / dt_myr  # erg/myr

    w = np.where(bh_dedt_low < 0.0)
    bh_dedt_low[w] = 0.0  # bad MPB track? CumEgy counter should be monotonically increasing

    # sort halo sample based on recent BH energy injection in low-state
    sort_inds = np.argsort(bh_dedt_low)[::-1]  # highest to lowest

    r["subInds"] = subInds[sort_inds]
    r["m200"] = m200[r["subInds"]]
    r["bh_dedt_low"] = bh_dedt_low

    # get fof halo IDs
    haloInds = sP.groupCat(fieldsSubhalos=["SubhaloGrNr"])
    r["haloInds"] = haloInds[r["subInds"]]

    # save cache
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]
    print("Saved [%s]." % saveFilename)

    return r


def selection_subbox_overlap(sP, sbNum, sel, verbose=False):
    """Determine intersection with a halo selection and evolving tracks through a given subbox."""
    path = sP.postPath + "SubboxSubhaloList/subbox%d_%d.hdf5" % (sbNum, sP.snap)

    with h5py.File(path, "r") as f:
        sbSubIDs = f["SubhaloIDs"][()]
        sbEverInFlag = f["EverInSubboxFlag"][()]

        numInside = sbEverInFlag[sel["subInds"]].sum()
        if verbose:
            print("number of selected halos ever inside [subbox %d]: %d" % (sbNum, numInside))

        if numInside == 0:
            return None

        # cross-match to locate target subhalos in these datasets
        sel_inds, subbox_inds = match(sel["subInds"], sbSubIDs)

        # load remaining datasets
        subboxScaleFac = f["SubboxScaleFac"][()]
        # minEdgeDistRedshifts = f["minEdgeDistRedshifts"][()]
        # sbMinEdgeDist = f["SubhaloMinEdgeDist"][subbox_inds, :]
        minSBsnap = f["SubhaloMinSBSnap"][()][subbox_inds]
        maxSBsnap = f["SubhaloMaxSBSnap"][()][subbox_inds]

        subhaloPos = f["SubhaloPos"][subbox_inds, :, :]

        # extended information available?
        extInfo = {}
        for key in f:
            if "SubhaloStars_" in key or "SubhaloGas_" in key or "SubhaloBH_" in key:
                extInfo[key] = f[key][subbox_inds, :, :]

        extInfo["mostBoundID"] = f["SubhaloMBID"][subbox_inds, :]

    return sel_inds, subbox_inds, minSBsnap, maxSBsnap, subhaloPos, subboxScaleFac, extInfo


def _getHaloEvoDataOneSnap(
    snap,
    sP,
    haloInds,
    minSnap,
    maxSnap,
    centerPos,
    scalarFields,
    loadFields,
    histoNames1D,
    histoNames2D,
    apertures,
    limits,
    histoNbins,
):
    """Multiprocessing target: load and process all data for one subbox/normal snap, returning results."""
    sP.setSnap(snap)
    if (sP.isSubbox and snap % 100 == 0) or (not sP.isSubbox):
        print("snap: ", snap)

    data = {"snap": snap}

    if 0:  # not sP.isSubbox:
        # temporary file already exists, then load now (i.e. skip)
        tempSaveName = sP.derivPath + "haloevo/evo_temp_sub_%d.dat" % snap
        if isfile(tempSaveName):
            import pickle

            print("Temporary file [%s] exists, loading..." % tempSaveName)
            f = open(tempSaveName, "rb")
            data = pickle.load(f)
            f.close()
            return data

    maxAperture_sq = np.max([np.max(limits["rad"]), np.max(apertures["histo2d"]), np.max(apertures["histo1d"])]) ** 2

    # particle data load
    for ptType in scalarFields.keys():
        data[ptType] = {}

        # first load global coordinates
        x = sP.snapshotSubsetP(ptType, "Coordinates", sq=False, float32=True)
        if x["count"] == 0:
            continue

        # create load mask
        mask = np.zeros(x["count"], dtype="bool")
        for i in range(len(haloInds)):
            if snap < minSnap[i] or snap > maxSnap[i]:
                continue

            # localize to this subhalo
            subPos = centerPos[i, snap, :]
            dists_sq = sP.periodicDistsSq(subPos, x["Coordinates"])

            w = np.where(dists_sq <= maxAperture_sq)
            mask[w] = True

        load_inds = np.where(mask)[0]
        mask = None

        if len(load_inds) == 0:
            continue

        x["Coordinates"] = x["Coordinates"][load_inds]

        # load remaining datasets, restricting each to those particles within relevant distances
        fieldsToLoad = list(set(scalarFields[ptType] + loadFields[ptType]))  # unique

        for field in fieldsToLoad:
            x[field] = sP.snapshotSubsetP(ptType, field, inds=load_inds)

        load_inds = None

        # subhalo loop
        for i, haloInd in enumerate(haloInds):
            data_loc = {}
            subPos = centerPos[i, snap, :]

            if snap < minSnap[i] or snap > maxSnap[i]:
                continue

            # localize to this subhalo
            dists_sq = sP.periodicDistsSq(subPos, x["Coordinates"])
            w_max = np.where(dists_sq <= maxAperture_sq)

            if len(w_max[0]) == 0:
                continue

            x_local = {}
            for key in x:
                if key == "count":
                    continue
                x_local[key] = x[key][w_max]

            x_local["dists_sq"] = dists_sq[w_max]

            # scalar fields: select relevant particles and save
            for key in scalarFields[ptType]:
                data_loc[key] = np.zeros(len(apertures["scalar"]), dtype="float32")

            for j, aperture in enumerate(apertures["scalar"]):
                w = np.where(x_local["dists_sq"] <= aperture**2)

                if len(w[0]) > 0:
                    for key in scalarFields[ptType]:
                        if ptType == "bhs":
                            data_loc[key][j] = x_local[key][w].max()  # MAX
                        if ptType in ["gas", "stars"]:
                            data_loc[key][j] = x_local[key][w].sum()  # TOTAL (sfr, masses)

            if len(histoNames1D[ptType]) + len(histoNames2D[ptType]) == 0:
                data[ptType][haloInd] = data_loc
                continue

            # common computations
            if ptType == "gas":
                # first compute an approximate subVel using gas
                w = np.where((x_local["dists_sq"] <= apertures["sfgas"] ** 2) & (x_local["StarFormationRate"] > 0.0))
                subVel = np.mean(x_local["vel"][w, :], axis=1)
                # todo: may need to smooth vel in time? alternatively, use MBID pos/vel evolution
                # or, we have the Potential saved in subboxes, could use particle with min(Potential) inside rad

            # calculate values only within maxAperture
            rad = np.sqrt(x_local["dists_sq"])  # i.e. 'rad', code units, [ckpc/h]
            vrad = sP.units.particleRadialVelInKmS(x_local["Coordinates"], x_local["vel"], subPos, subVel)

            vrel = sP.units.particleRelativeVelInKmS(x_local["vel"], subVel)
            vrel = np.sqrt(vrel[:, 0] ** 2 + vrel[:, 1] ** 2 + vrel[:, 2] ** 2)

            vals = {"rad": rad, "radlog": np.log10(rad), "vrad": vrad, "vrel": vrel}

            if ptType == "gas":
                vals["numdens"] = np.log10(x_local["numdens"])
                vals["temp"] = np.log10(x_local["temp"])

            # 2D histograms: compute and save
            for histoName in histoNames2D[ptType]:
                xaxis, yaxis, color = histoName.split("_")

                xlim = limits[xaxis]
                ylim = limits[yaxis]

                xvals = vals[xaxis]
                yvals = vals[yaxis]

                if color == "massfrac":
                    # mass distribution in this 2D plane
                    weight = x_local["mass"]
                    zz, _, _ = np.histogram2d(
                        xvals, yvals, bins=[histoNbins, histoNbins], range=[xlim, ylim], density=True, weights=weight
                    )
                else:
                    # each pixel colored according to its mean value of a third quantity
                    weight = vals[color]
                    zz, _, _, _ = binned_statistic_2d(
                        xvals, yvals, weight, "mean", bins=[histoNbins, histoNbins], range=[xlim, ylim]
                    )

                zz = zz.T
                if color != "vrad":
                    zz = logZeroNaN(zz)

                data_loc[histoName] = zz

            # 1D histograms (and X as a function of Y relationships): compute and save
            for histoName in histoNames1D[ptType]:
                xaxis, yaxis = histoName.split("_")
                xlim = limits[xaxis]
                xvals = vals[xaxis]

                data_loc[histoName] = np.zeros((len(apertures["histo1d"]), histoNbins), dtype="float32")

                # loop over apertures (always code units)
                for j, aperture in enumerate(apertures["histo1d"]):
                    w = np.where(x_local["dists_sq"] <= aperture**2)

                    if yaxis == "count":
                        # 1d histogram of a quantity
                        hh, _ = np.histogram(xvals[w], bins=histoNbins, range=xlim, density=True)
                    else:
                        # median yval (i.e. vrad) in bins of xval, which is typically e.g. radius
                        yvals = vals[yaxis]
                        hh, _, _ = binned_statistic(xvals[w], yvals[w], statistic="median", range=xlim, bins=histoNbins)

                    data_loc[histoName][j, :] = hh

            data[ptType][haloInd] = data_loc  # add dict for this subhalo to the byPartType dict, with haloInd as key

    # fullbox? save dump now so we can restart
    if 0:  # not sP.isSubbox:
        import pickle

        f = open(tempSaveName, "wb")
        pickle.dump(data, f)
        f.close()
        print("Wrote temp file %d." % snap)

    return data


def halosTimeEvo(sP, haloInds, haloIndsSnap, centerPos, minSnap, maxSnap, largeLimits=False):
    """Derive properties for one or more halos at all subbox/normal snapshots.

    Halos are defined by their evolving centerPos locations. minSnap/maxSnap define the range over which to consider
    each halo. sP can be a fullbox or subbox, which sets the data origin. One save file is made per halo.
    """
    # config
    scalarFields = {
        "gas": ["StarFormationRate", "mass"],
        "stars": ["mass"],
        "bhs": [
            "BH_CumEgyInjection_QM",
            "BH_CumEgyInjection_RM",
            "BH_Mass",
            "BH_Mdot",
            "BH_MdotEddington",
            "BH_MdotBondi",
            "BH_Progs",
        ],
    }
    histoNames1D = {
        "gas": [
            "rad_numdens",
            "rad_temp",
            "rad_vrad",
            "rad_vrel",
            "temp_vrad",
            "radlog_numdens",
            "radlog_temp",
            "radlog_vrad",
            "radlog_vrel",
            "vrad_count",
            "vrel_count",
            "temp_count",
        ],
        "stars": [],
        "bhs": [],
    }
    histoNames2D = {
        "gas": [
            "rad_vrad_massfrac",
            "rad_vrel_massfrac",
            "rad_vrad_temp",
            "numdens_temp_massfrac",
            "numdens_temp_vrad",
            "radlog_vrad_massfrac",
            "radlog_vrel_massfrac",
            "radlog_vrad_temp",
        ],
        "stars": ["rad_vrad_massfrac", "radlog_vrad_massfrac"],
        "bhs": [],
    }
    loadFields = {
        "gas": ["mass", "vel", "temp", "numdens"],
        "stars": ["mass", "vel"],
        "bhs": [],
    }  # everything needed to achieve histograms

    histoNbins = 300

    apertures = {
        "scalar": [10.0, 30.0, 100.0],  # code units, within which scalar quantities are accumulated
        "sfgas": 20.0,  # code units, select SFR>0 gas within this aperture to calculate subVel
        "histo1d": [10, 50, 100, 1000],  # code units, for 1D histograms/relations
        "histo2d": 1000.0,
    }  # code units, for 2D histograms where x is not rad/radlog (i.e. phase diagrams)

    limits = {
        "rad": [0.0, 800.0],
        "radlog": [0.0, 3.0],
        "vrad": [-400, 800],
        "vrel": [0, 800],
        "numdens": [-8.0, 2.0],
        "temp": [3.0, 8.0],
    }

    if largeLimits:
        # e.g. looking at M_halo > 12 with the action of the low-state BH winds, expand limits
        limits["rad"] = [0, 1200]
        limits["vrad"] = [-1000, 2500]
        limits["vrel"] = [0, 3500]

    # existence check, immediate load and return if so
    sbStr = "_" + sP.variant if "subbox" in sP.variant else ""
    hashStr = hashlib.sha256(
        "%s_%s_%s_%s_%d_%s_%s"
        % (
            str(scalarFields),
            str(histoNames1D),
            str(histoNames2D),
            str(loadFields),
            histoNbins,
            str(apertures),
            str(limits),
        )
    ).hexdigest()[::4]

    savePath = sP.derivPath + "/haloevo/"

    if not isdir(savePath):
        mkdir(savePath)

    savePath = savePath + "evo_%d_h%d%s_%s.hdf5"

    if len(haloInds) == 1:
        # single halo: try to load and return available data
        data = {}
        saveFilename = savePath % (haloIndsSnap, haloInds[0], sbStr, hashStr)

        if isfile(saveFilename):
            with h5py.File(saveFilename, "r") as f:
                for group in f.keys():
                    data[group] = {}
                    for dset in f[group].keys():
                        data[group][dset] = f[group][dset][()]
            return data

    # thread parallelize by snapshot
    nThreads = 1 if sP.isSubbox else 1  # assume ~full node memory usage when analyzing full boxes
    pool = mp.Pool(processes=nThreads)
    func = partial(
        _getHaloEvoDataOneSnap,
        sP=sP,
        haloInds=haloInds,
        minSnap=minSnap,
        maxSnap=maxSnap,
        centerPos=centerPos,
        scalarFields=scalarFields,
        loadFields=loadFields,
        histoNames1D=histoNames1D,
        histoNames2D=histoNames2D,
        apertures=apertures,
        limits=limits,
        histoNbins=histoNbins,
    )

    snaps = range(np.min(minSnap), np.max(maxSnap) + 1)  # [2687]

    if nThreads > 1:
        results = pool.map(func, snaps)
    else:
        results = []
        for snap in snaps:
            results.append(func(snap))

    # save each individually
    numSnaps = np.max(maxSnap) + 1  # centerPos.shape[1]

    for i, haloInd in enumerate(haloInds):
        data = {}

        # allocate a save data structure for this halo alone
        for ptType in scalarFields.keys():
            data[ptType] = {}
            for field in scalarFields[ptType]:
                data[ptType][field] = np.zeros((numSnaps, len(apertures["scalar"])), dtype="float32")
            for name in histoNames2D[ptType]:
                data[ptType]["histo2d_" + name] = np.zeros((numSnaps, histoNbins, histoNbins), dtype="float32")
            for name in histoNames1D[ptType]:
                data[ptType]["histo1d_" + name] = np.zeros(
                    (numSnaps, len(apertures["histo1d"]), histoNbins), dtype="float32"
                )

        data["global"] = {}
        data["global"]["mask"] = np.zeros(numSnaps, dtype="int16")  # 1 = in subbox
        data["global"]["mask"][minSnap[i] : maxSnap[i] + 1] = 1
        data["limits"] = limits
        data["apertures"] = apertures

        # stamp by snapshot
        for result in results:
            snap = result["snap"]

            for ptType in scalarFields.keys():
                # nothing for this halo/ptType combination (i.e. out of minSnap/maxSnap bounds)
                if haloInd not in result[ptType]:
                    continue

                for field in scalarFields[ptType]:
                    if field not in result[ptType][haloInd]:
                        continue
                    data[ptType][field][snap, :] = result[ptType][haloInd][field]

                for name in histoNames2D[ptType]:
                    if name not in result[ptType][haloInd]:
                        continue
                    data[ptType]["histo2d_" + name][snap, :, :] = result[ptType][haloInd][name]

                for name in histoNames1D[ptType]:
                    if name not in result[ptType][haloInd]:
                        continue
                    data[ptType]["histo1d_" + name][snap, :, :] = result[ptType][haloInd][name]

        # save
        saveFilename = savePath % (haloIndsSnap, haloInd, sbStr, hashStr)
        with h5py.File(saveFilename, "w") as f:
            for key in data:
                group = f.create_group(key)
                for dset in data[key]:
                    group[dset] = data[key][dset]
        print("Saved [%s]." % saveFilename)

    return data


def halosTimeEvoSubbox(sP, sbNum, selInds, minM200=11.5):
    """Record several properties for one or more halos at each subbox snapshot.

    Halos are specified by selInds, which
    index the result of selection_subbox_overlap() which intersects the SubboxSubhaloList
    catalog with the simple mass selection returned by halo_selection().
    """
    sel = halo_selection(sP, minM200=minM200)

    sel_inds, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = selection_subbox_overlap(sP, sbNum, sel)

    # indices, position evolution tracks, and min/max subbox snapshots for each
    selInds = iterable(selInds)

    haloInds = sel["haloInds"][sel_inds[selInds]]
    centerPos = subhaloPos[selInds, :, :]  # ndim == 3
    minSnap = minSBsnap[selInds]
    maxSnap = maxSBsnap[selInds]

    # compute and save, or return, time evolution data
    sP_sub = simParams(res=sP.res, run=sP.run, variant="subbox%d" % sbNum)

    largeLimits = False if minM200 == 11.5 else True

    return halosTimeEvo(sP_sub, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)


def halosTimeEvoFullbox(sP, haloInds):
    """Record several properties for one or more halos at each full box snapshot.

    Use SubLink MPB for positioning, extrapolating back to snapshot zero.
    """
    posSet = []
    minSnap = []
    maxSnap = []

    # acquire complete positional tracks at all snapshots
    for haloInd in haloInds:
        halo = sP.groupCatSingle(haloID=haloInd)
        snaps, _, pos = mpbPositionComplete(sP, halo["GroupFirstSub"])

        posSet.append(pos)
        minSnap.append(0)
        maxSnap.append(sP.snap)

        assert np.array_equal(snaps, range(0, sP.snap + 1))  # otherwise handle

    centerPos = np.zeros((len(posSet), posSet[0].shape[0], posSet[0].shape[1]), dtype=posSet[0].dtype)
    for i, pos in enumerate(posSet):
        centerPos[i, :, :] = pos

    # compute and save, or return, time evolution data
    largeLimits = True if sP.units.codeMassToLogMsun(halo["Group_M_Crit200"]) > 12.1 else False

    return halosTimeEvo(sP, haloInds, sP.snap, centerPos, minSnap, maxSnap, largeLimits=largeLimits)


def run():
    """Perform all the (possibly expensive) analysis for the paper."""
    redshift = 0.73  # last snapshot, 58

    TNG50 = simParams(res=2160, run="tng", redshift=redshift)
    TNG50_3 = simParams(res=540, run="tng", redshift=redshift)

    if 1:
        # print out subbox intersections with selection
        sel = halo_selection(TNG50, minM200=12.0)
        for sbNum in [0, 1, 2]:
            _ = selection_subbox_overlap(TNG50, sbNum, sel, verbose=True)

    if 1:
        # subbox: save data through time
        # halosTimeEvoSubbox(TNG50, sbNum=0, selInds=[0,1,2,3], minM200=11.5)
        halosTimeEvoSubbox(TNG50, sbNum=2, selInds=[0], minM200=12.0)

    if 0:
        # fullbox: save data through time, first 20 halos of 12.0 selection all at once
        sel = halo_selection(TNG50, minM200=12.0)
        halosTimeEvoFullbox(TNG50, haloInds=sel["haloInds"][0:20])

    if 0:
        # TNG50_3 test
        sel = halo_selection(TNG50_3, minM200=12.0)
        halosTimeEvoFullbox(TNG50_3, haloInds=sel["haloInds"][0:20])
