"""
Efficient analysis for time evolution of (sub)halos across snapshots.
"""

import hashlib
import multiprocessing as mp
from functools import partial
from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np
from scipy.stats import binned_statistic, binned_statistic_2d

from ..cosmo.mergertree import mpbPositionComplete
from ..util.helper import iterable, logZeroNaN
from ..util.match import match
from ..util.simParams import simParams


def subhalo_subbox_overlap(sP, sbNum, subInds, verbose=False):
    """Determine intersection with a subhalo selection and evolving tracks through a given subbox."""
    path = sP.postPath + "SubboxSubhaloList/subbox%d_%d.hdf5" % (sbNum, sP.snap)

    with h5py.File(path, "r") as f:
        sbSubIDs = f["SubhaloIDs"][()]
        sbEverInFlag = f["EverInSubboxFlag"][()]

        numInside = sbEverInFlag[subInds].sum()
        if verbose:
            print("number of selected halos ever inside [subbox %d]: %d" % (sbNum, numInside))

        if numInside == 0:
            return None

        # cross-match to locate target subhalos in these datasets
        sel_inds, subbox_inds = match(subInds, sbSubIDs)

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


def halosTimeEvo(sP, haloInds, haloIndsSnap, centerPos, minSnap, maxSnap):
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

    # appropriate if e.g. looking at M_halo > 12 with the action of the low-state BH winds, otherwise contract
    limits = {
        "rad": [0, 1200],  # [0.0, 800.0]
        "radlog": [0.0, 3.0],
        "vrad": [-1000, 2500],  # [-400, 800],
        "vrel": [0, 3500],  # [0, 800],
        "numdens": [-8.0, 2.0],
        "temp": [3.0, 8.0],
    }

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


def halosTimeEvoSubbox(sP, sbNum, sel, selInds):
    """Record several properties for one or more halos at each subbox snapshot.

    Halos are specified by selInds, which
    index the result of subbox_subbox_overlap() which intersects the SubboxSubhaloList
    catalog with the simple mass selection returned by halo_selection().
    """
    sel_inds, _, minSBsnap, maxSBsnap, subhaloPos, _, _ = subhalo_subbox_overlap(sP, sbNum, sel["subInds"])

    # indices, position evolution tracks, and min/max subbox snapshots for each
    selInds = iterable(selInds)

    haloInds = sel["haloInds"][sel_inds[selInds]]
    centerPos = subhaloPos[selInds, :, :]  # ndim == 3
    minSnap = minSBsnap[selInds]
    maxSnap = maxSBsnap[selInds]

    # compute and save, or return, time evolution data
    sP_sub = simParams(res=sP.res, run=sP.run, variant="subbox%d" % sbNum)

    return halosTimeEvo(sP_sub, haloInds, sP.snap, centerPos, minSnap, maxSnap)


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
    return halosTimeEvo(sP, haloInds, sP.snap, centerPos, minSnap, maxSnap)
