"""
Helper functions related to cosmo box simulations.
"""

from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np
from scipy import interpolate

from ..cosmo.mergertree import loadMPBs
from ..load.snapshot import subboxVals
from ..util.helper import closest, contiguousIntSubsets, evenlySample
from ..util.match import match
from ..util.simParams import simParams
from ..util.treeSearch import buildFullTree, calcQuantReduction


# --- snapshot configuration & spacing ---


def redshiftToSnapNum(redshifts=None, times=None, sP=None, recalculate=False, load=False):
    """Convert one or more input redshifts to closest matching snapshot numbers for a given sP."""
    assert sP is not None, "Must input sP."

    if redshifts is None:
        redshifts = np.array(sP.redshift)
    else:
        redshifts = np.array(redshifts)

    nSnaps = 1000  # maximum

    sbNum, sbStr1, sbStr2 = subboxVals(sP.subbox)
    if sP.subbox is not None:
        nSnaps *= 10
        if "tng" in sP.run:
            nSnaps = 16000  # maximum of 16,000 subbox snapshots for new TNG runs

    # load from file if it exists, otherwise create
    r = {}
    saveFilename = sP.derivPath + "snapnum." + sbStr1 + "redshift.hdf5"

    save = True
    try:
        if not isdir(sP.derivPath):
            mkdir(sP.derivPath)
    except PermissionError:
        print("Warning: Permission error creating directory [%s], skipping." % sP.derivPath)
        save = False

    if isfile(saveFilename) and not recalculate:
        with h5py.File(saveFilename, "r") as f:
            for key in f.keys():
                r[key] = f[key][()]
    else:
        r["redshifts"] = np.zeros(nSnaps, dtype="float32") - 1.0
        r["times"] = np.zeros(nSnaps, dtype="float32") - 1.0
        r["nFound"] = 0

        # attempt load for all snapshots
        for i in range(nSnaps):
            fileName = sP.snapPath(snapNum=i, subbox=sP.subbox, checkExists=True)

            if fileName is None:
                # allow for existence of groups only
                fileName = sP.gcPath(snapNum=i, checkExists=True)

                if fileName is None:
                    continue

            # snapshot exists, load redshift and scale factor and save to cache file
            with h5py.File(fileName, "r") as f:
                r["redshifts"][i] = f["Header"].attrs["Redshift"]
                r["times"][i] = f["Header"].attrs["Time"]

            # time is not equal to 1.0, but redshift is zero? non-cosmological run
            if r["redshifts"][i] == 0.0 and np.abs(r["times"][i] - 1.0) > 1e-10:
                r["redshifts"][i] = np.nan  # mark as such

            r["nFound"] += 1

        # save
        if save:
            with h5py.File(saveFilename, "w") as f:
                for key in r.keys():
                    f[key] = r[key]
        else:
            print("Warning: Permission error saving [%s], skipping." % saveFilename)

    if np.sum(redshifts) == -1:
        raise Exception("Old behavior, used to return !NULL.")

    # return everything
    if load:
        return r

    # return array of snapshot numbers
    snaps = np.zeros(redshifts.size, dtype="int32")

    # time instead of redshift? (non-cosmological run)
    if times is not None:
        for i, time in np.ndenumerate(times):
            # closest snapshot time to requested
            tFound, w = closest(r["times"], time)
            tErr = np.abs(tFound - time)

            if tErr > 0.1:
                print("Warning! [%s] Snapshot selected with time error = %g" % (sP.simName, tErr))

            snaps[i] = w
    else:
        # redshift search
        for i, redshift in np.ndenumerate(redshifts):
            # closest snapshot redshift to requested
            w_nan = np.where(r["redshifts"] == -1)[0]
            r["redshifts"][w_nan] = np.nan  # do not select non-existent snapshot

            zFound, w = closest(r["redshifts"], redshift)
            zErr = np.abs(zFound - redshift)

            if zErr > 0.1 and zFound > redshift:
                # try to recompute in case we have a partial save from a previously in progress run
                # and more snapshots exist than previously existed
                maxSnapPrev = np.where(np.isfinite(r["redshifts"]))[0].max()
                nextSnapExists = sP.snapPath(snapNum=maxSnapPrev + 1, subbox=sP.subbox, checkExists=True)

                if not recalculate and nextSnapExists is not None:
                    return redshiftToSnapNum(redshifts=redshifts, times=times, sP=sP, recalculate=True)
                else:
                    print("Warning! [%s] Snapshot selected with redshift error = %g" % (sP.simName, zErr))

            snaps[i] = w

    if snaps.size == 1:
        snaps = snaps[0]

    return snaps


def validSnapList(
    sP, maxNum=None, minRedshift=None, maxRedshift=None, onlyFull=False, reqTr=False, reqFluidQuants=False
):
    """Return a list of all snapshot numbers which exist."""
    if reqFluidQuants:
        assert reqTr

    if minRedshift is None:
        minRedshift = 0.0  # filter out -1 values indicating missing snaps
    if maxRedshift is None:
        maxRedshift = np.finfo("float32").max

    redshifts = snapNumToRedshift(sP, all=True)

    if maxNum is not None and sP.subbox is not None:
        # for subboxes (movie renderings), auto detect change of global timestep
        log_scalefacs = np.log10(1 / (1 + redshifts))
        dloga = log_scalefacs - np.roll(log_scalefacs, 1)
        dloga[0] = dloga[1]  # corrupted by roll

        dloga_target = np.median(dloga)  # np.median( dloga[ int(dloga.size*(2.0/4)):int(dloga.size*(3.0/4)) ])
        print("validSnapList(): subbox auto detect dloga_target = %f" % dloga_target)

        ww = np.where(dloga < 0.8 * dloga_target)[0]
        # print('  number snaps below target [%d] spanning [%d-%d]' % (len(ww),ww.min(),ww.max()))
        ww2 = np.where(dloga < 0.8 * 0.5 * dloga_target)[0]

        # assert len(ww2) == 0 # number of timesteps even one jump lower
        if len(ww2) > 0:
            print(" WARNING: %d snaps even one timestep below target" % len(ww2))

        # detect contiguous snapshot subsets in this list of integers
        ranges = contiguousIntSubsets(ww)
        # print(' identified contiguous snap ranges:',ranges)

        # find also ranges with the same dloga, but surrounded by larger regions of different dloga
        # (e.g. dips/spikes), currently unused
        ddloga = dloga - np.roll(dloga, 1)
        ddloga[0] = ddloga[1]  # corrupted by roll
        ww_const = np.where(np.abs(ddloga) < dloga_target / 10)[0]

        ranges2 = contiguousIntSubsets(ww_const)

        for i, loc_range in enumerate(ranges2):
            if i == 0:
                continue

            if loc_range[1] - loc_range[0] > 50:
                continue  # skip large contiguous ranges

            cur_dloga = np.mean(dloga[loc_range[0] : loc_range[1]])
            prev_dloga = np.mean(dloga[ranges2[i - 1][0] : ranges2[i - 1][1]])
            next_dloga = np.mean(dloga[ranges2[i + 1][0] : ranges2[i + 1][1]])

            if cur_dloga < prev_dloga and cur_dloga > next_dloga:
                # monotonic, ok
                continue

        # custom modifications by sim (independent of sbNum)
        ranges_global = []
        if sP.run == "tng" and sP.res == 2500:
            ranges_global.append((1092, 1099))
            ranges_global.append((1177, 1180))
            if ranges[-1] == (68, 274):
                ranges.pop()  # allow z<0.1 to switch to dloga_target/2

        # override every other snapshot in these ranges with a redshift of -1 so it is filtered out below
        for range_start, range_stop in ranges:
            # the first entry here corresponds to the first subbox snapshot whose delta time since the
            # previous is half of our target, so start removing here so that the dt across this gap
            # becomes constant
            snap_inds = ww[range_start:range_stop:2]
            redshifts[snap_inds] = -1.0
            # print(' in range [%d to %d] filter out %d snaps' % (range_start,range_stop,snap_inds.size))
        for range_start, range_stop in ranges_global:
            redshifts[range_start:range_stop:2] = -1.0

    w = np.where((redshifts >= minRedshift) & (redshifts <= maxRedshift))[0]

    if len(w) == 0:
        return None

    # require existence of trMC information? have to check now
    if reqTr:
        snaps = w
        w = []
        gName = "PartType" + str(sP.ptNum("tracer"))

        for snap in snaps:
            fileName = sP.snapPath(snap, subbox=sP.subbox, checkExists=True)
            if fileName is None:
                continue

            with h5py.File(fileName, "r") as f:
                if reqFluidQuants:
                    # TNG50/300/Cluster: although tracers exist in all snaps, FluidQuantities only in full
                    if gName in f and "FluidQuantities" in f[gName]:
                        w.append(snap)
                else:
                    # just require tracers in general (only exist in full snaps for TNG100)
                    if gName in f:
                        w.append(snap)

        w = np.array(w)

    # only full snapshots? have to check now
    if onlyFull:
        if sP.simName == "Eagle100":
            pass  # all snaps are full
        else:
            snaps = w
            w = []

            for snap in snaps:
                fileName = sP.snapPath(snap, subbox=sP.subbox, checkExists=True)

                if fileName is None:
                    continue

                with h5py.File(fileName, "r") as f:
                    # handle different criterion/simulation types
                    if "TNG" in sP.simName:
                        fullSnap = "/PartType0/MagneticField" in f
                    elif sP.winds == 4:  # MCST
                        fullSnap = "/PartType0/GrackleCoolTime" in f
                    else:
                        assert 0, "Unhandled case for determining full snaps."

                    # keep only full snaps
                    if fullSnap:
                        w.append(snap)

            w = np.array(w)

    # cap at a maximum number of snaps? (evenly spaced)
    if maxNum is not None:
        w = evenlySample(w, maxNum)

    return w


def multiRunMatchedSnapList(runList, method="expand", **kwargs):
    """Match snapshots across multiple simulations at a common set of redshifts.

    For an input runList of dictionaries containing a sP key corresponding to a simParams
    for each run, produce a 'matched'/unified set of snapshot numbers, one set per run, with
    all the same length, e.g. for comparative analysis at matched redshifts, or for rendering
    movie frames comparing runs at the same redshift. If method is 'expand', inflate the
    snapshot lists of all runs to the size of the maximal (duplicates are then guaranteed).
    If method is 'condense', shrink the snapshot lists of all runs to the size of the minimal
    (skips are then guaranteed).
    """
    assert method in ["expand", "condense"]

    snapLists = []
    numSnaps = []

    for run in runList:
        runSnaps = validSnapList(run["sP"], **kwargs)

        if runSnaps is None:
            raise Exception("Run [%s] has no snapshots within requested redshift range." % run["sP"].simName)

        numSnaps.append(len(runSnaps))

    # let method dictate target size of the matched snapshot lists and 'master' run
    if method == "expand":
        targetSize = np.max(numSnaps)
        targetRun = np.argmax(numSnaps)

    if method == "condense":
        targetSize = np.min(numSnaps)
        targetRun = np.argmin(numSnaps)

    print("Matched snapshot list [%s] to %d snaps of %s." % (method, targetSize, runList[targetRun]["sP"].simName))

    # choose the closest snapshot to each target redshift in each run
    targetSnaps = validSnapList(runList[targetRun]["sP"], **kwargs)
    targetRedshifts = snapNumToRedshift(runList[targetRun]["sP"], snap=targetSnaps)

    for run in runList:
        runSnaps = redshiftToSnapNum(targetRedshifts, sP=run["sP"])
        snapLists.append(runSnaps)

    # verify
    assert targetRedshifts.size == targetSize
    for snapList in snapLists:
        assert np.min(snapList) >= 0
        assert snapList.size == targetRedshifts.size

    return snapLists


def snapNumToRedshift(sP, snap=None, time=False, all=False):
    """Convert snapshot number(s) to redshift or time (scale factor or non-cosmological sim time)."""
    from ..load.snapshot import subboxVals

    if not all and snap is None:
        snap = sP.snap
        assert snap is not None, "Input either snap or sP.snap required."

    _, sbStr1, _ = subboxVals(sP.subbox)

    # load snapshot -> redshift mapping files
    snaps = redshiftToSnapNum(load=True, sP=sP)

    # scale factor or redshift?
    val = snaps["redshifts"]
    if time:
        val = snaps["times"]

    # all values or a given scalar or array list?
    if all:
        w = np.where(val >= 0.0)[0]  # remove empties past end of number of snaps
        return val[0 : w.max() + 1]

    return val[snap]


def snapNumToAgeFlat(sP, snap=None):
    """Convert snapshot number to approximate age of the universe at that time."""
    z = snapNumToRedshift(sP, snap=snap)
    return sP.units.redshiftToAgeFlat(z)


def crossMatchSubhalosBetweenRuns(sP_from, sP_to, subhaloInds_from_search, method="LHaloTree"):
    """Given a set of subhaloInds_from_search in sP_from, find matched subhalos in sP_to.

    Can implement many methods. For now, uses external (pre-generated) postprocessing/SubhaloMatching/
    for TNG_method runs, or postprocessing/SubhaloMatchingToDark/ for Illustris/TNG to DMO runs, or
    postprocessing/SubhaloMatchingToIllustris/ for TNG->Illustris runs.
    Return is an int32 array of the same size as input, where -1 indicates no match.
    """
    assert method in ["LHaloTree", "SubLink", "Lagrange", "Positional", "PositionalAll"]
    assert sP_from != sP_to

    # positional cross-match between two different runs?
    if method in ["Positional", "PositionalAll"]:
        subhaloInds_from_search = np.array(subhaloInds_from_search)
        r = np.zeros(subhaloInds_from_search.size, dtype="int32") - 1
        assert sP_to.boxSize == sP_from.boxSize

        # matches which differ by more than this amount are discarded
        massDeltaMaxDex = 0.3

        # filter subhaloInds_from_search to centrals only (ignore satellites if any requested)
        css = "cen" if method == "Positional" else "all"
        cen_inds_to = cenSatSubhaloIndices(sP_to, cenSatSelect=css)
        cen_inds_from = cenSatSubhaloIndices(sP_from, cenSatSelect=css)

        _, ind_from = match(cen_inds_from, subhaloInds_from_search)
        subhaloInds_from = subhaloInds_from_search[ind_from]

        # load halo masses and positions from both runs
        mhalo_to = sP_to.groupCat(fieldsSubhalos=["mhalo_200_log"])
        mhalo_from = sP_from.groupCat(fieldsSubhalos=["mhalo_200_log"])

        pos_from = sP_from.groupCat(fieldsSubhalos=["SubhaloPos"])
        pos_to = sP_to.groupCat(fieldsSubhalos=["SubhaloPos"])

        pos_to_cen = pos_to[cen_inds_to, :]

        # loop over each requested search subhalo
        for i, subhaloInd_from in enumerate(subhaloInds_from):
            # calculate distances to all centrals in illustris
            pos_from_loc = np.squeeze(pos_from[subhaloInd_from, :])
            dists = periodicDists(pos_from_loc, pos_to_cen, sP_from)
            pos_to_cen_ind = np.where(dists == dists.min())[0]
            subhaloInd_to = cen_inds_to[pos_to_cen_ind]

            # pass mass requirement? if not, do not save this match
            if np.abs(mhalo_from[subhaloInds_from[i]] - mhalo_to[subhaloInd_to]) > massDeltaMaxDex:
                continue

            r[ind_from[i]] = subhaloInd_to

        return r

    # are we cross-matching between Illustris and TNG with the Lagrange catalog?
    if sP_from.run == "illustris" and sP_to.run == "tng":
        assert method == "Lagrange"
        assert sP_from.res == sP_to.res == 1820  # otherwise generalize

        matchFilePath = sP_to.postPath + "/SubhaloMatchingToIllustris/"
        matchFileName = matchFilePath + "LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5" % sP_to.snap

        with h5py.File(matchFileName, "r") as f:
            inds_tng = f["SubhaloIndexFrom"][()]
            inds_illustris = f["SubhaloIndexTo"][()]
            # score = f["Score"][()]

        r = np.zeros(len(subhaloInds_from_search), dtype="int32") - 1

        # instead: pick central with minimum ID (oh my, do something better)
        cen_inds_to = cenSatSubhaloIndices(sP=sP_to, cenSatSelect="cen")
        print("Warning: inverse Lagrange mapping, taking most massive centrals (doesnt really work).")

        for i in range(len(subhaloInds_from_search)):
            w = np.where(inds_illustris == subhaloInds_from_search[i])[0]

            # candidates
            cand_inds_tng = inds_tng[w]

            # which are centrals? take min ID (most massive)
            cand_inds_tng, _ = match(cen_inds_to, cand_inds_tng)

            r[i] = cen_inds_to[cand_inds_tng].min()

        return r

    if sP_from.run == "tng" and sP_to.run == "illustris":
        assert method == "Lagrange", "Positional"
        r = np.zeros(len(subhaloInds_from_search), dtype="int32") - 1

        if method == "Lagrange":
            assert sP_from.res == sP_to.res == 1820  # otherwise generalize

            matchFilePath = sP_from.postPath + "/SubhaloMatchingToIllustris/"
            matchFileName = matchFilePath + "LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5" % sP_from.snap

            with h5py.File(matchFileName, "r") as f:
                inds_tng = f["SubhaloIndexFrom"][()]
                inds_illustris = f["SubhaloIndexTo"][()]
                # score = f['Score'][()]

            match_inds_source, match_inds_search = match(inds_tng, subhaloInds_from_search)
            r[match_inds_search] = inds_illustris[match_inds_source]

        return r

    assert sP_from.snap == sP_to.snap

    # are we cross-matching between two non-fiducial sims.rTNG_method runs?
    if (
        sP_from.run == "tng"
        and sP_from.variant.isdigit()
        and sP_to.run == "tng"
        and sP_to.variant.isdigit()
        and int(sP_to.variant) != 0
        and int(sP_from.variant) != 0
    ):
        # we route all such requests through the fiducial run as an intermediate step by self-calling
        assert sP_from.res == sP_to.res  # otherwise we need to implement: chain through resolution levels
        sP_fid = simParams(res=sP_from.res, run=sP_from.run, redshift=sP_from.redshift, variant=0000)

        # what are the matched halos of 'from' in 'fiducial'
        match_from_fid = crossMatchSubhalosBetweenRuns(sP_from, sP_fid, subhaloInds_from_search, method=method)

        # what are these match results in 'to'
        match_to_fid = crossMatchSubhalosBetweenRuns(sP_fid, sP_to, match_from_fid, method=method)

        w = np.where(match_from_fid == -1)
        match_to_fid[w] = -1  # flag any matches which failed first step as complete failures

        return match_to_fid

    # are we matching to the DMO analog of sP_from? or vice versa
    swapMatchDirection = False

    dmo1 = sP_to.run == sP_from.run + "_dm" and sP_to.res == sP_from.res and sP_to.hInd == sP_from.hInd
    dmo2 = sP_to.run + "_dm" == sP_from.run and sP_to.res == sP_from.res and sP_to.hInd == sP_from.hInd

    if dmo1:
        # yes, sP_to is the DMO version of sP_from
        basePath = sP_from.postPath + "SubhaloMatchingToDark/%s_" % method
        filePath = basePath + "%03d.hdf5" % sP_from.snap
    elif dmo2:
        # yes, sP_from is the DMO version of sP_to
        basePath = sP_to.postPath + "SubhaloMatchingToDark/%s_" % method
        filePath = basePath + "%03d.hdf5" % sP_to.snap

        swapMatchDirection = True
    else:
        # no, use a match subdirectory to a specific run
        basePath = sP_from.postPath + "SubhaloMatching/%s/%s_" % (sP_to.simName, method)
        filePath = basePath + "%03d.hdf5" % sP_from.snap

        # only have matching in the other direction?
        if not isfile(filePath):
            basePath = sP_to.postPath + "SubhaloMatching/%s/%s_" % (sP_from.simName, method)
            filePath = basePath + "%03d.hdf5" % sP_to.snap
            assert isfile(filePath)  # otherwise fail

            swapMatchDirection = True

    # load results of a matching catalog
    if method == "LHaloTree":
        with h5py.File(filePath, "r") as f:
            subhaloInds_from = f["SubhaloIndexFrom"][()]
            subhaloInds_to = f["SubhaloIndexTo"][()]

    if method == "SubLink":
        with h5py.File(filePath, "r") as f:
            subhaloInds_from = np.arange(f["DescendantIndex"].size)
            subhaloInds_to = f["DescendantIndex"][()]

        w = np.where(subhaloInds_to >= 0)
        subhaloInds_from = subhaloInds_from[w]
        subhaloInds_to = subhaloInds_to[w]

    # if we requested a DMO->Physics match, we have instead loaded [the only thing available] the
    # Physics->DMO SubhaloMatchingToDark information. so, swap the 'from' and 'to' subhaloInds
    # such that subhaloInds_from_search is matched against the DMO and the return is for the Physics
    if swapMatchDirection:
        subhaloInds_from, subhaloInds_to = subhaloInds_to, subhaloInds_from

    # find matches and make return
    match_inds_source, match_inds_search = match(subhaloInds_from, subhaloInds_from_search)

    r = np.zeros(len(subhaloInds_from_search), dtype="int32") - 1

    r[match_inds_search] = subhaloInds_to[match_inds_source]

    return r


# --- periodic B.C. ---


def correctPeriodicDistVecs(vecs, sP):
    """Enforce periodic B.C. for distance vectors (effectively component by component)."""
    vecs[np.where(vecs > sP.boxSize * 0.5)] -= sP.boxSize
    vecs[np.where(vecs <= -sP.boxSize * 0.5)] += sP.boxSize


def correctPeriodicPosVecs(vecs, sP):
    """Enforce periodic boundary conditions for positions.

    To do so, add boxSize to any negative points, subtract boxSize from any points outside box.
    """
    vecs[np.where(vecs < 0.0)] += sP.boxSize
    vecs[np.where(vecs >= sP.boxSize)] -= sP.boxSize


def correctPeriodicPosBoxWrap(vecs, sP):
    """Determine if an array of positions spans a periodic boundary and, if so, wrap them (for plotting).

    For an array of positions [N,3], determine if they span a periodic boundary (e.g. half are near
    x=0 and half are near x=BoxSize). If so, wrap the high coordinate value points by a BoxSize, making
    them negative. Suitable for plotting particle positions in global coordinates. Return indices of
    shifted coordinates so they can be shifted back, in the form of dict with an entry for each
    shifted dimension and key equal to the dimensional index.
    """
    r = {}

    for i in range(3):
        w1 = np.where(vecs[:, i] < sP.boxSize * 0.1)[0]
        w2 = np.where(vecs[:, i] > sP.boxSize * 0.9)[0]

        # satisfy wrap criterion for this axis?
        if len(w1) and len(w2):
            wCheck = np.where((vecs[:, i] > sP.boxSize * 0.5) & (vecs[:, i] < sP.boxSize * 0.8))[0]
            if len(wCheck):
                raise Exception("Positions spanning very large fraction of box, something strange.")

            wMove = np.where(vecs[:, i] > sP.boxSize * 0.8)[0]
            vecs[wMove, i] -= sP.boxSize

            # store indices of shifted coordinates for return
            r[i] = wMove

    return r


def periodicDists(pt, vecs, sP, chebyshev=False):
    """Calculate distances correctly taking into account periodic boundary conditions.

    Args:
      pt (list[3] or [N,3]): if pt is one point, distance from pt to all vecs.
        if pt is several points, distance from each pt to each vec (must have same number of points as vecs).
      vecs (list[3,N]): position array in periodic 3D space.
      sP (:py:class:`~util.simParams`): simulation instance.
      chebyshev (bool): use Chebyshev distance metric (greatest difference in positions along any one axis)
    """
    assert vecs.ndim in [1, 2]
    assert pt.ndim in [1, 2]

    if vecs.ndim == 1:
        # vecs.shape == [3], e.g. single 3-vector
        assert vecs.size == 3
        vecs = np.reshape(vecs, (1, 3))

    if len(pt) == 2:
        assert vecs.shape[1] == 2
        assert not chebyshev

        xDist = vecs[:, 0] - pt[0]
        yDist = vecs[:, 1] - pt[1]

        if not sP.isSubbox:
            correctPeriodicDistVecs(xDist, sP)
            correctPeriodicDistVecs(yDist, sP)

        return np.sqrt(xDist * xDist + yDist * yDist)

    # distances from one point (x,y,z) to a vector of other points [N,3]
    if pt.ndim == 1:
        xDist = vecs[:, 0] - pt[0]
        yDist = vecs[:, 1] - pt[1]
        zDist = vecs[:, 2] - pt[2]

    # distances from a vector of points [N,3] to another vector of other points [N,3]
    if pt.ndim == 2:
        assert vecs.shape[0] == pt.shape[0]
        xDist = vecs[:, 0] - pt[:, 0]
        yDist = vecs[:, 1] - pt[:, 1]
        zDist = vecs[:, 2] - pt[:, 2]

    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    if chebyshev:
        dists = np.abs(xDist)
        wy = np.where(np.abs(yDist) > dists)
        dists[wy] = np.abs(yDist[wy])
        wz = np.where(np.abs(zDist) > dists)
        dists[wz] = np.abs(zDist[wz])
    else:
        dists = np.sqrt(xDist * xDist + yDist * yDist + zDist * zDist)

    return dists


def periodicDists2D(pt, vecs, sP, chebyshev=False):
    """Calculate 2D distances correctly taking into account periodic boundary conditions.

    Args:
      pt (list[2] or [N,2]): if pt is one point, distance from pt to all vecs.
        if pt is several points, distance from each pt to each vec (must have same number of points as vecs).
      vecs (list[2,N]): position array in periodic 3D space.
      sP (:py:class:`~util.simParams`): simulation instance.
      chebyshev (bool): use Chebyshev distance metric (greatest difference in positions along any one axis)
    """
    assert vecs.ndim in [1, 2]
    assert pt.ndim in [1, 2]

    if vecs.ndim == 1:
        # vecs.shape == [3], e.g. single 3-vector
        assert vecs.size == 2
        vecs = np.reshape(vecs, (1, 2))

    # distances from one point (x,y) to a vector of other points [N,2]
    if pt.ndim == 1:
        xDist = vecs[:, 0] - pt[0]
        yDist = vecs[:, 1] - pt[1]

    # distances from a vector of points [N,2] to another vector of other points [N,2]
    if pt.ndim == 2:
        assert vecs.shape[0] == pt.shape[0]
        xDist = vecs[:, 0] - pt[:, 0]
        yDist = vecs[:, 1] - pt[:, 1]

    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)

    if chebyshev:
        dists = np.abs(xDist)
        wy = np.where(np.abs(yDist) > dists)
        dists[wy] = np.abs(yDist[wy])
    else:
        dists = np.sqrt(xDist * xDist + yDist * yDist)

    return dists


def periodicDistsSq(pt, vecs, sP):
    """As cosmo.util.periodicDists() but specialized, without error checking, and no sqrt.

    Works either for 2D or 3D, where the dimensions of pt and vecs should correspond.
    """
    if len(pt) == 2:
        assert vecs.shape[1] == 2

        xDist = vecs[:, 0] - pt[0]
        yDist = vecs[:, 1] - pt[1]

        if not sP.isSubbox:
            correctPeriodicDistVecs(xDist, sP)
            correctPeriodicDistVecs(yDist, sP)

        return xDist * xDist + yDist * yDist

    # fall through to normal 3D case
    xDist = vecs[:, 0] - pt[0]
    yDist = vecs[:, 1] - pt[1]
    zDist = vecs[:, 2] - pt[2]

    if not sP.isSubbox:
        correctPeriodicDistVecs(xDist, sP)
        correctPeriodicDistVecs(yDist, sP)
        correctPeriodicDistVecs(zDist, sP)

    return xDist * xDist + yDist * yDist + zDist * zDist


def periodicPairwiseDists(pts, sP):
    """Calculate pairwise distances between all 3D points, correctly taking into account periodic B.C."""
    nPts = pts.shape[0]
    num = int(nPts * (nPts - 1) / 2)

    ii = 0
    index0 = np.arange(nPts - 1, dtype="int32") + 1
    index1 = np.zeros(num, dtype="int32")
    index2 = np.zeros(num, dtype="int32")

    # set up indexing
    for i in np.arange(nPts - 1):
        n1 = nPts - (i + 1)
        index1[ii : ii + n1] = i
        index2[ii : ii + n1] = index0[0:n1] + i
        ii += n1

    # component wise difference
    xDist = pts[index1, 0] - pts[index2, 0]
    yDist = pts[index1, 1] - pts[index2, 1]
    zDist = pts[index1, 2] - pts[index2, 2]

    # correct for periodic distance function
    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    dists = np.sqrt(xDist * xDist + yDist * yDist + zDist * zDist)

    return dists, index1, index2


# --- other ---


def inverseMapPartIndicesToSubhaloIDs(
    sP, indsType, ptName, debug=False, flagFuzz=True, SubhaloLenType=None, SnapOffsetsSubhalo=None
):
    """Compute the subhalo ID that each particle/cell belongs to.

    For a particle type ptName and snapshot indices for that type indsType, compute the
    subhalo ID to which each particle index belongs. Optional: SubhaloLenType (from groupcat)
    and SnapOffsetsSubhalo (from groupCatOffsetListIntoSnap()), otherwise loaded on demand.
    If flagFuzz is True (default), particles in FoF fuzz are marked as outside any subhalo,
    otherwise they are attributed to the closest (prior) subhalo.
    """
    if SubhaloLenType is None:
        SubhaloLenType = sP.groupCat(fieldsSubhalos=["SubhaloLenType"])
    if SnapOffsetsSubhalo is None:
        SnapOffsetsSubhalo = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]

    gcLenType = SubhaloLenType[:, sP.ptNum(ptName)]
    gcOffsetsType = SnapOffsetsSubhalo[:, sP.ptNum(ptName)][:-1]

    # val gives the indices of gcOffsetsType such that, if each indsType was inserted
    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the
    # subhalo offset (i.e. first particle) works correctly
    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the
    # subhalo offset index -after- which the particle should be inserted
    val = np.searchsorted(gcOffsetsType - 1, indsType) - 1
    val = val.astype("int32")

    # search and flag all matches where the indices exceed the length of the
    # subhalo they have been assigned to, e.g. either in fof fuzz, in subhalos with
    # no particles of this type, or not in any subhalo at the end of the file
    if flagFuzz:
        gcOffsetsMax = gcOffsetsType + gcLenType - 1
        ww = np.where(indsType > gcOffsetsMax[val])[0]

        if len(ww):
            val[ww] = -1

    if debug:
        # for all inds we identified in subhalos, verify parents directly
        for i in range(len(indsType)):
            if val[i] < 0:
                continue
            assert indsType[i] >= gcOffsetsType[val[i]]
            if flagFuzz:
                assert indsType[i] < gcOffsetsType[val[i]] + gcLenType[val[i]]
                assert gcLenType[val[i]] != 0

    return val


def inverseMapPartIndicesToHaloIDs(sP, indsType, ptName, GroupLenType=None, SnapOffsetsGroup=None, debug=False):
    """Compute the halo ID that each particle/cell belongs to.

    For a particle type ptName and snapshot indices for that type indsType, compute the
    halo/fof ID to which each particle index belongs. Optional: GroupLenType (from groupcat)
    and SnapOffsetsGroup (from groupCatOffsetListIntoSnap()), otherwise loaded on demand.
    """
    if GroupLenType is None:
        GroupLenType = sP.groupCat(fieldsHalos=["GroupLenType"])
    if SnapOffsetsGroup is None:
        SnapOffsetsGroup = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]

    gcLenType = GroupLenType[:, sP.ptNum(ptName)]
    gcOffsetsType = SnapOffsetsGroup[:, sP.ptNum(ptName)][:-1]

    # val gives the indices of gcOffsetsType such that, if each indsType was inserted
    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the
    # subhalo offset (i.e. first particle) works correctly
    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the
    # subhalo offset index -after- which the particle should be inserted
    val = np.searchsorted(gcOffsetsType - 1, indsType) - 1
    val = val.astype("int32")

    # flag all matches where the indices are past the end of the fofs (at end of file)
    gcOffsetMax = gcOffsetsType[-1] + gcLenType[-1] - 1
    ww = np.where(indsType > gcOffsetMax)[0]

    if len(ww):
        val[ww] = -1

    if debug:
        # verify directly
        for i in range(len(indsType)):
            if val[i] < 0:
                continue
            assert indsType[i] >= gcOffsetsType[val[i]]
            assert indsType[i] < gcOffsetsType[val[i]] + gcLenType[val[i]]
            assert gcLenType[val[i]] != 0

    return val


def subhaloIDsToBoundingPartIndices(sP, subhaloIDs, groups=False, strictSubhalos=False):
    """For a list of subhalo IDs, identfy the particle index that bounds all their members.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      subhaloIDs (array-like): list of subhalo IDs.
      groups (bool): if True, input IDs are group (FoF) IDs, not subhalo IDs.
      strictSubhalos (bool): if True, do not use parent groups to bound subhalo members, instead return exact
        bounding index ranges.

    Return:
      dict: with an entry for each partType, whose value is a 2-tuple of the particle index range bounding the
        members of the parent groups of this list of subhalo IDs. Indices are inclusive as in snapshotSubset().
    """
    if strictSubhalos:
        assert not groups
    if groups:
        assert not strictSubhalos  # mutually exclusive

    first_sub = subhaloIDs[0]
    last_sub = subhaloIDs[-1]

    min_sub = np.min(subhaloIDs)
    max_sub = np.max(subhaloIDs)

    if first_sub != min_sub:
        print("Warning: First sub [%d] was not minimum of subhaloIDs [%d]." % (first_sub, min_sub))
        first_sub = min_sub
    if last_sub != max_sub:
        print("Warning: Last sub [%d] was not maximum of subhaloIDs [%d]." % (last_sub, max_sub))
        last_sub = max_sub

    if not groups:
        # get parent groups of extremum subhalos
        first_sub_groupID = sP.groupCatSingle(subhaloID=first_sub)["SubhaloGrNr"]
        last_sub_groupID = sP.groupCatSingle(subhaloID=last_sub)["SubhaloGrNr"]
    else:
        # input 'subhaloIDs' are already group IDs
        first_sub_groupID = first_sub
        last_sub_groupID = last_sub

    # load group offsets
    snapHeader = sP.snapshotHeader()
    snapOffsets = sP.groupCatOffsetListIntoSnap()
    offsets_pt = snapOffsets["snapOffsetsGroup"]

    if strictSubhalos:
        # use subhalo offsets instead
        offsets_pt = snapOffsets["snapOffsetsSubhalo"]

        # and change *_sub_groupID to actually be our subhalo IDs
        first_sub_groupID = first_sub
        last_sub_groupID = last_sub

        # load length of last subhalo
        last_sub_length = sP.groupCatSingle(subhaloID=last_sub)["SubhaloLenType"]

    r = {}
    for ptName in ["gas", "dm", "stars", "bhs"]:
        if snapHeader["NumPart"][sP.ptNum(ptName)] == 0:
            continue  # no particles of this type

        # bound upper range using the start of the next group/subhalo, minus one
        r[ptName] = offsets_pt[[first_sub_groupID, last_sub_groupID + 1], sP.ptNum(ptName)]
        # the final index is inclusive, as in snapshotSubset(), but not as in numpy indexing
        r[ptName][1] -= 1

        if strictSubhalos:
            # need to use the end of the final subhalo, instead of the beginning of the one after,
            # as they are not necessarily contiguous (if they span across groups)
            # also solves the issue of using the very last subhalo of the catalog
            r[ptName] = offsets_pt[[first_sub_groupID, last_sub_groupID], sP.ptNum(ptName)]
            r[ptName][1] += last_sub_length[sP.ptNum(ptName)] - 1

        if r[ptName][1] == -1:
            # can occur if all subhalos have no particles of this type (but snap does)
            subLenType = sP.subhalos("SubhaloLenType")
            subLenType = subLenType[:, sP.ptNum(ptName)]
            assert subLenType.sum() == 0 and r[ptName][0] == 0
            r[ptName][1] = 0
        else:
            # otherwise we read the last/unused element of offsets_pt
            assert r[ptName][1] >= 0 or snapHeader["NumPart"][sP.ptNum(ptName)] == 1

    if "stars" in r:
        r["wind"] = r["stars"]

    return r


def cenSatSubhaloIndices(sP=None, gc=None, cenSatSelect=None):
    """Return a tuple of three sets of subhalo IDs: centrals only, centrals & satellites, and satellites only."""
    if sP is None:
        assert "halos" in gc
        assert "GroupFirstSub" in gc["halos"] and "Group_M_Crit200" in gc["halos"]

    if "css_inds_w1" in sP.data:
        # load from cache
        w1, w2, w3 = sP.data["css_inds_w1"], sP.data["css_inds_w2"], sP.data["css_inds_w3"]
    else:
        if gc is None:
            # load what we need
            assert sP is not None

            gc = sP.groupCat(fieldsHalos=["GroupFirstSub", "Group_M_Crit200"])

        mask = np.zeros(sP.numSubhalos, dtype="int16")

        wHalo = np.where((gc["GroupFirstSub"] >= 0) & (gc["Group_M_Crit200"] > 0))

        # zoom simulation or virtual box (TNG-Cluster)?
        if sP.isZoomOrVirtualBox:
            # -only- include indices for centrals/satellites of primary targeted (i.e. uncontaminated) halos
            n_subs = sP.halos("GroupNsubs")
            pri_target = sP.halos("GroupPrimaryZoomTarget")

            wHaloPri = np.where((gc["GroupFirstSub"] >= 0) & (gc["Group_M_Crit200"] > 0) & (pri_target == 1))

            # indices
            w1 = gc["GroupFirstSub"][wHaloPri]  # centrals only
            w1_all = gc["GroupFirstSub"][wHalo]
            mask[w1] = 2  # pri-targeted centrals
            mask[w1_all] = 1  # other (ignored) centrals

            for halo_ind in wHaloPri[0]:
                start = gc["GroupFirstSub"][halo_ind]
                end = start + n_subs[halo_ind]
                mask[start:end] = 3  # satellites in pri-targeted centrals

            w2 = np.where(mask >= 2)[0]  # centrals + satellites
            w3 = np.where(mask == 3)[0]  # satellites

            assert n_subs[wHaloPri].sum() == w3.size
        else:
            # normal full box: halos with a primary subhalo
            w1 = gc["GroupFirstSub"][wHalo]  # centrals only
            w2 = np.arange(sP.numSubhalos)  # centrals + satellites

            mask[w1] = 1
            w3 = np.where(mask == 0)[0]  # satellites only
            # w3 = np.array( list(set(w2) - set(w1)) ) # satellites only (slow)

        # cache
        sP.data["css_inds_w1"] = w1
        sP.data["css_inds_w2"] = w2
        sP.data["css_inds_w3"] = w3

    if cenSatSelect is None:
        return w1, w2, w3

    if cenSatSelect in ["cen", "pri", "primary", "central", "centrals"]:
        return w1
    if cenSatSelect in ["sat", "sec", "secondary", "satellite", "satellites"]:
        return w3
    if cenSatSelect in ["all", "both"]:
        return w2


def subboxSubhaloCat(sP, sbNum):
    """Generate the SubboxSubhaloList catalog giving the intersection of fullbox subhalos with subboxes vs time.

    Use the merger trees from the fullbox and interpolate positions to subbox times.
    Determine interpolate properties of relevant subhalos at each subbox snapshot
    """
    minEdgeDistRedshifts = [100.0, 6.0, 4.0, 3.0, 2.0, 1.0, 0.0]

    def _inSubbox(pos):
        # return a vector of True or False entries, if pos (3-vector, or [N,3] vector) is inside subbox.
        # also return the minimum distance between pos and the subbox boundaries along any coordinate at each time.
        dist_x = np.abs(pos[:, 0] - subboxCen[0])
        dist_y = np.abs(pos[:, 1] - subboxCen[1])
        dist_z = np.abs(pos[:, 2] - subboxCen[2])

        # subbox cannot cross periodic boundary, so no need to account for periodic BCs here
        inside = (dist_x < subboxHalfSize) & (dist_y < subboxHalfSize) & (dist_z < subboxHalfSize)

        # calculate minimum distances along each axis (negative = outside, positive = inside)
        min_dists = np.min(
            np.vstack((subboxHalfSize - dist_x, subboxHalfSize - dist_y, subboxHalfSize - dist_z)), axis=0
        )

        return inside, min_dists

    fileBase = sP.postPath + "/SubboxSubhaloList/"
    filePath = fileBase + "subbox%d_%d.hdf5" % (sbNum, sP.snap)

    # check for existence and load
    if isfile(filePath):
        r = {}
        with h5py.File(filePath, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    r = {}

    # calculate new
    if not isdir(fileBase):
        mkdir(fileBase)

    # subbox properties
    subboxCen = np.array(sP.subboxCen[sbNum])
    subboxSize = np.array(sP.subboxSize[sbNum])

    subboxHalfSize = subboxSize / 2
    # subboxMin = subboxCen - subboxSize/2
    # subboxMax = subboxCen + subboxSize/2

    sP_sub = simParams(res=sP.res, run=sP.run, variant="subbox%d" % sbNum)

    # load snapshot meta-data
    snapTimes = snapNumToRedshift(sP, time=True, all=True)
    subboxTimes = snapNumToRedshift(sP_sub, time=True, all=True)

    minEdgeIndices = []

    for redshift in minEdgeDistRedshifts:
        # pre-search for indices of each redshift range within which we record minimum edge distances
        inds_local = np.where(1 / subboxTimes - 1 <= redshift)[0]

        if len(inds_local) == 0:
            assert redshift == 0.0  # i.e. no subbox saved exactly at z=0, use last one
            assert 1 / subboxTimes[-1] - 1 < 0.01
            inds_local = np.array([subboxTimes.size - 1])

        minEdgeIndices.append(inds_local)

    # locate the fullbox snapshot according to each subbox time for convenience, and vice versa (-1 indicates no match)
    r["SubboxScaleFac"] = subboxTimes
    r["SnapNumMapApprox"] = np.zeros(subboxTimes.size, dtype="float32")
    r["FullBoxSnapNum"] = np.zeros(subboxTimes.size, dtype="int16") - 1
    r["SubboxSnapNum"] = np.zeros(snapTimes.size, dtype="int16") - 1

    for i, time in enumerate(snapTimes):
        w = np.where(subboxTimes == time)[0]
        # if SubboxSyncModulo > 1, may not have exact matches anymore
        if len(w) == 0:
            foundTime, w = closest(subboxTimes, time)
            r["SnapNumMapApprox"][w] = foundTime - time
            w = [w]
        # assert len(w) in [1,2] # len(w)==2 occurs for duplicated z=1 subbox outputs, keep last
        r["SubboxSnapNum"][i] = w[-1]
        r["FullBoxSnapNum"][w[-1]] = i

    # load all MPBs (SubLink) of subhalos at this ending sP.snap
    ids = np.arange(sP.numSubhalos)
    fields = ["SubhaloPos", "SubhaloIDMostbound", "SnapNum"]

    mpbs = loadMPBs(sP, ids, fields=fields, treeName="SubLink")

    # loop over each MPB
    r["EverInSubboxFlag"] = np.zeros(ids.size, dtype="bool")

    print("Determining which of [%d] MPBs intersect subbox..." % ids.size)
    for i, subhaloID in enumerate(ids):
        if i % int(ids.size / 10.0) == 0:
            print(" %d%%" % np.ceil(float(i) / ids.size * 100))
        if subhaloID not in mpbs:
            continue

        pos = mpbs[subhaloID]["SubhaloPos"]
        flag, _ = _inSubbox(pos)

        if flag.sum() == 0:
            continue  # never inside

        if mpbs[subhaloID]["SnapNum"].size <= 3:  # minimum of 4 required for cspline
            continue  # no progenitor information in tree

        # set flag True if subhalo MPB is ever inside subbox, False if not
        r["EverInSubboxFlag"][i] = True

    # further processing of those whose MPBs are ever inside
    w = np.where(r["EverInSubboxFlag"])

    r["SubhaloIDs"] = ids[w]
    r["SubhaloPos"] = np.zeros((ids[w].size, subboxTimes.size, 3), dtype="float32")
    r["SubhaloPosExtrap"] = np.zeros((ids[w].size, subboxTimes.size), dtype="int16")
    r["SubhaloMBID"] = np.zeros((ids[w].size, subboxTimes.size), dtype=mpbs[0]["SubhaloIDMostbound"].dtype)
    r["SubhaloMinSBSnap"] = np.zeros(ids[w].size, dtype="int32") - 1
    r["SubhaloMaxSBSnap"] = np.zeros(ids[w].size, dtype="int32") - 1

    r["minEdgeDistRedshifts"] = np.array(minEdgeDistRedshifts, dtype="float32")
    r["SubhaloMinEdgeDist"] = np.zeros((ids[w].size, r["minEdgeDistRedshifts"].size), dtype="float32")

    print("\n[%d] subhalos intersect subbox, interpolating positions to subbox times..." % r["SubhaloIDs"].size)
    for i, subhaloID in enumerate(r["SubhaloIDs"]):
        if i % int(r["SubhaloIDs"].size / 10.0) == 0:
            print(" %d%%" % np.ceil(float(i) / r["SubhaloIDs"].size * 100))
        # create position interpolant
        mpb = mpbs[subhaloID]
        times = snapTimes[mpb["SnapNum"]]
        pos = np.zeros((subboxTimes.size, 3), dtype="float32")

        timeSnapMax = times.max()
        timeSnapMin = times.min()

        wInterp = np.where((subboxTimes >= timeSnapMin) & (subboxTimes <= timeSnapMax))
        wExtrap = np.where((subboxTimes < timeSnapMin) | (subboxTimes > timeSnapMax))

        for j in range(3):
            # each axis separately, first cubic spline interp
            f = interpolate.interp1d(times, mpb["SubhaloPos"][:, j], kind="cubic")
            pos[wInterp, j] = f(subboxTimes[wInterp])

            # linear extrapolation
            f = interpolate.interp1d(times, mpb["SubhaloPos"][:, j], kind="linear", fill_value="extrapolate")
            pos[wExtrap, j] = f(subboxTimes[wExtrap])

        # flag for extrapolated positions
        posExtrapolated = np.zeros(subboxTimes.size, dtype="int16")
        posExtrapolated[wExtrap] = 1

        # verify: positions should match where a subbox intersects a full box
        # for j, time in enumerate(times):
        #    w = np.where(subboxTimes == time)[0]
        #    assert np.sum(np.abs(mpb['SubhaloPos'][j,:]-pos[w,:])) < 1e-8

        # replicate most bound IDs to allow them to be re-located in subbox (alternative position possibility)
        mbID = np.zeros(subboxTimes.size, dtype=mpb["SubhaloIDMostbound"].dtype)
        mbID[r["SubboxSnapNum"][mpb["SnapNum"]]] = mpb["SubhaloIDMostbound"]
        for j in range(mbID.size - 2, -1, -1):
            if mbID[j] == 0:
                mbID[j] = mbID[j + 1]

        # store
        r["SubhaloPos"][i, :, :] = pos
        r["SubhaloPosExtrap"][i, :] = posExtrapolated
        r["SubhaloMBID"][i, :] = mbID

        # minimum distance to subbox boundary and bounding snapshots where this subhalo is inside the subbox
        flags, min_axis_dists = _inSubbox(pos)
        w = np.where(flags)[0]

        for j in range(r["minEdgeDistRedshifts"].size):  # negative = outside, positive = inside
            r["SubhaloMinEdgeDist"][i, j] = min_axis_dists[minEdgeIndices[j]].min()

        if len(w) == 0:
            # mpb['SubhaloPos'] is inside, but pos is not: should be rare (and require SnapNumMapApprox.sum()>0)
            print("Warning: subhalo [%d] ID [%d] interpolated positions never inside subbox..." % (i, subhaloID))
            continue

        r["SubhaloMinSBSnap"][i] = w.min()
        r["SubhaloMaxSBSnap"][i] = w.max()

    # intermediate save now
    with h5py.File(filePath, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved intermediate: [%s]" % filePath)

    return r


def subboxSubhaloCatExtend(sP, sbNum, redo=False):
    """Extend the SubboxSubhaloList catalog with custom (interpolated) properties.

    For example, 30 pkpc stellar masses. Separated into second step since this is a heavy calculation, restartable.
    """
    fileBase = sP.postPath + "/SubboxSubhaloList/"
    filePath = fileBase + "subbox%d_%d.hdf5" % (sbNum, sP.snap)

    # check for existence and load
    r = {}
    assert isfile(filePath)

    with h5py.File(filePath, "r") as f:
        for key in f:
            r[key] = f[key][()]

    # allocate for additional quantities
    nApertures = 3
    nSubSnaps = r["SubboxScaleFac"].size
    nSubs = r["SubhaloIDs"].size

    dataFieldKeys = [
        "SubhaloStars_Mass",
        "SubhaloGas_Mass",
        "SubhaloBH_Mass",
        "SubhaloBH_Mass2",
        "SubhaloBH_Mdot",
        "SubhaloBH_MdotEddington",
        "SubhaloGas_SFR",
        "SubhaloBH_CumEgyInjection_QM",
        "SubhaloBH_CumEgyInjection_RM",
        "SubhaloBH_Num",
    ]

    for key in dataFieldKeys:
        # if we are resuming, so these fields were loaded (and partially done) from the file, then do not recreate
        if key not in r:
            r[key] = np.zeros((nSubs, nApertures, nSubSnaps), dtype="float32")

    sP_sub = simParams(res=sP.res, run=sP.run, variant="subbox%d" % sbNum)

    # add done flag
    if "done" not in r or redo:
        r["done"] = np.zeros(nSubSnaps, dtype="int16")

    print("Loading subbox and deriving additional quantities...")

    for sbSnapNum in range(nSubSnaps):
        # load
        sP_sub.setSnap(sbSnapNum)
        z_sub = 1 / r["SubboxScaleFac"][sbSnapNum] - 1

        apertures_sq = [
            sP_sub.units.physicalKpcToCodeLength(30.0) ** 2,
            30.0**2,
            50.0**2,
        ]  # 30 pkpc, 30 ckpc/h, 50 ckpc/h
        print(" [%4d] z = %.2f (30pkpc rad_code = %.2f)" % (sbSnapNum, z_sub, np.sqrt(apertures_sq[0])), flush=True)

        if r["done"][sbSnapNum]:
            print(" skip, already done.")
            continue

        # loop over particle types
        for ptType in ["gas", "stars", "bh"]:
            loadFields = ["Masses", "Coordinates"]
            if ptType == "gas":
                loadFields.append("StarFormationRate")
            if ptType == "bh":
                loadFields.append("BH_CumEgyInjection_QM")
                loadFields.append("BH_CumEgyInjection_RM")
                loadFields.append("BH_Mass")
                loadFields.append("BH_Mdot")
                loadFields.append("BH_MdotEddington")

            # particle data load
            x = sP_sub.snapshotSubset(ptType, loadFields)

            if x["count"] == 0:
                continue

            # tree-based quantity reduction: build tree once per particle type
            pos = x["Coordinates"]
            posSearch = np.squeeze(r["SubhaloPos"][:, sbSnapNum, :])

            tree = buildFullTree(pos, boxSizeSim=0, treePrec="float64")

            for i, aperture_sq in enumerate(apertures_sq):
                # successive queries for each aperture search: mass sum
                hsml = np.sqrt(aperture_sq)
                result = calcQuantReduction(
                    pos, x["Masses"], hsml, op="sum", boxSizeSim=0, posSearch=posSearch, tree=tree
                )

                # save results
                saveStr = ptType.capitalize() if ptType != "bh" else "BH"
                r["Subhalo%s_Mass" % saveStr][:, i, sbSnapNum] = result

                if ptType == "gas":
                    result1 = calcQuantReduction(
                        pos, x["StarFormationRate"], hsml, op="sum", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    r["SubhaloGas_SFR"][:, i, sbSnapNum] = result1

                if ptType == "bh":
                    result1 = calcQuantReduction(
                        pos, x["BH_CumEgyInjection_QM"], hsml, op="max", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    result2 = calcQuantReduction(
                        pos, x["BH_CumEgyInjection_RM"], hsml, op="max", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    result3 = calcQuantReduction(
                        pos, x["BH_Mass"], hsml, op="max", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    result4 = calcQuantReduction(
                        pos, x["BH_Mass"], hsml, op="count", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    result5 = calcQuantReduction(
                        pos, x["BH_Mdot"], hsml, op="max", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )
                    result6 = calcQuantReduction(
                        pos, x["BH_MdotEddington"], hsml, op="max", boxSizeSim=0, posSearch=posSearch, tree=tree
                    )

                    r["SubhaloBH_CumEgyInjection_QM"][:, i, sbSnapNum] = result1
                    r["SubhaloBH_CumEgyInjection_RM"][:, i, sbSnapNum] = result2
                    r["SubhaloBH_Mass2"][:, i, sbSnapNum] = result3
                    r["SubhaloBH_Num"][:, i, sbSnapNum] = result4
                    r["SubhaloBH_Mdot"][:, i, sbSnapNum] = result5
                    r["SubhaloBH_MdotEddington"][:, i, sbSnapNum] = result6

        # save each subbox snap as we go (every field full, could optimize)
        r["done"][sbSnapNum] = 1

        with h5py.File(filePath, "r+") as f:
            for key in r:
                if key in f:
                    f[key][:] = r[key]
                else:
                    f[key] = r[key]

    print("Done: [%s]" % filePath)


def subsampleRandomSubhalos(sP, maxPerDex, mstarMinMax, mstar=None, cenOnly=False):
    """Sub-sample subhalos IDs randomly, uniformly across some quantity (can be generalized) such as stellar mass.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      maxPerDex (int): maximum number of subhalos to select per dex in the quantity.
      mstarMinMax (2-tuple): minimum and maximum values of the quantity to consider (e.g. stellar mass).
      mstar (ndarray): optional pre-loaded array of the quantity values for all subhalos..
      cenOnly (bool): if True, only consider central subhalos when mstar is not provided.
    """
    rng = np.random.default_rng(424242)
    binsize = 0.1  # dex
    numPerBin = np.max([1, int(maxPerDex * binsize)])

    if maxPerDex < 10:
        print("Note: subsampleRandomSubhalos() returning at least %d per %.1f dex bin." % (numPerBin, binsize))

    if mstar is None:
        mstar = sP.subhalos("mstar_30pkpc_log")
        if cenOnly:
            cen_flag = sP.subhalos("cen_flag")
            mstar[cen_flag == 0] = np.nan  # never select
    else:
        assert not cenOnly  # we don't know where mstar values came from, cannot apply

    inds = np.zeros(mstar.size, dtype="int32")
    count = 0
    xbin = mstarMinMax[0]

    while xbin < mstarMinMax[1]:
        with np.errstate(invalid="ignore"):
            w = np.where((mstar >= xbin) & (mstar < xbin + binsize))[0]

        if len(w) < numPerBin:
            inds[count : count + len(w)] = w
            count += len(w)
        else:
            inds_loc = rng.choice(w, numPerBin, replace=False)
            inds[count : count + inds_loc.size] = inds_loc
            count += inds_loc.size

        xbin += binsize

    inds = inds[0:count]

    return inds, mstar[inds]
