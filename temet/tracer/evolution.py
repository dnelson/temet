"""
Analysis for evolution of tracer quantities in time (for cosmo boxes/zooms).
"""

from collections import OrderedDict
from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np

from ..tracer.montecarlo import (
    defParPartTypes,
    globalAllTracersTimeEvo,
    globalTracerMPBMap,
    subhalosTracersTimeEvo,
    subhaloTracersTimeEvo,
)
from ..util.helper import pSplitRange
from ..util.simParams import simParams


# integer flags for accretion modes
ACCMODE_NONE = -1
ACCMODE_SMOOTH = 1
ACCMODE_MERGER = 2
ACCMODE_STRIPPED = 3

ACCMODES = OrderedDict(
    [("NONE", ACCMODE_NONE), ("SMOOTH", ACCMODE_SMOOTH), ("MERGER", ACCMODE_MERGER), ("STRIPPED", ACCMODE_STRIPPED)]
)  # same as above

# types of extrema which we know how to calculate
allowedExtTypes = ["min", "min_b015", "max", "max_b015"]

# default value: maximum redshift to track tracer properties back to
maxRedshift = 10.0


def zoomDataDriver(sP, fields, snapStep=1):
    """Run and save data files for tracer evolution in several quantities of interest."""
    # sP = simParams(res=11, run='zooms2', redshift=2.0, hInd=2)
    # fields = ['tracer_maxtemp','tracer_maxent','rad_rvir','vrad','entr','temp','sfr','subhalo_id']
    subhaloID = sP.zoomSubhaloID

    subhaloTracersTimeEvo(sP, subhaloID, fields, snapStep=snapStep)


def boxTracerDataCutout(snap=None):
    """Extract cutout tracer data for individual subhalos from a full box."""
    sP = simParams(res=1820, run="illustris", redshift=0.0)

    # config
    parPartTypes = ["gas", "stars"]
    toRedshift = 2.0
    trFields = ["tracer_windcounter"]
    parFields = ["pos", "vel", "temp", "sfr"]
    outPath = sP.postPath + "/tracer_tracks/cutout_%s" % sP.simName

    # subhalo list
    subhaloIDs = np.array([])  # specify

    subhalosTracersTimeEvo(sP, subhaloIDs, toRedshift, trFields, parFields, parPartTypes, outPath, onlySnap=snap)


def tracersTimeEvo(sP, fieldName, snapStep=None, all=True, pSplit=None):
    """Wrapper to handle zoom vs. box load."""
    # restricted load? task parallel
    if pSplit is not None:
        assert not sP.isZoom
        assert sP.haloInd is None and sP.subhaloInd is None and all is True
        assert fieldName != "meta"

        # determine index range
        _, nTracerTot = tracersMetaOffsets(sP, all="Halo")
        indRange = pSplitRange([0, nTracerTot], pSplit[1], pSplit[0])

        print(" tracersTimeEvo [%s] indRange [%d %d]" % (fieldName, indRange[0], indRange[1]))

        r = globalAllTracersTimeEvo(sP, fieldName, indRange=indRange)

        r["indRange"] = indRange
        return r

    # restricted load? full box, but only a [halo/subhalo] subset
    if all is False and (sP.haloInd is not None or sP.subhaloInd is not None):
        assert pSplit is None
        assert not sP.isZoom
        assert fieldName != "meta"

        # determine index range
        meta, nTracerTot = tracersMetaOffsets(sP)

        r = {}
        offset = 0

        for pt in meta.keys():
            if meta[pt]["length"] == 0:
                continue

            # request load of tracers for this halos/subhalo for this partType
            indRange = [meta[pt]["offset"], meta[pt]["offset"] + meta[pt]["length"]]
            rLocal = globalAllTracersTimeEvo(sP, fieldName, indRange=indRange)

            # save redshifts, snaps
            for key in ["redshifts", "snaps"]:
                r[key] = rLocal[key]

            # allocate main return if needed
            if fieldName not in r:
                nSnaps = rLocal[fieldName].shape[-1]
                r[fieldName] = np.zeros((nTracerTot, nSnaps), dtype=rLocal[fieldName].dtype)

            # stamp in return for this partType
            r[fieldName][offset : offset + meta[pt]["length"], ...] = rLocal[fieldName]
            offset += meta[pt]["length"]

        return r

    # global load
    if sP.isZoom:
        assert snapStep is not None
        r = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep)
    else:
        r = globalAllTracersTimeEvo(sP, fieldName)

    # global load requested?
    return r


def tracersMetaOffsets(sP, all=None, parIDs=None, trIDs=None, getPath=False):
    """Get offsets needed to load a halo/subhalo-restricted part of any of the tracer_tracks data.

    For a fullbox sP and either sP.haloInd or sP.subhaloInd specified.
    If all == 'Halo' or 'Subhalo', then return all respective offsets and lengths.
    If parIDs or trIDs is not None, then a ptName to load and return these IDs for directly.
    """
    assert ((sP.haloInd is not None) ^ (sP.subhaloInd is not None)) or (all is not None or getPath is True)
    if all is not None:
        assert all == "Halo" or all == "Subhalo"
    if parIDs or trIDs:
        assert all is None
    assert not sP.isZoom

    saveFilename = sP.postPath + "/tracer_tracks/tr_all_groups_%d_meta.hdf5" % (sP.snap)

    if getPath:
        return saveFilename

    if sP.haloInd is not None:
        gName = "Halo"
        dInd = sP.haloInd
    if sP.subhaloInd is not None:
        gName = "Subhalo"
        dInd = sP.subhaloInd

    # load
    r = {}

    with h5py.File(saveFilename, "r") as f:
        for ptName in defParPartTypes:
            r[ptName] = {}

            if all:
                # load all lengths and offsets
                r[ptName]["length"] = f[all]["TracerLength"][ptName][()]
                r[ptName]["offset"] = f[all]["TracerOffset"][ptName][()]
            else:
                # single subhalo/halo
                r[ptName]["length"] = f[gName]["TracerLength"][ptName][dInd]
                r[ptName]["offset"] = f[gName]["TracerOffset"][ptName][dInd]

        if parIDs:
            return f["ParentIDs"][r[parIDs]["offset"] : r[parIDs]["offset"] + r[parIDs]["length"]]
        if trIDs:
            return f["TracerIDs"][r[parIDs]["offset"] : r[parIDs]["offset"] + r[parIDs]["length"]]

    nTracerTot = np.sum([r[ptName]["length"] for ptName in r])

    return r, nTracerTot


def loadAllOrRestricted(sP, saveFilename, datasetName=None, indRange=None):
    """Load a dataset from a tracer_tracks file, either full or for a halo/subhalo subset."""
    # return for all tracers, or an indRange subset
    if sP.isZoom or (sP.haloInd is None and sP.subhaloInd is None):
        with h5py.File(saveFilename, "r") as f:
            if indRange is None:
                return f[datasetName][()]
            else:
                assert f[datasetName].ndim == 1
                return f[datasetName][indRange[0] : indRange[1]]

    # get offsets from meta and do [halo/subhalo]-restricted load and return (concat types)
    assert indRange is None
    meta, nTracerTot = tracersMetaOffsets(sP)

    if nTracerTot == 0:
        return None

    with h5py.File(saveFilename, "r") as f:
        r = np.zeros(nTracerTot, dtype=f[datasetName].dtype)
        offset = 0

        for pt in meta.keys():
            if meta[pt]["length"] == 0:
                continue

            r[offset : offset + meta[pt]["length"]] = f[datasetName][
                meta[pt]["offset"] : meta[pt]["offset"] + meta[pt]["length"]
            ]
            offset += meta[pt]["length"]

    return r


def accTime(sP, snapStep=1, rVirFac=1.0, pSplit=None, indRangeLoad=None):
    """Calculate 'halo accretion time' for each tracer, as the earliest (highest redshift) rvir crossing.

    Uses the 'rad_rvir' field.
    Argument: rVirFac = what fraction of the virial radius denotes the accretion time?
    """
    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + "/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_time_%d.hdf5" % (
            sP.zoomSubhaloID,
            True,
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
            rVirFac * 100,
        )
    else:
        splitStr = "" if pSplit is None else "_split-%d-%d"
        saveFilenameBase = sP.derivPath + "/trTimeEvo/acc_time_snap_%d-%d-%d_r%d%s.hdf5" % (
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
            rVirFac * 100,
            splitStr,
        )
        saveFilename = saveFilenameBase
        if pSplit is not None:
            saveFilename = saveFilename % (pSplit[0], pSplit[1])

    if not isdir(sP.derivPath + "/trTimeEvo"):
        mkdir(sP.derivPath + "/trTimeEvo")

    # check for existence of all split files for concatenation
    if isfile(saveFilename) and pSplit is not None:
        allExist = True
        allCount = 0

        for i in range(pSplit[1]):
            saveFileSplit_i = saveFilenameBase % (i, pSplit[1])
            if not isfile(saveFileSplit_i):
                allExist = False
                continue

            # record counts and dataset shape
            with h5py.File(saveFileSplit_i, "r") as f:
                allCount += f["accTimeInterp"].size

        if allExist:
            # all chunks exist, concatenate them now and continue
            accTimeInterp = np.zeros(allCount, dtype="float32") - 1.0
            print(" Concatenating into shape: ", accTimeInterp.shape)

            for i in range(pSplit[1]):
                saveFileSplit_i = saveFilenameBase % (i, pSplit[1])

                with h5py.File(saveFileSplit_i, "r") as f:
                    indRange = f["indRange"][()]
                    accTimeInterp[indRange[0] : indRange[1]] = f["accTimeInterp"][()]

            assert np.count_nonzero(accTimeInterp == -1.0) == 0  # all should be filled
            saveFilename = saveFilenameBase.replace(splitStr, "")
            assert not isfile(saveFilename)

            with h5py.File(saveFilename, "w") as f:
                f.create_dataset("accTimeInterp", data=accTimeInterp)

            print(" Concatenated new [%s] and saved." % saveFilename.split("/")[-1])
            print(" All chunks concatenated, please manually delete them now.")
        else:
            print("Chunk [%s] already exists, but all not yet done, exiting." % saveFilename.split("/")[-1])
            return None

    # load pre-existing
    if isfile(saveFilename):
        return loadAllOrRestricted(sP, saveFilename, "accTimeInterp", indRange=indRangeLoad)

    print("Calculating new accTime for [%s]..." % sP.simName)
    if pSplit is not None:
        print(" Split calculation [%d] of [%d]." % (pSplit[0], pSplit[1]))

    # calculate new: load radial histories
    data = tracersTimeEvo(sP, "rad_rvir", snapStep, all=True, pSplit=pSplit)

    # reverse so that increasing indices are increasing snapshot numbers
    data2d = data["rad_rvir"][:, ::-1]

    data["snaps"] = data["snaps"][::-1]
    data["redshifts"] = data["redshifts"][::-1]

    data2d[~np.isfinite(data2d)] = rVirFac * 10  # set NaN (untracked MPB) to large values (outside)

    # set mask to one for all radii less than factor
    mask2d = np.zeros_like(data2d, dtype="int16")
    ww = np.where(data2d < rVirFac)
    mask2d[ww] = 1

    # along second axis (snaps), take index (lowest snap number inside) which is nonzero
    firstSnapInsideInd = np.argmax(mask2d, axis=1)

    # interp between index and previous (one snap before first time inside) for non-discrete answer
    nTr = data["rad_rvir"].shape[0]
    accTimeInterp = np.zeros(nTr, dtype="float32")

    for i in range(nTr):
        if i % int(nTr / 100) == 0:
            print(" %4.1f%%" % (float(i) / nTr * 100.0))

        ind0 = firstSnapInsideInd[i]
        ind1 = firstSnapInsideInd[i] - 1

        if ind0 == 0:
            # never inside? flag with nan
            if mask2d[:, i].sum() == 0:
                accTimeInterp[i] = np.nan
                continue

            # actually inside from first available snapshot
            accTimeInterp[i] = data["redshifts"][0]
            continue

        assert ind0 > 0
        assert ind1 >= 0

        z0 = data["redshifts"][ind0]
        z1 = data["redshifts"][ind1]
        r0 = data2d[i, ind0]
        r1 = data2d[i, ind1]

        # linear interpolation, find redshift where rad_rvir=rVirFac
        accTimeInterp[i] = (rVirFac - r0) / (r1 - r0) * (z1 - z0) + z0

    # save
    with h5py.File(saveFilename, "w") as f:
        f["accTimeInterp"] = accTimeInterp

        if "indRange" in data:  # save pSplit portion
            f["pSplit"] = pSplit
            f["indRange"] = data["indRange"]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])
    return accTimeInterp


def redshiftsToClosestSnaps(data, redshifts, indsNotSnaps=False):
    """Return the nearest snapshot number to each redshift.

    Uses the data['redshifts'] and data['snaps'] mapping.
    By default, return simulation snapshot
    number, unless indsNotSnaps==True, in which case return the indices into the first dimension
    of data[field] for each tracer of the second dimension.
    """
    z_inds1 = np.searchsorted(data["redshifts"], redshifts)

    ww = np.where(z_inds1 == data["redshifts"].size)
    z_inds1[ww] -= 1

    z_inds0 = z_inds1 - 1

    z_dist1 = np.abs(redshifts - data["redshifts"][z_inds1])
    z_dist0 = np.abs(redshifts - data["redshifts"][z_inds0])

    if indsNotSnaps:
        accSnap = z_inds1
    else:
        accSnap = data["snaps"][z_inds1]

    with np.errstate(invalid="ignore"):  # ignore nan comparison RuntimeWarning
        ww = np.where(z_dist0 < z_dist1)

    if indsNotSnaps:
        accSnap[ww] = [z_inds0[ww]]
    else:
        accSnap[ww] = data["snaps"][z_inds0[ww]]

    # nan redshifts's (never inside rvir) got assigned to the earliest snapshot, flag them as -1
    accSnap[np.isnan(redshifts)] = -1

    return accSnap


def accMode(sP, snapStep=1, pSplit=None, indRangeLoad=None):
    """Calculate an 'accretion mode' categorization for each tracer based on its group membership history.

    Specifically, separate all tracers into one of [smooth/merger/stripped] defined as:
    * smooth: child of MPB or no subhalo at all z>=z_acc
    * merger: child of subhalo other than the MPB at z=z_acc
    * stripped: child of MPB or no subhalo at z=z_acc, but child of non-MPB subhalo at any z>z_acc
    Where z_acc is the accretion redshift defined as the first (highest z) crossing of the virial radius.
    """
    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + "/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_mode.hdf5" % (
            sP.zoomSubhaloID,
            True,
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
        )
    else:
        splitStr = "" if pSplit is None else "_split-%d-%d"
        saveFilenameBase = sP.derivPath + "/trTimeEvo/acc_mode_s%d-%d-%d%s.hdf5" % (
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
            splitStr,
        )
        saveFilename = saveFilenameBase
        if pSplit is not None:
            saveFilename = saveFilename % (pSplit[0], pSplit[1])

    # check for existence of all split files for concatenation
    if isfile(saveFilename) and pSplit is not None:
        allExist = True
        allCount = 0

        for i in range(pSplit[1]):
            saveFileSplit_i = saveFilenameBase % (i, pSplit[1])
            if not isfile(saveFileSplit_i):
                allExist = False
                continue

            # record counts and dataset shape
            with h5py.File(saveFileSplit_i, "r") as f:
                allCount += f["accMode"].size

        if allExist:
            # all chunks exist, concatenate them now and continue
            accMode = np.zeros(allCount, dtype="int8")
            print(" Concatenating into shape: ", accMode.shape)

            for i in range(pSplit[1]):
                saveFileSplit_i = saveFilenameBase % (i, pSplit[1])

                with h5py.File(saveFileSplit_i, "r") as f:
                    indRange = f["indRange"][()]
                    accMode[indRange[0] : indRange[1]] = f["accMode"][()]

            assert np.count_nonzero(accMode) == accMode.size  # all should be filled
            saveFilename = saveFilenameBase.replace(splitStr, "")
            assert not isfile(saveFilename)

            with h5py.File(saveFilename, "w") as f:
                f.create_dataset("accMode", data=accMode)

            print(" Concatenated new [%s] and saved." % saveFilename.split("/")[-1])
            print(" All chunks concatenated, please manually delete them now.")
        else:
            print("Chunk [%s] already exists, but all not yet done, exiting." % saveFilename.split("/")[-1])
            return None

    # load pre-existing
    if isfile(saveFilename):
        return loadAllOrRestricted(sP, saveFilename, "accMode", indRange=indRangeLoad)

    print("Calculating new accMode for [%s]..." % sP.simName)
    if pSplit is not None:
        print(" Split calculation [%d] of [%d]." % (pSplit[0], pSplit[1]))

    # load accTime, subhalo_id tracks, and MPB history
    data = tracersTimeEvo(sP, "subhalo_id", snapStep, all=True, pSplit=pSplit)
    data_snaps_min = data["snaps"].min()

    if sP.isZoom:
        mpb = sP.loadMPB(sP.zoomSubhaloID)
    else:
        if pSplit is not None:
            # load the subset of the tracer IDs we are working with and obtain only their MPBs
            metaPath = tracersMetaOffsets(sP, getPath=True)
            trIDs = loadAllOrRestricted(sP, metaPath, "TracerIDs", indRange=data["indRange"])
            mpbGlobal = globalTracerMPBMap(sP, halos=True, retMPBs=True, trIDs=trIDs, indRange=data["indRange"])
        else:
            mpbGlobal = globalTracerMPBMap(sP, halos=True, retMPBs=True)

    acc_time = accTime(sP, snapStep=snapStep, indRangeLoad=data["indRange"])

    # allocate return
    nTr = acc_time.size
    accMode = np.zeros(nTr, dtype="int8")

    # closest snapshot for each accretion time
    accSnap = redshiftsToClosestSnaps(data, acc_time)

    assert nTr == data["subhalo_id"].shape[0] == accSnap.size

    # prepare a mapping from snapshot number -> mpb[index]
    if sP.isZoom:
        mpbIndexMap = np.zeros(mpb["SnapNum"].max() + 1, dtype="int32") - 1
        mpbIndexMap[mpb["SnapNum"]] = np.arange(mpb["SnapNum"].max())
    else:
        mpbIndexMap = np.zeros(data["snaps"].max() + 1, dtype="int32")

    # make a mapping from snapshot number -> data[index]
    dataIndexMap = np.zeros(data["snaps"].max() + 1, dtype="int32") - 1
    dataIndexMap[data["snaps"]] = np.arange(data["snaps"].size)

    # start loop to determine each tracer
    for i in range(nTr):
        if i % int(nTr / 100) == 0:
            print(" %4.1f%%" % (float(i) / nTr * 100.0))

        # never inside rvir -> accMode is undetermined
        if accSnap[i] == -1:
            accMode[i] = ACCMODE_NONE
            continue

        # accretion time determined as earliest snapshot (e.g. z=10), we label this smooth
        if accSnap[i] == data_snaps_min:
            accMode[i] = ACCMODE_SMOOTH
            continue

        # (only needed for periodic boxes with multiple MPBs)
        if not sP.isZoom:
            # tracer is in FoF halo with no primary subhalo at sP.snap (no MPB)
            if mpbGlobal["subhalo_id"][i] == -1:
                accMode[i] = ACCMODE_NONE
                continue

            # extract MPB used for this tracer
            mpb = mpbGlobal["mpbs"][mpbGlobal["subhalo_id"][i]]

            # create new mapping into MPB for this tracer
            mpbIndexMap.fill(-1)
            mpbIndexMap[mpb["SnapNum"]] = np.arange(mpb["SnapNum"].size)

        # pull out indices
        mpbIndAcc = mpbIndexMap[accSnap[i]]
        dataIndAcc = dataIndexMap[accSnap[i]]

        # in fullboxes, we may not even have the MPB back to the accTime (not currently allowed for zooms)
        if mpbIndAcc == -1 and not sP.isZoom:
            accMode[i] = ACCMODE_NONE
            continue

        assert mpbIndAcc != -1
        assert dataIndAcc != -1
        assert data["snaps"][dataIndAcc] == mpb["SnapNum"][mpbIndAcc]
        assert data["snaps"][dataIndAcc] == accSnap[i]

        # merger?
        mpbSubfindID_AtAcc = mpb["SubfindID"][mpbIndAcc]
        trParSubfindID_AtAcc = data["subhalo_id"][dataIndAcc, i]

        if mpbSubfindID_AtAcc != trParSubfindID_AtAcc:
            # mismatch of MPB subfind ID and tracer parent subhalo ID at z_acc
            accMode[i] = ACCMODE_MERGER

            # assert trParSubfindID_AtAcc != -1 # this is allowed
            assert mpbSubfindID_AtAcc != -1  # guess this is techncially possible? if we have
            # for instance a skip and a ghost insert, then a rvir crossing could fall in a
            # snapshot where the mpb was not defined (we hit this for 104 override?)
            continue

        # smooth?
        trParAtAccAndEarlier_HaveAtSnapNums = data["snaps"][dataIndAcc:]
        mpbInds_AtMatchingSnapNums = mpbIndexMap[trParAtAccAndEarlier_HaveAtSnapNums]

        trParSubfindIDs_AtAccAndEarlier = np.squeeze(data["subhalo_id"][i, dataIndAcc:])
        mpbSubfindIDs_AtAccAndEarlier = mpb["SubfindID"][mpbInds_AtMatchingSnapNums].copy()

        # wherever mpbInds_AtMachingSnapNums is -1 (MPB is untracked at this snapshot),
        # rewrite the local mpbSubfindIDs_AtAccAndEarlier with -1 (i.e. once the MPB becomes
        # untracked, if at that point the tracer is within no subhalo, then we allow this to
        # count as smooth)
        ww = np.where(mpbInds_AtMatchingSnapNums == -1)
        mpbSubfindIDs_AtAccAndEarlier[ww] = -1

        # wherever trParSubfindIDs_AtAccAndEarlier is -1 (not in any subhalo), overwrite
        # the local mpbSubfindIDs_AtAccAndEarlier with these same values for the logic below
        ww = np.where(trParSubfindIDs_AtAccAndEarlier == -1)
        mpbSubfindIDs_AtAccAndEarlier[ww] = -1

        # debug verify:
        assert trParSubfindIDs_AtAccAndEarlier.size == mpbSubfindIDs_AtAccAndEarlier.size
        mpb_SnapVerify = mpb["SnapNum"][mpbInds_AtMatchingSnapNums]
        ww = np.where(mpbInds_AtMatchingSnapNums >= 0)  # mpb tracked only)
        assert np.array_equal(mpb_SnapVerify[ww], trParAtAccAndEarlier_HaveAtSnapNums[ww])

        # agreement of MPB subfind IDs and tracer parent subhalo IDs at all z>=z_acc
        if np.array_equal(trParSubfindIDs_AtAccAndEarlier, mpbSubfindIDs_AtAccAndEarlier):
            accMode[i] = ACCMODE_SMOOTH
            continue

        # stripped? by definition, if we make it here we have:
        #   mpbSubfindID_AtAcc == trParSubfindID_AtAcc
        #   trParSubfindIDs_AtAccAndEarlier != mpbSubfindIDs_AtAccAndEarlier
        accMode[i] = ACCMODE_STRIPPED

    # stats
    nBad = np.count_nonzero(accMode == 0)
    nNone = np.count_nonzero(accMode == ACCMODE_NONE)
    nSmooth = np.count_nonzero(accMode == ACCMODE_SMOOTH)
    nMerger = np.count_nonzero(accMode == ACCMODE_MERGER)
    nStrip = np.count_nonzero(accMode == ACCMODE_STRIPPED)

    assert nBad == 0
    nD = len(str(accMode.size))

    print(" Smooth:   [ %*d of %*d ] %4.1f%%" % (nD, nSmooth, nD, accMode.size, (100.0 * nSmooth / accMode.size)))
    print(" Merger:   [ %*d of %*d ] %4.1f%%" % (nD, nMerger, nD, accMode.size, (100.0 * nMerger / accMode.size)))
    print(" Stripped: [ %*d of %*d ] %4.1f%%" % (nD, nStrip, nD, accMode.size, (100.0 * nStrip / accMode.size)))
    print(" None:     [ %*d of %*d ] %4.1f%%" % (nD, nNone, nD, accMode.size, (100.0 * nNone / accMode.size)))

    # save
    with h5py.File(saveFilename, "w") as f:
        f["accMode"] = accMode

        if "indRange" in data:  # save pSplit portion
            f["pSplit"] = pSplit
            f["indRange"] = data["indRange"]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])
    return accMode


def valReduction(sP, fieldName, snapStep=1, op="max"):
    """Calculate a reduction operation (e.g. min, max, mean) for every tracer for a given property (e.g. temp).

    For [gas] values affected by the modified Utherm of
    the star-forming eEOS (temp, entr) we exclude times when SFR>0. This is then also consistent
    with what is done in the code for tracer_max* recorded values.
    """
    assert op in allowedExtTypes
    assert isinstance(fieldName, str)

    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + "/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_%s_%s.hdf5" % (
            sP.zoomSubhaloID,
            True,
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
            fieldName,
            op,
        )
    else:
        saveFilename = sP.derivPath + "/trTimeEvo/%s_%s_snap_%d-%s-%d.hdf5" % (
            fieldName,
            op,
            sP.snap,
            sP.redshiftToSnapNum(maxRedshift),
            snapStep,
        )

    # load pre-existing
    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # calculate new: load required field
    print("Calculating new valReduction [%s] for [%s]..." % (fieldName, sP.simName))

    data = tracersTimeEvo(sP, fieldName, snapStep, all=True)

    # mask sfr>0 points (for gas cell properties which are modified by eEOS)
    if fieldName in ["temp", "entr"]:
        sfr = tracersTimeEvo(sP, "sfr", snapStep, all=True)

        with np.errstate(invalid="ignore"):  # ignore nan comparison RuntimeWarning
            ww = np.where(sfr["sfr"] > 0.0)
        data[fieldName][ww] = np.nan

    # mask t>t_acc_015rvir points (only take extremum for time "before" first 0.15rvir crossing)
    if "_b015" in op:
        acc_time = accTime(sP, snapStep=snapStep, rVirFac=0.15)
        inds = redshiftsToClosestSnaps(data, acc_time, indsNotSnaps=True)

        for i in np.arange(inds.size):
            if inds[i] == -1:
                continue
            data[fieldName][i, : inds[i]] = np.nan

    # which functions to use
    if "min" in op:
        fval = np.nanmin
        fargval = np.nanargmin
    if "max" in op:
        fval = np.nanmax
        fargval = np.nanargmax

    # calculate extremum value
    r = {}
    r["val"] = fval(data[fieldName], axis=1)

    # calculate the redshift when it occured
    r["time"] = np.zeros(data[fieldName].shape[0], dtype="float32")
    r["time"].fill(np.nan)

    ww = np.where(~np.isnan(r["val"]))

    extInd = fargval(data[fieldName][ww, :], axis=1)
    r["time"][ww] = data["redshifts"][extInd]

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r.keys():
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])
    return r


def trValsAtRedshifts(sP, valName, redshifts, snapStep=1):
    """Return some property from the tracer evolution tracks (e.g. rad_rvir, temp) at specific redshifts.

    Given by valName at the times in the simulation given by redshifts. Extracted from closest
    snapshot, no interpolation of property value.
    """
    # load
    assert isinstance(valName, str)
    data = tracersTimeEvo(sP, valName, snapStep, all=False)

    assert data[valName].ndim == 2  # need to verify logic herein for ndim==3 (e.g. pos/vel) case

    # map times to data indices
    if isinstance(redshifts, (float)):
        # replicate into array
        redshifts = redshifts * np.ones(data[valName].shape[0], dtype="float32")

    inds = redshiftsToClosestSnaps(data, redshifts, indsNotSnaps=True)

    assert inds.max() < data[valName].shape[1]
    assert inds.size == data[valName].shape[0]

    # inds gives, for each tracer (first dimension of data[valName]), the index into the second
    # dimension of data[valName] that we want to extract. convert this implicit pair into 1d inds
    inds_dim1 = np.arange(inds.size)
    inds_1d = np.ravel_multi_index((inds_dim1, inds), data[valName].shape, mode="clip")

    # make a view to the contiguous flattened/1d array
    data_1d = np.ravel(data[valName])

    # pull out values and flag those which were always invalid as nan
    trVals = data_1d[inds_1d]

    ww = np.where(inds == -1)

    if trVals.dtype == "float32":
        trVals[ww] = np.nan
    if trVals.dtype == "int32":
        trVals[ww] = -1
    assert trVals.dtype == "float32" or trVals.dtype == "int32"

    if 1:  # stochastic debug verify of subset
        np.random.seed(42424242)
        for i in np.random.randint(0, trVals.size, 100):
            if np.isnan(trVals[i]):
                continue
            assert trVals[i] == data[valName][inds[i], i]

    return trVals


def trValsAtReductionTimes(sP, valName, redName, op="max", snapStep=1):
    """Wrap trValsAtRedshifts() to give trVals at the redshifts of these extremum (i.e. max) times."""
    red = valReduction(sP, redName, snapStep=snapStep, op=op)
    return trValsAtRedshifts(sP, valName, red["time"], snapStep=snapStep)


def trValsAtAccTimes(sP, valName, rVirFac=1.0, snapStep=1):
    """Wrap trValsAtRedshifts() to give trVals at the redshifts of the tracer halo accretion times."""
    acc_time = accTime(sP, snapStep=snapStep, rVirFac=rVirFac)
    return trValsAtRedshifts(sP, valName, acc_time, snapStep=snapStep)


def mpbValsAtRedshifts(sP, valName, redshifts, snapStep=1):
    """Return some halo property, per tracer, from the main progenitor branch (MPB) (e.g. tvir, spin).

    Given by valName at the times in the simulation given by redshifts.
    """
    # load
    assert sP.isZoom  # todo for boxes (handle sP.subhaloInd and sP.haloInd as well)
    assert isinstance(valName, str)

    # TODO: maybe whole function is divergent for sP.isZoom or not, split
    if sP.isZoom:
        mpb = sP.quantMPB(sP.zoomSubhaloID, valName, smooth=True)
    else:
        mpbGlobal = globalTracerMPBMap(sP, halos=True, retMPBs=True, extraFields=valName)

    data = {}

    # (only needed for periodic boxes with multiple MPBs)
    if not sP.isZoom:
        # TODO: loop over all tracers, for each get the respective MPB, and save the valName at the requested
        # redshifts[i] or redshifts (if constant)
        if len(redshifts) > 1:
            assert redshifts.size == mpbGlobal["subhalo_id"]

        # tracer is in FoF halo with no primary subhalo at sP.snap (no MPB)

        # extract MPB used for this tracer

        # create new mapping into MPB for this tracer

        assert 0  # todo: finish

    # pull out field straight from trees
    data["val"] = mpb[valName]

    assert data["val"].shape[0] == mpb["Redshift"].shape[0]
    assert data["val"].shape[0] == mpb["SnapNum"].shape[0]

    # map times to snapshot numbers
    data["redshifts"] = mpb["Redshift"]
    data["snaps"] = mpb["SnapNum"]

    inds = redshiftsToClosestSnaps(data, redshifts, indsNotSnaps=True)

    if data["val"].ndim == 1:
        return data["val"][inds]
    if data["val"].ndim == 2:
        return data["val"][inds, :]

    raise Exception("Should not reach here.")


def mpbValsAtExtremumTimes(sP, valName, extName, op="max", snapStep=1):
    """Wrap mpbValsAtRedshifts() to give mpbVals at the redshifts of these extremum (i.e. max) times."""
    ext = valReduction(sP, extName, snapStep=snapStep, op=op)
    return mpbValsAtRedshifts(sP, valName, ext["time"], snapStep=snapStep)


def mpbValsAtAccTimes(sP, valName, rVirFac=1.0, snapStep=1):
    """Wrap mpbValsAtRedshifts() to give mpbVals at the redshifts of the tracer halo accretion times."""
    acc_time = accTime(sP, snapStep=snapStep, rVirFac=rVirFac)
    return mpbValsAtRedshifts(sP, valName, acc_time, snapStep=snapStep)
