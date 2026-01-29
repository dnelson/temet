"""
Creation of 'virtual (uniform) boxes' from a number of zoom resimulations.
"""

from os import mkdir
from os.path import isdir, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numba import jit

from ..cosmo.zooms import _halo_ids_run
from ..tracer.montecarlo import globalTracerChildren, globalTracerLength
from ..util.helper import closest
from ..util.match import match
from ..util.simParams import simParams


@jit(nopython=True, nogil=True, cache=True)
def _mark_mask(mask, pxsize, pos, value):
    """Helper."""
    for i in range(pos.shape[0]):
        ix = int(np.floor(pos[i, 0] / pxsize))
        iy = int(np.floor(pos[i, 1] / pxsize))
        iz = int(np.floor(pos[i, 2] / pxsize))

        cur_val = mask[ix, iy, iz]
        if cur_val != -1 and cur_val != value:
            print(cur_val, value)
        assert cur_val == -1 or cur_val == value

        mask[ix, iy, iz] = value


@jit(nopython=True, nogil=True, cache=True)
def _volumes_from_mask(mask):
    """Compute the 'volume' (number of grid cells) per value in the mask.

    Return volumes[i] is the volume of halo ID i. The last entry is the unoccupied space.
    """
    maxval = np.max(mask)

    volumes = np.zeros(maxval + 2, dtype=np.int32)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                val = mask[i, j, k]
                volumes[val] += 1

    return volumes


def maskBoxRegionsByHalo():
    """Compute spatial volume fractions of zoom runs via discrete convex hull type approach."""
    # zoom config
    res = 13
    variant = "sf3"
    run = "tng_zoom"

    # hInds = [0,10,120,13,877,901,1041] # testing
    hInds = _halo_ids_run()[0:50]  # 184 for all complete until now

    snap = 99

    # mask config
    nGrid = 1024  # 2048

    sP = simParams(run=run, res=res, snap=snap, hInd=hInds[0], variant=variant)
    gridSize = sP.boxSize / nGrid  # linear, code units
    gridVol = sP.units.codeLengthToMpc(gridSize) ** 3  # volume, pMpc^3
    volumeTot = nGrid**3 * gridVol  # equals sP.units.codeLengthToMpc(sP.boxSize)**3

    # allocate mask
    mask = np.zeros((nGrid, nGrid, nGrid), dtype="int16") - 1

    numHalosTot = 0
    numSubhalosTot = 0

    numHalos14nocontam = 0
    numHalos12nocontam = 0

    # test
    for hInd in hInds:
        # load
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
        halos = sP.halos(["GroupPos", "GroupLenType", "Group_M_Crit200"])
        halo_contam = halos["GroupLenType"][:, 2] / halos["GroupLenType"][:, 1]
        halo_m200 = sP.units.codeMassToLogMsun(halos["Group_M_Crit200"])

        # offset
        halos["GroupPos"] -= sP.boxSize / 2
        halos["GroupPos"] += sP.zoomShiftPhys

        print(sP.simName, sP.numHalos, sP.numSubhalos, halos["GroupPos"][0, :])

        _mark_mask(mask, gridSize, halos["GroupPos"], hInd)

        # diagnostics
        numHalosTot += sP.numHalos
        numSubhalosTot += sP.numSubhalos

        with np.errstate(invalid="ignore"):
            w14 = np.where((halo_contam == 0) & (halo_m200 >= 14.0))
            w12 = np.where((halo_contam == 0) & (halo_m200 >= 12.0))

        numHalos14nocontam += len(w14[0])
        numHalos12nocontam += len(w12[0])

    print("\nTotal: targeted halos = %d, nHalos = %d, nSubhalos = %d" % (len(hInds), numHalosTot, numSubhalosTot))
    print("Number of halos w/o contam: [%d] above 14.0, [%d] above 12.0" % (numHalos14nocontam, numHalos12nocontam))

    # compute volume occupied by each halo
    totOccupiedVolFrac = 0.0
    volumes = _volumes_from_mask(mask) * gridVol

    for hInd in hInds + [-1]:
        frac = volumes[hInd] / volumeTot * 100
        if hInd != -1:
            totOccupiedVolFrac += frac
        print("[%4d] vol = [%8.1f pMpc^3], frac of total = [%8.6f%%]" % (hInd, volumes[hInd], frac))

    assert np.abs(totOccupiedVolFrac - (100 - frac)) < 1e-6  # total occupied should equal 1-(total unoccupied)


def combineZoomRunsIntoVirtualParentBox(snap=99):
    """Combine a set of individual zoom simulations into a 'virtual' parent simulation.

    We concatenate the output/group* and output/snap* of these runs, and process a single snapshot,
    since all are independent. Note that we write exactly one  output groupcat file per zoom halo,
    and exactly two output snapshot files.
    """
    outPath = "/u/dnelson/sims.TNG/L680n8192TNG/output/"
    parent_sim = simParams("tng-cluster")

    # zoom config
    res = 14
    variant = "sf3"
    run = "tng_zoom"

    hInds = _halo_ids_run(res=res, onlyDone=True)

    def _newpartid(old_ids, halo_ind, ptNum):
        """Define convention to offset particle/cell/tracer IDs based on zoom run halo ID.

        No zoom halo has more than 100M (halo 0 has ~85M) of any type. This requires conversion
        to LONGIDS by definition.
        """
        new_ids = old_ids.astype("uint64")

        # shift to start at 1 instead of 1000000000 (for IC-split/spawned types)
        if ptNum in [0, 3, 4, 5]:
            new_ids -= 1000000000 - 1

        # offset (increase) by hInd*1e9
        new_ids += halo_ind * 1000000000
        return new_ids

    # --- tracers ---
    if snap == 99:
        # tracers: is final snapshot? then decide tracer ordering and save for use on all snapshots
        GroupLenTypeTracers = np.zeros(len(hInds), dtype="int32")

        for i, hInd in enumerate(hInds):
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
            print("[%4d] z=0 tracers." % hInd)

            # cache
            saveFilename = outPath + "tracers_%d.hdf5" % hInd

            if isfile(saveFilename):
                with h5py.File(saveFilename, "r") as f:
                    for key in f["TracerLength_Halo"]:
                        GroupLenTypeTracers[i] += np.sum(f["TracerLength_Halo"][key][()])

                print(" skip")
                continue

            # get child tracers of all particle types in all FoFs
            # (ordered first by parent type: gas->stars->BHs, then by halo/subhalo membership)
            trIDs = globalTracerChildren(sP, halos=True)

            # TracerLength and TracerOffset by halo and subhalo
            trCounts_halo, trOffsets_halo = globalTracerLength(sP, halos=True)

            trCounts_sub, trOffsets_sub = globalTracerLength(sP, subhalos=True, haloTracerOffsets=trOffsets_halo)

            # load all TracerIDs, get those not in FoFs
            TracerID = sP.snapshotSubsetP("tracer", "TracerID")

            trInds, _ = match(TracerID, trIDs)
            assert trInds.size == trIDs.size

            mask = np.zeros(TracerID.size, dtype="int8")
            mask[trInds] = 1

            trInds_outside_halos = np.where(mask == 0)[0]

            # trIDs_outside_halos = TracerID[trInds_outside_halos]

            # make final, z=0 ordered, list of tracerIDs
            trInds_final = np.hstack((trInds, trInds_outside_halos))

            if 1:
                # debug verify
                mask2 = np.zeros(TracerID.size, dtype="int16")
                mask2[trInds_final] += 1
                assert mask2.min() == 1
                assert mask2.max() == 1

            TracerID = TracerID[trInds_final]

            # save
            with h5py.File(saveFilename, "w") as f:
                # lengths and offsets
                for key in trCounts_halo.keys():
                    f["TracerLength_Halo/%s" % key] = trCounts_halo[key]
                    f["TracerOffset_Halo/%s" % key] = trOffsets_halo[key]

                    f["TracerLength_Subhalo/%s" % key] = trCounts_sub[key]
                    f["TracerOffset_Subhalo/%s" % key] = trOffsets_sub[key]

                # z=0 ordered TracerIDs
                f["TracerID"] = TracerID

            # save total length of tracers in FoF halos, for each hInd (sum over all parent types)
            for key in trCounts_halo.keys():
                GroupLenTypeTracers[i] += np.sum(trCounts_halo[key])

        with h5py.File(outPath + "tracers_halolengths.hdf5", "w") as f:
            f["GroupLenTypeTracers"] = GroupLenTypeTracers

    with h5py.File(outPath + "tracers_halolengths.hdf5", "r") as f:
        GroupLenTypeTracers = f["GroupLenTypeTracers"][()]

    # --- groupcat ---

    savePath = outPath + "groups_%03d/" % snap
    if not isdir(savePath):
        mkdir(savePath)

    # load total number of halos and subhalos
    lengths = {"Group": np.zeros(len(hInds), dtype="int32"), "Subhalo": np.zeros(len(hInds), dtype="int32")}

    for i, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        lengths["Group"][i] = sP.numHalos
        lengths["Subhalo"][i] = sP.numSubhalos

        # verify
        if lengths["Subhalo"][i]:
            GroupNsubs = sP.groups("GroupNsubs")
            assert lengths["Subhalo"][i] == GroupNsubs.sum()  # h604 fails snap==53

    numHalosTot = np.sum(lengths["Group"], dtype="int32")
    numSubhalosTot = np.sum(lengths["Subhalo"], dtype="int32")

    print("\nSnapshot = [%2d], total [%d] halos, [%d] subhalos." % (snap, numHalosTot, numSubhalosTot))

    GroupLenType_hInd = np.zeros((len(hInds), 6), dtype="int32")

    offsets = {}
    offsets["Group"] = np.hstack((0, np.cumsum(lengths["Group"], dtype="int64")[:-1]))
    offsets["Subhalo"] = np.hstack((0, np.cumsum(lengths["Subhalo"], dtype="int64")[:-1]))
    offsets["Tracers"] = np.hstack((0, np.cumsum(GroupLenTypeTracers, dtype="int64")[:-1]))

    numFiles = sP.groupCatHeader()["NumFiles"]
    print("\nCombine [%d] zooms, re-writing group catalogs:" % len(hInds))

    # use first zoom run: load header-type groups
    headers = {}
    sP = simParams(run=run, res=res, snap=snap, hInd=hInds[0], variant=variant)

    with h5py.File(sP.gcPath(sP.snap, 0), "r") as f:
        for gName in ["Config", "Header", "Parameters"]:
            headers[gName] = dict(f[gName].attrs.items())

    # header adjustments
    fac = 1000.0

    if headers["Header"]["Redshift"] < 1e-10:
        assert headers["Header"]["Redshift"] > 0
        headers["Header"]["Redshift"] = 0.0
        headers["Header"]["Time"] = 1.0

    headers["Header"]["BoxSize"] *= fac  # Mpc -> kpc units
    headers["Header"]["Nids_ThisFile"] = 0  # always unused
    headers["Header"]["Nids_Total"] = 0  # always unused

    headers["Header"]["Ngroups_Total"] = numHalosTot
    headers["Header"]["Nsubgroups_Total"] = numSubhalosTot
    headers["Header"]["NumFiles"] = np.int32(len(hInds))

    headers["Parameters"]["BoxSize"] *= fac  # Mpc -> kpc units
    headers["Parameters"]["InitCondFile"] = "various"
    headers["Parameters"]["NumFilesPerSnapshot"] = np.int32(len(hInds) * 2)
    headers["Parameters"]["UnitLength_in_cm"] /= fac  # kpc units
    headers["Config"]["LONGIDS"] = ""  # True

    # loop over all zoom runs: load full group cats
    for hCount, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # loop over original files, load
        data = {"Group": {}, "Subhalo": {}}
        offsets_loc = {"Group": 0, "Subhalo": 0}

        for i in range(numFiles):
            with h5py.File(sP.gcPath(sP.snap, i), "r") as f:
                # loop over groups with datasets
                for gName in data.keys():
                    if len(f[gName]) == 0:
                        continue

                    start = offsets_loc[gName]
                    length = f[gName][list(f[gName].keys())[0]].shape[0]

                    for field in f[gName]:
                        if field in ["SubhaloBfldDisk", "SubhaloBfldHalo"]:
                            continue  # do not save (not fixed)

                        if i == 0:
                            # allocate
                            shape = list(f[gName][field].shape)
                            shape[0] = lengths[gName][hCount]  # override chunk length with global
                            data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)

                        # read chunk
                        data[gName][field][start : start + length] = f[gName][field][()]

                    offsets_loc[gName] += length

        if lengths["Group"][hCount]:
            # allocate fields to save originating zoom run IDs
            data["Group"]["GroupOrigHaloID"] = np.zeros(lengths["Group"][hCount], dtype="int32")

            # allocate new meta-data fields
            data["Group"]["GroupPrimaryZoomTarget"] = np.zeros(lengths["Group"][hCount], dtype="int32")
            data["Group"]["GroupContaminationFracByMass"] = np.zeros(lengths["Group"][hCount], dtype="float32")
            data["Group"]["GroupContaminationFracByNumPart"] = np.zeros(lengths["Group"][hCount], dtype="float32")

            w = np.where(data["Group"]["GroupMassType"][:, 1] > 0)
            data["Group"]["GroupContaminationFracByMass"][w] = data["Group"]["GroupMassType"][w, 2] / (
                data["Group"]["GroupMassType"][w, 1] + data["Group"]["GroupMassType"][w, 2]
            )

            w = np.where(data["Group"]["GroupLenType"][:, 1] > 0)
            data["Group"]["GroupContaminationFracByNumPart"][w] = data["Group"]["GroupLenType"][w, 2] / (
                data["Group"]["GroupLenType"][w, 1] + data["Group"]["GroupLenType"][w, 2]
            )

            w = np.where((data["Group"]["GroupLenType"][:, 2] > 0) & (data["Group"]["GroupLenType"][:, 2] == 0))
            data["Group"]["GroupContaminationFracByMass"][w] = 1.0
            data["Group"]["GroupContaminationFracByNumPart"][w] = 1.0

            # save originating zoom run halo ID
            data["Group"]["GroupOrigHaloID"][:] = hInd
            data["Group"]["GroupPrimaryZoomTarget"][0] = 1

            # make index adjustments
            w = np.where(data["Group"]["GroupFirstSub"] != -1)
            data["Group"]["GroupFirstSub"][w] += offsets["Subhalo"][hCount]

            # spatial offset adjustments: un-shift zoom center and periodic shift
            for field in ["GroupCM", "GroupPos"]:
                data["Group"][field] -= sP.boxSize / 2
                data["Group"][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data["Group"][field])

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in [
                "Group_R_Crit200",
                "Group_R_Crit500",
                "Group_R_Mean200",
                "Group_R_TopHat200",
                "GroupCM",
                "GroupPos",
            ]:
                data["Group"][field] *= fac

            data["Group"]["GroupBHMdot"] *= fac ** (-1)  # UnitLength^-1

            # record fof-scope lengths by type
            GroupLenType_hInd[hCount, :] = np.sum(data["Group"]["GroupLenType"], axis=0)

            # tracers: add Tracer{Length,Offset}Type at z=0
            if snap == 99:
                data["Group"]["TracerLengthType"] = np.zeros((lengths["Group"][hCount], 6), dtype="int32")
                data["Group"]["TracerOffsetType"] = np.zeros((lengths["Group"][hCount], 6), dtype="int64")

                with h5py.File(outPath + "tracers_%d.hdf5" % hInd, "r") as f:
                    for key in f["TracerLength_Halo"].keys():
                        data["Group"]["TracerLengthType"][:, sP.ptNum(key)] = f["TracerLength_Halo"][key][()]
                        data["Group"]["TracerOffsetType"][:, sP.ptNum(key)] = (
                            f["TracerOffset_Halo"][key][()] + offsets["Tracers"][hCount]
                        )

        if lengths["Subhalo"][hCount]:
            data["Subhalo"]["SubhaloOrigHaloID"] = np.zeros(lengths["Subhalo"][hCount], dtype="int32")

            data["Subhalo"]["SubhaloOrigHaloID"][:] = hInd

            # make index adjustments
            data["Subhalo"]["SubhaloGrNr"] += offsets["Group"][hCount]

            # SubhaloIDMostbound could be any type, identify any of PT1/PT2 (with low IDs) and offset those
            # such that they are unchanged after _newpartid(ptNum=0)
            w = np.where(data["Subhalo"]["SubhaloIDMostbound"] < 1000000000)
            data["Subhalo"]["SubhaloIDMostbound"][w] += 1000000000 - 1

            data["Subhalo"]["SubhaloIDMostbound"] = _newpartid(data["Subhalo"]["SubhaloIDMostbound"], hInd, ptNum=0)

            # spatial offset adjustments: un-shift zoom center and periodic shift
            for field in ["SubhaloCM", "SubhaloPos"]:
                data["Subhalo"][field] -= sP.boxSize / 2
                data["Subhalo"][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data["Subhalo"][field])

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in [
                "SubhaloHalfmassRad",
                "SubhaloHalfmassRadType",
                "SubhaloSpin",
                "SubhaloStellarPhotometricsRad",
                "SubhaloVmaxRad",
                "SubhaloCM",
                "SubhaloPos",
            ]:
                data["Subhalo"][field] *= fac

            data["Subhalo"]["SubhaloBHMdot"] *= fac ** (-1)

            for field in ["SubhaloBfldDisk", "SubhaloBfldHalo"]:
                if field in data["Subhalo"]:
                    data["Subhalo"][field] *= fac ** (-1.5)  # UnitLength^-1.5

            # tracers: add Tracer{Length,Offset}Type at z=0
            if snap == 99:
                data["Subhalo"]["TracerLengthType"] = np.zeros((lengths["Subhalo"][hCount], 6), dtype="int32")
                data["Subhalo"]["TracerOffsetType"] = np.zeros((lengths["Subhalo"][hCount], 6), dtype="int64")

                with h5py.File(outPath + "tracers_%d.hdf5" % hInd, "r") as f:
                    for key in f["TracerLength_Subhalo"].keys():
                        data["Subhalo"]["TracerLengthType"][:, sP.ptNum(key)] = f["TracerLength_Subhalo"][key][()]
                        data["Subhalo"]["TracerOffsetType"][:, sP.ptNum(key)] = (
                            f["TracerOffset_Subhalo"][key][()] + offsets["Tracers"][hCount]
                        )

        # per-halo header adjustments
        headers["Header"]["Ngroups_ThisFile"] = lengths["Group"][hCount]
        headers["Header"]["Nsubgroups_ThisFile"] = lengths["Subhalo"][hCount]

        # write this zoom halo into single file
        outFile = "fof_subhalo_tab_%03d.%d.hdf5" % (snap, hCount)

        with h5py.File(savePath + outFile, "w") as f:
            # add header groups
            for gName in headers:
                f.create_group(gName)
                for at in headers[gName]:
                    f[gName].attrs[at] = headers[gName][at]

            # add datasets
            for gName in data:
                f.create_group(gName)
                for field in data[gName]:
                    f[gName][field] = data[gName][field]

        print(
            " [%3d] Wrote [%s] (hInd = %4d) (offsets = %8d %8d)"
            % (hCount, outFile, hInd, offsets["Group"][hCount], offsets["Subhalo"][hCount])
        )

    # --- snapshot ---

    savePath = outPath + "snapdir_%03d/" % snap
    if not isdir(savePath):
        mkdir(savePath)

    print("\nCombine [%d] zooms, re-writing snapshots:" % len(hInds))

    # load total number of particles
    NumPart_Total = np.zeros((len(hInds), 6), dtype="int64")

    for i, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # load snapshot header from first run (reuse Parameters and Config from groupcats)
        with h5py.File(sP.snapPath(sP.snap, 0), "r") as f:
            Header_snap = dict(f["Header"].attrs.items())

        if i == 0:
            headers["Header"] = Header_snap
        else:
            assert np.allclose(headers["Header"]["MassTable"], Header_snap["MassTable"])

        # count particles
        assert np.sum(Header_snap["NumPart_Total_HighWord"]) == 0

        NumPart_Total[i, :] = Header_snap["NumPart_Total"]

    NumPart_Total_Global = np.sum(NumPart_Total, axis=0)

    # load total number of tracers in halos and overwrite GroupLenType[3]
    assert np.sum(GroupLenType_hInd[:, 3]) == 0
    assert GroupLenTypeTracers.size == GroupLenType_hInd.shape[0]

    GroupLenType_hInd[:, 3] = GroupLenTypeTracers

    # determine sizes/split between two files per halo
    OuterFuzzLenType_hInd = NumPart_Total - GroupLenType_hInd

    # quick save of offsets
    saveFilename = "lengths_hind_%03d.hdf5" % snap
    with h5py.File(outPath + saveFilename, "w") as f:
        # particle lengths in all fofs for this hInd (file 1)
        f["GroupLenType_hInd"] = GroupLenType_hInd
        # particle lengths outside fofs for this hInd (file 2)
        f["OuterFuzzLenType_hInd"] = OuterFuzzLenType_hInd
        # halo IDs of original zooms
        f["HaloIDs"] = np.array(hInds, dtype="int32")

    print(" Saved [%s%s]." % (outPath, saveFilename))

    # header adjustments
    if headers["Header"]["Redshift"] < 1e-10:
        assert headers["Header"]["Redshift"] > 0
        headers["Header"]["Redshift"] = 0.0
        headers["Header"]["Time"] = 1.0

    headers["Header"]["BoxSize"] *= fac  # Mpc -> kpc units
    headers["Header"]["NumFilesPerSnapshot"] = np.int32(len(hInds) * 2)
    headers["Header"]["UnitLength_in_cm"] /= fac  # kpc units

    headers["Header"]["NumPart_Total"] = np.uint32(NumPart_Total_Global & 0xFFFFFFFF)  # first 32 bits
    headers["Header"]["NumPart_Total_HighWord"] = np.uint32(NumPart_Total_Global >> 32)

    # loop over all zoom runs: load full group cats
    for hCount, hInd in enumerate(hInds):
        sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)

        # loop over original files, load
        data = {}
        attr = {}
        offsets_loc = {}

        for pt in [0, 1, 2, 3, 4, 5]:
            data["PartType%d" % pt] = {}
            attr["PartType%d" % pt] = {}
            offsets_loc["PartType%d" % pt] = 0

        for i in range(numFiles):
            with h5py.File(sP.snapPath(sP.snap, i), "r") as f:
                # loop over groups with datasets
                for gName in data.keys():
                    if gName not in f or len(f[gName]) == 0:
                        continue

                    start = offsets_loc[gName]
                    length = f[gName][list(f[gName].keys())[0]].shape[0]
                    length_global = f["Header"].attrs["NumPart_Total"][int(gName[-1])]

                    for field in f[gName]:
                        if field in ["TimeStep", "TimebinHydro", "HighResGasMass"]:
                            continue  # do not save

                        if gName == "PartType2" and field in ["SubfindDMDensity", "SubfindDensity", "SubfindVelDisp"]:
                            continue  # all zero for low-res DM, do not save

                        if field not in data[gName]:
                            # allocate (could be i>0)
                            shape = list(f[gName][field].shape)
                            shape[0] = length_global  # override chunk length with global (for this hInd)
                            data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)
                            attr[gName][field] = dict(f[gName][field].attrs)

                        # read chunk
                        data[gName][field][start : start + length] = f[gName][field][()]

                    offsets_loc[gName] += length

        # loop over all particle types to apply corrections
        for gName in data.keys():
            # spatial offset adjustments: un-shift zoom center and periodic shift
            ptNum = int(gName[-1])

            for field in ["CenterOfMass", "Coordinates", "BirthPos"]:
                if field not in data[gName]:
                    continue

                data[gName][field] -= sP.boxSize / 2
                data[gName][field] += sP.zoomShiftPhys
                sP.correctPeriodicPosVecs(data[gName][field])

            # reorder tracers into z=0 order
            if gName == "PartType3":
                with h5py.File(outPath + "tracers_%d.hdf5" % hInd, "r") as f:
                    # z=0 ordered TracerIDs
                    TracerID_z0 = f["TracerID"][()]

                inds_snap, inds_z0 = match(data[gName]["TracerID"], TracerID_z0)
                assert data[gName]["TracerID"].size == TracerID_z0.size
                assert inds_z0.size == TracerID_z0.size

                for trField in data[gName].keys():
                    data[gName][trField] = data[gName][trField][inds_snap]

            # make ID index adjustments
            if "ParticleIDs" in data[gName]:
                data[gName]["ParticleIDs"] = _newpartid(data[gName]["ParticleIDs"], hInd, ptNum)

            if gName == "PartType3":
                data[gName]["TracerID"] = _newpartid(data[gName]["TracerID"], hInd, ptNum)
                data[gName]["ParentID"] = _newpartid(data[gName]["ParentID"], hInd, ptNum)

            # spatial offset adjustments: unit system (Mpc -> kpc)
            for field in ["CenterOfMass", "Coordinates", "BirthPos", "SubfindHsml", "BH_Hsml"]:
                if field in data[gName]:  # UnitLength
                    data[gName][field] *= fac
                    attr[gName][field]["to_cgs"] /= fac

            for field in ["Density", "SubfindDensity", "SubfindDMDensity", "BH_Density"]:
                if field in data[gName]:  # UnitLength^-3
                    data[gName][field] *= fac ** (-3)
                    attr[gName][field]["to_cgs"] /= fac ** (-3)

        if "MagneticField" in data["PartType0"]:
            # unit meta-data missing in TNG codebase, add now
            data["PartType0"]["MagneticField"] *= fac ** (-1.5)  # UnitLength^-1.5
            attr["PartType0"]["MagneticField"]["a_scaling"] = -2.0
            attr["PartType0"]["MagneticField"]["h_scaling"] = 1.0
            attr["PartType0"]["MagneticField"]["length_scaling"] = -1.5
            attr["PartType0"]["MagneticField"]["mass_scaling"] = 0.5
            attr["PartType0"]["MagneticField"]["to_cgs"] = 2.60191e-06
            attr["PartType0"]["MagneticField"]["velocity_scaling"] = 1.0

            data["PartType0"]["MagneticFieldDivergence"] *= fac ** (-2.5)  # UnitLength^-2.5
            attr["PartType0"]["MagneticField"]["a_scaling"] = -3.0
            attr["PartType0"]["MagneticField"]["h_scaling"] = 2.0
            attr["PartType0"]["MagneticField"]["length_scaling"] = -2.5
            attr["PartType0"]["MagneticField"]["mass_scaling"] = 0.5
            attr["PartType0"]["MagneticFieldDivergence"]["to_cgs"] = 8.43220e-28
            attr["PartType0"]["MagneticField"]["velocity_scaling"] = 1.0

        if "EnergyDissipation" in data["PartType0"]:  # UnitLength^-1
            data["PartType0"]["EnergyDissipation"] *= fac ** (-1)
            # attr['PartType0']['EnergyDissipation']['to_cgs'] /= fac**(-1) # meta-data not present

        if len(data["PartType5"]):
            data["PartType5"]["BH_BPressure"] *= fac ** (-3)  # assume same as BH_Pressure (todo verify!)
            attr["PartType5"]["BH_BPressure"]["a_scaling"] = -4.0  # unit meta-data missing
            attr["PartType5"]["BH_BPressure"]["h_scaling"] = 2.0  # units are (MagneticField)^2
            attr["PartType5"]["BH_BPressure"]["length_scaling"] = -3.0
            attr["PartType5"]["BH_BPressure"]["mass_scaling"] = 1.0
            attr["PartType5"]["BH_BPressure"]["to_cgs"] = 6.76994e-12  # attr['PartType0']['MagneticField']['to_cgs']**2
            attr["PartType5"]["BH_BPressure"]["velocity_scaling"] = 2.0

            data["PartType5"]["BH_Pressure"] *= fac ** (-3)  # Pressure = UnitLength^-3 (assuming wrong in io_fields.c)
            attr["PartType5"]["BH_Pressure"]["to_cgs"] /= fac ** (-3)

            for field in ["BH_CumEgyInjection_RM", "BH_CumEgyInjection_QM"]:
                data["PartType5"][field] *= 1  # UnitLength^0
                attr["PartType5"][field]["to_cgs"] *= 1

            for field in ["BH_Mdot", "BH_MdotBondi", "BH_MdotEddington"]:  # UnitMass/UnitTime = UnitLength^-1
                data["PartType5"][field] *= fac ** (-1)  # (assuming wrong in io_fields.c)
                attr["PartType5"][field]["to_cgs"] /= fac ** (-1)

        # write this zoom halo into two files: one for fof-particles, one for outside-fof-particles
        for fileNum in [0, 1]:
            # per-halo header adjustments
            if fileNum == 0:
                headers["Header"]["NumPart_ThisFile"] = np.int32(GroupLenType_hInd[hCount, :])
                start = np.zeros(6, dtype="int32")
            else:
                headers["Header"]["NumPart_ThisFile"] = np.int32(OuterFuzzLenType_hInd[hCount, :])
                start = GroupLenType_hInd[hCount, :]

            outFile = "snap_%03d.%d.hdf5" % (snap, hCount + len(hInds) * fileNum)

            with h5py.File(savePath + outFile, "w") as f:
                # add header groups
                for gName in headers:
                    f.create_group(gName)
                    for at in headers[gName]:
                        f[gName].attrs[at] = headers[gName][at]

                # add datasets
                for gName in data:
                    ptNum = int(gName[-1])
                    length_loc = headers["Header"]["NumPart_ThisFile"][ptNum]

                    if length_loc == 0:
                        continue

                    f.create_group(gName)

                    for field in data[gName]:
                        # write
                        f[gName][field] = data[gName][field][start[ptNum] : start[ptNum] + length_loc]

                        # add unit meta-data
                        for at in attr[gName][field]:
                            f[gName][field].attrs[at] = attr[gName][field][at]

            print(
                " [%3d] hInd = %4d (gas %8d - %8d of %8d) Wrote: [%s]"
                % (
                    hCount,
                    hInd,
                    start[0],
                    start[0] + headers["Header"]["NumPart_ThisFile"][0],
                    NumPart_Total[hCount, 0],
                    outFile,
                )
            )

    # compute offsets and insert them (e.g. new/MTNG convention)
    parent_sim.snap = snap  # avoid setSnap() since our snap<->redshift mapping file is incomplete

    offsets = parent_sim.groupCatOffsetListIntoSnap()

    w_offset_halos = 0
    w_offset_subs = 0

    for hCount in range(len(hInds)):
        outFile = outPath + "groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap, snap, hCount)

        with h5py.File(outFile, "r+") as f:
            Nsubhalos = f["Header"].attrs["Nsubgroups_ThisFile"]
            Nhalos = f["Header"].attrs["Ngroups_ThisFile"]

            if Nhalos > 0:
                f["Group"]["GroupOffsetType"] = offsets["snapOffsetsGroup"][w_offset_halos : w_offset_halos + Nhalos]
            if Nsubhalos > 0:
                f["Subhalo"]["SubhaloOffsetType"] = offsets["snapOffsetsSubhalo"][
                    w_offset_subs : w_offset_subs + Nsubhalos
                ]

            w_offset_halos += Nhalos
            w_offset_subs += Nsubhalos

    print("Done.")


def testVirtualParentBoxGroupCat(snap=99):
    """Compare all group cat fields (1d histograms) vs TNG300-1 to check unit conversions, etc."""
    # config
    nBins = 50

    sP1 = simParams(run="tng-cluster", snap=snap)
    sP2 = simParams(run="tng300-1", snap=snap)

    # compare group catalogs: entire (un-contaminated) TNG-Cluster vs. ~same (first) N of TNG300-1
    contam = sP1.halos("GroupContaminationFracByMass")
    m200 = sP1.units.codeMassToLogMsun(sP1.halos("Group_M_Crit200"))
    m200_sP2 = sP2.units.codeMassToLogMsun(sP2.halos("Group_M_Crit200"))

    haloIDs_1 = np.where((contam < 0.01) & (m200 > 14.0))[0]
    haloIDs_2 = np.where(m200_sP2 > 14.0)[0]

    haloIDs = [haloIDs_1, haloIDs_2]

    # subhalos: all of these groups
    subIDs = []

    for i, sP in enumerate([sP1, sP2]):
        nSubs = sP.halos("GroupNsubs")[haloIDs[i]]
        firstSub = sP.halos("GroupFirstSub")[haloIDs[i]]

        subIDs_loc = np.hstack([np.arange(nSubs[i]) + firstSub[i] for i in range(haloIDs[i].size)])
        subIDs.append(subIDs_loc)

    for gName in ["Group", "Subhalo"]:
        # get list of halo/subhalo properties
        with h5py.File(sP2.gcPath(sP2.snap, 0), "r") as f:
            fields = list(f[gName].keys())

        # start pdf book
        pdf = PdfPages("compare_%s_%s_%s_%d.pdf" % (gName, sP1.simName, sP2.simName, snap))

        for field in fields:
            # start plot
            print(field)
            if field in ["SubhaloFlag", "SubhaloBfldDisk", "SubhaloBfldHalo"]:
                continue

            fig, ax = plt.subplots()

            ax.set_xlabel(field + " [log]")
            ax.set_ylabel("log N")

            # load and histogram
            for i, sP in enumerate([sP1, sP2]):
                if gName == "Group":
                    vals = sP.halos(field)
                    vals = vals[haloIDs[i]]

                if gName == "Subhalo":
                    vals = sP.subhalos(field)
                    vals = vals[subIDs[i]]

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel()  # 1D for all multi-D

                if field not in ["GroupCM", "GroupPos", "SubhaloCM", "SubhaloGrNr", "SubhaloIDMostbound"]:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, density=True, label=sP.simName)

            # finish plot
            ax.legend(loc="best")
            pdf.savefig()
            plt.close(fig)

        # finish
        pdf.close()


def testVirtualParentBoxSnapshot(snap=99):
    """Compare all snapshot fields (1d histograms) vs TNG300-1 to check unit conversions, etc."""
    # config
    haloID = 0  # for particle comparison, indexing primary targets of TNG-Cluster
    nBins = 50

    sP1 = simParams(run="tng-cluster", snap=snap)
    sP2 = simParams(run="tng300-1", snap=snap)

    # compare particle fields: one halo
    pri_target = sP1.halos("GroupPrimaryZoomTarget")

    sP1_hInd = np.where(pri_target)[0][haloID]
    sP1_m200 = sP1.halo(sP1_hInd)["Group_M_Crit200"]
    zoomOrigID = sP1.groupCatSingle(haloID=sP1_hInd)["GroupOrigHaloID"]

    # locate close mass in sP2 to compare to
    sP2_m200, sP2_hInd = closest(sP2.halos("Group_M_Crit200"), sP1_m200)

    haloIDs = [sP1_hInd, sP2_hInd]

    print(
        "Comparing hInd [%d (%d)] from TNG-Cluster to [%d] from TNG300-1 (%.1f vs %.1f log msun)."
        % (sP1_hInd, zoomOrigID, sP2_hInd, sP1.units.codeMassToLogMsun(sP1_m200), sP2.units.codeMassToLogMsun(sP2_m200))
    )

    if 0:
        # debugging: load one field
        pt = "dm"
        field = "ParticleIDs"

        vals1 = sP1.snapshotSubset(pt, field, haloID=sP1_hInd)
        vals2 = sP2.snapshotSubset(pt, field, haloID=sP2_hInd)

        print(sP1.simName, " min max mean: ", vals1.min(), vals1.max(), np.mean(vals1))
        print(sP2.simName, " min max mean: ", vals2.min(), vals2.max(), np.mean(vals2))

    # loop over part types
    for ptNum in [0, 1, 4, 5]:  # skip low-res DM (2) and tracers (3)
        gName = "PartType%d" % ptNum

        # get list of particle datasets
        with h5py.File(sP2.snapPath(sP2.snap, 0), "r") as f:
            fields = list(f[gName].keys())

        # start pdf book
        pdf = PdfPages("compare_%s_%s_%s_h%d_%d.pdf" % (gName, sP1.simName, sP2.simName, haloID, snap))

        for field in fields:
            # start plot
            print(gName, field)
            if field in ["InternalEnergyOld", "StellarHsml"]:
                continue

            fig, ax = plt.subplots()

            ax.set_xlabel(field + " [log]")
            ax.set_ylabel("N")
            ax.set_yscale("log")

            # load and histogram
            for i, sP in enumerate([sP1, sP2]):
                vals = sP.snapshotSubset(ptNum, field, haloID=haloIDs[i])

                if field == "ParticleIDs" and i == 0:  # verify ID spacing
                    offset = 1000000000 * zoomOrigID
                    assert (vals - offset).min() > 0 and (vals - offset).max() < 1000000000

                if field == "Potential":
                    vals *= -1

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel()  # 1D for all multi-D

                if field not in []:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, label=sP.simName)

            # finish plot
            ax.legend(loc="best")
            pdf.savefig()
            plt.close(fig)

        # finish
        pdf.close()


def check_groupcat_property():
    """Compare TNG300 vs TNG-Cluster property."""
    xprop = "Group_M_Crit200"
    yprop = "SubhaloBHMass"  #'GroupMassType', 'GroupSFR', 'GroupNsubs', 'GroupWindMass', 'GroupBHMass'
    ypropind = None

    snap = 99

    halo_inds = _halo_ids_run(onlyDone=True)
    halo_inds.pop(-1)  # remove h4

    # load
    cache = "cache_%s_%s.hdf5" % (yprop, ypropind)
    if isfile(cache):
        with h5py.File(cache, "r") as f:
            x = f["x"][()]
            y = f["y"][()]

    else:
        # compute now
        res = 13
        variant = "sf3"
        run = "tng_zoom"
        haloID = 0  # always use first fof

        # allocate
        x = np.zeros(len(halo_inds), dtype="float32")
        y = np.zeros(len(halo_inds), dtype="float32")

        # loop over halos
        for i, hInd in enumerate(halo_inds):
            sP = simParams(run=run, res=res, snap=snap, hInd=hInd, variant=variant)
            halo = sP.halo(haloID)
            subh = sP.subhalo(halo["GroupFirstSub"])

            x[i] = halo[xprop]
            if "Group" in yprop:
                y[i] = halo[yprop][ypropind] if ypropind is not None else halo[yprop]
            else:
                y[i] = subh[yprop][ypropind] if ypropind is not None else subh[yprop]

            print(i, hInd)

        with h5py.File(cache, "w") as f:
            f["x"] = x
            f["y"] = y
        print("Saved: [%s]" % cache)

    # TNG-Cluster unit conversions
    sP = simParams(run="tng-cluster")
    x = sP.units.codeMassToLogMsun(x)
    if "Mass" in yprop:
        y = sP.units.codeMassToLogMsun(y)
    else:
        y = np.log10(y)

    # plot
    fig, ax = plt.subplos()

    ax.set_xlabel(xprop)
    ax.set_ylabel(yprop + (" [" + str(ypropind) + "]" if ypropind is not None else ""))

    ax.set_xlim([14.0, 15.5])

    ax.scatter(x, y, marker="o", label="TNG-Cluster")

    for run in ["tng300-1", "tng300-2", "tng300-3"]:
        # load TNG300-x
        sP = simParams(run=run, snap=snap)

        x2 = sP.halos(xprop)

        if "Group" in yprop:
            y2 = sP.halos(yprop)[:, ypropind] if ypropind is not None else sP.halos(yprop)
        else:
            GroupFirstSub = sP.halos("GroupFirstSub")
            # y2 = np.zeros( sP.numHalos, dtype='float32' )
            # y2.fill(np.nan)
            y2 = (
                sP.subhalos(yprop)[GroupFirstSub, ypropind]
                if ypropind is not None
                else sP.subhalos(yprop)[GroupFirstSub]
            )

        x2 = sP.units.codeMassToLogMsun(x2)
        if "Mass" in yprop:
            y2 = sP.units.codeMassToLogMsun(y2)
        else:
            y2 = np.log10(y2)

        w = np.where(x2 >= 14.0)
        x2 = x2[w]
        y2 = y2[w]

        # plot
        ax.scatter(x2, y2, marker="s", label=sP.simName)

    ax.legend(loc="best")
    fig.savefig("check_%s_%s_%s.pdf" % (xprop, yprop, ypropind))
    plt.close(fig)


def check_particle_property():
    """Debug: compare TNG300 vs TNG-Cluster particle property."""
    snap = 99
    pt = "bh"
    prop = "BH_Mass"

    # load zoom
    hInd = 0
    haloID = 0  # always use first fof

    sP = simParams(run="tng_zoom", res=13, snap=snap, hInd=hInd, variant="sf3")

    x = sP.snapshotSubset(pt, prop, haloID=haloID)
    x = sP.units.codeMassToLogMsun(x)

    # load box
    sP = simParams(run="tng300-1", snap=snap)

    y = sP.snapshotSubset(pt, prop, haloID=haloID)
    y = sP.units.codeMassToLogMsun(y)

    print(10.0 ** y.mean() / 10.0 ** x.mean())
    print(10.0 ** x.mean() / 10.0 ** y.mean())

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel("%s %s" % (pt, prop))
    ax.set_ylabel("PDF")

    ax.hist(x, bins=40, label="TNG-Cluster")
    ax.hist(y, bins=40, label="TNG300")

    ax.legend(loc="best")
    fig.savefig("check_%s_%s.pdf" % (pt, prop))
    plt.close(fig)
