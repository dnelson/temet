"""
Various data exporters/converters, between different formats, etc.
"""

import csv
import glob
import struct
from io import BytesIO
from os import mkdir, path, remove
from os.path import isfile

import h5py
import numpy as np

from ..cosmo.util import multiRunMatchedSnapList, snapNumToAgeFlat, snapNumToRedshift
from ..util.helper import crossmatchHalosByCommonIDs, logZeroMin, nUnique, pSplitRange
from ..util.match import match
from ..util.simParams import simParams
from ..util.sphMap import sphGridWholeBox, sphMap


def concatSubboxFilesAndMinify():
    """Minify subbox snapshots, removing unwanted fields and resaving concatenated into a smaller number of chunks."""
    # config
    outputPath = path.expanduser("~") + "/sims.TNG/L35n2160TNG/output/"
    sbNum = 2
    sbSnapRange = [1200, 1201]
    numChunksSave = 8

    metaGroups = ["Config", "Header", "Parameters"]
    keepFields = {
        "PartType0": [
            "Coordinates",
            "Density",
            "ElectronAbundance",
            "EnergyDissipation",
            "GFM_Metallicity",
            "InternalEnergy",
            "Machnumber",
            "MagneticField",
            "Masses",
            "NeutralHydrogenAbundance",
            "ParticleIDs",
            "StarFormationRate",
            "Velocities",
        ],
        "PartType1": ["Coordinates", "ParticleIDs", "Velocities"],
        "PartType3": ["ParentID", "TracerID"],
        "PartType4": [
            "Coordinates",
            "GFM_InitialMass",
            "GFM_Metallicity",
            "GFM_StellarFormationTime",
            "Masses",
            "ParticleIDs",
            "Velocities",
        ],
        "PartType5": [
            "BH_BPressure",
            "BH_CumEgyInjection_QM",
            "BH_CumEgyInjection_RM",
            "BH_CumMassGrowth_QM",
            "BH_CumMassGrowth_RM",
            "BH_Density",
            "BH_HostHaloMass",
            "BH_Hsml",
            "BH_Mass",
            "BH_Mdot",
            "BH_MdotBondi",
            "BH_MdotEddington",
            "BH_Pressure",
            "BH_Progs",
            "BH_U",
            "Coordinates",
            "Masses",
            "ParticleIDs",
            "Potential",
            "Velocities",
        ],
    }  # all

    # set paths
    sbBasePath = outputPath + "subbox%d/" % sbNum
    saveBasePath = outputPath + "subbox%d_new/" % sbNum

    if not path.isdir(saveBasePath):
        mkdir(saveBasePath)

    def _oldChunkPath(snapNum, chunkNum):
        return sbBasePath + "snapdir_subbox%d_%03d/snap_subbox%d_%03d.%s.hdf5" % (
            sbNum,
            snapNum,
            sbNum,
            snapNum,
            chunkNum,
        )

    def _newChunkPath(snapNum, chunkNum):
        return saveBasePath + "snapdir_subbox%d_%03d/snap_subbox%d_%03d.%s.hdf5" % (
            sbNum,
            snapNum,
            sbNum,
            snapNum,
            chunkNum,
        )

    # from the first: dtypes, ndims, how many chunks?
    print("Save configuration:")
    dtypes = {}
    ndims = {}

    allDone = False
    i = 0

    while not allDone:
        # get dtypes and ndims for each field, but need to find a file that contains each
        with h5py.File(_oldChunkPath(1199, i), "r") as f:
            for gName in keepFields.keys():
                if gName not in f or keepFields[gName][0] not in f[gName]:
                    continue
                dtypes[gName] = {}
                ndims[gName] = {}
                for field in keepFields[gName]:
                    dtypes[gName][field] = f[gName][field].dtype
                    ndims[gName][field] = f[gName][field].ndim
                    if ndims[gName][field] > 1:
                        ndims[gName][field] = f[gName][field].shape[1]  # actually need shape of 2nd dim

        # all done?
        allDone = True
        for gName in keepFields.keys():
            if gName not in dtypes:
                allDone = False
        i += 1

    for gName in keepFields.keys():
        for field in keepFields[gName]:
            print(" ", gName, field, dtypes[gName][field], ndims[gName][field])

    numChunks = len(glob.glob(_oldChunkPath(0, "*")))
    print("  numChunks: %d" % numChunks)

    for sbSnap in range(sbSnapRange[0], sbSnapRange[1] + 1):
        # sbPath = sbBasePath + "snapdir_subbox%d_%d/" % (sbNum, sbSnap)
        oldSize = 0.0
        newSize = 0.0

        # load meta
        meta = {}
        with h5py.File(_oldChunkPath(sbSnap, 0), "r") as f:
            for gName in metaGroups:
                meta[gName] = {}
                for attr in f[gName].attrs:
                    meta[gName][attr] = f[gName].attrs[attr]

        NumPart = meta["Header"]["NumPart_Total"]
        assert meta["Header"]["NumPart_Total_HighWord"].sum() == 0
        print("[%4d] NumPart: " % sbSnap, NumPart)

        # allocate
        data = {}
        offsets = {}
        for gName in keepFields.keys():
            ptNum = int(gName[-1])
            # no particles of this type?
            if NumPart[ptNum] == 0:
                continue

            data[gName] = {}
            offsets[gName] = 0

            for field in keepFields[gName]:
                dtype = dtypes[gName][field]

                # allocate [N] or e.g. [N,3]
                if ndims[gName][field] == 1:
                    shape = NumPart[ptNum]
                else:
                    shape = (NumPart[ptNum], ndims[gName][field])
                data[gName][field] = np.zeros(shape, dtype=dtype)

                if dtype in [np.float32, np.float64]:
                    data[gName][field].fill(np.nan)  # for verification

        # load (requested fields only)
        print("[%4d] loading   [" % sbSnap, end="")
        for i in range(numChunks):
            print(".", end="")
            oldSize += path.getsize(_oldChunkPath(sbSnap, i)) / 1024.0**3

            with h5py.File(_oldChunkPath(sbSnap, i), "r") as f:
                for gName in keepFields.keys():
                    # PartTypeX not in file?
                    if gName not in f:
                        continue

                    # load each field of this PartTypeX
                    for field in keepFields[gName]:
                        ndim = ndims[gName][field]
                        off = offsets[gName]
                        loc_size = f[gName][field].shape[0]

                        # print('  %s off = %8d loc_size = %7d' % (gName,off,loc_size))
                        if ndim == 1:
                            data[gName][field][off : off + loc_size] = f[gName][field][()]
                        else:
                            data[gName][field][off : off + loc_size, :] = f[gName][field][()]

                    offsets[gName] += loc_size
        print("]")

        # verify
        for gName in offsets.keys():
            ptNum = int(gName[-1])
            assert offsets[gName] == NumPart[ptNum]
            for field in data[gName]:
                if dtypes[gName][field] in [np.float32, np.float64]:
                    assert np.count_nonzero(np.isnan(data[gName][field])) == 0

        # write
        print("[%4d] writing   [" % sbSnap, end="")
        assert not path.isdir(saveBasePath + "snapdir_subbox%d_%03d" % (sbNum, sbSnap))
        mkdir(saveBasePath + "snapdir_subbox%d_%03d" % (sbNum, sbSnap))

        start = {}
        stop = {}
        for gName in keepFields.keys():
            start[gName] = 0
            stop[gName] = 0

        for i in range(numChunksSave):
            # determine split, update header
            print(".", end="")

            meta["Header"]["NumFilesPerSnapshot"] = numChunksSave
            for gName in keepFields.keys():
                ptNum = int(gName[-1])
                if NumPart[ptNum] == 0:
                    continue

                start[gName], stop[gName] = pSplitRange([0, NumPart[ptNum]], numChunksSave, i)

                assert stop[gName] > start[gName]
                meta["Header"]["NumPart_ThisFile"][ptNum] = stop[gName] - start[gName]
                # print(i,gName,start[gName],stop[gName])

            with h5py.File(_newChunkPath(sbSnap, i), "w") as f:
                # save meta
                for gName in metaGroups:
                    g = f.create_group(gName)
                    for attr in meta[gName].keys():
                        g.attrs[attr] = meta[gName][attr]

                # save data
                for gName in keepFields.keys():
                    ptNum = int(gName[-1])
                    if NumPart[ptNum] == 0:
                        print(" skip pt %d (write)" % ptNum)
                        continue

                    g = f.create_group(gName)

                    for field in keepFields[gName]:
                        ndim = ndims[gName][field]

                        if ndim == 1:
                            g[field] = data[gName][field][start[gName] : stop[gName]]
                        else:
                            g[field] = data[gName][field][start[gName] : stop[gName], :]

            newSize += path.getsize(_newChunkPath(sbSnap, i)) / 1024.0**3

        print("]")
        print("[%4d] saved (old size = %5.1f GB new size %5.1f GB)" % (i, oldSize, newSize))

    print("Done.")


def groupCutoutFromSnap(run="tng"):
    """Create a [full] subhalo/fof cutout from a snapshot (as would be done by the Web API)."""
    ptTypes = ["gas", "dm", "bhs", "stars"]
    basePath = path.expanduser("~") + "/sims.TNG/L75n1820TNG/postprocessing/guinevere_cutouts/"

    # (A) subhalo indices (z=0): TNG100-1, Illustris-1 (Lagrangian match), Illustris-1 (positional match)
    if 0:
        sP = simParams(res=1820, run=run, redshift=0.0)
        samplePath = basePath + "new_mw_sample_fgas_sat.txt"

        data = np.genfromtxt(samplePath, delimiter=",", dtype="int32")

        if run == "tng":
            subhaloIDs = data[:, 0]
        if run == "illustris":
            subhaloIDs = data[:, 1]  # Lagrangian match

    # (B) L25n512_0000 (TNG), L25n512_0010 (Illustris), L25n512_3000 (no BH wind), L25n512_0012 (TNG w/ Illustris winds)
    if 1:
        sP = simParams(res=512, run="tng", redshift=0.0, variant="1000")
        samplePath = basePath + "new_mw_sample_L25_variants.txt"

        data = np.genfromtxt(samplePath, delimiter=",", dtype="int32")

        if sP.variant == "0000":
            subhaloIDs = data[:, 0]
        if sP.variant == "0010":
            subhaloIDs = data[:, 1]
        if sP.variant == "3000":
            subhaloIDs = data[:, 2]
        if sP.variant == "0012":
            subhaloIDs = data[:, 3]
        if sP.variant == "1000":
            subhaloIDs = data[:, 4]
        print(sP.variant, subhaloIDs)

    # get list of field names
    fields = {}

    fileName = sP.snapPath(sP.snap, chunkNum=0)

    with h5py.File(fileName, "r") as f:
        for partType in ptTypes:
            gName = "PartType" + str(sP.ptNum(partType))
            fields[gName] = f[gName].keys()

    # loop over subhalos
    for subhaloID in subhaloIDs:
        if subhaloID == -1:
            print("skip -1")
            continue

        saveFilename = "cutout_%s_%d_subhalo.hdf5" % (sP.simName, subhaloID)
        if path.isfile(saveFilename):
            print("skip, [%s] already exists" % saveFilename)
            continue

        data = {}

        # load (subhalo restricted)
        for partType in ptTypes:
            print(subhaloID, "sub", partType)
            gName = "PartType" + str(sP.ptNum(partType))

            data[gName] = sP.snapshotSubset(partType, fields[gName], subhaloID=subhaloID)

        # write
        with h5py.File(saveFilename, "w") as f:
            for gName in data:
                g = f.create_group(gName)
                for field in data[gName]:
                    g[field] = data[gName][field]

        # get parent fof, load (fof restricted)
        continue  # skip

        data = {}
        subh = sP.groupCatSingle(subhaloID=subhaloID)
        haloID = subh["SubhaloGrNr"]

        for partType in ptTypes:
            print(subhaloID, "fof", partType)
            gName = "PartType" + str(sP.ptNum(partType))

            data[gName] = sP.snapshotSubset(partType, fields[gName], haloID=haloID)

        # write
        with h5py.File("cutout_%s_%d_group.hdf5" % (sP.simName, subhaloID), "w") as f:
            for gName in data:
                g = f.create_group(gName)
                for field in data[gName]:
                    g[field] = data[gName][field]


def tracerCutoutFromTracerTracksCat():
    """Create a subhalo cutout of tracers from the full postprocessing/tracer_tracks/ catalog."""
    ptTypes = ["gas", "stars", "bhs"]

    # get subhaloIDs
    if 0:
        sP = simParams(res=1820, run="tng", redshift=0.0)
        data = np.genfromtxt(sP.postPath + "guinevere_cutouts/new_mw_sample_fgas_sat.txt", delimiter=",", dtype="int32")
        subhaloIDs = data[:, 0]

    if 1:
        sP = simParams(res=512, run="tng", redshift=0.0, variant="1000")
        basePath = path.expanduser("~") + "/sims.TNG/L75n1820TNG/postprocessing/guinevere_cutouts/"
        samplePath = basePath + "new_mw_sample_L25_variants.txt"
        data = np.genfromtxt(samplePath, delimiter=",", dtype="int32")

        if sP.variant == "0000":
            subhaloIDs = data[:, 0]
        if sP.variant == "3000":
            subhaloIDs = data[:, 2]
        if sP.variant == "0012":
            subhaloIDs = data[:, 3]
        if sP.variant == "1000":
            subhaloIDs = data[:, 4]

    # list of tracer quantities we know
    catBasePath = sP.postPath + "tracer_tracks/*.hdf5"
    cats = {}

    for catPath in glob.glob(catBasePath):
        catName = catPath.split("%d_" % sP.snap)[-1].split(".hdf5")[0]
        cats[catName] = catPath

    # load offset/length from meta
    offs = {}
    lens = {}
    with h5py.File(cats["meta"], "r") as f:
        for ptType in ptTypes:
            offs[ptType] = f["Subhalo"]["TracerOffset"][ptType][()]
            lens[ptType] = f["Subhalo"]["TracerLength"][ptType][()]

    # loop over subhaloIDs
    for subhaloID in subhaloIDs:
        data = {}
        for ptType in ptTypes:
            data[ptType] = {}

        saveFilename = "tracers_%s_%d_subhalo.hdf5" % (sP.simName, subhaloID)
        if path.isfile(saveFilename):
            print("skip, [%s] already exists" % saveFilename)
            continue

        # load from each existing cat
        for catName, catPath in cats.items():
            if "meta.hdf5" in catPath:
                continue
            print(" ", subhaloID, catName)

            with h5py.File(catPath, "r") as f:
                for ptType in ptTypes:
                    start = offs[ptType][subhaloID]
                    length = lens[ptType][subhaloID]
                    data[ptType][catName] = f[catName][:, start : start + length]

                snaps = f["snaps"][()]
                redshifts = f["redshifts"][()]

        # write
        with h5py.File(saveFilename, "w") as f:
            for ptType in ptTypes:
                g = f.create_group(ptType)
                for field in data[ptType]:
                    g[field] = data[ptType][field]
            f["snaps"] = snaps
            f["redshifts"] = redshifts


def makeSnapHeadersForLHaloTree():
    """Copy chunk 0 of each snapshot only and the header only (for LHaloTree B-HaloTree)."""
    nSnaps = 59  # 100

    pathFrom = path.expanduser("~") + "/sims.TNG/L35n2160TNG/output/snapdir_%03d/"
    pathTo = path.expanduser("~") + "/sims.TNG/L35n2160TNG/output/snapdir_%03d/"
    fileFrom = pathFrom + "snap_%03d.%s.hdf5"
    fileTo = pathTo + "snap_%03d.%s.hdf5"

    # loop over snapshots
    for i in range(nSnaps):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # open destination for writing
        j = 0
        if path.isfile(fileTo % (i, i, j)):
            print("Skip snap [%d], already exists" % j)
            continue

        fOut = h5py.File(fileTo % (i, i, j), "w")

        # open origin file for reading
        assert path.isfile(fileFrom % (i, i, j))

        with h5py.File(fileFrom % (i, i, j), "r") as f:
            # copy header
            fOut.create_group("Header")
            for attr in f["Header"].attrs:
                fOut["Header"].attrs[attr] = f["Header"].attrs[attr]

            fOut.close()
            print("%s" % fileTo % (i, i, j))


def makeSnapSubsetsForPostProcessing():
    """Copy snapshot chunks reducing to needed fields for tree/post-processing calculations."""
    nSnaps = 100

    deriveHsml = True

    # SubLink/SubLink_gal
    # copyFields = {'PartType0':['Masses','ParticleIDs','StarFormationRate'], #for SubLink_gal need 'StarFormationRate'
    #              'PartType1':['ParticleIDs'],
    #              'PartType4':['Masses','GFM_StellarFormationTime','ParticleIDs']}

    # Subfind-HBT
    copyFields = {
        "PartType0": ["Coordinates", "Velocities", "Masses", "ParticleIDs", "Density", "InternalEnergy"],
        "PartType1": ["Coordinates", "Velocities", "ParticleIDs"],
        "PartType4": ["Coordinates", "Velocities", "Masses", "ParticleIDs"],
        "PartType5": ["Coordinates", "Velocities", "Masses", "ParticleIDs"],
    }

    pathFrom = path.expanduser("~") + "/sims.TNG/TNG100-1/output/snapdir_%03d/"
    pathTo = path.expanduser("~") + "/data/gadget4/output/snapdir_%03d/"
    fileFrom = pathFrom + "snap_%03d.%s.hdf5"
    fileTo = pathTo + "snap_%03d.%s.hdf5"

    # verify number of chunks
    files = glob.glob(fileFrom % (0, 0, "*"))
    nChunks = len(files)
    print("Found [%d] file chunks, copying." % nChunks)

    # loop over snapshots
    for i in range(nSnaps):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # loop over chunks
        for j in range(nChunks):
            # open destination for writing
            fOut = h5py.File(fileTo % (i, i, j), "w")

            # open origin file for reading
            assert path.isfile(fileFrom % (i, i, j))

            with h5py.File(fileFrom % (i, i, j), "r") as f:
                # copy header
                if "Header" not in fOut:
                    g = fOut.create_group("Header")
                    for attr in f["Header"].attrs:
                        fOut["Header"].attrs[attr] = f["Header"].attrs[attr]

                if "PartType3" not in copyFields.keys():
                    # 'fix' header and set tracer counts to zero, if we aren't copying
                    keys = ["NumPart_ThisFile", "NumPart_Total", "NumPart_Total_HighWord", "MassTable"]
                    for key in keys:
                        attr_vals = fOut["Header"].attrs[key]
                        attr_vals[3] = 0
                        fOut["Header"].attrs[key] = attr_vals

                # loop over partTypes
                for gName in copyFields.keys():
                    # skip if not in origin
                    if gName not in f:
                        continue
                    # copy fields for this partType
                    g = fOut.create_group(gName)
                    for dName in copyFields[gName]:
                        g[dName] = f[gName][dName][()]

                    # calculate SmoothingLength for GADGET post-processing runs?
                    if deriveHsml and gName == "PartType0":
                        if "Volume" in f[gName]:
                            vol = f[gName]["Volume"][()]
                        else:
                            vol = f[gName]["Masses"][()] / f[gName]["Density"]
                        hsml = (vol * 3.0 / (4 * np.pi)) ** (1.0 / 3.0)
                        g["SmoothingLength"] = hsml

            fOut.close()
            print("%s" % fileTo % (i, i, j))


def finalizeSubfindHBTGroupCat(snap, prep=False):
    """Finish Subfind-HBT post-processing of an original Subfind simulation.

    Extract the new group ordered IDs from the rewritten snapshot, place them into the catalog itself, and
    cross-match to the original snapshot IDs, to write the index mapping into the catalog.
    Finally, rewrite the catalog into a single file, and add fof and subhalo cross-matching
    to original catalog. To run: (i) first create minimal snaps with makeSnapSubsetsForPostProcessing()
    above, (ii) run Gadget-4 SubfindHBT, (iii) run finalizeSubfindHBTGroupCat(prep=True),
    (iv) create SubLinkHBT trees, (v) run makeSubgroupOffsetsIntoSublinkTreeGlobal(basePath, treeName='SubLinkHBT'),
    (vi) run finalizeSubfindHBTGroupCat() to finish.
    """
    sP = simParams(run="tng100-2", snap=snap)
    basePath = path.expanduser("~") + "/data/gadget4/"

    snapPath = basePath + "output/snapdir_%03d/" % snap
    groupPath = basePath + "output/groups_%03d/fof_subhalo_tab_%03d" % (snap, snap) + ".%s.hdf5"
    saveFile = basePath + "postprocessing/subfind_hbt_%03d.hdf5" % snap
    saveFileIDs = basePath + "postprocessing/subfind_hbt_%03d_ids.hdf5" % snap
    treeOffsetFile = basePath + "postprocessing/offsets/SubLinkHBT_offsets_subgroup_%d.hdf5" % snap

    # number of chunks
    files = glob.glob(groupPath % "*")
    nChunks = len(files)

    if prep:
        # rewrite a few datasets/attrs to make dtypes consistent with TNG (for SubLink)
        print(snap)
        for i in range(nChunks):
            with h5py.File(groupPath % i, "r+") as f:
                # G4 renamed 'Nsubgroups*' -> 'Nsubhalos*'
                f["Header"].attrs["Nsubgroups_Total"] = f["Header"].attrs["Nsubhalos_Total"].astype("int32")
                f["Header"].attrs["Nsubgroups_ThisFile"] = f["Header"].attrs["Nsubhalos_ThisFile"].astype("int32")

                del f["Header"].attrs["Nsubhalos_Total"]
                del f["Header"].attrs["Nsubhalos_ThisFile"]

                f["Header"].attrs["Ngroups_Total"] = f["Header"].attrs["Ngroups_Total"].astype("int32")
                f["Header"].attrs["Ngroups_ThisFile"] = f["Header"].attrs["Ngroups_ThisFile"].astype("int32")

                # G4 renamed SubhaloGrNr to SubhaloGroupNr
                if "Subhalo" in f and "SubhaloGroupNr" in f["Subhalo"]:
                    SubhaloGrNr = f["Subhalo/SubhaloGroupNr"][()].astype("int32")  # G4: int64
                    f["Subhalo/SubhaloGrNr"] = SubhaloGrNr
                    del f["Subhalo/SubhaloGroupNr"]

                # G4 renamed SubhaloParent to SubhaloParentRank
                if "Subhalo" in f and "SubhaloParentRank" in f["Subhalo"]:
                    SubhaloParent = f["Subhalo/SubhaloParentRank"][()]
                    f["Subhalo/SubhaloParent"] = SubhaloParent
                    del f["Subhalo/SubhaloParentRank"]

                # dtypes
                if "Subhalo" in f and "SubhaloMassType" in f["Subhalo"]:
                    SubhaloMassType = f["Subhalo/SubhaloMassType"][()].astype("float32")  # G4: float64
                    del f["Subhalo/SubhaloMassType"]
                    f["Subhalo/SubhaloMassType"] = SubhaloMassType

                # mass-weighted scalefac of members (redundant if not lightcone)
                if "Group" in f and "GroupAscale" in f["Group"]:
                    del f["Group/GroupAscale"]

        for i in range(nChunks):
            with h5py.File(snapPath + "snap_%03d.%d.hdf5" % (snap, i), "r+") as f:
                f["Header"].attrs["NumPart_ThisFile"] = f["Header"].attrs["NumPart_ThisFile"].astype("int32")
        return

    # total member size by type
    GroupLenTypeTot = np.zeros(6, dtype="int64")

    for i in range(nChunks):
        with h5py.File(groupPath % i, "r") as f:
            assert f["Header"].attrs["NumFiles"] == nChunks
            if "Group" in f:
                GroupLenTypeTot += np.sum(f["Group/GroupLenType"][()], axis=0)

    # (A) create new group catalog single file
    print("A [snap = %d]" % snap)
    with h5py.File(saveFileIDs, "w") as fOut:
        ids = fOut.create_group("IDs")

        for i in range(GroupLenTypeTot.size):
            if GroupLenTypeTot[i] > 0:
                ids.create_dataset("PartType%d" % i, (GroupLenTypeTot[i],), dtype="int64")

        # read ordered IDs and write directly
        offsets = np.zeros(GroupLenTypeTot.size, dtype="int64")

        for i in range(nChunks):
            with h5py.File(snapPath + "snap_%03d.%d.hdf5" % (snap, i), "r") as f:
                for j in range(GroupLenTypeTot.size):
                    gName = "PartType%d" % j
                    if gName not in f or GroupLenTypeTot[j] == 0:
                        continue

                    # determine read/write size
                    rw_size = f[gName]["ParticleIDs"].size

                    if offsets[j] + rw_size > GroupLenTypeTot[j]:
                        rw_size = GroupLenTypeTot[j] - offsets[j]

                    # read
                    ids_loc = f[gName]["ParticleIDs"][0:rw_size]

                    # write into hdf5 directly
                    ids[gName][offsets[j] : offsets[j] + rw_size] = ids_loc
                    offsets[j] += rw_size

    # (B) concat catalog fields into new single file, update headers
    print("B")
    metaGroups = {"Config": {}, "Header": {}, "Parameters": {}}

    dtypes = {"Group": {}, "Subhalo": {}}
    shapes = {"Group": {}, "Subhalo": {}}
    totlen = {}

    with h5py.File(groupPath % (0), "r") as f:
        # record dtype and shapes of all datasets
        totlen["Group"] = f["Header"].attrs["Ngroups_Total"]
        totlen["Subhalo"] = f["Header"].attrs["Nsubgroups_Total"]

        for gName in dtypes.keys():
            if gName not in f:
                continue

            for key in f[gName].keys():
                dtypes[gName][key] = f[gName][key].dtype
                shapes[gName][key] = list(f[gName][key].shape)
                shapes[gName][key][0] = totlen[gName]

        # copy headers and update for single file write
        for gName in metaGroups.keys():
            metaGroups[gName] = dict(f[gName].attrs)

        metaGroups["Header"]["Nids_Total"] = GroupLenTypeTot.sum()  # original value not meaningful
        metaGroups["Header"]["Nsubgroups_ThisFile"] = metaGroups["Header"]["Nsubgroups_Total"]
        metaGroups["Header"]["Ngroups_ThisFile"] = metaGroups["Header"]["Ngroups_Total"]
        metaGroups["Header"]["Nids_ThisFile"] = metaGroups["Header"]["Nids_Total"]
        metaGroups["Header"]["NumFiles"] = 1

    offsets = {"Group": 0, "Subhalo": 0}

    with h5py.File(saveFile, "w") as fOut:
        # allocate datasets
        for gName in dtypes.keys():
            for key in dtypes[gName]:
                fOut["%s/%s" % (gName, key)] = np.zeros(shapes[gName][key], dtype=dtypes[gName][key])

        # copy datasets
        for i in range(nChunks):
            with h5py.File(groupPath % i, "r") as f:
                for gName in dtypes.keys():
                    if gName not in f:
                        continue
                    for key in dtypes[gName]:
                        length = f[gName][key].shape[0]
                        offset = offsets[gName]
                        fOut[gName][key][offset : offset + length] = f[gName][key][()]
                    offsets[gName] += length

        # write headers
        for gName in metaGroups.keys():
            g = fOut.create_group(gName)
            for key in metaGroups[gName].keys():
                g.attrs[key] = metaGroups[gName][key]

    # (C) cross-match particle IDs by type
    for i in range(GroupLenTypeTot.size):
        if GroupLenTypeTot[i] == 0:
            continue
        print("C", i)

        # load sorted ids of new fof/subhalos-hbt
        with h5py.File(saveFileIDs, "r") as f:
            ids_new = f["IDs/PartType%d" % i][()]

        # load sorted ids of old fof/subhalos (cannot restrict to haloSubset of old catalog,
        # possible that the FoFs differ after all)
        ids_old = sP.snapshotSubsetP(i, "ids")

        # indices gives, for each ids_new, the index where it is found in ids_old
        # such that indices[subhalohbt_start:subhalohbt_end] provides the particle
        # indices in the original snapshot of the new subfind-hbt subhalo
        indices, _ = match(ids_old, ids_new)

        assert indices.size == ids_new.size

        # write
        with h5py.File(saveFile, "r+") as fOut:
            fOut["SnapIndices/PartType%d" % i] = indices

    # (D) add group cross-matching results
    if GroupLenTypeTot.sum() == 0:
        print("Done.")
        return

    print("D")
    ptNum = 1  # DM particles only

    with h5py.File(saveFileIDs, "r") as f:
        ids_group1 = f["IDs/PartType%d" % ptNum][()]

    with h5py.File(saveFile, "r") as f:
        lengths_group1 = f["Group/GroupLenType"][:, ptNum]
        lengths_sub1 = f["Subhalo/SubhaloLenType"][:, ptNum]
        nsubs1 = f["Group/GroupNsubs"][()]

    ids_group2 = sP.snapshotSubsetP(ptNum, "ids")
    lengths_group2 = sP.groups("GroupLenType")[:, ptNum]
    lengths_sub2 = sP.subhalos("SubhaloLenType")[:, ptNum]
    nsubs2 = sP.groups("GroupNsubs")

    # match 1->2 and 2->1, then require bijective
    def match_bijective(ids1, lengths1, ids2, lengths2):
        index_12 = crossmatchHalosByCommonIDs(ids1, lengths1, ids2, lengths2)
        index_21 = crossmatchHalosByCommonIDs(ids2, lengths2, ids1, lengths1)

        # w = np.where(index_12 < 0)
        # print('Originally have [%d] objects of [%d] without matches.' % (len(w[0]),lengths1.size))

        w12 = np.where(index_21[index_12] != np.arange(lengths1.size))
        w21 = np.where(index_12[index_21] != np.arange(lengths2.size))
        # print(' Remove [%d] non-bijective 1->2, and [%d] non-bijective 2->1.' % (len(w12[0]),len(w21[0])))

        index_12[w12] = -1
        index_21[w21] = -1

        return index_12, index_21

    haloindex_12, haloindex_21 = match_bijective(ids_group1, lengths_group1, ids_group2, lengths_group2)

    with h5py.File(saveFile, "r+") as f:
        f["Matching/GroupOrigToHBT"] = haloindex_21
        f["Matching/GroupHBTToOrig"] = haloindex_12

    # (E) add subhalo cross-matching results
    print("E")

    def dense_subhalo_ids(lengths_sub, lengths_group, ids_group, nsubs):
        ids_sub = np.zeros(lengths_sub.sum(), dtype=ids_group.dtype)
        offsets_group = np.hstack([0, np.cumsum(lengths_group)[:-1]])
        w_offset = 0
        sub_i = 0

        for k in np.arange(lengths_group.size):
            # subhalo offsets depend on group (to allow fuzz)
            if nsubs[k] > 0:
                # first sub
                offset = offsets_group[k]
                length = lengths_sub[sub_i]
                ids_sub[w_offset : w_offset + length] = ids_group[offset : offset + length]
                w_offset += length
                sub_i += 1

                # satellite subs
                for _m in np.arange(1, nsubs[k]):
                    offset += lengths_sub[sub_i - 1]
                    length = lengths_sub[sub_i]
                    ids_sub[w_offset : w_offset + length] = ids_group[offset : offset + length]
                    w_offset += length
                    sub_i += 1

        assert ids_sub.min() > 0
        return ids_sub

    ids_sub1 = dense_subhalo_ids(lengths_sub1, lengths_group1, ids_group1, nsubs1)
    ids_sub2 = dense_subhalo_ids(lengths_sub2, lengths_group2, ids_group2, nsubs2)

    subindex_12, subindex_21 = match_bijective(ids_sub1, lengths_sub1, ids_sub2, lengths_sub2)

    with h5py.File(saveFile, "r+") as f:
        f["Matching/SubhaloOrigToHBT"] = subindex_21
        f["Matching/SubhaloHBTToOrig"] = subindex_12

    # (F) add offsets into SubLinkHBT merger tree (must be already created)
    # run: makeSubgroupOffsetsIntoSublinkTreeGlobal('/u/dnelson/gadget4/', treeName='SubLinkHBT')
    print("F")
    with h5py.File(treeOffsetFile, "r") as f:
        LastProgenitorID = f["LastProgenitorID"][()]
        MainLeafProgenitorID = f["MainLeafProgenitorID"][()]
        SubhaloID = f["SubhaloID"][()]
        RowNum = f["RowNum"][()]

    with h5py.File(saveFile, "r+") as f:
        f["Subhalo/SubLinkHBT/LastProgenitorID"] = LastProgenitorID
        f["Subhalo/SubLinkHBT/MainLeafProgenitorID"] = MainLeafProgenitorID
        f["Subhalo/SubLinkHBT/SubhaloID"] = SubhaloID
        f["Subhalo/SubLinkHBT/RowNum"] = RowNum

    remove(treeOffsetFile)
    remove(saveFileIDs)

    print("Done.")


def makeSdssSpecObjIDhdf5():
    """Transform some CSV files into a HDF5 for SDSS objid -> specobjid mapping."""
    files = sorted(glob.glob("z*.txt"))
    objid = np.zeros((10000000), dtype="uint64")
    specobjid = np.zeros((10000000), dtype="uint64")
    offset = 0

    # read
    for file in files:
        print(file, offset)
        x = np.loadtxt(file, delimiter=",", skiprows=2, dtype="uint64")
        num = x.shape[0]
        objid[offset : offset + num] = x[:, 0]
        specobjid[offset : offset + num] = x[:, 1]
        offset += num

    # look at uniqueness
    objid = objid[0:offset]
    specobjid = specobjid[0:offset]

    assert nUnique(objid) == objid.size
    assert nUnique(specobjid) == specobjid.size

    # write
    with h5py.File("sdss_objid_specobjid_z0.0-0.5.hdf5", "w") as f:
        f["objid"] = objid
        f["specobjid"] = specobjid


def createEmptyMissingGroupCatChunk():
    """Fix some issue in 4503 variation."""
    nChunks = 64
    basePath = path.expanduser("~") + "/sims.TNG_method/L25n512_4503/output/groups_004/"
    fileBase = basePath + "fof_subhalo_tab_004.%d.hdf5"
    fileMake = fileBase % 60

    # load all chunks, determine number of missing groups/subgroups
    nGroups = 0
    nSubgroups = 0

    for i in range(nChunks):
        if not path.isfile(fileBase % i):
            print("Skip: %s" % fileBase % i)
            continue
        with h5py.File(fileBase % i, "r") as f:
            nGroups += f["Header"].attrs["Ngroups_ThisFile"]
            nSubgroups += f["Header"].attrs["Nsubgroups_ThisFile"]

    # load data shapes and types, and write
    f = h5py.File(fileBase % 0, "r")
    fOut = h5py.File(fileMake, "w")

    nGroupsTot = f["Header"].attrs["Ngroups_Total"]
    nSubgroupsTot = f["Header"].attrs["Nsubgroups_Total"]

    nMissingGroups = nGroupsTot - nGroups
    nMissingSubgroups = nSubgroupsTot - nSubgroups

    print("Missing groups [%d] subgroups [%d]." % (nMissingGroups, nMissingSubgroups))

    fOut.create_group("Header")
    fOut.create_group("Group")
    fOut.create_group("Subhalo")

    # (header)
    for attr in f["Header"].attrs:
        fOut["Header"].attrs[attr] = f["Header"].attrs[attr]
    fOut["Header"].attrs["Ngroups_ThisFile"] = nMissingGroups
    fOut["Header"].attrs["Nsubgroups_ThisFile"] = nMissingSubgroups

    # (group)
    for key in f["Group"]:
        shape = np.array(f["Group"][key].shape)
        shape[0] = nMissingGroups
        fOut["Group"][key] = np.zeros(shape, dtype=f["Group"][key].dtype)

    # (subhalo)
    for key in f["Subhalo"]:
        shape = np.array(f["Subhalo"][key].shape)
        shape[0] = nMissingSubgroups
        fOut["Subhalo"][key] = np.zeros(shape, dtype=f["Subhalo"][key].dtype)

    f.close()
    fOut.close()
    print("Wrote: %s" % fileMake)


def combineAuxCatSubdivisions():
    """Combine a subdivision of a pSplit auxCat calculation."""
    basePath = path.expanduser("~") + "/sims.TNG/L205n2500TNG/data.files/auxCat/"
    field = "Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1"  # _rad30pkpc
    fileBase = field + "_099-split-%d-%d.hdf5"

    pSplitBig = [90, 100, 160]  # from
    pSplitSm = [9, 16]  # to

    # load properties
    allExist = True
    allCount = 0

    for i in range(pSplitBig[0], pSplitBig[1]):
        filePath_i = basePath + fileBase % (i, pSplitBig[2])
        print(filePath_i)

        if not isfile(filePath_i):
            allExist = False
            continue

        # record counts and dataset shape
        with h5py.File(filePath_i, "r") as f:
            allCount += f["subhaloIDs"].size
            allShape = f[field].shape

    assert allExist

    # allocate
    allShape = np.array(allShape)
    allShape[0] = allCount  # size
    offset = 0

    new_r = np.zeros(allShape, dtype="float32")
    subhaloIDs = np.zeros(allCount, dtype="int32")

    new_r.fill(-1.0)  # does validly contain nan
    subhaloIDs.fill(np.nan)

    # read
    for i in range(pSplitBig[0], pSplitBig[1]):
        filePath_i = basePath + fileBase % (i, pSplitBig[2])
        print(filePath_i)

        # record counts and dataset shape
        with h5py.File(filePath_i, "r") as f:
            length = f["subhaloIDs"].size
            subhaloIDs[offset : offset + length] = f["subhaloIDs"][()]

            new_r[offset : offset + length, :, :] = f[field][()]

            offset += length

    assert np.count_nonzero(np.isnan(subhaloIDs)) == 0
    assert np.count_nonzero(new_r == -1.0) == 0

    outPath = path.expanduser("~") + "/data/" + fileBase % (pSplitSm[0], pSplitSm[1])
    print("Write to: [%s]" % outPath)

    assert not isfile(outPath)
    with h5py.File(outPath, "w") as f:
        f.create_dataset(field, data=new_r)
        f.create_dataset("subhaloIDs", data=subhaloIDs)

    print("Saved.")

    verifyPath = basePath + fileBase % (pSplitSm[0], pSplitSm[1])
    if not isfile(verifyPath):
        print("Verify does not exist, skip [%s]." % verifyPath)
        return

    with h5py.File(verifyPath, "r") as f:
        verify_r = f[field][()]
        verify_ids = f["subhaloIDs"][()]

    assert np.array_equal(verify_ids, subhaloIDs)
    # np.array_equal() is False for NaN entries
    # roundoff differences:
    # assert ((verify_r == new_r) | (np.isnan(verify_r) & np.isnan(new_r))).all()
    assert np.allclose(verify_r, new_r, equal_nan=True)
    print("Verified.")


def snapRedshiftsTxt():
    """Output a text-file of snapshot redshifts, etc."""
    # config
    sP = simParams(res=1820, run="tng", redshift=0.0, variant="subbox0")
    minZ = 0.0
    maxZ = 50.0  # tng subboxes start at a=0.02, illustris at a=0.0078125
    maxNSnaps = 2700  # 90 seconds at 30 fps
    matchUse = "condense"

    # snapshots
    panels = [{"sP": sP}]
    snapNumLists = multiRunMatchedSnapList(panels, matchUse, maxNum=maxNSnaps, minRedshift=minZ, maxRedshift=maxZ)

    snapNums = snapNumLists[0]
    frameNums = np.arange(snapNums.size)
    redshifts = snapNumToRedshift(sP, snapNums)
    ages = snapNumToAgeFlat(sP, snap=snapNums)

    # write
    with open("frames.txt", "w") as f:
        f.write("# frame_number snapshot_number redshift age_universe_gyr\n")
        for i in range(frameNums.size):
            f.write("%04d %04d %6.3f %5.3f\n" % (frameNums[i], snapNums[i], redshifts[i], ages[i]))
    print("Wrote: [frames.txt].")


def tngVariantsLatexOrWikiTable(variants="all", fmt="wiki"):
    """Output latex-syntax table describing the TNG model variations runs."""
    run_file = path.expanduser("~") + "/sims.TNG_method/runs.csv"

    with open(run_file) as f:
        lines = f.readlines()

    # if variants is a list of lists, flatten
    if isinstance(variants[0], list):
        variants = [item for sublist in variants for item in sublist]

    # header
    if fmt == "wiki":
        print('{| class="eoTable2 wikiTable"')
        print(
            "! run || name || description || 128^3 || 256^3 || 512^3 || "
            + "parameter/option changed || fiducial value || changed value || notes"
        )
    if fmt == "latex":
        print(r"\begin{table*}")
        print(r"  \fontsize{8}{8}\selectfont")
        print(r"  \caption{Caption here.}")
        print(r"  \label{simTable}")
        print(r"  \begin{center}")
        print(r"    \\begin{tabular}{rlllll}")
        print(r"     \hline\hline")
        print(r"     \# & Run Name & Parameter(s) Changed & Fiducial Value & Modified Value & Reference \\ \hline")

    count = 1
    for line in csv.reader(lines, quoting=csv.QUOTE_ALL):
        if "_" in line[0]:
            (
                run,
                prio,
                largevol,
                who,
                stat512,
                stat256,
                stat128,
                recomp,
                name,
                desc,
                change,
                val_fiducial,
                val_changed,
                notes,
            ) = line
            run = run.split("_")[1]

            if stat512 != "done":
                continue
            if variants != "all":
                if run not in variants:
                    continue

            if fmt == "wiki":
                print("|-")
                runstat = "{{yes}} || {{yes}} || {{yes}}"  # 128, 256, 512
                print(
                    "|| %s || %s || %s || %s || %s || %s || %s || %s"
                    % (run, name, desc, runstat, change, val_fiducial, val_changed, notes)
                )
            if fmt == "latex":
                ref = "W17" if "BH" in name else "P17"  # needs to be corrected for other cases
                change = change.replace("_", r"\_").replace("#", r"\#")
                print(r"     %d & %s & %s & %s & %s & %s \\" % (count, name, change, val_fiducial, val_changed, ref))

            count += 1

    # footer
    if fmt == "wiki":
        print("|}")
    if fmt == "latex":
        print(r"    \hline")
        print(r"    \end{tabular}")
        print(r"  \end{center}")
        print(r"\end{table*}")


def splitSingleHDF5IntoChunks(snap=151):
    """Split a single-file snapshot/catalog/etc HDF5 into a number of roughly equally sized chunks."""
    basePath = path.expanduser("~") + "/sims.other/Simba-L25n512FP/output/snapdir_%03d/" % snap
    fileName = "snap_%03d.hdf5" % snap
    numChunksSave = 16

    # load header, dtypes, ndims, and data
    fGroups = {}
    dtypes = {}
    ndims = {}
    data = {}

    with h5py.File(basePath + fileName, "r") as f:
        header = dict(f["Header"].attrs.items())
        for k in f.keys():
            if k != "Header":
                fGroups[k] = []

        for gName in fGroups.keys():
            dtypes[gName] = {}
            ndims[gName] = {}
            data[gName] = {}
            fGroups[gName] = list(f[gName].keys())

            for field in fGroups[gName]:
                print(gName, field)
                dtypes[gName][field] = f[gName][field].dtype
                ndims[gName][field] = f[gName][field].ndim
                if ndims[gName][field] > 1:
                    ndims[gName][field] = f[gName][field].shape[1]  # actually need shape of 2nd dim
                data[gName][field] = f[gName][field][()]

    print("NumPartTot: ", header["NumPart_Total"])

    start = {}
    stop = {}
    for gName in fGroups.keys():
        start[gName] = 0
        stop[gName] = 0

    for i in range(numChunksSave):
        # determine split, update header
        header["NumFilesPerSnapshot"] = numChunksSave
        for gName in fGroups.keys():
            ptNum = int(gName[-1])
            if header["NumPart_Total"][ptNum] == 0:
                continue

            start[gName], stop[gName] = pSplitRange([0, header["NumPart_Total"][ptNum]], numChunksSave, i)
            if start[gName] == 0 and stop[gName] == 0:
                # e.g. just a few stars or BHs, these are assigned to the last chunk
                assert header["NumPart_Total"][ptNum] < numChunksSave
            else:
                assert stop[gName] > start[gName]

            header["NumPart_ThisFile"][ptNum] = stop[gName] - start[gName]
            print(i, gName, start[gName], stop[gName])

        newChunkPath = basePath + fileName.split(".hdf5")[0] + ".%d.hdf5" % i

        with h5py.File(newChunkPath, "w") as f:
            # save header
            g = f.create_group("Header")
            for attr in header.keys():
                g.attrs[attr] = header[attr]

            # save data
            for gName in fGroups.keys():
                ptNum = int(gName[-1])
                if header["NumPart_Total"][ptNum] == 0:
                    print(" skip pt %d (write)" % ptNum)
                    continue

                g = f.create_group(gName)

                for field in fGroups[gName]:
                    ndim = ndims[gName][field]

                    if ndim == 1:
                        g[field] = data[gName][field][start[gName] : stop[gName]]
                    else:
                        g[field] = data[gName][field][start[gName] : stop[gName], :]

    print("Done.")


def combineMultipleHDF5FilesIntoSingle():
    """Combine multiple groupcat file chunks into a single HDF5 file."""
    simName = "L35n2160TNG"
    snap = 33

    loadPath = path.expanduser("~") + "/sims.TNG/%s/output/groups_%03d/" % (simName, snap)
    savePath = "/mnt/nvme/cache/%s/output/groups_%03d/" % (simName, snap)
    fileBase = "fof_subhalo_tab_%03d.%s.hdf5"
    outFile = "fof_subhalo_tab_%03d.hdf5" % snap

    # metadata
    data = {}
    offsets = {}
    headers = {}

    with h5py.File(loadPath + fileBase % (snap, 0), "r") as f:
        for gName in f.keys():
            if len(f[gName]):
                # group with datasets, e.g. Group, Subhalo, PartType0
                data[gName] = {}
                offsets[gName] = 0
            else:
                # group with no datasets, i.e. only attributes, e.g. Header, Config, Parameters
                headers[gName] = dict(f[gName].attrs.items())

            for field in f[gName].keys():
                shape = list(f[gName][field].shape)

                # replace first dim with total length
                if gName == "Group":
                    shape[0] = f["Header"].attrs["Ngroups_Total"]
                elif gName == "Subhalo":
                    shape[0] = f["Header"].attrs["Nsubgroups_Total"]
                else:
                    assert 0  # handle

                # allocate
                data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # load
    nFiles = len(glob.glob(loadPath + fileBase % (snap, "*")))

    for i in range(nFiles):
        print(i, offsets["Group"], offsets["Subhalo"])
        with h5py.File(loadPath + fileBase % (snap, i), "r") as f:
            for gName in f.keys():
                if len(f[gName]) == 0:
                    continue

                offset = offsets[gName]

                # load and stamp
                for field in f[gName]:
                    length = f[gName][field].shape[0]

                    data[gName][field][offset : offset + length, ...] = f[gName][field][()]

                offsets[gName] += length

    # save
    if not path.isdir(savePath):
        mkdir(savePath)

    with h5py.File(savePath + outFile, "w") as f:
        # add header groups
        for gName in headers:
            f.create_group(gName)
            for attr in headers[gName]:
                f[gName].attrs[attr] = headers[gName][attr]

        f["Header"].attrs["Ngroups_ThisFile"] = f["Header"].attrs["Ngroups_Total"]
        f["Header"].attrs["Nsubgroups_ThisFile"] = f["Header"].attrs["Nsubgroups_Total"]
        f["Header"].attrs["NumFiles"] = 1

        # add datasets
        for gName in data:
            f.create_group(gName)
            for field in data[gName]:
                f[gName][field] = data[gName][field]
                assert data[gName][field].shape[0] == offsets[gName]

    print("Saved: [%s]" % outFile)


def convertVoronoiConnectivityVPPP(stage=1, thisTask=0):
    """Read the Voronoi mesh data from Chris Byrohl using his vppp (voro++ parallel) approach, save to HDF5."""
    sP = simParams(run="tng50-2", redshift=0.5)
    basepath = "/freya/ptmp/mpa/cbyrohl/public/vppp_dataset/IllustrisTNG50-2_z0.5_posdata"

    file1 = basepath + ".bin.nb"
    file2 = basepath + ".bin.nb2"

    outfile1 = path.expanduser("~") + "/sims.TNG/L35n1080TNG/data.files/voronoi/mesh_spatialorder_%02d.hdf5" % sP.snap
    outfile2 = path.expanduser("~") + "/sims.TNG/L35n1080TNG/data.files/voronoi/mesh_%02d.hdf5" % sP.snap

    dtype_nb = np.dtype(
        [
            ("x", np.float64),
            ("y", np.float64),
            ("z", np.float64),
            ("gidx", np.int64),  # snapshot index (1-indexed !!!!!)
            ("noffset", np.int64),  # offset in neighbor list (1-indexed !!!!!)
            ("ncount", np.int32),  # neighborcount
        ]
    )

    # convert stage (1): rewrite into HDF5
    if stage == 1:
        # chunked load
        chunksize = 100000000

        # get npart and ngb list size
        with open(file1, "rb") as f:
            f.seek(0)
            npart = np.fromfile(f, dtype=np.int64, count=1)[0]

        with open(file2, "rb") as f:
            f.seek(0)
            tot_num_entries = np.fromfile(f, dtype=np.int64, count=1)[0]

        # open save file
        fOut = h5py.File(outfile1, "w")

        snap_index = fOut.create_dataset("snap_index", (npart,), dtype="int64")
        num_ngb = fOut.create_dataset("num_ngb", (npart,), dtype="int16")
        offset_ngb = fOut.create_dataset("offset_ngb", (npart,), dtype="int64")
        x = fOut.create_dataset("x", (npart,), dtype="float32")
        ngb_inds = fOut.create_dataset("ngb_inds", (tot_num_entries,), dtype="int64")  # (1-indexed !!)

        # load all from file1
        with open(file1, "rb") as f:
            # get npart
            f.seek(0)

            nloaded = 0
            byte_offset = 8  # skip npart

            while nloaded < npart:
                print("loaded %4.1f%% [%10d] of [%10d]" % (nloaded / npart * 100, nloaded, npart))
                f.seek(byte_offset)
                data = np.fromfile(f, dtype=dtype_nb, count=chunksize)

                # save
                snap_index[nloaded : nloaded + data.size] = data["gidx"] - 1  # change from 1-based fortran indexing
                num_ngb[nloaded : nloaded + data.size] = data["ncount"]
                offset_ngb[nloaded : nloaded + data.size] = data["noffset"] - 1  # change from 1-based fortran indexing
                x[nloaded : nloaded + data.size] = data["x"]

                # continue
                nloaded += data.size
                byte_offset += data.size * dtype_nb.itemsize

        # load neighbor list from file2
        with open(file2, "rb") as f:
            f.seek(0)

            nloaded = 0
            byte_offset = 8  # skip tot_num_entries

            while nloaded < tot_num_entries:
                print("ngblist %4.1f%% [%10d] of [%10d]" % (nloaded / tot_num_entries * 100, nloaded, tot_num_entries))
                f.seek(byte_offset)
                data = np.fromfile(f, dtype=np.int64, count=chunksize * 10)

                if data.size == 0:
                    break

                # save
                ngb_inds[nloaded : nloaded + data.size] = data - 1  # change from 1-based fortran indexing

                # continue
                nloaded += data.size
                byte_offset += data.size * data.itemsize

        fOut.close()

    # convert stage (2): shuffle into snapshot order
    if stage == 2:
        with h5py.File(outfile1, "r+") as f:
            snap_index = f["snap_index"][()]

            sort_inds = np.argsort(snap_index)
            f["sort_inds"] = sort_inds

    # sanity checks A
    if stage == 3:
        with h5py.File(outfile1, "r") as f:
            snap_index = f["snap_index"][()]
            sort_inds = f["sort_inds"][()]

        # check: indices are dense
        new_snap_index = snap_index[sort_inds]

        numGas = sP.snapshotHeader()["NumPart"][sP.ptNum("gas")]
        lin_list = np.arange(numGas)
        print(np.array_equal(lin_list, new_snap_index))

    # sanity checks B
    if stage == 4:
        with h5py.File(outfile1, "r+") as f:
            x = f["x"][()] * sP.boxSize  # [0,1] -> [0,sP.boxSize]
            sort_inds = f["sort_inds"][()]

        # check: order is correct by comparing x-coordinates
        new_x = x[sort_inds]
        snap_x = sP.snapshotSubsetP("gas", "pos_x")

        print(new_x[0:5], snap_x[0:5])
        print(np.allclose(new_x, snap_x))

    # stage (5): save spatial domain information for non-groupordered datafile
    if stage == 5:
        with open(basepath.replace("posdata", "domains.txt")) as f:
            lines = f.readlines()

        # Format (4 lines per chunk): (all lengths/starts in numbers of entries, not bytes)
        # CHUNKID, NB2 start, NB2 length
        # NB1 start, NB1 length
        # xstart, ystart, zstart
        # xend, yend, zend
        nChunks = 512
        assert len(lines) == nChunks * 4  # 512 cubic spatial subsets

        # allocate
        r = {
            "chunk_id": np.zeros(nChunks, dtype="int32"),
            "offset_ngb": np.zeros(nChunks, dtype="int64"),
            "num_ngb": np.zeros(nChunks, dtype="int64"),
            "offset_cells": np.zeros(nChunks, dtype="int64"),
            "num_cells": np.zeros(nChunks, dtype="int64"),
            "xyz_min": np.zeros((nChunks, 3), dtype="float32"),
            "xyz_max": np.zeros((nChunks, 3), dtype="float32"),
        }

        # parse
        for i in np.arange(0, nChunks):
            r["chunk_id"][i], r["offset_ngb"][i], r["num_ngb"][i] = [int(x) for x in lines[i * 4 + 0].split()]
            r["offset_cells"][i], r["num_cells"][i] = [int(x) for x in lines[i * 4 + 1].split()]
            r["xyz_min"][i, :] = [float(x) for x in lines[i * 4 + 2].split()]
            r["xyz_max"][i, :] = [float(x) for x in lines[i * 4 + 3].split()]

        # sanity checks
        nCells = sP.snapshotHeader()["NumPart"][sP.ptNum("gas")]
        assert np.array_equal(r["chunk_id"], np.arange(nChunks))
        assert r["offset_ngb"].min() >= 0 and r["num_ngb"].min() >= 0
        assert r["offset_cells"].min() >= 0 and r["num_cells"].min() >= 0 and r["offset_cells"].max() < nCells
        assert r["xyz_min"].min() >= 0 and r["xyz_min"].max() <= 1.0
        assert r["xyz_max"].min() >= 0 and r["xyz_max"].max() <= 1.0

        # save into spatially ordered datafile
        with h5py.File(outfile1, "r+") as f:
            for key in r:
                f["meta/%s" % key] = r[key]
        print("Saved spatial metadata.")

    # convert stage (6): create new final 'mesh' file with shuffled num_ngb and offset_ngb
    if stage == 6:
        # metadata
        with h5py.File(outfile1, "r") as f:
            sort_inds = f["sort_inds"][()]
            tot_num_entries = f["ngb_inds"].size

        # read and write per-cell datasets
        for key in ["num_ngb", "offset_ngb"]:
            with h5py.File(outfile1, "r") as f:
                data = f[key][()]
            with h5py.File(outfile2, "a") as f:
                f[key] = data[sort_inds]
            print(key, flush=True)

        # allocate unfilled neighbor list
        with h5py.File(outfile2, "r+") as f:
            ngb_inds = f.create_dataset("ngb_inds", (tot_num_entries,), dtype="int64")
            for i in range(20):
                locrange = pSplitRange([0, tot_num_entries], 20, i)
                print(i, locrange, flush=True)
                ngb_inds[locrange[0] : locrange[1]] = -1

    # convert stage (7): rewrite ngb_inds into dense, contiguous subsets following snapshot order
    nTasks = 10  # 140

    if stage == 7:
        with h5py.File(outfile2, "r") as f:
            num_cells = f["num_ngb"].size

        locRange = pSplitRange([0, num_cells], nTasks, thisTask)

        # load original offsets
        with h5py.File(outfile2, "r") as f:
            num_ngb = f["num_ngb"][locRange[0] : locRange[1]]
            offset_ngb = f["offset_ngb"][locRange[0] : locRange[1]]

        # allocate sub-task output file
        subfile = outfile2.replace(".hdf5", "_%d_of_%d.hdf5" % (thisTask, nTasks))
        totNumNgbLoc = num_ngb.sum()
        ngb_inds = np.zeros(totNumNgbLoc, dtype="int64") - 1

        print("[%2d of %2d] starting... " % (thisTask, nTasks), locRange, flush=True)

        offset = 0

        with h5py.File(outfile1, "r") as f:  # open source
            for i in range(num_ngb.size):
                if i % 100000 == 0:
                    print("[%2d] [%10d] %.2f%%" % (thisTask, i, i / num_ngb.size * 100), flush=True)

                # read
                loc_inds = f["ngb_inds"][offset_ngb[i] : offset_ngb[i] + num_ngb[i]]

                ngb_inds[offset : offset + num_ngb[i]] = loc_inds

                offset += num_ngb[i]

        # save
        with h5py.File(subfile, "w") as f:
            f["ngb_inds"] = ngb_inds

    # convert stage (8): concatenate ngb_inds
    if stage == 8:
        # note: do not need to permute ngb_inds, since they are indices into the snapshot, not into the nb1 file
        global_min = np.inf
        global_max = -1

        offset = 0

        for i in range(nTasks):
            subfile = outfile2.replace(".hdf5", "_%d_of_%d.hdf5" % (i, nTasks))
            with h5py.File(subfile, "r") as f:
                loc_inds = f["ngb_inds"][()]

            print(i, loc_inds.min(), loc_inds.max(), flush=True)

            if loc_inds.max() > global_max:
                global_max = loc_inds.max()
            if loc_inds.min() < global_min:
                global_min = loc_inds.min()

            # save
            with h5py.File(outfile2, "r+") as f:
                f["ngb_inds"][offset : offset + loc_inds.size] = loc_inds

            offset += loc_inds.size

        print("global min: ", global_min)
        print("global max: ", global_max)
        print("final offset: ", offset)

        with h5py.File(outfile1, "r") as f:
            print("should equal: ", f["ngb_inds"].size)

    # convert stage (9): new offset_ngb
    if stage == 9:
        with h5py.File(outfile2, "r+") as f:
            num_ngb = f["num_ngb"][()]
            offset_ngb = np.zeros(num_ngb.size, dtype="int64")
            offset_ngb[1:] = np.cumsum(num_ngb)[:-1]
            f["offset_ngb"][:] = offset_ngb

    print("done.")


def exportSubhalosBinary():
    """Export a very minimal group catalog to a flat binary format (for WebGL/Explorer3D)."""
    # config
    sP = simParams(run="eagle", redshift=0.0)
    cenSatSelect = "cen"
    writeN = 100000  # None for all
    nValsPerHalo = 7  # x, y, z, r200, log_Tvir, log_M200, log_Lx

    # load
    pos = sP.groupCat(fieldsSubhalos=["SubhaloPos"])
    r200 = sP.groupCat(fieldsSubhalos=["rhalo_200"])
    tvir = sP.groupCat(fieldsSubhalos=["tvir_log"])
    m200 = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])

    # Lx = sP.auxCat(['Subhalo_XrayBolLum'])['Subhalo_XrayBolLum']
    # Lx = np.log10(Lx.astype('float64') * 1e30).astype('float32') # log erg/s
    Lx = np.zeros(tvir.size, dtype="float32")
    w = np.where(~np.isfinite(Lx))
    Lx[w] = 0.0

    # reduce precision/quantize, no need, and data is not public
    r200 = np.round(r200 * 1.0) / 1.0
    m200 = np.round(m200 * 10.0) / 10.0
    tvir = np.round(tvir * 10.0) / 10.0
    Lx = np.round(Lx * 10.0) / 10.0

    # restrict to css
    inds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

    if writeN is not None:
        inds = inds[0:writeN]

    if sP.simNameAlt == "L680n2048TNG_DM":
        # TNG1 testing: load custom
        from ..cosmo.zooms import _halo_ids_run

        halo_inds = _halo_ids_run()
        inds = sP.halos("GroupFirstSub")[halo_inds]

    nSubhalos = inds.size
    print("Writing [%d] %s subhalos..." % (nSubhalos, cenSatSelect))

    # linearize
    floatsOut = np.zeros(nValsPerHalo * nSubhalos, dtype="float32")

    for i in range(nSubhalos):
        floatsOut[i * nValsPerHalo + 0] = pos[inds[i], 0]
        floatsOut[i * nValsPerHalo + 1] = pos[inds[i], 1]
        floatsOut[i * nValsPerHalo + 2] = pos[inds[i], 2]
        floatsOut[i * nValsPerHalo + 3] = r200[inds[i]]
        floatsOut[i * nValsPerHalo + 4] = tvir[inds[i]]
        floatsOut[i * nValsPerHalo + 5] = m200[inds[i]]
        floatsOut[i * nValsPerHalo + 6] = Lx[inds[i]]

    # open output
    writeStr = "all" if writeN is None else "N%d" % writeN
    fileName = "subh_%s_%s_%s_z%.1f.dat" % (sP.simName, cenSatSelect, writeStr, sP.redshift)

    with open(fileName, "wb") as f:
        # header (28 bytes)
        binVersion = 1
        headerBytes = 7 * 4

        header = np.array([binVersion, headerBytes, nSubhalos, nValsPerHalo, sP.snap], dtype="int32")
        f.write(struct.pack("iiiii", *header))
        header = np.array([sP.redshift, sP.boxSize], dtype="float32")
        f.write(struct.pack("ff", *header))

        # write (x0,y0,z0,pa0,x1,y1,...)
        f.write(struct.pack("f" * len(floatsOut), *floatsOut))

        # write ID list
        id_list = inds[0:nSubhalos].astype("int32")
        f.write(struct.pack("i" * len(id_list), *id_list))

        # footer
        footer = np.array([99], dtype="int32")
        f.write(struct.pack("i", *footer))

    print("Saved: [%s]" % fileName)


def exportHierarchicalBoxGrids(
    sP,
    partType="gas",
    partField="mass",
    nCells=(32, 64, 128, 256, 512),
    haloID=None,
    haloSizeRvir=2.0,
    retData=False,
    memoryReturn=False,
):
    """Export one or more 3D uniform Cartesian grids, of different resolutions, to a flat binary format.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      partType (str): particle type, e.g. 'gas', 'dm', 'star'.
      partField (str): particle field to grid, e.g. 'mass', 'dens', 'temp', 'metallicity', etc.
      nCells (list[int]): one or more grid sizes (number of cells per linear dimension)
      haloID (int or None): if None then use full box, otherwise fof-restricted particle load.
      haloSizeRvir (float): if ``haloID`` is specified, then this gives the box side-length in rvir units.
      retData (bool): if True, immediately return python object (list of ndarrays) of data.
      memoryReturn (bool): if True, an actual file is not written, and instead a bytes array is returned.

    Return:
      Either a list, raw binary inside a :py:class:`io.BytesIO`, or no return if an actual file is written to disk.

    Note:
      This function is used to pre-compute the grids used in the Explorer3D WebGL volume rendering interface, as
      well as on-the-fly grid generation for halo-scope volume rendering.
    """
    # config
    label, limits, takeLog = sP.simParticleQuantity(partType, partField)

    # load
    massField = partField if "mass" in partField else "mass"

    pos = sP.snapshotSubsetP(partType, "pos", haloID=haloID)
    mass = sP.snapshotSubsetP(partType, massField, haloID=haloID) if partType != "dm" else sP.dmParticleMass
    hsml = sP.snapshotSubsetP(partType, "hsml", haloID=haloID)

    quant = None  # grid mass
    if "mass" not in partField:  # grid a different, mass-weighted quantity
        quant = sP.snapshotSubsetP(partType, partField, haloID=haloID)
        # assert partField != 'dens' # do mass instead

    # make series of grids at progressively better resolution
    grids = []

    for nCell in nCells:
        if haloID is None:
            grid = sphGridWholeBox(sP, pos, hsml, mass, quant, nCells=nCell)
        else:
            halo = sP.halo(haloID)
            boxSizeImg = halo["Group_R_Crit200"] * np.array([haloSizeRvir, haloSizeRvir, haloSizeRvir])
            boxCen = halo["GroupPos"]

            grid = sphMap(
                pos=pos,
                hsml=hsml,
                mass=mass,
                quant=quant,
                axes=[0, 1],
                ndims=3,
                boxSizeSim=sP.boxSize,
                boxSizeImg=boxSizeImg,
                boxCen=boxCen,
                nPixels=[nCell, nCell, nCell],
            )

        if "mass" in partField:  # unit conversion
            pxVol = (sP.boxSize / nCell) ** 3  # code units (ckpc/h)^3
            grid = sP.units.codeDensToPhys(grid / pxVol) * 1e10  # Msun/kpc^3

        if takeLog:
            grid = logZeroMin(grid)

        grid = grid.astype("float16")  # 2 bytes per value!
        grids.append(grid.ravel())

    if retData:
        return grids

    # save binary
    if memoryReturn:
        f = BytesIO()
    else:
        fileName = "boxgrid_%s_%s_%s_z%.1f.dat" % (sP.simName.replace("-2", ""), partType, partField, sP.redshift)

        f = open(fileName, "wb")

    # header (24 bytes + 12 bytes per grid)
    binVersion = 1
    headerBytes = 6 * 4 + len(nCells) * 12

    header = np.array([binVersion, headerBytes, len(nCells), sP.snap], dtype="int32")
    f.write(struct.pack("iiii", *header))
    header = np.array([sP.redshift, sP.boxSize], dtype="float32")
    f.write(struct.pack("ff", *header))

    # for each grid, write [nCells, startOffset, stopOffset], offsets are file global
    offset = headerBytes
    for i, grid in enumerate(grids):
        header = np.array([nCells[i], offset, offset + grid.nbytes], dtype="int32")
        offset += grid.nbytes
        f.write(struct.pack("i" * len(header), *header))

    # write each grid (nCells[i]**3*2 bytes each)
    # to write in float32: only change 'e' to 'f' and 'float16' to 'float32' above
    for grid in grids:
        f.write(struct.pack("e" * len(grid), *grid))

    # footer
    footer = np.array([99], dtype="int32")
    f.write(struct.pack("i", *footer))

    if memoryReturn:
        return f

    f.close()
    print("Saved: [%s]" % fileName)


def exportIltisCutout(sim, haloIDs, emLine="O  7 21.6020A", haloSizeRvir=2.0, sfrEmisFac=1):
    """Export a snapshot cutout (global scope) around a given halo, for use in ILTIS-RT.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      haloIDs (list[int]): list of FoF halo IDs to center the cutouts on.
      emLine (str): the emission line to save the emissivity of, which also specifies the ion for the density field.
      haloSizeRvir (float): the box side-length in rvir units.
      sfrEmisFac (float): if not unity, then multiplicative (boost) factor by which to change the emissivities
        for star-forming gas only, e.g. to generate simple models for bright central sources.

    Return:
      None. An actual file is written to disk.
    """
    savePath = path.expanduser("~") + "/data/public/OVII_iltis/inprogress/"
    sim.createCloudyCache = True

    # load gas of entire box (large-memory node, for multi-halo efficiency)
    pos = sim.gas("pos")
    vel = sim.gas("vel")

    emis = sim.gas("%s lum2phase" % emLine) * (1e30 / 1e42)  # in 1e42 erg/s

    if sfrEmisFac != 1:
        sfr = sim.gas("sfr")

    ionName = emLine.split()[0] + " " + emLine.split()[1]
    dens = sim.gas("%s numdens" % ionName)  # cm^-3

    temp = sim.gas("temp_sfcold")  # K

    # make halo-based spatial cutout
    for haloID in haloIDs:
        # filename
        fileName = savePath + "cutout_%s_%d_halo%d_size%d_b%s_%s.hdf5" % (
            sim.simName,
            sim.snap,
            haloID,
            int(haloSizeRvir),
            sfrEmisFac,
            emLine.replace(" ", "-"),
        )

        if path.isfile(fileName):
            continue

        # load halo, calculate spatial subset
        halo = sim.halo(haloID)

        dists_xyz = sim.periodicDists(halo["GroupPos"], pos, chebyshev=True)

        dists_xyz /= halo["Group_R_Crit200"]

        inds = np.where(dists_xyz <= haloSizeRvir)[0]

        print(
            f"Selected [{inds.size}] of [{dists_xyz.size}] total gas cells = {inds.size / dists_xyz.size * 100:.2f}%%"
        )

        # coordinates in box length units
        pos_loc = pos[inds] / sim.boxSize

        # velocities and emissivities
        vel_loc = sim.units.particleCodeVelocityToKms(vel[inds]) * 1e5  # cm/s

        emis_loc = emis[inds]

        if sfrEmisFac != 1:
            ww = np.where(sfr[inds] > 0)
            emis_loc[ww] *= sfrEmisFac

        dens_loc = dens[inds]
        temp_loc = temp[inds]

        # not yet used: dust, turbulent velocity
        dust_dens = np.zeros(dens_loc.size, dtype="float32")
        turb_vel = np.zeros(dens_loc.size, dtype="float32")

        # write hdf5 file
        with h5py.File(fileName, "w") as f:
            # write header attributes
            f.attrs["BoxSizeCGS"] = sim.units.codeLengthToKpc(sim.boxSize) * sim.units.kpc_in_cm
            f.attrs["Center"] = halo["GroupPos"] / sim.boxSize
            f.attrs["CutoutLength"] = (haloSizeRvir * halo["Group_R_Crit200"]) / sim.boxSize
            f.attrs["CutoutShape"] = "cube"
            f.attrs["Redshift"] = sim.redshift

            # Coordinates* are in boxlength units [0.0,1.0), Emissivity is in [1e42 erg/s], everything else in [cgs]
            f["CoordinateX"] = pos_loc[:, 0]
            f["CoordinateY"] = pos_loc[:, 1]
            f["CoordinateZ"] = pos_loc[:, 2]
            f["Emissivity"] = emis_loc
            f["density"] = dens_loc
            f["dust_density"] = dust_dens
            f["temperature"] = temp_loc
            f["turbulent_velocity"] = turb_vel
            f["velocity_x"] = vel_loc[:, 0]
            f["velocity_y"] = vel_loc[:, 1]
            f["velocity_z"] = vel_loc[:, 2]

            # keep indices so we can map back to global snapshot cells
            f["global_inds"] = inds

        print("Saved: [%s]" % fileName, flush=True)
