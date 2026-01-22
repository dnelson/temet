"""
Loading I/O - fof/subhalo group cataloges.
"""

import glob
from os import mkdir
from os.path import isdir, isfile

import h5py
import illustris_python as il
import numpy as np
from numba import njit

from ..util.helper import iterable, logZeroNaN


# registry for groupcat field metadata (in groupcat_fields.py)
groupcat_fields = {}

# and custom-derived catalog fields (in groupcat_fields_custom.py)
custom_cat_fields = {}
custom_cat_fields_aliases = {}
custom_cat_multi_fields = {}


def catalog_field(arg=None, **kwargs):
    """Decorator factory to save custom catalog field deriving functions into the registry."""

    def decorator(f):
        # add entry using the function name as the custom field name
        custom_cat_fields[f.__name__] = f

        # add entries for alias(es)
        aliases = iterable(kwargs.get("aliases", [])) + iterable(kwargs.get("alias", []))

        for alias in aliases:
            custom_cat_fields[alias] = f

        # is this a handler for multiple fields/wildcards?
        multi = kwargs.get("multi")

        if multi:
            if isinstance(multi, str):
                # value of 'multi' argument is the wildcard search key
                custom_cat_fields[multi] = f
                custom_cat_multi_fields[multi] = f
            else:
                # otherwise, name of the decorated function is the search key
                custom_cat_multi_fields[f.__name__] = f
        else:
            # for non-multi fields, keep track of primary name, and its list of aliases (for docs)
            custom_cat_fields_aliases[f.__name__] = aliases

        return f

    if callable(arg):
        # @catalog_field() is a function returning a decorator
        return decorator(arg)
    else:
        # catalog_field is just a decorator
        return decorator


def gcPath(basePath, snapNum, chunkNum=0, noLocal=False, checkExists=False):
    """Find and return absolute path to a group catalog HDF5 file.

    Can be used to redefine illustris_python version (il.groupcat.gcPath = load.groupcat.gcPath).
    """
    # local scratch test: call ourself with a basePath corresponding to local scratch (on freyator)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/mnt/nvme/cache/%s/%s/" % (bpSplit[-3], bpSplit[-2])
        localFT = gcPath(localBP, snapNum, chunkNum=chunkNum, noLocal=True, checkExists=True)

        if localFT:
            # print("Note: Reading groupcat from local scratch [%s]!" % localFT)
            return localFT

    # format snapshot number
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [  # both fof+subfind in single (non-split) file in root directory
        basePath + "/fof_subhalo_tab_" + ext + ".hdf5",
        # standard: both fof+subfind in >1 files per snapshot, in subdirectory
        basePath + "groups_" + ext + "/fof_subhalo_tab_" + ext + "." + str(chunkNum) + ".hdf5",
        # fof only, in >1 files per snapshot, in subdirectory
        basePath + "groups_" + ext + "/fof_tab_" + ext + "." + str(chunkNum) + ".hdf5",
        # rewritten new group catalogs with offsets
        basePath + "groups_" + ext + "/groups_" + ext + "." + str(chunkNum) + ".hdf5",
        # single (non-split) file in subdirectory (i.e. Millennium rewrite)
        basePath + "groups_" + ext + "/fof_subhalo_tab_" + ext + ".hdf5",
    ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    # failure:
    for fileName in fileNames:
        print(" " + fileName)
    raise Exception("No group catalog found.")


def groupCat(sP, sub=None, halo=None, group=None, fieldsSubhalos=None, fieldsHalos=None, sq=True):
    """Load HDF5 fof+subfind group catalog for a given snapshot, one or more fields, possibly custom.

    fieldsSubhalos : read only a subset of Subgroup fields from the catalog
    fieldsHalos    : read only a subset of Group fields from the catalog
    sub            : shorthand for fieldsSubhalos
    halo,group     : shorthands for fieldsHalos
    sq             : squeeze single field return into a numpy array instead of within a dict
    """
    assert sP.snap is not None, "Must specify sP.snap for groupCat() load."
    assert sP.subbox is None, "No groupCat() for subbox snapshots."

    if sub is not None:
        assert fieldsSubhalos is None
        fieldsSubhalos = sub
    if halo is not None:
        assert group is None and fieldsHalos is None
        fieldsHalos = halo
    if group is not None:
        assert halo is None and fieldsHalos is None
        fieldsHalos = group

    assert fieldsSubhalos is not None or fieldsHalos is not None, "Must specify fields type."

    r = {}

    # derived HALO fields
    if fieldsHalos is not None:
        fieldsHalos = list(iterable(fieldsHalos))

        for field in fieldsHalos:
            quant = field.lower()
            quantName = quant.lower().replace("_log", "")

            # fields defined only for TNG-Cluster, generalize to normal boxes
            if quantName in ["groupprimaryzoomtarget"]:
                if not groupCatHasField(sP, "Group", "GroupPrimaryZoomTarget"):
                    r[field] = np.ones(sP.numHalos, dtype="int32")  # all valid

        # for now, if a custom halo field requested, only 1 allowed, and cannot mix with anything else
        if len(r) > 0:
            assert len(r) == 1
            assert fieldsSubhalos is None
            if sq and len(r) == 1:
                # compress and return single field
                key = list(r.keys())[0]
                return r[key]

            return r

    # derived SUBHALO fields and unit conversions (mhalo_200_log, ...). Can request >=1 custom fields
    # and >=1 standard fields simultaneously, as opposed to snapshotSubset().
    if fieldsSubhalos is not None:
        fieldsSubhalos = list(iterable(fieldsSubhalos))

        # special behaviors (AREPO -> AREPO-2 conventions)
        field_renames = {
            "SubhaloGrNr": "SubhaloGroupNr",
            "SubhaloSFRinRad": "SubhaloSfrInRad",
            "SubhaloSFRinHalfRad": "SubhaloSfrInHalfRad",
            "SubhaloSFRinMaxRad": "SubhaloSfrInMaxRad",
            "SubhaloGasMetallicitySfrWeighted": "SubhaloGasMetallicityWeighted",
        }

        for field_old, field_new in field_renames.items():
            if field_old in fieldsSubhalos and not sP.groupCatHasField("Subhalo", field_old):
                fieldsSubhalos[fieldsSubhalos.index(field_old)] = field_new

        for field in fieldsSubhalos:
            quant = field.lower()

            # cache check
            cacheKey = "gc_subcustom_%s" % field
            if cacheKey in sP.data:
                r[field] = sP.data[cacheKey]
                continue

            quantName = quant.lower().replace("_log", "")

            # does (exact) field name exist in custom field registry?
            if quantName in custom_cat_fields:
                # yes: load/compute now
                data = custom_cat_fields[quantName](sP, quantName)

                # if return is None, then this is a fall-through to a normal load
                if data is not None:
                    assert data.size == sP.numSubhalos, "Size-mismatch intended?"
                    r[field] = data
            else:
                # if not, try wild-card matching for custom fields
                for search_key in custom_cat_multi_fields:
                    # requested field contains search key?
                    if search_key in quantName:
                        r[field] = custom_cat_multi_fields[search_key](sP, quantName)

            # log?
            if quant[-4:] == "_log":
                assert field in r, "Error: Can only request '_log' of a custom field. Likely a typo in the field name."
                r[field] = logZeroNaN(r[field])

            # save cache
            if field in r:
                sP.data[cacheKey] = r[field]

        if len(r) >= 1:
            # have at least one custom subhalo field, were halos also requested? not allowed
            assert fieldsHalos is None

            # do we also have standard fields requested? if so, load them now and combine
            if len(r) < len(fieldsSubhalos):
                standardFields = list(fieldsSubhalos)
                for key in r.keys():
                    standardFields.remove(key)
                gc = groupCat(sP, fieldsSubhalos=standardFields, sq=False)
                if isinstance(gc["subhalos"], np.ndarray):
                    assert len(standardFields) == 1
                    gc["subhalos"] = {standardFields[0]: gc["subhalos"]}  # pack into dictionary as expected
                gc["subhalos"].update(r)
                r = gc

            if sq and len(r) == 1:
                # compress and return single field
                key = list(r.keys())[0]
                assert len(r.keys()) == 1
                return r[key]
            else:
                # return dictionary of fields (no 'subhalos' wrapping)
                if "subhalos" in r:
                    return r["subhalos"]
                return r

    # override path function
    il.groupcat.gcPathOrig = il.groupcat.gcPath
    il.groupcat.gcPath = gcPath

    # read
    r["header"] = il.groupcat.loadHeader(sP.simPath, sP.snap)

    if fieldsSubhalos is not None:
        # check cache
        fieldsSubhalos = iterable(fieldsSubhalos)
        r["subhalos"] = {}

        for field in fieldsSubhalos:
            cacheKey = "gc_sub_%s" % field
            if cacheKey in sP.data:
                r["subhalos"][field] = sP.data[cacheKey]
                fieldsSubhalos.remove(field)

        # load
        if len(fieldsSubhalos):
            data = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=fieldsSubhalos)
            if isinstance(data, dict):
                r["subhalos"].update(data)
            else:
                assert isinstance(data, np.ndarray) and len(fieldsSubhalos) == 1
                r["subhalos"][fieldsSubhalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == "illustris":
            for field in fieldsSubhalos:
                if "Metallicity" in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig  # set to new catalogs
                    print("Note: Overriding subhalo [" + field + "] with groups_ new catalog values.")
                    r["subhalos"][field] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath  # restore

        # cache
        for field in r["subhalos"]:
            sP.data["gc_sub_%s" % field] = r["subhalos"][field]

        # reverse AREPO -> AREPO-2 conventions
        for field_old, field_new in field_renames.items():
            if field_new in r["subhalos"]:
                r["subhalos"][field_old] = r["subhalos"][field_new]
                r["subhalos"].pop(field_new)

        key0 = list(r["subhalos"].keys())[0]
        if len(r["subhalos"].keys()) == 1 and key0 != "count":  # keep old behavior of il.groupcat.loadSubhalos()
            r["subhalos"] = r["subhalos"][key0]

    if fieldsHalos is not None:
        # check cache
        fieldsHalos = iterable(fieldsHalos)
        r["halos"] = {}

        for field in fieldsHalos:
            cacheKey = "gc_halo_%s" % field
            if cacheKey in sP.data:
                r["halos"][field] = sP.data[cacheKey]
                fieldsHalos.remove(field)

        # load
        if len(fieldsHalos):
            data = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=fieldsHalos)
            if isinstance(data, dict):
                r["halos"].update(data)
            else:
                assert isinstance(data, np.ndarray) and len(fieldsHalos) == 1
                r["halos"][fieldsHalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == "illustris":
            for field in fieldsHalos:
                if "Metallicity" in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig  # set to new catalogs
                    print("Note: Overriding halo [" + field + "] with groups_ new catalog values.")
                    r["halos"][field] = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath  # restore

        # override HDF5 datatypes if needed (GroupFirstSub unsigned -> signed for -1 entries)
        if isinstance(r["halos"], dict):
            if "GroupFirstSub" in r["halos"]:
                r["halos"]["GroupFirstSub"] = r["halos"]["GroupFirstSub"].astype("int32")
        else:
            if iterable(fieldsHalos)[0] == "GroupFirstSub":
                assert len(iterable(fieldsHalos)) == 1
                r["halos"] = r["halos"].astype("int32")

        for field in r["halos"]:  # cache
            sP.data["gc_halo_%s" % field] = r["halos"][field]

        key0 = list(r["halos"].keys())[0]
        if len(r["halos"].keys()) == 1 and key0 != "count":  # keep old behavior of il.groupcat.loadHalos()
            r["halos"] = r["halos"][key0]

    if sq:
        # if possible: remove 'halos'/'subhalos' subdict, and field subdict
        if fieldsSubhalos is None:
            r = r["halos"]
        if fieldsHalos is None:
            r = r["subhalos"]

        if isinstance(r, dict) and len(r.keys()) == 1 and r["count"] > 0:
            r = r[list(r.keys())[0]]

    return r


def groupCat_subhalos(sP, fields):
    """Wrapper for above."""
    return groupCat(sP, fieldsSubhalos=fields)


def groupCat_halos(sP, fields):
    """Wrapper for above."""
    return groupCat(sP, fieldsHalos=fields)


def groupCatSingle(sP, haloID=None, subhaloID=None):
    """Return complete group catalog information for one halo or subhalo."""
    assert haloID is None or subhaloID is None, "Cannot specify both haloID and subhaloID."
    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No groupCatSingle() for subbox snapshots."

    gcName = "Subhalo" if subhaloID is not None else "Group"
    gcID = subhaloID if subhaloID is not None else haloID
    assert gcID >= 0

    # load groupcat offsets, calculate target file and offset
    groupFileOffsets = groupCatOffsetList(sP)["offsets" + gcName]
    groupFileOffsets = gcID - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load halo/subhalo fields into a dict
    r = {}

    with h5py.File(gcPath(sP.simPath, sP.snap, fileNum), "r") as f:
        for haloProp in f[gcName].keys():
            r[haloProp] = f[gcName][haloProp][groupOffset]

    if "SubhaloGroupNr" in r:
        r["SubhaloGrNr"] = r["SubhaloGroupNr"]  # AREPO-2 -> AREPO convention

    return r


def groupCatSingle_subhalo(sP, obj_id):
    """Wrapper for above."""
    return groupCatSingle(sP, subhaloID=obj_id)


def groupCatSingle_halo(sP, obj_id):
    """Wrapper for above."""
    return groupCatSingle(sP, haloID=obj_id)


def groupCatHeader(sP, fileName=None):
    """Load complete group catalog header."""
    if fileName is None:
        fileName = gcPath(sP.simPath, sP.snap)

    if fileName is None:
        return {"Ngroups_Total": 0, "Nsubgroups_Total": 0}

    with h5py.File(fileName, "r") as f:
        header = dict(f["Header"].attrs.items())

    # AREPO-2 -> AREPO convention
    if "Nsubhalos_Total" in header:
        # AREPO-2 also has uint64, but this causes np.arange() to return float, messing up many indexing opreations
        header["Nsubgroups_Total"] = np.int64(header["Nsubhalos_Total"])
        header["Nsubgroups_ThisFile"] = np.int64(header["Nsubhalos_ThisFile"])

    return header


def groupCatHasField(sP, objType, field):
    """True or False, does group catalog for objType=['Group','Subhalo'] have field?"""
    with h5py.File(gcPath(sP.simPath, sP.snap), "r") as f:
        if objType in f and field in f[objType]:
            return True

    return False


def groupCatFields(sP, objType):
    """Return list of all fields in the group catalog for either halos or subhalos."""
    for i in range(groupCatNumChunks(sP.basePath, sP.snap)):
        with h5py.File(gcPath(sP.simPath, sP.snap, i), "r") as f:
            if objType in f:
                fields = list(f[objType].keys())
                break

    return fields


def groupCatNumChunks(basePath, snapNum):
    """Find number of file chunks in a group catalog."""
    # load from header of first file
    path = gcPath(basePath, snapNum, chunkNum=0, checkExists=True)

    if path is None:
        return 0

    with h5py.File(path, "r") as f:
        nChunks = f["Header"].attrs["NumFiles"]

    return nChunks


def groupCatOffsetList(sP):
    """Make the offset table for the group catalog files, to determine the file for a given group/subgroup ID."""
    saveFilename = sP.derivPath + "offsets/groupcat_" + str(sP.snap) + ".hdf5"

    if not isdir(sP.derivPath + "offsets"):
        mkdir(sP.derivPath + "offsets")

    r = {}

    # local nvme? we use here only single files for efficiency
    path = gcPath(sP.simPath, sP.snap)
    if "/nvme/" in path:
        assert len(glob.glob(path.replace(".0.hdf5", ".hdf5"))) == 1  # make sure we are as expected
        r["offsetsGroup"] = np.array([0], dtype="int32")
        r["offsetsSubhalo"] = np.array([0], dtype="int32")
        return r

    # normal
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            r["offsetsGroup"] = f["offsetsGroup"][()]
            r["offsetsSubhalo"] = f["offsetsSubhalo"][()]
    else:
        nChunks = groupCatNumChunks(sP.simPath, sP.snap)
        r["offsetsGroup"] = np.zeros(nChunks, dtype="int32")
        r["offsetsSubhalo"] = np.zeros(nChunks, dtype="int32")

        for i in np.arange(1, nChunks + 1):
            f = h5py.File(gcPath(sP.simPath, sP.snap, chunkNum=i - 1), "r")

            key1 = "Ngroups_ThisFile"
            key2 = "Nsubgroups_ThisFile" if "Nsubgroups_ThisFile" in f["Header"].attrs else "Nsubhalos_ThisFile"

            if i < nChunks:
                r["offsetsGroup"][i] = r["offsetsGroup"][i - 1] + f["Header"].attrs[key1]
                r["offsetsSubhalo"][i] = r["offsetsSubhalo"][i - 1] + f["Header"].attrs[key2]

                f.close()

        with h5py.File(saveFilename, "w") as f:
            f["offsetsGroup"] = r["offsetsGroup"]
            f["offsetsSubhalo"] = r["offsetsSubhalo"]
            print("Wrote: " + saveFilename)

    return r


@njit
def _group_cat_offsets(groupLenType, subgroupLenType, groupNsubs, totGroups, totSubGroups, nTypes):
    """Loop over each particle type, then over groups, calculate offsets from length."""
    snapOffsetsGroup = np.zeros((totGroups + 1, nTypes), dtype=np.int64)
    snapOffsetsSubhalo = np.zeros((totSubGroups + 1, nTypes), dtype=np.int64)

    nTypes = snapOffsetsGroup.shape[1]

    for j in range(nTypes):
        subgroupCount = 0

        # compute group offsets first (first entry is zero!)
        snapOffsetsGroup[1:, j] = np.cumsum(groupLenType[:, j])

        for k in range(totGroups):
            # subhalo offsets depend on group (to allow fuzz)
            if groupNsubs[k] > 0:
                snapOffsetsSubhalo[subgroupCount, j] = snapOffsetsGroup[k, j]

                subgroupCount += 1
                for _m in np.arange(1, groupNsubs[k]):
                    snapOffsetsSubhalo[subgroupCount, j] = (
                        snapOffsetsSubhalo[subgroupCount - 1, j] + subgroupLenType[subgroupCount - 1, j]
                    )
                    subgroupCount += 1

    return snapOffsetsGroup, snapOffsetsSubhalo


def groupCatOffsetListIntoSnap(sP):
    """Make the particle offset table (by type) for every group/subgroup.

    This allows the global location of the particle/cell members of any group/subgroup to be quickly located.
    """
    saveFilename = sP.derivPath + "offsets/snap_groups_" + str(sP.snap) + ".hdf5"

    if not isdir(sP.derivPath + "offsets"):
        mkdir(sP.derivPath + "offsets")

    r = {}

    # check for existence of save file
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            r["snapOffsetsGroup"] = f["snapOffsetsGroup"][()]
            r["snapOffsetsSubhalo"] = f["snapOffsetsSubhalo"][()]

        if r["snapOffsetsGroup"].max() == 0 and sP.numHalos > 1:
            print("WARNING: [%s] seems corrupt, recomputing." % saveFilename)
        else:
            return r

    # calculate now: allocate
    with h5py.File(gcPath(sP.simPath, sP.snap, noLocal=True), "r") as f:
        shkey = "Nsubgroups_Total" if "Nsubgroups_Total" in f["Header"].attrs else "Nsubhalos_Total"

        totGroups = int(f["Header"].attrs["Ngroups_Total"])
        totSubGroups = int(f["Header"].attrs[shkey])  # int() otherwise +1 below casts result to float64

    shkey = shkey.replace("_Total", "_ThisFile")

    groupCount = 0
    subgroupCount = 0

    # load following 3 fields across all chunks
    groupLenType = np.zeros((totGroups, sP.nTypes), dtype=np.int32)
    groupNsubs = np.zeros((totGroups,), dtype=np.int32)
    subgroupLenType = np.zeros((totSubGroups, sP.nTypes), dtype=np.int32)

    nChunks = groupCatNumChunks(sP.simPath, sP.snap)
    print("Calculating new groupCatOffsetsListIntoSnap... [" + str(nChunks) + " chunks]")

    for i in range(1, nChunks + 1):
        # load header, get number of groups/subgroups in this file, and lengths
        f = h5py.File(gcPath(sP.simPath, sP.snap, chunkNum=i - 1, noLocal=True), "r")
        header = dict(f["Header"].attrs.items())

        Ngroups = int(header["Ngroups_ThisFile"])
        Nsubgroups = int(header[shkey])

        if header["Ngroups_ThisFile"] > 0:
            if "GroupLenType" in f["Group"]:
                groupLenType[groupCount : groupCount + Ngroups] = f["Group"]["GroupLenType"]
            else:
                assert sP.targetGasMass == 0.0  # Millennium DMO with no types
                groupLenType[groupCount : groupCount + Ngroups, sP.ptNum("dm")] = f["Group"]["GroupLen"]

            groupNsubs[groupCount : groupCount + Ngroups] = f["Group"]["GroupNsubs"]

        if Nsubgroups > 0:
            if "SubhaloLenType" in f["Subhalo"]:
                subgroupLenType[subgroupCount : subgroupCount + Nsubgroups] = f["Subhalo"]["SubhaloLenType"]
            else:
                assert sP.targetGasMass == 0.0  # Millennium DMO with no types
                subgroupLenType[subgroupCount : subgroupCount + Nsubgroups, sP.ptNum("dm")] = f["Subhalo"]["SubhaloLen"]

        groupCount += Ngroups
        subgroupCount += Nsubgroups

        f.close()

    # calculate group and subhalo offsets (allow fuzz gaps)
    snapOffsetsGroup, snapOffsetsSubhalo = _group_cat_offsets(
        groupLenType, subgroupLenType, groupNsubs, totGroups, totSubGroups, sP.nTypes
    )

    r = {"snapOffsetsGroup": snapOffsetsGroup, "snapOffsetsSubhalo": snapOffsetsSubhalo}

    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Wrote: " + saveFilename)

    return r


def groupOrderedValsToSubhaloOrdered(vals_group, sP):
    """Shuffle an array of values, one per halo, into a subhalo ID indexed array, assigning the values to each central.

    Non-centrals are left at NaN value.
    """
    groupFirstSubs = sP.groupCat(fieldsHalos=["GroupFirstSub"])
    assert groupFirstSubs.shape == vals_group.shape

    vals_sub = np.zeros(sP.numSubhalos, dtype="float64")
    vals_sub.fill(np.nan)
    vals_sub[groupFirstSubs] = vals_group

    return vals_sub
