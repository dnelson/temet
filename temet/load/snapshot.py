"""
Loading I/O - snapshots of AREPO cosmological simulations.
"""

import glob
import multiprocessing as mp
import multiprocessing.sharedctypes
from functools import partial  # wraps
from getpass import getuser
from os import makedirs
from os.path import isdir, isfile

import h5py
import illustris_python as il
import numpy as np

from ..load.groupcat import groupCatOffsetList, groupCatOffsetListIntoSnap
from ..util.helper import iterable, logZeroNaN, pSplitRange


# registry for snapshot field metadata (in snap_fields.py)
snapshot_fields = {}

# and custom-derived snapshot fields (in snap_fields_custom.py)
custom_fields = {}
custom_fields_aliases = {}
custom_multi_fields = {}


def snap_field(arg=None, **kwargs):
    """Decorator factory to save custom snapshot field deriving functions into the registry."""

    def decorator(f):
        # wrap the actual calculation function (not needed unless we want to actually add pre or
        # post functionality e.g. verify the return is an ndarray of the right size)
        # @wraps(f)
        # def wrapper(*args_, **kwargs_):
        #    return f(*args_, **kwargs_)

        # add entry using the function name as the custom field name
        custom_fields[f.__name__] = f  # wrapper

        # add entries for alias(es)
        aliases = iterable(kwargs.get("aliases", [])) + iterable(kwargs.get("alias", []))

        for alias in aliases:
            custom_fields[alias] = f  # wrapper

        # is this a handler for multiple fields/wildcards?
        multi = kwargs.get("multi")

        if multi:
            if isinstance(multi, str):
                # value of 'multi' argument is the wildcard search key
                custom_fields[multi] = f  # wrapper
                custom_multi_fields[multi] = f  # wrapper
            else:
                # otherwise, name of the decorated function is the search key
                custom_multi_fields[f.__name__] = f  # wrapper
        else:
            # for non-multi fields, keep track of primary name, and its list of aliases (for docs)
            custom_fields_aliases[f.__name__] = aliases

        return f
        # return wrapper

    if callable(arg):
        # @snap_field() is a function returning a decorator
        return decorator(arg)
    else:
        # snap_field is just a decorator
        return decorator


def subboxVals(subbox):
    """Return sbNum (integer) and sbStr1 and sbStr2 for use in locating subbox files."""
    sbNum = subbox if isinstance(subbox, int) else 0

    if subbox is not None:
        sbStr1 = "subbox" + str(sbNum) + "_"
        sbStr2 = "subbox" + str(sbNum) + "/"
    else:
        sbStr1 = ""
        sbStr2 = ""

    return sbNum, sbStr1, sbStr2


def snapPath(basePath, snapNum, chunkNum=0, subbox=None, checkExists=False):
    """Find and return absolute path to a snapshot HDF5 file.

    Can be used to redefine illustris_python version (il.snapshot.snapPath = load.snapshot.snapPath).
    """
    sbNum, sbStr1, sbStr2 = subboxVals(subbox)
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [  # standard: >1 file per snapshot, in subdirectory
        basePath + sbStr2 + "snapdir_" + sbStr1 + ext + "/snap_" + sbStr1 + ext + "." + str(chunkNum) + ".hdf5",
        # auriga, >1 file per snapshot, alternative base
        basePath + "snapdir_%s/snapshot_%s.%s.hdf5" % (ext, ext, chunkNum),
        # single file per snapshot
        basePath + sbStr2 + "snap_" + sbStr1 + ext + ".hdf5",
        # single file per snapshot (swift convention)
        basePath + sbStr2 + "snap_%s.hdf5" % str(snapNum).zfill(4),
        # single file per snapshot (smuggle convention)
        basePath + "snapshot_" + str(snapNum).zfill(3) + ".hdf5",
        # single file per snapshot, in subdirectory (i.e. Millennium rewrite)
        basePath + sbStr2 + "snapdir_" + sbStr1 + ext + "/snap_" + sbStr1 + ext + ".hdf5",
        # single groupordered file per snapshot
        basePath + sbStr2 + "snap-groupordered_" + ext + ".hdf5",
        # multiple groupordered files
        basePath
        + sbStr2
        + "snapdir_"
        + sbStr1
        + ext
        + "/snap-groupordered_"
        + sbStr1
        + ext
        + "."
        + str(chunkNum)
        + ".hdf5",
        # raw input (basePath actually contains a absolute path to a snapshot file already)
        basePath,
    ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    # failure:
    for fileName in fileNames:
        print(" " + fileName)
    raise Exception("ERROR: No snapshot found.")


def snapNumChunks(basePath, snapNum, subbox=None):
    """Find number of file chunks in a snapshot, by checking for existence of files inside directory."""
    _, sbStr1, sbStr2 = subboxVals(subbox)
    path = basePath + sbStr2 + "snapdir_" + sbStr1 + str(snapNum).zfill(3) + "/snap*.*.hdf5"
    nChunks = len(glob.glob(path))

    # check actual header (i.e. extra/duplicate files in the output folder)
    path = snapPath(basePath, snapNum, chunkNum=0, subbox=subbox, checkExists=True)
    if path is not None:
        with h5py.File(path, "r") as f:
            nChunksCheck = f["Header"].attrs["NumFilesPerSnapshot"]
        if nChunksCheck < nChunks:
            print("Note: Replacing snapshot nChunks [%d] with [%d] from header." % (nChunks, nChunksCheck))
            nChunks = nChunksCheck

    if nChunks == 0:
        nChunks = 1  # single file per snapshot

    return nChunks


def snapshotHeader(sP, fileName=None):
    """Load complete snapshot header."""
    if fileName is None:
        assert sP.snap is not None, "Must specify snapshot number to load header."
        fileName = snapPath(sP.simPath, sP.snap, subbox=sP.subbox)

    with h5py.File(fileName, "r") as f:
        header = dict(f["Header"].attrs.items())

        if "Cosmology" in f:
            header.update(f["Cosmology"].attrs.items())

    # calculate and include NumPart_Total
    header["NumPart"] = il.snapshot.getNumPart(header)

    del header["NumPart_Total"]
    if "NumPart_Total_HighWord" in header:
        del header["NumPart_Total_HighWord"]

    return header


def snapHasField(sP, partType, field):
    """True or False, does snapshot data for partType have field?"""
    gName = partType

    if "PartType" not in partType:
        gName = "PartType" + str(sP.ptNum(partType))

    # cache (for efficiency)
    cacheKey = "snapHasField_%s_%s" % (gName, field)
    if cacheKey in sP.data:
        return sP.data[cacheKey]

    # the first chunk could not have the field but it could exist in a later chunk (e.g. sparse file
    # contents of subboxes). to definitely return False, we have to check them all, but we can return
    # an early True if we find a(ny) chunk containing the field
    for i in range(snapNumChunks(sP.simPath, sP.snap, subbox=sP.subbox)):
        fileName = snapPath(sP.simPath, sP.snap, chunkNum=i, subbox=sP.subbox)

        with h5py.File(fileName, "r") as f:
            if "%s/%s" % (gName, field) in f:
                sP.data[cacheKey] = True
                return True

    sP.data[cacheKey] = False
    return False


def snapFields(sP, partType):
    """Return list of all fields for this particle type."""
    gName = partType
    if "PartType" not in partType:
        gName = "PartType" + str(sP.ptNum(partType))

    for i in range(snapNumChunks(sP.simPath, sP.snap, subbox=sP.subbox)):
        fileName = snapPath(sP.simPath, sP.snap, chunkNum=i, subbox=sP.subbox)

        with h5py.File(fileName, "r") as f:
            if gName in f:
                fields = list(f[gName].keys())
                break

    return fields


def snapConfigVars(sP):
    """Load Config.sh flags and values as stored in the /Config/ group of modern snapshots."""
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file, "r") as f:
        if "Config" in f:
            config = dict(f["Config"].attrs.items())
        else:
            config = None

    return config


def snapParameterVars(sP):
    """Load param.txt flags and values as stored in the /Parameters/ group of modern snapshots."""
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file, "r") as f:
        if "Parameters" in f:
            params = dict(f["Parameters"].attrs.items())
        else:
            params = None

    return params


def snapOffsetList(sP):
    """Make the offset table (by type) for the snapshot files, to find in which file a given offset+length exists."""
    _, sbStr1, sbStr2 = subboxVals(sP.subbox)
    saveFilename = sP.derivPath + "offsets/%ssnapshot_%s.hdf5" % (sbStr2, sP.snap)

    if not isdir(sP.derivPath + "offsets/%s" % sbStr2):
        makedirs(sP.derivPath + "offsets/%s" % sbStr2)

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            snapOffsets = f["offsets"][()]
    else:
        nChunks = snapNumChunks(sP.simPath, sP.snap, sP.subbox)
        snapOffsets = np.zeros((sP.nTypes, nChunks), dtype="int64")

        for i in np.arange(1, nChunks + 1):
            f = h5py.File(snapPath(sP.simPath, sP.snap, chunkNum=i - 1, subbox=sP.subbox), "r")

            if i < nChunks:
                for j in range(sP.nTypes):
                    snapOffsets[j, i] = snapOffsets[j, i - 1] + f["Header"].attrs["NumPart_ThisFile"][j]

                f.close()

        with h5py.File(saveFilename, "w") as f:
            f["offsets"] = snapOffsets
            print("Wrote: " + saveFilename)

    return snapOffsets


def haloOrSubhaloSubset(sP, haloID=None, subhaloID=None):
    """Return the subset dict{} for a given haloID/subhaloID, as needed by il.snapshot.loadSubset()."""
    gcName = "Group" if haloID is not None else "Subhalo"
    gcID = haloID if haloID is not None else subhaloID

    subset = {"snapOffsets": snapOffsetList(sP)}

    # calculate target groups file chunk which contains this id
    groupFileOffsets = groupCatOffsetList(sP)["offsets" + gcName]
    groupFileOffsets = int(gcID) - groupFileOffsets
    fileNum = np.max(np.where(groupFileOffsets >= 0))
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog, and its offset within the snapshot
    with h5py.File(sP.gcPath(sP.snap, fileNum), "r") as f:
        if gcName + "LenType" in f[gcName]:
            subset["lenType"] = f[gcName][gcName + "LenType"][groupOffset, :]
        else:
            assert sP.targetGasMass == 0.0
            print("Warning: Should be DMO (Millennium) simulation with no LenType.")
            subset["lenType"] = np.zeros(sP.nTypes, dtype="int64")
            subset["lenType"][sP.ptNum("dm")] = f[gcName][gcName + "Len"][groupOffset]

        subset["offsetType"] = groupCatOffsetListIntoSnap(sP)["snapOffsets" + gcName][gcID, :]
        assert subset["offsetType"].ndim == 1, (
            "Error: Make sure input haloID or subhaloID is a single number, not list."
        )

    return subset


def _global_indices_zoomorig(sP, partType, origZoomID=None):
    """Helper function for TNG-Cluster to calculate indices for a global zoom original load.

    If origZoomID is None, then determine from sP.subhaloInd.
    Returns two index ranges, one for the original zoom FoFs, and one for the outer fuzz.
    """
    pt = sP.ptNum(partType)

    with h5py.File(sP.postPath + "offsets/offsets_%03d.hdf5" % sP.snap, "r") as f:
        origIDs = f["OriginalZooms/HaloIDs"][()]
        offsets = f["OriginalZooms/GroupsSnapOffsetByType"][()]
        lengths = f["OriginalZooms/GroupsTotalLengthByType"][()]

        offsets2 = f["OriginalZooms/OuterFuzzSnapOffsetByType"][()]
        lengths2 = f["OriginalZooms/OuterFuzzTotalLengthByType"][()]

    if origZoomID is None:
        origZoomID = sP.groupCatSingle(subhaloID=sP.subhaloInd)["SubhaloOrigHaloID"]

    origZoomInd = np.where(origIDs == origZoomID)[0][0]

    indRange = [offsets[origZoomInd, pt], offsets[origZoomInd, pt] + lengths[origZoomInd, pt] - 1]
    indRange2 = [offsets2[origZoomInd, pt], offsets2[origZoomInd, pt] + lengths2[origZoomInd, pt] - 1]

    return indRange, indRange2


def _haloOrSubhaloIndRange(sP, partType, haloID=None, subhaloID=None):
    """Helper."""
    subset = haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    indStart = subset["offsetType"][sP.ptNum(partType)]
    indStop = indStart + subset["lenType"][sP.ptNum(partType)]

    indRange = [indStart, indStop - 1]

    return indRange


def snapshotSubset(
    sP,
    partType,
    fields,
    inds=None,
    indRange=None,
    haloID=None,
    subhaloID=None,
    mdi=None,
    sq=True,
    haloSubset=False,
    float32=False,
):
    """For a given snapshot load one or more field(s) for one particle type.

    The four arguments ``inds``, ``indRange``, ``haloID``, and ``subhaloID`` are all optional, but
    at most one can be specified.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      partType: e.g. [0,1,2,4] or ('gas','dm','tracer','stars').
      fields: e.g. ['ParticleIDs','Coordinates','temp',...].
      inds (ndarray[int]): known indices requested, optimize the load.
      indRange (list[int][2]): same, but specify only min and max indices **(--inclusive--!)**.
      haloID (int): if input, load particles only of the specified fof halo.
      subhaloID (int): if input, load particles only of the specified subalo.
      mdi (int or None): multi-dimensional index slice load (only used in recursive calls, don't input directly)
      sq (bool): squeeze single field return into a numpy array instead of within a dict.
      haloSubset (bool): return particle subset of only those in all FoF halos (no outer fuzz).
      float32 (bool): load any float64 datatype datasets directly as float32 (optimize for memory).

    Returns:
      ndarray or dict
    """
    kwargs = {"inds": inds, "indRange": indRange, "haloID": haloID, "subhaloID": subhaloID}
    subset = None

    if (inds is not None or indRange is not None) and (haloID is not None or subhaloID is not None):
        raise Exception("Can only specify one of (inds,indRange,haloID,subhaloID).")
    if inds is not None and indRange is not None:
        raise Exception("Cannot specify both inds and indRange.")
    if haloID is not None and subhaloID is not None:
        raise Exception("Cannot specify both haloID and subhaloID.")
    if ((haloID is not None) or (subhaloID is not None)) and not sP.groupOrdered:
        raise Exception("Not yet implemented (group/halo load in non-groupordered).")
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (
        not sP.groupOrdered
        or (haloID is not None)
        or (subhaloID is not None)
        or (inds is not None)
        or (indRange is not None)
    ):
        raise Exception("haloSubset only for groupordered snapshots, and not with halo/subhalo subset.")
    if sP.snap is None:
        raise Exception("Must specify sP.snap for snapshotSubset load.")

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)

    # make sure fields is not a single element, and don't modify input
    fields = list(iterable(fields))
    fieldsOrig = list(iterable(fields))

    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)["snapOffsetsGroup"]
        indRange = [0, offsets_pt[:, sP.ptNum(partType)].max()]
        kwargs["indRange"] = indRange

    # derived particle types (i.e. subsets of snapshot PartTypeN's)
    if "_" in str(partType):
        ptSnap = partType.split("_")[0]

        # load needed fields to define subset, and set w_sel
        if partType in ["star_real", "stars_real"]:
            sftime = snapshotSubset(sP, ptSnap, "sftime", **kwargs)
            w_sel = np.where(sftime >= 0.0)
        if partType == "wind_real":
            sftime = snapshotSubset(sP, ptSnap, "sftime", **kwargs)
            w_sel = np.where(sftime < 0.0)
        if partType in ["gas_sf", "gas_eos"]:
            sfr = snapshotSubset(sP, ptSnap, "sfr", **kwargs)
            w_sel = np.where(sfr > 0.0)
        if partType == "gas_nonsf":
            sfr = snapshotSubset(sP, ptSnap, "sfr", **kwargs)
            w_sel = np.where(sfr == 0.0)

        # load requested field(s), take subset and return
        ret = snapshotSubset(sP, ptSnap, fields, **kwargs)

        if isinstance(ret, dict):
            for key in ret.keys():
                if key == "count":
                    continue
                ret[key] = ret[key][w_sel]
            ret["count"] = len(w_sel[0])
            return ret
        return ret[w_sel]  # single ndarray

    # return dict
    r = {}

    # custom field functions must handle indRange, but they do not handle inds
    kwargs_custom = dict(kwargs)
    if inds is not None:
        kwargs_custom["indRange"] = [np.min(inds), np.max(inds)]
        kwargs_custom["inds"] = None

    # handle any custom fields: these include composite or derived fields (temp, vmag, ...),
    # unit conversions (bmag_uG, ...), and custom analysis (ionic masses, ...)
    for _i, fieldName in enumerate(fields):
        # field name: take lowercase, and strip optional '_log' postfix
        takeLog = False

        if fieldName[-len("_log") :].lower() == "_log":
            fieldName = fieldName[: -len("_log")]
            takeLog = True
        field = fieldName.lower()

        # does (exact) field name exist in custom field registry?
        if field in custom_fields:
            # yes: load/compute now
            data = custom_fields[field](sP, partType, field, kwargs_custom)

            # if return is None, then this is a fall-through to a normal load
            if data is not None:
                r[fieldName] = data
        else:
            # if not, try wild-card matching for custom fields
            for search_key in custom_multi_fields:
                # requested field contains search key?
                if search_key in field:
                    r[fieldName] = custom_multi_fields[search_key](sP, partType, fieldName, kwargs_custom)

        # inds subset?
        if inds is not None and fieldName in r:
            r[fieldName] = r[fieldName][inds - np.min(inds)]

        # unit-postprocessing
        if takeLog:
            # take log?
            r[fieldName] = logZeroNaN(r[fieldName])

            # change key name
            r[fieldName + "_log"] = r.pop(fieldName)

    # done:
    if len(r) >= 1:
        # have at least one custom field, do we also have standard fields requested? if so, load them now and combine
        if len(r) < len(fields):
            standardFields = list(fields)
            for key in r.keys():
                standardFields.remove(key)

            ss = snapshotSubset(sP, partType, standardFields, sq=False, **kwargs)
            ss.update(r)
            r = ss

        # just one field in total? compress and return single ndarray (by default)
        if len(r) == 1 and sq is True:
            key = list(r.keys())[0]
            return r[key]

        return r  # return dictionary

    # alternate field names mappings
    invNameMappings = {}

    altNames = [
        [["center_of_mass", "com", "center"], "CenterOfMass"],
        [["xyz", "positions", "pos"], "Coordinates"],
        [["dens", "rho"], "Density"],
        [["dmdens"], "SubfindDMDensity"],
        [["xe", "xelec"], "ElectronAbundance"],
        [["agnrad", "gfm_agnrad"], "GFM_AGNRadiation"],
        [["coolrate", "gfm_coolrate"], "GFM_CoolingRate"],
        [["winddmveldisp"], "GFM_WindDMVelDisp"],
        [["metal", "Z", "gfm_metal", "metallicity"], "GFM_Metallicity"],
        [["metals"], "GFM_Metals"],
        [["u", "utherm"], "InternalEnergy"],
        [["machnum", "shocks_machnum"], "Machnumber"],
        [["dedt", "energydiss", "shocks_dedt", "shocks_energydiss"], "EnergyDissipation"],
        [["b", "bfield"], "MagneticField"],
        [["mass"], "Masses"],
        [["numtr"], "NumTracers"],
        [["id", "ids"], "ParticleIDs"],
        [["pot"], "Potential"],
        [["pres"], "Pressure"],
        [["sfr"], "StarFormationRate"],
        [["vel"], "Velocities"],
        [["vol"], "Volume"],
        # stars only:
        [["initialmass", "ini_mass", "mass_ini"], "GFM_InitialMass"],
        [["stellarformationtime", "sftime", "birthtime"], "GFM_StellarFormationTime"],
        [["stellarphotometrics", "stellarphot", "sphot"], "GFM_StellarPhotometrics"],
        # blackholes only:
        [["bh_dens", "bh_rho"], "BH_Density"],
    ]

    for i, field in enumerate(fields):
        for altLabels, toLabel in altNames:
            # alternate field name map, lowercase versions accepted
            if field.lower() in altLabels or field == toLabel.lower():
                # invNameMappings[toLabel] = fields[i] # save inverse so we can undo
                fields[i] = toLabel

    # handle non-GFM runs which have the same fields without the name prefix
    # e.g. GFM_InitialMass -> InitialMass (also: StellarFormationTime, Metallicity) (MCST)
    for i, field in enumerate(fields):
        if field.startswith("GFM_"):
            if not snapHasField(sP, partType, field) and snapHasField(sP, partType, field.replace("GFM_", "")):
                toLabel = field.replace("GFM_", "")
                invNameMappings[toLabel] = fields[i]
                fields[i] = toLabel

    # check for snapshots written by other codes which require minor field remappings (SWIFT)
    swiftRenames = {
        "Density": "Densities",
        "Entropy": "Entropies",
        "InternalEnergy": "InternalEnergies",
        "Pressure": "Pressures",
        "SmoothingLength": "SmoothingLengths",
    }

    if sP.simCode == "SWIFT":
        for i, field in enumerate(fields):
            if field in swiftRenames:
                fields[i] = swiftRenames[field]

    # inds based subset
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        indRange = [np.min(inds), np.max(inds)]

        val = snapshotSubset(sP, partType, fields, indRange=indRange)
        return val[inds - np.min(inds)]

    # indRange based subset
    if indRange is not None:
        # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
        subset = {
            "offsetType": np.zeros(sP.nTypes, dtype="int64"),
            "lenType": np.zeros(sP.nTypes, dtype="int64"),
            "snapOffsets": snapOffsetList(sP),
        }

        subset["offsetType"][sP.ptNum(partType)] = indRange[0]
        subset["lenType"][sP.ptNum(partType)] = indRange[1] - indRange[0] + 1

    # multi-dimensional field slicing during load
    mdi = [None] * len(fields)  # multi-dimensional index to restrict load to
    trMCFields = sP.trMCFields if sP.trMCFields else np.repeat(-1, 14)

    multiDimSliceMaps = [
        {"names": ["x", "pos_x", "posx"], "field": "Coordinates", "fN": 0},
        {"names": ["y", "pos_y", "posy"], "field": "Coordinates", "fN": 1},
        {"names": ["z", "pos_z", "posz"], "field": "Coordinates", "fN": 2},
        {"names": ["vx", "vel_x", "velx"], "field": "Velocities", "fN": 0},
        {"names": ["vy", "vel_y", "vely"], "field": "Velocities", "fN": 1},
        {"names": ["vz", "vel_z", "velz"], "field": "Velocities", "fN": 2},
        {"names": ["bx", "b_x", "bfield_x"], "field": "MagneticField", "fN": 0},
        {"names": ["by", "b_y", "bfield_y"], "field": "MagneticField", "fN": 1},
        {"names": ["bz", "b_z", "bfield_z"], "field": "MagneticField", "fN": 2},
        {"names": ["tracer_maxtemp", "maxtemp"], "field": "FluidQuantities", "fN": trMCFields[0]},
        {"names": ["tracer_maxtemp_time", "maxtemp_time"], "field": "FluidQuantities", "fN": trMCFields[1]},
        {"names": ["tracer_maxtemp_dens", "maxtemp_dens"], "field": "FluidQuantities", "fN": trMCFields[2]},
        {"names": ["tracer_maxdens", "maxdens"], "field": "FluidQuantities", "fN": trMCFields[3]},
        {"names": ["tracer_maxdens_time", "maxdens_time"], "field": "FluidQuantities", "fN": trMCFields[4]},
        {"names": ["tracer_maxmachnum", "maxmachnum"], "field": "FluidQuantities", "fN": trMCFields[5]},
        {"names": ["tracer_maxent", "maxent"], "field": "FluidQuantities", "fN": trMCFields[6]},
        {"names": ["tracer_maxent_time", "maxent_time"], "field": "FluidQuantities", "fN": trMCFields[7]},
        {"names": ["tracer_laststartime", "laststartime"], "field": "FluidQuantities", "fN": trMCFields[8]},
        {"names": ["tracer_windcounter", "windcounter"], "field": "FluidQuantities", "fN": trMCFields[9]},
        {"names": ["tracer_exchcounter", "exchcounter"], "field": "FluidQuantities", "fN": trMCFields[10]},
        {"names": ["tracer_exchdist", "exchdist"], "field": "FluidQuantities", "fN": trMCFields[11]},
        {"names": ["tracer_exchdisterr", "exchdisterr"], "field": "FluidQuantities", "fN": trMCFields[12]},
        {"names": ["tracer_shockmaxmach", "shockmaxmach"], "field": "FluidQuantities", "fN": trMCFields[13]},
        {"names": ["phot_U", "U"], "field": "GFM_StellarPhotometrics", "fN": 0},
        {"names": ["phot_B", "B"], "field": "GFM_StellarPhotometrics", "fN": 1},
        {"names": ["phot_V", "V"], "field": "GFM_StellarPhotometrics", "fN": 2},
        {"names": ["phot_K", "K"], "field": "GFM_StellarPhotometrics", "fN": 3},
        {"names": ["phot_g", "g"], "field": "GFM_StellarPhotometrics", "fN": 4},
        {"names": ["phot_r", "r"], "field": "GFM_StellarPhotometrics", "fN": 5},
        {"names": ["phot_i", "i"], "field": "GFM_StellarPhotometrics", "fN": 6},
        {"names": ["phot_z", "z"], "field": "GFM_StellarPhotometrics", "fN": 7},
        # { 'names':['metals_H', 'hydrogen'],               'field':'GFM_Metals', 'fN':0 },
        # { 'names':['metals_He','helium'],                 'field':'GFM_Metals', 'fN':1 },
        # { 'names':['metals_C', 'carbon'],                 'field':'GFM_Metals', 'fN':2 },
        # { 'names':['metals_N', 'nitrogen'],               'field':'GFM_Metals', 'fN':3 },
        # { 'names':['metals_O', 'oxygen'],                 'field':'GFM_Metals', 'fN':4 },
        # { 'names':['metals_Ne','neon'],                   'field':'GFM_Metals', 'fN':5 },
        # { 'names':['metals_Mg','magnesium'],              'field':'GFM_Metals', 'fN':6 },
        # { 'names':['metals_Si','silicon'],                'field':'GFM_Metals', 'fN':7 },
        # { 'names':['metals_Fe','iron'],                   'field':'GFM_Metals', 'fN':8 },
        # { 'names':['metals_tot','metals_total'],          'field':'GFM_Metals', 'fN':9 },
        {"names": ["metaltag_SNIa", "metals_SNIa"], "field": "GFM_MetalsTagged", "fN": 0},
        {"names": ["metaltag_SNII", "metals_SNII"], "field": "GFM_MetalsTagged", "fN": 1},
        {"names": ["metaltag_AGB", "metals_AGB"], "field": "GFM_MetalsTagged", "fN": 2},
        {"names": ["metaltag_NSNS", "metals_NSNS"], "field": "GFM_MetalsTagged", "fN": 3},
        {"names": ["metaltag_FeSNIa", "metals_FeSNIa"], "field": "GFM_MetalsTagged", "fN": 4},
        {"names": ["metaltag_FeSNII", "metals_FeSNII"], "field": "GFM_MetalsTagged", "fN": 5},
    ]

    # map species abundance names into dataset indices (varies by simulation)
    if sP.metals is not None:
        metal_list = sP.metals

        if sP.star == 1:  # sP.snapHasField(partType, 'GFM_Metals'): # slow
            z_field = "GFM_Metals"
        elif sP.star in [2, 3]:  # sP.snapHasField(partType, 'ElementFraction'):
            z_field = "ElementFraction"
            if sP.isPartType(partType, "gas"):
                metal_list = sP.metals[2:]  # omit first 2 since ElementFraction does not have H,He for gas

        for i, metal in enumerate(metal_list):
            multiDimSliceMaps.append({"names": ["metals_" + metal], "field": z_field, "fN": i})

    for i, field in enumerate(fields):
        for multiDimMap in multiDimSliceMaps:
            if field in multiDimMap["names"]:
                invNameMappings[multiDimMap["field"]] = fields[i]  # save inverse so we can undo

                fields[i] = multiDimMap["field"]
                mdi[i] = multiDimMap["fN"]
                assert mdi[i] >= 0  # otherwise e.g. not assigned in sP

    if sum(m is not None for m in mdi) > 1:
        raise Exception("Not supported for multiple MDI at once.")

    # halo or subhalo based subset
    if haloID is not None or subhaloID is not None:
        assert not sP.isPartType(
            partType, "tracer"
        )  # not group-ordered (even in TNG-Cluster, since separated by parent type)
        subset = haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    # check memory cache (only simplest support at present, for indRange/full returns of global cache)
    if len(fields) == 1 and mdi[0] is None:
        cache_key = "snap%s_%s_%s" % (sP.snap, partType, fields[0].replace(" ", "_"))
        if cache_key in sP.data:
            # global? (or rather, whatever is in sP.data... be careful)
            if indRange is None:
                print("CAUTION: Cached return [%s], and indRange is None, returning all of sP.data field." % cache_key)
                if sq:
                    return sP.data[cache_key]
                else:
                    return {fields[0]: sP.data[cache_key]}

            print("NOTE: Returning [%s] from cache, indRange [%d - %d]!" % (cache_key, indRange[0], indRange[1]))
            if sq:
                return sP.data[cache_key][indRange[0] : indRange[1] + 1]
            else:
                return {fields[0]: sP.data[cache_key][indRange[0] : indRange[1] + 1]}

    # load from disk
    r = il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset, mdi=mdi, sq=sq, float32=float32)

    # optional unit post-processing
    if isinstance(r, np.ndarray) and len(fieldsOrig) == 1:
        if fieldsOrig[0] in ["tracer_maxent", "tracer_maxtemp"] and r.max() < 20.0:
            raise Exception("Unexpectedly low max for non-log values, something maybe changed.")

        if fieldsOrig[0] == "tracer_maxent":
            r = sP.units.tracerEntToCGS(r, log=True)  # [log cgs] = [log K cm^2]
        if fieldsOrig[0] == "tracer_maxtemp":
            r = logZeroNaN(r)  # [log Kelvin]

    # SWIFT: add little h (and/or little a) units back into particle fields as needed to match TNG/AREPO conventions
    if sP.simCode == "SWIFT":
        swiftFieldsH = {"Coordinates": 1, "Masses": 1, "Densities": 2, "InternalEnergies": 0, "SmoothingLengths": 1}
        # ,'Velocities':1.0/np.sqrt(sP.scalefac)} # generalize below, pull out first arg of np.power()

        if isinstance(r, np.ndarray) and len(fields) == 1 and fields[0] in swiftFieldsH:
            r *= np.power(sP.snapshotHeader()["h"][0], swiftFieldsH[fields[0]])
        else:
            for field in fields:
                if field in swiftFieldsH:
                    r[field] *= np.power(sP.snapshotHeader()["h"][0], swiftFieldsH[field])

        for field in fields:
            if field not in swiftFieldsH:
                raise Exception("Should fix h-units for consistency.")

    # inverse map multiDimSliceMaps such that return dict has key names exactly as requested
    # todo: could also do for altNames (just uncomment above, but need to refactor codebase)
    if isinstance(r, dict):
        for newLabel, origLabel in invNameMappings.items():
            r[origLabel] = r.pop(newLabel)  # change key label

    return r


def _parallel_load_func(sP, partType, field, indRangeLoad, indRangeSave, float32, shared_mem_array, dtype, shape):
    """Multiprocessing target, which calls snapshotSubset() and writes the result directly into a shared memory array.

    Always called with only one field.
    NOTE: sP has been pickled and shared between sub-processes (sP.data is likely common, careful!).
    """
    data = sP.snapshotSubset(partType, field, indRange=indRangeLoad, sq=True, float32=float32)

    numpy_array_view = np.frombuffer(shared_mem_array, dtype).reshape(shape)

    numpy_array_view[indRangeSave[0] : indRangeSave[1]] = data

    # note: could move this into il.snapshot.loadSubset() following the strategy of the
    # parallel groupCat() load, to actually avoid this intermediate memory usage


def snapshotSubsetParallel(
    sP,
    partType,
    fields,
    inds=None,
    indRange=None,
    haloID=None,
    subhaloID=None,
    sq=True,
    haloSubset=False,
    float32=False,
    nThreads=8,
):
    """Parallel version of :py:func:`snapshotSubset` that uses concurrent python+h5py reader processes."""
    import ctypes
    import traceback
    from functools import partial

    # enable global logging of multiprocessing to stderr:
    # logger = mp.log_to_stderr()
    # logger.setLevel(mp.SUBDEBUG)

    # method to disable parallel loading, which does not work with custom-subset cached fields
    # inside sP.data since indRange as computed below cannot know about this
    if "nThreads" in sP.data and sP.data["nThreads"] == 1:
        return snapshotSubset(
            sP,
            partType,
            fields,
            inds=inds,
            indRange=indRange,
            haloID=haloID,
            subhaloID=subhaloID,
            sq=sq,
            haloSubset=haloSubset,
            float32=float32,
        )

    # sanity checks
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (not sP.groupOrdered or (indRange is not None) or (inds is not None)):
        raise Exception("haloSubset only for groupordered snapshots, and not with indRange subset.")

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)
    fields = list(iterable(fields))

    # get total size
    h = sP.snapshotHeader()
    numPartTot = h["NumPart"][sP.ptNum(partType)]

    if numPartTot == 0:
        return {"count": 0}

    # low particle count (e.g. below ~1e7x4bytes=40MB there is little point) use serial
    serial = False
    minParallelCount = 1e7  # nThreads*10

    if (
        numPartTot < minParallelCount
        or (inds is not None and inds.size < minParallelCount)
        or (indRange is not None and (indRange[1] - indRange[0]) < minParallelCount)
    ):
        serial = True

    if not serial:
        # detect if we are already fetching data inside a parallelized load, and don't propagate
        stack = traceback.extract_stack(limit=6)
        serial = np.any(["_parallel_load_func" in frame.name for frame in stack])
        if serial and getuser() != "wwwrun":
            print("NOTE: Detected parallel-load request inside parallel-load, making serial.")

    if not serial:
        # detect if we are inside a daemonic child process already (e.g. multiprocessing spawned)
        # in which case we cannot start further child processes, so revert to serial load
        serial = mp.current_process().name != "MainProcess"
        if serial and getuser() != "wwwrun":
            print("NOTE: Detected parallel-load request inside daemonic child, making serial.")

    if not serial:
        # detect if we are requesting any field which, by default, spawns a highly threaded computation
        for field in fields:
            if field in ["baryon_frac", "f_b"]:
                print("NOTE: Detected parallel-load request for threaded calculation, making serial.")
                serial = False

    if serial:
        return snapshotSubset(
            sP,
            partType,
            fields,
            inds=inds,
            indRange=indRange,
            haloID=haloID,
            subhaloID=subhaloID,
            sq=sq,
            haloSubset=haloSubset,
            float32=float32,
        )

    # set indRange to load
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        assert indRange is None
        indRange = [inds.min(), inds.max()]

    if haloID is not None or subhaloID is not None:
        # convert halo or subhalo request into a particle index range
        assert indRange is None
        indRange = _haloOrSubhaloIndRange(sP, partType, haloID=haloID, subhaloID=subhaloID)

    if indRange is None:
        indRange = [0, numPartTot - 1]
    else:
        numPartTot = indRange[1] - indRange[0] + 1

    if numPartTot == 0:
        return {"count": 0}

    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)["snapOffsetsGroup"]
        indRange = [0, offsets_pt[:, sP.ptNum(partType)].max()]

    # get shape and dtype by loading one element
    sample = snapshotSubset(sP, partType, fields, indRange=[0, 0], sq=False, float32=float32)

    # prepare return
    r = {}

    # do different fields in a loop, if more than one requested
    for k in sample.keys():
        if k == "count":
            continue

        # prepare shape
        shape = [numPartTot]

        if sample[k].ndim > 1:
            shape.append(sample[k].shape[1])  # i.e. Coordinates, append 3 as second dimension

        # allocate global return
        size = int(np.prod(shape) * sample[k].dtype.itemsize)  # bytes
        ctype = ctypes.c_byte

        shared_mem_array = mp.sharedctypes.RawArray(ctype, size)
        numpy_array_view = np.frombuffer(shared_mem_array, sample[k].dtype).reshape(shape)

        # spawn processes with indRange subsets
        offset = 0
        processes = []

        for i in range(nThreads):
            indRangeLoad = pSplitRange(indRange, nThreads, i, inclusive=True)

            numLoadLoc = indRangeLoad[1] - indRangeLoad[0] + 1
            indRangeSave = [offset, offset + numLoadLoc]
            offset += numLoadLoc

            args = (sP, partType, k, indRangeLoad, indRangeSave, float32, shared_mem_array, sample[k].dtype, shape)

            p = mp.Process(target=_parallel_load_func, args=args)
            processes.append(p)

        # wrap in try, to help avoid zombie processes and system issues
        try:
            for p in processes:
                p.start()
        finally:
            for p in processes:
                p.join()
                # if exitcode == -9, then a sub-process was killed by the oom-killer, not good
                # (return will be corrupt/incompleted)
                assert p.exitcode == 0, "Insufficient memory for requested parallel load."

        if 0:
            # diagnostic
            nMallocs = mp.heap.BufferWrapper._heap._n_mallocs
            nFrees = mp.heap.BufferWrapper._heap._n_frees
            print("mp heap: nMallocs = %d, nFrees = %d" % (nMallocs, nFrees))
            for i, arena in enumerate(mp.heap.BufferWrapper._heap._arenas):
                print("mp arena[%d] size = %d (%.1f GB)" % (i, arena.size, arena.size / 1024**3))

        # add into dict
        if inds is not None:
            r[k] = numpy_array_view[inds - inds.min()]
        else:
            r[k] = numpy_array_view

    if len(r) == 1 and sq:
        # single ndarray return
        return r[list(r.keys())[0]]

    r["count"] = numPartTot
    return r


def snapshotSubsetLoadIndicesChunked(sP, partType, field, inds, sq=True, verbose=False):
    """Memory-efficient load of a subset of a snapshot field, as specified by a set of particle indices.

    If we only want to load a set of inds, and this is a small fraction of the
    total snapshot, then we do not ever need to do a global load or allocation, thus
    reducing the peak memory usage during load by a factor of nChunks or
    sP.numPart[partType]/inds.size, whichever is smaller. Note: currently only for
    a single field, could be generalized to multiple fields. Note: this effectively
    captures the multiblock I/O strategy of the previous codebase as well, with only
    a small efficiency loss since we do not exactly compute bounding local indRanges
    for contiguous index subsets, but rather process nChunks discretely.
    """
    numPartTot = sP.numPart[sP.ptNum(partType)]

    if verbose:
        ind_frac = inds.size / numPartTot * 100
        mask = np.zeros(inds.size)  # debugging only
        print("Loading [%s, %s], indices cover %.3f%% of snapshot total." % (partType, field, ind_frac))

    nChunks = 20

    # get shape and dtype by loading one element
    sample = sP.snapshotSubset(partType, field, indRange=[0, 0], sq=False)

    fieldName = list(sample.keys())[-1]
    assert fieldName != "count"  # check order guarantee

    sample = sample[fieldName]

    shape = [inds.size] if sample.ndim == 1 else [inds.size, sample.shape[1]]  # [N] or e.g. [N,3]

    # allocate
    data = np.zeros(shape, dtype=sample.dtype)

    # sort requested indices, to ease intersection with each indRange_loc
    sort_inds = np.argsort(inds)
    sorted_inds = inds[sort_inds]

    # chunk load
    for i in range(nChunks):
        if verbose:
            print(" %d%%" % (float(i) / nChunks * 100), end="", flush=True)

        indRange_loc = pSplitRange([0, numPartTot - 1], nChunks, i, inclusive=True)

        if indRange_loc[0] > sorted_inds.max() or indRange_loc[1] < sorted_inds.min():
            continue

        # which of the input indices are covered by this local indRange?
        ind0 = np.searchsorted(sorted_inds, indRange_loc[0], side="left")
        ind1 = np.searchsorted(sorted_inds, indRange_loc[1], side="right")

        if ind0 == ind1:
            continue

        # parallel load
        data_loc = sP.snapshotSubsetP(partType, field, indRange=indRange_loc)

        # sort_inds[ind0:ind1] gives us which inds are in this data_loc
        # the entires in data_loc are sorted_inds[ind0:ind1]-indRange_loc[0]
        stamp_inds = sort_inds[ind0:ind1]
        take_inds = sorted_inds[ind0:ind1] - indRange_loc[0]

        data[stamp_inds] = data_loc[take_inds]

        if verbose:
            mask[stamp_inds] += 1  # debugging only

    if verbose:
        assert mask.min() == 1 and mask.max() == 1
        print("")

    if sq:  # raw ndarray
        return data

    # wrap in dictionary with key equal to snapshot field name
    r = {fieldName: data}
    r["count"] = inds.size

    return r
