"""
Conversion of data (snapshots/catalogs) between different cosmological simulations.
"""

import glob
import struct
import time
from os import mkdir, path
from os.path import expanduser, isdir

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import map_coordinates

from ..cosmo.hydrogen import neutral_fraction
from ..plot.config import figsize
from ..util.helper import isUnique, rootPath
from ..util.match import match
from ..util.simParams import simParams


def convertGadgetICsToHDF5(aip=False):
    """Convert a Gadget-1/2 binary format ICs (dm-only, only pos/vel/IDs) into HDF5 format (keep original ordering)."""
    ptNum = 1
    longids = True  # 64 bit, else 32 bit

    loadPath = "/virgo/simulations/IllustrisTNG/InitialConditions/L35n2160TNG/output/ICs.%s"
    savePath = "/virgo/simulations/IllustrisTNG/L35n2160TNG/output/snap_ics.hdf5"

    if aip:
        loadPath = "/u/dnelson/sims.TNG/L500n512TNG_DM/ICs/"
        loadPath += "CF3_082719_BGc_CMB_RGmi0p0_err250_Rmax150_512_500_15545_z69.gadget"
        savePath = "/u/dnelson/sims.TNG/L500n512TNG_DM/ICs/ICs_CF3_15545.hdf5"

        longids = False

    # read header of first snapshot chunk
    nChunks = len(glob.glob(loadPath.replace("%s", "*")))
    print("Found [%d] chunks, loading..." % nChunks)

    fileName = loadPath % 0 if nChunks > 1 else loadPath
    with open(fileName, "rb") as f:
        header = f.read(260)

    if aip:
        # CLUES/CF3 ICs from Wang Peng/AIP, seems non-standard
        npart = struct.unpack("iiiiii", header[20 : 20 + 6 * 4])[ptNum]
        masstable = struct.unpack("dddddd", header[44 : 44 + 48])
        scalefac = struct.unpack("d", header[92 : 92 + 8])[0]
        redshift = struct.unpack("d", header[100 : 100 + 8])[0]
        # FlagSfr   = struct.unpack('h', header[108:108+2])[0]
        # FlagFB    = struct.unpack('h', header[110:110+2])[0]
        # dummy     = struct.unpack('i', header[112:112+4])[0]
        nPartTot = struct.unpack("iiiiii", header[116 : 116 + 6 * 4])[ptNum]
        # dummy     = struct.unpack('i', header[140:140+4])[0]
        nFiles = struct.unpack("h", header[144 : 144 + 2])[0]
        # dummy     = struct.unpack('h', header[146:146+2])[0]
        BoxSize = struct.unpack("d", header[148 : 148 + 8])[0]
        Omega0 = struct.unpack("d", header[156 : 156 + 8])[0]
        OmegaL = struct.unpack("d", header[164 : 164 + 8])[0]
        Hubble = struct.unpack("d", header[172 : 172 + 8])[0]
        # then: 48*2 bytes dummy
    else:
        # standard Gadget-2 header structure
        npart = struct.unpack("iiiiii", header[4 : 4 + 24])[ptNum]
        masstable = struct.unpack("dddddd", header[28 : 28 + 48])
        scalefac = struct.unpack("d", header[76 : 76 + 8])[0]
        redshift = struct.unpack("d", header[84 : 84 + 8])[0]
        # FlagSfr   = struct.unpack('i', header[92:92+4])[0]
        # FlagFB    = struct.unpack('i', header[96:96+4])[0]
        # nPartTot   = struct.unpack('iiiiii', header[100:100+24])[ptNum]
        # FlagCool  = struct.unpack('i', header[124:124+4])[0]
        nFiles = struct.unpack("i", header[128 : 128 + 4])[0]
        BoxSize = struct.unpack("d", header[132 : 132 + 8])[0]
        Omega0 = struct.unpack("d", header[140 : 140 + 8])[0]
        OmegaL = struct.unpack("d", header[148 : 148 + 8])[0]
        Hubble = struct.unpack("d", header[156 : 156 + 8])[0]

    assert nFiles == nChunks

    if longids:
        ids_type = "q"
        ids_size = 8
        ids_dtype = "int64"
    else:
        ids_type = "i"
        ids_size = 4
        ids_dtype = "int32"

    # nPartTot is wrong, has no highword, so read and accumulate manually
    if not aip:
        nPartTot = 0
        for i in range(nChunks):
            fileName = loadPath % i if nChunks > 1 else loadPath
            with open(fileName, "rb") as f:
                header = f.read(28)
                npart = struct.unpack("iiiiii", header[4 : 4 + 24])[ptNum]
            nPartTot += npart
        print("Found new nPartTot [%d]" % nPartTot)

    # open file for writing
    fOut = h5py.File(savePath, "w")

    # write header
    header = fOut.create_group("Header")
    numPartTot = np.zeros(6, dtype="int64")
    numPartTot[ptNum] = nPartTot
    header.attrs["BoxSize"] = BoxSize
    header.attrs["HubbleParam"] = Hubble
    header.attrs["MassTable"] = np.array(masstable, dtype="float64")
    header.attrs["NumFilesPerSnapshot"] = 1
    header.attrs["NumPart_ThisFile"] = numPartTot
    header.attrs["NumPart_Total"] = numPartTot
    header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype="int64")
    header.attrs["Omega0"] = Omega0
    header.attrs["OmegaLambda"] = OmegaL
    header.attrs["Redshift"] = redshift
    header.attrs["Time"] = scalefac
    if aip:
        # add flags needed to actually start an AREPO run
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 0

    # create group and datasets
    pt = fOut.create_group("PartType%d" % ptNum)

    particle_pos = pt.create_dataset("Coordinates", (nPartTot, 3), dtype="float32")
    particle_vel = pt.create_dataset("Velocities", (nPartTot, 3), dtype="float32")
    particle_ids = pt.create_dataset("ParticleIDs", (nPartTot,), dtype=ids_dtype)

    if masstable[ptNum] == 0:
        particle_mass = pt.create_dataset("Masses", (nPartTot,), dtype="float32")

    # load all snapshot IDs
    offset = 0

    for i in range(nChunks):
        # full read
        fileName = loadPath % i if nChunks > 1 else loadPath
        with open(fileName, "rb") as f:
            data = f.read()

        # local particle counts
        if aip:
            npart_local = struct.unpack("iiiiii", data[20 : 20 + 24])[ptNum]
            start_off = 276
            skip_off = 24  # 2 byte recordmarker + 22 bytes of something
        else:
            npart_local = struct.unpack("iiiiii", data[4 : 4 + 24])[ptNum]
            start_off = 260
            skip_off = 8

        # cast and save
        start_pos = start_off + 1 * skip_off + 0 * npart_local
        start_vel = start_off + 2 * skip_off + 12 * npart_local
        start_ids = start_off + 3 * skip_off + 24 * npart_local
        start_mass = start_off + 4 * skip_off + (24 + ids_size) * npart_local

        pos = struct.unpack("f" * npart_local * 3, data[start_pos : start_pos + npart_local * 12])
        vel = struct.unpack("f" * npart_local * 3, data[start_vel : start_vel + npart_local * 12])
        ids = struct.unpack(ids_type * npart_local * 1, data[start_ids : start_ids + npart_local * ids_size])

        if masstable[ptNum] == 0:
            mass = struct.unpack("f" * npart_local * 1, data[start_mass : start_mass + npart_local * 4])

        # write
        particle_pos[offset : offset + npart_local, :] = np.reshape(pos, (npart_local, 3))
        particle_vel[offset : offset + npart_local, :] = np.reshape(vel, (npart_local, 3))
        particle_ids[offset : offset + npart_local] = ids

        if masstable[ptNum] == 0:
            particle_mass[offset : offset + npart_local] = mass

        print(
            "[%3d] Snap chunk has [%8d] particles, from [%10d] to [%10d]."
            % (i, npart_local, offset, offset + npart_local)
        )
        offset += npart_local

    fOut.close()
    print("All done.")


def convertMillenniumSubhaloCatalog(snap=63):
    """Convert a subhalo catalog ('sub_tab_NNN.X' files), custom binary format of Millennium to Illustris-like HDF5."""
    savePath = path.expanduser("~") + "/sims.other/Millennium-1/output/"
    loadPath = "/virgo/simulations/Millennium/"

    dm_particle_mass = 0.0860657  # ~10^9 msun

    if not path.isdir(savePath + "groups_%03d" % snap):
        mkdir(savePath + "groups_%03d" % snap)

    def _chunkPath(snap, chunkNum):
        return loadPath + "postproc_%03d/sub_tab_%03d.%s" % (snap, snap, chunkNum)

    def _groupChunkPath(snap, chunkNum):
        return loadPath + "snapdir_%03d/group_tab_%03d.%s" % (snap, snap, chunkNum)

    nChunks = len(glob.glob(_chunkPath(snap, "*")))

    # no catalog? (snapshot <= 4) write empty catalog
    if nChunks == 0:
        print("Found [%d] chunks for snapshot[%03d]! Writing empty catalog!" % (nChunks, snap))
        # save into single hdf5
        with h5py.File(savePath + "groups_%03d/fof_subhalo_tab_%03d.hdf5" % (snap, snap), "w") as f:
            # header
            header = f.create_group("Header")
            header.attrs["Ngroups_ThisFile"] = 0
            header.attrs["Ngroups_Total"] = 0
            header.attrs["Nids_ThisFile"] = 0
            header.attrs["Nids_Total"] = 0
            header.attrs["Nsubgroups_ThisFile"] = 0
            header.attrs["Nsubgroups_Total"] = 0
            header.attrs["NumFiles"] = 1
        return

    print("Found [%d] chunks for snapshot [%03d], loading..." % (nChunks, snap))

    # reader header of first chunk
    with open(_chunkPath(snap, 0), "rb") as f:
        header = f.read(4 * 5)

    NGroups = struct.unpack("i", header[0:4])[0]
    #NIds = struct.unpack("i", header[4:8])[0]
    TotNGroups = struct.unpack("i", header[8:12])[0]
    NFiles = struct.unpack("i", header[12:16])[0]
    NSubs = struct.unpack("i", header[16:20])[0]

    # detect big-endianness
    endian = "@"  # native = little
    if NFiles != nChunks:
        TotNGroups = struct.unpack(">i", header[8:12])[0]
        NFiles = struct.unpack(">i", header[12:16])[0]
        endian = ">"

    assert NFiles == nChunks

    # no TotNSubs stored...
    TotNSubs = 0
    TotNGroupsCheck = 0

    for i in range(nChunks):
        with open(_chunkPath(snap, i), "rb") as f:
            header = f.read(4 * 5)
            NGroups = struct.unpack(endian + "i", header[0:4])[0]
            NSubs = struct.unpack(endian + "i", header[16:20])[0]

            TotNSubs += NSubs
            TotNGroupsCheck += NGroups

    print("Total: [%d] groups, [%d] subhalos, reading..." % (TotNGroups, TotNSubs))

    assert TotNGroupsCheck == TotNGroups

    # allocate (group files)
    GroupLen = np.zeros(TotNGroups, dtype="int32")
    GroupOffset = np.zeros(TotNGroups, dtype="int32")

    # load (group files)
    g_off = 0

    for i in range(nChunks):
        if i % int(nChunks / 10) == 0:
            print(" %d%%" % np.ceil(float(i) / nChunks * 100), end="")
        # full read
        with open(_groupChunkPath(snap, i), "rb") as f:
            data = f.read()

        # header (object counts)
        NGroups = struct.unpack(endian + "i", data[0:4])[0]

        GroupLen[g_off : g_off + NGroups] = struct.unpack(endian + "i" * NGroups, data[16 : 16 + 4 * NGroups])
        GroupOffset[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[16 + 4 * NGroups : 16 + 8 * NGroups]
        )

        g_off += NGroups

    print(" done with group files.")

    # allocate (subhalo files)
    NSubsPerHalo = np.zeros(TotNGroups, dtype="int32")
    FirstSubOfHalo = np.zeros(TotNGroups, dtype="int32")

    SubLen = np.zeros(TotNSubs, dtype="int32")
    SubOffset = np.zeros(TotNSubs, dtype="int32")
    SubParentHalo = np.zeros(TotNSubs, dtype="int32")
    SubFileNr = np.zeros(TotNSubs, dtype="int16")
    SubLocalIndex = np.zeros(TotNSubs, dtype="int32")

    Halo_M_Mean200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_Mean200 = np.zeros(TotNGroups, dtype="float32")
    Halo_M_Crit200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_Crit200 = np.zeros(TotNGroups, dtype="float32")
    Halo_M_TopHat200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_TopHat200 = np.zeros(TotNGroups, dtype="float32")

    SubPos = np.zeros((TotNSubs, 3), dtype="float32")
    SubVel = np.zeros((TotNSubs, 3), dtype="float32")
    SubVelDisp = np.zeros(TotNSubs, dtype="float32")
    SubVmax = np.zeros(TotNSubs, dtype="float32")
    SubSpin = np.zeros((TotNSubs, 3), dtype="float32")
    SubMostBoundID = np.zeros(TotNSubs, dtype="int64")
    SubHalfMass = np.zeros(TotNSubs, dtype="float32")

    # load (subhalo files)
    s_off = 0  # subhalos
    g_off = 0  # groups

    for i in range(nChunks):
        if i % int(nChunks / 10) == 0:
            print(" %d%%" % np.ceil(float(i) / nChunks * 100), end="")
        # full read
        with open(_chunkPath(snap, i), "rb") as f:
            data = f.read()

        # header (object counts)
        NGroups = struct.unpack(endian + "i", data[0:4])[0]
        NSubs = struct.unpack(endian + "i", data[16:20])[0]

        # per halo
        header_bytes = 20
        off = header_bytes
        NSubsPerHalo[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 0 * NGroups : off + 4 * NGroups]
        )
        FirstSubOfHalo[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 4 * NGroups : off + 8 * NGroups]
        )
        FirstSubOfHalo[g_off : g_off + NGroups] += s_off  # as stored is local to chunk files

        # per subhalo
        off = header_bytes + 8 * NGroups
        SubLen[s_off : s_off + NSubs] = struct.unpack(endian + "i" * NSubs, data[off + 0 * NSubs : off + 4 * NSubs])
        SubOffset[s_off : s_off + NSubs] = struct.unpack(endian + "i" * NSubs, data[off + 4 * NSubs : off + 8 * NSubs])
        SubParentHalo[s_off : s_off + NSubs] = struct.unpack(
            endian + "i" * NSubs, data[off + 8 * NSubs : off + 12 * NSubs]
        )
        SubParentHalo[s_off : s_off + NSubs] += g_off  # as stored is local to chunk files

        # per subhalo chunk-pointer information
        SubFileNr[s_off : s_off + NSubs] = i
        SubLocalIndex[s_off : s_off + NSubs] = np.arange(NSubs)

        # per halo
        off = header_bytes + 8 * NGroups + 12 * NSubs
        Halo_M_Mean200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 0 * NGroups : off + 4 * NGroups]
        )
        Halo_R_Mean200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 4 * NGroups : off + 8 * NGroups]
        )
        Halo_M_Crit200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 8 * NGroups : off + 12 * NGroups]
        )
        Halo_R_Crit200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 12 * NGroups : off + 16 * NGroups]
        )
        Halo_M_TopHat200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 16 * NGroups : off + 20 * NGroups]
        )
        Halo_R_TopHat200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 20 * NGroups : off + 24 * NGroups]
        )

        # per subhalo
        off = header_bytes + 8 * NGroups + 12 * NSubs + 24 * NGroups
        SubPos[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 0 * NSubs : off + 12 * NSubs]), (NSubs, 3)
        )
        SubVel[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 12 * NSubs : off + 24 * NSubs]), (NSubs, 3)
        )
        SubVelDisp[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 24 * NSubs : off + 28 * NSubs]
        )
        SubVmax[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 28 * NSubs : off + 32 * NSubs]
        )
        SubSpin[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 32 * NSubs : off + 44 * NSubs]), (NSubs, 3)
        )
        SubMostBoundID[s_off : s_off + NSubs] = struct.unpack(
            endian + "q" * 1 * NSubs, data[off + 44 * NSubs : off + 52 * NSubs]
        )
        SubHalfMass[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 52 * NSubs : off + 56 * NSubs]
        )

        # should have read entire file
        off = header_bytes + 8 * NGroups + 12 * NSubs + 24 * NGroups + 56 * NSubs
        assert off == len(data)

        g_off += NGroups
        s_off += NSubs

    # sanity checks
    assert SubLen.min() >= 20
    assert SubOffset.min() >= 0
    assert SubParentHalo.min() >= 0
    assert SubPos.min() >= 0.0 and SubPos.max() <= 500.0
    assert SubMostBoundID.min() >= 0

    # group (mass) ordering is LOCAL to each chunk! need a global sort and shuffle of all fields
    g_globalSort = np.argsort(-GroupLen, kind="mergesort")  # negative -> descending order, stable

    GroupLen_Orig = np.array(GroupLen)  # keep copy

    # re-order group properties
    GroupLen = GroupLen[g_globalSort]
    NSubsPerHalo = NSubsPerHalo[g_globalSort]
    FirstSubOfHalo = FirstSubOfHalo[g_globalSort]
    Halo_M_Mean200 = Halo_M_Mean200[g_globalSort]
    Halo_R_Mean200 = Halo_R_Mean200[g_globalSort]
    Halo_M_Crit200 = Halo_M_Crit200[g_globalSort]
    Halo_R_Crit200 = Halo_R_Crit200[g_globalSort]
    Halo_M_TopHat200 = Halo_M_TopHat200[g_globalSort]
    Halo_R_TopHat200 = Halo_R_TopHat200[g_globalSort]

    # fix subhalo parent indices, determine new subhalo ordering
    g_globalSortInv = np.zeros(g_globalSort.size, dtype="int32")
    g_globalSortInv[g_globalSort] = np.arange(g_globalSort.size)
    SubParentHalo = g_globalSortInv[SubParentHalo]

    s_globalSort = np.argsort(SubParentHalo, kind="mergesort")

    # check new subhalo ordering
    SubParentHalo_check = np.zeros(SubParentHalo.size, dtype="int32")
    for i in range(TotNGroups):
        if NSubsPerHalo[i] == 0:
            continue
        SubParentHalo_check[FirstSubOfHalo[i] : FirstSubOfHalo[i] + NSubsPerHalo[i]] = i
    assert np.array_equal(SubParentHalo, SubParentHalo_check)

    # fix first subhalo indices
    s_globalSortInv = np.zeros(s_globalSort.size, dtype="int32")
    s_globalSortInv[s_globalSort] = np.arange(s_globalSort.size)

    w_past = np.where(FirstSubOfHalo >= s_globalSort.size)
    assert np.sum(NSubsPerHalo[w_past]) == 0
    FirstSubOfHalo[w_past] = -1

    FirstSubOfHalo = s_globalSortInv[FirstSubOfHalo]

    # SubLen_Orig = np.array(SubLen)

    # re-order subhalo properties
    SubParentHalo = SubParentHalo[s_globalSort]
    SubLen = SubLen[s_globalSort]
    SubPos = SubPos[s_globalSort, :]
    SubVel = SubVel[s_globalSort, :]
    SubVelDisp = SubVelDisp[s_globalSort]
    SubVmax = SubVmax[s_globalSort]
    SubSpin = SubSpin[s_globalSort]
    SubMostBoundID = SubMostBoundID[s_globalSort]
    SubHalfMass = SubHalfMass[s_globalSort]

    # sanity checks
    assert FirstSubOfHalo[0] == 0
    w = np.where(NSubsPerHalo > 0)
    assert np.array_equal(SubParentHalo[FirstSubOfHalo[w]], w[0])
    assert GroupLen[0] == GroupLen.max()
    w = np.where(NSubsPerHalo == 0)
    if len(w[0]):
        a, b = match(w[0], SubParentHalo)
        assert a is None and b is None
        FirstSubOfHalo[w] = -1  # convention

    # GroupOffset/SubhaloOffset are local to chunk files, just recreate now with global offsets
    subgroupCount = 0

    snapOffsetsGroup = np.zeros(TotNGroups, dtype="int64")
    snapOffsetsSubhalo = np.zeros(TotNSubs, dtype="int64")

    snapOffsetsGroup[1:] = np.cumsum(GroupLen)[:-1]

    for k in np.arange(TotNGroups):
        # subhalo offsets depend on group (to allow fuzz)
        if NSubsPerHalo[k] > 0:
            snapOffsetsSubhalo[subgroupCount] = snapOffsetsGroup[k]

            subgroupCount += 1
            for _m in np.arange(1, NSubsPerHalo[k]):
                snapOffsetsSubhalo[subgroupCount] = snapOffsetsSubhalo[subgroupCount - 1] + SubLen[subgroupCount - 1]
                subgroupCount += 1

    # create a mapping of original (FileNr, SubLocalIndex) -> new index, to go from MPA trees to these group catalogs
    offset = 0
    FileNr_len = np.zeros(nChunks, dtype="int32")
    FileNr_off = np.zeros(nChunks, dtype="int32")
    new_index = np.zeros(TotNSubs, dtype="int32")

    for i in range(nChunks):
        w = np.where(SubFileNr == i)
        inds = s_globalSortInv[w]

        FileNr_len[i] = inds.size
        new_index[offset : offset + inds.size] = inds
        offset += inds.size

    FileNr_off[1:] = np.cumsum(FileNr_len)[:-1]

    with h5py.File(savePath + "groups_%03d/original_order_%03d.hdf5" % (snap, snap), "w") as f:
        f["Group_Reorder"] = g_globalSort
        f["Subhalo_Reorder"] = s_globalSort

        # e.g. the subhalo is at f['NewIndex'][ f['NewIndex_FileNrOffset'][FileNr] + SubhaloIndex ]
        f["NewIndex_FileNrOffset"] = FileNr_off
        f["NewIndex"] = new_index

    # create original GroupOffset to help particle rearrangement
    snapOffsetsGroup_Orig = np.zeros(TotNGroups, dtype="int64")
    snapOffsetsGroup_Orig[1:] = np.cumsum(GroupLen_Orig)[:-1]

    with h5py.File(savePath + "gorder_%d.hdf5" % snap, "w") as f:
        f["GroupLen_Orig"] = GroupLen_Orig
        f["GroupOffset_Orig"] = snapOffsetsGroup_Orig
        f["Group_Reorder"] = g_globalSort

    # save into single hdf5
    with h5py.File(savePath + "groups_%03d/fof_subhalo_tab_%03d.hdf5" % (snap, snap), "w") as f:
        # header
        header = f.create_group("Header")
        header.attrs["Ngroups_ThisFile"] = np.int32(TotNGroups)
        header.attrs["Ngroups_Total"] = np.int32(TotNGroups)
        header.attrs["Nids_ThisFile"] = np.int32(0)
        header.attrs["Nids_Total"] = np.int32(0)
        header.attrs["Nsubgroups_ThisFile"] = np.int32(TotNSubs)
        header.attrs["Nsubgroups_Total"] = np.int32(TotNSubs)
        header.attrs["NumFiles"] = np.int32(1)

        # groups
        groups = f.create_group("Group")
        groups["GroupFirstSub"] = FirstSubOfHalo
        groups["GroupLen"] = GroupLen
        groups["GroupMass"] = np.array(GroupLen * dm_particle_mass, dtype="float32")
        groups["GroupNsubs"] = NSubsPerHalo
        groups["Group_M_Crit200"] = Halo_M_Crit200
        groups["Group_R_Crit200"] = Halo_R_Crit200
        groups["Group_M_Mean200"] = Halo_M_Mean200
        groups["Group_R_Mean200"] = Halo_R_Mean200
        groups["Group_M_TopHat200"] = Halo_M_TopHat200
        groups["Group_R_TopHat200"] = Halo_R_TopHat200

        GroupLenType = np.zeros((GroupLen.size, 6), dtype=GroupLen.dtype)
        GroupMassType = np.zeros((GroupLen.size, 6), dtype="float32")
        GroupLenType[:, 1] = GroupLen
        GroupMassType[:, 1] = GroupLen * dm_particle_mass
        groups["GroupLenType"] = GroupLenType
        groups["GroupMassType"] = GroupMassType

        # subhalos
        subs = f.create_group("Subhalo")
        subs["SubhaloGrNr"] = SubParentHalo
        subs["SubhaloHalfmassRad"] = SubHalfMass
        subs["SubhaloIDMostbound"] = SubMostBoundID
        subs["SubhaloLen"] = SubLen
        subs["SubhaloMass"] = np.array(SubLen * dm_particle_mass, dtype="float32")
        subs["SubhaloPos"] = SubPos
        subs["SubhaloSpin"] = SubSpin
        subs["SubhaloVel"] = SubVel
        subs["SubhaloVelDisp"] = SubVelDisp
        subs["SubhaloVmax"] = SubVmax

        SubhaloLenType = np.zeros((SubLen.size, 6), dtype=SubLen.dtype)
        SubhaloMassType = np.zeros((SubLen.size, 6), dtype="float32")
        SubhaloLenType[:, 1] = SubLen
        SubhaloMassType[:, 1] = SubLen * dm_particle_mass
        groups["SubhaloLenType"] = SubhaloLenType
        groups["SubhaloMassType"] = SubhaloMassType

        # offsets (inside group files, similar to Illustris public data release, for convenience)
        offs = f.create_group("Offsets")
        offs["Group_Snap"] = snapOffsetsGroup
        offs["Subhalo_Snap"] = snapOffsetsSubhalo

    # return
    print(" All Done.")


def convertMillennium2SubhaloCatalog(snap=67):
    """Convert a subhalo catalog ('subhalo_tab_NNN.X' files), custom binary format of Millennium-2 to TNG-like HDF5."""
    savePath = path.expanduser("~") + "/sims.other/Millennium-2/output/"
    loadPath = "/virgo/simulations/Millennium2/BigRun/"

    header_bytes = 32

    if not path.isdir(savePath + "groups_%03d" % snap):
        mkdir(savePath + "groups_%03d" % snap)

    def _chunkPath(snap, chunkNum):
        return loadPath + "groups_%03d/subhalo_tab_%03d.%s" % (snap, snap, chunkNum)

    nChunks = len(glob.glob(_chunkPath(snap, "*")))

    print("Found [%d] chunks for snapshot [%03d], loading..." % (nChunks, snap))

    # reader header of first chunk
    with open(_chunkPath(snap, 0), "rb") as f:
        header = f.read(header_bytes)

    NGroups = struct.unpack("i", header[0:4])[0]
    TotNGroups = struct.unpack("i", header[4:8])[0]
    # NIds = struct.unpack("i", header[8:12])[0]
    # TotNids = struct.unpack("q", header[12:20])[0]
    NFiles = struct.unpack("i", header[20:24])[0]
    NSubs = struct.unpack("i", header[24:28])[0]
    TotNSubs = struct.unpack("i", header[28:32])[0]

    assert NFiles == nChunks  # verifies endianness
    endian = "@"  # native = little

    # no catalog? (snapshot <= 3) write empty catalog
    if TotNGroups == 0:
        assert TotNSubs == 0
        print("No groups or subgroups for snapshot[%03d]! Writing empty catalog!" % snap)

        # save into single hdf5
        with h5py.File(savePath + "groups_%03d/fof_subhalo_tab_%03d.hdf5" % (snap, snap), "w") as f:
            # header
            header = f.create_group("Header")
            header.attrs["Ngroups_ThisFile"] = 0
            header.attrs["Ngroups_Total"] = 0
            header.attrs["Nids_ThisFile"] = 0
            header.attrs["Nids_Total"] = 0
            header.attrs["Nsubgroups_ThisFile"] = 0
            header.attrs["Nsubgroups_Total"] = 0
            header.attrs["NumFiles"] = 1
        return

    print("Total: [%d] groups, [%d] subhalos, reading..." % (TotNGroups, TotNSubs))

    # allocate (groups)
    GroupLen = np.zeros(TotNGroups, dtype="int32")
    GroupOffset = np.zeros(TotNGroups, dtype="int32")  # note: wrong (overflow) (unused)
    GroupMass = np.zeros(TotNGroups, dtype="float32")
    GroupPos = np.zeros((TotNGroups, 3), dtype="float32")

    Halo_M_Mean200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_Mean200 = np.zeros(TotNGroups, dtype="float32")
    Halo_M_Crit200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_Crit200 = np.zeros(TotNGroups, dtype="float32")
    Halo_M_TopHat200 = np.zeros(TotNGroups, dtype="float32")
    Halo_R_TopHat200 = np.zeros(TotNGroups, dtype="float32")

    ContamCount = np.zeros(TotNGroups, dtype="uint32")  # unused, ==0
    ContamMass = np.zeros(TotNGroups, dtype="float32")  # unused, ==0
    NSubsPerHalo = np.zeros(TotNGroups, dtype="int32")
    FirstSubOfHalo = np.zeros(TotNGroups, dtype="int32")

    # allocate (subhalos)
    SubLen = np.zeros(TotNSubs, dtype="int32")
    SubOffset = np.zeros(TotNSubs, dtype="int32")  # note: wrong (overflow) (unused)
    SubParentHalo = np.zeros(TotNSubs, dtype="int32")  # note: is SubhaloParent? unlike in Millennium-1
    SubMass = np.zeros(TotNSubs, dtype="float32")

    SubPos = np.zeros((TotNSubs, 3), dtype="float32")
    SubVel = np.zeros((TotNSubs, 3), dtype="float32")
    SubCM = np.zeros((TotNSubs, 3), dtype="float32")
    SubSpin = np.zeros((TotNSubs, 3), dtype="float32")
    SubVelDisp = np.zeros(TotNSubs, dtype="float32")
    SubVmax = np.zeros(TotNSubs, dtype="float32")
    SubRVmax = np.zeros(TotNSubs, dtype="float32")
    SubHalfMass = np.zeros(TotNSubs, dtype="float32")
    SubMostBoundID = np.zeros(TotNSubs, dtype="int64")
    SubGrNr = np.zeros(TotNSubs, dtype="int32")

    # load (subhalo files)
    s_off = 0  # subhalos
    g_off = 0  # groups

    for i in range(nChunks):
        if i % int(nChunks / 10) == 0:
            print(" %d%%" % np.ceil(float(i) / nChunks * 100), end="", flush=True)
        # full read
        with open(_chunkPath(snap, i), "rb") as f:
            data = f.read()

        # header (object counts)
        NGroups = struct.unpack(endian + "i", data[0:4])[0]
        NSubs = struct.unpack(endian + "i", data[24:28])[0]

        # per halo
        off = header_bytes
        GroupLen[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 0 * NGroups : off + 4 * NGroups]
        )
        GroupOffset[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 4 * NGroups : off + 8 * NGroups]
        )
        GroupMass[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 8 * NGroups : off + 12 * NGroups]
        )
        GroupPos[g_off : g_off + NGroups, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NGroups, data[off + 12 * NGroups : off + 24 * NGroups]), (NGroups, 3)
        )

        off = header_bytes + 24 * NGroups
        Halo_M_Mean200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 0 * NGroups : off + 4 * NGroups]
        )
        Halo_R_Mean200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 4 * NGroups : off + 8 * NGroups]
        )
        Halo_M_Crit200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 8 * NGroups : off + 12 * NGroups]
        )
        Halo_R_Crit200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 12 * NGroups : off + 16 * NGroups]
        )
        Halo_M_TopHat200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 16 * NGroups : off + 20 * NGroups]
        )
        Halo_R_TopHat200[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 20 * NGroups : off + 24 * NGroups]
        )

        off = header_bytes + 48 * NGroups
        ContamCount[g_off : g_off + NGroups] = struct.unpack(
            endian + "I" * NGroups, data[off + 0 * NGroups : off + 4 * NGroups]
        )
        ContamMass[g_off : g_off + NGroups] = struct.unpack(
            endian + "f" * NGroups, data[off + 4 * NGroups : off + 8 * NGroups]
        )

        NSubsPerHalo[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 8 * NGroups : off + 12 * NGroups]
        )
        FirstSubOfHalo[g_off : g_off + NGroups] = struct.unpack(
            endian + "i" * NGroups, data[off + 12 * NGroups : off + 16 * NGroups]
        )  # global

        # per subhalo
        off = header_bytes + 64 * NGroups
        SubLen[s_off : s_off + NSubs] = struct.unpack(endian + "i" * NSubs, data[off + 0 * NSubs : off + 4 * NSubs])
        SubOffset[s_off : s_off + NSubs] = struct.unpack(endian + "i" * NSubs, data[off + 4 * NSubs : off + 8 * NSubs])
        SubParentHalo[s_off : s_off + NSubs] = struct.unpack(
            endian + "i" * NSubs, data[off + 8 * NSubs : off + 12 * NSubs]
        )  # group rel

        off = header_bytes + 64 * NGroups + 12 * NSubs
        SubMass[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 0 * NSubs : off + 4 * NSubs]
        )
        SubPos[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 4 * NSubs : off + 16 * NSubs]), (NSubs, 3)
        )
        SubVel[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 16 * NSubs : off + 28 * NSubs]), (NSubs, 3)
        )
        SubCM[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 28 * NSubs : off + 40 * NSubs]), (NSubs, 3)
        )
        SubSpin[s_off : s_off + NSubs, :] = np.reshape(
            struct.unpack(endian + "f" * 3 * NSubs, data[off + 40 * NSubs : off + 52 * NSubs]), (NSubs, 3)
        )

        SubVelDisp[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 52 * NSubs : off + 56 * NSubs]
        )
        SubVmax[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 56 * NSubs : off + 60 * NSubs]
        )
        SubRVmax[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 60 * NSubs : off + 64 * NSubs]
        )
        SubHalfMass[s_off : s_off + NSubs] = struct.unpack(
            endian + "f" * 1 * NSubs, data[off + 64 * NSubs : off + 68 * NSubs]
        )
        SubMostBoundID[s_off : s_off + NSubs] = struct.unpack(
            endian + "q" * 1 * NSubs, data[off + 68 * NSubs : off + 76 * NSubs]
        )
        SubGrNr[s_off : s_off + NSubs] = struct.unpack(
            endian + "i" * 1 * NSubs, data[off + 76 * NSubs : off + 80 * NSubs]
        )

        # should have read entire file
        off = header_bytes + 64 * NGroups + 92 * NSubs
        assert off == len(data)

        g_off += NGroups
        s_off += NSubs

    # sanity checks
    assert SubLen.min() >= 20
    assert SubGrNr.min() >= 0
    assert SubMass.min() > 0
    assert GroupLen.min() > 0
    assert GroupMass.min() > 0
    assert SubParentHalo.min() >= 0
    assert GroupPos.min() >= 0.0 and GroupPos.max() <= 100.0
    assert SubPos.min() >= 0.0 and SubPos.max() <= 100.0
    assert SubMostBoundID.min() >= 0
    assert SubGrNr.max() < TotNGroups

    assert g_off == TotNGroups
    assert s_off == TotNSubs

    # verify subhalo ordering
    SubGrNr_check = np.zeros(SubGrNr.size, dtype="int32")
    for i in range(TotNGroups):
        if NSubsPerHalo[i] == 0:
            continue
        SubGrNr_check[FirstSubOfHalo[i] : FirstSubOfHalo[i] + NSubsPerHalo[i]] = i
    assert np.array_equal(SubGrNr, SubGrNr_check)

    # sanity checks
    assert FirstSubOfHalo[0] == 0
    w = np.where(NSubsPerHalo > 0)
    assert np.array_equal(SubGrNr[FirstSubOfHalo[w]], w[0])
    assert GroupLen[0] == GroupLen.max()
    w = np.where(NSubsPerHalo == 0)
    if len(w[0]):
        a, b = match(w[0], SubGrNr)
        assert a is None and b is None
        FirstSubOfHalo[w] = -1  # modify GroupFirstSub, convention

    assert FirstSubOfHalo.max() < TotNSubs

    # GroupOffset/SubhaloOffset have overflow issues, just recreate now with global offsets
    subgroupCount = 0

    snapOffsetsGroup = np.zeros(TotNGroups, dtype="int64")
    snapOffsetsSubhalo = np.zeros(TotNSubs, dtype="int64")

    snapOffsetsGroup[1:] = np.cumsum(GroupLen)[:-1]

    for k in np.arange(TotNGroups):
        # subhalo offsets depend on group (to allow fuzz)
        if NSubsPerHalo[k] > 0:
            snapOffsetsSubhalo[subgroupCount] = snapOffsetsGroup[k]

            subgroupCount += 1
            for _m in np.arange(1, NSubsPerHalo[k]):
                snapOffsetsSubhalo[subgroupCount] = snapOffsetsSubhalo[subgroupCount - 1] + SubLen[subgroupCount - 1]
                subgroupCount += 1

    # save into single hdf5
    with h5py.File(savePath + "groups_%03d/fof_subhalo_tab_%03d.hdf5" % (snap, snap), "w") as f:
        # header
        header = f.create_group("Header")
        header.attrs["Ngroups_ThisFile"] = np.int32(TotNGroups)
        header.attrs["Ngroups_Total"] = np.int32(TotNGroups)
        header.attrs["Nids_ThisFile"] = np.int32(0)
        header.attrs["Nids_Total"] = np.int32(0)
        header.attrs["Nsubgroups_ThisFile"] = np.int32(TotNSubs)
        header.attrs["Nsubgroups_Total"] = np.int32(TotNSubs)
        header.attrs["NumFiles"] = np.int32(1)

        # groups
        groups = f.create_group("Group")
        groups["GroupFirstSub"] = FirstSubOfHalo
        groups["GroupLen"] = GroupLen
        groups["GroupPos"] = GroupPos
        groups["GroupMass"] = GroupMass
        groups["GroupNsubs"] = NSubsPerHalo
        groups["Group_M_Crit200"] = Halo_M_Crit200
        groups["Group_R_Crit200"] = Halo_R_Crit200
        groups["Group_M_Mean200"] = Halo_M_Mean200
        groups["Group_R_Mean200"] = Halo_R_Mean200
        groups["Group_M_TopHat200"] = Halo_M_TopHat200
        groups["Group_R_TopHat200"] = Halo_R_TopHat200

        GroupLenType = np.zeros((GroupLen.size, 6), dtype=GroupLen.dtype)
        GroupMassType = np.zeros((GroupLen.size, 6), dtype="float32")
        GroupLenType[:, 1] = GroupLen
        GroupMassType[:, 1] = GroupMass
        groups["GroupLenType"] = GroupLenType
        groups["GroupMassType"] = GroupMassType

        # subhalos
        subs = f.create_group("Subhalo")
        subs["SubhaloLen"] = SubLen
        subs["SubhaloMass"] = SubMass
        subs["SubhaloGrNr"] = SubGrNr
        subs["SubhaloHalfmassRad"] = SubHalfMass
        subs["SubhaloIDMostbound"] = SubMostBoundID
        subs["SubhaloParent"] = SubParentHalo
        subs["SubhaloPos"] = SubPos
        subs["SubhaloCM"] = SubCM
        subs["SubhaloSpin"] = SubSpin
        subs["SubhaloVel"] = SubVel
        subs["SubhaloVelDisp"] = SubVelDisp
        subs["SubhaloVmaxRad"] = SubRVmax
        subs["SubhaloVmax"] = SubVmax

        SubhaloLenType = np.zeros((SubLen.size, 6), dtype=SubLen.dtype)
        SubhaloMassType = np.zeros((SubLen.size, 6), dtype="float32")
        SubhaloLenType[:, 1] = SubLen
        SubhaloMassType[:, 1] = SubMass
        subs["SubhaloLenType"] = SubhaloLenType
        subs["SubhaloMassType"] = SubhaloMassType

        # offsets (inside group files, similar to Illustris public data release, for convenience)
        offs = f.create_group("Offsets")
        offs["Group_Snap"] = snapOffsetsGroup
        offs["Subhalo_Snap"] = snapOffsetsSubhalo

    # return
    print(" All Done.")


def convertMillenniumSnapshot(snap=63):
    """Convert a complete Millennium snapshot (+IDS) into Illustris-like group-ordered HDF5 format."""
    savePath = path.expanduser("~") + "/sims.other/Millennium-1/output/"
    loadPath = "/virgo/simulations/Recovered_Millennium/"
    # loadPath = '/virgo/simulations/MilliMillennium/'

    saveFile = savePath + "snapdir_%03d/snap_%03d.hdf5" % (snap, snap)

    if not path.isdir(savePath + "snapdir_%03d" % snap):
        mkdir(savePath + "snapdir_%03d" % snap)

    def _snapChunkPath(snap, chunkNum):
        return loadPath + "snapdir_%03d/snap_millennium_%03d.%s" % (snap, snap, chunkNum)
        # return loadPath + 'snapdir_%03d/snap_milli_%03d.%s' % (snap,snap,chunkNum)

    def _idChunkPath(snap, chunkNum):
        return loadPath + "postproc_%03d/sub_ids_%03d.%s" % (snap, snap, chunkNum)

    nChunks = len(glob.glob(_snapChunkPath(snap, "*")))
    nChunksIDs = len([fn for fn in glob.glob(_idChunkPath(snap, "*")) if "swapped" not in fn])
    assert nChunks == nChunksIDs or nChunksIDs == 0
    print("Found [%d] chunks for snapshot [%03d], loading..." % (nChunks, snap))

    # detect big-endianness
    endian = "@"  # native = little

    with open(_snapChunkPath(snap, 0), "rb") as f:
        header = f.read(260)
        NFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]

    if NFiles != nChunks:
        endian = ">"
        NFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]

    assert NFiles == nChunks

    # cycle one
    if not path.isfile(saveFile):
        # first, load all IDs from the sub files
        Nids_tot = 0

        if path.isfile(savePath + "gorder_%d.hdf5" % snap):
            offset = 0

            for i in range(nChunks):
                with open(_idChunkPath(snap, i), "rb") as f:
                    header = f.read(16)
                    Nids = struct.unpack(endian + "i", header[4:8])[0]
                    Nids_tot += Nids

            ids_groupordered_old = np.zeros(Nids_tot, dtype="int64")
            print("Reading a total of [%d] IDs now..." % Nids_tot)

            bitshift = (1 << 34) - 1  # from get_group_coordinates()

            for i in range(nChunks):
                # full read
                with open(_idChunkPath(snap, i), "rb") as f:
                    data = f.read()
                Nids = struct.unpack(endian + "i", data[4:8])[0]
                if Nids == 0:
                    continue
                ids = struct.unpack(endian + "q" * Nids, data[16 : 16 + Nids * 8])

                # transform into actual particle ID
                # particleid = (GroupIDs[i] << 30) >> 30 (seems wrong...)
                # hashkey    = GroupIDs[i] >> 34
                # ids_groupordered[offset : offset+Nids] = (np.array(ids) << 30) >> 30
                ids_groupordered_old[offset : offset + Nids] = np.array(ids) & bitshift

                print("[%3d] IDs [%8d] particles, from [%10d] to [%10d]." % (i, Nids, offset, offset + Nids))
                offset += Nids

            assert np.min(ids_groupordered_old) >= 0  # otherwise overflow or bad conversion above

            # ids_groupordered are in the chunk-local group ordering! reshuffle now
            print("Shuffling IDs into global group order...")

            gorder = {}
            with h5py.File(savePath + "gorder_%d.hdf5" % snap) as f:
                for key in f:
                    gorder[key] = f[key][()]

            # use original (chunk-ordered) GroupOffset and GroupLen accessed in global-order
            # to access ids_groupordered in non-contig blocks, stamping into ids_groupordered_new contiguously through
            offset = 0
            ids_groupordered = np.zeros(Nids_tot, dtype="int64")

            for i in gorder["Group_Reorder"]:
                read_offset = gorder["GroupOffset_Orig"][i]
                read_length = gorder["GroupLen_Orig"][i]

                if read_length == 0:
                    continue

                ids_groupordered[offset : offset + read_length] = ids_groupordered_old[
                    read_offset : read_offset + read_length
                ]
                offset += read_length

            ids_groupordered_old = None
        else:
            print("NO GROUP CATALOG! Assuming zero groups, and proceeding with snapshot rewrite!")

        # reader header of first snapshot chunk
        with open(_snapChunkPath(snap, 0), "rb") as f:
            header = f.read(260)

        npart = struct.unpack(endian + "iiiiii", header[4 : 4 + 24])[1]
        mass = struct.unpack(endian + "dddddd", header[28 : 28 + 48])
        scalefac = struct.unpack(endian + "d", header[76 : 76 + 8])[0]
        redshift = struct.unpack(endian + "d", header[84 : 84 + 8])[0]
        # nPartTot   = struct.unpack(endian+'iiiiii', header[100:100+24])[1]
        nFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]
        BoxSize = struct.unpack(endian + "d", header[132 : 132 + 8])[0]
        Omega0 = struct.unpack(endian + "d", header[140 : 140 + 8])[0]
        OmegaL = struct.unpack(endian + "d", header[148 : 148 + 8])[0]
        Hubble = struct.unpack(endian + "d", header[156 : 156 + 8])[0]

        assert nFiles == nChunks

        # nPartTot is wrong, has no highword, so read and accumulate manually
        nPartTot = 0
        for i in range(nChunks):
            with open(_snapChunkPath(snap, i), "rb") as f:
                header = f.read(28)
                npart = struct.unpack(endian + "iiiiii", header[4 : 4 + 24])[1]
            nPartTot += npart
        print("Found new nPartTot [%d]" % nPartTot)

        # load all snapshot IDs
        offset = 0
        ids_snapordered = np.zeros(nPartTot, dtype="int64")

        for i in range(nChunks):
            # full read
            with open(_snapChunkPath(snap, i), "rb") as f:
                data = f.read()

            # local particle counts
            npart_local = struct.unpack(endian + "iiiiii", data[4 : 4 + 24])[1]

            # cast and save
            start_ids = 284 + 24 * npart_local
            ids = struct.unpack(endian + "q" * npart_local * 1, data[start_ids : start_ids + npart_local * 8])

            ids_snapordered[offset : offset + npart_local] = ids

            print(
                "[%3d] Snap IDs [%8d] particles, from [%10d] to [%10d]."
                % (i, npart_local, offset, offset + npart_local)
            )
            offset += npart_local

        # crossmatch group catalog IDs and snapshot IDs
        if Nids_tot > 0:
            print("Matching two ID sets now...")
            start = time.time()

            ind_snapordered, ind_groupordered = match(ids_snapordered, ids_groupordered)
            # note: ids_snapordered[ind_snapordered] puts them into group ordering
            print(" took: " + str(round(time.time() - start, 2)) + " sec")

            assert ind_snapordered.size == ids_groupordered.size  # must have found them all
            ids_groupordered = None

            # create mask for outer FoF fuzz
            mask = np.zeros(ids_snapordered.size, dtype="bool")
            mask[ind_snapordered] = 1

            ind_outerfuzz = np.where(mask == 0)[0]

            # create master re-ordering index list
            inds_reorder = np.hstack((ind_snapordered, ind_outerfuzz))

            ind_snapordered = None
            ind_outerfuzz = None
        else:
            print("No cross-matching! Writing snapshot in ORIGINAL ORDER!")
            inds_reorder = np.arange(ids_snapordered.size)

        assert inds_reorder.size == ids_snapordered.size  # union must include all
        assert isUnique(inds_reorder)  # no duplicates

        # open file for writing
        fOut = h5py.File(saveFile, "w")

        header = fOut.create_group("Header")
        numPartTot = np.zeros(6, dtype="int64")
        numPartTot[1] = nPartTot
        header.attrs["BoxSize"] = BoxSize
        header.attrs["HubbleParam"] = Hubble
        header.attrs["MassTable"] = np.array(mass, dtype="float64")
        header.attrs["NumFilesPerSnapshot"] = np.int32(1)
        header.attrs["NumPart_ThisFile"] = np.int32(numPartTot)
        header.attrs["NumPart_Total"] = np.int32(numPartTot)
        header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype="int32")
        header.attrs["Omega0"] = Omega0
        header.attrs["OmegaLambda"] = OmegaL
        header.attrs["Redshift"] = redshift
        header.attrs["Time"] = scalefac

        pt1 = fOut.create_group("PartType1")

        # save IDs immediately
        pt1["ParticleIDs"] = ids_snapordered[inds_reorder]
        ids_snapordered = None
        fOut.close()

        # save order permutation, early quit (clear memory)
        with h5py.File(savePath + "reorder_%d.hdf5" % snap, "w") as f:
            f["Reorder"] = inds_reorder
        print("Wrote intermediate file, restart to finish.")
        return
    else:
        # intermediate file already exists, open
        print("Loading intermediate file...")
        with h5py.File(savePath + "reorder_%d.hdf5" % snap) as f:
            inds_reorder = f["Reorder"][()]

        fOut = h5py.File(saveFile)
        nPartTot = fOut["Header"].attrs["NumPart_Total"][1]
        pt1 = fOut["PartType1"]

        # load remaining particle fields, one at a time
        for ptName in ["Coordinates", "Velocities"]:
            offset = 0
            val = np.zeros((nPartTot, 3), dtype="float32")

            for i in range(nChunks):
                # full read
                with open(_snapChunkPath(snap, i), "rb") as f:
                    data = f.read()

                # local particle counts
                npart_local = struct.unpack(endian + "iiiiii", data[4 : 4 + 24])[1]

                # cast
                start_pos = 268
                start_vel = 276 + 12 * npart_local
                start = start_pos if ptName == "Coordinates" else start_vel

                val_local = struct.unpack(endian + "f" * npart_local * 3, data[start : start + npart_local * 12])

                # stamp
                val[offset : offset + npart_local, :] = np.reshape(val_local, (npart_local, 3))

                print(
                    "[%3d] %s [%8d] particles, from [%10d] to [%10d]."
                    % (i, ptName, npart_local, offset, offset + npart_local)
                )
                offset += npart_local

            # re-order and write
            val = val[inds_reorder, :]

            pt1[ptName] = val

        # close
        fOut.close()
        print("All done.")


def convertMillennium2Snapshot(snap=67):
    """Convert a complete Millennium-2 snapshot into TNG-like group-ordered HDF5 format.

    Note all snapshots except 4-7 (inclusive) are already group-ordered.
    """
    savePath = path.expanduser("~") + "/sims.other/Millennium-2/output/"
    loadPath = "/virgo/simulations/Millennium2/BigRun/"

    saveFile = savePath + "snapdir_%03d/snap_%03d.hdf5" % (snap, snap)

    unorderedSnaps = [4, 5, 6, 7]  # on /virgo/, only these snapshots are not yet in group order

    if not path.isdir(savePath + "snapdir_%03d" % snap):
        mkdir(savePath + "snapdir_%03d" % snap)

    def _snapChunkPath(snap, chunkNum):
        if snap in unorderedSnaps:
            return loadPath + "snapdir_%03d/snap_newMillen_%03d.%s" % (snap, snap, chunkNum)
        return loadPath + "snapdir_%03d/snap_newMillen_subidorder_%03d.%s" % (snap, snap, chunkNum)

    def _idChunkPath(snap, chunkNum):
        return loadPath + "groups_%03d/subhalo_ids_%03d.%s" % (snap, snap, chunkNum)

    nChunks = len(glob.glob(_snapChunkPath(snap, "*")))
    nChunksIDs = len(glob.glob(_idChunkPath(snap, "*")))

    # three cases for file organization
    assert nChunks == nChunksIDs or nChunksIDs == 0 or (nChunks == 512 and nChunksIDs == 2048)
    print("Found [%d] chunks for snapshot [%03d], loading..." % (nChunks, snap))

    # load header
    endian = "@"  # native = little
    with open(_snapChunkPath(snap, 0), "rb") as f:
        header = f.read(260)
        NFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]

    if NFiles != nChunks:
        endian = ">"  # big
        NFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]
        print("WARNING: Endian set to big, true for snapshots then?")

    assert NFiles == nChunks

    # reader header of first snapshot chunk
    with open(_snapChunkPath(snap, 0), "rb") as f:
        header = f.read(260)

    npart = struct.unpack(endian + "iiiiii", header[4 : 4 + 24])[1]
    mass = struct.unpack(endian + "dddddd", header[28 : 28 + 48])
    scalefac = struct.unpack(endian + "d", header[76 : 76 + 8])[0]
    redshift = struct.unpack(endian + "d", header[84 : 84 + 8])[0]
    # nPartTot   = struct.unpack(endian+'iiiiii', header[100:100+24])[1]
    nFiles = struct.unpack(endian + "i", header[128 : 128 + 4])[0]
    BoxSize = struct.unpack(endian + "d", header[132 : 132 + 8])[0]
    Omega0 = struct.unpack(endian + "d", header[140 : 140 + 8])[0]
    OmegaL = struct.unpack(endian + "d", header[148 : 148 + 8])[0]
    Hubble = struct.unpack(endian + "d", header[156 : 156 + 8])[0]

    assert nFiles == nChunks

    # nPartTot with highword, just read and accumulate manually
    nPartTot = 0
    for i in range(nChunks):
        with open(_snapChunkPath(snap, i), "rb") as f:
            header = f.read(28)
            npart = struct.unpack(endian + "iiiiii", header[4 : 4 + 24])[1]
        nPartTot += npart

    print("Found nPartTot [%d]" % nPartTot)
    assert nPartTot == 2160**3

    # load all snapshot IDs
    offset = 0
    ids_snapordered = np.zeros(nPartTot, dtype="int64")

    for i in range(nChunks):
        # full read
        with open(_snapChunkPath(snap, i), "rb") as f:
            data = f.read()

        # local particle counts
        npart_local = struct.unpack(endian + "iiiiii", data[4 : 4 + 24])[1]

        # cast and save
        start_ids = 284 + 24 * npart_local
        ids = struct.unpack(endian + "q" * npart_local * 1, data[start_ids : start_ids + npart_local * 8])

        ids_snapordered[offset : offset + npart_local] = ids

        min_val = np.min(ids)
        print(
            "[%4d] Snap IDs [%8d] particles, from [%10d] to [%10d] min = %10d"
            % (i, npart_local, offset, offset + npart_local, min_val)
        )
        offset += npart_local
        assert min_val != 0

    if nChunksIDs > 0:
        # need to reshuffle
        assert snap in unorderedSnaps

        # first, load all IDs from the sub files
        offset = 0

        # M-WAMP7/AQ/M2: Ngroups[int32], TotNgroups[int32], Nids[int32], TotNids[int64], NFiles[int32],
        #                SendOffset[int32(?)], ids[int64*]
        # M1: Ngroups[int32], Nids[int32], TotNgroups[int32], NTask[int32], ids[int64*]
        # P-M: Ngroups[int32], TotNgroups[int64], Nids[int32], TotNids[int64], NFiles[int32],
        #      SendOffset[int64], ids[int64*]
        for i in range(nChunksIDs):
            with open(_idChunkPath(snap, i), "rb") as f:
                header = f.read(28)
                Nids = struct.unpack(endian + "i", header[8:12])[0]
                TotNids = struct.unpack(endian + "q", header[12:20])[0]
                offset += Nids

        ids_groupordered = np.zeros(offset, dtype="int64")
        print("Reading a total of [%d] IDs now..." % offset)
        assert offset == TotNids

        bitshift = (1 << 34) - 1  # same as Millennium-1 (should be corrent)
        offset = 0

        for i in range(nChunksIDs):
            # full read
            with open(_idChunkPath(snap, i), "rb") as f:
                data = f.read()
            Nids = struct.unpack(endian + "i", data[8:12])[0]
            if Nids == 0:
                continue
            ids = struct.unpack(endian + "q" * Nids, data[28 : 28 + Nids * 8])

            # transform into actual particle ID and stamp
            ids_groupordered[offset : offset + Nids] = np.array(ids) & bitshift

            print("[%3d] IDs [%8d] particles, from [%10d] to [%10d]." % (i, Nids, offset, offset + Nids))
            offset += Nids

        assert np.min(ids_groupordered) >= 0  # otherwise overflow or bad conversion above

        # crossmatch group catalog IDs and snapshot IDs
        if TotNids > 0:
            print("Matching two ID sets now...")
            start = time.time()

            ind_snapordered, ind_groupordered = match(ids_snapordered, ids_groupordered)
            # note: ids_snapordered[ind_snapordered] puts them into group ordering
            print(" took: " + str(round(time.time() - start, 2)) + " sec")

            assert ind_snapordered.size == ids_groupordered.size  # must have found them all
            ids_groupordered = None

            # create mask for outer FoF fuzz
            mask = np.zeros(ids_snapordered.size, dtype="bool")
            mask[ind_snapordered] = 1

            ind_outerfuzz = np.where(mask == 0)[0]

            # create master re-ordering index list
            inds_reorder = np.hstack((ind_snapordered, ind_outerfuzz))

            ind_snapordered = None
            ind_outerfuzz = None

        assert inds_reorder.size == ids_snapordered.size  # union must include all
        assert isUnique(inds_reorder)  # no duplicates

        ids_snapordered = ids_snapordered[inds_reorder]

    else:
        print("No subhalo_ids files! Writing snapshot in ORIGINAL ORDER!")
        inds_reorder = None

    # open file for writing
    fOut = h5py.File(saveFile, "w")

    header = fOut.create_group("Header")
    numPartTot = np.zeros(6, dtype="int64")
    numPartTot[1] = nPartTot
    header.attrs["BoxSize"] = BoxSize
    header.attrs["HubbleParam"] = Hubble
    header.attrs["MassTable"] = np.array(mass, dtype="float64")
    header.attrs["NumFilesPerSnapshot"] = np.int32(1)
    header.attrs["NumPart_ThisFile"] = np.int64(numPartTot)
    header.attrs["NumPart_Total"] = np.int64(numPartTot)
    header.attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype="int32")
    header.attrs["Omega0"] = Omega0
    header.attrs["OmegaLambda"] = OmegaL
    header.attrs["Redshift"] = redshift
    header.attrs["Time"] = scalefac

    pt1 = fOut.create_group("PartType1")

    # save IDs immediately
    pt1["ParticleIDs"] = ids_snapordered
    ids_snapordered = None

    # load remaining particle fields, one at a time
    for ptName in ["Coordinates", "Velocities"]:
        offset = 0
        val = np.zeros((nPartTot, 3), dtype="float32")

        for i in range(nChunks):
            # full read
            with open(_snapChunkPath(snap, i), "rb") as f:
                data = f.read()

            # local particle counts
            npart_local = struct.unpack(endian + "iiiiii", data[4 : 4 + 24])[1]

            # cast
            start_pos = 268
            start_vel = 276 + 12 * npart_local
            start = start_pos if ptName == "Coordinates" else start_vel

            unpacker = struct.Struct(endian + "f" * npart_local * 3)
            val_local = unpacker.unpack(data[start : start + npart_local * 12])
            val_local = np.reshape(val_local, (npart_local, 3))

            # stamp
            val[offset : offset + npart_local, :] = val_local

            print(
                "[%4d] %s [%8d] particles, from [%10d] to [%10d]."
                % (i, ptName, npart_local, offset, offset + npart_local),
                flush=True,
            )
            offset += npart_local

        # re-order and write
        if inds_reorder is not None:
            val = val[inds_reorder, :]

        print("writing...")
        pt1[ptName] = val
        print("written.")

    # close
    fOut.close()
    print("All done.")


def convertEagleSnapshot(snap=20):
    """Convert an EAGLE simulation snapshot (HDF5) to a TNG-like snapshot (field names, units, etc)."""
    loadPath = "/virgo/simulations/Eagle/L0100N1504/REFERENCE/data/"
    # loadPath = '/virgo/simulations/EagleDM/L0100N1504/DMONLY/data/'
    # savePath = '/virgo/simulations/Illustris/Eagle-L68n1504DM/output/'
    savePath = "/u/dnelson/data/test/"

    gfmPhotoPath = expanduser("~") + "/data/Arepo_GFM_Tables_TNG/Photometrics/stellar_photometrics.hdf5"

    sP = simParams(res=1504, run="eagle")  # for units only

    metalNamesOrdered = ["Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron"]
    metalTagsOrdered = [
        "MetalMassFracFromSNIa",
        "MetalMassFracFromSNII",
        "MetalMassFracFromAGB",
        "skip",
        "SmoothedIronMassFracFromSNIa",
        "skip",
    ]
    photoBandsOrdered = ["U", "B", "V", "K", "g", "r", "i", "z"]

    fieldRenames = {
        "Velocity": "Velocities",
        "Mass": "Masses",
        "SmoothedMetallicity": "GFM_Metallicity",
        "InitialMass": "GFM_InitialMass",
        "StellarFormationTime": "GFM_StellarFormationTime",
        "BH_CumlAccrMass": "BH_CumMassGrowth_QM",
        "BH_CumlNumSeeds": "BH_Progs",
        "BH_AccretionLength": "BH_Hsml",
    }

    def snapPath(chunkNum):
        return loadPath + snapDir + "/%s.%s.hdf5" % (snapBase, chunkNum)

    def writePath(chunkNum):
        return savePath + "snapdir_%03d/snap_%03d.%d.hdf5" % (snap, snap, chunkNum)

    # locate the snapshot directory and find number of chunks
    snapPaths = glob.glob(loadPath + "snapshot_*")
    for snapPath in snapPaths:
        if "snapshot_%03d" % snap in snapPath:
            snapDir = snapPath.rsplit("/")[-1]

    snapBase = "snap_%03d_%s" % (snap, snapDir.split("_")[-1])
    nChunks = len(glob.glob(snapPath("*")))

    print("Loading [%d] chunks from: [%s]" % (nChunks, snapDir))

    # load the photometrics table first
    gfm_photo = {}
    with h5py.File(gfmPhotoPath, "r") as f:
        for key in f:
            gfm_photo[key] = f[key][()]

    # output directory
    if not isdir(savePath + "snapdir_%03d" % snap):
        mkdir(savePath + "snapdir_%03d" % snap)

    # loop over input chunks
    for chunkNum in range(nChunks):
        print(chunkNum, flush=True)

        # load full file
        data = {}

        gNames = ["PartType0", "PartType1", "PartType4", "PartType5"] if "DMONLY" not in loadPath else ["PartType1"]
        for gName in gNames:
            data[gName] = {}

        with h5py.File(snapPath(chunkNum), "r") as f:
            header = dict(f["Header"].attrs)
            config = dict(f["Config"].attrs)
            params = dict(f["RuntimePars"].attrs)

            # dm
            print(" dm")
            if "PartType1" in f:
                for key in ["Coordinates", "ParticleIDs", "Velocity"]:
                    data["PartType1"][key] = f["PartType1"][key][()]

            # gas
            print(" gas")
            if "PartType0" in f:
                for key in [
                    "Coordinates",
                    "Density",
                    "InternalEnergy",
                    "Mass",
                    "ParticleIDs",
                    "SmoothedMetallicity",
                    "StarFormationRate",
                    "Temperature",
                    "Velocity",
                ]:
                    data["PartType0"][key] = f["PartType0"][key][()]

            # stars
            print(" stars")
            if "PartType4" in f:
                for key in [
                    "Coordinates",
                    "Mass",
                    "ParticleIDs",
                    "InitialMass",
                    "SmoothedMetallicity",
                    "StellarFormationTime",
                    "Velocity",
                ]:
                    data["PartType4"][key] = f["PartType4"][key][()]

            # gas + stars
            print(" gas+stars")
            for pt in ["PartType0", "PartType4"]:
                if pt not in f:
                    continue
                data[pt]["GFM_Metals"] = np.zeros((data[pt]["ParticleIDs"].size, 10), dtype="float32")
                data[pt]["GFM_MetalsTagged"] = np.zeros((data[pt]["ParticleIDs"].size, 6), dtype="float32")

                for i, el in enumerate(metalNamesOrdered):
                    data[pt]["GFM_Metals"][:, i] = f[pt]["SmoothedElementAbundance"][el][()]
                for i, name in enumerate(metalTagsOrdered):
                    if name == "skip":
                        continue
                    data[pt]["GFM_MetalsTagged"][:, i] = f[pt][name][()]

            # BHs
            print(" bhs")
            if "PartType5" in f:
                for key in [
                    "BH_CumlAccrMass",
                    "BH_CumlNumSeeds",
                    "BH_Density",
                    "BH_Mass",
                    "BH_Mdot",
                    "BH_Pressure",
                    "BH_SoundSpeed",
                    "BH_SurroundingGasVel",
                    "BH_WeightedDensity",
                    "BH_AccretionLength",
                    "Coordinates",
                    "Mass",
                    "ParticleIDs",
                    "Velocity",
                ]:
                    data["PartType5"][key] = f["PartType5"][key][()]

                for key in [
                    "BH_BPressure",
                    "BH_CumEgyInjection_RM",
                    "BH_CumMassGrowth_RM",
                    "BH_HostHaloMass",
                    "BH_U",
                ]:  # missing
                    data["PartType5"][key] = np.zeros(data["PartType5"]["BH_Mass"].size, dtype="float32")

        # cleanup header
        for key in ["E(z)", "H(z)", "RunLabel"]:
            del header[key]
        header["Time"] = header.pop("ExpansionFactor")
        header["UnitLength_in_cm"] = 3.08568e21
        header["UnitMass_in_g"] = 1.989e43
        header["UnitVelocity_in_cm_per_s"] = 100000
        header["Flag_DoublePrecision"] = 1 if data["PartType1"]["Coordinates"].itemsize == 8 else 0

        # field renames
        for pt in data.keys():
            for from_name, to_name in fieldRenames.items():
                if from_name in data[pt]:
                    data[pt][to_name] = data[pt].pop(from_name)

        if "PartType0" in data:
            data["PartType0"]["CenterOfMass"] = data["PartType0"]["Coordinates"]

        # unit conversions
        for pt in data.keys():
            if "Coordinates" in data[pt]:
                data[pt]["Coordinates"] *= 1e3  # cMpc/h -> cKpc/h
            # wrong (in release paper)! also in sqrt(a) units like TNG, as per aexp-scale-exponent attr
            # if 'Velocities' in data[pt]:
            #    data[pt]['Velocities'] /= np.sqrt(header['Time']) # peculiar -> sqrt(a) units
            if "GFM_Metallicity" in data[pt]:
                w = np.where(data[pt]["GFM_Metallicity"] < 1e-20)
                data[pt]["GFM_Metallicity"][w] = 1e-20  # GFM_MIN_METAL
            if "GFM_MetalsTagged" in data[pt]:
                w = np.where(data[pt]["GFM_MetalsTagged"] < 0.0)
                data[pt]["GFM_MetalsTagged"][w] = 0.0  # enforce >=0

        if "PartType0" in data and "Density" in data["PartType0"]:
            data["PartType0"]["Density"] /= 1e9  # Mpc^-3 -> Kpc^-3
        if "PartType5" in data and "BH_Density" in data["PartType5"]:
            data["PartType5"]["BH_Density"] /= 1e9

        if "PartType0" in data:
            # gas: ne
            x_h = data["PartType0"]["GFM_Metals"][:, 0]
            mean_mol_wt = (
                data["PartType0"]["Temperature"]
                * sP.units.boltzmann
                / ((5.0 / 3.0 - 1) * data["PartType0"]["InternalEnergy"] * 1e10)
            )
            nelec = (sP.units.mass_proton * 4.0 / mean_mol_wt - 1.0 - 3.0 * x_h) / (4 * x_h)
            data["PartType0"]["ElectronAbundance"] = nelec

            # gas: nH
            sP.redshift = header["Redshift"]
            sP.units.scalefac = header["Time"]
            nH = (
                sP.units.codeDensToPhys(data["PartType0"]["Density"], cgs=True, numDens=True)
                * data["PartType0"]["GFM_Metals"][:, 0]
            )
            frac_nH0 = neutral_fraction(nH, sP=None, redshift=header["Redshift"])
            data["PartType0"]["NeutralHydrogenAbundance"] = frac_nH0

            # NOTE: In the currently written Eagle-L68n1504FP, {ne,nh,sfr} were overwritten by io_func_* recalculations,
            # but this has no actual impact, as StarFormationRate is OK (for RestartFlag==3), while Ne and Nh didn't
            # anyways exist in the original Eagle snapshots

        # stars: photometrics
        if "PartType4" in data and "Masses" in data["PartType4"]:
            data["PartType4"]["GFM_StellarPhotometrics"] = np.zeros(
                (data["PartType4"]["Masses"].size, 8), dtype="float32"
            )

            stars_formz = 1 / data["PartType4"]["GFM_StellarFormationTime"] - 1
            stars_logagegyr = np.log10(
                sP.units.redshiftToAgeFlat(header["Redshift"]) - sP.units.redshiftToAgeFlat(stars_formz)
            )
            stars_logz = np.log10(data["PartType4"]["GFM_Metallicity"])
            stars_masslogmsun = sP.units.codeMassToLogMsun(data["PartType4"]["GFM_InitialMass"])

            i1 = np.interp(
                stars_logz, gfm_photo["LogMetallicity_bins"], np.arange(gfm_photo["LogMetallicity_bins"].size)
            )
            i2 = np.interp(
                stars_logagegyr, gfm_photo["LogAgeInGyr_bins"], np.arange(gfm_photo["LogAgeInGyr_bins"].size)
            )
            iND = np.vstack((i1, i2))

            for i, band in enumerate(photoBandsOrdered):
                mags_1msun = map_coordinates(gfm_photo["Magnitude_%s" % band], iND, order=1, mode="nearest")
                data["PartType4"]["GFM_StellarPhotometrics"][:, i] = mags_1msun - 2.5 * stars_masslogmsun

        # BHs: accretion rates (1e3 from Mpc in UnitTime)
        if "PartType5" in data:
            data["PartType5"]["BH_Mdot"] /= 1e3

        # BHs: bondi and eddington mdot (should be checked more carefully)
        if "PartType5" in data and "BH_SurroundingGasVel" in data["PartType5"]:
            UnitMass_over_UnitTime = 10.22

            vrel = data["PartType5"]["BH_SurroundingGasVel"] / header["Time"]  # km/s
            vrel_mag = np.sqrt(vrel[:, 0] ** 2 + vrel[:, 1] ** 2 + vrel[:, 2] ** 2)  # km/s
            vel_term = (data["PartType5"]["BH_SoundSpeed"] ** 2 + vrel_mag**2) ** (3.0 / 2.0)  # (km/s)^3
            bh_mass = data["PartType5"]["BH_Mass"] / header["HubbleParam"]  # 10^10 msun
            dens = (
                data["PartType5"]["BH_Density"] * header["HubbleParam"] ** 2 / header["Time"] ** 3
            )  # 10^10 msun/kpc^3

            mdot_bondi = 4 * np.pi * sP.units.G**2 * bh_mass**2 * dens / vel_term  # (km/kpc) * (10^10 msun/s)
            mdot_bondi = mdot_bondi / sP.units.kpc_in_km * 1e10 * sP.units.s_in_yr  # msun/yr
            data["PartType5"]["BH_MdotBondi"] = mdot_bondi / UnitMass_over_UnitTime  # put into TNG units

            mdot_edd = sP.units.codeBHMassToMdotEdd(data["PartType5"]["BH_Mass"], eps_r=0.1)  # msun/yr
            data["PartType5"]["BH_MdotEddington"] = mdot_edd / UnitMass_over_UnitTime  # put into TNG units

            # BHs: cum egy injection
            bh_mass = sP.units.codeMassToMsun(data["PartType5"]["BH_CumMassGrowth_QM"])

            bh_E = 0.15 * 0.1 * bh_mass * sP.units.c_kpc_Gyr**2  # msun kpc^2/Gyr^2
            bh_E /= 1e9  # msun/yr kpc^2 / Gyr
            bh_E = bh_E * header["HubbleParam"] ** 2 / header["Time"] ** 2  # msun/yr (ckpc/h)^2 / Gyr
            bh_E = bh_E * header["HubbleParam"] / 0.978  # msun/yr (ckpc/h)^2 / (0.978Gyr/h)

            data["PartType5"]["BH_CumEgyInjection_QM"] = bh_E / 10.22  # put into TNG units

        # write
        with h5py.File(writePath(chunkNum), "w") as f:
            # headers
            h = f.create_group("Header")
            for key in header:
                h.attrs[key] = header[key]
            c = f.create_group("Config")
            for key in config:
                vals = config[key].decode("ascii").split(" ")
                if key == "SVN_Version":
                    vals = ["SVN_Version", config[key]]
                c.attrs[vals[0]] = vals[1]
            p = f.create_group("Parameters")
            for key in params:
                p.attrs[key] = params[key]

            # particle groups
            for gName in data:
                g = f.create_group(gName)

                for key in data[gName]:
                    g[key] = data[gName][key]

    print("Done.")


def convertSimbaSnapshot(snap=151):
    """Convert an SIMBA simulation snapshot (HDF5) to a TNG-like snapshot (field names, units, etc)."""
    run = "m25n512"  # 'm50n512', 'm100n1024'

    # derived paths
    basePath = "/u/dnelson/data/sims.other/Simba-"  #'/virgotng/universe/Simba/'
    loadPath = basePath + "%s/orig-snapshots/" % (run.replace("m", "L") + "FP")
    savePath = basePath + "%s/output/" % (run.replace("m", "L") + "FP")

    gfmPhotoPath = rootPath + "/tables/bc03/stellar_photometrics.hdf5"

    sP = simParams(run="simba")  # for units only

    # metals_TNG = ["Hydrogen", "Helium", "Carbon", "Nitrogen", "Oxygen", "Neon", "Magnesium", "Silicon", "Iron"]  # 9
    # metals_Simba = ["Z", "He", "C", "N", "O", "Ne", "Mg", "Si", "S", "Ca", "Fe"]  # 11. NOTE: First is Z, not H!
    metalInds = [0, 1, 2, 3, 4, 5, 6, 7, 10]  # skip sulphur and calcium, note: H (0) is computed and overrides
    photoBandsOrdered = ["U", "B", "V", "K", "g", "r", "i", "z"]  # TNG

    fieldRenames = {
        "StellarFormationTime": "GFM_StellarFormationTime",
        "BH_NProgs": "BH_Progs",
        "BH_AccretionLength": "BH_Hsml",
    }

    snapPath = loadPath + "/snap_%s_%03d.hdf5" % (run, snap)
    writePath = savePath + "snapdir_%03d/snap_%03d.hdf5" % (snap, snap)

    # load the photometrics table first
    gfm_photo = {}
    with h5py.File(gfmPhotoPath, "r") as f:
        for key in f:
            gfm_photo[key] = f[key][()]

    # output directory
    if not isdir(savePath + "snapdir_%03d" % snap):
        mkdir(savePath + "snapdir_%03d" % snap)

    # load full file
    data = {}

    gNames = ["PartType0", "PartType1", "PartType4", "PartType5"]
    for gName in gNames:
        data[gName] = {}

    with h5py.File(snapPath, "r") as f:
        header = dict(f["Header"].attrs)
        print(snap, " z = ", header["Redshift"])

        # dm
        print(" dm")
        if "PartType1" in f:
            # skipped: AGS-Softening, HaloID, ID_Generations
            for key in ["Coordinates", "ParticleIDs", "Masses", "Potential", "Velocities"]:
                data["PartType1"][key] = f["PartType1"][key][()]

        # gas
        print(" gas")
        if "PartType0" in f:
            # skipped: AGS-Softening, GrackleHI, GrackleHII, GrackleHM, GrackleHeI, GrackleHeII, GrackleHeIII,
            #          HaloID, ID_Generations, NWindLaunches, Sigma, SmoothingLength
            for key in [
                "Coordinates",
                "DelayTime",
                "Density",
                "ElectronAbundance",
                "InternalEnergy",
                "Masses",
                "NeutralHydrogenAbundance",
                "ParticleIDs",
                "Potential",
                "StarFormationRate",
                "Velocities",
            ]:
                data["PartType0"][key] = f["PartType0"][key][()]

            # add Simba_FractionH2
            data["PartType0"]["Simba_FractionH2"] = f["PartType0"]["FractionH2"][()]

        # stars
        print(" stars")
        if "PartType4" in f:
            # skipped: AGS-Softening, HaloID, ID_Generations
            for key in ["Coordinates", "Masses", "ParticleIDs", "Potential", "StellarFormationTime", "Velocities"]:
                data["PartType4"][key] = f["PartType4"][key][()]

        # gas + stars
        print(" gas+stars")
        for pt in ["PartType0", "PartType4"]:
            # handle GFM_Metals, GFM_Metallicity, and Simba analogous-dust fields
            if pt not in f:
                continue
            data[pt]["GFM_Metals"] = np.zeros((data[pt]["ParticleIDs"].size, 10), dtype="float32")
            data[pt]["Simba_DustMetals"] = np.zeros((data[pt]["ParticleIDs"].size, 10), dtype="float32")

            for i, src_ind in enumerate(metalInds):
                data[pt]["GFM_Metals"][:, i] = f[pt]["Metallicity"][:, src_ind]
                data[pt]["Simba_DustMetals"][:, i] = f[pt]["Dust_Metallicity"][:, src_ind]

            # store total metallicity separately
            data[pt]["GFM_Metallicity"] = data[pt]["GFM_Metals"][:, 0].copy()
            data[pt]["Simba_DustMetallicity"] = data[pt]["Simba_DustMetals"][:, 0].copy()

            # compute H and store (H = 1 - Z - He)
            data[pt]["GFM_Metals"][:, 0] = 1.0 - data[pt]["GFM_Metallicity"] - data[pt]["GFM_Metals"][:, 1]
            data[pt]["Simba_DustMetals"][:, 0] = (
                1.0 - data[pt]["Simba_DustMetallicity"] - data[pt]["Simba_DustMetals"][:, 1]
            )

            # compute 'total of all other i.e. untracked metals' and store
            data[pt]["GFM_Metals"][:, -1] = data[pt]["GFM_Metallicity"] - np.sum(data[pt]["GFM_Metals"][:, 2:], axis=1)
            data[pt]["Simba_DustMetals"][:, -1] = data[pt]["Simba_DustMetallicity"] - np.sum(
                data[pt]["Simba_DustMetals"][:, 2:], axis=1
            )

            # add Simba_DustMass, clip to zero
            data[pt]["Simba_DustMass"] = f[pt]["Dust_Masses"][()]
            w = np.where(data[pt]["Simba_DustMass"] < 0)
            data[pt]["Simba_DustMass"][w] = 0.0

            # fix Simba_DustMetallicity: should be either 0 (no dust) or 1 (dust)
            w = np.where(data[pt]["Simba_DustMetallicity"] < 0.5)
            data[pt]["Simba_DustMetallicity"][w] = 0.0
            w = np.where(data[pt]["Simba_DustMetallicity"] >= 0.5)
            data[pt]["Simba_DustMetallicity"][w] = 1.0

            w = np.where(data[pt]["Simba_DustMetals"] > 1.0)
            data[pt]["Simba_DustMetals"][w] = 1.0  # clip few erronous outliers
            w = np.where(data[pt]["Simba_DustMetals"] < 0.0)
            data[pt]["Simba_DustMetals"][w] = 0.0

            # convert Simba_DustMetals and Simba_DustMetallicity from mass ratio relative to total dust mass,
            # to mass ratio relative to total gas mass (for consistency with GFM_Metallicity and GFM_Metals)
            dust_mass_ratio = data[pt]["Simba_DustMass"] / data[pt]["Masses"]
            data[pt]["Simba_DustMetallicity"] *= dust_mass_ratio

            for i in range(data[pt]["Simba_DustMetals"].shape[1]):
                data[pt]["Simba_DustMetals"][:, i] *= dust_mass_ratio

            del data[pt]["Simba_DustMass"]  # redundant

        # BHs
        print(" bhs")
        if "PartType5" in f:
            # skipped: AGS-Softening, HaloID, ID_Generations, StellarFormationTime
            for key in [
                "BH_AccretionLength",
                "BH_Mass",
                "BH_Mass_AlphaDisk",
                "BH_Mdot",
                "BH_NProgs",
                "Coordinates",
                "Masses",
                "ParticleIDs",
                "Potential",
                "Velocities",
            ]:
                data["PartType5"][key] = f["PartType5"][key][()]

            # missing keys (leave blank)
            m_keys = [
                "BH_BPressure",
                "BH_CumEgyInjection_QM",
                "BH_CumMassGrowth_QM",
                "BH_CumEgyInjection_RM",
                "BH_CumMassGrowth_RM",
                "BH_Density",
                "BH_HostHaloMass",
                "BH_Pressure",
                "BH_U",
            ]
            for key in m_keys:
                data["PartType5"][key] = np.zeros(data["PartType5"]["BH_Mass"].size, dtype="float32")

    # cleanup header
    # note: Simba contains duplicate ParticleIDs (except for DM)
    #  -- particle splitting is enabled, but in MFM, all splitting is in enrichment from stars
    #  -- also, multiple stars are created from gas particles
    #  -- generated IDs are given by max(ParticleIDs,snap=0)+ProgenitorID, so if an ID is above
    #     max(ParticleIDs,snap=0) then taking the modulus by this value gives the ProgenitorID
    header["UnitLength_in_cm"] = 3.08568e21
    header["UnitMass_in_g"] = 1.989e43
    header["UnitVelocity_in_cm_per_s"] = 100000

    # field renames
    for pt in data.keys():
        for from_name, to_name in fieldRenames.items():
            if from_name in data[pt]:
                data[pt][to_name] = data[pt].pop(from_name)

    if "PartType0" in data:
        # removed in final output given its redundancy
        data["PartType0"]["CenterOfMass"] = data["PartType0"]["Coordinates"]

    # move wind particles from PT0 to PT4
    if "PartType0" in data:
        # DelayTime: non-zero for currently decoupled wind particles, in which case it gives the maximum
        # time (in code units) that the particle can remain decoupled (could also re-couple first with
        # the density criterion)
        w_wind = np.where(data["PartType0"]["DelayTime"] != 0)[0]
        w_gas = np.where(data["PartType0"]["DelayTime"] == 0)[0]

        # move wind to PT4, set properties (all PT4 datasets also exist for PT0)
        copy_keys = data["PartType4"].keys()
        if header["NumPart_Total"][4] == 0:
            # no stars exist yet: everything except GFM_InitialMass, StellarPhotometrics
            copy_keys = [
                "Coordinates",
                "GFM_Metallicity",
                "GFM_Metals",
                "GFM_StellarFormationTime",
                "Masses",
                "ParticleIDs",
                "Potential",
                "Simba_DustMetallicity",
                "Simba_DustMetals",
                "Velocities",
            ]

        for key in copy_keys:
            if key == "GFM_StellarFormationTime":
                # turn negative to meet TNG definition
                wind_data = -1.0 * data["PartType0"]["DelayTime"][w_wind]
            else:
                # all other fields
                wind_data = data["PartType0"][key][w_wind]

            if key in data["PartType4"]:
                # normal case, adding to existing stars
                axis = 0 if wind_data.ndim > 1 else None
                new_data = np.concatenate((data["PartType4"][key], wind_data), axis=axis)
            else:
                # high redshift, wind exists before the first star particle
                new_data = wind_data

            data["PartType4"][key] = new_data

        # subset gas to only real gas (we lose gas-only fields for wind, e.g. Density, InternalEnergy, and so on)
        for key in data["PartType0"].keys():
            data["PartType0"][key] = data["PartType0"][key][w_gas]

        # update part counts in header
        for key in ["NumPart_ThisFile", "NumPart_Total"]:
            header[key][0] -= w_wind.size
            header[key][4] += w_wind.size
        assert header["NumPart_Total"][0] == w_gas.size

        del data["PartType0"]["DelayTime"]

    # dark matter mass (constant)
    if "PartType1" in data:
        assert data["PartType1"]["Masses"].min() == data["PartType1"]["Masses"].max() == data["PartType1"]["Masses"][0]
        header["MassTable"][1] = data["PartType1"]["Masses"][0]
        del data["PartType1"]["Masses"]

    # fix any smbhs at exactly the same coordinate (breaks subfind)
    if 0 and snap >= 112:
        # find and fix
        dists, i1, i2 = sP.periodicPairwiseDists(data["PartType5"]["Coordinates"])

        w1 = i1[np.where(dists == 0)]
        w2 = i1[np.where(dists <= 1e-4)]

        assert np.array_equal(w1, w2)

        # take unique subset
        w1 = np.unique(w1)
        print("Found [%d] SMBHs at zero distance from some other SMBH." % len(w1))

        if len(w):
            # randomize positions within the softening length
            rng = np.random.default_rng(424242)
            pos_random = rng.uniform(0.0, 0.1, (len(w1), 3))

            data["PartType5"]["Coordinates"][w1, :] += pos_random

    # unit conversions
    for pt in data.keys():
        # if 'Velocities' in data[pt]: #no! already in sqrt(a) units, like TNG
        #    data[pt]['Velocities'] /= np.sqrt(header['Time']) # peculiar -> sqrt(a) units
        if "GFM_Metallicity" in data[pt]:
            w = np.where(data[pt]["GFM_Metallicity"] < 8e-10)
            data[pt]["GFM_Metallicity"][w] = 8e-10  # InitialAbundances > GFM_MIN_METAL

        if "GFM_Metals" in data[pt]:
            w = np.where(data[pt]["GFM_Metals"] < 1e-10)
            data[pt]["GFM_Metals"][w] = 1e-10  # InitialAbundances > GFM_MIN_METAL

        if "Simba_DustMetals" in data[pt]:
            w = np.where(data[pt]["Simba_DustMetals"] < 1e-20)
            data[pt]["Simba_DustMetals"][w] = 1e-20  # GFM_MIN_METAL

        if "Simba_DustMetallicity" in data[pt]:
            w = np.where(data[pt]["Simba_DustMetallicity"] < 1e-20)
            data[pt]["Simba_DustMetallicity"][w] = 1e-20  # GFM_MIN_METAL

    # stars: photometrics
    if header["NumPart_Total"][4] > 0:
        # construct PT4/GFM_InitialMass via BC03 interpolation (most consistent with Simba mass loss/yields)
        stars_formz = 1 / data["PartType4"]["GFM_StellarFormationTime"] - 1
        stars_logagegyr = np.log10(
            sP.units.redshiftToAgeFlat(header["Redshift"]) - sP.units.redshiftToAgeFlat(stars_formz)
        )
        stars_logz = np.log10(data["PartType4"]["GFM_Metallicity"])

        i1 = np.interp(stars_logz, gfm_photo["LogMetallicity_bins"], np.arange(gfm_photo["LogMetallicity_bins"].size))
        i2 = np.interp(stars_logagegyr, gfm_photo["LogAgeInGyr_bins"], np.arange(gfm_photo["LogAgeInGyr_bins"].size))
        iND = np.vstack((i1, i2))

        remaining_mass = map_coordinates(gfm_photo["RemainingMass"], iND, order=1, mode="nearest")
        assert remaining_mass.size == data["PartType4"]["Masses"].size

        data["PartType4"]["GFM_InitialMass"] = np.zeros(data["PartType4"]["Masses"].size, dtype="float32")

        # leave GFM_InitialMass zero for wind
        w = np.where(data["PartType4"]["GFM_StellarFormationTime"] > 0)
        data["PartType4"]["GFM_InitialMass"][w] = data["PartType4"]["Masses"][w] / remaining_mass[w]

        # derive BC03-based photometrics
        data["PartType4"]["GFM_StellarPhotometrics"] = np.zeros((data["PartType4"]["Masses"].size, 8), dtype="float32")

        stars_masslogmsun = sP.units.codeMassToLogMsun(data["PartType4"]["GFM_InitialMass"][w])

        for i, band in enumerate(photoBandsOrdered):
            mags_1msun = map_coordinates(gfm_photo["Magnitude_%s" % band], iND, order=1, mode="nearest")
            data["PartType4"]["GFM_StellarPhotometrics"][w, i] = mags_1msun[w] - 2.5 * stars_masslogmsun

    # BHs: eddington mdot
    if header["NumPart_Total"][5] > 0:
        UnitMass_over_UnitTime = 10.22

        mdot_edd = sP.units.codeBHMassToMdotEdd(data["PartType5"]["BH_Mass"], eps_r=0.1)  # msun/yr
        data["PartType5"]["BH_MdotEddington"] = mdot_edd / UnitMass_over_UnitTime  # put into TNG units

    # write
    with h5py.File(writePath, "w") as f:
        # headers
        h = f.create_group("Header")
        for key in header:
            h.attrs[key] = header[key]

        # particle groups
        for gName in data:
            g = f.create_group(gName)

            for key in data[gName]:
                g[key] = data[gName][key]

    print("Done.")


def fixSimbaSMBHs(snap=112):
    """SIMBA has some SMBHs at exactly the same position, which breaks Subfind."""
    sP = simParams("simba100", snap=snap)

    # global load
    ids = sP.bhs("ids")
    pos = sP.bhs("pos")

    # find and fix
    dists, i1, i2 = sP.periodicPairwiseDists(pos)

    w1 = i1[np.where(dists <= 1e-4)]  # i1[np.where(dists == 0)]

    # assert np.array_equal(w1,w2)

    # take unique subset
    w1 = np.unique(w1)
    print("[snap=%03d] Found [%d] SMBHs at zero distance from some other SMBH." % (snap, len(w1)))

    # randomize positions within the softening length
    rng = np.random.default_rng(424242)
    pos_random = rng.uniform(0.0, 0.01, (len(w1), 3))

    pos[w1, :] += pos_random

    # write
    nfiles = 16
    path = "/virgotng/universe/Illustris/Simba/L100n1024FP/output/snapdir_%03d/" % snap

    offset = 0

    for i in range(nfiles):
        filepath = path + "snap_%03d.%d.hdf5" % (snap, i)

        with h5py.File(filepath, "r+") as f:
            count = f["Header"].attrs["NumPart_ThisFile"][5]

            f["PartType5"]["Coordinates"][:] = pos[offset : offset + count, :]

        print(" ", filepath, offset, count)
        offset += count

    assert offset == ids.size


def testSimba(redshift=0.0):
    """Compare all snapshot fields (1d histograms) vs TNG to check unit conversions, etc."""
    # config (z=0,2,5)
    sP1 = simParams(run="tng50-3", redshift=redshift)
    sP2 = simParams(run="simba50", redshift=redshift)

    nBins = 50

    # loop over part types
    for ptNum in [0, 1, 4, 5]:
        gName = "PartType%d" % ptNum

        # get list of particle datasets
        with h5py.File(sP1.snapPath(sP1.snap, 0), "r") as f:
            fields1 = list(f[gName].keys())
        with h5py.File(sP2.snapPath(sP2.snap, 0), "r") as f:
            fields2 = list(f[gName].keys())

        for field in fields1:
            if field not in fields2:
                print("Note: %s/%s in TNG but not Simba!" % (gName, field))
        for field in fields2:
            if field not in fields1:
                print("Note: %s/%s in Simba but not TNG!" % (gName, field))

        # start pdf book
        pdf = PdfPages("compare_%s_%s_%s_%d_%d.pdf" % (gName, sP1.simName, sP2.simName, sP1.snap, sP2.snap))

        for field in set(fields1 + fields2):
            # start plot
            print(gName, field)

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(field + " [log]")
            ax.set_ylabel("N")
            ax.set_yscale("log")

            # load and histogram
            for i, sP in enumerate([sP1, sP2]):
                if (field not in fields1 and i == 0) or (field not in fields2 and i == 1):
                    print(" skip ", sP.simName)
                    continue

                vals = sP.snapshotSubsetP(ptNum, field)

                if field in ["Potential", "GFM_StellarPhotometrics"]:
                    vals *= -1

                num_zero = np.count_nonzero(np.where(vals == 0))
                num_neg = np.count_nonzero(np.where(vals < 0))
                num_inf = np.count_nonzero(np.where(~np.isfinite(vals)))

                if num_zero > 0 or num_neg > 0 or num_inf > 0:
                    print(" %s # zero = %d, negative = %d, inf = %d" % (sP.simName, num_zero, num_neg, num_inf))

                vals = vals[np.isfinite(vals) & (vals > 0)]
                vals = vals.ravel()  # 1D for all multi-D

                if field not in []:
                    vals = np.log10(vals)

                ax.hist(vals, bins=nBins, alpha=0.6, label=sP.simName)

            # finish plot
            ax.legend(loc="best")
            pdf.savefig()
            plt.close(fig)

        # some explicit multi-dimensional datasets, by dim
        if ptNum in [0, 4]:
            for field in ["GFM_Metals"]:
                print(gName, field)
                # load
                data1 = sP1.snapshotSubsetP(ptNum, field)
                data2 = sP2.snapshotSubsetP(ptNum, field)

                # loop over size of second dimension
                for i in range(data1.shape[1]):
                    print(i)
                    vals1 = data1[:, i]
                    vals2 = data2[:, i]

                    # histogram
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)

                    ax.set_xlabel(field + " [%d] [log]" % i)
                    ax.set_ylabel("N")
                    ax.set_yscale("log")

                    vals1 = np.log10(vals1)
                    vals2 = np.log10(vals2)

                    ax.hist(vals1, bins=nBins, alpha=0.6, label=sP1.simName)
                    ax.hist(vals2, bins=nBins, alpha=0.6, label=sP2.simName)

                    # finish plot
                    ax.legend(loc="best")
                    pdf.savefig()
                    plt.close(fig)

        # finish
        pdf.close()


def testSimbaCat(redshift=0.0):
    """Compare all group cat fields (1d histograms) to check unit conversions, etc."""
    # config
    nBins = 50

    sP1 = simParams(run="tng50-3", redshift=redshift)
    sP2 = simParams(run="simba50", redshift=redshift)

    for gName in ["Group", "Subhalo"]:
        # get list of halo/subhalo properties
        with h5py.File(sP2.gcPath(sP2.snap, 0), "r") as f:
            fields = list(f[gName].keys())

        # start pdf book
        pdf = PdfPages("compare_%s_%s_%s_%d_%d.pdf" % (gName, sP1.simName, sP2.simName, sP1.snap, sP2.snap))

        for field in fields:
            # start plot
            print(field)
            if field in ["SubhaloFlag", "SubhaloBfldDisk", "SubhaloBfldHalo"]:
                continue

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)

            ax.set_xlabel(field + " [log]")
            ax.set_ylabel("log N")

            # load and histogram
            for sP in [sP1, sP2]:
                if gName == "Group":
                    vals = sP.halos(field)

                if gName == "Subhalo":
                    vals = sP.subhalos(field)

                num_zero = np.count_nonzero(np.where(vals == 0))
                num_neg = np.count_nonzero(np.where(vals < 0))
                num_inf = np.count_nonzero(np.where(~np.isfinite(vals)))

                if num_zero > 0 or num_neg > 0 or num_inf > 0:
                    print(" %s # zero = %d, negative = %d, inf = %d" % (sP.simName, num_zero, num_neg, num_inf))

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
