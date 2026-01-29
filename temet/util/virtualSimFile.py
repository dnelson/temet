"""
Creation and update of the virtual 'simulation.hdf5' file.
"""

import glob
from os import getcwd, path

import h5py

from ..util.simParams import simParams


def _addPostprocessingCat(fSim, filepath, baseName, gNames, rootOnly=False):
    """Helper for createVirtualSimHDF5() below. Add one postprocessing catalog specification in."""
    # catalog exists?
    if not path.isfile(filepath):
        print(" MISSING [%s]! Skipping..." % filepath)
        return

    # open file
    with h5py.File(filepath, "r") as f:
        # loop over groups
        for gName in gNames:
            if gName not in f:
                print(" MISSING [%s]! Skipping..." % gName)
                continue

            # loop over fields
            for field in f[gName]:
                print(" %s %s" % (gName, field))

                if isinstance(f[gName][field], h5py.Dataset):
                    # establish virtual layout
                    shape = f[gName][field].shape
                    if len(shape) == 2 and shape[1] == 1:
                        shape = (shape[0],)  # squeeze

                    layout = h5py.VirtualLayout(shape=shape, dtype=f[gName][field].dtype)

                    layout[...] = h5py.VirtualSource(f[gName][field])

                    # add completed virtual dataset into container
                    # if len(f[gName]) == 1:
                    #    # don't think we ever get here in practice (all postprocessing catalogs have
                    #    # more than 1 dataset) (maybe with my Auxcat's)
                    #    import pdb; pdb.set_trace() # verify the following idea (no group, just dset)
                    #    key = '/%s/%s' % (baseName,field) # or (baseName,gName) ?
                    #    # doesn't work, e.g. Offsets/Group/SnapByType is only dset in Group/
                    # else:
                    #    key = '/%s/%s/%s' % (baseName,gName,field)

                    if len(gNames) == 1:
                        key = "/%s/%s" % (baseName, field)
                    else:
                        key = "/%s/%s/%s" % (baseName, gName, field)

                    if key in fSim:
                        # just redshifts,snaps of tracer_tracks for now
                        print(" skip [%s], already exists." % key)
                        continue

                    fSim.create_virtual_dataset(key, layout)
                else:
                    if rootOnly:
                        continue  # only add datasets directly in specified gNames

                    # nested group, traverse
                    assert isinstance(f[gName][field], h5py.Group)

                    for subfield in f[gName][field]:
                        print(" - %s" % subfield)

                        # establish virtual layout
                        shape = f[gName][field][subfield].shape
                        layout = h5py.VirtualLayout(shape=shape, dtype=f[gName][field][subfield].dtype)

                        layout[...] = h5py.VirtualSource(f[gName][field][subfield])

                        # add completed virtual dataset into container
                        if len(gNames) == 1:
                            key = "/%s/%s/%s" % (baseName, field, subfield)
                        else:
                            key = "/%s/%s/%s/%s" % (baseName, gName, field, subfield)

                        fSim.create_virtual_dataset(key, layout)


def createVirtualSimHDF5():
    """Create a single 'simulation.hdf5' file which is made up of virtual datasets (HDF5 1.1x/h5py 2.9.x features).

    Note: dataset details acquired from first chunk of last snapshot! Snapshot must be full, and first chunk must have
    at least one of every particle type! Note: run in simulation root dir, since we make relative path links.
    """
    sP = simParams(run="tng-cluster")
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd() or sP.simName.replace("-", "/") in getcwd()  # careful

    global_attr_skip = [
        "Ngroups_ThisFile",
        "Ngroups_Total",
        "Nids_ThisFile",
        "Nids_Total",
        "Nsubgroups_ThisFile",
        "Nsubgroups_Total",
        "NumFiles",
        "Redshift",
        "Time",
        "Composition_vector_length",
        "Flag_Cooling",
        "Flag_DoublePrecision",
        "Flag_Feedback",
        "Flag_Metals",
        "Flag_Sfr",
        "Flag_StellarAge",
        "NumFilesPerSnapshot",
        "NumPart_ThisFile",
        "NumPart_Total",
        "NumPart_Total_HighWord",
    ]
    local_attr_include = ["Ngroups_Total", "Nsubgroups_Total", "Redshift", "Time"]

    # initialize output
    fSim = h5py.File("simulation.hdf5", "w")

    snaps = sP.validSnapList()

    snapsToDo = snaps

    # two big iterations: first for snapshots, then for group catalogs
    for mode in ["snaps", "groups"]:
        print("Starting [%s]..." % mode)

        if mode == "snaps":
            chunkPath = sP.snapPath
            nChunks = sP.snapNumChunks(snaps[-1])
            gNames = ["PartType%d" % i for i in range(6)]
            baseName = "Snapshots"

        if mode == "groups":
            chunkPath = sP.gcPath
            nChunks = sP.groupCatNumChunks(snaps[-1])
            gNames = ["Group", "Subhalo"]
            baseName = "Groups"

        # acquire field names, shapes, dtypes, dimensionalities of all datasets from final snapshot
        filepath = chunkPath(snaps[-1], 0)
        print("Loading all dataset metadata from: %s" % filepath)

        shapes = {}
        dtypes = {}
        ndims = {}

        with h5py.File(filepath, "r") as f:
            for gName in gNames:
                if gName not in f:
                    continue

                shapes[gName] = {}
                dtypes[gName] = {}
                ndims[gName] = {}

                for field in f[gName].keys():
                    shapes[gName][field] = f["/%s/%s" % (gName, field)].shape
                    dtypes[gName][field] = f["/%s/%s" % (gName, field)].dtype
                    ndims[gName][field] = f["/%s/%s" % (gName, field)].ndim

            # insert global Header/Parameters/Config into root
            if mode == "snaps":
                for gName in ["Config", "Header", "Parameters"]:
                    if gName in f:
                        grp = fSim.create_group(gName)
                        for key, val in f[gName].attrs.items():
                            if key not in global_attr_skip:
                                grp.attrs[key] = val

        # loop over all snapshots
        for snap in snapsToDo:
            # load snapshot and group catalog headers
            print("snap [%3d]" % snap)
            sP.setSnap(snap)

            if sP.simName == "Illustris-1" and mode == "snaps" and snap in [53, 55]:
                print(" SKIPPING, corrupt...")
                continue
            if sP.simName == "Simba-L50n512" and mode == "snaps" and snap in [126]:
                print(" SKIPPING, corrupt...")
                continue

            if mode == "snaps":
                header = sP.snapshotHeader()
            if mode == "groups":
                header = sP.groupCatHeader()

            # loop over groups (particle types, or groups/subhalos)
            for gName in gNames:
                # get number of elements (particles of this type, or groups/subhalos)
                if mode == "snaps":
                    ptNum = int(gName[-1])
                    nPt = header["NumPart"][ptNum]
                else:
                    if gName == "Group":
                        nPt = header["Ngroups_Total"]
                    if gName == "Subhalo":
                        nPt = header["Nsubgroups_Total"]

                if nPt == 0:
                    continue  # group not present for this snapshot (or ever)

                print(" %s" % gName, flush=True)

                # loop over fields
                for field in shapes[gName].keys():
                    # print('[%3d] %s %s' % (snap,gName,field))

                    # get field dimensionality and dtype, set full field dataset size
                    if ndims[gName][field] == 1:
                        shape = (nPt,)
                    else:
                        shape = (nPt, shapes[gName][field][1])

                    # establish virtual layout
                    layout = h5py.VirtualLayout(shape=shape, dtype=dtypes[gName][field])

                    # loop over chunks
                    offset = 0
                    present = False

                    for i in range(nChunks):
                        if mode == "snaps":
                            fpath = "output/snapdir_%03d/snap_%03d.%d.hdf5" % (snap, snap, i)
                        if mode == "groups":
                            fpath = "output/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap, snap, i)

                        with h5py.File(fpath, "r") as f:
                            # attach virtual data source subset into layout
                            key = "/%s/%s" % (gName, field)

                            if key not in f:
                                continue  # empty chunk for this pt/field

                            present = True
                            vsource = h5py.VirtualSource(f[key])

                            if ndims[gName][field] == 1:
                                layout[offset : offset + vsource.shape[0]] = vsource
                            else:
                                layout[offset : offset + vsource.shape[0], :] = vsource

                            offset += vsource.shape[0]

                    if present:
                        # if not seen in any chunk, then skip (e.g. field missing from minisnap)
                        assert offset == shape[0]

                        # add completed virtual dataset into container
                        key = "/%s/%d/%s/%s" % (baseName, snap, gName, field)
                        fSim.create_virtual_dataset(key, layout)

            # add local Header attributes
            with h5py.File(chunkPath(snap, 0), "r") as f:
                if "Header" in f:
                    grp = fSim.create_group("/%s/%d/Header" % (baseName, snap))
                    for attr in local_attr_include:
                        if attr in f["Header"].attrs:
                            grp.attrs[attr] = f["Header"].attrs[attr]

            if mode == "snaps":
                grp.attrs["NumPart_Total"] = header["NumPart"]  # int64

        print("[%s] done.\n" % mode)

    # postprocessing, one file per snapshot
    modes = [
        "axisratios",
        "circularities/10Re",
        "circularities/allstars",
        # hih2
        # InfallCatalog
        # InSituFraction
        # MergerHistory
        # sizes_projected
        # skirt_images
        "StarFormationRates",
        "StellarAssembly/galaxies",
        "StellarAssembly/galaxies_in_rad",
        "StellarAssembly/stars",
        # stellar_light
        "StellarMasses/Group",
        "StellarMasses/Subhalo",
        "offsets",
        "SubhaloMatchingToDark/SubLink",
        "SubhaloMatchingToDark/LHaloTree",
    ]
    # SubhaloMatchingToIllustris
    # SubhaloMatchingToLowRes
    # VirialMassesType

    for mode in modes:
        print("Starting [%s]..." % mode)

        for snap in snapsToDo:
            # load snapshot and group catalog headers
            print("snap [%3d]" % snap)

            if mode == "axisratios":
                filepath = "postprocessing/axisratios/axisratios_%s%03d.hdf5" % (sP.simNameAlt, snap)
                baseName = "Groups/%d/Subhalo/axisratios" % snap
                gNames = ["/"]

            if mode == "circularities/10Re":
                filepath = "postprocessing/circularities/circularities_aligned_10Re_%s%03d.hdf5" % (sP.simNameAlt, snap)
                baseName = "Groups/%d/Subhalo/circularities_10Re" % snap
                gNames = ["/"]

            if mode == "circularities/allstars":
                filepath = "postprocessing/circularities/circularities_aligned_allstars_%s%03d.hdf5" % (
                    sP.simNameAlt,
                    snap,
                )
                baseName = "Groups/%d/Subhalo/circularities_allstars" % snap
                gNames = ["/"]

            if mode == "StarFormationRates":
                filepath = "postprocessing/StarFormationRates/Subhalo_SFRs_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/StarFormationRates" % snap
                gNames = ["Subhalo"]

            if mode == "StellarAssembly/galaxies":
                filepath = "postprocessing/StellarAssembly/galaxies_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/StellarAssembly" % snap
                gNames = ["/"]

            if mode == "StellarAssembly/galaxies_in_rad":
                filepath = "postprocessing/StellarAssembly/galaxies_in_rad_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/StellarAssemblyInRad" % snap
                gNames = ["/"]

            if mode == "StellarAssembly/stars":
                filepath = "postprocessing/StellarAssembly/stars_%03d.hdf5" % snap
                baseName = "Snapshots/%d/PartType4/StellarAssembly" % snap
                gNames = ["/"]

            if mode == "StellarMasses/Group":
                filepath = "postprocessing/StellarMasses/Group_3DStellarMasses_%03d.hdf5" % snap
                baseName = "Groups/%d/Group/StellarMasses" % snap
                gNames = ["Group"]

            if mode == "StellarMasses/Subhalo":
                filepath = "postprocessing/StellarMasses/Subhalo_3DStellarMasses_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/StellarMasses" % snap
                gNames = ["Subhalo"]

            if mode == "offsets":
                # could instead move into Snapshots/N/Group/Offsets/ and Snapshots/N/Subhalo/Offsets/
                filepath = "postprocessing/offsets/offsets_%03d.hdf5" % snap
                baseName = "Offsets/%d" % snap
                gNames = ["Group", "Subhalo"]

            if mode == "SubhaloMatchingToDark/SubLink":
                filepath = "postprocessing/SubhaloMatchingToDark/SubLink_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/SubhaloMatchingToDark/SubLink" % snap
                gNames = ["/"]

            if mode == "SubhaloMatchingToDark/LHaloTree":
                filepath = "postprocessing/SubhaloMatchingToDark/LHaloTree_%03d.hdf5" % snap
                baseName = "Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree" % snap
                gNames = ["/"]

            _addPostprocessingCat(fSim, filepath, baseName, gNames)

        print("[%s] done.\n" % mode)

    # postprocesisng, one file per simulation
    modes_sim = ["trees/SubLink", "trees/SubLink_gal"]
    #'trees/LHaloTree'] # terrible structure with too many groups...
    # SubboxSubhaloList

    for mode in modes_sim:
        print("Starting [%s]..." % mode)

        if mode == "trees/SubLink":
            filepath = "postprocessing/trees/SubLink/tree_extended.hdf5"
            baseName = "Trees/SubLink"
            gNames = ["/"]

        if mode == "trees/SubLink_gal":
            filepath = "postprocessing/trees/SubLink_gal/tree_extended.hdf5"
            baseName = "Trees/SubLink_gal"
            gNames = ["/"]

        _addPostprocessingCat(fSim, filepath, baseName, gNames)

        print("[%s] done.\n" % mode)

    # tracer_tracks: custom addition into PartType2 (final snapshot only)
    print("Starting [tracer_tracks]...")

    if 1:
        # meta
        snap = snaps[-1]

        filepath = "postprocessing/tracer_tracks/tr_all_groups_%d_meta.hdf5" % snap

        _addPostprocessingCat(fSim, filepath, "Groups/%d/Group" % snap, ["Halo"])
        _addPostprocessingCat(fSim, filepath, "Groups/%d/Subhalo" % snap, ["Subhalo"])

        _addPostprocessingCat(fSim, filepath, "Snapshots/%d/PartType2" % snap, ["/"], rootOnly=True)

        # add all other known tracks
        filepaths = glob.glob("postprocessing/tracer_tracks/tr_all_groups*.hdf5")
        for filepath in filepaths:
            if "_meta" in filepath:
                continue

            _addPostprocessingCat(fSim, filepath, "Snapshots/%d/PartType2" % snap, ["/"])

    print("[tracer_tracks] done.\n")

    print("All done.")
    fSim.close()


def supplementVirtualSimHDF5AddSnapField():
    """Add to existing 'simulation.hdf5' file (modify as needed, careful!)."""
    sP = simParams(res=1080, run="tng")
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd()  # careful

    # open (append mode)
    fSim = h5py.File("simulation.hdf5", "r+")
    snaps = sP.validSnapList()

    # start custom
    chunkPath = sP.snapPath
    nChunks = sP.snapNumChunks(snaps[-1])
    gName = "PartType4"
    field = "StellarHsml"
    baseName = "Snapshots"

    # acquire field name, shape, dtype of dataset from final snapshot
    filepath = chunkPath(snaps[-1], 0)

    with h5py.File(filepath, "r") as f:
        shape = f["/%s/%s" % (gName, field)].shape
        dtype = f["/%s/%s" % (gName, field)].dtype
        ndim = f["/%s/%s" % (gName, field)].ndim

    # loop over all snapshots
    for snap in snaps:
        # load snapshot and group catalog headers
        print("snap [%3d]" % snap)
        sP.setSnap(snap)

        if sP.simName == "Illustris-1" and snap in [53, 55]:
            print(" SKIPPING, corrupt...")
            continue

        header = sP.snapshotHeader()

        # get number of particles of this type
        ptNum = int(gName[-1])
        nPt = header["NumPart"][ptNum]

        if nPt == 0:
            continue  # group not present for this snapshot (or ever)

        # set full field dataset size
        if ndim == 1:
            shape = (nPt,)
        else:
            shape = (nPt, shape[1])

        # establish virtual layout
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

        # loop over chunks
        offset = 0
        present = False

        for i in range(nChunks):
            fpath = "output/snapdir_%03d/snap_%03d.%d.hdf5" % (snap, snap, i)

            with h5py.File(fpath, "r") as f:
                # attach virtual data source subset into layout
                key = "/%s/%s" % (gName, field)

                if key not in f:
                    continue  # empty chunk for this pt/field

                present = True
                vsource = h5py.VirtualSource(f[key])

                if ndim == 1:
                    layout[offset : offset + vsource.shape[0]] = vsource
                else:
                    layout[offset : offset + vsource.shape[0], :] = vsource

                offset += vsource.shape[0]

        if present:
            # if not seen in any chunk, then skip (e.g. field missing from minisnap)
            assert offset == shape[0]

            # completed virtual dataset: already exists?
            key = "/%s/%d/%s/%s" % (baseName, snap, gName, field)

            if key in fSim:
                # del fSim[key] # remove link
                import pdb

                pdb.set_trace()  # check

            # add completed virtual dataset into container
            fSim.create_virtual_dataset(key, layout)

    # finish custom
    fSim.close()


def supplementVirtualSimHDF5AddOrUpdateGroupcatField():
    """Add to existing 'simulation.hdf5' file (modify as needed, careful!)."""
    sP = simParams(res=1820, run="illustris")
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd()  # careful

    # open (append mode)
    fSim = h5py.File("simulation.hdf5", "r+")
    snaps = sP.validSnapList()

    # start custom
    chunkPath = sP.gcPath
    nChunks = sP.groupCatNumChunks(snaps[-1])
    gName = "Subhalo"
    field = "SubhaloFlag"
    baseName = "Groups"

    # acquire field name, shape, dtype of dataset from final snapshot
    filepath = chunkPath(snaps[-1], 0)

    with h5py.File(filepath, "r") as f:
        shape = f["/%s/%s" % (gName, field)].shape
        dtype = f["/%s/%s" % (gName, field)].dtype
        ndim = f["/%s/%s" % (gName, field)].ndim

    # loop over all snapshots
    for snap in snaps:
        # load snapshot and group catalog headers
        print("snap [%3d]" % snap)
        sP.setSnap(snap)

        header = sP.groupCatHeader()

        # get number of elements (particles of this type, or groups/subhalos)
        if gName == "Group":
            nPt = header["Ngroups_Total"]
        if gName == "Subhalo":
            nPt = header["Nsubgroups_Total"]

        if nPt == 0:
            continue  # group not present for this snapshot (or ever)

        # get field dimensionality and dtype, set full field dataset size
        if ndim == 1:
            shape = (nPt,)
        else:
            shape = (nPt, shape[1])

        # establish virtual layout
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

        # loop over chunks
        offset = 0
        present = False

        for i in range(nChunks):
            fpath = "output/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5" % (snap, snap, i)

            with h5py.File(fpath, "r") as f:
                # attach virtual data source subset into layout
                key = "/%s/%s" % (gName, field)

                if key not in f:
                    continue  # empty chunk for this pt/field

                present = True
                vsource = h5py.VirtualSource(f[key])

                if ndim == 1:
                    layout[offset : offset + vsource.shape[0]] = vsource
                else:
                    layout[offset : offset + vsource.shape[0], :] = vsource

                offset += vsource.shape[0]

        if present:
            # if not seen in any chunk, then skip (e.g. field missing from minisnap)
            assert offset == shape[0]

            # completed virtual dataset: already exists?
            key = "/%s/%d/%s/%s" % (baseName, snap, gName, field)

            if key in fSim:
                del fSim[key]  # remove link
                print(" - removed existing [%s] and created new." % key)

            # add completed virtual dataset into container
            fSim.create_virtual_dataset(key, layout)

    # finish custom
    fSim.close()


def supplementVirtualSimHDF5():
    """Add to existing 'simulation.hdf5' file (modify as needed, careful!)."""
    sP = simParams(run="tng50-2")
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd()  # careful

    # open (append mode)
    fSim = h5py.File("simulation.hdf5", "r+")
    snaps = sP.validSnapList()

    # start custom
    if 0:
        # add missing Offsets
        for snap in snaps:
            # load snapshot and group catalog headers
            print("snap [%3d]" % snap)

            filepath = "postprocessing/offsets/offsets_%03d.hdf5" % snap
            baseName = "Offsets/%d" % snap
            gNames = ["Group", "Subhalo"]

            _addPostprocessingCat(fSim, filepath, baseName, gNames)

    if 0:
        # add StellarAssembly (subhalo) catalogs
        for snap in snaps:
            # filepath = 'postprocessing/StellarAssembly/galaxies_%03d.hdf5' % snap
            # baseName = 'Groups/%d/Subhalo/StellarAssembly' % snap
            # gNames = ['/']
            # _addPostprocessingCat(fSim,filepath,baseName,gNames)

            filepath = "postprocessing/StellarAssembly/galaxies_in_rad_%03d.hdf5" % snap
            baseName = "Groups/%d/Subhalo/StellarAssemblyInRad" % snap
            gNames = ["/"]
            _addPostprocessingCat(fSim, filepath, baseName, gNames)

    if 0:
        # add StellarMasses (subhalo) catalogs
        for snap in snaps:
            # filepath = 'postprocessing/StellarMasses/Subhalo_3DStellarMasses_%03d.hdf5' % snap
            # baseName = 'Groups/%d/Subhalo/StellarMasses' % snap
            # gNames = ['Subhalo']
            # _addPostprocessingCat(fSim,filepath,baseName,gNames)

            # StarFormationRates
            filepath = "postprocessing/StarFormationRates/Subhalo_SFRs_%03d.hdf5" % snap
            baseName = "Groups/%d/Subhalo/StarFormationRates" % snap
            gNames = ["Subhalo"]
            _addPostprocessingCat(fSim, filepath, baseName, gNames)

    if 0:
        # add new LHaloTree catalogs and delete old
        for snap in snaps:
            print("snap [%3d]" % snap)

            filepath = "postprocessing/SubhaloMatchingToDark/LHaloTree_%03d.hdf5" % snap
            baseName = "Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree" % snap
            gNames = ["/"]

            _addPostprocessingCat(fSim, filepath, baseName, gNames)

        for snap in snaps:
            gName = "/Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree/" % snap
            keys = ["SubhaloIndexFrom", "SubhaloIndexTo"]

            for key in keys:
                if gName + key in fSim:
                    print("delete [%s]" % (gName + key))
                    del fSim[gName + key]

    if 0:
        # remove Bfld group cat fields from mini-snaps
        tngFullSnaps = [2, 3, 4, 6, 8, 11, 13, 17, 21, 25, 33, 40, 50, 59, 67, 72, 78, 84, 91, 99]
        fields = ["SubhaloBfldDisk", "SubhaloBfldHalo"]

        for snap in snaps:
            if snap in tngFullSnaps:
                continue
            for field in fields:
                key = "/Groups/%d/Subhalo/%s" % (snap, field)

                if key in fSim:
                    print("delete [%s]" % (key))
                    del fSim[key]

    if 0:
        # add Trees/
        modes_sim = ["trees/SubLink", "trees/SubLink_gal"]
        #'trees/LHaloTree'] # terrible structure with too many groups...

        for mode in modes_sim:
            if mode == "trees/SubLink":
                filepath = "postprocessing/trees/SubLink/tree_extended.hdf5"
                baseName = "Trees/SubLink"
                gNames = ["/"]

            if mode == "trees/SubLink_gal":
                filepath = "postprocessing/trees/SubLink_gal/tree_extended.hdf5"
                baseName = "Trees/SubLink_gal"
                gNames = ["/"]

            _addPostprocessingCat(fSim, filepath, baseName, gNames)

    if 0:
        # change to re-ordered tracer_tracks, first delete old then add new
        gName = "/Snapshots/99/PartType2/"
        for key in fSim[gName]:
            if key in ["ParentIDs", "TracerIDs", "redshifts", "snaps"]:
                continue
            print(f"delete old: {key}")
            del fSim[gName + key]

        filepaths = glob.glob("postprocessing/tracer_tracks/tr_all_groups*.hdf5")
        for filepath in filepaths:
            if "_meta" in filepath:
                continue

            _addPostprocessingCat(fSim, filepath, "Snapshots/99/PartType2", ["/"])

    # finish custom

    fSim.close()
