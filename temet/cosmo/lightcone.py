"""Cosmological simulations - generating and working with lightcones (w/ Andres Aramburo-Garcia)."""

import time
from os.path import isfile

import h5py
import numpy as np

from ..util.boxRemap import findCuboidRemapInds, remapPositions
from ..util.helper import pSplitRange
from ..util.simParams import simParams


def _load(sP, group, field, inds):
    """Helper: load a subset, specified by inds, of a particle/halo/subhalo field."""
    if "PartType" in group:
        ptNum = int(group[-1])
        data = sP.snapshotSubsetC(ptNum, field, inds)
    elif group == "Subhalo":
        data = sP.subhalos(field)[inds]
    elif group == "Group":
        data = sP.halos(field)[inds]
    else:
        raise Exception("Unhandled group.")

    return data


def get_cone(sP, group, config, snap_index):
    """Transform 3D periodic positions into lightcone geometry.

    Load coordinates of a particle type, or halos/subhalos, transform into the lightcone
    geometry, subset, and optionally load additional fields for lightcone mmembers.
    Return transformed positions, velocities, additional fields, and indices back into the
    original periodic snapshots.
    """
    # dataset config
    if "PartType" in group:
        nChunks = 10
        ptNum = int(group[-1])
        numPartTot = sP.numPart[ptNum]
        data_keys = sP.snapFields(group)
    elif group == "Subhalo":
        nChunks = 1
        numPartTot = sP.numSubhalos
        data_keys = sP.groupCatFields(group)
    elif group == "Group":
        nChunks = 1
        numPartTot = sP.numHalos
        data_keys = sP.groupCatFields(group)

    # remove datasets which will become inconsistent and are unwanted
    for key in ["CenterOfMass", "SubhaloCM", "GroupCM"]:
        if key in data_keys:
            data_keys.remove(key)

    # allocate
    global_index = np.zeros(numPartTot, dtype="int64")

    offset = 0
    count = 0

    # chunk load
    for i in range(nChunks):
        # determine local subset
        if "PartType" in group:
            indRange_loc = pSplitRange([0, numPartTot - 1], nChunks, i, inclusive=True)
            pos_local = sP.snapshotSubsetP(ptNum, "Coordinates", indRange=indRange_loc)
        elif group == "Subhalo":
            # just load group catalog field directly
            pos_local = sP.subhalos("SubhaloPos")
        elif group == "Group":
            pos_local = sP.halos("GroupPos")
        else:
            raise Exception("Unhandled group.")

        # transform the coordinates, all in [ckpc/h]
        pos_remapped, _ = remapPositions(sP, pos_local, config["remapShape"], nPixels=None)

        # get the indices of the particles/cells/groups that are inside the cone
        x_min = config["dists_mid"][snap_index]
        x_max = config["dists_mid"][snap_index + 1]

        y_min = config["cone_y_pos"] - pos_remapped[:, 0] * np.tan(config["dec_rad"])
        y_max = config["cone_y_pos"] + pos_remapped[:, 0] * np.tan(config["dec_rad"])
        z_min = config["cone_z_pos"] - pos_remapped[:, 0] * np.tan(config["ra_rad"])
        z_max = config["cone_z_pos"] + pos_remapped[:, 0] * np.tan(config["ra_rad"])

        index = np.where(
            (x_min < pos_remapped[:, 0])
            & (pos_remapped[:, 0] < x_max)
            & (y_min < pos_remapped[:, 1])
            & (pos_remapped[:, 1] < y_max)
            & (z_min < pos_remapped[:, 2])
            & (pos_remapped[:, 2] < z_max)
        )[0]

        # calculate the global index of the particles
        global_index[count : count + index.size] = offset + index

        count += index.size
        offset += pos_local.shape[0]

    assert offset == numPartTot
    global_index = global_index[:count]

    # load positions and velocities
    pos_keys = ["Coordinates", "SubhaloPos", "GroupPos"]
    vel_keys = ["Velocities", "SubhaloVel", "GroupVel"]

    for key in pos_keys:
        if key in data_keys:
            pos = _load(sP, group, key, global_index)

    for key in vel_keys:
        if key in data_keys:
            vel = _load(sP, group, key, global_index)

    for key in pos_keys + vel_keys:
        if key in data_keys:
            data_keys.remove(key)

    # creating minimal lightcone? do not load any extra fields
    if config["minimal"]:
        data_keys = []

    return pos, vel, data_keys, global_index


def lightcone_coordinates(sP, group, pos, vel, config, snap_index):
    """Compute the ra, dec, and redshift given the position and velocity within the lightcone."""
    # transform coordinates, center within the lightcone field of view
    pos, _ = remapPositions(sP, pos, config["remapShape"], nPixels=None)

    pos -= np.array([config["cone_x_pos"], config["cone_y_pos"], config["cone_z_pos"]])

    # compute line-of-sight distance from ..observer to points
    r = np.sqrt(np.square(pos).sum(axis=1))  # ckpc/h

    # create mapping from co-moving distance to redshift, for quicker interpolation
    zz = np.linspace(0.0, config["max_z"], 500)
    zz_dist = sP.units.redshiftToComovingDist(zz) * 1000 * sP.HubbleParam  # kpc/h

    # convert distances to redshift
    z_cosmo = np.interp(r, zz_dist, zz)  # linear
    a_cosmo = 1 / (z_cosmo + 1)

    # convert velocities into physical [km/s]
    if group == "Group":
        pec_vel = vel
    elif group == "Subhalo":
        pec_vel = vel / a_cosmo[:, np.newaxis]
    else:  # PartType
        pec_vel = vel * np.sqrt(a_cosmo)[:, np.newaxis]

    # unit vector for direction between observer and this point
    r_norm = pos / r[:, np.newaxis]

    # contribution to redshift of the line-of-sight peculiar velocity
    z_peculiar = np.sum(r_norm * pec_vel, axis=1) / sP.units.c_km_s

    # observed redshift
    z_obs = (1 + z_cosmo) * (1 + z_peculiar) - 1

    # calculate RA and DEC
    dec = np.rad2deg(np.arctan((pos[:, 1]) / pos[:, 0]))
    ra = np.rad2deg(np.arctan((pos[:, 2]) / pos[:, 0]))

    return z_cosmo, z_obs, dec, ra


def generate_lightcone(index_todo=None):
    """Generate a lightcone from a set of saved snapshots of a cubic periodic cosmological volume.

    Our technique is to apply the volume remapping to reshape the cubic box into an elongated cuboid
    domain with significant extent along the line-of-sight distance, taken to be the x-axis, while
    (ra,dec) are mapped from the (y,z) coordinates. If index_todo is not None, then process only this
    single snapshot.
    """
    start_time = time.time()

    # config
    sP = simParams(run="tng300-1")

    max_redshift = 1.0
    onlyFullSnaps = True  # can only use full if we need e.g. MagneticField
    minimal = False  # if True, no actual particle fields other than {snap,index} are copied

    remapShape = [10.0499, 0.2985, 0.3333]

    data_groups = ["Group", "Subhalo", "PartType5", "PartType4", "PartType1", "PartType0"]

    # decide snapshots to use
    snaps = sP.validSnapList(onlyFull=onlyFullSnaps)[::-1]
    redshifts = sP.snapNumToRedshift(snaps)

    w = np.where(redshifts <= max_redshift)
    snaps = snaps[w]
    redshifts = redshifts[w]

    # we take information from each snapshot until we reach the redshift/distance halfway to the next
    # e.g. z=0.05 between snap 99 (z=0) and snap 91 (z=0.1)
    redshifts_mid = np.hstack((redshifts[0], (redshifts[1:] + redshifts[:-1]) / 2, redshifts[-1]))

    # midpoints in cosmological distance between snapshots, i.e. the point where we switch to the
    # next snapshot. note: first value is special (z=0), and last value is special (max_redshift)
    dists_mid = sP.units.redshiftToComovingDist(redshifts_mid) * 1000 * sP.HubbleParam  # ckpc/h

    # new box shape after remapping
    remapMatrix, newBoxSize = findCuboidRemapInds(remapShape)

    newBoxSize *= sP.boxSize  # relative -> ckpc/h

    # lightcone config (extent in dec and ra)
    cone_dec = np.arctan(0.5 * newBoxSize[1] / newBoxSize[0])
    cone_ra = np.arctan(0.5 * newBoxSize[2] / newBoxSize[0])

    config = {
        "cone_x_pos": 0.0,
        "cone_y_pos": newBoxSize[1] / 2,
        "cone_z_pos": newBoxSize[2] / 2,
        "simulation": sP.simName,
        "dec_rad": cone_dec,
        "ra_rad": cone_ra,
        "dec": np.rad2deg(cone_dec) * 2,
        "ra": np.rad2deg(cone_ra) * 2,
        "NumFilesPerSnapshot": len(snaps),
        "snaps": snaps,
        "minimal": minimal,
        "max_z": max_redshift,
        "redshifts": redshifts,
        "redshifts_mid": redshifts_mid,
        "dists_mid": dists_mid,
        "remapShape": remapShape,
    }

    # loop over snapshots to process
    for i, snap in enumerate(snaps):
        # process subset for parallelization?
        if index_todo is not None and i != index_todo:
            continue

        sP.setSnap(snap)

        # create save file
        saveFilename = sP.postPath + "lightcones/lightcone.%d.hdf5" % i

        if not isfile(saveFilename):
            with h5py.File(saveFilename, "w") as f:
                header = f.create_group("Header")
                for key in config:
                    header.attrs[key] = config[key]
                header.attrs["SnapNum"] = snap

        # loop over requested datasets/groups to process
        for group in data_groups:
            print(i, snap, group)

            # load and restrict to lightcone geometry
            pos, vel, data_fields, snap_index = get_cone(sP, group, config, i)

            # derive ra, dec, and redshift
            z_cosmo, z_obs, dec, ra = lightcone_coordinates(sP, group, pos, vel, config, i)

            # write datasets
            with h5py.File(saveFilename, "r+") as f:
                f["%s/ra" % group] = ra
                f["%s/dec" % group] = dec
                f["%s/z_cosmo" % group] = z_cosmo
                f["%s/z_obs" % group] = z_obs
                f["%s/SnapIndex" % group] = snap_index

                # load and write extra datasets
                for field in data_fields:
                    print(field, flush=True)
                    data = _load(sP, group, field, snap_index)
                    f[group][field] = data

    print("Done [%.1f min]." % ((time.time() - start_time) / 60))


def finalize_lightcone():
    """Write total counts in analogy to normal snapshots to facilitate loading."""
    sP = simParams(run="tng300-1")

    # load metadata
    path = sP.postPath + "lightcones/lightcone.%d.hdf5"

    with h5py.File(path % 0, "r") as f:
        header = dict(f["Header"].attrs)

    # loop over files
    groups = ["PartType%d" % i for i in range(6)]
    NumPart_Total = np.zeros(6, dtype="int64")
    NumGroups = 0
    NumSubhalos = 0

    for i in range(header["NumFilesPerSnapshot"]):
        with h5py.File(path % i, "r+") as f:
            # use z_obs dataset to get size of this group
            NumGroups_loc = f["Group"]["z_obs"].size
            NumSubhalos_loc = f["Subhalo"]["z_obs"].size

            NumGroups += NumGroups_loc
            NumSubhalos += NumSubhalos_loc

            NumPart = np.zeros(6, dtype="int64")

            for group in groups:
                if group not in f:
                    continue
                ptNum = int(group[-1])

                NumPart[ptNum] += f[group]["z_obs"].size
                NumPart_Total[ptNum] += f[group]["z_obs"].size

            # write local header
            f["Header"].attrs["NumPart_ThisFile"] = NumPart
            f["Header"].attrs["Ngroups_ThisFile"] = NumGroups_loc
            f["Header"].attrs["Nsubgroups_ThisFile"] = NumSubhalos_loc

    # loop again, write global headers
    for i in range(header["NumFilesPerSnapshot"]):
        with h5py.File(path % i, "r+") as f:
            f["Header"].attrs["NumPart_Total"] = NumPart_Total
            f["Header"].attrs["Ngroups_Total"] = NumGroups
            f["Header"].attrs["Nsubgroups_Total"] = NumSubhalos


def lightcone3DtoSkyCoords(pos, vel, sP, velType):
    """Transform 3D positions and velocities into (ra,dec,redshift) for a lightcone geometry.

    pos and vel represent particle or galaxy positions in the periodic cube.
    The observer is assumed to be placed at the origin of the cube, (0,0,0), and the view direction is hardcoded.
    """
    # comoving distance from ..observer, removing little h
    pos = sP.units.codeLengthToComovingKpc(pos)

    rr = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)  # ckpc
    xy = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)  # ckpc

    # radial velocity (verify with other methods)
    ct = pos[:, 2] / rr
    st = np.sqrt(1 - ct**2)

    cp = pos[:, 0] / xy
    sp = pos[:, 1] / xy
    vrad = vel[:, 0] * st * cp + vel[:, 1] * st * sp + vel[:, 2] * ct

    # convert to [km/s], unfortunately depends on where these code velocities came from
    if velType == "subhalo":
        vrad = sP.units.subhaloCodeVelocityToKms(vrad)
    elif velType == "group":
        vrad = sP.units.groupCodeVelocityToKms(vrad)
    elif velType == "particle":
        vrad = sP.units.particleCodeVelocityToKms(vrad)

    # redshift: cosmological plus peculiar
    z_vals = np.arange(0.0, 2.0, 0.001)  # redshifts
    dists = sP.units.redshiftToComovingDist(z_vals)

    interpolant_dist_to_z = np.interp1d(dists, z_vals, kind="cubic")
    redshift_cosmo = interpolant_dist_to_z(rr)

    redshift = redshift_cosmo + (vrad / sP.units.c_km_s) * (1.0 + redshift_cosmo)

    # transform (x,y) -> (theta,phi) -> (ra,dec)
    theta = np.arccos(pos[:, 2] / rr)
    phi = np.arctan2(pos[:, 1], pos[:, 0])

    dec = theta - np.pi / 2
    ra = phi

    return ra, dec, redshift
