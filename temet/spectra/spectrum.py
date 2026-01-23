"""
Synthetic absorption spectra: generation.
"""

import glob
import threading
from os import mkdir, unlink
from os.path import isdir, isfile

import h5py
import numpy as np
from numba import jit

from ..cosmo.cloudy import cloudyIon
from ..spectra.util import (
    _equiv_width,
    _v90,
    _voigt_tau,
    create_wavelength_grid,
    instruments,
    line_params,
    lines,
    lsf_matrix,
    resample_spectrum,
    sP_units_boltzmann,
    sP_units_c_km_s,
    sP_units_mass_proton,
    sP_units_Mpc_in_cm,
    varconvolve,
)
from ..util.helper import pSplitRange
from ..util.voronoiRay import rayTrace


# default configuration for ray generation
# projAxis_def = 2
# nRaysPerDim_def = 2000 # 10000 for frm_los
# raysType_def = 'voronoi_rndfullbox'

projAxis_def = 2
nRaysPerDim_def = 1000
raysType_def = "voronoi_fullbox"


def generate_rays_voronoi_fullbox(
    sP,
    projAxis=projAxis_def,
    nRaysPerDim=nRaysPerDim_def,
    raysType=raysType_def,
    subhaloIDs=None,
    pSplit=None,
    integrateQuant=None,
    search=False,
):
    """Generate a large grid of (fullbox) rays by ray-tracing through the Voronoi mesh.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
      subhaloIDs (list): if raysType is 'sample_localized' (only), then a list of subhalo IDs.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      integrateQuant (str): if None, save rays for future use. otherwise, directly perform and save the
        integral of the specified gas quantity along each ray.
      search (bool): if True, return existing data only, do not calculate new files.
    """
    # paths and save file
    if not isdir(sP.derivPath + "rays"):
        mkdir(sP.derivPath + "rays")

    iqStr = "_%s" % integrateQuant if integrateQuant is not None else ""
    filename = "%s%s_n%dd%d_%03d.hdf5" % (raysType, iqStr, nRaysPerDim, projAxis, sP.snap)

    if pSplit is not None:
        filename = "%s%s_n%dd%d_%03d-split-%d-%d.hdf5" % (
            raysType,
            iqStr,
            nRaysPerDim,
            projAxis,
            sP.snap,
            pSplit[0],
            pSplit[1],
        )

    path = sP.derivPath + "rays/" + filename

    if not isfile(path) and isfile(sP.postPath + "AbsorptionSightlines/" + filename):
        # check also existing files in permanent, publicly released postprocessing/
        path = sP.postPath + "AbsorptionSightlines/" + filename

    # total requested pathlength (equal to box length)
    total_dl = sP.boxSize

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype="float64")
    ray_dir[projAxis] = 1.0

    inds = list({0, 1, 2} - {projAxis})  # e.g. [0,1] for projAxis == 2

    # check existence
    if isfile(path):
        print("Loading [%s]." % path)

        # TODO: if subhaloIDs is not None, verify consistent with existing file

        if integrateQuant is not None:
            with h5py.File(path, "r") as f:
                # integral results
                result = f["result"][()]
                ray_pos = f["ray_pos"][()]

                # metadata
                attrs = {}
                for attr in f.attrs:
                    attrs[attr] = f.attrs[attr]

            return result, ray_pos, ray_dir, attrs["total_dl"]

        with h5py.File(path, "r") as f:
            # ray results
            rays_off = f["rays_off"][()]
            rays_len = f["rays_len"][()]
            rays_dl = f["rays_dl"][()]
            rays_inds = f["rays_inds"][()]

            # ray config
            cell_inds = f["cell_inds"][()] if "cell_inds" in f else None
            ray_pos = f["ray_pos"][()]

            # metadata
            attrs = {}
            for attr in f.attrs:
                attrs[attr] = f.attrs[attr]

        return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, attrs["total_dl"]

    if search:
        # file does not exist, but we are only searching for existing files, so empty return
        return

    pSplitStr = " (split %d of %d)" % (pSplit[0], pSplit[1]) if pSplit is not None else ""
    print("Compute and save rays: [%s z=%.1f] [%s]%s" % (sP.simName, sP.redshift, raysType, pSplitStr))
    print("Total number of rays: %d x %d = %d" % (nRaysPerDim, nRaysPerDim, nRaysPerDim**2))

    # spatial decomposition
    nRaysPerDimOrig = nRaysPerDim

    if pSplit is not None and raysType != "sample_localized":
        nPerDimErr = np.abs(np.sqrt(pSplit[1]) - np.round(np.sqrt(pSplit[1])))
        assert nPerDimErr < 1e-6, "pSplitSpatial: Total number of jobs should have integer sqroot, e.g. 9, 16, 25, 64."
        nPerDim = int(np.sqrt(pSplit[1]))
        extent = sP.boxSize / nPerDim

        # [x,y] bounds of this spatial subset e.g. if projection direction is [z]
        ij = np.unravel_index(pSplit[0], (nPerDim, nPerDim))
        xmin = ij[0] * extent
        xmax = (ij[0] + 1) * extent
        ymin = ij[1] * extent
        ymax = (ij[1] + 1) * extent

        # number of rays in this spatial subset
        nRaysPerDim = nRaysPerDim / np.sqrt(pSplit[1])
        assert nRaysPerDim.is_integer(), "pSplitSpatial: nRaysPerDim is not divisable by sqroot(total number of jobs)."
        nRaysPerDim = int(nRaysPerDim)

        print(
            " pSplitSpatial: [%d of %d] ij (%d %d) extent [%g] x [%.1f - %.1f] y [%.1f - %.1f]"
            % (pSplit[0], pSplit[1], ij[0], ij[1], extent, xmin, xmax, ymin, ymax)
        )
        print(" subset of rays: %d x %d = %d" % (nRaysPerDim, nRaysPerDim, nRaysPerDim**2))
    else:
        xmin = ymin = 0.0
        xmax = ymax = sP.boxSize

    # ray starting positions
    if raysType == "voronoi_fullbox":
        # evenly spaced (skip last, which will be duplicate with first)
        numrays = nRaysPerDim**2

        xpts = np.linspace(xmin, xmax, nRaysPerDim + 1)[:-1]
        ypts = np.linspace(ymin, ymax, nRaysPerDim + 1)[:-1]

        xpts, ypts = np.meshgrid(xpts, ypts, indexing="ij")

    if raysType == "voronoi_rndfullbox":
        # stable, random
        numrays = nRaysPerDim**2

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res)

        xpts = rng.uniform(low=xmin, high=xmax, size=nRaysPerDim**2)
        ypts = rng.uniform(low=ymin, high=ymax, size=nRaysPerDim**2)

    if raysType == "sample_localized" and pSplit is None:
        # localized (e.g. <= rvir) sightlines around a given sample of subhalos, specified by a list of
        # subhaloIDs, taking nRaysPerDim**2 sightlines around each subhalo
        assert subhaloIDs is not None, "Error: For [sample_localized], specify subhaloIDs."

        numrays = nRaysPerDim**2 * len(subhaloIDs)
        virRadFactor = 1.5  # out to this factor times r200c in impact parameter

        # local pathlength? if None, then keep the full box
        # note: must be a constant, so is computed as the average across the subhaloIDs
        total_dl_local = 2.0  # plus/minus this factor times r200c in line-of-sight direction

        # load subhalo metadata
        SubhaloPos = sP.subhalos("SubhaloPos")
        grnr = sP.subhalos("SubhaloGrNr")[subhaloIDs]
        r200c = sP.halos("Group_R_Crit200")[grnr]

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res)

        xpts = np.zeros(numrays, dtype="float32")
        ypts = np.zeros(numrays, dtype="float32")
        zpts = np.zeros(numrays, dtype="float32")

        r200c_avg = r200c.mean()

        extent = np.max([virRadFactor, total_dl_local * 2]) * r200c.max()  # max

        for i, subhaloID in enumerate(subhaloIDs):
            randomAngle = rng.uniform(0, 2 * np.pi, nRaysPerDim**2)
            randomDistance = rng.uniform(0, virRadFactor * r200c[i], nRaysPerDim**2)

            offset = i * nRaysPerDim**2
            xpts[offset : offset + nRaysPerDim**2] = randomDistance * np.cos(randomAngle)
            ypts[offset : offset + nRaysPerDim**2] = randomDistance * np.sin(randomAngle)

            xpts[offset : offset + nRaysPerDim**2] += SubhaloPos[subhaloID, inds[0]]
            ypts[offset : offset + nRaysPerDim**2] += SubhaloPos[subhaloID, inds[1]]

            if total_dl_local is not None:
                zpts[offset : offset + nRaysPerDim**2] = -total_dl_local * r200c_avg
                zpts[offset : offset + nRaysPerDim**2] += SubhaloPos[subhaloID, projAxis]

                total_dl = total_dl_local * 2 * r200c_avg  # constant

    if raysType == "sample_localized" and pSplit is not None:
        # localized (e.g. <= rvir) sightlines around a given sample of subhalos, specified by a list of
        # subhaloIDs, taking nRaysPerDim**2 sightlines around each subhalo
        assert subhaloIDs is not None, "Error: For [sample_localized], specify subhaloIDs."
        assert pSplit[1] == len(subhaloIDs), "Error: pSplit size needs to equal subhaloIDs length."

        numrays = nRaysPerDim**2
        virRadFactor = 1.5  # out to this factor times r200c in impact parameter

        # local pathlength? if None, then keep the full box
        total_dl_local = 2.0  # plus/minus this factor times r200c in line-of-sight direction

        # load subhalo metadata
        subhaloID = subhaloIDs[pSplit[0]]

        rng = np.random.default_rng(424242 + nRaysPerDim + sP.snap + sP.res + subhaloID)

        subhalo = sP.subhalo(subhaloID)
        halo = sP.halo(subhalo["SubhaloGrNr"])
        SubhaloPos = subhalo["SubhaloPos"]
        r200c = halo["Group_R_Crit200"]

        extent = np.max([virRadFactor * 2, total_dl_local * 2]) * r200c  # max

        # generate sample of sightlines around this subhalo
        randomAngle = rng.uniform(0, 2 * np.pi, nRaysPerDim**2)
        randomDistance = rng.uniform(0, virRadFactor * r200c, nRaysPerDim**2)

        xpts = randomDistance * np.cos(randomAngle) + SubhaloPos[inds[0]]
        ypts = randomDistance * np.sin(randomAngle) + SubhaloPos[inds[1]]
        zpts = np.zeros(numrays, dtype="float32")

        if total_dl_local is not None:
            zpts += SubhaloPos[projAxis] - total_dl_local * r200c

            total_dl = total_dl_local * 2 * r200c

            print(subhaloID, r200c, total_dl_local, total_dl, zpts[0])

    # construct [N,3] list of ray starting locations
    ray_pos = np.zeros((numrays, 3), dtype="float64")

    ray_pos[:, inds[0]] = xpts.ravel()
    ray_pos[:, inds[1]] = ypts.ravel()
    ray_pos[:, projAxis] = zpts.ravel() if raysType == "sample_localized" else 0.0

    sP.correctPeriodicPosVecs(ray_pos)

    # determine spatial mask (cuboid with long side equal to boxlength in line-of-sight direction)
    if pSplit is not None:
        mask = np.zeros(sP.numPart[sP.ptNum("gas")], dtype="int8")
        mask += 1  # all required

        print(" pSplitSpatial:", end="")
        for ind, axis in enumerate([["x", "y", "z"][i] for i in inds]):
            print(" slice[%s]..." % axis, end="")
            dists = sP.snapshotSubsetP("gas", "pos_" + axis, float32=True)

            if raysType == "sample_localized":
                dists = SubhaloPos[ind] - dists  # 1D, along axis, from position of pSplit-targeted subhalo
                uniform_frac = (extent / sP.boxSize) ** (1 / 3)
            else:
                dists = (ij[ind] + 0.5) * extent - dists  # 1D, along axis, from center of subregion
                uniform_frac = 1 / pSplit[1]

            sP.correctPeriodicDistVecs(dists)

            # compute maxdist heuristic (in code units): the largest 1d distance we need for the calculation
            # second term: comfortably exceed size of largest (IGM) cells (~200 kpc for TNG100-1)
            maxdist = extent / 2 + sP.gravSoft * 1000

            w_spatial = np.where(np.abs(dists) > maxdist)
            mask[w_spatial] = 0  # outside bounding box along this axis

        cell_inds = np.nonzero(mask)[0]
        print(
            "\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%"
            % (cell_inds.size / mask.size * 100, uniform_frac * 100)
        )

        dists = None
        w_spatial = None
        mask = None
    else:
        # global load
        cell_inds = np.arange(sP.numPart[sP.ptNum("gas")])

    # load (reduced) cell spatial positions
    cell_pos = sP.snapshotSubsetC("gas", "pos", inds=cell_inds, verbose=True)

    # ray-trace and compute/save integral only
    if integrateQuant is not None:
        # load gas quantity
        loadQuant = integrateQuant
        if loadQuant.endswith("_los"):
            loadQuant = loadQuant.replace("_los", "") + "_" + ["x", "y", "z"][projAxis]

        cell_values = sP.snapshotSubsetC("gas", loadQuant, inds=cell_inds, verbose=True)  # units unchanged

        # integrate
        result = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, quant=cell_values, mode="quant_dx_sum")

        # special cases
        if integrateQuant == "frm_los":
            # unit conversion [code length] -> [pc] for pathlengths, such that the FRM is in [rad m^-2]
            result *= sP.units.codeLengthToPc(1.0)

        # save
        path = spectra_filepath(
            sP, ion=integrateQuant, projAxis=projAxis, nRaysPerDim=nRaysPerDimOrig, raysType=raysType, pSplit=pSplit
        )
        with h5py.File(path, "w") as f:
            f["result"] = result
            f["ray_pos"] = ray_pos

            f.attrs["nRaysPerDim"] = nRaysPerDim
            f.attrs["projAxis"] = projAxis
            f.attrs["ray_dir"] = ray_dir
            f.attrs["total_dl"] = total_dl

        print("Saved: [%s]" % path)

        return result, ray_pos, ray_dir, total_dl

    # full ray-trace to save rays
    print("Load done, tracing...", flush=True)

    rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode="full")

    # save
    with h5py.File(path, "w") as f:
        # ray results
        f["rays_off"] = rays_off
        f["rays_len"] = rays_len
        f["rays_dl"] = rays_dl
        f["rays_inds"] = rays_inds

        # indices index a spatial subset of the snapshot
        if cell_inds is not None:
            f["cell_inds"] = cell_inds

        # ray config and metadata
        f["ray_pos"] = ray_pos

        f.attrs["nRaysPerDim"] = nRaysPerDim
        f.attrs["projAxis"] = projAxis
        f.attrs["ray_dir"] = ray_dir
        f.attrs["total_dl"] = total_dl

    print("Saved: [%s]" % path)

    return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl


def generate_spectra_from_saved_rays(
    sP,
    ion="Si II",
    instrument="4MOST-HRS",
    nRaysPerDim=nRaysPerDim_def,
    raysType=raysType_def,
    subhaloIDs=None,
    pSplit=None,
    solar=False,
):
    """Generate a large number of spectra, based on already computed and saved rays.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
      subhaloIDs (ndarray[int]): if raysType is 'sample_localized' (only), then a list of subhalo IDs.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead
        use the (constant) solar value.
    """
    # adapt idealized grid to span (redshifted) central wavelength (optional, save space)
    if instrument == "idealized":
        wave_min_ion = np.inf
        wave_max_ion = 0.0

        for _, props in lines.items():
            if props["ion"] == ion:
                wave_min_ion = min(wave_min_ion, props["wave0"])
                wave_max_ion = max(wave_max_ion, props["wave0"])

        # note: must be int or float64, dangerous to be float32, can lead to
        # bizarre rounding issues in np.linspace during creation of master grid
        wave_min = int(np.floor((wave_min_ion * (1 + sP.redshift) - 50) / 100) * 100)
        wave_max = int(np.ceil((wave_max_ion * (1 + sP.redshift) + 50) / 100) * 100)

        if wave_min < 0:
            wave_min = 0
        instruments["idealized"]["wave_min"] = wave_min
        instruments["idealized"]["wave_max"] = wave_max

    # adapt master grid to span instrumental grid (optional, save some memory/efficiency)
    instruments["master"]["wave_min"] = instruments[instrument]["wave_min"] - 100
    instruments["master"]["wave_max"] = instruments[instrument]["wave_max"] + 100
    if instruments["master"]["wave_min"] < 0:
        instruments["master"]["wave_min"] = -10.0

    if 1:
        # if 10^K gas for this ion produces unresolved absorption lines, make master grid higher resolution
        temp = 1e4  # K
        ion_amu = {el["symbol"]: el["mass"] for el in cloudyIon._el}[ion.split(" ")[0]]
        ion_mass = ion_amu * sP_units_mass_proton  # g

        b = np.sqrt(2 * sP_units_boltzmann * temp / ion_mass) / 1e5  # km/s

        # check that master grid resolution is sufficient
        lineNames = [k for k, v in lines.items() if lines[k]["ion"] == ion]  # all transitions of this ion
        wave0 = lines[lineNames[0]]["wave0"]  # Angstrom
        b_dwave = b / sP_units_c_km_s * wave0  # v/c = dwave/wave

        if b_dwave < instruments["master"]["dwave"] * 10:
            print("NOTE: b_dwave is too small for the dwave_master, setting dwave_master 10x higher!")
            instruments["master"]["dwave"] /= 10

    # sample master grid
    wave_mid, _, tau = create_wavelength_grid(instrument=instrument)

    # list of lines to process for this ion
    lineCandidates = [k for k, v in lines.items() if lines[k]["ion"] == ion]  # all transitions of this ion

    # is (redshifted) line outside of the instrumental wavelength range? then skip
    lineNames = []

    for line in lineCandidates:
        wave_z = lines[line]["wave0"] * (1 + sP.redshift)
        if wave_z < wave_mid.min() or wave_z > wave_mid.max():
            print(
                f" [{line}] wave0 = {lines[line]['wave0']:.4f} at {wave_z = :.4f} outside of "
                + f"{instrument} spec range [{wave_mid.min():.1f} - {wave_mid.max():.1f}], skipping."
            )
            continue
        print(f" [{line}] wave0 = {lines[line]['wave0']:.4f} at {wave_z = :.4f} to compute.")
        lineNames.append(line)

    # save file
    saveFilename = spectra_filepath(
        sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=pSplit, solar=solar
    )
    saveFilenameConcat = spectra_filepath(
        sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None, solar=solar
    )

    if not isdir(sP.derivPath + "spectra/"):
        mkdir(sP.derivPath + "spectra/")

    # does save already exist, with all lines done?
    existing_lines = []

    if isfile(saveFilenameConcat):
        print(f"Final save [{saveFilenameConcat.split('/')[-1]}] already exists! Exiting.")
        return

    all_lines_done = False
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            # which lines are already done?
            existing_lines = [k.replace("EW_", "").replace("_", " ") for k in f.keys() if "EW_" in k]
            flux_done = "flux" in f and f.attrs.get("flux_done", False)

        all_lines_done = all(line in existing_lines for line in lineNames)
        all_done = all(line in existing_lines for line in lineNames) & flux_done
        if all_done:
            print(f"Save [{saveFilename.split('/')[-1]}] already exists and is done, exiting.")
            return

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = generate_rays_voronoi_fullbox(
        sP, nRaysPerDim=nRaysPerDim, raysType=raysType, subhaloIDs=subhaloIDs, pSplit=pSplit
    )

    if not all_lines_done:
        # load required gas cell properties
        projAxis = list(ray_dir).index(1)
        velLosField = "vel_" + ["x", "y", "z"][projAxis]

        cell_vellos = sP.snapshotSubsetP("gas", velLosField, inds=cell_inds)  # code
        cell_temp = sP.snapshotSubsetP("gas", "temp_sfcold", inds=cell_inds)  # K

        cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos)  # km/s

        # convert length units, all other units already appropriate
        rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # (re)start output
    EWs = {}
    N = {}
    v90 = {}
    densField = None

    with h5py.File(saveFilename, "a") as f:
        # not restarting? save metadata now
        if "wave" not in f:
            f["wave"] = wave_mid
            f["ray_pos"] = ray_pos
            f["ray_dir"] = ray_dir
            f["ray_total_dl"] = total_dl

            f.attrs["simName"] = sP.simName
            f.attrs["redshift"] = sP.redshift
            f.attrs["snapshot"] = sP.snap
            f.attrs["instrument"] = instrument
            f.attrs["lineNames"] = lineNames
            f.attrs["count"] = ray_pos.shape[0]

    # loop over requested line(s)
    for i, line in enumerate(lineNames):
        # load ion abundances per cell, unless we already have
        print(
            f"[{i + 1:02d}] of [{len(lineNames):02d}] computing: [{line}]",
            f"wave0 = {lines[line]['wave0']:.4f} at {wave_z = :.4f}",
            flush=True,
        )

        if line in existing_lines:
            print(" already exists, skipping...")
            continue

        # do we not already have the ion density loaded?
        if densField is None or lines[line]["ion"] != lines[lineNames[0]]["ion"]:
            densField = "%s numdens" % lines[line]["ion"]
            if solar:
                densField += "_solar"

            cell_dens = sP.snapshotSubsetP("gas", densField, inds=cell_inds)  # ions/cm^3

        # create spectra
        inst_wave, tau_local, EW_local, N_local, v90_local = create_spectra_from_traced_rays(
            sP, line, instrument, rays_off, rays_len, rays_dl, rays_inds, cell_dens, cell_temp, cell_vellos
        )

        assert np.array_equal(inst_wave, wave_mid)

        EWs[line] = EW_local
        N[line] = N_local
        v90[line] = v90_local

        chunks = (1000, tau_local.shape[1]) if tau_local.shape[1] < 10000 else (100, tau_local.shape[1])

        print(" saving...", flush=True)
        with h5py.File(saveFilename, "r+") as f:
            # save tau per line
            f.create_dataset("tau_%s" % line.replace(" ", "_"), data=tau_local, chunks=chunks, compression="gzip")
            # save EWs and coldens per line
            f.create_dataset("EW_%s" % line.replace(" ", "_"), data=EW_local)
            f.create_dataset("N_%s" % line.replace(" ", "_"), data=N_local)
            f.create_dataset("v90_%s" % line.replace(" ", "_"), data=v90_local)

        tau_local = None

    # sum optical depths across all lines, use to calculate flux array (i.e. the spectrum)
    print("Loading tau by line and saving total flux...", flush=True)

    with h5py.File(saveFilename, "r+") as f:
        # create flux dataset
        shape = f["tau_%s" % lineNames[0].replace(" ", "_")].shape
        chunks = (1000, shape[1]) if shape[1] < 10000 else (100, shape[1])

        if "flux" not in f:
            # create if not already present (may have been started but not finished)
            dset = f.create_dataset("flux", shape=shape, chunks=chunks, compression="gzip")
        else:
            dset = f["flux"]

        # process in 10 chunks
        n_chunks = 10

        offset = 0
        n_per_chunk = int(shape[0] / n_chunks)
        assert n_per_chunk == shape[0] / n_chunks

        tau = np.zeros((n_per_chunk, shape[1]), dtype="float32")

        for i in range(n_chunks):
            print(f" chunk [{i} of {n_chunks}]", flush=True)
            tau *= 0

            # loop over all lines, accumulate optical depth arrays
            for line in lineNames:
                # print(f' {reportMemory() = :.1f} GB now adding tau [{line}].', flush=True)
                tau_local = f["tau_%s" % line.replace(" ", "_")][offset : offset + n_per_chunk]
                tau += tau_local

            # compute and write flux
            flux = np.exp(-1 * tau)

            dset[offset : offset + n_per_chunk] = flux
            offset += n_per_chunk

        f.attrs["flux_done"] = True

    print(f"Saved: [{saveFilename}]")


def integrate_along_saved_rays(
    sP, field, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def, subhaloIDs=None, pSplit=None
):
    """Integrate a physical (gas) property along the line of sight, based on already computed and saved rays.

    The result has units of [pc] * [field] where [field] is the original units of the physical field as loaded,
    unless field is a number density, in which case the result (column density) is in [cm^-2].

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      field (str): any available gas field.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
      subhaloIDs (list): if raysType is 'sample_localized' (only), then a list of subhalo IDs.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note
        that we follow a spatial subdivision, so the total job number should be an integer squared.
    """
    # save file
    saveFilename = spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=pSplit)

    if isfile(saveFilename):
        print("Loading: [%s]" % saveFilename)
        with h5py.File(saveFilename, "r") as f:
            result = f["result"][()]
        return result

    # calculating, but no pSplit? rays are only kept split, so loop over now
    if pSplit is None:
        print(f"Calculating [{saveFilename}] now...")
        for i in range(16):
            _ = integrate_along_saved_rays(sP, field, nRaysPerDim, raysType, subhaloIDs, pSplit=[i, 16])
        concat_integrals(sP, field, nRaysPerDim, raysType)
        return integrate_along_saved_rays(sP, field, nRaysPerDim, raysType, subhaloIDs)

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = generate_rays_voronoi_fullbox(
        sP, nRaysPerDim=nRaysPerDim, raysType=raysType, subhaloIDs=subhaloIDs, pSplit=pSplit
    )

    projAxis = list(ray_dir).index(1)

    # load required gas cell properties
    if field.endswith("_los"):
        field = field.replace("_los", "") + "_" + ["x", "y", "z"][projAxis]

    cell_values = sP.snapshotSubsetP("gas", field, inds=cell_inds)  # units unchanged

    # convert length units
    if "numdens" in field:
        # result units: [cm^-2]
        rays_dl = sP.units.codeLengthToCm(rays_dl)
    else:
        # result units: [parsecs] * [field units]
        rays_dl = sP.units.codeLengthToPc(rays_dl)

    # start output
    with h5py.File(saveFilename, "w") as f:
        # attach ray configuration for reference
        f["ray_pos"] = ray_pos
        f["ray_dir"] = ray_dir
        f["ray_total_dl"] = total_dl

    # integrate
    result = _integrate_quantity_along_traced_rays(rays_off, rays_len, rays_dl, rays_inds, cell_values)

    with h5py.File(saveFilename, "r+") as f:
        f.create_dataset("result", data=result, compression="gzip")

    print(f"Saved: [{saveFilename}]")

    return result


# ---


@jit(nopython=True, nogil=True)
def deposit_single_line(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, debug=False):
    """Add the absorption profile of a single transition, from a single cell, to a spectrum.

    Global method, where the original master grid is assumed to be very high resolution, such that
    no sub-sampling is necessary (re-sampling onto an instrument grid done later).

    Args:
      wave_edges_master (array[float]): bin edges for master spectrum array [ang].
      tau_master (array[float]): master optical depth array.
      N (float): column density in [1/cm^2].
      b (float): doppler parameter in [km/s].
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0 (float): central wavelength, rest-frame [ang].
      z_eff (float): effective redshift, i.e. including both cosmological and peculiar components.
      debug (bool): if True, return local grid info and do checks.

    Return:
      None.
    """
    if N == 0:
        return  # empty

    # if the optical depth is larger than this by the edge of the local grid, redo
    edge_tol = 1e-4

    # check that grid resolution is sufficient
    dwave_master = wave_edges_master[1] - wave_edges_master[0]
    b_dwave = b / sP_units_c_km_s * wave0  # v/c = dwave/wave

    if b_dwave < dwave_master * 5:
        print("WARNING: b_dwave is too small for the dwave_master, ", b_dwave, dwave_master)
        # assert 0 # check

    # prep local grid where we will sample tau
    wave0_obsframe = wave0 * (1 + z_eff)

    line_width_safety = b / sP_units_c_km_s * wave0_obsframe

    n_iter = 0
    local_fac = 5.0
    tau = np.array([np.inf], dtype=np.float64)
    master_previnds = np.array([-1, -1], dtype=np.int32)

    while tau[0] > edge_tol or tau[-1] > edge_tol:
        # determine where local grid overlaps with master
        wave_min_local = wave0_obsframe - local_fac * line_width_safety
        wave_max_local = wave0_obsframe + local_fac * line_width_safety

        master_inds = np.searchsorted(wave_edges_master, [wave_min_local, wave_max_local])
        master_startind = master_inds[0] - 1
        master_finalind = master_inds[1]

        if master_startind == master_previnds[0] and master_finalind == master_previnds[1]:
            # increase of local_fac was too small to actually increase coverage of master grid, repeat
            local_fac *= 2.0
            n_iter += 1
            continue

        master_previnds[0] = master_startind
        master_previnds[1] = master_finalind

        # sanity checks
        if master_startind == -1:
            if debug:
                print("WARNING: min edge of local grid hit edge of master!")
            master_startind = 0

        if master_finalind == wave_edges_master.size:
            if debug:
                print("WARNING: max edge of local grid hit edge of master!")
            master_finalind = wave_edges_master.size - 1

        if master_startind == master_finalind:
            if n_iter < 20:
                # extend, see if wings of this feature will enter master spectrum
                local_fac *= 1.2
                n_iter += 1
                continue

            if debug:
                print("WARNING: absorber entirely off edge of master spectrum! skipping!")
            return

        # local grid
        wave_edges_local = wave_edges_master[master_startind:master_finalind]
        wave_mid_local = (wave_edges_local[1:] + wave_edges_local[:-1]) / 2

        # get optical depth
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        # if debug: print(f'  [iter {n_iter}] master inds [{master_startind} - {master_finalind}], {local_fac = },
        #                 f'{tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

        if n_iter > 100:
            break

        if master_startind == 0 and master_finalind == wave_edges_master.size - 1:
            break  # local grid already extended to entire master

        local_fac *= 2.0
        n_iter += 1

    # deposit local tau into each bin of master tau
    tau_master[master_startind : master_finalind - 1] += tau

    return


@jit(nopython=True, nogil=True)
def _create_spectra_from_traced_rays(
    f,
    gamma,
    wave0,
    ion_mass,
    rays_off,
    rays_len,
    rays_cell_dl,
    rays_cell_inds,
    cell_dens,
    cell_temp,
    cell_vellos,
    z_vals,
    z_lengths,
    master_mid,
    master_edges,
    inst_wavemid,
    inst_waveedges,
    lsf_mode,
    lsf_matrix,
    ind0,
    ind1,
):
    """JITed helper (see below)."""
    n_rays = ind1 - ind0 + 1
    scalefac = 1 / (1 + z_vals[0])

    # allocate: full spectra return as well as derived summary statistics
    tau_master = np.zeros(master_mid.size, dtype=np.float64)
    tau_allrays = np.zeros((n_rays, inst_wavemid.size), dtype=np.float32)
    EW_allrays = np.zeros(n_rays, dtype=np.float32)
    N_allrays = np.zeros(n_rays, dtype=np.float32)
    v90_allrays = np.zeros(n_rays, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # get local properties
        offset = rays_off[ind0 + i]  # start of intersected cells (in rays_cell*)
        length = rays_len[ind0 + i]  # number of intersected gas cells

        master_dx = rays_cell_dl[offset : offset + length]
        master_inds = rays_cell_inds[offset : offset + length]

        master_dens = cell_dens[master_inds]
        master_temp = cell_temp[master_inds]
        master_vellos = cell_vellos[master_inds]

        # column density
        N = master_dens * (master_dx * sP_units_Mpc_in_cm)  # cm^-2
        N_allrays[i] = np.sum(N)

        # skip rays with negligibly small total columns (in linear cm^-2)
        if N_allrays[i] < 1e8:
            continue

        # reset tau_master for each ray
        tau_master *= 0.0

        # cumulative pathlength, Mpc from start of box i.e. start of ray (at sP.redshift)
        cum_pathlength = np.zeros(length, dtype=np.float32)
        cum_pathlength[1:] = np.cumsum(master_dx)[:-1]  # pMpc
        cum_pathlength /= scalefac  # input in pMpc, convert to cMpc

        # cosmological redshift of each intersected cell
        z_cosmo = np.interp(cum_pathlength, z_lengths, z_vals)

        # doppler shift
        z_doppler = master_vellos / sP_units_c_km_s

        # effective redshift
        z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1

        # doppler parameter b = sqrt(2kT/m) where m is the particle mass
        b = np.sqrt(2 * sP_units_boltzmann * master_temp / ion_mass)  # cm/s
        b /= 1e5  # km/s

        # deposit each intersected cell as an absorption profile onto spectrum
        count = 0
        for j in range(length):
            # skip negligibly small columns (in linear cm^-2) for efficiency
            if N[j] < 1e6:
                continue

            deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[j], b[j], z_eff[j])
            count += 1

        # resample tau_master on to instrument wavelength grid
        if count == 0:
            continue  # no absorption, skip this ray

        tau_inst = resample_spectrum(master_mid, tau_master, inst_waveedges)

        # line spread function (LSF) in pixel space? convolve the instrumental (flux) spectrum now
        # note: in theory we would prefer to convolve the master spectrum prior to resampling, but
        # given the ~1e8 resolution of the master spectrum, the cost is prohibitive
        if lsf_mode == 1:
            flux_inst = 1 - np.exp(-tau_inst)
            flux_conv = varconvolve(flux_inst, lsf_matrix).astype(np.float64)

            # note: flux_conv can be 1.0, leading to tau_inst = inf
            # so set to 1-eps, such that tau is very large (~30 for this value of eps)
            flux_smallval = 1.0 - 1e-16
            flux_conv[flux_conv >= 1.0] = flux_smallval

            tau_inst = -np.log(1 - flux_conv).astype(np.float32)

        # also compute and save a reference EW and v90
        # note: are global values, i.e. not localized/restricted to a single absorber
        EW_allrays[i] = _equiv_width(tau_inst, inst_wavemid)

        v90_allrays[i] = _v90(tau_inst, inst_wavemid)

        # stamp
        tau_allrays[i, :] = tau_inst

        # debug: (verify EW is same in master and instrumental grids)
        if 0:
            EW_check = _equiv_width(tau_master, master_mid)
            # assert np.abs(EW_check - EW_allrays[i]) < 0.01
            if np.abs(EW_check - EW_allrays[i]) > 0.01:
                # where? ignore if it is in master grid outside of inst grid coverage
                ww = np.where(tau_master > 0)[0]
                wavemin = master_mid[ww.min()]
                wavemax = master_mid[ww.max()]
                if wavemin > inst_waveedges[0] and wavemax < inst_waveedges[-1]:
                    print(
                        "WARNING, EW delta = ",
                        EW_check - EW_allrays[i],
                        " from wavemin = ",
                        wavemin,
                        " to wavemax = ",
                        wavemax,
                        " EW_inst = ",
                        EW_check,
                        " EW_master = ",
                        EW_allrays[i],
                    )

    return tau_allrays, EW_allrays, N_allrays, v90_allrays


def create_spectra_from_traced_rays(
    sP,
    line,
    instrument,
    rays_off,
    rays_len,
    rays_cell_dl,
    rays_cell_inds,
    cell_dens,
    cell_temp,
    cell_vellos,
    nThreads=60,
):
    """Generate a fullset of mock absorption spectra given the pre-existing set of rays.

    The rays are a composite list of intersected cell pathlengths and indices. Using these, we extract the physical
    properties needed (dens, temp, vellos) and create the final absorption spectrum, depositing a Voigt absorption
    profile for each cell.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): string specifying the instrumental setup.
      rays_off (array[int]): first entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_len (array[int]): second entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_cell_dl (array[float]): third entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      rays_cell_inds (array[int]): fourth entry from tuple return of :py:func:`util.voronoiRay.rayTrace`.
      cell_dens (array[float]): gas per-cell densities of a given species [linear ions/cm^3]
      cell_temp (array[float]): gas per-cell temperatures [linear K]
      cell_vellos (array[float]): gas per-cell line of sight velocities [linear km/s]
      z_lengths (array[float]): the comoving distance to each z_vals relative to sP.redshift [pMpc]
      z_vals (array[float]): a sampling of redshifts, starting at sP.redshift
      nThreads (int): parallelize calculation using this threads (serial computation if one)
    """
    n_rays = rays_len.size

    # line properties
    f, gamma, wave0, ion_amu, ion_mass = line_params(line)

    # assign sP.redshift to the front intersection (beginning) of the box
    z_vals = np.linspace(sP.redshift, sP.redshift + 0.2, 400)
    assert sP.boxSize <= 100000, "Increase 0.2 factor above for boxes larger than TNG100."

    z_lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(sP.redshift)

    # sample master, and instrumental, grids
    master_mid, master_edges, _ = create_wavelength_grid(instrument="master")

    assert master_mid[1] > master_mid[0], "Error: dwave_master will be zero!"

    inst_wavemid, inst_waveedges, _ = create_wavelength_grid(instrument=instrument)

    assert inst_waveedges[0] >= master_edges[0], "Instrumental wavelength grid min extends off master."
    assert inst_waveedges[-1] <= master_edges[-1], "Instrumental wavelength grid max extends off master."

    lsf_mode, lsf, _ = lsf_matrix(instrument)

    if 0:
        indiv_index = 10910
        rays_len = rays_len[indiv_index : indiv_index + 10]
        rays_off = rays_off[indiv_index : indiv_index + 10]
        n_rays = rays_len.size
        print("TODO REMOVE SINGLE RAY DEBUG!!!")

    # single-threaded
    if nThreads == 1 or n_rays < nThreads:
        ind0 = 0
        ind1 = n_rays - 1

        tau, EW, N, v90 = _create_spectra_from_traced_rays(
            f,
            gamma,
            wave0,
            ion_mass,
            rays_off,
            rays_len,
            rays_cell_dl,
            rays_cell_inds,
            cell_dens,
            cell_temp,
            cell_vellos,
            z_vals,
            z_lengths,
            master_mid,
            master_edges,
            inst_wavemid,
            inst_waveedges,
            lsf_mode,
            lsf,
            ind0,
            ind1,
        )

        return inst_wavemid, tau, EW, N, v90

    # multi-threaded
    class specThread(threading.Thread):
        """Subclass Thread() to provide local storage."""

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # determine local slice
            self.ind0, self.ind1 = pSplitRange([0, n_rays - 1], nThreads, threadNum, inclusive=True)

        def run(self):
            # call JIT compiled kernel
            self.result = _create_spectra_from_traced_rays(
                f,
                gamma,
                wave0,
                ion_mass,
                rays_off,
                rays_len,
                rays_cell_dl,
                rays_cell_inds,
                cell_dens,
                cell_temp,
                cell_vellos,
                z_vals,
                z_lengths,
                master_mid,
                master_edges,
                inst_wavemid,
                inst_waveedges,
                lsf_mode,
                lsf,
                self.ind0,
                self.ind1,
            )

    # create threads
    threads = [specThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # all threads are done, determine return size and allocate
    tau_allrays = np.zeros((n_rays, inst_wavemid.size), dtype="float32")
    EW_allrays = np.zeros(n_rays, dtype="float32")
    N_allrays = np.zeros(n_rays, dtype="float32")
    v90_allrays = np.zeros(n_rays, dtype="float32")

    # add the result array from each thread to the global
    for thread in threads:
        tau_loc, EW_loc, N_loc, v90_loc = thread.result

        tau_allrays[thread.ind0 : thread.ind1 + 1, :] = tau_loc
        EW_allrays[thread.ind0 : thread.ind1 + 1] = EW_loc
        N_allrays[thread.ind0 : thread.ind1 + 1] = N_loc
        v90_allrays[thread.ind0 : thread.ind1 + 1] = v90_loc

    return inst_wavemid, tau_allrays, EW_allrays, N_allrays, v90_allrays


@jit(nopython=True, nogil=True)
def _integrate_quantity_along_traced_rays(rays_off, rays_len, rays_cell_dl, rays_cell_inds, cell_values):
    """Integrate a given physical quantity along each sightline.

    Return:
      float: the integral of the quantity along each sightline, with units given by [rays_cell_dl * cell_values].
    """
    n_rays = rays_len.size

    r = np.zeros(n_rays, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # get local properties
        offset = rays_off[i]  # start of intersected cells (in rays_cell*)
        length = rays_len[i]  # number of intersected gas cells

        master_dx = rays_cell_dl[offset : offset + length]
        master_inds = rays_cell_inds[offset : offset + length]

        master_values = cell_values[master_inds]

        r[i] = np.sum(master_dx * master_values)

    return r


def concat_integrals(sP, field, nRaysPerDim=nRaysPerDim_def, raysType=raysType_def):
    """Combine split files for line-of-sight quantity integrals into a single file.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      field (str): any available gas field.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
    """
    # search for chunks
    loadFilename = spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit="*")
    saveFilename = spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None)

    pSplitNum = len([f for f in glob.glob(loadFilename) if "_combined" not in f])
    assert pSplitNum > 0, "Error: No split spectra files found."

    # load all for count
    count = 0

    for i in range(pSplitNum):
        filename = spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i, pSplitNum])

        with h5py.File(filename, "r") as f:
            # first file: load number of spectra per chunk file, master wavelength grid, and other metadata
            if i == 0:
                n = f["result"].size

                ray_dir = f["ray_dir"][()] if "ray_dir" in f else f.attrs["ray_dir"]
                ray_total_dl = f["ray_total_dl"][()] if "ray_total_dl" in f else f.attrs["total_dl"]

                # allocate
                ray_pos = np.zeros((pSplitNum * n, 3), dtype="float32")
                result = np.zeros(pSplitNum * n, dtype=f["result"].dtype)

            else:
                # all other chunks: sanity checks
                assert n == f["result"].size  # should be constant

            # load ray starting positions
            ray_pos[count : count + n] = f["ray_pos"][()]
            result[count : count + n] = f["result"][()]

            print(f"[{count:7d} - {count + n:7d}] {filename}")
            count += n

    print(f"In total [{count}] line-of-sight integrals loaded.")
    assert count == n * pSplitNum, "Error: Unexpected total number of spectra."

    # save
    with h5py.File(saveFilename, "w") as f:
        # ray metadata and reuslt
        f["ray_pos"] = ray_pos
        f["ray_dir"] = ray_dir
        f["ray_total_dl"] = ray_total_dl

        f["result"] = result

        # metadata
        f.attrs["simName"] = sP.simName
        f.attrs["redshift"] = sP.redshift
        f.attrs["snapshot"] = sP.snap
        f.attrs["field"] = field
        f.attrs["count"] = count

    print("Saved: [%s]" % saveFilename)

    # remove split files
    if raysType == "sample_localized":
        return  # likely want to keep them (per-halo)

    for i in range(pSplitNum):
        filename = spectra_filepath(sP, ion=field, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=[i, pSplitNum])
        unlink(filename)

    print("Split files removed.")


def concat_spectra(
    sP, ion="Fe II", instrument="4MOST-HRS", nRaysPerDim=nRaysPerDim_def, raysType=raysType_def, solar=False
):
    """Combine split files for spectra into a single file.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead
        use the (constant) solar value.
    """
    # search for chunks
    lineNames = [k for k, v in lines.items() if lines[k]["ion"] == ion]  # all transitions of this ion

    loadFilename = spectra_filepath(
        sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit="*", solar=solar
    )
    saveFilename = spectra_filepath(
        sP, ion, instrument=instrument, nRaysPerDim=nRaysPerDim, raysType=raysType, pSplit=None, solar=solar
    )

    pSplitNum = len([f for f in glob.glob(loadFilename) if "_combined" not in f])
    assert pSplitNum > 0, "Error: No split spectra files found."

    # load all for count
    lines_present = []
    count = 0

    for i in range(pSplitNum):
        filename = spectra_filepath(
            sP,
            ion,
            instrument=instrument,
            nRaysPerDim=nRaysPerDim,
            raysType=raysType,
            pSplit=[i, pSplitNum],
            solar=solar,
        )

        with h5py.File(filename, "r") as f:
            # first file: load number of spectra per chunk file, master wavelength grid, and other metadata
            assert "flux" in f, "Error: No flux array found in [%s], likely OOM and did not finish." % filename

            if i == 0:
                n_wave = f["flux"].shape[1]
                n_spec = f["flux"].shape[0]

                ray_dir = f["ray_dir"][()]
                ray_total_dl = f["ray_total_dl"][()]
                wave = f["wave"][()]

                # allocate
                ray_pos = np.zeros((pSplitNum * n_spec, 3), dtype="float32")

                # which lines of this ion are present?
                for line in lineNames:
                    # this line is present?
                    key = "EW_%s" % line.replace(" ", "_")
                    if key in f:
                        lines_present.append(line)
                    else:
                        print("Skipping [%s], not present." % line)

            else:
                # all other chunks: sanity checks
                assert n_spec == f["flux"].shape[0]  # should be constant
                assert np.array_equal(wave, f["wave"][()])  # should be the same

            # load ray starting positions
            ray_pos[count : count + n_spec] = f["ray_pos"][()]

            print(f"[{count:7d} - {count + n_spec:7d}] {filename}")
            count += n_spec

    print(f"In total [{count}] spectra with: [{', '.join(lines_present)}]")

    lines_present = [line.replace(" ", "_") for line in lines_present]

    assert count > 0, "Error: All EWs are zero. Observed frame wavelengths outside instrument coverage?"
    assert count == n_spec * pSplitNum, "Error: Unexpected total number of spectra."

    # start save
    savedDatasets = []

    if not isfile(saveFilename):
        with h5py.File(saveFilename, "w") as f:
            # wavelength grid, flux array, ray positions
            f["wave"] = wave
            f["ray_pos"] = ray_pos
            f["ray_dir"] = ray_dir
            f["ray_total_dl"] = ray_total_dl

            # metadata
            f.attrs["simName"] = sP.simName
            f.attrs["redshift"] = sP.redshift
            f.attrs["snapshot"] = sP.snap
            f.attrs["instrument"] = instrument
            f.attrs["lineNames"] = lines_present
            f.attrs["count"] = count
    else:
        with h5py.File(saveFilename, "r") as f:
            savedDatasets = list(f.keys())

    # load large datasets, one at a time, and save
    dsets = ["flux"]
    dsets += ["EW_%s" % line for line in lines_present]
    dsets += ["N_%s" % line for line in lines_present]
    dsets += ["v90_%s" % line for line in lines_present]
    dsets += ["tau_%s" % line for line in lines_present]

    for dset in dsets:
        # already done?
        if dset in savedDatasets:
            print(f"Skipping [{dset}], already saved.")
            continue

        # set reasonable chunk shape (otherwise, automatic) (mandatory with compression)
        print(f"Re-writing [{dset}] -- [", end="")
        offset = 0

        if "EW_" in dset or "N_" in dset or "v90_" in dset:
            shape = count
            chunks = count
        else:
            shape = (count, n_wave)
            chunks = (1000, n_wave) if n_wave < 10000 else (100, n_wave)

        with h5py.File(saveFilename, "r+") as fOut:
            # initialize empty dataset
            data = fOut.create_dataset(dset, shape=shape, chunks=chunks, compression="gzip")

            # load and write by split chunk
            for i in range(pSplitNum):
                filename = spectra_filepath(
                    sP,
                    ion,
                    instrument=instrument,
                    nRaysPerDim=nRaysPerDim,
                    raysType=raysType,
                    pSplit=[i, pSplitNum],
                    solar=solar,
                )
                print(i, end=" ", flush=True)

                with h5py.File(filename, "r", rdcc_nbytes=0) as f_read:
                    data[offset : offset + n_spec] = f_read[dset][()]
                offset += n_spec

        print("] done.")

    print("Saved: [%s]" % saveFilename)

    # remove split files
    if raysType == "sample_localized":
        return  # likely want to keep them (per-halo)

    for i in range(pSplitNum):
        filename = spectra_filepath(
            sP,
            ion,
            instrument=instrument,
            nRaysPerDim=nRaysPerDim,
            raysType=raysType,
            pSplit=[i, pSplitNum],
            solar=solar,
        )
        unlink(filename)

    print("Split files removed.")


def spectra_filepath(
    sim,
    ion,
    projAxis=projAxis_def,
    nRaysPerDim=nRaysPerDim_def,
    raysType=raysType_def,
    instrument=None,
    pSplit=None,
    solar=False,
):
    """Return the path to a file of saved spectra.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      nRaysPerDim (int): number of rays per linear dimension (total is this value squared).
      raysType (str): either 'voronoi_fullbox' (equally spaced), 'voronoi_rndfullbox' (random), or
        'sample_localized' (distributed around a given set of subhalos).
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note
        that we follow a spatial subdivision, so the total job number should be an integer squared.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead
        use the (constant) solar value.
    """
    ionStr = ion.replace(" ", "")
    path = sim.derivPath + "spectra/"
    confStr = "n%dd%d-%s" % (nRaysPerDim, projAxis, raysType.replace("voronoi_", ""))  # e.g. 'n1000d2-fullbox'

    if instrument is not None:
        filebase = "spectra_%s_z%.1f_%s_%s_%s" % (sim.simName, sim.redshift, confStr, instrument, ionStr)
    else:
        filebase = "integral_%s_z%.1f_%s_%s" % (sim.simName, sim.redshift, confStr, ionStr)

    if isinstance(pSplit, list):
        # a specific chunk
        filename = filebase + "_%d-of-%d.hdf5" % (pSplit[0], pSplit[1])

    elif str(pSplit) == "*":
        # leave wildcard for glob search (would have to generalized if pSplit[1] is not two digits)
        filename = filebase + "_*of-*.hdf5"

    else:
        # concatenated set
        filename = filebase + "_combined.hdf5"

        if not isfile(path + filename) and isfile(sim.postPath + "AbsorptionSpectra/" + filename):
            path = sim.postPath + "AbsorptionSpectra/"  # permanent path in /postprocessing/

    if solar:
        filename = filename.replace(".hdf5", "_solar.hdf5")

    return path + filename
