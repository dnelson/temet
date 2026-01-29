"""
Synthetic absorption spectra generation (testing/non-production).
"""

from os.path import isfile

import h5py
import numpy as np

from ..spectra.spectrum import create_spectra_from_traced_rays
from ..spectra.util import _equiv_width, create_wavelength_grid, line_params, lines, sP_units_c_km_s
from ..util.simParams import simParams
from ..util.sphMap import sphGridWholeBox, sphMap
from ..util.treeSearch import buildFullTree
from ..util.voronoi import loadGlobalVPPP, loadSingleHaloVPPP
from ..util.voronoiRay import (
    rayTrace,
    trace_ray_through_voronoi_mesh_treebased,
    trace_ray_through_voronoi_mesh_with_connectivity,
)


def create_spectrum_from_traced_ray(sP, line, instrument, cell_dens, cell_dx, cell_temp, cell_vellos):
    """Given a completed (single) ray traced through a volume, compute the final absorption spectrum.

    Use the properties of all the intersected cells (dens, dx, temp, vellos) to deposit Voigt absorption
    profiles for each cell.
    """
    # prepare pass through
    rays_off = np.array([0], dtype="int32")
    rays_len = np.array([cell_dens.size], dtype="int32")
    rays_cell_inds = np.arange(cell_dens.size)

    return create_spectra_from_traced_rays(
        sP, line, instrument, rays_off, rays_len, cell_dx, rays_cell_inds, cell_dens, cell_temp, cell_vellos, nThreads=1
    )


# @jit(nopython=True, nogil=True, cache=False)
def deposit_single_line_local(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, debug=False):
    """Add the absorption profile of a single transition, from a single cell, to a spectrum.

    Local method, where master grid is sub-sampled and optical depth is deposited back onto master.

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
    from ..spectra.spectrum import _voigt_tau

    if N == 0:
        return  # empty

    # local (to the line), rest-frame wavelength grid
    dwave_local = 0.001  # ang
    edge_tol = 1e-4  # if the optical depth is larger than this by the edge of the local grid, redo

    b_dwave = b / sP_units_c_km_s * wave0  # v/c = dwave/wave

    # adjust local resolution to make sure we sample narrow lines
    while b_dwave < dwave_local * 10:
        dwave_local *= 0.5

        if dwave_local < 1e-6:
            print(b, b_dwave, dwave_local)
            assert 0  # check
            break

    # prep local grid
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
            # if debug: print('WARNING: min edge of local grid hit edge of master!')
            master_startind = 0

        if master_finalind == wave_edges_master.size:
            # if debug: print('WARNING: max edge of local grid hit edge of master!')
            master_finalind = wave_edges_master.size - 1

        if master_startind == master_finalind:
            if n_iter < 20:
                # extend, see if wings of this feature will enter master spectrum
                local_fac *= 1.2
                n_iter += 1
                continue

            # if debug: print('WARNING: absorber entirely off edge of master spectrum! skipping!')
            return

        # how does local grid size compare to master? approximate dwave_master as constant, while in reality it
        # may be variable (but locally ~constant). note dwave_local is in fact, always, constant.
        dwave_master = wave_edges_master[master_finalind] - wave_edges_master[master_finalind - 1]
        nloc_per_master = int(np.round(dwave_master / dwave_local))

        if nloc_per_master <= 0:
            # if debug: print('WARNING: local grid size actually smaller than master!')
            dwave_local *= 0.5
            n_iter += 1
            continue

        # create local grid specification aligned with master
        nmaster_covered = master_finalind - master_startind  # difference of bin edge indices
        num_bins_local = nmaster_covered * nloc_per_master

        wave_min_local = wave_edges_master[master_startind]
        wave_max_local = wave_edges_master[master_finalind]

        # create local grid
        wave_edges_local = np.linspace(wave_min_local, wave_max_local, num_bins_local + 1)
        wave_mid_local = (wave_edges_local[1:] + wave_edges_local[:-1]) / 2

        # get optical depth
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0)

        flux = np.exp(-tau)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        # if debug: print(f'  [iter {n_iter}] master inds [{master_startind} - {master_finalind}], {local_fac = },
        #  {tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

        if n_iter > 100:
            break

        if master_startind == 0 and master_finalind == wave_edges_master.size - 1:
            break  # local grid already extended to entire master

        local_fac *= 2.0
        n_iter += 1

    # if (tau[0] > edge_tol or tau[-1] > edge_tol):
    #    print('WARNING: final local grid edges still have high tau')
    #    #if not debug: assert 0

    # integrate local tau within each bin of master tau
    master_ind = master_startind
    # tau_bin = 0.0
    iflux_bin = 0.0

    for local_ind in range(wave_mid_local.size):
        # deposit partial integral of tau into master bin
        # tau_bin += tau[local_ind]
        iflux_bin += 1 - flux[local_ind]

        # print(f' add to tau_master[{master_ind:2d}] from {local_ind = :2d} with {tau[local_ind] = :.3g} i.e.
        # {wave_mid_local[local_ind]:.4f} [{wave_edges_local[local_ind]:.4f}-{wave_edges_local[local_ind+1]:.4f}]
        # Ang into [{wave_edges_master[master_ind]}-{wave_edges_master[master_ind+1]}] Ang')

        # has local wavelength bin moved into the next master bin? or are we finished with the local grid?
        if wave_mid_local[local_ind] > wave_edges_master[master_ind] or local_ind == wave_mid_local.size - 1:
            # midpoint rule, deposit accumulated tau into this master bin
            # (this is what trident does, converging tau, but not flux, which is not the behavior we want)
            # tau_master[master_ind] += tau_bin * (dwave_local/dwave_master)

            # mean flux in local grid?
            # local_meanflux = flux_bin / count
            # meanflux_to_tau = -np.log(local_meanflux)

            local_EW = iflux_bin * dwave_local
            master_height = local_EW / dwave_master  # h = area / width gives the 'height' of 1-F
            assert master_height >= 0.0
            assert master_height < 1.000001  # impossible to have >1, in which case np.log(negative) is nan

            if master_height > 1.0 - 1e-10:
                # entire master bin is saturated to zero flux, but rounding errors could place the height > 1.0
                # set to 1-eps, such that tau is very large (~20 for this value of eps), and final flux ~ 1e-10
                master_height = 1.0 - 1e-10

            localEW_to_tau = -np.log(1 - master_height)
            # assert master_height < 1.0 # otherwise np.log(negative) is nan
            # localEW_to_tau = -np.log(1-local_EW)

            # TESTING:
            tau_master[master_ind] += localEW_to_tau
            # tau_master[master_ind] += (localEW_to_tau / dwave_master)

            # how much total EW? (debugging only)
            # if N > 1e14 and debug:
            #    print(f'   updated tau_master[{master_ind:2d}] = {tau_master[master_ind]:.4f}, {local_EW = :.4f},
            # {localEW_to_tau = }')

            # move to next master bin
            master_ind += 1
            # tau_bin = 0.0
            iflux_bin = 0.0

    if N > 1e14 and debug:
        # debug check: if this is not the first deposition, then the local values should be lower than the master values
        wave_mid_master = (wave_edges_master[1:] + wave_edges_master[:-1]) / 2
        EW_local = _equiv_width(tau, wave_mid_local)
        EW_master = _equiv_width(tau_master, wave_mid_master)

        tau_local_tot = np.sum(tau * dwave_local)
        tau_master_tot = np.sum(tau_master * dwave_master)

        print(f"  {EW_local = :.6f}, {EW_master = :.6f}, {tau_local_tot = :.5f}, {tau_master_tot = :.5f}")
        print(f"  {dwave_local = }, {wave_mid_local.size = }")
        print(f"  {tau_master[116] = }")

    return


def generate_spectrum_uniform_grid():
    """Generate an absorption spectrum by ray-tracing through a uniform grid (deposit using sphMap)."""
    # config
    sP = simParams(run="tng50-4", redshift=0.5)

    line = "OVI 1032"  #'LyA'
    instrument = "idealized"  # 'SDSS-BOSS'
    nCells = 64
    haloID = 150  # if None, then full box

    posInds = [int(nCells * 0.5), int(nCells * 0.5)]  # [0,0] # (x,y) pixel indices to ray-trace along
    projAxis = 2  # z, to simplify vellos

    # quick caching
    cacheFile = f"cache_{line}_{nCells}_h{haloID}_{sP.snap}.hdf5"
    if isfile(cacheFile):
        # load now
        print(f"Loading [{cacheFile}].")
        with h5py.File(cacheFile, "r") as f:
            grid_dens = f["grid_dens"][()]
            grid_vel = f["grid_vel"][()]
            grid_temp = f["grid_temp"][()]
            if haloID is not None:
                boxSizeImg = f["boxSizeImg"][()]
    else:
        # load
        massField = "%s mass" % lines[line]["ion"]
        velField = "vel_" + ["x", "y", "z"][projAxis]

        pos = sP.snapshotSubsetP("gas", "pos", haloID=haloID)  # code
        vel_los = sP.snapshotSubsetP("gas", velField, haloID=haloID)  # code
        mass = sP.snapshotSubsetP("gas", massField, haloID=haloID)  # code
        hsml = sP.snapshotSubsetP("gas", "hsml", haloID=haloID)  # code
        temp = sP.snapshotSubsetP("gas", "temp_sfcold", haloID=haloID)  # K

        # grid
        if haloID is None:
            grid_mass = sphGridWholeBox(sP, pos, hsml, mass, None, nCells=nCells)
            grid_vel = sphGridWholeBox(sP, pos, hsml, mass, vel_los, nCells=nCells)
            grid_temp = sphGridWholeBox(sP, pos, hsml, mass, temp, nCells=nCells)

            pxVol = (sP.boxSize / nCells) ** 3  # code units (ckpc/h)^3
        else:
            halo = sP.halo(haloID)
            haloSizeRvir = 2.0
            boxSizeImg = halo["Group_R_Crit200"] * np.array([haloSizeRvir, haloSizeRvir, haloSizeRvir])
            boxCen = halo["GroupPos"]

            opts = {
                "axes": [0, 1],
                "ndims": 3,
                "boxSizeSim": sP.boxSize,
                "boxSizeImg": boxSizeImg,
                "boxCen": boxCen,
                "nPixels": [nCells, nCells, nCells],
            }

            grid_mass = sphMap(pos=pos, hsml=hsml, mass=mass, quant=None, **opts)
            grid_vel = sphMap(pos=pos, hsml=hsml, mass=mass, quant=vel_los, **opts)
            grid_temp = sphMap(pos=pos, hsml=hsml, mass=mass, quant=temp, **opts)

            pxVol = np.prod(boxSizeImg) / nCells**3  # code units

        # unit conversions: mass -> density
        f, gamma, wave0, ion_amu, ion_mass = line_params(line)

        grid_dens = sP.units.codeDensToPhys(grid_mass / pxVol, cgs=True, numDens=True)  # H atoms/cm^3
        grid_dens /= ion_amu  # [ions/cm^3]

        # unit conversions: line-of-sight velocity
        grid_vel = sP.units.particleCodeVelocityToKms(grid_vel)  # physical km/s

        # save
        with h5py.File(cacheFile, "w") as f:
            f["grid_dens"] = grid_dens
            f["grid_vel"] = grid_vel
            f["grid_temp"] = grid_temp
            if haloID is not None:
                f["boxSizeImg"] = boxSizeImg
        print(f"Saved [{cacheFile}].")

    # print ray starting location in global space (note: possible the grid is permuted/transposed still)
    print(f"{boxSizeImg = }")
    if haloID is None:
        boxCen = np.zeros(3) + sP.boxSize / 2
    else:
        halo = sP.halo(haloID)
        boxCen = halo["GroupPos"]
    pxScale = boxSizeImg[0] / grid_dens.shape[0]

    ray_x = boxCen[0] - boxSizeImg[0] / 2 + posInds[0] * pxScale
    ray_y = boxCen[1] - boxSizeImg[1] / 2 + posInds[1] * pxScale
    ray_z = boxCen[2] - boxSizeImg[2] / 2
    print(f"Starting {ray_x = :.4f} {ray_y = :.4f} {ray_z = :4f}")

    # create theory-space master grids
    master_dens = np.zeros(nCells, dtype="float32")  # density for each ray segment
    master_dx = np.zeros(nCells, dtype="float32")  # pathlength for each ray segment
    master_temp = np.zeros(nCells, dtype="float32")  # temp for each ray segment
    master_vellos = np.zeros(nCells, dtype="float32")  # line of sight velocity

    # init
    boxSize = sP.boxSize if haloID is None else boxSizeImg[projAxis]
    dx_Mpc = sP.units.codeLengthToMpc(boxSize / nCells)

    # 'ray trace' a single pixel from front of box to back of box
    for i in range(nCells):
        # store cell properties
        master_vellos[i] = grid_vel[posInds[0], posInds[1], i]
        master_dens[i] = grid_dens[posInds[0], posInds[1], i]
        master_temp[i] = grid_temp[posInds[0], posInds[1], i]
        master_dx[i] = dx_Mpc  # constant

    # create spectrum
    master_mid, tau_master, _ = create_spectrum_from_traced_ray(
        sP, line, instrument, master_dens, master_dx, master_temp, master_vellos
    )

    # plot
    # plotName = f"spectrum_box_{sP.simName}_{line}_{nCells}_h{haloID}_{posInds[0]}-{posInds[1]}_z{sP.redshift:.0f}.pdf"


def generate_spectrum_voronoi(use_precomputed_mesh=True, compare=False, debug=1, verify=True):
    """Generate a single absorption spectrum by ray-tracing through the Voronoi mesh.

    Args:
      use_precomputed_mesh (bool): if True, use pre-computed Voronoi mesh connectivity from VPPP,
        otherwise use tree-based, connectivity-free method.
      compare (bool): if True, run both methods and compare results.
      debug (int): verbosity level for diagnostic outputs: 0 (silent), 1, 2, or 3 (most verbose).
      verify (bool): if True, brute-force distance calculation verify parent cell at each step.
    """
    # config
    sP = simParams(run="tng50-4", redshift=0.5)

    line = "OVI 1032"  #'LyA'
    instrument = "idealized"  # 'SDSS-BOSS'
    haloID = 150  # if None, then full box

    ray_offset_x = 0.0  # relative to halo center, in units of rvir
    ray_offset_y = 0.5  # relative to halo center, in units of rvir
    ray_offset_z = -2.0  # relative to halo center, in units of rvir
    projAxis = 2  # z, to simplify vellos for now

    fof_scope_mesh = False

    # load halo
    halo = sP.halo(haloID)

    print(f"Halo [{haloID}] center {halo['GroupPos']} and Rvir = {halo['Group_R_Crit200']:.2f}")

    # ray starting position, and total requested pathlength
    ray_start_x = halo["GroupPos"][0] + ray_offset_x * halo["Group_R_Crit200"]
    ray_start_y = halo["GroupPos"][1] + ray_offset_y * halo["Group_R_Crit200"]
    ray_start_z = halo["GroupPos"][projAxis] + ray_offset_z * halo["Group_R_Crit200"]

    total_dl = np.abs(ray_offset_z * 2) * halo["Group_R_Crit200"]  # twice distance to center

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype="float64")
    ray_dir[projAxis] = 1.0

    # load cell properties (pos,vel,species dens,temp)
    densField = "%s numdens" % lines[line]["ion"]
    velLosField = "vel_" + ["x", "y", "z"][projAxis]

    haloIDLoad = haloID if fof_scope_mesh else None  # if global mesh, then global gas load

    cell_pos = sP.snapshotSubsetP("gas", "pos", haloID=haloIDLoad)  # code
    cell_vellos = sP.snapshotSubsetP("gas", velLosField, haloID=haloIDLoad)  # code
    cell_temp = sP.snapshotSubsetP("gas", "temp_sfcold", haloID=haloIDLoad)  # K
    cell_dens = sP.snapshotSubset("gas", densField, haloID=haloIDLoad)  # ions/cm^3

    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos)  # km/s

    # ray starting position
    ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z])

    # use precomputed connectivity method, or tree-based method?
    if use_precomputed_mesh or compare:
        # load mesh neighbor connectivity
        if fof_scope_mesh:
            num_ngb, ngb_inds, offset_ngb = loadSingleHaloVPPP(sP, haloID=haloID)
        else:
            num_ngb, ngb_inds, offset_ngb = loadGlobalVPPP(sP)

        # ray-trace
        master_dx, master_ind = trace_ray_through_voronoi_mesh_with_connectivity(
            cell_pos,
            num_ngb,
            ngb_inds,
            offset_ngb,
            ray_pos,
            ray_dir,
            total_dl,
            sP.boxSize,
            debug,
            verify,
            fof_scope_mesh,
        )

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

    if (not use_precomputed_mesh) or compare:
        # construct neighbor tree
        tree = buildFullTree(cell_pos, boxSizeSim=sP.boxSize, treePrec=cell_pos.dtype, verbose=debug)
        NextNode, length, center, sibling, nextnode = tree

        if compare:
            ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z])  # reset
            master_ind2 = master_ind.copy()
            master_dx2 = master_dx.copy()

        # ray-trace
        master_dx, master_ind = trace_ray_through_voronoi_mesh_treebased(
            cell_pos, NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, sP.boxSize, debug, verify
        )

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

        if compare:
            assert np.allclose(master_dx, master_dx2)
            assert np.array_equal(master_ind, master_ind2)
            print(master_dx, master_dx2, "Comparison success.")

    # convert length units, all other units already appropriate
    master_dx = sP.units.codeLengthToMpc(master_dx)

    # create spectrum
    master_mid, tau_master, _ = create_spectrum_from_traced_ray(
        sP, line, instrument, master_dens, master_dx, master_temp, master_vellos
    )

    # plot
    # meshStr = 'vppp' if use_precomputed_mesh else 'treebased'
    # plotName = f"spectrum_voronoi_{sP.simName}_{line}_{meshStr}_h{haloID}_z{sP.redshift:.0f}.pdf"


def generate_spectra_voronoi_halo():
    """Generate a large grid of (halocentric) absorption spectra by ray-tracing through the Voronoi mesh."""
    # config
    sP = simParams(run="tng50-1", redshift=0.5)

    lineNames = ["MgII 2796", "MgII 2803"]
    instrument = "4MOST-HRS"  # 'SDSS-BOSS'
    haloID = 150  # 150 for TNG50-1, 800 for TNG100-1

    nRaysPerDim = 50  # total number of rays is square of this number
    projAxis = 2  # z, to simplify vellos for now, keep axis-aligned

    fof_scope_mesh = True  # if False then full box load

    # caching file
    saveFilename = "spectra_%s_z%.1f_halo%d-%d_n%d_%s_%s.hdf5" % (
        sP.simName,
        sP.redshift,
        haloID,
        projAxis,
        nRaysPerDim,
        instrument,
        "-".join(lineNames),
    )

    if isfile(saveFilename):
        # load cache
        EWs = {}
        with h5py.File(saveFilename, "r") as f:
            master_wave = f["master_wave"][()]
            flux = f["flux"][()]
            for line in lineNames:
                EWs[line] = f["EW_%s" % line.replace(" ", "_")][()]

        print(f"Loaded: [{saveFilename}]")

        return master_wave, flux, EWs

    # load halo
    halo = sP.halo(haloID)
    cen = halo["GroupPos"]
    mass = sP.units.codeMassToLogMsun(halo["Group_M_Crit200"])[0]
    size = 2 * halo["Group_R_Crit200"]

    print(f"Halo [{haloID}] mass = {mass:.2f} and Rvir = {halo['Group_R_Crit200']:.2f}")

    # ray starting positions, and total requested pathlength
    xpts = np.linspace(cen[0] - size / 2, cen[0] + size / 2, nRaysPerDim)
    ypts = np.linspace(cen[1] - size / 2, cen[1] + size / 2, nRaysPerDim)

    xpts, ypts = np.meshgrid(xpts, ypts, indexing="ij")

    # construct [N,3] list of search positions
    ray_pos = np.zeros((nRaysPerDim**2, 3), dtype="float64")

    ray_pos[:, 0] = xpts.ravel()
    ray_pos[:, 1] = ypts.ravel()
    ray_pos[:, 2] = cen[2] - size / 2

    # total requested pathlength (twice distance to halo center)
    total_dl = size

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype="float64")
    ray_dir[projAxis] = 1.0

    # load cell properties (pos,vel,species dens,temp)
    haloIDLoad = haloID if fof_scope_mesh else None  # if global mesh, then global gas load

    cell_pos = sP.snapshotSubsetP("gas", "pos", haloID=haloIDLoad)  # code

    # ray-trace
    rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode="full", nThreads=4)

    # load other cell properties
    velLosField = "vel_" + ["x", "y", "z"][projAxis]

    cell_vellos = sP.snapshotSubsetP("gas", velLosField, haloID=haloIDLoad)  # code
    cell_temp = sP.snapshotSubsetP("gas", "temp_sfcold", haloID=haloIDLoad)  # K

    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos)  # km/s

    # convert length units, all other units already appropriate
    rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # sample master grid
    master_mid, master_edges, tau_master = create_wavelength_grid(instrument=instrument)
    tau_master = np.zeros((nRaysPerDim**2, tau_master.size), dtype=tau_master.dtype)

    EWs = {}

    # start cache
    with h5py.File(saveFilename, "w") as f:
        f["master_wave"] = master_mid

    # loop over requested line(s)
    for line in lineNames:
        densField = "%s numdens" % lines[line]["ion"]
        cell_dens = sP.snapshotSubset("gas", densField, haloID=haloIDLoad)  # ions/cm^3

        # create spectra
        master_wave, tau_local, EW_local = create_spectra_from_traced_rays(
            sP, line, instrument, rays_off, rays_len, rays_dl, rays_inds, cell_dens, cell_temp, cell_vellos
        )

        assert np.array_equal(master_wave, master_mid)

        tau_master += tau_local
        EWs[line] = EW_local

        with h5py.File(saveFilename, "r+") as f:
            # save tau per line
            f["tau_%s" % line.replace(" ", "_")] = tau_local
            # save EWs per line
            f["EW_%s" % line.replace(" ", "_")] = EW_local

    # calculate flux and total EW
    flux = np.exp(-1 * tau_master)

    with h5py.File(saveFilename, "r+") as f:
        f["flux"] = flux

    print(f"Saved: [{saveFilename}]")

    return master_wave, flux, EWs
