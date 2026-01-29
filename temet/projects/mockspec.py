"""
The Synthetic Absorption Line Spectral Almanac (SALSA).

https://arxiv.org/abs/2510.19904 (Nelson+ 2026)
"""

import glob
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from temet.cosmo.cloudy import cloudyIon
from temet.plot.config import colors, figsize, linestyles, markers, percs
from temet.spectra.plot import (
    EW_distribution,
    EW_vs_coldens,
    dNdz_evolution,
    instrument_lsf,
    spectra_gallery_indiv,
    spectrum_plot_single,
)
from temet.spectra.spectrum import spectra_filepath
from temet.spectra.util import instruments, lines
from temet.util import simParams
from temet.util.helper import closest, running_median
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def metalAbundancesVsSolar(sim, ion="Mg II"):
    """Diagnostic plot of how much various metal abundances actual vary vs. the solar abundance ratio."""
    n_thresh = -8.0  # for second histogram
    nbins = 200
    minmax = [-11.0, -1.0]  # abundance

    species = ion.split(" ")[0]

    # load
    abund = sim.gas("metals_%s" % species)
    abund = np.log10(abund)

    # only cells which contribute (non-negligible) absorption are relevant
    numdens = sim.gas("%s numdens" % ion)
    numdens = np.log10(numdens)

    # get solar abundance ratio (mass ratio, to total)
    cloudy = cloudyIon(sim, el=species, redshiftInterp=False)

    solar_abund = cloudy._solarMetalAbundanceMassRatio(species)
    solar_abund = np.log10(solar_abund)

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel("%s Abundance [ log ]" % species)
    ax.set_ylabel("PDF")
    ax.set_yscale("log")

    # global hist
    yy, xx = np.histogram(abund, bins=nbins, range=minmax, density=True)
    xx = xx[:-1] + (minmax[1] - minmax[0]) / nbins / 2

    # ax.hist(abund, bins=100, label='%s z=%.1f' % (sim.simName,sim.redshift))
    ax.plot(xx, yy, "-", label="%s (all gas)" % sim)

    # restricted hist
    yy, xx = np.histogram(abund[numdens > n_thresh], bins=nbins, range=minmax, density=True)
    xx = xx[:-1] + (minmax[1] - minmax[0]) / nbins / 2

    ax.plot(xx, yy, "-", label="%s ($n_{\\rm %s} > %.1f$)" % (sim, ion, n_thresh))

    # solar abundance value
    ax.plot([solar_abund, solar_abund], ax.get_ylim(), "-", color="black", alpha=0.8, label="solar")

    # finish plot
    ax.legend(loc="best")
    fig.savefig("abund_ratio_%s_%d_%s.pdf" % (sim.simName, sim.snap, species))
    plt.close(fig)


def lightconeSpectraConfig(sim, max_redshift=5.0):
    """Combine available pathlengths through single snapshots to create a cosmologically long sightline.

    To create a cosmological sightline, i.e. over a significant pathlength much larger than the box size,
    possible e.g. complete from z=0 to z=4, we need to combine available pathlengths as available at the
    discrete simulation snapshots. Compute the available snapshots, and the number of pathlengths to
    take from each.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      max_redshift (float): all spectra go from redshift 0.0 to max_redshift, i.e. of the background quasar.

    Return:
      a 3-tuple composed of

      - **snaps** (list[int]): the snapshot numbers which need to be used to cover the sightline.
      - **num_boxes** (ndarray[int]): the number of times a full-box pathlength
        needs to be replicated for each snapshot.
      - **z_init** (list[list[float]]): for each snapshot, a list of redshifts, corresponding to
        where each of the replications should be started.
    """
    # test: if we are at z=z_init, how many times to we need to repeat the box to get to z=z_final?
    if 0:
        z_init = 0.7
        z_final = 1.0

        dl_init = sim.units.redshiftToComovingDist(z_init)
        dl_final = sim.units.redshiftToComovingDist(z_final)

        dl = (dl_final - dl_init) * 1000  # ckpc

        num = dl / sim.units.codeLengthToComovingKpc(sim.boxSize)

        print(f"{num = }")

    # for interpolation
    zz = np.linspace(0.0, max_redshift, 1000)
    ll = sim.units.redshiftToComovingDist(zz) * 1000  # ckpc

    # automatic spacing: decide snapshots to use
    snaps = sim.validSnapList(onlyFull=True)[::-1]
    redshifts = sim.snapNumToRedshift(snaps)

    w = np.where(redshifts <= max_redshift + 0.01)
    snaps = snaps[w]
    redshifts = redshifts[w]

    num_boxes = np.zeros(snaps.size, dtype="float32")

    # we take information from each snapshot until we reach the redshift/distance halfway to the next
    # e.g. z=0.05 between snap 99 (z=0) and snap 91 (z=0.1)
    redshifts_mid = np.hstack((redshifts[0], (redshifts[1:] + redshifts[:-1]) / 2, redshifts[-1]))

    # midpoints in cosmological distance between snapshots, i.e. the point where we switch to the
    # next snapshot. note: first value is special (z=0), and last value is special (max_redshift)
    dists_mid = sim.units.redshiftToComovingDist(redshifts_mid) * 1000  # ckpc

    z_init = []

    for i in range(snaps.size):
        # print(f'[{i:2d}] snap = {snaps[i]}, redshift = {redshifts_mid[i]:4.2f}, dist = {dists_mid[i]:10.2f}')
        dist_start = dists_mid[i]
        dist_stop = dists_mid[i + 1]
        dl = dist_stop - dist_start

        num_rep = dl / sim.units.codeLengthToComovingKpc(sim.boxSize)

        # we have no partial sightlines, i.e. less than boxSize long, and cannot create them from existing spectra
        num_boxes[i] = np.round(num_rep)

        dStr = f"from [z = {redshifts_mid[i]:4.2f} D = {dists_mid[i]:9.1f}]"
        dStr += f" to [z = {redshifts_mid[i + 1]:4.2f} D = {dists_mid[i + 1]:9.1f}]"
        dStr += f" with [snap = {snaps[i]} at z = {redshifts[i]:4.2f}] dl = {dl:9.1f} N = {num_boxes[i]:4.1f}"

        print(dStr)

        # make list of the redshift at which each replication should begin
        dists_loc = np.arange(int(num_boxes[i]))
        dists_loc = dist_start + dists_loc * (dl / num_boxes[i])

        assert dists_loc[-1] < dist_stop

        dist_excess = dist_stop - (dists_loc[-1] + sim.units.codeLengthToComovingKpc(sim.boxSize))
        dists_loc += dist_excess / 2  # split gap on both sides

        # convert distances into redshift
        dists_z = np.interp(dists_loc, ll, zz)

        # assert dists_z[0] > redshifts_mid[i] and dists_z[-1] < redshifts_mid[i+1] # need not be true

        z_init.append(dists_z)  # list of lists, one per snapshot

        # print(' z = ' + ' '.join(['%.3f' % z for z in dists_z]))

    return snaps, num_boxes, z_init


def lightconeSpectra(sim, instrument, ion, solar=False, add_lines=None):
    """Create a composite spectrum spanning a cosmological distance.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      instrument (str): specify observational instrument (in temet.spectra.spectrum.instruments).
      ion (str): space-separated name of ion e.g. 'Mg II'.
      solar (bool): if True, then adopt solar abundance ratio for the given species, instead of snap value.
      add_lines (list[str] or None): if not None, then a list of lines to include. otherwise, include all for this ion.

    Return:
      a 2-tuple composed of

      - **wave** (:py:class:`~numpy.ndarray`): 1d array, observed-frame wavelength grid [Ang].
      - **flux** (:py:class:`~numpy.ndarray`): 1d array, normalized flux values, from 0 to 1.
    """
    rng = np.random.default_rng(424242)

    # get replication configuration
    snaps, num_boxes, z_inits = lightconeSpectraConfig(sim)

    # load metadata from first (available) snapshot
    for snap in snaps:
        sim.setSnap(snap)
        fname = spectra_filepath(sim, ion, instrument=instrument, solar=solar)

        if isfile(fname):
            break

    with h5py.File(fname, "r") as f:
        wave = f["wave"][()]
        ray_total_dl = f["ray_total_dl"][()]
        num_spec = f["ray_pos"].shape[0]

    assert ray_total_dl == sim.boxSize  # otherwise generalize lightconeSpectraConfig()

    # allocate
    tau_master = np.zeros(wave.size, dtype="float64")

    # loop over snapshots
    for snap, num_box, z_init in zip(snaps, num_boxes, z_inits):
        sim.setSnap(snap)
        print(f"[{snap = :3d}] at z = {sim.redshift:.2f}, num spec = {num_box}")

        # check existence
        fname = spectra_filepath(sim, ion, instrument=instrument, solar=solar)

        if not isfile(fname):
            # (hopefully, no lines at the relevant wavelength range at this redshift)
            print(" skip, does not exist.")
            continue

        # select N at random
        spec_inds = rng.integers(low=0, high=num_spec, size=int(num_box))

        # open file
        fname = spectra_filepath(sim, ion, instrument=instrument, solar=solar)

        with h5py.File(fname, "r") as f:
            # load each spectrum individually, shift, and accumulate
            for spec_ind, z_local in zip(spec_inds, z_init):
                # allocate
                tau_local = np.zeros(wave.size, dtype="float64")

                # combine optical depth arrays for all transitions of this ion
                for key in f:
                    # skip unrelated non-tau datasets
                    if "tau_" not in key:
                        continue

                    # skip if not among the specific lines requested, unless we are including all
                    if add_lines is not None and key.replace("tau_", "") not in add_lines:
                        continue

                    # load entire tau array for one transition of this ion
                    print(f" [spec {spec_ind:5d}] at {z_local = :.3f} adding [{key}]")
                    tau_local += f[key][spec_ind, :]

                # shift in redshift according to the cumulative pathlength (z_init)
                wave_redshifted = wave * ((1 + z_local) / (1 + sim.redshift))

                # interpolate back onto the master (rest-frame) wavelength grid, and accumulate
                tau_redshifted = np.interp(wave, wave_redshifted, tau_local, left=0.0, right=0.0)

                tau_master += tau_redshifted

    # convert optical depth to flux
    flux = np.exp(-1 * tau_master)

    return wave, flux


def plotLightconeSpectrum(sim, instrument, ion, add_lines=None, SNR=None):
    """Plot a single lightcone spectrum.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      instrument (str): specify observational instrument (in temet.spectra.spectrum.instruments).
      ion (str): space-separated name of ion e.g. 'Mg II'.
      add_lines (list[str] or None): if not None, then a list of lines to include. otherwise, include all for this ion.
      SNR (float): if not None, then add noise to achieve this signal to noise ratio.
    """
    # zoom panel specifications (could be made into arguments)
    if ion == "H I":
        zooms = [[3200, 3300], [5550, 5600], [6840, 6900]]
        zooms_y = [[0.02, 1.0], [-0.01, 0.4], [-0.01, 0.70]]
    if ion == "C IV":
        zooms = [[3400, 3450], [5430, 5450], [7870, 7900]]
        zooms_y = [[0.9, 1.03], [-0.03, 1.03], [0.58, 1.03]]

    # generate, quick caching
    linesStr = "" if add_lines is None else ("_" + "-".join(add_lines))
    cache_file = "spec_cache_%s_%s_%s%s.hdf5" % (sim.simName, instrument, ion.replace(" ", ""), linesStr)

    if isfile(cache_file):
        print(f"Loading from: [{cache_file}]")
        with h5py.File(cache_file, "r") as f:
            wave = f["wave"][()]
            flux = f["flux"][()]
    else:
        # create now
        wave, flux = lightconeSpectra(sim, instrument, ion, add_lines=add_lines)

        with h5py.File(cache_file, "w") as f:
            f["wave"] = wave
            f["flux"] = flux
        print(f"Saved: [{cache_file}]")

    # add noise? ("signal" is now 1.0)
    if SNR is not None:
        rng = np.random.default_rng(424242)
        noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux.shape)
        # flux_noisefree = flux.copy()
        flux += noise
        # achieved SNR = 1/stddev(noise)
        flux = np.clip(flux, 0, np.inf)  # clip negative values at zero

    # plot
    fig = plt.figure(figsize=(figsize[0] * 1.6, figsize[1] * 1.2))
    # (ax_top, ax_top_zoom), (ax_bottom, ax_bottom_zoom) = fig.subplots(nrows=2, ncols=2)
    gs = fig.add_gridspec(2, len(zooms))
    ax_top = fig.add_subplot(gs[0, :])
    ax_zooms = [fig.add_subplot(gs[1, i]) for i in range(len(zooms))]

    # top panel: strong absorbers, down to saturation
    ax_top.set_ylim([-0.03, 1.05])
    ax_top.set_xlim([wave.min(), wave.max()])
    ax_top.set_xlabel("Wavelength [ Ang ]")
    ax_top.set_ylabel("Normalized Flux")

    ax_top.step(wave, flux, "-", where="mid", lw=1, c="black", label="%s %s" % (instrument, ion))
    ax_top.legend(loc="best")

    # debugging: (we have some small wavelength regions which are covered by no volume, due to
    # requirement of sampling integer numbers of boxes -- not important for rare absorption, but
    # causes erroneous high flux spikes where we are absorption dominated e.g. high-z LyA forest)
    if 0:
        for z in [4.897, 4.795, 4.696, 4.600, 4.506]:
            z_obs = lines["LyA"]["wave0"] * (1 + z)
            ax_top.plot([z_obs, z_obs], ax_top.get_ylim(), "-", color="black")
        for z in [4.420, 4.340, 4.262, 4.186, 4.112, 4.039, 3.968, 3.898, 3.829, 3.762, 3.697, 3.632, 3.569, 3.507]:
            z_obs = lines["LyA"]["wave0"] * (1 + z)
            ax_top.plot([z_obs, z_obs], ax_top.get_ylim(), "-", color="green")

    # helper function for lower zoom detail(s)
    def _add_connecting_lines(ax):
        xlim = ax.get_xlim()
        ylim_top = ax_top.get_ylim()
        c1 = ConnectionPatch(
            xyA=[xlim[0], ax.get_ylim()[1]],
            xyB=[xlim[0], ax_top.get_ylim()[0]],
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax_top,
            color="black",
            lw=1,
            alpha=0.3,
            arrowstyle="-",
        )
        c2 = ConnectionPatch(
            xyA=[xlim[1], ax.get_ylim()[1]],
            xyB=[xlim[1], ax_top.get_ylim()[0]],
            coordsA="data",
            coordsB="data",
            axesA=ax,
            axesB=ax_top,
            color="black",
            lw=1,
            alpha=0.3,
            arrowstyle="-",
        )
        ax_top.add_artist(c1)
        ax_top.add_artist(c2)

        ax_top.fill_between(xlim, [ylim_top[0], ylim_top[0]], [ylim_top[1], ylim_top[1]], color="black", alpha=0.1)
        ax_top.plot([xlim[0], xlim[0]], ylim_top, color="black", alpha=0.3, lw=1)
        ax_top.plot([xlim[1], xlim[1]], ylim_top, color="black", alpha=0.3, lw=1)

        w = np.where((wave > xlim[0]) & (wave <= xlim[1]))
        ax_top.step(wave[w], flux[w], "-", where="mid", lw=1, c=ax.get_lines()[0].get_color())

    # plot lower zoom detail(s)
    for i in range(len(zooms)):
        ax_zooms[i].set_ylim(zooms_y[i])
        ax_zooms[i].set_xlim(zooms[i])
        ax_zooms[i].set_xlabel("Wavelength [ Ang ]")
        ax_zooms[i].set_ylabel("Normalized Flux")

        ax_zooms[i].step(wave, flux, "-", where="mid", c=colors[i])

        # if SNR is not None and i == 0: # custom
        #    ax_zooms[i].step(wave, flux_noisefree, '-', where='mid', c=colors[len(zooms)+i])

        _add_connecting_lines(ax_zooms[i])

    # finish
    fig.savefig("spectrum_lightcone_%s_%s_%s%s.pdf" % (sim.simName, ion.replace(" ", ""), instrument, linesStr))
    plt.close(fig)


def _galaxy_sightline_sample(sim, line, instrument, mstar_range, D_max):
    """Helper for below. Identify all sightlines passing near a galaxy sample.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): name of spectral line, e.g. 'Mg II'.
      instrument (str): name of instrument, e.g. 'SDSS-BOSS'.
      mstar_range (list[float]): 2-element list, min and max stellar mass [log Msun].
      D_max (float): maximum impact parameter [pkpc].
    """

    def _sim_posvel_to_wave(sim, pos_los_code, vel_los_code, wave0):
        """Convert 3D position and velocity into the corresponding redshifted wavelength of a transition.

        The line-of-sight position (i.e. z-coordinate) and line-of-sight peculiar velocity (i.e. v_z)
        become an effective redshift and thus redshifted wavelength of a specific rest-frame wavelength.

        Args:
          sim (:py:class:`~util.simParams`): simulation instance.
          pos_los_code (:py:class:`~numpy.ndarray`[float]): code units positions, along los coordinate.
          vel_los_code (:py:class:`~numpy.ndarray`[float]): code units velocities, in los direction.
          wave0 (float): rest-frame wavelength.
        """
        # create lookup table for (dist_los -> redshift)
        # note: sim.redshift occurs at the front intersection (beginning) of the box
        z_vals = np.linspace(sim.redshift, sim.redshift + 0.1, 200)

        z_lengths = sim.units.redshiftToComovingDist(z_vals) - sim.units.redshiftToComovingDist(sim.redshift)
        assert z_lengths.max() > sim.units.codeLengthToComovingMpc(sim.boxSize)

        dist_los = sim.units.codeLengthToComovingMpc(pos_los_code)  # [cMpc]
        z_cosmo = np.interp(dist_los, z_lengths, z_vals)

        # doppler shift
        vel_los = sim.units.particleCodeVelocityToKms(vel_los_code)
        z_doppler = vel_los / sim.units.c_km_s

        # effective redshift
        z_eff = (1 + z_doppler) * (1 + z_cosmo) - 1

        # expected wavelength
        gal_wave = wave0 * (1 + z_eff)

        return gal_wave

    # galaxy sample selection
    mstar = sim.subhalos("mstar_30kpc_log")

    subIDs = np.where((mstar > mstar_range[0]) & (mstar <= mstar_range[1]))[0]

    pos = sim.subhalos("SubhaloPos")[subIDs, :]
    vel = sim.subhalos("SubhaloVel")[subIDs, :]
    cen_flag = sim.subhalos("cen_flag")[subIDs]

    print(f"Selected [{len(subIDs)}] galaxies with M* in [{mstar_range[0]:.1f} - {mstar_range[1]:.1f}].")
    print(f"Note: [{np.sum(cen_flag)}] of [{len(subIDs)}] are centrals.")

    # load spectra metadata
    ion = lines[line]["ion"]
    filepath = spectra_filepath(sim, ion, instrument=instrument)

    with h5py.File(filepath, "r") as f:
        # load metadata
        ray_dir = f["ray_dir"][()]
        ray_pos = f["ray_pos"][()]
        ray_total_dl = f["ray_total_dl"][()]

    # find all spectra that pass within D_max of any galaxy in the sample
    spec_mask = np.zeros(ray_pos.shape[0], dtype="int32")

    dir_ind = np.where(ray_dir == 1)[0][0]
    sky_inds = list({0, 1, 2} - {dir_ind})  # e.g. [0,1] for projAxis == 2

    assert ray_pos[:, dir_ind].max() == 0  # otherwize generalize (dist cut in los direction)
    assert ray_total_dl == sim.boxSize  # otherwise generalize (dist cut in los direction)

    ray_pos = ray_pos[:, sky_inds]

    # allocate for intersections
    n_alloc = int(len(subIDs) * np.sqrt(spec_mask.size))  # heuristic

    rays_subID = np.zeros(n_alloc, dtype="int32")
    rays_subInd = np.zeros(n_alloc, dtype="int32")
    rays_dist = np.zeros(n_alloc, dtype="float32")
    rays_ind = np.zeros(n_alloc, dtype="int32")
    gals_nrays = np.zeros(len(subIDs), dtype="int32")
    rays_count = 0

    for i, (sub_id, sub_pos) in enumerate(zip(subIDs, pos)):
        if i % 10 == 0:
            print(f"[{i:3d}] of [{len(subIDs)}], now {sub_id = }, {rays_count = }.")

        # periodic transverse (i.e. plane of the sky) distances
        dists = sim.periodicDists(sub_pos[sky_inds], ray_pos)
        dists = sim.units.codeLengthToKpc(dists)  # pkpc

        ray_inds_loc = np.where(dists <= D_max)[0]

        # stamp
        n_loc = len(ray_inds_loc)

        spec_mask[ray_inds_loc] = 1  # could store count

        rays_subID[rays_count : rays_count + n_loc] = sub_id
        rays_subInd[rays_count : rays_count + n_loc] = i
        rays_dist[rays_count : rays_count + n_loc] = dists[ray_inds_loc]
        rays_ind[rays_count : rays_count + n_loc] = ray_inds_loc
        gals_nrays[i] = n_loc
        rays_count += n_loc

    # reduce
    rays_subID = rays_subID[0:rays_count]
    rays_subInd = rays_subInd[0:rays_count]
    rays_dist = rays_dist[0:rays_count]
    rays_ind = rays_ind[0:rays_count]

    # check coverage of galaxy sample
    nrays_near = len(np.where(spec_mask > 0)[0])

    print(f"Of [{spec_mask.size}] sightlines, found [{nrays_near}] within [{D_max = :.1f}] pkpc.")
    print(f"In total, have [{rays_count}] galaxy-sightline pairs. Loading [{line}] [{instrument}] spectra...")

    # calculate galaxy effective redshift, and corresponding wavelength for this transition
    gal_wave = _sim_posvel_to_wave(sim, pos[:, dir_ind], vel[:, dir_ind], lines[line]["wave0"])

    return rays_count, rays_subID, rays_subInd, rays_dist, rays_ind, gal_wave, subIDs


def _optical_depth_map_2d_calc(sim, line, instrument, mstar_range, D_max):
    """Helper for below. Calculate and cache a 2D stacked galaxy-centric optical depth map.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): name of spectral line, e.g. 'Mg II'.
      instrument (str): name of instrument, e.g. 'SDSS-BOSS'.
      mstar_range (list[float]): 2-element list, min and max stellar mass [log Msun].
      D_max (float): maximum impact parameter [pkpc].
    """
    # config
    dv_range = 800  # km/s

    # check cache
    lineName = line.replace(" ", "-")
    specStr = f"{mstar_range[0]:.1f}_{mstar_range[1]:.1f}_{dv_range:.0f}_{D_max:.0f}"
    cachefile = sim.cachePath + f"taumap_{sim.simName}_{sim.snap}_{lineName}_{instrument}_{specStr}.hdf5"

    if isfile(cachefile):
        # load
        r = {}
        with h5py.File(cachefile, "r") as f:
            for key in f:
                r[key] = f[key][()]
        print(f"Loaded [{cachefile}].")
        return r

    # galaxy sample
    rays_count, rays_subID, rays_subInd, rays_dist, rays_ind, gal_wave, subIDs = _galaxy_sightline_sample(
        sim, line, instrument, mstar_range, D_max
    )

    # load spectra metadata
    ion = lines[line]["ion"]
    filepath = spectra_filepath(sim, ion, instrument=instrument)

    with h5py.File(filepath, "r") as f:
        # load metadata
        wave = f["wave"][()]

    # determine common (wave -> dv) and size of dv_range in spectral bins
    wave_z = lines[line]["wave0"] * (1 + sim.redshift)
    dlambda = wave - wave_z
    wave_dv = dlambda / wave_z * sim.units.c_km_s

    w = np.where((wave_dv > -dv_range) & (wave_dv <= dv_range))[0]
    n = int(np.ceil(len(w) / 2))

    _, ind_cen = closest(wave, wave_z)
    wave_dv = wave_dv[ind_cen - n + 1 : ind_cen + n + 1]  # wave_dv[w]

    # allocate spec in terms of global wave, and narrower dv
    flux = np.zeros((rays_count, wave.size), dtype="float32")
    flux_dv = np.zeros((rays_count, wave_dv.size), dtype="float32")

    # load spectra (note: rays_ind contains duplicates, and is not sorted)
    dset = "flux"  #'tau_%s' % line.replace(' ','_')

    with h5py.File(filepath, "r") as f:
        for i, ray_ind in enumerate(rays_ind):
            if i % 100 == 0:
                print(i)
            flux[i] = f[dset][ray_ind, :]

    # stamp
    for i in range(rays_count):
        # select closest bin to center wavelength
        # TODO: is this good enough, or do we need to interpolate? (depends on how big dv bins are?)
        wave_cen, ind_cen = closest(wave, gal_wave[rays_subInd[i]])

        # wave_err = wave_cen - gal_wave[rays_subInd[i]]
        # print(i, rays_subInd[i], gal_wave[rays_subInd[i]], ind_cen, wave_err)

        # stamp
        flux_dv[i, :] = flux[i, ind_cen - n + 1 : ind_cen + n + 1]

    # save cache file
    r = {
        "rays_subID": rays_subID,
        "rays_subInd": rays_subInd,
        "rays_dist": rays_dist,
        "rays_ind": rays_ind,
        "rays_count": rays_count,
        "wave": wave,
        "flux": flux,
        "wave_dv": wave_dv,
        "flux_dv": flux_dv,
        "mstar_range": mstar_range,
        "dv_range": dv_range,
        "D_max": D_max,
        "subIDs": subIDs,
    }

    with h5py.File(cachefile, "w") as f:
        for key in r:
            f[key] = r[key]
        print(f"Saved [{cachefile}].")

    return r


def optical_depth_map_2d(sim, line, instrument, SNR=10.0, tau_minmax=(0.0, 0.3)):
    """Create 2D map of galaxy-centric apparent optical depth, as a function of transverse and LoS distance.

    Inspired by the usual Steidel+/KBSS style (e.g. https://arxiv.org/abs/2503.20037).

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      SNR (float): if not None, then add noise to achieve this signal to noise ratio.
      tau_minmax (list[float]): 2-tuple of min-max colorbar values for optical depth.
    """
    # config
    D_binsize = 10.0  # pkpc
    aspect = 1.8
    cmap = "inferno_r"

    # load/calculate
    mstar_range = [9.9, 10.0]  # [9.5,10.0] #[9.9, 10.0]
    D_max = 200.0  # 100 # pkpc

    stack = _optical_depth_map_2d_calc(sim, line, instrument, mstar_range, D_max)

    # add noise? ("signal" is 1.0)
    if SNR is not None:
        rng = np.random.default_rng(424242)
        noise = rng.normal(loc=0.0, scale=1 / SNR, size=stack["flux_dv"].shape)
        stack["flux_dv"] += noise  # # achieved SNR = 1/stddev(noise)
        stack["flux_dv"] = np.clip(stack["flux_dv"], 0, np.inf)  # clip negative values at zero

    # bin and stack in terms of projected (transverse) distance
    D_nbins = int(stack["D_max"] / D_binsize)
    D_binedges = np.linspace(0.0, stack["D_max"], D_nbins + 1)
    D_bincen = (D_binedges[:-1] + D_binedges[1:]) / 2

    flux_stack = np.zeros((stack["wave_dv"].size, D_nbins), dtype="float32")
    bin_count = np.zeros(D_nbins, dtype="int32")

    for i in range(stack["rays_count"]):
        bin_ind = int(np.floor(stack["rays_dist"][i] / D_binsize))
        flux_stack[:, bin_ind] += stack["flux_dv"][i, :]
        bin_count[bin_ind] += 1

    flux_stack /= bin_count

    # convert to apparent optical depth
    tau_stack = -np.log(flux_stack)

    # smooth?
    # sigma_xy = [1.0, 0.0]
    # tau_stack = gaussian_filter(tau_stack, sigma_xy, mode='reflect', truncate=5.0)

    # plot
    fig, ax = plt.subplots(figsize=[6.0, 6.0 * aspect])

    ax.set_xlabel(r"$D_{\rm transverse}$ [ kpc ]")
    ax.set_ylabel(r"$v_{\rm LOS}$ [ km/s ]")
    ax.set_title(line)

    xlim = [0, stack["D_max"]]
    ylim = [-stack["dv_range"], +stack["dv_range"]]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    norm = Normalize(vmin=tau_minmax[0], vmax=tau_minmax[1])
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    s = ax.imshow(tau_stack, extent=extent, norm=norm, origin="lower", aspect="auto", cmap=cmap)

    xx, yy = np.meshgrid(D_bincen, stack["wave_dv"], indexing="xy")
    ax.contour(xx, yy, tau_stack, np.linspace(tau_minmax[1] / 10, tau_minmax[1], 5), cmap=cmap.replace("_r", ""))

    # finish plot
    cax = make_axes_locatable(ax).append_axes("right", size="10%", pad=0.1)
    # cb = plt.colorbar(sm, cax=cax, ticks=N_vals)
    cb = plt.colorbar(s, cax=cax)
    cb.ax.set_ylabel(r"Apparent Optical Depth $\tau_{\rm app} = - \ln{(F)}$")

    fig.savefig(f"tau2d_{sim}_{line}_{instrument}.pdf".replace(" ", "-"))
    plt.close(fig)


def impact_parameter_profile(sim, line, instrument, spec=False, SNR=100):
    """Create a 1D profile of absorption EW vs impact parameter, for a given galaxy sample.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      spec (list[str]): if True, then add panels showing actual spectra for one of the galaxies.
      SNR (float): if not None, then add noise to achieve this signal to noise ratio.
    """
    mstar_range = [10.3, 10.4]  # [9.5,10.0] #[9.9, 10.0]
    D_max = 300.0  # 100 # pkpc
    spec_dwave = 10.0 if "SDSS" in instrument else 2.0  # Ang

    rays_count, rays_subID, rays_subInd, rays_dist, rays_ind, gal_wave, subIDs = _galaxy_sightline_sample(
        sim, line, instrument, mstar_range, D_max
    )

    # load EW
    ion = lines[line]["ion"]
    filepath = spectra_filepath(sim, ion, instrument=instrument)

    with h5py.File(filepath, "r") as f:
        # load metadata
        EW = f["EW_" + line.replace(" ", "_")][()]

    EW = EW[rays_ind]

    # plot
    if spec:
        fig = plt.figure(figsize=(figsize[0] * 1.6, figsize[1] * 0.8))
        gs = fig.add_gridspec(2, 4)
        ax = fig.add_subplot(gs[:, 0:2])
        # ax_spec = fig.add_subplot(gs[0,2]) # [0,2] [0,3] [1,2] [1,3]
        ax_spec = [fig.add_subplot(gs[i % 2, int(i / 2) + 2]) for i in range(len(spec))]
    else:
        fig, ax = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 0.8])

    ax.set_xlabel(r"Impact Parameter [ kpc ]")
    ax.set_ylabel(line + r" Equivalent Width [ $\AA$ ]")
    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

    xlim = [5.0, D_max]
    ylim = [1e-3, 2.0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.plot(rays_dist, EW, "o", ms=3, alpha=0.5, color="black", zorder=0)

    # 1-sigma band
    xm, ym, _, pm = running_median(rays_dist, EW, nBins=30, percs=percs)
    (l,) = ax.plot(xm, ym, "-", color="black", zorder=2)
    ax.fill_between(xm, pm[0, :], pm[-1, :], color=l.get_color(), alpha=0.3, zorder=2)

    # mark specific galaxy
    num_gals = 2
    spec_colors = []
    spec_inds = []
    spec_gal_wave = []

    for i in range(num_gals):
        if "MgII" in line:
            if i == 0:
                ww = np.where((rays_dist > 33) & (rays_dist < 35) & (EW > 1.0) & (EW < 1.1))[0][0]
            if i == 1:
                ww = np.where((rays_dist > 75) & (rays_dist < 80) & (EW > 1.2) & (EW < 3.0))[0][0]
        if "SiIII" in line:
            if i == 0:
                ww = np.where((rays_dist > 40) & (rays_dist < 45) & (EW > 0.7) & (EW < 1.0))[0][0]
            if i == 1:
                ww = np.where((rays_dist > 75) & (rays_dist < 80) & (EW > 0.6) & (EW < 1.0))[0][0]

        (l,) = ax.plot(rays_dist[ww], EW[ww], "o", ms=8, mfc="none", mew=5, zorder=3)

        spec_colors.append(l.get_color())
        spec_inds.append(rays_ind[ww])
        spec_gal_wave.append(gal_wave[rays_subInd[ww]])

        print(f" Marker: {rays_ind[ww] = }, {rays_subID[ww]}, {rays_subInd[ww]}, {rays_dist[ww]}, {EW[ww]}")

    # annotate details
    label = "%s\n%.1f < log M$_\\star$/M$_\\odot$ < %.1f\n%d galaxy-sightline pairs" % (
        sim,
        mstar_range[0],
        mstar_range[1],
        rays_count,
    )
    ax.annotate(label, xy=(0.02, 0.03), xycoords="axes fraction", fontsize="large")

    # add panels showing actual spectra?
    if spec:
        rng = np.random.default_rng(424242)

        for line_plot, ax in zip(spec, ax_spec):
            # setup sub-panel
            print(" Plotting spectrum for line: [%s]" % line_plot)
            ax.set_xlabel("Wavelength [ Ang ]", fontsize="x-large")
            ax.set_ylabel(f"{line_plot} Flux", fontsize="x-large")

            ion = lines[line_plot]["ion"]
            filepath = spectra_filepath(sim, ion, instrument=instrument)

            # get redshifted wavelength of this line at the galaxy systemic
            for i in range(num_gals):
                z_eff = (spec_gal_wave[i] / lines[line]["wave0"]) - 1
                wave_systemic = lines[line_plot]["wave0"] * (1 + z_eff)

                ax.set_xlim([wave_systemic - spec_dwave, wave_systemic + spec_dwave])
                ax.set_ylim([0.0, 1.05])

                ax.plot([wave_systemic, wave_systemic], ax.get_ylim(), "--", color=spec_colors[i], alpha=0.8)

                # load spectrum
                with h5py.File(filepath, "r") as f:
                    wave = f["wave"][()]
                    flux = f["flux"][spec_inds[i], :]

                if flux.min() > 0.9:
                    ax.set_ylim([0.9, 1.01])

                # add noise? ("signal" is now 1.0)
                if SNR is not None:
                    noise = rng.normal(loc=0.0, scale=1 / SNR, size=flux.shape)
                    flux += noise
                    flux = np.clip(flux, 0, np.inf)

                ax.step(wave, flux, "-", where="mid", c=spec_colors[i])

    # finish plot
    saveFilename = f"profileEW_{sim}_{line}_{instrument}_mstar={mstar_range[0]:.1f}-{mstar_range[1]:.1f}.pdf"
    fig.savefig(saveFilename.replace(" ", "-"))
    plt.close(fig)


def doublet_ratio(sim, line1, line2, instrument):
    """Create a 2D histogram of doublet ratio vs equivalent width.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line1 (str): first line transition.
      line2 (str): second line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
    """
    ion = lines[line1]["ion"]
    assert ion == lines[line2]["ion"], "Must be same ion."

    filepath = spectra_filepath(sim, ion, instrument=instrument)
    with h5py.File(filepath, "r") as f:
        # load metadata
        EW1 = f["EW_" + line1.replace(" ", "_")][()]
        EW2 = f["EW_" + line2.replace(" ", "_")][()]

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = EW1 / EW2

    theoretical_ratio = lines[line1]["f"] / lines[line2]["f"]

    # plot 1d histogram
    fig, ax = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 0.8])
    ax.set_xlabel(f"[{line1}/{line2}] Doublet Ratio")
    ax.set_ylabel("PDF")

    ax.hist(ratio, bins=50, histtype="stepfilled", density=True)
    ax.set_yscale("log")

    ax.plot([theoretical_ratio, theoretical_ratio], ax.get_ylim(), "--", color="black", label="Theoretical")
    ax.legend(loc="best")

    fig.savefig(f"doublet_ratio_{sim}_{line1}_{line2}_{instrument}.pdf".replace(" ", "-"))
    plt.close(fig)

    # plot 2d contours vs EW
    fig, ax = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 0.8])
    ax.set_xlabel(f"[{line1}] Equivalent Width [ $\\AA$ ]")
    ax.set_ylabel(f"[{line1}/{line2}] Doublet Ratio")

    # ax.set_xscale('log')

    ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized
    ax.hexbin(EW1, ratio, gridsize=50, mincnt=1, xscale="log", cmap="inferno", bins="log", zorder=0)

    ax.plot(ax.get_xlim(), [theoretical_ratio, theoretical_ratio], "--", color="black", label="Theoretical")
    ax.legend(loc="upper right")

    fig.savefig(f"doublet_ratio2d_{sim}_{line1}_{line2}_{instrument}.pdf".replace(" ", "-"))
    plt.close(fig)


def mean_transmitted_flux(sim, line, instrument, redshifts):
    """Calculate and plot the mean transmitted flux as a function of redshift.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      redshifts (list[float]): 1D array of redshift bin edges.
    """
    ion = lines[line]["ion"]

    # allocate
    mean_flux = np.zeros(len(redshifts), dtype="float32")
    percs_flux = np.zeros((len(redshifts), len(percs)), dtype="float32")

    for i in range(len(redshifts)):
        print(f" At redshift [{redshifts[i]:.2f}]...")
        # load
        sim.setRedshift(redshifts[i])
        filepath = spectra_filepath(sim, ion, instrument=instrument)

        with h5py.File(filepath, "r") as f:
            # load metadata
            wave = f["wave"][()]
            flux = f["flux"][()]  # [0:100000,:] # subsample for speed

        # what wavelength range is covered by this simulation box at this redshift?
        wave_min = lines[line]["wave0"] * (1 + sim.redshift)
        wave_max = lines[line]["wave0"] * (1 + sim.redshift + sim.dz)

        print(f"  Wavelength range: [{wave_min:.1f} - {wave_max:.1f}] Ang.")

        w = np.where((wave > wave_min) & (wave < wave_max))[0]

        mean_flux[i] = np.mean(flux[:, w])
        mean_tau = -np.log(mean_flux[i])
        percs_flux[i, :] = np.mean(np.percentile(flux[:, w], percs, axis=0), axis=1)

        print(mean_flux, mean_tau)

    # plot
    fig, ax = plt.subplots(figsize=[figsize[0] * 0.8, figsize[1] * 0.8])
    ax.set_xlabel("Redshift")
    ax.set_ylabel(f"Mean Transmitted Flux [{line}]")

    ax.set_xlim([1.4, 4.1])
    ax.set_ylim([0.0, 1.05])

    ax.plot(redshifts, mean_flux, "-", color="black")
    # ax.fill_between(redshifts, mean_flux - std_flux, mean_flux + std_flux, color='black', alpha=0.3)

    # see observational data e.g.
    # https://arxiv.org/abs/astro-ph/9911196
    # https://arxiv.org/abs/2310.00524

    fig.savefig(f"mean_transmitted_flux_{sim}_{line}_{instrument}.pdf".replace(" ", "-"))
    plt.close(fig)


def vis_overview(sP, haloID=None):
    """Visualize large-scale box, and halo-scale zoom, in an ion column density."""
    nPixels = 2000
    axes = [0, 2]  # x,y
    labelZ = True
    labelScale = True
    labelSim = True
    method = "sphMap"

    partType = "gas"
    partField = "O VI" if sP.redshift < 0.5 else "Ne VIII"

    panels = [{}]

    class plotConfig:
        plotStyle = "edged"
        # colorbars  = False
        colorbarOverlay = True

    if haloID is None:
        # large-scale box
        valMinMax = [12.0, 15.0]
        plotConfig.saveFilename = "./boxImage_%s_%s-%s.pdf" % (sP.simName, partType, partField)

        renderBox(panels, plotConfig, locals(), skipExisting=False)
    else:
        # single halo
        valMinMax = [13.5, 15.0]
        nPixels = 1000
        size = 3.5  # 2.5
        sizeType = "rVirial"
        labelScale = "physical"
        labelHalo = "mhalo,mstar,id"
        method = "sphMap_global"

        subhaloInd = sP.halo(haloID)["GroupFirstSub"]

        plotConfig.saveFilename = "./haloImage_%s_%d_%s-%s.pdf" % (sP.simName, haloID, partType, partField)
        plotConfig.rasterPx = [600, 600]

        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def ion_redshift_coverage(sim, single=False, all=True, lowz=False):
    """Schematic plot showing ion/redshift/instrument/etc coverage of available mock spectra files.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      single (bool): show only 1 marker per ion/full snap, instead of per instrument/config.
      all (bool): show all possible, otherwise show only existing.
      lowz (bool): make a log-log plot that focuses on low redshift.
    """
    # config
    line_alpha = 0.3

    # dataset selection
    datasets = []

    # (A) actually done
    path = sim.postPath + "AbsorptionSpectra/spectra_*fullbox*combined.hdf5"
    files = sorted(glob.glob(path))

    for file in files:
        with h5py.File(file, "r") as f:
            lineNames = f.attrs["lineNames"]

        filename = file.split("spectra_")[1]
        simname, z, config, inst, ion, _ = filename.split("_")
        z = float(z.replace("z", ""))
        datasets.append([ion, inst, z, config, lineNames, True])

    ions = list({ds[0] for ds in datasets})
    insts = list({ds[1] for ds in datasets})

    insts += ["COS-G230L"]

    # (B) all possible
    redshifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]

    if all:
        # ions = ['H I','Mg II','Fe II','Si II','Si III','Si IV','N V','C II','C IV','O VI','Ca II','Zn II']
        insts_to_fill = insts  # ['COS-G130M']#, 'SDSS-BOSS', '4MOST-HRS'] # insts
        config = "dummy"

        for line, props in lines.items():
            ion = props["ion"].replace(" ", "")
            for inst in insts_to_fill:
                for z in redshifts:
                    wave_z = props["wave0"] * (1 + z)

                    if wave_z > instruments[inst]["wave_min"] and wave_z < instruments[inst]["wave_max"]:
                        datasets.append([ion, inst, z, config, [line], False])

        # ions = list(dict.fromkeys([info['ion'] for line,info in lines.items()])) # unique
        ions = list({ds[0] for ds in datasets})

    # (c) only one marker per ion/full snap
    if single:
        datasets = []
        for line, props in lines.items():
            ion = props["ion"].replace(" ", "")
            inst = insts[0]
            config = "dummy"

            for z in redshifts:
                wave_z = props["wave0"] * (1 + z)

                # if wave_z > 0.0 and wave_z < instruments[inst]['wave_max']:
                datasets.append([ion, inst, z, config, [line], True])

        ions = list({ds[0] for ds in datasets})

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 1.2, figsize[1] * 0.9])
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(2)  # elements below z=1 are rasterized

    ax.set_xlabel("Observed Wavelength [ Ang ]")
    ax.set_ylabel("Redshift")

    if lowz:  # low-z focused (log-log)
        xlim = [700, 11500]  # [950, 10000]
        xticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 10000]
        ylim = [0.07, 7]  # [0.08, 6] # z=0.1 minimum currently used
        yticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ax.set_yscale("log")
        ax.set_xscale("log")
    else:  # high-z focused (linear)
        xlim = [900, 10500]
        if all:
            xlim[0] = 400
        xticks = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        ylim = [0.0, 5.2]
        yticks = [0.1, 0.5, 0.7, 1.0, 2.0, 3.0, 4.0, 5.0]

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_yticks(yticks)
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(xticks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())

    # loop over ions
    for i, ion in enumerate(ions):
        lines_loc = []
        wave0_loc = []
        for line, props in lines.items():
            if props["ion"].replace(" ", "") == ion:
                lines_loc.append(line)
                wave0_loc.append(props["wave0"])

        # draw line for each transition
        z = np.logspace(-2.0, np.log10(ylim[1]), 40)

        for j, wave0 in enumerate(wave0_loc):
            wave_z = wave0 * (1 + z)

            label = ion if j == 0 else ""
            ls = linestyles[i // len(colors)]
            (l,) = ax.plot(wave_z, z, ls, lw=1, c=colors[i], alpha=line_alpha, zorder=0)  # , label=label)

    # loop over datasets
    labels = []

    for dataset in datasets:
        ion, inst, z, config, lineNames, exists = dataset

        for lineName in lineNames:
            wave_z = lines[lineName.replace("_", " ")]["wave0"] * (1 + z)
            label = ion if ion not in labels else ""
            labels.append(ion)
            marker = markers[insts.index(inst)]

            style = {"fillstyle": "full", "ms": 8}
            if not exists:
                style["fillstyle"] = "none"
                style["ms"] = 4
                style["markeredgewidth"] = 1

            # vertical offset (in display coordinate space) based on inst, to separate out overlapping pts
            z_plot = z
            if len(insts) > 1:
                _, z_display = ax.transData.transform((wave_z, z))
                z_display += 7.0 * (insts.index(inst) - len(insts) / 2)
                _, z_plot = ax.transData.inverted().transform((wave_z, z_display))

            if single:
                _, z_display = ax.transData.transform((wave_z, z))
                z_display += 0.5 * (ions.index(ion) - len(ions) / 2)
                _, z_plot = ax.transData.inverted().transform((wave_z, z_display))

            ax.plot(wave_z, z_plot, marker, c=colors[ion], label=label, zorder=1, **style)

    # second legend (instrument markers)
    j_off = 1 if single else 0
    handles = [plt.Line2D([0], [0], color="black", lw=0, marker=markers[j + j_off]) for j in range(len(insts))]
    labels = insts

    legend2 = ax.legend(handles, labels, loc="lower right")
    ax.add_artist(legend2)

    # first legend
    handles = [plt.Line2D([0], [0], color=colors[i], ls=linestyles[i // len(colors)]) for i in range(len(ions))]
    labels = ions
    ax.legend(handles, labels, ncols=4 if lowz else 2, handlelength=1.3, columnspacing=0.8, loc="upper left")

    fig.savefig(
        "ion_redshift_inst_coverage_%s%s%s%s.pdf"
        % (sim.simName, "_single" if single else "", "_all" if all else "", "_lowz" if lowz else "")
    )
    plt.close(fig)


def paperPlots():
    """Generate plots for the mock spectra paper."""
    # fig 1: visual overview: full box TNG50 of N_ion, halo zoom, draw sightline going through
    if 0:
        sim = simParams("tng50-1", redshift=0.7)  # OVI at z=0.2 or NeVIII at z=0.7
        vis_overview(sim, haloID=None)  # full box
        vis_overview(sim, haloID=300)  # randomly selected, nice outflow features

        # single spectrum vis
        opts = {"num": None, "mode": "inds", "dv": True, "xlim": [-500, 500]}
        sim.setRedshift(2.0)
        spectra_gallery_indiv(sim, inds=[739610], ion="C IV", SNR=20, instrument="KECK-HIRES-B14", **opts)

        opts = {"num": None, "mode": "inds", "xlim": [1308, 1334]}  # [-2000,2000]}
        sim.setRedshift(0.7)
        spectra_gallery_indiv(sim, inds=[905323], ion="Ne VIII", SNR=20, instrument="COS-G130M", **opts)

    # fig 2: redshift coverage for transitions/instruments/configurations (given a sim)
    if 0:
        sim = simParams(run="tng50-1")
        # ion_redshift_coverage(sim, all=True, lowz=True)
        ion_redshift_coverage(sim, single=True, lowz=True)

    # fig 3: (dense) spectra galleries
    if 0:
        # C IV
        sim = simParams(run="tng50-1", redshift=2.0)
        inst = "SDSS-BOSS"  #'KECK-HIRES-B14'
        opts = {"ion": "C IV", "instrument": inst, "num": 121, "SNR": 50, "dv": True, "xlim": [-900, 900]}

        spectra_gallery_indiv(sim, EW_minmax=[0.1, 5.0], mode="evenly", style="grid", **opts)

    # fig 4: individual spectra galleries
    if 0:
        # Mg II
        sim = simParams(run="tng50-1", redshift=0.7)
        inst = "SDSS-BOSS"  #'4MOST-HRS'
        opts = {"ion": "Mg II", "instrument": inst, "num": 10, "solar": False, "SNR": 20}

        spectra_gallery_indiv(sim, EW_minmax=[0.1, 5.0], mode="evenly", **opts)
        spectra_gallery_indiv(sim, EW_minmax=[3.0, 6.0], mode="random", **opts)

    if 0:
        # C IV - fig 4
        sim = simParams(run="tng50-1", redshift=2.0)
        opts = {"ion": "C IV", "num": 10, "SNR": 50, "dv": True, "xlim": [-900, 900]}  # 'auto'

        for inst in ["SDSS-BOSS", "KECK-HIRES-B14"]:
            spectra_gallery_indiv(sim, EW_minmax=[0.1, 5.0], mode="evenly", instrument=inst, **opts)
            # spectra_gallery_indiv(sim, EW_minmax=[3.0, 6.0], mode='random', instrument=inst, **opts)

    if 0:
        # Ne VIII
        sim = simParams(run="tng50-1", redshift=0.7)
        opts = {"ion": "Ne VIII", "instrument": "COS-G130M", "num": 10, "dv": True, "xlim": [-900, 900], "SNR": 15}

        spectra_gallery_indiv(sim, EW_minmax=[0.1, 3.0], mode="evenly", **opts)
        spectra_gallery_indiv(sim, EW_minmax=[2.0, 3.5], mode="random", **opts)

    # fig 5: 2d spectra visualization
    if 0:
        # C IV
        sim = simParams(run="tng50-1", redshift=2.0)
        inst = "SDSS-BOSS"  #'KECK-HIRES-B14'
        opts = {"ion": "C IV", "instrument": inst, "num": None, "SNR": 50, "dv": True, "xlim": [-900, 900]}

        spectra_gallery_indiv(sim, EW_minmax=[0.1, 5.0], mode="all", style="2d", **opts)

    # fig 6: EW vs coldens vs CoG (CIV)
    if 0:
        sim = simParams("tng50-1", redshift=2.0)
        bvals = [5, 10, 25, 50]
        EW_vs_coldens(sim, line="CIV 1548", instrument="SDSS-BOSS")
        EW_vs_coldens(sim, line="MgII 2796", instrument="SDSS-BOSS", bvals=bvals, ylim=[-1.3, 1.3], xlim=[12.0, 19.0])
        # EW_vs_coldens(sim, line='MgII 2803', instrument='SDSS-BOSS', bvals=bvals, ylim=[-1.3,1.3], xlim=[12.0,19.0])
        # EW_vs_coldens(sim, line='HI 1215', instrument='SDSS-BOSS', bvals=bvals, ylim=[-1.3,2.5], xlim=[14.0,21.0])

    # fig 7: MgII EW distribution functions (dN/DW) and absorber incidence vs redshift (dN/dz) vs. data
    if 0:
        sim = simParams(run="tng50-1")
        line = "MgII 2796"
        inst = ["SDSS-BOSS", "XSHOOTER-NIR-04"]
        z1 = [0.3, 0.7, 1.0, 1.5, 2.0]
        z2 = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]
        indivEWs = False
        opts = {"xlim": [0, 6], "solar": False, "log": False}

        EW_distribution(sim, line=line, instrument=inst, redshifts=z1, indivEWs=False, **opts)
        dNdz_evolution(sim, line=line, instrument=inst, redshifts=z2, solar=opts["solar"])

    # fig 7: CIV EW distributions (dN/DW) and absorber incidence vs redshift (dN/dz) vs. data
    if 0:
        sim = simParams(run="tng50-1")
        line = "CIV 1548"
        inst = "SDSS-BOSS"
        redshifts = [1.5, 2.0, 3.0, 4.0, 5.0]
        indivEWs = False
        opts = {"xlim": [0.1, 2.5], "solar": False, "dX": True, "log": False}

        EW_distribution(sim, line=line, instrument=inst, redshifts=redshifts, indivEWs=indivEWs, **opts)
        dNdz_evolution(sim, line=line, instrument=inst, redshifts=redshifts, solar=opts["solar"])

    # fig 8: 2D optical depth map
    if 0:
        sim = simParams("tng50-1", redshift=2.0)
        inst = "KECK-HIRES-B14"  # 'KECK-LRIS-B-600' would better match KBSS

        optical_depth_map_2d(sim, line="CIV 1548", instrument=inst, tau_minmax=[0.0, 0.4])
        optical_depth_map_2d(sim, line="HI 1215", instrument=inst, tau_minmax=[0.0, 1.5])
        optical_depth_map_2d(sim, line="OI 1304", instrument=inst, tau_minmax=[0.0, 0.1])

    # fig 9: EW vs impact parameter, with example COS spectra
    if 0:
        sim = simParams("tng50-1", redshift=0.3)
        spec = ["MgII 2796", "MgII 2803", "FeI 3026", "NaI 5897"]
        impact_parameter_profile(sim, line="MgII 2796", instrument="SDSS-BOSS", spec=spec)

    if 0:
        sim = simParams("tng50-1", redshift=0.1)
        spec = ["NIII 990", "SiIII 1206", "NV 1238", "OVI 1031"]  # also have NII 1083, HI 1215
        impact_parameter_profile(sim, line="SiIII 1206", instrument="COS-G130M", spec=spec)

    # fig 10: example of a cosmological-distance (i.e. lightcone) spectrum
    if 0:
        sim = simParams(run="tng50-1")

        plotLightconeSpectrum(sim, instrument="KECK-HIRES-B14", ion="H I")
        plotLightconeSpectrum(sim, instrument="KECK-HIRES-B14", ion="C IV", SNR=100)

    # fig A: instrumental LSFs
    if 0:
        for inst in ["COS-G130M", "4MOST-HRS", "SDSS-BOSS", "DESI"]:
            instrument_lsf(inst)

    # fig X: doublet ratio
    if 0:
        sim = simParams(run="tng50-1", redshift=2.0)

        doublet_ratio(sim, line1="MgII 2796", line2="MgII 2803", instrument="SDSS-BOSS")
        doublet_ratio(sim, line1="CIV 1548", line2="CIV 1550", instrument="SDSS-BOSS")

    # fig X: mean transmitted LyA forest flux
    if 0:
        sim = simParams(run="tng50-1", redshift=2.0)
        mean_transmitted_flux(sim, line="HI 1215", instrument="SDSS-BOSS", redshifts=[2.0, 3.0, 4.0])
        mean_transmitted_flux(sim, line="HI 1215", instrument="KECK-HIRES-B14", redshifts=[2.0, 3.0, 4.0])

    # fig X: single spectrum plot (website/online API)
    if 0:
        sim = simParams(run="tng50-1")
        file = "spectra_TNG50-1_z2.0_n1000d2-fullbox_KECK-HIRES-B14_CIV_combined.hdf5"

        for _ in range(10):
            filepath = sim.postPath + "AbsorptionSpectra/" + file
            spectrum_plot_single(file=filepath, ind=None, pStyle="black")  # random index

    # fig X: 1000 random spectra plots (movie frames)
    if 0:
        sim = simParams(run="tng50-1")
        path = sim.postPath + "AbsorptionSpectra/"
        files = glob.glob(path + "spectra*_combined.hdf5")

        rng = np.random.default_rng(424242)
        for i in range(1000):
            filepath = rng.choice(files)
            savefile = "frame_%03d.png" % i
            if isfile(savefile):
                print(f" Frame {i:03d}: already exists, skipping.")
                continue
            spectrum_plot_single(file=filepath, ind=None, saveFilename=savefile, pStyle="black")  # random index

    # fig X: 2d stacked COS spectra visualization
    if 0:
        sim = simParams(run="tng50-1", redshift=0.1)
        ions = ["O VI", "N V", "N II", "N III", "Si III", "H I"]
        opts = {"ion": ions, "instrument": "COS-G130M", "EW_minmax": None, "num": None, "SNR": 10}

        spectra_gallery_indiv(sim, mode="inds", inds=np.arange(15000), style="2d", **opts)
        # spectra_gallery_indiv(sim, mode='all', style='2d', **opts)

    # table: transitions
    if 0:
        ions = list(dict.fromkeys([info["ion"] for line, info in lines.items()]))  # unique
        for ion in ions:
            s = ion + " & "
            for line, info in lines.items():
                if info["ion"] == ion:
                    s += line.split(" ")[1] + ", "
            print(s[:-2] + r" \\")


if __name__ == "__main__":
    paperPlots()
