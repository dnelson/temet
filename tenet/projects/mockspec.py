"""
One Billion Synthetic Absorption Sightlines (OBAS) project
The Billion Synthetic Absorption Spectra (BSAS) Project
The Billion Absorption Sightlines Project (BASP)
The Virtual Universe in Absorption (VUA): a billion synthetic absorption sightlines from cosmological hydrodynamical simulations
(in prep)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from os.path import isfile

from tenet.cosmo.spectrum import _spectra_filepath, lines
#from ..plot.general import plotParticleMedianVsSecondQuant, plotPhaseSpace2D
from tenet.plot.spectrum import spectra_gallery_indiv, EW_distribution
from tenet.plot.config import *

def metalAbundancesVsSolar(sim, ion='Mg II'):
    """ Diagnostic plot of how much various metal abundances actual vary vs. the solar abundance ratio. """
    from ..cosmo.cloudy import cloudyIon

    n_thresh = -8.0 # for second histogram
    nbins = 200
    minmax = [-11.0, -1.0] # abundance

    species = ion.split(' ')[0]

    # load
    abund = sim.gas('metals_%s' % species)
    abund = np.log10(abund)

    # only cells which contribute (non-negligible) absorption are relevant
    numdens = sim.gas('%s numdens' % ion)
    numdens = np.log10(numdens)

    # get solar abundance ratio (mass ratio, to total)
    cloudy = cloudyIon(sim, el=species, redshiftInterp=False)

    solar_abund = cloudy._solarMetalAbundanceMassRatio(species)
    solar_abund = np.log10(solar_abund)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('%s Abundance [ log ]' % species)
    ax.set_ylabel('PDF')
    ax.set_yscale('log')

    # global hist
    yy, xx = np.histogram(abund, bins=nbins, range=minmax, density=True)
    xx = xx[:-1] + (minmax[1]-minmax[0])/nbins/2

    #ax.hist(abund, bins=100, label='%s z=%.1f' % (sim.simName,sim.redshift))
    ax.plot(xx, yy, '-', lw=lw, label='%s (all gas)' % sim)

    # restricted hist
    yy, xx = np.histogram(abund[numdens > n_thresh], bins=nbins, range=minmax, density=True)
    xx = xx[:-1] + (minmax[1]-minmax[0])/nbins/2

    ax.plot(xx, yy, '-', lw=lw, label='%s ($n_{\\rm %s} > %.1f$)' % (sim,ion,n_thresh))

    # solar abundance value
    ax.plot([solar_abund, solar_abund], ax.get_ylim(), '-', color='black', lw=lw, alpha=0.8, label='solar')

    # finish plot
    ax.legend(loc='best')
    fig.savefig('abund_ratio_%s_%d_%s.pdf' % (sim.simName,sim.snap,species))
    plt.close(fig)

def lightconeSpectraConfig(sim, max_redshift=5.0):
    """ To create a cosmological sightline, i.e. over a significant pathlength much larger than the box size, 
    possible e.g. complete from z=0 to z=4, we need to combine available pathlengths as available at the 
    discrete simulation snapshots. Compute the available snapshots, and the number of pathlengths to 
    take from each.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      max_redshift (float): all spectra go from redshift 0.0 to max_redshift, i.e. of the background quasar.

    Return:
      a 3-tuple composed of

      - **snaps** (list[int]): the snapshot numbers which need to be used to cover the sightline.
      - **num_boxes** (:py:class:`~numpy.ndarray`[int]): the number of times a full-box pathlength 
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

        dl = (dl_final - dl_init) * 1000 # ckpc

        num = dl / sim.units.codeLengthToComovingKpc(sim.boxSize)

        print(f'{num = }')

    # for interpolation
    zz = np.linspace(0.0, max_redshift, 1000)
    ll = sim.units.redshiftToComovingDist(zz) * 1000 # ckpc

    # automatic spacing: decide snapshots to use
    snaps = sim.validSnapList(onlyFull=True)[::-1]
    redshifts = sim.snapNumToRedshift(snaps)

    w = np.where(redshifts <= max_redshift + 0.01)
    snaps = snaps[w]
    redshifts = redshifts[w]

    num_boxes = np.zeros(snaps.size, dtype='float32')

    # we take information from each snapshot until we reach the redshift/distance halfway to the next
    # e.g. z=0.05 between snap 99 (z=0) and snap 91 (z=0.1)
    redshifts_mid = np.hstack( (redshifts[0], (redshifts[1:] + redshifts[:-1]) / 2, redshifts[-1]) )

    # midpoints in cosmological distance between snapshots, i.e. the point where we switch to the
    # next snapshot. note: first value is special (z=0), and last value is special (max_redshift)
    dists_mid = sim.units.redshiftToComovingDist(redshifts_mid) * 1000 # ckpc

    z_init = []

    for i in range(snaps.size):
        #print(f'[{i:2d}] snap = {snaps[i]}, redshift = {redshifts_mid[i]:4.2f}, dist = {dists_mid[i]:10.2f}')
        dist_start = dists_mid[i]
        dist_stop = dists_mid[i+1]
        dl = dist_stop - dist_start

        num_rep = dl / sim.units.codeLengthToComovingKpc(sim.boxSize)

        # we have no partial sightlines, i.e. less than boxSize long, and cannot create them from existing spectra
        num_boxes[i] = np.round(num_rep)

        dStr = f'from [z = {redshifts_mid[i]:4.2f} D = {dists_mid[i]:9.1f}]'
        dStr += f' to [z = {redshifts_mid[i+1]:4.2f} D = {dists_mid[i+1]:9.1f}]'
        dStr += f' with [snap = {snaps[i]} at z = {redshifts[i]:4.2f}] dl = {dl:9.1f} N = {num_boxes[i]:4.1f}'

        print(dStr)

        # make list of the redshift at which each replication should begin
        dists_loc = np.arange(int(num_boxes[i]))
        dists_loc = dist_start + dists_loc * (dl / num_boxes[i])

        assert dists_loc[-1] < dist_stop

        dist_excess = dist_stop - (dists_loc[-1] + sim.units.codeLengthToComovingKpc(sim.boxSize))
        dists_loc += dist_excess/2 # split gap on both sides

        # convert distances into redshift
        dists_z = np.interp(dists_loc, ll, zz)

        #assert dists_z[0] > redshifts_mid[i] and dists_z[-1] < redshifts_mid[i+1] # need not be true

        z_init.append(dists_z) # list of lists, one per snapshot

        #print(' z = ' + ' '.join(['%.3f' % z for z in dists_z]))

    return snaps, num_boxes, z_init

def lightconeSpectra(sim, instrument, ion, solar=False, add_lines=None):
    """ Create a composite spectrum spanning a cosmological distance.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      instrument (str): specify observational instrument (from tenet.cosmo.spectrum.instruments).
      ion (str): space-separated name of ion e.g. 'Mg II'.
      solar (bool): if True, then adopt solar abundance ratio for the given species, instead of snap value.
      add_lines (list[str] or None): if not None, then a list of lines to include. otherwise, include all for this ion.

    Return:
      a 2-tuple composed of

      - **wave** (:py:class:`~numpy.ndarray`): 1d array, observed-frame wavelength grid [Ang].
      - **flux** (:py:class:`~numpy.ndarray`): 1d array, normalized flux values, from 0 to 1.
    """
    rng = np.random.default_rng(424242)

    # config
    projAxis = 2

    # get replication configuration
    snaps, num_boxes, z_inits = lightconeSpectraConfig(sim)

    # load metadata from first (available) snapshot
    for snap in snaps:
        sim.setSnap(snap)
        fname = _spectra_filepath(sim, projAxis, instrument, ion=ion, solar=solar)

        if isfile(fname):
            break

    with h5py.File(fname,'r') as f:
        master_wave = f['master_wave'][()]
        ray_total_dl = f['ray_total_dl'][()]
        num_spec = f['ray_pos'].shape[0]

    assert ray_total_dl == sim.boxSize # otherwise generalize lightconeSpectraConfig()

    # allocate
    tau_master = np.zeros(master_wave.size, dtype='float64')

    # loop over snapshots
    for snap, num_box, z_init in zip(snaps,num_boxes,z_inits):
        sim.setSnap(snap)
        print(f'[{snap = :3d}] at z = {sim.redshift:.2f}, num spec = {num_box}')

        # check existence
        fname = _spectra_filepath(sim, projAxis, instrument, ion=ion, solar=solar)

        if not isfile(fname):
            # (hopefully, no lines at the relevant wavelength range at this redshift)
            print(f' skip, does not exist.')
            continue

        # select N at random
        spec_inds = rng.integers(low=0, high=num_spec, size=int(num_box))

        # open file
        fname = _spectra_filepath(sim, projAxis, instrument, ion=ion, solar=solar)

        with h5py.File(fname,'r') as f:
            # load each spectrum individually, shift, and accumulate
            for spec_ind, z_local in zip(spec_inds,z_init):
                # allocate
                tau_local = np.zeros(master_wave.size, dtype='float64')

                # combine optical depth arrays for all transitions of this ion
                for key in f:
                    # skip unrelated non-tau datasets
                    if 'tau_' not in key:
                        continue

                    # skip if not among the specific lines requested, unless we are including all
                    if add_lines is not None and key.replace('tau_','') not in add_lines:
                        continue

                    # load entire tau array for one transition of this ion
                    print(f' [spec {spec_ind:5d}] at {z_local = :.3f} adding [{key}]')
                    tau_local += f[key][spec_ind,:]

                # shift in redshift according to the cumulative pathlength (z_init)
                wave_redshifted = master_wave * ((1 + z_local) / (1 + sim.redshift))

                # interpolate back onto the master (rest-frame) wavelength grid, and accumulate
                tau_redshifted = np.interp(master_wave, wave_redshifted, tau_local, left=0.0, right=0.0)

                tau_master += tau_redshifted

    # convert optical depth to flux
    flux = np.exp(-1*tau_master)

    return master_wave, flux

def plotLightconeSpectrum(sim, instrument, ion, add_lines=None):
    """ Plot a single lightcone spectrum.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      instrument (str): specify observational instrument (from tenet.cosmo.spectrum.instruments).
      ion (str): space-separated name of ion e.g. 'Mg II'.
      add_lines (list[str] or None): if not None, then a list of lines to include. otherwise, include all for this ion.
    """

    # generate, quick caching
    linesStr = '' if add_lines is None else ('_'+'-'.join(add_lines))
    cache_file = 'spec_cache_%s_%s_%s%s.hdf5' % (sim.simName,instrument,ion.replace(' ',''),linesStr)

    if isfile(cache_file):
        print(f'Loading from: [{cache_file}]')
        with h5py.File(cache_file, 'r') as f:
            wave = f['wave'][()]
            flux = f['flux'][()]
    else:
        # create now
        wave, flux = lightconeSpectra(sim, instrument, ion, add_lines=add_lines)

        with h5py.File(cache_file,'w') as f:
            f['wave'] = wave
            f['flux'] = flux
        print(f'Saved: [{cache_file}]')

    # plot
    fig = plt.figure(figsize=(figsize[0]*2,figsize[1]*1.5))
    #(ax_top, ax_top_zoom), (ax_bottom, ax_bottom_zoom) = fig.subplots(nrows=2, ncols=2)
    gs = GridSpec(2, 2, width_ratios=[2,1])
    ax_top, ax_top_zoom, ax_bottom, ax_bottom_zoom = [plt.subplot(spec) for spec in gs]

    # top panel: strong absorbers, down to saturation
    ax_top.set_ylim([-0.05,1.05])
    ax_top.set_xlim([wave.min(),wave.max()])
    ax_top.set_xlabel('Wavelength [ Ang ]')
    ax_top.set_ylabel('Continuum Normalized Flux')

    ax_top.step(wave, flux, '-', where='mid', lw=lw, label='%s %s' % (instrument,ion))
    ax_top.legend(loc='best')

    # top zoom panel (detail)
    ax_top_zoom.set_ylim([-0.05,1.05])
    ax_top_zoom.set_xlim([5550,5600])
    ax_top_zoom.set_xlabel('Wavelength [ Ang ]')

    ax_top_zoom.step(wave, flux, '-', where='mid', lw=lw)

    # bottom panel: weak absorbers, 1% from the continuum
    ax_bottom.set_ylim([0.95,1.001])
    ax_bottom.set_xlim([wave.min(),wave.max()])
    ax_bottom.set_xlabel('Wavelength [ Ang ]')
    ax_bottom.set_ylabel('Continuum Normalized Flux')

    ax_bottom.step(wave, flux, '-', where='mid', lw=lw)

    # debugging: (we have some small wavelength regions which are covered by no volume, due to 
    # requirement of sampling integer numbers of boxes -- not important for rare absorption, but
    # causes erroneous high flux spikes where we are absorption dominated e.g. high-z LyA forest)
    if 0:
        for z in [4.897,4.795,4.696,4.600,4.506]:
            z_obs = lines['LyA']['wave0'] * (1+z)
            ax_bottom.plot([z_obs,z_obs], ax_bottom.get_ylim(), '-', color='black')
        for z in [4.420,4.340,4.262,4.186,4.112,4.039,3.968,3.898,3.829,3.762,3.697,3.632,3.569,3.507]:
            z_obs = lines['LyA']['wave0'] * (1+z)
            ax_bottom.plot([z_obs,z_obs], ax_bottom.get_ylim(), '-', color='green')

    # bottom zoom panel (detail)
    ax_bottom_zoom.set_ylim([-0.01,0.20])
    ax_bottom_zoom.set_xlim([6000,6100])
    ax_bottom_zoom.set_xlabel('Wavelength [ Ang ]')

    ax_bottom_zoom.step(wave, flux, '-', where='mid', lw=lw)

    # finish
    fig.savefig('spectrum_lightcone_%s_%s_%s%s.pdf' % (sim.simName,ion.replace(' ',''),instrument,linesStr))
    plt.close(fig)

def paperPlots():
    from ..util.simParams import simParams

    # fig 1: individual spectra galleries
    if 0:
        # Mg II
        sim = simParams(run='tng50-1', redshift=0.5)
        inst = 'SDSS-BOSS' #'4MOST-HRS'
        solar = False
        num = 10

        opts = {'instrument':inst, 'num':num, 'solar':solar}
        spectra_gallery_indiv(sim, ion='Mg II', EW_minmax=[0.1, 5.0], mode='evenly', SNR=20, **opts)
        spectra_gallery_indiv(sim, ion='Mg II', EW_minmax=[0.01, 0.4], mode='evenly', SNR=50, **opts)
        spectra_gallery_indiv(sim, ion='Mg II', EW_minmax=[3.0, 6.0], mode='random', SNR=20, **opts)

    if 0:
        # C IV
        sim = simParams(run='tng50-1', redshift=2.0)
        inst = '4MOST-HRS'
        num = 10

        spectra_gallery_indiv(sim, ion='C IV', instrument=inst, EW_minmax=[0.1, 5.0], num=num, mode='evenly', SNR=20)
        spectra_gallery_indiv(sim, ion='C IV', instrument=inst, EW_minmax=[0.01, 0.4], num=num, mode='evenly', SNR=50)
        spectra_gallery_indiv(sim, ion='C IV', instrument=inst, EW_minmax=[3.0, 6.0], num=num, mode='random', SNR=20)

    if 0:
        # Fe II
        sim = simParams(run='tng50-1', redshift=2.0)
        inst = '4MOST-HRS'
        num = 10

        spectra_gallery_indiv(sim, ion='Fe II', instrument=inst, EW_minmax=[0.1, 5.0], num=num, mode='evenly', SNR=None)
        spectra_gallery_indiv(sim, ion='Fe II', instrument=inst, EW_minmax=[0.01, 0.4], num=num, mode='evenly', SNR=None)
        spectra_gallery_indiv(sim, ion='Fe II', instrument=inst, EW_minmax=[3.0, 6.0], num=num, mode='random', SNR=None)

    # fig 2: 2d spectra visualization
    if 0:
        pass

    # fig 3: EW distribution functions
    if 0:
        sim = simParams(run='tng50-1')
        line = 'MgII 2796'
        inst = 'SDSS-BOSS'
        redshifts = [0.5, 0.7, 1.0, 2.0]
        solar = False
        log = False

        EW_distribution(sim, line=line, instrument=inst, redshifts=redshifts, solar=solar, log=log)

    # fig X: abundances vs solar i.e. for mini-snaps
    if 0:
        sim = simParams(run='tng50-1', redshift=0.5)
        metalAbundancesVsSolar(sim, 'Mg II')

    # fig X: example of a cosmological-distance (i.e. lightcone) spectrum
    if 1:
        sim = simParams(run='tng50-1')

        plotLightconeSpectrum(sim, instrument='SDSS-BOSS', ion='Fe II')
        #plotLightconeSpectrum(sim, instrument='KECK-HIRES', ion='H I')
        #plotLightconeSpectrum(sim, instrument='KECK-HIRES', ion='H I')

    # fig X: Lyman-alpha forest sightlines (testing)
    if 0:
        sim = simParams(run='tng50-1', redshift=5.0)
        inst = 'KECK-HIRES'
        num = 10

        spectra_gallery_indiv(sim, ion='H I', instrument=inst, EW_minmax=None, num=num, mode='random', xlim='full', SNR=None)

    # fig X: EW_solar/EW_sim vs EW_sim test
    if 0:
        pass

    # fig X: redshift coverage for transitions, given an instrument
    if 0:
        pass # x-axis: redshift?

if __name__ == "__main__":
    paperPlots()
