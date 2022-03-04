"""
Diagnostic and production plots based on synthetic ray-traced absorption spectra.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt

from ..cosmo.spectrum import _line_params, _voigt_tau, _equiv_width, _spectra_filepath
from ..cosmo.spectrum import create_master_grid, deposit_single_line, lines
from ..util.helper import logZeroNaN
from ..util import units
from ..plot.config import *

def curve_of_growth(line='MgII 2803'):
    """ Plot relationship between EW and N for a given transition.

    Args:
      line (str): name of transition, e.g. 'MgII 2803' or 'LyA'.
    """
    f, gamma, wave0_ang, _, _ = _line_params(line)

    # run config
    nPts = 201
    wave_ang = np.linspace(wave0_ang-5, wave0_ang+5, nPts)
    dvel = (wave_ang/wave0_ang - 1) * units.c_cgs / 1e5 # cm/s -> km/s

    #dwave_ang = wave_ang[1] - wave_ang[0]
    #wave_edges_ang = np.hstack(((wave_ang - dwave_ang/2),(wave_ang[-1] + dwave_ang/2)))
    
    # run test
    #N = 21.0 # log 1/cm^2
    #b = 30.0 # km/s
    #tau = _voigt_tau(wave_ang/1e8, N, b, wave0_ang, f, gamma)
    #flux = np.exp(-1*tau)

    # plot flux
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Velocity Offset [ km/s ]')
    ax.set_ylabel('Relative Flux')

    for N in [12,13,14,15,18,19,20]:
        for j, sigma in enumerate([30]):
            b = sigma * np.sqrt(2)
            tau = _voigt_tau(wave_ang, 10.0**N, b, wave0_ang, f, gamma)
            flux = np.exp(-1*tau)
            EW = _equiv_width(tau,wave_ang)
            print(N,b,EW)

            ax.plot(dvel, flux, lw=lw, linestyle=linestyles[j], label='N = %.1f b = %d' % (N,b))

    ax.legend(loc='best')
    fig.savefig('flux_%s.pdf' % line)
    plt.close(fig)

    # plot cog
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Column Density [ log cm$^{-2}$ ]')
    ax.set_ylabel('Equivalent Width [ $\AA$ ]')
    ax.set_yscale('log')
    #ax.set_ylim([0.01,10])

    cols = np.linspace(12.0, 18.5, 100)
    bvals = [3,5,10,15]

    for b in bvals: # doppler parameter, km/s
        # draw EW targets
        xx = [cols.min(), cols.max()]
        ax.plot(xx, [0.4,0.4], '-', color='#444444', alpha=0.4)
        ax.plot(xx, [1.0,1.0], '-', color='#444444', alpha=0.4)

        ax.fill_between(xx, [0.3,0.3], [0.5,0.5], color='#444444', alpha=0.05)
        ax.fill_between(xx, [0.9,0.9], [1.1,1.1], color='#444444', alpha=0.05)

        # derive EWs as a function of column density
        EW = np.zeros( cols.size, dtype='float32')
        for i, col in enumerate(cols):
            tau = _voigt_tau(wave_ang, 10.0**col, b, wave0_ang, f, gamma)
            EW[i] = _equiv_width(tau,wave_ang)

        ax.plot(cols, EW, lw=lw, label='b = %d km/s' % b)

    ax.legend(loc='best')
    fig.savefig('cog_%s.pdf' % line)
    plt.close(fig)

def profile_single_line():
    """ Voigt profile deposition of a single absorption line: create spectrum and plot. """

    # transition, instrument, and spectrum type
    line = 'LyA'
    instrument = None
    
    # config for 'this cell'
    N = 15.0 # log 1/cm^2
    b = 40.0 # km/s

    vel_los = 0.0 #1000.0 # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(line=line, instrument=None)

    # deposit
    f, gamma, wave0, _, _ = _line_params(line)

    z_doppler = vel_los / units.c_km_s
    z_eff = (1+z_doppler)*(1+z_cosmo) - 1 # effective redshift

    wave_local, tau_local, flux_local = deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff, debug=True)

    # compute flux
    flux_master = np.exp(-1*tau_master)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(211)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Relative Flux')
    ax.plot(master_mid, flux_master, 'o-', lw=lw, label='method A')
    ax.plot(wave_local, flux_local, '-', lw=lw, label='local')

    ax.legend(loc='best')
    ax = fig.add_subplot(212)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Optical Depth $\\tau$')
    ax.plot(master_mid, tau_master, 'o-', lw=lw, label='method A')
    ax.plot(wave_local, tau_local, '-', lw=lw, label='local')

    ax.legend(loc='best')
    fig.savefig('spectrum_single_%s.pdf' % line)
    plt.close(fig)

def profiles_multiple_lines():
    """ Deposit Voigt absorption profiles for a number of transitions: create spectrum and plot. """

    # transition, instrument, and spectrum type
    lineNames = ['LyA'] + [line for line in lines.keys() if 'HI ' in line] # Lyman series
    instrument = 'test_EUV'
    
    # config for 'this cell'
    N = 15.0 # log 1/cm^2
    b = 40.0 # km/s

    vel_los = 0.0 #1000.0 # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)

    # deposit
    z_doppler = vel_los / units.c_km_s
    z_eff = (1+z_doppler)*(1+z_cosmo) - 1 # effective redshift

    for line in lineNames:
        f, gamma, wave0, _, _ = _line_params(line)

        deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff)

    # compute flux
    flux_master = np.exp(-1*tau_master)

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(211)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Relative Flux')
    label = f'{N = :.1f} cm$^{{-2}}$, {b = :.1f} km/s'
    ax.plot(master_mid, flux_master, '-', lw=lw, label=label)

    ax.legend(loc='best')
    ax = fig.add_subplot(212)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Optical Depth $\\tau$')
    ax.plot(master_mid, tau_master, '-', lw=lw, label=label)

    ax.legend(loc='best')
    fig.savefig('spectrum_multi_%s.pdf' % ('-'.join(lineNames)))
    plt.close(fig)

def LyA_profiles_vs_coldens():
    """ Reproduce Hummels+17 Figure 10 of LyA absorption profiles for various N_HI values. """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm
    from ..util.helper import sampleColorTable, loadColorTable

    line = 'LyA'
    
    # config for 'this cell'
    N_vals = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20] # log 1/cm^2
    b = 22.0 # km/s

    vel_los = 0.0
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(line=line)

    # setup
    z_doppler = vel_los / units.c_km_s
    z_eff = (1+z_doppler)*(1+z_cosmo) - 1 # effective redshift

    f, gamma, wave0, _, _ = _line_params(line)

    # start plot
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot()

    ax.set_ylabel('Relative Flux')
    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([wave0-1.0, wave0+1.0])

    # top x-axis (v/c = dwave/wave)
    ax2 = ax.twiny()
    ax2.set_xlabel('$\Delta$v [ km/s ]')

    dwave = np.array(ax.get_xlim()) - wave0 # ang
    dv = units.c_km_s * (dwave / wave0)

    ax2.set_xlim(dv)

    # colors
    cmap = loadColorTable('viridis')
    bounds = [N-0.5 for N in N_vals] + [N_vals[-1] + 0.5]
    norm = BoundaryNorm(bounds, cmap.N)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    # loop over N values, compute a local spectrum for each and plot
    for i, N in enumerate(N_vals):
        wave, tau, flux = deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N, b, z_eff, debug=True)

        # plot
        ax.plot(wave, flux, '-', lw=lw, color=sm.to_rgba(N))

    # finish plot
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.1)
    cb = plt.colorbar(sm, cax=cax, ticks=N_vals)
    cb.ax.set_ylabel('log N$_{\\rm HI}$ [ cm$^{-2}$ ]')

    fig.savefig('LyA_absflux_vs_coldens.pdf')
    plt.close(fig)

def _spectrum_debug_plot(line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos):
    """ Plot some quick diagnostic panels for spectra. """

    # calculate derivative quantities
    flux_master = np.exp(-1*tau_master)
    master_dl = np.cumsum(master_dx)

    # start
    fig = plt.figure(figsize=(26,14))

    ax = fig.add_subplot(321) # upper left
    w = np.where(tau_master > 0)[0]
    xminmax = master_mid[ [w[0]-5, w[-1]+5] ]
    xcen = np.mean(xminmax)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_xlim([xminmax[0], xminmax[1]]) # [xcen-25,xcen+25]
    ax.set_ylabel('Relative Flux')
    ax.plot(master_mid, flux_master, '-', lw=lw, label=line)
    ax.legend(loc='best')

    ax = fig.add_subplot(322) # upper right
    ax.set_xlabel('Distance Along Ray [ Mpc ]')
    ax.set_ylabel('Density [log cm$^{-3}$]')
    ax.plot(master_dl, logZeroNaN(master_dens), '-', lw=lw)

    ax = fig.add_subplot(323) # center left
    ax.set_xlabel('$\Delta$ v [ km/s ]')
    ax.set_xlim([-300, 300])
    ax.set_ylabel('Relative Flux')
    dv = (master_mid-xcen)/xcen * units.c_km_s
    ax.plot(dv, flux_master, '-', lw=lw, label=line)

    ax = fig.add_subplot(324) # center right
    ax.set_xlabel('Distance Along Ray [ Mpc ]')
    ax.set_ylabel('Column Density [log cm$^{-2}$]')
    ax.plot(master_dl, logZeroNaN(master_dens*master_dx*units.Mpc_in_cm), '-', lw=lw)

    ax = fig.add_subplot(325) # lower left
    ax.set_xlabel('Distance Along Ray [ Mpc ]')
    ax.set_ylabel('Line-of-sight Velocity [ km/s ]')
    ax.plot(master_dl, master_vellos, '-', lw=lw)
    #ax.set_ylabel('Wavelength Deposited [ Ang ]')
    #ax.plot(master_dl, master_towave, '-', lw=lw)

    ax = fig.add_subplot(326) # lower right
    ax.set_xlabel('Distance Along Ray [ Mpc ]')
    ax.set_ylabel('Temperature [ log K ]')
    ax.plot(master_dl, logZeroNaN(master_temp), '-', lw=lw)

    fig.savefig(plotName)
    plt.close(fig)

def concat_spectra_gallery(sim):
    """ Plot a gallery of absorption profiles within a given EW range based on a concatenated spectra file.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
    """

    # config data
    projAxis = 2
    instrument = '4MOST_HRS'
    lineNames = ['MgII 2796','MgII 2803']

    # config plot
    EW_min = 0.6
    EW_max = 0.7
    SNR = None #30.0 # if not None, add Gaussian noise for the given signal-to-noise ratio
    num = 10

    wave_minmax = [4750,4800] # z=0.5: [4180,4250]

    # load
    filepath = _spectra_filepath(sim, projAxis, instrument, lineNames)

    with h5py.File(filepath,'r') as f:
        flux = f['flux'][()]
        EW = f['EW_total'][()]
        wave = f['master_wave'][()]

    # select
    inds = np.where( (EW>EW_min) & (EW<=EW_max) )[0]
    print(f'Found [{len(inds)}] of [{EW.size}] spectra within EW range [{EW_min}-{EW_max}] Ang.')

    rng = np.random.default_rng(4242+inds[0]+inds[-1])
    rng.shuffle(inds)

    flux = flux[inds,:]
    EW = EW[inds]

    # add noise?
    if SNR is not None:
        noise = rng.normal(loc=0.0, scale=1/SNR, size=flux.shape)
        flux += noise
        # achieved SNR = 1/stddev(noise)
        flux = np.clip(flux, 0, np.inf) # clip negative values at zero

    # plot
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    ax.set_xlim(wave_minmax)
    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Relative Flux')

    for i in range(num):
        ax.step(wave, flux[i,:], '-', where='mid', lw=lw, label='EW = %.1f$\AA$' % EW[i])

    ax.legend(loc='best')
    fig.savefig('spectra_%.1f-%.1f.pdf' % (EW_min,EW_max))
    plt.close(fig)

def EW_distribution(sim_in, redshifts=[0.5,0.7,1.0], log=False):
    """ Plot the EW distribution of a given line based on a concatenated spectra file.

    Args:
      sim_in (:py:class:`~util.simParams`): simulation instance.
      redshifts (list[float]): list of redshifts to overplot.
      log (bool): plot log(EW) instead of linear EWs.
    """
    sim = sim_in.copy()

    # config
    projAxis = 2
    instrument = '4MOST_HRS'
    lineNames = ['MgII 2796','MgII 2803']

    xlim = [0, 12] # ang
    if log: xlim = [-1.0, 1.4] # log[ang]
    nBins = 40

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_xlabel('Equivalent Width [ %sAng ]' % ('Log ' if log else ''))
    ax.set_ylabel('PDF')
    ax.set_yscale('log')

    # loop over requested redshifts
    for redshift in redshifts:
        # load
        sim.setRedshift(redshift)
        filepath = _spectra_filepath(sim, projAxis, instrument, lineNames)

        EWs = {}

        with h5py.File(filepath,'r') as f:
            for key in f:
                if 'EW_' in key:
                    EWs[key] = f[key][()]

        # histogram
        x = EWs['EW_total']
        if log: x = np.log10(x)
        hh, bin_edges = np.histogram(x, bins=nBins, range=xlim)

        ax.stairs(hh, edges=bin_edges, lw=lw, label='z = %.1f' % sim.redshift)

    # finish plot
    ax.legend(loc='best')
    fig.savefig('EW_histogram_%s%s.pdf' % \
        ('-'.join([line.replace(' ','_') for line in lineNames]),'_log' if log else ''))
    plt.close(fig)

def n_cloud_distribution(sim, redshifts=[0.5,0.7]):
    """ Plot the N_cloud distribution of a given ion based on a rays statistics file.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      redshifts (list[float]): list of redshifts to overplot.
    """

    # config
    projAxis = 2
    ionName = 'Mg II'

    xlim = [0, 20]
    nBins = xlim[1]

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_xlabel('Number of Clouds per Sightline')
    ax.set_ylabel('PDF')
    ax.set_yscale('log')

    # loop over requested redshifts
    for redshift in redshifts:
        # load
        saveFilename = sim.derivPath + 'rays/stats_%s_z%.1f_%d_%s.hdf5' % \
          (sim.simName,redshift,projAxis,ionName.replace(' ','_'))

        with h5py.File(saveFilename,'r') as f:
            n_clouds = f['n_clouds'][()]

        # histogram
        w = np.where(n_clouds > 0)
        print(f'At z = {redshift:0.1f} [{len(w[0])}] of [{n_clouds.size}] sightlines have at least one cloud.')
        hh, bin_edges = np.histogram(n_clouds[w], bins=nBins, range=xlim)

        ax.stairs(hh, edges=bin_edges, lw=lw, label='z = %.1f' % redshift)

    # finish plot
    ax.legend(loc='best')
    fig.savefig('N_clouds_histogram_%s.pdf' % ionName.replace(' ','_'))
    plt.close(fig)

def n_clouds_vs_EW(sim):
    """ Plot relationship between number of discrete clouds along a sightline, and the EW of the transition.
    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
    """

    # config
    projAxis = 2
    ionName = 'Mg II' # n_cloud
    lineNames = ['MgII 2796','MgII 2803'] # EW
    instrument = '4MOST_HRS' # EW

    xlog = False

    # load n_clouds
    saveFilename = sim.derivPath + 'rays/stats_%s_z%.1f_%d_%s.hdf5' % \
      (sim.simName,sim.redshift,projAxis,ionName.replace(' ','_'))

    with h5py.File(saveFilename,'r') as f:
        n_clouds = f['n_clouds'][()]

    # load EWs
    filepath = _spectra_filepath(sim, projAxis, instrument, lineNames)

    EWs = {}
    inds = {}

    with h5py.File(filepath,'r') as f:
        for key in f:
            if 'EW_' in key:
                EWs[key] = f[key][()]
        for pSplitInd in f['inds']:
            inds[pSplitInd] = f['inds'][pSplitInd][()]

    # create subset of n_clouds matching measured EWs (concatenated spectra above threshold only)
    n_clouds = n_clouds[inds['global']]

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim([1e-2,1e1] if xlog else [0,10])
    ax.set_xlabel('Equivalent Width [ Ang ]')
    ax.set_ylabel('Number of Clouds per Sightline')
    if xlog: ax.set_xscale('log')

    ax.set_rasterization_zorder(1) # elements below z=1 are rasterized

    ax.scatter(EWs['EW_total'], n_clouds, marker='.', zorder=0, label='z = %.1f' % sim.redshift)

    # finish plot
    ax.legend(loc='upper right')
    fig.savefig('N_clouds_vs_EW_%s.pdf' % ionName.replace(' ','_'))
    plt.close(fig)
