"""
Diagnostic and production plots based on synthetic ray-traced absorption spectra.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile

from ..cosmo.spectrum import _line_params, _voigt_tau, _equiv_width, _spectra_filepath, lsf_matrix
from ..cosmo.spectrum import create_wavelength_grid, deposit_single_line, lines, instruments, absorber_catalog
from ..util.helper import logZeroNaN, sampleColorTable
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
    master_mid, master_edges, tau_master = create_wavelength_grid(line=line, instrument=None)

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
    master_mid, master_edges, tau_master = create_wavelength_grid(instrument=instrument)

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
    master_mid, master_edges, tau_master = create_wavelength_grid(line=line)

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

def instrument_lsf(instrument):
    """ Plot LSF(s) of a given instrument. For wavelength-dependent LSF matrices. """
    num = 6

    # get wavelength grid and wavelength-dependent LSF
    wave_mid, _, _ = create_wavelength_grid(instrument=instrument)
    lsf_mode, lsf, lsf_fwhm = lsf_matrix(instrument)

    print(f'{lsf_mode = }, {lsf.shape = }')

    # x-axis
    lsf_size = lsf.shape[1]
    cen_i = int(np.floor(lsf_size/2))

    xx = np.arange(lsf_size, dtype='int32') - cen_i

    # start plot
    fig, axes = plt.subplots(ncols=1, nrows=3, height_ratios=[1,1,0.5], figsize=(8,16.8))

    for ax in axes[0:2]:
        ax.set_xlabel('Pixel Number')
        ax.set_xticks(xx)
        ax.set_ylabel(f'{instrument} LSF')

        # first and second panels are identical, except second is y-log
        if ax == axes[1]: ax.set_yscale('log')

        # add a number of LSFs across the instrumental wavelength range
        for i in range(num):
            # evenly sample
            d_ind = lsf.shape[0] / num
            ind = int(i * d_ind + d_ind/2)

            lsf_kernel = lsf[ind,:]

            label = r'$\rm{\lambda = %.1f \AA}$' % wave_mid[ind]
            ax.plot(xx, lsf_kernel, 'o-', label=label)
        
        ax.plot([xx[cen_i],xx[cen_i]], [0, ax.get_ylim()[1]], '--', color='#ccc')

    # bottom panel: FWHM vs wave
    axes[-1].set_xlabel('Wavelength [ $\\rm{\AA}$ ]')
    axes[-1].set_ylabel('FWHM [ $\\rm{\AA}$ ]')
        
    axes[-1].plot(wave_mid, lsf_fwhm)

    # finish plot
    axes[0].legend(loc='upper right')
    fig.savefig('lsf_%s.pdf' % instrument)
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

def spectra_gallery_indiv(sim, ion='Mg II', instrument='4MOST-HRS', EW_minmax=[0.1,1.0], 
                          num=10, mode='random', inds=None, solar=False, SNR=None, xlim=None):
    """ Plot a gallery of individual absorption profiles within a given EW range.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      EW_minmax (list[float]): minimum and maximum EW to plot [Ang].
      num (int): how many individual spectra to show.
      mode (str): either 'random', 'evenly', or 'inds'.
      inds (list[int]): if mode is 'inds', then the list of specific spectra indices to plot. num is ignored.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
      SNR (float): if not None, then add noise to achieve this signal to noise ratio.
      xlim (str, list[float]): either 'full' or a 2-tuple of [min,max], or automatic if None (default)
    """
    assert mode in ['random','evenly','inds']
    if mode == 'inds': assert inds is not None
    if mode in ['random','evenly']: assert inds is None
    
    # config
    ctName = 'thermal'    

    # load
    filepath = _spectra_filepath(sim, ion, instrument=instrument, solar=solar)

    with h5py.File(filepath,'r') as f:
        # load metadata
        lineNames = f.attrs['lineNames']
        wave = f['wave'][()]

        # total EW (summing all transitions)
        EW = np.sum(np.vstack([f[key][()] for key in f.keys() if 'EW_' in key]), axis=0)

        # total EW (of a single transition)
        #EW = f['EW_MgII_2796'][()]

    # select
    if EW_minmax is not None:
        inds_all = np.where( (EW>EW_minmax[0]) & (EW<=EW_minmax[1]) )[0]
        print(f'[{ion}] Found [{len(inds_all)}] of [{EW.size}] spectra in EW range [{EW_minmax[0]}-{EW_minmax[1]}] Ang.')
    else:
        inds_all = np.arange(EW.size)
        print(f'[{ion}] Loaded [{len(inds_all)}] spectra, no EW range window.')

    rng = np.random.default_rng(4242+inds_all[0]+inds_all[-1])

    if mode == 'random':
        # randomly shuffle all spectra in the EW bin, then select num
        rng.shuffle(inds_all)
        inds = inds_all[0:num]

    if mode == 'evenly':
        # evenly sample across EW, selecting one spectrum in each of num equal bins
        binsize = (EW_minmax[1] - EW_minmax[0]) / num

        inds = []
        for i in range(num):
            w = np.where((EW>EW_minmax[0]+i*binsize) & (EW<=EW_minmax[0]+(i+1)*binsize))[0]
            rng.shuffle(w)

            inds.append(w[0])

    if mode == 'inds':
        num = len(inds)

    # partial load of selected spectra
    inds = np.sort(inds)

    with h5py.File(filepath,'r') as f:
        flux = f['flux'][inds,:]

    EW = EW[inds]

    # re-sort
    if mode == 'evenly':
        sort_inds = np.argsort(EW)
        flux = flux[sort_inds]
        EW = EW[sort_inds]

    # how many lines do we have? what is their span in wavelength?
    lines_wavemin = 0
    lines_wavemax = np.inf

    for line in lineNames:
        line = line.replace('_',' ')
        lines_wavemin = np.clip(lines_wavemin, lines[line]['wave0'], np.inf)
        lines_wavemax = np.clip(lines_wavemax, 0, lines[line]['wave0'])
        
    # determine wavelength (x-axis) bounds
    if str(xlim) == 'full':
        xlim = [np.min(wave), np.max(wave)]
    elif isinstance(xlim,list):
        # input directly
        pass
    else:
        # automatic
        xlim = [np.inf, 0]

        for i in range(num):
            w = np.where(flux[i,:] < 0.99)[0]
            w_min = wave[w].min()
            w_max = wave[w].max()

            if w_min < xlim[0]: xlim[0] = w_min
            if w_max > xlim[1]: xlim[1] = w_max

        dwave = (xlim[1] - xlim[0]) * 0.01
        xlim[0] = np.floor((xlim[0]-dwave)/10) * 10
        xlim[1] = np.ceil((xlim[1]+dwave)/10) * 10

    # determine flux (y-axis) bounds
    spacingFac = 1.0
    if EW_minmax is not None:
        if np.max(EW_minmax) <= 0.4:
            spacingFac = 0.5
        if np.max(EW_minmax) > 0.8:
            spacingFac = 1.2
    ylim = [-spacingFac/2, num*spacingFac + spacingFac/2]

    # add noise? ("signal" is now 1.0)
    if SNR is not None:
        noise = rng.normal(loc=0.0, scale=1/SNR, size=flux.shape)
        flux += noise
        # achieved SNR = 1/stddev(noise)
        flux = np.clip(flux, 0, np.inf) # clip negative values at zero

    # plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Relative Flux (+ constant offset)')

    ax.set_yticks(np.arange(num+1)*spacingFac)
    if spacingFac >= 1.0:
        ax.set_yticklabels(['%d' % i for i in range(num+1)])
    else:
        ax.set_yticklabels(['%.1f' % (i*spacingFac) for i in range(num+1)])

    colors = sampleColorTable(ctName, num, bounds=[0.0, 0.9])

    for i in range(num):
        # vertical offset by 1.0 for each spectrum
        y_offset = (i+1)*spacingFac - 1
        ax.step(wave, flux[i,:]+y_offset, '-', color=colors[i], where='mid', lw=lw)

        # label
        text_x = xlim[0] + (xlim[1]-xlim[0])/100
        text_y = y_offset + 1.0 - (num/50) * spacingFac
        if SNR is not None: text_y -= (num/50) * (5/SNR)
        label = 'EW = %.2f$\AA$' % EW[i]

        ax.text(text_x, text_y, label, color=colors[i], alpha=0.6, ha='left', va='top')

    # finish plot
    label = r'%s ($\rm{z \simeq %.1f}$)' % (ion,sim.redshift)
    ax.legend([plt.Line2D((0,1),(0,0),lw=0,marker='')], [label], loc='lower right')

    snrStr = '_snr%d' % SNR if SNR is not None else ''
    ewStr = '_%.1f-%.1f' % (EW_minmax[0],EW_minmax[1]) if EW_minmax is not None else ''
    fig.savefig('spectra_%s_%d_%s_%s%s_%d-%s%s.pdf' % \
        (sim.simName,sim.snap,ion.replace(' ',''),instrument,ewStr,num,mode,snrStr))
    plt.close(fig)

def EW_distribution(sim_in, line='MgII 2796', instrument='SDSS-BOSS', redshifts=[0.5,0.7,1.0], 
                    solar=False, indivEWs=False, log=False):
    """ Plot the EW distribution (dN/dWdz) of a given absorption line.

    Args:
      sim_in (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      redshifts (list[float]): list of redshifts to overplot.
      solar (bool): use the (constant) solar value instead of simulation-tracked metal abundances.
      indivEWs (bool): if True, then use/create absorber catalog, to handle multiple absorbers per sightline, 
        otherwise use the available 'global' EWs, one per sightline.
      log (bool): plot log(EW) instead of linear EWs.
    """
    sim = sim_in.copy()

    # plot config
    EW_min = 1e-3 # rest-frame ang

    xlim = [0, 8] # ang
    if log: xlim = [-1.0, 1.4] # log[ang]
    nBins = 80

    # load: loop over requested redshifts
    EWs = {}

    for redshift in redshifts:
        sim.setRedshift(redshift)
        print('EW distribution: ', sim, line)

        ion = lines[line]['ion']
        filepath = _spectra_filepath(sim, ion, instrument=instrument, solar=solar)

        # raw EWs (one per sightline), or re-processed EWs (one per individual absorber)?
        if indivEWs:
            EWs_orig, _, EWs_processed, counts_processed, _ = absorber_catalog(sim, ion, instrument=instrument, solar=solar)
            data = EWs_processed[line]
            count = EWs_orig[line].size
        else:
            with h5py.File(filepath,'r') as f:
                count = f.attrs['count']
                data = f['EW_%s' % line.replace(' ','_')][()]

        # exclude absolute zero EWs (i.e. no absorption)
        data = data[data > 0]

        # convert to rest-frame
        data /= (1+sim.redshift)

        # exclude unobservably small EWs
        data = data[data >= EW_min]

        EWs[redshift] = data

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_xlabel('Rest-frame Equivalent Width [ %s$\\rm{\AA}$ ]' % ('Log ' if log else ''))
    ax.set_ylabel('d$^2 N$/d$z$d$W$ (%s)' % line)
    ax.set_yscale('log')

    # loop over requested redshifts
    for redshift in redshifts:
        # load
        sim.setRedshift(redshift)
        x = EWs[redshift]

        if log: x = np.log10(x)

        # histogram
        hh, bin_edges = np.histogram(x, bins=nBins, range=xlim)

        # normalize by dz = total redshift path length = N_sightlines * boxSizeInDeltaRedshift
        hh = hh.astype('float32') / (count * sim.dz)

        # normalize by dW = equivalent width bin sizes [Ang]
        dW_norm = bin_edges[1:] - bin_edges[:-1] # constant (linear)
        if log: dW_norm = 10.0**bin_edges[1:] - 10.0**bin_edges[:-1] # variable (log)

        hh /= dW_norm

        ax.stairs(hh, edges=bin_edges, lw=lw, label='z = %.1f' % sim.redshift)

    # plot obs data
    if line == 'MgII 2796':
        # (Matejek+12 Table 3)
        # https://ui.adsabs.harvard.edu/abs/2012ApJ...761..112M/abstract
        # (!) see also https://ui.adsabs.harvard.edu/abs/2013ApJ...764....9M/abstract
        m13_x = [0.42, 0.94, 1.52, 2.11, 2.70, 4.34]
        m13_x_lower = [0.05, 0.64, 1.23, 1.82, 2.41, 3.00]
        m13_x_upper = [0.64, 1.23, 1.82, 2.41, 3.00, 5.68]
        m13_y = [1.570, 0.594, 0.291, 0.187, 0.083, 0.027]
        m13_yerr = [0.272, 0.119, 0.080, 0.064, 0.042, 0.011]
        m13_label = 'Matejek+12 (1.9 < z < 6.3)'

        xerr = np.vstack( (np.array(m13_x) - m13_x_lower,np.array(m13_x_upper) - m13_x) )

        opts = {'color':'#333333', 'ecolor':'#333333', 'alpha':0.6, 'capsize':0.0, 'fmt':'s'}
        ax.errorbar(m13_x, m13_y, yerr=m13_yerr, xerr=xerr, label=m13_label, **opts)

        # Chen+16 (updated/finished sample of Matejek+, identical EW bins)
        # https://ui.adsabs.harvard.edu/abs/2017ApJ...850..188C/abstract
        c16_y = [1.539, 0.591, 0.298, 0.185, 0.134, 0.026]
        c16_yerr = [0.215, 0.082, 0.055, 0.042, 0.035, 0.007]
        c16_label = 'Chen+16 (1.9 < z < 6.3)'

        opts = {'color':'#333333', 'ecolor':'#333333', 'alpha':0.9, 'capsize':0.0, 'fmt':'D'}
        ax.errorbar(m13_x, c16_y, yerr=c16_yerr, xerr=xerr, label=c16_label, **opts)

    # check: https://iopscience.iop.org/article/10.3847/1538-4357/abbb34/pdf

    # finish plot
    ax.legend(loc='best')
    fig.savefig('EW_histogram_%s_%s%s_%s.pdf' % (sim.simName,line.replace(' ','-'),'_log' if log else '',instrument))
    plt.close(fig)

def dNdz_evolution(sim_in, redshifts, line='MgII 2796', instrument='SDSS-BOSS', solar=False):
    """ Plot the redshift evolution (i.e. dN/dz) and comoving incidence rate (dN/dX) of a given absorption line.

    Args:
      sim_in (:py:class:`~util.simParams`): simulation instance.
      line (str): string specifying the line transition.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      redshifts (list[float]): list of redshifts to overplot.
      solar (bool): use the (constant) solar value instead of simulation-tracked metal abundances.
      log (bool): plot log(EW) instead of linear EWs.
    """
    from ..load.data import zhu13mgii

    sim = sim_in.copy()

    # config
    z13 = zhu13mgii()
    #EW_thresholds = [0.3,1.0,3.0] # thresholds for EW for vs. redshift plot
    EW_thresholds = z13['EW0'] # match to obs data

    xlim = [0.0, np.max(redshifts)+1]
    ylim = [8e-4, 4.0]

    # load: loop over all available redshifts
    zz = []
    dNdz = {thresh:[] for thresh in EW_thresholds}
    dNdX = {thresh:[] for thresh in EW_thresholds}

    for redshift in redshifts:
        sim.setRedshift(redshift)
        ion = lines[line]['ion']
        filepath = _spectra_filepath(sim, ion, instrument=instrument, solar=solar)

        if not isfile(filepath):
            continue

        with h5py.File(filepath,'r') as f:
            count = f.attrs['count']
            EWs = f['EW_%s' % line.replace(' ','_')][()]

        # convert to rest-frame and store
        EWs /= (1+sim.redshift)

        # loop over requested thresholds
        for EW_thresh in EW_thresholds:
            num = len(np.where(EWs >= EW_thresh)[0])

            # normalize by dz = total redshift path length = N_sightlines * boxSizeInDeltaRedshift
            num_dz = float(num) / (count * sim.dz)

            # normalize by dX = total comoving path length
            num_dX = float(num) / (count * sim.dX)

            # store
            dNdz[EW_thresh].append(num_dz)
            dNdX[EW_thresh].append(num_dX)

        zz.append(redshift)

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('d$N$/d$z$ (%s)' % line)
    ax.set_yscale('log')

    # plot the simulation dN/dz for each EW threshold
    colors = []
    for EW_thresh in EW_thresholds:
        l, = ax.plot(zz, dNdz[EW_thresh], '-', lw=lw, label='EW > %.1f$\,\\rm{\AA}$' % EW_thresh)
        colors.append(l.get_color())

    # observational data
    if line == 'MgII 2796':     
        for i, EW0 in enumerate(z13['EW0']):
            label = r'%s ($\rm{W_0 > %.1f \AA}$)' % (z13['label'],EW0)
            typical_error = 1e-3
            ax.errorbar(z13['z'], z13['dNdz'][EW0], yerr=typical_error, 
              color=colors[i], alpha=0.8, marker='s', linestyle='none', label=label)

        # Zou+21 (z>2)
        # https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.1023C/abstract

    if line == 'CIV': # also SiIV
        pass # https://ui.adsabs.harvard.edu/abs/2022ApJ...924...12H/abstract
        # https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4940C/abstract

    # finish plot
    ax.legend(loc='best')
    fig.savefig('dNdz_evolution_%s_%s.pdf' % (sim.simName,line.replace(' ','-')))
    plt.close(fig)

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('Redshift')
    ax.set_ylabel('d$N$/d$X$ (%s)' % line)
    ax.set_yscale('log')

    for EW_thresh in EW_thresholds:
        ax.plot(zz, dNdX[EW_thresh], '-', label='EW > %.1f$\,\\rm{\AA}$' % EW_thresh)

    # observational data
    if line == 'MgII 2796':
        pass
        # Zou+21
        # https://ui.adsabs.harvard.edu/abs/2017MNRAS.472.1023C/abstract

    if line == 'CIV': # also SiIV
        pass # https://ui.adsabs.harvard.edu/abs/2022ApJ...924...12H/abstract
        # https://ui.adsabs.harvard.edu/abs/2018MNRAS.481.4940C/abstract

    # finish plot
    ax.legend(loc='best')
    fig.savefig('dNdX_evolution_%s_%s.pdf' % (sim.simName,line.replace(' ','-')))
    plt.close(fig)

def n_cloud_distribution(sim, ion='Mg II', redshifts=[0.5,0.7]):
    """ Plot the N_cloud distribution of a given ion based on a rays statistics file.

    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      redshifts (list[float]): list of redshifts to overplot.
    """
    # config
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
        saveFilename = _spectra_filepath(sim, ion).replace('integral_','stats_').replace('_combined','')

        with h5py.File(saveFilename,'r') as f:
            n_clouds = f['n_clouds'][()]

        # histogram
        w = np.where(n_clouds > 0)
        print(f'At z = {redshift:0.1f} [{len(w[0])}] of [{n_clouds.size}] sightlines have at least one cloud.')
        hh, bin_edges = np.histogram(n_clouds[w], bins=nBins, range=xlim)

        ax.stairs(hh, edges=bin_edges, lw=lw, label='z = %.1f' % redshift)

    # finish plot
    ax.legend(loc='best')
    fig.savefig('N_clouds_histogram_%s.pdf' % ion.replace(' ','_'))
    plt.close(fig)

def n_clouds_vs_EW(sim):
    """ Plot relationship between number of discrete clouds along a sightline, and the EW of the transition.
    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
    """

    # config
    ion = 'Mg II'
    instrument = '4MOST-HRS'

    xlog = False

    # load n_clouds
    saveFilename = _spectra_filepath(sim, ion).replace('integral_','stats_').replace('_combined','')

    with h5py.File(saveFilename,'r') as f:
        n_clouds = f['n_clouds'][()]

    # load EWs
    filepath = _spectra_filepath(sim, ion, instrument=instrument)

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
    fig.savefig('N_clouds_vs_EW_%s.pdf' % ion.replace(' ','_'))
    plt.close(fig)
