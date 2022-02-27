"""
Synthetic absorption spectra generation.
Inspired by SpecWizard (Schaye), pygad (Rottgers), Trident (Hummels).
"""
import numpy as np
import h5py
import glob
import threading
from os.path import isfile, isdir, expanduser
from os import mkdir
import matplotlib.pyplot as plt
from scipy.special import wofz

from numba import jit
from numba.extending import get_cython_function_address
import ctypes

from ..plot.config import *
from ..util import units
from ..util.helper import logZeroNaN, closest, pSplitRange
from ..util.voronoiRay import trace_ray_through_voronoi_mesh_treebased, \
  trace_ray_through_voronoi_mesh_with_connectivity, rayTrace

# line data (mostly AtomDB)
# f [dimensionless]
# gamma [1/s], where tau=1/gamma is the ~lifetime (is the sum of A)
# wave0 [ang]
lines = {'LyA'        : {'f':0.4164,   'gamma':4.49e8,  'wave0':1215.670,  'ion':'H I'},
         'HI 1024'    : {'f':0.0791,   'gamma':1.897e8, 'wave0':1025.7223, 'ion':'H I'},
         'HI 973'     : {'f':0.0290,   'gamma':1.28e7,  'wave0':972.5367,  'ion':'H I'},
         'HI 950'     : {'f':1.395e-2, 'gamma':4.12e6,  'wave0':949.7430,  'ion':'H I'},
         'HI 938'     : {'f':7.803e-3, 'gamma':2.45e6,  'wave0':937.8034,  'ion':'H I'},
         'HI 931'     : {'f':4.814e-3, 'gamma':7.56e5,  'wave0':930.7482,  'ion':'H I'},
         'HI 926'     : {'f':3.183e-3, 'gamma':3.87e5,  'wave0':926.22564, 'ion':'H I'},
         'HI 923'     : {'f':2.216e-3, 'gamma':2.14e5,  'wave0':923.1503,  'ion':'H I'},
         'HI 921'     : {'f':1.605e-3, 'gamma':1.26e5,  'wave0':920.9630,  'ion':'H I'},
         'HI 919'     : {'f':1.20e-3,  'gamma':7.83e4,  'wave0':919.3514,  'ion':'H I'},
         'HI 918'     : {'f':9.21e-4,  'gamma':5.06e4,  'wave0':918.1293,  'ion':'H I'},
         'HI 917'     : {'f':7.226e-4, 'gamma':3.39e4,  'wave0':917.1805,  'ion':'H I'},
         'HI 916a'    : {'f':5.77e-4,  'gamma':2.34e4,  'wave0':916.4291,  'ion':'H I'},
         'HI 916b'    : {'f':4.69e-4,  'gamma':1.66e4,  'wave0':915.8238,  'ion':'H I'},
         'CIV 1548'   : {'f':1.908e-1, 'gamma':2.654e8, 'wave0':1548.195,  'ion':'C IV'},
         'CIV 1551'   : {'f':9.522e-2, 'gamma':2.641e8, 'wave0':1550.770,  'ion':'C IV'},
         'OVI 1038'   : {'f':6.580e-2, 'gamma':4.076e8, 'wave0':1037.6167, 'ion':'O VI'},
         'OVI 1032'   : {'f':1.325e-1, 'gamma':4.149e8, 'wave0':1031.9261, 'ion':'O VI'},
         'MgII 1239'  : {'f':2.675e-4, 'gamma':5.802e5, 'wave0':1239.9253, 'ion':'Mg II'},
         'MgII 1240'  : {'f':1.337e-4, 'gamma':5.796e5, 'wave0':1240.3947, 'ion':'Mg II'},
         'MgII 2796'  : {'f':0.5909,   'gamma':2.52e8,  'wave0':2796.3543, 'ion':'Mg II'},
         'MgII 2803'  : {'f':0.2958,   'gamma':2.51e8,  'wave0':2803.5315, 'ion':'Mg II'},
         'AlII 1671'  : {'f':1.880,    'gamma':1.46e9,  'wave0':1670.787,  'ion':'Al II'},
         'AlIII 1855' : {'f':0.539,    'gamma':2.00e8,  'wave0':1854.716,  'ion':'Al III'},
         'AlIII 1863' : {'f':0.268,    'gamma':2.00e8,  'wave0':1862.790,  'ion':'Al III'},
         'SiIV 1394'  : {'f':0.528,    'gamma':9.200e8, 'wave0':1393.755,  'ion':'Si IV'},
         'SiIV 1403'  : {'f':0.262,    'gamma':9.030e8, 'wave0':1402.770,  'ion':'Si IV'},
         'FeII 2587'  : {'f':6.457e-2, 'gamma':2.720e8, 'wave0':2586.650,  'ion':'Fe II'},
         'FeII 2600'  : {'f':2.239e-1, 'gamma':2.700e8, 'wave0':2600.1729, 'ion':'Fe II'},
         'FeII 2374'  : {'f':2.818e-2, 'gamma':2.990e8, 'wave0':2374.4612, 'ion':'Fe II'},
         'FeII 2383'  : {'f':3.006e-1, 'gamma':3.100e8, 'wave0':2382.765,  'ion':'Fe II'}}

# instrument characteristics (in Ang)
# R = lambda/dlambda = c/dv
# EW_restframe = W_obs / (1+z_abs)
instruments = {'idealized'  : {'wave_min':1000, 'wave_max':12000, 'dwave':0.1}, # used for EW map vis
               'COS-G130M'  : {'wave_min':1150, 'wave_max':1450,  'dwave':0.01},
               'COS-G140L'  : {'wave_min':1130, 'wave_max':2330,  'dwave':0.08},
               'COS-G160M'  : {'wave_min':1405, 'wave_max':1777,  'dwave':0.012},
               'test_EUV'   : {'wave_min':800,  'wave_max':1300,  'dwave':0.1}, # to see LySeries at rest
               'test_EUV2'  : {'wave_min':1200, 'wave_max':2000,  'dwave':0.1}, # testing redshift shifts
               'SDSS-BOSS'  : {'wave_min':3543, 'wave_max':10400, 'dlogwave':1e-4}, # constant log10(dwave)=1e-4
               '4MOST_LRS'  : {'wave_min':4000, 'wave_max':8860,  'dwave':0.8},  # assume R=5000 = lambda/dlambda
               '4MOST_HRS'  : {'wave_min':3926, 'wave_max':6790,  'R':20000},    # but gaps!
               'MIKE'       : {'wave_min':3350, 'wave_max':9500,  'dwave':0.07}, # approximate only
               'KECK-HIRES' : {'wave_min':3000, 'wave_max':9250,  'dwave':0.04}} # approximate only

# pull out some units for JITed functions
sP_units_Mpc_in_cm = 3.08568e24
sP_units_boltzmann = 1.380650e-16
sP_units_c_km_s = 2.9979e5
sP_units_c_cgs = 2.9979e10
sP_units_mass_proton = 1.672622e-24

def _line_params(line):
    """ Return 5-tuple of (f,Gamma,wave0,ion_amu,ion_mass). """
    from ..cosmo.cloudy import cloudyIon

    element = lines[line]['ion'].split(' ')[0]
    ion_amu = {el['symbol']:el['mass'] for el in cloudyIon._el}[element]
    ion_mass = ion_amu * sP_units_mass_proton # g

    return lines[line]['f'], lines[line]['gamma'], lines[line]['wave0'], ion_amu, ion_mass

# cpdef double complex wofz(double complex x0) nogil
addr = get_cython_function_address("scipy.special.cython_special", "wofz")
# first argument of CFUNCTYPE() is return type, which is actually 'complex double' but no support for this
# pass the complex value x0 on the stack as two adjacent double values
functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double) 
# Note: rather dangerous as the real part isn't strictly guaranteed to be the first 8 bytes
wofz_complex_fn_realpart = functype(addr)

@jit(nopython=True, nogil=True, cache=False)
def _voigt0(wave_cm, b, wave0_ang, gamma):
    """ Dimensionless Voigt profile (shape).

    Args:
      wave_cm (array[float]): wavelength grid in [cm] where the profile is calculated.
      b (float): doppler parameter in km/s.
      wave0_ang (float): central wavelength of transition in angstroms.
      gamma (float): sum of transition probabilities (Einstein A coefficients).
    """
    nu = sP_units_c_cgs / wave_cm # wave = c/nu
    wave_rest = wave0_ang * 1e-8 # angstrom -> cm
    nu0 = sP_units_c_cgs / wave_rest # Hz
    b_cgs = b * 1e5 # km/s -> cm/s
    dnu = b_cgs / wave_rest # Hz, "doppler width" = sigma/sqrt(2)

    # use Faddeeva for integral
    alpha = gamma / (4*np.pi*dnu)
    voigt_u = (nu - nu0) / dnu # z

    # numba wofz issue: https://github.com/numba/numba/issues/3086
    #voigt_wofz = wofz(voigt_u + 1j*alpha).real # H(alpha,z)
    voigt_wofz = np.zeros(voigt_u.size, dtype=np.float64)
    for i in range(voigt_u.size):
        voigt_wofz[i] = wofz_complex_fn_realpart(voigt_u[i], alpha)

    phi_wave = voigt_wofz / b_cgs # s/cm
    return phi_wave

@jit(nopython=True, nogil=True, cache=False)
def _voigt_tau(wave, N, b, wave0, f, gamma, wave0_rest=None, logwave=False):
    """ Compute optical depth tau as a function of wavelength for a Voigt absorption profile.

    Args:
      wave (array[float]): wavelength grid in [ang]
      N (float): column density of absorbing species in [cm^-2]
      b (float): doppler parameter, equal to sqrt(2kT/m) where m is the particle mass.
        b = sigma*sqrt(2) where sigma is the velocity dispersion.
      wave0 (float): central wavelength of the transition in [ang]
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0_rest (float): if not None, then rest-frame central wavelength, i.e. wave0 could be redshifted
      logwave (bool): if True, interpret wave input as [log ang].
    """
    if wave0_rest is None:
        wave0_rest = wave0

    # get dimensionless shape for voigt profile:
    if logwave:
        wave_cm = np.exp(wave) * 1e-8
    else:
        wave_cm = wave * 1e-8

    phi_wave = _voigt0(wave_cm, b, wave0, gamma)

    consts = 0.014971475 # sqrt(pi)*e^2 / m_e / c = cm^2/s
    wave0_rest_cm = wave0_rest * 1e-8

    tau_wave = (consts * N * f * wave0_rest_cm) * phi_wave
    return tau_wave

@jit(nopython=True, nogil=True, cache=True)
def _equiv_width(tau,wave_mid_ang):
    """ Compute the equivalent width by integrating the optical depth array across the given wavelength grid. """
    assert wave_mid_ang.size == tau.size

    # wavelength bin size
    dang = np.abs(np.diff(wave_mid_ang))

    # integrate (1-exp(-tau_lambda)) d_lambda from 0 to inf, composite trap rule
    integrand = 1 - np.exp(-tau)
    res = np.sum(dang * (integrand[1:] + integrand[:-1])/2)

    # (only for constant dwave):
    # dang = wave_mid_ang[1] - wave_mid_ang[0]
    # res = dang / 2 * (integrand[0] + integrand[-1] + np.sum(2*integrand[1:-1]))

    return res

@jit(nopython=True, nogil=True, cache=True)
def _equiv_width_flux(flux,wave_mid_ang):
    """ Compute the equivalent width by integrating the continuum normalized flux array across the given wavelength grid. """
    assert wave_mid_ang.size == flux.size

    # wavelength bin size
    dang = np.abs(np.diff(wave_mid_ang))

    # integrate 1-flux = (1-exp(-tau_lambda)) d_lambda from 0 to inf, composite trap rule
    integrand = 1 - flux
    res = np.sum(dang * (integrand[1:] + integrand[:-1])/2)

    return res

def curveOfGrowth(line='MgII 2803'):
    """ Plot relationship between EW and N for a given transition (e.g. HI, MgII line). """
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

    for N in [12,15,18,21]:
        for j, sigma in enumerate([30]):
            b = sigma * np.sqrt(2)
            tau = _voigt_tau(wave_ang, 10.0**N, b, wave0_ang, f, gamma)
            flux = np.exp(-1*tau)
            EW = _equiv_width(tau,wave_ang)
            print(N,b,EW)

            ax.plot(dvel, flux, lw=lw, linestyle=linestyles[j], label='N = %f b = %d' % (N,b))

    ax.legend()
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

    ax.legend()
    fig.savefig('cog_%s.pdf' % line)
    plt.close(fig)

def create_master_grid(line=None, instrument=None):
    """ Create a master grid (i.e. spectrum) to receieve absorption line depositions.
    Must specify one, but not both, of either 'line' or 'instrument'. In the first case, 
    a local spectrum is made around its rest-frame central wavelength. In the second case, 
    a global spectrum is made corresponding to the instrumental properties.
    """
    assert line is not None or instrument is not None

    if line is not None:
        f, gamma, wave0_restframe, _, _ = _line_params(line)

    # master wavelength grid, observed-frame [ang]
    dwave = None
    dlogwave = None

    if line is not None:
        wave_min = np.floor(wave0_restframe - 15.0)
        wave_max = np.ceil(wave0_restframe + 15.0)
        dwave = 0.1

    if instrument is not None:
        wave_min = instruments[instrument]['wave_min']
        wave_max = instruments[instrument]['wave_max']
        if 'dwave' in instruments[instrument]:
            dwave = instruments[instrument]['dwave']
        if 'dlogwave' in instruments[instrument]:
            dlogwave = instruments[instrument]['dlogwave']

    # if dwave is specified, use linear wavelength spacing
    if dwave is not None:
        print(f'Creating linear wavelength grid with {dwave = :.3f}')
        num_edges = int(np.floor((wave_max - wave_min) / dwave)) + 1
        wave_edges = np.linspace(wave_min, wave_max, num_edges)
        wave_mid = (wave_edges[1:] + wave_edges[:-1]) / 2

    # if dlogwave is specified, use log10-linear wavelength spacing
    if dlogwave is not None:
        log_wavemin = np.log10(wave_min)
        log_wavemax = np.log10(wave_max)
        log_wave_mid = np.arange(log_wavemin,log_wavemax+dlogwave,dlogwave)
        wave_mid = 10.0**log_wave_mid
        log_wave_edges = np.arange(log_wavemin-dlogwave/2,log_wavemax+dlogwave+dlogwave/2,dlogwave)
        wave_edges = 10.0**log_wave_edges

    # else, use spectral resolution R, and create linear in log(wave) grid
    if dwave is None and dlogwave is None:
        R = instruments[instrument]['R']
        print(f'Creating loglinear wavelength grid with {R = }')
        log_wavemin = np.log(wave_min)
        log_wavemax = np.log(wave_max)
        d_loglam = 1/R
        log_wave_mid = np.arange(log_wavemin,log_wavemax+d_loglam,d_loglam)
        wave_mid = np.exp(log_wave_mid)
        log_wave_edges = np.arange(log_wavemin-d_loglam/2,log_wavemax+d_loglam+d_loglam/2,d_loglam)
        wave_edges = np.exp(log_wave_edges)

    tau_master = np.zeros(wave_mid.size, dtype='float32')

    return wave_mid, wave_edges, tau_master

@jit(nopython=True, nogil=True, cache=False)
def deposit_single_line(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, logwave=False, debug=False):
    """ Add the absorption profile of a single transition, from a single cell, to a spectrum.

    Args:
      wave_edges_master (array[float]): bin edges for master spectrum array [ang].
      tau_master (array[float]): master optical depth array.
      N (float): column density in [1/cm^2].
      b (float): doppler parameter in [km/s].
      f (float): oscillator strength of the transition
      gamma (float): sum of transition probabilities (Einstein A coefficients) [1/s]
      wave0 (float): central wavelength, rest-frame [ang].
      z_eff (float): effective redshift, i.e. including both cosmological and peculiar components.
      logwave (bool): if True, interpret wave_edges_master as [log ang].
      debug (bool): if True, return local grid info and do checks.

    Return:
      None.
    """
    if N == 0:
        return # empty

    # local (to the line), rest-frame wavelength grid
    dwave_local = 0.01 # ang
    edge_tol = 1e-4 # if the optical depth is larger than this by the edge of the local grid, redo

    b_dwave = b / sP_units_c_km_s * wave0 # v/c = dwave/wave

    # adjust local resolution to make sure we sample narrow lines
    while b_dwave < dwave_local * 4:
        dwave_local *= 0.5

        if dwave_local < 1e-5:
            print(b, b_dwave, dwave_local)
            assert 0 # check
            break

    # prep local grid
    wave0_obsframe = wave0 * (1 + z_eff)

    line_width_safety = b / sP_units_c_km_s * wave0_obsframe

    dwave_master = wave_edges_master[1] - wave_edges_master[0]
    nloc_per_master = int(np.round(dwave_master / dwave_local))

    n_iter = 0
    local_fac = 5.0
    tau = np.array([np.inf], dtype=np.float64)

    while tau[0] > edge_tol or tau[-1] > edge_tol:
        # determine where local grid overlaps with master
        wave_min_local = wave0_obsframe - local_fac*line_width_safety
        wave_max_local = wave0_obsframe + local_fac*line_width_safety

        master_inds = np.searchsorted(wave_edges_master, [wave_min_local,wave_max_local])
        master_startind = master_inds[0] - 1
        master_finalind = master_inds[1]

        # sanity checks
        if master_startind == -1:
            if debug: print('WARNING: min edge of local grid hit edge of master!')
            master_startind = 0

        if master_finalind == wave_edges_master.size:
            if debug: print('WARNING: max edge of local grid hit edge of master!')
            master_finalind = wave_edges_master.size - 1

        if master_startind == master_finalind:
            if n_iter < 20:
                # extend, see if wings of this feature will enter master spectrum
                local_fac *= 1.2
                n_iter += 1
                continue

            if debug: print('WARNING: absorber entirely off edge of master spectrum! skipping!')
            return

        # create local grid specification aligned with master
        nmaster_covered = master_finalind - master_startind # difference of bin edge indices
        num_bins_local = nmaster_covered * nloc_per_master

        wave_min_local = wave_edges_master[master_startind]
        wave_max_local = wave_edges_master[master_finalind]

        # create local grid
        wave_edges_local = np.linspace(wave_min_local, wave_max_local, num_bins_local+1)
        wave_mid_local = (wave_edges_local[1:] + wave_edges_local[:-1]) / 2

        # get optical depth
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0, logwave=logwave)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        #if debug: print(f'{local_fac = }, {tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

        if n_iter > 100:
            break

        if master_startind == 0 and master_finalind == wave_edges_master.size - 1:
            break # local grid already extended to entire master

        local_fac *= 1.2
        n_iter += 1

    if (tau[0] > edge_tol or tau[-1] > edge_tol):
        print('WARNING: final local grid edges still have high tau')
        if not debug: assert 0

    # integrate local tau within each bin of master tau
    master_ind = master_startind
    count = 0
    tau_bin = 0.0

    for local_ind in range(wave_mid_local.size):
        # deposit partial integral of tau into master bin
        tau_bin += tau[local_ind]
        count += 1

        #print(f' add to tau_master[{master_ind:2d}] from {local_ind = :2d} with {tau[local_ind] = :.3g} i.e. {wave_mid_local[local_ind]:.4f} [{wave_edges_local[local_ind]:.4f}-{wave_edges_local[local_ind+1]:.4f}] Ang into {wave_mid[master_ind]:.2f} [{wave_edges_master[master_ind]}-{wave_edges_master[master_ind+1]}] Ang')

        if count == nloc_per_master:
            # midpoint rule
            tau_master[master_ind] += tau_bin * (dwave_local/dwave_master)
            #print(f'  midpoint tau_master[{master_ind:2d}] = {tau_master[master_ind]:.4f}')

            # move to next master bin
            master_ind += 1
            count = 0
            tau_bin = 0.0

    if debug:
        # debug check
        wave_mid_master = (wave_edges_master[1:] + wave_edges_master[:-1]) / 2
        EW_local = _equiv_width(tau,wave_mid_local)
        EW_master = _equiv_width(tau_master,wave_mid_master)

        tau_local_tot = np.sum(tau * dwave_local)
        tau_master_tot = np.sum(tau_master * dwave_master)

        #print(f'{EW_local = :.6f}, {EW_master = :.6f}, {tau_local_tot = :.5f}, {tau_master_tot = :.5f}')

    if debug:
        # get flux
        flux = np.exp(-1*tau)

        # return local wavelength, tau, and flux arrays
        return wave_mid_local, tau, flux

    return

def _spectrum_debug_plot(sP, line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos):
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
    dv = (master_mid-xcen)/xcen * sP.units.c_km_s
    ax.plot(dv, flux_master, '-', lw=lw, label=line)

    ax = fig.add_subplot(324) # center right
    ax.set_xlabel('Distance Along Ray [ Mpc ]')
    ax.set_ylabel('Column Density [log cm$^{-2}$]')
    ax.plot(master_dl, logZeroNaN(master_dens*master_dx*sP.units.Mpc_in_cm), '-', lw=lw)

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

def create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
    master_dens, master_dx, master_temp, master_vellos):
    """ Given a completed (single) ray traced through a volume, and the properties of all the intersected 
    cells (dens, dx, temp, vellos), create the final absorption spectrum, depositing a Voigt absorption 
    profile for each cell. """
    nCells = master_dens.size

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)

    # assign sP.redshift to the front intersectiom (beginning) of the box
    z_start = sP.redshift # 0.1 # change to imagine that this snapshot is at a different redshift

    z_vals = np.linspace(z_start, z_start+0.1, 200)
    lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(z_start)

    # cumulative pathlength, Mpc from start of box i.e. start of ray (at z_start)
    cum_pathlength = np.zeros(nCells, dtype='float32') 
    cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # Mpc

    # cosmological redshift of each intersected cell
    z_cosmo = np.interp(cum_pathlength, lengths, z_vals)

    # doppler shift
    z_doppler = master_vellos / units.c_km_s

    # effective redshift
    z_eff = (1+z_doppler)*(1+z_cosmo) - 1

    # column density
    N = master_dens * (master_dx * sP.units.Mpc_in_cm) # cm^-2

    # doppler parameter b = sqrt(2kT/m) where m is the particle mass
    b = np.sqrt(2 * sP.units.boltzmann * master_temp / ion_mass) # cm/s
    b /= 1e5 # km/s

    # deposit each intersected cell as an absorption profile onto spectrum
    for i in range(nCells):
        print(f'[{i:3d}] N = {logZeroNaN(N[i]):6.3f} {b[i] = :7.2f} {z_eff[i] = :.6f}')
        deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[i], b[i], z_eff[i])

    return master_mid, tau_master, z_eff

@jit(nopython=True, nogil=True, cache=False)
def _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                     rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                     cell_dens, cell_temp, cell_vellos, 
                                     z_vals, z_lengths,
                                     master_mid, master_edges, ind0, ind1):
    """ JITed helper (see below). """
    n_rays = ind1 - ind0 + 1

    # allocate: full spectra return as well as derived EWs
    tau_master = np.zeros(master_mid.size, dtype=np.float32)
    tau_allrays = np.zeros((n_rays,tau_master.size), dtype=np.float32)
    EW_master = np.zeros(n_rays, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # get local properties
        offset = rays_off[ind0+i] # start of intersected cells (in rays_cell*)
        length = rays_len[ind0+i] # number of intersected gas cells

        master_dx = rays_cell_dl[offset:offset+length]
        master_inds = rays_cell_inds[offset:offset+length]

        master_dens = cell_dens[master_inds]
        master_temp = cell_temp[master_inds]
        master_vellos = cell_vellos[master_inds]

        # reset tau_master for each ray
        tau_master *= 0.0

        # cumulative pathlength, Mpc from start of box i.e. start of ray (at sP.redshift)
        cum_pathlength = np.zeros(length, dtype=np.float32) 
        cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # Mpc

        # cosmological redshift of each intersected cell
        z_cosmo = np.interp(cum_pathlength, z_lengths, z_vals)

        # doppler shift
        z_doppler = master_vellos / sP_units_c_km_s

        # effective redshift
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1

        # column density
        N = master_dens * (master_dx * sP_units_Mpc_in_cm) # cm^-2

        # doppler parameter b = sqrt(2kT/m) where m is the particle mass
        b = np.sqrt(2 * sP_units_boltzmann * master_temp / ion_mass) # cm/s
        b /= 1e5 # km/s

        # deposit each intersected cell as an absorption profile onto spectrum
        for j in range(length):
            #print(f' [{j:3d}] N = {logZeroNaN(N[j]):6.3f} {b[j] = :7.2f} {z_eff[j] = :.6f}')
            deposit_single_line(master_edges, tau_master, f, gamma, wave0, N[j], b[j], z_eff[j])

        # stamp spectrum
        tau_allrays[i,:] = tau_master

        # also compute EW and save
        # note: is currently a global EW, i.e. not localized/restricted to a single absorber
        EW_master[i] = _equiv_width(tau_master,master_mid)

    return master_mid, tau_allrays, EW_master

def create_spectra_from_traced_rays(sP, line, instrument,
                                    rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                    cell_dens, cell_temp, cell_vellos, nThreads=20):
    """ Given many completed rays traced through a volume, in the form of a composite list of 
    intersected cell pathlengths and indices, extract the physical properties needed (dens, temp, vellos) 
    and create the final absorption spectrum, depositing a Voigt absorption profile for each cell.

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
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    # assign sP.redshift to the front intersectiom (beginning) of the box
    z_vals = np.linspace(sP.redshift, sP.redshift+0.1, 200)
    z_lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(sP.redshift)

    # sample master grid
    master_mid, master_edges, _ = create_master_grid(instrument=instrument)

    # single-threaded
    if nThreads == 1 or n_rays < nThreads:
        ind0 = 0
        ind1 = n_rays - 1

        return _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                                rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                master_mid, master_edges, ind0, ind1)

    # multi-threaded
    class specThread(threading.Thread):
        """ Subclass Thread() to provide local storage which can be retrieved after 
            this thread terminates and added to the global return. """
        def __init__(self, threadNum, nThreads):
            super(specThread, self).__init__()

            # determine local slice
            self.ind0, self.ind1 = pSplitRange([0, n_rays-1], nThreads, threadNum, inclusive=True)

        def run(self):
            # call JIT compiled kernel
            self.result = _create_spectra_from_traced_rays(f, gamma, wave0, ion_mass, instrument,
                                                rays_off, rays_len, rays_cell_dl, rays_cell_inds, 
                                                cell_dens, cell_temp, cell_vellos, z_vals, z_lengths,
                                                master_mid, master_edges, self.ind0, self.ind1)

    # create threads
    threads = [specThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

    # all threads are done, determine return size and allocate
    tau_allrays = np.zeros((n_rays,master_mid.size), dtype='float32')
    EW_master = np.zeros(n_rays, dtype='float32')

    # add the result array from each thread to the global
    for thread in threads:
        wave_loc, tau_loc, EW_loc = thread.result

        tau_allrays[thread.ind0 : thread.ind1 + 1,:] = tau_loc
        EW_master[thread.ind0 : thread.ind1 + 1] = EW_loc

    return master_mid, tau_allrays, EW_master

def generate_spectrum_uniform_grid():
    """ Generate an absorption spectrum by ray-tracing through a uniform grid (deposit using sphMap). """
    from ..util.simParams import simParams
    from ..util.sphMap import sphGridWholeBox, sphMap
    from ..cosmo.cloudy import cloudyIon

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1032' #'LyA'
    instrument = 'test_EUV2' # 'SDSS-BOSS' #
    nCells = 64
    haloID = 150 # if None, then full box

    posInds = [int(nCells*0.5),int(nCells*0.5)] # [0,0] # (x,y) pixel indices to ray-trace along
    projAxis = 2 # z, to simplify vellos

    # quick caching
    cacheFile = f"cache_{line}_{nCells}_h{haloID}_{sP.snap}.hdf5"
    if isfile(cacheFile):
        # load now
        print(f'Loading [{cacheFile}].')
        with h5py.File(cacheFile,'r') as f:
            grid_dens = f['grid_dens'][()]
            grid_vel = f['grid_vel'][()]
            grid_temp = f['grid_temp'][()]
            if haloID is not None:
                boxSizeImg = f['boxSizeImg'][()]
    else:
        # load
        massField = '%s mass' % lines[line]['ion']
        velField = 'vel_' + ['x','y','z'][projAxis]

        pos = sP.snapshotSubsetP('gas', 'pos', haloID=haloID) # code
        vel_los = sP.snapshotSubsetP('gas', velField, haloID=haloID) # code
        mass = sP.snapshotSubsetP('gas', massField, haloID=haloID) # code
        hsml = sP.snapshotSubsetP('gas', 'hsml', haloID=haloID) # code
        temp = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloID) # K

        # grid
        if haloID is None:
            grid_mass = sphGridWholeBox(sP, pos, hsml, mass, None, nCells=nCells)
            grid_vel = sphGridWholeBox(sP, pos, hsml, mass, vel_los, nCells=nCells)
            grid_temp = sphGridWholeBox(sP, pos, hsml, mass, temp, nCells=nCells)

            pxVol = (sP.boxSize / nCells)**3 # code units (ckpc/h)^3
        else:
            halo = sP.halo(haloID)
            haloSizeRvir = 2.0
            boxSizeImg = halo['Group_R_Crit200'] * np.array([haloSizeRvir,haloSizeRvir,haloSizeRvir])
            boxCen = halo['GroupPos']

            grid_mass = sphMap( pos=pos, hsml=hsml, mass=mass, quant=None, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )
            grid_vel  = sphMap( pos=pos, hsml=hsml, mass=mass, quant=vel_los, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )
            grid_temp = sphMap( pos=pos, hsml=hsml, mass=mass, quant=temp, axes=[0,1], 
                                ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, 
                                boxCen=boxCen, nPixels=[nCells, nCells, nCells] )

            pxVol = np.prod(boxSizeImg) / nCells**3 # code units

        # unit conversions: mass -> density
        f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

        grid_dens = sP.units.codeDensToPhys(grid_mass/pxVol, cgs=True, numDens=True) # H atoms/cm^3
        grid_dens /= ion_amu # [ions/cm^3]

        # unit conversions: line-of-sight velocity
        grid_vel = sP.units.particleCodeVelocityToKms(grid_vel) # physical km/s

        # save
        with h5py.File(cacheFile,'w') as f:
            f['grid_dens'] = grid_dens
            f['grid_vel'] = grid_vel
            f['grid_temp'] = grid_temp
            if haloID is not None:
                f['boxSizeImg'] = boxSizeImg
        print(f'Saved [{cacheFile}].')

    # print ray starting location in global space (note: possible the grid is permuted/transposed still)
    print(f'{boxSizeImg = }')
    if haloID is None:
        boxCen = np.zeros(3) + sP.boxSize/2
    else:
        halo = sP.halo(haloID)
        boxCen = halo['GroupPos']
    pxScale = boxSizeImg[0] / grid_dens.shape[0]

    ray_x = boxCen[0] - boxSizeImg[0]/2 + posInds[0]*pxScale
    ray_y = boxCen[1] - boxSizeImg[1]/2 + posInds[1]*pxScale
    ray_z = boxCen[2] - boxSizeImg[2]/2
    print(f'Starting {ray_x = :.4f} {ray_y = :.4f} {ray_z = :4f}')

    # create theory-space master grids
    master_dens   = np.zeros(nCells, dtype='float32') # density for each ray segment
    master_dx     = np.zeros(nCells, dtype='float32') # pathlength for each ray segment
    master_temp   = np.zeros(nCells, dtype='float32') # temp for each ray segment
    master_vellos = np.zeros(nCells, dtype='float32') # line of sight velocity

    # init
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    boxSize = sP.boxSize if haloID is None else boxSizeImg[projAxis]
    dx_Mpc = sP.units.codeLengthToMpc(boxSize / nCells)

    # 'ray trace' a single pixel from front of box to back of box
    for i in range(nCells):
        # store cell properties
        master_vellos[i] = grid_vel[posInds[0], posInds[1], i]
        master_dens[i] = grid_dens[posInds[0], posInds[1], i]
        master_temp[i] = grid_temp[posInds[0], posInds[1], i]
        master_dx[i] = dx_Mpc # constant

    # create spectrum
    master_mid, tau_master, z_eff = create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
                                      master_dens, master_dx, master_temp, master_vellos)

    # plot
    plotName = f"spectrum_box_{sP.simName}_{line}_{nCells}_h{haloID}_{posInds[0]}-{posInds[1]}_z{sP.redshift:.0f}.pdf"

    _spectrum_debug_plot(sP, line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos)

def generate_spectrum_voronoi(use_precomputed_mesh=True, compare=False, debug=1, verify=True):
    """ Generate an absorption spectrum by ray-tracing through the Voronoi mesh.

    Args:
      use_precomputed_mesh (bool): if True, use pre-computed Voronoi mesh connectivity from VPPP, 
        otherwise use tree-based, connectivity-free method.
      compare (bool): if True, run both methods and compare results.
      debug (int): verbosity level for diagnostic outputs: 0 (silent), 1, 2, or 3 (most verbose).
      verify (bool): if True, brute-force distance calculation verify parent cell at each step.
    """
    from ..util.simParams import simParams
    from ..util.voronoi import loadSingleHaloVPPP, loadGlobalVPPP
    from ..cosmo.cloudy import cloudyIon
    from ..util.treeSearch import buildFullTree

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1032' #'LyA'
    instrument = 'test_EUV2' # 'SDSS-BOSS'
    haloID = 150 # if None, then full box

    ray_offset_x = 0.0 # relative to halo center, in units of rvir
    ray_offset_y = 0.5 # relative to halo center, in units of rvir
    ray_offset_z = -2.0 # relative to halo center, in units of rvir
    projAxis = 2 # z, to simplify vellos for now

    fof_scope_mesh = False

    # load halo
    halo = sP.halo(haloID)

    print(f"Halo [{haloID}] center {halo['GroupPos']} and Rvir = {halo['Group_R_Crit200']:.2f}")

    # ray starting position, and total requested pathlength
    ray_start_x = halo['GroupPos'][0]        + ray_offset_x*halo['Group_R_Crit200']
    ray_start_y = halo['GroupPos'][1]        + ray_offset_y*halo['Group_R_Crit200']
    ray_start_z = halo['GroupPos'][projAxis] + ray_offset_z*halo['Group_R_Crit200']

    total_dl = np.abs(ray_offset_z*2) * halo['Group_R_Crit200'] # twice distance to center

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0

    # load cell properties (pos,vel,species dens,temp)
    densField = '%s numdens' % lines[line]['ion']
    velLosField = 'vel_'+['x','y','z'][projAxis]

    haloIDLoad = haloID if fof_scope_mesh else None # if global mesh, then global gas load

    cell_pos    = sP.snapshotSubsetP('gas', 'pos', haloID=haloIDLoad) # code
    cell_vellos = sP.snapshotSubsetP('gas', velLosField, haloID=haloIDLoad) # code
    cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloIDLoad) # K
    cell_dens   = sP.snapshotSubset('gas', densField, haloID=haloIDLoad) # ions/cm^3

    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

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
        master_dx, master_ind = trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, 
                                       num_ngb, ngb_inds, offset_ngb, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify, fof_scope_mesh)

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

    if (not use_precomputed_mesh) or compare:
        # construct neighbor tree
        tree = buildFullTree(cell_pos, boxSizeSim=sP.boxSize, treePrec=cell_pos.dtype, verbose=debug)
        NextNode, length, center, sibling, nextnode = tree

        if compare:
            ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z]) # reset
            master_ind2 = master_ind.copy()
            master_dx2 = master_dx.copy()

        # ray-trace
        master_dx, master_ind = trace_ray_through_voronoi_mesh_treebased(cell_pos, 
                                       NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify)

        master_dens = cell_dens[master_ind]
        master_temp = cell_temp[master_ind]
        master_vellos = cell_vellos[master_ind]
        assert np.abs(master_dx.sum() - total_dl) < 1e-4

        if compare:
            assert np.allclose(master_dx,master_dx2)
            assert np.array_equal(master_ind,master_ind2)
            print(master_dx,master_dx2,'Comparison success.')

    # create spectrum
    f, gamma, wave0, ion_amu, ion_mass = _line_params(line)

    # convert length units, all other units already appropriate
    master_dx = sP.units.codeLengthToMpc(master_dx)

    master_mid, tau_master, z_eff = create_spectrum_from_traced_ray(sP, f, gamma, wave0, ion_mass, instrument, 
                                      master_dens, master_dx, master_temp, master_vellos)

    # plot
    meshStr = 'vppp' if use_precomputed_mesh else 'treebased'
    plotName = f"spectrum_voronoi_{sP.simName}_{line}_{meshStr}_h{haloID}_z{sP.redshift:.0f}.pdf"

    _spectrum_debug_plot(sP, line, plotName, master_mid, tau_master, master_dens, master_dx, master_temp, master_vellos)

def generate_spectra_voronoi_halo():
    """ Generate a large grid of (halocentric) absorption spectra by ray-tracing through the Voronoi mesh. """
    from ..util.simParams import simParams
    from ..cosmo.cloudy import cloudyIon

    # config
    sP = simParams(run='tng50-1', redshift=0.5)

    lineNames = ['MgII 2796','MgII 2803']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'
    haloID = 150 # 150 for TNG50-1, 800 for TNG100-1

    nRaysPerDim = 50 # total number of rays is square of this number
    projAxis = 2 # z, to simplify vellos for now, keep axis-aligned

    fof_scope_mesh = True # if False then full box load

    # caching file
    saveFilename = 'spectra_%s_z%.1f_halo%d-%d_n%d_%s_%s.hdf5' % \
      (sP.simName,sP.redshift,haloID,projAxis,nRaysPerDim,instrument,'-'.join(lineNames))

    if not isfile(saveFilename):
        # load halo
        halo = sP.halo(haloID)
        cen = halo['GroupPos']
        mass = sP.units.codeMassToLogMsun(halo['Group_M_Crit200'])[0]
        size = 2 * halo['Group_R_Crit200']

        print(f"Halo [{haloID}] mass = {mass:.2f} and Rvir = {halo['Group_R_Crit200']:.2f}")

        # ray starting positions, and total requested pathlength
        xpts = np.linspace(cen[0]-size/2, cen[0]+size/2, nRaysPerDim)
        ypts = np.linspace(cen[1]-size/2, cen[1]+size/2, nRaysPerDim)

        xpts, ypts = np.meshgrid(xpts, ypts, indexing='ij')

        # construct [N,3] list of search positions
        ray_pos = np.zeros( (nRaysPerDim**2,3), dtype='float64')
        
        ray_pos[:,0] = xpts.ravel()
        ray_pos[:,1] = ypts.ravel()
        ray_pos[:,2] = cen[2] - size/2

        # total requested pathlength (twice distance to halo center)
        total_dl = size

        # ray direction
        ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
        ray_dir[projAxis] = 1.0

        # load cell properties (pos,vel,species dens,temp)
        haloIDLoad = haloID if fof_scope_mesh else None # if global mesh, then global gas load

        cell_pos = sP.snapshotSubsetP('gas', 'pos', haloID=haloIDLoad) # code

        # ray-trace
        rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode='full',nThreads=4)

        # load other cell properties
        velLosField = 'vel_'+['x','y','z'][projAxis]

        cell_vellos = sP.snapshotSubsetP('gas', velLosField, haloID=haloIDLoad) # code
        cell_temp   = sP.snapshotSubsetP('gas', 'temp_sfcold_linear', haloID=haloIDLoad) # K
        
        cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

        # convert length units, all other units already appropriate
        rays_dl = sP.units.codeLengthToMpc(rays_dl)

        # sample master grid
        master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)
        tau_master = np.zeros( (nRaysPerDim**2,tau_master.size), dtype=tau_master.dtype )

        EWs = {}

        # start cache
        with h5py.File(saveFilename,'w') as f:
            f['master_wave'] = master_mid

        # loop over requested line(s)
        for line in lineNames:
            densField = '%s numdens' % lines[line]['ion']
            cell_dens = sP.snapshotSubset('gas', densField, haloID=haloIDLoad) # ions/cm^3

            # create spectra
            master_wave, tau_local, EW_local = \
              create_spectra_from_traced_rays(sP, line, instrument, 
                                              rays_off, rays_len, rays_dl, rays_inds,
                                              cell_dens, cell_temp, cell_vellos)

            assert np.array_equal(master_wave,master_mid)

            tau_master += tau_local
            EWs[line] = EW_local

            with h5py.File(saveFilename,'r+') as f:
                # save tau per line
                f['tau_%s' % line.replace(' ','_')] = tau_local
                # save EWs per line
                f['EW_%s' % line.replace(' ','_')] = EW_local

        # calculate flux and total EW
        flux = np.exp(-1*tau_master)

        with h5py.File(saveFilename,'r+') as f:
            f['flux'] = flux

        print(f'Saved: [{saveFilename}]')
    else:
        # load cache
        EWs = {}
        with h5py.File(saveFilename,'r') as f:
            master_wave = f['master_wave'][()]
            flux = f['flux'][()]
            for line in lineNames:
                EWs[line] = f['EW_%s' % line.replace(' ','_')][()]
        print(f'Loaded: [{saveFilename}]')

    # plot config
    EW_targets = [0.1, 0.2, 0.4, 1.0, 2.0, 8.0] # Ang
    xlim = [4200,4230] # Ang

    # total EW
    EW_total = np.zeros(flux.shape[0], dtype='float32')
    for line in lineNames:
        EW_total += EWs[line]

    # plot hist of EW
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Equivalent Width [ Ang ]')
    ax.set_ylabel('Fraction')
    ax.set_yscale('log')

    valMinMax = [-2.0,1.0]
    nBins = 30

    for line in lineNames:
        yy, xx = np.histogram(logZeroNaN(EWs[line]), bins=nBins, density=True, range=valMinMax)
        xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins

        ax.plot(xx, yy, label=line)

    fig.savefig('spectra_EW_hist_%s_%s_%s.pdf' % (sP.simName,instrument,'-'.join(lineNames)))
    plt.close(fig)

    # plot some sample spectra
    ray_inds = []
    for EW_target in EW_targets:
        _, ind = closest(EW_total, EW_target)
        ray_inds.append(ind)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel('Wavelength [ Ang ]')
    ax.set_ylabel('Relative Flux')
    ax.set_xlim(xlim)

    for ray_ind in ray_inds:
        label = f'EW = {EW_total[ray_ind]:.2f} Ang' #'EW = %.2f Ang [#%d]' % (EW_total[ray_ind],ray_ind)
        ax.plot(master_wave, flux[ray_ind,:], '-', lw=lw, label=label)

    ax.legend(loc='lower left')
    fig.savefig('spectra_%s_%s_%s.pdf' % (sP.simName,instrument,'-'.join(lineNames)))
    plt.close(fig)

def generate_rays_voronoi_fullbox(sP, projAxis=2, pSplit=None):
    """ Generate a large grid of (fullbox) rays by ray-tracing through the Voronoi mesh.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      projAxis (int): either 0, 1, or 2. only axis-aligned allowed for now.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
    """

    # config
    nRaysPerDim = 1000 # total number of rays (per box, summing over all pSplits) is this number squared
    raysType = 'voronoi_fullbox'

    # paths and save file
    if not isdir(sP.derivPath + 'rays'):
        mkdir(sP.derivPath + 'rays')

    pathStr1 = sP.derivPath + 'rays/%s_n%dd%d_%03d.hdf5' % (raysType,nRaysPerDim,projAxis,sP.snap)
    pathStr2 = sP.derivPath + 'rays/%s_n%dd%d_%03d-split-%d-%d.hdf5' % \
      (raysType,nRaysPerDim,projAxis,sP.snap,pSplit[0],pSplit[1])

    path = pathStr2 if pSplit is not None else pathStr1

    # total requested pathlength (equal to box length)
    total_dl = sP.boxSize

    # ray direction
    ray_dir = np.array([0.0, 0.0, 0.0], dtype='float64')
    ray_dir[projAxis] = 1.0    

    # check existence
    if isfile(path):
        print('Loading [%s].' % path)
        with h5py.File(path, 'r') as f:
            # ray results
            rays_off = f['rays_off'][()]
            rays_len = f['rays_len'][()]
            rays_dl = f['rays_dl'][()]
            rays_inds = f['rays_inds'][()]

            # ray config
            cell_inds = f['cell_inds'][()] if 'cell_inds' in f else None
            ray_pos = f['ray_pos'][()]

            # metadata
            attrs = {}
            for attr in f.attrs:
                attrs[attr] = f.attrs[attr]

        return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, attrs['total_dl']

    pSplitStr = ' (split %d of %d)' % (pSplit[0],pSplit[1]) if pSplit is not None else ''
    print('Compute and save: [%s z=%.1f] [%s]%s' % (sP.simName,sP.redshift,raysType,pSplitStr))
    print('Total number of rays: %d x %d = %d' % (nRaysPerDim,nRaysPerDim,nRaysPerDim**2))

    # spatial decomposition
    if pSplit is not None:
        assert np.abs(np.sqrt(pSplit[1]) - np.round(np.sqrt(pSplit[1]))) < 1e-6, 'pSplitSpatial: Total number of jobs should have integer square root, e.g. 9, 16, 25, 64.'
        nPerDim = int(np.sqrt(pSplit[1]))
        extent = sP.boxSize / nPerDim

        # [x,y] bounds of this spatial subset
        ij = np.unravel_index(pSplit[0], (nPerDim,nPerDim))
        xmin = ij[0] * extent
        xmax = (ij[0]+1) * extent
        ymin = ij[1] * extent
        ymax = (ij[1]+1) * extent

        # number of rays in this spatial subset
        nRaysPerDim = nRaysPerDim/np.sqrt(pSplit[1])
        assert nRaysPerDim.is_integer(), 'pSplitSpatial: nRaysPerDim is not divisable by square root of total number of jobs.'
        nRaysPerDim = int(nRaysPerDim)

        print(' pSplitSpatial: [%d of %d] ij (%d %d) extent [%g] x [%.1f - %.1f] y [%.1f - %.1f]' % \
            (pSplit[0],pSplit[1],ij[0],ij[1],extent,xmin,xmax,ymin,ymax))
        print(' subset of rays: %d x %d = %d' % (nRaysPerDim,nRaysPerDim,nRaysPerDim**2))
    else:
        xmin = ymin = 0.0
        xmax = ymax = sP.boxSize

    # ray starting positions (skip last, which will be duplicate with first)
    xpts = np.linspace(xmin, xmax, nRaysPerDim+1)[:-1]
    ypts = np.linspace(ymin, ymax, nRaysPerDim+1)[:-1]

    xpts, ypts = np.meshgrid(xpts, ypts, indexing='ij')

    # construct [N,3] list of ray locations
    ray_pos = np.zeros( (nRaysPerDim**2,3), dtype='float64')
    
    ray_pos[:,0] = xpts.ravel()
    ray_pos[:,1] = ypts.ravel()
    ray_pos[:,2] = 0.0

    if 0:
        # DEBUG only
        ray_pos_new = np.zeros( (1,3), dtype='float64' )
        ray_pos_new[0,:] = ray_pos[4,:] # 4 62500 [    0. 17640.     0.]
        ray_pos = ray_pos_new

    # determine spatial mask (cuboid with long side equal to boxlength in line-of-sight direction)
    cell_inds = None

    if 1:
        if pSplit is not None:
            mask = np.zeros(sP.numPart[sP.ptNum('gas')], dtype='int8')
            mask += 1 # all required

            print(' pSplitSpatial:', end='')
            for ind, axis in enumerate(['x','y']):
                print(' slice[%s]...' % axis, end='')
                dists = sP.snapshotSubsetP('gas', 'pos_'+axis, float32=True)

                dists = (ij[ind] + 0.5) * extent - dists # 1D, along axis, from center of subregion
                sP.correctPeriodicDistVecs(dists)

                # compute maxdist heuristic (in code units): the largest 1d distance we need for the calculation
                # second term: comfortably exceed size of largest (IGM) cells (~200 kpc for TNG100-1)
                maxdist = extent / 2 + sP.gravSoft*1000

                w_spatial = np.where(np.abs(dists) > maxdist)
                mask[w_spatial] = 0 # outside bounding box along this axis

            cell_inds = np.nonzero(mask)[0]
            print('\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%' % \
                (cell_inds.size/mask.size*100, 1/pSplit[1]*100))

            dists = None
            w_spatial = None
            mask = None

        # load (reduced) cell spatial positions
        cell_pos = sP.snapshotSubsetC('gas', 'pos', inds=cell_inds, verbose=True)

        # DEBUG only: save pos and tree cache
        if 0:
            with h5py.File('pos_cache.hdf5','w') as f:
                f['cell_pos'] = cell_pos
            print('Saved: [pos_cache.hdf5]')

            from ..util.treeSearch import buildFullTree
            NextNode, length, center, sibling, nextnode = buildFullTree(cell_pos,sP.boxSize,cell_pos.dtype)
            with h5py.File('tree_cache.hdf5','w') as f:
                f['NextNode'] = NextNode
                f['length'] = length
                f['center'] = center
                f['sibling'] = sibling
                f['nextnode'] = nextnode
            print('Saved: [tree_cache.hdf5]')
            # immediate exit
            return

        # ray-trace
        print('Load done, tracing...', flush=True)

        rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, mode='full')

    if 0:
        # DEBUG only: load caches
        print('Loading pos_cache.')
        with h5py.File('pos_cache.hdf5','r') as f:
            cell_pos = f['cell_pos'][()]

        print('Loading tree cache.')
        with h5py.File('tree_cache.hdf5','r') as f:
            NextNode = f['NextNode'][()]
            length = f['length'][()]
            center = f['center'][()]
            sibling = f['sibling'][()]
            nextnode = f['nextnode'][()]
        tree = NextNode, length, center, sibling, nextnode

        # ray-trace
        print('Load done, tracing...')
        import time
        start_time = time.time()

        rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, cell_pos, 
              mode='full', tree=tree, nThreads=1)

        print(f'Time: [{time.time()-start_time}] sec.')
        return

    # save
    with h5py.File(path, 'w') as f:
        # ray results
        f['rays_off'] = rays_off
        f['rays_len'] = rays_len
        f['rays_dl'] = rays_dl
        f['rays_inds'] = rays_inds

        # indices index a spatial subset of the snapshot
        if cell_inds is not None:
            f['cell_inds'] = cell_inds

        # ray config and metadata
        f['ray_pos'] = ray_pos

        f.attrs['nRaysPerDim'] = nRaysPerDim
        f.attrs['projAxis'] = projAxis
        f.attrs['ray_dir'] = ray_dir
        f.attrs['total_dl'] = total_dl

    print('Saved: [%s]' % path)

    return rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl

def generate_spectra_from_saved_rays(sP, pSplit=None):
    """ Generate a large number of spectra, based on already computed and saved rays.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total]. Note 
        that we follow a spatial subdivision, so the total job number should be an integer squared.
    """
    # config
    projAxis = 2
    lineNames = ['MgII 2796','MgII 2803']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'

    # save file
    linesStr = '-'.join([line.replace(' ','_') for line in lineNames])
    saveFilename = sP.derivPath + 'rays/spectra_%s_z%.1f_%d_%s_%s_%d-of-%d.hdf5' % \
      (sP.simName,sP.redshift,projAxis,instrument,linesStr,pSplit[0],pSplit[1])

    # load rays
    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = \
      generate_rays_voronoi_fullbox(sP, projAxis=projAxis, pSplit=pSplit)

    # load required gas cell properties
    velLosField = 'vel_'+['x','y','z'][projAxis]

    cell_vellos = sP.snapshotSubsetC('gas', velLosField, inds=cell_inds, verbose=True) # code
    cell_temp   = sP.snapshotSubsetC('gas', 'temp_sfcold_linear', inds=cell_inds, verbose=True) # K
    
    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos) # km/s

    # convert length units, all other units already appropriate
    rays_dl = sP.units.codeLengthToMpc(rays_dl)

    # sample master grid
    master_mid, master_edges, tau_master = create_master_grid(instrument=instrument)
    tau_master = np.zeros((rays_len.size,tau_master.size), dtype=tau_master.dtype)

    EWs = {}

    # start output
    with h5py.File(saveFilename,'w') as f:
        f['master_wave'] = master_mid

        # attach ray configuration for reference
        f['ray_pos'] = ray_pos
        f['ray_dir'] = ray_dir
        f['ray_total_dl'] = total_dl

    # loop over requested line(s)
    for line in lineNames:
        densField = '%s numdens' % lines[line]['ion']
        cell_dens = sP.snapshotSubsetC('gas', densField, inds=cell_inds, verbose=True) # ions/cm^3

        # create spectra
        master_wave, tau_local, EW_local = \
          create_spectra_from_traced_rays(sP, line, instrument, 
                                          rays_off, rays_len, rays_dl, rays_inds,
                                          cell_dens, cell_temp, cell_vellos)

        assert np.array_equal(master_wave,master_mid)

        tau_master += tau_local
        EWs[line] = EW_local

        with h5py.File(saveFilename,'r+') as f:
            # save tau per line
            f['tau_%s' % line.replace(' ','_')] = tau_local
            # save EWs per line
            f['EW_%s' % line.replace(' ','_')] = EW_local

    # calculate flux and total EW
    flux = np.exp(-1*tau_master)

    with h5py.File(saveFilename,'r+') as f:
        f['flux'] = flux

    print(f'Saved: [{saveFilename}]')

def concat_and_filter_spectra(sP):
    """ Combine split files for spectra, and filter, keeping only those above an EW threshold. """

    # config
    projAxis = 2
    lineNames = ['MgII 2796','MgII 2803']
    instrument = '4MOST_HRS' # 'SDSS-BOSS'

    EW_threshold = 0.001 # applied to sum of lines [ang]

    # search for chunks
    linesStr = '-'.join([line.replace(' ','_') for line in lineNames])
    loadFilename = sP.derivPath + 'rays/spectra_%s_z%.1f_%d_%s_%s_*.hdf5' % \
      (sP.simName,sP.redshift,projAxis,instrument,linesStr)

    saveFilename = loadFilename.replace('_*','').replace(' ','-')

    files = glob.glob(loadFilename)

    # load all for count
    inds = []
    EW_total = []

    count = 0

    for file in files:
        print(file)
        with h5py.File(file,'r') as f:
            if 'flux' not in f:
                print(' skip')
                inds.append([])
                EWs.append([])
                continue
            n_wave = f['flux'].shape[1]
            n_spec = f['flux'].shape[0]
            EW_local = np.zeros(n_spec, dtype='float32')
            for line in lineNames:
                EW_local += f['EW_%s' % line.replace(' ','_')][()]
            #flux = f['flux'][()]
            #master_mid = f['master_wave'][()]

        # recalculate EW based on total flux (e.g. combine lines/doublets)
        #EW_local = np.zeros(flux.shape[0], dtype='float32')
        #for i in range(flux.shape[0]):
        #    EW_local[i] = _equiv_width_flux(flux[i,:],master_mid)

        # select
        w = np.where(EW_local >= EW_threshold)[0]
        count += len(w)

        inds.append(w)
        EW_total.append(EW_local[w])

    print(f'In total [{count}] spectra of [{n_spec*len(files)}] above {EW_threshold = }')

    # allocate
    flux = np.zeros((count,n_wave), dtype='float32')
    ray_pos = np.zeros((count,3), dtype='float32')

    EWs = []
    for line in lineNames:
        EWs.append( np.zeros(count, dtype='float32') )

    # load
    offset = 0

    for i, file in enumerate(files):
        if len(inds[i]) == 0:
            continue
        print(file, offset)
        with h5py.File(file,'r') as f:
            flux[offset:offset+len(inds[i])] = f['flux'][inds[i]]
            ray_pos[offset:offset+len(inds[i])] = f['ray_pos'][inds[i]]
            master_wave = f['master_wave'][()]

            for j, line in enumerate(lineNames):
                EWs[j][offset:offset+len(inds[i])] = f['EW_%s' % line.replace(' ','_')][inds[i]]

        offset += len(inds[i])

    # save
    with h5py.File(saveFilename,'w') as f:
        f['master_wave'] = master_wave
        f['flux'] = flux
        f['ray_pos'] = ray_pos
        f['EW_total'] = np.hstack(EW_total)

        for i, line in enumerate(lineNames):
            f['EW_%s' % line.replace(' ','_')] = EWs[i]

        f.attrs['projAxis'] = projAxis
        f.attrs['simName'] = sP.simName
        f.attrs['redshift'] = sP.redshift
        f.attrs['snapshot'] = sP.snap
        f.attrs['instrument'] = instrument
        f.attrs['lineNames'] = lineNames
        f.attrs['EW_threshold'] = EW_threshold

    print('Saved: [%s]' % saveFilename)

def plot_concat_spectra():
    """ Debug plots for concatenated spectra. """

    # config
    filename = "spectra_TNG50-1_z0.7_2_4MOST_HRS_MgII-2796-MgII-2803.hdf5"
    path = "/u/dnelson/sims.TNG/TNG50-1/data.files/rays/"

    EW_min = 0.9
    EW_max = 1.0
    SNR = None #30.0 # if not None, add Gaussian noise for the given signal-to-noise ratio
    num = 10

    wave_minmax = [4750,4800] # z=0.5: [4180,4250]

    # load
    with h5py.File(path + filename,'r') as f:
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

def single_line_test():
    """ Test for Voigt profile deposition of a single absorption line. """

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

def test_LyA_vs_coldens():
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

def multi_line_test():
    """ Testing. """

    # transition, instrument, and spectrum type
    lines = ['LyA', 'LyB', 'LyC', 'LyD', 'LyE']
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

    for line in lines:
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
    fig.savefig('spectrum_multi_%s.pdf' % ('-'.join(lines)))
    plt.close(fig)

def benchmark_line():
    """ Deposit many random lines. """
    import time

    line = 'MgII 2803'
    instrument = None

    # parameter ranges
    n = int(1e4)
    rng = np.random.default_rng(424242)

    N_vals = rng.uniform(low=10.0, high=16.0, size=n) # log cm^-2
    b_vals = rng.uniform(low=1.0, high=25.0, size=n) # km/s
    vel_los = rng.uniform(low=-300, high=300, size=n) # km/s
    z_cosmo = 0.0

    # create master grid
    master_mid, master_edges, tau_master = create_master_grid(line=line, instrument=instrument)

    f, gamma, wave0, _, _ = _line_params(line)

    # start timer
    start_time = time.time()

    # deposit
    for i in range(n):
        # effective redshift
        z_doppler = vel_los[i] / units.c_km_s
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1 

        if i % (n/10) == 0:
            print(i, N_vals[i], b_vals[i], vel_los[i], z_eff)

        deposit_single_line(master_edges, tau_master, f, gamma, wave0, 10.0**N_vals[i], b_vals[i], z_eff)

    tot_time = time.time() - start_time
    print('depositions took [%g] sec, i.e. [%g] each' % (tot_time, tot_time/n))
