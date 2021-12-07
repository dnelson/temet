"""
Synthetic absorption spectra generation.
Inspired by SpecWizard (Schaye), pygad (Rottgers), Trident (Hummels).
"""
import numpy as np
import h5py
from os.path import isfile, expanduser
import matplotlib.pyplot as plt
from scipy.special import wofz

from plot.config import *
from util import units
from util.helper import logZeroNaN
from util.voronoiRay import trace_ray_through_voronoi_mesh_treebased, trace_ray_through_voronoi_mesh_with_connectivity

# line data (Morton+2003)
# f [dimensionless]
# gamma [1/s], where tau=1/gamma is the ~lifetime (is the sum of A)
# wave0 [ang]
lines = {'LyA'       : {'f':0.4164,   'gamma':4.49e8,  'wave0':1215.670,  'ion':'H I'},
         'LyB'       : {'f':0.0791,   'gamma':1.897e8, 'wave0':1025.7223, 'ion':'H I'},
         'LyC'       : {'f':0.0290,   'gamma':1.28e7,  'wave0':972.5367,  'ion':'H I'},
         'LyD'       : {'f':1.395e-2, 'gamma':4.12e6,  'wave0':949.7430,  'ion':'H I'},
         'LyE'       : {'f':7.803e-3, 'gamma':2.45e6,  'wave0':937.8034,  'ion':'H I'},
         'OVI 1033'  : {'f':1.983e-1, 'gamma':0.0,     'wave0':1033.8160, 'ion':'O VI'}, # (this is placeholder for doublet?)
         'OVI 1037'  : {'f':6.580e-2, 'gamma':4.076e8, 'wave0':1037.6167, 'ion':'O VI'},
         'OVI 1031'  : {'f':1.325e-1, 'gamma':4.149e8, 'wave0':1031.9261, 'ion':'O VI'},
         'MgII 2803' : {'f':0.6155,   'gamma':2.592e8, 'wave0':2803.5315, 'ion':'Mg II'},
         'MgII 2796' : {'f':0.3058,   'gamma':2.612e8, 'wave0':2796.3543, 'ion':'Mg II'}}

# instrument characteristics (in Ang)
instruments = {'COS-G130M'  : {'wave_min':1150, 'wave_max':1450, 'dwave':0.01},
               'COS-G140L'  : {'wave_min':1130, 'wave_max':2330, 'dwave':0.08},
               'COS-G160M'  : {'wave_min':1405, 'wave_max':1777, 'dwave':0.012},
               'test_EUV'   : {'wave_min':800,  'wave_max':1300, 'dwave':0.1}, # to see LySeries at rest
               'test_EUV2'  : {'wave_min':1200, 'wave_max':2000, 'dwave':0.1}, # testing redshift shifts
               'SDSS-BOSS'  : {'wave_min':3600, 'wave_max':10400, 'dwave':1.0}, # dwave approx, constant in log(dwave) only
               'MIKE'       : {'wave_min':3350, 'wave_max':9500, 'dwave':0.07}, # approximate only
               'KECK-HIRES' : {'wave_min':3000, 'wave_max':9250, 'dwave':0.04}} # approximate only

def _line_params(line):
    """ Return 5-tuple of (f,Gamma,wave0,ion_amu,ion_mass). """
    from cosmo.cloudy import cloudyIon

    mass_proton = 1.672622e-24 # cgs

    element = lines[line]['ion'].split(' ')[0]
    ion_amu = {el['symbol']:el['mass'] for el in cloudyIon._el}[element]
    ion_mass = ion_amu * mass_proton # g

    return lines[line]['f'], lines[line]['gamma'], lines[line]['wave0'], ion_amu, ion_mass

def _voigt0(wave_cm, b, wave0_ang, gamma):
    """ Dimensionless Voigt profile (shape).

    Args:
      wave_cm (array[float]): wavelength grid in [cm] where the profile is calculated.
      b (float): doppler parameter in km/s.
      wave0_ang (float): central wavelength of transition in angstroms.
      gamma (float): sum of transition probabilities (Einstein A coefficients).
    """
    nu = units.c_cgs / wave_cm # wave = c/nu
    wave_rest = wave0_ang * 1e-8 # angstrom -> cm
    nu0 = units.c_cgs / wave_rest # Hz
    b_cgs = b * 1e5 # km/s -> cm/s
    dnu = b_cgs / wave_rest # Hz, "doppler width" = sigma/sqrt(2)

    # use Faddeeva for integral
    alpha = gamma / (4*np.pi*dnu)
    voigt_u = (nu - nu0) / dnu # z

    voigt_wofz = wofz(voigt_u + 1j*alpha).real # H(alpha,z)

    phi_wave = voigt_wofz / b_cgs # s/cm
    return phi_wave

def _voigt_tau(wave, N, b, wave0, f, gamma, wave0_rest=None):
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
    """
    if wave0_rest is None:
        wave0_rest = wave0

    # get dimensionless shape for voigt profile:
    wave_cm = wave * 1e-8
    phi_wave = _voigt0(wave_cm, b, wave0, gamma)

    consts = 0.014971475 # sqrt(pi)*e^2 / m_e / c = cm^2/s
    wave0_rest_cm = wave0_rest * 1e-8

    tau_wave = (consts * N * f * wave0_rest_cm) * phi_wave
    return tau_wave

def _equiv_width(tau,wave_ang):
    """ Compute the equivalent width by integrating the optical depth array across the given wavelength grid. """
    dang = wave_ang[1] - wave_ang[0]
    integrand = 1 - np.exp(-tau)

    # integrate (1-exp(-tau_lambda)) d_lambda from 0 to inf, composite trap rule
    res = dang / 2 * (integrand[0] + integrand[-1] + np.sum(2*integrand[1:-1]))

    return res

def curveOfGrowth(line='MgII 2803'):
    """ Plot relationship between EW and N for a given transition (e.g. HI, MgII line). """
    f, gamma, wave0_ang, _, _ = _line_params(line)

    # run config
    nPts = 201
    wave_ang = np.linspace(wave0_ang-5, wave0_ang+5, nPts)
    dvel = (wave_ang/wave0_ang - 1) * units.c_cgs / 1e5 # cm/s -> km/s
    
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
            tau = _voigt_tau(wave_ang/1e8, col, b, wave0_ang, f, gamma)
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
    if line is not None:
        f, gamma, wave0_restframe, _, _ = _line_params(line)

    # master wavelength grid, observed-frame [ang]
    if line is not None:
        wave_min = np.floor(wave0_restframe - 15.0)
        wave_max = np.ceil(wave0_restframe + 15.0)
        dwave = 0.1

    if instrument is not None:
        wave_min = instruments[instrument]['wave_min']
        wave_max = instruments[instrument]['wave_max']
        dwave = instruments[instrument]['dwave']

    num_edges = int(np.floor((wave_max - wave_min) / dwave)) + 1
    wave_edges = np.linspace(wave_min, wave_max, num_edges)
    wave_mid = (wave_edges[1:] + wave_edges[:-1]) / 2

    tau_master = np.zeros(wave_mid.size, dtype='float32')

    return wave_mid, wave_edges, tau_master

def deposit_single_line(wave_edges_master, tau_master, f, gamma, wave0, N, b, z_eff, debug=False):
    """ Add the absorption profile of a single transition, from a single cell, to a spectrum.

    Args:
      wave_edges_master (array[float]): bin edges for master spectrum array.
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
        return # empty

    # local (to the line), rest-frame wavelength grid
    dwave_local = 0.01 # ang
    edge_tol = 1e-4 # if the optical depth is larger than this by the edge of the local grid, redo

    b_dwave = b / units.c_km_s * wave0 # v/c = dwave/wave

    # adjust local resolution to make sure we sample narrow lines
    while b_dwave < dwave_local * 4:
        dwave_local *= 0.5

        if dwave_local < 1e-4:
            assert 0 # check
            break

    # prep local grid
    wave0_obsframe = wave0 * (1 + z_eff)

    line_width_safety = b / units.c_km_s * wave0_obsframe

    dwave_master = wave_edges_master[1] - wave_edges_master[0]
    nloc_per_master = int(np.round(dwave_master / dwave_local))

    n_iter = 0
    local_fac = 5.0
    tau = [np.inf]

    while tau[0] > edge_tol or tau[-1] > edge_tol:
        # determine where local grid overlaps with master
        wave_min_local = wave0_obsframe - local_fac*line_width_safety
        wave_max_local = wave0_obsframe + local_fac*line_width_safety

        master_inds = np.searchsorted(wave_edges_master, [wave_min_local,wave_max_local])
        master_startind = master_inds[0] - 1
        master_finalind = master_inds[1]

        # sanity checks
        if master_startind == -1:
            if debug:
                print('WARNING: min edge of local grid hit edge of master!')
            master_startind = 0

        if master_finalind == wave_edges_master.size:
            if debug:
                print('WARNING: max edge of local grid hit edge of master!')
            master_finalind = wave_edges_master.size - 1

        if master_startind == master_finalind:
            if n_iter < 20:
                # extend, see if wings of this feature will enter master spectrum
                local_fac *= 1.2
                n_iter += 1
                continue

            if debug:
                print('WARNING: absorber entirely off edge of master spectrum! skipping!')
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
        tau = _voigt_tau(wave_mid_local, N, b, wave0_obsframe, f, gamma, wave0_rest=wave0)

        # iterate and increase wavelength range of local grid if the optical depth at the edges is still large
        if debug:
            print(f'{local_fac = }, {tau[0] = :.3g}, {tau[-1] = :.3g}, {edge_tol = }')

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

        print(f'{EW_local = :.6f}, {EW_master = :.6f}, {tau_local_tot = :.5f}, {tau_master_tot = :.5f}')

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
    """ Given a completed ray traced through a volume, and the properties of all the intersected cells 
    (dens, dx, temp, vellos), create the final absorption spectrum, depositing a Voigt absorption 
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

def generate_spectrum_uniform_grid():
    """ Generate an absorption spectrum by ray-tracing through a uniform grid (deposit using sphMap). """
    from util.simParams import simParams
    from util.sphMap import sphGridWholeBox, sphMap
    from cosmo.cloudy import cloudyIon

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1031' #'LyA'
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
    from util.simParams import simParams
    from util.voronoi import loadSingleHaloVPPP, loadGlobalVPPP
    from cosmo.cloudy import cloudyIon
    from util.treeSearch import buildFullTree

    # config
    sP = simParams(run='tng50-4', redshift=0.5)

    line = 'OVI 1031' #'LyA'
    instrument = 'test_EUV2' # 'SDSS-BOSS'
    haloID = 150 # if None, then full box

    ray_offset_x = 0.0 # relative to halo center, in units of rvir
    ray_offset_y = 0.5 # relative to halo center, in units of rvir
    ray_offset_z = -2.0 # relative to halo center, in units of rvir
    projAxis = 2 # z, to simplify vellos for now

    fof_scope_mesh = True

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
        master_dens, master_dx, master_temp, master_vellos = \
          trace_ray_through_voronoi_mesh_with_connectivity(cell_pos, cell_vellos, cell_temp, cell_dens, 
                                       num_ngb, ngb_inds, offset_ngb, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify, fof_scope_mesh)

    if (not use_precomputed_mesh) or compare:
        # construct neighbor tree
        tree = buildFullTree(cell_pos, boxSizeSim=sP.boxSize, treePrec=cell_pos.dtype, verbose=debug)
        NextNode, length, center, sibling, nextnode = tree

        if compare:
            ray_pos = np.array([ray_start_x, ray_start_y, ray_start_z]) # reset
            master_dens2 = master_dens.copy()
            master_dx2 = master_dx.copy()
            master_temp2 = master_temp.copy()
            master_vellos2 = master_vellos.copy()

        # ray-trace
        master_dx, master_dens, master_temp, master_vellos = \
          trace_ray_through_voronoi_mesh_treebased(cell_pos, cell_dens, cell_temp, cell_vellos, 
                                       NextNode, length, center, sibling, nextnode, ray_pos, ray_dir, total_dl, 
                                       sP.boxSize, debug, verify)

        if compare:
            assert np.allclose(master_dens2,master_dens)
            assert np.allclose(master_dx,master_dx2)
            assert np.allclose(master_temp,master_temp2)
            assert np.allclose(master_vellos,master_vellos2)
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

def single_line_test():
    """ Testing.

    Args:
      mode (str): 'local' or 'box' or 'lightcone'. If 'local', then we neglect sP.redshift i.e. the 
      cosmological redshift of the box, as well as the location of the gas within the box, and simply 
      produce an absorption spectrum around the rest-frame wavelength of the transition (including only 
      peculiar velocities). If 'box', the transition is cosmologically redshifted to sP.redshift in 
      addition to the pathlength through the box, which is treated as a periodic cube. If 'lightcone', 
      then use a pre-existing lightcone geometry (e.g. for a very long pathlength), taking the redshift 
      of each gas cell as input.
    """

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
    from util.helper import sampleColorTable, loadColorTable

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
