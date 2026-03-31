"""
Utilities for analysis of turbulence.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic
from numba import njit
import sys
from numba.typed import List


def _to_uniform_grid(quant, coordinates, grid_size=100):
    """Map a quantity that is defined on an irregular grid to a uniform grid based on nearest neighbor sampling.
    """
    # Setup a grid of evenly spaced points
    mins = coordinates.min(axis=0)
    maxs = coordinates.max(axis=0)
    lengths = maxs - mins
    L = lengths.max()
    center = (mins + maxs) / 2
    cube_mins = center - L / 2
    cube_maxs = center + L / 2
    x = np.linspace(cube_mins[0], cube_maxs[0], grid_size)
    y = np.linspace(cube_mins[0], cube_maxs[0], grid_size)
    z = np.linspace(cube_mins[0], cube_maxs[0], grid_size)

    xx, yy, zz = np.meshgrid(x, y, z)
    grid = np.vstack((xx.ravel(), yy.ravel(), zz.ravel())).T

    # Value of the quantity at grid point is given by the closest point of the irregular grid, i.e. if it is inside its 
    # voronoi cell if the coordinates are generating points for the voronoi tesselation
    tree = cKDTree(coordinates)
    _, idx = tree.query(grid, k=1)
    print(quant.shape)
    if len(quant.shape) > 1:
        return quant[idx].reshape(grid_size, grid_size, grid_size, 3)
    else:
        return quant[idx].reshape(grid_size, grid_size, grid_size)


def _uniform_power_spectrum_1d(quant, boxsize=1.0, nBins=200):
    """Calculate the 1d power spectrum of a periodic quantity on a uniform grid. This method is based on the description 
    in section 2.5 of Bauer & Springel (2013) (https://arxiv.org/abs/1109.4413). 
    The 3d power spectrum is given by the absolute value squared of the fourier transformed quantity:

        E_3d(k) = (2pi/L)^3 * abs(F[quantity])^2.

    If the quantity is distributed isotropically, the 1d power spectrum is obtained by averaging over radial shells in 
    k-space:
    
        E(k) = 4pi* k^2 * <E_3d(k)>. 
    
    It is calculated below as the mean in logarithmically spaced bins.
    """
    N = quant.shape[0]

    quant_k = np.fft.fftn(quant)
    if len(quant.shape) > 3:
        power = np.sum(np.abs(quant_k)**2, axis=-1)
    else:
        power = np.abs(quant_k)**2
    power *= (2 * np.pi/boxsize)**3

    k = np.fft.fftfreq(N, d=boxsize/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    power = power.ravel()

    k_nonzero = k_mag[k_mag > 0]
    k_min = k_nonzero.min()
    k_max = k.max()

    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), nBins + 1)

    Pk, _, _ = binned_statistic(k_mag, power, bins=bin_edges, statistic='mean')
    k_bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    Pk *= 4 * np.pi * k_bin_centers**2
    Pk[np.isnan(Pk)] = 0
    return k_bin_centers, Pk


def power_spectrum_1d(quant, coordinates, boxsize=1.0, nBins=200, uniform_gridsize=200):
    """Calculate the 1d power spectrum of a periodic quantity on a non uniform grid such as the output data of 
    simulations based on Arepo. This method is based on the description in section 2.5 of Bauer & Springel (2013) (https://arxiv.org/abs/1109.4413). 
    First, the quantity is mapped to a uniform grid, from which the Fourier transform can be used to calculate the power spectrum.

    Args:
      quant (ndarray of float64, shape (n, 3) or (n,)): A 3d vector or scalar quantity given at coordinates in 3d space.
      coordinates (ndarray of float64, shape (n, 3)): The coordinates where the quantity is given.
      boxsize (float): Size of the simulation box.
      nBins (int): Number of radial, logarithmically spaced bins in k-space. 
      uniform_gridsize (int): Size of the uniform grid for nearest neighbor sampling.  
    """
    quant_uniform = _to_uniform_grid(quant, coordinates, grid_size=uniform_gridsize)
    return _uniform_power_spectrum_1d(quant_uniform, boxsize=boxsize, nBins=nBins)


def _uniform_helmholtz_decomposition(F, boxsize=1.0):
    """Calculate the helmholtz decomposition of a periodic 3d vector field. In Fourier space, helmholtz decomposition 
    corresponds to decomposing the vector field in components parallel and perpendicular to the wave vectors k. 
    """
    N = F.shape[0]
    
    Fx_k = np.fft.fftn(F[..., 0])
    Fy_k = np.fft.fftn(F[..., 1])
    Fz_k = np.fft.fftn(F[..., 2])
    
    k = np.fft.fftfreq(N, d=boxsize/N) * 2 * np.pi
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[0, 0, 0] = 1.0  # prevent division by zero at k = 0
    k_dot_F = kx * Fx_k + ky * Fy_k + kz * Fz_k

    # the curlfree or compressive part of the vector field is the projection along the k-vectors
    Fx_curlfree_k = k_dot_F * kx / k_squared
    Fy_curlfree_k = k_dot_F * ky / k_squared
    Fz_curlfree_k = k_dot_F * kz / k_squared

    # k = (0, 0, 0) corresponds to the bulk component which has vanishing curl and divergence. If the vector field has no bulk
    # part there is no issue. The code below assigns the bulk part to the divergence free part of the decomposition.
    Fx_curlfree_k[0, 0, 0] = 0
    Fy_curlfree_k[0, 0, 0] = 0
    Fz_curlfree_k[0, 0, 0] = 0

    Fx_curlfree = np.real(np.fft.ifftn(Fx_curlfree_k))
    Fy_curlfree = np.real(np.fft.ifftn(Fy_curlfree_k))
    Fz_curlfree = np.real(np.fft.ifftn(Fz_curlfree_k))

    F_curl_free = np.stack((Fx_curlfree, Fy_curlfree, Fz_curlfree), axis=-1)
    # curl free and divergence free part sum up to the total vector field since the projection in Fourier space and the 
    # Fourier transform are linear
    F_div_free = F - F_curl_free
    return F_div_free, F_curl_free


def helmholtz_decomposition(quant, coordinates, boxsize=1.0, uniform_gridsize=200):
    """Calculate the helmholtz decomposition of a periodic vector field on a non uniform grid such as the output data of 
    simulations based on Arepo. First, the quantity is mapped to a uniform grid, from which the Fourier transform can be 
    used to calculate the Helmholtz decomposition.

    Args:
      quant (ndarray of float64, shape (n, 3)): A 3d vector quantity given at coordinates in 3d space.
      coordinates (ndarray of float64, shape (n, 3)): The coordinates where the quantity is given.
      boxsize (float): Size of the simulation box.
      uniform_gridsize (int): Size of the uniform grid for nearest neighbor sampling.
    """
    quant_uniform = _to_uniform_grid(quant, coordinates, grid_size=uniform_gridsize)
    return _uniform_helmholtz_decomposition(quant_uniform, boxsize=boxsize)


def _update_progress(fraction_done, bar_length=40):
    """A small progress bar function.
    """
    percent = int(fraction_done * 100)
    filled_len = int(bar_length * fraction_done)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)
    sys.stdout.write(f'\r[{bar}] {percent:3d}%')
    sys.stdout.flush()


@njit
def _init_mean_vels(N, velx, vely, velz, inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, masses):
    """Function that initializes the turbulent velocities.
    """
    for i in range(N):
        bulkx = np.sum(velx[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])
        bulky = np.sum(vely[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])
        bulkz = np.sum(velz[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])
        previous_turbulentx[i] = velx[i] - bulkx
        previous_turbulenty[i] = vely[i] - bulky
        previous_turbulentz[i] = velz[i] - bulkz
        current_width[i] = max(current_width[i] + cell_radii[i], 1.05 * current_width[i])


@njit
def _calculate_mean_vels(N, velx, vely, velz, inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, shocks, coherence_length, masses):
    """Function that calculates the turbulent velocity decomposition based on the local bulk velocity. Numba is used to
    speed up the calculations.
    """
    for i in range(N):
        # if inds[i] contains less than two cell indices, the size of the spherical shell around the cell was set to zero
        # after convergence was reached for the cells turbulent velocity
        if len(inds[i]) < 2:
            continue
        # calculate mass weighted mean velocity of all cells that intersect with the spherical volume
        bulkx = np.sum(velx[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])
        bulky = np.sum(vely[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])
        bulkz = np.sum(velz[inds[i]] * masses[inds[i]])/np.sum(masses[inds[i]])

        # calculate the fractional change between new and old turbulent velocity, the iteration is stopped either if the 
        # fractional change is below 0.05 or a cell with mach number greater than 1.3 enters the spherical volume 
        if (abs(previous_turbulentx[i]) < 1e-10) or (abs(previous_turbulenty[i]) < 1e-10) or (abs(previous_turbulentz[i]) < 1e-10):
            delta = np.inf
        else:
            delta = np.max(np.abs(np.array([(velx[i] - bulkx)/previous_turbulentx[i], (vely[i] - bulky)/previous_turbulenty[i], (velz[i] - bulkz)/previous_turbulentz[i]]) - 1))
        shocked = False
        for l in inds[i]:
            if shocks[l] > 1.3:
                shocked = True
                break
        if (delta < 0.05) or shocked:
            # the convergence is reached. The coherence length corresponds to the radius of the spherical volume where
            # the iteration stopped. The current width is set to zero to prevent further calculation in the iteration
            coherence_length[i] = current_width[i]
            current_width[i] = 0.0
        else:
            # the convergence is not reached and the radius of the spherical volume is increased 
            current_width[i] = max(current_width[i] + cell_radii[i], 1.05 * current_width[i])

        previous_turbulentx[i] = velx[i] - bulkx
        previous_turbulenty[i] = vely[i] - bulky
        previous_turbulentz[i] = velz[i] - bulkz


def multiscale_filtering_velocity_decomposition(positions, velocities, masses, density, shocks, showProgress=False, workers=8):
    """An implementation of the multiscale velocity decomposition algorithm described in section 2.2 of 
    Vallés-Pérez et al. 2021 (https://arxiv.org/abs/2103.13449) for IllustrisTNG simulation outputs. The function takes 
    the gas velocities and separates the local velocity fluctuations from the bulk velocity. Instead of calculating
    the bulk velocity as a local average based on a fixed smoothing scale, the smoothing scale is iteratively increased
    until the turbulent velocity is converged. 

    Args:
      positions (ndarray of float64, shape (n, 3)): 'Coordinates' field of IllustrisTNG output.
      velocities (ndarray of float64, shape (n, 3)): 'Velocities' field of IllustrisTNG output or any other 3d vector.
      masses (ndarray of float64, shape (n,)): 'Masses' field of IllustrisTNG output.
      density (ndarray of float64, shape (n,)): 'Density' field of IllustrisTNG output.
      shocks (ndarray of float64, shape (n,)): 'Machnumber' field of IllustrisTNG output.
      showProgress (bool): If not False, show progress bar in command line.
      workers (int): Number of threads for parallel work, searching for the neigboring cells.
    """
    N = positions.shape[0]
    vols = masses/density
    # start the iteration by computing the bulk velocity within a spherical volume with radius three times the radius of
    # of the cell if it was spherical.
    cell_radii = (3/(4*np.pi) * vols)**(1/3)
    current_width = 3 * cell_radii
    velx = velocities[:, 0]
    vely = velocities[:, 1]
    velz = velocities[:, 2]
    previous_turbulentx = np.zeros_like(velx)
    previous_turbulenty = np.zeros_like(velx)
    previous_turbulentz = np.zeros_like(velx)
    # coherence scale corresponds to the radius of the spherical volume where the turbulent velocity is converged
    coherence_scale = np.zeros(len(positions))

    # use scipy.spatial.ckDTree for fast lookup of all the other cells that are within the smoothing radius
    tree = cKDTree(positions)
    inds = tree.query_ball_point(positions, current_width, workers=workers)
    # inds[i] contains the indices of the cells that are present in the spherical volume around cell i
    # convert this list to a numba list to be able to use it in a function with @njit later
    numba_inds = List()
    for ind_list in inds:
        numba_inds.append(np.array(ind_list))

    _init_mean_vels(N, velx, vely, velz, numba_inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, masses)

    # iteration: if convergence is reached, current_width is set to zero in _calculate_mean_vels, therefore all cells 
    # have reached convergence if the sum of all is zero
    while np.sum(current_width) > 1e-10:
        if showProgress:
            _update_progress(np.sum(current_width == 0.0)/len(positions))
        inds = tree.query_ball_point(positions, current_width, workers=8)
        numba_inds = List()
        for ind_list in inds:
            numba_inds.append(np.array(ind_list))
        _calculate_mean_vels(N, velx, vely, velz, numba_inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, shocks, coherence_scale, masses)
    
    previous_turbulent = np.column_stack((previous_turbulentx, previous_turbulenty, previous_turbulentz))
    return velocities - previous_turbulent, previous_turbulent, coherence_scale
