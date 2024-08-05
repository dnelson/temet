"""
Synthetic absorption spectra: analysis and derived quantities.
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from os.path import isfile
from numba import jit

from ..util.helper import contiguousIntSubsets
from ..cosmo.spectrum import generate_rays_voronoi_fullbox, integrate_along_saved_rays, \
    create_wavelength_grid, _spectra_filepath, lines
from ..plot.config import *

def absorber_catalog(sP, ion, instrument, solar=False):
    """ Detect and chatacterize absorbers, handling the possibility of one or possibly multiple per sightline, 
    defined as separated but contiguous regions of absorption. Create an absorber catalog, i.e. counts and 
    offsets per sightline, and compute their EWs.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
    """
    # config
    EW_threshold = 1e-3 # observed frame [ang]

    # lines of this ion
    lineNames = [k for k,v in lines.items() if lines[k]['ion'] == ion] # all transitions of this ion

    loadFilename = _spectra_filepath(sP, ion, instrument=instrument, solar=solar)
    saveFilename = loadFilename.replace('_combined','_abscat')

    # dicts (to load, one entry per line)
    tau = {}

    # dicts (to be generated)
    counts = {} # per spectrum (number of detected individual absorbers)
    offset = {} # per spectrum

    EWs       = {} # EW of this detected individual absorbers
    ind_spec  = {} # spectrum index of this absorber
    ind_start = {} # start index in the spectrum of this absorber
    ind_stop  = {} # stop index in the spectrum of this absorber

    # save file already exists? then load now and return
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for line in lineNames:
                EWs[line]  = f['EWs/'+line][()]
                counts[line] = f['counts/'+line][()]
                offset[line] = f['offset/'+line][()]
                ind_spec[line]  = f['ind_spec/'+line][()]
                ind_start[line] = f['ind_start/'+line][()]
                ind_stop[line]  = f['ind_stop/'+line][()]

        print('Loaded: [%s]' % saveFilename)

        return EWs, counts, offset, ind_spec, ind_start, ind_stop

    # loop over possible transitions
    for line in lineNames:

        with h5py.File(loadFilename,'r') as f:
            # load metadata
            wave = f['wave'][()]
            dang = np.abs(np.diff(wave)) # wavelength bin size

            key = 'EW_%s' % line.replace(' ','_')
            if key not in f:
                print(f'[{line}] skipping, not present.')
                continue

            # load original EWs: search above threshold (informational only)
            EWs_orig = f[key][()]

            for thresh in [1e-4, 1e-3, 1e-2, 1e-1]:
                count = np.count_nonzero(EWs_orig > thresh)
                print('[%s] [%s] have [%d] of [%d] above EW > %g.' % (sP,line,count,EWs_orig.size,thresh))

            # load spectra
            key = 'tau_%s' % line.replace(' ','_')
            tau[line] = f[key][()]

        # allocate for processed results
        nspec = tau[line].shape[0]
        count_abs = 0

        counts[line] = np.zeros(nspec, dtype='int32')

        EWs[line] = np.zeros(nspec*10, dtype='float32') # space for >1 absorber per sightline
        EWs[line].fill(np.nan)
        ind_spec[line]  = np.zeros(nspec*10, dtype='int32') - 1 # spectrum index of this absorber
        ind_start[line] = np.zeros(nspec*10, dtype='int32') - 1 # starting index in this spectrum
        ind_stop[line]  = np.zeros(nspec*10, dtype='int32') - 1 # ending index in this spectrum
        
        # loop over spectra, find deviations from tau==0, find contiguous regions, compute EW in each
        for i in range(nspec):
            if i % int(nspec/10) == 0: print(' %.1f%%' % (i/nspec*100), flush=True)
            # single spectrum
            local_tau = tau[line][i,:].flatten()

            # non-zero optical depth regions (i.e. normalized flux less than unity)
            local_tau_nonzero_inds = np.where(local_tau > 0)[0]

            # find contiguous tau > 0 regions
            ranges = contiguousIntSubsets(local_tau_nonzero_inds)

            counts[line][i] = len(ranges)
            
            # loop over each contiguous range
            for i_start,i_stop in ranges:
                # index range
                local_inds = local_tau_nonzero_inds[i_start:i_stop]

                ind_spec[line][count_abs]  = i
                ind_start[line][count_abs] = local_inds[0]
                ind_stop[line][count_abs]  = local_inds[-1]

                # save EW of this single absorption feature
                integrand = 1 - np.exp(-local_tau[local_inds])
                EWs[line][count_abs] = np.sum(dang[local_inds[:-1]] * (integrand[1:] + integrand[:-1])/2)

                count_abs += 1

        # sanity checks, and reduce arrays to used size
        assert count_abs == counts[line].sum()

        EWs[line] = EWs[line][0:count_abs]
        ind_spec[line]  = ind_spec[line][0:count_abs]
        ind_start[line] = ind_start[line][0:count_abs]
        ind_stop[line]  = ind_stop[line][0:count_abs]

        # create offsets
        offset[line] = np.zeros(nspec, dtype='int32')
        offset[line][1:] = np.cumsum(counts[line])[:-1]

    # open file for writing
    fOut = h5py.File(saveFilename,'w')

    # load subset
    with h5py.File(loadFilename,'r') as f:
        # copy metadata
        for attr in f.attrs:
            fOut.attrs[attr] = f.attrs[attr]
    fOut.attrs['EW_threshold'] = EW_threshold

    # loop over lines
    for line in lineNames:
        fOut['EWs/%s' % line] = EWs[line]
        fOut['counts/%s' % line] = counts[line]
        fOut['offset/%s' % line] = offset[line]
        fOut['ind_spec/%s' % line] = ind_spec[line]
        fOut['ind_start/%s' % line] = ind_start[line]
        fOut['ind_stop/%s' % line] = ind_stop[line]

    fOut.close()

    print('Saved: [%s]' % saveFilename)

    return EWs, counts, offset, ind_spec, ind_start, ind_stop

@jit(nopython=True, nogil=True, cache=False)
def _cell_absorber_map(wave0, abs_counts, abs_offset, abs_ind_spec, abs_ind_start, abs_ind_stop,
                       rays_off, rays_len, rays_cell_dl, rays_cell_inds, ray_offset,
                       cell_vellos, cell_dens, z_vals, z_lengths, inst_waveedges):
    """ JITed helper (see below). """
    n_rays = rays_len.size
    n_cells = cell_vellos.size

    sP_units_Mpc_in_cm = 3.08568e24
    sP_units_c_km_s = 2.9979e5

    scalefac = 1/(1+z_vals[0])

    # how many absorbers in these rays?
    n_abs = np.sum(abs_counts[ray_offset:ray_offset+n_rays])
    abs_count = 0

    # allocate: ordered list of gas cell indices that contribute to each absorber (length, offset style)
    global_count = 0
    spec_index = np.zeros(int(n_abs * n_cells**(1/3)), dtype=np.int32) - 1 # heuristic size

    # allocate: length and offset list of cells, per absorber
    counts_abs = np.zeros(n_abs, dtype=np.int32)
    offset_abs = np.zeros(n_abs, dtype=np.int32)
    coldens_abs = np.zeros(n_abs, dtype=np.float32)
    colfrac_abs = np.zeros(n_abs, dtype=np.float32)

    # loop over rays
    for i in range(n_rays):
        # if the spectrum of this ray has no absorbers, skip
        global_ray_index = ray_offset + i

        if abs_counts[global_ray_index] == 0:
            continue

        # get properties of cells intersected by this ray
        offset = rays_off[i] # start of intersected cells (in rays_cell*)
        length = rays_len[i] # number of intersected gas cells

        master_dx = rays_cell_dl[offset:offset+length]
        master_inds = rays_cell_inds[offset:offset+length]

        master_vellos = cell_vellos[master_inds]
        master_dens = cell_dens[master_inds]

        # cumulative pathlength, Mpc from start of box i.e. start of ray (at sP.redshift)
        cum_pathlength = np.zeros(length, dtype=np.float32) 
        cum_pathlength[1:] = np.cumsum(master_dx)[:-1] # pMpc
        cum_pathlength /= scalefac # input in pMpc, convert to cMpc

        # cosmological redshift of each intersected cell
        z_cosmo = np.interp(cum_pathlength, z_lengths, z_vals)

        # doppler shift
        z_doppler = master_vellos / sP_units_c_km_s

        # effective redshift
        z_eff = (1+z_doppler)*(1+z_cosmo) - 1

        # observed-frame (central) wavelength
        wave0_obsframe = wave0 * (1 + z_eff)

        # index in the spectrum where each cell will center its absorption
        spec_index_loc = np.searchsorted(inst_waveedges, wave0_obsframe)

        # gives the index (of the wave edges) beyond the absorption wavelength, shift back by one
        # such that these indices correspond to the wave mid (and so abs start/stop) array indices
        spec_index_loc -= 1

        # column density
        N = master_dens * (master_dx * sP_units_Mpc_in_cm) # cm^-2

        # loop over all absorbers in this ray
        for j in range(abs_counts[global_ray_index]):
            # each absorber has a start index and stop index
            abs_index = abs_offset[global_ray_index] + j
            ind_start_abs = abs_ind_start[abs_index]
            ind_stop_abs = abs_ind_stop[abs_index]

            assert abs_ind_spec[abs_index] == global_ray_index, 'Error: Absorber index does not match ray index.'

            # locate gas cells whose central absorption is within this absorber
            w = np.where((spec_index_loc >= ind_start_abs) & (spec_index_loc <= ind_stop_abs))[0]

            # calculate total column density
            N_abs = N[w]
            N_abs_tot = np.sum(N_abs)

            # sort the individual column density contributions from each cell
            sort_inds = np.argsort(N_abs)[::-1]
            
            # fractional contribution of each cell to the total
            N_abs_frac = N_abs[sort_inds] / N_abs_tot

            # cumulative sum, normalized to fraction of the total
            N_abs_cum = np.cumsum(N_abs[sort_inds]) / N_abs_tot
            
            # identify the cells required such that we reach 99% of the total column density
            # and include all cells with fractional contribution above 1e-3
            thresh1 = 0.99
            thresh2 = 1e-3

            for k in range(N_abs_cum.size):
                if N_abs_cum[k] > thresh1 and N_abs_frac[k] < thresh2:
                    break

            # store the column density of this absorber, and the fraction that we are including
            coldens_abs[abs_count] = N_abs_tot
            colfrac_abs[abs_count] = N_abs_cum[k]

            # store the indices of these cells that 'significantly' contribute to this absorber
            loc_count = k+1
            spec_index[global_count:global_count+loc_count] = master_inds[sort_inds[0:k+1]]

            counts_abs[abs_count] = loc_count # of gas cell inds
            global_count += loc_count # of gas cell inds
            abs_count += 1

    # create offsets and reduce spec_index to used size
    offset_abs[1:] = np.cumsum(counts_abs)[:-1]

    assert spec_index[global_count-1] != -1, 'Error: Last entry in spec_index is not set.'
    assert spec_index[global_count] == -1, 'Error: Past end of spec_index is not -1.'

    spec_index = spec_index[0:global_count]

    return spec_index, counts_abs, offset_abs, coldens_abs, colfrac_abs

def cell_to_absorber_map(sP, ion, instrument, solar=False):
    """ For each absorber, defined as a contiguous set of pixels in a spectrum, identify the 
    gas cells that contribute to this absorber. Return 'spec_index' is an ordered list of 
    global gas cell indices, stored with a length (counts_abs) and offset (offset_abs) approach, 
    where the length and offset are per absorber. Also derived and returned: the column 
    density per absorber, and the fraction of this (total) column density that is recovered 
    by the cells included here as 'significantly' contributing to the absorber.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
      instrument (str): specify wavelength range and resolution, must be known in `instruments` dict.
      solar (bool): if True, do not use simulation-tracked metal abundances, but instead 
        use the (constant) solar value.
    """
    # lines of this ion
    lineNames = [k for k,v in lines.items() if lines[k]['ion'] == ion] # all transitions of this ion
    
    loadFilename = _spectra_filepath(sP, ion, instrument=instrument, solar=solar)
    saveFilename = loadFilename.replace('_combined','_abscellmap')

    pSplitNum = 16 # hack, no easy way to get this from the combined spectra file

    # save file already exists? then load now and return
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            spec_index = {line:f['spec_index/'+line][()] for line in f['spec_index'].keys()}
            counts_abs = {line:f['counts_abs/'+line][()] for line in f['counts_abs'].keys()}
            offset_abs = {line:f['offset_abs/'+line][()] for line in f['offset_abs'].keys()}
            coldens_abs = {line:f['coldens_abs/'+line][()] for line in f['coldens_abs'].keys()}
            colfrac_abs = {line:f['colfrac_abs/'+line][()] for line in f['colfrac_abs'].keys()}

        print('Loaded: [%s]' % saveFilename)

        return spec_index, counts_abs, offset_abs, coldens_abs, colfrac_abs

    # load absorber catalog
    _, abs_counts, abs_offset, abs_ind_spec, abs_ind_start, abs_ind_stop = \
        absorber_catalog(sP, ion, instrument, solar=solar)
    
    # sample instrumental grid
    _, inst_waveedges, _ = create_wavelength_grid(instrument=instrument)

    # allocate
    spec_index = {}
    counts_abs = {}
    offset_abs = {}
    coldens_abs = {}
    colfrac_abs = {}

    for line in lineNames:
        n_abs = abs_counts[line].sum() # across all splits

        spec_index[line] = np.zeros(int(n_abs * sP.res), dtype='int32') - 1 # heuristic size
        counts_abs[line] = np.zeros(n_abs, dtype='int32')
        offset_abs[line] = np.zeros(n_abs, dtype='int32')
        coldens_abs[line] = np.zeros(n_abs, dtype='float32')
        colfrac_abs[line] = np.zeros(n_abs, dtype='float32')

    spec_w_off = {line:0 for line in lineNames}
    abs_w_off = {line:0 for line in lineNames}

    # assign sP.redshift to the front intersection (beginning) of the box
    z_vals = np.linspace(sP.redshift, sP.redshift+0.1, 200)
    assert sP.boxSize < 40000, 'Increase 0.1 factor above for boxes larger than TNG50.'

    z_lengths = sP.units.redshiftToComovingDist(z_vals) - sP.units.redshiftToComovingDist(sP.redshift)

    # load gas cells
    projAxis = 2 #list(ray_dir).index(1)
    cell_vellos_glob = sP.snapshotSubsetP('gas', 'vel_'+['x','y','z'][projAxis]) # code    
    cell_vellos_glob = sP.units.particleCodeVelocityToKms(cell_vellos_glob) # km/s

    densField = '%s numdens' % lines[line]['ion']
    if solar: densField += '_solar'

    cell_dens_glob = sP.snapshotSubsetP('gas', densField) # ions/cm^3

    # load rays (split into sub files, while spectra and absorber catalogs are not)
    ray_offset = 0

    for i in range(pSplitNum):
        rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = \
            generate_rays_voronoi_fullbox(sP, pSplit=[i,pSplitNum])
        
        # subset gas cells
        cell_vellos = cell_vellos_glob[cell_inds]
        cell_dens = cell_dens_glob[cell_inds]

        # convert length units, all other units already appropriate
        rays_dl = sP.units.codeLengthToMpc(rays_dl)

        # loop over requested line(s)
        for j, line in enumerate(lineNames):
            # load ion abundances per cell, unless we already have
            print(f' [{j+1:02d}] of [{len(lineNames):02d}] computing: [{line}] wave0 = {lines[line]["wave0"]:.4f}', flush=True)

            # determine cells which contribute to each absorber in each spectra
            spec_index_loc, counts_abs_loc, offset_abs_loc, coldens_abs_loc, colfrac_abs_loc = \
                _cell_absorber_map(lines[line]["wave0"], 
                abs_counts[line], abs_offset[line], abs_ind_spec[line], abs_ind_start[line], abs_ind_stop[line],
                rays_off, rays_len, rays_dl, rays_inds, ray_offset, 
                cell_vellos, cell_dens, z_vals, z_lengths, inst_waveedges)

            # save cell indices list (spec_index_loc, are local with respect to cell_inds)
            spec_index_loc = cell_inds[spec_index_loc]

            spec_index[line][spec_w_off[line]:spec_w_off[line]+spec_index_loc.size] = spec_index_loc

            # save per-absorber results
            counts_abs[line][abs_w_off[line]:abs_w_off[line]+counts_abs_loc.size] = counts_abs_loc
            offset_abs[line][abs_w_off[line]:abs_w_off[line]+offset_abs_loc.size] = offset_abs_loc
            coldens_abs[line][abs_w_off[line]:abs_w_off[line]+coldens_abs_loc.size] = coldens_abs_loc
            colfrac_abs[line][abs_w_off[line]:abs_w_off[line]+colfrac_abs_loc.size] = colfrac_abs_loc

            spec_w_off[line] += spec_index_loc.size
            abs_w_off[line] += counts_abs_loc.size

        ray_offset += rays_len.size

    # reduce spec_index to used size
    for line in lineNames:
        assert spec_index[line][spec_w_off[line]-1] != -1, 'Error: Last entry in spec_index is not set.'
        assert spec_index[line][spec_w_off[line]] == -1, 'Error: Past end of spec_index is not -1.'

        spec_index[line] = spec_index[line][0:spec_w_off[line]]

    # open file for writing
    with h5py.File(saveFilename,'w') as f:
        for line in lineNames:
            f['spec_index/%s' % line] = spec_index[line]
            f['counts_abs/%s' % line] = counts_abs[line]
            f['offset_abs/%s' % line] = offset_abs[line]
            f['coldens_abs/%s' % line] = coldens_abs[line]
            f['colfrac_abs/%s' % line] = colfrac_abs[line]

    print('Saved: [%s]' % saveFilename)

    return spec_index, counts_abs, offset_abs, coldens_abs, colfrac_abs

def calc_statistics_from_saved_rays(sP, ion):
    """ Calculate useful statistics based on already computed and saved rays.
    Results depend on ion, independent of actual transition.
    Results depend on the physical properties along each sightline, not on the absorption spectra.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ion (str): space separated species name and ionic number e.g. 'Mg II'.
    """
    # config
    dens_threshold = 1e-12 # ions/cm^3

    pSplitNum = 16

    # save file
    saveFilename = _spectra_filepath(sP, ion).replace('integral_','stats_').replace('_combined','')

    # (global) load required gas cell properties
    densField = '%s numdens' % ion
    cell_dens = sP.snapshotSubset('gas', densField) # ions/cm^3

    # loop over splits
    w_offset = 0

    for i in range(pSplitNum):
        pSplit = [i, pSplitNum]

        # load rays
        result = generate_rays_voronoi_fullbox(sP, pSplit=pSplit, search=True)
        if result is None:
            continue

        rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = result

        # convert indices local to this subset of the snapshot into global snapshot indices
        ray_cell_inds = cell_inds[rays_inds]

        # allocate (equal number of rays per split file)
        if i == 0:
            n_clouds = np.zeros(rays_len.size * pSplitNum, dtype='int32')

        # loop over each ray
        print_fac = int(rays_len.size/10)
        for j in range(rays_len.size):
            if j % print_fac == 0: print('[%2d of %2d] %.2f%%' % (i,pSplitNum,j/rays_len.size*100), flush=True)
            # get skewers of density, pathlength
            local_inds = ray_cell_inds[rays_off[j]:rays_off[j]+rays_len[j]]
            local_dens = cell_dens[local_inds]

            local_dl = rays_dl[rays_off[j]:rays_off[j]+rays_len[j]]

            # identify all intersected cells above ion density threshold
            w = np.where(local_dens > dens_threshold)[0]

            if len(w) == 0:
                # no cells above threshold == no clouds
                continue

            # find contiguous index ranges, identify breakpoints between contiguous ranges
            diff = np.diff(w)
            breaks = np.where(diff != 1)[0]

            # count number of discrete clouds
            n_clouds[w_offset+j] = len(breaks) + 1

        w_offset += rays_len.size

    # save output
    with h5py.File(saveFilename,'w') as f:
        f['n_clouds'] = n_clouds

    print(f'Saved: [{saveFilename}]')

def test_abs_coldens(sim, ion, instrument):
    """ Check if the column densities of all absorbers of a given spectrum sum to the 
    separately integrated total column density of that spectrum. """

    # load absorber catalog
    _, abs_counts, abs_offset, abs_ind_spec, abs_ind_start, abs_ind_stop = \
        absorber_catalog(sim, ion, instrument)
    
    # load column densities per absorber
    spec_index, counts_abs, offset_abs, coldens_abs, colfrac_abs = cell_to_absorber_map(sim, ion, instrument)

    # load integrals of column density
    field = ion + ' numdens'
    N = np.log10(integrate_along_saved_rays(sim, field)) # log cm^-2

    # pick a line
    line = list(counts_abs.keys())[0]

    diff = np.zeros(N.size, dtype='float32') # dex

    # loop over the spectra
    for i in range(N.size):
        # any absorbers present?
        if abs_counts[line][i] == 0:
            if N[i] > 7.5:
                print(i, 'zero abs but N = ', N[i])
            continue

        # sum up the column densities of all absorbers in this spectrum
        N_abs = 0
        for j in range(abs_counts[line][i]):
            N_abs += coldens_abs[line][abs_offset[line][i]+j]

        N_abs = np.log10(N_abs)

        diff[i] = N[i] - N_abs

        # compare to the integrated column density
        if np.abs(N[i]-N_abs) > 0.1 and N[i] > 8.0:
            assert N_abs <= N[i]
            print(i, abs_counts[line][i], N[i], N_abs, np.abs(N[i]-N_abs))
        assert np.isclose(N[i], N_abs, atol=np.inf), f'Error: Spectrum {i} has total column density mismatch.'

    print('Done.')

    # plot
    fig, (ax,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,8))
    ax.set_yscale('log')
    ax.set_xlabel('Ray Column Density [log cm$^{-2}$]')
    ax.set_ylabel('Difference [ dex ]')

    ax.set_rasterization_zorder(1) # elements below z=1 are rasterized
    ax.plot(N, diff, 'o', markersize=1, alpha=1.0, zorder=0)

    ww = np.where(diff == 0)
    ax.plot(N[ww], diff[ww] + ax.get_ylim()[0]*0.9, 'o', markersize=1, alpha=1.0, zorder=0)

    # plot relative diff
    ax2.set_xlabel('Ray Column Density [log cm$^{-2}$]')
    ax2.set_ylabel('Relative Difference [ dex ]')

    ax2.set_rasterization_zorder(1) # elements below z=1 are rasterized
    rel_diff = np.log10(diff / 10.0**N)
    ax2.plot(N, rel_diff, 'o', markersize=1, alpha=1.0, zorder=0)

    fig.savefig('test_abs_coldens_%s.pdf' % line)
    plt.close(fig)
