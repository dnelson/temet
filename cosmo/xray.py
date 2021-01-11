"""
cosmo/xray.py
  Generate x-ray emissivity tables using AtomDB/XSPEC and apply these to gas cells.
"""
import numpy as np
import h5py
import astropy.io.fits as pyfits
from os.path import expanduser
from scipy.integrate import cumtrapz

from util.helper import rootPath

basePath = rootPath + "tables/xray/"

apec_elem_names = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", 
                   "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr",
                   "Mn", "Fe", "Co", "Ni", "Cu", "Zn"] # COMMENT Atoms Included

abundance_tables = { # NOTE: AG89 is the assumed abundances inside APEC
    "AG89" : np.array([1.00E+00, 9.77E-02, 1.45E-11, 1.41E-11, 3.98E-10,
                       3.63E-04, 1.12E-04, 8.51E-04, 3.63E-08, 1.23E-04,
                       2.14E-06, 3.80E-05, 2.95E-06, 3.55E-05, 2.82E-07,
                       1.62E-05, 3.16E-07, 3.63E-06, 1.32E-07, 2.29E-06,
                       1.26E-09, 9.77E-08, 1.00E-08, 4.68E-07, 2.45E-07,
                       4.68E-05, 8.32E-08, 1.78E-06, 1.62E-08, 3.98E-08]),
    "aspl" : np.array([1.00E+00, 8.51E-02, 1.12E-11, 2.40E-11, 5.01E-10,
                       2.69E-04, 6.76E-05, 4.90E-04, 3.63E-08, 8.51E-05,
                       1.74E-06, 3.98E-05, 2.82E-06, 3.24E-05, 2.57E-07,
                       1.32E-05, 3.16E-07, 2.51E-06, 1.07E-07, 2.19E-06,
                       1.41E-09, 8.91E-08, 8.51E-09, 4.37E-07, 2.69E-07,
                       3.16E-05, 9.77E-08, 1.66E-06, 1.55E-08, 3.63E-08]),
    "wilm" : np.array([1.00E+00, 9.77E-02, 0.00, 0.00, 0.00, 2.40E-04,
                       7.59E-05, 4.90E-04, 0.00, 8.71E-05, 1.45E-06, 2.51E-05,
                       2.14E-06, 1.86E-05, 2.63E-07, 1.23E-05, 1.32E-07,
                       2.57E-06, 0.00, 1.58E-06, 0.00, 6.46E-08, 0.00,
                       3.24E-07, 2.19E-07, 2.69E-05, 8.32E-08, 1.12E-06,
                       0.00, 0.00]),
    "lodd" : np.array([1.00E+00, 7.92E-02, 1.90E-09, 2.57E-11, 6.03E-10,
                       2.45E-04, 6.76E-05, 4.90E-04, 2.88E-08, 7.41E-05,
                       1.99E-06, 3.55E-05, 2.88E-06, 3.47E-05, 2.88E-07,
                       1.55E-05, 1.82E-07, 3.55E-06, 1.29E-07, 2.19E-06,
                       1.17E-09, 8.32E-08, 1.00E-08, 4.47E-07, 3.16E-07,
                       2.95E-05, 8.13E-08, 1.66E-06, 1.82E-08, 4.27E-08])
}

def integrate_to_common_grid(bins_in, cont_in, bins_out):
    """ Convert a 'compressed' APEC spectrum into a normally binned one, by interpolating 
    from an input (bins,cont) pair to (bins_out). """
    
    # concatenate compressed spectrum bins (input) and requested bin edges (output)
    bins_all = np.append(bins_in, bins_out)
    
    # interpolate compressed spectrum emis (input) to requested bin edges (output)
    cont_tmp = np.interp(bins_out, bins_in, cont_in)

    cont_all = np.append(cont_in, cont_tmp)

    # generate mask and flag output entries
    mask = np.zeros( (bins_in.size + bins_out.size), dtype='bool' )
    mask[bins_in.size:] = True

    # sort
    sort_inds = np.argsort(bins_all)
    bins_all = bins_all[sort_inds]
    cont_all = cont_all[sort_inds]
    mask = mask[sort_inds]

    # cumulative integrate: composite trap rule
    cum_cont = cumtrapz(cont_all, bins_all, initial=0.0)

    # select our output points
    cont_out = cum_cont[mask]

    # convert to differential emissivity per bin
    cont = cont_out[1:] - cont_out[:-1]

    return cont

def apec_load():
    """ Testing. """
    from util.units import units

    base = expanduser("~") + '/code/atomdb/'
    path_line = base + 'apec_line.fits'
    path_cont = base + 'apec_coco.fits'

    # define our master wavelength grid
    dtype = 'float64'
    n_energy_bins = 1000

    master_grid = np.logspace(-3.5, 1.5, n_energy_bins+1) # 0.001 to 100 keV (EDGES)

    # define our abundance ratio choice
    abundanceSet = 'AG89'

    # define our metallicity binning
    n_metal_bins = 100
    metal_minmax = [-3.0, 1.0] # log solar

    # open continuum APEC file
    with pyfits.open(path_cont) as f:
        # get metadata for array sizes
        temp_kev = f[1].data.field('kT')
        n_temp_bins = temp_kev.size

        n_atom_bins = f[2].data.field('Z').size

        # allocate, both have units of [photon cm^3 s^-1 bin^-1]
        master_continuum = np.zeros( (n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype )
        master_pseudo    = np.zeros( (n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype )

        # there are n_temp_bins main blocks in the fits file
        for i in range(n_temp_bins):
            data = f[2+i].data

            # each block has 30 entries, one per element of interest
            if i > 0:
                # always the same
                assert np.array_equal(atomic_number_Z, data.field('Z'))

            atomic_number_Z = data.field('Z')
            #ion_number = data.field('rmJ') # e.g. 6 for VI, unused

            n_atoms = atomic_number_Z.size

            # all four datasets below have shape e.g. [n_atoms, n_energy_points]
            # continuum
            continuum_bins = data.field('E_Cont') # kev
            continuum = data.field('Continuum') # photon cm^3 s^-1 kev^-1

            # the pseudo-continuum consists of lines which are too weak to list individually
            # and so are accumulated here
            pseudo_bins = data.field('E_Pseudo') # kev
            pseudo = data.field('Pseudo') # photon cm^3 s^-1 kev^-1

            # how many non-zero entries?
            N_good_cont = data.field('N_cont')
            N_good_pseudo = data.field('N_pseudo')

            for j in range(n_atoms):
                # take first N (valid) points, and interpolate onto common grid
                N = N_good_cont[j]

                loc_result = integrate_to_common_grid(continuum_bins[j,:N], continuum[j,:N], master_grid)

                master_continuum[i,j,:] = loc_result

                # same for pseudo
                N = N_good_pseudo[j]

                loc_result = integrate_to_common_grid(pseudo_bins[j,:N], pseudo[j,:N], master_grid)

                master_pseudo[i,j,:] = loc_result

    # open line emission APEC file
    with pyfits.open(path_line) as f:

        assert np.array_equal(f[1].data.field('kT'), temp_kev) # same temp bins

        # allocate
        master_line = np.zeros( (n_temp_bins, n_atom_bins, n_energy_bins), dtype=dtype )

        # loop over temperature bins
        for i in range(n_temp_bins):
            atomic_number_Z = f[2+i].data.field('Element')
            #ion_number = f[2+i].data.field('Element') # unused

            assert atomic_number_Z.max() < n_atom_bins

            # in each temperature bin, we have an arbitrary number of entries here:
            waveang = f[2+i].data.field('Lambda') # wavelength of list [angtrom]
            emis = f[2+i].data.field('Epsilon') # line emissivity, at this temp and density [photon cm^3 s^-1]

            # convert wavelength to energy
            wave_kev = units.hc_kev_ang / waveang

            # deposit each line as a delta function (neglect thermal/velocity broadening...)
            # we combine all elements and ions together, could keep them separate if there was some reason
            ind = np.clip(np.searchsorted(master_grid, wave_kev, side='left') - 1, a_min=0, a_max=None)

            master_line[i,atomic_number_Z,ind] += emis

    for j in range(n_atom_bins):
        print('[%2d, %2s] total emis: cont = %e, pseudo = %e, line = %e' % \
            (j,apec_elem_names[j],master_continuum[:,j,:].sum(),master_pseudo[:,j,:].sum(),master_line[:,j,:].sum()))

    # convert these per-element emissivities to be instead as a function of metallicity, 
    # (must assume solar abundances, or otherwise input abundances per species of interest)
    # i.e. here we 'bake in' the abundances
    spec_prim  = np.zeros( (n_temp_bins, n_energy_bins), dtype=dtype )
    spec_metal = np.zeros( (n_temp_bins, n_energy_bins), dtype=dtype )

    apecAbundSet = 'AG89' # i.e. as assumed when generating APEC tables, never change

    # note: e.g. SOXS takes a different definition of 'metals', and includes several additional 
    # 'trace' elements beyond {H, He} into the non-metals category
    for i in range(2): # H, He
        abund_ratio = abundance_tables[abundanceSet][i] / abundance_tables[apecAbundSet][i]
        spec_prim += master_continuum[:,i,:] * abund_ratio
        spec_prim += master_pseudo[:,i,:] * abund_ratio
        spec_prim += master_line[:,i,:] * abund_ratio

    for i in range(2,n_atom_bins):
        abund_ratio = abundance_tables[abundanceSet][i] / abundance_tables[apecAbundSet][i]
        spec_metal += master_continuum[:,i,:] * abund_ratio
        spec_metal += master_pseudo[:,i,:] * abund_ratio
        spec_metal += master_line[:,i,:] * abund_ratio

    # save 3D table with grid config [temp_kev, metal_solar, master_grid] 
    # in units of [photon cm^3 s^-1 bin^-1]
    # such that multiplication by n*n*V gives [photon s^-1 bin^-1]
    spec_grid_3d = np.zeros( (n_temp_bins, n_metal_bins, n_energy_bins), dtype=dtype )

    metallicities = np.linspace(metal_minmax[0], metal_minmax[1], n_metal_bins)

    for i in range(n_metal_bins):
        linear_metallicity_in_solar = 10.0**metallicities[i]
        spec_grid_3d[:,i,:] = spec_prim + linear_metallicity_in_solar *  spec_metal

    # and save 2D table integrating e.g. 0.5-2.0 kev, and converting each photon to its energy, 
    # with grid config [temp_kev, metal_solar]
    # in units of [erg cm^3 s^-1]
    # such that multiplication by n*n*V gives [erg s^-1] in this band-pass
    # depends on redshift since photon energies change, and [emin,emax] are observed frame
    if 1:
        # config, variable
        redshift = 0.0
        emin = 0.5 # kev, observed frame
        emax = 2.0 # kev, observed frame

    # erg/photon for each bin of master_grid
    master_grid_mid = (master_grid[1:] + master_grid[:-1]) / 2 # midpoints
    energy_erg_emit = master_grid_mid / units.erg_in_kev # erg/photon
    energy_erg_obs = energy_erg_emit / (1.0 + redshift) # erg/photon

    # convert spec_grid_3d from photon/s to erg/s
    spec_grid_3d *= energy_erg_obs

    # collect bins inside bandpass and integrate
    emin_emit = (1.0 + redshift) * emin
    emax_emit = (1.0 + redshift) * emax

    w = np.where( (master_grid >= emin) & (master_grid <= emax) )[0]

    emis = np.sum( spec_grid_3d[:,:,w], axis=2 ) # [erg cm^3 s^-1]

    # save test output
    fileName = basePath + 'apec_z%02d.hdf5' % np.round(redshift*10)
    with h5py.File(fileName,'w') as f:
        f['emis_05-2kev'] = emis # erg cm^3 s^-1
        f['temp'] = temp_kev # linear kev
        f['metal'] = metallicities # log solar

    print('Done, saved: [%s].' % fileName)

class xrayEmission():
    """ Use pre-computed XSPEC table to derive x-ray emissivities for simulation gas cells. """
    def __init__(self, sP, instrument=None, order=3, use_apec=False):
        """ Load the table, optionally only for a given instrument(s).
        If instruments ends in '_NoMet', use this table instead. """
        self.sP = sP

        self.data = {}
        self.grid = {}

        self.order = order # quadcubic interpolation by default (1 = quadlinear)
        self.gridNames = ['Metallicity','Normalisation','Temperature']

        # which table file? note: 
        zStr = '%02d' % np.round(sP.redshift * 10) # '00' for z=0, '04' for z=0.4
        metalStr = ''
        if '_NoMet' in instrument:
            metalStr = '_NoMet'
            instrument = instrument.replace('_NoMet','')

        fileName = 'XSPEC_z_%s%s.hdf5' % (zStr,metalStr)

        # different set of tables?
        self.use_apec = use_apec
        if self.use_apec:
            # todo: add redshift
            self.gridNames = ['temp','metal']

            fileName = 'apec_z%02d.hdf5' % np.round(sP.redshift * 10)

        with h5py.File(basePath + fileName,'r') as f:
            # load grid specification
            for name in self.gridNames:
                self.grid[name] = f[name][()]

            # load 3D emissivity tables
            if instrument is None:
                for key in [e for e in list(f.keys()) if e not in self.gridNames]:
                    self.data[key] = f[key][()]
            else:
                # load just one 'instrument' (e.g. dataset name for now)
                # Flux_05_2, Luminosity_05_2, Count_Erosita_05_2_2ks, Count_Chandra_03_5_100ks
                assert instrument in f
                self.data[instrument] = f[instrument][()]

    def slice(self, instrument, metal=None, norm=None, temp=None):
        """ Return a 1D slice of the table specified by a value in all other dimensions (only one 
          input can remain None). """
        if sum(pt is not None for pt in (metal,norm,temp)) != 2:
            raise Exception('Must specify 2 of 3 grid positions.')

        # closest array indices
        _, i0 = closest( self.grid['Normalisation'], dens if dens else 0 )
        _, i1 = closest( self.grid['Temperature'], temp if temp else 0 )
        _, i2 = closest( self.grid['Metallicity'], metal if metal else 0 )

        if norm is None:
            return self.grid['Normalisation'], self.data[instrument][:,i1,i2]
        if temp is None:
            return self.grid['Temperature'], self.data[instrument][i0,:,i2]
        if metal is None:
            return self.grid['Metallicity'], self.data[instrument][i0,i1,:]

    def emis(self, instrument, metal, norm, temp):
        """ Interpolate the x-ray table, return fluxes [erg/s/cm^2], luminosities [10^44 erg/s], or counts [1/s].
            Input gas properties can be scalar or np.array(), in which case they must have the same size.
              instrument : name of the requested dataset
              metal : metallicity [log solar]
              norm  : usual XSPEC normalization [log cm^-5]
              temp  : boltzmann constant * temperature [log keV]
        """
        from scipy.ndimage import map_coordinates
        import time

        if instrument not in self.data:
            raise Exception('Requested instrument [' + instrument + '] not in grid.')

        start_time = time.time()
        # convert input interpolant point into fractional 3D array indices
        # Note: we are clamping here at [0,size-1], which means that although we never 
        # extrapolate below (nearest grid edge value is returned), there is no warning given
        if self.use_apec:
            i1 = np.interp( temp,  self.grid['temp'],  np.arange(self.grid['temp'].size) )
            i2 = np.interp( metal, self.grid['metal'], np.arange(self.grid['metal'].size) )

            iND = np.vstack( (i1,i2) )
        else:
            i1 = np.interp( norm,  self.grid['Normalisation'], np.arange(self.grid['Normalisation'].size) )
            i2 = np.interp( temp,  self.grid['Temperature'],   np.arange(self.grid['Temperature'].size) )
            i3 = np.interp( metal, self.grid['Metallicity'],   np.arange(self.grid['Metallicity'].size) )

            iND = np.vstack( (i1,i2,i3) )

        # do 3D interpolation on this data product sub-table at the requested order
        locData = self.data[instrument]

        emis = map_coordinates( locData, iND, order=self.order, mode='nearest')

        # clip negatives to zero
        w = np.where(emis < 0.0)
        emis[w] = 0.0

        #print('Emissivity interp took [%.2f] sec (%s points)' % ((time.time()-start_time),norm.size) )

        return emis

    def _prefac(self, redshift=0.01):
        """ Return pre-factor for normalisation used in tables. 
        Note that z=0.01 is hard-coded in the "z=0" table, i.e. we should set z=0.01 to 
        use the table at z=0. To use the z=0.4 table at z=0.4, we should use redshift=0.4.
        However, we can approximate any redshift by using the "z=0" table and multiplying 
        the normalisation by the ratio of two prefactors, (z0/zother). TBD! """
        assert redshift == 0.01 # otherwise not yet understood

        ang_diam = self.sP.units.redshiftToAngDiamDist(redshift) * self.sP.units.Mpc_in_cm

        prefac = 1e-14 / (4 * np.pi * ang_diam**2 * (1+redshift)**2) # 1/cm^2

        return prefac

    def calcGasEmission(self, sP, instrument, indRange=None, tempSfCold=True):
        """ Compute x-ray emission, either (i) flux [erg/s/cm^2], (ii) luminosity [erg/s], or 
         (iii) counts for a particular observational setup [count/s], always linear,
         for gas particles in the whole snapshot, optionally restricted to an indRange.
         tempSfCold : set temperature of SFR>0 gas to cold phase temperature (1000 K, i.e. no x-ray
         emission) instead of the effective EOS temp. """
        instrument = instrument.replace("_NoMet","") # this is used to change the table file itself in init()

        # load gas densities, volume, derive normalization
        nh = sP.snapshotSubset('gas', 'nh', indRange=indRange) # linear 1/cm^3
        xe = sP.snapshotSubset('gas', 'ElectronAbundance', indRange=indRange) # = n_e/n_H
        volume = sP.snapshotSubset('gas', 'volume_cm3', indRange=indRange) # linear 1/cm^3

        if self.use_apec:
            norm = xe * nh**2 * volume # = n_e*n_H*V [1/cm^3]
        else:
            prefac = self._prefac(redshift=0.01)
            norm = np.log10((prefac * volume) * xe * nh**2) # log [1/cm^2 1/cm^6 cm^3] = [log 1/cm^5]

        # load gas metallicities
        assert sP.snapHasField('gas', 'GFM_Metallicity')
        metal = sP.snapshotSubset('gas', 'metal', indRange=indRange)
        metal_logSolar = sP.units.metallicityInSolar(metal, log=True)

        # load gas temperatures
        tempField = 'temp_sfcold' if tempSfCold else 'temp' # use cold phase temperature for eEOS gas (by default)
        temp = sP.snapshotSubset('gas', tempField, indRange=indRange) # log K
        temp_keV = np.log10(10.0**temp / sP.units.boltzmann_keV) # log keV
        
        # interpolate for flux, luminosity, or counts
        vals = self.emis(instrument, metal_logSolar, norm, temp_keV)

        if self.use_apec:
            # vals now needs to be multiplied by norm to obtain a luminosity in [erg/s]
            vals *= norm
        else:
            # XSPEC: for luminosities, remove 1e44 factor in table
            if 'Luminosity' in instrument:
                vals *= 1e44

        # check for strange values, and avoid absolute zeros
        assert np.count_nonzero(np.isnan(vals)) == 0

        w = np.where(vals == 0)
        vals[w] = 1e10 # extremely small

        # todo: convert to float32?
        return vals

def compare_tables():
    """ Test. Compare XSPEC and APEC tables. """
    from util.helper import plothist
    from util.simParams import simParams

    # config
    sP = simParams(run='tng100-1', redshift=0.0)

    inst1 = 'Luminosity_05_2'
    inst2 = 'emis_05-2kev'

    # create interpolators
    xray1 = xrayEmission(sP, instrument=inst1, use_apec=False)
    xray2 = xrayEmission(sP, instrument=inst2, use_apec=True)

    # load a subset
    indRange = [0,1000000]

    vals1 = xray1.calcGasEmission(sP, inst1, indRange=indRange)
    vals2 = xray2.calcGasEmission(sP, inst2, indRange=indRange)

    ratio = vals1 / vals2 # old / new
    print('mean median: ',ratio.mean(), np.median(ratio))
    print('min max: ', ratio.min(), ratio.max())
    print('percs 5,16,84,95: ', np.percentile(ratio, [5,16,85,95]))
    plothist(ratio[vals2 > 1e10])

    # what are outliers?
    w_good = np.where( (ratio>0.5) & (ratio<2.0) )
    w_bad = np.where( (ratio>2) )

    print('bad outliers: [%d of %d]' % (len(w_bad[0]),vals1.size))

    # relation with temp/metallicity/dens
    temp = sP.snapshotSubset('gas', 'temp', indRange=indRange)
    metal = sP.snapshotSubset('gas', 'z_solar', indRange=indRange)
    dens_nh = sP.snapshotSubset('gas', 'nh', indRange=indRange)

    print('temp: ', temp[w_good].mean(), temp[w_bad].mean())
    print('metal: ', metal[w_good].mean(), metal[w_bad].mean())
    print('dens_nh: ', dens_nh[w_good].mean(), dens_nh[w_bad].mean())

    import pdb; pdb.set_trace()
