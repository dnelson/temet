"""
cosmo/xray.py
  Generate x-ray emissivity tables using AtomDB/XSPEC and apply these to gas cells.
"""
import numpy as np
import h5py

from util.helper import rootPath

basePath = rootPath + "tables/xray/"

class xrayEmission():
    """ Use pre-computed XSPEC table to derive x-ray emissivities for simulation gas cells. """
    def __init__(self, sP, instrument=None, order=3):
        """ Load the table, optionally only for a given instrument(s). """
        self.sP = sP

        self.data = {}
        self.grid = {}

        self.order = order # quadcubic interpolation by default (1 = quadlinear)
        self.gridNames = ['Metallicity','Normalisation','Temperature']

        with h5py.File(basePath + 'XSPEC_table.hdf5','r') as f:
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
              metal : metallicity [log solar] ???
              norm  : hydrogen number density * electron number density [log cm^-6] ???
              temp  : boltzmann constant * temperature [log keV] ???
        """
        from scipy.ndimage import map_coordinates
        import time

        if instrument not in self.data:
            raise Exception('Requested instrument [' + instrument + '] not in grid.')

        start_time = time.time()
        # convert input interpolant point into fractional 3D array indices
        # Note: we are clamping here at [0,size-1], which means that although we never 
        # extrapolate below (nearest grid edge value is returned), there is no warning given
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

    def calcGasEmission(self, sP, instrument, indRange=None, tempSfCold=True):
        """ Compute x-ray emission, either (i) flux [erg/s/cm^2], (ii) luminosity [erg/s], or 
         (iii) counts for a particular observational setup [count/s], always linear,
         for gas particles in the whole snapshot, optionally restricted to an indRange.
         tempSfCold : set temperature of SFR>0 gas to cold phase temperature (1000 K, i.e. no x-ray
         emission) instead of the effective EOS temp. """

        # load gas densities, volume, derive normalization
        nh = sP.snapshotSubset('gas', 'nh', indRange=indRange) # linear 1/cm^3
        xe = sP.snapshotSubset('gas', 'ElectronAbundance', indRange=indRange) # = n_e/n_H
        volume = sP.snapshotSubset('gas', 'volume_cm3', indRange=indRange) # linear 1/cm^3
        norm = np.log10(xe * nh**2) # = n_e*n_H [log 1/cm^6]

        redshift = 0.01 # hard-coded in table?
        ang_diam = sP.units.redshiftToAngDiamDist(redshift) * sP.units.Mpc_in_cm
        prefac = 1e-14 / (4 * np.pi * ang_diam**2 * (1+redshift)**2) # 1/cm^2
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

        # for luminosities, remove 1e44 factor in table
        if 'Luminosity' in instrument:
            vals *= 1e44

        return vals
