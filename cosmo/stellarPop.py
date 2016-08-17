"""
cosmo/stellarPop.py
  Stellar population synthesis, evolution, photometrics.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
import h5py

from os.path import isfile, expanduser
from util.loadExtern import loadSDSSData
from cosmo.kCorr import kCorrections, coeff
from util.helper import logZeroSafe, logZeroMin

gfmBands = ['U','B','V','K','g','r','i','z'] # unused

def stellarPhotToSDSSColor(photVector, bands):
    """ Convert the GFM_StellarPhotometrics[] or SubhaloStellarPhotometrics[] vector into a 
    specified color, by choosing the right elements and handling any necessary conversions. """
    colorName = ''.join(bands)

    if colorName == 'ui':
        # U is in Vega, i is in AB, and U_AB = U_Vega + 0.79 
        # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # assume Buser U = Johnson U filter (close-ish), and use Lupton2005 transformation
        # http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html
        u_sdss = photVector[:,4] + (-1.0/0.2906) * (photVector[:,2] - photVector[:,4] - 0.0885)
        return u_sdss - photVector[:,6] + 0.79

    if colorName == 'gr':
        return photVector[:,4] - photVector[:,5] + 0.0 # g,r in sdss AB magnitudes

    if colorName == 'ri':
        return photVector[:,5] - photVector[:,6] + 0.0 # r,i in sdss AB magnitudes

    if colorName == 'iz':
        return photVector[:,6] - photVector[:,7] + 0.0 # i,z in sdss AB magnitudes

    if colorName == 'gz':
        return photVector[:,4] - photVector[:,7] + 0.0 # g,z in sdss AB magnitudes

    raise Exception('Band combination not implemented.')

def calcSDSSColors(bands, redshiftRange=None, eCorrect=False, kCorrect=False):
    """ Load the SDSS data files and compute a requested color, optionally restricting to a given 
    galaxy redshift range, correcting for extinction, and/or doing a K-correction. """
    assert redshiftRange is None, 'Not implemented.'

    sdss = loadSDSSData()

    # extinction correction
    if not eCorrect:
        for key in sdss.keys():
            if 'extinction_' in key:
                sdss[key] *= 0.0

    sdss_color = (sdss[bands[0]]-sdss['extinction_'+bands[0]]) - \
                 (sdss[bands[1]]-sdss['extinction_'+bands[1]])
    sdss_Mstar = sdss['logMass_gran1']

    # K-correction (absolute_M = apparent_m - C - K) (color A-B = m_A-C-K_A-m_B+C+K_B=m_A-m_B+K_B-K_A)
    if kCorrect:
        kCorrs = {}

        for band in bands:
            availCorrections = [key.split('_')[1] for key in coeff.keys() if band+'_' in key]
            useCor = availCorrections[0]
            #print('Calculating K-corr for [%s] band using [%s-%s] color.' % (band,useCor[0],useCor[1]))

            cor_color = (sdss[useCor[0]]-sdss['extinction_'+useCor[0]]) - \
                        (sdss[useCor[1]]-sdss['extinction_'+useCor[1]])

            kCorrs[band] = kCorrections(band, sdss['redshift'], useCor, cor_color)

        sdss_color += (kCorrs[bands[1]] - kCorrs[bands[0]])

    return sdss_color, sdss_Mstar

def calcMstarColor2dKDE(bands, gal_Mstar, gal_color, Mstar_range, mag_range, sP=None):
    """ Quick caching of (slow) 2D KDE calculation of (Mstar,color) plane for SDSS z<0.1 points 
    if sP is None, otherwise for simulation (Mstar,color) points if sP is specified. """
    from scipy.stats import gaussian_kde

    if sP is None:
        saveFilename = expanduser("~") + "/obs/sdss_2dkde_%s.hdf5" % ''.join(bands)
        dName = 'kde_obs'
    else:
        saveFilename = sP.derivPath + "galMstarColor_2dkde_%s_%d.hdf5" % (''.join(bands),sP.snap)
        dName = 'kde_sim'

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            xx = f['xx'][()]
            yy = f['yy'][()]
            kde_obs = f[dName][()]

        return xx, yy, kde_obs

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    vv = np.vstack( [gal_Mstar, gal_color] )
    kde = gaussian_kde(vv)

    xx, yy = np.mgrid[Mstar_range[0]:Mstar_range[1]:200j, mag_range[0]:mag_range[1]:400j]
    xy = np.vstack( [xx.ravel(), yy.ravel()] )
    kde2d = np.reshape( np.transpose(kde(xy)), xx.shape)

    # save
    with h5py.File(saveFilename,'w') as f:
        f['xx'] = xx
        f['yy'] = yy
        f[dName] = kde2d
    print('Saved: [%s]' % saveFilename)

    return xx, yy, kde2d

class sps():
    """ Use pre-computed FSPS stellar photometrics tables to derive magnitudes for simulation stars. """
    basePath = expanduser("~") + '/code/fsps.run/'

    imfTypes   = {'salpeter':0, 'chabrier':1, 'kroupa':2}
    isoTracks  = ['padova07'] # basti,geneva,mist,parsec (cannot easily dynamically change at present)
    dustModels = ['none','bc00']

    def __init__(self, sP, iso='padova07', imf='chabrier', dustModel='bc00', order=3):
        """ Load the pre-computed stellar photometrics table, computing if it does not yet exist. """
        import fsps

        assert iso in self.isoTracks
        assert imf in self.imfTypes.keys()
        assert dustModel in self.dustModels

        self.sP    = sP
        self.data  = {}
        self.order = order # bicubic interpolation by default (1 = bilinear)
        self.bands = fsps.find_filter('') # do them all (138)

        saveFilename = self.basePath + 'mags_%s_%s_%s_bands-%d.hdf5' % (iso,imf,dustModel,len(self.bands))

        # no saved table? compute now
        if not isfile(saveFilename):
            print(' Computing new stellarPhotTable: [iso=%s imf=%s dust=%s]...' % (iso,imf,dustModel))
            self.computePhotTable(iso, imf, dustModel, saveFilename)

        # load
        with h5py.File(saveFilename,'r') as f:
            self.bands  = f['bands'][()]
            self.ages   = f['ages_logGyr'][()]
            self.metals = f['metals_log'][()]
            for key in f:
                if 'mags_' not in key:
                    continue
                self.data[key] = f[key][()]

    def computePhotTable(self, iso, imf, dustModel, saveFilename):
        """ Compute a new photometrics table for the given (iso,imf,dustModel) using fsps. """
        import fsps

        if dustModel == 'none':
            dust_type   = 0
            dust1       = 0.0
            dust2       = 0.0
            dust_index  = 0.0
            dust1_index = 0.0
            dust_tesc   = 7.0 # log(yr)

        if dustModel == 'bc00':
            # see Conroy+ (2009) or Bruzual & Charlot (2000)
            # tau_dust(lambda) = tau_1 * (lambda/lambda_0)^alpha_1      t_ssp <= t_bc
            #                    tau_2 * (lambda/lambda_0)^alpha_2      t_ssp  > t_bc
            dust_type   = 0    # powerlaw taking the above functional form
            dust1       = 1.0  # tau_1
            dust2       = 0.3  # tau_2
            dust_index  = -0.7 # alpha_2
            dust1_index = -0.7 # alpha_1
            dust_tesc   = 7.0  # t_bc [log(yr)] = 0.01 Gyr, timescale to escape/disrupt molecular birth cloud

        # init
        pop = fsps.StellarPopulation(sfh = 0, # SSP
                                     zmet = 1, # integer index of metallicity value (modified later)
                                     add_neb_continuum = True,
                                     add_dust_emission = True,
                                     imf_type = self.imfTypes[imf],
                                     dust_type = dust_type,
                                     dust1 = dust1,
                                     dust2 = dust2,
                                     dust_index = dust_index,
                                     dust_tesc = dust_tesc,
                                     dust1_index = dust1_index)

        # different tracks are available at discrete metallicities (linear mass_Z/mass_tot, not in solar!)
        # (unused)
        if iso == 'padova07':
            Zsolar = 0.019
            metals = [0.0002,0.0003,0.0004,0.0005,0.0006,0.0008,0.001,0.0012,0.0016,0.002,0.0025,
                      0.0031,0.0039,0.0049,0.0061,0.0077,0.0096,0.012,0.015,0.019,0.024,0.03]
        if iso == 'basti':
            Zsolar = 0.02
            metals = [0.0003,0.0006,0.001,0.002,0.004,0.008,0.01,0.02,0.03,0.04]
        if iso == 'geneva':
            Zsolar = 0.02
            metals = [0.001,0.004,0.008,0.02,0.04]
        if iso == 'mist':
            Zsolar = 0.0142
            metals = Zsolar * 10.0**np.array([-2.5,-2.0,-1.75,-1.5,-1.25,-1.0,-0.75,-0.5,-0.25,0.0,0.25,0.5])
        if iso == 'parsec':
            Zsolar = 0.0152
            metals = [0.0001,0.0002,0.0005,0.001,0.002,0.004,0.006,0.008,0.01,0.014,0.017,0.02,0.03,0.04,0.06]

        # save struct
        mags = {}

        for band in self.bands:
            mags[band] = np.zeros( (pop.zlegend.size, pop.log_age.size), dtype='float32' )

        # loop over metallicites, compute band magnitudes over an age grid for each
        for i in range(pop.zlegend.size):
            print('  [%d of %d] Z = %g' % (i,pop.zlegend.size,pop.zlegend[i]))

            # update metallicity step, request magnitudes in all bands
            pop.params['zmet'] = i + 1 # 1-indexed

            x = pop.get_mags(bands=self.bands)

            # put magnitudes into (age,Z) grids split by band
            for bandNum, bandName in enumerate(self.bands):
                mags[bandName][i,:] = x[:,bandNum]            

        with h5py.File(saveFilename, 'w') as f:
            f['bands']  = self.bands
            f['ages_logGyr'] = np.array(pop.log_age - 9.0, dtype='float32') # log(yr) -> log(Gyr)
            f['metals_log']  = np.array(np.log10(pop.zlegend), dtype='float32') # linear -> log

            for key in mags:
                f['mags_' + key] = mags[key]

        print('Saved: [%s]' % saveFilename)

    def filters(self, select=None):
        """ Return name of available filters. """
        if select is not None:
            return [band for band in self.bands if select in band]
        return self.bands

    def has_filter(self, filterName):
        """ Return True or False if the pre-computed table contains the specified filter/band. """
        return filterName.lower() in self.bands

    def mags(self, band, ages_logGyr, metals_log, masses_logMsun):
        """ Interpolate table to compute magnitudes in requested band for input stars. """
        from scipy.ndimage import map_coordinates

        assert band.lower() in self.bands
        assert ages_logGyr.size == metals_log.size == masses_logMsun.size
        assert ages_logGyr.ndim == metals_log.ndim == masses_logMsun.ndim == 1

        # verify units
        assert np.max(metals_log) <= 0.0
        assert np.min(ages_logGyr) >= -10.0
        assert np.max(ages_logGyr) < 2.0
        assert np.max(masses_logMsun) < 10.0
        assert np.min(masses_logMsun) > 1.5 # we have such small stars it seems

        # convert input interpolant point into fractional 2D (+bandNum) array indices
        # Note: we are clamping at [0,size-1], so no extrapolation (nearest grid edge value is returned)
        i1 = np.interp( metals_log,  self.metals, np.arange(self.metals.size) )
        i2 = np.interp( ages_logGyr, self.ages,   np.arange(self.ages.size) )

        iND = np.vstack( (i1,i2) )
        locData = self.data["mags_" + band.lower()]

        # do 2D interpolation on this band sub-table at the requested order
        mags = map_coordinates( locData, iND, order=self.order, mode='nearest')

        return mags

    def mags_code_units(self, sP, band, gfm_sftime, gfm_metallicity, masses_code, retFullPt4Size=False):
        """ Do unit conversions (and wind particle filtering) on inputs, and return mags() results. """
        wStars = np.where( gfm_sftime >= 0.0 )

        if len(wStars[0]) == 0:
            return None

        # ages
        curUniverseAgeGyr = sP.units.redshiftToAgeFlat(sP.redshift)
        birthRedshift = 1.0/gfm_sftime[wStars] - 1.0
        ages_logGyr = logZeroMin( curUniverseAgeGyr - sP.units.redshiftToAgeFlat(birthRedshift) )

        # masses
        masses_logMsun = sP.units.codeMassToLogMsun(masses_code[wStars])

        # truncate metallicities at GFM_MIN_METAL and take log
        metals_log = logZeroMin(gfm_metallicity[wStars])
        metals_log[metals_log < -20.0] = -20.0

        # magnitudes for 1 solar mass SSPs
        stellarMags = self.mags(band, ages_logGyr, metals_log, masses_logMsun)

        # account for population mass
        stellarMags -= 2.5 * masses_logMsun

        # return an array of the same size of the input, with nan for wind entries
        if retFullPt4Size:
            r = np.zeros( gfm_sftime.size, dtype='float32' )
            r.fill(np.nan)
            r[wStars] = stellarMags
            return r

        return stellarMags

    def calcStellarLuminosities(self, sP, band, indRange=None):
        """ Compute (linear) luminosities in the given band, using either snapshot-stored values or 
        on the fly sps calculation, optionally restricted to indRange. Note that wind is here 
        returned as NaN luminosity, assuming it is filtered out elsewhere, e.g. in gridBox(). """
        assert isinstance(band, basestring)

        if 'snap_' in band:
            # direct load snapshot saved stellar photometrics
            fields = ['sftime','phot_'+band.split("snap_")[1]]

            stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=fields, indRange=indRange)

            wWind = np.where( stars['GFM_StellarFormationTime'] < 0.0 )
            stars['GFM_StellarPhotometrics'][wWind] = np.nan

            mags = stars['GFM_StellarPhotometrics']
        else:
            # load age,Z,mass_ini, use FSPS on the fly
            assert band in self.bands
            fields = ['initialmass','sftime','metallicity']

            stars = cosmo.load.snapshotSubset(sP, partType='stars', fields=fields, indRange=indRange)

            mags = self.mags_code_units(sP, band, stars['GFM_StellarFormationTime'], 
                                        stars['GFM_Metallicity'], stars['GFM_InitialMass'], 
                                        retFullPt4Size=True)

        # convert to luminosities
        lums = np.zeros( mags.size, dtype='float32' )
        lums.fill(np.nan)

        ww = np.isfinite(mags)
        lums[ww] = np.power(10.0, -0.4 * mags[ww])
        
        return lums