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
from scipy.ndimage import map_coordinates

from util.loadExtern import loadSDSSData
from cosmo.kCorr import kCorrections, coeff
from cosmo.load import groupCat, auxCat
from cosmo.util import correctPeriodicDistVecs
from util.helper import logZeroMin, trapsum
from util.sphMap import sphMap
from vis.common import rotationMatrixFromVec, rotateCoordinateArray

# currently same for all sims, otherwise move into sP:
gfmBands = {'U':0, 'B':1, 'V':2, 'K':3,
            'g':4, 'r':5, 'i':6, 'z':7}

def loadSimGalColors(sP, simColorsModel, colorData=None, bands=None):
    """ Load band-magnitudes either from snapshot photometrics or from auxCat SPS modeling, 
    and convert to a color if bands is passed in, otherwise return loaded data. If loaded 
    data is passed in with bands, do then magnitude computation without re-loading."""
    if colorData is None:
        # load
        if simColorsModel == 'snap':
            colorData = groupCat(sP, fieldsSubhalos=['SubhaloStellarPhotometrics'])
        else:
            acKey = 'Subhalo_StellarPhot_' + simColorsModel
            colorData = auxCat(sP, fields=[acKey])

    # early exit with full data?
    if bands is None:
        return colorData

    # compute colors
    if simColorsModel == 'snap':
        gc_colors = stellarPhotToSDSSColor( colorData['subhalos'], bands )
    else:
        # auxcatPhotToSDSSColor():
        acBands = list(colorData[acKey+'_attrs']['bands'])
        i0 = acBands.index('sdss_'+bands[0])
        i1 = acBands.index('sdss_'+bands[1])
        gc_colors = colorData[acKey][:,i0] - colorData[acKey][:,i1]

    return gc_colors

def stellarPhotToSDSSColor(photVector, bands):
    """ Convert the GFM_StellarPhotometrics[] or SubhaloStellarPhotometrics[] vector into a 
    specified color, by choosing the right elements and handling any necessary conversions. """
    colorName = ''.join(bands)

    # dictionary of band name -> SubhaloStellarPhotometrics[:,i] index i
    ii = gfmBands

    if colorName == 'ui':
        # UBVK are in Vega, i is in AB, and U_AB = U_Vega + 0.79, V_AB = V_Vega + 0.02
        # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # assume Buser V = Johnson V filter (close-ish), and use Lupton2005 transformation
        # http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html
        u_sdss_AB = photVector[:,ii['g']] + (-1.0/0.2906) * \
                    (photVector[:,ii['V']]+0.02 - photVector[:,ii['g']] - 0.0885)
        return u_sdss_AB - photVector[:,ii['z']]

    if colorName == 'gr':
        return photVector[:,ii['g']] - photVector[:,ii['r']] + 0.0 # g,r in sdss AB magnitudes

    if colorName == 'ri':
        return photVector[:,ii['r']] - photVector[:,ii['i']] + 0.0 # r,i in sdss AB magnitudes

    if colorName == 'iz':
        return photVector[:,ii['i']] - photVector[:,ii['z']] + 0.0 # i,z in sdss AB magnitudes

    if colorName == 'gz':
        return photVector[:,ii['g']] - photVector[:,ii['z']] + 0.0 # g,z in sdss AB magnitudes

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
    dustModels = ['none','bc00','cf00','bc00_res_eff','bc00_res_conv','cf00_res_eff','cf00_res_conv']

    def __init__(self, sP, iso='padova07', imf='chabrier', dustModel='bc00', order=3):
        """ Load the pre-computed stellar photometrics table, computing if it does not yet exist. """
        import fsps

        assert iso in self.isoTracks
        assert imf in self.imfTypes.keys()
        assert dustModel in self.dustModels

        self.sP    = sP
        self.data  = {} # band magnitudes
        self.spec  = {} # spectra
        self.order = order # bicubic interpolation by default (1 = bilinear)
        self.bands = fsps.find_filter('') # do them all (138)

        self.dust = dustModel.split("_")[0]
        self.dustModel = dustModel

        saveFilename = self.basePath + 'mags_%s_%s_%s_bands-%d.hdf5' % (iso,imf,self.dust,len(self.bands))

        # no saved table? compute now
        if not isfile(saveFilename):
            print(' Computing new stellarPhotTable: [iso=%s imf=%s dust=%s]...' % (iso,imf,dustModel))
            self.computePhotTable(iso, imf, saveFilename)

        # load
        with h5py.File(saveFilename,'r') as f:
            self.bands  = f['bands'][()]
            self.ages   = f['ages_logGyr'][()]
            self.metals = f['metals_log'][()]
            self.wave   = f['wave_nm'][()]
            self.spec   = f['spec_lsun_hz'][()]

            for key in f:
                if 'mags_' in key:
                    self.data[key] = f[key][()]

        # pre-compute for dust model
        self.prep_filters()
        self.prep_dust_models()

    def computePhotTable(self, iso, imf, saveFilename):
        """ Compute a new photometrics table for the given (iso,imf,self.dust) using fsps. """
        import fsps

        if self.dust == 'none':
            dust_type   = 0
            dust1       = 0.0
            dust2       = 0.0
            dust_index  = 0.0
            dust1_index = 0.0
            dust_tesc   = 7.0 # log(yr)

        if self.dust == 'bc00':
            # see Conroy+ (2009) or Charlot & Fall (2000) - note 'bc00' is just a typo for 'cf00' kept 
            #   here with dust1=1.0 which, because the young populations have both attenuation terms 
            #   applied in FSPS, in reality means dust1=1.0+dust2 in the below equation:
            # tau_dust(lambda) = tau_1 * (lambda/lambda_0)^alpha_1      t_ssp <= t_bc
            #                    tau_2 * (lambda/lambda_0)^alpha_2      t_ssp  > t_bc
            dust_type   = 0    # powerlaw taking the above functional form
            dust1       = 1.0  # tau_1
            dust2       = 0.3  # tau_2
            dust_index  = -0.7 # alpha_2
            dust1_index = -0.7 # alpha_1
            dust_tesc   = 7.0  # t_bc [log(yr)] = 0.01 Gyr, timescale to escape/disrupt molecular birth cloud

        if self.dust == 'cf00':
            # same as 'bc00', with the real citation name, and dust1 changed to 0.7 such that 
            # tau_1 = 1.0 in the given functional form
            dust_type   = 0    # powerlaw taking the above functional form
            dust1       = 0.7  # tau_1
            dust2       = 0.3  # tau_2
            dust_index  = -0.7 # alpha_2
            dust1_index = -0.7 # alpha_1
            dust_tesc   = 7.0  # t_bc

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

        # get sizes of full spectra
        wave0, spec0 = pop.get_spectrum() # Lsun/Hz, Angstroms

        # save struct and spectral array
        mags = {}

        for band in self.bands:
            mags[band] = np.zeros( (pop.zlegend.size, pop.log_age.size), dtype='float32' )
        spec = np.zeros( (pop.zlegend.size, pop.log_age.size, wave0.size), dtype='float32' )

        # loop over metallicites, compute band magnitudes over an age grid for each
        for i in range(pop.zlegend.size):
            print('  [%d of %d] Z = %g' % (i,pop.zlegend.size,pop.zlegend[i]))

            # update metallicity step, request magnitudes in all bands
            pop.params['zmet'] = i + 1 # 1-indexed

            x = pop.get_mags(bands=self.bands)

            w, s = pop.get_spectrum(peraa=False) # Lsun/Hz, Angstroms
            assert np.array_equal(w,wave0) # we assume same wavelengths for all metal indices
            assert s.shape[0] == pop.log_age.size # should be same age grid in isochrones

            # put magnitudes into (age,Z) grids split by band
            for bandNum, bandName in enumerate(self.bands):
                mags[bandName][i,:] = x[:,bandNum]

            # save spectral array
            for j in range(pop.log_age.size):
                spec[i,j,:] = s[j,:]

        with h5py.File(saveFilename, 'w') as f:
            f['bands'] = self.bands
            f['ages_logGyr'] = np.array(pop.log_age - 9.0, dtype='float32') # log(yr) -> log(Gyr)
            f['metals_log']  = np.array(np.log10(pop.zlegend), dtype='float32') # linear -> log
            f['wave_nm']     = np.array(wave0 / 10.0, dtype='float32') # Ang -> nm

            for key in mags:
                f['mags_' + key] = mags[key]
            f['spec_lsun_hz'] = np.array(spec, dtype='float32') # Lsun/Hz

        print('Saved: [%s]' % saveFilename)

    def filters(self, select=None):
        """ Return name of available filters. """
        if select is not None:
            return [band for band in self.bands if select in band]
        return self.bands

    def has_filter(self, filterName):
        """ Return True or False if the pre-computed table contains the specified filter/band. """
        return filterName.lower() in self.bands

    def prep_dust_models(self):
        """ Do possibly expensive pre-calculations for (resolved) dust model. """
        self.lambda_nm = {}
        self.A_lambda_sol = {}
        self.f_scattering = {}
        self.gamma = {}

        self.beta = -0.5
        self.N_H0 = 2.1e21 # neutral hydrogen column density [cm^2]

        # Lsun/Hz to cgs at d=10pc
        self.mag2cgs = np.log10( self.sP.units.L_sun / (4.0 * np.pi * (10*self.sP.units.pc_in_cm)**2))

        for band in self.bands:
            if 'suprimecam' in band:
                continue # missing transmission data

            # get wavelength array
            if '_res_eff' in self.dustModel:
                # get single (lambda_eff) luminosity attenuation factor for each star
                lambda_nm = np.array(self.lambda_eff[band])

            if '_res_conv' in self.dustModel:
                # do full convolution of original stellar spectrum with tau(lambda)
                #lambda_nm = self.trans_lambda[band] # at the resolution of the transmission function
                lambda_nm = self.wave # at the resolution of the stellar spectra

            # get tau^a factor from absorption (Cardelli 1989 equations 1-3b)
            x = 1/(lambda_nm/1000) # inverse microns
            y = x - 1.82

            # we are outside the O/nearIR regime into the UV/FUV or IR
            # but: the master wavelength grid always spans a full range, just be cautious later
            #if x.min() < 0.3 or x.max() > 3.3: 
            #    continue

            R_V = 3.1 # e.g. MW/LMC value (2.7 for SMC)
            a_x = np.zeros( lambda_nm.size, dtype='float32' )
            b_x = np.zeros( lambda_nm.size, dtype='float32' )
            a_x[np.where(x < 1.1)] = 0.574 * x**1.61
            b_x[np.where(x < 1.1)] = -0.527 * x**1.61
            a_x[np.where(x >= 1.1)] = 1 + 0.17699*y**1 - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
                              + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
            b_x[np.where(x >= 1.1)] = 0 + 1.41338*y**1 + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
                              - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

            self.A_lambda_sol[band] = a_x + b_x / R_V

            ww = np.where(self.A_lambda_sol[band] < 0.0)
            self.A_lambda_sol[band][ww] = 0.0 # clip negative values

            gamma = np.zeros( lambda_nm.size, dtype='float32' )
            gamma[np.where(lambda_nm >= 200)] = 1.6
            gamma[np.where(lambda_nm < 200)] = 1.35            

            # get full tau factor accounting for scattering (Calzetti 1994 internal dust model #5)
            h_lambda = np.zeros( lambda_nm.size, dtype='float32' )
            omega_lambda = np.zeros( lambda_nm.size, dtype='float32' )

            yy = np.log10(lambda_nm*10.0)
            h_lambda = 1.0 - 0.561 * np.exp( -np.abs(yy-3.3112)**2.2 / 0.17 )
            omega_lambda[np.where(lambda_nm <= 346.0)] = 0.43 + 0.366 * (1 - np.exp(-(yy-3)*(yy-3)/0.2))
            omega_lambda[np.where(lambda_nm > 346.0)] = -0.48*yy + 2.41

            h_lambda = np.clip( h_lambda, 0.0, 1.0 )
            omega_lambda = np.clip( omega_lambda, 0.0, 1.0 )

            self.f_scattering[band] = (h_lambda * np.sqrt(1-omega_lambda) + (1-h_lambda)*(1-omega_lambda))

            self.lambda_nm[band] = lambda_nm
            self.gamma[band] = gamma

    def prep_filters(self):
        """ Extract filter properties in case we want them later. """
        import fsps

        self.lambda_eff = {}
        self.msun_ab = {}
        self.msun_vega = {}
        self.trans_lambda = {}
        self.trans_val = {}
        self.trans_normed = {}
        self.wave_ang = self.wave * 10.0

        for band in self.bands:
            if 'suprimecam' in band:
                continue # missing transmission data

            f = fsps.get_filter(band)

            # get filter general properties
            self.lambda_eff[band] = f.lambda_eff / 10.0 # nm
            self.msun_ab[band] = f.msun_ab
            self.msun_vega[band] = f.msun_vega

            # get transmission of filter
            trans_lambda, trans_val = f.transmission
            self.trans_lambda[band] = np.array(trans_lambda) / 10.0 # nm, make sure to copy
            self.trans_val[band] = np.array(trans_val)

            # interpolate transmission function onto master wavelength grid
            trans_val = np.interp( self.wave, self.trans_lambda[band], self.trans_val[band] )

            # normalize
            trans_norm = trapsum( self.wave_ang, trans_val/self.wave_ang )
            if trans_norm <= 0.0:
                trans_norm = 1.0 # band entirely outside wavelength array
            trans_val /= trans_norm
            trans_val[np.where(trans_val < 0.0)] = 0.0 # no negative values

            # pre-divide out Angstroms
            self.trans_normed[band] = trans_val / self.wave_ang

    def mags(self, band, ages_logGyr, metals_log, masses_logMsun):
        """ Interpolate table to compute magnitudes in requested band for input stars. """
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

        # unit conversions: ages, masses, metallicities
        ages_logGyr = self.sP.units.scalefacToAgeLogGyr(gfm_sftime[wStars])

        masses_logMsun = sP.units.codeMassToLogMsun(masses_code[wStars])

        metals_log = logZeroMin(gfm_metallicity[wStars])
        metals_log[metals_log < -20.0] = -20.0 # truncate at GFM_MIN_METAL

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

    def dust_tau_model_mags(self, band, N_H, Z_g, ages_logGyr, metals_log, masses_msun):
        """ For a set of stars characterized by their (age,Z,M) values as well as (N_H,Z_g) 
        calculated from the resolved gas distribution, do the Model (C) attenuation on the 
        full spectra, sum together, and convolve the resulting total L(lambda) with the band 
        transmission function, returning a magnitude, otherwise identical to mags_code_units(). """
        assert N_H.size == Z_g.size == ages_logGyr.size == metals_log.size == masses_msun.size
        assert band in self.bands

        # accumulate per star attenuated luminosity (wavelength dependent):
        obs_lum = np.zeros( self.wave.size, dtype='float32' )

        for i in range(N_H.size):
            # finish absorption tau and total tau
            tau_a = self.A_lambda_sol[band] * (1+self.sP.redshift)**self.beta * \
                    (Z_g[i]/self.sP.units.Z_solar)**self.gamma[band] * (N_H[i]/self.N_H0)
            tau_lambda = tau_a * self.f_scattering[band]

            # attenuation as a function of wavelength, interpolate onto stellar spec wavelengths
            atten = (1 - np.exp(-tau_lambda))
            tau_lambda[np.where(tau_lambda == 0.0)] = 1.0
            atten *= (1/tau_lambda) # where tau_lambda==0 (outside lambda range), set atten=0
            
            if self.lambda_nm[band].size > 1: # _conv models
                atten = np.interp( self.wave, self.lambda_nm[band], atten )

            # bilinear interpolation: stellar population spectrum
            x_ind = np.interp( metals_log[i],  self.metals, np.arange(self.metals.size) )
            y_ind = np.interp( ages_logGyr[i], self.ages,   np.arange(self.ages.size) )
            x = metals_log[i]
            y = ages_logGyr[i]

            # clip indices at [0,size] which leads to constant extrap (nearest grid edge value in that dim)
            x1_ind = np.int32(np.floor(x_ind))
            x2_ind = x1_ind + 1
            y1_ind = np.int32(np.floor(y_ind))
            y2_ind = y1_ind + 1

            x1_ind = np.clip(x1_ind, 0, self.metals.size-1) # change to size-2 for linear extrap
            x2_ind = np.clip(x2_ind, 0, self.metals.size-1)
            y1_ind = np.clip(y1_ind, 0, self.ages.size-1) # change to size-2 for linear extrap
            y2_ind = np.clip(y2_ind, 0, self.ages.size-1)

            spec_12 = np.squeeze( self.spec[x1_ind,y2_ind,:] )
            spec_21 = np.squeeze( self.spec[x2_ind,y1_ind,:] )
            spec_11 = np.squeeze( self.spec[x1_ind,y1_ind,:] )
            spec_22 = np.squeeze( self.spec[x2_ind,y2_ind,:] )
 
            x1 = self.metals[x1_ind]
            x2 = self.metals[x2_ind]
            y1 = self.ages[y1_ind]
            y2 = self.ages[y2_ind]

            # constant beyond edges (delete for linear extrap)
            if x2_ind == x1_ind == self.metals.size-1:
                x2 += (self.metals[-1]-self.metals[-2])
            if x2_ind == x1_ind == 0:
                x1 -= (self.metals[1]-self.metals[0])
            if y2_ind == y1_ind == self.ages.size-1:
                y2 += (self.ages[-1]-self.ages[-2])
            if y2_ind == y1_ind == 0:
                y1 -= (self.ages[1]-self.ages[0])

            # interpolated 1D spectrum
            spectrum_local = spec_11*(x2-x)*(y2-y) + spec_21*(x-x1)*(y2-y) + \
                             spec_12*(x2-x)*(y-y1) + spec_22*(x-x1)*(y-y1)
            spectrum_local /= ((x2-x1)*(y2-y1))

            # accumulate attenuated contribution of this stellar population
            obs_lum += (spectrum_local * masses_msun[i]) * atten

            if 0: # DEBUG
                # get band magnitude as computed by FSPS
                mags = self.mags(band, np.array([ages_logGyr[i]]), 
                    np.array([metals_log[i]]), np.log10(np.array([masses_msun[i]])))

                # do our method of band convolution here and compare
                spec_band = spectrum_local * self.trans_normed[band] # Lsun/Hz/Angstrom
                spec_band *= atten

                result = trapsum(self.wave_ang, spec_band)
                nn = self.wave.size
                local_lum = np.sum( np.abs(self.wave_ang[1:nn-1]-self.wave_ang[0:nn-2]) * \
                                    (spec_band[1:nn-1] + spec_band[0:nn-2])*0.5 )

                result_mag = -2.5 * np.log10(result) - 48.60 - 2.5*self.mag2cgs
                result_mag2 = -2.5 * np.log10(local_lum) - 48.60 - 2.5*self.mag2cgs

                mags_wmass = mags - 2.5 * np.log10(masses_msun[i])
                print(mags,mags_wmass,result_mag,result_mag2)
                import pdb; pdb.set_trace()

        # convolve with band (trapezoidal rule)
        obs_lum *= self.trans_normed[band]

        nn = self.wave.size
        band_lum = np.sum( np.abs(self.wave_ang[1:nn-1]-self.wave_ang[0:nn-2]) * \
                            (obs_lum[1:nn-1] + obs_lum[0:nn-2])*0.5 )

        result_mag = -2.5 * np.log10(band_lum) - 48.60 - 2.5*self.mag2cgs

        return result_mag

    def resolved_dust_mapping(self, pos_in, hsml, mass_nh, quant_z, pos_stars_in, 
                              projCen, projVec, pxSize=1.0):
        """ Compute line of sight quantities per star for a resolved dust attenuation calculation.
        Gas (pos,hsml,mass_nh,quant_z) and stars (pos_stars) are used for the gridding of the gas 
        and the target (star) list. projVec is a [3]-vector, and the particles are rotated about 
        projCen [3] such that it aligns with the projection direction. pxSize is in physical kpc. """
        assert projCen.size == 3 and projVec.size == 3

        # rotation
        axes = [0,1]
        targetVec = np.array( [0,0,1], dtype='float32' )
        rotMatrix = rotationMatrixFromVec(projVec, targetVec)

        pos, _ = rotateCoordinateArray(self.sP, pos_in, rotMatrix, projCen, shiftBack=True)
        pos_stars, extentStars = rotateCoordinateArray(self.sP, pos_stars_in, rotMatrix, 
                                                       projCen, shiftBack=True)

        # configure projection grid, note we use the symmetric covering extentStars around the 
        # original projCen, although we could decrease the grid size by re-centering.
        #extentStars = np.array([ extentStars[1], extentStars[0] ]) # permute for T (no)
        extentStars += pxSize*2.0

        boxSizeImg = np.array([extentStars[0], extentStars[1], self.sP.boxSize])
        pxSizeCode = self.sP.units.physicalKpcToCodeLength(pxSize)
        nPixels    = np.int32( np.ceil(extentStars/pxSizeCode) )[0:2]

        nThreads = 8

        if pos_in.shape[0] < 1e3 and pos_stars_in.shape[0] < 1e3:
            nThreads = 4
        if pos_in.shape[0] < 1e2 and pos_stars_in.shape[0] < 1e2:
            nThreads = 1

        #print('nStars: ',pos_stars.shape[0])
        #print('nGas: ',pos.shape[0])
        #print('extentStars: ',extentStars)
        #print('boxSizeImg: ',boxSizeImg)
        #print('boxCen: ',projCen)
        #print('nPixels: ',nPixels)
        #print('nThreads: ',nThreads)

        # get (N_H,Z_g) along line of sight to each star
        N_H, Z_g = sphMap( pos=pos, hsml=hsml, mass=mass_nh, quant=quant_z, axes=axes, ndims=3, 
                           boxSizeSim=self.sP.boxSize, boxSizeImg=boxSizeImg, boxCen=projCen, 
                           nPixels=nPixels, colDens=False, multi=True, nThreads=nThreads, posTarget=pos_stars )

        # normalize out mass weights of metallicity
        w = np.where(N_H > 0.0)
        Z_g[w] /= N_H[w]

        # convert N_H units from mass to coldens_code to (neutral H atoms)/cm^2
        pixelArea = (boxSizeImg[0]/nPixels[0]) * (boxSizeImg[1]/nPixels[1])

        N_H /= pixelArea
        N_H = self.sP.units.codeColDensToPhys(N_H, cgs=True, numDens=True)

        return N_H, Z_g
