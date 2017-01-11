"""
cosmo/stellarPop.py
  Stellar population synthesis, evolution, photometrics.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
import h5py
import time
from numba import jit

from os.path import isfile, expanduser
from scipy.ndimage import map_coordinates

from util.loadExtern import loadSDSSData
from cosmo.kCorr import kCorrections, coeff
from cosmo.load import groupCat, auxCat
from cosmo.util import correctPeriodicDistVecs
from util.helper import logZeroMin, trapsum, iterable
from util.sphMap import sphMap
from vis.common import rotationMatrixFromVec, rotateCoordinateArray

# currently same for all sims, otherwise move into sP:
gfmBands = {'U':0, 'B':1, 'V':2, 'K':3,
            'g':4, 'r':5, 'i':6, 'z':7}

def loadSimGalColors(sP, simColorsModel, colorData=None, bands=None, projs=None):
    """ Load band-magnitudes either from snapshot photometrics or from auxCat SPS modeling, 
    and convert to a color if bands is passed in, otherwise return loaded data. If loaded 
    data is passed in with bands, do then magnitude computation without re-loading."""
    acKey = 'Subhalo_StellarPhot_' + simColorsModel

    if colorData is None:
        # load
        if simColorsModel == 'snap':
            colorData = groupCat(sP, fieldsSubhalos=['SubhaloStellarPhotometrics'])
        else:
            colorData = auxCat(sP, fields=[acKey])

    # early exit with full data?
    if bands is None:
        return colorData

    subhaloIDs = None

    # compute colors
    if simColorsModel == 'snap':
        gc_colors = stellarPhotToSDSSColor( colorData['subhalos'], bands )
    else:
        # which subhaloIDs do these colors correspond to?
        if 'subhaloIDs' in colorData:
            subhaloIDs = colorData['subhaloIDs']
        else:
            # could generate with a range() if this came up
            print(' warning: subhaloIDs not in [%s] auxCat.' % acKey)

        # band indices
        acBands = list(colorData[acKey+'_attrs']['bands'])
        i0 = acBands.index('sdss_'+bands[0])
        i1 = acBands.index('sdss_'+bands[1])

        # multiple projections per subhalo?
        if colorData[acKey].ndim == 3:
            if projs is None:
                print(' Warning: loadSimGalColors() projs unspecified, returning [random] by default.')
                projs = 'random'
            
            if projs == 'all':
                # return all
                gc_colors = colorData[acKey][:,i0,:] - colorData[acKey][:,i1,:]
            elif projs == 'random':
                # return one per subhalo, randomly chosen
                np.random.seed(42424242)
                nums = np.random.randint(0,high=colorData[acKey].shape[2],size=colorData[acKey].shape[0])
                all_inds = range(colorData[acKey].shape[0])
                gc_colors = colorData[acKey][all_inds,i0,nums] - colorData[acKey][all_inds,i1,nums]
            else:
                # otherwise, projs had better be an integer or a tuple
                assert isinstance(projs, (int,long,list,tuple))
                gc_colors = colorData[acKey][:,i0,projs] - colorData[acKey][:,i1,projs]
        else:
            # just one projection per subhalo
            gc_colors = colorData[acKey][:,i0] - colorData[acKey][:,i1]

    return gc_colors, subhaloIDs

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

@jit(nopython=True, nogil=True, cache=True)
def _dust_tau_model_lum(N_H,Z_g,ages_logGyr,metals_log,masses_msun,wave,A_lambda_sol,redshift,
    beta,Z_solar,gamma,N_H0,f_scattering,metals,ages,wave_ang,spec):
    """ Helper for sps.dust_tau_model_mag(). Cannot JIT a class member function, so it sits here. """

    # accumulate per star attenuated luminosity (wavelength dependent):
    obs_lum = np.zeros( wave.size, dtype=np.float32 )
    atten = np.zeros( wave.size, dtype=np.float32 )
    gamma_term = np.zeros( wave.size, dtype=np.float32 )

    gamma_w_lt200 = np.where(wave_ang < 2000.0)
    gamma_w_gt200 = np.where(wave_ang >= 2000.0)

    for i in range(N_H.size):
        # taking np.power() using gamma as a full array does ~6000 powers, need to avoid for efficiency
        gamma_val_lt200 = np.power(Z_g[i]/Z_solar, 1.35)
        gamma_val_gt200 = np.power(Z_g[i]/Z_solar, 1.6)

        gamma_term[gamma_w_lt200] = gamma_val_lt200
        gamma_term[gamma_w_gt200] = gamma_val_gt200

        # finish absorption tau and total tau
        tau_a = A_lambda_sol * np.power(1+redshift,beta) * gamma_term * (N_H[i]/N_H0)

        tau_lambda = tau_a * f_scattering

        # attenuation as a function of wavelength
        atten *= 0.0
        atten += 1.0 # reset to one

        # leave atten at 1.0 (no change) for tau_lambda->0, and use 1e-5 threshold to avoid
        # numerical truncation setting atten=0 for tau_lambda~0 (very small)
        w = np.where(tau_lambda >= 1e-5)
        atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]

        # bilinear interpolation: stellar population spectrum
        x = metals_log[i]
        y = ages_logGyr[i]

        ##x_ind = np.interp( x, metals, np.arange(metals.size) )
        ##y_ind = np.interp( y, ages,   np.arange(ages.size) )
        for x_ind0 in range(metals.size):
            if x < metals[x_ind0]:
                break
        for y_ind0 in range(ages.size):
            if y < ages[y_ind0]:
                break

        xt = (x - metals[x_ind0-1]) / (metals[x_ind0]-metals[x_ind0-1])
        yt = (y - ages[y_ind0-1]) / (ages[y_ind0]-ages[y_ind0-1])
        x_ind = (x_ind0-1) * (1 - xt) + x_ind0 * xt
        y_ind = (y_ind0-1) * (1 - yt) + y_ind0 * yt           

        # set indices out of bounds on purpose if we are in extrapolation regime, which then causes 
        # below x1_ind==x2_ind or y1_ind==y2_ind such that constant (not linear) extrapolation occurs
        if x < metals[0]:
            x_ind = -1.0
        if x > metals[-1]:
            x_ind = metals.size - 1
        if y < ages[0]:
            y_ind = -1.0
        if y > ages[-1]:
            y_ind = ages.size - 1

        # clip indices at [0,size] which leads to constant extrap (nearest grid edge value in that dim)
        x1_ind = np.int32(np.floor(x_ind))
        x2_ind = x1_ind + 1
        y1_ind = np.int32(np.floor(y_ind))
        y2_ind = y1_ind + 1

        if x1_ind < 0 or x2_ind < 0:
            x1_ind = 0
            x2_ind = 0
        if y1_ind < 0 or y2_ind < 0:
            y1_ind = 0
            y2_ind = 0
        if x1_ind > metals.size-1 or x2_ind > metals.size-1:
            x1_ind = metals.size-1
            x2_ind = metals.size-1
        if y1_ind > ages.size-1 or y2_ind > ages.size-1:
            y1_ind = ages.size-1
            y2_ind = ages.size-1

        spec_12 = spec[x1_ind,y2_ind,:]
        spec_21 = spec[x2_ind,y1_ind,:]
        spec_11 = spec[x1_ind,y1_ind,:]
        spec_22 = spec[x2_ind,y2_ind,:]

        x1 = metals[x1_ind]
        x2 = metals[x2_ind]
        y1 = ages[y1_ind]
        y2 = ages[y2_ind]

        # constant beyond edges, make denominator nonzero
        if x2_ind == x1_ind == metals.size-1:
            x2 += (metals[-1]-metals[-2])
        if x2_ind == x1_ind == 0:
            x1 -= (metals[1]-metals[0])
        if y2_ind == y1_ind == ages.size-1:
            y2 += (ages[-1]-ages[-2])
        if y2_ind == y1_ind == 0:
            y1 -= (ages[1]-ages[0])

        # interpolated 1D spectrum
        spectrum_local = spec_11*(x2-x)*(y2-y) + spec_21*(x-x1)*(y2-y) + \
                         spec_12*(x2-x)*(y-y1) + spec_22*(x-x1)*(y-y1)
        spectrum_local /= ((x2-x1)*(y2-y1))

        #spectrum_local = np.clip(spectrum_local, 0.0, np.inf) # enforce everywhere positive
        w = np.where(spectrum_local < 0.0)
        spectrum_local[w] = 0.0

        # accumulate attenuated contribution of this stellar population
        obs_lum += (spectrum_local * masses_msun[i]) * atten

    # return full attenuated spectrum, for later convlution with some band
    return obs_lum

@jit(nopython=True, nogil=True, cache=True)
def _dust_tau_model_lum_indiv(N_H,Z_g,ages_logGyr,metals_log,masses_msun,wave,A_lambda_sol,redshift,
    beta,Z_solar,gamma,N_H0,f_scattering,metals,ages,wave_ang,spec):
    """ Helper for sps.dust_tau_model_mag(). Cannot JIT a class member function, so it sits here. """

    # accumulate per star attenuated luminosity (wavelength dependent):
    obs_lum_indiv = np.zeros( (N_H.size, wave.size), dtype=np.float32 )
    atten = np.zeros( wave.size, dtype=np.float32 )
    gamma_term = np.zeros( wave.size, dtype=np.float32 )

    gamma_w_lt200 = np.where(wave_ang < 2000.0)
    gamma_w_gt200 = np.where(wave_ang >= 2000.0)

    for i in range(N_H.size):
        # taking np.power() using gamma as a full array does ~6000 powers, need to avoid for efficiency
        gamma_val_lt200 = np.power(Z_g[i]/Z_solar, 1.35)
        gamma_val_gt200 = np.power(Z_g[i]/Z_solar, 1.6)

        gamma_term[gamma_w_lt200] = gamma_val_lt200
        gamma_term[gamma_w_gt200] = gamma_val_gt200

        # finish absorption tau and total tau
        tau_a = A_lambda_sol * np.power(1+redshift,beta) * gamma_term * (N_H[i]/N_H0)

        tau_lambda = tau_a * f_scattering

        # attenuation as a function of wavelength
        atten *= 0.0
        atten += 1.0 # reset to one

        # leave atten at 1.0 (no change) for tau_lambda->0, and use 1e-5 threshold to avoid
        # numerical truncation setting atten=0 for tau_lambda~0 (very small)
        w = np.where(tau_lambda >= 1e-5)
        atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]

        # bilinear interpolation: stellar population spectrum
        x = metals_log[i]
        y = ages_logGyr[i]

        ##x_ind = np.interp( x, metals, np.arange(metals.size) )
        ##y_ind = np.interp( y, ages,   np.arange(ages.size) )
        for x_ind0 in range(metals.size):
            if x < metals[x_ind0]:
                break
        for y_ind0 in range(ages.size):
            if y < ages[y_ind0]:
                break

        xt = (x - metals[x_ind0-1]) / (metals[x_ind0]-metals[x_ind0-1])
        yt = (y - ages[y_ind0-1]) / (ages[y_ind0]-ages[y_ind0-1])
        x_ind = (x_ind0-1) * (1 - xt) + x_ind0 * xt
        y_ind = (y_ind0-1) * (1 - yt) + y_ind0 * yt           

        # set indices out of bounds on purpose if we are in extrapolation regime, which then causes 
        # below x1_ind==x2_ind or y1_ind==y2_ind such that constant (not linear) extrapolation occurs
        if x < metals[0]:
            x_ind = -1.0
        if x > metals[-1]:
            x_ind = metals.size - 1
        if y < ages[0]:
            y_ind = -1.0
        if y > ages[-1]:
            y_ind = ages.size - 1

        # clip indices at [0,size] which leads to constant extrap (nearest grid edge value in that dim)
        x1_ind = np.int32(np.floor(x_ind))
        x2_ind = x1_ind + 1
        y1_ind = np.int32(np.floor(y_ind))
        y2_ind = y1_ind + 1

        if x1_ind < 0 or x2_ind < 0:
            x1_ind = 0
            x2_ind = 0
        if y1_ind < 0 or y2_ind < 0:
            y1_ind = 0
            y2_ind = 0
        if x1_ind > metals.size-1 or x2_ind > metals.size-1:
            x1_ind = metals.size-1
            x2_ind = metals.size-1
        if y1_ind > ages.size-1 or y2_ind > ages.size-1:
            y1_ind = ages.size-1
            y2_ind = ages.size-1

        spec_12 = spec[x1_ind,y2_ind,:]
        spec_21 = spec[x2_ind,y1_ind,:]
        spec_11 = spec[x1_ind,y1_ind,:]
        spec_22 = spec[x2_ind,y2_ind,:]

        x1 = metals[x1_ind]
        x2 = metals[x2_ind]
        y1 = ages[y1_ind]
        y2 = ages[y2_ind]

        # constant beyond edges, make denominator nonzero
        if x2_ind == x1_ind == metals.size-1:
            x2 += (metals[-1]-metals[-2])
        if x2_ind == x1_ind == 0:
            x1 -= (metals[1]-metals[0])
        if y2_ind == y1_ind == ages.size-1:
            y2 += (ages[-1]-ages[-2])
        if y2_ind == y1_ind == 0:
            y1 -= (ages[1]-ages[0])

        # interpolated 1D spectrum
        spectrum_local = spec_11*(x2-x)*(y2-y) + spec_21*(x-x1)*(y2-y) + \
                         spec_12*(x2-x)*(y-y1) + spec_22*(x-x1)*(y-y1)
        spectrum_local /= ((x2-x1)*(y2-y1))

        #spectrum_local = np.clip(spectrum_local, 0.0, np.inf) # enforce everywhere positive
        w = np.where(spectrum_local < 0.0)
        spectrum_local[w] = 0.0

        # accumulate attenuated contribution of this stellar population
        obs_lum_indiv[i,:] = (spectrum_local * masses_msun[i]) * atten

    # return full attenuated spectra, for later convlution with bands
    return obs_lum_indiv

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
        if '_res' not in self.dustModel:
            return
            
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
                lambda_nm = np.array([self.lambda_eff[band]])

            if '_res_conv' in self.dustModel:
                # do full convolution of original stellar spectrum with tau(lambda)
                #lambda_nm = self.trans_lambda[band] # at the resolution of the transmission function
                lambda_nm = self.wave # at the resolution of the stellar spectra

            # get tau^a factor from absorption (Cardelli 1989 equations 1-3b)
            x = 1/(lambda_nm/1000) # inverse microns

            R_V = 3.1 # e.g. MW/LMC value (2.7 for SMC)
            a_x = np.zeros( lambda_nm.size, dtype='float32' )
            b_x = np.zeros( lambda_nm.size, dtype='float32' )

            # infrared regime (0.91 um < lambda < 3.3 um) 
            w_lt = np.where(x < 1.1)

            a_x[w_lt] = 0.574 * x[w_lt]**1.61
            b_x[w_lt] = -0.527 * x[w_lt]**1.61

            # optical/NIR regime (0.3 um < lambda < 0.91)
            w_gt = np.where(x >= 1.1)
            y = x[w_gt] - 1.82

            a_x[w_gt] = 1 + 0.17699*y**1 - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 \
                              + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7
            b_x[w_gt] = 0 + 1.41338*y**1 + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 \
                              - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7

            # UV regime (0.125 um < lambda < 0.3 um)
            w_gt = np.where(x >= 3.3)
            w_gt59 = np.where(x >= 5.9)

            F_a = np.zeros( lambda_nm.size, dtype='float32' )
            F_b = np.zeros( lambda_nm.size, dtype='float32' )

            F_a[w_gt59] = -0.04473*(x[w_gt59]-5.9)**2 - 0.009779*(x[w_gt59]-5.9)**3
            F_b[w_gt59] =  0.21300*(x[w_gt59]-5.9)**2 + 0.120700*(x[w_gt59]-5.9)**3

            a_x[w_gt] = 1.752 - 0.316*x[w_gt] - 0.104/((x[w_gt]-4.67)**2 + 0.341) + F_a[w_gt]
            b_x[w_gt] = -3.09 + 1.825*x[w_gt] + 1.206/((x[w_gt]-4.62)**2 + 0.263) + F_b[w_gt]

            # far-UV regime (0.1 um < lambda < 0.125)
            w_gt = np.where(x >= 8.0)

            a_x[w_gt] = -1.073 - 0.628*(x[w_gt]-8) + 0.137*(x[w_gt]-8)**2 - 0.070*(x[w_gt]-8)**3
            b_x[w_gt] = 13.670 + 4.257*(x[w_gt]-8) - 0.420*(x[w_gt]-8)**2 + 0.374*(x[w_gt]-8)**3

            # outside scope
            w_gt = np.where(x > 10.0) # these values are >1 and growing, but divergent as lambda->0
            a_x[w_gt] = a_x[np.where(x < 10.0)].max()
            b_x[w_gt] = b_x[np.where(x < 10.0)].max()
            w_lt = np.where(x < 0.3) # these values anyways tend to zero
            a_x[w_lt] = 0.0
            b_x[w_lt] = 0.0

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

            w_lt = np.where(lambda_nm <= 346.0)
            w_gt = np.where(lambda_nm > 346.0)

            omega_lambda[w_lt] = 0.43 + 0.366 * (1 - np.exp(-(yy[w_lt]-3)*(yy[w_lt]-3)/0.2))
            omega_lambda[w_gt] = -0.48*yy[w_gt] + 2.41

            # note these are only valid in (100 nm < lambda < 1000 nm) and the given functional forms 
            # are not so well behaved outside this range, so we should probably enforce constant value 
            # at the edges of this range if we were ever to extend the used bands
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

        # account for population mass
        mags -= 2.5 * masses_logMsun

        return mags

    def mags_code_units(self, sP, band, gfm_sftime, gfm_metallicity, masses_code, retFullSize=False):
        """ Do unit conversions (and wind particle filtering) on inputs, and return mags() results. 
        If retFullSize is True, return same size as inputs with wind set to nan, otherwise filter 
        out wind/nan values and compress return size. """
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

        # return an array of the same size of the input, with nan for wind entries
        if retFullSize:
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
                                        retFullSize=True)

        # convert to luminosities
        lums = np.zeros( mags.size, dtype='float32' )
        lums.fill(np.nan)

        ww = np.isfinite(mags)
        lums[ww] = np.power(10.0, -0.4 * mags[ww])
        
        return lums

    def dust_tau_model_mags(self, bands, N_H, Z_g, ages_logGyr, metals_log, masses_msun, ret_indiv=False):
        """ For a set of stars characterized by their (age,Z,M) values as well as (N_H,Z_g) 
        calculated from the resolved gas distribution, do the Model (C) attenuation on the 
        full spectra, sum together, and convolve the resulting total L(lambda) with the  
        transmission function of multiple bands, returning a dict of magnitudes, one for 
        each band. """
        assert N_H.size == Z_g.size == ages_logGyr.size == metals_log.size == masses_msun.size
        assert N_H.ndim == Z_g.ndim == ages_logGyr.ndim == metals_log.ndim == masses_msun.ndim == 1

        bands = iterable(bands)
        for band in bands:
            assert band in self.bands

        r = {}

        if '_conv' in self.dustModel:
            # the aggregate spectrum is actually band-independent, get now from JITed helper function
            # band is any member of self.bands, since A_lambda_sol,gamma,f_scattering are all actually 
            # band-independent in this case where self.lambda_nm == self.wave
            if ret_indiv:
                obs_lum = _dust_tau_model_lum_indiv(N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                              self.wave,self.A_lambda_sol[band],self.sP.redshift,self.beta,
                              self.sP.units.Z_solar,self.gamma[band],self.N_H0,self.f_scattering[band],
                              self.metals,self.ages,self.wave_ang,self.spec)
            else:
                obs_lum = _dust_tau_model_lum(N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                                  self.wave,self.A_lambda_sol[band],self.sP.redshift,self.beta,
                                  self.sP.units.Z_solar,self.gamma[band],self.N_H0,self.f_scattering[band],
                                  self.metals,self.ages,self.wave_ang,self.spec)

        for band in bands:
            if '_eff' in self.dustModel:
                assert ret_indiv is False
                # the aggregate spectrum is band-dependent, but its calculation is very fast since
                # the lambda_nm is a single value instead of a ~6000 element array
                obs_lum = _dust_tau_model_lum(N_H,Z_g,ages_logGyr,metals_log,masses_msun,
                              self.wave,self.A_lambda_sol[band],self.sP.redshift,self.beta,
                              self.sP.units.Z_solar,self.gamma[band],self.N_H0,self.f_scattering[band],
                              self.metals,self.ages,self.wave_ang,self.spec)

            # convolve with band (trapezoidal rule)
            if not ret_indiv:
                # return total band magnitude of all star particles combined
                obs_lum_conv = obs_lum * self.trans_normed[band]

                nn = self.wave.size
                band_lum = np.sum( np.abs(self.wave_ang[1:nn-1]-self.wave_ang[0:nn-2]) * \
                    (obs_lum_conv[1:nn-1] + obs_lum_conv[0:nn-2])*0.5 )

                assert band_lum > 0.0

                r[band] = -2.5 * np.log10(band_lum) - 48.60 - 2.5*self.mag2cgs
            else:
                # return band magnitude individually for each star particle
                r[band] = np.zeros( obs_lum.shape[0], dtype='float32' )

                for i in range(obs_lum.shape[0]):
                    obs_lum_conv = np.squeeze(obs_lum[i,:]) * self.trans_normed[band]

                    nn = self.wave.size
                    band_lum = np.sum( np.abs(self.wave_ang[1:nn-1]-self.wave_ang[0:nn-2]) * \
                        (obs_lum_conv[1:nn-1] + obs_lum_conv[0:nn-2])*0.5 )

                    assert band_lum > 0.0

                    r[band][i] = -2.5 * np.log10(band_lum) - 48.60 - 2.5*self.mag2cgs

        return r

    def resolved_dust_mapping(self, pos_in, hsml, mass_nh, quant_z, pos_stars_in, 
                              projCen, projVec=None, rotMatrix=None, pxSize=1.0):
        """ Compute line of sight quantities per star for a resolved dust attenuation calculation.
        Gas (pos,hsml,mass_nh,quant_z) and stars (pos_stars) are used for the gridding of the gas 
        and the target (star) list. projVec is a [3]-vector, and the particles are rotated about 
        projCen [3] such that it aligns with the projection direction. pxSize is in physical kpc. """
        assert projCen.size == 3
        assert projVec is not None or rotMatrix is not None

        # rotation
        axes = [0,1]

        if rotMatrix is None:
            targetVec = np.array( [0,0,1], dtype='float32' )
            rotMatrix = rotationMatrixFromVec(projVec, targetVec)
            assert projVec.size == 3
        else:
            assert rotMatrix.size == 9

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

        # efficiency cut: neutral hydrogen mass in a cell less than 1e-6 times the target cell mass, 
        # and in large (h > 2.5 * softening) cells, clip to zero to avoid sph deposition calculation
        massThreshold = 1e-6 * self.sP.targetGasMass
        sizeThreshold = 2.5 * self.sP.gravSoft

        ww = np.where( (mass_nh < massThreshold) & (hsml > sizeThreshold) )
        mass_nh[ww] = 0.0

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

def debug_dust_plots():
    """ Plot intermediate aspects of the resolved dust calculation. """
    import matplotlib.pyplot as plt
    from util import simParams
    from util.helper import logZeroNaN

    sP = simParams(res=1820,run='tng',redshift=0.0)

    bands = ['sdss_u','sdss_g']#,'sdss_r','sdss_i','sdss_z','wfc_acs_f606w']
    iso = 'padova07'
    imf = 'chabrier'
    dust = 'cf00_res_conv' # _eff, _conv

    marker = 'o' if '_eff' in dust else ''
    xm_label = 'FSPS Master $\lambda$ [nm]'
    xf_label = 'Filter $\lambda_{eff}$ [nm]' if '_eff' in dust else xm_label
    master_lambda_range = [0,2000]

    pop = sps(sP, iso, imf, dust)

    for band in bands:
        # set up a calculation
        N_H = np.array([pop.N_H0 * 0.5])
        Z_g = np.array([sP.units.Z_solar * 0.8])
        age_logGyr = pop.ages[80]
        #age_logGyr = np.array(-3.9) # out of bounds left
        #age_logGyr = np.array(1.6) # out of bounds right
        mass_msun = 1e6
        metal_log = pop.metals[15]
        #print(band,N_H,Z_g,age_logGyr,mass_msun,metal_log)

        # go through a calculation
        tau_a = pop.A_lambda_sol[band] * (1+sP.redshift)**pop.beta * \
                (Z_g/sP.units.Z_solar)**pop.gamma[band] * (N_H/pop.N_H0)
        tau_lambda = tau_a * pop.f_scattering[band]

        atten = np.ones( tau_lambda.size, dtype='float32' )
        w = np.where(tau_lambda >= 1e-5)
        atten[w] = (1 - np.exp(-tau_lambda[w])) / tau_lambda[w]
        
        if pop.lambda_nm[band].size > 1: # _conv models
            atten = np.interp( pop.wave, pop.lambda_nm[band], atten )

        # bilinear interpolation: stellar population spectrum
        x_ind = np.interp( metal_log,  pop.metals, np.arange(pop.metals.size) )
        y_ind = np.interp( age_logGyr, pop.ages,   np.arange(pop.ages.size) )
        #assert x_ind == np.int32(x_ind)
        #assert y_ind == np.int32(y_ind)
        x_ind = np.int32(x_ind)
        y_ind = np.int32(y_ind)

        spectrum_local = np.array(pop.spec[x_ind,y_ind,:])
        spectrum_local = np.clip(spectrum_local, 0.0, np.inf) # enforce everywhere positive

        # accumulate attenuated contribution of this stellar population
        obs_lum = np.zeros( pop.wave.size, dtype='float32' )
        obs_lum += (spectrum_local * mass_msun) * atten

        obs_lum_noatten = np.zeros( pop.wave.size, dtype='float32' )
        obs_lum_noatten += (spectrum_local * mass_msun)

        # convolve with band (trapezoidal rule)
        obs_lum_conv = obs_lum * pop.trans_normed[band]
        obs_lum_noatten *= pop.trans_normed[band]

        nn = pop.wave.size
        band_lum = np.sum( np.abs(pop.wave_ang[1:nn-1]-pop.wave_ang[0:nn-2]) * \
                            (obs_lum_conv[1:nn-1] + obs_lum_conv[0:nn-2])*0.5 )
        band_lum_noatten = np.sum( np.abs(pop.wave_ang[1:nn-1]-pop.wave_ang[0:nn-2]) * \
                            (obs_lum_noatten[1:nn-1] + obs_lum_noatten[0:nn-2])*0.5 )

        assert band_lum > 0.0
        assert band_lum_noatten > 0.0

        result_mag = -2.5 * np.log10(band_lum) - 48.60 - 2.5*pop.mag2cgs
        result_mag_noatten = -2.5 * np.log10(band_lum_noatten) - 48.60 - 2.5*pop.mag2cgs

        # get magnitude without using our method of convolution
        ages_logGyr = np.array([age_logGyr])
        metals_log = np.array([metal_log])
        masses_msun = np.array([mass_msun])

        mag = pop.mags(band, ages_logGyr, metals_log, np.log10(masses_msun))

        # call our actual function and accelerated function to verify correctness
        NstarsTodo = 1000
        N_H = np.ones( NstarsTodo ) * N_H
        Z_g = np.ones( NstarsTodo ) * Z_g
        ages_logGyr = np.ones( NstarsTodo ) * ages_logGyr
        metals_log = np.ones( NstarsTodo ) * metals_log
        masses_msun = np.ones( NstarsTodo ) * masses_msun

        mag_f = pop.dust_tau_model_mag(band, N_H, Z_g, ages_logGyr, metals_log, masses_msun)

        print(band,mag_f,result_mag-2.5*np.log10(NstarsTodo)) #,result_mag_noatten,mag)

        # start figure
        fig = plt.figure(figsize=(22,14))

        ax = fig.add_subplot(3,4,1)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('$(A_\lambda / A_V)_\odot$')
        ax.plot( pop.lambda_nm[band], pop.A_lambda_sol[band], marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,2)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('f_scattering')
        ax.plot( pop.lambda_nm[band], pop.f_scattering[band], marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,3)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('$\gamma$')
        ax.plot( pop.lambda_nm[band], pop.gamma[band], marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,5)
        ax.set_xlabel('Filter $\lambda$ [nm]')
        ax.set_ylabel('Filter Transmission')
        ax.plot( pop.trans_lambda[band], pop.trans_val[band] )

        ax = fig.add_subplot(3,4,6)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel('Filter Interp-Trans')
        ax.plot( pop.wave, pop.trans_normed[band]*pop.wave_ang )

        # plots of spectrum
        ax = fig.add_subplot(3,4,4)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel('log spec [L$_\odot$/Hz]')
        ax.plot( pop.wave, logZeroNaN(spectrum_local), label='fsps mag=%g' % mag )
        ax.legend()

        ax = fig.add_subplot(3,4,8)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('log spec*mass*atten')
        ax.plot( pop.wave, logZeroNaN(obs_lum), label='mag=%g' % result_mag )
        ax.legend()
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,11)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel('log spec*mass*atten conv')
        ax.plot( pop.wave, logZeroNaN(obs_lum_conv) )

        ax = fig.add_subplot(3,4,12)
        ax.set_xlim(master_lambda_range)
        ax.set_xlabel(xm_label)
        ax.set_ylabel('log spec*mass convolved')
        ax.plot( pop.wave, logZeroNaN(obs_lum_noatten), label='mag=%g' % result_mag_noatten )
        ax.legend()

        # other plots
        ax = fig.add_subplot(3,4,7)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('$\\tau_a$')
        ax.plot( pop.lambda_nm[band], tau_a, marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,9)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('$\\tau_\lambda$')
        ax.plot( pop.lambda_nm[band], tau_lambda, marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        ax = fig.add_subplot(3,4,10)
        ax.set_xlabel(xf_label)
        ax.set_ylabel('L$_{obs}(\lambda)$ / L$_i(\lambda)$')
        ax.plot( pop.lambda_nm[band], atten, marker=marker )
        if pop.lambda_nm[band].size != 1:
            ax.set_xlim(master_lambda_range)

        fig.tight_layout()    
        fig.savefig('debug_%s_%s.pdf' % (dust,band))
        plt.close(fig)
