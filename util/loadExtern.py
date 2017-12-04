"""
loadExtern.py
  load external data files (observational points, etc) and Arepo txt files
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
from util.helper import evenlySample, running_median
from os.path import isfile, expanduser
from scipy import interpolate
from scipy.signal import savgol_filter
from collections import OrderedDict

logOHp12_solar = 8.69 # Asplund+ (2009) Table 1

dataBasePath = expanduser("~") + '/python/data/'

def behrooziSMHM(sP, logHaloMass=None, redshift=0.1):
    """ Load from data files: Behroozi+ (2013) abundance matching, stellar mass / halo mass relation. """
    basePath = dataBasePath + 'behroozi/release-sfh_z0_z8_052913/smmr/'
    if redshift == 0.0: redshift = 0.1
    fileName = 'c_smmr_z%.2f_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat' % redshift
    assert isfile(basePath + fileName)

    # columns: log10(halo_mass), log10(stellar_mass/halo_mass), err_up (dex), err_down (dex)
    data = np.loadtxt(basePath+fileName)

    r = { 'haloMass'  : data[:,0], 
          'smhmRatio' : data[:,1],
          'errorUp'   : data[:,2], 
          'errorDown' : data[:,3] }

    # if halo mass input, return the predicted stellar mass [log Msun] given the AM results
    if logHaloMass is not None:
        fRatio = interpolate.interp1d(r['haloMass'],r['smhmRatio'],  'cubic')
        logSmhmRatio = fRatio(logHaloMass)
        logStellarMass = 10.0**(logSmhmRatio) * 10.0**logHaloMass
        return np.log10(logStellarMass)

    r['y_low']  = 10.0**( r['smhmRatio']-r['errorDown'] ) / (sP.omega_b/sP.omega_m)
    r['y_mid']  = 10.0**r['smhmRatio'] / (sP.omega_b/sP.omega_m)
    r['y_high'] = 10.0**( r['smhmRatio']+r['errorUp'] ) / (sP.omega_b/sP.omega_m)

    # interpolated version for smoothness
    r['haloMass_i'] = np.linspace(r['haloMass'].min(), r['haloMass'].max(), 200)
    r['y_low_i']  = interpolate.interp1d(r['haloMass'],r['y_low'],  'cubic')(r['haloMass_i'])
    r['y_mid_i']  = interpolate.interp1d(r['haloMass'],r['y_mid'],  'cubic')(r['haloMass_i'])
    r['y_high_i'] = interpolate.interp1d(r['haloMass'],r['y_high'], 'cubic')(r['haloMass_i'])
    
    return r

def behrooziSFRAvgs():
    """ Load from data files: Behroozi+ (2013) average SFR histories in halo mass bins. """
    haloMassBins = np.linspace(10.0,15.0,26) # 0.2 spacing #[11.0,12.0,3.0,14.0,15.0]
    #basePath  = dataBasePath + 'behroozi/release-sfh_z0_z8_052913/sfr/' # from website
    basePath  = dataBasePath + 'behroozi/analysis/' # private communication
    fileNames = ['sfr_corrected_'+str(haloMass)+'.dat' for haloMass in haloMassBins]

    # filenames have halo mass at z=0 in log msun (e.g. 11.0 is halos from 11.0 to 11.2)
    # columns: Scale factor, SFR, Err_Up, Err_Down (all linear units)
    r = {}
    r['haloMassBins'] = haloMassBins
    r['haloBinSize']  = 0.2

    for i, fileName in enumerate(fileNames):
        data = np.loadtxt(basePath+fileName)

        massbin = { 'scaleFac'  : data[:,0], 
                    'sfr'       : savgol_filter(data[:,1],9,3), 
                    'errorUp'   : savgol_filter(data[:,2],9,3),
                    'errorDown' : savgol_filter(data[:,3],9,3) }

        massbin['redshift'] = 1/massbin['scaleFac'] - 1

        r[str(haloMassBins[i])] = massbin

    return r

def behrooziObsSFRD():
    """ Load observational data point compilation of SFRD(z) from Behroozi+ (2013). """
    basePath = dataBasePath + 'behroozi/behroozi-2013-data-compilation/'
    fileName = 'csfrs_new.dat'

    # Columns: redshift, Log10(CSFR) (Msun/yr/Mpc^3), Err+ (dex), Err- (dex)
    data = np.loadtxt(basePath+fileName)

    r = { 'redshift'  : data[:,0], 
          'sfrd'      : data[:,1],
          'errorUp'   : data[:,2], 
          'errorDown' : data[:,3] }

    # convert errorUp, errorDown to linear deltas
    r['errorUp']   = 10.0**(r['sfrd']+r['errorUp']) - 10.0**r['sfrd']
    r['errorDown'] = 10.0**r['sfrd'] - 10.0**(r['sfrd']-r['errorDown'])
    r['sfrd']      = 10.0**r['sfrd']
    r['label']     = 'Behroozi+ (2013 comp)'

    return r

def bouwensSFRD2014():
    """ Load observational data points from Bouwens+ (2014): arXiv:1211.2230. """
    z_vals    = np.array([3.8,   5.0,   5.9,   6.8,   8.0,   9.2,   10.0])
    z_errs    = np.array([0.5,   0.6,   0.4,   0.6,   0.7,   0.8,   0.8]) # approximate, visual
    sfrd_corr = np.array([-1.10, -1.36, -1.67, -1.83, -2.17, -2.87, -3.45]) # dust corrected (L>0.05Lstar)
    sfrd_errs = np.array([0.05,  0.06,  0.08,  0.10,  0.11,  0.45,  0.36])

    r = {}
    r['redshift']    = z_vals
    r['redshiftErr'] = z_errs

    r['errorUp']   = 10.0**(sfrd_corr+sfrd_errs) - 10.0**sfrd_corr
    r['errorDown'] = 10.0**sfrd_corr - 10.0**(sfrd_corr-sfrd_errs)
    r['sfrd']      = 10.0**sfrd_corr
    r['label']     = 'Bouwens+ (2014)'

    return r

def mosterSMHM(sP, redshift=0.0):
    """ Load from data files: Moster+ (2013) abundance matching, stellar mass / halo mass relation. """
    def f2013(mass, ind, redshift):
        """ Eqn. 2 of Moster+ (2013) with redshift dependent parameters from Eqns 11-14 and the 
            best fit values and 1sigma scatter as given in Table 1. 
              ind=1 : best fit relation, ind=0: 1sigma lower envelope, ind=2: 1sigma upper envelope
        """
        zFac = redshift / (1+redshift)

        M10 = 11.590
        M11 = 1.195
        N10 = 0.0351
        N11 = -0.0247
        beta_10  = 1.376
        beta_11  = -0.826
        gamma_10 = 0.608
        gamma_11 = 0.329

        M10s = 0.236
        M11s = 0.353
        N10s = 0.0058
        N11s = 0.0069
        beta_10s = 0.153
        beta_11s = 0.225
        gamma_10s = 0.059
        gamma_11s = 0.173

        # best-fit center line
        if ind == 1:
            log_M1_z = M10 + M11*zFac
            N_z      = N10 + N11*zFac
            beta_z   = beta_10  + beta_11*zFac
            gamma_z  = gamma_10 + gamma_11*zFac

            M1_z = 10.0**log_M1_z
            return 2.0 * N_z / ( (mass/M1_z)**(-beta_z) + (mass/M1_z)**(gamma_z) )

        # envelopes: sample and return max or min
        np.random.seed(424242)
        nPts = 500

        log_M1_z = np.random.normal(M10,M10s,nPts) + np.random.normal(M11,M11s,nPts)*zFac
        N_z      = np.random.normal(N10,N10s,nPts) + np.random.normal(N11,N11s,nPts)*zFac
        beta_z   = np.random.normal(beta_10,beta_10s,nPts) + np.random.normal(beta_11,beta_11s,nPts)*zFac
        gamma_z  = np.random.normal(gamma_10,gamma_10s,nPts) + np.random.normal(gamma_11,gamma_11s,nPts)*zFac       

        M1_z = 10.0**log_M1_z

        r = np.zeros( mass.size, dtype='float32' )

        for i in np.arange(mass.size):
            vals = 2.0 * N_z / ( (mass[i]/M1_z)**(-beta_z) + (mass[i]/M1_z)**(gamma_z) )
            if ind == 0:
                r[i] = vals.mean() - vals.std()
            if ind == 2:
                r[i] = vals.mean() + vals.std()

        return r

    def f2009(mass, ind=1):
        """ Eqn. 2 of Moster+ (2009) with default values of Table 2 (the model including scatter). 
              ind=1 : best fit relation, ind=0: 1sigma lower envelope, ind=2: 1sigma upper envelope
        """
        N      = 0.02817 + np.array( [-0.00057, 0.0, +0.00063])
        log_M1 = 11.899  + np.array( [-0.024,   0.0, +0.026])
        beta   = 1.068   + np.array( [-0.044,   0.0, +0.051])
        gamma  = 0.611   + np.array( [-0.010,   0.0, +0.012])

        M1 = 10.0**log_M1
        return 2.0 * N[ind] / ( (mass/M1[ind])**(-beta[ind]) + (mass/M1[ind])**(gamma[ind])  )

    r = {}
    r['haloMass'] = np.linspace(8.0, 16.0, num=200)
    r['y_low']  = f2013( 10.0**r['haloMass'], ind=0, redshift=redshift ) / (sP.omega_b/sP.omega_m)
    r['y_mid']  = f2013( 10.0**r['haloMass'], ind=1, redshift=redshift ) / (sP.omega_b/sP.omega_m)
    r['y_high'] = f2013( 10.0**r['haloMass'], ind=2, redshift=redshift ) / (sP.omega_b/sP.omega_m)

    return r

def kravtsovSMHM(sP):
    """ Load from data files: Kravtsov+ (2014) abundance matching, stellar mass / halo mass relation. """
    def f(x, alpha, delta, gamma):
        term1 = -1.0 * np.log10(10.0**(alpha*x) + 1.0)
        term2 = delta * (np.log10(1+np.exp(x)))**gamma / ( 1 + np.exp(10.0**(-x)) )
        return term1+term2

    def k2014(mass):
        """ Eqn. A3 and A4 of Kravtsov+ (2014) with Mvir (w/ scatter) or M200c (w/ scatter) best fit values. """
        # halo mass definition: M200c
        log_M1 = 11.35
        log_eps = -1.642
        alpha = -1.779
        delta = 4.394
        gamma = 0.547

        # halo mass definition: Mvir
        #log_M1 = 11.39
        #log_eps = -1.685
        #alpha = -1.740 # typo? Table 3 has positive values
        #delta = 4.335
        #gamma = 0.531

        M1 = 10.0**log_M1
        log_Mstar = np.log10(10.0**log_eps * M1) + \
                    f(np.log10(mass/M1),alpha,delta,gamma) - \
                    f(0.0,alpha,delta,gamma)
        return (10.0**log_Mstar / mass)

    r = {}
    r['haloMass'] = np.linspace(9.0, 16.0, num=200)
    #r['y_low']  = ( k2014( 10.0**r['haloMass'] )-10.0**0.05 ) / (sP.omega_b/sP.omega_m)
    r['y_mid']  = k2014( 10.0**r['haloMass'] ) / (sP.omega_b/sP.omega_m)
    #r['y_high'] = ( k2014( 10.0**r['haloMass'] )+10.0**0.05 ) / (sP.omega_b/sP.omega_m)

    return r

def kormendyHo2013():
    """ Best fit black hole / stellar bulge mass relations and observed points from Kormendy & Ho (2013). """
    M_0 = 10.0**11

    M_bulge = 10.0**np.linspace(8.0, 13.0, 100)
    M_BH  = 10.0**9 * (0.49) * (M_bulge/M_0)**1.16 # Msun, Eqn 10

    w = np.where(M_bulge > M_0)[0].min()
    errorUp   = np.zeros( M_BH.size )
    errorDown = np.zeros( M_BH.size )

    # below characteristic mass
    errorUp[:w]   = 10.0**9 * (0.49+0.06) * (M_bulge[:w]/M_0)**(1.16-0.08)
    errorDown[:w] = 10.0**9 * (0.49-0.05) * (M_bulge[:w]/M_0)**(1.16+0.08)

    # above characteristic mass
    errorUp[w:]   = 10.0**9 * (0.49+0.06) * (M_bulge[w:]/M_0)**(1.16+0.08)
    errorDown[w:] = 10.0**9 * (0.49-0.05) * (M_bulge[w:]/M_0)**(1.16-0.08)

    r = { 'M_bulge'   : np.log10(M_bulge),
          'M_BH'      : np.log10(M_BH),
          'errorUp'   : np.log10(errorUp),
          'errorDown' : np.log10(errorDown),
          'label'     : 'Kormendy & Ho (2013) M$_{\\rm BH}$-M$_{\\rm bulge}$' }

    return r

def mcconnellMa2013():
    """ Best fit black hole / stellar bulge mass relations from McConnell & Ma (2013). """
    # load data file (blackhole.berkeley.edu)
    r = {}
    path = dataBasePath + 'mcconnell/current_ascii.txt'

    # Columns: galName, dist, MBH [Msun], MBH lower (68%), MBH upper (68%), method, sigma [km/s]
    #          sigma lower, sigma upper, log(LV/Lsun), error in log(LV/Lsun), log(L3.6/Lsun),
    #          error in log(L3.6/Lsun), Mbulge [Msun], radius of influence [arcsec], morphology, 
    #          profile, reff (V), reff (I), reff (3.6)
    data = np.genfromtxt(path, dtype=None) # array of 20 lists

    galName   = np.array([d[0] for d in data])
    M_BH      = np.array([d[2] for d in data])
    M_BH_up   = np.array([d[3] for d in data])
    M_BH_down = np.array([d[4] for d in data])
    M_bulge   = np.array([d[13] for d in data])

    w = np.where(M_bulge > 0.0)

    r['pts'] = { 'galName'   : galName[w], 
                 'M_BH'      : np.log10( M_BH[w] ),
                 'M_BH_up'   : np.log10( M_BH_up[w] ) - np.log10( M_BH[w] ), 
                 'M_BH_down' : np.log10( M_BH[w] ) - np.log10( M_BH_down[w] ),
                 'M_bulge'   : np.log10( M_bulge[w] ),
                 'label'     : 'McConnell & Ma (2013 comp)' }

    # fit: Table 2, M_BH - M_bulge relation, "Dynamical Masses" with Method MPFITEXY
    alpha     = 8.46
    alpha_err = 0.08
    beta      = 1.05
    beta_err  = 0.11
    eps_0     = 0.34
    M_0       = 10.0**11

    M_bulge = 10.0**np.linspace(8.0, 13.0, 100)
    log_M_BH  = alpha + beta * np.log10(M_bulge/M_0) # Msun, Eqn 10

    w = np.where(M_bulge > M_0)[0].min()
    errorUp   = np.zeros( log_M_BH.size )
    errorDown = np.zeros( log_M_BH.size )

    # below characteristic mass
    errorUp[:w]   = alpha+alpha_err + (beta-beta_err) * np.log10(M_bulge[:w]/M_0) 
    errorDown[:w] = alpha-alpha_err + (beta+beta_err) * np.log10(M_bulge[:w]/M_0) 

    # above characteristic mass
    errorUp[w:]   = alpha+alpha_err + (beta+beta_err) * np.log10(M_bulge[w:]/M_0) 
    errorDown[w:] = alpha-alpha_err + (beta-beta_err) * np.log10(M_bulge[w:]/M_0) 

    r['M_bulge']   = np.log10(M_bulge)
    r['M_BH']      = log_M_BH
    r['errorUp']   = errorUp
    r['errorDown'] = errorDown
    r['label']     = 'McConnell & Ma (2013) M$_{\\rm BH}$-M$_{\\rm bulge}$'

    return r

def baldry2012SizeMass():
    """ Load observational data points from Baldry+ (2012). """
    path = dataBasePath + 'baldry/size-mass.txt'

    def logPcToKpc(x):
        return 10.0**x / 1000.0

    # Columns: Ngal, stellar mass [log Msun], size 16th percentile, median size log [pc], 84th percentile
    #   first 11 rows: "Blue galaxies"
    #   next 9 rows: "Red galaxies"
    data = np.loadtxt(path)
    n = 11

    r = {}
    r['blue'] = { 'stellarMass' : data[:n,1], 
                  'sizeKpc'     : logPcToKpc(data[:n,3]),
                  'sizeKpcUp'   : logPcToKpc(data[:n,2]), 
                  'sizeKpcDown' : logPcToKpc(data[:n,4]),
                  'label'       : 'Baldry+ (2012) GAMA R$_{\\rm e}$ blue' }

    r['red'] = { 'stellarMass' : data[n:,1], 
                 'sizeKpc'     : logPcToKpc(data[n:,3]),
                 'sizeKpcUp'   : logPcToKpc(data[n:,2]), 
                 'sizeKpcDown' : logPcToKpc(data[n:,4]),
                 'label'       : 'Baldry+ (2012) GAMA R$_{\\rm e}$ red' }

    for t in ['red','blue']:
        r[t]['errorUp']   = r[t]['sizeKpcUp'] - r[t]['sizeKpc']
        r[t]['errorDown'] = r[t]['sizeKpc'] - r[t]['sizeKpcDown']

    return r

def lange2016SizeMass():
    """ Load observational data points from Lange+ (2016) GAMA. """
    r = {}

    def _plawFit(log_Mstar,a,b):
        # errors not yet implemented
        return a * np.power(10.0**log_Mstar/1e10, b) # kpc

    # Table 1 (selected rows), and Eqn 2
    stellarMassRange  = np.log10( [1e8, 1e11] ) # Msun, effective covered by sample
    stellarMassRange2 = np.log10( [2e10, 3e11] ) # Msun, most massive elliptical fit

    stellarMass  = np.linspace( stellarMassRange[0], stellarMassRange[1], 2)
    stellarMass2 = np.linspace( stellarMassRange2[0], stellarMassRange2[1], 2)

    r['stellarMass']  = stellarMass
    r['stellarMass2'] = stellarMass2

    for k in ['hubbletype','structural','combined']: r[k] = {}

    for k in ['E_gt2e10']: r['hubbletype'][k] = {}
    for k in ['late_disc','late_bulge','early_disc','early_bulge']: r['structural'][k] = {}
    for k in ['all_discs','global_late','E_ETB']: r['combined'][k] = {}

    r['hubbletype']['E_gt2e10']['sizeKpc'] = _plawFit(stellarMass2, 0.999, 0.786) # errs: 0.089, 0.048
    r['hubbletype']['E_gt2e10']['label']   = 'Lange+ (2016) GAMA R$_{\\rm e}$ E (>2e10)'

    r['structural']['late_disc']['sizeKpc']   = _plawFit(stellarMass, 6.939, 0.245)
    r['structural']['late_bulge']['sizeKpc']  = _plawFit(stellarMass, 4.041, 0.339)
    r['structural']['early_disc']['sizeKpc']  = _plawFit(stellarMass, 4.55, 0.247)
    r['structural']['early_bulge']['sizeKpc'] = _plawFit(stellarMass, 1.836, 0.267)

    r['structural']['late_disc']['label']   = 'Lange+ (2016) GAMA R$_{\\rm e}$ late discs'
    r['structural']['late_bulge']['label']  = 'Lange+ (2016) GAMA R$_{\\rm e}$ late bulges'
    r['structural']['early_disc']['label']  = 'Lange+ (2016) GAMA R$_{\\rm e}$ early discs'
    r['structural']['early_bulge']['label'] = 'Lange+ (2016) GAMA R$_{\\rm e}$ early bulges'

    r['combined']['all_discs']['sizeKpc']   = _plawFit(stellarMass, 5.56, 0.274)
    r['combined']['global_late']['sizeKpc'] = _plawFit(stellarMass, 4.104, 0.208)
    r['combined']['E_ETB']['sizeKpc']       = _plawFit(stellarMass, 2.033, 0.318)

    r['combined']['all_discs']['label']   = 'Lange+ (2016) GAMA R$_{\\rm e}$ all discs'
    r['combined']['global_late']['label'] = 'Lange+ (2016) GAMA R$_{\\rm e}$ global late'
    r['combined']['E_ETB']['label']       = 'Lange+ (2016) GAMA R$_{\\rm e}$ E + ETB'

    return r

def shen2003SizeMass():
    """ Load observational data points from Shen+ (2013). Table 1 and Eqns 17-19 (Fig 11). """
    def earlyTypeR(stellar_mass_msun, b=2.88e-6, a=0.56): # see Erratum for b coefficient
        rKpc = b * (stellar_mass_msun)**a
        return rKpc

    def lateTypeR(stellar_mass_msun, alpha=0.14, beta=0.39, gamma=0.10, M0=3.98e10):
        rKpc = gamma * (stellar_mass_msun)**alpha * (1 + stellar_mass_msun/M0)**(beta-alpha)
        return rKpc

    def stddevR(stellar_mass_msun, M0=3.98e10, sigma1=0.47, sigma2=0.34):
        ln_sigmaR = sigma2 + (sigma1-sigma2) / (1 + (stellar_mass_msun/M0)**2)
        return np.e**ln_sigmaR

    r = {}
    r['early'] = { 'stellarMass' : np.linspace(9.5, 12.0, 2),
                   'label'       : 'Shen+ (2003) SDSS R$_{\\rm e}$ early (n>2.5)' }
    r['late']  = { 'stellarMass' : np.linspace(8.0, 12.0, 2),
                   'label'       : 'Shen+ (2003) SDSS R$_{\\rm e}$ late (n<2.5)' }

    r['early']['sizeKpc'] = earlyTypeR(10.0**r['early']['stellarMass'])
    r['late']['sizeKpc']  = lateTypeR(10.0**r['late']['stellarMass'])

    for t in ['early','late']:
        r[t]['sizeKpcUp']   = r[t]['sizeKpc'] + r[t]['sizeKpc']/stddevR(10.0**r[t]['stellarMass'])
        r[t]['sizeKpcDown'] = r[t]['sizeKpc'] - r[t]['sizeKpc']/stddevR(10.0**r[t]['stellarMass'])

    return r

def baldry2008SMF():
    """ Load observational data points from Baldry+ (2008). """
    path = dataBasePath + 'baldry/gsmf-BGD08.txt'

    # Columns: log stellar mass (bin center), Ngal, ndens (/Mpc^3/dex), Poisson error, min n, max n
    data = np.loadtxt(path)

    r = { 'stellarMass' : data[:,0],
          'numDens'     : data[:,2],
          'numDensDown' : data[:,4],
          'numDensUp'   : data[:,5],        
          'label'       : 'Baldry+ (2008) SDSS z~0' }

    r['errorUp']   = r['numDensUp'] - r['numDens']
    r['errorDown'] = r['numDens'] - r['numDensDown']

    return r

def baldry2012SMF():
    """ Load observational data points from Baldry+ (2012). """
    path = dataBasePath + 'baldry/gsmf-B12.txt'

    # Columns: log mass, bin width, num dens, error, number in sample
    # number density is per dex per 10^3 Mpc^3, assuming H0=70 km/s/Mpc
    data = np.loadtxt(path)

    r = { 'stellarMass' : data[:,0],
          'numDens'     : data[:,2] * 1e-3,
          'error'       : data[:,3] * 1e-3,
          'label'       : 'Baldry+ (2012) GAMA z<0.05' }

    return r

def liWhite2009SMF(little_h=0.704):
    """ Load observational data ponts from Li & White (2009). Triple-Schechter fit of Table 1. """
    def fSchechter(M, phi_star, M_star, alpha):
        return phi_star * (M/M_star)**alpha * np.exp(-M/M_star)

    massRanges = [ np.log10( 10.0**np.array([8.00,9.33]) / little_h**2 ),
                   np.log10( 10.0**np.array([9.33,10.67]) / little_h**2 ),
                   np.log10( 10.0**np.array([10.67,12.00]) / little_h**2 )]
    phiStar    = np.array([10.0**0.0146,10.0**0.0132,10.0**0.0044]) * little_h**3
    alpha      = np.array([-1.13,-0.90,-1.99])
    mStar      = np.array([10.0**9.61,10.0**10.37,10.0**10.71]) / little_h**2

    phiStar_err = [0.0005,0.0007,0.0006]
    alpha_err   = [0.09,0.04,0.18]
    mStar_err   = [0.24,0.02,0.04]

    r = { 'stellarMass' : np.linspace(8.31,12.3,200),
          'numDens'     : np.zeros(200, dtype='float32'),
          'label'       : 'Li & White (2009) SDSS DR7' }

    for i, massRange in enumerate(massRanges):
        w = np.where((r['stellarMass'] >= massRange[0]) & (r['stellarMass'] < massRange[1]))
        r['numDens'][w] = fSchechter(10.0**r['stellarMass'][w],phiStar[i],mStar[i],alpha[i])

    # single Schechter fit
    phiStar   = 0.0083 * little_h**3 # +/- 0.0002, Mpc^(-3)
    alpha     = -1.155 # +/- 0.008
    log_mStar = 10.525 # +/- 0.005
    mStar     = 10.0**log_mStar / little_h**2 # Msun
    r['numDensSingle'] = fSchechter(10.0**r['stellarMass'],phiStar,mStar,alpha)
    raise Exception('Not finished, needs to be checked.')

    return r

def bernardi2013SMF():
    """ Load observational data points from Bernardi+ (2013). """
    models = ['Ser','SerExp','Ser_Simard','cmodel']
    paths = [dataBasePath + 'bernardi/MsF_' + m + '.dat' for m in models]

    # Columns: stellar mass (log msun), num dens (all), err, num dens (Ell), err, num dens (S0), err, 
    #          num dens (Sab), err, num dens (Scd), err
    # number densities are in log10( Mpc^-3 dex^-1 )
    r = {}

    for i, path in enumerate(paths):
        data = np.loadtxt(path)
        r[models[i]] = { 'stellarMass' : data[:,0], 
                         'numDens'     : 10.0**data[:,1],
                         'errorUp'     : 10.0**(data[:,1]+data[:,2]) - 10.0**data[:,1],
                         'errorDown'   : 10.0**data[:,1] - 10.0**(data[:,1]-data[:,2]),
                         'label'       : 'Bernardi+ (2013) SDSS '+models[i] }

    return r

def dsouza2015SMF():
    """ Load observational data points from D'Souza+ (2015) Fig 7. """
    path = dataBasePath + 'dsouza/ds2015_fig7.txt'

    # columns: log10(M_star*h^2), log10(Phi/h^3 / Mpc^3 / log10(Mstar)), y_err
    data = np.loadtxt(path,delimiter=',')

    little_h = 0.72

    Mstar  = 10.0**data[:,0] / little_h**2.0
    valMid = 10.0**(data[:,1]) * little_h**3.0
    valUp  = 10.0**((data[:,1]+data[:,2])) * little_h**3.0

    r = {'stellarMass' : np.log10(Mstar),
          'numDens'    : valMid,
          'error'      : valUp - valMid,
          'label'      : 'D\'Souza+ (2015) SDSS z~0.1'}

    return r

def song2015SMF(redshift):
    """ Load observational data points from Song+ (2015). """
    path = dataBasePath + 'song/song2015_table2.txt'
    assert redshift in [4.0,5.0,6.0,7.0,8.0]

    # redshift log_Mstar[msun] log_phi[1/Mpc^3/dex] log_err_up[1sigma] log_err_down[1sigma]
    data = np.loadtxt(path,delimiter=' ')
    #data = np.genfromtxt(path, dtype=None)

    w = np.where(data[:,0] == redshift)

    r = { 'zMin'        : np.squeeze( data[w,0] ),
          'stellarMass' : np.squeeze( data[w,1] ),
          'numDens'     : np.squeeze( data[w,2] ),
          'errorUp'     : np.squeeze( data[w,3] ),   
          'errorDown'   : np.squeeze( data[w,4] ),
          'label'       : 'Song+ (2015) CANDELS/GOODS/HUDF z=%d' % redshift }

    r['errorUp']   = 10.0**(r['numDens']+r['errorUp']) - 10.0**r['numDens']
    r['errorDown'] = 10.0**r['numDens'] - 10.0**(r['numDens']+r['errorDown'])
    r['numDens']   = 10.0**r['numDens']

    # rescale stellar masses from Salpeter to Chabrier
    fac_from_Salpeter_to_Chabrier = 0.61 # 0.66 for Kroupa
    r['stellarMass'] = np.log10(10.0**r['stellarMass'] * fac_from_Salpeter_to_Chabrier)

    return r

def grazian2015SMF(redshift):
    """ Load observational data points from Grazian+ (2015). """
    path = dataBasePath + 'grazian/grazian2015_fig6.txt'
    assert redshift in [3.5,4.5,5.5,6.5] # lower bin edges

    # zmin zmax Mstar[msun/h70^2] log_phi[h70^3/Mpc^3/dex] log_phi_up log_phi_down lowerlimit
    data = np.loadtxt(path,delimiter=',')

    w = np.where( (data[:,0] == redshift) & (data[:,6] == 0) ) # remove lower limits

    r = { 'zMin'         : np.squeeze( data[w,0] ),
          'zMax'         : np.squeeze( data[w,1] ),
          'stellarMass'  : np.squeeze( data[w,2] ),
          'numDens'      : np.squeeze( data[w,3] ),
          'numDensUp'    : np.squeeze( data[w,4] ),   
          'numDensDown'  : np.squeeze( data[w,5] ),
          'label'        : 'Grazian+ (2015) GOODS-South/UDS %.1f<z<%.1f' % (data[w,0].min(),data[w,1].max()) }

    r['errorUp']   = 10.0**r['numDensUp'] - 10.0**r['numDens']
    r['errorDown'] = 10.0**r['numDens'] - 10.0**r['numDensDown']
    r['numDens']   = 10.0**r['numDens']

    # rescale stellar masses from Salpeter to Chabrier
    fac_from_Salpeter_to_Chabrier = 0.61 # 0.66 for Kroupa
    r['stellarMass'] = np.log10(r['stellarMass'] * fac_from_Salpeter_to_Chabrier)

    return r

def caputi2015SMF(redshift):
    """ Load observational data points from Caputi+ (2015). """
    path = dataBasePath + 'caputi/caputi2015_table1.txt'
    assert redshift in [3.0,4.0] # lower bin edges

    # zmin zmax log_Mstar[msun] log_phi[1/Mpc^3/dex] err_up err_down
    data = np.loadtxt(path,delimiter=' ')

    # flag lower limits
    #lowerLims = np.zeros( data[:,0].size, dtype='bool' )
    #lowerLims[np.where(data[:,4] >= 1.0)] = True
    # remove lower limits:

    w = np.where( (data[:,0] == redshift) & (data[:,4] < 1.0) )

    r = { 'zMin'        : np.squeeze( data[w,0] ),
          'zMax'        : np.squeeze( data[w,1] ),
          'stellarMass' : np.squeeze( data[w,2] ),
          'numDens'     : np.squeeze( data[w,3] ),
          'errorUp'     : np.squeeze( data[w,4] ),  
          'errorDown'   : np.squeeze( data[w,5] ),
          #'lowerLimits' : np.squeeze( lowerLims[w] ),
          'label'       : 'Caputi+ (2015) COSMOS %.1f<z<%.1f' % (data[w,0].min(),data[w,1].max()) }

    r['errorUp']   = 10.0**(r['numDens']+r['errorUp']) - 10.0**r['numDens']
    r['errorDown'] = 10.0**r['numDens'] - 10.0**(r['numDens']+r['errorDown'])
    r['numDens']   = 10.0**r['numDens']

    # rescale stellar masses from Salpeter to Chabrier
    # note in this paper: "stellar masses have been multiplied by a factor of 1.7 to convert from 
    #   a Chabrier to a Salpeter IMF over (0.1-100) Msun." (factor of 0.59)
    fac_from_Salpeter_to_Chabrier = 0.61 # 0.66 for Kroupa
    r['stellarMass'] = np.log10(10.0**r['stellarMass'] * fac_from_Salpeter_to_Chabrier)

    return r

def davidzon2017SMF(redshift):
    """ Load observational data points from Davidzon+ (2017). """
    path = dataBasePath + 'davidzon/davidzon17_fig8.txt'
    assert redshift in [2.5,3.0,3.5,4.5] # lower bin edges

    # Columns: zmin zmax log_M[msun] log_phi[1/Mpc^3/dex] log_phi_up, log_phi_down
    data = np.loadtxt(path,delimiter=',')

    w = np.where(data[:,0] == redshift)

    r = { 'zMin'        : np.squeeze( data[w,0] ),
          'zMax'        : np.squeeze( data[w,1] ),
          'stellarMass' : np.squeeze( data[w,2] ),
          'numDens'     : np.squeeze( data[w,3] ),
          'numDensUp'   : np.squeeze( data[w,4] ),   
          'numDensDown' : np.squeeze( data[w,5] ),
          'label'       : 'Davidzon+ (2017) COSMOS %.1f<z<%.1f' % (data[w,0].min(),data[w,1].max()) }

    r['errorUp']   = 10.0**r['numDensUp'] - 10.0**r['numDens']
    r['errorDown'] = 10.0**r['numDens'] - 10.0**r['numDensDown']
    r['numDens'] = 10.0**r['numDens']

    return r

def gallazzi2005(sP):
    """ Load observational data points (M-Z and ages) from Gallazzi+ (2005). """
    path = dataBasePath + 'gallazzi/table2.txt'

    # columns: log(Mstar/Msun), log(Z/Zun) [P50, P16, P84], log(tr/yr) [P50, P16, P84]
    # rescale metallicities from old Z_solar=0.02 to present GS10 (0.0127) value
    data = np.loadtxt(path)

    r = { 'stellarMass'  : data[:,0], 
          'Zstars'       : np.log10(10.0**data[:,1] * 0.02 / sP.units.Z_solar),
          'ZstarsDown'   : np.log10(10.0**data[:,2] * 0.02 / sP.units.Z_solar),
          'ZstarsUp'     : np.log10(10.0**data[:,3] * 0.02 / sP.units.Z_solar),
          'ageStars'     : 10.0**(data[:,4] - 9.0), # log yr -> Gyr
          'ageStarsDown' : 10.0**(data[:,5] - 9.0),
          'ageStarsUp'   : 10.0**(data[:,6] - 9.0),
          'label'        : 'Gallazzi+ (2005) SDSS z<0.2' }

    return r

def bernardi10():
    """ Load observational data points (stellar ages) from Bernardi+ (2010). """
    path = dataBasePath + 'bernardi/b10_fig10.txt'

    # columns: Mstar (log10 Msun), Age (Gyr), Age_up (Gyr), Age_down (Gyr)
    data = np.loadtxt(path, delimiter=',')

    r = { 'stellarMass'  : data[:,0], 
          'ageStars'     : data[:,1],
          'ageStarsUp'   : data[:,2],
          'ageStarsDown' : data[:,3],
          'label'        : 'Bernardi+ (2010) SDSS, HB09 Early-Types' }

    return r

def woo2008(sP):
    """ Load observational data points (M-Z of local group dwarfs) from Woo+ (2008). """
    path = dataBasePath + 'woo/table1.txt'

    # columns: Name, log(Mstar/Msun), log(Z) where log(Z/0.019)=[Fe/H]
    # note: using instead Z_solar = 0.019 below would convert to [Fe/H] instead of Z/Z_solar
    data = np.genfromtxt(path, dtype=None)

    r = { 'name'           : np.array([d[0] for d in data]), 
          'stellarMass'    : np.array([d[1] for d in data]),
          'Zstars'         : np.log10(10.0**np.array([d[2] for d in data]) / sP.units.Z_solar),
          'stellarMassErr' : 0.17, # dex, average
          'ZstarsErr'      : 0.2, # dex, average
          'label'          : 'Woo+ (2008) Local Group' }

    return r

def kirby2013():
    """ Load observational data points (M-Z of local group dwarfs) from Kirby+ (2013). """
    path = dataBasePath + 'kirby/2013_table4.txt'

    # columns: Name, Num, Lv, Lv_err, log(Mstar/Msun), err, <[Fe/H]>, err, sigma, err, 
    #          median, mad, IQR, skewness, err, kurtosis, err
    # "assume a solar abundance of 12 + log(Fe/H) = 7.52" (this is within 5% of GS10)
    data = np.genfromtxt(path, dtype=None)

    r = { 'name'           : np.array([d[0] for d in data]), 
          'stellarMass'    : np.array([d[4] for d in data]),
          'stellarMassErr' : np.array([d[5] for d in data]),
          'Zstars'         : np.array([d[6] for d in data]),
          'ZstarsErr'      : np.array([d[7] for d in data]),
          'label'          : 'Kirby+ (2013) Local Group' }

    return r

def giodini2009(sP):
    """ Load observational data points (gas/stellar mass fractions in r500crit) from Giodini+ (2009). """
    # Table 2 (masses are M500/h_72 [Msun]) (errors are symmetric, stddev of the mean)
    M500h72Msun   = np.array([2.1e13, 5.1e13, 1.2e14, 3.0e14, 7.1e14])

    r = {'m500_logMsun'  : np.log10( M500h72Msun/(0.72/sP.HubbleParam) ),
         'fStars500'     : np.array([0.062, 0.045, 0.036, 0.021, 0.019]),
         'fStars500Err'  : np.array([0.005, 0.002, 0.004, 0.002, 0.002]),
         'fGas500'       : np.array([0.074, 0.068, 0.080, 0.103, 0.123]),
         'fGas500Err'    : np.array([0.028, 0.005, 0.003, 0.008, 0.007]),
         'fBaryon500'    : np.array([0.136, 0.113, 0.116, 0.124, 0.141]),
         'fBaryon500Err' : np.array([0.028, 0.005, 0.005, 0.009, 0.007]),
         'label'         : 'Giodini+ (2009) z<0.2' }

    return r

def lovisari2015(sP):
    """ Load observational data points (gas/total mass within r500crit) from Lovisari+ (2015). """
    # Table 2, 7th column is M500 in [10^13 / h70 Msun], 9th column is M500gas in [10^12 * h70^{-5/2} Msun]
    path = dataBasePath + 'lovisari/lovisari2015_table2.txt'

    data = np.genfromtxt(path, dtype=None)

    m500_tot = np.array([d[6] for d in data])
    m500_tot_err = np.array([d[7] for d in data])
    m500_gas = np.array([d[8] for d in data])
    m500_gas_err = np.array([d[9] for d in data])

    # we assume their units mean that h=0.7 is assumed and included in their numbers, so slightly
    # compensate for our cosmology
    m500_tot_Msun     = m500_tot * 1e13 / (0.70/sP.HubbleParam)
    m500_tot_Msun_err = m500_tot_err * 1e13 / (0.70/sP.HubbleParam)
    m500_gas_Msun     = m500_gas * 1e12 / (0.70/sP.HubbleParam)
    m500_gas_Msun_err = m500_gas_err * 1e12 / (0.70/sP.HubbleParam)

    frac_500_gas = m500_gas_Msun / m500_tot_Msun

    # maximal error estimate:
    #frac_500_gas_errUp   = (m500_gas_Msun+m500_gas_Msun_err) / (m500_tot_Msun-m500_tot_Msun_err)
    #frac_500_gas_errDown = (m500_gas_Msun-m500_gas_Msun_err) / (m500_tot_Msun+m500_tot_Msun_err)
    #frac_500_gas_err = (frac_500_gas_errUp + frac_500_gas_errDown) / 2.0
    frac_500_gas_err = (m500_gas_Msun+m500_gas_Msun_err) / m500_tot_Msun

    r = { 'name'             : np.array([d[0] for d in data]), 
          'm500_logMsun'     : np.log10(m500_tot_Msun),
          'm500_logMsun_err' : np.log10(m500_tot_Msun_err),
          'fGas500'          : frac_500_gas,
          'fGas500Err'       : frac_500_gas_err,
          'label'            : 'Losivari+ (2015) z<0.04' }

    return r

def zafar2013():
    """ Load observational data points (HI absorption, f_HI(N) z=[1.51,5.0]) from Zafar+ (2013), sub-DLAs. """
    # Table 4
    r = {'log_NHI'         : np.array( [19.15, 19.45, 19.75, 20.00, 20.20] ),
         'log_NHI_xerr'    : 0.15, # symmetric, in both directions
         'log_fHI'         : np.array( [-20.43, -20.75, -21.15, -21.30, -21.61] ),
         'log_fHI_errUp'   : np.array( [0.09, 0.09, 0.09, 0.09, 0.11] ),
         'log_fHI_errDown' : -1*np.array( [-0.10, -0.10, -0.13, -0.12, -0.16] ),
         'label'           : 'Zafar+ (2013) combined sub-DLA sample (1.5 < z < 5)' }
    return r

def noterdaeme2012():
    """ Load observational data points (HI absorption, f_HI(N)) from Noterdaeme+ (2012), DLA range. """
    path = dataBasePath + 'noterdaeme/noterdaeme2012.txt'

    # columns: log_N(Hi)_low log_N(Hi)_high log_f(NHI,chi) log_f(NHI,chi)_corr sigma(log_f(NHI,chi))
    data = np.loadtxt(path)

    r = {'log_NHI'      : np.array( [np.mean([data[i,0],data[i,1]]) for i in range(len(data))]),
         'log_NHI_xerr' : data[0,1]-data[0,0],
         'log_fHI'      : data[:,3],
         'log_fHI_err'  : data[:,4], # corrected
         'label'        : 'Noterdaeme+ (2012) SDSS-DR9 <z> = 2.5'}

    return r

def noterdaeme2009():
    """ Load observational data points (HI absorption, f_HI(N)) from Noterdaeme+ (2009), DLA range. """
    # Gamma-function fit, Table 1 & Fig 11, http://adsabs.harvard.edu/abs/2009A%26A...505.1087N
    k_g = -22.75
    N_g = 21.26
    alpha_g = -1.27

    log_NHI = np.linspace(20.0, 22.0, 50)
    log_fHI = k_g * (log_NHI/N_g)**alpha_g * np.exp(-log_NHI/N_g)

    # we have the raw points instead from Fig 11 (priv comm)
    path = dataBasePath + 'noterdaeme/fhix.dat'
    data = np.loadtxt(path)

    r = {'log_NHI'      : np.array( [np.mean([data[i,0],-data[i,1]]) for i in range(len(data))]),
         'log_NHI_xerr' : 0.5*(data[0,1]+data[0,0]),
         'log_fHI'      : data[:,2],
         'log_fHI_err'  : data[:,3],
         'label'        : 'Noterdaeme+ (2009) SDSS-DR7 (2.1 < z < 5.2)'}

    return r

def kim2013cddf():
    """ Load observational data points (HI absorption, f_HI(N)) from Kim+ (2013), Lya forest range. """
    # Table A.3 http://adsabs.harvard.edu/abs/2013A%26A...552A..77K
    path = dataBasePath + 'kim/kim2013_A3.txt'

    # columns: log_NH [f +f -f]   [f +f -f]   [f +f -f]
    #                 [z=1.9-3.2] [z=1.9-2.4] [z=2.4-3.2]
    data = np.loadtxt(path)

    r = {'log_NHI'         : data[:,0],
         'log_fHI'         : data[:,7],
         'log_fHI_errUp'   : data[:,8],
         'log_fHI_errDown' : data[:,9],
         'label'           : 'Kim+ (2013) VLT/UVES (2.4 < z < 3.2)'}

    return r

def prochaska10cddf():
    """ Load observational data (f_HI(N) cddf) from Prochaska+ (2010), LLS range. """
    # Table 5 and Fig 10: five powerlaws of the form f_LLS(N_HI) = k * N_HI^beta
    # take envelope, but k_LLS by forcing f to agree at f19
    nPts = 50

    beta_lls  = np.array([-0.8, -0.9, -1.3, -0.1, -0.8])
    log_k_lls = np.array([-5.05, -3.15, 4.45, -17.95, -4.65])

    r = {'log_NHI'       : np.linspace(17.5, 19.0, nPts),
         'log_fHI'       : np.zeros(nPts),
         'log_fHI_upper' : np.zeros(nPts),
         'log_fHI_lower' : np.zeros(nPts),
         'label'         : 'Prochaska+ (2010) SDSS-DR7 (3.3 < z < 4.4)'}

    for i in range(nPts):
        N_HI = 10.0**r['log_NHI'][i]
        fHI_variations = 10.0**log_k_lls * N_HI**beta_lls

        r['log_fHI'][i] = np.log10( fHI_variations[0] )
        r['log_fHI_upper'][i] = np.log10( fHI_variations ).max()
        r['log_fHI_lower'][i] = np.log10( fHI_variations ).min()

    return r

def rafelski2012(sP, redshiftRange=[2.5, 3.5]):
    """ Load observational data (DLA metallicities) from Rafelski+ (2012). z>1.5. """
    # Tables 2 and 3 from online data
    # http://vizier.cfa.harvard.edu/viz-bin/VizieR-3?-source=J/ApJ/755/89/table3
    path1 = dataBasePath + 'rafelski/raf2012_table2.txt'
    path2 = dataBasePath + 'rafelski/raf2012_table3.txt'

    data1 = np.genfromtxt(path1, dtype=None)
    data2 = np.genfromtxt(path2, dtype=None)

    redshifts     = np.array([d[1] for d in data1] + [d[1] for d in data2])
    metallicities = np.array([d[11] for d in data1] + [d[11] for d in data2]) # 8=[Fe/H], 11=[M/H]

    #metalBins = [-2.8, -2.4, -2.0, -1.6, -1.2, -0.8, -0.4]
    #binSize = 0.4
    metalBins = [-2.7, -2.4, -2.1, -1.8, -1.5, -1.2, -0.9, -0.6, -0.3]
    binSize = 0.3

    ww = np.where((redshifts >= redshiftRange[0]) & (redshifts < redshiftRange[1]))

    metalX = np.zeros(len(metalBins)-1)
    metalY = np.zeros(len(metalBins)-1)
    metalS = np.zeros(len(metalBins)-1)
    count = 0

    for i in range(len(metalBins)-1):
        metalX[i] = 0.5*(metalBins[i]+metalBins[i+1])

        # which DLAs are in this redshift+metallicity bin
        wwZ = (metallicities[ww] >= metalBins[i]) & (metallicities[ww] < metalBins[i+1])

        # number/total, and their stddev
        metalY[i] = np.count_nonzero( wwZ )
        metalS[i] = np.sqrt(metalY[i]) # rough Poisson errors
        count += metalY[i]

    metalY /= ( len(ww[0]) * binSize )
    metalS /= ( len(ww[0]) * binSize )

    # metalY is [M/H] = total metal abundance = log10(N_M/N_H)_DLA -log10(N_M/N_H)_solar
    Z_solar_raf = 0.0182 # = MZ/MH = 0.0134/0.7381 (Asplund 2009 Table 1 http://arxiv.org/abs/0909.0948)

    r = {'log_Z'     : np.log10( 10.0**metalX * (Z_solar_raf/sP.units.Z_solar) ),
         'log_Z_err' : 0.5*binSize,
         'pdf'       : metalY,
         'pdf_err'   : metalS,
         'label'     : 'Rafelski+ (2012) DLA [M/H] 2.5 < z < 3.5'}

    return r

def zahid2012(pp04=False, redshift=0):
    """ Load observational data (gas MZR) from Zahid+ (2012). """
    if redshift not in [0,1,2]:
        raise Exception('Bad redshift')

    path = dataBasePath + 'zahid/z2012_table1_z%d.txt' % redshift
    # columns: log_Mstar_solar, log_OHn12, log_OHn12_err, E_BV, E_BV_err, SFR, SFR_err
    data = np.loadtxt(path)

    # Zahid+ (2014) uses KK04: to convert this to PP04_N2, use z_PP04 = a+b*x+c*x^2+d*x^3
    #   where x=KK04 metallicity following Kewley+ (2008)
    a = 916.7484
    b = -309.5448
    c = 35.051680
    d = -1.3188

    if pp04:
        logOHp12 = a + b*data[:,1] + c*data[:,1]**2 + d*data[:,1]**3.0
        logOHp12_err = a + b*(data[:,1]+data[:,2]) + c*(data[:,1]+data[:,2])**2 + d*(data[:,1]+data[:,2])**3.0
        logOHp12_err -= logOHp12
        label = 'Zahid+ (2012) PP04 SDSS-DR7 z~%d' % redshift
    else:
        logOHp12 = data[:,1]
        logOHp12_err = data[:,2]
        label = 'Zahid+ (2012) KK04 SDSS-DR7 z~%d' % redshift

    # metallicity traditionally defined as a number density of oxygen relative to hydrogen, and is 
    # given as 12 + log(O/H). To convert to the mass density of oxygen relative to hydrogen (equal to 
    # total oxygen mass divided by total hydrogen mass):
    # log(Z_gas) = 12 + log(O/H) - 12 - log( (M_O / M_H)/(X*M_H + Y*M_He) )
    #            = log(O/H) - log( 16.0*1.0079 / (0.75*1.0079 + 0.25*4.0) )

    OH_ratio = 10.0**(logOHp12 - 12.0) / 10.0**(logOHp12_solar - 12.0)

    r = { 'stellarMass'  : data[:,0], 
          'Zgas'         : np.log10(OH_ratio),#logOHp12,
          'Zgas_err'     : logOHp12_err,
          'label'        : label }

    return r

def zahid2014(pp04=False, redshift=0.08):
    """ Load observational data (gas MZR fit) from Zahid+ (2014). """
    # Eqn 5 with best-fit parameter values from Table 2 SDSS "BEST FIT"
    nPts = 50
    xx = np.linspace(9.0, 11.0, nPts)

    if redshift == 0.08:
        Z_0     = 9.102 # +/- 0.002
        log_M_0 = 9.219 # +/- 0.004
        gamma   = 0.513 # +/- 0.009
    elif redshift == 0.29:
        Z_0     = 9.102 # +/- 0.004
        log_M_0 = 9.52  # +/- 0.02
        gamma   = 0.52  # +/- 0.02
    elif redshift == 0.78:
        Z_0     = 9.10  # +/- 0.01
        log_M_0 = 9.80  # +/- 0.05
        gamma   = 0.52  # +/- 0.04
    elif redshift == 1.55:
        Z_0     = 9.08  # +/- 0.07
        log_M_0 = 10.06 # +/- 0.2
        gamma   = 0.61  # +/- 0.15
    else:
        raise Exception('Bad redshift')

    logOHp12 = Z_0 + np.log10( 1.0 - np.exp( -1.0*(10.0**xx/10.0**log_M_0)**gamma ) )

    # Zahid+ (2014) uses KK04: to convert this to PP04_N2, use z_PP04 = a+b*x+c*x^2+d*x^3
    #   where x=KK04 metallicity following Kewley+ (2008)
    a = 916.7484
    b = -309.5448
    c = 35.051680
    d = -1.3188

    if pp04:
        logOHp12 = a + b*logOHp12 + c*logOHp12**2 + d*logOHp12**3.0
        label = 'Zahid+ (2014) PP04 SDSS-fit z~'+str(redshift)
    else:
        label = 'Zahid+ (2014) KK04 SDSS-fit z~'+str(redshift)

    OH_ratio = 10.0**(logOHp12 - 12.0) / 10.0**(logOHp12_solar - 12.0)



    r = { 'stellarMass'  : xx, 
          'Zgas'         : np.log10(OH_ratio),
          'label'        : label }

    return r

def tremonti2004():
    """ Load observational data (gas MZR) from Tremonti+ (2004). """
    path = dataBasePath + 'tremonti/t2004_table3.txt'
    # columns: log_Mstar_solar, p2.5, p16, p50, p84, p97.5
    #          where p values are percentiles for 12+log(O/H) in bins of 0.1 dex, so p50 is the median
    data = np.loadtxt(path)

    OH_ratio      = 10.0**(data[:,3] - 12.0) / 10.0**(logOHp12_solar - 12.0)
    OH_ratio_up   = 10.0**(data[:,4] - 12.0) / 10.0**(logOHp12_solar - 12.0)
    OH_ratio_down = 10.0**(data[:,2] - 12.0) / 10.0**(logOHp12_solar - 12.0)

    r = { 'stellarMass'  : data[:,0], 
          'Zgas'         : np.log10(OH_ratio),
          'Zgas_Up'      : np.log10(OH_ratio_up),
          'Zgas_Down'    : np.log10(OH_ratio_down),
          'label'        : 'Tremonti+ (2004) CL01 SDSS-EDR z~0.1' }

    return r

def guo2016(O3O2=False):
    """ Load observational data (gas MZR dwarfs z~0.6) from Guo+ (2016). """
    nPts = 50
    xx = np.linspace(8.0, 10.5, nPts)

    # Eqn 1 with best-fit parameters (LINEAR) from Table 2
    if O3O2:
        # [OIII]/[OII]
        c_0     = 5.90
        c_0_err = 0.18
        c_1     = 0.30
        c_1_err = 0.02
        c_2     = 0.0
        label   = 'Guo+ (2016) DEEP3+TKRS [OIII]/[OII] 0.6<z<0.8'
    else:
        # [OIII]/Hbeta (upper+lower z)
        c_0     = 5.83
        c_0_err = 0.19
        c_1     = 0.30
        c_1_err = 0.02
        c_2     = 0.0
        label   = 'Guo+ (2016) DEEP3+TKRS [OIII]/H$\\beta$ 0.6<z<0.8'

    logOHp12 = c_0 + c_1 * xx + c_2 * xx**2.0
    OH_ratio = 10.0**(logOHp12 - 12.0) / 10.0**(logOHp12_solar - 12.0)

    r = { 'stellarMass'  : xx, 
          'Zgas'         : np.log10(OH_ratio),
          'label'        : label }

    return r

def thomChen2008():
    """ OVI CDDF (0.12 < z < 0.5) from Thom & Chen (2008), extracted from Figure 5 panel (a). """
    r = {'log_NOVI'          : np.array([14.311, 13.731, 13.483]),
         'log_NOVI_errRight' : np.array([0.60, 0.13, 0.13]),
         'log_NOVI_errLeft'  : np.array([0.38, 0.10, 0.12]),
         'log_fOVI'          : np.array([-14.720, -13.146, -12.893]),
         'log_fOVI_err'      : np.array([0.188, 0.149, 0.149]),
         'label'             : 'Thom+ (2008) STIS 0.12 < z < 0.5'}

    return r

def tripp2008():
    """ OVI CDDF (z < 0.5) from Tripp+ (2008), extracted from Figure 11. """
    tot_N = 91.0
    log_NOVI = np.array([13.2, 13.4, 13.6, 13.8, 14.0, 14.2, 14.4, 14.6, 14.8])
    dNdz     = np.array([2.81, 4.96, 5.15, 5.84, 1.94, 2.28, 1.00, 0.34, 0.04])

    # estimate a rough errorbar, assume dz constant among bins
    dN = dNdz/dNdz.sum() * tot_N
    ss = np.sqrt(dN+1) + 1.0 # Eqn 10, Gehrels 1986, as in Table 4 Tripp+ (2008)

    cddf = dNdz/10.0**log_NOVI

    r = {'log_NOVI'         : log_NOVI,
         'log_NOVI_err'     : 0.1, # dex
         'log_fOVI'         : np.log10(cddf),
         'log_fOVI_err'     : np.log10( (dNdz+ss)/10.0**log_NOVI ) - np.log10(cddf),
         'label'            : 'Tripp+ (2008) STIS/FUSE z < 0.5' }

    return r

def danforth2008():
    """ OVI CDDF (z < 0.4) from Danforth+ (2008), extracted from Figure 5 top-right panel. """
    log_NOVI     = np.array([13.0,  13.2,  13.4,  13.6,  13.8,  14.0,  14.2,  14.4,  14.6,  14.8])
    dNdz         = np.array([5.575, 4.474, 4.597, 3.705, 2.695, 1.809, 1.440, 1.009, 0.406, 0.271])
    dNdz_errUp   = np.array([3.373, 2.055, 1.692, 1.280, 1.065, 0.886, 0.775, 0.665, 0.505, 0.461])
    dNdz_errDown = np.array([2.265, 1.483, 1.274, 0.985, 0.769, 0.609, 0.529, 0.437, 0.264, 0.197])
    cddf         = dNdz/10.0**log_NOVI

    r = {'log_NOVI'         : log_NOVI,
         'log_NOVI_err'     : 0.1, # dex
         'log_fOVI'         : np.log10(cddf),
         'log_fOVI_errUp'   : np.log10( (dNdz+dNdz_errUp)/10.0**log_NOVI ) - np.log10(cddf),
         'log_fOVI_errDown' : np.log10(cddf) - np.log10( (dNdz-dNdz_errDown)/10.0**log_NOVI ),
         'label'            : 'Danforth+ (2008) STIS/FUSE z < 0.4' }

    return r

def danforth2016():
    """ Load observational data (OVI CDDF low-redshift) from Danforth+ (2015), Table 5. """
    dlog_N = 0.2 # dex
    log_N  = np.array([12.9, 13.1, 13.3, 13.5, 13.7,  13.9,  14.1,  14.3,  14.5,  14.7,  14.9 ])
    N_OVI  = np.array([2,    6,    17,   40,   62,    69,    47,    24,    10,    2,     1    ])
    dz_OVI = np.array([0.24, 0.91, 4.13, 9.16, 12.65, 13.81, 14.13, 14.36, 14.43, 14.46, 14.48])

    d2N_dlogNdz   = np.array([42,  33, 21, 22, 25, 25, 17, 8.4, 3.5, 0.69, 0.35])
    d2N_dlogNdz_p = np.array([150, 78, 33, 7,  4,  3,  3,  2.1, 1.5, 0.91, 0.79])
    d2N_dlogNdz_m = np.array([37,  25, 9,  5,  3,  3,  2,  1.7, 1.1, 0.45, 0.29])

    d2N_dNdz   = d2N_dlogNdz * dlog_N / 10.0**log_N
    d2N_dNdz_p = (d2N_dlogNdz+d2N_dlogNdz_p) * dlog_N / 10.0**log_N
    d2N_dNdz_m = (d2N_dlogNdz-d2N_dlogNdz_m) * dlog_N / 10.0**log_N

    r = {'log_NOVI'         : log_N,
         'log_NOVI_err'     : 0.5 * dlog_N,
         'log_fOVI'         : np.log10(d2N_dNdz),
         'log_fOVI_errUp'   : np.log10(d2N_dNdz_p) - np.log10(d2N_dNdz),
         'log_fOVI_errDown' : np.log10(d2N_dNdz) - np.log10(d2N_dNdz_m),
         'label'            : 'Danforth+ (2016) COS 0.1<z<0.73'}

    return r

def muzahid2011():
    """ Load observational data (OVI CDDF z~2.3) from Muzahid+ (2011). """
    path = dataBasePath + 'muzahid/muzahid11_ovi.txt'
    # columns: N_OVI_systems fN fN_yerrup fN_yerrdown fN_xerrleft fN_xerrright
    data = np.loadtxt(path)

    r = {'log_NOVI'         : data[:,0],
         'log_NOVI_errLow'  : data[:,0] - data[:,4],
         'log_NOVI_errHigh' : data[:,5] - data[:,0],
         'log_fOVI'         : data[:,1],
         'log_fOVI_errUp'   : data[:,2] - data[:,1],
         'log_fOVI_errDown' : data[:,1] - data[:,3],
         'label'            : 'Muzahid+ (2011) VLT/UVES 1.9<z<3.1'}

    return r

def bekeraite16VF():
    """ Load observational data points from Bekeraite+ (2016) Fig 3 extracted. """
    path = dataBasePath + 'bekeraite/b16.txt'

    data = np.loadtxt(path,delimiter=',')

    r = {'v_circ'      : data[:,0],
         'numDens'     : 1e-3 * data[:,1],
         'numDens_err' : 1e-3 * data[:,2],
         'label'       : 'Bekeraite+ (2016) HIPASS+CALIFA z < 0.05'}

    return r

def anderson2015(sP):
    """ Load observational x-ray data from Anderson+ (2015). Table 3. """

    # note: stellar masses are from 'SDSS photometry' (so use 30pkpc for comparison)
    Mstar = np.array([10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8, 10.9, 
                      11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 11.8, 11.9])
    Mstar_width = 0.1
    Mstar += Mstar_width/2

    # [erg/s] X-ray luminosities in (0.5-2.0 keV) soft band, within r500crit
    log_Lx_tot = np.array([0.0,   39.60, 40.0, 39.94, 38.96, 39.60, 40.10, 39.96, 40.40, 40.58,
                           40.97, 41.29, 41.52, 41.80, 42.34, 42.64, 42.98, 43.39, 43.46, 43.82])
    sigma_m_Ltot = np.array([0.0,  0.97, 0.19, 0.21, 0.86, 0.97, 0.19, 0.27, 0.09, 0.07,
                             0.04, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.02, 0.03])
    sigma_b_Ltot = np.array([0.40, 0.86, 0.47, 0.28, 0.83, 0.78, 0.63, 0.46, 0.19, 0.10,
                             0.11, 0.07, 0.05, 0.06, 0.06, 0.05, 0.06, 0.09, 0.11, 0.21])

    log_Lx_tot_up = log_Lx_tot + np.sqrt(sigma_m_Ltot**2.0 + sigma_b_Ltot**2.0)
    log_Lx_tot_down = log_Lx_tot - np.sqrt(sigma_m_Ltot**2.0 + sigma_b_Ltot**2.0)

    # [0.0-2keV -> bolometric] correction using Table 2
    C_bolo = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                       1.1, 1.1, 1.2, 1.3, 1.4, 1.8, 2.1, 2.5, 2.8, 3.2])

    log_Lx_bol = np.log10(10.0**log_Lx_tot * C_bolo)
    log_Lx_bol_up = np.log10(10.0**log_Lx_tot_up * C_bolo)
    log_Lx_bol_down = np.log10(10.0**log_Lx_tot_down * C_bolo)

    r = {'stellarMass'         : Mstar,
         'stellarMass_err'     : Mstar_width/2,
         'xray_LumBol'         : log_Lx_bol,
         'xray_LumBol_errUp'   : log_Lx_bol_up - log_Lx_bol,
         'xray_LumBol_errDown' : log_Lx_bol - log_Lx_bol_down,
         'label' : 'Anderson+ (2015) 0.05<z<0.4 ROSAT stacking'}

    return r

def werk2013(onlydict=False, tumlinsonOVI=True, coveringFractions=False):
    """ Load observational COS-Halos data from Werk+ (2013). """
    if coveringFractions:
        # obs data points (werk 2013 table 6)
        werk13 = {'rad':[ [0,75],[75,160] ]}
        werk13['all'] = {'cf':[83,69], 'cf_errup':[9,9], 'cf_errdown':[9,17]}
        werk13['ssfr_lt_n11'] = {'cf':[37,46], 'cf_errup':[24,14], 'cf_errdown':[24,30]}
        werk13['ssfr_gt_n11'] = {'cf':[96,84], 'cf_errup':[4,9], 'cf_errdown':[4,9]}
        werk13['mstar_lt_105'] = {'cf':[81+2,96+2], 'cf_errup':[13,4], 'cf_errdown':[13,4]}
        werk13['mstar_gt_105'] = {'cf':[81,50], 'cf_errup':[13,12], 'cf_errdown':[13,23]}
        return werk13

    path1 = dataBasePath + 'werk/galaxies_werk13.txt'
    path2 = dataBasePath + 'werk/lines_werk13.txt'

    if tumlinsonOVI: # use OVI columns from Tumlinson+ (2011)
        path2 = dataBasePath + 'tumlinson/ovi_tumlinson11.txt'

    with open(path1,'r') as f:
        gal_lines = f.readlines()
    with open(path2,'r') as f:
        abs_lines = f.readlines()

    galaxies = OrderedDict()

    for line in gal_lines:
        if line[0] == '#': continue
        name, redshift, R, Mstar, sfr, lim, sfr_err = line.split('\t')
        is_limit = True if lim == '<' else False
        # 0.22 is multiplicative correction factor of 0.61
        # for both, see https://ned.ipac.caltech.edu/level5/March14/Madau/Madau3.html
        galaxies[name] = {'z':float(redshift),
                          'R':float(R),
                          'logM':float(Mstar) - 0.22, # salpeter -> chabrier IMF correction
                          'sfr':float(sfr) * 0.63, # salpeter -> chabrier SFR correction
                          'sfr_limit':is_limit, # upper
                          'sfr_err':float(sfr_err) * 0.63, # propagation
                          'name':name,
                          'lines':{}}

    for line in abs_lines:
        if line[0] == '#': continue
        name, el, ion, lim, logN, err, flag = line.split('\t')
        ion = el + ' ' + ion
        lim = ['=','<','>'].index(lim)

        assert name in galaxies
        if ion in galaxies[name]:
            print('skip: [%s] for [%s]' % (ion,name))
            continue

        # flag: 1 = good, 3 = minorly blended, 5 = non-detection (2sigma upper limit)
        # 9 = saturated, and 11 = blended and saturated
        galaxies[name]['lines'][ion] = {'line_limit':int(lim), # 0=exact, 1=upper, 2=lower
                                        'logN':float(logN),
                                        'err':float(err),
                                        'flag':int(flag)} 

    # pull out some flat numpy arrays
    gals = [g for _, g in galaxies.iteritems() if 'O VI' in g['lines']]

    logM = np.array( [gal['logM'] for gal in gals] )
    z = np.array( [gal['z'] for gal in gals] )
    sfr = np.array( [gal['sfr'] for gal in gals] )
    sfr_err = np.array( [gal['sfr_err'] for gal in gals] )
    sfr_limit = np.array( [gal['sfr_limit'] for gal in gals] ) # True=upper
    R = np.array( [gal['R'] for gal in gals] )

    ovi_logN = np.array( [gal['lines']['O VI']['logN'] for gal in gals] )
    ovi_err = np.array( [gal['lines']['O VI']['err'] for gal in gals] )
    ovi_limit = np.array( [gal['lines']['O VI']['line_limit'] for gal in gals] ) # 0=exact, 1=upper, 2=lower

    if onlydict:
        return gals

    return gals, logM, z, sfr, sfr_err, sfr_limit, R, ovi_logN, ovi_err, ovi_limit

def johnson2015(surveys=['IMACS','SDSS'], coveringFractions=False):
    """ Load observational data/compendium from Johnson+ (2015). Only the given surveys, i.e. 
    exclude the COS-Halos points which are also included in this table. """
    if coveringFractions:
        # obs data points (johnson 2015 figure 4 bottom row)
        j15 = {}

        j15['all'] = {}
        j15['ssfr_gt_n11'] = {'rad':[0.25,0.68,2.0,5.4], 'rad_left':[0.1,0.5,1.0,3.1],
                              'rad_right':[0.5,1.0,3.1,10.0],'cf':[1.0,0.94,0.28,0.0], 
                              'cf_down':[0.9,0.82,0.2,0.0], 'cf_up':[1.0,0.96,0.43,0.03]}
        j15['ssfr_lt_n11'] = {'rad':[0.34,0.76,1.95,4.6], 'rad_left':[0.22,0.66,1.1,3.3],
                              'rad_right':[0.45,0.88,2.9,7.0],'cf':[0.62,0.33,0.05,0.0], 
                              'cf_down':[0.44,0.19,0.03,0.0], 'cf_up':[0.75,0.62,0.14,0.1]}
        j15['ssfr_lt_n11_I'] = {'rad':[0.25,0.64,1.9,5.1], 'rad_left':[0.1,0.5,1.1,3.3],
                                'rad_right':[0.5,0.95,2.97,9.7],'cf':[1.0,0.93,0.18,0.0], 
                                'cf_down':[0.88,0.8,0.12,0.0], 'cf_up':[1.0,0.95,0.35,0.05]}                  
        j15['ssfr_lt_n11_NI'] = {'rad':[0.35,0.75,2.2,4.95], 'rad_left':[0.22,0.66,1.5,3.7],
                                'rad_right':[0.45,0.87,2.94,7.0],'cf':[0.57,0.34,0.0,0.0], 
                                'cf_down':[0.39,0.19,0.0,0.0], 'cf_up':[0.72,0.62,0.09,0.15]}  
        j15['ssfr_gt_n11_I'] = {'rad':[0.5,2.25,6.03], 'rad_left':[0.16,1.85,3.16],
                                'rad_right':[0.97,2.62,7.67],'cf':[1.0,0.67,0.0], 
                                'cf_down':[0.74,0.38,0.0], 'cf_up':[1.0,0.80,0.12]} # note: 0.82->0.80 visual
        j15['ssfr_gt_n11_NI'] = {'rad':[0.28,1.55,3.9], 'rad_left':[0.28,1.1,3.3],
                                'rad_right':[0.28,2.08,4.5],'cf':[1.0,0.12,0.0], 
                                'cf_down':[0.4,0.08,0.0], 'cf_up':[1.0,0.32,0.2]} 
        return j15

    with open(dataBasePath + 'johnson/j15_table1.txt') as f:
        lines = f.readlines()

    # count and allocate
    nGals = 0
    for line in lines:
        for survey in surveys:
            if survey in line:
                nGals += 1

    logM = np.zeros( nGals, dtype='float32' )
    z = np.zeros( nGals, dtype='float32' )
    sfr = np.zeros( nGals, dtype='float32' ) # set to ssfr -11.5 for Class=Early, -10.5 for Class=Late
    sfr_err = np.zeros( nGals, dtype='float32' ) # left at zero
    sfr_limit = np.zeros( nGals, dtype='bool' ) # all false
    R = np.zeros( nGals, dtype='float32' )
    ovi_logN = np.zeros( nGals, dtype='float32' )
    ovi_err = np.zeros( nGals, dtype='float32' )
    ovi_limit = np.zeros( nGals, dtype='int16' )

    galaxies = OrderedDict()
    count = 0

    for line in lines:
        if line[0] == '#': continue
        name,RAJ2000,DEJ2000,zgal,logMstar,Class,Env,Survey,d,d_Rh,\
          l_logNHI,logNHI,e_logNHI,logNHIu,l_logNHOVI,logNHOVI,e_logNHOVI = line.split('|')

        if Survey.strip() not in surveys:
            continue

        # construct 'quasar_galaxy' name using QSO_RAgalDECgal where 
        # RA,DEC are truncated to nearest arcsec, sexagesimal with spaces removed
        name_qso = name.strip()
        name = name_qso + "_" + RAJ2000.split(".")[0].replace(" ","") + DEJ2000.split(".")[0].replace(" ","")
        z[count] = float(zgal)
        logM[count] = float(logMstar)
        R[count] = float(d)

        if Class.strip() == 'Early':
            sfr[count] = 10.0**logM[count] * 10.0**(-11.5) # msun/yr
        elif Class.strip() == 'Late':
            sfr[count] = 10.0**logM[count] * 10.0**(-10.0)
        else:
            assert 0

        ovi_limit[count] = [' ','<','not indicated'].index(l_logNHOVI) # 0=exact, 1=upper, 2=lower
        ovi_logN[count] = float(logNHOVI) if logNHOVI.strip() != '' else np.nan
        ovi_err[count] = float(e_logNHOVI) if e_logNHOVI != '\n' else np.nan

        # consistent with werk2013() return
        galaxies[name] = {'z':z[count],
                          'R':R[count],
                          'logM':logM[count],
                          'sfr':sfr[count],
                          'sfr_limit':sfr_limit[count], # always False
                          'sfr_err':sfr_err[count], # always zero
                          'name':name,
                          'R_Rh':float(d_Rh),
                          'survey':Survey.strip(),
                          'environment':['I','NI'].index(Env.strip()), # 0=I (isolated), 1=NI (not isolated)
                          'lines':{}}
        galaxies[name]['lines']['O VI'] = {'line_limit':int(ovi_limit[count]), # 0=exact, 1=upper, 2=lower
                                           'logN':float(ovi_logN[count]),
                                           'err':float(ovi_err[count]),
                                           'flag':-1} 
        count += 1

    assert count == nGals

    # consistent with werk2013() return
    gals = [g for _, g in galaxies.iteritems()]

    return gals, logM, z, sfr, sfr_err, sfr_limit, R, ovi_logN, ovi_err, ovi_limit

def rossetti17planck():
    """ Load observational data points from Rosetti+ (2017) Table 1, Planck clusters. """
    path = dataBasePath + 'rossetti/r17_table1.txt'

    # load
    with open(path,'r') as f:
        lines = f.readlines()

    nLines = 0
    for line in lines:
        if line[0] != '#': nLines += 1

    # allocate
    name  = []
    z     = np.zeros( nLines, dtype='float32' )
    m500  = np.zeros( nLines, dtype='float32' ) # [log msun]
    c     = np.zeros( nLines, dtype='float32' ) # concentration parameter
    c_err = np.zeros( nLines, dtype='float32' )

    # parse
    i = 0
    for line in lines:
        if line[0] == '#': continue
        line = line.replace("\\\\","").split(" & ")
        name.append(line[1])
        z[i] = float(line[5])
        m500[i] = float(line[6]) # 10^14 msun
        c[i] = float(line[7])
        c_err[i] = float(line[8])
        i += 1

    r = {'name'  : name,
         'z'     : z,
         'm500'  : np.log10(m500 * 1e14), # msun -> log[msun]
         'c'     : c,
         'c_err' : c_err,
         'label' : 'Rossetti+ (2017) Planck Sample'}

    return r

def cassano13():
    """ Load observational data points from Cassano+ (2013) Tables 1 and 2, radio/x-ray/SZ clusters. """
    path1 = dataBasePath + 'cassano/C13_table1.txt'
    path2 = dataBasePath + 'cassano/C13_table2.txt'

    # load first table
    with open(path1,'r') as f:
        lines = f.readlines()

    nLines = 0
    for line in lines:
        if line[0] != '#': nLines += 1

    # allocate
    name  = []
    z     = np.zeros( nLines, dtype='float32' )
    l500  = np.zeros( nLines, dtype='float32' ) # 0.1-2.4 kev x-ray luminosity within r500 [units = linear 10^44 erg/s ????]
    y500  = np.zeros( nLines, dtype='float32' ) # Y500 in [ log Mpc^2 ]
    m500  = np.zeros( nLines, dtype='float32' ) # mass within [log msun]
    p14   = np.zeros( nLines, dtype='float32' ) # k-corrected radio halo power at 1.4 ghz [units = linear 10^24 W/Hz ????]
    l500_err = np.zeros( nLines, dtype='float32' )
    y500_err = np.zeros( nLines, dtype='float32' )
    m500_err = np.zeros( nLines, dtype='float32' )
    p14_err  = np.zeros( nLines, dtype='float32' )

    # parse
    i = 0
    for line in lines:
        if line[0] == '#': continue
        line = line.split()
        name.append(line[0])
        z[i] = float(line[7])
        l500[i] = float(line[8])
        l500_err[i] = float(line[9])
        if line[13][0] == '<': line[13] = line[13][1:] # upper limit, denoted by err == -1.0
        p14[i] = float(line[13])
        p14_err[i] = float(line[14])
        i += 1

    # load second table
    with open(path2,'r') as f:
        lines = f.readlines()

    for line in lines:
        if line[0] == '#': continue
        line = line.split()
        index = name.index(line[0])
        y500[index] = float(line[2])
        y500_err[index] = float(line[3])
        m500[index] = float(line[4])
        m500_err[index] = float(line[5])

    # return only those with known m500
    w = np.where(m500 > 0.0)

    w_detection = np.where(p14_err > 0.0)
    p14_logerr = np.zeros( p14_err.size, dtype='float32' ) - 1.0
    p14_logerr[w_detection] = np.log10((p14[w_detection]+p14_err[w_detection]) * 1e24) - np.log10(p14[w_detection] * 1e24)

    r = {'name'     : np.array(name)[w],
         'z'        : z[w],
         'm500'     : m500[w], # log[msun]
         'm500_err' : m500_err[w], # log[msun]
         'y500'     : y500[w], # log[Mpc^2]
         'y500_err' : y500_err[w], # log[Mpc^2]
         'l500'     : np.log10(l500[w] * 1e44), # ? maybe 10^44 erg/s, return as [log erg/s]
         'l500_err' : np.log10(l500_err[w] * 1e44), # ? maybe 10^44 erg/s
         'p14'      : np.log10(p14[w] * 1e24), # ? maybe 10^24 W/Hz, return as [log W/Hz]
         'p14_err'  : p14_logerr[w], # ? maybe 10^24 W/Hz, return as [log W/Hz]
         'label' : 'Cassano+ (2013) EGRHS X-ray Flux Limited'}

    return r

def loadSDSSData(loadFields=None, redshiftBounds=[0.0,0.1], petro=False):
    """ Load some CSV->HDF5 files dumped from the SkyServer. """
    #SELECT
    #   p.objid,
    #   p.u,p.g,p.r,p.i,p.z,
    #   p.extinction_u,p.extinction_g,p.extinction_r,p.extinction_i,p.extinction_z,
    #   s.z as redshift,
    #   gran.cModelAbsMag_u,
    #   gran.cModelAbsMag_g,
    #   gran.cModelAbsMag_r,
    #   gran.cModelAbsMag_i,
    #   gran.cModelAbsMag_z,
    #   wisc1.mstellar_median as logMass_wisc1,
    #   wisc2.mstellar_median as logMass_wisc2,
    #   gran1.logMass as logMass_gran1,
    #   gran2.logMass as logMass_gran2,
    #   port1.logMass as logMass_port1,
    #   port2.logMass as logMass_port2
    #FROM PhotoObj AS p
    #   JOIN SpecObj AS s ON s.specobjid = p.specobjid
    #   JOIN stellarMassPCAWiscBC03 AS wisc1 ON wisc1.specobjid = p.specobjid
    #   JOIN stellarMassPCAWiscM11 AS wisc2 ON wisc2.specobjid = p.specobjid
    #   JOIN stellarMassFSPSGranWideDust AS gran1 ON gran1.specobjid = p.specobjid
    #   JOIN stellarMassFSPSGranEarlyDust AS gran2 ON gran2.specobjid = p.specobjid
    #   JOIN stellarMassStarFormingPort AS port1 ON port1.specobjid = p.specobjid
    #   JOIN stellarMassPassivePort AS port2 ON port2.specobjid = p.specobjid
    #WHERE
    #   s.z BETWEEN 0.0 and 0.1 # and so on
   
    # for petrosian magnitudes:
    #SELECT
    #   p.objid,
    #   p.petroMag_u,p.petroMag_g,p.petroMag_r,p.petroMag_i,p.petroMag_z,
    #   s.z as redshift,
    #   gran1.logMass as logMass_gran1,
    #   gran2.logMass as logMass_gran2
    #FROM PhotoObj AS p
    #   JOIN SpecObj AS s ON s.specobjid = p.specobjid
    #   JOIN stellarMassFSPSGranWideDust AS gran1 ON gran1.specobjid = p.specobjid
    #   JOIN stellarMassFSPSGranEarlyDust AS gran2 ON gran2.specobjid = p.specobjid
    #WHERE
    #   s.z BETWEEN 0.0 and 0.1

    assert redshiftBounds == [0.0,0.1]
    path_csv = expanduser("~") + '/obs/sdss_z0.0-0.1'
    nFloatFields = 22

    path = path_csv
    if petro: path += '_petro'

    r = {}

    # load HDF5
    if isfile(path+'.hdf5'):
        with h5py.File(path+'.hdf5','r') as f:
            if loadFields is None:
                loadFields = f.keys()
            for key in loadFields:
                r[key] = f[key][()]
        return r

    # convert CSV to HDF5 (first column is int64, all others are float32)
    data = np.genfromtxt(path_csv+'.csv',comments='#',delimiter=',',skip_header=1,names=True,
              dtype="i8,"+",".join(["f4" for _ in range(nFloatFields)]))

    # petrosian instead of cModelMag?
    if petro:
        path_csv += '_petro'
        nFloatFields = 8

        data_p = np.genfromtxt(path_csv+'.csv',comments='#',delimiter=',',skip_header=1,names=True,
                  dtype="i8,"+",".join(["f4" for _ in range(nFloatFields)]))

        # replace band magnitude fields in data, leave everything else
        for band in ['u','g','r','i','z']:
            data[band] = data_p['petroMag_'+band]

    with h5py.File(path+'.hdf5','w') as f:
        for key in data.dtype.names:
            f[key] = data[key]
    print('Saved: [%s.hdf5]' % path)

    return data

def loadSDSSFits(redshiftBounds=[0.0,0.1]):
    """ Load the fit results of the SDSS fiber spectrum MCMC chains. """
    from obs.sdss import sdssSpectraFitsCatName, spectralFitQuantities
    assert redshiftBounds == [0.0,0.1]
    path1 = expanduser("~") + '/obs/%s.hdf5' % sdssSpectraFitsCatName
    path2 = expanduser("~") + '/obs/sdss_z0.0-0.1.hdf5'
    assert isfile(path1) and isfile(path2)

    r = {}

    # load HDF5
    with h5py.File(path1,'r') as f:
        for key in f.keys():
            r[key] = f[key][()]
            for a in f[key].attrs:
                r[a] = f[key].attrs[a]

    # load the corresponding stellar masses
    with h5py.File(path2,'r') as f:
        r['objid2'] = f['objid'][()]
        r['logMass'] = f['logMass_gran1'][()]
        #r['redshift'] = f['redshift'][()]

    # make medians
    assert np.array_equal(r['objid'],r['objid2'])

    binSize = 0.2 # log stellar mass
    percentiles = [10,16,50,84,90,40,60]

    for ind, quantName in enumerate(spectralFitQuantities):
        vals = np.squeeze(r['sdss_mcmc_fits_z0.0-0.1'][:,ind,1]) # 1 = median
        if quantName in ['mass']: vals = np.log10(vals)
        w = np.where( np.isfinite(vals) & (r['logMass'] > 6.0) )
        xm, ym, sm, pm = running_median(r['logMass'][w],vals[w],binSize=binSize,percs=percentiles)
        r[quantName] = {'xm':xm,'ym':ym,'sm':sm,'pm':pm}

    r['label'] = 'SDSS z<0.1'

    return r

def sfrTxt(sP):
    """ Load and parse sfr.txt. """
    nPts = 2000

    # cached? in sP object or on disk?
    if 'sfrd' in sP.data:
        return sP.data['sfrd']

    saveFilenames = sorted(glob.glob(sP.derivPath + 'sfrtxt_*.hdf5'))

    if len(saveFilenames):
        print(' Loaded: [%s]' % saveFilenames[-1].split(sP.derivPath)[1])
        r = {}
        with h5py.File(saveFilenames[-1],'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # Illustris txt-files directories and Illustris-1 curie/supermuc split
    path = sP.simPath + 'sfr.txt'
    if not isfile(path):
        path = sP.simPath + 'txt-files/sfr.txt'
    if not isfile(path):
        path = sP.derivPath + 'sfr.txt'
    if not isfile(path):
        raise Exception('Cannot find sfr.txt file.')

    # columns: All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, cum_mass_stars
    data = np.loadtxt(path)

    r = { 'scaleFac'          : evenlySample(data[:,0],nPts,logSpace=True),
          'totalSm'           : evenlySample(data[:,1],nPts,logSpace=True),
          'totSfrRate'        : evenlySample(data[:,2],nPts,logSpace=True),
          'sfrMsunPerYr'      : evenlySample(data[:,3],nPts,logSpace=True),
          'totalSumMassStars' : evenlySample(data[:,4],nPts,logSpace=True),
          'cumMassStars'      : evenlySample(data[:,5],nPts,logSpace=True) }

    r['redshift'] = 1.0/r['scaleFac'] - 1.0
    r['sfrd']     = r['totSfrRate'] / sP.boxSizeCubicComovingMpc # a constant cMpc^3 (equals pMpc^3 at z=0)

    # save
    saveFilename = sP.derivPath + 'sfrtxt_%.2f.hdf5' % r['scaleFac'].max()
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print(' Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    sP.data['sfrd'] = r # attach to sP as cache

    return r
