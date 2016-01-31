"""
loadExtern.py
  load external data files (observational points, etc) and Arepo txt files
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from util.helper import evenlySample
from os.path import isfile

def behrooziSMHM(sP, logHaloMass=None):
    """ Load from data files: Behroozi+ (2013) abundance matching, stellar mass / halo mass relation. """
    from scipy import interpolate

    basePath = '/n/home07/dnelson/obs/behroozi/release-sfh_z0_z8_052913/smmr/'
    fileName = 'c_smmr_z0.10_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat'

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
    from scipy.signal import savgol_filter

    haloMassBins = [11.0,12.0,13.0,14.0,15.0]
    basePath  = '/n/home07/dnelson/obs/behroozi/release-sfh_z0_z8_052913/sfr/'
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

    basePath = '/n/home07/dnelson/obs/behroozi/behroozi-2013-data-compilation/'
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

def mosterSMHM(sP):
    """ Load from data files: Moster+ (2013) abundance matching, stellar mass / halo mass relation. """
    def f2013(mass, ind=1, redshift=0.0):
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
    r['y_low']  = f2013( 10.0**r['haloMass'], ind=0 ) / (sP.omega_b/sP.omega_m)
    r['y_mid']  = f2013( 10.0**r['haloMass'], ind=1 ) / (sP.omega_b/sP.omega_m)
    r['y_high'] = f2013( 10.0**r['haloMass'], ind=2 ) / (sP.omega_b/sP.omega_m)

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
    path = '/n/home07/dnelson/obs/mcconnell/current_ascii.txt'

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

def baldrySizeMass():
    """ Load observational data points from Baldry+ (2012). """
    path = '/n/home07/dnelson/obs/baldry/size-mass.txt'

    # Columns: Ngal, stellar mass [log Msun], size 16th percentile, median size log [pc], 84th percentile
    #   first 11 rows: "Blue galaxies"
    #   next 9 rows: "Red galaxies"
    data = np.loadtxt(path)
    n = 12

    r = {}
    r['blue'] = { 'stellarMass' : data[:n,1], 
                  'sizeKpc'     : 10.0**data[:n,3] / 1000.0,
                  'errorUp'     : 10.0**data[:n,2] / 1000.0, 
                  'errorDown'   : 10.0**data[:n,4] / 1000.0,
                  'label'       : 'Baldry+ (2012) GAMA blue' }

    r['red'] = { 'stellarMass' : data[n:,1], 
                 'sizeKpc'     : 10.0**data[n:,3] / 1000.0,
                 'errorUp'     : 10.0**data[n:,2] / 1000.0, 
                 'errorDown'   : 10.0**data[n:,4] / 1000.0,
                 'label'       : 'Baldry+ (2012) GAMA red' }

    # convert errorUp, errorDown to linear deltas
    #r['errorUp']   = 10.0**(r['sfrd']+r['errorUp']) - 10.0**r['sfrd']
    #r['errorDown'] = 10.0**r['sfrd'] - 10.0**(r['sfrd']-r['errorDown'])
    #r['sfrd']      = 10.0**r['sfrd']

    return r

def sfrTxt(sP):
    """ Load and parse sfr.txt. """
    path = sP.simPath + 'sfr.txt'
    nPts = 2000

    # cached?
    if 'sfrd' in sP.data:
        return sP.data['sfrd']

    # Illustris txt-files directories and Illustris-1 curie/supermuc split
    if sP.run == 'illustris':
        path = sP.simPath + 'txt-files/sfr.txt'

    # columns: All.Time, total_sm, totsfrrate, rate_in_msunperyear, total_sum_mass_stars, cum_mass_stars
    data = np.loadtxt(path)

    r = { 'scaleFac'          : evenlySample(data[:,0],nPts,logSpace=True),
          'totalSm'           : evenlySample(data[:,1],nPts,logSpace=True),
          'totSfrRate'        : evenlySample(data[:,2],nPts,logSpace=True),
          'sfrMsunPerYr'      : evenlySample(data[:,3],nPts,logSpace=True),
          'totalSumMassStars' : evenlySample(data[:,4],nPts,logSpace=True),
          'cumMassStars'      : evenlySample(data[:,5],nPts,logSpace=True) }

    r['redshift'] = 1.0/r['scaleFac'] - 1.0
    r['sfrd']     = r['totSfrRate'] / sP.boxSizeCubicMpc

    sP.data['sfrd'] = r # attach to sP as cache

    return r
