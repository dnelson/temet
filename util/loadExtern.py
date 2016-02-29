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

    haloMassBins = np.linspace(10.0,15.0,26) # 0.2 spacing #[11.0,12.0,3.0,14.0,15.0]
    #basePath  = '/n/home07/dnelson/obs/behroozi/release-sfh_z0_z8_052913/sfr/' # from website
    basePath  = '/n/home07/dnelson/obs/behroozi/analysis/' # private communication
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

def baldry2012SizeMass():
    """ Load observational data points from Baldry+ (2012). """
    path = '/n/home07/dnelson/obs/baldry/size-mass.txt'

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
    path = '/n/home07/dnelson/obs/baldry/gsmf-BGD08.txt'

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
    path = '/n/home07/dnelson/obs/baldry/gsmf-B12.txt'

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
    paths = ['/n/home07/dnelson/obs/bernardi/MsF_' + m + '.dat' for m in models]

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

def gallazzi2005(sP):
    """ Load observational data points (M-Z and ages) from Gallazzi+ (2005). """
    path = '/n/home07/dnelson/obs/gallazzi/table2.txt'

    # columns: log(Mstar/Msun), log(Z/Zun) [P50, P16, P84], log(tr/yr) [P50, P16, P84]
    # rescale metallicities from old Z_solar=0.02 to present GS10 (0.0127) value
    data = np.loadtxt(path)

    r = { 'stellarMass'  : data[:,0], 
          'Zstars'       : np.log10(10.0**data[:,1] * 0.02 / sP.units.Z_solar),
          'ZstarsDown'   : np.log10(10.0**data[:,2] * 0.02 / sP.units.Z_solar),
          'ZstarsUp'     : np.log10(10.0**data[:,3] * 0.02 / sP.units.Z_solar),
          'ageStars'     : data[:,4],
          'ageStarsDown' : data[:,5],
          'ageStarsUp'   : data[:,6],
          'label'        : 'Gallazzi+ (2005) SDSS z<0.2' }

    return r

def woo2008(sP):
    """ Load observational data points (M-Z of local group dwarfs) from Woo+ (2008). """
    path = '/n/home07/dnelson/obs/woo/table1.txt'

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
    path = '/n/home07/dnelson/obs/kirby/2013_table4.txt'

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

def sfrTxt(sP):
    """ Load and parse sfr.txt. """
    nPts = 2000

    # cached?
    if 'sfrd' in sP.data:
        return sP.data['sfrd']

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
    r['sfrd']     = r['totSfrRate'] / sP.boxSizeCubicMpc

    sP.data['sfrd'] = r # attach to sP as cache

    return r
