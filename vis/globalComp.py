"""
globalComp.py
  run summary plots and comparisons to constraints
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import pdb
#import h5py

def running_median(X, Y, nBins=100, binSize=None):
    """ desc """
    if binSize is not None:
        nBins = round( (X.max()-X.min()) / binSize )

    bins = np.linspace(X.min(),X.max(), nBins)
    delta = bins[1]-bins[0]
    #idx  = np.digitize(X,bins)
    #running_median = [np.nanmedian(Y[idx==k]) for k in range(nBins)]
    #running_std    = [Y[idx==k].std() for k in range(nBins)]
    #return bins-delta/2, running_median, running_std

    running_median = []
    running_std    = []
    bin_centers    = []

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= bin) & (X < binMax))

        if len(w[0]):
            running_median.append( np.nanmedian(Y[w]) )
            running_std.append( np.std(Y[w]) )
            bin_centers.append( np.nanmedian(X[w]) )

    return bin_centers, running_median, running_std

def loadBehroozi(sP):
    """ desc """
    basePath = '/n/home07/dnelson/obs/behroozi/release-sfh_z0_z8_052913/smmr/'
    fileName = 'c_smmr_z0.10_red_all_smf_m1p1s1_bolshoi_fullcosmos_ms.dat'

    # columns: log10(halo_mass), log10(stellar_mass/halo_mass), err_up (dex), err_down (dex)
    data = np.loadtxt(basePath+fileName)

    r = { 'haloMass'  : data[:,0], 
          'smhmRatio' : data[:,1],
          'errorUp'   : data[:,2], 
          'errorDown' : data[:,3] }

    r['y_low']  = 10.0**( r['smhmRatio']-r['errorDown'] ) / (sP.omega_b/sP.omega_m)
    r['y_mid']  = 10.0**r['smhmRatio'] / (sP.omega_b/sP.omega_m)
    r['y_high'] = 10.0**( r['smhmRatio']+r['errorUp'] ) / (sP.omega_b/sP.omega_m)

    return r

def loadMoster(sP):
    """ desc """
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

def loadKravtsov(sP):
    """ desc """
    def f(x, alpha, delta, gamma):
        term1 = -1.0 * np.log10(10.0**(alpha*x) + 1.0)
        term2 = delta * (np.log10(1+np.exp(x)))**gamma / ( 1 + np.exp(10.0**(-x)) )
        return term1+term2

    def k2014(mass):
        """ Eqn. A3 and A4 of Kravtsov+ (2014) with Mvir (w/ scatter) or M200c (w/ scatter) best fit values. """
        # halo mass definition: M200c
        #log_M1 = 11.35
        #log_eps = -1.642
        #alpha = 1.779
        #delta = 4.394
        #gamma = 0.547

        # halo mass definition: Mvir
        log_M1 = 11.39
        log_eps = -1.685
        alpha = -1.740 # typo? Table 3 has positive values
        delta = 4.335
        gamma = 0.531

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

def stellarMassHaloMass():
    """ desc """
    import cosmo
    import matplotlib.pyplot as plt
    from util import simParams

    # config
    runs = ['zoom_01_ill3','zoom_02_ill3','zoom_03_ill3','zoom_04_ill3']
    res = 3
    redshift = 0.0

    subhaloID = 0
    marker = 'D'

    # config
    sPs = []
    for run in runs:
        sPs.append( simParams(res=res, run=run, redshift=redshift))

    sPsFull = []
    sPsFull.append( simParams(res=1820, run='illustris', redshift=redshift) )
    sPsFull.append( simParams(res=910, run='illustris', redshift=redshift) )
    sPsFull.append( simParams(res=455, run='illustris', redshift=redshift) )

    linestyles = ['-','--','-.',':']

    # plot setup
    xrange = [10.0, 15.0]
    yrange = [0.0, 0.25]

    runNames = []
    for sP in sPs:
        runNames.append(sP.run)

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.set_title('z=' + '{:.1g}'.format(redshift))
    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('M$_\star$ / M$_{\\rm halo}$ $(\Omega_{\\rm b} / \Omega_{\\rm m})^{-1}$')

    # abundance matching constraints
    b = loadBehroozi(sPs[0])
    m = loadMoster(sPs[0])
    k = loadKravtsov(sPs[0])

    ax.plot(b['haloMass'], b['y_mid'] , color='#ff6677', label='Behroozi+ (2013)')
    ax.fill_between(b['haloMass'], b['y_low'], b['y_high'], color='#ffddee', interpolate=True, alpha=0.2)

    ax.plot(m['haloMass'], m['y_mid'] , color='#66ff77', label='Moster+ (2013)')
    ax.fill_between(m['haloMass'], m['y_low'], m['y_high'], color='#ddffee', interpolate=True, alpha=0.2)

    ax.plot(k['haloMass'], k['y_mid'] , color='orange', label='Kravtsov+ (2014)')

    # loop over each zoom run
    for sP in sPs:
        gc = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
        gh = cosmo.load.groupCatSingle(sP, haloID=gc['SubhaloGrNr'])

        # halo mass definition?
        xx_code = gc['SubhaloMass'] #gh['Group_M_Crit200']
        xx = sP.units.codeMassToLogMsun( xx_code )

        # stellar mass definition(s)
        yy = gc['SubhaloMassType'][4] / xx_code / (sP.omega_b/sP.omega_m)
        ax.plot(xx,yy,marker,color='blue')

        yy = gc['SubhaloMassInRadType'][4] / xx_code / (sP.omega_b/sP.omega_m)
        ax.plot(xx,yy,marker,color='purple')

        yy = gc['SubhaloMassInHalfRadType'][4] / xx_code / (sP.omega_b/sP.omega_m)
        ax.plot(xx,yy,marker,color='black')

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='blue', marker=marker, linestyle=''),
              plt.Line2D( (0,1), (0,0), color='purple', marker=marker, linestyle=''),
              plt.Line2D( (0,1), (0,0), color='black', marker=marker, linestyle='')]
    lExtra = [r'$M_\star^{\rm tot}$',
              r'$M_\star^{< 2r_{1/2}}$', 
              r'$M_\star^{< r_{1/2}}$']

    legend1 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')
    plt.gca().add_artist(legend1)

    # loop over each fullbox run
    lines = []

    for i, sP in enumerate(sPsFull):
        gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub'], 
                                     fieldsSubhalos=['SubhaloMass','SubhaloMassType'])

        # centrals only
        gc['halos'] = gc['halos'].astype('int32')
        w = np.where(gc['halos'] >= 0)
        w = gc['halos'][w]
        #w = w[0:100] # 100 most massive

        # halo mass definition?
        xx_code = gc['subhalos']['SubhaloMass'][w] #gh['Group_M_Crit200']
        xx = sP.units.codeMassToLogMsun( xx_code )

        # stellar mass definition(s)
        yy = gc['subhalos']['SubhaloMassType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
        #ax.plot(xx[0:200],yy[0:200],'.',color='blue')

        # median
        xm, ym, sm = running_median(xx,yy,binSize=0.1)

        l, = ax.plot(xm, ym, linestyles[i], color='blue', label=sP.run+' '+str(sP.res))
        lines.append(l)
        #ax.errorbar(xm, ym, sm, color='blue', alpha=0.2, fmt='none')

    # second legend
    ax.legend(handles=lines, loc='upper left')

    fig.tight_layout()    
    fig.savefig('smhm.pdf')