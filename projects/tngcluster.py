"""
projects/tngcluster.py
  TNG-Cluster: introduction paper.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import leastsq
from os.path import isfile

def satelliteVelocityDistribution(sP, minMasses, sub_N=1):
    """ Calculate relative velocity between Nth most massive satellite and central halo, in units 
    of V200 of the halo. (sub_N=1 means the 1st, most massive, satellite). """
    groupFields   = ['Group_M_Crit200','Group_R_Crit200','GroupFirstSub','GroupNsubs']
    subhaloFields = ['SubhaloVel','SubhaloPos','SubhaloMass']

    gc = sP.groupCat(fieldsSubhalos=subhaloFields, fieldsHalos=groupFields)

    haloMasses = sP.units.codeMassToMsun(gc['halos']['Group_M_Crit200'])

    r = {}

    for minMass in minMasses:
        # find all fof halos above this M200_Crit minimum value
        w = np.where( haloMasses >= minMass )[0]

        if len(w) == 0:
            continue

        # compute V200 circular velocity
        V200 = sP.units.codeM200R200ToV200InKmS(gc['halos']['Group_M_Crit200'][w], 
                                                gc['halos']['Group_R_Crit200'][w])

        V200b = sP.units.codeMassToVirVel(gc['halos']['Group_M_Crit200'][w]) # max 1% different

        # subhalo velocities
        assert np.min(gc['halos']['GroupFirstSub'][w]) >= 0
        assert np.min(gc['halos']['GroupNsubs'][w]) > sub_N

        priInds = gc['halos']['GroupFirstSub'][w]
        Vpri = sP.units.subhaloCodeVelocityToKms(gc['subhalos']['SubhaloVel'][priInds])

        subInds = gc['halos']['GroupFirstSub'][w] + sub_N
        Vsub = sP.units.subhaloCodeVelocityToKms(gc['subhalos']['SubhaloVel'][subInds])

        # relative subhalo velocity
        Vrel = Vsub - Vpri

        Vrel_mag = np.sqrt( Vrel[:,0]**2.0 + Vrel[:,1]**2.0 + Vrel[:,2]**2.0 ) 

        # ratio of V_sub/V200_central for Nth (most massive) satellite
        r[minMass] = Vrel_mag/V200
        wMax = np.argmax(Vrel_mag)
        massMax = sP.units.codeMassToLogMsun(gc['subhalos']['SubhaloMass'][subInds[wMax]])

        print(sP.redshift, sP.snap, minMass, len(w), Vrel_mag.max(), r[minMass].max(), massMax)

    return r

def plotRelativeVelDists():
    """ desc """
    from util import simParams
    
    for snap in range(99,60,-1):
        sP = simParams(res=625, run='tng_dm', snap=snap)

        minMasses = [5e14, 1e15] #[1e14, 3e14, 1e15] # Msun

        vv = satelliteVelocityDistribution(sP, minMasses, sub_N=1)

        # start plot
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        #ax.set_xlim([0.0,2.5])
        #ax.set_ylim([1e-4,1e0])
        #ax.set_yscale('log')

        ax.set_xlabel('V$_{\\rm sub,rel}$ / V$_{\\rm 200}$')
        ax.set_ylabel('N(>V$_{\\rm sub,rel}$) / N$_{\\rm tot}$')

        for minMass in vv.keys():
            label = "M$_{\\rm 200}$ > " + str(minMass) + " M$_\odot$"
            plt.hist( vv[minMass], normed=True, cumulative=True, label=label )

        ax.plot([1.9,1.9],[0,10],'--',label='Bullet')

        ax.legend(loc='best')
        fig.savefig('subVelDist_' + str(sP.snap) + '.pdf')
        plt.close(fig)

def clusterEntropyCores():
    """ Plot radial profiles of cluster entropy and measure CC/NCC central entropy values. """
    from util.simParams import simParams

    # config
    sP = simParams(res=2500, run='tng', redshift=0.4)
    haloIDs = np.arange(250)#[4,5]

    nBins = 40
    radMinMax = [0.3,3.0] # log(pKpc)

    nBins_lin = 60
    radMinMax_lin = [0.5, 1000] # pKpc

    # helper functions
    def plaw(x, params):
        """ Powerlaw of Cavagnolo+ (2009) Eqn. 5. """
        (alpha, K100) = params
        y = K100 * (x/100.0)**alpha
        return y
    def plaw_error(params, x, y):
        y_fit = plaw(x, params)
        return y_fit - y

    def plaw_cored(x, params):
        """ Powerlaw with core of Cavagnolo+ (2009) Eqn. 4. """
        (alpha, K100, K0) = params
        y = K0 + K100 * (x/100.0)**alpha
        return y
    def plaw_cored_error(params, x, y):
        y_fit = plaw_cored(x, params)
        return y_fit - y

    # check existence of temporary save?
    saveFilename = 'save2_entr_%s_%d_halos=%d.hdf5' % (sP.simName,sP.snap,len(haloIDs))

    if isfile(saveFilename):
        print('Loading: [%s]' % saveFilename)
        with h5py.File(saveFilename,'r') as f:
            cenVals = f['cenVals'][()]
            cenVals2 = f['cenVals2'][()]
            cenVals3 = f['cenVals3'][()]
            K0s = f['K0s'][()]
            K0s_lin = f['K0s_lin'][()]
            radProfiles = f['radProfiles'][()]
            radProfiles_lin = f['radProfiles_lin'][()]
            rad = f['rad'][()]
            rad_lin = f['rad_lin'][()]
    else:
        # allocate
        cenVals = np.zeros( len(haloIDs), dtype='float32' )
        cenVals2 = np.zeros( len(haloIDs), dtype='float32' )
        cenVals3 = np.zeros( len(haloIDs), dtype='float32' )
        K0s = np.zeros( len(haloIDs), dtype='float32' )
        K0s_lin = np.zeros( len(haloIDs), dtype='float32' )
        radProfiles = np.zeros( (nBins,len(haloIDs)), dtype='float32' )
        radProfiles_lin = np.zeros( (nBins_lin,len(haloIDs)), dtype='float32' )

        # loop over halo selection
        for i, haloID in enumerate(haloIDs):
            # load
            halo = sP.groupCatSingle(haloID=haloID)
            pos = sP.snapshotSubset('gas', 'pos', haloID=haloID)

            rr = sP.periodicDists(halo['GroupPos'], pos)
            rr = sP.units.codeLengthToKpc(rr)
            rr = np.log10(rr)

            # load entropy, note: xray version: K(r) = k * T_x(r) * n_e(r)^{-2/3}
            yvals = sP.snapshotSubset('gas', 'entr', haloID=haloID)
            yvals = 10.0**yvals / sP.units.boltzmann_keV # [K cm^2] -> [keV cm^2]

            # exclude sfr>0 and log(T)<5.8 gas
            sfr = sP.snapshotSubset('gas', 'sfr', haloID=haloID)
            temp = sP.snapshotSubset('gas', 'temp', haloID=haloID)

            w = np.where( (sfr == 0.0) & (temp >= 5.8) )
            rr = rr[w]
            yvals = yvals[w]

            # plot radial profile
            yy_mean = np.zeros( nBins, dtype='float32' ) + np.nan
            yy_med  = np.zeros( nBins, dtype='float32' ) + np.nan
            yy_mean_lin = np.zeros( nBins_lin, dtype='float32' ) + np.nan
            yy_med_lin  = np.zeros( nBins_lin, dtype='float32' ) + np.nan
            rad     = np.zeros( nBins, dtype='float32' )
            rad_lin = np.zeros( nBins_lin, dtype='float32' )

            binSize = (radMinMax[1]-radMinMax[0])/nBins
            binSize_lin = (radMinMax_lin[1]-radMinMax_lin[0])/nBins_lin

            for j in range(nBins):
                # log
                binStart = radMinMax[0] + j*binSize
                binEnd   = radMinMax[0] + (j+1)*binSize
                if j == 0: binStart = -0.5 # ~0.5 pKpc

                ww = np.where((rr >= binStart) & (rr < binEnd))
                rad[j] = 10.0**( (binStart+binEnd)/2.0 )

                if len(ww[0]) > 0:
                    yy_mean[j] = np.mean(yvals[ww])
                    yy_med[j]  = np.median(yvals[ww])

            rr_lin = 10.0**rr
            for j in range(nBins_lin):
                # linear
                binStart = radMinMax_lin[0] + j*binSize_lin
                binEnd   = radMinMax_lin[0] + (j+1)*binSize_lin
                if j == 0: binStart = 0.0 # 0.0 pKpc

                ww = np.where((rr_lin >= binStart) & (rr_lin < binEnd))
                rad_lin[j] = (binStart+binEnd)/2.0

                if len(ww[0]) > 0:
                    yy_mean_lin[j] = np.mean(yvals[ww])
                    yy_med_lin[j]  = np.median(yvals[ww])

            # fit profile with a powerlaw / calculate a central value
            r200 = sP.units.codeLengthToKpc(halo['Group_R_Crit200'])
            print('[%d] Halo r200: %.1f [pKpc]' % (haloID,r200))

            # central value
            w = np.where(rad <= 0.05 * r200)
            cenVal_mean = np.nanmean( yy_mean[w] )
            cenVal_med  = np.nanmean( yy_med[w] )

            w = np.where(rad <= 30.0)
            cenVal_mean2 = np.nanmean( yy_mean[w] )

            # central value without fit
            w = np.where(rr_lin <= 20.0)
            cenVal_mean3 = np.nanmean( yvals[w] )

            print(' Central value: %.2f (%.2f) %.2f %.2f' % (cenVal_mean,cenVal_med,cenVal_mean2,cenVal_mean3))

            # data points for fit
            w = np.where( (rad <= 0.5 * r200) & np.isfinite(yy_mean) )
            x_data = rad[w]
            y_data = yy_mean[w]

            # fit powerlaw and cored powerlaws
            params_best, _, _, _, _ = leastsq(plaw_error, [1.0, 100.0], 
                                              args=(x_data,y_data), full_output=True)
            params_best_cored, _, _, _, _ = leastsq(plaw_cored_error, [1.0, 100.0, 100.0], 
                                                    args=(x_data,y_data), full_output=True)

            print(' Best fit powerlaw:  K100 = %.1f  alpha = %.2f' % (params_best[1],params_best[0]))
            print(' Best fit coredplaw: K100 = %.1f  alpha = %.2f  K0 = %.1f' % \
                (params_best_cored[1],params_best_cored[0],params_best_cored[2]))

            # fit cored to linear
            w = np.where( (rad_lin <= 0.5 * r200) & np.isfinite(yy_mean_lin) )
            x_data = rad_lin[w]
            y_data = yy_mean_lin[w]

            params_best_cored_lin, _, _, _, _ = leastsq(plaw_cored_error, [1.0, 100.0, 100.0], 
                                                    args=(x_data,y_data), full_output=True)
            print(' Best fit cored_lin: K100 = %.1f  alpha = %.2f  K0 = %.1f' % \
                (params_best_cored_lin[1],params_best_cored_lin[0],params_best_cored_lin[2]))

            # store values
            cenVals[i] = cenVal_mean
            cenVals2[i] = cenVal_mean2
            cenVals3[i] = cenVal_mean3
            K0s[i] = params_best_cored[2]
            K0s_lin[i] = params_best_cored_lin[2]
            radProfiles[:,i] = yy_mean
            radProfiles_lin[:,i] = yy_mean_lin

        # save data
        with h5py.File(saveFilename,'w') as f:
            f['cenVals'] = cenVals
            f['cenVals2'] = cenVals2
            f['cenVals3'] = cenVals3
            f['K0s'] = K0s
            f['K0s_lin'] = K0s_lin
            f['radProfiles'] = radProfiles
            f['radProfiles_lin'] = radProfiles_lin
            f['rad'] = rad
            f['rad_lin'] = rad_lin
        print('Saved [%s]' % saveFilename)

    # start pdf
    pdf = PdfPages('entr2_%s_%d_halos=%d.pdf' % (sP.simName,sP.snap,len(haloIDs)))

    # Voit+ (2002) 'pure cooling model' approximated
    voit_x = [2.8, 1800] # kpc
    voit_y = [1e0, 3e3] # [keV cm^2]

    # Cavagnolo+ (2009) K0 histogram
    yy_cavagnolo = [1,2,2,3,0,7,18,20,25,19,9,10,21,23,27,21,16,7,2]
    xx_cavagnolo = 0.0 + 0.15 * np.arange(len(yy_cavagnolo))

    # (A) start histogram plot
    if 1:
        K0_minmax = np.log10( [0.4, 700] )
        K0_nBins = 20

        # (A1) fractional histograms
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)

        ax.set_xlabel('Fraction of Clusters')
        ax.set_xlabel('Central Entropy [ keV cm$^2$ ]')
        ax.set_xlim(10.0**K0_minmax)
        ax.set_xscale('log')

        # plot histogram of cenEntr (1)
        yy, xx = np.histogram(np.log10(cenVals), bins=K0_nBins, range=K0_minmax)
        yy = np.array(yy) / float(np.sum(yy))
        xx = xx[:-1] + 0.5*(K0_minmax[1]-K0_minmax[0])/K0_nBins

        ax.plot(10.0**xx, yy, '-', drawstyle='steps-mid', lw=3.0, label='Central Entropy (1)')

        # plot histogram of cenEntr (2)
        yy, xx = np.histogram(np.log10(cenVals2), bins=K0_nBins, range=K0_minmax)
        yy = np.array(yy) / float(np.sum(yy))
        xx = xx[:-1] + 0.5*(K0_minmax[1]-K0_minmax[0])/K0_nBins

        ax.plot(10.0**xx, yy, '-', drawstyle='steps-mid', lw=3.0, label='Central Entropy (2)')

        # plot histogram of cenEntr (3)
        yy, xx = np.histogram(np.log10(cenVals3), bins=K0_nBins, range=K0_minmax)
        yy = np.array(yy) / float(np.sum(yy))
        xx = xx[:-1] + 0.5*(K0_minmax[1]-K0_minmax[0])/K0_nBins

        ax.plot(10.0**xx, yy, '-', drawstyle='steps-mid', lw=3.0, label='Central Entropy (3)')

        # plot histogram of K0
        w = np.where(K0s > 0.0)
        yy, xx = np.histogram(np.log10(K0s[w]), bins=K0_nBins, range=K0_minmax)
        yy = np.array(yy) / float(np.sum(yy))
        xx = xx[:-1] + 0.5*(K0_minmax[1]-K0_minmax[0])/K0_nBins

        ax.plot(10.0**xx, yy, '-', drawstyle='steps-mid', lw=3.0, label='Central K$_0$')

        # plot histogram of K0_lin
        w = np.where(K0s_lin > 0.0)
        yy, xx = np.histogram(np.log10(K0s_lin[w]), bins=K0_nBins, range=K0_minmax)
        yy = np.array(yy) / float(np.sum(yy))
        xx = xx[:-1] + 0.5*(K0_minmax[1]-K0_minmax[0])/K0_nBins

        ax.plot(10.0**xx, yy, '-', drawstyle='steps-mid', lw=3.0, label='Central K$_0$ linear')

        # plot K0 histogram from Cavagnolo+ (2009)
        yy = np.array(yy_cavagnolo) / float(np.sum(yy_cavagnolo))
        ax.plot(10.0**xx_cavagnolo, yy, '-', drawstyle='steps-mid', 
                color='black', lw=3.0, label='Cavagnolo+ (2009) K$_0$')

        ax.legend(loc='best')

        # finish
        pdf.savefig()
        plt.close(fig)

    # (B) all rad profiles (log)
    if 1:
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)

        ax.set_title('%s z=%.1f [log]' % (sP.simName,sP.redshift))
        ax.set_xlabel('Radius [ pKpc ]')
        ax.set_xlim([0.2, 2000])
        ax.set_xscale('log')

        ax.set_ylabel('Entropy [ keV cm$^2$ ]')
        ax.set_ylim([1, 3000])
        ax.set_yscale('log')

        ax.plot(voit_x, voit_y, color='black', label='Voit+ (2002) pure cooling model')

        for i, haloID in enumerate(haloIDs):
            yy = np.squeeze( radProfiles[:,i] )
            ax.plot(rad, yy, '-')

        # finish plot
        ax.legend(loc='best')
        pdf.savefig()
        plt.close(fig)

    # (C) all rad profiles (linear)
    if 1:
        fig = plt.figure(figsize=(14,10))
        ax = fig.add_subplot(111)

        ax.set_title('%s z=%.1f [linear]' % (sP.simName,sP.redshift))
        ax.set_xlabel('Radius [ pKpc ]')
        ax.set_xlim([0.2, 2000])
        ax.set_xscale('log')

        ax.set_ylabel('Entropy [ keV cm$^2$ ]')
        ax.set_ylim([1, 3000])
        ax.set_yscale('log')

        ax.plot(voit_x, voit_y, color='black', label='Voit+ (2002) pure cooling model')

        for i, haloID in enumerate(haloIDs):
            yy = np.squeeze( radProfiles_lin[:,i] )
            ax.plot(rad_lin, yy, '-')

        # finish plot
        ax.legend(loc='best')
        pdf.savefig()
        plt.close(fig)

    # (D) individual rad profiles
    if 1:
        for i, haloID in enumerate(haloIDs):
            #halo = sP.groupCatSingle(haloID=haloID)
            fig = plt.figure(figsize=(14,10))
            ax = fig.add_subplot(111)

            ax.set_title('%s z=%.1f haloID=%d' % (sP.simName,sP.redshift,haloID))
            ax.set_xlabel('Radius [ pKpc ]')
            ax.set_xlim([0.2, 2000])
            ax.set_xscale('log')

            ax.set_ylabel('Entropy [ keV cm$^2$ ]')
            ax.set_ylim([1, 3000])
            ax.set_yscale('log')

            ax.plot(voit_x, voit_y, color='black')

            yy = np.squeeze( radProfiles[:,i] )
            yy_lin = np.squeeze( radProfiles_lin[:,i] )

            ax.plot(rad, yy, '-', label='cenVal = %.1f cenVal2 = %.1f K0 = %.1f' % (cenVals[i], cenVals2[i], K0s[i]))
            ax.plot(rad_lin, yy_lin, ':', label='linear cenVal3 = %.1f  K0 = %.1f' % (cenVals3[i], K0s_lin[i]))

            # finish plot
            ax.legend(loc='best')
            pdf.savefig()
            plt.close(fig) 

    pdf.close()

def barnes_check1():
    """ Test plots for Barnes paper. """
    from util.loadExtern import rossetti17planck
    data = rossetti17planck()

    # z=0 TNG300-1
    sP = simParams(res=2500, run='tng', redshift=0.0)
    m500 = sP.groupCat(fieldsSubhalos=['mhalo_500_log'])
    w500 = np.where(10.0**m500 >= 2e14)

    # redshifts
    zz = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    med_sim = np.zeros( zz.size, dtype='float32' )
    med_data = np.zeros( zz.size, dtype='float32' )

    for i, z in enumerate(zz):
        print(z)
        sPloc = simParams(res=2500, run='tng', redshift=z)
        m500loc = sPloc.groupCat(fieldsSubhalos=['mhalo_500'])
        w = np.where(m500loc >= 2e14)
        med_sim[i] = np.log10( np.median( m500loc[w] ) )
        w = np.where( np.abs(data['z'] - z) <= 0.05 )
        med_data[i] = np.log10( np.median( 10.0**data['m500'][w] ) )

    # plot
    fig = plt.figure(figsize=(18,8))
    ax = fig.add_subplot(121)

    ax.set_xlabel('Halo Mass [m500, log M$_{\\rm sun}$]')
    ax.set_ylabel('N$_{\\rm halos}$')
    label = sP.simName + ' z=%.1f' % sP.redshift + ' "high-mass" (M$_{\\rm halo} > 2*10^{14} M_{\\rm sun}$)'
    ax.hist(m500[w500], bins=10, range=[14.2,15.2], alpha=0.7, label=label)
    ax.hist(data['m500'], bins=10, range=[14.2,15.2], alpha=0.7, label=data['label'])
    ax.legend()

    ax = fig.add_subplot(122)
    ax.set_xlabel('Redshift')
    ax.set_xlim([-0.05,0.6])
    ax.set_ylabel('Median Halo Mass [m500, log M$_{\\rm sun}$]')

    ax.plot(zz, med_sim, '-', marker='o', label=sP.simName)
    ax.plot(zz, med_data, '-', marker='o', label='Planck Sample')
    ax.legend(loc='best')

    fig.savefig('check_halomass.pdf')
    plt.close(fig)

def barnes_check2():
    """ Test plots for Barnes paper. """
    from util.loadExtern import rossetti17planck
    data = rossetti17planck()

    c_thresh = 0.075 # from Rossetti

    binSize = 0.2
    binMin = 14.0
    nBins = 6

    cc_frac = np.zeros(nBins, dtype='float32')
    mass = np.zeros(nBins, dtype='float32')
    cc_frac.fill(np.nan)

    for i in range(nBins):
        mass0 = binMin + binSize*i
        mass1 = mass0 + binSize
        mass[i] = (mass0+mass1)/2

        w = np.where( (data['m500'] > mass0) & (data['m500'] <= mass1) )
        c_vals = data['c'][w]
        if c_vals.size == 0: continue

        n_below = np.count_nonzero(c_vals < c_thresh)
        n_above = np.count_nonzero(c_vals >= c_thresh)
        cc_frac[i] = float(n_above) / (n_above+n_below)

    # plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Halo Mass [m500, log M$_{\\rm sun}$]')
    ax.set_ylabel('Cool Core Fraction (Planck Obs, Rossetti+ 17 definition)')
    ax.plot(mass, cc_frac, '-', marker='o', label=data['label'])

    fig.savefig('check_ccfrac_vs_mass.pdf')
    plt.close(fig)

def barnes_check3():
    """ Line fitting, Figure 9 (oct9 draft) right panel. """
    x      = np.array([1.00, 0.75,  0.5,   0.4,   0.3,   0.2,   0.1,   0.0])
    y      = np.array([0.50, 0.384, 0.0,   0.048, 0.098, 0.219, 0.069, 0.097])
    y_up   = np.array([0.75, 0.525, 0.108, 0.147, 0.181, 0.303, 0.131, 0.157])
    y_down = np.array([0.25, 0.273, 0.001, 0.032, 0.066, 0.164, 0.046, 0.068])

    obs_x      = np.array([1.0, 0.75, 0.375, 0.125])
    obs_x_l    = np.array([1.2, 0.8, 0.6, 0.2])
    obs_x_r    = np.array([0.8, 0.6, 0.2, 0.0])
    obs_y      = np.array([0.07, 0.183, 0.170, 0.225])
    obs_y_up   = np.array([0.20, 0.26, 0.26, 0.27])
    obs_y_down = np.array([0.05, 0.14, 0.12, 0.19])

    # fit
    from scipy.stats import linregress
    from scipy.optimize import leastsq
    from numpy.polynomial.polynomial import polyfit

    result1 = linregress(x[::-1],y[::-1])
    print('Slope and intercept: %g, %g (linregress)' % (result1.slope,result1.intercept))

    sigma = ((y-y_down) + (y_up-y)) / 2 # avg
    weights = 1/sigma
    result2 = polyfit(x, y, deg=1, w=weights)
    print('Slope and intercept: %g, %g (polyfit avg)' % (result2[1], result2[0]))

    sigma_max = np.max( np.vstack( ((y-y_down), (y_up-y)) ), axis=0 ) # max
    result3 = polyfit(x, y, deg=1, w=1/sigma_max)
    print('Slope and intercept: %g, %g (polyfit max)' % (result3[1], result3[0]))

    def error_function(params, x, y, y_err_up, y_err_down):
        y_fit = params[1]*x + params[0]
        error = y - y_fit
        error_pos = np.where(error >= 0) # fit below data
        error_neg = np.where(error < 0) # fit above data
        error[error_pos] /= y_err_down[error_pos]
        error[error_neg] /= y_err_up[error_neg]
        return error

    params_init = [0.0, 0.0]
    args = (x,y,y_up,y_down)
    result4, params_cov, info, errmsg, retcode = \
      leastsq(error_function, params_init, args=args, full_output=True)

    print('Slope and intercept: %g, %g' % (result4[1], result4[0]))

    params_init = [0.0, 0.0]
    args = (obs_x,obs_y,obs_y_up,obs_y_down)
    result5, params_cov, info, errmsg, retcode = \
      leastsq(error_function, params_init, args=args, full_output=True)

    print('Slope and intercept: %g, %g (data)' % (result5[1], result5[0]))

    # requires: linfit.py
    #sigmay = y_up-y
    #result6, covarmat, info6 = linfit(x, y, sigmay=sigmay, relsigma=False, return_all=True)
    #print('Slope and intercept: %g, %g (error: %g, %g)' % (result6[0], result6[1],info6.fiterr[0],info6.fiterr[1]))

    #sigmay_obs = ((obs_y-obs_y_down) + (obs_y_up-obs_y)) / 2 # avg
    #result7, covarmat, info7 = linfit(obs_x, obs_y, sigmay=sigmay_obs, relsigma=False, return_all=True)
    #print('Slope and intercept: %g, %g (error: %g, %g) (data)' % (result7[0], result7[1],info7.fiterr[0],info7.fiterr[1]))

    # plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)

    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.errorbar(x, y, yerr=[y-y_down,y_up-y], marker='o', label='TNG300-1')
    ax.errorbar(obs_x, obs_y, linestyle='None', yerr=[obs_y-obs_y_down,obs_y_up-obs_y], marker='o', color='black', label='Obs')

    xx = np.array([-0.05,1.05])
    ax.plot(xx, result1.slope*xx + result1.intercept, '-', label='linregress un-weighted (%.3f)' % result1.slope)
    ax.plot(xx, result2[1]*xx + result2[0], '-', label='polyfit weighted mean symmetric error (%.3f)' % result2[1])
    ax.plot(xx, result3[1]*xx + result3[0], '-', label='polyfit weighted max symmetric error (%.3f)' % result3[1])
    ax.plot(xx, result4[1]*xx + result4[0], '-', label='leastsq weighted asymmetric error (%.3f)' % result4[1])
    ax.plot(xx, result5[1]*xx + result5[0], '--', color='black', label='obs-fit leastsq weighted asymmetric error (%.3f)' % result5[1])
    #ax.plot(xx, result6[0]*xx + result6[1], '-', label='linfit (%.3f)' % result6[0])

    ax.legend()
    fig.savefig('check_linfit_Figure9_right.pdf')
    plt.close(fig)
