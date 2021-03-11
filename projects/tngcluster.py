"""
TNG-Cluster: introduction paper.
(in prep)
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import leastsq
from os.path import isfile

from plot.cosmoGeneral import quantMedianVsSecondQuant
from plot.config import *

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
    """ Do we have a bullet cluster in the volume? Based on relative velocity of sub-component. """
    from util import simParams
    
    for snap in range(99,60,-1):
        sP = simParams(res=625, run='tng_dm', snap=snap)

        minMasses = [5e14, 1e15] #[1e14, 3e14, 1e15] # Msun

        vv = satelliteVelocityDistribution(sP, minMasses, sub_N=1)

        # start plot
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)

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

def vis_fullbox_virtual(sP, conf=0):
    """ Visualize the entire virtual reconstructed box. """
    from vis.box import renderBox

    axes       = [0,1] # x,y
    labelZ     = True
    labelScale = True
    labelSim   = True
    nPixels    = 2000

    # halo plotting
    plotHalos  = False

    if conf in [0,1,2,3,4]:
        pri = sP.groups('GroupPrimaryZoomTarget')
        plotHaloIDs = np.where(pri == 1)[0]

    # panel config
    if conf == 0:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[6.5,7.1]}]
    if conf == 1:
        method = 'sphMap' # is global, overlapping coarse cells
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[9.1,9.6]}]
    if conf == 2:
        method = 'sphMap_globalZoom'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]
    if conf == 3:
        method = 'sphMap' # is global
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[5.0,8.0]}]
    if conf == 4:
        method = 'sphMap' # is global
        panels = [{'partType':'gas', 'partField':'coldens_msunkpc2', 'valMinMax':[5.8,7.2]}]
        numBufferLevels = 3 # 2 or 3, free parameter

        maxGasCellMass = sP.targetGasMass
        if numBufferLevels >= 1:
            # first buffer level is 27x mass, then 8x mass for each subsequent level
            maxGasCellMass *= 27 * np.power(8,numBufferLevels-1)
            # add padding for x2 Gaussian distribution
            maxGasCellMass *= 3

        ptRestrictions = {'Masses':['lt',maxGasCellMass]}

    if conf == 5:
        sP = simParams(run='tng_dm',res=2048,redshift=0.0) # parent box
        method = 'sphMap'
        panels = [{'partType':'dm', 'partField':'coldens_msunkpc2', 'valMinMax':[7.0,8.4]}]

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = [nPixels,nPixels]
        colorbars  = True
        fontsize   = 32

        saveFilename = './boxImage_%s_%s-%s_%s_conf%d.pdf' % \
          (sP.simName,panels[0]['partType'],panels[0]['partField'],sP.snap,conf)

    renderBox(panels, plotConfig, locals(), skipExisting=False)

def vis_gallery(sP, conf=0, num=20):
    """ Visualize the entire virtual reconstructed box. """
    from vis.halo import renderSingleHalo

    rVirFracs  = [1.0]
    axes       = [0,1] # x,y
    labelZ     = False
    labelScale = False
    labelHalo  = True
    rotation   = None # random
    nPixels    = 600
    size       = 3.0
    sizeType   = 'rVirial'

    if num == 1:
        # for single halo showcase image
        nPixels = [1920, 1080]
        size = 4.0
        print('TODO: add insets of other properties, including zooms onto BCG')

    method = 'sphMap_globalZoomOrig' # all particles of original zoom run only

    if conf == 0:
        partType = 'gas'
        partField = 'coldens_msunkpc2'
        valMinMax = [5.8, 7.8]
    if conf == 1:
        partType = 'gas'
        partField = 'xray'
        valMinMax = [34.0, 39.5]

    # targets
    pri_target = sP.groups('GroupPrimaryZoomTarget')
    subIDs = sP.groups('GroupFirstSub')[np.where(pri_target == 1)]

    if num == 1:
        subIDs = [subIDs[2]]
    else:
        subIDs = subIDs[0:num]

    # panels
    panels = []

    for subID in subIDs:
        panels.append( {'subhaloInd':subID} )

    panels[0]['labelScale'] = 'physical'

    class plotConfig:
        plotStyle  = 'edged' # open, edged
        rasterPx   = nPixels
        colorbars  = True if num > 1 and num < 72 else False
        fontsize   = 24
        nCols      = int(np.floor(np.sqrt(num)))
        nRows      = int(np.ceil(num/nCols))

        saveFilename = './gallery_%s_%d_%s-%s_n%d.pdf' % (sP.simName,sP.snap,partType,partField,num)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

def mass_function():
    """ Plot halo mass function from the parent box (TNG300) and the zoom sample. """
    from cosmo.zooms import _halo_ids_run
    
    mass_range = [14.0, 15.5]
    binSize = 0.1
    redshift = 0.0
    
    sP_tng300 = simParams(res=2500,run='tng',redshift=redshift)
    sP_tng1 = simParams(res=2048, run='tng_dm', redshift=redshift)

    # load halos
    halo_inds = _halo_ids_run()

    print(len(halo_inds))
    print(halo_inds)

    # start figure
    fig = plt.figure(figsize=figsize)

    nBins = int((mass_range[1]-mass_range[0])/binSize)

    ax = fig.add_subplot(1,1,1)
    ax.set_xlim(mass_range)
    ax.set_xticks(np.arange(mass_range[0],mass_range[1],0.1))
    ax.set_xlabel('Halo Mass M$_{\\rm 200,crit}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('Number of Halos [%.1f dex$^{-1}$]' % binSize)
    ax.set_yscale('log')
    ax.yaxis.set_ticks_position('both')

    yy_max = 1.0

    hh = []
    labels = []

    for sP in [sP_tng300,sP_tng1]:
        if sP == sP_tng300:
            # tng300
            gc = sP_tng300.groupCat(fieldsHalos=['Group_M_Crit200'])
            masses = sP_tng300.units.codeMassToLogMsun(gc)
            label = 'TNG300-1'
        elif sP == sP_tng1:
            # tng1 - achieved targets (from runs.txt)
            gc = sP_tng1.groupCat(fieldsHalos=['Group_M_Crit200'])
            masses = sP_tng1.units.codeMassToLogMsun(gc[halo_inds])
            label = 'TNG-Cluster'
        else:
            # OLD: tng1 parent box for zooms (planned targets)
            gc = sP_tng1.groupCat(fieldsHalos=['Group_M_Crit200'])
            halo_inds = pick_halos()
            first_bin = 5 # >=14.5
            masses = [gc[inds] for inds in halo_inds[first_bin:]] # list of masses in each bin
            masses = np.hstack(masses)
            masses = sP_tng1.units.codeMassToLogMsun(masses)
            label = 'TNG-Cluster'

        w = np.where(~np.isnan(masses))
        yy, xx = np.histogram(masses[w], bins=nBins, range=mass_range)
        yy_max = np.nanmax([yy_max,np.nanmax(yy)])

        print(xx,yy)

        hh.append(masses[w])
        labels.append(label)

    # 'bonus': halos above 14.0 in the high-res regions of more massive zoom targets
    if 0:
        cacheFile = 'cache_mass_function_bonus.hdf5'
        if isfile(cacheFile):
            with h5py.File(cacheFile,'r') as f:
                masses = f['masses'][()]
        else:
            masses = []
            for i, hInd in enumerate(halo_inds):
                # only runs with existing data
                if not isdir('sims.TNG_zooms/L680n2048TNG_h%d_L13_sf3' % hInd):
                    print('[%3d of %3d]  skip' % (i,len(halo_inds)))
                    continue

                # load FoF catalog, record clusters with zero contamination
                sP = simParams(res=13, run='tng_zoom', redshift=redshift, hInd=hInd, variant='sf3')
                loc_masses = sP.halos('Group_M_Crit200')
                loc_masses = sP.units.codeMassToLogMsun(loc_masses[1:]) # skip FoF 0 (assume is target)

                loc_length = sP.halos('GroupLenType')[1:,:]
                contam_frac = loc_length[:,sP.ptNum('dm_lowres')] / loc_length[:,sP.ptNum('dm')]

                w = np.where( (loc_masses >= mass_range[0]) & (contam_frac == 0) )

                if len(w[0]):
                    masses = np.hstack( (masses,loc_masses[w]) )
                print('[%3d of %3d] ' % (i,len(halo_inds)), hInd, len(w[0]), len(masses))
            with h5py.File(cacheFile,'w') as f:
                f['masses'] = masses

        hh.append(masses)
        labels.append('TNG1 Bonus')

    # plot
    ax.hist(hh,bins=nBins,range=mass_range,label=labels,histtype='bar',alpha=0.9,stacked=True)

    ax.set_ylim([0.8,100])
    ax.legend(loc='upper right')

    fig.savefig('mass_functions.pdf')
    plt.close(fig)

def sample_halomasses_vs_redshift(sPs):
    """ Compare simulation vs observed cluster samples as a function of (redshift,mass). """
    from load.data import rossetti17planck, pintoscastro19, hilton20act, adami18xxl
    from load.data import bleem20spt, piffaretti11rosat

    redshifts = np.linspace(0.0, 0.6, 13) #[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    zspread = (redshifts[1]-redshifts[0]) / 3 # add random noise along redshift axis
    alpha = 0.6 # for data
    msize = 30 # scatter() default is 20

    np.random.seed(424242)

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_rasterization_zorder(1) # elements below z=1 are rasterized

    ax.set_xlabel('Redshift')
    ax.set_ylabel('Halo Mass M$_{\\rm 500,crit}$ [log M$_{\\rm sun}$]')

    ax.set_xlim([redshifts[0]-zspread*1.01, redshifts[-1]+zspread*1.01])
    ax.set_ylim([14.0, 15.6])

    # plot obs samples
    r17 = rossetti17planck()
    pc19 = pintoscastro19(sPs[0])
    h20 = hilton20act()
    a18 = adami18xxl()
    b20 = bleem20spt(sPs[0])
    p11 = piffaretti11rosat()

    d1 = ax.scatter(r17['z'], r17['m500'], s=msize+8, c='#000000', marker='s', alpha=alpha, label=r17['label'], zorder=0)

    d2 = ax.scatter(pc19['z'], pc19['m500'], s=msize+8, c='#222222', marker='*', alpha=alpha, label=pc19['label'], zorder=0)

    d3 = ax.scatter(h20['z'], h20['m500'], s=msize-9, c='#222222', marker='p', alpha=alpha-0.3, label=h20['label'], zorder=0)

    d4 = ax.scatter(a18['z'], a18['m500'], s=msize+8, c='#222222', marker='D', alpha=alpha, label=a18['label'], zorder=0)

    d5 = ax.scatter(b20['z'], b20['m500'], s=msize+8, c='#222222', marker='X', alpha=alpha, label=b20['label'], zorder=0)

    d6 = ax.scatter(p11['z'], p11['m500'], s=msize-4, c='#222222', marker='h', alpha=alpha-0.3, label=p11['label'], zorder=0)

    # add first legend
    legend1 = ax.legend(loc='upper right', frameon=True)
    legend1.get_frame().set_edgecolor('#bbbbbb')
    legend1.get_frame().set_linewidth(1.0)
    ax.add_artist(legend1)

    # load simulations and plot
    for i, sP in enumerate(sPs):
        color = next(ax._get_lines.prop_cycler)['color']

        for j, redshift in enumerate(redshifts):
            print(sP.simName, redshift)
            sP.setRedshift(redshift)
            m500 = sP.subhalos('mhalo_500_log')
            pri_target = sP.halos('GroupPrimaryZoomTarget')[ sP.subhalos('SubhaloGrNr') ]
                
            with np.errstate(invalid='ignore'):
                w = np.where((m500 > ax.get_ylim()[0]) & (pri_target == 1))

            # scatterplot
            xx = np.random.uniform(low=-zspread, high=zspread, size=len(w[0])) + redshift
            label = sP.simName if j == 0 else ''

            ax.scatter(xx, m500[w], s=msize, c=color, marker='o', alpha=1.0, label=label)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles[-2:], labels[-2:], loc='upper left', frameon=True)
    legend2.get_frame().set_edgecolor('#bbbbbb')
    legend2.get_frame().set_linewidth(1.0)

    # plot coma cluster
    def _plot_single_cluster(m500_msun, m500_err_up, m500_err_down, redshift, name):
        """ Helper. Input in linear msun. """
        m500 = np.log10([m500_msun, m500_err_up, m500_err_down])

        error_lower = m500[0] - m500[2]
        error_upper = m500[1] - m500[0]
        yerr = np.reshape( [error_lower, error_upper], (2,1) )

        d4 = ax.errorbar(redshift, m500[0], yerr=yerr, color='#000000', marker='H')
        ax.text(redshift+0.005, m500[0]+0.02, name, fontsize=14)

    # plot coma cluster (Okabe+2014 Table 8, g+ profile)
    coma_z = 0.0231
    coma_m500 = np.array([3.89, 3.89+1.04, 3.89-0.76]) * 1e14 / sP.HubbleParam

    _plot_single_cluster(coma_m500[0], coma_m500[1], coma_m500[2], coma_z, 'Coma')

    # plot pheonix cluster
    pheonix_z = 0.597 # currently off edge of plot
    pheonix_m500 = 2.34e15 # msun, Tozzi+15 (Section 3)
    m500_err = 0.71e15 # msun

    _plot_single_cluster(pheonix_m500, pheonix_m500+m500_err, pheonix_m500-m500_err, pheonix_z, 'Pheonix')

    # plot perseus cluster (note: virgo, fornax m500<1e14)
    perseus_z = 0.0183
    perseus_m500 = sP.units.m200_to_m500(6.65e14) # Simionescu+2011
    perseus_m500_errup = sP.units.m200_to_m500(6.65e14 + 0.43e14)
    perseus_m500_errdown = sP.units.m200_to_m500(6.65e14 - 0.46e14)

    _plot_single_cluster(perseus_m500, perseus_m500_errup, perseus_m500_errdown, perseus_z, 'Perseus')

    # plot eROSITA completeness goal
    erosita_minhalo = [0.20,0.32,0.47,0.65,0.86,1.12,1.44,1.87,2.33,2.91,3.46,4.19,4.86,5.80,6.68,7.33,7.79]
    erosita_z       = [0.05,0.08,0.11,0.14,0.17,0.21,0.25,0.32,0.38,0.47,0.56,0.69,0.82,1.03,1.30,1.60,1.92]

    erosita_minhalo = np.log10(sP.units.m200_to_m500(np.array(erosita_minhalo) * 1e14)) # log msun

    l, = ax.plot(erosita_z, erosita_minhalo, '-', lw=lw, alpha=alpha, color='#000000')
    ax.arrow(erosita_z[6], erosita_minhalo[6]+0.02, 0.0, 0.1, lw=lw, head_length=0.008, color=l.get_color())
    ax.text(erosita_z[7]-0.04, 14.02, 'eROSITA All-Sky Complete', color=l.get_color(), fontsize=14, rotation=21)

    fig.savefig('sample_halomass_vs_redshift.pdf')
    plt.close(fig)

def bfield_strength_vs_halomass(sPs, redshifts):
    """ Driver for quantMedianVsSecondQuant. """
    sPs_in = []
    for redshift in redshifts:
        for sP in sPs:
            sPloc = sP.copy()
            sPloc.setRedshift(redshift)
            sPs_in.append( sPloc )

    xQuant = 'mhalo_200_log'
    yQuant = 'bmag_halfr500_volwt'
    scatterColor = 'redshift'
    cenSatSelect = 'cen'

    xlim = [14.0, 15.4]
    ylim = [-0.65, 0.85]
    clim = [0.0, 2.0]
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    def _draw_data(ax):
        """ Draw data constraints on figure. """

        # Di Gennaro+2020 (https://arxiv.org/abs/2011.01628)
        #  -- measurements based on integrated flux within <~ 0.5*r500
        bmin = np.log10(1.0) # uG
        bmax = np.log10(3.0) # uG
        mass_range = sPs[0].units.m500_to_m200(np.array([5e14, 9e14])) # msun

        ax.fill_between(np.log10(mass_range), y1=[bmin,bmin], y2=[bmax,bmax], color='#cccccc', 
            label='Di Gennaro+20 ($z \sim 0.8$)')

        # Boehringer+2016 (https://arxiv.org/abs/1610.02887)
        # -- about ~90 measurements have mean r/r500 = 0.32, median r/r500 = 0.25
        bmin = np.log10(2.0) # uG
        bmax = np.log10(6.0) # uG
        mass_range = [2e14,4e14] # m200 msun

        ax.fill_between(np.log10(mass_range), y1=[bmin,bmin], y2=[bmax,bmax], color='#eeeeee', 
            label='B$\\rm\\"{o}$hringer+16 ($z \sim 0.1$)')

    def _draw_data2(ax):
        """ Draw additional data constraints on figure, individual halos. """
        b = np.log10(2.0) # uG, Bonafede+10 https://arxiv.org/abs/1002.0594
        yerr = np.reshape( [0.34, 0.34], (2,1) ) # 1-4.5 uG (center vs 1 Mpc), Bonafede+10
        m200 = 14.88 # Okabe+14 m500->m200
        xerr = np.reshape( [0.1, 0.1], (2,1) ) # Okabe+14

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color='#000000', marker='D', label='Bonafede+10 (Coma)')

        b = np.log10( (1.5+0.3)/2 ) # average of <B0>=1.5 uG and ~0.3 uG (volume average within 1 Mpc, ~1r500)
        yerr = np.reshape( [0.47, 0.22], (2,1) )
        m200 = np.log10(1e14 / 0.6774) # Govoni+17 Sec 4.2
        xerr = np.reshape( [0.1, 0.1], (2,1) ) # assumed, e.g. minimum of ~30% uncertainty

        ax.errorbar(m200, b, xerr=xerr, yerr=yerr, color='#000000', marker='H', label='Govoni+17 (Abell 194)')

        # TODO: Stuardi, C.+2021, Abell 2345
        # |B| = 2.8 +/- 0.1 uG (within 200 kpc)
        # M_500,SZ = 5.91e14 Msun

    quantMedianVsSecondQuant(sPs_in, pdf=None, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_pre=_draw_data, f_post=_draw_data2, legendLoc='lower right')

def stellar_mass_vs_halomass(sPs, conf=0):
    """ Plot various stellar mass quantities vs halo mass. """
    from load.data import behrooziSMHM, mosterSMHM, kravtsovSMHM

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [13.8, 15.4]
    clim = [-0.4, 0.0] # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    if conf == 0:
        yQuant = 'mstar_30pkpc'
        ylabel = 'BCG Stellar Mass [ log M$_{\\rm sun}$ ]'
        ylim = [10.9, 13.1]
        scatterColor = 'massfrac_exsitu2'

        def _draw_data(ax):
            # Kravtsov+ 2018 (Table 1 for M500crit + Table 4)
            label = 'Kravtsov+18'
            
            m500c = np.log10(np.array([15.60, 10.30, 7.00, 5.34, 2.35, 1.86, 1.34, 0.46, 0.47]) * 1e14)
            mstar_30pkpc = np.log10(np.array([5.18, 10.44, 7.12, 3.85, 3.67, 4.35, 4.71, 4.59, 6.76]) * 1e11)

            ax.scatter(m500c, mstar_30pkpc, s=markersize+20, c='#000000', marker='D', alpha=1.0, label=label)

            # empirical SMHM relations
            #b = behrooziSMHM(sPs[0], redshift=0.0)
            #m = mosterSMHM(sPs[0], redshift=0.0)
            #k = kravtsovSMHM(sPs[0])

            #ax.plot(b['m500c'], b['mstar_mid'], color='#333333', label='Behroozi+ (2013)')
            #ax.fill_between(b['m500c'], b['mstar_low'], b['mstar_high'], color='#333333', interpolate=True, alpha=0.3)

            #ax.plot(m['m500c'], m['mstar_mid'], color='#dddddd', label='Moster+ (2013)')
            #ax.fill_between(m['m500c'], m['mstar_low'], m['mstar_high'], color='#dddddd', interpolate=True, alpha=0.3)

            #ax.plot(k['m500c'], k['mstar_mid'], color='#888888', label='Kravtsov+ (2014)')

    if conf == 1:
        yQuant = 'mstar_r500'
        ylabel = 'Total Halo Stellar Mass [ log M$_{\\rm sun}$ ]' # BCG+SAT+ICL (e.g. <r500c)
        ylim = [11.9, 13.5]
        scatterColor = 'massfrac_exsitu'

        def _draw_data(ax):
            # Kravtsov+ 2018 (Figure 7 for r500c, Figure 8 for satellites within r500c)
            label = 'Kravtsov+18'
            m500c       = np.log10([5.31e13,5.68e13,1.29e14,1.79e14,2.02e14,5.40e14,5.87e14,8.59e14,1.19e15])
            mstar_r500c = np.log10([1.47e12,1.45e12,2.28e12,2.80e12,2.42e12,4.36e12,6.58e12,1.01e13,1.33e13])
            #mstar_sats  = np.log10([7.97e11,3.89e11,1.45e12,1.78e12,1.83e12,3.27e12,4.39e12,6.61e12,1.07e13])

            ax.scatter(m500c, mstar_r500c, s=markersize+20, c='#000000', marker='s', alpha=1.0, label=label)

            # Gonzalez+13 (Figure 7, mstar is <r500c, and msats is satellites within r500c)
            label = 'Gonzalez+13'
            m500c = [9.55e13,9.84e13,9.54e13,1.45e14,3.66e14,3.52e14,3.23e14,5.35e14,2.28e14,2.44e14,2.42e14,2.26e14]
            mstar = [2.82e12,3.21e12,4.18e12,3.06e12,4.99e12,6.07e12,7.53e12,7.04e12,5.95e12,5.95e12,5.56e12,5.50e12]
            #msats = [1.96e12,1.75e12,1.55e12,1.51e12,2.65e12,4.60e12,4.61e12,4.94e12,3.48e12,3.56e12,3.65e12,3.87e12]

            ax.scatter(np.log10(m500c), np.log10(mstar), s=markersize+20, c='#000000', marker='D', alpha=1.0, label=label)

            # Leauthaud+12 (obtained from Kravtsov+18 Fig 7)
            label = 'Leauthaud+12'
            m500c = [3e13, 4.26e14]
            mstar_r500c = [5.8e11, 5.6e12]

            ax.plot(np.log10(m500c), np.log10(mstar_r500c), '-', color='#000000', alpha=1.0, label=label)

            # Bahe+17 (Hydrangea sims, Fig 4 left) (arXiv:1703.10610)
            label = 'Bahe+17 (sims)'
            m500c = [13.83,13.92,13.88,13.97,14.04,14.07,14.29,14.31,14.35,14.40,14.42,14.48,14.55,14.58,
                     14.64,14.79,14.81,14.84,14.90,14.90,15.04,15.07] # 14.69,
            mstar = [12.02,12.14,12.21,12.25,12.32,12.29,12.47,12.53,12.49,12.60,12.66,12.69,12.71,12.76,
                     12.81,13.00,12.97,12.99,13.05,13.08,13.17,13.23] # 12.38,

            ax.scatter(m500c, mstar, s=markersize+20, c='#000000', marker='*', alpha=1.0, label=label)

    if conf == 2:
        yQuant = 'TODO'
        ylabel = 'Stellar Mass <100 pkpc)[ log M$_{\\rm sun}$ ]'

        def _draw_data(ax):
            pass

    quantMedianVsSecondQuant(sPs, pdf=None, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_post=_draw_data, ylabel=ylabel, legendLoc='lower right')

def gas_fraction_vs_halomass(sPs):
    """ Plot f_gas vs halo mass. """
    from load.data import giodini2009, lovisari2015

    xQuant = 'mhalo_500_log'
    cenSatSelect = 'cen'

    xlim = [13.8, 15.4]
    clim = [-0.4, 0.0] # log fraction
    scatterPoints = True
    drawMedian = False
    markersize = 40.0
    sizefac = 0.8 # for single column figure

    yQuant = 'fgas_r500'
    ylim = [0.05, 0.18]
    scatterColor = None #'massfrac_exsitu2'

    def _draw_data(ax):
        # observational points
        g = giodini2009(sPs[0])
        l = lovisari2015(sPs[0])

        ax.errorbar(g['m500_logMsun'], g['fGas500'], yerr=g['fGas500Err'],
                             color='#999999', alpha=0.9, 
                             fmt=markers[0]+linestyles[0],label=g['label'])
        ax.errorbar(l['m500_logMsun'], l['fGas500'], #yerr=l['fGas500Err'],
                             color='#555555', alpha=0.9, 
                             marker=markers[0],linestyle='',label=l['label'])

        # Tanimura+2020 (https://arxiv.org/abs/2007.02952) (xerr assumed)
        ax.errorbar( np.log10(0.9e14/0.6774), 0.13, xerr=0.1, yerr=0.03, marker='D', alpha=0.9, 
                     color='#333333', label='Tanimura+ (2020) z~0.4')

        # universal baryon fraction line
        OmegaU = sPs[0].omega_b / sPs[0].omega_m
        ax.plot( xlim, [OmegaU,OmegaU], '--', lw=1.0, color='#444444', alpha=0.2)
        ax.text( xlim[1]-0.2, OmegaU+0.003, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', size='large', alpha=0.2)

    quantMedianVsSecondQuant(sPs, pdf=None, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=cenSatSelect, 
                             xlim=xlim, ylim=ylim, clim=clim, drawMedian=drawMedian, markersize=markersize,
                             scatterPoints=scatterPoints, scatterColor=scatterColor, sizefac=sizefac, 
                             f_post=_draw_data, legendLoc='lower right')

def paperPlots():
    """ Plots for TNG-Cluster intro paper. """
    from util.simParams import simParams

    # all analysis at z=0 unless changed below
    TNG300 = simParams(run='tng300-1', redshift=0.0)
    TNG_C  = simParams(run='tng-cluster', redshift=0.0)

    sPs = [TNG300, TNG_C]

    # figure 1 - mass function
    if 0:
        mass_function()

    # figure 2 - samples
    if 0:
        sample_halomasses_vs_redshift(sPs)

    # figure 3 - virtual full box vis
    if 0:
        for conf in [0,1,2,3,4,5]:
            vis_fullbox_virtual(TNG_C, conf=conf)

    # figure 4 - individual halo/gallery vis (x-ray)
    if 0:
        vis_gallery(TNG_C, conf=1, num=1) # single in xray
        for conf in [0,1]:
            vis_gallery(TNG_C, conf=conf, num=72) # gallery

    # figure X - magnetic fields
    if 0:
        redshifts = [0.0, 1.0, 2.0]
        bfield_strength_vs_halomass(sPs, redshifts)

    # figure X - stellar mass contents
    if 0:
        for conf in [0,1]:
            stellar_mass_vs_halomass(sPs, conf=conf)

    # figure X - gas fractions
    if 0:
        gas_fraction_vs_halomass(sPs)

    # figure X - black hole mass scaling relation
    if 0:
        from plot.globalComp import blackholeVsStellarMass

        pdf = PdfPages('blackhole_masses_vs_mstar_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        blackholeVsStellarMass(sPs, pdf, twiceR=True, xlim=[11,13.0], ylim=[7.5,11], actualLargestBHMasses=True)
        pdf.close()

        #pdf = PdfPages('blackhole_masses_vs_mbulge_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        #blackholeVsStellarMass(sPs, pdf, vsBulgeMass=True, xlim=[11,13.0], ylim=[7.5,11], actualLargestBHMasses=True)
        #pdf.close()

        #pdf = PdfPages('blackhole_masses_vs_mhalo_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        #blackholeVsStellarMass(sPs, pdf, vsHaloMass=True, xlim=[13.8,15.4], ylim=[8.5,11], actualLargestBHMasses=True)
        #pdf.close()

    # figure X - BCG stellar sizes
    if 0:
        from plot.sizes import galaxySizes
        pdf = PdfPages('galaxy_stellar_sizes_%s_z%d.pdf' % ('-'.join(sP.simName for sP in sPs),sPs[0].redshift))
        galaxySizes(sPs, pdf, xlim=[11.0,13.0], ylim=[3,300], onlyRedData=True)
        pdf.close()

    # figure todo - entropy profiles
    # figure todo - cc/ncc fractions vs mass
    # figure todo - Lx vs mass
    # figure todo - SZ-y vs mass
