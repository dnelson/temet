"""
globalComp.py
  run summary plots and comparisons to constraints
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.loadExtern import *
from util.helper import running_median, running_histogram, logZeroSafe
from cosmo.load import groupCat, groupCatSingle, auxCat, groupCatHasField, snapHasField, snapshotHeader
from cosmo.util import validSnapList, periodicDists
from plot.galaxyColor import galaxyColorPDF, galaxyColor2DPDFs
from plot.sizes import galaxySizes
from plot.cosmoGeneral import addRedshiftAgeAxes
from plot.config import *

# still overriding figsize for globalComps:
figsize = (16,9) # (16,9) (8,6) (smf:10.6,7.0)

def stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=False, use30kpc=False, 
                        simRedshift=0.0, dataRedshift=0.0, fig_subplot=[None,None]):
    """ Stellar mass / halo mass relation, full boxes and zoom points vs. abundance matching lines. """
    # plot setup
    xrange = [10.0, 15.0]
    yrange = [0.0, 0.30]
    if dataRedshift > 0.0: yrange[1] = 0.25

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])

    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    if ylog:
        ax.set_yscale('log')
        ax.set_ylim([1e-3,1e0])

    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
    ax.set_ylabel('M$_\star$ / M$_{\\rm halo}$ $(\Omega_{\\rm b} / \Omega_{\\rm m})^{-1}$ [ only centrals ]')

    # observational data: abundance matching constraints
    b = behrooziSMHM(sPs[0], redshift=dataRedshift)
    m = mosterSMHM(sPs[0], redshift=dataRedshift)
    k = kravtsovSMHM(sPs[0])

    ax.plot(b['haloMass_i'], b['y_mid_i'], color='#333333', label='Behroozi+ (2013) z=%d' % dataRedshift)
    ax.fill_between(b['haloMass_i'], b['y_low_i'], b['y_high_i'], color='#333333', interpolate=True, alpha=0.3)

    ax.plot(m['haloMass'], m['y_mid'], color='#dddddd', label='Moster+ (2013) z=%d' % dataRedshift)
    ax.fill_between(m['haloMass'], m['y_low'], m['y_high'], color='#dddddd', interpolate=True, alpha=0.3)

    if dataRedshift == 0.0:
        ax.plot(k['haloMass'], k['y_mid'], color='#888888', label='Kravtsov+ (2014) z=0')

    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc='upper right')
    plt.gca().add_artist(legend1)

    # loop over each run
    lines = []

    for i, sP in enumerate(sPs):
        sP.setRedshift(simRedshift)
        print('SMHM (z=%d): %s (z=%d)' % (dataRedshift,sP.simName,sP.redshift))

        if sP.isZoom:
            gc = groupCatSingle(sP, subhaloID=sP.zoomSubhaloID)
            gh = groupCatSingle(sP, haloID=gc['SubhaloGrNr'])

            # halo mass definition
            xx_code = gh['Group_M_Crit200'] #gc['SubhaloMass'] 
            xx = sP.units.codeMassToLogMsun( xx_code )

            # stellar mass definition(s)
            yy = gc['SubhaloMassType'][4] / xx_code / (sP.omega_b/sP.omega_m)
            ax.plot(xx,yy,sP.marker,color=colors[0])

            yy = gc['SubhaloMassInRadType'][4] / xx_code / (sP.omega_b/sP.omega_m)
            ax.plot(xx,yy,sP.marker,color=colors[1])

            yy = gc['SubhaloMassInHalfRadType'][4] / xx_code / (sP.omega_b/sP.omega_m)
            ax.plot(xx,yy,sP.marker,color=colors[2])

        else:
            # fullbox:
            gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], 
              fieldsSubhalos=['SubhaloMass','SubhaloMassType',
                              'SubhaloMassInRadType','SubhaloMassInHalfRadType'])

            label = sP.simName + ' z=%d' % sP.redshift

            # centrals only
            wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0) & (gc['halos']['Group_M_Crit200'] > 0))
            w = gc['halos']['GroupFirstSub'][wHalo]

            # halo mass definition
            xx_code = gc['halos']['Group_M_Crit200'][wHalo]
            xx = sP.units.codeMassToLogMsun( xx_code )

            # stellar mass definition(s)
            c = ax._get_lines.prop_cycler.next()['color']

            if use30kpc:
                # load auxcat
                field = 'Subhalo_Mass_30pkpc_Stars'
                ac = auxCat(sP, fields=[field])

                yy = ac[field][w] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], color=c, lw=3.0, label=label)
                lines.append(l)

            if allMassTypes:
                yy = gc['subhalos']['SubhaloMassType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[1], color=c, lw=3.0)

            if not use30kpc or allMassTypes:
                # primary (in 2rhalf_stars)
                yy = gc['subhalos']['SubhaloMassInRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], lw=3.0, color=c, label=label)
                lines.append(l)

            if allMassTypes:
                yy = gc['subhalos']['SubhaloMassInHalfRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[2], lw=3.0, color=c)

    # second legend
    markers = []
    sExtra = []
    lExtra = []

    for handle in lines:
        sExtra.append( handle )
        lExtra.append( handle.get_label() )

    if allMassTypes:
        sExtra += [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[1]),
                  plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[0]),
                  plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[2])]
        lExtra += [r'$M_\star^{\rm tot}$',
                  r'$M_\star^{< 2r_{1/2}}$', 
                  r'$M_\star^{< r_{1/2}}$']
    if use30kpc:
        sExtra += [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[0])]
        lExtra += [r'$M_\star$ (< 30 pkpc)']

    for sP in sPs:
        if not sP.isZoom or sP.marker in markers:
            continue
        sExtra.append( plt.Line2D((0,1),(0,0),color='black',marker=sP.marker,linestyle='',label='test') )
        lExtra.append( sP.simName + ' z=%d' % sP.redshift )
        markers.append( sP.marker )

    legend2 = ax.legend(sExtra, lExtra, loc='upper left')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def stellarMassHaloMassMultiPanel(sPs, pdf, ylog=False, allMassTypes=False, use30kpc=False, 
                                  redshifts=[0,1,2,3]):
    """ Stellar mass / halo mass relation (2x2 panels at 4 different redshifts, matching simulation and 
    data redshifts), vs. abundance matching lines. """

    nRows = np.floor(np.sqrt(len(redshifts)))
    nCols = np.ceil(len(redshifts) / nRows)

    sizefac_loc = 0.9
    figsize_loc = [figsize[0]*nCols*sizefac_loc, figsize[1]*nRows*sizefac_loc]

    fig = plt.figure(figsize=figsize_loc)

    for i, redshift in enumerate(redshifts):
        # append each panel to existing figure, which is automatically saved into pdf at the end
        ind = int(nRows*100 + nCols*10 + (i+1))
        stellarMassHaloMass(sPs, pdf, ylog=ylog, allMassTypes=allMassTypes, use30kpc=use30kpc, 
                            simRedshift=redshift, dataRedshift=redshift, fig_subplot=[fig,ind])

def sfrAvgVsRedshift(sPs, pdf):
    """ Average SFRs in some halo mass bins vs. redshift vs. abundance matching lines. """
    from util import simParams

    # config
    plotMassBins  = [10.6,11.2,11.8]
    massBinColors = ['#333333','#666666','#999999']

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_ylim([8e-3, 5e2])
    addRedshiftAgeAxes(ax, sPs[0])
    ax.set_ylabel('<SFR> [ M$_{\\rm sun}$ / yr ] [ < 2r$_{1/2}$ ] [ only centrals ]')
    ax.set_yscale('log')    

    # calculate and cache from simulations function
    def _loadSfrAvg(sP, haloMassBins, haloBinSize, maxNumSnaps=60):
        """ Helper function to calculate average SFR in halo mass bins across snapshots. """
        snaps = validSnapList(sP, maxNum=maxNumSnaps)

        saveFilename = sP.derivPath + 'sfr_avgs_%d-%d_%d.hdf5' % (snaps.min(),snaps.max(),len(snaps))

        if isfile(saveFilename):
            print(' Loaded: [%s]' % saveFilename.split(sP.derivPath)[1])
            r = {}
            with h5py.File(saveFilename,'r') as f:
                for key in f:
                    r[key] = f[key][()]
            return r

        # allocate
        sfrFields = ['SubhaloSFR','SubhaloSFRinRad','SubhaloSFRinHalfRad']

        r = {}

        r['haloMassBins'] = haloMassBins
        r['haloBinSize'] = haloBinSize
        r['redshifts'] = np.zeros(len(snaps))

        r['sfrs_med']  = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')
        r['sfrs_mean'] = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')
        r['sfrs_std']   = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')
        r['sfrs_med_noZero']  = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')
        r['sfrs_mean_noZero'] = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')
        r['sfrs_std_noZero']   = np.zeros((len(snaps), len(haloMassBins), 3), dtype='float32')

        r['sfrs_med'].fill(np.nan)
        r['sfrs_mean'].fill(np.nan)
        r['sfrs_std'].fill(np.nan)
        r['sfrs_med_noZero'].fill(np.nan)
        r['sfrs_mean_noZero'].fill(np.nan)
        r['sfrs_std_noZero'].fill(np.nan)

        # loop over all snapshots
        for j, snap in enumerate(snaps):
            print(' snap %d [%d of %d]' % (snap,j,len(snaps)))
            sP.setSnap(snap)

            gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], fieldsSubhalos=sfrFields)

            if not gc['halos']['count']:
                continue # high redshift

            r['redshifts'][j] = sP.redshift

            # centrals only, given halo mass definition, in this halo mass bin
            for k, haloMassBin in enumerate(haloMassBins):
                haloMassesLogMsun = sP.units.codeMassToLogMsun( gc['halos']['Group_M_Crit200'] )

                wHalo = np.where( (gc['halos']['GroupFirstSub'] >= 0) & \
                                  (haloMassesLogMsun > haloMassBin-0.5*haloBinSize) & \
                                  (haloMassesLogMsun <= haloMassBin+0.5*haloBinSize) )

                if len(wHalo[0]) == 0:
                    continue

                w = gc['halos']['GroupFirstSub'][wHalo]

                # sfr definition(s)
                for m, sfrField in enumerate(sfrFields):
                    r['sfrs_med'][j,k,m]  = np.median( gc['subhalos'][sfrField][w] )
                    r['sfrs_mean'][j,k,m] = np.mean( gc['subhalos'][sfrField][w] )
                    r['sfrs_std'][j,k,m]  = np.std( gc['subhalos'][sfrField][w] )

                    # repeat but exclude all SFR==0 entries
                    loc_w = np.where( gc['subhalos'][sfrField][w] > 0.0 )

                    if len(loc_w[0]) == 0:
                        continue

                    r['sfrs_med_noZero'][j,k,m]  = np.median( gc['subhalos'][sfrField][w][loc_w] )
                    r['sfrs_mean_noZero'][j,k,m] = np.mean( gc['subhalos'][sfrField][w][loc_w] )
                    r['sfrs_std_noZero'][j,k,m]  = np.std( gc['subhalos'][sfrField][w][loc_w] )

        # save
        with h5py.File(saveFilename,'w') as f:
            for key in r:
                f[key] = r[key]
        print(' Saved: [%s]' % saveFilename.split(sP.derivPath)[1])
        return r

    # load observational data
    b = behrooziSFRAvgs()

    for i, massBin in enumerate(plotMassBins):
        xx = b[str(massBin)]['redshift']
        yy = b[str(massBin)]['sfr']
        yyDown = b[str(massBin)]['sfr'] - b[str(massBin)]['errorDown']
        yyUp   = b[str(massBin)]['sfr'] + b[str(massBin)]['errorUp']

        label = 'Behroozi+ (2013) $10^{' + str(massBin) + '}$ M$_{\\rm sun}$ Halos'
        l, = ax.plot(xx, yy, label=label, color=massBinColors[i])
        ax.fill_between(xx, yyDown, yyUp, color=l.get_color(), interpolate=True, alpha=0.3)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('SFRavg: '+sP.simName)

        # load saved simulation data
        simData = _loadSfrAvg(sP, b['haloMassBins'], b['haloBinSize'])
        xx = simData['redshifts']

        # plot line for each halo mass bin
        c = ax._get_lines.prop_cycler.next()['color']

        for haloMassBin in plotMassBins:
            # locate this mass bin in saved data
            k = np.where(simData['haloMassBins'] == haloMassBin)[0]
            assert len(k) == 1

            # different sfr definitions
            for j in [1]: # <2r1/2
                label = sP.simName if (haloMassBin==plotMassBins[0] and j==1) else ''

                #ax.plot(xx, simData['sfrs_med'][:,k,j], ':', color=c, lw=3.0, label=label)
                ax.plot(xx, simData['sfrs_med_noZero'][:,k,j], '-', color=c, lw=3.0, label=label)

                #if sP == sPs[0] and j == 1:
                #    yy_down = simData['sfrs_med_noZero'][:,k,j] - simData['sfrs_std_noZero'][:,k,j]
                #    yy_up = simData['sfrs_med_noZero'][:,k,j] + simData['sfrs_std_noZero'][:,k,j]
                #    ax.fill_between(xx, np.squeeze(yy_down), np.squeeze(yy_up), 
                #                    color=c, interpolate=True, alpha=0.2)

    # legend
    ax.legend(loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def sfrdVsRedshift(sPs, pdf, xlog=True):
    """ Star formation rate density of the universe, vs redshift, vs observational points. """
    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_ylim([5e-4, 5e-1])
    addRedshiftAgeAxes(ax, sPs[0], xlog=xlog)
    ax.set_ylabel('SFRD [ M$_{\\rm sun}$  yr$^{-1}$  Mpc$^{-3}$]')
    ax.set_yscale('log')

    # observational points
    be = behrooziObsSFRD()

    l1,_,_ = ax.errorbar(be['redshift'], be['sfrd'], yerr=[be['errorDown'],be['errorUp']], 
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='o')

    bo = bouwensSFRD2014()

    l2,_,_ = ax.errorbar(bo['redshift'], bo['sfrd'], 
                         xerr=bo['redshiftErr'], yerr=[bo['errorDown'],bo['errorUp']], 
                         color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='D')

    legend1 = ax.legend([l1,l2], [be['label'], bo['label']], loc='upper right')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        # load sfr.txt file
        print('SFRD: '+sP.simName)
        s = sfrTxt(sP)

        ax.plot(s['redshift'], s['sfrd'], '-', lw=2.5, label=sP.simName)

    # second legend
    legend2 = ax.legend(loc='lower left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def blackholeVsStellarMass(sPs, pdf, twiceR=False, vsHaloMass=False, vsBulgeMass=False, 
                           actualBHMasses=False, actualLargestBHMasses=True, simRedshift=0.0):
    """ Black hole mass vs. stellar (bulge) mass relation at z=0. """
    assert twiceR or vsHaloMass or vsBulgeMass

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([8.5, 13.0])
    ax.set_ylim([5.5, 11.0])

    ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ tot dyn sum, only cen ]')
    if actualBHMasses:
        ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ actual sum, only cen ]')
    if actualLargestBHMasses:
        ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ actual max, only cen ]')

    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 1r$_{1/2}$ ]')
    if twiceR:
        ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
    if vsHaloMass:
        ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([9,14.5])
        ax.set_ylim([5.0, 11.0])
    if vsBulgeMass:
        ax.set_xlabel('M$_{\\rm bulge,\star}$ [ log M$_{\\rm sun}$ ] [ 2*counter-rotating < 1r$_{1/2}$ ]')

    # observational points
    if not vsHaloMass:
        k = kormendyHo2013()
        m = mcconnellMa2013()

        l3,_,_ = ax.errorbar(m['pts']['M_bulge'], m['pts']['M_BH'], 
                             yerr=[m['pts']['M_BH_down'],m['pts']['M_BH_up']],
                             color='#bbbbbb', ecolor='#dddddd', alpha=0.9, capsize=0.0, fmt='D')

        l2, = ax.plot(m['M_bulge'], m['M_BH'], '-', color='#999999')
        ax.fill_between(m['M_bulge'], m['errorDown'], m['errorUp'], color='#999999', interpolate=True, alpha=0.3)

        l1, = ax.plot(k['M_bulge'], k['M_BH'], '-', color='#333333')
        ax.fill_between(k['M_bulge'], k['errorDown'], k['errorUp'], color='#333333', interpolate=True, alpha=0.3)

        legend1 = ax.legend([l1,l2,l3], [k['label'], m['label'], m['pts']['label']], loc='upper left')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        sP.setRedshift(simRedshift)

        numBHs = snapshotHeader(sP)['NumPart'][sP.ptNum('bhs')]
        if not groupCatHasField(sP, 'Subhalo', 'SubhaloBHMass') or numBHs == 0:
            print('BHMass: %s [SKIP: sim has no BHs]' % sP.simName)
            continue

        print('BHMass: '+sP.simName)

        fieldsSubhalos = ['SubhaloBHMass','SubhaloMassType']
        if vsBulgeMass: fieldsSubhalos.append('SubhaloMassInHalfRadType')
        if twiceR: fieldsSubhalos.append('SubhaloMassInRadType')

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], fieldsSubhalos=fieldsSubhalos)

        # centrals only
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        # stellar mass definition: would want to mimic bulge mass measurements
        if not twiceR and not vsHaloMass:
            xx_code = gc['subhalos']['SubhaloMassInHalfRadType'][w,sP.ptNum('stars')]
        if twiceR:
            xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]
        if vsHaloMass:
            xx_code = gc['halos']['Group_M_Crit200'][wHalo]
        if vsBulgeMass:
            # load auxCat and compute bulge-mass
            acField = 'Subhalo_StellarRotation_1rhalfstars'
            acIndex = 2

            ac = auxCat(sP, fields=[acField])
            ac = np.squeeze( ac[acField][w,acIndex] ) # counter-rotating mass fraction relative to total
            ac[np.where(np.isnan(ac))] = 0.0 # set NaN to zero (consistent with groupcat)

            # multiply 2 x (massfrac) x (stellar mass)
            mass_1rhalf = gc['subhalos']['SubhaloMassInHalfRadType'][w,sP.ptNum('stars')]
            xx_code = 2.0 * ac * np.squeeze(mass_1rhalf)

        xx = sP.units.codeMassToLogMsun( xx_code )

        # 'total' black hole mass in this subhalo, exclude those with no BHs
        # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here 
        # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
        if actualBHMasses:
            # "actual" BH masses, excluding gas reservoir
            yy = gc['subhalos']['SubhaloBHMass'][w]
        else:
            # dynamical (particle masses)
            yy = gc['subhalos']['SubhaloMassType'][w,sP.ptNum('bhs')]
        if actualLargestBHMasses:
            # load auxCat (fix this problem by using the most massive BH in each subhalo)
            acField = 'Subhalo_BH_Mass_largest'
            ac = auxCat(sP, fields=[acField])[acField]
            ac[np.where(np.isnan(ac))] = 0.0 # set NaN to zero (consistent with groupcat)
            yy = ac[w]

        yy = sP.units.codeMassToLogMsun(yy)
        ww = np.where(yy > 0.0)

        xm, ym, sm = running_median(xx[ww],yy[ww],binSize=binSize,skipZeros=True)
        ym2 = savgol_filter(ym,sKn,sKo)
        sm2 = savgol_filter(sm,sKn,sKo)
        l, = ax.plot(xm[:-1], ym2[:-1], '-', lw=3.0, label=sP.simName)

        if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2):
            y_down = np.array(ym2[:-1]) - sm2[:-1]
            y_up   = np.array(ym2[:-1]) + sm2[:-1]
            ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.2)

    # second legend
    legend2 = ax.legend(loc='lower right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def stellarMassFunction(sPs, pdf, highMassEnd=False, centralsOnly=False, use30kpc=False, use30H=False, 
                        useP10=False, simRedshift=0.0, dataRedshift=0.0, fig_subplot=[None,None]):
    """ Stellar mass function (number density of galaxies) at redshift zero, or above. """
    # config
    mts = ['SubhaloMassInRadType','SubhaloMassInHalfRadType','SubhaloMassType']

    cutClumps = False

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])
    
    ax.set_ylim([5e-6,2e-1])
    ax.set_xlim([7,12.5])
    if clean: ax.set_xlim([5,12.5])
    if dataRedshift >= 3.0: ax.set_ylim([5e-7,6e-2])

    if highMassEnd:
        #ax.set_ylim([1e-7,2e-2])
        #ax.set_xlim([10.0,12.5])
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < various ]')
    else:
        #ax.set_ylim([5e-4,2e-1])
        #ax.set_xlim([7,11.5])
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{\star,1/2}$ ]')

    if centralsOnly:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ only centrals ]')
    else:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ centrals & satellites ]')
    ax.set_yscale('log')
    if clean: ax.set_ylabel('Stellar Mass Function [ Mpc$^{-3}$ dex$^{-1}$ ]')

    if use30kpc:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 30 pkpc ]')
        if clean: ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]')
    if use30H:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < min(2r$_{\star,1/2}$,30 pkpc) ]')
    if useP10:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < puchwein2010 r$_{\\rm cut}$ ]')

    # observational points
    data = []
    lines = []

    if dataRedshift == 0.0:
        data.append( baldry2008SMF() )
        data.append( baldry2012SMF() )
        data.append( bernardi2013SMF()['SerExp'] )
        data.append( dsouza2015SMF() )
    if dataRedshift == 1.0:
        raise Exception('todo')
    if dataRedshift == 2.0:
        raise Exception('todo')
    if dataRedshift == 3.0:
        data.append( davidzon2017SMF(redshift=2.5) )
        data.append( davidzon2017SMF(redshift=3.0) )
        data.append( caputi2015SMF(redshift=3) )
    if dataRedshift == 4.0:
        data.append( davidzon2017SMF(redshift=3.5) )
        data.append( song2015SMF(redshift=4) )
        data.append( caputi2015SMF(redshift=3) )
        data.append( caputi2015SMF(redshift=4) )
        data.append( grazian2015SMF(redshift=3.5) )
    if dataRedshift == 5.0:
        data.append( davidzon2017SMF(redshift=4.5) )
        data.append( song2015SMF(redshift=5) )
        data.append( caputi2015SMF(redshift=4) )
        data.append( grazian2015SMF(redshift=4.5) )

    symbols = ['D','o','p','s','x']
    colors = ['#bbbbbb','#888888','#555555','#222222','#000000']

    for i, d in enumerate(data):
        ll = d['lowerLimits'] if 'lowerLimits' in d else False
        if 'errorUp' in d:
            l,_,_ = ax.errorbar(d['stellarMass'], d['numDens'], yerr=[d['errorUp'],d['errorDown']],
                         color=colors[i], ecolor=colors[i], alpha=0.9, capsize=0.0, fmt=symbols[i], lolims=ll)
        if 'error' in d:
            l,_,_ = ax.errorbar(d['stellarMass'], d['numDens'], yerr=d['error'], 
                         color=colors[i], ecolor=colors[i], alpha=0.9, capsize=0.0, fmt=symbols[i], lolims=ll)

        lines.append(l)

    if not clean:
        legend1 = ax.legend(lines, [d['label'] for d in data],loc='upper right')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('SMF: '+sP.simName)
        sP.setRedshift(simRedshift)

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], fieldsSubhalos=mts)

        # centrals only
        if centralsOnly:
            wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
            w = gc['halos']['GroupFirstSub'][wHalo]
        else:
            w = np.arange(gc['subhalos']['count'])

            if cutClumps:
                extraFieldsHalo = ['GroupPos','Group_R_Crit200']
                extraFields = ['SubhaloMass', 'SubhaloPos', 'SubhaloGrNr', 'SubhaloLenType']
                gc2 = groupCat(sP, fieldsHalos=extraFieldsHalo, fieldsSubhalos=extraFields)

                massFracStars = gc['subhalos']['SubhaloMassType'][:,sP.ptNum('stars')] / \
                                gc2['subhalos']['SubhaloMass']

                numDM = gc2['subhalos']['SubhaloLenType'][:,sP.ptNum('dm')]

                isPrimary = np.zeros( gc['subhalos']['count'], dtype='int32' )
                ww = np.where( gc['halos']['GroupFirstSub'] >= 0 )
                isPrimary[ gc['halos']['GroupFirstSub'][ww] ] = 1

                parentPos = np.zeros( (gc['subhalos']['count'],3), dtype='float32' )
                for i in range(3):
                    parentPos[:,i] = gc2['halos']['GroupPos'][ gc2['subhalos']['SubhaloGrNr'],i ]
                parentR200 = gc2['halos']['Group_R_Crit200'][ gc2['subhalos']['SubhaloGrNr'] ]

                ww = np.where(parentR200 == 0.0)
                parentR200[ww] = 1e-10 # make finite and small s.t. normalized dist is outside cut

                radialDist = periodicDists( gc2['subhalos']['SubhaloPos'], parentPos, sP)
                radialDistNormedByParentR200 = radialDist / parentR200

                wExclude = np.where( (numDM < 10) & \
                                     (massFracStars > 1 - 1e-3) & \
                                     (isPrimary == 0) & \
                                     (radialDistNormedByParentR200 < 0.2) )

                print(' cut clumps (exclude %d of %d)' % (wExclude[0].size, len(w)))

                mask = np.zeros( gc['subhalos']['count'], dtype='int32' )
                mask[wExclude] = 1
                w = np.where(mask == 0)

        # for each of the three stellar mass definitions, calculate SMF and plot
        count = 0

        for mt in mts:
            if not highMassEnd and mt != 'SubhaloMassInRadType':
                continue

            # temporary Mstar selection
            if use30kpc:
                field = 'Subhalo_Mass_30pkpc_Stars'
                ac = auxCat(sP, fields=[field])
                xx = sP.units.codeMassToLogMsun(ac[field][w])
            if use30H:
                field = 'Subhalo_Mass_min_30pkpc_2rhalf_Stars'
                ac = auxCat(sP, fields=[field])
                xx = sP.units.codeMassToLogMsun(ac[field][w])
            if useP10:
                field = 'Subhalo_Mass_puchwein10_Stars'
                ac = auxCat(sP, fields=[field])
                xx = sP.units.codeMassToLogMsun(ac[field][w])
            if not use30kpc and not useP10 and not use30H:
                xx = gc['subhalos'][mt][w,sP.ptNum('stars')]
                xx = sP.units.codeMassToLogMsun(xx)

            normFac = sP.boxSizeCubicComovingMpc * binSize
            xm, ym_i = running_histogram(xx, binSize=binSize, normFac=normFac, skipZeros=True)
            ym = savgol_filter(ym_i,sKn,sKo)

            label = sP.simName+' z=%d'%sP.redshift if count == 0 else ''
            color = l.get_color() if count > 0 else None
            l, = ax.plot(xm[1:], ym[1:], linestyles[count], color=color, lw=3.0, label=label)

            count += 1

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    if highMassEnd:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[2]),
                  plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[0]),
                  plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[1])]
        lExtra = [r'$M_\star^{\rm tot}$',
                  r'$M_\star^{< 2r_{\star,1/2}}$', 
                  r'$M_\star^{< r_{\star,1/2}}$']
    else:
        sExtra = []
        lExtra = []

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower left')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def stellarMassFunctionMultiPanel(sPs, pdf, highMassEnd=False, centralsOnly=False, use30kpc=False, 
                                  use30H=False, useP10=False, redshifts=[0,1,2,3]):
    """ Stellar mass function (number density of galaxies) in 2x2 panels at 4 different redshifts, 
    matching simulation and data redshifts. """
    nRows = np.floor(np.sqrt(len(redshifts)))
    nCols = np.ceil(len(redshifts) / nRows)

    sizefac_loc = 0.9
    figsize_loc = [figsize[0]*nCols*sizefac_loc, figsize[1]*nRows*sizefac_loc]

    fig = plt.figure(figsize=figsize_loc)

    for i, redshift in enumerate(redshifts):
        # append each panel to existing figure, which is automatically saved into pdf at the end
        ind = int(nRows*100 + nCols*10 + (i+1))
        stellarMassFunction(sPs, pdf, highMassEnd=highMassEnd, centralsOnly=centralsOnly, 
                            use30kpc=use30kpc, use30H=use30H, useP10=useP10, 
                            simRedshift=redshift, dataRedshift=redshift, fig_subplot=[fig,ind])

def massMetallicityStars(sPs, pdf, simRedshift=0.0, fig_subplot=[None,None]):
    """ Stellar mass-metallicity relation at z=0. """
    # config
    acMetalFields = ['Subhalo_StellarZ_4pkpc_rBandLumWt']
    metalFields   = ['SubhaloStarMetallicityHalfRad',
                     'SubhaloStarMetallicity',
                     'SubhaloStarMetallicityMaxRad']
    if clean: metalFields = []

    minNumStars = 1
    if clean: minNumStars = 100 # log(Mstar) ~= 8.2 (1820) or 9.1 (2500)

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])
    
    ax.set_xlim([8.0, 12.5])
    ax.set_ylim([-1.5,1.0])

    xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
    ylabel = 'Z$_{\\rm stars}$ [ log Z$_{\\rm sun}$ ]'
    if not clean:
        xlabel += ' [ < 2r$_{1/2}$ ]'
        ylabel += ' [ centrals and satellites ]'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # observational points
    g = gallazzi2005(sPs[0])
    w = woo2008(sPs[0])
    k = kirby2013()

    l1, = ax.plot(g['stellarMass'], g['Zstars'], '-', color='#333333',lw=2.0,alpha=0.7)
    ax.fill_between(g['stellarMass'], g['ZstarsDown'], g['ZstarsUp'], 
                    color='#333333', interpolate=True, alpha=0.3)

    l2,_,_ = ax.errorbar(w['stellarMass'], w['Zstars'], yerr=w['ZstarsErr'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')

    l3,_,_ = ax.errorbar(k['stellarMass'], k['Zstars'], 
                         xerr=[k['stellarMassErr'],k['stellarMassErr']], yerr=[k['ZstarsErr'],k['ZstarsErr']],
                         color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='o')

    legend1 = ax.legend([l1,l2,l3], [g['label'],w['label'],k['label']], loc='lower right')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print('MMStars: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        # load
        c = ax._get_lines.prop_cycler.next()['color']

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMass','SubhaloMassInRadType']+metalFields)
        ac = auxCat(sP, fields=acMetalFields)

        # include: centrals + satellites (no noticeable difference vs. centrals only)
        # stellar mass definition, enforce resolution limit
        xx_code = gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')]

        w = np.where( xx_code >= minNumStars * sP.targetGasMass )
        xx = sP.units.codeMassToLogMsun( xx_code[w] )

        # metallicities based on auxCat calculation?
        for i, acMetalField in enumerate(acMetalFields):

            iters = [0]
            #if clean: iters = [0,1] # disabled

            for i_num in iters:
                yy = logZeroSafe( ac[acMetalField][w] / sP.units.Z_solar )

                if i_num == 1:
                    # apply and plot Guidi+ (2016) correction from Z(Lum-W_fibre) to Z(OBS)
                    yy = (yy - 0.23) / 0.69

                # only include subhalos with non-nan entries (e.g. at least 1 real star within radial cut)
                ww = np.where(np.isfinite(yy))
                yy_loc = yy[ww]
                xx_loc = xx[ww]

                xm, ym_i, sm_i, pm_i = running_median(xx_loc,yy_loc,binSize=binSize,
                                                      skipZeros=True,percs=[10,25,75,90])
                ym = savgol_filter(ym_i,sKn,sKo)
                sm = savgol_filter(sm_i,sKn,sKo)
                pm = savgol_filter(pm_i,sKn,sKo,axis=1) # P[10,90]

                label = sP.simName if (i == 0 and i_num == 0) else ''
                ax.plot(xm[:-1], ym[:-1], linestyles[i+i_num*2], color=c, lw=3.0, label=label)

                ax.fill_between(xm[:-1], pm[0,:-1], pm[-1,:-1], color=c, interpolate=True, alpha=0.25)

        # metallicities from groupcat, measured within what radius?                    
        for i, metalField in enumerate(metalFields):
            # note: Vogelsberger+ (2014a) scales the simulation values by Z_solar=0.02 instead of 
            # correcting the observational Gallazzi/... points, resulting in the vertical shift 
            # with respect to this plot (sim,Gal,Woo all shift up, but I think Kirby is good as is)
            yy = logZeroSafe( gc['subhalos'][metalField][w] / sP.units.Z_solar )

            xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,skipZeros=True,percs=[10,25,75,90])
            ym2 = savgol_filter(ym,sKn,sKo)
            pm2 = savgol_filter(pm,sKn,sKo,axis=1) # P[10,90]

            ax.plot(xm[1:-1], ym2[1:-1], linestyles[i+len(acMetalFields)], color=c, lw=3.0)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []

    if not clean:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[0]),
                  plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[1]),
                  plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[2]),
                  plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[3])]
        lExtra = ['Z$_{\\rm stars}$ (r < 4pkpc, rBand-LumWt)',
                  'Z$_{\\rm stars}$ (r < 1r$_{1/2})$', 
                  'Z$_{\\rm stars}$ (r < 2r$_{1/2})$',
                  'Z$_{\\rm stars}$ (r < r$_{\\rm max})$']
    else:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[2])]
        lExtra = ['Guidi+ (2016) Correction'] 

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def massMetallicityGas(sPs, pdf, simRedshift=0.0):
    """ Gas mass-metallicity relation at z=0. 
        (Torrey 2013 Figure 10) (Schaye Figure 13)"""
    from util import simParams

    # config
    metalFields = ['SubhaloGasMetallicitySfrWeighted','SubhaloGasMetallicitySfr']

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    if simRedshift == 0.0:
        ax.set_xlim([8.0, 11.5])
        ax.set_ylim([-1.0,1.0])
    if simRedshift == 0.7:
        ax.set_xlim([7.5, 11.5])
        ax.set_ylim([-1.0,0.75])
    
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
    ax.set_ylabel('Z$_{\\rm gas}$ [ log Z$_{\\rm sun}$ ] [ centrals and satellites ]')

    # observational points
    if simRedshift == 0.0:
        z12a = zahid2012(pp04=False, redshift=0)
        z12b = zahid2012(pp04=True, redshift=0)
        z14a = zahid2014(pp04=False, redshift=0.08)
        z14b = zahid2014(pp04=True, redshift=0.08)
        t04  = tremonti2004()

        l1,_,_ = ax.errorbar(z12a['stellarMass'], z12a['Zgas'], yerr=z12a['Zgas_err'],
                             color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='D')

        l2,_,_ = ax.errorbar(z12b['stellarMass'], z12b['Zgas'], yerr=z12b['Zgas_err'],
                             color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='s')

        #l3,_,_ = ax.errorbar(t04['stellarMass'], t04['Zgas'], yerr=[t04['Zgas_errDown'],t04['Zgas_errUp']],
        #                     color='#bbbbbb', ecolor='#bbbbbb', alpha=0.9, capsize=0.0, fmt='o')
        l3, = ax.plot(t04['stellarMass'], t04['Zgas'], ':', color='#bbbbbb', alpha=0.9)
        ax.fill_between(t04['stellarMass'], t04['Zgas_Down'], t04['Zgas_Up'], 
                        color='#bbbbbb', interpolate=True, alpha=0.2)

        l4, = ax.plot(z14a['stellarMass'], z14a['Zgas'], '-', color='#999999', lw=2.0, alpha=0.9)
        l5, = ax.plot(z14a['stellarMass'], z14b['Zgas'], '--', color='#999999', lw=2.0, alpha=0.9)

        labels  = [z12a['label'],z12b['label'],t04['label'],z14a['label'],z14b['label']]
        legend1 = ax.legend([l1,l2,l3,l4,l5], labels, loc='lower right')
        ax.add_artist(legend1)

    if simRedshift == 0.7:
        g16a = guo2016(O3O2=False)
        g16b = guo2016(O3O2=True)
        z12a = zahid2012(pp04=False, redshift=1)
        z12b = zahid2012(pp04=True, redshift=1)
        z14a = zahid2014(pp04=False, redshift=0.78)
        z14b = zahid2014(pp04=True, redshift=0.78)

        l1, = ax.plot(g16a['stellarMass'], g16a['Zgas'], '-', color='#666666', lw=2.0, alpha=0.9)
        l2, = ax.plot(g16b['stellarMass'], g16b['Zgas'], '--', color='#666666', lw=2.0, alpha=0.9)

        l3,_,_ = ax.errorbar(z12a['stellarMass'], z12a['Zgas'], yerr=z12a['Zgas_err'],
                             color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')

        l4,_,_ = ax.errorbar(z12b['stellarMass'], z12b['Zgas'], yerr=z12b['Zgas_err'],
                             color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='s')

        l5, = ax.plot(z14a['stellarMass'], z14a['Zgas'], '-', color='#bbbbbb', lw=2.0, alpha=0.9)
        l6, = ax.plot(z14a['stellarMass'], z14b['Zgas'], '--', color='#bbbbbb', lw=2.0, alpha=0.9)

        labels  = [g16a['label'],g16b['label'],z12a['label'],z12b['label'],z14a['label'],z14b['label']]
        legend1 = ax.legend([l1,l2,l3,l4,l5,l6], labels, loc='lower right')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('MMGas (z=%3.1f): %s' % (simRedshift,sP.simName))
        sP.setRedshift(simRedshift)

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMassInRadType']+metalFields)

        # include: centrals + satellites (no noticeable difference vs. centrals only)
        w = np.arange(gc['subhalos']['count'])
        #w = np.where(gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')] > 0.0)[0]

        # stellar mass definition
        xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]
        xx = sP.units.codeMassToLogMsun( xx_code )

        # metallicity measured how/within what radius?
        c = ax._get_lines.prop_cycler.next()['color']
                
        for i, metalField in enumerate(metalFields):
            # only subhalos with nonzero metalField (some star-forming gas)
            wNz = np.where( gc['subhalos'][metalField][w] > 0.0 )

            # log (Z_gas/Z_solar)
            yy = logZeroSafe( gc['subhalos'][metalField][w][wNz] / sP.units.Z_solar )

            xm, ym, sm = running_median(xx[wNz],yy,binSize=binSize)
            ym2 = savgol_filter(ym,sKn,sKo)
            sm2 = savgol_filter(sm,sKn,sKo)

            label = sP.simName + ' z=%3.1f' % simRedshift if i==0 else ''
            ax.plot(xm[:-1], ym2[:-1], linestyles[i], color=c, lw=3.0, label=label)

            if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0:
                ax.fill_between(xm[:-1], ym2[:-1]-sm2[:-1], ym2[:-1]+sm2[:-1], 
                color=c, interpolate=True, alpha=0.3)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[0]),
              plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[1])]
    lExtra = ['Z$_{\\rm gas}$ (sfr>0 sfr-weighted)',
              'Z$_{\\rm gas}$ (sfr>0 mass-weighted)']

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def baryonicFractionsR500Crit(sPs, pdf, simRedshift=0.0):
    """ Gas, star, and total baryonic fractions within r500_crit (for massive systems).
        (Genel Fig 10) (Schaye Fig 15) """
    # config
    markers   = ['o','D','s']  # gas, stars, baryons
    fracTypes = ['gas','stars','baryons']

    acField = 'Group_Mass_Crit500_Type'

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([11.0, 15.0])
    ax.set_ylim([0,0.25])
    ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ < r$_{\\rm 500c}$ ]')
    ax.set_ylabel('Gas/Star/Baryon Fraction [ M / M$_{\\rm 500c}$ ]')

    # observational points
    g = giodini2009(sPs[0])
    l = lovisari2015(sPs[0])

    l1,_,_ = ax.errorbar(g['m500_logMsun'], g['fGas500'], yerr=g['fGas500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[0]+linestyles[0])
    l2,_,_ = ax.errorbar(g['m500_logMsun'], g['fStars500'], yerr=g['fStars500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[1]+linestyles[1])
    l3,_,_ = ax.errorbar(g['m500_logMsun'], g['fBaryon500'], yerr=g['fBaryon500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[2]+linestyles[2])
    l4,_,_ = ax.errorbar(l['m500_logMsun'], l['fGas500'], #yerr=l['fGas500Err'],
                         color='#555555', ecolor='#555555', alpha=0.9, capsize=0.0, 
                         marker=markers[0],linestyle='')

    legend1 = ax.legend([l1,l2,l3,l4], [g['label']+' f$_{\\rm gas}$',
                                        g['label']+' f$_{\\rm stars}$',
                                        g['label']+' f$_{\\rm baryons}$',
                                        l['label']+' f$_{\\rm gas}$'], loc='upper left')
    ax.add_artist(legend1)

    # universal baryon fraction line
    OmegaU = sPs[0].omega_b / sPs[0].omega_m
    ax.plot( [11.0,15.0], [OmegaU,OmegaU], ':', lw=1.0, color='#444444', alpha=0.2)
    ax.text( 12.5, OmegaU+0.003, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', size='large', alpha=0.2)

    # loop over each fullbox run
    for sP in sPs:
        print('Fracs500Crit: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            ac = auxCat(sP, fields=[field])
            ac[field] = ac[field][sP.zoomSubhaloID,:]

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum( ac[field] )
            xx = sP.units.codeMassToLogMsun( xx_code )

            for i, fracType in enumerate(fracTypes):
                if fracType == 'gas':
                    val = ac[field][1]
                if fracType == 'stars':
                    val = ac[field][2]
                if fracType == 'baryons':
                    val = ac[field][1] + ac[field][2]

                yy = val / xx_code # fraction with respect to total
                ax.plot(xx,yy,markers[i],color=colors[0])
        else:
            ac = auxCat(sP, fields=[acField]) #, searchExists=True)
            #if ac[acField] is None:
            #    print(' SKIP missing')
            #    continue
            data = ac[acField]

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum( data, axis=1 )

            # handle NaNs
            ww = np.isnan(xx_code)
            xx_code[ww] = 1e-10
            xx_code[xx_code == 0.0] = 1e-10
            data[ww,0] = 1e-10
            data[ww,1:2] = 0.0

            xx = sP.units.codeMassToLogMsun( xx_code )

            # metallicity measured within what radius?
            c = ax._get_lines.prop_cycler.next()['color']
                    
            for i, fracType in enumerate(fracTypes):
                if fracType == 'gas':
                    val = data[:,1]
                if fracType == 'stars':
                    val = data[:,2]
                if fracType == 'baryons':
                    val = data[:,1] + data[:,2]

                yy = val / xx_code # fraction with respect to total

                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)

                label = sP.simName if i==0 else ''
                ax.plot(xm[:], ym2[:], linestyles[i], color=c, lw=3.0, label=label)

                #if fracType == 'gas':
                #    ax.fill_between(xm[:-1], ym2[:-1]-sm[:-1], ym2[:-1]+sm[:-1], 
                #                    color=c, interpolate=True, alpha=0.3)

    # zoom legend
    # setup the 'iClusters' line with 3 different symbols simultaneously
    # todo: http://nbviewer.jupyter.org/gist/leejjoon/5603703
    #p0 = plt.Line2D( (0,1),(0,0),color=colors[0],lw=3.0,marker=markers[0],linestyle='')
    #p1 = plt.Line2D( (0,1),(0,0),color=colors[0],lw=3.0,marker=markers[1],linestyle='')
    #p2 = plt.Line2D( (0,1),(0,1),color=colors[0],lw=3.0,marker=markers[2],linestyle='')
    #sExtra = [(p0,p1,p2)]
    #lExtra = ['iClusters']

    # f_labels legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = ['f$_{\\rm '+t+'}$' for t in fracTypes]

    legend3 = ax.legend(sExtra, lExtra, loc='lower right')
    ax.add_artist(legend3)

    # sim legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,alpha=0.0,marker='')]
    lExtra = ['[ sims z=%3.1f ]' % simRedshift]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def nHIcddf(sPs, pdf, moment=0, simRedshift=3.0):
    """ CDDF (column density distribution function) of neutral (atomic) hydrogen in the whole box.
        (Vog 14a Fig 4) """
    from util import simParams

    # config
    speciesList = ['nHI_noH2','nHI'] #,'nHI2','nHI3']

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([17.0, 23.0])
    ax.set_xlabel('log N$_{\\rm HI}$ [ cm$^{-2}$ ]')

    if moment == 0:
        ax.set_ylim([-27, -18])
        ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm HI}$)  [ cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-4, 0])
        ax.set_ylabel('CDDF (1$^{\\rm st}$ moment):  log N$_{\\rm HI}$ f(N$_{\\rm HI}$)')

    # observational points
    z13 = zafar2013()
    n12 = noterdaeme2012()
    n09 = noterdaeme2009()
    k13 = kim2013cddf()
    p10 = prochaska10cddf()

    if moment == 1:
        z13['log_fHI'] = np.log10( 10.0**z13['log_fHI'] * 10.0**z13['log_NHI'] )
        n12['log_fHI'] = np.log10( 10.0**n12['log_fHI'] * 10.0**n12['log_NHI'] )
        n09['log_fHI'] = np.log10( 10.0**n09['log_fHI'] * 10.0**n09['log_NHI'] )
        k13['log_fHI'] = np.log10( 10.0**k13['log_fHI'] * 10.0**k13['log_NHI'] )
        p10['log_fHI_lower'] = np.log10( 10.0**p10['log_fHI_lower'] * 10.0**p10['log_NHI'] )
        p10['log_fHI_upper'] = np.log10( 10.0**p10['log_fHI_upper'] * 10.0**p10['log_NHI'] )

    l1,_,_ = ax.errorbar(z13['log_NHI'], z13['log_fHI'], yerr=[z13['log_fHI_errDown'],z13['log_fHI_errUp']],
               xerr=z13['log_NHI_xerr'], color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')
    l2,_,_ = ax.errorbar(n12['log_NHI'], n12['log_fHI'], yerr=n12['log_fHI_err'], xerr=n12['log_NHI_xerr'], 
                         color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='s')
    l3,_,_ = ax.errorbar(n09['log_NHI'], n09['log_fHI'], yerr=n09['log_fHI_err'], xerr=n12['log_NHI_xerr'], 
                         color='#cccccc', ecolor='#cccccc', alpha=0.9, capsize=0.0, fmt='o')
    l4,_,_ = ax.errorbar(k13['log_NHI'], k13['log_fHI'], yerr=[k13['log_fHI_errDown'],k13['log_fHI_errUp']],
                         color='#444444', ecolor='#444444', alpha=0.9, capsize=0.0, fmt='D')

    l5 = ax.fill_between(p10['log_NHI'], p10['log_fHI_lower'], p10['log_fHI_upper'], 
                    color='#dddddd', interpolate=True, alpha=0.3)

    labels = [z13['label'],n12['label'],n09['label'],k13['label'],p10['label']]
    legend1 = ax.legend([l1,l2,l3,l4,l5], labels, loc='lower left')
    ax.add_artist(legend1)

    # colDens definitions, plot vertical dotted lines [cm^-2] at dividing points
    limitDLA = 20.3
    ax.plot( [limitDLA,limitDLA], ax.get_ylim(), '--', color='#dddddd', alpha=0.5 )

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('CDDF HI: '+sP.simName)
        sP.setRedshift(simRedshift)

        c = ax._get_lines.prop_cycler.next()['color']

        # once including H2 modeling, once without
        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            ac = auxCat(sP, fields=['Box_CDDF_'+species])

            n_HI  = ac['Box_CDDF_'+species][0,:]
            fN_HI = ac['Box_CDDF_'+species][1,:]

            # plot
            xx = np.log10(n_HI)

            if moment == 0:
                yy = logZeroSafe(fN_HI, zeroVal=np.nan)
            if moment == 1:
                yy = logZeroSafe(fN_HI*n_HI, zeroVal=np.nan)

            label = sP.simName if i == 0 else ''
            ax.plot(xx, yy, '-', lw=3.0, linestyle=linestyles[i], color=c, label=label)

    # variations legend
    #legend3 = ax.legend(sExtra, lExtra, loc='lower right')
    #ax.add_artist(legend3)

    # sim legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=0.0,alpha=0.0,marker='')]
    lExtra = ['[ sims z=%3.1f ]' % simRedshift]

    sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra += [str(s) for s in speciesList]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def nOVIcddf(sPs, pdf, moment=0, simRedshift=0.2):
    """ CDDF (column density distribution function) of O VI in the whole box.
        (Schaye Fig 17) (Suresh+ 2016 Fig 11) """
    from util import simParams

    # config
    speciesList = ['nOVI','nOVI_solar','nOVI_10','nOVI_25']

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([12.5, 15.5])
    ax.set_xlabel('log N$_{\\rm OVI}$ [ cm$^{-2}$ ]')

    if moment == 0:
        ax.set_ylim([-17, -11])
        ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm OVI}$)  [ cm$^{2}$ ]')
        if clean:
            ax.set_ylabel('log f(N$_{\\rm OVI}$) [ cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-0.5, 1.5])
        ax.set_ylabel('CDDF (1$^{\\rm st}$ moment):  log N$_{\\rm OVI}$ f(N$_{\\rm OVI}$)')

    # observational points
    d16  = danforth2016()
    d08  = danforth2008()
    tc08 = thomChen2008()
    t08  = tripp2008()

    if moment == 1:
        d16['log_fOVI']  = np.log10( 10.0**d16['log_fOVI'] * 10.0**d16['log_NOVI'] )
        d08['log_fOVI']  = np.log10( 10.0**d08['log_fOVI'] * 10.0**d08['log_NOVI'] )
        tc08['log_fOVI'] = np.log10( 10.0**tc08['log_fOVI'] * 10.0**tc08['log_NOVI'] )
        t08['log_fOVI']  = np.log10( 10.0**t08['log_fOVI'] * 10.0**t08['log_NOVI'] )

    l1,_,_ = ax.errorbar(d16['log_NOVI'], d16['log_fOVI'], yerr=[d16['log_fOVI_errDown'],d16['log_fOVI_errUp']],
               xerr=d16['log_NOVI_err'], color='#555555', ecolor='#555555', alpha=0.9, capsize=0.0, fmt='s')

    l2,_,_ = ax.errorbar(d08['log_NOVI'], d08['log_fOVI'], yerr=[d08['log_fOVI_errDown'],d08['log_fOVI_errUp']],
               xerr=d08['log_NOVI_err'], color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')

    l3,_,_ =  ax.errorbar(tc08['log_NOVI'], tc08['log_fOVI'], yerr=tc08['log_fOVI_err'],
               xerr=[tc08['log_NOVI_errLeft'],tc08['log_NOVI_errRight']], 
               color='#cccccc', ecolor='#cccccc', alpha=0.9, capsize=0.0, fmt='s')

    l4,_,_ =  ax.errorbar(t08['log_NOVI'], t08['log_fOVI'], yerr=t08['log_fOVI_err'],
               xerr=t08['log_NOVI_err'], color='#aaaaaa', ecolor='#aaaaaa', alpha=0.9, capsize=0.0, fmt='o')

    if not clean:
        labels = [d16['label'],d08['label'],tc08['label'],t08['label']]
        legend1 = ax.legend([l1,l2,l3,l4], labels, loc='lower left')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('CDDF OVI: '+sP.simName)
        sP.setRedshift(simRedshift)

        c = ax._get_lines.prop_cycler.next()['color']

        # pre-computed CDDF: first species for sizes
        ac = auxCat(sP, fields=['Box_CDDF_'+speciesList[0]])
        n_OVI  = ac['Box_CDDF_'+speciesList[0]][0,:]
        fN_OVI = ac['Box_CDDF_'+speciesList[0]][1,:]

        # pre-computed CDDF: allocate for max/min bounds of our variations
        fN_OVI_min = fN_OVI * 0.0 + np.inf
        fN_OVI_max = fN_OVI * 0.0

        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            ac = auxCat(sP, fields=['Box_CDDF_'+species])

            assert np.array_equal(ac['Box_CDDF_'+species][0,:], n_OVI) # require same x-pts
            fN_OVI = ac['Box_CDDF_'+species][1,:]

            fN_OVI_min = np.nanmin(np.vstack( (fN_OVI_min, fN_OVI) ), axis=0)
            fN_OVI_max = np.nanmax(np.vstack( (fN_OVI_max, fN_OVI) ), axis=0)

        # plot 'uncertainty' band
        xx = np.log10(n_OVI)

        if moment == 0:
            yy_min = logZeroSafe(fN_OVI_min, zeroVal=np.nan)
            yy_max = logZeroSafe(fN_OVI_max, zeroVal=np.nan)
            yy = logZeroSafe( 0.5*(fN_OVI_min+fN_OVI_max), zeroVal=np.nan )
        if moment == 1:
            yy_min = logZeroSafe(fN_OVI_min*n_OVI, zeroVal=np.nan)
            yy_max = logZeroSafe(fN_OVI_max*n_OVI, zeroVal=np.nan)
            yy = logZeroSafe( 0.5*(fN_OVI_min*n_OVI+fN_OVI_max*n_OVI), zeroVal=np.nan )

        ax.fill_between(xx, yy_min, yy_max, color=c, alpha=0.2, interpolate=True)

        # plot middle line
        label = sP.simName
        ax.plot(xx, yy, '-', lw=3.0, color=c, label=label)

    # legend
    sExtra = [] #[plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = [] #[str(s) for s in speciesList]

    if not clean:
        sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,alpha=0.0,marker='')]
        lExtra += ['[ sims z=%3.1f ]' % simRedshift]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def dlaMetallicityPDF(sPs, pdf, simRedshift=3.0):
    """ PDF of log of DLA (nHI>20.3) metallicities in the whole box colDens grid.
        (Vog 14a Fig 4) """
    from util import simParams

    # config
    speciesList = ['nHI_noH2','nHI']
    log_nHI_limitDLA = 20.3
    log_Z_nBins = 50
    log_Z_range = [-3.0, 0.0]

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim(log_Z_range)
    ax.set_xlabel('log ( Z / Z$_{\\rm solar}$ )')
    ax.set_ylim([0.0,1.2])
    ax.set_ylabel('PDF of DLA Metallicities')

    # observational points
    sPs[0].setRedshift(simRedshift)
    r12 = rafelski2012(sPs[0])

    l1,_,_ = ax.errorbar(r12['log_Z'], r12['pdf'], yerr=r12['pdf_err'], xerr=r12['log_Z_err'], 
               color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='D')

    legend1 = ax.legend([l1], [r12['label']], loc='upper right')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue
        else:
            print('DLA Z PDF: '+sP.simName)
            sP.setRedshift(simRedshift)

            c = ax._get_lines.prop_cycler.next()['color']

            # once including H2 modeling, once without
            for i, species in enumerate(speciesList):
                # load pre-computed Z PDF
                ac = auxCat(sP, fields=['Box_Grid_'+species,'Box_Grid_Z'])
                ww = np.where(ac['Box_Grid_'+species] > log_nHI_limitDLA)

                #ac = auxCat(sP, fields=['Box_CDDF_'+species])
                #n_HI  = ac['Box_CDDF_'+species][0,:]
                #fN_HI = ac['Box_CDDF_'+species][1,:]

                # plot (xx in log(Z/Zsolar) already)
                yy, xx = np.histogram(ac['Box_Grid_Z'][ww], bins=log_Z_nBins, range=log_Z_range, density=True)

                xx = xx[:-1] + 0.5*(log_Z_range[1]-log_Z_range[0])/log_Z_nBins

                label = sP.simName+' z=%3.1f' % sP.redshift if i == 0 else ''
                ax.plot(xx, yy, '-', lw=3.0, linestyle=linestyles[i], color=c, label=label)

    # second legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = [str(s) for s in speciesList]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def velocityFunction(sPs, pdf, centralsOnly=True, simRedshift=0.0):
    """ Velocity function (galaxy counts as a function of v_circ/v_max). """
    binSizeLogKms = 0.03

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([50,350])
    ax.set_ylim([1e-3,3e-1])
    ax.set_xlabel('v$_{\\rm circ}$ [ km/s ] [ sim = v$_{\\rm max}$ ]')

    if centralsOnly:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ only centrals ]')
    else:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ centrals & satellites ]')
    ax.set_yscale('log')

    # observational points
    b16 = bekeraite16VF()

    l1,_,_ = ax.errorbar(b16['v_circ'], b16['numDens'], yerr=b16['numDens_err'], 
               color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='s')

    legend1 = ax.legend([l1], [b16['label']], loc='upper right')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print('VF: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloVmax'])

        # centrals only?
        if centralsOnly:
            wHalo = np.where((gc['halos'] >= 0))
            w = gc['halos'][wHalo]
        else:
            w = np.arange(gc['subhalos'].size)

        # histogram in log(v) and plot in linear(v)
        xx = np.log10(gc['subhalos'][w])
        normFac = sP.boxSizeCubicPhysicalMpc*binSizeLogKms

        xm_i, ym_i = running_histogram(xx, binSize=binSizeLogKms, normFac=normFac, skipZeros=True)
        xm = 10.0**xm_i
        ym = savgol_filter(ym_i,sKn,sKo)

        l, = ax.plot(xm[1:-1], ym[1:-1], linestyles[0], lw=3.0, label=sP.simName)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles, labels, loc='lower left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def stellarAges(sPs, pdf, centralsOnly=False, simRedshift=0.0, fig_subplot=[None,None]):
    """ Luminosity or mass weighted stellar ages, as a function of Mstar (Vog 14b Fig 25). """
    ageTypes = ['Subhalo_StellarAge_4pkpc_rBandLumWt',
                'Subhalo_StellarAge_NoRadCut_MassWt',
                'Subhalo_StellarAge_NoRadCut_rBandLumWt']
    if clean: ageTypes = [ageTypes[0]]

    minNumStars = 1
    if clean: minNumStars = 100 # log(Mstar) ~= 8.2 (1820) or 9.1 (2500)

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])
    
    ax.set_xlim([8.0,12.5])
    ax.set_ylim([0,14])

    xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
    if not clean: xlabel += ' [ < 2r$_{\star,1/2}$ ]'
    ax.set_xlabel(xlabel)

    cenSatStr = ' [ centrals only ]' if centralsOnly else ' [ centrals & satellites ]'
    if clean: cenSatStr = ''
    ax.set_ylabel('Stellar Age [ Gyr ]%s' % cenSatStr)

    # observational points
    g05 = gallazzi2005(sPs[0])
    b10 = bernardi10()

    l1, = ax.plot(g05['stellarMass'], g05['ageStars'], '-', color='#333333',lw=2.0,alpha=0.7)
    ax.fill_between(g05['stellarMass'], g05['ageStarsDown'], g05['ageStarsUp'], 
                    color='#333333', interpolate=True, alpha=0.2)

    l2, = ax.plot(b10['stellarMass'], b10['ageStars'], '-', color='#777777',lw=2.0,alpha=0.7)
    ax.fill_between(b10['stellarMass'], b10['ageStarsDown'], b10['ageStarsUp'], 
                    color='#333333', interpolate=True, alpha=0.1)

    legend1 = ax.legend([l1,l2], [g05['label'],b10['label']], loc='upper left')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print('AGES: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        # load
        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
        ac = auxCat(sP, fields=ageTypes)

        # include: centrals + satellites, or centrals only
        if centralsOnly:
            gcLoad = groupCat(sP, fieldsHalos=['GroupFirstSub'])
            wHalo = np.where((gcLoad['halos'] >= 0))
            w = gcLoad['halos'][wHalo]
        else:
            w = np.arange(gc['subhalos'].shape[0])

        # stellar mass definition, and enforce resolution limit
        xx_code = gc['subhalos'][:,sP.ptNum('stars')]

        wResLimit = np.where( xx_code >= minNumStars * sP.targetGasMass )[0]
        w = np.intersect1d(w,wResLimit)

        xx = sP.units.codeMassToLogMsun( xx_code[w] )

        # loop through ages measured through different techniques
        c = ax._get_lines.prop_cycler.next()['color']
                
        for i, ageType in enumerate(ageTypes):

            iters = [0]
            #if clean: iters = [0,1] # disabled

            for i_num in iters:
                yy = ac[ageType][w]

                if i_num == 1: # apply Guidi+ (2016) correction from Age(Lum-W_fibre) to Age(OBS)
                    yy = (yy - 2.37) / 0.97

                # only include subhalos with non-nan age entries (e.g. at least 1 real star within radial cut)
                ww = np.where(np.isfinite(yy))
                yy_loc = yy[ww]
                xx_loc = xx[ww]

                xm, ym_i, sm_i, pm_i = running_median(xx_loc,yy_loc,binSize=binSize,percs=[10,25,75,90])

                ym = savgol_filter(ym_i,sKn,sKo)
                sm = savgol_filter(sm_i,sKn,sKo)
                pm = savgol_filter(pm_i,sKn,sKo,axis=1)

                label = sP.simName if (i == 0 and i_num == 0) else ''
                ax.plot(xm[:-1], ym[:-1], linestyles[i+2*i_num], color=c, lw=3.0, label=label)

                if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0: # P[10,90]
                    ax.fill_between(xm[:-1], pm[0,:-1], pm[-1,:-1], color=c, interpolate=True, alpha=0.25)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []

    if not clean:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[i]) \
                    for i,ageType in enumerate(ageTypes)]
        lExtra = [', '.join(ageType.split("_")[2:]) for ageType in ageTypes]
    else:
        pass
        #sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[2])]
        #lExtra = ['Guidi+ (2016) Correction'] 

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower right')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def haloXrayLum(sPs, pdf, centralsOnly=True, use30kpc=True, simRedshift=0.0, fig_subplot=[None,None]):
    """ X-ray bolometric luminosity scaling relation vs halo mass (e.g. Schaye Fig 16). """
    lumTypes = ['Subhalo_XrayBolLum','Group_XrayBolLum_Crit500'] #,'Subhalo_XrayBolLum_2rhalfstars']
    if clean: lumTypes = [lumTypes[0]]

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])
    
    ax.set_xlim([10.0,12.0])
    ax.set_ylim([38,45])

    xlabel = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
    ylabel = 'L$_{\\rm X}$ Bolometric [ log erg/s ]'
    cenSatStr = ''

    if not clean:
        if use30kpc: xlabel += ' [ < 30 pkpc ]'
        if not use30kpc: xlabel += ' [ < 2r$_{\star,1/2}$ ]'
        if centralsOnly: ylabel += ' [ centrals only ]'
        if not centralsOnly: ylabel += ' [ centrals & satellites ]'

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # observational points
    a15 = anderson2015(sPs[0])

    l1,_,_ = ax.errorbar(a15['stellarMass'], a15['xray_LumBol'],xerr=a15['stellarMass_err'],
                         yerr=[a15['xray_LumBol_errDown'],a15['xray_LumBol_errUp']],
                         color='#666666', ecolor='#666666', alpha=0.9, capsize=0.0, fmt='D')

    legend1 = ax.legend([l1], [a15['label']], loc='upper left')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print('XRAY: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            continue

        # load
        gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
        ac = auxCat(sP, fields=lumTypes)

        # include: centrals + satellites, or centrals only
        if centralsOnly:
            gcLoad = groupCat(sP, fieldsHalos=['GroupFirstSub'])
            wHalo = np.where((gcLoad['halos'] >= 0))
            w = gcLoad['halos'][wHalo]
        else:
            w = np.arange(gc['subhalos'].shape[0])

        # stellar mass definition, and enforce resolution limit
        xx_code = gc['subhalos'][:,sP.ptNum('stars')]

        if use30kpc:
            # load auxcat
            field = 'Subhalo_Mass_30pkpc_Stars'
            xx_code = auxCat(sP, fields=[field])[field]

        xx = sP.units.codeMassToLogMsun( xx_code[w] )

        # loop through ages measured through different techniques
        c = ax._get_lines.prop_cycler.next()['color']
                
        for i, lumType in enumerate(lumTypes):
            # auxCat values are per subhalo or per group?
            if 'Group_' in lumType:
                yy = ac[lumType][wHalo]
            else:
                yy = ac[lumType][w]

            # unit conversion: [10^-30 erg/s] -> [log erg/s]
            yy = logZeroSafe(yy.astype('float64') * 1e30)

            # only include subhalos with non-nan Lum entries
            ww = np.where(np.isfinite(xx) & np.isfinite(yy))
            yy_loc = yy[ww]
            xx_loc = xx[ww]

            xm, ym_i, sm_i, pm_i = running_median(xx_loc,yy_loc,binSize=binSize,percs=[10,25,75,90])

            ym = savgol_filter(ym_i,sKn,sKo)
            sm = savgol_filter(sm_i,sKn,sKo)
            pm = savgol_filter(pm_i,sKn,sKo,axis=1)

            label = sP.simName if (i == 0) else ''
            ax.plot(xm[:-1], ym[:-1], linestyles[i], color=c, lw=3.0, label=label)

            if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0: # P[10,90]
                ax.fill_between(xm[:-1], pm[0,:-1], pm[-1,:-1], color=c, interpolate=True, alpha=0.25)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []

    if not clean and len(lumTypes) > 1:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[i]) \
                    for i,lumType in enumerate(lumTypes)]
        lExtra = [' '.join(lumType.split("_")) for lumType in lumTypes]

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower right')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def plots():
    """ Plot portfolio of global population comparisons between runs. """
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    from util import simParams

    sPs = []
    # add runs: zooms
    #sPs.append( simParams(res=3, run='iClusters', variant='TNG_11', hInd=1) )
    #sPs.append( simParams(res=3, run='iClusters', variant='TNG_11', hInd=2) )
    #sPs.append( simParams(res=3, run='iClusters', variant='TNG_11', hInd=3) )
    #sPs.append( simParams(res=3, run='iClusters', variant='TNG_11', hInd=4) )
    #sPs.append( simParams(res=2, run='iClusters', variant='TNG_11', hInd=1) )

    # add runs: fullboxes
    sPs.append( simParams(res=1820, run='tng') )
    #sPs.append( simParams(res=910, run='tng') )
    #sPs.append( simParams(res=455, run='tng') )

    #sPs.append( simParams(res=1820, run='illustris') )
    #sPs.append( simParams(res=910, run='illustris') )
    #sPs.append( simParams(res=455, run='illustris') )

    #sPs.append( simParams(res=512, run='tng') )
    #sPs.append( simParams(res=256, run='tng') )
    #sPs.append( simParams(res=256, run='tng', variant='wmap') )

    #for i in range(1,11):
    #    sPs.append( simParams(res=256, run='tng', variant='r%03d' % i) )

    #sPs.append( simParams(res=2500, run='tng') )
    #sPs.append( simParams(res=1250, run='tng') )
    #sPs.append( simParams(res=625, run='tng') )  

    #sPs.append( simParams(res=2160, run='tng') )  
    #sPs.append( simParams(res=1080, run='tng') )  
    #sPs.append( simParams(res=540, run='tng') )  
    #sPs.append( simParams(res=270, run='tng') )

    # add runs: TNG_methods
    #sPs.append( simParams(res=512, run='tng', variant=0000) )
    #sPs.append( simParams(res=256, run='tng', variant=0000) )
    #sPs.append( simParams(res=256, run='tng', variant=4601) )
    #sPs.append( simParams(res=256, run='tng', variant=4602) )

    # make multipage PDF
    pdf = PdfPages('globalComps_sizes_z4_%s.pdf' % (datetime.now().strftime('%d-%m-%Y')))

    zZero = 4.0 # change to plot simulations at z>0 against z=0 observational data

    # TEST AREA
    galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=zZero, addHalfLightRad=['p07c_cf00dust_res_conv_efr','sdss_r',False])
    galaxySizes(sPs, pdf, vsHaloMass=True, simRedshift=zZero, addHalfLightRad=['p07c_cf00dust_res_conv_efr','sdss_r',False])
    pdf.close()
    return
    # END TEST AREA

    stellarMassHaloMass(sPs, pdf, ylog=False, use30kpc=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=True, use30kpc=True, simRedshift=zZero)

    stellarMassHaloMassMultiPanel(sPs, pdf, ylog=False, use30kpc=True)
    stellarMassHaloMassMultiPanel(sPs, pdf, ylog=True, use30kpc=True)

    sfrAvgVsRedshift(sPs, pdf)
    sfrdVsRedshift(sPs, pdf, xlog=True)
    sfrdVsRedshift(sPs, pdf, xlog=False)
    blackholeVsStellarMass(sPs, pdf, vsBulgeMass=True, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, twiceR=True, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, vsHaloMass=True, simRedshift=zZero)
    galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=zZero, addHalfLightRad=None)
    galaxySizes(sPs, pdf, vsHaloMass=True, simRedshift=zZero, addHalfLightRad=['p07c_cf00dust_efr','sdss_r',False])
    galaxySizes(sPs, pdf, vsHaloMass=True, simRedshift=zZero, addHalfLightRad=None)
    stellarMassFunction(sPs, pdf, highMassEnd=False, use30kpc=True, simRedshift=zZero)
    stellarMassFunction(sPs, pdf, highMassEnd=True, simRedshift=zZero)
    stellarMassFunctionMultiPanel(sPs, pdf, use30kpc=True, highMassEnd=False, redshifts=[3,4])

    massMetallicityStars(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=0.7)
    baryonicFractionsR500Crit(sPs, pdf, simRedshift=zZero)

    if 0:
        nHIcddf(sPs, pdf) # z=3
        nHIcddf(sPs, pdf, moment=1)
        nOVIcddf(sPs, pdf) # z=0.2
        nOVIcddf(sPs, pdf, moment=1)
        dlaMetallicityPDF(sPs, pdf) # z=3

    if 1:
        cheapDustModel = 'p07c_cf00dust_rad30pkpc' #'p07c_cf00dust_res_conv_ns1_rad30pkpc' is very expensive to run
        galaxyColorPDF(sPs, pdf, bands=['u','i'], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
        galaxyColorPDF(sPs, pdf, bands=['g','r'], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
        galaxyColorPDF(sPs, pdf, bands=['r','i'], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
        galaxyColorPDF(sPs, pdf, bands=['i','z'], splitCenSat=False, simRedshift=zZero, simColorsModels=[cheapDustModel])
        galaxyColor2DPDFs(sPs, pdf, simRedshift=zZero, simColorsModel=cheapDustModel)

    velocityFunction(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=True, simRedshift=zZero)
    haloXrayLum(sPs, pdf, centralsOnly=True, use30kpc=True, simRedshift=zZero)

    # todo: Vmax vs Mstar (tully-fisher) (Torrey Fig 9) (Vog 14b Fig 23) (Schaye Fig 12)
    # todo: Mbaryon vs Mstar (baryonic tully-fisher) (Vog 14b Fig 23)
    # todo: SFR main sequence (Schaye Fig 11)
    # todo: active/passive fraction vs Mstar (Schaye Fig 11) (or red/blue Vog Fig ?)
    # todo: SFRD decomposed into contribution by halo mass bin (Genel Fig ?)

    # with additional modeling:
    # todo: M_HI vs Mstar (Vog 14a Fig 3), HI-vs-halo mass relation
    # todo: R_HI vs Mstar
    # todo: other metal CDDFs (e.g. Schaye Fig 17) (Bird 2016 Fig 6 Carbon) (HI z=0.1 Gurvich2016)
    # todo: Omega_X(z) (e.g. Bird? Fig ?)
    # todo: B/T distributions in Mstar bins, early/late fraction vs Mstar (kinematic)
    # todo: other cluster observables: SZ, radio

    pdf.close()
