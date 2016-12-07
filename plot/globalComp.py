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
from illustris_python.util import partTypeNum
from cosmo.load import groupCat, groupCatSingle, auxCat
from cosmo.util import validSnapList, periodicDists
from plot.galaxyColor import galaxyColorPDF, galaxyColor2DPDFs
from plot.cosmoGeneral import addRedshiftAgeAxes

# global configuration
sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar mass/halo mass for median lines
figsize = (16,9) # (16,9) (8,6) (smf:10.6,7.0)
clean   = False # make visually clean plots with less information

linestyles = ['-',':','--','-.']       # typically for analysis variations per run
colors     = ['blue','purple','black'] # colors for zoom markers only (cannot vary linestyle with 1 point)

def stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=False, use30kpc=False, simRedshift=0.0):
    """ Stellar mass / halo mass relation, full boxes and zoom points vs. abundance matching lines. """
    # plot setup
    xrange = [10.0, 15.0]
    yrange = [0.0, 0.30]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    if ylog:
        ax.set_yscale('log')
        ax.set_ylim([1e-3,1e0])

    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
    ax.set_ylabel('M$_\star$ / M$_{\\rm halo}$ $(\Omega_{\\rm b} / \Omega_{\\rm m})^{-1}$ [ only centrals ]')

    # observational data: abundance matching constraints
    b = behrooziSMHM(sPs[0])
    m = mosterSMHM(sPs[0])
    k = kravtsovSMHM(sPs[0])

    ax.plot(b['haloMass_i'], b['y_mid_i'], color='#333333', label='Behroozi+ (2013)')
    ax.fill_between(b['haloMass_i'], b['y_low_i'], b['y_high_i'], color='#333333', interpolate=True, alpha=0.3)

    ax.plot(m['haloMass'], m['y_mid'], color='#dddddd', label='Moster+ (2013)')
    ax.fill_between(m['haloMass'], m['y_low'], m['y_high'], color='#dddddd', interpolate=True, alpha=0.3)

    ax.plot(k['haloMass'], k['y_mid'], color='#888888', label='Kravtsov+ (2014)')

    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend(handles, labels, loc='upper right')
    plt.gca().add_artist(legend1)

    # loop over each run
    lines = []

    for i, sP in enumerate(sPs):
        print('SMHM: '+sP.simName)
        sP.setRedshift(simRedshift)

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
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], color=c, lw=3.0, label=sP.simName)
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
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], lw=3.0, color=c, label=sP.simName)
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
        lExtra.append( sP.simName )
        markers.append( sP.marker )

    legend2 = ax.legend(sExtra, lExtra, loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

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

def blackholeVsStellarMass(sPs, pdf, twiceR=False, vsHaloMass=False, actualBHMasses=False, simRedshift=0.0):
    """ Black hole mass vs. stellar (bulge) mass relation at z=0. """
    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([8.5, 13.0])
    ax.set_ylim([5.5, 11.0])

    ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ tot, only centrals ]')
    if actualBHMasses:
        ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ actual, only centrals ]')

    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 1r$_{1/2}$ ]')
    if twiceR:
        ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
    if vsHaloMass:
        ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([9,14.5])
        ax.set_ylim([5.0, 11.0])

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

        print('BHMass: '+sP.simName)
        sP.setRedshift(simRedshift)

        fieldsSubhalos = ['SubhaloBHMass','SubhaloMassType']
        if not twiceR and not vsHaloMass: fieldsSubhalos.append('SubhaloMassInHalfRadType')
        if twiceR: fieldsSubhalos.append('SubhaloMassInRadType')

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], fieldsSubhalos=fieldsSubhalos)

        # centrals only
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        # stellar mass definition: would want to mimic bulge mass measurements
        if not twiceR and not vsHaloMass:
            xx_code = gc['subhalos']['SubhaloMassInHalfRadType'][w,partTypeNum('stars')]
        if twiceR:
            xx_code = gc['subhalos']['SubhaloMassInRadType'][w,partTypeNum('stars')]
        if vsHaloMass:
            xx_code = gc['halos']['Group_M_Crit200'][wHalo]

        xx = sP.units.codeMassToLogMsun( xx_code )

        # 'total' black hole mass in this subhalo, exclude those with no BHs
        # note: some subhalos (particularly the ~50=~1e-5 most massive) have N>1 BHs, then we here 
        # are effectively taking the sum of all their BH masses (better than mean, but max probably best)
        if actualBHMasses:
            # "actual" BH masses, excluding gas reservoir
            yy = gc['subhalos']['SubhaloBHMass'][w]
        else:
            # dynamical (particle masses)
            yy = gc['subhalos']['SubhaloMassType'][w,partTypeNum('bhs')]

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

def galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=0.0):
    """ Galaxy sizes (half mass radii) vs stellar mass or halo mass, at redshift zero. """
    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_ylim([0.3,2e2])
    ax.set_ylabel('Galaxy Size [ kpc ] [ r$_{\\rm 1/2, stars/gas}$ ] [ only centrals ]')
    ax.set_yscale('log')

    if vsHaloMass:
        ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([9,14.5])
        ax.set_ylim([1.0,8e2])
    else:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
        #ax.set_xlim( behrooziSMHM(sPs[0], logHaloMass=np.array(ax.get_xlim())) )
        ax.set_xlim([7,12.0])

    # observational points
    if not vsHaloMass:
        b = baldry2012SizeMass()
        s = shen2003SizeMass()

        l1,_,_ = ax.errorbar(b['red']['stellarMass'], b['red']['sizeKpc'], 
                             yerr=[b['red']['errorDown'],b['red']['errorUp']],
                             color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')
        l2,_,_ = ax.errorbar(b['blue']['stellarMass'], b['blue']['sizeKpc'], 
                             yerr=[b['blue']['errorDown'],b['blue']['errorUp']],
                             color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='o')

        l4, = ax.plot(s['late']['stellarMass'], s['late']['sizeKpc'], '-', color='#cccccc')
        ax.fill_between(s['late']['stellarMass'], s['late']['sizeKpcDown'], s['late']['sizeKpcUp'], 
                        color='#cccccc', interpolate=True, alpha=0.3)

        l3, = ax.plot(s['early']['stellarMass'], s['early']['sizeKpc'], '-', color='#aaaaaa')
        ax.fill_between(s['early']['stellarMass'], s['early']['sizeKpcDown'], s['early']['sizeKpcUp'], 
                        color='#aaaaaa', interpolate=True, alpha=0.3)

        legend1 = ax.legend([l1,l2,l3,l4], 
                            [b['red']['label'], b['blue']['label'], 
                             s['early']['label'], s['late']['label']], 
                            loc='lower right')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('Sizes: '+sP.simName)
        sP.setRedshift(simRedshift)

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMassInRadType','SubhaloHalfmassRadType'])

        # centrals only
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        # x-axis: mass definition
        if vsHaloMass:
            xx_code = gc['halos']['Group_M_Crit200'][wHalo]
        else:
            xx_code = gc['subhalos']['SubhaloMassInRadType'][w,partTypeNum('stars')]

        xx = sP.units.codeMassToLogMsun( xx_code )

        # stellar mass definition(s)
        yy_gas   = gc['subhalos']['SubhaloHalfmassRadType'][w,partTypeNum('gas')]
        yy_gas   = sP.units.codeLengthToKpc( yy_gas )
        yy_stars = gc['subhalos']['SubhaloHalfmassRadType'][w,partTypeNum('stars')]
        yy_stars = sP.units.codeLengthToKpc( yy_stars )

        # if plotting vs halo mass, restrict our attention to those galaxies with sizes (e.g. nonzero 
        # number of either gas cells or star particles)
        if vsHaloMass:
            ww = np.where( (yy_gas > 0.0) & (yy_stars > 0.0) )
            yy_gas = yy_gas[ww]
            yy_stars = yy_stars[ww]
            xx = xx[ww]

        xm_gas, ym_gas, sm_gas       = running_median(xx,yy_gas,binSize=binSize,skipZeros=True)
        xm_stars, ym_stars, sm_stars = running_median(xx,yy_stars,binSize=binSize,skipZeros=True)

        ww_gas   = np.where(ym_gas > 0.0)
        ww_stars = np.where(ym_stars > 0.0)

        ym_gas   = savgol_filter(ym_gas[ww_gas],sKn,sKo)
        ym_stars = savgol_filter(ym_stars[ww_stars],sKn,sKo)
        sm_gas   = savgol_filter(sm_gas[ww_gas],sKn,sKo)
        sm_stars = savgol_filter(sm_stars[ww_stars],sKn,sKo)

        xm_gas = xm_gas[ww_gas]
        xm_stars = xm_stars[ww_stars]

        l, = ax.plot(xm_stars[1:-1], ym_stars[1:-1], linestyles[0], lw=3.0, label=sP.simName)
        l, = ax.plot(xm_gas[1:-1], ym_gas[1:-1], linestyles[1], color=l.get_color(), lw=3.0)

        if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2):
            y_down = np.array(ym_stars[1:-1]) - sm_stars[1:-1]
            y_up   = np.array(ym_stars[1:-1]) + sm_stars[1:-1]
            ax.fill_between(xm_stars[1:-1], y_down, y_up, 
                            color=l.get_color(), interpolate=True, alpha=0.3)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[0]),
              plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[1])]
    lExtra = [r'stars',
              r'gas']

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def stellarMassFunction(sPs, pdf, highMassEnd=False, centralsOnly=False, 
                        use30kpc=False, use30H=False, useP10=False, simRedshift=0.0):
    """ Stellar mass function (number density of galaxies) at redshift zero. """
    # config
    mts = ['SubhaloMassInRadType','SubhaloMassInHalfRadType','SubhaloMassType']

    cutClumps = False

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_ylim([5e-6,2e-1])
    ax.set_xlim([7,12.5])
    if clean: ax.set_xlim([5,12.5])

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
    b08 = baldry2008SMF()
    b12 = baldry2012SMF()
    b13 = bernardi2013SMF()
    d15 = dsouza2015SMF()

    l1,_,_ = ax.errorbar(b08['stellarMass'], b08['numDens'], yerr=[b08['errorDown'],b08['errorUp']],
                         color='#bbbbbb', ecolor='#bbbbbb', alpha=0.9, capsize=0.0, fmt='D')
    l2,_,_ = ax.errorbar(b12['stellarMass'], b12['numDens'], yerr=b12['error'],
                         color='#888888', ecolor='#888888', alpha=0.9, capsize=0.0, fmt='o')
    l3,_,_ = ax.errorbar(b13['SerExp']['stellarMass'], b13['SerExp']['numDens'], 
                         yerr=[b13['SerExp']['errorUp'],b13['SerExp']['errorDown']],
                         color='#555555', ecolor='#555555', alpha=0.9, capsize=0.0, fmt='p')
    l4,_,_ = ax.errorbar(d15['stellarMass'], d15['numDens'], yerr=d15['error'],
                         color='#222222', ecolor='#222222', alpha=0.9, capsize=0.0, fmt='s')

    if not clean:
        legend1 = ax.legend([l1,l2,l3,l4], 
            [b08['label'], b12['label'], b13['SerExp']['label'], d15['label']], loc='upper right')
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
                xx = gc['subhalos'][mt][w,partTypeNum('stars')]
                xx = sP.units.codeMassToLogMsun(xx)

            normFac = sP.boxSizeCubicPhysicalMpc * binSize
            xm, ym_i = running_histogram(xx, binSize=binSize, normFac=normFac, skipZeros=True)
            ym = savgol_filter(ym_i,sKn,sKo)

            label = sP.simName if count == 0 else ''
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

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def massMetallicityStars(sPs, pdf, simRedshift=0.0):
    """ Stellar mass-metallicity relation at z=0. """
    # config
    metalFields = ['SubhaloStarMetallicityHalfRad','SubhaloStarMetallicity','SubhaloStarMetallicityMaxRad']

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([-2.0,1.0])
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
    ax.set_ylabel('Z$_{\\rm stars}$ [ log Z$_{\\rm sun}$ ] [ centrals and satellites ]')

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

    legend1 = ax.legend([l1,l2,l3], [g['label'],w['label'],k['label']], loc='upper left')
    ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        print('MMStars: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            gc = groupCatSingle(sP, subhaloID=sP.zoomSubhaloID)
            gh = groupCatSingle(sP, haloID=gc['SubhaloGrNr'])

            # stellar mass definition
            xx_code = gc['SubhaloMassInRadType'][partTypeNum('stars')]
            xx = sP.units.codeMassToLogMsun( xx_code )

            for i, metalField in enumerate(metalFields):
                # metallicity definition(s)
                yy = np.log10( gc[metalFields[i]] / sP.units.Z_solar )
                ax.plot(xx,yy,sP.marker,color=colors[i])

        else:
            gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
                fieldsSubhalos=['SubhaloMassInRadType']+metalFields)

            # include: centrals + satellites (no noticeable difference vs. centrals only)
            w = np.arange(gc['subhalos']['count'])
            #w = np.where(gc['subhalos']['SubhaloMassInRadType'][:,partTypeNum('stars')] > 0.0)[0]

            # stellar mass definition
            xx_code = gc['subhalos']['SubhaloMassInRadType'][w,partTypeNum('stars')]
            xx = sP.units.codeMassToLogMsun( xx_code )

            # metallicity measured within what radius?
            c = ax._get_lines.prop_cycler.next()['color']
                    
            for i, metalField in enumerate(metalFields):
                # note: Vogelsberger+ (2014a) scales the simulation values by Z_solar=0.02 instead of 
                # correcting the observational Gallazzi/... points, resulting in the vertical shift 
                # with respect to this plot (sim,Gal,Woo all shift up, but I think Kirby is good as is)
                yy = logZeroSafe( gc['subhalos'][metalField][w] / sP.units.Z_solar )

                xm, ym, sm = running_median(xx,yy,binSize=binSize,skipZeros=True)
                ym2 = savgol_filter(ym,sKn,sKo)

                label = sP.simName if i==0 else ''
                ax.plot(xm[1:-1], ym2[1:-1], linestyles[i], color=c, lw=3.0, label=label)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[0]),
              plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[1]),
              plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[2])]
    lExtra = ['Z$_{\\rm stars}$ (r < 1r$_{1/2})$', 
              'Z$_{\\rm stars}$ (r < 2r$_{1/2})$',
              'Z$_{\\rm stars}$ (r < r$_{\\rm max})$']

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower right')

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
        #w = np.where(gc['subhalos']['SubhaloMassInRadType'][:,partTypeNum('stars')] > 0.0)[0]

        # stellar mass definition
        xx_code = gc['subhalos']['SubhaloMassInRadType'][w,partTypeNum('stars')]
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

    field = 'Group_Mass_Crit500_Type'

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([11.0, 15.0])
    ax.set_ylim([0,0.25])
    ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ < r$_{\\rm 500c}$ ]')
    ax.set_ylabel('Gas/Star/Baryon Fraction [ M / M$_{\\rm 500c}$ ]')

    # observational points
    g = giodini2009(sPs[0])

    l1,_,_ = ax.errorbar(g['m500_logMsun'], g['fGas500'], yerr=g['fGas500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[0]+linestyles[0])
    l2,_,_ = ax.errorbar(g['m500_logMsun'], g['fStars500'], yerr=g['fStars500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[1]+linestyles[1])
    l3,_,_ = ax.errorbar(g['m500_logMsun'], g['fBaryon500'], yerr=g['fBaryon500Err'],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, 
                         fmt=markers[2]+linestyles[2])

    legend1 = ax.legend([l1,l2,l3], [g['label']+' f$_{\\rm gas}$',
                                     g['label']+' f$_{\\rm stars}$',
                                     g['label']+' f$_{\\rm baryons}$'], loc='upper left')
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
            ac = auxCat(sP, fields=[field])

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum( ac[field], axis=1 )

            # handle NaNs
            ww = np.isnan(xx_code)
            xx_code[ww] = 1e-10
            xx_code[xx_code == 0.0] = 1e-10
            ac[field][ww,0] = 1e-10
            ac[field][ww,1:2] = 0.0

            xx = sP.units.codeMassToLogMsun( xx_code )

            # metallicity measured within what radius?
            c = ax._get_lines.prop_cycler.next()['color']
                    
            for i, fracType in enumerate(fracTypes):
                if fracType == 'gas':
                    val = ac[field][:,1]
                if fracType == 'stars':
                    val = ac[field][:,2]
                if fracType == 'baryons':
                    val = ac[field][:,1] + ac[field][:,2]

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

def stellarAges(sPs, pdf, centralsOnly=False, simRedshift=0.0):
    """ Luminosity or mass weighted stellar ages, as a function of Mstar (Vog 14b Fig 25). """
    ageTypes = ['Subhalo_StellarAge_4pkpc_rBandLumWt',
                'Subhalo_StellarAge_NoRadCut_MassWt',
                'Subhalo_StellarAge_NoRadCut_rBandLumWt']
    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    ax.set_xlim([8.0,12.0])
    ax.set_ylim([0,14])
    ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{\star,1/2}$ ]')

    cenSatStr = '[ centrals only ]' if centralsOnly else '[ centrals & satellites ]'
    ax.set_ylabel('Stellar Age [ Gyr ] %s' % cenSatStr)

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

        # stellar mass definition
        xx_code = gc['subhalos'][w,partTypeNum('stars')]
        xx = sP.units.codeMassToLogMsun( xx_code )

        # loop through ages measured through different techniques
        c = ax._get_lines.prop_cycler.next()['color']
                
        for i, ageType in enumerate(ageTypes):
            yy = ac[ageType][w]

            # only include subhalos with non-nan age entries (e.g. at least 1 real star within radial cut)
            ww = np.where(np.isfinite(yy))
            yy_loc = yy[ww]
            xx_loc = xx[ww]

            xm, ym_i, sm_i = running_median(xx_loc,yy_loc,binSize=binSize)
            ym = savgol_filter(ym_i,sKn,sKo)
            sm = savgol_filter(sm_i,sKn,sKo)

            label = sP.simName if i == 0 else ''
            ax.plot(xm[:-1], ym[:-1], linestyles[i], color=c, lw=3.0, label=label)

            if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2) and i == 0:
                ax.fill_between(xm[:-1], ym[:-1]-sm[:-1], ym[:-1]+sm[:-1], 
                                color=c, interpolate=True, alpha=0.25)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', lw=3.0, marker='', linestyle=linestyles[i]) \
                for i,ageType in enumerate(ageTypes)]
    lExtra = [', '.join(ageType.split("_")[2:]) for ageType in ageTypes]

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower right')

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
    #sPs.append( simParams(res=1820, run='tng') )
    #sPs.append( simParams(res=910, run='tng') )
    #sPs.append( simParams(res=455, run='tng') )

    #sPs.append( simParams(res=1820, run='illustris') )
    #sPs.append( simParams(res=910, run='illustris') )
    #sPs.append( simParams(res=455, run='illustris') )

    #sPs.append( simParams(res=512, run='cosmo0_v6') )
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
    sPs.append( simParams(res=512, run='tng', variant=0000) )
    sPs.append( simParams(res=512, run='tng', variant=4401) )
    sPs.append( simParams(res=512, run='tng', variant=4402) )
    sPs.append( simParams(res=512, run='tng', variant=4501) )
    sPs.append( simParams(res=512, run='tng', variant=4502) )
    sPs.append( simParams(res=512, run='tng', variant=4503) )

    # make multipage PDF
    pdf = PdfPages('globalComps_440x_450x_' + datetime.now().strftime('%d-%m-%Y')+'.pdf')

    zZero = 0.0 # change to plot simulations at z>0 against z=0 observational data

    stellarMassHaloMass(sPs, pdf, ylog=False, use30kpc=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True, simRedshift=zZero)
    stellarMassHaloMass(sPs, pdf, ylog=True, use30kpc=True, simRedshift=zZero)
    sfrAvgVsRedshift(sPs, pdf)
    sfrdVsRedshift(sPs, pdf, xlog=True)
    sfrdVsRedshift(sPs, pdf, xlog=False)
    blackholeVsStellarMass(sPs, pdf, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, twiceR=True, simRedshift=zZero)
    blackholeVsStellarMass(sPs, pdf, vsHaloMass=True, actualBHMasses=True, simRedshift=zZero)
    galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=zZero)
    galaxySizes(sPs, pdf, vsHaloMass=True, simRedshift=zZero)
    stellarMassFunction(sPs, pdf, highMassEnd=False, use30kpc=True, simRedshift=zZero)
    stellarMassFunction(sPs, pdf, highMassEnd=True, simRedshift=zZero)
    massMetallicityStars(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=zZero)
    massMetallicityGas(sPs, pdf, simRedshift=0.7)
    baryonicFractionsR500Crit(sPs, pdf, simRedshift=zZero)

    nHIcddf(sPs, pdf) # z=3
    nHIcddf(sPs, pdf, moment=1)
    nOVIcddf(sPs, pdf) # z=0.2
    nOVIcddf(sPs, pdf, moment=1)
    dlaMetallicityPDF(sPs, pdf) # z=3

    galaxyColorPDF(sPs, pdf, bands=['u','i'], splitCenSat=False, simRedshift=zZero)
    ###galaxyColorPDF(sPs, pdf, bands=['u','i'], splitCenSat=True, simRedshift=zZero)
    galaxyColorPDF(sPs, pdf, bands=['g','r'], splitCenSat=False, simRedshift=zZero)
    galaxyColorPDF(sPs, pdf, bands=['r','i'], splitCenSat=False, simRedshift=zZero)
    galaxyColorPDF(sPs, pdf, bands=['i','z'], splitCenSat=False, simRedshift=zZero)
    galaxyColor2DPDFs(sPs, pdf, simRedshift=zZero)

    velocityFunction(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=False, simRedshift=zZero)
    stellarAges(sPs, pdf, centralsOnly=True, simRedshift=zZero)

    # todo: SMF 2x2 at z=0,1,2,3 (Torrey Fig 1)
    # todo: Vmax vs Mstar (tully-fisher) (Torrey Fig 9) (Vog 14b Fig 23) (Schaye Fig 12)
    # todo: Mbaryon vs Mstar (baryonic tully-fisher) (Vog 14b Fig 23)
    # todo: SFR main sequence (Schaye Fig 11) (sSFR vs Mstar colored by Sersic index, e.g. Wuyts)
    # todo: active/passive fraction vs Mstar (Schaye Fig 11) (or red/blue Vog Fig ?)
    # todo: SFRD decomposed into contribution by halo mass bin (Genel Fig ?)

    # with additional modeling:
    # todo: M_HI vs Mstar (Vog 14a Fig 3)
    # todo: R_HI vs Mstar
    # todo: other metal CDDFs (e.g. Schaye Fig 17) (Bird 2016 Fig 6 Carbon) (HI z=0.1 Gurvich2016)
    # todo: Omega_X(z) (e.g. Bird? Fig ?)
    # todo: B/T distributions in Mstar bins, early/late fraction vs Mstar (kinematic)
    # todo: X-ray (e.g. Schaye Fig 16)

    pdf.close()
