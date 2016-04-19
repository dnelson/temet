"""
globalComp.py
  run summary plots and comparisons to constraints
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from util.loadExtern import *
from util.helper import running_median, running_histogram, logZeroSafe
from illustris_python.util import partTypeNum
from cosmo.load import groupCat, groupCatSingle, auxCat
from cosmo.util import addRedshiftAgeAxes, validSnapList

# global configuration
sKn     = 5   # savgol smoothing kernel length (1=disabled)
sKo     = 3   # savgol smoothing kernel poly order
binSize = 0.2 # dex in stellar/halo mass for median lines

linestyles = ['-',':','--','-.']       # typically for analysis variations per run
colors     = ['blue','purple','black'] # colors for zoom markers only (cannot vary linestyle with 1 point)

def stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=False, simRedshift=0.0):
    """ Stellar mass / halo mass relation, full boxes and zoom points vs. abundance matching lines. """
    # plot setup
    xrange = [10.0, 15.0]
    yrange = [0.0, 0.30]

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    if ylog:
        ax.set_yscale('log')
        ax.set_ylim([1e-3,1e0])

    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
    ax.set_ylabel('M$_\star$ / M$_{\\rm halo}$ $(\Omega_{\\rm b} / \Omega_{\\rm m})^{-1}$ [ only centrals ]')

    # abundance matching constraints
    b = behrooziSMHM(sPs[0])
    m = mosterSMHM(sPs[0])
    k = kravtsovSMHM(sPs[0])

    ax.plot(b['haloMass_i'], b['y_mid_i'], color='#333333', label='Behroozi+ (2013)')
    ax.fill_between(b['haloMass_i'], b['y_low_i'], b['y_high_i'], color='#333333', interpolate=True, alpha=0.3)

    ax.plot(m['haloMass'], m['y_mid'], color='#dddddd', label='Moster+ (2013)')
    ax.fill_between(m['haloMass'], m['y_low'], m['y_high'], color='#dddddd', interpolate=True, alpha=0.3)

    ax.plot(k['haloMass'], k['y_mid'], color='#888888', label='Kravtsov+ (2014)')

    # first legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color=colors[0], lw=3.0, marker='', linestyle=linestyles[0]),
              plt.Line2D( (0,1), (0,0), color=colors[1], lw=3.0, marker='', linestyle=linestyles[1]),
              plt.Line2D( (0,1), (0,0), color=colors[2], lw=3.0, marker='', linestyle=linestyles[2])]
    lExtra = [r'$M_\star^{\rm tot}$',
              r'$M_\star^{< 2r_{1/2}}$', 
              r'$M_\star^{< r_{1/2}}$']

    legend1 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')
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

            if allMassTypes:
                yy = gc['subhalos']['SubhaloMassType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], color=c, lw=3.0)

            yy = gc['subhalos']['SubhaloMassInRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
            xm, ym, sm = running_median(xx,yy,binSize=binSize)
            ym2 = savgol_filter(ym,sKn,sKo)
            l, = ax.plot(xm[:-1], ym2[:-1], linestyles[1], lw=3.0, color=c, label=sP.simName)
            lines.append(l)

            if allMassTypes:
                yy = gc['subhalos']['SubhaloMassInHalfRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[2], lw=3.0, color=c)

    # second legend
    markers = []; sExtra = []; lExtra = [];
    for handle in lines:
        sExtra.append( handle )
        lExtra.append( handle.get_label() )

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
    maxNumSnaps = 20

    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_ylim([8e-3, 5e2])
    addRedshiftAgeAxes(ax, sPs[0])
    ax.set_ylabel('<SFR> [ M$_{\\rm sun}$ / yr ] [ < 2r$_{1/2}$ ] [ only centrals ]')
    ax.set_yscale('log')    

    # abundance matching constraints
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
        snaps, nSnaps = validSnapList(sP, maxNum=maxNumSnaps)

        redshifts = np.zeros(nSnaps)
        sfrs = np.zeros( (nSnaps, len(plotMassBins), 3), dtype='float32' )
        stds = np.zeros( (nSnaps, len(plotMassBins), 3), dtype='float32' )

        # loop over all snapshots
        for j, snap in enumerate(snaps):
            sP.setSnap(snap)

            gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], 
              fieldsSubhalos=['SubhaloSFR','SubhaloSFRinRad','SubhaloSFRinHalfRad'])

            if not gc['halos']['count']:
                continue # high redshift

            redshifts[j] = sP.redshift

            # centrals only, given halo mass definition, in this halo mass bin
            for k, haloMassBin in enumerate(plotMassBins):
                haloMassesLogMsun = sP.units.codeMassToLogMsun( gc['halos']['Group_M_Crit200'] )

                wHalo = np.where( (gc['halos']['GroupFirstSub'] >= 0) & \
                                  (haloMassesLogMsun > haloMassBin) & \
                                  (haloMassesLogMsun <= haloMassBin+b['haloBinSize']) )
                w = gc['halos']['GroupFirstSub'][wHalo]

                # sfr definition(s)
                sfrs[j,k,0] = np.median( gc['subhalos']['SubhaloSFR'][w] )
                stds[j,k,0] = np.std( gc['subhalos']['SubhaloSFR'][w] )

                sfrs[j,k,1] = np.median( gc['subhalos']['SubhaloSFRinRad'][w] )
                stds[j,k,1] = np.std( gc['subhalos']['SubhaloSFRinRad'][w] )

                sfrs[j,k,2] = np.median( gc['subhalos']['SubhaloSFRinHalfRad'][w] )
                stds[j,k,2] = np.std( gc['subhalos']['SubhaloSFRinHalfRad'][w] )

        # plot line for each halo mass bin
        c = ax._get_lines.prop_cycler.next()['color']

        for k, haloMassBin in enumerate(plotMassBins):
            # different sfr definitions
            for j in [1]: # <2r1/2
                label = sP.simName if (k==0 and j==1) else ''

                ax.plot(redshifts, sfrs[:,k,j], '-', color=c, marker='o', lw=3.0, label=label)
                #ax.errorbar(redshifts, sfrs[:,k,j], stds[:,k,j], ecolor=c, alpha=1.0, capthick='', fmt='none')

    # legend
    ax.legend(loc='upper left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def sfrdVsRedshift(sPs, pdf, xlog=True, simRedshift=0.0):
    """ Star formation rate density of the universe, vs redshift, vs observational points. """
    # plot setup
    fig = plt.figure(figsize=(16,9))
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
        sP.setRedshift(simRedshift)

        s = sfrTxt(sP)

        ax.plot(s['redshift'], s['sfrd'], '-', lw=2.5, label=sP.simName)

    # second legeng
    legend2 = ax.legend(loc='lower left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def blackholeVsStellarMass(sPs, pdf, simRedshift=0.0):
    """ Black hole mass vs. stellar (bulge) mass relation at z=0. """
    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_xlim([8.5, 13.0])
    ax.set_ylim([5.5, 11.0])
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 1r$_{1/2}$ ]')
    ax.set_ylabel('Black Hole Mass [ log M$_{\\rm sun}$ ] [ only centrals ]')

    # observational points
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

        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMassType','SubhaloMassInHalfRadType'])

        # centrals only
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        # stellar mass definition: would want to mimic bulge mass measurements
        xx_code = gc['subhalos']['SubhaloMassInHalfRadType'][w,partTypeNum('stars')]
        xx = sP.units.codeMassToLogMsun( xx_code )

        # stellar mass definition(s)
        yy = sP.units.codeMassToLogMsun( gc['subhalos']['SubhaloMassType'][w,partTypeNum('bhs')] )

        xm, ym, sm = running_median(xx,yy,binSize=binSize)
        ym2 = savgol_filter(ym,sKn,sKo)
        l, = ax.plot(xm[:-1], ym2[:-1], '-', lw=3.0, label=sP.simName)

    # second legeng
    legend2 = ax.legend(loc='lower right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=0.0):
    """ Galaxy sizes (half mass radii) vs stellar mass or halo mass, at redshift zero. """
    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_ylim([0.3,2e2])
    ax.set_ylabel('Galaxy Size [ kpc ] [ r$_{\\rm 1/2, stars/gas}$ ] [ only centrals ]')
    ax.set_yscale('log')

    if vsHaloMass:
        ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([10,14.5])   
    else:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
        #ax.set_xlim( behrooziSMHM(sPs[0], logHaloMass=np.array(ax.get_xlim())) )
        ax.set_xlim([7,12.5])

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
                            loc='upper left')
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

        xm, ym_gas, sm_gas     = running_median(xx,yy_gas,binSize=binSize)
        xm, ym_stars, sm_stars = running_median(xx,yy_stars,binSize=binSize)

        ym_gas   = savgol_filter(ym_gas,sKn,sKo)
        ym_stars = savgol_filter(ym_stars,sKn,sKo)
        sm_gas   = savgol_filter(sm_gas,sKn,sKo)
        sm_stars = savgol_filter(sm_stars,sKn,sKo)

        l, = ax.plot(xm[:-1], ym_stars[:-1], linestyles[0], lw=3.0, label=sP.simName)
        l, = ax.plot(xm[:-1], ym_gas[:-1], linestyles[1], color=l.get_color(), lw=3.0)

        if sP.run == 'illustris':
            y_down = np.array(ym_stars[:-1]) - sm_stars[:-1]
            y_up   = np.array(ym_stars[:-1]) + sm_stars[:-1]
            ax.fill_between(xm[:-1], y_down, y_up, 
                            color=l.get_color(), interpolate=True, alpha=0.3)

    # second legeng
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[0]),
              plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[1])]
    lExtra = [r'stars',
              r'gas']

    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='lower right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def stellarMassFunction(sPs, pdf, highMassEnd=False, centralsOnly=False, simRedshift=0.0):
    """ Stellar mass function (number density of galaxies) at redshift zero. """
    # config
    mts = ['SubhaloMassInRadType','SubhaloMassInHalfRadType','SubhaloMassType']

    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_ylim([5e-6,2e-1])
    ax.set_xlim([7,12.5])

    if highMassEnd:
        #ax.set_ylim([1e-7,2e-2])
        #ax.set_xlim([10.0,12.5])
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 1r$_{1/2}$, < 2r$_{1/2}$, or total ]')
    else:
        #ax.set_ylim([5e-4,2e-1])
        #ax.set_xlim([7,11.5])
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')

    if centralsOnly:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ only centrals ]')
    else:
        ax.set_ylabel('$\Phi$ [ Mpc$^{-3}$ dex$^{-1}$ ] [ centrals & satellites ]')
    ax.set_yscale('log')

    # observational points
    b08 = baldry2008SMF()
    b12 = baldry2012SMF()
    b13 = bernardi2013SMF()

    l1,_,_ = ax.errorbar(b08['stellarMass'], b08['numDens'], yerr=[b08['errorDown'],b08['errorUp']],
                         color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')
    l2,_,_ = ax.errorbar(b12['stellarMass'], b12['numDens'], yerr=b12['error'],
                         color='#555555', ecolor='#555555', alpha=0.9, capsize=0.0, fmt='o')

    l3,_,_ = ax.errorbar(b13['SerExp']['stellarMass'], b13['SerExp']['numDens'], 
                         yerr=[b13['SerExp']['errorUp'],b13['SerExp']['errorDown']],
                         color='#333333', ecolor='#333333', alpha=0.9, capsize=0.0, fmt='s')

    legend1 = ax.legend([l1,l2,l3], [b08['label'], b12['label'], b13['SerExp']['label']], loc='upper right')
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

        # for each of the three stellar mass definitions, calculate SMF and plot
        count = 0

        for mt in mts:
            if not highMassEnd and mt != 'SubhaloMassInRadType':
                continue

            xx = gc['subhalos'][mt][w,partTypeNum('stars')]
            xx = sP.units.codeMassToLogMsun(xx)

            xm, ym = running_histogram(xx, binSize=binSize, normFac=sP.boxSizeCubicMpc*binSize)
            ym = savgol_filter(ym,sKn,sKo)

            label = sP.simName if count == 0 else ''
            color = l.get_color() if count > 0 else None
            l, = ax.plot(xm[:-1], ym[:-1], linestyles[count], color=color, lw=3.0, label=label)

            count += 1

    # second legeng
    handles, labels = ax.get_legend_handles_labels()
    if highMassEnd:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[2]),
                  plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[0]),
                  plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[1])]
        lExtra = [r'$M_\star^{\rm tot}$',
                  r'$M_\star^{< 2r_{1/2}}$', 
                  r'$M_\star^{< r_{1/2}}$']
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
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([-2.0,1.0])
    ax.set_xlabel('Stellar Mass [ log M$_{\\rm sun}$ ] [ < 2r$_{1/2}$ ]')
    ax.set_ylabel('Z$_{\\rm stars}$ [ log Z$_{\\rm sun}$ ] [ centrals and satellites ]')

    # observational points
    g = gallazzi2005(sPs[0])
    w = woo2008(sPs[0])
    k = kirby2013()

    l1, = ax.plot(g['stellarMass'], g['Zstars'], '-', color='#333333')
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
                yy = np.log10( gc['subhalos'][metalField][w] / sP.units.Z_solar )

                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)

                label = sP.simName if i==0 else ''
                ax.plot(xm[:-1], ym2[:-1], linestyles[i], color=c, lw=3.0, label=label)

    # second legeng
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
    fig = plt.figure(figsize=(16,9))
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
            yy = np.log10( gc['subhalos'][metalField][w][wNz] / sP.units.Z_solar )

            xm, ym, sm = running_median(xx[wNz],yy,binSize=binSize)
            ym2 = savgol_filter(ym,sKn,sKo)
            sm2 = savgol_filter(sm,sKn,sKo)

            label = sP.simName + ' z=%3.1f' % simRedshift if i==0 else ''
            ax.plot(xm[:-1], ym2[:-1], linestyles[i], color=c, lw=3.0, label=label)

            if i == 0:
                ax.fill_between(xm[:-1], ym2[:-1]-sm2[:-1], ym2[:-1]+sm2[:-1], 
                color=c, interpolate=True, alpha=0.3)

    # second legeng
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

    # plot setup
    fig = plt.figure(figsize=(16,9))
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
    ax.plot( [14.75,15.0], [OmegaU,OmegaU], '--', color='#444444', alpha=0.4)
    ax.text( 14.79, OmegaU+0.003, '$\Omega_{\\rm b} / \Omega_{\\rm m}$', size='large', alpha=0.4)

    # loop over each fullbox run
    for sP in sPs:
        print('MMStars: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.isZoom:
            ac = auxCat(sP, fields=['Group/Mass_Crit500_Type'])
            ac['Group/Mass_Crit500_Type'] = ac['Group/Mass_Crit500_Type'][sP.zoomSubhaloID,:]

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum( ac['Group/Mass_Crit500_Type'] )
            xx = sP.units.codeMassToLogMsun( xx_code )

            for i, fracType in enumerate(fracTypes):
                if fracType == 'gas':
                    val = ac['Group/Mass_Crit500_Type'][1]
                if fracType == 'stars':
                    val = ac['Group/Mass_Crit500_Type'][2]
                if fracType == 'baryons':
                    val = ac['Group/Mass_Crit500_Type'][1] + ac['Group/Mass_Crit500_Type'][2]

                yy = val / xx_code # fraction with respect to total
                ax.plot(xx,yy,markers[i],color=colors[0])
        else:
            ac = auxCat(sP, fields=['Group/Mass_Crit500_Type'])

            # halo mass definition (xx_code == gc['halos']['Group_M_Crit500'] by construction)
            xx_code = np.sum( ac['Group/Mass_Crit500_Type'], axis=1 )

            # handle NaNs
            ww = np.isnan(xx_code)
            xx_code[ww] = 1e-10
            xx_code[xx_code == 0.0] = 1e-10
            ac['Group/Mass_Crit500_Type'][ww,0] = 1e-10
            ac['Group/Mass_Crit500_Type'][ww,1:2] = 0.0

            xx = sP.units.codeMassToLogMsun( xx_code )

            # metallicity measured within what radius?
            c = ax._get_lines.prop_cycler.next()['color']
                    
            for i, fracType in enumerate(fracTypes):
                if fracType == 'gas':
                    val = ac['Group/Mass_Crit500_Type'][:,1]
                if fracType == 'stars':
                    val = ac['Group/Mass_Crit500_Type'][:,2]
                if fracType == 'baryons':
                    val = ac['Group/Mass_Crit500_Type'][:,1] + ac['Group/Mass_Crit500_Type'][:,2]

                yy = val / xx_code # fraction with respect to total

                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,sKn,sKo)

                label = sP.simName if i==0 else ''
                ax.plot(xm[:-1], ym2[:-1], linestyles[i], color=c, lw=3.0, label=label)

                #if fracType == 'gas':
                #    ax.fill_between(xm[:-1], ym2[:-1]-sm[:-1], ym2[:-1]+sm[:-1], 
                #                    color=c, interpolate=True, alpha=0.3)

    # second legend
    # setup the 'iClusters' line with 3 different symbols simultaneously
    # todo: http://nbviewer.jupyter.org/gist/leejjoon/5603703
    p0 = plt.Line2D( (0,1),(0,0),color=colors[0],lw=3.0,marker=markers[0],linestyle='')
    p1 = plt.Line2D( (0,1),(0,0),color=colors[0],lw=3.0,marker=markers[1],linestyle='')
    p2 = plt.Line2D( (0,1),(0,1),color=colors[0],lw=3.0,marker=markers[2],linestyle='')
    sExtra = [(p0,p1,p2)]
    lExtra = ['iClusters']

    # f_labels
    sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra += ['f$_{\\rm '+t+'}$' for t in fracTypes]

    # render
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right', numpoints=1)

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
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_xlim([17.0, 23.0])
    ax.set_xlabel('log N$_{\\rm HI}$ [ cm$^{-2}$ ]')

    if moment == 0:
        ax.set_ylim([-27, -18])
        ax.set_ylabel('CDDF$_{0}$:  log f(N$_{\\rm HI}$)  [ cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-4, 0])
        ax.set_ylabel('CDDF$_{1}$:  log N$_{\\rm HI}$ f(N$_{\\rm HI}$)')

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
            ac = auxCat(sP, fields=['Box/CDDF_'+species])

            n_HI  = ac['Box/CDDF_'+species][0,:]
            fN_HI = ac['Box/CDDF_'+species][1,:]

            # plot
            xx = np.log10(n_HI)

            if moment == 0:
                yy = logZeroSafe(fN_HI, zeroVal=np.nan)
            if moment == 1:
                yy = logZeroSafe(fN_HI*n_HI, zeroVal=np.nan)

            label = sP.simName+' z=%3.1f' % sP.redshift if i == 0 else ''
            ax.plot(xx, yy, '-', lw=3.0, linestyle=linestyles[i], color=c, label=label)

    # second legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = [str(s) for s in speciesList]

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
    speciesList = ['nOVI']

    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_xlim([12.5, 15.5])
    ax.set_xlabel('log N$_{\\rm OVI}$ [ cm$^{-2}$ ]')

    if moment == 0:
        ax.set_ylim([-18, -11])
        ax.set_ylabel('CDDF$_{0}$:  log f(N$_{\\rm OVI}$)  [ cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-4, 0]) # TODO
        ax.set_ylabel('CDDF$_{1}$:  log N$_{\\rm OVI}$ f(N$_{\\rm OVI}$)')

    # observational points
    d14 = danforth2014()

    #if moment == 1:
    #    d14['log_fOVI'] = np.log10( 10.0**d14['log_fOVI'] * 10.0**d14['log_NOVI'] )

    #l1,_,_ = ax.errorbar(d14['log_NOVI'], d14['log_fOVI'], yerr=[d14['log_fOVI_errDown'],d14['log_fOVI_errUp']],
    #           xerr=d14['log_NOVI_xerr'], color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='D')

    #labels = [d14['label']]
    #legend1 = ax.legend([l1], labels, loc='lower left')
    #ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('CDDF OVI: '+sP.simName)
        sP.setRedshift(simRedshift)

        c = ax._get_lines.prop_cycler.next()['color']

        # TODO: remove speciesList
        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            ac = auxCat(sP, fields=['Box/CDDF_'+species])# , reCalculate=True)

            n_OVI  = ac['Box/CDDF_'+species][0,:]
            fN_OVI = ac['Box/CDDF_'+species][1,:]

            print('n_OVI: ', n_OVI)
            print('f_OVI: ', fN_OVI)

            # plot
            xx = np.log10(n_OVI)

            if moment == 0:
                yy = logZeroSafe(fN_OVI, zeroVal=np.nan)
            if moment == 1:
                yy = logZeroSafe(fN_OVI*n_OVI, zeroVal=np.nan)

            label = sP.simName+' z=%3.1f' % sP.redshift if i == 0 else ''
            ax.plot(xx, yy, '-', lw=3.0, linestyle=linestyles[i], color=c, label=label)

    # second legend
    sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = [str(s) for s in speciesList]

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
    fig = plt.figure(figsize=(16,9))
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
                ac = auxCat(sP, fields=['Box/Grid_'+species,'Box/Grid_Z'])
                ww = np.where(ac['Box/Grid_'+species] > log_nHI_limitDLA)

                #ac = auxCat(sP, fields=['Box/CDDF_'+species])
                #n_HI  = ac['Box/CDDF_'+species][0,:]
                #fN_HI = ac['Box/CDDF_'+species][1,:]

                # plot (xx in log(Z/Zsolar) already)
                yy, xx = np.histogram(ac['Box/Grid_Z'][ww], bins=log_Z_nBins, range=log_Z_range, density=True)

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

def galaxyColorPDF(sPs, pdf, simRedshift=0.0):
    """ PDF of galaxy colors (by default: (u-i)), with no dust corrections. (Vog 14b Fig 13) """
    from util import simParams

    # config
    bands = ['u','i']
    stellarMassBins = ( [9.0,9.5],   [9.5,10.0],  [10.0,10.5], 
                        [10.5,11.0], [11.0,11.5], [11.5,12.0] )
    nRows = 2
    nCols = 3
    iLeg  = 4 # which panel to place simNames legend in

    # plot setup
    mag_range = [0.5,3.5]
    fig = plt.figure(figsize=(16*1.5,9*1.5))
    axes = []

    if bands[0] != 'u' or bands[1] != 'i':
        raise Exception('Not implemented')

    cind1 = 0 # U
    cind2 = 6 # i
    cFac  = 0.79 # U is in Vega, i is in AB, and U_AB = U_Vega + 0.79 
                 # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html

    # load observational points

    #select stellarMass,u_mag,g_mag,r_mag,i_mag,z_mag
    # from Guo2010a..MRII
    # where snapnum=63
    # and stellarMass between 10.0 and 20.0

    # loop over each mass bin
    for i, stellarMassBin in enumerate(stellarMassBins):

        # panel setup
        ax = fig.add_subplot(nRows,nCols,i+1)
        axes.append(ax)
        
        ax.set_xlim(mag_range)
        ax.set_xlabel('(%s-%s) color [ mag ]' % (bands[0],bands[1]))
        #ax.set_ylim([0.0,3.0])
        ax.set_ylabel('PDF [no dust corr, cen+sat]')

        # add stellar mass bin legend
        sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=0.0,marker='',linestyle=linestyles[0])]
        lExtra = ['%.1f < M$_{\\rm \star}$ < %.1f' % (stellarMassBin[0],stellarMassBin[1])]

        legend1 = ax.legend(sExtra, lExtra, loc='upper right')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('Color PDF: '+sP.simName)
        sP.setRedshift(simRedshift)

        c = ax._get_lines.prop_cycler.next()['color']

        # fullbox:
        gc = groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], 
                          fieldsSubhalos=['SubhaloMassInRadType','SubhaloStellarPhotometrics'])

        # galaxy selection
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0) & (gc['halos']['Group_M_Crit200'] > 0))
        w1 = gc['halos']['GroupFirstSub'][wHalo] # centrals only
        w2 = np.arange(gc['subhalos']['count']) # centrals + satellites
        w3 = np.array( list(set(w2) - set(w1)) ) # satellites only

        # selection:
        for j in range(1): #disabled, all only
            if j == 0: w = w2
            if j == 1: w = w1
            if j == 2: w = w3

            # galaxy mass definition and color
            stellar_mass = sP.units.codeMassToLogMsun( gc['subhalos']['SubhaloMassInRadType'][w,4] )
            galaxy_color = gc['subhalos']['SubhaloStellarPhotometrics'][w,cind1] - \
                           gc['subhalos']['SubhaloStellarPhotometrics'][w,cind2] + cFac
            
            # loop over each mass bin
            for i, stellarMassBin in enumerate(stellarMassBins):
                wBin = np.where((stellar_mass >= stellarMassBin[0]) & (stellar_mass < stellarMassBin[1]))

                nBins = np.max( [16, np.floor( np.sqrt( len(wBin[0] ))) * 1.2] ) # adaptive
                #print(sP.simName,i,nBins)

                yy, xx = np.histogram(galaxy_color[wBin], bins=nBins, range=mag_range, density=True)
                xx = xx[:-1] + 0.5*(mag_range[1]-mag_range[0])/nBins

                label = sP.simName if i == iLeg and j == 0 else ''
                alpha = 1.0 if j == 0 else 0.4
                l, = axes[i].plot(xx, yy, linestyles[j], color=c, label=label, alpha=alpha, lw=3.0)

    # legend
    #sExtra = [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    #lExtra = ['all galaxies','centrals only','satellites only']
    sExtra = []
    lExtra = []

    handles, labels = axes[iLeg].get_legend_handles_labels()
    legend2 = axes[iLeg].legend(handles+sExtra, labels+lExtra, loc='upper left')

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
    sPs.append( simParams(res=1820, run='illustris') )
    #sPs.append( simParams(res=512, run='L25n512_PRVS_0116') )
    #sPs.append( simParams(res=512, run='L25n512_PRVS_0311') )
    sPs.append( simParams(res=512, run='cosmo0_v6') )
    sPs.append( simParams(res=512, run='L25n512_PRVS_0404') )

    #sPs.append( simParams(res=270, run='realizations/L35n270_TNG_WMAP7') )
    #sPs.append( simParams(res=270, run='realizations/L35n270_TNG_PLANCK15') )

    #sPs.append( simParams(res=128, run='L12.5n256_discrete_dm0.0') )
    #sPs.append( simParams(res=128, run='L12.5n256_discrete_dm0.0001') )
    #sPs.append( simParams(res=128, run='L12.5n256_discrete_dm0.00001') )

    #sPs.append( simParams(res=1820, run='illustrisprime') )
    #sPs.append( simParams(res=910, run='illustris') )
    #sPs.append( simParams(res=455, run='illustris') )
    #sPs.append( simParams(res=455, run='illustris_prime') )
    #sPs.append( simParams(res=256, run='feedback') )

    # make multipage PDF
    pdf = PdfPages('globalComps_' + datetime.now().strftime('%d-%m-%Y')+'.pdf')

    stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True)
    stellarMassHaloMass(sPs, pdf, ylog=True, allMassTypes=True)
    sfrAvgVsRedshift(sPs, pdf)
    sfrdVsRedshift(sPs, pdf, xlog=True)
    sfrdVsRedshift(sPs, pdf, xlog=False)
    blackholeVsStellarMass(sPs, pdf)
    galaxySizes(sPs, pdf, vsHaloMass=False)
    galaxySizes(sPs, pdf, vsHaloMass=True)
    stellarMassFunction(sPs, pdf, highMassEnd=False)
    stellarMassFunction(sPs, pdf, highMassEnd=True)
    massMetallicityStars(sPs, pdf)
    massMetallicityGas(sPs, pdf, simRedshift=0.0)
    massMetallicityGas(sPs, pdf, simRedshift=0.7)
    baryonicFractionsR500Crit(sPs, pdf)
    nHIcddf(sPs, pdf)
    nHIcddf(sPs, pdf, moment=1)
    #nOVIcddf(sPs, pdf)
    #nOVIcddf(sPs, pdf, moment=1)
    dlaMetallicityPDF(sPs, pdf)
    galaxyColorPDF(sPs, pdf)

    # todo: stellar ages vs Mstar (Vog 14b Fig 25), luminosity or mass weighted?
    # todo: SMF 2x2 at z=0,1,2,3 (Torrey Fig 1)
    # todo: Vmax vs Mstar (tully-fisher) (Torrey Fig 9) (Vog 14b Fig 23) (Schaye Fig 12)
    # todo: Mbaryon vs Mstar (baryonic tully-fisher) (Vog 14b Fig 23)
    # todo: SFR main sequence (Schaye Fig 11)
    # todo: active/passive fraction vs Mstar (Schaye Fig 11)

    # with additional modeling:
    # todo: M_HI vs Mstar (Vog 14a Fig 3)
    # todo: R_HI vs Mstar
    # todo: metal CDDF (e.g. Schaye Fig 17) (Bird 2016 Fig 6 Carbon) (Suresh 2015 Fig 11 Oxygen)
    # todo: Omega_X(z) (e.g. Bird? Fig ?)
    # todo: B/T distributions in Mstar bins, early/late fraction vs Mstar (kinematic)
    # todo: X-ray (e.g. Schaye Fig 16)

    pdf.close()
