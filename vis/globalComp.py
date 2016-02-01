"""
globalComp.py
  run summary plots and comparisons to constraints
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
import cosmo
from util.loadExtern import *
from util.helper import running_median, running_histogram
from illustris_python.util import partTypeNum
from scipy.signal import savgol_filter

def stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=False):
    """ Stellar mass / halo mass relation, full boxes and zoom points vs. abundance matching lines. """

    # config
    subhaloID  = 0 # plot which subhalo for zoom runs
    colors     = ['blue','purple','black'] # Mtot, M<2r1/2, M<r1/2 (for markers)
    linestyles = [':','-','--','-.'] # 
    binSize    = 0.2 # for running median lines

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

    #ax.set_title('')
    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
    ax.set_ylabel('M$_\star$ / M$_{\\rm halo}$ $(\Omega_{\\rm b} / \Omega_{\\rm m})^{-1}$ [ only centrals ]')

    # abundance matching constraints
    b = behrooziSMHM(sPs[0])
    m = mosterSMHM(sPs[0])
    k = kravtsovSMHM(sPs[0])

    # '#ff6677', '#66ff77', 'cyan'
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

        if sP.isZoom:
            gc = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
            gh = cosmo.load.groupCatSingle(sP, haloID=gc['SubhaloGrNr'])

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
            gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], 
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
                ym2 = savgol_filter(ym,5,3)
                l, = ax.plot(xm[:-1], ym2[:-1], linestyles[0], color=c, lw=3.0)

            yy = gc['subhalos']['SubhaloMassInRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
            xm, ym, sm = running_median(xx,yy,binSize=binSize)
            ym2 = savgol_filter(ym,5,3)
            l, = ax.plot(xm[:-1], ym2[:-1], linestyles[1], lw=3.0, color=c, label=sP.simName)
            lines.append(l)

            if allMassTypes:
                yy = gc['subhalos']['SubhaloMassInHalfRadType'][w,4] / xx_code / (sP.omega_b/sP.omega_m)
                xm, ym, sm = running_median(xx,yy,binSize=binSize)
                ym2 = savgol_filter(ym,5,3)
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
    massBinColors = ['#333333','#666666','#999999'] #['#333333','#555555','#777777','#999999']
    maxNumSnaps = 20

    # plot setup
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)
    
    ax.set_ylim([8e-3, 5e2])
    cosmo.util.addRedshiftAgeAxes(ax, sPs[0])
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
        snaps, nSnaps = cosmo.util.validSnapList(sP, maxNum=maxNumSnaps)

        redshifts = np.zeros(nSnaps)
        sfrs = np.zeros( (nSnaps, len(plotMassBins), 3), dtype='float32' )
        stds = np.zeros( (nSnaps, len(plotMassBins), 3), dtype='float32' )

        # loop over all snapshots
        for j, snap in enumerate(snaps):
            sP = simParams(res=sP.res, run=sP.run, snap=snap)

            gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'], 
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

def sfrdVsRedshift(sPs, pdf, xlog=True):
    """ Star formation rate density of the universe, vs redshift, vs observational points. """
    from cosmo.util import addRedshiftAgeAxes
    
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
        s = sfrTxt(sP)

        ax.plot(s['redshift'], s['sfrd'], '-', lw=2.5, label=sP.simName)

    # second legeng
    legend2 = ax.legend(loc='lower left')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def blackholeVsStellarMass(sPs, pdf):
    """ Black hole mass vs. stellar (bulge) mass relation at z=0. """
    from cosmo.load import groupCat

    binSize = 0.2

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
        ym2 = savgol_filter(ym,5,3)
        l, = ax.plot(xm[:-1], ym2[:-1], '-', lw=3.0, label=sP.simName)

    # second legeng
    legend2 = ax.legend(loc='lower right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def galaxySizes(sPs, pdf, vsHaloMass=False):
    """ Galaxy sizes (half mass radii) vs stellar mass or halo mass, at redshift zero. """
    from cosmo.load import groupCat

    linestyles = ['-',':']
    binSize = 0.2

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

        ym_gas   = savgol_filter(ym_gas,5,3)
        ym_stars = savgol_filter(ym_stars,5,3)
        sm_gas   = savgol_filter(sm_gas,5,3)
        sm_stars = savgol_filter(sm_stars,5,3)

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

def stellarMassFunction(sPs, pdf, highMassEnd=False, centralsOnly=False):
    """ Stellar mass function (number density of galaxies) at redshift zero. """
    from cosmo.load import groupCat

    linestyles = ['-',':','--'] # 2rhalf, 1rhalf, total
    binSize = 0.2 # dex, stellar mass
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
            ym = savgol_filter(ym,5,3)

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

def plots():
    """ Plot portfolio of global population comparisons between runs. """
    from matplotlib.backends.backend_pdf import PdfPages
    from datetime import datetime
    from util import simParams

    # config
    sPs = []

    # zooms
    zoomNums = [1,2,3,4]
    redshift = 0.0

    for hInd in zoomNums:
        sPs.append( simParams(res=3, run='iClusters', variant='TNG_11', hInd=hInd, redshift=redshift) )
    sPs.append( simParams(res=2, run='iClusters', variant='TNG_11', hInd=1, redshift=redshift) )

    # fullboxes
    sPs.append( simParams(res=1820, run='illustris', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L25n512_PRVS', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L12.5n256_PR02_04', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L12.5n256_PRVS_B', redshift=redshift) )

    ##sPs.append( simParams(res=512, run='L12.5n256_PR06_01', redshift=redshift) )
    ##sPs.append( simParams(res=512, run='L12.5n256_PR06_02', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L12.5n256_PR06_03', redshift=redshift) )

    #sPs.append( simParams(res=512, run='L12.5n256_PR07_01', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L12.5n256_PR07_02', redshift=redshift) )
    #sPs.append( simParams(res=512, run='L12.5n256_PR07_03', redshift=redshift) )

    sPs.append( simParams(res=128, run='L12.5n256_count', redshift=redshift) )
    sPs.append( simParams(res=128, run='L12.5n256_discrete', redshift=redshift) )

    #sPs.append( simParams(res=1820, run='illustrisprime', redshift=0.5) )
    #sPs.append( simParams(res=910, run='illustris', redshift=redshift) )
    #sPs.append( simParams(res=455, run='illustris', redshift=redshift) )
    #sPs.append( simParams(res=256, run='L25n256_PR00', redshift=redshift) )

    # make multipage PDF
    pdf = PdfPages('globalComps_' + datetime.now().strftime('%d-%m-%Y')+'.pdf')

    stellarMassHaloMass(sPs, pdf, ylog=False)
    stellarMassHaloMass(sPs, pdf, ylog=False, allMassTypes=True)
    stellarMassHaloMass(sPs, pdf, ylog=True)
    sfrAvgVsRedshift(sPs, pdf)
    sfrdVsRedshift(sPs, pdf, xlog=True)
    sfrdVsRedshift(sPs, pdf, xlog=False)
    blackholeVsStellarMass(sPs, pdf)
    galaxySizes(sPs, pdf, vsHaloMass=False)
    galaxySizes(sPs, pdf, vsHaloMass=True)
    stellarMassFunction(sPs, pdf, highMassEnd=False)
    stellarMassFunction(sPs, pdf, highMassEnd=True)

    pdf.close()