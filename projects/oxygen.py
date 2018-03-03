"""
projects/oxygen.py
  Plots: Oxygen (OVI, OVII and OVIII) TNG paper.
  http://arxiv.org/abs/1712.00016
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker
from matplotlib.colors import Normalize, colorConverter
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d
from os.path import isfile
from functools import partial

from util import simParams
from util.loadExtern import werk2013, johnson2015
from plot.config import *
from util.helper import running_median, logZeroNaN, iterable, contourf, loadColorTable, getWhiteBlackColors, closest, reducedChiSq
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.cloudy import cloudyIon
from plot.general import plotPhaseSpace2D
from plot.quantities import simSubhaloQuantity, bandMagRange, quantList
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from plot.cloudy import ionAbundFracs2DHistos
from vis.common import setAxisColors
from cosmo.util import cenSatSubhaloIndices, redshiftToSnapNum, periodicDists
from obs.galaxySample import obsMatchedSample, addIonColumnPerSystem, ionCoveringFractions

def nOVIcddf(sPs, pdf, moment=0, simRedshift=0.2, boxDepth10=False, boxDepth125=False):
    """ CDDF (column density distribution function) of O VI in the whole box at z~0.
        (Schaye Fig 17) (Suresh+ 2016 Fig 11) """
    from util.loadExtern import danforth2008, danforth2016, thomChen2008, tripp2008

    # config
    lw = 3.5
    speciesList = ['nOVI','nOVI_solar','nOVI_10','nOVI_25']
    if boxDepth10:
        for i in range(len(speciesList)):
            speciesList[i] += '_depth10'
    if boxDepth125:
        speciesList = ['nOVII_solarz_depth125']#,'nOVII_10_solarz_depth125']

    # plot setup
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([12.5, 15.5])
    ax.set_xlabel('N$_{\\rm OVI}$ [ log cm$^{-2}$ ]')

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

    labels = [d16['label'],d08['label'],tc08['label'],t08['label']]
    legend1 = ax.legend([l1,l2,l3,l4], labels, loc='lower left')
    ax.add_artist(legend1)

    # loop over each fullbox run
    prevName = ''
    lwMod = 0.0

    for sP in sPs:
        if sP.isZoom:
            continue

        print('CDDF OVI: '+sP.simName)
        sP.setRedshift(simRedshift)

        if sP.simName.split("-")[0] == prevName:
            # decrease line thickness
            lwMod += 1.0
        else:
            # next color
            c = ax._get_lines.prop_cycler.next()['color']
            prevName = sP.simName.split("-")[0]
            lwMod = 0.0

        # pre-computed CDDF: first species for sizes
        ac = auxCat(sP, fields=['Box_CDDF_'+speciesList[0]])
        n_OVI  = ac['Box_CDDF_'+speciesList[0]][0,:]
        fN_OVI = ac['Box_CDDF_'+speciesList[0]][1,:]

        # pre-computed CDDF: allocate for max/min bounds of our variations
        fN_OVI_min = fN_OVI * 0.0 + np.inf
        fN_OVI_max = fN_OVI * 0.0

        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            acField = 'Box_CDDF_'+species

            if sP.simName in ['Illustris-1','TNG300-1'] and i > 0:
                continue # skip expensive variations we won't use for the oxygen paper

            ac = auxCat(sP, fields=[acField], searchExists=True)
            if ac[acField] is None:
                print(' skip: %s %s' % (sP.simName,species))
                continue

            assert np.array_equal(ac['Box_CDDF_'+species][0,:], n_OVI) # require same x-pts
            fN_OVI = ac[acField][1,:]

            fN_OVI_min = np.nanmin(np.vstack( (fN_OVI_min, fN_OVI) ), axis=0)
            fN_OVI_max = np.nanmax(np.vstack( (fN_OVI_max, fN_OVI) ), axis=0)

        # plot 'uncertainty' band
        xx = np.log10(n_OVI)

        if moment == 0:
            yy_min = logZeroNaN(fN_OVI_min)
            yy_max = logZeroNaN(fN_OVI_max)
            yy = logZeroNaN( 0.5*(fN_OVI_min+fN_OVI_max) )
        if moment == 1:
            yy_min = logZeroNaN(fN_OVI_min*n_OVI)
            yy_max = logZeroNaN(fN_OVI_max*n_OVI)
            yy = logZeroNaN( 0.5*(fN_OVI_min*n_OVI+fN_OVI_max*n_OVI) )

        ax.fill_between(xx, yy_min, yy_max, color=c, alpha=0.2, interpolate=True)

        # plot middle line
        label = sP.simName
        ax.plot(xx, yy, '-', lw=lw-lwMod, color=c, label=label)

        # calculate and print reduced (mean) chi^2
        chi2v = reducedChiSq(xx, yy, d16['log_NOVI'], d16['log_fOVI'], 
                             data_yerr_up=d16['log_fOVI_errUp'], data_yerr_down=d16['log_fOVI_errDown'])
        print('[%s] vs Danforth+ (2016) reduced chi^2: %g' % (sP.simName,chi2v))

    # legend
    sExtra = [] #[plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = [] #[str(s) for s in speciesList]

    if not clean:
        sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,alpha=0.0,marker='')]
        lExtra += ['[ sims z=%3.1f ]' % simRedshift]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def cddfRedshiftEvolution(sPs, saveName, moment=0, ions=['OVI','OVII'], redshifts=[0,1,2,3], 
                          boxDepth10=False, colorOff=0):
    """ CDDF (column density distribution function) of O VI in the whole box.
        (Schaye Fig 17) (Suresh+ 2016 Fig 11) """
    from util.loadExtern import danforth2016, muzahid2011

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    if len(redshifts) > 1: sizefac *= 0.7
    heightFac = 1.0 if ('main' in saveName or len(redshifts) > 1) else 0.95
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac*heightFac])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([12.5, 16.0])
    ax.set_xlabel('N$_{\\rm oxygen}$ [ log cm$^{-2}$ ]')
    if len(ions) == 1: ax.set_xlabel('N$_{\\rm %s}$ [ log cm$^{-2}$ ]' % ions[0])

    if moment == 0:
        ax.set_ylim([-18, -12])
        if len(ions) == 1: ax.set_ylim([-19, -11])
        ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm oxygen}$)  [ cm$^{2}$ ]')
        if len(ions) == 1: ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm %s}$)  [ cm$^{2}$ ]' % ions[0])
        if clean:
            ax.set_ylabel('f(N$_{\\rm oxygen}$) [ log cm$^{2}$ ]')
            if len(ions) == 1: ax.set_ylabel('f(N$_{\\rm %s}$) [ log cm$^{2}$ ]' % ions[0])
    if moment == 1:
        ax.set_ylim([-2.5, 1.5])
        ax.set_ylabel('CDDF (1$^{\\rm st}$ moment):  log N$_{\\rm oxygen}$ f(N$_{\\rm oxygen}$)')
        if clean:
            ax.set_ylabel('N$_{\\rm oxygen}$ $\cdot$ f(N$_{\\rm oxygen}$) [ log ]')
            if len(ions) == 1: ax.set_ylabel('N$_{\\rm %s}$ $\cdot$ f(N$_{\\rm %s}$) [ log ]' % (ions[0],ions[0]))
    if moment == 2:
        ax.set_ylim([-1.5, 2.2])
        ax.set_ylabel('CDDF (2$^{\\rm nd}$ moment):  log N$_{\\rm oxygen}^2$ f(N$_{\\rm oxygen}$)')
        if clean:
            ax.set_ylabel('[N$_{\\rm oxygen}$$^2$ / 10$^{13}$] $\cdot$ f(N$_{\\rm oxygen}$) [ log ]')
            if len(ions) == 1: ax.set_ylabel('N$_{\\rm %s}$ $\cdot$ f(N$_{\\rm %s}$) [ log ]' % (ions[0],ions[0]))

    # observational OVI points (not in paper)
    if 0:
        d16 = danforth2016()
        m11 = muzahid2011()

        if moment == 1:
            d16['log_fOVI']  = np.log10( 10.0**d16['log_fOVI'] * 10.0**d16['log_NOVI'] )
            m11['log_fOVI']  = np.log10( 10.0**m11['log_fOVI'] * 10.0**m11['log_NOVI'] )

        l1,_,_ = ax.errorbar(d16['log_NOVI'], d16['log_fOVI'], yerr=[d16['log_fOVI_errDown'],d16['log_fOVI_errUp']],
                   xerr=d16['log_NOVI_err'], color='#555555', ecolor='#555555', alpha=0.9, capsize=0.0, fmt='s')

        l2,_,_ =  ax.errorbar(m11['log_NOVI'][2:], m11['log_fOVI'][2:], yerr=[m11['log_fOVI_errDown'][2:],m11['log_fOVI_errUp'][2:]],
                   xerr=[m11['log_NOVI_errLow'][2:],m11['log_NOVI_errHigh'][2:]], color='#999999', ecolor='#999999', alpha=0.9, capsize=0.0, fmt='o')
        _,_,_  =  ax.errorbar(m11['log_NOVI'][:2], m11['log_fOVI'][:2], yerr=[m11['log_fOVI_errDown'][:2],m11['log_fOVI_errUp'][:2]],
                   xerr=[m11['log_NOVI_errLow'][:2],m11['log_NOVI_errHigh'][:2]], color='#999999', ecolor='#999999', alpha=0.3, capsize=0.0, fmt='o')

        legend1 = ax.legend([l1,l2], [d16['label'],m11['label']], loc='upper right')
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        txt = []

        for j, ion in enumerate(ions):
            print('[%s]: %s' % (ion,sP.simName))

            if j == 0:
                for _ in range(colorOff+1):
                    c = ax._get_lines.prop_cycler.next()['color']
            else:
                c = ax._get_lines.prop_cycler.next()['color']

            for i, redshift in enumerate(redshifts):
                sP.setRedshift(redshift)

                # Omega_ion value: compute
                if len(sPs) > 8:
                    fieldName = 'Box_Omega_' + ion
                    boxOmega = auxCat(sP, fields=[fieldName])[fieldName]

                # pre-computed CDDF: load at this redshift
                fieldName = 'Box_CDDF_n'+ion
                if boxDepth10: fieldName += '_depth10'

                ac = auxCat(sP, fields=[fieldName])
                N_ion  = ac[fieldName][0,:]
                fN_ion = ac[fieldName][1,:]

                xx = np.log10(N_ion)

                if moment == 0:
                    yy = logZeroNaN( fN_ion )
                if moment == 1:
                    yy = logZeroNaN( fN_ion*N_ion )
                if moment == 2:
                    yy = logZeroNaN( fN_ion*N_ion*(N_ion/1e13) )

                txt_loc = {}
                txt_loc['N'] = xx
                txt_loc['fN'] = yy
                txt_loc['z'] = redshift
                txt.append(txt_loc)

                # plot middle line
                label = '%s %s z=%.1f' % (sP.simName, ion, redshift) if len(redshifts) ==1 else '%s %s' % (sP.simName, ion)
                if clean:
                    label = ion
                    if len(ions) == 1 and not boxDepth10: label = sP.simName
                    if len(sPs) > 1: label = '%s %s' % (ion,sP.simName)
                    if len(sPs) > 8: label = '%s (%.1f)' % (sP.simName,boxOmega*1e7)
                if i > 0: label = ''
                c = 'black' if (len(sPs) > 5 and sP.variant == '0000') else c
                lwLoc = lw if not (len(sPs) == 12 and sP.variant == '0000') else 2*lw

                ls = linestyles[i]
                if i == 0 and len(sPs) > 8 and 'BH' in sP.simName: ls = '--'

                ax.plot(xx, yy, lw=lwLoc, color=c, linestyle=ls, label=label)

    # print
    if len(ions) == 1 and len(sPs) == 1:
        filename = 'fig5_cddf_%s.txt' % ion
        out = '# Nelson+ (2018) http://arxiv.org/abs/1712.00016\n'
        out += '# Figure 5 CDDFs (%s z=%.1f)\n' % (sP.simName, sP.redshift)
        if boxDepth10: out += '# Note: Calculated for a projection depth of 10 cMpc/h (=14.8 pMpc at z=0)\n'
        out += '# N_%s [log cm^-2]' % (ion)
        for redshift in redshifts: out += ' f(N)_z%d' % redshift
        out += ' (all [log cm^2])\n'

        for i in range(len(txt)-1): # make sure columns are the same at each redshift (for each column)
            assert np.array_equal(txt[i+1]['N'], txt[0]['N'])

        for i in range(txt[0]['N'].size):
            if txt[0]['N'][i] > 20.0: continue
            out += '%7.2f' % txt[0]['N'][i]
            for j in range(len(txt)): # loop over redshifts
                out += ' %7.3f' % txt[j]['fN'][i]
            out += '\n'

        with open(filename, 'w') as f:
            f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(redshifts) > 1:
        for i, redshift in enumerate(redshifts):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['z = %3.1f' % redshift]

    handles, labels = ax.get_legend_handles_labels()

    if len(sPs) == 13: # main variants, split into 2 legends
        legend1 = ax.legend(handles[0:4], labels[0:4], loc='upper right', prop={'size':18})
        ax.add_artist(legend1)
        legend2 = ax.legend(handles[4:]+sExtra, labels[4:]+lExtra, loc='lower left', prop={'size':18})
    else: # default
        loc = 'upper right' if len(ions) > 1 else 'lower left'
        legend2 = ax.legend(handles+sExtra, labels+lExtra, loc=loc)

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def totalIonMassVsHaloMass(sPs, saveName, ions=['OVI','OVII'], cenSatSelect='cen', redshift=0.0, 
                           vsHaloMass=True, secondTopAxis=False, toAvgColDens=False, colorOff=2, toyFacs=None):
    """ Plot total [gravitationally bound] mass of various ions, or e.g. cold/hot/total CGM mass, 
    versus halo or stellar mass at a given redshift. If toAvgColDens, then instead of total mass 
    plot average column density computed geometrically as (Mtotal/pi/rvir^2). 
    If secondTopAxis, add the other (halo/stellar) mass as a secondary top axis, average relation. """

    binSize = 0.2 # log mass
    renames = {'AllGas':'Total Gas / 100','AllGas_Metal':'Total Metals','AllGas_Oxygen':'Total Oxygen'}
    ionColors = {'AllGas':'#444444','AllGas_Metal':'#777777','AllGas_Oxygen':'#cccccc'}

    runToyModel = False # testing Lars' idea

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    heightFac = 1.1 if secondTopAxis else 1.0
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac*heightFac])
    ax = fig.add_subplot(111)
    
    mHaloLabel = 'M$_{\\rm halo}$ [ < r$_{\\rm 200,crit}$, log M$_{\\rm sun}$ ]'
    mHaloField = 'mhalo_200_log'
    mStarLabel = 'M$_{\star}$ [ < 30 pkpc, log M$_{\\rm sun}$ ]'
    mStarField = 'mstar_30pkpc_log'

    if vsHaloMass:
        ax.set_xlim([11.0, 15.0])
        ax.set_xlabel(mHaloLabel)
        massField = mHaloField
    else:
        ax.set_xlim([9.0, 12.0])
        ax.set_xlabel(mStarLabel)
        massField = mStarField

    if toAvgColDens:
        ax.set_ylim([12.0, 16.0])
        ax.set_ylabel('Average Column Density $<N_{\\rm oxygen}>$ [ log cm$^{-2}$ ]')
    else:
        ax.set_ylim([5.0, 9.0])
        if 'AllGas' in ions: ax.set_ylim([4.0, 12.0])
        ax.set_ylabel('Total Bound Gas Mass [ log M$_{\\rm sun}$ ]')

    if secondTopAxis:
        # add the other mass value as a secondary x-axis on the top of the panel
        axTop = ax.twiny()

        if vsHaloMass: # x=halo, top=stellar
            topMassVals = [8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
            axTop.set_xlabel(mStarLabel)
            topMassField = mStarField
        else: # x=stellar, top=halo
            topMassVals = [11.0, 11.5, 12.0, 13.0, 14.0, 15.0]
            axTop.set_xlabel(mHaloLabel)
            topMassField = mHaloField

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())

    # loop over each fullbox run
    colors = []

    for i, sP in enumerate(sPs):
        # load halo masses and CSS
        sP.setRedshift(redshift)
        xx = groupCat(sP, fieldsSubhalos=[massField])

        cssInds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
        xx = xx[cssInds]

        if secondTopAxis and i == 0:
            # load mass values for top x-axis, construct median relation interpolant, assign values
            xx_top = groupCat(sP, fieldsSubhalos=[topMassField])
            xx_top = xx_top[cssInds]
            xm, ym, _ = running_median(xx_top,xx,binSize=binSize,skipZeros=True,minNumPerBin=10)
            f = interp1d(xm, ym, kind='linear', bounds_error=False, fill_value='extrapolate')

            axTickVals = f(topMassVals) # values of bottom x-axis for each topMassVals

            axTop.set_xticks(axTickVals)
            axTop.set_xticklabels(topMassVals)

        if toAvgColDens:
            # load virial radii
            rad = groupCat(sP, fieldsSubhalos=['rhalo_200_code'])
            rad = rad[cssInds]
            ionData = cloudyIon(None)

        if not toAvgColDens and runToyModel:
            # TEST: toy model for total grav. bound mass
            from cosmo.cloudy import cloudyIon
            ion = cloudyIon(sP, el=['Oxygen'])

            #tempFac = 0.6
            #densFac = 2.0
            #metalFac = 1.5
            tempFac, densFac, metalFac = toyFacs

            # load total grav. bound gas mass
            field = 'Subhalo_Mass_AllGas'
            tot_gas_mass = auxCat(sP, fields=[field])[field][cssInds]

            field2 = 'Subhalo_Mass_AllGas_Metal' #_Metal, _Oxygen
            tot_metal_mass = auxCat(sP, fields=[field2])[field2][cssInds]

            # calculate mean metallicity
            ww = np.where(xx >= 11.0) # min halo mass to consider for mean metal frac
            mean_metal_frac = np.nanmean(tot_metal_mass[ww] / tot_gas_mass[ww])
            virMetallicity = mean_metal_frac / ion.solar_Z

            print(' mean metal mass fraction: ',mean_metal_frac)
            print(' mean virial metallicity in solar units: ',virMetallicity)
            print(' oxygen to total mass ratio (solar): ',ion._solarMetalAbundanceMassRatio('Oxygen'))

            # assume solar abundance of oxygen, scale to virMetallicity and use to compute total oxy mass
            oxyMassFracAtFacVirMetallicity = ion._solarMetalAbundanceMassRatio('Oxygen') * metalFac*virMetallicity
            virTotOxygenMass = tot_gas_mass * oxyMassFracAtFacVirMetallicity # code units
            yy_toy = {}
            yy_toy['AllGas_Oxygen'] = sP.units.codeMassToLogMsun(virTotOxygenMass)

            # define temp, dens per halo
            haloMassesCode = sP.units.logMsunToCodeMass(xx)
            virTemp = sP.units.codeMassToVirTemp(haloMassesCode*tempFac, log=True) # log K
            virVolume = (4.0/3.0) * np.pi * sP.units.codeMassToVirRad(haloMassesCode)**3.0
            virDens = (tot_gas_mass * ion.solar_X) / virVolume # hydrogen, code units
            virDensPhys = np.log10(sP.units.codeDensToPhys(virDens*densFac, cgs=True, numDens=True)) # log(1/cm^3)

            # run cloudy for each ion, predict total grav. bound ionic mass
            metal = np.zeros( virTemp.size, dtype='float32') + np.log10(virMetallicity)
            for ionNum in [6,7,8]:
                log_ionFrac = ion.frac('Oxygen', ionNum, virDensPhys, metal, virTemp)
                virTotIonMass = 10.0**log_ionFrac * virTotOxygenMass
                yy_toy['O' + ion.numToRoman(ionNum)] = sP.units.codeMassToLogMsun(virTotIonMass)

        for j, ion in enumerate(ions):
            print('[%s]: %s' % (ion,sP.simName))

            # load and apply CSS
            fieldName = 'Subhalo_Mass_%s' % ion

            ac = auxCat(sP, fields=[fieldName])
            if ac[fieldName] is None: continue
            ac[fieldName] = ac[fieldName][cssInds]

            # unit conversions
            if toAvgColDens:
                # per subhalo normalization, from [code mass] -> [ions/cm^2]
                assert ion[0] == 'O' # oxygen

                # [code mass] -> [code mass / code length^2]
                yy = ac[fieldName] / (np.pi * rad * rad)
                # [code mass/code length^2] -> [H atoms/cm^2]
                yy = sP.units.codeColDensToPhys(yy, cgs=True, numDens=True) 
                yy /= ionData.atomicMass(ion[0]) # [H atoms/cm^2] to [ions/cm^2]
                yy = logZeroNaN(yy)
            else:
                yy = sP.units.codeMassToLogMsun(ac[fieldName])

            if ion == 'AllGas': yy -= 2.0 # offset!
            
            # calculate median and smooth
            xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,
                                            skipZeros=True,percs=[10,25,75,90], minNumPerBin=10)

            if xm.size > sKn:
                ym = savgol_filter(ym,sKn,sKo)
                sm = savgol_filter(sm,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1) # P[10,90]

            # determine color
            if i == 0:
                if ion in ionColors: # preset color
                    c = ionColors[ion]
                else: # cycle
                    for _ in range(colorOff+1):
                        c = ax._get_lines.prop_cycler.next()['color']
                    if colorOff > 0: colorOff = 0 # only once
                colors.append(c)
            else:
                c = colors[j]

            # plot median line
            label = ion if i == 0 else ''
            if ion in renames.keys() and i == 0: label = renames[ion]
            ax.plot(xm, ym, lw=lw, color=c, linestyle=linestyles[i], label=label)

            if i == 0:
                # show percentile scatter only for 'all galaxies'
                ax.fill_between(xm, pm[0,:], pm[-1,:], color=c, interpolate=True, alpha=0.2)

            if ion in ['AllGas_Oxygen','OVI','OVII','OVIII']:
                # TOY PLOT
                xm, ym, sm, pm = running_median(xx,yy_toy[ion],binSize=binSize,
                                                skipZeros=True,percs=[10,25,75,90], minNumPerBin=10)
                ym = savgol_filter(ym,sKn,sKo)
                sm = savgol_filter(sm,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1) # P[10,90]
                ax.plot(xm, ym, lw=lw, color=c, linestyle=':')

    # legend
    sExtra = []
    lExtra = []

    if clean:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['%s' % sP.simName]
        if runToyModel:
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=':',marker='')]
            lExtra += ['Toy Model, f$_{\\rm T}$=%.1f, f$_{\\rm \\rho}$=%.1f, f$_{\\rm Z}$=%.1f' % \
                         (tempFac,densFac,metalFac)]
        loc = 'upper right' if toAvgColDens else 'lower right'
        legend1 = ax.legend(sExtra, lExtra, loc=loc)
        ax.add_artist(legend1)

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles, labels, loc='upper left')

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def _resolutionLineHelper(ax, sP, radRelToVirRad=False, rvirs=None, massBins=None, corrMaxBox=False):
    """ Helper: add some resolution lines at small radius. """
    resBandPKpc = 2.0 * sP.gravSoft
    yOff = 0.15
    xOff = 0.02
    textOpts = {'ha':'right', 'va':'bottom', 'rotation':90, 'color':'#555555', 'alpha':0.2}

    yy = np.array(ax.get_ylim())
    xx = np.array(ax.get_xlim())

    if not radRelToVirRad:
        if not corrMaxBox:
            ax.text(xx[1]-xOff,yy[0]+yOff,"%d Mpc" % (10.0**xx[1]/1000.0), **textOpts)

        xx[1] = np.log10(resBandPKpc) # log [pkpc]
        ax.fill_between(xx, [yy[0],yy[0]], [yy[1],yy[1]], color='#555555', alpha=0.1)
        ax.text(xx[1]-xOff, yy[0]+yOff, "Resolution Limit", **textOpts)
    else:
        minMpc = (10.0**xx[1])*sP.units.codeLengthToKpc(rvirs[0]) / 1000.0
        maxMpc = (10.0**xx[1])*sP.units.codeLengthToKpc(rvirs[-1]) / 1000.0
        ax.text(xx[1]-xOff, yy[0]+yOff, "%d Mpc $\\rightarrow$ %d Mpc" % (minMpc,maxMpc), **textOpts)

        for k, massBin in enumerate(massBins):
            xx[1] = np.log10(resBandPKpc / sP.units.codeLengthToKpc(rvirs[k]))
            ax.fill_between(xx, [yy[0],yy[0]], [yy[1],yy[1]], color='#555555', alpha=0.1+0.01*k)
            if k == 0:
                ax.text(xx[1]-xOff, yy[0]+yOff, "Resolution Limit", **textOpts)

    if corrMaxBox:
        # show maximum separation scale at which tpcf is trustable (~5 Mpc/h for TNG100, ~20 Mpc/h for TNG300)
        boxBandPKpc = sP.units.codeLengthToKpc( sP.boxSize / 15.0 ) # default
        if sP.boxSize == 75000.0: boxBandPKpc = sP.units.codeLengthToKpc( 5000.0 )
        if sP.boxSize == 205000.0: boxBandPKpc = sP.units.codeLengthToKpc( 5000.0*(50/20) )

        xx = np.array(ax.get_xlim())
        xx[0] = np.log10(boxBandPKpc)

        ax.fill_between(xx, [yy[0],yy[0]], [yy[1],yy[1]], color='#333333', alpha=0.05)
        ax.text(xx[0]-xOff, yy[0]+yOff+2.0, "Box Size Limit", **textOpts)


def stackedRadialProfiles(sPs, saveName, ions=['OVI'], redshift=0.0, cenSatSelect='cen', projDim='3D',
                          radRelToVirRad=False, massDensity=False, haloMassBins=None, stellarMassBins=None,
                          combine2Halo=False):
    """ Plot average/stacked radial number/mass density profiles for a series of halo or 
    stellar mass bins. One or more ions, one or more runs, at a given redshift. Specify one of 
    haloMassBins or stellarMassBins. If radRelToVirRad, then [r/rvir] instead of [pkpc]. If 
    massDensity, then [g/cm^3] instead of [1/cm^3]. If combine2Halo, then combine the other-halo and 
    diffuse terms. """
    from tracer.tracerMC import match3

    # config
    percs = [10,90]

    fieldTypes = ['GlobalFoF'] # Global, Subfind, GlobalFoF, SubfindGlobal

    fieldNames = []
    for fieldType in fieldTypes:
        fieldNames.append( 'Subhalo_RadProfile%s_'+fieldType+'_%s_Mass' )

    radNames = ['total','self (1-halo)','other (2-halo)','diffuse']

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
    ax = fig.add_subplot(111)
    
    radStr = 'Radius' if '3D' in projDim else 'Impact Parameter'
    if radRelToVirRad:
        ax.set_xlim([-2.0, 2.0])
        ax.set_xlabel('%s / Virial Radius [ log ]' % radStr)
    else:
        ax.set_xlim([0.0, 4.0])
        ax.set_xlabel('%s [ log pkpc ]' % radStr)

    speciesStr = ions[0] if len(ions) == 1 else 'oxygen'

    if '3D' in projDim:
        # 3D mass/number density
        if massDensity:
            ax.set_ylim([-37.0,-30.0])
            ax.set_ylabel('Mass Density $\\rho_{\\rm %s}$ [ log g cm$^{-3}$ ]' % speciesStr)
        else:
            ax.set_ylim([-14.0, -6.0])
            ax.set_ylabel('Number Density $n_{\\rm %s}$ [ log cm$^{-3}$ ]' % speciesStr)
    else:
        # 2D mass/column density
        if massDensity:
            ax.set_ylim([-12.0,-6.0])
            ax.set_ylabel('Column Mass Density $\\rho_{\\rm %s}$ [ log g cm$^{-2}$ ]' % speciesStr)
        else:
            ax.set_ylim([11.0, 16.0])
            ax.set_ylabel('Column Number Density $N_{\\rm %s}$ [ log cm$^{-2}$ ]' % speciesStr)

    # init
    ionData = cloudyIon(None)
    colors = []
    rvirs = []

    if haloMassBins is not None:
        massField = 'mhalo_200_log'
        massBins = haloMassBins
    else:
        massField = 'mstar_30pkpc_log'
        massBins = stellarMassBins

    # loop over each fullbox run
    txt = []

    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        sP.setRedshift(redshift)
        masses = groupCat(sP, fieldsSubhalos=[massField])

        cssInds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii
        rad = groupCat(sP, fieldsSubhalos=['rhalo_200_code'])
        rad = rad[cssInds]

        for j, ion in enumerate(ions):
            print('[%s]: %s' % (ion,sP.simName))

            # load and apply CSS
            for fieldName in fieldNames:
                fieldName = fieldName % (projDim,ion)
                ac = auxCat(sP, fields=[fieldName])
                if ac[fieldName] is None: continue

                # crossmatch 'subhaloIDs' to cssInds
                ac_inds, css_inds = match3( ac['subhaloIDs'], cssInds )
                ac[fieldName] = ac[fieldName][ac_inds,:]
                masses_loc = masses[css_inds]
                rad_loc = rad[css_inds]

                # unit conversions: mass per bin to (space mass density) or (space number density)
                yy = ac[fieldName]

                if '3D' in projDim:
                    normField = 'bin_volumes_code'
                    unitConversionFunc = sP.units.codeDensToPhys
                else:
                    normField = 'bin_areas_code' # 2D
                    unitConversionFunc = sP.units.codeColDensToPhys

                if ac[fieldName].ndim == 2:
                    yy /= ac[fieldName+'_attrs'][normField]
                    nRadTypes = 1
                else:
                    for radType in range(ac[fieldName].shape[2]):
                        yy[:,:,radType] /= ac[fieldName+'_attrs'][normField]
                    nRadTypes = 4

                if massDensity:
                    # from e.g. [code mass / code length^3] -> [g/cm^3]
                    yy = unitConversionFunc(yy, cgs=True)
                else:
                    # from e.g. [code mass / code length^3] -> [ions/cm^3]
                    assert ion[0] == 'O' # oxygen
                    yy = unitConversionFunc(yy, cgs=True, numDens=True) 
                    yy /= ionData.atomicMass(ion[0]) # [H atoms/cm^3] to [ions/cm^3]

                # loop over mass bins
                for k, massBin in enumerate(massBins):
                    txt_mb = []
                    # select
                    w = np.where( (masses_loc >= massBin[0]) & (masses_loc < massBin[1]) )

                    print(' %s [%d] %.1f - %.1f : %d' % (projDim,k,massBin[0],massBin[1],len(w[0])))
                    assert len(w[0])

                    # radial bins: normalize to rvir if requested
                    avg_rvir_code = np.nanmedian( rad_loc[w] )
                    if i == 0 and j == 0: rvirs.append( avg_rvir_code )

                    # sum and calculate percentiles in each radial bin
                    for radType in range(nRadTypes):
                        if yy.ndim == 3:
                            yy_local = np.squeeze( yy[w,:,radType] )

                            # combine diffuse into other-halo term, and skip separate line?
                            if combine2Halo and radType == 2:
                                yy_local += np.squeeze( yy[w,:,radType+1] )
                            if combine2Halo and radType == 3:
                                continue
                        else:
                            yy_local = np.squeeze( yy[w,:] )

                        if radRelToVirRad:
                            rr = 10.0**ac[fieldName+'_attrs']['rad_bins_code'] / avg_rvir_code
                        else:
                            rr = ac[fieldName+'_attrs']['rad_bins_pkpc']

                        # for low res runs, combine the inner bins which are poorly sampled
                        if sP.boxSize == 25000.0:
                            nInner = int( 20 / (sP.res/256) )
                            rInner = np.mean( rr[0:nInner] )

                            for dim in range(yy_local.shape[0]):
                                yy_local[dim,nInner-1] = np.nanmedian( yy_local[dim,0:nInner] )
                            yy_local = yy_local[:,nInner-1:]
                            rr = np.hstack( [rInner,rr[nInner:]] )

                        # replace zeros by nan so they are not included in percentiles
                        yy_local[yy_local == 0.0] = np.nan

                        # calculate mean profile and scatter
                        yy_mean = np.nansum( yy_local, axis=0 ) / len(w[0])
                        yp = np.nanpercentile( yy_local, percs, axis=0 )

                        # log both axes
                        yy_mean = logZeroNaN(yy_mean)
                        yp = logZeroNaN(yp)
                        rr = np.log10(rr)

                        if rr.size > sKn:
                            yy_mean = savgol_filter(yy_mean,sKn,sKo)
                            yp = savgol_filter(yp,sKn,sKo,axis=1) # P[10,90]

                        # determine color
                        if i == 0 and radType == 0:
                            c = ax._get_lines.prop_cycler.next()['color']
                            colors.append(c)
                        else:
                            c = colors[k]

                        # plot median line
                        label = '%.1f < $M_{\\rm halo}$ < %.1f' \
                          % (massBin[0],massBin[1]) if (i == 0 and radType == 0) else ''
                        label = '$M_{\\rm halo}$ = %.1f' \
                          % (0.5*(massBin[0]+massBin[1])) if (i == 0 and radType == 0) else ''
                        ax.plot(rr, yy_mean, lw=lw, color=c, linestyle=linestyles[radType], label=label)

                        txt_loc = {}
                        txt_loc['bin'] = massBin
                        txt_loc['rr'] = rr
                        txt_loc['yy'] = yy_mean
                        txt_loc['yy_0'] = yp[0,:]
                        txt_loc['yy_1'] = yp[-1,:]
                        txt_mb.append(txt_loc)

                        # draw rvir lines (or 300pkpc lines if x-axis is already relative to rvir)
                        yrvir = ax.get_ylim()
                        yrvir = np.array([ yrvir[1], yrvir[1] - (yrvir[1]-yrvir[0])*0.1]) - 0.25

                        if not radRelToVirRad:
                            xrvir = np.log10( [avg_rvir_code, avg_rvir_code] )
                            textStr = 'R$_{\\rm vir}$'
                            if '3D' in projDim:
                                yrvir[1] -= 0.4 * k
                            else:
                                yrvir[1] -= 0.1 * k
                        else:
                            rvir_300pkpc_ratio = sP.units.physicalKpcToCodeLength(300.0) / avg_rvir_code
                            xrvir = np.log10( [rvir_300pkpc_ratio, rvir_300pkpc_ratio] )
                            textStr = '300 kpc'
                            if '3D' in projDim:
                                yrvir[1] -= 0.4 * (len(massBins)-k)
                            else:
                                yrvir[1] -= 0.1 * (len(massBins)-k)

                        ax.plot(xrvir, yrvir, lw=lw*1.5, color=c, alpha=0.1)
                        ax.text(xrvir[0]-0.02, yrvir[1], textStr, color=c, va='bottom', ha='right', 
                                alpha=0.1, rotation=90)

                        if i == 0 and radType == 0:
                            # show percentile scatter only for first run
                            ax.fill_between(rr, yp[0,:], yp[-1,:], color=c, interpolate=True, alpha=0.2)

                    txt.append(txt_mb)

    # gray resolution band at small radius
    _resolutionLineHelper(ax, sPs[0], radRelToVirRad, rvirs=rvirs, massBins=massBins)

    # print
    for k in range(len(txt)): # loop over mass bins (separate file for each)
        filename = 'fig9_%sdens_rad%s_m-%.2f.txt' % \
          ('num' if projDim=='3D' else 'col', 'rvir' if radRelToVirRad else 'kpc', np.mean(txt[k][0]['bin']))
        out = '# Nelson+ (2018) http://arxiv.org/abs/1712.00016\n'
        out += '# Figure 9 Left Panel n_OVI [log cm^-3] (%s z=%.1f)\n' % (sP.simName, sP.redshift)
        out += '# Halo Mass Bin [%.1f - %.1f]\n' % (txt[k][0]['bin'][0], txt[k][0]['bin'][1])
        out += '# rad_logpkpc'
        for j in range(len(txt[k])): # loop over rad types
            radName = radNames[j].split(" ")[0]
            out += ' n_%s n_%s_err0 n_%s_err1' % (radName,radName,radName)
        out += '\n'
        for i in range(1,txt[k][0]['rr'].size): # loop over radial bins
            out += '%8.4f ' % txt[k][j]['rr'][i]
            for j in range(len(txt[k])): # loop over rad types
                out += '%8.4f %8.4f %8.4f' % (txt[k][j]['yy'][i], txt[k][j]['yy_0'][i], txt[k][j]['yy_1'][i])
            out += '\n'
        with open(filename, 'w') as f:
            f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if clean:
        if len(sPs) > 1:
            for i, sP in enumerate(sPs):
                sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
                lExtra += ['%s' % sP.simName]
        for i in range(nRadTypes - int(combine2Halo)):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['%s' % radNames[i]]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def oxygenTwoPointCorrelation(sPs, saveName, ions=['OVI'], redshift=0.0, order=0, colorOff=0):
    """ Plot the real-space 3D two point correlation function of e.g. OVI mass. """
    from cosmo.clustering import twoPointAutoCorrelationParticle

    # visual config
    lw = 3.0
    alphaFill = 0.15
    drawError = True
    symSize = 7.0
    alpha = 1.0

    # quick helper mapping from ions[] inputs to snapshotSubset() particle field names
    ionNameToPartFieldMap = {'OVI':'O VI mass','OVII':'O VII mass','OVIII':'O VIII mass',
                             'O':'metalmass_O','Z':'metalmass','gas':'mass','bhmass':'BH_Mass'}

    # plot setup
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([0.0, 4.0])
    ax.set_xlabel('Radius [ log pkpc ]')

    ax.set_ylim([-1.0,6.0])
    if order == 1: ax.set_ylim([1.0,7.0])
    if order == 2: ax.set_ylim([1.0,8.0])
    if ions[0] == 'bhmass' and order == 0: ax.set_ylim([-1.0,5.0])
    ionStr = '_{\\rm %s}'%ions[0] if len(ions) == 1 else ''
    ax.set_ylabel('%s$\\xi%s(r)$ [ log ]' % (['','r','r$^2$'][order],ionStr))

    # loop over each particle type/property
    for j, ion in enumerate(ions):
        if j == 0:
            for _ in range(colorOff+1):
                c = ax._get_lines.prop_cycler.next()['color']
        else:
            c = ax._get_lines.prop_cycler.next()['color']

        if ion == 'bhmass':
            partType = 'bh'
        else:
            partType = 'gas'
        partField = ionNameToPartFieldMap[ion]

        # loop over each fullbox run
        for i, sP in enumerate(sPs):
            print('[%s]: %s' % (ion,sP.simName))
            sP.setRedshift(redshift)

            # load tpcf
            rad, xi, xi_err, _ = twoPointAutoCorrelationParticle(sP, partType=partType, partField=partField)

            xx = sP.units.codeLengthToKpc(rad)
            xx = rad
            ww = np.where( xi > 0.0 )

            # y-axis multiplier
            if order == 0: yFac = 1.0
            if order == 1: yFac = xx[ww]
            if order == 2: yFac = xx[ww]**2

            x_plot = logZeroNaN(xx[ww])
            y_plot = logZeroNaN(yFac * xi[ww])

            label = ion if i == 0 else ''
            if label == 'O': label = 'O, Z$_{\\rm tot}$'
            l, = ax.plot(x_plot, y_plot, lw=lw, linestyle=linestyles[i], label=label, color=c, alpha=alpha)

            # todo, symbols, bands, etc
            if xi_err is not None and drawError:
                nSigma = 10.0 # 5sigma up and down
                if 1:
                    yy0 = y_plot - logZeroNaN( yFac*(xi[ww] - nSigma*xi_err[ww]/2) )
                    yy1 = logZeroNaN( yFac*(xi[ww] + nSigma*xi_err[ww]/2) ) - y_plot
                    ax.errorbar(x_plot, y_plot, yerr=[yy0,yy1], markerSize=symSize, 
                         color=l.get_color(), ecolor=l.get_color(), alpha=alphaFill*2, capsize=0.0, fmt='o')

                if 0:
                    yy0 = logZeroNaN( yFac*(xi[ww] - nSigma*xi_err[ww]/2) )
                    yy1 = logZeroNaN( yFac*(xi[ww] + nSigma*xi_err[ww]/2) )

                    ax.fill_between(x_plot, yy0, yy1, color=l.get_color(), interpolate=True, alpha=alphaFill)

    # gray resolution band at small radius
    _resolutionLineHelper(ax, sPs[0], corrMaxBox=True)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['%s' % sP.simName]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='best')

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def obsSimMatchedGalaxySamples(sPs, saveName, config='COS-Halos'):
    """ Plot the COS-Halos (or other observed) galaxies data, and our mock sample."""
    from scipy.stats import binned_statistic_2d

    # config
    detLimitAlpha = 0.6 # alpha transparency for upper/lower limits
    lw = 3.0

    cmap = loadColorTable('RdYlGn')
    colorMinMax = [0.1,0.3] # redshift
    cbarTextSize = 13
    nBinsHist = 8

    cmap2D = loadColorTable('gray_r')
    nBinsHist2D = 30

    xlim = [9.5, 11.5] # log mstar [msun]
    ylim = [-13.0, -9.0] # log ssfr [1/yr]
    if config in ['eCGM','eCGMfull']: xlim = [9.0, 11.5]

    # load data
    if config == 'COS-Halos': datafunc = werk2013
    if config == 'eCGM': datafunc = johnson2015
    if config == 'eCGMfull': datafunc = partial(johnson2015, surveys=['IMACS','SDSS','COS-Halos'])

    gals, logM, z, sfr, sfr_err, sfr_limit, R, ovi_logN, ovi_err, ovi_limit = datafunc()
    log_ssfr = np.log10(sfr/10.0**logM)

    sim_samples = []
    for sP in sPs:
        sim_samples.append( obsMatchedSample(sP, datasetName=config) )

    # plot geometry setup
    left = 0.12
    bottom = 0.12
    width = 0.64
    height = 0.66
    hist_pad = 0.02
    cbar_pad = 0.03

    rect_scatter = [left, bottom, width, height]
    rect_hist_top = [left, bottom+height+hist_pad, width, 1.0-height-bottom-hist_pad*2]
    rect_hist_right = [left+width+hist_pad, bottom, 1.0-left-width-hist_pad*2, height]
    rect_cbar = [left+cbar_pad, bottom+cbar_pad, 0.03, height/2]

    # plot setup
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac*0.9, figsize[1]*sizefac])

    ax = fig.add_axes(rect_scatter)

    ax.set_ylim(ylim)
    ax.set_ylabel('Galaxy sSFR [ 1/yr ]')
    ax.set_xlim(xlim)
    ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]')

    # plot sim 2d histogram in background
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBinsHist2D, int(nBinsHist2D*(bbox.height/bbox.width))])

    sim_xvals = sim_samples[0]['mstar_30pkpc_log'].ravel()
    sim_yvals = sim_samples[0]['ssfr_30pkpc_log'].ravel()
    sim_cvals = np.zeros( sim_xvals.size )

    cc, xBins, yBins, inds = binned_statistic_2d(sim_xvals, sim_yvals, sim_cvals, 'count', 
                                                 bins=nBins2D, range=[xlim,ylim])

    cc = cc.T # imshow convention
    cc2d = cc
    if config in ['eCGM','eCGMfull']: cc2d = logZeroNaN(cc2d)

    cMinMax = [np.nanmax(cc2d)*0.1, np.nanmax(cc2d)*1.1]
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap2D(norm(cc2d))

    cc2d_rgb[cc == 0.0,:] = colorConverter.to_rgba('white') # empty bins

    plt.imshow(cc2d_rgb, extent=[xlim[0],xlim[1],ylim[0],ylim[1]], origin='lower', 
               interpolation='nearest', aspect='auto')

    # plot obs scatterpoints on top
    for limitType in [2,1,0]: # upper, lower, exact
        w = np.where(ovi_limit == limitType)

        label = config if limitType == 0 else ''
        marker = ['o','v','^'][limitType]
        alpha = 1.0 if limitType == 0 else detLimitAlpha

        s = ax.scatter(logM[w], log_ssfr[w], s=80, marker=marker, c=z[w], label=label, alpha=alpha, 
                       edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1])

    # top histogram: setup
    ax_h1 = fig.add_axes(rect_hist_top)
    ax_h1.set_ylabel('PDF')
    ax_h1.set_xlabel('')
    ax_h1.set_xlim(xlim)
    ax_h1.set_ylim([0,1.5])
    ax_h1.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_h1.yaxis.set_major_formatter(ticker.NullFormatter())

    # top histogram: obs and sim
    c_obs = 'black'
    c_sim = []

    ax_h1.hist(logM, bins=nBinsHist, range=xlim, normed=True, histtype='bar', 
               alpha=0.5, color=c_obs, orientation='vertical', label=config)

    for i, sP in enumerate(sPs):
        c_sim.append( ax._get_lines.prop_cycler.next()['color'] )
        ax_h1.hist(sim_samples[i]['mstar_30pkpc_log'].ravel(), 
                   bins=nBinsHist*4, range=xlim, normed=True, histtype='bar', 
                   alpha=0.5, color=c_sim[-1], orientation='vertical', label=sP.simName)

    ax_h1.legend(loc='upper right', prop={'size':13})

    # right histogram: setup
    ax_h2 = fig.add_axes(rect_hist_right)
    ax_h2.set_xlabel('PDF')
    ax_h2.set_ylabel('')
    ax_h2.set_ylim(ylim)
    ax_h2.set_xlim([0,1.0])
    ax_h2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_h2.yaxis.set_major_formatter(ticker.NullFormatter())

    ax_h2.hist(log_ssfr, bins=nBinsHist, range=ylim, normed=True, histtype='bar', 
               alpha=0.5, color=c_obs, orientation='horizontal', label=config)

    for i, sP in enumerate(sPs):
        ax_h2.hist(sim_samples[i]['ssfr_30pkpc_log'].ravel(), 
                   bins=nBinsHist*4, range=ylim, normed=True, histtype='bar', 
                   alpha=0.5, color=c_sim[i], orientation='horizontal', label=sP.simName)

    # colorbar
    cbar_ax = fig.add_axes(rect_cbar)
    cb = fig.colorbar(s, cax=cbar_ax)
    cb.locator = ticker.MaxNLocator(nbins=1) # nbins = Nticks+1
    cb.update_ticks()
    cb.ax.set_ylabel('Galaxy Redshift', size=cbarTextSize+5)

    # colorbar labels
    cb.ax.tick_params(labelsize=0)
    cb.ax.text(0.5, 0.06, '%.1f' % colorMinMax[0], ha='center', va='center', size=cbarTextSize)
    cb.ax.text(0.5, 0.5, '%.1f' % np.mean(colorMinMax), ha='center', va='center', size=cbarTextSize)
    cb.ax.text(0.5, 0.94, '%.1f' % colorMinMax[1], ha='center', va='center', size=cbarTextSize)

    fig.savefig(saveName)
    plt.close(fig)

def cosOVIDataPlot(sP, saveName, radRelToVirRad=False, config='COS-Halos'):
    """ Plot COS-Halos N_OVI data, and our mock COS-Halos galaxy sample analysis."""

    # load data
    if config == 'COS-Halos': datafunc = werk2013
    if config == 'eCGM': datafunc = johnson2015
    if config == 'eCGMfull': datafunc = partial(johnson2015, surveys=['IMACS','SDSS','COS-Halos'])

    gals, logM, z, sfr, sfr_err, sfr_limit, R, ovi_logN, ovi_err, ovi_limit = datafunc()
    log_ssfr = np.log10(sfr/10.0**logM)

    sim_sample = obsMatchedSample(sP, datasetName=config)
    sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

    for iter in [0,1]:
        # plot setup
        sizefac = 1.0 if not clean else sfclean
        fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])

        ax = fig.add_subplot(111)

        if iter == 0:
            # x axis = impact parameter
            if radRelToVirRad:
                assert 0 # not implemented yet
                ax.set_xlim([0, 2.0])
                ax.set_xlabel('Projected Distance / Virial Radius')
                if config in ['eCGM','eCGMfull']: ax.set_xlim([0, 10])
            else:
                ax.set_xlim([0, 200])
                ax.set_xlabel('Projected Distance [ pkpc ]')
                if config in ['eCGM','eCGMfull']: ax.set_xlim([0, 1000])

        if iter == 1:
            # x axis = sSFR
            ax.set_xlim([-13.0, -9.0])
            ax.set_xlabel('sSFR [ 1/yr ]')

        ax.set_ylim([12.5, 15.5])
        if config in ['eCGM','eCGMfull']: ax.set_ylim([11.5, 15.5])
        ax.set_ylabel('Column Density $N_{\\rm OVI}$ [ log cm$^{-2}$ ]')

        # plot obs
        for limitType in [2,1,0]: # upper, lower, exact
            w = np.where(ovi_limit == limitType)

            label = config if limitType == 0 else ''
            marker = ['o','v','^'][limitType]

            if iter == 0:
                x_vals = R[w]
                c_vals = log_ssfr[w]
                c_label = 'sSFR [ log 1/yr ]'
                colorMinMax = [-12.0, -10.0]
                cmap = loadColorTable('coolwarm_r')
            if iter == 1:
                x_vals = log_ssfr[w]
                c_vals = logM[w]
                c_label = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
                colorMinMax = [9.0,11.2]
                cmap = loadColorTable('coolwarm')

            y_vals = ovi_logN[w]

            s = ax.scatter(x_vals, y_vals, s=80, marker=marker, c=c_vals, label=label, alpha=1.0, 
                           edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1])

        # plot sim
        if iter == 0:
            x_vals = sim_sample['impact_parameter'].ravel()
            c_vals = sim_sample['ssfr_30pkpc_log'].ravel()
            c_label = 'sSFR [ log 1/yr ]'
            colorMinMax = [-12.0, -10.0]
            cmap = loadColorTable('coolwarm_r')
        if iter == 1:
            x_vals = sim_sample['ssfr_30pkpc_log'].ravel()
            c_vals = sim_sample['mstar_30pkpc_log'].ravel()
            c_label = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
            colorMinMax = [9.0,11.2]
            cmap = loadColorTable('coolwarm')

        y_vals = sim_sample['column'].ravel()

        s = ax.scatter(x_vals, y_vals, s=10, marker='s', c=c_vals, label=sP.simName, alpha=0.3, 
                       edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1])

        # legend
        legend2 = ax.legend(loc='upper right')

        fig.tight_layout()

        # colorbar
        fig.subplots_adjust(right=0.84)
        cbar_ax = fig.add_axes([0.86, 0.12, 0.03, 0.84])
        cb = fig.colorbar(s, cax=cbar_ax)
        cb.ax.set_ylabel(c_label)

        cb.set_alpha(1) # fix stripes
        cb.draw_all()

        fig.savefig(saveName.split('.pdf')[0] + '_v%d.pdf' % iter)
        plt.close(fig)

def cosOVIDataPlotExtended(sP, saveName, config='COS-Halos'):
    """ Plot COS-Halos N_OVI data, and our mock COS-Halos galaxy sample analysis. Here to the 
    right of the plots we add stacked offset 1d KDEs of each realization vs observed point. """
    ylim = [12.5, 15.5]
    if config in ['eCGM','eCGMfull']: ylim = [11.5, 15.5]
    ylabel = 'Column Density $N_{\\rm OVI}$ [ log cm$^{-2}$ ]'

    lw = 2.5
    cbarTextSize = 13
    nKDE1D = 100
    kdeHeightFac = 4.0 # multiplicative horizontal size beyond individual bounds
    if config in ['eCGM','eCGMfull']: kdeHeightFac = 10.0

    # geometry
    left = 0.06
    bottom = 0.12
    width = 0.35
    height = 0.84
    hist_pad = 0.02
    cbar_pad = 0.015
    cbar_width = 0.02

    rect_mainpanel = [left, bottom, width, height]
    rect_right = [left+width+hist_pad, bottom, 1.0-left-width-hist_pad*2, height]
    rect_cbar1 = [left+width-cbar_pad*2-cbar_width, bottom+cbar_pad*(height/width), cbar_width, height/2]
    rect_cbar2 = [left+width/2-width/4, bottom+cbar_pad*4, width/2, cbar_width*2]

    # load data
    if config == 'COS-Halos': datafunc = werk2013
    if config == 'eCGM': datafunc = johnson2015
    if config == 'eCGMfull': datafunc = partial(johnson2015, surveys=['IMACS','SDSS','COS-Halos'])

    gals, logM, z, sfr, sfr_err, sfr_limit, R, ovi_logN, ovi_err, ovi_limit = datafunc()
    log_ssfr = np.log10(sfr/10.0**logM)

    sim_sample = obsMatchedSample(sP, datasetName=config)
    sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

    for ind in range(sim_sample['impact_parameter'].shape[0]):
        x_vals = sim_sample['impact_parameter'][ind,:].ravel()
        diff = x_vals.mean() - R[ind]
        assert np.abs(diff) < 1.0 # make sure we are matched and not mixed up

    for iter in [0,1]:
        # plot setup
        sizefac = 1.0 if not clean else sfclean
        fig = plt.figure(figsize=[figsize[0]*sizefac*2.0, figsize[1]*sizefac])
        
        ax = fig.add_axes(rect_mainpanel)

        if iter == 0:
            # x axis = impact parameter, color=sSFR
            ax.set_xlim([0, 200])
            if config in ['eCGM','eCGMfull']: ax.set_xlim([0, 1000])
            ax.set_xlabel('Projected Distance [ pkpc ]')

            c_label = 'sSFR [ log 1/Gyr ]'
            colorMinMax = [-3.0, -1.0]
            cmap = loadColorTable('coolwarm_r')

        if iter == 1:
            # x axis = sSFR, color=Mstar
            ax.set_xlim([-13.0, -9.0])
            ax.set_xlabel('sSFR [ log 1/yr ]')

            c_label = 'Stellar Mass [ log M$_{\\rm sun}$ ]'
            colorMinMax = [9.8,11.0]
            cmap = loadColorTable('coolwarm')

        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)

        # setup right panel
        ax_right = fig.add_axes(rect_right)

        ax_right.set_ylim(ylim)
        ax_right.set_xlim([0,1])
        ax_right.xaxis.set_major_formatter(ticker.NullFormatter())
        ax_right.yaxis.set_major_formatter(ticker.NullFormatter())

        xtick_vals = np.linspace(0.0, 1.0-(1.0/(len(gals)+2))*(kdeHeightFac-1.5), len(gals)+2)
        ax_right.set_xticks([])

        #ax_right.spines["bottom"].set_visible(False)
        #ax_right.spines["top"].set_visible(False)

        # main panel: plot obs
        for limitType in [2,1,0]: # upper, lower, exact (OVI)
            w = np.where(ovi_limit == limitType)

            if not len(w[0]): continue

            label = config if limitType == 0 else ''
            marker = ['o','v','^'][limitType]

            if iter == 0:
                x_vals = R[w]
                c_vals = np.log10(10.0**log_ssfr[w] * 1e9)
            if iter == 1:
                x_vals = log_ssfr[w]
                c_vals = logM[w]

            y_vals = ovi_logN[w]

            # add points to main panel
            s = ax.scatter(x_vals, y_vals, s=120, marker=marker, c=c_vals, label=label, alpha=1.0, 
                           edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1])

            if iter == 1:
                # x-axis values could also be limits: replicate markers into list and adjust
                ww = np.where(sfr_limit[w])
                x_off = 0.14               

                if len(ww[0]):
                    s = ax.scatter(x_vals[ww]-x_off, y_vals[ww], s=50, marker='<', c=c_vals[ww], alpha=1.0, 
                                   edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1])

                    norm = Normalize(vmin=colorMinMax[0], vmax=colorMinMax[1], clip=False)
                    for i in ww[0]:
                        ax.plot( [x_vals[i],x_vals[i]-x_off], [y_vals[i],y_vals[i]], '-', 
                                 color=cmap(norm(c_vals[i])) )
                #for i in ww[0]:
                #    print(x_vals[i],y_vals[i])
                #    ax.annotate(s='Q', xy=(x_vals[i], y_vals[i]), xytext=(x_vals[i]-0.5, y_vals[i]),
                #                arrowprops=dict(arrowstyle='->'))
                #    #ax.errorbar(x_vals, y_vals, 
                #    #    xuplims=ww, #xlolims=xlolims, uplims=uplims, lolims=lolims,
                #    #    marker='o', markersize=8,linestyle='-')

        # main panel: plot simulation
        if iter == 0:
            x_vals = sim_sample['impact_parameter'].ravel()
            c_vals = sim_sample['ssfr_30pkpc_log'].ravel()
            c_vals = np.log10(10.0**c_vals * 1e9) # 1/yr -> 1/Gyr
        if iter == 1:
            x_vals = sim_sample['ssfr_30pkpc_log'].ravel()
            c_vals = sim_sample['mstar_30pkpc_log'].ravel()

        y_vals = sim_sample['column'].ravel()

        s = ax.scatter(x_vals, y_vals, s=10, marker='s', c=c_vals, label=sP.simName, alpha=0.3, 
                       edgecolors='none', cmap=cmap, vmin=colorMinMax[0], vmax=colorMinMax[1], 
                       rasterized=True) # rasterize the 3700 squares into a single image

        # right panel: sort by order along x-axis of main panel
        if iter == 0:
            sort_inds = np.argsort( R )
        if iter == 1:
            sort_inds = np.argsort( log_ssfr )

        xx = xtick_vals[1:-1]

        # save some data for later
        pvals = np.zeros(R.size, dtype='float32')
        plims = np.zeros(R.size, dtype='int32')

        for i, sort_ind in enumerate(sort_inds):
            # plot vertical line and obs. galaxy name
            ax_right.plot( [ xx[i],xx[i] ], ylim, '-', color='black', alpha=0.04 )

            textOpts = {'ha':'center', 'va':'bottom', 'rotation':90, 'color':'#555555', 'fontsize':8}
            ax_right.text( xx[i], ylim[0], gals[sort_ind]['name'], **textOpts)

            # plot sim 1D KDE
            sim_cols = np.squeeze( sim_sample['column'][sort_ind,:] )

            kde_x = np.linspace(ylim[0], ylim[1], nKDE1D)
            kde = gaussian_kde(sim_cols, bw_method='scott')
            kde_y = kde(kde_x) * (1.0/len(gals)) * kdeHeightFac

            l, = ax_right.plot(kde_y + xx[i], kde_x, '-', alpha=1.0, lw=lw)
            ax_right.fill_betweenx(kde_x, xx[i], kde_y+xx[i], facecolor=l.get_color(), alpha=0.05)

            # locate 'height' of observed point beyond xx[i], i.e. the KDE value at its N_OVI
            _, kde_ind_obs = closest(kde_x, ovi_logN[sort_ind])

            # mark observed data point
            marker = ['o','v','^'][ovi_limit[sort_ind]]
            ax_right.plot( xx[i] + kde_y[kde_ind_obs], ovi_logN[sort_ind], markersize=12, marker=marker, 
                           color=l.get_color(), alpha=1.0)

            # add observational error as vertical line
            if ovi_err[sort_ind] > 0.0:
                obs_xerr = [xx[i], xx[i]] + kde_y[kde_ind_obs]
                obs_yerr = [ovi_logN[sort_ind]-ovi_err[sort_ind], ovi_logN[sort_ind]+ovi_err[sort_ind]]
                ax_right.plot( obs_xerr, obs_yerr, '-', color=l.get_color(), lw=lw, alpha=1.0 )

            # calculate and print a quantitative probability number
            z1 = kde.integrate_box_1d(-np.inf, ovi_logN[sort_ind])
            z2 = kde.integrate_box_1d(ovi_logN[sort_ind], np.inf)

            if ovi_limit[sort_ind] == 0: pvals[i] = 2*np.min([z1,z2]) # detection, 2*PDF area more extreme
            if ovi_limit[sort_ind] == 1: pvals[i] = z1 # upper limit, PDF area which is consistent
            if ovi_limit[sort_ind] == 2: pvals[i] = z2 # lower limit, PDF area which is consistent
            plims[i] = ovi_limit[sort_ind]

            print(gals[sort_ind]['name'], ovi_logN[sort_ind], pvals[i])

        # print summary of pvals statistic
        percs = np.nanpercentile(pvals, [16,50,84])
        print('all percentiles: ',percs)
        for lim in [0,1,2]:
            w = np.where(plims == lim)
            percs = np.nanpercentile(pvals[w], [16,50,84])
            print('limit [%d] percs:' % lim, percs)
        print('counts: ',np.count_nonzero(pvals < 0.05),np.count_nonzero(pvals < 0.01),pvals.size)

        # print summary of sim vs. obs mean/1sigma column densities
        percs = np.nanpercentile(ovi_logN, [16,50,84])
        print('obs OVI logN: %.2f (-%.2f +%.2f)' % (percs[1],percs[1]-percs[0],percs[2]-percs[1]))
        percs = np.nanpercentile(sim_sample['column'].ravel(), [16,50,84])
        print('sim OVI logN: %.2f (-%.2f +%.2f)' % (percs[1],percs[1]-percs[0],percs[2]-percs[1]))

        w_sf = np.where(log_ssfr >= -11.0)
        w_qq = np.where(log_ssfr < -11.0)
        percs_sf = np.nanpercentile(ovi_logN[w_sf], [16,50,84])
        percs_qq = np.nanpercentile(ovi_logN[w_qq], [16,50,84])
        print('obs SF OVI logN: %.2f (-%.2f +%.2f)' % (percs_sf[1],percs_sf[1]-percs_sf[0],percs_sf[2]-percs_sf[1]))
        print('obs QQ OVI logN: %.2f (-%.2f +%.2f)' % (percs_qq[1],percs_qq[1]-percs_qq[0],percs_qq[2]-percs_qq[1]))
        percs_sf = np.nanpercentile(sim_sample['column'][w_sf,:].ravel(), [16,50,84])
        percs_qq = np.nanpercentile(sim_sample['column'][w_qq,:].ravel(), [16,50,84])
        print('sim SF OVI logN: %.2f (-%.2f +%.2f)' % (percs_sf[1],percs_sf[1]-percs_sf[0],percs_sf[2]-percs_sf[1]))
        print('sim QQ OVI logN: %.2f (-%.2f +%.2f)' % (percs_qq[1],percs_qq[1]-percs_qq[0],percs_qq[2]-percs_qq[1]))

        # main panel: legend
        loc = ['upper right','upper left'][iter]
        legend2 = ax.legend(loc=loc, markerscale=1.8)
        legend2.legendHandles[0].set_color('#000000')
        legend2.legendHandles[1].set_color('#000000')

        # colorbar
        cbar_ax = fig.add_axes(rect_cbar1 if iter == 0 else rect_cbar2)
        cb = fig.colorbar(s, cax=cbar_ax, orientation=['vertical','horizontal'][iter])
        cb.set_alpha(1) # fix stripes
        cb.draw_all()

        cb.locator = ticker.MaxNLocator(nbins=1) # nbins = Nticks+1
        cb.update_ticks()
        if iter == 0: cb.ax.set_ylabel(c_label, size=cbarTextSize+3)
        if iter == 1: cb.ax.set_xlabel(c_label, size=cbarTextSize+3, labelpad=4)

        # colorbar labels
        cb.ax.tick_params(labelsize=0)
        if iter == 0:
            # vertical, custom labeling
            cbx = [0.5,  0.5, 0.5]
            cby = [0.06, 0.5, 0.94]
        if iter == 1:
            # horizontal, custom labeling
            cbx = [0.06, 0.5, 0.94]
            cby = [0.5, 0.5, 0.5]

        cb.ax.text(cbx[0], cby[0], '%.1f' % colorMinMax[0], ha='center', va='center', size=cbarTextSize)
        cb.ax.text(cbx[1], cby[1], '%.1f' % np.mean(colorMinMax), ha='center', va='center', size=cbarTextSize)
        cb.ax.text(cbx[2], cby[2], '%.1f' % colorMinMax[1], ha='center', va='center', size=cbarTextSize)

        fig.savefig(saveName.split('.pdf')[0] + '_v%d.pdf' % iter)
        plt.close(fig)

def coveringFractionVsDist(sPs, saveName, ions=['OVI'], config='COS-Halos', 
                           colDensThresholds=[13.5, 14.5], conf=0):
    """ Covering fraction of OVI versus impact parameter, COS-Halos data versus mock simulated sample, 
    or exploration of physics variations with respect to fiducial model. 
    colDensThresholds is a list in [1/cm^2] to compute. """
    assert len(ions) == 1
    assert ions[0] == 'OVI'

    gsNames = {'all':'All Galaxies',
               'mstar_lt_105':'$M_\star < 10^{10.5} \,$M$_{\!\odot}$',
               'mstar_gt_105':'$M_\star > 10^{10.5} \,$M$_{\!\odot}$',
               'ssfr_lt_n11':'sSFR < 10$^{-11}$ yr$^{-1}$',
               'ssfr_gt_n11':'sSFR > 10$^{-11}$ yr$^{-1}$',
               'ssfr_lt_n11_I':'sSFR < 10$^{-11}$ yr$^{-1}$ (I)',
               'ssfr_lt_n11_NI':'sSFR < 10$^{-11}$ yr$^{-1}$ (NI)',
               'ssfr_gt_n11_I':'sSFR > 10$^{-11}$ yr$^{-1}$ (I)',
               'ssfr_gt_n11_NI':'sSFR > 10$^{-11}$ yr$^{-1}$ (NI)'}

    if config == 'COS-Halos':
        werk13 = werk2013(coveringFractions=True)

        if conf == 0: galaxySets = ['all']
        if conf == 1: galaxySets = ['ssfr_gt_n11','ssfr_lt_n11','mstar_lt_105','mstar_gt_105']
    if config in ['eCGM','eCGMfull']:
        j15 = johnson2015(coveringFractions=True)

        if conf == 0: galaxySets = ['all']
        if conf == 1: galaxySets = ['ssfr_gt_n11','ssfr_lt_n11']
        if conf == 2: galaxySets = ['ssfr_gt_n11_I','ssfr_lt_n11_I','ssfr_gt_n11_NI','ssfr_lt_n11_NI']
    if config == 'SimHalos_115-125':
        galaxySets = ['all']

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    heightFac = 1.0 if ('main' in saveName or len(sPs) == 1) else 0.95
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac*heightFac])
    ax = fig.add_subplot(111)
    
    if config in ['COS-Halos','SimHalos_115-125']:
        ax.set_xlim([0.0, 400.0])
        ax.set_xlabel('Impact Parameter [ pkpc ]')
    if config in ['eCGM','eCGMfull']:
        ax.set_xlim([-1.05, 1.05])
        ax.set_xlabel('Impact Parameter / Virial Radius [ log ]') 

    yLabelExtra = ''
    if len(colDensThresholds) == 1:
        yLabelExtra = ' (N$_{\\rm OVI}$ > 10$^{%.2f}$ cm$^{-2}$)' % colDensThresholds[0]
        if int(colDensThresholds[0]*10)/10.0 == colDensThresholds[0]:
            yLabelExtra = ' (N$_{\\rm OVI}$ > 10$^{%.1f}$ cm$^{-2}$)' % colDensThresholds[0]
        if int(colDensThresholds[0]*1)/1.0 == colDensThresholds[0]:
            yLabelExtra = ' (N$_{\\rm OVI}$ > 10$^{%.0f}$ cm$^{-2}$)' % colDensThresholds[0]
    ax.set_ylabel('Covering Fraction $\kappa_{\\rm OVI}$%s' % yLabelExtra)
    if conf == 0: ax.set_ylim([0.0, 1.04])
    if conf == 1: ax.set_ylim([0.1, 1.04])
    if config in ['eCGM','eCGMfull']: ax.set_ylim([-0.1, 1.04])

    # overplot obs
    colors = []

    if config == 'COS-Halos':
        for j, gs in enumerate( galaxySets ):
            c = 'black' if j == 0 and len(galaxySets) == 1 else ax._get_lines.prop_cycler.next()['color']
            colors.append(c)

            for i in range(len(werk13['rad'])):
                x = np.mean( werk13['rad'][i] ) + [0,4,0,4][j] # horizontal offset for visual clarity
                y = werk13[gs]['cf'][i]
                xerr = (x - werk13['rad'][i][0]) * 0.96 # reduce a few percent for clarity
                yerr = np.array( [werk13[gs]['cf_errdown'][i], werk13[gs]['cf_errup'][i]] )
                yerr = np.reshape( yerr/100.0, (2,1) )

                print(gs,i,x,xerr,y)

                label = ''
                if gs != 'all': label = 'Werk+ (2013) ' + gsNames[gs]
                if gs == 'all':
                    ax.text(124, 0.64, 'Werk+ (2013)', ha='left', size=18)
                    ax.text(124, 0.59, 'N$_{\\rm OVI}$ > 10$^{14.15}$ cm$^{-2}$', ha='left', size=18)
                if i > 0: label = ''
                ax.errorbar(x, y/100.0, xerr=xerr, yerr=yerr, fmt='o', 
                            color=c, markersize=11.0, lw=1.6, capthick=1.6, label=label)

    if config in ['eCGM','eCGMfull']:
        for j, gs in enumerate( galaxySets ):
            c = 'black' if j == 0 and len(galaxySets) == 1 else ax._get_lines.prop_cycler.next()['color']
            colors.append(c)

            if len(j15[gs]) == 0:
                continue # no data points (all)

            for i in range(len(j15[gs]['rad'])):
                x = np.log10( j15[gs]['rad'][i] )
                y = j15[gs]['cf'][i]

                xerr = [x - np.log10( j15[gs]['rad_left'][i] ), np.log10( j15[gs]['rad_right'][i] ) - x]
                xerr = np.reshape( xerr, (2,1) )
                yerr = [y - j15[gs]['cf_down'][i], j15[gs]['cf_up'][i] - y]
                yerr = np.reshape( yerr, (2,1) )

                label = ''
                if gs != 'all': label = 'Johnson+ (2015) ' + gsNames[gs]
                if i > 0: label = ''
                ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', 
                            color=c, markersize=11.0, lw=1.6, capthick=1.6, label=label)

    # loop over each column density threshold (different colors)
    for j, thresh in enumerate(colDensThresholds):

        if len(colDensThresholds) > 1:
            c = ax._get_lines.prop_cycler.next()['color']

        # loop over each fullbox run (different linestyles)
        for i, sP in enumerate(sPs):
            # load
            sim_sample = obsMatchedSample(sP, datasetName=config)
            sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)
            cf = ionCoveringFractions(sP, sim_sample, config=config)

            print('[%s]: %s' % (sP.simName,thresh))

            if len(colDensThresholds) == 1 and len(sPs) > 1:
                c = ax._get_lines.prop_cycler.next()['color']

            # which index for the requested col density threshold?
            assert thresh in cf['colDensThresholds']
            ind = np.where(cf['colDensThresholds'] == thresh)[0]

            relStr = ''
            if config in ['eCGM','eCGMfull']: relStr = '_rel'

            # different galaxy samples?
            for k, gs in enumerate(galaxySets):
                xx = cf['radBins%s' % relStr]
                yy = np.squeeze( cf['%s_percs%s' % (gs,relStr)][ind,:,3] )
                yy_min = np.squeeze( cf['%s_percs%s' % (gs,relStr)][ind,:,2] ) # -half sigma
                yy_max = np.squeeze( cf['%s_percs%s' % (gs,relStr)][ind,:,4] ) # +half sigma
                assert list(cf['perc_vals'][[3,2,4]]) == [50,38,62] # verify as expected

                # plot middle line
                label = 'N > %.2f cm$^{-2}$' % thresh
                if int(thresh*10)/10.0 == thresh: label = 'N > %.1f cm$^{-2}$' % thresh
                if len(colDensThresholds) == 1: label = sP.simName
                if gs != 'all': label += ' (%s)' % gsNames[gs]
                if i > 0 and len(colDensThresholds) > 1: label = ''
                
                if len(galaxySets) > 1:
                    c = colors[k]
                else:
                    c = 'black' if (len(sPs) > 5 and sP.variant == '0000') else c

                ls = linestyles[i] if len(colDensThresholds) > 1 else linestyles[0]
                if len(sPs) > 8 and 'BH' in sP.simName: ls = '--'

                ax.plot(xx, yy, lw=lw, color=c, linestyle=ls, label=label)

                # percentiles
                if i != len(sPs)-1:
                    continue

                ax.fill_between(xx, yy_min, yy_max, color=c, alpha=0.1, interpolate=True)

                # add TNG100-2 (i.e. TNG300-1) line to COS-Halos figure?
                if gs == 'all' and sP.simName == 'TNG100-1' and thresh == 14.15:
                    print('add')
                    sP_ill = simParams(res=910,run='tng',redshift=0.0)
                    sim_sample_ill = obsMatchedSample(sP_ill, datasetName=config)
                    sim_sample_ill = addIonColumnPerSystem(sP_ill, sim_sample_ill, config=config)
                    cf_ill = ionCoveringFractions(sP_ill, sim_sample_ill, config=config)
                    xx = cf_ill['radBins%s' % relStr]
                    yy = np.squeeze( cf_ill['%s_percs%s' % (gs,relStr)][ind,:,3] )
                    ax.plot(xx, yy, lw=lw, linestyle='--', color=c)
                    ax.text(245,0.33,'TNG300',color=c,fontsize=18)
                    ax.text(345,0.29,'TNG100',color=c,fontsize=18)

                #if conf == 1 and sP.simName == 'TNG100-2' and gs in ['mstar_lt_105','ssfr_gt_n11']:
                #    # +/- 1 sigma lines
                #    yy_min = np.squeeze( cf['%s_percs%s' % (gs,relStr)][ind,:,1] ) # -1 sigma
                #    yy_max = np.squeeze( cf['%s_percs%s' % (gs,relStr)][ind,:,5] ) # +1 sigma
                #    assert list(cf['perc_vals'][[1,5]]) == [16,84] # verify as expected
                #    ax.plot(xx, yy_min, '-', lw=lw-1, color=c, linestyle=linestyles[i], alpha=0.2)
                #    ax.plot(xx, yy_max, '-', lw=lw-1, color=c, linestyle=linestyles[i], alpha=0.2)

    # legend
    handles, labels = ax.get_legend_handles_labels()

    if config == 'eCGMfull' and conf == 2:
        prop = {'size':15}        
        legend1 = ax.legend(handles[0:4], labels[0:4], loc='upper right', prop=prop)
        legend2 = ax.legend(handles[4:], labels[4:], loc='lower left', prop=prop)
        ax.add_artist(legend1)
    else:
        if len(sPs) == 13: # main variants, split into 2 legends
            legend1 = ax.legend(handles[0:5], labels[0:5], loc='upper right', prop={'size':18})
            ax.add_artist(legend1)
            legend2 = ax.legend(handles[5:], labels[5:], loc='lower left', prop={'size':18})
        else: # default
            loc = 'upper right' if len(sPs) == 1 else 'lower left'
            prop = {}
            if config in ['eCGM','eCGMfull']: prop['size'] = 15
            legend2 = ax.legend(handles, labels, loc=loc, ncol=1, prop=prop)

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def milkyWaySampleNumbers():
    """ Construct our Milky Way sample, and run some small analysis to print a few interesting numbers. """
    #sP = simParams(res=512,run='tng',redshift=0.0,variant='0000')
    sP = simParams(res=1820,run='tng',redshift=0.0)

    acFieldName = 'Subhalo_StellarRotation_2rhalfstars'
    mhalo = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])
    ssfr  = groupCat(sP, fieldsSubhalos=['ssfr_gyr_log'])
    kappa = auxCat(sP, fields=[acFieldName])[acFieldName][:,1]

    # sample selection
    w = np.where( (mhalo >= 11.85) & (mhalo < 12.28) &\
                  (ssfr >= -1.3) & (ssfr < -0.3) & \
                  (kappa >= 0.6) & (kappa < 1.0) )

    print('Sample size: ', len(w[0]))

    # loop info from auxcats
    fields = ['Subhalo_Mass_250pkpc_Gas','Subhalo_Mass_250pkpc_Stars',
              'Subhalo_Mass_50pkpc_Gas','Subhalo_Mass_50pkpc_Stars',
              'Subhalo_Mass_250pkpc_Gas_Global','Subhalo_Mass_250pkpc_Stars_Global']
    ac = auxCat(sP, fields=fields)

    for field in fields + ['baryon_50','baryon_250','baryon_global_250']:

        if field == 'baryon_50':
            data = ac['Subhalo_Mass_50pkpc_Gas'] + ac['Subhalo_Mass_50pkpc_Stars']
        elif field == 'baryon_250':
            data = ac['Subhalo_Mass_250pkpc_Gas'] + ac['Subhalo_Mass_250pkpc_Stars']
        elif field == 'baryon_global_250':
            data = ac['Subhalo_Mass_250pkpc_Gas_Global'] + ac['Subhalo_Mass_250pkpc_Stars_Global']
        else:
            data = ac[field]

        if data.size != len(w[0]):
            data = data[w] # partial auxCat not already computed just on these subhaloIDs

        percs = np.nanpercentile( data, [16,50,84] )
        percs = sP.units.codeMassToLogMsun(percs)
        print(field,percs)

    import pdb; pdb.set_trace()

def test_lambda_statistic():
    """ Test the behavior of the lambda statistic depending on the sim vs obs draws. """
    # config
    N_sim = 100
    loc = 15.0
    scale = 0.8
    sim_cols = np.random.normal(loc=loc, scale=scale, size=N_sim)
    kde = gaussian_kde(sim_cols, bw_method='scott')

    # obs
    N_obs = 10000
    scale_fac = 0.5

    obs_cols = np.random.normal(loc=loc, scale=scale*scale_fac, size=N_obs) # drawn from same distribution
    #obs_cols = np.zeros( N_obs, dtype='float32' ) + loc # all exact

    #lim_types = np.zeros( N_obs, dtype='int32' ) # all detections (0=detections, 1=upper lim, 2=lower lim)
    lim_types = np.random.randint( low=0, high=3, size=N_obs ) # random assortment of limit types

    lambdas = np.zeros( obs_cols.size, dtype='float32' )

    for i in range(obs_cols.size):
        z1 = kde.integrate_box_1d(-np.inf, obs_cols[i])
        z2 = kde.integrate_box_1d(obs_cols[i], np.inf)

        if lim_types[i] == 0: lambdas[i] = 2*np.min([z1,z2]) # detection, 2*PDF area more extreme
        if lim_types[i] == 1: lambdas[i] = z1 # upper limit, PDF area which is consistent
        if lim_types[i] == 2: lambdas[i] = z2 # lower limit, PDF area which is consistent

    print('mean lambda: ',lambdas.mean())

# -------------------------------------------------------------------------------------------------

variants1 = ['0100','0401','0402','0501','0502','0601','0602','0701','0703','0000']
variants2 = ['0201','0202','0203','0204','0205','0206','0801','0802','1100','0000']
variants3 = ['1000','1002','1003','1004','1005','4302','1200','1301','1302','0000']
variants4 = ['2002','2101','2102','2201','2202','2203','2302','4601','4602','0000']
variants5 = ['3000','3100','3001','3010','3101','3102','3201','3203','3002','0000']
variants6 = ['3403','3404','3501','3502','3401','3402','3601','3602','3901','0000']
variants7 = ['3301','3302','3303','3304','3701','3702','3801','3802','3902','0000']
variants8 = ['4000','4100','4410','4412','4420','4501','4502','4503','4506','0000']
variantSets = [variants1,variants2,variants3,variants4,variants5,variants6,variants7,variants8]

variantsMain = ['0501','0502','0801','2002','2302','2102','2202','3000','3001','3010','3404','0010','0000']

def paperPlots():
    """ Construct all the final plots for the paper. """
    TNG100    = simParams(res=1820,run='tng',redshift=0.0)
    TNG100_2  = simParams(res=910,run='tng',redshift=0.0)
    TNG100_3  = simParams(res=455,run='tng',redshift=0.0)
    TNG300    = simParams(res=2500,run='tng',redshift=0.0)
    TNG300_2  = simParams(res=1250,run='tng',redshift=0.0)
    TNG300_3  = simParams(res=625,run='tng',redshift=0.0)
    Illustris = simParams(res=1820,run='illustris',redshift=0.0)

    ions = ['OVI','OVII','OVIII'] # whenever we are not just doing OVI

    # figure 1, 2: full box composite image components, and full box OVI/OVIII ratio
    if 0:
        from vis.boxDrivers import TNG_oxygenPaperImages
        for part in [3]: #[0,1,2,3]:
            TNG_oxygenPaperImages(part=part)

    # figure 3a: ionization data for OVI, OVII, and OVIII
    if 0:
        element = 'Oxygen'
        ionNums = [6,7,8]
        redshift = 0.0
        metal = -1.0 # log solar

        saveName = 'abundance_fractions_%s_%s_z%d_Z%d.pdf' % \
          (element, '-'.join([str(i) for i in ionNums]),redshift*100,10**metal * 1000)

        ionAbundFracs2DHistos(saveName, element=element, ionNums=ionNums, redshift=redshift, metal=metal)

    # figure 3b: global box phase-diagrams weighted by gas mass in OVI, OVII, and OVIII
    if 0:
        sP = TNG100
        ptType = 'gas'
        xQuant = 'hdens'
        yQuant = 'temp'
        weights = ['O VI mass','O VII mass','O VIII mass']
        xMinMax = [-9.0,0.0]
        yMinMax = [3.0,8.0]
        contours = [-3.0, -2.0, -1.0]
        massFracMinMax = [-4.0, 0.0] #[-10.0, 0.0]
        hideBelow = True
        smoothSigma = 1.0

        plotPhaseSpace2D(sP, ptType, xQuant, yQuant, weights=weights, haloID=None, 
                         clim=massFracMinMax, xlim=xMinMax, ylim=yMinMax, 
                         contours=contours, smoothSigma=smoothSigma, hideBelow=True)

    # figure 4, CDDF of OVI at z~0 compared to observations
    if 0:
        moment = 0
        simRedshift = 0.2
        boxDepth10 = True # use 10 Mpc/h projection depth
        sPs = [TNG100, TNG100_2, TNG100_3, TNG300, Illustris]

        pdf = PdfPages('cddf_ovi_z%02d_moment%d_%s%s.pdf' % \
            (10*simRedshift,moment,'_'.join([sP.simName for sP in sPs]),'_10Mpch' if boxDepth10 else ''))
        nOVIcddf(sPs, pdf, moment=moment, simRedshift=simRedshift, boxDepth10=boxDepth10)
        pdf.close()

    # figure 5, CDDF redshift evolution of multiple ions (combined panel, and individual panels)
    if 0:
        moment = 0
        sPs = [TNG100] #, Illustris]
        boxDepth10 = True
        redshifts = [0,1,2,4]

        saveName = 'cddf_%s_zevo-%s_moment%d_%s.pdf' % \
            ('-'.join(ions), '-'.join(['%d'%z for z in redshifts]), moment, 
             '_'.join([sP.simName for sP in sPs]))
        cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=ions, redshifts=redshifts, 
                              boxDepth10=boxDepth10, colorOff=2)

        for i, ion in enumerate(ions):
            saveName = 'cddf_%s_zevo-%s_moment%d_%s%s.pdf' % \
                (ion, '-'.join(['%d'%z for z in redshifts]), moment, 
                 '_'.join([sP.simName for sP in sPs]),'_10Mpch' if boxDepth10 else '')
            cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=[ion], redshifts=redshifts, 
                                  boxDepth10=boxDepth10, colorOff=i+2)

    # figure 6, CDDF at z=0 with physics variants (L25n512)
    if 0:
        simRedshift = 0.0
        moment = 1

        sPs = []
        for variant in variantsMain:
            sPs.append( simParams(res=512,run='tng',redshift=simRedshift,variant=variant) )

        saveName = 'cddf_ovi_z%02d_moment%d_variants-main.pdf' % (10*simRedshift,moment)
        cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=['OVI'], redshifts=[simRedshift])

    # figure 7: 2pcf
    if 0:
        redshift = 0.0
        sPs = [TNG100] #[TNG100, TNG300]
        ions = ['OVI','OVII','OVIII','O','gas']

        # compute time for one split:
        # TNG100 [days] = (1820^3/256^3)^2 * (1148/60/60/60) * (8*100/nSplits) * (16/nThreads)
        # for nSplits=200000, should finish each in 1.5 days (nThreads=32) (each has 60,000 cells)
        # for TNG300, nSplits=500000, should finish each in 4 days (nThreads=32)
        for order in [0,1,2]:
            saveName = 'tpcf_order%d_%s_%s_z%02d.pdf' % \
              (order,'-'.join(ions),'_'.join([sP.simName for sP in sPs]),redshift)

            oxygenTwoPointCorrelation(sPs, saveName, ions=ions, redshift=redshift, order=order, colorOff=2)

    # figure 8, bound mass of O ions vs halo/stellar mass
    if 0:
        sPs = [TNG300, TNG100]
        cenSatSelect = 'cen'
        redshift = 0.0
        ionsLoc = ['AllGas','AllGas_Metal','AllGas_Oxygen'] + ions

        for vsHaloMass in [True,False]:
            massStr = '%smass' % ['stellar','halo'][vsHaloMass]

            saveName = 'ions_masses_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ionsLoc, cenSatSelect=cenSatSelect, 
                redshift=redshift, vsHaloMass=vsHaloMass, secondTopAxis=True)

            saveName = 'ions_avgcoldens_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, cenSatSelect=cenSatSelect, 
                redshift=redshift, vsHaloMass=vsHaloMass, toAvgColDens=True, secondTopAxis=True)

    # figure 9: average radial profiles
    if 0:
        redshift = 0.0
        sPs = [TNG100]
        ions = ['OVI'] # OVII, OVIII
        cenSatSelect = 'cen'
        haloMassBins = [[10.9,11.1], [11.4,11.6], [11.9,12.1], [12.4,12.6]]
        projSpecs = ['2Dz_2Mpc'] #['3D','2Dz_2Mpc']
        combine2Halo = False

        simNames = '_'.join([sP.simName for sP in sPs])

        for massDensity in [False]: #[True,False]:
            for radRelToVirRad in [False]: #[True,False]:
                for projDim in projSpecs:

                    saveName = 'radprofiles_%s_%s_%s_z%02d_%s_rho%d_rvir%d.pdf' % \
                      (projDim,'-'.join(ions),simNames,redshift,cenSatSelect,massDensity,radRelToVirRad)
                    stackedRadialProfiles(sPs, saveName, ions=ions, redshift=redshift, massDensity=massDensity,
                                          radRelToVirRad=radRelToVirRad, cenSatSelect='cen', projDim=projDim, 
                                          haloMassBins=haloMassBins, combine2Halo=combine2Halo)


    # figure 10: mock COS-Halos samples
    if 0:
        sPs = [TNG100, TNG300]

        simNames = '_'.join([sP.simName for sP in sPs])
        obsSimMatchedGalaxySamples(sPs, 'coshalos_sample_%s.pdf' % simNames, config='COS-Halos')

    # figure 11: COS-Halos: N_OVI vs impact parameter and vs sSFR bimodality
    if 0:
        sP = TNG100

        #cosOVIDataPlot(sP, saveName='coshalos_ovi_%s.pdf' % sP.simName, config='COS-Halos')
        cosOVIDataPlotExtended(sP, saveName='coshalos_ovi_%s_ext.pdf' % sP.simName, config='COS-Halos')
        
    # figure 12: covering fractions, OVI vs obs (all galaxies, and subsamples)
    if 0:
        sPs = [TNG100] #, TNG100_2]

        # All Galaxies
        novi_vals = [13.5, 14.0, 14.15, 14.5, 15.0]
        saveName = 'coshalos_covering_frac_%s.pdf' % '_'.join([sP.simName for sP in sPs])
        coveringFractionVsDist(sPs, saveName, ions=['OVI'], colDensThresholds=novi_vals, conf=0)

        # sSFR / M* subsets
        novi_vals = [14.15]
        saveName = 'coshalos_covering_frac_subsets_%s.pdf' % '_'.join([sP.simName for sP in sPs])
        coveringFractionVsDist(sPs, saveName, ions=['OVI'], colDensThresholds=novi_vals, conf=1)


    # figure 13: nums 11-14 repeated for the eCGM dataset instead of COS-Halos
    if 0:
        sP = TNG100
        cf = 'eCGMfull' # eCGM

        obsSimMatchedGalaxySamples([sP], '%s_sample_%s.pdf' % (cf,sP.simName), config=cf)
        cosOVIDataPlot(sP, saveName='%s_ovi_%s.pdf' % (cf,sP.simName), config=cf)
        cosOVIDataPlotExtended(sP, saveName='%s_ovi_%s_ext.pdf' % (cf,sP.simName), config=cf)

        coveringFractionVsDist([sP], '%s_covering_frac_%s.pdf' % (cf,sP.simName), ions=['OVI'], 
            colDensThresholds=[13.5, 14.0, 14.5, 15.0], config=cf, conf=0)
        for conf in [1,2]:
            coveringFractionVsDist([sP], '%s_covering_frac_%s_conf%d.pdf' % (cf,sP.simName,conf), 
                ions=['OVI'], colDensThresholds=[13.5], config=cf, conf=conf)

    # figure 14: covering fractions, with main physics variants (L25n512)
    if 0:
        novi_vals = [14.0]

        sPs = []
        for variant in variantsMain:
            sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

        saveName = 'covering_frac_ovi_variants-main.pdf'
        coveringFractionVsDist(sPs, saveName, ions=['OVI'], colDensThresholds=novi_vals, 
                               config='SimHalos_115-125')

    # figure 15, 16: OVI red/blue image samples
    if 0:
        from vis.haloDrivers import tngFlagship_galaxyStellarRedBlue
        tngFlagship_galaxyStellarRedBlue(evo=False, redSample=1, conf=1)
        tngFlagship_galaxyStellarRedBlue(evo=False, blueSample=1, conf=1)

    # figure 17: OVI vs color at fixed stellar/halo mass
    if 0:
        sPs = [TNG100, TNG300]
        css = 'cen'
        quant = 'mass_ovi'
        xQuant = 'color_C_gr'

        for iter in [0,1]:
            if iter == 0:
                sQuant = 'mstar_30pkpc_log'
                sRange = [10.4,10.6]
            if iter == 1:
                sQuant = 'mhalo_200_log'
                sRange = [12.0, 12.1]

            pdf = PdfPages('slice_%s_%s_%s-%.1f-%.1f_%s.pdf' % \
                ('_'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))
            quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, 
                         sRange=sRange, cenSatSelect=css)
            pdf.close()

    # figure 18, 19, 20: 2d histos
    if 0:
        sP = TNG300
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        cQuant = 'mass_ovi'

        yQuants1 = ['ssfr','Z_gas','fgas2','size_gas','temp_halo_volwt','mass_z']
        yQuants2 = ['surfdens1_stars','Z_stars','color_C_gr','size_stars','Krot_oriented_stars2','Krot_oriented_gas2']
        yQuants3 = ['nh_halo_volwt','fgas_r200','pratio_halo_volwt','BH_CumEgy_low','M_BH_actual','_dummy_']

        yQuantSets = [yQuants1, yQuants2, yQuants3]

        for i, xQuant in enumerate(xQuants):
            yQuants3[-1] = xQuants[1-i] # include the other

            for j, yQuants in enumerate(yQuantSets):
                params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant}

                pdf = PdfPages('histo2d_x=%s_set-%d_%sb.pdf' % (xQuant,j,sP.simName))
                fig = plt.figure(figsize=figsize_loc)
                quantHisto2D(sP, pdf, yQuant=yQuants[0], fig_subplot=[fig,321], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[1], fig_subplot=[fig,322], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[2], fig_subplot=[fig,323], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[3], fig_subplot=[fig,324], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[4], fig_subplot=[fig,325], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[5], fig_subplot=[fig,326], **params)
                pdf.close()

    # ------------ appendix ---------------

    # figure A1, all CDDFs at z=0 with physics variants (L25n512)
    if 0:
        simRedshift = 0.0
        moment = 1

        for i, variants in enumerate(variantSets):
            sPs = []
            for variant in variants:
                sPs.append( simParams(res=512,run='tng',redshift=simRedshift,variant=variant) )

            saveName = 'cddf_ovi_z%02d_moment%d_variants-%d.pdf' % (10*simRedshift,moment,i)
            cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=['OVI'], redshifts=[simRedshift])

    # figure 14: all covering fractions with physics variants (L25n512)
    if 0:
        novi_vals = [14.0]

        for i, variants in enumerate(variantSets):
            sPs = []
            for variant in variants:
                sPs.append( simParams(res=512,run='tng',redshift=0.0,variant=variant) )

            saveName = 'covering_frac_ovi_variants-%d.pdf' % (i)
            coveringFractionVsDist(sPs, saveName, ions=['OVI'], colDensThresholds=novi_vals, 
                                   config='SimHalos_115-125')

    # ------------ exploration ------------

    # exploration: OVI average column vs everything at fixed stellar/halo mass
    if 0:
        sPs = [TNG100, TNG300]
        css = 'cen'
        quant = 'mass_ovi'
        xQuants = quantList(wCounts=False, wTr=False, wMasses=True)

        for iter in [0,1]:
            if iter == 0:
                sQuant = 'mstar_30pkpc_log'
                sRange = [10.4,10.6]
            if iter == 1:
                sQuant = 'mhalo_200_log'
                sRange = [12.0, 12.1]
            
            pdf = PdfPages('slices_%s_x=all_%s-%.1f-%.1f_%s.pdf' % \
                ('_'.join([sP.simName for sP in sPs]),sQuant,sRange[0],sRange[1],css))
            for xQuant in xQuants:
                quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, 
                             sRange=sRange, cenSatSelect=css)
            pdf.close()

    # exploration: median OVI column vs stellar/halo mass, split by everything else
    if 0:
        sPs = [TNG300]
        simNames = '-'.join([sP.simName for sP in sPs])

        css = 'cen'
        quants = quantList(wCounts=False, wTr=False, wMasses=True)
        priQuant = 'mass_ovi'
        sLowerPercs = [10,50]
        sUpperPercs = [90,50]

        for xQuant in ['mstar_30pkpc','mhalo_200_log']:
            # individual plot per y-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_slice=%s.pdf' % (simNames,xQuant,css,priQuant))
            for yQuant in quants:
                quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=priQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)
            pdf.close()

            # individual plot per s-quantity:
            pdf = PdfPages('medianTrends_%s_x=%s_%s_y=%s.pdf' % (simNames,xQuant,css,priQuant))
            for sQuant in quants:
                quantMedianVsSecondQuant(sPs, pdf, yQuants=[priQuant], xQuant=xQuant, cenSatSelect=css,
                                         sQuant=sQuant, sLowerPercs=sLowerPercs, sUpperPercs=sUpperPercs)

            pdf.close()

    # exploration: OVI vs everything else in the median
    if 0:
        sPs = [TNG100, TNG300]
        simNames = '-'.join([sP.simName for sP in sPs])

        css = 'cen'
        xQuants = quantList(wCounts=False, wTr=False, wMasses=True)
        yQuant = 'mass_ovi'

        rQuant = 'mhalo_200_log' # only include systems satisfying this restriction
        rRange = [11.0, 16.0] # restriction range

        # individual plot per y-quantity:
        pdf = PdfPages('medianTrends_%s_y=%s_vs-all%d_%s_%s_in_%.1f-%.1f.pdf' % \
            (simNames,yQuant,len(xQuants),css,rQuant,rRange[0],rRange[1]))
        for xQuant in xQuants:
            quantSlice1D(sPs, pdf, xQuant=xQuant, yQuants=[yQuant], sQuant=rQuant, 
                         sRange=rRange, cenSatSelect=css)
            # for most quantities, is dominated everywhere by low-mass halos with very small mass_ovi:
            #quantMedianVsSecondQuant(sPs, pdf, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css)
        pdf.close()

    # exploration: cloudy ionization table
    if 0:
        from cosmo.cloudy import plotIonAbundances
        plotIonAbundances(elements=['Oxygen'])

    # figure 4: testing toy model for total mass in different ions
    if 0:
        sPs = [TNG300]
        cenSatSelect = 'cen'
        ionsLoc = ['AllGas','AllGas_Metal','AllGas_Oxygen'] + ions

        toyFacsList = [ [1.0,1.0,1.0], [0.6,1.5,1.5], [0.8,1.0,1.5], [1.0,1.0,1.5], [1.0,1.0,2.0],
                        [0.6,1.0,1.5], [0.6,2.0,1.5], [1.5,1.0,1.5], [2.0,1.0,1.5] ]

        for toyFacs in toyFacsList:
            toyStr = 'toy=%.1f_%.1f_%.1f' % (toyFacs[0],toyFacs[1],toyFacs[2])
            saveName = 'ions_masses_cen_z0_%s_%s.pdf' % \
                ('_'.join([sP.simName for sP in sPs]),toyStr)
            totalIonMassVsHaloMass(sPs, saveName, ions=ionsLoc, cenSatSelect='cen', 
                redshift=0.0, vsHaloMass=True, secondTopAxis=True, toyFacs=toyFacs)
