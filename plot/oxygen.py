"""
plot/oxygen.py
  Plots: OVI paper.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from os.path import isfile

from util import simParams
from plot.config import *
from util.helper import running_median, logZeroNaN, iterable
from cosmo.load import groupCat, groupCatSingle, auxCat
from cosmo.cloudy import cloudyIon
from plot.general import simSubhaloQuantity, getWhiteBlackColors, bandMagRange
from plot.cosmoGeneral import quantHisto2D, quantSlice1D, quantMedianVsSecondQuant
from vis.common import setAxisColors
from cosmo.util import cenSatSubhaloIndices

def nOVIcddf(sPs, pdf, moment=0, simRedshift=0.2):
    """ CDDF (column density distribution function) of O VI in the whole box at z~0.
        (Schaye Fig 17) (Suresh+ 2016 Fig 11) """
    from util.loadExtern import danforth2008, danforth2016, thomChen2008, tripp2008

    # config
    speciesList = ['nOVI','nOVI_solar','nOVI_10','nOVI_25']

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

def cddfRedshiftEvolution(sPs, pdf, moment=0, ions=['OVI','OVII'], redshifts=[0,1,2,3]):
    """ CDDF (column density distribution function) of O VI in the whole box.
        (Schaye Fig 17) (Suresh+ 2016 Fig 11) """

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
    ax = fig.add_subplot(111)
    
    ax.set_xlim([12.5, 16.5])
    ax.set_xlabel('N$_{\\rm oxygen}$ [ log cm$^{-2}$ ]')

    if moment == 0:
        ax.set_ylim([-18, -11])
        ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm oxygen}$)  [ cm$^{2}$ ]')
        if clean:
            ax.set_ylabel('f(N$_{\\rm oxygen}$) [ log cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-2.5, 1.5])
        ax.set_ylabel('CDDF (1$^{\\rm st}$ moment):  log N$_{\\rm oxygen}$ f(N$_{\\rm oxygen}$)')
        if clean:
            ax.set_ylabel('N$_{\\rm oxygen}$ $\cdot$ f(N$_{\\rm oxygen}$) [ log ]')

    # loop over each fullbox run
    for sP in sPs:
        for ion in ions:
            print('[%s]: %s' % (ion,sP.simName))
            c = ax._get_lines.prop_cycler.next()['color']

            for i, redshift in enumerate(redshifts):
                sP.setRedshift(redshift)

                # pre-computed CDDF: load at this redshift
                ac = auxCat(sP, fields=['Box_CDDF_n'+ion])
                n_ion  = ac['Box_CDDF_n'+ion][0,:]
                fN_ion = ac['Box_CDDF_n'+ion][1,:]

                xx = np.log10(n_ion)

                if moment == 0:
                    yy = logZeroNaN( fN_ion )
                if moment == 1:
                    yy = logZeroNaN( fN_ion*n_ion )

                # plot middle line
                label = '%s %s z=%.1f' % (sP.simName, ion, redshift)
                if clean: label = '%s' % (ion)
                if i > 0: label = ''
                ax.plot(xx, yy, lw=lw, color=c, linestyle=linestyles[i], label=label)

    # legend
    sExtra = []
    lExtra = []

    if clean:
        for i, redshift in enumerate(redshifts):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['z = %3.1f' % redshift]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper right')

    fig.tight_layout()
    pdf.savefig()
    plt.close(fig)

def totalIonMassVsHaloMass(sPs, saveName, ions=['OVI','OVII'], redshift=0.0, cenSatSelect='cen', 
                           vsHaloMass=True, toAvgColDens=False):
    """ Plot total [gravitationally bound] mass of various ions, or e.g. cold/hot/total CGM mass, 
    versus halo or stellar mass at a given redshift. If toAvgColDens, then instead of total mass 
    plot average column density computed geometrically as (Mtotal/pi/rvir^2). """

    binSize = 0.2 # log mass

    # plot setup
    lw = 3.0
    sizefac = 1.0 if not clean else sfclean
    fig = plt.figure(figsize=[figsize[0]*sizefac, figsize[1]*sizefac])
    ax = fig.add_subplot(111)
    
    if vsHaloMass:
        ax.set_xlim([11.0, 15.0])
        ax.set_xlabel('M$_{\\rm halo}$ [ < r$_{\\rm 200,crit}$, log M$_{\\rm sun}$ ]')
        massField = 'mhalo_200_log'
    else:
        ax.set_xlim([9.0, 12.0])
        ax.set_xlabel('M$_{\star}$ [ < 30 pkpc, log M$_{\\rm sun}$ ]')
        massField = 'mstar_30pkpc_log'

    if toAvgColDens:
        ax.set_ylim([12.0, 16.0])
        ax.set_ylabel('Average Column Density $<N_{\\rm oxygen}>$ [ log cm$^2$ ]')
    else:
        ax.set_ylim([5.0, 9.0])
        ax.set_ylabel('Total Bound Gas Mass [ log M$_{\\rm sun}$ ]')

    # loop over each fullbox run
    colors = []

    for i, sP in enumerate(sPs):
        sP.setRedshift(redshift)

        # load halo masses and CSS
        xx = groupCat(sP, fieldsSubhalos=[massField])

        cssInds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
        xx = xx[cssInds]

        if toAvgColDens:
            # load virial radii
            rad = groupCat(sP, fieldsSubhalos=['rhalo_200_code'])
            rad = rad[cssInds]
            ionData = cloudyIon(None)

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
            
            # calculate median and smooth
            xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,
                                            skipZeros=True,percs=[10,25,75,90], minNumPerBin=10)

            if xm.size > sKn:
                ym = savgol_filter(ym,sKn,sKo)
                sm = savgol_filter(sm,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1) # P[10,90]

            # plot middle line
            if i == 0:
                c = ax._get_lines.prop_cycler.next()['color']
                colors.append(c)
            else:
                c = colors[j]

            label = ion if i == 0 else ''
            ax.plot(xm, ym, lw=lw, color=c, linestyle=linestyles[i], label=label)

            if i == 0:
                # show percentile scatter only for 'all galaxies'
                ax.fill_between(xm, pm[0,:], pm[-1,:], color=c, interpolate=True, alpha=0.2)

    # legend
    sExtra = []
    lExtra = []

    if clean:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['%s' % sP.simName]

    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc='upper left')

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def paperPlots():
    """ Construct all the final plots for the paper. """
    TNG100 = simParams(res=1820,run='tng',redshift=0.0)
    TNG100_2 = simParams(res=910,run='tng',redshift=0.0)
    TNG100_3 = simParams(res=455,run='tng',redshift=0.0)
    TNG300 = simParams(res=2500,run='tng',redshift=0.0)
    Illustris = simParams(res=1820,run='illustris',redshift=0.0)

    ions = ['OVI','OVII','OVIII'] # whenever we are not just doing OVI

    # figure 1a, bound mass of O ions vs halo/stellar mass
    if 0:
        sPs = [TNG100_2, TNG100] # [TNG300, TNG100] #, TNG100_3]
        cenSatSelect = 'cen'
        redshift = 0.0

        saveName = 'ionmasses_vs_halomass_%s_z%d_%s.pdf' % \
            (cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
        totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                               cenSatSelect=cenSatSelect, vsHaloMass=True)

        saveName = 'ionmasses_vs_stellarmass_%s_z%d_%s.pdf' % \
            (cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
        totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                               cenSatSelect=cenSatSelect, vsHaloMass=False)

        saveName = 'avgcoldens_vs_halomass_%s_z%d_%s.pdf' % \
            (cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
        totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                               cenSatSelect=cenSatSelect, vsHaloMass=True, toAvgColDens=True)

        saveName = 'avgcoldens_vs_stellarmass_%s_z%d_%s.pdf' % \
            (cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
        totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                               cenSatSelect=cenSatSelect, vsHaloMass=False, toAvgColDens=True)

    # figure 2a, CDDF of OVI at z~0 compared to observations
    if 0:
        moment = 0
        simRedshift = 0.2
        sPs = [TNG100, Illustris, TNG300]

        pdf = PdfPages('cddf_ovi_z%02d_moment%d_%s.pdf' % \
            (moment,10*simRedshift,'_'.join([sP.simName for sP in sPs])))
        nOVIcddf(sPs, pdf, moment=moment, simRedshift=simRedshift)
        pdf.close()

    # figure 2b, CDDF redshift evolution of multiple ions
    if 1:
        moment = 0
        sPs = [TNG100, Illustris]
        redshifts = [0,1,2,4]

        pdf = PdfPages('cddf_%s_zevo-%s_moment%d_%s.pdf' % \
            ('-'.join(ions), '-'.join(['%d'%z for z in redshifts]), moment, 
             '_'.join([sP.simName for sP in sPs])))
        cddfRedshiftEvolution(sPs, pdf, moment=moment, ions=ions, redshifts=redshifts)
        pdf.close()
