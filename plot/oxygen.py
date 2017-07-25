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
from plot.general import simSubhaloQuantity, getWhiteBlackColors, bandMagRange, quantList
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

def cddfRedshiftEvolution(sPs, saveName, moment=0, ions=['OVI','OVII'], redshifts=[0,1,2,3], colorOff=0):
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
        ax.set_ylim([-18, -12])
        if len(ions) == 1: ax.set_ylim([-19, -11])
        ax.set_ylabel('CDDF (O$^{\\rm th}$ moment):  log f(N$_{\\rm oxygen}$)  [ cm$^{2}$ ]')
        if clean:
            ax.set_ylabel('f(N$_{\\rm oxygen}$) [ log cm$^{2}$ ]')
    if moment == 1:
        ax.set_ylim([-2.5, 1.5])
        ax.set_ylabel('CDDF (1$^{\\rm st}$ moment):  log N$_{\\rm oxygen}$ f(N$_{\\rm oxygen}$)')
        if clean:
            ax.set_ylabel('N$_{\\rm oxygen}$ $\cdot$ f(N$_{\\rm oxygen}$) [ log ]')
    if moment == 2:
        ax.set_ylim([-1.5, 2.2])
        ax.set_ylabel('CDDF (2$^{\\rm nd}$ moment):  log N$_{\\rm oxygen}^2$ f(N$_{\\rm oxygen}$)')
        if clean:
            ax.set_ylabel('[N$_{\\rm oxygen}$$^2$ / 10$^{13}$] $\cdot$ f(N$_{\\rm oxygen}$) [ log ]') 

    # loop over each fullbox run
    for sP in sPs:
        for j, ion in enumerate(ions):
            print('[%s]: %s' % (ion,sP.simName))

            if j == 0:
                for _ in range(colorOff+1):
                    c = ax._get_lines.prop_cycler.next()['color']
            else:
                c = ax._get_lines.prop_cycler.next()['color']

            for i, redshift in enumerate(redshifts):
                sP.setRedshift(redshift)

                # pre-computed CDDF: load at this redshift
                ac = auxCat(sP, fields=['Box_CDDF_n'+ion])
                N_ion  = ac['Box_CDDF_n'+ion][0,:]
                fN_ion = ac['Box_CDDF_n'+ion][1,:]

                xx = np.log10(N_ion)

                if moment == 0:
                    yy = logZeroNaN( fN_ion )
                if moment == 1:
                    yy = logZeroNaN( fN_ion*N_ion )
                if moment == 2:
                    yy = logZeroNaN( fN_ion*N_ion*(N_ion/1e13) )

                # plot middle line
                label = '%s %s z=%.1f' % (sP.simName, ion, redshift)
                if clean: label = ion
                if clean and len(ions) == 1: label = sP.simName
                if i > 0: label = ''
                c = 'black' if (len(sPs) > 5 and sP.variant == '0000') else c

                ax.plot(xx, yy, lw=lw, color=c, linestyle=linestyles[i], label=label)

    # legend
    sExtra = []
    lExtra = []

    if clean and len(redshifts) > 1:
        for i, redshift in enumerate(redshifts):
            sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
            lExtra += ['z = %3.1f' % redshift]

    handles, labels = ax.get_legend_handles_labels()
    loc = 'upper right' if len(ions) > 1 else 'lower left'
    legend2 = ax.legend(handles+sExtra, labels+lExtra, loc=loc)

    fig.tight_layout()
    fig.savefig(saveName)
    plt.close(fig)

def totalIonMassVsHaloMass(sPs, saveName, ions=['OVI','OVII'], redshift=0.0, cenSatSelect='cen', 
                           vsHaloMass=True, toAvgColDens=False, colorOff=2):
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

            # determine color
            if i == 0:
                if j == 0:
                    for _ in range(colorOff+1):
                        c = ax._get_lines.prop_cycler.next()['color']
                else:
                    c = ax._get_lines.prop_cycler.next()['color']
                colors.append(c)
            else:
                c = colors[j]

            # plot median line
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
        sPs = [TNG300, TNG100] #, TNG100_3]
        cenSatSelect = 'cen'
        redshift = 0.0

        for vsHaloMass in [True,False]:
            massStr = '%smass' % ['stellar','halo'][vsHaloMass]

            saveName = 'ions_masses_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                                   cenSatSelect=cenSatSelect, vsHaloMass=vsHaloMass)

            saveName = 'ions_avgcoldens_vs_%s_%s_z%d_%s.pdf' % \
                (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, redshift=redshift, 
                                   cenSatSelect=cenSatSelect, vsHaloMass=vsHaloMass, toAvgColDens=True)

    # figure 2a, CDDF of OVI at z~0 compared to observations
    if 0:
        moment = 0
        simRedshift = 0.2
        sPs = [TNG100, Illustris] #, TNG300]

        pdf = PdfPages('cddf_ovi_z%02d_moment%d_%s.pdf' % \
            (10*simRedshift,moment,'_'.join([sP.simName for sP in sPs])))
        nOVIcddf(sPs, pdf, moment=moment, simRedshift=simRedshift)
        pdf.close()

    # figure 2b, CDDF redshift evolution of multiple ions (combined panel, and individual panels)
    if 0:
        moment = 0
        sPs = [TNG100] #, Illustris]
        redshifts = [0,1,2,4]

        saveName = 'cddf_%s_zevo-%s_moment%d_%s.pdf' % \
            ('-'.join(ions), '-'.join(['%d'%z for z in redshifts]), moment, 
             '_'.join([sP.simName for sP in sPs]))
        cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=ions, redshifts=redshifts, colorOff=2)

        for i, ion in enumerate(ions):
            saveName = 'cddf_%s_zevo-%s_moment%d_%s.pdf' % \
                (ion, '-'.join(['%d'%z for z in redshifts]), moment, 
                 '_'.join([sP.simName for sP in sPs]))
            cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=[ion], redshifts=redshifts, colorOff=i+2)

    # figure 3, CDDF at z=0 with physics variants (L25n512)
    if 0:
        simRedshift = 0.0
        moment = 2
        variants1 = ['0100','0401','0402','0501','0502','0601','0602','0701','0703','0000']
        variants2 = ['0201','0202','0203','0204','0205','0206','0801','0802','1100','0000']
        variants3 = ['1000','1001','1002','1003','1004','1005','2002','2101','2102','0000']
        variants4 = ['2201','2202','2203','3000','3001','3010','3101','3102','3201','0000'] # 2302
        variantSets = [variants1, variants2, variants3, variants4]

        for i, variants in enumerate(variantSets):
            sPs = []
            for variant in variants:
                sPs.append( simParams(res=512,run='tng',redshift=simRedshift,variant=variant) )

            saveName = 'cddf_ovi_z%02d_moment%d_variants-%d.pdf' % (10*simRedshift,moment,i)
            cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=['OVI'], redshifts=[simRedshift])

    # figure 4: average radial profiles
    if 0:
        pass

    # figure 5: OVI red/blue image samples
    if 0:
        from vis.haloDrivers import tngFlagship_galaxyStellarRedBlue
        tngFlagship_galaxyStellarRedBlue(evo=False, redSample=1, conf=1)
        tngFlagship_galaxyStellarRedBlue(evo=False, blueSample=1, conf=1)

    # figure 6: OVI vs color at fixed stellar/halo mass
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

            for xQuant in xQuants:
                pdf = PdfPages('slice_%s_%s_%s-%.1f-%.1f_%s.pdf' % \
                    ('_'.join([sP.simName for sP in sPs]),xQuant,sQuant,sRange[0],sRange[1],css))
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

    # exploration: 2D OVI
    if 0:
        sP = TNG300
        figsize_loc = [figsize[0]*2*0.7, figsize[1]*3*0.7]
        xQuants = ['mstar_30pkpc_log','mhalo_200_log']
        cQuant = 'mass_ovi'

        yQuants1 = ['ssfr','surfdens1_stars','fgas2','stellarage','bmag_2rhalf_masswt','pratio_halo_masswt']
        yQuants2 = ['Z_gas','Z_stars','Krot_oriented_stars2','Krot_oriented_gas2','size_gas','size_stars']
        yQuants3 = ['color_C_gr','xray_r500','zform_mm5','BH_CumEgy_low','M_BH_actual','filled_below']

        yQuantSets = [yQuants1, yQuants2, yQuants3]

        for i, xQuant in enumerate(xQuants):
            yQuants3[-1] = xQuants[1-i] # include the other

            for j, yQuants in enumerate(yQuantSets):
                params = {'cenSatSelect':'cen', 'cStatistic':'median_nan', 'cQuant':cQuant, 'xQuant':xQuant}

                pdf = PdfPages('histo2d_x=%s_set-%d_%s.pdf' % (xQuant,j,sP.simName))
                fig = plt.figure(figsize=figsize_loc)
                quantHisto2D(sP, pdf, yQuant=yQuants[0], fig_subplot=[fig,321], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[1], fig_subplot=[fig,322], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[2], fig_subplot=[fig,323], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[3], fig_subplot=[fig,324], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[4], fig_subplot=[fig,325], **params)
                quantHisto2D(sP, pdf, yQuant=yQuants[5], fig_subplot=[fig,326], **params)
                pdf.close()
