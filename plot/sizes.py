"""
sizes.py
  Galaxy sizes, half mass and half light radii.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import savgol_filter
from collections import OrderedDict

from util import simParams
from util.helper import running_median, logZeroNaN
from plot.config import *

def galaxySizes(sPs, pdf, vsHaloMass=False, simRedshift=0.0, fig_subplot=[None,None], addHalfLightRad=None):
    """ Galaxy sizes (half mass radii) vs stellar mass or halo mass, at redshift zero. 
    If addHalfLightRad is not None, then addHalfLightRad = [dustModel,band,show3D] e.g.
    addHalfLightRad = ['p07c_cf00dust_res_conv_efr_rad30pkpc','sdss_r',False]. """
    from util.loadExtern import baldry2012SizeMass, shen2003SizeMass, lange2016SizeMass

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])

    ax.set_ylim([0.3,1e2])

    ylabel = 'Galaxy Size [ kpc ]'
    if not clean: ylabel += ' [ r$_{\\rm 1/2, stars/gas}$ ] [ only centrals ]'
    if clean: ylabel += ' [ Halfmass Radius ]'
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    if vsHaloMass:
        ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([9,14.5])
        ax.set_ylim([0.7,8e2])
    else:
        xlabel = 'Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]'
        if not clean: xlabel += ' [ < 2r$_{1/2}$ ]'
        ax.set_xlabel(xlabel)
        #ax.set_xlim( behrooziSMHM(sPs[0], logHaloMass=np.array(ax.get_xlim())) )
        ax.set_xlim([7,12.0])
        if clean: ax.set_xlim([8.0,12.0])

    # observational points
    if not vsHaloMass:
        b = baldry2012SizeMass()
        s = shen2003SizeMass()
        l = lange2016SizeMass()

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

        l5, = ax.plot(l['stellarMass2'], l['hubbletype']['E_gt2e10']['sizeKpc'], '--', color='#777777')
        l6, = ax.plot(l['stellarMass'], l['combined']['all_discs']['sizeKpc'], '--', color='#333333')

        legend1 = ax.legend([l1,l2,l3,l4,l5,l6], 
          [ b['red']['label'], b['blue']['label'], 
            s['early']['label'], s['late']['label'],
            l['hubbletype']['E_gt2e10']['label'], l['combined']['all_discs']['label'] ], 
          loc='upper left') # lower right
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('Sizes: '+sP.simName)
        sP.setRedshift(simRedshift)

        gc = sP.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMassInRadType','SubhaloHalfmassRadType'])

        # centrals only
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        # x-axis: mass definition
        if vsHaloMass:
            xx_code = gc['halos']['Group_M_Crit200'][wHalo]
        else:
            xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]

        xx = sP.units.codeMassToLogMsun( xx_code )

        # sizes
        yy_gas   = gc['subhalos']['SubhaloHalfmassRadType'][w,sP.ptNum('gas')]
        yy_gas   = sP.units.codeLengthToKpc( yy_gas )
        yy_stars = gc['subhalos']['SubhaloHalfmassRadType'][w,sP.ptNum('stars')]
        yy_stars = sP.units.codeLengthToKpc( yy_stars )

        if addHalfLightRad is not None:
            # load auxCat half light radii
            acField = 'Subhalo_HalfLightRad_' + addHalfLightRad[0]
            ac = sP.auxCat(fields=[acField])
            assert addHalfLightRad[1] in ac[acField+'_attrs']['bands']
            bandNum = list(ac[acField+'_attrs']['bands']).index( addHalfLightRad[1] )

            # hard-coded structure of these files for now
            assert ac[acField].shape[2] in [6,10]

            if ac[acField].shape[2] == 6:
                # non-resolved (models A,B)
                Re_labels = ['edge-on','face-on','edge-on-smallest','edge-on-random','z-axis','3d']
            if ac[acField].shape[2] == 10:
                # resolved (model C): even 3d radii are projection dependent
                Re_labels = ['edge-on 2d','face-on 2d','edge-on-smallest 2d','edge-on-random 2d','z-axis 2d',
                             'edge-on 3d','face-on 3d','edge-on-smallest 3d','edge-on-random 3d','z-axis 3d']
                if addHalfLightRad[2]:
                    Re_labels = Re_labels[5:]
                    ac[acField] = ac[acField][:,:,5:]
                else:
                    Re_labels = Re_labels[0:5]
                    ac[acField] = ac[acField][:,:,:5]

            # split by each projection
            yy_stars_Re = []
            
            for i in range(ac[acField].shape[2]):
                yy_stars_Re.append( sP.units.codeLengthToKpc(ac[acField][w,bandNum,i]) )         

        # if plotting vs halo mass, restrict our attention to those galaxies with sizes (e.g. nonzero 
        # number of either gas cells or star particles)
        if vsHaloMass:
            ww = np.where( (yy_gas > 0.0) & (yy_stars > 0.0) )
            yy_gas = yy_gas[ww]
            yy_stars = yy_stars[ww]
            xx = xx[ww]

        if vsHaloMass and addHalfLightRad is not None:
            for i in range(ac[acField].shape[2]):
                yy_stars_Re[i] = yy_stars_Re[i][ww]

        # take median vs mass and smooth
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

        label = sP.simName
        if sP.redshift > 0.0: label += ' z=%.1f' % sP.redshift
        l, = ax.plot(xm_stars[1:-1], ym_stars[1:-1], linestyles[0], lw=3.0, label=label)

        if not clean:
            l, = ax.plot(xm_gas[1:-1], ym_gas[1:-1], linestyles[1], color=l.get_color(), lw=3.0)

        if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2):
            y_down = np.array(ym_stars[1:-1]) - sm_stars[1:-1]
            y_up   = np.array(ym_stars[1:-1]) + sm_stars[1:-1]
            ax.fill_between(xm_stars[1:-1], y_down, y_up, 
                            color=l.get_color(), interpolate=True, alpha=0.3)

        # add all of the half-light radii size measurements, one line per projection
        if addHalfLightRad is not None:
            for i in range(ac[acField].shape[2]):
                xm_stars, ym_stars, sm_stars = running_median(xx,yy_stars_Re[i],binSize=binSize,skipZeros=True)
                ww_stars = np.where(ym_stars > 0.0)
                ym_stars = savgol_filter(ym_stars[ww_stars],sKn,sKo)
                sm_stars = savgol_filter(sm_stars[ww_stars],sKn,sKo)
                xm_stars = xm_stars[ww_stars]

                l, = ax.plot(xm_stars[1:-1], ym_stars[1:-1], linestyles[0], lw=3.0, 
                             label='R$_{\\rm e}$ '+addHalfLightRad[1]+' '+Re_labels[i])

                if i == 0:
                    y_down = np.array(ym_stars[1:-1]) - sm_stars[1:-1]
                    y_up   = np.array(ym_stars[1:-1]) + sm_stars[1:-1]
                    ax.fill_between(xm_stars[1:-1], y_down, y_up, 
                            color=l.get_color(), interpolate=True, alpha=0.3)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = []
    lExtra = []

    if not clean:
        sExtra = [plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[0]),
                  plt.Line2D( (0,1), (0,0), color='black', marker='', lw=3.0, linestyle=linestyles[1])]
        lExtra = [r'stars',r'gas']

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

def galaxyHISizeMass(sPs, pdf, simRedshift=0.0, fig_subplot=[None,None]):
    """ Galaxy HI size-mass relation, at redshift zero. """

    # plot setup
    if fig_subplot[0] is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    else:
        # add requested subplot to existing figure
        fig = fig_subplot[0]
        ax = fig.add_subplot(fig_subplot[1])

    cenSatSelect = 'cen'
    ylabel = 'D$_{\\rm HI}$ [ log kpc ]'
    if clean: ylabel += ' [ Halfmass Radius ] [ only centrals ]'
    ax.set_ylabel(ylabel)

    xlabel = 'M$_{\\rm HI}$ [ log M$_{\\rm sun}$ ]'
    if not clean: xlabel += ' [ < 2r$_{1/2}$ ]'
    ax.set_xlabel(xlabel)

    ax.set_ylim([-0.7, 2.5])
    ax.set_xlim([5.0, 11.5])

    # observational points (Wang, J.+ 2016)
    # todo
    xx_obs = np.array([5.0, 11.5])
    yy_obs = 0.506 * xx_obs - 3.293 # log pkpc

    ax.plot(xx_obs, yy_obs, '-', color='black', label='Wang+ (2016)')
    ax.plot(xx_obs, yy_obs + 0.06*3, ':', color='black') # 3sigma
    ax.plot(xx_obs, yy_obs - 0.06*3, ':', color='black') # 3sigma

    # loop over each fullbox run
    for sP in sPs:
        if sP.isZoom:
            continue

        print('HI Sizes: '+sP.simName)
        sP.setRedshift(simRedshift)

        # centrals only
        w = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

        # x-axis: mass definition
        fieldName = 'Subhalo_Mass_100pkpc_HI'
        ac = sP.auxCat(fieldName)[fieldName]
        xx = sP.units.codeMassToLogMsun( ac[w] )

        # sizes
        fieldName = 'Subhalo_Gas_HI_HalfRad'
        ac = sP.auxCat(fieldName)[fieldName]
        yy = logZeroNaN( sP.units.codeLengthToKpc( ac[w] ) )

        # take median vs mass and smooth
        xm, ym, sm, pm = running_median(xx,yy,binSize=binSize,percs=[5,16,50,84,95],skipZeros=True)

        if xm.size > sKn:
            ym = savgol_filter(ym,sKn,sKo)
            sm = savgol_filter(sm,sKn,sKo)
            pm = savgol_filter(pm,sKn,sKo,axis=1)

        label = sP.simName
        if sP.redshift > 0.0: label += ' z=%.1f' % sP.redshift
        l, = ax.plot(xm, ym, linestyles[0], lw=lw, label=label)

        if ((len(sPs) > 2 and sP == sPs[0]) or len(sPs) <= 2):
            ax.fill_between(xm, pm[0,:], pm[-1,:], color=l.get_color(), interpolate=True, alpha=0.05)
            ax.fill_between(xm, pm[1,:], pm[-2,:], color=l.get_color(), interpolate=True, alpha=0.25)
            #ax.fill_between(xm[1:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.3)

    # second legend
    handles, labels = ax.get_legend_handles_labels()
    legend2 = ax.legend(handles, labels, loc='lower right')

    # finish figure
    finishFlag = False
    if fig_subplot[0] is not None: # add_subplot(abc)
        digits = [int(digit) for digit in str(fig_subplot[1])]
        if digits[2] == digits[0] * digits[1]: finishFlag = True

    if fig_subplot[0] is None or finishFlag:
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

def sizeModelsRatios():
    """ Look at calculated HalfLightRadii, plot ratios to evaluate impact of dust. """
    sP = simParams(res=1820,run='tng',redshift=0.0)

    acPre = 'Subhalo_HalfLightRad_p07c_'
    acField1 = 'nodust_efr'
    acField2 = 'cf00dust_efr' #'cf00dust_res_conv_efr' #

    # non-resolved (models A,B)
    Re_labels_nonres = ['edge-on 2d','face-on 2d','edge-on-smallest 2d','edge-on-random 2d','z-axis 2d','3d']

    # resolved (model C): even 3d radii are projection dependent
    Re_labels_res = ['edge-on 2d','face-on 2d','edge-on-smallest 2d','edge-on-random 2d','z-axis 2d',
                     'edge-on 3d','face-on 3d','edge-on-smallest 3d','edge-on-random 3d','z-axis 3d']

    # load auxCat
    ac = sP.auxCat(fields=[acPre+acField1,acPre+acField2])
    bands = list(ac[acPre+acField1+'_attrs']['bands'])

    # load groupCat
    gc = sP.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
        fieldsSubhalos=['SubhaloMassInRadType','SubhaloHalfmassRadType'])

    # centrals only, x-axis mass definition, calculate sizes
    wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
    w = gc['halos']['GroupFirstSub'][wHalo]

    xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]
    xx = sP.units.codeMassToLogMsun( xx_code )

    # which half light radii do we have?
    if ac[acPre+acField1].shape[2] == 6:  Re_labels_1 = Re_labels_nonres
    if ac[acPre+acField1].shape[2] == 10: Re_labels_1 = Re_labels_res
    if ac[acPre+acField2].shape[2] == 6:  Re_labels_2 = Re_labels_nonres
    if ac[acPre+acField2].shape[2] == 10: Re_labels_2 = Re_labels_res

    # start pdf
    pdf = PdfPages('HalfLightRadii_%s_z%.1f_%s.pdf' % (sP.simName,sP.redshift,datetime.now().strftime('%d-%m-%Y')))

    for band in bands:
        # start plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title('%s z=%.1f band=%s' % (sP.simName,sP.redshift,band))

        xlabel = 'Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]' + ' [ < 2r$_{1/2}$ ]'
        ax.set_xlabel(xlabel)
        ax.set_xlim([7.0,12.5])

        ylabel = 'Galaxy Size Ratio'
        ax.set_ylabel(ylabel)
        #ax.set_yscale('log')

        # calculate
        bandNum1 = list(ac[acPre+acField1+'_attrs']['bands']).index( band )
        bandNum2 = list(ac[acPre+acField2+'_attrs']['bands']).index( band )
        assert bandNum1 == bandNum2

        sizes1 = np.squeeze( ac[acPre+acField1][:,bandNum1,:] )
        sizes2 = np.squeeze( ac[acPre+acField2][:,bandNum2,:] )

        # plot auxCat: dust / nodust
        for i in range(len(Re_labels_2)):
            # no-dust index: to corresponding projection for 2d, or to last for 3d (only 1, since in 
            # the no-dust case the 3d half light radii does not depend on projection)
            ind1 = i if i < 5 else -1
            ind2 = i # we are iterating directly over the 10 res_conv radii outputs

            sizes1_loc = np.squeeze( sizes1[w,ind1] )
            sizes2_loc = np.squeeze( sizes2[w,ind2] )
            label1 = Re_labels_1[ind1]
            label2 = Re_labels_2[ind2]

            ratio = sizes2_loc / sizes1_loc

            print(band,i,ind1,ind2,label1,label2,ratio[0:3])
            assert ratio.shape == xx.shape

            xm_stars, ym_stars, sm_stars = running_median(xx,ratio,binSize=binSize,skipZeros=True)
            ym_stars = savgol_filter(ym_stars,sKn,sKo)

            label = '(%s, %s) / (%s, %s)' % (acField2,label2,acField1,label1)
            l, = ax.plot(xm_stars[:-1], ym_stars[:-1], linestyles[0], lw=3.0, label=label)

        # plot auxCat: dust3d projections / dust3d first
        for i in range(6,10):
            ind1 = 5 # compared to the first
            ind2 = i # we are iterating directly over the 5 3d res_conv radii outputs

            if ind2 >= sizes2.shape[1]:
                continue # e.g. model B

            sizes1_loc = np.squeeze( sizes2[w,ind1] )
            sizes2_loc = np.squeeze( sizes2[w,ind2] )
            label1 = Re_labels_2[ind1]
            label2 = Re_labels_2[ind2]

            ratio = sizes2_loc / sizes1_loc

            print(band,i,ind1,ind2,label1,label2,ratio[0:3])
            assert ratio.shape == xx.shape

            xm_stars, ym_stars, sm_stars = running_median(xx,ratio,binSize=binSize,skipZeros=True)

            label = '(%s, %s) / (%s, %s)' % (acField2,label2,acField1,label1)
            l, = ax.plot(xm_stars[:-1], ym_stars[:-1], linestyles[1], lw=3.0, label=label)

        # finish
        ax.legend(loc='best',prop={'size':13})
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

    # close pdf
    pdf.close()

def lumModelsRatios(res=1820, run='tng', redshifts=[0.0]):
    """ Look at calculated StellarPhot, plot ratios to evaluate impact of dust. """
    acPre = 'Subhalo_StellarPhot_p07c_'
    acField1 = 'nodust'
    acField2 = 'cf00dust_res_conv_ns1' # or 'cf00dust', but no rad restriction (unless also in nodust)

    # start pdf
    sP = simParams(res=res,run=run,redshift=redshifts[0])
    pdf = PdfPages('StellarPhotRatios_%s_%s_%s.pdf' % (sP.simName,acField2,datetime.now().strftime('%d-%m-%Y')))

    for redshift in redshifts:
        sP = simParams(res=res,run=run,redshift=redshift)

        # load auxCat
        ac = sP.auxCat(fields=[acPre+acField1,acPre+acField2])
        bands = list(ac[acPre+acField1+'_attrs']['bands'])
        nProj = ac[acPre+acField2].shape[2] if ac[acPre+acField2].ndim == 3 else 1

        # load groupCat
        gc = sP.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
            fieldsSubhalos=['SubhaloMassInRadType','SubhaloHalfmassRadType'])

        # centrals only, x-axis mass definition, calculate sizes
        wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0))
        w = gc['halos']['GroupFirstSub'][wHalo]

        xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]
        xx = sP.units.codeMassToLogMsun( xx_code )

        # start plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_title('%s z=%.1f' % (sP.simName,sP.redshift))

        xlabel = 'Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]' + ' [ < 2r$_{1/2}$ ]'
        ax.set_xlabel(xlabel)
        ax.set_xlim([7.0,12.5])

        ylabel = 'Band-Luminosity Ratio'
        ax.set_ylabel(ylabel)
        
        for band in bands:
            # in this band
            c = next(ax._get_lines.prop_cycler)['color']

            bandNum1 = list(ac[acPre+acField1+'_attrs']['bands']).index( band )
            bandNum2 = list(ac[acPre+acField2+'_attrs']['bands']).index( band )
            assert bandNum1 == bandNum2

            mags1 = np.squeeze( ac[acPre+acField1][w,bandNum1] )
            lums1 = sP.units.absMagToLuminosity(mags1)

            print(sP.redshift,band)

            for projNum in range(nProj):
                # calculate ratio and plot
                if ac[acPre+acField2].ndim == 3:
                    mags2 = np.squeeze( ac[acPre+acField2][w,bandNum2,projNum] )
                else:
                    mags2 = np.squeeze( ac[acPre+acField2][w,bandNum2] )

                lums2 = sP.units.absMagToLuminosity(mags2)

                ratio = lums2 / lums1
                assert ratio.shape == xx.shape

                xm_stars, ym_stars, sm_stars = running_median(xx,ratio,binSize=binSize,skipZeros=True)

                label = '%s / %s (%s)' % (acField2,acField1,band) if projNum == 0 else ''
                lw = 3.0 if projNum == 0 else 1.0
                alpha = 1.0 if projNum == 0 else 0.2
                l, = ax.plot(xm_stars[:-1], ym_stars[:-1], linestyles[0], 
                             lw=lw, color=c, alpha=alpha, label=label)

        # finish page (for one redshift)
        ax.legend(loc='best',prop={'size':13})
        fig.tight_layout()
        pdf.savefig()
        plt.close(fig)

    # close pdf
    pdf.close()

def clumpSizes(sP):
    """ Galaxy sizes of the very small things vs stellar mass or halo mass, at redshift zero. """
    from util.loadExtern import baldry2012SizeMass, shen2003SizeMass, lange2016SizeMass
    from cosmo.util import cenSatSubhaloIndices

    centralsOnly = False
    vsMstarXaxis = True

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_title(sP.simName + ' z=%.1f' % sP.redshift)

    ylabel = 'Subhalo Size [ kpc ] [ r$_{\\rm 1/2, stars}$ ]'
    if centralsOnly: ylabel += ' [ only centrals ]'
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    ax.set_xlabel('Parent Group Mass [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
    ax.set_xlim([8,14.5])
    ax.set_ylim([0.1,10])

    # load
    gc = sP.groupCat(fieldsHalos=['GroupFirstSub','Group_M_Crit200'],
        fieldsSubhalos=['SubhaloMassInRadType','SubhaloHalfmassRadType','SubhaloGrNr'])

    # centrals only?
    inds_cen, inds_all, _ = cenSatSubhaloIndices(sP, gc=gc)

    if centralsOnly:
        w = inds_cen
    else:
        w = inds_all
    wHalo = gc['subhalos']['SubhaloGrNr'][w]

    # x-axis: mass definition
    xx_code = gc['halos']['Group_M_Crit200'][wHalo]
    xx = sP.units.codeMassToLogMsun( xx_code )

    if vsMstarXaxis:
        xx_code = gc['subhalos']['SubhaloMassInRadType'][w,sP.ptNum('stars')]
        xx = sP.units.codeMassToLogMsun( xx_code )
        ax.set_xlim([6.0,11.5])
        ax.set_xlabel('Subhalo Stellar Mass [ log M$_{\\rm sun}$ ] [ <2r$_{\\rm 1/2, stars}$ ]')

    # sizes
    yy_stars = gc['subhalos']['SubhaloHalfmassRadType'][w,sP.ptNum('stars')]
    yy_stars = sP.units.codeLengthToKpc( yy_stars )

    # plot (xx, yy_gas) and (xx, yy_stars)
    ax.plot(xx, yy_stars, '.', alpha=0.7, label=sP.simName)

    # plot median
    ww = np.where(yy_stars > 0.0)
    xx = xx[ww]
    yy_stars = yy_stars[ww]

    xm_stars, ym_stars, _ = running_median(xx,yy_stars,binSize=binSize,skipZeros=True)

    ax.plot(xm_stars[1:-1], ym_stars[1:-1], '-', lw=3.0, label=sP.simName)

    # finish figure
    fig.tight_layout()
    plt.savefig('sizes_diagnostic_cenOnly=%s_vsMstar=%s_%s_z=%.1f.png' % (centralsOnly,vsMstarXaxis,sP.simName,sP.redshift))
    plt.close(fig)

def characteristicSizes(sP, vsHaloMass=False):
    """ Compare many different 'characteristic' halo/galaxy sizes as a function of mass. """
    from util.loadExtern import baldry2012SizeMass, shen2003SizeMass, lange2016SizeMass

    reBand = 'jwst_f115w' # for half light radii

    labels = {'stars'    : 'r$_{\\rm 1/2,\star}$',
              'dm'       : 'r$_{\\rm 1/2,DM}$',
              'gas'      : 'r$_{\\rm 1/2,gas}$',
              'gas_sf'   : 'R$_{\\rm SF,H\\alpha}$',
              'gas_hi'   : 'R$_{\\rm HI}$',
              'stars_re' : 'R$_{\\rm e,stars}$',
              'rvir'     : 'r$_{\\rm vir,halo}$'}

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_ylim([0.2,4e2])

    ylabel = 'Galaxy or Halo Size [ kpc ]'
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')

    if vsHaloMass:
        ax.set_xlabel('Halo Mass [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200c}$ ]')
        ax.set_xlim([10.0,14.0])
        ax.set_ylim([0.7,8e2])
    else:
        ax.set_xlabel('Galaxy Stellar Mass [ log M$_{\\rm sun}$ ]')
        ax.set_xlim([6.5,11.5])

    # observational points
    if not vsHaloMass:
        b = baldry2012SizeMass()

        l1,_,_ = ax.errorbar(b['red']['stellarMass'], b['red']['sizeKpc'], 
                             yerr=[b['red']['errorDown'],b['red']['errorUp']],
                             color='#777777', ecolor='#777777', alpha=0.8, capsize=0.0, fmt='D')
        l2,_,_ = ax.errorbar(b['blue']['stellarMass'], b['blue']['sizeKpc'], 
                             yerr=[b['blue']['errorDown'],b['blue']['errorUp']],
                             color='#444444', ecolor='#444444', alpha=0.8, capsize=0.0, fmt='o')

        legend1 = ax.legend([l1,l2], [ b['red']['label'], b['blue']['label'] ], loc='lower right')
        ax.add_artist(legend1)

    # sim: load, select centrals only
    gc = sP.groupCat(fieldsSubhalos=['central_flag','SubhaloHalfmassRadType'])
    w = np.where(gc['central_flag'])

    # x-axis: mass definition
    if vsHaloMass:
        xx = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])[w]
    else:
        xx = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log'])[w]

    yy = OrderedDict()

    # gas half mass radii
    yy['gas'] = gc['SubhaloHalfmassRadType'][:,sP.ptNum('gas')]
    yy['gas'] = sP.units.codeLengthToKpc( yy['gas'][w] )

    # dark matter
    #yy['dm'] = gc['SubhaloHalfmassRadType'][:,sP.ptNum('dm')]
    #yy['dm'] = sP.units.codeLengthToKpc( yy['dm'][w] )

    # stellar half mass radii
    yy['stars'] = gc['SubhaloHalfmassRadType'][:,sP.ptNum('stars')]
    yy['stars'] = sP.units.codeLengthToKpc( yy['stars'][w] )

    # halo virial radii
    yy['rvir'] = sP.groupCat(fieldsSubhalos=['rhalo_200'])[w] # r200,crit [pkpc]

    # stellar half light radii
    fieldName = 'Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc'
    ac = sP.auxCat(fieldName)
    bandInd = list(ac[fieldName + '_attrs']['bands']).index(reBand)
    yy['stars_re'] = ac[fieldName][:,bandInd] # code units
    yy['stars_re'] = sP.units.codeLengthToKpc( yy['stars_re'][w] )     

    # gas halpha half light radii (half SFR radii)
    fieldName = 'Subhalo_Gas_SFR_HalfRad'
    ac = sP.auxCat(fieldName)[fieldName]
    yy['gas_sf'] = sP.units.codeLengthToKpc( ac[w] )
    sf_nan = np.where(np.isnan(yy['gas_sf']))
    yy['gas_sf'][sf_nan] = 0.0 # convention, filtered below

    # gas HI radii
    fieldName = 'Subhalo_Gas_HI_HalfRad'
    ac = sP.auxCat(fieldName)[fieldName]
    yy['gas_hi'] = sP.units.codeLengthToKpc( ac[w] )
    sf_nan = np.where(np.isnan(yy['gas_hi']))
    yy['gas_hi'][sf_nan] = 0.0 # convention, filtered below

    # if plotting vs halo mass, restrict our attention to those galaxies with sizes (e.g. nonzero 
    # number of either gas cells or star particles)
    if vsHaloMass:
        ww = np.where( (yy['gas'] > 0.0) & (yy['stars'] > 0.0) )
        xx = xx[ww]
        for key in yy:
            yy[key] = yy[key][ww]

    # loop over size types
    for key in yy:
        # take median vs mass and smooth
        xm, ym, sm, pm = running_median(xx, yy[key], binSize=binSize, skipZeros=True, percs=[16,84])

        ww = np.where(ym > 0.0)
        xm = xm[ww]

        ym = savgol_filter(ym[ww], sKn, sKo)
        sm = savgol_filter(sm[ww], sKn, sKo)
        pm = savgol_filter(pm[:,ww[0]], sKn, sKo, axis=1)

        # plot median
        l, = ax.plot(xm, ym, linestyles[0], lw=lw, label=labels[key])

        # band
        y_down = np.array(ym) - sm
        y_up   = np.array(ym) + sm
        ax.fill_between(xm, pm[0,:], pm[1,:], color=l.get_color(), interpolate=True, alpha=0.3)

    ax.legend(loc='upper left', ncol=2)

    # finish figure
    fig.tight_layout()
    fig.savefig('characteristic_sizes_%s_%d.pdf' % (sP.simName,sP.snap))
    plt.close(fig)
