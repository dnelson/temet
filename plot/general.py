"""
general.py
  General exploratory/diagnostic plots of single halos or entire boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter

import illustris_python as il
from util import simParams
from util.helper import loadColorTable, running_median
from cosmo.load import groupCat, groupCatSingle, auxCat, snapshotSubset
from cosmo.util import periodicDists

def plotPhaseSpace2D(yAxis):
    """ Plot a 2D phase space plot (gas density on x-axis), for a single halo or for an entire box. """
    assert yAxis in ['temp','P_B','P_tot','P_tot_dens','sfr','mass_sfr_dt','mass_sfr_dt_hydro','dt_yr']

    sP = simParams(res=2160, run='tng', redshift=6.0)
    haloID = 0 # None for fullbox

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('%s z=%.1f %s' % (sP.simName,sP.redshift,hStr))
    ax.set_xlabel('Gas Density [ log cm$^{-3}$ ]')

    # load
    dens = snapshotSubset(sP, 'gas', 'dens', haloID=haloID)
    dens = sP.units.codeDensToPhys(dens, cgs=True, numDens=True)
    ###dens = sP.units.codeDensToCritRatio(dens, baryon=True, log=False)
    dens = np.log10(dens)

    xMinMax = [-8.0,9.0]
    #xMinMax = [-2.0,8.0]

    mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)

    if yAxis == 'temp':
        yvals = snapshotSubset(sP, 'gas', 'temp', haloID=haloID)
        ax.set_ylabel('Gas Temperature [ log K ]')
        yMinMax = [2.0, 8.0]

    if yAxis == 'P_B':
        yvals = snapshotSubset(sP, 'gas', 'P_B', haloID=haloID)
        ax.set_ylabel('Gas Magnetic Pressure [ log K cm$^{-3}$ ]')
        yMinMax = [-15.0, 16.0]

    if yAxis == 'P_tot':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        ax.set_ylabel('Gas Total Pressure [ log K cm$^{-3}$ ]')
        yMinMax = [-15.0, 16.0]

    if yAxis == 'P_tot_dens':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        yvals = np.log10( 10.0**yvals/10.0**dens )
        ax.set_ylabel('Gas Total Pressure / Gas Density [ log arbitrary units ]')
        yMinMax = [2.0, 10.0]

    if yAxis == 'sfr':
        yvals = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        yvals = np.log10( yvals )
        ax.set_ylabel('Star Formation Rate [ log M$_\odot$ / yr ]')
        yMinMax = [-5.0, 1.0]

    if yAxis == 'mass_sfr_dt':
        mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)
        mass = sP.units.codeMassToMsun(mass)
        sfr  = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        dt   = snapshotSubset(sP, 'gas', 'TimeStep', haloID=haloID)

        dt_yr = sP.units.codeTimeStepToYears(dt)
        yvals = np.log10( mass / sfr / dt_yr )

        ax.set_ylabel('Gas Mass / SFR / Timestep [ log dimensionless ]')
        yMinMax = [-2.0,5.0]

    if yAxis == 'mass_sfr_dt_hydro':
        mass = snapshotSubset(sP, 'gas', 'mass', haloID=haloID)
        mass = sP.units.codeMassToMsun(mass)
        sfr  = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)

        soundspeed = snapshotSubset(sP, 'gas', 'soundspeed', haloID=haloID)
        cellrad = snapshotSubset(sP, 'gas', 'cellrad', haloID=haloID)
        cellrad_kpc = sP.units.codeLengthToKpc(cellrad)
        cellrad_km  = cellrad_kpc * sP.units.kpc_in_km

        dt_hydro_s = 0.3 * cellrad_km / soundspeed
        dt_hydro_yr = dt_hydro_s / sP.units.s_in_yr
        yvals = np.log10( mass / sfr / dt_hydro_yr )

        ax.set_ylabel('Gas Mass / SFR / HydroTimestep [ log dimensionless ]')
        yMinMax = [-2.0,5.0]

    if yAxis == 'dt_yr':
        dt = snapshotSubset(sP, 'gas', 'TimeStep', haloID=haloID)
        yvals = np.log10( sP.units.codeTimeStepToYears(dt) )

        ax.set_ylabel('Gas Timestep [ log yr ]')
        yMinMax = [1.0,6.0]

    nBinsX = 800
    nBinsY = 400

    # plot
    zz, xc, yc = np.histogram2d(dens, yvals, bins=[nBinsX, nBinsY], range=[xMinMax,yMinMax], 
                                normed=True, weights=mass)

    zz = np.transpose(zz)
    zz = np.log10(zz)

    cmap = loadColorTable('viridis')
    plt.imshow(zz, extent=[xMinMax[0],xMinMax[1],yMinMax[0],yMinMax[1]], 
               cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

    # colorbar and save
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('Relative Gas Mass [ log ]')

    fig.savefig('phase_%s_%s_%s.pdf' % (sP.simName,yAxis,hStr))
    plt.close(fig)

def plotRadialProfile1D(quant='Potential'):
    """ Quick radial profile of some quantity vs. radius (FoF restricted). """
    sP = simParams(res=2160, run='tng', redshift=6.0)
    haloID = 0

    nBins = 200
    valMinMax = [-3.0,2.5]

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    ax.set_title('%s z=%.1f halo%d' % (sP.simName,sP.redshift,haloID))
    ax.set_xlabel('Radius [ log pkpc ]')
    ax.set_xlim(valMinMax)

    # load
    halo = groupCatSingle(sP, haloID=haloID)
    pos = snapshotSubset(sP, 'gas', 'pos', haloID=haloID)

    rad = periodicDists(halo['GroupPos'], pos, sP)
    rad = sP.units.codeLengthToKpc(rad)
    rad = np.log10(rad)

    # load quant
    if quant == 'P_gas':
        yvals = snapshotSubset(sP, 'gas', 'P_gas', haloID=haloID)
        ax.set_ylabel('Gas Pressure [ log K cm$^{-3}$ ]')
        ax.set_ylim([1.0,13.0])

    if quant == 'dens':
        yvals = snapshotSubset(sP, 'gas', 'dens', haloID=haloID)
        yvals = sP.units.codeDensToPhys(yvals, cgs=True, numDens=True)
        yvals = np.log10(yvals)
        ax.set_ylabel('Gas Density [ log cm$^{-3}$ ]')

    if quant == 'P_tot':
        yvals = snapshotSubset(sP, 'gas', 'P_tot', haloID=haloID)
        ax.set_ylabel('Gas Total Pressure [ log K cm$^{-3}$ ]')
        ax.set_ylim([1.0,13.0])

    if quant == 'Potential':
        yvals = snapshotSubset(sP, 'gas', 'Potential', haloID=haloID)
        yvals *= sP.units.scalefac
        ax.set_ylabel('Gravitational Potential [ (km/s)$^2$ ]')

    if quant == 'sfr':
        yvals = snapshotSubset(sP, 'gas', 'sfr', haloID=haloID)
        ax.set_ylabel('Star Formation Rate [ Msun/yr ]')
        ax.set_yscale('log')

    # plot radial profile of quant
    yy_mean = np.zeros( nBins, dtype='float32' ) + np.nan
    yy_med  = np.zeros( nBins, dtype='float32' ) + np.nan
    xx      = np.zeros( nBins, dtype='float32' )

    binSize = (valMinMax[1]-valMinMax[0])/nBins

    for i in range(nBins):
        binStart = valMinMax[0] + i*binSize
        binEnd   = valMinMax[0] + (i+1)*binSize

        ww = np.where((rad >= binStart) & (rad < binEnd))
        xx[i] = (binStart+binEnd)/2.0

        if len(ww[0]) > 0:
            yy_mean[i] = np.mean(yvals[ww])
            yy_med[i]  = np.median(yvals[ww])

    ax.plot(xx, yy_med, label='median')
    ax.plot(xx, yy_mean, label='mean')

    # finish plot
    ax.legend(loc='best')
    fig.savefig('radProfile_%s_halo%d.pdf' % (quant,haloID))
    plt.close(fig)

def bFieldStrengthComparison():
    """ Plot histogram of B field magnitude comparing runs etc. """
    sPs = []

    haloID = None # None for fullbox
    redshift = 0.5
    nBins = 100
    valMinMax = [-7.0,4.0]

    sPs.append( simParams(res=1820, run='tng', redshift=redshift) )
    sPs.append( simParams(res=910, run='tng', redshift=redshift) )
    sPs.append( simParams(res=455, run='tng', redshift=redshift) )

    # start plot
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111)

    hStr = 'fullbox' if haloID is None else 'halo%d' % haloID
    ax.set_title('z=%.1f %s' % (redshift,hStr))
    ax.set_xlim(valMinMax)
    ax.set_xlabel('Magnetic Field Magnitude [ log $\mu$G ]')
    ax.set_ylabel('N$_{\\rm cells}$ PDF $\int=1$')
    ax.set_yscale('log')

    for sP in sPs:
        # load
        b_mag = snapshotSubset(sP, 'gas', 'bmag', haloID=haloID)
        b_mag *= 1e6 # Gauss to micro-Gauss
        b_mag = np.log10(b_mag) # log uG

        # add to plot
        yy, xx = np.histogram(b_mag, bins=nBins, density=True, range=valMinMax)
        xx = xx[:-1] + 0.5*(valMinMax[1]-valMinMax[0])/nBins

        ax.plot(xx, yy, label=sP.simName)

    # finish plot
    ax.legend(loc='best')

    fig.savefig('bFieldStrengthComparison_%s.pdf' % hStr)
    plt.close(fig)

def depletionVsDynamicalTimescale():
    """ Andi: depletion vs dynamical timescale.
      t_dep = M_H2/SFR   M_H2 the cold, star-forming gas or take total gas mass instead
      t_dyn = r12 / v_rot  r12 the half mass radius of the gaseous disk, v_rot its characteristic rot. vel
    """

    # config
    figsize = (14,9)
    sP = simParams(res=1820,run='illustris',redshift=0.0)

    gc = groupCat(sP, fieldsHalos=['GroupFirstSub'], 
                      fieldsSubhalos=['SubhaloHalfmassRadType','SubhaloVmax','SubhaloSFR'])
    ac = auxCat(sP, fields=['Subhalo_Mass_SFingGas','Subhalo_Mass_30pkpc_Stars'])

    # t_dep [Gyr]
    M_cold = sP.units.codeMassToMsun(ac['Subhalo_Mass_SFingGas'])
    SFR = gc['subhalos']['SubhaloSFR'] # Msun/yr
    t_dep = M_cold / SFR / 1e9

    # t_dyn [Gyr]
    r12 = sP.units.codeLengthToKpc(gc['subhalos']['SubhaloHalfmassRadType'][:,sP.ptNum('stars')])
    v_rot = gc['subhalos']['SubhaloVmax'] * sP.units.kmS_in_kpcGyr
    t_dyn = r12 / v_rot

    # stellar masses and central selection
    m_star = sP.units.codeMassToLogMsun(ac['Subhalo_Mass_30pkpc_Stars'])

    w_central = np.where( gc['halos'] >= 0 )
    
    centralsMask = np.zeros( gc['subhalos']['count'], dtype=np.int16 )
    centralsMask[gc['halos'][w_central]] = 1

    centrals = np.where(centralsMask & (SFR > 0.0) & (r12 > 0.0))

    t_dep = t_dep[centrals]
    t_dyn = t_dyn[centrals]
    m_star = m_star[centrals]

    # plot config
    title = sP.simName + ' z=%.1f' % sP.redshift + ' [only centrals with SFR>0 and r12>0]'
    tDynMinMax = [0,0.2]
    tDepMinMax = [0,4]
    mStarMinMax = [9.0,12.0]
    ratioMinMax = [0,0.05] # tdyn/tdep
    nBinsX = 200
    nBinsY = 150
    binSizeMed = 0.01

    # (A) 2d histogram of t_dep vs. t_dyn for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(tDynMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('t$_{\\rm dyn}$ [Gyr]')
        ax.set_ylabel('t$_{\\rm dep}$ [Gyr]')

        # 2d histo
        zz, xc, yc = np.histogram2d(t_dyn, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[tDynMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[tDynMinMax[0],tDynMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        #xm, ym, sm = running_median(t_dyn,t_dep,binSize=binSizeMed)
        #ym2 = savgol_filter(ym,3,2)
        #sm2 = savgol_filter(sm,3,2)
        #ax.plot(xm[:-1], ym2[:-1], '-', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='black', lw=2.0)
        #ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_a.pdf' % sP.simName)
        plt.close(fig)

    # (B) 2d histogram of ratio (t_dep/t_dyn) vs. m_star for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(ratioMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dyn}$ / t$_{\\rm dep}$')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dyn/t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,ratioMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],ratioMinMax[0],ratioMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dyn/t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_b.pdf' % sP.simName)
        plt.close(fig)

    # (C) t_dep vs m_star
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel('M$_{\\rm star}$ [ log M$_\odot$ ]')
        ax.set_ylabel('t$_{\\rm dep}$ [ Gyr ]')

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dep, bins=[nBinsX, nBinsY], 
                                    range=[mStarMinMax,tDepMinMax], normed=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable('viridis')
        plt.imshow(zz, extent=[mStarMinMax[0],mStarMinMax[1],tDepMinMax[0],tDepMinMax[1]], 
                   cmap=cmap, origin='lower', interpolation='nearest', aspect='auto')

        # median
        xm, ym, sm = running_median(m_star,t_dep,binSize=binSizeMed*10)
        ym2 = savgol_filter(ym,3,2)
        sm2 = savgol_filter(sm,3,2)
        ax.plot(xm[:-3], ym2[:-3], '-', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]+sm2[:-3], ':', color='black', lw=2.0)
        ax.plot(xm[:-3], ym2[:-3]-sm2[:-3], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Number of Galaxies [ log ]')

        fig.tight_layout()
        fig.savefig('tdyn_vs_tdep_%s_c.pdf' % sP.simName)
        plt.close(fig)