"""
cosmoGeneral.py
  General and misc plots for cosmological boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from os.path import isfile
from scipy.signal import savgol_filter

from util import simParams
from util.helper import running_median
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat

def addRedshiftAxis(ax, sP, zVals=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,10.0]):
    """ Add a redshift axis as a second x-axis on top (assuming bottom axis is Age of Universe [Gyr]). """
    axTop = ax.twiny()
    axTickVals = sP.units.redshiftToAgeFlat( np.array(zVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(zVals)
    axTop.set_xlabel("Redshift")

def addUniverseAgeAxis(ax, sP, ageVals=[0.7,1.0,1.5,2.0,3.0,4.0,6.0,9.0]):
    """ Add a age of the universe [Gyr] axis as a second x-axis on top (assuming bottom is redshift). """
    axTop = ax.twiny()

    ageVals.append( sP.units.redshiftToAgeFlat([0.0]).round(2) )
    axTickVals = sP.units.ageFlatToRedshift( np.array(ageVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(ageVals)
    axTop.set_xlabel("Age of the Universe [Gyr]")

def addRedshiftAgeAxes(ax, sP, xrange=[-1e-4,8.0], xlog=True):
    """ Add bottom vs. redshift (and top vs. universe age) axis for standard X vs. redshift plots. """
    ax.set_xlim(xrange)
    ax.set_xlabel('Redshift')

    if xlog:
        ax.set_xscale('symlog')
        zVals = [0,0.5,1,1.5,2,3,4,5,6,7,8] # [10]
    else:
        ax.set_xscale('linear')
        zVals = [0,1,2,3,4,5,6,7,8]

    ax.set_xticks(zVals)
    ax.set_xticklabels(zVals)

    addUniverseAgeAxis(ax, sP)

def plotRedshiftSpacings():
    """ Compare redshift spacing of snapshots of different runs. """

    # config
    sPs = []
    sPs.append( simParams(res=512,run='tracer') )
    sPs.append( simParams(res=512,run='feedback') )
    sPs.append( simParams(res=1820,run='illustris') )

    # plot setup
    xrange = [0.0, 14.0]
    yrange = [0.5, len(sPs) + 0.5]

    runNames = []
    for sP in sPs:
        runNames.append(sP.run)

    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.set_xlabel('Age of Universe [Gyr]')
    ax.set_ylabel('')

    ax.set_yticks( np.arange(len(sPs))+1 )
    ax.set_yticklabels(runNames)
    
    # loop over each run
    for i, sP in enumerate(sPs):
        zVals = snapNumToRedshift(sP,all=True)
        zVals = sP.units.redshiftToAgeFlat(zVals)

        yLoc = (i+1) + np.array([-0.4,0.4])

        for zVal in zVals:
            ax.plot([zVal,zVal],yLoc,lw=0.5,color=sP.colors[1])

    # redshift axis
    addRedshiftAxis(ax, sP)

    fig.tight_layout()    
    fig.savefig(sP.plotPath + 'redshift_spacing.pdf')
    plt.close(fig)

def plotMassFunctions():
    """ Plot DM halo and stellar mass functions comparing multiple boxes, at one redshift. """
    # config
    mass_ranges = [ [6.6, 16.0], [6.6, 13.0] ] # m_halo, m_star
    binSize = 0.2
    
    sPs = []
    sPs.append( simParams(res=1820,run='tng',redshift=2.0) )
    sPs.append( simParams(res=2500,run='tng',redshift=2.0) )

    # plot setup
    fig = plt.figure(figsize=(18,8))

    # halo or stellar mass function
    for j, mass_range in enumerate(mass_ranges):
        nBins = (mass_range[1]-mass_range[0])/binSize

        ax = fig.add_subplot(1,2,j+1)
        ax.set_xlim(mass_range)
        if j == 0: ax.set_xlabel('Halo Mass [ M$_{\\rm 200,crit}$  log M$_\odot$ ]')
        if j == 1: ax.set_xlabel('Stellar Mass [ M$_\star(<2r_{\\rm 1/2,stars})$  centrals  log M$_\odot$ ]')
        ax.set_ylabel('N$_{\\rm bin=%.1f}$' % binSize)
        ax.set_xticks(np.arange(np.int32(mass_range[0]),np.int32(mass_range[1])+1))
        ax.set_yscale('log')

        yy_max = 1.0

        for i, sP in enumerate(sPs):
            print(j,sP.simName)

            if j == 0:
                gc = cosmo.load.groupCat(sP, fieldsHalos=['Group_M_Crit200'])
                masses = sP.units.codeMassToLogMsun(gc['halos'])
            if j == 1:
                gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloMassInRadType'])
                masses = gc['subhalos'][ gc['halos'] ][:,sP.ptNum('stars')] # Mstar (<2*r_{1/2,stars})
                masses = sP.units.codeMassToLogMsun(masses)

            yy, xx = np.histogram(masses, bins=nBins, range=mass_range)
            yy_max = np.max([yy_max,yy.max()])

            label = sP.simName + ' z=%.1f' % sP.redshift
            ax.hist(masses,bins=nBins,range=mass_range,lw=2.0,label=label,histtype='step',alpha=0.9)

        ax.set_ylim([1,yy_max*1.4])
        ax.legend()

    fig.tight_layout()    
    fig.savefig('mass_functions.pdf')
    plt.close(fig)

def haloMassesVsDMOMatched():
    """ Plot the ratio of halo masses matched between baryonic and DMO runs. """
    # config
    runList = ['tng','illustris']
    redshift = 0.0
    resList = [1820, 910, 455]
    cenSatSelect = 'cen' #all, cen, sat

    binSize = 0.1
    linestyles = ['-','--',':']
    sKn = 3 #5
    sKo = 2 #3
    lw = 2.5
    xrange = [8.0, 15.0]
    yrange = [0.6, 1.2]

    # start plot
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)
    ax.set_title('z=%.1f %s [bijective only]' % (redshift,cenSatSelect))

    ax.set_xlabel('M$_{\\rm halo,DM}$ [ log M$_{\\rm sun}$ subhalo ]')
    ax.set_ylabel('M$_{\\rm halo,DM}$ / M$_{\\rm halo,baryonic}$')

    # loop over runs
    for run in runList:
        c = ax._get_lines.prop_cycler.next()['color']

        for i, res in enumerate(resList):
            sP = simParams(res=res,run=run,redshift=redshift)
            sPdm = simParams(res=res,run=run+'_dm',redshift=redshift)
            print(sP.simName)

            # load masses from group catalogs for TNG and DMO runs
            gc_b = groupCat(sP, fieldsSubhalos=['SubhaloMass'])['subhalos']
            gc_dm = groupCat(sPdm, fieldsSubhalos=['SubhaloMass'])['subhalos']

            # restrict to central subhalos of DMO, and valid (!= -1) matches
            wSelect_b = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
            mask_b = np.zeros( gc_b.size, dtype='bool' )
            mask_b[wSelect_b] = 1

            # loop over matching methods
            for j, method in enumerate(['LHaloTree']): #,'SubLink'
                # load matching catalog
                if method == 'SubLink':
                    catPath = sP.postPath + '/SubhaloMatchingToDark/SubLink_%03d.hdf5' % sP.snap
                    assert isfile(catPath)

                    with h5py.File(catPath,'r') as f:
                        dm_inds = f['DescendantIndex'][()]

                    gcInds_b = np.where( (dm_inds >= 0) & (mask_b == 1) )
                    gcInds_dm = dm_inds[ gcInds_b ]
                    assert gcInds_dm.min() >= 0

                if method == 'LHaloTree':
                    catPath = sP.postPath + '/SubhaloMatchingToDark/LHaloTree_%03d.hdf5' % sP.snap
                    assert isfile(catPath)

                    with h5py.File(catPath,'r') as f:
                        b_inds = f['SubhaloIndexFrom'][()]
                        dm_inds = f['SubhaloIndexTo'][()]

                    cs_take_matched = np.where( mask_b[b_inds] == 1 )

                    gcInds_b = b_inds[cs_take_matched]
                    gcInds_dm = dm_inds[cs_take_matched]

                # calculate mass ratios of matched
                masses = sP.units.codeMassToLogMsun(gc_dm[gcInds_dm])
                mass_ratios = gc_b[gcInds_b] / gc_dm[gcInds_dm]

                # plot
                xm, ym, sm, pm = running_median(masses,mass_ratios,binSize=binSize,percs=[10,25,75,90])
                xm = xm[1:-1]
                ym2 = savgol_filter(ym,sKn,sKo)[1:-1]
                sm2 = savgol_filter(sm,sKn,sKo)[1:-1]
                pm2 = savgol_filter(pm,sKn,sKo,axis=1)[:,1:-1]

                ax.plot(xm, ym2, linestyles[i], lw=lw, color=c, label=sP.simName)
                if i == 0:
                    ax.fill_between(xm, pm2[1,:], pm2[-2,:], facecolor=c, alpha=0.1, interpolate=True)

    ax.plot(xrange, [1.0,1.0], '-', color='black', alpha=0.2)

    ax.legend()
    fig.tight_layout()    
    fig.savefig('haloMassRatioVsDMO_L75.pdf')
    plt.close(fig)

