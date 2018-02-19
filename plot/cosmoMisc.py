"""
cosmoMisc.py
  Misc plots related to cosmological boxes.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from os.path import isfile, expanduser
from scipy.signal import savgol_filter

from util import simParams
from util.helper import running_median, logZeroNaN
from cosmo.util import cenSatSubhaloIndices
from cosmo.load import groupCat, snapshotSubset
from plot.config import *

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
    mass_ranges = [ [5.0, 16.0], [4.4, 13.0] ] # m_halo, m_star
    binSize = 0.2
    
    sPs = []
    #sPs.append( simParams(res=2160,run='tng',redshift=0.0) )
    #sPs.append( simParams(res=1080,run='tng',redshift=0.8) )
    #sPs.append( simParams(res=540,run='tng',redshift=0.8) )
    #sPs.append( simParams(res=270,run='tng',redshift=0.8) )
    sPs.append( simParams(res=1024,run='tng_dm',redshift=0.0) )
    sPs.append( simParams(res=1820,run='tng_dm',redshift=0.0) )
    sPs.append( simParams(res=2500,run='tng_dm',redshift=0.0) )

    # plot setup
    fig = plt.figure(figsize=(18,8))

    # halo or stellar mass function
    for j, mass_range in enumerate(mass_ranges):
        nBins = int((mass_range[1]-mass_range[0])/binSize)

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
                gc = groupCat(sP, fieldsHalos=['Group_M_Crit200'])
                masses = sP.units.codeMassToLogMsun(gc['halos'])
                #print(sP.simName, np.where(masses >= 14.0)[0].size)
                #continue
            if j == 1:
                gc = groupCat(sP, fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloMassInRadType'])
                masses = gc['subhalos'][ gc['halos'] ][:,sP.ptNum('stars')] # Mstar (<2*r_{1/2,stars})
                masses = sP.units.codeMassToLogMsun(masses)
                #print(sP.simName, np.where( (masses >= 10.4) & (masses < 10.6) )[0].size)
                #continue

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
    runList = { 'tng':[1820,910,455], 'illustris':[1820,910,455], 'tng':[2500,1250,625] }
    redshift = 0.0
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
    for run in runList.keys():
        c = ax._get_lines.prop_cycler.next()['color']

        for i, res in enumerate(runList[run]):
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

def plotClumpsEvo():
    """ Plot evolution of clumps (smallest N subhalos in halfmassrad) using SubLink_gal tree. """
    import cosmo

    sP = simParams(res=1820,run='tng',redshift=0.0)
    figsize = (22.4,12.6) # (16,9)*1.4
    treeName = 'SubLink_gal'
    selectQuant = 'SubhaloHalfmassRad' # SubhaloHalfmassRadType
    selectN = 1000 # number 
    lw = 2.0
    reverseSort = False # if True, then descending

    # load and select
    gc = groupCat(sP, fieldsHalos=['GroupFirstSub'], fieldsSubhalos=['SubhaloMass', selectQuant])
    sort_inds = np.argsort( gc['subhalos'][selectQuant] )
    if reverseSort: sort_inds = sort_inds[::-1]

    snapRedshifts = cosmo.util.snapNumToRedshift(sP, all=True)
    snapAgeGyr = sP.units.redshiftToAgeFlat( snapRedshifts )
    snapDtGyr = snapAgeGyr - np.roll(snapAgeGyr,1)
    snapDtGyr[0] = snapAgeGyr[0] - sP.units.redshiftToAgeFlat(127.0)

    # start pdf
    pdf = PdfPages('clumps_evo_%s_first%d.pdf' % (sP.simName,selectN))

    # halo or stellar mass function
    for i in range(selectN):
        print('[%3d] subhaloID = %d' % (i,sort_inds[i]))
        fig = plt.figure(figsize=figsize)

        # load MPB
        mpbFields = ['SnapNum','SubhaloPos','SubhaloHalfmassRad','SubhaloHalfmassRadType','SubhaloSFR',
                     'SubhaloMass','SubhaloMassType','SubhaloGrNr','Group_M_Crit200','SubhaloParent']
        mpb = cosmo.mergertree.loadMPB(sP, sort_inds[i], fields=mpbFields, treeName=treeName)

        xx = cosmo.util.snapNumToRedshift(sP, mpb['SnapNum'])

        # get the MPB of the z=0 parent halo
        halo = cosmo.load.groupCatSingle(sP, haloID=mpb['SubhaloGrNr'][0])
        mpbParent = cosmo.mergertree.loadMPB(sP, halo['GroupFirstSub'], fields=mpbFields, treeName=treeName)

        xxPar = cosmo.util.snapNumToRedshift(sP, mpbParent['SnapNum'])

        # allocate parent relative properties
        radFromPar = np.zeros( xx.size, dtype='float32')
        inParFlag = np.zeros( xx.size, dtype='float32' )
        inParFlag2 = np.zeros( xx.size, dtype='float32' )
        radFromPar.fill(np.nan)

        for j in range(xx.size):
            # cross-match
            w = np.where(mpbParent['SnapNum'] == mpb['SnapNum'][j])[0]
            if len(w) == 0:
                continue
            assert len(w) == 1
            w = w[0]

            # calculate radial distance of clump from this z=0 parent halo MPB
            xyzPar = mpbParent['SubhaloPos'][w,:].reshape( (1,3) )
            xyzSub = mpb['SubhaloPos'][j,:].reshape( (1,3) )

            radFromPar[j] = cosmo.util.periodicDists(xyzPar, xyzSub, sP)

            # calculate when clump is/isnot within this z=0 parent halo
            if mpbParent['SubhaloGrNr'][w] == mpb['SubhaloGrNr'][j]:
                inParFlag[j] = 1
            if mpbParent['SubhaloParent'][w] == mpb['SubhaloParent'][j]:
                inParFlag2[j] = 1

        # load member star particle formation times and calculate a SFR(t) based on them
        sfh = np.zeros( snapRedshifts.size, dtype='float32' )
        sfh.fill(np.nan)

        if mpb['SubhaloMassType'][0,sP.ptNum('stars')] > 0:
            stars = cosmo.load.snapshotSubset(sP, 'stars', ['masses','sftime'], subhaloID=sort_inds[i])

            snapScalefacs = 1.0 / (1+snapRedshifts)
            for j in range(snapRedshifts.size-1):
                aMin = snapScalefacs[j]
                aMax = snapScalefacs[j+1]
                w = np.where( (stars['GFM_StellarFormationTime'] > aMin) & \
                              (stars['GFM_StellarFormationTime'] <= aMax) )

                # compute SFR in [Msun/yr] in this redshift bin between two successive snapshots
                sfh[j] = sP.units.codeMassToMsun(np.sum(stars['Masses'][w])) / (snapDtGyr[j]*1e9)

            #verifyMass1 = np.log10( np.nanmean(sfh) * sP.units.redshiftToAgeFlat(0.0)*1e9 )
            #verifyMass2 = sP.units.codeMassToLogMsun( mpb['SubhaloMassType'][0,sP.ptNum('stars')] )
            #assert np.abs(verifyMass1-verifyMass2)/verifyMass1 < 0.5 # 50% agreement in log

        # modify symbol to a single circle for clumps with no tracked snapshots
        sym = '-' if xx.size > 1 else 'o'

        # six quantities
        for j in range(6):
            ax = fig.add_subplot(2,3,j+1)

            redshift_max = 2.0 if xx.max() > 2.0 else xx.max()+0.1
            if xx.size == 1: redshift_max = 2.0
            ax.set_xlim([redshift_max, -0.1])
            ax.set_xlabel('Redshift')

            if j == 0:
                # quant (A): mass by type
                ax.set_title(sP.simName + ' ['+str(i)+'] shID='+str(sort_inds[i]))
                ax.set_ylabel('Subhalo Mass [ log M$_{\\rm sun}$ ]')
                yy0 = sP.units.codeMassToLogMsun( mpb['SubhaloMass'] )
                ax.plot(xx, yy0, sym, lw=lw, label='total')

                for ptName in ['gas','stars','dm','bhs']:
                    yy = sP.units.codeMassToLogMsun( mpb['SubhaloMassType'][:,sP.ptNum(ptName)])
                    if ptName == 'bhs' and yy.size == 1 and yy == 0.0: continue
                    ax.plot(xx, yy, sym, lw=lw, label=ptName)

                ax.legend(loc='best')

            if j == 1:
                # quant (B): mass fractions
                ax.set_title('z=0 size: %.3f ckpc/h' % mpb['SubhaloHalfmassRad'][0])
                ax.set_ylabel('log ( Subhalo Mass Fraction )')
                c = ax._get_lines.prop_cycler.next()['color'] # skip total color

                for ptName in ['gas','stars','dm']:
                    yy = mpb['SubhaloMassType'][:,sP.ptNum(ptName)] / mpb['SubhaloMass']
                    yy = logZeroNaN(yy)
                    ax.plot(xx, yy, sym, lw=lw, label=ptName+'/total')

                ax.legend(loc='best')

            if j == 2:
                # quant (C): sizes
                ax.set_title('z$_{\\rm trackedto}$=%.1f numSnapsTracked=%d' % (xx.max(),xx.size))
                ax.set_ylabel('Subhalo Size [ log ckpc/h ]')
                ax.plot(xx, logZeroNaN(mpb['SubhaloHalfmassRad']), sym, lw=lw, label='total')

                for ptName in ['gas','stars','dm']:
                    yy = mpb['SubhaloHalfmassRadType'][:,sP.ptNum(ptName)]
                    yy = logZeroNaN(yy)
                    ax.plot(xx, yy, sym, lw=lw, label=ptName)

                ax.legend(loc='best')

            if j == 3:
                # quant (D): parent halo mass
                ax.set_ylabel('Parent Halo M$_{200}$ [ log M$_{\\rm sun}$ ]')

                yy = sP.units.codeMassToLogMsun( mpbParent['Group_M_Crit200'] )
                ax.plot(xxPar, yy, '-', lw=lw, color='black')

            if j == 4:
                # quant (E): radial distance from parent halo
                ax.set_ylabel('Radial Dist from Parent [ log ckpc/h ]')

                w1 = np.where( inParFlag == 0 )
                w2 = np.where( inParFlag2 == 0 )

                if len(w1[0]):
                    ax.plot(xx[w1], logZeroNaN(radFromPar)[w1], 'o', markeredgecolor='red', alpha=0.8, label='outside z=0 parentGr')
                if len(w2[0]):
                    ax.plot(xx[w2], logZeroNaN(radFromPar)[w2], 's', markeredgecolor='green', alpha=0.8, label='subhParent differs from mpb')
                if len(w1[0]) or len(w2[0]):
                    ax.legend()

                ax.plot(xx, logZeroNaN(radFromPar), sym, lw=lw, color='black')

            if j == 5:
                # quant (F): SFR(t) and SFH histogram of constitutient star particles
                ax.set_ylabel('Subhalo SFR [ log M$_{\\rm sun}$/yr ]')
                yy = logZeroNaN( mpb['SubhaloSFR'] )

                ax.plot(xx, yy, sym, lw=lw, color='black')

                if mpb['SubhaloMassType'][0,sP.ptNum('stars')] > 0:
                    ax.plot(snapRedshifts, logZeroNaN(sfh), '-', lw=lw, color='red', label='from star ages')
                    ax.legend()

        fig.tight_layout()    
        pdf.savefig()
        plt.close(fig)

    pdf.close()

def compareEOSFiles(doTempNotPres=False):
    """ Compare eos.txt files from different runs (in eosFiles), and actual runs as well (in sPs). """
    sPs = []
    sPs.append( simParams(res=455,run='tng',redshift=0.0) )
    sPs.append( simParams(res=910,run='tng',redshift=0.0) )
    sPs.append( simParams(res=1820,run='tng',redshift=0.0) )

    eosBasePath = expanduser("~") + '/python/data/sim.tng/'
    eosFiles = ['eos_q03.txt','eos_q1.txt','eos_poly.txt']
    eosLabels = ['normal eEOS q=0.3',
                 'normal eEOS q=1.0',
                 'normal eEOS q=0.3 + polytropic 4/3 above 100$\\rho_{\\rm crit}$']

    binSize = 0.1 # in log(dens)

    # start plot
    fig = plt.figure(figsize=(12.0,8.0))
    ax = fig.add_subplot(111)

    ax.set_ylim([0,13])
    if doTempNotPres: ax.set_ylim([4,8])
    ax.set_xlim([-6,5])
    
    # load eos.txt details and plot
    for i, eosFile in enumerate(eosFiles):
        # load and unit conversion
        print(eosFile)
        sP = simParams(res=455,run='tng',redshift=0.0) # eos.txt has density in [code/a^3] units with a=1

        data = np.loadtxt(eosBasePath + eosFile)
        dens = sP.units.codeDensToPhys(np.squeeze( data[:,0] ), cgs=True, numDens=True)
        dens *= sP.units.hydrogen_massfrac
        pres = np.squeeze( data[:,1] ) * sP.units.UnitPressure_in_cgs / sP.units.boltzmann # K/cm^3 (CGS)
        temp = pres / (sP.units.gamma-1.0) / dens # K

        # plot
        if doTempNotPres:
            ax.plot(np.log10(dens),np.log10(temp),'-', alpha=0.9, label=eosLabels[i])
        else:
            ax.plot(np.log10(dens),np.log10(pres),'-', alpha=0.9, label=eosLabels[i])

    # load actual sim data and plot
    for sP in sPs:
        print(sP.simName)
        
        sim_dens = snapshotSubset(sP, 'gas', 'dens')
        sim_dens = sP.units.codeDensToPhys(sim_dens, cgs=True, numDens=True) * sP.units.hydrogen_massfrac
        sim_dens = np.log10(sim_dens)

        if doTempNotPres:
            sim_temp = snapshotSubset(sP, 'gas', 'temp')
            xm1, ym1, sm1 = running_median(sim_dens,sim_temp,binSize=binSize)

            ax.plot(xm1[:-1], ym1[:-1], '-', alpha=0.9, label=sP.simName+' median T$_{\\rm gas}$')
        else:
            # pressures
            sim_pres_gas = snapshotSubset(sP, 'gas', 'P_gas')
            sim_pres_B   = snapshotSubset(sP, 'gas', 'P_B')

            xm1, ym1, sm1 = running_median(sim_dens,sim_pres_gas,binSize=binSize)
            xm2, ym2, sm2 = running_median(sim_dens,sim_pres_B,binSize=binSize)

            l, = ax.plot(xm1[:-1], ym1[:-1], '-', alpha=0.9, label=sP.simName+' median P$_{\\rm gas}$')
            ax.plot(xm2[:-1], ym2[:-1], ':', alpha=0.9, color=l.get_color(), label=sP.simName+' median P$_{\\rm B}$')

    if doTempNotPres:
        ax.set_ylabel('Temperature [log K]')
    else:
        ax.set_ylabel('Pressure [log cgs K/cm^3]')

    ax.set_xlabel('Density [log physical 1/cm^3]')

    fig.tight_layout()
    ax.legend(loc='best')
    if doTempNotPres:
        plt.savefig('compareTwoEosFiles_temp.pdf')
    else:
        plt.savefig('compareTwoEosFiles.pdf')
    plt.close()


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
