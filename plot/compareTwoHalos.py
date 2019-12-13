"""
compareTwoHalos.py
  summary plots comparing two matched halos
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import illustris_python as il
from util import units
from util.helper import isUnique, nUnique, iterable, logZeroNaN, sampleColorTable, getWhiteBlackColors
from cosmo.util import crossMatchSubhalosBetweenRuns, snapNumToRedshift, cenSatSubhaloIndices
from cosmo.color import loadSimGalColors, gfmBands
from cosmo.mergertree import loadMPB
from plot.quantities import simSubhaloQuantity
from plot.config import *
from cosmo.load import groupCat, groupCatSingle, snapshotSubset
    
def matchedUniqueGCIDs(gc1,gc2,matchPath,snapNum):
    """ Return i1,i2 two sets of indices into gc1,gc2 based on matching results, such that
        gc1[i1] -> gc2[i2]. The match file contains 'SubhaloIndex' which gives, for every 
        subhalo of gc1, the match index in gc2. """
    
    f = h5py.File(matchPath + "subhalos_Illustris1_" + str(snapNum).zfill(3) + ".hdf5")
    ind2 = f['SubhaloIndex'][:]
    f.close()
    
    # match between runs
    ind1 = np.arange(0,gc1['count'])
    
    w = np.where(ind2 >= 0)
    print('Number matched: ' + str(len(w[0])) + ' of ' + str(gc1['count']) + ' ('+str(len(ind2))+')')
    
    # non-unique, take first (highest mass) IllustrisPrime target for each Illustris subhalo
    _, ind2_uniq_inds = np.unique(ind2, return_index=True)
    w_new = np.intersect1d(w[0],ind2_uniq_inds)
    
    print('Number unique targets overall: ' + str(len(ind2_uniq_inds)))
    print('Number unique matched targets: ' + str(len(w_new)))
    
    #TODO
    w = w_new
    
    # replace ind1 with ind1[w] before this point (move unique+matches indices to func)
    ind1 = ind1[w]
    ind2 = ind2[w]
    
    return ind1,ind2
    
def priSecMatchedGCIDs(ind1,ind2,basePath1,snapNum):
    """ Given indices into groupcats of two runs, ind1 and ind2, separate them into 
        pri/sec (centrals/satellites) based on status in run1. """
        
    # get pri/sec indices of run1
    ps1_pri, ps1_all, ps1_sec = cenSatSubhaloIndices(sP)
    
    # intersect pri/sec with run1 indices
    pri_inds = np.in1d( ind1, ps1_pri, assume_unique=True ).nonzero()[0]
    sec_inds = np.in1d( ind1, ps1_sec, assume_unique=True ).nonzero()[0]
    
    # create new run1 and run2 indices separated by pri/sec
    r = {}

    r['pri2'] = ind2[ pri_inds ]
    r['sec2'] = ind2[ sec_inds ]
    r['pri1'] = ind1[ pri_inds ]
    r['sec1'] = ind1[ sec_inds ]
    
    # debug verify
    if True:
        if not isUnique(ind1) or not isUnique(ind2):
            raise Exception('failed')
        if not isUnique(ps1_pri) or not isUnique(ps1_sec):
            raise Exception('failed')
            
        print(' pri: ' + str(len(r['pri1'])) + ' sec: ' + str(len(r['sec1'])))
        
        if len(r['pri1']) + len(r['sec1']) != len(ind1):
            raise Exception('failed')
        if len(r['pri2']) + len(r['sec2']) != len(ind2):
            raise Exception('failed')
                
        check1 = np.sort( np.concatenate( (r['pri1'],r['sec1']) ) )
        
        if not np.array_equal( check1, ind1 ):
            raise Exception('failed')
        if len(r['pri1']) != len(r['pri2']) or len(r['sec1']) != len(r['sec2']):
            raise Exception('failed')
    
    return r
    
def globalCatComparison(sP1,sP22,matchPath):
    """ Compare population statistic plots/ratios between two runs using matched objects. """
    fields = ['SubhaloMass','SubhaloHalfmassRad','SubhaloMassType']
    
    # plot config
    nBins = 100
    xMinMax = [9.0,14.0]
    yMinMax = [-0.4,0.4]
    hMinMax = [xMinMax, yMinMax]  #[[xmin,xmax], [ymin,ymax]]
    yLineVals = [-0.1249,0.0,0.0969] #-25%, equal, +25%
    
    # load
    gc1 = il.groupcat.loadSubhalos(sP1.simPath,sP1.snap,fields=fields)
    gc2 = il.groupcat.loadSubhalos(sP2.simPath,sP2.snap,fields=fields)
    
    # match indices between runs, split into pri/sec
    ind1, ind2 = matchedUniqueGCIDs(gc1,gc2,matchPath,snapNum)
    inds = priSecMatchedGCIDs(ind1,ind2,basePath1,snapNum)   
    
    # restrict all fields to matched halos, separated by pri/sec
    gc1['pri'] = {}
    gc1['sec'] = {}
    gc2['pri'] = {}
    gc2['sec'] = {}
    
    for field in fields:
        gc1['pri'][field] = gc1[field][inds['pri1']]
        gc1['sec'][field] = gc1[field][inds['sec1']]
        gc2['pri'][field] = gc2[field][inds['pri2']]
        gc2['sec'][field] = gc2[field][inds['sec2']]
        gc1[field] = gc1[field][ind1]
        gc2[field] = gc2[field][ind2]
    
    for i in [0,1]:
        # init
        fig, ax = plt.subplots(1, 3, figsize=(12,4))

        if i == 0:
            fieldName = 'SubhaloMass'
            ytitle = 'log ( $M_{halo}^{IP} / M_{halo}^{I}$ )'
        if i == 1:
            fieldName = 'SubhaloHalfmassRad'
            ytitle = 'log ( $r_{1/2}^{IP} / r_{1/2}^{I}$ )'
            
        # plot (1) left, central
        ratio = np.log10( gc2['pri'][fieldName] / gc1['pri'][fieldName] )
        ax[0].set_ylabel(ytitle)
        
        xval = units.codeMassToLogMsun( gc1['pri']['SubhaloMass'] )
        ax[0].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[0].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[0].set_xlabel('Halo Mass [log $M_\odot$]')      
        ax[0].set_title('centrals')
        
        # plot (2) center, satellites
        ratio = np.log10( gc2[fieldName] / gc1[fieldName] )
        
        xval = units.codeMassToLogMsun( gc1['SubhaloMass'] )        
        ax[1].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[1].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[1].set_xlabel('Halo Mass [log $M_\odot$]')        
        ax[1].set_title('all')
        
        # plot (3) right, all
        ratio = np.log10( gc2['sec'][fieldName] / gc1['sec'][fieldName] )
        xval = units.codeMassToLogMsun( gc1['sec']['SubhaloMass'] )
        
        ax[2].hist2d(xval,ratio,bins=nBins,range=hMinMax,norm=LogNorm())
        for yLineVal in yLineVals:
            ax[1].plot( xMinMax, [yLineVal,yLineVal], '--', color='black' )
        ax[2].set_xlabel('Halo Mass [log $M_\odot$]')    
        ax[2].set_title('satellites')
        
        # finalize
        fig.tight_layout()        
        fig.savefig('test_' + fieldName + '.pdf')
        plt.close(fig)    

def timeSeriesMultiPanelComp(sP1, shID1, sP2, shID2):
    """ A few panels of time-series evolution of two subhalos shID1 and shID2 from sP1 and sP2, 
    respectively. Can be lists, in which case all are individually plotted. """
    from vis.common import setAxisColors

    # visual config
    xMinMax = [0.0, 2.0]
    xLabel = 'Redshift'
    sizefac = 0.7 * 2.5
    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    lw = 2.5
    nPanels = 7

    color_1 = sampleColorTable( 'tableau10', ['red'] )[0] # illustris
    color_2 = sampleColorTable( 'tableau10', ['blue'] )[0] # tng

    # load MPBs
    mpbs1 = []
    mpbs2 = []

    shID1 = iterable(shID1)
    shID2 = iterable(shID2)

    for shID in shID1:
        mpbs1.append( loadMPB(sP1, shID) )
    for shID in shID2:
        mpbs2.append( loadMPB(sP2, shID) )
       
    # start figure
    fig = plt.figure(figsize=(figsize[0]*sizefac,figsize[1]*sizefac*(1+0.1*nPanels)),facecolor=color1)

    # top panel: mu_{red,blue}
    for i in range(nPanels):
        if i == 0:
            ax0 = fig.add_subplot(nPanels,1,i+1, facecolor=color1)
            ax = ax0
        else:
            ax = fig.add_subplot(nPanels,1,i+1, facecolor=color1, sharex=ax0)
        setAxisColors(ax, color2)

        ax.set_xlim(xMinMax)
        ax.set_xlabel(xLabel)

        if i == 0:
            # stellar mass
            ax.set_ylabel('$M_\star$ [ log Msun ]')
            ax.set_ylim([10.0,11.5])
        if i == 1:
            # sSFR
            ax.set_ylabel('sSFR [ log Gyr$^{-1}$ ]')
            ax.set_ylim([-3.5,0.5])
        if i == 2:
            # gas fraction
            ax.set_ylabel('Gas Fraction [ log ]')
            ax.set_ylim([-2.5,0.0])
        if i == 3:
            # color
            ax.set_ylabel('(g-r) color [ mag ]')
            ax.set_ylim([0.1,0.9])
        if i == 4:
            # bh mdot
            ax.set_ylabel('BH Mdot [ log Msun/yr ]')
            ax.set_ylim([-6.0,0.0])
        if i == 5:
            # stellar size
            ax.set_ylabel('r$_{\star,1/2}$ [ kpc ]')
            ax.set_ylim([1.0,30.0])
        if i == 6:
            # stellar surface density inside 1rhalf
            ax.set_ylabel('$\Sigma_{\star}$ [ log M$_{\\rm sun}$/kpc$^2$ ]')
            ax.set_ylim([7.0, 9.0])

        for sP,mpbs,color,shID in zip([sP1,sP2],[mpbs1,mpbs2],[color_1,color_2],[shID1,shID2]):
            for j in range(len(mpbs)):
                x_vals = mpbs[j]['SnapNum']
                x_vals = snapNumToRedshift(sP, snap=x_vals)

                mstar = sP.units.codeMassToMsun( mpbs[j]['SubhaloMassInRadType'][:,4] )
                mgas  = sP.units.codeMassToMsun( mpbs[j]['SubhaloMassInRadType'][:,0] )
                photVector = mpbs[j]['SubhaloStellarPhotometrics']
                gr    = photVector[:,gfmBands['g']] - photVector[:,gfmBands['r']]

                bh_mdot = mpbs[j]['SubhaloBHMdot'] * 10.22 # msun/yr
                bh_mass = sP.units.codeMassToMsun( mpbs[j]['SubhaloBHMass'] )

                rstar = sP.units.codeLengthToKpc( mpbs[j]['SubhaloHalfmassRadType'][:,4] )
                rgas  = sP.units.codeLengthToKpc( mpbs[j]['SubhaloHalfmassRadType'][:,0] )

                if i == 0:
                    y_vals = sP.units.codeMassToLogMsun( mpbs[j]['SubhaloMassInRadType'][:,4] )
                if i == 1:
                    y_vals = logZeroNaN(mpbs[j]['SubhaloSFRinRad'] * 1e9 / mstar)
                if i == 2:
                    y_vals = logZeroNaN(mgas / (mgas+mstar))
                if i == 3:
                    y_vals = gr
                if i == 4:
                    y_vals = logZeroNaN(bh_mdot)
                if i == 5:
                    y_vals = rstar
                if i == 6:
                    y_vals = logZeroNaN(mstar / (np.pi * rstar**2))

                # todo: determine BH in low vs high state
                bh_medd = sP.units.codeBHMassToMdotEdd( mpbs[j]['SubhaloBHMass'] )

                bh_lowStateFlag = np.zeros( bh_mdot.size, dtype='int32' )
                w = np.where( bh_mdot/bh_medd < sP.units.BH_chi(bh_mass) )
                bh_lowStateFlag[w] = 1

                # split plot with np.where into low,high with squares,circles respectively
                w_low = np.where(bh_lowStateFlag == 1)
                #w_high = np.where(bh_lowStateFlag == 0)

                ax.plot(x_vals, y_vals, 'o-', label=sP.simName+'#'+str(shID[j]), color=color)
                ax.plot(x_vals[w_low], y_vals[w_low], 's', color=color)

        ax.legend(loc='best')#, prop={'size':11})

    # finish plot and save
    fig.tight_layout()
    fig.savefig('timeSeriesMultiPanelComp_%s_%s_%s.pdf' % (sP1.run,sP2.run,'-'.join([str(s) for s in shID2])))
    plt.close(fig)

def illustrisVsTNG_RedEvoComp(candInd=None):
    """ Driver for Illustris-1 vs TNG100-1 comparison of time evolution of massive galaxies which 
    are red/quenched in TNG but still blue/star-forming in Illustris. If candInd is not None, just 
    an individual system, otherwise all of them. """
    from util import simParams

    # config
    sP1 = simParams(res=1820, run='illustris', redshift=0.0)
    sP2 = simParams(res=1820, run='tng', redshift=0.0)

    matchMethod = 'Positional' # Lagrange
    mstarBin    = [11.0,11.5] # log msun
    colorThresh = 0.7 # g-r

    if 0:
        # load TNG catalog for selection
        mstar2, _, _, _ = simSubhaloQuantity(sP2, 'mstar_30pkpc_log', clean=clean)

        color_gr1, _ = loadSimGalColors(sP1, defSimColorModel, bands=['g','r'])
        color_gr2, _ = loadSimGalColors(sP2, defSimColorModel, bands=['g','r'])

        # select: massive and red
        with np.errstate(invalid='ignore'):
            w = np.where( (mstar2 > mstarBin[0]) & (mstar2 < mstarBin[1]) & (color_gr2 > colorThresh) )
        subhaloInds2 = w[0]

        # centrals only
        from tracer.tracerMC import match3
        cen_inds2 = cenSatSubhaloIndices(sP2, cenSatSelect='cen')

        _,i2 = match3(cen_inds2, subhaloInds2)
        subhaloInds2 = subhaloInds2[i2]

        # cross-match TNG -> Illustris
        subhaloInds1 = crossMatchSubhalosBetweenRuns(sP2, sP1, subhaloInds2, method=matchMethod)

        # load Illustris catalog to finish selection
        with np.errstate(invalid='ignore'):
            ww = np.where( color_gr1[subhaloInds1] < colorThresh )[0] # blue in illustris

        subhaloInds1 = subhaloInds1[ww]
        subhaloInds2 = subhaloInds2[ww]
    if 1:
        # just the above, cached
        assert mstarBin == [11.0,11.5] and colorThresh == 0.7

        subhaloInds1 = [224435, 199142, 262551, 290464, 296519, 300120, 291771, 305507,
                       308748, 317769, 311140, 323212, 313084, 340843, 333168, 336253,
                       326247, 341618, 332327, 332721, 326929, 337947, 353280, 343552,
                       351433, 331830, 366317, 354641, 353029, 350919, 362079, 357561,
                       376048, 374140, 383346, 366034, 364109]
        subhaloInds2 = [210763, 217485, 260862, 270963, 322160, 333672, 342689, 348038,
                       350486, 352359, 358016, 364263, 368436, 368697, 375020, 376963,
                       379320, 379732, 380934, 384705, 387438, 388249, 392961, 394600,
                       394986, 396073, 398110, 401192, 403110, 403787, 409761, 414050,
                       417032, 429237, 429989, 438368, 445271]

    if candInd is not None:
        subhaloInds1 = subhaloInds1[candInd] # illustris
        subhaloInds2 = subhaloInds2[candInd] # tng

    # plot
    timeSeriesMultiPanelComp(sP1, subhaloInds1, sP2, subhaloInds2)
