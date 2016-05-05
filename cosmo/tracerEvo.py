"""
tracerEvo.py
  Analysis and plotting of evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter
import scipy.ndimage as ndimage
from os.path import isfile

from util.tracerMC import subhaloTracersTimeEvo, subhalosTracersTimeEvo
from cosmo.mergertree import mpbSmoothedProperties
from util.helper import loadColorTable, logZeroSafe
from cosmo.util import addRedshiftAgeAxes, redshiftToSnapNum

# integer flags for accretion modes
ACCMODE_NONE     = -1
ACCMODE_SMOOTH   = 1
ACCMODE_MERGER   = 2
ACCMODE_STRIPPED = 3

def zoomDataDriver(sP, fields):
    """ Run and save data files for tracer evolution in several quantities of interest. """
    from util import simParams

    #sP = simParams(res=11, run='zooms2', redshift=2.0, hInd=2)
    #fields = ['tracer_maxtemp','tracer_maxent','rad_rvir','vrad','entr','temp','sfr','subhalo_id']

    subhaloID = sP.zoomSubhaloID
    snapStep  = 1

    subhaloTracersTimeEvo(sP, subhaloID, fields, snapStep=snapStep)

def addRedshiftAgeImageAxes(ax, sP):
    """ Add a redshift (bottom) and age (top) pair of axes for imshow plots. Top axis does not work 
    when a colorbar is also added to the plot. """
    from cosmo.util import redshiftToSnapNum

    zVals = np.array([2,3,4,5,6,7,8,9,10])
    snaps = redshiftToSnapNum( zVals, sP )
    ax.set_xticks(snaps)
    ax.set_xticklabels(zVals)
    ax.set_xlabel('Redshift')

    if 0:
        axTop = ax.twiny()

        ageVals = [0.7,1.0,1.5,2.0,3.0]
        ageVals.append( sP.units.redshiftToAgeFlat([zVals.min()]).round(2) )
        axTickVals = redshiftToSnapNum(sP.units.ageFlatToRedshift( np.array(ageVals) ), sP)

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(axTickVals)
        axTop.set_xticklabels(ageVals)
        axTop.set_xlabel("Age of the Universe [Gyr]")

def plotConfig(fieldName):
    """ Store some common plot configuration parameters. """
    print(fieldName)

    if fieldName == "tracer_maxtemp":
        ctName    = "jet"
        label     = "Tracer Tmax [ log K ]"
        valMinMax = [4.0,6.5]
        takeLog   = True

    if fieldName == "tracer_maxent":
        pass #todo

    if fieldName == "rad_rvir":
        ctName    = "jet"
        label     = "R / Rvir"
        valMinMax = [0.0,2.0]
        takeLog   = False

    if fieldName == "vrad":
        pass #todo

    if fieldName == "entr":
        ctName    = "jet"
        label     = "Gas Entropy [ log K cm^2 ]"
        valMinMax = None #[4.0,6.5]
        takeLog   = False

    if fieldName == "temp":
        ctName    = "jet"
        label     = "Gas Temp [ log K ]"
        valMinMax = [4.0,6.5]
        takeLog   = False

    if fieldName == "sfr":
        pass #todo

    return ctName, label, valMinMax, takeLog

def plotEvo2D():
    """ Plot various full 2D blocks showing evolution of 'all' tracer tracks vs redshift/radius. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    fieldNames = ["tracer_maxtemp","rad_rvir"] # "tracer_maxent", "vrad", "entr", "temp", "sfr"

    # load accretion times
    trAccTimes = accTime(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: overview 2D plot of all tracker tracks
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID')

            # data transform
            data2d = np.transpose(data[fieldName].copy())

            # resize tracerInd axis to reasonable raster size
            zoomFac = (1080*4) / data2d.shape[0]
            data2d = ndimage.zoom( data2d, [zoomFac,1], order=1 )

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = 0
            y_max = data['TracerIDs'].size - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

        if 1:
            # PLOT 2: overview 2D plot of all tracker tracks, sort by accTime
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID [Sorted by Accretion Time]')

            # data transform, sort
            data2d = np.transpose(data[fieldName].copy())

            sort_inds = np.argsort(trAccTimes)
            data2d = data2d[sort_inds,:]

            # resize tracerInd axis to reasonable raster size
            zoomFac = (1080*4) / data2d.shape[0]
            data2d = ndimage.zoom( data2d, [zoomFac,1], order=1 )

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = 0
            y_max = data['TracerIDs'].size - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_accTimeSorted_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

        if 1:
            # PLOT 3: zoom view of 2D
            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)

            ax.set_ylabel('TracerID')

            # data transform
            data2d = np.transpose(data[fieldName].copy())

            startInd = 100000
            zoomSize = 1080*1
            data2d = data2d[startInd:startInd+zoomSize,:]

            # resize tracerInd axis to reasonable raster size (unneeded)
            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            # color mapping
            cmap = loadColorTable(ctName)

            # axes ranges and place image
            x_min = int( data['snaps'].max() )
            x_max = int( data['snaps'].min() )
            y_min = startInd
            y_max = startInd + zoomSize - 1

            extent = [x_min, x_max, y_min, y_max]

            plt.imshow(data2d, cmap=cmap, extent=extent, aspect='auto', origin='lower')

            if valMinMax is not None:
                plt.clim( valMinMax )

            # colobar and axes
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

            cb = plt.colorbar(cax=cax) #, format=FormatStrFormatter('%.1f'))
            cb.ax.set_ylabel(label)

            addRedshiftAgeImageAxes(ax, sP)

            # finish
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo2D_zoom_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

def plotEvo1D():
    """ Plot various 1D views showing evolution of tracer tracks vs redshift/radius. """
    from util import simParams

    # config
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    fieldNames = ["tracer_maxtemp","rad_rvir"] # "temp","entr"

    # load accretion times
    trAccTimes = accTime(sP, snapStep)

    for fieldName in fieldNames:
        ctName, label, valMinMax, takeLog = plotConfig(fieldName)

        # load
        data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep=snapStep)

        if 1:
            # PLOT 1: little 1D plot of a few tracer tracks
            inds = [0,10,100,1000]

            fig = plt.figure( figsize=(16,10) )
            ax = fig.add_subplot(1,1,1)
            #ax.set_xlim([2.0,10.0])
            ax.set_xlabel('Redshift')
            ax.set_ylabel(label)

            # data transform, resize tracerInd axis to reasonable raster size
            data2d = np.transpose(data[fieldName].copy())

            if valMinMax is not None:
                ax.set_ylim(valMinMax)

            if takeLog:
                data2d = logZeroSafe( data2d )

                w = np.where(data2d == 0.0)
                data2d[w] = np.nan # flag missing data

            for ind in inds:
                ax.plot( data['redshifts'], data2d[ind,:], label='TracerID '+str(ind) )

            ax.legend(loc='upper right')
            fig.tight_layout()    
            fig.savefig(sP.plotPath+'tracerEvo1D_%s_%s_step%d.pdf' % (sP.simName,fieldName,snapStep))
            plt.close(fig)

def accTime(sP, snapStep=1, rVirFac=1.0):
    """ Calculate accretion time for each tracer (and cache), as the earliest (highest redshift) crossing 
    of the virial radius of the MPB halo. Uses the 'rad_rvir' field. 
    Argument: rVirFac = what fraction of the virial radius denotes the accretion time? """

    # check for existence
    saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_time_%d.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(10.0,sP),snapStep,rVirFac*10)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            return f['accTimeInterp'][()]

    # load
    data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, ['rad_rvir'], snapStep)

    # reverse so that increasing indices are increasing snapshot numbers
    data2d = data['rad_rvir'][::-1,:]

    data['snaps'] = data['snaps'][::-1]
    data['redshifts'] = data['redshifts'][::-1]
    
    # set mask to one for all radii less than factor
    mask2d = np.zeros_like( data2d, dtype='int16' )
    ww = np.where( data2d < rVirFac )
    mask2d[ww] = 1

    # along second axis (trInds), take first index (lowest snap number inside) which is nonzero
    firstSnapInsideInd = np.argmax( mask2d, axis=0 )

    # interp between index and previous (one snap before first time inside) for non-discrete answer
    accTimeInterp = np.zeros( data['TracerIDs'].size, dtype='float32' )

    for i in range(data['TracerIDs'].size):
        if i % int(data['TracerIDs'].size/10) == 0:
            print(' %4.1f%%' % (float(i)/data['TracerIDs'].size*100.0))

        ind0 = firstSnapInsideInd[i]
        ind1 = firstSnapInsideInd[i] - 1

        if ind0 == 0:
            # never inside? flag with nan
            if mask2d[:,i].sum() == 0:
                accTimeInterp[i] = np.nan
                continue

            # actually inside from first available snapshot
            accTimeInterp[i] = data['redshifts'][0]
            continue

        assert ind0 > 0
        assert ind1 >= 0

        z0 = data['redshifts'][ind0]
        z1 = data['redshifts'][ind1]
        r0 = data2d[ind0,i]
        r1 = data2d[ind1,i]

        # linear interpolation, find redshift where rad_rvir=1.0
        accTimeInterp[i] = (1.0-r0)/(r1-r0) * (z1-z0) + z0

    # save
    with h5py.File(saveFilename,'w') as f:
        f['accTimeInterp'] = accTimeInterp

    print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])
    return accTimeInterp

def accMode():
    """ Derive an 'accretion mode' categorization for each tracer based on its group membership history. 
    Specifically, separate all tracers into one of [smooth/merger/stripped] defined as:
      - smooth: child of MPB or no subhalo at all z>=z_acc 
      - merger: child of subhalo other than the MPB at z=z_acc 
      - stripped: child of MPB or no subhalo at z=z_acc, but child of non-MPB subhalo at any z>z_acc 
    Where z_acc is the accretion redshift defined as the first (highest z) crossing of the virial radius. """
    # config (testing)
    from util import simParams
    sP = simParams(res=9, run='zooms2', redshift=2.0, hInd=2)
    snapStep = 1

    # check for existence
    saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_mode.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(10.0,sP),snapStep)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            return f['accTimeInterp'][()]

    # load accTime, subhalo_id tracks, and MPB history
    mpb  = mpbSmoothedProperties(sP, sP.zoomSubhaloID)
    data = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, ['subhalo_id'], snapStep=snapStep)
    acc_time = accTime(sP, snapStep=snapStep)

    # allocate return
    accMode = np.zeros( acc_time.size, dtype='int8' )

    # closest snapshot for each accretion time
    z_inds1 = np.searchsorted( data['redshifts'], acc_time )

    ww = np.where(z_inds1 == data['redshifts'].size)
    z_inds1[ww] = z_inds1[ww] - 1

    z_inds0 = z_inds1 - 1

    z_dist1 = np.abs( acc_time - data['redshifts'][z_inds1] )
    z_dist0 = np.abs( acc_time - data['redshifts'][z_inds0] )

    accSnap = data['snaps'][z_inds1]

    with np.errstate(invalid='ignore'): # ignore nan comparison RuntimeWarning
        ww = np.where( z_dist0 < z_dist1 )[0]

    if len(ww):
        accSnap[ww] = data['snaps'][z_inds0[ww]]

    # nan acc_time's (never inside rvir) got assigned to the earliest snapshot, flag them as -1
    accSnap[np.isnan(acc_time)] = -1

    # make a mapping from snapshot number -> mpb[index]
    mpbIndexMap = np.zeros( mpb['SnapNum'].max()+1, dtype='int32' ) - 1
    mpbIndexMap[ mpb['SnapNum'] ] = np.arange(mpb['SnapNum'].max())

    # make a mapping from snapshot number -> data[index]
    dataIndexMap = np.zeros( data['snaps'].max()+1, dtype='int32' ) - 1
    dataIndexMap[ data['snaps'] ] = np.arange(data['snaps'].max())

    # start loop to determine each tracer
    for i in range(data['TracerIDs'].size):
        if i % int(data['TracerIDs'].size/10) == 0:
            print(' %4.1f%%' % (float(i)/data['TracerIDs'].size*100.0))

        # never inside rvir -> accMode is undetermined
        if accSnap[i] == -1:
            accMode[i] = ACCMODE_NONE
            continue

        # accretion time determined as earliest snapshot (e.g. z=10), we label this smooth
        if accSnap[i] == data['snaps'].min():
            accMode[i] = ACCMODE_SMOOTH
            continue

        # pull out indices
        mpbIndAcc  = mpbIndexMap[accSnap[i]]
        dataIndAcc = dataIndexMap[accSnap[i]]

        assert mpbIndAcc != -1
        assert dataIndAcc != -1
        assert data['snaps'][dataIndAcc] == mpb['SnapNum'][mpbIndAcc]
        assert data['snaps'][dataIndAcc] == accSnap[i]

        # merger?
        mpbSubfindID_AtAcc = mpb['SubfindID'][ mpbIndAcc ]
        trParSubfindID_AtAcc = data['subhalo_id'][ dataIndAcc, i]

        if mpbSubfindID_AtAcc != trParSubfindID_AtAcc:
            # mismatch of MPB subfind ID and tracer parent subhalo ID at z_acc
            accMode[i] = ACCMODE_MERGER

            #assert trParSubfindID_AtAcc != -1 # this is allowed
            assert mpbSubfindID_AtAcc != -1 # guess this is techncially possible? if we have 
            # for instance a skip and a ghost insert, then a rvir crossing could fall in a 
            # snapshot where the mpb was not defined (we hit this for 104 override?)
            continue

        # smooth?
        trParAtAccAndEarlier_HaveAtSnapNums = data['snaps'][ dataIndAcc : ]
        mpbInds_AtMatchingSnapNums = mpbIndexMap[ trParAtAccAndEarlier_HaveAtSnapNums ]

        trParSubfindIDs_AtAccAndEarlier = data['subhalo_id'][ dataIndAcc : , i ] # squeeze?
        mpbSubfindIDs_AtAccAndEarlier = mpb['SubfindID'][ mpbInds_AtMatchingSnapNums ].copy()

        # wherever trParSubfindIDs_AtAccAndEarlier is -1 (not in any subhalo), overwrite 
        # the local mpbSubfindIDs_AtAccAndEarlier with these same values for the logic below
        ww = np.where( trParSubfindIDs_AtAccAndEarlier == -1 )[0]
        if len(ww):
            mpbSubfindIDs_AtAccAndEarlier[ww] = -1

        # debug verify:
        assert trParSubfindIDs_AtAccAndEarlier.size == mpbSubfindIDs_AtAccAndEarlier.size
        mpb_SnapVerify = mpb['SnapNum'][ mpbInds_AtMatchingSnapNums ]
        assert np.array_equal(mpb_SnapVerify, trParAtAccAndEarlier_HaveAtSnapNums)

        # agreement of MPB subfind IDs and tracer parent subhalo IDs at all z>=z_acc
        if np.array_equal(trParSubfindIDs_AtAccAndEarlier, mpbSubfindIDs_AtAccAndEarlier):
            accMode[i] = ACCMODE_SMOOTH
            continue

        # stripped? by definition, if we make it here we have:
        #   mpbSubfindID_AtAcc == trParSubfindID_AtAcc
        #   trParSubfindIDs_AtAccAndEarlier != mpbSubfindIDs_AtAccAndEarlier
        accMode[i] = ACCMODE_STRIPPED

    # stats
    nNone   = len( np.where(accMode == ACCMODE_NONE)[0] )
    nBad    = len( np.where(accMode == 0)[0] )
    nSmooth = len( np.where(accMode == ACCMODE_SMOOTH)[0] )
    nMerger = len( np.where(accMode == ACCMODE_MERGER)[0] )
    nStrip  = len( np.where(accMode == ACCMODE_STRIPPED)[0] )

    assert nBad == 0
    nD = len(str(accMode.size))

    print(' Smooth:   [ %*d of %*d ] %4.1f%%' % (nD,nSmooth,nD,accMode.size,(100.0*nSmooth/accMode.size)) )
    print(' Merger:   [ %*d of %*d ] %4.1f%%' % (nD,nMerger,nD,accMode.size,(100.0*nMerger/accMode.size)) )
    print(' Stripped: [ %*d of %*d ] %4.1f%%' % (nD,nStrip,nD,accMode.size,(100.0*nStrip/accMode.size)) )
    print(' None:     [ %*d of %*d ] %4.1f%%' % (nD,nNone,nD,accMode.size,(100.0*nNone/accMode.size)) )

    # save
    with h5py.File(saveFilename,'w') as f:
        f['accMode'] = accMode

    print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])
    return accMode

# --- old ---

def guinevereData():
    """ Data for Guinevere. """
    # config
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    parPartTypes = ['gas','stars']
    toRedshift   = 0.5
    trFields     = ['tracer_windcounter'] 
    parFields    = ['pos','vel','temp','sfr']
    outPath      = sP.derivPath

    # subhalo list
    subhaloIDs = np.loadtxt(sP.derivPath + 'guinevere.list.subs.txt', dtype='int32')

    subhalosTracersTimeEvo(sP, subhaloIDs, toRedshift, trFields, parFields, parPartTypes, outPath)

def plotPosTempVsRedshift():
    """ Plot trMC position (projected) and temperature evolution vs redshift. """
    from cosmo.util import correctPeriodicPosBoxWrap

    # config
    axis1 = 0
    axis2 = 2
    alpha = 0.05
    boxSize = 1000.0 # ckpc/h
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    shNums = [int(s[:-5].rsplit('_',1)[1]) for s in glob.glob(sP.derivPath + 'subhalo_*.hdf5')]
    shNum = shNums[0]

    # load
    with h5py.File(sP.derivPath + 'subhalo_'+str(shNum)+'.hdf5') as f:
        pos  = f['pos'][()]
        temp = f['temp'][()]
        sfr  = f['sfr'][()]
        redshift = f['Redshift'][()]

        #pt = cosmo.load.groupCatSingle(sP, subhaloID=f['SubhaloID'][0])['SubhaloPos']

    # plot
    if 0:
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        ax.set_xlim(pos[:,:,axis1].mean() + np.array([-boxSize,boxSize]))
        ax.set_ylim(pos[:,:,axis2].mean() + np.array([-boxSize,boxSize]))
        ax.set_aspect(1.0)

        ax.set_title('Evolution of tracer positions with time check')
        ax.set_xlabel('x [ckpc/h]')
        ax.set_ylabel('y [ckpc/h]')

        # make relative and periodic correct
        correctPeriodicPosBoxWrap(pos, sP)

        #for i in np.arange(pos.shape[1]):
        for i in np.arange(10000):
            ax.plot(pos[:,i,axis1], pos[:,i,axis2], '-', color='#333333', alpha=alpha, lw=1.0)

        fig.tight_layout()
        plt.savefig('trMC_checkPos_'+sP.simName+'_sh'+str(shNum)+'.pdf')
        plt.close(fig)

    # plot 2
    if 1:
        fig = plt.figure(figsize=(16,8))
        ax = fig.add_subplot(111)
        ax.set_xlim([0.0,0.5])
        ax.set_ylim([3.5,8.0])

        ax.set_title('Evolution of tracer temperatures with time check')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Temp [log K]')

        #for i in np.arange(temp.shape[1]):
        for i in [205]:
            # plot only snapshots with temp (in gas) and sfr=0 (not eEOS)
            #ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
            #if not np.count_nonzero(ww):
            #    continue
            #ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', color='#333333', alpha=alpha*5, lw=2.0)

            # plot only those tracers which have been always in gas with sfr=0 their whole track
            #ww = np.isnan(temp[:,i]) | (sfr[:,i] > 0.0)
            #if np.count_nonzero(ww):
            #    continue
            #ax.plot(redshift, np.squeeze(temp[:,i]), '-', color='#333333', alpha=alpha, lw=1.0)

            # test
            ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
            ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', alpha=alpha*10, lw=2.0, label='gas sfr==0')

            ww = np.isfinite(temp[:,i])
            ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '--', alpha=alpha*10, lw=2.0, label='gas sfr any')

            ax.plot(redshift, np.squeeze(temp[:,i]), 'o', alpha=alpha*10, lw=2.0, label='star or gas')

            print(temp[:,i])

        # test
        ax.legend()

        fig.tight_layout()
        plt.savefig('trMC_checkTempB_'+sP.simName+'_sh'+str(shNum)+'.pdf')
        plt.close(fig)

def plotStarFracVsRedshift():
    """ Plot the fraction of tracers in stars vs. gas parents vs redshift. """
    # config
    alpha = 0.3
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    shNums = [int(s[:-5].rsplit('_',1)[1]) for s in glob.glob(sP.derivPath + 'subhalo_*.hdf5')]

    # plot
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.set_xlim([0.0,0.5])
    #ax.set_ylim([0.0,0.4])

    ax.set_title('')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Fraction of trMC in Stellar Parents')

    for shNum in shNums:
        # load
        with h5py.File(sP.derivPath + 'subhalo_'+str(shNum)+'.hdf5') as f:
            temp = f['temp'][()]
            sfr  = f['sfr'][()]
            redshift = f['Redshift'][()]

        # calculate fraction at each snapshot (using temp=nan->in star)
        fracInStars = np.zeros( temp.shape[0] )
        for i in np.arange(temp.shape[0]):
            numInStars = np.count_nonzero(np.isfinite(temp[i,:]))
            fracInStars[i] = numInStars / float(temp.shape[1])

        ax.plot(redshift, fracInStars, '-', color='#333333', alpha=alpha, lw=1.0)

    fig.tight_layout()
    plt.savefig('trMC_starFracs_'+sP.simName+'.pdf')
    plt.close(fig)