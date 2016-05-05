"""
tracerEvo.py
  Analysis for evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import h5py
import numpy as np
from os.path import isfile

from tracer.tracerMC import subhaloTracersTimeEvo, subhalosTracersTimeEvo
from cosmo.mergertree import mpbSmoothedProperties
from cosmo.util import redshiftToSnapNum

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

def accMode(sP, snapStep=1):
    """ Derive an 'accretion mode' categorization for each tracer based on its group membership history. 
    Specifically, separate all tracers into one of [smooth/merger/stripped] defined as:
      - smooth: child of MPB or no subhalo at all z>=z_acc 
      - merger: child of subhalo other than the MPB at z=z_acc 
      - stripped: child of MPB or no subhalo at z=z_acc, but child of non-MPB subhalo at any z>z_acc 
    Where z_acc is the accretion redshift defined as the first (highest z) crossing of the virial radius. """

    # check for existence
    saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_mode.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(10.0,sP),snapStep)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            return f['accMode'][()]

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
