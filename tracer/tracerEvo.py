"""
tracerEvo.py
  Analysis for evolution of tracer quantities in time (for cosmo boxes/zooms).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import h5py
import numpy as np
from os.path import isfile, isdir
from os import mkdir

from tracer.tracerMC import subhaloTracersTimeEvo, subhalosTracersTimeEvo, \
  globalAllTracersTimeEvo, globalTracerMPBMap, defParPartTypes
from cosmo.mergertree import mpbSmoothedProperties
from cosmo.util import redshiftToSnapNum

# integer flags for accretion modes
ACCMODE_NONE     = -1
ACCMODE_SMOOTH   = 1
ACCMODE_MERGER   = 2
ACCMODE_STRIPPED = 3

# types of extrema which we know how to calculate
allowedExtTypes = ['min','min_b015','max','max_b015']

# default value: maximum redshift to track tracer properties back to
maxRedshift = 10.0

def zoomDataDriver(sP, fields, snapStep=1):
    """ Run and save data files for tracer evolution in several quantities of interest. """
    from util import simParams

    #sP = simParams(res=11, run='zooms2', redshift=2.0, hInd=2)
    #fields = ['tracer_maxtemp','tracer_maxent','rad_rvir','vrad','entr','temp','sfr','subhalo_id']
    subhaloID = sP.zoomSubhaloID

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

def tracersTimeEvo(sP, fieldName, snapStep=None, all=True):
    """ Wrapper to handle zoom vs. box load. """
    if sP.isZoom:
        assert snapStep is not None
        r = subhaloTracersTimeEvo(sP, sP.zoomSubhaloID, [fieldName], snapStep)
    else:
        r = globalAllTracersTimeEvo(sP, fieldName)

    # global load
    if all is True or (sP.haloInd is None and sP.subhaloInd is None):
        return r

    # restrict based on [halo/subhalo] for a fullbox
    # Note! For globalAllTracersTimeEvo(), we could move this inside and actually do a restricted 
    # load, instead of a load and then subset... going to be mandatory
    assert not sP.isZoom
    assert fieldName != 'meta'

    meta, nTracerTot = tracersMetaOffsets(sP)

    for key in r:
        if key == 'redshifts' or key == 'snaps':
            continue

        # size is (Nsnaps,NtrSubset)
        subset = np.zeros( (r[key].shape[0],nTracerTot), dtype=r[key].dtype )
        offset = 0

        for pt in meta.keys():
            if meta[pt]['length'] == 0:
                continue

            for i in range(r[key].shape[0]):
                # we anyways made these non-contiguous on disk, so load looping over snap dim
                subset[i, offset : offset + meta[pt]['length']] = \
                  r[key][ i, meta[pt]['offset'] : meta[pt]['offset']+meta[pt]['length'] ]

            offset += meta[pt]['length']

        r[key] = subset # replace

    return r

def tracersMetaOffsets(sP):
    """ For a fullbox sP and either sP.haloInd or sP.subhaloInd specified, load and return the needed 
    offsets to load a [halo/subhalo]-restricted part of any of the tracer_tracks data. """
    assert (sP.haloInd is not None) ^ (sP.subhaloInd is not None)
    assert not sP.isZoom

    saveFilename = sP.postPath + '/tracer_tracks/tr_all_groups_%d_meta.hdf5' % (sP.snap)

    if sP.haloInd is not None:
        gName = 'Halo'
        dInd = sP.haloInd
    if sP.subhaloInd is not None:
        gName = 'Subhalo'
        dInd = sP.subhaloInd

    # load
    r = {}

    with h5py.File(saveFilename,'r') as f:
        for ptName in defParPartTypes:
            r[ptName] = {}
            r[ptName]['length'] = f[gName]['TracerLength'][ptName][dInd]
            r[ptName]['offset'] = f[gName]['TracerOffset'][ptName][dInd]

    nTracerTot = np.sum([r[ptName]['length'] for ptName in r])

    return r, nTracerTot

def loadAllOrRestricted(sP, saveFilename, datasetName=None):
    """ Load a single datasetName from a HDF5 tracer save file, either the entire dataset or 
    restricted to a [halo/subhalo] subset if sP.haloInd or sP.subhaloInd is specified for a 
    fullbox run. """

    # return for all tracers
    if sP.isZoom or (sP.haloInd is None and sP.subhaloInd is None):
        with h5py.File(saveFilename,'r') as f:
            return f[datasetName][()]

    # get offsets from meta and do [halo/subhalo]-restricted load and return (concat types)
    meta, nTracerTot = tracersMetaOffsets(sP)
    
    if nTracerTot == 0:
        return None        

    with h5py.File(saveFilename,'r') as f:
        r = np.zeros( nTracerTot, dtype=f[datasetName].dtype )
        offset = 0

        for pt in meta.keys():
            if meta[pt]['length'] == 0:
                continue

            r[offset : offset + meta[pt]['length']] = \
              f[datasetName][ meta[pt]['offset'] : meta[pt]['offset']+meta[pt]['length'] ]
            offset += meta[pt]['length']

    return r

def accTime(sP, snapStep=1, rVirFac=1.0):
    """ Calculate accretion time for each tracer (and cache), as the earliest (highest redshift) crossing 
    of the virial radius of the MPB halo. Uses the 'rad_rvir' field. 
    Argument: rVirFac = what fraction of the virial radius denotes the accretion time? """

    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_time_%d.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep,rVirFac*100)
    else:
        saveFilename = sP.derivPath + '/trTimeEvo/acc_time_snap_%d-%d-%d_r%d.hdf5' % \
          (sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep,rVirFac*100)

    if not isdir(sP.derivPath + '/trTimeEvo'):
        mkdir(sP.derivPath + '/trTimeEvo')

    # load pre-existing
    if isfile(saveFilename):
        return loadAllOrRestricted(sP,saveFilename,'accTimeInterp')

    print('Calculating new accTime for [%s]...' % sP.simName)

    # calculate new: load radial histories
    data = tracersTimeEvo(sP, 'rad_rvir', snapStep, all=True)

    # reverse so that increasing indices are increasing snapshot numbers
    data2d = data['rad_rvir'][::-1,:]

    data['snaps'] = data['snaps'][::-1]
    data['redshifts'] = data['redshifts'][::-1]
    
    data2d[~np.isfinite(data2d)] = rVirFac * 10 # set NaN (untracked MPB) to large values (outside)

    # set mask to one for all radii less than factor
    mask2d = np.zeros_like( data2d, dtype='int16' )
    ww = np.where( data2d < rVirFac )
    mask2d[ww] = 1

    # along second axis (trInds), take first index (lowest snap number inside) which is nonzero
    firstSnapInsideInd = np.argmax( mask2d, axis=0 )

    # interp between index and previous (one snap before first time inside) for non-discrete answer
    nTr = data['rad_rvir'].shape[1]
    accTimeInterp = np.zeros( nTr, dtype='float32' )

    for i in range(nTr):
        if i % int(nTr/100) == 0:
            print(' %4.1f%%' % (float(i)/nTr*100.0))

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

def accTimesToClosestSnaps(data, acc_time, indsNotSnaps=False):
    """ Return the nearest snapshot number to each acc_time (which are redshifts) using the 
    data['redshifts'] and data['snaps'] mapping. By default, return simulation snapshot 
    number, unless indsNotSnaps==True, in which case return the indices into the first dimension 
    of data[field] for each tracer of the second dimension. 
    Note that acc_time can be any array of redshifts (e.g. also extremum times). """
    z_inds1 = np.searchsorted( data['redshifts'], acc_time )

    ww = np.where(z_inds1 == data['redshifts'].size)
    z_inds1[ww] -= 1

    z_inds0 = z_inds1 - 1

    z_dist1 = np.abs( acc_time - data['redshifts'][z_inds1] )
    z_dist0 = np.abs( acc_time - data['redshifts'][z_inds0] )

    if indsNotSnaps:
        accSnap = z_inds1
    else:
        accSnap = data['snaps'][z_inds1]

    with np.errstate(invalid='ignore'): # ignore nan comparison RuntimeWarning
        ww = np.where( z_dist0 < z_dist1 )

    if indsNotSnaps:
        accSnap[ww] = [z_inds0[ww]]
    else:
        accSnap[ww] = data['snaps'][z_inds0[ww]]

    # nan acc_time's (never inside rvir) got assigned to the earliest snapshot, flag them as -1
    accSnap[np.isnan(acc_time)] = -1

    return accSnap

def accMode(sP, snapStep=1):
    """ Derive an 'accretion mode' categorization for each tracer based on its group membership history. 
    Specifically, separate all tracers into one of [smooth/merger/stripped] defined as:
      - smooth: child of MPB or no subhalo at all z>=z_acc 
      - merger: child of subhalo other than the MPB at z=z_acc 
      - stripped: child of MPB or no subhalo at z=z_acc, but child of non-MPB subhalo at any z>z_acc 
    Where z_acc is the accretion redshift defined as the first (highest z) crossing of the virial radius. """

    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_acc_mode.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep)
    else:
        saveFilename = sP.derivPath + '/trTimeEvo/acc_mode_s%d-%d-%d.hdf5' % \
          (sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep)

    # load pre-existing
    if isfile(saveFilename):
        return loadAllOrRestricted(sP,saveFilename,'accMode')

    print('Calculating new accMode for [%s]...' % sP.simName)

    # load accTime, subhalo_id tracks, and MPB history
    data = tracersTimeEvo(sP, 'subhalo_id', snapStep, all=True)

    if sP.isZoom:
        mpb = mpbSmoothedProperties(sP, sP.zoomSubhaloID)
    else:
        mpbGlobal = globalTracerMPBMap(sP, halos=True, retMPBs=True)

    acc_time = accTime(sP, snapStep=snapStep)

    # allocate return
    nTr = acc_time.size
    accMode = np.zeros( nTr, dtype='int8' )

    # closest snapshot for each accretion time
    accSnap = accTimesToClosestSnaps(data, acc_time)

    assert nTr == data['subhalo_id'].shape[1] == accSnap.size

    # prepare a mapping from snapshot number -> mpb[index]
    if sP.isZoom:
        mpbIndexMap = np.zeros( mpb['SnapNum'].max()+1, dtype='int32' ) - 1
        mpbIndexMap[ mpb['SnapNum'] ] = np.arange(mpb['SnapNum'].max())
    else:
        mpbIndexMap = np.zeros( data['snaps'].max()+1, dtype='int32' )

    # make a mapping from snapshot number -> data[index]
    dataIndexMap = np.zeros( data['snaps'].max()+1, dtype='int32' ) - 1
    dataIndexMap[ data['snaps'] ] = np.arange(data['snaps'].max())

    # start loop to determine each tracer
    for i in range(nTr):
        if i % int(nTr/100) == 0:
            print(' %4.1f%%' % (float(i)/nTr*100.0))

        # never inside rvir -> accMode is undetermined
        if accSnap[i] == -1:
            accMode[i] = ACCMODE_NONE
            continue

        # accretion time determined as earliest snapshot (e.g. z=10), we label this smooth
        if accSnap[i] == data['snaps'].min():
            accMode[i] = ACCMODE_SMOOTH
            continue

        # (only needed for periodic boxes with multiple MPBs)
        if not sP.isZoom:
            # tracer is in FoF halo with no primary subhalo at sP.snap (no MPB)
            if mpbGlobal['subhalo_id'][i] == -1:
                accMode[i] = ACCMODE_NONE
                continue

            # extract MPB used for this tracer
            mpb = mpbGlobal['mpbs'][ mpbGlobal['subhalo_id'][i] ]

            # create new mapping into MPB for this tracer
            mpbIndexMap.fill(-1)
            mpbIndexMap[ mpb['SnapNum'] ] = np.arange(mpb['SnapNum'].max())

        # pull out indices
        mpbIndAcc  = mpbIndexMap[accSnap[i]]
        dataIndAcc = dataIndexMap[accSnap[i]]

        # in fullboxes, we may not even have the MPB back to the accTime (not currently allowed for zooms)
        if mpbIndAcc == -1 and not sP.isZoom:
            accMode[i] = ACCMODE_NONE
            continue

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

        # wherever mpbInds_AtMachingSnapNums is -1 (MPB is untracked at this snapshot), 
        # rewrite the local mpbSubfindIDs_AtAccAndEarlier with -1 (i.e. once the MPB becomes 
        # untracked, if at that point the tracer is within no subhalo, then we allow this to 
        # count as smooth)
        ww = np.where( mpbInds_AtMatchingSnapNums == -1 )
        mpbSubfindIDs_AtAccAndEarlier[ww] = -1

        # wherever trParSubfindIDs_AtAccAndEarlier is -1 (not in any subhalo), overwrite 
        # the local mpbSubfindIDs_AtAccAndEarlier with these same values for the logic below
        ww = np.where( trParSubfindIDs_AtAccAndEarlier == -1 )
        mpbSubfindIDs_AtAccAndEarlier[ww] = -1

        # debug verify:
        assert trParSubfindIDs_AtAccAndEarlier.size == mpbSubfindIDs_AtAccAndEarlier.size
        mpb_SnapVerify = mpb['SnapNum'][ mpbInds_AtMatchingSnapNums ]
        ww = np.where( mpbInds_AtMatchingSnapNums >= 0 ) # mpb tracked only)
        assert np.array_equal(mpb_SnapVerify[ww], trParAtAccAndEarlier_HaveAtSnapNums[ww])

        # agreement of MPB subfind IDs and tracer parent subhalo IDs at all z>=z_acc
        if np.array_equal(trParSubfindIDs_AtAccAndEarlier, mpbSubfindIDs_AtAccAndEarlier):
            accMode[i] = ACCMODE_SMOOTH
            continue

        # stripped? by definition, if we make it here we have:
        #   mpbSubfindID_AtAcc == trParSubfindID_AtAcc
        #   trParSubfindIDs_AtAccAndEarlier != mpbSubfindIDs_AtAccAndEarlier
        accMode[i] = ACCMODE_STRIPPED

    # stats
    nBad    = np.count_nonzero(accMode == 0)
    nNone   = np.count_nonzero(accMode == ACCMODE_NONE)
    nSmooth = np.count_nonzero(accMode == ACCMODE_SMOOTH)
    nMerger = np.count_nonzero(accMode == ACCMODE_MERGER)
    nStrip  = np.count_nonzero(accMode == ACCMODE_STRIPPED)

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

def valExtremum(sP, fieldName, snapStep=1, extType='max'):
    """ Calculate an extremum (e.g. min or max) or other single quantity (e.g. avg) for every 
    tracer for a given property (e.g. temp). For [gas] values affected by the modified Utherm of 
    the star-forming eEOS (temp, entr) we exclude times when SFR>0. This is then also consistent 
    with what is done in the code for tracer_max* recorded values. """
    assert extType in allowedExtTypes
    assert isinstance(fieldName,basestring)

    # check for existence
    if sP.isZoom:
        saveFilename = sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_%s_%s.hdf5' % \
          (sP.zoomSubhaloID,True,sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep,fieldName,extType)
    else:
        saveFilename = sP.derivPath + '/trTimeEvo/%s_%s_snap_%d-%s-%d.hdf5' % \
          (fieldName,extType,sP.snap,redshiftToSnapNum(maxRedshift,sP),snapStep)

    # load pre-existing
    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # calculate new: load required field
    print('Calculating new valExtremum [%s] for [%s]...' % (fieldName,sP.simName))

    data = tracersTimeEvo(sP, fieldName, snapStep, all=True)

    # mask sfr>0 points (for gas cell properties which are modified by eEOS)
    if fieldName in ['temp','entr']:
        sfr = tracersTimeEvo(sP, 'sfr', snapStep, all=True)

        with np.errstate(invalid='ignore'): # ignore nan comparison RuntimeWarning
            ww = np.where( sfr['sfr'] > 0.0 )
        data[fieldName][ww] = np.nan

    # mask t>t_acc_015rvir points (only take extremum for time "before" first 0.15rvir crossing)
    if '_b015' in extType:
        acc_time = accTime(sP, snapStep=snapStep, rVirFac=0.15)
        inds = accTimesToClosestSnaps(data, acc_time, indsNotSnaps=True)

        for i in np.arange( inds.size ):
            if inds[i] == -1:
                continue
            data[fieldName][:inds[i],i] = np.nan

    # which functions to use
    if 'min' in extType:
        fval = np.nanmin
        fargval = np.nanargmin
    if 'max' in extType:
        fval = np.nanmax
        fargval = np.nanargmax

    # calculate extremum value
    r = {}
    r['val'] = fval( data[fieldName], axis=0 )

    # calculate the redshift when it occured
    r['time'] = np.zeros( data[fieldName].shape[1], dtype='float32' )
    r['time'][:] = np.nan

    ww = np.where( ~np.isnan(r['val']) )

    extInd = fargval( data[fieldName][:,ww], axis=0 )
    r['time'][ww] = data['redshifts'][extInd]

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r.keys():
            f[key] = r[key]

    print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])
    return r

def trValsAtRedshifts(sP, valName, redshifts, snapStep=1):
    """ Return some property from the tracer evolution tracks (e.g. rad_rvir, temp, tracer_maxent) 
    given by valName at the times in the simulation given by redshifts. """
    raise Exception('todo, seems not to work at all, unclear')
    # load
    assert isinstance(valName,basestring)
    data = tracersTimeEvo(sP, valName, snapStep, all=False)

    assert data[valName].ndim == 2 # need to verify logic herein for ndim==3 (e.g. pos/vel) case

    # map times to data indices
    inds = accTimesToClosestSnaps(data, redshifts, indsNotSnaps=True)

    assert inds.max() < data[valName].shape[0]
    assert inds.size == data[valName].shape[1]

    # inds gives, for each tracer (second dimension of data[valName]), the index into the first 
    # dimension of data[valName] that we want to extract. convert this implicit pair into 1d inds
    inds_dim2 = np.arange(inds.shape[0])
    inds_1d = np.ravel_multi_index( (inds, inds_dim2), data[valName].shape, mode='clip' )

    # make a view to the contiguous flattened/1d array
    data_1d = np.ravel( data[valName] )

    # pull out values and flag those which were always invalid as nan
    trVals = data_1d[ inds_1d ]

    ww = np.where( inds == -1 )
    
    if trVals.dtype == 'float32' : trVals[ww] = np.nan
    if trVals.dtype == 'int32'   : trVals[ww] = -1
    assert trVals.dtype == 'float32' or trVals.dtype == 'int32'

    if 0: # debug verify
        for i in np.arange(trVals.size):
            if np.isnan(trVals[i]):
                continue
            assert trVals[i] == data[valName][inds[i],i]

    return trVals

def trValsAtExtremumTimes(sP, valName, extName, extType='max', snapStep=1):
    """ Wrap trValsAtRedshifts() to specifically give trVals at the redshifts corresponding 
    to the extremum times corresponding to extName, extType. """
    ext = valExtremum(sP, extName, snapStep=snapStep, extType=extType)
    return trValsAtRedshifts(sP, valName, ext['time'], snapStep=snapStep)

def trValsAtAccTimes(sP, valName, rVirFac=1.0, snapStep=1):
    """ Wrap trValsAtRedshifts() to specifically give trVals at the redshifts corresponding 
    to the accretion times determined as the first crossing of rVirFac times the virial radius. """
    acc_time = accTime(sP, snapStep=snapStep, rVirFac=rVirFac)
    return trValsAtRedshifts(sP, valName, acc_time, snapStep=snapStep)

def mpbValsAtRedshifts(sP, valName, redshifts, snapStep=1):
    """ Return some halo property, per tracer, from the main progenitor branch (MPB) (e.g. tvir, spin) 
    given by valName at the times in the simulation given by redshifts. """

    # load
    assert sP.isZoom # todo for boxes (handle sP.subhaloInd and sP.haloInd as well)
    assert isinstance(valName,basestring)

    if sP.isZoom:
        mpb = mpbSmoothedProperties(sP, sP.zoomSubhaloID, extraFields=valName)
    else:
        mpbGlobal = globalTracerMPBMap(sP, halos=True, retMPBs=True, extraFields=valName)

    data = {}

    # pull out either smoothed value or raw field straight from trees
    if valName in mpb['sm'].keys():
        data['val'] = mpb['sm'][valName]
    if valName in mpb.keys():
        data['val'] = mpb[valName]

    assert 'val' in data
    assert data['val'].shape[0] == mpb['Redshift'].shape[0]
    assert data['val'].shape[0] == mpb['SnapNum'].shape[0]

    # map times to snapshot numbers
    data['redshifts'] = mpb['Redshift']
    data['snaps']     = mpb['SnapNum']

    inds = accTimesToClosestSnaps(data, redshifts, indsNotSnaps=True)

    if data['val'].ndim == 1:
        return data['val'][inds]
    if data['val'].ndim == 2:
        return data['val'][inds,:]

    raise Exception('Should not reach here.')

def mpbValsAtExtremumTimes(sP, valName, extName, extType='max', snapStep=1):
    """ Wrap mpbValsAtRedshifts() to specifically give mpbVals at the redshifts corresponding 
    to the extremum times corresponding to extName, extType. """
    ext = valExtremum(sP, extName, snapStep=snapStep, extType=extType)
    return mpbValsAtRedshifts(sP, valName, ext['time'], snapStep=snapStep)

def mpbValsAtAccTimes(sP, valName, rVirFac=1.0, snapStep=1):
    """ Wrap mpbValsAtRedshifts() to specifically give mpbVals at the redshifts corresponding 
    to the accretion times determined as the first crossing of rVirFac times the virial radius. """
    acc_time = accTime(sP, snapStep=snapStep, rVirFac=rVirFac)
    return mpbValsAtRedshifts(sP, valName, acc_time, snapStep=snapStep)
    