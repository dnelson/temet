"""
util/tracerMC.py
  Helper functions to efficiently work with the Monte Carlo tracers.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

from util import simParams
from illustris_python.util import partTypeNum
from os.path import isfile
import cosmo.load
import numpy as np
import h5py
import pdb
import glob
import matplotlib.pyplot as plt

debug = False # enable expensive debug consistency checks and verbose output

def match(ar1, ar2, uniq=False):
    """ My version of numpy.in1d with invert=False. Return is a ndarray of indices into ar1, 
        corresponding to elements which exist in ar2. Meant to be used e.g. as ar1=all IDs in
        snapshot, and ar2=some IDs to search for, where ar2 could be e.g. ParentID from the 
        tracers, in which case they are generally not unique (multiple tracers can exist in the 
        same parent). """
    import time
    start = time.time()

    # flatten both arrays (behavior for the first array could be different)
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # tuning: special case for small B arrays (significantly faster than the full sort)
    if len(ar2) < 10 * len(ar1) ** 0.145:
        mask = np.zeros(len(ar1), dtype=np.bool)
        for a in ar2:
            mask |= (ar1 == a)
        return mask

    # otherwise use sorting of the concatenated array: here we use a stable 'mergesort', 
    # such that the values from the first array are always before the values from the second
    start_uniq = time.time()
    if not uniq:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)
    end_uniq = time.time()

    ar = np.concatenate((ar1, ar2))

    start_sort = time.time()
    order = ar.argsort(kind='mergesort')
    end_sort = time.time()

    # construct the output index list
    ar = ar[order]
    bool_ar = (ar[1:] == ar[:-1])

    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = bool_ar

    if uniq:
        inds = ret[:len(ar1)].nonzero()[0]
    else:
        inds = ret[rev_idx].nonzero()[0]

    if debug:
        print(' match: '+str(round(time.time()-start,2))+' sec '+\
              '[sort: '+str(round(end_sort-start_sort,2))+' sec] '+\
              '[uniq: '+str(round(end_uniq-start_uniq,2))+' sec]')

    return inds

def match2(ar1, ar2):
    """ My alternative version of numpy.in1d with invert=False, which is more similar to calcMatch(). 
        Return is two ndarrays. The first is indices into ar1, the second is indices into ar2, such 
        that ar1[inds1] = ar2[inds2]. Both ar1 and ar2 are assumed to contain unique values. Can be 
        used to e.g. crossmatch between two TracerID sets from different snapshots, or between some 
        ParentIDs and ParticleIDs of other particle types. The approach is a concatenated mergesort 
        of ar1,ar2 combined, therefore O( (N_ar1+N_ar2)*log(N_ar1+N_ar2) ) complexity. """
    import time
    start = time.time()

    # flatten both arrays (behavior for the first array could be different)
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # make concatenated list of ar1,ar2 and a combined list of indices, and a flag for which array
    # each index belongs to (0=ar1, 1=ar2)
    c = np.concatenate((ar1, ar2))
    ind = np.concatenate(( np.arange(ar1.size), np.arange(ar2.size) ))
    vec = np.concatenate(( np.zeros(ar1.size, dtype='int16'), np.zeros(ar2.size, dtype='int16')+1 ))

    # sort combined list
    order = c.argsort(kind='mergesort')

    c   = c[order]
    ind = ind[order]
    vec = vec[order]

    # find duplicates in sorted combined list
    firstdup = np.where( (c == np.roll(c,-1)) & (vec != np.roll(vec,-1)) )[0]

    if firstdup.size == 0:
        return None,None

    dup = np.zeros( firstdup.size*2, dtype='uint64' )
    even = np.arange( firstdup.size, dtype='uint64' )*2

    dup[even] = firstdup
    dup[even+1] = firstdup+1

    ind = ind[dup]
    vec = vec[dup]

    inds1 = ind[np.where(vec == 0)]
    inds2 = ind[np.where(vec == 1)]

    if debug:
        if not np.array_equal(ar1[inds1],ar2[inds2]):
            raise Exception('match2 fail')
        print(' match2: '+str(round(time.time()-start,2))+' sec')

    return inds1, inds2

def match3(ar1, ar2, firstSorted=False):
    """ Returns index arrays i1,i2 of the matching elementes between ar1 and ar2. The elements of ar2 need 
        not be unique. For every matched element of ar2, the return i1 gives the index in ar1 where it can 
        be found. For every matched element of ar1, the return i2 gives the index in ar2 where it can be 
        found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves the order of ar2. Therefore, 
        if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, ar2=set of TracerIDs to locate) 
        then ar2[i2] = ar2. The approach is one sort of ar1 followed by bisection search for each element 
        of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= O(N_ar1*log(N_ar1)) complexity so 
        long as N_ar2 << N_ar1. """
    import time
    start = time.time()

    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)
        ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)
        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index, mode="clip")

    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None,None

    if debug:
        if not np.array_equal(ar1[ar1_inds],ar2[ar2_inds]):
            raise Exception('match3 fail')
        print(' match3: '+str(round(time.time()-start,2))+' sec')

    return ar1_inds, ar2_inds

def getTracerChildren(sP, parentSearchIDs, inds=False, ParentID=None, ParentIDSortInds=None):
    """ For an input list of parent IDs (a UNIQUE list of an unknown mixture of gas/star/BH IDs), return 
        the complete list of child MC tracers belonging to those parents (either their IDs or their 
        global indices in the snap). """
    # if global ParentID for the snapshot is not previously loaded and passed in, load it now
    if ParentID is None:
        ParentID = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID')

    if debug:
        print(' ParentID: Size: '+str(ParentID.size)+' min: '+str(ParentID.min())+' max: '+str(ParentID.max()))
        print(' parentSearchIDs: Size: '+str(parentSearchIDs.size)+\
              ' min: '+str(parentSearchIDs.min())+\
              ' max: '+str(parentSearchIDs.max()))

    # ID crossmatch: find matching elements
    # trInds,_ = match3(ParentID,parentSearchIDs) is not appropraite here: we have possibly multiple 
    # matches inside ParentID for each element of parentSearchIDs. Instead, we can do 
    # _,trInds = match3(parentSearchIDs,ParentID) and get all the indices of ParentID which are found 
    # anywhere inside parentSearchIDs
    _,trInds = match3(parentSearchIDs, ParentID)
    
    if debug:
        # old method: keep for debugging
        comp1 = match(ParentID, parentSearchIDs)
        
        if not np.array_equal(trInds,comp1):
            raise Exception('trInds comp1 mismatch')

    if debug:
        print(' trInds: Size: '+str(trInds.size)+' min: '+str(trInds.min())+' max: '+str(trInds.max()))

    # dealing with a pre-sorted ParentID? If so, take the inverse of our locate indices
    if ParentIDSortInds is not None:
        trInds = ParentIDSortInds[trInds]

    # just the snapshot indices requested? then we can return the now
    if inds:
        return trInds

    # otherwise, the tracer IDs need to be loaded    
    trIDs = cosmo.load.snapshotSubset(sP, 'tracer', 'TracerID', inds=trInds)

    if debug:
        print(' trIDs: Size: '+str(trIDs.size)+' min: '+str(trIDs.min())+' max: '+str(trIDs.max()))
    return trIDs

def mapParentIDsToIndsByType(sP, parentIDs):
    """ For an input list of parent IDs (an unknown mixture of possibly non-unique gas, star, and BH IDs), 
        locate these cells/particles in the snapshot and return the by-type global snapshot indices 
        (one per tracer, and so then possibly containing duplicates). """

    # transform parentIDs into a unique array
    parentIDsUniq, parentIDsUniqInvInds = np.unique(parentIDs, return_inverse=True)

    if debug:
        print(' mapParentIDsToIndsByType: parentIDs has ['+str(parentIDs.size)+\
              '] unique ['+str(parentIDsUniq.size)+']')

    r = { 'partTypes'   : ['gas','stars','bhs'],
          'parentInds'  : np.zeros( parentIDs.size, dtype='int64' ),
          'parentTypes' : np.zeros( parentIDs.size, dtype='int16' ) - 1 } # start at -1

    nMatched = 0

    for ptName in r['partTypes']:
        parIDsType = cosmo.load.snapshotSubset(sP, ptName, 'id')
        if debug:
            print(' mapParentIDsToIndsByType: '+ptName+' searching ['+str(parIDsType.size)+'] ids...')

        # crossmatch the two unique ID lists and request the match indices into both
        parIndsType, wMatched = match3(parIDsType, parentIDs)

        if parIndsType is None:
            continue # parentIDs contain none of this particle type

        if debug:
            # old method, keep as debug check
            parIndsTypeUniq, wMatchedUniq = match2(parIDsType, parentIDsUniq)

            # create mask for matched elements of parentIDsUniq, transform into mask of parentIDs, then
            # make index list of matched elements of parentIDs
            wMatchedUniqMask = np.zeros( parentIDsUniq.size, dtype='int16' )
            wMatchedUniqMask[wMatchedUniq] = 1
            wMatchedMask = wMatchedUniqMask[parentIDsUniqInvInds]
            comp2 = wMatchedMask.nonzero()[0]

            # create 'mask' for parIndsType of matched elements, which contains actual indices as values, 
            # transform into value array for all parentIDs, then extract out values
            parIndsTypeUniqMask = np.zeros( parentIDsUniq.size, dtype='int64' ) - 1
            parIndsTypeUniqMask[wMatchedUniq] = parIndsTypeUniq
            parIndsTypeMask = parIndsTypeUniqMask[parentIDsUniqInvInds]
            comp1 = parIndsTypeMask[np.where(parIndsTypeMask >= 0)[0]]

            if not np.array_equal(comp1,parIndsType):
                raise Exception('match disagreement parIndsType')
            if not np.array_equal(comp2,wMatched):
                raise Exception('match disagreement wMatched')

        r['parentInds'][wMatched]  = parIndsType
        r['parentTypes'][wMatched] = partTypeNum(ptName) 

        nMatched += parIndsType.size

    # verify that we found all parents
    if r['parentTypes'].min() < 0 or nMatched != parentIDs.size:
        raise Exception('Failed to locate all requested parents through their IDs.')

    return r

def subhaloTracerChildren(sP, inds=False, haloID=None, subhaloID=None, 
                          parPartTypes=['gas','stars','bhs'], concatTypes=True,
                          ParentID=None, ParentIDSortInds=None):
    """ For a given haloID or subhaloID, return all the child tracers of parents in that object (their 
        IDs or their global indices in the snap), by default for parents of all particle types, 
        optionally restricted to input particle type(s). """
    trIDsByParType = {}

    # quick caching mechanism
    saveFilename = sP.derivPath + 'trChildren/snap_' + str(sP.snap) + '_sh_' + str(subhaloID) + \
                   '_i' + str(int(inds)) + '_' + '-'.join(parPartTypes) + '.hdf5'

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f.keys():
                trIDsByParType[key] = f[key][()]
    else:
        # calculate now
        trIDsByParType = {}

        # consider each parent type
        for parPartType in parPartTypes:
            # load IDs of this type in the requested object
            parIDsType = cosmo.load.snapshotSubset(sP, parPartType, 'id', haloID=haloID, subhaloID=subhaloID)

            if debug:
                print(' subhaloTracerChildren: '+parPartType+' '+str(len(parIDsType)))

            # crossmatch
            trIDs = getTracerChildren(sP, parIDsType, inds=inds, 
                                      ParentID=ParentID, ParentIDSortInds=ParentIDSortInds)

            # save
            trIDsByParType[parPartType] = trIDs

            if debug:
                # (A) (we use this as a check, but could instead use it as a given and then avoid this PT 
                # loop, as well as the outer loop over subhalos, with a concat->locate->split approach, since 
                # we would then know ahead of time the number of tracers in each type/subhalo)
                numTr = cosmo.load.snapshotSubset(sP, parPartType, 'numtr', haloID=haloID, subhaloID=subhaloID)
                if numTr.sum() != trIDs.size:
                    raise Exception('Tracer number count mismatch.')

                # (B): consider duplicates and consistency
                wParMoreThanOneTracer = np.where(numTr > 1)[0]
                wParZeroTracers = np.where(numTr == 0)[0]
                totalDupeTracers = int((numTr[wParMoreThanOneTracer]-1).sum())

                expectedNumChildTracers = parIDsType.size + totalDupeTracers - len(wParZeroTracers)

                if expectedNumChildTracers != trIDs.size:
                    raise Exception('Inconsistency in tracer counts.')

                # (C): verify self consistency of par -> tr -> par
                debugTracerIDs = cosmo.load.snapshotSubset(sP, 'tracer', 'TracerID')
                debugTracerInds = match(debugTracerIDs, trIDs, uniq=True)
                debugTracerParIDs = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID', inds=debugTracerInds)
                debugTracerPars = mapParentIDsToIndsByType(sP, debugTracerParIDs)
                debugwType = np.where( debugTracerPars['parentTypes'] == partTypeNum(parPartType) )[0]
                debugIndsType = debugTracerPars['parentInds'][debugwType]
                debugTracerParIDsType = cosmo.load.snapshotSubset(sP, parPartType, 'id', inds=debugIndsType)

                # in this direction, we have acquired duplicate parents and lost parents with no tracers
                debugTracerParIDsType_uniq = np.unique(debugTracerParIDsType)

                debugParIDsTypeZeroTracers = []
                if len(wParZeroTracers) > 0:
                    debugParIDsTypeZeroTracers = parIDsType[wParZeroTracers]

                debugTracerParIDsTypeAllUniq = np.concatenate((debugTracerParIDsType_uniq,debugParIDsTypeZeroTracers))

                # re-sort the two for comparison
                debugTracerParIDsTypeAllUniq_sorted = np.sort(debugTracerParIDsTypeAllUniq)
                debugParIDsType_sorted = np.sort(parIDsType)

                if not np.array_equal(debugTracerParIDsTypeAllUniq_sorted, debugParIDsType_sorted):
                    raise Exception(' sTC: Debug check on tr -> par -> tr self-consistency fail.')

        # save
        with h5py.File(saveFilename,'w') as f:
            for key in trIDsByParType.keys():
                f[key] = trIDsByParType[key]
        print('Wrote: ' + saveFilename)

    # concatenate child tracer IDs disregarding type?
    if concatTypes:
        totNumTracers = np.array([trIDsByParType[k].size for k in trIDsByParType.keys()]).sum()
        trIDsAllTypes = np.zeros( totNumTracers, dtype=trIDsByParType.values()[0].dtype )

        offset = 0

        for parPartType in parPartTypes:
            trIDsAllTypes[offset : offset + trIDsByParType[parPartType].size] = trIDsByParType[parPartType]
            offset += trIDsByParType[parPartType].size

        return trIDsAllTypes

    return trIDsByParType

def tracersTimeEvo(sP, tracerSearchIDs, toRedshift, trFields, parFields):
    """ For a given set of tracerIDs at sP.redshift, walk through snapshots either forwards or backwards 
        until reaching toRedshift. At each snapshot, re-locate the tracers and record trFields as a 
        time sequence (from fluid_properties). Then, locate their parents at each snapshot and record 
        parFields (from valid snapshot fields). For gas fields (e.g. temp), if the tracer is in a parent of 
        a different type at that snapshot, record NaN. If the tracer is in a BH, set Velocity=0 as a flag. """

    # create inverse sort indices for tracerSearchIDs
    sortIndsSearch = np.argsort(tracerSearchIDs)
    revIndsSearch = np.zeros_like( sortIndsSearch )
    revIndsSearch[sortIndsSearch] = np.arange(sortIndsSearch.size)

    # snapshot config
    startSnap = sP.snap
    finalSnap = cosmo.util.redshiftToSnapNum(toRedshift,sP)
    snapStep  = 2 if finalSnap > startSnap else -2
    print('SNAP STEPPING 2')

    snaps     = np.arange(startSnap,finalSnap+snapStep,snapStep)
    redshifts = cosmo.util.snapNumToRedshift(sP, snaps)

    if debug:
        print('tracersTimeEvo: ['+str(startSnap)+'] to ['+str(finalSnap)+'] step = '+str(snapStep))

    # allocate return struct
    r = {}
    for field in parFields+trFields:
        # hardcode some dtypes and dimensionality for now
        dtype = 'float32' if 'tracer_' not in field else 'int32'

        if field in ['pos','vel']: # [N,3] vector
            r[field] = np.zeros( (len(snaps),tracerSearchIDs.size,3), dtype=dtype )
        else: # [N] vector
            r[field] = np.zeros( (len(snaps),tracerSearchIDs.size), dtype=dtype )

    # walk through snapshots
    for m, snap in enumerate(snaps):
        sP = simParams(res=sP.res, run=sP.run, snap=snap)
        print('['+str(sP.snap).zfill(3)+'] z = '+str(sP.redshift))

        # for the tracers we search for: get their indices at this snapshot
        tracerIDsLocal  = cosmo.load.snapshotSubset(sP, 'tracer', 'TracerID')

        tracerIndsLocal,_ = match3(tracerIDsLocal, tracerSearchIDs)
        tracerIDsLocalCheck = tracerIDsLocal[tracerIndsLocal]

        if 0:
            # old method which is not order preserving
            tracerIndsLocal = match(tracerIDsLocal, tracerSearchIDs, uniq=True)
            tracerIDsLocal  = tracerIDsLocal[tracerIndsLocal]

            # shuffle tracer indices at this snapshot to be in the same order as our search order
            sortIndsLocal   = np.argsort(tracerIDsLocal)
            tracerIDsLocalCheck = tracerIDsLocal[sortIndsLocal[revIndsSearch]]
            tracerIndsLocal = tracerIndsLocal[sortIndsLocal[revIndsSearch]]

        if not np.array_equal(tracerIDsLocalCheck, tracerSearchIDs):
            raise Exception('Failure to match TracerID set between snapshots.')

        # record tracer properties
        for field in trFields:
            if debug:
                print(' '+field)

            r[field][m,:] = cosmo.load.snapshotSubset(sP, 'tracer', field, inds=tracerIndsLocal)

        # get parent IDs and then indices by-type
        tracerParIDsLocal = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID', inds=tracerIndsLocal)
        tracerParsLocal = mapParentIDsToIndsByType(sP, tracerParIDsLocal)

        if debug:
            # go full circle, calculate the tracer children of these parents, and verify
            debugTracerParIDs = np.zeros( tracerParsLocal['parentInds'].size, dtype='uint64' )
            offset = 0

            for ptName in tracerParsLocal['partTypes']:
                wType = np.where( tracerParsLocal['parentTypes'] == partTypeNum(ptName) )[0]
                indsType = tracerParsLocal['parentInds'][wType]

                if not indsType.size:
                    continue

                debugTypeIDs = cosmo.load.snapshotSubset(sP, ptName, 'id', inds=indsType)
                debugTracerParIDs[offset:offset+debugTypeIDs.size] = debugTypeIDs
                offset += debugTypeIDs.size

            debugTracerIDs = getTracerChildren(sP, np.array(debugTracerParIDs), inds=False)

            # at any snap other than startSnap, the parents may have additional child tracers in them, 
            # so at best we verify that our search group is a subset of all current children
            debugTracerIDsIndMatch,_ = match3(debugTracerIDs,tracerSearchIDs)
            #if not np.array_equal(np.sort(debugTracerIDs),np.sort(tracerSearchIDs)): # only true at startSnap
            if not np.array_equal(debugTracerIDs[debugTracerIDsIndMatch],tracerSearchIDs):
                raise Exception(' tTE: Debug check on tr -> par -> tr self-consistency fail.')

        # record parent properties
        for field in parFields:

            # load parent cells/particles by type
            for ptName in tracerParsLocal['partTypes']:
                if debug:
                    print(' '+field+' '+ptName)

                wType = np.where( tracerParsLocal['parentTypes'] == partTypeNum(ptName) )[0]
                indsType = tracerParsLocal['parentInds'][wType]

                if not indsType.size:
                    continue # no parents of this type

                # does this property exist for parents of this type? flag NaN (hardcoded for now)
                if field in ['temp','sfr'] and ptName != 'gas':
                    r[field][m,wType] = np.nan
                    continue

                # load parent property and save by dimension
                dataRead = cosmo.load.snapshotSubset(sP, ptName, field, inds=indsType)
                if dataRead.ndim == 1:
                    r[field][m,wType] = dataRead
                else:
                    r[field][m,wType,:] = dataRead

    return r, snaps, redshifts

def subhalosTracersTimeEvo(sP,subhaloIDs,toRedshift,trFields,parFields,parPartTypes,outPath):
    """ For a set of subhaloIDs, determine all their child tracers at sP.redshift then record their 
        properties back in time. """
    # load global ParentID for all tracers at the starting snapshot, and pre-sort
    #ParentID = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID')
    #ParentIDSortInds = np.argsort(ParentID, kind='mergesort')
    #ParentID = ParentID[ParentIDSortInds]
    # or: if we have already cached all the initial subhaloTracerChildren(), can skip
    ParentID = None
    ParentIDSortInds = None

    # for each subhalo, get a list of all child tracer IDs of parPartTypes
    trIDsBySubhalo = {}
    trCounts = np.zeros( subhaloIDs.size, dtype='uint32' )

    for i, subhaloID in enumerate(subhaloIDs):
        shDetails = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)

        if debug:
            print('['+str(i).zfill(3)+'] subhaloID = '+str(subhaloID) + \
                  '  LenType: '+' '.join([str(l) for l in shDetails['SubhaloLenType']]))

        subhaloTrIDs = subhaloTracerChildren(sP,subhaloID=subhaloID,parPartTypes=parPartTypes,
                                             ParentID=ParentID,ParentIDSortInds=ParentIDSortInds)

        trIDsBySubhalo[str(subhaloID)] = subhaloTrIDs
        trCounts[i] = len(subhaloTrIDs)

    # concatenate all tracer IDs into a single search list
    trSearchIDs = np.zeros( trCounts.sum(), dtype=subhaloTrIDs.dtype )
    offset = 0

    for i, subhaloID in enumerate(subhaloIDs):
        trSearchIDs[offset : offset+trCounts[i]] = trIDsBySubhalo[str(subhaloID)]
        offset += trCounts[i]

    # follow tracer and tracer parent properties over the requested snapshot range
    tracerProps, snaps, redshifts = tracersTimeEvo(sP, trSearchIDs, toRedshift, trFields, parFields)

    # separate back out by subhalo and save one data file per subhalo
    offset = 0

    for i, subhaloID in enumerate(subhaloIDs):
        outFilePath = outPath + 'subhalo_' + str(subhaloID) + '.hdf5'

        f = h5py.File(outFilePath,'w')

        # header
        f['TracerID']  = trSearchIDs[offset : offset + trCounts[i]]
        f['SubhaloID'] = [subhaloID]
        f['Snapshot']  = snaps
        f['Redshift']  = redshifts

        for key in tracerProps.keys():
            if tracerProps[key].ndim == 3:
                f[key] = tracerProps[key][:,offset : offset + trCounts[i],:]
            else:
                f[key] = tracerProps[key][:,offset : offset + trCounts[i]]

        f.close()
        print('Saved: ' + outFilePath)
        offset += trCounts[i]

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
    # config
    axis1 = 0
    axis2 = 1
    alpha = 0.05
    boxSize = 2000.0 # ckpc/h
    sP = simParams(res=1820, run='illustris', redshift=0.0)

    shNums = [int(s[:-5].rsplit('_',1)[1]) for s in glob.glob(sP.derivPath + 'subhalo_*.hdf5')]
    shNum = shNums[75]

    # load
    with h5py.File(sP.derivPath + 'subhalo_'+str(shNum)+'.hdf5') as f:
        pos  = f['pos'][()]
        temp = f['temp'][()]
        sfr  = f['sfr'][()]
        redshift = f['Redshift'][()]

        pt = cosmo.load.groupCatSingle(sP, subhaloID=f['SubhaloID'][0])['SubhaloPos']

    # plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.set_xlim(pos[:,:,axis1].mean() + np.array([-boxSize,boxSize]))
    ax.set_ylim(pos[:,:,axis2].mean() + np.array([-boxSize,boxSize]))
    ax.set_aspect(1.0)

    ax.set_title('Evolution of tracer positions with time check')
    ax.set_xlabel('x [ckpc/h]')
    ax.set_ylabel('y [ckpc/h]')

    # make relative and periodic correct
    xDist = vecs[:,0] - pt[0]
    yDist = vecs[:,1] - pt[1]
    zDist = vecs[:,2] - pt[2]

    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    for i in np.arange(pos.shape[1]): #np.arange(1000)
        ax.plot(pos[:,i,axis1], pos[:,i,axis2], '-', color='#333333', alpha=alpha, lw=1.0)

    fig.tight_layout()
    plt.savefig('trMC_checkPos_'+sP.simName+'_sh'+str(shNum)+'.pdf')
    plt.close(fig)

    # plot 2
    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)
    ax.set_xlim([0.0,0.5])
    ax.set_ylim([3.5,8.0])

    ax.set_title('Evolution of tracer temperatures with time check')
    ax.set_xlabel('Redshift')
    ax.set_ylabel('Temp [log K]')

    #for i in np.arange(temp.shape[1]):
    for i in np.arange(100):
        ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
        if not np.count_nonzero(ww[0]):
            print('skip: '+str(i))
            continue

        ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', color='#333333', alpha=alpha, lw=1.0)

    fig.tight_layout()
    plt.savefig('trMC_checkTemp_'+sP.simName+'_sh'+str(shNum)+'.pdf')
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