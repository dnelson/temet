"""
tracerMC.py
  Helper functions to efficiently work with the Monte Carlo tracers.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
import time
from os.path import isfile, isdir
from os import mkdir

import cosmo.load
from util.helper import iterable, nUnique
from cosmo.mergertree import mpbSmoothedProperties
from cosmo.util import periodicDists

debug = False # enable expensive debug consistency checks and verbose output

defParPartTypes = ['gas','stars','bhs']

def match(ar1, ar2, uniq=False):
    """ My version of numpy.in1d with invert=False. Return is a ndarray of indices into ar1, 
        corresponding to elements which exist in ar2. Meant to be used e.g. as ar1=all IDs in
        snapshot, and ar2=some IDs to search for, where ar2 could be e.g. ParentID from the 
        tracers, in which case they are generally not unique (multiple tracers can exist in the 
        same parent). """
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
    """ Returns index arrays i1,i2 of the matching elements between ar1 and ar2. While the elements of ar1 
        must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1 
        gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives 
        the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves 
        the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap, 
        ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by 
        bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~= 
        O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1. """
    if debug:
        start = time.time()
        assert np.unique(ar1).size == len(ar1)

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
    # trInds,_ = match3(ParentID,parentSearchIDs) is not appropriate here: we have possibly multiple 
    # matches inside ParentID for each element of parentSearchIDs. Instead, we can do 
    # _,trInds = match3(parentSearchIDs,ParentID) and get all the indices of ParentID which are found 
    # anywhere inside parentSearchIDs
    _,trInds = match3(parentSearchIDs, ParentID)

    if trInds is None:
        return None # no children
    
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

    r = { 'partTypes'   : defParPartTypes,
          'parentInds'  : np.zeros( parentIDs.size, dtype='int64' ),
          'parentTypes' : np.zeros( parentIDs.size, dtype='int16' ) - 1 } # start at -1

    nMatched = 0

    for ptName in r['partTypes']:
        parIDsType = cosmo.load.snapshotSubset(sP, ptName, 'id')

        # no particles of this type in the snapshot
        if isinstance(parIDsType,dict):
            if parIDsType['count'] == 0:
                continue

        if debug:
            print(' mapParentIDsToIndsByType: '+ptName+' searching ['+str(parIDsType.size)+'] ids...')

        # crossmatch the two ID lists and request the match indices into both
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
        r['parentTypes'][wMatched] = sP.ptNum(ptName) 

        nMatched += parIndsType.size

    # verify that we found all parents
    if r['parentTypes'].min() < 0 or nMatched != parentIDs.size:
        raise Exception('Failed to locate all requested parents through their IDs.')

    return r

def concatTracersByType(trIDsByParType, parPartTypes):
    """ Collapse trIDsByParType dictionary into 1D tracer ID/index list. Ordering preserved according 
    to the order of parPartTypes. """
    totNumTracers = np.array([trIDsByParType[k].size for k in trIDsByParType.keys()]).sum()
    trIDsAllTypes = np.zeros( totNumTracers, dtype=trIDsByParType.values()[0].dtype )

    offset = 0

    for parPartType in parPartTypes:
        # no particles of this type in the snapshot
        if parPartType not in trIDsByParType:
            continue

        trIDsAllTypes[offset : offset + trIDsByParType[parPartType].size] = trIDsByParType[parPartType]
        offset += trIDsByParType[parPartType].size

    return trIDsAllTypes

def subhaloTracerChildren(sP, inds=False, haloID=None, subhaloID=None, 
                          parPartTypes=defParPartTypes, concatTypes=True,
                          ParentID=None, ParentIDSortInds=None):
    """ For a given haloID or subhaloID, return all the child tracers of parents in that object (their 
        IDs or their global indices in the snap), by default for parents of all particle types, 
        optionally restricted to input particle type(s). """
    trIDsByParType = {}

    doCache = False

    # quick caching mechanism
    saveFilename = sP.derivPath + 'trChildren/snap_' + str(sP.snap) + '_sh_' + str(subhaloID) + \
                   '_i' + str(int(inds)) + '_' + '-'.join(parPartTypes) + '.hdf5'

    if isfile(saveFilename) and doCache:
        with h5py.File(saveFilename,'r') as f:
            for key in f.keys():
                trIDsByParType[key] = f[key][()]
    else:
        # consider each parent type
        for parPartType in parPartTypes:
            # load IDs of this type in the requested object
            parIDsType = cosmo.load.snapshotSubset(sP, parPartType, 'id', haloID=haloID, subhaloID=subhaloID)

            # no particles of this type in the snapshot
            if isinstance(parIDsType,dict):
                if parIDsType['count'] == 0:
                    continue

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
                debugwType = np.where( debugTracerPars['parentTypes'] == sP.ptNum(parPartType) )[0]
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
        if doCache:
            with h5py.File(saveFilename,'w') as f:
                for key in trIDsByParType.keys():
                    f[key] = trIDsByParType[key]
            print('Wrote: ' + saveFilename)

    # concatenate child tracer IDs disregarding type?
    if concatTypes:
        return concatTracersByType(trIDsByParType, parPartTypes)

    return trIDsByParType

def globalTracerChildren(sP, inds=False, halos=False, subhalos=False, parPartTypes=defParPartTypes,
                         concatTypes=True, ParentID=None, ParentIDSortInds=None):
    """ For all subhalos or halos in a snapshot, return all the child tracers of parents in that object 
        (their IDs or their global indices in the snap), by default for parents of all particle types, 
        optionally restricted to input particle type(s). """
    trIDsByParType = {}

    assert halos == True or subhalos == True # pick one

    if ParentID is None:
        assert ParentIDSortInds is None # both should be None, or both should be not None

        # load and sort the parent IDs of all tracers in this snapshot
        ParentID = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID')
        ParentIDSortInds = np.argsort(ParentID, kind='mergesort')
        ParentID = ParentID[ParentIDSortInds]

    # consider each parent type
    for parPartType in parPartTypes:
        # determine index range of particles of this type
        ptNum = sP.ptNum(parPartType)

        inds = None
        indRange = None

        if halos:
            # in FoFs (contiguous at the start of each snap)
            gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupLenType'])
            haloOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']

            nPartTotInHalosThisType = np.sum(gc['halos'][:,ptNum])
            indRange = [0,nPartTotInHalosThisType-1]

            if debug:
                offset = 0
                indsDebug = np.zeros( nPartTotInHalosThisType, dtype='int64' ) - 1

                for i in range(gc['header']['Ngroups_Total']):
                    # slice starting/ending indices for stars local to this FoF
                    i0 = haloOffsetType[i,ptNum]
                    i1 = i0 + gc['halos'][i,ptNum]

                    if i1 == i0:
                        continue # zero length of this type

                    indsDebug[offset:offset+(i1-i0)] = np.arange(i0,i1)
                    offset += (i1-i0)

                assert np.array_equal( indsDebug, np.arange(indRange[0],indRange[1]+1) )
                assert np.min(indsDebug) >= 0
                assert nUnique(indsDebug) == indsDebug.size
                assert indsDebug[0] == 0 and indsDebug[-1] == indsDebug.size-1

        if subhalos:
            # in subhalos (make non-contiguous index list)
            gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType'])
            subhaloOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

            nPartTotInSubhalosThisType = np.sum(gc['subhalos'][:,ptNum])
            inds = np.zeros( nPartTotInSubhalosThisType, dtype='int64' )
            offset = 0

            for i in range(gc['header']['Nsubgroups_Total']):
                # slice starting/ending indices for stars local to this FoF
                i0 = subhaloOffsetType[i,ptNum]
                i1 = i0 + gc['subhalos'][i,ptNum]

                if i1 == i0:
                    continue # zero length of this type
                
                inds[offset:offset+(i1-i0)] = np.arange(i0,i1)
                offset += (i1-i0)

        # load global IDs of this type
        parIDsType = cosmo.load.snapshotSubset(sP, parPartType, 'id', inds=inds, indRange=indRange)

        # no particles of this type in the snapshot
        if isinstance(parIDsType,dict) and parIDsType['count'] == 0:
            continue

        # ID crossmatch: find matching elements, where trInds gives the indices list of all child tracers 
        # in all the parIDsType parents. But, they are ordered according to ParentID, therefore also 
        # capture parInds and sort them, then rearrange trInds according to the sort of parInds
        parInds,trInds = match3(parIDsType, ParentID)
        parIndsSortInds = np.argsort(parInds, kind='mergesort')

        # dealing with a pre-sorted ParentID, so take the inverse of our locate indices
        trInds = ParentIDSortInds[trInds]

        # convert indices to IDs and save into dict
        trIDs = cosmo.load.snapshotSubset(sP, 'tracer', 'TracerID', inds=trInds)
        trIDs = trIDs[parIndsSortInds] # make order consistent with parIDsType

        trIDsByParType[parPartType] = trIDs

    # concatenate child tracer IDs disregarding type?
    if concatTypes:
        return concatTracersByType(trIDsByParType, parPartTypes)

    return trIDsByParType

def tracersTimeEvo(sP, tracerSearchIDs, trFields, parFields, toRedshift=None, snapStep=1, mpb=None):
    """ For a given set of tracerIDs at sP.redshift, walk through snapshots either forwards or backwards 
        until reaching toRedshift. At each snapshot, re-locate the tracers and record trFields as a 
        time sequence (from fluid_properties). Then, locate their parents at each snapshot and record 
        parFields (from valid snapshot fields). For gas fields (e.g. temp), if the tracer is in a parent of 
        a different type at that snapshot, record NaN. If the tracer is in a BH, set Velocity=0 as a flag. """

    # config
    gas_only_fields = ['temp','sfr','entr']   # tag with NaN for values not in gas parents at some snap
    n_3d_fields     = ['pos','vel']           # store [N,3] vector instead of [N] vector
    d_int32_fields  = ['tracer_windcounter',  # use int32 dtype to store, otherwise default to float32
                       'subhalo_id']
                       
    halo_rel_fields = {'rad'        : ['pos'], # require mpb as fields are relative to halo properties
                       'rad_rvir'   : ['pos'], # (in which case the mapping is the snapshot quantities needed)
                       'vrad'       : ['pos','vel'], 
                       'vrad_vvir'  : ['pos','vel'],
                       'angmom'     : ['pos','vel','mass'],
                       'subhalo_id' : []}

    # create inverse sort indices for tracerSearchIDs
    sortIndsSearch = np.argsort(tracerSearchIDs)
    revIndsSearch = np.zeros_like( sortIndsSearch )
    revIndsSearch[sortIndsSearch] = np.arange(sortIndsSearch.size)

    # snapshot config
    startSnap = sP.snap

    if toRedshift is not None:
        finalSnap = cosmo.util.redshiftToSnapNum(toRedshift,sP)
    else:
        finalSnap = 0

    if finalSnap > startSnap:
        snapStep = +1 * np.abs(snapStep)
    else:
        snapStep = -1 * np.abs(snapStep)

    snaps = np.arange(startSnap,finalSnap+snapStep,snapStep)
    snaps = snaps[snaps >= 0]

    # intersect with valid snapshots (skip missing, etc) with tracers (e.g. skip mini's in L75TNG)
    validSnaps = cosmo.util.validSnapList(sP, reqTr=True)
    snaps = [snap for snap in snaps if snap in validSnaps]

    redshifts = cosmo.util.snapNumToRedshift(sP, snaps)

    if debug:
        print('tracersTimeEvo: ['+str(startSnap)+'] to ['+str(finalSnap)+'] step = '+str(snapStep))

    # allocate return struct
    r = { 'snaps'     : snaps, 
          'redshifts' : redshifts }

    for field in parFields+trFields:
        # hardcode some dtypes and dimensionality for now
        dtype = 'float32'
        if field in d_int32_fields: dtype = 'int32'

        if field in n_3d_fields:
            r[field] = np.zeros( (len(snaps),tracerSearchIDs.size,3), dtype=dtype )
        else:
            r[field] = np.zeros( (len(snaps),tracerSearchIDs.size), dtype=dtype )

    # walk through snapshots
    for m, snap in enumerate(snaps):
        sP.setSnap(snap)
        print(' ['+str(sP.snap).zfill(3)+'] z = '+str(sP.redshift))

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
        if len(parFields):
            tracerParIDsLocal = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID', inds=tracerIndsLocal)
            tracerParsLocal = mapParentIDsToIndsByType(sP, tracerParIDsLocal)

            if debug:
                # go full circle, calculate the tracer children of these parents, and verify
                debugTracerParIDs = np.zeros( tracerParsLocal['parentInds'].size, dtype='uint64' )
                offset = 0

                for ptName in tracerParsLocal['partTypes']:
                    wType = np.where( tracerParsLocal['parentTypes'] == sP.ptNum(ptName) )[0]
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

            # load anything independent of particle type
            if field == 'subhalo_id':
                # sims.zooms2/h2_L9: corrupt groups_104 override
                if sP.run == 'zooms2' and sP.res == 9 and sP.hInd == 2 and sP.snap == 104:
                    print('WARNING: sims.zooms2/h2_L9: corrupt gc subhalo_id override')
                    r[field][m,:] = np.zeros( tracerSearchIDs.size, dtype='int32' ) - 1
                    continue

                gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType'])
                gcOffsets = cosmo.load.groupCatOffsetListIntoSnap(sP)

            # load parent cells/particles by type
            for ptName in tracerParsLocal['partTypes']:
                if debug:
                    print('  '+field+' '+ptName)

                wType = np.where( tracerParsLocal['parentTypes'] == sP.ptNum(ptName) )[0]
                indsType = tracerParsLocal['parentInds'][wType]

                if not indsType.size:
                    continue # no parents of this type

                # does this property exist for parents of this type? flag NaN if not
                if field in gas_only_fields and ptName != 'gas':
                    r[field][m,wType] = np.nan
                    continue

                if field not in halo_rel_fields:
                    # load parent property
                    data = cosmo.load.snapshotSubset(sP, ptName, field, inds=indsType)

                    # save directly (by dimension) if not calculating further
                    if data.ndim == 1:
                        r[field][m,wType] = data
                    else:
                        r[field][m,wType,:] = data

                    continue

                # field is relative to halo properties? extract halo values at this snapshot
                if mpb is None:
                    raise Exception('Error, mpb track required as inputs.')

                mpbInd = np.where( mpb['SnapNum'] == snap )[0]
                if len(mpbInd) == 0:
                    raise Exception('Error, snap ['+str(snap)+'] not found in mpb.')

                haloCenter = mpb['sm']['pos'][mpbInd[0],:]
                haloVel    = mpb['sm']['vel'][mpbInd[0],:]
                haloVirRad = mpb['sm']['rvir'][mpbInd[0]]
                haloVirVel = mpb['sm']['vvir'][mpbInd[0]]

                # load required raw fields(s) from the snapshot
                data = {}
                for fName in halo_rel_fields[field]:
                    data[fName] = cosmo.load.snapshotSubset(sP, ptName, fName, inds=indsType)

                # compute
                if field in ['rad','rad_rvir']:
                    # radial distance from halo center, optionally normalized by rvir (r200crit)
                    val = periodicDists(haloCenter, data['pos'], sP) # code units (e.g. ckpc/h)
                    
                    if field == 'rad_rvir':
                        val /= haloVirRad # normalized, unitless

                if field in ['vrad','vrad_vvir']:
                    # radial velocity relative to halo CM motion [km/s], hubble expansion added in, 
                    # optionally normalized by the vvir (v200) of the halo at this snapshot
                    val = sP.units.particleRadialVelInKmS(data['pos'], data['vel'], haloCenter, haloVel)

                    if field == 'vrad_vvir':
                        val /= haloVirVel # normalized, unitless

                if field in ['angmom']:
                    # magnitude of specific angular momentum in [kpc km/s]
                    val = sP.units.particleSpecAngMomMagInKpcKmS(data['pos'], data['vel'], data['mass'], 
                                                                 haloCenter, haloVel, log=True)

                if field in ['subhalo_id']:
                    # determine parent subhalo ID                    
                    gcLenType = gc['subhalos'][:,sP.ptNum(ptName)]
                    gcOffsetsType = gcOffsets['snapOffsetsSubhalo'][:,sP.ptNum(ptName)][:-1]

                    # val gives the indices of gcOffsetsType such that, if each indsType was inserted 
                    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
                    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the 
                    # subhalo offset (i.e. first particle) works correctly
                    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the 
                    # subhalo offset index -after- which the particle should be inserted
                    val = np.searchsorted( gcOffsetsType - 1, indsType ) - 1
                    val = val.astype('int32')

                    # search and flag all tracers with parents whose indices exceed the length of the 
                    # subhalo they have been assigned to, e.g. either in fof fuzz, in subhalos with 
                    # no particles of this type, or not in any subhalo at the end of the file
                    gcOffsetsMax = gcOffsetsType + gcLenType - 1
                    ww = np.where( indsType > gcOffsetsMax[val] )[0]

                    if len(ww):
                        val[ww] = -1

                    if debug:
                        # for tracers we identified in subhalos, verify parents directly
                        for i in range(indsType.size):
                            if val[i] < 0:
                                continue
                            assert indsType[i] >= gcOffsetsType[val[i]]
                            assert indsType[i] < gcOffsetsType[val[i]]+gcLenType[val[i]]
                            assert gcLenType[val[i]] != 0

                        # for tracers we identified in no subhalos, load particle IDs of this type 
                        # from all subhalos at this snapshot, then tracerParIDsLocal[wType[noPar]] matched 
                        # to allIDsOfThisTypeInSubhalos should be empty
                        wwNoPar = np.where( val < 0 )[0]

                        if len(wwNoPar) > 0:
                            ParIDsToMatch = np.zeros( gcLenType.sum(), dtype=tracerParIDsLocal.dtype )
                            offset = 0

                            for i in range(gcLenType.size):
                                if gcLenType[i] == 0:
                                    continue

                                loadIDs = cosmo.load.snapshotSubset(sP, ptName, 'ids', subhaloID=i)
                                ParIDsToMatch[ offset : offset+gcLenType[i] ] = loadIDs
                                offset += gcLenType[i]

                            tracerParIDsShouldBeOutsideAllSubhalos = tracerParIDsLocal[wType[wwNoPar]]
                            ind1, ind2 = match3(ParIDsToMatch, tracerParIDsShouldBeOutsideAllSubhalos)

                            assert ind1 is None and ind2 is None

                if val.ndim != 1:
                    raise Exception('Unexpected.')

                r[field][m,wType] = val

    sP.setSnap(startSnap)
    return r

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
        if debug:
            shDetails = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
            print('['+str(i).zfill(3)+'] subhaloID = '+str(subhaloID) + \
                  '  LenType: '+' '.join([str(l) for l in shDetails['SubhaloLenType']]))

        subhaloTrIDs = subhaloTracerChildren(sP,subhaloID=subhaloID,parPartTypes=parPartTypes,
                                             ParentID=ParentID,ParentIDSortInds=ParentIDSortInds)

        trIDsBySubhalo[subhaloID] = subhaloTrIDs
        trCounts[i] = len(subhaloTrIDs)

    # concatenate all tracer IDs into a single search list
    trSearchIDs = np.zeros( trCounts.sum(), dtype=subhaloTrIDs.dtype )
    offset = 0

    for i, subhaloID in enumerate(subhaloIDs):
        trSearchIDs[offset : offset+trCounts[i]] = trIDsBySubhalo[subhaloID]
        offset += trCounts[i]

    # follow tracer and tracer parent properties over the requested snapshot range
    tracerProps = tracersTimeEvo(sP, trSearchIDs, trFields, parFields, toRedshift)

    # separate back out by subhalo and save one data file per subhalo
    offset = 0

    for i, subhaloID in enumerate(subhaloIDs):
        outFilePath = outPath + 'subhalo_' + str(subhaloID) + '.hdf5'

        f = h5py.File(outFilePath,'w')

        # header
        f['TracerIDs'] = trSearchIDs[offset : offset + trCounts[i]]
        f['SubhaloID'] = [subhaloID]

        for key in tracerProps.keys():
            if key in ['snaps','redshifts']:
                f[key] = tracerProps[key]
                continue

            if tracerProps[key].ndim == 3:
                f[key] = tracerProps[key][:,offset : offset + trCounts[i],:]
            else:
                f[key] = tracerProps[key][:,offset : offset + trCounts[i]]

        f.close()
        print('Saved: ' + outFilePath)
        offset += trCounts[i]

def subhaloTracersTimeEvo(sP, subhaloID, fields, snapStep=1, toRedshift=10.0, fullHaloTracers=True):
    """ For a single subhaloID, determine all its child tracers at sP.redshift and then record 
    their properties back in time until the beginning of the simulation ('tracks'). 
    Note: Nearly identical to subhaloTracersTimeEvo() and can be merged in the future. Currently, 
    here we do a separate snapshot loop for each quantity and save each quantity in a separate file. 
    Memory load is smaller, run time is longer, parallelized to multiple quantities easily, and only 
    one subhaloID supported at a time. """

    if not isdir(sP.derivPath + '/trTimeEvo'):
        mkdir(sP.derivPath + '/trTimeEvo')
    
    def saveFilename():
        """ Output file name (all vars are taken from locals at the time of call). """
        snapFinal = cosmo.util.redshiftToSnapNum(toRedshift, sP)
        return sP.derivPath + '/trTimeEvo/shID_%d_hf%d_snap_%d-%d-%d_%s.hdf5' % \
          (subhaloID,fullHaloTracers,sP.snap,snapFinal,snapStep,field)

    # single load requested? do now and return
    fields = iterable(fields)

    if len(fields) == 1:
        field = fields[0]

        if isfile(saveFilename()):
            with h5py.File(saveFilename(),'r') as f:
                r = {}
                for key in f:
                    r[key] = f[key][()]

            return r

    if fullHaloTracers:
        # get child tracers of all particle types in entire parent halo
        gc = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
        haloID = gc['SubhaloGrNr']

        trIDs = subhaloTracerChildren(sP, haloID=haloID)
    else:
        # get child tracers of all particle types in subhalo only
        trIDs = subhaloTracerChildren(sP, subhaloID=subhaloID)

    # get smoothed MPB tracks since some properties are relative to the parent halo MPB values
    mpb = mpbSmoothedProperties(sP, subhaloID)

    # follow tracer and tracer parent properties (one at a time and save) from sP.snap back to snap=0
    # note, could simply eliminate this whole loop and hand all of trFields+parFields to tracersTimeEvo()
    #   in one call, then save identical files as now by splitting up the vals return. only keep it as is 
    #   to be embarassingly parallel and let different jobs handle different quantities
    for field in fields:
        # check for existence of save
        if isfile(saveFilename()):
            print('Already exists, skipping computation: [%s]' % saveFilename().split(sP.derivPath)[1])
        else:           
            print('Computing: [%s]' % saveFilename().split(sP.derivPath)[1])

            if 'tracer_' in field: # trField
                trVals = tracersTimeEvo(sP, trIDs, [field], [], toRedshift, snapStep, mpb)
            else: # parField
                trVals = tracersTimeEvo(sP, trIDs, [], [field], toRedshift, snapStep, mpb)

            # save
            with h5py.File(saveFilename(),'w') as f:
                # header
                trVals['TracerIDs'] = trIDs
                trVals['SubhaloID'] = [subhaloID]

                for key in trVals:
                    f[key] = trVals[key]

            print('Saved: [%s]' % saveFilename().split(sP.derivPath)[1])

        if len(fields) == 1:
            return trVals

def globalTracerLength(sP, halos=False, subhalos=False):
    """ Return a 1D array of the number of tracers per halo or subhalo, for all in the snapshot, in 
    direct analogy to LenType in the group catalogs. Compute the offsets as well. """
    assert halos is True or subhalos is True # pick one

    # load and sort the parent IDs of all tracers in this snapshot
    ParentID = cosmo.load.snapshotSubset(sP, 'tracer', 'ParentID')
    ParentIDSortInds = np.argsort(ParentID, kind='mergesort')
    ParentID = ParentID[ParentIDSortInds]

    # allocate counts
    h = cosmo.load.groupCatHeader(sP)
    if halos: nObjs = h['Ngroups_Total']
    if subhalos: nObjs = h['Nsubgroups_Total']

    trCounts  = {}
    trOffsets = {}

    for pt in defParPartTypes:
        trCounts[pt]  = np.zeros( nObjs, dtype='int32' )
        trOffsets[pt] = np.zeros( nObjs, dtype='int64' )

    # load offsets
    if halos:
        gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupLenType'])['halos']
        objOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']

    if subhalos:
        gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType'])['subhalos']
        objOffsetType = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    # consider each parent type
    for parPartType in defParPartTypes:
        # load IDs of this type in the requested object
        print(parPartType)

        ptNum = sP.ptNum(parPartType)
        parIDsType = cosmo.load.snapshotSubset(sP, parPartType, 'id')

        # no particles of this type in the snapshot
        if isinstance(parIDsType,dict) and parIDsType['count'] == 0:
            continue

        # loop over each halo/subhalo, calculate lengths
        for i in np.arange(nObjs):
            if i % int(nObjs/10) == 0 and i <= nObjs:
                print(' %4.1f%%' % (float(i+1)*100.0/nObjs))

            # get parent IDs of this type local to this halo/subhalo
            i0 = objOffsetType[i,ptNum]
            i1 = i0 + gc[i,ptNum]

            if i1 == i0:
                continue # zero length of this type
            
            parIDsTypeLocal = parIDsType[i0:i1]

            # crossmatch
            trIDs = getTracerChildren(sP, parIDsTypeLocal, inds=True, 
                                      ParentID=ParentID, ParentIDSortInds=ParentIDSortInds)

            # save counts
            if trIDs is not None:
                trCounts[parPartType][i] = len(trIDs)

    # build offsets
    trOffsets['gas'][1:]   = np.cumsum( trCounts['gas'] )[:-1] # gas are first, obj by obj
    trOffsets['stars'][1:] = np.cumsum( trCounts['stars'] )[:-1] # stars/pt4 are next
    trOffsets['bhs'][1:]   = np.cumsum( trCounts['bhs'] )[:-1] # bhs are last
    trOffsets['stars'] += np.sum( trCounts['gas'] )
    trOffsets['bhs']   += np.sum( trCounts['gas'] ) + np.sum( trCounts['stars'] )

    return trCounts, trOffsets

def globalAllTracersTimeEvo(sP, field, halos=False, subhalos=False):
    """ For all tracers in all FoFs at the simulation endtime, record time evolution tracks of one 
    field for all snapshots (which contain tracers). """
    assert halos is True or subhalos is True # pick one

    if not isdir(sP.derivPath + '/trTimeEvo'):
        mkdir(sP.derivPath + '/trTimeEvo')

    if halos: selectStr = 'groups'
    if subhalos: selectStr = 'subhalos'

    saveFilename = sP.derivPath + '/trTimeEvo/tr_all_%s_%d_%s.hdf5' % (selectStr,sP.snap,field)

    # single load requested? do now and return
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            r = {}
            for key in f:
                r[key] = f[key][()]

        return r

    # get child tracers of all particle types in all FoFs
    trIDs = globalTracerChildren(sP, halos=halos, subhalos=subhalos)

    # mpb?
    # todo!

    # follow tracer and tracer parent properties (one at a time and save) from sP.snap back to snap=0
    print('Computing: [%s]' % saveFilename.split(sP.derivPath)[1])

    if field == 'meta':
        # save the metadata (ordered tracer IDs, lengths/offsets of tracers by group/subhalo)
        trVals = {}

        # save the tracer IDs in the order we are saving the tracks
        trVals['TracerIDs'] = trIDs

        # compute lengths/offsets
        trCounts, trOffsets = globalTracerLength(sP, halos=halos, subhalos=subhalos)

        for key in trCounts:
            trVals['TracerLength_'+key] = trCounts[key]
            trVals['TracerOffset_'+key] = trOffsets[key]

    else:
        if 'tracer_' in field: # trField
            trVals = tracersTimeEvo(sP, trIDs, [field], [])
        else: # parField
            trVals = tracersTimeEvo(sP, trIDs, [], [field])

    # save tracks
    with h5py.File(saveFilename,'w') as f:
        for key in trVals:
            f[key] = trVals[key]

    print('Saved: [%s]' % saveFilename.split(sP.derivPath)[1])

    return trVals
