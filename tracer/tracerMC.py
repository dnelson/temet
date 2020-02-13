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

from util.helper import iterable, nUnique, bincount, reportMemory
from cosmo.mergertree import mpbSmoothedProperties, loadTreeFieldnames
from cosmo.util import inverseMapPartIndicesToSubhaloIDs, inverseMapPartIndicesToHaloIDs

debug = False # enable expensive debug consistency checks and verbose output

# global configuration for tracer calculations
defParPartTypes = ['gas','stars','bhs']   # all possible tracer parent types (ordering important)

# helper information for recording different parent properties
gas_only_fields  = ['temp','sfr','entr','hdens','beta','netcoolrate','tcool'] # tag with NaN for values not in gas parents at that snap
star_only_fields = ['sftime']                   # same but for stars
n_3d_fields      = ['pos','vel']                # store [N,3] vector instead of [N] vector
d_int_fields     = {'subhalo_id':'int32',       # use int dtype to store, otherwise default to float32
                    'halo_id':'int32',
                    'tracer_windcounter':'int16',
                    'parent_indextype':'int64'}
fields_in_log    = ['temp','entr','angmom']

# require MPB(s) as fields are relative to halo properties (mapping gives the snapshot quantities needed)
halo_rel_fields = {'rad'        : ['pos'],
                   'rad_rvir'   : ['pos'],
                   'vrad'       : ['pos','vel'], 
                   'vrad_vvir'  : ['pos','vel'],
                   'angmom'     : ['pos','vel','mass']}

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
    if not isinstance(ar1,np.ndarray): ar1 = np.array(ar1)
    if not isinstance(ar2,np.ndarray): ar2 = np.array(ar2)
    assert ar1.ndim == ar2.ndim == 1
    
    if debug:
        start = time.time()
        assert np.unique(ar1).size == len(ar1)

    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)
        ar1_sorted = None
        ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
        ar1_sorted_index = None
        index = None
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

from numba import jit
@jit(nopython=True, nogil=True, cache=True)
def _match3(ar1, ar2, firstSorted=False):
    """ Test. """
    assert ar1.ndim == ar2.ndim == 1
    
    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)

        for i in range(ar1_sorted_index.size): # mode="clip"
            if ar1_sorted_index[i] >= index.size:
                ar1_sorted_index[i] = index.size

        ar1_inds = np.take(index, ar1_sorted_index)
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)

        for i in range(ar1_sorted_index.size): # mode="clip"
            if ar1_sorted_index[i] >= index.size:
                ar1_sorted_index[i] = index.size

        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index)

    mask = (ar1[ar1_inds] == ar2)
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None,None

    return ar1_inds, ar2_inds

def getTracerChildren(sP, parentSearchIDs, inds=False, ParentID=None, ParentIDSortInds=None):
    """ For an input list of parent IDs (a UNIQUE list of an unknown mixture of gas/star/BH IDs), return 
        the complete list of child MC tracers belonging to those parents (either their IDs or their 
        global indices in the snap). """
    # if global ParentID for the snapshot is not previously loaded and passed in, load it now
    if ParentID is None:
        ParentID = sP.snapshotSubsetP('tracer', 'ParentID')

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
    trIDs = sP.snapshotSubsetP('tracer', 'TracerID', inds=trInds)

    if debug:
        print(' trIDs: Size: '+str(trIDs.size)+' min: '+str(trIDs.min())+' max: '+str(trIDs.max()))
    return trIDs

def mapParentIDsToIndsByType(sP, parentIDs):
    """ For an input list of parent IDs (an unknown mixture of possibly non-unique gas, star, and BH IDs), 
        locate these cells/particles in the snapshot and return the by-type global snapshot indices 
        (one per tracer, and so then possibly containing duplicates). """

    if debug:
        # transform parentIDs into a unique array (old method)
        parentIDsUniq, parentIDsUniqInvInds = np.unique(parentIDs, return_inverse=True)

        print(' mapParentIDsToIndsByType: parentIDs has ['+str(parentIDs.size)+\
              '] unique ['+str(parentIDsUniq.size)+']')

    r = { 'partTypes'   : defParPartTypes,
          'parentInds'  : np.zeros( parentIDs.size, dtype='int64' ),
          'parentTypes' : np.zeros( parentIDs.size, dtype='int16' ) - 1 } # start at -1

    nMatched = 0

    for ptName in r['partTypes'][::-1]:
        parIDsType = sP.snapshotSubsetP(ptName, 'id')

        # no particles of this type in the snapshot
        if isinstance(parIDsType,dict) and parIDsType['count'] == 0:
            continue

        # crossmatch the two ID lists and request the match indices into both
        parIndsType, wMatched = match3(parIDsType, parentIDs)

        if parIndsType is None:
            continue # parentIDs contain none of this particle type

        if debug:
            print(' mapParentIDsToIndsByType: %s searching [%d] ids (%d unique), [%s] matches.' % \
              (ptName,parIDsType.size,nUnique(parIDsType),wMatched.size))
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
    if r['parentTypes'].min() < 0 or nMatched < parentIDs.size:
        raise Exception('Failed to locate all requested parents through their IDs.')
    if nMatched != parentIDs.size and not sP.winds:
        raise Exception('We have located more than 1 parent per search ID, inconsistency.')
    if nMatched != parentIDs.size and sP.winds:
        assert nMatched > parentIDs.size
        nOver = nMatched - parentIDs.size
        # we do not know here actually which parent these tracers belong to, because spawned winds 
        # in some bugged GFM commits do not change their ID from the progenitor gas cell. the ordering of 
        # defParPartTypes will then assign the reversed(last)==the first (e.g. the prog gas cell) as the parent
        print(' WARNING: More than 1 parent located for [%d] tracers (GFM.wind ID == prog cell).' % nOver)

    return r

def concatTracersByType(trIDsByParType, parPartTypes):
    """ Collapse trIDsByParType dictionary into 1D tracer ID/index list. Ordering preserved according 
    to the order of parPartTypes. """
    totNumTracers = np.array([trIDsByParType[k].size for k in trIDsByParType.keys()]).sum()
    trIDsAllTypes = np.zeros( totNumTracers, dtype=list(trIDsByParType.values())[0].dtype )

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

    doCache = True

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
            parIDsType = sP.snapshotSubset(parPartType, 'id', haloID=haloID, subhaloID=subhaloID)

            # no particles of this type in the snapshot
            if isinstance(parIDsType,dict) and parIDsType['count'] == 0:
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
                if sP.snapHasField('tracer', 'NumTracers'):
                    numTr = sP.snapshotSubset(parPartType, 'numtr', haloID=haloID, subhaloID=subhaloID)
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
                debugTracerIDs = sP.snapshotSubsetP('tracer', 'TracerID')
                debugTracerInds = match(debugTracerIDs, trIDs, uniq=True)
                debugTracerParIDs = sP.snapshotSubset('tracer', 'ParentID', inds=debugTracerInds)
                debugTracerPars = mapParentIDsToIndsByType(sP, debugTracerParIDs)
                debugwType = np.where( debugTracerPars['parentTypes'] == sP.ptNum(parPartType) )[0]
                debugIndsType = debugTracerPars['parentInds'][debugwType]
                debugTracerParIDsType = sP.snapshotSubset(parPartType, 'id', inds=debugIndsType)

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
    """ For all subhalos or halos in a snapshot, return all the child tracers of parents in those objects 
        (their IDs or their global indices in the snap), by default for parents of all particle types, 
        optionally restricted to input particle type(s). """
    trIDsByParType = {}

    assert halos == True or subhalos == True or sP.isSubbox # pick one

    # check cache
    hStr = 'halos' if halos else 'subhalos'
    if sP.isSubbox: hStr = 'subbox%d' % sP.subbox
    saveFilename = sP.derivPath + 'tracer_tracks/globalTracerChildren_%s_%d.hdf5' % (hStr,sP.snap)
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                trIDsByParType[key] = f[key][()]
        print('Loaded cached [%s]' % saveFilename)

        if concatTypes:
            return concatTracersByType(trIDsByParType, parPartTypes)

        return trIDsByParType

    if ParentID is None:
        assert ParentIDSortInds is None # both should be None, or both should be not None

        # load and sort the parent IDs of all tracers in this snapshot
        ParentID = sP.snapshotSubsetP('tracer', 'ParentID')
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
            gc = sP.groupCat(fieldsHalos=['GroupLenType'])
            haloOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']

            nPartTotThisType = np.sum(gc[:,ptNum])
            indRange = [0,int(nPartTotThisType-1)]

            if debug:
                offset = 0
                indsDebug = np.zeros( nPartTotThisType, dtype='int64' ) - 1

                for i in range(sP.numHalos):
                    # slice starting/ending indices for stars local to this FoF
                    i0 = haloOffsetType[i,ptNum]
                    i1 = i0 + gc[i,ptNum]

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
            gc = sP.groupCat(fieldsSubhalos=['SubhaloLenType'])
            subhaloOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']

            nPartTotThisType = np.sum(gc[:,ptNum])
            inds = np.zeros( nPartTotThisType, dtype='int64' )
            offset = 0

            for i in range(sP.numSubhalos):
                # slice starting/ending indices for stars local to this FoF
                i0 = subhaloOffsetType[i,ptNum]
                i1 = i0 + gc[i,ptNum]

                if i1 == i0:
                    continue # zero length of this type
                
                inds[offset:offset+(i1-i0)] = np.arange(i0,i1)
                offset += (i1-i0)

        if sP.isSubbox:
            # in entire subbox snapshot (no group catalogs)
            nPartTotThisType = sP.numPart[ptNum]
            indRange = [0, nPartTotThisType-1]

        # load global IDs of this type
        if nPartTotThisType == 0:
            continue
           
        parIDsType = sP.snapshotSubsetP(parPartType, 'id', inds=inds, indRange=indRange)

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
        trIDs = sP.snapshotSubsetP('tracer', 'TracerID', inds=trInds)
        trIDs = trIDs[parIndsSortInds] # make order consistent with parIDsType

        trIDsByParType[parPartType] = trIDs

    # save cache?
    if 0:
        with h5py.File(saveFilename,'w') as f:
            for key in trIDsByParType:
                f[key] = trIDsByParType[key]
        print('Saved: [%s]' % saveFilename)

    # concatenate child tracer IDs disregarding type?
    if concatTypes:
        return concatTracersByType(trIDsByParType, parPartTypes)

    return trIDsByParType

def getEvoSnapList(sP, toRedshift=None, snapStep=1):
    """ Helper for below. """
    startSnap = sP.snap

    if toRedshift is not None:
        finalSnap = sP.redshiftToSnapNum(toRedshift)
    else:
        finalSnap = 0

    if finalSnap > startSnap:
        snapStep = +1 * np.abs(snapStep)
    else:
        snapStep = -1 * np.abs(snapStep)

    snaps = np.arange(startSnap,finalSnap+snapStep,snapStep)
    snaps = snaps[snaps >= 0]

    # intersect with valid snapshots (skip missing, etc) with tracers (e.g. skip mini's in L75TNG)
    validSnaps = sP.validSnapList(reqTr=(True if not sP.isSubbox else False))
    snaps = [snap for snap in snaps if snap in validSnaps]

    redshifts = sP.snapNumToRedshift(snaps)

    if debug:
        print('tracersTimeEvo: ['+str(startSnap)+'] to ['+str(finalSnap)+'] step = '+str(snapStep))

    return snaps, redshifts

def tracersTimeEvo(sP, tracerSearchIDs, trFields, parFields, toRedshift=None, snapStep=1, mpb=None, 
                   saveFilename=None, onlySnap=None, exitAfterOneSnap=False):
    """ For a given set of tracerIDs at sP.redshift, walk through snapshots either forwards or backwards 
        until reaching toRedshift. At each snapshot, re-locate the tracers and record trFields as a 
        time sequence (from fluid_properties). Then, locate their parents at each snapshot and record 
        parFields (from valid snapshot fields). For gas fields (e.g. temp), if the tracer is in a parent of 
        a different type at that snapshot, record NaN. If the tracer is in a BH, set Velocity=0 as a flag. 
        If saveFilename specified, then we save inside this function one snapshot at a time (avoid large 
        memory allocation, and can be restarted). Otherwise, return for external save. If exitAfterOneSnap, 
        then process only one snapshot and return, mostly for memory efficiency. """

    # snapshot config
    snaps, redshifts = getEvoSnapList(sP, toRedshift, snapStep)
    startSnap = sP.snap

    sP_start = sP.copy()

    # allocate return struct
    r = { 'snaps'     : snaps, 
          'redshifts' : redshifts }

    # prepare save file (actually saved one snap at a time within tracersTimeEvo)
    if saveFilename is not None:
        if not isfile(saveFilename):
            done = np.zeros( len(snaps), dtype='int32' )
            with h5py.File(saveFilename,'w') as f:
                for key in r:
                    f[key] = r[key]
                f['done'] = done
            print('Started new save file: [%s]' % saveFilename.split("/")[-1], flush=True)
        else:
            with h5py.File(saveFilename,'r') as f:
                done = f['done'][()]

            if done.sum() == len(snaps):
                print('Done, now loading from [%s]...' % saveFilename.split("/")[-1])
                with h5py.File(saveFilename,'r') as f:
                    for key in f:
                        if key == 'done': continue
                        r[key] = f[key][()]
                return r
            
            print('Restarting: [%s] [numDone = %d]' % (saveFilename.split("/")[-1],done.sum()), flush=True)

    for field in parFields+trFields:
        # hardcode some dtypes and dimensionality for now
        dtype = 'float32'
        if field in d_int_fields.keys(): dtype = d_int_fields[field]

        # memory allocation
        numAllocSnaps = len(snaps) if saveFilename is None else 1 # 1 vs global memory allocation

        if field in n_3d_fields:
            r[field] = np.zeros( (numAllocSnaps,tracerSearchIDs.size,3), dtype=dtype )
        else:
            r[field] = np.zeros( (numAllocSnaps,tracerSearchIDs.size), dtype=dtype )

        # file pre-allocation? if we are saving herein, and not restarting
        shape = (len(snaps),tracerSearchIDs.size,3) if field in n_3d_fields else (len(snaps),tracerSearchIDs.size)

        if saveFilename is not None and done.sum() == 0:
            with h5py.File(saveFilename,'r+') as f:
                if field not in f:
                    print('Allocating field to: [%s]' % saveFilename.split("/")[-1])
                    dset = f.create_dataset(field, shape, dtype=dtype)

    # walk through snapshots
    for m, snap in enumerate(snaps):

        if onlySnap is not None:
            if snap != onlySnap:
                continue

        sP.setSnap(snap)
        print(' ['+str(sP.snap).zfill(3)+'] z = '+str(sP.redshift), flush=True)

        if saveFilename is not None:
            if done[m]:
                print('  Skip, already in save file.')
                continue
            saveSnapInd = m
            m = 0 # write over previous snapshot in r[], and save in this loop

        # try to load pre-existing parent indexes
        parent_indextype = None
        if len(trFields) == 0:
            parent_indextype = globalAllTracersTimeEvo(sP_start, 'parent_indextype', indRange=[saveSnapInd])

        # for the tracers we search for: get their indices at this snapshot
        if parent_indextype is None:
            tracerIDsLocal  = sP.snapshotSubsetP('tracer', 'TracerID')

            tracerIndsLocal, tracerSearchIndsLocal = match3(tracerIDsLocal, tracerSearchIDs)
            tracerIDsLocalCheck = tracerIDsLocal[tracerIndsLocal]

            if not np.array_equal(tracerIDsLocalCheck, tracerSearchIDs) and not sP.isSubbox:
                raise Exception('Failure to match TracerID set between snapshots.')

            if sP.isSubbox:
                frac = tracerIndsLocal.size / tracerSearchIDs.size * 100
                print('  subbox: %10d of %10d original tracers inside, %.3f%%' % (tracerIndsLocal.size,tracerSearchIDs.size,frac))
            else:
                # only need this for subboxes where we may not find all tracers searched for
                # in this case, must write to r[] in the correct locations (normally, since 
                # match3 is order preserving, tracerIndsLocal is in the order of tracerSearchIDs)
                tracerSearchIndsLocal = None

            tracerIDsLocal = None
            tracerIDsLocalCheck = None

        # record tracer properties
        for field in trFields:
            print(' '+field, flush=True)

            if sP.isSubbox:
                r[field][m,tracerSearchIndsLocal] = sP.snapshotSubsetP('tracer', field, inds=tracerIndsLocal)
            else:
                r[field][m,:] = sP.snapshotSubsetP('tracer', field, inds=tracerIndsLocal)

        # get parent IDs and then indices by-type
        if len(parFields):
            if parent_indextype is None:
                # cross-match to identify parents
                tracerParIDsLocal = sP.snapshotSubsetP('tracer', 'ParentID', inds=tracerIndsLocal)
                tracerIndsLocal = None
                tracerParsLocal = mapParentIDsToIndsByType(sP, tracerParIDsLocal)
                tracerParIDsLocal = None
            else:
                # already have saved parent indices, reconstruct by type
                assert parent_indextype['snaps'][saveSnapInd] == snap
                parent_indextype = parent_indextype['parent_indextype']

                tracerParsLocal = { 'partTypes'   : defParPartTypes,
                                    'parentInds'  : np.zeros( parent_indextype.size, dtype='int64' ),
                                    'parentTypes' : np.zeros( parent_indextype.size, dtype='int16' ) - 1 } # start at -1

                for parPt in defParPartTypes:
                    ptNum = sP.ptNum(parPt)
                    startVal = int(ptNum * 1e11)
                    endVal = int((ptNum+1) * 1e11)
                    w = np.where( (parent_indextype >= startVal) & (parent_indextype < endVal) )

                    if len(w[0]) == 0:
                        continue

                    tracerParsLocal['parentInds'][w] = parent_indextype[w[0]] - startVal
                    tracerParsLocal['parentTypes'][w] = ptNum

            if debug:
                # go full circle, calculate the tracer children of these parents, and verify
                debugTracerParIDs = np.zeros( tracerParsLocal['parentInds'].size, dtype='uint64' )
                offset = 0

                for ptName in tracerParsLocal['partTypes']:
                    wType = np.where( tracerParsLocal['parentTypes'] == sP.ptNum(ptName) )[0]
                    indsType = tracerParsLocal['parentInds'][wType]

                    if not indsType.size:
                        continue

                    debugTypeIDs = sP.snapshotSubset(ptName, 'id', inds=indsType)
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
                SubhaloLenType = sP.groupCat(fieldsSubhalos=['SubhaloLenType'])
                SnapOffsetsSubhalo = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']

            if field == 'halo_id':
                GroupLenType = sP.groupCat(fieldsHalos=['GroupLenType'])
                SnapOffsetsGroup = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup'] 

            if field in halo_rel_fields and not sP.isZoom:
                # for global catalogs (tracers spanning many/all halos) we have pre-computed the 
                # subhalo ID tracks of the MPB of each halo at the final snapshot, here load the 
                # group catalog halo properties at this snapshot and later use the mapping to 
                # assign a halo center, vel, virrad, and virvel for each tracer
                fieldsSH = ['SubhaloPos','SubhaloVel','SubhaloGrNr']
                fieldsH = ['Group_R_Crit200','Group_M_Crit200']

                gcHalo = sP.groupCat(fieldsSubhalos=fieldsSH, fieldsHalos=fieldsH)

                if 'vel' in halo_rel_fields[field]:
                    ac = sP.auxCat('Subhalo_StellarMeanVel')
                    Subhalo_StellarMeanVel = sP.units.particleCodeVelocityToKms(ac['Subhalo_StellarMeanVel'])

            # load parent cells/particles by type
            for ptName in tracerParsLocal['partTypes'][::-1]:
                print('  '+field+' '+ptName, reportMemory(), flush=True)

                wType = np.where( tracerParsLocal['parentTypes'] == sP.ptNum(ptName) )[0]
                indsType = tracerParsLocal['parentInds'][wType]

                if sP.isSubbox:
                    # update stamp indices: handle non-full search returns
                    wType = tracerSearchIndsLocal[wType]

                if exitAfterOneSnap:
                    # memory optimization, do gas last (largest), and erase things we don't need anymore
                    if ptName == tracerParsLocal['partTypes'][0]:
                        tracerParsLocal = None

                if not indsType.size:
                    continue # no parents of this type

                # does this property exist for parents of this type? flag NaN if not
                if field in gas_only_fields and ptName != 'gas':
                    r[field][m,wType] = np.nan
                    continue

                if field in star_only_fields and ptName != 'stars':
                    r[field][m,wType] = np.nan
                    continue

                if field in ['metal'] and ptName == 'bhs':
                    r[field][m,wType] = np.nan
                    continue

                # specialized properties
                if field in ['parent_indextype']:
                    # encode the snapshot index of the parent (by type) as well as its type in an int64
                    # as parent_indextype = type*1e11 + index
                    r[field][m,wType] = data = sP.ptNum(ptName)*1e11 + indsType
                    continue

                if field in ['subhalo_id']:
                    # determine parent subhalo ID
                    r[field][m,wType] = inverseMapPartIndicesToSubhaloIDs(sP, indsType, ptName, 
                                          SubhaloLenType=SubhaloLenType, SnapOffsetsSubhalo=SnapOffsetsSubhalo)
                    continue

                if field in ['halo_id']:
                    # determine parent subhalo ID
                    r[field][m,wType] = inverseMapPartIndicesToHaloIDs(sP, indsType, ptName, 
                                          GroupLenType=GroupLenType, SnapOffsetsGroup=SnapOffsetsGroup)
                    continue

                # general properties
                if field not in halo_rel_fields:
                    # load parent property
                    data = sP.snapshotSubsetP(ptName, field, inds=indsType)

                    # save directly (by dimension) if not calculating further
                    if data.ndim == 1:
                        r[field][m,wType] = data
                    else:
                        r[field][m,wType,:] = data

                    continue

                # field is relative to halo properties? extract halo values at this snapshot
                if mpb is None:
                    raise Exception('Error, mpb track required as inputs.')

                if sP.isZoom:
                    # for zoom runs, we use the smoothed, single MPB track of the target for all particles
                    mpbInd = np.where( mpb['SnapNum'] == snap )[0]

                    if len(mpbInd) == 0:
                        raise Exception('Error, snap ['+str(snap)+'] not found in mpb.')

                    haloCenter = mpb['sm']['pos'][mpbInd[0],:]
                    haloVel    = mpb['sm']['vel'][mpbInd[0],:]
                    haloVirRad = mpb['sm']['rvir'][mpbInd[0]]
                    haloVirVel = mpb['sm']['vvir'][mpbInd[0]]
                else:
                    # for global catalogs (tracers spanning many/all halos) map groupcat values at 
                    # this snapshot into one [halo pos,vel,virrad,virvel] per tracer

                    # get subhalo ID targets at this snapshot (subhaloIDs_target contains -1 values)
                    subhaloID_trackIndexByTr = mpb['subhalo_evo_index'][wType]
                    subhaloIDs_target = mpb['subhalo_ids_evo'][m,subhaloID_trackIndexByTr]
                    subhaloID_trackIndexByTr = None

                    if exitAfterOneSnap:
                        # memory optimization, do gas last (largest), and erase things we don't need anymore
                        if tracerParsLocal is None:
                            mpb = None

                    # assumed in m indexing above, e.g. we called getEvoSnapList(sP) with no args in the driver
                    assert toRedshift is None and snapStep == 1 

                    # map (N,3) per-subhalo quantities
                    if field not in ['rad','rad_rvir']: # use periodicDistsIndexed instead to save memory
                        haloCenter = np.zeros( (indsType.size,3), dtype='float32' )

                        for i in range(3):
                            haloCenter[:,i] = gcHalo['subhalos']['SubhaloPos'][subhaloIDs_target,i]

                    # map (N) quantities, which happen to be per-halo
                    haloIDs_target = gcHalo['subhalos']['SubhaloGrNr'][subhaloIDs_target]

                    haloVirRad = gcHalo['halos']['Group_R_Crit200'][haloIDs_target]

                    if field in ['vrad','vrad_vvir','angmom']:
                        haloVel = np.zeros( (indsType.size,3), dtype='float32' )
                        for i in range(3):
                            haloVel[:,i] = Subhalo_StellarMeanVel[subhaloIDs_target,i]
                        haloVirVel = gcHalo['halos']['Group_M_Crit200'][haloIDs_target]
                        haloVirVel = sP.units.codeMassToVirVel(haloVirVel)

                    haloIDs_target = None

                # load required raw fields(s) from the snapshot
                data = {}
                for fName in halo_rel_fields[field]:
                    data[fName] = sP.snapshotSubset(ptName, fName, float32=True)
                    data[fName] = data[fName][indsType]

                # compute (the 4 halo properties can be scalar or arrays of the same size of indsType)
                if field in ['rad','rad_rvir']:
                    # radial distance from halo center, optionally normalized by rvir (r200crit)
                    #val = sP.periodicDistsN(haloCenter, data['pos']) # code units (e.g. ckpc/h)
                    val = sP.periodicDistsIndexed(gcHalo['subhalos']['SubhaloPos'], data['pos'], subhaloIDs_target)

                    data = None

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

                if val.ndim != 1:
                    raise Exception('Unexpected.')

                # save
                r[field][m,wType] = val

                # handle untracked halos at this snapshot which are marked by -1 (fill with NaN)
                if not sP.isZoom:
                    w_untracked = np.where(subhaloIDs_target < 0)[0]
                    if len(w_untracked):
                        r[field][m,wType[w_untracked]] = np.nan

            # internal saving of this field? do so now, and mark this snapshot as done
            if saveFilename is not None:
                with h5py.File(saveFilename,'r+') as f:
                    f['done'][saveSnapInd] = 1
                    done[saveSnapInd] = 1

                    if r[field].ndim == 2:
                        f[field][saveSnapInd,:] = r[field][m,:]
                    else:
                        f[field][saveSnapInd,:,:] = r[field][m,:,:]

                print('  Saved snapshot index [%d] to [%s].' % (saveSnapInd,saveFilename.split("/")[-1]))

        if exitAfterOneSnap and parFields[0] != 'halo_id': # so we can compute globalTracerMPBMap() for 'rad' catalog
            print('Exiting for now (one snap at a time)!')
            sP.setSnap(startSnap)
            return

    sP.setSnap(startSnap)
    return r

def subhalosTracersTimeEvo(sP,subhaloIDs,toRedshift,trFields,parFields,parPartTypes,outPath,onlySnap=None):
    """ For a set of subhaloIDs, determine all their child tracers at sP.redshift then record their 
        properties back in time. """
    # load global ParentID for all tracers at the starting snapshot, and pre-sort
    #ParentID = sP.snapshotSubsetP('tracer', 'ParentID')
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
            shDetails = sP.groupCatSingle(subhaloID=subhaloID)
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
    tracerProps = tracersTimeEvo(sP, trSearchIDs, trFields, parFields, toRedshift, saveFilename='/u/dnelson/data/temp.hdf5', onlySnap=onlySnap)

    # separate back out by subhalo and save one data file per subhalo
    offset = 0

    for i, subhaloID in enumerate(subhaloIDs):
        outFilePath = outPath + '_' +  str(subhaloID) + '_subhalo.hdf5'

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
        snapFinal = sP.redshiftToSnapNum(toRedshift)
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
        haloID = sP.groupCatSingle(subhaloID=subhaloID)['SubhaloGrNr']
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

def globalTracerLength(sP, halos=False, subhalos=False, histoMethod=True, haloTracerOffsets=None):
    """ Return a 1D array of the number of tracers per halo or subhalo, for all in the snapshot, in 
    direct analogy to LenType in the group catalogs. Compute the offsets as well. """
    assert halos is True or subhalos is True # pick one
    if subhalos is True: assert haloTracerOffsets is not None # required for subhalo-based offsets

    if histoMethod:
        # load and count the parent IDs of all tracers in this snapshot
        ParentID = sP.snapshotSubsetP('tracer', 'ParentID')

        # if the IDs of parents are dense enough, use a histogram counting approach
        ParentID_min = ParentID.min()
        ParentID_max = ParentID.max()
        assert ParentID_max - ParentID_min <= 68e9 # up to 507GB memory allocation here

        # offset ParentIDs by their minimum, cast into signed type, then histogram
        ParentID -= ParentID_min

        assert ParentID.max() < np.iinfo('int64').max
        ParentID.dtype = np.int64 # change dtype (unsafe cast)

        ParentHisto = bincount(ParentID, dtype=np.int32)

        assert ParentHisto.max() < np.iinfo('int32').max

        ParentID = None
    else:
        # load and sort the parent IDs of all tracers in this snapshot
        ParentID = sP.snapshotSubsetP('tracer', 'ParentID')

        # otherwise do a pre-sort and we will later do many bisection searches
        ParentIDSortInds = np.argsort(ParentID, kind='mergesort')
        ParentID = ParentID[ParentIDSortInds]

    # allocate counts
    h = sP.groupCatHeader()
    if halos: nObjs = h['Ngroups_Total']
    if subhalos: nObjs = h['Nsubgroups_Total']

    trCounts  = {}
    trOffsets = {}

    for pt in defParPartTypes:
        trCounts[pt]  = np.zeros( nObjs, dtype='int32' )
        trOffsets[pt] = np.zeros( nObjs, dtype='int64' )

    # load offsets
    if halos:
        gc = sP.groupCat(fieldsHalos=['GroupLenType'])
        objOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']

    if subhalos:
        gc = sP.groupCat(fieldsSubhalos=['SubhaloLenType'])
        groupNsubs = sP.groupCat(fieldsHalos=['GroupNsubs'])
        objOffsetType = sP.groupCatOffsetListIntoSnap()['snapOffsetsSubhalo']

    # consider each parent type
    for parPartType in defParPartTypes:
        # load IDs of this type in the requested object
        print(parPartType)

        ptNum = sP.ptNum(parPartType)
        parIDsType = sP.snapshotSubsetP(parPartType, 'id')

        # no particles of this type in the snapshot
        if isinstance(parIDsType,dict) and parIDsType['count'] == 0:
            continue

        if histoMethod:
            parIDsType -= ParentID_min # offset

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

            if histoMethod:
                # truncate any ParentID's which exceed ParentHisto bounds (by definition have 0 tracers)
                parIDsTypeLocal = parIDsTypeLocal[(parIDsTypeLocal >= 0) & \
                                                  (parIDsTypeLocal < ParentHisto.size)]

                # sum histogram entries of offset parIDsType
                trCounts[parPartType][i] = ParentHisto[parIDsTypeLocal].sum()
            else:
                # crossmatch
                trIDs = getTracerChildren(sP, parIDsTypeLocal, inds=True, 
                                          ParentID=ParentID, ParentIDSortInds=ParentIDSortInds)

                # save counts
                if trIDs is not None:
                    trCounts[parPartType][i] = len(trIDs)

    # build offsets
    if halos:
        trOffsets['gas'][1:]   = np.cumsum( trCounts['gas'] )[:-1] # gas are first, obj by obj
        trOffsets['stars'][1:] = np.cumsum( trCounts['stars'] )[:-1] # stars/pt4 are next
        trOffsets['bhs'][1:]   = np.cumsum( trCounts['bhs'] )[:-1] # bhs are last
        trOffsets['stars'] += np.sum( trCounts['gas'] )
        trOffsets['bhs']   += np.sum( trCounts['gas'] ) + np.sum( trCounts['stars'] )

    if subhalos:
        # need to handle relative to halo offsets to account for inner fuzz
        for pt in ['gas','stars','bhs']:
            shCount = 0

            for i in np.arange(h['Ngroups_Total']):
                if groupNsubs[i] == 0:
                    continue

                # handle GroupFirstSub
                trOffsets[pt][shCount] = haloTracerOffsets[pt][i]
                shCount += 1

                # handle all secondary subhalos
                for j in np.arange(1, groupNsubs[i]):
                    trOffsets[pt][shCount] = trOffsets[pt][shCount-1] + trCounts[pt][shCount-1]
                    shCount += 1

    return trCounts, trOffsets

def globalTracerMPBMap(sP, halos=False, subhalos=False, trIDs=None, retMPBs=False, 
                       extraFields=[], indRange=None):
    """ Load all MPBs of a global tracer set and create a mapping between unique MPBs and tracers.
    If indRange is not None (only so far when calling from accMode()), attempt to get halo_id's 
    using already computed parent_indextype tracer_tracks (for speed and efficiency only). """
    assert halos is True or subhalos is True # pick one

    treeName = 'SubLink'

    mpb = {}

    # check cache file
    hStr = 'halos' if halos else 'subhalos'
    saveFilename = sP.derivPath + 'tracer_tracks/globalTracerMPBMap_%s_%d.hdf5' % (hStr,sP.snap)
    if isfile(saveFilename) and not retMPBs and len(extraFields) == 0 and indRange is None:
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                mpb[key] = f[key][()]
        print('Loaded cached [%s]' % saveFilename)

        return mpb

    if trIDs is None: # lazy load
        trIDs = globalTracerChildren(sP, halos=halos, subhalos=subhalos)

    # where do we start each tree? all tracers in a given FoF are assigned to follow the MPB
    # of the central subhalo of that FoF at sP.snap! e.g. tracers in satellites at sP.snap 
    # will derive a rad/rad_rvir corresponding to the ~distance of the satellite from the halo center
    print('Identifying central subhalo IDs for all [%d] tracers at snap [%d]...' % (trIDs.size,sP.snap))

    if indRange is None:
        trVals = tracersTimeEvo(sP, trIDs, [], ['halo_id'], toRedshift=sP.redshift) 
        trVals['halo_id'] = np.squeeze(trVals['halo_id']) # single snap
    else:
        # load parent_indextype of the subset
        indRange3 = [indRange[0], indRange[1], 0] # take i=0 slice along snapshot axis
        par_indtype = globalAllTracersTimeEvo(sP, 'parent_indextype', halos=True, indRange=indRange3)
        assert par_indtype['snaps'][0] == sP.snap
        assert par_indtype['parent_indextype'].size == trIDs.size

        par_indtype = par_indtype['parent_indextype']

        GroupLenType = sP.groupCat(fieldsHalos=['GroupLenType'])
        SnapOffsetsGroup = sP.groupCatOffsetListIntoSnap()['snapOffsetsGroup']

        trVals = {}
        trVals['halo_id'] = np.zeros( trIDs.size, dtype='int32' )
        trVals['halo_id'].fill(np.nan)

        for ptName in defParPartTypes:
            ptMin = 1e11 * sP.ptNum(ptName) # see parent_indextype documentation
            ptMax = 1e11 * (sP.ptNum(ptName)+1)

            wType = np.where( (par_indtype >= ptMin) & (par_indtype < ptMax) )[0]
            indsType = par_indtype[wType] - ptMin

            trVals['halo_id'][wType] = inverseMapPartIndicesToHaloIDs(sP, indsType, ptName, 
                                         GroupLenType, SnapOffsetsGroup)

        assert np.count_nonzero( np.isnan(trVals['halo_id']) ) == 0

    # map the FoF IDs at z=0 into subhalo_ids with GroupFirstSub
    GroupFirstSub = sP.groupCat(fieldsHalos=['GroupFirstSub'])
    trVals['subhalo_id'] = GroupFirstSub[trVals['halo_id']]

    # debug: how many FoF's have GroupFirstSub==-1 already at z=0 (these by definition will be 
    # filled entirely with NaN for halo_rel_fields, how much space wasted?)
    ww = np.where(trVals['subhalo_id'] == -1)[0]
    print(' note: [%d] of [%d] tracers are in FoFs with no central subhalo' % (ww.size,trIDs.size))

    # reduce to a unique subset
    uniqSubhaloIDs = np.unique(trVals['subhalo_id'])
    print('Finding [%d] unique subhalo MPB tracks, of [%d] total...' % (uniqSubhaloIDs.size,trIDs.size))

    # allocate, -1 indicates untracked at that snapshot
    evoSnaps, _ = getEvoSnapList(sP)
    mpb['subhalo_ids_evo'] = np.zeros( (len(evoSnaps),uniqSubhaloIDs.size), dtype='int32' ) - 1

    # load MPBs of all subhalos
    treeFileFields = loadTreeFieldnames(sP, treeName=treeName)
    fields = ['SnapNum','SubfindID']

    for field in iterable(extraFields):
        if field not in fields and field in treeFileFields:
            fields.append(field)

    mpbs = sP.loadMPBs(uniqSubhaloIDs, fields=fields, treeName=treeName)

    for i, id in enumerate(uniqSubhaloIDs):
        # untracked?
        if id == -1: continue # parent fof of tracer has no central subhalo at sP.snap
        if id not in mpbs: continue # central subhalo not in tree at sP.snap

        # crossmatch MPB snapshots we have with target snaps, and save subhalo IDs
        w1, w2 = match3(mpbs[id]['SnapNum'], evoSnaps)

        mpb['subhalo_ids_evo'][w2,i] = mpbs[id]['SubfindID'][w1]

    # save a mapping between the subhalo IDs of each tracer here at sP.snap and the unique 
    # subhalo tracks we have saved in subhalo_ids_evo
    wUniq, _ = match3(uniqSubhaloIDs, trVals['subhalo_id'])
    mpb['subhalo_evo_index'] = wUniq

    assert wUniq.size == trIDs.size == trVals['subhalo_id'].size == trVals['halo_id'].size

    # save cache
    if 1 and not retMPBs:
        with h5py.File(saveFilename,'w') as f:
            for key in mpb:
                f[key] = mpb[key]
        print('Saved: [%s]' % saveFilename)

    if retMPBs: # attach to return
        mpb['mpbs'] = mpbs
        mpb['subhalo_id'] = trVals['subhalo_id'] # indices into mpb['mpbs'] for each tracer

    return mpb

def globalAllTracersTimeEvo(sP, field, halos=True, subhalos=False, indRange=None, toRedshift=None):
    """ For all tracers in all FoFs at the simulation endtime, record time evolution tracks of one 
    field for all snapshots (which contain tracers). """
    assert halos is True or subhalos is True or sP.isSubbox # pick one

    savePath = sP.postPath + '/tracer_tracks/'
    if not isdir(savePath):
        mkdir(savePath)

    if halos: selectStr = 'groups'
    if subhalos: selectStr = 'subhalos'
    if sP.isSubbox:
        assert field not in halo_rel_fields
        halos = False
        subhalos = False
        selectStr = 'subbox%d' % sP.subbox

    saveFilename = savePath + 'tr_all_%s_%d_%s.hdf5' % (selectStr,sP.snap,field)
    if toRedshift is not None and toRedshift < sP.redshift: # forward in time
        saveFilename = saveFilename.replace("_%d_" % sP.snap, "_+%d_" % sP.snap)

    # single load requested? do now and return
    if isfile(saveFilename):
        r = {}

        done = 1
        with h5py.File(saveFilename,'r') as f:
            # restartable? check if we are done, otherwise continue calculation
            if 'done' in f:
                done = f['done'][()].min()

        if done or indRange is not None:
            # skip all of this and fall through to computation if any snapshots are not yet done
            with h5py.File(saveFilename,'r') as f:

                for k1 in f:
                    if isinstance(f[k1], h5py.Dataset):
                        # we have a dataset, full or subset read?
                        if indRange is None:
                            # read full dataset
                            r[k1] = f[k1][()]
                        else:
                            if f[k1].ndim == 1:
                                # read full dataset for 1D (e.g. redshifts)
                                r[k1] = f[k1][()]
                            else:
                                # read partial dataset for 2D/3D (shape is [Nsnaps,Ntr] or [Nsnaps,Ntr,3])
                                assert len(indRange) in [1,2,3]
                                if len(indRange) == 2:
                                    r[k1] = f[k1][:, indRange[0]:indRange[1]]
                                elif len(indRange) == 3:
                                    r[k1] = f[k1][indRange[2], indRange[0]:indRange[1]]
                                else:
                                    r[k1] = f[k1][indRange[0],:]

                        continue

                    # handle meta: nested Halo/TracerLength/gas type structure
                    for k2 in f[k1]:
                        for k3 in f[k1][k2]:
                            r[k1+'_'+k2+'_'+k3] = f[k1][k2][k3][()]

            # handle TNG temperature correction (for Powell term) in subboxes
            if 'TNG' in sP.simName and sP.isSubbox and field in ['temp','temp_sfcold']:
                w = np.where(r[field] < 3.9)
                r[field][w] = 4.0
                print(' notice: set [%d] of [%d] subbox tracer temperatures from <3.9 to 4.0.' % (len(w[0]),r[field].size))
                
            return r

    if indRange is not None:
        print('Warning: globalAllTracersTimeEvo() returning None, indRange input but [%s] does not exist.' % saveFilename)
        return None # if we requested a subset, were expecting existence, indicate problem

    # get child tracers of all particle types in all FoFs
    trIDs = globalTracerChildren(sP, halos=halos, subhalos=subhalos)

    # mpb? are we calculating parent properties which are relative to [evolving] halo properties?
    mpb = None

    if field in halo_rel_fields:
        mpb = globalTracerMPBMap(sP, halos=halos, subhalos=subhalos, trIDs=trIDs)

    # follow tracer and tracer parent properties (one at a time and save) from sP.snap back to snap=0
    print('Computing: [%s]' % saveFilename.split(savePath)[1], flush=True)

    if field == 'meta':
        # save the metadata (ordered tracer/parent IDs, lengths/offsets of tracers by group/subhalo)
        trVals = {}

        # save the tracer IDs in the order we are saving the tracks
        trVals['TracerIDs'] = trIDs

        if not sP.isSubbox:
            # compute lengths/offsets by halo
            trCounts, trOffsets = globalTracerLength(sP, halos=True)

            for key in trCounts:
                trVals['Halo/TracerLength/'+key] = trCounts[key]
                trVals['Halo/TracerOffset/'+key] = trOffsets[key]

            # compute lengths/offsets by halo
            trCounts, trOffsets = globalTracerLength(sP, subhalos=True, haloTracerOffsets=trOffsets)

            for key in trCounts:
                trVals['Subhalo/TracerLength/'+key] = trCounts[key]
                trVals['Subhalo/TracerOffset/'+key] = trOffsets[key]

        # save the parent IDs of these tracers in the same order
        TracerID = sP.snapshotSubsetP('tracer', 'TracerID')

        trInds,_ = match3(TracerID,trIDs)
        assert trInds.size == trIDs.size

        trVals['ParentIDs'] = sP.snapshotSubsetP('tracer', 'ParentID', inds=trInds)

        # save
        with h5py.File(saveFilename,'w') as f:
            for key in trVals:
                f[key] = trVals[key]

        print('Saved: [%s]' % saveFilename.split(savePath)[1])

    else:
        # saved inside tracersTimeEvo() one snapshot at a time, can be restarted
        oneSnap = (sP.res == 2160 and not sP.isSubbox)

        if 'tracer_' in field: # trField
            trVals = tracersTimeEvo(sP, trIDs, [field], [], toRedshift=toRedshift, mpb=mpb, saveFilename=saveFilename, exitAfterOneSnap=oneSnap)
        else: # parField
            trVals = tracersTimeEvo(sP, trIDs, [], [field], toRedshift=toRedshift, mpb=mpb, saveFilename=saveFilename, exitAfterOneSnap=oneSnap)

    return trVals

def checkTracerMeta(sP):
    """ Verify globalAllTracersTimeEvo() ordering. """
    nRandom = 100
    nIndiv = 5
    ptTypes = ['gas','stars','bhs']

    # load
    meta = globalAllTracersTimeEvo(sP, 'meta', halos=True)
    h = sP.groupCatHeader()
    
    # load and sort the parent IDs of all tracers in this snapshot
    ParentIDorig = sP.snapshotSubsetP('tracer', 'ParentID')
    TracerID = sP.snapshotSubsetP('tracer', 'TracerID')
    ParentIDSortInds = np.argsort(ParentIDorig, kind='mergesort')
    ParentID = ParentIDorig[ParentIDSortInds]

    # for nRandom halos and nRandom subhalos, verify child tracers are correct
    for i in range(nRandom):
        haloID = np.random.randint(0,h['Ngroups_Total'])
        subhaloID = np.random.randint(0,h['Nsubgroups_Total'])
        print(i,haloID,subhaloID)

        # get actual children
        trIDs_halo_check = subhaloTracerChildren(sP, haloID=haloID, concatTypes=False,
                                            ParentID=ParentID, ParentIDSortInds=ParentIDSortInds)
        trIDs_subhalo_check = subhaloTracerChildren(sP, subhaloID=subhaloID, concatTypes=False,
                                            ParentID=ParentID, ParentIDSortInds=ParentIDSortInds)

        # loop over types
        for pt in ptTypes:
            # handle empty cases consistently
            if pt not in trIDs_halo_check: trIDs_halo_check[pt] = np.array([])
            if pt not in trIDs_subhalo_check: trIDs_subhalo_check[pt] = np.array([])
            if trIDs_halo_check[pt] is None: trIDs_halo_check[pt] = np.array([])
            if trIDs_subhalo_check[pt] is None: trIDs_subhalo_check[pt] = np.array([])
            print(' %s %d %d' % (pt,trIDs_halo_check[pt].size,trIDs_subhalo_check[pt].size))

            # halo
            i0 = meta['Halo_TracerOffset_'+pt][haloID]
            i1 = meta['Halo_TracerLength_'+pt][haloID] + i0
            trIDs_halo_pt = meta['TracerIDs'][i0:i1]

            assert np.array_equal(np.sort(trIDs_halo_pt),np.sort(trIDs_halo_check[pt]))

            # subhalo
            i0 = meta['Subhalo_TracerOffset_'+pt][subhaloID]
            i1 = meta['Subhalo_TracerLength_'+pt][subhaloID] + i0
            trIDs_subhalo_pt = meta['TracerIDs'][i0:i1]

            assert np.array_equal(np.sort(trIDs_subhalo_pt),np.sort(trIDs_subhalo_check[pt]))

            # check first five individual parents ordered children
            subhalo_pt_ids = sP.snapshotSubset(pt, 'id', subhaloID=subhaloID)
            if isinstance(subhalo_pt_ids,dict) and subhalo_pt_ids['count'] == 0:
                continue

            children_ids = []

            for j in range(nIndiv):
                if j >= subhalo_pt_ids.size:
                    continue
                ww = np.where(ParentIDorig == subhalo_pt_ids[j])[0]
                if not len(ww):
                    continue
                children_ids.append( TracerID[ww] )

            children_ids = np.array(children_ids)
            if children_ids.size:
                children_ids = np.hstack(children_ids)
                assert np.array_equal(children_ids, trIDs_subhalo_pt[0:children_ids.size])
