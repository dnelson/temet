"""
load/auxcat.py
  Loading I/O - postprocessing auxcats.
"""
import numpy as np
import h5py
from os.path import isfile, isdir
from os import mkdir, unlink
from getpass import getuser

from util.helper import iterable, curRepoVersion

def auxCat(sP, fields=None, pSplit=None, reCalculate=False, searchExists=False, indRange=None, onlyMeta=False, expandPartial=False):
    """ Load field(s) from the auxiliary group catalog, computing missing datasets on demand. 
      reCalculate  : force redo of computation now, even if data is already saved in catalog
      searchExists : return None if data is not already computed, i.e. do not calculate right now 
      indRange     : if a tuple/list, load only the specified range of data (field and e.g. subhaloIDs)
      onlyMeta     : load only attributes and coverage information 
      expandPartial : if data was only computed for a subset of all subhalos, expand this now into a total nSubs sized array """

    epStr = '_ep' if expandPartial else ''
    if len(iterable(fields)) == 1 and 'ac_'+iterable(fields)[0]+epStr in sP.data and not reCalculate:
        return sP.data['ac_'+iterable(fields)[0]+epStr].copy() # cached, avoid view

    def _comparatorListInds(fieldName):
        # transform e.g. 'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_13' into 13 (an integer) for sorting comparison
        num = fieldName.rsplit('_', 1)[-1]
        return int(num)

    def _concatSplitFiles(field, datasetName):
        # specified chunk exists, do all exist? check and record sizes
        allExist = True
        allCount = 0
        allCountPart = 0

        for i in range(pSplit[1]):
            auxCatPathSplit_i = pathStr2 % (field,sP.snap,i,pSplit[1])
            if not isfile(auxCatPathSplit_i):
                allExist = False
                continue

            # record counts and dataset shape
            with h5py.File(auxCatPathSplit_i,'r') as f:
                allShape = f[datasetName].shape

                if len(allShape) > 1 and allShape[condAxis] > condThresh:
                    # very high dimensionality auxCat (e.g. full spectra), save only non-nan
                    assert f[datasetName].ndim == 2
                    assert 'partInds' not in f or f['partInds'][()] == -1

                    temp_r = f[datasetName][()]
                    w = np.where( np.isfinite(np.sum(temp_r,axis=condAxis)) )
                    allCount += len(w[0])
                    print(' [%2d] keeping %d of %d non-nan.' % (i,len(w[0]),f['subhaloIDs'].size))
                else:
                    # normal, save full Subhalo size
                    allCount += f['subhaloIDs'].size
                    if 'partInds' in f: allCountPart += f['partInds'].size

        if not allExist:
            print('Chunk [%s] already exists, but all not yet done, exiting.' % auxCatPathSplit)
            #r[field] = None
            return
            
        # for full saves we want (auxCat size is groupcat size), assuming we computed a subset of objects
        numSubs = sP.groupCatHeader()['Nsubgroups_Total']
        writeSparseCalcFullSize = False
        if numSubs > allCount:
            print('Note: Increasing save size from [%d] computed, to full groupcat size [%d].' % (allCount,numSubs))
            allCount = numSubs
            writeSparseCalcFullSize = True

        # all chunks exist, concatenate them now and continue
        catIndFieldName = 'subhaloIDs'

        if allCountPart > 0:
            # can relax this if we ever do Particle* auxCat with pts other than stars:
            assert allCountPart == snapshotHeader(sP)['NumPart'][sP.ptNum('stars')]
            
            # proceed with 1-per-particle based concatenation, then 'subhaloIDs' = 'partInds'
            allCount = allCountPart
            catIndFieldName = 'partInds'

        allShape = np.array(allShape)
        allShape[0] = allCount # size
        offset = 0

        new_r = np.zeros( allShape, dtype='float32' )
        subhaloIDs = np.zeros( allCount, dtype='int32' )
        attrs = {}

        new_r.fill(np.nan) # note: will be nan for unprocessed subhalos, and e.g. empty bins in processed subs
        subhaloIDs.fill(-1)

        print(' Concatenating into shape: ', new_r.shape)

        for i in range(pSplit[1]):
            auxCatPathSplit_i = pathStr2 % (field,sP.snap,i,pSplit[1])

            with h5py.File(auxCatPathSplit_i,'r') as f:
                # load
                length = f[catIndFieldName].size
                temp_r = f[datasetName][()]

                if length == 0: continue

                if len(allShape) > 1 and allShape[condAxis] >= condThresh or f[datasetName].ndim > 3:
                    # want to condense, stamp in to dense indices
                    assert not writSparseCalcFullSize # oops, contradiction, if we really want to condense 
                    # then disable 'allCount = numSubs' increase above, otherwise change condThresh etc 
                    # here to avoid 'condensing' back into a smaller size
                    subhaloIDsToSave = f[catIndFieldName][()]

                    if f[datasetName].ndim == 2 and 'Subhalo_SDSSFiberSpectra' in field:
                        # splits saved all subhalos (sdss spectra)
                        #assert 'Subhalo_SDSSFiberSpectra' in field # otherwise check
                        w = np.where( np.isfinite(np.sum(temp_r,axis=condAxis)) )[0]
                        length = len(w)
                        temp_r = temp_r[w,:]
                        subhaloIDsToSave = subhaloIDsToSave[w]

                    subhaloIDs[offset : offset+length] = subhaloIDsToSave
                    subhaloIndsStamp = np.arange(offset,offset+length)
                else:
                    # full, stamp in to indices corresponding to subhalo indices
                    subhaloIndsStamp = f[catIndFieldName][()]
                    subhaloIDs[subhaloIndsStamp] = subhaloIndsStamp

                # save into final array
                new_r[subhaloIndsStamp,...] = temp_r

                for attr in f[datasetName].attrs:
                    attrs[attr] = f[datasetName].attrs[attr]

                offset += length

                if 'wavelength' in f: attrs['wavelength'] = f['wavelength'][()]

        if not writeSparseCalcFullSize:
            assert np.count_nonzero(np.where(subhaloIDs < 0)) == 0
            assert np.count_nonzero(new_r == -1.0) == 0
        else:
            print(' WARNING: Skipping checks that we wrote data for all subhalos, since we wrote a sparse calc to a full size auxCat.')

        # auxCat already exists? only allowed if we are processing multiple fields
        if isfile(auxCatPath):
            with h5py.File(auxCatPath,'r') as f:
                assert field+'_0' in f
                assert datasetName not in f

        # save (or append to) new auxCat
        with h5py.File(auxCatPath,'a') as f:
            f.create_dataset(datasetName, data=new_r)
            if catIndFieldName == 'subhaloIDs' and catIndFieldName not in f:
                f.create_dataset(catIndFieldName, data=subhaloIDs)
            for attr, attrValue in attrs.items():
                if attr in largeAttrNames:
                    f.create_dataset(attr, data=attrValue)
                    continue # typically too large to store as an attribute
                f[datasetName].attrs[attr] = attrValue

        # remove split files
        for i in range(pSplit[1]):
            auxCatPathSplit_i = pathStr2 % (field,sP.snap,i,pSplit[1])
            unlink(auxCatPathSplit_i)

        print(' Concatenated new [%s] and saved, split files deleted.' % auxCatPath.split("/")[-1])
        return

    def _expand_partial():
        """ Helper, expand a subhalo-partial aC into a subhalo-complete array. """
        nSubsTot = sP.numSubhalos

        assert 'subhaloIDs' in r # else check why we are here

        if r['subhaloIDs'].size == nSubsTot:
            return r[field]
        
        if r['subhaloIDs'].size < nSubsTot:
            shape = np.array(r[field].shape)
            shape[0] = nSubsTot
            new_data = np.zeros( shape, dtype=r[field].dtype )
            new_data.fill(np.nan)
            new_data[r['subhaloIDs'],...] = r[field]
            print(' Auxcat Expanding [%d] to [%d] elements for [%s].' % (r[field].shape[0],new_data.shape[0],field))
            return new_data

    from cosmo import auxcatalog
    import datetime

    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No auxCat() for subbox snapshots."

    condThresh = 150 # threshold size of any dimension such that we save a condensed auxCat
    condAxis = 1 # currently only implemented for auxCat.shape = [subhaloIDs,*] (i.e. ndim==2)
    largeAttrNames = ['subhaloIDs','partInds','wavelength'] # save these as separate datasets, if present

    pathStr1 = sP.derivPath + 'auxCat/%s_%03d.hdf5'
    pathStr2 = sP.derivPath + 'auxCat/%s_%03d-split-%d-%d.hdf5'

    r = {}

    if not isdir(sP.derivPath + 'auxCat'):
        mkdir(sP.derivPath + 'auxCat')

    for field in iterable(fields):
        if field not in list(auxcatalog.fieldComputeFunctionMapping.keys()) + auxcatalog.manualFieldNames:
            raise Exception('Unrecognized field ['+field+'] for auxiliary catalog.')

        # check for existence of auxiliary catalog file for this dataset
        auxCatPath = pathStr1 % (field,sP.snap)

        # split the calculation over multiple jobs? check if all chunks already exist
        if pSplit is not None:
            assert indRange is None # for load only

            auxCatPathSplit = pathStr2 % (field,sP.snap,pSplit[0],pSplit[1])

            if isfile(auxCatPathSplit) and not reCalculate:
                # setup combine, for 1 dataset or for multiple
                with h5py.File(auxCatPathSplit,'r') as f:
                    readFields = [field]
                    if field not in f:
                        assert field+'_0' in f # multiple datasets
                        readFields = sorted([key for key in f.keys() if field in key])

                # combine now
                for readField in readFields:
                    _concatSplitFiles(field, readField)

        # just checking for existence? (do not calculate right now if missing)
        if not isfile(auxCatPath) and searchExists:
            r[field] = None
            continue

        # load previously computed values
        if isfile(auxCatPath) and not reCalculate:
            #print('Load existing: ['+field+'] '+auxCatPath)
            with h5py.File(auxCatPath,'r') as f:
                # load data
                readFields = [field]
                if field not in f:
                    assert field+'_0' in f # list of ndarrays
                    readFields = sorted([key for key in f.keys() if field in key], key=_comparatorListInds)

                if not onlyMeta:
                    # read 1 or more datasets, keep list only if >1
                    rr = []

                    for readField in readFields:
                        if indRange is None:
                            data = f[readField][()]
                        else:
                            data = f[readField][indRange[0]:indRange[1],...]
                            data = np.squeeze(data) # remove any degenerate dimensions
                        rr.append(data)

                    if len(rr) == 1:
                        r[field] = rr[0]
                    else:
                        r[field] = rr

                # load metadata
                r[field+'_attrs'] = {}
                for attr in f[readFields[0]].attrs:
                    r[field+'_attrs'][attr] = f[readFields[0]].attrs[attr]

                # load subhaloIDs, partInds if present
                for attrName in largeAttrNames:
                    if attrName in f:
                        if indRange is None:
                            r[attrName] = f[attrName][()]
                        else:
                            r[attrName] = f[attrName][indRange[0]:indRange[1]]

            # subhaloIDs indicates computation only for partial set of subhalos?
            if expandPartial:
                assert len(readFields) == 1
                r[field] = _expand_partial()

            # cache
            sP.data['ac_'+field+epStr] = {}
            for key in r:
                sP.data['ac_'+field+epStr][key] = r[key]

            continue

        # either does not exist yet, or reCalculate requested
        if field in auxcatalog.manualFieldNames:
            raise Exception('Error: auxCat [%s] does not yet exist, but must be manually created.' % field)
            
        pSplitStr = ''
        savePath = auxCatPath

        if pSplit is not None:
            pSplitStr = ' (split %d of %d)' % (pSplit[0],pSplit[1])
            savePath = auxCatPathSplit

        print('Compute and save: [%s] [%s]%s' % (sP.simName,field,pSplitStr))

        r[field], attrs = auxcatalog.fieldComputeFunctionMapping[field] (sP, pSplit)
        r[field+'_attrs'] = attrs

        # include subhaloIDs, partInds if present
        for attrName in largeAttrNames:
            if attrName in attrs:
                if indRange is None:
                    r[attrName] = attrs[attrName]
                else:
                    r[attrName] = attrs[attrName][indRange[0]:indRange[1]]

        # save new dataset (or overwrite existing)
        with h5py.File(savePath,'w') as f:
            if isinstance(r[field],list):
                # list of ndarrays, save each as separate dataset
                for i in range(len(r[field])):
                    f.create_dataset(field+'_'+str(i), data=r[field][i])
                fieldAttrSave = field + '_0' # save attributes to this dataset
            else:
                # single ndarray (typical case)
                f.create_dataset(field, data=r[field])
                fieldAttrSave = field

            # save metadata and any additional descriptors as attributes
            f[fieldAttrSave].attrs['CreatedOn']   = datetime.date.today().strftime('%d %b %Y')
            f[fieldAttrSave].attrs['CreatedRev']  = curRepoVersion()
            f[fieldAttrSave].attrs['CreatedBy']   = getuser()
            
            for attrName, attrValue in attrs.items():
                if attrName in largeAttrNames:
                    f.create_dataset(attrName, data=r[field+'_attrs'][attrName])
                    continue # typically too large to store as an attribute
                f[fieldAttrSave].attrs[attrName] = attrValue

        if not reCalculate:
            print(' Saved new [%s].' % savePath)
        else:
            print(' Saved over existing [%s].' % savePath)

        # subhaloIDs indicates computation only for partial set of subhalos? modify for immediate return
        if expandPartial:
            r[field] = _expand_partial()

    return r