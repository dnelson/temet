"""
cosmo/load.py
  Cosmological simulations - loading procedures (snapshots, fof/subhalo group cataloges, merger trees).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import glob
from functools import partial
from os.path import isfile, isdir
from os import mkdir, makedirs, unlink
from getpass import getuser

import illustris_python as il
from illustris_python.util import partTypeNum as ptNum
from util.helper import iterable, logZeroNaN, curRepoVersion, pSplitRange, numPartToChunkLoadSize

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

        new_r.fill(-1.0) # does validly contain nan
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

        assert np.count_nonzero(np.where(subhaloIDs < 0)) == 0
        assert np.count_nonzero(new_r == -1.0) == 0

        # auxCat already exists? only allowed if we are processing multiple fields
        if isfile(auxCatPath):
            with h5py.File(auxCatPath,'r') as f:
                assert field+'_0' in f
                assert datasetName not in f

        # save (or append to) new auxCat
        with h5py.File(auxCatPath) as f:
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

    condThresh = 100 # threshold size of any dimension such that we save a condensed auxCat
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

def gcPath(basePath, snapNum, chunkNum=0, noLocal=False, checkExists=False):
    """ Find and return absolute path to a group catalog HDF5 file.
        Can be used to redefine illustris_python version (il.groupcat.gcPath = cosmo.load.gcPath). """

    # local scratch test: call ourself with a basePath corresponding to local stratch (on freyator)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/mnt/nvme/cache/%s/%s/" % (bpSplit[-3], bpSplit[-2])
        localFT = gcPath(localBP, snapNum, chunkNum=chunkNum, noLocal=True, checkExists=True)

        if localFT:
            #print("Note: Reading group catalog from local scratch!")
            return localFT

    # format snapshot number
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # both fof+subfind in single (non-split) file in root directory
                  basePath + '/fof_subhalo_tab_' + ext + '.hdf5',
                  # standard: both fof+subfind in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # fof only, in >1 files per snapshot, in subdirectory
                  basePath + 'groups_' + ext + '/fof_tab_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # rewritten new group catalogs with offsets
                  basePath + 'groups_' + ext + '/groups_' + ext + '.' + str(chunkNum) + '.hdf5',
                  # single (non-split) file in subdirectory (i.e. Millennium rewrite)
                  basePath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.hdf5',
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    # failure:
    for fileName in fileNames:
        print(' '+fileName)
    raise Exception("No group catalog found.")

def groupCat(sP, readIDs=False, skipIDs=False, fieldsSubhalos=None, fieldsHalos=None, sq=True):
    """ Load HDF5 fof+subfind group catalog for a given snapshot.
                         
       readIDs=1 : by default, skip IDs since we operate under the group ordered snapshot assumption, but
                   if this flag is set then read IDs and include them (if they exist)
       skipIDs=1 : acknowledge we are working with a STOREIDS type .hdf5 group cat and don't warn
       fields*   : read only a subset fields from the catalog
       sq        : squeeze single field return into a numpy array instead of within a dict
    """
    assert sP.snap is not None, "Must specify sP.snap for groupCat() load."
    assert sP.subbox is None, "No groupCat() for subbox snapshots."
    assert fieldsSubhalos is not None or fieldsHalos is not None, "Must specify fields type."

    r = {}

    # derived SUBHALO fields and unit conversions (mhalo_200_log, ...). Can request >=1 custom fields 
    # and >=1 standard fields simultaneously, as opposed to snapshotSubset().
    if fieldsSubhalos is not None:

        for i, field in enumerate(fieldsSubhalos):
            quant = field.lower()

            # cache check
            cacheKey = 'gc_subcustom_%s' % field
            if cacheKey in sP.data:
                r[field] = sP.data[cacheKey]
                continue

            # halo mass (m200 or m500) of parent halo [code, msun, or log msun]
            if quant in ['mhalo_200','mhalo_200_log','mhalo_200_code',
                         'mhalo_500','mhalo_500_log','mhalo_200_code']:
                od = 200 if '_200' in quant else 500

                gc = groupCat(sP, fieldsHalos=['Group_M_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])

                r[field] = gc['halos']['Group_M_Crit%d'%od][gc['subhalos']]

                if '_code' not in quant: r[field] = sP.units.codeMassToMsun( r[field] )
                if '_log' in quant: r[field] = logZeroNaN(r[field])

                # satellites given nan
                mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                mask[ gc['halos']['GroupFirstSub'] ] = 1
                wSat = np.where(mask == 0)
                r[field][wSat] = np.nan

            # subhalo mass [msun or log msun]
            if quant in ['mhalo_subfind','mhalo_subfind_log']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMass'])
                r[field] = sP.units.codeMassToMsun( gc )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # subhalo stellar mass (<30 pkpc definition, with auxCat) [msun or log msun]
            if quant in ['mstar_30pkpc','mstar_30pkpc_log']:
                acField = 'Subhalo_Mass_30pkpc_Stars'
                ac = auxCat(sP, fields=[acField])
                r[field] = sP.units.codeMassToMsun( ac[acField] )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # subhalo stellar mass (<1 or <2 rhalf definition, from groupcat) [msun or log msun]
            if quant in ['mstar1','mstar2','mstar1_log','mstar2_log']:
                field = 'SubhaloMassInRadType' if '2' in quant else 'SubhaloMassInHalfRadType'
                mass = groupCat(sP, fieldsSubhalos=[field])[:,sP.ptNum('stars')]
                r[field] = sP.units.codeMassToMsun(mass)

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # central flag (1 if central, 0 if not)
            if quant in ['central_flag','cen_flag','is_cen','is_central']:
                gc = groupCat(sP, fieldsHalos=['GroupFirstSub'])
                gc = gc[ np.where(gc >= 0) ]

                # satellites given zero
                r[field] = np.zeros( sP.numSubhalos, dtype='int16' )
                r[field][ gc ] = 1

            # isolated flag (1 if 'isolated' according to criterion, 0 if not, -1 if unprocessed)
            if 'isolated3d,' in quant:
                from cosmo.clustering import isolationCriterion3D

                # e.g. 'isolated3d,mstar_30pkpc,max,in_300pkpc'
                _, quant, max_type, dist = quant.split(',')
                dist = float( dist.split('in_')[1].split('pkpc')[0] )

                ic3d = isolationCriterion3D(sP, dist) #defaults: cenSatSelect='all', mstar30kpc_min=9.0

                r[field] = ic3d['flag_iso_%s_%s' % (quant,max_type)]

            # auxCat: photometric color
            if 'color_' in quant:
                from cosmo.color import loadColors

                r[field] = loadColors(sP, quant)

            # ssfr (1/yr or 1/Gyr) (SFR and Mstar both within 2r1/2stars) (optionally Mstar in 30pkpc)
            if quant in ['ssfr','ssfr_gyr','ssfr_30pkpc','ssfr_30pkpc_gyr',
                         'ssfr_log','ssfr_gyr_log','ssfr_30pkpc_log','ssfr_30pkpc_gyr_log']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
                mstar = sP.units.codeMassToMsun( gc['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

                # replace stellar masses with auxcat values within constant aperture, if requested
                if '_30pkpc' in quant:
                    mstar = groupCat(sP, fieldsSubhalos=['mstar_30pkpc'])

                # set mstar=0 subhalos to nan 
                w = np.where(mstar == 0.0)[0]
                if len(w):
                    mstar[w] = 1.0
                    gc['SubhaloSFRinRad'][w] = np.nan

                r[field] = gc['SubhaloSFRinRad'] / mstar

                if '_gyr' in quant: r[field] *= 1e9 # 1/yr to 1/Gyr
                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # virial radius (r200 or r500) of parent halo [code, pkpc, log pkpc]
            if quant in ['rhalo_200_code', 'rhalo_200','rhalo_200_log','rhalo_500','rhalo_500_log']:
                od = 200 if '_200' in quant else 500

                gc = groupCat(sP, fieldsHalos=['Group_R_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])
                r[field] = gc['halos']['Group_R_Crit%d'%od][gc['subhalos']]

                if '_code' not in quant: r[field] = sP.units.codeLengthToKpc( r[field] )
                if '_log' in quant: r[field] = logZeroNaN(r[field])

                # satellites given nan
                mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                mask[ gc['halos']['GroupFirstSub'] ] = 1
                wSat = np.where(mask == 0)
                r[field][wSat] = np.nan

            # virial velocity (v200) of parent halo [km/s]
            if quant in ['vhalo','v200','vhalo_log','v200_log']:
                gc = groupCat(sP, fieldsSubhalos=['mhalo_200_code','rhalo_200_code'])
                r[field] = sP.units.codeM200R200ToV200InKmS(gc['mhalo_200_code'], gc['rhalo_200_code'])

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # stellar half mass radii [code, pkpc, log pkpc]
            if quant in ['size_stars_code','size_stars','size_stars_log','rhalf_stars_code','rhalf_stars','rhalf_stars_log']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloHalfmassRadType'])
                r[field] = gc[:,sP.ptNum('stars')]

                if '_code' not in quant: r[field] = sP.units.codeLengthToKpc( r[field] )
                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # radial distance to parent halo [code, pkpc, log pkpc, r200frac] (centrals will have 0)
            if quant in ['rdist_code','rdist','rdist_log','rdist_rvir']:
                gc = groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], 
                                  fieldsSubhalos=['SubhaloPos','SubhaloGrNr'])

                parInds = gc['subhalos']['SubhaloGrNr']
                r[field] = sP.periodicDists( gc['halos']['GroupPos'][parInds,:], 
                                             gc['subhalos']['SubhaloPos'])

                if quant in ['rdist','rdist_log']:
                    r[field] = sP.units.codeLengthToKpc( r[field] )

                if '_rvir' in quant: r[field] /= gc['halos']['Group_R_Crit200'][parInds]
                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # virial temperature of parent halo
            if quant in ['tvir', 'tvir_log']:
                # get mass with self-call
                mass = groupCat(sP, fieldsSubhalos=['mhalo_200_code'])
                r[field] = sP.units.codeMassToVirTemp(mass)

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # save cache
            if field in r:
                sP.data[cacheKey] = r[field]

        if len(r) >= 1:
            # have at least one custom subhalo field, were halos also requested? not allowed
            assert fieldsHalos is None

            # do we also have standard fields requested? if so, load them now and combine
            if len(r) < len(fieldsSubhalos):
                standardFields = list(fieldsSubhalos)
                for key in r.keys():
                    standardFields.remove(key)
                gc = groupCat(sP, fieldsSubhalos=standardFields, sq=False)
                if isinstance(gc['subhalos'],np.ndarray):
                    assert len(standardFields) == 1
                    gc['subhalos'] = {standardFields[0]:gc['subhalos']} # pack into dictionary as expected
                gc['subhalos'].update(r)
                r = gc

            if sq and len(r) == 1:
                # compress and return single field (by default, unlike for standard fields)
                key = list(r.keys())[0]
                assert len(r.keys()) == 1
                return r[key]
            else:
                # return dictionary of fields (no 'subhalos' wrapping)
                if 'subhalos' in r: return r['subhalos']
                return r

    # override path function
    il.groupcat.gcPathOrig = il.groupcat.gcPath
    il.groupcat.gcPath = gcPath

    # IDs exist? either read or skip
    with h5py.File(gcPath(sP.simPath,sP.snap),'r') as f:
        if 'IDs' in f.keys() and len(f['IDs']):
            if readIDs:
                r['ids'] = f['IDs']['ID'][:]
            else:
                if not skipIDs:
                    print("Warning: readIDs not requested, but IDs present in group catalog!")

    # read
    r['header'] = il.groupcat.loadHeader(sP.simPath,sP.snap)

    if fieldsSubhalos is not None:
        # check cache
        fieldsSubhalos = list(fieldsSubhalos)
        r['subhalos'] = {}

        for field in fieldsSubhalos:
            cacheKey = 'gc_sub_%s' % field
            if cacheKey in sP.data:
                r['subhalos'][field] = sP.data[cacheKey]
                fieldsSubhalos.remove(field)

        # load
        if len(fieldsSubhalos):
            data = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=fieldsSubhalos)
            if isinstance(data,dict):
                r['subhalos'].update(data)
            else:
                assert isinstance(data,np.ndarray) and len(fieldsSubhalos) == 1
                r['subhalos'][fieldsSubhalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == 'illustris':
            for field in fieldsSubhalos:
                if 'Metallicity' in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig # set to new catalogs
                    print('Note: Overriding subhalo ['+field+'] with groups_ new catalog values.')
                    r['subhalos'][field] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath # restore

        for field in r['subhalos']: # cache
            sP.data['gc_sub_%s' % field] = r['subhalos'][field]

        key0 = list(r['subhalos'].keys())[0]
        if len(r['subhalos'].keys()) == 1 and key0 != 'count': # keep old behavior of il.groupcat.loadSubhalos()
            r['subhalos'] = r['subhalos'][key0]

    if fieldsHalos is not None:
        # check cache
        fieldsHalos = list(fieldsHalos)
        r['halos'] = {}

        for field in fieldsHalos:
            cacheKey = 'gc_halo_%s' % field
            if cacheKey in sP.data:
                r['halos'][field] = sP.data[cacheKey]
                fieldsHalos.remove(field)

        # load
        if len(fieldsHalos):
            data = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=fieldsHalos)
            if isinstance(data,dict):
                r['halos'].update(data)
            else:
                assert isinstance(data,np.ndarray) and len(fieldsHalos) == 1
                r['halos'][fieldsHalos[0]] = data

        # Illustris-1 metallicity fixes if needed
        if sP.run == 'illustris':
            for field in fieldsHalos:
                if 'Metallicity' in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig # set to new catalogs
                    print('Note: Overriding halo ['+field+'] with groups_ new catalog values.')
                    r['halos'][field] = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath # restore

        # override HDF5 datatypes if needed (GroupFirstSub unsigned -> signed for -1 entries)
        if isinstance(r['halos'],dict):
            if 'GroupFirstSub' in r['halos']:
                r['halos']['GroupFirstSub'] = r['halos']['GroupFirstSub'].astype('int32')
        else:
            if iterable(fieldsHalos)[0] == 'GroupFirstSub':
                assert len(iterable(fieldsHalos)) == 1
                r['halos'] = r['halos'].astype('int32')

        for field in r['halos']: # cache
            sP.data['gc_halo_%s' % field] = r['halos'][field]

        key0 = list(r['halos'].keys())[0]
        if len(r['halos'].keys()) == 1 and key0 != 'count': # keep old behavior of il.groupcat.loadHalos()
            r['halos'] = r['halos'][key0]

    if sq:
        # if possible: remove 'halos'/'subhalos' subdict, and field subdict
        if fieldsSubhalos is None: r = r['halos']
        if fieldsHalos is None: r = r['subhalos']

        if isinstance(r,dict) and len(r.keys()) == 1:
            r = r[ list(r.keys())[0] ]

    return r

def groupCatSingle(sP, haloID=None, subhaloID=None):
    """ Return complete group catalog information for one halo or subhalo. """
    assert haloID is None or subhaloID is None, "Cannot specify both haloID and subhaloID."
    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No groupCatSingle() for subbox snapshots."
        
    gcName = "Subhalo" if subhaloID is not None else "Group"
    gcID = subhaloID if subhaloID is not None else haloID
 
    # load groupcat offsets, calculate target file and offset
    groupFileOffsets = groupCatOffsetList(sP)['offsets'+gcName]
    groupFileOffsets = gcID - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]

    # load halo/subhalo fields into a dict
    r = {}
    
    with h5py.File(gcPath(sP.simPath,sP.snap,fileNum),'r') as f:
        for haloProp in f[gcName].keys():
            r[haloProp] = f[gcName][haloProp][groupOffset]
            
    return r

def groupCatHeader(sP, fileName=None):
    """ Load complete group catalog header. """
    if fileName is None:
        fileName = gcPath(sP.simPath, sP.snap)

    if fileName is None:
        return {'Ngroups_Total':0,'Nsubgroups_Total':0}

    with h5py.File(fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )

    return header

def groupCatHasField(sP, objType, field):
    """ True or False, does group catalog for objType=['Group','Subhalo'] have field? """
    with h5py.File(gcPath(sP.simPath,sP.snap),'r') as f:
        if objType in f and field in f[objType]:
            return True

    return False

def subboxVals(subbox):
    """ Return sbNum (integer) and sbStr1 and sbStr2 for use in locating subbox files. """
    sbNum = subbox if isinstance(subbox, int) else 0

    if subbox is not None:
        sbStr1 = 'subbox' + str(sbNum) + '_'
        sbStr2 = 'subbox' + str(sbNum) + '/'
    else:
        sbStr1 = ''
        sbStr2 = ''

    return sbNum, sbStr1, sbStr2

def snapPath(basePath, snapNum, chunkNum=0, subbox=None, checkExists=False):
    """ Find and return absolute path to a snapshot HDF5 file.
        Can be used to redefine illustris_python version (il.snapshot.snapPath = cosmo.load.snapPath). """
    sbNum, sbStr1, sbStr2 = subboxVals(subbox)
    ext = str(snapNum).zfill(3)

    # file naming possibilities
    fileNames = [ # standard: >1 file per snapshot, in subdirectory
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + \
                  '/snap_' + sbStr1 + ext + '.' + str(chunkNum) + '.hdf5',
                  # single file per snapshot
                  basePath + sbStr2 + 'snap_' + sbStr1 + ext + '.hdf5',
                  # single file per snapshot, in subdirectory (i.e. Millennium rewrite)
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + '/snap_' + sbStr1 + ext + '.hdf5',
                  # single groupordered file per snapshot
                  basePath + sbStr2 + 'snap-groupordered_' + ext + '.hdf5',
                  # multiple groupordered files
                  basePath + sbStr2 + 'snapdir_' + sbStr1 + ext + \
                  '/snap-groupordered_' + sbStr1 + ext + '.' + str(chunkNum) + '.hdf5',
                  # raw input (basePath actually contains a absolute path to a snapshot file already)
                  basePath
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    # failure:
    for fileName in fileNames:
        print(' '+fileName)
    raise Exception("ERROR: No snapshot found.")

def snapNumChunks(basePath, snapNum, subbox=None):
    """ Find number of file chunks in a snapshot, by checking for existence of files inside directory. """
    _, sbStr1, sbStr2 = subboxVals(subbox)
    path = basePath + sbStr2 + 'snapdir_' + sbStr1 + str(snapNum).zfill(3) + '/snap*.*.hdf5'
    nChunks = len(glob.glob(path))

    if nChunks == 0:
        nChunks = 1 # single file per snapshot

    return nChunks

def groupCatNumChunks(basePath, snapNum, subbox=None):
    """ Find number of file chunks in a group catalog. """
    _, sbStr1, sbStr2 = subboxVals(subbox)
    path = basePath + sbStr2 + 'groups_' + sbStr1 + str(snapNum).zfill(3)

    nChunks = len(glob.glob(path+'/fof_*.*.hdf5'))
    if nChunks == 0:
        # only if original 'fof_subhalo_tab' files are not present, then count 'groups' files instead
        nChunks += len(glob.glob(path+'/groups_*.*.hdf5'))

    if nChunks == 0:
        nChunks = 1 # single file per snapshot

    return nChunks

def snapshotHeader(sP, fileName=None):
    """ Load complete snapshot header. """
    if fileName is None:
        fileName = snapPath(sP.simPath, sP.snap, subbox=sP.subbox)

    with h5py.File(fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )

    # calculate and include NumPart_Total
    header['NumPart'] = il.snapshot.getNumPart(header)
    del header['NumPart_Total']
    del header['NumPart_Total_HighWord']

    return header

def snapHasField(sP, partType, field):
    """ True or False, does snapshot data for partType have field? """
    gName = 'PartType' + str(ptNum(partType))

    # the first chunk could not have the field but it could exist in a later chunk (e.g. sparse file 
    # contents of subboxes). to definitely return False, we have to check them all, but we can return 
    # an early True if we find a(ny) chunk containing the field
    for i in range(snapNumChunks(sP.simPath, sP.snap, subbox=sP.subbox)):
        fileName = snapPath(sP.simPath, sP.snap, chunkNum=i, subbox=sP.subbox)

        with h5py.File(fileName,'r') as f:
            if gName in f and field in f[gName]:
                return True

    return False

def snapConfigVars(sP):
    """ Load Config.sh flags and values as stored in the /Config/ group of modern snapshots. """
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file,'r') as f:
        if 'Config' in f:
            config = dict( f['Config'].attrs.items() )
        else:
            config = None

    return config

def snapParameterVars(sP):
    """ Load param.txt flags and values as stored in the /Parameters/ group of modern snapshots. """
    file = snapPath(sP.simPath, sP.snap, chunkNum=0, subbox=sP.subbox)

    with h5py.File(file,'r') as f:
        if 'Parameters' in f:
            params = dict( f['Parameters'].attrs.items() )
        else:
            params = None
            
    return params

def snapOffsetList(sP):
    """ Make the offset table (by type) for the snapshot files, to be able to quickly determine within 
        which file(s) a given offset+length will exist. """
    _, sbStr1, sbStr2 = subboxVals(sP.subbox)
    saveFilename = sP.derivPath + 'offsets/%ssnapshot_%s.hdf5' % (sbStr2,sP.snap)

    if not isdir(sP.derivPath+'offsets/%s' % sbStr2):
        makedirs(sP.derivPath+'offsets/%s' % sbStr2)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            snapOffsets = f['offsets'][()]
    else:
        nChunks = snapNumChunks(sP.simPath, sP.snap, sP.subbox)
        snapOffsets = np.zeros( (sP.nTypes, nChunks), dtype='int64' )

        for i in np.arange(1,nChunks+1):
            f = h5py.File( snapPath(sP.simPath,sP.snap,chunkNum=i-1,subbox=sP.subbox), 'r' )

            if i < nChunks:
                for j in range(sP.nTypes):
                    snapOffsets[j,i] = snapOffsets[j,i-1] + f['Header'].attrs['NumPart_ThisFile'][j]

                f.close()

        with h5py.File(saveFilename,'w') as f:
            f['offsets'] = snapOffsets
            print('Wrote: ' + saveFilename)

    return snapOffsets

def groupCatOffsetList(sP):
    """ Make the offset table for the group catalog files, to be able to quickly determine which
        which file a given group/subgroup number exists. """
    saveFilename = sP.derivPath + 'offsets/groupcat_' + str(sP.snap) + '.hdf5'

    if not isdir(sP.derivPath+'offsets'):
        mkdir(sP.derivPath+'offsets')

    r = {}

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            r['offsetsGroup']   = f['offsetsGroup'][()]
            r['offsetsSubhalo'] = f['offsetsSubhalo'][()]
    else:
        nChunks = groupCatNumChunks(sP.simPath, sP.snap)
        r['offsetsGroup']   = np.zeros( nChunks, dtype='int32' )
        r['offsetsSubhalo'] = np.zeros( nChunks, dtype='int32' )

        for i in np.arange(1,nChunks+1):
            f = h5py.File( gcPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )

            if i < nChunks:
                r['offsetsGroup'][i]   = r['offsetsGroup'][i-1]   + f['Header'].attrs['Ngroups_ThisFile']
                r['offsetsSubhalo'][i] = r['offsetsSubhalo'][i-1] + f['Header'].attrs['Nsubgroups_ThisFile']

                f.close()

        with h5py.File(saveFilename,'w') as f:
            f['offsetsGroup']   = r['offsetsGroup']
            f['offsetsSubhalo'] = r['offsetsSubhalo']
            print('Wrote: ' + saveFilename)

    return r

def groupCatOffsetListIntoSnap(sP):
    """ Make the offset table (by type) for every group/subgroup, such that the global location of 
        the members of any group/subgroup can be quickly located. """
    saveFilename = sP.derivPath + 'offsets/snap_groups_' + str(sP.snap) + '.hdf5'

    if not isdir(sP.derivPath+'offsets'):
        mkdir(sP.derivPath+'offsets')

    r = {}

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            r['snapOffsetsGroup']   = f['snapOffsetsGroup'][()]
            r['snapOffsetsSubhalo'] = f['snapOffsetsSubhalo'][()]
    else:
        # allocate
        with h5py.File( gcPath(sP.simPath,sP.snap), 'r' ) as f:
            totGroups    = f['Header'].attrs['Ngroups_Total']
            totSubGroups = f['Header'].attrs['Nsubgroups_Total']

        r['snapOffsetsGroup']   = np.zeros( (totGroups+1, sP.nTypes), dtype=np.int64 )
        r['snapOffsetsSubhalo'] = np.zeros( (totSubGroups+1, sP.nTypes), dtype=np.int64 )
        
        groupCount    = 0
        subgroupCount = 0
        
        # load following 3 fields across all chunks
        groupLenType    = np.zeros( (totGroups, sP.nTypes), dtype=np.int32 )
        groupNsubs      = np.zeros( (totGroups,), dtype=np.int32 )
        subgroupLenType = np.zeros( (totSubGroups, sP.nTypes), dtype=np.int32 )

        nChunks = groupCatNumChunks(sP.simPath, sP.snap)
        print('Calculating new groupCatOffsetsListIntoSnap... ['+str(nChunks)+' chunks]')

        for i in range(1,nChunks+1):
            # load header, get number of groups/subgroups in this file, and lengths
            f = h5py.File( gcPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )
            header = dict( f['Header'].attrs.items() )
            
            if header['Ngroups_ThisFile'] > 0:
                if 'GroupLenType' in f['Group']:
                    groupLenType[groupCount:groupCount+header['Ngroups_ThisFile']] = f['Group']['GroupLenType']
                else:
                    assert sP.targetGasMass == 0.0 # Millennium DMO with no types
                    groupLenType[groupCount:groupCount+header['Ngroups_ThisFile'],sP.ptNum('dm')] = f['Group']['GroupLen']

                groupNsubs[groupCount:groupCount+header['Ngroups_ThisFile']]   = f['Group']['GroupNsubs']
            if header['Nsubgroups_ThisFile'] > 0:
                if 'SubhaloLenType' in f['Subhalo']:
                    subgroupLenType[subgroupCount:subgroupCount+header['Nsubgroups_ThisFile']] = f['Subhalo']['SubhaloLenType']
                else:
                    assert sP.targetGasMass == 0.0 # Millennium DMO with no types
                    subgroupLenType[subgroupCount:subgroupCount+header['Nsubgroups_ThisFile'],sP.ptNum('dm')] = f['Subhalo']['SubhaloLen']
            
            groupCount += header['Ngroups_ThisFile']
            subgroupCount += header['Nsubgroups_ThisFile']
            
            f.close()
            
        # loop over each particle type, then over groups, calculate offsets from length
        for j in range(sP.nTypes):
            subgroupCount = 0
            
            # compute group offsets first
            r['snapOffsetsGroup'][1:,j] = np.cumsum( groupLenType[:,j] )
            
            for k in np.arange(totGroups):
                # subhalo offsets depend on group (to allow fuzz)
                if groupNsubs[k] > 0:
                    r['snapOffsetsSubhalo'][subgroupCount,j] = r['snapOffsetsGroup'][k,j]
                    
                    subgroupCount += 1
                    for m in np.arange(1, groupNsubs[k]):
                        r['snapOffsetsSubhalo'][subgroupCount,j] = \
                          r['snapOffsetsSubhalo'][subgroupCount-1,j] + subgroupLenType[subgroupCount-1,j]
                        subgroupCount += 1

        with h5py.File(saveFilename,'w') as f:
            f['snapOffsetsGroup']   = r['snapOffsetsGroup']
            f['snapOffsetsSubhalo'] = r['snapOffsetsSubhalo']
            print('Wrote: ' + saveFilename)

    return r    

def _haloOrSubhaloSubset(sP, haloID=None, subhaloID=None):
    """ Return the offset and length as a subset dict{} for a given haloID/subhaloID, as needed by il.snapshot.loadSubset(). """
    gcName = 'Group' if haloID is not None else 'Subhalo'
    gcID = haloID if haloID is not None else subhaloID

    subset = { 'snapOffsets' : snapOffsetList(sP) }

    # calculate target groups file chunk which contains this id
    groupFileOffsets = groupCatOffsetList(sP)['offsets'+gcName]
    groupFileOffsets = int(gcID) - groupFileOffsets
    fileNum = np.max( np.where(groupFileOffsets >= 0) )
    groupOffset = groupFileOffsets[fileNum]

    # load the length (by type) of this group/subgroup from the group catalog, and its offset within the snapshot
    with h5py.File(gcPath(sP.simPath,sP.snap,fileNum),'r') as f:
        if gcName+'LenType' in f[gcName]:
            subset['lenType'] = f[gcName][gcName+'LenType'][groupOffset,:]
        else:
            assert sP.targetGasMass == 0.0
            print('Warning: Should be DMO (Millennium) simulation with no LenType.')
            subset['lenType'] = np.zeros(sP.nTypes, dtype='int64')
            subset['lenType'][sP.ptNum('dm')] = f[gcName][gcName+'Len'][groupOffset]

        subset['offsetType'] = groupCatOffsetListIntoSnap(sP)['snapOffsets'+gcName][gcID,:]

    return subset

def _ionLoadHelper(sP, partType, field, kwargs):
    """ Helper to load (with particle level caching) ionization fraction, or total ion mass, 
    values values for gas cells. Or, total line flux for emission. """

    if 'flux' in field:
        lineName, prop = field.rsplit(" ",1)
        lineName = lineName.replace("-"," ") # e.g. "O--8-16.0067A" -> "O  8 16.0067A"
    else:
        element, ionNum, prop = field.split() # e.g. "O VI mass" or "Mg II frac"

    # use cold-phase temperature for SFR>0 instead of eEOS temperature?
    tempSfCold = False
    if '_sfcold' in prop:
        prop = prop.replace('_sfcold','')
        tempSfCold = True

    assert sP.isPartType(partType, 'gas')
    assert prop in ['mass','frac','flux']

    # indRange subset
    indRangeOrig = kwargs['indRange']

    # haloID or subhaloID subset
    if kwargs['haloID'] is not None or kwargs['subhaloID'] is not None:
        assert indRangeOrig is None
        subset = _haloOrSubhaloSubset(sP, haloID=kwargs['haloID'], subhaloID=kwargs['subhaloID'])
        offset = subset['offsetType'][sP.ptNum(partType)]
        length = subset['lenType'][sP.ptNum(partType)]
        indRangeOrig = [offset, offset+length-1] # inclusive below

    # check memory cache (only simplest support at present, for indRange returns of global cache)
    if indRangeOrig is not None:
        cache_key = 'snap%d_%s_%s' % (sP.snap,partType,field.replace(" ","_"))
        if cache_key in sP.data:
            print('NOTE: Returning [%s] from cache, indRange [%d - %d]!' % \
                (cache_key,indRangeOrig[0],indRangeOrig[1]))
            return sP.data[cache_key][indRangeOrig[0]:indRangeOrig[1]+1]

    # full snapshot-level caching, create during normal usage but not web (always use if exists)
    useCache = True
    createCache = True if getuser() == 'dnelson' else False

    cachePath = sP.derivPath + 'cache/'
    cacheFile = cachePath + 'cached_%s_%s_%d.hdf5' % (partType,field.replace(" ","-"),sP.snap)
    indRangeAll = [0, snapshotHeader(sP)['NumPart'][sP.ptNum(partType)] ]

    if useCache:
        # does not exist yet, and should create?
        if createCache and not isfile(cacheFile):
            if not isdir(cachePath):
                mkdir(cachePath)
            # compute for indRange == None (whole snapshot) with a reasonable pSplit
            nChunks = numPartToChunkLoadSize(indRangeAll[1])
            print('Creating [%s] for [%d] particles in [%d] chunks.' % \
                (cacheFile.split(sP.derivPath)[1], indRangeAll[1], nChunks) )

            # create file and init ionization calculator
            with h5py.File(cacheFile, 'w') as f:
                dset = f.create_dataset('field', (indRangeAll[1],), dtype='float32')

            from cosmo.cloudy import cloudyIon, cloudyEmission
            if prop in ['mass','frac']:
                ion = cloudyIon(sP, el=element, redshiftInterp=True)
            else:
                emis = cloudyEmission(sP, line=lineName, redshiftInterp=True)
                wavelength = emis.lineWavelength(lineName)

            # process chunked
            for i in range(nChunks):
                indRangeLocal = pSplitRange(indRangeAll, nChunks, i)

                # indRange is inclusive for snapshotSubset(), so skip saving the very last 
                # element, which is included in the next return of pSplitRange()
                indRangeLocal[1] -= 1
                kwargs['indRange'] = indRangeLocal

                if indRangeLocal[0] == indRangeLocal[1]:
                    continue # we are done

                if prop in ['mass','frac']:
                    # either ionization fractions, or total mass in the ion
                    values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeLocal, tempSfCold=tempSfCold)
                    if prop == 'mass':
                        values *= sP.snapshotSubset(partType, 'Masses', **kwargs)
                else:
                    # emission flux
                    lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeLocal, tempSfCold=tempSfCold)
                    values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2] @ sP.redshift

                with h5py.File(cacheFile) as f:
                    f['field'][indRangeLocal[0]:indRangeLocal[1]+1] = values

                print(' [%2d] saved %d - %d' % (i,indRangeLocal[0],indRangeLocal[1]))
            print('Saved: [%s].' % cacheFile.split(sP.derivPath)[1])
            kwargs['indRange'] = indRangeOrig # restore

        # load from existing cache if it exists
        if isfile(cacheFile):
            if getuser() == 'dnelson':
                print('Loading [%s] [%s] from [%s].' % (partType,field,cacheFile.split(sP.derivPath)[1]))

            with h5py.File(cacheFile, 'r') as f:
                assert f['field'].size == indRangeAll[1]
                if indRangeOrig is None:
                    values = f['field'][()]
                else:
                    values = f['field'][indRangeOrig[0] : indRangeOrig[1]+1]
    
    if not useCache or not isfile(cacheFile):
        # don't use cache, or tried to use and it doesn't exist yet, so run computation now
        from cosmo.cloudy import cloudyIon, cloudyEmission
        if prop in ['mass','frac']:
            ion = cloudyIon(sP, el=element, redshiftInterp=True)
        else:
            emis = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            wavelength = emis.lineWavelength(lineName)

        if prop in ['mass','frac']:
            # either ionization fractions, or total mass in the ion
            values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeOrig, tempSfCold=tempSfCold)
            if prop == 'mass':
                values *= sP.snapshotSubset(partType, 'Masses', **kwargs)
        else:
            # emission flux
            lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeOrig)
            values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2]

    return values

def snapshotSubset(sP, partType, fields, 
                   inds=None, indRange=None, haloID=None, subhaloID=None, 
                   mdi=None, sq=True, haloSubset=False, float32=False):
    """ For a given snapshot load one or more field(s) for one particle type
          partType = e.g. [0,1,2,4] or ('gas','dm','tracer','stars')
          fields   = e.g. ['ParticleIDs','Coordinates','temp',...]

          the following four arguments are optional, but at most one can be specified:
            * inds      : known indices requested, optimize the load
            * indRange  : same, but specify only min and max indices (inclusive)
            * haloID    : if input, load particles only of the specified fof halo
            * subhaloID : if input, load particles only of the specified subalo
          mdi : multi-dimensional index slice load (only used in recursive calls, don't input directly)
          sq  : squeeze single field return into a numpy array instead of within a dict
          haloSubset : return particle subset of only those in all FoF halos (no outer fuzz)
          float32 : load any float64 datatype datasets directly as float32 (optimize for memory)
    """
    kwargs = {'inds':inds, 'indRange':indRange, 'haloID':haloID, 'subhaloID':subhaloID}
    subset = None

    if (inds is not None or indRange is not None) and (haloID is not None or subhaloID is not None):
        raise Exception("Can only specify one of (inds,indRange,haloID,subhaloID).")
    if inds is not None and indRange is not None:
        raise Exception("Cannot specify both inds and indRange.")
    if haloID is not None and subhaloID is not None:
        raise Exception("Cannot specify both haloID and subhaloID.")
    if ((haloID is not None) or (subhaloID is not None)) and not sP.groupOrdered:
        raise Exception("Not yet implemented (group/halo load in non-groupordered).")
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (not sP.groupOrdered or (haloID is not None) or \
        (subhaloID is not None) or (inds is not None) or (indRange is not None)):
        raise Exception("haloSubset only for groupordered snapshots, and not with halo/subhalo subset.")
    if sP.snap is None:
        raise Exception("Must specify sP.snap for snapshotSubset load.")

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)

    # make sure fields is not a single element, and don't modify input
    fields = list(iterable(fields))
    fieldsOrig = list(iterable(fields))

    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        indRange = [0, offsets_pt[:,sP.ptNum(partType)].max()]
        kwargs['indRange'] = indRange

    # derived particle types (i.e. subsets of snapshot PartTypeN's)
    if '_' in str(partType):
        ptSnap = partType.split('_')[0]

        # load needed fields to define subset, and set w_sel
        if partType in ['star_real','stars_real']:
            sftime = snapshotSubset(sP, ptSnap, 'sftime', **kwargs)
            w_sel = np.where(sftime >= 0.0)
        if partType == 'wind_real':
            sftime = snapshotSubset(sP, ptSnap, 'sftime', **kwargs)
            w_sel = np.where(sftime < 0.0)
        if partType in ['gas_sf','gas_eos']:
            sfr = snapshotSubset(sP, ptSnap, 'sfr', **kwargs)
            w_sel = np.where(sfr > 0.0)
        if partType == 'gas_nonsf':
            sfr = snapshotSubset(sP, ptSnap, 'sfr', **kwargs)
            w_sel = np.where(sfr == 0.0)

        # load requested field(s), take subset and return
        ret = snapshotSubset(sP, ptSnap, fields, **kwargs)

        if isinstance(ret,dict):
            for key in ret.keys():
                ret[key] = ret[key][w_sel]
            return ret
        return ret[w_sel] # single ndarray

    # composite and derived fields (temp, vmag, ...), unit conversions (bmag_uG, ...), and custom analysis (ionic masses, ...)
    r = {}

    for i,field in enumerate(fields):
        # temperature (from u,nelec) [log K]
        if field.lower() in ["temp", "temperature"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=True)

        # temperature (uncorrected values for TNG runs) [log K]
        if field.lower() in ["temp_old"]:
            u  = snapshotSubset(sP, partType, 'InternalEnergyOld', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=True)

        # temperature (from u,nelec) [log K] (where star forming gas is set to the cold-phase temperature instead of eEOS temperature)
        if field.lower() in ["temp_sfcold"]:
            r[field] = snapshotSubset(sP, partType, 'temp', **kwargs)
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            w = np.where(sfr > 0.0)
            r[field][w] = 3.0 # fiducial Illustris/TNG model: T_clouds = 1000 K, T_SN = 5.73e7 K

        # temperature (from u,nelec) [linear K]
        if field.lower() in ["temp_linear"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.UToTemp(u,ne,log=False)

        # hydrogen number density (from rho) [linear 1/cm^3]
        if field.lower() in ["nh","hdens"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True,numDens=True)
            r[field] = sP.units.hydrogen_massfrac * dens # constant 0.76 assumed

        # number density (from rho) [linear 1/cm^3]
        if field.lower() in ["numdens"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.codeDensToPhys(dens,cgs=True,numDens=True)

        # mass density to critical baryon density [linear dimensionless]
        if field.lower() in ['dens_critratio','dens_critb']:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.codeDensToCritRatio(dens, baryon=('_critb' in field.lower()), log=False)

        # particle/cell mass [linear solar masses]
        if field.lower() in ['mass_msun']:
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)
            r[field] = sP.units.codeMassToMsun(mass)

        # catch DM particle mass request [code units]
        if field.lower() in ['mass','masses'] and sP.isPartType(partType,'dm'):
            dummy = snapshotSubset(sP, partType, 'pos_x', **kwargs)
            r[field] = dummy*0.0 + sP.snapshotHeader()['MassTable'][sP.ptNum('dm')]

        # entropy (from u,dens) [log cgs] == [log K cm^2]
        if field.lower() in ["ent", "entr", "entropy"]:
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            r[field] = sP.units.calcEntropyCGS(u,dens,log=True)

        # velmag (from 3d velocity) [physical km/s]
        if field.lower() in ["vmag", "velmag"]:
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            vel = sP.units.particleCodeVelocityToKms(vel)
            r[field] = np.sqrt( vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2] )

        # Bmag (from vector field) [physical Gauss]
        if field.lower() in ["bmag", "bfieldmag"]:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            b = sP.units.particleCodeBFieldToGauss(b)
            bmag = np.sqrt( b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2] )
            r[field] = bmag

        # Bmag in micro-Gauss [physical uG]
        if field.lower() in ['bmag_ug', 'bfieldmag_ug']:
            r[field] = snapshotSubset(sP, partType, 'bmag', **kwargs) * 1e6

        # scalar B^2 (from vector field) [code units]
        if field.lower() in ["b2","bsq"]:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            r[field] = ( b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2] )

        # Alfven velocity magnitude (of electron plasma) [physical km/s]
        if field.lower() in ["vmag_alfven", "velmag_alfven"]:
            assert 0 # todo

        # volume [ckpc/h]^3
        if field.lower() in ["vol", "volume"]:
            # Volume eliminated in newer outputs, calculate as necessary, otherwise fall through
            if not snapHasField(sP, partType, 'Volume'):
                mass = snapshotSubset(sP, partType, 'mass', **kwargs)
                dens = snapshotSubset(sP, partType, 'dens', **kwargs)
                r[field] = (mass / dens)

        # volume in physical [cm^3] or [kpc^3]
        if field.lower() in ['vol_cm3','volume_cm3']:
            r[field] = sP.units.codeVolumeToCm3( snapshotSubset(sP, partType, 'volume', **kwargs) )
        if field.lower() in ['vol_kpc3','volume_kpc3']:
            r[field] = sP.units.codeVolumeToKpc3( snapshotSubset(sP, partType, 'volume', **kwargs) )

        # cellsize (from volume) [ckpc/h]
        if field.lower() in ["cellsize", "cellrad"]:
            vol = snapshotSubset(sP, partType, 'volume', **kwargs)                
            r[field] = (vol * 3.0 / (4*np.pi))**(1.0/3.0)

        # cellsize [physical kpc]
        if field.lower() in ["cellsize_kpc","cellrad_kpc"]:
            cellsize_code = snapshotSubset(sP, partType, 'cellsize', **kwargs)
            r[field] = sP.units.codeLengthToKpc(cellsize_code)

        # particle volume (from subfind hsml of N nearest DM particles) [ckpc/h]
        if field.lower() in ["subfind_vol","subfind_volume"]:
            hsml = snapshotSubset(sP, partType, 'SubfindHsml', **kwargs)
            r[field] = (4.0/3.0) * np.pi * hsml**3.0

        # metallicity in linear(solar) units
        if field.lower() in ["metal_solar","z_solar"]:
            metal = snapshotSubset(sP, partType, 'metal', **kwargs) # metal mass / total mass ratio
            r[field] = sP.units.metallicityInSolar(metal,log=False)

        # stellar age in Gyr (convert GFM_StellarFormationTime scalefactor)
        if field.lower() in ["star_age","stellar_age"]:
            curUniverseAgeGyr = sP.units.redshiftToAgeFlat(sP.redshift)
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            birthRedshift = 1.0/birthTime - 1.0
            r[field] = curUniverseAgeGyr - sP.units.redshiftToAgeFlat(birthRedshift)

        # formation redshift (convert GFM_StellarFormationTime scalefactor)
        if field.lower() in ["z_formation","z_form"]:
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            r[field] = 1.0/birthTime - 1.0

        # pressure_ratio (linear ratio of magnetic to gas pressure)
        if field.lower() in ['pres_ratio','pressure_ratio']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            b    = snapshotSubset(sP, partType, 'MagneticField', **kwargs)

            P_gas = sP.units.calcPressureCGS(u, dens)
            P_B   = sP.units.calcMagneticPressureCGS(b)
            r[field] = P_B/P_gas

        # u_B_ke_ratio (linear ratio of magnetic to kinetic energy density)
        if field.lower() in ['u_b_ke_ratio','magnetic_kinetic_edens_ratio','b_ke_edens_ratio']:
            u_b = snapshotSubset(sP, partType, 'p_b', **kwargs) # [log K/cm^3]
            u_ke = snapshotSubset(sP, partType, 'u_ke', **kwargs) # [log K/cm^3]
            r[field] = 10.0**u_b / 10.0**u_ke

        # gas pressure [log K/cm^3]
        if field.lower() in ['gas_pres','gas_pressure','p_gas']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            r[field] = sP.units.calcPressureCGS(u, dens, log=True)

        if field.lower() in ['p_gas_linear']:
            r[field] = 10.0**snapshotSubset(sP, partType, 'p_gas', **kwargs)

        # magnetic pressure [log K/cm^3]
        if field.lower() in ['mag_pres','magnetic_pressure','p_b','p_magnetic']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            r[field] = sP.units.calcMagneticPressureCGS(b, log=True)

        if field.lower() in ['p_b_linear']:
            r[field] = 10.0**snapshotSubset(sP, partType, 'p_b', **kwargs)

        # kinetic energy density [log erg/cm^3]
        if field.lower() in ['kinetic_energydens','kinetic_edens','u_ke']:
            dens_code = snapshotSubset(sP, partType, 'Density', **kwargs)
            vel_kms = snapshotSubset(sP, partType, 'velmag', **kwargs)
            r[field] = sP.units.calcKineticEnergyDensityCGS(dens_code, vel_kms, log=True)

        # total pressure, magnetic plus gas [K/cm^3]
        if field.lower() in ['p_tot','pres_tot','pres_total','pressure_tot','pressure_total']:
            P_B = 10.0**snapshotSubset(sP, partType, 'P_B', **kwargs)
            P_gas = 10.0**snapshotSubset(sP, partType, 'P_gas', **kwargs)
            r[field] = ( P_B + P_gas )

        # escape velocity (based on Potential field) [physical km/s]
        if field.lower() in ['vesc','escapevel']:
            pot = snapshotSubset(sP, partType, 'Potential', **kwargs)
            r[field] = sP.units.codePotentialToEscapeVelKms(pot)

        # ------------------------------------------------------------------------------------------------------

        # blackhole bolometric luminosity [erg/s] (model-dependent)
        if field.lower() in ['bh_lbol','bh_bollum']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            r[field] = sP.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot)

        if field.lower() in ['bh_lbol_basic','bh_bollum_basic']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            r[field] = sP.units.codeBHMassMdotToBolLum(bh_mass, bh_mdot, basic_model=True)

        # blackhole eddington ratio [dimensionless linear]
        if field.lower() in ['lambda_edd','edd_ratio','eddington_ratio','bh_eddratio']:
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            bh_mdot_edd = snapshotSubset(sP, partType, 'BH_MdotEddington', **kwargs)
            r[field] = (bh_mdot / bh_mdot_edd) # = (lum_bol / lum_edd)

        # blackhole eddington luminosity [erg/s linear]
        if field.lower() in ['ledd','bh_ledd','lumedd','bh_lumedd','eddington_lum']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            r[field] = sP.units.codeBHMassToLumEdd(bh_mass)

        # blackhole feedback energy injection rate [erg/s linear]
        if field.lower() in ['bh_dedt','bh_edot']:
            bh_mass = snapshotSubset(sP, partType, 'BH_Mass', **kwargs)
            bh_mdot = snapshotSubset(sP, partType, 'BH_Mdot', **kwargs)
            bh_mdot_bondi = snapshotSubset(sP, partType, 'BH_MdotBondi', **kwargs)
            bh_mdot_edd = snapshotSubset(sP, partType, 'BH_MdotEddington', **kwargs)
            bh_density  = snapshotSubset(sP, partType, 'BH_Density', **kwargs)

            r[field] = sP.units.codeBHMassMdotToInstantaneousEnergy(bh_mass, bh_mdot, bh_density, bh_mdot_bondi, bh_mdot_edd)

        # wind model (gas cells): feedback energy injection rate [10^51 erg/s linear]
        if field.lower() in ['wind_dedt','wind_edot','sn_dedt','sn_edot','sf_dedt','sf_edot']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)

            r[field] = sP.units.codeSfrZToWindEnergyRate(sfr, metal)

        # wind model (gas cells): feedback momentum injection rate [10^51 g*cm/s^2 linear]
        if field.lower() in ['wind_dpdt','wind_pdot','sn_dpdt','sn_pdot','sf_dpdt','sf_pdot']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)

            r[field] = sP.units.codeSfrZToWindMomentumRate(sfr, metal, dm_veldisp)

        # wind model (gas cells): launch velocity [km/s]
        if field.lower() in ['wind_vel']:
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)
            r[field] = sP.units.sigmaDMToWindVel(dm_veldisp)

        # wind model (gas cells): mass loading factor [linear dimensionless]
        if field.lower() in ['wind_eta','wind_etam','wind_massloading']:
            sfr = snapshotSubset(sP, partType, 'sfr', **kwargs)
            metal = snapshotSubset(sP, partType, 'metallicity', **kwargs)
            dm_veldisp = snapshotSubset(sP, partType, 'SubfindVelDisp', **kwargs)

            r[field] = sP.units.codeSfrZSigmaDMToWindMassLoading(sfr, metal, dm_veldisp)

        # ------------------------------------------------------------------------------------------------------

        # synchrotron power (simple model) [W/Hz]
        if field.lower() in ['p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            vol = snapshotSubset(sP, partType, 'Volume', **kwargs)

            modelArgs = {}
            if '_ska' in field: modelArgs['telescope'] = 'SKA'
            if '_vla' in field: modelArgs['telescope'] = 'VLA'
            if '_eta43' in field: modelArgs['eta'] = (4.0/3.0)
            if '_alpha15' in field: modelArgs['alpha'] = 1.5
            r[field] = sP.units.synchrotronPowerPerFreq(b, vol, watts_per_hz=True, log=False, **modelArgs)

        # bolometric x-ray luminosity (simple model) [10^30 erg/s]
        if field.lower() in ['xray_lum','xray']:
            sfr  = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            mass = snapshotSubset(sP, partType, 'Masses', **kwargs)
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            ne   = snapshotSubset(sP, partType, 'ne', **kwargs)
            r[field] = sP.units.calcXrayLumBolometric(sfr, u, ne, mass, dens)

        # h-alpha line luminosity (simple model: linear conversion from SFR) [10^30 erg/s]
        if field.lower() in ['halpha_lum','halpha','sfr_halpha']:
            sfr  = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            r[field] = sP.units.sfrToHalphaLuminosity(sfr)

        # 850 micron submilliter flux (simple model) [linear mJy]
        if field.lower() in ['s850um_flux', 'submm_flux', 's850um_flux_ismcut', 'submm_flux_ismcut']:
            sfr = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            metalmass = snapshotSubset(sP, partType, 'metalmass_msun', **kwargs)
            if '_ismcut' in field.lower():
                temp = snapshotSubset(sP, partType, 'temp', **kwargs)
                dens = snapshotSubset(sP, partType, 'Density', **kwargs)
                ismCut = True
            else:
                temp = None
                dens = None
                ismCut = False

            r[field] = sP.units.gasSfrMetalMassToS850Flux(sfr, metalmass, temp, dens, ismCut=ismCut)

        # cloudy based ionic mass (or emission flux) calculation, if field name has a space in it
        if " " in field:
            # hydrogen model mass calculation (todo: generalize to different molecular models)
            if field.lower() in ['h i mass', 'hi mass', 'himass', 'h1mass', 'hi_mass']:
                assert haloID is None and subhaloID is None # otherwise handle, construct indRange
                from cosmo.hydrogen import hydrogenMass
                r[field] = hydrogenMass(None, sP, atomic=True, indRange=indRange)

            elif field.lower() in ['h 2 mass', 'h2 mass', 'h2mass'] or 'h2mass_' in field.lower():
                # todo: we are inside the (" " in field) block, will never catch
                assert haloID is None and subhaloID is None # otherwise handle, construct indRange
                from cosmo.hydrogen import hydrogenMass
                if 'h2mass_' in field.lower():
                    molecularModel = field.lower().split('_')[1]
                else:
                    molecularModel = 'BL06'
                    print('Warning: using [%s] model for H2 by default since unspecified.' % molecularModel)

                r[field] = hydrogenMass(None, sP, molecular=molecularModel, indRange=indRange)

            else:
                # cloudy-based calculation
                r[field] = _ionLoadHelper(sP, partType, field, kwargs)

        # pre-computed H2/other particle-level data
        if '_popping' in field.lower():
            # use Popping+2019 pre-computed results in 'hydrogen' postprocessing catalog
            # e.g. 'MH2BR_popping', 'MH2GK_popping', 'MH2KMT_popping', 'MHIBR_popping', 'MHIGK_popping', 'MHIKMT_popping'
            assert haloID is None and subhaloID is None # otherwise handle, construct indRange
            
            path = sP.postPath + 'hydrogen/gas_%03d.hdf5' % sP.snap
            key = field.split('_popping')[0]

            with h5py.File(path,'r') as f:
                if indRange is None:
                    r[field] = f[key][()]
                else:
                    r[field] = f[key][indRange[0]:indRange[1]+1]

        if '_diemer' in field.lower():
            # use Diemer+2019 pre-computed results in 'hydrogen' postprocessing catalog
            # e.g. 'MH2_GD14_diemer', 'MH2_GK11_diemer', 'MH2_K13_diemer', 'MH2_S14_diemer'
            # or 'MHI_GD14_diemer', 'MHI_GK11_diemer', 'MHI_K13_diemer', 'MHI_S14_diemer'
            assert haloID is None and subhaloID is None # otherwise handle, construct indRange
            
            path = sP.postPath + 'hydrogen/diemer_%03d.hdf5' % sP.snap
            key = 'f_mol_' + field.split('_')[1]

            with h5py.File(path,'r') as f:
                if indRange is None:
                    f_mol = f[key][()]
                    f_neutral_H = f['f_neutral_H'][()]
                else:
                    f_mol = f[key][indRange[0]:indRange[1]+1]
                    f_neutral_H = f['f_neutral_H'][indRange[0]:indRange[1]+1]

            # file contains f_mol, for M_H2 = Mass_gas * f_neutral_H * f_mol, while for M_HI = MasS_gas * f_neutral_H * (1-f_mol)
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)

            if 'mh2_' in field.lower():
                r[field] = mass * f_neutral_H * f_mol
            if 'mhi_' in field.lower():
                r[field] = mass * f_neutral_H * (1.0 - f_mol)

        if '_ionmassratio' in field:
            # per-cell ratio between two ionic masses, e.g. "O6_O8_ionmassratio"
            from cosmo.cloudy import cloudyIon
            ion = cloudyIon(sP=None)
            ion1, ion2, _ = field.split('_')

            mass1 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion1), **kwargs)
            mass2 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion2), **kwargs)
            r[field] = ( mass1 / mass2 )

        if '_numratio' in field:
            # metal number density ratio e.g. "Si_H_numratio", relative to solar, [Si/H] = log(n_Si/n_H)_cell - log(n_Si/n_H)_solar
            from cosmo.cloudy import cloudyIon
            el1, el2, _ = field.split('_')

            ion = cloudyIon(sP=None)
            el1_massratio = snapshotSubset(sP, partType, 'metals_'+el1, **kwargs)
            el2_massratio = snapshotSubset(sP, partType, 'metals_'+el2, **kwargs)
            el_ratio = el1_massratio / el2_massratio

            r[field] = ion._massRatioToRelSolarNumDensRatio(el_ratio, el1, el2)

        if '_massratio' in field:
            # metal mass ratio e.g. "Si_H_massratio", absolute linear (not relative to solar)
            el1, el2, _ = field.split('_')

            el1_massratio = snapshotSubset(sP, partType, 'metals_'+el1, **kwargs)
            el2_massratio = snapshotSubset(sP, partType, 'metals_'+el2, **kwargs)
            r[field] = (el1_massratio / el2_massratio)

        # metal mass (total or by species): convert fractions to masses [code units] or [msun]
        if "metalmass" in field.lower():
            assert sP.isPartType(partType, 'gas') or sP.isPartType(partType, 'stars')

            solarUnits = False
            if "msun" in field.lower(): # e.g. "metalmass_msun" or "metalmass_He_msun"
                solarUnits = True
                field = field.split('_', 1)[0]

            fracFieldName = "metal" # e.g. "metalmass" = total metal mass
            if "_" in field: # e.g. "metalmass_O" or "metalmass_Mg"
                fracFieldName = "metals_" + field.split("_")[1].capitalize()

            masses = snapshotSubset(sP, partType, 'Masses', **kwargs)
            masses *= snapshotSubset(sP, partType, fracFieldName, **kwargs)
            if solarUnits:
                masses = sP.units.codeMassToMsun(masses)

            r[field] = masses

        # metal mass density [linear g/cm^3]
        if "metaldens" in field.lower():
            fracFieldName = "metal" # e.g. "metaldens" = total metal mass density
            if "_" in field: # e.g. "metaldens_O" or "metaldens_Mg"
                fracFieldName = "metals_" + field.split("_")[1].capitalize()

            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True)
            r[field] = dens * snapshotSubset(sP, partType, fracFieldName, **kwargs)

        # gravitational potential, in linear [(km/s)^2]
        if field.lower() in ['gravpot','gravpotential']:
            pot = snapshotSubset(sP, partType, 'Potential', **kwargs)
            r[field] = pot * sP.units.scalefac

        # GFM_MetalsTagged: ratio of iron mass [linear] produced in SNIa versus SNII
        if field.lower() in ['sn_iaii_ratio_fe']:
            metals_FeSNIa = snapshotSubset(sP, partType, 'metals_FeSNIa', **kwargs)
            metals_FeSNII = snapshotSubset(sP, partType, 'metals_FeSNII', **kwargs)
            r[field] = ( metals_FeSNIa / metals_FeSNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus SNII
        if field.lower() in ['sn_iaii_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_SNII = snapshotSubset(sP, partType, 'metals_SNII', **kwargs)
            r[field] = ( metals_SNIa / metals_SNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus AGB stars
        if field.lower() in ['sn_ia_agb_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_AGB = snapshotSubset(sP, partType, 'metals_AGB', **kwargs)
            r[field] = ( metals_SNIa / metals_AGB )

        # sound speed (hydro only version) [physical km/s]
        if field.lower() in ['csnd','soundspeed']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            r[field] = sP.units.calcSoundSpeedKmS(u,dens)

        # cooling time (computed from saved GFM_CoolingRate) [Gyr]
        if field.lower() in ['tcool','cooltime']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            r[field] = sP.units.coolingTimeGyr(dens, coolrate, u)

        # cooling rate, specific (computed from saved GFM_CoolingRate) [erg/s/g]
        if field.lower() in ['coolrate','coolingrate']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            coolheat = sP.units.coolingRateToCGS(dens, coolrate)
            w = np.where(coolheat >= 0.0)
            coolheat[w] = np.nan # cooling only
            r[field] = -1.0 * coolheat # positive

        # cooling rate, specific (computed from saved GFM_CoolingRate) [erg/s/g]
        if field.lower() in ['heatrate','heatingrate']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            coolheat = sP.units.coolingRateToCGS(dens, coolrate)
            w = np.where(coolheat <= 0.0)
            coolheat[w] = np.nan # heating only, positive
            r[field] = coolheat

        # 'cooling rate' of Powell source term, specific (computed from saved DivB, GFM_CoolingRate) [erg/s/g]
        if field.lower() in ['coolrate_powell']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            divb = snapshotSubset(sP, partType, 'MagneticFieldDivergence', **kwargs)
            bfield = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            vel = snapshotSubset(sP, partType, 'Velocities', **kwargs)
            vol = snapshotSubset(sP, partType, 'Volume', **kwargs)
            coolheat = sP.units.powellEnergyTermCGS(dens, divb, bfield, vel, vol)
            w = np.where(coolheat >= 0.0)
            coolheat[w] = np.nan # cooling only
            r[field] = -1.0 * coolheat # positive

        # ratio of the above two terms, only gas with powell<0 (cooling) and coolrate>0 (heating)
        if field.lower() in ['coolrate_ratio']:
            heatrate = snapshotSubset(sP, partType, 'heatrate', **kwargs)
            powell = snapshotSubset(sP, partType, 'coolrate_powell', **kwargs)
            r[field] = powell / heatrate # positive by definition

        # total effective timestep, from snapshot [years]
        if field.lower() == 'dt_yr':
            dt = snapshotSubset(sP, 'gas', 'TimeStep', **kwargs)
            r[field] = sP.units.codeTimeStepToYears(dt)

        # gas cell hydrodynamical timestep [years]
        if field.lower() == 'dt_hydro_yr':
            soundspeed = snapshotSubset(sP, 'gas', 'soundspeed', **kwargs)
            cellrad = snapshotSubset(sP, 'gas', 'cellrad', **kwargs)
            cellrad_kpc = sP.units.codeLengthToKpc(cellrad)
            cellrad_km  = cellrad_kpc * sP.units.kpc_in_km

            dt_hydro_s = sP.units.CourantFac * cellrad_km / soundspeed
            dt_yr = dt_hydro_s / sP.units.s_in_yr
            r[field] = dt_yr

        # ratio of (cell mass / sfr / timestep), either hydro-only or actual timestep [dimensionless linear ratio]
        if field.lower() in ['mass_sfr_dt','mass_sfr_dt_hydro']:
            mass = snapshotSubset(sP, 'gas', 'mass', **kwargs)
            mass = sP.units.codeMassToMsun(mass)
            sfr  = snapshotSubset(sP, 'gas', 'sfr', **kwargs)

            dt_type = 'dt_hydro_yr' if '_hydro' in field.lower() else 'dt_yr'
            dt = snapshotSubset(sP, 'gas', dt_type, **kwargs)

            r[field] = (mass / sfr / dt)

        # ------------------------------------------------------------------------------------------------------
        # halo centric analysis (currently require one explicit haloID/suhaloID)
        # (TODO: generalize to e.g. full snapshots)
        # should be able to do this with an inverse mapping of indRange to subhaloIDs (check the case of indRange==None 
        # mapping correctly to all). Then, replace groupCatSingle()'s with global groupCat() loads, and loop over 
        # each subhaloID, for each call the appropriate unit function with the group/subhalo particle subset and obj 
        # position. note: decision for satellite subhalos, properties are relative to themselves or to their central.
        # ------------------------------------------------------------------------------------------------------

        # 3D xyz position, relative to halo/subhalo center, [code] or [physical kpc] of [dimensionless fraction of rvir=r200crit]
        if field.lower() in ['pos_rel','pos_rel_kpc','pos_rel_rvir']:
            assert not sP.isZoom # otherwise as below for 'rad'
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            # get haloID and load halo regardless, even for non-centrals
            # take center position as subhalo center (same as group center for centrals)
            if subhaloID is None:
                halo = sP.groupCatSingle(sP, haloID=haloID)
                haloPos = halo['GroupPos']
            if subhaloID is not None:
                sub = groupCatSingle(sP, subhaloID=subhaloID)
                halo = groupCatSingle(sP, haloID=sub['SubhaloGrNr'])
                haloID = sub['SubhaloGrNr']
                haloPos = sub['SubhaloPos']

            for j in range(3):
                pos[:,j] -= haloPos[j]

            sP.correctPeriodicDistVecs(pos)
            if '_kpc' in field: pos = sP.units.codeLengthToKpc(pos)
            if '_rvir' in field: pos /= halo['Group_R_Crit200']

            r[field] = pos

        # 3D radial distance from halo center, [code] or [physical kpc] or [dimensionless fraction of rvir=r200crit]
        if field.lower() in ['rad','rad_kpc','rad_kpc_linear','halo_rad','halo_rad_kpc','rad_rvir','halo_rad_rvir']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return
            
            if subhaloID is not None: haloID = groupCatSingle(sP, subhaloID=subhaloID)['SubhaloGrNr']
            halo = groupCatSingle(sP, haloID=haloID)
            haloPos = halo['GroupPos'] # note: is identical to SubhaloPos of GroupFirstSub

            rad = sP.periodicDists(haloPos, pos)
            if '_kpc' in field: rad = sP.units.codeLengthToKpc(rad)
            if '_rvir' in field: rad = rad / halo['Group_R_Crit200']

            r[field] = rad

        # radial velocity, negative=inwards, relative to the central subhalo pos/vel, including hubble correction [km/s]
        if field.lower() in ['vrad','halo_vrad','radvel','halo_radvel']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            if haloID is None and subhaloID is None:
                assert sP.refVel is not None
                print('WARNING: snapshotSubset() using refVel [%.1f %.1f %.1f] as non-zoom run to compute [%s]!' % \
                    (sP.refVel[0],sP.refVel[1],sP.refVel[2],field))
                refPos = sP.refPos
                refVel = sP.refVel
            else:
                shID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
                firstSub = groupCatSingle(sP, subhaloID=shID)
                refPos = firstSub['SubhaloPos']
                refVel = firstSub['SubhaloVel']

            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            r[field] = sP.units.particleRadialVelInKmS(pos, vel, refPos, refVel)

        # velocity 3-vector, relative to the central subhalo pos/vel, [km/s] for each component
        if field.lower() in ['vrel','halo_vrel','relvel','halo_relvel','relative_vel']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            if haloID is None and subhaloID is None:
                assert sP.refVel is not None
                print('WARNING: snapshotSubset() using refVel [%.1f %.1f %.1f] as non-zoom run to compute [%s]!' % \
                    (sP.refVel[0],sP.refVel[1],sP.refVel[2],field))
                refVel = sP.refVel
            else:
                shID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
                firstSub = groupCatSingle(sP, subhaloID=shID)
                refVel = firstSub['SubhaloVel']

            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(vel, dict) and vel['count'] == 0: return vel # no particles of type, empty return

            r[field] = sP.units.particleRelativeVelInKmS(vel, refVel)

        if field.lower() in ['vrelmag','halo_vrelmag','relvelmag','halo_relvelmag','relative_velmag','relative_vmag']:
            vel = snapshotSubset(sP, partType, 'halo_relvel', **kwargs)
            r[field] = np.sqrt( vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2 )

        # angular momentum, relative to the central subhalo pos/vel, either the 3-vector [Msun kpc km/s] or specific magnitude [kpc km/s]
        if field.lower() in ['specangmom_mag','specj_mag','angmom_vec','j_vec']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)

            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            shID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
            firstSub = groupCatSingle(sP, subhaloID=shID)

            if '_mag' in field.lower():
                r[field] = sP.units.particleSpecAngMomMagInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])
            if '_vec' in field.lower():
                r[field] = sP.units.particleAngMomVecInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])

        # done:
        if len(r) >= 1:
            # have at least one custom field, do we also have standard fields requested? if so, load them now and combine
            if len(r) < len(fields):
                standardFields = list(fields)
                for key in r.keys():
                    standardFields.remove(key)

                ss = snapshotSubset(sP, partType, standardFields, sq=False, **kwargs)
                ss.update(r)
                r = ss

            # just one field in total? compress and return single ndarray (by default)
            if len(r) == 1 and sq is True:
                key = list(r.keys())[0]
                return r[key]

            return r # return dictionary

    # alternate field names mappings
    invNameMappings = {}

    altNames = [ [['center_of_mass','com','center'], 'CenterOfMass'],
                 [['xyz','positions','pos'], 'Coordinates'],
                 [['dens','rho'], 'Density'],
                 [['ne','nelec'], 'ElectronAbundance'],
                 [['agnrad','gfm_agnrad'], 'GFM_AGNRadiation'],
                 [['coolrate','gfm_coolrate'], 'GFM_CoolingRate'],
                 [['winddmveldisp'], 'GFM_WindDMVelDisp'],
                 [['metal','Z','gfm_metal','metallicity'], 'GFM_Metallicity'],
                 [['metals'], 'GFM_Metals'],
                 [['u','utherm'], 'InternalEnergy'],
                 [['machnum','shocks_machnum'], 'Machnumber'],
                 [['dedt','energydiss','shocks_dedt','shocks_energydiss'], 'EnergyDissipation'],
                 [['b','bfield'], 'MagneticField'],
                 [['mass'], 'Masses'],
                 [['numtr'], 'NumTracers'],
                 [['id','ids'], 'ParticleIDs'],
                 [['pot'], 'Potential'],
                 [['pres'], 'Pressure'],
                 [['hsml'], 'SmoothingLength'],
                 [['sfr'], 'StarFormationRate'],
                 [['vel'], 'Velocities'],
                 [['vol'], 'Volume'],
                 # stars only:
                 [['initialmass','ini_mass','mass_ini'], 'GFM_InitialMass'],
                 [['stellarformationtime','sftime','birthtime'], 'GFM_StellarFormationTime'],
                 [['stellarphotometrics','stellarphot','sphot'], 'GFM_StellarPhotometrics'],
                 # blackholes only:
                 [['bh_dens','bh_rho'], 'BH_Density'], \
               ]

    for i,field in enumerate(fields):
        for altLabels,toLabel in altNames:
            # alternate field name map, lowercase versions accepted
            if field.lower() in altLabels or field == toLabel.lower():
                #invNameMappings[toLabel] = fields[i] # save inverse so we can undo
                fields[i] = toLabel

    # inds and indRange based subset
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        indRange = [inds.min(), inds.max()]

        val = snapshotSubset(sP, partType, fields, indRange=indRange)
        return val[ inds-inds.min() ]

    if indRange is not None:
        # load a contiguous chunk by making a subset specification in analogy to the group ordered loads
        subset = { 'offsetType'  : np.zeros(sP.nTypes, dtype='int64'),
                   'lenType'     : np.zeros(sP.nTypes, dtype='int64'),
                   'snapOffsets' : snapOffsetList(sP) }

        subset['offsetType'][ptNum(partType)] = indRange[0]
        subset['lenType'][ptNum(partType)]    = indRange[1]-indRange[0]+1

    # multi-dimensional field slicing during load
    mdi = [None] * len(fields) # multi-dimensional index to restrict load to
    trMCFields = sP.trMCFields if sP.trMCFields else np.repeat(-1,14)

    multiDimSliceMaps = [ \
      { 'names':['x','pos_x','posx'],                   'field':'Coordinates',     'fN':0 },
      { 'names':['y','pos_y','posy'],                   'field':'Coordinates',     'fN':1 },
      { 'names':['z','pos_z','posz'],                   'field':'Coordinates',     'fN':2 },
      { 'names':['vx','vel_x','velx'],                  'field':'Velocities',      'fN':0 },
      { 'names':['vy','vel_y','vely'],                  'field':'Velocities',      'fN':1 },
      { 'names':['vz','vel_z','velz'],                  'field':'Velocities',      'fN':2 },
      { 'names':['bx','b_x','bfield_x'],                'field':'MagneticField',   'fN':0 },
      { 'names':['by','b_y','bfield_y'],                'field':'MagneticField',   'fN':1 },
      { 'names':['bz','b_z','bfield_z'],                'field':'MagneticField',   'fN':2 },
      { 'names':['tracer_maxtemp','maxtemp'],           'field':'FluidQuantities', 'fN':trMCFields[0] },
      { 'names':['tracer_maxtemp_time','maxtemp_time'], 'field':'FluidQuantities', 'fN':trMCFields[1] },
      { 'names':['tracer_maxtemp_dens','maxtemp_dens'], 'field':'FluidQuantities', 'fN':trMCFields[2] },
      { 'names':['tracer_maxdens','maxdens'],           'field':'FluidQuantities', 'fN':trMCFields[3] },
      { 'names':['tracer_maxdens_time','maxdens_time'], 'field':'FluidQuantities', 'fN':trMCFields[4] },
      { 'names':['tracer_maxmachnum','maxmachnum'],     'field':'FluidQuantities', 'fN':trMCFields[5] },
      { 'names':['tracer_maxent','maxent'],             'field':'FluidQuantities', 'fN':trMCFields[6] },
      { 'names':['tracer_maxent_time','maxent_time'],   'field':'FluidQuantities', 'fN':trMCFields[7] },
      { 'names':['tracer_laststartime','laststartime'], 'field':'FluidQuantities', 'fN':trMCFields[8] },
      { 'names':['tracer_windcounter','windcounter'],   'field':'FluidQuantities', 'fN':trMCFields[9] },
      { 'names':['tracer_exchcounter','exchcounter'],   'field':'FluidQuantities', 'fN':trMCFields[10] },
      { 'names':['tracer_exchdist','exchdist'],         'field':'FluidQuantities', 'fN':trMCFields[11] },
      { 'names':['tracer_exchdisterr','exchdisterr'],   'field':'FluidQuantities', 'fN':trMCFields[12] },
      { 'names':['tracer_shockmaxmach','shockmaxmach'], 'field':'FluidQuantities', 'fN':trMCFields[13] },
      { 'names':['phot_U','U'],                         'field':'GFM_StellarPhotometrics', 'fN':0 },
      { 'names':['phot_B','B'],                         'field':'GFM_StellarPhotometrics', 'fN':1 },
      { 'names':['phot_V','V'],                         'field':'GFM_StellarPhotometrics', 'fN':2 },
      { 'names':['phot_K','K'],                         'field':'GFM_StellarPhotometrics', 'fN':3 },
      { 'names':['phot_g','g'],                         'field':'GFM_StellarPhotometrics', 'fN':4 },
      { 'names':['phot_r','r'],                         'field':'GFM_StellarPhotometrics', 'fN':5 },
      { 'names':['phot_i','i'],                         'field':'GFM_StellarPhotometrics', 'fN':6 },
      { 'names':['phot_z','z'],                         'field':'GFM_StellarPhotometrics', 'fN':7 },
      { 'names':['metals_H', 'hydrogen'],               'field':'GFM_Metals', 'fN':0 },
      { 'names':['metals_He','helium'],                 'field':'GFM_Metals', 'fN':1 },
      { 'names':['metals_C', 'carbon'],                 'field':'GFM_Metals', 'fN':2 },
      { 'names':['metals_N', 'nitrogen'],               'field':'GFM_Metals', 'fN':3 },
      { 'names':['metals_O', 'oxygen'],                 'field':'GFM_Metals', 'fN':4 },
      { 'names':['metals_Ne','neon'],                   'field':'GFM_Metals', 'fN':5 },
      { 'names':['metals_Mg','magnesium'],              'field':'GFM_Metals', 'fN':6 },
      { 'names':['metals_Si','silicon'],                'field':'GFM_Metals', 'fN':7 },
      { 'names':['metals_Fe','iron'],                   'field':'GFM_Metals', 'fN':8 },
      { 'names':['metals_tot','metals_total'],          'field':'GFM_Metals', 'fN':9 },
      { 'names':['metaltag_SNIa',  'metals_SNIa'],      'field':'GFM_MetalsTagged', 'fN':0 },
      { 'names':['metaltag_SNII',  'metals_SNII'],      'field':'GFM_MetalsTagged', 'fN':1 },
      { 'names':['metaltag_AGB',   'metals_AGB'],       'field':'GFM_MetalsTagged', 'fN':2 },
      { 'names':['metaltag_NSNS',  'metals_NSNS'],      'field':'GFM_MetalsTagged', 'fN':3 },
      { 'names':['metaltag_FeSNIa','metals_FeSNIa'],    'field':'GFM_MetalsTagged', 'fN':4 },
      { 'names':['metaltag_FeSNII','metals_FeSNII'],    'field':'GFM_MetalsTagged', 'fN':5 } \
    ]

    for i,field in enumerate(fields):
        for multiDimMap in multiDimSliceMaps:
            if field in multiDimMap['names']:
                invNameMappings[multiDimMap['field']] = fields[i] # save inverse so we can undo

                fields[i] = multiDimMap['field']
                mdi[i] = multiDimMap['fN']
                assert mdi[i] >= 0 # otherwise e.g. not assigned in sP

    if sum(m is not None for m in mdi) > 1:
        raise Exception('Not supported for multiple MDI at once.')

    # halo or subhalo based subset
    if haloID is not None or subhaloID is not None:
        assert not sP.isPartType(partType, 'tracer') # not group-ordered
        subset = _haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    # check memory cache (only simplest support at present, for indRange/full returns of global cache)
    if len(fields) == 1 and mdi[0] is None:
        cache_key = 'snap%s_%s_%s' % (sP.snap,partType,fields[0].replace(" ","_"))
        if cache_key in sP.data:
            # global? (or rather, whatever is in sP.data... be careful)
            if indRange is None:
                indRange = [0,sP.data[cache_key].shape[0]-1]
                #print('CAUTION: Cached return, and indRange is None, returning all of sP.data field.')

            #print('NOTE: Returning [%s] from cache, indRange [%d - %d]!' % (cache_key,indRange[0],indRange[1]))
            if sq:
                return sP.data[cache_key][indRange[0]:indRange[1]+1]
            else:
                return {fields[0]:sP.data[cache_key][indRange[0]:indRange[1]+1]}

    # load from disk
    r = il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset, mdi=mdi, sq=sq, float32=float32)

    # optional unit post-processing
    if isinstance(r, np.ndarray) and len(fieldsOrig) == 1:
        if fieldsOrig[0] in ['tracer_maxent','tracer_maxtemp'] and r.max() < 20.0:
            raise Exception('Unexpectedly low max for non-log values, something maybe changed.')
            
        if fieldsOrig[0] == 'tracer_maxent':
            r = sP.units.tracerEntToCGS(r, log=True) # [log cgs] = [log K cm^2]
        if fieldsOrig[0] == 'tracer_maxtemp':
            r = logZeroNaN(r) # [log Kelvin]

    # inverse map multiDimSliceMaps such that return dict has key names exactly as requested
    # todo: could also do for altNames (just uncomment above, but need to refactor codebase)
    if isinstance(r,dict):
        for newLabel,origLabel in invNameMappings.items():
            r[origLabel] = r.pop(newLabel) # change key label

    return r

def _func(sP,partType,field,indRangeLoad,indRangeSave,float32,array):
    """ Multiprocessing target, which simply calls snapshotSubset() and writes the result 
    directly into a shared memory array. Always called with only one field. """
    data = sP.snapshotSubset(partType, field, indRange=indRangeLoad, sq=True, float32=float32)
    array[ indRangeSave[0]:indRangeSave[1], ... ] = data
    # note: could move this into il.snapshot.loadSubset() following the strategy of the 
    # parallel groupCat() load, to actually avoid this intermediate memory usage

def snapshotSubsetParallel(sP, partType, fields, inds=None, indRange=None, haloID=None, subhaloID=None, 
                           sq=True, haloSubset=False, float32=False, nThreads=8):
    """ Identical to snapshotSubset() except split filesystem load over a number of 
    concurrent python+h5py reader processes and gather the result. """
    import multiprocessing as mp
    import multiprocessing.sharedctypes
    import ctypes
    from functools import partial

    # method to disable parallel loading, which does not work with custom-subset cached fields 
    # inside sP.data since indRange as computed below cannot know about this
    if 'nThreads' in sP.data and sP.data['nThreads'] == 1:
        return snapshotSubset(sP, partType, fields, inds=inds, indRange=indRange, haloID=haloID, 
            subhaloID=subhaloID, sq=sq, haloSubset=haloSubset, float32=float32)

    # sanity checks
    if indRange is not None:
        assert indRange[0] >= 0 and indRange[1] >= indRange[0]
    if haloSubset and (not sP.groupOrdered or (indRange is not None)):
        raise Exception('haloSubset only for groupordered snapshots, and not with indRange subset.')
    if haloID is not None or subhaloID is not None:
        raise Exception('Not yet supported.')

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)
    fields = list(iterable(fields))

    # get total size
    h = sP.snapshotHeader()
    numPartTot = h['NumPart'][sP.ptNum(partType)]

    if numPartTot == 0:
        return {'count':0}
    if numPartTot < nThreads*2:
        # low particle count, use serial
        return snapshotSubset(sP, partType, fields, inds=inds, indRange=indRange,
                              sq=sq, haloSubset=haloSubset, float32=float32)

    # set indRange to load
    if inds is not None:
        # load the range which bounds the minimum and maximum indices, then return subset
        indRange = [inds.min(), inds.max()]

    if indRange is None:
        indRange = [0, numPartTot-1]
    else:
        numPartTot = indRange[1] - indRange[0] + 1

    if numPartTot == 0:
        return {'count':0}
    
    # haloSubset only? update indRange and continue
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        indRange = [0, offsets_pt[:,sP.ptNum(partType)].max()]

    # get shape and dtype by loading one element
    sample = snapshotSubset(sP, partType, fields, indRange=[0,0], sq=False, float32=float32)

    # prepare return
    r = {}

    # do different fields in a loop, if more than one requested
    for k in sample.keys():
        if k == 'count':
            continue

        # prepare shape
        shape = [numPartTot] 

        if sample[k].ndim > 1:
            shape.append( sample[k].shape[1] ) # i.e. Coordinates, append 3 as second dimension

        # allocate global return
        size = int(np.prod(shape) * sample[k].dtype.itemsize) # bytes
        ctype = ctypes.c_byte

        shared_mem_array = mp.sharedctypes.RawArray(ctype, size)
        numpy_array_view = np.frombuffer(shared_mem_array, sample[k].dtype).reshape(shape)

        # spawn threads with indRange subsets
        offset = 0
        processes = []

        for i in range(nThreads):
            indRangeLoad = pSplitRange(indRange, nThreads, i)
            if i < nThreads-1:
                # pSplitRange() returns overlap as per np indexing convention, but snapshotSubset()
                # indRange is inclusive of the last index
                indRangeLoad[1] -= 1

            numLoadLoc   = indRangeLoad[1] - indRangeLoad[0] + 1
            indRangeSave = [offset, offset + numLoadLoc]
            offset += numLoadLoc

            #_func(sP,partType,k,indRangeLoad,indRangeSave,float32,numpy_array_view) # debugging only
            args = (sP,partType,k,indRangeLoad,indRangeSave,float32,numpy_array_view)
            p = mp.Process(target=_func, args=args)
            processes.append(p)

        # wrap in try, to help avoid zombie processes and system issues
        try:
            for p in processes:
                p.start()
        finally:
            for p in processes:
                p.join()

        # add into dict
        if inds is not None:
            r[k] = numpy_array_view[inds-inds.min()]
        else:
            r[k] = numpy_array_view

    if len(r) == 1 and sq:
        # single ndarray return
        return r[list(r.keys())[0]]

    r['count'] = numPartTot
    return r
