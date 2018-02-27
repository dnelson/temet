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
from os import mkdir

import illustris_python as il
from illustris_python.util import partTypeNum as ptNum
from util.helper import iterable, logZeroNaN, curRepoVersion, pSplitRange, numPartToChunkLoadSize
from cosmo.util import periodicDists

def auxCat(sP, fields=None, pSplit=None, reCalculate=False, searchExists=False, indRange=None, onlyMeta=False, expandPartial=False):
    """ Load field(s) from the auxiliary group catalog, computing missing datasets on demand. 
      reCalculate  : force redo of computation now, even if data is already saved in catalog
      searchExists : return None if data is not already computed, i.e. do not calculate right now 
      indRange     : if a tuple/list, load only the specified range of data (field and  e.g. subhaloIDs)
      onlyMeta     : load only attributes and coverage information 
      expandPartial : if data was only computed for a subset of all subhalos, expand this now into a total nSubs sized array """
    from cosmo import auxcatalog
    import datetime
    import getpass

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
        if field not in auxcatalog.fieldComputeFunctionMapping.keys() + auxcatalog.manualFieldNames:
            raise Exception('Unrecognized field ['+field+'] for auxiliary catalog.')

        # check for existence of auxiliary catalog file for this dataset
        auxCatPath = pathStr1 % (field,sP.snap)

        # split the calculation over multiple jobs? check if all chunks already exist
        if pSplit is not None:
            assert indRange is None # for load only
            auxCatPathSplit = pathStr2 % (field,sP.snap,pSplit[0],pSplit[1])

            if isfile(auxCatPathSplit) and not reCalculate:
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
                        allShape = f[field].shape

                        if len(allShape) > 1 and allShape[condAxis] > condThresh:
                            # very high dimensionality auxCat (e.g. full spectra), save only non-nan
                            assert f[field].ndim == 2
                            assert 'partInds' not in f or f['partInds'][()] == -1

                            temp_r = f[field][()]
                            w = np.where( np.isfinite(np.sum(temp_r,axis=condAxis)) )
                            allCount += len(w[0])
                            print(' [%2d] keeping %d of %d non-nan.' % (i,len(w[0]),f['subhaloIDs'].size))
                        else:
                            # normal, save full Subhalo size
                            allCount += f['subhaloIDs'].size
                            if 'partInds' in f: allCountPart += f['partInds'].size

                if allExist:
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
                            temp_r = f[field][()]

                            if length == 0: continue

                            if len(allShape) > 1 and allShape[condAxis] >= condThresh:
                                # want to condense, stamp in to dense indices
                                subhaloIDsToSave = f[catIndFieldName][()]

                                if f[field].ndim == 2 and 'Subhalo_SDSSFiberSpectra' in field:
                                    # splits saved all subhalos (sdss spectra)
                                    #assert 'Subhalo_SDSSFiberSpectra' in field # otherwise check
                                    w = np.where( np.isfinite(np.sum(temp_r,axis=condAxis)) )[0]
                                    length = len(w)
                                    temp_r = temp_r[w,:]
                                    subhaloIDsToSave = subhaloIDsToSave[w]

                                if f[field].ndim == 3:
                                    # splits saved a subset of subhalos
                                    assert 'Subhalo_RadProfile' in field # otherwise check why we are here

                                subhaloIDs[offset : offset+length] = subhaloIDsToSave
                                subhaloIndsStamp = np.arange(offset,offset+length)
                            else:
                                # full, stamp in to indices corresponding to subhalo indices
                                subhaloIndsStamp = f[catIndFieldName][()]
                                subhaloIDs[subhaloIndsStamp] = subhaloIndsStamp

                            # save into final array
                            if f[field].ndim == 1:
                                new_r[subhaloIndsStamp] = temp_r
                            if f[field].ndim == 2:
                                new_r[subhaloIndsStamp, :] = temp_r
                            if f[field].ndim == 3:
                                new_r[subhaloIndsStamp, :, :] = temp_r

                            assert f[field].ndim in [1,2,3]

                            for attr in f[field].attrs:
                                attrs[attr] = f[field].attrs[attr]

                            offset += length

                            if 'wavelength' in f: attrs['wavelength'] = f['wavelength'][()]

                    assert np.count_nonzero(np.where(subhaloIDs < 0)) == 0
                    assert np.count_nonzero(new_r == -1.0) == 0

                    # save new auxCat
                    assert not isfile(auxCatPath)
                    with h5py.File(auxCatPath,'w') as f:
                        f.create_dataset(field, data=new_r)
                        if catIndFieldName == 'subhaloIDs':
                            f.create_dataset(catIndFieldName, data=subhaloIDs)
                        for attrName, attrValue in attrs.iteritems():
                            f[field].attrs[attrName] = attrValue

                    print(' Concatenated new [%s] and saved.' % auxCatPath.split("/")[-1])
                    print(' All chunks concatenated, please manually delete them now.')
                else:
                    print('Chunk [%s] already exists, but all not yet done, exiting.' % auxCatPathSplit)
                    r[field] = None
                    continue

        # done with chunk logic.
        # just checking for existence? (do not calculate right now if missing)
        if not isfile(auxCatPath) and searchExists:
            r[field] = None
            continue

        # load previously computed values
        if isfile(auxCatPath) and not reCalculate:
            #print('Load existing: ['+field+'] '+auxCatPath)
            with h5py.File(auxCatPath,'r') as f:
                # load data
                if not onlyMeta:
                    if indRange is None:
                        r[field] = f[field][()]
                    else:
                        if f[field].ndim == 1: r[field] = f[field][indRange[0]:indRange[1]]
                        if f[field].ndim == 2: r[field] = f[field][indRange[0]:indRange[1],:]
                        if f[field].ndim == 3: r[field] = f[field][indRange[0]:indRange[1],:,:]
                        r[field] = np.squeeze(r[field]) # remove any degenerate dimensions
                        assert f[field].ndim in [1,2,3]

                # load metadata
                r[field+'_attrs'] = {}
                for attr in f[field].attrs:
                    r[field+'_attrs'][attr] = f[field].attrs[attr]

                # load subhaloIDs, partInds if present
                for attrName in largeAttrNames:
                    if attrName in f:
                        if indRange is None:
                            r[attrName] = f[attrName][()]
                        else:
                            r[attrName] = f[attrName][indRange[0]:indRange[1]]

            # subhaloIDs indicates computation only for partial set of subhalos?
            if expandPartial:
                nSubsTot = groupCatHeader(sP)['Nsubgroups_Total']
                
                if 'subhaloIDs' in r and (r['subhaloIDs'].size < nSubsTot) and expandPartial:
                    shape = np.array(r[field].shape)
                    shape[0] = nSubsTot
                    new_data = np.zeros( shape, dtype=r[field].dtype )
                    new_data.fill(np.nan)
                    if r[field].ndim == 1: new_data[r['subhaloIDs']] = r[field]
                    if r[field].ndim == 2: new_data[r['subhaloIDs'],:] = r[field]
                    if r[field].ndim == 3: new_data[r['subhaloIDs'],:,:] = r[field]
                    print(' Auxcat Expanding [%d] to [%d] elements for [%s].' % (r[field].size,new_data.size,field))
                    r[field] = new_data

            continue

        # either does not exist yet, or reCalculate requested
        pSplitStr = ''
        savePath = auxCatPath

        if pSplit is not None:
            pSplitStr = ' (split %d of %d)' % (pSplit[0],pSplit[1])
            savePath = auxCatPathSplit

        print('Compute and save: [%s]%s' % (field,pSplitStr))

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
            f.create_dataset(field, data=r[field])

            # save metadata and any additional descriptors as attributes
            f[field].attrs['CreatedOn']   = datetime.date.today().strftime('%d %b %Y')
            f[field].attrs['CreatedRev']  = curRepoVersion()
            f[field].attrs['CreatedBy']   = getpass.getuser()
            
            for attrName, attrValue in attrs.iteritems():
                if attrName in largeAttrNames:
                    f.create_dataset(attrName, data=r[field+'_attrs'][attrName])
                    continue # typically too large to store as an attribute
                f[field].attrs[attrName] = attrValue

        if not reCalculate:
            print(' Saved new [%s].' % savePath)
        else:
            print(' Saved over existing [%s].' % savePath)
                    
    return r

def gcPath(basePath, snapNum, chunkNum=0, noLocal=False, checkExists=False):
    """ Find and return absolute path to a group catalog HDF5 file.
        Can be used to redefine illustris_python version (il.groupcat.gcPath = cosmo.load.gcPath). """

    # local scratch test: call ourself with a basePath corresponding to local stratch (on dev node)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/scratch/" + bpSplit[-3] + "/" + bpSplit[-2] + "/"
        localFT = gcPath(localBP, snapNum, noLocal=True, checkExists=True)

        if localFT:
            print("Note: Reading group catalog from local scratch!")
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
                  basePath + 'groups_' + ext + '/groups_' + ext + '.' + str(chunkNum) + '.hdf5'
                ]

    for fileName in fileNames:
        if isfile(fileName):
            return fileName

    if checkExists:
        return None

    raise Exception("No group catalog found.")

def groupCat(sP, readIDs=False, skipIDs=False, fieldsSubhalos=None, fieldsHalos=None, sq=False):
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

            # halo mass (m200 or m500) of parent halo [code, msun, or log msun]
            if quant in ['mhalo_200','mhalo_200_log','mhalo_200_code',
                         'mhalo_500','mhalo_500_log','mhalo_200_code']:
                od = 200 if '_200' in quant else 500

                gc = groupCat(sP, fieldsHalos=['Group_M_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])

                r[field] = gc['halos']['Group_M_Crit%d'%od][gc['subhalos']]

                if '_code' not in quant:
                    # conversion: code -> physical units
                    r[field] = sP.units.codeMassToMsun( r[field] )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

                # satellites given nan
                mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                mask[ gc['halos']['GroupFirstSub'] ] = 1
                wSat = np.where(mask == 0)
                r[field][wSat] = np.nan

            # subhalo mass [msun or log msun]
            if quant in ['mhalo_subfind','mhalo_subfind_log']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMass'])
                r[field] = sP.units.codeMassToMsun( gc['subhalos'] )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # subhalo stellar mass (<30 pkpc definition, with auxCat) [msun or log msun]
            if quant in ['mstar_30pkpc','mstar_30pkpc_log']:
                acField = 'Subhalo_Mass_30pkpc_Stars'
                ac = auxCat(sP, fields=[acField])
                r[field] = sP.units.codeMassToMsun( ac[acField] )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # central flag (1 if central, 0 if not)
            if quant in ['central_flag','cen_flag','is_cen','is_central']:
                gc = groupCat(sP, fieldsHalos=['GroupFirstSub'])

                # satellites given zero
                r[field] = np.zeros( gc['header']['Nsubgroups_Total'], dtype='int16' )
                r[field][ gc['halos'] ] = 1

            # isolated flag (1 if 'isolated' according to criterion, 0 if not, -1 if unprocessed)
            if 'isolated3d,' in quant:
                from cosmo.clustering import isolationCriterion3D

                # e.g. 'isolated3d,mstar_30pkpc,max,in_300pkpc'
                _, quant, max_type, dist = quant.split(',')
                dist = float( dist.split('in_')[1].split('pkpc')[0] )

                ic3d = isolationCriterion3D(sP, dist) #defaults: cenSatSelect='all', mstar30kpc_min=9.0

                r[field] = ic3d['flag_iso_%s_%s' % (quant,max_type)]

            # ssfr (1/yr or 1/Gyr) (SFR and Mstar both within 2r1/2stars) (optionally Mstar in 30pkpc)
            if quant in ['ssfr','ssfr_gyr','ssfr_30pkpc','ssfr_30pkpc_gyr',
                         'ssfr_log','ssfr_gyr_log','ssfr_30pkpc_log','ssfr_30pkpc_gyr_log']:
                gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType','SubhaloSFRinRad'])
                mstar = sP.units.codeMassToMsun( gc['subhalos']['SubhaloMassInRadType'][:,sP.ptNum('stars')] )

                # replace stellar masses with auxcat values within constant aperture, if requested
                if '_30pkpc' in quant:
                    mstar = groupCat(sP, fieldsSubhalos=['mstar_30pkpc'])

                # set mstar=0 subhalos to nan 
                w = np.where(mstar == 0.0)[0]
                if len(w):
                    mstar[w] = 1.0
                    gc['subhalos']['SubhaloSFRinRad'][w] = np.nan

                r[field] = gc['subhalos']['SubhaloSFRinRad'] / mstar

                if '_gyr' in quant:
                    r[field] *= 1e9 # 1/yr to 1/Gyr
                if '_log' in quant: r[field] = logZeroNaN(r[field])

            # virial radius (r200 or r500) of parent halo [code, pkpc, log pkpc]
            if quant in ['rhalo_200_code', 'rhalo_200','rhalo_200_log','rhalo_500','rhalo_500_log']:
                od = 200 if '_200' in quant else 500

                gc = groupCat(sP, fieldsHalos=['Group_R_Crit%d'%od,'GroupFirstSub'], fieldsSubhalos=['SubhaloGrNr'])
                r[field] = gc['halos']['Group_R_Crit%d'%od][gc['subhalos']]

                if '_code' not in quant:
                    # conversion: code -> physical units
                    r[field] = sP.units.codeLengthToKpc( r[field] )

                if '_log' in quant: r[field] = logZeroNaN(r[field])

                # satellites given nan
                mask = np.zeros( gc['subhalos'].size, dtype='int16' )
                mask[ gc['halos']['GroupFirstSub'] ] = 1
                wSat = np.where(mask == 0)
                r[field][wSat] = np.nan

            # radial distance to parent halo [code, pkpc, log pkpc, r200frac] (centrals will have 0)
            if quant in ['rdist_code','rdist','rdist_log','rdist_rvir']:
                from cosmo.util import periodicDists
                gc = groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], 
                                  fieldsSubhalos=['SubhaloPos','SubhaloGrNr'])

                parInds = gc['subhalos']['SubhaloGrNr']
                r[field] = periodicDists( gc['halos']['GroupPos'][parInds,:], 
                                          gc['subhalos']['SubhaloPos'], sP)

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

        if len(r) >= 1:
            # have at least one custom subhalo field, were halos also requested? not allowed
            assert fieldsHalos is None

            # do we also have standard fields requested? if so, load them now and combine
            if len(r) < len(fieldsSubhalos):
                standardFields = list(fieldsSubhalos)
                for key in r.keys():
                    standardFields.remove(key)
                gc = groupCat(sP, fieldsSubhalos=standardFields, sq=False)
                gc['subhalos'].update(r)
                r = gc

            if len(r) == 1:
                # compress and return single field (by default, unlike for standard fields)
                key = r.keys()[0]
                assert len(r.keys()) == 1
                return r[key]
            elif len(r) > 1:
                # return dictionary (no 'subhalos' wrapping)
                assert sq is False
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
        r['subhalos'] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=fieldsSubhalos)

        # Illustris-1 metallicity fixes if needed
        if sP.run == 'illustris':
            for field in fieldsSubhalos:
                if 'Metallicity' in field:
                    il.groupcat.gcPath = il.groupcat.gcPathOrig # set to new catalogs
                    print('Note: Overriding subhalo ['+field+'] with groups_ new catalog values.')
                    r['subhalos'][field] = il.groupcat.loadSubhalos(sP.simPath, sP.snap, fields=field)
            il.groupcat.gcPath = gcPath # restore

    if fieldsHalos is not None:
        r['halos'] = il.groupcat.loadHalos(sP.simPath, sP.snap, fields=fieldsHalos)
    
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

    if sq:
        # remove 'halos'/'subhalos' subdict, and field subdict
        assert fieldsSubhalos is None or fieldsHalos is None

        if fieldsSubhalos is not None: r = r['subhalos']
        if fieldsHalos is not None: r = r['halos']

        if isinstance(r,dict):
            assert len(r.keys()) == 1
            r = r[ r.keys()[0] ]

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
    sbNum = subbox if isinstance(subbox, (int,long)) else 0

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
    path = basePath + sbStr2 + 'snapdir_' + sbStr1 + str(snapNum).zfill(3) + '/*.hdf5'

    nChunks = len(glob.glob(path))

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

def snapOffsetList(sP):
    """ Make the offset table (by type) for the snapshot files, to be able to quickly determine within 
        which file(s) a given offset+length will exist. """
    saveFilename = sP.derivPath + 'offsets/snapshot_' + str(sP.snap) + '.hdf5'

    if not isdir(sP.derivPath+'offsets'):
        mkdir(sP.derivPath+'offsets')

    if isfile(saveFilename):
            with h5py.File(saveFilename,'r') as f:
                snapOffsets = f['offsets'][()]
    else:
        nChunks = snapNumChunks(sP.simPath, sP.snap, sP.subbox)
        snapOffsets = np.zeros( (sP.nTypes, nChunks), dtype='int64' )

        for i in np.arange(1,nChunks+1):
            f = h5py.File( snapPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )

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
        nChunks = snapNumChunks(sP.simPath, sP.snap)
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
        nChunks = snapNumChunks(sP.simPath, sP.snap)
        print('Calculating new groupCatOffsetsListIntoSnap... ['+str(nChunks)+' chunks]')

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

        for i in range(1,nChunks+1):
            # load header, get number of groups/subgroups in this file, and lengths
            f = h5py.File( gcPath(sP.simPath,sP.snap,chunkNum=i-1), 'r' )
            header = dict( f['Header'].attrs.items() )
            
            if header['Ngroups_ThisFile'] > 0:
                groupLenType[groupCount:groupCount+header['Ngroups_ThisFile']] = f['Group']['GroupLenType']
                groupNsubs[groupCount:groupCount+header['Ngroups_ThisFile']]   = f['Group']['GroupNsubs']
            if header['Nsubgroups_ThisFile'] > 0:
                subgroupLenType[subgroupCount:subgroupCount+header['Nsubgroups_ThisFile']] = f['Subhalo']['SubhaloLenType']
            
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
        subset['lenType'] = f[gcName][gcName+'LenType'][groupOffset,:]
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

    # caching
    useCache = True

    if useCache:
        cachePath = sP.derivPath + 'cache/'
        cacheFile = cachePath + 'cached_%s_%s_%d.hdf5' % (partType,field.replace(" ","-"),sP.snap)
        indRangeAll = [0, snapshotHeader(sP)['NumPart'][sP.ptNum(partType)] ]

        if not isdir(cachePath):
            mkdir(cachePath)

        if not isfile(cacheFile):
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
                    values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeLocal)
                    if prop == 'mass':
                        values *= snapshotSubset(sP, partType, 'Masses', **kwargs)
                else:
                    # emission flux
                    lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeLocal)
                    values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2]

                with h5py.File(cacheFile) as f:
                    f['field'][indRangeLocal[0]:indRangeLocal[1]+1] = values

                print(' [%2d] saved %d - %d' % (i,indRangeLocal[0],indRangeLocal[1]))
            print('Saved: [%s].' % cacheFile.split(sP.derivPath)[1])

        # load from existing cache
        print('Loading [%s] [%s] from [%s].' % (partType,field,cacheFile.split(sP.derivPath)[1]))

        with h5py.File(cacheFile, 'r') as f:
            assert f['field'].size == indRangeAll[1]
            if indRangeOrig is None:
                values = f['field'][()]
            else:
                values = f['field'][indRangeOrig[0] : indRangeOrig[1]+1]

    else:
        # old, don't create or use cache
        from cosmo.cloudy import cloudyIon, cloudyEmission
        if prop in ['mass','frac']:
            ion = cloudyIon(sP, el=element, redshiftInterp=True)
        else:
            emis = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            wavelength = emis.lineWavelength(lineName)

        if prop in ['mass','frac']:
            # either ionization fractions, or total mass in the ion
            values = ion.calcGasMetalAbundances(sP, element, ionNum, indRange=indRangeOrig)
            if prop == 'mass':
                values *= snapshotSubset(sP, partType, 'Masses', **kwargs)
        else:
            # emission flux
            lum = emis.calcGasLineLuminosity(sP, lineName, indRange=indRangeOrig)
            values = sP.units.luminosityToFlux(lum, wavelength=wavelength) # [photon/s/cm^2]

    return values

def snapshotSubset(sP, partType, fields, 
                   inds=None, indRange=None, haloID=None, subhaloID=None, 
                   mdi=None, sq=True, haloSubset=False):
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
    if '_' in partType:
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
    # TODO: combining composite fields with len(fields)>1 currently skips any others, returns single ndarray
    for i,field in enumerate(fields):
        # temperature (from u,nelec) [log K]
        if field.lower() in ["temp", "temperature"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            return sP.units.UToTemp(u,ne,log=True)

        # temperature (from u,nelec) [linear K]
        if field.lower() in ["temp_linear"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            return sP.units.UToTemp(u,ne,log=False)

        # hydrogen number density (from rho) [linear 1/cm^3]
        if field.lower() in ["nh","hdens"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True,numDens=True)
            return sP.units.hydrogen_massfrac * dens # constant 0.76 assumed

        # number density (from rho) [linear 1/cm^3]
        if field.lower() in ["numdens"]:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            return sP.units.codeDensToPhys(dens,cgs=True,numDens=True)

        # mass density to critical baryon density [linear dimensionless]
        if field.lower() in ['dens_critratio','dens_critb']:
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            return sP.units.codeDensToCritRatio(dens, baryon=('_critb' in field.lower()), log=False)

        # particle/cell mass [linear solar masses]
        if field.lower() in ['mass_msun']:
            mass = snapshotSubset(sP, partType, 'mass', **kwargs)
            return sP.units.codeMassToMsun(mass)

        # entropy (from u,dens) [log cgs] == [log K cm^2]
        if field.lower() in ["ent", "entr", "entropy"]:
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            return sP.units.calcEntropyCGS(u,dens,log=True)

        # velmag (from 3d velocity) [physical km/s]
        if field.lower() in ["vmag", "velmag"]:
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)
            vel = sP.units.particleCodeVelocityToKms(vel)
            return np.sqrt( vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1] + vel[:,2]*vel[:,2] )

        # Bmag (from vector field) [physical Gauss]
        if field.lower() in ["bmag", "bfieldmag"]:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            b = sP.units.particleCodeBFieldToGauss(b)
            bmag = np.sqrt( b[:,0]*b[:,0] + b[:,1]*b[:,1] + b[:,2]*b[:,2] )
            return bmag

        # Bmag in micro-Gauss [physical uG]
        if field.lower() in ['bmag_ug', 'bfieldmag_ug']:
            return snapshotSubset(sP, partType, 'bmag', **kwargs) * 1e6

        # Alfven velocity magnitude (of electron plasma) [physical km/s]
        if field.lower() in ["vmag_alfven", "velmag_alfven"]:
            assert 0 # todo

        # volume [ckpc/h]^3
        if field.lower() in ["vol", "volume"]:
            # Volume eliminated in newer outputs, calculate as necessary, otherwise fall through
            if not snapHasField(sP, partType, 'Volume'):
                mass = snapshotSubset(sP, partType, 'mass', **kwargs)
                dens = snapshotSubset(sP, partType, 'dens', **kwargs)
                return (mass / dens)

        # volume in physical [cm^3] or [kpc^3]
        if field.lower() in ['vol_cm3','volume_cm3']:
            return sP.units.codeVolumeToCm3( snapshotSubset(sP, partType, 'volume', **kwargs) )
        if field.lower() in ['vol_kpc3','volume_kpc3']:
            return sP.units.codeVolumeToKpc3( snapshotSubset(sP, partType, 'volume', **kwargs) )

        # cellsize (from volume) [ckpc/h]
        if field.lower() in ["cellsize", "cellrad"]:
            # Volume eliminated in newer outputs, calculate as necessary from Mass/Density
            if snapHasField(sP, partType, 'Volume'):
                vol = snapshotSubset(sP, partType, 'vol', **kwargs)
            else:
                mass = snapshotSubset(sP, partType, 'mass', **kwargs)
                dens = snapshotSubset(sP, partType, 'dens', **kwargs)
                vol = mass / dens
                
            return (vol * 3.0 / (4*np.pi))**(1.0/3.0)

        # cellsize [physical kpc]
        if field.lower() in ["cellsize_kpc","cellrad_kpc"]:
            cellsize_code = snapshotSubset(sP, partType, 'cellsize', **kwargs)
            return sP.units.codeLengthToKpc(cellsize_code)

        # particle volume (from subfind hsml of N nearest DM particles) [ckpc/h]
        if field.lower() in ["subfind_vol","subfind_volume"]:
            hsml = snapshotSubset(sP, partType, 'SubfindHsml', **kwargs)
            return (4.0/3.0) * np.pi * hsml**3.0

        # metallicity in linear(solar) units
        if field.lower() in ["metal_solar","z_solar"]:
            metal = snapshotSubset(sP, partType, 'metal', **kwargs) # metal mass / total mass ratio
            return sP.units.metallicityInSolar(metal,log=False)

        # stellar age in Gyr (convert GFM_StellarFormationTime scalefactor)
        if field.lower() in ["star_age","stellar_age"]:
            curUniverseAgeGyr = sP.units.redshiftToAgeFlat(sP.redshift)
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            birthRedshift = 1.0/birthTime - 1.0
            return curUniverseAgeGyr - sP.units.redshiftToAgeFlat(birthRedshift)

        # bolometric x-ray luminosity (simple model) [erg/s]
        if field.lower() in ['xray_lum','xray']:
            sfr  = snapshotSubset(sP, partType, 'StarFormationRate', **kwargs)
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            mass = snapshotSubset(sP, partType, 'Masses', **kwargs)
            u    = snapshotSubset(sP, partType, 'u', **kwargs)
            ne   = snapshotSubset(sP, partType, 'ne', **kwargs)
            return sP.units.calcXrayLumBolometric(sfr, u, ne, mass, dens)

        # pressure_ratio (linear ratio of magnetic to gas pressure)
        if field.lower() in ['pres_ratio','pressure_ratio']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            b    = snapshotSubset(sP, partType, 'MagneticField', **kwargs)

            P_gas = sP.units.calcPressureCGS(u, dens)
            P_B   = sP.units.calcMagneticPressureCGS(b)
            return P_B/P_gas

        # u_B_ke_ratio (linear ratio of magnetic to kinetic energy density)
        if field.lower() in ['u_b_ke_ratio','magnetic_kinetic_edens_ratio','b_ke_edens_ratio']:
            u_b = snapshotSubset(sP, partType, 'p_b', **kwargs) # [log K/cm^3]
            u_ke = snapshotSubset(sP, partType, 'u_ke', **kwargs) # [log K/cm^3]
            return 10.0**u_b / 10.0**u_ke

        # gas pressure [log K/cm^3]
        if field.lower() in ['gas_pres','gas_pressure','p_gas']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            return sP.units.calcPressureCGS(u, dens, log=True)

        if field.lower() in ['p_gas_linear']:
            return 10.0**snapshotSubset(sP, partType, 'p_gas', **kwargs)

        # magnetic pressure [log K/cm^3]
        if field.lower() in ['mag_pres','magnetic_pressure','p_b','p_magnetic']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            return sP.units.calcMagneticPressureCGS(b, log=True)

        if field.lower() in ['p_b_linear']:
            return 10.0**snapshotSubset(sP, partType, 'p_b', **kwargs)

        # kinetic energy density [log erg/cm^3]
        if field.lower() in ['kinetic_energydens','kinetic_edens','u_ke']:
            dens_code = snapshotSubset(sP, partType, 'Density', **kwargs)
            vel_kms = snapshotSubset(sP, partType, 'velmag', **kwargs)
            return sP.units.calcKineticEnergyDensityCGS(dens_code, vel_kms, log=True)

        # total pressure, magnetic plus gas [K/cm^3]
        if field.lower() in ['p_tot','pres_tot','pres_total','pressure_tot','pressure_total']:
            P_B = 10.0**snapshotSubset(sP, partType, 'P_B', **kwargs)
            P_gas = 10.0**snapshotSubset(sP, partType, 'P_gas', **kwargs)
            return ( P_B + P_gas )

        # ------------------------------------------------------------------------------------------------------

        # synchrotron power model [W/Hz]
        if field.lower() in ['p_sync_ska','p_sync_ska_eta43','p_sync_ska_alpha15','p_sync_vla']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            vol = snapshotSubset(sP, partType, 'Volume', **kwargs)

            modelArgs = {}
            if '_ska' in field: modelArgs['telescope'] = 'SKA'
            if '_vla' in field: modelArgs['telescope'] = 'VLA'
            if '_eta43' in field: modelArgs['eta'] = (4.0/3.0)
            if '_alpha15' in field: modelArgs['alpha'] = 1.5
            return sP.units.synchrotronPowerPerFreq(b, vol, watts_per_hz=True, log=False, **modelArgs)

        # hydrogen model mass calculation (todo: generalize to different molecular models)
        if field.lower() in ['h i mass', 'hi mass', 'himass', 'h1mass', 'hi_mass']:
            assert haloID is None and subhaloID is None # otherwise handle, construct indRange
            from cosmo.hydrogen import hydrogenMass
            return hydrogenMass(None, sP, atomic=True, indRange=indRange)

        if field.lower() in ['h 2 mass', 'h2 mass', 'h2mass'] or 'h2mass_' in field.lower():
            assert haloID is None and subhaloID is None # otherwise handle, construct indRange
            from cosmo.hydrogen import hydrogenMass
            if 'h2mass_' in field.lower():
                molecularModel = field.lower().split('_')[1]
            else:
                molecularModel = 'BL06'
                print('Warning: using [%s] model for H2 by default since unspecified.' % molecularModel)

            return hydrogenMass(None, sP, molecular=molecularModel, indRange=indRange)

        # cloudy based ionic mass (or emission flux) calculation, if field name has a space in it
        if " " in field:
            return _ionLoadHelper(sP, partType, field, kwargs)

        if '_ionmassratio' in field:
            # per-cell ratio between two ionic masses, e.g. "O6_O8_ionmassratio"
            from cosmo.cloudy import cloudyIon
            ion = cloudyIon(sP=None)
            ion1, ion2, _ = field.split('_')

            mass1 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion1), **kwargs)
            mass2 = snapshotSubset(sP, partType, '%s mass' % ion.formatWithSpace(ion2), **kwargs)
            return ( mass1 / mass2 )

        if '_numratio' in field:
            # metal number density ratio e.g. "Si_H_numratio", relative to solar, [Si/H] = log(n_Si/n_H)_cell - log(n_Si/n_H)_solar
            from cosmo.cloudy import cloudyIon
            el1, el2, _ = field.split('_')

            ion = cloudyIon(sP=None)
            el1_massratio = snapshotSubset(sP, partType, 'metals_'+el1, **kwargs)
            el2_massratio = snapshotSubset(sP, partType, 'metals_'+el2, **kwargs)
            el_ratio = el1_massratio / el2_massratio

            return ion._massRatioToRelSolarNumDensRatio(el_ratio, el1, el2)

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
                masses = sP.units.codeMassToMsun(sP, masses)

            return masses

        # metal mass density [linear g/cm^3]
        if "metaldens" in field.lower():
            fracFieldName = "metal" # e.g. "metaldens" = total metal mass density
            if "_" in field: # e.g. "metaldens_O" or "metaldens_Mg"
                fracFieldName = "metals_" + field.split("_")[1].capitalize()

            dens = snapshotSubset(sP, partType, 'dens', **kwargs)
            dens = sP.units.codeDensToPhys(dens,cgs=True)
            return dens * snapshotSubset(sP, partType, fracFieldName, **kwargs)

        # gravitational potential, in linear [(km/s)^2]
        if field.lower() in ['gravpot','gravpotential']:
            pot = snapshotSubset(sP, partType, 'Potential', **kwargs)
            return pot * sP.units.scalefac

        # GFM_MetalsTagged: ratio of iron mass [linear] produced in SNIa versus SNII
        if field.lower() in ['sn_iaii_ratio_fe']:
            metals_FeSNIa = snapshotSubset(sP, partType, 'metals_FeSNIa', **kwargs)
            metals_FeSNII = snapshotSubset(sP, partType, 'metals_FeSNII', **kwargs)
            return ( metals_FeSNIa / metals_FeSNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus SNII
        if field.lower() in ['sn_iaii_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_SNII = snapshotSubset(sP, partType, 'metals_SNII', **kwargs)
            return ( metals_SNIa / metals_SNII )

        # GFM_MetalsTagged: ratio of total metals [linear] produced in SNIa versus AGB stars
        if field.lower() in ['sn_ia_agb_ratio_metals']:
            metals_SNIa = snapshotSubset(sP, partType, 'metals_SNIa', **kwargs)
            metals_AGB = snapshotSubset(sP, partType, 'metals_AGB', **kwargs)
            return ( metals_SNIa / metals_AGB )

        # sound speed (hydro only version) [physical km/s]
        if field.lower() in ['csnd','soundspeed']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            return sP.units.calcSoundSpeedKmS(u,dens)

        # cooling time (computed from saved GFM_CoolingRate) [Gyr]
        if field.lower() in ['tcool','cooltime']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            coolrate = snapshotSubset(sP, partType, 'GFM_CoolingRate', **kwargs)
            return sP.units.coolingTimeGyr(dens, coolrate, u)

        # total effective timestep, from snapshot [years]
        if field.lower() == 'dt_yr':
            dt = snapshotSubset(sP, 'gas', 'TimeStep', **kwargs)
            return sP.units.codeTimeStepToYears(dt)

        # gas cell hydrodynamical timestep [years]
        if field.lower() == 'dt_hydro_yr':
            soundspeed = snapshotSubset(sP, 'gas', 'soundspeed', **kwargs)
            cellrad = snapshotSubset(sP, 'gas', 'cellrad', **kwargs)
            cellrad_kpc = sP.units.codeLengthToKpc(cellrad)
            cellrad_km  = cellrad_kpc * sP.units.kpc_in_km

            dt_hydro_s = sP.units.CourantFac * cellrad_km / soundspeed
            dt_yr = dt_hydro_s / sP.units.s_in_yr
            return dt_yr

        # ratio of (cell mass / sfr / timestep), either hydro-only or actual timestep [dimensionless linear ratio]
        if field.lower() in ['mass_sfr_dt','mass_sfr_dt_hydro']:
            mass = snapshotSubset(sP, 'gas', 'mass', **kwargs)
            mass = sP.units.codeMassToMsun(mass)
            sfr  = snapshotSubset(sP, 'gas', 'sfr', **kwargs)

            dt_type = 'dt_hydro_yr' if '_hydro' in field.lower() else 'dt_yr'
            dt = snapshotSubset(sP, 'gas', dt_type, **kwargs)

            return (mass / sfr / dt)

        # ------------------------------------------------------------------------------------------------------
        # halo centric analysis (currently require one explicit haloID/suhaloID)
        # (TODO: generalize to e.g. full snapshots)
        # should be able to do this with an inverse mapping of indRange to subhaloIDs (check the case of indRange==None 
        # mapping correctly to all). Then, replace groupCatSingle()'s with global groupCat() loads, and loop over 
        # each subhaloID, for each call the appropriate unit function with the group/subhalo particle subset and obj 
        # position. note: decision for satellite subhalos, properties are relative to themselves or to their central.
        # ------------------------------------------------------------------------------------------------------

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

            rad = periodicDists(haloPos, pos, sP)
            if '_kpc' in field: rad = sP.units.codeLengthToKpc(rad)
            if '_rvir' in field: rad = rad / halo['Group_R_Crit200']

            return rad

        # radial velocity, negative=inwards, relative to the central subhalo pos/vel, including hubble correction [km/s]
        if field.lower() in ['vrad','halo_vrad','radvel','halo_radvel']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            pos = snapshotSubset(sP, partType, 'pos', **kwargs)
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(pos, dict) and pos['count'] == 0: return pos # no particles of type, empty return

            shID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
            firstSub = groupCatSingle(sP, subhaloID=shID)

            return sP.units.particleRadialVelInKmS(pos, vel, firstSub['SubhaloPos'], firstSub['SubhaloVel'])

        # velocity 3-vector, relative to the central subhalo pos/vel, [km/s] for each component
        if field.lower() in ['vrel','halo_vrel','relvel','halo_relvel','relative_vel']:
            if sP.isZoom:
                subhaloID = sP.zoomSubhaloID
                print('WARNING: snapshotSubset() using zoomSubhaloID [%d] for zoom run to compute [%s]!' % (subhaloID,field))
            assert haloID is not None or subhaloID is not None
            vel = snapshotSubset(sP, partType, 'vel', **kwargs)

            if isinstance(vel, dict) and vel['count'] == 0: return vel # no particles of type, empty return

            shID = groupCatSingle(sP, haloID=haloID)['GroupFirstSub'] if subhaloID is None else subhaloID
            firstSub = groupCatSingle(sP, subhaloID=shID)

            return sP.units.particleRelativeVelInKmS(vel, firstSub['SubhaloVel'])

        if field.lower() in ['vrelmag','halo_vrelmag','relvelmag','halo_relvelmag','relative_velmag','relative_vmag']:
            vel = snapshotSubset(sP, partType, 'halo_relvel', **kwargs)
            return np.sqrt( vel[:,0]**2 + vel[:,1]**2 + vel[:,2]**2 )

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
                return sP.units.particleSpecAngMomMagInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])
            if '_vec' in field.lower():
                return sP.units.particleAngMomVecInKpcKmS(pos, vel, mass, firstSub['SubhaloPos'], firstSub['SubhaloVel'])

        # TODO: DM particle mass (use stride_tricks to allow virtual DM 'Masses' load)
        # http://stackoverflow.com/questions/13192089/fill-a-numpy-array-with-the-same-number

    # alternate field names mappings
    altNames = [ [['center_of_mass','com'], 'Center-of-Mass'],
                 [['xyz','positions','pos'], 'Coordinates'],
                 [['dens','rho'], 'Density'],
                 [['ne','nelec'], 'ElectronAbundance'],
                 [['agnrad','gfm_agnrad'], 'GFM_AGNRadiation'],
                 [['coolrate','gfm_coolrate'], 'GFM_CoolingRate'],
                 [['winddmveldisp'], 'GFM_WindDMVelDisp'],
                 [['metal','Z','gfm_metal','metallicity'], 'GFM_Metallicity'],
                 [['metals'], 'GFM_Metals'],
                 [['u'], 'InternalEnergy'],
                 [['machnum','shocks_machnum'], 'Machnumber'],
                 [['dedt','energydiss','shocks_dedt','shocks_energydiss'], 'EnergyDissipation'],
                 [['b','bfield'], 'MagneticField'],
                 [['mass'], 'Masses'],
                 [['nh'], 'NeutralHydrogenAbundance'],
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
            if field.lower() in altLabels: # alternate field name map
                fields[i] = toLabel            
            if field == toLabel.lower(): # lowercase versions accepted
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
                #print('Multi-dimensional slice load: ' + field + ' -> ' + \
                #      multiDimMap['field'] + ' [mdi=' + str(multiDimMap['fN']) + ']')

                fields[i] = multiDimMap['field']
                mdi[i] = multiDimMap['fN']

    if sum(m is not None for m in mdi) > 1:
        raise Exception('Not supported for multiple MDI at once.')

    # halo or subhalo based subset
    if haloID is not None or subhaloID is not None:
        subset = _haloOrSubhaloSubset(sP, haloID=haloID, subhaloID=subhaloID)

    # check memory cache (only simplest support at present, for indRange returns of global cache)
    if len(fields) == 1 and mdi[0] is None and indRange is not None:
        cache_key = 'snap%s_%s_%s' % (sP.snap,partType,fields[0].replace(" ","_"))
        if cache_key in sP.data:
            print('NOTE: Returning [%s] from cache, indRange [%d - %d]!' % (cache_key,indRange[0],indRange[1]))
            return sP.data[cache_key][indRange[0]:indRange[1]+1]

    # load from disk
    r = il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset, mdi=mdi, sq=sq)

    # optional unit post-processing
    if isinstance(r, np.ndarray) and len(fieldsOrig) == 1:
        if fieldsOrig[0] in ['tracer_maxent','tracer_maxtemp'] and r.max() < 20.0:
            raise Exception('Unexpectedly low max for non-log values, something maybe changed.')
            
        if fieldsOrig[0] == 'tracer_maxent':
            r = sP.units.tracerEntToCGS(r, log=True) # [log cgs] = [log K cm^2]
        if fieldsOrig[0] == 'tracer_maxtemp':
            r = logZeroNaN(r) # [log Kelvin]

    return r
