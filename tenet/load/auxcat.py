"""
Loading I/O - postprocessing auxcats, and the full list of specified auxcatalogs.
"""
import numpy as np
import h5py
import datetime
from functools import partial
from os.path import isfile, isdir
from os import mkdir, unlink
from getpass import getuser

from .snapshot import snapshotHeader
from ..util.helper import iterable, curRepoVersion

# generative functions
from ..cosmo.auxcatalog import fofRadialSumType, subhaloRadialReduction, subhaloStellarPhot, \
  wholeBoxColDensGrid, wholeBoxCDDF, mergerTreeQuant, tracerTracksQuant, subhaloCatNeighborQuant, \
  subhaloRadialProfile

from ..projects.outflows_analysis import instantaneousMassFluxes, massLoadingsSN, outflowVelocities
from ..projects.rshock import healpixThresholdedRadius

# save these as separate datasets, if present
largeAttrNames = ['subhaloIDs','partInds','wavelength']

# threshold size of elements per subhalo such that we save a condensed (non-full) auxCat
condThresh = 50

def auxCat(sP, fields=None, pSplit=None, reCalculate=False, searchExists=False, indRange=None, 
           subhaloIDs=None, onlyMeta=False, expandPartial=False):
    """ Load field(s) from the auxiliary group catalog, computing missing datasets on demand. 

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      fields (str or list[str]): requested field(s) by name.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      reCalculate (bool): force redo of computation now, even if data is already saved in catalog.
      searchExists (bool): return None if data is not already computed, i.e. do not calculate right now.
      indRange (list[int][2]): if a tuple/list, load only the specified range of data (field and e.g. subhaloIDs).
      subhaloIDs (ndarray[int]): if a tuple/list, load only the requests subhaloIDs (assumed sparse).
      onlyMeta (bool): load only attributes and coverage information.
      expandPartial (bool): if data was only computed for a subset of all subhalos, expand this now into a total nSubs sized array.

    Return:
      dict: A catalog dict with three entries for each ``field``, which are 'field' (ndarray), 'field_attrs' 
      (dict of metadata), and 'subhaloIDs', which correspond to the first dimension of the 'field' array.

    Note:
      Catalogs constructed without pSplit parallelism will be saved sparse if possible, with an array whose 
      size is smaller than sP.numSubhalos and with subhaloIDs providing the mapping. However, catalogs 
      constructed with pSplit parallelism will usually be expanded to full rather than sparse 
      representations (denoted by writeSparseCalcFullSize). In this second case, unprocessed always have 
      a result of np.nan which should always be filtered out, each with a corresponding entry of -1 in 
      subhaloIDs.
    """
    assert np.sum([el is not None for el in [indRange,subhaloIDs]]) in [0,1] # specify at most one

    epStr = '_ep' if expandPartial else ''
    if len(iterable(fields)) == 1 and 'ac_'+iterable(fields)[0]+epStr in sP.data \
      and not reCalculate and not onlyMeta and not searchExists and indRange is None and subhaloIDs is None:
        return sP.data['ac_'+iterable(fields)[0]+epStr].copy() # cached, avoid view

    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No auxCat() for subbox snapshots."

    pathStr1 = sP.derivPath + 'auxCat/%s_%03d.hdf5'
    pathStr2 = sP.derivPath + 'auxCat/%s_%03d-split-%d-%d.hdf5'

    r = {}

    if not isdir(sP.derivPath + 'auxCat'):
        mkdir(sP.derivPath + 'auxCat')

    for field in iterable(fields):
        if field not in list(fieldComputeFunctionMapping.keys()) + manualFieldNames:
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
                    ret = _concatSplitFiles(sP, pSplit, field, readField, 
                                            auxCatPath, auxCatPathSplit, pathStr2)
                    if ret is False:
                        return # requested, but not all, exist

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

                # load specific subhalos?
                if subhaloIDs is not None:
                    from ..tracer.tracerMC import match3
                    
                    subhaloIDs = iterable(subhaloIDs)
                    subIDs_file = f['subhaloIDs'][()]
                    subInds_file, _ = match3(subIDs_file, subhaloIDs)
                    assert subInds_file.size == len(subhaloIDs), 'Failed to find all subhaloIDs in auxCat!'

                # load data
                if not onlyMeta:
                    rr = []

                    for readField in readFields:
                        # read subset or entire dataset
                        if indRange is not None:
                            data = f[readField][indRange[0]:indRange[1],...]
                            data = np.squeeze(data) # remove any degenerate dimensions
                        elif subhaloIDs is not None:
                            data = f[readField][subInds_file,...]
                            data = np.squeeze(data)
                        else:
                            data = f[readField][()]

                        rr.append(data)

                    # read 1 or more datasets, keep list only if >1
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
                        if indRange is not None:
                            r[attrName] = f[attrName][indRange[0]:indRange[1]]
                        elif subhaloIDs is not None:
                            r[attrName] = f[attrName][subInds_file]
                        else:
                            r[attrName] = f[attrName][()]
                            
            # subhaloIDs indicates computation only for partial set of subhalos?
            if expandPartial:
                assert len(readFields) == 1
                r[field] = _expand_partial(sP, r, field)

            # announce warning if return size does not match full subhalo catalog size
            if not expandPartial:
                if isinstance(r[field],list):
                    # RadialMassFlux/special auxCats with multiple datasets
                    checkSize = r[field][0].shape[0]
                else:
                    checkSize = r[field].shape[0]

                if field.startswith('Subhalo_'):
                    verifySize = sP.numSubhalos
                elif field.startswith('Group_'):
                    verifySize = sP.numHalos
                else:
                    verifySize = 0 # can generalize

                if checkSize != verifySize:
                    print(f'WARNING: Return partial auxCat [{field}] without expanding to fill!')

            # cache
            sP.data['ac_'+field+epStr] = {}
            for key in r:
                sP.data['ac_'+field+epStr][key] = r[key]

            continue

        # either does not exist yet, or reCalculate requested
        if field in manualFieldNames:
            raise Exception('Error: auxCat [%s] does not yet exist, but must be manually created.' % field)
            
        pSplitStr = ''
        savePath = auxCatPath

        if pSplit is not None:
            pSplitStr = ' (split %d of %d)' % (pSplit[0],pSplit[1])
            savePath = auxCatPathSplit

        print('Compute and save: [%s] [%s]%s' % (sP.simName,field,pSplitStr))

        r[field], attrs = fieldComputeFunctionMapping[field] (sP, pSplit)
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
            r[field] = _expand_partial(sP, r, field)

    return r

def _comparatorListInds(fieldName):
    """ Transform e.g. 'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_13' into 13 (an integer) 
    for sorting comparison. """
    num = fieldName.rsplit('_', 1)[-1]
    return int(num)

def _expand_partial(sP, r, field):
    """ Expand a subhalo-partial aC into a subhalo-complete array. """
    nSubsTot = sP.numSubhalos

    assert 'subhaloIDs' in r # else check why we are here

    if r['subhaloIDs'].size == nSubsTot:
        return r[field]

    assert r['subhaloIDs'].min() >= 0, 'Have a -1 entry in subhaloIDs, should not occur.'
    
    if r['subhaloIDs'].size < nSubsTot:
        shape = np.array(r[field].shape)
        shape[0] = nSubsTot
        new_data = np.zeros( shape, dtype=r[field].dtype )
        new_data.fill(np.nan)
        new_data[r['subhaloIDs'],...] = r[field]
        print(' Auxcat Expanding [%d] to [%d] elements for [%s].' % (r[field].shape[0],new_data.shape[0],field))
        return new_data

def _concatSplitFiles(sP, pSplit, field, datasetName, auxCatPath, auxCatPathSplit, pathStr2):
    """ Concatenate a number of partial auxCat files into a single completed file. """
    allExist = True
    allCount = 0
    allCountPart = 0

    # specified chunk exists, do all exist? check and record sizes
    for i in range(pSplit[1]):
        auxCatPathSplit_i = pathStr2 % (field,sP.snap,i,pSplit[1])
        if not isfile(auxCatPathSplit_i):
            allExist = False
            continue

        # record counts and dataset shape
        with h5py.File(auxCatPathSplit_i,'r') as f:
            allShape = f[datasetName].shape
            if f[datasetName].ndim == 0: # scalar, i.e. single subhalo
                allShape = [f[datasetName].size]

            nElemPerSub = np.prod(allShape[1:])

            if len(allShape) > 1 and nElemPerSub > condThresh:
                # high dimensionality auxCat, save condensed
                assert 'partInds' not in f or f['partInds'][()] == -1

                # sparse large data (e.g. full spectra), save only non-nan
                if 'Subhalo_SDSSFiberSpectra' in field:
                    temp_r = f[datasetName][()]
                    w = np.where( np.isfinite(np.sum(temp_r,axis=1)) )
                    allCount += len(w[0])
                    print(' [%2d] keeping %d of %d non-nan.' % (i,len(w[0]),f['subhaloIDs'].size))
                else:
                    allCount += allShape[0]
                    print(' [%2d] saving all %d elements condensed.' % (i,allShape[0]))
            else:
                # normal, save full Subhalo size
                allCount += f['subhaloIDs'].size
                if 'partInds' in f: allCountPart += f['partInds'].size

    if not allExist:
        print('Chunk [%s] already exists, but all not yet done, exiting.' % auxCatPathSplit)
        #r[field] = None
        return False
        
    # for full saves we want (auxCat size is groupcat size), assuming we computed a subset of objects
    numSubs = sP.groupCatHeader()['Nsubgroups_Total']
    writeSparseCalcFullSize = False
    if numSubs > allCount and nElemPerSub <= condThresh:
        print('Note: Increasing save size from [%d], shape = %s computed, to full groupcat size [%d].' % (allCount,allShape,numSubs))
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

            if len(allShape) > 1 and nElemPerSub > condThresh:
                # want to condense, saving only a subset of subhalos, stamp in to dense indices
                subhaloIDsToSave = f[catIndFieldName][()]

                if 'Subhalo_SDSSFiberSpectra' in field:
                    # splits saved all subhalos (sdss spectra)
                    w = np.where( np.isfinite(np.sum(temp_r,axis=1)) )[0]
                    length = len(w)
                    temp_r = temp_r[w,:]
                    subhaloIDsToSave = subhaloIDsToSave[w]

                subhaloIDs[offset : offset+length] = subhaloIDsToSave
                subhaloIndsStamp = np.arange(offset,offset+length)
            else:
                # full, save all subhalos, stamp in to indices corresponding to subhalo indices
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
    return True

# common particle-level restrictions
sfrgt0 = {'StarFormationRate':['gt',0.0]}
sfreq0 = {'StarFormationRate':['eq',0.0]}

# common option sets
sphericalSamplesOpts = {'op':'kernel_mean', 'scope':'global_spatial',
                        'radMin':0.0, 'radMax':5.0, 'radNumBins':400, 'Nside':16, 'Nngb':20}

# this dictionary contains a mapping between all auxCatalogs and their generating functions, where the 
# first sP,pSplit inputs are stripped out with a partial func and the remaining arguments are hardcoded
fieldComputeFunctionMapping = \
  {'Group_Mass_Crit500_Type' : \
     partial(fofRadialSumType,ptProperty='Masses',ptType='all',rad='Group_R_Crit500'),
   'Group_XrayBolLum_Crit500' : \
     partial(fofRadialSumType,ptProperty='xray_lum',ptType='gas',rad='Group_R_Crit500'),
   'Group_XrayLum_05-2kev_Crit500' : \
     partial(fofRadialSumType,ptProperty='xray_lum_05-2kev',ptType='gas',rad='Group_R_Crit500'),
   'Group_XrayLum_0.5-2.0kev_Crit500' : \
     partial(fofRadialSumType,ptProperty='xray_lum_0.5-2.0kev',ptType='gas',rad='Group_R_Crit500'),

   # subhalo: masses
   'Subhalo_Mass_5pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=5.0),
   'Subhalo_Mass_5pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=5.0),
   'Subhalo_Mass_5pkpc_DM' : \
     partial(subhaloRadialReduction,ptType='dm',ptProperty='Masses',op='sum',rad=5.0),
   'Subhalo_Mass_5pkpc_BH' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='Masses',op='sum',rad=5.0),

   'Subhalo_Mass_25pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=25.0),
   'Subhalo_Mass_30pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=30.0),
   'Subhalo_Mass_2rhalf_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='2rhalfstars'),
   'Subhalo_Mass_100pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=100.0,minStellarMass=10.0),
   'Subhalo_Mass_min_30pkpc_2rhalf_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='30h'),
   'Subhalo_Mass_puchwein10_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='p10'),
   'Subhalo_Mass_SFingGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestrictions=sfrgt0),

   'Subhalo_Mass_30pkpc_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=30.0),
   'Subhalo_Mass_100pkpc_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=100.0),
   'Subhalo_Mass_2rstars_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad='2rhalfstars'),
   'Subhalo_Mass_FoF_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None,scope='fof',cenSatSelect='cen'),
   'Subhalo_Mass_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None),

   'Subhalo_Mass_2rstars_MHI_GK' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MHI_GK',op='sum',rad='2rhalfstars'),
   'Subhalo_Mass_70pkpc_MHI_GK' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MHI_GK',op='sum',rad=70.0),
   'Subhalo_Mass_FoF_MHI_GK' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MHI_GK',op='sum',rad=None,scope='fof',cenSatSelect='cen'),

   'Subhalo_Mass_10pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_Mass_10pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_Mass_10pkpc_DM' : \
     partial(subhaloRadialReduction,ptType='dm',ptProperty='Masses',op='sum',rad=10.0),
   'Subhalo_EscapeVel_10pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='vesc',op='mean',rad='10pkpc_shell'),
   'Subhalo_EscapeVel_rvir_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='vesc',op='mean',rad='rvir_shell'),
   'Subhalo_Potential_10pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Potential',op='mean',rad='10pkpc_shell'),
   'Subhalo_Potential_rvir_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Potential',op='mean',rad='rvir_shell'),

   'Subhalo_Mass_HI_GK' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MHI_GK',op='sum',rad=None),
   'Subhalo_Mass_MgII' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Mg II mass',op='sum',rad=None),
   'Subhalo_Mass_OV' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O V mass',op='sum',rad=None),
   'Subhalo_Mass_OVI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VI mass',op='sum',rad=None),
   #'Group_Mass_OVI' : \
   #  partial(subhaloRadialReduction,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof'),
   'Subhalo_Mass_OVII' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VII mass',op='sum',rad=None),
   'Subhalo_Mass_OVIII' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O VIII mass',op='sum',rad=None),

   'Subhalo_Mass_AllGas_Mg' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_Mg',op='sum',rad=None),
   'Subhalo_Mass_AllGas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad=None),
   'Subhalo_Mass_AllGas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None),
   'Subhalo_Mass_AllGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None),
   'Subhalo_Mass_SF0Gas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad=None,ptRestrictions=sfreq0),
   'Subhalo_Mass_SF0Gas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestrictions=sfreq0),
   'Subhalo_Mass_SF0Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad=None,ptRestrictions=sfreq0),

   'Subhalo_Mass_SFGas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestrictions=sfrgt0),
   'Subhalo_Mass_SFGas_Hydrogen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_H',op='sum',rad=None,ptRestrictions=sfrgt0),
   'Subhalo_Mass_SFGas_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None,ptRestrictions=sfrgt0),
   'Subhalo_Mass_nHgt05_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestrictions={'nh':['gt',0.05]}),
   'Subhalo_Mass_nHgt05_Hydrogen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_H',op='sum',rad=None,ptRestrictions={'nh':['gt',0.05]}),
   'Subhalo_Mass_nHgt05_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None,ptRestrictions={'nh':['gt',0.05]}),
   'Subhalo_Mass_nHgt025_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad=None,ptRestrictions={'nh':['gt',0.025]}),
   'Subhalo_Mass_nHgt025_Hydrogen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_H',op='sum',rad=None,ptRestrictions={'nh':['gt',0.025]}),
   'Subhalo_Mass_nHgt025_HI' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='sum',rad=None,ptRestrictions={'nh':['gt',0.025]}),

   'Subhalo_Mass_HaloGas_Oxygen' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass_O',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGas_Metal' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='metalmass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGas_Cold' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad='r015_1rvir_halo',ptRestrictions={'temp_log':['lt',4.5]}),
   'Subhalo_Mass_HaloGas_SFCold' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',rad='r015_1rvir_halo',ptRestrictions={'temp_sfcold_log':['lt',4.5]}),
   'Subhalo_Mass_HaloStars_Metal' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metalmass',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloStars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='mass',op='sum',rad='r015_1rvir_halo'),

   'Subhalo_Mass_HaloGasFoF' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',scope='fof',cenSatSelect='cen',rad='r015_1rvir_halo'),
   'Subhalo_Mass_HaloGasFoF_SFCold' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='sum',scope='fof',cenSatSelect='cen',rad='r015_1rvir_halo',ptRestrictions={'temp_sfcold_log':['lt',4.5]}),

   'Subhalo_Mass_50pkpc_Gas': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=50.0),
   'Subhalo_Mass_50pkpc_Stars': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=50.0),
   'Subhalo_Mass_250pkpc_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=250.0),
   'Subhalo_Mass_250pkpc_Gas_Global' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad=250.0,scope='global'),
   'Subhalo_Mass_250pkpc_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=250.0),
   'Subhalo_Mass_250pkpc_Stars_Global' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad=250.0,scope='global'),
   'Subhalo_Mass_r200_Gas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad='r200crit'),
   'Subhalo_Mass_r200_Gas_Global' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad='r200crit',scope='global',minStellarMass=9.0),

   'Subhalo_Mass_r500_Gas_FoF': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Masses',op='sum',rad='r500crit',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),
   'Subhalo_Mass_r500_Stars_FoF': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='r500crit',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),

   'Subhalo_Mass_1pkpc_2D_Stars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Masses',op='sum',rad='1pkpc_2d'),

   # cooling properties
   'Subhalo_CoolingTime_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',rad='r015_1rvir_halo',ptRestrictions=sfreq0),
   'Subhalo_CoolingTime_OVI_HaloGas' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='tcool',op='mean',weighting='O VI mass',rad='r015_1rvir_halo',ptRestrictions=sfreq0),

   # star formation rates
   'Subhalo_StellarMassFormed_10myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='initialmass',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.01]}),
   'Subhalo_StellarMassFormed_50myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='initialmass',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.05]}),
   'Subhalo_StellarMassFormed_100myr_30pkpc': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='initialmass',op='sum',rad=30.0,ptRestrictions={'stellar_age':['lt',0.1]}),
   'Subhalo_GasSFR_30pkpc': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='sfr',op='sum',rad=30.0),

   # sizes
   'Subhalo_Gas_SFR_HalfRad': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='sfr',op='halfrad',rad=None),
   'Subhalo_Gas_Halpha_HalfRad': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='halpha_lum',op='halfrad',rad=None),
   'Subhalo_Gas_HI_HalfRad': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='HI mass',op='halfrad',rad=None),
   'Subhalo_Gas_Dist256': \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='mass',op='dist256',scope='fof',cenSatSelect='cen',minStellarMass=9.0,rad=None),
   'Subhalo_Stars_R50': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='mass',op='halfrad',rad=None),
   'Subhalo_Stars_R80': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='mass',op='rad80',rad=None),

   # emission: x-rays
   'Subhalo_XrayBolLum' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad=None),
   'Subhalo_XrayLum_05-2kev' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum_05-2kev',op='sum',rad=None), 
   'Subhalo_XrayLum_0.5-2.0kev' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum_0.5-2.0kev',op='sum',rad=None), 
   'Subhalo_XrayBolLum_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum',op='sum',rad='2rhalfstars'),
   'Subhalo_XrayLum_05-2kev_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='xray_lum_05-2kev',op='sum',rad='2rhalfstars'),

   'Subhalo_OVIIr_GalaxyLum_30pkpc' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O  7 21.6020A lum2phase',op='sum',rad=30.0,ptRestrictions=sfrgt0),
   'Subhalo_OVIIr_DiffuseLum_30pkpc' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='O  7 21.6020A lum2phase',op='sum',rad=30.0,ptRestrictions=sfreq0),

   # emission
   'Subhalo_S850um' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='s850um_flux',op='sum',rad=None),
   'Subhalo_S850um_25pkpc' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='s850um_flux',op='sum',rad=25.0),

   'Subhalo_MgII_Lum_DustDepleted' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='sum',rad=None),
   'Subhalo_MgII_LumSize_DustDepleted' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='halfrad',rad=None),
   'Subhalo_MgII_LumConcentration_DustDepleted' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='concentration',rad=None),

   'Subhalo_SynchrotronPower_SKA' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska',op='sum',rad=None),
   'Subhalo_SynchrotronPower_SKA_eta43' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska_eta43',op='sum',rad=None),
   'Subhalo_SynchrotronPower_SKA_alpha15' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_ska_alpha15',op='sum',rad=None),
   'Subhalo_SynchrotronPower_VLA' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_sync_vla',op='sum',rad=None),

   # black holes
   'Subhalo_BH_Mass_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mass',op='max',rad=None),
   'Subhalo_BH_Mdot_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_Mdot',op='max',rad=None),
   'Subhalo_BH_MdotEdd_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_MdotEddington',op='max',rad=None),
   'Subhalo_BH_BolLum_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_BolLum',op='max',rad=None),
   'Subhalo_BH_BolLum_basic_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_BolLum_basic',op='max',rad=None),
   'Subhalo_BH_EddRatio_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_EddRatio',op='max',rad=None),
   'Subhalo_BH_dEdt_largest' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_dEdt',op='max',rad=None),
   'Subhalo_BH_mode' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_mode',op='mean',rad=None), # if not zero or unity, >1 BH

   # wind-model
   'Subhalo_Gas_Wind_vel' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_vel',op='mean',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_dEdt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_dEdt',op='sum',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_dPdt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_dPdt',op='sum',rad='2rhalfstars'),
   'Subhalo_Gas_Wind_etaM' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='wind_etaM',op='mean',rad='2rhalfstars'),

   # kinematics and morphology
   'Subhalo_StellarRotation' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad=None),
   'Subhalo_StellarRotation_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad='2rhalfstars'),
   'Subhalo_StellarRotation_1rhalfstars' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='Krot',op='ufunc',rad='1rhalfstars'),
   'Subhalo_GasRotation' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Krot',op='ufunc',rad=None),
   'Subhalo_GasRotation_2rhalfstars' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='Krot',op='ufunc',rad='2rhalfstars'),

   'Subhalo_EllipsoidShape_Stars_1rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='shape_ellipsoid_1r',op='ufunc',weighting='mass',rad=None),
   'Subhalo_EllipsoidShape_Gas_SFRgt0_1rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='shape_ellipsoid_1r',op='ufunc',weighting='mass',ptRestrictions=sfrgt0,rad=None),
   'Subhalo_EllipsoidShape_Stars_2rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='shape_ellipsoid',op='ufunc',weighting='mass',rad=None),
   'Subhalo_EllipsoidShape_Gas_SFRgt0_2rhalfstars_shell' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='shape_ellipsoid',op='ufunc',weighting='mass',ptRestrictions=sfrgt0,rad=None),

   # stellar age/metallicity
   'Subhalo_StellarAge_NoRadCut_MassWt'       : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='mass'),
   'Subhalo_StellarAge_NoRadCut_rBandLumWt' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=4.0,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_SDSSFiber_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_SDSSFiber4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='sdss_fiber_4pkpc',weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad=4.0,weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_SDSSFiber_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber',weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_SDSSFiber4pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='sdss_fiber_4pkpc',weighting='bandLum-sdss_r'),
   'Subhalo_StellarZ_2rhalf_rBandLumWt': \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='metal',op='mean',rad='2rhalfstars',weighting='bandLum-sdss_r',minHaloMass='1000dm'),

   'Subhalo_StellarAge_2rhalf_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='2rhalfstars',weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_30pkpc_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=30.0,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_NoRadCut_rBandLumWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='bandLum-sdss_r'),
   'Subhalo_StellarAge_2rhalf_MassWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_StellarAge_30pkpc_MassWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=30.0,weighting='mass'),
   'Subhalo_StellarAge_NoRadCut_MassWt'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='stellar_age',op='mean',rad=None,weighting='mass'),

   'Subhalo_StellarZform_VIMOS_Slit'    : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='z_form',op='mean',rad='legac_slit',weighting='mass'),

   'Subhalo_StellarMeanVel' : \
     partial(subhaloRadialReduction,ptType='stars',ptProperty='vel',op='mean',rad=None,weighting='mass'),

   # magnetic fields
   'Subhalo_Bmag_SFingGas_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='mass',ptRestrictions=sfrgt0),
   'Subhalo_Bmag_SFingGas_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='volume',ptRestrictions=sfrgt0),
   'Subhalo_Bmag_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Bmag_2rhalfstars_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='2rhalfstars',weighting='volume'),
   'Subhalo_Bmag_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Bmag_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_Bmag_subhalo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='mass'),
   'Subhalo_Bmag_subhalo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad=None,weighting='volume'),
   'Subhalo_Bmag_fof_r500_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r500crit',weighting='mass',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),
   'Subhalo_Bmag_fof_r500_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='r500crit',weighting='volume',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),
   'Subhalo_Bmag_fof_halfr500_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='0.5r500crit',weighting='mass',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),
   'Subhalo_Bmag_fof_halfr500_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='bmag',op='mean',rad='0.5r500crit',weighting='volume',scope='fof',cenSatSelect='cen',minHaloMass='10000dm'),

   # CGM gas properties
   'Subhalo_Temp_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='temp',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Temp_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='temp',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_nH_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_nH_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_nH_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='nh',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Gas_RadialVel_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='radvel',op='ufunc',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Gas_RadialVel_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='radvel',op='ufunc',rad='2rhalfstars',weighting='mass'),

   'Subhalo_Pratio_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_Pratio_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_Pratio_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pres_ratio',op='mean',rad='r015_1rvir_halo',weighting='volume'),
   'Subhalo_uB_uKE_ratio_2rhalfstars_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='2rhalfstars',weighting='mass'),
   'Subhalo_uB_uKE_ratio_halo_massWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='r015_1rvir_halo',weighting='mass'),
   'Subhalo_uB_uKE_ratio_halo_volWt' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='B_KE_edens_ratio',op='mean',rad='r015_1rvir_halo',weighting='volume'),

   'Subhalo_Ptot_gas_halo' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_gas',op='sum',rad='r015_1rvir_halo'),
   'Subhalo_Ptot_B_halo' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='p_b',op='sum',rad='r015_1rvir_halo'),

   # light: rest-frame/absolute
   'Subhalo_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none'),
   'Subhalo_StellarPhot_p07c_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00'),
   'Subhalo_StellarPhot_p07c_cf00dust_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00dust_allbands' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', bands='all', minStellarMass=9.0),
   'Subhalo_StellarPhot_p07k_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='none'),
   'Subhalo_StellarPhot_p07k_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='kroupa', dust='cf00'),
   'Subhalo_StellarPhot_p07s_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='none'),
   'Subhalo_StellarPhot_p07s_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='salpeter', dust='cf00'),

   'Subhalo_StellarPhot_p07c_cf00dust_res_eff_ns1' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_eff', Nside=1),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=1),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, # main model, with 12 projections per
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=1, rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_30pkpc' : partial(subhaloStellarPhot, # main model, with 1 projection per
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00dust_z_30pkpc' : partial(subhaloStellarPhot, # model B, with 1 projection per
                                         iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', rad=30.0, minHaloMass='1000dm'),
   'Subhalo_StellarPhot_p07c_cf00dust_z_2rhalf' : partial(subhaloStellarPhot, # model B, with 1 projection per
                                         iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', rad='2rhalfstars', minHaloMass='1000dm'),
   'Subhalo_StellarPhot_p07c_cf00b_dust_res_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00b_res_conv', Nside=1, rad=30.0),
   'Subhalo_StellarPhot_p07c_cf00dust_res3_conv_ns1_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res3_conv', Nside=1, rad=30.0),

   'Subhalo_StellarPhot_p07c_ns8_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=8),
   'Subhalo_StellarPhot_p07c_ns4_demo' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=4),
   'Subhalo_StellarPhot_p07c_ns8_demo_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=8, rad=30.0),
   'Subhalo_StellarPhot_p07c_ns4_demo_rad30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside=4, rad=30.0),

   'Subhalo_HalfLightRad_p07c_nodust' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside=None, sizes=True),
   'Subhalo_HalfLightRad_p07c_nodust_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside='z-axis', sizes=True),
   'Subhalo_HalfLightRad_p07c_nodust_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside='efr2d', sizes=True), 
   'Subhalo_HalfLightRad_p07c_cf00dust_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='efr2d', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_efr2d_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='efr2d', rad=30.0, sizes=True),   
   'Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', rad=100.0, sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', sizes=True, minHaloMass='1000dm'), # main model, with 1 projection per
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='efr2d', sizes=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_efr2d_rad30pkpc' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='efr2d', rad=30.0, sizes=True),

   'Particle_StellarPhot_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00', indivStarMags=True),
   'Particle_StellarPhot_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', indivStarMags=True, Nside='z-axis'),

   # spectral mocks
   'Subhalo_SDSSFiberSpectra_NoVel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis'),
   'Subhalo_SDSSFiberSpectra_Vel_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='sdss_fiber', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=2, Nside='z-axis'),

   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_p07c_cf00dust_res_conv_z_restframe' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', minStellarMass=9.8),

   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_Seeing_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, seeing=0.4, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_NoEm_Seeing_p07s_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='salpeter', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, seeing=0.4, minStellarMass=9.8),

   'Subhalo_LEGA-C_SlitSpectra_NoVel_Em_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit', 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', 
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, emlines=True, minStellarMass=9.8),
   'Subhalo_LEGA-C_SlitSpectra_NoVel_Em_Seeing_p07c_cf00dust_res_conv_z' : partial(subhaloStellarPhot, rad='legac_slit',
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv',
                                         fullSubhaloSpectra=1, Nside='z-axis', redshifted=True, emlines=True, seeing=0.4, minStellarMass=9.8),


   # stellar light: UVJ colors (Donnari)
   'Subhalo_StellarPhot_UVJ_p07c_nodust'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j']),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_5pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad=5.0),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad=30.0),
   'Subhalo_StellarPhot_UVJ_p07c_nodust_2rhalf'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['u','v','2mass_j'], rad='2rhalfstars'),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j']),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_5pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad=5.0),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad=30.0),
   'Subhalo_StellarPhot_UVJ_p07c_cf00dust_res_conv_z_2rhalf'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['u','v','2mass_j'], rad='2rhalfstars'),
   'Subhalo_StellarPhot_vistaK_p07c_cf00dust_res_conv_z_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['vista_k'], rad=30.0),
   'Subhalo_StellarPhot_ugr_p07c_cf00dust_res_conv_z_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['sdss_u','sdss_g','sdss_r'], rad=30.0),

   'Subhalo_StellarPhot_NUV_cfht-i_p07c_nodust_30pkpc'   : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='none', bands=['galex_nuv','cfht_i'], rad=30.0),
   'Subhalo_StellarPhot_NUV_cfht-i_p07c_cf00dust_res_conv_z_30pkpc' : partial(subhaloStellarPhot, 
                                         iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', bands=['galex_nuv','cfht_i'], rad=30.0),                           

   # light: redshifted/apparent
   'Subhalo_StellarPhot_p07c_nodust_red'   : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='none', redshifted=True),
   'Subhalo_StellarPhot_p07c_cf00dust_red' : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='cf00', redshifted=True),
   'Subhalo_StellarPhot_p07c_cf00dust_res_conv_z_rad30pkpc_red' : partial(subhaloStellarPhot,
                                             iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', rad=30.0, redshifted=True),

   'Particle_StellarPhot_p07c_nodust_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='none', indivStarMags=True, redshifted=True),
   'Particle_StellarPhot_p07c_cf00dust_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='cf00', indivStarMags=True, redshifted=True),
   'Particle_StellarPhot_p07c_cf00dust_res_conv_z_red' : partial(subhaloStellarPhot, 
                                            iso='padova07', imf='chabrier', dust='cf00_res_conv', indivStarMags=True, Nside='z-axis', redshifted=True),

   'Subhalo_HalfLightRad_p07c_nodust_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='none', Nside=None, sizes=True, redshifted=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_z_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00', Nside='z-axis', sizes=True, redshifted=True),
   'Subhalo_HalfLightRad_p07c_cf00dust_res_conv_z_red' : \
      partial(subhaloStellarPhot, iso='padova07', imf='chabrier', dust='cf00_res_conv', Nside='z-axis', sizes=True, redshifted=True),

   # fullbox
   'Box_Grid_nHI'            : partial(wholeBoxColDensGrid,species='HI'),
   'Box_Grid_nHI_noH2'       : partial(wholeBoxColDensGrid,species='HI_noH2'),
   'Box_Grid_Z'              : partial(wholeBoxColDensGrid,species='Z'),
   'Box_Grid_nOVI'           : partial(wholeBoxColDensGrid,species='O VI'),
   'Box_Grid_nOVI_10'        : partial(wholeBoxColDensGrid,species='O VI 10'),
   'Box_Grid_nOVI_25'        : partial(wholeBoxColDensGrid,species='O VI 25'),
   'Box_Grid_nOVI_solar'     : partial(wholeBoxColDensGrid,species='O VI solar'),
   'Box_Grid_nOVII'          : partial(wholeBoxColDensGrid,species='O VII'),
   'Box_Grid_nOVIII'         : partial(wholeBoxColDensGrid,species='O VIII'),

   'Box_Grid_nH2_BR_depth10'  : partial(wholeBoxColDensGrid,species='MH2_BR_depth10'),
   'Box_Grid_nH2_GK_depth10'  : partial(wholeBoxColDensGrid,species='MH2_GK_depth10'),
   'Box_Grid_nH2_KMT_depth10' : partial(wholeBoxColDensGrid,species='MH2_KMT_depth10'),
   'Box_Grid_nHI_GK_depth10'  : partial(wholeBoxColDensGrid,species='MHI_GK_depth10'),

   'Box_Grid_nH2_GK'         : partial(wholeBoxColDensGrid,species='MH2_GK'),
   'Box_Grid_nH2_GK_depth20' : partial(wholeBoxColDensGrid,species='MH2_GK_depth20'),
   'Box_Grid_nH2_GK_depth5'  : partial(wholeBoxColDensGrid,species='MH2_GK_depth5'),
   'Box_Grid_nH2_GK_depth1'  : partial(wholeBoxColDensGrid,species='MH2_GK_depth1'),

   'Box_CDDF_nHI'            : partial(wholeBoxCDDF,species='HI'),
   'Box_CDDF_nHI_noH2'       : partial(wholeBoxCDDF,species='HI_noH2'),
   'Box_CDDF_nOVI'           : partial(wholeBoxCDDF,species='OVI'),
   'Box_CDDF_nOVI_10'        : partial(wholeBoxCDDF,species='OVI_10'),
   'Box_CDDF_nOVI_25'        : partial(wholeBoxCDDF,species='OVI_25'),
   'Box_CDDF_nOVII'          : partial(wholeBoxCDDF,species='OVII'),
   'Box_CDDF_nOVIII'         : partial(wholeBoxCDDF,species='OVIII'),

   'Box_CDDF_nH2_BR_depth10'  : partial(wholeBoxCDDF,species='H2_BR_depth10'),
   'Box_CDDF_nH2_GK_depth10'  : partial(wholeBoxCDDF,species='H2_GK_depth10'),
   'Box_CDDF_nH2_KMT_depth10' : partial(wholeBoxCDDF,species='H2_KMT_depth10'),
   'Box_CDDF_nHI_GK_depth10'  : partial(wholeBoxCDDF,species='HI_GK_depth10'),

   'Box_CDDF_nH2_GK'         : partial(wholeBoxCDDF,species='H2_GK'), # fullbox depth
   'Box_CDDF_nH2_GK_depth20' : partial(wholeBoxCDDF,species='H2_GK_depth20'),
   'Box_CDDF_nH2_GK_depth5'  : partial(wholeBoxCDDF,species='H2_GK_depth5'),
   'Box_CDDF_nH2_GK_depth1'  : partial(wholeBoxCDDF,species='H2_GK_depth1'),

   'Box_Grid_nH2_GD14_depth10' : partial(wholeBoxColDensGrid,species='MH2_GD14_depth10'),
   'Box_Grid_nH2_GK11_depth10' : partial(wholeBoxColDensGrid,species='MH2_GK11_depth10'),
   'Box_Grid_nH2_K13_depth10'  : partial(wholeBoxColDensGrid,species='MH2_K13_depth10'),
   'Box_Grid_nH2_S14_depth10'  : partial(wholeBoxColDensGrid,species='MH2_S14_depth10'),
   'Box_CDDF_nH2_GD14_depth10' : partial(wholeBoxCDDF,species='H2_GD14_depth10'),
   'Box_CDDF_nH2_GK11_depth10' : partial(wholeBoxCDDF,species='H2_GK11_depth10'),
   'Box_CDDF_nH2_K13_depth10'  : partial(wholeBoxCDDF,species='H2_K13_depth10'),
   'Box_CDDF_nH2_S14_depth10'  : partial(wholeBoxCDDF,species='H2_S14_depth10'),

   'Box_Grid_nH2_GK_depth10_onlySFRgt0' : partial(wholeBoxColDensGrid,species='MH2_GK_depth10',onlySFR=True),
   'Box_Grid_nH2_GK_depth10_allSFRgt0' : partial(wholeBoxColDensGrid,species='MH2_GK_depth10',allSFR=True),
   'Box_CDDF_nH2_GK_depth10_onlySFRgt0' : partial(wholeBoxCDDF,species='H2_GK_depth10_onlySFRgt0'),
   'Box_CDDF_nH2_GK_depth10_allSFRgt0' : partial(wholeBoxCDDF,species='H2_GK_depth10_allSFRgt0'),

   'Box_Grid_nH2_GK_depth10_gridSize=3.0' : partial(wholeBoxColDensGrid,species='MH2_GK_depth10',gridSize=3.0),
   'Box_Grid_nH2_GK_depth10_gridSize=1.0' : partial(wholeBoxColDensGrid,species='MH2_GK_depth10',gridSize=1.0),
   'Box_Grid_nH2_GK_depth10_gridSize=0.5' : partial(wholeBoxColDensGrid,species='MH2_GK_depth10',gridSize=0.5),

   'Box_CDDF_nH2_GK_depth10_cell3' : partial(wholeBoxCDDF,species='H2_GK_depth10',gridSize=3.0),
   'Box_CDDF_nH2_GK_depth10_cell1' : partial(wholeBoxCDDF,species='H2_GK_depth10',gridSize=1.0),
   'Box_CDDF_nH2_GK_depth10_cell05' : partial(wholeBoxCDDF,species='H2_GK_depth10',gridSize=0.5),

   'Box_Grid_nOVI_depth10'           : partial(wholeBoxColDensGrid,species='O VI_depth10'),
   'Box_Grid_nOVI_10_depth10'        : partial(wholeBoxColDensGrid,species='O VI 10_depth10'),
   'Box_Grid_nOVI_25_depth10'        : partial(wholeBoxColDensGrid,species='O VI 25_depth10'),
   'Box_Grid_nOVII_depth10'          : partial(wholeBoxColDensGrid,species='O VII_depth10'),
   'Box_Grid_nOVIII_depth10'         : partial(wholeBoxColDensGrid,species='O VIII_depth10'),
   'Box_CDDF_nOVI_depth10'           : partial(wholeBoxCDDF,species='OVI_depth10'),
   'Box_CDDF_nOVI_10_depth10'        : partial(wholeBoxCDDF,species='OVI_10_depth10'),
   'Box_CDDF_nOVI_25_depth10'        : partial(wholeBoxCDDF,species='OVI_25_depth10'),
   'Box_CDDF_nOVII_depth10'          : partial(wholeBoxCDDF,species='OVII_depth10'),
   'Box_CDDF_nOVIII_depth10'         : partial(wholeBoxCDDF,species='OVIII_depth10'),

   'Box_Grid_nOVI_solar_depth10'     : partial(wholeBoxColDensGrid,species='O VI solar_depth10'),
   'Box_CDDF_nOVI_solar_depth10'     : partial(wholeBoxCDDF,species='OVI_solar_depth10'),
   'Box_Grid_nOVII_solarz_depth10'    : partial(wholeBoxColDensGrid,species='O VII solarz_depth10'),
   'Box_CDDF_nOVII_solarz_depth10'    : partial(wholeBoxCDDF,species='OVII_solarz_depth10'),
   'Box_Grid_nOVIII_solarz_depth10'   : partial(wholeBoxColDensGrid,species='O VIII solarz_depth10'),
   'Box_CDDF_nOVIII_solarz_depth10'   : partial(wholeBoxCDDF,species='OVIII_solarz_depth10'),
   'Box_Grid_nOVII_solarz_depth125'    : partial(wholeBoxColDensGrid,species='O VII solarz_depth125'),
   'Box_CDDF_nOVII_solarz_depth125'    : partial(wholeBoxCDDF,species='OVII_solarz_depth125'),
   'Box_Grid_nOVII_10_solarz_depth125'    : partial(wholeBoxColDensGrid,species='O VII 10 solarz_depth125'),
   'Box_CDDF_nOVII_10_solarz_depth125'    : partial(wholeBoxCDDF,species='OVII_10_solarz_depth125'),

   'Box_Omega_HI'                    : partial(wholeBoxCDDF,species='H I',omega=True),
   'Box_Omega_H2'                    : partial(wholeBoxCDDF,species='H 2',omega=True),
   'Box_Omega_OVI'                   : partial(wholeBoxCDDF,species='O VI',omega=True),
   'Box_Omega_OVII'                  : partial(wholeBoxCDDF,species='O VII',omega=True),
   'Box_Omega_OVIII'                 : partial(wholeBoxCDDF,species='O VIII',omega=True),

    # temporal
   'Subhalo_SubLink_zForm_mm5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['mm',5,'snap']),
   'Subhalo_SubLink_zForm_ma5' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['ma',5,'snap']),
   'Subhalo_SubLink_zForm_poly7' : partial(mergerTreeQuant,treeName='SubLink',quant='zForm',
                                         smoothing=['poly',7,'snap']),

   'Subhalo_SubLinkGal_isSat_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='isSat_atForm'),
   'Subhalo_SubLinkGal_dmFrac_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='dmFrac_atForm'),
   'Subhalo_SubLinkGal_rad_rvir_atForm' : partial(mergerTreeQuant,treeName='SubLink_gal',quant='rad_rvir_atForm'),

   'Subhalo_Tracers_zAcc_mean'   : partial(tracerTracksQuant,quant='acc_time_1rvir',op='mean',time=None),
   'Subhalo_Tracers_dtHalo_mean' : partial(tracerTracksQuant,quant='dt_halo',op='mean',time=None),
   'Subhalo_Tracers_angmom_tAcc' : partial(tracerTracksQuant,quant='angmom',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_entr_tAcc'   : partial(tracerTracksQuant,quant='entr',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_temp_tAcc'   : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir'),
   'Subhalo_Tracers_tempTviracc_tAcc' : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir',norm='tvir_tacc'),
   'Subhalo_Tracers_tempTvircur_tAcc' : partial(tracerTracksQuant,quant='temp',op='mean',time='acc_time_1rvir',norm='tvir_cur'),

   'Subhalo_BH_CumEgyInjection_Low' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumEgyInjection_RM',op='sum',rad=None),
   'Subhalo_BH_CumEgyInjection_High' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumEgyInjection_QM',op='sum',rad=None),
   'Subhalo_BH_CumMassGrowth_Low' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumMassGrowth_RM',op='sum',rad=None),
   'Subhalo_BH_CumMassGrowth_High' : \
     partial(subhaloRadialReduction,ptType='bhs',ptProperty='BH_CumMassGrowth_QM',op='sum',rad=None),

   # subhalo neighbors/catalog
   'Subhalo_Env_StellarMass_Max_1Mpc' : \
     partial(subhaloCatNeighborQuant,quant='mstar_30pkpc_log',op='max',rad=1000.0,subRestrictions=None,cenSatSelect='cen'),
   'Subhalo_Env_sSFR_Median_1Mpc_Mstar_9-10' : \
     partial(subhaloCatNeighborQuant,quant='ssfr',op='median',rad=1000.0,subRestrictions=[['mstar_30pkpc_log',9.0,10.0]],cenSatSelect='cen'),

   'Subhalo_Env_Closest_Distance_Mstar_Gt8' : \
     partial(subhaloCatNeighborQuant,quant=None,op='closest_rad',subRestrictions=[['mstar_30pkpc_log',8.0,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_d5_Mstar_Gt10' : \
     partial(subhaloCatNeighborQuant,quant=None,op='d5_rad',subRestrictions=[['mstar_30pkpc_log',10.0,np.inf]]),
   'Subhalo_Env_d5_Mstar_Gt8' : \
     partial(subhaloCatNeighborQuant,quant=None,op='d5_rad',subRestrictions=[['mstar_30pkpc_log',8.0,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_d5_Mstar_Gt7' : \
     partial(subhaloCatNeighborQuant,quant=None,op='d5_rad',subRestrictions=[['mstar_30pkpc_log',7.0,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_Closest_Distance_MstarRel_GtHalf' : \
     partial(subhaloCatNeighborQuant,quant=None,op='closest_rad',subRestrictionsRel=[['mstar_30pkpc',0.5,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_d5_MstarRel_GtHalf' : \
     partial(subhaloCatNeighborQuant,quant=None,op='d5_rad',subRestrictionsRel=[['mstar_30pkpc',0.5,np.inf]],cenSatSelect='cen'),

   'Subhalo_Env_Closest_SubhaloID_MstarRel_GtHalf' : \
     partial(subhaloCatNeighborQuant,quant='id',op='closest_quant',subRestrictionsRel=[['mstar_30pkpc',0.5,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_Count_Mstar_Gt8_2rvir' : \
     partial(subhaloCatNeighborQuant,quant=None,op='count',rad='2rvir',subRestrictions=[['mstar_30pkpc_log',8.0,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_Count_Mstar_Gt7_2rvir' : \
     partial(subhaloCatNeighborQuant,quant=None,op='count',rad='2rvir',subRestrictions=[['mstar_30pkpc_log',7.0,np.inf]],cenSatSelect='cen'),
   'Subhalo_Env_Count_MstarRel_GtTenth_2rvir' : \
     partial(subhaloCatNeighborQuant,quant=None,op='count',rad='2rvir',subRestrictionsRel=[['mstar_30pkpc',0.1,np.inf]],cenSatSelect='cen'),

   # radial profiles: oxygen
   'Subhalo_RadProfile3D_Global_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global_fof'), 
   'Subhalo_RadProfile3D_SubfindGlobal_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind_global'), 
   'Subhalo_RadProfile3D_Subfind_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind'),
   'Subhalo_RadProfile3D_FoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='fof'),

   'Subhalo_RadProfile2Dz_2Mpc_Global_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='global_fof',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_Subfind_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='subfind',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_FoF_OVI_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VI mass',op='sum',scope='fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_Global_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='global_fof',proj2D=[2,2000]),
   'Subhalo_RadProfile3D_Global_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_GlobalFoF_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='global_fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_FoF_OVII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII mass',op='sum',scope='fof'),
   'Subhalo_RadProfile3D_FoF_OVII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII flux',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_2Mpc_FoF_OVII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII flux',op='sum',scope='fof',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VII flux',op='sum',scope='global_fof',proj2D=[2,2000]),
   'Subhalo_RadProfile3D_FoF_OVIII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII mass',op='sum',scope='fof'),
   'Subhalo_RadProfile3D_FoF_OVIII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII flux',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_2Mpc_FoF_OVIII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII flux',op='sum',scope='fof',proj2D=[2,2000]),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_OVIII_Flux' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='O VIII flux',op='sum',scope='global_fof',proj2D=[2,2000]),

   # radial profiles
   'Subhalo_RadProfile3D_Global_Gas_O_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass_O',op='sum',scope='global'),
   'Subhalo_RadProfile3D_Global_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='global'),
   'Subhalo_RadProfile3D_Global_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='global'),

   'Subhalo_RadProfile3D_GlobalFoF_MgII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='Mg II mass',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_6Mpc_GlobalFoF_MgII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='Mg II mass',op='sum',scope='global_fof',proj2D=[2,5700]),
   'Subhalo_RadProfile2Dz_30Mpc_GlobalFoF_MgII_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='Mg II mass',op='sum',scope='global_fof',minHaloMass=12.5,proj2D=[2,30500]),

   'Subhalo_RadProfile3D_GlobalFoF_HI_GK_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='MHI_GK',op='sum',scope='global_fof'),
   'Subhalo_RadProfile2Dz_2Mpc_GlobalFoF_HIGK_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='MHI_GK',op='sum',scope='global_fof',proj2D=[2,2000]),

   'Subhalo_RadProfile3D_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global'),
   'Subhalo_RadProfile2Dz_2Mpc_Global_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='global',proj2D=[2,2000]),
   'Subhalo_RadProfile3D_FoF_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Stars_Mass' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='mass',op='sum',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile3D_FoF_DM_Mass' : \
     partial(subhaloRadialProfile,ptType='dm',ptProperty='mass',op='sum',scope='fof'),
   'Subhalo_RadProfile3D_Global_DM_Mass' : \
     partial(subhaloRadialProfile,ptType='dm',ptProperty='mass',op='sum',scope='global_spatial',minHaloMass='1000dm'),

   'Subhalo_RadProfile3D_FoF_SFR' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='sfr',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_SFR' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='sfr',op='sum',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile3D_FoF_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='mass',op='sum',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile3D_FoF_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metal_Mass' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='metalmass',op='sum',scope='fof',proj2D=[2,None]),

   'Subhalo_RadProfile3D_FoF_Gas_Bmag' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='bmag',op='mean',scope='fof'),

   'Subhalo_RadProfile3D_FoF_Gas_Metallicity' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',scope='fof'),
   'Subhalo_RadProfile3D_FoF_Gas_Metallicity_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',weighting='sfr',scope='fof'),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metallicity' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_Metallicity_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='z_solar',op='mean',weighting='sfr',scope='fof',proj2D=[2,None]),

   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVel_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel_abs',op='mean',weighting='sfr',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVel_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel_abs',op='mean',weighting='sfr',scope='fof',proj2D=['edge-on',None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dz_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=[2,None]),
   'Subhalo_RadProfile2Dedgeon_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=['edge-on',None]),
   'Subhalo_RadProfile2Dfaceon_FoF_Gas_LOSVelSigma_sfrWt' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='losvel',op=np.std,weighting='sfr',scope='fof',proj2D=['face-on',None]),

   'Subhalo_RadProfile3D_FoF_Gas_SFR0_CellSizeKpc_Mean' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op='mean',scope='fof',ptRestrictions=sfreq0),
   'Subhalo_RadProfile3D_FoF_Gas_SFR0_CellSizeKpc_Median' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op='median',scope='fof',ptRestrictions=sfreq0),
   'Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Mean' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op='mean',scope='fof'),
   'Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Median' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op='median',scope='fof'),
   'Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Min' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op='min',scope='fof'),
   'Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_p10' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='cellsize_kpc',op=lambda x: np.percentile(x,10),scope='fof'),

   'Subhalo_CGM_Inflow_MeanX' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pos_x',op='mean',rad='r015_1rvir_halo',ptRestrictions={'vrad':['lt',0.0]},scope='fof',cenSatSelect='cen',minStellarMass=1.0),
   'Subhalo_CGM_Inflow_MeanY' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pos_y',op='mean',rad='r015_1rvir_halo',ptRestrictions={'vrad':['lt',0.0]},scope='fof',cenSatSelect='cen',minStellarMass=1.0),
   'Subhalo_CGM_Inflow_MeanZ' : \
     partial(subhaloRadialReduction,ptType='gas',ptProperty='pos_z',op='mean',rad='r015_1rvir_halo',ptRestrictions={'vrad':['lt',0.0]},scope='fof',cenSatSelect='cen',minStellarMass=1.0),

   # radial profiles: 
   'Subhalo_RadProfile3D_Global_Gas_Temp' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='temp_sfcold',op='mean',weighting='mass',scope='global_spatial',radMin=0.0,radMax=2.0,radNumBins=100,radRvirUnits=True,minHaloMass='1000dm'),
   'Subhalo_RadProfile3D_Global_Gas_Tcool' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='tcool',op='mean',weighting='mass',scope='global_spatial',radMin=0.0,radMax=2.0,radNumBins=100,radRvirUnits=True,minHaloMass='1000dm'),

   # spherical sampling/healpix sightlines
   'Subhalo_SphericalSamples_Global_Gas_Temp_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='temp',minHaloMass='10000dm',**sphericalSamplesOpts),
   'Subhalo_SphericalSamples_Global_Gas_Entropy_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='entropy',minHaloMass='10000dm',**sphericalSamplesOpts),
   'Subhalo_SphericalSamples_Global_Gas_ShocksMachNum_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='shocks_machnum',minHaloMass='10000dm',**sphericalSamplesOpts),
   'Subhalo_SphericalSamples_Global_Gas_ShocksEnergyDiss_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='shocks_dedt',minHaloMass='10000dm',**sphericalSamplesOpts),

   'Subhalo_SphericalSamples_Global_Gas_RadVel_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='radvel',minHaloMass='10000dm',**sphericalSamplesOpts),
   'Subhalo_SphericalSamples_Global_Stars_RadVel_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='stars',ptProperty='radvel',minHaloMass='10000dm',**sphericalSamplesOpts),
   'Subhalo_SphericalSamples_Global_DM_RadVel_5rvir_400rad_16ns' : \
     partial(subhaloRadialProfile,ptType='dm',ptProperty='radvel',minHaloMass='10000dm',**sphericalSamplesOpts),

   'Subhalo_SphericalSamples_Global_Gas_Temp_5rvir_400rad_8ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='temp',minHaloMass='10000dm',**{**sphericalSamplesOpts,'Nside':8}),
   'Subhalo_SphericalSamples_Global_Gas_ShocksMachNum_10rvir_800rad_16ns' : \
     partial(subhaloRadialProfile,ptType='gas',ptProperty='shocks_machnum',minHaloMass='10000dm',**{**sphericalSamplesOpts,'radMax':10.0,'radNumBins':800}),

   # shock/splashback radii
   'Subhalo_VirShockRad_Temp_400rad_16ns' : partial(healpixThresholdedRadius,ptType='Gas',quant='Temp',radNumBins=400, Nside=16),
   'Subhalo_VirShockRad_Temp_400rad_8ns'  : partial(healpixThresholdedRadius,ptType='Gas',quant='Temp',radNumBins=400, Nside=8),
   'Subhalo_VirShockRad_Entropy_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='Gas',quant='Entropy',radNumBins=400, Nside=16),
   'Subhalo_VirShockRad_ShocksMachNum_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='Gas',quant='ShocksMachNum',radNumBins=400, Nside=16),
   'Subhalo_VirShockRad_ShocksMachNum_10rvir_800rad_16ns' : partial(healpixThresholdedRadius,ptType='Gas',quant='ShocksMachNum',radNumBins=800, radMax=10, Nside=16),
   'Subhalo_VirShockRad_ShocksEnergyDiss_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='Gas',quant='ShocksEnergyDiss',radNumBins=400, Nside=16),
   'Subhalo_VirShockRad_RadVel_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='Gas',quant='RadVel',radNumBins=400, Nside=16),
   'Subhalo_SplashbackRad_DM_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='DM',quant='RadVel',radNumBins=400, Nside=16),
   'Subhalo_SplashbackRad_Stars_400rad_16ns'  : partial(healpixThresholdedRadius,ptType='Stars',quant='RadVel',radNumBins=400, Nside=16),

   # outflows/inflows
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz'),
   'Subhalo_RadialMassFlux_Global_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='global'),
   'Subhalo_RadialMassFlux_Global_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='global'),

   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_MgII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Mg II mass'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_SiII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Si II mass'),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_NaI' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',massField='Na I mass'),

   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',
                                                            rawMass=True,fluxMass=False,proj2D=True),
   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',
                                                             rawMass=True,fluxMass=False,proj2D=True),
   'Subhalo_RadialMass2DProj_SubfindWithFuzz_Gas_SiII' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',
                                                             rawMass=True,fluxMass=False,proj2D=True,massField='Si II mass'),
   'Subhalo_RadialMass_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',rawMass=True,fluxMass=False),

   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr' : partial(massLoadingsSN,sfr_timescale=100,outflowMethod='instantaneous'),
   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-50myr' : partial(massLoadingsSN,sfr_timescale=50,outflowMethod='instantaneous'),
   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-10myr' : partial(massLoadingsSN,sfr_timescale=10,outflowMethod='instantaneous'),

   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-100myr' : partial(massLoadingsSN,sfr_timescale=100,outflowMethod='instantaneous',massField='MgII'),
   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-50myr' : partial(massLoadingsSN,sfr_timescale=50,outflowMethod='instantaneous',massField='MgII'),
   'Subhalo_MassLoadingSN_MgII_SubfindWithFuzz_SFR-10myr' : partial(massLoadingsSN,sfr_timescale=10,outflowMethod='instantaneous',massField='MgII'),

   'Subhalo_RadialEnergyFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',fluxKE=True,fluxMass=False),
   'Subhalo_RadialEnergyFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',fluxKE=True,fluxMass=False),
   'Subhalo_RadialMomentumFlux_SubfindWithFuzz_Gas' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',fluxP=True,fluxMass=False),
   'Subhalo_RadialMomentumFlux_SubfindWithFuzz_Wind' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',fluxP=True,fluxMass=False),
   'Subhalo_EnergyLoadingSN_SubfindWithFuzz' : partial(massLoadingsSN,outflowMethod='instantaneous',fluxKE=True),
   'Subhalo_MomentumLoadingSN_SubfindWithFuzz' : partial(massLoadingsSN,outflowMethod='instantaneous',fluxP=True),

   'Subhalo_OutflowVelocity_SubfindWithFuzz' : partial(outflowVelocities),
   'Subhalo_OutflowVelocity_MgII_SubfindWithFuzz' : partial(outflowVelocities,massField='MgII'),
   'Subhalo_OutflowVelocity_SiII_SubfindWithFuzz' : partial(outflowVelocities,massField='SiII'),
   'Subhalo_OutflowVelocity_NaI_SubfindWithFuzz' : partial(outflowVelocities,massField='NaI'),

   'Subhalo_OutflowVelocity2DProj_SubfindWithFuzz' : partial(outflowVelocities,proj2D=True),
   'Subhalo_OutflowVelocity2DProj_SiII_SubfindWithFuzz' : partial(outflowVelocities,proj2D=True,massField='SiII'),

   'Subhalo_RadialMassFlux_SubfindWithFuzz_Gas_v200norm' : partial(instantaneousMassFluxes,ptType='gas',scope='subhalo_wfuzz',v200norm=True),
   'Subhalo_RadialMassFlux_SubfindWithFuzz_Wind_v200norm' : partial(instantaneousMassFluxes,ptType='wind',scope='subhalo_wfuzz',v200norm=True),
   'Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr_v200norm' : partial(massLoadingsSN,sfr_timescale=100,outflowMethod='instantaneous',v200norm=True),
   'Subhalo_OutflowVelocity_SubfindWithFuzz_v200norm' : partial(outflowVelocities,v200norm=True),

   'Subhalo_MgII_Emission_Grid2D_Shape' : partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='grid2d_isophot_shape',rad=None,scope='fof',cenSatSelect='cen',minStellarMass=7.0),
   'Subhalo_MgII_Emission_Grid2D_Area' : partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='grid2d_isophot_area',rad=None,scope='fof',cenSatSelect='cen',minStellarMass=7.0),
   'Subhalo_MgII_Emission_Grid2D_Gini' : partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='grid2d_isophot_gini',rad=None,scope='fof',cenSatSelect='cen',minStellarMass=7.0),
   'Subhalo_MgII_Emission_Grid2D_M20' : partial(subhaloRadialReduction,ptType='gas',ptProperty='MgII lum_dustdepleted',op='grid2d_m20',rad=None,scope='fof',cenSatSelect='cen',minStellarMass=7.0),
  }

# this list contains the names of auxCatalogs which are computed manually (e.g. require more work than 
# a single generative function), but are then saved in the same format and so can be loaded normally
manualFieldNames = \
[   'Subhalo_SDSSFiberSpectraFits_NoVel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-NoRealism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_NoVel-Realism_p07c_cf00dust_res_conv_z',
    'Subhalo_SDSSFiberSpectraFits_Vel-Realism_p07c_cf00dust_res_conv_z'
]
