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
from util.helper import iterable, logZeroSafe, curRepoVersion

def auxCat(sP, fields=None, reCalculate=False, searchExists=False):
    """ Load field(s) from the auxiliary group catalog, computing missing datasets on demand. 
      reCalculate  : force redo of computation now, even if data is already saved in catalog
      searchExists : return None if data is not already computed, i.e. do not calculate right now """
    from cosmo import auxcatalog
    import datetime
    import getpass

    assert sP.snap is not None, "Must specify sP.snap for snapshotSubset load."
    assert sP.subbox is None, "No auxCat() for subbox snapshots."

    r = {}

    if not isdir(sP.derivPath + 'auxCat'):
        mkdir(sP.derivPath + 'auxCat')

    for field in iterable(fields):
        if field not in auxcatalog.fieldComputeFunctionMapping:
            raise Exception('Unrecognized field ['+field+'] for auxiliary catalog.')

        # check for existence of auxiliary catalog file for this dataset
        auxCatPath = sP.derivPath + 'auxCat/%s_%03d.hdf5' % (field,sP.snap)

        # checking for existence? (do not calculate right now if missing)
        if not isfile(auxCatPath) and searchExists:
            r[field] = None
            continue

        # load previously computed values
        if isfile(auxCatPath) and not reCalculate:
            #print('Load existing: ['+field+']')
            with h5py.File(auxCatPath,'r') as f:
                # load data
                r[field] = f[field][()]

                # load metadata
                r[field+'_attrs'] = {}
                for attr in f[field].attrs:
                    r[field+'_attrs'][attr] = f[field].attrs[attr]

            continue

        # either does not exist yet, or reCalculate requested
        print('Compute and save: ['+field+']')
        r[field], attrs = auxcatalog.fieldComputeFunctionMapping[field] (sP)
        r[field+'_attrs'] = attrs

        # save new dataset (or overwrite existing)
        with h5py.File(auxCatPath,'w') as f:
            f.create_dataset(field, data=r[field])

            if not reCalculate:
                print(' Saved new.')
            else:
                print(' Saved over existing.')

            # save metadata and any additional descriptors as attributes
            f[field].attrs['CreatedOn']   = datetime.date.today().strftime('%d %b %Y')
            f[field].attrs['CreatedRev']  = curRepoVersion()
            f[field].attrs['CreatedBy']   = getpass.getuser()
            for attrName, attrValue in attrs.iteritems():
                f[field].attrs[attrName] = attrValue
                    
    return r

def gcPath(basePath, snapNum, chunkNum=0, noLocal=False):
    """ Find and return absolute path to a group catalog HDF5 file.
        Can be used to redefine illustris_python version (il.groupcat.gcPath = cosmo.load.gcPath). """

    # local scratch test: call ourself with a basePath corresponding to local stratch (on dev node)
    if not noLocal:
        bpSplit = basePath.split("/")
        localBP = "/scratch/" + bpSplit[-3] + "/" + bpSplit[-2] + "/"
        localFT = gcPath(localBP, snapNum, noLocal=True)

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

    return None

def groupCat(sP, readIDs=False, skipIDs=False, fieldsSubhalos=None, fieldsHalos=None):
    """ Load HDF5 fof+subfind group catalog for a given snapshot.
                         
       readIDs=1 : by default, skip IDs since we operate under the group ordered snapshot assumption, but
                   if this flag is set then read IDs and include them (if they exist)
       skipIDs=1 : acknowledge we are working with a STOREIDS type .hdf5 group cat and don't warn
       fields*   : read only a subset fields from the catalog
    """
    assert sP.snap is not None, "Must specify sP.snap for groupCat() load."
    assert sP.subbox is None, "No groupCat() for subbox snapshots."

    # override path function
    il.groupcat.gcPathOrig = il.groupcat.gcPath
    il.groupcat.gcPath = gcPath

    r = {}

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

    raise Exception("No snapshot found.")

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

def snapshotSubset(sP, partType, fields, 
                   inds=None, indRange=None, haloID=None, subhaloID=None, 
                   mdi=None, sq=True, haloSubset=False):
    """ For a given snapshot load only one field for one particle type
          partType = e.g. [0,1,2,4] or ('gas','dm','tracer','stars')
          fields   = e.g. ['ParticleIDs','Coordinates',...]

          the following four optional, but at most one can be specified:
            * inds      : known indices requested, optimize the load
            * indRange  : same, but specify only min and max indices (inclusive)
            * haloID    : if input, load particles only of the specified fof halo
            * subhaloID : if input, load particles only of the specified subalo
          mdi : multi-dimensional index slice load
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
    if haloSubset and (not sP.groupOrdered or (haloID is not None) or (subhaloID is not None) or (inds is not None)):
        raise Exception("haloSubset only for groupordered snapshots, and not with halo/subhalo subset.")
    if sP.snap is None:
        raise Exception("Must specify sP.snap for snapshotSubset load.")

    # override path function
    il.snapshot.snapPath = partial(snapPath, subbox=sP.subbox)

    # make sure fields is not a single element, and don't modify input
    fields = list(iterable(fields))
    fieldsOrig = list(iterable(fields))

    # haloSubset only? construct indRange and self-call
    if haloSubset:
        offsets_pt = groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']
        kwargs['indRange'] = [0, offsets_pt[:,sP.ptNum(partType)].max()]
        return snapshotSubset(sP, partType, fields, sq=sq, **kwargs)

    # composite fields (temp, vmag, ...)
    # TODO: combining composite fields with len(fields)>1 currently skips any others, returns single ndarray
    for i,field in enumerate(fields):
        # temperature (from u,nelec) [log K]
        if field.lower() in ["temp", "temperature"]:
            u  = snapshotSubset(sP, partType, 'u', **kwargs)
            ne = snapshotSubset(sP, partType, 'ne', **kwargs)
            return sP.units.UToTemp(u,ne,log=True)

        # entropy (from u,dens) [log cgs]
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

        # metallicity in log(solar) units
        if field.lower() in ["metal_solar","Z_solar"]:
            metal = snapshotSubset(sP, partType, 'metal', **kwargs) # metal mass / total mass ratio
            return sP.units.metallicityInSolar(metal,log=False)

        # stellar age in Gyr (convert GFM_StellarFormationTime scalefactor)
        if field.lower() in ["star_age","stellar_age"]:
            curUniverseAgeGyr = sP.units.redshiftToAgeFlat(sP.redshift)
            birthTime = snapshotSubset(sP, partType, 'birthtime', **kwargs)
            birthRedshift = 1.0/birthTime - 1.0
            return curUniverseAgeGyr - sP.units.redshiftToAgeFlat(birthRedshift)

        # sound speed (hydro only version) [physical km/s]
        if field.lower() in ['csnd','soundspeed']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            return sP.units.calcSoundSpeedKmS(u,dens)

        # pressure_ratio (linear ratio of magnetic to gas pressure)
        if field.lower() in ['pres_ratio','pressure_ratio']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            b    = snapshotSubset(sP, partType, 'MagneticField', **kwargs)

            P_gas = sP.units.calcPressureCGS(u, dens)
            P_B   = sP.units.calcMagneticPressureCGS(b)
            return P_B/P_gas

        # gas pressure [log K/cm^3]
        if field.lower() in ['gas_pres','gas_pressure','p_gas']:
            dens = snapshotSubset(sP, partType, 'Density', **kwargs)
            u    = snapshotSubset(sP, partType, 'InternalEnergy', **kwargs)
            return sP.units.calcPressureCGS(u, dens, log=True)

        # magnetic prssure [log K/cm^3]
        if field.lower() in ['mag_pres','magnetic_pressure','p_b','p_magnetic']:
            b = snapshotSubset(sP, partType, 'MagneticField', **kwargs)
            return sP.units.calcMagneticPressureCGS(b, log=True)

        # total pressure, magnetic plus gas [log K/cm^3]
        if field.lower() in ['p_tot','pres_tot','pres_total','pressure_tot','pressure_total']:
            P_B = 10.0**snapshotSubset(sP, partType, 'P_B', **kwargs)
            P_gas = 10.0**snapshotSubset(sP, partType, 'P_gas', **kwargs)
            return np.log10( P_B + P_gas )

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
                 [['machnum'], 'MachNumber'],
                 [['b','bfield'], 'MagneticField'],
                 [['mass'], 'Masses'],
                 [['nh'], 'NeutralHydrogenAbundance'],
                 [['numtr'], 'NumTracers'],
                 [['id','ids'], 'ParticleIDs'],
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
      { 'names':['metals_tot','metals_total'],          'field':'GFM_Metals', 'fN':9 } \
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

    # load
    r = il.snapshot.loadSubset(sP.simPath, sP.snap, partType, fields, subset=subset, mdi=mdi, sq=sq)

    # optional unit post-processing
    if isinstance(r, np.ndarray) and len(fieldsOrig) == 1:
        if fieldsOrig[0] in ['tracer_maxent','tracer_maxtemp'] and r.max() < 20.0:
            raise Exception('Unexpectedly low max for non-log values, something maybe changed.')
            
        if fieldsOrig[0] == 'tracer_maxent':
            r = sP.units.tracerEntToCGS(r, log=True) # [log cgs] = [log K cm^2]
        if fieldsOrig[0] == 'tracer_maxtemp':
            r = logZeroSafe(r) # [log Kelvin]

    return r
