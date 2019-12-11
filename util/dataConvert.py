"""
dataConvert.py
  Various data exporters/converters, between different formats, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os import path, mkdir, getcwd
import glob
import time
import struct

def concatSubboxFilesAndMinify():
    """ Minify a series of subbox snapshots but removing unwanted fields, and re-save concatenated 
    into a smaller number of chunks. """
    from util.helper import pSplitRange

    # config
    outputPath = '/home/extdylan/sims.TNG/L35n2160TNG/output/'
    sbNum = 2
    sbSnapRange = [1200,1201]
    numChunksSave = 8

    metaGroups = ['Config','Header','Parameters']
    keepFields = {'PartType0':['Coordinates','Density','ElectronAbundance','EnergyDissipation',
                               'GFM_Metallicity','InternalEnergy','Machnumber','MagneticField',
                               'Masses','NeutralHydrogenAbundance','ParticleIDs','StarFormationRate',
                               'Velocities'],
                  'PartType1':['Coordinates','ParticleIDs','Velocities'],
                  'PartType3':['ParentID','TracerID'],
                  'PartType4':['Coordinates','GFM_InitialMass','GFM_Metallicity','GFM_StellarFormationTime',
                               'Masses','ParticleIDs','Velocities'],
                  'PartType5':['BH_BPressure','BH_CumEgyInjection_QM','BH_CumEgyInjection_RM', 
                               'BH_CumMassGrowth_QM','BH_CumMassGrowth_RM','BH_Density','BH_HostHaloMass',
                               'BH_Hsml','BH_Mass','BH_Mdot','BH_MdotBondi','BH_MdotEddington','BH_Pressure',
                               'BH_Progs','BH_U','Coordinates','Masses','ParticleIDs','Potential',
                               'Velocities']} # all

    # set paths
    sbBasePath = outputPath + 'subbox%d/' % sbNum
    saveBasePath = outputPath + 'subbox%d_new/' % sbNum

    if not path.isdir(saveBasePath):
        mkdir(saveBasePath)

    def _oldChunkPath(snapNum, chunkNum):
        return sbBasePath + 'snapdir_subbox%d_%03d/snap_subbox%d_%03d.%s.hdf5' % \
          (sbNum,snapNum,sbNum,snapNum,chunkNum)
    def _newChunkPath(snapNum, chunkNum):
        return saveBasePath + 'snapdir_subbox%d_%03d/snap_subbox%d_%03d.%s.hdf5' % \
          (sbNum,snapNum,sbNum,snapNum,chunkNum)

    # from the first: dtypes, ndims, how many chunks?
    print('Save configuration:')
    dtypes = {}
    ndims = {}

    allDone = False
    i = 0

    while not allDone:
        # get dtypes and ndims for each field, but need to find a file that contains each
        with h5py.File(_oldChunkPath(1199, i),'r') as f:
            for gName in keepFields.keys():
                if gName not in f or keepFields[gName][0] not in f[gName]:
                    continue
                dtypes[gName] = {}
                ndims[gName] = {}
                for field in keepFields[gName]:
                    dtypes[gName][field] = f[gName][field].dtype
                    ndims[gName][field] = f[gName][field].ndim
                    if ndims[gName][field] > 1:
                        ndims[gName][field] = f[gName][field].shape[1] # actually need shape of 2nd dim

        # all done?
        allDone = True
        for gName in keepFields.keys():
            if gName not in dtypes:
                allDone = False
        i += 1

    for gName in keepFields.keys():
        for field in keepFields[gName]:
            print(' ',gName, field, dtypes[gName][field], ndims[gName][field])

    numChunks = len( glob.glob(_oldChunkPath(0,'*')) )
    print('  numChunks: %d' % numChunks)

    for sbSnap in range(sbSnapRange[0], sbSnapRange[1]+1):
        sbPath = sbBasePath + 'snapdir_subbox%d_%d/' % (sbNum, sbSnap)
        oldSize = 0.0
        newSize = 0.0

        # load meta
        meta = {}
        with h5py.File(_oldChunkPath(sbSnap, 0),'r') as f:
            for gName in metaGroups:
                meta[gName] = {}
                for attr in f[gName].attrs:
                    meta[gName][attr] = f[gName].attrs[attr]

        NumPart = meta['Header']['NumPart_Total']
        assert meta['Header']['NumPart_Total_HighWord'].sum() == 0
        print('[%4d] NumPart: ' % sbSnap,NumPart)

        # allocate
        data = {}
        offsets = {}
        for gName in keepFields.keys():
            ptNum = int(gName[-1])
            # no particles of this type?
            if NumPart[ptNum] == 0:
                continue

            data[gName] = {}
            offsets[gName] = 0

            for field in keepFields[gName]:
                dtype = dtypes[gName][field]

                # allocate [N] or e.g. [N,3]
                if ndims[gName][field] == 1:
                    shape = (NumPart[ptNum])
                else:
                    shape = (NumPart[ptNum], ndims[gName][field])
                data[gName][field] = np.zeros( shape, dtype=dtype )

                if dtype in [np.float32, np.float64]:
                    data[gName][field].fill(np.nan) # for verification

        # load (requested fields only)
        print('[%4d] loading   [' % sbSnap, end='')
        for i in range(numChunks):
            print('.', end='')
            oldSize += path.getsize(_oldChunkPath(sbSnap, i)) / 1024.0**3

            with h5py.File(_oldChunkPath(sbSnap, i),'r') as f:
                for gName in keepFields.keys():
                    # PartTypeX not in file?
                    if gName not in f:
                        continue

                    # load each field of this PartTypeX
                    for field in keepFields[gName]:
                        ndim = ndims[gName][field]
                        off = offsets[gName]
                        loc_size = f[gName][field].shape[0]

                        #print('  %s off = %8d loc_size = %7d' % (gName,off,loc_size))
                        if ndim == 1:
                            data[gName][field][off:off+loc_size] = f[gName][field][()]
                        else:
                            data[gName][field][off:off+loc_size,:] = f[gName][field][()]

                    offsets[gName] += loc_size
        print(']')

        # verify
        for gName in offsets.keys():
            ptNum = int(gName[-1])
            assert offsets[gName] == NumPart[ptNum]
            for field in data[gName]:
                if dtypes[gName][field] in [np.float32, np.float64]:
                    assert np.count_nonzero( np.isnan( data[gName][field] )) == 0

        # write
        print('[%4d] writing   [' % sbSnap, end='')
        assert not path.isdir(saveBasePath + 'snapdir_subbox%d_%03d' % (sbNum,sbSnap))
        mkdir(saveBasePath + 'snapdir_subbox%d_%03d' % (sbNum,sbSnap))

        start = {}
        stop = {}
        for gName in keepFields.keys():
            start[gName] = 0
            stop[gName] = 0

        for i in range(numChunksSave):
            # determine split, update header
            print('.', end='')

            meta['Header']['NumFilesPerSnapshot'] = numChunksSave
            for gName in keepFields.keys():
                ptNum = int(gName[-1])
                if NumPart[ptNum] == 0:
                    continue

                start[gName], stop[gName] = pSplitRange([0,NumPart[ptNum]], numChunksSave, i)

                assert stop[gName] > start[gName]
                meta['Header']['NumPart_ThisFile'][ptNum] = stop[gName] - start[gName]
                #print(i,gName,start[gName],stop[gName])

            with h5py.File(_newChunkPath(sbSnap,i),'w') as f:
                # save meta
                for gName in metaGroups:
                    g = f.create_group(gName)
                    for attr in meta[gName].keys():
                        g.attrs[attr] = meta[gName][attr]

                # save data
                for gName in keepFields.keys():
                    ptNum = int(gName[-1])
                    if NumPart[ptNum] == 0:
                        print(' skip pt %d (write)' % ptNum)
                        continue

                    g = f.create_group(gName)

                    for field in keepFields[gName]:
                        ndim = ndims[gName][field]

                        if ndim == 1:
                            g[field] = data[gName][field][start[gName]:stop[gName]]
                        else:
                            g[field] = data[gName][field][start[gName]:stop[gName],:]

            newSize += path.getsize(_newChunkPath(sbSnap, i)) / 1024.0**3

        print(']')
        print('[%4d] saved (old size = %5.1f GB new size %5.1f GB)' % (i,oldSize,newSize))

    print('Done.')

def groupCutoutFromSnap(run='tng'):
    """ Create a [full] subhalo/fof cutout from a snapshot (as would be done by the Web API). """
    from util.simParams import simParams
    import cosmo

    ptTypes = ['gas','dm','bhs','stars']

    # (A) subhalo indices (z=0): TNG100-1, Illustris-1 (Lagrangian match), Illustris-1 (positional match)
    if 0:
        sP = simParams(res=1820,run=run,redshift=0.0)
        samplePath = path.expanduser('~') + '/sims.TNG/L75n1820TNG/postprocessing/guinevere_cutouts/new_mw_sample_fgas_sat.txt'

        data = np.genfromtxt(samplePath, delimiter=',', dtype='int32')

        if run == 'tng':
            subhaloIDs = data[:,0]
        if run == 'illustris':
            subhaloIDs = data[:,1] # Lagrangian match

    # (B) L25n512_0000 (TNG), L25n512_0010 (Illustris), L25n512_3000 (no BH wind), L25n512_0012 (TNG w/ Illustris winds)
    if 1:
        sP = simParams(res=512,run='tng',redshift=0.0,variant='1000')
        samplePath = path.expanduser('~') + '/sims.TNG/L75n1820TNG/postprocessing/guinevere_cutouts/new_mw_sample_L25_variants.txt'

        data = np.genfromtxt(samplePath, delimiter=',', dtype='int32')

        if sP.variant == '0000': subhaloIDs = data[:,0]
        if sP.variant == '0010': subhaloIDs = data[:,1]
        if sP.variant == '3000': subhaloIDs = data[:,2]
        if sP.variant == '0012': subhaloIDs = data[:,3]
        if sP.variant == '1000': subhaloIDs = data[:,4]
        print(sP.variant,subhaloIDs)

    # get list of field names
    fields = {}

    fileName = cosmo.load.snapPath(sP.simPath, sP.snap, chunkNum=0)

    with h5py.File(fileName,'r') as f:
        for partType in ptTypes:
            gName = 'PartType' + str(sP.ptNum(partType))
            fields[gName] = f[gName].keys()

    # loop over subhalos
    for subhaloID in subhaloIDs:
        if subhaloID == -1:
            print('skip -1')
            continue

        saveFilename = 'cutout_%s_%d_subhalo.hdf5' % (sP.simName,subhaloID)
        if path.isfile(saveFilename):
            print('skip, [%s] already exists' % saveFilename)
            continue

        data = {}

        # load (subhalo restricted)
        for partType in ptTypes:
            print(subhaloID,'sub',partType)
            gName = 'PartType' + str(sP.ptNum(partType))

            data[gName] = cosmo.load.snapshotSubset(sP, partType, fields[gName], subhaloID=subhaloID)

        # write
        with h5py.File(saveFilename,'w') as f:
            for gName in data:
                g = f.create_group(gName)
                for field in data[gName]:
                    g[field] = data[gName][field]

        # get parent fof, load (fof restricted)
        continue # skip

        data = {}
        subh = sP.groupCatSingle(subhaloID=subhaloID)
        haloID = subh['SubhaloGrNr']

        for partType in ptTypes:
            print(subhaloID,'fof',partType)
            gName = 'PartType' + str(sP.ptNum(partType))

            data[gName] = cosmo.load.snapshotSubset(sP, partType, fields[gName], haloID=haloID)

        # write
        with h5py.File('cutout_%s_%d_group.hdf5' % (sP.simName,subhaloID),'w') as f:
            for gName in data:
                g = f.create_group(gName)
                for field in data[gName]:
                    g[field] = data[gName][field]

def tracerCutoutFromTracerTracksCat():
    """ Create a subhalo cutout of tracers from the full postprocessing/tracer_tracks/ catalog. """
    from util.simParams import simParams
    ptTypes = ['gas','stars','bhs']

    # get subhaloIDs
    if 0:
        sP = simParams(res=1820,run='tng',redshift=0.0)
        data = np.genfromtxt(sP.postPath + 'guinevere_cutouts/new_mw_sample_fgas_sat.txt', delimiter=',', dtype='int32')
        subhaloIDs = data[:,0]

    if 1:
        sP = simParams(res=512,run='tng',redshift=0.0,variant='1000')
        samplePath = path.expanduser('~') + '/sims.TNG/L75n1820TNG/postprocessing/guinevere_cutouts/new_mw_sample_L25_variants.txt'

        data = np.genfromtxt(samplePath, delimiter=',', dtype='int32')

        if sP.variant == '0000': subhaloIDs = data[:,0]
        if sP.variant == '3000': subhaloIDs = data[:,2]
        if sP.variant == '0012': subhaloIDs = data[:,3]
        if sP.variant == '1000': subhaloIDs = data[:,4]

    # list of tracer quantities we know
    catBasePath = sP.postPath + 'tracer_tracks/*.hdf5'
    cats = {}

    for catPath in glob.glob(catBasePath):
        catName = catPath.split("%d_" % sP.snap)[-1].split(".hdf5")[0]
        cats[catName] = catPath

    # load offset/length from meta
    offs = {}
    lens = {}
    with h5py.File(cats['meta'],'r') as f:
        for ptType in ptTypes:
            offs[ptType] = f['Subhalo']['TracerOffset'][ptType][()]
            lens[ptType] = f['Subhalo']['TracerLength'][ptType][()]

    # loop over subhaloIDs
    for subhaloID in subhaloIDs:
        data = {}
        for ptType in ptTypes: data[ptType] = {}

        saveFilename = 'tracers_%s_%d_subhalo.hdf5' % (sP.simName,subhaloID)
        if path.isfile(saveFilename):
            print('skip, [%s] already exists' % saveFilename)
            continue

        # load from each existing cat
        for catName, catPath in cats.items():
            if 'meta.hdf5' in catPath: continue
            print(' ',subhaloID,catName)

            with h5py.File(catPath,'r') as f:
                for ptType in ptTypes:
                    start = offs[ptType][subhaloID]
                    length = lens[ptType][subhaloID]
                    data[ptType][catName] = f[catName][:, start:start+length]

                snaps = f['snaps'][()]
                redshifts = f['redshifts'][()]

        # write
        with h5py.File(saveFilename,'w') as f:
            for ptType in ptTypes:
                g = f.create_group(ptType)
                for field in data[ptType]:
                    g[field] = data[ptType][field]
            f['snaps'] = snaps
            f['redshifts'] = redshifts

def makeSnapHeadersForLHaloTree():
    """ Copy chunk 0 of each snapshot only and the header only (for LHaloTree B-HaloTree). """
    nSnaps = 59 #100

    pathFrom = '/home/extdylan/data/out/L35n2160TNG/output/snapdir_%03d/'
    pathTo   = '/home/extdylan/data/out/L35n2160TNG/output/snapdir_%03d/'
    fileFrom = pathFrom + 'snap_%03d.%s.hdf5'
    fileTo   = pathTo + 'snap_%03d.%s.hdf5'

    # loop over snapshots
    for i in range(nSnaps):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # open destination for writing
        j = 0
        if path.isfile(fileTo % (i,i,j)):
            print('Skip snap [%d], already exists' % j)
            continue

        fOut = h5py.File(fileTo % (i,i,j), 'w')

        # open origin file for reading
        assert path.isfile(fileFrom % (i,i,j))

        with h5py.File(fileFrom % (i,i,j),'r') as f:
            # copy header
            g = fOut.create_group('Header')
            for attr in f['Header'].attrs:
                fOut['Header'].attrs[attr] = f['Header'].attrs[attr]

            fOut.close()
            print('%s' % fileTo % (i,i,j))

def makeSnapSubsetsForMergerTrees():
    """ Copy snapshot chunks reducing to needed fields for tree calculation. """
    nSnaps  = 100
    copyFields = {'PartType0':['Masses','ParticleIDs','StarFormationRate'], #for SubLink_gal need 'StarFormationRate'
                  'PartType1':['ParticleIDs'],
                  'PartType4':['Masses','GFM_StellarFormationTime','ParticleIDs']}

    #copyFields = {'PartType1':['Coordinates','SubfindHsml']}
    #copyFields = {'PartType4':['Coordinates','ParticleIDs']}
    #copyFields = {'PartType0':['Masses','Coordinates','Density','GFM_Metals','GFM_Metallicity']}

    pathFrom = '/home/extdylan/sims.TNG/L35n1080TNG_DM/output/snapdir_%03d/'
    pathTo   = '/home/extdylan/data/out/L35n1080TNG_DM/output/snapdir_%03d/'
    fileFrom = pathFrom + 'snap_%03d.%s.hdf5'
    fileTo   = pathTo + 'snap_%03d.%s.hdf5'

    # L205 config
    if 'L205n2500' in pathFrom:
        nChunks = 600
        if '_DM' in pathFrom: nChunks = 75
    if 'L35n2160' in pathFrom:
        nChunks = 680
        if '_DM' in pathFrom: nChunks = 128
    if 'L35n1080' in pathFrom:
        nChunks = 128
        if '_DM' in pathFrom: nChunks = 85

    # verify number of chunks
    files = glob.glob(fileFrom % (0,0,'*'))
    assert len(files) == nChunks

    # loop over snapshots
    for i in range(82,nSnaps):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # loop over chunks
        for j in range(nChunks):
            # open destination for writing
            fOut = h5py.File(fileTo % (i,i,j)) # 'w' removed, append to existing file

            # open origin file for reading
            assert path.isfile(fileFrom % (i,i,j))

            with h5py.File(fileFrom % (i,i,j),'r') as f:
                # copy header
                if 'Header' not in fOut:
                    g = fOut.create_group('Header')
                    for attr in f['Header'].attrs:
                        fOut['Header'].attrs[attr] = f['Header'].attrs[attr]
                # loop over partTypes
                for gName in copyFields.keys():
                    # skip if not in origin
                    if gName not in f:
                        continue
                    # copy fields for this partType
                    g = fOut.create_group(gName)
                    for dName in copyFields[gName]:
                        g[dName] = f[gName][dName][()]

            fOut.close()
            print('%s' % fileTo % (i,i,j))

def makeSdssSpecObjIDhdf5():
    """ Transform some CSV files into a HDF5 for SDSS objid -> specobjid mapping. """
    from util.helper import nUnique

    files = sorted(glob.glob('z*.txt'))
    objid = np.zeros( (10000000), dtype='uint64' )
    specobjid = np.zeros( (10000000), dtype='uint64' )
    offset = 0

    # read
    for file in files:
        print(file,offset)
        x = np.loadtxt(file,delimiter=',',skiprows=2,dtype='uint64')
        num = x.shape[0]
        objid[offset:offset+num] = x[:,0]
        specobjid[offset:offset+num] = x[:,1]
        offset += num

    # look at uniqueness
    objid = objid[0:offset]
    specobjid = specobjid[0:offset]

    assert nUnique(objid) == objid.size
    assert nUnique(specobjid) == specobjid.size

    # write
    with h5py.File('sdss_objid_specobjid_z0.0-0.5.hdf5','w') as f:
        f['objid'] = objid
        f['specobjid'] = specobjid

def createEmptyMissingGroupCatChunk():
    nChunks = 64
    basePath = '/u/dnelson/sims.TNG_method/L25n512_4503/output/groups_004/'
    fileBase = basePath + 'fof_subhalo_tab_004.%d.hdf5'
    fileMake = fileBase % 60

    # load all chunks, determine number of missing groups/subgroups
    nGroups = 0
    nSubgroups = 0

    for i in range(nChunks):
        if not path.isfile(fileBase % i):
            print('Skip: %s' % fileBase % i)
            continue
        with h5py.File(fileBase % i, 'r') as f:
            nGroups += f['Header'].attrs['Ngroups_ThisFile']
            nSubgroups += f['Header'].attrs['Nsubgroups_ThisFile']

    # load data shapes and types, and write
    f = h5py.File(fileBase % 0, 'r')
    fOut = h5py.File(fileMake, 'w')

    nGroupsTot = f['Header'].attrs['Ngroups_Total']
    nSubgroupsTot = f['Header'].attrs['Nsubgroups_Total']

    nMissingGroups = nGroupsTot - nGroups
    nMissingSubgroups = nSubgroupsTot - nSubgroups

    print('Missing groups [%d] subgroups [%d].' % (nMissingGroups,nMissingSubgroups))

    fOut.create_group('Header')
    fOut.create_group('Group')
    fOut.create_group('Subhalo')

    # (header)
    for attr in f['Header'].attrs:
        fOut['Header'].attrs[attr] = f['Header'].attrs[attr]
    fOut['Header'].attrs['Ngroups_ThisFile'] = nMissingGroups
    fOut['Header'].attrs['Nsubgroups_ThisFile'] = nMissingSubgroups

    # (group)
    for key in f['Group']:
        shape = np.array( f['Group'][key].shape )
        shape[0] = nMissingGroups
        fOut['Group'][key] = np.zeros( shape, dtype=f['Group'][key].dtype )

    # (subhalo)
    for key in f['Subhalo']:
        shape = np.array( f['Subhalo'][key].shape )
        shape[0] = nMissingSubgroups
        fOut['Subhalo'][key] = np.zeros( shape, dtype=f['Subhalo'][key].dtype )

    f.close()
    fOut.close()
    print('Wrote: %s' % fileMake)

def combineAuxCatSubdivisions():
    """ Combine a subdivision of a pSplit auxCat calculation. """
    from os.path import isfile

    basePath = '/n/home07/dnelson/sims.TNG/L205n2500TNG/data.files/auxCat/'
    field    = 'Subhalo_StellarPhot_p07c_cf00dust_res_conv_ns1' # _rad30pkpc
    fileBase = field + '_099-split-%d-%d.hdf5'

    pSplitBig = [90,100,160] # from
    pSplitSm  = [9,16] # to

    # load properties
    allExist = True
    allCount = 0

    for i in range(pSplitBig[0],pSplitBig[1]):
        filePath_i = basePath + fileBase % (i,pSplitBig[2])
        print(filePath_i)

        if not isfile(filePath_i):
            allExist = False
            continue

        # record counts and dataset shape
        with h5py.File(filePath_i,'r') as f:
            allCount += f['subhaloIDs'].size
            allShape = f[field].shape

    assert allExist

    # allocate
    allShape = np.array(allShape)
    allShape[0] = allCount # size
    offset = 0

    new_r = np.zeros( allShape, dtype='float32' )
    subhaloIDs = np.zeros( allCount, dtype='int32' )

    new_r.fill(-1.0) # does validly contain nan
    subhaloIDs.fill(np.nan)

    # read
    for i in range(pSplitBig[0],pSplitBig[1]):
        filePath_i = basePath + fileBase % (i,pSplitBig[2])
        print(filePath_i)

        # record counts and dataset shape
        with h5py.File(filePath_i,'r') as f:
            length = f['subhaloIDs'].size
            subhaloIDs[offset : offset+length] = f['subhaloIDs'][()]

            new_r[offset : offset+length, :, :] = f[field][()]

            offset += length

    assert np.count_nonzero(np.isnan(subhaloIDs)) == 0
    assert np.count_nonzero(new_r == -1.0) == 0

    outPath = '/n/home07/dnelson/data7/' + fileBase % (pSplitSm[0],pSplitSm[1])
    print('Write to: [%s]' % outPath)

    assert not isfile(outPath)
    with h5py.File(outPath,'w') as f:
        f.create_dataset(field, data=new_r)
        f.create_dataset('subhaloIDs', data=subhaloIDs)

    print('Saved.')

    verifyPath = basePath + fileBase % (pSplitSm[0],pSplitSm[1])
    if not isfile(verifyPath):
        print('Verify does not exist, skip [%s].' % verifyPath)
        return

    with h5py.File(verifyPath,'r') as f:
        verify_r = f[field][()]
        verify_ids = f['subhaloIDs'][()]

    assert np.array_equal( verify_ids, subhaloIDs )
    # np.array_equal() is False for NaN entries
    # roundoff differences:
    #assert ((verify_r == new_r) | (np.isnan(verify_r) & np.isnan(new_r))).all() 
    assert np.allclose( verify_r, new_r, equal_nan=True)
    print('Verified.')

def redshiftWikiTable():
    """ Output wiki-syntax table of snapshot spacings. """
    fname = '/n/home07/dnelson/sims.TNG/outputs.txt'
    with open(fname,'r') as f:
        lines = f.readlines()

    for i,line in enumerate([l.strip() for l in lines]):
        scaleFac, snapType = line.split()
        scaleFac = float(scaleFac)
        snapType = int(snapType)

        if snapType == 1:
            print('| %3d || %6.4g || %5.2g || {{yes}} || - ' % (i,scaleFac,1/scaleFac-1.0))
            print('|-')
        if snapType == 3:
            print('| %3d || %6.4g || %5.2g || -       || {{yes}} ' % (i,scaleFac,1/scaleFac-1.0))
            print('|-')

    sP = simParams(res=455,run='tng')
    z = cosmo.util.snapNumToRedshift(sP, all=True)

    w = np.where(z >= 0.0)[0]
    print(len(w))
    for redshift in z[w]:
        pass

def snapRedshiftsTxt():
    """ Output a text-file of snapshot redshifts, etc. """
    from util.simParams import simParams
    from cosmo.util import multiRunMatchedSnapList, snapNumToRedshift, snapNumToAgeFlat

    # config
    sP = simParams(res=1820,run='tng',redshift=0.0,variant='subbox0')
    minZ      = 0.0
    maxZ      = 50.0 # tng subboxes start at a=0.02, illustris at a=0.0078125
    maxNSnaps = 2700 # 90 seconds at 30 fps
    matchUse  = 'condense'

    # snapshots
    panels = [{'sP':sP}]
    snapNumLists = multiRunMatchedSnapList(panels, matchUse, maxNum=maxNSnaps, 
                                           minRedshift=minZ, maxRedshift=maxZ)

    snapNums = snapNumLists[0]
    frameNums = np.arange(snapNums.size)
    redshifts = snapNumToRedshift(sP, snapNums)
    ages      = snapNumToAgeFlat(sP, snap=snapNums)

    # write
    with file('frames.txt','w') as f:
        f.write('# frame_number snapshot_number redshift age_universe_gyr\n')
        for i in range(frameNums.size):
            f.write('%04d %04d %6.3f %5.3f\n' % (frameNums[i],snapNums[i],redshifts[i],ages[i]))
    print('Wrote: [frames.txt].')

def tngVariantsLatexOrWikiTable(variants='all', fmt='wiki'):
    """ Output latex-syntax table describing the TNG model variations runs. """
    import csv

    run_file = '/home/extdylan/sims.TNG_method/runs.csv'

    with open(run_file) as f:
        lines = f.readlines()

    # if variants is a list of lists, flatten
    if isinstance(variants[0],list):
        variants = [item for sublist in variants for item in sublist]

    # header
    if fmt == 'wiki':
        print('{| class="eoTable2 wikiTable"')
        print('! run || name || description || 128^3 || 256^3 || 512^3 || ' + \
              'parameter/option changed || fiducial value || changed value || notes')
    if fmt == 'latex':
        print('\\begin{table*}')
        print('  \\fontsize{8}{8}\selectfont')
        print('  \caption{Caption here.}')
        print('  \label{simTable}')
        print('  \\begin{center}')
        print('    \\begin{tabular}{rlllll}')
        print('     \hline\hline')
        print('     \# & Run Name & Parameter(s) or Option(s) Changed & Fiducial Value & Modified Value & Reference \\\\ \hline')

    count = 1
    for line in csv.reader(lines,quoting=csv.QUOTE_ALL):
        if '_' in line[0]:
            run, prio, largevol, who, stat512, stat256, stat128, recomp, \
            name, desc, change, val_fiducial, val_changed, notes = line
            run = run.split("_")[1]

            if stat512 != 'done': continue
            if variants != 'all':
                if run not in variants: continue

            if fmt == 'wiki':
                print('|-')
                runstat = '{{yes}} || {{yes}} || {{yes}}' # 128, 256, 512
                print('|| %s || %s || %s || %s || %s || %s || %s || %s' % \
                    (run,name,desc,runstat,change,val_fiducial,val_changed,notes))
            if fmt == 'latex':
                ref = 'W17' if 'BH' in name else 'P17' # needs to be corrected for other cases
                change = change.replace("_","\_").replace("#","\#")
                print('     %d & %s & %s & %s & %s & %s \\\\' % \
                    (count,name,change,val_fiducial,val_changed,ref))

            count += 1

    # footer
    if fmt == 'wiki':
        print('|}')
    if fmt == 'latex':
        print('    \hline')
        print('    \end{tabular}')
        print('  \end{center}')
        print('\end{table*}')

def export_ovi_phase():
    """ Export raw data points for OVI phase diagram. """
    from cosmo.load import snapshotSubset
    from cosmo.util import inverseMapPartIndicesToSubhaloIDs, cenSatSubhaloIndices
    from tracer.tracerMC import match3
    from util.simParams import simParams
    
    sP = simParams(res=1820,run='tng',redshift=0.0)
    N = int(1e7) # None for all
    partType = 'gas'

    # one of the these three:
    cenSubsOnly = False # select from particles within central subhalos only, otherwise all in box
    fofOnly = True # select from particles within FoF halos only
    haloID = None # if not None, only this fof halo

    # load
    xvals = snapshotSubset(sP, partType, 'hdens', haloID=haloID, haloSubset=fofOnly)
    xvals = np.log10(xvals) # log=True
    yvals = snapshotSubset(sP, partType, 'temp', haloID=haloID, haloSubset=fofOnly)
    weight = snapshotSubset(sP, partType, 'O VI mass', haloID=haloID, haloSubset=fofOnly)
    weight = sP.units.codeMassToMsun(weight)
    N_tot = xvals.size

    # subsample?
    if N is not None:
        np.random.seed(424242)
        inds = np.arange(xvals.size)

        selection = 'all particles in box'
        if cenSubsOnly:
            assert haloID is None and not fofOnly
            inds2 = inverseMapPartIndicesToSubhaloIDs(sP, inds, partType)
            cen_inds = cenSatSubhaloIndices(sP, cenSatSelect='cen')
            _, in_cen_inds = match3(cen_inds, inds2)
            inds = inds[in_cen_inds]
            selection = 'central subhalo particles only (all masses)'
            selStr = '_censub'
        if fofOnly:
            assert haloID is None and not cenSubsOnly
            selection = 'fof halo particles only (all masses)'
            selStr = '_fof'

        if haloID is not None:
            selection = 'all member particles of fof halo [%d] only' % haloID
            selStr = '_halo%d' % haloID
        else:
            inds_sel = np.random.choice(inds, size=N, replace=False)
            xvals = xvals[inds_sel]
            yvals = yvals[inds_sel]
            weight = weight[inds_sel]

    # save
    with h5py.File('ovi_%s_z%s_%s%s.hdf5' % (sP.simName,sP.redshift,N,selStr), 'w') as f:
        h = f.create_group('Header')
        h.attrs['num_pts'] = N if haloID is None else N_tot
        h.attrs['num_pts_total'] = N_tot
        h.attrs['written_by'] = 'dnelson'
        h.attrs['selection'] = selection
        h.attrs['sim_name'] = sP.simName
        h.attrs['sim_redshift'] = sP.redshift

        h['hdens_logcm3'] = xvals
        h['temp_logk'] = yvals
        h['ovi_mass_msun'] = weight

def makeCohnVsuiteCatalog(redshift=0.0):
    """ Write a .txt file for input into Joanne Cohn's validation-suite. """
    from util.simParams import simParams
    sP = simParams(res=1820,run='illustris',redshift=redshift)

    # load
    mstar = sP.groupCat(fieldsSubhalos=['mstar_30pkpc_log']) # log msun
    sfr   = sP.groupCat(fieldsSubhalos=['SubhaloSFRinRad']) # msun/yr
    cen   = sP.groupCat(fieldsSubhalos=['central_flag'])
    mhalo = sP.groupCat(fieldsSubhalos=['mhalo_subfind_log']) # log msun
    m200  = sP.groupCat(fieldsSubhalos=['mhalo_200_log'])

    sat = (~cen.astype('bool')).astype('int16')

    w = np.where(np.isnan(mstar)) # zero stellar mass
    mstar[w] = 0.0

    w = np.where(cen == 1)
    mhalo[w] = m200[w] # use m200,crit values for centrals at least

    w = np.where(mstar >= 7.8)
    mstar = mstar[w]
    sfr = sfr[w]
    cen = cen[w]
    mhalo = mhalo[w]
    sat = sat[w]

    with open('inputstats_%s_z%s.txt' % (sP.simName,int(redshift)),'w') as f:
        f.write('# Illustris-1 simulation (z=%s)\n' % redshift)
        f.write('# Nelson+ (2015) (https://arxiv.org/abs/1504.00362) (http://www.illustris-project.org/data/)\n')
        f.write('# note: M* is 30pkpc aperture values, SFR is SubhaloSFRinRad, M_halo is Group_M_Crit200 for centrals, SubhaloMass for satellites\n')
        f.write('# note: minimum M* of 7.8 log msun for inclusion here (50 stars)\n')
        f.write('# \log_10 M^* [M_\odot], SFR (M^*[M_\odot]/yr), RA, DEC, zred, ifsat, \log_10 M_halo [M_\odot]\n')
        for i in range(mstar.size):
            f.write('%7.3f %7.3f 1.0 1.0 %.1f %d %7.3f\n' % (mstar[i],sfr[i],redshift,sat[i],mhalo[i]))

def convertMillenniumSubhaloCatalog(snap=63):
    """ Convert a subhalo catalog ('sub_tab_NNN.X' files), custom binary format of Millennium simulation to Illustris-like HDF5. """
    from tracer.tracerMC import match3

    savePath = '/u/dnelson/data/sims.other/Millennium-1/output/'
    loadPath = '/virgo/simulations/Millennium/'

    dm_particle_mass = 0.0860657 # ~10^9 msun

    if not path.isdir(savePath + 'groups_%03d' % snap):
        mkdir(savePath + 'groups_%03d' % snap)

    def _chunkPath(snap, chunkNum):
        return loadPath + 'postproc_%03d/sub_tab_%03d.%s' % (snap,snap,chunkNum)
    def _groupChunkPath(snap, chunkNum):
        return loadPath + 'snapdir_%03d/group_tab_%03d.%s' % (snap,snap,chunkNum)

    nChunks = len( glob.glob(_chunkPath(snap,'*')) )

    # no catalog? (snapshot <= 4) write empty catalog
    if nChunks == 0:
        print('Found [%d] chunks for snapshot[%03d]! Writing empty catalog!' % (nChunks,snap))
        # save into single hdf5
        with h5py.File(savePath + 'groups_%03d/fof_subhalo_tab_%03d.hdf5' % (snap,snap), 'w') as f:
            # header
            header = f.create_group('Header')
            header.attrs['Ngroups_ThisFile'] = 0
            header.attrs['Ngroups_Total'] = 0
            header.attrs['Nids_ThisFile'] = 0
            header.attrs['Nids_Total'] = 0
            header.attrs['Nsubgroups_ThisFile'] = 0
            header.attrs['Nsubgroups_Total'] = 0
            header.attrs['NumFiles'] = 1
        return

    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    # reader header of first chunk
    with open(_chunkPath(snap,0),'rb') as f:
        header = f.read(4*5)

    NGroups    = struct.unpack('i', header[0:4])[0]
    NIds       = struct.unpack('i', header[4:8])[0]
    TotNGroups = struct.unpack('i', header[8:12])[0]
    NFiles     = struct.unpack('i', header[12:16])[0]
    NSubs      = struct.unpack('i', header[16:20])[0]

    # detect big-endianness
    endian = '@' # native = little
    if NFiles != nChunks:
        TotNGroups = struct.unpack('>i', header[8:12])[0]
        NFiles = struct.unpack('>i', header[12:16])[0]
        endian = '>'

    assert NFiles == nChunks

    # no TotNSubs stored...
    TotNSubs = 0
    TotNGroupsCheck = 0

    for i in range(nChunks):
        with open(_chunkPath(snap,i),'rb') as f:
            header  = f.read(4*5)
            NGroups = struct.unpack(endian+'i', header[0:4])[0]
            NSubs   = struct.unpack(endian+'i', header[16:20])[0]

            TotNSubs += NSubs
            TotNGroupsCheck += NGroups

    print('Total: [%d] groups, [%d] subhalos, reading...' % (TotNGroups,TotNSubs))

    assert TotNGroupsCheck == TotNGroups

    # allocate (group files)
    GroupLen       = np.zeros( TotNGroups, dtype='int32' )
    GroupOffset    = np.zeros( TotNGroups, dtype='int32' )

    # load (group files)
    g_off = 0

    for i in range(nChunks):
        if i % int(nChunks/10) == 0: print(' %d%%' % np.ceil(float(i)/nChunks*100), end='')
        # full read
        with open(_groupChunkPath(snap,i),'rb') as f:
            data = f.read()

        # header (object counts)
        NGroups = struct.unpack(endian+'i', data[0:4])[0]

        GroupLen[g_off : g_off + NGroups]    = struct.unpack(endian+'i' * NGroups, data[16 : 16 + 4*NGroups])
        GroupOffset[g_off : g_off + NGroups] = struct.unpack(endian+'i' * NGroups, data[16 + 4*NGroups : 16 + 8*NGroups])

        g_off += NGroups

    print(' done with group files.')

    # allocate (subhalo files)
    NSubsPerHalo   = np.zeros( TotNGroups, dtype='int32' )
    FirstSubOfHalo = np.zeros( TotNGroups, dtype='int32' )

    SubLen         = np.zeros( TotNSubs, dtype='int32' )
    SubOffset      = np.zeros( TotNSubs, dtype='int32' )
    SubParentHalo  = np.zeros( TotNSubs, dtype='int32' )
    SubFileNr      = np.zeros( TotNSubs, dtype='int16' )
    SubLocalIndex  = np.zeros( TotNSubs, dtype='int32' )

    Halo_M_Mean200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_Mean200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_M_Crit200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_Crit200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_M_TopHat200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_TopHat200 = np.zeros( TotNGroups, dtype='float32' )

    SubPos         = np.zeros( (TotNSubs,3), dtype='float32' )
    SubVel         = np.zeros( (TotNSubs,3), dtype='float32' )
    SubVelDisp     = np.zeros( TotNSubs, dtype='float32' )
    SubVmax        = np.zeros( TotNSubs, dtype='float32' )
    SubSpin        = np.zeros( (TotNSubs,3), dtype='float32' )
    SubMostBoundID = np.zeros( TotNSubs, dtype='int64' )
    SubHalfMass    = np.zeros( TotNSubs, dtype='float32' )

    # load (subhalo files)
    s_off = 0 # subhalos
    g_off = 0 # groups

    for i in range(nChunks):
        if i % int(nChunks/10) == 0: print(' %d%%' % np.ceil(float(i)/nChunks*100), end='')
        # full read
        with open(_chunkPath(snap,i),'rb') as f:
            data = f.read()

        # header (object counts)
        NGroups    = struct.unpack(endian+'i', data[0:4])[0]
        NSubs      = struct.unpack(endian+'i', data[16:20])[0]

        # per halo
        header_bytes = 20
        off = header_bytes
        NSubsPerHalo[g_off : g_off + NGroups]   = struct.unpack(endian+'i' * NGroups, data[off + 0*NGroups : off + 4*NGroups])
        FirstSubOfHalo[g_off : g_off + NGroups] = struct.unpack(endian+'i' * NGroups, data[off + 4*NGroups : off + 8*NGroups])
        FirstSubOfHalo[g_off : g_off + NGroups] += s_off # as stored is local to chunk files

        # per subhalo
        off = header_bytes + 8*NGroups
        SubLen[s_off : s_off + NSubs]        = struct.unpack(endian+'i' * NSubs, data[off + 0*NSubs : off + 4*NSubs])
        SubOffset[s_off : s_off + NSubs]     = struct.unpack(endian+'i' * NSubs, data[off + 4*NSubs : off + 8*NSubs])
        SubParentHalo[s_off : s_off + NSubs] = struct.unpack(endian+'i' * NSubs, data[off + 8*NSubs : off + 12*NSubs])
        SubParentHalo[s_off : s_off + NSubs] += g_off # as stored is local to chunk files

        # per subhalo chunk-pointer information
        SubFileNr[s_off : s_off + NSubs] = i
        SubLocalIndex[s_off : s_off + NSubs] = np.arange(NSubs)

        # per halo
        off = header_bytes + 8*NGroups + 12*NSubs
        Halo_M_Mean200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 0*NGroups: off + 4*NGroups])
        Halo_R_Mean200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 4*NGroups: off + 8*NGroups])
        Halo_M_Crit200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 8*NGroups: off + 12*NGroups])
        Halo_R_Crit200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 12*NGroups: off + 16*NGroups])
        Halo_M_TopHat200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 16*NGroups: off + 20*NGroups])
        Halo_R_TopHat200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 20*NGroups: off + 24*NGroups])

        # per subhalo
        off = header_bytes + 8*NGroups + 12*NSubs + 24*NGroups
        SubPos[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 0*NSubs : off + 12*NSubs]), (NSubs,3) )
        SubVel[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 12*NSubs : off + 24*NSubs]), (NSubs,3) )
        SubVelDisp[s_off : s_off + NSubs]             = struct.unpack(endian+'f' * 1*NSubs, data[off + 24*NSubs : off + 28*NSubs])
        SubVmax[s_off : s_off + NSubs]                = struct.unpack(endian+'f' * 1*NSubs, data[off + 28*NSubs : off + 32*NSubs])
        SubSpin[s_off : s_off + NSubs,:]  = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 32*NSubs : off + 44*NSubs]), (NSubs,3) )
        SubMostBoundID[s_off : s_off + NSubs]         = struct.unpack(endian+'q' * 1*NSubs, data[off + 44*NSubs : off + 52*NSubs])
        SubHalfMass[s_off : s_off + NSubs]            = struct.unpack(endian+'f' * 1*NSubs, data[off + 52*NSubs : off + 56*NSubs])

        # should have read entire file
        off = header_bytes + 8*NGroups + 12*NSubs + 24*NGroups + 56*NSubs
        assert off == len(data)

        g_off += NGroups
        s_off += NSubs

    # sanity checks
    assert SubLen.min() >= 20
    assert SubOffset.min() >= 0
    assert SubParentHalo.min() >= 0
    assert SubPos.min() >= 0.0 and SubPos.max() <= 500.0
    assert SubMostBoundID.min() >= 0

    # group (mass) ordering is LOCAL to each chunk! need a global sort and shuffle of all fields
    g_globalSort = np.argsort(-GroupLen, kind='mergesort') # negative -> descending order, stable

    GroupLen_Orig = np.array(GroupLen) # keep copy

    # re-order group properties
    GroupLen = GroupLen[g_globalSort]
    NSubsPerHalo = NSubsPerHalo[g_globalSort]
    FirstSubOfHalo = FirstSubOfHalo[g_globalSort]
    Halo_M_Mean200 = Halo_M_Mean200[g_globalSort]
    Halo_R_Mean200 = Halo_R_Mean200[g_globalSort]
    Halo_M_Crit200 = Halo_M_Crit200[g_globalSort]
    Halo_R_Crit200 = Halo_R_Crit200[g_globalSort]
    Halo_M_TopHat200 = Halo_M_TopHat200[g_globalSort]
    Halo_R_TopHat200 = Halo_R_TopHat200[g_globalSort]

    # fix subhalo parent indices, determine new subhalo ordering
    g_globalSortInv = np.zeros( g_globalSort.size, dtype='int32' )
    g_globalSortInv[g_globalSort] = np.arange(g_globalSort.size)
    SubParentHalo = g_globalSortInv[SubParentHalo]

    s_globalSort = np.argsort(SubParentHalo, kind='mergesort')

    # check new subhalo ordering
    SubParentHalo_check = np.zeros( SubParentHalo.size, dtype='int32' )
    for i in range(TotNGroups):
        if NSubsPerHalo[i] == 0:
            continue
        SubParentHalo_check[ FirstSubOfHalo[i] : FirstSubOfHalo[i] + NSubsPerHalo[i] ] = i
    assert np.array_equal(SubParentHalo,SubParentHalo_check)

    # fix first subhalo indices
    s_globalSortInv = np.zeros( s_globalSort.size, dtype='int32' )
    s_globalSortInv[s_globalSort] = np.arange(s_globalSort.size)

    w_past = np.where(FirstSubOfHalo >= s_globalSort.size)
    assert np.sum(NSubsPerHalo[w_past]) == 0
    FirstSubOfHalo[w_past] = -1

    FirstSubOfHalo = s_globalSortInv[FirstSubOfHalo]

    SubLen_Orig = np.array(SubLen)

    # re-order subhalo properties
    SubParentHalo = SubParentHalo[s_globalSort]
    SubLen = SubLen[s_globalSort]
    SubPos = SubPos[s_globalSort,:]
    SubVel = SubVel[s_globalSort,:]
    SubVelDisp = SubVelDisp[s_globalSort]
    SubVmax = SubVmax[s_globalSort]
    SubSpin = SubSpin[s_globalSort]
    SubMostBoundID = SubMostBoundID[s_globalSort]
    SubHalfMass = SubHalfMass[s_globalSort]

    # sanity checks
    assert FirstSubOfHalo[0] == 0
    w = np.where(NSubsPerHalo > 0)
    assert np.array_equal(SubParentHalo[FirstSubOfHalo[w]], w[0])
    assert GroupLen[0] == GroupLen.max()
    w = np.where(NSubsPerHalo == 0)
    if len(w[0]):
        a, b = match3(w[0],SubParentHalo)
        assert a is None and b is None
        FirstSubOfHalo[w] = -1 # convention

    # GroupOffset/SubhaloOffset are local to chunk files, just recreate now with global offsets
    subgroupCount = 0

    snapOffsetsGroup = np.zeros( TotNGroups, dtype='int64' )
    snapOffsetsSubhalo = np.zeros( TotNSubs, dtype='int64' )
    
    snapOffsetsGroup[1:] = np.cumsum( GroupLen )[:-1]
    
    for k in np.arange(TotNGroups):
        # subhalo offsets depend on group (to allow fuzz)
        if NSubsPerHalo[k] > 0:
            snapOffsetsSubhalo[subgroupCount] = snapOffsetsGroup[k]
            
            subgroupCount += 1
            for m in np.arange(1, NSubsPerHalo[k]):
                snapOffsetsSubhalo[subgroupCount] = snapOffsetsSubhalo[subgroupCount-1] + SubLen[subgroupCount-1]
                subgroupCount += 1

    # create a mapping of original (FileNr, SubLocalIndex) -> new index, which we will need to go from MPA trees to these group catalogs
    offset = 0
    FileNr_len = np.zeros( nChunks, dtype='int32' )
    FileNr_off = np.zeros( nChunks, dtype='int32' )
    new_index  = np.zeros( TotNSubs, dtype='int32' )

    for i in range(nChunks):
        w = np.where(SubFileNr == i)
        inds = s_globalSortInv[w]

        FileNr_len[i] = inds.size
        new_index[offset : offset + inds.size] = inds
        offset += inds.size

    FileNr_off[1:] = np.cumsum( FileNr_len )[:-1]

    with h5py.File(savePath + 'groups_%03d/original_order_%03d.hdf5' % (snap,snap), 'w') as f:
        f['Group_Reorder'] = g_globalSort
        f['Subhalo_Reorder'] = s_globalSort

        # e.g. the subhalo is at f['NewIndex'][ f['NewIndex_FileNrOffset'][FileNr] + SubhaloIndex ]
        f['NewIndex_FileNrOffset'] = FileNr_off
        f['NewIndex'] = new_index

    # create original GroupOffset to help particle rearrangement
    snapOffsetsGroup_Orig = np.zeros( TotNGroups, dtype='int64' )    
    snapOffsetsGroup_Orig[1:] = np.cumsum( GroupLen_Orig )[:-1]

    with h5py.File(savePath + 'gorder_%d.hdf5' % snap,'w') as f:
        f['GroupLen_Orig'] = GroupLen_Orig
        f['GroupOffset_Orig'] = snapOffsetsGroup_Orig
        f['Group_Reorder'] = g_globalSort

    # save into single hdf5
    with h5py.File(savePath + 'groups_%03d/fof_subhalo_tab_%03d.hdf5' % (snap,snap), 'w') as f:
        # header
        header = f.create_group('Header')
        header.attrs['Ngroups_ThisFile'] = np.int32(TotNGroups)
        header.attrs['Ngroups_Total'] = np.int32(TotNGroups)
        header.attrs['Nids_ThisFile'] = np.int32(0)
        header.attrs['Nids_Total'] = np.int32(0)
        header.attrs['Nsubgroups_ThisFile'] = np.int32(TotNSubs)
        header.attrs['Nsubgroups_Total'] = np.int32(TotNSubs)
        header.attrs['NumFiles'] = np.int32(1)

        # groups
        groups = f.create_group('Group')
        groups['GroupFirstSub'] = FirstSubOfHalo
        groups['GroupLen'] = GroupLen
        groups['GroupMass'] = np.array(GroupLen * dm_particle_mass, dtype='float32')
        groups['GroupNsubs'] = NSubsPerHalo
        groups['Group_M_Crit200'] = Halo_M_Crit200
        groups['Group_R_Crit200'] = Halo_R_Crit200
        groups['Group_M_Mean200'] = Halo_M_Mean200
        groups['Group_R_Mean200'] = Halo_R_Mean200
        groups['Group_M_TopHat200'] = Halo_M_TopHat200
        groups['Group_R_TopHat200'] = Halo_R_TopHat200

        GroupLenType  = np.zeros( (GroupLen.size,6), dtype=GroupLen.dtype )
        GroupMassType = np.zeros( (GroupLen.size,6), dtype='float32' )
        GroupLenType[:,1]  = GroupLen
        GroupMassType[:,1] = GroupLen * dm_particle_mass
        groups['GroupLenType']  = GroupLenType
        groups['GroupMassType'] = GroupMassType

        # subhalos
        subs = f.create_group('Subhalo')
        subs['SubhaloGrNr'] = SubParentHalo
        subs['SubhaloHalfmassRad'] = SubHalfMass
        subs['SubhaloIDMostbound'] = SubMostBoundID
        subs['SubhaloLen'] = SubLen
        subs['SubhaloMass'] = np.array(SubLen * dm_particle_mass, dtype='float32')
        subs['SubhaloPos'] = SubPos
        subs['SubhaloSpin'] = SubSpin
        subs['SubhaloVel'] = SubVel
        subs['SubhaloVelDisp'] = SubVelDisp
        subs['SubhaloVmax'] = SubVmax

        SubhaloLenType  = np.zeros( (SubLen.size,6), dtype=SubLen.dtype )
        SubhaloMassType = np.zeros( (SubLen.size,6), dtype='float32' )
        SubhaloLenType[:,1]  = SubLen
        SubhaloMassType[:,1] = SubLen * dm_particle_mass
        groups['SubhaloLenType']  = SubhaloLenType
        groups['SubhaloMassType'] = SubhaloMassType

        # offsets (inside group files, similar to Illustris public data release, for convenience)
        offs = f.create_group('Offsets')
        offs['Group_Snap'] = snapOffsetsGroup
        offs['Subhalo_Snap'] = snapOffsetsSubhalo

    # return
    print(' All Done.')

def convertMillennium2SubhaloCatalog(snap=67):
    """ Convert a subhalo catalog ('subhalo_tab_NNN.X' files), custom binary format of Millennium-2 simulation to TNG-like HDF5. """
    from tracer.tracerMC import match3

    savePath = '/u/dnelson/data/sims.other/Millennium-2/output/'
    loadPath = '/virgo/simulations/Millennium2/BigRun/'

    header_bytes = 32

    if not path.isdir(savePath + 'groups_%03d' % snap):
        mkdir(savePath + 'groups_%03d' % snap)

    def _chunkPath(snap, chunkNum):
        return loadPath + 'groups_%03d/subhalo_tab_%03d.%s' % (snap,snap,chunkNum)

    nChunks = len( glob.glob(_chunkPath(snap,'*')) )

    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    # reader header of first chunk
    with open(_chunkPath(snap,0),'rb') as f:
        header = f.read(header_bytes)

    NGroups    = struct.unpack('i', header[0:4])[0]
    TotNGroups = struct.unpack('i', header[4:8])[0]
    NIds       = struct.unpack('i', header[8:12])[0]
    TotNids    = struct.unpack('q', header[12:20])[0]
    NFiles     = struct.unpack('i', header[20:24])[0]
    NSubs      = struct.unpack('i', header[24:28])[0]
    TotNSubs   = struct.unpack('i', header[28:32])[0]

    assert NFiles == nChunks # verifies endianness
    endian = '@' # native = little

    # no catalog? (snapshot <= 3) write empty catalog
    if TotNGroups == 0:
        assert TotNSubs == 0
        print('No groups or subgroups for snapshot[%03d]! Writing empty catalog!' % snap)

        # save into single hdf5
        with h5py.File(savePath + 'groups_%03d/fof_subhalo_tab_%03d.hdf5' % (snap,snap), 'w') as f:
            # header
            header = f.create_group('Header')
            header.attrs['Ngroups_ThisFile'] = 0
            header.attrs['Ngroups_Total'] = 0
            header.attrs['Nids_ThisFile'] = 0
            header.attrs['Nids_Total'] = 0
            header.attrs['Nsubgroups_ThisFile'] = 0
            header.attrs['Nsubgroups_Total'] = 0
            header.attrs['NumFiles'] = 1
        return

    print('Total: [%d] groups, [%d] subhalos, reading...' % (TotNGroups,TotNSubs))

    # allocate (groups)
    GroupLen       = np.zeros( TotNGroups, dtype='int32' )
    GroupOffset    = np.zeros( TotNGroups, dtype='int32' ) # note: wrong (overflow) (unused)
    GroupMass      = np.zeros( TotNGroups, dtype='float32' )
    GroupPos       = np.zeros( (TotNGroups,3), dtype='float32' )

    Halo_M_Mean200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_Mean200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_M_Crit200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_Crit200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_M_TopHat200 = np.zeros( TotNGroups, dtype='float32' )
    Halo_R_TopHat200 = np.zeros( TotNGroups, dtype='float32' )

    ContamCount    = np.zeros( TotNGroups, dtype='uint32' ) # unused, ==0
    ContamMass     = np.zeros( TotNGroups, dtype='float32' ) # unused, ==0
    NSubsPerHalo   = np.zeros( TotNGroups, dtype='int32' )
    FirstSubOfHalo = np.zeros( TotNGroups, dtype='int32' )

    # allocate (subhalos)
    SubLen         = np.zeros( TotNSubs, dtype='int32' )
    SubOffset      = np.zeros( TotNSubs, dtype='int32' ) # note: wrong (overflow) (unused)
    SubParentHalo  = np.zeros( TotNSubs, dtype='int32' ) # note: is SubhaloParent? unlike in Millennium-1
    SubMass        = np.zeros( TotNSubs, dtype='float32' )

    SubPos         = np.zeros( (TotNSubs,3), dtype='float32' )
    SubVel         = np.zeros( (TotNSubs,3), dtype='float32' )
    SubCM          = np.zeros( (TotNSubs,3), dtype='float32' )
    SubSpin        = np.zeros( (TotNSubs,3), dtype='float32' )
    SubVelDisp     = np.zeros( TotNSubs, dtype='float32' )
    SubVmax        = np.zeros( TotNSubs, dtype='float32' )
    SubRVmax       = np.zeros( TotNSubs, dtype='float32' )
    SubHalfMass    = np.zeros( TotNSubs, dtype='float32' )
    SubMostBoundID = np.zeros( TotNSubs, dtype='int64' )
    SubGrNr        = np.zeros( TotNSubs, dtype='int32' )

    # load (subhalo files)
    s_off = 0 # subhalos
    g_off = 0 # groups

    for i in range(nChunks):
        if i % int(nChunks/10) == 0: print(' %d%%' % np.ceil(float(i)/nChunks*100), end='', flush=True)
        # full read
        with open(_chunkPath(snap,i),'rb') as f:
            data = f.read()

        # header (object counts)
        NGroups = struct.unpack(endian+'i', data[0:4])[0]
        NSubs   = struct.unpack(endian+'i', data[24:28])[0]

        # per halo
        off = header_bytes
        GroupLen[g_off : g_off + NGroups]       = struct.unpack(endian+'i' * NGroups, data[off + 0*NGroups : off + 4*NGroups])
        GroupOffset[g_off : g_off + NGroups]    = struct.unpack(endian+'i' * NGroups, data[off + 4*NGroups : off + 8*NGroups])
        GroupMass[g_off : g_off + NGroups]      = struct.unpack(endian+'f' * NGroups, data[off + 8*NGroups : off + 12*NGroups])
        GroupPos[g_off : g_off + NGroups,:] = np.reshape( struct.unpack(endian+'f' * 3*NGroups, data[off + 12*NGroups : off + 24*NGroups]), (NGroups,3) )

        off = header_bytes + 24*NGroups
        Halo_M_Mean200[g_off : g_off + NGroups]   = struct.unpack(endian+'f' * NGroups, data[off + 0*NGroups: off + 4*NGroups])
        Halo_R_Mean200[g_off : g_off + NGroups]   = struct.unpack(endian+'f' * NGroups, data[off + 4*NGroups: off + 8*NGroups])
        Halo_M_Crit200[g_off : g_off + NGroups]   = struct.unpack(endian+'f' * NGroups, data[off + 8*NGroups: off + 12*NGroups])
        Halo_R_Crit200[g_off : g_off + NGroups]   = struct.unpack(endian+'f' * NGroups, data[off + 12*NGroups: off + 16*NGroups])
        Halo_M_TopHat200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 16*NGroups: off + 20*NGroups])
        Halo_R_TopHat200[g_off : g_off + NGroups] = struct.unpack(endian+'f' * NGroups, data[off + 20*NGroups: off + 24*NGroups])

        off = header_bytes + 48*NGroups
        ContamCount[g_off : g_off + NGroups]    = struct.unpack(endian+'I' * NGroups, data[off + 0*NGroups: off + 4*NGroups])
        ContamMass[g_off : g_off + NGroups]     = struct.unpack(endian+'f' * NGroups, data[off + 4*NGroups: off + 8*NGroups])

        NSubsPerHalo[g_off : g_off + NGroups]   = struct.unpack(endian+'i' * NGroups, data[off + 8*NGroups : off + 12*NGroups])
        FirstSubOfHalo[g_off : g_off + NGroups] = struct.unpack(endian+'i' * NGroups, data[off + 12*NGroups : off + 16*NGroups]) # global

        # per subhalo
        off = header_bytes + 64*NGroups
        SubLen[s_off : s_off + NSubs]        = struct.unpack(endian+'i' * NSubs, data[off + 0*NSubs : off + 4*NSubs])
        SubOffset[s_off : s_off + NSubs]     = struct.unpack(endian+'i' * NSubs, data[off + 4*NSubs : off + 8*NSubs])
        SubParentHalo[s_off : s_off + NSubs] = struct.unpack(endian+'i' * NSubs, data[off + 8*NSubs : off + 12*NSubs]) # group rel

        off = header_bytes + 64*NGroups + 12*NSubs
        SubMass[s_off : s_off + NSubs]                = struct.unpack(endian+'f' * 1*NSubs, data[off + 0*NSubs : off + 4*NSubs])
        SubPos[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 4*NSubs : off + 16*NSubs]), (NSubs,3) )
        SubVel[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 16*NSubs : off + 28*NSubs]), (NSubs,3) )
        SubCM[s_off : s_off + NSubs,:]    = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 28*NSubs : off + 40*NSubs]), (NSubs,3) )
        SubSpin[s_off : s_off + NSubs,:]  = np.reshape( struct.unpack(endian+'f' * 3*NSubs, data[off + 40*NSubs : off + 52*NSubs]), (NSubs,3) )

        SubVelDisp[s_off : s_off + NSubs]             = struct.unpack(endian+'f' * 1*NSubs, data[off + 52*NSubs : off + 56*NSubs])
        SubVmax[s_off : s_off + NSubs]                = struct.unpack(endian+'f' * 1*NSubs, data[off + 56*NSubs : off + 60*NSubs])
        SubRVmax[s_off : s_off + NSubs]               = struct.unpack(endian+'f' * 1*NSubs, data[off + 60*NSubs : off + 64*NSubs])
        SubHalfMass[s_off : s_off + NSubs]            = struct.unpack(endian+'f' * 1*NSubs, data[off + 64*NSubs : off + 68*NSubs])
        SubMostBoundID[s_off : s_off + NSubs]         = struct.unpack(endian+'q' * 1*NSubs, data[off + 68*NSubs : off + 76*NSubs])
        SubGrNr[s_off : s_off + NSubs]                = struct.unpack(endian+'i' * 1*NSubs, data[off + 76*NSubs : off + 80*NSubs])

        # should have read entire file
        off = header_bytes + 64*NGroups + 92*NSubs
        assert off == len(data)

        g_off += NGroups
        s_off += NSubs

    # sanity checks
    assert SubLen.min() >= 20
    assert SubGrNr.min() >= 0
    assert SubMass.min() > 0
    assert GroupLen.min() > 0
    assert GroupMass.min() > 0
    assert SubParentHalo.min() >= 0
    assert GroupPos.min() >= 0.0 and GroupPos.max() <= 100.0
    assert SubPos.min() >= 0.0 and SubPos.max() <= 100.0
    assert SubMostBoundID.min() >= 0
    assert SubGrNr.max() < TotNGroups

    assert g_off == TotNGroups
    assert s_off == TotNSubs

    # verify subhalo ordering
    SubGrNr_check = np.zeros( SubGrNr.size, dtype='int32' )
    for i in range(TotNGroups):
        if NSubsPerHalo[i] == 0:
            continue
        SubGrNr_check[ FirstSubOfHalo[i] : FirstSubOfHalo[i] + NSubsPerHalo[i] ] = i
    assert np.array_equal(SubGrNr,SubGrNr_check)

    # sanity checks
    assert FirstSubOfHalo[0] == 0
    w = np.where(NSubsPerHalo > 0)
    assert np.array_equal(SubGrNr[FirstSubOfHalo[w]], w[0])
    assert GroupLen[0] == GroupLen.max()
    w = np.where(NSubsPerHalo == 0)
    if len(w[0]):
        a, b = match3(w[0],SubGrNr)
        assert a is None and b is None
        FirstSubOfHalo[w] = -1 # modify GroupFirstSub, convention

    assert FirstSubOfHalo.max() < TotNSubs

    # GroupOffset/SubhaloOffset have overflow issues, just recreate now with global offsets
    subgroupCount = 0

    snapOffsetsGroup = np.zeros( TotNGroups, dtype='int64' )
    snapOffsetsSubhalo = np.zeros( TotNSubs, dtype='int64' )
    
    snapOffsetsGroup[1:] = np.cumsum( GroupLen )[:-1]
    
    for k in np.arange(TotNGroups):
        # subhalo offsets depend on group (to allow fuzz)
        if NSubsPerHalo[k] > 0:
            snapOffsetsSubhalo[subgroupCount] = snapOffsetsGroup[k]
            
            subgroupCount += 1
            for m in np.arange(1, NSubsPerHalo[k]):
                snapOffsetsSubhalo[subgroupCount] = snapOffsetsSubhalo[subgroupCount-1] + SubLen[subgroupCount-1]
                subgroupCount += 1

    # save into single hdf5
    with h5py.File(savePath + 'groups_%03d/fof_subhalo_tab_%03d.hdf5' % (snap,snap), 'w') as f:
        # header
        header = f.create_group('Header')
        header.attrs['Ngroups_ThisFile'] = np.int32(TotNGroups)
        header.attrs['Ngroups_Total'] = np.int32(TotNGroups)
        header.attrs['Nids_ThisFile'] = np.int32(0)
        header.attrs['Nids_Total'] = np.int32(0)
        header.attrs['Nsubgroups_ThisFile'] = np.int32(TotNSubs)
        header.attrs['Nsubgroups_Total'] = np.int32(TotNSubs)
        header.attrs['NumFiles'] = np.int32(1)

        # groups
        groups = f.create_group('Group')
        groups['GroupFirstSub'] = FirstSubOfHalo
        groups['GroupLen'] = GroupLen
        groups['GroupPos'] = GroupPos
        groups['GroupMass'] = GroupMass
        groups['GroupNsubs'] = NSubsPerHalo
        groups['Group_M_Crit200'] = Halo_M_Crit200
        groups['Group_R_Crit200'] = Halo_R_Crit200
        groups['Group_M_Mean200'] = Halo_M_Mean200
        groups['Group_R_Mean200'] = Halo_R_Mean200
        groups['Group_M_TopHat200'] = Halo_M_TopHat200
        groups['Group_R_TopHat200'] = Halo_R_TopHat200

        GroupLenType  = np.zeros( (GroupLen.size,6), dtype=GroupLen.dtype )
        GroupMassType = np.zeros( (GroupLen.size,6), dtype='float32' )
        GroupLenType[:,1]  = GroupLen
        GroupMassType[:,1] = GroupMass
        groups['GroupLenType']  = GroupLenType
        groups['GroupMassType'] = GroupMassType

        # subhalos
        subs = f.create_group('Subhalo')
        subs['SubhaloLen'] = SubLen
        subs['SubhaloMass'] = SubMass
        subs['SubhaloGrNr'] = SubGrNr
        subs['SubhaloHalfmassRad'] = SubHalfMass
        subs['SubhaloIDMostbound'] = SubMostBoundID
        subs['SubhaloParent'] = SubParentHalo
        subs['SubhaloPos'] = SubPos
        subs['SubhaloCM'] = SubCM
        subs['SubhaloSpin'] = SubSpin
        subs['SubhaloVel'] = SubVel
        subs['SubhaloVelDisp'] = SubVelDisp
        subs['SubhaloVmaxRad'] = SubRVmax
        subs['SubhaloVmax'] = SubVmax

        SubhaloLenType  = np.zeros( (SubLen.size,6), dtype=SubLen.dtype )
        SubhaloMassType = np.zeros( (SubLen.size,6), dtype='float32' )
        SubhaloLenType[:,1]  = SubLen
        SubhaloMassType[:,1] = SubMass
        subs['SubhaloLenType']  = SubhaloLenType
        subs['SubhaloMassType'] = SubhaloMassType

        # offsets (inside group files, similar to Illustris public data release, for convenience)
        offs = f.create_group('Offsets')
        offs['Group_Snap'] = snapOffsetsGroup
        offs['Subhalo_Snap'] = snapOffsetsSubhalo

    # return
    print(' All Done.')

def convertMillenniumSnapshot(snap=63):
    """ Convert a complete Millennium snapshot (+IDS) into Illustris-like group-ordered HDF5 format. """
    from tracer.tracerMC import match3
    from util.helper import isUnique

    savePath = '/u/dnelson/data/sims.millennium/Millennium1/output/'
    loadPath = '/virgo/simulations/Recovered_Millennium/' 
    #loadPath = '/virgo/simulations/MilliMillennium/'

    saveFile = savePath + 'snapdir_%03d/snap_%03d.hdf5' % (snap,snap)

    if not path.isdir(savePath + 'snapdir_%03d' % snap):
        mkdir(savePath + 'snapdir_%03d' % snap)

    def _snapChunkPath(snap, chunkNum):
        return loadPath + 'snapdir_%03d/snap_millennium_%03d.%s' % (snap,snap,chunkNum)
        #return loadPath + 'snapdir_%03d/snap_milli_%03d.%s' % (snap,snap,chunkNum)
    def _idChunkPath(snap, chunkNum):
        return loadPath + 'postproc_%03d/sub_ids_%03d.%s' % (snap,snap,chunkNum)

    nChunks = len( glob.glob(_snapChunkPath(snap,'*')) )
    nChunksIDs = len( [fn for fn in glob.glob(_idChunkPath(snap,'*')) if 'swapped' not in fn] )
    assert nChunks == nChunksIDs or nChunksIDs == 0
    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    # detect big-endianness
    endian = '@' # native = little

    with open(_snapChunkPath(snap,0),'rb') as f:
        header = f.read(260)
        NFiles = struct.unpack(endian+'i', header[128:128+4])[0]

    if NFiles != nChunks:
        endian = '>'
        NFiles = struct.unpack(endian+'i', header[128:128+4])[0]

    assert NFiles == nChunks

    # cycle one
    if not path.isfile(saveFile):
        # first, load all IDs from the sub files
        Nids_tot = 0
        
        if path.isfile(savePath + 'gorder_%d.hdf5' % snap):
            offset = 0

            for i in range(nChunks):
                with open(_idChunkPath(snap,i),'rb') as f:
                    header = f.read(16)
                    Nids = struct.unpack(endian+'i', header[4:8])[0]
                    Nids_tot += Nids

            ids_groupordered_old = np.zeros( Nids_tot, dtype='int64' )
            print('Reading a total of [%d] IDs now...' % Nids_tot)

            bitshift = ((1 << 34) - 1) # from get_group_coordinates()

            for i in range(nChunks):
                # full read
                with open(_idChunkPath(snap,i),'rb') as f:
                    data = f.read()
                Nids = struct.unpack(endian+'i', data[4:8])[0]
                if Nids == 0:
                    continue
                ids = struct.unpack(endian+'q' * Nids, data[16:16 + Nids*8])

                # transform into actual particle ID
                # particleid = (GroupIDs[i] << 30) >> 30 (seems wrong...)
                # hashkey    = GroupIDs[i] >> 34
                #ids_groupordered[offset : offset+Nids] = (np.array(ids) << 30) >> 30
                ids_groupordered_old[offset : offset+Nids] = np.array(ids) & bitshift

                print('[%3d] IDs [%8d] particles, from [%10d] to [%10d].' % (i, Nids, offset, offset+Nids))
                offset += Nids

            assert np.min(ids_groupordered_old) >= 0 # otherwise overflow or bad conversion above

            # ids_groupordered are in the chunk-local group ordering! reshuffle now
            print('Shuffling IDs into global group order...')

            gorder = {}
            with h5py.File(savePath + 'gorder_%d.hdf5' % snap) as f:
                for key in f:
                    gorder[key] = f[key][()]

            # use original (chunk-ordered) GroupOffset and GroupLen accessed in global-order
            # to access ids_groupordered in non-contig blocks, stamping into ids_groupordered_new contiguously through
            offset = 0
            ids_groupordered = np.zeros( Nids_tot, dtype='int64' )

            for i in gorder['Group_Reorder']:
                read_offset = gorder['GroupOffset_Orig'][i]
                read_length = gorder['GroupLen_Orig'][i]

                if read_length == 0:
                    continue

                ids_groupordered[offset : offset+read_length] = ids_groupordered_old[read_offset : read_offset + read_length]
                offset += read_length

            ids_groupordered_old = None
        else:
            print('NO GROUP CATALOG! Assuming zero groups, and proceeding with snapshot rewrite!')

        # reader header of first snapshot chunk
        with open(_snapChunkPath(snap,0),'rb') as f:
            header = f.read(260)

        npart      = struct.unpack(endian+'iiiiii', header[4:4+24])[1]
        mass       = struct.unpack(endian+'dddddd', header[28:28+48])
        scalefac   = struct.unpack(endian+'d', header[76:76+8])[0]
        redshift   = struct.unpack(endian+'d', header[84:84+8])[0]
        #nPartTot   = struct.unpack(endian+'iiiiii', header[100:100+24])[1]
        nFiles     = struct.unpack(endian+'i', header[128:128+4])[0]
        BoxSize    = struct.unpack(endian+'d', header[132:132+8])[0]
        Omega0     = struct.unpack(endian+'d', header[140:140+8])[0]
        OmegaL     = struct.unpack(endian+'d', header[148:148+8])[0]
        Hubble     = struct.unpack(endian+'d', header[156:156+8])[0]

        assert nFiles == nChunks

        # nPartTot is wrong, has no highword, so read and accumulate manually
        nPartTot = 0
        for i in range(nChunks):
            with open(_snapChunkPath(snap,i),'rb') as f:
                header = f.read(28)
                npart  = struct.unpack(endian+'iiiiii', header[4:4+24])[1]
            nPartTot += npart
        print('Found new nPartTot [%d]' % nPartTot)

        # load all snapshot IDs
        offset = 0
        ids_snapordered = np.zeros( nPartTot, dtype='int64' )

        for i in range(nChunks):
            # full read
            with open(_snapChunkPath(snap,i),'rb') as f:
                data = f.read()

            # local particle counts
            npart_local = struct.unpack(endian+'iiiiii', data[4:4+24])[1]

            # cast and save
            start_ids = 284 + 24*npart_local
            ids = struct.unpack(endian+'q' * npart_local*1, data[start_ids:start_ids + npart_local*8])

            ids_snapordered[offset : offset+npart_local] = ids

            print('[%3d] Snap IDs [%8d] particles, from [%10d] to [%10d].' % (i, npart_local, offset, offset+npart_local))
            offset += npart_local

        # crossmatch group catalog IDs and snapshot IDs
        if Nids_tot > 0:
            print('Matching two ID sets now...')
            start = time.time()

            ind_snapordered, ind_groupordered = match3(ids_snapordered, ids_groupordered)
            # note: ids_snapordered[ind_snapordered] puts them into group ordering
            print(' took: '+str(round(time.time()-start,2))+' sec')

            assert ind_snapordered.size == ids_groupordered.size # must have found them all
            ids_groupordered = None
            ind_groupordered = None

            # create mask for outer FoF fuzz
            mask = np.zeros( ids_snapordered.size, dtype='bool' )
            mask[ind_snapordered] = 1

            ind_outerfuzz = np.where(mask == 0)[0]

            # create master re-ordering index list
            inds_reorder = np.hstack( (ind_snapordered,ind_outerfuzz) )

            ind_snapordered = None
            ind_outerfuzz = None
        else:
            print('No cross-matching! Writing snapshot in ORIGINAL ORDER!')
            inds_reorder = np.arange(ids_snapordered.size)

        assert inds_reorder.size == ids_snapordered.size # union must include all
        assert isUnique(inds_reorder) # no duplicates

        # open file for writing
        fOut = h5py.File(saveFile, 'w')

        header = fOut.create_group('Header')
        numPartTot = np.zeros( 6, dtype='int64' )
        numPartTot[1] = nPartTot
        header.attrs['BoxSize'] = BoxSize
        header.attrs['HubbleParam'] = Hubble
        header.attrs['MassTable'] = np.array(mass, dtype='float64')
        header.attrs['NumFilesPerSnapshot'] = np.int32(1)
        header.attrs['NumPart_ThisFile'] = np.int32(numPartTot)
        header.attrs['NumPart_Total'] = np.int32(numPartTot)
        header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype='int32')
        header.attrs['Omega0'] = Omega0
        header.attrs['OmegaLambda'] = OmegaL
        header.attrs['Redshift'] = redshift
        header.attrs['Time'] = scalefac

        pt1 = fOut.create_group('PartType1')

        # save IDs immediately
        pt1['ParticleIDs'] = ids_snapordered[inds_reorder]
        ids_snapordered = None
        fOut.close()

        # save order permutation, early quit (clear memory)
        with h5py.File(savePath + 'reorder_%d.hdf5' % snap,'w') as f:
            f['Reorder'] = inds_reorder
        print('Wrote intermediate file, restart to finish.')
        return
    else:
        # intermediate file already exists, open
        print('Loading intermediate file...')
        with h5py.File(savePath + 'reorder_%d.hdf5' % snap) as f:
            inds_reorder = f['Reorder'][()]

        fOut = h5py.File(saveFile)
        nPartTot = fOut['Header'].attrs['NumPart_Total'][1]
        pt1 = fOut['PartType1']

        # load remaining particle fields, one at a time
        for ptName in ['Coordinates','Velocities']:
            offset = 0
            val = np.zeros( (nPartTot,3), dtype='float32' )

            for i in range(nChunks):
                # full read
                with open(_snapChunkPath(snap,i),'rb') as f:
                    data = f.read()

                # local particle counts
                npart_local = struct.unpack(endian+'iiiiii', data[4:4+24])[1]

                # cast
                start_pos = 268
                start_vel = 276 + 12*npart_local
                start = start_pos if ptName == 'Coordinates' else start_vel

                val_local = struct.unpack(endian+'f' * npart_local*3, data[start:start + npart_local*12])

                # stamp
                val[offset:offset+npart_local,:] = np.reshape(val_local, (npart_local,3))

                print('[%3d] %s [%8d] particles, from [%10d] to [%10d].' % (i, ptName, npart_local, offset, offset+npart_local))
                offset += npart_local

            # re-order and write
            val = val[inds_reorder,:]

            pt1[ptName] = val

        # close
        fOut.close()
        print('All done.')

def convertMillennium2Snapshot(snap=67):
    """ Convert a complete Millennium-2 snapshot into TNG-like group-ordered HDF5 format. 
    Note all snapshots except 4-7 (inclusive) are already group-ordered. """
    from tracer.tracerMC import match3
    from util.helper import isUnique

    savePath = '/u/dnelson/data/sims.other/Millennium-2/output/'
    loadPath = '/virgo/simulations/Millennium2/BigRun/'

    saveFile = savePath + 'snapdir_%03d/snap_%03d.hdf5' % (snap,snap)

    unorderedSnaps = [4,5,6,7] # on /virgo/, only these snapshots are not yet in group order

    if not path.isdir(savePath + 'snapdir_%03d' % snap):
        mkdir(savePath + 'snapdir_%03d' % snap)

    def _snapChunkPath(snap, chunkNum):
        if snap in unorderedSnaps:
            return loadPath + 'snapdir_%03d/snap_newMillen_%03d.%s' % (snap,snap,chunkNum)
        return loadPath + 'snapdir_%03d/snap_newMillen_subidorder_%03d.%s' % (snap,snap,chunkNum)
    def _idChunkPath(snap, chunkNum):
        return loadPath + 'groups_%03d/subhalo_ids_%03d.%s' % (snap,snap,chunkNum)

    nChunks = len( glob.glob(_snapChunkPath(snap,'*')) )
    nChunksIDs = len( glob.glob(_idChunkPath(snap,'*')) )

    # three cases for file organization
    assert nChunks == nChunksIDs or nChunksIDs == 0 or (nChunks == 512 and nChunksIDs == 2048)
    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    # load header
    endian = '@' # native = little
    with open(_snapChunkPath(snap,0),'rb') as f:
        header = f.read(260)
        NFiles = struct.unpack(endian+'i', header[128:128+4])[0]

    if NFiles != nChunks:
        endian = '>' # big
        NFiles = struct.unpack(endian+'i', header[128:128+4])[0]
        print('WARNING: Endian set to big, true for snapshots then?')

    assert NFiles == nChunks 

    # reader header of first snapshot chunk
    with open(_snapChunkPath(snap,0),'rb') as f:
        header = f.read(260)

    npart      = struct.unpack(endian+'iiiiii', header[4:4+24])[1]
    mass       = struct.unpack(endian+'dddddd', header[28:28+48])
    scalefac   = struct.unpack(endian+'d', header[76:76+8])[0]
    redshift   = struct.unpack(endian+'d', header[84:84+8])[0]
    #nPartTot   = struct.unpack(endian+'iiiiii', header[100:100+24])[1]
    nFiles     = struct.unpack(endian+'i', header[128:128+4])[0]
    BoxSize    = struct.unpack(endian+'d', header[132:132+8])[0]
    Omega0     = struct.unpack(endian+'d', header[140:140+8])[0]
    OmegaL     = struct.unpack(endian+'d', header[148:148+8])[0]
    Hubble     = struct.unpack(endian+'d', header[156:156+8])[0]

    assert nFiles == nChunks

    # nPartTot with highword, just read and accumulate manually
    nPartTot = 0
    for i in range(nChunks):
        with open(_snapChunkPath(snap,i),'rb') as f:
            header = f.read(28)
            npart  = struct.unpack(endian+'iiiiii', header[4:4+24])[1]
        nPartTot += npart

    print('Found nPartTot [%d]' % nPartTot)
    assert nPartTot == 2160**3

    # load all snapshot IDs
    offset = 0
    ids_snapordered = np.zeros( nPartTot, dtype='int64' )

    for i in range(nChunks):
        # full read
        with open(_snapChunkPath(snap,i),'rb') as f:
            data = f.read()

        # local particle counts
        npart_local = struct.unpack(endian+'iiiiii', data[4:4+24])[1]

        # cast and save
        start_ids = 284 + 24*npart_local
        ids = struct.unpack(endian+'q' * npart_local*1, data[start_ids:start_ids + npart_local*8])

        ids_snapordered[offset : offset+npart_local] = ids

        min_val = np.min(ids)
        print('[%4d] Snap IDs [%8d] particles, from [%10d] to [%10d] min = %10d' % \
            (i, npart_local, offset, offset+npart_local, min_val))
        offset += npart_local
        if min_val == 0:
            import pdb; pdb.set_trace()

    if nChunksIDs > 0:
        # need to reshuffle
        assert snap in unorderedSnaps

        # first, load all IDs from the sub files
        offset = 0

        # M-WAMP7/AQ/M2: Ngroups[int32], TotNgroups[int32], Nids[int32], TotNids[int64], NFiles[int32], SendOffset[int32(?)], ids[int64*]
        # M1: Ngroups[int32], Nids[int32], TotNgroups[int32], NTask[int32], ids[int64*]
        # P-M: Ngroups[int32], TotNgroups[int64], Nids[int32], TotNids[int64], NFiles[int32], SendOffset[int64], ids[int64*]
        for i in range(nChunksIDs):
            with open(_idChunkPath(snap,i),'rb') as f:
                header = f.read(28)
                Nids    = struct.unpack(endian+'i', header[8:12])[0]
                TotNids = struct.unpack(endian+'q', header[12:20])[0]
                offset += Nids

        ids_groupordered = np.zeros( offset, dtype='int64' )
        print('Reading a total of [%d] IDs now...' % offset)
        assert offset == TotNids

        bitshift = ((1 << 34) - 1) # same as Millennium-1 (should be corrent)
        offset = 0

        for i in range(nChunksIDs):
            # full read
            with open(_idChunkPath(snap,i),'rb') as f:
                data = f.read()
            Nids = struct.unpack(endian+'i', data[8:12])[0]
            if Nids == 0:
                continue
            ids = struct.unpack(endian+'q' * Nids, data[28:28 + Nids*8])

            # transform into actual particle ID and stamp
            ids_groupordered[offset : offset+Nids] = np.array(ids) & bitshift

            print('[%3d] IDs [%8d] particles, from [%10d] to [%10d].' % (i, Nids, offset, offset+Nids))
            offset += Nids

        assert np.min(ids_groupordered) >= 0 # otherwise overflow or bad conversion above

        # crossmatch group catalog IDs and snapshot IDs
        if TotNids > 0:
            print('Matching two ID sets now...')
            start = time.time()

            ind_snapordered, ind_groupordered = match3(ids_snapordered, ids_groupordered)
            # note: ids_snapordered[ind_snapordered] puts them into group ordering
            print(' took: '+str(round(time.time()-start,2))+' sec')

            assert ind_snapordered.size == ids_groupordered.size # must have found them all
            ids_groupordered = None
            ind_groupordered = None

            # create mask for outer FoF fuzz
            mask = np.zeros( ids_snapordered.size, dtype='bool' )
            mask[ind_snapordered] = 1

            ind_outerfuzz = np.where(mask == 0)[0]

            # create master re-ordering index list
            inds_reorder = np.hstack( (ind_snapordered,ind_outerfuzz) )

            ind_snapordered = None
            ind_outerfuzz = None

        assert inds_reorder.size == ids_snapordered.size # union must include all
        assert isUnique(inds_reorder) # no duplicates

        ids_snapordered = ids_snapordered[inds_reorder]

    else:
        print('No subhalo_ids files! Writing snapshot in ORIGINAL ORDER!')
        inds_reorder = None

    # open file for writing
    fOut = h5py.File(saveFile, 'w')

    header = fOut.create_group('Header')
    numPartTot = np.zeros( 6, dtype='int64' )
    numPartTot[1] = nPartTot
    header.attrs['BoxSize'] = BoxSize
    header.attrs['HubbleParam'] = Hubble
    header.attrs['MassTable'] = np.array(mass, dtype='float64')
    header.attrs['NumFilesPerSnapshot'] = np.int32(1)
    header.attrs['NumPart_ThisFile'] = np.int64(numPartTot)
    header.attrs['NumPart_Total'] = np.int64(numPartTot)
    header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype='int32')
    header.attrs['Omega0'] = Omega0
    header.attrs['OmegaLambda'] = OmegaL
    header.attrs['Redshift'] = redshift
    header.attrs['Time'] = scalefac

    pt1 = fOut.create_group('PartType1')

    # save IDs immediately
    pt1['ParticleIDs'] = ids_snapordered
    ids_snapordered = None

    # load remaining particle fields, one at a time
    for ptName in ['Coordinates','Velocities']:
        offset = 0
        val = np.zeros( (nPartTot,3), dtype='float32' )

        for i in range(nChunks):
            # full read
            with open(_snapChunkPath(snap,i),'rb') as f:
                data = f.read()

            # local particle counts
            npart_local = struct.unpack(endian+'iiiiii', data[4:4+24])[1]

            # cast
            start_pos = 268
            start_vel = 276 + 12*npart_local
            start = start_pos if ptName == 'Coordinates' else start_vel

            unpacker = struct.Struct(endian+'f' * npart_local*3)
            val_local = unpacker.unpack(data[start:start + npart_local*12])
            val_local = np.reshape(val_local, (npart_local,3))

            # stamp
            val[offset:offset+npart_local,:] = val_local

            print('[%4d] %s [%8d] particles, from [%10d] to [%10d].' % (i, ptName, npart_local, offset, offset+npart_local), flush=True)
            offset += npart_local

        # re-order and write
        if inds_reorder is not None:
            val = val[inds_reorder,:]

        print('writing...')
        pt1[ptName] = val
        print('written.')

    # close
    fOut.close()
    print('All done.')

def convertGadgetICsToHDF5():
    """ Convert a Gadget-1 binary format ICs (dm-only, 8 byte IDs, 4 byte pos/vel) into HDF5 format (keep original ordering). """
    loadPath = '/u/dnelson/sims.TNG/InitialConditions/L35n2160TNG/output/ICs.%s' 
    savePath = '/u/dnelson/sims.TNG/L35n2160TNG/output/snap_ics.hdf5'

    ptNum = 1
    longids = True # 64 bit, else 32 bit

    # read header of first snapshot chunk
    nChunks = len( glob.glob(loadPath % '*') )
    print('Found [%d] chunks, loading...' % nChunks)

    with open(loadPath % 0,'rb') as f:
        header = f.read(260)

    npart      = struct.unpack('iiiiii', header[4:4+24])[ptNum]
    masstable  = struct.unpack('dddddd', header[28:28+48])
    scalefac   = struct.unpack('d', header[76:76+8])[0]
    redshift   = struct.unpack('d', header[84:84+8])[0]
    #nPartTot   = struct.unpack('iiiiii', header[100:100+24])[ptNum]
    nFiles     = struct.unpack('i', header[128:128+4])[0]
    BoxSize    = struct.unpack('d', header[132:132+8])[0]
    Omega0     = struct.unpack('d', header[140:140+8])[0]
    OmegaL     = struct.unpack('d', header[148:148+8])[0]
    Hubble     = struct.unpack('d', header[156:156+8])[0]

    assert nFiles == nChunks

    if longids:
        ids_type = 'q'
        ids_size = 8
        ids_dtype = 'int64'
    else:
        ids_type = 'i'
        ids_size = 4
        ids_dtype = 'int32'

    # nPartTot is wrong, has no highword, so read and accumulate manually
    nPartTot = 0
    for i in range(nChunks):
        with open(loadPath % i,'rb') as f:
            header = f.read(28)
            npart  = struct.unpack('iiiiii', header[4:4+24])[ptNum]
        nPartTot += npart
    print('Found new nPartTot [%d]' % nPartTot)

    # open file for writing
    fOut = h5py.File(savePath, 'w')

    # write header
    header = fOut.create_group('Header')
    numPartTot = np.zeros( 6, dtype='int64' )
    numPartTot[ptNum] = nPartTot
    header.attrs['BoxSize'] = BoxSize
    header.attrs['HubbleParam'] = Hubble
    header.attrs['MassTable'] = np.array(masstable, dtype='float64')
    header.attrs['NumFilesPerSnapshot'] = 1
    header.attrs['NumPart_ThisFile'] = numPartTot
    header.attrs['NumPart_Total'] = numPartTot
    header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype='int64')
    header.attrs['Omega0'] = Omega0
    header.attrs['OmegaLambda'] = OmegaL
    header.attrs['Redshift'] = redshift
    header.attrs['Time'] = scalefac

    # create group and datasets
    pt = fOut.create_group('PartType%d' % ptNum)

    particle_pos = pt.create_dataset('Coordinates', (nPartTot,3), dtype='float32' )
    particle_vel = pt.create_dataset('Velocities', (nPartTot,3), dtype='float32' )
    particle_ids = pt.create_dataset('ParticleIDs', (nPartTot,), dtype=ids_dtype )

    if masstable[ptNum] == 0:
        particle_mass = pt.create_dataset('Masses', (nPartTot,), dtype='float32' )

    # load all snapshot IDs
    offset = 0

    for i in range(nChunks):
        # full read
        with open(loadPath % i,'rb') as f:
            data = f.read()

        # local particle counts
        npart_local = struct.unpack('iiiiii', data[4:4+24])[ptNum]

        # cast and save
        start_pos = 268 + 0*npart_local
        start_vel = 276 + 12*npart_local
        start_ids = 284 + 24*npart_local
        start_mass = 292 + (24+ids_size)*npart_local

        pos = struct.unpack('f' * npart_local*3, data[start_pos:start_pos + npart_local*12])
        vel = struct.unpack('f' * npart_local*3, data[start_vel:start_vel + npart_local*12])
        ids = struct.unpack(ids_type * npart_local*1, data[start_ids:start_ids + npart_local*ids_size])

        if masstable[ptNum] == 0:
            mass = struct.unpack('f' * npart_local*1, data[start_mass:start_mass + npart_local*4])

        # write
        particle_pos[offset : offset+npart_local,:] = np.reshape( pos, (npart_local,3) )
        particle_vel[offset : offset+npart_local,:] = np.reshape( vel, (npart_local,3) )
        particle_ids[offset : offset+npart_local] = ids

        if masstable[ptNum] == 0:
            particle_mass[offset : offset+npart_local] = mass

        print('[%3d] Snap chunk has [%8d] particles, from [%10d] to [%10d].' % (i, npart_local, offset, offset+npart_local))
        offset += npart_local

    fOut.close()
    print('All done.')

def splitSingleHDF5IntoChunks():
    """ Split a single-file snapshot/catalog/etc HDF5 into a number of roughly equally sized chunks. """
    from util.helper import pSplitRange
    basePath = '/u/dnelson/sims.millennium/Millennium1/output/snapdir_063/'
    fileName = 'milli_063.hdf5'
    numChunksSave = 16

    # load header, dtypes, ndims, and data
    fGroups = {}
    dtypes = {}
    ndims = {}
    data = {}

    with h5py.File(basePath + fileName,'r') as f:
        header = dict( f['Header'].attrs.items() )
        for k in f.keys():
            if k != 'Header': fGroups[k] = []

        for gName in fGroups.keys():
            dtypes[gName] = {}
            ndims[gName] = {}
            data[gName] = {}
            fGroups[gName] = f[gName].keys()

            for field in fGroups[gName]:
                dtypes[gName][field] = f[gName][field].dtype
                ndims[gName][field] = f[gName][field].ndim
                if ndims[gName][field] > 1:
                    ndims[gName][field] = f[gName][field].shape[1] # actually need shape of 2nd dim
                data[gName][field] = f[gName][field][()]

    print('NumPartTot: ', header['NumPart_Total'])

    start = {}
    stop = {}
    for gName in fGroups.keys():
        start[gName] = 0
        stop[gName] = 0

    for i in range(numChunksSave):
        # determine split, update header
        header['NumFilesPerSnapshot'] = numChunksSave
        for gName in fGroups.keys():
            ptNum = int(gName[-1])
            if header['NumPart_Total'][ptNum] == 0:
                continue

            start[gName], stop[gName] = pSplitRange([0,header['NumPart_Total'][ptNum]], numChunksSave, i)

            assert stop[gName] > start[gName]
            header['NumPart_ThisFile'][ptNum] = stop[gName] - start[gName]
            print(i,gName,start[gName],stop[gName])

        newChunkPath = basePath + fileName.split('.hdf5')[0] + '.%d.hdf5' % i

        with h5py.File(newChunkPath,'w') as f:
            # save header
            g = f.create_group('Header')
            for attr in header.keys():
                g.attrs[attr] = header[attr]

            # save data
            for gName in fGroups.keys():
                ptNum = int(gName[-1])
                if header['NumPart_Total'][ptNum] == 0:
                    print(' skip pt %d (write)' % ptNum)
                    continue

                g = f.create_group(gName)

                for field in fGroups[gName]:
                    ndim = ndims[gName][field]

                    if ndim == 1:
                        g[field] = data[gName][field][start[gName]:stop[gName]]
                    else:
                        g[field] = data[gName][field][start[gName]:stop[gName],:]

    print('Done.')

def combineMultipleHDF5FilesIntoSingle():
    """ Combine multiple groupcat file chunks into a single HDF5 file. """
    simName = 'L35n2160TNG'
    snap    = 33

    loadPath = '/u/dnelson/sims.TNG/%s/output/groups_%03d/' % (simName,snap)
    savePath = '/mnt/nvme/cache/%s/output/groups_%03d/' % (simName,snap)
    fileBase = "fof_subhalo_tab_%03d.%s.hdf5"
    outFile  = "fof_subhalo_tab_%03d.hdf5" % snap

    # metadata
    data = {}
    offsets = {}
    headers = {}

    with h5py.File(loadPath + fileBase % (snap,0), 'r') as f:
        for gName in f.keys():
            if len(f[gName]):
                # group with datasets, e.g. Group, Subhalo, PartType0
                data[gName] = {}
                offsets[gName] = 0
            else:
                # group with no datasets, i.e. only attributes, e.g. Header, Config, Parameters
                headers[gName] = dict( f[gName].attrs.items() )

            for field in f[gName].keys():
                shape = list(f[gName][field].shape)

                # replace first dim with total length
                if gName == 'Group':
                    shape[0] = f['Header'].attrs['Ngroups_Total']
                elif gName == 'Subhalo':
                    shape[0] = f['Header'].attrs['Nsubgroups_Total']
                else:
                    assert 0 # handle
                
                # allocate
                data[gName][field] = np.zeros(shape, dtype=f[gName][field].dtype)

    # load
    nFiles = len(glob.glob(loadPath + fileBase % (snap,'*')))

    for i in range(nFiles):
        print(i,offsets['Group'],offsets['Subhalo'])
        with h5py.File(loadPath + fileBase % (snap,i), 'r') as f:
            for gName in f.keys():
                if len(f[gName]) == 0:
                    continue

                offset = offsets[gName]

                # load and stamp
                for field in f[gName]:
                    length = f[gName][field].shape[0]

                    data[gName][field][offset:offset+length,...] = f[gName][field][()]

                offsets[gName] += length

    # save
    if not path.isdir(savePath):
        mkdir(savePath)

    with h5py.File(savePath + outFile, 'w') as f:
        # add header groups
        for gName in headers:
            f.create_group(gName)
            for attr in headers[gName]:
                f[gName].attrs[attr] = headers[gName][attr]

        f['Header'].attrs['Ngroups_ThisFile'] = f['Header'].attrs['Ngroups_Total']
        f['Header'].attrs['Nsubgroups_ThisFile'] = f['Header'].attrs['Nsubgroups_Total']
        f['Header'].attrs['NumFiles'] = 1

        # add datasets
        for gName in data:
            f.create_group(gName)
            for field in data[gName]:
                f[gName][field] = data[gName][field]
                assert data[gName][field].shape[0] == offsets[gName]

    print('Saved: [%s]' % outFile)

def convertEagleSnapshot(snap=20):
    """ Convert an EAGLE simulation snapshot (HDF5) to a TNG-like snapshot (field names, units, etc). """
    from util.simParams import simParams
    from cosmo.hydrogen import neutral_fraction
    from scipy.ndimage.interpolation import map_coordinates
    from os.path import isdir
    from os import mkdir

    loadPath = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/data/'
    #loadPath = '/virgo/simulations/EagleDM/L0100N1504/DMONLY/data/'
    savePath = '/virgo/simulations/Illustris/Eagle-L68n1504FP/output/'

    gfmPhotoPath = '/u/dnelson/data/Arepo_GFM_Tables_TNG/Photometrics/stellar_photometrics.hdf5'

    sP = simParams(res=1504,run='eagle') # for units only

    metalNamesOrdered = ['Hydrogen','Helium','Carbon','Nitrogen','Oxygen','Neon','Magnesium','Silicon','Iron']
    metalTagsOrdered  = ['MetalMassFracFromSNIa','MetalMassFracFromSNII','MetalMassFracFromAGB','skip','SmoothedIronMassFracFromSNIa','skip']
    photoBandsOrdered = ['U','B','V','K','g','r','i','z']

    fieldRenames = {'Velocity':'Velocities',
                    'Mass':'Masses',
                    'SmoothedMetallicity':'GFM_Metallicity',
                    'InitialMass':'GFM_InitialMass',
                    'StellarFormationTime':'GFM_StellarFormationTime',
                    'BH_CumlAccrMass':'BH_CumMassGrowth_QM',
                    'BH_CumlNumSeeds':'BH_Progs',
                    'BH_AccretionLength':'BH_Hsml'}

    def snapPath(chunkNum):
        return loadPath + snapDir + '/%s.%s.hdf5' % (snapBase,chunkNum)

    def writePath(chunkNum):
        return savePath + 'snapdir_%03d/snap_%03d.%d.hdf5' % (snap,snap,chunkNum)

    # locate the snapshot directory and find number of chunks
    snapPaths = glob.glob(loadPath + 'snapshot_*')
    for path in snapPaths:
        if 'snapshot_%03d' % snap in path:
            snapDir = path.rsplit('/')[-1]

    snapBase = 'snap_%03d_%s' % (snap,snapDir.split('_')[-1])
    nChunks = len(glob.glob(snapPath('*')))

    print('Loading [%d] chunks from: [%s]' % (nChunks,snapDir))

    # load the photometrics table first
    gfm_photo = {}
    with h5py.File(gfmPhotoPath,'r') as f:
        for key in f:
            gfm_photo[key] = f[key][()]

    # output directory
    if not isdir(savePath + 'snapdir_%03d' % snap):
        mkdir(savePath + 'snapdir_%03d' % snap)

    # loop over input chunks
    for chunkNum in range(nChunks):
        print(chunkNum)

        # load full file
        data = {}

        for gName in ['PartType0','PartType1','PartType4','PartType5']:
            data[gName] = {}

        with h5py.File(snapPath(chunkNum), 'r') as f:
            header = dict(f['Header'].attrs)

            # dm
            print(' dm')
            if 'PartType1' in f:
                for key in ['Coordinates','ParticleIDs','Velocity']:
                    data['PartType1'][key] = f['PartType1'][key][()]

            if 'DMONLY' in loadPath:
                continue

            # gas
            print(' gas')
            if 'PartType0' in f:
                for key in ['Coordinates','Density','InternalEnergy','Mass','ParticleIDs',
                            'SmoothedMetallicity','StarFormationRate','Temperature','Velocity']:
                    data['PartType0'][key] = f['PartType0'][key][()]

            # stars
            print(' stars')
            if 'PartType4' in f:
                for key in ['Coordinates','Mass','ParticleIDs','InitialMass',
                            'SmoothedMetallicity','StellarFormationTime','Velocity']:
                    data['PartType4'][key] = f['PartType4'][key][()]

            # gas + stars
            print(' gas+stars')
            for pt in ['PartType0','PartType4']:
                if pt not in f:
                    continue
                data[pt]['GFM_Metals'] = np.zeros( (data[pt]['ParticleIDs'].size,10), dtype='float32' )
                data[pt]['GFM_MetalsTagged'] = np.zeros( (data[pt]['ParticleIDs'].size,6), dtype='float32' )

                for i, el in enumerate(metalNamesOrdered):
                    data[pt]['GFM_Metals'][:,i] = f[pt]['SmoothedElementAbundance'][el][()]
                for i, name in enumerate(metalTagsOrdered):
                    if name == 'skip': continue
                    data[pt]['GFM_MetalsTagged'][:,i] = f[pt][name][()]

            # BHs
            print(' bhs')
            if 'PartType5' in f:
                for key in ['BH_CumlAccrMass','BH_CumlNumSeeds','BH_Density','BH_Mass','BH_Mdot','BH_Pressure',
                            'BH_SoundSpeed', 'BH_SurroundingGasVel', 'BH_WeightedDensity', 'BH_AccretionLength',
                            'Coordinates','Mass','ParticleIDs','Velocity']:
                    data['PartType5'][key] = f['PartType5'][key][()]

                for key in ['BH_BPressure','BH_CumEgyInjection_RM','BH_CumMassGrowth_RM','BH_HostHaloMass','BH_U']: # missing
                    data['PartType5'][key] = np.zeros( data['PartType5']['BH_Mass'].size, dtype='float32' )


        # cleanup header
        for key in ['E(z)','H(z)','RunLabel']:
            del header[key]
        header['Time'] = header.pop('ExpansionFactor')
        header['UnitLength_in_cm'] = 3.08568e+21
        header['UnitMass_in_g'] = 1.989e+43
        header['UnitVelocity_in_cm_per_s'] = 100000
        header['Flag_DoublePrecision'] = 1 if data['PartType0']['Coordinates'].itemsize == 8 else 0

        # field renames
        for pt in data.keys():
            for from_name, to_name in fieldRenames.items():
                if from_name in data[pt]:
                    data[pt][to_name] = data[pt].pop(from_name)

        data['PartType0']['CenterOfMass'] = data['PartType0']['Coordinates']

        # unit conversions
        for pt in data.keys():
            if 'Coordinates' in data[pt]:
                data[pt]['Coordinates'] *= 1e3 # cMpc/h -> cKpc/h
            if 'Velocities' in data[pt]:
                data[pt]['Velocities'] /= np.sqrt(header['Time']) # peculiar -> sqrt(a) units
            if 'GFM_Metallicity' in data[pt]:
                w = np.where(data[pt]['GFM_Metallicity'] < 1e-20)
                data[pt]['GFM_Metallicity'][w] = 1e-20 # GFM_MIN_METAL
            if 'GFM_MetalsTagged' in data[pt]:
                w = np.where(data[pt]['GFM_MetalsTagged'] < 0.0)
                data[pt]['GFM_MetalsTagged'][w] = 0.0 # enforce >=0

        data['PartType0']['Density'] /= 1e9 # Mpc^-3 -> Kpc^-3
        if 'BH_Density' in data['PartType5']:
            data['PartType5']['BH_Density'] /= 1e9

        # gas: ne
        x_h = data['PartType0']['GFM_Metals'][:,0]
        mean_mol_wt = data['PartType0']['Temperature'] * sP.units.boltzmann / ((5.0/3.0-1) * data['PartType0']['InternalEnergy'] * 1e10)
        nelec = (sP.units.mass_proton * 4.0 / mean_mol_wt - 1.0 - 3.0*x_h) / (4*x_h)
        data['PartType0']['ElectronAbundance'] = nelec

        # gas: nH
        sP.redshift = header['Redshift']
        sP.units.scalefac = header['Time']
        nH = sP.units.codeDensToPhys(data['PartType0']['Density'], cgs=True, numDens=True) * data['PartType0']['GFM_Metals'][:,0]
        frac_nH0 = neutral_fraction(nH, sP=None, redshift=header['Redshift'])
        data['PartType0']['NeutralHydrogenAbundance'] = frac_nH0

        # stars: photometrics
        if 'Masses' in data['PartType4']:
            data['PartType4']['GFM_StellarPhotometrics'] = np.zeros( (data['PartType4']['Masses'].size,8), dtype='float32' )

            stars_formz = 1/data['PartType4']['GFM_StellarFormationTime'] - 1
            stars_logagegyr = np.log10( sP.units.redshiftToAgeFlat(header['Redshift']) - sP.units.redshiftToAgeFlat(stars_formz) )
            stars_logz = np.log10( data['PartType4']['GFM_Metallicity'] )
            stars_masslogmsun = sP.units.codeMassToLogMsun( data['PartType4']['GFM_InitialMass'])

            i1 = np.interp( stars_logz, gfm_photo['LogMetallicity_bins'], np.arange(gfm_photo['LogMetallicity_bins'].size) )
            i2 = np.interp( stars_logagegyr,  gfm_photo['LogAgeInGyr_bins'],  np.arange(gfm_photo['LogAgeInGyr_bins'].size) )
            iND = np.vstack( (i1,i2) )

            for i, band in enumerate(photoBandsOrdered):
                mags_1msun = map_coordinates( gfm_photo['Magnitude_%s' % band], iND, order=1, mode='nearest')
                data['PartType4']['GFM_StellarPhotometrics'][:,i] = mags_1msun - 2.5 * stars_masslogmsun
        
        # BHs: bondi and eddington mdot (should be checked more carefully)
        if 'BH_SurroundingGasVel' in data['PartType5']:
            vrel = data['PartType5']['BH_SurroundingGasVel']
            vrel_mag = np.sqrt(vrel[:,0]**2 + vrel[:,1]**2 + vrel[:,2]**2)
            vel_term = (data['PartType5']['BH_SoundSpeed']**2 + vrel_mag**2)**(3.0/2.0)
            mdot_bondi = 4 * np.pi * sP.units.G**2 * data['PartType5']['BH_Mass']**2 * data['PartType5']['BH_Density'] / vel_term
            data['PartType5']['BH_MdotBondi'] = mdot_bondi / 10.22

            data['PartType5']['BH_MdotEddington'] = sP.units.codeBHMassToMdotEdd(data['PartType5']['BH_Mass'], eps_r=0.1) / 10.22

            # BHs: cum egy injection
            bh_E = 0.15 * 0.1 * sP.units.codeMassToMsun(data['PartType5']['BH_CumMassGrowth_QM']) * header['HubbleParam']**2  / header['Time']**2
            bh_E *= sP.units.c_kpc_Gyr**2 * sP.units.redshiftToAgeFlat(header['Redshift']) * 1e9 # (msun/yr) (ckpc/h)^2
            #data['PartType5']['BH_CumEgyInjection_QM'] = bh_E / 10.22 # (msun/yr) -> (1e10 Msun/h)/(0.978 Gyr/h) # needs fix
            data['PartType5']['BH_CumEgyInjection_QM'] = np.zeros( data['PartType5']['BH_Mass'].size, dtype='float32' )

        # debug
        #for pt in data.keys():
        #    for field in data[pt].keys():
        #        print(pt,field,data[pt][field].min(), data[pt][field].max(), data[pt][field].mean())

        # write
        with h5py.File(writePath(chunkNum), 'w') as f:
            # header
            h = f.create_group('Header')
            for key in header:
                h.attrs[key] = header[key]

            # particle groups
            for gName in data:
                g = f.create_group(gName)

                for key in data[gName]:
                    g[key] = data[gName][key]

    print('Done.')

def _addPostprocessingCat(fSim,filepath,baseName,gNames,rootOnly=False):
    """ Helper for createVirtualSimHDF5() below. Add one postprocessing catalog specification in. """
    # catalog exists?
    if not path.isfile(filepath):
        print(' MISSING [%s]! Skipping...' % filepath)
        return

    # open file
    with h5py.File(filepath,'r') as f:
        # loop over groups
        for gName in gNames:
            if gName not in f:
                print(' MISSING [%s]! Skipping...' % gName)
                continue

            # loop over fields
            for field in f[gName]:
                print(' %s %s' % (gName,field))

                if isinstance(f[gName][field], h5py.Dataset):
                    # establish virtual layout
                    shape = f[gName][field].shape
                    if len(shape) == 2 and shape[1] == 1:
                        shape = (shape[0],) # squeeze

                    layout = h5py.VirtualLayout(shape=shape, dtype=f[gName][field].dtype)

                    layout[...] = h5py.VirtualSource(f[gName][field])

                    # add completed virtual dataset into container
                    #if len(f[gName]) == 1:
                    #    # don't think we ever get here in practice (all postprocessing catalogs have 
                    #    # more than 1 dataset) (maybe with my Auxcat's)
                    #    import pdb; pdb.set_trace() # verify the following idea (no group, just dset)
                    #    key = '/%s/%s' % (baseName,field) # or (baseName,gName) ?
                    #    # doesn't work, e.g. Offsets/Group/SnapByType is only dset in Group/
                    #else:
                    #    key = '/%s/%s/%s' % (baseName,gName,field)

                    if len(gNames) == 1:
                        key = '/%s/%s' % (baseName,field)
                    else:
                        key = '/%s/%s/%s' % (baseName,gName,field)

                    if key in fSim:
                        # just redshifts,snaps of tracer_tracks for now
                        print(' skip [%s], already exists.' % key)
                        continue

                    fSim.create_virtual_dataset(key, layout)
                else:
                    if rootOnly:
                        continue # only add datasets directly in specified gNames

                    # nested group, traverse
                    assert isinstance(f[gName][field], h5py.Group)

                    for subfield in f[gName][field]:
                        print(' - %s' % subfield)

                        # establish virtual layout
                        shape = f[gName][field][subfield].shape
                        layout = h5py.VirtualLayout(shape=shape, dtype=f[gName][field][subfield].dtype)

                        layout[...] = h5py.VirtualSource(f[gName][field][subfield])

                        # add completed virtual dataset into container
                        if len(gNames) == 1:
                            key = '/%s/%s/%s' % (baseName,field,subfield)
                        else:
                            key = '/%s/%s/%s/%s' % (baseName,gName,field,subfield)

                        fSim.create_virtual_dataset(key, layout)

def createVirtualSimHDF5():
    """ Create a single 'simulation.hdf5' file which is made up of virtual datasets (HDF5 1.1x/h5py 2.9.x features). 
    Note: dataset details acquired from first chunk of last snapshot! Snapshot must be full, and first chunk must have 
    at least one of every particle type! Note: run in simulation root dir, since we make relative path links. """
    from util.simParams import simParams
    import cosmo

    sP = simParams(run='millennium-2')
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd() # careful

    global_attr_skip = ['Ngroups_ThisFile','Ngroups_Total','Nids_ThisFile','Nids_Total','Nsubgroups_ThisFile','Nsubgroups_Total',
                        'NumFiles','Redshift','Time','Composition_vector_length','Flag_Cooling','Flag_DoublePrecision',
                        'Flag_Feedback','Flag_Metals','Flag_Sfr','Flag_StellarAge','NumFilesPerSnapshot','NumPart_ThisFile',
                        'NumPart_Total','NumPart_Total_HighWord']
    local_attr_include = ['Ngroups_Total','Nsubgroups_Total','Redshift','Time']

    # initialize output
    fSim = h5py.File('simulation.hdf5','w')

    snaps = sP.validSnapList()

    snapsToDo = snaps

    # two big iterations: first for snapshots, then for group catalogs
    for mode in ['snaps','groups']:
        print('Starting [%s]...' % mode)

        if mode == 'snaps':
            chunkPath = cosmo.load.snapPath
            nChunks = cosmo.load.snapNumChunks(sP.simPath, snaps[-1])
            gNames = ['PartType%d' % i for i in range(6)]
            baseName = 'Snapshots'

        if mode == 'groups':
            chunkPath = cosmo.load.gcPath
            nChunks = cosmo.load.groupCatNumChunks(sP.simPath, snaps[-1])
            gNames = ['Group','Subhalo'] 
            baseName = 'Groups'

        # acquire field names, shapes, dtypes, dimensionalities of all datasets from final snapshot
        filepath = chunkPath(sP.simPath, snaps[-1], 0)
        print('Loading all dataset metadata from: %s' % filepath)

        shapes = {}
        dtypes = {}
        ndims  = {}

        with h5py.File(filepath,'r') as f:
            for gName in gNames:
                if gName not in f:
                    continue

                shapes[gName] = {}
                dtypes[gName] = {}
                ndims[gName]  = {}

                for field in f[gName].keys():
                    shapes[gName][field] = f['/%s/%s' % (gName,field)].shape
                    dtypes[gName][field] = f['/%s/%s' % (gName,field)].dtype
                    ndims[gName][field]  = f['/%s/%s' % (gName,field)].ndim

            # insert global Header/Parameters/Config into root
            if mode == 'snaps':
                for gName in ['Config','Header','Parameters']:
                    if gName in f:
                        grp = fSim.create_group(gName)
                        for key,val in f[gName].attrs.items():
                            if key not in global_attr_skip:
                                grp.attrs[key] = val

        # loop over all snapshots
        for snap in snapsToDo:
            # load snapshot and group catalog headers
            print('snap [%3d]' % snap)
            sP.setSnap(snap)

            if sP.simName == 'Illustris-1' and mode == 'snaps' and snap in [53,55]:
                print(' SKIPPING, corrupt...')
                continue 
            
            if mode == 'snaps': header = sP.snapshotHeader()
            if mode == 'groups': header = sP.groupCatHeader()

            # loop over groups (particle types, or groups/subhalos)
            for gName in gNames:
                # get number of elements (particles of this type, or groups/subhalos)
                if mode == 'snaps':
                    ptNum = int(gName[-1])
                    nPt = header['NumPart'][ptNum]
                else:
                    if gName == 'Group': nPt = header['Ngroups_Total']
                    if gName == 'Subhalo': nPt = header['Nsubgroups_Total']

                if nPt == 0:
                    continue # group not present for this snapshot (or ever)

                print(' %s' % gName)

                # loop over fields
                for field in shapes[gName].keys():
                    #print('[%3d] %s %s' % (snap,gName,field))

                    # get field dimensionality and dtype, set full field dataset size
                    if ndims[gName][field] == 1:
                        shape = (nPt,)
                    else:
                        shape = (nPt,shapes[gName][field][1])

                    # establish virtual layout
                    layout = h5py.VirtualLayout(shape=shape, dtype=dtypes[gName][field])

                    # loop over chunks
                    offset = 0
                    present = False

                    for i in range(nChunks):
                        if mode == 'snaps':
                            fpath = 'output/snapdir_%03d/snap_%03d.%d.hdf5' % (snap,snap,i)
                        if mode == 'groups':
                            fpath = 'output/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5' % (snap,snap,i)

                        with h5py.File(fpath,'r') as f:
                            # attach virtual data source subset into layout
                            key = '/%s/%s' % (gName,field)

                            if key not in f:
                                continue # empty chunk for this pt/field

                            present = True
                            vsource = h5py.VirtualSource(f[key])

                            if ndims[gName][field] == 1:
                                layout[offset : offset + vsource.shape[0]] = vsource
                            else:
                                layout[offset : offset + vsource.shape[0],:] = vsource

                            offset += vsource.shape[0]

                    if present:
                        # if not seen in any chunk, then skip (e.g. field missing from minisnap)
                        assert offset == shape[0]

                        # add completed virtual dataset into container
                        key = '/%s/%d/%s/%s' % (baseName,snap,gName,field)
                        fSim.create_virtual_dataset(key, layout)

            # add local Header attributes
            with h5py.File(chunkPath(sP.simPath, snap, 0),'r') as f:
                if 'Header' in f:
                    grp = fSim.create_group('/%s/%d/Header' % (baseName,snap))
                    for attr in local_attr_include:
                        if attr in f['Header'].attrs:
                            grp.attrs[attr] = f['Header'].attrs[attr]

            if mode == 'snaps':
                grp.attrs['NumPart_Total'] = header['NumPart'] # int64

        print('[%s] done.\n' % mode)

    # postprocessing, one file per snapshot
    modes = ['axisratios',
             'circularities/10Re',
             'circularities/allstars',
             #hih2
             #InfallCatalog
             #InSituFraction
             #MergerHistory
             #sizes_projected
             #skirt_images
             'StarFormationRates',
             'StellarAssembly/galaxies',
             'StellarAssembly/galaxies_in_rad',
             'StellarAssembly/stars',
             #stellar_light
             'StellarMasses/Group',
             'StellarMasses/Subhalo',
             'offsets',
             'SubhaloMatchingToDark/SubLink',
             'SubhaloMatchingToDark/LHaloTree']
             #SubhaloMatchingToIllustris
             #SubhaloMatchingToLowRes
             #VirialMassesType

    for mode in modes:
        print('Starting [%s]...' % mode)

        for snap in snapsToDo:
            # load snapshot and group catalog headers
            print('snap [%3d]' % snap)

            if mode == 'axisratios':
                filepath = 'postprocessing/axisratios/axisratios_%s%03d.hdf5' % (sP.simNameAlt,snap)
                baseName = 'Groups/%d/Subhalo/axisratios' % snap
                gNames = ['/']

            if mode == 'circularities/10Re':
                filepath = 'postprocessing/circularities/circularities_aligned_10Re_%s%03d.hdf5' % (sP.simNameAlt,snap)
                baseName = 'Groups/%d/Subhalo/circularities_10Re' % snap
                gNames = ['/']

            if mode == 'circularities/allstars':
                filepath = 'postprocessing/circularities/circularities_aligned_allstars_%s%03d.hdf5' % (sP.simNameAlt,snap)
                baseName = 'Groups/%d/Subhalo/circularities_allstars' % snap
                gNames = ['/']

            if mode == 'StarFormationRates':
                filepath = 'postprocessing/StarFormationRates/Subhalo_SFRs_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/StarFormationRates' % snap
                gNames = ['Subhalo']

            if mode == 'StellarAssembly/galaxies':
                filepath = 'postprocessing/StellarAssembly/galaxies_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/StellarAssembly' % snap
                gNames = ['/']

            if mode == 'StellarAssembly/galaxies_in_rad':
                filepath = 'postprocessing/StellarAssembly/galaxies_in_rad_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/StellarAssemblyInRad' % snap
                gNames = ['/']

            if mode == 'StellarAssembly/stars':
                filepath = 'postprocessing/StellarAssembly/stars_%03d.hdf5' % snap
                baseName = 'Snapshots/%d/PartType4/StellarAssembly' % snap
                gNames = ['/']

            if mode == 'StellarMasses/Group':
                filepath = 'postprocessing/StellarMasses/Group_3DStellarMasses_%03d.hdf5' % snap
                baseName = 'Groups/%d/Group/StellarMasses' % snap
                gNames = ['Group']

            if mode == 'StellarMasses/Subhalo':
                filepath = 'postprocessing/StellarMasses/Subhalo_3DStellarMasses_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/StellarMasses' % snap
                gNames = ['Subhalo']

            if mode == 'offsets':
                # could instead move into Snapshots/N/Group/Offsets/ and Snapshots/N/Subhalo/Offsets/
                filepath = 'postprocessing/offsets/offsets_%03d.hdf5' % snap
                baseName = 'Offsets/%d' % snap
                gNames = ['Group','Subhalo']

            if mode == 'SubhaloMatchingToDark/SubLink':
                filepath = 'postprocessing/SubhaloMatchingToDark/SubLink_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/SubhaloMatchingToDark/SubLink' % snap
                gNames = ['/']

            if mode == 'SubhaloMatchingToDark/LHaloTree':
                filepath = 'postprocessing/SubhaloMatchingToDark/LHaloTree_%03d.hdf5' % snap
                baseName = 'Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree' % snap
                gNames = ['/']

            _addPostprocessingCat(fSim,filepath,baseName,gNames)

        print('[%s] done.\n' % mode)

    # postprocesisng, one file per simulation
    modes_sim = ['trees/SubLink',
                 'trees/SubLink_gal']
                 #'trees/LHaloTree'] # terrible structure with too many groups...
                 #SubboxSubhaloList

    for mode in modes_sim:
        print('Starting [%s]...' % mode)

        if mode == 'trees/SubLink':
            filepath = 'postprocessing/trees/SubLink/tree_extended.hdf5'
            baseName = 'Trees/SubLink' % snap
            gNames = ['/']

        if mode == 'trees/SubLink_gal':
            filepath = 'postprocessing/trees/SubLink_gal/tree_extended.hdf5'
            baseName = 'Trees/SubLink_gal' % snap
            gNames = ['/']

        _addPostprocessingCat(fSim,filepath,baseName,gNames)

        print('[%s] done.\n' % mode)

    # tracer_tracks: custom addition into PartType2 (final snapshot only)
    print('Starting [tracer_tracks]...')

    if 1:
        # meta
        snap = snaps[-1]

        filepath = 'postprocessing/tracer_tracks/tr_all_groups_%d_meta.hdf5' % snap

        _addPostprocessingCat(fSim,filepath,'Groups/%d/Group' % snap,['Halo'])
        _addPostprocessingCat(fSim,filepath,'Groups/%d/Subhalo' % snap,['Subhalo'])

        _addPostprocessingCat(fSim,filepath,'Snapshots/%d/PartType2' % snap,['/'],rootOnly=True)

        # add all other known tracks
        filepaths = glob.glob('postprocessing/tracer_tracks/tr_all_groups*.hdf5')
        for filepath in filepaths:
            if '_meta' in filepath:
                continue

            _addPostprocessingCat(fSim,filepath,'Snapshots/%d/PartType2' % snap,['/'])

    print('[tracer_tracks] done.\n')

    print('All done.')
    fSim.close()

def supplementVirtualSimHDF5AddSnapField():
    """ Add to existing 'simulation.hdf5' file (modify as needed, careful!). """
    from util.simParams import simParams
    import cosmo

    sP = simParams(res=2500,run='tng')
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd() # careful

    # open (append mode)
    fSim = h5py.File('simulation.hdf5','r+')
    snaps = sP.validSnapList()

    # start custom
    chunkPath = cosmo.load.snapPath
    nChunks  = cosmo.load.snapNumChunks(sP.simPath, snaps[-1])
    gName    = 'PartType0'
    field    = 'InternalEnergyOld'
    baseName = 'Snapshots'

    # acquire field name, shape, dtype of dataset from final snapshot
    filepath = chunkPath(sP.simPath, snaps[-1], 0)

    with h5py.File(filepath,'r') as f:
        shape = f['/%s/%s' % (gName,field)].shape
        dtype = f['/%s/%s' % (gName,field)].dtype
        ndim  = f['/%s/%s' % (gName,field)].ndim

    # loop over all snapshots
    for snap in snaps:
        # load snapshot and group catalog headers
        print('snap [%3d]' % snap)
        sP.setSnap(snap)

        if sP.simName == 'Illustris-1' and snap in [53,55]:
            print(' SKIPPING, corrupt...')
            continue 
        
        header = sP.snapshotHeader()

        # get number of particles of this type
        ptNum = int(gName[-1])
        nPt = header['NumPart'][ptNum]

        if nPt == 0:
            continue # group not present for this snapshot (or ever)

        # set full field dataset size
        if ndim == 1:
            shape = (nPt,)
        else:
            shape = (nPt,shape[1])

        # establish virtual layout
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

        # loop over chunks
        offset = 0
        present = False

        for i in range(nChunks):
            fpath = 'output/snapdir_%03d/snap_%03d.%d.hdf5' % (snap,snap,i)

            with h5py.File(fpath,'r') as f:
                # attach virtual data source subset into layout
                key = '/%s/%s' % (gName,field)

                if key not in f:
                    continue # empty chunk for this pt/field

                present = True
                vsource = h5py.VirtualSource(f[key])

                if ndim == 1:
                    layout[offset : offset + vsource.shape[0]] = vsource
                else:
                    layout[offset : offset + vsource.shape[0],:] = vsource

                offset += vsource.shape[0]

        if present:
            # if not seen in any chunk, then skip (e.g. field missing from minisnap)
            assert offset == shape[0]

            # completed virtual dataset: already exists?
            key = '/%s/%d/%s/%s' % (baseName,snap,gName,field)

            if key in fSim:
                #del fSim[key] # remove link
                import pdb; pdb.set_trace() # check

            # add completed virtual dataset into container
            fSim.create_virtual_dataset(key, layout)

    # finish custom
    fSim.close()

def supplementVirtualSimHDF5AddOrUpdateGroupcatField():
    """ Add to existing 'simulation.hdf5' file (modify as needed, careful!). """
    from util.simParams import simParams
    import cosmo

    sP = simParams(res=1820,run='illustris')
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd() # careful

    # open (append mode)
    fSim = h5py.File('simulation.hdf5','r+')
    snaps = sP.validSnapList()

    # start custom
    chunkPath = cosmo.load.gcPath
    nChunks   = cosmo.load.groupCatNumChunks(sP.simPath, snaps[-1])
    gName     = 'Subhalo'
    field     = 'SubhaloFlag'
    baseName  = 'Groups'

    # acquire field name, shape, dtype of dataset from final snapshot
    filepath = chunkPath(sP.simPath, snaps[-1], 0)

    with h5py.File(filepath,'r') as f:
        shape = f['/%s/%s' % (gName,field)].shape
        dtype = f['/%s/%s' % (gName,field)].dtype
        ndim  = f['/%s/%s' % (gName,field)].ndim

    # loop over all snapshots
    for snap in snaps:
        # load snapshot and group catalog headers
        print('snap [%3d]' % snap)
        sP.setSnap(snap)
        
        header = sP.groupCatHeader()

        # get number of elements (particles of this type, or groups/subhalos)
        if gName == 'Group': nPt = header['Ngroups_Total']
        if gName == 'Subhalo': nPt = header['Nsubgroups_Total']

        if nPt == 0:
            continue # group not present for this snapshot (or ever)

        # get field dimensionality and dtype, set full field dataset size
        if ndim == 1:
            shape = (nPt,)
        else:
            shape = (nPt,shape[1])

        # establish virtual layout
        layout = h5py.VirtualLayout(shape=shape, dtype=dtype)

        # loop over chunks
        offset = 0
        present = False

        for i in range(nChunks):
            fpath = 'output/groups_%03d/fof_subhalo_tab_%03d.%d.hdf5' % (snap,snap,i)

            with h5py.File(fpath,'r') as f:
                # attach virtual data source subset into layout
                key = '/%s/%s' % (gName,field)

                if key not in f:
                    continue # empty chunk for this pt/field

                present = True
                vsource = h5py.VirtualSource(f[key])

                if ndim == 1:
                    layout[offset : offset + vsource.shape[0]] = vsource
                else:
                    layout[offset : offset + vsource.shape[0],:] = vsource

                offset += vsource.shape[0]

        if present:
            # if not seen in any chunk, then skip (e.g. field missing from minisnap)
            assert offset == shape[0]

            # completed virtual dataset: already exists?
            key = '/%s/%d/%s/%s' % (baseName,snap,gName,field)

            if key in fSim:
                del fSim[key] # remove link
                print(' - removed existing [%s] and created new.' % key)

            # add completed virtual dataset into container
            fSim.create_virtual_dataset(key, layout)

    # finish custom
    fSim.close()

def supplementVirtualSimHDF5():
    """ Add to existing 'simulation.hdf5' file (modify as needed, careful!). """
    from util.simParams import simParams
    import cosmo

    sP = simParams(res=1820,run='tng_dark')
    assert sP.simName in getcwd() or sP.simNameAlt in getcwd() # careful

    # open (append mode)
    fSim = h5py.File('simulation.hdf5','r+')
    snaps = sP.validSnapList()

    # start custom
    if 1:
        # add missing Offsets
        for snap in snaps:
            # load snapshot and group catalog headers
            print('snap [%3d]' % snap)

            filepath = 'postprocessing/offsets/offsets_%03d.hdf5' % snap
            baseName = 'Offsets/%d' % snap
            gNames = ['Group','Subhalo']

            _addPostprocessingCat(fSim,filepath,baseName,gNames)

    if 0:
        # add new LHaloTree catalogs and delete old
        for snap in snaps:
            print('snap [%3d]' % snap)

            filepath = 'postprocessing/SubhaloMatchingToDark/LHaloTree_%03d.hdf5' % snap
            baseName = 'Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree' % snap
            gNames = ['/']

            _addPostprocessingCat(fSim,filepath,baseName,gNames)

        for snap in snaps:
            gName = '/Groups/%d/Subhalo/SubhaloMatchingToDark/LHaloTree/' % snap
            keys = ['SubhaloIndexFrom','SubhaloIndexTo']

            for key in keys:
                if gName+key in fSim:
                    print('delete [%s]' % (gName+key))
                    del fSim[gName+key]

    if 0:
        # remove Bfld group cat fields from mini-snaps
        tngFullSnaps = [2,3,4,6,8,11,13,17,21,25,33,40,50,59,67,72,78,84,91,99]
        fields = ['SubhaloBfldDisk','SubhaloBfldHalo']

        for snap in snaps:
            if snap in tngFullSnaps:
                continue
            for field in fields:
                key = '/Groups/%d/Subhalo/%s' % (snap,field)

                if key in fSim:
                    print('delete [%s]' % (key))
                    del fSim[key]
    # finish custom

    fSim.close()

def vdsTest():

    import numpy as np
    np.random.seed(424242)

    size_full = 10
    size_sparse = 5
    num_files = 3

    # Create source files (1.h5 to 3.h5)
    for n in range(1, num_files+1):
        with h5py.File('{}.h5'.format(n), 'w') as f:
            data = np.random.uniform(size=size_sparse) #np.arange(size_sparse) + 10**n
            f['data'] = data

            inds = np.arange(size_full)
            np.random.shuffle(inds) # create dummy indices which cover the full dset size
            inds = np.sort(inds[0:size_sparse]) # must be ascending
            f['inds'] = inds

    import pdb; pdb.set_trace()

    # Assemble virtual dataset
    layout = h5py.VirtualLayout(shape=(num_files, size_full), dtype=data.dtype)

    for n in range(1, num_files+1):
        filename = "{}.h5".format(n)
        with h5py.File(filename,'r') as f:
            inds = f['inds'][()]
            vsource = h5py.VirtualSource(f['data'])
            layout[n - 1, inds] = vsource

    # Add virtual dataset to output file
    with h5py.File("VDS.h5", 'w', libver='latest') as f:
        f.create_virtual_dataset('data', layout, fillvalue=np.nan)
        print("Virtual dataset:")
        print(f['data'][:,:])

def convertVoronoiConnectivityVPPP(stage=1, thisTask=0):
    """ Read the Voronoi mesh data from Chris Byrohl using his vppp (voro++ parallel) approach, save to HDF5. """
    from util.simParams import simParams
    from util.helper import pSplitRange

    sP = simParams(run='tng50-2', redshift=0.5)
    basepath = "/freya/ptmp/mpa/cbyrohl/public/vppp_dataset/IllustrisTNG50-2_z0.5_posdata"

    file1 = basepath + ".bin.nb"
    file2 = basepath + ".bin.nb2"

    outfile1 = "/u/dnelson/sims.TNG/L35n1080TNG/data.files/voronoi/mesh_spatialorder_%02d.hdf5" % sP.snap
    outfile2 = "/u/dnelson/sims.TNG/L35n1080TNG/data.files/voronoi/mesh_%02d.hdf5" % sP.snap

    dtype_nb = np.dtype([
        ("x", np.float64),
        ("y", np.float64),
        ("z", np.float64),
        ("gidx", np.int64), # snapshot index (1-indexed !!!!!)
        ("noffset", np.int64), #offset in neighbor list (1-indexed !!!!!)
        ("ncount", np.int32), #neighborcount
    ])

    # convert stage (1): rewrite into HDF5
    if stage == 1:
        # chunked load
        chunksize = 100000000

        # get npart and ngb list size
        with open(file1, "rb") as f:
            f.seek(0)
            npart = np.fromfile(f,dtype=np.int64,count=1)[0]

        with open(file2,"rb") as f:
            f.seek(0)
            tot_num_entries = np.fromfile(f, dtype=np.int64, count=1)[0]

        # open save file
        fOut = h5py.File(outfile1,"w")

        snap_index = fOut.create_dataset("snap_index", (npart,), dtype='int64')
        num_ngb    = fOut.create_dataset("num_ngb", (npart,), dtype='int16')
        offset_ngb = fOut.create_dataset("offset_ngb", (npart,), dtype='int64')
        x          = fOut.create_dataset("x", (npart,), dtype='float32')
        ngb_inds   = fOut.create_dataset("ngb_inds", (tot_num_entries,), dtype='int64') # (1-indexed !!)

        # load all from file1
        with open(file1, "rb") as f:
            # get npart
            f.seek(0)

            nloaded = 0
            byte_offset = 8 # skip npart

            while nloaded < npart:
                print('loaded %4.1f%% [%10d] of [%10d]' % (nloaded/npart*100,nloaded,npart))
                f.seek(byte_offset)
                data = np.fromfile(f, dtype=dtype_nb, count=chunksize)

                # save
                snap_index[nloaded : nloaded + data.size] = data['gidx'] - 1 # change from 1-based fortran indexing
                num_ngb[nloaded : nloaded + data.size] = data['ncount']
                offset_ngb[nloaded : nloaded + data.size] = data['noffset'] - 1 # change from 1-based fortran indexing
                x[nloaded : nloaded + data.size] = data['x']

                # continue
                nloaded += data.size
                byte_offset += data.size * dtype_nb.itemsize

        # load neighbor list from file2
        with open(file2,"rb") as f:
            f.seek(0)

            nloaded = 0
            byte_offset = 8 # skip tot_num_entries

            while nloaded < tot_num_entries:
                print('ngblist %4.1f%% [%10d] of [%10d]' % (nloaded/tot_num_entries*100,nloaded,tot_num_entries))
                f.seek(byte_offset)
                data = np.fromfile(f, dtype=np.int64, count=chunksize*10)

                if data.size == 0:
                    break

                # save
                ngb_inds[nloaded : nloaded + data.size] = data - 1 # change from 1-based fortran indexing

                # continue
                nloaded += data.size
                byte_offset += data.size * data.itemsize

        fOut.close()

    # convert stage (2): shuffle into snapshot order
    if stage == 2:
        with h5py.File(outfile1,'r+') as f:
            snap_index = f['snap_index'][()]

            sort_inds = np.argsort(snap_index)
            f['sort_inds'] = sort_inds

    # sanity checks A
    if stage == 3:
        with h5py.File(outfile1,'r') as f:
            snap_index = f['snap_index'][()]
            sort_inds = f['sort_inds'][()]

        # check: indices are dense
        new_snap_index = snap_index[sort_inds]

        numGas = sP.snapshotHeader()['NumPart'][sP.ptNum('gas')]
        lin_list = np.arange(numGas)
        print(np.array_equal(lin_list,new_snap_index))

    # sanity checks B
    if stage == 4:
        with h5py.File(outfile1,'r+') as f:
            x = f['x'][()] * sP.boxSize # [0,1] -> [0,sP.boxSize]
            sort_inds = f['sort_inds'][()]

        # check: order is correct by comparing x-coordinates
        new_x = x[sort_inds]
        snap_x = sP.snapshotSubsetP('gas','pos_x')

        print(new_x[0:5], snap_x[0:5])
        print(np.allclose(new_x,snap_x))

    # stage (5): save spatial domain information for non-groupordered datafile
    if stage == 5:
        with open(basepath.replace("posdata","domains.txt")) as f:
            lines = f.readlines()

        #Format (4 lines per chunk): (all lengths/starts in numbers of entries, not bytes)
        #CHUNKID, NB2 start, NB2 length
        #NB1 start, NB1 length
        #xstart, ystart, zstart
        #xend, yend, zend
        nChunks = 512
        assert len(lines) == nChunks * 4 # 512 cubic spatial subsets

        # allocate
        r = {'chunk_id'     : np.zeros( nChunks, dtype='int32' ),
             'offset_ngb'   : np.zeros( nChunks, dtype='int64' ),
             'num_ngb'      : np.zeros( nChunks, dtype='int64' ),
             'offset_cells' : np.zeros( nChunks, dtype='int64' ),
             'num_cells'    : np.zeros( nChunks, dtype='int64' ),
             'xyz_min'      : np.zeros( (nChunks,3), dtype='float32' ),
             'xyz_max'      : np.zeros( (nChunks,3), dtype='float32' )}

        # parse
        for i in np.arange(0, nChunks):
            r['chunk_id'][i], r['offset_ngb'][i], r['num_ngb'][i] = [int(x) for x in lines[i*4+0].split()]
            r['offset_cells'][i], r['num_cells'][i] = [int(x) for x in lines[i*4+1].split()]
            r['xyz_min'][i,:] = [float(x) for x in lines[i*4+2].split()]
            r['xyz_max'][i,:] = [float(x) for x in lines[i*4+3].split()]

        # sanity checks
        nCells = sP.snapshotHeader()['NumPart'][sP.ptNum('gas')]
        assert np.array_equal(r['chunk_id'], np.arange(nChunks))
        assert r['offset_ngb'].min() >= 0 and r['num_ngb'].min() >= 0
        assert r['offset_cells'].min() >= 0 and r['num_cells'].min() >= 0 and r['offset_cells'].max() < nCells
        assert r['xyz_min'].min() >= 0 and r['xyz_min'].max() <= 1.0
        assert r['xyz_max'].min() >= 0 and r['xyz_max'].max() <= 1.0

        # save into spatially ordered datafile
        with h5py.File(outfile1,'r+') as f:
            for key in r:
                f['meta/%s' % key] = r[key]
        print('Saved spatial metadata.')

    # convert stage (6): create new final 'mesh' file with shuffled num_ngb and offset_ngb
    if stage == 6:
        # metadata
        with h5py.File(outfile1,'r') as f:
            sort_inds = f['sort_inds'][()]
            tot_num_entries = f['ngb_inds'].size

        # read and write per-cell datasets
        for key in ['num_ngb','offset_ngb']:
            with h5py.File(outfile1,'r') as f:
                data = f[key][()]
            with h5py.File(outfile2,'a') as f:
                f[key] = data[sort_inds]
            print(key, flush=True)

        # allocate unfilled neighbor list
        with h5py.File(outfile2,'r+') as f:
            ngb_inds = f.create_dataset("ngb_inds", (tot_num_entries,), dtype='int64')
            for i in range(20):
                locrange = pSplitRange( [0,tot_num_entries], 20, i)
                print(i,locrange,flush=True)
                ngb_inds[locrange[0]:locrange[1]] = -1
        
    # convert stage (7): rewrite ngb_inds into dense, contiguous subsets following snapshot order
    nTasks = 10 #140

    if stage == 7:
        with h5py.File(outfile2,'r') as f:
            num_cells = f['num_ngb'].size

        locRange = pSplitRange( [0,num_cells], nTasks, thisTask)

        # load original offsets
        with h5py.File(outfile2,'r') as f:
            num_ngb = f['num_ngb'][locRange[0]:locRange[1]]
            offset_ngb = f['offset_ngb'][locRange[0]:locRange[1]]

        # allocate sub-task output file
        subfile = outfile2.replace(".hdf5","_%d_of_%d.hdf5" % (thisTask,nTasks))
        totNumNgbLoc = num_ngb.sum()
        ngb_inds = np.zeros( totNumNgbLoc, dtype='int64' ) - 1

        print('[%2d of %2d] starting... ' % (thisTask,nTasks), locRange, flush=True)

        offset = 0

        with h5py.File(outfile1,'r') as f: # open source
            for i in range(num_ngb.size):
                if i % 100000 == 0:
                    print('[%2d] [%10d] %.2f%%' % (thisTask,i,i/num_ngb.size*100), flush=True)

                # read
                loc_inds = f['ngb_inds'][offset_ngb[i] : offset_ngb[i]+num_ngb[i]]

                ngb_inds[offset:offset+num_ngb[i]] = loc_inds

                offset += num_ngb[i]

        # save
        with h5py.File(subfile,'w') as f:
            f['ngb_inds'] = ngb_inds

    # convert stage (8): concatenate ngb_inds
    if stage == 8:
        # note: do not need to permute ngb_inds, since they are indices into the snapshot, not into the nb1 file
        global_min = np.inf
        global_max = -1

        offset = 0

        for i in range(nTasks):
            subfile = outfile2.replace(".hdf5","_%d_of_%d.hdf5" % (i,nTasks))
            with h5py.File(subfile,'r') as f:
                loc_inds = f['ngb_inds'][()]

            print(i, loc_inds.min(), loc_inds.max(), flush=True)

            if loc_inds.max() > global_max:
                global_max = loc_inds.max()
            if loc_inds.min() < global_min:
                global_min = loc_inds.min()

            # save
            with h5py.File(outfile2,'r+') as f:
                f['ngb_inds'][offset:offset+loc_inds.size] = loc_inds

            offset += loc_inds.size

        print('global min: ', global_min)
        print('global max: ', global_max)
        print('final offset: ', offset)

        with h5py.File(outfile1,'r') as f:
            print('should equal: ', f['ngb_inds'].size)

    # convert stage (9): new offset_ngb
    if stage == 9:
        with h5py.File(outfile2,'r+') as f:
            num_ngb = f['num_ngb'][()]
            offset_ngb = np.zeros(num_ngb.size, dtype='int64')
            offset_ngb[1:] = np.cumsum(num_ngb)[:-1]
            f['offset_ngb'][:] = offset_ngb

    print('done.')
