"""
dataConvert.py
  Various data exporters/converters, between different formats, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os import path, mkdir
import glob

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

    sP = simParams(res=1820,run=run,redshift=0.0)
    ptTypes = ['gas','dm','bhs','stars']

    # subhaloIDs
    samplePath = '/home/extdylan/sims.TNG/L75n1820TNG/data.files/new_mw_sample_fgas.txt'
    data = np.genfromtxt(samplePath, delimiter=',', dtype='int32')

    # subhalo indices (z=0): TNG100-1, Illustris-1 (Lagrangian match), Illustris-1 (positional match)
    if run == 'tng':
        subhaloIDs = data[:,0]
    if run == 'illustris':
        subhaloIDs = data[:,1] # Lagrangian match

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

        data = {}

        # load (subhalo restricted)
        for partType in ptTypes:
            print(subhaloID,'sub',partType)
            gName = 'PartType' + str(sP.ptNum(partType))

            data[gName] = cosmo.load.snapshotSubset(sP, partType, fields[gName], subhaloID=subhaloID)

        # write
        with h5py.File('cutout_%s_%d_subhalo.hdf5' % (sP.simName,subhaloID),'w') as f:
            for gName in data:
                g = f.create_group(gName)
                for field in data[gName]:
                    g[field] = data[gName][field]

        # get parent fof, load (fof restricted)
        data = {}
        subh = cosmo.load.groupCatSingle(sP, subhaloID=subhaloID)
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
    sP = simParams(res=1820,run='tng',redshift=0.0)
    data = np.genfromtxt(sP.derivPath + 'new_mw_sample_fgas.txt', delimiter=',', dtype='int32')
    subhaloIDs = data[:,0]

    # list of tracer quantities we know
    catBasePath = sP.postPath + 'tracer_tracks/*.hdf5'
    cats = {}

    for catPath in glob.glob(catBasePath):
        catName = catPath.split("99_")[-1].split(".hdf5")[0]
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

        # load from each existing cat
        for catName, catPath in cats.iteritems():
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
        with h5py.File('tracers_%s_%d_subhalo.hdf5' % (sP.simName,subhaloID),'w') as f:
            for ptType in ptTypes:
                g = f.create_group(ptType)
                for field in data[ptType]:
                    g[field] = data[ptType][field]
            f['snaps'] = snaps
            f['redshifts'] = redshifts

def makeSnapHeadersForLHaloTree():
    """ Copy chunk 0 of each snapshot only and the header only (for LHaloTree B-HaloTree). """
    nSnaps = 100

    pathFrom = '/home/extdylan/sims.TNG/L205n2500TNG_DM/output/snapdir_%03d/'
    pathTo   = '/home/extdylan/test/snapdir_%03d/'
    fileFrom = pathFrom + 'snap_%03d.%s.hdf5'
    fileTo   = pathTo + 'snap_%03d.%s.hdf5'

    # loop over snapshots
    for i in range(nSnaps):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # open destination for writing
        j = 0
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
    copyFields = {'PartType0':['Masses','Coordinates','Density','GFM_Metals'],
                  'PartType1':['ParticleIDs'],
                  'PartType4':['Masses','GFM_StellarFormationTime','ParticleIDs']}

    copyFields = {'PartType4':['Coordinates','ParticleIDs']}
    copyFields = {'PartType0':['Masses','Coordinates','Density','GFM_Metals','GFM_Metallicity']}

    pathFrom = '/home/extdylan/sims.TNG/L205n2500TNG/output/snapdir_%03d/'
    pathTo   = '/home/extdylan/data/out/L205n2500TNG/output/snapdir_%03d/'
    fileFrom = pathFrom + 'snap_%03d.%s.hdf5'
    fileTo   = pathTo + 'snap_%03d.%s.hdf5'

    # L205 config
    nChunks = 600
    if '_DM' in pathFrom: nChunks = 75

    # verify number of chunks
    files = glob.glob(fileFrom % (0,0,'*'))
    assert len(files) == nChunks

    # loop over snapshots
    for i in [99]: #range(50,61):
        if not path.isdir(pathTo % i):
            mkdir(pathTo % i)

        # loop over chunks
        for j in range(nChunks):
            # open destination for writing
            fOut = h5py.File(fileTo % (i,i,j), 'w')

            # open origin file for reading
            assert path.isfile(fileFrom % (i,i,j))

            with h5py.File(fileFrom % (i,i,j),'r') as f:
                # copy header
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
