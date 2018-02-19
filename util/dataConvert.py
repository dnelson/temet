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
    from cosmo.load import groupCat
    sP = simParams(res=1820,run='illustris',redshift=redshift)

    # load
    mstar = groupCat(sP, fieldsSubhalos=['mstar_30pkpc_log']) # log msun
    sfr   = groupCat(sP, fieldsSubhalos=['SubhaloSFRinRad'])['subhalos'] # msun/yr
    cen   = groupCat(sP, fieldsSubhalos=['central_flag'])
    mhalo = groupCat(sP, fieldsSubhalos=['mhalo_subfind_log']) # log msun
    m200  = groupCat(sP, fieldsSubhalos=['mhalo_200_log'])

    sat = (~cen.astype('bool')).astype('int16')

    w = np.where(np.isnan(mstar)) # zero stellar mass
    mstar[w] = 0.0

    w = np.where(cen == 1)
    mhalo[w] = m200 # use m200,crit values for centrals at least

    w = np.where(mstar >= 7.8)
    mstar = mstar[w]
    sfr = sfr[w]
    cen = cen[w]
    mhalo = mhalo[w]
    sat = sat[w]

    with open('inputstats_%s_z%s_new.txt' % (sP.simName,int(redshift)),'w') as f:
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

    savePath = '/u/dnelson/data/sims.millennium/Millennium1/output/'
    loadPath = '/virgo/simulations/Millennium/' 

    dm_particle_mass = 0.0860657 # ~10^9 msun

    if not path.isdir(savePath + 'groups_%03d' % snap):
        mkdir(savePath + 'groups_%03d' % snap)

    def _chunkPath(snap, chunkNum):
        return loadPath + 'postproc_%03d/sub_tab_%03d.%s' % (snap,snap,chunkNum)
    def _groupChunkPath(snap, chunkNum):
        return loadPath + 'snapdir_%03d/group_tab_%03d.%s' % (snap,snap,chunkNum)

    nChunks = len( glob.glob(_chunkPath(snap,'*')) )
    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    # reader header of first chunk
    with open(_chunkPath(snap,0),'rb') as f:
        header = f.read(4*5)

    NGroups    = struct.unpack('i', header[0:4])[0]
    NIds       = struct.unpack('i', header[4:8])[0]
    TotNGroups = struct.unpack('i', header[8:12])[0]
    NFiles     = struct.unpack('i', header[12:16])[0]
    NSubs      = struct.unpack('i', header[16:20])[0]

    assert NFiles == nChunks

    # no TotNSubs stored...
    TotNSubs = 0
    TotNGroupsCheck = 0

    for i in range(nChunks):
        with open(_chunkPath(snap,i),'rb') as f:
            header  = f.read(4*5)
            NGroups = struct.unpack('i', header[0:4])[0]
            NSubs   = struct.unpack('i', header[16:20])[0]

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
        NGroups = struct.unpack('i', data[0:4])[0]

        GroupLen[g_off : g_off + NGroups]    = struct.unpack('i' * NGroups, data[16 : 16 + 4*NGroups])
        GroupOffset[g_off : g_off + NGroups] = struct.unpack('i' * NGroups, data[16 + 4*NGroups : 16 + 8*NGroups])

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
        NGroups    = struct.unpack('i', data[0:4])[0]
        NSubs      = struct.unpack('i', data[16:20])[0]

        # per halo
        header_bytes = 20
        off = header_bytes
        NSubsPerHalo[g_off : g_off + NGroups]   = struct.unpack('i' * NGroups, data[off + 0*NGroups : off + 4*NGroups])
        FirstSubOfHalo[g_off : g_off + NGroups] = struct.unpack('i' * NGroups, data[off + 4*NGroups : off + 8*NGroups])
        FirstSubOfHalo[g_off : g_off + NGroups] += s_off # as stored is local to chunk files

        # per subhalo
        off = header_bytes + 8*NGroups
        SubLen[s_off : s_off + NSubs]        = struct.unpack('i' * NSubs, data[off + 0*NSubs : off + 4*NSubs])
        SubOffset[s_off : s_off + NSubs]     = struct.unpack('i' * NSubs, data[off + 4*NSubs : off + 8*NSubs])
        SubParentHalo[s_off : s_off + NSubs] = struct.unpack('i' * NSubs, data[off + 8*NSubs : off + 12*NSubs])
        SubParentHalo[s_off : s_off + NSubs] += g_off # as stored is local to chunk files

        # per subhalo chunk-pointer information
        SubFileNr[s_off : s_off + NSubs] = i
        SubLocalIndex[s_off : s_off + NSubs] = np.arange(NSubs)

        # per halo
        off = header_bytes + 8*NGroups + 12*NSubs
        Halo_M_Mean200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 0*NGroups: off + 4*NGroups])
        Halo_R_Mean200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 4*NGroups: off + 8*NGroups])
        Halo_M_Crit200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 8*NGroups: off + 12*NGroups])
        Halo_R_Crit200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 12*NGroups: off + 16*NGroups])
        Halo_M_TopHat200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 16*NGroups: off + 20*NGroups])
        Halo_R_TopHat200[g_off : g_off + NGroups] = struct.unpack('f' * NGroups, data[off + 20*NGroups: off + 24*NGroups])

        # per subhalo
        off = header_bytes + 8*NGroups + 12*NSubs + 24*NGroups
        SubPos[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack('f' * 3*NSubs, data[off + 0*NSubs : off + 12*NSubs]), (NSubs,3) )
        SubVel[s_off : s_off + NSubs,:]   = np.reshape( struct.unpack('f' * 3*NSubs, data[off + 12*NSubs : off + 24*NSubs]), (NSubs,3) )
        SubVelDisp[s_off : s_off + NSubs]             = struct.unpack('f' * 1*NSubs, data[off + 24*NSubs : off + 28*NSubs])
        SubVmax[s_off : s_off + NSubs]                = struct.unpack('f' * 1*NSubs, data[off + 28*NSubs : off + 32*NSubs])
        SubSpin[s_off : s_off + NSubs,:]  = np.reshape( struct.unpack('f' * 3*NSubs, data[off + 32*NSubs : off + 44*NSubs]), (NSubs,3) )
        SubMostBoundID[s_off : s_off + NSubs]         = struct.unpack('q' * 1*NSubs, data[off + 44*NSubs : off + 52*NSubs])
        SubHalfMass[s_off : s_off + NSubs]            = struct.unpack('f' * 1*NSubs, data[off + 52*NSubs : off + 56*NSubs])

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
        header.attrs['Ngroups_ThisFile'] = TotNGroups
        header.attrs['Ngroups_Total'] = TotNGroups
        header.attrs['Nids_ThisFile'] = 0
        header.attrs['Nids_Total'] = 0
        header.attrs['Nsubgroups_ThisFile'] = TotNSubs
        header.attrs['Nsubgroups_Total'] = TotNSubs
        header.attrs['NumFiles'] = 1

        # groups
        groups = f.create_group('Group')
        groups['GroupFirstSub'] = FirstSubOfHalo
        groups['GroupLen'] = GroupLen
        groups['GroupMass'] = GroupLen * dm_particle_mass
        groups['GroupNsubs'] = NSubsPerHalo
        groups['Group_M_Crit200'] = Halo_M_Crit200
        groups['Group_R_Crit200'] = Halo_R_Crit200
        groups['Group_M_Mean200'] = Halo_M_Mean200
        groups['Group_R_Mean200'] = Halo_R_Mean200
        groups['Group_M_TopHat200'] = Halo_M_TopHat200
        groups['Group_R_TopHat200'] = Halo_R_TopHat200

        # subhalos
        subs = f.create_group('Subhalo')
        subs['SubhaloGrNr'] = SubParentHalo
        subs['SubhaloHalfmassRad'] = SubHalfMass
        subs['SubhaloIDMostbound'] = SubMostBoundID
        subs['SubhaloLen'] = SubLen
        subs['SubhaloMass'] = SubLen * dm_particle_mass
        subs['SubhaloPos'] = SubPos
        subs['SubhaloSpin'] = SubSpin
        subs['SubhaloVel'] = SubVel
        subs['SubhaloVelDisp'] = SubVelDisp
        subs['SubhaloVmax'] = SubVmax

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
    loadPath = '/virgo/simulations/Millennium/' 
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
    assert nChunks == nChunksIDs
    print('Found [%d] chunks for snapshot [%03d], loading...' % (nChunks,snap))

    if not path.isfile(saveFile):
        # first, load all IDs
        Nids_tot = 0
        offset = 0

        for i in range(nChunks):
            with open(_idChunkPath(snap,i),'rb') as f:
                header = f.read(16)
                Nids = struct.unpack('i', header[4:8])[0]
                Nids_tot += Nids

        ids_groupordered_old = np.zeros( Nids_tot, dtype='int64' )
        print('Reading a total of [%d] IDs now...' % Nids_tot)

        bitshift = ((1 << 34) - 1) # from get_group_coordinates()

        for i in range(nChunks):
            # full read
            with open(_idChunkPath(snap,i),'rb') as f:
                data = f.read()
            Nids = struct.unpack('i', data[4:8])[0]
            if Nids == 0:
                continue
            ids = struct.unpack('q' * Nids, data[16:16 + Nids*8])

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

        # reader header of first snapshot chunk
        with open(_snapChunkPath(snap,0),'rb') as f:
            header = f.read(260)

        npart      = struct.unpack('iiiiii', header[4:4+24])[1]
        mass       = struct.unpack('dddddd', header[28:28+48])
        scalefac   = struct.unpack('d', header[76:76+8])[0]
        redshift   = struct.unpack('d', header[84:84+8])[0]
        #nPartTot   = struct.unpack('iiiiii', header[100:100+24])[1]
        nFiles     = struct.unpack('i', header[128:128+4])[0]
        BoxSize    = struct.unpack('d', header[132:132+8])[0]
        Omega0     = struct.unpack('d', header[140:140+8])[0]
        OmegaL     = struct.unpack('d', header[148:148+8])[0]
        Hubble     = struct.unpack('d', header[156:156+8])[0]

        assert nFiles == nChunks

        # nPartTot is wrong, has no highword, so read and accumulate manually
        nPartTot = 0
        for i in range(nChunks):
            with open(_snapChunkPath(snap,i),'rb') as f:
                header = f.read(28)
                npart  = struct.unpack('iiiiii', header[4:4+24])[1]
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
            npart_local = struct.unpack('iiiiii', data[4:4+24])[1]

            # cast and save
            start_ids = 284 + 24*npart_local
            ids = struct.unpack('q' * npart_local*1, data[start_ids:start_ids + npart_local*8])

            ids_snapordered[offset : offset+npart_local] = ids

            print('[%3d] Snap IDs [%8d] particles, from [%10d] to [%10d].' % (i, npart_local, offset, offset+npart_local))
            offset += npart_local

        # crossmatch
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

        # open file for writing
        fOut = h5py.File(saveFile, 'w')

        header = fOut.create_group('Header')
        numPartTot = np.zeros( 6, dtype='int64' )
        numPartTot[1] = nPartTot
        header.attrs['BoxSize'] = BoxSize
        header.attrs['HubbleParam'] = Hubble
        header.attrs['MassTable'] = np.array(mass, dtype='float64')
        header.attrs['NumFilesPerSnapshot'] = 1
        header.attrs['NumPart_ThisFile'] = numPartTot
        header.attrs['NumPart_Total'] = numPartTot
        header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype='int64')
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
                npart_local = struct.unpack('iiiiii', data[4:4+24])[1]

                # cast
                start_pos = 268
                start_vel = 276 + 12*npart_local
                start = start_pos if ptName == 'Coordinates' else start_vel

                val_local = struct.unpack('f' * npart_local*3, data[start:start + npart_local*12])

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

def convertGadgetICsToHDF5():
    """ Convert a Gadget-1 binary format ICs (dm-only, 8 byte IDs, 4 byte pos/vel) into HDF5 format (keep original ordering). """
    loadPath = '/u/dnelson/sims.TNG/InitialConditions/L680n1024/output/ICs.%s' 
    savePath = '/u/dnelson/sims.TNG/L680n1024TNG_DM/output/snap_ics.hdf5'

    nChunks = len( glob.glob(loadPath % '*') )
    print('Found [%d] chunks, loading...' % nChunks)

    # reader header of first snapshot chunk
    with open(loadPath % 0,'rb') as f:
        header = f.read(260)

    npart      = struct.unpack('iiiiii', header[4:4+24])[1]
    mass       = struct.unpack('dddddd', header[28:28+48])
    scalefac   = struct.unpack('d', header[76:76+8])[0]
    redshift   = struct.unpack('d', header[84:84+8])[0]
    #nPartTot   = struct.unpack('iiiiii', header[100:100+24])[1]
    nFiles     = struct.unpack('i', header[128:128+4])[0]
    BoxSize    = struct.unpack('d', header[132:132+8])[0]
    Omega0     = struct.unpack('d', header[140:140+8])[0]
    OmegaL     = struct.unpack('d', header[148:148+8])[0]
    Hubble     = struct.unpack('d', header[156:156+8])[0]

    assert nFiles == nChunks

    # nPartTot is wrong, has no highword, so read and accumulate manually
    nPartTot = 0
    for i in range(nChunks):
        with open(loadPath % i,'rb') as f:
            header = f.read(28)
            npart  = struct.unpack('iiiiii', header[4:4+24])[1]
        nPartTot += npart
    print('Found new nPartTot [%d]' % nPartTot)

    # load all snapshot IDs
    offset = 0
    particle_pos = np.zeros( (nPartTot,3), dtype='float32' )
    particle_vel = np.zeros( (nPartTot,3), dtype='float32' )
    particle_ids = np.zeros( nPartTot, dtype='int64' )

    for i in range(nChunks):
        # full read
        with open(loadPath % i,'rb') as f:
            data = f.read()

        # local particle counts
        npart_local = struct.unpack('iiiiii', data[4:4+24])[1]

        # cast and save
        start_pos = 268 + 0*npart_local
        start_vel = 276 + 12*npart_local
        start_ids = 284 + 24*npart_local

        pos = struct.unpack('f' * npart_local*3, data[start_pos:start_pos + npart_local*12])
        vel = struct.unpack('f' * npart_local*3, data[start_vel:start_vel + npart_local*12])
        ids = struct.unpack('q' * npart_local*1, data[start_ids:start_ids + npart_local*8])

        particle_pos[offset : offset+npart_local,:] = np.reshape( pos, (npart_local,3) )
        particle_vel[offset : offset+npart_local,:] = np.reshape( vel, (npart_local,3) )
        particle_ids[offset : offset+npart_local] = ids

        print('[%3d] Snap chunk has [%8d] particles, from [%10d] to [%10d].' % (i, npart_local, offset, offset+npart_local))
        offset += npart_local

    # open file for writing
    with h5py.File(savePath, 'w') as fOut:
        # header
        header = fOut.create_group('Header')
        numPartTot = np.zeros( 6, dtype='int64' )
        numPartTot[1] = nPartTot
        header.attrs['BoxSize'] = BoxSize
        header.attrs['HubbleParam'] = Hubble
        header.attrs['MassTable'] = np.array(mass, dtype='float64')
        header.attrs['NumFilesPerSnapshot'] = 1
        header.attrs['NumPart_ThisFile'] = numPartTot
        header.attrs['NumPart_Total'] = numPartTot
        header.attrs['NumPart_Total_HighWord'] = np.zeros(6, dtype='int64')
        header.attrs['Omega0'] = Omega0
        header.attrs['OmegaLambda'] = OmegaL
        header.attrs['Redshift'] = redshift
        header.attrs['Time'] = scalefac

        pt1 = fOut.create_group('PartType1')

        # particle data
        pt1['Coordinates'] = particle_pos
        pt1['Velocities'] = particle_vel
        pt1['ParticleIDs'] = particle_ids

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
