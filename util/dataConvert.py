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
