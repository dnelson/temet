import numpy as np
import h5py
import pdb

def loadGroupCatSingle(fileBase,snapNum,searchID,getGroup):
    """ Return complete subhalo (subfind) information on one group. 
        Search for subhalo/subgroup if getGroup==0, else search for halo/fof group. """

    dsetName = "Subhalo"
    if getGroup > 0:
        dsetName = "Group" 
 
    # load groupcat offsets, calculate target file and offset
    filePath = fileBase + '/postprocessing/offsets/offsets_'+dsetName.lower()+'_'+str(snapNum)+'.npy'
    offsets = np.load(filePath)
 
    offsets = searchID - offsets
    fileNum = np.max( np.where(offsets >= 0) )
    groupOffset = offsets[fileNum]
 
    # load subhalo fields into a dict
    filePath = fileBase + '/output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(fileNum) + '.hdf5'
 
    result = {}
 
    f = h5py.File(filePath,'r')
 
    for haloProp in f[dsetName].keys():
        result[haloProp] = f[dsetName][haloProp][groupOffset]
 
    f.close()
    return result
 
def loadSnapSubset(fileBase,snapNum,searchID,getGroup,partType,fields):
    """ Return requested fields for one particle type for all members of group/subgroup. """
 
    groupName = "PartType" + str(partType)
    dsetName = "Subhalo"
    if getGroup > 0:
        dsetName = "Group"
 
    # load the length (by type) of this group/subgroup from the group catalog
    filePath = fileBase + '/postprocessing/offsets/offsets_' + dsetName.lower() + '_'+str(snapNum)+'.npy'
    offsets = np.load(filePath)
 
    offsets = searchID - offsets
    fileNum = np.max( np.where(offsets >= 0) )
    groupOffset = offsets[fileNum]
 
    filePath = fileBase + '/output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(fileNum) + '.hdf5'
 
    f = h5py.File(filePath,'r')
    lenType = f[dsetName][dsetName+"LenType"][groupOffset]
    f.close()
 
    # load the offset (by type) of this group/subgroup within the snapshot chunks
    filePath = fileBase + '/postprocessing/offsets/snap_offsets_' + dsetName.lower() + '_'+str(snapNum)+'.hdf5'
 
    f = h5py.File(filePath,'r')
    offsetType = f["Offsets"][ searchID ]
    f.close()
 
    # load the offsets for the snapshot chunks
    filePath = fileBase + '/postprocessing/offsets/offsets_snap_'+str(snapNum)+'.npy'
    offsets = np.load(filePath)
 
    # determine first snapshot chunk we need to load for this type
    wOffset = 0
    result  = {}
 
    offsetsThisType = offsetType[partType] - offsets[partType,:]
    fileNum         = np.max( np.where(offsetsThisType >= 0) )
    fileOffset      = offsetsThisType[fileNum]
 
    numLeftToRead = lenType[partType]
 
    while numLeftToRead:
 
        # loop over files, for each load the overlapping chunk into the hdf5 file
        curSnapFilePath = fileBase + '/output/snapdir_' + str(snapNum).zfill(3) + '/'
        curSnapFilePath += 'snap_' + str(snapNum).zfill(3) + '.' + str(fileNum) + '.hdf5'
        fSnap = h5py.File(curSnapFilePath,'r')

        # set local read length for this file
        readLen = numLeftToRead
 
        if fileNum < offsets.shape[1]-1: # if in last file, assume group is entirely contained in this file
            # if the local length after requested offset is less than the read length, modify read length
            if fileOffset+readLen+offsets[partType,fileNum]-1 >= offsets[partType,fileNum+1]:
                readLen = (offsets[partType,fileNum+1]-offsets[partType,fileNum]) - fileOffset
 
        # loop over each requested field for this particle type
        for fieldName in fields:
            # shape and type
            dtype = fSnap[groupName][fieldName].dtype
            shape = fSnap[groupName][fieldName].shape
 
            # read data local to the current file (allocate dataset if it does not already exist)
            if len(shape) == 1:
                if fieldName not in result:
                    result[fieldName] = np.zeros( (lenType[partType],), dtype=dtype )
                result[fieldName][wOffset:wOffset+readLen] = fSnap[groupName][fieldName][fileOffset:fileOffset+readLen]
            else:
                if fieldName not in result:
                    result[fieldName] = np.zeros( (lenType[partType],shape[1]), dtype=dtype )
                result[fieldName][wOffset:wOffset+readLen,:] = fSnap[groupName][fieldName][fileOffset:fileOffset+readLen,:]
 
        wOffset += readLen
        numLeftToRead -= readLen
        fileNum += 1
        fileOffset = 0
 
        fSnap.close()
 
    # reads across all files done, for all fields of this type
    return result
    
# config
def testRun():
    fields    = ["Coordinates","Masses"]
    partType  = 0
    searchID  = 6557 # group or subgroup number
    getGroup  = 0 # 0=get subhalo, 1=get group
    snapNum   = 121
    fileBase  = '/n/ghernquist/Illustris/Runs/Illustris-3/'
    
    x = loadSnapSubset(fileBase,snapNum,searchID,getGroup,partType,fields)
    return x

def testMassProblem():
    fields    = ["Masses"]
    partType  = 0
    searchID  = 6557 # group or subgroup number
    getGroup  = 0 # 0=get subhalo, 1=get group
    snapNum   = 135 #121
    fileBase  = '/n/ghernquist/Illustris/Runs/Illustris-3/'

    # load group catalog and particle masses
    for searchID in range(0,100):
    	gc = loadGroupCatSingle(fileBase,snapNum,searchID,getGroup)
    	x = loadSnapSubset(fileBase,snapNum,searchID,getGroup,partType,fields)

        snapMass = 0.0
        windMass = 0.0
        if 'Masses' in x:
            snapMass = sum(x['Masses'])

        # also wind
        x = loadSnapSubset(fileBase,snapNum,searchID,getGroup,4,['GFM_StellarFormationTime','Masses'])

        if 'Masses' in x:
            inds = np.where( x['GFM_StellarFormationTime'] < 0.0 )
            if len(inds) > 0:
                windMass = sum(x['Masses'][inds])

        massDiff = gc['SubhaloMassType'][0]-snapMass

        if massDiff > 0.0:
            print searchID,massDiff,massDiff-windMass
    #return x

def testGrifIDs():
    fields    = ["ParticleIDs"]
    partType  = 4
    searchID  = 0 # group or subgroup number
    getGroup  = 0 # 0=get subhalo, 1=get group
    snapNum   = 135
    fileBase  = '/n/ghernquist/Illustris/Runs/Illustris-1/'
    
    x = loadSnapSubset(fileBase,snapNum,searchID,getGroup,partType,fields)
    return x
