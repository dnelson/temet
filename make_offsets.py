# file/DB utility functions
import numpy as np
import h5py
from os import path

def makeSubgroupOffsetsIntoSublinkTree(fileBase, snapNum):
    """For every subgroup, save its (SubhaloID, LastProgenitorID, rowNum) for SubLink."""

    # load first groupcat and get counts for allocation
    filePath = fileBase + 'output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(0) + '.hdf5'

    f = h5py.File(filePath,'r')
    header = f['Header']
        
    totSubGroups = header.attrs['Nsubgroups_Total']
        
    f.close()
    print("totSubGroups: "+str(totSubGroups))
    
    # allocate save arrays
    SubhaloID        = np.zeros( (totSubGroups), dtype=np.int64 )
    LastProgenitorID = np.zeros( (totSubGroups), dtype=np.int64 )
    RowNum           = np.zeros( (totSubGroups), dtype=np.int32 )
    
    # load minimal SubLink tree
    if path.isfile(fileBase + 'trees/SubLink/tree.hdf5'):
        f = h5py.File(fileBase + 'trees/SubLink/tree.hdf5','r')
    else:
        f = h5py.File(fileBase + 'output/sublink/tree.hdf5','r')

    Tree = f["Tree"][:]
    f.close()
    
    # locate entries for this snapshot
    snap_inds = ( np.where(Tree["SnapNum"] == snapNum) )[0]
    
    if not len(snap_inds) or np.max(snap_inds) > 2147483646:
        print 'ERROR'
        return
    
    print snap_inds.dtype, np.min(snap_inds), np.max(snap_inds)
    
    # get the subgroup ids that index the group catalogs at this snapshot
    loc_subgroup_ids = Tree["SubfindID"][snap_inds]
    
    # place their tree number and subindex
    LastProgenitorID[ loc_subgroup_ids ] = Tree["LastProgenitorID"][snap_inds]
    SubhaloID       [ loc_subgroup_ids ] = Tree["SubhaloID"][snap_inds]
    RowNum          [ loc_subgroup_ids ] = snap_inds
    
    # save offset tables
    f = h5py.File('sublink_offsets_subgroup_'+str(snapNum)+'.hdf5','w')
    dset = f.create_dataset("LastProgenitorID", data=LastProgenitorID)
    dset = f.create_dataset("SubhaloID", data=SubhaloID)
    dset = f.create_dataset("RowNum", data=RowNum)
    f.close()
    
    print("Saved: [sublink_offsets_subgroup_"+str(snapNum)+".hdf5]")
    print("COUNTS: [%d %d]" % (totSubGroups,len(loc_subgroup_ids)))

def makeSubgroupOffsetsIntoMergerTree(fileBase, snapNum, nChunks):
    """For every subgroup, save its tree number and subindex within that tree."""
    lastSnapNum = 135
    
    # load first groupcat and get counts for allocation
    filePath = fileBase + 'output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(0) + '.hdf5'

    f = h5py.File(filePath,'r')
    header = f['Header']
        
    totSubGroups = header.attrs['Nsubgroups_Total']
        
    treeFile  = np.zeros( (totSubGroups), dtype=np.int32 )
    treeNum   = np.zeros( (totSubGroups), dtype=np.int32 )
    treeIndex = np.zeros( (totSubGroups), dtype=np.int32 )
    treeMask  = np.zeros( (totSubGroups), dtype=np.int32 ) # debugging
    
    f.close()
    print("totSubGroups: "+str(totSubGroups))
    
    subgroupCount = 0
    
    for i in range(nChunks):
        filePath = fileBase + 'trees/treedata/trees_sf1_' + str(lastSnapNum).zfill(3)
        filePath += '.' + str(i) +'.hdf5'

        if not path.isfile(filePath):
            print("ERROR: File not found: " + filePath)
            return
    
        f = h5py.File(filePath,'r')
        header = f['Header']
                    
        if i % 100 == 0:
            print(filePath)
        
        # loop over each tree
        for j in range(header.attrs['NtreesPerFile']):
        
            # first select on snapshot number
            loc_snaps = f['Tree'+str(j)]['SnapNum'][:]
            snap_inds = ( np.where(loc_snaps == snapNum) )[0]
            
            if not len(snap_inds):
                continue
                
            if np.max(snap_inds) > 2147483646:
                print 'ERROR'
                return
                
            snap_inds = np.array(snap_inds, dtype='int32')
            
            # get the subgroup ids that index the group catalogs at this snapshot
            loc_subgroup_ids = f['Tree'+str(j)]['SubhaloNumber'][:]
            loc_subgroup_ids = loc_subgroup_ids[snap_inds]
            
            # place their tree number and subindex
            treeFile [ loc_subgroup_ids ] = i
            treeNum  [ loc_subgroup_ids ] = j
            treeIndex[ loc_subgroup_ids ] = snap_inds
            treeMask [ loc_subgroup_ids ] += 1
            
            subgroupCount += len(loc_subgroup_ids)
            
        f.close()
        
    # save offset tables (note: last offset of each represents EOF)
    #if totSubGroups == subgroupCount and np.min(treeMask) == 1 and np.max(treeMask) == 1:
    f = h5py.File('tree_offsets_subgroup_'+str(snapNum)+'_'+str(lastSnapNum)+'.hdf5','w')
    dset = f.create_dataset("TreeFile", data=treeFile)
    dset = f.create_dataset("TreeNum", data=treeNum)
    dset = f.create_dataset("TreeIndex", data=treeIndex)
    f.close()
    print("Saved: [tree_offsets_subgroup_"+str(snapNum)+".hdf5]")
    #else:
    #    print("Error, save canceled. [%d %d] [%d %d]" % (totSubGroups,subgroupCount,np.min(treeMask),np.max(treeMask)))

        
def makeGroupOffsetsIntoSnap(fileBase, snapNum, nChunks):
    """Make the offset table (by type) for every group/subgroup, such that the global location of 
       the members of any group/subgroup can be quickly located."""
    nTypes = 6

    # load first groupcat and get counts for allocation
    filePath = fileBase + 'output/groups_' + str(snapNum).zfill(3) + '/'
    filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(0) + '.hdf5'

    f = h5py.File(filePath,'r')
    header = f['Header']
        
    totGroups    = header.attrs['Ngroups_Total']
    totSubGroups = header.attrs['Nsubgroups_Total']
        
    offsetsGroup    = np.zeros( (totGroups+1,nTypes), dtype=np.int64 )
    offsetsSubgroup = np.zeros( (totSubGroups+1,nTypes), dtype=np.int64 )
    print("totGroups: "+str(totGroups)+" totSubGroups: "+str(totSubGroups))
    
    groupCount    = 0
    subgroupCount = 0
    
    # load following 3 fields across all chunks
    groupLenType    = np.zeros( (totGroups,nTypes), dtype=np.int32 )
    groupNsubs      = np.zeros( (totGroups,), dtype=np.int32 )
    subgroupLenType = np.zeros( (totSubGroups,nTypes), dtype=np.int32 )
    
    for i in range(1,nChunks+1):
        filePath = fileBase + 'output/groups_' + str(snapNum).zfill(3) + '/'
        filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(i-1) + '.hdf5'
        
        if not path.isfile(filePath):
            print("ERROR: File not found: " + filePath)
            return    
    
        # load header, get number of groups/subgroups in this file, and lengths
        f = h5py.File(filePath,'r')
        header = f['Header']
        
        nGroups    = header.attrs['Ngroups_ThisFile']
        nSubgroups = header.attrs['Nsubgroups_ThisFile']
        #print(filePath+"  ["+str(nGroups)+" "+str(nSubgroups)+"]")
        
        if header.attrs['Ngroups_ThisFile'] > 0:
            groupLenType[groupCount:groupCount+nGroups] = f['Group']['GroupLenType']
            groupNsubs[groupCount:groupCount+nGroups]   = f['Group']['GroupNsubs']
        if header.attrs['Nsubgroups_ThisFile'] > 0:
            subgroupLenType[subgroupCount:subgroupCount+nSubgroups] = f['Subhalo']['SubhaloLenType']
        
        groupCount += nGroups
        subgroupCount += nSubgroups
        
        f.close()
        
    # loop over each particle type, then over groups, calculate offsets from length
    for j in range(nTypes):
        #print "Processing type ["+str(j)+"]."
        subgroupCount = 0
        
        # compute group offsets first
        offsetsGroup[1:,j] = np.cumsum( groupLenType[:,j] )
        
        for k in range(totGroups):
            # subhalo offsets depend on group (to allow fuzz)
            if groupNsubs[k] > 0:
                offsetsSubgroup[subgroupCount,j] = offsetsGroup[k,j]
                
                subgroupCount += 1
                for m in range(1, groupNsubs[k]):
                    offsetsSubgroup[subgroupCount,j] = offsetsSubgroup[subgroupCount-1,j] + subgroupLenType[subgroupCount-1,j]
                    subgroupCount += 1
    
    # save offset tables (note: last offset of each represents EOF)
    if totGroups == groupCount and totSubGroups == subgroupCount:
        f = h5py.File('snap_offsets_group_'+str(snapNum)+'.hdf5','w')
        dset = f.create_dataset("Offsets", data=offsetsGroup)
        f.close()
        print("Saved: [snap_offsets_group_"+str(snapNum)+".hdf5]")
        
        f = h5py.File('snap_offsets_subhalo_'+str(snapNum)+'.hdf5','w')
        dset = f.create_dataset("Offsets", data=offsetsSubgroup)
        f.close()
        print("Saved: [snap_offsets_subhalo_"+str(snapNum)+".hdf5]")
    else:
        print("Error, save canceled. [%d %d] [%d %d]" % (totGroups,groupCount,totSubGroups,subgroupCount))
    
    #import pdb; pdb.set_trace()
    
def makeSnapOffsetList(fileBase, snapNum, nChunks):
    """Make the offset table (by type) for the snapshot files, to be able to quickly determine within 
       which file(s) a given offset+length will exist."""
    nTypes = 6
    
    offsets = np.zeros( (nTypes,nChunks), dtype=np.int64 )
    NumPartTot = np.zeros( nTypes, dtype=np.int64 )
    
    for i in range(1,nChunks+1):
        filePath = fileBase + 'output/snapdir_' + str(snapNum).zfill(3)
        filePath += '/snap_' + str(snapNum).zfill(3) + '.' + str(i-1) + '.hdf5'

        if not path.isfile(filePath):
            print("ERROR: File not found: " + filePath)
            return
        
        # load header, get NumPart_ThisFile
        f = h5py.File(filePath,'r')
        header = f['Header']
        
        # load total number of particles across all files
        if i == 1:
            for j in range(nTypes):
                NumPartTot[j] = header.attrs['NumPart_Total'][j] | (header.attrs['NumPart_Total_HighWord'][j] << 32)
        
        # set next offsets
        if i < nChunks:
            for j in range(nTypes):
                offsets[j,i] = offsets[j,i-1] + header.attrs['NumPart_ThisFile'][j]
        
    # save offset table
    totCounted = offsets[:,-1] + header.attrs['NumPart_ThisFile']
    if (totCounted == NumPartTot).sum() == nTypes:
        np.save('offsets_snap_'+str(snapNum), offsets)
        print("Saved: [offsets_snap_"+str(snapNum)+"]")
    else:
        print("Error, save canceled.")
        
def makeGroupCatOffsetList(fileBase, snapNum, nChunks):
    """Make the offset table for the group catalog files, to be able to quickly determine which
       which file a given group/subgroup number exists."""
    offsetsGroup    = np.zeros( nChunks, dtype=np.int32 )
    offsetsSubgroup = np.zeros( nChunks, dtype=np.int32 )
    totSubGroups = 0
    totGroups = 0
    
    for i in range(1,nChunks+1):
        filePath = fileBase + 'output/groups_' + str(snapNum).zfill(3) + '/'
        filePath += 'fof_subhalo_tab_' + str(snapNum).zfill(3) + '.' + str(i-1) + '.hdf5'

        if not path.isfile(filePath):
            print("ERROR: File not found: " + filePath)
            return

        # load header, get NumPart_ThisFile
        f = h5py.File(filePath,'r')
        header = f['Header']
        
        # load total number of subgroups across all files
        if i == 1:
            totGroups = header.attrs['Ngroups_Total']
            totSubGroups = header.attrs['Nsubgroups_Total']
        
        # set next offsets
        if i < nChunks:
            offsetsGroup[i] = offsetsGroup[i-1] + header.attrs['Ngroups_ThisFile']
            offsetsSubgroup[i] = offsetsSubgroup[i-1] + header.attrs['Nsubgroups_ThisFile']
    
    # save offset tables
    totGroupCounted = offsetsGroup[-1] + header.attrs['Ngroups_ThisFile']
    totSubgroupCounted = offsetsSubgroup[-1] + header.attrs['Nsubgroups_ThisFile']
    
    if totGroupCounted == totGroups and totSubgroupCounted == totSubGroups:
        np.save('offsets_group_'+str(snapNum), offsetsGroup)
        np.save('offsets_subhalo_'+str(snapNum), offsetsSubgroup)
        
        print("Saved: [offsets_group_"+str(snapNum)+"]")
        print("Saved: [offsets_subhalo_"+str(snapNum)+"]")
    else:
        print("Error, save canceled. [%d %d] [%d %d]" % (totGroups,totGroupCounted,totSubGroups,totSubgroupCounted))

def makeOffsets():
    # Illustris-1
    #fileBase = '/n/ghernquist/Illustris/Runs/L75n1820FP/'
    #snapNum = 68
    #nChunksSnap = 512
    #nChunksTree = 4096
    
    # Illustris-2
    #fileBase = '/n/ghernquist/Illustris/Runs/L75n910FP/'
    #snapNum = 135
    #nChunksSnap = 256
    #nChunksTree = 1024
   
    # Illustris-3
    #fileBase = '/n/ghernquist/Illustris/Runs/L75n455FP/'
    #snapNum = 135
    #nChunksSnap = 32
    #nChunksTree = 256

    # Illustris-1-DM
    #fileBase = '/n/hernquistfs1/Illustris/Runs/L75n1820DM/'
    #snapNum = 135
    #nChunksSnap = 128

    # Illustris-2-DM
    #fileBase = '/n/hernquistfs1/Illustris/Runs/L75n910DM/'
    #snapNum = 135
    #nChunksSnap = 32

    # Illustris-3-DM
    #fileBase = '/n/hernquistfs1/Illustris/Runs/L75n455DM/'
    #snapNum = 135
    #nChunksSnap = 8
 
    # sims.zooms.128_20Mpc
    fileBase = '/n/home07/dnelson/sims.zooms/128_20Mpc_h9_L10/'
    snapNum = 58
    #nChunksSnap = 2 #L9=2, L10=2, L11=4

    #for i in range(0,136): # Illustris=136
    #	makeGroupCatOffsetList(fileBase, i, nChunksSnap)
    #	makeSnapOffsetList(fileBase, i, nChunksSnap)
    #	makeGroupOffsetsIntoSnap(fileBase, i, nChunksSnap)

    #makeSubgroupOffsetsIntoMergerTree(fileBase, snapNum, nChunksTree)
    makeSubgroupOffsetsIntoSublinkTree(fileBase, snapNum)

