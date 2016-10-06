"""
cosmo/mergertree.py
  Cosmological simulations - working with merger trees (SubLink, LHaloTree, C-Trees).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import illustris_python as il
from scipy.signal import savgol_filter
from cosmo.util import snapNumToRedshift, correctPeriodicPosBoxWrap
from util.helper import running_sigmawindow, iterable

treeName_default = "SubLink_gal"

def loadMPB(sP, id, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load fields of main-progenitor-branch (MPB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        return il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        return il.lhalotree.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True)

    raise Exception('Unrecognized treeName.')

def loadMPBs(sP, ids, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load multiple MPBs at once (e.g. all of them), optimized for speed. 
    Basically a rewrite of illustris_python/sublink.py under specific conditions (hopefully temporary). 
      Return: a dictionary whose keys are subhalo IDs, and the contents of each dict value is another 
      dictionary of identical stucture to the return of loadMPB().
    """
    
    # (Step 0) prep
    basePath = sP.simPath
    snapNum  = sP.snap

    from os.path import isfile
    from glob import glob
    assert isfile(il.groupcat.offsetPath(basePath,snapNum)) # otherwise need to generalize offset loading
    assert treeName in ['SubLink','SubLink_gal'] # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]

    # create quick offset table for rows in the SubLink files
    # if you are loading thousands or millions of sub-trees, you may wish to cache this offsets array
    numTreeFiles = len(glob(il.sublink.treePath(basePath,treeName,'*')))
    offsets = np.zeros( numTreeFiles, dtype='int64' )

    for i in range(numTreeFiles-1):
        with h5py.File(il.sublink.treePath(basePath,treeName,i),'r') as f:
            offsets[i+1] = offsets[i] + f['SubhaloID'].shape[0]

    result = {}

    # (Step 1) treeOffsets()
    offsetFile = il.groupcat.offsetPath(basePath,snapNum)
    prefix = 'Subhalo/' + treeName + '/'

    with h5py.File(offsetFile,'r') as f:
        groupFileOffsets = f['FileOffsets/Subhalo'][()]

        # load all merger tree offsets
        RowNums     = f[prefix+'RowNum'][()]
        LastProgIDs = f[prefix+'LastProgenitorID'][()]
        SubhaloIDs  = f[prefix+'SubhaloID'][()]

    # now subhalos one at a time (one tree file opening and len(fields)+1 reads for each)
    for i, id in enumerate(ids):
        if i % int(ids.size/10) == 0 and i <= ids.size:
            print(' %4.1f%%' % (float(i+1)*100.0/ids.size))

        if id == -1: continue # skip requests for e.g. fof halos which had no central subhalo

        # calculate target groups file chunk which contains this id
        groupFileOffsetsLoc = int(id) - groupFileOffsets
        fileNum = np.max( np.where(groupFileOffsetsLoc >= 0) )
        groupOffset = groupFileOffsetsLoc[fileNum]

        # (Step 2) loadTree()
        RowNum = RowNums[groupOffset]
        LastProgID = LastProgIDs[groupOffset]
        SubhaloID  = SubhaloIDs[groupOffset]

        if RowNum == -1:
            print('   warning, subhalo [%d] at snapNum [%d] not in tree.' % (id,snapNum))
            continue

        rowStart = RowNum
        rowEnd   = RowNum + (LastProgID - SubhaloID)
        nRows    = rowEnd - rowStart + 1
    
        # find the tree file chunk containing this row
        rowOffsets = rowStart - offsets

        fileNum = np.max(np.where( rowOffsets >= 0 ))
        fileOff = rowOffsets[fileNum]
    
        # load only main progenitor branch: get MainLeafProgenitorID now
        with h5py.File(il.sublink.treePath(basePath,treeName,fileNum),'r') as f:
            MainLeafProgenitorID = f['MainLeafProgenitorID'][fileOff]
            
            # re-calculate nRows
            rowEnd = RowNum + (MainLeafProgenitorID - SubhaloID)
            nRows  = rowEnd - rowStart + 1
        
            # read
            result[id] = {'count':nRows}
         
            # loop over each requested field and read, no error checking
            for field in fields:
                result[id][field] = f[field][fileOff:fileOff+nRows]
           
    return result

def loadTreeFieldnames(sP, treeName=treeName_default):
    """ Load names of fields available in a mergertree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        with h5py.File(il.sublink.treePath(sP.simPath, treeName), 'r') as f:
            return f.keys()
    if treeName in ['LHaloTree']:
        with h5py.File(il.lhalotree.treePath(sP.simPath, chunkNum=0), 'r') as f:
            return f['Tree0'].keys()

    raise Exception('Unrecognized treeName.')

def insertMPBGhost(mpb, snapToInsert=None):
    """ Insert a ghost entry into a MPB dict by interpolating over neighboring snapshot information.
    Appropriate if e.g. a group catalog if corrupt but snapshot files are ok. Could also be used to 
    selectively wipe out outlier points in the MPB. """
    assert snapToInsert is not None

    indAfter = np.where(mpb['SnapNum'] == snapToInsert - 1)[0]
    assert len(indAfter) > 0

    mpb['SubfindID'] = np.insert( mpb['SubfindID'], indAfter, -1 ) # ghost

    for key in mpb:
        if key in ['count','SubfindID']:
            continue

        if mpb[key].ndim == 1: # [N]
            interpVal = np.mean([mpb[key][indAfter],mpb[key][indAfter-2]])
            mpb[key] = np.insert( mpb[key], indAfter, interpVal )

        if mpb[key].ndim == 2: # [N,3]
            interpVal = np.mean( np.vstack((mpb[key][indAfter,:],mpb[key][indAfter-2,:])), axis=0)
            mpb[key] = np.insert( mpb[key], indAfter, interpVal, axis=0)

    return mpb

def mpbSmoothedProperties(sP, id, fillSkippedEntries=True, extraFields=[]):
    """ Load a particular subset of MPB properties of subhalo id, and smooth them in time. These are 
    currently: position, mass (m200_crit), virial radius (r200_crit), virial temperature (derived), 
    velocity (subhalo), and are inside ['sm'] for smoothed versions. Also attach time with snap/redshift.
    Note: With the Sublink* trees, the group properties are always present for all subhalos and are 
    identical for all subhalos in the same group. """

    fields = ['SubfindID','SnapNum','SubhaloPos','SubhaloVel','Group_R_Crit200','Group_M_Crit200']

    # any extra fields to be loaded?
    treeFileFields = loadTreeFieldnames(sP)

    for field in iterable(extraFields):
        if field not in fields and field in treeFileFields:
            fields.append(field)

    # load
    mpb = loadMPB(sP, id, fields=fields)

    # fill any missing snapshots with ghost entries? (e.g. actual trees can skip a snapshot when 
    # locating a descendant but we may need a continuous position for all snapshots)
    if fillSkippedEntries:
        for snap in np.arange( mpb['SnapNum'].min(), mpb['SnapNum'].max() ):
            if snap in mpb['SnapNum']:
                continue
            mpb = insertMPBGhost(mpb, snapToInsert=snap)
            print(' mpb inserted [%d] ghost' % snap)

    # sims.zooms2/h2_L9: corrupt groups_104 override (insert interpolated snap 104 values for MPB)
    if sP.run == 'zooms2' and sP.res == 9 and sP.hInd == 2:
        print('WARNING: sims.zooms2/h2_L9: mpb corrupt 104 ghost inserted')
        mpb = insertMPBGhost(mpb, snapToInsert=104)

    # determine sK parameters
    sKn = int(len(mpb['SnapNum'])/10) # savgol smoothing kernel length (1=disabled)
    if sKn % 2 == 0:
        sKn = sKn + 1 # make odd
    sKo = 3 # savgol smoothing kernel poly order

    # attach redshift, virial temp, savgol parameters
    mpb['Redshift']    = snapNumToRedshift( sP, mpb['SnapNum'] )
    mpb['Group_T_vir'] = sP.units.codeMassToVirTemp(mpb['Group_M_Crit200'], log=True)
    mpb['Group_S_vir'] = sP.units.codeMassToVirEnt(mpb['Group_M_Crit200'], log=True)
    mpb['Group_V_vir'] = sP.units.codeMassToVirVel(mpb['Group_M_Crit200'])

    mpb['sm'] = {}
    mpb['sm']['sKn'] = sKn
    mpb['sm']['sKo'] = sKo

    # smoothing: velocity
    mpb['sm']['vel'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['vel_moved'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['v_sm'] = mpb['SubhaloVel'].astype('float32')
    #mpb['sm']['v_sigma'] = mpb['SubhaloVel'].astype('float32')

    for i in range(3):
        # outlier rejection: running median estimator
        medWindowSize = int( len(mpb['SnapNum'])/sKn*2 )
        sigmaThresh = 2.0
        iterations = 0 # disabled

        for j in range(iterations):
            print('DEBUG TESTING')
            mpb['sm']['v_sigma'][:,i] = running_sigmawindow( mpb['Redshift'], mpb['sm']['vel'][:,i], medWindowSize)
            mpb['sm']['v_sm'][:,i] = savgol_filter( mpb['sm']['vel'][:,i], sKn*5, sKo )

            w = np.where( np.abs(mpb['sm']['vel'][:,i]-mpb['sm']['v_sm'][:,i])/mpb['sm']['v_sigma'][:,i] > sigmaThresh )

            print(i,mpb['sm']['vel'][w,i],mpb['sm']['v_sm'][w,i])
            mpb['sm']['vel'][w,i] = mpb['sm']['v_sm'][w,i]
            mpb['sm']['vel_moved'][w,i] = mpb['sm']['v_sm'][w,i]
            # END DEBUG

        # smooth
        mpb['sm']['vel'][:,i] = savgol_filter( mpb['sm']['vel'][:,i], sKn, sKo )

    # smoothing: positions with box-edge shifting
    posShiftInds = correctPeriodicPosBoxWrap(mpb['SubhaloPos'], sP)
    mpb['sm']['pos'] = mpb['SubhaloPos'].astype('float32')

    for i in range(3):
        mpb['sm']['pos'][:,i] = savgol_filter( mpb['sm']['pos'][:,i], sKn, sKo )
        
        # shifted? then unshift now
        if i in posShiftInds:
            unShift = np.zeros( len(mpb['Redshift']), dtype='float32' )
            unShift[posShiftInds[i]] = sP.boxSize

            mpb['SubhaloPos'][:,i] = mpb['SubhaloPos'][:,i] + unShift
            mpb['sm']['pos'][:,i] = mpb['sm']['pos'][:,i] + unShift

    # smoothing: all others
    mpb['sm']['mass'] = savgol_filter( sP.units.codeMassToLogMsun( mpb['Group_M_Crit200'] ), sKn, sKo )
    mpb['sm']['rvir'] = savgol_filter( mpb['Group_R_Crit200'], sKn, sKo )
    mpb['sm']['tvir'] = savgol_filter( mpb['Group_T_vir'], sKn, sKo )
    mpb['sm']['svir'] = savgol_filter( mpb['Group_S_vir'], sKn, sKo )
    mpb['sm']['vvir'] = savgol_filter( mpb['Group_V_vir'], sKn, sKo )

    return mpb    

def debugPlot():
    """ Testing MPB loading and smoothing. """
    from util import simParams
    import matplotlib.pyplot as plt
    from cosmo.util import addRedshiftAgeAxes

    # config
    #sP = simParams(res=455, run='tng', redshift=0.0) 
    sP = simParams(res=11, run='zooms', redshift=2.0, hInd=0)

    # load
    mpb = mpbSmoothedProperties(sP, sP.zoomSubhaloID)

    label = 'savgol n='+str(mpb['sm']['sKn']) + ' o=' + str(mpb['sm']['sKo'])

    if 1:
        # PLOT 1: position
        fig, axs = plt.subplots(1, 3, figsize=(20,10))

        for i in range(3):
            addRedshiftAgeAxes(axs[i], sP, xlog=True)
            axs[i].set_xlim([1.9,9.0])
            axs[i].set_ylabel(['x','y','z'][i] + ' [ckpc/h]')

            # plot original and smoothed x,y,z
            axs[i].plot( mpb['Redshift'], mpb['SubhaloPos'][:,i], 'o', label='raw' )      
            axs[i].plot( mpb['Redshift'], mpb['sm']['pos'][:,i], linestyle='-', label=label )

            # fitting
            #for deg in [3,7]:
            #    pcoeffs = np.polyfit( mpb['Redshift'], mpb['SubhaloPos'][:,i], deg)
            #    poly = np.poly1d(pcoeffs)
            #    xx = np.linspace(2.0, 10.0, 400)
            #    yy = poly(xx)
            #    axs[i].plot( xx, yy, linestyle='-', label='poly '+str(deg))

        axs[0].legend(loc='upper right')
        fig.tight_layout()    
        fig.savefig('tree_debug_pos_'+str(sP.res)+'.pdf')
        plt.close(fig)

    if 1:
        # PLOT 2: vel
        fig, axs = plt.subplots(1, 3, figsize=(20,10))

        for i in range(3):
            addRedshiftAgeAxes(axs[i], sP, xlog=True)
            axs[i].set_xlim([1.9,9.0])
            axs[i].set_ylabel(['vel_x','vel_y','vel_z'][i] + ' [km/s]')

            # plot original and smoothed v_x,v_y,v_z
            axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i], 'o', label='raw orig' ) 
            #axs[i].plot( mpb['Redshift'], mpb['sm']['vel_moved'][:,i], 'o', color='red', label='raw moved' )      
            axs[i].plot( mpb['Redshift'], mpb['sm']['vel'][:,i], linestyle='-', label=label )

            # sigma bounds
            #axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i]+2*mpb['sm']['v_sigma'][:,i], 
            #    linestyle=':', label='p2 sigma' )
            #axs[i].plot( mpb['Redshift'], mpb['SubhaloVel'][:,i]-2*mpb['sm']['v_sigma'][:,i], 
            #    linestyle=':', label='n2 sigma' )

        axs[0].legend(loc='upper right')
        fig.tight_layout()    
        fig.savefig('tree_debug_vel_'+str(sP.res)+'.pdf')
        plt.close(fig)

    if 1:
        # PLOT 3: other properties
        fig, axs = plt.subplots(2, 2, figsize=(20,10))

        # vir rad
        addRedshiftAgeAxes(axs[0,0], sP, xlog=True)
        axs[0,0].set_xlim([1.9,9.0])
        axs[0,0].set_ylabel('Group_R_Crit200 [ckpc/h]')

        axs[0,0].plot( mpb['Redshift'], mpb['Group_R_Crit200'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_R_Crit200'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[0,0].plot( mpb['Redshift'], yy2, linestyle='-', label=label )
        axs[0,0].legend(loc='upper right')

        # vir temp
        addRedshiftAgeAxes(axs[1,0], sP, xlog=True)
        axs[1,0].set_xlim([1.9,9.0])
        axs[1,0].set_ylabel('Vir Temp [ log K ]')

        axs[1,0].plot( mpb['Redshift'], mpb['Group_T_vir'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_T_vir'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[1,0].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # mass
        addRedshiftAgeAxes(axs[0,1], sP, xlog=True)
        axs[0,1].set_xlim([1.9,9.0])
        axs[0,1].set_ylabel('Mass [ 10$^{10}$ Msun/h ]')

        axs[0,1].plot( mpb['Redshift'], mpb['Group_M_Crit200'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_M_Crit200'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[0,1].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # vir entropy
        addRedshiftAgeAxes(axs[1,1], sP, xlog=True)
        axs[1,1].set_xlim([1.9,9.0])
        axs[1,1].set_ylabel('Vir Entropy [ log cgs ]')

        axs[1,1].plot( mpb['Redshift'], mpb['Group_S_vir'], 'o', label='raw' )

        yy2 = savgol_filter(mpb['Group_S_vir'],mpb['sm']['sKn'],mpb['sm']['sKo'])
        axs[1,1].plot( mpb['Redshift'], yy2, linestyle='-', label=label )

        # finish
        fig.tight_layout()    
        fig.savefig('tree_debug_props_'+str(sP.res)+'.pdf')
        plt.close(fig)    
