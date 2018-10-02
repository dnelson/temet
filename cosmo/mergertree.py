"""
cosmo/mergertree.py
  Cosmological simulations - working with merger trees (SubLink, LHaloTree, C-Trees).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import illustris_python as il
from os import path

import cosmo
from scipy.signal import savgol_filter
from scipy import interpolate
from cosmo.util import snapNumToRedshift, validSnapList, correctPeriodicPosBoxWrap
from util.helper import running_sigmawindow, iterable

treeName_default = "SubLink"

def loadMPB(sP, id, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load fields of main-progenitor-branch (MPB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        return il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        return il.lhalotree.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True)

    raise Exception('Unrecognized treeName.')

def loadMDB(sP, id, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load fields of main-descendant-branch (MDB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        return il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMDB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        return il.lhalotree.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMDB=True)

    raise Exception('Unrecognized treeName.')

def loadMPBs(sP, ids, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load multiple MPBs at once (e.g. all of them), optimized for speed, with a full tree load (high mem).
    Basically a rewrite of illustris_python/sublink.py under specific conditions (hopefully temporary). 
      Return: a dictionary whose keys are subhalo IDs, and the contents of each dict value is another 
      dictionary of identical stucture to the return of loadMPB().
    """
    from glob import glob
    assert treeName in ['SubLink','SubLink_gal'] # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, basestring):
        fields = [fields]
        
    fieldsLoad = fields + ['MainLeafProgenitorID']    

    # find full tree data sizes and attributes
    numTreeFiles = len(glob(il.sublink.treePath(sP.simPath,treeName,'*')))

    lengths = {}
    dtypes = {}
    seconddims = {}

    for field in fieldsLoad:
        lengths[field] = 0
        seconddims[field] = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath,treeName,i),'r') as f:
            for field in fieldsLoad:
                dtypes[field] = f[field].dtype
                lengths[field] += f[field].shape[0]
                if len(f[field].shape) > 1:
                    seconddims[field] = f[field].shape[1]

    # allocate for a full load
    fulltree = {}

    for field in fieldsLoad:
        if seconddims[field] == 0: 
            fulltree[field] = np.zeros( lengths[field], dtype=dtypes[field] )
        else:
            fulltree[field] = np.zeros( (lengths[field],seconddims[field]), dtype=dtypes[field] )

    # load full tree
    offset = 0

    for i in range(numTreeFiles):
        with h5py.File(il.sublink.treePath(sP.simPath,treeName,i),'r') as f:
            for field in fieldsLoad:
                if seconddims[field] == 0:
                    fulltree[field][offset : offset + f[field].shape[0]] = f[field][()]
                else:
                    fulltree[field][offset : offset + f[field].shape[0],:] = f[field][()]
            offset += f[field].shape[0]

    result = {}

    # (Step 1) treeOffsets()
    offsetFile = il.groupcat.offsetPath(sP.simPath,sP.snap)
    prefix = 'Subhalo/' + treeName + '/'

    with h5py.File(offsetFile,'r') as f:
        # load all merger tree offsets
        RowNums     = f[prefix+'RowNum'][()]
        SubhaloIDs  = f[prefix+'SubhaloID'][()]

    # now subhalos one at a time (memory operations only)
    for i, id in enumerate(ids):
        if id == -1:
            continue # skip requests for e.g. fof halos which had no central subhalo

        # (Step 2) loadTree()
        RowNum = RowNums[id]
        SubhaloID  = SubhaloIDs[id]
        MainLeafProgenitorID = fulltree['MainLeafProgenitorID'][RowNum]

        if RowNum == -1:
            continue

        # load only main progenitor branch
        rowStart = RowNum
        rowEnd   = RowNum + (MainLeafProgenitorID - SubhaloID)
        nRows    = rowEnd - rowStart + 1
    
        # init dict
        result[id] = {'count':nRows}
     
        # loop over each requested field and copy, no error checking
        for field in fields:
            result[id][field] = fulltree[field][RowNum:RowNum+nRows]
           
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
            interpVal = np.mean([mpb[key][indAfter],mpb[key][indAfter-2]], dtype=mpb[key].dtype)
            mpb[key] = np.insert( mpb[key], indAfter, interpVal )

        if mpb[key].ndim == 2: # [N,3]
            interpVal = np.mean( np.vstack((mpb[key][indAfter,:],mpb[key][indAfter-2,:])), dtype=mpb[key].dtype, axis=0)
            mpb[key] = np.insert( mpb[key], indAfter, interpVal, axis=0)

    return mpb

def mpbPositionComplete(sP, id, extraFields=[]):
    """ Load a particular MPB of subhalo id, and return it along with a filled version of SubhaloPos 
    which interpolates for any skipped intermediate snapshots as well as back beyond the end of the 
    tree to the beginning of the simulation. The return indexed by snapshot number. """
    from tracer.tracerMC import match3

    fields = ['SubfindID','SnapNum','SubhaloPos']

    # any extra fields to be loaded?
    treeFileFields = loadTreeFieldnames(sP)

    for field in iterable(extraFields):
        if field not in fields and field in treeFileFields:
            fields.append(field)

    # load MPB
    mpb = loadMPB(sP, id, fields=fields)

    # load all valid snapshots, then make (contiguous) list from [0, sP.snap]
    snaps = validSnapList(sP)
    times = snapNumToRedshift(sP, snap=snaps, time=True)
    assert snaps.shape == times.shape

    w = np.where(snaps <= sP.snap)
    snaps = snaps[w]
    times = times[w]

    assert len(snaps) == snaps.max() - snaps.min() + 1 # otherwise think more about missing snaps
    assert snaps.min() == 0 # otherwise think more

    # fill any missing [intermediate] snapshots with ghost entries
    for snap in np.arange( mpb['SnapNum'].min(), mpb['SnapNum'].max() ):
        if snap in mpb['SnapNum']:
            continue
        mpb = insertMPBGhost(mpb, snapToInsert=snap)
        #print(' mpb inserted [%d] ghost' % snap)

    # rearrange into ascending snapshot order, and are we already done?
    SubhaloPos  = mpb['SubhaloPos'][::-1,:] # ascending snapshot order
    SnapNum     = mpb['SnapNum'][::-1] # ascending snapshot order

    mpbTimes = cosmo.util.snapNumToRedshift(sP, snap=SnapNum, time=True)

    if np.array_equal(SnapNum, snaps):
        return SnapNum, mpbTimes, SubhaloPos

    # extrapolate back to t=0 beyond the end of the (resolved) tree
    posComplete = np.zeros( (times.size,3), dtype=SubhaloPos.dtype )
    wExtrap = np.where( (times < mpbTimes.min()) | (times > mpbTimes.max()) )

    ind0, ind1 = match3(snaps, SnapNum)
    posComplete[ind0,:] = SubhaloPos[ind1,:]

    for j in range(3):
        # each axis separately, linear extrapolation
        f = interpolate.interp1d(mpbTimes, SubhaloPos[:,j], kind='linear', fill_value='extrapolate')
        assert posComplete[wExtrap,j].sum() == 0.0 # should be empty
        posComplete[wExtrap,j] = f(times[wExtrap])

    return snaps, times, posComplete

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
    from plot.cosmoGeneral import addRedshiftAgeAxes

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

def stellarMergerContribution(sP):
    """ Analysis routine for TNG flagship paper on stellar mass content. """
    from cosmo.util import periodicDistsSq

    # config
    haloMassBins = [[11.4, 11.6], [11.9, 12.1], [12.4,12.6], [12.9,13.1], 
                    [13.4,13.6], [13.9, 14.1], [14.3, 14.7]]
    rad_pkpc = 30.0
    nHistBins = 50
    histMinMax = [8.0, 12.0] # log msun
    threshVals = [0.1, 0.5, 0.9]
    minHaloMassIndiv = 12.0 # log msun
    pt = sP.ptNum('stars')

    # check if we saved some results already?
    r = {}
    saveFilename = 'stellarMergerData_%s_%d.hdf5' % (sP.simName,sP.snap)

    if path.isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # no, start new calculation: load stellar assembly cat
    fName = sP.postPath + 'StellarAssembly/stars_%03d_supp.hdf5' % sP.snap

    with h5py.File(fName,'r') as f:
        InSitu = f['InSitu'][()]
        MergerMass = f['MergerMass'][()]

    MergerMass = sP.units.codeMassToLogMsun( MergerMass )

    # load stellar particle data
    stars = cosmo.load.snapshotSubset(sP, 'stars', ['mass','pos','sftime'])
    assert stars['Masses'].shape == InSitu.shape

    # load groupcat
    radSqMax = sP.units.physicalKpcToCodeLength(rad_pkpc)**2
    gc = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloPos','SubhaloLenType','SubhaloMass','SubhaloGrNr'])['subhalos']
    halos = cosmo.load.groupCat(sP, fieldsHalos=['Group_M_Crit200'])['halos']
    gc['SubhaloOffsetType'] = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    gc['SubhaloMass'] = sP.units.codeMassToLogMsun( gc['SubhaloMass'] )
    halo_masses = sP.units.codeMassToLogMsun( halos )
    halo_masses = halo_masses[ gc['SubhaloGrNr'] ] # re-index to subhalos

    # process centrals only
    sat_inds = cosmo.util.cenSatSubhaloIndices(sP, cenSatSelect='sat')
    halo_masses[sat_inds] = np.nan

    # allocate returns
    binSizeHalf = (histMinMax[1]-histMinMax[0])/nHistBins/2

    r['totalStarMass'] = np.zeros( len(haloMassBins), dtype='float32' )
    r['mergerMassHisto'] = np.zeros( (len(haloMassBins),nHistBins), dtype='float32' )

    # (A) in halo mass bins
    for i, haloMassBin in enumerate(haloMassBins):
        w_sub = np.where( (halo_masses >= haloMassBin[0]) & (halo_masses < haloMassBin[1]) )

        if len(w_sub[0]) == 0:
            continue # empty mass bin for this simulation

        print(haloMassBin, len(w_sub[0]))

        for subhaloID in w_sub[0]:
            # get local indices
            i0 = gc['SubhaloOffsetType'][subhaloID,pt]
            i1 = i0 + gc['SubhaloLenType'][subhaloID,pt]

            if i1 == i0:
                continue # zero length of this type

            # radial restrict
            rr = periodicDistsSq( gc['SubhaloPos'][subhaloID,:], 
                                  stars['Coordinates'][i0:i1,:], sP )

            w_valid = np.where( (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) & \
                                 (rr <= radSqMax) & \
                                 (InSitu[i0:i1] == 0) )

            if len(w_valid[0]) == 0:
                continue # zero stars

            # local properties
            loc_masses = stars['Masses'][i0:i1][w_valid]
            loc_mergermass = MergerMass[i0:i1][w_valid]

            # histogram and save
            loc_hist, hist_bins = np.histogram( loc_mergermass, bins=nHistBins, range=histMinMax, weights=loc_masses )

            r['totalStarMass'][i] += loc_masses.sum()
            r['mergerMassHisto'][i,:] += loc_hist

    # (B) individual halos: intersection with threshold values
    w_sub = np.where( halo_masses >= minHaloMassIndiv )

    nApertures = 4 # <10, <30, >100, all subhalo particles

    r['indivSubhaloIDs'] = w_sub[0]
    r['indivHaloMasses'] = halo_masses[w_sub]
    r['indivHisto'] = np.zeros( (len(w_sub[0]),nApertures,len(threshVals)), dtype='float32' )
    r['indivHisto'].fill(np.nan) # nan value indicates not filled

    print('Processing individuals: ',len(r['indivSubhaloIDs']))

    for i, subhaloID in enumerate(r['indivSubhaloIDs']):
        # get local indices
        if i % 100 == 0: print(i)
        i0 = gc['SubhaloOffsetType'][subhaloID,pt]
        i1 = i0 + gc['SubhaloLenType'][subhaloID,pt]

        if i1 == i0:
            continue # zero length of this type

        # radial restrict
        rr = periodicDistsSq( gc['SubhaloPos'][subhaloID,:], 
                              stars['Coordinates'][i0:i1,:], sP )

        for apertureIter in range(nApertures):
            # aperture selections
            if apertureIter == 0:
                # < 10 pkpc
                w_valid = np.where( (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) & \
                                     (rr <= 10.0) & \
                                     (InSitu[i0:i1] == 0) )
            if apertureIter == 1:
                # < 30 pkpc
                w_valid = np.where( (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) & \
                                     (rr <= 30.0) & \
                                     (InSitu[i0:i1] == 0) )
            if apertureIter == 2:
                # > 100 pkpc
                w_valid = np.where( (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) & \
                                     (rr >= 100.0) & \
                                     (InSitu[i0:i1] == 0) )
            if apertureIter == 3:
                # all subhalo particles
                w_valid = np.where( (stars['GFM_StellarFormationTime'][i0:i1] >= 0.0) & 
                                     (InSitu[i0:i1] == 0) )

            if len(w_valid[0]) == 0:
                continue # zero stars

            # local properties
            loc_masses = stars['Masses'][i0:i1][w_valid]
            loc_mergermass = MergerMass[i0:i1][w_valid]

            # histogram
            loc_hist, hist_bins = np.histogram( loc_mergermass, bins=nHistBins, range=histMinMax, 
                                                weights=loc_masses )

            # normalized cumulative sum
            yy = loc_hist / loc_hist.sum()                
            yy_cum = yy[::-1].cumsum()[::-1] # above a given subhalo mass threshold

            # loop over thresholds, find intersections
            for threshIter, thresh in enumerate(threshVals):
                mass_ind = np.where( yy_cum >= thresh )[0]
                if len(mass_ind) == 0:
                    continue # never crosses threshold? i.e. no ex-situ stars, no stars, ...

                mass_val = hist_bins[mass_ind.max()] + binSizeHalf

                r['indivHisto'][i,apertureIter,threshIter] = mass_val

    # some extra info to save
    r['hist_bins'] = hist_bins[:-1] + binSizeHalf
    r['haloMassBins'] = haloMassBins
    r['histMinMax'] = histMinMax
    r['threshVals'] = threshVals

    # save
    with h5py.File(saveFilename, 'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved: [%s].' % saveFilename)

    return r

def stellarMergerContributionPlot():
    """ Driver. """
    from util.simParams import simParams
    from plot.config import sKn, sKo
    from util.helper import running_median
    from scipy.signal import savgol_filter
    import matplotlib.pyplot as plt
    import csv

    sPs = []

    sPs.append( simParams(res=1820,run='tng',redshift=0.0) )
    sPs.append( simParams(res=2500,run='tng',redshift=0.0) )

    haloBinSize   = 0.1
    apertureNames = ['< 10 kpc', '< 30 kpc', '> 100 kpc', 'central + ICL']
    lw = 2.0
    linestyles = ['-',':']
    nApertures = 4

    # load
    md = {}
    for sP in sPs:
        md[sP.simName] = stellarMergerContribution(sP)

    histMinMax = md[sPs[0].simName]['histMinMax']
    hist_bins  = md[sPs[0].simName]['hist_bins']

    # dump text files
    for i, sP in enumerate(sPs):
        save = []
        save.append(hist_bins)
        for j, haloMassBin in enumerate(md[sP.simName]['haloMassBins']):
            yy = md[sP.simName]['mergerMassHisto'][j,:]
            yy /= md[sP.simName]['mergerMassHisto'][j,:].sum()
            yy_cum = yy[::-1].cumsum()[::-1]
            save.append(yy_cum)

        massBinStrs = ['halo_%.1f_%.1f' % (mb[0],mb[1]) for mb in md[sP.simName]['haloMassBins']]
        header = 'subhalo_mass %s' % ' '.join(massBinStrs)
        np.savetxt('top_panel_%s.txt' % sP.simName, np.array(save).T, fmt='%.5f', header=header)

        save = []
        for apertureIter in range(nApertures):
            for threshIter in range(3):
                x_vals = md[sP.simName]['indivHaloMasses']
                y_vals = md[sP.simName]['indivHisto'][:,apertureIter,threshIter]

                xm, ym, _, pm = running_median(x_vals,y_vals,binSize=haloBinSize,percs=[16,84])
                ym = savgol_filter(ym,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1)

                data = np.zeros( (4,len(xm)), dtype=xm.dtype )
                data[0,:] = xm
                data[1,:] = ym
                data[2,:] = pm[0,:]
                data[3,:] = pm[1,:]

                filename = 'bottom_%s_aperture=%d_thresh=%.1f.txt' % \
                  (sP.simName, apertureIter, md[sPs[0].simName]['threshVals'][threshIter])
                header = '%s\nhalo_m200crit median lower16 upper84' % apertureNames[apertureIter]
                np.savetxt(filename, data.T, fmt='%.5f', header=header)

    # plot
    fig = plt.figure(figsize=(10,16))

    # top panel
    ax = fig.add_subplot(211)
    ax.set_xlabel('M$_{\\rm stars,progenitor}$ [ log M$_{\\rm sun}$ ]')
    ax.set_ylabel('Cumulative Ex-Situ Mass Frac [>= Mass]')
    ax.set_xlim(histMinMax)
    ax.set_ylim([0.0,1.0])

    ax.plot(histMinMax, [0.1,0.1], ':', color='black', alpha=0.05)
    ax.plot(histMinMax, [0.5,0.5], ':', color='black', alpha=0.05)
    ax.plot(histMinMax, [0.9,0.9], ':', color='black', alpha=0.05)

    colors = []
    
    for j, haloMassBin in enumerate(md[sP.simName]['haloMassBins']):
        c = ax._get_lines.prop_cycler.next()['color']
        colors.append(c)

        for i, sP in enumerate(sPs):

            alpha = 1.0
            if (i == 0 and j > 3) or (i == 1 and j < 2):
                alpha = 0.0 # skip lowest two bins for TNG300, highest 3 bins for TNG100

            yy = md[sP.simName]['mergerMassHisto'][j,:]
            # if mergermass==nan (originally -1), then we will not sum all 
            # the weights, such that mergerMassHisto[j,:].sum() <= totalStarMass[j]
            yy /= md[sP.simName]['mergerMassHisto'][j,:].sum() 
            
            yy_cum = yy[::-1].cumsum()[::-1] # above a given subhalo mass threshold
            #label = '%.1f < M$_{\\rm halo}$ < %.1f' % (haloMassBin[0],haloMassBin[1])

            ax.plot(hist_bins, yy_cum, linestyles[i], lw=lw, color=c, label='', alpha=alpha)

    # legend
    sExtra = []
    lExtra = []

    for j, haloMassBin in enumerate(md[sP.simName]['haloMassBins']):
        label = '%.1f < M$_{\\rm halo}$ < %.1f' % (haloMassBin[0],haloMassBin[1])
        sExtra.append( plt.Line2D( (0,1),(0,0),color=colors[j],lw=lw,marker='',linestyle='-' ) )
        lExtra.append( label )
    for i, sP in enumerate(sPs):
        sExtra.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[i] ) )
        lExtra.append( sP.simName )

    legend1 = ax.legend(sExtra, lExtra, loc='upper right', fontsize=13)
    ax.add_artist(legend1)

    # bottom panel
    ax = fig.add_subplot(212)
    ax.set_xlabel('M$_{\\rm halo}$ [ log M$_{\\rm sun}$ ] [ M$_{\\rm 200,crit}$ ]')
    ax.set_ylabel('Satellite Progenitor Threshold Mass [ log M$_{\\rm sun}$ ]')
    ax.set_xlim([12.0,15.0])
    ax.set_ylim([8.0,12.0])

    colors = []
    threshInds = [1,2]

    for apertureIter in range(nApertures):
        c = ax._get_lines.prop_cycler.next()['color']
        colors.append(c)

        for i, sP in enumerate(sPs):
            for threshIter in threshInds:
                x_vals = md[sP.simName]['indivHaloMasses']
                y_vals = md[sP.simName]['indivHisto'][:,apertureIter,threshIter]

                xm, ym, _, pm = running_median(x_vals,y_vals,binSize=haloBinSize,percs=[16,84])

                ym = savgol_filter(ym,sKn,sKo)
                pm = savgol_filter(pm,sKn,sKo,axis=1)

                alpha = [0.0,0.3,1.0][threshIter]

                l, = ax.plot(xm[:-1], ym[:-1], linestyles[i], lw=lw, color=c, alpha=alpha)

                if apertureIter > 0 or threshIter == 1 or i > 0:
                    continue # show percentile scatter only for first aperture
                
                ax.fill_between(xm[:-1], pm[0,:-1], pm[-1,:-1], color=c, interpolate=True, alpha=0.1)

    # legend
    sExtra = []
    lExtra = []

    for apertureIter in range(nApertures):
        sExtra.append( plt.Line2D( (0,1),(0,0),color=colors[apertureIter],lw=lw,marker='',linestyle='-' ) )
        lExtra.append( apertureNames[apertureIter] )
    for threshIter in threshInds:
        alpha = [0.0,0.1,1.0][threshIter]
        sExtra.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle='-',alpha=alpha ) )
        lExtra.append( 'f$_{\\rm ex-situ}$ > %.1f' % md[sPs[0].simName]['threshVals'][threshIter] )
    for i, sP in enumerate(sPs):
        sExtra.append( plt.Line2D( (0,1),(0,0),color='black',lw=lw,marker='',linestyle=linestyles[i] ) )
        lExtra.append( sP.simName )

    legend1 = ax.legend(sExtra, lExtra, loc='upper left', fontsize=13)
    ax.add_artist(legend1)

    # finish plot
    fig.tight_layout()    
    fig.savefig('merger_progmass_%s_%d.pdf' % ('-'.join([sP.simName for sP in sPs]),sP.snap))
    plt.close(fig)
