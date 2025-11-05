"""
Cosmological simulations - working with merger trees (SubLink, LHaloTree, C-Trees).
"""
import numpy as np
import h5py
from numba import jit
from scipy.signal import savgol_filter
from scipy import interpolate

import illustris_python as il
from ..util.helper import logZeroNaN, running_sigmawindow, iterable, closest, cache

treeName_default = "SubLink"

def loadMPB(sP, id, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load fields of main-progenitor-branch (MPB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        tree = il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        tree = il.lhalotree.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMPB=True)

    tree['Redshift'] = sP.snapNumToRedshift(tree['SnapNum'])

    return tree

def loadMDB(sP, id, fields=None, treeName=treeName_default, fieldNamesOnly=False):
    """ Load fields of main-descendant-branch (MDB) of subhalo id from the given tree. """
    assert sP.snap is not None, "sP.snap required"

    if treeName in ['SubLink','SubLink_gal']:
        tree = il.sublink.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMDB=True, treeName=treeName)
    if treeName in ['LHaloTree']:
        tree = il.lhalotree.loadTree(sP.simPath, sP.snap, id, fields=fields, onlyMDB=True)
    
    tree['Redshift'] = sP.snapNumToRedshift(tree['SnapNum'])

    return tree

def loadMPBs(sP, ids, fields=None, treeName=treeName_default):
    """ Load multiple MPBs at once (e.g. all of them), optimized for speed, with a full tree load (high mem).
    Basically a rewrite of illustris_python/sublink.py under specific conditions (hopefully temporary).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      ids (list[int]): list of subhalo IDs to load.
      fields (list[str]): list of field names to load, or None for all (not recommended).
      treeName (str): which merger tree to use? 'SubLink' or 'SubLink_gal'.

    Returns: 
      dict: Dictionary of MPBs where keys are subhalo IDs, and the contents of each dict value is another 
      dictionary of identical stucture to the return of loadMPB().
    """
    from glob import glob
    assert treeName in ['SubLink','SubLink_gal'] # otherwise need to generalize tree loading

    # make sure fields is not a single element
    if isinstance(fields, str):
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
        if prefix+'RowNum' not in f:
            return result # early snapshots, no tree offset

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

def insertMPBGhost(mpb, snap):
    """ Insert a ghost entry into a MPB dict by interpolating over neighboring snapshot information.
    Appropriate if e.g. a group catalog if corrupt but snapshot files are ok. Could also be used to 
    selectively wipe out outlier points in the MPB. """
    indAfter = np.where(mpb['SnapNum'] == snap - 1)[0]
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

def mpbPositionComplete(sP, id, extraFields=None):
    """ Load a particular MPB of subhalo id, and return it along with a filled version of SubhaloPos 
    which interpolates for any skipped intermediate snapshots as well as back beyond the end of the 
    tree to the beginning of the simulation. The return indexed by snapshot number. """
    from ..tracer.tracerMC import match3
    if extraFields is None: extraFields = []

    fields = ['SubfindID','SnapNum','SubhaloPos']

    # any extra fields to be loaded?
    treeFileFields = loadTreeFieldnames(sP)

    for field in iterable(extraFields):
        if field not in fields and field in treeFileFields:
            fields.append(field)

    # load MPB
    mpb = loadMPB(sP, id, fields=fields)

    # load all valid snapshots, then make (contiguous) list from [0, sP.snap]
    snaps = sP.validSnapList()
    times = sP.snapNumToRedshift(snap=snaps, time=True)
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
        mpb = insertMPBGhost(mpb, snap)
        #print(' mpb inserted [%d] ghost' % snap)

    # rearrange into ascending snapshot order, and are we already done?
    SubhaloPos  = mpb['SubhaloPos'][::-1,:] # ascending snapshot order
    SnapNum     = mpb['SnapNum'][::-1] # ascending snapshot order

    mpbTimes = sP.snapNumToRedshift(snap=SnapNum, time=True)

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

@cache
def quantMPB(sim, subhaloInd, quants, add_ghosts=False, z_vals=None, smooth=False):
    """ Return particular quantit(ies) from a MPB. 
    A simplified version of e.g. simSubhaloQuantity(). Can be generalized in the future. 
    Returned units should be consistent with simSubhaloQuantity() of the same quantity name.
    
    Args:
      sim (:py:class:`~util.simParams`): simulation instance.
      subhaloInd (int): subhalo index.
      quants (list[str]): quantities to return.
      add_ghosts (bool): fill any missing snapshots with ghost entries?
      z_vals (list[float]): if not None, restrict to these redshift values only.
      smooth (bool): smooth the returned quantities in time.
    """
    quants = iterable(quants)

    # load main progenitor branch
    mpb = sim.loadMPB(subhaloInd)

    r = {}

    # fill any missing snapshots with ghost entries? (e.g. actual trees can skip a snapshot when 
    # locating a descendant but we may need a continuous position for all snapshots)
    if add_ghosts:
        for snap in np.arange(mpb['SnapNum'].min(), mpb['SnapNum'].max()):
            if snap in mpb['SnapNum']:
                continue
            mpb = insertMPBGhost(mpb, snap=snap)
            print(' mpb inserted [%d] ghost' % snap)

    # add redshift
    mpb_z = sim.snapNumToRedshift(mpb['SnapNum'])
    mpb_a = 1 / (1 + mpb_z)
    r['z'] = mpb_z

    # restrict to a set of discrete redshifts?
    inds = np.where(r['z'])[0]
    if z_vals is not None:
        inds = np.array([closest(r['z'], z_val)[1] for z_val in z_vals])
        r['z'] = r['z'][inds]

    # loop over requested quantities
    for quant in quants:
        prop = quant.lower().replace('_log','') # new
        log = True # default
        vals = None

        if prop in ['mhalo_200','mhalo']:
            vals = mpb['Group_M_Crit200']
            vals = sim.units.codeMassToMsun(vals)

        if prop in ['rhalo_200','rhalo','r200','r200c','rvir','r_vir','rvirial']:
            vals = mpb['Group_R_Crit200']
            #vals *= mpb_a # comoving -> physical
            # todo: the following inconsistency wrt the units of simSubhaloQuantity() 
            # is for its historical use in tracer_Mc and renderSingleHaloFrames()
            print(f'NOTE: [{sim}] quantMPB [{prop}] in code (comoving) units!')

        if prop in ['t_vir']:
            vals = sim.units.codeMassToVirTemp(mpb['Group_M_Crit200'])
        
        if prop in ['s_vir']:
            vals = sim.units.codeMassToVirEnt(mpb['Group_M_Crit200'])

        if prop in ['v_vir']:
            vals = sim.units.codeMassToVirVel(mpb['Group_M_Crit200'])

        if prop in ['mstar']:
            vals = mpb['SubhaloMassType'][:,sim.ptNum('stars')]
            vals = sim.units.codeMassToMsun(vals)

        if prop in ['mstar2']:
            vals = mpb['SubhaloMassInRadType'][:,sim.ptNum('stars')]
            vals = sim.units.codeMassToMsun(vals)

        if prop in ['mgas2']:
            vals = mpb['SubhaloMassInRadType'][:,sim.ptNum('gas')]
            vals = sim.units.codeMassToMsun(vals)

        if prop in ['mass_smbh']:
            vals = mpb['SubhaloBHMass']
            vals = sim.units.codeMassToMsun(vals)

        if prop in ['sfr2']:
            vals = mpb['SubhaloSFRinRad']

        if prop in ['size_stars','rhalf_stars']:
            vals = mpb['SubhaloHalfmassRadType'][:,sim.ptNum('stars')]
            vals = sim.units.codeLengthToComovingKpc(vals)
            vals *= mpb_a # comoving -> physical

        if prop in ['size_gas']:
            vals = mpb['SubhaloHalfmassRadType'][:,sim.ptNum('gas')]
            vals = sim.units.codeLengthToComovingKpc(vals)
            vals *= mpb_a # comoving -> physical

        if prop in ['z_stars']:
            if 'SubhaloStarMetallicity' in mpb: # not in MCST
                vals = sim.units.metallicityInSolar(mpb['SubhaloStarMetallicity'])

        if prop in ['z_gas']:
            if 'SubhaloGasMetallicity' in mpb: # not in MCST
                vals = sim.units.metallicityInSolar(mpb['SubhaloGasMetallicity'])

        if prop in ['z_gas_sfr','z_gas_sfrwt']:
            if 'SubhaloGasMetallicitySfrWeighted' in mpb: # not in MCST
                vals = sim.units.metallicityInSolar(mpb['SubhaloGasMetallicitySfrWeighted'])

        # unchanged fields from the tree
        if quant.replace('_log','') in mpb.keys():
            vals = mpb[quant.replace('_log','')]

        if vals is None:
            # unrecognized quant name, likely custom/auxcat-type, need to compute separately for each snap
            print(f'Computing custom MPB quant [{quant}] for [{sim}].')
            sim_loc = sim.copy()

            for i, tree_ind in enumerate(inds):
                sim_loc.setSnap(mpb['SnapNum'][tree_ind])

                vals_loc = sim_loc.subhalos(quant)
                val_loc = vals_loc[mpb['SubfindID'][tree_ind]]

                if i == 0:
                    vals = np.zeros(mpb['SnapNum'].size, dtype=vals_loc.dtype)
                vals[tree_ind] = val_loc

        # smooth?
        if smooth and quant not in ['SubfindID','SnapNum']:
            from ..plot.config import sKn, sKo

            # 3-vectors
            if vals.ndim > 1:
                if prop in ['pos','subhalopos']:
                    # positions with box-edge shifting
                    posShiftInds = sim.correctPeriodicPosBoxWrap(mpb['SubhaloPos'])

                for i in range(vals.shape[1]):
                    vals[:,i] = savgol_filter(vals[:,i], sKn, sKo)

                    if prop in ['pos','subhalopos'] and i in posShiftInds:
                        assert 0 # old code, should be checked
                        unShift = np.zeros( len(mpb['Redshift']), dtype='float32' )
                        unShift[posShiftInds[i]] = sim.boxSize
                        vals[:,i] = vals[:,i] + unShift
            else:
                vals = savgol_filter(vals, sKn, sKo)

        # take redshift subset
        vals = vals[inds]

        # take log?
        if '_log' in quant and log:
            log = False
            vals = logZeroNaN(vals)

        # attach
        r[quant] = vals

    # return
    return r

@jit(nopython=True, nogil=True, cache=True)
def _helper_plot_tree(SnapNum,SubhaloID,DescendantID,FirstProgenitorID):
    """ JITed helper to do the structural loops over the tree, which can be slow for big trees. """
    nrows = SnapNum.size
    snapnum_min = np.min(SnapNum)
    snapnum_max = np.max(SnapNum)

    max_progenitors = np.zeros(nrows, dtype=np.float32)
    
    # iterate over snapshots
    for snapnum in range(snapnum_min, snapnum_max):
        # iterate over subhalos from current snapshot
        locs = np.where(SnapNum == snapnum)[0]

        for rownum in locs:
            sub_id  = SubhaloID[rownum]
            desc_id = DescendantID[rownum]
            first_prog_id = FirstProgenitorID[rownum]

            if first_prog_id == -1:
                assert max_progenitors[rownum] == 0
                max_progenitors[rownum] = 1

            assert desc_id != -1

            rownum_desc = rownum - (sub_id - desc_id)
            max_progenitors[rownum_desc] += max_progenitors[rownum]

    xref  = np.zeros(nrows, dtype=np.float32)
    dx    = np.ones(nrows, dtype=np.float32)
    xc    = np.zeros(nrows, dtype=np.float32) + 0.5
    yc    = SnapNum.copy()
    lines = np.zeros( (nrows,2,2), dtype=np.float32 )

    # iterate over snapshots and subhalos again, but this time starting from the last snapshot
    for snapnum in np.arange(snapnum_max-1, snapnum_min-1, -1): # reversed(range(snapnum_min,snapnum_max))
        locs = np.where(SnapNum == snapnum)[0]

        for rownum in locs:
            sub_id = SubhaloID[rownum]
            desc_id = DescendantID[rownum]
            rownum_desc = rownum - (sub_id - desc_id)

            dx[rownum]   = dx[rownum_desc] * max_progenitors[rownum] / max_progenitors[rownum_desc]
            xref[rownum] = xref[rownum_desc]
            xc[rownum]   = xref[rownum_desc] + 0.5 * dx[rownum]

            xref[rownum_desc] += dx[rownum]

            # store lines
            lines[rownum,0,0] = xc[rownum] # x0
            lines[rownum,0,1] = yc[rownum] # y0
            lines[rownum,1,0] = xc[rownum_desc] # x1
            lines[rownum,1,1] = yc[rownum_desc] # y1

    return xc, yc, lines

def plot_tree(sP, subhaloID, saveFilename, treeName=treeName_default, dpi=100, ctName='inferno', output_fmt='png'):
    """ Visualize a full merger tree of a given subhalo. saveFilename can either be a string, BytesIO buffer, or None. 
    If a buffer, output_fmt should specify the required format (e.g. 'pdf', 'png', 'jpg'). If None, then the plot 
    is rendered herein and the final image array (uint8) is returned, for e.g. image manipulation procedures. """
    from ..plot.config import figsize
    from ..util.helper import loadColorTable, logZeroNaN
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.collections import LineCollection
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    alpha = 0.7 # markers
    color = '#000000' # lines

    # load tree
    fields = ['SubhaloID', 'DescendantID', 'FirstProgenitorID', 'SnapNum', 'SubhaloMass', 'SubhaloMassType', 'SubhaloSFR']
    if sP.isDMO:
        fields = ['SubhaloID', 'DescendantID', 'FirstProgenitorID', 'SnapNum', 'SubhaloMass', 'SubhaloVel']
    tree = il.sublink.loadTree(sP.simPath, sP.snap, subhaloID, fields=fields, treeName=treeName)

    if tree is None:
        # subhalo not in tree
        return None

    nrows = tree['count']

    alpha2 = np.clip( 0.1*200/np.sqrt(nrows), 0.01, 0.4 )
    lw     = np.clip( 1.0*200/np.sqrt(nrows), 0.4, 1.0 )
    markerSizeFac = np.clip( 80/nrows**(1.0/4.0), 5.0, 10.0 )

    if nrows > 5e5:
        minMarkerSize = 4.0
    elif nrows > 1e5:
        minMarkerSize = 3.5
    elif nrows > 1e4:
        minMarkerSize = 3.0
    elif nrows > 1e3:
        minMarkerSize = 2.0
    else:
        minMarkerSize = 1.5

    # calculate marker sizes
    markersize = markerSizeFac * (tree['SubhaloMass'])**(1.0/4.0)

    # the calibration above accounts mainly for trees of different mass halos calibrated at ~TNG100-1 resolution
    # but at different resolutions, trees of the same halo mass (e.g. typical circle sizes) have much fewer
    # nrows and so are excessively boosted - here we scale them back to a canonical mean size
    if sP.isDMO:
        targetDMMass1820 = 0.00050557
        resFac = (targetDMMass1820/sP.dmParticleMass)**(1.0/5.0)
    else:
        targetGasMass1820 = 9.4395e-05
        resFac = (targetGasMass1820/sP.targetGasMass)**(1.0/5.0)
    markersize *= resFac

    #print(' min to plot: ',minMarkerSize)
    #print(' markersizefac: ', markerSizeFac, ' mean markersize: ', markersize.mean())
    #print(' count: ', nrows)
    #print(' lw:', lw, ' alpha2: ', alpha2)

    # calculate color quantity
    if sP.isDMO:
        vmag = np.sqrt( tree['SubhaloVel'][:,0]**2 + tree['SubhaloVel'][:,1]**2 + tree['SubhaloVel'][:,2]**2 )
        vmag = sP.units.subhaloCodeVelocityToKms(vmag)
        tree['colorField'] = logZeroNaN(vmag)
        clabel = 'log(Velocity) [km/s]'
        cminmax = [1.4, 3.0] # default
    else:
        # sSFR
        mstar = sP.units.codeMassToMsun( tree['SubhaloMassType'][:,sP.ptNum('stars')] )
        w = np.where(mstar == 0.0)
        mstar[w] = np.nan

        with np.errstate(invalid='ignore'):
            tree['colorField'] = logZeroNaN(tree['SubhaloSFR'] / mstar ) # 1/yr
        clabel = 'log(sSFR) [yr$^{-1}$]'
        cminmax = [-12.0, -8.0] # default

    # call JITed helper
    xc, yc, lines = _helper_plot_tree(tree['SnapNum'],tree['SubhaloID'],tree['DescendantID'],tree['FirstProgenitorID'])

    # start plot
    fig = plt.figure(figsize=(figsize[0]*1.0,figsize[1]*1.0), dpi=dpi) # 1400x1000 px
    #fig = plt.figure(figsize=(19.2,10.8)) # 1920x1080 px
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.02,1.02])

    redshift_ticks = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0])
    snapnum_ticks  = sP.redshiftToSnapNum(redshift_ticks)

    marker0pad = int(np.log10(nrows) * sP.numSnaps / 100.0) # pretty hacky
    ymin_snap = np.clip(snapnum_ticks[-1]-4, 0, np.inf) # sets TNG at 0, but sets e.g. Illustris just past z=10 (many snaps away from 0)
    ymax_snap = np.clip(sP.snap+marker0pad, snapnum_ticks[3], np.inf) # scale y-axis, but start at z=2 at earliest
    ax.set_ylim([ymin_snap, ymax_snap]) 

    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    w = np.where(snapnum_ticks <= ymax_snap)[0]
    ax.set_yticks(snapnum_ticks[w])
    ax.set_yticklabels(['z = %.1f' % z for z in redshift_ticks[w]])

    ax.set_ylabel('%s (subhaloID = %d at snap %d)' % (sP.simName,subhaloID,sP.snap))

    # plot 'root' subhalo marker
    markersize0 = markerSizeFac * (tree['SubhaloMass'][-1])**(1.0/4.0)
    ax.plot(xc[0], yc[0], 'o', markersize=markersize0, color=color, alpha=1.0)

    # add markers
    cmap = loadColorTable(ctName)
    if np.count_nonzero( np.isfinite(tree['colorField']) ):
        vmin = np.round(np.nanmin(tree['colorField']) - 0.5)
        vmax = np.round(np.nanmax(tree['colorField']))
    else:
        vmin = cminmax[0]
        vmax = cminmax[1]
    norm = Normalize(vmin=vmin, vmax=vmax)

    w_minsize = np.where(markersize >= minMarkerSize)

    with np.errstate(invalid='ignore'):
        colors = cmap(norm(tree['colorField']))

    ax.scatter(xc[w_minsize], yc[w_minsize], s=markersize[w_minsize]**2, marker='o', color=colors[w_minsize], alpha=alpha)

    # add connecting lines
    lc = LineCollection(lines, color=color, lw=lw, alpha=alpha2)
    ax.add_collection(lc)

    # colorbar
    cax = inset_axes(ax, width='4%', height='40%', borderpad=1.0, loc='upper left')
    cb = ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')
    cb.ax.set_ylabel(clabel)

    # finish (note: saveFilename could be in-memory buffer)
    if saveFilename is not None:
        fig.savefig(saveFilename, format=output_fmt, dpi=dpi)
        plt.close(fig)

        return True
    else:
        # return image array itself, i.e. draw the canvas then extract the (Nx,Ny,3) array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()

        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        plt.close(fig)

        return image

def test_plot_tree():
    from ..util.simParams import simParams
    sP = simParams(res=1820,run='tng',redshift=0.0)

    for haloID in [200]: #[20,200,210,600,2000,20000]:
        print(haloID)
        subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']
        saveFilename = 'mergertree_%s_%d.png' % (sP.simName,haloID)

        plot_tree(sP, subhaloID, saveFilename)

def test_plot_tree_mem(haloID=19):
    from io import BytesIO
    import time
    from imageio import imread, imwrite
    from scipy.misc import imresize
    from ..util.simParams import simParams

    # config
    sP = simParams(res=1820,run='tng',snap=99)

    supersample = 4
    output_fmt = 'png'

    # start
    buf = None # return image array by default

    if output_fmt == 'pdf':
        # fill memory buffer instead
        buf = BytesIO()

    subhaloID = sP.groupCatSingle(haloID=haloID)['GroupFirstSub']

    # render
    start_time = time.time()

    im = plot_tree(sP, subhaloID, saveFilename=buf, dpi=100*supersample, output_fmt=output_fmt)

    print('plotting done, [%.1f sec]' % (time.time()-start_time))

    # 'resize'
    if output_fmt in ['png','jpg']:
        im = imresize(im, 1.0/supersample, interp='bicubic')

        # 'save'
        buf = BytesIO()
        imwrite(buf, im, format=output_fmt)

    with open('mergertree_%s_snap-%d_%d.%s' % (sP.simName,sP.snap,haloID,output_fmt),'wb') as f:
        f.write(buf.getvalue())
