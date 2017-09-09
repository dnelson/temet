"""
cosmo/util.py
  Helper functions related to cosmo box simulations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import illustris_python as il
import cosmo.load
import copy
from os.path import isfile, isdir
from os import mkdir

# --- snapshot configuration & spacing ---

def redshiftToSnapNum(redshifts=None, sP=None):
    """ Convert one or more input redshifts to closest matching snapshot numbers for a given sP. """
    from util.helper import closest

    assert sP is not None, "Must input sP."

    if redshifts is None:
        redshifts = np.array(sP.redshift)
    else:
        redshifts = np.array(redshifts)

    nSnaps = 400 # maximum

    sbNum, sbStr1, sbStr2 = cosmo.load.subboxVals(sP.subbox)
    if sP.subbox is not None:
        nSnaps *= 10
        if 'tng' in sP.run:
            nSnaps = 16000 # maximum of 16,000 subbox snapshots for new TNG runs

    # load if exists, otherwise create
    r = {}
    saveFilename = sP.derivPath + sP.savPrefix + '_' + sbStr1 + 'snapnum.redshift.hdf5'

    if not isdir(sP.derivPath):
        mkdir(sP.derivPath)

    if isfile(saveFilename):
        with h5py.File(saveFilename, 'r') as f:
            for key in f.keys():
                r[key] = f[key][()]
    else:
        r['redshifts'] = np.zeros(nSnaps, dtype='float32') - 1.0
        r['times']     = np.zeros(nSnaps, dtype='float32') - 1.0
        r['nFound']    = 0

        # attempt load for all snapshots
        for i in range(nSnaps):
            ext = str(i).zfill(3)

            # standard: >1 file per snapshot
            fileName = sP.simPath + sbStr2 + 'snapdir_' + sbStr1 + ext + '/snap_' + sbStr1 + ext + '.0.hdf5'

            # single file per snapshot
            if not isfile(fileName):
                fileName = sP.simPath + sbStr2 + 'snap_' + sbStr1 + ext + '.hdf5'

            # single groupordered file per snapshot
            if not isfile(fileName):
                fileName = sP.simPath + sbStr2 + 'snap-groupordered_' + ext + '.hdf5'

            # allow for existence of groups only (e.g. dev.prime)
            if not isfile(fileName):
                fileName = sP.simPath + 'groups_' + ext + '/fof_subhalo_tab_' + ext + '.0.hdf5'

            # if file doesn't exist yet, skip (e.g. missing/deleted snapshots are ok)
            if not isfile(fileName):
                continue
                
            with h5py.File(fileName, 'r') as f:
                r['redshifts'][i] = f['Header'].attrs['Redshift']
                r['times'][i]     = f['Header'].attrs['Time']

            r['nFound'] += 1

        # save
        with h5py.File(saveFilename, 'w') as f:
            for key in r.keys():
                f[key] = r[key]

    if np.sum(redshifts) == -1:
        raise Exception("Old behavior, used to return !NULL.")

    # return array of snapshot numbers
    snaps = np.zeros( redshifts.size, dtype='int32' )

    for i,redshift in np.ndenumerate(redshifts):
        # closest snapshot redshift to requested
        zFound, w = closest( r['redshifts'], redshift )

        if np.abs(zFound-redshift) > 0.1:
            print("Warning! [%s] Snapshot selected with redshift error = %g" % (sP.simName,np.abs(zFound-redshift)))

        snaps[i] = w

    if snaps.size == 1:
        snaps = snaps[0]

    return snaps
  
def validSnapList(sP, maxNum=None, minRedshift=None, maxRedshift=None, reqTr=False):
    """ Return a list of all snapshot numbers which exist. """
    from util.helper import evenlySample, closest, contiguousIntSubsets

    if minRedshift is None:
        minRedshift = 0.0 # filter out -1 values indicating missing snaps
    if maxRedshift is None:
        maxRedshift = np.finfo('float32').max

    redshifts = snapNumToRedshift(sP, all=True)

    if maxNum is not None and sP.subbox is not None:
        # for subboxes (movie renderings), auto detect change of global timestep
        log_scalefacs = np.log10(1 / (1+redshifts))
        dloga = log_scalefacs - np.roll(log_scalefacs, 1)
        dloga[0] = dloga[1] # corrupted by roll

        dloga_target = np.median( dloga[ int(dloga.size*(2.0/4)):int(dloga.size*(3.0/4)) ])
        print(' validSnapList(): subbox auto detect dloga_target = %f' % dloga_target)

        ww = np.where(dloga < 0.8 * dloga_target)[0]
        print('  number snaps below target [%d] spanning [%d-%d]' % (len(ww),ww.min(),ww.max()))
        ww2 = np.where(dloga < 0.8 * 0.5 * dloga_target)[0]

        #assert len(ww2) == 0 # number of timesteps even one jump lower
        if len(ww2) > 0: print('  WARNING: %d snaps even one timestep below target' % len(ww2))

        # detect contiguous snapshot subsets in this list of integers
        ranges = contiguousIntSubsets(ww)
        print('  identified contiguous snap ranges:',ranges)

        # override every other snapshot in these ranges with a redshift of -1 so it is filtered out below
        for range_start, range_stop in ranges:
            # the first entry here corresponds to the first subbox snapshot whose delta time since the 
            # previous is half of our target, so start removing here so that the dt across this gap 
            # becomes constant
            snap_inds = ww[ range_start : range_stop : 2 ]
            redshifts[snap_inds] = -1.0
            print('  in range [%d to %d] filter out %d snaps' % (range_start,range_stop,snap_inds.size))

    w = np.where((redshifts >= minRedshift) & (redshifts < maxRedshift))[0]

    if len(w) == 0:
        return None

    # require existence of trMC information? have to check now
    if reqTr:
        snaps = w
        w = []

        for snap in snaps:
            fileName = cosmo.load.snapPath(sP.simPath, snap, checkExists=True)
            with h5py.File(fileName,'r') as f:
                if 'PartType'+str(sP.ptNum('tracer')) in f:
                    w.append(snap)

        w = np.array(w)

    # cap at a maximum number of snaps? (evenly spaced)
    if maxNum is not None:
        w = evenlySample(w, maxNum)

    return w

def multiRunMatchedSnapList(runList, method='expand', **kwargs):
    """ For an input runList of dictionaries containing a sP key corresponding to a simParams 
    for each run, produce a 'matched'/unified set of snapshot numbers, one set per run, with 
    all the same length, e.g. for comparative analysis at matched shifts, or for rendering 
    movie frames comparing runs at the same redshift. If method is 'expand', inflate the 
    snapshot lists of all runs to the size of the maximal (duplicates are then guaranteed). 
    If method is 'condense', shrink the snapshot lists of all runs to the size of the minimal 
    (skips are then guaranteed). """
    assert method in ['expand','condense']

    snapLists = []
    numSnaps  = []

    for run in runList:
        runSnaps = validSnapList(run['sP'], **kwargs)

        if runSnaps is None:
            raise Exception('Run [%s] has no snapshots within requested redshift range.' % run['sP'].simName)

        numSnaps.append( len(runSnaps) )

    # let method dictate target size of the matched snapshot lists and 'master' run
    if method == 'expand':
        targetSize = np.max(numSnaps)
        targetRun  = np.argmax(numSnaps)

    if method == 'condense':
        targetSize = np.min(numSnaps)
        targetRun  = np.argmin(numSnaps)

    print('Matched snapshot list [%s] to %d snaps of %s.' % (method,targetSize,runList[targetRun]['sP'].simName))

    # choose the closest snapshot to each target redshift in each run
    targetSnaps = validSnapList(runList[targetRun]['sP'], **kwargs)
    targetRedshifts = snapNumToRedshift(runList[targetRun]['sP'], snap=targetSnaps)

    for run in runList:
        runSnaps = redshiftToSnapNum(targetRedshifts, sP=run['sP'])
        snapLists.append( runSnaps )

    # verify
    assert targetRedshifts.size == targetSize
    for snapList in snapLists:
        assert np.min(snapList) >= 0
        assert snapList.size == targetRedshifts.size

    return snapLists

def snapNumToRedshift(sP, snap=None, time=False, all=False):
    """ Convert snapshot number(s) to redshift or time (scale factor). """
    if not all and snap is None:
        snap = sP.snap
        assert snap is not None, "Input either snap or sP.snap required."

    _, sbStr1, _ = cosmo.load.subboxVals(sP.subbox)

    # load snapshot -> redshift mapping files
    r = {}
    saveFilename = sP.derivPath + sP.savPrefix + '_' + sbStr1 + 'snapnum.redshift.hdf5'

    if not isfile(saveFilename):
        # redshiftToSnapNum() not yet run, do it now
        _ = redshiftToSnapNum(2.0, sP=sP)

    with h5py.File(saveFilename, 'r') as f:
        for key in f.keys():
            r[key] = f[key][()]

    # scale factor or redshift?
    val = r['redshifts']
    if time:
        val = r['time']

    # all values or a given scalar or array list?
    if all:
        w = np.where(val >= 0.0)[0] # remove empties past end of number of snaps
        return val[0 : w.max()+1]

    return val[snap]

def snapNumToAgeFlat(sP, snap=None):
    """ Convert snapshot number to approximate age of the universe at that time. """
    z = snapNumToRedshift(sP, snap=snap)
    return sP.units.redshiftToAgeFlat(z)

def crossMatchSubhalosBetweenRuns(sP_from, sP_to, subhaloInds_from_search, method='LHaloTree'):
    """ Given a set of subhaloInds_from_search in sP_from, find matched subhalos in sP_to.
    Can implement many methods. For now, uses external (pre-generated) postprocessing/SubhaloMatching/ 
    for TNG_method runs, or postprocessing/SubhaloMatchingToDark/ for Illustris/TNG to DMO runs, or 
    postprocessing/SubhaloMatchingToIllustris/ for TNG->Illustris runs.
    Return is an int32 array of the same size as input, where -1 indicates no match. """
    from tracer.tracerMC import match3
    from util.simParams import simParams

    assert method in ['LHaloTree','SubLink','Lagrange','Positional','PositionalAll']
    assert sP_from != sP_to

    # positional cross-match between two different runs?
    if method in ['Positional','PositionalAll']:
        subhaloInds_from_search = np.array(subhaloInds_from_search)
        r = np.zeros( subhaloInds_from_search.size, dtype='int32' ) - 1
        assert sP_to.boxSize == sP_from.boxSize

        # matches which differ by more than this amount are discarded
        massDeltaMaxDex = 0.3

        # filter subhaloInds_from_search to centrals only (ignore satellites if any requested)
        css = 'cen' if method == 'Positional' else 'all'
        cen_inds_to = cenSatSubhaloIndices(sP_to, cenSatSelect=css)
        cen_inds_from = cenSatSubhaloIndices(sP_from, cenSatSelect=css)

        _, ind_from = match3(cen_inds_from, subhaloInds_from_search)
        subhaloInds_from = subhaloInds_from_search[ind_from]

        # load halo masses and positions from both runs
        mhalo_to = cosmo.load.groupCat(sP_to, fieldsSubhalos=['mhalo_200_log'])
        mhalo_from = cosmo.load.groupCat(sP_from, fieldsSubhalos=['mhalo_200_log'])

        pos_from = cosmo.load.groupCat(sP_from, fieldsSubhalos=['SubhaloPos'])['subhalos']
        pos_to = cosmo.load.groupCat(sP_to, fieldsSubhalos=['SubhaloPos'])['subhalos']

        pos_to_cen = pos_to[cen_inds_to,:]

        # loop over each requested search subhalo
        for i, subhaloInd_from in enumerate(subhaloInds_from):
            # calculate distances to all centrals in illustris
            pos_from_loc = np.squeeze(pos_from[subhaloInd_from,:])
            dists = periodicDists(pos_from_loc, pos_to_cen, sP_from)
            pos_to_cen_ind = np.where(dists == dists.min())[0]
            subhaloInd_to = cen_inds_to[pos_to_cen_ind]

            # pass mass requirement? if not, do not save this match
            if np.abs( mhalo_from[subhaloInds_from[i]] - mhalo_to[subhaloInd_to] ) > massDeltaMaxDex:
                continue

            r[ind_from[i]] = subhaloInd_to

        return r

    # are we cross-matching between Illustris and TNG with the Lagrange catalog?
    if sP_from.run == 'illustris' and sP_to.run == 'tng':
        assert method == 'Lagrange'
        assert sP_from.res == sP_to.res == 1820 # otherwise generalize

        matchFilePath = sP_to.postPath + '/SubhaloMatchingToIllustris/'
        matchFileName = matchFilePath + 'LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5' % sP_to.snap

        with h5py.File(matchFileName,'r') as f:
            inds_tng = f['SubhaloIndexFrom'][()]
            inds_illustris = f['SubhaloIndexTo'][()]
            score = f['Score'][()]

        r = np.zeros( len(subhaloInds_from_search), dtype='int32' ) - 1

        # instead: pick central with minimum ID (oh my, do something better)
        cen_inds_to = cenSatSubhaloIndices(sP=sP_to, cenSatSelect='cen')
        print('Warning: inverse Lagrange mapping, taking most massive centrals (doesnt really work).')

        for i in range(len(subhaloInds_from_search)):
            w = np.where(inds_illustris == subhaloInds_from_search[i])[0]

            # candidates
            cand_inds_tng = inds_tng[w]

            # which are centrals? take min ID (most massive)
            cand_inds_tng, _ = match3(cen_inds_to, cand_inds_tng)

            r[i] = cen_inds_to[cand_inds_tng].min()

        return r

    if sP_from.run == 'tng' and sP_to.run == 'illustris':
        assert method == 'Lagrange','Positional'
        r = np.zeros( len(subhaloInds_from_search), dtype='int32' ) - 1

        if method == 'Lagrange':
            assert sP_from.res == sP_to.res == 1820 # otherwise generalize

            matchFilePath = sP_from.postPath + '/SubhaloMatchingToIllustris/'
            matchFileName = matchFilePath + 'LagrangeMatches_L75n1820TNG_L75n1820FP_%03d.hdf5' % sP_from.snap

            with h5py.File(matchFileName,'r') as f:
                inds_tng = f['SubhaloIndexFrom'][()]
                inds_illustris = f['SubhaloIndexTo'][()]
                score = f['Score'][()]

            match_inds_source, match_inds_search = match3(inds_tng, subhaloInds_from_search)
            r[match_inds_search] = inds_illustris[match_inds_source]

        return r

    assert sP_from.snap == sP_to.snap

    # are we cross-matching between two non-fiducial sims.rTNG_method runs?
    if sP_from.run == 'tng' and sP_from.variant.isdigit() and \
       sP_to.run == 'tng' and sP_to.variant.isdigit() and \
       int(sP_to.variant) != 0 and int(sP_from.variant) != 0:
        # we route all such requests through the fiducial run as an intermediate step by self-calling
        assert sP_from.res == sP_to.res # otherwise we need to implement: chain through resolution levels
        sP_fid = simParams(res=sP_from.res, run=sP_from.run, redshift=sP_from.redshift, variant=0000)

        # what are the matched halos of 'from' in 'fiducial'
        match_from_fid = crossMatchSubhalosBetweenRuns(sP_from, sP_fid, subhaloInds_from_search, method=method)

        # what are these match results in 'to'
        match_to_fid = crossMatchSubhalosBetweenRuns(sP_fid, sP_to, match_from_fid, method=method)

        w = np.where(match_from_fid == -1)
        match_to_fid[w] = -1 # flag any matches which failed first step as complete failures

        return match_to_fid

    # are we matching to the DMO analog of sP_from? or vice versa
    swapMatchDirection = False

    dmo1 = (sP_to.run == sP_from.run+'_dm' and sP_to.res == sP_from.res and sP_to.hInd == sP_from.hInd)
    dmo2 = (sP_to.run+'_dm' == sP_from.run and sP_to.res == sP_from.res and sP_to.hInd == sP_from.hInd)

    if dmo1:
        # yes, sP_to is the DMO version of sP_from
        basePath = sP_from.postPath + 'SubhaloMatchingToDark/%s_' % method
        filePath = basePath + '%03d.hdf5' % sP_from.snap
    elif dmo2:
        # yes, sP_from is the DMO version of sP_to
        basePath = sP_to.postPath + 'SubhaloMatchingToDark/%s_' % method
        filePath = basePath + '%03d.hdf5' % sP_to.snap

        swapMatchDirection = True
    else:
        # no, use a match subdirectory to a specific run
        basePath = sP_from.postPath + 'SubhaloMatching/%s/%s_' % (sP_to.simName,method)
        filePath = basePath + '%03d.hdf5' % sP_from.snap

        # only have matching in the other direction?
        if not isfile(filePath):
            basePath = sP_to.postPath + 'SubhaloMatching/%s/%s_' % (sP_from.simName,method)
            filePath = basePath + '%03d.hdf5' % sP_to.snap
            assert isfile(filePath) # otherwise fail

            swapMatchDirection = True

    # load results of a matching catalog
    if method == 'LHaloTree':
        with h5py.File(filePath,'r') as f:
            subhaloInds_from = f['SubhaloIndexFrom'][()]
            subhaloInds_to = f['SubhaloIndexTo'][()]

    if method == 'SubLink':
        with h5py.File(filePath,'r') as f:
            subhaloInds_from = np.arange( f['DescendantIndex'].size )
            subhaloInds_to = f['DescendantIndex'][()]

        w = np.where(subhaloInds_to >= 0)
        subhaloInds_from = subhaloInds_from[w]
        subhaloInds_to = subhaloInds_to[w]

    # if we requested a DMO->Physics match, we have instead loaded [the only thing available] the 
    # Physics->DMO SubhaloMatchingToDark information. so, swap the 'from' and 'to' subhaloInds 
    # such that subhaloInds_from_search is matched against the DMO and the return is for the Physics
    if swapMatchDirection:
        subhaloInds_from, subhaloInds_to = subhaloInds_to, subhaloInds_from

    # find matches and make return
    match_inds_source, match_inds_search = match3(subhaloInds_from, subhaloInds_from_search)

    r = np.zeros( len(subhaloInds_from_search), dtype='int32' ) - 1

    r[match_inds_search] = subhaloInds_to[match_inds_source]

    return r

# --- periodic B.C. ---

def correctPeriodicDistVecs(vecs, sP):
    """ Enforce periodic B.C. for distance vectors (effectively component by component). """
    assert sP.subbox is None
    
    vecs[ np.where(vecs > sP.boxSize*0.5)  ]  -= sP.boxSize
    vecs[ np.where(vecs <= -sP.boxSize*0.5) ] += sP.boxSize

def correctPeriodicPosVecs(vecs, sP):
    """ Enforce periodic B.C. for positions (add boxSize to any negative points, subtract boxSize from any 
        points outside box).
    """
    assert sP.subbox is None

    vecs[ np.where(vecs < 0.0) ]         += sP.boxSize
    vecs[ np.where(vecs >= sP.boxSize) ] -= sP.boxSize

def correctPeriodicPosBoxWrap(vecs, sP):
    """ For an array of positions [N,3], determine if they span a periodic boundary (e.g. half are near 
    x=0 and half are near x=BoxSize). If so, wrap the high coordinate value points by a BoxSize, making 
    them negative. Suitable for plotting particle positions in global coordinates. Return indices of 
    shifted coordinates so they can be shifted back, in the form of dict with an entry for each 
    shifted dimension and key equal to the dimensional index. """
    r = {}

    for i in range(3):
        w1 = np.where(vecs[:,i] < sP.boxSize * 0.1)[0]
        w2 = np.where(vecs[:,i] > sP.boxSize * 0.9)[0]

        # satisfy wrap criterion for this axis?
        if len(w1) and len(w2):
            wCheck = np.where( (vecs[:,i] > sP.boxSize * 0.5) & (vecs[:,i] < sP.boxSize * 0.8) )[0]
            if len(wCheck):
                raise Exception('Positions spanning very large fraction of box, something strange.')

            wMove = np.where(vecs[:,i] > sP.boxSize * 0.8)[0]
            vecs[wMove,i] -= sP.boxSize

            # store indices of shifted coordinates for return
            r[i] = wMove

    return r

def periodicDists(pt, vecs, sP, chebyshev=False):
    """ Calculate distances correctly taking into account periodic B.C.
          if pt is one point: distance from pt to all vecs
          if pt is several points: distance from each pt to each vec (must have same number of points as vecs)
          Chebyshev=1 : use Chebyshev distance metric (greatest difference in positions along any one axis)
    """
    assert vecs.ndim in [1,2]
    assert pt.ndim in [1,2]
    assert vecs.shape[1] == 3 # cannot currently handle vecs.shape == [3] e.g. single vec

    # distances from one point (x,y,z) to a vector of other points [N,3]
    if pt.ndim == 1:
        xDist = vecs[:,0] - pt[0]
        yDist = vecs[:,1] - pt[1]
        zDist = vecs[:,2] - pt[2]

    # distances from a vector of points [N,3] to another vector of other points [N,3]
    if pt.ndim == 2:
        xDist = vecs[:,0] - pt[:,0]
        yDist = vecs[:,1] - pt[:,1]
        zDist = vecs[:,2] - pt[:,2]

    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    if chebyshev:
        dists = np.zeros( xDist.size, dtype='float32' )
        dists[ np.where(xDist > dists) ] = xDist
        dists[ np.where(yDist > dists) ] = yDist
        dists[ np.where(zDist > dists) ] = zDist
    else:
        dists = np.sqrt( xDist*xDist + yDist*yDist + zDist*zDist )

    return dists

def periodicDistsSq(pt, vecs, sP):
    """ As cosmo.util.periodicDists() but specialized, without error checking, and no sqrt. 
    Works either for 2D or 3D, where the dimensions of pt and vecs should correspond. """
    if len(pt) == 2:
        assert vecs.shape[1] == 2

        xDist = vecs[:,0] - pt[0]
        yDist = vecs[:,1] - pt[1]

        correctPeriodicDistVecs(xDist, sP)
        correctPeriodicDistVecs(yDist, sP)

        return xDist*xDist + yDist*yDist
    
    # fall through to normal 3D case
    xDist = vecs[:,0] - pt[0]
    yDist = vecs[:,1] - pt[1]
    zDist = vecs[:,2] - pt[2]

    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    return xDist*xDist + yDist*yDist + zDist*zDist

def periodicPairwiseDists(pts, sP):
    """ Calculate pairwise distances between all 3D points, correctly taking into account periodic B.C. """
    nPts = pts.shape[0]
    num  = nPts*(nPts-1)/2

    ii = 0
    index0 = np.arange(nPts - 1, dtype='int32') + 1
    index1 = np.zeros(num, dtype='int32')
    index2 = np.zeros(num, dtype='int32')

    # set up indexing
    for i in np.arange(nPts-2):
        n1 = nPts - (i+1)
        index1[ii:ii+n1] = i
        index2[ii] = index0[0:n1] + i
        ii += n1

    # component wise difference
    xDist = pts[index1,0] - pts[index2,0]
    yDist = pts[index1,1] - pts[index2,1]
    zDist = pts[index1,2] - pts[index2,2]

    # correct for periodic distance function
    correctPeriodicDistVecs(xDist, sP)
    correctPeriodicDistVecs(yDist, sP)
    correctPeriodicDistVecs(zDist, sP)

    dists = np.sqrt( xDist*xDist + yDist*yDist + zDist*zDist )

    return dists

# --- other ---

def inverseMapPartIndicesToSubhaloIDs(sP, indsType, ptName, debug=False, flagFuzz=True,
                                      SubhaloLenType=None, SnapOffsetsSubhalo=None):
    """ For a particle type ptName and snapshot indices for that type indsType, compute the 
        subhalo ID to which each particle index belongs. Optional: SubhaloLenType (from groupcat) 
        and SnapOffsetsSubhalo (from groupCatOffsetListIntoSnap()), otherwise loaded on demand.
        If flagFuzz is True (default), particles in FoF fuzz are marked as outside any subhalo, 
        otherwise they are attributed to the closest (prior) subhalo.
    """
    if SubhaloLenType is None:
        SubhaloLenType = cosmo.load.groupCat(sP, fieldsSubhalos=['SubhaloLenType'])['subhalos']
    if SnapOffsetsSubhalo is None:
        SnapOffsetsSubhalo = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsSubhalo']

    gcLenType = SubhaloLenType[:,sP.ptNum(ptName)]
    gcOffsetsType = SnapOffsetsSubhalo[:,sP.ptNum(ptName)][:-1]

    # val gives the indices of gcOffsetsType such that, if each indsType was inserted 
    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the 
    # subhalo offset (i.e. first particle) works correctly
    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the 
    # subhalo offset index -after- which the particle should be inserted
    val = np.searchsorted( gcOffsetsType - 1, indsType ) - 1
    val = val.astype('int32')

    # search and flag all matches where the indices exceed the length of the 
    # subhalo they have been assigned to, e.g. either in fof fuzz, in subhalos with 
    # no particles of this type, or not in any subhalo at the end of the file
    if flagFuzz:
        gcOffsetsMax = gcOffsetsType + gcLenType - 1
        ww = np.where( indsType > gcOffsetsMax[val] )[0]

        if len(ww):
            val[ww] = -1

    if debug:        
        # for all inds we identified in subhalos, verify parents directly
        for i in range(len(indsType)):
            if val[i] < 0:
                continue
            assert indsType[i] >= gcOffsetsType[val[i]]
            if flagFuzz:
                assert indsType[i] < gcOffsetsType[val[i]]+gcLenType[val[i]]
                assert gcLenType[val[i]] != 0

    return val

def inverseMapPartIndicesToHaloIDs(sP, indsType, ptName, 
                                   GroupLenType=None, SnapOffsetsGroup=None, debug=False):
    """ For a particle type ptName and snapshot indices for that type indsType, compute the 
        halo/fof ID to which each particle index belongs. Optional: GroupLenType (from groupcat) 
        and SnapOffsetsGroup (from groupCatOffsetListIntoSnap()), otherwise loaded on demand.
    """
    if GroupLenType is None:
        GroupLenType = cosmo.load.groupCat(sP, fieldsHalos=['GroupLenType'])['halos']
    if SnapOffsetsGroup is None:
        SnapOffsetsGroup = cosmo.load.groupCatOffsetListIntoSnap(sP)['snapOffsetsGroup']

    gcLenType = GroupLenType[:,sP.ptNum(ptName)]
    gcOffsetsType = SnapOffsetsGroup[:,sP.ptNum(ptName)][:-1]

    # val gives the indices of gcOffsetsType such that, if each indsType was inserted 
    # into gcOffsetsType just -before- its index, the order of gcOffsetsType is unchanged
    # note 1: (gcOffsetsType-1) so that the case of the particle index equaling the 
    # subhalo offset (i.e. first particle) works correctly
    # note 2: np.ss()-1 to shift to the previous subhalo, since we want to know the 
    # subhalo offset index -after- which the particle should be inserted
    val = np.searchsorted( gcOffsetsType - 1, indsType ) - 1
    val = val.astype('int32')

    if debug:
        # verify directly
        for i in range(len(indsType)):
            if val[i] < 0:
                continue
            assert indsType[i] >= gcOffsetsType[val[i]]
            assert indsType[i] < gcOffsetsType[val[i]]+gcLenType[val[i]]
            assert gcLenType[val[i]] != 0

    return val

def subhaloIDListToBoundingPartIndices(sP, subhaloIDs, groups=False, strictSubhalos=False):
    """ For a list of subhalo IDs, return a dictionary with an entry for each partType, 
    whose value is a 2-tuple of the particle index range bounding the members of the 
    parent groups of this list of subhalo IDs. Indices are inclusive in the sense of 
    cosmo.load.snapshotSubset(). If groups==True, then the input IDs are assumed to be 
    actually group (FoF) IDs, and we return bounding ranges for them. If strictSubhalos==True, 
    then do not use parent groups to bound subhalo members, instead return exact bounding 
    index ranges. """
    if strictSubhalos: assert not groups
    if groups: assert not strictSubhalos #  mutually exclusive

    first_sub = subhaloIDs[0]
    last_sub = subhaloIDs[-1]

    min_sub = np.min(subhaloIDs)
    max_sub = np.max(subhaloIDs)

    if first_sub != min_sub:
        print('Warning: First sub [%d] was not minimum of subhaloIDs [%d].' % (first_sub,min_sub))
        first_sub = min_sub
    if last_sub != max_sub:
        print('Warning: Last sub [%d] was not maximum of subhaloIDs [%d].' % (last_sub,max_sub))
        last_sub = max_sub

    if not groups:
        # get parent groups of extremum subhalos
        first_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=first_sub)['SubhaloGrNr']
        last_sub_groupID = cosmo.load.groupCatSingle(sP, subhaloID=last_sub)['SubhaloGrNr']
    else:
        # input 'subhaloIDs' are already group IDs
        first_sub_groupID = first_sub
        last_sub_groupID = last_sub

    # load group offsets
    snapHeader = cosmo.load.snapshotHeader(sP)
    snapOffsets = cosmo.load.groupCatOffsetListIntoSnap(sP)
    offsets_pt = snapOffsets['snapOffsetsGroup']

    if strictSubhalos:
        # use subhalo offsets instead
        offsets_pt = snapOffsets['snapOffsetsSubhalo']

        # and change *_sub_groupID to actually be our subhalo IDs
        first_sub_groupID = first_sub
        last_sub_groupID = last_sub

        # load length of last subhalo
        last_sub_length = cosmo.load.groupCatSingle(sP, subhaloID=last_sub)['SubhaloLenType']

    r = {}
    for ptName in ['gas','dm','stars','bhs']:
        if snapHeader['NumPart'][sP.ptNum(ptName)] == 0:
            continue # no particles of this type

        # bound upper range using the start of the next group/subhalo, minus one
        r[ptName] = offsets_pt[ [first_sub_groupID,last_sub_groupID+1], sP.ptNum(ptName) ]
        # the final index is inclusive, as in snapshotSubset(), but not as in numpy indexing
        r[ptName][1] -= 1

        if strictSubhalos:
            # need to use the end of the final subhalo, instead of the beginning of the one after, 
            # as they are not necessarily contiguous (if they span across groups)
            # also solves the issue of using the very last subhalo of the catalog
            r[ptName] = offsets_pt[ [first_sub_groupID,last_sub_groupID], sP.ptNum(ptName) ]
            r[ptName][1] += last_sub_length[ sP.ptNum(ptName) ] - 1

        assert r[ptName][1] >= 0 # otherwise indicates we read the last/unused element of offsets_pt

    return r

def cenSatSubhaloIndices(sP=None, gc=None, cenSatSelect=None):
    """ Return a tuple of three sets of indices into the group catalog for subhalos: 
      centrals only, centrals & satellites together, and satellites only. """
    if sP is None:
        assert 'halos' in gc
        assert 'GroupFirstSub' in gc['halos'] and 'Group_M_Crit200' in gc['halos']

    if gc is None:
        # load what we need
        assert sP is not None
        gc = cosmo.load.groupCat(sP, fieldsHalos=['GroupFirstSub','Group_M_Crit200'])

    nSubhalos = cosmo.load.groupCatHeader(sP)['Nsubgroups_Total']

    # halos with a primary subhalo
    wHalo = np.where((gc['halos']['GroupFirstSub'] >= 0) & (gc['halos']['Group_M_Crit200'] > 0))

    # indices
    w1 = gc['halos']['GroupFirstSub'][wHalo] # centrals only
    w2 = np.arange(nSubhalos) # centrals + satellites
    w3 = np.array( list(set(w2) - set(w1)) ) # satellites only

    if cenSatSelect is None:
        return w1, w2, w3

    if cenSatSelect in ['cen','pri','primary','central','centrals']: return w1
    if cenSatSelect in ['sat','sec','secondary','satellite','satellites']: return w3
    if cenSatSelect in ['all','both']: return w2
