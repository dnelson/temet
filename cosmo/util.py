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
from os.path import isfile, isdir
from os import mkdir

def redshiftToSnapNum(redshifts=None, sP=None, subbox=None):
    """ Convert one or more input redshifts to closest matching snapshot numbers for a given sP. """
    from util.helper import closest

    if sP is None:
        raise Exception("Must input sP.")
    if redshifts is None:
        redshifts = np.array(sP.redshift)
    else:
        redshifts = np.array(redshifts)

    nSnaps = 400 # maximum

    sbNum, sbStr1, sbStr2 = cosmo.load.subboxVals(subbox)
    if subbox is not None:
        nSnaps *= 10

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
            print("Warning! Snapshot selected with redshift error = " + str(np.abs(zFound-redshift)))

        snaps[i] = w

    if snaps.size == 1:
        snaps = snaps[0]

    return snaps
  
def validSnapList(sP, maxNum=None):
    """ Return a list of all snapshot numbers which exist. """
    from util.helper import evenlySample

    redshifts = snapNumToRedshift(sP, all=True)
    w = np.where(redshifts >= 0.0)[0]

    if len(w) == 0:
        return None, 0

    # cap at a maximum number of snaps? (evenly spaced)
    if maxNum is not None:
        w = evenlySample(w, maxNum)

    return w, len(w)

def snapNumToRedshift(sP, snap=None, time=False, all=False, subbox=None):
    """ Convert snapshot number to redshift or time (scale factor). """
    if not all:
        if snap is None:
            snap = sP.snap
        if sP.snap is None:
            raise Exception("Input either snap or sP.snap required.")

    _, sbStr1, _ = cosmo.load.subboxVals(subbox)

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
        return val

    return val[snap]

def snapNumToAgeFlat(sP, snap=None):
    """ Convert snapshot number to approximate age of the universe at that time. """
    z = snapNumToRedshift(sP, snap=snap)
    return sP.units.redshiftToAgeFlat(z)

# --- periodic B.C. ---

def correctPeriodicDistVecs(vecs, sP):
    """ Enforce periodic B.C. for distance vectors (effectively component by component). """
    vecs[ np.where(vecs > sP.boxSize*0.5)  ]  -= sP.boxSize
    vecs[ np.where(vecs <= -sP.boxSize*0.5) ] += sP.boxSize

def correctPeriodicPosVecs(vecs, sP):
    """ Enforce periodic B.C. for positions (add boxSize to any negative points, subtract boxSize from any 
        points outside box).
    """
    vecs[ np.where(vecs < 0.0) ]         += sP.boxSize
    vecs[ np.where(vecs >= sP.boxSize) ] -= sP.boxSize

def correctPeriodicPosBoxWrap(vecs, sP):
    """ For an array of positions [N,3], determine if they span a periodic boundary (e.g. half are near 
    x=0 and half are near x=BoxSize). If so, wrap the high coordinate value points by a BoxSize, making 
    them negative. Suitable for plotting particle positions in global coordinates. """

    for i in range(2):
        w1 = np.where(vecs[:,i] < sP.boxSize * 0.1)[0]
        w2 = np.where(vecs[:,i] > sP.boxSize * 0.9)[0]

        # satisfy wrap criterion for this axis?
        if len(w1) and len(w2):
            wCheck = np.where(vecs[:,i] > sP.boxSize * 0.5 and vecs[:,i] < sP.boxSize * 0.8)[0]
            if len(wCheck):
                raise Exception('Positions spanning very large fraction of box, something strange.')

            wMove = np.where(vecs[:,i] > sP.boxSize * 0.8)[0]
            vecs[wMove,i] -= sP.boxSize

def periodicDists(pt, vecs, sP, chebyshev=False):
    """ Calculate distances correctly taking into account periodic B.C.
          if pt is one point: distance from pt to all vecs
          if pt is several points: distance from each pt to each vec (must have same number of points as vecs)
          Chebyshev=1 : use Chebyshev distance metric (greatest difference in positions along any one axis)
    """

    if (vecs.ndim != 1 and vecs.ndim != 2) or vecs.shape[1] != 3:
        raise Exception("Input vecs not in expected shape.")
    if pt.ndim not in [1,2]:
        raise Exception("Something in wrong, pt has strange dimensionality.")

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

def gasMassesFromIDs(search_ids, sP):
    """ Return individual gas cell/particle masses given input ID list. """
    from util.helper import getIDIndexMap

    if sP.snap is None:
        raise Exception("Need sP.snap")

    h = cosmo.load.snapshotHeader(sP)
    
    if sP.trMCPerCell == 0:
        # SPH case: all particles have constant mass
        masses = np.zeros( h.NumPart[il.util.partTypeNum('gas')], dtype='float32' )
        masses += sP.targetGasMass
    else:
        # Arepo case: variable gas cell masses
        ids = cosmo.load.snapshotSubset(sP, partType='gas', fields='ids')

        idIndexMap, minID = getIDIndexMap(ids)
        inds = idIndexMap[search_ids - minID]

        del ids
        del idIndexMap

        masses = cosmo.load.snapshotSubset(sP, partType='gas', fields='mass', inds=inds)

    return masses

# --- plotting ---

def addRedshiftAxis(ax, sP, zVals=[0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0,10.0]):
    """ Add a redshift axis as a second x-axis on top (assuming bottom axis is Age of Universe [Gyr]). """
    axTop = ax.twiny()
    axTickVals = sP.units.redshiftToAgeFlat( np.array(zVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(zVals)
    axTop.set_xlabel("Redshift")

def addUniverseAgeAxis(ax, sP, ageVals=[0.7,1.0,1.5,2.0,3.0,4.0,6.0,9.0]):
    """ Add a age of the universe [Gyr] axis as a second x-axis on top (assuming bottom is redshift). """
    axTop = ax.twiny()

    ageVals.append( sP.units.redshiftToAgeFlat([0.0]).round(2) )
    axTickVals = sP.units.ageFlatToRedshift( np.array(ageVals) )

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals)
    axTop.set_xticklabels(ageVals)
    axTop.set_xlabel("Age of the Universe [Gyr]")

def addRedshiftAgeAxes(ax, sP, xrange=[-1e-4,8.0], xlog=True):
    """ Add bottom vs. redshift (and top vs. universe age) axis for standard X vs. redshift plots. """
    ax.set_xlim(xrange)
    ax.set_xlabel('Redshift')

    if xlog:
        ax.set_xscale('symlog')
        zVals = [0,0.5,1,1.5,2,3,4,5,6,7,8] # [10]
    else:
        ax.set_xscale('linear')
        zVals = [0,1,2,3,4,5,6,7,8]

    ax.set_xticks(zVals)
    ax.set_xticklabels(zVals)

    addUniverseAgeAxis(ax, sP)

def plotRedshiftSpacings():
    """ Compare redshift spacing of snapshots of different runs. """
    import matplotlib.pyplot as plt
    from util import simParams

    # config
    sPs = []
    sPs.append( simParams(res=512,run='tracer') )
    sPs.append( simParams(res=512,run='feedback') )
    sPs.append( simParams(res=1820,run='illustris') )

    # plot setup
    xrange = [0.0, 14.0]
    yrange = [0.5, len(sPs) + 0.5]

    runNames = []
    for sP in sPs:
        runNames.append(sP.run)

    fig = plt.figure(figsize=(16,6))

    ax = fig.add_subplot(111)
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)

    ax.set_xlabel('Age of Universe [Gyr]')
    ax.set_ylabel('')

    ax.set_yticks( np.arange(len(sPs))+1 )
    ax.set_yticklabels(runNames)
    
    # loop over each run
    for i, sP in enumerate(sPs):
        zVals = snapNumToRedshift(sP,all=True)
        zVals = sP.units.redshiftToAgeFlat(zVals)

        yLoc = (i+1) + np.array([-0.4,0.4])

        for zVal in zVals:
            ax.plot([zVal,zVal],yLoc,lw=0.5,color=sP.colors[1])

    # redshift axis
    addRedshiftAxis(ax, sP)

    fig.tight_layout()    
    fig.savefig(sP.plotPath + 'redshift_spacing.pdf')