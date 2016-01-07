"""
cosmo/util.py
  Helper functions related to cosmo box simulations.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import illustris_python as il
from os.path import isfile
import cosmo.load
from util.helper import getIDIndexMap

def redshiftToSnapNum(redshifts=None, sP=None, subbox=None):
    """ Convert one or more input redshifts to closest matching snapshot numbers for a given sP. """
    if sP is None:
        raise Exception("Must input sP.")
    if redshifts is None:
        redshifts = np.array(sP.redshift)
    else:
        redshifts = np.array(redshifts)

    nSnaps = 400 # maximum

    # subbox support
    sbNum = subbox if isinstance(subbox, (int,long)) else 0

    if subbox is not None:
        sbStr1 = 'subbox' + str(sbNum) + '_'
        sbStr2 = 'subbox' + str(sbNum) + '/'
        nSnaps *= 10
    else:
        sbStr1 = ''
        sbStr2 = ''

    # load if exists, otherwise create
    r = {}
    saveFilename = sP.derivPath + sP.savPrefix + '_' + sbStr1 + 'snapnum.redshift.hdf5'

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
        dists = np.abs(r['redshifts'] - redshift)
        w = np.argmin( dists )

        if dists[w] > 0.1:
            print("Warning! Snapshot selected with redshift error = " + str(dists[w]))

        snaps[i] = w

    if snaps.size == 1:
        snaps = snaps[0]

    return snaps
    
def gasMassesFromIDs(search_ids, sP):
    """ Return individual gas cell/particle masses given input ID list. """
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

def snapNumToRedshift(snap=None, sP=None):
    return 0.0 # TODO
    