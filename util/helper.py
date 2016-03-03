"""
util/helper.py
  General helper functions, small algorithms, basic I/O, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np

# --- utility functions ---

def nUnique(x):
    """ Return number of unique elements in input numpy array x. """
    return (np.unique(x)).size

def isUnique(x):
    """ Does input array contain only unique values? """
    return x.size == (np.unique(x)).size

def closest(array, value):
    """ Return closest element of array to input value. """
    ind = ( np.abs(array-value) ).argmin()
    return array[ind], ind

def evenlySample(sequence, num, logSpace=False):
    """ Return num samples from sequence roughly equally spaced. """
    if sequence.size <= num:
        return sequence

    if logSpace:
        inds = np.logspace(0.0, np.log10(float(sequence.size)-1), num)
    else:
        inds = np.linspace(0.0, float(sequence.size)-1, num)

    return sequence[inds.astype('int32')]

def reportMemory():
    """ Return current Python process memory usage in GB. """
    import os
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0**3 # GB

# --- general algorithms ---

def running_median(X, Y, nBins=100, binSize=None):
    """ Create a adaptive median line of a (x,y) point set using some number of bins. """
    if binSize is not None:
        nBins = round( (X.max()-X.min()) / binSize )

    bins = np.linspace(X.min(),X.max(), nBins)
    delta = bins[1]-bins[0]

    running_median = []
    running_std    = []
    bin_centers    = []

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= bin) & (X < binMax))

        if len(w[0]):
            running_median.append( np.nanmedian(Y[w]) )
            running_std.append( np.std(Y[w]) )
            bin_centers.append( np.nanmedian(X[w]) )

    return bin_centers, running_median, running_std

def running_histogram(X, nBins=100, binSize=None, normFac=None):
    """ Create a adaptive histogram of a (x) point set using some number of bins. """
    if binSize is not None:
        nBins = round( (X.max()-X.min()) / binSize )

    bins = np.linspace(X.min(),X.max(), nBins)
    delta = bins[1]-bins[0]

    running_h   = []
    bin_centers = []

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= bin) & (X < binMax))

        if len(w[0]):
            running_h.append( len(w[0]) )
            bin_centers.append( np.nanmedian(X[w]) )

    if normFac is not None:
        running_h /= normFac

    return bin_centers, running_h

def replicateVar(childCounts, subsetInds=None):
    """ Given a number of children for each parent, replicate the parent indices for each child.
          subset_inds : still need to walk the full child_counts, but only want parent indices of a subset.
    """
    offset = 0

    if subsetInds is None:
        # full
        parentInds = np.array( np.sum(childCounts), dtype='uint32' )

        for i in np.arange(childCounts.size):
            if childCounts[i] > 0:
                parentInds[ offset : offset+childCounts[i] ] = np.repeat(i, childCounts[i])
            offset += childCounts[i]

        return parentInds

    else:
        # subset
        totChildren = np.sum( childCounts[subsetInds] )

        # we also return the child index array (i.e. which children belong to the subsetInds parents)
        r = { parentInds : np.array( totChildren, dtype='uint32' ),
              childInds  : np.array( totChildren, dtype='uint32' ) }

        offsetSub = 0

        subsetMask = np.zeros( childCounts.size, dtype='int8' )
        subsetMask[subsetInds] = 1

        for i in np.arange(childCounts.size):
            if subsetMask[i] == 1 and childCounts[i] > 0:
                r['parentInds'][ offsetSub : offsetSub+childCounts[i] ] = np.repeat(i, childCounts[i])
                r['childInds'][ offsetSub : offsetSub+childCounts[i] ] = np.arange(childCounts[i]) + offset

                offsetSub += childCounts[i]
            offset += childCounts[i]

        return r

def pSplit(array, numProcs, curProc):
    """ Divide work for embarassingly parallel problems. """
    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return array # no split, return whole job load to caller

    # split array into numProcs segments, and return the curProc'th segment
    splitSize  = np.round( len(array) / numProcs )
    arraySplit = array[curProc*splitSize : (curProc+1)*splitSize ]

    # for last split, make sure it takes any leftovers
    if curProc-1 == numProcs:
        arraySplit = array[curProc*splitSize:]

    return arraySplit

def getIDIndexMapSparse(ids):
    """ Return an array which maps ID->indices within dense, disjoint subsets which are 
        allowed to be sparse in the entire ID range. within each subset i of size binsize
        array[ID-minids[i]+offset[i]] is the index of the original array ids where ID 
        is found (assumes no duplicate IDs).
    """
    raise Exception("Not implemented.")

def getIDIndexMap(ids):
    """ Return an array of size max(ids)-min(ids) such that array[ID-min(ids)] is the 
        index of the original array ids where ID is found (assumes a one to one mapping, 
        not repeated indices as in the case of parentIDs for tracers).
    """
    minid = np.min(ids)
    maxid = np.max(ids)

    dtype = 'uint32'
    if ids.size >= 2e9:
        dtype = 'uint64'

    # direct indexing approach (pretty fast)
    arr = np.zeros( maxid-minid+1, dtype=dtype )
    arr[ids-minid] = np.arange( ids.size, dtype=dtype )

    # C-style loop approach (good for sparse IDs)
    #arr = ulonarr(maxid-minid+1)
    #for i=0ULL,n_elements(ids)-1L do arr[ids[i]-minid] = i

    # looped where approach (never a good idea)
    #arr = l64indgen(maxid-minid+1)
    #for i=minid,maxid do begin
    #  w = where(ids eq i,count)
    #  if (count gt 0) then arr[i] = w[0]
    #endfor

    # reverse histogram approach (good for dense ID sampling, maybe better by factor of ~2)
    #arr = l64indgen(maxid-minid+1)
    #h = histogram(ids,rev=rev,omin=omin)
    #for i=0L,n_elements(h)-1 do if (rev[i+1] gt rev[i]) then arr[i] = rev[rev[i]:rev[i+1]-1]

    return arr, minid

# --- vis ---

def loadColorTable(ctName):
    """ Load a custom or built-in color table.
          rgb_table : do not load for active plotting, just return table as an array.
    """
    raise Exception("Not implemented.")

def sampleColorTable(ctName, num, bounds=None):
    """ Grab a sequence of colors, evenly spaced, from a given colortable. """
    raise Exception("Not implemented.")

# --- I/O ---

def curRepoVersion():
    """ Return a hash of the current state of the mercurial python repo. """
    import subprocess
    from os import getcwd, chdir

    oldCwd = getcwd()
    chdir('/n/home07/dnelson/python/')
    repoRevStr = subprocess.check_output(["hg", "id"]).strip()
    chdir(oldCwd)

    return repoRevStr