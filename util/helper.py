"""
util/helper.py
  General helper functions, small algorithms, basic I/O, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import matplotlib.pyplot as plt
import collections

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

def contiguousIntSubsets(x):
    """ Return a list of index pairs corresponding to contiguous integer subsets of the input array. """
    assert x.dtype in ['int32','int64']

    ranges = []
    inRange = False

    for i in range(x.size-1):
        # start new range?
        if not inRange:
            if x[i+1] == x[i] + 1:
                inRange = True
                rangeStart = i
        else:
            if x[i+1] == x[i] + 1:
                continue # range continues
            else:
                inRange = False
                rangeEnd = i
                ranges.append( (rangeStart,rangeEnd) )
    if inRange:
        ranges.append( (rangeStart,i) ) # final range

    return ranges

def logZeroSafe(x, zeroVal=1.0):
    """ Take log10 of input variable or array, keeping zeros at some value. """
    if not isinstance(x, (int,long,float)) and x.ndim: # array
        # another approach: if type(x).__module__ == np.__name__: print('is numpy object')
        w = np.where(x <= 0.0)
        x[w] = zeroVal
    else: # scalar
        if x <= 0.0:
            x = zeroVal

    return np.log10(x)

def logZeroMin(x):
    """ Take log10 of input variable, setting zeros to 100 times less than the minimum. """
    w = np.where(x > 0.0)
    minVal = x[w].min() if len(w[0]) > 0 else 1.0
    return logZeroSafe(x, minVal*0.01)

def logZeroNaN(x):
    """ Take log10, setting zeros to NaN silently and leaving NaN as NaN (same as the default 
    behavior, but suppress warnings). """
    r = x.copy()
    r[~np.isfinite(r)] = 0.0
    return logZeroSafe(r, np.nan)

def iterable(x):
    """ Protect against non-list/non-tuple (e.g. scalar or single string) value of x, to guarantee that 
        a for loop can iterate over this object correctly. """
    if isinstance(x, collections.Iterable) and not isinstance(x, basestring):
        return x
    else:
        return [x]

def rebin(x, shape):
    """ Resize input array x, must be 2D, to new shape, probably smaller, by averaging. """
    assert x.ndim == 2
    assert shape[0] <= x.shape[0]
    assert shape[1] <= x.shape[1]
    
    sh = shape[0],x.shape[0]//shape[0],shape[1],x.shape[1]//shape[1]
    return x.reshape(sh).mean(-1).mean(1)

def reportMemory():
    """ Return current Python process memory usage in GB. """
    import os
    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0**3 # GB

# --- general algorithms ---

def running_median(X, Y, nBins=100, binSize=None, skipZeros=False, percs=None):
    """ Create a adaptive median line of a (x,y) point set using some number of bins. """
    minVal = X.min()
    if skipZeros:
        minVal = X[X > 0.0].min()

    if binSize is not None:
        nBins = round( (X.max()-minVal) / binSize )

    bins = np.linspace(minVal,X.max(), nBins)
    delta = bins[1]-bins[0]

    running_median = []
    running_std    = []
    bin_centers    = []
    if percs is not None: running_percs = [[] for p in percs]

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= bin) & (X < binMax))

        # non-empty bin
        if len(w[0]):
            running_median.append( np.nanmedian(Y[w]) )
            running_std.append( np.std(Y[w]) )
            bin_centers.append( np.nanmedian(X[w]) )

            # compute percentiles also?
            if percs is not None:
                for j, perc in enumerate(percs):
                    running_percs[j].append( np.percentile(Y[w], perc,interpolation='linear') )

    bin_centers = np.array(bin_centers)
    running_median = np.array(running_median)
    running_std = np.array(running_std)

    if percs is not None:
        running_percs = np.array(running_percs)
        return bin_centers, running_median, running_std, running_percs

    return bin_centers, running_median, running_std

def running_sigmawindow(X, Y, windowSize=None):
    """ Create an local/adaptive estimate of the stddev of a (x,y) point set using a sliding 
    window of windowSize points. """
    assert X.size == Y.size
    if windowSize is None:
        windowSize = 3
 
    windowHalf = round(windowSize / 2.0)

    if windowHalf < 1:
        raise Exception('Window half size is too small.')

    running_std = np.zeros( X.size, dtype='float32' )

    for i in np.arange( X.size ):
        indMin = np.max( [0, i - windowHalf] )
        indMax = np.min( [i + windowHalf, X.size] )

        running_std[i] = np.std( Y[indMin:indMax] )

    return running_std

def running_histogram(X, nBins=100, binSize=None, normFac=None, skipZeros=False):
    """ Create a adaptive histogram of a (x) point set using some number of bins. """
    minVal = X.min()
    if skipZeros:
        minVal = X[X > 0.0].min()

    if binSize is not None:
        nBins = round( (X.max()-minVal) / binSize )

    bins = np.linspace(minVal,X.max(), nBins)
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

    return np.array(bin_centers), np.array(running_h)

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
    splitSize  = int(np.floor( len(array) / numProcs ))
    arraySplit = array[curProc*splitSize : (curProc+1)*splitSize]

    # for last split, make sure it takes any leftovers
    if curProc == numProcs-1:
        arraySplit = array[curProc*splitSize:]

    return arraySplit

def pSplitRange(indrange, numProcs, curProc):
    """ As pSplit(), but accept a 2-tuple of [start,end] indices and return a new range subset. """
    assert len(indrange) == 2 and indrange[1] > indrange[0]

    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return indrange

    # split array into numProcs segments, and return the curProc'th segment
    splitSize = int(np.floor( (indrange[1]-indrange[0]) / numProcs ))
    start = indrange[0] + curProc*splitSize
    end   = indrange[0] + (curProc+1)*splitSize

    # for last split, make sure it takes any leftovers
    if curProc == numProcs-1:
        end = indrange[1]

    return [start,end]

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

def trapsum(xin,yin):
    """ Trapezoidal rule numerical quadrature. """
    assert xin.size == yin.size
    assert xin.size >= 2
    nn = xin.size
    return np.sum( np.abs(xin[1:nn-1]-xin[0:nn-2]) * (yin[1:nn-1]+yin[0:nn-2])*0.5 )

# --- vis ---

def loadColorTable(ctName, valMinMax=None):
    """ Load a custom or built-in color table.
          rgb_table : do not load for active plotting, just return table as an array.
    """
    if ctName is None: return None
    
    from matplotlib.pyplot import cm
    from matplotlib.colors import LinearSegmentedColormap
    cmap = None

    # matplotlib
    if ctName in cm.cmap_d:
        cmap = cm.get_cmap(ctName)

    # cubehelix (with arbitrary parameters)
    # ...

    # custom
    if ctName == 'bluered_black0':
        # blue->red with a sharp initial start in black
        cdict = {'red'   : ((0.0,  0.0, 0.0), # x0, r_i(x0), r_f(x0)
                            (0.004, 1.0, 1.0), # x1, r_i(x1), r_f(x1)
                            (1.0,  0.0, 0.0)),
                 'green' : ((0.0,  0.0, 0.0), # xj, g_initial(xj), g_final(xj)
                            (1.0,  0.2, 0.1)),
                 'blue'  : ((0.0,  0.0, 0.0),
                            (0.004, 0.0, 0.0),
                            (1.0,  1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=512)

    if ctName == 'blgrrd_black0':
        # blue->green->red with a sharp initial start in black
        cdict = {'red'   : ((0, 0, 0), (0.01, 0.1, 0.1), (0.5, 0.1, 0.1), (1, 1, 1)),
                 'green' : ((0, 0, 0), (0.2, 0, 0), (0.5, 0.8, 0.8), (0.8, 0, 0), (1, 0, 0)),
                 'blue'  : ((0, 0, 0), (0.01, 1, 1), (0.5, 0.1, 0.1), (1, 0.1, 0.1))}
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == 'dmdens':
        # illustris dark matter density (originally from Mark)
        cdict = {'red'   : ((0.0, 0.0, 0.0), (0.3,0.0,0.0), (0.6, 0.8, 0.8), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.0, 0.0), (0.3,0.3,0.3), (0.6, 0.4, 0.4), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.05, 0.05), (0.3,0.5,0.5), (0.6, 0.6, 0.6), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == 'dmdens_tng':
        # TNG dark matter density
        #cdict = {'red'   : ((0.0, 0.0, 0.0), (0.3,0.1,0.1), (0.6, 0.76, 0.76), (1.0, 1.0, 1.0)),
        #         'green' : ((0.0, 0.0, 0.0), (0.3,0.3,0.3), (0.6, 0.53, 0.53), (1.0, 1.0, 1.0)),
        #         'blue'  : ((0.0, 0.05, 0.05), (0.3,0.5,0.5), (0.6, 0.33, 0.33), (1.0, 1.0, 1.0))}
        cdict = {'red'   : ((0.0, 0.0, 0.0), (0.15,0.1,0.1), (0.3,0.1,0.1), (0.6, 0.76, 0.76), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.0, 0.0), (0.15,0.13,0.13), (0.3,0.3,0.3), (0.6, 0.53, 0.53), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.05, 0.05), (0.15,0.26,0.26), (0.3,0.5,0.5), (0.6, 0.33, 0.33), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == 'HI_segmented':
        # discontinuous colormap for column densities, split at 10^20 and 10^19 cm^(-3)
        assert valMinMax is not None # need for placing discontinuities at correct physical locations
        valCut1 = 19.0
        valCut2 = 20.0

        fCut1 = (valCut1-valMinMax[0]) / (valMinMax[1]-valMinMax[0])
        fCut2 = (valCut2-valMinMax[0]) / (valMinMax[1]-valMinMax[0])

        color1 = np.array([114,158,206]) / 255.0 # tableau10_medium[0] (blue)
        color2 = np.array([103,191,92]) / 255.0 # tableau10_medium[2] (green)
        color3 = np.array([255,158,74]) / 255.0 # tabluea10_medium[1] (orange) or e.g. white
        cFac = 0.2 # compress start of each segment to 20% of its original intensity (e.g. towards black)

        cdict = {'red'  :  ((0.0, 0.0, 0.0), 
                            (fCut1, color1[0], color2[0]*cFac), 
                            (fCut2, color2[0], color3[0]*cFac), 
                            (1.0,   color3[0], color3[0])),
                 'green':  ((0.0, 0.0, 0.0), 
                            (fCut1, color1[1], color2[1]*cFac), 
                            (fCut2, color2[1], color3[1]*cFac), 
                            (1.0,   color3[1], color3[1])),
                 'blue' :  ((0.0, 0.0, 0.0), 
                            (fCut1, color1[2], color2[2]*cFac), 
                            (fCut2, color2[2], color3[2]*cFac), 
                            (1.0,   color3[2], color3[2])) }
        cmap = LinearSegmentedColormap(ctName, cdict, N=512)

    if cmap is None:
        raise Exception('Unrecognized colormap request ['+ctName+'] or not implemented.')

    return cmap

def sampleColorTable(ctName, num, bounds=None):
    """ Grab a sequence of colors, evenly spaced, from a given colortable. """
    from matplotlib.pyplot import cm
    cmap = cm.get_cmap(ctName)
    return cmap( np.linspace(0,1,num) )

def contourf(*args, **kwargs):
    """ Wrap matplotlib.contourf() for a graphical fix in PDF output. """
    cnt = plt.contourf(*args, **kwargs)

    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.1) # must be nonzero to fix sub-pixel AA issue

    return cnt

# --- I/O ---

def curRepoVersion():
    """ Return a hash of the current state of the mercurial python repo. """
    import subprocess
    from os import getcwd, chdir
    from os.path import expanduser

    oldCwd = getcwd()
    chdir(expanduser("~") + '/python/')
    repoRevStr = subprocess.check_output(["hg", "id"]).strip()
    chdir(oldCwd)

    return repoRevStr
