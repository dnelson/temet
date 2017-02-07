"""
util/helper.py
  General helper functions, small algorithms, basic I/O, etc.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import copy
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

def loadColorTable(ctName, valMinMax=None, plawScale=None, cmapCenterVal=None):
    """ Load a custom or built-in color table specified by ctName.
      valMinMax: required for some custom colormaps, and for some adjustments.
      plawScale: return the colormap modified as cmap_new = cmap_old**plawScale
      cmapCenterVal: return the colormap modified such that its middle point lands at 
        the numerical value cmapCenterVal, given the bounds valMinMax (e.g. zero, 
        for any symmetric colormap, say for positive/negative radial velocities)
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
        #cdict = {'red'   : ((0.0, 0.0, 0.0), (0.15,0.1,0.1), (0.3,0.1,0.1), (0.6, 0.76, 0.76), (0.9, 1.0, 1.0), (1.0, 1.0, 1.0)),
        #         'green' : ((0.0, 0.0, 0.0), (0.15,0.13,0.13), (0.3,0.3,0.3), (0.6, 0.53, 0.53), (0.9, 1.0, 1.0), (1.0, 1.0, 1.0)),
        #         'blue'  : ((0.0, 0.05, 0.05), (0.15,0.26,0.26), (0.3,0.5,0.5), (0.6, 0.33, 0.33), (0.9, 1.0, 1.0), (1.0, 1.0, 1.0))}
        # with powerlaw scaling of 1.1 added:
        cdict = {'red'   : ((0.0, 0.0, 0.0), (0.124,0.1,0.1), (0.266,0.1,0.1), (0.570, 0.76, 0.76), (0.891, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.0, 0.0), (0.124,0.13,0.13), (0.266,0.3,0.3), (0.570, 0.53, 0.53), (0.891, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.05, 0.05), (0.124,0.26,0.26), (0.266,0.5,0.5), (0.570, 0.33, 0.33), (0.891, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

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

    if ctName == 'perula':
        # matlab new default colortable: https://github.com/BIDS/colormap/blob/master/parula.py
        cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905],  [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
          [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
          [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
          [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
          [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
          [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
          [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 0.8037619048], 
          [0.0230904762, 0.6417857143, 0.7912666667],  [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476], 
          [0.0383714286, 0.6742714286, 0.743552381],  [0.0589714286, 0.6837571429, 0.7253857143], [0.0843, 0.6928333333, 0.7061666667], 
          [0.1132952381, 0.7015, 0.6858571429], [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333], 
          [0.2178285714, 0.7250428571, 0.6192619048],  [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 0.5711857143], 
          [0.3481666667, 0.7424333333, 0.5472666667],  [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857], 
          [0.4871238095, 0.7490619048, 0.4839761905],  [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 0.4493904762], 
          [0.609852381, 0.7473142857, 0.4336857143],  [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
          [0.7184095238, 0.7411333333, 0.3904761905],  [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 0.3632714286], 
          [0.8185047619, 0.7327333333, 0.3497904762],  [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
          [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857,   0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
          [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857,   0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
          [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857],  [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
          [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 0.0948380952], [0.9661, 0.9514428571, 0.0755333333], [0.9763, 0.9831, 0.0538]]

        cm_data = [[0.125, 0.143, 0.406], 
          [0.137, 0.172, 0.473], [0.12, 0.20, 0.55], [0.10, 0.26, 0.66], 
          [0.053, 0.324, 0.780], [0.0116952381, 0.3875095238, 0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
          [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
          [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
          [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
          [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 0.8037619048], 
          [0.0230904762, 0.6417857143, 0.7912666667],  [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 0.7607190476], 
          [0.0383714286, 0.6742714286, 0.743552381],  [0.0589714286, 0.6837571429, 0.7253857143], [0.0843, 0.6928333333, 0.7061666667], 
          [0.1132952381, 0.7015, 0.6858571429], [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 0.6424333333], 
          [0.2178285714, 0.7250428571, 0.6192619048],  [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 0.5711857143], 
          [0.3481666667, 0.7424333333, 0.5472666667],  [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 0.5033142857], 
          [0.4871238095, 0.7490619048, 0.4839761905],  [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 0.4493904762], 
          [0.609852381, 0.7473142857, 0.4336857143],  [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
          [0.7184095238, 0.7411333333, 0.3904761905],  [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 0.3632714286], 
          [0.8185047619, 0.7327333333, 0.3497904762],  [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
          [0.9139333333, 0.7257857143, 0.33], [0.9449571429, 0.7261142857,   0.36], [0.9738952381, 0.7313952381, 0.39], 
          [0.9937714286, 0.7454571429, 0.42], [0.9990428571, 0.7653142857,   0.45], [0.9955333333, 0.7860571429, 0.47], 
          [0.988, 0.8066, 0.50], [0.9788571429, 0.8271428571, 0.53],  [0.9697, 0.8481380952, 0.56], [0.9625857143, 0.8705142857, 0.59], 
          [0.9588714286, 0.8949, 0.62], [0.96, 0.92, 0.65], [0.98, 0.95, 0.67], [1.0, 1.0, 0.7]]

        cmap = LinearSegmentedColormap.from_list('parula', cm_data)

    if ctName == 'gasdens_tng':
        # TNG gas matter density
        cdict = {'red'   : ((0.0, 0.027, 0.027), (0.25, 0.106, 0.106), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.055, 0.055), (0.25, 0.204, 0.204), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.075, 0.075), (0.25, 0.286, 0.286), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'gasdens_tng2':
        # TNG gas matter density (blue replaced with dark orange)
        cdict = {'red'   : ((0.0, 0.075, 0.075), (0.25,0.561,0.561), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.039, 0.039), (0.25,0.235,0.235), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.020, 0.020), (0.25,0.094,0.094), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'gasdens_tng2b':
        # TNG gas matter density (blue replaced with darker orange)
        cdict = {'red'   : ((0.0, 0.075, 0.075), (0.25,0.384,0.384), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.039, 0.039), (0.25,0.161,0.161), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.020, 0.020), (0.25,0.067,0.067), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'gasdens_tng3':
        # TNG gas matter density (blue replaced with orange)
        cdict = {'red'   : ((0.0, 0.075, 0.075), (0.25,0.964,0.964), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.039, 0.039), (0.25,0.494,0.494), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.020, 0.020), (0.25,0.192,0.192), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'gasdens_tng4':
        # TNG gas matter density (low density blue is brighter)
        cdict = {'red'   : ((0.0, 0.027, 0.027), (0.25, 0.168, 0.168), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.055, 0.055), (0.25, 0.322, 0.322), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.075, 0.075), (0.25, 0.447, 0.447), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'test_dnA':
        cm_data = [[ 0.29466208, 0.00672935, 0.15437978],
                   [ 0.29730967, 0.0140011 , 0.16168024],
                   [ 0.29984988, 0.021738  , 0.16893544],
                   [ 0.30228407, 0.02994866, 0.17614398],
                   [ 0.30461426, 0.03864027, 0.18330337],
                   [ 0.30684299, 0.04734316, 0.19041051],
                   [ 0.30897339, 0.0555057 , 0.19746169],
                   [ 0.31100808, 0.06326312, 0.20445433],
                   [ 0.31294873, 0.07070323, 0.21138726],
                   [ 0.31479799, 0.07788695, 0.21825817],
                   [ 0.3165587 , 0.08485875, 0.22506482],
                   [ 0.31823387, 0.09165195, 0.23180503],
                   [ 0.31982542, 0.09829343, 0.23847817],
                   [ 0.32133617, 0.10480348, 0.24508275],
                   [ 0.32276897, 0.11119826, 0.25161752],
                   [ 0.32412654, 0.11749101, 0.25808163],
                   [ 0.32541148, 0.12369267, 0.26447459],
                   [ 0.32662676, 0.129812  , 0.27079579],
                   [ 0.32777537, 0.13585628, 0.27704485],
                   [ 0.3288596 , 0.14183217, 0.28322227],
                   [ 0.32988251, 0.14774473, 0.28932804],
                   [ 0.33084732, 0.15359812, 0.29536231],
                   [ 0.33175708, 0.15939603, 0.30132564],
                   [ 0.33261392, 0.16514225, 0.30721944],
                   [ 0.33342052, 0.17083967, 0.31304488],
                   [ 0.33418059, 0.17649012, 0.31880266],
                   [ 0.33489706, 0.18209564, 0.32449422],
                   [ 0.33557229, 0.18765838, 0.33012154],
                   [ 0.33620728, 0.19318105, 0.33568753],
                   [ 0.33680651, 0.19866394, 0.34119314],
                   [ 0.33737253, 0.20410839, 0.34664067],
                   [ 0.33790777, 0.20951567, 0.35203259],
                   [ 0.33841195, 0.21488841, 0.35737281],
                   [ 0.33888846, 0.22022695, 0.36266339],
                   [ 0.33933992, 0.22553199, 0.36790685],
                   [ 0.33976795, 0.23080465, 0.37310628],
                   [ 0.34017352, 0.23604634, 0.378265  ],
                   [ 0.34055458, 0.24125983, 0.38338755],
                   [ 0.34091519, 0.24644462, 0.38847576],
                   [ 0.34125585, 0.25160199, 0.39353293],
                   [ 0.34157678, 0.25673326, 0.39856232],
                   [ 0.34187713, 0.26184017, 0.40356746],
                   [ 0.34215355, 0.26692549, 0.40855258],
                   [ 0.34240863, 0.2719891 , 0.41351945],
                   [ 0.34264124, 0.27703258, 0.41847099],
                   [ 0.34284994, 0.28205755, 0.42341001],
                   [ 0.34303294, 0.2870657 , 0.4283391 ],
                   [ 0.34318467, 0.29206024, 0.43326188],
                   [ 0.34330501, 0.29704192, 0.43817967],
                   [ 0.34339222, 0.30201213, 0.44309408],
                   [ 0.34344334, 0.30697274, 0.44800676],
                   [ 0.34345511, 0.31192561, 0.45291909],
                   [ 0.34342402, 0.31687262, 0.45783217],
                   [ 0.34334389, 0.32181662, 0.46274758],
                   [ 0.34321126, 0.32675925, 0.46766564],
                   [ 0.34302392, 0.33170166, 0.47258589],
                   [ 0.3427776 , 0.3366457 , 0.47750821],
                   [ 0.34246789, 0.34159319, 0.48243215],
                   [ 0.34209025, 0.34654591, 0.48735696],
                   [ 0.34164002, 0.3515056 , 0.49228157],
                   [ 0.34111248, 0.35647393, 0.49720464],
                   [ 0.34050091, 0.36145315, 0.50212502],
                   [ 0.33980058, 0.36644474, 0.5070407 ],
                   [ 0.33900839, 0.37144959, 0.51194896],
                   [ 0.33811942, 0.3764691 , 0.51684731],
                   [ 0.33712872, 0.38150459, 0.52173299],
                   [ 0.33603134, 0.3865573 , 0.52660301],
                   [ 0.33482236, 0.39162839, 0.53145412],
                   [ 0.33349684, 0.3967189 , 0.53628283],
                   [ 0.33204993, 0.40182981, 0.54108544],
                   [ 0.33047678, 0.40696196, 0.54585803],
                   [ 0.32877263, 0.41211611, 0.55059646],
                   [ 0.32693281, 0.4172929 , 0.5552964 ],
                   [ 0.32495274, 0.42249286, 0.55995334],
                   [ 0.32282797, 0.42771641, 0.5645626 ],
                   [ 0.32055417, 0.43296384, 0.56911933],
                   [ 0.31812721, 0.43823531, 0.57361853],
                   [ 0.31554315, 0.44353087, 0.57805509],
                   [ 0.31279826, 0.44885043, 0.58242377],
                   [ 0.30988908, 0.45419377, 0.58671925],
                   [ 0.30681246, 0.45956051, 0.59093613],
                   [ 0.30356557, 0.46495017, 0.59506899],
                   [ 0.30014595, 0.4703621 , 0.59911237],
                   [ 0.29655157, 0.47579551, 0.60306083],
                   [ 0.29278086, 0.48124949, 0.60690898],
                   [ 0.28883279, 0.48672295, 0.61065152],
                   [ 0.28470686, 0.4922147 , 0.61428325],
                   [ 0.28040182, 0.49772369, 0.61779905],
                   [ 0.27591604, 0.50324893, 0.62119386],
                   [ 0.27125494, 0.50878801, 0.62446323],
                   [ 0.26642099, 0.51433921, 0.62760283],
                   [ 0.26141755, 0.5199007 , 0.63060867],
                   [ 0.25624892, 0.52547057, 0.63347711],
                   [ 0.25092041, 0.53104681, 0.63620489],
                   [ 0.24543843, 0.53662738, 0.63878919],
                   [ 0.23980655, 0.54221083, 0.64122701],
                   [ 0.23403759, 0.54779434, 0.64351699],
                   [ 0.2281426 , 0.55337561, 0.6456579 ],
                   [ 0.22213336, 0.5589525 , 0.6476489 ],
                   [ 0.21602319, 0.56452292, 0.64948964],
                   [ 0.20982717, 0.5700848 , 0.6511803 ],
                   [ 0.20356113, 0.5756363 , 0.65272131],
                   [ 0.19724633, 0.58117519, 0.65411429],
                   [ 0.1909043 , 0.58669965, 0.65536092],
                   [ 0.18455896, 0.59220797, 0.65646327],
                   [ 0.17823707, 0.59769859, 0.65742379],
                   [ 0.1719692 , 0.60316997, 0.65824547],
                   [ 0.16579061, 0.60862062, 0.65893196],
                   [ 0.15973697, 0.61404954, 0.65948625],
                   [ 0.15384948, 0.61945575, 0.65991193],
                   [ 0.14817364, 0.62483839, 0.66021274],
                   [ 0.14275943, 0.63019674, 0.66039253],
                   [ 0.13766277, 0.63553007, 0.66045574],
                   [ 0.13294449, 0.64083766, 0.66040727],
                   [ 0.12866207, 0.6461196 , 0.66024989],
                   [ 0.12487837, 0.65137569, 0.65998747],
                   [ 0.12165607, 0.65660583, 0.65962383],
                   [ 0.11905514, 0.66181003, 0.65916267],
                   [ 0.11712987, 0.66698837, 0.65860756],
                   [ 0.11592626, 0.67214096, 0.65796216],
                   [ 0.11548149, 0.67726769, 0.65723154],
                   [ 0.11581074, 0.68236926, 0.65641681],
                   [ 0.1169185 , 0.68744602, 0.65552086],
                   [ 0.11879407, 0.69249831, 0.65454636],
                   [ 0.12141287, 0.69752653, 0.65349584],
                   [ 0.12473876, 0.70253109, 0.65237159],
                   [ 0.12872687, 0.70751239, 0.65117575],
                   [ 0.13332681, 0.71247086, 0.64991025],
                   [ 0.13848551, 0.7174069 , 0.64857686],
                   [ 0.14414977, 0.72232092, 0.64717714],
                   [ 0.1502682 , 0.72721327, 0.64571276],
                   [ 0.1567925 , 0.73208426, 0.64418574],
                   [ 0.1636778 , 0.73693434, 0.64259604],
                   [ 0.17088403, 0.74176383, 0.64094464],
                   [ 0.17837554, 0.746573  , 0.63923238],
                   [ 0.18612096, 0.75136209, 0.63745999],
                   [ 0.19409296, 0.75613129, 0.63562808],
                   [ 0.20226789, 0.76088077, 0.63373721],
                   [ 0.21062546, 0.76561065, 0.63178782],
                   [ 0.21914836, 0.77032099, 0.62978031],
                   [ 0.22782189, 0.77501182, 0.627715  ],
                   [ 0.23663365, 0.77968313, 0.62559221],
                   [ 0.24557326, 0.78433483, 0.62341218],
                   [ 0.25463207, 0.7889668 , 0.62117517],
                   [ 0.26380294, 0.79357889, 0.61888143],
                   [ 0.27308004, 0.79817085, 0.6165312 ],
                   [ 0.28245865, 0.80274243, 0.61412476],
                   [ 0.29193506, 0.80729329, 0.61166242],
                   [ 0.30150637, 0.81182305, 0.60914455],
                   [ 0.31117041, 0.81633128, 0.60657158],
                   [ 0.32092565, 0.82081749, 0.60394402],
                   [ 0.33077109, 0.82528112, 0.60126247],
                   [ 0.34070621, 0.82972159, 0.59852766],
                   [ 0.35073088, 0.83413823, 0.59574047],
                   [ 0.36084534, 0.83853033, 0.59290189],
                   [ 0.37105013, 0.84289711, 0.59001315],
                   [ 0.38134605, 0.84723772, 0.58707563],
                   [ 0.39173412, 0.85155129, 0.58409097],
                   [ 0.40221559, 0.85583683, 0.58106109],
                   [ 0.41279188, 0.86009333, 0.57798814],
                   [ 0.42346574, 0.86431958, 0.57487318],
                   [ 0.43423802, 0.86851449, 0.57172021],
                   [ 0.44511055, 0.8726768 , 0.56853254],
                   [ 0.45608527, 0.87680519, 0.56531394],
                   [ 0.46716417, 0.88089826, 0.56206876],
                   [ 0.47834931, 0.88495455, 0.55880197],
                   [ 0.48964276, 0.88897248, 0.55551925],
                   [ 0.50104661, 0.89295043, 0.55222711],
                   [ 0.5125629 , 0.89688668, 0.54893297],
                   [ 0.52419362, 0.90077942, 0.54564531],
                   [ 0.53594066, 0.90462678, 0.54237384],
                   [ 0.54780892, 0.90842633, 0.53912641],
                   [ 0.55980324, 0.91217548, 0.53591253],
                   [ 0.57192009, 0.91587279, 0.53275172],
                   [ 0.58416045, 0.91951606, 0.52966042],
                   [ 0.59652485, 0.92310307, 0.52665741],
                   [ 0.60901317, 0.92663162, 0.52376406],
                   [ 0.62163739, 0.93009711, 0.52099363],
                   [ 0.63438922, 0.93349841, 0.51838028],
                   [ 0.64726118, 0.93683432, 0.5159595 ],
                   [ 0.66024868, 0.94010307, 0.51376686],
                   [ 0.67336538, 0.94329888, 0.51182786],
                   [ 0.68658726, 0.94642378, 0.51019993],
                   [ 0.69989887, 0.94947826, 0.50893557],
                   [ 0.71330517, 0.95245817, 0.50807772],
                   [ 0.72677221, 0.95536849, 0.50769586],
                   [ 0.74027498, 0.95821272, 0.50785354],
                   [ 0.75379702, 0.96099257, 0.50861057],
                   [ 0.76728662, 0.96371966, 0.51003893],
                   [ 0.78071329, 0.96640105, 0.51219577],
                   [ 0.79402528, 0.96905086, 0.51513573],
                   [ 0.80717422, 0.97168382, 0.51889853],
                   [ 0.8201124 , 0.97431615, 0.52350733],
                   [ 0.83279768, 0.97696396, 0.52896682],
                   [ 0.84517833, 0.97964827, 0.53525864],
                   [ 0.85724398, 0.98237785, 0.54235406],
                   [ 0.86895555, 0.98517241, 0.55019815],
                   [ 0.88030569, 0.98804171, 0.55873107],
                   [ 0.89130564, 0.99098892, 0.56789694],
                   [ 0.90195877, 0.9940201 , 0.57763056],
                   [ 0.91225818, 0.99714584, 0.58784834]]

        cmap = LinearSegmentedColormap.from_list('test_dnA', cm_data)

    if cmap is None:
        raise Exception('Unrecognized colormap request ['+ctName+'] or not implemented.')

    def _plawScale(cmap, plaw_index):
        assert plaw_index > 0.0
        cdict = {}

        for k in ['red','green','blue']:
            cdict[k] = []
            for j in range(len(cmap._segmentdata[k])):
                xx = cmap._segmentdata[k][j]
                cdict[k].append( [xx[0]**plaw_index, xx[1], xx[2]] )
            assert (cdict[k][0] < 0 or cdict[k][-1] > 1) # outside [0,1]
        return LinearSegmentedColormap(ctName+'_p', cdict, N=1024)

    if plawScale is not None:
        cmap = _plawScale(cmap, plawScale)

    if cmapCenterVal is not None:
        assert cmapCenterVal > valMinMax[0] and cmapCenterVal < valMinMax[1]
        center_rel = np.abs(cmapCenterVal - valMinMax[0]) / np.abs(valMinMax[1] - valMinMax[0])
        plaw_index = np.log(center_rel) / np.log(0.5)
        cmap = _plawScale(cmap, plaw_index)

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
