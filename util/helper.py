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
from scipy.optimize import leastsq, least_squares, curve_fit
from scipy.stats import binned_statistic
from numba import jit

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
    ind_nd = np.unravel_index( ind, array.shape )
    return array[ind_nd], ind

def array_equal_nan(a, b):
    """ As np.array_equal(a,b) but allowing NaN==NaN. """
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()

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
    if np.isfinite(zeroVal):
        pass
        #print(' logZeroSafe: This was always ill-advised, migrate towards deleting this function.')
    if not isinstance(x, (int,float)) and x.ndim: # array
        # another approach: if type(x).__module__ == np.__name__: print('is numpy object')
        with np.errstate(invalid='ignore'):
            w = np.where(x <= 0.0)
        x[w] = zeroVal
    else: # scalar
        if x <= 0.0:
            x = zeroVal

    return np.log10(x)

def logZeroMin(x):
    """ Take log10 of input variable, setting zeros to 100 times less than the minimum. """
    if isinstance(x,np.number) and not isinstance(x,np.ndarray) and x.size == 1:
        x = np.array([x])

    with np.errstate(invalid='ignore'):
        w = np.where(x > 0.0)

    minVal = x[w].min() if len(w[0]) > 0 else 1.0
    return logZeroSafe(x, minVal*0.01)

def logZeroNaN(x):
    """ Take log10, setting zeros to NaN silently and leaving NaN as NaN (same as the default 
    behavior, but suppress warnings). """
    r = x.copy()
    r[~np.isfinite(r)] = 0.0
    return logZeroSafe(r, np.nan)

def last_nonzero(array, axis, invalid_val=-1):
    """ Return the indices of the last nonzero entries of the array, along the given axis. """
    mask = (array != 0)

    val = array.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def iterable(x):
    """ Protect against non-list/non-tuple (e.g. scalar or single string) value of x, to guarantee that 
        a for loop can iterate over this object correctly. """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return np.reshape(x, 1) # scalar to 1d array of 1 element
    elif isinstance(x, collections.Iterable) and not isinstance(x, str):
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

def numPartToChunkLoadSize(numPart):
    """ For a given snapshot size, in terms of total particle count, decide on a good 
    chunk loading size for a reasonable compute/memory balance. """
    nChunks = np.max( [4, int(numPart**(1.0/3.0) / 10.0)] )
    return nChunks

def tail(fileName, nLines):
    """ Wrap linux tail command line utility. """
    import subprocess
    lines = subprocess.check_output( ['tail', '-n', str(nLines), fileName] )
    if isinstance(lines,bytes):
        lines = lines.decode('utf-8')
    return lines

# --- general algorithms ---

def running_median(X, Y, nBins=100, binSize=None, binSizeLg=None, skipZeros=False, percs=None, minNumPerBin=10, mean=False, weights=None):
    """ Create a adaptive median line of a (x,y) point set using some number of bins. """
    assert X.shape == Y.shape
    if weights is not None:
        assert mean
        assert weights.size == Y.size

    minVal = np.nanmin(X)
    maxVal = np.nanmax(X)
    if skipZeros:
        minVal = np.nanmin( X[X != 0.0] )

    if np.isnan(minVal):
        print('Bad inputs, going to fail in linspace.')

    if binSize is not None:
        nBins = round( (maxVal-minVal) / binSize )

    if nBins <= 0: nBins = 1
    bins = np.linspace(minVal, maxVal, nBins)

    if binSizeLg is not None:
        # small bins for low x values (e.g. halo mass)
        splitX = np.nanpercentile(X,90) # rough heuristic
        nBins0 = int( (splitX-minVal) / binSize )
        nBins1 = int( (maxVal-splitX) / binSizeLg )
        bins0 = np.linspace(minVal, splitX, nBins0)
        bins1 = np.linspace(splitX, maxVal, nBins1)[1:]
        bins = np.hstack( (bins0,bins1) )

    running_median = []
    running_std    = []
    bin_centers    = []
    if percs is not None: running_percs = [[] for p in percs]

    binLeft = bins[0]

    for i in range(bins.size):
        binMax = bins[i+1] if i+1 < bins.size else np.inf

        with np.errstate(invalid='ignore'): # expect X may contain NaN which should not be included
            w = np.where((X >= binLeft) & (X < binMax))

        # non-empty bin, or last bin with at least minNumPerBin/2 elements
        if len(w[0]) >= minNumPerBin or (i == len(bins)-1 and len(w[0]) >= minNumPerBin/2):

            if np.isnan(Y[w]).all():
                continue

            binLeft = binMax
            if mean:
                if weights is None:
                    running_median.append( np.nanmean(Y[w]) )
                else:
                    loc_value = np.nansum(Y[w]*weights[w]) / np.nansum(weights[w])
                    running_median.append( loc_value )
            else:
                running_median.append( np.nanmedian(Y[w]) )
            running_std.append( np.nanstd(Y[w]) )
            bin_centers.append( np.nanmedian(X[w]) )

            # compute percentiles also?
            if percs is not None:
                for j, perc in enumerate(percs):
                    running_percs[j].append( np.nanpercentile(Y[w], perc,interpolation='linear') )

    bin_centers = np.array(bin_centers)
    running_median = np.array(running_median)
    running_std = np.array(running_std)

    if percs is not None:
        running_percs = np.array(running_percs)
        return bin_centers, running_median, running_std, running_percs

    return bin_centers, running_median, running_std

def running_median_clipped(X, Y_in, nBins=100, minVal=None, maxVal=None, binSize=None, skipZerosX=False, skipZerosY=False, clipPercs=[10,90]):
    """ Create a constant-bin median line of a (x,y) point set, clipping outliers above/below clipPercs. """
    if minVal is None:
        if skipZerosX:
            minVal = np.nanmin( X[X != 0.0] )
        else:
            minVal = np.nanmin(X)
    if maxVal is None:
        maxVal = np.nanmax(X)

    Y = Y_in
    if skipZerosY:
        # filter out
        Y = Y_in.copy()
        w = np.where(Y == 0.0)
        Y[w] = np.nan

    if binSize is not None:
        nBins = round( (maxVal-minVal) / binSize ) + 1

    if nBins <= 0: nBins = 1
    bins = np.linspace(minVal, maxVal, nBins)
    delta = bins[1]-bins[0] if nBins >= 2 else np.inf

    running_median = np.zeros(nBins, dtype='float32')
    bin_centers    = np.zeros(nBins, dtype='float32')

    running_median.fill(np.nan)
    bin_centers.fill(np.nan)

    with np.errstate(invalid='ignore'): # both X and Y contain nan, silence np.where() warnings
        for i, bin in enumerate(bins):
            binMin = bin
            binMax = bin + delta
            w = np.where((X >= binMin) & (X < binMax))

            if len(w[0]) == 0:
                continue

            # compute percentiles
            percs = np.nanpercentile(Y[w], clipPercs,interpolation='linear')

            # filter
            w_unclipped = np.where( (Y[w] >= percs[0]) & (Y[w] <= percs[1]) )

            if len(w_unclipped[0]) == 0:
                continue

            # compute median on points inside sigma-clipped region only
            running_median[i] = np.nanmedian(Y[w][w_unclipped])
            bin_centers[i] = np.nanmedian(X[w][w_unclipped])

    return bin_centers, running_median, bins

def running_median_sub(X, Y, S, nBins=100, binSize=None, skipZeros=False, sPercs=[25,50,75], 
                       percs=[16,84], minNumPerBin=10):
    """ Create a adaptive median line of a (x,y) point set using some number of bins, where in each 
    bin only the sub-sample of points obtained by slicing a third value (S) above and/or below one or 
    more percentile thresholds is used. """
    minVal = np.nanmin(X)
    if skipZeros:
        minVal = np.nanmin( X[X != 0.0] )

    if binSize is not None:
        nBins = round( (np.nanmax(X)-minVal) / binSize )

    if nBins <= 0: nBins = 1
    bins = np.linspace(minVal,np.nanmax(X), nBins)
    delta = bins[1]-bins[0] if nBins >= 2 else np.inf

    bin_centers     = []
    running_medianA = [[] for p in sPercs]
    running_medianB = [[] for p in sPercs]
    running_percsA  = [[] for p in sPercs]
    running_percsB  = [[] for p in sPercs]

    binLeft = bins[0]

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= binLeft) & (X < binMax))

        # non-empty bin
        #if len(w[0]):
        # non-empty bin, or last bin with at least minNumPerBin/2 elements
        if len(w[0]) >= minNumPerBin or (i == len(bins)-1 and len(w[0]) >= minNumPerBin/2):
            # slice third quantity
            slice_perc_vals = np.nanpercentile(S[w], sPercs, interpolation='linear')

            bin_centers.append( np.nanmedian(X[w]) )

            binLeft = binMax

            for i, sPerc in enumerate(sPercs):
                # which points in this bin are above/below threshold percentile (e.g. median)?
                with np.errstate(invalid='ignore'):
                    w_sliceA = np.where( S[w] > slice_perc_vals[i] )
                    w_sliceB = np.where( S[w] <= slice_perc_vals[i] )

                running_medianA[i].append( np.nanmedian(Y[w][w_sliceA]) )
                running_medianB[i].append( np.nanmedian(Y[w][w_sliceB]) )
                
                # compute percentiles also
                running_percsA[i] = np.nanpercentile(Y[w][w_sliceA], percs, interpolation='linear')
                running_percsB[i] = np.nanpercentile(Y[w][w_sliceB], percs, interpolation='linear')

    bin_centers = np.array(bin_centers)
    running_medianA = np.array(running_medianA)
    running_medianB = np.array(running_medianB)
    running_percsA = np.array(running_percsA)
    running_percsB = np.array(running_percsB)

    return bin_centers, running_medianA, running_medianB, running_percsA, running_percsB

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
    minVal = np.nanmin(X)
    if skipZeros:
        minVal = np.nanmin( X[X != 0.0] )

    if binSize is not None:
        nBins = round( (np.nanmax(X)-minVal) / binSize )

    bins = np.linspace(minVal,np.nanmax(X), nBins)
    delta = bins[1]-bins[0]

    running_h   = []
    bin_centers = []

    for i, bin in enumerate(bins):
        binMax = bin + delta
        with np.errstate(invalid='ignore'):
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

def pSplitRange(indrange, numProcs, curProc, inclusive=False):
    """ As pSplit(), but accept a 2-tuple of [start,end] indices and return a new range subset. 
    If inclusive==True, then assume the range subset will be used e.g. as input to snapshotSubseet(), 
    which unlike numpy convention is inclusive in the indices."""
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

    if inclusive and curProc < numProcs-1:
        # not for last split/final index, because this should be e.g. NumPart[0]-1 already
        end -= 1

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

def leastsq_fit(func, params_init, args=None):
    """ Wrap scipy.optimize.leastsq() by making the error function and handling returns.
    If args is not None, then the standard errors are also computed, but this ASSUMES that 
    args[0] is the x data points and args[1] is the y data points. """
    def error_function(params, x, y, fixed=None):
        y_fit = func(x, params, fixed)
        return y_fit - y

    params_best, params_cov, info, errmsg, retcode = \
      leastsq(error_function, params_init, args=args, full_output=True)

    # estimate errors (unused)
    params_stddev = np.zeros( len(params_best), dtype='float32' )

    if params_cov is not None and args is not None:
        # assume first two elements of args are x_data and y_data
        assert len(args) >= 2 and args[0].shape == args[1].shape
        x_data, y_data = args[0], args[1]

        # reduced chi^2 (i.e. residual variance)
        chi2 = np.sum(error_function(params_best, x_data, y_data)**2.0)
        reduced_chi2 = chi2 / (y_data.size - len(params_best))

        # incorporate into fractional covariance matrix
        params_cov *= reduced_chi2

        # square root of diagonal elements estimates stddev of each parameter
        for j in range(len(params_best)):
            params_stddev[j] = np.abs(np.sqrt( params_cov[j][j] ))

    return params_best, params_stddev

def least_squares_fit(func, params_init, params_bounds, args=None):
    """ Wrap scipy.optimize.least_squares() using the Trust Region Reflective algorithm for 
    fitting with (optional) constraints on each parameter. """
    def error_function(params, x, y, fixed=None):
        y_fit = func(x, params, fixed)
        return y_fit - y

    # e.g. two parameters, require that x[1] >= 1.5, and x[0] left unconstrained
    # (lower bounds, upper bounds)
    # where each is a scalar or an array of the same size as parameters
    #bounds2 = ([-np.inf, 1.5], np.inf)

    result = least_squares(error_function, params_init, bounds=params_bounds, args=args, method='trf')

    return result.x

def reducedChiSq(sim_x, sim_y, data_x, data_y, data_yerr=None, data_yerr_up=None, data_yerr_down=None):
    """ Helper, computed reduced (i.e.) mean chi squared between a simulation 'line' and observed 
    relation given by a set of points with errorbars. """
    from scipy.interpolate import interp1d
    assert data_yerr is None or (data_yerr_up is None and data_yerr_down is None)
    assert np.sum([data_yerr_up is None,data_yerr_down is None]) in [0,2] # both or neither

    sim_f = interp1d(sim_x,sim_y,kind='linear')
    sim_y_at_data_x = sim_f(data_x)

    if data_yerr is not None:
        data_error = data_yerr
    else:
        data_error = data_yerr_down
        w = np.where(sim_y_at_data_x > data_y)
        data_error[w] = data_yerr_up[w]

    # weighted squared deviations
    devs = (sim_y_at_data_x - data_y)**2 / data_error**2

    chi2 = np.sum(devs)
    chi2v = chi2 / data_x.size
    
    return chi2v

def sgolay2d(z, window_size, order, derivative=None):
    """ Szalay-golay filter in 2D using FFT convolutions. """
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0
   
    if window_size % 2 == 0:
        raise ValueError('window_size must be odd')
   
    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')
    half_size = window_size // 2
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]
   
    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])
       
    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z
   
    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )
   
    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )
   
    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')       
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')       
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return sfftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid')

def binned_stat_2d(x, y, c, bins, range_x, range_y, stat='median'):
    """ Replacement of binned_statistic_2d for mean_nan or median_nan. """
    assert stat in ['mean','median']

    nbins = bins[0] * bins[1]
    binsize_x = (range_x[1] - range_x[0]) / bins[0]
    binsize_y = (range_y[1] - range_y[0]) / bins[1]

    # finite only
    w = np.where( np.isfinite(x) & np.isfinite(y) & np.isfinite(c) )
    x = x[w]
    y = y[w]
    c = c[w]

    # in bounds only
    w = np.where( (x>=range_x[0]) & (x<range_x[1]) & (y>=range_y[0]) & (y<range_y[1]) )

    # bin
    inds_x = np.floor( (x[w] - range_x[0]) / binsize_x ).astype('int32')
    inds_y = np.floor( (y[w] - range_y[0]) / binsize_y ).astype('int32')
    c = c[w]

    ind_1d = np.ravel_multi_index([inds_x,inds_y], bins)

    count = np.bincount(ind_1d, minlength=nbins).astype('float64')

    if stat == 'mean':
        # statistic: mean
        total = np.bincount(ind_1d, c, minlength=nbins)
        
        # only non-zero bins
        w = np.where(count > 0)
        
        mean = np.zeros( nbins, dtype=c.dtype )
        mean.fill(np.nan)
        mean[w] = total[w] / count[w]

        result = np.reshape( mean, bins )

    if stat == 'median':
        # statistic: median
        sort_inds = np.argsort(ind_1d)
        ind_1d_sorted = ind_1d[sort_inds]
        c_sorted = c[sort_inds]

        # bin edges and mid points
        bin_edges_inds = (ind_1d_sorted[1:] != ind_1d_sorted[:-1]).nonzero()[0] + 1
        target_inds = ind_1d_sorted[bin_edges_inds-1]

        # allocate
        result = np.zeros( nbins, dtype=c.dtype )
        result.fill(np.nan)

        if 0:
            # handle odd/even definition for np.median
            result[w] = (c_sorted[med_inds[w_even]] + c_sorted[med_inds[w_even]-1]) * 0.5
            result[w] = c_sorted[med_inds[w_odd]]

        # fastest method (unfinished!)
        if 0:
            med_inds = ( np.r_[0, bin_edges_inds] + np.r_[bin_edges_inds, len(w)] ) * 0.5
            med_inds = med_inds.astype('int32')
            #sort_inds_c = np.argsort(c_sorted) # need to rearrange vals
            # med_inds must sample the s-sorted index list for each bin
            result[target_inds] = c_sorted[med_inds]

        # clearer method (effectively a sort on each bin subset)
        result[target_inds] = [np.median(i) for i in np.split(c_sorted, bin_edges_inds[:-1])]
        
        if 0: # debug
            for bin_ind in range(10):
                ww = np.where(ind_1d == bin_ind)
                print(bin_ind, result[bin_ind], np.median(c[ww]), result[bin_ind]/np.median(c[ww]))

        result = np.reshape( result, bins )

    return result, np.reshape(count,bins)

def binned_statistic_weighted(x, values, statistic, bins, weights=None, weights_w=None):
    """ If weights == None, straight passthrough to scipy.stats.binned_statistic(). Otherwise, 
    compute once for values*weights, again for weights alone, then normalize and return. 
    If weights_w is not None, apply this np.where() result to the weights array. """
    if weights is None:
        return binned_statistic(x, values, statistic=statistic, bins=bins)

    weights_loc = weights[weights_w] if weights_w is not None else weights

    if statistic == 'mean':
        # weighted mean (nan for bins where wt_sum == 0)
        valwt_sum, bin_edges, bin_number = binned_statistic(x, values*weights_loc, statistic='sum', bins=bins)
        wt_sum, _, _ = binned_statistic(x, weights_loc, statistic='sum', bins=bins)

        return (valwt_sum/wt_sum), bin_edges, bin_number

    if statistic == np.std:
        # weighted standard deviation (note: numba accelerated)
        std = weighted_std_binned(x, values, weights, bins)
        return std, None, None

# --- numba accelerated ---

@jit(nopython=True, nogil=True, cache=True)
def weighted_std_binned(x, vals, weights, bins):
    """ For a given set of bins (edges), histogram x into those bins, and then compute and 
    return the standard deviation (unbiased) of vals weighted by weights, per-bin. Assumes 
    'reliability' (non-random) weights, following 
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance. """

    # histogram, bins[i] < x < bins[i+1], same convention as np.histogram()
    bin_inds = np.digitize(x, bins) - 1 
    
    # protect against x.min() < bins[0]
    if np.min(bin_inds) < 0:
        for i in range(bin_inds.size):
            if bin_inds[i] < 0:
                bin_inds[i] = 0

    # sort values and weights by bin
    bin_counts = np.bincount(bin_inds)

    sort_inds = np.argsort(bin_inds)

    vals_sorted = vals[sort_inds]
    weights_sorted = weights[sort_inds]

    # allocate
    std = np.zeros( bins.size-1, dtype=np.float32 )

    # loop over bins
    offset = 0
    for i in range(bin_counts.size):
        if bin_counts[i] <= 1:
            std[i] = np.nan
            continue

        # get values and weights in this bin
        end_i = offset + bin_counts[i]

        loc_vals = vals_sorted[offset : end_i]
        loc_wts  = weights_sorted[offset : end_i]

        offset += bin_counts[i]

        # sum weights
        wt_sum = np.sum(loc_wts)
        if wt_sum == 0.0:
            std[i] = np.nan
            continue

        num_nonzero = 0
        for j in range(bin_counts[i]):
            if loc_wts[j] > 0.0:
                num_nonzero += 1

        if num_nonzero <= 1:
            std[i] = np.nan
            continue

        # calculate weighted std
        valwt_sum = np.sum(loc_vals*loc_wts)

        weighted_mean = valwt_sum / wt_sum
        diff_sq = (loc_vals - weighted_mean)**2.0

        stdwt_sum = np.sum(diff_sq*loc_wts)

        if stdwt_sum == 0.0:
            std[i] = np.nan
            continue

        wt2_sum = np.sum(loc_wts*loc_wts)
        normalization = (wt_sum - wt2_sum/wt_sum)

        if normalization == 0.0:
            # protect against wt_sum == wt2_sum (i.e. N=1)
            normalization = wt_sum

        variance = stdwt_sum / normalization

        std[i] = np.sqrt(variance)
        
    return std

@jit(nopython=True, nogil=True, cache=True)
def bincount(x, dtype):
    """ Same behavior as np.bincount() except can specify dtype different than x.dtype to save memory. """
    c = np.zeros(np.max(x)+1, dtype=dtype)

    for i in range(x.size):
        c[x[i]] += 1

    return c

@jit(nopython=True, nogil=True, cache=True)
def periodicDistsN(pos1, pos2, BoxSize, squared=False):
    """ Compute periodic distance between each (x,y,z) coordinate in pos1 vs. the 
    corresponding (x,y,z) point in pos2. Either pos1 and pos2 have the same shapes, 
    and are matched pairwise, or pos1 is a tuple (i.e. reference position). """
    BoxHalf = BoxSize * 0.5

    dists = np.zeros( pos2.shape[0], dtype=pos2.dtype )
    assert pos1.shape[0] == pos2.shape[0] or pos1.size == 3
    assert pos2.shape[1] == 3
    if pos1.ndim == 2:
        assert pos1.shape[1] == 3

    for i in range(pos2.shape[0]):
        for j in range(3):
            if pos1.ndim == 1:
                xx = pos1[j] - pos2[i,j]
            else:
                xx = pos1[i,j] - pos2[i,j]

            if xx > BoxHalf:
                xx -= BoxSize
            if xx < -BoxHalf:
                xx += BoxSize

            dists[i] += xx * xx

    if not squared:
        for i in range(pos2.shape[0]):
            dists[i] = np.sqrt( dists[i] )

    return dists

@jit(nopython=True, nogil=True, cache=True)
def periodicDistsIndexed(pos1, pos2, indices, BoxSize):
    """ Compute periodic distance between each (x,y,z) coordinate in pos1 vs. the 
    corresponding (x,y,z) point in pos2. Here pos1.shape[0] != pos2.shape[0], 
    e.g. in the case that pos1 are group centers, and pos2 are particle positions.
    Then indices gives, for each pos2, the index of the corresponding pos1 element 
    to compute the distance to, e.g. the group ID of each particle. Return size is 
    the length of pos2. """
    BoxHalf = BoxSize * 0.5

    dists = np.zeros( pos2.shape[0], dtype=pos1.dtype )
    assert pos1.shape[0] != pos2.shape[0] # not generically expected
    assert pos1.shape[1] == 3 and pos2.shape[1] == 3
    assert indices.ndim == 1 and indices.size == pos2.shape[0]

    for i in range(pos2.shape[0]):
        pos1_loc = pos1[indices[i],:]
        for j in range(3):
            xx = pos1_loc[j] - pos2[i,j]

            if xx > BoxHalf:
                xx -= BoxSize
            if xx < -BoxHalf:
                xx += BoxSize

            dists[i] += xx * xx
        dists[i] = np.sqrt( dists[i] )

    return dists

# --- vis ---

def getWhiteBlackColors(pStyle):
    """ Plot style helper. """
    assert pStyle in ['white','black']

    if pStyle == 'white':
        color1 = 'white' # background
        color2 = 'black' # axes etc
        color3 = '#777777' # color bins with only NaNs
        color4 = '#cccccc' # color bins with value 0.0
    if pStyle == 'black':
        color1 = 'black'
        color2 = 'white'
        color3 = '#333333'
        color4 = '#222222'

    return color1, color2, color3, color4

def validColorTableNames():
    """ Return a list of whitelisted colormap names. """
    from matplotlib.pyplot import cm

    names1 = list(cm.cmap_d.keys()) # matplotlib
    names2 = [n.replace('.cmo','') for n in cm.cmap_d] # cmocean
    names3 = ['dmdens','dmdens_tng','HI_segmented','perula'] # custom

    return names1 + names2 + names3

def loadColorTable(ctName, valMinMax=None, plawScale=None, cmapCenterVal=None, fracSubset=None, numColors=None):
    """ Load a custom or built-in color table specified by ctName. Note that appending '_r' to most default colormap names 
    requests the colormap in reverse order (e.g. changing light->dark to dark->light).
      valMinMax: required for some custom colormaps, and for some adjustments.
      plawScale: return the colormap modified as cmap_new = cmap_old**plawScale
      cmapCenterVal: return the colormap modified such that its middle point lands at 
        the numerical value cmapCenterVal, given the bounds valMinMax (e.g. zero, 
        for any symmetric colormap, say for positive/negative radial velocities)
      fracSubset: a 2-tuple in [0,1] e.g. [0.2,0.8] to use only part of the original colormap range
      numColors: if not None, integer number of discrete colors of the desired colortable (matplotlib colormaps only)
    """
    if ctName is None: return None

    from matplotlib.pyplot import cm
    from matplotlib.colors import LinearSegmentedColormap
    import cmocean
    cmap = None

    # matplotlib
    if ctName in cm.cmap_d:
        cmap = cm.get_cmap(ctName, lut=numColors)

    # cmocean
    if 'cmo.%s' % ctName in cm.cmap_d:
        cmap = cm.get_cmap('cmo.%s' % ctName, lut=numColors)

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

    if ctName == 'BdRd_r_black':
        # brewer blue->red diverging, with central white replaced with black (psychodelic)
        data = ((0.40392156862745099,  0.0                ,  0.12156862745098039),
                (0.69803921568627447,  0.09411764705882353,  0.16862745098039217),
                (0.83921568627450982,  0.37647058823529411,  0.30196078431372547),
                (0.95686274509803926,  0.6470588235294118 ,  0.50980392156862742),
                (0.99215686274509807,  0.85882352941176465,  0.7803921568627451 ),
                (0.96862745098039216,  0.96862745098039216,  0.96862745098039216),
                (0.81960784313725488,  0.89803921568627454,  0.94117647058823528),
                (0.5725490196078431 ,  0.77254901960784317,  0.87058823529411766),
                (0.2627450980392157 ,  0.57647058823529407,  0.76470588235294112),
                (0.12941176470588237,  0.4                ,  0.67450980392156867),
                (0.0196078431372549 ,  0.18823529411764706,  0.38039215686274508)
                )

        cdict = {'red':[], 'green':[], 'blue':[]}
        for i, rgb in enumerate(data):
            new_r = 1.0 - rgb[0]
            new_g = 1.0 - rgb[1]
            new_b = 1.0 - rgb[2]
            frac = float(i) / (len(data)-1)
            cdict['red'].append( (frac,new_r,new_r) )
            cdict['green'].append( (frac,new_g,new_g) )
            cdict['blue'].append( (frac,new_b,new_b) )
        cmap = LinearSegmentedColormap(ctName, cdict)

    if ctName == 'BdRd_r_black2':
        # brewer blue->red diverging, with central white replaced with black (try #2)
        cdict = {'red'   : ((0.0, 0.043, 0.043), (0.5, 0.0, 0.0), (1, 0.8, 0.8)),
                 'green' : ((0, 0.396, 0.396), (0.5, 0.0, 0.0), (1, 0, 0)),
                 'blue'  : ((0.0, 0.8, 0.8), (0.5, 0.0, 0.0), (1, 0.2353, 0.2352))}
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

    if ctName in ['HI_segmented','H2_segmented']:
        # discontinuous colormap for column densities, split at 10^20 and 10^19 cm^(-3)
        assert valMinMax is not None # need for placing discontinuities at correct physical locations
        valCut1 = 17.0 # sub-LLS and LLS boundary # 19.0 previously
        valCut2 = 20.3 # LLS and DLA boundary # 20.0 previously

        if ctName == 'H2_segmented':
            valCut1 = 18.0
            valCut2 = 21.0

        fCut1 = (valCut1-valMinMax[0]) / (valMinMax[1]-valMinMax[0])
        fCut2 = (valCut2-valMinMax[0]) / (valMinMax[1]-valMinMax[0])

        if fCut1 <= 0 or fCut1 >= 1 or fCut2 <= 0 or fCut2 >= 1:
            # if valMinMax does not span these values, we create a corrupt cmap which cannot be rendered
            fCut1 = 0.33
            fCut2 = 0.66

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

    if ctName in ['tarn0','diff0','curl0','delta0','topo0','balance0']:
        # reshape a diverging colormap, which is otherwise centered at its midpoint, such that the center occurs at value zero
        valCut = 0.0 # e.g. log10(1) for tcool/tff, delta_rho

        fCut = (valCut-valMinMax[0]) / (valMinMax[1]-valMinMax[0])
        if fCut <= 0 or fCut >= 1: fCut = 0.5

        # sample from each side of the colormap
        x1 = np.linspace(0.0, 0.5, int(1024*fCut))
        x2 = np.linspace(0.5, 1.0, int(1024*(1-fCut)))

        cmap = getattr(cmocean.cm, ctName[:-1]) # acquire object member via string
        colors1 = cmap(x1)
        colors2 = cmap(x2)

        # combine them and construct a new colormap
        colors = np.vstack((colors1, colors2))
        cmap = LinearSegmentedColormap.from_list('magma_gray', colors)

        return cmap

    if ctName == 'magma_gray':
        # discontinuous colormap: magma on the upper half, grayscale on the lower half, split at 1e-16 (e.g. surface brightness)
        assert valMinMax is not None # need for placing discontinuities at correct physical locations
        valCut = 14.5 #-15.0 #np.log10(1e14) #-17.0

        fCut = (valCut-valMinMax[0]) / (valMinMax[1]-valMinMax[0])
        if fCut <= 0 or fCut >= 1:
            print('Warning: strange fCut, fix!')
            fCut = 0.5

        # sample from both colormaps
        x1 = np.linspace(0.1, 1.0, int(512*(1-fCut))) # avoid darkest (black) region of magma
        x2 = np.linspace(0.0, 0.8, int(512*fCut)) # avoid brightness whites
        colors1 = plt.cm.magma(x1)
        colors2 = plt.cm.gray(x2)

        # combine them and construct a new colormap
        colors = np.vstack((colors2, colors1))
        cmap = LinearSegmentedColormap.from_list('magma_gray', colors)

        return cmap

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

    if ctName == 'gasdens_tng5':
        # TNG gas matter density (shift yellow/dark blue transition lower)
        cdict = {'red'   : ((0.0, 0.027, 0.027), (0.1, 0.168, 0.168), (0.4,0.980,0.980), (0.55,0.286,0.286), (0.7, 0.282, 0.282), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'green' : ((0.0, 0.055, 0.055), (0.1, 0.322, 0.322), (0.4,0.898,0.898), (0.55,0.702,0.702), (0.7, 0.557, 0.557), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0)),
                 'blue'  : ((0.0, 0.075, 0.075), (0.1, 0.447, 0.447), (0.4,0.357,0.357), (0.55,0.302,0.302), (0.7, 0.792, 0.792), (0.99, 1.0, 1.0), (1.0, 1.0, 1.0))}
        cmap = LinearSegmentedColormap(ctName, cdict, N=1024)

    if ctName == 'blue_red_t10':
        # pure blue -> red using the tableau10 colors
        red_r = 214.0/255 ; red_g = 39.0/255 ; red_b = 40.0/255
        blue_r = 31.0/255 ; blue_g = 119.0/255 ; blue_b = 180.0/255
        cm_data = [ [red_r, red_g, red_b], [blue_r, blue_g, blue_b]]
        cmap = LinearSegmentedColormap.from_list('blue_red_t10', cm_data)

    if cmap is None:
        raise Exception('Unrecognized colormap request ['+ctName+'] or not implemented.')

    def _plawScale(cmap, plaw_index):
        assert plaw_index > 0.0
        cdict = {}

        # ListedColormap has no _segmentdata
        if not hasattr(cmap, '_segmentdata'):
            cmap = copy.deepcopy(cmap)
            cmap._segmentdata = {'red':[],'green':[],'blue':[]}
            for i, color in enumerate(cmap.colors):
                ind = float(i) / (len(cmap.colors)-1) # [0,255] -> [0,1] typically
                cmap._segmentdata['red'].append( [ind,color[0],color[0]] )
                cmap._segmentdata['green'].append( [ind,color[1],color[1]] )
                cmap._segmentdata['blue'].append( [ind,color[2],color[2]] )

        # pull out RGB triplets and scale
        N = 1024

        for k in ['red','green','blue']:
            cdict[k] = []
            nElem = len(cmap._segmentdata[k]) if not callable(cmap._segmentdata[k]) else N # detect lambda

            for j in range(nElem):
                if callable(cmap._segmentdata[k]):
                    # sample lambda function through [0,1]
                    pos = float(j)/(N-1)
                    val = cmap._segmentdata[k](pos)
                    xx = [pos, val, val]
                else:
                    # pull out actual discrete entries
                    xx = cmap._segmentdata[k][j]
                cdict[k].append( [xx[0]**plaw_index, xx[1], xx[2]] )
            #assert (cdict[k][0] < 0 or cdict[k][-1] > 1) # outside [0,1]
            
        return LinearSegmentedColormap(ctName+'_p', cdict, N=N)

    if plawScale is not None:
        cmap = _plawScale(cmap, plawScale)

    if cmapCenterVal is not None:
        assert cmapCenterVal > valMinMax[0] and cmapCenterVal < valMinMax[1]
        center_rel = np.abs(cmapCenterVal - valMinMax[0]) / np.abs(valMinMax[1] - valMinMax[0])
        plaw_index = np.log(center_rel) / np.log(0.5)
        cmap = _plawScale(cmap, plaw_index)

    if fracSubset is not None:
        cmap = LinearSegmentedColormap.from_list(
          'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=fracSubset[0], b=fracSubset[1]),
          cmap(np.linspace(fracSubset[0], fracSubset[1], 512)) )

    return cmap

def sampleColorTable(ctName, num, bounds=None):
    """ Grab a sequence of colors, evenly spaced, from a given colortable. """
    from matplotlib.pyplot import cm

    if ctName == 'tableau10':
        # current custom implementation of name-based color picking from this cm
        # note: exists in matplotlib 2.0+ as 'tab10'
        colors = {'blue':'#1F77B4','orange':'#FF7F0E','green':'#2CA02C','red':'#D62728','purple':'#9467BD',
                  'brown':'#8C564B','pink':'#E377C2','gray':'#BCBD22','yellow':'#17BECF','lightblue':'#7F7F7F'}
        r = [colors[name] for name in iterable(num)]
        if len(r) == 1: return r[0]
        return r

    cmap = cm.get_cmap(ctName)
    if bounds is None: bounds = [0,1]
    return cmap( np.linspace(bounds[0],bounds[1],num) )

def contourf(*args, **kwargs):
    """ Wrap matplotlib.contourf() for a graphical fix in PDF output. """
    cnt = plt.contourf(*args, **kwargs)

    for c in cnt.collections:
        c.set_edgecolor("face")
        c.set_linewidth(0.1) # must be nonzero to fix sub-pixel AA issue

    return cnt

def plothist(x, filename='out.pdf', nBins=50, norm=False, skipzeros=True):
    """ Plot a quick 1D histogram of an array x and save it to a PDF. """
    if skipzeros: x = x[x != 0.0]

    # linear (x)
    x_range = [ np.nanmin(x), np.nanmax(x) ]
    binSize = (x_range[1] - x_range[0]) / nBins
    yy_lin, xx_lin = np.histogram(x, bins=nBins, range=x_range, density=norm)
    xx_lin = xx_lin[:-1] + 0.5*binSize

    # log (x)
    x = logZeroNaN(x)
    x_range_log = [ np.nanmin(x), np.nanmax(x) ]
    binSize = (x_range_log[1] - x_range_log[0]) / nBins
    if np.isfinite(x_range_log[0]):
        yy_log, xx_log = np.histogram(x, bins=nBins, range=x_range_log, density=norm)
        xx_log = xx_log[:-1] + 0.5*binSize
    else:
        xx_log, yy_log = np.nan, np.nan # skip

    # figure
    figsize = np.array([14,10]) * 0.8 * 2
    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.set_xlabel(['x','log(x)','x','log(x)'][i])
        ax.set_ylabel(['N','N','log(N)','log(N)'][i])

        if i in [1,3]:
            x_plot = xx_log
            y_plot = yy_log
            ax.set_xlim(x_range_log)
        else:
            x_plot = xx_lin
            y_plot = yy_lin
            ax.set_xlim(x_range)
            
        if i in [2,3]: ax.set_yscale('log')

        ax.plot(x_plot,y_plot, '-', lw=2.5)
        ax.step(x_plot,y_plot, lw=2.5, where='mid',color='black',alpha=0.5)

    fig.savefig(filename)
    plt.close(fig)

def plotxy(x, y, filename='plot.pdf'):
    """ Plot a quick 1D line plot of x vs. y and save it to a PDF. """
    xx_log = logZeroNaN(x)
    yy_log = logZeroNaN(y)

    # figure
    figsize = np.array([14,10]) * 0.8 * 2
    fig = plt.figure(figsize=figsize)

    for i in range(4):
        ax = fig.add_subplot(2,2,i+1)
        ax.set_xlabel(['x','log(x)','x','log(x)'][i])
        ax.set_ylabel(['y','y','log(y)','log(y)'][i])

        if i == 0:
            ax.plot(x, y, 'o-', lw=2.5)
        if i == 1:
            ax.plot(xx_log, y, 'o-', lw=2.5)
        if i == 2:
            ax.plot(x, yy_log, 'o-', lw=2.5)
        if i == 3:
            ax.plot(xx_log, yy_log, 'o-', lw=2.5)

    fig.savefig(filename)
    plt.close(fig)

def plot2d(grid, label='', filename='plot.pdf'):
    """ Plot a quick image plot of a 2d array/grid and save it to a PDF. """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    figsize = np.array([14,10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    cmap = 'viridis'

    # plot
    plt.imshow(grid, cmap=cmap, aspect=grid.shape[0]/grid.shape[1])

    ax.autoscale(False)
    #plt.clim([minval,maxval])

    # colorbar
    cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)

    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(label)   

    fig.savefig(filename)
    plt.close(fig)

# --- I/O ---

def curRepoVersion():
    """ Return a hash of the current state of the mercurial python repo. """
    import subprocess
    from os import getcwd, chdir
    from os.path import expanduser
    from getpass import getuser

    oldCwd = getcwd()
    if getuser() != 'wwwrun':
        chdir(expanduser("~") + '/python/')
    else:
        chdir('/var/www/python/')
    
    command = ["git", "rev-parse", "--short", "HEAD"]
    repoRevStr = subprocess.check_output(command, stderr=subprocess.DEVNULL).strip()
    chdir(oldCwd)

    return repoRevStr
