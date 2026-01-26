"""
General helper functions, small algorithms, basic I/O, etc.
"""

import collections.abc as collections
from functools import wraps
from hashlib import sha256
from inspect import getsource
from os.path import abspath, dirname, isfile

import h5py
import numpy as np
from numba import jit
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares, leastsq
from scipy.stats import binned_statistic, gaussian_kde


# --- root path (e.g. '/home/username/temet/temet/') ----
rootPath = abspath(dirname(__file__) + "/../") + "/"

# --- utility functions ---


def nUnique(x):
    """Return number of unique elements in input numpy array x."""
    return (np.unique(x)).size


def isUnique(x):
    """Does input array contain only unique values?"""
    return x.size == (np.unique(x)).size


def closest(array, value):
    """Return closest element of array to input value."""
    ind = np.nanargmin(np.abs(array - value))
    ind_nd = np.unravel_index(ind, array.shape)
    return array[ind_nd], ind


def array_equal_nan(a, b):
    """As np.array_equal(a,b) but allowing NaN==NaN."""
    return ((a == b) | (np.isnan(a) & np.isnan(b))).all()


def evenlySample(sequence, num, logSpace=False):
    """Return num samples from sequence roughly equally spaced."""
    if sequence.size <= num:
        return sequence

    if logSpace:
        inds = np.logspace(0.0, np.log10(float(sequence.size) - 1), num)
    else:
        inds = np.linspace(0.0, float(sequence.size) - 1, num)

    return sequence[inds.astype("int32")]


@jit(nopython=True, nogil=True, cache=True)
def contiguousIntSubsets(x):
    """Return a list of index pairs corresponding to contiguous integer subsets of the input array.

    Final index of each pair is exclusive as in numpy syntax, so to obtain all the
    elements of a range from a numpy array, x[ranges[0][0]:ranges[0][1]].
    """
    # assert x.dtype in [np.int32,np.int64]

    ranges = []
    inRange = False

    for i in range(x.size - 1):
        # start new range?
        if not inRange:
            if x[i + 1] == x[i] + 1:
                inRange = True
                rangeStart = i
        else:
            if x[i + 1] == x[i] + 1:
                continue  # range continues
            else:
                # range is over, save and start next
                inRange = False
                rangeEnd = i + 1
                ranges.append((rangeStart, rangeEnd))
    if inRange:
        rangeEnd = i + 2
        ranges.append((rangeStart, rangeEnd))  # final range

    return ranges


def logZeroSafe(x, zeroVal=1.0):
    """Take log10 of input variable or array, keeping zeros at some value."""
    if np.isfinite(zeroVal):
        pass
        # print(' logZeroSafe: This was always ill-advised, migrate towards deleting this function.')
    if not isinstance(x, (int, float)) and x.ndim:  # array
        # another approach: if type(x).__module__ == np.__name__: print('is numpy object')
        with np.errstate(invalid="ignore"):
            w = np.where(x <= 0.0)
        x[w] = zeroVal
    else:  # scalar
        if x <= 0.0:
            x = zeroVal

    return np.log10(x)


def logZeroMin(x):
    """Take log10 of input variable, setting zeros to 100 times less than the minimum."""
    if isinstance(x, np.number) and not isinstance(x, np.ndarray) and x.size == 1:
        x = np.array([x])

    with np.errstate(invalid="ignore"):
        w = np.where(x > 0.0)

    minVal = x[w].min() if len(w[0]) > 0 else 1.0
    return logZeroSafe(x, minVal * 0.01)


def logZeroNaN(x):
    """Take log10, setting zeros to NaN and leaving NaN as NaN (same as default behavior, but suppress warnings)."""
    r = x.copy()
    if r.ndim == 0:
        r = np.array(r)
    r[~np.isfinite(r)] = 0.0
    return logZeroSafe(r, np.nan)


def last_nonzero(array, axis, invalid_val=-1):
    """Return the indices of the last nonzero entries of the array, along the given axis."""
    mask = array != 0

    val = array.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)


def iterable(x):
    """Wrap input as needed so that a for loop can iterate over this object correctly.

    Protect against non-list/non-tuple (e.g. scalar or single string) values.
    """
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return np.reshape(x, 1)  # scalar to 1d array of 1 element
    elif isinstance(x, collections.Iterable) and not isinstance(x, str):
        return x
    else:
        return [x]


def rebin(x, shape):
    """Resize input array x, must be 2D, to new shape, probably smaller, by averaging."""
    assert x.ndim == 2
    assert shape[0] <= x.shape[0]
    assert shape[1] <= x.shape[1]

    sh = shape[0], x.shape[0] // shape[0], shape[1], x.shape[1] // shape[1]
    return x.reshape(sh).mean(-1).mean(1)


def reportMemory():
    """Return current Python process memory usage in GB."""
    import os

    import psutil

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0**3  # GB


def numPartToChunkLoadSize(numPart):
    """For a given snapshot size, in terms of total particle count, decide on a good chunk loading size.

    Goal: a reasonable compute/memory balance.
    """
    nChunks = np.max([4, int(numPart ** (1.0 / 3.0) / 20.0)])
    return nChunks


def tail(fileName, nLines):
    """Wrap linux tail command line utility."""
    import subprocess

    lines = subprocess.check_output(["tail", "-n", str(nLines), fileName])
    if isinstance(lines, bytes):
        lines = lines.decode("utf-8")
    return lines


# --- decorators ---


def cache(_func=None, *, overwrite=False):
    """Decorator to cache the return (dict) of a function. Cache filename depends on arguments."""
    from ..util.simParams import simParams

    def decorator_cache(func):
        @wraps(_func)
        def wrapper(*args, **kwargs):
            # identify simParams in args
            sim = None
            for arg in args:
                if isinstance(arg, simParams):
                    assert sim is None  # should only be one
                    sim = arg
            assert sim is not None  # should be one, our path is sP-dependent

            cachefile = sim.cachePath + f"f_{func.__name__}.hdf5"

            # cachefile name includes hash that depends on args and source code of func
            # (this hash could be within the file, and used to invalidate the cache and overwrite)
            # unless the function has only a single argument that is a simParams object
            if len(args) > 0 or len(kwargs) > 0 and not (len(args) == 1 and isinstance(args[0], simParams)):
                # get function source (note: includes decorator line(s), comments, and so on)
                hashstr = getsource(func)
                hashstr = hashstr[hashstr.find("def ") :]  # strip leading decorator line(s)

                # append args and kwargs as string pairs
                for arg in args:
                    hashstr += str(arg) + ";"
                for key in sorted(kwargs.keys()):
                    hashstr += key + "=" + str(kwargs[key]) + ";"

                # append simParams details (otherwise is not unique in e.g. snap)
                hashstr += str(sim.snap)

                hashval = sha256(hashstr.encode("utf-8")).hexdigest()[::4]

                cachefile = cachefile.replace(".hdf5", f"_{hashval}.hdf5")

            if isfile(cachefile) and not overwrite:
                # load cached
                with h5py.File(cachefile, "r") as f:
                    data = {key: f[key][()] for key in f.keys()}
                if "_return" in data and len(data) == 1:
                    # single ndarray
                    data = data["_return"]
                print(f"Loaded [{cachefile}].")
            else:
                # call i.e. compute
                data = func(*args, **kwargs)

                if isinstance(data, dict):
                    # save cache
                    with h5py.File(cachefile, "w") as f:
                        for key in data:
                            f[key] = data[key]
                    print(f"Saved [{cachefile}].")
                elif isinstance(data, np.ndarray):
                    # save cache
                    with h5py.File(cachefile, "w") as f:
                        f["_return"] = data
                    print(f"Saved [{cachefile}].")
                else:
                    print(f"Function [{func.__name__}] did not return a ndarray or dict, not saving cache.")

            return data

        return wrapper

    if _func is None:
        return decorator_cache
    else:
        return decorator_cache(_func)


# --- running median ---


def running_median(
    X,
    Y,
    nBins=100,
    binSize=None,
    binSizeLg=None,
    skipZeros=False,
    percs=None,
    minNumPerBin=10,
    mean=False,
    weights=None,
):
    """Create a adaptive median line of a (x,y) point set using some number of bins."""
    assert X.shape == Y.shape
    if weights is not None:
        assert mean
        assert weights.size == Y.size

    minVal = np.nanmin(X[np.isfinite(X)])
    maxVal = np.nanmax(X[np.isfinite(X)])
    if skipZeros:
        minVal = np.nanmin(X[X != 0.0])

    if np.isnan(minVal):
        print("Bad inputs, going to fail in linspace.")

    if binSize is not None:
        nBins = round((maxVal - minVal) / binSize)

    if nBins <= 0:
        nBins = 1
    bins = np.linspace(minVal, maxVal, nBins)

    if binSizeLg is not None:
        # small bins for low x values (e.g. halo mass)
        splitX = np.nanpercentile(X, 90)  # rough heuristic
        nBins0 = int((splitX - minVal) / binSize)
        nBins1 = int((maxVal - splitX) / binSizeLg)
        bins0 = np.linspace(minVal, splitX, nBins0)
        bins1 = np.linspace(splitX, maxVal, nBins1)[1:]
        bins = np.hstack((bins0, bins1))

    running_median = []
    running_std = []
    bin_centers = []
    if percs is not None:
        running_percs = [[] for p in percs]

    binLeft = bins[0]

    for i in range(bins.size):
        binMax = bins[i + 1] if i + 1 < bins.size else np.inf

        with np.errstate(invalid="ignore"):  # expect X may contain NaN which should not be included
            w = np.where((X >= binLeft) & (X < binMax))

        # non-empty bin, or last bin with at least minNumPerBin/2 elements
        if len(w[0]) >= minNumPerBin or (i == len(bins) - 1 and len(w[0]) >= minNumPerBin / 2):
            if np.isnan(Y[w]).all():
                continue

            binLeft = binMax
            if mean:
                if weights is None:
                    running_median.append(np.nanmean(Y[w]))
                else:
                    loc_value = np.nansum(Y[w] * weights[w]) / np.nansum(weights[w])
                    running_median.append(loc_value)
            else:
                running_median.append(np.nanmedian(Y[w]))
            running_std.append(np.nanstd(Y[w]))
            bin_centers.append(np.nanmedian(X[w]))

            # compute percentiles also?
            if percs is not None:
                for j, perc in enumerate(percs):
                    running_percs[j].append(np.nanpercentile(Y[w], perc, interpolation="linear"))

    bin_centers = np.array(bin_centers)
    running_median = np.array(running_median)
    running_std = np.array(running_std)

    if percs is not None:
        running_percs = np.array(running_percs)
        return bin_centers, running_median, running_std, running_percs

    return bin_centers, running_median, running_std


def running_median_clipped(
    X, Y_in, nBins=100, minVal=None, maxVal=None, binSize=None, skipZerosX=False, skipZerosY=False, clipPercs=(0, 90)
):
    """Create a constant-bin median line of a (x,y) point set, clipping outliers above/below clipPercs."""
    if minVal is None:
        if skipZerosX:
            minVal = np.nanmin(X[X != 0.0])
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
        nBins = round((maxVal - minVal) / binSize) + 1

    if nBins <= 0:
        nBins = 1
    bins = np.linspace(minVal, maxVal, nBins)
    delta = bins[1] - bins[0] if nBins >= 2 else np.inf

    running_median = np.zeros(nBins, dtype="float32")
    bin_centers = np.zeros(nBins, dtype="float32")

    running_median.fill(np.nan)
    bin_centers.fill(np.nan)

    with np.errstate(invalid="ignore"):  # both X and Y contain nan, silence np.where() warnings
        for i, bin in enumerate(bins):
            binMin = bin
            binMax = bin + delta
            w = np.where((X >= binMin) & (X < binMax))

            if len(w[0]) == 0:
                continue

            # compute percentiles
            percs = np.nanpercentile(Y[w], clipPercs, interpolation="linear")

            # filter
            w_unclipped = np.where((Y[w] >= percs[0]) & (Y[w] <= percs[1]))

            if len(w_unclipped[0]) == 0:
                continue

            # compute median on points inside sigma-clipped region only
            running_median[i] = np.nanmedian(Y[w][w_unclipped])
            bin_centers[i] = np.nanmedian(X[w][w_unclipped])

    return bin_centers, running_median, bins


def running_median_sub(
    X, Y, S, nBins=100, binSize=None, skipZeros=False, sPercs=(25, 50, 75), percs=(16, 84), minNumPerBin=10
):
    """Create a adaptive median line of a (x,y) point set, by slicing in a third value on percentiles.

    Use some number of bins, where in each bin only the sub-sample of points obtained by slicing a third value
    (S) above and/or below one or more percentile thresholds is used.
    """
    minVal = np.nanmin(X)
    if skipZeros:
        minVal = np.nanmin(X[X != 0.0])

    if binSize is not None:
        nBins = round((np.nanmax(X) - minVal) / binSize)

    if nBins <= 0:
        nBins = 1
    bins = np.linspace(minVal, np.nanmax(X), nBins)
    delta = bins[1] - bins[0] if nBins >= 2 else np.inf

    bin_centers = []
    running_medianA = [[] for p in sPercs]
    running_medianB = [[] for p in sPercs]
    running_percsA = [[] for p in sPercs]
    running_percsB = [[] for p in sPercs]

    binLeft = bins[0]

    for i, bin in enumerate(bins):
        binMax = bin + delta
        w = np.where((X >= binLeft) & (X < binMax))

        # non-empty bin
        # if len(w[0]):
        # non-empty bin, or last bin with at least minNumPerBin/2 elements
        if len(w[0]) >= minNumPerBin or (i == len(bins) - 1 and len(w[0]) >= minNumPerBin / 2):
            # slice third quantity
            slice_perc_vals = np.nanpercentile(S[w], sPercs, interpolation="linear")

            bin_centers.append(np.nanmedian(X[w]))

            binLeft = binMax

            for i, slice_perc_val in enumerate(slice_perc_vals):
                # which points in this bin are above/below threshold percentile (e.g. median)?
                with np.errstate(invalid="ignore"):
                    w_sliceA = np.where(S[w] > slice_perc_val)
                    w_sliceB = np.where(S[w] <= slice_perc_val)

                running_medianA[i].append(np.nanmedian(Y[w][w_sliceA]))
                running_medianB[i].append(np.nanmedian(Y[w][w_sliceB]))

                # compute percentiles also
                running_percsA[i] = np.nanpercentile(Y[w][w_sliceA], percs, interpolation="linear")
                running_percsB[i] = np.nanpercentile(Y[w][w_sliceB], percs, interpolation="linear")

    bin_centers = np.array(bin_centers)
    running_medianA = np.array(running_medianA)
    running_medianB = np.array(running_medianB)
    running_percsA = np.array(running_percsA)
    running_percsB = np.array(running_percsB)

    return bin_centers, running_medianA, running_medianB, running_percsA, running_percsB


def running_sigmawindow(X, Y, windowSize=None):
    """Create an local/adaptive estimate of the stddev of a (x,y) point set using a sliding window."""
    assert X.size == Y.size
    if windowSize is None:
        windowSize = 3

    windowHalf = round(windowSize / 2.0)

    if windowHalf < 1:
        raise Exception("Window half size is too small.")

    running_std = np.zeros(X.size, dtype="float32")

    for i in np.arange(X.size):
        indMin = np.max([0, i - windowHalf])
        indMax = np.min([i + windowHalf, X.size])

        running_std[i] = np.std(Y[indMin:indMax])

    return running_std


def running_histogram(X, nBins=100, binSize=None, normFac=None, skipZeros=False):
    """Create a adaptive histogram of a (x) point set using some number of bins."""
    minVal = np.nanmin(X[np.isfinite(X)])
    maxVal = np.nanmax(X[np.isfinite(X)])
    if skipZeros:
        minVal = np.nanmin(X[np.isfinite(X) & (X != 0.0)])

    if binSize is not None:
        nBins = round((maxVal - minVal) / binSize)

    bins = np.linspace(minVal, maxVal, nBins)
    delta = bins[1] - bins[0]

    running_h = []
    bin_centers = []

    for _i, bin in enumerate(bins):
        binMax = bin + delta
        with np.errstate(invalid="ignore"):
            w = np.where((X >= bin) & (X < binMax))

        if len(w[0]):
            running_h.append(len(w[0]))
            bin_centers.append(np.nanmedian(X[w]))

    if normFac is not None:
        running_h /= normFac

    return np.array(bin_centers), np.array(running_h)


# --- general algorithms ---


def dist_theta_grid(size, nPixels):
    """Compute impact parameter and angle for every pixel of a map.

    Args:
      size (float): physical size of the map [pkpc].
      nPixels (int or 2-tuple): number of pixels along each axis.
    """
    # pixel size: [pkpc] if size in [pkpc], else units of size
    if isinstance(nPixels, int):
        nPixels = [nPixels, nPixels]

    pxSize = size / nPixels[0]

    xx, yy = np.mgrid[0 : nPixels[0], 0 : nPixels[1]]
    xx = xx.astype("float64") - nPixels[0] / 2 + 0.5
    yy = yy.astype("float64") - nPixels[1] / 2 + 0.5
    dist = np.sqrt(xx**2 + yy**2) * pxSize

    theta = np.rad2deg(np.arctan2(xx, yy))  # 0 and +/- 180 is major axis, while +/- 90 is minor axis
    theta = np.abs(theta)  # 0 -> 90 -> 180 is major -> minor -> major

    w = np.where(theta >= 90.0)
    theta[w] = 180.0 - theta[w]  # 0 is major, 90 is minor

    return dist, theta


def shrinking_center(xyz, boxSize, frac_stop=0.1, drop_frac_per_iter=0.05):
    """Shrinking center algorithm: iteratively search for a center position given a [N,3] coordinate set."""
    # starting state
    mask = np.zeros(xyz.shape[0], dtype="int16") + 1

    # config
    max_iter = 100
    num_stop = int(xyz.shape[0] * frac_stop)  # until the inner 10% are left

    for _i in range(max_iter):
        # which points remain
        w = np.where(mask)[0]
        num_left = len(w)

        # compute center and distances
        cen = np.mean(xyz[w], axis=0)
        dists = periodicDistsN(cen, xyz[w], boxSize)

        # sort
        sort_inds = np.argsort(dists)

        # exclude 5% most distant
        ind = int(sort_inds.size * (1 - drop_frac_per_iter))

        if ind == 0 or ind == sort_inds.size or num_left <= num_stop:
            break

        exclude_inds = w[sort_inds[ind:]]
        mask[exclude_inds] = 0

    return cen


def replicateVar(childCounts, subsetInds=None):
    """Given a number of children for each parent, replicate the parent indices for each child.

    subset_inds : still need to walk the full child_counts, but only want parent indices of a subset.
    """
    offset = 0

    if subsetInds is None:
        # full
        parentInds = np.array(np.sum(childCounts), dtype="uint32")

        for i in np.arange(childCounts.size):
            if childCounts[i] > 0:
                parentInds[offset : offset + childCounts[i]] = np.repeat(i, childCounts[i])
            offset += childCounts[i]

        return parentInds

    else:
        # subset
        totChildren = np.sum(childCounts[subsetInds])

        # we also return the child index array (i.e. which children belong to the subsetInds parents)
        r = {"parentInds": np.array(totChildren, dtype="uint32"), "childInds": np.array(totChildren, dtype="uint32")}

        offsetSub = 0

        subsetMask = np.zeros(childCounts.size, dtype="int8")
        subsetMask[subsetInds] = 1

        for i in np.arange(childCounts.size):
            if subsetMask[i] == 1 and childCounts[i] > 0:
                r["parentInds"][offsetSub : offsetSub + childCounts[i]] = np.repeat(i, childCounts[i])
                r["childInds"][offsetSub : offsetSub + childCounts[i]] = np.arange(childCounts[i]) + offset

                offsetSub += childCounts[i]
            offset += childCounts[i]

        return r


def pSplit(array, numProcs, curProc):
    """Divide work for embarassingly parallel problems."""
    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return array  # no split, return whole job load to caller

    # split array into numProcs segments, and return the curProc'th segment
    splitSize = int(np.floor(len(array) / numProcs))
    arraySplit = array[curProc * splitSize : (curProc + 1) * splitSize]

    # for last split, make sure it takes any leftovers
    if curProc == numProcs - 1:
        arraySplit = array[curProc * splitSize :]

    return arraySplit


def pSplitRange(indrange, numProcs, curProc, inclusive=False):
    """As pSplit(), but accept a 2-tuple of [start,end] indices and return a new range subset.

    If inclusive==True, then assume the range subset will be used e.g. as input to snapshotSubseet(),
    which unlike numpy convention is inclusive in the indices.
    """
    assert len(indrange) == 2 and indrange[1] > indrange[0]

    if numProcs == 1:
        if curProc != 0:
            raise Exception("Only a single processor but requested curProc>0.")
        return indrange

    # split array into numProcs segments, and return the curProc'th segment
    splitSize = int(np.floor((indrange[1] - indrange[0]) / numProcs))
    start = indrange[0] + curProc * splitSize
    end = indrange[0] + (curProc + 1) * splitSize

    # for last split, make sure it takes any leftovers
    if curProc == numProcs - 1:
        end = indrange[1]

    if inclusive and curProc < numProcs - 1:
        # not for last split/final index, because this should be e.g. NumPart[0]-1 already
        end -= 1

    return [start, end]


def getIDIndexMapSparse(ids):
    """Return an array which maps ID->indices within dense, disjoint subsets which can be sparse in the entire ID range.

    Within each subset i of size binsize array[ID-minids[i]+offset[i]] is the index of the original array ids
    where ID is found (assumes no duplicate IDs).
    """
    raise Exception("Not implemented.")


def getIDIndexMap(ids):
    """Return an array such that array[ID-min(ids)] is the index of the original array ids where ID is found.

    The return has size max(ids)-min(ids).
    Assumes a one to one mapping, not repeated indices as in the case of parentIDs for tracers).
    """
    minid = np.min(ids)
    maxid = np.max(ids)

    dtype = "uint32"
    if ids.size >= 2e9:
        dtype = "uint64"

    # direct indexing approach (pretty fast)
    arr = np.zeros(maxid - minid + 1, dtype=dtype)
    arr[ids - minid] = np.arange(ids.size, dtype=dtype)

    # C-style loop approach (good for sparse IDs)
    # arr = ulonarr(maxid-minid+1)
    # for i=0ULL,n_elements(ids)-1L do arr[ids[i]-minid] = i

    # looped where approach (never a good idea)
    # arr = l64indgen(maxid-minid+1)
    # for i=minid,maxid do begin
    #  w = where(ids eq i,count)
    #  if (count gt 0) then arr[i] = w[0]
    # endfor

    # reverse histogram approach (good for dense ID sampling, maybe better by factor of ~2)
    # arr = l64indgen(maxid-minid+1)
    # h = histogram(ids,rev=rev,omin=omin)
    # for i=0L,n_elements(h)-1 do if (rev[i+1] gt rev[i]) then arr[i] = rev[rev[i]:rev[i+1]-1]

    return arr, minid


def trapsum(xin, yin):
    """Trapezoidal rule numerical quadrature."""
    assert xin.size == yin.size
    assert xin.size >= 2
    nn = xin.size
    return np.sum(np.abs(xin[1 : nn - 1] - xin[0 : nn - 2]) * (yin[1 : nn - 1] + yin[0 : nn - 2]) * 0.5)


def leastsq_fit(func, params_init, args=None):
    """Wrap scipy.optimize.leastsq() by making the error function and handling returns.

    If args is not None, then the standard errors are also computed, but this ASSUMES that
    args[0] is the x data points and args[1] is the y data points.
    """

    def error_function(params, x, y, fixed=None):
        y_fit = func(x, params, fixed)
        return y_fit - y

    params_best, params_cov, info, errmsg, retcode = leastsq(error_function, params_init, args=args, full_output=True)

    # estimate errors (unused)
    params_stddev = np.zeros(len(params_best), dtype="float32")

    if params_cov is not None and args is not None:
        # assume first two elements of args are x_data and y_data
        assert len(args) >= 2 and args[0].shape == args[1].shape
        x_data, y_data = args[0], args[1]

        # reduced chi^2 (i.e. residual variance)
        chi2 = np.sum(error_function(params_best, x_data, y_data) ** 2.0)
        reduced_chi2 = chi2 / (y_data.size - len(params_best))

        # incorporate into fractional covariance matrix
        params_cov *= reduced_chi2

        # square root of diagonal elements estimates stddev of each parameter
        for j in range(len(params_best)):
            params_stddev[j] = np.abs(np.sqrt(params_cov[j][j]))

    return params_best, params_stddev


def least_squares_fit(func, params_init, params_bounds, args=None):
    """Wrap least_squares() using the Trust Region Reflective method for fitting with parameter constraints."""

    def error_function(params, x, y, fixed=None):
        y_fit = func(x, params, fixed)
        return y_fit - y

    # e.g. two parameters, require that x[1] >= 1.5, and x[0] left unconstrained
    # (lower bounds, upper bounds)
    # where each is a scalar or an array of the same size as parameters
    # bounds2 = ([-np.inf, 1.5], np.inf)

    result = least_squares(error_function, params_init, bounds=params_bounds, args=args, method="trf")

    return result.x


def reducedChiSq(sim_x, sim_y, data_x, data_y, data_yerr=None, data_yerr_up=None, data_yerr_down=None):
    """Compute reduced (i.e.) mean chi squared between a simulation 'line' and observed points with errors."""
    from scipy.interpolate import interp1d

    assert data_yerr is None or (data_yerr_up is None and data_yerr_down is None)
    assert np.sum([data_yerr_up is None, data_yerr_down is None]) in [0, 2]  # both or neither

    sim_f = interp1d(sim_x, sim_y, kind="linear")
    sim_y_at_data_x = sim_f(data_x)

    if data_yerr is not None:
        data_error = data_yerr
    else:
        data_error = data_yerr_down
        w = np.where(sim_y_at_data_x > data_y)
        data_error[w] = data_yerr_up[w]

    # weighted squared deviations
    devs = (sim_y_at_data_x - data_y) ** 2 / data_error**2

    chi2 = np.sum(devs)
    chi2v = chi2 / data_x.size

    return chi2v


def sgolay2d(z, window_size, order, derivative=None):
    """Szalay-golay filter in 2D using FFT convolutions."""
    from scipy.signal import fftconvolve

    # number of terms in the polynomial expression
    n_terms = (order + 1) * (order + 2) / 2.0

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    if window_size**2 < n_terms:
        raise ValueError("order is too high for the window size")
    half_size = window_size // 2
    exps = [(k - n, n) for k in range(order + 1) for n in range(k + 1)]

    # coordinates of points
    ind = np.arange(-half_size, half_size + 1, dtype=np.float64)
    dx = np.repeat(ind, window_size)
    dy = np.tile(ind, [window_size, 1]).reshape(window_size**2)

    # build matrix of system of equation
    A = np.empty((window_size**2, len(exps)))
    for i, exp in enumerate(exps):
        A[:, i] = (dx ** exp[0]) * (dy ** exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2 * half_size, z.shape[1] + 2 * half_size
    Z = np.zeros(new_shape)
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] = band - np.abs(np.flipud(z[1 : half_size + 1, :]) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band + np.abs(np.flipud(z[-half_size - 1 : -1, :]) - band)
    # left band
    band = np.tile(z[:, 0].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs(np.fliplr(z[:, 1 : half_size + 1]) - band)
    # right band
    band = np.tile(z[:, -1].reshape(-1, 1), [1, half_size])
    Z[half_size:-half_size, -half_size:] = band + np.abs(np.fliplr(z[:, -half_size - 1 : -1]) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0, 0]
    Z[:half_size, :half_size] = band - np.abs(np.flipud(np.fliplr(z[1 : half_size + 1, 1 : half_size + 1])) - band)
    # bottom right corner
    band = z[-1, -1]
    Z[-half_size:, -half_size:] = band + np.abs(
        np.flipud(np.fliplr(z[-half_size - 1 : -1, -half_size - 1 : -1])) - band
    )

    # top right corner
    band = Z[half_size, -half_size:]
    Z[:half_size, -half_size:] = band - np.abs(np.flipud(Z[half_size + 1 : 2 * half_size + 1, -half_size:]) - band)
    # bottom left corner
    band = Z[-half_size:, half_size].reshape(-1, 1)
    Z[-half_size:, :half_size] = band - np.abs(np.fliplr(Z[-half_size:, half_size + 1 : 2 * half_size + 1]) - band)

    # solve system and convolve
    if derivative is None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode="valid")
    elif derivative == "col":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode="valid")
    elif derivative == "row":
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode="valid")
    elif derivative == "both":
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode="valid"), fftconvolve(Z, -c, mode="valid")


def kde_2d(x, y, xrange, yrange):
    """Simple 2D KDE calculation using scipy gaussian_kde."""
    vv = np.vstack([x, y])
    kde = gaussian_kde(vv)

    xx, yy = np.mgrid[xrange[0] : xrange[1] : 200j, yrange[0] : yrange[1] : 400j]
    xy = np.vstack([xx.ravel(), yy.ravel()])
    kde2d = np.reshape(np.transpose(kde(xy)), xx.shape)

    return xx, yy, kde2d


def gaussian_filter_nan(zz, sigma):
    """Filter 2D array zz by a Gaussian with input sigma, ignoring any NaN entries."""
    if sigma == 0:
        return zz

    zz1 = zz.copy()
    zz1[np.isnan(zz)] = 0.0
    zz1 = gaussian_filter(zz1, sigma)

    zz2 = 0 * zz.copy() + 1.0
    zz2[np.isnan(zz)] = 0.0
    zz2 = gaussian_filter(zz2, sigma)

    with np.errstate(invalid="ignore"):
        zz = zz1 / zz2

    return zz


def mvbe(points, tol=0.005):
    """Find the 'minimum volume bounding ellipsoid' of a given set of points (iterative).

    Returns the ellipse equation in "center form" (x-c).T * A * (x-c) = 1
    Based on MVEE in Matlab by Nima Moshtagh.
    """
    N, d = points.shape
    Q = np.column_stack((points, np.ones(N))).T
    err = tol + 1.0
    u = np.ones(N) / N

    count = 0

    while err > tol:
        # assert u.sum() == 1 # invariant
        X = np.dot(np.dot(Q, np.diag(u)), Q.T)
        M = np.diag(np.dot(np.dot(Q.T, np.linalg.inv(X)), Q))
        jdx = np.argmax(M)
        step_size = (M[jdx] - d - 1.0) / ((d + 1) * (M[jdx] - 1.0))
        new_u = (1 - step_size) * u
        new_u[jdx] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

        if count > 1000:
            # print('WARNING, mvbe count >100, err = ', err, ' tol = ', tol)
            break
        count += 1

    cen = np.dot(u, points)
    A = np.linalg.inv(np.dot(np.dot(points.T, np.diag(u)), points) - np.multiply.outer(cen, cen)) / d

    # convert from center form into more typical (majoraxis,minoraxis,angle) form
    eig_w, eig_v = np.linalg.eig(A)

    # eigenvectors give principal axes of the ellipse
    eig_v1 = eig_v[0, :]
    theta = np.degrees(np.arccos(eig_v1[0]))

    # semimajor axis is the 1/eigenvalue with smaller absolute value
    axislengths = np.sqrt(1.0 / eig_w)  # larger is semi-major, smallest is semi-minor

    return axislengths, theta, cen


def binned_stat_2d(x, y, c, bins, range_x, range_y, stat="median"):
    """Replacement of binned_statistic_2d for mean_nan or median_nan."""
    assert stat in ["mean", "median"]

    nbins = bins[0] * bins[1]
    binsize_x = (range_x[1] - range_x[0]) / bins[0]
    binsize_y = (range_y[1] - range_y[0]) / bins[1]

    # finite only
    w = np.where(np.isfinite(x) & np.isfinite(y) & np.isfinite(c))
    x = x[w]
    y = y[w]
    c = c[w]

    # in bounds only
    w = np.where((x >= range_x[0]) & (x < range_x[1]) & (y >= range_y[0]) & (y < range_y[1]))

    # bin
    inds_x = np.floor((x[w] - range_x[0]) / binsize_x).astype("int32")
    inds_y = np.floor((y[w] - range_y[0]) / binsize_y).astype("int32")
    c = c[w]

    ind_1d = np.ravel_multi_index([inds_x, inds_y], bins)

    count = np.bincount(ind_1d, minlength=nbins).astype("float64")

    if stat == "mean":
        # statistic: mean
        total = np.bincount(ind_1d, c, minlength=nbins)

        # only non-zero bins
        w = np.where(count > 0)

        mean = np.zeros(nbins, dtype=c.dtype)
        mean.fill(np.nan)
        mean[w] = total[w] / count[w]

        result = np.reshape(mean, bins)

    if stat == "median":
        # statistic: median
        sort_inds = np.argsort(ind_1d)
        ind_1d_sorted = ind_1d[sort_inds]
        c_sorted = c[sort_inds]

        # bin edges and mid points
        bin_edges_inds = (ind_1d_sorted[1:] != ind_1d_sorted[:-1]).nonzero()[0] + 1
        target_inds = ind_1d_sorted[bin_edges_inds - 1]

        # allocate
        result = np.zeros(nbins, dtype=c.dtype)
        result.fill(np.nan)

        # handle odd/even definition for np.median
        # result[w] = (c_sorted[med_inds[w_even]] + c_sorted[med_inds[w_even]-1]) * 0.5
        # result[w] = c_sorted[med_inds[w_odd]]

        # fastest method (unfinished!)
        if 0:
            med_inds = (np.r_[0, bin_edges_inds] + np.r_[bin_edges_inds, len(w)]) * 0.5
            med_inds = med_inds.astype("int32")
            # sort_inds_c = np.argsort(c_sorted) # need to rearrange vals
            # med_inds must sample the s-sorted index list for each bin
            result[target_inds] = c_sorted[med_inds]

        # clearer method (effectively a sort on each bin subset)
        result[target_inds] = [np.median(i) for i in np.split(c_sorted, bin_edges_inds[:-1])]

        if 0:  # debug
            for bin_ind in range(10):
                ww = np.where(ind_1d == bin_ind)
                print(bin_ind, result[bin_ind], np.median(c[ww]), result[bin_ind] / np.median(c[ww]))

        result = np.reshape(result, bins)

    return result, np.reshape(count, bins)


def binned_statistic_weighted(x, values, statistic, bins, weights=None, weights_w=None):
    """Compute a binned statistic with optional weights.

    If weights == None, straight passthrough to scipy.stats.binned_statistic(). Otherwise,
    compute once for values*weights, again for weights alone, then normalize and return.
    If weights_w is not None, apply this np.where() result to the weights array.
    """
    if weights is None:
        return binned_statistic(x, values, statistic=statistic, bins=bins)

    weights_loc = weights[weights_w] if weights_w is not None else weights

    if statistic == "mean":
        # weighted mean (nan for bins where wt_sum == 0)
        valwt_sum, bin_edges, bin_number = binned_statistic(x, values * weights_loc, statistic="sum", bins=bins)
        wt_sum, _, _ = binned_statistic(x, weights_loc, statistic="sum", bins=bins)

        with np.errstate(invalid="ignore"):
            weighted_stat = valwt_sum / wt_sum  # zeros in wt_sum result in nan
        return weighted_stat, bin_edges, bin_number

    if statistic == np.std:
        # weighted standard deviation (note: numba accelerated)
        std = weighted_std_binned(x, values, weights, bins)
        return std, None, None


def lowess(x, y, x0, degree=1, l=1, robust=False):
    """
    Locally smoothed regression with the LOWESS algorithm (https://github.com/arokem/lowess).

    x - values of x for which f(x) is input, with shape (ndim,y.size)
    y - values of f(x) at these points (1d)
    x0 - x values to output LOWESS estimate for (can be the same as x)
    degree - degree of smoothing functions (0=locally constant, 1=linear, ...)
    l - metric window size for the kernel (scalar, or array of y.size)
    robust - if True, apply the robustification procedure from Cleveland (1979), pg 831
    """
    import numpy.linalg as la

    def epanechnikov(xx, l):
        ans = np.zeros(xx.shape)
        xx_norm = xx / l
        idx = np.where(xx_norm <= 1)
        ans[idx] = 0.75 * (1 - xx_norm[idx] ** 2)
        return ans

    def tri_cube(xx, **kwargs):
        ans = np.zeros(xx.shape)
        idx = np.where(xx <= 1)
        ans[idx] = (1 - np.abs(xx[idx]) ** 3) ** 3
        return ans

    def bi_square(xx, **kwargs):
        ans = np.zeros(xx.shape)
        idx = np.where(xx < 1)
        ans[idx] = (1 - xx[idx] ** 2) ** 2
        return ans

    if robust:
        # We use the procedure described in Cleveland+1979
        # Start by calling this function with robust set to false and the x0
        # input being equal to the x input:
        y_est = lowess(x, y, x, l=l, robust=False)
        resid = y_est - y
        median_resid = np.nanmedian(np.abs(resid))

        # calculate the bi-cube function on the residuals for robustness weights
        robustness_weights = bi_square(resid / (6 * median_resid))

    # for the case where x0 is provided as a scalar
    if not np.iterable(x0):
        x0 = np.asarray([x0])

    ans = np.zeros(x0.shape[-1], dtype=np.float32)

    # we only need one design matrix for fitting
    B = [np.ones(x.shape[-1])]
    for _d in range(1, degree + 1):
        B.append(x**degree)

    B = np.vstack(B).T
    for idx, this_x0 in enumerate(x0.T):
        # handle 1d case
        if not np.iterable(this_x0):
            this_x0 = np.asarray([this_x0])

        # Different weighting kernel for each x0
        xx = np.sqrt(np.sum(np.power(x - this_x0[:, np.newaxis], 2), 0))
        W = np.diag(epanechnikov(xx, l=l))
        # W = np.diag(do_kernel(this_x0, x, l=l, kernel=kernel))

        if robust:
            # apply the robustness weights to the weighted least-squares procedure
            robustness_weights[np.isnan(robustness_weights)] = 0
            W = np.dot(W, np.diag(robustness_weights))

        # try:
        # Equation 6.8 in HTF:
        BtWB = np.dot(np.dot(B.T, W), B)
        BtW = np.dot(B.T, W)
        # Get the params:
        beta = np.dot(np.dot(la.pinv(BtWB), BtW), y.T)
        # We create a design matrix for this coordinate for back-predicting:
        B0 = [1]
        for _d in range(1, degree + 1):
            B0 = np.hstack([B0, this_x0**degree])
        B0 = np.vstack(B0).T
        # Estimate the answer based on the parameters:
        ans[idx] += np.dot(B0, beta)

    # If we are trying to sample far away from where the function is
    # defined, we will be trying to invert a singular matrix. In that case,
    # the regression should not work for you and you should get a nan:
    # except la.LinAlgError :
    #    ans[idx] += np.nan
    return ans.T


# --- numba accelerated ---


@jit(nopython=True, nogil=True, cache=True)
def weighted_std_binned(x, vals, weights, bins):
    """Weighted standard deviation within bins.

    For a given set of bins (edges), histogram x into those bins, and then compute and
    return the standard deviation (unbiased) of vals weighted by weights, per-bin. Assumes
    'reliability' (non-random) weights, following
    https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance.
    """
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
    std = np.zeros(bins.size - 1, dtype=np.float32)

    # loop over bins
    offset = 0
    for i in range(bin_counts.size):
        if bin_counts[i] <= 1:
            std[i] = np.nan
            continue

        # get values and weights in this bin
        end_i = offset + bin_counts[i]

        loc_vals = vals_sorted[offset:end_i]
        loc_wts = weights_sorted[offset:end_i]

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
        valwt_sum = np.sum(loc_vals * loc_wts)

        weighted_mean = valwt_sum / wt_sum
        diff_sq = (loc_vals - weighted_mean) ** 2.0

        stdwt_sum = np.sum(diff_sq * loc_wts)

        if stdwt_sum == 0.0:
            std[i] = np.nan
            continue

        wt2_sum = np.sum(loc_wts * loc_wts)
        normalization = wt_sum - wt2_sum / wt_sum

        if normalization == 0.0:
            # protect against wt_sum == wt2_sum (i.e. N=1)
            normalization = wt_sum

        variance = stdwt_sum / normalization

        std[i] = np.sqrt(variance)

    return std


@jit(nopython=True, nogil=True, cache=True)
def bincount(x, dtype):
    """Same behavior as np.bincount() except can specify dtype different than x.dtype to save memory."""
    c = np.zeros(np.max(x) + 1, dtype=dtype)

    for i in range(x.size):
        c[x[i]] += 1

    return c


@jit(nopython=True, nogil=True, cache=True)
def periodicDistsN(pos1, pos2, BoxSize, squared=False):
    """Compute periodic distance between each (x,y,z) coordinate in pos1 vs. the corresponding point in pos2.

    Either pos1 and pos2 have the same shapes, and are matched pairwise, or pos1 is a tuple (i.e. reference position).
    """
    BoxHalf = BoxSize * 0.5

    dists = np.zeros(pos2.shape[0], dtype=pos2.dtype)
    assert pos1.shape[0] == pos2.shape[0] or pos1.size == 3
    assert pos2.shape[1] == 3
    if pos1.ndim == 2:
        assert pos1.shape[1] == 3

    for i in range(pos2.shape[0]):
        for j in range(3):
            if pos1.ndim == 1:
                xx = pos1[j] - pos2[i, j]
            else:
                xx = pos1[i, j] - pos2[i, j]

            if xx > BoxHalf:
                xx -= BoxSize
            if xx < -BoxHalf:
                xx += BoxSize

            dists[i] += xx * xx

    if not squared:
        for i in range(pos2.shape[0]):
            dists[i] = np.sqrt(dists[i])

    return dists


@jit(nopython=True, nogil=True, cache=True)
def periodicDistsIndexed(pos1, pos2, indices, BoxSize):
    """Compute periodic distance between each (x,y,z) coordinate in pos1 vs. the corresponding point in pos2.

    Here pos1.shape[0] != pos2.shape[0], e.g. in the case that pos1 are group centers, and pos2 are particle positions.
    Then indices gives, for each pos2, the index of the corresponding pos1 element to compute the distance to,
    e.g. the group ID of each particle.
    Return size is the length of pos2.
    """
    BoxHalf = BoxSize * 0.5

    dists = np.zeros(pos2.shape[0], dtype=pos1.dtype)
    assert pos1.shape[0] != pos2.shape[0]  # not generically expected
    assert pos1.shape[1] == 3 and pos2.shape[1] == 3
    assert indices.ndim == 1 and indices.size == pos2.shape[0]

    for i in range(pos2.shape[0]):
        pos1_loc = pos1[indices[i], :]
        for j in range(3):
            xx = pos1_loc[j] - pos2[i, j]

            if xx > BoxHalf:
                xx -= BoxSize
            if xx < -BoxHalf:
                xx += BoxSize

            dists[i] += xx * xx
        dists[i] = np.sqrt(dists[i])

    return dists


@jit(nopython=True, nogil=True, cache=True)
def crossmatchHalosByCommonIDs(ids1, lengths1, ids2, lengths2):
    """For each object in the first set, find the best-matching object in the second set, based on common IDs.

    For two sets of objects (e.g. halos or subhalos) which contain member IDs,
    finding the best matching index of the second set, for each object in each first set,
    coresponding to the object in the second set with the largest number of shared IDs.
    """
    assert lengths1.sum() <= ids1.size
    assert lengths2.sum() <= ids2.size

    # allocate
    cand_scores2 = np.zeros(lengths2.size, dtype=np.int32)  # change to float32 if weighted scores
    index2 = np.zeros(lengths1.size, dtype=np.int32)

    # ids must be dense, containing the members of set 1 packed by [lengths1[0], lengths1[1], ...]
    offset = 0

    # create mapping of ID -> index for set 2
    id_to_haloindex2 = np.zeros(ids2.max() + 1, dtype=np.int32) - 1

    for i in range(lengths2.size):
        for _j in range(lengths2[i]):
            id_to_haloindex2[ids2[offset]] = i
            offset += 1

    # loop over objects in set 1
    offset = 0

    for i in range(lengths1.size):
        # zero candidate scores
        cand_scores2 *= 0

        # loop over member IDs
        for _j in range(lengths1[i]):
            # select id
            loc_id = ids1[offset]
            offset += 1

            # what halo in set 2 does this id belong to?
            halo_index2 = id_to_haloindex2[loc_id]

            # if it belongs to a halo
            if halo_index2 >= 0:
                # add to ranking, could optionally use a weighting here
                weight = 1.0
                cand_scores2[halo_index2] += weight

        # decide (first is chosen if two or more equally ranked candidates exist)
        index2[i] = np.argmax(cand_scores2)

    return index2


@jit(nopython=True, nogil=True)
def faddeeva985(x, y):
    """Real arguments represent z = x + im*y and return only the real part."""
    # constants
    theta = 0.5641895835477563  # 1/np.sqrt(np.pi)

    a0 = 122.60793
    a1 = 214.38239
    a2 = 181.92853
    a3 = 93.15558
    a4 = 30.180142
    a5 = 5.9126262
    a6 = 0.5641895835477563  # 1/np.sqrt(np.pi)

    b0 = 122.60793
    b1 = 352.73063
    b2 = 457.33448
    b3 = 348.70392
    b4 = 170.35400
    b5 = 53.992907
    b6 = 10.479857

    g0 = 36183.31
    g1 = 3321.99
    g2 = 1540.787
    g3 = 219.031
    g4 = 35.7668
    g5 = 1.320522
    g6 = 1 / np.sqrt(np.pi)

    l0 = 32066.6
    l1 = 24322.84
    l2 = 9022.228
    l3 = 2186.181
    l4 = 364.2191
    l5 = 61.57037
    l6 = 1.841439

    s0 = 38000.0
    s1 = 256.0
    s2 = 62.0
    s3 = 30.0
    t3 = 1.0e-13
    s4 = 2.5
    t4 = 5.0e-9
    t5 = 0.072

    x_sq = x * x
    y_sq = y * y
    s = x_sq + y_sq

    # regions 1-3 are optimized/simplified for real-only return
    # region 1: Laplace continued fractions, 1 convergent
    if s >= s0:
        # print('region 1')
        return y * theta / s

    # region 2: Laplace continued fractions, 2 convergents
    if s >= s1:
        # print('region 2')
        return y * (0.5 + s) * (theta / ((s**2 + (y_sq - x_sq)) + 0.25))

    # region 3: Laplace continued fractions, 3 convergents
    if s >= s2:
        # print('region 3')
        q = y_sq - x_sq + 1.5
        r = 4.0 * x_sq * y_sq
        return theta * (y * ((q - 0.5) * q + r + x_sq)) / (s * (q * q + r))

    # regions 4+ are not optimized for real-only return
    if s >= s3 and y_sq >= t3:
        z = complex(x, y)
        z_sq = z**2
        result = (theta * complex(-y, x)) * (z_sq - 2.5) / (z_sq * (z_sq - 3.0) + 0.75)
        return result.real

    if s > s4 and y_sq < t4:
        z = complex(x, y)
        z_sq = z**2
        r = g0 + z_sq * (g1 + z_sq * (g2 + z_sq * (g3 + z_sq * (g4 + z_sq * (g5 + z_sq * g6)))))
        t = l0 + z_sq * (l1 + z_sq * (l2 + z_sq * (l3 + z_sq * (l4 + z_sq * (l5 + z_sq * (l6 + z_sq))))))

        result = np.exp(-x_sq) + z * r / t * (0 + 1j)
        return result.real
    elif s > s4 and y_sq < t5:
        z = complex(x, y)
        z_sq = z**2
        r = g0 + z_sq * (g1 + z_sq * (g2 + z_sq * (g3 + z_sq * (g4 + z_sq * (g5 + z_sq * g6)))))
        t = l0 + z_sq * (l1 + z_sq * (l2 + z_sq * (l3 + z_sq * (l4 + z_sq * (l5 + z_sq * (l6 + z_sq))))))
        result = np.exp(-z_sq) + z * r / t * (0 + 1j)
        return result.real

    # region 6
    q = complex(y, -x)  # y - j*x
    r = a0 + q * (a1 + q * (a2 + q * (a3 + q * (a4 + q * (a5 + q * a6)))))
    t = b0 + q * (b1 + q * (b2 + q * (b3 + q * (b4 + q * (b5 + q * (b6 + q))))))
    return (r / t).real


# --- I/O ---


def xypairs_to_np(s):
    """Convert a string of comma-separated x,y pairs, each separated by a newline, into two separate arrays."""
    x = np.array([float(x.split(",")[0]) for x in s.split("\n")])
    y = np.array([float(x.split(",")[1]) for x in s.split("\n")])

    return x, y


def curRepoVersion():
    """Return a hash of the current state of the mercurial python repo."""
    import subprocess
    from getpass import getuser
    from os import chdir, getcwd
    from pathlib import Path

    oldCwd = getcwd()
    if getuser() != "wwwrun":
        chdir(Path(__file__).parent.parent.absolute())
    else:
        chdir("/var/www/python/")

    command = ["git", "rev-parse", "--short", "HEAD"]
    repoRevStr = subprocess.check_output(command, stderr=subprocess.DEVNULL).strip()
    chdir(oldCwd)

    return repoRevStr
