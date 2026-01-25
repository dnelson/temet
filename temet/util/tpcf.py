"""
Two-point correlation functions (pairwise distances).
"""

import threading

import numpy as np
from numba import jit

from ..util.helper import pSplitRange
from ..util.sphMap import _NEAREST


@jit(nopython=True, nogil=True, cache=True)
def _calcTPCFBinned(pos, pos2, rad_bins_sq, boxSizeSim, xi_int, start_ind, stop_ind):
    """Core routine for tpcf(), see below."""
    numPart = pos2.shape[0]
    boxHalf = boxSizeSim / 2.0

    # radial_bins_max = np.max(rad_bins_sq[:-1])  # skip inf

    # loop over all particles
    for i in range(start_ind, stop_ind):
        pi_0 = pos[i, 0]
        pi_1 = pos[i, 1]
        pi_2 = pos[i, 2]

        for j in range(0, numPart):
            if i == j:
                continue

            pj_0 = pos2[j, 0]
            pj_1 = pos2[j, 1]
            pj_2 = pos2[j, 2]

            # calculate 3d periodic squared distance
            dx = _NEAREST(pi_0 - pj_0, boxHalf, boxSizeSim)
            dy = _NEAREST(pi_1 - pj_1, boxHalf, boxSizeSim)
            dz = _NEAREST(pi_2 - pj_2, boxHalf, boxSizeSim)

            r2 = dx * dx + dy * dy + dz * dz

            # find histogram bin and accumulate
            k = 1
            while r2 > rad_bins_sq[k]:
                k += 1

            xi_int[k - 1] += 1

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _calcTPCFBinnedWeighted(pos, weights, pos2, weights2, rad_bins_sq, boxSizeSim, xi_float, start_ind, stop_ind):
    """Core routine for tpcf(), see below."""
    numPart = pos2.shape[0]
    boxHalf = boxSizeSim / 2.0

    rad_bins_sq_max = np.max(rad_bins_sq[:-1])  # skip inf

    # loop over all particles
    for i in range(start_ind, stop_ind):
        pi_0 = pos[i, 0]
        pi_1 = pos[i, 1]
        pi_2 = pos[i, 2]

        for j in range(0, numPart):
            if i == j:
                continue

            pj_0 = pos2[j, 0]
            pj_1 = pos2[j, 1]
            pj_2 = pos2[j, 2]

            # calculate 3d periodic squared distance
            dx = _NEAREST(pi_0 - pj_0, boxHalf, boxSizeSim)
            if dx > rad_bins_sq_max:
                continue
            dy = _NEAREST(pi_1 - pj_1, boxHalf, boxSizeSim)
            if dy > rad_bins_sq_max:
                continue
            dz = _NEAREST(pi_2 - pj_2, boxHalf, boxSizeSim)
            if dz > rad_bins_sq_max:
                continue

            r2 = dx * dx + dy * dy + dz * dz
            if r2 > rad_bins_sq_max:
                continue

            # find histogram bin and accumulate
            k = 1
            while r2 > rad_bins_sq[k]:
                k += 1

            xi_float[k - 1] += weights[i] * weights2[j]

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _reduceQuantsInRad(
    pos_search, pos_target, radial_bins, quants, reduced_quants, reduce_type, boxSizeSim, start_ind, stop_ind
):
    """Core routine for quantReductionInRad()."""
    numQuants = quants.shape[1]
    numTarget = pos_target.shape[0]
    boxHalf = boxSizeSim / 2.0

    radial_bins_sq = np.power(radial_bins, 2.0)
    radial_bins_sq_max = np.max(radial_bins_sq)
    radial_bins_max = np.max(radial_bins)

    for i in range(start_ind, stop_ind):
        pi_0 = pos_search[i, 0]
        pi_1 = pos_search[i, 1]
        pi_2 = pos_search[i, 2]

        i_save = i - start_ind

        for j in range(0, numTarget):
            pj_0 = pos_target[j, 0]
            pj_1 = pos_target[j, 1]
            pj_2 = pos_target[j, 2]

            # do not count self if these two samples are the same
            if pi_0 == pj_0 and pi_1 == pj_1 and pi_2 == pj_2:
                continue

            # calculate 3d periodic squared distance
            dx = _NEAREST(pi_0 - pj_0, boxHalf, boxSizeSim)
            if dx > radial_bins_max:
                continue
            dy = _NEAREST(pi_1 - pj_1, boxHalf, boxSizeSim)
            if dy > radial_bins_max:
                continue
            dz = _NEAREST(pi_2 - pj_2, boxHalf, boxSizeSim)
            if dz > radial_bins_max:
                continue

            r2 = dx * dx + dy * dy + dz * dz

            # within radial search aperture?
            if r2 > radial_bins_sq_max:
                continue

            # find radial bin index
            r_ind = 0
            while r2 > radial_bins_sq[r_ind + 1]:
                r_ind += 1

            # MAX
            if reduce_type == 0:
                for k in range(0, numQuants):
                    if quants[j, k] > reduced_quants[i_save, r_ind, k]:
                        reduced_quants[i_save, r_ind, k] = quants[j, k]

            # MIN
            if reduce_type == 1:
                for k in range(0, numQuants):
                    if quants[j, k] < reduced_quants[i_save, r_ind, k]:
                        reduced_quants[i_save, r_ind, k] = quants[j, k]

            # SUM
            if reduce_type == 2:
                for k in range(0, numQuants):
                    reduced_quants[i_save, r_ind, k] += quants[j, k]

    # void return


def tpcf(pos, radialBins, boxSizeSim, weights=None, pos2=None, weights2=None, nThreads=32):
    """Calculate and simultaneously histogram the results of a two-point auto correlation function.

    Approach: compute all the pairwise (periodic) distances in pos. 3D only.

      pos[N,3]      : array of 3-coordinates for the galaxies/points
      radialBins[M] : array of (inner) bin edges in radial distance (code units)
      boxSizeSim[1] : the physical size of the simulation box for periodic wrapping (0=non periodic)
      weights[N]    : if not None, then use these scalars for a weighted correlation function
      pos2[L,3]     : if not None, then cross-correlation instead of auto, of pos vs. pos1
      weights2[L]   : must be None if weights is None, and vice versa

      return is xi(r),DD,RR where xi[i]=(DD/RR-1) is computed between radialBins[i:i+1] (size == M-1)
    """
    # input sanity checks
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] <= 1:
        raise Exception("Strange dimensions of pos.")
    if radialBins.ndim != 1 or radialBins.size < 2:
        raise Exception("Strange dimensions of radialBins.")
    if pos.dtype not in [np.float32, np.float64]:
        raise Exception("pos not in float32/64")
    if weights is not None:
        assert weights.size == pos.shape[0]
        assert weights.min() >= 0.0  # all finite and non-negative
        meanWeight = np.mean(weights)
        xi_dtype = "float64"
    else:
        meanWeight = 1.0
        xi_dtype = "int64"

    # cross-correlation sanity checks
    if pos2 is not None:
        if weights is not None:
            assert weights2 is not None
        if pos2.ndim != 2 or pos2.shape[1] != 3 or pos2.shape[0] <= 1:
            raise Exception("Strange dimensions of pos2.")
        if pos2.dtype not in [np.float32, np.float64]:
            raise Exception("pos2 not in float32/64")
    else:
        assert weights2 is None
        pos2 = pos  # views
        weights2 = weights

    # square radial bins
    nPts = pos.shape[0]
    nPts2 = pos2.shape[0]
    rad_bins_sq = np.copy(radialBins) ** 2
    cutFirst = False

    # add a inner bin edge at zero, and an outer bin edge at np.inf if not already present
    if rad_bins_sq[0] != 0.0:
        rad_bins_sq = np.insert(rad_bins_sq, 0, 0.0)
        cutFirst = True
    if rad_bins_sq[-1] != np.inf:
        rad_bins_sq = np.append(rad_bins_sq, np.inf)

    def _analytic_estimator(xi_int):
        # calculate RR expectation for periodic cube
        vol_enclosed = 4.0 / 3 * np.pi * np.sqrt(rad_bins_sq[:-1]) ** 3.0  # spheres
        bin_volume = np.diff(vol_enclosed)  # spherical shells

        # if e.g. pos is the whole sample (or the larger part), this will be less noisy:
        meanWeight2 = meanWeight  # if pos2 is None else np.mean(weights2)

        mean_num_dens = float(nPts * nPts2) / boxSizeSim**3.0 * (meanWeight * meanWeight2)  # N^2 / V_box
        RR = bin_volume * mean_num_dens

        # xi = DD / RR - 1.0
        DD = xi_int[:-1].astype("float32").copy()
        xi = DD / RR - 1.0

        # user did not start with a 0.0 inner bin edge, so throw this first bin out
        if cutFirst:
            xi = xi[1:]
            DD = DD[1:]
            RR = RR[1:]

        return xi, DD, RR

    # allocate return
    xi_int = np.zeros(rad_bins_sq.size - 1, dtype=xi_dtype)

    # single threaded?
    # ----------------
    if nThreads == 1:
        # call JIT compiled kernel
        start_ind = 0
        stop_ind = nPts

        if weights is not None:
            _calcTPCFBinnedWeighted(pos, weights, pos2, weights2, rad_bins_sq, boxSizeSim, xi_int, start_ind, stop_ind)
        else:
            _calcTPCFBinned(pos, pos2, rad_bins_sq, boxSizeSim, xi_int, start_ind, stop_ind)

        # transform integer counts into the correlation function with an analytic estimate for RR
        xi, DD, RR = _analytic_estimator(xi_int)

        return xi, DD, RR

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """Subclass Thread() to provide local storage."""

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # allocate local returns as attributes of the function
            self.xi_int = np.zeros(rad_bins_sq.size - 1, dtype=xi_dtype)

            # determine local slice (these are views not copies, even better)
            self.threadNum = threadNum
            self.nThreads = nThreads

            self.start_ind, self.stop_ind = pSplitRange([0, nPts], nThreads, threadNum)

            # make local view of pos (non-self inputs to JITed function appears to prevent GIL release)
            self.pos = pos
            self.weights = weights
            self.pos2 = pos2
            self.weights2 = weights2
            self.rad_bins_sq = rad_bins_sq
            self.boxSizeSim = boxSizeSim

        def run(self):
            # call JIT compiled kernel
            if weights is not None:
                _calcTPCFBinnedWeighted(
                    self.pos,
                    self.weights,
                    self.pos2,
                    self.weights2,
                    self.rad_bins_sq,
                    self.boxSizeSim,
                    self.xi_int,
                    self.start_ind,
                    self.stop_ind,
                )
            else:
                _calcTPCFBinned(
                    self.pos, self.pos2, self.rad_bins_sq, self.boxSizeSim, self.xi_int, self.start_ind, self.stop_ind
                )

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        xi_int += thread.xi_int

    # transform integer counts into the correlation function with an analytic estimate for RR
    xi, DD, RR = _analytic_estimator(xi_int)

    return xi, DD, RR


def quantReductionInRad(pos_search, pos_target, radial_bins, quants, reduce_op, boxSizeSim, nThreads=8):
    """Calculate a reduction operation on one or more quantities for each search point.

    In each case, consider all target points falling within a 3D periodic search radius.

    pos_search[N,3] : array of 3-coordinates for the galaxies/points to search from
    pos_target[M,3] : array of the 3-coordiantes of the galaxies/points to search over
    radial_bins[M]  : array of bin edges in radial distance (code units)
    quants[M]/[M,P] : 1d or P-d array of quantities, one per pos_target, to process
    reduce_op[str]  : one of 'min', 'max', 'sum'
    boxSizeSim[1]   : the physical size of the simulation box for periodic wrapping (0=non periodic)

    return is reduced_quants[N,M-1]/[N,M-1,P]
    """
    # input sanity checks
    if pos_search.ndim != 2 or pos_search.shape[1] != 3:
        raise Exception("Strange dimensions of pos_search.")
    if pos_target.ndim != 2 or pos_target.shape[1] != 3 or pos_target.shape[0] <= 1:
        raise Exception("Strange dimensions of pos_target.")
    if radial_bins.dtype not in [np.float32, np.float64] or radial_bins.size < 2:
        raise Exception("Strange type or size of radial_bins.")
    if pos_search.dtype != np.float32 and pos_search.dtype != np.float64:
        raise Exception("pos_search not in float32/64")
    if pos_target.dtype != np.float32 and pos_target.dtype != np.float64:
        raise Exception("pos_target not in float32/64")
    if quants.ndim not in [1, 2] or (quants.ndim == 2 and quants.shape[0] != pos_target.shape[0]):
        raise Exception("Strange dimensions of quants.")

    # prepare
    reduce_op = reduce_op.lower()
    reduceOps = {"max": 0, "min": 1, "sum": 2}
    assert reduce_op in reduceOps.keys()
    reduce_type = reduceOps[reduce_op]

    nSearch = pos_search.shape[0]
    #nTarget = pos_target.shape[0]
    nQuants = quants.shape[1] if quants.ndim == 2 else 1

    # radial bin(s)
    rad_bins_sq = np.copy(radial_bins) ** 2
    nRadBins = rad_bins_sq.size - 1

    # allocate return
    reduced_quants = np.zeros((nSearch, nRadBins, nQuants), dtype=quants.dtype)
    if reduce_op == "max":
        reduced_quants.fill(-np.inf)
    if reduce_op == "min":
        reduced_quants.fill(np.inf)

    # single threaded?
    # ----------------
    if nThreads == 1:
        # call JIT compiled kernel
        start_ind = 0
        stop_ind = nSearch

        _reduceQuantsInRad(
            pos_search, pos_target, radial_bins, quants, reduced_quants, reduce_type, boxSizeSim, start_ind, stop_ind
        )

        return reduced_quants

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """Subclass Thread() to provide local storage."""

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # determine local slice
            self.threadNum = threadNum
            self.nThreads = nThreads

            self.start_ind, self.stop_ind = pSplitRange([0, nSearch], nThreads, threadNum)

            # allocate local returns as attributes of the function
            nSearchLocal = self.stop_ind - self.start_ind
            self.reduced_quants = np.zeros((nSearchLocal, nRadBins, nQuants), dtype=quants.dtype)

            if reduce_op == "max":
                self.reduced_quants.fill(-np.inf)
            if reduce_op == "min":
                self.reduced_quants.fill(np.inf)

            # make local view of pos (non-self inputs to JITed function appears to prevent GIL release)
            self.pos_search = pos_search
            self.pos_target = pos_target
            self.radial_bins = radial_bins
            self.quants = quants
            self.reduce_type = reduce_type
            self.boxSizeSim = boxSizeSim

        def run(self):
            # call JIT compiled kernel
            _reduceQuantsInRad(
                self.pos_search,
                self.pos_target,
                self.radial_bins,
                self.quants,
                self.reduced_quants,
                self.reduce_type,
                self.boxSizeSim,
                self.start_ind,
                self.stop_ind,
            )

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        reduced_quants[thread.start_ind : thread.stop_ind, :, :] = thread.reduced_quants

    return reduced_quants
