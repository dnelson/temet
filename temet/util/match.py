"""
Helper function to efficiently match elements between two arrays.
"""

import time

import numpy as np
from numba import jit


try:
    from ..util.parallelSort import argsort as p_argsort
except ImportError:
    from numpy import argsort as p_argsort  # fallback


def match(ar1, ar2, firstSorted=False, parallel=True, debug=False):
    """Calculate the common elements between two arrays and their respective indices.

    Returns index arrays i1,i2 of the matching elements between ar1 and ar2.While the elements of ar1
    must be unique, the elements of ar2 need not be. For every matched element of ar2, the return i1
    gives the index in ar1 where it can be found. For every matched element of ar1, the return i2 gives
    the index in ar2 where it can be found. Therefore, ar1[i1] = ar2[i2]. The order of ar2[i2] preserves
    the order of ar2. Therefore, if all elements of ar2 are in ar1 (e.g. ar1=all TracerIDs in snap,
    ar2=set of TracerIDs to locate) then ar2[i2] = ar2. The approach is one sort of ar1 followed by
    bisection search for each element of ar2, therefore O(N_ar1*log(N_ar1) + N_ar2*log(N_ar1)) ~=
    O(N_ar1*log(N_ar1)) complexity so long as N_ar2 << N_ar1.
    """
    if not isinstance(ar1, np.ndarray):
        ar1 = np.array(ar1)
    if not isinstance(ar2, np.ndarray):
        ar2 = np.array(ar2)
    assert ar1.ndim == ar2.ndim == 1

    if debug:
        start = time.time()
        assert np.unique(ar1).size == len(ar1)

    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        if parallel:
            index = p_argsort(ar1)
        else:
            index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)
        ar1_sorted = None
        ar1_inds = np.take(index, ar1_sorted_index, mode="clip")
        ar1_sorted_index = None
        index = None
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)
        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index, mode="clip")

    mask = ar1[ar1_inds] == ar2
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None, None

    if debug:
        if not np.array_equal(ar1[ar1_inds], ar2[ar2_inds]):
            raise Exception("match fail")
        print(" match: " + str(round(time.time() - start, 2)) + " sec")

    return ar1_inds, ar2_inds


def match1(ar1, ar2, uniq=False, debug=False):
    """Calculate the common elements between two arrays and their respective indices (unused version 1).

    My version of numpy.in1d with invert=False. Return is a ndarray of indices into ar1,
    corresponding to elements which exist in ar2. Meant to be used e.g. as ar1=all IDs in
    snapshot, and ar2=some IDs to search for, where ar2 could be e.g. ParentID from the
    tracers, in which case they are generally not unique (multiple tracers can exist in the
    same parent).
    """
    start = time.time()

    # flatten both arrays (behavior for the first array could be different)
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # tuning: special case for small B arrays (significantly faster than the full sort)
    if len(ar2) < 10 * len(ar1) ** 0.145:
        mask = np.zeros(len(ar1), dtype="bool")
        for a in ar2:
            mask |= ar1 == a
        return mask

    # otherwise use sorting of the concatenated array: here we use a stable 'mergesort',
    # such that the values from the first array are always before the values from the second
    start_uniq = time.time()
    if not uniq:
        ar1, rev_idx = np.unique(ar1, return_inverse=True)
        ar2 = np.unique(ar2)
    end_uniq = time.time()

    ar = np.concatenate((ar1, ar2))

    start_sort = time.time()
    order = ar.argsort(kind="mergesort")
    end_sort = time.time()

    # construct the output index list
    ar = ar[order]
    bool_ar = ar[1:] == ar[:-1]

    ret = np.empty(ar.shape, dtype=bool)
    ret[order] = bool_ar

    if uniq:
        inds = ret[: len(ar1)].nonzero()[0]
    else:
        inds = ret[rev_idx].nonzero()[0]

    if debug:
        print(
            " match1: "
            + str(round(time.time() - start, 2))
            + " sec "
            + "[sort: "
            + str(round(end_sort - start_sort, 2))
            + " sec] "
            + "[uniq: "
            + str(round(end_uniq - start_uniq, 2))
            + " sec]"
        )

    return inds


def match2(ar1, ar2, debug=False):
    """Calculate the common elements between two arrays and their respective indices (unused version 2).

    My alternative version of numpy.in1d with invert=False, which is more similar to calcMatch().
    Return is two ndarrays. The first is indices into ar1, the second is indices into ar2, such
    that ar1[inds1] = ar2[inds2]. Both ar1 and ar2 are assumed to contain unique values. Can be
    used to e.g. crossmatch between two TracerID sets from different snapshots, or between some
    ParentIDs and ParticleIDs of other particle types. The approach is a concatenated mergesort
    of ar1,ar2 combined, therefore O( (N_ar1+N_ar2)*log(N_ar1+N_ar2) ) complexity.
    """
    start = time.time()

    # flatten both arrays (behavior for the first array could be different)
    ar1 = np.asarray(ar1).ravel()
    ar2 = np.asarray(ar2).ravel()

    # make concatenated list of ar1,ar2 and a combined list of indices, and a flag for which array
    # each index belongs to (0=ar1, 1=ar2)
    c = np.concatenate((ar1, ar2))
    ind = np.concatenate((np.arange(ar1.size), np.arange(ar2.size)))
    vec = np.concatenate((np.zeros(ar1.size, dtype="int16"), np.zeros(ar2.size, dtype="int16") + 1))

    # sort combined list
    order = c.argsort(kind="mergesort")

    c = c[order]
    ind = ind[order]
    vec = vec[order]

    # find duplicates in sorted combined list
    firstdup = np.where((c == np.roll(c, -1)) & (vec != np.roll(vec, -1)))[0]

    if firstdup.size == 0:
        return None, None

    dup = np.zeros(firstdup.size * 2, dtype="uint64")
    even = np.arange(firstdup.size, dtype="uint64") * 2

    dup[even] = firstdup
    dup[even + 1] = firstdup + 1

    ind = ind[dup]
    vec = vec[dup]

    inds1 = ind[np.where(vec == 0)]
    inds2 = ind[np.where(vec == 1)]

    if debug:
        if not np.array_equal(ar1[inds1], ar2[inds2]):
            raise Exception("match2 fail")
        print(" match2: " + str(round(time.time() - start, 2)) + " sec")

    return inds1, inds2


@jit(nopython=True, nogil=True, cache=True)
def _match_jit(ar1, ar2, firstSorted=False):
    """Test."""
    assert ar1.ndim == ar2.ndim == 1

    if not firstSorted:
        # need a sorted copy of ar1 to run bisection against
        index = np.argsort(ar1)
        ar1_sorted = ar1[index]
        ar1_sorted_index = np.searchsorted(ar1_sorted, ar2)

        for i in range(ar1_sorted_index.size):  # mode="clip"
            if ar1_sorted_index[i] >= index.size:
                ar1_sorted_index[i] = index.size

        ar1_inds = np.take(index, ar1_sorted_index)
    else:
        # if we can assume ar1 is already sorted, then proceed directly
        ar1_sorted_index = np.searchsorted(ar1, ar2)

        for i in range(ar1_sorted_index.size):  # mode="clip"
            if ar1_sorted_index[i] >= index.size:
                ar1_sorted_index[i] = index.size

        ar1_inds = np.take(np.arange(ar1.size), ar1_sorted_index)

    mask = ar1[ar1_inds] == ar2
    ar2_inds = np.where(mask)[0]
    ar1_inds = ar1_inds[ar2_inds]

    if not len(ar1_inds):
        return None, None

    return ar1_inds, ar2_inds
