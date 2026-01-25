"""
Adaptive estimation of a smoothing length (radius of sphere enclosing N nearest neighbors) using oct-tree.
"""

import threading
import time

import numpy as np
from numba import jit

from ..util.helper import pSplit
from ..util.sphMap import _NEAREST, _getkernel


@jit(nopython=True, nogil=True)  # , cache=True)
def _updateNodeRecursive(no, sib, NumPart, last, suns, nextnode, next_node, sibling):
    """Helper routine for calcHsml(), see below."""
    pp = 0
    nextsib = 0

    if no >= NumPart:
        if last >= 0:
            if last >= NumPart:
                nextnode[last - NumPart] = no
            else:
                next_node[last] = no

        last = no

        for i in range(8):
            p = suns[i, no - NumPart]

            if p >= 0:
                # check if we have a sibling on the same level
                j = i + 1
                while j < 8:
                    pp = suns[j, no - NumPart]
                    if pp >= 0:
                        break
                    j += 1  # unusual syntax so that j==8 at the end of the loop if we never break

                if j < 8:  # yes, we do
                    nextsib = pp
                else:
                    nextsib = sib

                last = _updateNodeRecursive(p, nextsib, NumPart, last, suns, nextnode, next_node, sibling)

        sibling[no - NumPart] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                nextnode[last - NumPart] = no
            else:
                next_node[last] = no

        last = no

    return last  # avoid use of global in numba


@jit(nopython=True, nogil=True)  # , cache=True)
def _constructTree(pos, boxSizeSim, next_node, length, center, suns, sibling, nextnode):
    """Core routine for calcHsml(), see below."""
    subnode = 0
    parent = -1
    lenHalf = 0.0

    # Nodes_base and Nodes are both pointers to the arrays of NODE structs
    # Nodes_base is allocated with size >NumPart, and entries >=NumPart are "internal nodes"
    #  while entries from 0 to NumPart-1 are leafs (actual particles)
    #  Nodes just points to Nodes_base-NumPart (such that Nodes[no]=Nodes_base[no-NumPart])
    xyzMin = np.zeros(3, dtype=np.float32)
    xyzMax = np.zeros(3, dtype=np.float32)

    # select first node
    NumPart = pos.shape[0]
    nFree = NumPart

    # create an empty root node
    if boxSizeSim > 0.0:
        # periodic
        for j in range(3):
            center[j, nFree - NumPart] = 0.5 * boxSizeSim
        length[0] = boxSizeSim
    else:
        # non-periodic
        for j in range(3):
            xyzMin[j] = 1.0e35  # MAX_REAL_NUMBER
            xyzMax[j] = -1.0e35  # MAX_REAL_NUMBER

        for i in range(NumPart):
            for j in range(3):
                if pos[i, j] > xyzMax[j]:
                    xyzMax[j] = pos[i, j]
                if pos[i, j] < xyzMin[j]:
                    xyzMin[j] = pos[i, j]

        # determine maximum extension
        extent = 0.0

        for j in range(3):
            if xyzMax[j] - xyzMin[j] > extent:
                extent = xyzMax[j] - xyzMin[j]

            center[j, nFree - NumPart] = 0.5 * (xyzMin[j] + xyzMax[j])
        length[0] = extent

    # daughter slots of root node all start empty
    for i in range(8):
        suns[i, nFree - NumPart] = -1

    numNodes = 1
    nFree += 1

    # now insert all particles and so construct the tree
    for i in range(NumPart):
        # start at the root node
        no = NumPart

        # insert particle i
        while 1:
            if no >= NumPart:  # we are dealing with an internal node
                # to which subnode will this particle belong
                subnode = 0
                ind = no - NumPart

                if pos[i, 0] > center[0, ind]:
                    subnode += 1
                if pos[i, 1] > center[1, ind]:
                    subnode += 2
                if pos[i, 2] > center[2, ind]:
                    subnode += 4

                # get the next node
                nn = suns[subnode, ind]

                if nn >= 0:  # ok, something is in the daughter slot already, need to continue
                    parent = no  # note: subnode can still be used in the next step of the walk
                    no = nn
                else:
                    # here we have found an empty slot where we can attach the new particle as a leaf
                    suns[subnode, ind] = i
                    break  # done for this particle
            else:
                # we try to insert into a leaf with a single particle - need to generate a new internal
                # node at this point, because every leaf is only allowed to contain one particle
                suns[subnode, parent - NumPart] = nFree
                ind1 = parent - NumPart
                ind2 = nFree - NumPart

                length[ind2] = 0.5 * length[ind1]
                lenHalf = 0.25 * length[ind1]

                if subnode & 1:
                    center[0, ind2] = center[0, ind1] + lenHalf
                else:
                    center[0, ind2] = center[0, ind1] - lenHalf

                if subnode & 2:
                    center[1, ind2] = center[1, ind1] + lenHalf
                else:
                    center[1, ind2] = center[1, ind1] - lenHalf

                if subnode & 4:
                    center[2, ind2] = center[2, ind1] + lenHalf
                else:
                    center[2, ind2] = center[2, ind1] - lenHalf

                for j in range(8):
                    suns[j, ind2] = -1

                # which subnode
                subnode = 0

                if pos[no, 0] > center[0, ind2]:
                    subnode += 1
                if pos[no, 1] > center[1, ind2]:
                    subnode += 2
                if pos[no, 2] > center[2, ind2]:
                    subnode += 4

                if length[ind2] < 1e-4:
                    # may have particles at identical locations, in which case randomize the subnode
                    # index to put the particle into a different leaf (happens well below the
                    # gravitational softening scale)
                    subnode = int(np.random.rand() * 8)
                    subnode = max(subnode, 7)

                suns[subnode, ind2] = no

                no = nFree  # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                if numNodes >= length.shape[0]:
                    # exceeding tree allocated size, need to increase and redo
                    return -1

    # now compute the (sibling,nextnode,next_node) recursively
    last = np.int32(-1)

    last = _updateNodeRecursive(NumPart, -1, NumPart, last, suns, nextnode, next_node, sibling)

    if last >= NumPart:
        nextnode[last - NumPart] = -1
    else:
        next_node[last] = -1

    return numNodes


@jit(nopython=True, nogil=True, cache=True)
def _treeSearch(xyz, h, NumPart, boxSizeSim, pos, next_node, length, center, sibling, nextnode, quant, op):
    """Helper routine for calcHsml(), see below."""
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h
    hinv = 1.0 / h

    numNgbInH = 0
    numNgbWeightedInH = 0.0

    quantResult = 0.0
    if op == 2:  # max
        quantResult = -np.inf
    if op == 3:  # min
        quantResult = np.inf

    # 3D-normalized kernel
    C1 = 2.546479089470  # COEFF_1
    C2 = 15.278874536822  # COEFF_2
    C3 = 5.092958178941  # COEFF_5
    CN = 4.188790204786  # NORM_COEFF (4pi/3)

    # start search
    no = NumPart

    while no >= 0:
        if no < NumPart:
            # single particle
            assert next_node[no] != no  # Going into infinite loop.

            p = no
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST(pos[p, 0] - xyz[0], boxHalf, boxSizeSim)
            if dx < -h or dx > h:
                continue

            dy = _NEAREST(pos[p, 1] - xyz[1], boxHalf, boxSizeSim)
            if dy < -h or dy > h:
                continue

            dz = _NEAREST(pos[p, 2] - xyz[2], boxHalf, boxSizeSim)
            if dz < -h or dz > h:
                continue

            # spherical exclusion if we've made it this far
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= h2:
                continue

            # count
            numNgbInH += 1

            # weighted count
            kval = _getkernel(hinv, r2, C1, C2, C3)
            numNgbWeightedInH += CN * kval

            # reduction operation on particle quantity
            if op == 1:  # sum
                quantResult += quant[p]
            if op == 2:  # max
                if quant[p] > quantResult:
                    quantResult = quant[p]
            if op == 3:  # min
                if quant[p] < quantResult:
                    quantResult = quant[p]
            if op == 4:  # unweighted mean
                quantResult += quant[p]
            if op == 5:  # kernel-weighted mean
                quantResult += quant[p] * kval
            if op == 6:  # count
                quantResult += 1

        else:
            # internal node
            ind = no - NumPart
            no = sibling[ind]  # in case the node can be discarded

            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            no = nextnode[ind]  # we need to open the node

    if op == 4:  # mean
        if numNgbInH != 0.0:
            quantResult /= numNgbInH
    if op == 5:  # kernel-weighted mean
        if numNgbWeightedInH != 0.0:
            quantResult /= numNgbWeightedInH

    return numNgbInH, numNgbWeightedInH, quantResult


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchIndices(xyz, h, NumPart, boxSizeSim, pos, posMask, next_node, length, center, sibling, nextnode):
    """Helper routine for calcParticleIndices(), see below."""
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h

    # allocate, unfortunately unclear how safe we have to be here
    numNgbInH = 0
    inds = np.empty(NumPart, dtype=np.int64)

    # start search
    no = NumPart

    while no >= 0:
        if no < NumPart:
            # single particle
            assert next_node[no] != no  # Going into infinite loop.

            p = no
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST(pos[p, 0] - xyz[0], boxHalf, boxSizeSim)
            if dx < -h or dx > h:
                continue

            dy = _NEAREST(pos[p, 1] - xyz[1], boxHalf, boxSizeSim)
            if dy < -h or dy > h:
                continue

            dz = _NEAREST(pos[p, 2] - xyz[2], boxHalf, boxSizeSim)
            if dz < -h or dz > h:
                continue

            # spherical exclusion if we've made it this far
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= h2:
                continue

            if posMask[p] == 0:
                continue

            # count
            inds[numNgbInH] = p
            numNgbInH += 1

        else:
            # internal node
            ind = no - NumPart
            no = sibling[ind]  # in case the node can be discarded

            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            no = nextnode[ind]  # we need to open the node

    if numNgbInH > 0:
        inds = inds[0:numNgbInH]
        return inds

    return None


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchNearest(xyz, h, NumPart, boxSizeSim, pos, posMask, next_node, length, center, sibling, nextnode):
    """Helper routine for calcParticleIndices(), see below."""
    boxHalf = 0.5 * boxSizeSim
    h2 = h * h

    # allocate, unfortunately unclear how safe we have to be here
    min_index = -1
    min_dist2 = np.inf

    # start search
    no = NumPart

    while no >= 0:
        if no < NumPart:
            # single particle
            assert next_node[no] != no  # Going into infinite loop.

            p = no
            no = next_node[no]

            dx = _NEAREST(pos[p, 0] - xyz[0], boxHalf, boxSizeSim)
            dy = _NEAREST(pos[p, 1] - xyz[1], boxHalf, boxSizeSim)
            dz = _NEAREST(pos[p, 2] - xyz[2], boxHalf, boxSizeSim)

            r2 = dx * dx + dy * dy + dz * dz

            if r2 == 0:
                continue  # no self

            if r2 >= h2:
                continue

            if posMask[p] == 0:
                continue  # masked

            # new closest?
            if r2 < min_dist2:
                min_dist2 = r2
                min_index = p

        else:
            # internal node
            ind = no - NumPart
            no = sibling[ind]  # in case the node can be discarded

            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                continue

            no = nextnode[ind]  # we need to open the node

    return min_index, min_dist2


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchNearestSingle(xyz, pos, boxSizeSim, next_node, length, center, sibling, nextnode, h=1.0):
    """Iterate on tree-search until we have at least one neighbor, return nearest."""
    NumPart = pos.shape[0]
    loc_index = -1
    min_dist2 = np.inf
    iter_num = 0
    boxHalf = 0.5 * boxSizeSim

    while loc_index == -1:
        # loc_index, loc_dist2 = _treeSearchNearest(xyz,h_guess,NumPart,boxSizeSim,pos,posMask,
        #                                          NextNode,length,center,sibling,nextnode)

        h2 = h * h

        # start search
        no = NumPart

        while no >= 0:
            if no < NumPart:
                # single particle
                assert next_node[no] != no  # Going into infinite loop.

                p = no
                no = next_node[no]

                dx = _NEAREST(pos[p, 0] - xyz[0], boxHalf, boxSizeSim)
                dy = _NEAREST(pos[p, 1] - xyz[1], boxHalf, boxSizeSim)
                dz = _NEAREST(pos[p, 2] - xyz[2], boxHalf, boxSizeSim)

                r2 = dx * dx + dy * dy + dz * dz

                if r2 == 0:
                    continue  # no self

                if r2 >= h2:
                    continue

                # new closest?
                if r2 < min_dist2:
                    min_dist2 = r2
                    loc_index = p
            else:
                # internal node
                ind = no - NumPart
                no = sibling[ind]  # in case the node can be discarded

                if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                    continue
                if _NEAREST(center[0, ind] - xyz[0], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                    continue

                if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                    continue
                if _NEAREST(center[1, ind] - xyz[1], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                    continue

                if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) + 0.5 * length[ind] < -h:
                    continue
                if _NEAREST(center[2, ind] - xyz[2], boxHalf, boxSizeSim) - 0.5 * length[ind] > h:
                    continue

                no = nextnode[ind]  # we need to open the node

        h *= 2.0
        iter_num += 1

        if iter_num > 10000:
            print("ERROR: Failed to converge.")
            break

    return loc_index, np.sqrt(min_dist2)


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchHsmlIterate(
    xyz, h_guess, nNGB, nNGBDev, NumPart, boxSizeSim, pos, next_node, length, center, sibling, nextnode, weighted_num
):
    """Helper routine for calcHsml(), see below."""
    left = 0.0
    right = 0.0
    one_third = 1.0 / 3.0

    if h_guess == 0.0:
        h_guess = 1.0

    iter_num = 0
    dummy_in = -1
    dummy_in2 = np.zeros(1, dtype=np.int32)

    while 1:
        iter_num += 1

        assert iter_num < 1000  # Convergence failure, too many iterations.

        numNgbInH, numNgbWeightedInH, dummy = _treeSearch(
            xyz, h_guess, NumPart, boxSizeSim, pos, next_node, length, center, sibling, nextnode, dummy_in2, dummy_in
        )

        if iter_num > 990:
            print("Convergence failure in _treeSearchHsmlIterate()", iter_num, xyz, h_guess, numNgbInH, left, right)

        if iter_num == 999:
            # TNG-Cluster snap == 77 have e.g. 39 stars at 'identical' positions [558761.75 , 555277.75 ,  16362.459]
            print(" break")
            h_guess = 0.01
            break

        # looking for h enclosing the SPH kernel weighted number, instead of the actual number?
        if weighted_num:
            numNgbInH = numNgbWeightedInH

        # success
        if numNgbInH > (nNGB - nNGBDev) and numNgbInH <= (nNGB + nNGBDev):
            break

        # fail, the number of neighbors we found within h_guess is outside bounds
        if left > 0.0 and right > 0.0:
            if right - left < 0.001 * left:
                break  # particle is OK

        if numNgbInH < nNGB:  # -nNGBDev:
            left = max(h_guess, left)
        else:
            if right != 0.0:
                if h_guess < right:
                    right = h_guess
            else:
                right = h_guess

        if right > 0.0 and left > 0.0:
            h_guess = np.power(0.5 * (left * left * left + right * right * right), one_third)
        else:
            assert right > 0.0 or left > 0.0  # Cannot occur that both are zero.

            if right == 0.0 and left > 0.0:
                h_guess *= 1.26
            if right > 0.0 and left == 0.0:
                h_guess /= 1.26

    return h_guess


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchHsmlSet(
    posSearch, ind0, ind1, nNGB, nNGBDev, boxSizeSim, pos, next_node, length, center, sibling, nextnode, weighted_num
):
    """Core routine for calcHsml(), see below."""
    numSearch = ind1 - ind0 + 1
    NumPart = pos.shape[0]

    h_guess = 1.0
    if boxSizeSim > 0.0:
        h_guess = boxSizeSim / NumPart ** (1.0 / 3.0)

    hsml = np.zeros(numSearch, dtype=np.float32)

    for i in range(numSearch):
        # single ball search using octtree, requesting hsml which enclosed nNGB+/-nNGBDev around xyz
        xyz = posSearch[ind0 + i, :]

        hsml[i] = _treeSearchHsmlIterate(
            xyz,
            h_guess,
            nNGB,
            nNGBDev,
            NumPart,
            boxSizeSim,
            pos,
            next_node,
            length,
            center,
            sibling,
            nextnode,
            weighted_num,
        )

        # use previous result as guess for the next (any spatial ordering will greatly help)
        h_guess = hsml[i]

    return hsml


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchNearestIterate(
    posSearch, ind0, ind1, boxSizeSim, pos, posMask, next_node, length, center, sibling, nextnode
):
    """Core routine for calcHsml(), see below."""
    numSearch = ind1 - ind0 + 1
    NumPart = pos.shape[0]

    h_guess = 1.0  # i.e. code units
    if boxSizeSim > 0.0:
        h_guess = boxSizeSim / NumPart ** (1.0 / 3.0) * 0.1

    dists = np.zeros(numSearch, dtype=np.float32)
    indices = np.zeros(numSearch, dtype=np.int64)

    for i in range(numSearch):
        # iterate ball search using octtree until we have >= 1 neighbor result
        xyz = posSearch[ind0 + i, :]

        loc_index = -1
        iter_num = 0

        while loc_index == -1:
            loc_index, loc_dist2 = _treeSearchNearest(
                xyz, h_guess, NumPart, boxSizeSim, pos, posMask, next_node, length, center, sibling, nextnode
            )
            # print(iter_num,loc_index,np.sqrt(loc_dist2),h_guess)
            h_guess *= 2.0
            iter_num += 1

            if iter_num > 10000:
                print("ERROR: Failed to converge.")
                break

        # store result
        dists[i] = np.sqrt(loc_dist2)
        indices[i] = loc_index

        # use previous result as guess for the next (any spatial ordering will greatly help)
        h_guess = dists[i]

    return dists, indices


@jit(nopython=True, nogil=True, cache=True)
def _treeSearchQuantReduction(
    posSearch, hsml, ind0, ind1, boxSizeSim, pos, quant, opnum, next_node, length, center, sibling, nextnode
):
    """Core routine for calcQuantReduction(), see below."""
    numSearch = ind1 - ind0 + 1
    NumPart = pos.shape[0]

    result = np.zeros(numSearch, dtype=np.float32)

    for i in range(numSearch):
        # single ball search using octtree, requesting reduction of a given type on the
        # quantity of all the particles within a fixed distance hsml around xyz
        xyz = posSearch[ind0 + i, :]
        h = hsml[ind0 + i] if hsml.shape[0] > 1 else hsml[0]  # variable or constant

        _, _, result[i] = _treeSearch(
            xyz, h, NumPart, boxSizeSim, pos, next_node, length, center, sibling, nextnode, quant, opnum
        )

    return result


def buildFullTree(pos, boxSizeSim, treePrec=None, verbose=False):
    """Helper. See below."""
    treePrecs = {"single": "float32", "double": "float64", "np32": np.float32, "np64": np.float64}
    intType = "int64"  # can be int64, or could be made automatic if MaxNodes is too large

    if treePrec is None:
        treePrec = pos.dtype
    if treePrec in treePrecs.keys():
        treePrec = treePrecs[treePrec]
    assert treePrec in treePrecs.values()

    start_time = time.time()

    NumPart = pos.shape[0]
    NextNode = np.zeros(NumPart, dtype=intType)

    # tree allocation and construction (iterate in case we need to re-allocate for larger number of nodes)
    for num_iter in range(10):
        # allocate
        MaxNodes = int((num_iter**2 + 0.7) * NumPart) + 1
        if MaxNodes < 1000:
            MaxNodes = 1000

        assert MaxNodes < np.iinfo(intType).max, "Too many points to make tree with int32 dtype."

        length = np.zeros(MaxNodes, dtype=treePrec)  # NODE struct member
        center = np.zeros((3, MaxNodes), dtype=treePrec)  # NODE struct member
        suns = np.zeros((8, MaxNodes), dtype=intType)  # NODE.u first union member
        sibling = np.zeros(MaxNodes, dtype=intType)  # NODE.u second union member (NODE.u.d member)
        nextnode = np.zeros(MaxNodes, dtype=intType)  # NODE.u second union member (NODE.u.d member)

        # construct: call JIT compiled kernel
        numNodes = _constructTree(pos, boxSizeSim, NextNode, length, center, suns, sibling, nextnode)

        if numNodes > 0:
            break

        print(" tree: increase alloc %g to %g and redo..." % (num_iter + 0.7, num_iter + 1.7), flush=True)

    assert numNodes > 0, "Tree: construction failed!"
    if verbose:
        print(" tree: construction took [%g] sec." % (time.time() - start_time))

    # memory optimization: subset arrays to used portions
    length = length[0:numNodes]
    center = center[0:numNodes]
    sibling = sibling[0:numNodes]
    nextnode = nextnode[0:numNodes]

    return NextNode, length, center, sibling, nextnode


def calcHsml(
    pos,
    boxSizeSim,
    posSearch=None,
    posMask=None,
    nNGB=32,
    nNGBDev=1,
    nDims=3,
    weighted_num=False,
    treePrec="single",
    tree=None,
    nThreads=32,
    nearest=False,
    verbose=False,
):
    """Calculate a characteristic size (i.e. smoothing length) given a set of input particle coordinates.

    The size is defined as the radius of the sphere (or circle in 2D) enclosing the nNGB nearest
    neighbors. If posSearch==None, then pos defines both the neighbor and search point sets, otherwise
    a radius for each element of posSearch is calculated by searching for nearby points in pos.

    Args:
      pos (ndarray[float][N,3]/[N,2]): array of 3-coordinates for the particles (or 2-coords for 2D).
      boxSizeSim (float): the physical size of the simulation box for periodic wrapping (0=non periodic).
      posSearch (ndarray[float][M,3]): search sites.
      posMask (ndarray[bool][N]): if not None, bool mask, only True entries are considered in search.
      nNGB (int): number of nearest neighbors to search for in order to define HSML.
      nNGBDev (int): allowed deviation (+/-) from the requested number of neighbors.
      nDims (int): number of dimensions of simulation (1,2,3), to set SPH kernel coefficients.
      weighted_num (bool): if True, use the SPH kernel weighted number of neighbors, instead of real number.
      treePrec (str): construct the tree using 'single' or 'double' precision for coordinates.
      tree (list or None): if not None, should be a list of all the needed tree arrays (pre-computed),
        i.e the exact return of :py:func`~util.treeSearch.buildFullTree`.
      nThreads (int): do multithreaded calculation (on treefind, while tree construction remains serial).
      nearest (bool): if True, then instead of returning hsml values based on nNGB, return indices and
        distances to single closest match only.
      verbose (bool): print timing and warning messages.

    Returns:
      ndarray[float or int]: derived smoothing length for each input point (if not ``nearest``). Instead if
      ``nearest == True``, then a 2-tuple of ``(dists,indices)``.
    """
    # input sanity checks
    treeDims = [3]

    assert pos.ndim == 2 and pos.shape[1] in treeDims, "Strange dimensions of pos."
    assert pos.dtype in [np.float32, np.float64], "pos not in float32/64."
    assert nDims in treeDims, "Invalid ndims specification (3D only)."
    if posMask is not None:
        assert nearest  # otherwise generalize to also work for hsml

    # handle small inputs
    if pos.shape[0] < nNGB - nNGBDev:
        nNGBDev = nNGB - pos.shape[0] + 1
        if verbose:
            print("WARNING: Less particles than requested neighbors. Increasing nNGBDev to [%d]!" % nNGBDev)

    if treePrec == "single" and pos.dtype == np.float64:
        treePrec = "float64"

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos, boxSizeSim, treePrec, verbose=verbose)
    else:
        NextNode, length, center, sibling, nextnode = tree  # split out list

    if posSearch is None:
        posSearch = pos  # set search coordinates as a view onto the same pos used to make the tree
    if posMask is None:
        posMask = np.ones(pos.shape[0], dtype="bool")

    if posSearch.shape[0] < nThreads:
        nThreads = 1
        if verbose:
            print("WARNING: Less particles than requested threads. Just running in serial.")

    start_time = time.time()

    # single threaded?
    # ----------------
    if nThreads == 1:
        ind0 = 0
        ind1 = posSearch.shape[0] - 1

        if nearest:
            dists, indices = _treeSearchNearestIterate(
                posSearch, ind0, ind1, boxSizeSim, pos, posMask, NextNode, length, center, sibling, nextnode
            )
            if verbose:
                print(" calcHsml(): search took [%g] sec (serial)." % (time.time() - start_time))
            return dists, indices
        else:
            hsml = _treeSearchHsmlSet(posSearch, ind0, ind1, nNGB, nNGBDev, boxSizeSim, pos, NextNode, length,
                                      center, sibling, nextnode, weighted_num)  # fmt: skip

            if verbose:
                print(" calcHsml(): search took [%g] sec (serial)." % (time.time() - start_time))
            return hsml

    # else, multithreaded
    # -------------------
    class searchThread(threading.Thread):
        """Subclass Thread() to provide local storage (hsml).

        Performance note (Ody2): this algorithm with the serial overhead of the tree construction has ~55% scaling
        effeciency to 16 threads (~8x speedup), drops to ~32% effeciency at 32 threads (~10x speedup).
        """

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # determine local slice (this is a view instead of a copy, even better)
            searchInds = np.arange(posSearch.shape[0])
            inds = pSplit(searchInds, nThreads, threadNum)

            self.ind0 = inds[0]
            self.ind1 = inds[-1]

            # copy other parameters (non-self inputs to _calc() appears to prevent GIL release)
            self.nNGB = nNGB
            self.nNGBDev = nNGBDev
            self.boxSizeSim = boxSizeSim
            self.weighted_num = weighted_num

            # create views to other arrays
            self.posSearch = posSearch
            self.posMask = posMask
            self.pos = pos

            self.NextNode = NextNode
            self.length = length
            self.center = center
            self.sibling = sibling
            self.nextnode = nextnode

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            if nearest:
                self.dists, self.indices = _treeSearchNearestIterate(self.posSearch, self.ind0, self.ind1,
                                                                     self.boxSizeSim, self.pos, self.posMask,
                                                                     self.NextNode, self.length, self.center,
                                                                     self.sibling, self.nextnode)  # fmt: skip
            else:
                self.hsml = _treeSearchHsmlSet(self.posSearch, self.ind0, self.ind1, self.nNGB, self.nNGBDev,
                                               self.boxSizeSim, self.pos, self.NextNode, self.length, self.center,
                                               self.sibling, self.nextnode, self.weighted_num)  # fmt: skip

    # create threads
    threads = [searchThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # allocate master return grids
    if nearest:
        indices = np.zeros(posSearch.shape[0], dtype=np.int64)
        dists = np.zeros(posSearch.shape[0], dtype=np.float32)
    else:
        hsml = np.zeros(posSearch.shape[0], dtype=np.float32)

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        if nearest:
            dists[thread.ind0 : thread.ind1 + 1] = thread.dists
            indices[thread.ind0 : thread.ind1 + 1] = thread.indices
        else:
            hsml[thread.ind0 : thread.ind1 + 1] = thread.hsml

    if verbose:
        print(" calcHsml(): search took [%g] sec (nThreads=%d)." % (time.time() - start_time, nThreads))

    if nearest:
        return dists, indices
    return hsml


def calcQuantReduction(pos, quant, hsml, op, boxSizeSim, posSearch=None, treePrec="single", tree=None, nThreads=32):
    """Reduction operation on all particles/cells within a fixed search distance around each position.

    Calculate a reduction of a given quantity on all the particles within a fixed search
    distance hsml around each pos. If posSearch==None, then pos defines both the neighbor and search point
    sets, otherwise a reduction at the location of each posSearch is calculated by searching for nearby points in pos.

    Args:
      pos (ndarray[float][N,3]/[N,2]): array of 3-coordinates for the particles (or 2-coords for 2D).
      quant (ndarray[float][N]): array of quantity values (i.e. mass, temperature).
      hsml (ndarray[float][1 or N]): array of search distances, or scalar value if constant.
      op (str): 'min', 'max', 'mean', 'kernel_mean', 'sum', 'count'.
      boxSizeSim (float): the physical size of the simulation box for periodic wrapping (0=non periodic).
      posSearch (ndarray[float][N,3]): search coordinates (optional).
      tree (list or None): if not None, should be a list of all the needed tree arrays (pre-computed),
                        i.e the exact return of buildFullTree().
      treePrec (str): construct the tree using 'single' or 'double' precision for coordinates.
      nThreads (int): do multithreaded calculation (on treefind, while tree construction remains serial).

    Returns:
      ndarray[float]: reduction operation applied for each input.
    """
    # input sanity checks
    ops = {"sum": 1, "max": 2, "min": 3, "mean": 4, "kernel_mean": 5, "count": 6}
    treeDims = [3]

    if isinstance(hsml, float):
        hsml = [hsml]
    hsml = np.array(hsml, dtype="float32")

    assert pos.ndim == 2 and pos.shape[1] in treeDims, "Strange dimensions of pos."
    assert pos.dtype in [np.float32, np.float64], "pos not in float32/64."
    assert pos.shape[1] in treeDims, "Invalid ndims specification (3D only)."
    assert quant.ndim == 1 and quant.size == pos.shape[0], "Strange quant shape."
    assert op in ops.keys(), "Unrecognized reduction operation."

    if posSearch is None:
        assert hsml.size in [1, quant.size], "Strange hsml shape."
    else:
        assert hsml.size in [1, posSearch.shape[0]], "Strange hsml shape."

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos, boxSizeSim, treePrec)
    else:
        NextNode, length, center, sibling, nextnode = tree  # split out list

    if posSearch is None:
        posSearch = pos  # set search coordinates as a view onto the same pos used to make the tree

    if posSearch.shape[0] < nThreads:
        nThreads = 1
        print("WARNING: Less particles than requested threads. Just running in serial.")

    opnum = ops[op]

    # single threaded?
    # ----------------
    if nThreads == 1:
        ind0 = 0
        ind1 = posSearch.shape[0] - 1

        result = _treeSearchQuantReduction(
            posSearch, hsml, ind0, ind1, boxSizeSim, pos, quant, opnum, NextNode, length, center, sibling, nextnode
        )

        return result

    # else, multithreaded
    # -------------------
    class searchThread(threading.Thread):
        """Subclass Thread() to provide local storage."""

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # determine local slice (this is a view instead of a copy, even better)
            searchInds = np.arange(posSearch.shape[0])
            inds = pSplit(searchInds, nThreads, threadNum)

            self.ind0 = inds[0]
            self.ind1 = inds[-1]

            # copy other parameters (non-self inputs to _calc() appears to prevent GIL release)
            self.boxSizeSim = boxSizeSim
            self.opnum = opnum

            # create views to other arrays
            self.posSearch = posSearch
            self.pos = pos
            self.hsml = hsml
            self.quant = quant

            self.NextNode = NextNode
            self.length = length
            self.center = center
            self.sibling = sibling
            self.nextnode = nextnode

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            self.result = _treeSearchQuantReduction(self.posSearch, self.hsml, self.ind0, self.ind1, self.boxSizeSim,
                                                    self.pos, self.quant, self.opnum, self.NextNode, self.length,
                                                    self.center, self.sibling, self.nextnode)  # fmt: skip

    # create threads
    threads = [searchThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # allocate master return grids
    result = np.zeros(posSearch.shape[0], dtype=np.float32)

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        result[thread.ind0 : thread.ind1 + 1] = thread.result

    # print(' calcQuantReduction(): took [%g] sec.' % (time.time()-build_done_time))
    return result


def calcParticleIndices(pos, posSearch, hsmlSearch, boxSizeSim, posMask=None, treePrec="single", tree=None):
    """Find and return the actual particle indices (indexing pos, hsml) within the search radii of posSearch locations.

    Serial by construction, since we do only one search.

    Args:
      pos (ndarray[float][N,3]/[N,2]): array of 3-coordinates for the particles (or 2-coords for 2D).
      posSearch (ndarray[float][3]): search position.
      hsmlSearch (float): search distance.
      boxSizeSim (float): the physical size of the simulation box for periodic wrapping (0=non periodic).
      posMask (ndarray[bool][N]): if not None, then only True entries are considered in the search.
      treePrec (str): construct the tree using 'single' or 'double' precision for coordinates.
      tree (list or None): if not None, should be a list of all the needed tree arrays (pre-computed),
                        i.e the exact return of buildFullTree().

    Returns:
      ndarray[int]: list of indices into `pos` of the neighbors within the search distance of
      the search position.
    """
    # input sanity checks
    treeDims = [3]

    if isinstance(hsmlSearch, int):
        hsmlSearch = float(hsmlSearch)
    assert isinstance(hsmlSearch, (float, np.float32))
    hsmlSearch = np.array(hsmlSearch, dtype="float32")

    assert pos.ndim == 2 and pos.shape[1] in treeDims, "Strange dimensions of pos."
    assert pos.dtype in [np.float32, np.float64], "pos not in float32/64."
    assert pos.shape[1] in treeDims, "Invalid ndims specification (3D only)."

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos, boxSizeSim, treePrec)
    else:
        NextNode, length, center, sibling, nextnode = tree  # split out list

    # single threaded
    NumPart = pos.shape[0]

    if posMask is None:
        posMask = np.ones(pos.shape[0], dtype="bool")

    result = _treeSearchIndices(
        posSearch, hsmlSearch, NumPart, boxSizeSim, pos, posMask, NextNode, length, center, sibling, nextnode
    )

    return result
