"""
util/treeSearch.py
  Adaptive estimation of a smoothing length (radius of sphere enclosing N nearest neighbors) using oct-tree.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import threading
import time

from numba import jit, void, int32
from util.helper import pSplit
from util.sphMap import _NEAREST, _getkernel

#@jit(void(int32,int32,int32,int32,int32[:],int32[:],int32[:],int32[:]))
@jit(nopython=True, nogil=True)
def _updateNodeRecursive(no,sib,NumPart,last,suns,nextnode,next_node,sibling):
    """ Helper routine for calcHsml(), see below. """
    #print(' _updateNodeRecursive(%d,%d,%d,%d)' % (no,sib,NumPart,last))
    pp = 0 #np.int32(0)
    nextsib = 0 #np.int32(0)

    if no >= NumPart:
        if last >= 0:
            if last >= NumPart:
                nextnode[last-NumPart] = no
            else:
                next_node[last] = no

        last = no

        for i in range(8):
            p = suns[i,no-NumPart]

            if p >= 0:
                # check if we have a sibling on the same level
                j = i + 1
                while j < 8:
                    pp = suns[j,no-NumPart]
                    if pp >= 0:
                        break
                    j += 1 # unusual syntax so that j==8 at the end of the loop if we never break

                if j < 8: # yes, we do
                    nextsib = pp
                else:
                    nextsib = sib

                last = _updateNodeRecursive(p,nextsib,NumPart,last,suns,nextnode,next_node,sibling)

        sibling[no-NumPart] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                nextnode[last-NumPart] = no
            else:
                next_node[last] = no

        last = no

    return last # avoid use of global in numba

@jit(nopython=True, nogil=True)
def _constructTree(pos,boxSizeSim,next_node, #father_node,
                   length,center,suns,sibling,nextnode): # Nodes_base
                   #father,maxsoft,s,mass,bitflags): # Nodes_base unused
    """ Core routine for calcHsml(), see below. """
    subnode = 0
    parent  = -1
    lenHalf = 0.0

    # Nodes_base and Nodes are both pointers to the arrays of NODE structs
    # Nodes_base is allocated with size >NumPart, and entries >=NumPart are "internal nodes"
    #  while entries from 0 to NumPart-1 are leafs (actual particles)
    #  Nodes just points to Nodes_base-NumPart (such that Nodes[NumPart]=Nodes_base[0])
    #  Nodes[no]=Nodes_base[no-NumPart]
    xyzMin = np.zeros( 3, dtype=np.float32 )
    xyzMax = np.zeros( 3, dtype=np.float32 )
    
    # select first node
    NumPart = pos.shape[0]
    nFree = NumPart

    # create an empty root node
    if boxSizeSim > 0.0:
        # periodic
        for j in range(3):
            center[j,nFree-NumPart] = 0.5 * boxSizeSim
            length[j] = boxSizeSim
    else:
        # non-periodic
        for j in range(3):
            xyzMin[j] = 1.0e35 # MAX_REAL_NUMBER
            xyzMax[j] = 1.0e35 # MAX_REAL_NUMBER

        for i in range(NumPart):
            for j in range(3):
                if pos[i,j] > xyzMax[j]:
                    xyzMax[j] = pos[i,j]
                if pos[i,j] < xyzMin[j]:
                    xyzMin[j] = pos[i,j]

        # determine maximum extension
        extent = 0.0

        for j in range(3):
            if xyzMax[j] - xyzMin[j] > extent:
                extent = xyzMax[j] - xyzMin[j]

            center[j,nFree-NumPart] = 0.5 * (xyzMin[j] + xyzMax[j])
            length[j] = extent

    # daughter slots of root node all start empty
    for i in range(8):
        suns[i,nFree-NumPart] = -1

    numNodes = 1
    nFree += 1

    # now insert all particles and so construct the tree
    for i in range(NumPart):
        # start at the root node
        no = NumPart

        # insert particle i
        while 1:
            if no >= NumPart: # we are dealing with an internal node
                # to which subnode will this particle belong
                subnode = 0
                ind = no-NumPart

                if pos[i,0] > center[0,ind]:
                    subnode += 1
                if pos[i,1] > center[1,ind]:
                    subnode += 2
                if pos[i,2] > center[2,ind]:
                    subnode += 4

                # get the next node
                nn = suns[subnode,ind]

                if nn >= 0: # ok, something is in the daughter slot already, need to continue
                    parent = no # note: subnode can still be used in the next step of the walk
                    no = nn
                else:
                    # here we have found an empty slot where we can attach the new particle as a leaf
                    suns[subnode,ind] = i
                    break # done for this particle
            else:
                # we try to insert into a leaf with a single particle - need to generate a new internal
                # node at this point, because every leaf is only allowed to contain one particle
                suns[subnode,parent-NumPart] = nFree
                ind1 = parent-NumPart
                ind2 = nFree-NumPart

                length[ind2] = 0.5 * length[ind1]
                lenHalf = 0.25 * length[ind1]

                if subnode & 1: # np.bitwise_and(subnode,1)
                    center[0,ind2] = center[0,ind1] + lenHalf
                else:
                    center[0,ind2] = center[0,ind1] - lenHalf

                if subnode & 2:
                    center[1,ind2] = center[1,ind1] + lenHalf
                else:
                    center[1,ind2] = center[1,ind1] - lenHalf

                if subnode & 4:
                    center[2,ind2] = center[2,ind1] + lenHalf
                else:
                    center[2,ind2] = center[2,ind1] - lenHalf

                for j in range(8):
                    suns[j,ind2] = -1

                # which subnode
                subnode = 0

                if pos[no,0] > center[0,ind2]:
                    subnode += 1
                if pos[no,1] > center[1,ind2]:
                    subnode += 2
                if pos[no,2] > center[2,ind2]:
                    subnode += 4

                #if length[nFree] < 1.0e-3 * Softening:
                    # randomize subnode index, or terminate

                suns[subnode,ind2] = no

                no = nFree # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                assert numNodes < length.shape[0] # Exceed tree allocated size, need to increase and redo.

    # now compute the multipole moments recursively
    last = np.int32(-1)
    #dummy = np.int32(-1)

    last = _updateNodeRecursive(NumPart,-1,NumPart,last,suns,nextnode,next_node,sibling)

    if last >= NumPart:
        nextnode[last-NumPart] = -1
    else:
        next_node[last] = -1

    return numNodes

@jit(nopython=True, nogil=True)
def _treeSearchNumNgb(xyz,h,NumPart,boxSizeSim,pos,next_node,length,center,sibling,nextnode):
    """ Helper routine for calcHsml(), see below. """
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h
    hinv = 1.0 / h

    numNgbInH = 0
    numNgbWeightedInH = 0.0

    # 3D-normalized kernel
    C1 = 2.546479089470  # COEFF_1
    C2 = 15.278874536822 # COEFF_2
    C3 = 5.092958178941  # COEFF_5
    CN = 4.188790204786  # NORM_COEFF (4pi/3)

    # start search
    no = NumPart

    while no >= 0:
        if no < NumPart:
            # single particle
            #print(' single_particle: no=%d to %d' % (no,next_node[no]))
            p = no
            assert next_node[no] != no # Going into infinite loop.
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST( pos[p,0] - xyz[0], boxHalf, boxSizeSim )
            if dx < -h or dx > h:
                continue

            dy = _NEAREST( pos[p,1] - xyz[1], boxHalf, boxSizeSim )
            if dy < -h or dy > h:
                continue

            dz = _NEAREST( pos[p,2] - xyz[2], boxHalf, boxSizeSim )
            if dz < -h or dz > h:
                continue

            # spherical exclusion if we've made it this far
            r2 = dx*dx + dy*dy + dz*dz
            if r2 >= h2:
                continue

            # count
            numNgbInH += 1

            # weighted count
            numNgbWeightedInH += CN * _getkernel(hinv, r2, C1, C2, C3)

        else:
            # internal node
            #print(' internal_node: no=%d to %d if discarded or %d if opened' % (no,sibling[no-NumPart],nextnode[no-NumPart]))
            ind = no-NumPart # struct NODE *this = &Nodes[no];
            no = sibling[ind] # in case the node can be discarded

            #print('  evaluate opening: center [%g %g %g]' % (center[0,ind],center[1,ind],center[2,ind]))

            if _NEAREST( center[0,ind] - xyz[0], boxHalf, boxSizeSim ) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST( center[0,ind] - xyz[0], boxHalf, boxSizeSim ) - 0.5 * length[ind] > h:
                continue

            if _NEAREST( center[1,ind] - xyz[1], boxHalf, boxSizeSim ) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST( center[1,ind] - xyz[1], boxHalf, boxSizeSim ) - 0.5 * length[ind] > h:
                continue

            if _NEAREST( center[2,ind] - xyz[2], boxHalf, boxSizeSim ) + 0.5 * length[ind] < -h:
                continue
            if _NEAREST( center[2,ind] - xyz[2], boxHalf, boxSizeSim ) - 0.5 * length[ind] > h:
                continue

            #print('  open')

            no = nextnode[ind] # we need to open the node

    return numNgbWeightedInH

@jit(nopython=True, nogil=True)
def _treeSearchHsmlSingle(xyz,h_guess,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                          next_node,length,center,sibling,nextnode):
    """ Helper routine for calcHsml(), see below. """
    left  = 0.0
    right = 0.0
    one_third = 1.0 / 3.0

    if h_guess == 0.0:
        h_guess = 1.0

    iter_num = 0

    while 1:
        iter_num += 1

        assert iter_num < 1000 # Convergence failure, too many iterations.

        numNgbInH = _treeSearchNumNgb(xyz,h_guess,NumPart,boxSizeSim,pos,
                                      next_node,length,center,sibling,nextnode)

        #print(' iter=%d h=%g nNGB=%g (left=%g right=%g)' % (iter_num,h_guess,numNgbInH,left,right))

        # success
        if numNgbInH > (nNGB-nNGBDev) and numNgbInH <= (nNGB+nNGBDev):
            break

        # fail, the number of neighbors we found within h_guess is outside bounds
        if left > 0.0 and right > 0.0:
            if right-left < 0.001 * left:
                break # particle is OK

        if numNgbInH < nNGB-nNGBDev:
            left = max(h_guess, left)
        else:
            if right != 0.0:
                if h_guess < right:
                    right = h_guess
            else:
                right = h_guess

        if right > 0.0 and left > 0.0:
            h_guess = np.power( 0.5 * (left*left*left + right*right*right), one_third )
        else:
            assert right > 0.0 or left > 0.0 # Cannot occur that both are zero.

            if right == 0.0 and left > 0.0:
                h_guess *= 1.26
            if right > 0.0 and left == 0.0:
                h_guess /= 1.26

    #print(' found h = %g iter_num = %d' % (h_guess,iter_num))
    return h_guess

@jit(nopython=True, nogil=True)
def _treeSearchHsmlSet(posSearch,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                       next_node,length,center,sibling,nextnode):
    """ Core routine for calcHsml(), see below. """
    numSearch = posSearch.shape[0]

    h_guess = 1.0
    if boxSizeSim > 0.0:
        h_guess = boxSizeSim / NumPart**(1.0/3.0)

    hsml = np.zeros( numSearch, dtype=np.float32 )

    for i in range(numSearch):
        # single ball search using octtree, requesting hsml which enclosed nNGB+/-nNGBDev around xyz
        xyz = posSearch[i,:]
        #print('[%d] xyz %g %g %g' % (i,xyz[0],xyz[1],xyz[2]))

        hsml[i] = _treeSearchHsmlSingle(xyz,h_guess,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                                        next_node,length,center,sibling,nextnode)

        # use previous result as guess for the next (any spatial ordering will greatly help)
        h_guess = hsml[i]

    return hsml

def calcHsml(pos, boxSizeSim, posSearch=None, nNGB=32, nNGBDev=1, nDims=3, treePrec='single', nThreads=1):
    """ Calculate a characteristic 'size' ('smoothing length') given a set of input particle coordinates, 
    where the size is defined as the radius of the sphere (or circle in 2D) enclosing the nNGB nearest 
    neighbors. If posSearch==None, then pos defines both the neighbor and search point sets, otherwise 
    a radius for each element of posSearch is calculated by searching for nearby points in pos.
        
      pos[N,3]/[N,2] : array of 3-coordinates for the particles (or 2-coords for 2D)
      boxSizeSim[1]  : the physical size of the simulation box for periodic wrapping (0=non periodic)
      nNGB           : number of nearest neighbors to search for in order to define HSML
      nNGBDev        : allowed deviation (+/-) from the requested number of neighbors
      nDims          : number of dimensions of simulation (1,2,3), to set SPH kernel coefficients
      nThreads       : do multithreaded calculation (on treefind, while tree construction remains serial)
    """
    # input sanity checks
    assert pos.ndim == 2 and pos.shape[1] in [3], 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert nDims in [3], 'Invalid ndims specification (3D only).'
    assert treePrec in ['single','double'], 'Invalid treePrec specification.'

    # tree allocation and construction
    treeDtype = 'float32' if treePrec == 'single' else 'float64'

    NumPart = pos.shape[0]
    MaxNodes = np.int(1.1*NumPart) + 1

    assert NumPart >= nNGB-nNGBDev, 'Less particles then requested neighbors.'

    length   = np.zeros( MaxNodes, dtype=treeDtype )     # NODE struct member
    center   = np.zeros( (3,MaxNodes), dtype=treeDtype ) # NODE struct member
    #maxsoft  = np.zeros( MaxNodes, dtype=treeDtype )     # NODE struct member
    suns     = np.zeros( (8,MaxNodes), dtype='int32' )    # NODE.u first union member
    #s        = np.zeros( (3,MaxNodes), dtype=treeDtype )  # NODE.u second union member (NODE.u.d struct member)
    #mass     = np.zeros( MaxNodes, dtype=treeDtype )       # (NODE.u.d struct member)
    #bitflags = np.zeros( MaxNodes, dtype='uint32' )        # (NODE.u.d struct member)
    sibling  = np.zeros( MaxNodes, dtype='int32' )         # (NODE.u.d struct member)
    nextnode = np.zeros( MaxNodes, dtype='int32' )         # (NODE.u.d struct member)
    #father   = np.zeros( MaxNodes, dtype='int32' )         # (NODE.u.d struct member)

    NextNode   = np.zeros( NumPart, dtype='int32' )
    #FatherNode = np.zeros( NumPart, dtype='int32' )

    # call JIT compiled kernel
    start_time = time.time()

    numNodes = _constructTree(pos,boxSizeSim,NextNode, #Father,
                              length,center,suns,sibling,nextnode) #,father)
                              #,maxsort,s,mass,bitflags) # unused

    build_done_time = time.time()

    # tree search
    #print('numNodes: %d (MaxNodes=%d)' % (numNodes,MaxNodes))
    #for i in range(numNodes):
    #    print(' [%d] length=%d [%d %d %d %d %d %d %d %d] sibling=%d nextnode=%d' % \
    #      (i, length[i], suns[0,i], suns[1,i], suns[2,i], suns[3,i], suns[4,i], 
    #       suns[5,i], suns[6,i], suns[7,i], sibling[i], nextnode[i]))

    #print('NumPart: %d' % (NumPart))
    #for i in range(NumPart):
    #    print(' [%d] next_node = %d' % (i,NextNode[i]))

    if posSearch is None:
        posSearch = pos # set search coordinates as a view onto the same pos used to make the tree

    hsml = _treeSearchHsmlSet(posSearch,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                              NextNode,length,center,sibling,nextnode)

    done_time = time.time()

    print(' calcHsml(): tree construction took [%g] sec, search took [%g] sec.' % \
        (build_done_time-start_time, done_time-build_done_time))

    return hsml

def benchmark():
    """ Benchmark performance of calcHsml(). """
    np.random.seed(424242)
    from cosmo.load import snapshotSubset
    from util.simParams import simParams

    # config data
    if 1:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 80000

        posDtype = 'float32'
        pos = np.random.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)
        #pos = np.zeros( (nPts,3), dtype=posDtype )
        #pos[0,:] = [10,20,30]
        #pos[1,:] = [40,45,10]

    if 0:
        # load some gas in a box
        sP = simParams(res=128, run='tracer', redshift=0.0)
        pos = snapshotSubset(sP, 'gas', 'pos')

    # config
    nNGB = 32
    nNGBDev = 1
    treePrec = 'single'
    nThreads = 1

    # warmup (compile)
    hsml = calcHsml(pos,sP.boxSize,nNGB=nNGB,nNGBDev=nNGBDev,treePrec=treePrec,nThreads=nThreads)

    # calculate and time
    start_time = time.time()
    nLoops = 2

    for i in np.arange(nLoops):
        hsml = calcHsml(pos,sP.boxSize,nNGB=nNGB,nNGBDev=nNGBDev,treePrec=treePrec,nThreads=nThreads)

    print('%d estimates of HSMLs took [%g] sec on avg' % (nLoops,(time.time()-start_time)/nLoops))
    print('hsml min %g max %g mean %g' % (np.max(hsml), np.min(hsml), np.mean(hsml)))
