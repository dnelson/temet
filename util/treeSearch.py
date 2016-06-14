"""
util/treeSearch.py
  Adaptive estimation of a smoothing length (radius of sphere enclosing N nearest neighbors) using oct-tree.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import threading
from numba import jit, void, int32
from util.helper import pSplit

#@jit(void(int32,int32,int32,int32,int32[:],int32[:],int32[:],int32[:]))
@jit(nopython=True, nogil=True)
def _updateNodeRecursive(no,sib,NumPart,last,suns,nextnode,next_node,sibling):
    """ Helper routine for calcHsml(), see below. """
    #print(' _updateNodeRecursive(%d,%d)' % (no,sib))
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
                for j in range(i+1,8):
                    pp = suns[j,no-NumPart]
                    if pp >= 0:
                        break

                    if j < 8: # yes, we do
                        nextsib = pp
                    else:
                        nextsib = sib

                    #p = np.int32(p)
                    #nextsib = np.int32(nextsib)
                    #NumPart = np.int32(NumPart)
                    #last = np.int32(last)

                    _updateNodeRecursive(p,nextsib,NumPart,last,suns,nextnode,next_node,sibling)

        sibling[no-NumPart] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                nextnode[last-NumPart] = no
            else:
                next_node[last] = no

        last = no

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
    #nfreep = Nodes[nFree]

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

                if pos[i,0] > center[0,no-NumPart]:
                    subnode += 1
                if pos[i,1] > center[1,no-NumPart]:
                    subnode += 2
                if pos[i,2] > center[2,no-NumPart]:
                    subnode += 4

                #print('p[%d] subnode = %d (no=%d no-NumPart=%d)' % (i,subnode,no,no-NumPart))

                # get the next node
                nn = suns[subnode,no-NumPart]

                if nn >= 0: # ok, something is in the daughter slot already, need to continue
                    parent = no # note: subnode can still be used in the next step of the walk
                    no = nn
                else:
                    # here we have found an empty slot where we can attach the new particle as a leaf
                    suns[subnode,no-NumPart] = i
                    #print(' attached as leaf suns[%d,%d]' % (subnode,no-NumPart))
                    break # done for this particle
            else:
                # we try to insert into a leaf with a single particle - need to generate a new internal
                # node at this point, because every leaf is only allowed to contain one particle
                suns[subnode,parent-NumPart] = nFree

                length[nFree-NumPart] = 0.5 * length[parent-NumPart]
                lenHalf = 0.25 * length[parent-NumPart]

                if subnode & 1: # np.bitwise_and(subnode,1)
                    center[0,nFree-NumPart] = center[0,parent-NumPart] + lenHalf
                else:
                    center[0,nFree-NumPart] = center[0,parent-NumPart] - lenHalf

                if subnode & 2:
                    center[1,nFree-NumPart] = center[1,parent-NumPart] + lenHalf
                else:
                    center[1,nFree-NumPart] = center[1,parent-NumPart] - lenHalf

                if subnode & 4:
                    center[2,nFree-NumPart] = center[2,parent-NumPart] + lenHalf
                else:
                    center[2,nFree-NumPart] = center[2,parent-NumPart] - lenHalf

                for j in range(8):
                    suns[j,nFree-NumPart] = -1

                # which subnode
                subnode = 0

                if pos[no-NumPart,0] > center[0,nFree-NumPart]:
                    subnode += 1
                if pos[no-NumPart,1] > center[1,nFree-NumPart]:
                    subnode += 2
                if pos[no-NumPart,2] > center[2,nFree-NumPart]:
                    subnode += 4

                #if length[nFree] < 1.0e-3 * Softening:
                    # randomize subnode index, or terminate

                suns[subnode,nFree-NumPart] = no

                no = nFree # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                if numNodes >= length.shape[0]:
                    return 0 # e.g. request reallocation and retry

    # now compute the multipole moments recursively
    #last = np.zeros( 1, dtype=np.int32) -1
    #dummy = np.zeros( 1, dtype=np.int32) -1

    last = np.int32(-1)
    dummy = np.int32(-1)
    #NumPart = np.int32(NumPart)

    #print(last.dtype)
    ##print(NumPart.dtype)
    #print(dummy.dtype)
    #print(suns.dtype)

    _updateNodeRecursive(NumPart,dummy,NumPart,last,suns,nextnode,next_node,sibling)

    #print('last: ',last)
    if last >= NumPart:
        nextnode[last-NumPart] = -1
    else:
        next_node[last] = -1

    return numNodes

# ------------------------------------------------------------------------------

@jit(nopython=True, nogil=True)
def _updateNodeRecursive2(last):
    if last > 3:
        return

    for i in range(4):
        last += 1
        _updateNodeRecursive2(last)

@jit(nopython=True, nogil=True)
def _constructTree2():
    last = np.int32(-1)
    _updateNodeRecursive2(last)

def testMe():
    _constructTree2()

# ------------------------------------------------------------------------------

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
    assert pos.ndim == 2 and pos.shape[1] in [2,3], 'Strange dimensions of pos.'
    assert posSearch is None, 'Not implemented (separate posSearch array, need still to port from IDL).'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert nDims in [1,2,3], 'Invalid ndims specification.'
    assert treePrec in ['single','double'], 'Invalid treePrec specification.'

    # tree allocation and construction
    treeDtype = 'float32' if treePrec == 'single' else 'float64'

    NumPart = pos.shape[0]
    MaxNodes = np.int(0.7*NumPart) + 1

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
    numNodes = _constructTree(pos,boxSizeSim,NextNode, #Father,
                              length,center,suns,sibling,nextnode) #,father)
                              #,maxsort,s,mass,bitflags) # unused

    # tree search
    print('numNodes: %d (MaxNodes=%d)' % (numNodes,MaxNodes))

    #for i in range(numNodes):
    #    print(' [%d] length=%d [%d %d %d %d %d %d %d %d] sibling=%d nextnode=%d' % \
    #      (i, length[i], suns[0,i], suns[1,i], suns[2,i], suns[3,i], suns[4,i], 
    #s       suns[5,i], suns[6,i], suns[7,i], sibling[i], nextnode[i]))


def benchmark():
    """ Benchmark performance of calcHsml(). """
    np.random.seed(424242)
    from cosmo.load import snapshotSubset
    from util.simParams import simParams
    import time

    # config data
    if 1:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 2000

        posDtype = 'float32'

        pos = np.random.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)

    if 0:
        # load some gas in a box
        sP = simParams(res=128, run='tracer', redshift=0.0)
        pos = snapshotSubset(sP, 'gas', 'pos')

    # config
    nNGB = 32
    nNGBDev = 1
    treePrec = 'single'
    nThreads = 1

    # calculate and time
    start_time = time.time()
    nLoops = 1

    for i in np.arange(nLoops):
        hsml = calcHsml(pos,sP.boxSize,nNGB=nNGB,nNGBDev=nNGBDev,treePrec=treePrec,nThreads=nThreads)

    print('%d estimates of HSMLs took [%g] sec' % (nLoops,(time.time()-start_time)/nLoops))
