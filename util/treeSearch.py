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

@jit(nopython=True, nogil=True)
def _updateNodeRecursive(no,sib,NumPart,last,suns,nextnode,next_node,sibling):
    """ Helper routine for calcHsml(), see below. """
    pp = 0
    nextsib = 0

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
def _constructTree(pos,boxSizeSim,next_node,length,center,suns,sibling,nextnode):
    """ Core routine for calcHsml(), see below. """
    subnode = 0
    parent  = -1
    lenHalf = 0.0

    # Nodes_base and Nodes are both pointers to the arrays of NODE structs
    # Nodes_base is allocated with size >NumPart, and entries >=NumPart are "internal nodes"
    #  while entries from 0 to NumPart-1 are leafs (actual particles)
    #  Nodes just points to Nodes_base-NumPart (such that Nodes[no]=Nodes_base[no-NumPart])
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

                if subnode & 1:
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

                suns[subnode,ind2] = no

                no = nFree # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                if numNodes >= length.shape[0]:
                    # exceeding tree allocated size, need to increase and redo
                    return -1

    # now compute the (sibling,nextnode,next_node) recursively
    last = np.int32(-1)

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
            assert next_node[no] != no # Going into infinite loop.

            p = no
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
            ind = no-NumPart
            no = sibling[ind] # in case the node can be discarded

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

    return h_guess

@jit(nopython=True, nogil=True)
def _treeSearchHsmlSet(posSearch,ind0,ind1,nNGB,nNGBDev,boxSizeSim,pos,
                       next_node,length,center,sibling,nextnode):
    """ Core routine for calcHsml(), see below. """
    numSearch = ind1 - ind0 + 1
    NumPart = pos.shape[0]

    h_guess = 1.0
    if boxSizeSim > 0.0:
        h_guess = boxSizeSim / NumPart**(1.0/3.0)

    hsml = np.zeros( numSearch, dtype=np.float32 )

    for i in range(numSearch):
        # single ball search using octtree, requesting hsml which enclosed nNGB+/-nNGBDev around xyz
        xyz = posSearch[ind0+i,:]

        hsml[i] = _treeSearchHsmlSingle(xyz,h_guess,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                                        next_node,length,center,sibling,nextnode)

        # use previous result as guess for the next (any spatial ordering will greatly help)
        h_guess = hsml[i]

    return hsml

def calcHsml(pos, boxSizeSim, posSearch=None, nNGB=32, nNGBDev=1, nDims=3, treePrec='single', nThreads=16):
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
    treePrecs = ['single','double']
    treeDims  = [3]

    assert pos.ndim == 2 and pos.shape[1] in treeDims, 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert nDims in treeDims, 'Invalid ndims specification (3D only).'
    assert treePrec in ['single','double'], 'Invalid treePrec specification.'
    assert pos.shape[0] >= nNGB-nNGBDev, 'Less particles then requested neighbors.'

    treeDtype = 'float32' if treePrec == 'single' else 'float64'

    NumPart = pos.shape[0]

    # tree allocation and construction (iterate in case we need to re-allocate for larger number of nodes)
    start_time = time.time()
    print(' calcHsml(): starting...')

    NextNode = np.zeros( NumPart, dtype='int32' )

    for num_iter in range(10):
        # allocate
        MaxNodes = np.int( (num_iter+1.1)*NumPart ) + 1

        length   = np.zeros( MaxNodes, dtype=treeDtype )     # NODE struct member
        center   = np.zeros( (3,MaxNodes), dtype=treeDtype ) # NODE struct member
        suns     = np.zeros( (8,MaxNodes), dtype='int32' )   # NODE.u first union member
        sibling  = np.zeros( MaxNodes, dtype='int32' )       # NODE.u second union member (NODE.u.d member)
        nextnode = np.zeros( MaxNodes, dtype='int32' )       # NODE.u second union member (NODE.u.d member)

        # construct: call JIT compiled kernel
        numNodes = _constructTree(pos,boxSizeSim,NextNode,length,center,suns,sibling,nextnode)

        if numNodes > 0:
            break

        print(' calcHsml(): increase alloc %g to %g and redo...' % (num_iter+1.1,num_iter+2.1))

    build_done_time = time.time()
    assert numNodes > 0, 'Tree construction failed (try double precision if you used single).'
    print(' calcHsml(): tree construction took [%g] sec.' % (build_done_time-start_time))

    if posSearch is None:
        posSearch = pos # set search coordinates as a view onto the same pos used to make the tree

    # single threaded?
    # ----------------
    if nThreads == 1:
        ind0 = 0
        ind1 = posSearch.shape[0] - 1

        hsml = _treeSearchHsmlSet(posSearch,ind0,ind1,nNGB,nNGBDev,boxSizeSim,pos,
                                  NextNode,length,center,sibling,nextnode)

        return hsml

    # else, multithreaded
    # -------------------
    class searchThread(threading.Thread):
        """ Subclass Thread() to provide local storage (hsml) which can be retrieved after 
            this thread terminates and added to the global return. Note (on Ody2): This algorithm with 
            the serial overhead of the tree construction has ~55% scaling effeciency to 16 threads (~8x
            speedup), drops to ~32% effeciency at 32 threads (~10x speedup). """
        def __init__(self, threadNum, nThreads):
            super(searchThread, self).__init__()

            # determine local slice (this is a view instead of a copy, even better)
            searchInds = np.arange(posSearch.shape[0])
            inds = pSplit(searchInds, nThreads, threadNum)

            self.ind0 = inds[0]
            self.ind1 = inds[-1]

            # copy other parameters (non-self inputs to _calc() appears to prevent GIL release)
            self.nNGB       = nNGB
            self.nNGBDev    = nNGBDev
            self.boxSizeSim = boxSizeSim

            # create views to other arrays
            self.posSearch = posSearch
            self.pos       = pos
            self.NextNode  = NextNode
            self.length    = length
            self.center    = center
            self.sibling   = sibling
            self.nextnode  = nextnode

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            self.hsml = _treeSearchHsmlSet(self.posSearch,self.ind0,self.ind1,self.nNGB,self.nNGBDev,
                                           self.boxSizeSim,self.pos,self.NextNode,self.length,
                                           self.center,self.sibling,self.nextnode)

    # create threads
    threads = [searchThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # allocate master return grids
    hsml = np.zeros( posSearch.shape[0], dtype=np.float32 )

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        hsml[thread.ind0 : thread.ind1 + 1] = thread.hsml

    print(' calcHsml(): search took [%g] sec.' % (time.time()-build_done_time))
    return hsml

def benchmark():
    """ Benchmark performance of calcHsml(). """
    np.random.seed(424242)
    from cosmo.load import snapshotSubset
    from util.simParams import simParams

    # config data
    if 0:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 100000

        posDtype = 'float32'
        pos = np.random.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)

    if 1:
        # load some gas in a box
        sP = simParams(res=128, run='tracer', redshift=0.0)
        pos = snapshotSubset(sP, 'gas', 'pos')

    # config
    nNGB = 32
    nNGBDev = 1
    treePrec = 'single'
    nThreads = 16
    posSearch = None #pos[0:5000,:]

    # warmup (compile)
    hsml = calcHsml(pos,sP.boxSize,posSearch=posSearch,
                    nNGB=nNGB,nNGBDev=nNGBDev,treePrec=treePrec,nThreads=nThreads)

    # calculate and time
    start_time = time.time()
    nLoops = 4

    for i in np.arange(nLoops):
        hsml = calcHsml(pos,sP.boxSize,posSearch=posSearch,
                        nNGB=nNGB,nNGBDev=nNGBDev,treePrec=treePrec,nThreads=nThreads)

    print('%d estimates of HSMLs took [%g] sec on avg' % (nLoops,(time.time()-start_time)/nLoops))
    print('hsml min %g max %g mean %g' % (np.min(hsml), np.max(hsml), np.mean(hsml)))

def checkVsSubfindHsml():
    """ Compare our result vs SubfindHsml output. """
    from util import simParams
    from cosmo.load import snapshotSubset
    import matplotlib.pyplot as plt

    sP = simParams(res=455,run='tng',redshift=0.0)

    pos = snapshotSubset(sP, 'gas', 'pos')
    subfind_hsml = snapshotSubset(sP, 'gas', 'SubfindHsml')

    N = 1e6
    subfind_hsml = subfind_hsml[0:N]
    pos = pos[0:N,:]

    hsml = calcHsml(pos, sP.boxSize, nNGB=64, nNGBDev=1)

    # plot
    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(111)
    ax.set_xlabel('SubfindHSML')
    ax.set_ylabel('CalcHsml')

    ax.set_xlim([0,50])
    ax.set_ylim([0,50])

    ax.scatter(subfind_hsml,hsml)
    ax.plot([0,45],[0,45],'r')

    fig.tight_layout()    
    fig.savefig('hsml.png')
    plt.close(fig)