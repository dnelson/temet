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

@jit(nopython=True, nogil=True) #, cache=True)
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

@jit(nopython=True, nogil=True) #, cache=True)
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
        length[0] = boxSizeSim
    else:
        # non-periodic
        for j in range(3):
            xyzMin[j] = 1.0e35 # MAX_REAL_NUMBER
            xyzMax[j] = -1.0e35 # MAX_REAL_NUMBER

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
        length[0] = extent

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

                if(length[ind2] < 1e-4):
                    # may have particles at identical locations, in which case randomize the subnode 
                    # index to put the particle into a different leaf (happens well below the 
                    # gravitational softening scale)
                    subnode = np.int(np.random.rand())
                    subnode = max(subnode,7)

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

@jit(nopython=True, nogil=True, cache=True)
def _treeSearch(xyz,h,NumPart,boxSizeSim,pos,next_node,length,center,sibling,nextnode,quant,op):
    """ Helper routine for calcHsml(), see below. """
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h
    hinv = 1.0 / h

    numNgbInH = 0
    numNgbWeightedInH = 0.0

    quantResult = 0.0
    if op == 2: # max
        quantResult = -np.inf
    if op == 3: # min
        quantResult = np.inf

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
            kval = _getkernel(hinv, r2, C1, C2, C3)
            numNgbWeightedInH += CN * kval

            # reduction operation on particle quantity
            if op == 1: # sum
                quantResult += quant[p]
            if op == 2: # max
                if quant[p] > quantResult: quantResult = quant[p]
            if op == 3: # min
                if quant[p] < quantResult: quantResult = quant[p]
            if op == 4: # unweighted mean
                quantResult += quant[p]
            if op == 5: # kernel-weighted mean
                quantResult += quant[p] * kval
            if op == 6: # count
                quantResult += 1

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

    if op == 4: # mean
        if numNgbInH != 0.0: quantResult /= numNgbInH
    if op == 5: # kernel-weighted mean
        if numNgbWeightedInH != 0.0: quantResult /= numNgbWeightedInH

    return numNgbInH, numNgbWeightedInH, quantResult

@jit(nopython=True, nogil=True, cache=True)
def _treeSearchIndices(xyz,h,NumPart,boxSizeSim,pos,next_node,length,center,sibling,nextnode):
    """ Helper routine for calcParticleIndices(), see below. """
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h
    hinv = 1.0 / h

    numNgbInH = 0

    # allocate, unfortunately unclear how safe we have to be here
    inds = np.empty( NumPart, dtype=np.int64 )

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
            inds[numNgbInH] = p
            numNgbInH += 1

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

    if numNgbInH > 0:
        inds = inds[0:numNgbInH]
        return inds

    return None

@jit(nopython=True, nogil=True, cache=True)
def _treeSearchHsmlIterate(xyz,h_guess,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                          next_node,length,center,sibling,nextnode,weighted_num):
    """ Helper routine for calcHsml(), see below. """
    left  = 0.0
    right = 0.0
    one_third = 1.0 / 3.0

    if h_guess == 0.0:
        h_guess = 1.0

    iter_num = 0
    dummy_in = -1
    dummy_in2 = np.zeros(1,dtype=np.int32)

    while 1:
        iter_num += 1

        assert iter_num < 1000 # Convergence failure, too many iterations.

        numNgbInH, numNgbWeightedInH, dummy = _treeSearch(xyz,h_guess,NumPart,boxSizeSim,pos,
                                                          next_node,length,center,sibling,nextnode,
                                                          dummy_in2,dummy_in)

        # looking for h enclosing the SPH kernel weighted number, instead of the actual number?
        if weighted_num:
            numNgbInH = numNgbWeightedInH

        # success
        if numNgbInH > (nNGB-nNGBDev) and numNgbInH <= (nNGB+nNGBDev):
            break

        # fail, the number of neighbors we found within h_guess is outside bounds
        if left > 0.0 and right > 0.0:
            if right-left < 0.001 * left:
                break # particle is OK

        if numNgbInH < nNGB:#-nNGBDev:
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

@jit(nopython=True, nogil=True, cache=True)
def _treeSearchHsmlSet(posSearch,ind0,ind1,nNGB,nNGBDev,boxSizeSim,pos,
                       next_node,length,center,sibling,nextnode,weighted_num):
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

        hsml[i] = _treeSearchHsmlIterate(xyz,h_guess,nNGB,nNGBDev,NumPart,boxSizeSim,pos,
                                         next_node,length,center,sibling,nextnode,weighted_num)

        # use previous result as guess for the next (any spatial ordering will greatly help)
        h_guess = hsml[i]

    return hsml

@jit(nopython=True, nogil=True, cache=True)
def _treeSearchQuantReduction(posSearch,hsml,ind0,ind1,boxSizeSim,pos,quant,opnum,
                              next_node,length,center,sibling,nextnode):
    """ Core routine for calcQuantReduction(), see below. """
    numSearch = ind1 - ind0 + 1
    NumPart = pos.shape[0]

    result = np.zeros( numSearch, dtype=np.float32 )

    for i in range(numSearch):
        # single ball search using octtree, requesting reduction of a given type on the 
        # quantity of all the particles within a fixed distance hsml around xyz
        xyz = posSearch[ind0+i,:]
        h = hsml[ind0+i] if hsml.shape[0] > 1 else hsml[0] # variable or constant

        _, _, result[i] = _treeSearch(xyz,h,NumPart,boxSizeSim,pos,next_node,length,center,
                                      sibling,nextnode,quant,opnum)

    return result

def buildFullTree(pos, boxSizeSim, treePrec, verbose=False):
    """ Helper. See below. """
    treePrecs = {'single':'float32','double':'float64'}
    if treePrec in treePrecs.keys(): treePrec = treePrecs[treePrec]
    assert treePrec in treePrecs.values()

    start_time = time.time()

    NumPart = pos.shape[0]
    NextNode = np.zeros( NumPart, dtype='int32' )

    # tree allocation and construction (iterate in case we need to re-allocate for larger number of nodes)
    for num_iter in range(10):
        # allocate
        MaxNodes = np.int( (num_iter+0.7)*NumPart ) + 1
        if MaxNodes < 1000: MaxNodes = 1000

        length   = np.zeros( MaxNodes, dtype=treePrec )     # NODE struct member
        center   = np.zeros( (3,MaxNodes), dtype=treePrec ) # NODE struct member
        suns     = np.zeros( (8,MaxNodes), dtype='int32' )   # NODE.u first union member
        sibling  = np.zeros( MaxNodes, dtype='int32' )       # NODE.u second union member (NODE.u.d member)
        nextnode = np.zeros( MaxNodes, dtype='int32' )       # NODE.u second union member (NODE.u.d member)

        # construct: call JIT compiled kernel
        numNodes = _constructTree(pos,boxSizeSim,NextNode,length,center,suns,sibling,nextnode)

        if numNodes > 0:
            break

        print(' tree: increase alloc %g to %g and redo...' % (num_iter+1.1,num_iter+2.1))

    assert numNodes > 0, 'Tree: construction failed!'
    if verbose:
        print(' tree: construction took [%g] sec.' % (time.time()-start_time))

    # memory optimization: subset arrays to used portions
    length = length[0:numNodes]
    center = center[0:numNodes]
    sibling = sibling[0:numNodes]
    nextnode = nextnode[0:numNodes]

    return NextNode, length, center, sibling, nextnode

def calcHsml(pos, boxSizeSim, posSearch=None, nNGB=32, nNGBDev=1, nDims=3, weighted_num=False, 
             treePrec='single', tree=None, nThreads=40, verbose=False):
    """ Calculate a characteristic 'size' ('smoothing length') given a set of input particle coordinates, 
    where the size is defined as the radius of the sphere (or circle in 2D) enclosing the nNGB nearest 
    neighbors. If posSearch==None, then pos defines both the neighbor and search point sets, otherwise 
    a radius for each element of posSearch is calculated by searching for nearby points in pos.
        
      pos[N,3]/[N,2] : array of 3-coordinates for the particles (or 2-coords for 2D)
      boxSizeSim[1]  : the physical size of the simulation box for periodic wrapping (0=non periodic)
      nNGB           : number of nearest neighbors to search for in order to define HSML
      nNGBDev        : allowed deviation (+/-) from the requested number of neighbors
      nDims          : number of dimensions of simulation (1,2,3), to set SPH kernel coefficients
      weighted_num   : if True, search for SPH kernel weighted number of neighbors, instead of real number
      treePrec       : construct the tree using 'single' or 'double' precision for coordinates
      tree           : if not None, should be a list of all the needed tree arrays (pre-computed), 
                       i.e the exact return of buildFullTree()
      nThreads       : do multithreaded calculation (on treefind, while tree construction remains serial)
    """
    # input sanity checks
    treeDims  = [3]

    assert pos.ndim == 2 and pos.shape[1] in treeDims, 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert nDims in treeDims, 'Invalid ndims specification (3D only).'

    # handle small inputs
    if pos.shape[0] < nNGB-nNGBDev:
        nNGBDev = nNGB - pos.shape[0] + 1
        print('WARNING: Less particles than requested neighbors. Increasing nNGBDev to [%d]!' % nNGBDev)

    build_start_time = time.time()

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos,boxSizeSim,treePrec)
    else:
        NextNode, length, center, sibling, nextnode = tree # split out list
        
    build_done_time = time.time()
    #print(' calcHsml(): tree build took [%g] sec (serial).' % (time.time()-build_start_time))

    if posSearch is None:
        posSearch = pos # set search coordinates as a view onto the same pos used to make the tree

    if posSearch.shape[0] < nThreads:
        nThreads = 1
        #print('WARNING: Less particles than requested threads. Just running in serial.') 

    # single threaded?
    # ----------------
    if nThreads == 1:
        ind0 = 0
        ind1 = posSearch.shape[0] - 1

        hsml = _treeSearchHsmlSet(posSearch,ind0,ind1,nNGB,nNGBDev,boxSizeSim,pos,
                                  NextNode,length,center,sibling,nextnode,weighted_num)

        #print(' calcHsml(): search took [%g] sec (serial).' % (time.time()-build_done_time))
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
            self.nNGB         = nNGB
            self.nNGBDev      = nNGBDev
            self.boxSizeSim   = boxSizeSim
            self.weighted_num = weighted_num

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
                                           self.center,self.sibling,self.nextnode,self.weighted_num)

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

    #print(' calcHsml(): search took [%g] sec (nThreads=%d).' % (time.time()-build_done_time, nThreads))
    return hsml

def calcQuantReduction(pos, quant, hsml, op, boxSizeSim, posSearch=None, treePrec='single', tree=None, nThreads=16):
    """ Calculate a reduction of a given quantity reduction on all the particles within a fixed search 
    distance hsml around each pos. If posSearch==None, then pos defines both the neighbor and search point 
    sets, otherwise a reduction at the location of each posSearch is calculated by searching for nearby points in pos.
        
      pos[N,3]/[N,2]  : array of 3-coordinates for the particles (or 2-coords for 2D)
      quant[N]        : array of quantity values (i.e. mass, temperature)
      hsml[N]/hsml[1] : array of search distances, or scalar value if constant
      op              : 'min', 'max', 'mean', 'kernel_mean', 'sum', 'count'
      boxSizeSim[1]   : the physical size of the simulation box for periodic wrapping (0=non periodic)
      posSearch[N,3]  : search coordinates (optional)
      treePrec        : construct the tree using 'single' or 'double' precision for coordinates
      tree            : if not None, should be a list of all the needed tree arrays (pre-computed), 
                        i.e the exact return of buildFullTree()
      nThreads        : do multithreaded calculation (on treefind, while tree construction remains serial)
    """
    # input sanity checks
    ops = {'sum':1, 'max':2, 'min':3, 'mean':4, 'kernel_mean':5, 'count':6}
    treeDims  = [3]

    if isinstance(hsml,float):
        hsml = [hsml]
    hsml = np.array(hsml, dtype='float32')

    assert pos.ndim == 2 and pos.shape[1] in treeDims, 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert pos.shape[1] in treeDims, 'Invalid ndims specification (3D only).'
    assert quant.ndim == 1 and quant.size == pos.shape[0], 'Strange quant shape.'
    assert op in ops.keys(), 'Unrecognized reduction operation.'
    
    if posSearch is None:
        assert hsml.size in [1,quant.size], 'Strange hsml shape.'
    else:
        assert hsml.size in [1,posSearch.shape[0]], 'Strange hsml shape.'

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos,boxSizeSim,treePrec)
    else:
        NextNode, length, center, sibling, nextnode = tree # split out list

    build_done_time = time.time()

    if posSearch is None:
        posSearch = pos # set search coordinates as a view onto the same pos used to make the tree

    if posSearch.shape[0] < nThreads:
        nThreads = 1
        print('WARNING: Less particles than requested threads. Just running in serial.') 

    opnum = ops[op]

    # single threaded?
    # ----------------
    if nThreads == 1:
        ind0 = 0
        ind1 = posSearch.shape[0] - 1

        result = _treeSearchQuantReduction(posSearch,hsml,ind0,ind1,boxSizeSim,pos,quant,opnum,
                                           NextNode,length,center,sibling,nextnode)

        #print(' calcQuantReduction(): took [%g] sec (serial).' % (time.time()-build_done_time))
        return result

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
            self.boxSizeSim = boxSizeSim
            self.opnum      = opnum

            # create views to other arrays
            self.posSearch = posSearch
            self.pos       = pos
            self.hsml      = hsml
            self.quant     = quant

            self.NextNode  = NextNode
            self.length    = length
            self.center    = center
            self.sibling   = sibling
            self.nextnode  = nextnode

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            self.result = _treeSearchQuantReduction(self.posSearch,self.hsml,self.ind0,self.ind1,self.boxSizeSim,
                                                    self.pos,self.quant,self.opnum,
                                                    self.NextNode,self.length,self.center,self.sibling,self.nextnode)

    # create threads
    threads = [searchThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # allocate master return grids
    result = np.zeros( posSearch.shape[0], dtype=np.float32 )

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        result[thread.ind0 : thread.ind1 + 1] = thread.result

    #print(' calcQuantReduction(): took [%g] sec.' % (time.time()-build_done_time))
    return result

def calcParticleIndices(pos, posSearch, hsmlSearch, boxSizeSim, treePrec='single', tree=None):
    """ Find and return the actual particle indices (indexing pos, hsml) within the search radius hsml 
    of the posSearch location. Serial by construction, since we do only one search.

      pos[N,3]/[N,2] : array of 3-coordinates for the particles (or 2-coords for 2D)
      posSearch[3]   : search postion
      hsmlSearch[1]  : search distance
      boxSizeSim[1]  : the physical size of the simulation box for periodic wrapping (0=non periodic)
      treePrec       : construct the tree using 'single' or 'double' precision for coordinates
      tree           : if not None, should be a list of all the needed tree arrays (pre-computed), 
                       i.e the exact return of buildFullTree()
    """
    # input sanity checks
    treeDims  = [3]

    if isinstance(hsmlSearch,int): hsmlSearch = float(hsmlSearch)
    assert isinstance(hsmlSearch,(float))
    hsmlSearch = np.array(hsmlSearch, dtype='float32')

    assert pos.ndim == 2 and pos.shape[1] in treeDims, 'Strange dimensions of pos.'
    assert pos.dtype in [np.float32, np.float64], 'pos not in float32/64.'
    assert pos.shape[1] in treeDims, 'Invalid ndims specification (3D only).'

    # build tree
    if tree is None:
        NextNode, length, center, sibling, nextnode = buildFullTree(pos,boxSizeSim,treePrec)
    else:
        NextNode, length, center, sibling, nextnode = tree # split out list

    build_done_time = time.time()

    # single threaded
    NumPart = pos.shape[0]

    result = _treeSearchIndices(posSearch,hsmlSearch,NumPart,boxSizeSim,pos,NextNode,length,center,sibling,nextnode)

    #print(' calcParticleIndices(): took [%g] sec (serial).' % (time.time()-build_done_time))
    return result

def benchmark():
    """ Benchmark performance of calcHsml(). """
    np.random.seed(424242)
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
        sP = simParams(res=128, run='tng', redshift=0.0, variant='0000')
        pos = sP.snapshotSubset('gas', 'pos')

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
    import matplotlib.pyplot as plt

    nNGB = 64
    nNGBDev = 1

    sP = simParams(res=128,run='tng',redshift=0.0,variant='0000')

    pos = sP.snapshotSubset('dm', 'pos')
    subfind_hsml = sP.snapshotSubset('dm', 'SubfindHsml')

    N = int(1e5)
    subfind_hsml = subfind_hsml[0:N]
    posSearch = pos[0:N,:]

    hsml = calcHsml(pos, sP.boxSize, posSearch=posSearch, nNGB=nNGB, nNGBDev=nNGBDev, treePrec='double')

    # check deviations
    ratio = hsml/subfind_hsml
    print('ratio, min max mean: ',ratio.min(),ratio.max(),ratio.mean())

    allowed_dev = 2.0*nNGBDev / nNGB # in NGB, not necessarily in hsml
    w_high = np.where( ratio > (1+allowed_dev) )
    w_low = np.where( ratio < (1-allowed_dev) )

    print('allowed dev = %.3f (above: %d = %.4f%%) (below: %d = %.4f%%)' % \
        (allowed_dev,len(w_high[0]),len(w_high[0])/ratio.size,len(w_low[0]),len(w_low[0])/ratio.size))

    # verify
    checkInds = np.hstack( (np.arange(10),w_high[0][0:10],w_low[0][0:10]) )

    for i in checkInds:
        dists = sP.periodicDists(posSearch[i,:],pos)
        dists = np.sort(dists)
        ww = np.where(dists < hsml[i])
        ww2 = np.where(dists < subfind_hsml[i])
        numInHsml = len(ww[0])
        numInHsmlSnap = len(ww2[0])
        passMine = (numInHsml >= (nNGB-nNGBDev)) & (numInHsml <= (nNGB+nNGBDev))
        passSnap = (numInHsmlSnap >= (nNGB-nNGBDev)) & (numInHsmlSnap <= (nNGB+nNGBDev))
        print('[%2d] hsml: %.3f hsmlSnap: %.3f, myNumInHsml: %d (pass: %s) numInHsmlSnap: %d (pass: %s)' % \
            (i,hsml[i],subfind_hsml[i],numInHsml,passMine,numInHsmlSnap,passSnap))
    import pdb; pdb.set_trace()

    # plot
    fig = plt.figure(figsize=(20,20))

    ax = fig.add_subplot(111)
    ax.set_xlabel('SubfindHSML')
    ax.set_ylabel('CalcHsml')

    ax.set_xlim([0,50])
    ax.set_ylim([0,50])

    ax.scatter(subfind_hsml,hsml)
    ax.plot([0,45],[0,45],'r')

    fig.savefig('hsml.png')
    plt.close(fig)

