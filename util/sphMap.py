"""
util/sphMap.py
  Interpolation of scattered point sets onto a uniform grid using the SPH spline kernel deposition method.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import threading
from numba import jit
from util.helper import pSplit

@jit(nopython=True, nogil=True, cache=True)
def _NEAREST(x, BoxHalf, BoxSize):
    """ Periodic wrap distance. 
        #define NEAREST(x) (((x)>BoxHalf)?((x)-BoxSize):(((x)<-BoxHalf)?((x)+BoxSize):(x)))
    """
    if BoxSize == 0.0:
        return x
    
    if x > BoxHalf:
        return x-BoxSize
    else:
        if x < -BoxHalf:
            return x+BoxSize
        else:
            return x

@jit(nopython=True, nogil=True, cache=True)
def _NEAREST_POS(x, BoxSize):
    """ Periodic wrap position. 
        #define NEAREST_POS(x) (((x)>BoxSize)?((x)-BoxSize):(((x)<0)?((x)+BoxSize):(x)))
    """
    if BoxSize == 0.0:
        return x

    if x > BoxSize:
        return x-BoxSize
    else:
        if x < 0:
            return x+BoxSize
        else:
            return x

@jit(nopython=True, nogil=True, cache=True)
def _getkernel(hinv, r2, C1, C2, C3):
    """ Project spline kernel. """
    u = np.sqrt(r2) * hinv

    if u < 0.5:
        return (C1 + C2 * (u - 1.0) * u * u)
    else:
        return C3 * (1.0 - u) * (1.0 - u) * (1.0 - u)

@jit(nopython=True, nogil=True, cache=True)
def _calcSphMap(pos,hsml,mass,quant,dens_out,quant_out,
                boxSizeImg,boxSizeSim,boxCen,axes,ndims,nPixels,
                normQuant,normColDens):
    """ Core routine for sphMap(), see below. """
    # init
    NumPart = pos.shape[0]
    BoxHalf = boxSizeSim / 2.0
    axis3   = 3 - axes[0] - axes[1]

    # coefficients for SPH spline kernel and its derivative
    if ndims == 1:
        COEFF_1 = 4.0/3
        COEFF_2 = 8.0
        COEFF_3 = 2.6666666667
    if ndims == 2:
        COEFF_1 = 5.0/7*2.546479089470
        COEFF_2 = 5.0/7*15.278874536822
        COEFF_3 = 5.0/7*5.092958178941
    if ndims == 3:
        COEFF_1 = 2.546479089470
        COEFF_2 = 15.278874536822
        COEFF_3 = 5.092958178941

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]

    pixelArea   = pixelSizeX * pixelSizeY # e.g. (ckpc/h)^2

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 50.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 50.0

    # loop over all particles (Note: np.arange() seems to have a huge penalty here instead of range())
    for k in range(NumPart):
        p0 = pos[k,axes[0]]
        p1 = pos[k,axes[1]]
        p2 = pos[k,axis3] if pos.shape[1] == 3 else 0.0
        h  = hsml[k]
        v  = mass[k] if mass.size != 2 else mass[0]
        w  = quant[k] if quant.size > 1 else 0.0

        # clip points ouside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs( _NEAREST(p2-boxCen[2],BoxHalf,boxSizeSim) ) > 0.5 * boxSizeImg[2]:
                continue

        # clamp smoothing length
        if h < hsmlMin:
            h = hsmlMin
        if h > hsmlMax:
            h = hsmlMax

        # clip points outside box (x,y) dimensions
        if np.abs( _NEAREST(p0-boxCen[0],BoxHalf,boxSizeSim) ) > 0.5*boxSizeImg[0]+h or \
           np.abs( _NEAREST(p1-boxCen[1],BoxHalf,boxSizeSim) ) > 0.5*boxSizeImg[1]+h:
           continue

        # position relative to box (x,y) minimum
        pos0 = p0 - (boxCen[0] - 0.5*boxSizeImg[0])
        pos1 = p1 - (boxCen[1] - 0.5*boxSizeImg[1])

        h2 = h*h
        hinv = 1.0 / h

        # number of pixels covered by particle
        nx = np.floor(h / pixelSizeX + 1)
        ny = np.floor(h / pixelSizeY + 1)

        # coordinates of pixel center of particle
        x = (np.floor( pos0 / pixelSizeX ) + 0.5) * pixelSizeX
        y = (np.floor( pos1 / pixelSizeY ) + 0.5) * pixelSizeY

        # calculate sum (normalization)
        kSum = 0.0
        
        for dx in range(-nx,nx+1):
            for dy in range(-ny,ny+1):
                # distance of covered pixel from actual position
                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                r2 = xx * xx + yy * yy

                if r2 < h2:
                    kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

        if kSum < 1e-10:
            continue

        v_over_sum = v / kSum # normalization such that all kernel values sum to the weight v

        # calculate contribution
        for dx in range(-nx,nx+1):
            for dy in range(-ny,ny+1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim)
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim)

                # pixel array indices
                i = np.int(xxx / pixelSizeX)
                j = np.int(yyy / pixelSizeY)

                # skip if desired pixel is out of bounds
                if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1]:
                    continue

                # calculate kernel contribution at pixel center
                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                r2 = xx * xx + yy * yy

                if r2 < h2:
                    # divide by sum for normalization
                    kVal = _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

                    dens_out [i, j] += kVal * v_over_sum
                    quant_out[i, j] += kVal * v_over_sum * w

    # normalize mass weighted quantity
    # this only works for all-at-once calcs, otherwise maps need first to be collected, then divided
    if normQuant:
        for i in range(nPixels[0]):
            for j in range(nPixels[1]):
                if dens_out[i, j] > 0:
                    quant_out[i, j] /= dens_out[i, j]

    # for total column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    if normColDens:
        dens_out /= pixelArea

    # void return

def sphMap(pos, hsml, mass, quant, axes, boxSizeImg, boxSizeSim, boxCen, nPixels, ndims, 
           colDens=False, nThreads=16):
    """ Simultaneously calculate a gridded map of projected density and some other mass weighted 
        quantity (e.g. temperature) with the sph spline kernel. If quant=None, the map of mass is 
        returned, optionally converted to a column density map if colDens=True. If quant is specified, 
        the mass-weighted map of the quantity is instead returned.
        
          Note: transpose of _calcSphMap() is taken such that with default plotting approaches e.g. 
                axes=[0,1] gives imshow(return[i,j]) with x and y axes corresponding correctly to 
                code coordinates. 
          Note: both boxSizeImg and boxCenter [0,1,2] correspond to [axes[0],axes[1],axes2], meaning
                pos[:,axes[0]] is compared against the first entries boxSizeImg[0] and boxCenter[0]
                and not compared against e.g. boxSizeImg[axes[0]].

      pos[N,3]/[N,2] : array of 3-coordinates for the particles (or 2-coords only, to ignore z-axis)
      hsml[N]        : array of smoothing lengths to use for the particles, same units as pos
      mass[N]/[1]    : array of mass to deposit (or scalar value if all particles have the same mass)
      quant[N]       : array of some quantity to calculate a mass-weighted projection of (None=skip)
      axes[2]        : list of two axis indices, e.g. [0,1] for x,y (project along z-axis)
      boxSizeImg[3]  : the physical size the image should cover, same units as pos
      boxSizeSim[1]  : the physical size of the simulation box for periodic wrapping (0=non periodic)
      boxCen[3]      : (x,y,z) coordinates of the center of the imaging box, same units as pos
      nPixels[2]     : number of pixels in x,y directions for output image
      ndims          : number of dimensions of simulation (1,2,3), to set SPH kernel coefficients
      colDens        : if True, normalize each grid value by its area (default=False)
      nThreads       : do multithreaded calculation (mem required=nThreads times more)
    """
    # input sanity checks
    if len(boxSizeImg) != 3 or not isinstance(boxSizeSim,(float)) or len(boxCen) != 3:
        raise Exception('Strange size of box input(s).')
    if len(nPixels) != 2 or len(axes) != 2:
        raise Exception('Strange size of imaging input(s).')

    if pos.ndim != 2 or (pos.shape[1] != 3 and pos.shape[1] != 2):
        raise Exception('Strange dimensions of pos.')
    if hsml.ndim != 1 or (mass.ndim != 1 and mass.size > 1) or (quant is not None and quant.ndim != 1):
        raise Exception('Strange dimensions of hsml/mass/quant.')
    if (mass.ndim == 0 and mass.size != 1) or (mass.size == 2):
        raise Exception('Strange shape of mass.')
    if pos.shape[0] != hsml.size or (pos.shape[0] != mass.size and mass.size > 1):
        raise Exception('Dimension mismatch for inputs (hsml/mass).')
    if quant is not None and pos.shape[0] != quant.size:
        raise Exception('Dimension mismatch for inputs (quant).')

    if pos.dtype != np.float32 and pos.dtype != np.float64:
        raise Exception('pos not in float32/64')
    if hsml.dtype != np.float32 or mass.dtype != np.float32:
        raise Exception('hsml/mass not in float32')
    if quant is not None and quant.dtype != np.float32:
        raise Exception('quant not in float32')

    if axes[0] not in [0,1,2] or axes[1] not in [0,1,2]:
        raise Exception('Invalid axes specification.')
    if ndims not in [1,2,3]:
        raise Exception('Invalid ndims specification.')

    # only input two coordinates per particle in 3D?
    if ndims == 3 and pos.shape[1] == 2:
        if axes[0] != 0 or axes[1] != 1:
            raise Exception('Must have axes=[0,1] for 3D projection with two coordinates only.')

    # massage quant if not specified
    if quant is None:
        quant = np.array([0])

    # massage mass if single scalar
    if mass.size == 1:
        mass = np.array( [mass,mass], dtype='float32' ) # single element array kills numba type inference

    # single threaded?
    # ----------------
    if nThreads == 1:
        # allocate return grids
        rDens = np.zeros( nPixels, dtype='float32' )
        rQuant = np.zeros( nPixels, dtype='float32' )

        # call JIT compiled kernel (normQuant=True since we do only one computation)
        _calcSphMap(pos,hsml,mass,quant,rDens,rQuant,
                    boxSizeImg,boxSizeSim,boxCen,axes,ndims,nPixels,
                    True,colDens)

        if quant.size > 1:
            return rQuant.T
        return rDens.T

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """ Subclass Thread() to provide local storage (rDens,rQuant) which can be retrieved after 
            this thread terminates and added to the global return. Note (on Ody2): This technique with this 
            algorithm has ~94 percent scaling efficiency to 16 threads, drops to ~70 percent at 32. """
        def __init__(self, threadNum, nThreads):
            super(mapThread, self).__init__()

            # allocate local return grids as attributes of the function
            self.rDens = np.zeros( nPixels, dtype='float32' )
            self.rQuant = np.zeros( nPixels, dtype='float32' )

            # determine local slice (these are views not copies, even better)
            self.threadNum = threadNum
            self.nThreads = nThreads

            self.pos  = pSplit(pos, nThreads, threadNum)
            self.hsml = pSplit(hsml, nThreads, threadNum)
            self.mass = pSplit(mass, nThreads, threadNum) if mass.size != 2 else mass
            self.quant = pSplit(quant, nThreads, threadNum) if quant.size > 1 else quant

            # copy others into local space (non-self inputs to _calc() appears to prevent GIL release)
            self.boxSizeImg = boxSizeImg
            self.boxSizeSim = boxSizeSim
            self.boxCen     = boxCen

            self.axes    = axes
            self.ndims   = ndims
            self.nPixels = nPixels
            self.colDens = colDens

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            _calcSphMap(self.pos,self.hsml,self.mass,self.quant,self.rDens,self.rQuant,
                        self.boxSizeImg,self.boxSizeSim,self.boxCen,self.axes,self.ndims,self.nPixels,
                        False,self.colDens)

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # allocate master return grids
    rDens = np.zeros( nPixels, dtype='float32' )
    rQuant = np.zeros( nPixels, dtype='float32' )

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()
        
    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        rQuant += thread.rQuant
        rDens += thread.rDens

    # finalize and return
    if quant.size > 1:
        w = np.where(rDens > 0.0)
        rQuant[w] /= rDens[w]

        return rQuant.T
    return rDens.T

def sphMapWholeBox(pos, hsml, mass, quant, axes, nPixels, sP, colDens=False, nThreads=16):
    """ Wrap sphMap() specialized to projecting an entire cosmological/periodic box. Specifically, 
        (ndims,boxSizeSim,boxSizeImg,boxCen) are derived from sP, and nPixels should be input 
        as a single scalar which is then assumed to be square. """
    return sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, 
                   ndims=3, boxSizeSim=sP.boxSize, boxSizeImg=sP.boxSize*np.array([1.0,1.0,1.0]), 
                   boxCen=sP.boxSize*np.array([0.5,0.5,0.5]), nPixels=[nPixels,nPixels], 
                   colDens=colDens, nThreads=nThreads )

def benchmark():
    """ Benchmark performance of sphMap(). """
    np.random.seed(424242)
    from cosmo.load import snapshotSubset
    from util.simParams import simParams
    import time

    # config data
    if 0:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 200000
        posDtype = 'float32'
        hsmlMinMax = [1.0,10.0]
        massMinMax = [1e-5,1e-4]

        pos   = np.random.uniform(low=0.0, high=sP.boxSize, size=(nPts,3)).astype(posDtype)
        hsml  = np.random.uniform(low=hsmlMinMax[0], high=hsmlMinMax[1], size=nPts).astype('float32')
        mass  = np.random.uniform(low=massMinMax[0], high=massMinMax[1], size=nPts).astype('float32')
        quant = np.zeros( nPts, dtype='float32' ) + 10.0

    if 1:
        # load some gas in a box
        sP = simParams(res=128, run='tracer', redshift=0.0)
        pos   = snapshotSubset(sP, 'gas', 'pos')
        hsml  = snapshotSubset(sP, 'gas', 'cellrad')
        mass  = snapshotSubset(sP, 'gas', 'mass')
        quant = snapshotSubset(sP, 'gas', 'temp')

    # config imaging
    nPixels    = 100
    axes       = [0,1]

    # map and time
    start_time = time.time()
    nLoops = 1

    for i in np.arange(nLoops):
        densMap  = sphMapWholeBox(pos, hsml, mass, None, axes, nPixels, sP)
        quantMap = sphMapWholeBox(pos, hsml, mass, quant, axes, nPixels, sP)

    print('2 maps took [' + str((time.time()-start_time)/nLoops) + '] sec')
