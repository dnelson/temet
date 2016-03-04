"""
util/sphMap.py
  Interpolation of scattered point sets onto a uniform grid using the SPH spline kernel deposition method.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
from numba import jit # TODO

def _NEAREST(x, BoxHalf, BoxSize):
    """ Periodic wrap distance. """
    if BoxSize == 0.0:
        return x
    
    #define NEAREST(x) (((x)>BoxHalf)?((x)-BoxSize):(((x)<-BoxHalf)?((x)+BoxSize):(x)))
    if x > BoxHalf:
        return x-BoxSize
    else:
        if x < -BoxHalf:
            return x+BoxSize
        else:
            return x

def _NEAREST_POS(x, BoxSize):
    """ Periodic wrap position. """
    if BoxSize == 0.0:
        return x

    #define NEAREST_POS(x) (((x)>BoxSize)?((x)-BoxSize):(((x)<0)?((x)+BoxSize):(x)))
    if x > BoxSize:
        return x-BoxSize
    else:
        if x < 0:
            return x+BoxSize
        else:
            return x

def _getkernel(hinv, r2, C1, C2, C3):
    """ Project spline kernel. """
    u = np.sqrt(r2) * hinv

    if u < 0.5:
        return (C1 + C2 * (u - 1.0) * u * u)
    else:
        return C3 * (1.0 - u) * (1.0 - u) * (1.0 - u)

def _calcSphMap(pos,hsml,mass,quant,dens_out,quant_out,
                boxSizeImg,boxSizeSim,boxCen,axes,ndims,nPixels,
                normQuant,normColDens):
    """ Core routine for sphMap(), see below. """
    # init
    NumPart = pos.shape[0]
    BoxHalf = boxSizeSim / 2.0
    axis3   = 3 - axes[0] - axes[1]

    p = np.zeros( 3, dtype='float32' )

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
    pixelArea  = pixelSizeX * pixelSizeY # e.g. (ckpc/h)^2

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 50.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 50.0

    # loop over all particles
    for k in np.arange(NumPart):
        if k % int(NumPart/10) == 0:
            print(str(k/NumPart*100.0) + '%')

        p[0] = pos[k,0]
        p[1] = pos[k,1]
        p[2] = pos[k,2] if pos.shape[1] == 3 else 0.0
        h    = hsml[k]
        v    = mass[k] if mass.size > 1 else mass[0]
        w    = quant[k] if quant is not None else 0.0

        # clip points ouside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs( _NEAREST(p[axis3]-boxCen[2],BoxHalf,boxSizeSim) ) > 0.5 * boxSizeImg[2]:
                continue

        # position relative to box (x,y) minimum
        pos0 = p[axes[0]] - (boxCen[0] - 0.5*boxSizeImg[0])
        pos1 = p[axes[1]] - (boxCen[1] - 0.5*boxSizeImg[1])

        # clamp smoothing length
        if h < hsmlMin:
            h = hsmlMin
        if h > hsmlMax:
            h = hsmlMax

        # clip points outside box (x,y) dimensions
        if np.abs( _NEAREST(p[axes[0]]-boxCen[0],BoxHalf,boxSizeSim) ) > 0.5*boxSizeImg[0]+h or \
           np.abs( _NEAREST(p[axes[1]]-boxCen[1],BoxHalf,boxSizeSim) ) > 0.5*boxSizeImg[1]+h:
           continue

        h2 = h*h
        hinv = 1.0 / h

        # number of pixels covered by particle
        nx = int(h / pixelSizeX + 1)
        ny = int(h / pixelSizeY + 1)

        # coordinates of pixel center of particle
        x = (np.floor( pos0 / pixelSizeX ) + 0.5) * pixelSizeX
        y = (np.floor( pos1 / pixelSizeY ) + 0.5) * pixelSizeY

        # calculate sum (normalization)
        kSum = 0.0
        
        for dx in np.arange(-nx,nx+1):
            for dy in np.arange(-ny,ny+1):
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
        for dx in np.arange(-nx,nx+1):
            for dy in np.arange(-ny,ny+1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim)
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim)

                # pixel array indices
                i = int(xxx / pixelSizeX)
                j = int(yyy / pixelSizeY)

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

                    dens_out [j, i] += kVal * v_over_sum
                    quant_out[j, i] += kVal * v_over_sum * w

    # normalize mass weighted quantity
    # this only works for all-at-once calcs, otherwise maps need first to be collected, then divided
    if normQuant:
        for j in np.arange(nPixels[1]):
            for i in np.arange(nPixels[0]):
                if dens_out[j, i] > 0:
                    quant_out[j, i] /= dens_out[j, i]

    # for column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    if normColDens:
        dens_out /= pixelArea

    print('Done')

def sphMap(pos, hsml, mass, quant, axes, boxSizeImg, boxSizeSim, boxCen, nPixels, ndims, colDens=False):
    """ Simultaneously calculate a gridded map of projected density and some other mass weighted 
        quantity (e.g. temperature) with the sph spline kernel. If quant=None, the map of mass is 
        returned, optionally converted to a column density map if colDens=True. If quant is specified, 
        the mass-weighted map of the quantity is instead returned.

      pos[N,3]/[N,2] : array of 3-coordinates for the particles (or 2-coords only, to ignore z-axis)
      hsml[N]        : array of smoothing lengths to use for the particles, same units as pos
      mass[N]/[1]    : array of mass to deposit (or scalar value if all particles have the same mass)
      quant[N]       : array of some quantity to calculate a mass-weighted projection of (None=skip)
      axes[2]        : list of two axis indices, e.g. [0,1] for x,y (project along z-axis)
      boxSizeImg[3]  : the physical size the image should cover, same units as pos
      boxSizeSim[1]  : the physical size of the simulation box for periodic wrapping (0=non periodic)
      boxCen[3]      : (x,y,z) coordinates of the center of the imaging box, same units as pos
    """
    # input sanity checks
    if len(boxSizeImg) != 3 or not isinstance(boxSizeSim,(float)) or len(boxCen) != 3:
        raise Exception('Strange size of box input(s).')
    if len(nPixels) != 2 or len(axes) != 2:
        raise Exception('Strange size of imaging input(s).')

    if pos.ndim != 2 or (pos.shape[1] != 3 and pos.shape[1] != 2):
        raise Exception('Strange dimensions of pos.')
    if hsml.ndim != 1 or mass.ndim != 1 or (quant is not None and quant.ndim != 1):
        raise Exception('Strange dimensions of hsml/mass/quant.')
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

    # allocate return grids
    rDens = np.zeros( nPixels, dtype='float32' )
    rQuant = np.zeros( nPixels, dtype='float32' )

    normQuant   = True # must be false and then done at the end if multiple
    normColDens = colDens

    # call JIT compiled kernel
    # TODO: could here split pos,hsml,mass,quant among threads, give them separate rArrays to write, and sum
    _calcSphMap(pos,hsml,mass,quant,rDens,rQuant,
                boxSizeImg,boxSizeSim,boxCen,axes,ndims,nPixels,
                normQuant,normColDens)

    if quant is not None:
        return rQuant

    return rDens

def test():
    """ Debugging plots for accuracy/performance of sphMap(). """
    np.random.seed(424242)
    import pdb
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from cosmo.load import snapshotSubset
    from util.simParams import simParams

    # config data
    if 0:
        # generate random testing data
        class sP:
            boxSize = 100.0

        nPts = 2000
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
        pos   = snapshotSubset(sP, 'gas', 'pos', inds=np.arange(10000)*100)
        hsml  = snapshotSubset(sP, 'gas', 'cellrad', inds=np.arange(10000)*100)
        mass  = snapshotSubset(sP, 'gas', 'mass', inds=np.arange(10000)*100)
        quant = snapshotSubset(sP, 'gas', 'temp', inds=np.arange(10000)*100)

    # config imaging
    nPixels    = [30,30]
    ndims      = 3
    boxCen     = sP.boxSize * np.array([0.5,0.5,0.5])
    boxSizeImg = np.array([sP.boxSize,sP.boxSize,sP.boxSize])
    axes       = [0,1]

    # map
    densMap  = sphMap(pos, hsml, mass, None, axes, boxSizeImg, sP.boxSize, boxCen, nPixels, ndims)
    colMap   = sphMap(pos, hsml, mass, None, axes, boxSizeImg, sP.boxSize, boxCen, nPixels, ndims, colDens=True)
    quantMap = sphMap(pos, hsml, mass, quant, axes, boxSizeImg, sP.boxSize, boxCen, nPixels, ndims)

    # plot
    extent = [ boxCen[axes[0]] - 0.5*boxSizeImg[0], boxCen[axes[0]] + 0.5*boxSizeImg[0], 
               boxCen[axes[1]] - 0.5*boxSizeImg[1], boxCen[axes[1]] + 0.5*boxSizeImg[1]]

    fig = plt.figure(figsize=(26,12))

    ax = fig.add_subplot(131)
    ax.set_xlabel('x [ ckpc/h ]')
    ax.set_ylabel('y [ ckpc/h ]')
    plt.imshow(np.log10(densMap), extent=extent, origin='lower', interpolation='nearest')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('Column Mass [log 10$^{10}$ Msun/h]')

    ax = fig.add_subplot(132)
    ax.set_xlabel('x [ ckpc/h ]')
    ax.set_ylabel('y [ ckpc/h ]')
    plt.imshow(colMap, extent=extent, origin='lower', interpolation='nearest')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('Column Density [10$^{10}$ Msun * h / ckpc$^2$]')

    ax = fig.add_subplot(133)
    ax.set_xlabel('x [ ckpc/h ]')
    ax.set_ylabel('y [ ckpc/h ]')
    plt.imshow(quantMap, extent=extent, origin='lower', interpolation='nearest')
    cax = make_axes_locatable(ax).append_axes('right', size='5%', pad=0.1)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel('Temperature [ log K ]')

    fig.tight_layout()    
    fig.savefig('sphMap_test.pdf')
    plt.close(fig)

    pdb.set_trace()