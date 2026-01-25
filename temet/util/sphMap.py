"""
Interpolation of scattered point sets onto a uniform grid using the SPH spline kernel deposition method.
"""

import threading

import numpy as np
from numba import jit

from ..util.helper import pSplit


@jit(nopython=True, nogil=True, cache=True)
def _NEAREST(x, BoxHalf, BoxSize):
    """Periodic wrap distance.

    #define NEAREST(x) (((x)>BoxHalf)?((x)-BoxSize):(((x)<-BoxHalf)?((x)+BoxSize):(x)))
    """
    if BoxSize == 0.0:
        return x

    if x > BoxHalf:
        return x - BoxSize
    else:
        if x < -BoxHalf:
            return x + BoxSize
        else:
            return x


@jit(nopython=True, nogil=True, cache=True)
def _NEAREST_POS(x, BoxSize):
    """Periodic wrap position.

    #define NEAREST_POS(x) (((x)>BoxSize)?((x)-BoxSize):(((x)<0)?((x)+BoxSize):(x)))
    """
    if BoxSize == 0.0:
        return x

    if x > BoxSize:
        return x - BoxSize
    else:
        if x < 0:
            return x + BoxSize
        else:
            return x


@jit(nopython=True, nogil=True, cache=True)
def _getkernel(hinv, r2, C1, C2, C3):
    """Project spline kernel."""
    u = np.sqrt(r2) * hinv

    if u < 0.5:
        return C1 + C2 * (u - 1.0) * u * u
    else:
        return C3 * (1.0 - u) * (1.0 - u) * (1.0 - u)


@jit(nopython=True, nogil=True, cache=True)
def _getkernel_asym(x2_over_h0sq, y2_over_h1sq, C1, C2, C3):
    """Project spline kernel, asymmetric shape."""
    u = np.sqrt(x2_over_h0sq + y2_over_h1sq)

    if u < 0.5:
        return C1 + C2 * (u - 1.0) * u * u
    else:
        return C3 * (1.0 - u) * (1.0 - u) * (1.0 - u)


@jit(nopython=True, nogil=True, cache=True)
def _kernelCoefficients(ndims):
    """Return the normalization coefficients for the SPH spline kernel and its derivative, depending on ndims."""
    if ndims == 1:
        COEFF_1 = 4.0 / 3
        COEFF_2 = 8.0
        COEFF_3 = 2.6666666667
    if ndims == 2:
        COEFF_1 = 5.0 / 7 * 2.546479089470
        COEFF_2 = 5.0 / 7 * 15.278874536822
        COEFF_3 = 5.0 / 7 * 5.092958178941
    if ndims == 3:
        COEFF_1 = 2.546479089470
        COEFF_2 = 15.278874536822
        COEFF_3 = 5.092958178941

    return COEFF_1, COEFF_2, COEFF_3


@jit(nopython=True, nogil=True, cache=True)
def _calcSphMap(
    pos,
    hsml,
    mass,
    quant,
    dens_out,
    quant_out,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    axes,
    ndims,
    nPixels,
    normColDens,
    maxIntProj,
    minIntProj,
):
    """Core routine for sphMap(), see below."""
    # init
    NumPart = pos.shape[0]
    # axis along which to project
    axis3 = 3 - axes[0] - axes[1]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = boxSizeSim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernelCoefficients(ndims)

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]

    pixelArea = pixelSizeX * pixelSizeY  # e.g. (ckpc/h)^2

    if pixelSizeX < pixelSizeY:
        hsmlMax = pixelSizeX * 500.0
    else:
        hsmlMax = pixelSizeY * 500.0

    # loop over all particles (Note: np.arange() seems to have a huge penalty here instead of range())
    for k in range(NumPart):
        p0 = pos[k, axes[0]]
        p1 = pos[k, axes[1]]
        p2 = pos[k, axis3] if pos.shape[1] == 3 else 0.0
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        # clip points outside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs(_NEAREST(p2 - boxCen[2], BoxHalf[2], boxSizeSim[2])) > 0.5 * boxSizeImg[2] + h:
                continue

        # clamp smoothing length
        if h > hsmlMax:
            h = hsmlMax

        # clip points outside box (x,y) dimensions
        if (
            np.abs(_NEAREST(p0 - boxCen[0], BoxHalf[0], boxSizeSim[0])) > 0.5 * boxSizeImg[0] + h
            or np.abs(_NEAREST(p1 - boxCen[1], BoxHalf[1], boxSizeSim[1])) > 0.5 * boxSizeImg[1] + h
        ):
            continue

        # position relative to box (x,y) corner
        pos0 = _NEAREST(p0 - (boxCen[0] - 0.5 * boxSizeImg[0]), BoxHalf[0], boxSizeSim[0])
        pos1 = _NEAREST(p1 - (boxCen[1] - 0.5 * boxSizeImg[1]), BoxHalf[1], boxSizeSim[1])

        h2 = h * h
        hinv = 1.0 / h

        # coordinates of pixel center of pixel containing the particle
        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY

        # number of pixels covered by particle
        nx = int(np.floor(h / pixelSizeX + 1))
        ny = int(np.floor(h / pixelSizeY + 1))

        # if particle contained in one cell, dump everything into it
        if nx * ny == 1:
            # coordinates of pixel center of covering pixels
            xxx = _NEAREST_POS(x, boxSizeSim[0])
            yyy = _NEAREST_POS(y, boxSizeSim[1])

            # pixel array indices
            i = int(xxx / pixelSizeX)
            j = int(yyy / pixelSizeY)

            # skip if desired pixel is out of bounds
            if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1]:
                continue

            if minIntProj:
                # minimum intensity projection (kernel and mass weighted to determine max)
                if v * w < quant_out[i, j]:
                    dens_out[i, j] = v
                    quant_out[i, j] = v * w
            elif maxIntProj:
                # maximum intensity projection (kernel and mass weighted to determine max)
                if v * w > quant_out[i, j]:
                    dens_out[i, j] = v
                    quant_out[i, j] = v * w
            else:
                # normal weighted projection including contributions from all overlapping kernels
                dens_out[i, j] += v
                quant_out[i, j] += v * w
            continue

        # calculate sum (normalization), skip for MIP
        if not minIntProj and not maxIntProj:
            kSum = 0.0
            v_over_sum = 0.0
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    # distance of covered pixel from actual position
                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    r2 = xx * xx + yy * yy

                    if r2 < h2:
                        kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

            if kSum < 1e-10:
                continue

            v_over_sum = v / kSum  # normalization such that all kernel values sum to the weight v

        # calculate contribution
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim[0])
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim[1])

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

                    if minIntProj:
                        # minimum intensity projection (kernel and mass weighted to determine max)
                        if kVal * v * w < quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    elif maxIntProj:
                        # maximum intensity projection (kernel and mass weighted to determine max)
                        if kVal * v * w > quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    else:
                        # normal weighted projection including contributions from all overlapping kernels
                        dens_out[i, j] += kVal * v_over_sum
                        quant_out[i, j] += kVal * v_over_sum * w

    # for total column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    if normColDens:
        dens_out /= pixelArea

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _calcSphGrid(
    pos,
    hsml,
    mass,
    quant,
    dens_out,
    quant_out,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    ndims,
    nPixels,
    normVolDens,
    maxIntProj,
    minIntProj,
):
    """Core routine for sphMap() that does 3D gridding."""
    # init
    NumPart = pos.shape[0]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = boxSizeSim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernelCoefficients(ndims)

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]
    pixelSizeZ = boxSizeImg[2] / nPixels[2]

    minSize = min(pixelSizeX, pixelSizeY, pixelSizeZ)
    hsmlMin = 1.001 * minSize * 0.5
    hsmlMax = minSize * 65.0

    # loop over all particles
    for k in range(NumPart):
        p0 = pos[k, 0]
        p1 = pos[k, 1]
        p2 = pos[k, 2]
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        # clamp smoothing length
        if h < hsmlMin:
            h = hsmlMin
        elif h > hsmlMax:
            h = hsmlMax

        # clip points outside subbox to grid
        if (
            np.abs(_NEAREST(p0 - boxCen[0], BoxHalf[0], boxSizeSim[0])) > 0.5 * boxSizeImg[0] + h
            or np.abs(_NEAREST(p1 - boxCen[1], BoxHalf[1], boxSizeSim[1])) > 0.5 * boxSizeImg[1] + h
            or np.abs(_NEAREST(p2 - boxCen[2], BoxHalf[2], boxSizeSim[2])) > 0.5 * boxSizeImg[2] + h
        ):
            continue

        # position relative to subbox (x, y, z) corner
        pos0 = p0 - (boxCen[0] - 0.5 * boxSizeImg[0])
        pos1 = p1 - (boxCen[1] - 0.5 * boxSizeImg[1])
        pos2 = p2 - (boxCen[2] - 0.5 * boxSizeImg[2])

        h2 = h * h
        hinv = 1.0 / h

        # coordinates of center of pixel containing the particle relative to subbox corner
        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY
        z = (np.floor(pos2 / pixelSizeZ) + 0.5) * pixelSizeZ

        # number of pixels covered by particle (at least one in each direction)
        nx = np.floor(h / pixelSizeX + 1)
        ny = np.floor(h / pixelSizeY + 1)
        nz = np.floor(h / pixelSizeZ + 1)

        # if particle contained in one cell, dump everything into it
        if nx * ny * nz == 1:
            # pixel array indices
            i = int(x / pixelSizeX)
            j = int(y / pixelSizeY)
            k = int(z / pixelSizeZ)
            # skip if desired pixel is out of bounds
            if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1] or k < 0 or k >= nPixels[2]:
                continue
            if minIntProj:
                # minimum intensity projection (kernel and mass
                # weighted to determine max)
                if v * w < quant_out[i, j, k]:
                    dens_out[i, j, k] = v
                    quant_out[i, j, k] = v * w
            elif maxIntProj:
                # maximum intensity projection (kernel and mass
                # weighted to determine max)
                if v * w > quant_out[i, j, k]:
                    dens_out[i, j, k] = v
                    quant_out[i, j, k] = v * w
            else:
                # normal weighted projection including
                # contributions from all overlapping kernels
                dens_out[i, j, k] += v
                quant_out[i, j, k] += v * w
            continue

        # calculate sum (normalization), skip for MIP
        if not minIntProj and not maxIntProj:
            kSum = 0.0
            v_over_sum = 0.0
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    for dz in range(-nz, nz + 1):
                        # distance of covered pixel from actual position
                        xx = x + dx * pixelSizeX - pos0
                        yy = y + dy * pixelSizeY - pos1
                        zz = z + dz * pixelSizeZ - pos2
                        r2 = xx * xx + yy * yy + zz * zz

                        if r2 < h2:
                            kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

            if kSum < 1e-10:
                continue

            v_over_sum = v / kSum  # normalization such that all kernel values
            # sum to the weight v

        # calculate contribution
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                for dz in range(-nz, nz + 1):
                    # coordinates of pixel center of covering pixels
                    xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim[0])
                    yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim[1])
                    zzz = _NEAREST_POS(z + dz * pixelSizeZ, boxSizeSim[2])

                    # pixel array indices
                    i = int(xxx / pixelSizeX)
                    j = int(yyy / pixelSizeY)
                    k = int(zzz / pixelSizeZ)

                    # skip if desired pixel is out of bounds
                    if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1] or k < 0 or k >= nPixels[2]:
                        continue

                    # calculate kernel contribution at pixel center
                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    zz = z + dz * pixelSizeZ - pos2
                    r2 = xx * xx + yy * yy + zz * zz

                    if r2 < h2:
                        # divide by sum for normalization
                        kVal = _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

                        if minIntProj:
                            # minimum intensity projection (kernel and mass
                            # weighted to determine max)
                            if kVal * v * w < quant_out[i, j, k]:
                                dens_out[i, j, k] = kVal * v
                                quant_out[i, j, k] = kVal * v * w
                        elif maxIntProj:
                            # maximum intensity projection (kernel and mass
                            # weighted to determine max)
                            if kVal * v * w > quant_out[i, j, k]:
                                dens_out[i, j, k] = kVal * v
                                quant_out[i, j, k] = kVal * v * w
                        else:
                            # normal weighted projection including
                            # contributions from all overlapping kernels
                            dens_out[i, j, k] += kVal * v_over_sum
                            quant_out[i, j, k] += kVal * v_over_sum * w

    # for volume density, normalize by the pixel volume, e.g. [10^10 Msun/h] -> [10^10 Msun * h^2 / ckpc^3]
    if normVolDens:
        pixelVol = pixelSizeX * pixelSizeY * pixelSizeZ
        dens_out /= pixelVol

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _calcSphMapPixelRel(
    pos,
    hsml,
    mass,
    quant,
    dens_out,
    quant_out,
    ref_grid,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    axes,
    ndims,
    nPixels,
    normColDens,
    maxIntProj,
    minIntProj,
):
    """Core routine for sphMap(), see below.

    For quant, instead of summing mass-weighted values,
    sum squares of differences between the mass-weighted particle quantity and the value (i.e. mean)
    of a reference_grid at that pixel location (i.e. for standard deviation per pixel).
    """
    # init
    NumPart = pos.shape[0]
    axis3 = 3 - axes[0] - axes[1]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = boxSizeSim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernelCoefficients(ndims)

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]

    pixelArea = pixelSizeX * pixelSizeY  # e.g. (ckpc/h)^2

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 500.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 500.0

    # loop over all particles (Note: np.arange() seems to have a huge penalty here instead of range())
    for k in range(NumPart):
        p0 = pos[k, axes[0]]
        p1 = pos[k, axes[1]]
        p2 = pos[k, axis3] if pos.shape[1] == 3 else 0.0
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        # clip points ouside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs(_NEAREST(p2 - boxCen[2], BoxHalf[2], boxSizeSim[2])) > 0.5 * boxSizeImg[2]:
                continue

        # clamp smoothing length
        if h < hsmlMin:
            h = hsmlMin
        if h > hsmlMax:
            h = hsmlMax

        # clip points outside box (x,y) dimensions
        if (
            np.abs(_NEAREST(p0 - boxCen[0], BoxHalf[0], boxSizeSim[0])) > 0.5 * boxSizeImg[0] + h
            or np.abs(_NEAREST(p1 - boxCen[1], BoxHalf[1], boxSizeSim[1])) > 0.5 * boxSizeImg[1] + h
        ):
            continue

        # position relative to box (x,y) center
        pos0 = p0 - (boxCen[0] - 0.5 * boxSizeImg[0])
        pos1 = p1 - (boxCen[1] - 0.5 * boxSizeImg[1])

        h2 = h * h
        hinv = 1.0 / h

        # number of pixels covered by particle
        nx = np.floor(h / pixelSizeX + 1)
        ny = np.floor(h / pixelSizeY + 1)

        # coordinates of pixel center of pixel containing the particle
        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY

        # calculate sum (normalization), skip for MIP
        kSum = 0.0
        v_over_sum = 0.0

        if not minIntProj and not maxIntProj:
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    # distance of covered pixel from actual position
                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    r2 = xx * xx + yy * yy

                    if r2 < h2:
                        kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

            if kSum < 1e-10:
                continue

            v_over_sum = v / kSum  # normalization such that all kernel values sum to the weight v

        # calculate contribution
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim[0])
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim[1])

                # pixel array indices
                i = int(xxx / pixelSizeX)
                j = int(yyy / pixelSizeY)

                # skip if desired pixel is out of bounds
                if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1]:
                    continue

                # reference grid is NaN at this pixel? skip
                if np.isnan(ref_grid[i, j]):
                    continue

                # calculate kernel contribution at pixel center
                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                r2 = xx * xx + yy * yy

                if r2 < h2:
                    # divide by sum for normalization
                    kVal = _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

                    # normal weighted projection including contributions from all overlapping kernels
                    dens_out[i, j] += kVal * v_over_sum
                    quant_out[i, j] += kVal * v_over_sum * (w - ref_grid[i, j]) * (w - ref_grid[i, j])

    # for total column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    if normColDens:
        dens_out /= pixelArea

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _calcSphMapAsymmetric(
    pos,
    hsml_0,
    hsml_1,
    mass,
    quant,
    dens_out,
    quant_out,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    axes,
    ndims,
    nPixels,
    normColDens,
    maxIntProj,
    minIntProj,
):
    """Core routine for sphMap(), see below. Asymmetric kernel with different hsml_0, hsml_1."""
    # init
    NumPart = pos.shape[0]
    axis3 = 3 - axes[0] - axes[1]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = boxSizeSim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernelCoefficients(ndims)

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 500.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 500.0

    # loop over all particles (Note: np.arange() seems to have a huge penalty here instead of range())
    for k in range(NumPart):
        p0 = pos[k, axes[0]]
        p1 = pos[k, axes[1]]
        p2 = pos[k, axis3] if pos.shape[1] == 3 else 0.0
        h0 = hsml_0[k]
        h1 = hsml_1[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        # clip points ouside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs(_NEAREST(p2 - boxCen[2], BoxHalf[2], boxSizeSim[2])) > 0.5 * boxSizeImg[2]:
                continue

        # clamp smoothing length
        if h0 < hsmlMin:
            h0 = hsmlMin
        if h0 > hsmlMax:
            h0 = hsmlMax
        if h1 < hsmlMin:
            h1 = hsmlMin
        if h1 > hsmlMax:
            h1 = hsmlMax

        # clip points outside box (x,y) dimensions
        if (
            np.abs(_NEAREST(p0 - boxCen[0], BoxHalf[0], boxSizeSim[0])) > 0.5 * boxSizeImg[0] + h0
            or np.abs(_NEAREST(p1 - boxCen[1], BoxHalf[1], boxSizeSim[1])) > 0.5 * boxSizeImg[1] + h1
        ):
            continue

        # position relative to box (x,y) center
        pos0 = p0 - (boxCen[0] - 0.5 * boxSizeImg[0])
        pos1 = p1 - (boxCen[1] - 0.5 * boxSizeImg[1])

        h0sq = h0 * h0
        h1sq = h1 * h1
        h0sqInv = 1.0 / h0sq
        h1sqInv = 1.0 / h1sq

        # number of pixels covered by particle
        nx = np.floor(h0 / pixelSizeX + 1)
        ny = np.floor(h1 / pixelSizeY + 1)

        # coordinates of pixel center of pixel containing the particle
        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY

        # calculate sum (normalization), skip for MIP
        kSum = 0.0
        v_over_sum = 0.0

        if not minIntProj and not maxIntProj:
            for dx in range(-nx, nx + 1):
                for dy in range(-ny, ny + 1):
                    # distance of covered pixel from actual position
                    xx = x + dx * pixelSizeX - pos0
                    yy = y + dy * pixelSizeY - pos1
                    x2_over_h0sq = xx * xx * h0sqInv
                    y2_over_h1sq = yy * yy * h1sqInv

                    if x2_over_h0sq + y2_over_h1sq < 1.0:
                        kSum += _getkernel_asym(x2_over_h0sq, y2_over_h1sq, COEFF_1, COEFF_2, COEFF_3)

            if kSum < 1e-10:
                continue

            v_over_sum = v / kSum  # normalization such that all kernel values sum to the weight v

        # calculate contribution
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim[0])
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim[1])

                # pixel array indices
                i = int(xxx / pixelSizeX)
                j = int(yyy / pixelSizeY)

                # skip if desired pixel is out of bounds
                if i < 0 or i >= nPixels[0] or j < 0 or j >= nPixels[1]:
                    continue

                # calculate kernel contribution at pixel center
                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                x2_over_h0sq = xx * xx * h0sqInv
                y2_over_h1sq = yy * yy * h1sqInv

                if x2_over_h0sq + y2_over_h1sq < 1.0:
                    # divide by sum for normalization
                    kVal = _getkernel_asym(x2_over_h0sq, y2_over_h1sq, COEFF_1, COEFF_2, COEFF_3)

                    if minIntProj:
                        # minimum intensity projection (kernel and mass weighted to determine max)
                        if kVal * v * w < quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    elif maxIntProj:
                        # maximum intensity projection (kernel and mass weighted to determine max)
                        if kVal * v * w > quant_out[i, j]:
                            dens_out[i, j] = kVal * v
                            quant_out[i, j] = kVal * v * w
                    else:
                        # normal weighted projection including contributions from all overlapping kernels
                        dens_out[i, j] += kVal * v_over_sum
                        quant_out[i, j] += kVal * v_over_sum * w

    # for total column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    pixelArea = pixelSizeX * pixelSizeY  # e.g. (ckpc/h)^2 or (rad)^2
    if normColDens:
        dens_out /= pixelArea

    # void return


@jit(nopython=True, nogil=True, cache=True)
def _gridInterp(
    posTarget,
    kT,
    axes,
    boxCen,
    boxSizeImg,
    boxSizeSim,
    nPixels,
    pixelSizeX,
    pixelSizeY,
    invPixelArea,
    dens_grid,
    quant_grid,
):
    """Helper for _calcSphMapTargets()."""
    p0T = posTarget[kT, axes[0]]
    p1T = posTarget[kT, axes[1]]

    pos0T = _NEAREST_POS(p0T - (boxCen[0] - 0.5 * boxSizeImg[0]), boxSizeSim[0])
    pos1T = _NEAREST_POS(p1T - (boxCen[1] - 0.5 * boxSizeImg[1]), boxSizeSim[1])

    # pixel coordinates (lower-left) of projected target position
    i0 = int(np.floor(pos0T / pixelSizeX))
    j0 = int(np.floor(pos1T / pixelSizeY))
    i1 = i0 + 1
    j1 = j0 + 1

    if i0 < 0 or j0 < 0 or i0 >= nPixels[0] or j0 >= nPixels[1]:
        raise Exception()  # not allowed

    if i1 >= nPixels[0] or j1 >= nPixels[1]:
        # fall back to NGP method since we have no outside row/col to interpolate with
        x = (np.floor(pos0T / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1T / pixelSizeY) + 0.5) * pixelSizeY

        xxx = _NEAREST_POS(x, boxSizeSim[0])
        yyy = _NEAREST_POS(y, boxSizeSim[1])

        i = int(xxx / pixelSizeX)
        j = int(yyy / pixelSizeY)

        dens_interp = dens_grid[i, j]
        quant_interp = quant_grid[i, j]

        # print('[ind=%3d] pos [%4.1f %4.1f] use coords [%3d %3d  na  na] vals = %7f %7f' % \
        #    (kT,p0T,p1T,i,j,dens_interp,quant_interp))
    else:
        # default: bilinear interpolation method
        x0 = i0 * pixelSizeX
        x1 = i1 * pixelSizeX
        y0 = j0 * pixelSizeY
        y1 = j1 * pixelSizeY

        dens_interp = invPixelArea * (
            dens_grid[i0, j0] * (x1 - pos0T) * (y1 - pos1T)
            + dens_grid[i1, j0] * (pos0T - x0) * (y1 - pos1T)
            + dens_grid[i0, j1] * (x1 - pos0T) * (pos1T - y0)
            + dens_grid[i1, j1] * (pos0T - x0) * (pos1T - y0)
        )

        quant_interp = invPixelArea * (
            quant_grid[i0, j0] * (x1 - pos0T) * (y1 - pos1T)
            + quant_grid[i1, j0] * (pos0T - x0) * (y1 - pos1T)
            + quant_grid[i0, j1] * (x1 - pos0T) * (pos1T - y0)
            + quant_grid[i1, j1] * (pos0T - x0) * (pos1T - y0)
        )

        # print('[ind=%3d] pos [%4.1f %4.1f] use coords [%3d %3d %3d %3d] vals = %7f %7f' % \
        #    (kT,p0T,p1T,i0,i1,j0,j1,dens_interp,quant_interp))

    return dens_interp, quant_interp


@jit(nopython=True, nogil=True, cache=True)
def _calcSphMapTargets(
    pos,
    hsml,
    mass,
    quant,
    posTarget,
    dens_out,
    quant_out,
    densT_out,
    quantT_out,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    axes,
    ndims,
    nPixels,
    normColDens,
):
    """Core routine for sphMap(), see below."""
    # init
    NumPart = pos.shape[0]
    NumTarget = posTarget.shape[0]
    axis3 = 3 - axes[0] - axes[1]

    BoxHalf = np.zeros(3, dtype=np.float32)
    for i in range(3):
        BoxHalf[i] = boxSizeSim[i] / 2.0

    COEFF_1, COEFF_2, COEFF_3 = _kernelCoefficients(ndims)

    # find minimum of axis3 coordinate
    min_axis3 = boxSizeSim[axis3] * 2.0
    pos_axis3 = np.zeros(NumPart, dtype=np.float32)
    posTarget_axis3 = np.zeros(NumTarget, dtype=np.float32)

    for i in range(NumPart):
        pos_axis3[i] = pos[i, axis3]  # copy
        if pos_axis3[i] < min_axis3:  # min
            min_axis3 = pos_axis3[i]
    for i in range(NumTarget):
        posTarget_axis3[i] = posTarget[i, axis3]
        if posTarget_axis3[i] < min_axis3:
            min_axis3 = posTarget_axis3[i]

    # sort indices along periodic(axis3) for both pos and posTarget
    for i in range(NumPart):
        pos_axis3[i] = _NEAREST(pos_axis3[i] - min_axis3, BoxHalf[2], boxSizeSim[2])
    for i in range(NumTarget):
        posTarget_axis3[i] = _NEAREST(posTarget_axis3[i] - min_axis3, BoxHalf[2], boxSizeSim[2])

    pos_sortInds = np.argsort(pos_axis3)  # requires numba 0.28+
    posTarget_sortInds = np.argsort(posTarget_axis3)

    # setup pixel sizes
    pixelSizeX = boxSizeImg[0] / nPixels[0]
    pixelSizeY = boxSizeImg[1] / nPixels[1]

    pixelArea = pixelSizeX * pixelSizeY  # e.g. (ckpc/h)^2
    invPixelArea = 1.0 / pixelArea

    if pixelSizeX < pixelSizeY:
        hsmlMin = 1.001 * pixelSizeX * 0.5
        hsmlMax = pixelSizeX * 500.0
    else:
        hsmlMin = 1.001 * pixelSizeY * 0.5
        hsmlMax = pixelSizeY * 500.0

    # set up first target
    targetInd = 0
    kT = posTarget_sortInds[targetInd]
    p2T = posTarget[kT, axis3]

    # loop over all particles in axis3-order
    for kLinear in range(NumPart):
        k = pos_sortInds[kLinear]

        p0 = pos[k, axes[0]]
        p1 = pos[k, axes[1]]
        p2 = pos[k, axis3]
        h = hsml[k]
        v = mass[k] if mass.size != 2 else mass[0]
        w = quant[k] if quant.size > 1 else 0.0

        # (A) check target list
        while _NEAREST(p2T - p2, BoxHalf[axis3], boxSizeSim[2]) <= 0.0:
            # target z-coordinate has been passed, record grid value now
            dens_interp, quant_interp = _gridInterp(
                posTarget,
                kT,
                axes,
                boxCen,
                boxSizeImg,
                boxSizeSim,
                nPixels,
                pixelSizeX,
                pixelSizeY,
                invPixelArea,
                dens_out,
                quant_out,
            )

            assert np.isnan(densT_out[targetInd])
            assert np.isnan(quantT_out[targetInd])

            densT_out[targetInd] = dens_interp
            quantT_out[targetInd] = quant_interp

            # move to following target
            targetInd += 1

            if targetInd >= NumTarget:
                # exit while loop and never re-enter
                p2T = boxSizeSim[2] * 2.0
                break

            kT = posTarget_sortInds[targetInd]
            p2T = posTarget[kT, axis3]

        # (B) proceed with normal sph mapping

        # immediately skip points with e.g. no mass (weight == 0)
        if v == 0.0:
            continue

        # clip points ouside box (z) dimension
        if pos.shape[1] == 3:
            if np.abs(_NEAREST(p2 - boxCen[2], BoxHalf[2], boxSizeSim[2])) > 0.5 * boxSizeImg[2]:
                continue

        # clamp smoothing length
        if h < hsmlMin:
            h = hsmlMin
        if h > hsmlMax:
            h = hsmlMax

        # clip points outside box (x,y) dimensions
        if (
            np.abs(_NEAREST(p0 - boxCen[0], BoxHalf[0], boxSizeSim[0])) > 0.5 * boxSizeImg[0] + h
            or np.abs(_NEAREST(p1 - boxCen[1], BoxHalf[1], boxSizeSim[1])) > 0.5 * boxSizeImg[1] + h
        ):
            continue

        # position relative to box (x,y) center
        pos0 = p0 - (boxCen[0] - 0.5 * boxSizeImg[0])
        pos1 = p1 - (boxCen[1] - 0.5 * boxSizeImg[1])

        h2 = h * h
        hinv = 1.0 / h

        # number of pixels covered by particle
        nx = np.floor(h / pixelSizeX + 1)
        ny = np.floor(h / pixelSizeY + 1)

        # coordinates of pixel center of pixel containing the particle
        x = (np.floor(pos0 / pixelSizeX) + 0.5) * pixelSizeX
        y = (np.floor(pos1 / pixelSizeY) + 0.5) * pixelSizeY

        # calculate sum (normalization)
        kSum = 0.0

        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                # distance of covered pixel from actual position
                xx = x + dx * pixelSizeX - pos0
                yy = y + dy * pixelSizeY - pos1
                r2 = xx * xx + yy * yy

                if r2 < h2:
                    kSum += _getkernel(hinv, r2, COEFF_1, COEFF_2, COEFF_3)

        if kSum < 1e-10:
            continue

        v_over_sum = v / kSum  # normalization such that all kernel values sum to the weight v

        # calculate contribution
        for dx in range(-nx, nx + 1):
            for dy in range(-ny, ny + 1):
                # coordinates of pixel center of covering pixels
                xxx = _NEAREST_POS(x + dx * pixelSizeX, boxSizeSim[0])
                yyy = _NEAREST_POS(y + dy * pixelSizeY, boxSizeSim[0])

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

                    dens_out[i, j] += kVal * v_over_sum
                    quant_out[i, j] += kVal * v_over_sum * w

    # any targets beyond pos z-distribution? sample completed grid now
    pos_max3 = np.max(pos_axis3)

    while targetInd < NumTarget:
        kT = posTarget_sortInds[targetInd]

        # verify
        p2T = posTarget[kT, axis3]
        assert p2T > pos_max3

        # record grid value now
        dens_interp, quant_interp = _gridInterp(
            posTarget,
            kT,
            axes,
            boxCen,
            boxSizeImg,
            boxSizeSim,
            nPixels,
            pixelSizeX,
            pixelSizeY,
            invPixelArea,
            dens_out,
            quant_out,
        )

        assert np.isnan(densT_out[targetInd])
        assert np.isnan(quantT_out[targetInd])

        densT_out[targetInd] = dens_interp
        quantT_out[targetInd] = quant_interp

        # move to following target
        targetInd += 1

    # verify
    assert targetInd == NumTarget
    # assert np.sum(~np.isfinite(densT_out)) == 0 # not true for threaded
    # assert np.sum(~np.isfinite(quantT_out)) == 0 # not true for threaded

    # for total column density, normalize by the pixel area, e.g. [10^10 Msun/h] -> [10^10 Msun * h / ckpc^2]
    if normColDens:
        dens_out /= pixelArea
        densT_out /= pixelArea

    # void return


def sphMap(
    pos,
    hsml,
    mass,
    quant,
    axes,
    boxSizeImg,
    boxSizeSim,
    boxCen,
    nPixels,
    ndims,
    hsml_1=None,
    colDens=False,
    nThreads=16,
    multi=False,
    posTarget=None,
    maxIntProj=False,
    minIntProj=False,
    refGrid=None,
):
    """Calculate gridded maps of scattered pointsets using the SPH spline kernel.

    Simultaneously calculate a gridded map of projected mass and some other mass weighted
    quantity (e.g. temperature) with the sph spline kernel. If quant=None, the map of mass is
    returned, optionally converted to a column density map if colDens=True. If quant is specified,
    the mass-weighted map of the quantity is additionally returned. If len(nPixels) is three, compute
    grid and not image.

    Args:
      pos (ndarray[float][N,3 or N,2]): array of 3-coordinates for the particles (or 2-coords only, to ignore z-axis).
      hsml (ndarray[float][N]): smoothing lengths to use for the particles, same units as pos.
      mass (ndarray[float][N or 1]): mass to deposit, can be scalar value if all particles have the same mass.
      quant (ndarray[float][N]): some quantity to calculate a mass-weighted projection of (None=skip).
      axes (list[int][2]): two axis indices defining the projection direction,
        e.g. [0,1] for x,y (project along z-axis).
      boxSizeImg (list[float][3]): the physical size the image should cover, same units as pos.
      boxSizeSim (list[float][3]): the physical size of the simulation box for periodic wrapping (0=non periodic).
      boxCen (list[float][3]): (x,y,z) coordinates of the center of the imaging box, same units as pos.
      nPixels (list[int][2 or 3]): number of pixels in x,y directions for output image or number of cells
        in x,y,z directions for output grid.
      ndims (int): number of dimensions of simulation (1,2,3), to set SPH kernel coefficients.
      hsml_1 (ndarray[float][N]): if not None, do asymmetric sph kernel projection with
        ``hsml==hsml_0`` (x), and ``hsml_1`` (y).
      colDens (bool): if True, normalize each grid value by its area (default=False).
      nThreads (int): do multithreaded calculation (mem required=nThreads times more).
      multi (bool): if True, return the tuple (dens,quant) instead of selecting one, under the
        assumption that we are using sphMap() in a chunked mode and have to normalize later.
      posTarget (ndarray[float][N,3]): array of 3-coordinates for a set of targets (e.g. star particle
        locations). If not None, then both pos and posTarget will be sorted on the 3rd (projection
        direction) coordinate, and mapping will be front to back, such that the grid value at the position
        (2D+depth) of each posTarget is estimated. In this case, the return is of shape [N(posTarget)]
        giving the projected density or mass weighted quantity along the line of sight to each posTarget.
      maxIntProj (bool): perform a maximum intensity projection (MIP) instead of the usual weighting.
      minIntProj (bool): perform a minimum intensity projection, instead of the usual weighting.
      refGrid (ndarray[float][nPixels[0],nPixels[1]]): if not None, calculation is the sum of squared differences
        between the mass-weighted particle quantity and the value of this reference grid at its pixel location
        (i.e. to calculate standard deviation per pixel).

    Returns:
      ndarray[float][nPixels]: computed 2D or 3D array, depositing either `mass` or `mass`-weighted `quant`.
      If ``multi`` is True, then a 2-tuple of the ``(dens_map, quant_map)`` arrays separately.

    Notes:
      * the transpose of :py:func:`_calcSphMap()` is taken such that with default plotting approaches e.g.
        ``axes=[0,1]`` gives ``imshow(return[i,j])`` with x and y axes corresponding correctly to
        code coordinates. This does not apply if ``len(nPixels) == 3`` and a 3D grid is computed.
        No transpose is taken in that case.
      * the entries of ``boxSizeImg``, ``boxCenter``, and ``boxSizeSim`` correspond to
        ``[axes[0],axes[1],axes[2]]``, meaning ``pos[:,axes[0]]`` is compared against the first entries
        ``boxSizeImg[0]`` and ``boxCenter[0]`` and not compared against e.g. ``boxSizeImg[axes[0]]``.
    """
    # input sanity checks
    if len(boxSizeImg) != pos.shape[1] or len(boxCen) != pos.shape[1]:
        raise Exception("Strange size of box input(s).")
    if len(nPixels) not in [2, 3] or len(axes) != 2:
        raise Exception("Strange size of imaging input(s).")

    if not isinstance(boxSizeSim, (float, int)):
        assert len(boxSizeSim) == 3
    else:
        boxSizeSim = [boxSizeSim, boxSizeSim, boxSizeSim]  # same in all 3 coordinate directions

    if pos.ndim != 2 or (pos.shape[1] != 3 and pos.shape[1] != 2) or pos.shape[0] <= 1:
        raise Exception("Strange dimensions of pos.")
    if hsml.ndim != 1 or (mass.ndim != 1 and mass.size > 1) or (quant is not None and quant.ndim != 1):
        raise Exception("Strange dimensions of hsml/mass/quant.")
    if hsml_1 is not None:
        if hsml_1.ndim != 1 or hsml_1.size != pos.shape[0] or hsml_1.size != hsml.size:
            raise Exception("Strange dimensions of hsml_1.")
        assert posTarget is None  # not implemented with asymmetric hsml
    if mass.ndim == 0 and mass.size != 1:
        raise Exception("Strange shape of mass.")
    if pos.shape[0] != hsml.size or (pos.shape[0] != mass.size and mass.size > 1):
        raise Exception("Dimension mismatch for inputs (hsml/mass).")
    if quant is not None and pos.shape[0] != quant.size:
        raise Exception("Dimension mismatch for inputs (quant).")

    if pos.dtype != np.float32 and pos.dtype != np.float64:
        raise Exception("pos not in float32/64")
    if hsml.dtype == np.float64:
        hsml = hsml.astype("float32")
        # print('Warning: sphMap() converting hsml from float64 to float32.')
    if hsml_1 is not None:
        if hsml_1.dtype == np.float64:
            hsml_1 = hsml.astype("float32")
            # print('Warning: sphMap() converting hsml_1 from float64 to float32.')
    if mass.dtype == np.float64:
        mass = mass.astype("float32")
        # print('Warning: sphMap() converting mass from float64 to float32.')

    if hsml.dtype != np.float32 or mass.dtype != np.float32:
        raise Exception("hsml/mass not in float32")
    if quant is not None and quant.dtype != np.float32:
        raise Exception("quant not in float32")

    if axes[0] not in [0, 1, 2] or axes[1] not in [0, 1, 2]:
        raise Exception("Invalid axes specification.")
    if ndims not in [1, 2, 3]:
        raise Exception("Invalid ndims specification.")
    if maxIntProj and minIntProj:
        raise Exception("Cannot do both maximum and minimum intensity projections at the same time.")
    if (maxIntProj or minIntProj) and posTarget is not None:
        raise Exception("MIP not yet supported with posTarget.")
    if refGrid is not None:
        refGrid = np.transpose(refGrid)  # expect input to be a previous output of sphMap()
        assert refGrid.shape[0] == nPixels[0] and refGrid.shape[1] == nPixels[1]
        assert refGrid.dtype in [np.float32, np.float64]
        assert hsml_1 is None  # not supported
        assert maxIntProj is False and minIntProj is False  # doesn't make sense

    # _calcSphMapTargets() mode?
    nTargets = 0

    if posTarget is not None:
        nTargets = posTarget.shape[0]
        assert ndims == 3
        assert pos.shape[1] == 3
        assert posTarget.ndim == 2
        assert posTarget.shape[1] == 3

    # only input two coordinates per particle in 3D?
    if ndims == 3 and pos.shape[1] == 2:
        if axes[0] != 0 or axes[1] != 1:
            raise Exception("Must have axes=[0,1] for 3D projection with two coordinates only.")

    # single-threaded for small calculations (avoid zero-size pSplits)
    if mass.size < nThreads * 10:
        nThreads = 1

    # massage quant if not specified
    if quant is None:
        quant = np.array([0])

    # numba: handle deprecation for type 'reflected list' (or convert into typed Lists)
    axes = np.array(axes)
    boxSizeSim = np.array(boxSizeSim)
    boxSizeImg = np.array(boxSizeImg)
    boxCen = np.array(boxCen)
    nPixels = np.array(nPixels)

    # massage mass if single scalar
    if mass.size == 1:
        mass = np.array([mass, mass], dtype="float32")  # single element array kills numba type inference

    # allocate return grids
    rDens = np.zeros(nPixels, dtype="float32")
    rQuant = np.zeros(nPixels, dtype="float32")
    rDensT = np.zeros(nTargets, dtype="float32")
    rQuantT = np.zeros(nTargets, dtype="float32")

    if minIntProj:
        rQuant.fill(np.inf)

    # single threaded?
    # ----------------
    if nThreads == 1:
        # debugging:
        rDensT.fill(np.nan)
        rQuantT.fill(np.nan)

        # call JIT compiled kernel
        if posTarget is None:
            if hsml_1 is None:
                if refGrid is None:
                    if len(nPixels) == 2:
                        # normal operation, produce 2D array
                        _calcSphMap(pos, hsml, mass, quant, rDens, rQuant, boxSizeImg, boxSizeSim, boxCen, axes, ndims,
                                    nPixels, colDens, maxIntProj, minIntProj)  # fmt: skip
                    elif len(nPixels) == 3:
                        # gridding, produce 3D array
                        _calcSphGrid(pos, hsml, mass, quant, rDens, rQuant, boxSizeImg, boxSizeSim, boxCen, ndims,
                                     nPixels, colDens, maxIntProj, minIntProj)  # fmt: skip
                else:
                    # symmetric, but relative to reference grid
                    _calcSphMapPixelRel(pos, hsml, mass, quant, rDens, rQuant, refGrid, boxSizeImg, boxSizeSim, boxCen,
                                        axes, ndims, nPixels, colDens, maxIntProj, minIntProj)  # fmt: skip
            else:
                # asymmetric
                _calcSphMapAsymmetric(pos, hsml, hsml_1, mass, quant, rDens, rQuant, boxSizeImg, boxSizeSim, boxCen,
                                      axes, ndims, nPixels, colDens, maxIntProj, minIntProj)  # fmt: skip
        else:
            _calcSphMapTargets(pos, hsml, mass, quant, posTarget, rDens, rQuant, rDensT, rQuantT, boxSizeImg,
                               boxSizeSim, boxCen, axes, ndims, nPixels, colDens)  # fmt: skip

            # replace rDens,rQuant with rDensT,rQuantT for returns
            rDens = rDensT
            rQuant = rQuantT

        if multi:
            if len(nPixels) == 2:
                return (rDens.T, rQuant.T)
            elif len(nPixels) == 3:
                return (rDens, rQuant)

        if quant.size > 1:
            w = np.where(rDens > 0.0)
            rQuant[w] /= rDens[w]

            if len(nPixels) == 2:
                return rQuant.T
            elif len(nPixels) == 3:
                return rQuant
        if len(nPixels) == 2:
            return rDens.T
        elif len(nPixels) == 3:
            return rDens

    # else, multithreaded
    # -------------------
    class mapThread(threading.Thread):
        """Subclass Thread() to provide local storage (rDens,rQuant).

        This local data can be retrieved after this thread terminates and added to the global return.
        Performance note (Ody2): ~94 percent scaling efficiency to 16 threads, drops to ~70 percent at 32 threads.
        """

        def __init__(self, threadNum, nThreads):
            super().__init__()

            # allocate local return grids as attributes of the function
            self.rDens = np.zeros(nPixels, dtype="float32")
            self.rQuant = np.zeros(nPixels, dtype="float32")
            self.rDensT = np.zeros(nTargets, dtype="float32")
            self.rQuantT = np.zeros(nTargets, dtype="float32")

            if minIntProj:
                self.rQuant.fill(np.inf)

            # debugging:
            self.rDensT.fill(np.nan)
            self.rQuantT.fill(np.nan)

            # determine local slice (these are views not copies, even better)
            self.threadNum = threadNum
            self.nThreads = nThreads

            self.pos = pSplit(pos, nThreads, threadNum)
            self.hsml = pSplit(hsml, nThreads, threadNum)
            self.mass = pSplit(mass, nThreads, threadNum) if mass.size != 2 else mass
            self.quant = pSplit(quant, nThreads, threadNum) if quant.size > 1 else quant

            if posTarget is not None:
                self.posTarget = posTarget  # full view
            if hsml_1 is not None:
                self.hsml_1 = pSplit(hsml_1, nThreads, threadNum)
            if refGrid is not None:
                self.refGrid = refGrid  # full view

            # copy others into local space (non-self inputs to _calc() appears to prevent GIL release)
            self.boxSizeImg = boxSizeImg
            self.boxSizeSim = boxSizeSim
            self.boxCen = boxCen
            self.axes = axes
            self.ndims = ndims
            self.nPixels = nPixels
            self.colDens = colDens
            self.maxIntProj = maxIntProj
            self.minIntProj = minIntProj

        def run(self):
            # call JIT compiled kernel (normQuant=False since we handle this later)
            if posTarget is None:
                if hsml_1 is None:
                    if refGrid is None:
                        if len(self.nPixels) == 2:
                            # normal operation
                            _calcSphMap(self.pos, self.hsml, self.mass, self.quant, self.rDens, self.rQuant,
                                        self.boxSizeImg, self.boxSizeSim, self.boxCen, self.axes, self.ndims,
                                        self.nPixels, self.colDens, self.maxIntProj, self.minIntProj, )  # fmt: skip
                        elif len(self.nPixels) == 3:
                            # gridding
                            _calcSphGrid(self.pos, self.hsml, self.mass, self.quant, self.rDens, self.rQuant,
                                         self.boxSizeImg, self.boxSizeSim, self.boxCen, self.ndims, self.nPixels,
                                         self.colDens, self.maxIntProj, self.minIntProj, )  # fmt: skip
                    else:
                        # symmetric, but relative to reference grid
                        _calcSphMapPixelRel(self.pos, self.hsml, self.mass, self.quant, self.rDens, self.rQuant,
                            self.refGrid, self.boxSizeImg, self.boxSizeSim, self.boxCen, self.axes,
                            self.ndims, self.nPixels, self.colDens, self.maxIntProj, self.minIntProj)  # fmt: skip
                else:
                    # asymmetric
                    _calcSphMapAsymmetric(self.pos, self.hsml, self.hsml_1, self.mass, self.quant, self.rDens,
                        self.rQuant, self.boxSizeImg, self.boxSizeSim, self.boxCen, self.axes,
                        self.ndims, self.nPixels, self.colDens, self.maxIntProj, self.minIntProj, )  # fmt: skip
            else:
                _calcSphMapTargets(self.pos, self.hsml, self.mass, self.quant, self.posTarget, self.rDens, self.rQuant,
                                   self.rDensT, self.rQuantT, self.boxSizeImg, self.boxSizeSim, self.boxCen, self.axes,
                                   self.ndims, self.nPixels, self.colDens, )  # fmt: skip

    # create threads
    threads = [mapThread(threadNum, nThreads) for threadNum in np.arange(nThreads)]

    # launch each thread, detach, and then wait for each to finish
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

        # after each has finished, add its result array to the global
        if posTarget is None:
            if maxIntProj:
                # maximum intensity projection
                w = np.where(thread.rQuant > rQuant)
                rQuant[w] = thread.rQuant[w]
                rDens[w] = thread.rDens[w]
            elif minIntProj:
                # minimum intensity projection
                w = np.where(thread.rQuant < rQuant)
                rQuant[w] = thread.rQuant[w]
                rDens[w] = thread.rDens[w]
            else:
                # normal
                rQuant += thread.rQuant
                rDens += thread.rDens
        else:
            rQuantT += np.nan_to_num(thread.rQuantT)
            rDensT += np.nan_to_num(thread.rDensT)

    # with targets? then replace rQuant/rDens with rQuantT/rDensT for return
    if posTarget is not None:
        rDens = rDensT
        rQuant = rQuantT

    # finalize and return
    if multi:
        if len(nPixels) == 2:
            return (rDens.T, rQuant.T)
        elif len(nPixels) == 3:
            return (rDens, rQuant)

    if quant.size > 1:
        w = np.where(rDens > 0.0)
        rQuant[w] /= rDens[w]

        if len(nPixels) == 2:
            return rQuant.T
        elif len(nPixels) == 3:
            return rQuant

    if len(nPixels) == 2:
        return rDens.T
    elif len(nPixels) == 3:
        return rDens


def sphMapWholeBox(
    pos, hsml, mass, quant, axes, nPixels, sP, colDens=False, nThreads=8, multi=False, posTarget=None, sliceFac=1.0
):
    """Wrap sphMap() specialized to projecting an entire cosmological/periodic box (or subbox).

    Specifically, (ndims,boxSizeSim,boxSizeImg,boxCen) are derived from sP, and nPixels should
    be input as a single scalar which is then assumed to be square.
    """
    boxSizeImg = sP.boxSize * np.array([1.0, 1.0, 1.0 * sliceFac])
    boxCen = sP.boxSize * np.array([0.5, 0.5, 0.5])

    if sP.isSubbox:
        boxSizeImg = sP.subboxSize[sP.subbox] * np.array([1.0, 1.0, 1.0 * sliceFac])
        boxCen = sP.subboxCen[sP.subbox]

    return sphMap(
        pos=pos,
        hsml=hsml,
        mass=mass,
        quant=quant,
        axes=axes,
        ndims=3,
        boxSizeSim=sP.boxSize,
        boxSizeImg=boxSizeImg,
        boxCen=boxCen,
        nPixels=[nPixels, nPixels],
        colDens=colDens,
        nThreads=nThreads,
        multi=multi,
        posTarget=posTarget,
    )


def sphGridWholeBox(sP, pos, hsml, mass, quant, nCells=32):
    """Wrap sphMap(), produce a 3D grid of the given pt/field combination for an entire box."""
    boxSizeImg = sP.boxSize * np.array([1.0, 1.0, 1.0])
    boxCen = sP.boxSize * np.array([0.5, 0.5, 0.5])

    if sP.isSubbox:
        boxSizeImg = sP.subboxSize[sP.subbox] * np.array([1.0, 1.0, 1.0])
        boxCen = sP.subboxCen[sP.subbox]

    return sphMap(
        pos=pos,
        hsml=hsml,
        mass=mass,
        quant=quant,
        axes=[0, 1],
        ndims=3,
        boxSizeSim=sP.boxSize,
        boxSizeImg=boxSizeImg,
        boxCen=boxCen,
        nPixels=[nCells, nCells, nCells],
    )
