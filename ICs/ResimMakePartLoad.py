"""
ics/ResimMakePartLoad.py
  Cosmological zoom/resimulation ICs: re-write of Volker's P-Resim-MakePartLoad.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
#from builtins import * # future.types.newint interferes with numba typing

import numpy as np
import h5py
import time

from ICs.utilities import write_ic_file
from numba import jit

@jit(nopython=True, cache=True)
def _fof_periodic_wrap(x, BoxSize):
    """ Helper. Equivalent to util.sphMap._NEAREST_POS(). """
    while (x >= BoxSize):
        x -= BoxSize

    while (x < 0):
        x += BoxSize

    return x

@jit(nopython=True, cache=True)
def _fof_periodic(x, BoxSize):
    """ Helper. Equivalent to util.sphMap._NEAREST(). """
    if x >= 0.5*BoxSize:
        x -= BoxSize

    if x < -0.5*BoxSize:
        x += BoxSize

    return x

@jit(nopython=True, cache=True)
def _PER(x, dim):
    """ Helper. #define PER(x) (x < 0 ? (x+dim) : (x >= dim ? (x-dim):(x))) """
    if x < 0:
        return x + dim
    else:
        if x >= dim:
            return x - dim
        else:
            return x

@jit(nopython=True, cache=True)
def _get_center_of_mass(posInitial, BoxSize):
    """ See generate(). """
    cm = np.zeros( 3, dtype=np.float64 )

    ref = posInitial[0,:]
    N = posInitial.shape[0]

    for i in range(N):
        for j in range(3):
            cm[j] += _fof_periodic(posInitial[i,j] - ref[j], BoxSize)

    cm /= N

    for j in range(3):
        cm[j] = _fof_periodic_wrap(cm[j] + ref[j], BoxSize)

    return cm

@jit(nopython=True, cache=True)
def _mark_high_res_cells(Grids, GridsOffset, level, BoxSize, posInitial, cmInitial):
    """ See generate(). """
    dim = 1 << level
    fac = dim / BoxSize

    N = posInitial.shape[0]

    for m in range(N):
        x = _fof_periodic_wrap( posInitial[m,0] - cmInitial[0] + 0.5 * BoxSize, BoxSize )
        y = _fof_periodic_wrap( posInitial[m,1] - cmInitial[1] + 0.5 * BoxSize, BoxSize )
        z = _fof_periodic_wrap( posInitial[m,2] - cmInitial[2] + 0.5 * BoxSize, BoxSize )

        i = int(fac * x)
        j = int(fac * y)
        k = int(fac * z)

        if i >= dim:
            i = dim - 1
        if j >= dim:
            j = dim - 1
        if k >= dim:
            k = dim - 1

        ind = (i * dim + j) * dim + k + GridsOffset[level]

        Grids[ind] = 1

@jit(nopython=True, cache=True)
def _enlarge_high_res_cells(Grids, GridsOffset, MaxLevel, BoxSize, EnlargeHighResFactor):
    """ See generate(). """
    count = 0
    dim = 1 << MaxLevel
    size = dim * dim * dim

    a_in = np.zeros(size, dtype=np.int8)
    a_out = np.zeros(size, dtype=np.int8)

    for i in range(size):
        a_in[i] = Grids[i + GridsOffset[MaxLevel]]

        if a_in[i]:
            count += 1

    radius = np.power(count/(4*np.pi/3), 1.0/3.0) * BoxSize / dim

    #print('We start with %d cells, using radius = %g' % (count,radius))

    count_now = 0

    while count_now < EnlargeHighResFactor * count:
        # loop over all cells at the highest grid level
        for x in range(dim):
            for y in range(dim):
                for z in range(dim):
                    ind = (x * dim + y) * dim + z # no GridsOffset, since a_in is just the high-res grid

                    # flagged high res cell, also flag its neighbors
                    if a_in[ind]:
                        for xx in range(-1,2):
                            for yy in range(-1,2):
                                for zz in range(-1,2):
                                    xxx = _PER(x+xx, dim)
                                    yyy = _PER(y+yy, dim)
                                    zzz = _PER(z+zz, dim)

                                    ind_out = (xxx * dim + yyy) * dim + zzz # similarly no GridsOffset
                                    a_out[ind_out] = 1

        # transfer a_out to a_in, and count flagged cells
        count_now = 0

        for i in range(size):
            a_in[i] = a_out[i]
            a_out[i] = 0

            if a_in[i]:
                count_now += 1

        #print(' iter, now have %d cells' % count_now)

    radius = np.power(count_now/(4*np.pi/3), 1.0/3.0) * BoxSize / dim

    #print('Finished, now we use radius = %g' % radius)

    for i in range(size):
        Grids[i + GridsOffset[MaxLevel]] = a_in[i]

    return radius

@jit(nopython=True, cache=True)
def _build_parent_grid(Grids, GridsOffset, MaxLevel):
    """ See generate(). """
    for level in range(MaxLevel-1, -1, -1):
        dim = 1 << level

        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    ind = (i * dim + j) * dim + k + GridsOffset[level]

                    # examine next high-res grid, mark this current level as high-res if it is covered
                    for x in [0,1]:
                        for y in [0,1]:
                            for z in [0,1]:
                                ind2 = ((2*i+x) * 2*dim + (2*j+y)) * 2*dim + (2*k+z) + GridsOffset[level + 1]

                                if Grids[ind2]:
                                    Grids[ind] = 1

@jit(nopython=True, cache=True)
def _find_partload_size(level, i, j, k, Radius, Angle, PartCount, BoxSize, MaxLevel, MinLevel, ZoomFactor, Grids, GridsOffset):
    """ See generate(). Recursively called. (pIndex < 0 case)."""
    dim = 1 << level
    ind = (i * dim + j) * dim + k + GridsOffset[level]
    cell = BoxSize / dim

    sx = (i + 0.5) * cell - 0.5 * BoxSize
    sy = (j + 0.5) * cell - 0.5 * BoxSize
    sz = (k + 0.5) * cell - 0.5 * BoxSize

    dist = np.sqrt(sx*sx + sy*sy + sz*sz)
    if dist > Radius:
        theta = cell / (dist - Radius)
    else:
        theta = 2 * Angle

    if (Grids[ind] & (level < MaxLevel)) or (level < MinLevel) or ( (theta > Angle) & (level < MaxLevel) ):
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    # recurse to next higher res level
                    _find_partload_size(level+1, i*2+x, j*2+y, k*2+z, 
                        Radius, Angle, PartCount, BoxSize, MaxLevel, MinLevel, ZoomFactor, Grids, GridsOffset)
    else:
        if level == MaxLevel:
            if Grids[ind]:
                # high-res cell particles
                PartCount[1] += ZoomFactor**3
            else:
                # medium (original res)
                PartCount[2] += 1
        else:
            # coarse
            PartCount[3] += 1

@jit(nopython=True, cache=True)
def _generate_grid(level, i, j, k, Radius, Angle, pIndex, BoxSize, MaxLevel, MinLevel, ZoomFactor, Grids, GridsOffset, 
                   P_Type, P_Pos, P_Mass):
    """ See generate(). Recursively called. (pIndex >= 0 case). """
    dim = 1 << level
    ind = (i * dim + j) * dim + k + GridsOffset[level]
    cell = BoxSize / dim

    sx = (i + 0.5) * cell - 0.5 * BoxSize
    sy = (j + 0.5) * cell - 0.5 * BoxSize
    sz = (k + 0.5) * cell - 0.5 * BoxSize

    dist = np.sqrt(sx*sx + sy*sy + sz*sz)
    if dist > Radius:
        theta = cell / (dist - Radius)
    else:
        theta = 2 * Angle

    if (Grids[ind] & (level < MaxLevel)) or (level < MinLevel) or ( (theta > Angle) & (level < MaxLevel) ):
        for x in [0,1]:
            for y in [0,1]:
                for z in [0,1]:
                    # recurse to next higher res level
                    _generate_grid(level+1, i*2+x, j*2+y, k*2+z, 
                        Radius, Angle, pIndex, BoxSize, MaxLevel, MinLevel, ZoomFactor, Grids, GridsOffset, P_Type, P_Pos, P_Mass)
    else:
        sx = (i + 0.5) * cell
        sy = (j + 0.5) * cell
        sz = (k + 0.5) * cell

        if level == MaxLevel:
            if Grids[ind]:
                # generate ZoomFactor**3 particles in high-res cell
                for x in range(ZoomFactor):
                    for y in range(ZoomFactor):
                        for z in range(ZoomFactor):
                            P_Type[pIndex] = 1
                            P_Pos[pIndex,0] = sx + (-0.5 + (x + 0.5) / ZoomFactor) * cell
                            P_Pos[pIndex,1] = sy + (-0.5 + (y + 0.5) / ZoomFactor) * cell
                            P_Pos[pIndex,2] = sz + (-0.5 + (z + 0.5) / ZoomFactor) * cell
                            pIndex += 1
            else:
                # generate 1 particle in medium-res cell
                P_Type[pIndex] = 2
                P_Pos[pIndex,0] = sx
                P_Pos[pIndex,1] = sy
                P_Pos[pIndex,2] = sz
                pIndex += 1
        else:
            # generate 1 particle in coarse cell
            P_Type[pIndex] = 3
            P_Pos[pIndex,0] = sx
            P_Pos[pIndex,1] = sy
            P_Pos[pIndex,2] = sz
            P_Mass[pIndex] = 1.0 / (dim*dim*dim)
            pIndex += 1

def generate():
    """ Create zoom particle load and save. """
    from util.simParams import simParams
    from cosmo.load import groupCatSingle, snapshotSubset
    from tracer.tracerMC import match3

    # config
    saveFilename = 'out.hdf5'
    sP = simParams(res=455,run='tng',redshift=0.0)
    fofID = 10
    MaxLevel = 7 # 9=512^3
    MinLevel = 4
    ZoomFactor = 3
    Angle = 0.1
    EnlargeHighResFactor = 2.5
    floatType = 'float64' # float64 == DOUBLEPRECISION, otherwise float32
    idType = 'int64' # int64 == LONGIDS, otherwise int32

    # load halo DM positions and IDs at target snapshot
    start_time = time.time()
    halo = groupCatSingle(sP, haloID=fofID)
    haloLen = halo['GroupLenType'][sP.ptNum('dm')]
    print('Halo [%d] length: [%d], loading IDs and crossmatching for positions...' % (fofID,haloLen))

    dmIDs_halo = snapshotSubset(sP, 'dm', 'ids', haloID=fofID)
    assert haloLen == dmIDs_halo.size

    # load parent box initial conditions
    sP.setSnap('ics')
    dmIDs_ics = snapshotSubset(sP, 'dm', 'ids')

    inds_ics, inds_halo = match3(dmIDs_ics, dmIDs_halo)
    assert inds_ics.size == inds_halo.size == dmIDs_halo.size
    print(' done, took [%g] sec.' % (time.time()-start_time))

    # locate dm particles in ICs, load positions of halo DM particles
    dmPos_ics = snapshotSubset(sP, 'dm', 'pos')
    posInitial = dmPos_ics[inds_ics,:]

    # initialize grid
    GridDim = np.zeros( MaxLevel+1, dtype='int32')
    GridsSize = 0
    GridsOffset = np.zeros( [MaxLevel+1], dtype='int64')

    for level in range(MaxLevel+1):
        dim = 1 << level
        GridDim[level] = dim
        GridsOffset[level] = GridsSize
        GridsSize += dim*dim*dim

    Grids = np.zeros(GridsSize, dtype='int8')
    print('Allocated grid of [%d] elements (%.3f GB), creating particle load...' % (Grids.size,float(Grids.size)*1/1024/1024/1024))
    start_time = time.time()

    # mark high resolution region, enlarge, and build parent grid
    cmInitial = _get_center_of_mass(posInitial, sP.boxSize)

    _mark_high_res_cells(Grids, GridsOffset, MaxLevel, sP.boxSize, posInitial, cmInitial)

    Radius = _enlarge_high_res_cells(Grids, GridsOffset, MaxLevel, sP.boxSize, EnlargeHighResFactor)

    _build_parent_grid(Grids, GridsOffset, MaxLevel)

    # count particles that will be generated, and allocate P[]
    PartCount = np.zeros( 6, dtype='int32' )

    _find_partload_size(0,0,0,0,Radius,Angle,PartCount,sP.boxSize,MaxLevel,MinLevel,ZoomFactor,Grids,GridsOffset)

    NumPartTot = PartCount.sum()
    P_Type = np.zeros( NumPartTot, dtype='int32' )
    P_Pos  = np.zeros( (NumPartTot,3), dtype=floatType )
    P_Mass = np.zeros( NumPartTot, dtype=floatType )

    # create grid
    pIndex = np.zeros(1, dtype=idType)
    _generate_grid(0,0,0,0,Radius,Angle,pIndex,sP.boxSize,MaxLevel,MinLevel,ZoomFactor,Grids,GridsOffset,P_Type,P_Pos,P_Mass)

    assert pIndex == NumPartTot
    print(' done, took [%g] sec.' % (time.time()-start_time))

    # save
    for i in range(6):
        print(' partType [%d] has [%10d] particles.' % (i,PartCount[i]))
    print('Saving [%s]...' % saveFilename, end='')

    dim = 1 << MaxLevel
    size = dim * dim * dim

    massTable = np.zeros( 6, dtype='float64' )
    if PartCount[1]:
        massTable[1] = 1.0 / size / ZoomFactor**3
    if PartCount[2]:
        massTable[2] = 1.0 / size

    # generate a partTypes dict
    idOffset = 0
    partTypes = {}

    for ptNum in [1,2,3]:
        # separate out into different types
        if PartCount[ptNum] == 0:
            continue

        w = np.where(P_Type == ptNum)
        assert len(w[0]) == PartCount[ptNum]

        # generate IDs
        ids = np.arange( PartCount[ptNum], dtype=idType ) + idOffset
        idOffset += PartCount[ptNum]

        gName = 'PartType%d' % ptNum
        partTypes[gName] = {'Coordinates':np.squeeze(P_Pos[w,:]), 'ParticleIDs':ids}

        if ptNum == 3:
            # add masses for variable mass particle type
            partTypes[gName]['Masses'] = P_Mass[w]

    write_ic_file(saveFilename, partTypes, sP.boxSize, massTable=massTable, headerExtra={'GroupCM':cmInitial})
    print(' Done.')
