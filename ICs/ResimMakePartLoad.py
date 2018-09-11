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

def _get_ic_inds(sP, dmIDs_halo, simpleMethod=False):
    """ Helper function for below, return the DM particle indices from the ICs snapshot corresponding to dmIDs_halo. """
    from util.helper import pSplitRange
    from tracer.tracerMC import match3
    from os.path import isfile
    from sys import stdout

    if simpleMethod:
        # OLD: non-caching method, identical return
        start_time = time.time()
        sP.setSnap('ics')
        dmIDs_ics = sP.snapshotSubsetP('dm', 'ids')
        print(' load done, took [%g] sec.' % (time.time()-start_time))
        next_time = time.time()

        inds_ics, inds_halo = match3(dmIDs_ics, dmIDs_halo)
        assert inds_ics.size == inds_halo.size == dmIDs_halo.size
        print(' match done, took [%g] sec.' % (time.time()-next_time))

        return inds_ics

    # NEW method
    assert 0 # needs to be debugged
    idCacheFile = sP.derivPath + 'cache/sorted_dm_ids_ics.hdf5'
    if not isfile(idCacheFile):
        # make new
        sP.setSnap('ics')
        dmIDs_ics = sP.snapshotSubsetP('dm', 'ids')
        print(' load done, took [%g] sec.' % (time.time()-start_time))
        next_time = time.time()

        # sort and save
        sort_inds = np.argsort(dmIDs_ics)
        ids_sorted = dmIDs_ics[sort_inds]
        dmIDs_ics = None

        with h5py.File(idCacheFile,'w') as f:
            f['sort_inds'] = sort_inds
            f['ids_sorted'] = ids_sorted

        print('Wrote [%s], quit and restart for memory.')
        return

    # load sorted IDs chunk by chunk, run an independent match() on each
    print('Using cache file: [%s]' % idCacheFile)
    nDMPart = sP.snapshotHeader()['NumPart'][sP.ptNum('dm')]
    nChunks = 10

    nMatched = 0
    inds_ics = np.zeros( dmIDs_halo.size, dtype='int64' )

    for i in range(nChunks):
        range_loc = pSplitRange([0,nDMPart], nChunks, i)
        print(' %d%%' % (float(i)/nChunks*100), end='')
        stdout.flush()

        with h5py.File(idCacheFile,'r') as f:
            sort_inds = f['sort_inds'][range_loc[0]:range_loc[1]]
            ids_sorted = f['ids_sorted'][range_loc[0]:range_loc[1]]

        next_time = time.time()
        inds_ics_loc, inds_halo = match3(ids_sorted, dmIDs_halo, firstSorted=True)

        if inds_halo is None:
            continue # no matches in current chunk

        # unsort indices and stamp
        inds_ics[nMatched : nMatched+inds_ics_loc.size] = sort_inds[inds_ics_loc]
        nMatched += inds_ics_loc.size

        if nMatched > dmIDs_halo.size:
            raise Exception('Not expected.')
        if nMatched == dmIDs_halo.size:
            break # done early

    assert nMatched == dmIDs_halo.size
    print(' done.')
    return inds_ics

def generate(fofID, EnlargeHighResFactor=None):
    """ Create zoom particle load and save. """
    from util.simParams import simParams

    # config
    sP = simParams(res=2048,run='tng_dm',redshift=0.0)
    MaxLevel = 11 # e.g. 9=512^3, 11=2048^3
    MinLevel = 6 # 2^5=32 coarsest background, 2^6=64
    ZoomFactor = 3
    Angle = 0.1
    if EnlargeHighResFactor is None: EnlargeHighResFactor = 2.0

    floatType = 'float64' # float64 == DOUBLEPRECISION, otherwise float32
    idType = 'int64' # int64 == LONGIDS, otherwise int32
    saveFilename = 'partload_%s_halo%d_L%d_sf%.1f.hdf5' % (sP.simName, fofID, MaxLevel+(ZoomFactor-1), EnlargeHighResFactor)

    # load halo DM positions and IDs at target snapshot
    sP_snap = sP.snap
    start_time = time.time()

    halo = sP.groupCatSingle(haloID=fofID)
    haloLen    = halo['GroupLenType'][sP.ptNum('dm')]
    haloPos    = halo['GroupPos']
    haloVirRad = sP.units.codeLengthToMpc(halo['Group_R_Crit200'])
    haloM200   = sP.units.codeMassToLogMsun(halo['Group_M_Crit200'])

    print('Halo [%d] pos: %.1f %.1f %.1f, length: [%d], m200 [%.2f] rvir [%.2f pMpc], loading IDs and crossmatching for positions...' % \
        (fofID,haloPos[0],haloPos[1],haloPos[2],haloLen,haloM200,haloVirRad))

    dmIDs_halo = sP.snapshotSubset('dm', 'ids', haloID=fofID)
    assert haloLen == dmIDs_halo.size

    # locate dm particle indices in ICs of this halo
    inds_ics = _get_ic_inds(sP, dmIDs_halo, simpleMethod=True)

    # for dm particles in ICs, load positions of halo DM particles
    posInitial = sP.snapshotSubsetP('dm', 'pos', inds=inds_ics)
    ext_min = np.min(posInitial, axis=0)
    ext_max = np.max(posInitial, axis=0)
    extent_frac = np.max( ext_max - ext_min ) / sP.boxSize

    print('Initial DM positions extent, x [%.1f - %.1f], y [%.1f - %.1f], z [%.1f - %.1f], max box fraction = %.3f' % \
        (ext_min[0], ext_max[0], ext_min[1], ext_max[1], ext_min[2], ext_max[2], extent_frac))

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
    next_time = time.time()

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
    print(' done, took [%g] sec.' % (time.time()-next_time))

    w = np.where(Grids[GridsOffset[MaxLevel]:] == 1)
    vol_frac = float(len(w[0])) / Grids[GridsOffset[MaxLevel]:].size * 100
    print('Volume fraction of box occupied by high-res region in ICs = %.3f%%' % vol_frac)

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

    headerExtra = {'GroupCM':cmInitial, 'MinLevel':MinLevel, 'MaxLevel':MaxLevel, 'ZoomFactor':ZoomFactor, 'Boxsize':sP.boxSize, 
                   'Sim_name':np.string_(sP.simName), 'Sim_snap':sP_snap, 'Sim_fofID':fofID, 'InitTime':sP.scalefac, 
                   'InitBoxVolFrac':vol_frac, 'InitBoxExtentFrac':extent_frac, 'EnlargeHighResFactor':EnlargeHighResFactor}

    write_ic_file(saveFilename, partTypes, sP.boxSize, massTable=massTable, headerExtra=headerExtra)
    print(' Done (starting z = %.1f, a = %f) (total: %.1f sec).' % (sP.redshift, sP.scalefac, time.time()-start_time))

def generate_set():
    haloIDs = [3232] #[50,51,90]
    sizeFacs = [3.0] #[2.0,3.0,4.0]

    for haloID in haloIDs:
        for sizeFac in sizeFacs:
            generate(fofID=haloID, EnlargeHighResFactor=sizeFac)

