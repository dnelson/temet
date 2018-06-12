"""
util/subfind.py
  Implementation of (serial) subfind algorithm.
"""
from __future__ import (absolute_import,division,print_function)#,unicode_literals)
# unicode_literals: https://github.com/numpy/numpy/issues/2407 (to be fixed in numpy 1.15)
from builtins import *

import numpy as np
import time
import glob
from numba import jit
from util.sphMap import _NEAREST

# DOUBLEPRECISION == 1
MyFloat = np.float64
MySingle = np.float64
MyDouble = np.float64
MyIDType = np.uint64 # LONGIDS
integertime = np.int32 # ~ENLARGE_DYNAMIC_RANGE_IN_TIME

GFM_N_CHEM_ELEMENTS = 10
GFM_N_CHEM_TAGS = 6
GRAVCOSTLEVELS = 6
MAXSCALARS = GFM_N_CHEM_ELEMENTS + 1 + 1 # SECOND +1: BUG FROM GFM_DUST

NTYPES = 6
NSOFTTYPES_HYDRO = 64 # ADAPTIVE_HYDRO_SOFTENING
AdaptiveHydroSofteningSpacing = 1.2
GasSoftFactor = 2.5
ErrTolThetaSubfind = 0.7
DesLinkNgb = 20

# L35n2160TNG:
NSOFTTYPES = 6
MinimumComovingHydroSoftening = 0.05
SofteningComoving = [0.390, 0.390, 0.390, 0.670, 1.15, 2.0]
SofteningMaxPhys  = [0.195, 0.195, 0.390, 0.670, 1.15, 2.0]

# L25n256_0000 TESTING:
#NSOFTTYPES = 5
#MinimumComovingHydroSoftening = 0.25
#SofteningComoving = [2.0, 2.0, 2.0, 3.42, 5.84]
#SofteningMaxPhys  = [1.0, 1.0, 2.0, 3.42, 5.84]

# define data types
grad_data_dtype = np.dtype([
    ('dhro', MySingle, 3),
    ('dvel', MySingle, (3,3)),
    ('dpress', MySingle, 3),
    ('dB', MySingle, (3,3)), # MHD
    ('dscalars', MySingle, (MAXSCALARS,3)), # MAXSCALARS
])

SphP_dtype = np.dtype([
    # conserved variables
    ('Energy', MyFloat),
    ('Momentum', MyFloat, 3),
    ('Volume', MyFloat),
    ('OldMass', MyFloat),
    # primitive variables
    ('Density', MyFloat),
    ('Pressure', MyFloat),
    ('Utherm', MySingle),
    ('FullGravAccel', MySingle, 3), # HIERARCHICAL_GRAVITY
    ('MaxMach', MyFloat), # BUG: defined(TRACER_MC_MACHMAX)
    # variables for mesh
    ('Center', MyDouble, 3),
    ('VelVertex', MySingle, 3),
    ('MaxDelaunayRadius', MySingle),
    ('Hsml', MySingle),
    ('SurfaceArea', MySingle),
    ('MaxFaceAngle', MySingle), # REGULARIZE_MESH_FACE_ANGLE
    ('ActiveArea', MySingle),
    ('Machnumber', MySingle), # SHOCK_FINDER_BEFORE_OUTPUT
    ('EnergyDissipation', MySingle), # SHOCK_FINDER_BEFORE_OUTPUT
    ('CurlVel', MySingle), # REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED
    ('CurrentMaxTiStep', MySingle), # TREE_BASED_TIMESTEPS
    ('Csnd', MySingle), # TREE_BASED_TIMESTEPS
    # MHD
    ('B', MyFloat, 3),
    ('BConserved', MyFloat, 3),
    ('DivB', MyFloat),
    # GFM_STELLAR_EVOLUTION
    ('Metallicity', MyFloat),
    ('MassMetallicity', MyFloat),
    ('MetalsFraction', MyFloat, GFM_N_CHEM_ELEMENTS),
    ('MassMetals', MyFloat, GFM_N_CHEM_ELEMENTS),
    # GFM_CHEMTAGS
    ('MassMetalsChemTags', MyFloat, GFM_N_CHEM_TAGS),
    ('MassMetalsChemTagsFraction', MyFloat, GFM_N_CHEM_TAGS),
    ('MinimumEdgeDistance', MySingle), # REFINEMENT_SPLIT_CELLS
    ('Ne', MyFloat), # COOLING
    ('Sfr', MySingle), # USE_SFR
    ('HostHaloMass', MySingle), # GFM_WINDS_VARIABLE
    #('DMVelDisp', MySingle), # GFM_WINDS_VARIABLE (UNION)
    ('AGNBolIntensity', MySingle), # GFM_AGN_RADIATION
    ('Injected_BH_Energy', MySingle), # BH_THERMALFEEDBACK
    ('Injected_BH_Wind_Momentum', MyFloat, 3), # BH_ADIOS_WIND
    ('Grad', grad_data_dtype),
    ('first_connection', np.int32), # VORONOI_DYNAMIC_UPDATE
    ('last_connection', np.int32), # VORONOI_DYNAMIC_UPDATE
    ('SepVector', MySingle, 3), # REFINEMENT_SPLIT_CELLS
    ('TimeLastPrimUpdate', np.float64),
])

SphP_dtype_mem = np.dtype([
    # conserved variables
    ('Energy', MyFloat),
    ('Momentum', MyFloat, 3),
    ('Volume', MyFloat),
    # primitive variables
    ('Density', MyFloat),
    ('Pressure', MyFloat),
    ('Utherm', MySingle),
    # variables for mesh
    ('Center', MyDouble, 3),
    ('Machnumber', MySingle), # SHOCK_FINDER_BEFORE_OUTPUT
    ('EnergyDissipation', MySingle), # SHOCK_FINDER_BEFORE_OUTPUT
    # MHD
    ('B', MyFloat, 3),
    ('DivB', MyFloat),
    # GFM_STELLAR_EVOLUTION
    ('Metallicity', MyFloat),
    ('MetalsFraction', MyFloat, GFM_N_CHEM_ELEMENTS),
    ('Ne', MyFloat), # COOLING
    ('Sfr', MySingle), # USE_SFR
])

StarP_dtype = np.dtype([ # GFM
    ('PID', np.uint32),
    ('pad_0', np.int32), # STRUCT PADDING
    ('BirthTime', MyFloat),
    ('BirthPos', MyFloat, 3),
    ('BirthVel', MyFloat, 3),
    # GFM_STELLAR_EVOLUTION
    ('InitialMass', MyDouble),
    ('MassMetals', MyFloat, GFM_N_CHEM_ELEMENTS),
    ('Metallicity', MyFloat),
    ('SNIaRate', MyFloat),
    ('SNIIRate', MyFloat),
    ('lastEnrichTime', MyDouble), # GFM_DISCRETE_ENRICHMENT
    ('MassMetalsChemTags', MyFloat, GFM_N_CHEM_TAGS), # GFM_CHEMTAGS
    ('Hsml', MyFloat), # GFM_STELLAR EVOLUTION || GFM_WINDS
    ('Utherm', MyFloat) # GFM_WINDS
])

BHP_dtype = np.dtype([ # BLACK_HOLES
    ('PID', np.uint32),
    ('BH_CountProgs', np.int32),
    ('BH_NumNgb', MyFloat),
    ('BH_Hsml', MyFloat),
    ('BH_Mass', MyFloat),
    ('BH_Mdot', MyFloat),
    ('BH_MdotBondi', MyFloat),
    ('BH_MdotEddington', MyFloat),
    ('BH_CumMass_QM', MyFloat),
    ('BH_CumEgy_QM', MyFloat),
    ('BH_CumMass_RM', MyFloat),
    ('BH_CumEgy_RM', MyFloat),
    ('BH_DtGasNeighbor', MyFloat),
    ('BH_VolSum', MyFloat),
    ('BH_Density', MyFloat),
    ('BH_U', MyFloat),
    ('BH_Pressure', MyFloat),
    ('BH_SurroundingGasVel', MyFloat, 3),
    ('SwallowID', MyIDType),
    ('HsmlCentering', MyFloat), # BH_NEW_CENTERING
    # DRAINGAS
    ('NearestDist', MyFloat),
    ('DrainID', MyIDType),
    ('CellDensity', MyFloat),
    ('CellUtherm', MyFloat),
    ('DrainBucketMass', MyDouble),
    ('BH_WindEnergy', MyFloat), # BH_ADIOS_WIND
    ('WindDir', MyFloat, 3), # BH_ADIOS_RANDOMIZED
    ('HostHaloMass', MyFloat), # GFM_AGN_RADIATION
    ('BH_DMVelDisp', MyFloat), # GFM_WINDS_VARIABLE
    ('BH_ThermEnergy', MyFloat), # BH_THERMALFEEDBACK,
    ('BH_Bpress', MyFloat) # BH_USE_ALFVEN_SPEED_IN_BONDI
])

P_dtype = np.dtype([
    ('Pos', MyDouble, 3),
    ('Mass', MyDouble),
    ('Vel', MyFloat, 3),
    ('GravAccel', MySingle, 3),
    ('GravPM', MySingle, 3), # PMGRID
    #('BirthPos', MySingle, 3), # L35n2160TNG_FIX_E7BF4CF
    #('BirthVel', MySingle, 3), # L35n2160TNG_FIX_E7BF4CF
    ('Potential', MySingle), # EVALPOTENTIAL
    ('PM_Potential', MySingle), # EVALPOTENTIAL & PMGRID
    ('AuxDataID', MyIDType), # GFM || BLACK_HOLES
    # TRACER_MC
    ('TracerHead', np.int32),
    ('NumberOfTracers', np.int32),
    ('OriginTask', np.int32),
    # general
    ('ID', MyIDType),
    ('pad_0', np.int32, 2), # STRUCT PADDING
    ('TI_Current', integertime),
    ('OldAcc', np.float32),
    ('GravCost', np.float32, GRAVCOSTLEVELS),
    ('Type', np.uint8),
    ('SofteningType', np.uint8),
    ('TimeBinGrav', np.int8),
    ('TimeBinHydro', np.int8)
])

P_dtype_mem = np.dtype([
    ('Pos', MyDouble, 3),
    ('Mass', MyDouble),
    ('Vel', MyFloat, 3),
    ('Potential', MySingle), # EVALPOTENTIAL
    ('AuxDataID', MyIDType), # GFM || BLACK_HOLES
    ('ID', MyIDType),
    ('Type', np.uint8),
    ('SofteningType', np.uint8),
])

PS_dtype = np.dtype([
    ('OriginIndex', np.int32),
    ('OriginTask', np.int32),
    ('TargetIndex', np.int32),
    ('TargetTask', np.int32),
    ('GrNr', np.int32),
    # SUBFIND
    ('SubNr', np.int32),
    ('OldIndex', np.int32),
    ('submark', np.int32),
    ('originindex', np.int32),
    ('origintask', np.int32),
    ('Utherm', MyFloat),
    ('Density', MyFloat),
    ('Potential', MyFloat),
    ('Hsml', MyFloat),
    ('BindingEnergy', MyFloat),
    ('Center', MyDouble, 3), # CELL_CENTER_GRAVITY
    # SAVE_HSML_IN_SNAPSHOT
    ('SubfindHsml', MyFloat),
    ('SubfindDensity', MyFloat),
    ('SubfindDMDensity', MyFloat),
    ('SubfindVelDisp', MyFloat)
])

PS_dtype_mem = np.dtype([
    ('GrNr', np.int32),
    # SUBFIND
    ('SubNr', np.int32),
    ('OldIndex', np.int32),
    ('Utherm', MyFloat),
    ('Density', MyFloat),
    ('Potential', MyFloat),
    ('Hsml', MyFloat),
    ('BindingEnergy', MyFloat),
    ('Center', MyDouble, 3), # CELL_CENTER_GRAVITY
    # SAVE_HSML_IN_SNAPSHOT
    ('SubfindHsml', MyFloat),
    ('SubfindDensity', MyFloat),
    ('SubfindDMDensity', MyFloat),
    ('SubfindVelDisp', MyFloat)
])

cand_dtype = np.dtype([
    ('head', np.int32),
    ('len', np.int32),
    ('nsub', np.int32),
    ('rank', np.int32),
    ('subnr', np.int32),
    ('parent', np.int32),
    ('bound_length', np.int32)
])

ud_dtype = np.dtype([
    ('index', np.int32)
])

#@jit(nopython=True, nogil=True) #, cache=True)

def load_custom_dump(sP, GrNr):
    """ Load groups_{snapNum}/fof_{fofID}.{taskNum} custom binary data dump. """
    import struct

    filePath = sP.simPath + 'groups_%03d/fof_%d/fof_%d' % (sP.snap,GrNr,GrNr)

    headerSize = 4*5 + 4*5 + 4*4

    # first loop: load all headers, accumulate total counts
    nChunks = len(glob.glob(filePath + '.*'))
    print('nChunks: [%d], reading headers...' % nChunks)

    NumP = 0
    NumSphP = 0
    NumStarP = 0
    NumBHP = 0

    for i in range(nChunks):
        file = '%s.%d' % (filePath,i)
        if i % 100 == 0: print(file)
        with open(file,'rb') as f:
            header = f.read(headerSize)

        TargetGrNr, NumP_loc, NumSphP_loc, NumStarP_loc, NumBHP_loc = struct.unpack('iiiii', header[0:20])
        min_ind, max_ind, min_sph_ind, max_sph_ind = struct.unpack('iiii', header[20:36])
        size_PS, size_P, size_SphP, size_StarP, size_BHP = struct.unpack('iiiii', header[36:56])

        NumP += NumP_loc
        NumSphP += NumSphP_loc
        NumStarP += NumStarP_loc
        NumBHP += NumBHP_loc

        # verify struct configurations (ugh padding)
        assert TargetGrNr == GrNr
        assert size_PS == PS_dtype.itemsize
        assert size_P == P_dtype.itemsize
        assert size_SphP == SphP_dtype.itemsize
        assert size_StarP == StarP_dtype.itemsize
        assert size_BHP == BHP_dtype.itemsize

    # allocate
    group = sP.groupCatSingle(haloID=GrNr)
    print('GrNr', GrNr, ' has LenType: ', group['GroupLenType'], ' Len:', group['GroupLen'])

    NumDM = NumP - NumSphP - NumStarP - NumBHP
    print('Save files have total lenType: ', NumSphP, NumDM, NumStarP, NumBHP, ' Len:', NumP)

    sizeBytes = NumP*size_P + NumP*size_PS + NumSphP*size_SphP + NumStarP*size_StarP + NumBHP*size_BHP
    sizeGB = sizeBytes / 1024.0**3
    print('Memory allocation for full arrays, all particles, would require [%.2f GB]' % sizeGB)

    sizeBytes = group['GroupLen'] * size_P + \
                group['GroupLen'] * size_PS + \
                group['GroupLenType'][sP.ptNum('gas')] * size_SphP + \
                group['GroupLenType'][sP.ptNum('stars')] * size_StarP + \
                group['GroupLenType'][sP.ptNum('bhs')] * size_BHP
    sizeGB = sizeBytes / 1024.0**3
    print('Memory allocation for full arrays, group members only, would require [%.2f GB]' % sizeGB)

    sizeBytes = group['GroupLen'] * P_dtype_mem.itemsize + \
                group['GroupLen'] * PS_dtype_mem.itemsize + \
                group['GroupLenType'][sP.ptNum('gas')] * SphP_dtype_mem.itemsize + \
                group['GroupLenType'][sP.ptNum('stars')] * StarP_dtype.itemsize + \
                group['GroupLenType'][sP.ptNum('bhs')] * BHP_dtype.itemsize
    sizeGB = sizeBytes / 1024.0**3
    print('Memory allocation for partial arrays, group members only, will require [%.2f GB]' % sizeGB)

    P     = np.zeros(group['GroupLen'], dtype=P_dtype_mem)
    PS    = np.zeros(group['GroupLen'], dtype=PS_dtype_mem) # memset(0) in fof.c
    SphP  = np.empty(group['GroupLenType'][sP.ptNum('gas')], dtype=SphP_dtype_mem)
    StarP = np.empty(group['GroupLenType'][sP.ptNum('stars')], dtype=StarP_dtype)
    BHP   = np.empty(group['GroupLenType'][sP.ptNum('bhs')], dtype=BHP_dtype)

    NumP = 0
    NumSphP = 0
    NumStarP = 0
    NumBHP = 0

    for i in range(nChunks):
        file = '%s.%d' % (filePath,i)

        with open(file,'rb') as f:
            header = f.read(headerSize)
            TargetGrNr, NumP_loc, NumSphP_loc, NumStarP_loc, NumBHP_loc = struct.unpack('iiiii', header[0:20])
            min_ind, max_ind, min_sph_ind, max_sph_ind = struct.unpack('iiii', header[20:36])

            PS_temp    = np.fromfile(f, dtype=PS_dtype, count=NumP_loc)
            P_temp     = np.fromfile(f, dtype=P_dtype, count=NumP_loc)
            SphP_temp  = np.fromfile(f, dtype=SphP_dtype, count=NumSphP_loc)
            StarP_temp = np.fromfile(f, dtype=StarP_dtype, count=NumStarP_loc)
            BHP_temp   = np.fromfile(f, dtype=BHP_dtype, count=NumBHP_loc)

        # particles
        w = np.where(PS_temp['GrNr'] == GrNr)
        gr_NumP = len(w[0])
        print('[%4d] [%7d of %7d] particles belong to group, now have [%9d of %9d] total.' % (i,gr_NumP,PS_temp.size,NumP+gr_NumP,group['GroupLen']))

        # only save needed fields to optimize memory usage
        for field in PS_dtype_mem.names:
            PS[field][NumP:NumP+gr_NumP] = PS_temp[field][w]
        for field in P_dtype_mem.names:
            P[field][NumP:NumP+gr_NumP] = P_temp[field][w]

        assert P_temp['Type'].min() >= 0 and P_temp['Type'].max() < NTYPES # sanity check for struct padding success
        assert P_temp['TimeBinGrav'].min() >= 25 and P_temp['TimeBinGrav'].max() <= 52
        assert P_temp['Pos'].min() >= 0.0 and P_temp['Pos'].max() <= sP.boxSize

        # gas
        if NumSphP_loc:
            w_gas = np.where(P_temp['Type'] == 0)
            assert min_sph_ind == 0
            if len(w_gas[0]) == NumSphP_loc: # full
                assert max_sph_ind == NumSphP_loc - 1
            else:
                print(' first partial gas')

            w_gas = np.where( (P_temp['Type'] == 0) & (PS_temp['GrNr'] == GrNr) )
            assert len(w_gas[0]) == NumSphP_loc

            for field in SphP_dtype_mem.names:
                # only save needed fields to optimize memory usage
                SphP[field][NumSphP:NumSphP+NumSphP_loc] = SphP_temp[field][w_gas]

            NumSphP += len(w_gas[0]) # == NumSphP_loc

        # note: AuxDataID's will be good (file local), since we always write full StarP/BHP,
        # but aux[].PID will need to be adjusted by min_ind (not used anyways in subfind)

        # stars
        if NumStarP_loc:
            w_star = np.where( (P_temp['Type'] == 4) & (PS_temp['GrNr'] == GrNr) )
            Pw_aux = P_temp['AuxDataID'][w_star]
            assert Pw_aux.min() >= 0 and Pw_aux.max() < NumStarP_loc

            StarP[NumStarP:NumStarP+NumStarP_loc] = StarP_temp[Pw_aux]

            # reassign AuxData ID as indices into new global StarP
            #P[NumP:NumP+gr_NumP][w_star]['AuxDataID'] = np.arange( len(w_star[0]) ) + NumStarP # TODO
            StarP[NumStarP:NumStarP+NumStarP_loc]['PID'] = w_star[0] + gr_NumP
            #assert np.array_equal(P_temp['AuxDataID'][ StarP_temp[Pw_aux]['PID'] ], Pw_aux) # TODO

            NumStarP += len(w_star[0]) # != NumStarP_loc !

        # bhs
        if NumBHP_loc:
            w_bhs = np.where( (P_temp['Type'] == 5) & (PS_temp['GrNr'] == GrNr) )
            Pw_aux = P_temp['AuxDataID'][w_bhs]
            assert Pw_aux.min() >= 0 and Pw_aux.max() < NumBHP_loc

            BHP[NumBHP:NumBHP+NumBHP_loc] = BHP_temp[Pw_aux]

            # reassign AuxData ID as indices into new global BHP
            #P[NumP:NumP+gr_NumP][w_bhs]['AuxDataID'] = np.arange( len(w_bhs[0]) ) + NumBHP # TODO
            StarP[NumBHP:NumBHP+NumBHP_loc]['PID'] = w_bhs[0] + gr_NumP
            #assert np.array_equal(P_temp['AuxDataID'][ BHP_temp[Pw_aux]['PID'] ], Pw_aux) # TODO

            NumBHP += len(w_bhs[0]) # != NumBHP_loc !

        NumP += gr_NumP        

    # final checks
    assert NumP == group['GroupLen']
    assert NumSphP == group['GroupLenType'][sP.ptNum('gas')]
    assert NumStarP == group['GroupLenType'][sP.ptNum('stars')]
    assert NumBHP == group['GroupLenType'][sP.ptNum('bhs')]

    print('Particle counts of all types verified, match expected Group [%d] lengths.' % GrNr)

    PS['OldIndex'] = np.arange(group['GroupLen'])

    return P, PS, SphP, StarP, BHP

def load_snapshot_data(sP, GrNr):
    """ Load the FoF particles from an actual snapshot for testing. """

    # load group metadata
    dmMass = sP.dmParticleMass
    group = sP.groupCatSingle(haloID=GrNr)
    numPartType = group['GroupLenType']

    print('Group [%d] has total length [%d], and [%d] subhalos.' % (GrNr,group['GroupLen'],group['GroupNsubs']))

    for i in range(group['GroupNsubs']):
        sub = sP.groupCatSingle(subhaloID=group['GroupFirstSub']+i)
        print('subnr ', i, ' len', sub['SubhaloLen'], ' pos:',sub['SubhaloPos'],' mostboundid',sub['SubhaloIDMostbound'],' lentype',sub['SubhaloLenType'])

    # allocate
    P     = np.empty(group['GroupLen'], dtype=P_dtype)
    PS    = np.zeros(group['GroupLen'], dtype=PS_dtype) # memset(0) in fof.c
    SphP  = np.empty(numPartType[sP.ptNum('gas')], dtype=SphP_dtype)
    StarP = np.empty(numPartType[sP.ptNum('stars')], dtype=StarP_dtype)
    BHP   = np.empty(numPartType[sP.ptNum('bhs')], dtype=BHP_dtype)

    # load and fill
    P_offset = 0

    for ptNum in range(numPartType.size):
        print(' loading ptNum [%d]...' % ptNum)
        
        # general
        for field in ['Pos','Vel','Mass','Potential','ID']:
            if field == 'Mass' and sP.isPartType(ptNum, 'dm'):
                P[P_offset:P_offset+numPartType[ptNum]][field] = dmMass
                continue

            P[P_offset:P_offset+numPartType[ptNum]][field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

        P[P_offset:P_offset+numPartType[ptNum]]['Type'] = ptNum
        P[P_offset:P_offset+numPartType[ptNum]]['SofteningType'] = 1 # L35n2160TNG (DM and Stars) (gas/BHs overwritten later)

        # gas only, and handle SphP
        if sP.isPartType(ptNum, 'gas'):
            for field in ['Density','Sfr','Utherm','Center','Volume']:
                SphP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

            PS[P_offset:P_offset+numPartType[ptNum]]['Center'] = SphP['Center']
            PS[P_offset:P_offset+numPartType[ptNum]]['Utherm'] = SphP['Utherm']

        # stars only, and handle StarP
        if sP.isPartType(ptNum, 'stars'):
            StarP['PID'] = np.arange(P_offset, P_offset + numPartType[ptNum])
            for field in ['BirthTime']:
                StarP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

            P[P_offset:P_offset+numPartType[ptNum]]['AuxDataID'] = np.arange(numPartType[ptNum])

        # BHs only, and handle BHP
        if sP.isPartType(ptNum, 'bhs'):
            BHP['PID'] = np.arange(P_offset, P_offset + numPartType[ptNum])
            for field in ['BH_Mass','BH_Mdot']:
                BHP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

            P[P_offset:P_offset+numPartType[ptNum]]['AuxDataID'] = np.arange(numPartType[ptNum])

        for field in ['SubfindHsml','SubfindDensity','SubfindDMDensity','SubfindVelDisp']:
            PS[P_offset:P_offset+numPartType[ptNum]][field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

        P_offset += numPartType[ptNum]

    # PS
    PS['GrNr'] = GrNr
    PS['Hsml'] = PS['SubfindHsml']
    PS['Density'] = PS['SubfindDensity']
    PS['OldIndex'] = np.arange(group['GroupLen'])

    return P, PS, SphP, StarP, BHP

@jit(nopython=True, nogil=True)
def _updateNodeRecursive(no,sib,last,next_node,tree_nodes):
    """ Helper routine for calcHsml(), see below. """
    pp = 0
    nextsib = 0

    NumPart = next_node.size

    if no >= NumPart:
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last-NumPart]['nextnode'] = no
            else:
                next_node[last] = no

        last = no

        for i in range(8):
            p = tree_nodes[no-NumPart]['suns'][i]

            if p >= 0:
                # check if we have a sibling on the same level
                j = i + 1
                while j < 8:
                    pp = tree_nodes[no-NumPart]['suns'][j]
                    if pp >= 0:
                        break
                    j += 1 # unusual syntax so that j==8 at the end of the loop if we never break

                if j < 8: # yes, we do
                    nextsib = pp
                else:
                    nextsib = sib

                last = _updateNodeRecursive(p,nextsib,last,next_node,tree_nodes)

        tree_nodes[no-NumPart]['sibling'] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last-NumPart]['nextnode'] = no
            else:
                next_node[last] = no

        last = no

    return last # avoid use of global in numba

@jit(nopython=True, nogil=True)
def _updateNodeRecursiveExtra(no,sib,last,next_node,tree_nodes,P,P_Pos,ForceSoftening):
    """ As _updateNodeRecursive(), but also compute additional information for the tree nodes such as masses, softenings. """
    pp = 0
    nextsib = 0

    NumPart = next_node.size

    if no >= NumPart:
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last-NumPart]['nextnode'] = no
            else:
                next_node[last] = no

        last = no

        # initial values
        mass = 0.0
        com = np.zeros( 3, dtype=np.float64 )
        mass_per_type = np.zeros( NSOFTTYPES, dtype=np.float64 )

        maxsofttype = np.uint8(NSOFTTYPES + NSOFTTYPES_HYDRO)
        maxhydrosofttype = NSOFTTYPES
        minhydrosofttype = NSOFTTYPES + NSOFTTYPES_HYDRO - 1

        # loop over each of the 8 daughters
        for i in range(8):
            p = tree_nodes[no-NumPart]['suns'][i]

            if p >= 0:
                # check if we have a sibling on the same level
                j = i + 1
                while j < 8:
                    pp = tree_nodes[no-NumPart]['suns'][j]
                    if pp >= 0:
                        break
                    j += 1 # unusual syntax so that j==8 at the end of the loop if we never break

                if j < 8: # yes, we do
                    nextsib = pp
                else:
                    nextsib = sib

                last = _updateNodeRecursiveExtra(p,nextsib,last,next_node,tree_nodes,P,P_Pos,ForceSoftening)

                if p < NumPart:
                    # individual particle
                    mass   += P[p]['Mass']
                    com[0] += P[p]['Mass'] * P_Pos[p,0]
                    com[1] += P[p]['Mass'] * P_Pos[p,1]
                    com[2] += P[p]['Mass'] * P_Pos[p,2]

                    if ForceSoftening[maxsofttype] < ForceSoftening[P[p]['SofteningType']]:
                        maxsofttype = P[p]['SofteningType']

                    if P[p]['Type'] == 0:
                        mass_per_type[0] += P[p]['Mass']

                        if maxhydrosofttype < P[p]['SofteningType']:
                            maxhydrosofttype = P[p]['SofteningType']
                        if minhydrosofttype > P[p]['SofteningType']:
                            minhydrosofttype = P[p]['SofteningType']
                    else:
                        mass_per_type[P[p]['SofteningType']] += P[p]['Mass']
                else:
                    # internal node
                    ind = p-NumPart

                    mass   += tree_nodes[ind]['mass']
                    com[0] += tree_nodes[ind]['mass'] * tree_nodes[ind]['com'][0]
                    com[1] += tree_nodes[ind]['mass'] * tree_nodes[ind]['com'][1]
                    com[2] += tree_nodes[ind]['mass'] * tree_nodes[ind]['com'][2]

                    if(ForceSoftening[maxsofttype] < ForceSoftening[tree_nodes[ind]['maxsofttype']]):
                        maxsofttype = tree_nodes[ind]['maxsofttype']

                    for k in range(NSOFTTYPES):
                        mass_per_type[k] += tree_nodes[ind]['mass_per_type'][k]

                    if maxhydrosofttype < tree_nodes[ind]['maxhydrosofttype']:
                        maxhydrosofttype = tree_nodes[ind]['maxhydrosofttype']
                    if minhydrosofttype < tree_nodes[ind]['minhydrosofttype']:
                        minhydrosofttype = tree_nodes[ind]['minhydrosofttype']

        # update node properties
        ind = no - NumPart

        if mass > 0.0:
            com /= mass
        else:
            com[0] = tree_nodes[ind]['center'][0]
            com[1] = tree_nodes[ind]['center'][1]
            com[2] = tree_nodes[ind]['center'][2]

        tree_nodes[ind]['com'][0] = com[0]
        tree_nodes[ind]['com'][1] = com[1]
        tree_nodes[ind]['com'][2] = com[2]

        tree_nodes[ind]['mass'] = mass
        tree_nodes[ind]['maxsofttype'] = maxsofttype
        for k in range(NSOFTTYPES):
            tree_nodes[ind]['mass_per_type'][k] = mass_per_type[k]
        tree_nodes[ind]['maxhydrosofttype'] = maxhydrosofttype
        tree_nodes[ind]['minhydrosofttype'] = minhydrosofttype

        tree_nodes[ind]['sibling'] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last-NumPart]['nextnode'] = no
            else:
                next_node[last] = no

        last = no

    return last # avoid use of global in numba

@jit(nopython=True, nogil=True)
def _treeExtent(pos):
    """ Determine extent for non-periodic (local) tree. """
    NumPart = pos.shape[0]

    xyzMin = np.zeros( 3, dtype=np.float64 )
    xyzMax = np.zeros( 3, dtype=np.float64 )

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

    return xyzMin, xyzMax, extent

@jit(nopython=True, nogil=True) #, cache=True)
def _constructTree(pos,boxSizeSim,xyzMin,xyzMax,extent,next_node,tree_nodes,P,P_Pos,ForceSoftening):
    """ Core routine for calcHsml(), see below. """
    subnode = 0
    parent  = -1
    lenHalf = 0.0

    # Nodes_base and Nodes are both pointers to the arrays of NODE structs
    # Nodes_base is allocated with size >NumPart, and entries >=NumPart are "internal nodes"
    #  while entries from 0 to NumPart-1 are leafs (actual particles)
    #  Nodes just points to Nodes_base-NumPart (such that Nodes[no]=Nodes_base[no-NumPart])

    # select first node
    NumPart = pos.shape[0]
    nFree = NumPart

    # create an empty root node
    if boxSizeSim > 0.0:
        # periodic, set center position and extent
        for j in range(3):
            tree_nodes[0]['center'][j] = 0.5 * boxSizeSim
            tree_nodes[0]['length'] = boxSizeSim
    else:
        # non-periodic
        if extent == 0.0:
            # do not have a pre-computed xyzMin, xyzMax, extent, so determine now
            xyzMin, xyzMax, extent = _treeExtent(pos)

        # set center position and extent
        for j in range(3):
            tree_nodes[0]['center'][j] = 0.5 * (xyzMin[j] + xyzMax[j])
        tree_nodes[0]['length'] = extent

    # daughter slots of root node all start empty
    for i in range(8):
        tree_nodes[0]['suns'][i] = -1

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

                if pos[i,0] > tree_nodes[ind]['center'][0]:
                    subnode += 1
                if pos[i,1] > tree_nodes[ind]['center'][1]:
                    subnode += 2
                if pos[i,2] > tree_nodes[ind]['center'][2]:
                    subnode += 4

                # get the next node
                nn = tree_nodes[ind]['suns'][subnode]

                if nn >= 0: # ok, something is in the daughter slot already, need to continue
                    parent = no # note: subnode can still be used in the next step of the walk
                    no = nn
                else:
                    # here we have found an empty slot where we can attach the new particle as a leaf
                    tree_nodes[ind]['suns'][subnode] = i
                    break # done for this particle
            else:
                # we try to insert into a leaf with a single particle - need to generate a new internal
                # node at this point, because every leaf is only allowed to contain one particle
                tree_nodes[parent-NumPart]['suns'][subnode] = nFree
                ind1 = parent-NumPart
                ind2 = nFree-NumPart

                tree_nodes[ind2]['length'] = 0.5 * tree_nodes[ind1]['length']
                lenHalf = 0.25 * tree_nodes[ind1]['length']

                if subnode & 1:
                    tree_nodes[ind2]['center'][0] = tree_nodes[ind1]['center'][0] + lenHalf
                else:
                    tree_nodes[ind2]['center'][0] = tree_nodes[ind1]['center'][0] - lenHalf

                if subnode & 2:
                    tree_nodes[ind2]['center'][1] = tree_nodes[ind1]['center'][1] + lenHalf
                else:
                    tree_nodes[ind2]['center'][1] = tree_nodes[ind1]['center'][1] - lenHalf

                if subnode & 4:
                    tree_nodes[ind2]['center'][2] = tree_nodes[ind1]['center'][2] + lenHalf
                else:
                    tree_nodes[ind2]['center'][2] = tree_nodes[ind1]['center'][2] - lenHalf

                for j in range(8):
                    tree_nodes[ind2]['suns'][j] = -1

                # which subnode
                subnode = 0

                if pos[no,0] > tree_nodes[ind2]['center'][0]:
                    subnode += 1
                if pos[no,1] > tree_nodes[ind2]['center'][1]:
                    subnode += 2
                if pos[no,2] > tree_nodes[ind2]['center'][2]:
                    subnode += 4

                if(tree_nodes[ind2]['length'] < 1e-4):
                    # may have particles at identical locations, in which case randomize the subnode 
                    # index to put the particle into a different leaf (happens well below the 
                    # gravitational softening scale)
                    subnode = np.int(np.random.rand())
                    subnode = max(subnode,7)

                tree_nodes[ind2]['suns'][subnode] = no

                no = nFree # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                if numNodes >= tree_nodes.size:
                    # exceeding tree allocated size, need to increase and redo
                    return -1

    # now compute the (sibling,nextnode,next_node) recursively
    last = np.int32(-1)

    last = _updateNodeRecursiveExtra(NumPart,-1,last,next_node,tree_nodes,P,P_Pos,ForceSoftening)

    if last >= NumPart:
        tree_nodes[last-NumPart]['nextnode'] = -1
    else:
        next_node[last] = -1

    return numNodes

@jit(nopython=True, nogil=True)
def _treeSearchIndices(xyz,h,boxSizeSim,pos,next_node,tree_nodes):
    """ Helper routine for calcParticleIndices(), see below. """
    boxHalf = 0.5 * boxSizeSim

    h2 = h * h
    hinv = 1.0 / h

    numNgbInH = 0

    # allocate, unfortunately unclear how safe we have to be here
    NumPart = next_node.size
    inds = np.empty( NumPart, dtype=np.int64 )
    dists2 = np.empty( NumPart, dtype=np.float64 )

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
            dists2[numNgbInH] = r2
            numNgbInH += 1

        else:
            # internal node
            ind = no - NumPart
            no = tree_nodes[ind]['sibling'] # in case the node can be discarded

            if _NEAREST( tree_nodes[ind]['center'][0] - xyz[0], boxHalf, boxSizeSim ) + 0.5 * tree_nodes[ind]['length'] < -h:
                continue
            if _NEAREST( tree_nodes[ind]['center'][0] - xyz[0], boxHalf, boxSizeSim ) - 0.5 * tree_nodes[ind]['length'] > h:
                continue

            if _NEAREST( tree_nodes[ind]['center'][1] - xyz[1], boxHalf, boxSizeSim ) + 0.5 * tree_nodes[ind]['length'] < -h:
                continue
            if _NEAREST( tree_nodes[ind]['center'][1] - xyz[1], boxHalf, boxSizeSim ) - 0.5 * tree_nodes[ind]['length'] > h:
                continue

            if _NEAREST( tree_nodes[ind]['center'][2] - xyz[2], boxHalf, boxSizeSim ) + 0.5 * tree_nodes[ind]['length'] < -h:
                continue
            if _NEAREST( tree_nodes[ind]['center'][2] - xyz[2], boxHalf, boxSizeSim ) - 0.5 * tree_nodes[ind]['length'] > h:
                continue

            no = tree_nodes[ind]['nextnode'] # we need to open the node

    if numNgbInH > 0:
        inds = inds[0:numNgbInH]
        dists2 = dists2[0:numNgbInH]
        return numNgbInH, inds, dists2

    return 0, inds, dists2

@jit(nopython=True, nogil=True, cache=True)
def treeSearchIndicesIterate(xyz,h_guess,nNGB,boxSizeSim,pos,next_node,tree_nodes):
    """ Helper routine for subfind(), see below. 
    Note: no nNGBDev, instead we terminate if we ever find >=nNGB, and the return is sorted by distance. """
    if h_guess == 0.0:
        h_guess = 1.0

    iter_num = 0

    while 1:
        iter_num += 1

        assert iter_num < 1000 # Convergence failure, too many iterations.

        numNgbInH, inds, dists_sq = _treeSearchIndices(xyz,h_guess,boxSizeSim,pos,next_node,tree_nodes)

        # enough
        if numNgbInH >= nNGB:
            break

        h_guess *= 1.26

    # sort
    dists = np.sqrt(dists_sq)
    sort_inds = np.argsort(dists)
    dists = dists[sort_inds]
    inds = inds[sort_inds]

    return inds, dists

node_dtype = np.dtype([
    ('length', np.float64),
    ('center', np.float64, 3),
    ('suns', np.int32, 8),  # pointers to daughter nodes
    ('sibling', np.int32),  # next node in the walk, in case the current node can be used
    ('nextnode', np.int32), # next node in the walk, in case the current node needs to be opened
    ('com', np.float64, 3), # center of mass
    ('mass', np.float64),   # mass of node
    ('maxsofttype', np.uint8),
    ('maxhydrosofttype', np.uint8),
    ('minhydrosofttype', np.uint8),
    ('mass_per_type', np.float64, NSOFTTYPES)
])

@jit(nopython=True)
def buildFullTree(pos, boxSizeSim, xyzMin, xyzMax, extent, P, P_Pos, ForceSoftening):
    """ As above, but minimal and JITed. """
    NumPart = pos.shape[0]
    NextNode = np.zeros( NumPart, dtype=np.int32 )

    # tree allocation and construction (iterate in case we need to re-allocate for larger number of nodes)
    for num_iter in range(10):

        # allocate
        MaxNodes = np.int( (num_iter+1.1)*NumPart ) + 1
        if MaxNodes < 100: MaxNodes = 100

        TreeNodes = np.zeros( MaxNodes, dtype=node_dtype )

        # construct: call JIT compiled kernel
        numNodes = _constructTree(pos,boxSizeSim,xyzMin,xyzMax,extent,NextNode,TreeNodes,P,P_Pos,ForceSoftening)

        if numNodes > 0:
            break

    return NextNode, TreeNodes

@jit(nopython=True, nogil=True) #, cache=True)
def subfind_treeevaluate_potential(target, P_Pos, P, ForceSoftening, next_node, tree_nodes, boxHalf, boxSizeSim):
    pos = P_Pos[target]
    h_i = ForceSoftening[P[target]['SofteningType']]

    pot = 0
    indi_flag1 = -1
    indi_flag2 = 0

    # start search
    NumPart = next_node.size
    no = NumPart # note: NumPart here is len (local), as opposed to LocMaxPart in arepo, because our local tree construction does 
                 # not place the root node at LocMaxPart (i.e. aware of P_Pos size) but rather at len (i.e. only aware of loc_pos size)

    while no >= 0:
        if no < NumPart:
            # single particle
            assert next_node[no] != no # Going into infinite loop.

            p = no
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST( P_Pos[p,0] - pos[0], boxHalf, boxSizeSim )
            dy = _NEAREST( P_Pos[p,1] - pos[1], boxHalf, boxSizeSim )
            dz = _NEAREST( P_Pos[p,2] - pos[2], boxHalf, boxSizeSim )

            r2 = dx*dx + dy*dy + dz*dz

            mass = P[p]['Mass']

            h_j = ForceSoftening[P[p]['SofteningType']]

            if h_j > h_i:
                hmax = h_j
            else:
                hmax = h_i
        else:
            # internal node
            ind = no - NumPart
            mass = tree_nodes[ind]['mass']

            dx = _NEAREST( tree_nodes[ind]['com'][0] - pos[0], boxHalf, boxSizeSim )
            dy = _NEAREST( tree_nodes[ind]['com'][0] - pos[0], boxHalf, boxSizeSim )
            dz = _NEAREST( tree_nodes[ind]['com'][0] - pos[0], boxHalf, boxSizeSim )

            r2 = dx*dx + dy*dy + dz*dz

            # check Barnes-hut opening criterion
            if tree_nodes[ind]['length']**2 > r2 * ErrTolThetaSubfind**2:
                # open the node
                if mass:
                    no = tree_nodes[ind]['nextnode']
                    continue

            h_j = ForceSoftening[tree_nodes[ind]['maxsofttype']]

            if h_j > h_i:
                # multiple hydro softenings in this node? compare to maximum
                if tree_nodes[ind]['maxhydrosofttype'] != tree_nodes[ind]['minhydrosofttype']:
                    if tree_nodes[ind]['mass_per_type'][0] > 0:
                        if r2 < ForceSoftening[tree_nodes[ind]['maxhydrosofttype']]**2:
                            # open the node
                            no = tree_nodes[ind]['nextnode']
                            continue

                indi_flag1 = 0
                indi_flag2 = NSOFTTYPES
                hmax = h_j
            else:
                hmax = h_i

            no = tree_nodes[ind]['sibling'] # node can be used

        # proceed (use node)
        ind = no - NumPart
        r = np.sqrt(r2)

        for ptype in range(indi_flag1, indi_flag2):
            if ptype >= 0:
                mass = tree_nodes[ind]['mass_per_type'][ptype]
                if ptype == 0:
                    h_j = ForceSoftening[tree_nodes[ind]['maxhydrosofttype']]
                else:
                    h_j = ForceSoftening[ptype]

                if h_j > h_i:
                    hmax = h_j
                else:
                    hmax = h_i

            if mass:
                if r >= hmax:
                    pot -= mass / r
                else:
                    h_inv = 1.0 / hmax
                    u = r * h_inv

                    if u < 0.5:
                        wp = -2.8 + u * u * (5.333333333333 + u * u * (6.4 * u - 9.6))
                    else:
                        wp = -3.2 + 0.066666666667 / u + u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)))

                    pot += mass * h_inv * wp

    return pot

@jit(nopython=True, nogil=True) #, cache=True)
def subfind_unbind(P_Pos, P, PS, ud, len, vel_to_phys, H_of_a, G, atime, boxsize, SofteningTable, ForceSoftening, xyzMin, xyzMax, extent):
    """ Unbinding. """
    max_iter = 1000
    weakly_bound_limit = 0
    len_non_gas = 0
    minpot = 0
    boxhalf = boxsize * 0.5
    unbound = 0

    iter_num = 0
    phaseflag = 0 # this means we will recompute the potential for all particles

    bnd_energy = np.zeros( len, dtype=np.float64 )
    v  = np.zeros( 3, dtype=np.float64 )
    s  = np.zeros( 3, dtype=np.float64 )
    dv = np.zeros( 3, dtype=np.float64 )
    dx = np.zeros( 3, dtype=np.float64 )

    while 1:
        iter_num += 1
        #print(' subfind_unbind(): iter =',iter_num,' length = ',len,' unbound = ',unbound,' phase = ',phaseflag)

        # build local tree, including only particles still inside the candidate
        loc_pos = np.zeros( (len,3), dtype=np.float64 )
        for i in range(len):
            loc_pos[i,:] = P_Pos[ud[i]['index']]

        NextNode, TreeNodes = buildFullTree(loc_pos, 0.0, xyzMin, xyzMax, extent, P, P_Pos, ForceSoftening)

        # compute the potential
        if phaseflag == 0:
            # redo for all particles
            minindex = -1
            minpot = 1.0e30

            # find particle with the minimum potential
            for i in range(len):
                p = ud[i]['index']
                pot = subfind_treeevaluate_potential(p, P_Pos, P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize)

                PS[p]['Potential'] = G / atime * pot
                #print(' [%2d] p = %3d pot = %g (%g)' % (i,p,pot,PS[p]['Potential']))

                if PS[p]['Potential'] < minpot or minindex == -1:
                    # new minimum potential found
                    minpot = PS[p]['Potential']
                    minindex = p

            # position of minimum potential (CELL_CENTER_GRAVITY)
            pos = P_Pos[minindex]
        else:
            # only repeat for those particles close to the unbinding threshold
            for i in range(len):
                p = ud[i]['index']

                if PS[p]['BindingEnergy'] >= weakly_bound_limit:
                    pot = subfind_treeevaluate_potential(p, P_Pos, P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize)
                    PS[p]['Potential'] *= G / atime

        # calculate the bulk velocity and center of mass
        v *= 0
        s *= 0
        TotMass = 0

        for i in range(len):
            p = ud[i]['index']

            for j in range(3):
                ddxx = _NEAREST( P_Pos[p,j] - pos[j], boxhalf, boxsize )
                s[j] += P[p]['Mass'] * ddxx
                v[j] += P[p]['Mass'] * P[p]['Vel'][j]

            TotMass += P[p]['Mass']

        for j in range(3):
            v[j] /= TotMass
            s[j] /= TotMass # center of mass
            s[j] += pos[j]

            while(s[j] < 0): # PERIODIC
                s[j] += boxsize
            while(s[j] >= boxsize):
                s[j] -= boxsize

        # compute binding energy for all particles
        for i in range(len):
            p = ud[i]['index']

            for j in range(3):
                dv[j] = vel_to_phys * (P[p]['Vel'][j] - v[j])
                dx[j] = atime * _NEAREST( P_Pos[p,j] - s[j], boxhalf, boxsize )
                dv[j] += H_of_a * dx[j] # Hubble expansion, per coordinate

            PS[p]['BindingEnergy'] = PS[p]['Potential'] + 0.5 * (dv[0]*dv[0] + dv[1]*dv[1] + dv[2]*dv[2])
            PS[p]['BindingEnergy'] += G / atime * P[p]['Mass'] / SofteningTable[P[p]['SofteningType']]

            if P[p]['Type'] == 0:
                PS[p]['BindingEnergy'] += PS[p]['Utherm']

            bnd_energy[i] = PS[p]['BindingEnergy']

        # sort by binding energy, largest first
        bnd_energy_sorted = sorted(bnd_energy, reverse=True) # reverse=True ? double check

        quarter_ind = np.int(np.floor(0.25*len))
        energy_limit = bnd_energy_sorted[quarter_ind]
        unbound = 0

        for i in range(len-1):
            if bnd_energy_sorted[i] > 0:
                unbound += 1
            else:
                unbound -= 1

            if unbound <= 0:
                break

        weakly_bound_limit = bnd_energy_sorted[i]

        # now omit unbound particles, but at most 1/4 of the original size (not really)
        unbound = 0
        len_non_gas = 0

        for i in range(len):
            p = ud[i]['index']

            if PS[p]['BindingEnergy'] > 0 and PS[p]['BindingEnergy'] > energy_limit:
                unbound += 1
                ud[i] = ud[len-1]
                i -= 1
                len -= 1
            elif P[p]['Type'] != 0:
                len_non_gas += 1

        # already too small?
        if len < DesLinkNgb:
            break

        # alternate full vs. partial potential calculations
        if phaseflag == 0:
            if unbound > 0:
                phaseflag = 1
        else:
            if unbound == 0:
                phaseflag = 0
                unbound = 1

        if iter_num > max_iter:
            raise Exception('Not good.')

        # convergence, we are done
        if unbound <= 0:
            break

    return len, len_non_gas

@jit(nopython=True, nogil=True)
def subfind(P, PS, SphP, StarP, BHP, atime, H_of_a, G, boxsize, SofteningTable, ForceSoftening):
    """ Run serial subfind. (Offs = 0). """

    # estimate the maximum number of substructures we need to store (conservative upper limit)
    MaxNsubgroups = P.size / DesLinkNgb
    N = P.size

    # generate P_Pos, pure ndarray and handle CELL_CENTER_GRAVITY
    P_Pos = np.zeros( (N,3), dtype=np.float64 )

    for i in range(N):
        if P[i]['Type'] == 0:
            P_Pos[i,:] = PS[i]['Center']
        else:
            P_Pos[i,:] = P[i]['Pos']

    # allocate
    candidates = np.zeros( N, dtype=cand_dtype )
    Head = np.zeros( N, dtype=np.int32 ) - 1
    Next = np.zeros( N, dtype=np.int32 ) - 1
    Tail = np.zeros( N, dtype=np.int32 ) - 1
    Len  = np.zeros( N, dtype=np.int32 )
    ud   = np.zeros( N, dtype=ud_dtype )

    for i in range(N):
        ud[i]['index'] = i

    # order particles (P, PS) in the order of decreasing density
    sort_inds = np.argsort(PS['Density'])[::-1] # descending
    sort_inds_inv = np.zeros( sort_inds.size, dtype=np.int32 )
    sort_inds_inv[sort_inds] = np.arange(sort_inds.size)

    # note: temporarily break the association with SphP[] and other arrays!
    PS = PS[sort_inds]
    P  = P[sort_inds]
    P_Pos = P_Pos[sort_inds]

    for i in range(StarP.size):
        StarP[i]['PID'] = sort_inds_inv[ StarP[i]['PID'] ]
    for i in range(BHP.size):
        BHP[i]['PID'] = sort_inds_inv[ BHP[i]['PID'] ]

    # build tree for all particles of this group
    BoxSizeSim = 0.0 # tree searches are non-periodic, and we use local (non-box-global) extents
    xyzMin, xyzMax, extent = _treeExtent(P_Pos)
    NextNode, TreeNodes = buildFullTree(P_Pos, BoxSizeSim, xyzMin, xyzMax, extent, P, P_Pos, ForceSoftening)

    # process every particle
    head = 0
    count_cand = 0
    listofdifferent = np.zeros( 2, dtype=np.int32 )

    print('Tree built and arrays sorted, beginning neighbor search...')

    for i in range(N):
        # find neighbors, note: returned neighbors are already sorted by distance (ascending)
        if i % np.int(N/10) == 0: print(' ',np.round(float(i)/N*100),'%')
        pos = P_Pos[i]
        h_guess = PS[i]['Hsml']

        inds, dists = treeSearchIndicesIterate(pos,h_guess,DesLinkNgb,BoxSizeSim,P_Pos,NextNode,TreeNodes)

        # process neighbors
        ndiff = 0
        ngbs = 0

        for j in range(DesLinkNgb):
            ngb_index = inds[j]

            if ngbs >= 2:
                break

            if ngb_index == i:
                continue # to exclude the particle itself

            # we only look at neighbors that are denser
            if PS[ngb_index]['Density'] > PS[i]['Density']:
                ngbs += 1

                if Head[ngb_index] >= 0: # neighbor is attached to a group
                    if ndiff == 1:
                        if listofdifferent[0] == Head[ngb_index]:
                            continue

                    # a new group has been found
                    listofdifferent[ndiff] = Head[ngb_index]
                    ndiff += 1
                else:
                    raise Exception('this may not occur')

        # treat the different possible cases
        if ndiff == 0:
            # this appears to be a lonely maximum -> new group
            head = i
            Head[i] = i
            Tail[i] = i
            Len[i] = 1
            Next[i] = -1

        elif ndiff == 1:
            # the particle is attached to exactly one group
            head = listofdifferent[0]
            Head[i] = head
            Next[Tail[head]] = i
            Tail[head] = i
            Len[head] += 1
            Next[i] = -1

        elif ndiff == 2:
            # the particle mergers two groups together
            head = listofdifferent[0]
            head_attach = listofdifferent[1]

            if Len[head_attach] > Len[head] or (Len[head_attach] == Len[head] and head_attach < head):
                # other group is longer, swap them. for equal length, take the larger head value
                head = listofdifferent[1]
                head_attach = listofdifferent[0]

            # only in case the attached group is long enough do we register it as a subhalo candidate
            if Len[head_attach] >= DesLinkNgb:
                candidates[count_cand]['len'] = Len[head_attach]
                candidates[count_cand]['head'] = Head[head_attach]
                count_cand += 1

            # now join the two groups
            Next[Tail[head]] = head_attach
            Tail[head] = Tail[head_attach]
            Len[head] += Len[head_attach]

            ss = head_attach
            while 1:
                Head[ss] = head
                ss = Next[ss]
                if ss < 0:
                    break

            # finally, attach the particle
            Head[i] = head
            Next[Tail[head]] = i
            Tail[head] = i
            Len[head] += 1
            Next[i] = -1

        else:
            raise Exception('Cannot occur.')

    # add the full thing as a subhalo candidate
    prev = -1

    for i in range(N):
        if Head[i] != i:
            continue
        if Next[Tail[i]] == -1:
            if prev < 0:
                head = i
            if prev >= 0:
                Next[prev] = i

            prev = Tail[i]

    candidates[count_cand]['len'] = N
    candidates[count_cand]['head'] = head
    count_cand += 1

    print('Searches done, ended with [',count_cand,'] candidates, now unbinding...')

    # go through them once and assign the rank
    p = head
    rank = 0

    for i in range(N):
        Len[p] = rank
        rank += 1
        p = Next[p]

    # for each candidate, we now pull out the rank of its head
    for i in range(count_cand):
        candidates[i]['rank'] = Len[candidates[i]['head']]

    for i in range(N):
        Tail[i] = -1

    # do gravitational unbinding on each candidate
    vel_to_phys = 1.0 / atime
    nsubs = 0

    for i in range(count_cand):
        p = candidates[i]['head']
        len = 0

        # create local index list for members of this candidate
        for _ in range(candidates[i]['len']):
            if Tail[p] < 0:
                assert p >= 0
                ud[len]['index'] = p
                len += 1
            p = Next[p]

        print(' cand: [',i,'] of [',count_cand,'] with length ',len)
        #if i == count_cand-1:
        #    print(' SKIP JUST FOR NOW')
        #    continue

        if len >= DesLinkNgb:
            len, len_non_gas = subfind_unbind(P_Pos, P, PS, ud, len, vel_to_phys, H_of_a, G, atime, boxsize, 
                SofteningTable, ForceSoftening, xyzMin, xyzMax, extent)

        if len >= DesLinkNgb:
            # we found a substructure
            for j in range(len):
                Tail[ud[j]['index']] = nsubs # use this to flag the substructures

            candidates[i]['nsub'] = nsubs
            candidates[i]['bound_length'] = len
            nsubs += 1
        else:
            candidates[i]['nsub'] = -1
            candidates[i]['bound_length'] = 0

        p = Next[p]

    return count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP

def subfind_properties_1(candidates):
    """ Determine the parent subhalo for each candidate. """

    # sort candidates on (bound_length,rank)
    candidates['bound_length'] *= -1
    sort_inds = np.argsort(candidates, order=['bound_length','rank']) # bound_length descending, rank ascending
    candidates['bound_length'] *= -1

    candidates = candidates[sort_inds]

    # reduce to actual (non-blank) candidates. note: not done in subfind
    count_cand = np.count_nonzero(candidates['bound_length'])
    candidates = candidates[0:count_cand]

    # sort with comparator function:
    # https://stackoverflow.com/questions/29200353/how-to-speed-quicksort-with-numba

    for i in range(count_cand):
        candidates[i]['subnr'] = i
        candidates[i]['parent'] = 0

    candidates['len'] *= -1
    sort_inds = np.argsort(candidates, order=['rank','len']) # rank ascending, len descending
    candidates['len'] *= -1

    candidates = candidates[sort_inds]

    for k in range(count_cand):
        for j in range(k+1, count_cand):
            if candidates[j]['rank'] > candidates[k]['rank'] + candidates[k]['len']:
                break

            if candidates[k]['rank'] + candidates[k]['len'] >= candidates[j]['rank'] + candidates[j]['len']:
                if candidates[k]['bound_length'] >= DesLinkNgb:
                    candidates[j]['parent'] = candidates[k]['subnr']
            else:
                raise Exception('Not good.')

    sort_inds = np.argsort(candidates['subnr'])
    candidates = candidates[sort_inds]

    return candidates

#@jit(nopython=True, nogil=True)
def subfind_properties_2(candidates, Tail, Next, P, PS, SphP, StarP, BHP, atime, H_of_a, G, boxsize):
    """ Determine the properties of each subhalo. """
    vel_to_phys = 1.0 / atime
    #Group['Nsubs'] = nsubs
    #Group['Pos'] = Group['CM']

    #allocate Group, SubGroup[]
    subnr = 0
    totlen = 0

    for k in range(candidates.size):
        len = candidates[k]['bound_length']
        totlen += len

        p = candidates[k]['head']
        ud_index = np.zeros( len, dtype=np.int32 )
        count = 0

        for i in range(candidates[k]['len']):
            if Tail[p] == candidates[k]['nsub']:
                ud_index[count] = p
                count += 1
            p = Next[p]

        if count != len:
            raise Exception('Mismatch.')

        # subfind_determine_sub_halo_properties(); start
        # temporary allocations
        len_type = np.zeros( NTYPES, dtype=np.int32 )
        #len_type_loc = np.zeros( NTYPES, dtype=np.int32 )
        pos = np.zeros( 3, dtype=np.float64 )

        sfr = 0.0
        bh_Mass = 0.0
        bh_Mdot = 0.0
        windMass = 0.0

        minindex = -1
        minpot = 1.0e30

        for i in range(len):
            p = ud_index[i]
            auxid = P[p]['AuxDataID']

            if PS[p]['Potential'] < minpot or minindex == -1:
                minpot = PS[p]['Potential']
                minindex = p

            len_type[P[p]['Type']] += 1

            if P[p]['Type'] == 0:
                sfr += SphP[PS[p]['OldIndex']]['Sfr']

            if P[p]['Type'] == 5:
                bh_Mass += BHP[auxid]['BH_Mass']
                bh_Mdot += BHP[auxid]['BH_Mdot']

            if P[p]['Type'] == 4 and StarP[auxid]['BirthTime'] < 0:
                windMass += P[p]['Mass']

        #for i in range(NTYPES):
        #    len_type_loc[i] = len_type[i]

        assert minindex != -1

        # pos[] now holds the position of the minimum potential, we'll take it as the center
        for j in range(3):
            if P[minindex]['Type'] == 0:
                pos[j] = SphP[ PS[minindex]['OldIndex'] ]['Center'][j]
            else:
                pos[j] = P[minindex]['Pos'][j]
        #pos = P_Pos[minindex]

        # determine the particle ID with the smallest binding energy
        minindex = -1
        minpot = 1.0e30

        for i in range(len):
            p = ud_index[i]
            if PS[p]['BindingEnergy'] < minpot or minindex == -1:
                minpot = PS[p]['BindingEnergy']
                minindex = p

        assert minindex != -1

        mostboundid = P[minindex]['ID']

        # let's get bulk velocity and the center-of-mass, here we still take all particles

        print('subnr ', subnr, ' len', len, ' pos:',pos,' mostboundid',mostboundid,' lentype',len_type)
        # ...
        # subfind_determine_sub_halo_properties(); end

        #SubGroup[subnr]['SubParent'] = candidates[k]['parent']
        #SubGroup[subnr]['SubNr'] = subnr
        #SubGroup[subnr]['GrNr'] = Group['GrNr']

        if subnr == 0:
            pass
            #Group['Pos'] = SubGroup[subnr]['Pos']

        # let us now assign the subgroup number
        #for i in range(len):
        #    PS[ud_index[i]]['SubNr'] = subnr
        subnr += 1

    import pdb; pdb.set_trace()
    


def set_softenings(P, SphP, sP):
    """ Generate SofteningTable following grav_softening.c, and see P[].SofteningType values (based on sizes/masses). """
    def get_softeningtype_for_hydro_cell(volume):
        radius = np.power(volume * 3.0 / (4.0 * np.pi), 1.0 / 3)
        soft = GasSoftFactor * radius

        types = np.zeros( volume.size, dtype=np.uint8 )

        w = np.where(soft <= ForceSoftening[NSOFTTYPES])
        types[w] = NSOFTTYPES

        w = np.where(soft > ForceSoftening[NSOFTTYPES])
        k = 0.5 * np.log(soft[w] / ForceSoftening[NSOFTTYPES]) / np.log(AdaptiveHydroSofteningSpacing)

        w_above = np.where(k >= NSOFTTYPES_HYDRO)
        k[w_above] = NSOFTTYPES_HYDRO - 1

        types[w] = NSOFTTYPES + k

        return types

    def get_softening_type_from_mass(mass):
        # get_desired_softening_from_mass():
        eps = np.zeros( mass.size, dtype=np.float64 )
        w = np.where(mass <= sP.dmParticleMass)
        eps[w] = 2.8 * SofteningComoving[1]
        w = np.where(mass > sP.dmParticleMass)
        eps[w] = 2.8 * SofteningComoving[1] * np.power(mass[w] / sP.dmParticleMass, 1.0/3)

        types = np.zeros( mass.size, dtype=np.uint8 )
        min_dln = np.zeros( mass.size, dtype=np.float64 )
        min_dln.fill( np.finfo(np.float64).max )

        for i in range(1,NSOFTTYPES): # MULTIPLE_NODE_SOFTENING & ADAPTIVE_HYDRO_SOFTENING
            if ForceSoftening[i] > 0:
                dln = np.abs( np.log(eps) - np.log(ForceSoftening[i]) )

                w = np.where(dln < min_dln)
                types[w] = i
        return types

    SofteningTable = np.zeros( NSOFTTYPES + NSOFTTYPES_HYDRO, dtype=np.float64 ) # current (comoving)

    for i in range(NSOFTTYPES):
        if SofteningComoving[i] * sP.scalefac > SofteningMaxPhys[i]:
            SofteningTable[i] = SofteningMaxPhys[i] / sP.scalefac
        else:
            SofteningTable[i] = SofteningComoving[i]

    for i in range(NSOFTTYPES_HYDRO):
        SofteningTable[i + NSOFTTYPES] = MinimumComovingHydroSoftening * np.power(AdaptiveHydroSofteningSpacing, i)

    ForceSoftening = np.zeros(SofteningTable.size + 1, dtype=SofteningTable.dtype)
    ForceSoftening[:-1] = 2.8 * SofteningTable
    ForceSoftening[NSOFTTYPES + NSOFTTYPES_HYDRO] = 0

    # handle P[].SofteningType (snapshot load only)
    if 1:
        print('Setting P[].SofteningType for snapshot load.')
        # INDIVIDUAL_GRAVITY_SOFTENING=32 for L35n2160TNG
        w = np.where(P['Type'] == sP.ptNum('bhs'))[0]
        assert len(w) == w.max() - w.min() + 1
        # note: P[w]['field'] = array assignment seems to silently fail... only works for min:max index range
        P[w.min() : w.max()+1]['SofteningType'] = get_softening_type_from_mass(P[w]['Mass'])

        # ADAPTIVE_HYDRO_SOFTENING
        w = np.where(P['Type'] == sP.ptNum('gas'))[0]
        assert len(w) == w.max() - w.min() + 1
        P[w.min() : w.max()+1]['SofteningType'] = get_softeningtype_for_hydro_cell(SphP['Volume'])

    return SofteningTable, ForceSoftening, P

def run_test():
    """ Testing. """
    from util.simParams import simParams
    sP = simParams(res=256,run='tng',redshift=0.0,variant='0000')

    # load
    P, PS, SphP, StarP, BHP = load_snapshot_data(sP, GrNr=100) # testing

    SofteningTable, ForceSoftening, P = set_softenings(P, SphP, sP)

    # execute
    count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP = subfind(P, PS, SphP, StarP, BHP, 
        sP.scalefac, sP.units.H_z, sP.units.G, sP.boxSize, SofteningTable, ForceSoftening)

    print("Number of substructures: %d (before unbinding: %d)" % (nsubs,count_cand))

    candidates = subfind_properties_1(candidates)
    test = subfind_properties_2(candidates, Tail, Next, P, PS, SphP, StarP, BHP, sP.scalefac, sP.units.H_z, sP.units.G, sP.boxSize)

    import pdb; pdb.set_trace()

def run_test2():
    """ Testing. """
    from util.simParams import simParams
    # snap=5 is fof0 SKIPPED (with new fof_0 dump), snap=4 is normal (with corrupt fof_0 dump)
    #sP = simParams(res=512,run='tng',snap=5,variant='5006')

    # L35n2160TNG started skipping fof0 subfind at snapshot 69 and onwards
    sP = simParams(res=2160,run='tng',snap=69)

    P, PS, SphP, StarP, BHP = load_custom_dump(sP, GrNr=0)
    import pdb; pdb.set_trace()