"""
Implementation of (serial) subfind algorithm.
"""

import glob
import multiprocessing as mp
import struct
import time
from os.path import isfile

import h5py
import numpy as np
from numba import jit, prange

from ..util.match import match
from ..util.simParams import simParams
from ..util.sphMap import _NEAREST


# DOUBLEPRECISION == 1
MyFloat = np.float64
MySingle = np.float64
MyDouble = np.float64
MyIDType = np.uint64  # LONGIDS
integertime = np.int32  # ~ENLARGE_DYNAMIC_RANGE_IN_TIME

GFM_N_CHEM_ELEMENTS = 10
GFM_N_CHEM_TAGS = 6
GRAVCOSTLEVELS = 6
MAXSCALARS = GFM_N_CHEM_ELEMENTS + 1 + 1  # SECOND +1: BUG FROM GFM_DUST

NTYPES = 6
NSOFTTYPES_HYDRO = 64  # ADAPTIVE_HYDRO_SOFTENING
AdaptiveHydroSofteningSpacing = 1.2
GasSoftFactor = 2.5
ErrTolThetaSubfind = 0.7
DesLinkNgb = 20

SUBFIND_GAL_RADIUS_FAC = 2.0
GFM_STELLAR_PHOTOMETRICS_BANDS = 8  # number of bands
GFM_STELLAR_PHOTOMETRICS_DIRECTIONS = 100  # number of directions for surface brightness calculation
GFM_STELLAR_PHOTOMETRICS_RADII = 100  # number of radii (rings) for surface brightness calculation (must be >=3)
GFM_STELLAR_PHOTOMETRICS_K_LIMIT = 20.7  # limiting surface brightness determining 'detectable radius'
GFM_MIN_METAL = -20.0  # minimum metallicity
MAX_FLOAT_NUMBER = 1e37

# L35n2160TNG: (also change P.BirthPos/Vel!) (can shorten *_mem.dtype for phase1)
NSOFTTYPES = 6
MinimumComovingHydroSoftening = 0.05
SofteningComoving = [0.390, 0.390, 0.390, 0.670, 1.15, 2.0]
SofteningMaxPhys = [0.195, 0.195, 0.390, 0.670, 1.15, 2.0]

# L25n128_0000 TESTING:
# NSOFTTYPES = 4
# MinimumComovingHydroSoftening = 0.5
# SofteningComoving = [4.0, 4.0, 4.0, 6.84]
# SofteningMaxPhys  = [2.0, 2.0, 4.0, 6.84]

# L25n256_0000 TESTING:
# NSOFTTYPES = 5
# MinimumComovingHydroSoftening = 0.25
# SofteningComoving = [2.0, 2.0, 2.0, 3.42, 5.84]
# SofteningMaxPhys  = [1.0, 1.0, 2.0, 3.42, 5.84]

# L25n512_0000 TESTING:
# NSOFTTYPES = 6
# MinimumComovingHydroSoftening = 0.125
# SofteningComoving = [1.0, 1.0, 1.0, 1.71, 2.92, 5.0]
# SofteningMaxPhys  = [0.5, 0.5, 1.0, 1.71, 2.92, 5.0]

# define data types
grad_data_dtype = np.dtype(
    [
        ("dhro", MySingle, 3),
        ("dvel", MySingle, (3, 3)),
        ("dpress", MySingle, 3),
        ("dB", MySingle, (3, 3)),  # MHD
        ("dscalars", MySingle, (MAXSCALARS, 3)),  # MAXSCALARS
    ]
)

SphP_dtype = np.dtype(
    [
        # conserved variables
        ("Energy", MyFloat),
        ("Momentum", MyFloat, 3),
        ("Volume", MyFloat),
        ("OldMass", MyFloat),
        # primitive variables
        ("Density", MyFloat),
        ("Pressure", MyFloat),
        ("Utherm", MySingle),
        ("FullGravAccel", MySingle, 3),  # HIERARCHICAL_GRAVITY
        ("MaxMach", MyFloat),  # BUG: defined(TRACER_MC_MACHMAX)
        # variables for mesh
        ("Center", MyDouble, 3),
        ("VelVertex", MySingle, 3),
        ("MaxDelaunayRadius", MySingle),
        ("Hsml", MySingle),
        ("SurfaceArea", MySingle),
        ("MaxFaceAngle", MySingle),  # REGULARIZE_MESH_FACE_ANGLE
        ("ActiveArea", MySingle),
        ("Machnumber", MySingle),  # SHOCK_FINDER_BEFORE_OUTPUT
        ("EnergyDissipation", MySingle),  # SHOCK_FINDER_BEFORE_OUTPUT
        ("CurlVel", MySingle),  # REGULARIZE_MESH_CM_DRIFT_USE_SOUNDSPEED
        ("CurrentMaxTiStep", MySingle),  # TREE_BASED_TIMESTEPS
        ("Csnd", MySingle),  # TREE_BASED_TIMESTEPS
        # MHD
        ("B", MyFloat, 3),
        ("BConserved", MyFloat, 3),
        ("DivB", MyFloat),
        # GFM_STELLAR_EVOLUTION
        ("Metallicity", MyFloat),
        ("MassMetallicity", MyFloat),
        ("MetalsFraction", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("MassMetals", MyFloat, GFM_N_CHEM_ELEMENTS),
        # GFM_CHEMTAGS
        ("MassMetalsChemTags", MyFloat, GFM_N_CHEM_TAGS),
        ("MassMetalsChemTagsFraction", MyFloat, GFM_N_CHEM_TAGS),
        ("MinimumEdgeDistance", MySingle),  # REFINEMENT_SPLIT_CELLS
        ("Ne", MyFloat),  # COOLING
        ("Sfr", MySingle),  # USE_SFR
        ("HostHaloMass", MySingle),  # GFM_WINDS_VARIABLE
        # ('DMVelDisp', MySingle), # GFM_WINDS_VARIABLE (UNION)
        ("AGNBolIntensity", MySingle),  # GFM_AGN_RADIATION
        ("Injected_BH_Energy", MySingle),  # BH_THERMALFEEDBACK
        ("Injected_BH_Wind_Momentum", MyFloat, 3),  # BH_ADIOS_WIND
        ("Grad", grad_data_dtype),
        ("first_connection", np.int32),  # VORONOI_DYNAMIC_UPDATE
        ("last_connection", np.int32),  # VORONOI_DYNAMIC_UPDATE
        ("SepVector", MySingle, 3),  # REFINEMENT_SPLIT_CELLS
        ("TimeLastPrimUpdate", np.float64),
    ]
)

SphP_dtype_mem = np.dtype(
    [
        ("Volume", MyFloat),  # PHASE 1 DISABLE
        ("Utherm", MySingle),
        ("Center", MyDouble, 3),  # PHASE 1 DISABLE, NOTE: phase1 directly stamped into P['Pos'] when loading
        ("B", MyFloat, 3),  # PHASE 1 DISABLE
        ("Metallicity", MyFloat),  # PHASE 1 DISABLE
        ("MetalsFraction", MyFloat, GFM_N_CHEM_ELEMENTS),  # PHASE 1 DISABLE
        ("Sfr", MySingle),  # USE_SFR # PHASE 1 DISABLE
    ]
)

StarP_dtype = np.dtype(
    [  # GFM
        ("PID", np.uint32),
        ("pad_0", np.int32),  # STRUCT PADDING
        ("BirthTime", MyFloat),
        ("BirthPos", MyFloat, 3),
        ("BirthVel", MyFloat, 3),
        # GFM_STELLAR_EVOLUTION
        ("InitialMass", MyDouble),
        ("MassMetals", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("Metallicity", MyFloat),
        ("SNIaRate", MyFloat),
        ("SNIIRate", MyFloat),
        ("lastEnrichTime", MyDouble),  # GFM_DISCRETE_ENRICHMENT
        ("MassMetalsChemTags", MyFloat, GFM_N_CHEM_TAGS),  # GFM_CHEMTAGS
        ("Hsml", MyFloat),  # GFM_STELLAR EVOLUTION || GFM_WINDS
        ("Utherm", MyFloat),  # GFM_WINDS
    ]
)

StarP_dtype_mem = np.dtype(
    [  # GFM
        ("BirthTime", MyFloat),  # PHASE 1 DISABLE
        ("InitialMass", MyDouble),  # PHASE 1 DISABLE
        ("MassMetals", MyFloat, GFM_N_CHEM_ELEMENTS),  # PHASE 1 DISABLE
        ("Metallicity", MyFloat),  # PHASE 1 DISABLE
    ]
)

BHP_dtype = np.dtype(
    [  # BLACK_HOLES
        ("PID", np.uint32),
        ("BH_CountProgs", np.int32),
        ("BH_NumNgb", MyFloat),
        ("BH_Hsml", MyFloat),
        ("BH_Mass", MyFloat),
        ("BH_Mdot", MyFloat),
        ("BH_MdotBondi", MyFloat),
        ("BH_MdotEddington", MyFloat),
        ("BH_CumMass_QM", MyFloat),
        ("BH_CumEgy_QM", MyFloat),
        ("BH_CumMass_RM", MyFloat),
        ("BH_CumEgy_RM", MyFloat),
        ("BH_DtGasNeighbor", MyFloat),
        ("BH_VolSum", MyFloat),
        ("BH_Density", MyFloat),
        ("BH_U", MyFloat),
        ("BH_Pressure", MyFloat),
        ("BH_SurroundingGasVel", MyFloat, 3),
        ("SwallowID", MyIDType),
        ("HsmlCentering", MyFloat),  # BH_NEW_CENTERING
        # DRAINGAS
        ("NearestDist", MyFloat),
        ("DrainID", MyIDType),
        ("CellDensity", MyFloat),
        ("CellUtherm", MyFloat),
        ("DrainBucketMass", MyDouble),
        ("BH_WindEnergy", MyFloat),  # BH_ADIOS_WIND
        ("WindDir", MyFloat, 3),  # BH_ADIOS_RANDOMIZED
        ("HostHaloMass", MyFloat),  # GFM_AGN_RADIATION
        ("BH_DMVelDisp", MyFloat),  # GFM_WINDS_VARIABLE
        ("BH_ThermEnergy", MyFloat),  # BH_THERMALFEEDBACK,
        ("BH_Bpress", MyFloat),  # BH_USE_ALFVEN_SPEED_IN_BONDI
    ]
)

P_dtype = np.dtype(
    [
        ("Pos", MyDouble, 3),
        ("Mass", MyDouble),
        ("Vel", MyFloat, 3),
        ("GravAccel", MySingle, 3),
        ("GravPM", MySingle, 3),  # PMGRID
        # ('BirthPos', MySingle, 3), # L35n2160TNG_FIX_E7BF4CF (absent in TNG50-1)
        # ('BirthVel', MySingle, 3), # L35n2160TNG_FIX_E7BF4CF (absent in TNG50-1)
        ("Potential", MySingle),  # EVALPOTENTIAL
        ("PM_Potential", MySingle),  # EVALPOTENTIAL & PMGRID
        ("AuxDataID", MyIDType),  # GFM || BLACK_HOLES
        # TRACER_MC
        ("TracerHead", np.int32),
        ("NumberOfTracers", np.int32),
        ("OriginTask", np.int32),
        # general
        ("pad_0", np.int32),  # STRUCT PADDING
        ("ID", MyIDType),
        ("pad_1", np.int32),  # STRUCT PADDING
        ("TI_Current", integertime),
        ("OldAcc", np.float32),
        ("GravCost", np.float32, GRAVCOSTLEVELS),
        ("Type", np.uint8),
        ("SofteningType", np.uint8),
        ("TimeBinGrav", np.int8),
        ("TimeBinHydro", np.int8),
    ]
)

P_dtype_mem = np.dtype(
    [
        ("Pos", MyDouble, 3),
        ("Mass", MyDouble),
        ("Vel", MyFloat, 3),
        ("AuxDataID", MyIDType),  # GFM || BLACK_HOLES # PHASE 1 DISABLE
        ("ID", MyIDType),  # PHASE 1 DISABLE
        ("Type", np.uint8),
        ("SofteningType", np.uint8),
    ]
)

PS_dtype = np.dtype(
    [
        ("OriginIndex", np.int32),
        ("OriginTask", np.int32),
        ("TargetIndex", np.int32),
        ("TargetTask", np.int32),
        ("GrNr", np.int32),
        # SUBFIND
        ("SubNr", np.int32),
        ("OldIndex", np.int32),
        ("submark", np.int32),
        ("originindex", np.int32),
        ("origintask", np.int32),
        ("Utherm", MyFloat),
        ("Density", MyFloat),
        ("Potential", MyFloat),
        ("Hsml", MyFloat),
        ("BindingEnergy", MyFloat),
        ("Center", MyDouble, 3),  # CELL_CENTER_GRAVITY
        # SAVE_HSML_IN_SNAPSHOT
        ("SubfindHsml", MyFloat),
        ("SubfindDensity", MyFloat),
        ("SubfindDMDensity", MyFloat),
        ("SubfindVelDisp", MyFloat),
    ]
)

PS_dtype_mem = np.dtype(
    [
        ("SubNr", np.int32),
        ("OldIndex", np.int32),  # could remove and save mem, if we shuffle all PartType0 to be first
        ("Density", MyFloat),
        ("Potential", MyFloat),
        ("Hsml", MyFloat),
        ("BindingEnergy", MyFloat),
    ]
)

cand_dtype = np.dtype(
    [
        ("head", np.int32),
        ("len", np.int32),
        ("nsub", np.int32),
        ("rank", np.int32),
        ("subnr", np.int32),
        ("parent", np.int32),
        ("bound_length", np.int32),
    ]
)

ud_dtype = np.dtype([("index", np.int32)])

Subgroup_dtype = np.dtype(
    [
        ("Len", np.int32),
        ("LenType", np.int32, NTYPES),
        ("GrNr", np.int32),
        ("SubNr", np.int32),
        ("Parent", np.int32),
        ("IDMostbound", MyIDType),
        ("Mass", MyFloat),
        ("MassType", MyFloat, NTYPES),
        ("VelDisp", MyFloat),
        ("Vmax", MyFloat),
        ("VmaxRad", MyFloat),
        ("HalfmassRad", MyFloat),
        ("HalfmassRadType", MyFloat, NTYPES),
        ("MassInRad", MyFloat),
        ("MassInRadType", MyFloat, NTYPES),
        ("MassInHalfRad", MyFloat),
        ("MassInHalfRadType", MyFloat, NTYPES),
        ("MassInMaxRad", MyFloat),
        ("MassInMaxRadType", MyFloat, NTYPES),
        ("Pos", MyFloat, 3),
        ("CM", MyFloat, 3),
        ("Vel", MyFloat, 3),
        ("Spin", MyFloat, 3),
        ("BfldHalo", MyFloat),
        ("BfldDisk", MyFloat),
        ("SFR", MyFloat),
        ("SFRinRad", MyFloat),
        ("SFRinHalfRad", MyFloat),
        ("SFRinMaxRad", MyFloat),
        # ('GasMassSFR', MyFloat),
        ("GasMetallicity", MyFloat),
        ("GasMetallicityHalfRad", MyFloat),
        ("GasMetallicityMaxRad", MyFloat),
        ("GasMetalFractions", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("GasMetalFractionsHalfRad", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("GasMetalFractionsMaxRad", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("StarMetallicity", MyFloat),
        ("StarMetallicityHalfRad", MyFloat),
        ("StarMetallicityMaxRad", MyFloat),
        ("StarMetalFractions", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("StarMetalFractionsHalfRad", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("StarMetalFractionsMaxRad", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("GasMetalFractionsSfr", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("GasMetalFractionsSfrWeighted", MyFloat, GFM_N_CHEM_ELEMENTS),
        ("GasMetallicitySfr", MyFloat),
        ("GasMetallicitySfrWeighted", MyFloat),
        ("BHMass", MyFloat),
        ("BHMdot", MyFloat),
        ("WindMass", MyFloat),
        ("StellarPhotometrics", MyFloat, GFM_STELLAR_PHOTOMETRICS_BANDS),  # changed
        ("StellarPhotometricsRad", MyFloat),
        ("StellarPhotometricsMassInRad", MyFloat),
    ]
)


def load_custom_dump(sP, GrNr, phase1=False):
    """Load groups_{snapNum}/fof_{fofID}.{taskNum} custom binary data dump."""
    filePath = sP.simPath + "groups_%03d/fof_%d" % (sP.snap, GrNr)

    headerSize = 4 * 5 + 4 * 5 + 4 * 4

    # first loop: load all headers, accumulate total counts
    nChunks = len(glob.glob(filePath + ".*"))
    print("nChunks: [%d], reading headers..." % nChunks)
    assert nChunks > 0, "Custom FoF0 save files not found."

    NumP = 0
    NumSphP = 0
    NumStarP = 0
    NumBHP = 0

    i = 0
    nFound = 0

    while nFound < nChunks:
        file = "%s.%d" % (filePath, i)
        if i % 100 == 0:
            print(file)
        if not isfile(file):
            print("WARNING: [%s] does not exist! skipping..." % file)
            i += 1
            continue

        with open(file, "rb") as f:
            header = f.read(headerSize)

        TargetGrNr, NumP_loc, NumSphP_loc, NumStarP_loc, NumBHP_loc = struct.unpack("iiiii", header[0:20])
        min_ind, max_ind, min_sph_ind, max_sph_ind = struct.unpack("iiii", header[20:36])
        size_PS, size_P, size_SphP, size_StarP, size_BHP = struct.unpack("iiiii", header[36:56])

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

        i += 1
        nFound += 1

    # allocate
    group = sP.groupCatSingle(haloID=GrNr)
    print("GrNr", GrNr, " has LenType: ", group["GroupLenType"], " Len:", group["GroupLen"])

    NumDM = NumP - NumSphP - NumStarP - NumBHP
    print("Save files have total lenType: ", NumSphP, NumDM, NumStarP, NumBHP, " Len:", NumP)

    sizeBytes = NumP * size_P + NumP * size_PS + NumSphP * size_SphP + NumStarP * size_StarP + NumBHP * size_BHP
    sizeGB = sizeBytes / 1024.0**3
    print("Memory allocation for full arrays, all particles, would require [%.2f GB]" % sizeGB)

    sizeBytes = (
        group["GroupLen"] * size_P
        + group["GroupLen"] * size_PS
        + group["GroupLenType"][sP.ptNum("gas")] * size_SphP
        + group["GroupLenType"][sP.ptNum("stars")] * size_StarP
        + group["GroupLenType"][sP.ptNum("bhs")] * size_BHP
    )
    sizeGB = sizeBytes / 1024.0**3
    print("Memory allocation for full arrays, group members only, would require [%.2f GB]" % sizeGB)

    sizeBytes = (
        group["GroupLen"] * P_dtype_mem.itemsize
        + group["GroupLen"] * PS_dtype_mem.itemsize
        + group["GroupLenType"][sP.ptNum("gas")] * SphP_dtype_mem.itemsize
        + group["GroupLenType"][sP.ptNum("stars")] * StarP_dtype_mem.itemsize
        + group["GroupLenType"][sP.ptNum("bhs")] * BHP_dtype.itemsize
    )
    sizeGB = sizeBytes / 1024.0**3
    print("Memory allocation for partial arrays, group members only, will require [%.2f GB]" % sizeGB)

    sizeGB = int(1.1 * group["GroupLen"]) * node_dtype.itemsize / 1024.0**3
    print("Memory allocation for tree will require another at least [%.2f GB]" % sizeGB)
    sizeGB = group["GroupLen"] * (ud_dtype.itemsize + cand_dtype.itemsize + 16) / 1024.0**3
    print("Memory allocation for subfind identification will require another at least [%.2f GB]" % sizeGB)

    P = np.zeros(group["GroupLen"], dtype=P_dtype_mem)
    PS = np.zeros(group["GroupLen"], dtype=PS_dtype_mem)  # memset(0) in fof.c
    SphP = np.empty(group["GroupLenType"][sP.ptNum("gas")], dtype=SphP_dtype_mem)
    StarP = np.empty(group["GroupLenType"][sP.ptNum("stars")], dtype=StarP_dtype_mem)
    BHP = np.empty(group["GroupLenType"][sP.ptNum("bhs")], dtype=BHP_dtype)

    if phase1:
        gas_Center = np.zeros((NumSphP, 3), dtype=MyDouble)

    NumP = 0
    NumSphP = 0
    NumStarP = 0
    NumBHP = 0

    i = 0
    nFound = 0

    while nFound < nChunks:
        file = "%s.%d" % (filePath, i)

        if not isfile(file):
            print("WARNING: [%s] does not exist! skipping..." % file)
            i += 1
            continue

        with open(file, "rb") as f:
            header = f.read(headerSize)
            TargetGrNr, NumP_loc, NumSphP_loc, NumStarP_loc, NumBHP_loc = struct.unpack("iiiii", header[0:20])
            min_ind, max_ind, min_sph_ind, max_sph_ind = struct.unpack("iiii", header[20:36])

            PS_temp = np.fromfile(f, dtype=PS_dtype, count=NumP_loc)
            P_temp = np.fromfile(f, dtype=P_dtype, count=NumP_loc)
            SphP_temp = np.fromfile(f, dtype=SphP_dtype, count=NumSphP_loc)
            StarP_temp = np.fromfile(f, dtype=StarP_dtype, count=NumStarP_loc)
            BHP_temp = np.fromfile(f, dtype=BHP_dtype, count=NumBHP_loc)

        # particles
        w = np.where(PS_temp["GrNr"] == GrNr)
        gr_NumP = len(w[0])
        print(
            "[%4d] [%7d of %7d] particles belong to group, now have [%9d of %9d] total."
            % (i, gr_NumP, PS_temp.size, NumP + gr_NumP, group["GroupLen"]),
            flush=True,
        )
        # print(' file: sphp=%d, starp=%d, bhp=%d, min_ind=%d' % (NumSphP_loc, NumStarP_loc, NumBHP_loc, min_ind))

        # only save needed fields to optimize memory usage
        for field in PS_dtype_mem.names:
            PS[field][NumP : NumP + gr_NumP] = PS_temp[field][w]
        for field in P_dtype_mem.names:
            P[field][NumP : NumP + gr_NumP] = P_temp[field][w]

        assert P_temp["Type"].min() >= 0 and P_temp["Type"].max() < NTYPES  # sanity check for struct padding success
        assert P_temp["ID"].min() > 0 and P_temp["ID"].max() < 1000000000000000000
        assert P_temp["TimeBinGrav"].min() >= 25 and P_temp["TimeBinGrav"].max() <= 52
        assert P_temp["Pos"].min() >= 0.0 and P_temp["Pos"].max() <= sP.boxSize

        # gas
        if NumSphP_loc:
            w_gas = np.where(P_temp["Type"] == 0)
            assert min_sph_ind == 0
            if len(w_gas[0]) == NumSphP_loc:  # full
                assert max_sph_ind == NumSphP_loc - 1

            w_gas = np.where((P_temp["Type"] == 0) & (PS_temp["GrNr"] == GrNr))
            assert len(w_gas[0]) == NumSphP_loc

            for field in SphP_dtype_mem.names:
                # only save needed fields to optimize memory usage
                SphP[field][NumSphP : NumSphP + NumSphP_loc] = SphP_temp[field][w_gas]

            if phase1:  # save 'Center' separately (phase 1 only)
                gas_Center[NumSphP : NumSphP + NumSphP_loc, :] = SphP_temp["Center"][w_gas]

            NumSphP += len(w_gas[0])  # == NumSphP_loc

        # note: AuxDataID's will be good (file local), since we always write full StarP/BHP,
        # but aux[].PID will need to be adjusted by min_ind (not used anyways in subfind)

        # stars
        if NumStarP_loc:
            w_star = np.where((P_temp["Type"] == 4) & (PS_temp["GrNr"] == GrNr))
            Pw_aux = P_temp["AuxDataID"][w_star]
            assert Pw_aux.min() >= 0 and Pw_aux.max() < NumStarP_loc

            for field in StarP_dtype_mem.names:
                StarP[field][NumStarP : NumStarP + NumStarP_loc] = StarP_temp[field][Pw_aux]

            # reassign P.AuxData ID as indices into new global StarP
            global_p_inds = np.where(P[NumP : NumP + gr_NumP]["Type"] == 4)[0] + NumP

            if not phase1:
                P["AuxDataID"][global_p_inds] = np.arange(len(w_star[0])) + NumStarP

            if 0:
                # and likewise for StarP.PID (unused)
                StarP[NumStarP : NumStarP + NumStarP_loc]["PID"] = global_p_inds

                assert np.array_equal(StarP_temp[Pw_aux]["PID"] - min_ind, w_star[0])
                assert np.array_equal(P_temp["AuxDataID"][StarP_temp[Pw_aux]["PID"] - min_ind], Pw_aux)

            NumStarP += len(w_star[0])  # != NumStarP_loc !

        # bhs
        if NumBHP_loc:
            w_bhs = np.where((P_temp["Type"] == 5) & (PS_temp["GrNr"] == GrNr))
            Pw_aux = P_temp["AuxDataID"][w_bhs]
            assert Pw_aux.min() >= 0 and Pw_aux.max() < NumBHP_loc

            BHP[NumBHP : NumBHP + NumBHP_loc] = BHP_temp[Pw_aux]

            # reassign P.AuxData ID as indices into new global BHP, and likewise for BHP.PID
            global_p_inds = np.where(P[NumP : NumP + gr_NumP]["Type"] == 5)[0] + NumP

            if not phase1:
                P["AuxDataID"][global_p_inds] = np.arange(len(w_bhs[0])) + NumBHP

            BHP[NumBHP : NumBHP + NumBHP_loc]["PID"] = global_p_inds

            assert np.array_equal(BHP_temp[Pw_aux]["PID"] - min_ind, w_bhs[0])
            assert np.array_equal(P_temp["AuxDataID"][BHP_temp[Pw_aux]["PID"] - min_ind], Pw_aux)

            NumBHP += len(w_bhs[0])  # != NumBHP_loc !

        NumP += gr_NumP

        i += 1
        nFound += 1

    # final checks: counts
    assert NumP == group["GroupLen"]
    assert NumSphP == group["GroupLenType"][sP.ptNum("gas")]
    assert NumStarP == group["GroupLenType"][sP.ptNum("stars")]
    assert NumBHP == group["GroupLenType"][sP.ptNum("bhs")]

    # final checks: PID <-> AuxDataID mapping (global)
    # w_stars = np.where(P['Type'] == 4)
    # assert np.array_equal(StarP[P['AuxDataID'][w_stars]]['PID'], w_stars[0]) # unused
    # w_bhs = np.where(P['Type'] == 5)
    # assert np.array_equal(BHP[P['AuxDataID'][w_bhs]]['PID'], w_bhs[0])

    print("Particle counts of all types verified, match expected Group [%d] lengths.\n" % GrNr, flush=True)

    # verify we have collected the -right- particles
    for pt in []:  # [0,1,4,5]: # DISABLED
        w = np.where(P["Type"] == pt)
        loc_ids = P[w]["ID"]
        snap_ids = sP.snapshotSubset(pt, "ids", haloID=GrNr)
        print("Verifying PT [%d], loaded len = %d, snap len = %d" % (pt, loc_ids.size, snap_ids.size), flush=True)
        loc_ids = np.sort(loc_ids)
        snap_ids = np.sort(snap_ids)
        assert np.array_equal(loc_ids, snap_ids)

    # construct OldIndex
    offset = 0
    for i in range(P.size):
        if P[i]["Type"] == 0:
            PS["OldIndex"][i] = offset

            if phase1:  # stamp gas_Center into P['Pos'] (phase1 only)
                P[i]["Pos"] = gas_Center[offset]

            offset += 1
        else:
            PS["OldIndex"][i] = -1

    assert PS["OldIndex"].max() < SphP.size

    return P, PS, SphP, StarP, BHP


def load_snapshot_data(sP, GrNr):
    """Load the FoF particles from an actual snapshot for testing."""
    # load group metadata
    dmMass = sP.dmParticleMass
    group = sP.groupCatSingle(haloID=GrNr)
    numPartType = group["GroupLenType"]

    print("Group [%d] has total length [%d], and [%d] subhalos." % (GrNr, group["GroupLen"], group["GroupNsubs"]))

    for i in range(group["GroupNsubs"]):
        sub = sP.groupCatSingle(subhaloID=group["GroupFirstSub"] + i)
        print(
            "subnr ",
            i,
            " len",
            sub["SubhaloLen"],
            " pos:",
            sub["SubhaloPos"],
            " mostboundid",
            sub["SubhaloIDMostbound"],
            " lentype",
            sub["SubhaloLenType"],
        )

    # allocate
    P = np.empty(group["GroupLen"], dtype=P_dtype)
    PS = np.zeros(group["GroupLen"], dtype=PS_dtype)  # memset(0) in fof.c
    SphP = np.empty(numPartType[sP.ptNum("gas")], dtype=SphP_dtype)
    StarP = np.empty(numPartType[sP.ptNum("stars")], dtype=StarP_dtype)
    BHP = np.empty(numPartType[sP.ptNum("bhs")], dtype=BHP_dtype)

    # load and fill
    P_offset = 0

    for ptNum in range(numPartType.size):
        print(" loading ptNum [%d]..." % ptNum)

        # general
        for field in ["Pos", "Vel", "Mass", "Potential", "ID"]:
            if field == "Mass" and sP.isPartType(ptNum, "dm"):
                P[P_offset : P_offset + numPartType[ptNum]][field] = dmMass
                continue

            P[P_offset : P_offset + numPartType[ptNum]][field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

        P[P_offset : P_offset + numPartType[ptNum]]["Type"] = ptNum
        P[P_offset : P_offset + numPartType[ptNum]]["SofteningType"] = (
            1  # L35n2160TNG (DM and Stars) (gas/BHs overwritten later)
        )

        # gas only, and handle SphP
        if sP.isPartType(ptNum, "gas"):
            for field in ["Density", "Sfr", "Utherm", "Center", "Volume"]:
                SphP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

        # stars only, and handle StarP
        if sP.isPartType(ptNum, "stars"):
            # StarP['PID'] = np.arange(P_offset, P_offset + numPartType[ptNum]) # unused
            for field in ["BirthTime"]:
                StarP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

            P[P_offset : P_offset + numPartType[ptNum]]["AuxDataID"] = np.arange(numPartType[ptNum])

        # BHs only, and handle BHP
        if sP.isPartType(ptNum, "bhs"):
            BHP["PID"] = np.arange(P_offset, P_offset + numPartType[ptNum])
            for field in ["BH_Mass", "BH_Mdot"]:
                BHP[field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

            P[P_offset : P_offset + numPartType[ptNum]]["AuxDataID"] = np.arange(numPartType[ptNum])

        for field in ["SubfindHsml", "SubfindDensity", "SubfindDMDensity", "SubfindVelDisp"]:
            PS[P_offset : P_offset + numPartType[ptNum]][field] = sP.snapshotSubset(ptNum, field, haloID=GrNr)

        P_offset += numPartType[ptNum]

    # PS
    PS["Hsml"] = PS["SubfindHsml"]
    PS["Density"] = PS["SubfindDensity"]

    w = np.where(P["Type"] == 0)  # todo check
    PS["OldIndex"][w] = np.arange(group["GroupLenType"][0])  # todo check

    return P, PS, SphP, StarP, BHP


@jit(nopython=True, nogil=True)
def _updateNodeRecursiveExtra(no, sib, last, next_node, tree_nodes, P, ForceSoftening):
    """As _updateNodeRecursive(), but also compute additional information for the tree nodes such as masses."""
    pp = 0
    nextsib = 0

    NumPart = next_node.size

    if no >= NumPart:
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last - NumPart]["nextnode"] = no
            else:
                next_node[last] = no

        last = no

        # initial values
        mass = 0.0
        com = np.zeros(3, dtype=np.float64)
        mass_per_type = np.zeros(NSOFTTYPES, dtype=np.float64)

        maxsofttype = np.uint8(NSOFTTYPES + NSOFTTYPES_HYDRO)
        maxhydrosofttype = NSOFTTYPES
        minhydrosofttype = NSOFTTYPES + NSOFTTYPES_HYDRO - 1

        # loop over each of the 8 daughters
        for i in range(8):
            p = tree_nodes[no - NumPart]["suns"][i]

            if p >= 0:
                # check if we have a sibling on the same level
                j = i + 1
                while j < 8:
                    pp = tree_nodes[no - NumPart]["suns"][j]
                    if pp >= 0:
                        break
                    j += 1  # unusual syntax so that j==8 at the end of the loop if we never break

                if j < 8:  # yes, we do
                    nextsib = pp
                else:
                    nextsib = sib

                last = _updateNodeRecursiveExtra(p, nextsib, last, next_node, tree_nodes, P, ForceSoftening)

                if p < NumPart:
                    # individual particle
                    mass += P[p]["Mass"]
                    com += P[p]["Mass"] * P[p]["Pos"][:]

                    if ForceSoftening[maxsofttype] < ForceSoftening[P[p]["SofteningType"]]:
                        maxsofttype = P[p]["SofteningType"]

                    if P[p]["Type"] == 0:
                        mass_per_type[0] += P[p]["Mass"]

                        if maxhydrosofttype < P[p]["SofteningType"]:
                            maxhydrosofttype = P[p]["SofteningType"]
                        if minhydrosofttype > P[p]["SofteningType"]:
                            minhydrosofttype = P[p]["SofteningType"]
                    else:
                        mass_per_type[P[p]["SofteningType"]] += P[p]["Mass"]
                else:
                    # internal node
                    ind = p - NumPart

                    mass += tree_nodes[ind]["mass"]
                    com += tree_nodes[ind]["mass"] * tree_nodes[ind]["com"]

                    if ForceSoftening[maxsofttype] < ForceSoftening[tree_nodes[ind]["maxsofttype"]]:
                        maxsofttype = tree_nodes[ind]["maxsofttype"]

                    for k in range(NSOFTTYPES):
                        mass_per_type[k] += tree_nodes[ind]["mass_per_type"][k]

                    if maxhydrosofttype < tree_nodes[ind]["maxhydrosofttype"]:
                        maxhydrosofttype = tree_nodes[ind]["maxhydrosofttype"]
                    if minhydrosofttype > tree_nodes[ind]["minhydrosofttype"]:
                        minhydrosofttype = tree_nodes[ind]["minhydrosofttype"]

        # update node properties
        ind = no - NumPart

        if mass > 0.0:
            com /= mass
        else:
            com = tree_nodes[ind]["center"][:]

        tree_nodes[ind]["com"][:] = com[:]

        tree_nodes[ind]["mass"] = mass
        tree_nodes[ind]["maxsofttype"] = maxsofttype
        tree_nodes[ind]["mass_per_type"][:] = mass_per_type[:]
        tree_nodes[ind]["maxhydrosofttype"] = maxhydrosofttype
        tree_nodes[ind]["minhydrosofttype"] = minhydrosofttype

        tree_nodes[ind]["sibling"] = sib

    else:
        # single particle or pseudo particle
        if last >= 0:
            if last >= NumPart:
                tree_nodes[last - NumPart]["nextnode"] = no
            else:
                next_node[last] = no

        last = no

    return last  # avoid use of global in numba


@jit(nopython=True, nogil=True)
def _treeExtent(P):
    """Determine extent for non-periodic (local) tree."""
    xyzMin = np.zeros(3, dtype=np.float64)
    xyzMax = np.zeros(3, dtype=np.float64)

    for j in range(3):
        xyzMin[j] = 1.0e35  # MAX_REAL_NUMBER
        xyzMax[j] = -1.0e35  # MAX_REAL_NUMBER

    for i in range(P.size):
        for j in range(3):
            if P[i]["Pos"][j] > xyzMax[j]:
                xyzMax[j] = P[i]["Pos"][j]
            if P[i]["Pos"][j] < xyzMin[j]:
                xyzMin[j] = P[i]["Pos"][j]

    # determine maximum extension
    extent = 0.0

    for j in range(3):
        if xyzMax[j] - xyzMin[j] > extent:
            extent = xyzMax[j] - xyzMin[j]

    return xyzMin, xyzMax, extent


@jit(nopython=True, nogil=True)
def _constructTree(P, boxSizeSim, xyzMin, xyzMax, extent, next_node, tree_nodes, ForceSoftening):
    """Core routine for calcHsml(), see below."""
    subnode = 0
    parent = -1
    lenHalf = 0.0

    # Nodes_base and Nodes are both pointers to the arrays of NODE structs
    # Nodes_base is allocated with size >NumPart, and entries >=NumPart are "internal nodes"
    #  while entries from 0 to NumPart-1 are leafs (actual particles)
    #  Nodes just points to Nodes_base-NumPart (such that Nodes[no]=Nodes_base[no-NumPart])

    # select first node
    NumPart = P.size
    nFree = NumPart

    # create an empty root node
    if boxSizeSim > 0.0:
        # periodic, set center position and extent
        for j in range(3):
            tree_nodes[0]["center"][j] = 0.5 * boxSizeSim
            tree_nodes[0]["length"] = boxSizeSim
    else:
        # non-periodic
        if extent == 0.0:
            # do not have a pre-computed xyzMin, xyzMax, extent, so determine now
            xyzMin, xyzMax, extent = _treeExtent(P)

        # set center position and extent
        for j in range(3):
            tree_nodes[0]["center"][j] = 0.5 * (xyzMin[j] + xyzMax[j])
        tree_nodes[0]["length"] = extent

    # daughter slots of root node all start empty
    for i in range(8):
        tree_nodes[0]["suns"][i] = -1

    numNodes = 1
    nFree += 1

    # now insert all particles and so construct the tree
    for i in range(NumPart):
        # start at the root node
        no = NumPart

        # insert particle i
        while 1:
            if no >= NumPart:  # we are dealing with an internal node
                # to which subnode will this particle belong
                subnode = 0
                ind = no - NumPart

                if P[i]["Pos"][0] > tree_nodes[ind]["center"][0]:
                    subnode += 1
                if P[i]["Pos"][1] > tree_nodes[ind]["center"][1]:
                    subnode += 2
                if P[i]["Pos"][2] > tree_nodes[ind]["center"][2]:
                    subnode += 4

                # get the next node
                nn = tree_nodes[ind]["suns"][subnode]

                if nn >= 0:  # ok, something is in the daughter slot already, need to continue
                    parent = no  # note: subnode can still be used in the next step of the walk
                    no = nn
                else:
                    # here we have found an empty slot where we can attach the new particle as a leaf
                    tree_nodes[ind]["suns"][subnode] = i
                    break  # done for this particle
            else:
                # we try to insert into a leaf with a single particle - need to generate a new internal
                # node at this point, because every leaf is only allowed to contain one particle
                tree_nodes[parent - NumPart]["suns"][subnode] = nFree
                ind1 = parent - NumPart
                ind2 = nFree - NumPart

                tree_nodes[ind2]["length"] = 0.5 * tree_nodes[ind1]["length"]
                lenHalf = 0.25 * tree_nodes[ind1]["length"]

                if subnode & 1:
                    tree_nodes[ind2]["center"][0] = tree_nodes[ind1]["center"][0] + lenHalf
                else:
                    tree_nodes[ind2]["center"][0] = tree_nodes[ind1]["center"][0] - lenHalf

                if subnode & 2:
                    tree_nodes[ind2]["center"][1] = tree_nodes[ind1]["center"][1] + lenHalf
                else:
                    tree_nodes[ind2]["center"][1] = tree_nodes[ind1]["center"][1] - lenHalf

                if subnode & 4:
                    tree_nodes[ind2]["center"][2] = tree_nodes[ind1]["center"][2] + lenHalf
                else:
                    tree_nodes[ind2]["center"][2] = tree_nodes[ind1]["center"][2] - lenHalf

                for j in range(8):
                    tree_nodes[ind2]["suns"][j] = -1

                # which subnode
                subnode = 0

                if P[no]["Pos"][0] > tree_nodes[ind2]["center"][0]:
                    subnode += 1
                if P[no]["Pos"][1] > tree_nodes[ind2]["center"][1]:
                    subnode += 2
                if P[no]["Pos"][2] > tree_nodes[ind2]["center"][2]:
                    subnode += 4

                if tree_nodes[ind2]["length"] < 1e-8:  # 1e-4 in the past
                    # may have particles at identical locations, in which case randomize the subnode
                    # index to put the particle into a different leaf (happens well below the
                    # gravitational softening scale)
                    subnode = int(np.random.rand())
                    subnode = max(subnode, 7)

                tree_nodes[ind2]["suns"][subnode] = no

                no = nFree  # resume trying to insert the new particle at the newly created internal node

                numNodes += 1
                nFree += 1

                if numNodes >= tree_nodes.size:
                    # exceeding tree allocated size, need to increase and redo
                    return -1

    # now compute the (sibling,nextnode,next_node) recursively
    last = np.int32(-1)

    last = _updateNodeRecursiveExtra(NumPart, -1, last, next_node, tree_nodes, P, ForceSoftening)

    if last >= NumPart:
        tree_nodes[last - NumPart]["nextnode"] = -1
    else:
        next_node[last] = -1

    return numNodes


@jit(nopython=True, nogil=True)
def _treeSearchIndices(P, xyz, h, boxSizeSim, next_node, tree_nodes):
    """Helper routine for calcParticleIndices(), see below."""
    boxHalf = 0.5 * boxSizeSim
    h2 = h * h
    numNgbInH = 0

    # allocate, unfortunately unclear how safe we have to be here
    NumPart = next_node.size
    inds = np.empty(NumPart, dtype=np.int64)
    dists2 = np.empty(NumPart, dtype=np.float64)

    # 3D-normalized kernel
    # C1 = 2.546479089470  # COEFF_1
    # C2 = 15.278874536822 # COEFF_2
    # C3 = 5.092958178941  # COEFF_5
    # CN = 4.188790204786  # NORM_COEFF (4pi/3)

    # start search
    no = NumPart

    while no >= 0:
        if no < NumPart:
            # single particle
            assert next_node[no] != no  # Going into infinite loop.

            p = no
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST(P[p]["Pos"][0] - xyz[0], boxHalf, boxSizeSim)
            if dx < -h or dx > h:
                continue

            dy = _NEAREST(P[p]["Pos"][1] - xyz[1], boxHalf, boxSizeSim)
            if dy < -h or dy > h:
                continue

            dz = _NEAREST(P[p]["Pos"][2] - xyz[2], boxHalf, boxSizeSim)
            if dz < -h or dz > h:
                continue

            # spherical exclusion if we've made it this far
            r2 = dx * dx + dy * dy + dz * dz
            if r2 >= h2:
                continue

            # count
            inds[numNgbInH] = p
            dists2[numNgbInH] = r2
            numNgbInH += 1

        else:
            # internal node
            ind = no - NumPart
            no = tree_nodes[ind]["sibling"]  # in case the node can be discarded

            if (
                _NEAREST(tree_nodes[ind]["center"][0] - xyz[0], boxHalf, boxSizeSim) + 0.5 * tree_nodes[ind]["length"]
                < -h
            ):
                continue
            if (
                _NEAREST(tree_nodes[ind]["center"][0] - xyz[0], boxHalf, boxSizeSim) - 0.5 * tree_nodes[ind]["length"]
                > h
            ):
                continue

            if (
                _NEAREST(tree_nodes[ind]["center"][1] - xyz[1], boxHalf, boxSizeSim) + 0.5 * tree_nodes[ind]["length"]
                < -h
            ):
                continue
            if (
                _NEAREST(tree_nodes[ind]["center"][1] - xyz[1], boxHalf, boxSizeSim) - 0.5 * tree_nodes[ind]["length"]
                > h
            ):
                continue

            if (
                _NEAREST(tree_nodes[ind]["center"][2] - xyz[2], boxHalf, boxSizeSim) + 0.5 * tree_nodes[ind]["length"]
                < -h
            ):
                continue
            if (
                _NEAREST(tree_nodes[ind]["center"][2] - xyz[2], boxHalf, boxSizeSim) - 0.5 * tree_nodes[ind]["length"]
                > h
            ):
                continue

            no = tree_nodes[ind]["nextnode"]  # we need to open the node

    if numNgbInH > 0:
        inds = inds[0:numNgbInH]
        dists2 = dists2[0:numNgbInH]
        return numNgbInH, inds, dists2

    return 0, inds, dists2


@jit(nopython=True, nogil=True, cache=True)
def treeSearchIndicesIterate(P, xyz, h_guess, nNGB, boxSizeSim, next_node, tree_nodes):
    """Helper routine for subfind(), see below.

    Note: no nNGBDev, instead we terminate if we ever find >=nNGB, and the return is sorted by distance.
    """
    if h_guess == 0.0:
        h_guess = 1.0

    iter_num = 0

    while 1:
        iter_num += 1

        assert iter_num < 1000  # Convergence failure, too many iterations.

        numNgbInH, inds, dists_sq = _treeSearchIndices(P, xyz, h_guess, boxSizeSim, next_node, tree_nodes)

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


node_dtype = np.dtype(
    [
        ("length", np.float64),
        ("center", np.float64, 3),
        ("suns", np.int32, 8),  # pointers to daughter nodes
        ("sibling", np.int32),  # next node in the walk, in case the current node can be used
        ("nextnode", np.int32),  # next node in the walk, in case the current node needs to be opened
        ("com", np.float64, 3),  # center of mass
        ("mass", np.float64),  # mass of node
        ("maxsofttype", np.uint8),
        ("maxhydrosofttype", np.uint8),
        ("minhydrosofttype", np.uint8),
        ("mass_per_type", np.float32, NSOFTTYPES),  # TODO: changed from np.float64 for memory (snap 99)
    ]
)


@jit(nopython=True)
def buildFullTree(P, boxSizeSim, xyzMin, xyzMax, extent, ForceSoftening):
    """As above, but minimal and JITed."""
    NumPart = P.size
    NextNode = np.zeros(NumPart, dtype=np.int32)

    # tree allocation and construction (iterate in case we need to re-allocate for larger number of nodes)
    for num_iter in range(10):
        # allocate
        MaxNodes = int((num_iter + 1.1) * NumPart) + 1
        if MaxNodes < 100:
            MaxNodes = 100

        TreeNodes = np.zeros(MaxNodes, dtype=node_dtype)

        # construct: call JIT compiled kernel
        numNodes = _constructTree(P, boxSizeSim, xyzMin, xyzMax, extent, NextNode, TreeNodes, ForceSoftening)

        if numNodes > 0:
            break

    return NextNode, TreeNodes


@jit(nopython=True, nogil=True)
def subfind_treeevaluate_potential(target, P, ForceSoftening, next_node, tree_nodes, boxHalf, boxSizeSim):
    """Evaluate gravitational potential using the tree."""
    pos = P[target]["Pos"]
    h_i = ForceSoftening[P[target]["SofteningType"]]

    pot = 0

    # start search
    # note: NumPart here is len (local), as opposed to LocMaxPart in arepo, because our local tree
    # construction does not place the root node at LocMaxPart (i.e. aware of P size) but rather at len
    # (i.e. only aware of loc_P size)
    NumPart = next_node.size
    no = NumPart

    while no >= 0:
        indi_flag1 = -1  # MULTIPLE_NODE_SOFTENING
        indi_flag2 = 0

        if no < NumPart:
            # single particle
            assert next_node[no] != no  # Going into infinite loop.

            p = no
            no = next_node[no]

            # box-exclusion along each axis
            dx = _NEAREST(P[p]["Pos"][0] - pos[0], boxHalf, boxSizeSim)
            dy = _NEAREST(P[p]["Pos"][1] - pos[1], boxHalf, boxSizeSim)
            dz = _NEAREST(P[p]["Pos"][2] - pos[2], boxHalf, boxSizeSim)

            r2 = dx * dx + dy * dy + dz * dz

            mass = P[p]["Mass"]

            h_j = ForceSoftening[P[p]["SofteningType"]]

            if h_j > h_i:
                hmax = h_j
            else:
                hmax = h_i
        else:
            # internal node
            ind_nop = no - NumPart
            mass = tree_nodes[ind_nop]["mass"]

            dx = _NEAREST(tree_nodes[ind_nop]["com"][0] - pos[0], boxHalf, boxSizeSim)
            dy = _NEAREST(tree_nodes[ind_nop]["com"][1] - pos[1], boxHalf, boxSizeSim)
            dz = _NEAREST(tree_nodes[ind_nop]["com"][2] - pos[2], boxHalf, boxSizeSim)

            r2 = dx * dx + dy * dy + dz * dz

            # check Barnes-hut opening criterion
            if tree_nodes[ind_nop]["length"] ** 2 > r2 * ErrTolThetaSubfind**2:
                # open the node
                if mass:
                    no = tree_nodes[ind_nop]["nextnode"]
                    continue

            h_j = ForceSoftening[tree_nodes[ind_nop]["maxsofttype"]]

            if h_j > h_i:
                # multiple hydro softenings in this node? compare to maximum
                if tree_nodes[ind_nop]["maxhydrosofttype"] != tree_nodes[ind_nop]["minhydrosofttype"]:
                    if tree_nodes[ind_nop]["mass_per_type"][0] > 0:
                        if r2 < ForceSoftening[tree_nodes[ind_nop]["maxhydrosofttype"]] ** 2:
                            # open the node
                            no = tree_nodes[ind_nop]["nextnode"]
                            continue

                indi_flag1 = 0
                indi_flag2 = NSOFTTYPES
                hmax = h_j
            else:
                hmax = h_i

            no = tree_nodes[ind_nop]["sibling"]  # node can be used

        # proceed (use node)
        r = np.sqrt(r2)

        for ptype in range(indi_flag1, indi_flag2):
            if ptype >= 0:
                mass = tree_nodes[ind_nop]["mass_per_type"][ptype]
                if ptype == 0:
                    h_j = ForceSoftening[tree_nodes[ind_nop]["maxhydrosofttype"]]
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
                        wp = (
                            -3.2
                            + 0.066666666667 / u
                            + u * u * (10.666666666667 + u * (-16.0 + u * (9.6 - 2.133333333333 * u)))
                        )

                    pot += mass * h_inv * wp

    return pot


@jit(nopython=True, parallel=True)
def subfind_unbind_calculate_potential(
    num, ud, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize, PS, G, atime
):
    """Loop to parallelize."""
    for i in prange(num):
        p = ud[i]["index"]

        pot = subfind_treeevaluate_potential(i, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize)

        PS[p]["Potential"] = G / atime * pot


@jit(nopython=True, parallel=True)
def subfind_unbind_calculate_potential_weak(
    num, ud, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize, PS, G, atime, weakly_bound_limit
):
    """Loop to parallelize."""
    for i in prange(num):
        p = ud[i]["index"]

        if PS[p]["BindingEnergy"] >= weakly_bound_limit:
            # note: this is a bugfix from TNG codebase subfind, which has a typo making pot completely unused
            # pot = subfind_treeevaluate_potential(i, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize)
            # PS[p]['Potential'] = G / atime * pot

            PS[p]["Potential"] *= G / atime  # TODO: BUG ACTIVE FOR 'nopot' run


@jit(nopython=True)
def subfind_unbind(
    P,
    SphP,
    PS,
    ud,
    num,
    vel_to_phys,
    H_of_a,
    G,
    atime,
    boxsize,
    SofteningTable,
    ForceSoftening,
    xyzMin,
    xyzMax,
    extent,
    central_flag,
):
    """Unbinding."""
    max_iter = 10000
    unbind_percent_threshold = (
        0.00001  # if we remove <unbind_percent_threshold*N of the subhalo particles in an iter, stop
    )

    weakly_bound_limit = 0
    len_non_gas = 0
    minpot = 0
    boxhalf = boxsize * 0.5
    unbound = 0

    iter_num = 0
    phaseflag = 0

    bnd_energy = np.zeros(num, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    s = np.zeros(3, dtype=np.float64)
    dv = np.zeros(3, dtype=np.float64)
    dx = np.zeros(3, dtype=np.float64)

    while 1:
        iter_num += 1

        # build local tree, including only particles still inside the candidate
        loc_P = np.zeros(num, dtype=P_dtype_mem)

        for i in range(num):
            loc_P[i] = P[ud[i]["index"]]

        NextNode, TreeNodes = buildFullTree(loc_P, 0.0, xyzMin, xyzMax, extent, ForceSoftening)

        # compute the potential
        if phaseflag == 0:
            # redo for all particles (threaded target)
            subfind_unbind_calculate_potential(
                num, ud, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize, PS, G, atime
            )

            # find particle with the minimum potential
            minindex = -1
            minpot = 1.0e30

            for i in range(num):
                p = ud[i]["index"]

                if PS[p]["Potential"] < minpot or minindex == -1:
                    # new minimum potential found
                    minpot = PS[p]["Potential"]
                    minindex = p

            # position of minimum potential (CELL_CENTER_GRAVITY)
            pos = P[minindex]["Pos"]
        else:
            # only repeat for those particles close to the unbinding threshold
            subfind_unbind_calculate_potential_weak(
                num, ud, loc_P, ForceSoftening, NextNode, TreeNodes, boxhalf, boxsize, PS, G, atime, weakly_bound_limit
            )

        # calculate the bulk velocity and center of mass
        v *= 0
        s *= 0
        TotMass = 0

        for i in range(num):
            p = ud[i]["index"]

            for j in range(3):
                ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                s[j] += P[p]["Mass"] * ddxx
                v[j] += P[p]["Mass"] * P[p]["Vel"][j]

            TotMass += P[p]["Mass"]

        for j in range(3):
            v[j] /= TotMass
            s[j] /= TotMass  # center of mass
            s[j] += pos[j]

            while s[j] < 0:  # PERIODIC
                s[j] += boxsize
            while s[j] >= boxsize:
                s[j] -= boxsize

        # compute binding energy for all particles
        for i in range(num):
            p = ud[i]["index"]

            for j in range(3):
                dv[j] = vel_to_phys * (P[p]["Vel"][j] - v[j])
                dx[j] = atime * _NEAREST(P[p]["Pos"][j] - s[j], boxhalf, boxsize)
                dv[j] += H_of_a * dx[j]  # Hubble expansion, per coordinate

            PS[p]["BindingEnergy"] = PS[p]["Potential"] + 0.5 * (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2])
            PS[p]["BindingEnergy"] += G / atime * P[p]["Mass"] / SofteningTable[P[p]["SofteningType"]]

            if P[p]["Type"] == 0:
                PS[p]["BindingEnergy"] += SphP[PS[p]["OldIndex"]]["Utherm"]

            bnd_energy[i] = PS[p]["BindingEnergy"]

        # sort by binding energy, largest first
        bnd_energy = np.sort(bnd_energy)[::-1]

        quarter_ind = int(np.floor(0.25 * num))
        energy_limit = bnd_energy[quarter_ind]
        unbound = 0

        for i in range(num - 1):
            if bnd_energy[i] > 0:
                unbound += 1
            else:
                unbound -= 1

            if unbound <= 0:
                break

        weakly_bound_limit = bnd_energy[i]

        # now omit unbound particles, but at most 1/4 of the original size
        unbound = 0
        len_non_gas = 0
        i = 0

        while i < num:
            p = ud[i]["index"]

            if PS[p]["BindingEnergy"] > 0 and PS[p]["BindingEnergy"] > energy_limit:
                unbound += 1
                ud[i] = ud[num - 1]
                i -= 1
                num -= 1

            if P[p]["Type"] != 0:
                len_non_gas += 1

            i += 1

        if central_flag:
            print("central: iter,num,unbound,phaseflag =", iter_num, num, unbound, phaseflag)

        # already too small?
        if num < DesLinkNgb:
            break

        # alternate full vs. partial potential calculations
        if phaseflag == 0:
            if unbound > 0:
                phaseflag = 1

            # note: this earlier termination is an optimization not in the original subfind
            if central_flag and unbound < int(unbind_percent_threshold * num):
                break
        else:
            if unbound == 0:
                phaseflag = 0  # repeat everything once more for all particles
                unbound = 1

        if iter_num > max_iter:
            raise Exception("Not good.")

        # convergence, we are done
        if unbound <= 0:
            break

    return num, len_non_gas


@jit(nopython=True)
def subfind(P, PS, SphP, StarP, BHP, atime, H_of_a, G, boxsize, SofteningTable, ForceSoftening):
    """Run serial subfind. (Offs = 0)."""
    # estimate the maximum number of substructures we need to store (conservative upper limit)
    N = P.size

    # allocate
    candidates = np.zeros(N, dtype=cand_dtype)
    Head = np.zeros(N, dtype=np.int32) - 1
    Next = np.zeros(N, dtype=np.int32) - 1
    Tail = np.zeros(N, dtype=np.int32) - 1
    Len = np.zeros(N, dtype=np.int32)
    ud = np.zeros(N, dtype=ud_dtype)

    for i in range(N):
        ud[i]["index"] = i

    # order particles (P, PS) in the order of decreasing density
    sort_inds = np.argsort(PS["Density"])[::-1]  # descending
    sort_inds_inv = np.zeros(sort_inds.size, dtype=np.int32)
    sort_inds_inv[sort_inds] = np.arange(sort_inds.size)

    # note: temporarily break the association with SphP[] and other arrays!
    PS = PS[sort_inds]
    P = P[sort_inds]

    # for i in range(StarP.size):
    #    StarP[i]['PID'] = sort_inds_inv[ StarP[i]['PID'] ] # unused, save memory
    for i in range(BHP.size):
        BHP[i]["PID"] = sort_inds_inv[BHP[i]["PID"]]

    # build tree for all particles of this group
    BoxSizeSim = 0.0  # tree searches are non-periodic, and we use local (non-box-global) extents
    xyzMin, xyzMax, extent = _treeExtent(P)
    NextNode, TreeNodes = buildFullTree(P, BoxSizeSim, xyzMin, xyzMax, extent, ForceSoftening)

    # process every particle
    head = 0
    count_cand = 0
    listofdifferent = np.zeros(2, dtype=np.int32)

    print("Tree built and arrays sorted, beginning neighbor search...")

    for i in range(N):
        # find neighbors, note: returned neighbors are already sorted by distance (ascending)
        # if i % int(N/10) == 0: print(' ',np.round(float(i)/N*100),'%')
        pos = P[i]["Pos"]
        h_guess = PS[i]["Hsml"]

        inds, dists = treeSearchIndicesIterate(P, pos, h_guess, DesLinkNgb, BoxSizeSim, NextNode, TreeNodes)

        # process neighbors
        ndiff = 0
        ngbs = 0

        for j in range(DesLinkNgb):
            ngb_index = inds[j]

            if ngbs >= 2:
                break

            if ngb_index == i:
                continue  # to exclude the particle itself

            # we only look at neighbors that are denser
            if PS[ngb_index]["Density"] > PS[i]["Density"]:
                ngbs += 1

                if Head[ngb_index] >= 0:  # neighbor is attached to a group
                    if ndiff == 1:
                        if listofdifferent[0] == Head[ngb_index]:
                            continue

                    # a new group has been found
                    listofdifferent[ndiff] = Head[ngb_index]
                    ndiff += 1
                else:
                    raise Exception("this may not occur")

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
            # the particle merges two groups together
            head = listofdifferent[0]
            head_attach = listofdifferent[1]

            if Len[head_attach] > Len[head] or (Len[head_attach] == Len[head] and head_attach < head):
                # other group is longer, swap them. for equal length, take the larger head value
                head = listofdifferent[1]
                head_attach = listofdifferent[0]

            # only in case the attached group is long enough do we register it as a subhalo candidate
            if Len[head_attach] >= DesLinkNgb:
                candidates[count_cand]["len"] = Len[head_attach]
                candidates[count_cand]["head"] = Head[head_attach]
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
            raise Exception("Cannot occur.")

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

    candidates[count_cand]["len"] = N
    candidates[count_cand]["head"] = head
    count_cand += 1

    print("Searches done, ended with [", count_cand, "] candidates, now unbinding...")

    vel_to_phys = 1.0 / atime
    nsubs = 0

    # go through them once and assign the rank
    p = head
    rank = 0

    for _i in range(N):
        Len[p] = rank
        rank += 1
        p = Next[p]

    # for each candidate, we now pull out the rank of its head
    for i in range(count_cand):
        candidates[i]["rank"] = Len[candidates[i]["head"]]

    for i in range(N):
        Tail[i] = -1

    # do gravitational unbinding on each candidate
    for i in range(count_cand):
        p = candidates[i]["head"]
        len = 0

        # create local index list for members of this candidate
        for _ in range(candidates[i]["len"]):
            if Tail[p] < 0:
                assert p >= 0
                ud[len]["index"] = p
                len += 1
            p = Next[p]

        central_flag = False
        if i == count_cand - 1:
            central_flag = True

        if len >= DesLinkNgb:
            len, len_non_gas = subfind_unbind(
                P,
                SphP,
                PS,
                ud,
                len,
                vel_to_phys,
                H_of_a,
                G,
                atime,
                boxsize,
                SofteningTable,
                ForceSoftening,
                xyzMin,
                xyzMax,
                extent,
                central_flag,
            )

        if len >= DesLinkNgb:
            # we found a substructure
            for j in range(len):
                Tail[ud[j]["index"]] = nsubs  # use this to flag the substructures

            candidates[i]["nsub"] = nsubs
            candidates[i]["bound_length"] = len
            nsubs += 1
        else:
            candidates[i]["nsub"] = -1
            candidates[i]["bound_length"] = 0

    return count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP


def subfind_properties_1(candidates):
    """Determine the parent subhalo for each candidate."""
    # sort candidates on (bound_length,rank)
    candidates["bound_length"] *= -1
    sort_inds = np.argsort(candidates, order=["bound_length", "rank"])  # bound_length descending, rank ascending
    candidates["bound_length"] *= -1

    candidates = candidates[sort_inds]

    # reduce to actual (non-blank) candidates. note: not done in subfind
    count_cand = np.count_nonzero(candidates["bound_length"])
    candidates = candidates[0:count_cand]

    # sort with comparator function:
    for i in range(count_cand):
        candidates[i]["subnr"] = i
        candidates[i]["parent"] = 0

    candidates["len"] *= -1
    sort_inds = np.argsort(candidates, order=["rank", "len"])  # rank ascending, len descending
    candidates["len"] *= -1

    candidates = candidates[sort_inds]

    for k in range(count_cand):
        for j in range(k + 1, count_cand):
            if candidates[j]["rank"] > candidates[k]["rank"] + candidates[k]["len"]:
                break

            if candidates[k]["rank"] + candidates[k]["len"] >= candidates[j]["rank"] + candidates[j]["len"]:
                if candidates[k]["bound_length"] >= DesLinkNgb:
                    candidates[j]["parent"] = candidates[k]["subnr"]
            else:
                raise Exception("Not good.")

    sort_inds = np.argsort(candidates["subnr"])
    candidates = candidates[sort_inds]

    return candidates


@jit(nopython=True)
def get_time_difference_in_Gyr(a0, a1):
    """Time difference between two cosmological scale factors (from timestep.c)."""
    OmegaLambda = np.double(0.6911)  # TODO: not generalized, can take all from sP
    Omega0 = np.double(0.3089)
    HUBBLE = np.double(3.24078e-18)
    HubbleParam = np.double(0.6774)
    SEC_PER_GYR = np.double(3.15576e16)

    factor1 = 2.0 / (3.0 * np.sqrt(OmegaLambda))
    factor3 = np.sqrt(OmegaLambda / Omega0)

    term1 = factor3 * a0**1.5
    term2 = np.sqrt(1 + OmegaLambda / Omega0 * a0**3)

    t0 = factor1 * np.log(term1 + term2)

    term1 = factor3 * a1**1.5
    term2 = np.sqrt(1 + OmegaLambda / Omega0 * a1**3)

    t1 = factor1 * np.log(term1 + term2)

    result = t1 - t0

    time_diff = result / (HUBBLE * HubbleParam)  # seconds
    time_diff /= SEC_PER_GYR  # Gyr

    return time_diff


@jit(nopython=True)
def assign_stellar_photometrics(i, P, StarP, mags, LogMetallicity_bins, LogAgeInGyr_bins, TableMags, atime):
    """Interpolate for stellar magnitudes (from GFM/stellar_photometrics.c)."""
    if P[i]["Type"] != 4 or P[i]["Mass"] == 0 or StarP[P[i]["AuxDataID"]]["BirthTime"] <= 0:
        return

    # prepare SSP inputs
    mass = StarP[P[i]["AuxDataID"]]["InitialMass"]
    metallicity = StarP[P[i]["AuxDataID"]]["Metallicity"]
    birth_a = StarP[P[i]["AuxDataID"]]["BirthTime"]

    ageInGyr = get_time_difference_in_Gyr(birth_a, atime)
    log_AgeInGyr = np.log10(ageInGyr)
    mass_in_msun = mass * 1e10 / 0.6774  # TODO: not generalized

    if metallicity < 0:
        log_Metallicity = GFM_MIN_METAL
    else:
        log_Metallicity = np.log10(metallicity)

    # search stellar age
    if log_AgeInGyr > LogAgeInGyr_bins[0]:
        i1 = 0
        while i1 < LogAgeInGyr_bins.size - 2 and log_AgeInGyr > LogAgeInGyr_bins[i1 + 1]:
            i1 += 1

        i2 = i1 + 1

        if i2 >= LogAgeInGyr_bins.size:
            i2 = LogAgeInGyr_bins.size - 1

        if log_AgeInGyr >= LogAgeInGyr_bins[0] and log_AgeInGyr <= LogAgeInGyr_bins[-1]:
            d_i_local = log_AgeInGyr - LogAgeInGyr_bins[i1]
        else:
            d_i_local = 0.0

        delta_i = LogAgeInGyr_bins[i2] - LogAgeInGyr_bins[i1]

        if delta_i > 0:
            d_i_local = d_i_local / delta_i
        else:
            d_i_local = 0.0
    else:
        i1 = 0
        i2 = 0
        d_i_local = 0.0

    d_i = d_i_local

    # search metallicity
    if log_Metallicity > LogMetallicity_bins[0]:
        k1 = 0
        while k1 < LogMetallicity_bins.size - 2 and log_Metallicity > LogMetallicity_bins[k1 + 1]:
            k1 += 1

        k2 = k1 + 1

        if k2 >= LogMetallicity_bins.size:
            k2 = LogMetallicity_bins.size - 1

        if log_Metallicity >= LogMetallicity_bins[0] and log_Metallicity <= LogMetallicity_bins[-1]:
            d_k_local = log_Metallicity - LogMetallicity_bins[k1]
        else:
            d_k_local = 0.0

        delta_k = LogMetallicity_bins[k2] - LogMetallicity_bins[k1]

        if delta_k > 0:
            d_k_local = d_k_local / delta_k
        else:
            d_k_local = 0.0
    else:
        k1 = 0
        k2 = 0
        d_k_local = 0.0

    d_k = d_k_local

    # absolute band magnitudes, interpol_2d
    for j in range(GFM_STELLAR_PHOTOMETRICS_BANDS):
        mags[j] = (
            (1 - d_k) * (1 - d_i) * TableMags[j, k1, i1]
            + (1 - d_k) * d_i * TableMags[j, k1, i1 + 1]
            + d_k * (1 - d_i) * TableMags[j, k1 + 1, i1]
            + d_k * d_i * TableMags[j, k1 + 1, i1 + 1]
        )

    # account for mass
    mags -= 2.5 * np.log10(mass_in_msun)

    return


@jit(nopython=True)
def subfind_properties_2(
    candidates,
    Tail,
    Next,
    P,
    PS,
    SphP,
    StarP,
    BHP,
    atime,
    H_of_a,
    G,
    boxsize,
    LogMetallicity_bins,
    LogAgeInGyr_bins,
    TableMags,
    SofteningTable,
    GrNr,
):
    """Determine the properties of each subhalo."""
    vel_to_phys = np.double(1.0) / atime

    UnitLength_in_cm = np.double(3.085678e21)  # TODO: generalize (all three available in sP)
    PARSEC = np.double(3.085678e18)
    HubbleParam = np.double(0.6774)
    boxsize = np.double(boxsize)
    boxhalf = boxsize * 0.5

    # allocate SubGroup[]
    Subgroup = np.zeros(candidates.size, dtype=Subgroup_dtype)

    # allocations for property allocations
    subnr = 0
    totlen = 0

    v = np.zeros(3, dtype=np.float64)
    s = np.zeros(3, dtype=np.float64)
    pos = np.zeros(3, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64)
    spin = np.zeros(3, dtype=np.float64)
    cm = np.zeros(3, dtype=np.float64)
    dv = np.zeros(3, dtype=np.float64)
    dx = np.zeros(3, dtype=np.float64)

    lums = np.zeros(GFM_STELLAR_PHOTOMETRICS_BANDS, dtype=MyFloat)
    mags_local = np.zeros(GFM_STELLAR_PHOTOMETRICS_BANDS, dtype=MyFloat)
    mags = np.zeros(GFM_STELLAR_PHOTOMETRICS_BANDS, dtype=MyFloat)

    len_type = np.zeros(NTYPES, dtype=np.int32)
    len_type_loc = np.zeros(NTYPES, dtype=np.int32)

    mass_tab = np.zeros(NTYPES, dtype=np.float64)
    massinrad_tab = np.zeros(NTYPES, dtype=np.float64)
    massinhalfrad_tab = np.zeros(NTYPES, dtype=np.float64)
    massinmaxrad_tab = np.zeros(NTYPES, dtype=np.float64)
    halfmassradtype = np.zeros(NTYPES, dtype=np.float64)

    gasMassMetals = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    gasMassMetalsHalfRad = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    gasMassMetalsMaxRad = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    stellarMassMetals = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    stellarMassMetalsHalfRad = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    stellarMassMetalsMaxRad = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    gasMassMetalsSfr = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)
    gasMassMetalsSfrWeighted = np.zeros(GFM_N_CHEM_ELEMENTS, dtype=np.float64)

    rad_grid = np.zeros(GFM_STELLAR_PHOTOMETRICS_RADII, dtype=np.float64)
    lum_grid = np.zeros((GFM_STELLAR_PHOTOMETRICS_DIRECTIONS, GFM_STELLAR_PHOTOMETRICS_RADII - 1), dtype=MyFloat)
    mass_grid = np.zeros((GFM_STELLAR_PHOTOMETRICS_DIRECTIONS, GFM_STELLAR_PHOTOMETRICS_RADII - 1), dtype=np.float64)
    sblim_list_rr = np.zeros(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS, dtype=np.float64)
    sblim_list_mass = np.zeros(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS, dtype=np.float64)

    # RandomAngles: directly from arepo given TNG codebase init sequence
    StellarPhotometricsRandomAngles = np.zeros((GFM_STELLAR_PHOTOMETRICS_DIRECTIONS, 2), dtype=np.float32)

    StellarPhotometricsRandomAngles[:, 0] = [1.22469,2.5156,1.89036,1.12041,1.65378,1.36382,0.727205,1.67738,2.04625,
      1.66005,2.64311,1.24508,2.08531,1.01916,0.874651,1.13585,1.44381,1.77666,2.18747,1.87275,1.0667,2.28515,2.62462,
      1.66419,0.793788,1.63055,1.49722,1.61738,0.844168,0.376032,2.0015,1.59893,1.28786,1.75146,2.36355,1.9561,
      0.990663,2.41072,1.66573,1.49335,1.34853,2.48411,2.16556,3.08907,0.956437,1.88387,2.74096,2.28772,2.97042,
      0.0940844,1.52121,1.00048,1.35537,2.36729,2.17125,0.67752,0.692307,1.8943,0.407554,0.657783,0.426495,0.0747759,
      1.44043,2.00924,0.542286,2.68916,2.0342,2.62093,1.91206,0.970703,1.87628,1.14982,2.50939,0.848391,0.58381,
      1.56003,1.5534,1.54524,2.80665,2.79289,0.730284,0.390309,1.74462,2.20372,3.08843,1.45806,0.755776,0.302779,
      0.91858,1.35422,2.27953,1.89358,0.840655,2.64957,2.77734,3.01598,1.14885,2.81684,0.998777,1.63618]  # fmt: skip
    StellarPhotometricsRandomAngles[:, 1] = [1.68477,1.96426,3.13277,4.17738,2.57749,5.56388,4.91011,5.49633,4.68758,
      5.6185,2.02901,3.01843,1.88062,1.40816,0.903572,3.71739,1.67312,3.6844,0.171446,1.39946,0.604645,4.24954,2.09295,
      2.25746,1.30573,5.95015,1.85119,5.18499,0.387137,3.14278,1.15531,2.92628,2.90757,1.54623,3.7789,5.87517,3.47764,
      2.06749,2.15919,0.590433,0.553423,3.71127,2.67805,1.01639,5.14023,1.178,5.73582,1.37643,3.16482,3.33026,2.93006,
      2.96468,3.60805,2.04047,3.06509,3.06867,3.61902,4.00447,4.75386,4.42716,4.15663,0.664922,1.79173,0.763853,
      0.24633,6.02,4.96996,5.03009,3.81081,0.31155,3.20804,5.91206,5.40538,6.01432,3.05248,4.84851,5.54481,1.52201,
      2.06171,1.76919,0.933271,1.4099,6.25792,4.30342,0.28824,1.12354,5.03985,5.94228,1.98936,6.0363,1.56903,3.63016,
      3.988,0.161651,4.70989,2.396,1.74586,1.83267,0.232479,1.31449]  # fmt: skip

    # generate P_Pos, pure ndarray and handle CELL_CENTER_GRAVITY
    P_Pos = np.zeros((P.size, 3), dtype=np.float64)

    for i in range(P.size):
        if P[i]["Type"] == 0:
            P_Pos[i, :] = SphP[PS[i]["OldIndex"]]["Center"]
        else:
            P_Pos[i, :] = P[i]["Pos"]

    # loop over candidates
    for k in range(candidates.size):
        num = candidates[k]["bound_length"]
        totlen += num

        p = candidates[k]["head"]
        ud_index = np.zeros(num, dtype=np.int32)
        count = 0

        for _i in range(candidates[k]["len"]):
            if Tail[p] == candidates[k]["nsub"]:
                ud_index[count] = p
                count += 1
            p = Next[p]

        if count != num:
            raise Exception("Mismatch.")

        # subfind_determine_sub_halo_properties(); start

        # allocations and zeroing
        s *= 0
        v *= 0
        pos *= 0
        vel *= 0
        spin *= 0
        cm *= 0
        dv *= 0
        dx *= 0

        lums *= 0
        mags_local *= 0
        mags *= 0

        len_type *= 0
        len_type_loc *= 0

        mass_tab *= 0
        massinrad_tab *= 0
        massinhalfrad_tab *= 0
        massinmaxrad_tab *= 0
        halfmassradtype *= 0

        gasMassMetals *= 0
        gasMassMetalsHalfRad *= 0
        gasMassMetalsMaxRad *= 0
        stellarMassMetals *= 0
        stellarMassMetalsHalfRad *= 0
        stellarMassMetalsMaxRad *= 0
        gasMassMetalsSfr *= 0
        gasMassMetalsSfrWeighted *= 0

        rad_grid *= 0
        lum_grid *= 0
        mass_grid *= 0
        sblim_list_rr *= 0
        sblim_list_mass *= 0

        mass = np.double(0.0)
        massinrad = np.double(0.0)
        massinhalfrad = np.double(0.0)
        massinmaxrad = np.double(0.0)

        sfr = np.double(0.0)
        sfrinrad = np.double(0.0)
        sfrinhalfrad = np.double(0.0)
        sfrinmaxrad = np.double(0.0)
        gasMassSfr = np.double(0.0)

        bh_Mass = np.double(0.0)
        bh_Mdot = np.double(0.0)
        windMass = np.double(0.0)

        lx = np.double(0.0)
        ly = np.double(0.0)
        lz = np.double(0.0)
        disp = np.double(0.0)
        halfmassrad = np.double(0.0)
        max_stellar_rad = np.double(0.0)

        brightness_limit_rad = np.double(0.0)
        stellar_mass_in_phot_rad = np.double(0.0)

        gasMassMetallicity = np.double(0.0)
        gasMassMetallicityHalfRad = np.double(0.0)
        gasMassMetallicityMaxRad = np.double(0.0)
        stellarMassMetallicity = np.double(0.0)
        stellarMassMetallicityHalfRad = np.double(0.0)
        stellarMassMetallicityMaxRad = np.double(0.0)
        gasMassMetallicitySfr = np.double(0.0)
        gasMassMetallicitySfrWeighted = np.double(0.0)

        bfld_halo = np.double(0.0)
        bfld_disk = np.double(0.0)
        bfld_vol_halo = np.double(0.0)
        bfld_vol_disk = np.double(0.0)

        minindex = -1
        minpot = np.double(1.0e30)

        # start
        for i in range(num):
            p = ud_index[i]
            sphp = PS[p]["OldIndex"]
            auxid = P[p]["AuxDataID"]

            if PS[p]["Potential"] < minpot or minindex == -1:
                minpot = PS[p]["Potential"]
                minindex = p

            len_type[P[p]["Type"]] += 1

            if P[p]["Type"] == 0:
                assert sphp >= 0
                sfr += SphP[sphp]["Sfr"]

            if P[p]["Type"] == 5:
                bh_Mass += BHP[auxid]["BH_Mass"]
                bh_Mdot += BHP[auxid]["BH_Mdot"]

            if P[p]["Type"] == 4 and StarP[auxid]["BirthTime"] < 0:
                windMass += P[p]["Mass"]

        assert minindex != -1

        # pos[] now holds the position of the minimum potential, we'll take it as the center
        for j in range(3):
            if P[minindex]["Type"] == 0:
                pos[j] = SphP[PS[minindex]["OldIndex"]]["Center"][j]
            else:
                pos[j] = P[minindex]["Pos"][j]

        # determine the particle ID with the smallest binding energy
        minindex = -1
        minpot = np.double(1.0e30)

        for i in range(num):
            p = ud_index[i]
            if PS[p]["BindingEnergy"] < minpot or minindex == -1:
                minpot = PS[p]["BindingEnergy"]
                minindex = p

        assert minindex != -1

        mostboundid = P[minindex]["ID"]

        # print('subnr ', subnr, ' len', num, ' pos:',pos,' mostboundid',mostboundid,' lentype',len_type)

        # get bulk velocity and the center-of-mass, here we still take all particles
        for i in range(num):
            p = ud_index[i]

            for j in range(3):
                ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                s[j] += P[p]["Mass"] * ddxx
                v[j] += P[p]["Mass"] * P[p]["Vel"][j]

            mass += P[p]["Mass"]

            # mass by type
            ptype = P[p]["Type"]
            if ptype == 4 and StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                ptype = 0

            mass_tab[ptype] += P[p]["Mass"]

        s /= mass  # center of mass
        v /= mass
        vel[:] = vel_to_phys * v[:]

        for j in range(3):
            s[j] += pos[j]

            while s[j] < 0:
                s[j] += boxsize
            while s[j] >= boxsize:
                s[j] -= boxsize

            cm[j] = s[j]  # in comoving coordinates

        # see note below under len_type_loc
        rr_list = np.zeros(num, dtype=np.float64)
        mass_rr_sorted = np.zeros(num, dtype=np.float64)

        for i in range(num):
            p = ud_index[i]

            rr_tmp = np.double(0.0)
            disp_tmp = np.double(0.0)

            for j in range(3):
                ddxx = _NEAREST(P[p]["Pos"][j] - s[j], boxhalf, boxsize)
                dx[j] = atime * ddxx
                dv[j] = vel_to_phys * (P[p]["Vel"][j] - v[j])
                dv[j] += H_of_a * dx[j]

                disp_tmp += P[p]["Mass"] * dv[j] * dv[j]
                # for rotation curve computation, take minimum of potential as center
                ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                ddxx *= atime
                rr_tmp += ddxx * ddxx

            lx += P[p]["Mass"] * (dx[1] * dv[2] - dx[2] * dv[1])
            ly += P[p]["Mass"] * (dx[2] * dv[0] - dx[0] * dv[2])
            lz += P[p]["Mass"] * (dx[0] * dv[1] - dx[1] * dv[0])

            rr_tmp = np.sqrt(rr_tmp)
            rr_list[i] = rr_tmp
            mass_rr_sorted[i] = P[p]["Mass"]
            disp += disp_tmp

        spin[0] = lx / mass
        spin[1] = ly / mass
        spin[2] = lz / mass

        veldisp = np.sqrt(disp / (3 * mass))  # 3D -> 1D velocity dispersion

        # sort in increasing radial distance, calculate cumulative mass
        sort_inds = np.argsort(rr_list)
        rr_list = rr_list[sort_inds]
        mass_rr_sorted = mass_rr_sorted[sort_inds]

        for i in range(1, num):
            mass_rr_sorted[i] += mass_rr_sorted[i - 1]

        # vmax, vmaxrad, and half mass radius
        i = num - 1
        max = 0
        maxrad = 0

        while i >= 0:
            if i > 5 and mass_rr_sorted[i] > max * rr_list[i]:
                # outside of the very center, search for...
                max = mass_rr_sorted[i] / rr_list[i]
                maxrad = rr_list[i]

            if i < num - 1:
                # linearly interpolate to locate half mass radius
                if mass_rr_sorted[i] < 0.5 * mass and mass_rr_sorted[i + 1] >= 0.5 * mass:
                    halfmassrad = 0.5 * (rr_list[i] + rr_list[i + 1])

            i -= 1

        halfmassrad /= atime
        vmax = np.sqrt(G * max)
        vmaxrad = maxrad / atime

        # half mass radii for different types
        for i in range(num):
            p = ud_index[i]

            # count wind as gas for mass, but not for LenType since we use this to construct offset tables
            ptype = P[p]["Type"]
            if ptype == 4 and StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                ptype = 0

            len_type_loc[ptype] += 1

        for type in range(NTYPES):
            # NOTE: have size len_type_loc[type]+1 in subfind_properties.c, which is a harmless bug since the
            # mysort() is passed the correct len_type_loc[type]+0, but in our case picks up an extra zero entry
            rr_list = np.zeros(len_type_loc[type], dtype=np.float64)
            mass_rr_sorted = np.zeros(len_type_loc[type], dtype=np.float64)

            itmp = 0
            for i in range(num):
                p = ud_index[i]

                ptype = P[p]["Type"]
                if ptype == 4 and StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                    ptype = 0

                if ptype == type:
                    rr_tmp = 0.0
                    for j in range(3):
                        ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                        rr_tmp += ddxx * ddxx

                    rr_tmp = np.sqrt(rr_tmp)

                    mass_rr_sorted[itmp] = P[p]["Mass"]
                    rr_list[itmp] = rr_tmp
                    itmp += 1

            assert itmp == len_type_loc[type]

            # by type: sort in increasing radial distance, calculate cumulative mass
            sort_inds = np.argsort(rr_list)
            rr_list = rr_list[sort_inds]
            mass_rr_sorted = mass_rr_sorted[sort_inds]

            for i in range(1, len_type_loc[type]):
                mass_rr_sorted[i] += mass_rr_sorted[i - 1]

            # by type: half mass radii
            i = len_type_loc[type] - 1
            halfmass_type = 0.5 * mass_tab[type]

            while i >= 0:
                if mass_rr_sorted[i] < halfmass_type and mass_rr_sorted[i + 1] >= halfmass_type:
                    halfmassradtype[type] = 0.5 * (rr_list[i] + rr_list[i + 1])
                i -= 1

            # GFM_STELLAR_PHOTOMETRICS: maximum stellar radius
            if type == 4:
                if len_type_loc[type]:
                    max_stellar_rad = rr_list[len_type_loc[type] - 1]
                max_stellar_rad *= 1.01  # protect against round-off errors resulting in rxy > max_stellar_rad later

        # properties of 'central galaxies', defined in several ways as particles within some radius:
        # either (stellar half mass radius) or SUBFIND_GAL_RADIUS_FAC*(stellar half mass radius) or (radius of Vmax)
        for i in range(num):
            p = ud_index[i]
            sphp = PS[p]["OldIndex"]

            ptype = P[p]["Type"]
            if ptype == 4 and StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                ptype = 0

            # calculate particle radius
            rr_tmp = 0.0
            for j in range(3):
                ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                rr_tmp += ddxx * ddxx

            rr_tmp = np.sqrt(rr_tmp)

            # properties inside SUBFIND_GAL_RADIUS_FAC*(stellar half mass radius)
            if rr_tmp < SUBFIND_GAL_RADIUS_FAC * halfmassradtype[4]:
                massinrad += P[p]["Mass"]
                massinrad_tab[ptype] += P[p]["Mass"]

                if ptype == 0:
                    if P[p]["Type"] == 0:
                        sfrinrad += SphP[sphp]["Sfr"]

                        gasMassMetallicity += SphP[sphp]["Metallicity"] * P[p]["Mass"]
                        gasMassMetals += SphP[sphp]["MetalsFraction"] * P[p]["Mass"]

                    if P[p]["Type"] == 4:
                        gasMassMetallicity += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                        gasMassMetals += StarP[P[p]["AuxDataID"]]["MassMetals"]

                if ptype == 4:
                    stellarMassMetallicity += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                    stellarMassMetals += StarP[P[p]["AuxDataID"]]["MassMetals"]

            # properties inside (stellar half mass radius)
            if rr_tmp < 1.0 * halfmassradtype[4]:
                massinhalfrad += P[p]["Mass"]
                massinhalfrad_tab[ptype] += P[p]["Mass"]

                if ptype == 0:
                    if P[p]["Type"] == 0:
                        sfrinhalfrad += SphP[sphp]["Sfr"]
                        gasMassMetallicityHalfRad += SphP[sphp]["Metallicity"] * P[p]["Mass"]
                        gasMassMetalsHalfRad += SphP[sphp]["MetalsFraction"] * P[p]["Mass"]

                    if P[p]["Type"] == 4:
                        gasMassMetallicityHalfRad += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                        gasMassMetalsHalfRad += StarP[P[p]["AuxDataID"]]["MassMetals"]

                if ptype == 4:
                    stellarMassMetallicityHalfRad += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                    stellarMassMetalsHalfRad += StarP[P[p]["AuxDataID"]]["MassMetals"]

            # properties inside (radius of Vmax)
            if rr_tmp < 1.0 * vmaxrad:
                massinmaxrad += P[p]["Mass"]
                massinmaxrad_tab[ptype] += P[p]["Mass"]

                if ptype == 0:
                    if P[p]["Type"] == 0:
                        sfrinmaxrad += SphP[sphp]["Sfr"]

                        gasMassMetallicityMaxRad += SphP[sphp]["Metallicity"] * P[p]["Mass"]
                        gasMassMetalsMaxRad += SphP[sphp]["MetalsFraction"] * P[p]["Mass"]

                    if P[p]["Type"] == 4:
                        gasMassMetallicityMaxRad += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                        gasMassMetalsMaxRad += StarP[P[p]["AuxDataID"]]["MassMetals"]

                if ptype == 4:
                    stellarMassMetallicityMaxRad += StarP[P[p]["AuxDataID"]]["Metallicity"] * P[p]["Mass"]
                    stellarMassMetalsMaxRad += StarP[P[p]["AuxDataID"]]["MassMetals"]

        # properties of star-forming gas
        for i in range(num):
            p = ud_index[i]
            sphp = PS[p]["OldIndex"]

            if P[p]["Type"] != 0:
                continue

            if SphP[sphp]["Sfr"] <= 0:
                continue

            gasMassSfr += P[p]["Mass"]

            gasMassMetallicitySfr += SphP[sphp]["Metallicity"] * P[p]["Mass"]
            gasMassMetalsSfr += SphP[sphp]["MetalsFraction"] * P[p]["Mass"]

            gasMassMetallicitySfrWeighted += SphP[sphp]["Metallicity"] * SphP[sphp]["Sfr"]
            gasMassMetalsSfrWeighted += SphP[sphp]["MetalsFraction"] * SphP[sphp]["Sfr"]

        # MHD
        for i in range(num):
            p = ud_index[i]
            sphp = PS[p]["OldIndex"]

            if P[p]["Type"] != 0:
                continue

            bfld2 = SphP[sphp]["B"][0] ** 2 + SphP[sphp]["B"][1] ** 2 + SphP[sphp]["B"][2] ** 2

            bfld_halo += bfld2 * SphP[sphp]["Volume"]
            bfld_vol_halo += SphP[sphp]["Volume"]

            # calculate particle radius
            rr_tmp = 0.0

            for j in range(3):
                ddxx = _NEAREST(P[p]["Pos"][j] - pos[j], boxhalf, boxsize)
                rr_tmp += ddxx * ddxx

            rr_tmp = np.sqrt(rr_tmp)

            if rr_tmp < SUBFIND_GAL_RADIUS_FAC * halfmassradtype[4]:
                bfld_disk += bfld2 * SphP[sphp]["Volume"]
                bfld_vol_disk += SphP[sphp]["Volume"]

        if bfld_vol_halo > 0.0:
            bfld_halo = np.sqrt(bfld_halo / bfld_vol_halo)
        if bfld_vol_disk > 0.0:
            bfld_disk = np.sqrt(bfld_disk / bfld_vol_disk)

        # GFM_STELLAR_PHOTOMETRICS
        for i in range(num):
            p = ud_index[i]

            if P[p]["Type"] != 4 or StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                continue

            mags_local *= 0
            assign_stellar_photometrics(
                p, P, StarP, mags_local, LogMetallicity_bins, LogAgeInGyr_bins, TableMags, atime
            )
            lums += np.power(10.0, -0.4 * mags_local)

        if np.sum(lums <= 0) == 0:
            mags = -2.5 * np.log10(lums)
        else:
            mags[:] = MAX_FLOAT_NUMBER

        # calculate surface brightness profile and find the maximum radius that is still above the threshold
        SofteningTypeOfPartType_4 = 1  # TODO: not generalized, true for L25n128_0000 and L35n2160TNG, and typically
        def_pt4_softening = SofteningTable[SofteningTypeOfPartType_4]  # get_default_softening_of_particletype(4)

        if mass_tab[4] > 0:
            factor1 = 5 * np.log10(3600 * 360 / 10 / 2 / np.pi)
            factor2 = 2.5 * np.log10(np.power(atime * UnitLength_in_cm / PARSEC / HubbleParam, 2))

            for irad in range(1, GFM_STELLAR_PHOTOMETRICS_RADII):
                if def_pt4_softening < max_stellar_rad:
                    rad_frac = (irad - 1) / (GFM_STELLAR_PHOTOMETRICS_RADII - 2)
                    rad_grid[irad] = def_pt4_softening * np.power(max_stellar_rad / def_pt4_softening, rad_frac)
                else:
                    rad_grid[irad] = max_stellar_rad

            for idir in range(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS):
                sintheta = np.sin(StellarPhotometricsRandomAngles[idir][0])
                costheta = np.cos(StellarPhotometricsRandomAngles[idir][0])
                sinphi = np.sin(StellarPhotometricsRandomAngles[idir][1])
                cosphi = np.cos(StellarPhotometricsRandomAngles[idir][1])

                for i in range(num):
                    p = ud_index[i]

                    if P[p]["Type"] != 4 or StarP[P[p]["AuxDataID"]]["BirthTime"] < 0:
                        continue

                    ddxx = _NEAREST(P[p]["Pos"][0] - pos[0], boxhalf, boxsize)
                    ddyy = _NEAREST(P[p]["Pos"][1] - pos[1], boxhalf, boxsize)
                    ddzz = _NEAREST(P[p]["Pos"][2] - pos[2], boxhalf, boxsize)

                    rotxy_0 = ddxx * cosphi + ddyy * sinphi
                    rotxy_1 = -1 * ddxx * costheta * sinphi + ddyy * costheta * cosphi + ddzz * sintheta

                    rxy = np.sqrt(rotxy_0 * rotxy_0 + rotxy_1 * rotxy_1)

                    for irad in range(GFM_STELLAR_PHOTOMETRICS_RADII - 1):
                        if (rxy == 0 and irad == 0) or (rxy > rad_grid[irad] and rxy <= rad_grid[irad + 1]):
                            mags_local *= 0
                            assign_stellar_photometrics(
                                p, P, StarP, mags_local, LogMetallicity_bins, LogAgeInGyr_bins, TableMags, atime
                            )
                            lum_grid[idir][irad] += np.power(10.0, -0.4 * mags_local[3])  # Magnitude_K
                            mass_grid[idir][irad] += P[p]["Mass"]
                            break

            for idir in range(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS):
                sblim_list_rr[idir] = rad_grid[1]

                irad = GFM_STELLAR_PHOTOMETRICS_RADII - 2
                while irad > 0:
                    # convert luminosity/pc^2 to mag/arcsec^2, including cosmological surface brightness dimming
                    if lum_grid[idir][irad] > 0:
                        tmp_sb = (
                            factor1
                            - 2.5 * np.log10(lum_grid[idir][irad])
                            + 2.5 * np.log10(np.pi * (rad_grid[irad + 1] ** 2 - rad_grid[irad] ** 2))
                            + factor2
                            - 2.5 * np.log10(atime**4)
                        )

                        if GFM_STELLAR_PHOTOMETRICS_K_LIMIT > tmp_sb:
                            sblim_list_rr[idir] = rad_grid[irad + 1]
                            break

                    irad -= 1

                while irad >= 0:
                    sblim_list_mass[idir] += mass_grid[idir][irad]
                    irad -= 1

            sort_inds = np.argsort(sblim_list_rr, kind="mergesort")  # stable, since many repeated rr values
            sblim_list_rr = sblim_list_rr[sort_inds]
            sblim_list_mass = sblim_list_mass[sort_inds]

            # take median over directions
            brightness_limit_rad = sblim_list_rr[int(np.floor(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS / 2))]
            stellar_mass_in_phot_rad = sblim_list_mass[int(np.floor(GFM_STELLAR_PHOTOMETRICS_DIRECTIONS / 2))]

        # fill save structure
        Subgroup[subnr]["Len"] = num
        Subgroup[subnr]["Mass"] = mass
        Subgroup[subnr]["MassInRad"] = massinrad
        Subgroup[subnr]["MassInHalfRad"] = massinhalfrad
        Subgroup[subnr]["MassInMaxRad"] = massinmaxrad

        Subgroup[subnr]["MassType"][:] = mass_tab[:]
        Subgroup[subnr]["LenType"][:] = len_type[:]

        Subgroup[subnr]["BfldHalo"] = bfld_halo * np.sqrt(4 * np.pi)
        Subgroup[subnr]["BfldDisk"] = bfld_disk * np.sqrt(4 * np.pi)

        Subgroup[subnr]["HalfmassRadType"][:] = halfmassradtype[:]
        Subgroup[subnr]["MassInRadType"][:] = massinrad_tab[:]
        Subgroup[subnr]["MassInHalfRadType"][:] = massinhalfrad_tab[:]
        Subgroup[subnr]["MassInMaxRadType"][:] = massinmaxrad_tab[:]

        Subgroup[subnr]["Pos"][:] = pos[:]
        Subgroup[subnr]["Vel"][:] = vel[:]
        Subgroup[subnr]["CM"][:] = cm[:]
        Subgroup[subnr]["Spin"][:] = spin[:]

        Subgroup[subnr]["IDMostbound"] = mostboundid
        Subgroup[subnr]["VelDisp"] = veldisp
        Subgroup[subnr]["Vmax"] = vmax
        Subgroup[subnr]["VmaxRad"] = vmaxrad
        Subgroup[subnr]["HalfmassRad"] = halfmassrad

        Subgroup[subnr]["SFR"] = sfr
        Subgroup[subnr]["SFRinRad"] = sfrinrad
        Subgroup[subnr]["SFRinHalfRad"] = sfrinhalfrad
        Subgroup[subnr]["SFRinMaxRad"] = sfrinmaxrad
        # Subgroup[subnr]['GasMassSFR']   = gasMassSfr

        if massinrad_tab[0] > 0:
            Subgroup[subnr]["GasMetallicity"] = gasMassMetallicity / massinrad_tab[0]
            Subgroup[subnr]["GasMetalFractions"][:] = gasMassMetals[:] / massinrad_tab[0]
        if massinhalfrad_tab[0] > 0:
            Subgroup[subnr]["GasMetallicityHalfRad"] = gasMassMetallicityHalfRad / massinhalfrad_tab[0]
            Subgroup[subnr]["GasMetalFractionsHalfRad"][:] = gasMassMetalsHalfRad[:] / massinhalfrad_tab[0]
        if massinmaxrad_tab[0] > 0:
            Subgroup[subnr]["GasMetallicityMaxRad"] = gasMassMetallicityMaxRad / massinmaxrad_tab[0]
            Subgroup[subnr]["GasMetalFractionsMaxRad"][:] = gasMassMetalsMaxRad[:] / massinmaxrad_tab[0]

        if massinrad_tab[4] > 0:
            Subgroup[subnr]["StarMetallicity"] = stellarMassMetallicity / massinrad_tab[4]
            Subgroup[subnr]["StarMetalFractions"][:] = stellarMassMetals[:] / massinrad_tab[4]
        if massinhalfrad_tab[4] > 0:
            Subgroup[subnr]["StarMetallicityHalfRad"] = stellarMassMetallicityHalfRad / massinhalfrad_tab[4]
            Subgroup[subnr]["StarMetalFractionsHalfRad"][:] = stellarMassMetalsHalfRad[:] / massinhalfrad_tab[4]
        if massinmaxrad_tab[4] > 0:
            Subgroup[subnr]["StarMetallicityMaxRad"] = stellarMassMetallicityMaxRad / massinmaxrad_tab[4]
            Subgroup[subnr]["StarMetalFractionsMaxRad"][:] = stellarMassMetalsMaxRad[:] / massinmaxrad_tab[4]

        if gasMassSfr > 0:
            Subgroup[subnr]["GasMetallicitySfr"] = gasMassMetallicitySfr / gasMassSfr
            Subgroup[subnr]["GasMetalFractionsSfr"][:] = gasMassMetalsSfr[:] / gasMassSfr

        if sfr > 0:
            Subgroup[subnr]["GasMetallicitySfrWeighted"] = gasMassMetallicitySfrWeighted / sfr
            Subgroup[subnr]["GasMetalFractionsSfrWeighted"][:] = gasMassMetalsSfrWeighted[:] / sfr

        Subgroup[subnr]["BHMass"] = bh_Mass
        Subgroup[subnr]["BHMdot"] = bh_Mdot
        Subgroup[subnr]["WindMass"] = windMass

        Subgroup[subnr]["StellarPhotometrics"][:] = mags[:]
        Subgroup[subnr]["StellarPhotometricsRad"] = brightness_limit_rad
        Subgroup[subnr]["StellarPhotometricsMassInRad"] = stellar_mass_in_phot_rad

        # subfind_determine_sub_halo_properties(); end

        Subgroup[subnr]["Parent"] = candidates[k]["parent"]
        Subgroup[subnr]["SubNr"] = subnr
        Subgroup[subnr]["GrNr"] = GrNr

        # let us now assign the subgroup number to member particles/cells
        for i in range(num):
            PS[ud_index[i]]["SubNr"] = subnr

        subnr += 1

    # note: for fuzz have PS[i].SubNr = TotNgroups + 1;
    # set binding energy of fuzz to zero, was overwritten with Hsml before; needed for proper snapshot sorting of fuzz
    for i in range(P.size):
        if PS[i]["SubNr"] > subnr:
            PS[i]["BindingEnergy"] = 0

    return Subgroup


def subfind_particle_order(P, PS):
    """Prepare a particle-level output order according to our new PS (subgroup) assignment."""
    # write PS.SubNr into P.AuxDataID and PS.BindingEnergy into P.Mass for convenience
    P["AuxDataID"] = PS["SubNr"]
    P["Mass"] = PS["BindingEnergy"]

    # process per type
    sort_inds = {}

    for i in range(NTYPES):
        # select
        w = np.where(P["Type"] == i)

        if len(w[0]) == 0:
            continue

        # subset
        loc_P = P[w]

        # sort kernel: fof_compare_aux_sort_GrNr = GrNr, SubNr, BindingEnergy, ID
        sort_loc = np.argsort(
            loc_P, order=["AuxDataID", "Mass", "ID"]
        )  # SubNr ascending, BindingEnergy ascending, ID ascending

        # cast to int32
        if sort_loc.dtype == np.int64 and sort_loc.max() < np.iinfo(np.int32).max:
            sort_loc = sort_loc.astype(np.int32)

        # save IDs in sorted order, i.e. shuffling the snapshot IDs into this order is the required permutation
        sort_inds["%d_ids" % i] = loc_P["ID"][sort_loc]

    tot_inds = np.sum(sort_inds[key].size for key in sort_inds.keys())
    assert tot_inds == P.size

    return sort_inds


def set_softenings(P, SphP, sP, snapLoaded=False):
    """Generate SofteningTable following grav_softening.c, and set P[].SofteningType values (based on sizes/masses)."""

    def get_softeningtype_for_hydro_cell(volume):
        radius = np.power(volume * 3.0 / (4.0 * np.pi), 1.0 / 3)
        soft = GasSoftFactor * radius

        types = np.zeros(volume.size, dtype=np.uint8)

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
        eps = np.zeros(mass.size, dtype=np.float64)
        w = np.where(mass <= sP.dmParticleMass)
        eps[w] = 2.8 * SofteningComoving[1]
        w = np.where(mass > sP.dmParticleMass)
        eps[w] = 2.8 * SofteningComoving[1] * np.power(mass[w] / sP.dmParticleMass, 1.0 / 3)

        types = np.zeros(mass.size, dtype=np.uint8)
        min_dln = np.zeros(mass.size, dtype=np.float64)
        min_dln.fill(np.finfo(np.float64).max)

        for i in range(1, NSOFTTYPES):  # MULTIPLE_NODE_SOFTENING & ADAPTIVE_HYDRO_SOFTENING
            if ForceSoftening[i] > 0:
                dln = np.abs(np.log(eps) - np.log(ForceSoftening[i]))

                w = np.where(dln < min_dln)
                types[w] = i
        return types

    SofteningTable = np.zeros(NSOFTTYPES + NSOFTTYPES_HYDRO, dtype=np.float64)  # current (comoving)

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
    if snapLoaded:
        print("Setting P[].SofteningType for snapshot load.")
        # INDIVIDUAL_GRAVITY_SOFTENING=32 for L35n2160TNG
        w = np.where(P["Type"] == sP.ptNum("bhs"))[0]
        assert len(w) == w.max() - w.min() + 1
        # note: P[w]['field'] = array assignment seems to silently fail... only works for min:max index range
        P[w.min() : w.max() + 1]["SofteningType"] = get_softening_type_from_mass(P[w]["Mass"])

        # ADAPTIVE_HYDRO_SOFTENING
        w = np.where(P["Type"] == sP.ptNum("gas"))[0]
        assert len(w) == w.max() - w.min() + 1
        P[w.min() : w.max() + 1]["SofteningType"] = get_softeningtype_for_hydro_cell(SphP["Volume"])

    return SofteningTable, ForceSoftening, P


def load_gfm_stellar_photometrics():
    """Load photometrics table for interpolation later."""
    from os.path import expanduser

    gfmPhotoPath = expanduser("~") + "/data/Arepo_GFM_Tables_TNG/Photometrics/stellar_photometrics.hdf5"
    photoBandsOrdered = ["U", "B", "V", "K", "g", "r", "i", "z"]

    gfm_photo = {}
    with h5py.File(gfmPhotoPath, "r") as f:
        for key in f:
            gfm_photo[key] = f[key][()]

    # construct pure ndarray
    shape = (len(photoBandsOrdered), gfm_photo["N_LogMetallicity"][0], gfm_photo["N_LogAgeInGyr"][0])

    TableMags = np.zeros(shape, dtype=gfm_photo["Magnitude_B"].dtype)

    for i, band in enumerate(photoBandsOrdered):
        TableMags[i, :, :] = gfm_photo["Magnitude_%s" % band]

    return gfm_photo["LogMetallicity_bins"], gfm_photo["LogAgeInGyr_bins"], TableMags


def run_subfind_snapshot(sP, GrNr):
    """Run complete Subfind algorithm on a single group, using the saved snaphshot itself."""
    atime = sP.snapshotHeader()["Time"]

    # load
    P, PS, SphP, StarP, BHP = load_snapshot_data(sP, GrNr=GrNr)  # testing

    SofteningTable, ForceSoftening, P = set_softenings(P, SphP, sP, snapLoaded=True)

    # execute subfind
    count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP = subfind(
        P, PS, SphP, StarP, BHP, atime, sP.units.H_of_a, sP.units.G, sP.boxSize, SofteningTable, ForceSoftening
    )

    # derive subhalo properties
    LogMetallicity_bins, LogAgeInGyr_bins, TableMags = load_gfm_stellar_photometrics()

    candidates = subfind_properties_1(candidates)
    Subgroup = subfind_properties_2(
        candidates,
        Tail,
        Next,
        P,
        PS,
        SphP,
        StarP,
        BHP,
        atime,
        sP.units.H_of_a,
        sP.units.G,
        sP.boxSize,
        LogMetallicity_bins,
        LogAgeInGyr_bins,
        TableMags,
        SofteningTable,
        GrNr,
    )

    # sort order for Fof members (by type)
    ParticleOrder = subfind_particle_order(P, PS)

    return Subgroup, ParticleOrder


def run_subfind_customfof0save_phase1(sP, GrNr=0):
    """Run complete Subfind algorithm on custom FOF0 save files, phase one (TNG50-1)."""
    atime = sP.snapshotHeader()["Time"]
    phase1_save_file = sP.derivPath + "fof0_save_phase1_%s_%d.hdf5" % (sP.simName, sP.snap)

    # load
    P, PS, SphP, StarP, BHP = load_custom_dump(sP, GrNr=GrNr, phase1=True)

    SofteningTable, ForceSoftening, P = set_softenings(P, SphP, sP)

    # execute subfind
    print("Now executing subfind...", flush=True)
    start_time = time.time()

    count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP = subfind(
        P, PS, SphP, StarP, BHP, atime, sP.units.H_of_a, sP.units.G, sP.boxSize, SofteningTable, ForceSoftening
    )

    print(
        "Found [%d] substructures (before unbinding: %d), in [%g sec], now determining properties..."
        % (nsubs, count_cand, time.time() - start_time)
    )
    start_time = time.time()

    candidates = subfind_properties_1(candidates)
    print("subfind_properties_1: %g sec" % (time.time() - start_time))

    # need to save: candidates, Tail, Next, PS['Potential'], PS['BindingEnergy']
    with h5py.File(phase1_save_file, "w") as f:
        f["Tail"] = Tail
        f["Next"] = Next
        f["Potential"] = PS["Potential"]
        f["BindingEnergy"] = PS["BindingEnergy"]

    with open(phase1_save_file.replace("hdf5", "bin"), "wb") as f:
        candidates.tofile(f)

    print("All data saved, terminating.")


def run_subfind_customfof0save_phase2(sP, GrNr=0):
    """Run complete Subfind algorithm on custom FOF0 save files, phase two (TNG50-1)."""
    # load
    print("Loading...")

    atime = sP.snapshotHeader()["Time"]
    final_save_file = sP.derivPath + "fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)
    phase1_save_file = final_save_file.replace("save_", "save_phase1_")

    P, PS, SphP, StarP, BHP = load_custom_dump(sP, GrNr=GrNr)

    SofteningTable, ForceSoftening, P = set_softenings(P, SphP, sP)

    # redo sort: order particles (P, PS) in the order of decreasing density
    sort_inds = np.argsort(PS["Density"])[::-1]  # descending
    P = P[sort_inds]
    PS = PS[sort_inds]

    # fuzz: set a default SubNr that is larger than reasonable group number
    PS["SubNr"] = sP.groupCatHeader()["Ngroups_Total"] + 1

    # load phase1 data
    with h5py.File(phase1_save_file, "r") as f:
        Tail = f["Tail"][()]
        Next = f["Next"][()]
        PS["Potential"][:] = f["Potential"][()]
        PS["BindingEnergy"][:] = f["BindingEnergy"][()]

    with open(phase1_save_file.replace("hdf5", "bin"), "rb") as f:
        candidates = np.fromfile(f, dtype=cand_dtype)

    # derive subhalo properties
    start_time = time.time()

    LogMetallicity_bins, LogAgeInGyr_bins, TableMags = load_gfm_stellar_photometrics()

    Subgroup = subfind_properties_2(
        candidates,
        Tail,
        Next,
        P,
        PS,
        SphP,
        StarP,
        BHP,
        atime,
        sP.units.H_of_a,
        sP.units.G,
        sP.boxSize,
        LogMetallicity_bins,
        LogAgeInGyr_bins,
        TableMags,
        SofteningTable,
        GrNr,
    )

    print("subfind_properties_2: %g sec" % (time.time() - start_time))
    start_time = time.time()

    # save Subgroup
    with h5py.File(final_save_file, "w") as f:
        for field in Subgroup_dtype.names:
            f["Subhalo" + field] = Subgroup[field]

    print("Saved [Subgroup] to [%s]." % final_save_file)

    # save sort order for Fof0 members (by type)
    ParticleOrder = subfind_particle_order(P, PS)

    print("subfind_particle_order: %g sec" % (time.time() - start_time))

    with h5py.File(final_save_file, "a") as f:
        for pt in ParticleOrder:
            f["sort_inds_%s" % pt] = ParticleOrder[pt]

        # need to modify the Group later with:
        f["Group_nsubs"] = candidates.size
        f["Group_Pos"] = Subgroup[0]["Pos"]

        # note: still need to calculate Subfind{Density,DMDensity,Hsml,VelDisp} (do after rearrangement)

    print("Saved [ParticleOrder] to [%s]." % final_save_file)


def verify_results(sP, GrNr=0):
    """Check results vs. actual group catalog for a test run where the subgroups of FOF0 are computed in AREPO."""
    # load group catalog
    fof0 = sP.groupCatSingle(haloID=GrNr)
    print("Actual Fof0 has [%d] subhalos." % fof0["GroupNsubs"])

    actual_lengths = sP.groupCat(fieldsSubhalos=["SubhaloLen"])[0 : fof0["GroupNsubs"]]

    # load our results file
    final_save_file = sP.derivPath + "fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)

    result = {}
    with h5py.File(final_save_file, "r") as f:
        for key in f:
            result[key] = f[key][()]

    # candidate subhalo lengths
    print("actual lengths: ", actual_lengths)
    print("recovered lengths: ", result["SubhaloLen"], "\n")
    # assert np.array_equal(actual_lengths,result['SubhaloLen'])

    # check all subhalo properties
    for i in range(result["Group_nsubs"]):
        failed = []

        sub = sP.groupCatSingle(subhaloID=i)
        for key in sub:
            if key in ["SubhaloBfldDisk", "SubhaloBfldHalo"]:
                continue
            if not np.allclose(sub[key], result[key][i]):
                failed.append(key)
                # print(sub[key])
                # print(result[key][i])
                # assert 0

        if len(failed) > 0:
            print("sub = %4d (length = %7d) failed allclose: [%d] fields" % (i, sub["SubhaloLen"], len(failed)))
        else:
            print("sub = %4d (length = %7d) passed all checks." % (i, sub["SubhaloLen"]))

    # check particle ordering
    for pt in [0, 1, 4, 5]:
        snap_ids = sP.snapshotSubset(pt, "ids", haloID=GrNr)

        # minor differences in BindingEnergy lead to slightly permuted particle ordering inside subhalos
        ind0, ind1 = match(snap_ids, result["sort_inds_%d_ids" % pt])
        assert ind0.size == ind1.size == snap_ids.size

        eb_rank_diff = ind0 - np.arange(ind0.size)
        print("[pt %d] maximal eB rank difference in particle order = %d" % (pt, np.abs(eb_rank_diff).max()))

        # check subhalo and fuzz assignment based on ordering is absolutely correct
        offset = 0
        for i in range(fof0["GroupNsubs"]):
            loc_snap_ids = snap_ids[offset : offset + result["SubhaloLenType"][i, pt]]
            loc_res_ids = result["sort_inds_%d_ids" % pt][offset : offset + result["SubhaloLenType"][i, pt]]

            loc_snap_ids = np.sort(loc_snap_ids)
            loc_res_ids = np.sort(loc_res_ids)

            if not np.array_equal(loc_snap_ids, loc_res_ids):
                print("[%4d] sub particle members mismatch with snap" % i)
            offset += result["SubhaloLenType"][i, pt]

        # check fuzz
        fuzz_snap_ids = snap_ids[offset:]
        fuzz_res_ids = result["sort_inds_%d_ids" % pt][offset:]

        fuzz_snap_ids = np.sort(fuzz_snap_ids)
        fuzz_res_ids = np.sort(fuzz_res_ids)

        if not np.array_equal(fuzz_snap_ids, fuzz_res_ids):
            print("halo fuzz mismatch with snap")

    print("done.")


def rewrite_groupcat(sP, GrNr=0):
    """Rewrite a group catalog which is missing FOF0 subhalos using the phase2 (final) results."""
    assert GrNr == 0  # no generalization
    assert "fof0test" in sP.arepoPath  # testing, only modify L35n2160TNG_fof0test/ files

    final_save_file = sP.derivPath + "fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)

    # load our new FoF0 subhalos
    subs = {}

    with h5py.File(final_save_file, "r") as f:
        # all subhalo fields
        for key in f:
            if "Subhalo" not in key:
                continue
            subs[key] = f[key][()]
            print(key)

        # Fof0 fields
        Group_nsubs = f["Group_nsubs"][()]
        Group_Pos = f["Group_Pos"][()]

    # verify first groupcat chunk has no subhalos and has FoF0
    with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r") as f:
        nChunks = f["Header"].attrs["NumFiles"]
        assert f["Header"].attrs["Ngroups_ThisFile"] == 1
        assert f["Header"].attrs["Nsubgroups_ThisFile"] == 0

    # update GroupFirstSub across all halos of all chunks
    for i in range(nChunks):
        with h5py.File(sP.gcPath(sP.snap, chunkNum=i), "r+") as f:
            if f["Header"].attrs["Ngroups_ThisFile"] > 0:
                gfs_loc = f["Group"]["GroupFirstSub"][()]
                w = np.where(gfs_loc >= 0)
                gfs_loc[w] += Group_nsubs
                f["Group"]["GroupFirstSub"][:] = gfs_loc

    # update first chunk
    print("Updating groupcat files...")

    with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r+") as f:
        # write all FoF0 subhalo fields
        for key in subs:
            if subs[key].dtype == np.float64:
                # all such float fields saved as float32
                subs[key] = subs[key].astype("float32")
            f["Subhalo"][key] = subs[key]

        # update FoF0: number of subhalos and position
        f["Group"]["GroupFirstSub"][0] = 0
        f["Group"]["GroupNsubs"][0] = Group_nsubs
        f["Group"]["GroupPos"][0, :] = Group_Pos

    # update headers of all chunks
    with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r+") as f:
        f["Header"].attrs["Nsubgroups_ThisFile"] = np.int32(Group_nsubs)
        Nsubs_Total_old = f["Header"].attrs["Nsubgroups_Total"]

    Nsubgroups_Total = Nsubs_Total_old + Group_nsubs

    for i in range(nChunks):
        with h5py.File(sP.gcPath(sP.snap, chunkNum=i), "r+") as f:
            f["Header"].attrs["Nsubgroups_Total"] = np.int32(Nsubgroups_Total)

    print("Done.")


@jit(nopython=True)
def _find_so_quantities(dists, mass, rhoBack, Deltas):
    """Helper."""
    cur_mass = 0.0
    cur_overdensity = 0.0

    R200 = np.zeros(len(Deltas), dtype=np.float32)
    M200 = np.zeros(len(Deltas), dtype=np.float32)

    for i in range(dists.size):
        cur_mass += mass[i]
        if dists[i] > 0:
            cur_overdensity = cur_mass / (4 * np.pi / 3.0 * dists[i] ** 3) / rhoBack

        for delta_ind in range(Deltas.size):
            if cur_overdensity > Deltas[delta_ind]:
                R200[delta_ind] = dists[i]
                M200[delta_ind] = cur_mass

    return R200, M200


def add_so_quantities(sP, GrNr=0):
    """FoF0 is missing SO quantities (Group_R_Crit200, etc). Need to derive now."""
    import gc

    from ..util.helper import pSplitRange, reportMemory

    assert GrNr == 0  # no generalization for chunkNum
    assert "fof0test" in sP.arepoPath  # only modify L35n2160TNG_fof0test/ files

    DeltaMean200 = 200.0
    DeltaCrit200 = 200.0 / sP.units.Omega_z
    DeltaCrit500 = 500.0 / sP.units.Omega_z

    x = sP.units.Omega_z - 1.0
    DeltaTopHat = (18 * np.pi**2 + 82 * x - 39 * x**2) / sP.units.Omega_z

    Deltas = np.array([DeltaMean200, DeltaTopHat, DeltaCrit200, DeltaCrit500])

    # load meta
    ptTypes = [0, 1, 4, 5]
    NumPart = sP.snapshotHeader()["NumPart"]
    NumPartTot = np.sum([NumPart[pt] for pt in ptTypes])

    fof = sP.groupCatSingle(haloID=GrNr)

    # global load pos and compute distances
    dists = np.zeros(NumPartTot, dtype="float32")
    offset = 0

    for pt in ptTypes:
        indRange = [0, NumPart[pt]]
        pSplitNum = 10 if pt in [0, 1, 4] else 1

        for i in range(pSplitNum):
            # local range, snapshotSubset inclusive on last index
            locRange = pSplitRange(indRange, pSplitNum, i)
            locRange[1] -= 1

            print("load pos: ", pt, i, locRange, reportMemory(), flush=True)

            # load
            pos_loc = sP.snapshotSubsetP(pt, "pos", indRange=locRange, float32=True)

            # distances
            dists[offset : offset + pos_loc.shape[0]] = sP.periodicDists(fof["GroupPos"], pos_loc)
            offset += pos_loc.shape[0]

            # hack: https://bugs.python.org/issue32759 (fixed only in python 3.8x)
            del pos_loc
            mp.heap.BufferWrapper._heap = mp.heap.Heap()
            gc.collect()

    # sort
    sort_inds = np.argsort(dists)

    # truncate unused entries to save memory
    num_save = int(4e9)  # first 3 billion closest only, plenty to capture local environment
    sort_inds = sort_inds[0:num_save]

    dists = dists[sort_inds]  # shuffle and truncate

    # global load mass
    mass = np.zeros(NumPartTot, dtype="float32")
    offset = 0

    for pt in ptTypes:
        print("load mass: ", pt, reportMemory(), flush=True)
        mass[offset : offset + NumPart[pt]] = sP.snapshotSubsetP(pt, "mass")
        offset += NumPart[pt]

        # hack: https://bugs.python.org/issue32759 (fixed only in python 3.8x)
        mp.heap.BufferWrapper._heap = mp.heap.Heap()
        gc.collect()

    mass = mass[sort_inds]  # shuffle and truncate

    # overdensity
    R200, M200 = _find_so_quantities(dists, mass, sP.units.rhoBack, Deltas)

    print("R200: ", R200)
    print("M200: ", M200)

    if 1:
        with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r+") as f:
            f["Group"]["Group_R_Mean200"][GrNr] = R200[0]
            f["Group"]["Group_R_TopHat200"][GrNr] = R200[1]
            f["Group"]["Group_R_Crit200"][GrNr] = R200[2]
            f["Group"]["Group_R_Crit500"][GrNr] = R200[3]

            f["Group"]["Group_M_Mean200"][GrNr] = M200[0]
            f["Group"]["Group_M_TopHat200"][GrNr] = M200[1]
            f["Group"]["Group_M_Crit200"][GrNr] = M200[2]
            f["Group"]["Group_M_Crit500"][GrNr] = M200[3]

    print("Written.")


def rewrite_snapshot(sP, GrNr=0):
    """Rewrite a snapshot which is missing FOF0 subhalos using the phase2 (final) results."""
    assert GrNr == 0  # no generalization
    assert "fof0test" in sP.arepoPath  # testing, only modify L35n2160TNG_fof0test/ files

    ptTypes = [0, 1, 4, 5]

    final_save_file = sP.derivPath + "fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)

    # how many snapshot chunk files are covered by Fof0 (by type)?
    fof0 = sP.groupCatSingle(haloID=GrNr)

    if 1:
        nchunks_type = np.zeros(NTYPES, dtype=np.int32) - 1
        cumlen_type = np.zeros(NTYPES, dtype=np.int64)

        i = 0

        while np.any(nchunks_type[ptTypes] == -1):
            with h5py.File(sP.snapPath(sP.snap, chunkNum=i), "r") as f:
                for pt in ptTypes:
                    if nchunks_type[pt] >= 0:
                        continue

                    loc_len = f["Header"].attrs["NumPart_ThisFile"][pt]
                    cumlen_type[pt] += loc_len

                    # print('[%3d] pt = %d, len = %7d, cumlen [%7d of %8d]' % \
                    # (i,pt,loc_len,cumlen_type[pt],fof0['GroupLenType'][pt]))
                    if cumlen_type[pt] >= fof0["GroupLenType"][pt]:
                        nchunks_type[pt] = i + 1

            i += 1

        for pt in ptTypes:
            print("For partType = %d have to rewrite through chunk [%d]." % (pt, nchunks_type[pt]))

    # load particle sort indices to shuffle FoF0 member particles into proper Subfind order
    particle_sort = {}

    with h5py.File(final_save_file, "r") as f:
        for key in f:
            if "sort_inds" not in key:
                continue
            particle_sort[key] = f[key][()]

    # verify against FoF0 size
    for pt in ptTypes:
        assert particle_sort["sort_inds_%d_ids" % pt].size == fof0["GroupLenType"][pt]

    # derive shuffle order
    for pt in ptTypes:
        print("finding shuffle: ", pt)
        snap_ids = sP.snapshotSubset(pt, "id", haloID=GrNr)
        particle_sort["sort_inds_%d" % pt], _ = match(snap_ids, particle_sort["sort_inds_%d_ids" % pt])
        assert particle_sort["sort_inds_%d" % pt].size == snap_ids.size

    # list of all fields we will need to rewrite
    fields = {}

    with h5py.File(sP.snapPath(sP.snap, chunkNum=0), "r") as f:
        for pt in ptTypes:
            fields[pt] = []
            for key in f["PartType%d" % pt]:
                fields[pt].append(key)
            print("PartType = %d have [%d] fields." % (pt, len(fields[pt])))

    # rewrite loop
    print("Writing...")

    for pt in ptTypes:
        for field in fields[pt]:
            print(pt, field)

            # load using normal routines
            data = sP.snapshotSubset(pt, field, haloID=GrNr)

            # reshuffle
            data = data[particle_sort["sort_inds_%d" % pt]]

            # loop over chunks to rewrite
            offset = 0

            for i in range(nchunks_type[pt]):
                file = sP.snapPath(sP.snap, chunkNum=i)

                num_remaining = fof0["GroupLenType"][pt] - offset

                with h5py.File(file, "r+") as f:
                    write_size = f["Header"].attrs["NumPart_ThisFile"][pt]

                    if write_size > num_remaining:
                        write_size = num_remaining

                    # print(' [%s] write [0-%d] from data[%d-%d] left = %d' % \
                    #    (file,write_size,offset,offset+write_size,num_remaining))

                    # stamp
                    f["PartType%d" % pt][field][0:write_size] = data[offset : offset + write_size]

                offset += write_size

            # wrote full length?
            assert offset == fof0["GroupLenType"][pt]

    print("Done.")


def rewrite_particle_level_cat(sP, filename, partType):
    """Shuffle the FoF0 segment of a dataset from a full snapshot into the new order."""
    GrNr = 0
    final_save_file = sP.derivPath + "fof0/fof0_save_%s_%d.hdf5" % (sP.simName, sP.snap)

    ptNum = sP.ptNum(partType)
    fof0_size = sP.groupCatSingle(haloID=GrNr)["GroupLenType"][ptNum]

    # sanity check: such a filename probably has sP.snap encoded in it, otherwise remove this check
    assert "_%03d" % sP.snap in filename

    # load particle sort indices to shuffle FoF0 member particles into proper Subfind order
    print("Loading sort indices...")
    with h5py.File(final_save_file, "r") as f:
        if "sort_inds_%d" % ptNum in f:
            indsDone = True
            sort_inds = f["sort_inds_%d" % ptNum][()]
        else:
            indsDone = False
            particle_sort = f["sort_inds_%d_ids" % ptNum][()]

    if not indsDone:
        assert particle_sort.size == fof0_size

        # derive shuffle order
        print("Loading snap IDs...")
        snap_ids = sP.snapshotSubset(partType, "id", haloID=GrNr)

        print("Finding shuffle...")
        sort_inds, _ = match(snap_ids, particle_sort)
        assert sort_inds.size == snap_ids.size

        # save for future uses
        with h5py.File(final_save_file, "r+") as f:
            f["sort_inds_%d" % ptNum] = sort_inds
        print("Saved sort inds for future uses.")

    # load catalog file
    print("Loading and rewriting datafile...")
    with h5py.File(filename, "r+") as f:
        dsetNames = list(f.keys())

        # assert len(dsetNames) == 1 # otherwise which?
        # dset = dsetNames[0]

        for dset in dsetNames:
            if dset == "Header":
                continue
            print(dset)

            assert f[dset].size == sP.snapshotHeader()["NumPart"][ptNum]
            assert f[dset].ndim == 1  # otherwise generalize below

            # load
            fof0_data = f[dset][0:fof0_size]

            # re-shuffle
            fof0_data = fof0_data[sort_inds]

            # write
            f[dset][0:fof0_size] = fof0_data

    print("Done.")


def compare_subhalos_all_quantities(snap_start=67):
    """Plot diagnostic histograms."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    nBins = 50

    sPs = []
    sPs.append(simParams(res=2160, run="tng", snap=snap_start + 0))
    sPs.append(simParams(res=2160, run="tng", snap=snap_start + 1))
    sPs.append(simParams(res=2160, run="tng", snap=snap_start + 2))

    # sPs.append( simParams(res=512, run='tng', snap=3, variant='5011') )
    # sPs.append( 'fof0save' ) # load final save file of this run itself

    # start pdf book
    pdf = PdfPages("compare_subhalos_%s_%d.pdf" % (sPs[0].simName, snap_start))

    # get list of subhalo properties
    with h5py.File(sPs[0].gcPath(sPs[0].snap, chunkNum=0), "r") as f:
        fields = list(f["Subhalo"].keys())

    # get GroupNsubs[0] for each sim
    fof0len = []
    for sP in sPs:
        if str(sP) == "fof0save":
            final_save_file = sPs[0].derivPath + "fof0_save_%s_%d.hdf5" % (sPs[0].simName, sPs[0].snap)
            with h5py.File(final_save_file, "r") as f:
                fof0len.append(f["Group_nsubs"][()])
        else:
            fof0len.append(sP.groupCatSingle(haloID=0)["GroupNsubs"])

    for field in fields:
        # start plot
        print(field)

        fig, ax = plt.subplots()

        ax.set_xlabel(field + " [log]")
        ax.set_ylabel("log N")
        # ax.set_yscale('log')

        # load and histogram
        for i, sP in enumerate(sPs):
            if str(sP) == "fof0save":
                label = "fof0save" + " snap=%d" % sPs[0].snap
                with h5py.File(final_save_file, "r") as f:
                    vals = f[field][()]
            else:
                label = sP.simName + " snap=%d" % sP.snap
                with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r") as f:
                    if field in f["Subhalo"]:
                        vals = f["Subhalo"][field][()]
                    else:
                        assert "Bfld" in field  # only occasionally missing fields
                        vals = np.zeros(fof0len[i], dtype="float32")
                        vals.fill(np.nan)

            assert vals.shape[0] == fof0len[i]

            vals = vals.ravel()  # 1D for all multi-D

            if field not in ["SubhaloCM", "SubhaloGrNr", "SubhaloIDMostbound"]:
                vals = np.log10(vals)
            vals = vals[np.isfinite(vals)]

            ax.hist(vals, bins=nBins, alpha=0.6, label=label)

        # finish plot
        ax.legend(loc="best")
        pdf.savefig()
        plt.close(fig)

    # by type
    for field in fields:
        if field[-4:] != "Type":
            continue
        print(field)

        # load
        data = []
        labels = []

        for sP in sPs:
            if str(sP) == "fof0save":
                label = "fof0save" + " snap=%d" % sPs[0].snap
                with h5py.File(final_save_file, "r") as f:
                    vals = f[field][()]
            else:
                label = sP.simName + " snap=%d" % sP.snap
                with h5py.File(sP.gcPath(sP.snap, chunkNum=0), "r") as f:
                    vals = f["Subhalo"][field][()]

            data.append(vals)
            labels.append(label)

        # separate plot for each type
        for pt in [0, 1, 4, 5]:
            fig, ax = plt.subplots()

            ax.set_xlabel(field + " [Type=%d] [log]" % pt)
            ax.set_ylabel("log N")

            for i in range(len(sPs)):
                vals = np.squeeze(data[i][:, pt])
                w = np.where(vals == 0)
                print(field, pt, " number of zeros: ", len(w[0]), " of ", vals.size)
                vals = np.log10(vals)
                vals = vals[np.isfinite(vals)]

                ax.hist(vals, bins=nBins, alpha=0.6, label=labels[i])

            # finish plot
            ax.legend(loc="best")
            pdf.savefig()
            plt.close(fig)

    # finish
    pdf.close()


def run_subfind(snap):
    """Main driver."""
    # sP = simParams(res=128,run='tng',snap=snap,variant='0000') # note: collective vs. serial algorithm
    # sP = simParams(res=512,run='tng',snap=snap,variant='0000')

    # L35n2160TNG started skipping fof0 subfind at snapshot 69 and onwards
    sP = simParams(res=2160, run="tng", snap=snap)

    run_subfind_customfof0save_phase1(sP, GrNr=0)
    run_subfind_customfof0save_phase2(sP, GrNr=0)

    # rewrite
    rewrite_groupcat(sP, GrNr=0)
    add_so_quantities(sP, GrNr=0)
    rewrite_snapshot(sP, GrNr=0)

    compare_subhalos_all_quantities(snap_start=snap)
    verify_results(sP, GrNr=0)


def benchmark():
    """Benchmark."""
    sP = simParams(res=256, run="tng", snap=4, variant="0000")
    # sP = simParams(res=2160,run='tng',snap=69)

    nLoops = 4
    GrNr = 0

    # load
    P, PS, SphP, StarP, BHP = load_custom_dump(sP, GrNr=GrNr)
    atime = sP.snapshotHeader()["Time"]

    SofteningTable, ForceSoftening, P = set_softenings(P, SphP, sP)
    LogMetallicity_bins, LogAgeInGyr_bins, TableMags = load_gfm_stellar_photometrics()

    # run Subfind a few times
    print("Running [%d] Subfind loops now..." % nLoops)
    start_time = time.time()

    for i in range(nLoops):
        print(i)
        count_cand, nsubs, candidates, Tail, Next, P, PS, SphP, StarP, BHP = subfind(
            P, PS, SphP, StarP, BHP, atime, sP.units.H_of_a, sP.units.G, sP.boxSize, SofteningTable, ForceSoftening
        )
        candidates = subfind_properties_1(candidates)
        Subgroup = subfind_properties_2(
            candidates,
            Tail,
            Next,
            P,
            PS,
            SphP,
            StarP,
            BHP,
            atime,
            sP.units.H_of_a,
            sP.units.G,
            sP.boxSize,
            LogMetallicity_bins,
            LogAgeInGyr_bins,
            TableMags,
            SofteningTable,
            GrNr,
        )

    print(
        "Ran Subfind [%d] times on [%s], took [%g] sec on avg (Subgroup size = %d)."
        % (nLoops, sP.simName, (time.time() - start_time) / nLoops, len(Subgroup))
    )
