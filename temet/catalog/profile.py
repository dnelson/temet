"""
Cosmological simulations - auxiliary catalogs (of radial profiles).
"""

import healpy
import numpy as np

from ..catalog.common import pSplitBounds, userCustomFields
from ..util.helper import binned_statistic_weighted, numPartToChunkLoadSize
from ..util.rotation import momentOfInertiaTensor, rotateCoordinateArray, rotationMatricesFromInertiaTensor
from ..util.treeSearch import buildFullTree, calcHsml, calcParticleIndices, calcQuantReduction


def healpix_shells_points(nRad, Nside, radMin=0.0, radMax=5.0):
    """Return a set of spherical shell sample points as defined by healpix."""
    # generate one set sample positions on unit sphere
    nProj = healpy.nside2npix(Nside)
    projVecs = np.array(healpy.pix2vec(Nside, range(nProj), nest=True)).T  # [nProj,3]

    # broadcast into nRad shells, radial coordinates in units of rvir
    samplePoints = np.repeat(projVecs[:, np.newaxis, :], nRad, axis=1)  # [nProj,nRad,3]

    radPts = np.linspace(radMin, radMax, nRad, endpoint=True)

    # shift shells to radial distances
    pts = samplePoints * radPts[np.newaxis, :, np.newaxis]

    pts = np.reshape(pts, (nProj * nRad, 3))  # [N,3] for tree/search operations

    # bin sizes
    radBinSize = radPts[1] - radPts[0]  # r/rvir
    # thetaBinSize = np.sqrt(180**2 / (3*np.pi*Nside**2)) # deg
    # thetaBinSizeRvir = np.tan(np.deg2rad(thetaBinSize)) # angular spacing @ rvir, in units of rvir

    return pts, nProj, radPts, radBinSize


def subhaloRadialProfile(
    sP,
    pSplit,
    ptType,
    ptProperty,
    op,
    scope,
    weighting=None,
    proj2D=None,
    ptRestrictions=None,
    subhaloIDsTodo=None,
    radMin=-1.0,
    radMax=3.7,
    radBinsLog=True,
    radNumBins=100,
    radRvirUnits=False,
    Nside=None,
    Nngb=None,
    minHaloMass=None,
    minStellarMass=None,
    cenSatSelect="cen",
):
    """Compute subhalo radial profiles (e.g. total/sum, weighted mean) of a particle/cell property (e.g. mass).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle type, e.g. 'gas', 'stars', 'dm', 'bhs'.
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'min', 'max', 'mean', 'median', and 'sum' for example.
      scope (str): which particles/cells are included for each profile.
        If ``scope=='global'``, then all snapshot particles are used, and we do the accumulation in a
        chunked snapshot load. Self/other halo terms are decided based on subfind membership, unless
        scope=='global_fof', then on group membership.
        If ``scope=='fof'`` or 'subfind' then restrict to FoF/subhalo particles only, respectively, and do a
        restricted load according to pSplit. In this case, only the self-halo term is computed.
        If ``scope=='subfind_global'`` then only the other-halo term is computed, approximating the particle
        distribution using an already computed subhalo-based accumlation auxCat, e.g. 'Subhalo_Mass_OVI'.
        If ``scope=='global_spatial'``, then use pSplit to decompose the work via a spatial subset of the box,
        such that we are guaranteed to have access to all the particles/cells within radMax of all halos
        being processed, to enable more complex global scope operations.
        If ``scope=='global_tngcluster'``, then use the file structure of the reconstructed TNG-Cluster
        simulation to achieve global scope profiles.
      weighting (str): if not None, then use this additional particle/cell property as the weight.
      proj2D (list or None): if not None, do 3D profiles, otherwise 2-tuple specifying (i) integer coordinate axis in
        [0,1,2] to project along or 'face-on' or 'edge-on', and (ii) depth in code units (None for full box).
      ptRestrictions (dict): apply cuts to which particles/cells are included. Each key,val pair in the dict
        specifies a particle/cell field string in key, and a [min,max] pair in value, where e.g. np.inf can be
        used as a maximum to enforce a minimum threshold only.
      subhaloIDsTodo (list): if not None, then process this explicit list of subhalos.
      radMin (int): minimum radius for profiles, should be [log] if radBinsLog == True, else [linear].
        should be [code units] if radRvirUnits == False, else [dimensionless rvir units].
      radMax (int): maximum radius for profiles, as above.
      radBinsLog (bool): if True, input radMin and radMax are in log, otherwise linear.
      radNumBins (int): number of radial bins for profiles.
      radRvirUnits (bool): if True, radMin and radMax are in units of rvir of each halo.
        Note that in this case binning is always linear, whereas if False, binning is always logarithmic.
      Nside (int or None): if not None, should be a healpix parameter (2,4,8,etc). In this case, we do not
        compute a spherically averaged radial profile per halo, but instead a spherical healpix sampled set of
        shells/radial profiles, where the quantity sampling at each point uses a SPH-kernel with ``Nngb``.
      Nngb (int or None): must be specified, if and only if ``Nside`` is also specified. The neighbor number.
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 2d array, radial profile for each subhalo.
      - **attrs** (dict): metadata.

    Note:
        For scopes `global` and `global_fof`, four profiles are saved: [all, self-halo, other-halo, diffuse],
        otherwise only a single 'all' profile is computed.
    """
    assert op in ["sum", "mean", "median", "min", "max", "count", "kernel_mean", np.std]  # todo: or is a lambda
    assert scope in ["global", "global_fof", "global_spatial", "global_tngcluster", "subfind", "fof", "subfind_global"]

    if scope in ["global", "global_fof"]:
        assert op in ["sum"]  # not generalized to non-accumulation stats w/ chunk loading

    if scope == "global_tngcluster":
        # requires one pSplit request per original zoom halo
        assert sP.name == "TNG-Cluster"
        assert pSplit is not None and pSplit[1] == 352, "TNG-Cluster global requires pSplit[1] == 352."

    if Nside is not None:
        assert Nngb is not None
        assert op in ["kernel_mean"]  # can generalize, calcQuantReduction() accepts other operations
        assert weighting is None  # otherwise need to add support in calcQuantReduction()
        assert ptRestrictions is None  # otherwise need to rearrange order below
        assert proj2D is None
        assert cenSatSelect == "cen"  # otherwise generalize r/rvir scaling of sample points

    useTree = True if scope == "global_spatial" else False  # can be generalized, or made a parameter

    # determine ptRestriction
    if ptType == "stars":
        if ptRestrictions is None:
            ptRestrictions = {}
        ptRestrictions["GFM_StellarFormationTime"] = ["gt", 0.0]  # real stars

    # config
    ptLoadType = sP.ptNum(ptType)

    radDesc = "code units" if not radRvirUnits else "rvir units"
    radDesc = "log " + radDesc if radBinsLog else "linear " + radDesc
    desc = "Quantity [%s] (%s) radial profile" % (ptProperty, op) if op != "count" else "[Count] radial profile"
    desc += " for [%s] from [%.1f - %.1f] %s, with [%d] bins." % (ptType, radMin, radMax, radDesc, radNumBins)

    if not radRvirUnits:
        desc += " Note: first/inner-most bin is extra, and extends from r=0 to r=%.1f." % radMin
    if Nside is not None:
        desc = "Quantity [%s,%s] spherical healpix sampling, [%.1f - %.1f] r/rvir with [%d] bins. " % (
            ptType,
            ptProperty,
            radMin,
            radMax,
            radNumBins,
        )
        desc += "Nside = [%d] Nngb = [%d]." % (Nside, Nngb)
    if ptRestrictions is not None:
        desc += " (restriction = %s)." % ",".join(list(ptRestrictions))

    if weighting is not None:
        desc += " (weighting = %s)." % weighting

        assert op not in ["sum"]  # meaningless
        assert op in ["mean", np.std]  # currently only supported

    if proj2D is not None:
        assert len(proj2D) == 2
        assert scope != "global_spatial"  # otherwise generalize i0g,i1g indices below
        proj2Daxis, proj2Ddepth = proj2D

        if proj2Daxis == 0:
            p_inds = [1, 2, 3]  # seems wrong (unused), should be e.g. [1,2,0]
        if proj2Daxis == 1:
            p_inds = [0, 2, 1]
        if proj2Daxis == 2:
            p_inds = [0, 1, 2]
        if isinstance(proj2Daxis, str):
            p_inds = [0, 1, 2]  # by convention, after rotMatrix is applied, index 2 is the projection direction

        proj2D_halfDepth = proj2Ddepth / 2 if proj2Ddepth is not None else None  # code units

        depthStr = "fullbox" if proj2Ddepth is None else "%.1f" % proj2Ddepth
        desc += " (2D projection axis = %s, depth = %s)." % (proj2Daxis, depthStr)

    desc += " (scope = %s). " % scope
    if subhaloIDsTodo is None:
        select = "Subhalos [%s]." % cenSatSelect
        if minStellarMass is not None:
            select += " (Only with stellar mass >= %.2f)" % minStellarMass
        if minHaloMass is not None:
            select += " (Only with halo mass >= %s)" % minHaloMass
    else:
        nSubsSelected = len(subhaloIDsTodo)
        select = "Subhalos [%d] specifically input." % nSubsSelected

    # load group information and make selection
    gc = sP.groupCat(fieldsSubhalos=["SubhaloPos", "SubhaloLenType"])
    gc["header"] = sP.groupCatHeader()

    # no explicit ID list input, choose subhalos to process now
    pSplitSpatial = None
    load_inds = None

    if scope == "global_spatial":
        # for spatial subdivision, disable the normal subhalo-based subdivision
        pSplitSpatial = pSplit
        pSplit = None
        indRange = None

    if subhaloIDsTodo is None:
        subhaloIDsTodo, indRange_scoped, nSubsSelected = pSplitBounds(
            sP,
            pSplit,
            partType="dm",
            minStellarMass=minStellarMass,
            minHaloMass=minHaloMass,
            cenSatSelect=cenSatSelect,
            equalSubSplit=False,
        )
    else:
        assert pSplit is None  # otherwise check, don't think we actually subdivide the work
        indRange_scoped = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)

    nChunks = 1  # chunk load disabled by default

    # need for scope=='subfind' and scope=='global' (for self/other halo terms)
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]

    if scope in ["fof", "global_fof"]:
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        # for scope=='global_fof' take all FoF particles for the respective halo terms
        GroupLenType = sP.groupCat(fieldsHalos=["GroupLenType"])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=["SubhaloGrNr"])

        gc["SubhaloLenType"] = GroupLenType[SubhaloGrNr, :]
        gc["SubhaloOffsetType"] = GroupOffsetType[SubhaloGrNr, :]

    if scope in ["global", "global_fof"]:
        # enable chunk loading
        h = sP.snapshotHeader()
        nChunks = numPartToChunkLoadSize(h["NumPart"][sP.ptNum(ptType)])
        chunkSize = int(h["NumPart"][sP.ptNum(ptType)] / nChunks)

        # default particle load range is set inside chunkLoadLoop
        print(" Total # Snapshot Load Chunks: " + str(nChunks) + " (" + str(chunkSize) + " particles per load)")
        indRange = None
        prevMaskInd = 0

    if scope == "global_tngcluster":
        # override indRange with first particle range
        from ..load.snapshot import _global_indices_zoomorig

        orig_halo_id = sP.subhalos("SubhaloOrigHaloID")
        orig_halo_id_uniq = np.unique(orig_halo_id)
        origZoomID = orig_halo_id_uniq[pSplit[0]]

        indRange, indRange2 = _global_indices_zoomorig(sP, ptType, origZoomID=origZoomID)

    if scope in ["subfind", "fof"]:
        # non-global load, use restricted index range covering our input/selected/pSplit subhaloIDsTodo
        indRange = indRange_scoped[ptType]

    # determine radial binning
    if Nside is None:
        # normal radial profiles
        radMin_log = radMin if radBinsLog else np.log10(radMin)
        radMin_linear = 10.0**radMin_log
        radMax_log = radMax if radBinsLog else np.log10(radMax)
        radMax_linear = 10.0**radMax_log

        if radRvirUnits:
            # radMin, radMax in rvir units
            # bin edges (always linear), including inner and outer boundary
            if radBinsLog:
                rad_bin_edges = 10.0 ** np.linspace(radMin_log, radMax_log, radNumBins + 1)
            else:
                rad_bin_edges = np.linspace(radMin_linear, radMax_linear, radNumBins + 1)

            # load virial radii (code units)
            gc["Subhalo_Rvir"] = sP.subhalos("rhalo_200_code")

            radMaxCode = gc["Subhalo_Rvir"][subhaloIDsTodo].max() * radMax_linear
        else:
            # radMin, radMax in code units
            # bin edges (always linear), including inner and outer boundary
            if radBinsLog:
                rad_bin_edges = np.linspace(radMin_log, radMax_log, radNumBins + 1)
                rad_bin_edges = np.hstack([radMin_log - 1.0, rad_bin_edges])  # include an inner bin complete to r=0
                rad_bin_edges = 10.0**rad_bin_edges
            else:
                rad_bin_edges = np.linspace(radMin_linear, radMax_linear, radNumBins + 1)
                rad_bin_edges = np.hstack([0.0, rad_bin_edges])  # include an inner bin complete to r=0

            rad_bins_code = 0.5 * (rad_bin_edges[1:] + rad_bin_edges[:-1])  # bin centers
            rad_bins_pkpc = sP.units.codeLengthToKpc(rad_bins_code)

            radMaxCode = radMax_linear

            # bin (spherical shells in 3D, circular annuli in 2D) volumes/areas [code units]
            r_outer = rad_bin_edges[1:]
            r_inner = rad_bin_edges[:-1]
            r_inner[0] = 0.0

            bin_volumes_code = 4.0 / 3.0 * np.pi * (r_outer**3.0 - r_inner**3.0)
            bin_areas_code = np.pi * (r_outer**2.0 - r_inner**2.0)  # 2D annuli e.g. if proj2D is not None

        # allocation: for global particle scope: [all, self-halo, other-halo, diffuse]
        # or for subfind/fof scope: [self-halo] only, or for subfind_global scope: [other-halo] only
        numProfTypes = 4 if scope in ["global", "global_fof"] else 1
    else:
        # spherical sampling: get sample points (centered at 0,0,0 in units of rvir)
        pts, nProj, rad_bins_rvir, _ = healpix_shells_points(nRad=radNumBins, Nside=Nside, radMin=radMin, radMax=radMax)

        nRad = rad_bins_rvir.size

        # load virial radii (code units)
        gc["Subhalo_Rvir"] = sP.subhalos("rhalo_200_code")

    if pSplitSpatial:
        # spatial decomposition: determine extent and child subhalos
        assert np.abs(pSplitSpatial[1] ** (1 / 3) - np.round(pSplitSpatial[1] ** (1 / 3))) < 1e-6, (
            "pSplitSpatial: Total number of jobs should have integer cube root, e.g. 8, 27, 64."
        )
        nPerDim = int(pSplitSpatial[1] ** (1 / 3))
        extent = sP.boxSize / nPerDim

        ijk = np.unravel_index(pSplitSpatial[0], (nPerDim, nPerDim, nPerDim))
        xmin = ijk[0] * extent
        xmax = (ijk[0] + 1) * extent
        ymin = ijk[1] * extent
        ymax = (ijk[1] + 1) * extent
        zmin = ijk[2] * extent
        zmax = (ijk[2] + 1) * extent

        print(
            " pSplitSpatial: [%d of %d] ijk (%d %d %d) extent [%g] x [%.1f - %.1f] y [%.1f - %.1f] z [%.1f - %.1f]"
            % (pSplitSpatial[0], pSplitSpatial[1], ijk[0], ijk[1], ijk[2], extent, xmin, xmax, ymin, ymax, zmin, zmax)
        )

        # which subhalos?
        pos = gc["SubhaloPos"][subhaloIDsTodo]
        w_spatial = np.where(
            (pos[:, 0] > xmin)
            & (pos[:, 0] <= xmax)
            & (pos[:, 1] > ymin)
            & (pos[:, 1] <= ymax)
            & (pos[:, 2] > zmin)
            & (pos[:, 2] <= zmax)
        )

        subhaloIDsTodo = subhaloIDsTodo[w_spatial]

        # generate list of particle indices sufficient for (subhalos,binning specifications)
        mask = np.zeros(sP.numPart[sP.ptNum(ptType)], dtype="int8")
        mask += 1  # all required

        print(" pSplitSpatial:", end="")
        for ind, axis in enumerate(["x", "y", "z"]):
            print(" slice[%s]..." % axis, end="")
            dists = sP.snapshotSubsetP(ptType, "pos_" + axis, float32=True)

            dists = (ijk[ind] + 0.5) * extent - dists  # 1D, along axis, from center of subregion
            sP.correctPeriodicDistVecs(dists)

            # compute maxdist (in code units): the largest 1d distance we need for the calculation
            if Nside is None and not radRvirUnits:
                maxdist = extent / 2 + np.ceil(10.0**radMax)  # radMax in log code units
            else:
                # radMax in linear rvir units
                radMaxCode = np.nanmax(gc["Subhalo_Rvir"][subhaloIDsTodo]) * radMax * 1.05
                maxdist = extent / 2 + radMaxCode

            w_spatial = np.where(np.abs(dists) > maxdist)
            mask[w_spatial] = 0  # outside bounding box along this axis

        load_inds = np.nonzero(mask)[0]
        print(
            "\n pSplitSpatial: particle load fraction = %.2f%% vs. uniform expectation = %.2f%%"
            % (load_inds.size / mask.size * 100, 1 / pSplitSpatial[1] * 100)
        )

        dists = None
        w_spatial = None
        mask = None

    nSubsDo = len(subhaloIDsTodo)

    # info
    print(" " + desc)
    print(" " + select)
    print(
        " Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now..."
        % (gc["header"]["Nsubgroups_Total"], nSubsSelected, nSubsDo)
    )

    # allocate
    if Nside is None:
        # normal profiles" NaN indicates not computed except for mass where 0 will do
        r = np.zeros((nSubsDo, rad_bin_edges.size - 1, numProfTypes), dtype="float32")
        if numProfTypes == 1:
            r = np.squeeze(r, axis=2)
    else:
        # spherical sampling
        r = np.zeros((nSubsDo, nProj, nRad), dtype="float32")

    # if op not in ['sum']: # does not work with r[i,:] += below!
    #    r.fill(np.nan) # set NaN value for subhalos with e.g. no particles for op=mean

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = ["pos"]

    if ptRestrictions is not None:
        for restrictionField in ptRestrictions:
            fieldsLoad.append(restrictionField)

    if proj2D is not None and isinstance(proj2Daxis, str):
        # needed for moment of intertia tensor for rotations
        assert ptType == "gas"  # otherwise need to make a separate load to fill gasLocal
        fieldsLoad.append("mass")
        fieldsLoad.append("sfr")
        gc["SubhaloHalfmassRadType"] = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])

    if ptProperty == "radvel":
        fieldsLoad.append("pos")
        fieldsLoad.append("vel")
        gc["SubhaloVel"] = sP.groupCat(fieldsSubhalos=["SubhaloVel"])

    if ptProperty in ["losvel", "losvel_abs"]:
        assert proj2D is not None  # some 2D projection direction must be defined

        if proj2Daxis in [0, 1, 2]:
            # load component along one of the cartesian axes (line of sight direction)
            vel_key = "vel_%s" % ["x", "y", "z"][proj2Daxis]
            fieldsLoad.append(vel_key)
            gc["SubhaloVel"] = sP.groupCat(fieldsSubhalos=["SubhaloVel"])[:, proj2Daxis]
        else:
            # load full velocity 3-vector for later, per-subhalo rotation
            fieldsLoad.append("vel")
            gc["SubhaloVel"] = sP.groupCat(fieldsSubhalos=["SubhaloVel"])

    fieldsLoad = list(set(fieldsLoad))

    # so long as scope is not global, load the full particle set we need for these subhalos now
    if scope not in ["global", "global_fof", "global_spatial", "subfind_global"] or (
        scope == "global_spatial" and pSplitSpatial is None
    ):
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

        if ptProperty not in userCustomFields:
            particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)
            assert particles[ptProperty].ndim == 1

        if "count" not in particles:
            particles["count"] = particles[list(particles.keys())[0]].shape[0]

        # load weights, e.g. use particle masses or volumes (linear) as weights
        if weighting is not None:
            particles["weights"] = sP.snapshotSubsetP(partType=ptType, fields=weighting, indRange=indRange)

            assert particles["weights"].ndim == 1 and particles["weights"].size == particles["count"]

    if scope == "global_tngcluster":
        # second set of loads to get the non-fof particles from this original zoom sim
        particles2 = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange2)

        if ptProperty not in userCustomFields:
            particles2[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange2)

        if weighting is not None:
            particles2["weights"] = sP.snapshotSubsetP(partType=ptType, fields=weighting, indRange=indRange2)

        # combine with first set of particle load
        particles["count"] = particles["count"] + particles2["count"]
        for key in particles.keys():
            if key == "count":
                continue
            particles[key] = np.concatenate((particles[key], particles2[key]), axis=0)

    # if spatial decomposition, load the full particle set we need for these subhalos now
    if pSplitSpatial:
        # use snapshotSubsetC() to avoid ever having the global arrays in memory
        particles = {}

        for field in fieldsLoad:
            data = sP.snapshotSubsetC(partType=ptType, field=field, inds=load_inds, sq=False)
            for key in data:  # includes 'count'
                particles[key] = data[key]

        if ptProperty not in userCustomFields:
            particles[ptProperty] = sP.snapshotSubsetC(partType=ptType, field=ptProperty, inds=load_inds)
            assert particles[ptProperty].ndim == 1

        # load weights
        if weighting is not None:
            particles["weights"] = sP.snapshotSubsetC(partType=ptType, field=weighting, inds=load_inds)

            assert particles["weights"].ndim == 1 and particles["weights"].size == particles["count"]

    # chunk load: loop (possibly just once if chunk load is disabled)
    for chunkNum in range(nChunks):
        # load chunk now (we are simply accumulating, so no need to load everything at once)
        if scope in ["global", "global_fof"]:
            # calculate load indices (snapshotSubset is inclusive on last index, make sure we get to the end)
            indRange = [chunkNum * chunkSize, (chunkNum + 1) * chunkSize - 1]
            if chunkNum == nChunks - 1:
                indRange[1] = h["NumPart"][ptLoadType] - 1
            print("  [%2d] %9d - %d" % (chunkNum, indRange[0], indRange[1]))

            particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

            if ptProperty not in userCustomFields:
                particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)
                assert particles[ptProperty].ndim == 1

            assert "count" in particles
            assert weighting is None  # load not implemented

        # if approximating the particle distribution with pre-accumulated subhalo based values, load now
        if scope in ["subfind_global"]:
            assert nChunks == 1  # no need for more
            assert ptRestrictions is None  # cannot apply particle restriction to subhalos
            particles = {}

            if len(ptProperty.split(" ")) == 3:
                species, ion, prop = ptProperty.split(" ")
                acName = "Subhalo_%s_%s" % (prop.capitalize(), species + ion)
                print(" subfind_global: loading [%s] as effective particle data..." % acName)
            else:
                assert 0  # handle other cases, could e.g. call subhaloRadialReduction directly as:
                # (subhaloRadialReduction(sP,ptType='gas',ptProperty='O VI mass',op='sum',rad=None,scope='fof')

            particles[ptProperty] = sP.auxCat(acName)[acName]
            particles["count"] = particles[ptProperty].size
            particles["Coordinates"] = gc["SubhaloPos"]

            # for now assume full subhalo coverage, otherwise need to handle and remap subhalo positions
            # based on ac['subhaloIDs'], e.g. for a subhaloRadialReduction(..., scope='fof', css='cen')
            assert particles["count"] == gc["header"]["Nsubgroups_Total"]

        # construct a global octtree to accelerate searching?
        tree = None
        if useTree:
            tree = buildFullTree(particles["Coordinates"], boxSizeSim=sP.boxSize, treePrec="float64", verbose=True)

        if ptProperty in userCustomFields and Nside is not None:  # allocate for local halocentric calculations
            loc_val = np.zeros(particles["count"], dtype="float32")

        if scope in ["global", "global_fof", "subfind", "fof"]:
            indRangeSize = indRange[1] - indRange[0] + 1  # size of load

        if scope in ["global", "global_fof"]:
            # make mask corresponding to particles in all subhalos (other-halo term)
            subhalo_particle_mask = np.zeros(indRangeSize, dtype="int16")

            # loop from where we previously stopped, to the end of all subhalos
            for i in range(prevMaskInd, gc["header"]["Nsubgroups_Total"]):
                ind0 = gc["SubhaloOffsetType"][i, ptLoadType] - indRange[0]
                ind1 = ind0 + gc["SubhaloLenType"][i, ptLoadType]

                # out of loaded chunk, stop marking for now
                if ind0 >= indRangeSize:
                    break

                # is this subhalo entirely outside the current chunk? [0,indRangeSize]
                if ind1 <= 0:
                    continue

                # clip indices to be local to this loaded chunk
                ind0 = np.max([ind0, 0])
                ind1 = np.min([ind1, indRangeSize])

                # stamp
                subhalo_particle_mask[ind0:ind1] = 1

            prevMaskInd = i - 1

        # loop over subhalos
        for i, subhaloID in enumerate(subhaloIDsTodo):
            if i % np.max([1, int(nSubsDo / 10.0)]) == 0 and i <= nSubsDo:
                print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo), flush=True)

            # slice starting/ending indices for stars local to this subhalo/FoF
            i0 = 0
            i1 = particles["count"]  # set default i1 to full chunk size for 'global' scope

            if scope in ["subfind", "fof"]:
                i0 = gc["SubhaloOffsetType"][subhaloID, ptLoadType] - indRange[0]
                i1 = i0 + gc["SubhaloLenType"][subhaloID, ptLoadType]

                assert i0 >= 0 and i1 <= indRangeSize

                if i1 == i0:
                    continue  # zero length of this type

                if op == np.std and i1 - i0 == 1:
                    continue  # need at least 2 of this type

            # healpix spherical sampling? calculate now
            if Nside is not None:
                # shift sampling points to be halo local, account for periodic BCs
                pts_loc = pts.copy()
                pts_loc *= gc["Subhalo_Rvir"][subhaloID]
                pts_loc += gc["SubhaloPos"][subhaloID][np.newaxis, :]

                sP.correctPeriodicPosVecs(pts_loc)

                # derive hsml (one per sample point)
                loc_hsml = calcHsml(particles["Coordinates"], sP.boxSize, posSearch=pts_loc, nNGB=Nngb, tree=tree)

                # property
                if ptProperty not in userCustomFields:
                    loc_val = particles[ptProperty]
                elif ptProperty == "radvel":
                    # take tree search spatial subset
                    radMaxCode = gc["Subhalo_Rvir"][subhaloID] * radMax * 1.05  # maxRad in linear rvir units

                    loc_inds = calcParticleIndices(
                        particles["Coordinates"],
                        gc["SubhaloPos"][subhaloID, :],
                        radMaxCode,
                        boxSizeSim=sP.boxSize,
                        tree=tree,
                    )

                    # TODO: de-duplicate 'radvel' logic from below
                    p_pos = np.squeeze(particles["Coordinates"][loc_inds, :])
                    p_vel = np.squeeze(particles["Velocities"][loc_inds, :])

                    haloPos = gc["SubhaloPos"][subhaloID, :]
                    haloVel = gc["SubhaloVel"][subhaloID, :]

                    # only compute particleRadialVelInKmS() on the local subset for efficiency
                    loc_val *= 0
                    loc_val[loc_inds] = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)
                else:
                    raise Exception("Unhandled.")

                # sample (note: cannot modify e.g. subset loc_pos, must correspond to the constructed tree)
                loc_pos = particles["Coordinates"]

                result = calcQuantReduction(loc_pos, loc_val, loc_hsml, op, sP.boxSize, posSearch=pts_loc, tree=tree)
                result = np.reshape(result, (nProj, nRad))

                # stamp and continue to next subhalo
                r[i, :, :] = result

                continue

            # tree based search?
            if tree is not None:
                maxSearchRad = radMaxCode
                if radRvirUnits:
                    maxSearchRad = gc["Subhalo_Rvir"][subhaloID] * radMax * 1.05

                loc_inds = calcParticleIndices(
                    particles["Coordinates"],
                    gc["SubhaloPos"][subhaloID, :],
                    maxSearchRad,
                    boxSizeSim=sP.boxSize,
                    tree=tree,
                )

                if loc_inds is None:
                    continue  # zero particles of this type within search radius
                loc_size = loc_inds.size
            else:
                loc_inds = np.s_[i0:i1]  # slice object (will create view)
                loc_size = i1 - i0

            # particle pos subset
            particles_pos = particles["Coordinates"][loc_inds]

            # rotation?
            rotMatrix = None

            if proj2D is not None and isinstance(proj2Daxis, str) and (loc_size > 1):  # at least 2 particles
                # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                rHalf = gc["SubhaloHalfmassRadType"][subhaloID, sP.ptNum("stars")]
                shPos = gc["SubhaloPos"][subhaloID, :]

                # local particle set: even if we are computing global radial profiles
                i0g = gc["SubhaloOffsetType"][subhaloID, ptLoadType] - indRange[0]
                i1g = i0g + gc["SubhaloLenType"][subhaloID, ptLoadType]

                gasLocal = {
                    "Masses": particles["Masses"][i0g:i1g],
                    "Coordinates": np.squeeze(particles["Coordinates"][i0g:i1g, :]),
                    "StarFormationRate": particles["StarFormationRate"][i0g:i1g],
                    "count": i1g - i0g,
                }
                starsLocal = {"count": 0}

                I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos, useStars=False)

                rots = rotationMatricesFromInertiaTensor(I)
                rotMatrix = rots[proj2Daxis]

                # rotate coordinates (velocities handled below)
                particles_pos = particles_pos.copy()
                particles_pos, _ = rotateCoordinateArray(sP, particles_pos, rotMatrix, shPos)

            # use squared radii and sq distance function
            validMask = np.ones(particles_pos.shape[0], dtype="bool")

            if proj2D is None:
                # apply in 3D
                rr = sP.periodicDists(gc["SubhaloPos"][subhaloID, :], particles_pos)
            else:
                # apply in 2D projection, along the specified axis
                pt_2d = gc["SubhaloPos"][subhaloID, :]
                pt_2d = np.array([pt_2d[p_inds[0]], pt_2d[p_inds[1]]])
                vecs_2d = np.zeros((particles_pos.shape[0], 2), dtype=particles_pos.dtype)
                vecs_2d[:, 0] = particles_pos[:, p_inds[0]]
                vecs_2d[:, 1] = particles_pos[:, p_inds[1]]

                rr = sP.periodicDists(pt_2d, vecs_2d)  # handles 2D

                # enforce depth restriction
                if proj2Ddepth is not None:
                    dist_projDir = particles_pos[:, p_inds[2]].copy()  # careful of view
                    dist_projDir -= gc["SubhaloPos"][subhaloID, p_inds[2]]
                    sP.correctPeriodicDistVecs(dist_projDir)
                    validMask &= np.abs(dist_projDir) <= proj2D_halfDepth

            if scope in ["subfind_global"]:
                # do not self count, we are accumulating the other-halo term
                validMask[subhaloID] = 0

            validMask &= rr <= radMaxCode

            # apply particle-level restrictions
            if ptRestrictions is not None:
                for restrictionField in ptRestrictions:
                    inequality, val = ptRestrictions[restrictionField]

                    if inequality == "gt":
                        validMask &= particles[restrictionField][loc_inds] > val
                    if inequality == "lt":
                        validMask &= particles[restrictionField][loc_inds] <= val
                    if inequality == "eq":
                        validMask &= particles[restrictionField][loc_inds] == val

            wValid = np.where(validMask)

            if len(wValid[0]) == 0:
                continue  # zero length of particles satisfying radial cut and restriction

            if radRvirUnits:
                # distance, in units of rvir [dimensionless linear]
                loc_rr = rr[wValid] / gc["Subhalo_Rvir"][subhaloID]
            else:
                # distance, in code units [linear]
                loc_rr = rr[wValid]

            loc_wt = particles["weights"][loc_inds][wValid] if weighting is not None else None

            if ptProperty not in userCustomFields:
                loc_val = particles[ptProperty][loc_inds][wValid]
            else:
                # user function reduction operations, set loc_val now
                if ptProperty == "radvel":
                    p_pos = np.squeeze(particles_pos[wValid, :])
                    p_vel = np.squeeze(particles["Velocities"][loc_inds, :][wValid, :])

                    haloPos = gc["SubhaloPos"][subhaloID, :]
                    haloVel = gc["SubhaloVel"][subhaloID, :]

                    loc_val = sP.units.particleRadialVelInKmS(p_pos, p_vel, haloPos, haloVel)

                if ptProperty in ["losvel", "losvel_abs"]:
                    if rotMatrix is None:
                        p_vel = sP.units.particleCodeVelocityToKms(particles[vel_key][loc_inds][wValid])
                        assert p_vel.ndim == 1  # otherwise, do the following (old)
                        # p_vel   = p_vel[:,p_inds[2]]
                        haloVel = sP.units.subhaloCodeVelocityToKms(gc["SubhaloVel"][subhaloID])  # [p_inds[2]]
                    else:
                        p_vel = sP.units.particleCodeVelocityToKms(
                            np.squeeze(particles["Velocities"][loc_inds, :][wValid, :])
                        )
                        haloVel = sP.units.subhaloCodeVelocityToKms(gc["SubhaloVel"][subhaloID, :])

                        p_vel = np.array(np.transpose(np.dot(rotMatrix, p_vel.transpose())))
                        p_vel = np.squeeze(p_vel[:, p_inds[2]])  # slice index 2 by convention of rotMatrix

                        haloVel = np.array(np.transpose(np.dot(rotMatrix, haloVel.transpose())))[p_inds[2]][0]

                    loc_val = p_vel - haloVel
                    if ptProperty == "losvel_abs":
                        loc_val = np.abs(loc_val)

                if ptProperty in ["tff", "tcool_tff"]:
                    # do per-halo load (not scalable)
                    if scope == "subhalo":
                        loc_val = sP.snapshotSubset(ptType, ptProperty, subhaloID=subhaloID)
                    elif scope == "fof":
                        haloID = sP.subhalo(subhaloID)["SubhaloGrNr"]
                        loc_val = sP.snapshotSubset(ptType, ptProperty, haloID=haloID)
                    loc_val = loc_val[wValid]

            # weighted histogram (or other op) of rr_log distances
            if scope in ["global", "global_fof"]:
                # (1) all
                result, _, _ = binned_statistic_weighted(
                    loc_rr, loc_val, statistic=op, bins=rad_bin_edges, weights=loc_wt
                )
                r[i, :, 0] += result

                # (2) self-halo
                restoreSelf = False
                if gc["SubhaloLenType"][subhaloID, ptLoadType]:
                    is0 = gc["SubhaloOffsetType"][subhaloID, ptLoadType] - indRange[0]
                    is1 = is0 + gc["SubhaloLenType"][subhaloID, ptLoadType]

                    # update mask to specifically mark this halo (do not include in the other-halo term)
                    if not ((is0 < 0 and is1 <= 0) or (is0 >= indRangeSize and is1 > indRangeSize)):
                        is0 = np.max([is0, 0])
                        is1 = np.min([is1, indRangeSize])
                        subhalo_particle_mask[is0:is1] = 2
                        restoreSelf = True

                # extract mask portion corresponding to current valid particle selection
                loc_mask = subhalo_particle_mask[wValid]

                w = np.where(loc_mask == 2)

                if len(w[0]):
                    # this subhalo at least partially in the currently loaded data
                    result, _, _ = binned_statistic_weighted(
                        loc_rr[w], loc_val[w], statistic=op, bins=rad_bin_edges, weights=loc_wt, weights_w=w
                    )
                    r[i, :, 1] += result

                # (3) other-halo
                w = np.where(loc_mask == 1)

                if len(w[0]):
                    result, _, _ = binned_statistic_weighted(
                        loc_rr[w], loc_val[w], statistic=op, bins=rad_bin_edges, weights=loc_wt, weights_w=w
                    )
                    r[i, :, 2] += result

                if restoreSelf:
                    subhalo_particle_mask[is0:is1] = 1  # restore

                # (4) diffuse
                w = np.where(loc_mask == 0)

                if len(w[0]):
                    result, _, _ = binned_statistic_weighted(
                        loc_rr[w], loc_val[w], statistic=op, bins=rad_bin_edges, weights=loc_wt, weights_w=w
                    )
                    r[i, :, 3] += result
            else:
                # subhalo/fof/global_spatial scope, only compute the self-term, or 'subfind_global' technique
                result, _, _ = binned_statistic_weighted(
                    loc_rr, loc_val, statistic=op, bins=rad_bin_edges, weights=loc_wt
                )
                r[i, :] += result

    # return
    if Nside is None:
        attrs = {
            "Description": desc.encode("ascii"),
            "Selection": select.encode("ascii"),
            "ptType": ptType.encode("ascii"),
            "ptProperty": ptProperty.encode("ascii"),
            "weighting": str(weighting).encode("ascii"),
            "rad_bin_edges": rad_bin_edges,
            "radBinsLog": radBinsLog,
            "radRvirUnits": radRvirUnits,
            "subhaloIDs": subhaloIDsTodo,
        }

        if not radRvirUnits:
            attrs["rad_bins_code"] = rad_bins_code
            attrs["rad_bins_pkpc"] = rad_bins_pkpc
            attrs["bin_volumes_code"] = bin_volumes_code
            attrs["bin_areas_code"] = bin_areas_code
    else:
        attrs = {
            "Description": desc.encode("ascii"),
            "Selection": select.encode("ascii"),
            "ptType": ptType.encode("ascii"),
            "ptProperty": ptProperty.encode("ascii"),
            "weighting": str(weighting).encode("ascii"),
            "Nside": Nside,
            "Nngb": Nngb,
            "op": op,
            "rad_bins_rvir": rad_bins_rvir,
            "subhaloIDs": subhaloIDsTodo,
        }

    return r, attrs
