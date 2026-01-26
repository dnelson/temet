"""
Cosmological simulations - common utilities for auxiliary catalog calculation.
"""

import numpy as np

from ..util.helper import pSplit as pSplitArr
from ..util.helper import pSplitRange


# todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can
# eliminate the entire specialized ufunc logic herein
userCustomFields = [
    "Krot",
    "radvel",
    "losvel",
    "losvel_abs",
    "veldisp3d",
    "veldisp1d",
    "veldisp_z",
    "shape_ellipsoid",
    "shape_ellipsoid_1r",
    "tff",
    "tcool_tff",
]


def _radialRestriction(sP, nSubsTot, rad):
    """Handle an input 'rad' specification of a radial restriction.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      nSubsTot (int): total number of subhalos at this snapshot.
      rad (float or str): should match one of the options below, which specifies the details which are
        applied to particle/cell distances in order to achieve the aperture/radial restriction.

    Return:
      a 4-tuple composed of

      - **radRestrictIn2D** (bool): apply the cut to 2d projected, instead of 3d, distances.
      - **radSqMin** (list[float]): for each subhalo, the minimum distance to consider, squared.
      - **radSqMax** (list[float]): for each subhalo, the maximum distance to consider, squared.
      - **slit_code** (list[float]): if not None, represents 2d x,y aperture geometry of a slit.
    """
    radRestrictIn2D = False
    radSqMin = np.zeros(nSubsTot, dtype="float32")  # leave at zero unless modified below
    radSqMax = None
    slit_code = None  # used to return aperture geometry for weighted inclusions

    if isinstance(rad, float):
        # constant scalar, convert [pkpc] -> [ckpc/h] (code units) at this redshift
        rad_code = sP.units.physicalKpcToCodeLength(rad)
        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += rad_code * rad_code
    elif rad is None:
        # no radial restriction (all particles in subhalo)
        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += sP.boxSize**2.0
    elif rad == "p10":
        # load group m200_crit values
        gcLoad = sP.groupCat(fieldsHalos=["Group_M_Crit200"], fieldsSubhalos=["SubhaloGrNr"])
        parentM200 = gcLoad["halos"][gcLoad["subhalos"]]

        # r_cut = 27.3 kpc/h * (M200crit / (10^15 Msun/h))^0.29 from Puchwein+ (2010) Eqn 1
        r_cut = 27.3 * (parentM200 / 1e5) ** (0.29) / sP.HubbleParam
        radSqMax = r_cut * r_cut
    elif rad == "30h":
        # hybrid, minimum of [constant scalar 30 pkpc] and [the usual, 2rhalf,stars]
        rad_code = sP.units.physicalKpcToCodeLength(30.0)

        subHalfmassRadType = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])
        twiceStellarRHalf = 2.0 * subHalfmassRadType[:, sP.ptNum("stars")]

        ww = np.where(twiceStellarRHalf > rad_code)
        twiceStellarRHalf[ww] = rad_code
        radSqMax = twiceStellarRHalf**2.0
    elif rad == "10pkpc":
        # r < 10 pkpc
        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += (sP.units.physicalKpcToCodeLength(10.0)) ** 2
    elif rad == "10pkpc_shell":
        # shell at 10 +/- 2 pkpc
        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += (sP.units.physicalKpcToCodeLength(12.0)) ** 2
        radSqMin += (sP.units.physicalKpcToCodeLength(8.0)) ** 2
    elif rad == "rvir_shell":
        # shell at 1.0rvir +/- 0.1 rvir
        gcLoad = sP.groupCat(fieldsHalos=["Group_R_Crit200"], fieldsSubhalos=["SubhaloGrNr"])
        parentR200 = gcLoad["halos"][gcLoad["subhalos"]]

        radSqMax = (1.1 * parentR200) ** 2
        radSqMin = (0.9 * parentR200) ** 2
    elif rad == "r015_1rvir_halo":
        # classic 'halo' definition, 0.15rvir < r < 1.0rvir (meaningless for non-centrals)
        gcLoad = sP.groupCat(fieldsHalos=["Group_R_Crit200"], fieldsSubhalos=["SubhaloGrNr"])
        parentR200 = gcLoad["halos"][gcLoad["subhalos"]]

        radSqMax = (1.00 * parentR200) ** 2
        radSqMin = (0.15 * parentR200) ** 2
    elif rad == "20pkpc_halfrvir":
        # 'inner halo' definition: from 20 pkpc to 0.5 r200c (must be centrals only to avoid nans)
        parentR200 = sP.subhalos("rhalo_200_code")

        radSqMax = (0.5 * parentR200) ** 2
        radSqMin += (sP.units.physicalKpcToCodeLength(20.0)) ** 2
    elif rad == "halfrvir_rvir":
        # 'outer halo' definition: from 0.5 r200c to r200c (centrals only)
        parentR200 = sP.subhalos("rhalo_200_code")

        radSqMax = (1.0 * parentR200) ** 2
        radSqMin = (0.5 * parentR200) ** 2
    elif rad in ["r200crit", "rvir"]:
        # within the virial radius (r200,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=["Group_R_Crit200"], fieldsSubhalos=["SubhaloGrNr"])
        parentR200 = gcLoad["halos"][gcLoad["subhalos"]]

        radSqMax = (1.00 * parentR200) ** 2
    elif rad in ["2r200crit", "2rvir"]:
        # within twice the virial radius (r200,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=["Group_R_Crit200"], fieldsSubhalos=["SubhaloGrNr"])
        parentR200 = gcLoad["halos"][gcLoad["subhalos"]]

        radSqMax = (2.00 * parentR200) ** 2
    elif rad in ["0.1r500crit", "0.5r500crit", "r500crit"]:
        # within the (r500,crit definition) (centrals only)
        gcLoad = sP.groupCat(fieldsHalos=["Group_R_Crit500"], fieldsSubhalos=["SubhaloGrNr"])
        parentR500 = gcLoad["halos"][gcLoad["subhalos"]]

        if rad == "r500crit":
            radSqMax = (1.0 * parentR500) ** 2
        elif rad == "0.5r500crit":
            radSqMax = (0.5 * parentR500) ** 2
        elif rad == "0.1r500crit":
            radSqMax = (0.1 * parentR500) ** 2
    elif rad == "2rhalfstars":
        # classic Illustris galaxy definition, r < 2*r_{1/2,mass,stars}
        subHalfmassRadType = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])
        twiceStellarRHalf = 2.0 * subHalfmassRadType[:, sP.ptNum("stars")]

        radSqMax = twiceStellarRHalf**2
    elif rad == "1rhalfstars":
        # inner galaxy definition, r < 1*r_{1/2,mass,stars}
        subHalfmassRadType = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])
        stellarRHalf = 1.0 * subHalfmassRadType[:, sP.ptNum("stars")]

        radSqMax = stellarRHalf**2
    elif rad == "0.5rhalfstars":
        #  < 0.5*r_{1/2,mass,stars}
        subHalfmassRadType = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])
        stellarRHalf = 1.0 * subHalfmassRadType[:, sP.ptNum("stars")]

        radSqMax = (stellarRHalf / 2) ** 2
    elif rad == "2rhalfstars_fof":
        # classic galaxy definition, r < 2*r_{1/2,mass,stars}, except based on re-computed
        # stellar half mass radii that consider all FoF stars, not just those assigned to the subhalo
        subHalfmassRadType = sP.subhalos("rhalf_stars_fof")
        twiceStellarRHalf = 2.0 * subHalfmassRadType

        radSqMax = twiceStellarRHalf**2
    elif rad == "1pkpc_2d":
        # 1 pkpc in 2D projection (e.g. for Sigma_1)
        rad_code = sP.units.physicalKpcToCodeLength(1.0)
        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += rad_code * rad_code

        radRestrictIn2D = True
    elif rad == "sdss_fiber":
        # SDSS fiber is 3" diameter, convert to physical radius at this redshift for all z>0
        # for z=0.0 snapshots only, for this purpose we fake the angular diameter distance at z=0.1
        fiber_z = sP.redshift if sP.redshift > 0.0 else 0.1
        fiber_arcsec = 3.0  # note: 2.0 for BOSS, 3.0 for legacy SDSS
        fiber_diameter = sP.units.arcsecToAngSizeKpcAtRedshift(fiber_arcsec, z=fiber_z)
        print(" SDSS fiber diameter [%.2f pkpc] at redshift [z = %.2f]. NOTE: 2D!" % (fiber_diameter, fiber_z))

        # convert [pkpc] -> [ckpc/h] (code units) at this redshift
        fiber_diameter = sP.units.physicalKpcToCodeLength(fiber_diameter)

        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += (fiber_diameter / 2.0) ** 2.0
        radRestrictIn2D = True
    elif rad == "sdss_fiber_4pkpc":
        # keep old 4pkpc 'fiber radius' approximation but with 2D
        rad_pkpc = sP.units.physicalKpcToCodeLength(4.0)

        radSqMax = np.zeros(nSubsTot, dtype="float32")
        radSqMax += rad_pkpc * rad_pkpc
        radRestrictIn2D = True
    elif rad == "legac_slit":
        # slit is 1" x 8" minimum (length is variable and depends on galaxy size? TODO)
        slit_arcsec = np.array([1.0, 4.0])
        slit_kpc = sP.units.arcsecToAngSizeKpcAtRedshift(slit_arcsec, z=sP.redshift)  # arcsec -> pkpc
        slit_code = sP.units.physicalKpcToCodeLength(slit_kpc)  # pkpc -> ckpc/h

        radSqMax = np.zeros((nSubsTot, 2), dtype="float32")  # second dim: (x,y) in projection
        radSqMax[:, 0] += (slit_code[0] / 2.0) ** 2
        radSqMax[:, 1] += (slit_code[1] / 2.0) ** 2
        radRestrictIn2D = True

    assert radSqMax is not None, "Unrecognized [%s] rad specification." % rad
    assert radSqMax.size == nSubsTot or (radSqMax.size == nSubsTot * 2 and radSqMax.ndim == 2)
    assert radSqMin.size == nSubsTot or (radSqMin.size == nSubsTot * 2 and radSqMin.ndim == 2)

    return radRestrictIn2D, radSqMin, radSqMax, slit_code


def pSplitBounds(
    sP,
    pSplit,
    minStellarMass=None,
    minHaloMass=None,
    indivStarMags=False,
    partType=None,
    cenSatSelect=None,
    equalSubSplit=True,
):
    """Determine an efficient split of subhalos (and global snapshot index range needed) to process for a pSplit task.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      minStellarMass (float): apply lower limit on ``mstar_30pkpc_log``.
      minHaloMass (float): apply lower limit on ``mhalo_200_log``.
      indivStarMags (bool): make sure return covers the full PartType4 size.
      partType (str or None): if not None, use this to decide particle-based split, otherwise use 'gas'.
      cenSatSelect (str or None): if not None, restrict to 'cen', 'sat', or 'all'.
      equalSubSplit (bool): subdivide a pSplit based on equal numbers of subhalos, rather than particles.

    Return:
      a 3-tuple composed of

      - **subhaloIDsTodo** (list[int]): the list of subhalo IDs to process by this task.
      - **indRange** (dict): the index range for the particle load required to cover these subhalos.
      - **nSubsSelection** (int): the number of subhalos to be processed.
    """
    nSubsTot = sP.numSubhalos
    subhaloIDsTodo = np.arange(nSubsTot, dtype="int32")

    # stellar mass select
    if minStellarMass is not None:
        if str(minStellarMass) == "100stars":
            # sP dependent: one hundred stellar particles
            minStellarMass = sP.units.codeMassToLogMsun(sP.targetGasMass * 100)
            minStellarMass = np.round(minStellarMass[0] * 10) / 10  # round to 0.1

        masses = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])
        with np.errstate(invalid="ignore"):
            wSelect = np.where(masses >= minStellarMass)

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # m200 halo mass select
    if minHaloMass is not None:
        if str(minHaloMass) in ["1000dm", "10000dm"]:
            # sP dependent: one thousand dm particles
            numDM = 10000 if minHaloMass == "10000dm" else 1000
            minHaloMass = sP.units.codeMassToLogMsun(sP.dmParticleMass * numDM)
            minHaloMass = np.round(minHaloMass[0] * 10) / 10  # round to 0.1

        halo_masses = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])
        if minStellarMass is not None:
            halo_masses = halo_masses[wSelect]

        with np.errstate(invalid="ignore"):
            wSelect = np.where(halo_masses >= minHaloMass)
        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # cen/sat select?
    if cenSatSelect is not None:
        cssSubIDs = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        subhaloIDsTodo = np.intersect1d(subhaloIDsTodo, cssSubIDs)

    nSubsSelection = subhaloIDsTodo.size

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)

    invSubs = [0, 0]

    if pSplit is not None:
        if 0:
            # split up subhaloIDs in round-robin scheme (equal number of massive/centrals per job)
            # works perfectly for balance, but retains global load of all haloSubset particles
            modSplit = subhaloIDsTodo % pSplit[1]
            subhaloIDsTodo = np.where(modSplit == pSplit[0])[0]

        if sP.name == "TNG-Cluster" and pSplit[1] == 352:
            # global load: restrict subhalo IDs to those from this original zoom sim
            orig_halo_id = sP.subhalos("SubhaloOrigHaloID")
            orig_halo_id_uniq = np.unique(orig_halo_id)
            cur_halo_id = orig_halo_id_uniq[pSplit[0]]

            orig_halo_id = orig_halo_id[subhaloIDsTodo]

            inds_todo = np.where(orig_halo_id == cur_halo_id)[0]
            subhaloIDsTodo = subhaloIDsTodo[inds_todo]
            indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)

        elif equalSubSplit:
            # do contiguous subhalo ID division and reduce global haloSubset load
            # to the particle sets which cover the subhalo subset of this pSplit, but the issue is
            # that early tasks take all the large halos and all the particles, very imbalanced
            subhaloIDsTodo = pSplitArr(subhaloIDsTodo, pSplit[1], pSplit[0])

            indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)
        else:
            # subdivide the global cell/particle set, then map this back into a division of
            # subhalo IDs which will be better work-load balanced among tasks
            ptType = partType if partType is not None else "gas"
            gasSplit = pSplitRange(indRange[ptType], pSplit[1], pSplit[0])

            invSubs = sP.inverseMapPartIndicesToSubhaloIDs(gasSplit, ptType, flagFuzz=False)

            if pSplit[0] == pSplit[1] - 1:
                invSubs[1] = nSubsTot

            assert invSubs[1] != -1

            if invSubs[1] == invSubs[0]:
                # split is actually zero size, this is ok
                return [], {"gas": [0, 1], "stars": [0, 1], "dm": [0, 1]}, nSubsSelection

            invSubIDs = np.arange(invSubs[0], invSubs[1])
            subhaloIDsTodo = np.intersect1d(subhaloIDsTodo, invSubIDs)
            indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)

    if indivStarMags:
        # make subhalo-strict bounding index range and compute number of PT4 particles we will do
        if invSubs[0] > 0:
            # except for first pSplit, move coverage to include the last subhalo of the previous
            # split, then increment the indRange[0] by the length of that subhalo. in this way we
            # avoid any gaps for full PT4 coverage
            subhaloIDsTodo_extended = np.arange(invSubs[0] - 1, invSubs[1])

            indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo_extended, strictSubhalos=True)

            lastPrevSub = sP.groupCatSingle(subhaloID=invSubs[0] - 1)
            indRange["stars"][0] += lastPrevSub["SubhaloLenType"][sP.ptNum("stars")]
        else:
            indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo, strictSubhalos=True)

    return subhaloIDsTodo, indRange, nSubsSelection


def findHalfLightRadius(rad, vals, frac=0.5, mags=True):
    """Find radius that encloses half of the total quantity, e.g. half light or half mass radii.

    Linearly interpolate in rr (squared radii) to find the half light radius, given a list of
    values[i] corresponding to each particle at rad[i].

    Args:
      rad (:py:class:`~numpy.ndarray`): list of **squared** radii.
      vals (:py:class:`~numpy.ndarray`): list of values.
      frac (float): if 0.5, then half-light radius, otherwise e.g. 0.2 for r20.
      mags (bool): input ``vals`` are magnitudes, i.e. conversion to linear luminosity needed.

    Return:
      float: half-light radius in 3D or 2D (if rad is input 3D or 2D).
    """
    assert rad.size == vals.size

    # take input values unchanged (assume e.g. linear masses or light quantities already)
    lums = vals.copy()

    if mags:
        # convert individual mags to luminosities [arbitrary units]
        lums = np.power(10.0, -0.4 * lums)

    radii = rad.copy()
    totalLum = np.nansum(lums)

    sort_inds = np.argsort(radii)

    radii = radii[sort_inds]
    lums = lums[sort_inds]

    # cumulative sum luminosities in radial-distance order
    w = np.where(~np.isfinite(lums))  # wind particles have mags==nan -> lums==nan
    lums[w] = 0.0

    lums_cum = np.cumsum(lums)

    # locate radius where sum equals half of total (half-light radius)
    w = np.where(lums_cum >= frac * totalLum)[0]
    if len(w) == 0:
        return np.nan

    w1 = np.min(w)

    # linear interpolation in linear(rad) and linear(lum), find radius where lums_cum = totalLum/2
    if w1 == 0:
        # half of total luminosity could be within the radius of the first star
        r1 = lums_cum[w1]
        halfLightRad = (frac * totalLum - 0.0) / (r1 - 0.0) * (radii[w1] - 0.0) + 0.0

        assert (halfLightRad >= 0.0 and halfLightRad <= radii[w1]) or np.isnan(halfLightRad)
    else:
        # more generally valid case
        w0 = w1 - 1
        assert w0 >= 0 and w1 < lums.size

        r0 = lums_cum[w0]
        r1 = lums_cum[w1]
        halfLightRad = (frac * totalLum - r0) / (r1 - r0) * (radii[w1] - radii[w0]) + radii[w0]

        assert halfLightRad >= radii[w0] and (halfLightRad - radii[w1]) < 1e-4

    return halfLightRad
