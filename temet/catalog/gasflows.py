"""
Cosmological simulations - auxiliary catalogs for analyzing gas (in/out)flows.
"""

from collections import OrderedDict
from os.path import isfile

import h5py
import numpy as np
from scipy.interpolate import interp1d

from ..util.helper import pSplitRange
from ..util.match import match
from ..util.rotation import momentOfInertiaTensor, rotateCoordinateArray, rotationMatricesFromInertiaTensor
from ..util.treeSearch import buildFullTree, calcParticleIndices


def instantaneousMassFluxes(
    sP,
    pSplit=None,
    ptType="gas",
    scope="subhalo_wfuzz",
    massField="Masses",
    rawMass=False,
    fluxMass=True,
    fluxKE=False,
    fluxP=False,
    proj2D=False,
    v200norm=False,
):
    """Derive radial mass, energy, or momentum flux rates (outflowing/inflowing).

    For every subhalo, use the instantaneous kinematics of gas to derive radial mass, energy, or
    momemtum flux rates (outflowing/inflowing), and compute high dimensional histograms of this gas
    mass/energy/mom flux as a function of (rad,vrad,dens,temp,metallicity), as well as a few particular
    2D marginalized histograms of interest and 1D marginalized histograms. To run,
    choose one of: ``rawMass``, ``fluxMass``, ``fluxKE``, or ``fluxP`` (see below).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int][2]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle/cell type, can be either 'gas' (PartType0) or 'wind' (PartType4).
      scope (str): analysis scope, can be one of 'subhalo', 'subhalo_wfuzz', or 'global' (slow).
      massField (str): if not 'Masses' (total gas cell mass), then use instead this field and derive fluxes
        only for this mass subset (e.g. 'Mg II mass').

      rawMass (bool): histogrammed quantity is mass [msun] (per bin, e.g. mass in a given radial+vrad bin).
      fluxMass (bool): histogrammed quantity is radial mass flux [msun/yr] (default behavior, used in paper).
      fluxKE (bool): histogrammed quantity is radial kinetic energy flux [10^30 erg/s].
      fluxP (bool): histogrammed quantity is radial momentum flux [10^30 g*cm/s^2].

      proj2D (bool): if True, then all 'rad' bins become 'rad2d' bins (projected distance, z-axis direction,
        i.e. these are -annular- apertures on the sky), and all 'vrad' bins become 'vlos' bins (1D line of
        sight velocity). In this case, only rawMass is supported, since we have no shell volume element to
        normalize by. Additionally, a 'down the barrel' geometry is assumed, e.g. only material in front
        of the galaxy contributes.
      v200norm (bool): if True, then all velocities are binned in thresholds which are fractions of v200 of
        the halo, rather than in absolute physical [km/s] units.

    Returns:
      tuple: a 2-tuple composed of:

      - **result** (:py:class:`~numpy.ndarray`): 1d or 2d array, containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    minStellarMass = 7.4  # log msun (30pkpc values)
    cenSatSelect = "cen"  # cen, sat, all

    if sP.boxSize > 65000:
        minStellarMass = 7.9
    if sP.boxSize > 200000:
        minStellarMass = 8.4

    assert ptType in ["gas", "wind"]
    assert scope in ["subhalo", "subhalo_wfuzz", "global"]
    assert rawMass + fluxMass + fluxKE + fluxP in [1]  # choose one
    if massField != "Masses":
        assert ptType == "gas"  # no other masses for accumulation, that can be computed for stars
    if proj2D:
        assert rawMass

    # set distance and velocity field names
    rad, vrad = ("rad", "vrad") if not proj2D else ("rad2d", "vlos")

    # multi-D histogram config, [bin_edges] for each field
    binConfig1 = OrderedDict()
    binConfig1[rad] = [0, 5, 15, 25, 35, 45, 55, 75, 125, 175, 225, 375, 525, 1475]
    binConfig1[vrad] = [-np.inf, -450, -350, -250, -150, -50, 0, 50, 150, 250, 350, 450, 550, 1450, 2550, np.inf]

    if ptType == "gas":
        binConfig1["temp"] = [0, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, np.inf]
        binConfig1["z_solar"] = [-np.inf, -3.0, -2.0, -1.0, -0.5, 0.0, 0.5, np.inf]
        binConfig1["numdens"] = [-np.inf, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, np.inf]

    binConfigs = [binConfig1]

    # fine-binning of vrad (for both gas and wind)
    binConfig7 = OrderedDict()
    binConfig7[rad] = binConfig1[rad]
    binConfig7[vrad] = np.linspace(-500, 3500, 401)  # 10 km/s spacing

    if v200norm:
        assert not proj2D  # otherwise check it all makes sense
        binConfig1[vrad] = [-np.inf,-2.0,-1.5,-1.2,-1.0,-0.8,-0.6,-0.4,-0.2,0.0,
                            0.2,0.4,0.6,0.8,1.0,1.2,1.5,2.0,3.0,4.0,5.0,10.0,np.inf,]  # fmt: skip
        binConfig7[vrad] = np.linspace(-2.0, 50.0, 521)  # 0.1v200 spacing

    # secondary histogram configs (semi-marginalized, 1D and 2D, always binned by rad,vrad)
    if ptType == "gas":
        # 1D
        binConfig2 = OrderedDict()
        binConfig2[rad] = binConfig1[rad]
        binConfig2[vrad] = binConfig1[vrad]
        binConfig2["temp"] = np.linspace(3.0, 9.0, 121)  # 0.05 dex spacing

        binConfig2b = OrderedDict()
        binConfig2b[rad] = binConfig1[rad]
        binConfig2b[vrad] = binConfig1[vrad]
        binConfig2b["temp_sfcold"] = np.linspace(3.0, 9.0, 121)  # 0.05 dex spacing

        binConfig3 = OrderedDict()
        binConfig3[rad] = binConfig1[rad]
        binConfig3[vrad] = binConfig1[vrad]
        binConfig3["z_solar"] = np.linspace(-3.0, 1.5, 91)  # 0.05 dex spacing

        binConfig4 = OrderedDict()
        binConfig4[rad] = binConfig1[rad]
        binConfig4[vrad] = binConfig1[vrad]
        binConfig4["numdens"] = np.linspace(-8.0, 2.0, 201)  # 0.05 dex spacing

        # 2D
        binConfig5 = OrderedDict()
        binConfig5[rad] = binConfig1[rad]
        binConfig5[vrad] = binConfig1[vrad]
        binConfig5["numdens"] = np.linspace(-8.0, 2.0, 41)  # 0.25 dex spacing
        binConfig5["temp"] = np.linspace(3.0, 9.0, 31)  # 0.2 dex spacing

        binConfig6 = OrderedDict()
        binConfig6[rad] = binConfig1[rad]
        binConfig6[vrad] = binConfig1[vrad]
        binConfig6["z_solar"] = np.linspace(-3.0, 1.5, 26)  # 0.2 dex spacing
        binConfig6["temp"] = np.linspace(3.0, 9.0, 31)  # 0.2 dex spacing

        binConfig8 = OrderedDict()
        binConfig8[rad] = binConfig1[rad]
        binConfig8["temp"] = np.linspace(3.0, 9.0, 31)  # 0.2 dex spacing
        binConfig8[vrad] = np.linspace(-500, 3500, 81)  # 50 km/s spacing

        if v200norm:
            binConfig8[vrad] = np.linspace(-2.0, 10.0, 121)  # 0.1v200 spacing

        binConfig8b = OrderedDict()
        binConfig8b[rad] = binConfig1[rad]
        binConfig8b["temp_sfcold"] = binConfig8["temp"]
        binConfig8b[vrad] = binConfig8[vrad]

        # binning of angular theta
        binConfig9 = OrderedDict()
        binConfig9[rad] = binConfig1[rad]
        binConfig9[vrad] = binConfig1[vrad]
        binConfig9["theta"] = np.linspace(-np.pi, np.pi, 73)  # 5 deg spacing

        binConfig10 = OrderedDict()
        binConfig10[rad] = binConfig1[rad]
        binConfig10[vrad] = binConfig1[vrad]
        binConfig10["temp"] = binConfig1["temp"]
        binConfig10["theta"] = np.linspace(-np.pi, np.pi, 37)  # 10 deg spacing

        binConfig11 = OrderedDict()
        binConfig11[rad] = binConfig1[rad]
        binConfig11[vrad] = binConfig1[vrad]
        binConfig11["z_solar"] = binConfig1["z_solar"]
        binConfig11["theta"] = np.linspace(-np.pi, np.pi, 37)  # 10 deg spacing

        binConfigs += [binConfig2, binConfig2b, binConfig3, binConfig4, binConfig5, binConfig6]
        binConfigs += [binConfig7, binConfig8, binConfig8b, binConfig9, binConfig10, binConfig11]

    if ptType == "wind":
        binConfigs += [binConfig7]  # fine-binning of vrad

    # derived from binning
    maxRad = np.max(binConfig1[rad])

    h_bins = []  # histogramdd() input
    for binConfig in binConfigs:
        h_bins.append([binConfig[field] for field in binConfig])

    # load group catalog
    ptNum = sP.ptNum(ptType)
    ptNum_gas = sP.ptNum("gas")
    ptNum_stars = sP.ptNum("stars")

    fieldsSubhalos = ["SubhaloPos", "SubhaloVel", "SubhaloLenType", "SubhaloHalfmassRadType"]

    if v200norm:
        fieldsSubhalos += ["mhalo_200_code", "rhalo_200_code"]

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]
    nSubsTot = sP.numSubhalos

    if v200norm:
        gc["vhalo_200"] = sP.units.codeM200R200ToV200InKmS(gc["mhalo_200_code"], gc["rhalo_200_code"])

    subhaloIDsTodo = np.arange(nSubsTot, dtype="int32")

    if scope == "subhalo_wfuzz":
        # add new 'ParentGroup_LenType' and 'ParentGroup_OffsetType' (FoF group values) (for both cen/sat)
        Groups = sP.groupCat(fieldsHalos=["GroupLenType", "GroupFirstSub", "GroupNsubs"])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=["SubhaloGrNr"])

        gc["ParentGroup_LenType"] = Groups["GroupLenType"][SubhaloGrNr, ptNum]
        gc["ParentGroup_GroupFirstSub"] = Groups["GroupFirstSub"][SubhaloGrNr]
        gc["ParentGroup_GroupNsubs"] = Groups["GroupNsubs"][SubhaloGrNr]
        gc["ParentGroup_OffsetType"] = GroupOffsetType[SubhaloGrNr, ptNum]

        if cenSatSelect != "cen":
            print("WARNING: Is this really the measurement to make? Satellite bound gas is excluded from themselves.")

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)

    if pSplit is not None and scope != "global":
        # subdivide the global [variable ptType!] particle set, then map this back into a division of
        # subhalo IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange(indRange[ptType], pSplit[1], pSplit[0])

        invSubs = sP.inverseMapPartIndicesToSubhaloIDs(gasSplit, ptType, debug=True, flagFuzz=False)

        if pSplit[0] == pSplit[1] - 1:
            invSubs[1] = nSubsTot
        else:
            assert invSubs[1] != -1

        subhaloIDsTodo = np.arange(invSubs[0], invSubs[1])
        indRange = sP.subhaloIDsToBoundingPartIndices(subhaloIDsTodo)  # dict by type

    if scope == "global":
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = {}
        for pt in ["gas", "stars", "wind"]:
            indRange[pt] = [0, h["NumPart"][sP.ptNum(pt)] - 1]
        i0 = 0  # never changes
        i1 = indRange[ptType][1]  # never changes

    # stellar mass select
    if minStellarMass is not None:
        masses = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])
        masses = masses[subhaloIDsTodo]
        with np.errstate(invalid="ignore"):
            wSelect = np.where(masses >= minStellarMass)

        print(" min M* [%.2f] gives [%d] of [%d] subhalos." % (minStellarMass, len(wSelect[0]), subhaloIDsTodo.size))

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    if cenSatSelect != "all":
        central_flag = sP.groupCat(fieldsSubhalos=["central_flag"])
        central_flag = central_flag[subhaloIDsTodo]
        if cenSatSelect == "sat":
            central_flag = ~central_flag
        wSelect = np.where(central_flag)

        print(" css [%s] reduces to [%d] of [%d] subhalos." % (cenSatSelect, len(wSelect[0]), subhaloIDsTodo.size))

        subhaloIDsTodo = subhaloIDsTodo[wSelect]

    # allocate
    nSubsDo = len(subhaloIDsTodo)

    rr = []
    saveSizeGB = []

    for binConfig in binConfigs:
        allocSize = [nSubsDo]
        for field in binConfig:
            allocSize.append(len(binConfig[field]) - 1)

        locSize = np.prod(allocSize) * 4.0 / 1024**3
        print("  ", binConfig.keys(), allocSize, "%.2f GB" % locSize)
        saveSizeGB.append(locSize)
        rr.append(np.zeros(allocSize, dtype="float32"))

    print(
        " Processing [%d] of [%d] total subhalos (allocating %.1f GB + %.1f GB = save size)..."
        % (nSubsDo, nSubsTot, saveSizeGB[0], np.sum(saveSizeGB) - saveSizeGB[0])
    )

    # load snapshot
    fieldsLoad = ["Coordinates", "Velocities", "Masses"]
    if massField != "Masses":
        fieldsLoad += [massField]

    if ptType == "gas":
        fieldsLoad += ["temp_log", "temp_sfcold_log", "z_solar_log", "numdens_log"]
        fieldsLoad += ["StarFormationRate"]  # for rotation

    particles = sP.snapshotSubset(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange[ptType])

    if ptType == "wind":
        # processing wind mass fluxes: zero mass of all real stars
        sftime = sP.snapshotSubset(partType=ptType, fields="sftime", sq=True, indRange=indRange[ptType])
        wStars = np.where(sftime >= 0.0)
        particles["Masses"][wStars] = 0.0
        sftime = None

    # load stellar/gas fields, as required for rotation
    if ptType == "gas":
        # particles is gas, now load stars
        gas = particles

        fieldsLoad = ["Coordinates", "Masses", "GFM_StellarFormationTime"]
        stars = sP.snapshotSubset(partType="stars", fields=fieldsLoad, sq=False, indRange=indRange["stars"])
    else:
        # particles is wind, now load gas
        stars = particles
        stars["Masses"] = sP.snapshotSubset(partType="stars", fields="mass", sq=True, indRange=indRange["stars"])
        stars["GFM_StellarFormationTime"] = sP.snapshotSubset(
            partType="stars", fields="sftime", sq=True, indRange=indRange["stars"]
        )

        fieldsLoad = ["Coordinates", "Masses", "StarFormationRate"]
        gas = sP.snapshotSubset(partType="gas", fields=fieldsLoad, sq=False, indRange=indRange["gas"])

    # global? build octtree now
    if scope == "global":
        print(" Start build of global oct-tree...")
        tree = buildFullTree(particles["Coordinates"], boxSizeSim=sP.boxSize, treePrec="float64")
        print(" Tree finished.")

    # loop over subhalos
    printFac = 100.0 if (sP.res > 512 or scope == "global") else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1, int(nSubsDo / printFac)]) == 0 and i <= nSubsDo:
            print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo))

        # slice starting/ending indices for gas local to this halo
        if scope == "subhalo":
            i0 = gc["SubhaloOffsetType"][subhaloID, ptNum] - indRange[ptType][0]
            i1 = i0 + gc["SubhaloLenType"][subhaloID, ptNum]
        if scope == "subhalo_wfuzz":
            i0 = gc["ParentGroup_OffsetType"][subhaloID] - indRange[ptType][0]
            i1 = i0 + gc["ParentGroup_LenType"][subhaloID]
        if scope == "global":
            pass  # use constant i0, i1

        assert i0 >= 0 and i1 <= (indRange[ptType][1] - indRange[ptType][0] + 1)

        if i1 == i0:
            continue  # zero length of this type

        # halo properties
        haloPos = gc["SubhaloPos"][subhaloID, :]
        haloVel = gc["SubhaloVel"][subhaloID, :]

        # extract local particle subset
        p_local = {}

        if scope == "global":
            # global? tree-search now within maximum radius
            loc_inds = calcParticleIndices(particles["Coordinates"], haloPos, maxRad, boxSizeSim=sP.boxSize, tree=tree)

            if 0:  # brute-force verify
                dists = sP.periodicDists(haloPos, particles["Coordinates"])
                ww = np.where(dists <= maxRad)

                zz = np.argsort(loc_inds)
                zz = loc_inds[zz]
                assert np.array_equal(zz, ww[0])

            for key in particles:
                if key == "count":
                    continue
                p_local[key] = particles[key][loc_inds]
        else:
            # halo-based particle selection: extract now
            for key in particles:
                if key == "count":
                    continue
                p_local[key] = particles[key][i0:i1]

        # restriction: eliminate satellites by zeroing mass of their member particles
        if scope == "subhalo_wfuzz":
            GroupFirstSub = gc["ParentGroup_GroupFirstSub"][subhaloID]
            GroupNsubs = gc["ParentGroup_GroupNsubs"][subhaloID]

            if GroupNsubs > 1:
                firstSat_ind0 = gc["SubhaloOffsetType"][GroupFirstSub + 1, ptNum] - i0
                # firstSat_ind1 = firstSat_ind0 + gc["SubhaloLenType"][GroupFirstSub + 1, ptNum]
                lastSat_ind0 = gc["SubhaloOffsetType"][GroupFirstSub + GroupNsubs - 1, ptNum] - i0
                lastSat_ind1 = lastSat_ind0 + gc["SubhaloLenType"][GroupFirstSub + GroupNsubs - 1, ptNum]

                p_local[massField][firstSat_ind0:lastSat_ind1] = 0.0

        # compute halo-centric quantities
        p_local["rad"] = sP.units.codeLengthToKpc(sP.periodicDists(haloPos, p_local["Coordinates"]))
        p_local["vrad"] = sP.units.particleRadialVelInKmS(
            p_local["Coordinates"], p_local["Velocities"], haloPos, haloVel
        )

        # 2D projected distance and line-of-sight velocity (hard-coded z-axis projection direction)
        p_inds = [0, 1]  # x,y (Nside == 'z-axis')
        p_ind3 = 3 - p_inds[0] - p_inds[1]
        pt_2d = [haloPos[p_inds[0]], haloPos[p_inds[1]]]
        vecs_2d = np.zeros((p_local["rad"].size, 2), dtype=p_local["Coordinates"].dtype)
        vecs_2d[:, 0] = p_local["Coordinates"][:, p_inds[0]]
        vecs_2d[:, 1] = p_local["Coordinates"][:, p_inds[1]]

        rad2d = np.sqrt(sP.periodicDistsSq(pt_2d, vecs_2d))  # handles 2D
        p_local["rad2d"] = sP.units.codeLengthToKpc(rad2d)
        vel_los = p_local["Velocities"][:, p_ind3]
        # [physical km/s], relative to systemic, positive = towards observer (outflow), in contrast to typical
        # blueshifted absorption observational convention, in order to keep reasonable vrad bins the same
        p_local["vlos"] = haloVel[p_ind3] - sP.units.particleCodeVelocityToKms(vel_los)

        if proj2D:
            # down the barrel: consider only material between the observer and the galaxy, i.e. 'in front of'
            # the galaxy, whereby outflow moves towards the observer (positive sign in our convention), in contrast
            # to outflow on the other side which would be moving away from the observer
            los_dist_rel = haloPos[p_ind3] - p_local["Coordinates"][:, p_ind3]
            sP.correctPeriodicDistVecs(los_dist_rel)
            w = np.where(los_dist_rel < 0.0)  # note approximation of infinitely thin, aligned 'galaxy'
            p_local[massField][w] = 0.0

        assert particles["Masses"].shape == particles[massField].shape

        # compute weight, i.e. the halo-centric quantity 'radial mass flux'
        if rawMass:
            massflux = p_local[massField]  # codemass
        else:
            massflux = p_local["vrad"] * p_local[massField]  # codemass km/s

        # these multiply massflux from above
        if fluxKE:
            massflux *= p_local["vrad"] * p_local["vrad"] * 0.5  # 1/2 mv^2
        if fluxP:
            massflux *= p_local["vrad"]  # mv

        # velociy normalization: do now, such that it affects binning, but not mass flux calculations
        if v200norm:
            haloV200 = gc["vhalo_200"][subhaloID]
            assert np.isfinite(haloV200)
            p_local["vrad"] /= haloV200  # physical km/s -> unitless

        # compute rotation matrix for edge-on projection
        i0g = gc["SubhaloOffsetType"][subhaloID, ptNum_gas] - indRange["gas"][0]
        i0s = gc["SubhaloOffsetType"][subhaloID, ptNum_stars] - indRange["stars"][0]
        i1g = i0g + gc["SubhaloLenType"][subhaloID, ptNum_gas]
        i1s = i0s + gc["SubhaloLenType"][subhaloID, ptNum_stars]

        assert i0g >= 0 and i1g <= (indRange["gas"][1] - indRange["gas"][0] + 1)
        assert i0s >= 0 and i1s <= (indRange["stars"][1] - indRange["stars"][0] + 1)

        rHalf = gc["SubhaloHalfmassRadType"][subhaloID, sP.ptNum("stars")]
        shPos = gc["SubhaloPos"][subhaloID, :]

        gasLocal = {
            "Masses": gas["Masses"][i0g:i1g],
            "Coordinates": np.squeeze(gas["Coordinates"][i0g:i1g, :]),
            "StarFormationRate": gas["StarFormationRate"][i0g:i1g],
            "count": (i1g - i0g),
        }
        starsLocal = {
            "Masses": stars["Masses"][i0s:i1s],
            "Coordinates": np.squeeze(stars["Coordinates"][i0s:i1s, :]),
            "GFM_StellarFormationTime": stars["GFM_StellarFormationTime"][i0s:i1s],
            "count": (i1s - i0s),
        }

        I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
        rotMatrix = rotationMatricesFromInertiaTensor(I)["edge-on"]  # is edge-on-'largest'

        # do rotation and calculate theta angle
        projCen = gc["SubhaloPos"][subhaloID, :]
        p_pos = p_local["Coordinates"]
        p_pos_rot, _ = rotateCoordinateArray(sP, p_pos, rotMatrix, projCen, shiftBack=False)

        x_2d = p_pos_rot[:, 0]  # realize axes=[0,1]
        y_2d = p_pos_rot[:, 1]  # realize axes=[0,1]

        p_local["theta"] = np.arctan2(
            y_2d, x_2d
        )  # theta=0 along +x axis (major axis), theta=+/-pi=180 deg along -x axis (major axis)
        # theta=pi/2=90 deg along +y (minor axis), theta=-pi/2=-90 deg along -y (minor axis)

        # loop over binning configurations
        for j, binConfig in enumerate(binConfigs):
            # construct dense array of quantities to be binned
            sample = np.zeros((massflux.size, len(binConfig)), dtype="float32")
            for k, field in enumerate(binConfig):
                sample[:, k] = p_local[field]

            # multi-D histogram and stamp
            hh, _ = np.histogramdd(sample, bins=h_bins[j], density=False, weights=massflux)
            rr[j][i, ...] = hh

    # final unit handling: masses code->msun, and normalize out shell thicknesses
    if fluxKE:
        # msun/yr (km/s)^2 -> g/s (cm/s)^2, but work in unit system of [10^30 erg/s] to avoid overflows
        desc = "instantaneousEnergyOutflowRates [10^30 erg/s] (scope=%s)" % scope
        conv_fac = sP.units.Msun_in_g / sP.units.s_in_yr * sP.units.km_in_cm**2 / 1e30
    elif fluxP:
        # msun/yr (km/s) -> g*cm/s^2, but work in unit system of [10^30 g*cm/s^2] to avoid overflows
        desc = "instantaneousMomentumOutflowRates [10^30 g*cm/s^2] (scope=%s)" % scope
        conv_fac = sP.units.Msun_in_g / sP.units.s_in_yr * sP.units.km_in_cm / 1e30
    elif fluxMass:
        desc = "instantaneousMassOutflowRates [msun/yr] (scope=%s)" % scope
        conv_fac = 1.0
    elif rawMass:
        desc = "instantaneousMass [msun] (scope=%s)" % scope
        conv_fac = 1.0

    for i, binConfig in enumerate(binConfigs):
        rr[i] = sP.units.codeMassToMsun(rr[i])  # codemass -> msun
        rr[i] *= conv_fac  # handle fluxKE/fluxP unit conversions

        if not rawMass:
            # for fluxMass, fluxKE, or fluxP:
            rr[i] *= sP.units.kmS_in_kpcYr  # km/s -> kpc/yr
            for j in range(len(binConfig["rad"]) - 1):
                bin_width = binConfig["rad"][j + 1] - binConfig["rad"][j]  # pkpc
                rr[i][:, j, ...] /= bin_width  # msun kpc/yr -> msun/yr

            assert list(binConfig.keys()).index("rad") == 0  # otherwise we normalized along the wrong dimension

    # return quantities for save, as expected by load.auxCat()
    select = "subhalos, minStellarMass = %.2f (30pkpc values), [%s] only" % (minStellarMass, cenSatSelect)

    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "ptType": ptType.encode("ascii"),
        "subhaloIDs": subhaloIDsTodo,
    }

    for j, binConfig in enumerate(binConfigs):
        attrs["bins_%d" % j] = ".".join(binConfig.keys()).encode("ascii")
        for key in binConfig:
            attrs["bins_%d_%s" % (j, key)] = binConfig[key]

    return rr, attrs


def radialMassFluxes(
    sP,
    scope,
    ptType,
    thirdQuant=None,
    fourthQuant=None,
    firstQuant="rad",
    secondQuant="vrad",
    massField="Masses",
    selNum=None,
    fluxKE=False,
    fluxP=False,
    rawMass=False,
    inflow=False,
    v200norm=False,
):
    """Compute the total mass, energy, or momentum outflow or inflow fluxes (i.e. rates) in arbitrary bins.

    Principally, fluxes are computed in radial and radial velocity bins, but are optionally also binned
    by a third and fourth quantity (e.g. temperature, metallicity, density, or angular position theta).
    Loads the radialMassFlux aux catalogs and computes the total mass flux rate (msun/yr), energy
    flux rate (10^30 erg/s), or momentum flux rate (10^30 g*cm/s^2), according to fluxKE/fluxP.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      scope (str): 'subhalo', 'subhalo_wfuzz', or 'global' (see auxCat documentation).
      ptType (str): 'Gas' or 'Wind'.
      firstQuant (str): first quantity to bin by, typically 'rad'.
      secondQuant (str): second quantity to bin by, typically 'vrad'.
      thirdQuant (str or None): if provided, then should be one of temp,z_solar,numdens,theta, in which case the
        returned flux is not [Nsubs,nRad,nVradcuts] but instead [Nsubs,nRad,nVradcuts,nThirdQuantBins]. That is,
        the dependence on this parameter is given instead of integrated over, and the return has one more dimension.
      fourthQuant (str or None): as for thirdQuant, and then return is then 5D.
      massField (str): if not 'Masses' (total gas cell mass), then use instead this field and derive fluxes
        only for this mass subset (e.g. 'Mg II mass').
      selNum (int or None): if provided, then directly use this 'bin_X' from the auxCat,
        instead of searching for the appropriate one based on the quants.
      fluxKE (bool): compute energy loading factors instead of mass loadings.
      fluxP (bool): compute momentum loading factors instead of mass loadings.
      rawMass (bool): if True, then return total mass in each bin [msun], rather than mass flux rates [msun/yr].
      inflow (bool): if True, then compute inflow rates (vrad < 0) rather than outflow rates (vrad > 0).
      v200norm (bool): if True, then all velocities are binned in thresholds which are fractions of v200 of
        the halo, rather than in absolute physical [km/s] units.
    """
    import pickle

    validQuants = ["temp", "temp_sfcold", "z_solar", "numdens", "theta", "vrad", "vlos"]
    assert ptType in ["Gas", "Wind"]
    assert fluxKE + fluxP in [0, 1]  # at most one True

    if thirdQuant is not None:
        assert thirdQuant in validQuants
    if fourthQuant is not None:
        assert thirdQuant is not None
        assert fourthQuant in validQuants

    propStr = "MassFlux"
    if fluxKE:
        propStr = "EnergyFlux"
    if fluxP:
        propStr = "MomentumFlux"
    if rawMass:
        propStr = "Mass"

    # line-of-sight projected velocities?
    if thirdQuant == "vlos":
        propStr = "Mass2DProj"  #  load the proj2D auxCat()
        rad = "rad2d"
        vel = "vlos"
    else:
        rad = "rad"
        vel = "vrad"

    massStr = "_" + massField if massField != "Masses" else ""
    v200Str = "_v200norm" if v200norm else ""
    flowStr = "_inflow" if inflow else ""
    acField = "Subhalo_Radial%s_%s_%s%s%s" % (propStr, scope, ptType, massStr, v200Str)

    if ptType == "Gas":
        # will use this 3D histogram and collapse the temperature axis if thirdQuant == None
        dsetName = "%s.%s.temp" % (firstQuant, secondQuant)
        if thirdQuant is not None:
            dsetName = "%s.%s.%s" % (firstQuant, secondQuant, thirdQuant)
        if fourthQuant is not None:
            dsetName = "%s.%s.%s.%s" % (firstQuant, secondQuant, thirdQuant, fourthQuant)
    if ptType == "Wind":
        dsetName = "%s.%s" % (firstQuant, secondQuant)
        if thirdQuant is not None:
            dsetName = "%s.%s.%s" % (firstQuant, secondQuant, thirdQuant)

    selStr = "" if selNum is None else "_sel%d" % selNum
    cacheFile = sP.cachePath + "%s_%s-%s-%s-%s%s%s_%d.hdf5" % (
        acField,
        firstQuant,
        secondQuant,
        thirdQuant,
        fourthQuant,
        selStr,
        flowStr,
        sP.snap,
    )

    # overrides (after cache filename)
    dsetNameOrig = dsetName
    if dsetName == "%s.%s.%s" % (rad, rad, vel):
        # fine-grained vrad/vlos sampling (need to differentiate from '%s.%s.temp' case above, could be fixed)
        dsetName = "%s.%s" % (rad, vel)
    if dsetName == "%s.%s.%s" % (rad, vel, vel):
        # fine-grained vrad/vlos sampling (e.g. 1D plot of outflow rates vs vrad)
        dsetName = "%s.%s" % (rad, vel)
        thirdQuant = None

    # quick file cache, since these auxCat's are large
    if isfile(cacheFile):
        with h5py.File(cacheFile, "r") as f:
            flux = f["mdot"][()]
            mstar = f["mstar"][()]
            subhaloIDs = f["subhaloIDs"][()]
            binConfig = pickle.loads(f["binConfig"][()], encoding="latin1")
            numBins = pickle.loads(f["numBins"][()], encoding="latin1")
            vcut_vals = f["vcut_vals"][()]

        print("Loading from cached [%s]." % cacheFile)
        return flux, mstar, subhaloIDs, binConfig, numBins, vcut_vals

    # load radial mass fluxes auxCat
    ac = sP.auxCat(acField)

    # locate dataset we want and its binning configuration
    # todo: does not handle duplicates well, e.g. rad.vrad x2 for wind (fix)
    if selNum is None:
        for key, value in ac[acField + "_attrs"].items():
            if isinstance(value, str):
                if value == dsetName:
                    selNum = int(key.split("_")[1])
            if isinstance(value, bytes):
                if value.decode("ascii") == dsetName:
                    selNum = int(key.split("_")[1])

    assert selNum is not None, "Dataset not found."

    if ptType == "Wind" and selNum == 1 and dsetName == "rad.vrad":
        # want for mass loading plots, not for velocity plots
        print("NOTE: Switching selNum from 1 to 0 for correct Wind save (not high-res vrad histo).")
        selNum = 0

    binConfig = OrderedDict()
    numBins = OrderedDict()

    for field in dsetName.split("."):
        key = "bins_%d_%s" % (selNum, field)
        binConfig[field] = np.array(ac[acField + "_attrs"][key])
        numBins[field] = len(binConfig[field]) - 1

    if isinstance(ac[acField], list):
        dset = ac[acField][selNum]  # Gas
    else:
        # NOTE: bug! will hit this assert if pSplit was used to construct e.g. 'Subhalo_RadialMassFlux*'
        # because the concatenation of the partial files does not correctly preserve the multiple datasets
        dset = ac[acField]
        assert selNum == 0  # Wind, only 1 histogram and so not returned as a list

    assert dset.ndim == len(binConfig.keys()) + 1
    for i, field in enumerate(binConfig):
        assert dset.shape[i + 1] == numBins[field]  # first dimension is subhalos

    if secondQuant == vel and dsetNameOrig not in ["%s.%s.%s" % (rad, vel, vel)]:
        assert vel == "vrad"  # otherwise generalize this branch and consider what it means
        # standard case, i.e. rad.vrad or rad.vrad.* datasets

        # collapse (sum over) temperature bins, since we don't care here (dsetName == 'rad.vrad.temp')
        # note: for 'rad.vrad' this 2D hist only exists for Wind, so skip and go straight to vcut processing
        if ptType == "Gas" and thirdQuant is None and dsetNameOrig not in ["%s.%s" % (rad, vel)]:
            dset = np.sum(dset, axis=(3,))

        # now have a [nSubhalos,nRad,nVRad] shaped array, derive scalar quantities for each subhalo in auxCat
        #  --> in each radial shell, sum massflux over all temps
        if inflow:
            # inflow: sum for vrad <= vcut, taking vcut as all <= 0 vrad bins
            vcut_inds = np.where(binConfig["vrad"] <= 0.0)[0][1:]  # first is -np.inf
        else:
            # outflow: sum for vrad > vcut, taking vcut as all >= 0 vrad bins
            vcut_inds = np.where(binConfig["vrad"] >= 0.0)[0][:-1]  # last is np.inf

        if rawMass:
            vcut_inds = np.arange(binConfig["vrad"].size - 1)  # first is -inf (all), last is inf (not useful)
        vcut_vals = binConfig["vrad"][vcut_inds]

        # allocate
        flux_shape = [ac["subhaloIDs"].size, numBins["rad"], vcut_vals.size]
        if thirdQuant is not None:
            flux_shape.append(numBins[thirdQuant])
        if fourthQuant is not None:
            flux_shape.append(numBins[fourthQuant])

        flux = np.zeros(flux_shape, dtype="float32")

        for i, vcut_ind in enumerate(vcut_inds):
            # sum over all vrad > vcut bins for this vcut value
            if thirdQuant is None:
                # return is 3D
                if inflow:
                    flux[:, :, i] = np.sum(dset[:, :, :vcut_ind], axis=2)
                    import pdb

                    pdb.set_trace()
                else:
                    flux[:, :, i] = np.sum(dset[:, :, vcut_ind:], axis=2)
            else:
                if fourthQuant is None:
                    # return is 4D
                    if inflow:
                        flux[:, :, i, :] = np.sum(dset[:, :, :vcut_ind, :], axis=2)
                    else:
                        flux[:, :, i, :] = np.sum(dset[:, :, vcut_ind:, :], axis=2)
                else:
                    # return is 5D
                    if inflow:
                        flux[:, :, i, :, :] = np.sum(dset[:, :, :vcut_ind, :, :], axis=2)
                    else:
                        flux[:, :, i, :, :] = np.sum(dset[:, :, vcut_ind:, :, :], axis=2)

    else:
        # non-standard case, no manipulation or accumulation for vcut values
        flux = dset
        vcut_vals = np.zeros(1)

    # add a r=all bin by accumulating over all radial shells
    if firstQuant == rad:
        shape = np.array(flux.shape)
        assert shape[1] == numBins[rad]
        shape[1] += 1
        flux_rall = np.zeros(shape, dtype=flux.dtype)
        flux_rall[:, :-1, ...] = flux
        flux_rall[:, -1, ...] = np.sum(flux, axis=1)
        flux = flux_rall

    # load some group catalog properties and crossmatch for convenience
    gcIDs = np.arange(0, sP.numSubhalos)

    gc_inds, ac_inds = match(gcIDs, ac["subhaloIDs"])
    assert ac_inds.size == ac["subhaloIDs"].size

    mstar = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])
    mstar = mstar[gc_inds]

    # save cache
    with h5py.File(cacheFile, "w") as f:
        f["mdot"] = flux  # note: should be called 'flux', can be of mass, energy, or momentum
        f["mstar"] = mstar
        f["subhaloIDs"] = ac["subhaloIDs"]
        f["binConfig"] = np.void(pickle.dumps(binConfig))
        f["numBins"] = np.void(pickle.dumps(numBins))
        f["vcut_vals"] = vcut_vals

    print("Saved cached [%s]." % cacheFile)

    return flux, mstar, ac["subhaloIDs"], binConfig, numBins, vcut_vals


def massLoadingsSN(
    sP,
    pSplit,
    sfr_timescale=100,
    scope="SubfindWithFuzz",
    thirdQuant=None,
    massField="Masses",
    fluxKE=False,
    fluxP=False,
    v200norm=False,
):
    """Compute mass, energy, or momemtum loading factors.

    For every subhalo, compute one of:
    * mass loading factor ``eta_M^SN = eta_M = Mdot_out / SFR``.
    * energy loading factor ``eta_E^SN = Edot_out / Edot_SFR`` (if fluxKE == True)
    * momentum loading factor ``eta_P^SN = Pdot_out / Pdot_SFR`` (if fluxP == True)
    In the case of mass loadings, the outflow rates are derived using the instantaneous kinematic/flux method.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int][2]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      sfr_timescale (float): the star formation rates can be instantaneous or smoothed over some appropriate timescale.
      scope (str): analysis scope, can be one of 'subhalo', 'subhalo_wfuzz', or 'global' (slow).
      thirdQuant (str): if not None, then can be one of (temp,z_solar,numdens,theta), in which the
        dependence on this parameter is given instead of integrated over, and the return has one more dimension.
      massField (str): if not 'Masses' (total gas cell mass), then use instead this field and derive fluxes
        only for this mass subset (e.g. 'Mg II mass').
      fluxKE (bool): compute energy loading factors instead of mass loadings.
      fluxP (bool): compute momentum loading factors instead of mass loadings.
      v200norm (bool): if True, then all velocities are binned in thresholds which are fractions of v200 of
        the halo, rather than in absolute physical [km/s] units.

    Returns:
      tuple: a 2-tuple composed of:

      - **result** (ndarray[float][nSubsInAuxCat,nRadBins,nVradCuts]): 3d array,
        containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    assert sfr_timescale in [0, 10, 50, 100]  # Myr
    assert fluxKE + fluxP in [0, 1]  # at most one True
    assert pSplit is None  # not supported

    # todo: could also compute a 'blackhole mass loading' value by considering the BH Mdot instead of the SFR.
    #       instead of outflow_rate/BH_Mdot, maybe outflow_rate/(BH_dE/c^2)

    params = {"thirdQuant": thirdQuant, "massField": massField, "fluxKE": fluxKE, "fluxP": fluxP, "v200norm": v200norm}

    # load fluxes of gas cells as well as wind-phase gas particles
    gas_mdot, _, ac_subhaloIDs, gas_binconf, gas_nbins, gas_vcutvals = radialMassFluxes(sP, scope, "Gas", **params)

    if massField == "Masses":
        wind_mdot, _, wind_subids, _, _, _ = radialMassFluxes(sP, scope, "Wind", **params)

        assert np.array_equal(ac_subhaloIDs, wind_subids)
        assert gas_mdot.shape == wind_mdot.shape

        # sum the two
        outflow_rates = gas_mdot + wind_mdot
    else:
        # gas mass sub-component (phase), so skip wind (i.e. current assumption: wind particles have no ionic mass)
        outflow_rates = gas_mdot

    # prepare metadata
    binConfig = "none" if thirdQuant is None else gas_binconf[thirdQuant]
    numBins = "none" if thirdQuant is None else gas_nbins[thirdQuant]

    attrs = {"binConfig": binConfig, "numBins": numBins, "rad": gas_binconf["rad"], "vcut_vals": gas_vcutvals}

    # cross-match with group catalog
    gcIDs = np.arange(0, sP.numSubhalos)
    gc_inds, ac_inds = match(gcIDs, ac_subhaloIDs)
    assert ac_inds.size == ac_subhaloIDs.size

    select = "All Subfind subhalos (the subset which have computed radial mass fluxes)."

    if not fluxKE and not fluxP:
        # load star formation rates with the requested temporal smoothing
        sfr, _, _, _ = sP.simSubhaloQuantity("sfr_30pkpc_%dmyr" % sfr_timescale)  # msun/yr
        norm_quant = sfr[gc_inds]
        desc = "Mass loading, using instantaneous gas fluxes [%s] and 30pkpc %d Myr SFRs." % (massField, sfr_timescale)

    if fluxKE:
        # load energy injection rates of the star-forming gas according to the wind model
        outflow_rates = outflow_rates.astype("float64") * 1e30  # unit conversion: [10^30 erg/s] -> [erg/s]
        wind_dEdt, _, _, _ = sP.simSubhaloQuantity("wind_dEdt")
        norm_quant = wind_dEdt[gc_inds]
        desc = "Energy loading, using instantaneous gas fluxes [%s], 2rhalfstars wind_dEdt." % (massField)

    if fluxP:
        # load momentum injection rates of the star-forming gas (also at injection)
        outflow_rates = outflow_rates.astype("float64") * 1e30  # unit conversion: [10^30 g*cm/s^2] -> [g*cm/s^2]
        wind_dPdt, _, _, _ = sP.simSubhaloQuantity("wind_dPdt")
        norm_quant = wind_dPdt[gc_inds]
        desc = "Momentum loading, using instantaneous gas fluxes [%s], 2rhalfstars wind_dPdt." % (massField)

    # compute eta [linear dimensionless], simultaneously for all radial bins and vrad cuts
    eta = outflow_rates
    w = np.where(norm_quant > 0.0)[0]
    eta[w, :, :] /= norm_quant[w, np.newaxis, np.newaxis]

    # set zero in the case of zero outflow, and NaN in the case of zero SFR/Edot_SN/Pdot_SN
    w = np.where(outflow_rates == 0.0)
    eta[w] = 0.0
    w = np.where(norm_quant == 0.0)[0]
    eta[w, :, :] = np.nan

    # return
    attrs = dict(attrs)
    attrs["Description"] = desc.encode("ascii")
    attrs["Selection"] = select.encode("ascii")
    attrs["subhaloIDs"] = ac_subhaloIDs
    return eta, attrs


def outflowVelocities(
    sP,
    pSplit,
    percs=(25, 50, 75, 90, 95, 99),
    scope="SubfindWithFuzz",
    massField="Masses",
    proj2D=False,
    v200norm=False,
):
    """Compute an outflow velocity for every subhalo.

    Return has shape [nSubsInAuxCat,nRadBins+1,nPercentiles], where the final radial bin considers gas at all radii
    (in the scope). If proj2D is True, then we compute the line-of-sight projected velocities.
    """
    assert pSplit is None  # not supported

    if not proj2D:
        q1 = "rad"
        q2 = "vrad"
        q3 = "vrad"
    else:
        q1 = "rad2d"
        q2 = "vlos"
        q3 = "vlos"

    # load fluxes of gas cells as well as wind-phase gas particles
    gas_mdot, _, ac_subhaloIDs, gas_binconf, gas_nbins, _ = radialMassFluxes(
        sP, scope, "Gas", firstQuant=q1, secondQuant=q2, thirdQuant=q3, massField=massField, v200norm=v200norm
    )

    if massField == "Masses":
        wind_mdot, _, wind_subids, _, _, _ = radialMassFluxes(
            sP,
            scope,
            "Wind",
            firstQuant=q1,
            secondQuant=q2,
            thirdQuant=q3,
            selNum=1,
            massField=massField,
            v200norm=v200norm,
        )

        assert np.array_equal(ac_subhaloIDs, wind_subids)

        # sum the two = [msun/yr] ~ [msun] since the normalization is everywhere constant
        outflow_mass = gas_mdot + wind_mdot
    else:
        # gas mass sub-component (phase), so skip wind (i.e. current assumption: wind particles have no ionic mass)
        outflow_mass = gas_mdot

    # get radial velocity bins, restrict to vrad>0 (outflow)
    binConfig = gas_binconf[q3]
    assert binConfig.size == outflow_mass.shape[2] + 1

    minPosInd = np.where(binConfig >= 0.0)[0].min()
    outflow_mass = outflow_mass[:, :, minPosInd:]

    vradBinEdges = binConfig[minPosInd:]
    vradBins = (vradBinEdges[1:] + vradBinEdges[:-1]) / 2
    numVradBins = gas_nbins[q3] - minPosInd
    assert numVradBins == outflow_mass.shape[2]

    vradBinsRev = vradBins[::-1]
    numRadBins = outflow_mass.shape[1]

    # todo: finish

    # answer for v_{out,90} is 'radial velocity above which 10% of the outflowing mass is moving'
    perc_fracs = 1.0 - np.array(percs) / 100.0

    outflow_mass_rad_tot = np.sum(outflow_mass, axis=2)
    outflow_mass_vrad_tot = np.sum(outflow_mass, axis=1)
    outflow_mass_tot = np.sum(outflow_mass, axis=(1, 2))

    # allocate
    nSubs = outflow_mass.shape[0]
    vout = np.zeros((nSubs, numRadBins + 1, len(percs)), dtype="float32")  # [nSubs,nRadBins]
    vout.fill(np.nan)

    # loop over subhalos individually
    for i in range(nSubs):
        # cumulative sum from high to low vrad (corresponding now to vradBinsRev)
        cum = np.cumsum(outflow_mass[i, :, ::-1], axis=1)

        # loop over radial bins
        for j in range(numRadBins):
            if outflow_mass_rad_tot[i, j] == 0.0:
                continue

            # normalize such that integral is unity
            cum_rad = cum[j, :] / outflow_mass_rad_tot[i, j]

            # make interpolator, flipping 'axes' such that x=cum mass, y=vrad
            f = interp1d(cum_rad, vradBinsRev, kind="linear", fill_value="extrapolate")

            vout[i, j, :] = f(perc_fracs)

        # once at the end for all radial bins combined
        if outflow_mass_tot[i] == 0.0:
            continue

        cum = np.cumsum(outflow_mass_vrad_tot[i, ::-1]) / outflow_mass_tot[i]
        f = interp1d(cum, vradBinsRev, kind="linear", fill_value="extrapolate")
        vout[i, j + 1, :] = f(perc_fracs)

    # return
    attrs = {
        "vradBinEdges": vradBinEdges,
        "vradBins": vradBins,
        "numVradBins": numVradBins,
        "rad": gas_binconf[q1],
        "percs": percs,
    }
    desc = "Outflow velocities, computed using instantaneous gas [%s] fluxes." % massField
    if proj2D:
        desc += " Line-of-sight 1D projected velocities (rad are 2D annular apertures), down the barrel treatment."
    select = "All Subfind subhalos (the subset which have computed radial mass fluxes)."

    attrs = dict(attrs)
    attrs["Description"] = desc.encode("ascii")
    attrs["Selection"] = select.encode("ascii")
    attrs["subhaloIDs"] = ac_subhaloIDs
    return vout, attrs
