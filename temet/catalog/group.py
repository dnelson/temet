"""
Cosmological simulations - auxiliary (group-based) catalogs for additional derived properties.
"""

import numpy as np

from ..util.helper import pSplitRange


def fofRadialSumType(sP, pSplit, ptProperty, rad, method="B", ptType="all"):
    """Compute total/sum of a particle property (e.g. mass) for particles within an available SO radius.

    Use one of four methods:

    * Method A: do individual halo loads per halo, one loop over all halos.
    * Method B: do a full snapshot load per type, then halo loop and slice per FoF, to cut down on I/O ops.
    * Method C: per type: full snapshot load, spherical aperture search per FoF (brute-force global).
    * Method D: per type: full snapshot load, construct octtree, spherical aperture search per FoF (global).

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      rad (str): a radius available in the group catalog, e.g. 'Group_R_Crit200' [code units].
      method (str): see above. **Note!** Methods A and B restrict this calculation to FoF particles only,
        whereas method C does a full particle search over the entire box in order to compute the total/sum
        for each FoF halo.
      ptType (str): if 'all', then sum over all types (dm, gas, and stars), otherwise just for the single
        specified type.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, result for each subhalo.
      - **attrs** (dict): metadata.

    Warning:
      This was an early example of a catalog generating function, and is left mostly for reference as a
      particularly simple example. In practice, its functionality can be superseded by
      :py:func:`subhaloRadialReduction`.
    """
    # config
    if ptType == "all":
        # (0=dm, 1=gas+wind, 2=stars-wind), instead of making a half-empty Nx6 array
        ptSaveTypes = {"dm": 0, "gas": 1, "stars": 2}
        ptLoadTypes = {"dm": sP.ptNum("dm"), "gas": sP.ptNum("gas"), "stars": sP.ptNum("stars")}
        assert ptProperty == "Masses"  # not yet implemented any other fields which apply to all partTypes
    else:
        ptSaveTypes = {ptType: 0}
        ptLoadTypes = {ptType: sP.ptNum(ptType)}

    desc = "Mass by type enclosed within a radius of " + rad + " (only FoF particles included). "
    desc += "Type indices: " + " ".join([t + "=" + str(i) for t, i in ptSaveTypes.items()]) + "."
    select = "All FoF halos."

    # load group information
    gc = sP.groupCat(fieldsHalos=["GroupPos", "GroupLen", "GroupLenType", rad])
    gc["GroupOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]

    h = sP.snapshotHeader()

    nGroupsTot = sP.numHalos
    haloIDsTodo = np.arange(nGroupsTot, dtype="int32")

    # if no task parallelism (pSplit), set default particle load ranges
    indRange = sP.subhaloIDsToBoundingPartIndices(haloIDsTodo, groups=True)

    if pSplit is not None:
        ptSplit = ptType if ptType != "all" else "gas"

        # subdivide the global [variable ptType!] particle set, then map this back into a division of
        # group IDs which will be better work-load balanced among tasks
        gasSplit = pSplitRange(indRange[ptSplit], pSplit[1], pSplit[0])

        invGroups = sP.inverseMapPartIndicesToHaloIDs(gasSplit, ptSplit, debug=True)

        if pSplit[0] == pSplit[1] - 1:
            invGroups[1] = nGroupsTot
        else:
            assert invGroups[1] != -1

        haloIDsTodo = np.arange(invGroups[0], invGroups[1])
        indRange = sP.subhaloIDsToBoundingPartIndices(haloIDsTodo, groups=True)

    nHalosDo = len(haloIDsTodo)

    # info
    print(" " + desc)
    print(" Total # Halos: %d, processing [%d] halos..." % (nGroupsTot, nHalosDo))

    # allocate return, NaN indicates not computed
    r = np.zeros((nHalosDo, len(ptSaveTypes.keys())), dtype="float32")
    r.fill(np.nan)

    # square radii, and use sq distance function
    gc[rad] = gc[rad] * gc[rad]

    if method == "A":
        # loop over all halos
        for i, haloID in enumerate(haloIDsTodo):
            if nHalosDo >= 50 and i % int(nHalosDo / 50) == 0 and i <= nHalosDo:
                print(" %4.1f%%" % (float(i + 1) * 100.0 / nHalosDo))

            # For each type:
            #   1. Load pos (DM), pos/mass (gas), pos/mass/sftime (stars) for this FoF.
            #   2. Calculate periodic distances, (DM: count num within rad, sum massTable*num)
            #      gas/stars: sum mass of those within rad (gas = gas+wind, stars=real stars only)

            # DM
            if "dm" in ptLoadTypes:
                dm = sP.snapshotSubsetP(partType="dm", fields=["pos"], haloID=haloID, sq=False)

                if dm["count"]:
                    rDM = sP.periodicDistsSq(gc["GroupPos"][haloID, :], dm["Coordinates"])
                    wDM = np.where(rDM <= gc[rad][haloID])

                    r[i, ptSaveTypes["dm"]] = len(wDM[0]) * h["MassTable"][ptLoadTypes["dm"]]

            # GAS
            if "gas" in ptLoadTypes:
                gas = sP.snapshotSubsetP(partType="gas", fields=["pos", ptProperty], haloID=haloID)
                assert gas[ptProperty].ndim == 1

                if gas["count"]:
                    rGas = sP.periodicDistsSq(gc["GroupPos"][haloID, :], gas["Coordinates"])
                    wGas = np.where(rGas <= gc[rad][haloID])

                    r[i, ptSaveTypes["gas"]] = np.sum(gas[ptProperty][wGas])

            # STARS
            if "stars" in ptLoadTypes:
                stars = sP.snapshotSubsetP(partType="stars", fields=["pos", "sftime", ptProperty], haloID=haloID)
                assert stars[ptProperty].ndim == 1

                if stars["count"]:
                    rStars = sP.periodicDistsSq(gc["GroupPos"][haloID, :], stars["Coordinates"])
                    wWind = np.where((rStars <= gc[rad][haloID]) & (stars["GFM_StellarFormationTime"] < 0.0))
                    wStars = np.where((rStars <= gc[rad][haloID]) & (stars["GFM_StellarFormationTime"] >= 0.0))

                    r[i, ptSaveTypes["gas"]] += np.sum(stars[ptProperty][wWind])
                    r[i, ptSaveTypes["stars"]] = np.sum(stars[ptProperty][wStars])

    if method == "B":
        # (A): DARK MATTER
        if "dm" in ptLoadTypes:
            print(" [DM]")
            dm = sP.snapshotSubsetP(partType="dm", fields=["pos"], sq=False, indRange=indRange["dm"])

            if ptProperty == "Masses":
                dm[ptProperty] = np.zeros(dm["count"], dtype="float32") + h["MassTable"][ptLoadTypes["dm"]]
            else:
                dm[ptProperty] = sP.snapshotSubsetP(partType="dm", fields=ptProperty, haloSubset=True)

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if nHalosDo >= 10 and i % int(nHalosDo / 10) == 0 and i <= nHalosDo:
                    print("  %4.1f%%" % (float(i + 1) * 100.0 / nHalosDo))

                # slice starting/ending indices for dm local to this FoF
                i0 = gc["GroupOffsetType"][haloID, ptLoadTypes["dm"]] - indRange["dm"][0]
                i1 = i0 + gc["GroupLenType"][haloID, ptLoadTypes["dm"]]

                assert i0 >= 0 and i1 <= (indRange["dm"][1] - indRange["dm"][0] + 1)

                if i1 == i0:
                    continue  # zero length of this type

                rr = sP.periodicDistsSq(gc["GroupPos"][haloID, :], dm["Coordinates"][i0:i1, :])
                ww = np.where(rr <= gc[rad][haloID])

                r[i, ptSaveTypes["dm"]] = np.sum(dm[ptProperty][i0:i1][ww])
            del dm

        # (B): GAS
        if "gas" in ptLoadTypes:
            print(" [GAS]")
            gas = sP.snapshotSubsetP(partType="gas", fields=["pos"], sq=False, indRange=indRange["gas"])
            gas[ptProperty] = sP.snapshotSubsetP(partType="gas", fields=ptProperty, indRange=indRange["gas"])
            assert gas[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if nHalosDo >= 10 and i % int(nHalosDo / 10) == 0 and i <= nHalosDo:
                    print("  %4.1f%%" % (float(i + 1) * 100.0 / nHalosDo))

                # slice starting/ending indices for gas local to this FoF
                i0 = gc["GroupOffsetType"][haloID, ptLoadTypes["gas"]] - indRange["gas"][0]
                i1 = i0 + gc["GroupLenType"][haloID, ptLoadTypes["gas"]]

                assert i0 >= 0 and i1 <= (indRange["gas"][1] - indRange["gas"][0] + 1)

                if i1 == i0:
                    continue  # zero length of this type

                rr = sP.periodicDistsSq(gc["GroupPos"][haloID, :], gas["Coordinates"][i0:i1, :])
                ww = np.where(rr <= gc[rad][haloID])

                r[i, ptSaveTypes["gas"]] = np.sum(gas[ptProperty][i0:i1][ww])
            del gas

        # (C): STARS
        if "stars" in ptLoadTypes:
            print(" [STARS]")
            stars = sP.snapshotSubsetP(partType="stars", fields=["pos", "sftime"], sq=False, indRange=indRange["stars"])
            stars[ptProperty] = sP.snapshotSubsetP(partType="stars", fields=ptProperty, indRange=indRange["stars"])
            assert stars[ptProperty].ndim == 1

            # loop over halos
            for i, haloID in enumerate(haloIDsTodo):
                if nHalosDo >= 10 and i % int(nHalosDo / 10) == 0 and i <= nHalosDo:
                    print("  %4.1f%%" % (float(i + 1) * 100.0 / nHalosDo))

                # slice starting/ending indices for stars local to this FoF
                i0 = gc["GroupOffsetType"][haloID, ptLoadTypes["stars"]] - indRange["stars"][0]
                i1 = i0 + gc["GroupLenType"][haloID, ptLoadTypes["stars"]]

                assert i0 >= 0 and i1 <= (indRange["stars"][1] - indRange["stars"][0] + 1)

                if i1 == i0:
                    continue  # zero length of this type

                rr = sP.periodicDistsSq(gc["GroupPos"][haloID, :], stars["Coordinates"][i0:i1, :])
                wWind = np.where((rr <= gc[rad][haloID]) & (stars["GFM_StellarFormationTime"][i0:i1] < 0.0))
                wStars = np.where((rr <= gc[rad][haloID]) & (stars["GFM_StellarFormationTime"][i0:i1] >= 0.0))

                r[i, ptSaveTypes["gas"]] += np.sum(stars[ptProperty][i0:i1][wWind])
                r[i, ptSaveTypes["stars"]] = np.sum(stars[ptProperty][i0:i1][wStars])

    if method == "C":
        # proceed with loads as in B and simply do rr calculation against all particles
        raise Exception("Not implemented.")

    if method == "D":
        # use our numba tree implementation for 3d ball searches (todo)
        raise Exception("Not implemented.")

    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "ptType": ptType.encode("ascii"),
        "ptProperty": ptProperty.encode("ascii"),
        "subhaloIDs": haloIDsTodo,
    }

    r = np.squeeze(r)

    return r, attrs
