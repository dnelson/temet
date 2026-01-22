"""
Cosmological simulations - auxiliary (whole box-based) catalogs for additional derived properties.
"""

import numpy as np

from ..cosmo import hydrogen
from ..cosmo.cloudy import cloudyIon
from ..cosmo.hydrogen import calculateCDDF
from ..util.helper import numPartToChunkLoadSize, reportMemory


def wholeBoxColDensGrid(sP, pSplit, species, gridSize=None, onlySFR=False, allSFR=False):
    """Compute a 2D grid of gas column densities [cm^-2] covering the entire simulation box.

    For example to derive the neutral hydrogen CDDF. Strategy is a chunked load of the snapshot files,
    for each using SPH-kernel deposition to distribute the mass of the requested species (e.g. HI,
    CIV) in all gas cells onto the grid.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      species (str): the gas species/sub-component to grid.
      gridSize (int or None): if specified, override the default grid cell size [code units].
      onlySFR (bool): if True, only include SFR > 0 gas cells.
      allSFR (bool): if True, assume that all gas with SFR > 0 gas a unity fraction of the given
        species. E.g. for H2, assume star formation gas cells are entirely made of molecular hydrogen.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 2d array, gridded column densities.
      - **attrs** (dict): metadata.
    """
    from ..util.sphMap import sphMapWholeBox

    assert pSplit is None  # not implemented

    # hard-coded parameters for whole box (halo-independent) computations, could generalize
    boxGridSizeHI = 1.5  # code units, e.g. ckpc/h
    boxGridSizeMetals = 5.0  # code units, e.g. ckpc/h

    # adjust projection depth
    projDepthCode = sP.boxSize

    if "_depth10" in species:
        projDepthCode = 10000.0  # 10 cMpc/h
        species = species.split("_depth10")[0]
    if "_depth20" in species:
        projDepthCode = 20000.0  # 20 cMpc/h
        species = species.split("_depth20")[0]
    if "_depth5" in species:
        projDepthCode = 5000.0  # 5 cMpc/h
        species = species.split("_depth5")[0]
    if "_depth1" in species:
        projDepthCode = 1000.0  # 1 cMpc/h
        species = species.split("_depth1")[0]
    if "_depth125" in species:
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac)  # 12.5 cMpc
        species = species.split("_depth125")[0]

    # check
    hDensSpecies = ["HI", "HI_noH2"]
    preCompSpecies = ["MH2_BR", "MH2_GK", "MH2_KMT", "MHI_BR", "MHI_GK", "MHI_KMT", "MH2_GD14", "MH2_GK11", "MH2_K13",
                      "MH2_S14", "MHI_GD14", "MHI_GK11", "MHI_K13", "MHI_S14"]  # fmt: skip
    zDensSpecies = ["O VI", "O VI 10", "O VI 25", "O VI solar", "O VII", "O VIII", "O VII solarz", "O VII 10 solarz"]

    if species not in hDensSpecies + zDensSpecies + preCompSpecies + ["Z"]:
        raise Exception("Not implemented.")

    # config
    h = sP.snapshotHeader()
    nChunks = numPartToChunkLoadSize(h["NumPart"][sP.ptNum("gas")])
    axes = [0, 1]  # x,y projection

    # info
    h = sP.snapshotHeader()

    if species in zDensSpecies:
        boxGridSize = boxGridSizeMetals
    else:
        boxGridSize = boxGridSizeHI

    # adjust grid size
    if species in ["O VI 10", "O VII 10 solarz"]:
        boxGridSize = 10.0  # test, x2 bigger
    if species == "O VI 25":
        boxGridSize = 2.5  # test, x2 smaller

    if gridSize is not None:
        print(" Seting gridSize = %f [code units]" % gridSize)
        boxGridSize = gridSize

    boxGridDim = round(sP.boxSize / boxGridSize)
    chunkSize = int(h["NumPart"][sP.ptNum("gas")] / nChunks)

    if species in hDensSpecies + zDensSpecies + preCompSpecies:
        desc = "Square grid of integrated column densities of [" + species + "] in units of [cm^-2]. "
    if species == "Z":
        desc = "Square grid of mean gas metallicity in units of [log solar]."

    if species == "HI":
        desc += "Atomic only, H2 calculated and removed."
    if species == "HI_noH2":
        desc += "All neutral hydrogen included, any contribution of H2 ignored."

    select = "Grid dimensions: %dx%d pixels (cell size = %06.2f codeunits) along axes=[%d,%d]." % (
        boxGridDim,
        boxGridDim,
        boxGridSize,
        axes[0],
        axes[1],
    )

    print(" " + desc)
    print(" " + select)
    print(" Total # Snapshot Load Chunks: " + str(nChunks) + " (" + str(chunkSize) + " cells per load)")

    # specify needed data load, and allocate accumulation array(s)
    if species in hDensSpecies:
        fields = ["Coordinates", "Masses", "Density", "metals_H", "NeutralHydrogenAbundance"]

        r = np.zeros((boxGridDim, boxGridDim), dtype="float32")

    if species in zDensSpecies:
        fields = ["Coordinates", "Masses", "Density"]

        r = np.zeros((boxGridDim, boxGridDim), dtype="float32")

    if species in preCompSpecies:
        fields = ["Coordinates", "Masses", species]

        r = np.zeros((boxGridDim, boxGridDim), dtype="float32")

    if species == "Z":
        fields = ["Coordinates", "Masses", "Density", "GFM_Metallicity"]

        rM = np.zeros((boxGridDim, boxGridDim), dtype="float32")
        rZ = np.zeros((boxGridDim, boxGridDim), dtype="float32")

    if onlySFR or allSFR:
        fields += ["StarFormationRate"]

    # determine projection depth fraction
    boxWidthFrac = projDepthCode / sP.boxSize

    # loop over chunks (we are simply accumulating, so no need to load everything at once)
    for i in np.arange(nChunks):
        # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
        indRange = [i * chunkSize, (i + 1) * chunkSize - 1]
        if i == nChunks - 1:
            indRange[1] = int(h["NumPart"][sP.ptNum("gas")] - 1)
        print("  [%2d] %9d - %d" % (i, indRange[0], indRange[1]), reportMemory())

        # load
        gas = sP.snapshotSubsetP("gas", fields, indRange=indRange)

        # calculate smoothing size (V = 4/3*pi*h^3)
        if "Masses" in gas and "Density" in gas:
            vol = gas["Masses"] / gas["Density"]
            hsml = (vol * 3.0 / (4 * np.pi)) ** (1.0 / 3.0)
        else:
            # equivalent calculation
            hsml = sP.snapshotSubsetP("gas", "cellsize", indRange=indRange)

        hsml = hsml.astype("float32")

        # modifications
        if onlySFR:
            # only SFR>0 gas contributes
            w = np.where(gas["StarFormationRate"] == 0)
            gas[species][w] = 0.0
            gas["Masses"][w] = 0.0

        if allSFR:
            # SFR>0 gas has a fraction=1 of the given species
            assert species in preCompSpecies  # otherwise handle

        if species in hDensSpecies:
            # calculate atomic hydrogen mass (HI) or total neutral hydrogen mass (HI+H2) [10^10 Msun/h]
            atomic = species == "HI" or species == "HI2" or species == "HI3"
            neutral = species == "HI_noH2"
            mHI = hydrogen.hydrogenMass(gas, sP, atomic=atomic, totalNeutral=neutral)

            # simplified models (difference is quite small in CDDF)
            # mHI = gas['Masses'] * gas['GFM_Metals'] * gas['NeutralHydrogenAbundance']
            # mHI = gas['Masses'] * sP.units.hydrogen_massfrac * gas['NeutralHydrogenAbundance']

            # grid gas mHI using SPH kernel, return in units of [10^10 Msun * h / ckpc^2]
            ri = sphMapWholeBox(
                pos=gas["Coordinates"],
                hsml=hsml,
                mass=mHI,
                quant=None,
                axes=axes,
                nPixels=boxGridDim,
                sP=sP,
                colDens=True,
                sliceFac=boxWidthFrac,
            )

            r += ri

        if species in zDensSpecies:
            # calculate metal ion mass, and grid column densities
            element = species.split()[0]
            ionNum = species.split()[1]

            ion = cloudyIon(sP, el=element, redshiftInterp=True)

            if species[-5:] == "solar":
                # assume solar abundances
                mMetal = gas["Masses"] * ion.calcGasMetalAbundances(
                    sP, element, ionNum, indRange=indRange, solarAbunds=True
                )
            elif species[-6:] == "solarz":
                # assume solar abundances and solar metallicity
                mMetal = gas["Masses"] * ion.calcGasMetalAbundances(
                    sP, element, ionNum, indRange=indRange, solarAbunds=True, solarMetallicity=True
                )
            else:
                # default (use cached ion masses)
                mMetal = sP.snapshotSubset("gas", "%s %s mass" % (element, ionNum), indRange=indRange)

            # determine projection depth fraction
            boxWidthFrac = projDepthCode / sP.boxSize

            # project
            ri = sphMapWholeBox(
                pos=gas["Coordinates"],
                hsml=hsml,
                mass=mMetal,
                quant=None,
                axes=axes,
                nPixels=boxGridDim,
                sP=sP,
                colDens=True,
                sliceFac=boxWidthFrac,
            )

            r += ri

        if species in preCompSpecies:
            # anything directly loaded from the snapshots, return in units of [10^10 Msun * h / ckpc^2]
            nThreads = 8
            if boxGridDim > 60000:
                nThreads = 4
            if boxGridDim > 100000:
                nThreads = 2

            if allSFR:
                w = np.where(gas["StarFormationRate"] > 0)
                gas[species][w] = gas["Masses"][w]

            ri = sphMapWholeBox(
                pos=gas["Coordinates"],
                hsml=hsml,
                mass=gas[species],
                quant=None,
                axes=axes,
                nPixels=boxGridDim,
                sP=sP,
                colDens=True,
                nThreads=nThreads,
                sliceFac=boxWidthFrac,
            )

            r += ri

        if species == "Z":
            # grid total gas mass using SPH kernel, return in units of [10^10 Msun / h]
            rMi = sphMapWholeBox(
                pos=gas["Coordinates"],
                hsml=hsml,
                mass=gas["Masses"],
                quant=None,
                axes=axes,
                nPixels=boxGridDim,
                sP=sP,
                colDens=False,
                sliceFac=boxWidthFrac,
            )

            # grid total gas metal mass
            mMetal = gas["Masses"] * gas["GFM_Metallicity"]

            rZi = sphMapWholeBox(
                pos=gas["Coordinates"],
                hsml=hsml,
                mass=mMetal,
                quant=None,
                axes=axes,
                nPixels=boxGridDim,
                sP=sP,
                colDens=False,
                sliceFac=boxWidthFrac,
            )

            rM += rMi
            rZ += rZi

    # finalize
    if species in hDensSpecies + zDensSpecies + preCompSpecies:
        # column density: convert units from [code column density, above] to [H atoms/cm^2] and take log
        rr = sP.units.codeColDensToPhys(r, cgs=True, numDens=True)

        if species in zDensSpecies:
            ion = cloudyIon(None)
            rr /= ion.atomicMass(species.split()[0])  # [H atoms/cm^2] to [ions/cm^2]

        if "MH2" in species:
            print("Converting [H atoms/cm^2] to [H molecules/cm^2].")
            rr /= 2  # [H atoms/cm^2] to [H molecules/cm^2]

        rr = np.log10(rr)

    if species == "Z":
        # metallicity: take Z = mass_tot/mass_gas for each pixel, normalize by solar, take log
        rr = rZ / rM
        rr = np.log10(rr / sP.units.Z_solar)

    attrs = {"Description": desc.encode("ascii"), "Selection": select.encode("ascii")}

    return rr, attrs


def wholeBoxCDDF(sP, pSplit, species, gridSize=None, omega=False):
    """Compute the column density distribution function (CDDF, i.e. histogram) given a full box column density grid.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      species (str): the gas species/sub-component to grid.
      gridSize (int or None): if specified, override the default grid cell size [code units].
      omega (bool): if True, then instead compute the single number
        Omega_species = rho_species / rho_crit,0.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, the CDDF.
      - **attrs** (dict): metadata.

    Note:
      There is unfortunate duplication/lack of generality between this function and
      :py:func:`wholeBoxColDensGrid` (e.g. in the projection depth specification) which is always called.
      To define a new catalog for a CDDF, it must be specified twice: the actual grid, and the CDDF.
    """
    assert pSplit is None  # not implemented

    if omega:
        mass = sP.snapshotSubset("gas", species + " mass")
        code_dens = np.sum(mass) / sP.boxSize**3  # code units
        rr = sP.units.codeDensToCritRatio(code_dens, redshiftZero=True)
        desc = "Omega_%s = (rho_%s / rho_crit,z=0)" % (species, species)
        attrs = {"Description": desc.encode("ascii"), "Selection": "All gas cells in box.".encode("ascii")}
        return rr, attrs

    # config
    binSize = 0.1  # log cm^-2
    binMinMax = [11.0, 28.0]  # log cm^-2

    desc = "Column density distribution function (CDDF) for [" + species + "]. "
    desc += "Return has shape [2,nBins] where the first slice gives n [cm^-2], the second fN [cm^-2]."
    select = "Binning min: [%g] max: [%g] size: [%g]." % (binMinMax[0], binMinMax[1], binSize)

    # load
    acField = "Box_Grid_n" + species
    if gridSize is not None:
        acField += "_gridSize=%.1f" % gridSize

    ac = sP.auxCat(fields=[acField])

    # depth
    projDepthCode = sP.boxSize
    if "_depth1" in species:
        projDepthCode = 1000.0
    if "_depth10" in species:  # must be after '_depth1'...
        projDepthCode = 10000.0
    if "_depth5" in species:
        projDepthCode = 5000.0
    if "_depth20" in species:
        projDepthCode = 20000.0
    if "_depth125" in species:
        projDepthCode = sP.units.physicalKpcToCodeLength(12500.0 * sP.units.scalefac)
    assert not ("_depth" in species and projDepthCode == sP.boxSize)  # handle

    # calculate
    depthFrac = projDepthCode / sP.boxSize

    fN, n = calculateCDDF(ac[acField], binMinMax[0], binMinMax[1], binSize, sP, depthFrac=depthFrac)

    rr = np.vstack((n, fN))
    attrs = {"Description": desc.encode("ascii"), "Selection": select.encode("ascii")}

    return rr, attrs
