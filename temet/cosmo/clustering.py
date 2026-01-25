"""
Galaxy clustering statistics, e.g. two-point correlation functions.
"""

import glob
import time
from collections import OrderedDict
from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np

from ..cosmo.color import loadSimGalColors
from ..util.helper import pSplitRange
from ..util.tpcf import quantReductionInRad, tpcf


def _covar_matrix(x_avg, x_subs):
    """Compute covariance matrix from a jackknife set of samplings x_subs.

    Slightly different than normal we use x_avg from the input (i.e. the answer computed from the full spatial
    sample) instead of the median of the x_subs.
    """
    n = x_avg.size
    n_subs = x_subs.shape[1]

    covar = np.zeros((n, n), dtype="float32")

    for j in range(n):
        for k in range(n):
            covar[j, k] = (n_subs - 1.0) / n_subs * np.sum((x_subs[k, :] - x_avg[k]) * (x_subs[j, :] - x_avg[j]))

    # normalize and take diagonal entries as standard deviations
    covar /= n
    errs = np.sqrt(np.diag(covar))

    return covar, errs


def twoPointAutoCorrelationPeriodicCube(
    sP,
    cenSatSelect="all",
    minRad=10.0,
    numRadBins=20,
    colorBin=None,
    cType=None,
    mstarBin=None,
    mType=None,
    jackKnifeNumSub=4,
):
    r"""Calculate the two-point auto-correlation function (of galaxies/subhalos) in a periodic cube geometry.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      cenSatSelect (str): restrict to one of 'cen', 'sat', or 'all' (default).
      minRad (float): minimum radius [code units].
      numRadBins (int): number of radial bins.
      colorBin (2-tuple or None): if not None, a ``[min,max]`` tuple giving the minimum and maximum
        value of galaxy color to be included in the calculation. In this case we also require the
        corresponding ``cType`` to be specified.
      cType (2-tuple or None): if ``colorBin`` is not None, then should be ``[bands,simColorsModel]``
        specifying the two bands to derive the color, and the model (iso,imf,dust).
      mstarBin (2-tuple or None): if not None, a ``[min,max]]`` tuple giving the minimum and maximum
        value of galaxy stellar mass to be included in the calculation. In this case we also require
        the corresponding ``mType`` to be specified.
      mType (str or None): if ``mstarBin`` is not None, then should be a string defining the
        stellar mass field (of :py:func:`plot.quantities.simSubhaloQuantity`) and so also units.
      jackKnifeNumSub (int or None): if not None, gives the number :math:`N` of linear subdivisions
        for jackknife error estimation, requiring :math:`N^3` additional tpcf calculations.

    Returns:
      a 4-tuple composed of

      - **rad** (ndarray): distance bin mid-points [code units].
      - **xi** (ndarray): the two-point auto correlation function :math:`\chi(r) = \rm{DD}/\rm{RR} - 1`
        where ``xi[i]`` is computed between ``rad[i]:rad[i+1]``.
      - **xi_err** (ndarray): one sigma errors derived from the covariance (None if ``jackKnifeNumSub`` is None).
      - **covar** (ndarray): 2d covariance matrix (None if ``jackKnifeNumSub`` is None).

    Note:
      Automatically caching. Result is saved to ``sP.derivPath/clustering/``.
    """
    assert cenSatSelect in ["all", "cen", "sat"]
    savePath = sP.derivPath + "/clustering/"

    saveStr = "rad-%d-%.1f" % (numRadBins, minRad)
    if colorBin is not None:
        assert cType is not None
        bands, simColorsModel = cType
        saveStr += "_color-%s-%.1f-%.1f-%s" % ("".join(bands), colorBin[0], colorBin[1], simColorsModel)
    if mstarBin is not None:
        assert mType is not None
        saveStr += "_mass-%.1f-%.1f-%s" % (mstarBin[0], mstarBin[1], mType)
    if jackKnifeNumSub is not None:
        saveStr += "_err-%d" % jackKnifeNumSub

    saveFilename = savePath + "tpcf_%d_%s_%s.hdf5" % (sP.snap, cenSatSelect, saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            rad = f["rad"][()]
            xi = f["xi"][()]

            xi_err = f["xi_err"][()] if "xi_err" in f else None
            covar = f["covar"][()] if "covar" in f else None

        return rad, xi, xi_err, covar

    # calculate
    print("Calculating new: [%s]..." % saveFilename.split(sP.basePath)[1])

    # get cenSatSelect indices, load and restrict if requested
    wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

    pos = sP.groupCat(fieldsSubhalos=["SubhaloPos"])
    pos = np.squeeze(pos[wSelect, :])

    if colorBin is not None:
        # load simulation colors
        gc_colors, gc_ids = loadSimGalColors(sP, simColorsModel, bands=bands)
        assert np.array_equal(gc_ids, np.arange(sP.numSubhalos))
        gc_colors = gc_colors[wSelect]

        with np.errstate(invalid="ignore"):
            wColor = np.where((gc_colors >= colorBin[0]) & (gc_colors < colorBin[1]))

        pos = np.squeeze(pos[wColor, :])

    if mstarBin is not None:
        # load stellar masses
        gc_masses, _, _, _ = sP.simSubhaloQuantity(mType)
        gc_masses = gc_masses[wSelect]

        if colorBin is not None:
            # apply existing color restriction if applicable
            gc_masses = np.squeeze(gc_masses[wColor])

        with np.errstate(invalid="ignore"):
            wMass = np.where((gc_masses >= mstarBin[0]) & (gc_masses < mstarBin[1]))
        pos = np.squeeze(pos[wMass, :])

    # radial bins
    maxRad = sP.boxSize / 2
    radialBins = np.logspace(np.log10(minRad), np.log10(maxRad), numRadBins)

    rrBinSizeLog = (np.log10(maxRad) - np.log10(minRad)) / numRadBins
    rad = 10.0 ** (np.log10(radialBins) + rrBinSizeLog / 2)[:-1]

    # quick time estimate
    nPts = pos.shape[0]
    calc_time_sec = (float(pos.shape[0]) / 1e5) ** 2 * 600.0 / 16.0
    print(
        " nPts = %d, estimated time = %.1f sec (%.1f min) (%.2f hours) (%.2f days)"
        % (nPts, calc_time_sec, calc_time_sec / 60.0, calc_time_sec / 3600.0, calc_time_sec / 3600.0 / 24.0)
    )

    # calculate two-point correlation function
    xi, _, _ = tpcf(pos, radialBins, sP.boxSize)

    # if requested, calculate NSub^3 additional tpcf for jackknife error estimation
    xi_err = None
    covar = None

    if jackKnifeNumSub is not None:
        nSubs = jackKnifeNumSub**3
        subSize = sP.boxSize / jackKnifeNumSub

        xi_sub = np.zeros((xi.size, nSubs), dtype="float32")
        count = 0

        for i in range(jackKnifeNumSub):
            for j in range(jackKnifeNumSub):
                for k in range(jackKnifeNumSub):
                    # define spatial region
                    x0 = i * subSize
                    x1 = (i + 1) * subSize
                    y0 = j * subSize
                    y1 = (j + 1) * subSize
                    z0 = k * subSize
                    z1 = (k + 1) * subSize

                    # exclude this sub-region and create reduced point set
                    w = np.where(
                        ((pos[:, 0] <= x0) | (pos[:, 0] > x1))
                        | ((pos[:, 1] <= y0) | (pos[:, 1] > y1))
                        | ((pos[:, 2] <= z0) | (pos[:, 2] > z1))
                    )

                    print(
                        " [%2d] (%d %d %d) x=[%7d,%7d] y=[%7d,%7d] z=[%7d,%7d] keep %6d points..."
                        % (count, i, j, k, x0, x1, y0, y1, z0, z1, len(w[0]))
                    )

                    # calculcate and save tpcf
                    pos_loc = np.squeeze(pos[w, :])
                    xi_sub[:, count], _, _ = tpcf(pos_loc, radialBins, sP.boxSize)
                    count += 1

        # calculate covariance matrix
        covar, xi_err = _covar_matrix(xi, xi_sub)

    with h5py.File(saveFilename, "w") as f:
        f["rad"] = rad
        f["xi"] = xi
        if xi_err is not None:
            f["xi_err"] = xi_err
            f["covar"] = covar
            f["xi_sub"] = xi_sub
    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return rad, xi, xi_err, covar


def twoPointAutoCorrelationParticle(sP, partType, partField, pSplit=None):
    """Calculate the [weighted] two-point auto-correlation function in a periodic cube geometry.

    Instead of per-subhalo/galaxy, this is a per-particle based calculation. Given the
    prohibitive expense, we do this with a Monte Carlo sampling of the first term of the
    particle-particle tpcf, saving intermediate results which are then accumulated as available.
    If pSplit==None and no partial files exist, full/single compute and return.
    If pSplit!=None and the requested partial file exists, concatenate all and return intermediate result.
    if pSplit!=None and the requested partial file does not exist, then run a partial computation and save.
    """
    # for oxygen paper: 2.0, 20000, 40
    minRad = 0.1  # ckpc/h
    maxRad = 10000.0  # ckpc/h
    numRadBins = 50

    savePath = sP.derivPath + "/clustering/"
    saveStr = "%s_%s_rad-%d-%d-%d" % (partType, partField.replace(" ", "-"), numRadBins, minRad, maxRad)
    if pSplit is not None:
        saveStr += "_split-%d-of-%d" % (pSplit[0], pSplit[1])
    saveFilename = savePath + "tpcfp_%d_%s.hdf5" % (sP.snap, saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence of single (non-pSplit) computation
    if isfile(saveFilename) and pSplit is None:
        with h5py.File(saveFilename, "r") as f:
            rad = f["rad"][()]
            xi = f["xi"][()]
            covar = f["covar"][()] if "covar" in f else None
            xi_err = f["xi_err"][()] if "xi_err" in f else None

        return rad, xi, xi_err, covar

    # check if there are any pSplit* files, if so concatenate them on the fly and return
    if isfile(saveFilename) and pSplit is not None:
        fileSearch = saveFilename.replace("_split-%d" % pSplit[0], "_split-*")
        files = glob.glob(fileSearch)
        print("twoPointAutoCorrelationParticle() computing result from [%d] partial files..." % len(files))

        DD_sub = np.zeros((numRadBins - 1, len(files)), dtype="float32")
        RR_sub = np.zeros((numRadBins - 1, len(files)), dtype="float32")

        for i, file in enumerate(files):
            with h5py.File(file, "r") as f:
                rad = f["rad"][()]
                DD_sub[:, i] = f["DD"][()]
                RR_sub[:, i] = f["RR"][()]

        # recompute with accumulated numerator/denominators with the natural estimator
        DD = np.sum(DD_sub, axis=1)
        RR = np.sum(RR_sub, axis=1)
        xi = DD / RR - 1.0

        # calculate a covariance matrix from our sub-samples
        xi_sub = np.zeros((numRadBins - 1, len(files)), dtype="float32")
        for i in range(len(files)):
            xi_sub[:, i] = (DD - DD_sub[:, i]) / (RR - RR_sub[:, i]) - 1.0
        covar, xi_err = _covar_matrix(xi, xi_sub)

        # "permanently" save into non-split result file?
        saveFilename = fileSearch.replace("_split-*-of-%d" % pSplit[1], "")
        if not isfile(saveFilename):
            with h5py.File(saveFilename, "w") as f:
                f["rad"] = rad
                f["xi"] = xi
                f["covar"] = covar
                f["xi_err"] = xi_err

            frac = float(len(files)) / pSplit[1] * 100.0
            print("Saved: [%s]" % saveFilename.split(savePath)[1])
            print("NOTE: PARTIAL ONLY! [%d] of [%d] splits, [%.2f%%]." % (len(files), pSplit[1], frac))

        return rad, xi, xi_err, covar

    # radial bins
    maxRad = sP.boxSize / 2
    radialBins = np.logspace(np.log10(minRad), np.log10(maxRad), numRadBins)

    rrBinSizeLog = (np.log10(maxRad) - np.log10(minRad)) / numRadBins
    rad = 10.0 ** (np.log10(radialBins) + rrBinSizeLog / 2)[:-1]

    # load
    pos = sP.snapshotSubsetP(partType, "pos")
    weights = sP.snapshotSubsetP(partType, partField)

    # process weights
    w = np.where(weights < 0.0)
    weights[w] = 0.0  # non-negative
    assert np.count_nonzero(np.isfinite(weights)) == weights.size  # finite
    weights /= weights.mean()  # adjust mean to unity for better floating point accumulation accuracy

    # partial split?
    if pSplit is not None:
        shuffleSelectInd = pSplit[0]
        numProcs = pSplit[1]
        # splitSize = np.ceil(weights.size / numProcs)

        # take a contiguous chunk, but randomly permute the piece of the snapshot we handle
        np.random.seed(424242)
        splitInds = np.arange(numProcs)
        np.random.shuffle(splitInds)
        curProc = splitInds[shuffleSelectInd]

        indRangeLoc = pSplitRange([0, weights.size], numProcs, curProc)

        pos2 = pos[indRangeLoc[0] : indRangeLoc[1], :]
        weights2 = weights[indRangeLoc[0] : indRangeLoc[1]]
    else:
        # global load
        pos2 = pos
        weights2 = weights

    # quick time estimate
    nPts1 = pos.shape[0]
    nPts2 = pos2.shape[0]
    calc_sec = float(nPts1 * nPts2) / 1e10 * 600.0 / 16.0
    print(
        " nPts = %d x %d (~%d^2), estimated time = %.1f sec (%.1f min) (%.2f hours) (%.2f days)"
        % (nPts1, nPts2, np.sqrt(nPts1 * nPts2), calc_sec, calc_sec / 60.0, calc_sec / 3600.0, calc_sec / 3600.0 / 24.0)
    )

    # calculate weighted two-point [cross] correlation function
    xi, DD, RR = tpcf(pos, radialBins, sP.boxSize, weights=weights, pos2=pos2, weights2=weights2)

    with h5py.File(saveFilename, "w") as f:
        f["rad"] = rad
        f["xi"] = xi
        f["DD"] = DD
        f["RR"] = RR
    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return rad, xi, None, None


def isolationCriterion3D(sP, rad_pkpc, cenSatSelect="all", mstar30kpc_min=9.0):
    """For every subhalo, record the maximum nearby subhalo mass within some distance.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      rad_pkpc (float): 3d spherical aperture.
      cenSatSelect (str): Filter to consider only "all", "cen" (central), or "sat" (satellite) subhalos in the search.
      mstar30kpc_min (float or None): if not None, only consider subhalos above this stellar mass [log Msun].
    """
    assert cenSatSelect in ["all", "cen", "sat"]
    savePath = sP.derivPath + "/clustering/"

    saveStr = "_rad-%d-pkpc" % rad_pkpc
    if mstar30kpc_min is not None:
        saveStr += "_mass-%.1f" % mstar30kpc_min
    saveFilename = savePath + "isolation_crit_%d_%s%s.hdf5" % (sP.snap, cenSatSelect, saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence
    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]

        return r

    # calculate
    print("Calculating new: [%s]..." % saveFilename.split(sP.basePath)[1])

    # load and unit conversions
    gc = sP.groupCat(
        fieldsHalos=["Group_M_Crit200"],
        fieldsSubhalos=["SubhaloPos", "SubhaloMassInRadType", "SubhaloMass", "SubhaloGrNr"],
    )
    ac = sP.auxCat(fields=["Subhalo_Mass_30pkpc_Stars"])

    nSubhalos = sP.numSubhalos

    masses = OrderedDict()

    masses["halo_m200"] = sP.units.codeMassToLogMsun(gc["halos"])[gc["subhalos"]["SubhaloGrNr"]]
    masses["mstar2"] = sP.units.codeMassToLogMsun(gc["subhalos"]["SubhaloMassInRadType"][:, sP.ptNum("stars")])
    masses["mtotal"] = sP.units.codeMassToLogMsun(gc["subhalos"]["SubhaloMass"])
    masses["mstar30kpc"] = sP.units.codeMassToLogMsun(ac["Subhalo_Mass_30pkpc_Stars"])

    # handle cenSatSelect, reduce masses and stack into 2d ndarray, create new pos
    inds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
    quants = np.zeros((inds.size, len(masses.keys())), dtype="float32")

    for i, key in enumerate(masses):
        quants[:, i] = masses[key][inds]

    pos_target = np.squeeze(gc["subhalos"]["SubhaloPos"][inds, :])

    rad_bin_code = np.array([0, sP.units.physicalKpcToCodeLength(rad_pkpc)])

    # handle mstar30kpc_min on pos_search if requested
    if mstar30kpc_min is None:
        wMinMass = np.arange(nSubhalos)

        pos_search = gc["subhalos"]["SubhaloPos"]
    else:
        with np.errstate(invalid="ignore"):
            wMinMass = np.where(masses["mstar30kpc"] >= mstar30kpc_min)[0]
        print(" reducing [%d] to [%d] subhalo searches..." % (nSubhalos, len(wMinMass)))

        pos_search = np.squeeze(gc["subhalos"]["SubhaloPos"][wMinMass, :])

    # call reduction
    start_time = time.time()
    print(" start...")

    qred = quantReductionInRad(pos_search, pos_target, rad_bin_code, quants, "max", sP.boxSize)

    sec = time.time() - start_time
    print(" took: %.1f sec (%.2f min)" % (sec, sec / 60.0))

    r = {}
    for i, key in enumerate(masses):
        r[key] = np.zeros(nSubhalos, dtype=quants.dtype)
        r[key].fill(np.nan)
        r[key][wMinMass] = np.squeeze(qred[:, 0, i])

    # verify
    nVerify = 10
    np.random.seed(4242 + wMinMass.size + nVerify)
    verifyInds = np.random.choice(wMinMass.size, size=nVerify, replace=False)

    for verifyInd in verifyInds:
        cen_pos = np.squeeze(pos_search[verifyInd, :])
        dists = sP.periodicDistsSq(cen_pos, pos_target)
        w = np.where((dists <= rad_bin_code[-1] ** 2) & (dists > 0.0))

        for i, key in enumerate(masses):
            if np.count_nonzero(~np.isnan(quants[w, i])):
                assert r[key][wMinMass[verifyInd]] == np.nanmax(quants[w, i])
            else:
                assert r[key][wMinMass[verifyInd]] == -np.inf

    # calculate some useful isolation flags
    flagNames = [
        "flag_iso_mstar2_max_half",
        "flag_iso_mstar30kpc_max_half",
        "flag_iso_mstar30kpc_max_third",
        "flag_iso_mstar30kpc_max",
        "flag_iso_mhalo_lt_12",
    ]

    for flagName in flagNames:
        r[flagName] = np.zeros(nSubhalos, dtype="int16")
        r[flagName].fill(-1)  # -1 denotes unprocessed, 0 denotes not isolated, 1 denotes isolated

    subhalo_mstar2 = 10.0 ** masses["mstar2"]
    subhalo_mstar30kpc = 10.0 ** masses["mstar30kpc"]
    ngb_max_mstar2 = 10.0 ** r["mstar2"]
    ngb_max_mstar30kpc = 10.0 ** r["mstar30kpc"]

    with np.errstate(invalid="ignore"):
        w1 = np.where(ngb_max_mstar2 <= subhalo_mstar2 / 2)
        w2 = np.where(ngb_max_mstar2 > subhalo_mstar2 / 2)
        r["flag_iso_mstar2_max_half"][w1] = 1
        r["flag_iso_mstar2_max_half"][w2] = 0

        w1 = np.where(ngb_max_mstar30kpc <= subhalo_mstar30kpc / 2)
        w2 = np.where(ngb_max_mstar30kpc > subhalo_mstar30kpc / 2)
        r["flag_iso_mstar30kpc_max_half"][w1] = 1
        r["flag_iso_mstar30kpc_max_half"][w2] = 0

        w1 = np.where(ngb_max_mstar30kpc <= subhalo_mstar30kpc / 3)
        w2 = np.where(ngb_max_mstar30kpc > subhalo_mstar30kpc / 3)
        r["flag_iso_mstar30kpc_max_third"][w1] = 1
        r["flag_iso_mstar30kpc_max_third"][w2] = 0

        w1 = np.where(ngb_max_mstar30kpc <= subhalo_mstar30kpc)
        w2 = np.where(ngb_max_mstar30kpc > subhalo_mstar30kpc)
        r["flag_iso_mstar30kpc_max"][w1] = 1
        r["flag_iso_mstar30kpc_max"][w2] = 0

        w1 = np.where(r["halo_m200"] <= 12.0)
        w2 = np.where(r["halo_m200"] > 12.0)
        r["flag_iso_mhalo_lt_12"][w1] = 1
        r["flag_iso_mhalo_lt_12"][w2] = 0

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return r


def conformityRedFrac(
    sP,
    radRange=(0.0, 20.0),
    numRadBins=40,
    isolationRadPKpc=500.0,
    colorBin=None,
    cType=None,
    mstarBin=None,
    mType=None,
    jackKnifeNumSub=4,
    cenSatSelectSec="all",
    colorSplitSec=None,
):
    """Calculate the conformity signal for -isolated- primaries.

    Subject to a sample selection of: {colorBin, cType, mstarBin, mType}. Specifications for these are as in
    twoPointAutoCorrelationPeriodicCube(). A series of numRadBins are linearly spaced from
    radRange[0] to radRange[1] in physical Mpc. cenSatSelect is applied to the secondary sample,
    as are {colorSplitSec,cType} which define the red/blue split for secondaries.
    """
    assert cenSatSelectSec in ["all", "cen", "sat"]
    assert colorSplitSec is not None and cType is not None
    bands, simColorsModel = cType

    savePath = sP.derivPath + "/clustering/"

    saveStr = "rad-n%d-%.1f-%.1f" % (numRadBins, radRange[0], radRange[1])
    if colorBin is not None:
        saveStr += "_color-%s-%.1f-%.1f-%s" % ("".join(bands), colorBin[0], colorBin[1], simColorsModel)
    if mstarBin is not None:
        assert mType is not None
        saveStr += "_mass-%.1f-%.1f-%s" % (mstarBin[0], mstarBin[1], mType)
    if jackKnifeNumSub is not None:
        saveStr += "_err-%d" % jackKnifeNumSub

    saveFilename = savePath + "conformity_redfrac_%d_%s_%s.hdf5" % (sP.snap, cenSatSelectSec, saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence
    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]

        return r

    # calculate
    print("Calculating new: [%s]..." % saveFilename.split(sP.basePath)[1])

    # load
    pos_all = sP.groupCat(fieldsSubhalos=["SubhaloPos"])
    gc_colors, gc_ids = loadSimGalColors(sP, simColorsModel, bands=bands)
    assert np.array_equal(gc_ids, np.arange(sP.numSubhalos))

    # PRIMARIES:
    iso = isolationCriterion3D(sP, rad_pkpc=isolationRadPKpc)

    wIsolated = np.where(iso["flag_iso_mstar30kpc_max_half"] == 1)

    pos_pri = pos_all.copy()
    if colorBin is not None:
        # color bin
        gc_colors_pri = gc_colors[wIsolated]

        with np.errstate(invalid="ignore"):
            wColor = np.where((gc_colors_pri >= colorBin[0]) & (gc_colors_pri < colorBin[1]))

        pos_pri = np.squeeze(pos_pri[wColor, :])

    if mstarBin is not None:
        # stellar masses bin
        gc_masses, _, _, _ = sP.simSubhaloQuantity(mType)
        gc_masses = gc_masses[wIsolated]

        if colorBin is not None:
            # apply existing color restriction if applicable
            gc_masses = np.squeeze(gc_masses[wColor])

        with np.errstate(invalid="ignore"):
            wMass = np.where((gc_masses >= mstarBin[0]) & (gc_masses < mstarBin[1]))
        pos_pri = np.squeeze(pos_pri[wMass, :])

    # SECONDARIES:
    wSelectSec = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelectSec)
    pos_sec = np.squeeze(pos_all[wSelectSec, :])

    gc_colors_sec = gc_colors[wSelectSec]

    with np.errstate(invalid="ignore"):
        wRedSec = np.where(gc_colors_sec >= colorSplitSec)
        wBlueSec = np.where(gc_colors_sec < colorSplitSec)

    # generate quants[nSecondaries,2] marked for blue/red
    quants = np.zeros((pos_sec.shape[0], 2), dtype="int32")
    quants[wBlueSec, 0] = 1
    quants[wRedSec, 1] = 1

    # radial bins
    r = {}

    radialBins = np.linspace(radRange[0], radRange[1], numRadBins + 1)
    radBinSize = (radRange[1] - radRange[0]) / numRadBins
    radialBins_code = sP.units.physicalMpcToCodeLength(radialBins)

    r["rad"] = (radialBins + radBinSize / 2)[:-1]

    # call reduction
    start_time = time.time()
    print(" start...")

    quants_reduced = quantReductionInRad(pos_pri, pos_sec, radialBins_code, quants, "sum", sP.boxSize)

    sec = time.time() - start_time
    print(" took: %.1f sec %.2f min" % (sec, sec / 60.0))

    # compute red fraction as equally weighted average across primaries (not first stacked), in each
    # case only including bins with at least one secondary (following Bray+ 2015 end of Sec 2.1)
    def _redfrac_helper(qred):
        """Helper function, transform return of quantReductionInRad() to radial profiles of red frac."""
        redfrac = np.zeros(numRadBins, dtype="float32")
        redfrac_stacked = np.zeros(numRadBins, dtype="float32")

        for i in range(numRadBins):
            qred_frac = np.zeros(qred.shape[0], dtype="float32")
            qred_frac.fill(np.nan)

            # counts (per halo) in this radial bin
            red_counts = np.squeeze(qred[:, i, 1])
            blue_counts = np.squeeze(qred[:, i, 0])

            # avoid division by zero and only count nonzero bins
            w = np.where((red_counts + blue_counts) > 0)
            qred_frac[w] = red_counts[w] / (red_counts[w] + blue_counts[w])

            # halo average
            redfrac[i] = np.nanmean(qred_frac)

            # stacked
            redfrac_stacked[i] = red_counts.sum() / (red_counts.sum() + blue_counts.sum())

        return redfrac, redfrac_stacked

    r["redfrac"], r["redfrac_stacked"] = _redfrac_helper(quants_reduced)

    # if requested, calculate NSub^3 additional times for jackknife error estimation
    if jackKnifeNumSub is not None:
        nSubs = jackKnifeNumSub**3
        subSize = sP.boxSize / jackKnifeNumSub

        r["redfrac_sub"] = np.zeros((numRadBins, nSubs), dtype="float32")
        r["redfrac_stacked_sub"] = np.zeros((numRadBins, nSubs), dtype="float32")
        count = 0

        for i in range(jackKnifeNumSub):
            for j in range(jackKnifeNumSub):
                for k in range(jackKnifeNumSub):
                    # define spatial region
                    x0 = i * subSize
                    x1 = (i + 1) * subSize
                    y0 = j * subSize
                    y1 = (j + 1) * subSize
                    z0 = k * subSize
                    z1 = (k + 1) * subSize

                    # exclude this sub-region and create reduced point set
                    w_pri = np.where(
                        ((pos_pri[:, 0] <= x0) | (pos_pri[:, 0] > x1))
                        | ((pos_pri[:, 1] <= y0) | (pos_pri[:, 1] > y1))
                        | ((pos_pri[:, 2] <= z0) | (pos_pri[:, 2] > z1))
                    )
                    w_sec = np.where(
                        ((pos_sec[:, 0] <= x0) | (pos_sec[:, 0] > x1))
                        | ((pos_sec[:, 1] <= y0) | (pos_sec[:, 1] > y1))
                        | ((pos_sec[:, 2] <= z0) | (pos_sec[:, 2] > z1))
                    )

                    print(
                        " [%2d] (%d %d %d) x=[%7d,%7d] y=[%7d,%7d] z=[%7d,%7d] keep (%6d pri,%6d sec) points..."
                        % (count, i, j, k, x0, x1, y0, y1, z0, z1, len(w_pri[0]), len(w_sec[0]))
                    )

                    # calculcate and save counts on this subset
                    pos_pri_loc = np.squeeze(pos_pri[w_pri, :])
                    pos_sec_loc = np.squeeze(pos_sec[w_sec, :])
                    quants_loc = np.squeeze(quants[w_sec, :])

                    if pos_pri_loc.ndim == 1:
                        count += 1
                        continue

                    quants_reduced = quantReductionInRad(
                        pos_pri_loc, pos_sec_loc, radialBins_code, quants_loc, "sum", sP.boxSize
                    )

                    r["redfrac_sub"][:, count], r["redfrac_stacked_sub"][:, count] = _redfrac_helper(quants_reduced)

                    count += 1

        # normalize and take standard deivations from the diagonal
        r["covar"], r["redfrac_err"] = _covar_matrix(r["redfrac"], r["redfrac_sub"])
        r["covar_stacked"], r["redfrac_stacked_err"] = _covar_matrix(r["redfrac_stacked"], r["redfrac_stacked_sub"])

    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return r
