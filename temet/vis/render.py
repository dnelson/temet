"""
Visualizations: render actual image.
"""

import hashlib
from getpass import getuser
from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from ..cosmo.cloudy import cloudyEmission
from ..cosmo.stellarPop import sps
from ..spectra.spectrum import create_spectra_from_traced_rays
from ..util.boxRemap import remapPositions
from ..util.helper import logZeroMin, pSplitRange
from ..util.match import match
from ..util.rotation import perspectiveProjection, rotateCoordinateArray
from ..util.sphMap import sphMap
from ..util.treeSearch import calcHsml
from ..util.voronoiRay import rayTrace
from ..vis.quantities import (
    colDensityFields,
    gridOutputProcess,
    haloCentricFields,
    totSumFields,
    velCompFieldNames,
    velLOSFieldNames,
    volDensityFields,
)


def getHsmlForPartType(sP, partType, nNGB=64, indRange=None, useSnapHsml=False, alsoSFRgasForStars=False, pSplit=None):
    """Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given type.

    By default for the full snapshot, optionally restricted to an input indRange.
    """
    _, sbStr, _ = sP.subboxVals()
    irStr = "" if indRange is None else ".%d-%d" % (indRange[0], indRange[1])
    shStr = "" if useSnapHsml is False else ".sv"
    ngStr = "" if nNGB == 64 else ".ngb%d" % nNGB
    sfStr = "" if alsoSFRgasForStars is False else ".sfgas"
    saveFilename = sP.derivPath + "hsml/hsml.%s%d.%s%s%s%s%s.hdf5" % (
        sbStr,
        sP.snap,
        partType,
        irStr,
        shStr,
        ngStr,
        sfStr,
    )

    if not isdir(sP.derivPath + "hsml/"):
        mkdir(sP.derivPath + "hsml/")

    if pSplit is not None:
        # assert 0 # only for testing
        saveFilename = saveFilename.replace(".hdf5", "_%d_of_%d.hdf5" % (pSplit[0], pSplit[1]))
        print("Running pSplit! ", pSplit, saveFilename)

    # cache?
    useCache = (sP.isPartType(partType, "stars") and (not useSnapHsml)) or (
        sP.isPartType(partType, "dm") and not sP.snapHasField(partType, "SubfindHsml")
    )

    if sP.isPartType(partType, "stars"):
        if sP.isSubbox:
            useCache = True  # StellarHsml not saved for subboxes
        if sP.snapHasField(partType, "StellarHsml"):
            useCache = False  # if present, always use these values for stars

    if useSnapHsml:
        useCache = False
        assert sP.isPartType(partType, "stars")  # don't have any logic for dm/gas below not to use snapshot values

    if useCache and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename, "r") as f:
            hsml = f["hsml"][()]
        # print(' loaded: [%s]' % saveFilename.split(sP.derivPath)[1])

    else:
        # dark matter
        if sP.isPartType(partType, "dm") or sP.isPartType(partType, "dmlowres"):
            if not sP.snapHasField(partType, "SubfindHsml"):
                if indRange is None:
                    print("Warning: Computing DM hsml for global snapshot.")
                pos = sP.snapshotSubsetP(partType, "pos", indRange=indRange)
                # treePrec = 'single' if pos.dtype == np.float32 else 'double'
                nNGBDev = int(np.sqrt(nNGB) / 2)
                hsml = calcHsml(pos, sP.boxSize, nNGB=nNGB, nNGBDev=nNGBDev, treePrec="double")
            else:
                hsml = sP.snapshotSubsetP(partType, "SubfindHsml", indRange=indRange)

        # gas
        if sP.isPartType(partType, "gas"):
            if sP.simCode == "SWIFT":
                # use direcly the SPH smoothing length, instead of the volume derived spherically-equivalent radius
                hsml = sP.snapshotSubsetP(partType, "SmoothingLengths", indRange=indRange)
                hsml /= defaultHsmlFac(partType)  # cancel out since we will multiply by this value later
            else:
                hsml = sP.snapshotSubsetP(partType, "cellrad", indRange=indRange)

        # stars
        if sP.isPartType(partType, "stars"):
            # SubfindHsml is a density estimator of the local DM, don't generally use for stars
            if useSnapHsml:
                if sP.snapHasField("stars", "SubfindHsml"):
                    hsml = sP.snapshotSubsetP(partType, "SubfindHsml", indRange=indRange)
                else:
                    # we will generate SubfindHsml
                    indRange_dm = None
                    assert indRange is None  # otherwise generalize, derive indRange_dm to e.g. load only FoF-scope DM

                    pos_stars = sP.snapshotSubsetP(partType, "pos", indRange=indRange)
                    pos_dm = sP.snapshotSubsetP("dm", "pos", indRange=indRange_dm)
                    hsml = calcHsml(pos_dm, sP.boxSize, posSearch=pos_stars, nNGB=64, nNGBDev=4, treePrec="double")
            elif sP.snapHasField("stars", "StellarHsml"):
                # use pre-saved nNGB=32 values
                hsml = sP.snapshotSubsetP(partType, "StellarHsml", indRange=indRange)
            elif alsoSFRgasForStars:
                # compute: using SFR>0 gas as well as stars to define neighbors
                indRange_gas = None
                assert indRange is None  # otherwise generalize, derive indRange_gas to e.g. load only FoF-scope gas

                pos_stars = sP.snapshotSubsetP(partType, "pos", indRange=indRange)
                pos_sfgas = sP.snapshotSubsetP("gas_sf", "pos", indRange=indRange_gas)
                pos = np.vstack((pos_stars, pos_sfgas))

                hsml = calcHsml(pos, sP.boxSize, posSearch=pos_stars, nNGB=nNGB, nNGBDev=1, treePrec="double")
            else:
                # compute: use only stars to define neighbors
                pos = sP.snapshotSubsetP(partType, "pos", indRange=indRange)
                if isinstance(pos, dict) and pos["count"] == 0:
                    hsml = np.array([])
                else:
                    posSearch = pos  # default
                    if pSplit is not None:
                        indRangeLoc = pSplitRange([0, pos.shape[0]], pSplit[1], pSplit[0])
                        posSearch = pos[indRangeLoc[0] : indRangeLoc[1]]
                        print(" posSearch range: ", indRangeLoc)

                    # check for identical coordinates due to float32 snapshot (random sampling from first few particles)
                    if sP.isZoom and pos.dtype == np.float32:
                        flag = False
                        for i in range(np.min([10, pos.shape[0]])):
                            for j in range(i + 1, np.min([i + 10, pos.shape[0]])):
                                flag |= np.array_equal(pos[i, :], pos[j, :])
                        if flag:
                            print("NOTE: Found duplicate Coordinates, perturbing randomly for calcHsml().")
                            pos = pos.astype("float64")
                            rng = np.random.default_rng(42424242)
                            pos += rng.uniform(size=pos.shape) * (sP.gravSoft / 1e6)

                    hsml = calcHsml(pos, sP.boxSize, posSearch=posSearch, nNGB=nNGB, nNGBDev=1, treePrec="double")

        # bhs (unused)
        if sP.isPartType(partType, "bhs"):
            hsml = sP.snapshotSubsetP(partType, "BH_Hsml", indRange=indRange)

        # save
        if useCache:
            with h5py.File(saveFilename, "w") as f:
                f["hsml"] = hsml
            print(" saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    return hsml.astype("float32")


def defaultHsmlFac(partType):
    """Helper, set default hsmlFac for a given partType if not input."""
    if partType == "gas":
        return 2.5  # times cellsize
    if partType == "stars":
        return 1.0  # times nNGB=32 CalcHsml search
    if partType == "bhs":
        return 1.0  # times BH_Hsml, currently unused
    if partType == "dm":
        return 0.5  # times SubfindHsml or nNGB=64 CalcHsml search
    if partType == "dmlowres":
        return 4.0

    raise Exception("Unrecognized partType [%s]." % partType)


def clipStellarHSMLs(hsml, sP, pxScale, nPixels, indRange, method=2):
    """Clip input stellar HSMLs/sizes to minimum/maximum values. Work in progress."""
    # use a minimum/maximum size for stars in outskirts
    if method == 0:
        # constant based on numerical resolution
        hsml[hsml < 0.05 * sP.gravSoft] = 0.05 * sP.gravSoft
        hsml[hsml > 2.0 * sP.gravSoft] = 2.0 * sP.gravSoft  # can decouple, leads to strageness

        # print(' [m0] stellar hsml clip above [%.1f px] below [%.1f px]' % (2.0*sP.gravSoft,0.05*sP.gravSoft))

    # adaptively clip in proportion to pixel scale of image, depending on ~pixel number
    if method == 1:
        # adaptive technique 2 (used for Gauss proposal stellar composite figure)
        clipAboveNumPx = 30.0 * (np.max(nPixels) / 1920)
        clipAboveToPx = np.max([5.0, 6.0 - 2 * 1920 / np.max(nPixels)])  # was 3.0 not 5.0 before composite tests
        hsml[hsml > clipAboveNumPx * pxScale] = clipAboveToPx * pxScale

        # print(' [m1] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method == 2:
        # adaptive technique 1 (preferred) (used for TNG subbox movies)
        # minClipVal = 4.0 # was 3.0 before composite tests # previous
        minClipVal = 30.0 * (np.max(nPixels) / 1920)  # testing for tng.methods2

        # if 'sdss_g' in partField:
        #    minClipVal = 20.0
        #    print(' set minClipVal from 3 to 20 for Blue-channel')

        clipAboveNumPx = np.max([minClipVal, minClipVal * 2 / (1920 / np.max(nPixels))])
        clipAboveToPx = clipAboveNumPx  # coupled
        hsml[hsml > clipAboveNumPx * pxScale] = clipAboveToPx * pxScale

        # print(' [m2] stellar hsml above [%.1f px] to [%.1f px] (%.1f to %.1f kpc)' % \
        #    (clipAboveNumPx,clipAboveToPx,clipAboveNumPx*pxScale,clipAboveToPx*pxScale))

    if method == 3:
        # print(' custom AGE HSMLFAC MOD!') # (now default behavior)
        age_min = 1.0
        age_max = 3.0
        max_mod = 2.0

        # load stellar ages
        ages = sP.snapshotSubset("stars", "stellar_age", indRange=indRange)

        # ramp such that hsml*=1.0 at <=1Gyr, linear to hsml*=2.0 at >=4 Gyr
        rampFac = np.ones(ages.size, dtype="float32")
        ages = np.clip((ages - age_min) / (age_max - age_min), 0.0, 1.0) * max_mod
        rampFac += ages
        hsml *= rampFac

    if method is None:
        pass
        # print(' hsml clip DISABLED!')

    return hsml


def _stellar_3bands(partField):
    """Helper, parse 3-band stellar composite field name."""
    bands = partField.split("-")[1:]

    if len(bands) == 0:
        bands = ["jwst_f200w", "jwst_f115w", "jwst_f070w"]  # default

    assert len(bands) == 3

    bandsStr = ", ".join(bands)
    if all(band.startswith("jwst_") for band in bands):
        bandsStr = "JWST %s" % ", ".join([band.replace("jwst_", "") for band in bands])
    label = "Stellar Composite [%s]" % bandsStr

    return bands, label


def stellar3BandCompositeImage(
    sP,
    partField,
    method,
    nPixels,
    axes,
    projType,
    projParams,
    boxCenter,
    boxSizeImg,
    hsmlFac,
    rotMatrix,
    rotCenter,
    remapRatio,
    forceRecalculate,
    smoothFWHM,
    snapHsmlForStars,
    alsoSFRgasForStars,
    excludeSubhaloFlag,
    skipCellIndices,
    ptRestrictions,
    weightField,
    randomNoise,
    autoLimits,
):
    """Generate 3-band RGB composite using starlight in three different passbands. Work in progress."""
    bands, label = _stellar_3bands(partField)

    assert projType == "ortho"

    fieldPrefix = "stellarBandObsFrame-" if "ObsFrame" in partField else "stellarBand-"

    # print('Generating stellar composite with %s [%s %s %s]' % (fieldPrefix,bands[0],bands[1],bands[2]))
    band_grids = []  # in mags
    for i in range(3):
        grid_loc, _, _ = gridBox(
            sP,
            method,
            "stars",
            fieldPrefix + bands[i],
            nPixels,
            axes,
            projType,
            projParams,
            boxCenter,
            boxSizeImg,
            hsmlFac,
            rotMatrix,
            rotCenter,
            remapRatio,
            forceRecalculate,
            smoothFWHM,
            snapHsmlForStars,
            alsoSFRgasForStars,
            excludeSubhaloFlag,
            skipCellIndices,
            ptRestrictions,
            weightField,
            randomNoise,
        )
        band_grids.append(grid_loc)

    # convert magnitudes to linear luminosities
    ww = np.where(band_grids[0] < 99)  # these left at zero
    band0_grid = band_grids[0].copy() * 0.0
    band0_grid[ww] = np.power(10.0, -0.4 * band_grids[0][ww])

    ww = np.where(band_grids[1] < 99)
    band1_grid = band_grids[1].copy() * 0.0
    band1_grid[ww] = np.power(10.0, -0.4 * band_grids[1][ww])

    ww = np.where(band_grids[2] < 99)
    band2_grid = band_grids[2].copy() * 0.0
    band2_grid[ww] = np.power(10.0, -0.4 * band_grids[2][ww])

    grid_master = np.zeros((nPixels[1], nPixels[0], 3), dtype="float32")
    grid_master_u = np.zeros((nPixels[1], nPixels[0], 3), dtype="uint8")

    if 0:
        # old trials, KBU is similar to method used in many Auriga papers
        fac = (1 / sP.res) ** 2 * (boxSizeImg[0] / nPixels[0]) ** 2  # check

        dranges = {
            "snap_K": [400, 80000],  # red
            "snap_B": [20, 13000],  # green
            "snap_U": [13, 20000],  # blue
            "2mass_ks": [40, 8000],  # red
            "b": [2, 3300],  # green
            "u": [1, 1500],  # blue
            "wfc_acs_f814w": [60, 8000],  # red
            "wfc_acs_f606w": [20, 50000],  # green
            "wfc_acs_f475w": [3, 20000],  # blue
            "jwst_f070w": [4000, 30000],  # red  #[400, 30000]
            "jwst_f115w": [2000, 85000],  # green  #[200, 85000]
            "jwst_f200w": [1000, 75000],  # blue  #[100, 75000]
            "sdss_z": [100, 1000],  # red
            "sdss_i": [30, 5000],  # red
            "sdss_r": [30, 6000],  # green
            "sdss_g": [1, 7000],  # blue
            "sdss_u": [5, 3000],
        }  # blue

        for i in range(3):
            drange = dranges[bands[i]]
            drange = np.array(drange) * 1.0  # fac
            drange_log = np.log10(drange)

            if i == 0:
                grid_loc = band0_grid
            if i == 1:
                grid_loc = band1_grid
            if i == 2:
                grid_loc = band2_grid

            print(" ", i, bands[i], drange, grid_loc.mean(), grid_loc.min(), grid_loc.max())

            grid_log = np.log10(np.clip(grid_loc, drange[0], drange[1]))
            grid_stretch = (grid_log - drange_log[0]) / (drange_log[1] - drange_log[0])

            grid_master[:, :, i] = grid_stretch
            grid_master_u[:, :, i] = grid_stretch * np.uint8(255)

    if "-".join(bands) in ["sdss_g-sdss_r-sdss_i", "sdss_r-sdss_i-sdss_z"]:
        # gri-composite, following the method of Lupton+2004 (as in SDSS/HSC RGB cutouts)
        if "-".join(bands) == "sdss_r-sdss_i-sdss_z":
            fac = {"g": 1.0, "r": 1.0, "i": 1.2}  # RGB = riz
        else:
            fac = {"g": 1.0, "r": 1.0, "i": 0.8}  # RGB = gri

        lupton_alpha = 2.0  # 1/stretch
        lupton_Q = 10.0  # lower values clip highlights more (more contrast)
        scale_min = 0.1  # 1e5 # units of linear luminosity

        # make RGB array using arcsinh scaling following Lupton (1e7 shift to avoid truncation issues)
        band0_grid = sP.units.absMagToLuminosity(band_grids[0]) * fac["g"] * 1e7
        band1_grid = sP.units.absMagToLuminosity(band_grids[1]) * fac["r"] * 1e7
        band2_grid = sP.units.absMagToLuminosity(band_grids[2]) * fac["i"] * 1e7

        if "ObsFrame" in partField:  # scaling is sensitive to this value (i.e. mean flux), needs some generalization
            band0_grid *= 1e15
            band1_grid *= 1e15
            band2_grid *= 1e15  # 2e16 for RIZ, 5e16 for all for GRI

        inten = (band0_grid + band1_grid + band2_grid) / 3.0
        val = np.arcsinh(lupton_alpha * lupton_Q * (inten - scale_min)) / lupton_Q

        grid_master[:, :, 0] = band0_grid * val / inten
        grid_master[:, :, 1] = band1_grid * val / inten
        grid_master[:, :, 2] = band2_grid * val / inten

        if 0:
            # rescale and clip (not needed)
            maxval = np.max(grid_master, axis=2)  # for every pixel, across the 3 bands

            w = np.where(maxval > 1.0)
            for i in range(3):
                grid_master[w[0], w[1], i] /= maxval[w]

            # minval = np.min(grid_master, axis=2)

            w = np.where((maxval < 0.0) | (inten < 0.0))
            for i in range(3):
                grid_master[w[0], w[1], i] = 0.0

        # construct RGB
        grid_master = np.clip(grid_master, 0.0, 1.0)

        grid_master_u[:, :, 0] = grid_master[:, :, 2] * np.uint8(255)
        grid_master_u[:, :, 1] = grid_master[:, :, 1] * np.uint8(255)
        grid_master_u[:, :, 2] = grid_master[:, :, 0] * np.uint8(255)

    else:
        # typical custom technique for JWST (rest-frame) composites
        pxArea = (boxSizeImg[axes[1]] / nPixels[0]) * (boxSizeImg[axes[0]] / nPixels[1])

        if 0:
            pxArea0 = (80.0 / 960) ** 2.0  # at which the following ranges were calibrated
            resFac = 1.0  # (512.0/sP.res)**2.0

            # 2.2 for twelve, 1.4 for thirteen
            minVal = 2.2  # 0.6, 1.4, previous: 2.2 = best recent option, 2.8, 3.3 (nice control of low-SB features)
            maxVal = 5.60  # control clipping of high-SB features

            # if band0_grid.max() < 1e2:
            #    minVal = 0.6
            #    maxVal = 3.60

            if band0_grid.max() > 5e4:
                minVal = 3.2
                maxVal = 6.6

            minValLog = np.array([minVal, minVal, minVal])
            minValLog = np.log10((10.0**minValLog) * (pxArea / pxArea0 * resFac))

            # maxValLog = np.array([5.71, 5.68, 5.36])*0.9 # jwst f200w, f115w, f070w # previous
            maxValLog = np.array(
                [maxVal, maxVal + 0.08, maxVal - 0.24]
            )  # little less clipping, more yellow/red color (fiducial)
            # maxValLog = np.array([5.40, 5.48, 5.16]) # TNG50 sb0sh481167 movie: galaxy three only
            # maxValLog = np.array([6.70, 6.78, 6.46]) # TNG50 sb2sh0 movie: galaxy two only

            maxValLog = np.log10((10.0**maxValLog) * (pxArea / pxArea0 * resFac))
            # print('pxArea*res mod: ',(pxArea/pxArea0*resFac))

        if autoLimits:
            # new auto bounds
            band_vals = np.log10(band0_grid[band0_grid > 0])

            percs = np.nanpercentile(band_vals, [20, 99.5])

            if band_vals.size > 0:
                minValLog = percs[[0, 0, 0]]
                maxValLog = np.array([percs[1], percs[1] + 0.08, percs[1] - 0.24])
            else:
                minValLog = [0, 0, 0]
                maxValLog = [1, 1, 1]
        else:
            minValLog = np.array([-1.7, -1.7, -1.7])
            maxValLog = np.array([1.2, 1.3, 1.0])

            pxAreaFac = np.sqrt((960 / nPixels[0]) * (960 / nPixels[1]))
            minValLog = np.log10(10.0**minValLog * pxAreaFac)
            maxValLog = np.log10(10.0**maxValLog * pxAreaFac)

        for i in range(3):
            if i == 0:
                grid_loc = band0_grid
            if i == 1:
                grid_loc = band1_grid
            if i == 2:
                grid_loc = band2_grid

            # handle zero values
            ww = np.where(grid_loc == 0.0)
            if 0:
                # old: leads to flickering/issues in movies
                ww_nonzero = np.where(grid_loc > 0.0)

                if len(ww_nonzero[0]):
                    grid_loc[ww] = grid_loc[ww_nonzero].min() * 0.1  # 10x less than min
                else:
                    grid_loc[ww] = 1e-10  # full empty/zero image (leave as all black)
            else:
                grid_loc[ww] = 1e-10  # full empty/zero image (leave as all black)

            grid_log = np.log10(grid_loc)

            # clip and stretch within [minValLog,maxValLog]
            grid_log = np.clip(grid_log, minValLog[i], maxValLog[i])
            grid_stretch = (grid_log - minValLog[i]) / (maxValLog[i] - minValLog[i])

            grid_master[:, :, i] = grid_stretch
            grid_master_u[:, :, i] = grid_stretch * np.uint8(255)

            # print(' grid: ',i,grid_stretch.min(),grid_stretch.max(),\
            #       grid_master_u[:,:,i].min(),grid_master_u[:,:,i].max())

        # saturation adjust
        if 0:
            satVal = 1.5  # 0.0 -> b&w, 0.5 -> reduce color saturation by half, 1.0 -> unchanged
            R = grid_master_u[:, :, 0]
            G = grid_master_u[:, :, 1]
            B = grid_master_u[:, :, 2]
            P = np.sqrt(R * R * 0.299 + G * G * 0.587 + B * B * 0.144)  # standard luminance weights

            ww = np.where(B > 150)
            # grid_master_u[:,:,0] = np.uint8(np.clip( P + (R-P)*satVal, 0, 255 ))
            # grid_master_u[:,:,1] = np.uint8(np.clip( P + (G-P)*satVal, 0, 255 ))
            B[ww] = np.uint8(np.clip(P[ww] + (B[ww] - P[ww]) * satVal, 0, 255))
            grid_master_u[:, :, 2] = B
            print(" adjusted saturation")

        # contrast adjust
        if 0:
            C = 20.0
            F = 259 * (C + 255) / (255 * (259 - C))
            for i in range(3):
                new_i = F * (np.float32(grid_master_u[:, :, i]) - 128) + 128
                grid_master_u[:, :, i] = np.uint8(np.clip(new_i, 0, 255))
            # print(' adjusted contrast ',F)

    if 0:
        # DEBUG: dump 16 bit tiff without clipping
        im = np.zeros((nPixels[0], nPixels[1], 3), dtype="uint16")

        for i in range(3):
            if i == 0:
                grid_loc = band0_grid
            if i == 1:
                grid_loc = band1_grid
            if i == 2:
                grid_loc = band2_grid

            ww = np.where(grid_loc == 0.0)
            grid_loc[ww] = grid_loc[np.where(grid_loc > 0.0)].min() * 0.1  # 10x less than min
            grid_loc = np.log10(grid_loc)

            # rescale log(lum) into [0,65535]
            mVal = np.uint16(65535)
            grid_out = (grid_loc - grid_loc.min()) / (grid_loc.max() - grid_loc.min()) * mVal
            im[:, :, i] = grid_out.T
            print(" tiff: ", i, grid_loc.min(), grid_loc.max())

        import skimage.io

        skimage.io.imsave("out_%s.tif" % "-".join(bands), im, plugin="tifffile")

    config = {"ctName": "gray", "label": label, "vMM_guess": None}
    return grid_master_u, config, grid_master


def loadMassAndQuantity(sP, partType, partField, rotMatrix, rotCenter, method, weightField, indRange=None):
    """Load the field(s) needed to make a projection type grid, with any unit preprocessing."""
    # mass/weights
    if weightField != "mass":
        print("NOTE: Weighting by particle property [%s] instead of mass!" % weightField)

    if partType in ["dm"]:
        mass = sP.dmParticleMass
    else:
        mass = sP.snapshotSubsetP(partType, weightField, indRange=indRange).astype("float32")
        massTot = mass.sum()  # for checks

    # neutral hydrogen (do column densities, ignore atomic vs molecular complication)
    if partField in ["HI", "HI_segmented"]:
        mass *= sP.snapshotSubsetP(partType, "xhi", indRange=indRange)

    # molecular hydrogen (Popping pre-computed files, here with abbreviated names)
    if partField in ["H2_BR", "H2_GK", "H2_KMT", "HI_BR", "HI_GK", "HI_KMT"]:
        # should generalize to colDens fields
        mass = sP.snapshotSubsetP(partType, "M" + partField, indRange=indRange).astype("float32")

    # elemental mass fraction (do column densities)
    if "metals_" in partField:
        elem_mass_frac = sP.snapshotSubsetP(partType, partField, indRange=indRange)
        mass *= elem_mass_frac

    # metal ion mass (do column densities) [e.g. "O VI", "O VI mass", "O VI frac", "O VI fracmass"]
    if " " in partField and "EW_" not in partField:
        assert "sb_" not in partField
        element = partField.split()[0]
        ionNum = partField.split()[1]
        field = "mass"

        # use cache or calculate on the fly, as needed
        mass = sP.snapshotSubsetP("gas", "%s %s %s" % (element, ionNum, field), indRange=indRange)

        mass[mass < 0] = 0.0  # clip -eps values to 0.0

    # other total sum fields (replace mass)
    # todo: should generalize such that we automatically load any such field
    if "xray" in partField:
        # xray: replace 'mass' with x-ray luminosity [10^-30 erg/s], which is then accumulated into a
        # total Lx [erg/s] per pixel, and normalized by spatial pixel size into [erg/s/kpc^2]
        mass = sP.snapshotSubsetP(partType, partField, indRange=indRange)

    if partField in ["sfr_msunyrkpc2"]:
        mass = sP.snapshotSubsetP(partType, "sfr", indRange=indRange)

    if partField in ["sfr_halpha", "halpha"]:
        mass = sP.snapshotSubsetP(partType, "sfr_halpha", indRange=indRange)

    if "tau0_" in partField:
        mass = sP.snapshotSubsetP(partType, partField, indRange=indRange)

    if partField in ["sz_yparam", "ksz_yparam", "p_sync_ska"]:
        mass = sP.snapshotSubsetP(partType, partField, indRange=indRange)

    # flux/surface brightness (replace mass)
    if "sb_" in partField:  # e.g. ['sb_H-alpha','sb_Lyman-alpha','sb_OVIII','sb_O--8-16.0067A'], no spaces!
        # zero contribution from SFing gas cells?
        zeroSfr = False
        lumUnits = False
        ergUnits = False
        dustDepletion = False

        if "_sf0" in partField:
            partField = partField.split("_sf0")[0]
            zeroSfr = True
        if "_lum" in partField:
            partField = partField.split("_lum")[0]
            lumUnits = True
        if "_ergs" in partField:
            partField = partField.replace("_ergs", "")
            ergUnits = True
        if "_dustdeplete" in partField:
            partField = partField.replace("_dustdeplete", "")
            dustDepletion = True

        partField = partField.replace("_ster", "").replace("_kpc", "").replace("_ergs", "")  # options handled later

        lineName = partField.split("_")[1].replace("-", " ")  # e.g. "O--8-16.0067A" -> "O  8 16.0067A"

        # compute line emission flux for each gas cell in [erg/s/cm^2] or [photon/s/cm^2]
        if 1:
            # use cache
            assert not zeroSfr  # not implemented in cache
            assert not lumUnits  # not implemented in cache
            assert not dustDepletion  # not implemented in cache
            mass = sP.snapshotSubsetP("gas", "%s flux" % lineName, indRange=indRange)

            if ergUnits:
                # [photon/s/cm^2] -> [erg/s/cm^2]
                cloudy_e = cloudyEmission(sP, line=lineName, redshiftInterp=True)
                wavelength = cloudy_e.lineWavelength(lineName)
                mass /= sP.units.photonWavelengthToErg(wavelength)
        else:
            e_interp = cloudyEmission(sP, line=lineName, redshiftInterp=True)
            lum = e_interp.calcGasLineLuminosity(sP, lineName, indRange=indRange, dustDepletion=dustDepletion)

            redshift = sP.redshift
            if 0:  # collab.rubin.hubbleMCT_gibleVis()
                redshift = 0.36  # mock_redshift
                print(f"Pretending snapshot is at z={redshift:.2f} instead of z={sP.redshift:.2f} for flux.")

            if lumUnits:
                mass = lum / 1e30  # 10^30 erg/s
            else:
                wavelength = e_interp.lineWavelength(lineName)
                # photon/s/cm^2 if wavelength is not None, else erg/s/cm^2
                mass = sP.units.luminosityToFlux(
                    lum, wavelength=wavelength if not ergUnits else None, redshift=redshift
                )

            assert mass.min() >= 0.0
            assert np.count_nonzero(np.isnan(mass)) == 0

        if zeroSfr:
            sfr = sP.snapshotSubsetP(partType, "sfr", indRange=indRange)
            w = np.where(sfr > 0.0)
            mass[w] = 0.0

    # equivalent width map via synthetic spectra (e.g. "EW_MgII 2803")
    if partField.startswith("EW_"):
        # only possible via voronoi projection
        assert "voronoi_proj" in method

        # load number density of relevant species as 'mass'
        from ..spectra.util import lines

        line = partField.replace("EW_", "")
        element, ionNum = lines[line]["ion"].split(" ")
        field = "mass"

        # use cache or calculate on the fly, as needed
        mass = sP.snapshotSubsetP("gas", "%s %s %s" % (element, ionNum, field), indRange=indRange)

        assert mass.min() >= 0.0
        # mass[mass < 0] = 0.0 # clip -eps values to 0.0

    # single stellar band, replace mass array with linear luminosity of each star particle
    if "stellarBand-" in partField or "stellarBandObsFrame-" in partField:
        bands = partField.split("-")[1:]
        assert len(bands) == 1

        pop = sps(sP, redshifted=("ObsFrame" in partField), dustModel="none")
        mass = pop.calcStellarLuminosities(sP, bands[0], indRange=indRange, rotMatrix=rotMatrix, rotCenter=rotCenter)

    # quantity relies on a non-trivial computation / load of another quantity
    partFieldLoad = partField

    if partField in velLOSFieldNames + velCompFieldNames:
        partFieldLoad = "vel"

    if partField in ["masspart", "particle_mass"]:
        partFieldLoad = "mass"

    # quantity and column density normalization
    normCol = False

    if (
        partFieldLoad in volDensityFields + colDensityFields + totSumFields
        or " " in partFieldLoad
        or "metals_" in partFieldLoad
        or "stellarBand-" in partFieldLoad
        or "stellarBandObsFrame-" in partFieldLoad
        or "sb_" in partFieldLoad
    ):
        # distribute 'mass' and calculate column/volume density grid
        quant = None

        if partField != "mass" and "coldens" not in partField:
            assert mass.sum() != massTot, "Error! Mass array not replaced by [%s]!" % partField

        if partFieldLoad in volDensityFields + colDensityFields or (
            " " in partFieldLoad and "mass" not in partFieldLoad and "frac" not in partFieldLoad
        ):
            normCol = True
        # if 'stellarBand-' in partFieldLoad or 'stellarBandObsFrame-' in partFieldLoad and method == 'histo':
        #    normCol = True
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        if partFieldLoad in haloCentricFields or partFieldLoad.startswith("delta_"):
            if method in ["sphMap_global", "sphMap_globalZoom", "sphMap_globalZoomOrig", "voronoi_slice"]:
                # likely in chunked load, will use refPos and refVel as set in haloImgSpecs
                quant = sP.snapshotSubsetP(partType, partFieldLoad, indRange=indRange)
            else:
                # temporary override, switch to halo specified load (for halo-centric quantities)
                assert sP.subhaloInd is not None and "_global" not in method  # must be fof-scope or subhalo-scope
                assert sP.haloInd is None or sP.subhaloInd is None  # just one

                if "_subhalo" not in method:
                    haloID = sP.subhalo(sP.subhaloInd)["SubhaloGrNr"]
                    quant = sP.snapshotSubset(partType, partFieldLoad, haloID=haloID)
                else:
                    quant = sP.snapshotSubset(partType, partFieldLoad, subhaloID=sP.subhaloInd)

                assert quant.size == indRange[1] - indRange[0] + 1
        else:
            quant = sP.snapshotSubsetP(partType, partFieldLoad, indRange=indRange)

        # nan values will corrupt imaging, in general should not have any
        w = np.where(np.isnan(quant))
        if len(w[0]):  # only expected for tcool, tcool_tff
            print(
                "Warning: Zeroing mass of [%d] of [%d] particles with NaN quant [%s]"
                % (len(w[0]), quant.size, partField)
            )
            quant[w] = 0.0
            mass[w] = 0.0

    # for an actual voronoi-based ray-tracing, we don't spread out a 'mass', but instead
    # integrate a density * pathlength directly to obtain a column/surface density
    if "voronoi_proj" in method and partFieldLoad not in volDensityFields + totSumFields:
        print("Note: Normalizing 'mass' field [%s,%s] by Volume for [%s]!" % (weightField, partField, method))
        assert partType == "gas"  # could extend to dm,stars with a auto volume estimate
        vol = sP.snapshotSubsetP(partType, "volume", indRange=indRange)
        mass /= vol  # [code mass / code volume]

    # quantity pre-processing
    if partField in ["coldens_sq_msunkpc2"]:
        # DM annihilation radiation (see Schaller 2015, Eqn 2 for real units)
        # load density estimate, square, convert back to effective mass (still col dens like)
        dm_vol = sP.snapshotSubsetP(partType, "subfind_volume", indRange=indRange)
        mass = (mass / dm_vol) ** 2.0 * dm_vol

    if partField in velLOSFieldNames + velCompFieldNames:
        quant = sP.units.particleCodeVelocityToKms(quant)  # could add hubble expansion

    if partField in ["TimebinHydro", "id"]:  # cast integers to float
        quant = np.float32(quant)

    # protect against scalar/0-dimensional (e.g. single particle) arrays
    if quant is not None and quant.size == 1 and quant.ndim == 0:
        quant = np.array([quant])

    return mass, quant, normCol


def gridBox(
    sP,
    method,
    partType,
    partField,
    nPixels,
    axes,
    projType,
    projParams,
    boxCenter,
    boxSizeImg,
    hsmlFac,
    rotMatrix,
    rotCenter,
    remapRatio,
    forceRecalculate=False,
    smoothFWHM=None,
    snapHsmlForStars=False,
    alsoSFRgasForStars=False,
    excludeSubhaloFlag=False,
    skipCellIndices=None,
    ptRestrictions=None,
    weightField="mass",
    randomNoise=None,
    **kwargs,
):
    """Caching gridding/imaging of a simulation box."""
    optionalStr = ""
    if projType != "ortho":
        optionalStr += "_%s-%s" % (projType, "_".join([str(k) + "=" + str(v) for k, v in projParams.items()]))
    if remapRatio is not None:
        optionalStr += "_remap-%g-%g-%g" % (remapRatio[0], remapRatio[1], remapRatio[2])
    if snapHsmlForStars:
        optionalStr += "_snapHsmlForStars"
    if alsoSFRgasForStars:
        optionalStr += "_alsoSFRgasForStars"
    if excludeSubhaloFlag:
        optionalStr += "_excludeSubhaloFlag"
    if skipCellIndices is not None:
        optionalStr += "_skip-%s" % str(skipCellIndices)
    if ptRestrictions is not None:
        optionalStr += "_restrict-%s" % str(ptRestrictions)
    if weightField != "mass":
        optionalStr += "_wt-%s" % weightField
    if rotCenter is not None:  # need to add rotCenter, post 17 Sep 2018
        optionalStr += str(rotCenter)
    if len(nPixels) == 3:
        optionalStr += "grid3d-%d" % nPixels[2]

    hashstr = "nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d.%g.rot-%s%s" % (
        nPixels[0],
        nPixels[1],
        boxCenter[0],
        boxCenter[1],
        boxCenter[2],
        boxSizeImg[0],
        boxSizeImg[1],
        boxSizeImg[2],
        axes[0],
        axes[1],
        hsmlFac,
        str(rotMatrix),
        optionalStr,
    )
    hashval = hashlib.sha256(hashstr.encode("utf-8")).hexdigest()[::4]

    _, sbStr, _ = sP.subboxVals()

    # if loaded/gridded data is the same, just processed differently, don't save twice
    partFieldSave = partField.replace(" fracmass", " mass")
    partFieldSave = partField.replace("_msunckpc2", "_msunkpc2")
    partFieldSave = partFieldSave.replace(" ", "_")  # convention for filenames

    saveFilename = sP.derivPath + "grids/%s/%s.%s%d.%s.%s.%s.hdf5" % (
        sbStr.replace("_", "/"),
        method,
        sbStr,
        sP.snap,
        partType,
        partFieldSave,
        hashval,
    )

    if not isdir(sP.derivPath + "grids/"):
        mkdir(sP.derivPath + "grids/")
    if not isdir(sP.derivPath + "grids/%s" % sbStr.replace("_", "/")):
        mkdir(sP.derivPath + "grids/%s" % sbStr.replace("_", "/"))

    # no particles of type exist? blank grid return (otherwise die in getHsml and wind removal)
    h = sP.snapshotHeader()

    def emptyReturn():
        print("Skip empty: [%s]!" % saveFilename.split(sP.derivPath)[1])
        grid = np.zeros(nPixels, dtype="float32")
        grid, config, data_grid = gridOutputProcess(
            sP, grid, partType, partField, boxSizeImg, nPixels, projType, method
        )
        return grid, config, data_grid

    if h["NumPart"][sP.ptNum(partType)] <= 2:
        return emptyReturn()

    # generate a 3-band composite stellar image from 3 bands
    if "stellarComp" in partField or "stellarCompObsFrame" in partField:
        autoLimits = True if "autoLimits" not in kwargs else kwargs["autoLimits"]
        return stellar3BandCompositeImage(sP, partField, method, nPixels, axes, projType, projParams, boxCenter,
            boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio, forceRecalculate,
            smoothFWHM, snapHsmlForStars, alsoSFRgasForStars, excludeSubhaloFlag,
            skipCellIndices, ptRestrictions, weightField, randomNoise, autoLimits)  # fmt: skip

    # map
    if not forceRecalculate and isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename, "r") as f:
            grid_master = f["grid"][...]
        if getuser() != "wwwrun":
            print("Loaded: [%s]" % saveFilename.split(sP.derivPath)[1])
    else:
        # will we use a complete load or a subset particle load?
        indRange = None
        indRange2 = None
        boxSizeSim = sP.boxSize
        boxSizeImgMap = boxSizeImg
        boxCenterMap = boxCenter

        nChunks = 1

        # non-zoom simulation and subhaloInd specified (plotting around a single halo): do FoF restricted load
        if not sP.isZoom and sP.subhaloInd is not None and "_global" not in method:
            sh = sP.groupCatSingle(subhaloID=sP.subhaloInd)
            gr = sP.groupCatSingle(haloID=sh["SubhaloGrNr"])

            if not sP.groupOrdered:
                raise Exception("Want to do a group-ordered load but cannot.")

            # calculate indRange
            pt = sP.ptNum(partType)
            if "_subhalo" in method:
                # subhalo scope
                startInd = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"][sP.subhaloInd, pt]
                indRange = [startInd, startInd + sh["SubhaloLenType"][pt] - 1]
            else:
                # fof scope
                startInd = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"][sh["SubhaloGrNr"], pt]
                indRange = [startInd, startInd + gr["GroupLenType"][pt] - 1]

        if method == "sphMap_globalZoom":
            # virtual box 'global' scope: all fof-scope particles of all original zooms, plus h0 outer fuzz
            assert not sP.isZoom

            pt = sP.ptNum(partType)

            with h5py.File(sP.postPath + "offsets/offsets_%03d.hdf5" % sP.snap, "r") as f:
                OuterFuzzSnapOffsetByType = f["OriginalZooms/OuterFuzzSnapOffsetByType"][()]

            indRange = [0, OuterFuzzSnapOffsetByType[1, pt]]  # end at beginning of outer fuzz of second halo

        if method == "sphMap_globalZoomOrig":
            # virtual box 'global original zoom' scope: all particles of a single original zoom
            from ..load.snapshot import _global_indices_zoomorig

            assert not sP.isZoom and sP.subhaloInd is not None

            indRange, indRange2 = _global_indices_zoomorig(sP, partType, origZoomID=None)
            nChunks = 2

        if indRange is not None and indRange[1] - indRange[0] < 1:
            return emptyReturn()

        # quantity is computed with respect to a pre-existing grid? load now
        refGrid = None
        if partField in ["velsigma_los"]:
            partFieldRef = partField.replace("sigma", "")  # e.g. 'velsigma_los' -> 'vel_los'
            projParamsLoc = dict(projParams)
            projParamsLoc["noclip"] = True
            refGrid, _, _ = gridBox(sP, method, partType, partFieldRef, nPixels, axes, projType, projParamsLoc,
                                    boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
                                    smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

        # allocate
        grid_dens = np.zeros(nPixels[::-1], dtype="float32")
        grid_quant = np.zeros(nPixels[::-1], dtype="float32")

        # if doing a minimum intensity projection, pre-fill grid_quant with infinity as we
        # accumulate per chunk by using a minimum reduction between the master grid and each chunk grid
        if "_minIP" in method:
            grid_quant.fill(np.inf)

        disableChunkLoad = (
            sP.isPartType(partType, "dm") and not sP.snapHasField(partType, "SubfindHsml") and method != "histo"
        )
        disableChunkLoad |= sP.isPartType(partType, "stars")  # use custom CalcHsml always for stars now
        disableChunkLoad |= "voronoi_" in method  # need complete mesh at once
        disableChunkLoad |= indRange is None and h["NumPart"][sP.ptNum(partType)] < 1e8

        if len(sP.data) and np.count_nonzero([key for key in sP.data.keys() if "snap%d" % sP.snap in key]):
            print(" gridBox(): have fields in sP.data, disabling chunking (possible spatial subset already applied)")
            disableChunkLoad = True
            sP.data["nThreads"] = 1  # disable parallel snapshot loading

        # if indRange is still None (full snapshot load), we will proceed chunked, unless we need
        # a full tree construction to calculate hsml values
        if indRange is None and sP.subbox is None and not disableChunkLoad:
            nChunks = np.max([1, int(h["NumPart"][sP.ptNum(partType)] ** (1.0 / 3.0) / 20.0)])
            chunkSize = int(h["NumPart"][sP.ptNum(partType)] / nChunks)
            if nChunks > 10:
                print(" gridBox(): proceeding for (%s %s) with [%d] chunks..." % (partType, partField, nChunks))

        for chunkNum in np.arange(nChunks):
            # only if nChunks>1 do we here modify indRange
            if nChunks > 1 and indRange2 is None:
                # calculate load indices (snapshotSubset is inclusive on last index) (make sure we get to the end)
                indRange = [chunkNum * chunkSize, (chunkNum + 1) * chunkSize - 1]
                if chunkNum == nChunks - 1:
                    indRange[1] = h["NumPart"][sP.ptNum(partType)] - 1
                if nChunks > 10:
                    print("  [%2d] %11d - %d" % (chunkNum, indRange[0], indRange[1]))

            if indRange2 is not None and chunkNum == 1:
                # support for second range load in second iteration (e.g. for TNG-Cluster)
                assert nChunks == 2
                indRange = indRange2

            # load: 3D positions
            pos = sP.snapshotSubsetP(partType, "pos", indRange=indRange)

            # rotation? shift points to subhalo center, rotate, and shift back
            if rotMatrix is not None:
                if rotCenter is None:
                    # use subhalo center at this snapshot
                    sh = sP.groupCatSingle(subhaloID=sP.subhaloInd)
                    rotCenter = sh["SubhaloPos"]

                    if not sP.isZoom and sP.subhaloInd is None:
                        raise Exception("Rotation in periodic box must be about a halo center.")

                pos, _ = rotateCoordinateArray(sP, pos, rotMatrix, rotCenter)

            # cuboid remapping? transform points
            if remapRatio is not None:
                boxSizeSim = 0.0  # disable periodic boundaries in mapping
                pos, _ = remapPositions(sP, pos, remapRatio, nPixels)

            # load: sizes (hsml) and manipulate as needed
            hsml = None
            if (method != "histo") and ("voronoi" not in method):
                pxScale = np.max(np.array(boxSizeImg)[axes] / nPixels)

                if "stellarBand" in partField or (partType == "stars" and "coldens" in partField):
                    if sP.star in [2, 3]:
                        # high-res i.e. single-star type models, render as point-like
                        hsml = np.zeros(pos.shape[0], dtype="float32")
                        hsml += pxScale * 1.0
                    else:
                        # TNG-type SSP models, render with neighbor-based size smoothing
                        hsml = getHsmlForPartType(
                            sP,
                            partType,
                            indRange=indRange,
                            nNGB=32,
                            useSnapHsml=snapHsmlForStars,
                            alsoSFRgasForStars=alsoSFRgasForStars,
                        )
                else:
                    hsml = getHsmlForPartType(sP, partType, indRange=indRange)

                hsml *= hsmlFac  # modulate hsml values by hsmlFac

                if sP.isPartType(partType, "stars") and sP.star == 1:
                    hsml = clipStellarHSMLs(hsml, sP, pxScale, nPixels, indRange, method=3)  # custom age-based clipping

            # load: mass/weights, quantity, and render specifications required
            mass, quant, normCol = loadMassAndQuantity(
                sP, partType, partField, rotMatrix, rotCenter, method, weightField, indRange=indRange
            )

            if (method != "histo") and ("voronoi" not in method):
                assert mass.size == 1 or (mass.size == hsml.size)

            if mass.sum() == 0:
                return emptyReturn()

            # load: skip certain cells/particles?
            if ptRestrictions is not None:
                mask = np.ones(mass.size, dtype="bool")

                for restrictionField in ptRestrictions:
                    # load
                    restriction_vals = sP.snapshotSubset(partType, restrictionField, indRange=indRange)

                    # process
                    inequality, val = ptRestrictions[restrictionField]

                    if inequality == "gt":
                        mask &= restriction_vals > val
                    if inequality == "lt":
                        mask &= restriction_vals <= val
                    if inequality == "eq":
                        mask &= restriction_vals == val

                # zero mass/weight of excluded particles/cells
                w = np.where(mask == 0)
                mass[w] = 0.0

            if skipCellIndices is not None:
                print("Erasing %.3f%% of cells." % (skipCellIndices.size / mass.size))
                mass[skipCellIndices] = 0.0

            if excludeSubhaloFlag and method == "sphMap":
                # exclude any subhalos flagged as clumps, currently for fof-scope renders only
                SubhaloFlag = sP.subhalos("SubhaloFlag")
                sub_ids = sP.snapshotSubset(partType, "subhalo_id", indRange=indRange)

                flagged_ids = np.where(SubhaloFlag == 0)[0]  # 0=bad, 1=ok
                if len(flagged_ids):
                    # cross-match
                    inds_flag, inds_snap = match(flagged_ids, sub_ids)
                    if inds_snap is not None and len(inds_snap):
                        mass[inds_snap] = 0.0
            if excludeSubhaloFlag and method != "sphMap":
                print("WARNING: excludeSubhaloFlag only implemented for method == sphMap!")

            if projType in ["perspective"]:
                assert len(projParams) == 6  # 6 parameters for perspectiveProjection()
                assert projParams["n"] != 0.0

                # instead of specifying (l,r,b,t) could specify FOV
                if "fov" in projParams:
                    assert 0  # check
                    tangent = np.tan(np.deg2rad(projParams["fov"] / 2))  # tangent of half vertical FOV angle
                    halfHeight = projParams["n"] * tangent  # half height of near plane
                    halfWidth = halfHeight * (nPixels[0] / nPixels[1])  # half width of near plane (ar = w/h)

                    projParams["l"] = halfWidth
                    projParams["r"] = -halfWidth
                    projParams["t"] = halfHeight
                    projParams["b"] = -halfHeight
                else:
                    tangent = (projParams["t"] - projParams["b"]) / projParams["n"]
                    # fov = np.rad2deg(np.arctan(tangent) * 2.0)
                    # print('fov: %.1f' % fov)

                # shift pos to boxCenter
                axis_proj = 3 - axes[0] - axes[1]

                pos[:, axes[0]] -= boxCenter[0]
                pos[:, axes[1]] -= boxCenter[1]
                pos[:, axis_proj] -= boxCenter[2]

                sP.correctPeriodicDistVecs(pos)

                cameraShift = boxSizeImg[2] / 2
                pos[:, axis_proj] -= (
                    cameraShift  # switch to camera-frame; currently assumes that camera is exactly in front
                )
                # of the image domain, but maybe should be passed in projParams?
                sP.correctPeriodicDistVecs(pos)

                if hsml is None:
                    hsml = np.zeros(pos.shape[0], dtype="float32")  # dummy (e.g. method == histo)

                pos, hsml = perspectiveProjection(
                    projParams["n"],
                    projParams["f"],
                    projParams["l"],
                    projParams["r"],
                    projParams["b"],
                    projParams["t"],
                    pos,
                    hsml,
                    np.array(axes),
                )

                # switch back from camera-frame
                pos[:, axis_proj] += cameraShift
                sP.correctPeriodicDistVecs(pos)

                boxCenterMap = [0, 0, 0]

            # non-orthographic projection? project now, converting pos from a 3-vector into a 2-vector
            hsml_1 = None

            if projType in ["equirectangular", "mollweide"]:
                assert axes == [0, 1]  # by convention
                assert projParams["fov"] == 360.0
                assert nPixels[0] == nPixels[1] * 2  # we expect to make a 2:1 aspect ratio image

                hsml_orig = hsml.copy()

                # shift pos to boxCenter
                for i in range(3):
                    pos[:, i] -= boxCenter[i]
                sP.correctPeriodicDistVecs(pos)

                # cartesian to spherical coordinates
                s_rad = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
                s_lat = np.arctan2(
                    pos[:, 2], np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
                )  # latitude (phi) in [-pi/2,pi/2], defined from XY plane up
                s_long = np.arctan2(pos[:, 1], pos[:, 0])  # longitude (lambda) in [-pi,pi]

                # restrict to sphere, instead of cube, to avoid differential ray lengths
                w = np.where(s_rad < sP.boxSize / 2)
                mass[w] = 0.0

                # hsml: convert from kpc to deg (compute angular diameter)
                w = np.where(hsml_orig > 2 * s_rad)
                hsml_orig[w] = 1.999 * s_rad[w]  # otherwise outside arcsin

                hsml = 2 * np.arcsin(hsml_orig / (2 * s_rad))

                # handle differential distortion along x/y directions
                hsml_1 = hsml.astype("float32")  # hsml_1 (hsml_y) unmodified
                hsml = hsml / np.cos(s_lat)  # hsml_0 (i.e. hsml_x) only
                hsml = hsml.astype("float32")

                # we will project in this space, periodic on the boundaries
                pos = np.zeros((s_rad.size, 2), dtype=pos.dtype)
                pos[:, 0] = s_long + np.pi  # [0,2pi]
                pos[:, 1] = s_lat + np.pi / 2  # [0,pi]

                boxSizeImgMap = [2 * np.pi, np.pi]
                boxCenterMap = [np.pi, np.pi / 2]
                boxSizeSim = [2 * np.pi, np.pi, 0.0]  # periodic on projected coordinate system extent

            if projType in ["azimuthalequidistant"]:
                assert axes == [0, 1]  # by convention
                assert projParams["fov"] == 180.0
                assert nPixels[0] == nPixels[1]  # we expect to make a 1:1 aspect ratio image

                hsml_orig = hsml.copy()

                # shift pos to boxCenter
                for i in range(3):
                    pos[:, i] -= boxCenter[i]
                sP.correctPeriodicDistVecs(pos)

                # cartesian to spherical coordinates
                s_rad = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2 + pos[:, 2] ** 2)
                s_lat = (
                    -np.arctan2(pos[:, 2], np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)) + np.pi / 2.0
                )  # latitude (phi) in [0,pi], with 0 in viewing direction
                s_long = np.arctan2(pos[:, 1], pos[:, 0])  # longitude (lambda) in [-pi,pi]

                # restrict to sphere, instead of cube, to avoid differential ray lengths
                w = np.where(s_rad > sP.boxSize / 2)
                mass[w] = 0.0

                # remove all particles which are behind the camera
                w = np.where(s_lat > np.pi / 2.0)
                mass[w] = 0.0

                # hsml: convert from kpc to deg (compute angular diameter)
                w = np.where(hsml_orig > 2 * s_rad)
                hsml_orig[w] = 1.999 * s_rad[w]  # otherwise outside arcsin

                hsml = 2 * np.arcsin(hsml_orig / (2 * s_rad))

                # handle differential distortion along x/y directions
                # i.e. the distortion in s_long is equal to s_lat/sin(s_lat) or approx by the formula below
                # (does not seem necessary, future todo)
                hsml = hsml.astype("float32")
                if 0:
                    hsml_1 = hsml.astype("float32")  # hsml_1 (hsml_r) unmodified
                    hsml = hsml * (1 + s_lat**2 / 6.0 + 7.0 * s_lat**4 / 360.0)  # hsml_0 (i.e. hsml_phi) only

                # we will project in this space, periodic on the boundaries
                pos = np.zeros((s_rad.size, 2), dtype=pos.dtype)

                pos[:, 0] = +s_lat * np.sin(s_long) + np.pi / 2.0  # [0.0, pi]
                pos[:, 1] = -s_lat * np.cos(s_long) + np.pi / 2.0  # [0.0, pi]

                boxSizeImgMap = [np.pi, np.pi]
                boxCenterMap = [np.pi / 2, np.pi / 2]
                boxSizeSim = [np.pi, np.pi, 0.0]

            # rotation? handle for view dependent quantities (e.g. velLOS) (any 3-vector really...)
            if partField in velLOSFieldNames + velCompFieldNames:
                # first compensate for subhalo CM motion (if this is a halo plot)
                if sP.isZoom or sP.subhaloInd is not None:
                    sh = sP.groupCatSingle(subhaloID=sP.zoomSubhaloID if sP.isZoom else sP.subhaloInd)
                    for i in range(3):
                        # SubhaloVel already peculiar, quant converted already in loadMassAndQuantity()
                        quant[:, i] -= sh["SubhaloVel"][i]
                else:
                    assert sP.refVel is not None
                    for i in range(3):
                        quant[:, i] -= sP.refVel[i]

                if partField in velLOSFieldNames:
                    # slice corresponding to (optionally rotated) LOS component
                    sliceIndNoRot = 3 - axes[0] - axes[1]
                    sliceIndRot = 2

                if partField in velCompFieldNames:
                    # slice corresponding to (optionally rotated) _x or _y velocity component
                    if "_x" in partField:
                        sliceIndRot = 0
                    if "_y" in partField:
                        sliceIndRot = 1
                    if "_z" in partField:
                        sliceIndRot = 2
                    sliceIndNoRot = sliceIndRot

                # do slice (convert 3-vector into scalar)
                if rotMatrix is None:
                    quant = quant[:, sliceIndNoRot]
                else:
                    quant = np.transpose(np.dot(rotMatrix, quant.transpose()))
                    quant = np.squeeze(np.array(quant[:, sliceIndRot]))
                    quant = quant.astype("float32")  # rotMatrix was posssibly in double

            assert quant is None or quant.ndim == 1  # must be scalar

            # stars requested in run with winds? if so, load SFTime to remove contaminating wind particles
            wMask = None

            if partType == "stars" and sP.winds:
                sftime = sP.snapshotSubsetP(partType, "sftime", indRange=indRange)
                assert sftime.size == mass.size
                wMask = np.where(sftime > 0.0)[0]
                if len(wMask) <= 2 and nChunks == 1:
                    return emptyReturn()

                mass = mass[wMask]
                pos = pos[wMask, :]
                if method != "histo":
                    hsml = hsml[wMask]
                if quant is not None:
                    quant = quant[wMask]

            # render
            if method in [
                "sphMap",
                "sphMap_global",
                "sphMap_globalZoom",
                "sphMap_globalZoomOrig",
                "sphMap_subhalo",
                "sphMap_minIP",
                "sphMap_maxIP",
            ]:
                # particle by particle (unordered) splat using standard SPH cubic spline kernel

                # further sub-method specification?
                maxIntProj = True if "_maxIP" in method else False
                minIntProj = True if "_minIP" in method else False

                # render
                grid_d, grid_q = sphMap(
                    pos=pos,
                    hsml=hsml,
                    mass=mass,
                    quant=quant,
                    axes=axes,
                    ndims=3,
                    boxSizeSim=boxSizeSim,
                    boxSizeImg=boxSizeImgMap,
                    boxCen=boxCenterMap,
                    nPixels=nPixels,
                    hsml_1=hsml_1,
                    colDens=normCol,
                    multi=True,
                    maxIntProj=maxIntProj,
                    minIntProj=minIntProj,
                    refGrid=refGrid,
                )

            elif method in ["histo", "histo_global", "histo_maxIP", "histo_minIP"]:
                # simple 2D histogram, particles assigned to the bin which contains them
                from scipy.stats import binned_statistic_2d

                assert hsml_1 is None  # not supported

                stat = "sum"
                if "_minIP" in method:
                    stat = "min"
                if "_maxIP" in method:
                    stat = "max"

                xMinMax = [-boxSizeImgMap[0] / 2, +boxSizeImgMap[0] / 2]
                yMinMax = [-boxSizeImgMap[1] / 2, +boxSizeImgMap[1] / 2]

                # make pos periodic relative to boxCenterMap, and slice in axes[2] dimension
                for i in range(3):
                    pos[:, i] -= boxCenterMap[i]
                sP.correctPeriodicDistVecs(pos)

                zvals = np.squeeze(pos[:, 3 - axes[0] - axes[1]])
                w = np.where(np.abs(zvals) <= boxSizeImgMap[2] * 0.5)

                xvals = np.squeeze(pos[w, axes[0]])
                yvals = np.squeeze(pos[w, axes[1]])

                if mass.ndim == 0:
                    mass = np.zeros(len(w[0]), dtype=mass.dtype) + mass
                else:
                    mass = mass[w]

                # compute mass sum grid
                grid_d, _, _, _ = binned_statistic_2d(xvals, yvals, mass, stat, bins=nPixels, range=[xMinMax, yMinMax])
                grid_d = grid_d.T

                if normCol:
                    pixelArea = (boxSizeImg[0] / nPixels[0]) * (boxSizeImg[1] / nPixels[1])
                    grid_d /= pixelArea

                # mass-weighted quantity? compute mass*quant sum grid
                grid_q = np.zeros(grid_d.shape, dtype=grid_d.dtype)

                if quant is not None:
                    quant = quant[w]

                if quant is not None and partField != "velsigma_los":
                    grid_q, _, _, _ = binned_statistic_2d(
                        xvals, yvals, mass * quant, stat, bins=nPixels, range=[xMinMax, yMinMax]
                    )
                    grid_q = grid_q.T

                # special behavior
                # def _weighted_std(values, weights):
                #    """ Return weighted standard deviation.
                #        Would enable e.g. velsigma_los with sfr/X-ray weights, except that user functions in
                #        binned_statistic_2d() cannot accept any arguments beyond the list of values in a bin.
                #        So we cannot do a weighted f(), without rewriting the internals therein. """
                #    avg = np.average(values, weights=weights)
                #    delta = values - avg
                #    var = np.average(delta*delta, weights=weights)
                #    return np.sqrt(var)
                if partField == "velsigma_los":
                    assert nChunks == 1  # otherwise not supported
                    # refGrid loaded but not used, we let np.var() re-compute the per pixel mean (var->stddev below)
                    grid_q, binx, biny, inds = binned_statistic_2d(
                        xvals, yvals, quant, np.var, bins=nPixels, range=[xMinMax, yMinMax]
                    )
                    grid_q = grid_q.T  # convention
                    grid_q *= grid_d  # pre-emptively undo normalization below == unweighted var(los_vel)
                    grid_q[np.isnan(grid_q)] = (
                        0.0  # NaN not allowed in final map, but returned by binned_statistic_2d() for empty bins
                    )
                else:
                    assert refGrid is None  # not supported except for this one field

            elif method in ["voronoi_slice", "voronoi_slice_subhalo"]:
                # Voronoi mesh slice, fof-scope or subhalo-scope, using tree nearest neighbor search (no gradients)
                # note: only for a specified non-mass quantity, in which case the map gives the direct value of the
                # nearest gas cell (no weighting relevant)
                assert axes == [0, 1]  # otherwise check everything
                assert quant is not None  # otherwise we would be imaging mass (change coldens to dens)
                assert not normCol  # meaningless
                assert hsml_1 is None  # meaningless
                assert ("_minIP" not in method) and ("_maxIP" not in method)  # meaningless

                # define (x,y) pixel centers
                pxSize = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]

                x0 = boxCenter[0] - boxSizeImg[0] / 2
                x1 = boxCenter[0] + boxSizeImg[0] / 2 - pxSize[0]
                xpts = np.linspace(x0, x1, nPixels[0]) + pxSize[0] / 2

                y0 = boxCenter[1] - boxSizeImg[1] / 2
                y1 = boxCenter[1] + boxSizeImg[1] / 2 - pxSize[1]
                ypts = np.linspace(y0, y1, nPixels[1]) + pxSize[1] / 2

                # explode into nPixels[0]*nPixels[1] arrays
                xpts, ypts = np.meshgrid(xpts, ypts, indexing="ij")

                # construct [N,3] list of search positions
                search_pos = np.zeros((nPixels[0] * nPixels[1], 3), dtype=pos.dtype)

                search_pos[:, 0] = xpts.ravel()
                search_pos[:, 1] = ypts.ravel()
                search_pos[:, 2] = boxCenter[2]  # slice location along line-of-sight

                # periodic wrap search positions
                sP.correctPeriodicPosVecs(search_pos)

                # construct tree, find nearest gas cell (parent Voronoi cell) to each pixel center
                dist, index = calcHsml(pos, sP.boxSize, posSearch=search_pos, nearest=True)

                assert index.min() >= 0 and index.max() < pos.shape[0]

                # sample values from cells onto grid pixels
                grid_d = np.ones(nPixels, dtype="float32")
                grid_q = quant[index].reshape(nPixels).T

            elif method in ["voronoi_proj", "voronoi_proj_subhalo", "voronoi_proj_global"]:
                # Voronoi mesh-based projection, either fof-scope, subhalo-scope, or global, using the tree-based
                # ray-tracing algorithm. note: chunk-based loading is here disabled.
                assert axes == [0, 1]  # otherwise check everything
                assert hsml_1 is None  # unsupported
                assert ("_minIP" not in method) and ("_maxIP" not in method)  # need to add 'mode' support

                # define (x,y) pixel centers
                pxSize = [boxSizeImg[0] / nPixels[0], boxSizeImg[1] / nPixels[1]]

                x0 = boxCenter[0] - boxSizeImg[0] / 2
                x1 = boxCenter[0] + boxSizeImg[0] / 2 - pxSize[0]
                xpts = np.linspace(x0, x1, nPixels[0]) + pxSize[0] / 2

                y0 = boxCenter[1] - boxSizeImg[1] / 2
                y1 = boxCenter[1] + boxSizeImg[1] / 2 - pxSize[1]
                ypts = np.linspace(y0, y1, nPixels[1]) + pxSize[1] / 2

                # explode into nPixels[0]*nPixels[1] arrays
                xpts, ypts = np.meshgrid(xpts, ypts, indexing="ij")

                # construct [N,3] list of search positions
                ray_pos = np.zeros((nPixels[0] * nPixels[1], 3), dtype=pos.dtype)

                ray_pos[:, 0] = xpts.ravel()
                ray_pos[:, 1] = ypts.ravel()
                ray_pos[:, 2] = boxCenter[2] - boxSizeImg[2] / 2

                total_dl = boxSizeImg[2]  # path length

                ray_dir = np.array([0, 0, 1.0], dtype="float32")  # given axes

                # periodic wrap search positions
                sP.correctPeriodicPosVecs(ray_pos)

                # decide mode and ray-trace
                if partField in colDensityFields or (
                    " " in partField and "mass" not in partField and "frac" not in partField and "EW_" not in partField
                ):
                    # sum of dl_i * quant_i for each intersected cell, and here we want quant to be a
                    # number density (or mass density), so the result is a column density (or mass surface density).
                    # we have loaded 'Masses' [code] or a mass-like field (HI mass) or a luminosity-like
                    # field (e.g. xray, erg/s), and already normalized by Volume, so we have volume densities
                    assert normCol  # we have effectively already done this in our integration
                    mode = "quant_dx_sum"
                    # mass has units of [code mass / code volume], so grid_q has [code mass / code area]
                    result = rayTrace(sP, ray_pos, ray_dir, total_dl, pos, quant=mass, mode=mode)
                    grid_d = result.reshape(nPixels).T
                    grid_q = np.zeros(nPixels, dtype="float32").T  # dummy

                elif partField in volDensityFields:
                    assert not normCol  # check what it would mean
                    mode = "quant_mean"  # unweighted
                    result = rayTrace(sP, ray_pos, ray_dir, total_dl, pos, quant=mass, mode=mode)
                    grid_d = result.reshape(nPixels).T
                    grid_q = np.zeros(nPixels, dtype="float32").T  # dummy

                elif partField in totSumFields:
                    assert 0  # needs to be checked, implement tau0 and yparam direct integrations to verify
                    assert not normCol  # check what it would mean
                    mode = "quant_sum"
                    result = rayTrace(sP, ray_pos, ray_dir, total_dl, pos, quant=mass, mode=mode)
                    grid_d = result.reshape(nPixels).T
                    grid_q = np.zeros(nPixels, dtype="float32").T  # dummy

                elif "EW_" in partField:
                    # equivalent width map via synthetic spectra: obtain full rays
                    rays_off, rays_len, rays_dl, rays_inds = rayTrace(sP, ray_pos, ray_dir, total_dl, pos, mode="full")

                    # load additional required properties
                    assert normCol
                    assert nChunks == 1
                    lineName = partField.replace("EW_", "")

                    cell_temp = sP.snapshotSubsetP("gas", "temp_sfcold", indRange=indRange)  # K

                    velLosField = "vel_" + ["x", "y", "z"][3 - axes[0] - axes[1]]
                    cell_vellos = sP.snapshotSubsetP("gas", velLosField, indRange=indRange)  # code
                    cell_vellos = sP.units.particleCodeVelocityToKms(cell_vellos)  # km/s

                    assert cell_temp.size == cell_vellos.size == mass.size

                    # create spectra and derive EW (per ray) at the same time
                    instrument = "idealized"
                    _, _, result = create_spectra_from_traced_rays(
                        sP, lineName, instrument, rays_off, rays_len, rays_dl, rays_inds, mass, cell_temp, cell_vellos
                    )

                    grid_d = result.reshape(nPixels).T
                    grid_q = np.zeros(nPixels, dtype="float32").T  # dummy

                else:
                    # weighted average of quant_i using dens_i*dl (column density) as the weight
                    assert not normCol  # check what it would mean
                    mode = "quant_weighted_dx_mean"
                    result = rayTrace(sP, ray_pos, ray_dir, total_dl, pos, quant=quant, quant2=mass, mode=mode)
                    grid_q = result.reshape(nPixels).T
                    grid_d = np.ones(nPixels, dtype="float32").T  # dummy normalization

            else:
                raise Exception("Method not implemented.")

            # accumulate for chunked processing
            if "_minIP" in method:
                w = np.where(grid_q < grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            elif "_maxIP" in method:
                w = np.where(grid_q > grid_quant)
                grid_dens[w] = grid_d[w]
                grid_quant[w] = grid_q[w]
            else:
                grid_dens += grid_d
                grid_quant += grid_q

        # normalize quantity
        grid_master = grid_dens

        if quant is not None:
            # multi=True, so global normalization by per pixel 'mass' now
            w = np.where(grid_dens > 0.0)
            grid_quant[w] /= grid_dens[w]
            grid_master = grid_quant

        # save
        with h5py.File(saveFilename, "w") as f:
            f["grid"] = grid_master
        if getuser() != "wwwrun":
            print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1], flush=True)

    # smooth down to some resolution by convolving with a Gaussian? (before log if applicable)
    if smoothFWHM is not None:
        # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
        sigma_xy = (smoothFWHM / 2.3548) / (np.array(boxSizeImg)[axes] / nPixels)
        # print('smoothFWHM: [%.2f pkpc] = sigma of [%.1f px]: ' % (smoothFWHM,sigma_xy[0]))
        grid_master = gaussian_filter(grid_master, sigma_xy, mode="reflect", truncate=5.0)

    # add random noise level/floor, e.g. sky background level
    if randomNoise is not None:
        seed = int(hashval[::2], base=16)
        np.random.seed(seed)

        noise_vals = np.random.normal(loc=0.0, scale=randomNoise, size=grid_master.shape)  # pos and neg

        grid_master += np.abs(noise_vals)  # for now, absolute value

    # handle units and come up with units label
    grid_master, config, data_grid = gridOutputProcess(
        sP, grid_master, partType, partField, boxSizeImg, nPixels, projType, method
    )

    config["boxCenter"] = boxCenter
    config["boxSizeImg"] = boxSizeImg

    if projType == "mollweide":
        # we do not yet support actual projection onto mollweide (or healpix) coordinate systems
        # instead we produce an equirectangular projection, then re-map the 2d pixel image
        # from equirectangular into mollweide (image) coordinates
        print("NOTE: Mollweide not fully tested.")
        s_long0 = 0.0

        # image (x,y) coordinates in moll
        dx = 4 * np.sqrt(2) / grid_master.shape[1]
        dy = 2 * np.sqrt(2) / grid_master.shape[0]
        x_moll = np.linspace(-2 * np.sqrt(2), 2 * np.sqrt(2) - dx, grid_master.shape[1]) + dx / 2
        y_moll = np.linspace(-np.sqrt(2), np.sqrt(2) - dy, grid_master.shape[0]) + dy / 2
        R = 1.0

        # convert x,y coordinate lists into 2d grid
        x_moll, y_moll = np.meshgrid(x_moll, y_moll, indexing="xy")

        # corresponding lat,long coordinates
        theta = np.arcsin(y_moll / (R * np.sqrt(2)))
        s_lat = np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi)  # [-pi/2, +pi/2]
        s_long = s_long0 + (np.pi * x_moll) / (2 * R * np.sqrt(2) * np.cos(theta))

        w_bad = np.where((s_long < -np.pi) | (s_long > np.pi))  # outside egg?
        # print('bad: ', len(w_bad[0]), s_long.size)
        s_long[w_bad] = 0.0

        # find pixel indices of these lat,long coordinates in the equirectangular grid
        dlong = (2 * np.pi) / grid_master.shape[0]  # long per px
        dlat = (np.pi) / grid_master.shape[1]  # lat per px
        equi_long = np.linspace(0.0, 2 * np.pi - dlong, grid_master.shape[0]) + dlong / 2 - np.pi  # [-pi,pi]
        equi_lat = np.linspace(0.0, np.pi - dlat, grid_master.shape[1]) + dlat / 2 - np.pi / 2  # [-pi/2,pi/2]

        # bilinear interpolation
        from scipy.ndimage import map_coordinates

        i2 = np.interp(s_lat, equi_lat, np.arange(equi_lat.size)).ravel()
        i1 = np.interp(s_long, equi_long, np.arange(equi_long.size)).ravel()

        if 0:
            # test: shift/change center location in vertical (latitude) direction
            frac_shift = 0.1
            i2 = (i2 + grid_master.shape[1] * frac_shift) % grid_master.shape[1]
        if 0:
            # test: shift/change center location in vertical (latitude) direction
            frac_shift = 0.5
            i1 = (i1 + grid_master.shape[0] * frac_shift) % grid_master.shape[0]

        grid_master_new = map_coordinates(grid_master, np.vstack((i1, i2)), order=1, mode="nearest")
        grid_master_new = grid_master_new.reshape(grid_master.shape)

        data_grid_new = map_coordinates(data_grid, np.vstack((i1, i2)), order=1, mode="nearest")
        data_grid_new = data_grid_new.reshape(data_grid.shape)

        # flag empty pixels
        grid_master_new[w_bad] = np.nan
        data_grid_new[w_bad] = np.nan

        # replace
        grid_master = grid_master_new
        data_grid = data_grid_new

    # temporary: something a bit peculiar here, request an entirely different grid and
    # clip the line of sight to zero (or nan) where log(n_HI)<19.0 cm^(-2)
    if 0 and partField in velLOSFieldNames:
        print("Clipping LOS velocity, visible at log(n_HI) > 19.0 only.")
        grid_nHI, _, _ = gridBox(sP, method, "gas", "HI_segmented", nPixels, axes, projType, projParams,
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
            smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

        grid_master[grid_nHI < 19.0] = np.nan

    if 0 and partField in velLOSFieldNames:
        if "noclip" not in projParams:
            print("Clipping LOS velocity, visible at SFR surface density > 0.01 msun/yr/kpc^2 only.")
            grid_sfrsd, _, _ = gridBox(sP, method, "gas", "sfr_msunyrkpc2", nPixels, axes, projType, projParams,
                boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
                smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

            grid_master[grid_sfrsd < -3.0] = np.nan

    # temporary: similar, truncate stellar_age projection at a stellar column density of
    # ~log(3.2) [msun/kpc^2] equal to the bottom of the color scale for the illustris/tng sb0 box renders
    if partField == "stellar_age":
        grid_stellarColDens, _, _ = gridBox(sP, method, "stars", "coldens_msunkpc2", nPixels, axes, projType,
            projParams, boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
            smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

        w = np.where(grid_stellarColDens < 3.0)
        grid_master[w] = 0.0  # black

    # temporary: similar, fractional total mass sum of a sub-component relative to the full, request
    # the 'full' mass grid of this particle type now and normalize
    if " fracmass" in partField:
        grid_totmass, _, data_grid_totmass = gridBox(sP, method, partType, "mass", nPixels, axes, projType, projParams,
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
            smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

        grid_master = logZeroMin(10.0**grid_master / 10.0**grid_totmass)
        data_grid = logZeroMin(10.0**data_grid / 10.0**data_grid_totmass)

    # temporary: protect kSZ LOS
    if partField == "ksz_yparam":
        assert axes[0] == 0 and axes[1] == 1

    # temporary: ayromlou baryon fraction map by summing gas+stars coldens, then normalizing by gas+stars+DM
    if 0 and partField == "coldens" and partType == "gas":
        print("NOTE: Converting gas coldens map to baryon fraction map!")

        grid_stars, _, data_grid_stars = gridBox(sP, method, "stars", "coldens", nPixels, axes, projType, projParams,
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
            smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip
        grid_dm, _, data_grid_dm = gridBox(sP, method, "dm", "coldens", nPixels, axes, projType, projParams,
            boxCenter, boxSizeImg, hsmlFac, rotMatrix, rotCenter, remapRatio,
            smoothFWHM=smoothFWHM, ptRestrictions=ptRestrictions)  # fmt: skip

        grid_gas = 10.0**grid_master
        grid_stars = 10.0**grid_stars
        grid_dm = 10.0**grid_dm

        data_grid_gas = 10.0**data_grid
        data_grid_stars = 10.0**data_grid_stars
        data_grid_dm = 10.0**data_grid_dm

        grid_master = (grid_gas + grid_stars) / (grid_gas + grid_stars + grid_dm) / sP.units.f_b  # linear
        data_grid = (data_grid_gas + data_grid_stars) / (data_grid_gas + data_grid_stars + data_grid_dm) / sP.units.f_b

    # temporary: line integral convolution test
    if "licMethod" in kwargs and kwargs["licMethod"] is not None:
        from ..vis.lic import line_integral_convolution

        # temp config
        vecSliceWidth = kwargs["licSliceDepth"]
        pixelFrac = kwargs["licPixelFrac"]
        field_pt = kwargs["licPartType"]
        field_name = kwargs["licPartField"]

        # compress vector grids along third direction to more thin slice
        boxSizeImgLoc = np.array(boxSizeImg)
        boxSizeImgLoc[3 - axes[0] - axes[1]] = sP.units.physicalKpcToCodeLength(vecSliceWidth)

        # load two grids of vector length in plot-x and plot-y directions
        vel_field = np.zeros((nPixels[0], nPixels[1], 2), dtype="float32")
        field_x = field_name + "_" + ["x", "y", "z"][axes[0]]
        field_y = field_name + "_" + ["x", "y", "z"][axes[1]]

        vel_field[:, :, 1], _, _ = gridBox(sP, method, field_pt, field_x, nPixels, axes, projType, projParams,
            boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter, remapRatio)  # fmt: skip

        vel_field[:, :, 0], _, _ = gridBox(sP, method, field_pt, field_y, nPixels, axes, projType, projParams,
            boxCenter, boxSizeImgLoc, hsmlFac, rotMatrix, rotCenter, remapRatio)  # fmt: skip

        # smoothing kernel
        from scipy.stats import norm

        gauss_kernel = norm.pdf(np.linspace(-3, 3, 25 * 2))
        # TODO: this 50 should likely scale with nPixels to maintain same image (check)
        # TODO: Perlin noise

        # create noise field and do LIC
        np.random.seed(424242)

        if kwargs["licMethod"] == 1:
            # first is half pixels black, second is 99% pixels black (make into parameter)
            noise_50 = np.random.random(nPixels) < pixelFrac

            grid_master = line_integral_convolution(noise_50, vel_field, gauss_kernel)

        if kwargs["licMethod"] == 2:
            # noise field biased by the data field, or the data field itself somehow...
            noise_template = np.random.random(nPixels) < pixelFrac
            noise_field = noise_template

            lic_output = line_integral_convolution(noise_field, vel_field, gauss_kernel)
            lic_output = np.clip(lic_output, 1e-8, 1.0)

            # multiply the LIC field [0,1] by the logged, or the linear, actual data field
            print(np.nanmin(lic_output), np.nanmax(lic_output), np.nanmean(lic_output))
            # grid_master *= lic_output # if linear, e.g. velmag
            grid_master = logZeroMin(10.0**grid_master * lic_output)  # if log, e.g. coldens

    return grid_master, config, data_grid
