"""
Calculations for optical stellar light of galaxies and galaxy colors.
"""

import numpy as np

from ..cosmo.kCorr import coeff, kCorrections


# mapping of band name -> SubhaloStellarPhotometrics[:,i] index i (currently same for all sims, otherwise move into sP)
gfmBands = {"U": 0, "B": 1, "V": 2, "K": 3, "g": 4, "r": 5, "i": 6, "z": 7}

vegaMagCorrections = {"V": 0.02, "U": 0.79, "B": -0.09}

colorModelNames = {
    "A": "p07c_nodust",
    "B": "p07c_cf00dust",
    "Br": "p07c_cf00dust_rad30pkpc",
    "C": "p07c_cf00dust_res_conv_ns1_rad30pkpc",
    "nodust": "p07c_nodust",  # same as A
    "C-30kpc-z": "p07c_cf00dust_res_conv_z_30pkpc",  # z-axis only instead of 12 healpix projections
    "snap": "snap",
}

# abbreviations or alternative band names, mapped to FSPS appropriate names
bandRenamesToFSPS = {"J": "2mass_j"}


def loadColors(sP, quantName):
    """Load either a color (difference of two band magntiudes), or just a band magnitude, for every subhalo."""
    names = {}
    for key, val in colorModelNames.items():
        names[key.lower()] = val

    # determine which color model/bands are requested
    _, model, bands = quantName.split("_")
    simColorsModel = names[model]

    if len(bands) == 2:  # otherwise single band, returning magnitudes
        bands = [bands[0], bands[1]]

    for i, band in enumerate(bands):
        if band in bandRenamesToFSPS:
            bands[i] = bandRenamesToFSPS[band]

    # load
    vals, _ = loadSimGalColors(sP, simColorsModel, bands=bands)

    return vals


def loadSimGalColors(sP, simColorsModel, colorData=None, bands=None, projs=None, rad=""):
    """Load band-magnitudes from snapshot or auxcats, and return magnitudes, colors, or complete data.

    Return band magnitudes (if bands is a single band), convert to a color (if bands
    contains two bands), or return complete loaded data (if bands is None). If loaded
    data is passed in with bands, do then magnitude computation/slicing without re-loading.
    """
    acSet = ""
    if bands is not None:
        if bands[0] in ["U", "V", "J"]:
            acSet = "UVJ_"
        if len(bands) == 2 and bands[1] in ["U", "V", "J"]:
            acSet = "UVJ_"

    acKey = "Subhalo_StellarPhot_" + acSet + simColorsModel + rad

    if colorData is None:
        # load
        if simColorsModel == "snap":
            colorData = sP.groupCat(fieldsSubhalos=["SubhaloStellarPhotometrics"])
        else:
            colorData = sP.auxCat(fields=[acKey])

    # early exit with full data?
    if bands is None:
        return colorData

    subhaloIDs = None

    # compute colors
    if simColorsModel == "snap":
        if len(bands) == 1 and bands[0] in ["g", "r", "i", "z"]:
            gc_colors = colorData[:, gfmBands[bands[0]]]
        else:
            assert len(bands) == 2  # otherwise handle (AB conversions)
            gc_colors = stellarPhotToSDSSColor(colorData, bands)
    else:
        # which subhaloIDs do these colors correspond to? skip 'subhaloIDs' in colorData for legacy reasons
        gcH = sP.groupCatHeader()
        assert gcH["Nsubgroups_Total"] == colorData[acKey].shape[0]  # otherwise need auxCat stored subIDs
        subhaloIDs = np.arange(colorData[acKey].shape[0])

        # band indices
        acBands = list(np.squeeze(colorData[acKey + "_attrs"]["bands"]))
        bandname0 = "sdss_" + bands[0] if bands[0] in ["u", "g", "r", "i", "z"] else bands[0]
        bandname0 = bandname0.lower()
        i0 = acBands.index(bandname0) if bandname0 in acBands else acBands.index(bandname0.encode())

        if len(bands) == 2:
            bandname1 = "sdss_" + bands[1] if bands[1] in ["u", "g", "r", "i", "z"] else bands[1]
            bandname1 = bandname1.lower()
            i1 = acBands.index(bandname1.lower()) if bandname1 in acBands else acBands.index(bandname1.encode())

        # multiple projections per subhalo?
        if colorData[acKey].ndim == 3:
            if projs is None:
                print(" Warning: loadSimGalColors() projs unspecified, returning [random] by default.")
                projs = "random"

            if projs == "all":
                # return all
                if len(bands) == 1:
                    gc_colors = colorData[acKey][:, i0, :]
                else:
                    gc_colors = colorData[acKey][:, i0, :] - colorData[acKey][:, i1, :]
            elif projs == "random":
                # return one per subhalo, randomly chosen
                np.random.seed(42424242)
                nums = np.random.randint(0, high=colorData[acKey].shape[2], size=colorData[acKey].shape[0])
                all_inds = range(colorData[acKey].shape[0])

                if len(bands) == 1:
                    gc_colors = colorData[acKey][all_inds, i0, nums]
                else:
                    gc_colors = colorData[acKey][all_inds, i0, nums] - colorData[acKey][all_inds, i1, nums]
            else:
                # otherwise, projs had better be an integer or a tuple
                assert isinstance(projs, (int, list, tuple))
                if len(bands) == 1:
                    gc_colors = colorData[acKey][:, i0, projs]
                else:
                    gc_colors = colorData[acKey][:, i0, projs] - colorData[acKey][:, i1, projs]
        else:
            # just one projection per subhalo
            if len(bands) == 1:
                gc_colors = colorData[acKey][:, i0]
            else:
                gc_colors = colorData[acKey][:, i0] - colorData[acKey][:, i1]

    return gc_colors, subhaloIDs


def stellarPhotToSDSSColor(photVector, bands):
    """Convert GFM_StellarPhotometrics or SubhaloStellarPhotometrics into a specified color.

    Choose the right elements and handle any necessary conversions.
    """
    colorName = "".join(bands)

    if colorName == "ui":
        # UBVK are in Vega, i is in AB, and U_AB = U_Vega + 0.79, V_AB = V_Vega + 0.02
        # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # assume Buser V = Johnson V filter (close-ish), and use Lupton2005 transformation
        # http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html
        u_sdss_AB = photVector[:, gfmBands["g"]] + (-1.0 / 0.2906) * (
            photVector[:, gfmBands["V"]] + 0.02 - photVector[:, gfmBands["g"]] - 0.0885
        )
        return u_sdss_AB - photVector[:, gfmBands["z"]]

    if colorName == "gr":
        return photVector[:, gfmBands["g"]] - photVector[:, gfmBands["r"]] + 0.0  # g,r in sdss AB magnitudes

    if colorName == "ri":
        return photVector[:, gfmBands["r"]] - photVector[:, gfmBands["i"]] + 0.0  # r,i in sdss AB magnitudes

    if colorName == "iz":
        return photVector[:, gfmBands["i"]] - photVector[:, gfmBands["z"]] + 0.0  # i,z in sdss AB magnitudes

    if colorName == "gz":
        return photVector[:, gfmBands["g"]] - photVector[:, gfmBands["z"]] + 0.0  # g,z in sdss AB magnitudes

    raise Exception("Band combination not implemented.")


def calcSDSSColors(bands, redshiftRange=None, eCorrect=False, kCorrect=False, petro=False):
    """Load the SDSS data files and compute a requested color.

    Optionally restrict to a given galaxy redshift range, correcting for extinction, and/or doing a K-correction.
    """
    from ..load.data import loadSDSSData

    assert redshiftRange is None, "Not implemented."

    sdss = loadSDSSData(petro=petro)

    # extinction correction
    if not eCorrect:
        for key in sdss.keys():
            if "extinction_" in key:
                sdss[key] *= 0.0

    sdss_color = (sdss[bands[0]] - sdss["extinction_" + bands[0]]) - (sdss[bands[1]] - sdss["extinction_" + bands[1]])
    sdss_Mstar = sdss["logMass_gran1"]

    ww = np.where(sdss["redshift"] == 0.0)[0]
    assert len(ww) == 0

    # K-correction (absolute_M = apparent_m - C - K) (color A-B = m_A-C-K_A-m_B+C+K_B=m_A-m_B+K_B-K_A)
    if kCorrect:
        kCorrs = {}

        for band in bands:
            availCorrections = [key.split("_")[1] for key in coeff.keys() if band + "_" in key]
            useCor = availCorrections[0]
            # print('Calculating K-corr for [%s] band using [%s-%s] color.' % (band,useCor[0],useCor[1]))

            cor_color = (sdss[useCor[0]] - sdss["extinction_" + useCor[0]]) - (
                sdss[useCor[1]] - sdss["extinction_" + useCor[1]]
            )

            kCorrs[band] = kCorrections(band, sdss["redshift"], useCor, cor_color)

        sdss_color += kCorrs[bands[1]] - kCorrs[bands[0]]

    return sdss_color, sdss_Mstar
