"""
TNG flagship paper: galaxy colors, color bimodality.

https://arxiv.org/abs/1707.03395
"""

from os import mkdir
from os.path import expanduser, isdir, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import gaussian_kde

from temet.cosmo.color import calcSDSSColors, loadSimGalColors
from temet.plot import subhalos
from temet.plot.config import binSize, colors, cssLabels, figsize, linestyles, lw, pStyle, sKn, sKo
from temet.plot.quantities import bandMagRange, simSubhaloQuantity
from temet.plot.util import getWhiteBlackColors, loadColorTable, setAxisColors
from temet.projects.color_analysis import (
    calcColorEvoTracks,
    characterizeColorMassPlane,
    colorTransitionTimes,
    defSimColorModel,
)
from temet.util import simParams
from temet.util.helper import closest, kde_2d, leastsq_fit, logZeroNaN, running_median
from temet.util.match import match


def galaxyColorPDF(
    sPs,
    pdf,
    bands=("u", "i"),
    simColorsModels=(defSimColorModel,),
    simRedshift=0.0,
    splitCenSat=False,
    cenOnly=False,
    stellarMassBins=None,
    addPetro=False,
    minDMFrac=None,
):
    """PDF of galaxy colors (by default: (u-i)), with no dust corrections (Vog 14b Fig 13)."""
    if cenOnly:
        assert splitCenSat is False
    allOnly = True if (splitCenSat is False and cenOnly is False) else False
    assert not isinstance(simColorsModels, str)  # should be iterable
    assert len(sPs) == 1 or len(simColorsModels) == 1

    # config
    if stellarMassBins is None:
        # default, 2 cols 3 rows
        stellarMassBins = ([9.0, 9.5], [9.5, 10.0], [10.0, 10.5], [10.5, 11.0], [11.0, 11.5], [11.5, 12.0])
    obs_color = "#333333"
    petro_color = "purple"

    eCorrect = True  # True, False
    kCorrect = True  # True, False

    # start plot
    if len(stellarMassBins) >= 4:
        figsize = (16, 9)  # 2 rows, N columns
        # figsize = (9,16) # 3 rows, N (2) columns
    else:
        figsize = (5.3, 13.5)

    fig = plt.figure(figsize=figsize)
    axes = []

    if bands[0] == "u" and bands[1] == "i":
        mag_range = [0.0, 4.5]
    if bands[0] == "g" and bands[1] == "r":
        mag_range = [-0.2, 1.2]
    if bands[0] == "r" and bands[1] == "i":
        mag_range = [0.0, 0.6]
    if bands[0] == "i" and bands[1] == "z":
        mag_range = [0.0, 0.6]

    # loop over each mass bin
    for i, stellarMassBin in enumerate(stellarMassBins):
        # panel setup
        if len(stellarMassBins) >= 4:
            iLeg = 5  # 2 # lower right (2x3)
            ax = fig.add_subplot(2, int(len(stellarMassBins) / 2), i + 1)  # 2 rows, N columns
            # ax = fig.add_subplot(3,int(len(stellarMassBins)/3),i+1) #3 rows, N columns
        else:  # N rows, 1 column
            iLeg = 0  # top (3x1)
            ax = fig.add_subplot(len(stellarMassBins), 1, i + 1)

        axes.append(ax)

        ax.set_xlim(mag_range)
        xlabel = "(%s-%s) color [ mag ]" % (bands[0], bands[1])
        ylabel = "PDF"
        Mlabel = r"%.1f < M$_{\rm \star}$ < %.1f"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # add stellar mass bin legend
        sExtra = [plt.Line2D([0], [0], color="black", lw=0.0, marker="", ls=linestyles[0])]
        lExtra = [Mlabel % (stellarMassBin[0], stellarMassBin[1])]

        legendPos = "upper left"
        legend1 = ax.legend(sExtra, lExtra, loc=legendPos, handlelength=0, handletextpad=0)
        ax.add_artist(legend1)

    # load observational points, restrict colors to mag_range as done for sims (for correct normalization)
    sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=eCorrect, kCorrect=kCorrect)

    w = np.where((sdss_color >= mag_range[0]) & (sdss_color <= mag_range[1]))
    sdss_color = sdss_color[w]
    sdss_Mstar = sdss_Mstar[w]

    if addPetro:
        sdss_c_petro, sdss_m_petro = calcSDSSColors(bands, eCorrect=eCorrect, kCorrect=kCorrect, petro=True)
        w = np.where((sdss_c_petro >= mag_range[0]) & (sdss_c_petro <= mag_range[1]))
        sdss_c_petro = sdss_c_petro[w]
        sdss_m_petro = sdss_m_petro[w]

    # loop over each fullbox run
    pMaxVals = np.zeros(len(stellarMassBins), dtype="float32")

    for sP in sPs:
        if sP.isZoom:
            continue

        # loop over dustModels, for model comparison plot
        for simColorsModel in simColorsModels:
            print("Color PDF [%s] [%s]: %s" % ("-".join(bands), simColorsModel, sP.simName))
            sP.setRedshift(simRedshift)

            # load fullbox stellar masses
            gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
            gc_masses = sP.units.codeMassToLogMsun(gc[:, sP.ptNum("stars")])

            # galaxy selection
            w_cen, w_all, w_sat = sP.cenSatSubhaloIndices()

            # determine color
            c = None
            if len(sPs) == 2 and sPs[1].simName == "TNG100-1" and sPs[0].simName == "Illustris-1":
                # if sP.simName == 'Illustris-1': c = '#9467BD' # tableau10 fifth (purple) for Illustris-1
                # if sP.simName == 'Illustris-1': c = '#8C564B' # tableau10 sixth (brown) for Illustris-1
                if sP.simName == "Illustris-1":
                    c = "#D62728"  # tableau10 fourth (red) for Illustris-1
                if sP.simName == "TNG100-1":
                    c = "#1F77B4"  # tableau10 first (blue) for TNG100-1

            # load simulation colors
            if simColorsModel[-4:] == "_all":
                # request all 12*Nside^2 projections per subhalo, flatten into 1D array (sh0p0,sh0p1,...)
                gc_colors, _ = loadSimGalColors(sP, simColorsModel[:-4], bands=bands, projs="all")
                gc_colors = np.reshape(gc_colors, gc_colors.shape[0] * gc_colors.shape[1])

                # replicate stellar masses
                from re import findall  # could replace with actual Nside return from loadSimGalColors()

                Nside = np.int32(findall(r"ns\d+", simColorsModel)[0][2:])
                assert Nside == 1

                gc_inds = np.arange(gc_masses.size, dtype="int32")
                gc_inds = np.repeat(gc_inds, 12 * Nside**2)
                gc_masses = gc_masses[gc_inds]

                # replicate galaxy selections by crossmatching original selection indices with replicated list
                origSatSize = w_sat.size
                origCenSize = w_cen.size
                _, w_cen = match(w_cen, gc_inds)
                _, w_sat = match(w_sat, gc_inds)
                _, w_all = match(w_all, gc_inds)
                assert w_sat.size == 12 * Nside**2 * origSatSize
                assert w_cen.size == 12 * Nside**2 * origCenSize
            else:
                # request a single random color per subhalo (for "_res" models), and/or for simple models
                # without multiple projections even saved
                gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands)

            assert gc_colors.size == gc_masses.size
            assert w_all.size == gc_masses.size

            if minDMFrac is not None:
                subMass = sP.groupCat(fieldsSubhalos=["SubhaloMass", "SubhaloMassType"])
                subDMMassRatio = subMass["SubhaloMassType"][:, sP.ptNum("dm")] / subMass["SubhaloMass"]

                # w = np.where(subDMMassRatio < minDMFrac)
                # print('overall %d of %d fail DM cut' % (w[0].size,subDMMassRatio.size))
                # gc2 = sP.groupCat(fieldsSubhalos=['SubhaloMassInRadType'])
                # gc_masses2 = sP.units.codeMassToLogMsun( gc2[:,sP.ptNum('stars')] )
                # w = np.where(gc_masses2 >= 9.0)
                # subDMMassRatio2 = subDMMassRatio[w]
                # w = np.where(subDMMassRatio2 < minDMFrac)
                # print('above 9.0 have %d of %d fail DM cut' % (w[0].size,subDMMassRatio.size))

                if simColorsModel[-4:] == "_all":
                    subDMMassRatio = subDMMassRatio[gc_inds]
                assert subDMMassRatio.size == gc_masses.size

            # selection:
            normFacs = np.zeros(len(stellarMassBins))
            binSize = np.zeros(len(stellarMassBins))
            nBins = np.zeros(len(stellarMassBins), dtype="int32")

            if allOnly:
                loopInds = [0, 1]  # total only, except we add centrals for the first mass bin only
            if splitCenSat:
                loopInds = [0, 1, 2]  # show total, and cen/sat decomposition all at once
            if cenOnly:
                loopInds = [1]  # centrals only

            for j in loopInds:
                if j == 0:
                    w = w_all
                if j == 1:
                    w = w_cen
                if j == 2:
                    w = w_sat

                # galaxy mass definition and color
                stellar_mass = gc_masses[w]
                galaxy_color = gc_colors[w]

                # cut on DM fractions?
                if minDMFrac is not None:
                    subDMMassRatio_loc = subDMMassRatio[w]
                    w_dm = np.where(subDMMassRatio_loc >= minDMFrac)
                    # print('keep %d of %d' % (w_dm[0].size,subDMMassRatio_loc.size))
                    # w_check = np.where(subDMMassRatio_loc < minDMFrac)
                    # print('min mean max color failing: %f %f %f' % ( np.nanmin(galaxy_color[w_check]),
                    #    np.nanmean(galaxy_color[w_check]), np.nanmax(galaxy_color[w_check])))

                    stellar_mass = stellar_mass[w_dm]
                    galaxy_color = galaxy_color[w_dm]

                # filter out subhalos with e.g. no stars
                wNotNan = np.isfinite(galaxy_color)
                galaxy_color = galaxy_color[wNotNan]
                stellar_mass = stellar_mass[wNotNan]

                # loop over each mass bin
                for i, stellarMassBin in enumerate(stellarMassBins):
                    if allOnly and j == 1 and stellarMassBin[0] > 9.0:
                        continue  # add centrals for first mass bin only, if showing total only

                    wBin = np.where(
                        (stellar_mass >= stellarMassBin[0])
                        & (stellar_mass < stellarMassBin[1])
                        & (galaxy_color >= mag_range[0])
                        & (galaxy_color < mag_range[1])
                    )

                    if j == 0 or (cenOnly and j == loopInds[0]):
                        # set normalization (such that integral of PDF is one) based on 'all galaxies'
                        nBins[i] = np.max([16, np.int32(np.sqrt(len(wBin[0])) * 1.4)])  # adaptive
                        binSize[i] = (mag_range[1] - mag_range[0]) / nBins[i]
                        normFacs[i] = 1.0 / (binSize[i] * len(wBin[0]))

                    # plot panel config
                    label = sP.simName if i == iLeg and j == loopInds[0] and splitCenSat else ""
                    alpha = 1.0 if j == loopInds[0] else 0.7
                    if not splitCenSat:
                        alpha = 0.1

                    # obs histogram
                    wObs = np.where((sdss_Mstar >= stellarMassBin[0]) & (sdss_Mstar < stellarMassBin[1]))
                    yy, xx = np.histogram(sdss_color[wObs], bins=nBins[i], range=mag_range, density=True)
                    xx = xx[:-1] + 0.5 * binSize[i]

                    # obs kde
                    xx = np.linspace(mag_range[0], mag_range[1], 200)
                    # bw_scotthalf = sdss_color[wObs].size ** (-1.0 / (sdss_color.ndim + 4.0)) * 0.5
                    kde2 = gaussian_kde(sdss_color[wObs], bw_method="scott")
                    yy_obs = kde2(xx)
                    axes[i].plot(xx, yy_obs, "-", color=obs_color, alpha=1.0, lw=3.0)
                    # axes[i].fill_between(xx, 0.0, yy_obs, facecolor=obs_color, alpha=0.1, interpolate=True)

                    if addPetro:
                        wObs = np.where((sdss_m_petro >= stellarMassBin[0]) & (sdss_m_petro < stellarMassBin[1]))
                        kde2 = gaussian_kde(sdss_c_petro[wObs], bw_method="scott")
                        yy_obs = kde2(xx)
                        axes[i].plot(xx, yy_obs, "-", color=petro_color, alpha=1.0, lw=3.0)

                    if len(wBin[0]) <= 1:
                        print(" skip sim kde no data: ", sP.simName, i)
                        continue

                    # sim histogram
                    yy, xx = np.histogram(galaxy_color[wBin], bins=nBins[i], range=mag_range)
                    # yy2 = yy.astype("float32") * normFacs[i]
                    xx = xx[:-1] + 0.5 * binSize[i]

                    # sim kde
                    if not splitCenSat:
                        xx = np.linspace(mag_range[0], mag_range[1], 200)
                        # bw_scotthalf = galaxy_color[wBin].size ** (-1.0 / (galaxy_color.ndim + 4.0)) * 0.5
                        kde1 = gaussian_kde(galaxy_color[wBin], bw_method="scott")  # scott, silvermann, or scalar
                        yy_sim = kde1(xx)

                        if len(simColorsModels) == 1:
                            # label by simulation
                            label = sP.simName if i == iLeg and j == loopInds[0] else ""
                        else:
                            # label by dust model
                            label = simColorsModel if i == iLeg and j == loopInds[0] else ""

                        # replace dust model labels by paper versions
                        label = label.replace("p07c_cf00dust_res3_conv_ns1_rad30pkpc", "Model D")
                        label = label.replace("p07c_cf00dust_res_conv_ns1_rad30pkpc_all", "Model C (all)")
                        label = label.replace("p07c_cf00dust_res_conv_ns1_rad30pkpc", "Model C")
                        label = label.replace("p07c_cf00dust", "Model B")
                        label = label.replace("p07c_nodust", "Model A")

                        alpha = 0.85 if "Illustris" in sP.simName else 1.0
                        lw = 2.0 if "Illustris" in sP.simName else 4.0
                        axes[i].plot(xx, yy_sim, linestyles[j], label=label, color=c, alpha=alpha, lw=lw)
                        alpha = 0.1 if len(stellarMassBins) >= 4 else 0.05
                        if j == 0:
                            axes[i].fill_between(xx, 0.0, yy_sim, color=c, alpha=alpha, interpolate=True)

                    pMaxVals[i] = np.max([pMaxVals[i], np.max([yy_obs.max(), yy_sim.max()])])

    # y-ranges
    for i in range(len(stellarMassBins)):
        # fix y-axis limits for talk series
        if len(stellarMassBins) == 6:
            y_max = [8.0, 6.0, 6.0, 8.0, 10.0, 10.0][i]
        if len(stellarMassBins) == 3:
            y_max = [5.0, 6.0, 9.0][i]
        axes[i].set_ylim([0.0, y_max])

    # legend (simulations) (obs)
    legendPos = "upper right"
    if len(stellarMassBins) >= 4:  # 2 rows, N columns
        legendPos = "lower left"

    handles, labels = axes[iLeg].get_legend_handles_labels()
    handlesO = [plt.Line2D([0], [0], color=obs_color, lw=3.0, marker="", ls="-")]
    labelsO = ["SDSS z<0.1"]  # DR12 fspsGranWideDust

    if addPetro:
        labelsO[0] = "SDSS z<0.1 cModelMag"
        handlesO.append(plt.Line2D([0], [0], color=petro_color, lw=3.0, marker="", ls="-"))
        labelsO.append("SDSS z<0.1 Petrosian")

    axes[iLeg].legend(handlesO + handles, labelsO + labels, loc=legendPos)

    # legend (central/satellite split)
    if splitCenSat:
        sExtra = [plt.Line2D([0], [0], color="black", lw=3.0, marker="", ls=ls) for ls in linestyles]
        lExtra = ["all galaxies", "centrals", "satellites"]

        handles, labels = axes[iLeg + 1].get_legend_handles_labels()
        axes[iLeg + 1].legend(handles + sExtra, labels + lExtra, loc="upper right")

    if allOnly and len(stellarMassBins) > 3:
        sExtra = [plt.Line2D([0], [0], color="black", lw=3.0, marker="", ls=ls) for ls in linestyles[0:2]]
        lExtra = ["all galaxies", "centrals only"]

        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles + sExtra, labels + lExtra, loc="upper right")

    pdf.savefig()
    plt.close(fig)


def calcMstarColor2dKDE(
    bands, gal_Mstar, gal_color, Mstar_range, mag_range, sP=None, simColorsModel=None, kCorrected=True
):
    """Calculate and cache (slow) 2D KDE calculation of (Mstar,color) plane for sim or data.

    Args:
      bands (list[str,2]): the two band names that define the color.
      gal_Mstar (np.ndarray): stellar masses of galaxies.
      gal_color (np.ndarray): colors of galaxies.
      Mstar_range (list[float,2]): min/max stellar mass range to consider.
      mag_range (list[float,2]): min/max color range to consider.
      sP (:py:class:`~util.simParams`): simulation instance. If None, then do observational points (SDSS z < 0.1).
      simColorsModel (str or None): required if sP is specified, the simulation color model to use.
      kCorrected (bool): whether the colors have been k-corrected.
        if False, then tag filename (assume this is handled prior in calcSDSSColors)
    """
    if sP is None:
        kStr = "" if kCorrected else "_noK"
        saveFilename = expanduser("~") + "/obs/SDSS/sdss_2dkde_%s_%d-%d_%d-%d%s.hdf5" % (
            "".join(bands),
            Mstar_range[0] * 10,
            Mstar_range[1] * 10,
            mag_range[0] * 10,
            mag_range[1] * 10,
            kStr,
        )
        dName = "kde_obs"
    else:
        assert simColorsModel is not None
        savePath = sP.derivPath + "/galMstarColor/"

        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + "galMstarColor_2dkde_%s_%s_%d_%d-%d_%d-%d.hdf5" % (
            "".join(bands),
            simColorsModel,
            sP.snap,
            Mstar_range[0] * 10,
            Mstar_range[1] * 10,
            mag_range[0] * 10,
            mag_range[1] * 10,
        )
        dName = "kde_sim"

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            xx = f["xx"][()]
            yy = f["yy"][()]
            kde_obs = f[dName][()]

        return xx, yy, kde_obs

    # calculate
    print("Calculating new: [%s]..." % saveFilename)

    xx, yy, kde2d = kde_2d(gal_Mstar, gal_color, Mstar_range, mag_range)

    # save
    with h5py.File(saveFilename, "w") as f:
        f["xx"] = xx
        f["yy"] = yy
        f[dName] = kde2d
    print("Saved: [%s]" % saveFilename)

    return xx, yy, kde2d


def galaxyColor2DPDFs(sPs, pdf, simColorsModel=defSimColorModel, splitCenSat=False, simRedshift=0.0):
    """2D contours of galaxy colors/Mstar plane, multiple bands."""
    # config
    obs_color = "#000000"
    Mstar_range = [9.0, 12.0]
    bandCombos = [["u", "i"], ["g", "r"], ["r", "i"], ["u", "r"]]  # rz, iz

    eCorrect = True  # True, False (for sdss points)
    kCorrect = False  # True, False (for sdss points)

    def _discreteReSampleMatched(obs_1dhist, sim_1dhist, nBinsDS):
        """Draw a Mstar distribution from the simulation matching that from SDSS.

        Uses a quasi inverse transform sampling method. Enables a fair comparison of
        the full 1D histogram of a color over the Mstar_range.
        """
        obsMstarHist = obs_1dhist / obs_1dhist.sum()
        simInds = np.zeros(sim_1dhist.size + 1, dtype="int32")
        binSize = (Mstar_range[1] - Mstar_range[0]) / nBinsDS

        numAdded = 0

        for k in range(nBinsDS):
            binMin = Mstar_range[0] + k * binSize
            binMax = Mstar_range[0] + (k + 1) * binSize
            simIndsBin = np.where((sim_1dhist >= binMin) & (sim_1dhist < binMax))

            nWantedToMatchObs = np.int32(obsMstarHist[k] * sim_1dhist.size)

            if len(simIndsBin[0]) == 0 or nWantedToMatchObs == 0:
                continue  # failure in small boxes to have massive halos, or low res to have small halos

            # print(k,'sim size: ',simIndsBin[0].size,'wanted: ',nWantedToMatchObs)
            indsToAdd = np.random.choice(simIndsBin[0], size=nWantedToMatchObs, replace=True)
            simInds[numAdded : numAdded + indsToAdd.size] = indsToAdd

            numAdded += indsToAdd.size

        return simInds[0:numAdded]

    # create an entire plot PER run, only one 2D sim contour set each
    for sP_target in sPs:
        # start plot
        fig = plt.figure(figsize=(figsize[0] / 0.8, figsize[1] / 0.8))
        axes = []
        axes2 = []

        # loop over each requested color
        obs1DHistos = {}

        for i, bands in enumerate(bandCombos):
            print("Color 2D PDFs [%s-%s]: obs" % (bands[0], bands[1]))

            # panel setup
            ax = fig.add_subplot(2, int(len(bandCombos) / 2), i + 1)
            axes.append(ax)
            mag_range = bandMagRange(bands)

            ax.set_xlim(Mstar_range)
            ax.set_ylim(mag_range)
            ax.set_xlabel(r"M$_{\rm \star}$ [ log M$_{\rm sun}$ ]")
            ax.set_ylabel("(%s-%s) color [ mag ]" % (bands[0], bands[1]))

            # load observational points, restrict colors to mag_range as done for sims (for correct normalization)
            sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=eCorrect, kCorrect=kCorrect)

            w = np.where(
                (sdss_color >= mag_range[0])
                & (sdss_color <= mag_range[1])
                & (sdss_Mstar >= Mstar_range[0])
                & (sdss_Mstar <= Mstar_range[1])
            )

            sdss_color = sdss_color[w]
            sdss_Mstar = sdss_Mstar[w]

            # config
            extent = [Mstar_range[0], Mstar_range[1], mag_range[0], mag_range[1]]
            cLevels = [0.2, 0.5, 0.75, 0.98]
            cAlphas = [0.05, 0.2, 0.5, 1.0]
            nKDE1D = 200
            nBins1D = 100
            nBinsDS = 40  # discrete re-sampling
            # nBins2D = [50, 100]

            # (A) create kde of observations
            xx, yy, kde_obs = calcMstarColor2dKDE(
                bands, sdss_Mstar, sdss_color, Mstar_range, mag_range, kCorrected=kCorrect
            )

            for k in range(kde_obs.shape[0]):
                kde_obs[k, :] /= kde_obs[k, :].max()  # by column normalization

            for k, cLevel in enumerate(cLevels):
                ax.contour(
                    xx, yy, kde_obs, [cLevel], colors=[obs_color], alpha=cAlphas[k], linewidths=3.0, extent=extent
                )

            # (B) hist approach
            # cc, xBins, yBins = np.histogram2d(sdss_Mstar, sdss_color, bins=nBins2D, range=[Mstar_range,mag_range])
            # for k in range(c.shape[0]):
            #    cc[k,:] /= cc[k,:].max() # by column normalization
            # ax.contour(xBins[:-1], yBins[:-1], cc.T, cLevels, extent=extent)

            # vertical 1D histogram on the right side
            ax2 = make_axes_locatable(ax).append_axes("right", size="20%", pad=0.1)
            axes2.append(ax2)

            yy, xx = np.histogram(sdss_color, bins=nBins1D, range=mag_range, density=True)
            xx = xx[:-1] + 0.5 * (mag_range[1] - mag_range[0]) / nBins1D
            ax2.plot(yy, xx, "-", color=obs_color, alpha=0.2, lw=3.0)

            obs1DHistos["".join(bands)], _ = np.histogram(sdss_Mstar, bins=nBinsDS, range=Mstar_range)

            # obs 1D kde
            xx = np.linspace(mag_range[0], mag_range[1], nKDE1D)
            kde2 = gaussian_kde(sdss_color, bw_method="scott")
            ax2.plot(kde2(xx), xx, "-", color=obs_color, alpha=0.9, lw=3.0, label="SDSS z<0.1")
            ax2.fill_betweenx(xx, 0.0, kde2(xx), facecolor=obs_color, alpha=0.05)

            ax2.set_ylim(mag_range)
            ax2.get_xaxis().set_ticks([])
            ax2.get_yaxis().set_ticks([])

        # loop over each fullbox run
        for sP in sPs:
            if sP.isZoom:
                continue

            print("Color 2D PDFs [%s] [%s]: %s" % ("-".join(bands), simColorsModel, sP.simName))
            sP.setRedshift(simRedshift)

            # load fullbox stellar masses and photometrics
            gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
            gc_masses = sP.units.codeMassToLogMsun(gc[:, sP.ptNum("stars")])

            # load simulation colors
            colorData = loadSimGalColors(sP, simColorsModel)

            # galaxy selection
            w_cen, w_all, w_sat = sP.cenSatSubhaloIndices()

            # loop over each requested color
            for i, bands in enumerate(bandCombos):
                # calculate simulation colors
                gc_colors, _ = loadSimGalColors(sP, simColorsModel, colorData=colorData, bands=bands, projs="random")

                # config for this band
                mag_range = bandMagRange(bands)
                extent = [Mstar_range[0], Mstar_range[1], mag_range[0], mag_range[1]]

                loopInds = range(1)  # total only
                if splitCenSat:
                    loopInds = range(3)

                for j in loopInds:
                    if j == 0:
                        w = w_all
                    if j == 1:
                        w = w_cen
                    if j == 2:
                        w = w_sat

                    # galaxy mass definition and color
                    stellar_mass = gc_masses[w]
                    galaxy_color = gc_colors[w]

                    wNotNan = np.isfinite(galaxy_color)  # filter out subhalos with e.g. no stars
                    galaxy_color = galaxy_color[wNotNan]
                    stellar_mass = stellar_mass[wNotNan]

                    # select in bounds
                    wBin = np.where(
                        (stellar_mass >= Mstar_range[0])
                        & (stellar_mass < Mstar_range[1])
                        & (galaxy_color >= mag_range[0])
                        & (galaxy_color < mag_range[1])
                    )

                    # 1d: resample simulated Mstar distribution to roughly matched SDSS Mstar distribution
                    binInds = _discreteReSampleMatched(obs1DHistos["".join(bands)], stellar_mass[wBin], nBinsDS)
                    simInds = wBin[0][binInds]

                    # sim 1D histogram on the side
                    # yy, xx = np.histogram(galaxy_color[simInds], bins=nBins1D, range=mag_range, density=True)
                    # xx = xx[:-1] + 0.5*(mag_range[1]-mag_range[0])/nBins1D
                    # axes2[i].plot(yy, xx, '-', color=c, alpha=0.2, lw=3.0)

                    # sim 1D KDE on the side
                    xx = np.linspace(mag_range[0], mag_range[1], nKDE1D)
                    kde = gaussian_kde(galaxy_color[simInds], bw_method="scott")

                    label = sP.simName if j == 0 else ""
                    (l,) = axes2[i].plot(kde(xx), xx, "-", alpha=1.0, lw=3.0, label=label)
                    axes2[i].fill_betweenx(xx, 0.0, kde(xx), facecolor=l.get_color(), alpha=0.1)

                    # (only one 2D contour set per plot)
                    if sP != sP_target:
                        continue

                    # (A) sim 2D kde approach
                    xx, yy, kde_sim = calcMstarColor2dKDE(
                        bands,
                        stellar_mass[wBin],
                        galaxy_color[wBin],
                        Mstar_range,
                        mag_range,
                        sP=sP,
                        simColorsModel=simColorsModel,
                    )

                    for k in range(kde_sim.shape[0]):
                        kde_sim[k, :] /= kde_sim[k, :].max()  # by column normalization

                    for k, cLevel in enumerate(cLevels):
                        axes[i].contour(
                            xx,
                            yy,
                            kde_sim,
                            [cLevel],
                            colors=[l.get_color()],
                            alpha=cAlphas[k],
                            linewidths=3.0,
                            extent=extent,
                        )

                    # (B) sim 2D histogram approach
                    # cc, xBins, yBins = np.histogram2d(stellar_mass[wBin], galaxy_color[wBin], bins=nBins2D, \
                    #                                 range=[Mstar_range,mag_range])
                    # for k in range(cc.shape[0]):
                    #    cc[k,:] /= cc[k,:].max() # by column normalization
                    # for k, cLevel in enumerate(cLevels):
                    #    axes[i].contour(xBins[:-1], yBins[:-1], cc.T, [cLevel],
                    #               colors=[c], alpha=cAlphas[k], linewidths=3.0, extent=extent)

        # legend (simulations) (obs)
        hExtra = []  # [plt.Line2D([0],[0],color=obs_color,lw=3.0,marker='',linestyle='-')]
        lExtra = []  # ['SDSS z<0.1']

        handles, labels = axes2[0].get_legend_handles_labels()
        axes[0].legend(handles + hExtra, labels + lExtra, loc="upper left")

        # legend (central/satellite split)
        if splitCenSat:
            sExtra = [plt.Line2D([0], [0], color="black", lw=3.0, marker="", linestyle=ls) for ls in linestyles]
            lExtra = ["all galaxies", "centrals", "satellites"]

            axes[1].legend(sExtra, lExtra, loc="upper left")

        pdf.savefig()
        plt.close(fig)


def viewingAngleVariation():
    """Variation of (one or two) galaxy colors as a function of viewing angle (Nside > 1). 1D Histogram."""
    # config
    nBins = 250

    sP = simParams(res=1820, run="tng", redshift=0.0)

    ac_modelA = "p07c_nodust"
    ac_modelB = "p07c_cf00dust"
    ac_modelD = "p07c_bc00dust"  # debugging, not shown
    ac_modelC_demos = ["p07c_ns4_demo_rad30pkpc", "p07c_ns8_demo_rad30pkpc"]

    bands = ["g", "r"]

    # load
    modelA_colors, _ = loadSimGalColors(sP, ac_modelA, bands=bands)
    modelB_colors, _ = loadSimGalColors(sP, ac_modelB, bands=bands)
    modelD_colors, _ = loadSimGalColors(sP, ac_modelD, bands=bands)
    modelC_colors = {}
    modelC_ids = {}

    for ac_demo in ac_modelC_demos:
        modelC_colors[ac_demo], modelC_ids[ac_demo] = loadSimGalColors(sP, ac_demo, bands=bands, projs="all")

    # start plot
    fig, ax = plt.subplots()

    mag_range = [0.3, 0.85]  # bandMagRange(bands)
    markers = ["o", "s", "D"]
    linestyles = [":", "-"]
    binSize = (mag_range[1] - mag_range[0]) / nBins

    ax.set_xlim(mag_range)
    ax.set_yscale("log")
    ax.set_ylim([4, 800])
    ax.set_xlabel("(%s-%s) color [ mag ]" % (bands[0], bands[1]))
    ax.set_ylabel(r"PDF $\int=1$")

    # loop over multiple Nside demos
    lineColors = {}

    for i, ac_demo in enumerate(ac_modelC_demos):
        # loop over each subhalo included in the demo
        for j in range(modelC_colors[ac_demo].shape[0]):
            # histogram demo color distribution
            colors = modelC_colors[ac_demo][j, :]
            sub_id = modelC_ids[ac_demo][j]

            yy, xx = np.histogram(colors, bins=nBins, range=mag_range, density=True)
            xx = xx[:-1] + 0.5 * binSize

            label = ""
            subhalo = sP.groupCatSingle(subhaloID=sub_id)
            mstar = sP.units.codeMassToLogMsun(subhalo["SubhaloMassInRadType"][sP.ptNum("star")])
            sSFR = np.log10(subhalo["SubhaloSFR"] / 10.0**mstar)
            label = r"M$_\star$=10$^{%.1f}$ sSFR=%.1f" % (mstar, sSFR)

            # keep same color per subhalo, across different Ns demos
            if i == 0:
                (l,) = ax.plot(xx, yy, linestyle=linestyles[i], lw=2.5)
                lineColors[j] = l.get_color()
            else:
                (l,) = ax.plot(xx, yy, linestyle=linestyles[i], label=label, lw=2.5, color=lineColors[j])

            # plot model A and model B values for this subhalo
            if i == 0:
                color_A = modelA_colors[sub_id]
                color_B = modelB_colors[sub_id]
                color_D = modelD_colors[sub_id]
                print(sub_id, color_A, color_B, color_D, colors.mean())

                ax.plot([color_A], [10], color=lineColors[j], marker=markers[0], lw=2.5)
                ax.plot([color_B], [12], color=lineColors[j], marker=markers[1], lw=2.5)
                # ax.plot([color_D], [16], color=lineColors[j], marker=markers[2], lw=2.5)

    # legend
    handles, labels = ax.get_legend_handles_labels()
    sExtra = [
        plt.Line2D([0], [0], color="black", marker=markers[0], lw=0.0),
        plt.Line2D([0], [0], color="black", marker=markers[1], lw=0.0),
        # plt.Line2D([0], [0], color="black", marker=markers[2], lw=0.0),
        plt.Line2D([0], [0], color="black", linestyle=linestyles[0], lw=2.5),
        plt.Line2D([0], [0], color="black", linestyle=linestyles[1], lw=2.5),
    ]
    lExtra = ["Model A", "Model B", "Model C, N$_{\\rm side}$ = 4", "Model C, N$_{\\rm side}$ = 8"]  #'Model D',

    ax.legend(handles + sExtra, labels + lExtra, loc="upper left")

    # finish plot and save
    fig.savefig("appendix1_viewing_angle_variation.pdf")
    plt.close(fig)


def colorFluxArrows2DEvo(
    sP, pdf, bands, toRedshift, cenSatSelect="cen", minCount=None, simColorsModel=defSimColorModel, arrowMethod="arrow"
):
    """Plot 'flux' arrows in the (color,Mstar) plane showing the median evolution of all galaxies in each bin."""
    assert cenSatSelect in ["all", "cen", "sat"]

    # hard-coded config
    xQuant = "mstar2_log"
    contourColor = "#555555"  #'orange'
    # arrowColor = "black"
    arrowAlpha = 0.8
    contourLw = 2.0

    nBins = 12  # or 20
    rndProjInd = 0

    if arrowMethod in ["stream", "stream_mass"]:
        nBins = 30

    mag_range = bandMagRange(bands)

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    if "sim_xvals" in sP.data:
        sim_xvals, xlabel, xMinMax = sP.data["sim_xvals"], sP.data["xlabel"], sP.data["xMinMax"]
    else:
        sim_xvals, xlabel, xMinMax, _ = simSubhaloQuantity(sP, xQuant)
        sP.data["sim_xvals"], sP.data["xlabel"], sP.data["xMinMax"] = sim_xvals, xlabel, xMinMax

    # y-axis: load/calculate evolution of simulation colors, cached in sP.data
    if "sim_colors_evo" in sP.data:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = (
            sP.data["sim_colors_evo"],
            sP.data["shID_evo"],
            sP.data["subhalo_ids"],
            sP.data["snaps"],
        )
    else:
        sim_colors_evo, shID_evo, subhalo_ids, snaps = calcColorEvoTracks(
            sP, bands=bands, simColorsModel=simColorsModel
        )
        sP.data["sim_colors_evo"], sP.data["shID_evo"], sP.data["subhalo_ids"], sP.data["snaps"] = (
            sim_colors_evo,
            shID_evo,
            subhalo_ids,
            snaps,
        )

    ylabel = "(%s-%s) color [ mag ]" % (bands[0], bands[1])

    # pick initial and final color corresponding to timewindow
    savedRedshifts = sP.snapNumToRedshift(snap=snaps)
    _, zIndTo = closest(savedRedshifts, toRedshift)
    _, zIndFrom = closest(savedRedshifts, sP.redshift)

    assert zIndTo > 0 and zIndTo < len(snaps)
    assert snaps[zIndFrom] == sP.snap

    sim_colors_from = np.squeeze(sim_colors_evo[:, rndProjInd, zIndFrom])
    sim_colors_to = np.squeeze(sim_colors_evo[:, rndProjInd, zIndTo])

    # load x-axis quantity at final (to) time
    origSnap = sP.snap
    sP.setSnap(snaps[zIndTo])
    sim_xvals_to, _, _, _ = simSubhaloQuantity(sP, xQuant)
    ageTo = sP.tage

    sP.setSnap(origSnap)

    subhalo_ids_to = np.squeeze(shID_evo[:, zIndTo])
    sim_xvals_to = sim_xvals_to[subhalo_ids_to]

    # restrict xvals to the subhaloIDs for which we have saved colors
    sim_xvals_from = sim_xvals[subhalo_ids]

    # central/satellite selection?
    wSelect_orig = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
    wSelect, _ = match(subhalo_ids, wSelect_orig)

    frac_global = float(wSelect_orig.size) / sim_xvals.size * 100
    frac_local = float(wSelect.size) / subhalo_ids.size * 100
    print(sP.simName, "-".join(bands), simColorsModel, xQuant, cenSatSelect, minCount)
    print(" time interval [%.2f] Gyr (z=%.1f to z=%.1f)" % (sP.tage - ageTo, sP.redshift, toRedshift))
    print(
        " css (%s): [%d] of global [%d] = %.1f%% (reduced to [%d] of the [%d] in colorEvo = %.1f%%)"
        % (cenSatSelect, wSelect_orig.size, sim_xvals.size, frac_global, wSelect.size, subhalo_ids.size, frac_local)
    )

    sim_colors_from = sim_colors_from[wSelect]
    sim_colors_to = sim_colors_to[wSelect]
    sim_xvals_from = sim_xvals_from[wSelect]
    sim_xvals_to = sim_xvals_to[wSelect]

    # reduce to the subset with non-NaN colors at both ends of the time interval
    wFiniteColor = np.isfinite(sim_colors_from) & np.isfinite(sim_colors_to)

    sim_colors_from = sim_colors_from[wFiniteColor]
    sim_colors_to = sim_colors_to[wFiniteColor]
    sim_xvals_from = sim_xvals_from[wFiniteColor]
    sim_xvals_to = sim_xvals_to[wFiniteColor]

    # start plot
    fig = plt.figure(figsize=figsize, facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 2d bin configuration
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins * (bbox.height / bbox.width))])
    extent = [xMinMax[0], xMinMax[1], mag_range[0], mag_range[1]]

    binSize_x = (xMinMax[1] - xMinMax[0]) / nBins2D[0]
    binSize_y = (mag_range[1] - mag_range[0]) / nBins2D[1]
    print(" nBins2D: ", nBins2D, "binSizes (x,y): ", binSize_x, binSize_y)

    # calculate arrows ('velocity') for each bin (across whole grid)
    arrows_start_x = np.zeros(nBins2D, dtype="float32")
    arrows_start_y = np.zeros(nBins2D, dtype="float32")
    arrows_end_x = np.zeros(nBins2D, dtype="float32")
    arrows_end_y = np.zeros(nBins2D, dtype="float32")
    counts = np.zeros(nBins2D, dtype="int32")

    arrows_end_x.fill(np.nan)
    arrows_end_y.fill(np.nan)

    xx = np.zeros(nBins2D[0], dtype="float32")
    yy = np.zeros(nBins2D[1], dtype="float32")

    for i in range(nBins2D[0]):
        for j in range(nBins2D[1]):
            x0 = xMinMax[0] + i * binSize_x
            x1 = xMinMax[0] + (i + 1) * binSize_x
            y0 = mag_range[0] + j * binSize_y
            y1 = mag_range[0] + (j + 1) * binSize_y

            xx[i] = 0.5 * (x0 + x1)
            yy[j] = 0.5 * (y0 + y1)

            # select in bin (at sP.redshift, e.g. z=0)
            w = np.where(
                (sim_xvals_from > x0) & (sim_xvals_from <= x1) & (sim_colors_from > y0) & (sim_colors_from <= y1)
            )

            counts[i, j] = len(w[0])
            if counts[i, j] == 0:
                continue

            # arrow end points are 2d bin centers
            arrows_end_x[i, j] = 0.5 * (x0 + x1)
            arrows_end_y[i, j] = 0.5 * (y0 + y1)

            # arrow start points are median ending (Mstar,color) values of members of this bin
            # at toRedshift, e.g. where did the z=0 occupants of this bin come from?
            arrows_start_x[i, j] = np.median(sim_xvals_to[w])
            arrows_start_y[i, j] = np.median(sim_colors_to[w])

    # delta vectors
    delta_x = arrows_end_x - arrows_start_x
    delta_y = arrows_end_y - arrows_start_y

    # smoothing? interpolation? outlier exclusion?
    for _iter in range(3):
        for i in range(1, nBins2D[0] - 1):
            for j in range(1, nBins2D[1] - 1):
                # are all neighbors missing? then set counts such that -arrows- are skipped
                dx_ngb = [
                    delta_x[i - 1, j - 1],
                    delta_x[i - 1, j],
                    delta_x[i - 1, j + 1],
                    delta_x[i + 1, j - 1],
                    delta_x[i + 1, j],
                    delta_x[i + 1, j + 1],
                    delta_x[i, j - 1],
                    delta_x[i, j + 1],
                ]
                dy_ngb = [
                    delta_y[i - 1, j - 1],
                    delta_y[i - 1, j],
                    delta_y[i - 1, j + 1],
                    delta_y[i + 1, j - 1],
                    delta_y[i + 1, j],
                    delta_y[i + 1, j + 1],
                    delta_y[i, j - 1],
                    delta_y[i, j + 1],
                ]

                dx_ngood = np.count_nonzero(np.isfinite(dx_ngb))
                dy_ngood = np.count_nonzero(np.isfinite(dy_ngb))
                if dx_ngood == 0 and dy_ngood == 0 and counts[i, j] >= 0:
                    counts[i, j] = -1

    for _iter in range(10):
        for i in range(1, nBins2D[0] - 1):
            for j in range(1, nBins2D[1] - 1):
                # is the current pixel missing (nan)? if so, make a bilinear interpolation from the
                # four immediate neighbors, so long as >=3 are non-nan, to fill in this missing pt
                if np.isnan(delta_x[i, j]):
                    ngb = [delta_x[i - 1, j], delta_x[i + 1, j], delta_x[i, j - 1], delta_x[i, j + 1]]
                    ngood = np.count_nonzero(np.isfinite(ngb))
                    if ngood >= 3:
                        delta_x[i, j] = np.nanmean(ngb)

                if np.isnan(delta_y[i, j]):
                    ngb = [delta_y[i - 1, j], delta_y[i + 1, j], delta_y[i, j - 1], delta_y[i, j + 1]]
                    ngood = np.count_nonzero(np.isfinite(ngb))
                    if ngood >= 3:
                        delta_y[i, j] = np.nanmean(ngb)

    # (A): draw individual arrows
    if arrowMethod in ["arrow", "comp"]:
        for i in range(nBins2D[0]):
            for j in range(nBins2D[1]):
                if counts[i, j] < minCount:
                    continue

                # http://matplotlib.org/examples/pylab_examples/fancyarrow_demo.html
                # http://matplotlib.org/devdocs/api/patches_api.html#matplotlib.patches.FancyArrowPatch
                posA = [arrows_start_x[i, j], arrows_start_y[i, j]]
                posB = [arrows_start_x[i, j] + delta_x[i, j], arrows_start_y[i, j] + delta_y[i, j]]
                arrowstyle = "fancy, head_width=12, head_length=12, tail_width=20"
                # arrowstyle = 'simple, head_width=12, head_length=12, tail_width=4'

                # alternating color by row (in color)
                c = "#555555" if j % 2 == 0 else "#68af5a"
                p = FancyArrowPatch(posA=posA, posB=posB, arrowstyle=arrowstyle, alpha=arrowAlpha, color=c)
                ax.add_artist(p)

    # (B): quiver
    if arrowMethod == "quiver":
        # mid,head,tail
        ax.quiver(arrows_start_x, arrows_start_y, delta_x, delta_y, color="black", angles="xy", pivot="tail")

    # (C): draw streamlines using 2d vector field
    if arrowMethod in ["stream", "stream_mass", "comp"]:
        cmap = loadColorTable("jet", plawScale=1.0, fracSubset=[0.15, 0.95])

        # image gives stellar mass growth rate (e.g. dex/Gyr) or color change (e.g. mag/Gyr)
        if arrowMethod == "stream_mass":
            delta_per_Gyr = delta_x.T / (sP.tage - ageTo)
            # delta_per_Gyr = np.sqrt( delta_x**2.0 + delta_y**2.0 ).T / (sP.tage-ageTo)
            vMinMax = [-0.05, 0.15]
            clabel = r"Rate of $M_\star$ Evolution [ log M$_{\rm sun}$ / Gyr ]"
        else:
            delta_per_Gyr = delta_y.T / (sP.tage - ageTo)
            vMinMax = [-0.06, 0.1]
            clabel = r"Rate of (%s-%s) Evolution [ mag / Gyr ]" % (bands[0], bands[1])

        img = ax.imshow(
            delta_per_Gyr,
            extent=[xMinMax[0], xMinMax[1], mag_range[0], mag_range[1]],
            alpha=1.0,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            vmin=vMinMax[0],
            vmax=vMinMax[1],
        )

        # colorbar
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(img, cax=cax, drawedges=False)

        color2 = "black"
        cb.ax.set_ylabel(clabel, color=color2)
        cb.outline.set_edgecolor(color2)
        cb.ax.yaxis.set_tick_params(color=color2)
        plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=color2)

        # bugfix for alpha<1 striping in colorbar
        # cb.solids.set_rasterized(True)
        # cb.solids.set_edgecolor("face")

        # do stream plot
        ax.streamplot(xx, yy, delta_x.T, delta_y.T, density=[1.0, 1.0], linewidth=2.0, arrowsize=1.4, color="black")

    # single box showing grid size
    if 0:
        box_x0 = xx[-3] - binSize_x / 2
        box_x1 = xx[-2] - binSize_x / 2
        box_y0 = yy[-2] - binSize_y / 2
        box_y1 = yy[-1] - binSize_y / 2

        x = [box_x0, box_x1, box_x1, box_x0, box_x0]
        y = [box_y0, box_y0, box_y1, box_y1, box_y0]

        ax.plot(x, y, ":", color="#000000", alpha=0.6)

    # full box grid
    for i in range(nBins2D[0]):
        box_x0 = xx[i] - binSize_x / 2
        box_x1 = box_x0 + binSize_x

        for j in range(nBins2D[1]):
            box_y0 = yy[j] - binSize_y / 2
            box_y1 = box_y0 + binSize_y

            x = [box_x0, box_x1, box_x1, box_x0, box_x0]
            y = [box_y0, box_y0, box_y1, box_y1, box_y0]
            ax.plot(x, y, "-", color="#000000", alpha=0.05, linewidth=0.5)

    # contours
    extent = [xMinMax[0], xMinMax[1], mag_range[0], mag_range[1]]
    cLevels = [0.25, 0.5, 0.90]
    cAlphas = [0.1, 0.25, 0.4]

    sim_colors_1d = np.squeeze(sim_colors_evo[:, rndProjInd, zIndFrom])
    xx, yy, kde_sim = calcMstarColor2dKDE(
        bands, sim_xvals, sim_colors_1d, xMinMax, mag_range, sP=sP, simColorsModel=simColorsModel
    )

    for k in range(kde_sim.shape[0]):
        kde_sim[k, :] /= kde_sim[k, :].max()  # by column normalization

    for k, cLevel in enumerate(cLevels):
        ax.contour(
            xx, yy, kde_sim, [cLevel], colors=[contourColor], alpha=cAlphas[k], linewidths=contourLw, extent=extent
        )

    # finish plot and save
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        fig.savefig(
            "arrows2d_z=%.1f_%s_%s_%s_%s_%s_%s.pdf"
            % (toRedshift, "-".join(bands), simColorsModel, xQuant, cenSatSelect, minCount, arrowMethod),
            facecolor=fig.get_facecolor(),
        )
    plt.close(fig)


def _get_red_blue_2params(params, method, iterNum):
    """Helper function for plotting color-mass plane fits."""
    if "rel" in method:
        (mu1, sigma1, mu2, sigma2, A_rel) = params
        A1 = A_rel
        A2 = 1.0 - A_rel
    else:
        (A1, mu1, sigma1, A2, mu2, sigma2) = params

    if iterNum == 0:
        val_blue = sigma1
        val_red = sigma2
    if iterNum == 1:
        val_blue = mu1
        val_red = mu2
    if iterNum == 2:
        val_blue = A1
        val_red = A2

    return val_blue, val_red


def _get_red_blue_errors(errors, method, iterNum, params, errInds=(1, 3)):
    """Helper function for plotting color-mass plane fits."""
    if "rel" in method:
        mu1 = errors[errInds, 0]
        sigma1 = errors[errInds, 1]
        mu2 = errors[errInds, 2]
        sigma2 = errors[errInds, 3]
        A1 = errors[errInds, 4]
        A2 = A1
    else:
        (A1, mu1, sigma1, A2, mu2, sigma2) = params
        mu1 = errors[errInds, 1]
        sigma1 = errors[errInds, 2]
        mu2 = errors[errInds, 4]
        sigma2 = errors[errInds, 5]
        A1 = errors[errInds, 0]
        A2 = errors[errInds, 3]

    if iterNum == 0:
        val_blue = sigma1
        val_red = sigma2
    if iterNum == 1:
        val_blue = mu1
        val_red = mu2
    if iterNum == 2:
        val_blue = A1
        val_red = A2

    return val_blue[0], val_blue[1], val_red[0], val_red[1]


def _get_red_frac(params, method):
    """Helper function for plotting color-mass plane fits."""
    if "rel" in method:
        (mu1, sigma1, mu2, sigma2, A_rel) = params
        # A1 = A_rel * ...
        fraction_red = 1.0 - A_rel
    else:
        (A1, mu1, sigma1, A2, mu2, sigma2) = params
        # integral_1 = A1 * sigma1 * np.sqrt(2 * np.pi)
        # integral_2 = A2 * sigma2 * np.sqrt(2 * np.pi)

        fraction_red = (A2 * sigma2) / (A1 * sigma1 + A2 * sigma2)  # area = A*sigma*sqrt(2pi)
    return fraction_red


def colorMassPlaneFitSummary(sPs, bands=("g", "r"), simColorsModel=defSimColorModel):
    """Plot a double panel of the red/blue mu and sigma fits vs stellar mass, simulation(s) vs SDSS."""
    # analysis config
    cenSatSelect = "all"
    method = "Crel"  # MCMC fit with relative amplitudes
    nBurnIn = 2000  # 400, 2000, 10000
    newErrors = True

    # visual config
    xMinMax = [9.0, 12.0]  # log Mstar
    mMinMax = [0.3, 0.9]  # mu
    sMinMax = [0.0, 0.16]  # sigma

    xLabel = r"M$_{\star}$ [ log M$_{\rm sun}$ ]"
    mLabel = r"$\mu_{\rm red,blue}$ [peak location in (%s-%s) mag]" % (bands[0], bands[1])
    sLabel = r"$\sigma_{\\rm red,blue}$ [width in (%s-%s) mag]" % (bands[0], bands[1])

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    alpha = 0.8
    af = 5.0  # reduce alpha by this factor for low-mass red and high-mass blue

    cBlue = "blue"
    cRed = "red"

    # load obs and sim(s)
    fits_obs = characterizeColorMassPlane(
        None,
        bands=bands,
        cenSatSelect=cenSatSelect,
        simColorsModel=simColorsModel,
        nBurnIn=nBurnIn,
        remakeFlag=False,
        newErrors=False,
    )

    masses = fits_obs["mStar"]  # bin centers
    ind_r = fits_obs["skipNBinsRed"]
    ind_b = fits_obs["skipNBinsBlue"]

    fits_sim = []
    for sP in sPs:
        fits = characterizeColorMassPlane(
            sP,
            bands=bands,
            cenSatSelect=cenSatSelect,
            simColorsModel=simColorsModel,
            nBurnIn=nBurnIn,
            remakeFlag=False,
            newErrors=newErrors,
        )
        assert np.array_equal(fits_obs["mStar"], fits["mStar"])
        fits_sim.append(fits)

    medianBin = int(fits_sim[0][method + "_errors"].shape[0] / 2)

    def _fig_helper(ax, iterNum):
        val_red = np.zeros(masses.size, dtype="float32")
        val_blue = np.zeros(masses.size, dtype="float32")

        err_red_down = np.zeros(masses.size, dtype="float32")
        err_red_up = np.zeros(masses.size, dtype="float32")
        err_blue_down = np.zeros(masses.size, dtype="float32")
        err_blue_up = np.zeros(masses.size, dtype="float32")

        sExtra = []
        lExtra = []

        # loop over simulations, then obs at the end
        for sPnum, sP in enumerate(sPs + [None]):
            if sPnum < len(sPs):
                # sim
                fits = fits_sim[sPnum]
                ls = linestyles[0 + sPnum]
                marker = ["o", "s"][sPnum]
                lw = 3.0
                lExtra.append(sP.simName)
            else:
                # obs
                fits = fits_obs
                ls = linestyles[1]
                marker = "s"
                lw = 2.0
                lExtra.append("SDSS z<0.1")

            sExtra.append(plt.Line2D([0], [0], color="black", lw=lw, marker=marker, linestyle=ls))

            params = fits["%s_errors" % method][medianBin, :, :]  # medians
            errors = fits["%s_errors" % method]

            for i in range(len(masses)):
                val_blue[i], val_red[i] = _get_red_blue_2params(params[:, i], method, iterNum)
                err_blue_down[i], err_blue_up[i], err_red_down[i], err_red_up[i] = _get_red_blue_errors(
                    errors[:, :, i], method, iterNum, params
                )

            ax.plot(masses[:-ind_b], val_blue[:-ind_b], marker + ls, color=cBlue, alpha=alpha, lw=lw)
            ax.plot(masses[ind_b:], val_blue[ind_b:], marker + ls, color=cBlue, alpha=alpha / af, lw=lw)
            ax.plot(masses[ind_r:], val_red[ind_r:], marker + ls, color=cRed, alpha=alpha, lw=lw)
            ax.plot(masses[: ind_r + 1], val_red[: ind_r + 1], marker + ls, color=cRed, alpha=alpha / af, lw=lw)

            ax.fill_between(
                masses[:-ind_b],
                err_blue_down[:-ind_b],
                err_blue_up[:-ind_b],
                facecolor=cBlue,
                alpha=0.2,
                interpolate=True,
            )
            ax.fill_between(
                masses[ind_b + 1 :],
                err_blue_down[ind_b + 1 :],
                err_blue_up[ind_b + 1 :],
                facecolor=cBlue,
                alpha=0.2 / af,
                interpolate=True,
            )

            ax.fill_between(
                masses[ind_r:], err_red_down[ind_r:], err_red_up[ind_r:], facecolor=cRed, alpha=0.2, interpolate=True
            )
            ax.fill_between(
                masses[: ind_r + 1],
                err_red_down[: ind_r + 1],
                err_red_up[: ind_r + 1],
                facecolor=cRed,
                alpha=0.2 / af,
                interpolate=True,
            )

        # make legends
        loc = "upper left" if iterNum == 0 else "lower right"
        ax.legend(sExtra, lExtra, loc=loc)

    # start figure
    fig = plt.figure(figsize=(figsize[0], figsize[1] * 2), facecolor=color1)

    # top panel: mu_{red,blue}
    ax = fig.add_subplot(212, facecolor=color1)
    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(mLabel)
    ax.set_ylim(mMinMax)

    _fig_helper(ax, iterNum=1)

    # bottom panel: sigma_{red,blue}
    ax = fig.add_subplot(211, facecolor=color1)
    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_xlabel(xLabel)
    ax.set_ylabel(sLabel)
    ax.set_ylim(sMinMax)

    _fig_helper(ax, iterNum=0)

    # finish plot and save
    zStr = "_z=%.1f" % sP.redshift if sP.redshift > 0.0 else ""
    fig.savefig(
        "figure4_colorMassPlaneFits-%s_%s_%s_%s_%s%s_mcmc%d.pdf"
        % (method, "-".join([sP.simName for sP in sPs]), "-".join(bands), cenSatSelect, simColorsModel, zStr, nBurnIn)
    )
    plt.close(fig)


def colorMassPlaneFits(sP, bands=("g", "r"), css="all", simColorsModel=defSimColorModel):
    """Plot diagnostics of double gaussian fits in the color-mass plane with different methods."""
    assert css in ["all", "cen", "sat"]

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    mag_range = [-0.5, 1.5]  # bandMagRange(bands)
    yMinMax = [0.0, 4.0]  # PDFs
    mMinMax = [9.0, 12.0]  # log Mstar
    # sMinMax = [0.0, (mag_range[1] - mag_range[0]) / 10]  # sigma

    xlabel = r"(%s-%s) color [ mag ]" % (bands[0], bands[1])
    ylabel = "PDF"
    mlabel = r"%.2f < M$_{\rm \star}$ < %.2f"

    xx = np.linspace(mag_range[0], mag_range[1], 100)

    # load obs and sim
    # fits_obs = characterizeColorMassPlane(None, bands=bands, cenSatSelect=css,
    #                                      simColorsModel=simColorsModel, remakeFlag=False)
    fits_obs = None
    fits = characterizeColorMassPlane(
        sP, bands=bands, cenSatSelect=css, simColorsModel=simColorsModel, remakeFlag=False
    )

    masses = fits["mStar"]  # bin centers
    # methods = ['A','Arel']
    # methods = ['A','Arel','B','Brel','C','Crel']
    methods = ["Arel", "Brel", "Crel"]

    def _get_y1_y2(params, method):
        """Helper function for plotting color-mass plane fits."""
        if "rel" in method:
            (mu1, sigma1, mu2, sigma2, A_rel) = params

            A1 = A_rel / np.sqrt(2 * np.pi) / sigma1
            A2 = (1.0 - A_rel) / np.sqrt(2 * np.pi) / sigma2
            y1 = A1 * np.exp(-((xx - mu1) ** 2.0) / (2.0 * sigma1**2.0))  # blue
            y2 = A2 * np.exp(-((xx - mu2) ** 2.0) / (2.0 * sigma2**2.0))  # red
        else:
            (A1, mu1, sigma1, A2, mu2, sigma2) = params

            y1 = A1 * np.exp(-((xx - mu1) ** 2.0) / (2.0 * sigma1**2.0))  # blue
            y2 = A2 * np.exp(-((xx - mu2) ** 2.0) / (2.0 * sigma2**2.0))  # red

        return y1, y2

    # (A) start plot, debugging double gaussians (two plots of 10 panels each to cover 20 mass bins)
    nCols = 2 * 0.6
    nRows = 5 * 0.6  # 0.6=visual adjust fac

    for iterNum in [0, 1]:
        fig = plt.figure(figsize=(figsize[0] * nCols, figsize[1] * nRows), facecolor=color1)

        # loop over half the mass bins, with a stride of two (one panel per mass bin)
        for i, mass in enumerate(masses[iterNum::2]):
            # get index of this mass bin in params
            data_index = np.where(masses == mass)[0][0]
            print(iterNum, i, data_index, mass)

            # start plot
            ax = fig.add_subplot(5, 2, i + 1, facecolor=color1)

            setAxisColors(ax, color2)
            ax.set_xlim(mag_range)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # histogram which has been fit
            xx_hist = fits["mColorBins"]
            yy_hist = fits["mHists"][data_index, :]
            ax.step(xx_hist, yy_hist, where="mid", color="black", alpha=0.7, label="Hist")

            # load results for a particular method, including the model
            for j, method in enumerate(methods):
                params = fits["%s_params" % method]
                y1, y2 = _get_y1_y2(params[:, data_index], method)

                ax.plot(xx, y1, linestyles[j], color="blue", alpha=0.8)
                ax.plot(xx, y2, linestyles[j], color="red", alpha=0.8)

                # obs
                if fits_obs is not None:
                    params = fits_obs["%s_params" % method]
                    y1, y2 = _get_y1_y2(params[:, data_index], method)

                    ax.plot(xx, y1, linestyles[j], color="black", alpha=0.8)
                    ax.plot(xx, y2, linestyles[j], color="gray", alpha=0.8)

            # make legend
            sExtra = []
            lExtra = []

            for j, method in enumerate(methods):
                sExtra.append(plt.Line2D([0], [0], color="black", marker="", linestyle=linestyles[j]))
                lExtra.append("Method (%s)" % method)

            sExtra.append(plt.Line2D([0], [0], color="white", marker="", linestyle=linestyles[j]))
            lExtra.append(mlabel % (mass, mass + fits["binSizeMass"]))

            ax.legend(sExtra, lExtra, loc="best", prop={"size": 11})

        # finish plot and save
        fig.savefig(
            "colorMassPlaneFits%d_%s_%s_%s_%s.pdf" % (iterNum, sP.simName, "-".join(bands), css, simColorsModel)
        )
        plt.close(fig)

    # (B,C,D) start plot, sigma/mu/A vs Mstar
    for iterNum in [0, 1, 2]:
        fig = plt.figure(figsize=figsize, facecolor=color1)

        ax = fig.add_subplot(111, facecolor=color1)

        setAxisColors(ax, color2)
        ax.set_xlim(mMinMax)
        ax.set_xlabel(r"M$_{\star}$ [ log M$_{\rm sun}$ ]")

        if iterNum == 0:
            ax.set_ylabel(r"$\sigma$ [standard deviation (%s-%s)]" % (bands[0], bands[1]))
            saveStr = "Sigma"
            ax.set_ylim([0.0, 0.4])
        if iterNum == 1:
            ax.set_ylabel(r"$\mu$ [mean (%s-%s) peak location]" % (bands[0], bands[1]))
            saveStr = "Mu"
            ax.set_ylim(mag_range)
        if iterNum == 2:
            ax.set_ylabel(r"$A$ [peak amplitude (%s-%s)]" % (bands[0], bands[1]))
            saveStr = "A"
            ax.set_ylim([0.0, 3.0])

        val_red = np.zeros(masses.size, dtype="float32")
        val_blue = np.zeros(masses.size, dtype="float32")

        # load results for a particular method, including the model
        for j, method in enumerate(methods):
            params = fits["%s_params" % method]

            for i in range(len(masses)):
                val_blue[i], val_red[i] = _get_red_blue_2params(params[:, i], method, iterNum)

            ax.plot(masses, val_blue, "o" + linestyles[j], color="blue", alpha=0.8)
            ax.plot(masses, val_red, "o" + linestyles[j], color="red", alpha=0.8)

            if fits_obs is not None:
                params = fits_obs["%s_params" % method]

                for i in range(len(masses)):
                    val_blue[i], val_red[i] = _get_red_blue_2params(params[:, i], method, iterNum)

                ax.plot(masses, val_blue, "o" + linestyles[j], color="black", alpha=0.8)
                ax.plot(masses, val_red, "o" + linestyles[j], color="gray", alpha=0.8)

        # make legend
        sExtra = [plt.Line2D([0], [0], color="black", marker="", ls=linestyles[j]) for j in range(len(methods))]
        lExtra = ["Method (%s)" % m for m in methods]
        ax.legend(sExtra, lExtra, loc="best")

        # finish plot and save
        fig.savefig("colorMassPlane-%s_%s_%s_%s_%s.pdf" % (saveStr, sP.simName, "-".join(bands), css, simColorsModel))
        plt.close(fig)

    # (E) start plot, red fraction (ratio of counts in red vs red+blue gaussians)
    fig = plt.figure(figsize=figsize, facecolor=color1)

    ax = fig.add_subplot(111, facecolor=color1)

    setAxisColors(ax, color2)
    ax.set_xlim(mMinMax)
    ax.set_xlabel(r"M$_{\star}$ [ log M$_{\rm sun}$ ]")
    ax.set_ylabel(r"Red Fraction [in (%s-%s) double Gaussian fit]" % (bands[0], bands[1]))
    ax.set_ylim([0.0, 1.0])

    fraction_red = np.zeros(masses.size, dtype="float32")

    # load results for a particular method, including the model
    for j, method in enumerate(methods):
        params = fits["%s_params" % method]

        for i in range(len(masses)):
            fraction_red[i] = _get_red_frac(params[:, i], method)

        ax.plot(masses, fraction_red, linestyles[j], color="black", alpha=0.8, lw=lw)

        if fits_obs is not None:
            params = fits_obs["%s_params" % method]

            for i in range(len(masses)):
                fraction_red[i] = _get_red_frac(params[:, i], method)

            ax.plot(masses, fraction_red, linestyles[j], color="green", alpha=0.8, lw=lw)

    # make legend
    sExtra = [plt.Line2D([0], [0], color="black", marker="", ls=linestyles[j]) for j in range(len(methods))]
    lExtra = ["Method (%s)" % m for m in methods]
    ax.legend(sExtra, lExtra, loc="best")

    # finish plot and save
    fig.savefig("colorMassPlane-RedFrac_%s_%s_%s_%s.pdf" % (sP.simName, "-".join(bands), css, simColorsModel))
    plt.close(fig)

    # (F) 2d histogram of counts given the binning setup
    fig = plt.figure(figsize=figsize, facecolor=color1)
    ax = fig.add_subplot(1, 1, 1, facecolor=color1)

    setAxisColors(ax, color2)
    ax.set_ylim(mag_range)
    ax.set_xlim(fits["xMinMax"])
    ax.set_ylabel(xlabel)
    ax.set_xlabel("Stellar Mass [ log Msun ]")

    h2d = fits["mHists"].T

    norm = Normalize(vmin=0.0, vmax=h2d.max(), clip=False)
    cmap = loadColorTable("plasma")

    h2d_rgb = cmap(norm(h2d))
    extent = [fits["xMinMax"][0], fits["xMinMax"][1], mag_range[0], mag_range[1]]
    plt.imshow(h2d_rgb, extent=extent, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap, norm=norm)

    fig.savefig("colorMassPlane2DHist_%s_%s_%s_%s.pdf" % (sP.simName, "-".join(bands), css, simColorsModel))
    plt.close(fig)


def colorTransitionTimescale(sPs, bands=("g", "r"), simColorsModel=defSimColorModel):
    """Plot the distribution of 'color transition' timescales (e.g. Delta_t_green)."""
    # analysis config
    cenSatSelects = ["all", "cen", "sat"]  # make plots with each of these together combined
    maxRedshift = 1.0  # track galaxy color evolution back from sP.redshift to maxRedshift
    nBurnIn = 2000  # 400, 2000, 10000, e.g. which mcmc results to use

    f_red = 1.0  # mu_red - f_red*sigma_red defines lower boundary for red population
    f_blue = 1.0  # mu_blue + f_blue*sigma_blue defines upper boundary for blue population

    reqPosDtGreen = True  # exclude data contaminated by boundary by requiring dt_green>0
    mStarRange = [10.5, 12.5]  # if not None, only include galaxies with z=0 stellar mass in this bin

    # visual config
    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    alpha = 0.9

    tMinMax = [0.0, 8.0]  # Gyr
    mMinMax = [9.0, 12.0]  # log Msun
    zMinMax = [0.0, 1.0]  # redshift
    nBins = 50 if mStarRange is None else 30  # for all 1d histograms and running medians
    pdfMinLog = 1e-3 if mStarRange is None else 7e-2  # set minimum y-axis value for PDFs in log(y)

    fieldLabels = {
        "dt_green": r"$\Delta t_{\rm green}$ [ Gyr ]",
        "dt_rejuv": r"$\Delta t_{\rm rejuv}$ [ Gyr ]",
        "M_blue": r"$M_\star$ at $t_{\rm blue}$ (Depature from the Blue Population) [ log M$_{\rm sun}$ ]",
        "M_redini": r"$M_\star$ at $t_{\rm red,ini}$ (Arrival into the Red Population) [ log M$_{\rm sun}$ ]",
        "z_blue": r"$t_{\rm blue}$ (Redshift leaving Blue Population)",
        "z_redini": r"$t_{\rm red,ini}$ (Redshift entering Red Population)",
        "dM_green": r"$\Delta M_{\rm \star,green}$ (Mass Growth in Green Valley) [ log M$_{\rm sun}$ ]",
        "dM_red": r"$\Delta M_{\rm \star,red}$ (Mass Growth within Red Population) [ log M$_{\rm sun}$ ]",
        "dM_redfr": r"$\Delta M_{\rm \star,red}$ / $M_{\rm \star,z=0}$",
        "mstar": r"$M_{\rm \star,z=0}$ [ log M$_{\rm sun}$ ]",
    }

    fieldLabelsShort = {
        "dt_green": fieldLabels["dt_green"],
        "dt_rejuv": fieldLabels["dt_rejuv"],
        "M_blue": r"$M_\star (t_{\rm blue})$ [ log M$_{\rm sun}$ ]",
        "M_redini": r"$M_\star (t_{\rm red,ini})$ [ log M$_{\rm sun}$ ]",
        "z_blue": r"$t_{\rm blue}$ (Redshift)",
        "z_redini": r"$t_{\rm red,ini}$ (Redshift)",
        "dM_green": r"$\Delta M_{\rm \star,green}$ [ log M$_{\rm sun}$ ]",
        "dM_red": r"$\Delta M_{\rm \star,red}$ [ log M$_{\rm sun}$ ]",
        "dM_redfr": fieldLabels["dM_redfr"],
        "mstar": fieldLabels["mstar"],
        "kappa_stars": r"$\kappa_{\rm \star,rot}  (J_z > 0)$",
        "bh_cumegy_ratio": r"BH $\int$ E$_{\rm high}$ / E$_{\rm low}$ [ log ]",
    }

    # min/max bounds when the field is the x-axis (or is being histogrammed)
    fieldMinMax = {
        "dt_green": tMinMax,
        "dt_rejuv": tMinMax,
        "M_blue": mMinMax,
        "M_redini": mMinMax,
        "z_blue": zMinMax,
        "z_redini": zMinMax,
        "dM_green": [-1.5, 1.5],
        "dM_red": [-2.0, 2.0],
        "dM_redfr": [-1.0, 1.0],
        "mstar": mMinMax,
    }

    # min/max bounds when the field is the y-axis of a median (e.g. we don't see outliers here)
    fieldMinMaxTight = {
        "dt_green": [0.0, 5.0],
        "dt_rejuv": [0.0, 5.0],
        "M_blue": mMinMax,
        "M_redini": mMinMax,
        "z_blue": zMinMax,
        "z_redini": zMinMax,
        "dM_green": [-0.1, 0.3],
        "dM_red": [-0.1, 0.3],
        "dM_redfr": [-0.2, 0.4],
        "mstar": mMinMax,
    }

    def _lognormal_pdf(x, params, fixed=None):
        """Lognormal for fitting. Note fixed is unused."""
        (mu, sigma) = params
        A = 1.0 / np.sqrt(2 * np.pi) / sigma
        y = A * np.exp(-((np.log(x) - mu) ** 2.0) / (2.0 * sigma**2.0))
        return y

    # load transition times
    data = []

    for sP in sPs:
        print(sP.simName)
        evo = colorTransitionTimes(
            sP,
            f_red=f_red,
            f_blue=f_blue,
            maxRedshift=maxRedshift,
            nBurnIn=nBurnIn,
            bands=bands,
            simColorsModel=simColorsModel,
        )

        N = evo["subhalo_ids"].size
        dataKeys = [
            "dt_green",
            "dt_rejuv",
            "z_blue",
            "M_blue",
            "z_redini",
            "M_redini",
            "N_rejuv",
            "dM_red",
            "dM_redfr",
            "dM_green",
        ]

        # matching: evo subset <-> full groupcat subhalo sample
        subhalo_id_map = np.zeros(sP.numSubhalos, dtype="int32")
        subhalo_id_map[evo["subhalo_ids"]] = np.arange(N)  # such that subhalo_id_map[subhaloID] = evo index

        css_inds = {}
        css_inds["cen"], css_inds["all"], css_inds["sat"] = sP.cenSatSubhaloIndices()
        assert cenSatSelects[0] == "all"  # otherwise logic below fails for PDF norms

        # calculate: allocate
        for k in ["dt_green", "dt_rejuv", "dM_red", "dM_redfr", "dM_green"]:
            evo[k] = np.zeros(N, dtype="float32")
            evo[k].fill(np.nan)

        sub_snap = {}
        for k in dataKeys:
            sub_snap[k] = {}

        # calculate: dt_green
        ww = np.where(np.isfinite(evo["z_redini"]) & np.isfinite(evo["z_blue"]))
        t_blue_exit = sP.units.redshiftToAgeFlat(evo["z_blue"][ww])
        t_red_entry = sP.units.redshiftToAgeFlat(evo["z_redini"][ww])

        evo["dt_green"][ww] = t_red_entry - t_blue_exit  # Gyr

        # calculate: dt_rejuv
        for subhaloID in evo["z_rejuv_start"]:
            if subhaloID not in evo["z_rejuv_stop"]:
                # never found end of rejuvation event, skip
                continue

            # loop over possibly multiple events per galaxy
            for ind in range(len(evo["z_rejuv_start"][subhaloID])):
                if ind >= len(evo["z_rejuv_stop"][subhaloID]):
                    # end of this event not reached
                    continue

                t_rejuv_start = sP.units.redshiftToAgeFlat(evo["z_rejuv_start"][subhaloID][ind])
                t_rejuv_stop = sP.units.redshiftToAgeFlat(evo["z_rejuv_stop"][subhaloID][ind])

                # store all timescales for all subhalos, unordered, for histogramming
                if t_rejuv_stop < t_rejuv_start:
                    print(" skip rejuv stop=%.2f start=%.2f" % (t_rejuv_stop, t_rejuv_start))
                    continue

                if ind == 0:
                    # store first timescale for each subhalo for corelations with other quantities
                    evo_index = subhalo_id_map[subhaloID]
                    evo["dt_rejuv"][evo_index] = t_rejuv_stop - t_rejuv_start

        # sP.redshift stellar massses and other simulation quantities
        sim_quants = {"mstar": {}, "kappa_stars": {}, "bh_cumegy_ratio": {}}

        gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
        gc_mstar = sP.units.codeMassToLogMsun(gc[:, sP.ptNum("stars")])

        ac_kappa, fieldLabels["kappa_stars"], fieldMinMax["kappa_stars"], takeLog = simSubhaloQuantity(
            sP, "Krot_oriented_stars2"
        )
        fieldMinMaxTight["kappa_stars"] = fieldMinMax["kappa_stars"]
        if takeLog:
            ac_kappa = logZeroNaN(ac_kappa)

        ac_bhratio, fieldLabels["bh_cumegy_ratio"], fieldMinMax["bh_cumegy_ratio"], takeLog = simSubhaloQuantity(
            sP, "BH_CumEgy_ratio"
        )
        fieldMinMaxTight["bh_cumegy_ratio"] = fieldMinMax["bh_cumegy_ratio"]
        if takeLog:
            ac_bhratio = logZeroNaN(ac_bhratio)

        for css in cenSatSelects:
            sim_quants["mstar"][css] = gc_mstar[css_inds[css]]
            sim_quants["kappa_stars"][css] = ac_kappa[css_inds[css]]
            sim_quants["bh_cumegy_ratio"][css] = ac_bhratio[css_inds[css]]

        # calculate: dM_green
        evo["dM_green"] = evo["M_redini"] - evo["M_blue"]

        # calculate: dM_red, dM_redfr (fractional)
        Mstar_evo = sim_quants["mstar"]["all"][evo["subhalo_ids"]]
        ww = np.where(np.isfinite(Mstar_evo) & np.isfinite(evo["M_redini"]))
        evo["dM_red"][ww] = Mstar_evo[ww] - evo["M_redini"][ww]

        evo["dM_redfr"][ww] = (10.0 ** Mstar_evo[ww] - 10.0 ** evo["M_redini"][ww]) / 10.0 ** Mstar_evo[ww]

        # stamp all quantities, arranged as snapshot subhalos
        for css in cenSatSelects:
            # cross-match evo subhalo_ids to css ids
            snap_indices, evo_indices = match(css_inds[css], evo["subhalo_ids"])

            # allocate and store split by cen/sat/all selection
            for k in dataKeys:
                sub_snap[k][css] = np.zeros(css_inds[css].size, dtype="float32")
                sub_snap[k][css].fill(np.nan)

                sub_snap[k][css][snap_indices] = evo[k][evo_indices]

        # print out some statistics
        for css in cenSatSelects:
            print(" [%s]" % css)
            N_rejuv_loc = sub_snap["N_rejuv"][css]

            for massBin in [[0.0, 15.0], [11.0, 15.0]]:
                # select in mass bin
                wMassBin = np.where(
                    np.isfinite(N_rejuv_loc)
                    & (sim_quants["mstar"][css] >= massBin[0])
                    & (sim_quants["mstar"][css] < massBin[1])
                )
                nMassBin = len(wMassBin[0])

                for num in range(evo["N_rejuv"].max() + 1):
                    count_loc = len(np.where(N_rejuv_loc[wMassBin] == num)[0])
                    frac_loc = float(count_loc) / nMassBin * 100

                    print(
                        "  massbin [%.1f %.1f] for N_rejuv=%d we have %7d [of %d] galaxies = %.2f%%"
                        % (massBin[0], massBin[1], num, count_loc, nMassBin, frac_loc)
                    )

        for css in cenSatSelects:
            print(" [%s]" % css)
            for fieldName in ["dM_red", "dM_redfr", "dM_green"]:
                field_loc = sub_snap[fieldName][css]
                mass_loc = sim_quants["mstar"][css]

                for massBin in [[0.0, 15.0], [0.0, 10.5], [9.0, 10.5], [10.5, 15.0], [11.0, 15.0]]:
                    # select in mass bin
                    wMassBin = np.where(np.isfinite(field_loc) & (mass_loc >= massBin[0]) & (mass_loc < massBin[1]))

                    field_mb = field_loc[wMassBin]

                    print(
                        "  massbin [%.1f %.1f] %s mean = %.2f median = %.2f stddev = %.2f p10,90 = [%.2f, %.2f]"
                        % (
                            massBin[0],
                            massBin[1],
                            fieldName,
                            field_mb.mean(),
                            np.median(field_mb),
                            np.std(field_mb),
                            np.percentile(field_mb, 10.0),
                            np.percentile(field_mb, 90.0),
                        )
                    )

                    if fieldName == "dM_redfr":
                        count_loc = len(np.where(field_mb >= 0.5)[0])
                        frac_loc = float(count_loc) / field_mb.size * 100
                        print("   fraction dM_redfr > 0.5 is %.2f%% [%d of %d]" % (frac_loc, count_loc, field_mb.size))

        for css in cenSatSelects:
            print(" [%s]" % css)
            dt_green_loc = sub_snap["dt_green"][css]
            w = np.where(np.isfinite(dt_green_loc))

            for dt_bin in [[-10, 0], [0, 1], [1, 2], [2, 10], [4, 10]]:
                count_loc = len(np.where((dt_green_loc[w] >= dt_bin[0]) & (dt_green_loc[w] < dt_bin[1]))[0])
                frac_loc = float(count_loc) / dt_green_loc[w].size * 100

                print(
                    "  for dt_green bin [%5.1f Gyr to %5.1f Gyr] we have %7d [of %d] galaxies = %.2f%%"
                    % (dt_bin[0], dt_bin[1], count_loc, dt_green_loc.size, frac_loc)
                )

            ww = np.where(dt_green_loc[w] > 0.0)
            dt_green_loc = dt_green_loc[w][ww]

            ww = np.where(np.isfinite(sub_snap["dt_rejuv"][css]))
            dt_rejuv_loc = sub_snap["dt_rejuv"][css][ww]

            print(
                "  *dt_green mean = %.2f median = %.2f stddev = %.2f p10,90 = [%.2f, %.2f]"
                % (
                    dt_green_loc.mean(),
                    np.median(dt_green_loc),
                    np.std(dt_green_loc),
                    np.percentile(dt_green_loc, 10.0),
                    np.percentile(dt_green_loc, 90.0),
                )
            )
            print(
                "  *dt_rejuv mean = %.2f median = %.2f stddev = %.2f p10,90 = [%.2f, %.2f]"
                % (
                    dt_rejuv_loc.mean(),
                    np.median(dt_rejuv_loc),
                    np.std(dt_rejuv_loc),
                    np.percentile(dt_rejuv_loc, 10.0),
                    np.percentile(dt_rejuv_loc, 90.0),
                )
            )

        for key in sim_quants.keys():
            sub_snap[key] = sim_quants[key]

        data.append(sub_snap)

    def _fig_helper(fieldName, yscale, saveBase, plotCSS=cenSatSelects, xLabel=None, xMinMax=None, sizeAdjust=1.0):
        """Helper for histogram plots with all css combined."""
        fig = plt.figure(figsize=(figsize[0] * sizeAdjust, figsize[1] * sizeAdjust), facecolor=color1)
        ax = fig.add_subplot(111, facecolor=color1)

        if xMinMax is None:
            xMinMax = fieldMinMax[fieldName]
        xLabel = fieldLabels[fieldName] if sizeAdjust >= 1.0 else fieldLabelsShort[fieldName]

        setAxisColors(ax, color2)
        ax.set_xlim(xMinMax)
        if yscale == "log":
            ax.set_ylim(ymin=pdfMinLog)
        ax.set_xlabel(xLabel)
        ax.set_ylabel("PDF")
        ax.set_yscale(yscale)

        yy_max = 0.0

        # plot, looping over all runs and CSSes
        for i, sP in enumerate(sPs):
            nBinsLoc = nBins if sP.res > 2000 else int(nBins / 2)

            for j, cenSatSelect in enumerate(plotCSS):
                # load quantity and restrict to histogram range (for correct normalization)
                hh = data[i][fieldName][cenSatSelect]
                ww1 = np.where(np.isfinite(hh))
                hh = hh[ww1]
                ww2 = np.where((hh >= xMinMax[0]) & (hh <= xMinMax[1]))
                hh = hh[ww2]

                if reqPosDtGreen:
                    # remove contamination from time boundaries (by requiring dt_green>0)
                    dt_green = data[i]["dt_green"][cenSatSelect][ww1][ww2]
                    with np.errstate(invalid="ignore"):
                        ww3 = np.where(dt_green > 0.0)
                    hh = hh[ww3]

                if mStarRange is not None:
                    # only include galaxies in a mstar_z0 bin
                    mstar_z0 = data[i]["mstar"][cenSatSelect][ww1][ww2]
                    if reqPosDtGreen:
                        mstar_z0 = mstar_z0[ww3]
                    with np.errstate(invalid="ignore"):
                        ww4 = np.where((mstar_z0 >= mStarRange[0]) & (mstar_z0 < mStarRange[1]))
                    hh = hh[ww4]
                    mstar_z0 = mstar_z0[ww4]

                if j == 0:
                    # set normalization (such that integral of PDF is one) based on 'all galaxies'
                    binSize = (xMinMax[1] - xMinMax[0]) / nBinsLoc
                    normFac = 1.0 / (binSize * hh.size)

                yy, xx = np.histogram(hh, bins=nBinsLoc, range=xMinMax)
                yy = yy.astype("float32") * normFac
                xx = xx[:-1] + 0.5 * binSize

                label = sP.simName if j == 0 else ""
                (l,) = ax.plot(xx, yy, lw=lw, alpha=alpha, linestyle=linestyles[j], label=label)

                if fieldName == "dM_redfr":
                    # for figure 13, add Mstar>11.0 also as dotted lines
                    mStarRangeExtras = [11.0, 12.5]
                    with np.errstate(invalid="ignore"):
                        ww5 = np.where((mstar_z0 >= mStarRangeExtras[0]) & (mstar_z0 < mStarRangeExtras[1]))
                    hh = hh[ww5]
                    mstar_z0 = mstar_z0[ww5]

                    nBinsLoc = int(nBinsLoc * 0.6)
                    binSize = (xMinMax[1] - xMinMax[0]) / nBinsLoc
                    normFac = 1.0 / (binSize * hh.size)

                    yy, xx = np.histogram(hh, bins=nBinsLoc, range=xMinMax)
                    yy = yy.astype("float32") * normFac
                    xx = xx[:-1] + 0.5 * binSize

                    ax.plot(xx, yy, lw=lw, alpha=alpha, color=l.get_color(), linestyle=linestyles[j + 1])

                if fieldName == "M_redini" and plotCSS == ["cen"]:
                    # for figure 11, add mstar at z=0 for comparison
                    hh = data[i]["mstar"][cenSatSelect]
                    ww1 = np.where(np.isfinite(hh))
                    hh = hh[ww1]
                    ww2 = np.where((hh >= xMinMax[0]) & (hh <= xMinMax[1]))
                    hh = hh[ww2]

                    if reqPosDtGreen:
                        # remove contamination from time boundaries (by requiring dt_green>0)
                        dt_green = data[i]["dt_green"][cenSatSelect][ww1][ww2]
                        with np.errstate(invalid="ignore"):
                            ww3 = np.where(dt_green > 0.0)
                        hh = hh[ww3]

                    if mStarRange is not None:
                        # only include galaxies in a mstar_z0 bin
                        mstar_z0 = data[i]["mstar"][cenSatSelect][ww1][ww2]
                        if reqPosDtGreen:
                            mstar_z0 = mstar_z0[ww3]
                        with np.errstate(invalid="ignore"):
                            ww4 = np.where((mstar_z0 >= mStarRange[0]) & (mstar_z0 < mStarRange[1]))
                        hh = hh[ww4]

                    yy, xx = np.histogram(hh, bins=nBinsLoc, range=xMinMax)
                    yy = yy.astype("float32") / (binSize * hh.size)
                    xx = xx[:-1] + 0.5 * binSize

                    alphaFac = 0.7
                    ax.plot(xx, yy, lw=lw, alpha=alpha * alphaFac, color=l.get_color(), linestyle=linestyles[2])

                if yy.max() > yy_max:
                    yy_max = yy.max()

                # fit lognormal to dt_green and plot
                if 0:
                    params_guess = [0.5, 0.1]
                    params_best, _ = leastsq_fit(_lognormal_pdf, params_guess, args=(xx, yy))
                    print("%s lognormal fit: " % fieldName, cenSatSelect, params_best)
                    xx = np.linspace(xMinMax[0] + 1e-6, xMinMax[1], 100)
                    yy = _lognormal_pdf(xx, params_best)
                    ax.plot(xx, yy, ":", color="black", lw=lw / 2, alpha=alpha)

        if yscale == "log":
            ax.set_ylim(ymax=yy_max * 1.6)
            if fieldName == "dM_redfr":
                ax.set_ylim(ymax=1e1)

        # legends
        if sizeAdjust >= 1.0:
            handles, labels = ax.get_legend_handles_labels()  # show simNames plus the css below
        else:
            handles, labels = [], []  # skip simNames

        sExtra = []
        lExtra = []
        if len(plotCSS) > 1:
            loc = "upper right"
            for j, css in enumerate(plotCSS):
                sExtra.append(plt.Line2D([0], [0], color="black", lw=lw, marker="", ls=linestyles[j]))
                if sizeAdjust >= 1.0:
                    lExtra.append(cssLabels[css])  # expanded
                else:
                    lExtra.append(css.capitalize())  # abbreviated
        else:
            labels = [label + " " + plotCSS[0].capitalize() for label in labels]
            if fieldName == "M_redini":
                # figure 11 additions
                sExtra.append(
                    plt.Line2D([0], [0], color="black", lw=lw, alpha=alpha * alphaFac, marker="", ls=linestyles[2])
                )
                lExtra.append(r"$M_\star (z=0)$ Cen")
            if fieldName == "dM_redfr":
                # figure 13 additions
                sExtra.append(plt.Line2D([0], [0], color="black", lw=lw, alpha=alpha, marker="", ls=linestyles[0]))
                sExtra.append(plt.Line2D([0], [0], color="black", lw=lw, alpha=alpha, marker="", ls=linestyles[1]))
                lExtra.append(r"$M_{\star,z=0} > 10^{10.5} M_{\rm sun}$")
                lExtra.append(r"$M_{\star,z=0} > 10^{11.0} M_{\rm sun}$")

            loc = "upper left"
            if fieldName == "dM_redfr":
                loc = "lower left"

        legend1 = ax.legend(handles + sExtra, labels + lExtra, loc=loc)
        ax.add_artist(legend1)

        # finish plot and save
        cssStr = "_css-%s" % "-".join(plotCSS)
        reqStr = "_req" if reqPosDtGreen else ""
        msStr = "_mstar-%.1f-%.1f" % (mStarRange[0], mStarRange[1]) if mStarRange is not None else ""
        fig.savefig(
            "%s%s_%s_%s_y-%s%s%s%s.pdf"
            % (
                saveBase,
                fieldName,
                "_".join([sP.simName for sP in sPs]),
                simColorsModel,
                yscale,
                cssStr,
                reqStr,
                msStr,
            ),
            transparent=(sizeAdjust < 1.0),
        )  # leave PDF background transparent for inset
        plt.close(fig)

    def _fig_helper2(xAxis, yAxis, saveBase="", plotCSS=cenSatSelects, sizeAdjust=1.0):
        """Helper for histogram plots with all css combined."""
        fig = plt.figure(figsize=(figsize[0] * sizeAdjust, figsize[1] * sizeAdjust), facecolor=color1)
        ax = fig.add_subplot(111, facecolor=color1)

        xLabel = fieldLabels[xAxis] if sizeAdjust >= 1.0 else fieldLabelsShort[xAxis]

        setAxisColors(ax, color2)
        ax.set_xlim(fieldMinMax[xAxis])
        ax.set_ylim(fieldMinMaxTight[yAxis])
        ax.set_xlabel(xLabel)
        ax.set_ylabel(fieldLabelsShort[yAxis])
        ax.set_yscale("linear")

        # plot
        binSize = (fieldMinMax[xAxis][1] - fieldMinMax[xAxis][0]) / nBins

        for i, sP in enumerate(sPs):
            for j, cenSatSelect in enumerate(plotCSS):
                x_vals = data[i][xAxis][cenSatSelect]
                y_vals = data[i][yAxis][cenSatSelect]

                ww = np.where(np.isfinite(x_vals) & np.isfinite(y_vals))
                x_vals = x_vals[ww]
                y_vals = y_vals[ww]

                if reqPosDtGreen:
                    # remove contamination from time boundaries (by requiring dt_green>0)
                    dt_green = data[i]["dt_green"][cenSatSelect][ww]
                    with np.errstate(invalid="ignore"):
                        ww2 = np.where(dt_green > 0.0)
                    x_vals = x_vals[ww2]
                    y_vals = y_vals[ww2]

                if mStarRange is not None:
                    # only include galaxies in a mstar_z0 bin
                    mstar_z0 = data[i]["mstar"][cenSatSelect][ww]
                    if reqPosDtGreen:
                        mstar_z0 = mstar_z0[ww2]
                    with np.errstate(invalid="ignore"):
                        ww3 = np.where((mstar_z0 >= mStarRange[0]) & (mstar_z0 < mStarRange[1]))
                    x_vals = x_vals[ww3]
                    y_vals = y_vals[ww3]

                if x_vals.size <= 2:
                    continue

                # median of y_vals as a function of x_vals
                assert x_vals.min() != 0.0  # otherwise check
                xm, ym, sm, pm = running_median(
                    x_vals, y_vals, binSize=binSize, skipZeros=False, percs=[10, 25, 75, 90]
                )

                if xm.size > sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)  # P[10,90]

                label = sP.simName if j == 0 else ""
                ax.plot(xm[:-1], ym[:-1], linestyles[j], color=colors[i], lw=lw, label=label)

                if j > 0:
                    # show percentile scatter only for 'all galaxies'
                    continue

                ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color=colors[i], interpolate=True, alpha=0.2)

        # legends
        if sizeAdjust >= 1.0:
            handles, labels = ax.get_legend_handles_labels()  # show simNames plus the css below
        else:
            handles, labels = [], []  # skip simNames

        sExtra = []
        lExtra = []
        if len(plotCSS) > 1:
            for j, css in enumerate(plotCSS):
                sExtra.append(plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[j]))
                if sizeAdjust >= 1.0:
                    lExtra.append(cssLabels[css])  # expanded
                else:
                    lExtra.append(css.capitalize())  # abbreviated

        legend1 = ax.legend(handles + sExtra, labels + lExtra, loc="best")
        ax.add_artist(legend1)

        # finish plot and save
        cssStr = "css-%s" % "-".join(plotCSS)
        reqStr = "_req" if reqPosDtGreen else ""
        msStr = "_mstar-%.1f-%.1f" % (mStarRange[0], mStarRange[1]) if mStarRange is not None else ""

        fig.savefig(
            "%s%s_vs_%s_%s_%s_%s%s%s.pdf"
            % (saveBase, yAxis, xAxis, "_".join([sP.simName for sP in sPs]), simColorsModel, cssStr, reqStr, msStr),
            transparent=(sizeAdjust < 1.0),
        )  # leave PDF background transparent for inset
        plt.close(fig)

    # figure 10: color transition timescale histogram
    # _fig_helper('dt_green', 'linear', 'figure10_')

    # figure 11: histogram of Mstar when joining red sequence, inset: Mstar growth in green valley
    # _fig_helper('M_redini', 'linear', 'figure11_', xMinMax=[9.5, 12.0], plotCSS=['cen'])
    # _fig_helper('dM_green', 'log', 'figure11_inset_', xMinMax=[-1.0,1.5], sizeAdjust=0.55)

    # figure 12: color transition timescale as a function of Mstar when joining the red sequence
    # _fig_helper2(xAxis='M_redini', yAxis='dt_green', saveBase='figure12_')

    # figure 13: histogram of delta mstar red restricted to Mstar_z0>limit, inset: dM_red vs Mstar_z0
    _fig_helper("dM_redfr", "log", "figure13_", xMinMax=[0.0, 1.0], plotCSS=["cen"])
    # _fig_helper2(xAxis='mstar', yAxis='dM_redfr', saveBase='figure13_inset_', sizeAdjust=0.55)

    return

    # fully generic exploration (histogram everything, everything vs everything else)
    for fieldName in fieldLabels.keys():
        for yscale in ["linear", "log"]:
            _fig_helper(fieldName, yscale, saveBase="fig_hist_")
        for xAxis in fieldLabels.keys():
            if xAxis == fieldName:
                continue
            _fig_helper2(xAxis=xAxis, yAxis=fieldName)


def colorTracksSchematic(sP, bands, simColorsModel=defSimColorModel, pageNum=None):
    """Schematic diagram: smoothed contours in the color-mass plane and some individual galaxy evolution tracks."""
    # config
    trackIDs = [530874, 495177, 456725]  # blue population occupants
    trackIDs += [377212, 477919, 423657]  # zero dM_red
    trackIDs += [366565, 184828]  # large dM_red, 427314, 376356, 206226
    trackIDs += [531178]  # green valley occupants, 424485
    trackIDs += [146176, 170540]  # rejuv events, 108012, 154493

    dust_Br = "p07c_cf00dust_rad30pkpc"  # include as high time resolution
    dust_C = "p07c_cf00dust_res_conv_ns1_rad30pkpc"  # include as fiducial model
    projInd_C = 0  # viewing angle direction if color model has multiple projections
    minSnap = 33  # evo tracks from z=0 back to this snapshot

    # visual config
    cRedshifts = [0.0, 0.0]  # draw filled contour at first, line contours at subsequent (move manually)
    cColors = ["#dddddd", "#cccccc", "#bbbbbb", "#ffffff"]
    cLevels = [0.4, 0.6, 0.8, 1.1]
    cAlpha = 0.9
    cLevels2 = [0.4, 0.6, 0.8]
    cColors2 = ["#bbbbbb", "#aaaaaa", "#999999"]
    cAlpha2 = 0.7

    tAlpha = 0.9

    xMinMax = [9.0, 12.0]
    mag_range = bandMagRange(bands)

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)
    xlabel = "Galaxy Stellar Mass [ log Msun ]"
    ylabel = "Galaxy (%s-%s) Color" % (bands[0], bands[1])
    xSizeFac = 1.1

    xTicks = [9, 10, 11, 12]
    yTicks = [0.3, 0.6, 0.9]
    zTicks = [2.0, 1.0, 0.5, 0.0]

    # helper: possible red sequence subhalo ID selection, make paged display
    if pageNum is not None:
        nPerPage = 10
        print(" pageNum = %d" % pageNum)

        if "cen_inds" not in sP.data:
            cen_inds = sP.cenSatSubhaloIndices(cenSatSelect="cen")
            sP.data["cen_inds"] = cen_inds
        else:
            cen_inds = sP.data["cen_inds"]

        # subhaloIDs_inds = np.arange( pageNum*nPerPage, (pageNum+1)*nPerPage ) # contiguous N sets
        stride = 340  # e.g. number of total pages, until Mstarz0 ~ 10 (is 3400 centrals)
        subhaloIDs_inds = np.arange(cen_inds.size)[0 : nPerPage * stride][pageNum::stride]
        trackIDs = cen_inds[subhaloIDs_inds]

        print(" inds = ", subhaloIDs_inds)
        print(" ids = ", trackIDs)

    # load/calculate evolution of simulation colors
    gal_colors = {}
    gal_masses = {}
    gal_redshifts = {}

    for dustModel in [dust_C]:  # [dust_C,dust_Br]:
        # load evo and restrict to z<=1
        print(" [%s]" % dustModel)
        colors_evo, shID_evo, _, gal_snaps = calcColorEvoTracks(sP, bands=bands, simColorsModel=dustModel)

        ww = np.where(gal_snaps >= minSnap)

        if len(ww[0]) < gal_snaps.size:
            gal_snaps = gal_snaps[ww]
            if colors_evo.ndim == 2:
                colors_evo = np.squeeze(colors_evo[:, ww])
            if colors_evo.ndim == 3:
                colors_evo = np.squeeze(colors_evo[:, :, ww])
            shID_evo = np.squeeze(shID_evo[:, ww])

        # load evolving stellar masses with individual MPBs, make linear color/mass tracks
        gal_colors[dustModel] = np.zeros((gal_snaps.size, len(trackIDs)), dtype="float32")
        gal_masses[dustModel] = np.zeros_like(gal_colors[dustModel])
        gal_masses[dustModel].fill(np.nan)

        for i, shID in enumerate(trackIDs):
            print("  tracking subhalo [%d]..." % shID)
            assert shID in shID_evo[:, 0]
            evoInd = tuple(shID_evo[:, 0]).index(shID)

            # color track
            if colors_evo.ndim == 2:
                gal_colors[dustModel][:, i] = colors_evo[evoInd, :]
            else:
                gal_colors[dustModel][:, i] = colors_evo[evoInd, projInd_C, :]

            # load MPB
            mpb = sP.loadMPB(shID)
            mpb_mstar = sP.units.codeMassToLogMsun(mpb["SubhaloMassInRadType"][:, sP.ptNum("stars")])

            # pull out masses
            for j, snap in enumerate(gal_snaps):
                if snap not in mpb["SnapNum"]:
                    snap -= 1  # look for adjacent match
                if snap not in mpb["SnapNum"]:
                    snap += 2  # look for adjacent match
                if snap not in mpb["SnapNum"]:
                    continue
                mpbInd = tuple(mpb["SnapNum"]).index(snap)
                if snap == gal_snaps[j]:
                    assert mpb["SubfindID"][mpbInd] == shID_evo[evoInd, j]

                gal_masses[dustModel][j, i] = mpb_mstar[mpbInd]

        gal_redshifts[dustModel] = sP.snapNumToRedshift(snap=gal_snaps)
        gal_redshifts[dustModel] = np.around(gal_redshifts[dustModel], decimals=1)

    # start plot
    fig = plt.figure(figsize=(figsize[0] * 0.9 * xSizeFac, figsize[1] * 0.9), facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)
    setAxisColors(ax, color2)

    if 0:
        # disable top and right axes
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    ax.set_xlim(xMinMax)
    ax.set_ylim(mag_range)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(xTicks)
    ax.set_xticklabels(["%d" % xTick for xTick in xTicks])
    ax.set_yticks(yTicks)

    # contours
    for i, redshift in enumerate(cRedshifts):
        sP.snap = sP.redshiftToSnapNum(redshifts=redshift)

        extent = [xMinMax[0], xMinMax[1], mag_range[0], mag_range[1]]

        xx, yy, kde_sim = calcMstarColor2dKDE(bands, None, None, xMinMax, mag_range, sP=sP, simColorsModel=dust_C)

        for k in range(kde_sim.shape[0]):
            kde_sim[k, :] /= kde_sim[k, :].max()  # by column normalization

        # smooth
        kde_sim = savgol_filter(kde_sim, 31, 3, axis=0)
        kde_sim = savgol_filter(kde_sim, 61, 3, axis=1)

        # filled contour
        if i == 0:
            ax.contourf(xx, yy, kde_sim, cLevels, colors=cColors, alpha=cAlpha, extent=extent)
        else:
            ax.contour(xx, yy, kde_sim, cLevels2, colors=cColors2, alpha=cAlpha2, extent=extent, linestyles="dashed")

    # individual tracks
    for i, shID in enumerate(trackIDs):
        # dust_C
        xx = gal_masses[dust_C][:, i]
        yy = gal_colors[dust_C][:, i]

        (l,) = ax.plot(xx, yy, "o-", alpha=tAlpha, label="ID #%d" % shID)
        for j in range(xx.size):
            if gal_redshifts[dust_C][j] not in zTicks:
                continue
            ax.annotate(
                "%.1f" % gal_redshifts[dust_C][j],
                xy=(xx[j], yy[j]),
                textcoords="offset points",
                xytext=(-5, 5),
                ha="right",
                color=l.get_color(),
                fontsize=8,
                alpha=tAlpha / 2,
            )

        # dust_Br
        if pageNum is None and 0:
            xx = gal_masses[dust_Br][:, i]
            yy = gal_colors[dust_Br][:, i]
            (l,) = ax.plot(xx, yy, "-", alpha=tAlpha / 3, color=l.get_color())

    ax.legend(loc="upper left", prop={"size": 8})

    # finish plot and save
    oStr = "_page-%d" % pageNum if pageNum is not None else ""
    fig.savefig("figure16_schematic_%s_%s%s.pdf" % (sP.simName, "-".join(bands), oStr), facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------


def paperPlots():
    """Construct all the final plots for the paper."""
    L75 = simParams(res=1820, run="tng", redshift=0.0)
    L205 = simParams(res=2500, run="tng", redshift=0.0)
    # L75FP = simParams(res=1820, run="illustris", redshift=0.0)

    dust_A = "p07c_nodust"
    dust_B = "p07c_cf00dust"
    dust_C = "p07c_cf00dust_res_conv_ns1_rad30pkpc"  # one random projection per subhalo
    dust_C_all = "p07c_cf00dust_res_conv_ns1_rad30pkpc_all"  # all projections shown
    dust_D = "p07c_cf00dust_res3_conv_ns1_rad30pkpc"  # geometrical model #3
    # dust_D_all = "p07c_cf00dust_res3_conv_ns1_rad30pkpc"  # geometrical model #3, all projections

    bands = ["g", "r"]

    # figure 1, (g-r) 1D color PDFs in six mstar bins (3x2) Illustris vs TNG100 vs SDSS
    if 0:
        simRedshift = 0.0
        sPs = [L75]  # [L75FP, L75] # order reversed to put TNG100 on top, colors hardcoded
        dust = dust_C_all

        pdf = PdfPages("figure1_%s_%s.pdf" % ("_".join([sP.simName for sP in sPs]), dust))
        galaxyColorPDF(sPs, pdf, bands=bands, simColorsModels=[dust], simRedshift=simRedshift, addPetro=False)
        pdf.close()

    # figure 2, 2x2 grid of different 2D color PDFs, TNG100 vs SDSS
    if 0:
        simRedshift = 0.0
        sPs = [L75]
        dust = dust_C

        pdf = PdfPages("figure2_%s_%s_noK.pdf" % (sPs[0].simName, dust))
        galaxyColor2DPDFs(sPs, pdf, simColorsModel=dust, simRedshift=simRedshift)
        pdf.close()

    # figure 3, stellar ages and metallicities vs mstar (2x1 in a row)
    if 1:
        from temet.plot.driversObs import massMetallicityStars, stellarAges

        sPs = [L75, L205]  # L75FP
        simRedshift = 0.1
        sdssFiberFits = False

        pdf = PdfPages("figure3a_stellarAges_%s.pdf" % "_".join([sP.simName for sP in sPs]))
        stellarAges(sPs, pdf, simRedshift=simRedshift, sdssFiberFits=sdssFiberFits, centralsOnly=True)
        pdf.close()
        pdf = PdfPages("figure3b_massMetallicityStars_%s.pdf" % "_".join([sP.simName for sP in sPs]))
        massMetallicityStars(sPs, pdf, sdssFiberFits=sdssFiberFits, simRedshift=simRedshift)
        pdf.close()

    # figure 4: double gaussian fits, [peak/scatter vs Mstar] 2-panel
    if 0:
        L75.setRedshift(0.0)
        sPs = [L75]
        colorMassPlaneFitSummary(sPs)
        # for sP in sPs: colorMassPlaneFits(sP)

    # figure 5: fullbox demonstratrion projections
    if 0:
        from temet.vis.boxDrivers import TNG_colorFlagshipBoxImage

        # render each fullbox image used in the composite
        for part in [0, 1, 2, 3, 4]:
            TNG_colorFlagshipBoxImage(part=part)

    # figure 6, grid of L205_cen 2d color histos vs. several properties (2x3)
    if 0:
        sP = L205
        yQuant = "color_C_gr"
        params = {"cenSatSelect": "cen", "cStatistic": "median_nan"}

        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="ssfr", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="Z_gas", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="fgas2", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="stellarage", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="bmag_2rhalf_masswt", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="pratio_halo_masswt", **params)

    # figure 7: slice through 2d histo (one property)
    if 0:
        sPs = [L75, L205]
        xQuant = "color_C_gr"
        sQuant = "mstar2_log"
        sRange = [10.4, 10.6]
        css = "cen"
        quant = "pratio_halo_masswt"

        slice(sPs, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, sRange=sRange, cenSatSelect=css)

    # figure 8: BH cumegy vs mstar, model line on top (eddington transition to low-state?)
    if 0:
        sPs = [L75, L205]
        xQuant = "mstar2_log"
        yQuant = "BH_CumEgy_ratioInv"
        css = "cen"

        def _add_theory_line(ax):
            """Special case: yQuant == BH_CumEgy_ratio add theory curve on top from BH model."""
            # NOTE: moved here later, may need additional inputs (sP, cenSatSelect, ...) to work
            # make a second y-axis on the right
            color2 = "#999999"

            ax2 = ax.twinx()
            ax2.set_ylim([8e-5, 0.12])
            ax2.set_yscale("log")

            # ax2.set_ylabel('BH Low State Transition Threshold ($\chi$)', color=color2)
            ax2.set_ylabel("Blackhole Accretion Rate / Eddington Rate", color=color2)
            ax2.tick_params("y", which="both", colors=color2)

            # need median M_BH as a function of x-axis (e.g. M_star)
            for bhIterNum, bhRedshift in enumerate([0.0]):
                # more than 1 redshift
                sP_loc = sP.copy()
                sP_loc.setRedshift(bhRedshift)

                sim_x_loc, _, _, take_log2 = sP_loc.simSubhaloQuantity(xQuant)
                if take_log2:
                    sim_x_loc = logZeroNaN(sim_x_loc)  # match

                # same filters as above
                wSelect = sP_loc.cenSatSubhaloIndices(cenSatSelect=css)
                sim_x_loc = sim_x_loc[wSelect]

                for bhPropNum, bhPropName in enumerate(["BH_mass", "BH_Mdot_edd"]):
                    sim_m_bh, _, _, take_log2 = sP_loc.simSubhaloQuantity(bhPropName)
                    if not take_log2:
                        sim_m_bh = 10.0**sim_m_bh  # undo log then

                    # same filters as above
                    sim_m_bh = sim_m_bh[wSelect]

                    wFinite = np.isfinite(sim_x_loc) & np.isfinite(sim_m_bh)
                    sim_x_loc2 = sim_x_loc[wFinite]
                    sim_m_bh = sim_m_bh[wFinite]

                    xm_bh, ym_bh, _ = running_median(sim_x_loc2, sim_m_bh, binSize=binSize * 2, skipZeros=True)
                    ym_bh = savgol_filter(ym_bh, sKn, sKo)
                    w = np.where(ym_bh > 0.0)  # & (xm_bh > xMinMax[0]) & (xm_bh < xMinMax[1]))
                    xm_bh = xm_bh[w]
                    ym_bh = ym_bh[w]

                    # derive eddington ratio transition as a function of x-axis (e.g. M_star)
                    linestyle = "-" if (bhIterNum == 0 and bhPropNum == 0) else ":"
                    if bhPropName == "BH_mass":
                        ym_bh = sP.units.BH_chi(ym_bh)

                    ax2.plot(xm_bh, ym_bh, linestyle=linestyle, lw=lw, color=color2)

        subhalos.median(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css, f_post=_add_theory_line)

    # figure 9: flux arrows in color-mass plane (9c unused)
    if 0:
        sP = L205

        opts = {"bands": bands, "simColorsModel": dust_C, "cenSatSelect": "cen", "minCount": 1, "toRedshift": 0.3}

        colorFluxArrows2DEvo(sP, arrowMethod="arrow", **opts)

        colorFluxArrows2DEvo(sP, arrowMethod="stream", **opts)

        colorFluxArrows2DEvo(sP, arrowMethod="stream_mass", **opts)

    # figure 10: timescale histogram for color transition
    # figure 11: distribution of initial M* when entering red sequence (crossing color cut) (Q1)
    # figure 12: as a function of M*ini, the Delta_M* from t_{red,ini} to z=0 (Q2)
    # figure 13: as a function of M*(z=0), the t_{red,ini} PDF (Q3)
    if 0:
        sPs = [L75, L205]
        simColorsModel = dust_C  #'p07c_cf00dust_rad30pkpc' # Br
        colorTransitionTimescale(sPs, bands=bands, simColorsModel=simColorsModel)

    # figures 14-15: stellar image stamps of galaxies (red/blue samples)
    if 0:
        from temet.vis.haloDrivers import tngFlagship_galaxyStellarRedBlue

        tngFlagship_galaxyStellarRedBlue(evo=False, redSample=1)
        tngFlagship_galaxyStellarRedBlue(evo=False, blueSample=1)

    # figure 16: schematic / few N characteristic evolutionary tracks through color-mass 2d plane
    if 0:
        colorTracksSchematic(L75, bands=bands)
        # for pageNum in range(0,340):
        #    colorTracksSchematic(L75, bands=bands, pageNum=pageNum)

    # appendix figure 1, viewing angle variation (1 panel)
    if 0:
        viewingAngleVariation()

    # appendix figure 2, dust model dependence (1x3 1D histos in a column)
    if 0:
        sPs = [L75]
        dusts = [dust_D, dust_C_all, dust_C, dust_B, dust_A]
        massBins = ([9.5, 10.0], [10.0, 10.5], [10.5, 11.0])

        pdf = PdfPages("appendix2.pdf")
        galaxyColorPDF(sPs, pdf, bands=bands, simColorsModels=dusts, stellarMassBins=massBins)
        pdf.close()

    # appendix figure 3, resolution convergence (1x3 1D histos in a column)
    if 0:
        L75n910 = simParams(res=910, run="tng", redshift=0.0)
        L75n455 = simParams(res=455, run="tng", redshift=0.0)
        sPs = [L75, L75n910, L75n455]
        dust = dust_C_all
        massBins = ([9.5, 10.0], [10.0, 10.5], [10.5, 11.0])

        pdf = PdfPages("appendix3_%s.pdf" % dust)
        galaxyColorPDF(sPs, pdf, bands=bands, simColorsModels=[dust], stellarMassBins=massBins)
        pdf.close()

    # appendix figure X, 2d density histos (3x1 in a row) all_L75, cen_L75, cen_L205
    if 0:
        subhalos.histogram2d(L75, bands, cenSatSelect="all", cQuant=None)
        subhalos.histogram2d(L75, bands, cenSatSelect="cen", cQuant=None)
        subhalos.histogram2d(L205, bands, cenSatSelect="cen", cQuant=None)

    # supplemental figures:
    # ---------------------
    if 0:
        # 6 other properties, 2d histos
        sP = L205
        yQuant = "color_C_gr"
        params = {"cenSatSelect": "cen", "cStatistic": "median_nan"}

        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="surfdens1_stars", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="Z_stars", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="Krot_oriented_stars2", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="Krot_oriented_gas2", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="xray_r500", **params)
        subhalos.histogram2d(sP, yQuant=yQuant, cQuant="size_stars", **params)

    if 0:
        # slices of other properties, pratio components
        sPs = [L75, L205]
        xQuant = "color_C_gr"
        sQuant = "mstar2_log"
        sRange = [10.4, 10.6]
        css = "cen"
        quants = [
            "ptot_gas_halo",
            "ptot_b_halo",
            "ssfr",
            "fgas2",
            "Z_gas",
            "Z_stars",
            "bmag_2rhalf_masswt",
            "surfdens1_stars",
            "surfdens2_stars",
            "Krot_oriented_stars2",
            "Krot_oriented_gas2",
            "xray_r500",
            "size_stars",
        ]

        for quant in quants:
            slice(sPs, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, sRange=sRange, cenSatSelect=css)

    if 0:
        # star formation main sequence
        sP = L205
        params = {"cenSatSelect": "cen", "cStatistic": "median_nan"}

        subhalos.histogram2d(sP, yQuant="ssfr", cQuant=None, **params)
        subhalos.histogram2d(sP, yQuant="ssfr", cQuant="color_C_gr", **params)
