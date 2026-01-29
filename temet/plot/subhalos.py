"""
Generalized plots based on group catalog objects (i.e. subhalos) of cosmological boxes.
"""

import warnings
from getpass import getuser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, colorConverter
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from ..cosmo.util import subsampleRandomSubhalos
from ..plot.config import figsize, linestyles, sKn, sKo
from ..plot.util import getWhiteBlackColors, loadColorTable, sampleColorTable, setAxisColors, setColorbarColors
from ..util.helper import binned_stat_2d, iterable, kde_2d, logZeroNaN, lowess, running_median, running_median_sub


def addRedshiftAxis(ax, sP, zVals=(0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0)):
    """Add a redshift axis as a second x-axis on top (assuming bottom axis is Age of Universe [Gyr])."""
    zVals = np.array(zVals)
    axTop = ax.twiny()
    tlim = ax.get_xlim()

    axTickVals = sP.units.redshiftToAgeFlat(zVals)
    w = np.where((axTickVals >= tlim[0]) & (axTickVals <= tlim[1]))

    axTop.set_xlim(ax.get_xlim())
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals[w])
    axTop.set_xticklabels(zVals[w])
    axTop.set_xlabel("Redshift")


def addUniverseAgeAxis(ax, sP, ageVals=(0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0)):
    """Add a age of the universe [Gyr] axis as a second x-axis on top (assuming bottom is redshift)."""
    axTop = ax.twiny()
    zlim = ax.get_xlim()

    ageVals.append(sP.units.redshiftToAgeFlat([0.0]).round(2))
    ageVals = np.array(ageVals)
    axTickVals = sP.units.ageFlatToRedshift(ageVals)

    if zlim[0] < zlim[1]:
        w = np.where((axTickVals >= zlim[0]) & (axTickVals <= zlim[1]))
    else:
        w = np.where((axTickVals <= zlim[0]) & (axTickVals >= zlim[1]))

    axTop.set_xlim(zlim)
    axTop.set_xscale(ax.get_xscale())
    axTop.set_xticks(axTickVals[w])
    axTop.set_xticklabels(ageVals[w])
    axTop.set_xlabel("Age of the Universe [Gyr]", labelpad=8.0)


def addRedshiftAgeAxes(ax, sP, xrange=(-1e-4, 8.0), xlog=True):
    """Add bottom vs. redshift (and top vs. universe age) axis for standard X vs. redshift plots."""
    ax.set_xlim(xrange)
    ax.set_xlabel("Redshift")

    if xlog:
        ax.set_xscale("symlog")
        zVals = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]  # [10]
    else:
        ax.set_xscale("linear")
        zVals = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    ax.set_xticks(zVals)
    ax.set_xticklabels(zVals)

    addUniverseAgeAxis(ax, sP)


def histogram2d(
    sP,
    yQuant,
    xQuant="mstar2_log",
    cenSatSelect="cen",
    cQuant=None,
    xlim=None,
    ylim=None,
    clim=None,
    cStatistic=None,
    cNaNZeroToMin=False,
    minCount=None,
    cRel=None,
    cFrac=None,
    nBins=None,
    qRestrictions=None,
    filterFlag=False,
    medianLine=True,
    sizeFac=1.0,
    pStyle="white",
    ctName=None,
    saveFilename=None,
    output_fmt=None,
    pdf=None,
):
    """Make a 2D histogram comparing two subhalo properties, optionally coloring by a third.

    minCount specifies the minimum number of
    points a bin must contain to show it as non-white. If '_nan' is not in cStatistic, then by default,
    empty bins are white, and bins whose cStatistic is NaN (e.g. any NaNs in bin) are gray. Or, if
    '_nan' is in cStatistic, then empty bins remain white, while the cStatistic for bins with any
    non-NaN values is computed ignoring NaNs (e.g. np.nanmean() instead of np.mean()), and bins
    which are non-empty but contain only NaN values are gray. If cRel is not None, then should be a
    3-tuple of [relMin,relMax,takeLog] in which case the colors are not of the physical cQuant itself,
    but rather the value of that quantity relative to the median at that value of the x-axis (e.g. mass).
    If cFrac is not None, then a 4-tuple of [fracMin,fracMax,takeLog,label] specifying a criterion on the values
    of cQuant such that the colors are not of the physical cQuant itself, but rather represent the fraction of
    subhalos in each pixel satisfying (fracMin <= cQuant < fracMax), where +/-np.inf is allowed for one-sided,
    takeLog should be True or False, and label is either a string or None for automatic.
    If qRestrictions, then a list containing 3-tuples, each of [fieldName,min,max], to restrict all points by.
    If filterFlag, exclude SubhaloFlag==0 (non-cosmological) objects.
    If xlim, ylim, or clim are not None, then override the respective axes ranges with these [min,max] bounds.
    If cNanZeroToMin, then change the color of the NaN-only bins from the usual gray to the colormap minimum.
    """
    assert cenSatSelect in ["all", "cen", "sat"]
    assert cStatistic in [None, "mean", "median", "count", "sum", "mean_nan", "median_nan"]  # or any user function
    assert np.sum([cRel is not None, cFrac is not None]) <= 1  # at most one is not None

    # hard-coded config
    if nBins is None:
        nBins = 80

    cmap = loadColorTable(ctName if ctName is not None else "viridis")  # , numColors=13
    colorMed = "black"

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    colorContours = False
    if cQuant is None:
        colorMed = "orange"

    # x-axis: load fullbox galaxy properties and set plot options, cached in sP.data
    sim_xvals, xlabel, xMinMax, xLog = sP.simSubhaloQuantity(xQuant)
    if xMinMax[0] > xMinMax[1]:
        xMinMax = xMinMax[::-1]  # reverse
    if xLog is True:
        sim_xvals = logZeroNaN(sim_xvals)
    if xlim is not None:
        xMinMax = xlim

    # y-axis: load/calculate simulation colors, cached in sP.data
    sim_yvals, ylabel, yMinMax, yLog = sP.simSubhaloQuantity(yQuant)
    if yLog is True:
        sim_yvals = logZeroNaN(sim_yvals)
    if ylim is not None:
        yMinMax = ylim

    # c-axis: load properties for color mappings
    if cQuant is None:
        sim_cvals = np.zeros(sim_xvals.size, dtype="float32")

        # overrides for density distribution
        cStatistic = "count"
        if ctName is None:
            ctName = "gray_r" if pStyle == "white" else "gray"
        cmap = loadColorTable(ctName)

        clabel = "log N$_{\\rm gal}$+1"
        cMinMax = [0.0, 2.0] if clim is None else clim
        if sP.boxSize > 100000:
            cMinMax = [0.0, 2.5]
    else:
        if cStatistic is None:
            cStatistic = "median_nan"  # default if not specified with cQuant
        sim_cvals, clabel, cMinMax, cLog = sP.simSubhaloQuantity(cQuant)
        if clim is not None:
            cMinMax = clim

    if sim_cvals is None:
        return  # property is not calculated for this run (e.g. expensive auxCat)

    # flagging?
    sim_flag = np.ones(sim_xvals.shape).astype("bool")
    if filterFlag and sP.groupCatHasField("Subhalo", "SubhaloFlag"):
        # load SubhaloFlag and override sim_flag (0=bad, 1=good)
        sim_flag = sP.groupCat(fieldsSubhalos=["SubhaloFlag"])

    # central/satellite selection?
    wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

    sim_xvals = sim_xvals[wSelect]
    sim_cvals = sim_cvals[wSelect]
    sim_yvals = sim_yvals[wSelect]
    sim_flag = sim_flag[wSelect]

    # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
    wFinite = np.isfinite(sim_yvals)  # & np.isfinite(sim_xvals)

    # reduce to the good-flagged subset
    wFinite &= sim_flag

    sim_yvals = sim_yvals[wFinite]
    sim_cvals = sim_cvals[wFinite]
    sim_xvals = sim_xvals[wFinite]

    # arbitrary property restriction(s)?
    if qRestrictions is not None:
        assert (
            len(qRestrictions) == 1
        )  # otherwise check, does it really make sense? second+ vals hasn't had wRestrict applied...
        for rFieldName, rFieldMin, rFieldMax in qRestrictions:
            # load and restrict
            vals, _, _, _ = sP.simSubhaloQuantity(rFieldName)
            vals = vals[wSelect][wFinite]

            wRestrict = np.where((vals >= rFieldMin) & (vals < rFieldMax))

            sim_yvals = sim_yvals[wRestrict]
            sim_xvals = sim_xvals[wRestrict]
            sim_cvals = sim_cvals[wRestrict]

    # _nan cStatistic? separate points into two sets
    nanFlag = False
    if "_nan" in cStatistic:
        nanFlag = True

        wFiniteCval = np.isfinite(sim_cvals)
        wNaNCval = np.isnan(sim_cvals)
        wInfCval = np.isinf(sim_cvals)

        if np.count_nonzero(wInfCval) > 0:  # unusual
            print(" warning: [%d] infinite color values [%s]." % (np.count_nonzero(wInfCval), cQuant))

        assert np.count_nonzero(wFiniteCval) + np.count_nonzero(wNaNCval) + np.count_nonzero(wInfCval) == sim_cvals.size

        # save points with NaN cvals
        sim_yvals_nan = sim_yvals[wNaNCval]
        sim_cvals_nan = sim_cvals[wNaNCval]
        sim_xvals_nan = sim_xvals[wNaNCval]

        # override default binning to only points with finite cvals
        sim_yvals = sim_yvals[wFiniteCval]
        sim_cvals = sim_cvals[wFiniteCval]
        sim_xvals = sim_xvals[wFiniteCval]

        # replace cStatistic string
        cStatistic = cStatistic.split("_nan")[0]

    # start plot
    fig = plt.figure(figsize=(figsize[0] * sizeFac, figsize[1] * sizeFac), facecolor=color1)
    ax = fig.add_subplot(111, facecolor=color1)

    setAxisColors(ax, color2)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if getuser() != "wwwrun":
        print(" ", xQuant, yQuant, cQuant, sP.simName, cenSatSelect)

    cssStrings = {"all": "all galaxies", "cen": "centrals only", "sat": "satellites"}
    if getuser() == "wwwrun":
        ax.set_title(sP.simName + ": " + cssStrings[cenSatSelect])

    # 2d histogram
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBins, int(nBins * (bbox.height / bbox.width))])
    extent = [xMinMax[0], xMinMax[1], yMinMax[0], yMinMax[1]]

    # statistic reduction (e.g. median, sum, count) color by bin
    if cStatistic == "median" and nanFlag:
        # custom version
        cc, nn = binned_stat_2d(
            sim_xvals, sim_yvals, sim_cvals, bins=nBins2D, range_x=xMinMax, range_y=yMinMax, stat="median"
        )
    else:
        cc, xBins, yBins, inds = binned_statistic_2d(
            sim_xvals, sim_yvals, sim_cvals, cStatistic, bins=nBins2D, range=[xMinMax, yMinMax]
        )
        nn, _, _, _ = binned_statistic_2d(
            sim_xvals, sim_yvals, sim_cvals, "count", bins=nBins2D, range=[xMinMax, yMinMax]
        )

    # imshow convention
    cc = cc.T
    nn = nn.T

    # relative coloring as a function of the x-axis?
    if cRel is not None:
        # override min,max of color and whether or not to log
        cMinMax[0], cMinMax[1], cLog = cRel

        # normalize each column by median, ignore empty pixels
        if cStatistic == "count":
            w = np.where(cc == 0.0)
            cc[w] = np.nan

        # convert warnings into error causing pdb break
        # with warnings.catch_warnings():
        #    warnings.filterwarnings('error')
        #    medVals = np.nanmedian(cc, axis=0)

        with warnings.catch_warnings():
            # if any column contains no valid values (poor statistics), throws warning messages
            warnings.filterwarnings("ignore")
            medVals = np.nanmedian(cc, axis=0)

        with np.errstate(invalid="ignore"):
            cc /= medVals[np.newaxis, :]

        cmap = loadColorTable("coolwarm")  # diverging
        clabel = "Relative " + clabel.split("[")[0] + ("[ log ]" if cLog else "")

    # color based on fraction of systems in a pixel which satisfy some criterion?
    if cFrac is not None:
        # override min,max of color and whether or not to log
        fracMin, fracMax, cLog, fracLabel = cFrac
        if clim is None:
            cMinMax = [0.0, 1.0] if not cLog else [-1.5, 0.0]

        # select sim values which satisfy criterion, and re-count
        w = np.where((sim_cvals >= fracMin) & (sim_cvals < fracMax))

        nn_sat, _, _, _ = binned_statistic_2d(
            sim_xvals[w], sim_yvals[w], sim_cvals[w], "count", bins=nBins2D, range=[xMinMax, yMinMax]
        )
        nn_sat = nn_sat.T

        with np.errstate(invalid="ignore"):
            # set each pixel value to the fraction (= N_sat / N_tot)
            cc = nn_sat / nn

            # set absolute zeros to a small finite value, to avoid special (gray) coloring
            w = np.where(cc == 0.0)
        cc[w] = 1e-10

        # modify colortable and label
        cmap = loadColorTable("matter_r")  # haline, thermal, solar, deep_r, dense_r, speed_r, amp_r, matter_r

        qStr = clabel.split("[")[0]  # everything to the left of the units
        # if '$' in clabel:
        #    qStr = '$%s$' % clabel.split('$')[1] # just the label symbol, if one is present
        qUnitStr = ""
        if "[" in clabel:
            qUnitStr = " " + clabel.split("[")[1].split("]")[0].strip()
            if qUnitStr == " log":
                qUnitStr = ""

        # 1 digit after the decimal point if the bounds numbers are not roundable to ints, else just integers
        qDigits = (
            1
            if (
                (np.isfinite(fracMin) & ~float(fracMin).is_integer())
                | (np.isfinite(fracMax) & ~float(fracMax).is_integer())
            )
            else 0
        )

        clabel = "Fraction ("
        if np.isinf(fracMin):
            clabel += "%s < %.*f%s)" % (qStr, qDigits, fracMax, qUnitStr)
        elif np.isinf(fracMax):
            clabel += "%s > %.*f%s)" % (qStr, qDigits, fracMin, qUnitStr)
        else:
            clabel += "%.*f < %s [%s] < %.*f)" % (qDigits, fracMin, qStr, qUnitStr, qDigits, fracMax)
        clabel += " [ log ]" if cLog else ""

        # manually specified label?
        if fracLabel is not None:
            clabel = fracLabel

    # for now: log on density and all color quantities
    cc2d = cc
    if cQuant is None:
        cc2d += 1.0  # add 1 to count
    if cQuant is None or cLog is True:
        cc2d = logZeroNaN(cc)

    # normalize and color map
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap(norm(cc2d))

    # mask bins with median==0 and map to special color, which right now have been set to log10(0)=NaN
    color3 = colorConverter.to_rgba(color3)
    color4 = colorConverter.to_rgba(color4)

    if cNaNZeroToMin:
        color3 = cmap(0.0)
        color4 = cmap(0.0)

    if cQuant is not None:
        cc2d_rgb[cc == 0.0, :] = color4

    if nanFlag:
        # bin NaN point set counts
        nn_nan, _, _, _ = binned_statistic_2d(
            sim_xvals_nan, sim_yvals_nan, sim_cvals_nan, "count", bins=nBins2D, range=[xMinMax, yMinMax]
        )
        nn_nan = nn_nan.T

        # flag bins with nn_nan>0 and nn==0 (only NaNs in bin) as second gray color
        cc2d_rgb[((nn_nan > 0) & (nn == 0)), :] = color3

        nn += nn_nan  # accumulate total counts
    else:
        # mask bins with median==NaN (nonzero number of NaNs in bin) to gray
        cc2d_rgb[~np.isfinite(cc), :] = color3

    # mask empty bins to white
    cc2d_rgb[(nn == 0), :] = colorConverter.to_rgba(color1)

    if minCount is not None:
        cc2d_rgb[nn < minCount] = colorConverter.to_rgba(color1)

    # plot
    im = plt.imshow(
        cc2d_rgb, extent=extent, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap, norm=norm
    )

    # method (B) unused
    # reduceMap = {'mean':np.mean, 'median':np.median, 'count':np.size, 'sum':np.sum}
    # reduceFunc = reduceMap[cStatistic] if cStatistic in reduceMap else cStatistic
    # plt.hexbin(sim_xvals, sim_yvals, C=None, gridsize=nBins, extent=extent, bins='log',
    #          mincnt=minCount, cmap=cmap, marginals=False)
    # plt.hexbin(sim_xvals, sim_yvals, C=sim_cvals, gridsize=nBins, extent=extent, bins='log',
    #          mincnt=minCount, cmap=cmap, marginals=False, reduce_C_function=reduceFunc)

    # median line?
    if np.count_nonzero(np.isnan(sim_xvals)) == sim_xvals.size:
        warnStr = (
            "Warning! All x-axis values are NaN, so nothing to plot (for example, mhalo_200 is NaN for satellites)."
        )
        ax.text(
            np.mean(ax.get_xlim()),
            np.mean(ax.get_ylim()),
            warnStr,
            ha="center",
            va="center",
            color="black",
            fontsize=11,
        )
        medianLine = False  # all x-axis values are nan (i.e. mhalo_200 for cenSatSelect=='sat')

    if medianLine:
        binSizeMed = (xMinMax[1] - xMinMax[0]) / nBins * 2

        xm, ym, sm, pm = running_median(sim_xvals, sim_yvals, binSize=binSizeMed, percs=[5, 10, 25, 75, 90, 95])
        if xm.size > sKn:
            ym = savgol_filter(ym, sKn, sKo)
            sm = savgol_filter(sm, sKn, sKo)
            pm = savgol_filter(pm, sKn, sKo, axis=1)

        ax.plot(xm[:-1], ym[:-1], "-", color=colorMed, label="median")

        ax.plot(xm[:-1], pm[1, :-1], ":", color=colorMed, label="P[10,90]")
        ax.plot(xm[:-1], pm[-2, :-1], ":", color=colorMed)

    # contours?
    if colorContours:
        extent = [xMinMax[0], xMinMax[1], yMinMax[0], yMinMax[1]]
        cLevels = [0.75, 0.95]
        cAlphas = [0.5, 0.8]

        # run 2d kernel density estimate
        xx, yy, kde_sim = kde_2d(sim_xvals, sim_yvals, xMinMax, yMinMax)

        for k in range(kde_sim.shape[0]):
            kde_sim[k, :] /= kde_sim[k, :].max()  # by column normalization

        for k, cLevel in enumerate(cLevels):
            ax.contour(xx, yy, kde_sim, [cLevel], colors=[color2], alpha=cAlphas[k], extent=extent)

    # special behaviors
    if yQuant == "size_gas":
        # add virial radius median line
        aux_yvals, _, _, _ = sP.simSubhaloQuantity("rhalo_200_log")
        aux_yvals = aux_yvals[wSelect][wFinite]
        if nanFlag:
            aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals, aux_yvals, binSize=binSizeMed, percs=[5, 10, 25, 75, 90, 95])
        if xm.size > sKn:
            ym = savgol_filter(ym, sKn, sKo)

        color = sampleColorTable("tableau10", "purple")
        ax.plot(xm[:-1], ym[:-1], "--", color=color, label=r"Halo $R_{\rm 200,crit}$")
        ax.legend(loc="upper left")

    if yQuant in ["temp_halo", "temp_halo_volwt"]:
        # add virial temperature median line
        aux_yvals = sP.groupCat(fieldsSubhalos=["tvir_log"])
        aux_yvals = aux_yvals[wSelect][wFinite]
        if nanFlag:
            aux_yvals = aux_yvals[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals, aux_yvals, binSize=binSizeMed, percs=[5, 10, 25, 75, 90, 95])
        if xm.size > sKn:
            ym = savgol_filter(ym, sKn, sKo)

        color = sampleColorTable("tableau10", "purple")
        ax.plot(xm[:-1], ym[:-1], "--", color=color, label=r"Halo $T_{\rm vir}$")
        ax.legend(loc="upper left")

    if yQuant == "fgas_r200":
        # add constant f_b line
        f_b = np.log10(sP.units.f_b)

        color = sampleColorTable("tableau10", "purple")
        ax.plot(xMinMax, [f_b, f_b], "--", color=color)
        ax.text(np.mean(ax.get_xlim()), f_b + 0.05, r"$\Omega_{\rm b} / \Omega_{\rm m}$", color=color, size=17)

    if yQuant in ["BH_CumEgy_low", "BH_CumEgy_high"]:
        # add approximate halo binding energy line = (3/5)*GM^2/R
        G = sP.units.G / 1e10  # kpc (km/s)**2 / msun
        r_halo, _, _, _ = sP.simSubhaloQuantity("rhalo_200")  # pkpc
        m_halo, _, _, _ = sP.simSubhaloQuantity("mhalo_200")  # msun
        e_b = (3.0 / 5.0) * G * m_halo**2 * sP.units.f_b / r_halo  # (km/s)**2 * msun
        e_b = np.array(e_b, dtype="float64") * 1e10 * sP.units.Msun_in_g  # cm^2/s^2 * g
        e_b = logZeroNaN(e_b).astype("float32")  # log(cm^2/s^2 * g)
        e_b = e_b[wSelect][wFinite]
        if nanFlag:
            e_b = e_b[wFiniteCval]

        xm, ym, _, _ = running_median(sim_xvals, e_b, binSize=binSizeMed, percs=[5, 10, 25, 75, 90, 95])
        if xm.size > sKn:
            ym = savgol_filter(ym, sKn, sKo)

        color = sampleColorTable("tableau10", "purple")
        ax.plot(xm[:-1], ym[:-1], "--", color=color, label=r"Halo $E_{\rm B}$")
        ax.legend(loc="upper left")

    if xQuant in ["color_nodust_VJ", "color_C-30kpc-z_VJ"] and yQuant in ["color_nodust_UV", "color_C-30kpc-z_UV"]:
        # UVJ color-color diagram, add Tomczak+2014 separation of passive and SFing galaxies
        xx = [0.0, 0.7, 1.4, 1.4]
        yy = [1.4, 1.4, 2.0, 2.45]
        ax.plot(xx, yy, ":", color="red", label="Tomczak+14")

        # Muzzin+2013b separation line (Equations 1-3)
        if sP.redshift <= 1.0:
            off = 0.69
        if sP.redshift > 1.0:
            off = 0.59
        xx = [0.0, (1.3 - off) / 0.88, 1.5, 1.5]
        yy = [1.3, 1.3, 1.5 * 0.88 + off, 2.45]
        ax.plot(xx, yy, ":", color="orange", label="Muzzin+13b")
        ax.legend(loc="upper left")

    # colorbar
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.ax.set_ylabel(clabel)
    if len(clabel) > 45:
        newsize = 27 - (len(clabel) - 45) / 5
        cb.ax.set_ylabel(clabel, size=newsize)  # default: 24.192 (14 * x-large)
    setColorbarColors(cb, color2)

    # finish plot and save
    if pdf is not None:
        pdf.savefig(facecolor=fig.get_facecolor())
    else:
        # note: saveFilename could be an in-memory buffer
        if saveFilename is None:
            saveFilename = "histo2d_%s_%s_%s_%s_%s_%s.pdf" % (
                yQuant,
                xQuant,
                cQuant,
                cStatistic,
                cenSatSelect,
                minCount,
            )
        fig.savefig(saveFilename, format=output_fmt, facecolor=fig.get_facecolor())
    plt.close(fig)

    return True


def slice(
    sPs,
    xQuant,
    yQuants,
    sQuant,
    sRange,
    cenSatSelect="cen",
    yRel=None,
    xlim=None,
    ylim=None,
    filterFlag=False,
    sizefac=None,
    saveFilename=None,
    pdf=None,
):
    """Make a 1D 'slice' showing the dependence of one quantity on another, for subhalos selected by a range in a third.

    This can be thought of as a slice through the 2D histogram by restricting to some range sRange of some quantity
    sQuant which is typically Mstar (e.g. 10.4<log_Mstar<10.6 to slice in the middle of the bimodality).

    Supports multiple sPs which are overplotted.
    For all subhalos in this slice, optically restricted by cenSatSelect, load a set of quantities
    yQuants (could be just one) and plot this (y-axis) against xQuant, with any additional configuration
    provided by xQuantSpec. Multiple yQuants results in a grid.
    If xlim or ylim are not None, then override the respective axes ranges with these [min,max] bounds.
    If sRange is a list of lists, then overplot multiple different slice ranges.
    If yRel is not None, then should be a 3-tuple of [relMin,relMax,takeLog] or 4-tuple of
    [relMin,relMax,takeLog,yLabel] in which case the y-axis is not of the physical yQuants themselves,
    but rather the value of the quantity relative to the median in the slice (e.g. mass).
    If filterFlag, exclude SubhaloFlag==0 (non-cosmological) objects.
    """
    assert cenSatSelect in ["all", "cen", "sat"]

    if len(yQuants) == 0:
        return
    nRows = int(np.floor(np.sqrt(len(yQuants))))
    nCols = int(np.ceil(len(yQuants) / nRows))

    # just a single sRange? wrap in an outer list
    sRanges = sRange
    if not isinstance(sRange[0], list):
        sRanges = [sRange]

    # hard-coded config
    ptPlotThresh = 2000

    if sizefac is None:
        sizefac = 0.8 if nCols > 4 else 1.0  # enlarge text for big panel grids

    # start plot
    fig = plt.figure(figsize=[figsize[0] * nCols * sizefac, figsize[1] * nRows * sizefac])

    # loop over each yQuantity (panel)
    for i, yQuant in enumerate(yQuants):
        ax = fig.add_subplot(nRows, nCols, i + 1)

        for sP in sPs:
            # loop over each run and add to the same plot
            print(" ", yQuant, sP.simName, xQuant, cenSatSelect, sQuant, sRange)

            # y-axis: load galaxy properties (in histo2D were the color mappings)
            sim_yvals, ylabel, yMinMax, yLog = sP.simSubhaloQuantity(yQuant)
            if ylim is not None:
                yMinMax = ylim

            if sim_yvals is None:
                print("   skip")
                continue  # property is not calculated for this run (e.g. expensive auxCat)
            if yLog is True:
                sim_yvals = logZeroNaN(sim_yvals)

            # slice values: load fullbox galaxy property to slice on (e.g. Mstar or Mhalo)
            sim_svals, slabel, _, _ = sP.simSubhaloQuantity(sQuant)

            if sim_svals is None:
                print("   skip")
                continue

            # x-axis: load/calculate x-axis quantity (e.g. simulation colors), cached in sP.data
            sim_xvals, xlabel, xMinMax, xLog = sP.simSubhaloQuantity(xQuant)
            if xlim is not None:
                xMinMax = xlim

            if sim_xvals is None:
                print("   skip")
                continue
            if xLog is True:
                sim_xvals = logZeroNaN(sim_xvals)

            # relative coloring relative to the median in the slice?
            if yRel is not None:
                # override min,max of y-axis and whether or not to log
                assert yLog is False  # otherwise handle in a general way (and maybe undo?)
                if len(yRel) == 3:
                    yMinMax[0], yMinMax[1], yLog = yRel
                    ylabel = r"$\Delta$ " + ylabel.split("[")[0] + ("[ log ]" if yLog else "")
                if len(yRel) == 4:
                    yMinMax[0], yMinMax[1], yLog, ylabel = yRel

            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # flagging?
            sim_flag = np.ones(sim_xvals.shape).astype("bool")
            if filterFlag and sP.groupCatHasField("Subhalo", "SubhaloFlag"):
                # load SubhaloFlag and override sim_flag (0=bad, 1=good)
                sim_flag = sP.groupCat(fieldsSubhalos=["SubhaloFlag"])

            # central/satellite selection?
            wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

            sim_xvals = sim_xvals[wSelect]
            sim_yvals = sim_yvals[wSelect]
            sim_svals = sim_svals[wSelect]
            sim_flag = sim_flag[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.isfinite(sim_xvals) & np.isfinite(sim_yvals)

            # reduce to the good-flagged subset
            wFinite &= sim_flag

            sim_xvals = sim_xvals[wFinite]
            sim_yvals = sim_yvals[wFinite]
            sim_svals = sim_svals[wFinite]

            # loop over slice ranges
            for sRange in sRanges:
                # make slice selection
                wSlice = np.where((sim_svals >= sRange[0]) & (sim_svals < sRange[1]))
                xx = sim_xvals[wSlice]
                yy = sim_yvals[wSlice]

                # relative coloring relative to the median in the slice?
                if yRel is not None:
                    yy /= np.nanmedian(yy)

                # median and 10/90th percentile lines
                nBins = 30
                if xx.size >= ptPlotThresh:
                    nBins *= 2

                binSize = (xMinMax[1] - xMinMax[0]) / nBins

                xm, ym, sm, pm = running_median(xx, yy, binSize=binSize, percs=[5, 10, 25, 75, 90, 95])
                if xm.size > sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)

                sName = slabel.split("[")[0].rstrip()  # shortened version (remove units) of split quant name for legend
                label = sP.simName if len(sRanges) == 1 else "%.1f < %s < %.1f" % (sRange[0], sName, sRange[1])
                (l,) = ax.plot(xm, ym, linestyles[0], label=label)

                # percentile band:
                if xx.size >= ptPlotThresh:
                    ax.fill_between(xm, pm[1, :], pm[-2, :], facecolor=l.get_color(), alpha=0.1, interpolate=True)

                # plot points
                if xx.size < ptPlotThresh:
                    ax.plot(xx, yy, "o", color=l.get_color(), alpha=0.3)

        # legend
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            handlesO = []
            labelsO = []

            ax.legend(handles + handlesO, labels + labelsO, loc="best")

    # finish plot and save
    if pdf is not None:
        pdf.savefig()
    else:
        if saveFilename is None:
            saveFilename = "slice1d_%s_%s_%s_%s.pdf" % ("-".join(yQuants), xQuant, sQuant, cenSatSelect)
        fig.savefig(saveFilename)
    plt.close(fig)


def median(
    sPs,
    yQuants,
    xQuant,
    cenSatSelect="cen",
    sQuant=None,
    sLowerPercs=None,
    sUpperPercs=None,
    sizefac=1.0,
    alpha=1.0,
    nBins=50,
    qRestrictions=None,
    indivRestrictions=False,
    f_pre=None,
    f_post=None,
    xlabel=None,
    ylabel=None,
    lowessSmooth=False,
    scatterPoints=None,
    markersize=6.0,
    maxPointsPerDex=None,
    scatterColor=None,
    ctName=None,
    markSubhaloIDs=None,
    cRel=None,
    mark1to1=False,
    drawMedian=True,
    medianLabel=None,
    extraMedians=None,
    legendLoc="best",
    labelSims=True,
    xlim=None,
    ylim=None,
    clim=None,
    cbarticks=None,
    filterFlag=False,
    colorbarInside=False,
    saveFilename=None,
    pdf=None,
):
    """Plot the running median (optionally with scatter points) of some quantity vs another for all subhalos.

    Args:
      sPs (:py:class:`~util.simParams` or list): simulation instance(s).
      yQuants (str or list[str]): names of quantities (could be just one) to plot on the y-axis.
        Multiple yQuants results in a grid of panels.
      xQuant (str): name of quantity to plot on the x-axis.
      cenSatSelect (str): restrict subhalo sample to one of 'cen', 'sat', or 'all'.
      sQuant (str or None): if not None, then in addition to the median, load this third quantity and
        split the subhalos on it according to (mandatory) ``sLowerPercs`` and ``sUpperPercs``. Each
        such split adds another median line derived from that sub-sample alone.
      sLowerPercs (tuple[float][2]): a list of percentiles (e.g. ``[16,84]``), to split the sample on.
      sUpperPercs (tuple[float][2]): a list of percentiles (e.g. ``[16,84]``), to split the sample on.
      sizefac (float): overrides the default plot sizefac.
      alpha (float): controls only the scattered points (if plotted).
      nBins (int): number of bins along the x-axis quantity for computing the median lines.
      qRestrictions (list): one or more 3-tuples, each containing ``[fieldName,min,max]``, which are then
        used to restrict all points by.
      indivRestrictions (bool): if True, then each item in ``qRestrictions`` is applied independently, and
        added to the plot, otherwise all are applied simultaneously and only one sample is shown.
      f_pre (function): if not None, this 'custom' function hook is called just before plotting.
        It must accept the figure axis as its single argument.
      f_post (function): if not None, this 'custom' function hook is called just after plotting.
        It must accept the figure axis as its single argument.
      xlabel (str): if not None, override x-axis label.
      ylabel (str): if not None, override y-axis label.
      lowessSmooth (bool): smooth the resulting color distribution (slow for large number of points).
      scatterPoints (bool): include all raw points with a scatterplot.
      markersize (float): if ``scatterPoints`` then override the default marker size (of 6).
      maxPointsPerDex (int): if not None, then randomly sub-sample down to at most this number (equal
        number per 0.1 dex bin) as a maximum, to reduce confusion at the low-mass end.
      scatterColor (str): color each point by a third property.
      ctName (str or list): if not None, then specify a different colormap name to use for the points or medians.
        If a list, then the first entry should be the string name, while the second should be a bounds 2-tuple.
      markSubhaloIDs (bool): highlight these subhalos especially on the plot.
      cRel: if not None, then should be a 3-tuple of ``[relMin,relMax,takeLog]`` in which case the colors
        are not scatterColor itself, but the value of that quantity relative to the median at that value
        of the x-axis (e.g. mass).
      mark1to1 (bool): show a 1-to-1 line (i.e. assuming x and y axes could be closely related).
      drawMedian (bool): include median line and 1-sigma band.
      medianLabel (str): if not None, then override the median label with this string.
      extraMedians (list[str]): if not None, add more median lines for these (y-axis) quantities as well.
      legendLoc (str): override 'best' default location. None to disable legend.
      labelSims (bool): if True, then label each simulation (of sPs) in the legend.
      xlim (list[float][2]): if not None, override default x-axis limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      clim (list[float][2]): if not None, override default colorbar limits.
      cbarticks (list[float]): if not None, override automatic colorbar tick values.
      filterFlag (bool): if True, exclude SubhaloFlag==0 (non-cosmological) objects.
      colorbarInside (bool): place colorbar (assuming scatterColor is used) inside the panel.
      saveFilename (str): name (and extension, setting format) of output plot. Automatic if None.
      pdf (PdfPages or None): if None, an actual PDF file is written to disk with the figure.
        If not None, then the figure is added to this existing pdf collection.

    Returns:
      None. PDF figure is saved in current directory, or added to ``pdf`` if input.
    """
    assert cenSatSelect in ["all", "cen", "sat"]
    if extraMedians is None:
        extraMedians = []  # avoid mutable keyword argument
    if sQuant is not None:
        assert sLowerPercs is not None and sUpperPercs is not None
    if scatterColor is not None:
        scatterPoints = True
    if lowessSmooth:
        assert scatterPoints and scatterColor is not None, "Only LOWESS smooth scattered points."

    yQuants = iterable(yQuants)
    sPs = iterable(sPs)

    nRows = int(np.floor(np.sqrt(len(yQuants))))
    nCols = int(np.ceil(len(yQuants) / nRows))

    # hard-coded config
    ptPlotThresh = 2000
    if nCols > 4 and sizefac == 1.0:
        sizefac = 0.8

    # start plot
    fig = plt.figure(figsize=[figsize[0] * nCols * sizefac, figsize[1] * nRows * sizefac])

    # loop over each yQuantity (panel)
    for i, yQuant in enumerate(yQuants):
        ax = fig.add_subplot(nRows, nCols, i + 1)

        if f_pre is not None:
            f_pre(ax)

        for _j, sP in enumerate(sPs):
            # loop over each run and add to the same plot
            print(" ", yQuant, xQuant, sP.simName, cenSatSelect)

            # y-axis: load fullbox galaxy properties
            sim_yvals, ylabel_def, yMinMax, yLog = sP.simSubhaloQuantity(yQuant)
            if ylim is not None:
                yMinMax = ylim
            if ylabel is None or i > 0:
                ylabel = ylabel_def

            if sim_yvals is None:
                print("   skip")
                continue  # property is not calculated for this run (e.g. expensive auxCat)
            if yLog:
                # check for zero values (should generalize)
                w_zero = np.where(sim_yvals == 0)[0]
                if len(w_zero) and yQuant in ["sfr_30pkpc", "sfr", "sfr2"]:
                    print("Warning: setting [%d] zero SFRs to random around minimum y-axis value!" % len(w_zero))
                    rng = np.random.default_rng(424242)
                    low_val = ylim[0] + 1 * (ylim[1] - ylim[0]) / 100  # 1% above the bottom
                    high_val = ylim[0] + 15 * (ylim[1] - ylim[0]) / 100  # to 8% above the bottom
                    sim_yvals[w_zero] = 10.0 ** (rng.uniform(low=low_val, high=high_val, size=len(w_zero)))

                sim_yvals = logZeroNaN(sim_yvals)

            # x-axis: load fullbox galaxy properties
            sim_xvals, xlabel_def, xMinMax, xLog = sP.simSubhaloQuantity(xQuant)
            if xLog:
                sim_xvals = logZeroNaN(sim_xvals)
            if xlim is not None:
                xMinMax = xlim
            if xlabel is None:
                xlabel = xlabel_def

            ax.set_xlim(xMinMax)
            ax.set_ylim(yMinMax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            # splitting on third quantity? load now
            if sQuant is not None:
                sim_svals, slabel, _, sLog = sP.simSubhaloQuantity(sQuant)
                if sim_svals is None:
                    print("   skip")
                    continue
                if sLog:
                    sim_svals = logZeroNaN(sim_svals)

            # coloring points by third quantity? load now
            sim_cvals = np.zeros(sim_xvals.size, dtype="float32")
            if scatterColor is not None:
                sim_cvals, clabel, cMinMax, cLog = sP.simSubhaloQuantity(scatterColor)
                if cLog:
                    sim_cvals = logZeroNaN(sim_cvals)
                cMinMax = cMinMax if clim is None else clim

            # flagging?
            sim_flag = np.ones(sim_xvals.shape[0]).astype("bool")
            if filterFlag and sP.groupCatHasField("Subhalo", "SubhaloFlag"):
                # load SubhaloFlag and override sim_flag (0=bad, 1=good)
                sim_flag = sP.groupCat(fieldsSubhalos=["SubhaloFlag"])

            # central/satellite selection?
            wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

            sim_yvals_orig = np.array(sim_yvals)
            sim_xvals_orig = np.array(sim_xvals)

            sim_yvals = sim_yvals[wSelect]
            sim_xvals = sim_xvals[wSelect]
            sim_cvals = sim_cvals[wSelect]
            sim_flag = sim_flag[wSelect]

            # reduce to the subset with non-NaN x/y-axis values (galaxy colors, i.e. minimum 1 star particle)
            wFinite = np.ones(sim_xvals.shape[0], dtype="bool")

            wFinite &= np.isfinite(sim_xvals) if sim_xvals.ndim == 1 else np.any(np.isfinite(sim_xvals), axis=1)
            wFinite &= np.isfinite(sim_yvals) if sim_yvals.ndim == 1 else np.any(np.isfinite(sim_yvals), axis=1)

            # reduce to the good-flagged subset
            wFinite &= sim_flag

            sim_xvals = sim_xvals[wFinite]
            sim_yvals = sim_yvals[wFinite]
            sim_cvals = sim_cvals[wFinite]

            # loop over (one or more) subsets, i.e. subhalo property restrictions, and plot
            # if none are requested, then loop just once and apply no restrictions
            subsets = [None]

            if qRestrictions is not None:
                if indivRestrictions:
                    subsets = [[qR] for qR in qRestrictions]  # one at a time in serial (several outputs)
                else:
                    subsets = [qRestrictions]  # all at once (one output)

                # qStr = ', '.join(['%g < %s < %g' % (q[1],q[0],q[2]) for q in qRestrictions])
                # ax.set_title(qStr)

            for k, locRestrictions in enumerate(subsets):
                sim_yy = sim_yvals.view()
                sim_xx = sim_xvals.view()
                sim_cc = sim_cvals.view()

                wRestrictions = []
                if locRestrictions is not None:
                    # apply one or more restrictions
                    rDesc = ""

                    for rFieldName, rFieldMin, rFieldMax in locRestrictions:
                        # load and restrict
                        vals, rLabel, _, _ = sP.simSubhaloQuantity(rFieldName)
                        vals = vals[wSelect][wFinite]

                        rDesc += "%s = %.1f" % (rLabel, np.mean([rFieldMin, rFieldMax]))

                        for wPastRestrict in wRestrictions:
                            vals = vals[wPastRestrict]  # AND

                        with np.errstate(invalid="ignore"):
                            wRestrict = np.where((vals >= rFieldMin) & (vals < rFieldMax))

                        sim_yy = sim_yy[wRestrict]
                        sim_xx = sim_xx[wRestrict]
                        sim_cc = sim_cc[wRestrict]

                        wRestrictions.append(wRestrict)
                        assert len(sim_xx)  # otherwise, no galaxies left

                # decide color
                c = ax.plot([], [])[0].get_color()

                if ctName is not None:
                    ct = ctName
                    bounds = None
                    if isinstance(ctName, list):
                        ct, bounds = ctName

                    c = sampleColorTable(ct, len(subsets), bounds=bounds)[k]

                # plot points if sample size is small enough, and we are otherwise just showing a median
                if sim_xx.size < ptPlotThresh and scatterPoints is None:
                    ax.plot(sim_xx, sim_yy, "o", color=c, alpha=alpha)

                # median and 10/90th percentile lines
                binSize = (xMinMax[1] - xMinMax[0]) / nBins
                if sP.boxSize < 205000.0:
                    binSize *= 2.0

                if drawMedian:
                    xm, ym, _, pm = running_median(sim_xx, sim_yy, binSize=binSize, minNumPerBin=20, percs=[16, 50, 84])

                    if xm.size > sKn:
                        ym = savgol_filter(ym, sKn, sKo)
                        pm = savgol_filter(pm, sKn, sKo, axis=1)

                    label = sP.simName + " z=%.1f" % sP.redshift if len(sPs) > 1 else ""
                    if medianLabel is not None:
                        label = medianLabel
                    if extraMedians:
                        label = yQuant
                    if len(subsets) > 1:
                        label += rDesc
                    color = "black" if (len(sPs) == 1 and len(subsets) == 1) else c

                    (l,) = ax.plot(xm, ym, linestyles[0], color=color, alpha=1.0, label=label)
                    if i == 0 and k == 0:
                        ax.fill_between(xm, pm[0, :], pm[-1, :], facecolor=l.get_color(), alpha=0.1, interpolate=True)

                for k, medianProp in enumerate(extraMedians):
                    # load new (y-axis) quantity, subset as before, and median
                    sim_mvals, mlabel, mMinMax, mLog = sP.simSubhaloQuantity(medianProp)
                    mlabel_orig = mlabel
                    if mLog:
                        sim_mvals = logZeroNaN(sim_mvals)

                    # verify units (in theory, must match with y-axis units...)
                    munits = ""
                    yunits = ""

                    if "[" in ylabel:
                        yunits = ylabel.split("[")[1][:-1].strip()
                        munits = mlabel.split("[")[1][:-1].strip()
                        mlabel = mlabel.split("[")[0]  # delete units

                    if "log " in munits and munits.replace("log ", "") == yunits:
                        # munits is log(yunits), can fix this
                        assert mLog
                        sim_mvals = 10.0**sim_mvals
                        munits = munits.replace("log ", "")

                    if medianProp == "size_gas":
                        sim_mvals /= 10
                        mlabel = "0.1" + mlabel

                    if munits != yunits:
                        print(
                            "WARNING: Extra median [%s] has units [%s] mismatch with existing y-units [%s]!"
                            % (medianProp, munits, yunits)
                        )

                    sim_mvals = sim_mvals[wSelect]
                    sim_mvals = sim_mvals[wFinite]
                    for wRestrict in wRestrictions:
                        sim_mvals = sim_mvals[wRestrict]

                    # make sure these new values are also finite
                    w = np.where(np.isfinite(sim_mvals))
                    sim_xx_e = sim_xx[w]
                    sim_mvals = sim_mvals[w]

                    xm, ym, _, pm = running_median(
                        sim_xx_e, sim_mvals, binSize=binSize, minNumPerBin=20, percs=[16, 50, 84]
                    )
                    if xm.size > sKn:
                        ym = savgol_filter(ym, sKn, sKo)
                        # sm = savgol_filter(sm,sKn,sKo)
                        pm = savgol_filter(pm, sKn, sKo, axis=1)

                    if mlabel in [label, ylabel] or mlabel_orig in [label, ylabel]:
                        mlabel = medianProp  # plotting very similar quantities, be explicit
                    mlabel = mlabel.replace("mg2_shape_", "SB > ").replace("mg2_area_", "SB > ")  # custom

                    ax.plot(xm, ym, linestyles[k + 1], color="black", alpha=0.8, label=mlabel)

                # slice value?
                if sQuant is not None:
                    svals_loc = sim_svals[wSelect][wFinite]
                    binSizeS = binSize * 2

                    xm, yma, ymb, pma, pmb = running_median_sub(
                        sim_xx, sim_yy, svals_loc, binSize=binSizeS, sPercs=sLowerPercs
                    )

                    for k, sLowerPerc in enumerate(sLowerPercs):
                        label = "%s < P[%d]" % (slabel, sLowerPerc)
                        ax.plot(xm, ymb[k], linestyles[1 + k], label=label)

                    lsOffset = 0

                    xm, yma, ymb, pma, pmb = running_median_sub(
                        sim_xx, sim_yy, svals_loc, binSize=binSizeS, sPercs=sUpperPercs
                    )

                    for k, sUpperPerc in enumerate(sUpperPercs):
                        label = "%s > P[%d]" % (slabel, sUpperPerc)
                        ax.plot(xm, yma[k], linestyles[1 + k + lsOffset], label=label)

                # contours (optionally conditional, i.e. independently normalized for each x-axis value)
                # todo

                # handle multi-dimensional (i.e. multiple values per subhalo) arrays
                maxdim = np.max([sim_xx.ndim, sim_yy.ndim, sim_cc.ndim])

                if maxdim > 1:
                    # how many entries per subhalo in the multi-d array?
                    maxdim_ind = np.argmax([sim_xx.ndim, sim_yy.ndim, sim_cc.ndim])
                    shape = [sim_xx.shape, sim_yy.shape, sim_cc.shape][maxdim_ind][1]

                    # of {x,y,c}, tile the remaining to match this shape
                    if sim_xx.ndim == 1:
                        sim_xx = np.tile(sim_xx.reshape((sim_xx.size, 1)), (1, shape))
                    if sim_yy.ndim == 1:
                        sim_yy = np.tile(sim_yy.reshape((sim_yy.size, 1)), (1, shape))
                    if sim_cc.ndim == 1:
                        sim_cc = np.tile(sim_cc.reshape((sim_cc.size, 1)), (1, shape))

                # scatter all points?
                if scatterPoints:
                    # reduce PDF weight, skip points outside of visible plot
                    w = np.where((sim_xx >= xMinMax[0]) & (sim_xx <= xMinMax[1]))

                    xx = sim_xx[w]
                    yy = sim_yy[w]
                    cc = sim_cc[w]

                    if xx.size > 100:
                        ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

                    ct = "viridis"

                    # relative coloring as a function of the x-axis?
                    if cRel is not None:
                        # override min,max of color and whether or not to log
                        if cLog:
                            cc = 10.0**cc  # remove log

                        cMinMax[0], cMinMax[1], cLog = cRel

                        bins = np.arange(np.nanmin(xx), np.nanmax(xx) + binSize, binSize)
                        # loop through bins
                        for k in range(bins.size - 1):
                            w = np.where((xx > bins[k]) & (xx <= bins[k + 1]))[0]
                            if len(w) == 0:
                                continue

                            # normalize all points in this bin by the bin median
                            normval = np.nanmedian(cc[w])
                            if normval == 0:
                                normval = np.nanmean(cc[w])
                            cc[w] /= normval

                        if cLog:  # log relative?
                            cc = logZeroNaN(cc)

                        ct = "curl"  # diverging
                        clabel = r"$\Delta$ " + clabel.split("[")[0] + ("[ log ]" if cLog else "")

                    if maxPointsPerDex is not None:
                        inds, _ = subsampleRandomSubhalos(sP, maxPointsPerDex, xMinMax, mstar=xx)

                        xx = xx[inds]
                        yy = yy[inds]
                        cc = cc[inds]

                    if lowessSmooth:
                        in1 = np.vstack((xx, yy))
                        cc = lowess(in1, cc, in1, degree=1, l=0.2)

                    # scatter color and marker
                    opts = {"color": c}

                    if scatterColor is not None:
                        # override constant color
                        fracSubset = None  # all

                        if ctName is not None:
                            ct = ctName
                            if isinstance(ctName, list):
                                ct, fracSubset = ctName

                        cmap = loadColorTable(ct, fracSubset=fracSubset)

                        opts = {"vmin": cMinMax[0], "vmax": cMinMax[1], "c": cc, "cmap": cmap}
                        # opts['label'] = '%s z=%.1f' % (sP.simName,sP.redshift) if len(sPs) > 1 else ''
                        opts["marker"] = "s" if sP.simName == "TNG-Cluster" else "o"

                    # plot scatter
                    zIsInt = np.abs(sP.redshift - int(sP.redshift)) < 0.01
                    zStr = " z=%d" % sP.redshift if zIsInt else " z=%.1f" % sP.redshift
                    label = (sP.simName + zStr) if len(sPs) > 1 else ""
                    if not labelSims:
                        label = ""  # do not add to legend (may contain other items)
                    if drawMedian:
                        label = ""  # only if median lines are not already labeled
                    sc = ax.scatter(xx, yy, s=markersize, alpha=alpha, **opts, label=label, zorder=0)

                # 1-to-1 line?
                if mark1to1:
                    x0 = np.min([ax.get_xlim()[0], ax.get_ylim()[0]])
                    x1 = np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
                    ax.plot([x0, x1], [x0, x1], ":", color="black", alpha=0.9, label="1-to-1")

                # highlight/overplot a single subhalo or a few subhalos?
                if markSubhaloIDs is not None:
                    c = sampleColorTable("tableau10", "red")
                    for subID in markSubhaloIDs:
                        label = "Subhalo #%d" % subID if len(markSubhaloIDs) <= 2 else ""
                        ax.scatter(
                            sim_xvals_orig[subID],
                            sim_yvals_orig[subID],
                            s=markersize * 2.4,
                            marker="o",
                            linewidth=1.5,
                            color=c,
                            facecolor="none",
                            alpha=1.0,
                            label=label,
                        )

        # legend
        if f_post is not None:
            f_post(ax)

        if i == 0 and legendLoc is not None:
            handles, labels = ax.get_legend_handles_labels()
            handlesO = []
            labelsO = []

            ax.legend(handles + handlesO, labels + labelsO, loc=legendLoc)

    # colorbar?
    if scatterPoints and scatterColor is not None:
        if colorbarInside:  # can generalize to 'upper left', etc
            # cax = inset_locator.inset_axes(ax, width="40%", height="4%", loc='upper left')

            # rect = [0.5,0.25,0.4,0.04] # lower right
            rect = [0.15, 0.84, 0.38, 0.04]  # upper left
            cax = fig.add_axes(rect)
            orientation = "horizontal"
            # cax.patch.set_facecolor('white') # doesn't work
            # cax.patch.set_alpha(1.0)

            cb = plt.colorbar(sc, cax=cax, orientation=orientation, ticks=cbarticks)
        else:
            orientation = "vertical"
            cb = fig.colorbar(sc, ax=ax, ticks=cbarticks, pad=0.02)

        # cb.set_alpha(1)  # fix stripes
        # cb.draw_all()
        if orientation == "vertical":
            cb.ax.set_ylabel(clabel)
        if orientation == "horizontal":
            cb.ax.set_title(clabel)
        if len(clabel) > 45:
            newsize = 27 - (len(clabel) - 45) / 5
            cb.ax.set_ylabel(clabel, size=newsize)  # default: 24.192 (14 * x-large)

    # finish plot and save
    if pdf is not None:
        pdf.savefig()
    else:
        if saveFilename is None:
            simNames = "-".join(sorted({sP.simName for sP in sPs}))
            colorStr = "-%s" % scatterColor if scatterColor is not None else ""
            yQuantsStr = "-".join(list(yQuants))
            saveFilename = "median_%s_%s-%s%s_%s.pdf" % (simNames, yQuantsStr, xQuant, colorStr, cenSatSelect)
        fig.savefig(saveFilename)

    plt.close(fig)
