"""
Plots related to galaxy clustering statistics, two-point correlation functions, and conformity.
"""

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from ..cosmo.clustering import conformityRedFrac, twoPointAutoCorrelationPeriodicCube
from ..plot.config import colors, cssLabels, figsize, linestyles, lw
from ..plot.util import sampleColorTable, setAxisColors
from ..util.helper import logZeroNaN
from ..util.match import match


def galaxyTwoPoint(sPs, cenSatSelects=("all", "cen", "sat"), colorBin=None, cType=None, mstarBin=None, mType=None):
    """Plot the galaxy two-point correlation function for a run or multiple runs."""
    # visual config
    rMinMax = [0.01, 100.0]  # log Mpc
    yMinMax = [1e-2, 5e4]

    rLabel = "r [ Mpc ]"
    yLabel = r"$\xi(r \pm \Delta r)$  [ real space two-point autocorr ]"

    # load/calculate
    cfs = []
    for sP in sPs:
        for cenSatSelect in cenSatSelects:
            rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(
                sP, cenSatSelect=cenSatSelect, colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType
            )

            cfs.append({"rad": rad, "xi": xi, "xi_err": xi_err, "css": cenSatSelect, "sP": sP})

    # start plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    setAxisColors(ax)

    ax.set_xlim(rMinMax)
    ax.set_ylim(yMinMax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(rLabel)
    ax.set_ylabel(yLabel)

    # plot
    for _i, cf in enumerate(cfs):
        xx = cf["sP"].units.codeLengthToMpc(cf["rad"])
        ww = np.where(cf["xi"] > 0.0)

        label = cf["sP"].simName
        if len(cenSatSelects) > 1:
            label += " " + cssLabels[cf["css"]]
        if cf["sP"].redshift > 0.0:
            label += " z = %.1f" % cf["sP"].redshift

        (l,) = ax.plot(xx, cf["xi"], "-", lw=lw, label=label)

        yy0 = cf["xi"][ww] - cf["xi_err"][ww] / 2
        yy1 = cf["xi"][ww] + cf["xi_err"][ww] / 2

        ax.fill_between(xx[ww], yy0, yy1, color=l.get_color(), interpolate=True, alpha=0.15)

    # finish plot
    ax.legend()
    fig.savefig("tpcf_%s.pdf" % ("_".join([sP.simName for sP in sPs])))
    plt.close(fig)


def galaxyTwoPointQuantBounds(
    sPs,
    cenSatSelect="all",
    ratioSubPlot=False,
    colorBins=None,
    cType=None,
    mstarBins=None,
    mType=None,
    redshiftBins=None,
):
    """Plot the galaxy two-point correlation function for a run or multiple runs.

    Show a range of bins in either color, stellar mass, or redshift (choose one).
    """
    if colorBins is not None and mstarBins is not None:
        assert len(colorBins) == 1 or len(mstarBins) == 1  # only one of the two can vary
    assert redshiftBins is None  # not implemented yet

    # visual config
    rMinMax = [-2.0, 2.0]  # log Mpc
    yMinMaxes = [[-2.0, 6.0], [-1.0, 3.5], [0.0, 3.0]]
    yMinMaxSub = [0.1, 10]
    alpha = 0.9
    alphaFill = 0.05

    ratioSubPlotHSpace = 0.02  # zero for none
    drawSymbols = True
    symSize = 7.0

    rLabel = "r [ log Mpc ]"
    yLabel = "log %s$\\xi(r)$"

    # load/calculate
    cfs = OrderedDict()

    loadByColor = False
    loadByMass = False
    if colorBins is not None and len(colorBins) > 1:
        loadByColor = True
    if not loadByColor:
        loadByMass = True

    if loadByColor:
        # can specify no mstarBin, or a single mstarBin, within which these color bins are applied
        mstarBin = mstarBins[0] if mstarBins is not None else mstarBins

        for colorBin in colorBins:
            # colorBins a dict? then key as label, otherwise a list of 2-tuples, so make label
            if isinstance(colorBins, dict):
                label = colorBin
                colorBin = colorBins[colorBin]
            else:
                label = "%.1f < (%s-%s) < %.1f" % (colorBin[0], cType[0][0], cType[0][1], colorBin[1])

            cfs[label] = []

            for sP in sPs:
                rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(
                    sP, cenSatSelect=cenSatSelect, colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType
                )

                cfs[label].append({"rad": rad, "xi": xi, "xi_err": xi, "sP": sP})

    if loadByMass:
        # can specify no colorBin, or a single colorBin, within which these mass bins are applied
        colorBin = colorBins[0] if colorBins is not None else colorBins

        for mstarBin in mstarBins:
            # mstarBins a dict? then key as label, otherwise a list of 2-tuples, so make label
            if isinstance(mstarBins, dict):
                label = mstarBin
                mstarBin = mstarBins[mstarBin]
            else:
                label = r"%4.1f < log($M_\star$/M$_{\rm sun}$) < %4.1f" % (mstarBin[0], mstarBin[1])

            cfs[label] = []

            for sP in sPs:
                rad, xi, xi_err, _ = twoPointAutoCorrelationPeriodicCube(
                    sP, cenSatSelect=cenSatSelect, colorBin=colorBin, cType=cType, mstarBin=mstarBin, mType=mType
                )

                cfs[label].append({"rad": rad, "xi": xi, "xi_err": xi, "sP": sP})

    if redshiftBins is not None:
        assert 0

    # specific color table?
    cm = None

    if colorBins is not None and len(colorBins) == 1 and len(mstarBins) > 2:
        # many mass bins, separate plots for red and blue samples
        cm = sampleColorTable("Oranges", np.sum([len(cfs[k]) for k in cfs.keys()]), bounds=[0.4, 1.0])

    if mstarBins is not None and len(mstarBins) == 1 and len(colorBins) > 2:
        # many color bins, separate plots for high-mass and low-mass samples
        cm = sampleColorTable("Purples", np.sum([len(cfs[k]) for k in cfs.keys()]), bounds=[0.4, 1.0])

    if colorBins is not None and len(colorBins) == 2:
        # binary blue/red split in one panel
        cm = sampleColorTable("tableau10", ["blue", "red"])

    if mstarBins is not None and len(mstarBins) == 2:
        # binary lowmass/highmass split in one panel
        cm = sampleColorTable("tableau10", ["green", "purple"])

    # iterate over y-axes: xi(r), r*xi(r), r^2*xi(r)
    for iterNum in [0, 1, 2]:
        # start plot
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.2**ratioSubPlot))
        gs = gridspec.GridSpec(1 + ratioSubPlot, 1, height_ratios=[3.5, 1])
        ax = fig.add_subplot(gs[0])
        setAxisColors(ax)

        if not ratioSubPlot:
            ax.set_xlim(rMinMax)
            ax.set_xlabel(rLabel)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)

        ax.set_ylim(yMinMaxes[iterNum])
        ax.set_ylabel(yLabel % ["", "r", "r$^2$"][iterNum])

        # ratio sub-plot on the bottom?
        if ratioSubPlot:
            ax_sub = fig.add_subplot(gs[1], sharex=ax)
            ax_sub.set_xlim(rMinMax)
            ax_sub.set_xlabel(rLabel)
            ax_sub.set_ylim(yMinMaxSub)
            ax_sub.set_yscale("log")

            if len(cfs) != 2:
                ax_sub.set_ylabel(r"$\xi(r)_{M_i}$ / $\xi(r)_{M_0}$")
            else:
                if loadByColor:
                    ax_sub.set_ylabel(r"$\xi(r)_{\\rm red}$ / $\xi(r)_{\rm blue}$")
                if loadByMass:
                    ax_sub.set_ylabel(r"$\xi(r)_{\\rm high}$ / $\xi(r)_{\rm low}$")
                if len(sPs) > 1:
                    # multi-redshift
                    ax_sub.set_ylabel(r"$\xi(r)_{z_i}$ / $\xi(r)_{z=0}$")

            for yVal in [0.5, 1.0, 2.0]:
                ax_sub.plot(rMinMax, [yVal, yVal], ":", color="black", alpha=0.1, lw=1.0)

            # remove last tick label for the second subplot
            # yticks = ax_sub.yaxis.get_major_ticks()
            # yticks[-1].label1.set_visible(False)

            # remove vertical space
            ax_points = ax.get_position().get_points()
            ax_sub_pos = ax_sub.get_position()
            ax_sub_points = ax_sub_pos.get_points()
            ax_sub_points[1][1] = ax_points[0][1] - ratioSubPlotHSpace
            ax_sub_pos.set_points(ax_sub_points)
            ax_sub.set_position(ax_sub_pos)

        yy_max = 0.0
        yy_max_sub = 0.0

        # plot: loop over each bin
        for k, cfBoundSet in enumerate(cfs.keys()):
            # loop over each run/redshift for this bin
            for i, cf in enumerate(cfs[cfBoundSet]):
                xx = cf["sP"].units.codeLengthToComovingMpc(cf["rad"])
                ww = np.where(cf["xi"] > 0.0)

                label = cfBoundSet if i == 0 else ""

                # y-axis multiplier
                yFac = 1.0
                if iterNum == 1:
                    yFac = xx[ww]
                if iterNum == 2:
                    yFac = xx[ww] ** 2

                x_plot = logZeroNaN(xx[ww])
                y_plot = logZeroNaN(yFac * cf["xi"][ww])

                if y_plot.max() > yy_max:
                    yy_max = y_plot.max()

                c = cm[k] if cm is not None else colors[k]
                (l,) = ax.plot(x_plot, y_plot, lw=lw, linestyle=linestyles[i], alpha=alpha, label=label, color=c)

                if drawSymbols:
                    yy0 = y_plot - logZeroNaN(yFac * (cf["xi"][ww] - cf["xi_err"][ww] / 2))
                    yy1 = logZeroNaN(yFac * (cf["xi"][ww] + cf["xi_err"][ww] / 2)) - y_plot
                    ax.errorbar(
                        x_plot,
                        y_plot,
                        yerr=[yy0, yy1],
                        markerSize=symSize,
                        color=l.get_color(),
                        ecolor=l.get_color(),
                        alpha=alpha,
                        capsize=0.0,
                        fmt="o",
                    )

                if i == 0:
                    yy0 = logZeroNaN(yFac * (cf["xi"][ww] - cf["xi_err"][ww] / 2))
                    yy1 = logZeroNaN(yFac * (cf["xi"][ww] + cf["xi_err"][ww] / 2))

                    ax.fill_between(x_plot, yy0, yy1, color=l.get_color(), interpolate=True, alpha=alphaFill)

                # add ratio to sub plot?
                if not ratioSubPlot:
                    continue

                if (len(sPs) == 1 and k == 0) or (len(sPs) > 1 and i == 0):
                    # save
                    k0_x = x_plot
                    k0_y = y_plot
                    k0_yerr = yFac * cf["xi_err"][ww]
                else:
                    # plot ratio
                    w0, w = match(k0_x, x_plot)
                    assert np.array_equal(k0_x[w0], x_plot[w])
                    ysub_plot = 10.0 ** y_plot[w] / 10.0 ** k0_y[w0]

                    # inherit color, unless we just have 2 lines (1 ratio) then make it black
                    c = "black" if (len(cfs) == 2 and len(cfs[cfBoundSet]) == 1) else c

                    ax_sub.plot(x_plot[w], ysub_plot, lw=lw, linestyle=linestyles[i], label=label, color=c)

                    if ysub_plot.max() > yy_max_sub:
                        yy_max_sub = ysub_plot.max()

                    # add errors in fractional quadrature
                    k0_frac_err = k0_yerr[w0] / 10.0 ** k0_y[w0]
                    sub_frac_err = cf["xi_err"][w] / 10.0 ** y_plot[w]

                    yerr_sub = np.sqrt(k0_frac_err**2 + sub_frac_err**2) * ysub_plot / 2

                    # import pdb; pdb.set_trace() # fix low errors in ratios

                    if drawSymbols:
                        ax_sub.errorbar(
                            x_plot[w],
                            ysub_plot,
                            markerSize=symSize,
                            color=c,
                            ecolor=c,
                            alpha=alpha,
                            capsize=0.0,
                            fmt="o",
                        )
                    if (len(sPs) == 1 and i == 0) or (len(sPs) > 1 and i == 1):
                        ax_sub.fill_between(
                            x_plot[w],
                            ysub_plot - yerr_sub,
                            ysub_plot + yerr_sub,
                            color=c,
                            interpolate=True,
                            alpha=alphaFill,
                        )

        # set y maximum
        ax.set_ylim(ymax=np.ceil(yy_max))

        # legends
        loc = "lower left" if iterNum < 2 else "upper left"
        legend1 = ax.legend(loc=loc)
        ax.add_artist(legend1)

        if len(sPs) > 0:
            handles, labels = [], []
            for i, sP in enumerate(sPs):
                handles.append(plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[i]))
                labels.append(sP.simName + " z=%.1f" % sP.redshift)
            if colorBins is not None and len(colorBins) == 1:
                handles.append(plt.Line2D([0], [0], color="black", lw=0.0, marker=""))

            legend2 = ax.legend(handles, labels, loc="upper right")
            ax.add_artist(legend2)

        if ratioSubPlot:
            yy_max_sub = np.max([10, yy_max_sub * 1.1])
            if yy_max_sub > 10:
                yy_max_sub = np.ceil(yy_max_sub / 10.0) * 10  # round
            ax_sub.set_ylim(ymax=yy_max_sub)
            # ax_sub.legend()

        fig.savefig("tpcf_%s_%s.pdf" % (["xi", "rxi", "r2xi"][iterNum], "_".join([sP.simName for sP in sPs])))
        plt.close(fig)


def conformityWithRedFrac(sP, cenSatSelectSec="all"):
    """Plot the galaxy two-point correlation function for a run or multiple runs.

    Show a range of bins in either color, stellar mass, or redshift (choose one).
    NOTE: Conformity analysis incomplete (need to reconcile with Bray+/Illustris).
    """
    # analysis config
    cheapDustColorModel = "p07c_cf00dust_rad30pkpc"
    colorSplitSec = 0.6
    cType = [["g", "r"], cheapDustColorModel]
    colorBinsPri = [[0.0, 0.6], [0.6, 1.0]]
    massBinsPri = [[9.5, 10.0], [10.0, 10.5], [10.5, 11.0]]
    mTypePri = "mstar_30pkpc_log"

    # visual config
    rMinMax = [0.0, 20.0]  # Mpc
    yMinMax = [0.2, 0.8]  # red fraction
    alpha = 0.9
    alphaFill = 0.05

    drawSymbols = False
    symSize = 7.0

    rLabel = "r [ Mpc ]"
    yLabel = "Secondary Red Fraction"

    # load/calculate
    confs = OrderedDict()

    for massBinPri in massBinsPri:
        label = r"%4.1f < log($M_\star$/M$_{\rm sun}$) < %4.1f" % (massBinPri[0], massBinPri[1])
        confs[label] = OrderedDict()

        for colorBinPri in colorBinsPri:
            conf = conformityRedFrac(
                sP,
                colorBin=colorBinPri,
                cType=cType,
                mstarBin=massBinPri,
                mType=mTypePri,
                cenSatSelectSec=cenSatSelectSec,
                colorSplitSec=colorSplitSec,
            )

            colorLabel = "%.1f < (%s-%s) < %.1f" % (colorBinPri[0], cType[0][0], cType[0][1], colorBinPri[1])
            confs[label][colorLabel] = conf

    # start plot
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[0])
    setAxisColors(ax)

    ax.set_xlim(rMinMax)
    ax.set_xlabel(rLabel)
    ax.set_ylim(yMinMax)
    ax.set_ylabel(yLabel)

    cm = None

    # loop over each mass bin
    for k, massSet in enumerate(confs.keys()):
        # loop over each color (or secondary property in general) split for this mass bin
        for i, secSet in enumerate(confs[massSet].keys()):
            rf = confs[massSet][secSet]

            xx = rf["rad"]
            ww = np.where(rf["redfrac"] > 0.0)

            label = massSet + " " + secSet  # if i == 0 else ''

            # y-axis multiplier
            c = cm[k] if cm is not None else colors[k]
            (l,) = ax.plot(xx[ww], rf["redfrac"][ww], lw=lw, linestyle=linestyles[i], alpha=alpha, label=label, color=c)

            if drawSymbols:
                ax.errorbar(
                    xx[ww],
                    rf["redfrac"][ww],
                    yerr=rf["redfrac_error"][ww],
                    markerSize=symSize,
                    color=l.get_color(),
                    ecolor=l.get_color(),
                    alpha=alpha,
                    capsize=0.0,
                    fmt="o",
                )

            if i == 0:
                yy0 = rf["redfrac"][ww] - rf["redfrac_err"][ww] / 2
                yy1 = rf["redfrac"][ww] + rf["redfrac_err"][ww] / 2

                ax.fill_between(xx[ww], yy0, yy1, color=l.get_color(), interpolate=True, alpha=alphaFill)

    # legends
    legend1 = ax.legend(loc="upper left")
    ax.add_artist(legend1)

    handles = [plt.Line2D([0], [0], color="black", lw=lw, marker="", linestyle=linestyles[i])]
    labels = [sP.simName]
    legend2 = ax.legend(handles, labels, loc="upper right")
    ax.add_artist(legend2)

    fig.savefig("conformity_redfrac_%s.pdf" % (sP.simName))
    plt.close(fig)
