"""
Plots of galaxy/halo-scale gas flow rates, velocities, and related properties.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata, interp1d
from scipy.signal import savgol_filter

from ..catalog.gasflows import radialMassFluxes
from ..plot.config import colors, figsize, linestyles, lw, sKn, sKo
from ..plot.util import loadColorTable
from ..util.helper import logZeroNaN, running_median, sgolay2d
from ..util.match import match
from ..util.simParams import simParams


labels = {
    "rad": r"Radius [ pkpc ]",
    "vrad": r"Radial Velocity [ km/s ]",
    "vcut": r"Minimum Outflow Velocity Cut [ km/s ]",
    "temp": r"Gas Temperature [ log K ]",
    "temp_sfcold": r"Gas Temperature (eEOS=$10^3$K) [ log K ]",
    "z_solar": r"Gas Metallicity [ log Z$_{\rm sun}$ ]",
    "numdens": r"Gas Density [ log cm$^{-3}$ ]",
    "theta": r"Galactocentric Angle [ 0, $\pm\pi$ = major axis ]",
}


def outflowRates(
    sP, ptType, xQuant="mstar_30pkpc", eta=False, config=None, massField="Masses", v200norm=False, f_post=None
):
    """Plot radial mass flux (single value per galaxy) as a function of stellar mass or other galaxy properties."""
    # config
    scope = "SubfindWithFuzz"  # or 'Global'
    ptTypes = ["Gas", "Wind", "total"]

    assert ptType in ptTypes
    if eta and massField == "Masses":
        assert ptType == "total"  # avoid ambiguity, since massLoadingsSN() is always total
    if massField != "Masses":
        assert ptType == "Gas"  # avoid ambiguity, since other massField's only exist for Gas

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == "mstar_30pkpc":
        xlim = [7.5, 11.25]
        xlabel = r"Stellar Mass [ log M$_{\rm sun}$ ]"

    if config is not None and "xlim" in config:
        xlim = config["xlim"]

    # plot config (y)
    if eta:
        saveBase = "massLoading"
        pStr1 = ""
        pStr2 = "w"  # 'wind'
        ylim = [-1.15, 1.65]  # mass loadings default
        if massField != "Masses":
            pStr1 = r"_{\rm %s}" % massField
            pStr2 = massField
            ylim = [-10.5, -2.0]
            saveBase += massField
        ylabel = r"Mass Loading $\eta%s = \dot{M}_{\rm %s} / \dot{M}_\star$ [ log ]" % (pStr1, pStr2)
    else:
        saveBase = "outflowRate"
        pStr = "%s " % ptType if ptType != "total" else ""
        if massField != "Masses":
            pStr = "%s " % massField
        ylabel = r"%sOutflow Rate [ log M$_{\rm sun}$ / yr ]" % pStr
        ylim = [-2.8, 2.5]  # outflow rates default

    ptStr = "_%s" % ptType
    binSize = 0.2  # in M*
    markersize = 0.0  # 4.0, or 0.0 to disable
    malpha = 0.2
    percs = [16, 84]

    def _plotHelper(vcutIndsPlot, radIndsPlot, saveName=None, pdf=None, ylimLoc=None, stat="median", addModelTNG=False):
        """Plot a radii series, vcut series, or both."""
        # plot setup
        fig, ax = plt.subplots()

        if ylimLoc is None:
            ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if config is not None and "xlabel" in config:
            ax.set_xlabel(config["xlabel"])
        if config is not None and "ylabel" in config:
            ax.set_ylabel(config["ylabel"])

        labels_sec = []
        colors = []

        if addModelTNG:
            # load mass loading (of TNG model at injection) analysis
            GFM_etaM_mean = sP.subhalos("wind_etaM")

            # load x-axis property
            GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
            if takeLog:
                GFM_xquant = logZeroNaN(GFM_xquant)

            if xQuant == "mstar_30pkpc":
                fit_xlim = [7.5, 11.5]  # override

            # plot points
            # with np.errstate(invalid='ignore'):
            #    w = np.where(GFM_etaM_mean > 0)
            # ax.plot(GFM_xquant[w], logZeroNaN(GFM_etaM_mean[w]), 'o', color='red', alpha=0.2)

            # fit
            with np.errstate(invalid="ignore"):
                w_fit = np.where((GFM_etaM_mean > 0) & (GFM_xquant > fit_xlim[0]) & (GFM_xquant < fit_xlim[1]))
            x_fit = GFM_xquant[w_fit]
            y_fit = logZeroNaN(GFM_etaM_mean[w_fit])

            result, _, _, _, _ = np.polyfit(x_fit, y_fit, 2, full=True, cov=False)
            xx = np.linspace(fit_xlim[0], fit_xlim[1], 30)
            yy = np.polyval(result, xx)

            # plot fit
            ax.fill_between(xx, yy - 0.1, yy + 0.1, color="black", interpolate=True, alpha=0.1)
            ax.plot(xx, yy, "-", color="black", alpha=0.6)
            if len(radIndsPlot) == 1:
                ax.text(10.5, 0.03, "TNG Model (at Injection)", color="black", alpha=0.6, rotation=-43.0)
            else:
                ax.text(10.78, -0.24, "TNG Model", color="black", alpha=0.6, rotation=-43.0)
                ax.text(10.69, -0.32, "(at Injection)", color="black", alpha=0.6, rotation=-43.0)

        txt = []

        # loop over radii and/or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):
            for j, vcut_ind in enumerate(vcutIndsPlot):
                # local data
                yy = np.squeeze(vals[:, rad_ind, vcut_ind]).copy()  # zero flux -> nan, skipped in median

                # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                if 0:
                    w_zero = np.where(yy == 0.0)
                    yy[w_zero] = np.nan
                    # note: currently does nothing given the logZeroNaN() below, which skips zeros regardless

                    yy = logZeroNaN(yy)  # zero flux -> nan (skipped in median)

                # label and color
                if rad_ind < binConfig["rad"].size - 1:
                    radMidPoint = "%3d kpc" % (0.5 * (binConfig["rad"][rad_ind] + binConfig["rad"][rad_ind + 1]))
                else:
                    radMidPoint = "all"

                if len(vcutIndsPlot) == 1:
                    label = "r = %s" % radMidPoint
                    labelFixed = r"v$_{\rm rad}$ > %3d km/s" % vcut_vals[vcut_ind]
                    if v200norm:
                        labelFixed = r"v$_{\rm rad}$ > %.1f v$_{\rm 200}$" % vcut_vals[vcut_ind]
                if len(radIndsPlot) == 1:
                    label = r"v$_{\rm rad}$ > %3d km/s" % vcut_vals[vcut_ind]
                    if v200norm:
                        label = r"v$_{\rm rad}$ > %.1f v$_{\rm 200}$" % vcut_vals[vcut_ind]
                    labelFixed = "r = %s" % radMidPoint
                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    label = "r = %s" % radMidPoint  # primary label radius, by color
                    if not v200norm:
                        # second legend: vcut by ls
                        labels_sec.append(r"v$_{\rm rad}$ > %3d km/s" % vcut_vals[vcut_ind])
                    else:
                        labels_sec.append(r"v$_{\rm rad}$ > %.1fv$_{\rm 200}$" % vcut_vals[vcut_ind])
                    if j > 0:
                        label = ""

                if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
                    # one color per v_rad, if cycling over both
                    if i == 0:
                        ax.set_prop_cycle(None)  # reset color cycle

                # symbols for each system
                if markersize > 0:  # or (i==1 and j==1): # hard-coded option
                    size = markersize if markersize > 0 else 4.0
                    yy_mark = logZeroNaN(yy)
                    (l,) = ax.plot(xvals, yy_mark, "s", markersize=size, alpha=malpha, rasterized=True)

                    # mark those at absolute zero just above the bottom of the y-axis
                    off = 0.2
                    w_zero = np.where(np.isnan(yy_mark))
                    yy_zero = np.random.uniform(size=len(w_zero[0]), low=ylim[0] + off / 2, high=ylim[0] + off)
                    ax.plot(xvals[w_zero], yy_zero, "s", alpha=malpha / 2, markersize=size, color=l.get_color())

                # median line and 1sigma band
                xm, ym, sm, pm = running_median(xvals, yy, binSize=binSize, percs=percs, mean=(stat == "mean"))

                # take log after running mean/median, instead of before, such that zeros are considered
                ym = logZeroNaN(ym)
                sm = logZeroNaN(sm)
                pm = logZeroNaN(pm)

                if xm.size > sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)

                lsInd = i if len(vcutIndsPlot) < 4 else j
                if markersize > 0:
                    lsInd = 0
                (l,) = ax.plot(xm, ym, linestyles[lsInd], label=label)

                txt.append({"mstar": xm, "eta": ym, "rad": radMidPoint, "vcut": vcut_vals[vcut_ind]})

                # shade percentile band?
                if i == j or markersize > 0 or "mstar" not in xQuant:
                    y_down = pm[0, :]  # np.array(ym[:-1]) - sm[:-1]
                    y_up = pm[-1, :]  # np.array(ym[:-1]) + sm[:-1]

                    # repairs
                    w = np.where(np.isnan(y_up))[0]
                    if len(w) and len(w) < len(y_up):
                        lastGoodInd = np.max(w) + 1
                        lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                        y_up[w] = ym[:][w] + lastGoodVal

                    w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                    y_down[w] = ylimLoc[0]  # off bottom

                    # plot bottom
                    ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # special plotting behavior (including observational data sets)
        if f_post is not None:
            f_post(ax, locals())

        # legends and finish plot
        legParams = {"frameon": 1, "framealpha": 0.9, "fancybox": False}  # to add white background to legends

        loc1 = "upper right" if eta else "upper left"
        loc2 = "lower right" if len(radIndsPlot) > 1 else "lower left"
        if config is not None and "loc1" in config:
            loc1 = config["loc1"]
        if config is not None and "loc2" in config:
            loc2 = config["loc2"]

        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            if loc2 is not None:
                line = plt.Line2D([0], [0], color="white", marker="", lw=0.0)
                legend2 = ax.legend([line], [labelFixed], loc=loc2, handlelength=-0.5, **legParams)
                # for text in legend2.get_texts(): text.set_color('white')
                # frame = legend2.get_frame()
                # frame.set_facecolor('white')
                # ax.add_artist(legend2) # r = X kpc, z = Y

            locParams = {} if (config is None or "leg1white" not in config) else legParams
            if loc1 is not None:
                legend1 = ax.legend(loc=loc1, **locParams)  # vrad > ...

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [plt.Line2D([0], [0], color=colors[j], marker="", ls="-") for j in range(len(vcutIndsPlot))]
            legend2 = ax.legend(lines, labels_sec, loc="upper right")
            ax.add_artist(legend2)

            legend1 = ax.legend(loc="lower right" if eta else "upper left")
            for handle in legend1.legendHandles:
                handle.set_color("black")

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    mdot = {}
    mdot["Gas"], mstar_30pkpc_log, sub_ids, binConfig, numBins, vcut_vals = radialMassFluxes(
        sP, scope, "Gas", massField=massField, v200norm=v200norm
    )

    if massField == "Masses":
        mdot["Wind"], _, sub_ids, binConfig, numBins, vcut_vals = radialMassFluxes(
            sP, scope, "Wind", massField=massField, v200norm=v200norm
        )
        mdot["total"] = mdot["Gas"] + mdot["Wind"]
    else:
        mdot["total"] = mdot["Gas"]

    # load mass loadings (total)
    acField = "Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr"
    if massField != "Masses":
        acField = "Subhalo_MassLoadingSN_%s_SubfindWithFuzz_SFR-100myr" % massField
    if v200norm:
        acField += "_v200norm"

    etaM = sP.auxCat(acField)[acField]

    if eta:
        vals = etaM
    else:
        vals = mdot[ptType]

    # restrict Mdot/etaM values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
    if config is not None and "minMstar" in config:
        w = np.where(mstar_30pkpc_log < config["minMstar"])
        vals[w] = np.nan

    # load x-axis values, stellar mass or other?
    xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
    xvals = xvals[sub_ids]
    if takeLog:
        xvals = logZeroNaN(xvals)

    if xlabel is None:
        xlabel = xlabel2  # use suggestions if not hard-coded above
    if xlim is None:
        xlim = xlim2

    # one specific plot requested? make now and exit
    if config is not None:
        saveName = "%s%s_%s_%s_%d_v%dr%d_%s_%s.pdf" % (
            saveBase,
            ptStr,
            xQuant,
            sP.simName,
            sP.snap,
            len(config["vcutInds"]),
            len(config["radInds"]),
            config["stat"],
            "_v200norm" if v200norm else "",
        )

        if "saveName" in config:
            saveName = config["saveName"]
        if "markersize" in config:
            markersize = config["markersize"]
        if "addModelTNG" not in config:
            config["addModelTNG"] = False
        if "ylim" not in config:
            config["ylim"] = None
        if "percs" in config:
            percs = config["percs"]

        _plotHelper(
            vcutIndsPlot=config["vcutInds"],
            radIndsPlot=config["radInds"],
            saveName=saveName,
            stat=config["stat"],
            ylimLoc=config["ylim"],
            addModelTNG=config["addModelTNG"],
        )
        return

    # plot
    for stat in ["mean"]:  # ['mean','median']:
        print(ptType, stat, "eta:", eta)
        # (A) plot for a given vcut, at many radii
        radInds = [1, 3, 4, 5, 6, 7]

        pdf = PdfPages("%s%s_%s_A_%s_%d_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, sP.snap, stat))
        for vcut_ind in range(vcut_vals.size):
            _plotHelper(vcutIndsPlot=[vcut_ind], radIndsPlot=radInds, pdf=pdf, stat=stat)
        pdf.close()

        # (B) plot for a given radii, at many vcuts
        vcutInds = [0, 1, 2, 3, 4]

        pdf = PdfPages("%s%s_%s_B_%s_%d_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, sP.snap, stat))
        for rad_ind in range(numBins["rad"]):
            _plotHelper(vcutIndsPlot=vcutInds, radIndsPlot=[rad_ind], pdf=pdf, stat=stat)
        pdf.close()

        # (C) single-panel combination of both radial and vcut variations
        if ptType in ["Gas", "total"]:
            vcutIndsPlot = [0, 2, 3]
            radIndsPlot = [1, 2, 5]
            ylimLoc = [-2.5, 2.0] if not eta else ylim

        if ptType == "Wind":
            vcutIndsPlot = [0, 2, 4]
            radIndsPlot = [1, 2, 5]
            ylimLoc = [-3.0, 1.0]

        saveName = "%s%s_%s_C_%s_%d_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, sP.snap, stat)
        _plotHelper(vcutIndsPlot, radIndsPlot, saveName, ylimLoc=ylimLoc, stat=stat)


def outflowRatesVsRedshift(sP, ptType, eta=False, config=None, massField="Masses", v200norm=False):
    """Plot radial mass fluxes (single value per galaxy) as a function of redshift (for bins of other gal props)."""
    # config
    scope = "SubfindWithFuzz"  # or 'Global'
    ptTypes = ["Gas", "Wind", "total"]

    assert ptType in ptTypes
    if eta and massField == "Masses":
        assert ptType == "total"  # to avoid ambiguity, since massLoadingsSN() is always total
    if massField != "Masses":
        assert ptType == "Gas"  # to avoid ambiguity, since other massField's only exist for Gas

    binQuant = "mstar_30pkpc"
    bins = [[7.9, 8.1], [8.9, 9.1], [9.4, 9.6], [9.9, 10.1], [10.4, 10.6], [10.6, 11.4]]

    redshifts = [0.2, 0.5, 1.0, 2.0, 4.0, 5.0, 6.0]

    xlim = [0, 6.0]
    xlabel = "Redshift"

    # plot config (y)
    if eta:
        saveBase = "massLoadingVsRedshift"
        pStr1 = ""
        pStr2 = "w"  # 'wind'
        ylim = [-1.15, 1.65]  # mass loadings default
        if massField != "Masses":
            pStr1 = r"_{\rm %s}" % massField
            pStr2 = massField
            ylim = [-10.5, -2.0]
            saveBase += massField
        ylabel = r"Mass Loading $\eta%s = \dot{M}_{\rm %s} / \dot{M}_\star$ [ log ]" % (pStr1, pStr2)
    else:
        saveBase = "outflowRateVsRedshift"
        pStr = "%s " % ptType if ptType != "total" else ""
        if massField != "Masses":
            pStr = "%s " % massField
        ylabel = r"%sOutflow Rate [ log M$_{\rm sun}$ / yr ]" % pStr
        ylim = [-2.8, 2.5]  # outflow rates default

    ptStr = "_%s" % ptType
    percs = [16, 84]

    def _plotHelper(vcutIndsPlot, radIndsPlot, saveName=None, pdf=None, ylimLoc=None, stat="median"):
        """Plot a radii series, vcut series, or both."""
        # plot setup
        fig, ax = plt.subplots()

        if ylimLoc is None:
            ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # ax.set_xscale('symlog')
        # ax.set_xticks([0,0.5,1,2,3,4,5,6])

        if config is not None and "xlabel" in config:
            ax.set_xlabel(config["xlabel"])
        if config is not None and "ylabel" in config:
            ax.set_ylabel(config["ylabel"])

        labels_sec = []
        colors = []

        # loop over radii and/or vcut selections
        for i, rad_ind in enumerate(radIndsPlot):
            for j, vcut_ind in enumerate(vcutIndsPlot):
                # allocate
                binned_result = np.zeros((len(bins), len(redshifts)), dtype="float32")
                binned_percs = np.zeros((len(percs), len(bins), len(redshifts)), dtype="float32")

                # loop over redshifts
                for zInd in len(redshifts):
                    # local data at this redshift
                    yy = np.squeeze(data[zInd][:, rad_ind, vcut_ind]).copy()  # zero flux -> nan, skipped in median
                    loc_binvals = data_binning[zInd]

                    # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                    if 0:
                        yy = logZeroNaN(yy)  # zero flux -> nan (skipped in median)

                    # median value and save
                    for binInd, bin_edges in enumerate(bins):
                        w = np.where((loc_binvals >= bin_edges[0]) & (loc_binvals < bin_edges[1]))

                        result = np.nanmedian(yy[w])

                        if stat == "mean":
                            result = np.nanmean(yy[w])

                        binned_result[binInd, zInd] = result

                        # percentiles
                        binned_percs[:, binInd, zInd] = np.nanpercentile(yy[w], percs)

                # plot (once per radInd/vcutInd/quantBin)
                for binInd, bin_edges in enumerate(bins):
                    # local
                    xm = redshifts
                    ym = binned_result[binInd, :]
                    pm = binned_percs[:, binInd, :]

                    cmap = loadColorTable("viridis", numColors=None)
                    c = cmap(float(binInd) / len(bins))

                    # take log after running mean/median, instead of before, allows zeros to be considered
                    ym = logZeroNaN(ym)
                    pm = logZeroNaN(pm)

                    # if binInd == 5: # remove last nan
                    #    xm = xm[:-1]
                    #    ym = ym[:-1]
                    #    pm = pm[:,-1]

                    if ym.size > sKn:
                        ym = savgol_filter(ym, sKn, sKo)
                        pm = savgol_filter(pm, sKn, sKo, axis=1)

                    # lsInd = i if len(vcutIndsPlot) < 4 else j
                    label = r"$M_\star / \rm{M}_\odot = 10^{%.1f}$" % np.mean(bin_edges)
                    (l,) = ax.plot(xm, ym, "-", ls=linestyles[i], color=c, label=label)

                    if binInd == 5:
                        diff0 = pm[-1, :] - ym
                        diff1 = ym - pm[0, :]
                        sym_diff = np.max(np.vstack((diff0, diff1)), axis=0)
                        pm[-1, :] = ym + sym_diff
                        pm[0, :] = ym - sym_diff
                        pm[-1, -1] /= 1.2  # out of statistics, expand visually
                        pm[0, -1] *= 1.2

                    # shade percentile band
                    if i == j:
                        ax.fill_between(xm, pm[0, :], pm[-1, :], color=l.get_color(), interpolate=True, alpha=0.05)

        # legends and finish plot
        legParams = {"frameon": 1, "framealpha": 0.9, "fancybox": False}  # to add white background to legends

        loc1 = "upper right" if eta else "upper left"
        loc2 = "lower right" if len(radIndsPlot) > 1 else "lower left"
        if config is not None and "loc1" in config:
            loc1 = config["loc1"]
        if config is not None and "loc2" in config:
            loc2 = config["loc2"]

        if len(vcutIndsPlot) == 1 or len(radIndsPlot) == 1:
            # label
            if rad_ind < binConfig["rad"].size - 1:
                radMidPoint = "%3d kpc" % (0.5 * (binConfig["rad"][rad_ind] + binConfig["rad"][rad_ind + 1]))
            else:
                radMidPoint = "all"

            labelFixed = r"r = %s, v$_{\rm rad}$ > %3d km/s" % (radMidPoint, vcut_vals[vcut_ind])
            line = plt.Line2D([0], [0], color="white", marker="", lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc=loc2, handlelength=-0.5, **legParams)
            ax.add_artist(legend2)

            locParams = {} if (config is None or "leg1white" not in config) else legParams
            if loc1 is not None:
                legend1 = ax.legend(loc=loc1, ncol=2, **locParams)  # vrad > ...

        if len(vcutIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [plt.Line2D([0], [0], color=colors[j], marker="", ls="-") for j in range(len(vcutIndsPlot))]
            legend2 = ax.legend(lines, labels_sec, loc="upper right")
            ax.add_artist(legend2)

            legend1 = ax.legend(loc="lower right" if eta else "upper left")
            for handle in legend1.legendHandles:
                handle.set_color("black")

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates
    data = []
    data_binning = []

    for redshift in redshifts:
        sP_loc = simParams(res=sP.res, run=sP.run, redshift=redshift)
        mdot = {}
        mdot["Gas"], mstar_30pkpc_log, sub_ids, binConfig, numBins, vcut_vals = radialMassFluxes(
            sP_loc, scope, "Gas", massField=massField, v200norm=v200norm
        )

        if massField == "Masses":
            mdot["Wind"], _, sub_ids, binConfig, numBins, vcut_vals = radialMassFluxes(
                sP_loc, scope, "Wind", massField=massField, v200norm=v200norm
            )
            mdot["total"] = mdot["Gas"] + mdot["Wind"]
        else:
            mdot["total"] = mdot["Gas"]

        # load mass loadings (total)
        acField = "Subhalo_MassLoadingSN_SubfindWithFuzz_SFR-100myr"
        if massField != "Masses":
            acField = "Subhalo_MassLoadingSN_%s_SubfindWithFuzz_SFR-100myr" % massField
        if v200norm:
            acField += "_v200norm"

        etaM = sP_loc.auxCat(acField)[acField]

        if eta:
            vals = etaM
        else:
            vals = mdot[ptType]

        # restrict Mdot/etaM values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
        if config is not None and "minMstar" in config:
            w = np.where(mstar_30pkpc_log < config["minMstar"])
            vals[w] = np.nan

        # load binning values, stellar mass or other?
        binvals, _, _, takeLog = sP_loc.simSubhaloQuantity(binQuant)
        binvals = binvals[sub_ids]
        if takeLog:
            binvals = logZeroNaN(binvals)

        # append
        data.append(vals)
        data_binning.append(binvals)

    # one specific plot
    v200str = "_v200norm" if v200norm else ""
    saveName = "%s%s_%s_%d_v%dr%d_%s_%s.pdf" % (
        saveBase,
        ptStr,
        sP.simName,
        sP.snap,
        len(config["vcutInds"]),
        len(config["radInds"]),
        config["stat"],
        v200str,
    )
    if "saveName" in config:
        saveName = config["saveName"]
    if "ylim" not in config:
        config["ylim"] = None
    if "percs" in config:
        percs = config["percs"]

    _plotHelper(
        vcutIndsPlot=config["vcutInds"],
        radIndsPlot=config["radInds"],
        saveName=saveName,
        stat=config["stat"],
        ylimLoc=config["ylim"],
    )


def outflowVel(
    sP_in,
    xQuant="mstar_30pkpc",
    ylog=False,
    redshifts=(None,),
    config=None,
    massField="Masses",
    proj2D=False,
    v200norm=False,
    f_post=None,
):
    """Plot outflow velocity (single value per galaxy) versus stellar mass or other gal/halo properties.

    If massField is not 'Masses', then e.g. the ion mass ('SiII', 'MgII') to use to compute massflux-weighted
    outflow velocities. If proj2D, then line-of-sight 1D projected velocities are computed in the down-the-barrel
    treatment, instead of the usual 3D radial velocities.
    """
    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant)  # copy

    # config
    scope = "SubfindWithFuzz"  # or 'Global'

    mdotThreshVcutInd = 0  # vrad>0 km/s
    mdotThreshValue = 0.0  # msun/yr

    # plot config (y)
    ylim = [0, 1200]

    if massField == "Masses":
        ylabel = "Outflow Velocity [ km/s ]"
        saveBase = "outflowVelocity"
        ptStr = "_total"
    else:
        ylabel = "%s Outflow Velocity [ km/s ]" % massField
        saveBase = "outflowVelocity%s" % massField
        ptStr = "_Gas"

    if proj2D:
        ylabel = "Line-of-sight " + ylabel
        saveBase += "2DProj"

    if ylog:
        ylabel = ylabel.replace("km/s", "log km/s")
        ylim = [1.4, 3.2]
    if v200norm:
        ylabel = ylabel.replace("km/s", r"v$_{\rm 200}$")

    # plot config (x): values not hard-coded here set automatically by simSubhaloQuantity() below
    xlim = None
    xlabel = None

    if xQuant == "mstar_30pkpc":
        xlim = [7.5, 11.0]
        xlabel = r"Stellar Mass [ log M$_{\rm sun}$ ]"
    if "etaM" in xQuant:
        xlim = [0.0, 2.7]  # explore

    binSize = 0.2
    markersize = 0.0  # 4.0, or 0.0 to disable
    malpha = 0.2
    percs = [16, 84]
    zStr = str(sP.snap) if len(redshifts) == 1 else "z=" + "-".join(["%.1f" % z for z in redshifts])

    def _plotHelper(percIndsPlot, radIndsPlot, saveName=None, pdf=None, ylimLoc=None, stat="median", addModelTNG=False):
        """Plot a radii series, vcut series, or both."""
        if len(redshifts) > 1:
            assert len(radIndsPlot) == 1  # otherwise needs generalization

        # plot setup
        fig, ax = plt.subplots()

        if ylimLoc is None:
            ylimLoc = ylim

        ax.set_xlim(xlim)
        ax.set_ylim(ylimLoc)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if config is not None and "xlabel" in config:
            ax.set_xlabel(config["xlabel"])
        if config is not None and "ylabel" in config:
            ax.set_ylabel(config["ylabel"])

        labels_sec = []
        color_ind = 0

        # TNG minimum velocity band
        if ("mstar" in xQuant or "mhalo" in xQuant) and not v200norm:
            minVel = 350.0 if not ylog else np.log10(350.0)
            minVelTextY = 370.0 if not ylog else 2.58
            ax.fill_between(xlim, [0, 0], [minVel, minVel], color="#cccccc", alpha=0.05)
            ax.plot(xlim, [minVel, minVel], "-", color="#cccccc", alpha=0.5)
            label = r"TNG v$_{\rm wind,min}$ = 350 km/s"
            ax.text(xlim[0] + (xlim[1] - xlim[0]) * 0.04, minVelTextY, label, color="black", alpha=0.6)

        if addModelTNG:
            # loop over multiple-redshifts if requested
            redshiftsToDo = redshifts if redshifts[0] is not None else [sP.redshift]
            for k, redshift in enumerate(redshiftsToDo):
                sP.setRedshift(redshift)

                # load velocity (of TNG model at injection) analysis
                GFM_windvel_mean, _, _, _ = sP.simSubhaloQuantity("wind_vel")
                if ylog:
                    GFM_windvel_mean = logZeroNaN(GFM_windvel_mean)

                # load x-axis property
                GFM_xquant, _, fit_xlim, takeLog = sP.simSubhaloQuantity(xQuant)
                if takeLog:
                    GFM_xquant = logZeroNaN(GFM_xquant)

                assert GFM_windvel_mean.shape == GFM_xquant.shape

                # median
                with np.errstate(invalid="ignore"):
                    w_fit = np.where(GFM_windvel_mean > 0)

                x_fit = GFM_xquant[w_fit]
                y_fit = GFM_windvel_mean[w_fit]

                xm, ym, sm, pm = running_median(x_fit, y_fit, binSize=binSize, percs=[16, 84])

                if xm.size > sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)
                    pm = savgol_filter(pm, sKn, sKo, axis=1)

                alpha = 0.1 if len(redshifts) == 1 else 0.05
                if k == 0:
                    ax.fill_between(xm, pm[0, :], pm[-1, :], color="black", interpolate=True, alpha=alpha)

                xm2 = np.linspace(xm.min(), xm.max(), 100)
                ym2 = interp1d(xm, ym, kind="cubic", fill_value="extrapolate")(xm2)

                label = "TNG Model (at Injection)" if k == 0 else ""
                # if len(redshifts) > 1: label = 'TNG model (z=%.1f)' % redshift
                if len(redshifts) > 1:  # special case labeling
                    ax.text(9.66, 720.0, "$z$ = 6", color="#888888", rotation=70.0)
                    ax.text(9.94, 730.0, "$z$ < 4", color="#888888", rotation=65.0)

                alpha = 0.6 if len(redshifts) == 1 else 0.5
                ax.plot(xm2, ym2, linestyle=linestyles[k], color="black", alpha=alpha, label=label)

        # loop over redshifts
        data_z = []

        for k, redshift in enumerate(redshifts):
            # get local data
            mdot, xx, binConfig, numBins, vals, percs = data[k]

            # loop over radii or vcut selections
            for i, rad_ind in enumerate(radIndsPlot):
                if (len(percIndsPlot) > 1 and len(radIndsPlot) > 1) or len(redshifts) > 1:
                    color_ind += 1  # one color per rad, if cycling over both

                # local data (outflow rates in this radial bin)
                if rad_ind < mdot.shape[1]:
                    mdot_local = mdot[:, rad_ind]
                else:
                    mdot_local = np.sum(mdot, axis=1)  # 'all'

                for j, perc_ind in enumerate(percIndsPlot):
                    # local data (velocities in this radial/perc bin)
                    yy = np.squeeze(vals[:, rad_ind, perc_ind]).copy()  # zero flux -> nan, skipped in median

                    # decision on mdot==0 (or etaM==0) systems: include (in medians/means and percentiles) or exclude?
                    if 0:
                        w_zero = np.where(yy == 0.0)
                        yy[w_zero] = np.nan

                    # mdot < threshold: exclude
                    w_below = np.where(mdot_local < mdotThreshValue)
                    assert mdotThreshValue == 0.0  # otherwise have a mismatch of sub_ids subset here, verify size match
                    yy[w_below] = np.nan

                    if ylog:
                        yy = logZeroNaN(yy)

                    # label and color
                    labelFixed = None
                    if rad_ind == numBins["rad"]:
                        radMidPoint = "all"
                    else:
                        radMidPoint = "%3d kpc" % (0.5 * (binConfig["rad"][rad_ind] + binConfig["rad"][rad_ind + 1]))

                    if len(percIndsPlot) == 1:
                        label = "r = %s" % radMidPoint
                        labelFixed = r"v$_{\rm out,%d}$" % percs[perc_ind]
                    if len(radIndsPlot) == 1:
                        label = r"v$_{\rm out,%d}$" % percs[perc_ind]
                        labelFixed = "r = %s" % radMidPoint
                    if len(percIndsPlot) > 1 and (len(radIndsPlot) > 1 or len(redshifts) > 1):
                        label = "r = %s" % radMidPoint  # primary label radius, by color
                        labels_sec.append(r"v$_{\rm out,%d}$" % percs[perc_ind])  # second legend: vcut by ls
                        if j > 0:
                            label = ""
                    if len(redshifts) == 1:
                        if labelFixed is None or "mstar" not in xQuant:
                            labelFixed = "z = %.1f" % sP.redshift
                        else:
                            labelFixed += ", z = %.1f" % sP.redshift

                    if len(redshifts) > 1:
                        label = "z = %.1f" % redshift
                    if len(redshifts) > 1 and j > 0:
                        label = ""  # move percs to separate labels

                    if (len(percIndsPlot) == 1 or len(radIndsPlot) == 1) and len(redshifts) == 1:
                        color_ind += 1

                    # symbols for each system
                    if markersize > 0:
                        ax.plot(
                            xx, yy, "s", color=colors[color_ind], markersize=markersize, alpha=malpha, rasterized=True
                        )

                        # mark those at absolute zero just above the bottom of the y-axis
                        off = 10
                        w_zero = np.where(np.isnan(yy))
                        yy_zero = np.random.uniform(size=len(w_zero[0]), low=ylim[0] + off / 2, high=ylim[0] + off)
                        ax.plot(
                            xx[w_zero], yy_zero, "s", alpha=malpha / 2, markersize=markersize, color=colors[color_ind]
                        )

                    # median line and 1sigma band
                    minNum = 2 if "etaM" in xQuant else 5  # for xQuants = mstar, SFR, Lbol, ...
                    if redshift is not None and redshift > 7.0:
                        minNum = 2
                    xm, ym, sm, pm = running_median(
                        xx, yy, binSize=binSize, percs=percs, mean=(stat == "mean"), minNumPerBin=minNum
                    )

                    if xm.size > sKn:
                        extra = 0
                        ym = savgol_filter(ym, sKn + 2 + extra, sKo + 2)
                        sm = savgol_filter(sm, sKn + 2 + extra, sKo + 2)
                        pm = savgol_filter(pm, sKn + 2 + extra, sKo + 2, axis=1)

                    lsInd = j if len(percIndsPlot) < 4 else i
                    if markersize > 0:
                        lsInd = 0

                    # xm2 = np.linspace(xm.min(), xm.max(), 100)
                    # ym2 = interp1d(xm, ym, kind='cubic', fill_value='extrapolate')(xm2)

                    if xm[0] > xlim[0]:
                        xm[0] = xlim[0]  # visual

                    (l,) = ax.plot(xm, ym, linestyles[lsInd], color=colors[color_ind], label=label)

                    data_z.append(
                        {"redshift": redshift, "vperc": percs[perc_ind], "rad": radMidPoint, "xm": xm, "ym": ym}
                    )

                    # shade percentile band?
                    if i == j or markersize > 0 or "mstar" not in xQuant:
                        y_down = pm[0, :]  # np.array(ym[:-1]) - sm[:-1]
                        y_up = pm[-1, :]  # np.array(ym[:-1]) + sm[:-1]

                        # repairs
                        w = np.where(np.isnan(y_up))[0]
                        if len(w) and len(w) < len(y_up) and w.max() + 1 < y_up.size:
                            lastGoodInd = np.max(w) + 1
                            lastGoodVal = y_up[lastGoodInd] - ym[:][lastGoodInd]
                            y_up[w] = ym[:][w] + lastGoodVal

                        w = np.where(np.isnan(y_down) & np.isfinite(ym[:]))
                        y_down[w] = ylimLoc[0]  # off bottom

                        # plot bottom
                        ax.fill_between(xm[:], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

        # special plotting behavior (including observational data sets)
        if f_post is not None:
            f_post(ax, locals())

        # legends and finish plot
        legParams = {"frameon": 1, "framealpha": 0.9, "borderpad": 0.2, "fancybox": False}

        loc1 = "lower right" if (config is None or "loc1" not in config) else config["loc1"]
        if len(percIndsPlot) == 1 or len(radIndsPlot) == 1:
            line = plt.Line2D([0], [0], color="white", marker="", lw=0.0)
            legend2 = ax.legend([line], [labelFixed], loc=loc1)
            if loc1 is not None:
                ax.add_artist(legend2)

        if len(percIndsPlot) > 1 and len(radIndsPlot) > 1:
            lines = [
                plt.Line2D([0], [0], color="black", marker="", linestyle=linestyles[j])
                for j in range(len(percIndsPlot))
            ]
            legend2 = ax.legend(lines, labels_sec, loc=loc1)
            ax.add_artist(legend2)

        handles, labels = ax.get_legend_handles_labels()
        if len(redshifts) > 1 and len(percIndsPlot) > 1:
            handles += [
                plt.Line2D([0], [0], color="black", marker="", linestyle=linestyles[j])
                for j in range(len(percIndsPlot))
            ]
            labels += labels_sec
        loc2 = "upper right" if (config is None or "loc2" not in config) else config["loc2"]
        if "sfr" in xQuant:
            ax.legend(handles, labels, loc=loc2, **legParams)
        else:
            ax.legend(handles, labels, loc=loc2)

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load outflow rates and outflow velocities (total)
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)

        mdot, mstar_30pkpc_log, sub_ids, binConfig, numBins, _ = radialMassFluxes(sP, scope, "Gas", massField=massField)
        mdot = mdot[:, :, mdotThreshVcutInd]

        projStr = "2DProj" if proj2D else ""
        acField = "Subhalo_OutflowVelocity%s_%s" % (projStr, scope)
        if massField != "Masses":
            acField = "Subhalo_OutflowVelocity%s_%s_%s" % (projStr, massField, scope)
        if v200norm:
            acField += "_v200norm"

        ac = sP.auxCat(acField)

        vals = ac[acField]
        percs = ac[acField + "_attrs"]["percs"]

        # restrict included v_out values to a minimum M*? E.g. if plotting against something other than M* on the x-axis
        if config is not None and "minMstar" in config:
            w = np.where(mstar_30pkpc_log < config["minMstar"])
            vals[w] = np.nan

        # load x-axis values, stellar mass or other?
        xvals, xlabel2, xlim2, takeLog = sP.simSubhaloQuantity(xQuant)
        xvals = xvals[sub_ids]
        if takeLog:
            xvals = logZeroNaN(xvals)

        if xlabel is None:
            xlabel = xlabel2  # use suggestions if not hard-coded above
        if xlim is None:
            xlim = xlim2

        # save one data-list per redshift
        data.append([mdot, xvals, binConfig, numBins, vals, percs])

    allRadInd = vals.shape[1] - 1  # last bin is not a radial bin, but all radii combined

    # one specific plot requested? make now and exit
    if config is not None:
        v200Str = "_v200norm" if v200norm else ""
        saveName = "%s%s_%s_%s_%s_nr%d_np%d_%s_%s.pdf" % (
            saveBase,
            ptStr,
            xQuant,
            sP.simName,
            zStr,
            len(config["radInds"]),
            len(config["percInds"]),
            config["stat"],
            v200Str,
        )
        if "saveName" in config:
            saveName = config["saveName"]
        if "ylim" not in config:
            config["ylim"] = None
        if "xlim" in config:
            xlim = config["xlim"]
        if "addModelTNG" not in config:
            config["addModelTNG"] = False
        if "markersize" in config:
            markersize = config["markersize"]
        if "binSize" in config:
            binSize = config["binSize"]
        if "percs" in config:
            percs = config["percs"]

        _plotHelper(
            percIndsPlot=config["percInds"],
            radIndsPlot=config["radInds"],
            saveName=saveName,
            ylimLoc=config["ylim"],
            stat=config["stat"],
            addModelTNG=config["addModelTNG"],
        )
        return

    # plot
    for stat in ["mean"]:  # ['mean','median']:
        # (A) plot for a given perc, at many radii
        radInds = [1, 3, 4, 5, 6, 7, allRadInd]

        pdf = PdfPages("%s%s_%s_A_%s_%s_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, zStr, stat))
        for perc_ind in range(percs.size):
            _plotHelper(percIndsPlot=[perc_ind], radIndsPlot=radInds, pdf=pdf, stat=stat)
        pdf.close()

        # (B) plot for a given radii, at many percs
        percInds = [0, 1, 2, 3, 4, 5]

        pdf = PdfPages("%s%s_%s_B_%s_%s_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, zStr, stat))
        for rad_ind in range(numBins["rad"] + 1):  # last one is 'all'
            _plotHelper(percIndsPlot=percInds, radIndsPlot=[rad_ind], pdf=pdf, stat=stat)
        pdf.close()

        # (C) single-panel combination of both radial and perc variations
        percIndsPlot = [1, 2, 4]
        radIndsPlot = [1, 2, 13]
        ylimLoc = [0, 800] if not ylog else [1.5, 3.0]

        saveName = "%s%s_%s_C_%s_%s_%s.pdf" % (saveBase, ptStr, xQuant, sP.simName, zStr, stat)
        _plotHelper(percIndsPlot, radIndsPlot, saveName, ylimLoc=ylimLoc, stat=stat)


def outflowRatesStacked(sP_in, quant, mStarBins, redshifts=(None,), config=None, inflow=False, f_post=None):
    """Plot radial mass flux as a function of a histogrammed quantity, for single or stacked galaxies.

    Binning in stellar mass. Optionally at multiple redshifts.
    """
    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant)  # copy

    # config
    scope = "SubfindWithFuzz"  # or 'Global'
    ptType = "Gas"

    # plot config
    ylim = [-3.0, 2.0] if (config is None or "ylim" not in config) else config["ylim"]
    vcuts = [0, 1, 2, 3, 4] if quant != "vrad" else [None]
    zStr = str(sP.snap) if len(redshifts) == 1 else "z=" + "-".join(["%.1f" % z for z in redshifts])

    limits = {
        "temp": [2.9, 8.1],
        "temp_sfcold": [2.9, 8.1],
        "z_solar": [-3.0, 1.0],
        "numdens": [-5.0, 2.0],
        "vrad": [0, 3000],
        "theta": [-np.pi, np.pi],
    }

    if len(redshifts) > 1:
        # multi-z plots restricted to smaller M* bins, modify limits
        limits = {
            "temp": [3.0, 8.0],
            "temp_sfcold": [3.0, 8.0],
            "z_solar": [-2.0, 1.0],
            "numdens": [-5.0, 2.0],
            "vrad": [0, 1800],
            "theta": [-np.pi, np.pi],
        }

    def _plotHelper(vcut_ind, rad_ind, quant, mStarBins=None, stat="mean", saveName=None, pdf=None):
        """Plot a radii series, vcut series, or both."""
        # plot setup
        fig, ax = plt.subplots()

        ax.set_xlim(limits[quant])
        ax.set_ylim(ylim)

        ylabel = r"%s Outflow Rate [ log M$_{\rm sun}$ / yr ]" % ptType
        if inflow:
            ylabel = ylabel.replace("Outflow", "Inflow")
        ax.set_xlabel(labels[quant])
        ax.set_ylabel(ylabel)

        if quant == "theta":  # and (config is None or 'sterNorm' not in config):
            # special x-axis labels for angle theta
            ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
            ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$ (minor axis)", r"$0$", r"$+\pi/2$ (minor axis)", r"$+\pi$"])
            ax.plot([-np.pi / 2, -np.pi / 2], ylim, "-", color="#aaaaaa", alpha=0.3)
            ax.plot([+np.pi / 2, +np.pi / 2], ylim, "-", color="#aaaaaa", alpha=0.3)

        # loop over redshifts
        txt = []

        for j, redshift in enumerate(redshifts):
            # get local data
            mdot, mstar, subids, binConfig, numBins, vcut_vals = data[j]

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):
                # local data
                w = np.where((mstar > mStarBin[0]) & (mstar <= mStarBin[1]))
                # print(mStarBin, ' number of galaxies: ',len(w[0]))

                if vcut_ind is not None:
                    mdot_local = np.squeeze(mdot[w, rad_ind, vcut_ind, :]).copy()
                else:
                    mdot_local = np.squeeze(mdot[w, rad_ind, :]).copy()  # plot: mdot vs. vrad directly

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if 0:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                # normalize by angular units: convert from [Msun/yr] to [Msun/yr/ster]
                if config and "sterNorm" in config and config["sterNorm"]:
                    ax.set_ylabel(ylabel.replace("/ yr", "/ yr / ster"))
                    ax.set_xlim([0, np.pi / 2])
                    ax.set_yscale("symlog")

                    ster_per_bin = 4 * np.pi / mdot_local.shape[-1]
                    mdot_local /= ster_per_bin

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # avoid RuntimeWarning: Mean of empty slice (single galaxies with only zero values)
                    if stat == "median":
                        yy = np.nanmedian(mdot_local, axis=0)  # median on subhalo axis
                    if stat == "mean":
                        yy = np.nanmean(mdot_local, axis=0)  # mean

                    # median line and 1sigma band
                    pm = np.nanpercentile(mdot_local, [16, 84], axis=0, interpolation="linear")
                    pm = logZeroNaN(pm)
                    sm = np.nanstd(logZeroNaN(mdot_local), axis=0)

                if not isinstance(yy, np.ndarray):
                    continue  # single number

                if config and ("sterNorm" not in config or (not config["sterNorm"])):
                    yy = logZeroNaN(yy)  # zero flux -> nan

                # label and color
                xx = 0.5 * (binConfig[quant][:-1] + binConfig[quant][1:])

                radMidPoint = 0.5 * (binConfig["rad"][rad_ind] + binConfig["rad"][rad_ind + 1])
                mStarMidPoint = 0.5 * (mStarBin[0] + mStarBin[1])

                labelFixed = "r = %3d kpc" % radMidPoint
                if vcut_ind is not None:
                    labelFixed += r", v$_{\rm rad}$ > %3d km/s" % vcut_vals[vcut_ind]
                if len(redshifts) == 1:
                    labelFixed += r", z = %.1f" % sP.redshift

                label = r"M$^\star$ = %.1f" % mStarMidPoint if j == 0 else ""  # label M* only once

                # yy = savgol_filter(yy,sKn,sKo)
                if pm.ndim > 1:
                    pm = savgol_filter(pm, sKn, sKo, axis=1)
                sm = savgol_filter(sm, sKn, sKo)

                # l, = ax.plot(xm[:-1], ym[:-1], linestyles[i], color=c, label=label)
                (l,) = ax.plot(xx, yy, linestyle=linestyles[j], color=colors[i], label=label)

                txt.append({"vout": xx, "outflowrate": yy, "redshift": redshift, "mstar": mStarMidPoint})

                if j == 0 and (i == 0 or i == len(mStarBins) - 1):
                    # w = np.where( np.isfinite(pm[0,:]) & np.isfinite(pm[-1,:]) )[0]
                    # ax.fill_between(xx[w], pm[0,w], pm[-1,w], color=l.get_color(), alpha=0.05)
                    w = np.where(np.isfinite(yy))
                    ax.fill_between(xx[w], yy[w] - sm[w], yy[w] + sm[w], color=l.get_color(), alpha=0.05)

                if 0:
                    # plot some vertical line markers (Fig 8)
                    yy_sum = np.nansum(10.0**yy)
                    yy_cumsum = np.nancumsum(10.0**yy / yy_sum)
                    facs = [0.5, 0.95, 0.99]
                    for k, fac in enumerate(facs):
                        w = np.min(np.where(yy_cumsum >= fac)[0])
                        print(fac, xx[w])
                        ls = ["-", "--", ":"][k]
                        ymax = -2.8 if k == 0 else -2.85
                        ax.plot([xx[w], xx[w]], [-3.0, ymax], color=l.get_color(), lw=lw - 0.5, linestyle=ls, alpha=0.5)

        # special plotting behavior (including observational data sets)
        if f_post is not None:
            f_post(ax, locals())

        # legends and finish plot
        if len(redshifts) > 1:
            sExtra = []
            lExtra = []

            for j, redshift in enumerate(redshifts):
                sExtra += [plt.Line2D([0], [0], color="black", linestyle=linestyles[j], marker="")]
                lExtra += ["z = %.1f" % redshift]

            legend2 = ax.legend(sExtra, lExtra, loc="lower right")
            ax.add_artist(legend2)

        line = plt.Line2D([0], [0], color="white", marker="", lw=0.0)
        loc = "upper left" if (len(redshifts) > 1 or quant == "vrad" or quant == "numdens") else "lower right"
        if quant in ["temp", "temp_sfcold"]:
            loc = "upper right"
        legend3 = ax.legend([line], [labelFixed], handlelength=-0.5, loc=loc)
        ax.add_artist(legend3)  # "r = X kpc" or "r = X kpc, z = Y"

        ax.legend(loc="upper right" if quant not in ["temp", "temp_sfcold"] else "upper left")  # M* bins

        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)
        data.append(radialMassFluxes(sP, scope, ptType, thirdQuant=quant, inflow=inflow))

    if config is not None:
        saveName = "outflowRate_%s_%s_mstar_%s_%s_%s.pdf" % (ptType, quant, sP.simName, zStr, config["stat"])
        if inflow:
            saveName = saveName.replace("outflowRate", "inflowRate")
        if "vcutInd" not in config:
            config["vcutInd"] = None  # quant == vrad
        _plotHelper(config["vcutInd"], config["radInd"], quant, mStarBins, config["stat"], saveName=saveName)
        return

    for stat in ["mean"]:  # ['mean','median']:
        print(quant, stat)

        # (A) vs quant, booklet across rad and vcut variations
        pdf = PdfPages("outflowRate_A_%s_%s_mstar_%s_%s_%s.pdf" % (ptType, quant, sP.simName, zStr, stat))

        for radInd in [1, 3, 4, 5, 6, 7]:
            for vcutInd in vcuts:
                _plotHelper(vcutInd, radInd, quant, mStarBins, stat, pdf=pdf)

        pdf.close()


def outflowRates2DStacked(
    sP_in,
    xAxis,
    yAxis,
    mStarBins,
    redshifts=(None,),
    clims=([-3.0, 2.0],),
    config=None,
    eta=False,
    rawMass=False,
    rawDens=False,
    discreteColors=False,
    contours=None,
    v200norm=False,
):
    """Explore radial mass flux data, 2D panels where color indicates Mdot_out.

    Give clims as a list, one per mStarBin, or if just one element, use the same for all bins.
    If config is None, generate many exploration plots, otherwise just create the single desired plot.
    If eta is True, plot always mass-loadings instead of mass-outflow rates.
    if rawMass is True, plot always total mass, instead of mass-outflow rates.
    if rawDens i True, plot always total mass density, instead of mass-outflow rates.
    If discreteColors is True, split the otherwise continuous colormap into discrete segments.
    """
    sP = simParams(res=sP_in.res, run=sP_in.run, redshift=sP_in.redshift, variant=sP_in.variant)  # copy

    # config
    scope = "SubfindWithFuzz"  # or 'Global'
    ptType = "Gas"
    cStr = "_contour" if contours is not None else ""

    if eta:
        cbarlabel = r"%s Mass Loading $\eta = \dot{M}_{\rm w} / \dot{M}_\star$ [ log ]" % ptType
        cbarlabel2 = r"%s Mass Loading (Inflow) [ log ]" % ptType
        saveBase = "massLoading2D"
        contourlabel = r"log $\eta$"
    else:
        cbarlabel = r"%s Outflow Rate [ log M$_{\rm sun}$ / yr ]" % ptType
        cbarlabel2 = r"%s Inflow Rate [ log M$_{\rm sun}$ / yr ]" % ptType
        saveBase = "outflowRate2D"
        contourlabel = r"log $\dot{M}_{\rm out}$"
    if rawMass:
        assert not eta and not rawDens
        cbarlabel = r"%s Mass [ log M$_{\rm sun}$ ]" % ptType
        cbarlabel2 = r"%s Mass [ log ]" % ptType
        saveBase = "mass2D"
        contourlabel = r"log $M_{\rm %s}$" % ptType
    if rawDens:
        assert not eta and not rawMass
        cbarlabel = r"%s $\delta \rho / <\rho>$ [ log ]" % ptType
        cbarlabel2 = r"%s $\delta \rho / <\rho>$ [ log ]" % ptType
        saveBase = "densityRelative2D"
        contourlabel = r"log $\delta \rho / <\rho>$"

    saveBase += f"_{ptType}_{xAxis}-{yAxis}"

    if len(clims) == 1:
        clims = [clims[0]] * len(mStarBins)  # one for each
    assert yAxis != "rad"  # keep on x-axis if wanted
    assert xAxis != "vcut"  # keep on y-axis if wanted

    # plot config
    limits = {
        "rad": None,  # discrete labels (small number of bins)
        "vrad": False,  # fill from binConfig
        "vcut": None,  # discrete labels (small number of bins)
        "temp": False,  # fill from binConfig
        "temp_sfcold": False,  # fill from binConfig
        "z_solar": False,  # fill from binConfig
        "numdens": False,  # fill from binConfig
        "theta": [-np.pi, np.pi],
    }  # always fixed

    def _plotHelper(xAxis, yAxis, mStarBins=None, stat="mean", saveName=None, vcut_ind=None, rad_ind=None, pdf=None):
        """Plot a number of 2D histogram panels. mdot_2d should shape: [subhalo_ids, xAxis_quant, yAxis_quant]."""
        # replicate vcut/rad indices into lists, one per mass bin, if not already
        if not isinstance(vcut_ind, list):
            vcut_ind = [vcut_ind] * len(mStarBins)
        if not isinstance(rad_ind, list):
            rad_ind = [rad_ind] * len(mStarBins)

        if contours is not None:
            # only 1 panel for all M*/redshift variations, make panel now
            fig, ax = plt.subplots()

            lines = []
            labels1 = []
        else:
            # non-contour plot setup: multi-panel
            nRows = int(np.floor(np.sqrt(len(mStarBins))))
            nCols = int(np.ceil(len(mStarBins) / nRows))
            nRows, nCols = nCols, nRows

            fig = plt.figure(figsize=[figsize[0] * nCols, figsize[1] * nRows])

        # loop over redshifts
        for j, redshift in enumerate(redshifts):
            # get local data
            mdot_in, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed = data[j]

            # axes are always (rad,vcut), i.e. we never actually slice mdot_in
            if all(ind is None for ind in rad_ind + vcut_ind):
                mdot_2d = mdot_in.copy()

            # loop over stellar mass bins and stack
            for i, mStarBin in enumerate(mStarBins):
                # create local mdot, resize if needed (final dimensions are [subhalos, xaxis_quant, yaxis_quant])
                if vcut_ind[i] is not None and rad_ind[i] is None:
                    mdot_2d = np.squeeze(mdot_in[:, :, vcut_ind[i], :]).copy()
                if rad_ind[i] is not None and vcut_ind[i] is None:
                    mdot_2d = np.squeeze(mdot_in[:, rad_ind[i], :, :]).copy()
                    if yAxis == "vcut":
                        mdot_2d = np.swapaxes(mdot_2d, 1, 2)  # put vcut as last (y) axis
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    mdot_2d = np.squeeze(mdot_in[:, rad_ind[i], vcut_ind[i], :, :]).copy()

                # start panel
                if contours is None:
                    ax = fig.add_subplot(nRows, nCols, i + 1)

                xlim = limits[xAxis] if limits[xAxis] is not None else [0, mdot_2d.shape[1]]
                ylim = limits[yAxis] if limits[yAxis] is not None else [0, mdot_2d.shape[2]]
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                if mdot_2d.shape[1] == binConfig[xAxis].size and xAxis == "rad":
                    # remove new r=rall bin at the end
                    mdot_2d = mdot_2d[:, :-1, :]

                assert mdot_2d.shape[1] == binConfig[xAxis].size - 1
                assert mdot_2d.shape[2] == binConfig[yAxis].size - 1 if yAxis != "vcut" else binConfig[yAxis].size

                ax.set_xlabel(labels[xAxis])
                ax.set_ylabel(labels[yAxis])
                if v200norm:
                    ax.set_ylabel(labels[yAxis].replace("km/s", r"v$_{\rm 200}$"))

                # local data
                w = np.where((mstar > mStarBin[0]) & (mstar <= mStarBin[1]))
                mdot_local = np.squeeze(mdot_2d[w, :, :]).copy()
                print("z=", redshift, " M* bin:", mStarBin, " number of galaxies: ", len(w[0]))

                if eta:
                    # normalize each included system by its smoothed SFR (expand dimensionality for broadcasting)
                    sfr_norm = sfr_smoothed[w[0], None, None]
                    w = np.where(sfr_norm > 0.0)
                    mdot_local[w, :, :] /= sfr_norm[w, :, :]

                    w = np.where(sfr_norm == 0.0)
                    mdot_local[w, :, :] = np.nan

                # decision on mdot==0 systems: include (in medians/means and percentiles) or exclude?
                if 0:
                    w_zero = np.where(mdot_local == 0.0)
                    mdot_local[w_zero] = np.nan

                if stat == "median":
                    h2d = np.nanmedian(mdot_local, axis=0)  # median on subhalo axis
                if stat == "mean":
                    h2d = np.nanmean(mdot_local, axis=0)  # mean

                if rawDens:
                    # relative to azimuthal average in each radial bin: delta_rho/<rho>
                    radial_means = np.nanmean(h2d, axis=1)
                    h2d /= radial_means[:, np.newaxis]

                # handle negative values (inflow) by separating the matrix into positive and negative components
                h2d_pos = h2d.copy()
                h2d_neg = h2d.copy()
                h2d_pos[np.where(h2d < 0.0)] = np.nan
                h2d_neg[np.where(h2d >= 0.0)] = np.nan

                h2d = logZeroNaN(h2d)  # zero flux -> nan
                h2d_pos = logZeroNaN(h2d_pos)
                h2d_neg = logZeroNaN(-h2d_neg)

                # h2d = sgolay2d(h2d,sKn,sKo) # smoothing

                # set NaN/blank to minimum color
                with np.errstate(invalid="ignore"):
                    w_neg_clip = np.where(h2d_neg < (clims[i][0] + (clims[i][1] - clims[i][0]) * 0.1))
                    # 10% clip near bottom (black) edge of colormap, let these pixels stay as pos background color
                    h2d_neg[w_neg_clip] = np.nan

                w = np.where(np.isnan(h2d_pos) & np.isnan(h2d_neg))
                h2d_pos[w] = clims[i][0]
                # h2d_neg[w] = clims[i][0] # let h2d_pos assign 'background' color for all nan pixels

                # set special x/y axis labels? on a small, discrete number of bins
                if limits[xAxis] is None:
                    xx = list(0.5 * (binConfig[xAxis][:-1] + binConfig[xAxis][1:]))
                    if np.isinf(xx[-1]):
                        # last bin was some [finite,np.inf] range
                        xx[-1] = ">%s" % binConfig[xAxis][-2]

                    if np.isinf(xx[0]) or binConfig[xAxis][0] == 0.0:
                        # first bin was some [-np.inf,finite] or [0,finite] range (midpoint means little)
                        xx[0] = "<%s" % binConfig[xAxis][1]

                    xticklabels = []
                    for xval in xx:
                        xticklabels.append(xval if isinstance(xval, str) else "%d" % xval)

                    ax.set_xticks(np.arange(mdot_2d.shape[1]) + 0.5)
                    ax.set_xticklabels(xticklabels)
                    assert len(xx) == mdot_2d.shape[1]

                if limits[yAxis] is None:
                    yy = (
                        list(0.5 * (binConfig[yAxis][:-1] + binConfig[yAxis][1:]))
                        if yAxis != "vcut"
                        else list(binConfig[yAxis])
                    )

                    yticklabels = []
                    for yval in yy:
                        curStr = "%d" % yval if not v200norm else "%.1f" % yval
                        yticklabels.append(yval if isinstance(yval, str) else curStr)

                    ax.set_yticks(np.arange(mdot_2d.shape[2]) + 0.5)
                    ax.set_yticklabels(yticklabels)
                    assert len(yy) == mdot_2d.shape[2]

                # label
                mStarMidPoint = 0.5 * (mStarBin[0] + mStarBin[1])
                label1 = r"M$^\star$ = %.1f" % mStarMidPoint
                label2 = None

                if len(mStarBins) == 1 and len(redshifts) > 1:
                    label1 = "z = %.1f" % redshift

                if j > 0:
                    label1 = ""  # only label on first redshift

                if vcut_ind[i] is not None and np.isfinite(vcut_vals[vcut_ind[i]]):
                    label2 = r"v$_{\rm rad}$ > %3d km/s" % vcut_vals[vcut_ind[i]]
                    if v200norm:
                        label2 = r"v$_{\rm rad}$ > %.1f v$_{\rm 200}$" % vcut_vals[vcut_ind[i]]
                if rad_ind[i] is not None:
                    radMidPoint = 0.5 * (binConfig["rad"][rad_ind[i]] + binConfig["rad"][rad_ind[i] + 1])
                    label2 = r"r = %3d kpc" % radMidPoint
                if vcut_ind[i] is not None and rad_ind[i] is not None:
                    radMidPoint = 0.5 * (binConfig["rad"][rad_ind[i]] + binConfig["rad"][rad_ind[i] + 1])
                    label2 = r"r = %3d kpc, v$_{\rm rad}$ > %3d km/s" % (radMidPoint, vcut_vals[vcut_ind[i]])
                    if v200norm:
                        label2 = r"r = %3d kpc, v$_{\rm rad}$ > %.1f v$_{\rm 200}$" % (
                            radMidPoint,
                            vcut_vals[vcut_ind[i]],
                        )

                # plot: positive and negative components separately
                norm = Normalize(vmin=clims[i][0], vmax=clims[i][1])

                numColors = None  # continuous cmap
                if discreteColors:
                    numColors = (clims[i][1] - clims[i][0]) * 2  # discrete for each 0.5 interval
                cmap_pos = loadColorTable("viridis", numColors=numColors)
                cmap_neg = loadColorTable("inferno", numColors=numColors)

                imOpts = {
                    "extent": [xlim[0], xlim[1], ylim[0], ylim[1]],
                    "origin": "lower",
                    "interpolation": "nearest",
                    "aspect": "auto",
                }

                if contours is None:
                    # 2D histogram image
                    im_neg = plt.imshow(h2d_neg.T, cmap=cmap_neg, norm=norm, **imOpts)
                    im_pos = plt.imshow(h2d_pos.T, cmap=cmap_pos, norm=norm, **imOpts)

                    # set background color inside plot to lowest value instead of white, to prevent boundary artifacts
                    ax.set_facecolor(cmap_pos(0.0))
                else:
                    # 2D contour: first resample, increasing resolution
                    XX = np.linspace(xlim[0], xlim[1], mdot_2d.shape[1])
                    YY = np.linspace(ylim[0], ylim[1], mdot_2d.shape[2])

                    # origin space
                    grid_x, grid_y = np.meshgrid(XX, YY, indexing="ij")
                    grid_xy = np.zeros((grid_x.size, 2), dtype=grid_x.dtype)
                    grid_xy[:, 0] = grid_x.reshape(grid_x.shape[0] * grid_x.shape[1])  # flatten
                    grid_xy[:, 1] = grid_y.reshape(grid_y.shape[0] * grid_y.shape[1])  # flatten

                    grid_z = h2d_pos.copy().reshape(h2d_pos.shape[0] * h2d_pos.shape[1])  # flatten

                    # target space
                    nn = 50

                    # only above some minimum size (always true for now)
                    if h2d_pos.shape[0] < nn or h2d_pos.shape[1] < nn:
                        # remove any NaN's (for 2d sg)
                        w = np.where(np.isnan(grid_z))
                        grid_z[w] = clims[i][0]

                        # only if 2D histogram is actually small(er) than this
                        XX_out = np.linspace(xlim[0], xlim[1], nn)
                        YY_out = np.linspace(ylim[0], ylim[1], nn)
                        grid_out_x, grid_out_y = np.meshgrid(XX_out, YY_out, indexing="ij")

                        grid_out = np.zeros((grid_out_x.size, 2), dtype=grid_out_x.dtype)
                        grid_out[:, 0] = grid_out_x.reshape(nn * nn)  # flatten
                        grid_out[:, 1] = grid_out_y.reshape(nn * nn)  # flatten

                        # resample and smooth
                        grid_z_out = griddata(grid_xy, grid_z, grid_out, method="cubic").reshape(nn, nn)

                        if yAxis == "vrad":
                            # vrad crosses the zero boundary separating inflow/outflow, do not smooth across
                            min_pos_ind = np.where(YY_out > 0.0)[0].min()
                            grid_z_out[:, min_pos_ind:] = sgolay2d(grid_z_out[:, min_pos_ind:], sKn * 3, sKo)
                        else:
                            grid_z_out = sgolay2d(grid_z_out, sKn * 3, sKo)

                    # render contour (different linestyles for different contour values)
                    color = cmap_pos(float(i) / len(mStarBins))
                    if j == 0:
                        lines.append(plt.Line2D([0], [0], color=color, marker="", lw=lw))
                    c_ls = linestyles if len(redshifts) == 1 else linestyles[j]  # linestyle per redshift
                    im_pos = ax.contour(XX_out, YY_out, grid_z_out.T, contours, ls=c_ls, linewidths=lw, colors=[color])
                    # im_neg

                # special x-axis labels for angle theta
                if yAxis == "theta":
                    ax.set_yticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
                    ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
                    ax.plot(xlim, [-np.pi / 2, -np.pi / 2], "-", color="#aaaaaa", alpha=0.3)
                    ax.plot(xlim, [+np.pi / 2, +np.pi / 2], "-", color="#aaaaaa", alpha=0.3)
                if xAxis == "theta":
                    ax.set_xticks([-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi])
                    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$+\pi/2$", r"$+\pi$"])
                    ax.plot([-np.pi / 2, -np.pi / 2], ylim, "-", color="#aaaaaa", alpha=0.3)
                    ax.plot([+np.pi / 2, +np.pi / 2], ylim, "-", color="#aaaaaa", alpha=0.3)

                # legend
                if contours is not None:
                    if j == 0:
                        labels1.append(label1)
                    continue  # no colorbars, and no 'per panel' legends (add one at the end)

                line = plt.Line2D([0], [0], color="white", marker="", lw=0.0)
                if label2 is not None:
                    legend2 = ax.legend([line, line], [label1, label2], handlelength=0.0, loc="upper right")
                else:
                    legend2 = ax.legend([line], [label1], handlelength=0.0, loc="upper right")
                ax.add_artist(legend2)
                plt.setp(legend2.get_texts(), color="white")

                # colorbar(s)
                if len(mStarBins) > 1 or yAxis != "vrad":
                    # some panels with each colorbar
                    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
                    if yAxis == "vrad" and i % 2 == 1:
                        cb = plt.colorbar(im_neg, cax=cax)
                        cb.ax.set_ylabel(cbarlabel2)
                    else:
                        cb = plt.colorbar(im_pos, cax=cax)
                        cb.ax.set_ylabel(cbarlabel)
                else:
                    # both colorbars on the same, single panel
                    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.7)
                    cb = plt.colorbar(im_pos, cax=cax)
                    cb.ax.set_ylabel(cbarlabel.replace("Outflow", "Inflow/Outflow"))

                    bbox_ax = cax.get_position()
                    cax2 = fig.add_axes([0.814, bbox_ax.y0 + 0.004, 0.038, bbox_ax.height + 0.0785])  # manual tweaks
                    cb2 = plt.colorbar(im_neg, cax=cax2)
                    cb2.ax.set_yticklabels("")

                # special labels/behavior
                if yAxis == "theta":
                    opts = {"color": "#000000", "fontsize": 17, "ha": "center", "va": "center"}
                    xx = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.25
                    ax.text(xx, np.pi / 2, "minor axis", **opts)
                    ax.text(xx, -np.pi / 2, "minor axis", **opts)

                if xAxis == "rad" and yAxis == "theta":
                    binsize_theta = (limits["theta"][1] - limits["theta"][0]) / numBins["theta"]
                    theta_vals = binConfig["theta"][1:] + binsize_theta / 2

                    for iternum in [0, 1]:  # theta>0, theta<0
                        y_lower = np.zeros(numBins["rad"], dtype="float32")
                        y_upper = np.zeros(numBins["rad"], dtype="float32")

                        for radbinnum in range(numBins["rad"]):
                            theta_dist_loc = 10.0 ** h2d_pos[radbinnum, :]  # linear msun/yr

                            # select either theta>0 or theta<0
                            if iternum == 0:
                                dist = theta_dist_loc[int(theta_dist_loc.size / 2) :]
                                thetavals = theta_vals[int(theta_dist_loc.size / 2) :]
                            if iternum == 1:
                                dist = theta_dist_loc[: int(theta_dist_loc.size / 2)]
                                thetavals = theta_vals[: int(theta_dist_loc.size / 2)]

                            # locate 25-75 percentiles, i.e. derive opening angle of 'half mass flux'
                            csum = np.cumsum(dist) / np.sum(dist)

                            y_lower[radbinnum], y_upper[radbinnum] = np.interp([0.25, 0.75], csum, thetavals)

                        lastIndPlot = 8 if i == 0 else 9  # stop before noise dominates
                        rad_vals = np.arange(binConfig["rad"].size - 1) + 0.5
                        # opening_angle = np.rad2deg(y_upper - y_lower)
                        # print(mStarBin,iternum,rad_vals[:lastIndPlot],opening_angle[:lastIndPlot])

                        ax.plot(rad_vals[:lastIndPlot], y_lower[:lastIndPlot], "-", color="white", alpha=0.3)
                        ax.plot(rad_vals[:lastIndPlot], y_upper[:lastIndPlot], "-", color="white", alpha=0.3)

                # for massBins done
            # for redshifts done

        # single legend?
        if contours is not None:
            # labels for M* bins
            leg_lines = lines
            leg_labels = labels1
            leg_lines2 = []
            leg_labels2 = []

            if len(contours) > 1:  # labels for contour levels
                for i, contour in enumerate(contours):
                    leg_lines2.append(plt.Line2D([0], [0], color="black", marker="", lw=lw, ls=linestyles[i]))
                    leg_labels2.append("%s = %.1f" % (contourlabel, contour))

            if len(redshifts) > 1:  # labels for redshifts
                for j, redshift in enumerate(redshifts):
                    leg_lines2.append(plt.Line2D([0], [0], color="black", lw=lw, ls=linestyles[j], marker=""))
                    leg_labels2.append("z = %.1f" % redshift)

            legend3 = ax.legend(leg_lines, leg_labels, loc="lower left")
            ax.add_artist(legend3)

            legend4 = ax.legend(leg_lines2, leg_labels2, loc="upper right")
            ax.add_artist(legend4)

            # label for r,vcut?
            if label2 is not None:
                if len(contours) == 1:
                    label2 += ", %s = %.1f" % (contourlabel, contours[0])  # not enumerated, so show the 1 choice
                line = plt.Line2D([0], [0], color="white", marker="", lw=lw)
                legend4 = ax.legend([line], [label2], handlelength=0.0, loc="upper left")
                ax.add_artist(legend4)

        # finish plot
        if saveName is not None:
            fig.savefig(saveName)
        if pdf is not None:
            pdf.savefig()
        plt.close(fig)

    # load
    thirdQuant = None if (xAxis == "rad" and yAxis == "vcut") else xAxis  # use to load x-axis quantity
    if thirdQuant == "rad":
        thirdQuant = yAxis  # if x-axis is rad, then use to load y-axis quantity (which is something other than vcut)

    # use to load y-axis quantity (which is neither rad nor vcut)
    fourthQuant = None if (xAxis == "rad" or yAxis == "vcut") else yAxis

    # load
    data = []

    for redshift in redshifts:
        if redshift is not None:
            sP.setRedshift(redshift)

        if xAxis == "vrad" or yAxis == "vrad":
            # non-standard dataset, i.e. not rad.vrad.*
            secondQuant = xAxis
            thirdQuant = yAxis
            fourthQuant = None

            mdot, mstar, subids, binConfig, numBins, vcut_vals = radialMassFluxes(
                sP,
                scope,
                ptType,
                secondQuant=secondQuant,
                thirdQuant=thirdQuant,
                v200norm=v200norm,
                rawMass=(rawMass or rawDens),
            )
        else:
            # default behavior
            mdot, mstar, subids, binConfig, numBins, vcut_vals = radialMassFluxes(
                sP,
                scope,
                ptType,
                thirdQuant=thirdQuant,
                fourthQuant=fourthQuant,
                v200norm=v200norm,
                rawMass=(rawMass or rawDens),
            )

        binConfig["vcut"] = vcut_vals
        numBins["vcut"] = vcut_vals.size

        # update bounds based on the loaded dataset
        for quant in [xAxis, yAxis]:
            assert quant in binConfig
            if limits[quant] is not False:
                continue  # hard-coded

            limits[quant] = [binConfig[quant].min(), binConfig[quant].max()]  # always linear spacing

            if np.any(np.isinf(limits[quant])):
                limits[quant] = None  # discrete labels (small number of bins)

        # load smoothed star formation rates, and crossmatch to subhalos with mdot
        sfr_smoothed = None

        if eta:
            sfr_timescale = 100.0  # Myr
            sfr_smoothed, _, _, _ = sP.simSubhaloQuantity("sfr_30pkpc_%dmyr" % sfr_timescale)  # msun/yr

            gcIDs = np.arange(0, sP.numSubhalos)
            assert sP.numSubhalos == sfr_smoothed.size
            gc_inds, _ = match(gcIDs, subids)

            sfr_smoothed = sfr_smoothed[gc_inds]

        # append to data list
        data.append((mdot, mstar, subids, binConfig, numBins, vcut_vals, sfr_smoothed))

    # single plot: if config passed in
    zStr = str(sP.snap) if len(redshifts) == 1 else "z=" + "-".join(["%.1f" % z for z in redshifts])
    v200Str = "_v200norm" if v200norm else ""

    if config is not None:
        saveName = "%s_mstar_%s_%s_%s_%s%s.pdf" % (saveBase, sP.simName, zStr, config["stat"], cStr, v200Str)
        if "saveName" in config:
            saveName = config["saveName"]
        if "vcutInd" not in config:
            config["vcutInd"] = None
        if "radInd" not in config:
            config["radInd"] = None

        _plotHelper(
            xAxis,
            yAxis,
            mStarBins,
            config["stat"],
            vcut_ind=config["vcutInd"],
            rad_ind=config["radInd"],
            saveName=saveName,
        )
        return

    # plots: explore all
    for stat in ["mean"]:  # ['mean','median']:
        print(xAxis, yAxis, stat, "eta:", eta)
        # (A) 2D histogram, where axes consider all (rad,vcut) or (rad,vrad) values
        if xAxis == "rad" and yAxis in ["vcut", "vrad"]:
            saveName = "%s_mstar_%s_%s_%s_%s%s.pdf" % (saveBase, sP.simName, zStr, stat, cStr, v200Str)
            _plotHelper(xAxis, yAxis, mStarBins, stat, saveName=saveName)

            continue

        # (B) 2D histogram, where xAxis is still rad, so make separate plots for each (vcut) value
        if xAxis == "rad":
            pdf = PdfPages("%s_B_mstar_%s_%s_%s_%s%s.pdf" % (saveBase, sP.simName, zStr, stat, cStr, v200Str))

            for vcutInd in range(numBins["vcut"]):
                _plotHelper(xAxis, yAxis, mStarBins, stat, vcut_ind=vcutInd, pdf=pdf)

            pdf.close()
            continue

        # (C) 2D histogram, where yAxis is (vcut) values, so make separate plots for each (rad) value
        if yAxis in ["vcut", "vrad"]:
            pdf = PdfPages("%s_C_mstar_%s_%s_%s_%s%s.pdf" % (saveBase, sP.simName, zStr, stat, cStr, v200Str))

            for radInd in range(numBins["rad"]):
                _plotHelper(xAxis, yAxis, mStarBins, stat, rad_ind=radInd, pdf=pdf)

            pdf.close()
            continue

        # (D) 2D histogram, where neither xAxis nor yAxis cover any (rad,vcut) values, so need to iterate over both
        pdf = PdfPages("%s_D_mstar_%s_%s_%s_%s%s.pdf" % (saveBase, sP.simName, zStr, stat, cStr, v200Str))

        for vcutInd in range(numBins["vcut"]):
            for radInd in range(numBins["rad"]):
                _plotHelper(xAxis, yAxis, mStarBins, stat, vcut_ind=vcutInd, rad_ind=radInd, pdf=pdf)

        pdf.close()
