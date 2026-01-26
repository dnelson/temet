"""
LRG small-scale cold clouds CGM paper (TNG50).

https://arxiv.org/abs/2005.09654
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic, binned_statistic_2d

from temet.cosmo.util import subboxSubhaloCat
from temet.load.data import berg2019, chen2018zahedy2019, werk2013
from temet.plot import snapshot, subhalos
from temet.plot.config import colors, figsize, figsize_sm, linestyles, percs, sKn, sKo
from temet.plot.util import loadColorTable
from temet.projects.oxygen import (
    ionTwoPointCorrelation,
    obsColumnsDataPlotExtended,
    obsColumnsLambdaVsR,
    obsSimMatchedGalaxySamples,
    stackedRadialProfiles,
    totalIonMassVsHaloMass,
)
from temet.tracer import evolution as tracerEvo
from temet.tracer.montecarlo import globalAllTracersTimeEvo
from temet.util import simParams
from temet.util.helper import logZeroNaN, running_median, shrinking_center
from temet.util.match import match
from temet.util.voronoi import voronoiThresholdSegmentation
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def radialResolutionProfiles(
    sPs, saveName, redshift=0.5, cenSatSelect="cen", radRelToVirRad=False, haloMassBins=None, stellarMassBins=None
):
    """Plot average/stacked radial gas cellsize profiles in stellar mass bins.

    Specify one of haloMassBins or stellarMassBins. If radRelToVirRad, then [r/rvir] instead of [pkpc].
    """
    # config
    percs = [10, 90]
    # # 'Mean' or 'Median' or 'Min' or 'p10', 'Gas_SFReq0' or 'Gas'
    fieldNames = ["Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_Mean", "Subhalo_RadProfile3D_FoF_Gas_CellSizeKpc_p10"]

    # plot setup
    lw = 3.0
    fig, ax = plt.subplots()

    if radRelToVirRad:
        ax.set_xlim([-2.0, 0.0])
        ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
        ax.set_xlabel("Radius / Virial Radius [ log ]")
    else:
        ax.set_xlim([0.0, 3.2])
        ax.set_xlabel("Radius [ log pkpc ]")

    ax.set_ylim([-0.9, 1.0])
    ax.set_ylabel(r"Gas Resolution $r_{\rm cell}$ [ log kpc ]")

    # init
    rvirs = []

    if haloMassBins is not None:
        massField = "mhalo_200_log"
        massBins = haloMassBins
    else:
        massField = "mstar_30pkpc_log"
        massBins = stellarMassBins

    # mark 1 and 2 pkpc
    ax.plot(ax.get_xlim(), np.log10([2.0, 2.0]), lw=lw - 0.5, color="#cccccc", alpha=0.6)
    ax.plot(ax.get_xlim(), np.log10([1.0, 1.0]), lw=lw - 0.5, color="#cccccc", alpha=0.2)
    ax.plot(ax.get_xlim(), np.log10([0.5, 0.5]), lw=lw - 0.5, color="#cccccc", alpha=0.8)
    ax.text(-1.4, np.log10(2.0) - 0.025, "2 pkpc", color="#cccccc", va="top", ha="left", fontsize=20.0, alpha=0.6)
    ax.text(
        -0.5, np.log10(0.5) + 0.02, "500 parsecs", color="#cccccc", va="bottom", ha="left", fontsize=20.0, alpha=0.8
    )

    # loop over each fullbox run
    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        sP.setRedshift(redshift)
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii
        rad = sP.groupCat(fieldsSubhalos=["rhalo_200_code"])
        rad = rad[cssInds]

        # load and apply CSS
        for j, fieldName in enumerate(fieldNames):
            if "p10" in fieldName and sP.res != 2160:
                continue  # only add 10th percentile curves for TNG50-1

            ac = sP.auxCat(fields=[fieldName])
            if ac[fieldName] is None:
                continue
            yy = ac[fieldName]

            # crossmatch 'subhaloIDs' to cssInds
            ac_inds, css_inds = match(ac["subhaloIDs"], cssInds)
            ac[fieldName] = ac[fieldName][ac_inds, :]
            masses_loc = masses[css_inds]
            rad_loc = rad[css_inds]

            # loop over mass bins
            for k, massBin in enumerate(massBins):
                if k in [0, 1] and sP.res in [540, 1080]:
                    continue  # only add all 3 massbins for TNG50-1

                # select
                w = np.where((masses_loc >= massBin[0]) & (masses_loc < massBin[1]))

                print("%s [%d] %.1f - %.1f : %d" % (sP.simName, k, massBin[0], massBin[1], len(w[0])))
                assert len(w[0])

                # radial bins: normalize to rvir if requested
                avg_rvir_code = np.nanmedian(rad_loc[w])
                if i == 0:
                    rvirs.append(avg_rvir_code)

                # y-quantity
                yy_local = np.squeeze(yy[w, :])  # check

                # x-quantity
                if radRelToVirRad:
                    rr = 10.0 ** ac[fieldName + "_attrs"]["rad_bins_code"] / avg_rvir_code
                else:
                    rr = ac[fieldName + "_attrs"]["rad_bins_pkpc"]

                # for low res runs, combine the inner bins which are poorly sampled
                if 0 and sP.res in [540, 1080]:
                    nInner = int(20 / (sP.res / 540))
                    rInner = np.mean(rr[0:nInner])

                    for dim in range(yy_local.shape[0]):
                        yy_local[dim, nInner - 1] = np.nanmedian(yy_local[dim, 0:nInner])
                    yy_local = yy_local[:, nInner - 1 :]
                    rr = np.hstack([rInner, rr[nInner:]])

                # replace zeros by nan so they are not included in percentiles
                yy_local[yy_local == 0.0] = np.nan

                # calculate mean profile and scatter
                if yy_local.ndim > 1:
                    yy_mean = np.nansum(yy_local, axis=0) / len(w[0])
                    yp = np.nanpercentile(yy_local, percs, axis=0)
                else:
                    yy_mean = yy_local  # single profile
                    yp = np.vstack((yy_local, yy_local))  # no scatter

                # log both axes
                yy_mean = logZeroNaN(yy_mean)
                yp = logZeroNaN(yp)
                rr = np.log10(rr)

                if rr.size > sKn:
                    sKn_loc = sKn + 8 if j == 0 else sKn + 16
                    yy_mean = savgol_filter(yy_mean, sKn_loc, sKo + 1)
                    yp = savgol_filter(yp, sKn_loc, sKo + 1, axis=1)  # P[10,90]

                # plot median line
                label = r"%.1f < $M_{\rm halo}$ < %.1f" % (massBin[0], massBin[1]) if (i == 0 and j == 0) else ""
                label = r"$M_{\rm halo}$ = %.1f" % (0.5 * (massBin[0] + massBin[1])) if (i == 0 and j == 0) else ""
                alpha = 1.0 if j == 0 else 0.3
                linewidth = lw if j == 0 else lw - 1

                (l,) = ax.plot(rr, yy_mean, lw=linewidth, color=colors[k], ls=linestyles[i], label=label, alpha=alpha)

                # draw rvir lines (or 300pkpc lines if x-axis is already relative to rvir)
                yrvir = ax.get_ylim()
                yrvir = np.array([yrvir[0], yrvir[0] + (yrvir[1] - yrvir[0]) * 0.15]) + 0.05

                if not radRelToVirRad:
                    xrvir = np.log10([avg_rvir_code, avg_rvir_code])
                    textStr = r"R$_{\rm vir}$"
                    yrvir[1] += 0.0 * k
                else:
                    rvir_150pkpc_ratio = sP.units.physicalKpcToCodeLength(150.0) / avg_rvir_code
                    xrvir = np.log10([rvir_150pkpc_ratio, rvir_150pkpc_ratio])
                    textStr = "150 kpc"
                    yrvir[1] += 0.0 * (len(massBins) - k)

                if i == 0 and j == 0:
                    ax.plot(xrvir, yrvir, lw=lw * 1.5, color=l.get_color(), alpha=0.1)
                    opts = {"va": "bottom", "ha": "right", "fontsize": 20.0, "alpha": 0.1, "rotation": 90}
                    ax.text(xrvir[0] - 0.02, yrvir[0], textStr, color=l.get_color(), **opts)

                    # show percentile scatter only for first run
                    ax.fill_between(rr, yp[0, :], yp[-1, :], color=l.get_color(), interpolate=True, alpha=0.2)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            lExtra += ["%s" % sP.simName]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, ncol=2, loc="upper left")

    fig.savefig(saveName)
    plt.close(fig)


def _getStackedGrids(sP, ion, haloMassBin, fullDepth, radRelToVirRad, ConfigLan=False, indiv=False, axesSets=([0, 1],)):
    """Helper: return (and cache) a concatenated {N_ion,dist} set of pixels for all halos in the given mass bin."""
    # grid config
    method = "sphMap"
    nPixels = [1000, 1000]
    axes = [0, 1]
    rotation = "edge-on"
    size = 400.0
    sizeType = "kpc"

    if fullDepth:
        # global accumulation with appropriate depth along the projection direction
        method = "sphMap_global"
        dv = 500.0  # +/- km/s (Zahedy / COS-LRG config) or +/- 600 km/s (Werk / COS-Halos config)
        depth_code_units = (2 * dv) / sP.units.H_of_a  # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

    eStr = ""
    if ConfigLan:
        size = 2000.0
        nPixels = [2000, 2000]
        rotation = None  # random

        dv = 1000.0  # tbd to match the stacking procedure
        depth_code_units = (2 * dv) / sP.units.H_of_a  # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

        eStr = "_2k"

    # quick caching
    cacheSaveFile = sP.cachePath + "ionColumnsVsImpact2D_%s_%d_%s_%.1f-%.1f_rvir=%s_fd=%s%s_a%d.hdf5" % (
        sP.simName,
        sP.snap,
        ion,
        haloMassBin[0],
        haloMassBin[1],
        radRelToVirRad,
        fullDepth,
        eStr,
        len(axesSets),
    )

    if isfile(cacheSaveFile):
        # load previous result
        with h5py.File(cacheSaveFile, "r") as f:
            dist_global = f["dist_global"][()]
            grid_global = f["grid_global"][()]
        print("Loaded: [%s]" % cacheSaveFile)
    else:
        # get halo IDs in mass bin (centrals only by definition)
        gc = sP.groupCat(fieldsSubhalos=["mhalo_200_log", "rhalo_200_code"])

        with np.errstate(invalid="ignore"):
            subInds = np.where((gc["mhalo_200_log"] > haloMassBin[0]) & (gc["mhalo_200_log"] < haloMassBin[1]))[0]

        # load grids
        dist_global = np.zeros((nPixels[0] * nPixels[1], len(subInds) * len(axesSets)), dtype="float32")
        grid_global = np.zeros((nPixels[0] * nPixels[1], len(subInds) * len(axesSets)), dtype="float32")

        for i, subhaloInd in enumerate(subInds):
            print(i, len(subInds), subhaloInd)

            class plotConfig:
                saveFilename = "dummy"

            # compute impact parameter for every pixel
            pxSize = size / nPixels[0]  # pkpc

            xx, yy = np.mgrid[0 : nPixels[0], 0 : nPixels[1]]
            xx = xx.astype("float64") - nPixels[0] / 2
            yy = yy.astype("float64") - nPixels[1] / 2
            dist = np.sqrt(xx**2 + yy**2) * pxSize

            if radRelToVirRad:
                dist /= gc["rhalo_200_code"][subInds[i]]

            for j, axes in enumerate(axesSets):  # noqa: B007
                # loop over projection directions and render
                panels = [{"partType": "gas", "partField": ion, "valMinMax": [-1.4, 0.2]}]
                grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                # flatten and stamp
                dist_global[:, i * len(axesSets) + j] = dist.ravel()
                grid_global[:, i * len(axesSets) + j] = grid.ravel()

        # save cache
        with h5py.File(cacheSaveFile, "w") as f:
            f["dist_global"] = dist_global
            f["grid_global"] = grid_global

        print("Saved: [%s]" % cacheSaveFile)

    if not indiv:
        # flatten
        dist_global = dist_global.ravel()
        grid_global = grid_global.ravel()

    return dist_global, grid_global


def ionColumnsVsImpact2D(sP, haloMassBin, ion, radRelToVirRad=False, ycum=False, fullDepth=False):
    """Use gridded N_ion maps to plot a 2D pixel histogram of (N_ion vs impact parameter)."""
    ylim = [11.0, 17.0]  # N_ion
    if "MHI" in ion:
        ylim = [13.0, 22.0]

    minVals = [-np.inf, ylim[0]] if ion == "Mg II" else [-np.inf]

    xlog = False
    nBins = 100
    ctName = "viridis"
    medianLine = True
    colorMed = "white"

    ionName = ion
    if ionName == "MHI_GK":
        ionName = "HI"

    if ycum:
        cMinMax = [-2.0, 0.0]  # log fraction
        clabel = r"Covering Fraction $\,\kappa\,$(N$_{\rm %s} \geq$ N)" % ionName
    else:
        cMinMax = [-3.0, -1.8]  # log fraction
        # if 'MHI' in ion: cMinMax = [-3.0, -1.4]
        clabel = "Conditional Covering Fraction = N [log]"

    # load projected column density grids of all halos
    dist_global, grid_global = _getStackedGrids(sP, ion, haloMassBin, fullDepth, radRelToVirRad)

    # start plot
    fig, ax = plt.subplots()

    if xlog:
        if radRelToVirRad:
            ax.set_xlim([-2.0, 0.0])
            ax.set_xticks([-2.0, -1.5, -1.0, -0.5, 0.0])
            ax.set_xlabel("Impact Parameter / Virial Radius [ log ]")
        else:
            ax.set_xlim([0.5, 2.5])
            ax.set_xlabel("Impact Parameter [ log pkpc ]")
    else:
        if radRelToVirRad:
            ax.set_xlim([0.0, 1.0])
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlabel("Impact Parameter / Virial Radius")
        else:
            ax.set_xlim([0, 200])
            ax.set_xlabel("Impact Parameter [ pkpc ]")

    ax.set_ylim(ylim)
    ax.set_ylabel(r"N$_{\rm %s}$ [ log cm$^{-2}$ ]" % ionName)

    # plot
    w = np.where((dist_global > 0) & np.isfinite(grid_global))

    dist_global = dist_global[w]  # pkpc or r/rvir
    grid_global = grid_global[w]  # log cm^2

    if xlog:
        dist_global = np.log10(dist_global)

    sim_cvals = np.zeros(dist_global.size, dtype="float32")  # unused currently

    # histogram 2d
    bbox = ax.get_window_extent()
    xlim = ax.get_xlim()

    nBins2D = np.array([nBins, int(nBins * (bbox.height / bbox.width))])
    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    cc, xBins, yBins, inds = binned_statistic_2d(
        dist_global, grid_global, sim_cvals, "count", bins=nBins2D, range=[xlim, ylim]
    )

    cc = cc.T  # imshow convention

    # histogram again, this time extending the y-axis bounds over all values, such that every pixel is counted
    # required for proper normalizations
    nn, _, _, _ = binned_statistic_2d(
        dist_global, grid_global, sim_cvals, "count", bins=nBins2D, range=[xlim, [grid_global.min(), grid_global.max()]]
    )
    nn = nn.T

    # normalize each column separately: cc value is [fraction of sightlines, at this impact parameter, with this column]
    with np.errstate(invalid="ignore"):
        totVals = np.nansum(nn, axis=0)
        totVals[totVals == 0] = 1
        cc /= totVals[np.newaxis, :]

    # cumulative y? i.e. each cc value becomes [fraction of sightlines, at this impact parameter, with >= this column]
    if ycum:
        cc = np.cumsum(cc[::-1, :], axis=0)[::-1, :]  # flips give >= this column, otherwise is actually <= this column

    # units and colormap
    if not ycum:
        cc2d = logZeroNaN(cc)
    else:
        # linear fraction for cumulative version
        cc2d = cc
        cMinMax = [0, 0.8]

    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)

    cmap = loadColorTable(ctName, numColors=8 if ycum else None)
    cc2d_rgb = cmap(norm(cc2d))

    # mask empty bins to white
    # cc2d_rgb[(cc == 0),:] = colorConverter.to_rgba('white')

    plt.imshow(cc2d_rgb, extent=extent, origin="lower", interpolation="nearest", aspect="auto", cmap=cmap, norm=norm)

    if medianLine:
        # show a second median for MgII, restricting to detectable columns
        for minVal in minVals:
            binSizeMed = (xlim[1] - xlim[0]) / nBins * 2

            w = np.where(grid_global >= minVal)
            xm, ym, sm, pm = running_median(dist_global[w], grid_global[w], binSize=binSizeMed, percs=percs)
            if xm.size > sKn:
                ym = savgol_filter(ym, sKn, sKo)
                sm = savgol_filter(sm, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)

            ls = "-" if (len(minVals) == 1 or np.isfinite(minVal)) else "--"
            ax.plot(xm, ym, ls, color=colorMed, alpha=0.9)
            # if ~np.isfinite(minVal) and len(minVals) == 1:
            #    ax.plot(xm, pm[0,:], ':', color=colorMed)
            #    ax.plot(xm, pm[-1,:], ':', color=colorMed)
            if len(minVals) == 1 or np.isfinite(minVal):
                ax.fill_between(xm, pm[0, :], pm[-1, :], color=colorMed, alpha=0.05)
                ax.plot(xm, pm[0, :], "-", color=colorMed, lw=1, alpha=0.2 if len(minVals) == 2 else 0.5)
                ax.plot(xm, pm[-1, :], "-", color=colorMed, lw=1, alpha=0.2 if len(minVals) == 2 else 0.5)

    # add obs data points
    colors = ["white", "tab:orange", "#d34d29"]
    markersize = 8.0
    off = 0.2  # force all obs points to be within plot limits

    if "HI" in ion:
        # COS-Halos, only red/massive? maybe Werk+14 Fig 14(b)
        _, logM, _, sfr, _, _, d, N_HI, N_HI_err, N_HI_lim = werk2013(ionName="H I")

        ssfr = sfr / 10.0**logM
        w = np.where(ssfr < 1e-11)
        d = d[w]
        N_HI = N_HI[w]
        N_HI_err = N_HI_err[w]
        N_HI_lim = N_HI_lim[w]  # 0=exact, 1=upper, 2=lower

        w = np.where(N_HI_lim == 0)
        ax.plot(
            d[w],
            np.clip(N_HI[w], ylim[0] + off, ylim[1] - off),
            "o",
            color=colors[1],
            markersize=markersize,
            label="COS-Halos (red)",
        )
        w = np.where(N_HI_lim == 1)
        ax.plot(d[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "v", color=colors[1], markersize=markersize)
        w = np.where(N_HI_lim == 2)
        ax.plot(d[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "^", color=colors[1], markersize=markersize)

        # COS-LRG
        _, _, _, _, _, _, d, N_HI, N_HI_err, N_MgII, N_MgII_err = chen2018zahedy2019()

        w = np.where((N_HI_err != "-") & (N_HI_err != ">") & (N_HI_err != "<"))
        ax.plot(
            d[w],
            np.clip(N_HI[w], ylim[0] + off, ylim[1] - off),
            "o",
            color=colors[0],
            markersize=markersize,
            label="COS-LRG",
        )
        w = np.where(N_HI_err == ">")
        ax.plot(d[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "^", color=colors[0], markersize=markersize)
        w = np.where(N_HI_err == "<")
        ax.plot(d[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "v", color=colors[0], markersize=markersize)

        # LRG-RDR
        _, _, _, _, _, _, b, N_HI, N_HI_err, N_HI_lim = berg2019()  # RDR survey

        w = np.where(N_HI_lim == 0)
        ax.plot(
            b[w],
            np.clip(N_HI[w], ylim[0] + off, ylim[1] - off),
            "o",
            color=colors[2],
            markersize=markersize,
            label="LRG-RDR",
        )
        w = np.where(N_HI_lim == 1)
        ax.plot(
            b[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "v", color=colors[2], markersize=markersize
        )  # visual offset
        w = np.where(N_HI_lim == 2)
        ax.plot(b[w], np.clip(N_HI[w], ylim[0] + off, ylim[1] - off), "^", color=colors[2], markersize=markersize)

        # legend
        legend = ax.legend(loc="upper right")
        for handle, text in zip(legend.legendHandles, legend.get_texts()):
            text.set_color(handle.get_color())

    if "Mg II" in ion:
        # COS-Halos
        _, logM, _, sfr, _, _, d, N_MgII, N_MgII_err, N_MgII_lim = werk2013(ionName="Mg II")

        # restrict to 'red'
        ssfr = sfr / 10.0**logM
        w = np.where(ssfr < 1e-11)
        d = d[w]
        N_MgII = N_MgII[w]
        N_MgII_err = N_MgII_err[w]
        N_MgII_lim = N_MgII_lim[w]  # 0=exact, 1=upper, 2=lower

        w = np.where(N_MgII_lim == 0)
        ax.plot(
            d[w],
            np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off),
            "o",
            color=colors[1],
            markersize=markersize,
            label="COS-Halos (red)",
        )
        w = np.where(N_MgII_lim == 1)
        ax.plot(d[w], np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off), "v", color=colors[1], markersize=markersize)
        w = np.where(N_MgII_lim == 2)
        ax.plot(d[w], np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off), "^", color=colors[1], markersize=markersize)

        # COS-LRG
        _, _, _, _, _, _, d, N_HI, N_HI_err, N_MgII, N_MgII_err = chen2018zahedy2019()

        w = np.where((N_MgII_err != "-") & (N_MgII_err != ">") & (N_MgII_err != "<"))
        ax.plot(
            d[w],
            np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off),
            "o",
            color=colors[0],
            markersize=markersize,
            label="COS-LRG",
        )
        w = np.where(N_MgII_err == ">")
        ax.plot(d[w], np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off), "^", color=colors[0], markersize=markersize)
        w = np.where(N_MgII_err == "<")
        ax.plot(d[w], np.clip(N_MgII[w], ylim[0] + off, ylim[1] - off), "v", color=colors[0], markersize=markersize)

        # legend
        legend = ax.legend(loc="upper right")
        for handle, text in zip(legend.legendHandles, legend.get_texts()):
            text.set_color(handle.get_color())

    # colorbar and finish plot
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
    cb = plt.colorbar(cax=cax)
    cb.ax.set_ylabel(clabel)
    # ax.set_title(r'%.1f < M$_{\rm halo}$ [log M$_\odot$] < %.1f' % (haloMassBin[0],haloMassBin[1]))

    fig.savefig(
        "ionColumnsVsImpact2D_%s_%s_%.1f-%.1f_rvir=%d_xlog=%d_ycum=%d_fd=%d.pdf"
        % (sP.simName, ion, haloMassBin[0], haloMassBin[1], radRelToVirRad, xlog, ycum, fullDepth)
    )
    plt.close(fig)


def ionCoveringFractionVsImpact2D(sPs, haloMassBin, ion, Nthresh, sPs2=None, radRelToVirRad=False, fullDepth=False):
    """Use gridded N_ion maps to plot covering fraction f(N_ion>N_thresh) vs impact parameter."""
    nBins = 50
    xlim = [1, 3.1]  # log pkpc
    ylim = [1e-3, 1]  # Lan MgII LRGs
    ylog = True

    def _get_grids(sPs):
        """Helper to load grids from a set of runs."""
        for i, sP in enumerate(sPs):
            dist_loc, grid_loc = _getStackedGrids(
                sP,
                ion,
                haloMassBin,
                fullDepth,
                radRelToVirRad,
                ConfigLan=True,
                indiv=True,
                axesSets=[[0, 1], [0, 2], [1, 2]],
            )

            if i == 0:
                dist_global = dist_loc
                grid_global = grid_loc
            else:
                dist_global = np.hstack((dist_global, dist_loc))
                grid_global = np.hstack((grid_global, grid_loc))

        print(i)
        dist_global[dist_global == 0] = dist_global[dist_global > 0].min() / 2

        dist_global = np.log10(dist_global)
        print(i)

        return dist_global, grid_global

    def _covering_fracs(dist, grid, col_N):
        """Helper, compute fc for all halos individually, and also the stack."""
        numHalos = dist.shape[1]

        fc = np.zeros((nBins, numHalos + 1), dtype="float32")

        for i in range(numHalos + 1):
            # derive covering fraction
            loc_dist = dist[:, i] if i < numHalos else dist.ravel()
            loc_grid = grid[:, i] if i < numHalos else grid.ravel()

            mask = np.zeros(loc_dist.size, dtype="int16")  # 0 = below, 1 = above
            w = np.where(loc_grid >= col_N)
            mask[w] = 1

            count_above, _, _ = binned_statistic(loc_dist, mask, "sum", bins=nBins, range=[xlim])
            count_total, bin_edges, _ = binned_statistic(loc_dist, mask, "count", bins=nBins, range=[xlim])

            # take ratio and smooth
            xx = bin_edges[:-1] + (xlim[1] - xlim[0]) / nBins / 2
            with np.errstate(invalid="ignore"):
                fc[:, i] = count_above / count_total

            fc[:, i] = savgol_filter(fc[:, i], sKn, sKo)

        return xx, fc

    # load projected column density grids of all halos (possibly across more than one run/redshift)
    dist_global, grid_global = _get_grids(sPs)

    # start plot
    fig, ax = plt.subplots()

    ax.set_xlim([12, 1200])
    ax.set_xscale("log")
    ax.set_xlabel("Impact Parameter [ log pkpc ]")

    ax.set_ylim(ylim)
    if ylog:
        ax.set_yscale("log")
    ax.set_ylabel("%s Covering Fraction" % (ion))

    # loop over requested column density limits
    for N in Nthresh:
        print(N)
        # loop over individual halos, and one extra iter for all
        xx, fc = _covering_fracs(dist_global, grid_global, N)

        # plot global
        label = r"N$_{\rm %s}$ > 10$^{%.1f}$ cm$^{-2}$" % (ion, N)
        (l,) = ax.plot(10**xx, fc[:, -1], "-", alpha=1.0, label=label)

        # fill band (1 sigma halo-to-halo variation) for first column threshold only
        if N == Nthresh[0]:
            fc_percs = np.percentile(fc[:, :-1], [16, 84], axis=1)
            fc_percs = savgol_filter(fc_percs, sKn, sKo, axis=1)

            ax.fill_between(10**xx, fc_percs[0, :], fc_percs[1, :], color=l.get_color(), alpha=0.1, interpolate=True)

    # second sim set? only do for last column threshold
    if sPs2 is not None:
        dist_global = None
        grid_global = None

        lines = []
        for i, sPset in enumerate(sPs2):
            print(i, sPset[0].simName)

            # load grid
            dist_global, grid_global = _get_grids(sPset)

            # covering fraction calculation and plot
            xx, fc = _covering_fracs(dist_global, grid_global, Nthresh[-1])
            (l,) = ax.plot(10**xx, fc[:, -1], linestyles[i + 1], alpha=1.0, color=l.get_color())
            lines.append(l)

    # observations: Bowen+11: LRGs
    b14_label = "Bowen+ (2011) LRGs"
    b14_rp = [25, 75, 125, 175]
    b14_fc = np.array([0.093, 0.089, 1e-5, 1e-5])
    b14_up = np.array([0.154, 0.130, 0.02, 0.02])
    b14_low = np.array([0.035, 0.050, 1e-6, 1e-6])

    ax.errorbar(
        b14_rp,
        b14_fc,
        yerr=[b14_fc - b14_low, b14_up - b14_fc],
        markerSize=8,
        color="black",
        ecolor="black",
        alpha=0.4,
        capsize=0.0,
        fmt="D",
        label=b14_label,
    )

    # observations: Lan+14 Fig 8: fc (W > 1 Ang, "passive" i < 20.6)
    lan14_label = r"Lan+ (2014) W$_{\rm 0}^{\rm MgII}$ > 1$\AA$"
    lan14_rp = [25.0, 36.0, 50.0, 70.0, 100, 150, 210, 300, 430]
    lan14_fc = np.array([0.1, 0.147, 0.093, 0.048, 0.034, 0.019, 0.013, 0.0088, 0.0071])
    lan14_up = np.array([0.149, 0.189, 0.118, 0.064, 0.044, 0.026, 0.018, 0.012, 0.010])
    lan14_low = np.array([0.0508, 0.104, 0.068, 0.033, 0.024, 0.012, 0.008, 0.0052, 0.0042])

    ax.errorbar(
        lan14_rp,
        lan14_fc,
        yerr=[lan14_fc - lan14_low, lan14_up - lan14_fc],
        markerSize=8,
        color="black",
        ecolor="black",
        alpha=0.6,
        capsize=0.0,
        fmt="s",
        label=lan14_label,
    )

    # observations: Lan+18: fc (W > 0.4 Ang, LRGs, DR14)
    lan18_label = r"Lan+ (2018) W$_{\rm 0}^{\rm MgII}$ > 0.4$\AA$"
    lan18_rp = [23, 35, 48, 68, 95, 135, 189, 265, 380, 525, 740, 1.0e3]
    lan18_fc = np.array([0.205, 0.187, 0.291, 0.142, 0.196, 0.068, 0.061, 0.047, 0.027, 0.026, 0.014, 0.007])
    lan18_up = np.array([0.371, 0.417, 0.383, 0.203, 0.250, 0.097, 0.089, 0.061, 0.038, 0.034, 0.020, 0.011])
    lan18_down = np.array([0.037, 0.037, 0.201, 0.086, 0.142, 0.039, 0.033, 0.032, 0.016, 0.017, 0.009, 0.004])

    ax.errorbar(
        lan18_rp,
        lan18_fc,
        yerr=[lan18_fc - lan18_down, lan18_up - lan18_fc],
        markerSize=8,
        color="black",
        ecolor="black",
        alpha=0.9,
        capsize=0.0,
        fmt="p",
        label=lan18_label,
    )

    # observations: Zahedy+19 COS-LRG: fc (N_MgII > 10^13 cm^-3, poor statistics)
    z19_label = r"Zahedy+ (2019) N$_{\rm MgII}$ > 10$^{13}$ cm$^{-3}$"
    z19_rp = [50, 130]  # very rough averages of "d<100kpc" and "d = 100-160kpc"
    z19_fc = np.array([0.60, 0.0])
    z19_up = np.array([0.85, 0.2])
    z19_down = np.array([0.30, 0.0])

    ax.errorbar(
        z19_rp,
        z19_fc,
        yerr=[z19_fc - z19_down, z19_up - z19_fc],
        markerSize=8,
        color="black",
        ecolor="black",
        alpha=0.3,
        capsize=0.0,
        fmt="s",
        label=z19_label,
    )

    # finish plot
    if sPs2 is not None:
        legend2 = ax.legend(lines, [sPset[0].simName for sPset in sPs2], loc="upper right")
        ax.add_artist(legend2)

    ax.legend(loc="lower left")

    fig.savefig(
        "ionCoveringFracVsImpact2D_%s_%s_%.1f-%.1f_rvir=%d_fd=%d.pdf"
        % (sPs[0].simName, ion.replace(" ", ""), haloMassBin[0], haloMassBin[1], radRelToVirRad, fullDepth)
    )
    plt.close(fig)


def lrgHaloVisualization(sP, haloIDs, conf=3, gallery=False, globalDepth=True, testClumpRemoval=False):
    """Configure single halo and multi-halo gallery visualizations."""
    rVirFracs = [0.25]
    method = "sphMap"
    nPixels = [1000, 1000]
    axes = [0, 1]
    labelZ = True
    labelScale = "physical"
    labelSim = False
    labelHalo = True
    relCoords = False
    size = 400.0
    sizeType = "kpc"

    if not sP.isZoom:
        rotation = "edge-on"
    else:
        # MHD vs noMHD comparison
        nPixels = [1000, 500]
        cenShift = [0, +size / 4, 0]  # center on upper half, 2x1 aspect ratio

    # global with ~appropriate depth (same as in ionColumnsVsImpact2D)
    if globalDepth:
        method = "sphMap_global"
        dv = 500.0  # +/- km/s (Zahedy), or +/- 1000 km/s (Berg)
        depth_code_units = (2 * dv) / sP.units.H_of_a  # ckpc/h
        depthFac = sP.units.codeLengthToKpc(depth_code_units) / size

    # which conf?
    if conf == 1:
        panel = {"partType": "gas", "partField": "metal_solar", "valMinMax": [-1.4, 0.2]}
    if conf == 2:
        panel = {"partType": "gas", "partField": "MHI_GK", "valMinMax": [15.0, 21.0]}
        if sP.isZoom:
            panel["partField"] = "HI"
    if conf == 3:
        panel = {"partType": "gas", "partField": "Mg II", "valMinMax": [12.0, 16.5]}
    if conf == 4:
        panel = {"partType": "stars", "partField": "stellarComp"}
    if conf == 8:
        panel = {"partType": "gas", "partField": "tcool_tff", "valMinMax": [0.0, 2.0]}
        depthFac = 0.01  # test
    if conf == 9:
        panel = {"partType": "gas", "partField": "delta_rho", "valMinMax": [-0.3, 1.0]}
        # depthFac = 0.01 # test, = 4 kpc
    if conf == 10:
        panel = {"partType": "gas", "partField": "entropy", "valMinMax": [8.0, 9.0]}
        depthFac = 0.01  # test, = 4 kpc
    if conf == 11:
        panel = {"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [6.5, 9.5]}

    if conf in [5, 6, 7]:
        # NARROW SLICE! h19
        haloIDs = [19]

        nPixels = [2000, 1500]
        method = "sphMap"
        rVirFracs = [0.05, 0.1]
        depthFac = 0.2
        partType = "gas"
        size = 80.0
        cenShift = [-size / 2, -size / 2, 0]  # center on lower-left

        if 0:  # conf == 5
            # h1 test
            haloIDs = [1]
            rVirFracs = [0.01, 0.02, 0.05]
            depthFac = 0.15
            size = 50.0
            cenShift = [size / 2 + 4, size / 2 * (nPixels[1] / nPixels[0]) + 4, 0]  # center on upper-right quadrant

        labelZ = False
        labelHalo = False

        panel = []
        panel.append({"partField": "cellsize_kpc", "valMinMax": [-0.7, 0.0]})
        if conf == 5:
            panel.append({"partField": "P_gas", "valMinMax": [4.4, 6.0]})
            panel.append({"partField": "bmag_uG", "valMinMax": [-1.0, 1.0]})  # [2.8,5.2] 'P_B' for h1
            panel.append({"partField": "metal_solar", "valMinMax": [-1.0, -0.4]})  # [-1.5,1.5], 'pressure_ratio' for h1
        if conf == 6:
            panel.append({"partField": "P_gas", "valMinMax": [3.2, 4.6]})
            # panel.append( {'partField':'P_B', 'valMinMax':[2.0,3.8]} )
            panel.append({"partField": "metal_solar", "valMinMax": [-1.0, -0.4]})
            panel.append({"partField": "pressure_ratio", "valMinMax": [-1.0, 1.0]})
        if conf == 7:
            panel.append({"partField": "tcool", "valMinMax": [-1.0, 2.0]})
            panel.append({"partField": "tff", "valMinMax": [-1.5, -0.5]})
            panel.append({"partField": "tcool_tff", "valMinMax": [0.0, 2.0]})  # midpoint at tcool/tff=10

    if gallery:
        # multi-panel
        panels = []
        labelZ = False

        for haloID in haloIDs[0:12]:
            panel_loc = dict(panel)
            panel_loc["subhaloInd"] = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

            panels.append(panel_loc)

        class plotConfig:
            plotStyle = "edged"
            rasterPx = nPixels[0] * 2
            nRows = 3  # 3x4
            colorbars = True
            saveFilename = "./vis_%s_%d_%s.pdf" % (sP.simName, sP.snap, panels[0]["partField"].replace(" ", "_"))

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    else:
        # single image
        for haloID in haloIDs:
            subhaloInd = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]

            panels = [panel] if not isinstance(panel, list) else panel

        # test: remove single clump visually
        if len(haloIDs) == 1 and testClumpRemoval:
            th = {"propName": "Mg II numdens", "propThreshComp": "gt", "propThresh": 1e-8}
            objs, props = voronoiThresholdSegmentation(
                sP,
                haloID=haloIDs[0],
                propName=th["propName"],
                propThresh=th["propThresh"],
                propThreshComp=th["propThreshComp"],
            )

            clumpID = np.where(objs["lengths"] == 100)[0][8]
            print("Testing clump removal [ID = %d]." % clumpID)

            # halo-local indices of member cells
            def _getClumpInds(clumpID):
                offset = objs["offsets"][clumpID]
                length = objs["lengths"][clumpID]
                inds = objs["cell_inds"][offset : offset + length]
                return inds

            # inds1 = _getClumpInds(clumpID)
            # inds2 = _getClumpInds(3416)  # close in space
            # inds3 = _getClumpInds(1147)  # close in space
            # skipCellIndices = np.hstack((inds1, inds2, inds3))

            assert method == "sphMap"  # cell_inds to remove must index fof-scope indRange

        class plotConfig:
            plotStyle = "edged"
            rasterPx = nPixels
            colorbars = True
            saveFilename = saveFilename = "./vis_%s_%d_h%d_%s.pdf" % (
                sP.simName,
                sP.snap,
                haloID,
                panels[0]["partField"].replace(" ", "_"),
            )

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def lrgHaloVisResolution(sP, haloIDs, sPs_other):
    """Visualization: one halo, for four different resolution runs."""
    # cross match
    from temet.cosmo.util import crossMatchSubhalosBetweenRuns

    subIDs = [sP.halos("GroupFirstSub")[haloIDs]]

    for sPo in sPs_other:
        subIDs.append(crossMatchSubhalosBetweenRuns(sP, sPo, subIDs[0], method="Positional"))

    # panel config
    rVirFracs = [0.05, 0.1, 0.25]
    method = "sphMap"
    nPixels = [800, 800]
    axes = [1, 0]  # [0,1]
    labelZ = False
    labelScale = False
    labelSim = True
    labelHalo = False
    relCoords = False

    size = 200.0
    sizeType = "kpc"
    cenShift = [-size / 2, +size / 2, 0]  # center on left (half)

    partType = "gas"
    partField = "Mg II"
    valMinMax = [12.0, 16.5]

    # single image per halo
    for i, haloID in enumerate(haloIDs):
        panels = []

        for j, sPo in enumerate([sP] + sPs_other):
            panels.append({"run": sPo.run, "res": sPo.res, "redshift": sPo.redshift, "subhaloInd": subIDs[j][i]})

        panels[0]["labelScale"] = "physical"

        class plotConfig:
            plotStyle = "edged"
            rasterPx = nPixels
            colorbars = True
            nRows = 1
            saveFilename = saveFilename = "./vis_%s_%d_res_h%d.pdf" % (sP.simName, sP.snap, haloID)

        # render
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def cloudEvoVis(sP, haloID, clumpID, sbNum, sizeParam=False):
    """Visualize a series of time-evolution frames tracking a single cloud."""
    # vis conifg
    method = "sphMap_global"  # for subboxes
    nPixels = [800, 800]  # [x,y] shape of boxSizeImg must match nPixels aspect ratio
    axes = [0, 1]
    labelZ = False
    labelScale = True
    labelSim = True
    labelHalo = False
    plotHalos = False

    # subboxes: use snap-based HI since no subboxes are mini and no popping exists
    # full snaps: use popping catalog which is snap-complete, since many snaps are mini == no NeutralHydrogenAbundance
    partType = "gas"
    partField = "HI" if sbNum is not None else "MHI_GK"
    valMinMax = [16.0, 21.0]

    if sizeParam == 0:
        # constant zoomed in size
        size = 60.0
        depth = 20.0
    elif sizeParam == 1:
        # constant, more zoomed out size
        size = 400.0
        depth = 40.0
    elif sizeParam == 2:
        # adaptive, starting at these specifications
        size = 60.0 if sbNum is not None else 240.0  # code units
        depth = 20.0 if sbNum is not None else 80.0  # code units

    indivCircSize = 0.3 if sbNum is not None else 2.0

    # load tracer-based evolution tracks
    data = clumpTracerTracksLoad(sP, haloID, clumpID, sbNum, posOnly=True)

    # frameNums = np.arange(data['snaps'].size) # all
    frameNums = np.where(data["dt"] < 0)[0]  # only backwards in time

    if sbNum is None:
        # full box, get subhalo IDs for fof-scope loads
        method = "sphMap"
        subhaloID = sP.halo(haloID)["GroupFirstSub"]
        mpb = sP.loadMPB(subhaloID)
    else:
        variant = "subbox%d" % sbNum

    nearInds = np.arange(data["pos"].shape[1])  # init

    for i in frameNums:
        # decide frame center position and size
        snap = data["snaps"][i]

        boxSizeImg = [size, size, depth]

        # center: shrinking sphere on a subset of tracers, determined as those which were still
        # within a threshold distance at the previous snapshot
        boxCenter = shrinking_center(data["pos"][i, nearInds, :], sP.boxSize)  # track center with shrinking sphere

        dists = sP.periodicDists(boxCenter, data["pos"][i, :, :])
        nearInds = np.where(dists <= data["size_halfmassrad"][0] * 3)[0]

        extent = [
            boxCenter[0] - 0.5 * boxSizeImg[0],
            boxCenter[0] + 0.5 * boxSizeImg[0],
            boxCenter[1] - 0.5 * boxSizeImg[1],
            boxCenter[1] + 0.5 * boxSizeImg[1],
        ]

        trExtents = np.max(data["pos"][i, :, :], axis=0) - np.min(data["pos"][i, :, :], axis=0)

        if sizeParam == 2:
            if (trExtents[axes[0]] > size or trExtents[axes[1]] > size) and size < 400:
                size *= 2
            if (trExtents[3 - axes[0] - axes[1]] > depth) and depth < 300:
                depth *= 2

        print("[%2d] snap = %2d size = %4d depth = %3d" % (i, snap, size, depth))

        if trExtents[axes[0]] > size:
            print("Warning: Tracers have spread in extent along axes0 more than view size [%.2f]" % trExtents[axes[0]])
        if trExtents[axes[1]] > size:
            print("Warning: Tracers have spread in extent along axes1 more than view size [%.2f]" % trExtents[axes[1]])
        if trExtents[3 - axes[0] - axes[1]] > depth:
            print(
                "Warning: Tracers have spread in extent along LOS-axis more than depth [%.2f]"
                % trExtents[3 - axes[0] - axes[1]]
            )

        # full box snapshots: fof-scope load and render (global render for subbox snaps)
        subhaloInd = None
        if sbNum is None:
            w = np.where(mpb["SnapNum"] == snap)[0][0]
            subhaloInd = mpb["SubfindID"][w]
            print(" fullbox, located snap [%d] at ind [%d] with subhaloInd [%d]" % (snap, w, subhaloInd))

        panels = [{"run": sP.run, "res": sP.res}]

        # mark position of cloud, and member cells
        customCircles = {}
        customCrosses = {}

        customCircles["pos"] = boxCenter
        customCrosses["pos"] = data["pos"][i, :, :]

        customCircles["rad"] = data["size_halfmassrad"][0] * 2
        customCrosses["rad"] = np.zeros(data["pos"].shape[1]) + indivCircSize

        labelCustom = [r"$\Delta t$ = %.2f Myr" % data["dt"][i]]

        class plotConfig:
            plotStyle = "open"  # edged
            rasterPx = nPixels
            colorbars = True
            saveFilename = "./vis_cloudevo_%s_%s_%d_sb%s_h%d_c%d_size%d_frame%02d.png" % (
                partField,
                sP.simName,
                sP.snap,
                sbNum,
                haloID,
                clumpID,
                sizeParam,
                i,
            )

        # render
        renderBox(panels, plotConfig, locals(), skipExisting=False)


def cloudEvoVisFigure(sP, haloID, clumpID, sbNum, constSize=False):
    """Make figure for a series of time-evolution frames tracking a single cloud."""
    # vis conifg
    run = sP.run
    res = sP.res
    method = "sphMap_global"  # for subboxes
    nPixels = [150, 150]  # [x,y] shape of boxSizeImg must match nPixels aspect ratio
    axes = [0, 1]
    labelZ = False
    labelScale = False
    labelSim = False
    labelHalo = False
    plotHalos = False

    # subboxes: use snap-based HI since no subboxes are mini and no popping exists
    # full snaps: use popping catalog which is snap-complete, since many snaps are mini == no NeutralHydrogenAbundance
    partType = "gas"
    partField = "HI" if sbNum is not None else "MHI_GK"
    valMinMax = [16.0, 21.0]
    ctName = "thermal"

    indivCircSize = 0.3 if sbNum is not None else 2.0

    # load tracer-based evolution tracks
    data = clumpTracerTracksLoad(sP, haloID, clumpID, sbNum, posOnly=True)

    size = 10.0
    depth = 10.0

    boxSizeImg = [size, size, depth]

    frameNums = [0, 8, 20, 30, 60, 98, 128, 146]

    if sbNum is None:
        # full box, get subhalo IDs for fof-scope loads
        method = "sphMap"
        subhaloID = sP.halo(haloID)["GroupFirstSub"]
        mpb = sP.loadMPB(subhaloID)
    else:
        variant = "subbox%d" % sbNum

    # start panels
    panels = []

    for i in frameNums:
        # decide frame center position and size
        loc_snap = data["snaps"][i]

        boxCenter_loc = shrinking_center(data["pos"][i, :, :], sP.boxSize)  # track center with shrinking sphere
        extent_loc = [
            boxCenter_loc[0] - 0.5 * boxSizeImg[0],
            boxCenter_loc[0] + 0.5 * boxSizeImg[0],
            boxCenter_loc[1] - 0.5 * boxSizeImg[1],
            boxCenter_loc[1] + 0.5 * boxSizeImg[1],
        ]

        # full box snapshots: fof-scope load and render (global render for subbox snaps)
        subhaloInd_loc = None
        if sbNum is None:
            w = np.where(mpb["SnapNum"] == loc_snap)[0][0]
            subhaloInd_loc = mpb["SubfindID"][w]
            print(" fullbox, located snap [%d] at ind [%d] with subhaloInd [%d]" % (loc_snap, w, subhaloInd_loc))

        # mark position of cloud, and member cells
        customCircles_loc = {}
        customCrosses_loc = {}

        customCircles_loc["pos"] = boxCenter_loc
        customCrosses_loc["pos"] = data["pos"][i, :, :]

        customCircles_loc["rad"] = data["size_halfmassrad"][0] * 2
        customCrosses_loc["rad"] = np.zeros(data["pos"].shape[1]) + indivCircSize

        labelCustom_loc = [r"$\Delta t$ = %d Myr" % data["dt"][i]]

        # panel
        panel = {
            "snap": loc_snap,
            "boxCenter": boxCenter_loc,
            "extent": extent_loc,
            "subhaloInd": subhaloInd_loc,
            "customCircles": customCircles_loc,
            "customCrosses": customCrosses_loc,
            "labelCustom": labelCustom_loc,
        }

        panels.append(panel)

    panels = panels[::-1]  # forwards in time, from left to right
    panels[0]["labelScale"] = True

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels  # [nPixels[0]*len(frameNums),nPixels[1]]
        colorbars = True
        nRows = 1
        fontsize = 12
        saveFilename = "./vis_cloudevo_%s_%s_sb%s_h%d_c%d.pdf" % (partField, sP.simName, sbNum, haloID, clumpID)

    # render
    renderBox(panels, plotConfig, locals(), skipExisting=False)


# clump plot config
lims = {
    "size": [0, 3.0],  # linear pkpc
    "mass": [4.5, 8.5],  # log msun
    "ncells": [0, 40],  # linear
    "dist": [0, 500],  # linear pkpc
    "dens": [-3.0, 2.5],  # log cc
    "temp": [3.5, 5.0],  # log K
    "bmag": [-1.0, 2.0],  # log G
    "beta": [-2.0, 1.0],  # log
    "sfr": [0, 0.01],  # linear msun/yr
    "vrad": [-500, 200],  # linear km/s
    "specj": [4.0, 6.5],  # log kpc km/s
    "metal": [-1.4, 0.4],  # log solar
    "rcell1": [0, 800],  # linear parsec
    "rcell2": [0, 800],  # linear parsec
    "mg2_mass": [0.0, 5.0],  # log msun
    "hi_mass": [2.0, 7.0],
}  # log msun

labels = {
    "size": r"Clump Radius [ kpc ]",
    "mass": r"Clump Total Mass [ log M$_{\rm sun}$ ]",
    "ncells": r"Number of Gas Cells [ linear ]",
    "dist": r"Halocentric Distance [ kpc ]",
    "dens": r"Mean Hydrogen Number Density [ log cm$^{-3}$ ]",
    "temp": r"Mean Clump Temperature [ log K ]",
    "bmag": r"Mean Clump Magnetic Field Strength [ log $\mu$G ]",
    "beta": r"Mean $\beta = \rm{P}_{\rm gas} / \rm{P}_{\rm B}$ [ log ]",
    "sfr": r"Total Clump Star Formation Rate [ M$_{\rm sun}$ / yr ]",
    "vrad": r"Mean Halo-centric Radial Velocity [ km/s ]",
    "specj": r"Total Specific Angular Momentum [ log kpc km/s ]",
    "metal": r"Mean Clump Gas Metallicity [ log Z$_{\rm sun}$ ]",
    "rcell1": r"Average Member Gas r$_{\rm cell}$ [ parsec ]",
    "rcell2": r"Smallest Member Gas r$_{\rm cell}$ [ parsec ]",
    "mg2_mass": r"Total MgII Mass [ log M$_{\rm sun}$ ]",
    "hi_mass": r"Total Neutral HI Mass [ log M$_{\rm sun}$ ]",
    "number": r"Number of Discrete Clouds",
}

# default segmentation config
thPropName = "Mg II numdens"
thPropThresh = 1e-8
thPropThreshComp = "gt"


def _clump_values(sP, objs, props):
    """Helper: some common unit conversions."""
    values = {}
    values["size"] = sP.units.codeLengthToKpc(props["radius"])
    values["mass"] = sP.units.codeMassToLogMsun(props["mass"])
    values["ncells"] = objs["lengths"]
    values["dist"] = sP.units.codeLengthToKpc(props["distance"])
    values["dens"] = np.log10(props["dens_mean"])
    values["temp"] = np.log10(props["temp_mean"])
    values["bmag"] = np.log10(props["bmag_mean"] * 1e6)
    values["beta"] = np.log10(props["beta_mean"])
    values["sfr"] = props["sfr_tot"]
    values["vrad"] = props["vrad_mean"]
    values["specj"] = np.log10(props["specj_tot"])
    values["metal"] = np.log10(props["metal_mean"])
    values["rcell1"] = sP.units.codeLengthToKpc(props["rcell_mean"]) * 1000
    values["rcell2"] = sP.units.codeLengthToKpc(props["rcell_min"]) * 1000
    values["mg2_mass"] = sP.units.codeMassToLogMsun(props["mg2_mass"])
    values["hi_mass"] = sP.units.codeMassToLogMsun(props["hi_mass"])

    return values


def clumpDemographics(sPs, haloID, stackHaloIDs=None, trAnalysis=False):
    """Plot demographics of clump population for a single halo."""
    if not isinstance(sPs, list):
        sPs = [sPs]

    # config
    threshSets = []

    for val in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:  # [1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-15]:
        label = r"n$_{\rm Mg II}$ > %s cm$^{-3}$" % val
        threshSets.append({"propName": "Mg II numdens", "comp": "gt", "propThresh": val, "label": label})

    # for val in [4.0,4.2,4.4,4.8]:
    #    label = "log(T) < %.1f K" % val
    #    threshSets.append( {'propName':'temp_sfcold', 'comp':'lt', 'propThresh':val, 'label':label})

    # threshSets.append( {'propName':'sfr', 'comp':'gt', 'propThresh':1e-10, 'label':'SFR > 0'})
    # threshSets.append( {'propName':'nh', 'comp':'gt', 'propThresh':0.02, 'label':r'n$_{\rm H}$ > 0.02 cm$^{-3}$'})
    # threshSets.append( {'propName':'nh', 'comp':'gt', 'propThresh':0.1, 'label':r'n$_{\rm H}$ > 0.1 cm$^{-3}$'})
    # threshSets.append( {'propName':'nh', 'comp':'gt', 'propThresh':0.5, 'label':r'n$_{\rm H}$ > 0.5 cm$^{-3}$'})

    nBins1D = 100  # 1d histograms

    configs_2d = [
        "size-mass",
        "ncells-size",
        "size-dist",
        "ncells-dist",
        "dens-size",
        "vrad-dist",
        "vrad-size",
        "specj-dist",
        "specj-mass",
        "temp-size",
        "bmag-size",
        "beta-size",
        "sfr-size",
        "metal-size",
        "rcell1-size",
        "rcell2-size",
        "mg2_mass-size",
        "hi_mass-size",
        "metal-dist",
    ]

    # load
    data = []
    data_stack = []

    for sP in sPs:
        for i, th in enumerate(threshSets):
            if len(sPs) > 1:
                continue  # only multiple thresholds for a single sP
            objs, props = voronoiThresholdSegmentation(
                sP, haloID=haloID, propName=th["propName"], propThresh=th["propThresh"], propThreshComp=th["comp"]
            )

            # some common unit conversions
            values = _clump_values(sP, objs, props)

            data.append([objs, props, values])
            print(i, "prop = ", th["propName"], " ", th["comp"], " ", th["propThresh"], " tot objs = ", objs["count"])

        if stackHaloIDs is not None:
            # load all requested halos and combine, just for one threshold
            th = threshSets[2]  # 1e-8

            for sP in sPs:
                data_stack_loc = []

                for hID in stackHaloIDs:
                    print("load stack: ", sP.simName, hID)
                    objs, props = voronoiThresholdSegmentation(
                        sP, haloID=hID, propName=th["propName"], propThresh=th["propThresh"], propThreshComp=th["comp"]
                    )

                    values = _clump_values(sP, objs, props)
                    data_stack_loc.append([objs, props, values])

                data_stack.append(data_stack_loc)

    # tracer-analysis of accretion origin
    if trAnalysis:
        # load tracer catalogs for this halo
        sP = sPs[0]
        sP.haloInd = haloID  # specifies halo index in fullbox simulation

        print("Loading tracer data...")
        accTime = tracerEvo.accTime(sP)
        accMode = tracerEvo.accMode(sP)

        parIDs = tracerEvo.tracersMetaOffsets(sP, parIDs="gas")  # come first in concat'ed trIDs
        parIDs4 = tracerEvo.tracersMetaOffsets(sP, parIDs="stars")  # second
        parIDs5 = tracerEvo.tracersMetaOffsets(sP, parIDs="bhs")  # third

        assert accTime.size == parIDs.size + parIDs4.size + parIDs5.size
        assert accTime.size == accMode.size

        accTime = accTime[0 : parIDs.size]  # gas comes first
        accMode = accMode[0 : parIDs.size]

        # get child gas IDs in cloud
        print("Loading gas IDs and properties...")

        gasIDs = sP.snapshotSubset("gas", "ids", haloID=haloID)
        gasRad = sP.snapshotSubset("gas", "rad_rvir", haloID=haloID)
        gasTemp = sP.snapshotSubset("gas", "temp_log", haloID=haloID)

        clumpIDs = [1592, 3416, 3851, 430, 797, 2165, 2438, 4087]

        rr = [0.5, 1.0]  # radial restriction

        for i in range(len(clumpIDs) + 4):
            # single cloud analysis
            if i < len(clumpIDs):
                clumpID = clumpIDs[i]
                print("single cloud [%d]:" % clumpID)

                start_ind = objs["offsets"][clumpID]
                end_ind = objs["offsets"][clumpID] + objs["lengths"][clumpID]

                inds = objs["cell_inds"][start_ind:end_ind]

            # global 'all clouds' tracer analysis
            if i == len(clumpIDs) + 0:
                print("global all clouds (rad_slice):")

                inds = objs["cell_inds"]
                inds = inds[np.where((gasRad[inds] > rr[0]) & (gasRad[inds] <= rr[1]))]

            # global 'halo' tracer analysis: all gas cells >0.15<1.0 rvir
            if i == len(clumpIDs) + 1:
                print("global halo gas (rad_slice):")
                inds = np.where((gasRad > rr[0]) & (gasRad <= rr[1]))[0]

            # hot halo only
            if i == len(clumpIDs) + 2:
                print("global hot halo gas (rad_slice >5e5K):")
                inds = np.where((gasRad > rr[0]) & (gasRad <= rr[1]) & (gasTemp >= np.log10(5e5)))[0]

            # cold gas selection only
            if i == len(clumpIDs) + 3:
                print("global cold halo gas (rad_slice <1e5K):")
                inds = np.where((gasRad > rr[0]) & (gasRad <= rr[1]) & (gasTemp < 4.5))[0]

            # common: cross-match all gas-parent tracers in this halo with the target cells
            gasIDs_loc = gasIDs[inds]

            print(" matching [%d cells]..." % inds.size)
            _, trInds = match(gasIDs_loc, parIDs)

            accTime_loc = accTime[trInds]
            accMode_loc = accMode[trInds]

            print(" median acc redshift: ", np.nanmedian(accTime_loc))

            for accModeName, accModeVal in tracerEvo.ACCMODES.items():
                w = np.where(accMode_loc == accModeVal)
                frac = len(w[0]) / accMode_loc.size * 100
                print(" [%3d of %3d] tracers [%.3f%%] = %s" % (len(w[0]), accMode_loc.size, frac, accModeName))

        return

    # print cumulative mass by size statistics
    if 0:
        size = data[2][2]["size"]
        mass = data[2][2]["mass"]
        sort_inds = np.argsort(size)[::-1]  # descending
        size = size[sort_inds]
        mass = mass[sort_inds] / np.sum(mass)
        mass_cum = np.cumsum(mass)
        for mass_perc in [50, 90]:
            ww = np.where(mass_cum >= mass_perc)[0].min()
            print(mass_perc, size[ww])

    # C: 1D histograms of all properties (stacked), versus resolution
    for config in ["size"]:
        if len(sPs) == 1:
            continue  # skip for a single sP

        figsize_loc = (7, 5) if config == "size" else figsize
        fig = plt.figure(figsize=figsize_loc)
        ax = fig.add_subplot(111)

        lim = lims[config]
        ax.set_xlabel(labels[config])
        ax.set_ylabel("Number of Clouds")

        nBins = nBins1D if config != "ncells" else lim[1]

        if config == "size":  # paper figure
            ax.set_yscale("log")
            lim = [0, 6]  # pkpc
            nBins = nBins / 2

        ax.set_xlim(lim)

        binsize = (lim[1] - lim[0]) / nBins
        bins = np.linspace(lim[0] - binsize, lim[1] + binsize, nBins + 3)

        # stacked
        if stackHaloIDs is not None:
            valuesInd = 2
            for i, sP in enumerate(sPs):
                print(i, sP.simName)
                num_vals = np.sum([d[valuesInd][config].size for d in data_stack[i]])
                vals = np.zeros(num_vals, dtype=data_stack[i][0][valuesInd][config].dtype)
                offset = 0
                for d in data_stack[i]:
                    count = d[2][config].size
                    vals[offset : offset + count] = d[valuesInd][config][:]
                    offset += count

                yy, xx = np.histogram(vals, bins=bins)
                xx = xx[:-1] + binsize / 2  # mid
                yy = np.array(yy) * 2 / len(data_stack[i])  # mean*2

                (l,) = ax.plot(xx, yy, "-", drawstyle="steps-mid", label=sP.simName)

        # finish
        ymin = 1 if config == "size" else 0
        ax.set_ylim([ymin, ax.get_ylim()[1]])
        ax.legend()

        fig.savefig("clumpDemographics_sP%d_%s.pdf" % (len(sPs), config))
        plt.close(fig)

    if len(sPs) > 1:
        return  # skip remaining plots for multiple sPs

    # A: 1D histograms of all properties
    for config in lims.keys():
        fig, ax = plt.subplots()

        lim = lims[config]
        ax.set_xlabel(labels[config])
        ax.set_xlim(lim)
        ax.set_ylabel("Number of Clouds")

        nBins = nBins1D if config != "ncells" else lim[1]

        binsize = (lim[1] - lim[0]) / nBins
        bins = np.linspace(lim[0] - binsize, lim[1] + binsize, nBins + 3)

        for i, th in enumerate(threshSets):
            # load
            objs, props, values = data[i]

            vals = values[config]

            # histogram
            yy, xx = np.histogram(vals, bins=bins)
            xx = xx[:-1] + binsize / 2  # mid

            # label = '%s %s %s' % (th['dispName'], {'gt':'>','lt':'<'}[propThreshComp], propThresh)
            (l,) = ax.plot(xx, yy, "-", drawstyle="steps-mid", label=th["label"])
            ax.fill_between(xx, np.zeros(yy.size), yy, step="mid", color=l.get_color(), alpha=0.05)

        # stacked
        if stackHaloIDs is not None:
            valuesInd = 2
            num_vals = np.sum([d[valuesInd][config].size for d in data_stack])
            vals = np.zeros(num_vals, dtype=data_stack[0][valuesInd][config].dtype)
            offset = 0
            for d in data_stack:
                count = d[2][config].size
                vals[offset : offset + count] = d[valuesInd][config][:]
                offset += count

            yy, xx = np.histogram(vals, bins=bins)
            xx = xx[:-1] + binsize / 2  # mid
            yy = np.array(yy) * 2 / len(data_stack)  # mean*2

            (l,) = ax.plot(xx, yy, ":", drawstyle="steps-mid", color="black")

        # finish
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.legend()

        fig.savefig("clumpDemographics_%s_h%d_%s.pdf" % (sP.simName, haloID, config))
        plt.close(fig)

    # B: 2d xquant vs yquant plots
    for config in configs_2d:
        fig, ax = plt.subplots()

        yname, xname = config.split("-")

        ax.set_xlim(lims[xname])
        ax.set_ylim(lims[yname])
        ax.set_xlabel(labels[xname])
        ax.set_ylabel(labels[yname])

        for i, th in enumerate(threshSets):
            # load
            objs, props, values = data[i]

            xvals = values[xname]
            yvals = values[yname]

            if xvals.size == 1:
                ax.plot(xvals, yvals, "o", label=th["label"])
                continue

            # running median
            binSize = (lims[xname][1] - lims[xname][0]) / nBins1D
            if xname == "ncells":
                binSize = 1
            xm, ym, sm, pm = running_median(xvals, yvals, binSize=binSize, percs=percs)

            (l,) = ax.plot(xm, ym, "-", alpha=0.8, label=th["label"])
            if i in [0, len(threshSets) - 1]:
                ax.fill_between(xm, pm[0, :], pm[-1, :], facecolor=l.get_color(), alpha=0.2, interpolate=True)

        ax.legend(loc="lower right")

        fig.savefig("clumpDemographics_%s_h%d_%s.pdf" % (sP.simName, haloID, config))
        plt.close(fig)


def clumpTracerTracksLoad(sP, haloID, clumpID, sbNum=None, posOnly=False):
    """Load subbox time evolution tracks and analyze time evolution of clump cell/integral properties.

    If sbNum is None, then use fullbox snapshot (time spacing), otherwise use specified subbox.
    """
    saveFilename = sP.cachePath + "cache_clump_%s_%d-%d_sb%s.hdf5" % (sP.simName, haloID, clumpID, sbNum)

    if posOnly:
        saveFilename = saveFilename.replace(".hdf5", "_pos.hdf5")

    # check for cache existence
    if isfile(saveFilename):
        print("Loading [%s]..." % saveFilename)
        data = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                data[key] = f[key][()]

        return data

    # locate the target subhalo at z=0 via its descendant tree
    sP_z0 = sP.copy()
    sP_z0.setRedshift(0.0)

    subhaloID = sP.halo(haloID)["GroupFirstSub"]
    subMDB = sP.loadMDB(subhaloID)

    subhaloInd_z0 = np.where(subMDB["SnapNum"] == sP_z0.snap)[0]
    subhaloID_z0 = subMDB["SubfindID"][subhaloInd_z0][0]

    if sbNum is not None:
        # use subbox catalog
        sbCat = subboxSubhaloCat(sP_z0, sbNum)

        sbCatInd = np.where(sbCat["SubhaloIDs"] == subhaloID_z0)[0]
        subhaloCen = np.squeeze(sbCat["SubhaloPos"][sbCatInd, :, :])

        sP_tracks = sP.subboxSim(sbNum)
    else:
        # use fullbox merger tree
        subMPB = sP.loadMPB(subhaloID)
        subhaloCen = subMPB["SubhaloPos"]

        sP_tracks = sP

    # load segmentation at sP.redshift
    objs, props = voronoiThresholdSegmentation(
        sP, haloID=haloID, propName=thPropName, propThresh=thPropThresh, propThreshComp=thPropThreshComp
    )

    print("Selected clump [%d], properties:" % clumpID)

    for prop in props:
        print(" ", prop, props[prop][clumpID])

    offset = objs["offsets"][clumpID]
    length = objs["lengths"][clumpID]
    cell_inds = objs["cell_inds"][offset : offset + length]

    # load cell IDs
    cell_ids = sP.snapshotSubset("gas", "ids", haloID=haloID)[cell_inds]

    # load tracer tracks meta at this starting snapshot
    print("Load tracer meta and cross-matching...")
    tr_meta = globalAllTracersTimeEvo(sP_tracks, "meta")

    # cross-match IDs, get tracer catalog indices of the clump member gas cells
    _, inds_cat = match(cell_ids, tr_meta["ParentIDs"])
    indRange = [inds_cat.min(), inds_cat.max() + 1]
    inds_cat -= inds_cat.min()

    loadSize = (indRange[1] - indRange[0]) * 200 * 3 * 8 / 1024**3  # approx GB
    if loadSize > 10:
        print(" Load size [%.1f GB], skipping..." % loadSize)
        return None

    print("Found [%d] tracers in [%d] parent cells." % (inds_cat.size, cell_ids.size))

    # load other tracer property tracks
    trProps = ["hdens", "temp", "pos"]
    if sbNum is not None:
        trProps = ["hdens", "temp", "pos", "metal", "beta", "tcool", "sftime", "parent_indextype"]
    if posOnly:
        trProps = ["pos", "parent_indextype"]

    data = {}

    # loop over properties to load
    for prop in trProps:
        # load: backwards in time
        data_loc = globalAllTracersTimeEvo(sP_tracks, prop, indRange=indRange)

        if prop == "pos":
            # save snapshots/redshifts
            data["redshifts"] = data_loc["redshifts"]
            data["snaps"] = data_loc["snaps"]

        w_notdone = np.where(data_loc["done"] == 0)
        if len(w_notdone[0]):
            data_loc[prop][w_notdone] = np.nan

        data[prop] = data_loc[prop][:, inds_cat]  # subset within indRange

        # load: forwards in time (only for subboxes)
        if sbNum is not None:
            data_loc = globalAllTracersTimeEvo(sP_tracks, prop, indRange=indRange, toRedshift=sP.redshift - 0.1)

            if data_loc is not None:
                w_notdone = np.where(data_loc["done"] == 0)
                if len(w_notdone[0]):
                    data_loc[prop][w_notdone] = np.nan

                data[prop] = np.vstack((data[prop], data_loc[prop][:, inds_cat]))

            if prop == "pos":
                # append forward times
                data["redshifts"] = np.hstack((data["redshifts"], data_loc["redshifts"]))
                data["snaps"] = np.hstack((data["snaps"], data_loc["snaps"]))

        # if prop == 'temp': # old: 'temp' now linear, but may need for any old stored datafiles
        #    data[prop] = 10.0**data[prop] # remove log

    # data manipulation
    for prop in trProps:
        # time averages across all member cells
        data[prop + "_avg"] = np.nanmean(data[prop], axis=1)

    # count fractions in different parent types, and in wind-phase
    if sbNum is not None:
        for ptNum in [0, 4, 5]:
            mask = (data["parent_indextype"] >= ptNum * 1e11) & (data["parent_indextype"] < (ptNum + 1) * 1e11)
            data["parent_type_%d" % ptNum] = mask
            data["parent_frac_%d" % ptNum] = np.sum(mask, axis=1) / mask.shape[1]

        if not posOnly:
            with np.errstate(invalid="ignore"):
                # data['sftime'] is nan if parent type is not 4, so this the fraction of pt4 parents that are wind?
                data["parent_type_wind"] = data["sftime"] < 0.0
                data["parent_frac_wind"] = np.sum(data["parent_type_wind"], axis=1) / data["parent_type_wind"].shape[1]

    data["tage"] = sP.units.redshiftToAgeFlat(data["redshifts"]) * 1e3  # Gyr -> Myr
    data["dt"] = data["tage"] - data["tage"][0]
    data["dt"][0] -= 1e-6  # place starting time negative

    # derive center of clump and clump extent(s) from pos
    data["pos_avg"] = np.mean(data["pos"], axis=1)

    # derive distance of each tracer to center of clump, and clump 'size' as ~half mass radius
    nSnaps = data["pos"].shape[0]
    nTr = data["pos"].shape[1]

    data["rad"] = np.zeros((nSnaps, nTr), dtype="float32")
    data["dist"] = np.zeros((nSnaps, nTr), dtype="float32")
    data["pos_rel"] = np.zeros((nSnaps, nTr, 3), dtype="float32")
    data["size_maxseparation"] = np.zeros(nSnaps, dtype="float32")

    if not posOnly:
        maxTempsCold = [3e4, 1e5]
        data["size_maxsep_cold"] = np.zeros((nSnaps, len(maxTempsCold)), dtype="float32")

    if sbNum is not None:
        # sbCat fields are such that index = subbox snapnum (ascending), take directly
        SubhaloPos_trSnaps = subhaloCen[data["snaps"], :]
    else:
        # MPB fields are never guaranteed to be complete (and also descending), cross-match
        i1, i2 = match(subMPB["SnapNum"], data["snaps"])
        inds = np.zeros(data["snaps"].size, dtype="int32") - 1
        inds[i2] = i1  # unmatched snapshots will take last value
        SubhaloPos_trSnaps = subhaloCen[inds, :]

    for i in range(nSnaps):
        # distance of each tracer to center of clump
        xx = data["pos"][i, :, 0] - data["pos_avg"][i, 0]
        yy = data["pos"][i, :, 1] - data["pos_avg"][i, 1]
        zz = data["pos"][i, :, 2] - data["pos_avg"][i, 2]

        sP.correctPeriodicDistVecs(xx)
        sP.correctPeriodicDistVecs(yy)
        sP.correctPeriodicDistVecs(zz)

        data["rad"][i, :] = np.sqrt(xx**2 + yy**2 + zz**2)

        # distance of each tracer to center of halo
        data["pos_rel"][i, :, 0] = data["pos"][i, :, 0] - SubhaloPos_trSnaps[i, 0]
        data["pos_rel"][i, :, 1] = data["pos"][i, :, 1] - SubhaloPos_trSnaps[i, 1]
        data["pos_rel"][i, :, 2] = data["pos"][i, :, 2] - SubhaloPos_trSnaps[i, 2]

        sP.correctPeriodicDistVecs(data["pos_rel"])

        data["dist"][i, :] = np.sqrt(
            data["pos_rel"][i, :, 0] ** 2 + data["pos_rel"][i, :, 1] ** 2 + data["pos_rel"][i, :, 2] ** 2
        )

        # maximum pairwise distance between clump members
        data["size_maxseparation"][i], _, _ = sP.periodicPairwiseDists(data["pos"][i, :, :]).max()

        if not posOnly:
            for j, maxTempCold in enumerate(maxTempsCold):
                w_cold = np.where(data["temp"][i, :] < maxTempCold)
                data["size_maxsep_cold"][i, j], _, _ = sP.periodicPairwiseDists(
                    np.squeeze(data["pos"][i, w_cold, :])
                ).max()

    data["size_halfmassrad"] = np.median(data["rad"], axis=1)
    data["dist_rvir"] = data["dist"] / sP.halo(haloID)["Group_R_Crit200"]  # take constant
    data["pos_rel"] = sP.units.codeLengthToKpc(data["pos_rel"])

    # convert dist to dist_avg, i.e. radial distance of clump from interpolated halo center position
    data["dist_avg"] = np.mean(data["dist"], axis=1)
    data["pos_rel_avg"] = np.mean(data["pos_rel"], axis=1)
    data["dist_rvir_avg"] = np.mean(data["dist_rvir"], axis=1)

    # calculate medians as an alternative to means
    for key in list(data.keys()):
        if key + "_avg" in data:
            data[key + "_median"] = np.nanmedian(data[key], axis=1)

    # save cache
    with h5py.File(saveFilename, "w") as f:
        for key in data:
            f[key] = data[key]
    print("Saved [%s]." % saveFilename)

    return data


def clumpTracerTracks(sP, haloID, clumpID, sbNum=None, posOnly=False):
    """Analyze and plot time evolution of cold clump properties vs time.

    Intersect the LRG halo sample with the subbox catalogs, find which halos are available for high time resolution
    tracking, and then make our analysis and plots of the time evolution of clump cell/integral properties vs time.
    """
    labels = {
        "size_halfmassrad": "Clump Half-mass Radius [ kpc ]",
        "size_maxseparation": "Clump Size: Max Pairwise Separation [ kpc ]",
        "dist": "Halocentric Distance [ kpc ]",
        "dist_rvir": r"Halocentric Distance / r$_{\rm vir}$",
        "hdens": "Hydrogen Number Density [ log cm$^{-3}$ ]",
        "temp": "Temperature [ log K ]",
        "metal": "Metallicity [ log (not solar) ]",
        "parent_frac_wind": "Fraction of PT4 parents which are wind phase [ log ]",
        "tcool": "Cooling Time [ log Gyr ]",
        "beta": r"$\beta = \rm{P}_{\rm gas} / \rm{P}_{\rm B}$ [ log ]",
    }

    time_xlim = [-1000, 300] if sbNum is not None else [-3000, 100]  # [-500,300]
    lineAlpha = 0.05  # for individual tracers
    lineW = 1  # for individual tracers

    lims = {
        "temp": [3.8, 8.2] if sbNum else [1.8, 8.2],
        "hdens": [-3.5, 2.0] if sbNum else [-4.5, 1.0],
        "dist": [-20, 400] if sbNum else [-20, 1000],
    }  # [-20,300]

    cBack = "tab:blue"
    cForward = "tab:red"

    noForwardData = ["metal", "netcoolrate"]  # fields without tracer_tracks into the future

    circOpts = {
        "markeredgecolor": "white",
        "markerfacecolor": "None",
        "markersize": 10,
        "markeredgewidth": 2,
    }  # marking t=0

    # load
    data = clumpTracerTracksLoad(sP, haloID, clumpID, sbNum, posOnly=posOnly)

    if data is None:
        return

    # plot (A) - time series
    xx = data["dt"]

    w_back = np.where(xx < 0)
    w_forward = np.where(xx >= 0)

    for prop in labels.keys():
        if prop not in data:
            continue
        if posOnly and prop not in ["dist"]:
            continue
        print(" plot ", prop)

        fig, ax = plt.subplots()

        ax.set_xlabel("Time since $z=0.5$ [Myr]")
        ax.set_ylabel(labels[prop])
        ax.set_xlim(time_xlim)
        if prop in lims.keys():
            ax.set_ylim(lims[prop])
        ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

        logf = logZeroNaN if ("dist" not in prop and "coolrate" not in prop) else lambda x: x  # identity

        if prop + "_avg" in data:
            for i in range(data[prop].shape[1]):  # individuals
                yy = logf(data[prop][:, i])
                ax.plot(xx[w_back], yy[w_back], "-", lw=lineW, color="black", alpha=lineAlpha, zorder=0)
                if prop not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], "-", lw=lineW, color="black", alpha=lineAlpha, zorder=0)

            for i in [0, 1]:
                if i == 0:
                    yy = logf(data[prop + "_avg"])  # mean across member cells
                    s = "o-"
                if i == 1:
                    yy = logf(data[prop + "_median"])  # median across member cells
                    s = ":"

                if prop not in noForwardData:
                    label = "Cloud Mean ($t>0$)" if i == 0 else ""
                    ax.plot(xx[w_forward], yy[w_forward], s, color=cForward, label=label)
                ax.plot(xx[w_back], yy[w_back], s, color=cBack, label="Cloud Mean ($t<0$)" if i == 0 else "")
            if prop not in noForwardData and len(w_forward[0]):
                ax.plot(xx[w_forward][0], yy[w_forward][0], "o", **circOpts)

        else:
            yy = logf(data[prop])  # quantity is 1 number per snapshot
            if prop not in noForwardData:
                (l2,) = ax.plot(xx[w_forward], yy[w_forward], "o-", color=cForward, label="($t>0$)")
            (l1,) = ax.plot(xx[w_back], yy[w_back], "o-", color=cBack, label="($t<0)$")

            if prop == "size_maxseparation" and not posOnly:  # add _cold
                # yy = logf(data['size_maxsep_cold'][:,0])
                # ax.plot(xx[w_forward], yy[w_forward], ':', color=l2.get_color(), label='log(T) < 5.0 ($t>0$)')
                # ax.plot(xx[w_back], yy[w_back], ':', color=l1.get_color(), label='log(T) < 5.0 ($t<0$)')
                yy = logf(data["size_maxsep_cold"][:, 1])
                ax.plot(xx[w_forward], yy[w_forward], "--", color=cForward, label="T < 30,000K ($t>0$)")
                ax.plot(xx[w_back], yy[w_back], "--", color=cBack, label="T < 30,000K ($t<0$)")

        ax.legend(loc="upper left" if prop == "temp" else "best")
        fig.savefig("clumpEvo_%s_h%d_clumpID=%d_sb%s_%s.pdf" % (sP.simName, haloID, clumpID, sbNum, prop))
        plt.close(fig)

    # plot (B) - spatial tracks
    axes = [0, 1]
    prop = "pos_rel"

    if prop in data:
        fig, ax = plt.subplots()
        ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized
        aspect = ax.get_window_extent().height / ax.get_window_extent().width

        ax.set_xlabel(r"$\Delta$ %s [kpc]" % ["x", "y", "z"][axes[0]])
        ax.set_ylabel(r"$\Delta$ %s [kpc]" % ["x", "y", "z"][axes[1]])

        xylim = np.array([data[prop][:, :, axes].min(), data[prop][:, :, axes].max()]) * 0.8
        ax.set_xlim(xylim)
        ax.set_ylim(xylim * aspect)  # ax is non-square, so make limits reflect the correct aspect ratio

        for i in range(data[prop].shape[1]):  # individuals
            xx = data[prop][:, i, axes[0]]
            yy = data[prop][:, i, axes[1]]
            ax.plot(xx[w_forward], yy[w_forward], "-", color=cForward, lw=lineW, alpha=lineAlpha, zorder=0)
            ax.plot(xx[w_back], yy[w_back], "-", color=cBack, lw=lineW, alpha=lineAlpha, zorder=0)

        xx = data[prop + "_avg"][:, axes[0]]
        yy = data[prop + "_avg"][:, axes[1]]  # mean across member cells
        ax.plot([0.0, 0.0], "o", color="black")

        fig.savefig("clumpEvo_%s_h%d_clumpID=%d_sb%s_%s.pdf" % (sP.simName, haloID, clumpID, sbNum, prop))
        plt.close(fig)

    if posOnly:
        return

    # plot (C) - phase diagram with time track
    for xval in ["hdens", "temp", "dist_rvir"]:
        for yval in labels.keys():
            if xval == yval:
                continue
            if yval not in data:
                continue
            print(" plot ", xval, yval)

            fig, ax = plt.subplots()
            ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized

            ax.set_xlabel(labels[xval])
            ax.set_ylabel(labels[yval])

            logx = logZeroNaN if (("dist" not in xval) and ("coolrate" not in xval)) else lambda x: x  # identity
            logy = logZeroNaN if (("dist" not in yval) and ("coolrate" not in yval)) else lambda x: x  # identity

            if yval + "_avg" in data:
                for i in range(data[xval].shape[1]):  # individuals
                    xx = logx(data[xval][:, i])
                    yy = logy(data[yval][:, i])
                    ax.plot(xx[w_back], yy[w_back], "-", lw=lineW, color="black", alpha=lineAlpha, zorder=0)
                    if xval not in noForwardData and yval not in noForwardData:
                        ax.plot(xx[w_forward], yy[w_forward], "-", lw=lineW, color="black", alpha=lineAlpha, zorder=0)

                xx = logx(data[xval + "_median"])  # mean across member cells
                yy = logy(data[yval + "_median"])
                if xval not in noForwardData and yval not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], "o-", color=cForward, label="Cloud Mean ($t>0$)")
                ax.plot(xx[w_back], yy[w_back], "o-", color=cBack, label="Cloud Mean ($t<0$)")
                if xval not in noForwardData and yval not in noForwardData and len(w_forward[0]):
                    ax.plot(xx[w_forward][0], yy[w_forward][0], "o", **circOpts)
                ax.legend()
            else:
                xx = (
                    logx(data[xval]) if data[xval].ndim == 1 else logx(data[xval + "_avg"])
                )  # quantity is 1 number per snapshot
                yy = logy(data[yval])
                ax.plot(xx[w_back], yy[w_back], "o-", color=cBack)
                if xval not in noForwardData and yval not in noForwardData:
                    ax.plot(xx[w_forward], yy[w_forward], "o-", color=cForward)

            fig.savefig(
                "clumpEvo_%s_h%d_clumpID=%d_sb%s_x=%s_y=%s.pdf" % (sP.simName, haloID, clumpID, sbNum, xval, yval)
            )
            plt.close(fig)

    # plot (D) - spatial distribution at each snapshot
    axes = [1, 2]

    for i in range(15):  #  data["snaps"][0:15]
        fig, ax = plt.subplots()
        ax.set_rasterization_zorder(1)  # elements below z=1 are rasterized
        aspect = ax.get_window_extent().height / ax.get_window_extent().width

        ax.set_xlabel(r"$\Delta$ %s [kpc]" % ["x", "y", "z"][axes[0]])
        ax.set_ylabel(r"$\Delta$ %s [kpc]" % ["x", "y", "z"][axes[1]])

        # shrinking center, relative coordinates
        xx = data["pos_rel"][i, :, axes[0]]
        yy = data["pos_rel"][i, :, axes[1]]
        cen = shrinking_center(data["pos_rel"][i, :, :], sP.boxSize)

        xx -= cen[axes[0]]
        yy -= cen[axes[1]]

        xylim = np.array([np.vstack((xx, yy)).min(), np.vstack((xx, yy)).max()]) * 1.2
        # xylim = np.clip(xylim, 10, np.inf)
        print(i, xylim)
        ax.set_xlim(xylim)
        ax.set_ylim(xylim * aspect)  # ax is non-square, so make limits reflect the correct aspect ratio

        # plot
        ax.plot(xx, yy, "o", markersize=2, linestyle="None", label=r"$\Delta t$ = %.3f Myr" % data["dt"][i])

        ax.plot([0.0, 0.0], ax.get_ylim(), "-", color="#777777", alpha=0.4)
        ax.plot(ax.get_xlim(), [0.0, 0.0], "-", color="#777777", alpha=0.4)
        ax.legend()

        fig.savefig(
            "clumpEvo_%s_h%d_clumpID=%d_sb%s_axes%d-%d_xyz%02d.png"
            % (sP.simName, haloID, clumpID, sbNum, axes[0], axes[1], i)
        )
        plt.close(fig)


def clumpPropertiesVsHaloMass(sPs):
    """Run segmentation on a flat mass-selection of halos, plot clump properties / abundance vs halo mass."""
    from temet.vis.halo import selectHalosFromMassBins

    # limits tailored to resolution convergence (showing all runs)
    lims = {
        "size": [0, 15.0],  # linear pkpc
        "mass": [6.0, 11.0],  # log msun
        "ncells": [0, 200],  # linear
        "dist": [0, 500],  # linear pkpc
        "dens": [-2.5, 0.0],  # log cc
        "temp": [3.6, 4.4],  # log K
        "bmag": [-0.5, 1.5],  # log G
        "beta": [-1.0, 1.0],  # log
        "sfr": [0, 0.1],  # linear msun/yr
        "metal": [-1.4, 0.4],  # log solar
        "rcell1": [0, 3500],  # linear parsec
        "rcell2": [0, 2000],  # linear parsec
        "number": [0.7, 2e4],  # linear
        "mg2_mass": [1.5, 7.0],  # log msun
        "hi_mass": [4.0, 11.0],
    }  # log msun

    # config
    minMaxHaloMass = [11.0, 14.0]
    numPerBin = 10
    xQuant = "mhalo_200_log"
    xlabel = r"Halo Mass [ log M$_{\rm sun}$ ]"
    xlim = [10.95, 14.05]

    minCellsPerClump = 10

    # halo binning config
    binSize = 0.1
    numMassBins = int((minMaxHaloMass[1] - minMaxHaloMass[0]) / binSize) + 1
    bins = [[x + 0.0, x + binSize] for x in np.linspace(minMaxHaloMass[0], minMaxHaloMass[1], numMassBins)]

    # loop over simulations and load
    x_vals = []
    clump_props = []

    for sP in sPs:
        # make halo selection
        print("load: ", sP.simName)
        haloIDs = selectHalosFromMassBins(sP, bins, numPerBin, "random")
        haloIDs = np.hstack(list(haloIDs)).astype("int32")

        # allocate
        locProps = {}
        for prop in labels.keys():
            locProps[prop] = np.zeros(haloIDs.size, dtype="float32")
        locProps["number"] = np.zeros(haloIDs.size, dtype="int32")
        locProps["number2"] = np.zeros(haloIDs.size, dtype="int32")

        # load/create segmentations, and accumulate mean properties per halo
        for i, haloID in enumerate(haloIDs):
            objs, props = voronoiThresholdSegmentation(
                sP, haloID=haloID, propName=thPropName, propThresh=thPropThresh, propThreshComp=thPropThreshComp
            )

            if objs["count"] == 0:
                continue

            values = _clump_values(sP, objs, props)

            w = np.where(objs["lengths"] >= minCellsPerClump)[0]
            locProps["number"][i] = len(w)
            w = np.where(values["size"] >= 2.0)[0]
            locProps["number2"][i] = len(w)

            for prop in labels.keys():
                if prop in ["number", "vrad", "specj"]:
                    continue
                locProps[prop][i] = np.median(values[prop][w])

        # load x-quant and make median
        x_vals_loc, x_label, minMax, takeLog = sP.simSubhaloQuantity(xQuant)
        if takeLog:
            x_vals_loc = np.log10(x_vals_loc)

        x_vals_loc = x_vals_loc[sP.halos("GroupFirstSub")[haloIDs]]

        # save
        x_vals.append(x_vals_loc)
        clump_props.append(locProps)

    # loop over clump properties to plot
    for prop in labels.keys():
        print(prop)

        # plot
        fig, ax = plt.subplots()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(labels[prop])
        ax.set_xlim(xlim)
        if prop in lims.keys():
            ax.set_ylim(lims[prop])
        if prop == "number":
            ax.set_yscale("log")
        if prop == "beta":
            ax.plot(ax.get_xlim(), [0.0, 0.0], ":", color="#555555", alpha=0.6)

        # loop over runs
        for i, sP in enumerate(sPs):
            # calculate median
            xm, ym, sm, pm = running_median(
                x_vals[i], clump_props[i][prop], binSize=binSize * 2, percs=percs, minNumPerBin=3
            )
            if xm.size > sKn:
                ym = savgol_filter(ym, sKn, sKo)
                sm = savgol_filter(sm, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)

            # plot individual halo markers, median line, and scatter band
            (l,) = ax.plot(x_vals[i], clump_props[i][prop], marker="o", alpha=0.8, linestyle="None")
            ax.fill_between(xm, pm[0, :], pm[-1, :], color=l.get_color(), alpha=0.2)
            ax.plot(xm, ym, "-", color=l.get_color(), label=sP.simName)

            # alternate number count (above a minimum size of 2pkpc)
            if prop == "number":
                xm, ym, sm, pm = running_median(
                    x_vals[i], clump_props[i]["number2"], binSize=binSize * 2, percs=percs, minNumPerBin=3
                )
                if xm.size > sKn:
                    ym = savgol_filter(ym, sKn, sKo)
                    sm = savgol_filter(sm, sKn, sKo)

                ax.plot(xm, ym, "--", color=l.get_color())

        # finish plot
        ax.legend()

        fig.savefig("clumps_%s_vs_%s_sP%d_%d_min%d.pdf" % (prop, xQuant, len(sPs), sP.snap, minCellsPerClump))
        plt.close(fig)


def clumpRadialProfiles(sP, haloID, selections, norm=False):
    """Compute and plot clump-centric radial profiles, for all clumps satisfying the selection in this halo."""
    xlim = [0.0, 10.0]
    xlabel = "Cloud-centric Distance [ pkpc ]"

    partType = "gas"
    props = [
        "temp_sfcold",
        "hdens",
        "vrel",
        "sfr",
        "entropy",
        "bmag_ug",
        "beta",
        "z_solar",
        "Mg II numdens",
        "cellsize_kpc",
        "pres_ratio",
        "p_b",
        "p_gas",
        "p_tot",
        "tcool",
        "tcool_tff",
        "nHI_GK",
        "H I numdens",
    ]
    nBins = 50
    cenfield = "cen_propwt"  # 'cen', 'cen_denswt', 'cen_propwt' (MgII weighted CoM)

    bins = np.linspace(xlim[0], xlim[1], nBins)  # edges, including inner and outer boundary
    bin_cens = bins[:-1] + (xlim[1] - xlim[0]) / nBins / 2
    maxAllocNumElem = int(1e9)  # one billion entries, 8 bytes per, 8 GB

    if sP.snap != 67:
        raise Exception("Should convert bin_cens to pkpc, check!.")

    if norm:
        props = ["z_solar"]
        assert len(props) == 1  # otherwise generalize

    # load segmentation
    objs, obj_props = voronoiThresholdSegmentation(
        sP, haloID=haloID, propName=thPropName, propThresh=thPropThresh, propThreshComp=thPropThreshComp
    )

    values = _clump_values(sP, objs, obj_props)

    # loop over selections (add each to final plot)
    results = []

    for selection in selections:
        selStr = "-".join(["%s-%g-%g" % (key, bound[0], bound[1]) for key, bound in selection.items()])
        if xlim[1] != 5.0:
            selStr += "_xmax=%d" % xlim[1]
        if norm:
            selStr += "_normT5"
        saveFilename = sP.cachePath + "cache_clumpprofs_%s_%d_%s_props%d.hdf5" % (
            sP.simName,
            haloID,
            selStr,
            len(props),
        )

        result = {}

        # check for cache existence
        if isfile(saveFilename):
            print("Loading [%s]..." % saveFilename)
            with h5py.File(saveFilename, "r") as f:
                for key in f:
                    result[key] = f[key][()]
            results.append(result)
            continue

        # calculate now: first apply selection
        mask = np.zeros(objs["lengths"].size, dtype="int16")

        for sel, bounds in selection.items():
            print(sel, bounds)
            w = np.where((values[sel] >= bounds[0]) & (values[sel] < bounds[1]))
            mask[w] = 1

        # locate accepted clouds
        w_cloud = np.where(mask == 1)

        cloud_pos = obj_props[cenfield][w_cloud]
        nClouds = len(w_cloud[0])

        print("Processing [%d] clouds..." % len(w_cloud[0]))

        # allocate (per particle dist)
        if not norm:
            dists = np.zeros(maxAllocNumElem, dtype="float32")
            p_inds = np.zeros(maxAllocNumElem, dtype="int32")
            c_inds = np.zeros(maxAllocNumElem, dtype="int32")
            offset = 0
        else:
            prop = sP.snapshotSubset(partType, props[0], haloID=haloID)
            temp = sP.snapshotSubset(partType, "temp_log", haloID=haloID)

            profs = np.zeros((nBins - 1, nClouds), dtype="float32")
            profs.fill(np.nan)

            assert percs[1] == 50  # we only do median

        # load fof-scope particle data
        pos = sP.snapshotSubset(partType, "pos", haloID=haloID)

        # loop over clouds
        for j in range(nClouds):
            # distances to all particles, accumulate
            if j % 10 == 0:
                print("dists: ", j)
            dists_loc = sP.periodicDists(cloud_pos[j, :], pos)

            w = np.where(dists_loc <= xlim[1])

            if norm:
                # compute profile per cloud, normalize to its value at the largest distances considered
                prop_loc = prop[w]
                temp_loc = temp[w]
                dists_loc = dists_loc[w]

                # restrict to hot gas beyond size
                w_zero = np.where((dists_loc > bounds[1]) & (temp_loc < 5.0))
                dists_loc[w_zero] = xlim[0] - 1  # move outside any bin

                # profile
                prof, _, _ = binned_statistic(dists_loc, prop_loc, "mean", bins=nBins - 1, range=[xlim])
                norm_val = np.mean(prof[-int(nBins / 10) :])

                profs[:, j] = prof / norm_val
            else:
                # save distances and indices, we will compute mean stacked profiles all together later
                dists[offset : offset + len(w[0])] = dists_loc[w]
                p_inds[offset : offset + len(w[0])] = w[0]
                c_inds[offset : offset + len(w[0])] = w_cloud[0][j]

                offset += len(w[0])

        if norm:
            # save and skip the rest
            result[props[0]] = np.zeros((nBins - 1, len(percs)), dtype="float32")
            result[props[0]].fill(np.nan)

            result[props[0]][:, 1] = np.nanmedian(profs, axis=1)

        else:
            # truncate
            dists = dists[0:offset]
            p_inds = p_inds[0:offset]
            c_inds = c_inds[0:offset]

            # loop over each requested property
            for prop in props:
                # allocate
                print(prop)
                result[prop] = np.zeros((nBins - 1, len(percs)), dtype="float32")
                result[prop].fill(np.nan)

                # load particle data, and sort on distance
                vals = sP.snapshotSubset(partType, prop, haloID=haloID)
                vals = vals[p_inds]

                if prop == "vrel":
                    # use this to compute radial velocity with respect to the cloud
                    refVel = obj_props["vrel"][c_inds]
                    refPos = obj_props[cenfield][c_inds]
                    # sP.units.particleRadialVelInKmS(pos, vel, refPos, refVel)

                    loc_pos = sP.snapshotSubset(partType, "pos", haloID=haloID)
                    loc_pos = loc_pos[p_inds]

                    # calculate position, relative to subhalo center (pkpc)
                    for i in range(3):
                        loc_pos[:, i] -= refPos[:, i]

                    loc_pos = sP.units.codeLengthToKpc(loc_pos)
                    rad = np.linalg.norm(loc_pos, 2, axis=1)

                    # correct velocities for cloud CM motion
                    for i in range(3):
                        vals[:, i] -= refVel[:, i]

                    # overwrite vals with radial velocity (km/s), negative=inwards
                    vals = (vals[:, 0] * loc_pos[:, 0] + vals[:, 1] * loc_pos[:, 1] + vals[:, 2] * loc_pos[:, 2]) / rad

                # binned statistic, and stamp
                for i in range(nBins - 1):
                    w = np.where((dists > bins[i]) & (dists <= bins[i + 1]))
                    if len(w[0]) == 0:
                        continue
                    result[prop][i, :] = np.nanpercentile(vals[w], percs)

        # save
        with h5py.File(saveFilename, "w") as f:
            for key in result:
                f[key] = result[key]
        print("Saved: [%s]" % saveFilename)

        results.append(result)

    # plot
    for prop in props:
        print("plot: ", prop)

        fig = plt.figure(figsize=figsize_sm)
        ax = fig.add_subplot(111)

        ylabel, ylim, ylog = sP.simParticleQuantity("gas", prop)

        if prop == "vrel":
            ylabel = "Local Radial Velocity [ km/s ]"

        if norm:
            ylog = False
            ylabel = r"Z$_{\rm gas}$ / Z$_{\rm gas,10kpc}$ [linear]"

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)
        # ax.set_ylim(ylim)
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if prop in ["beta", "pres_ratio", "tcool_tff", "vrel"]:  #'z_solar',
            ax.plot(ax.get_xlim(), [0.0, 0.0], ":", color="#555555", alpha=0.6)
        if prop == "cellsize_kpc":
            ax.plot(ax.get_xlim(), np.log10([0.5, 0.5]), ":", color="#555555", alpha=0.6)
        if prop in ["z_solar"] and norm:
            ax.plot(ax.get_xlim(), [1.0, 1.0], ":", color="#555555", alpha=0.6)

        yy_txt = []

        for i, selection in enumerate(selections):
            if ylog:
                yy = logZeroNaN(results[i][prop])
            else:
                yy = results[i][prop]

            if yy.size > sKn:
                yy = savgol_filter(yy, sKn + 4, sKo, axis=0)

            yy_txt.append(yy)

            # plot median and percentile band
            label = ", ".join(["%.1f < %s < %.1f" % (bound[0], key, bound[1]) for key, bound in selection.items()])
            if len(selection) == 1 and "size" in selection:
                label = r"r$_{\rm cloud}$ = %.1f kpc" % selection["size"][0]

            (l,) = ax.plot(bin_cens, yy[:, 1], "-", label=label)

            # if i in [3]:
            #    ax.fill_between(bin_cens, yy[:,0], yy[:,-1], color=l.get_color(), interpolate=True, alpha=0.1)

            if prop == "p_tot":
                # compute a 'ram pressure' of the radial inflow and add it to the total pressure
                dens = results[i]["hdens"] * sP.units.mass_proton  # g/cm^3
                vel = results[i]["vrel"] * sP.units.km_in_cm  # cm/s
                P_ram = dens * vel**2  # g/cm/s^2 = erg/cm^3
                P_ram /= sP.units.boltzmann  # K/cm^3

                yy2 = logZeroNaN(results[i]["p_tot"] + P_ram)  # sum radial profiles together
                yy2 = savgol_filter(yy2, sKn + 4, sKo, axis=0)
                ax.plot(bin_cens, yy2[:, 1], ":", color=l.get_color())

        # finish plot
        ax.legend()
        fig.savefig("clumps_radprof_%s_%s_%d_%d.pdf" % (prop, sP.simName, sP.snap, haloID))
        plt.close(fig)

        # write data file
        filename = "fig10_%s_z%.1f_h%d_%s.txt" % (sP.simName, sP.redshift, haloID, prop)

        out = "# Nelson+ (2020) http://arxiv.org/abs/xxxx.xxxxx\n"
        out += "# Figure 10: stacked radial profiles of cloud property [%s = %s]\n" % (prop, ylabel)
        out += "# %s (z = %.1f) haloID = %d (nClouds stacked = 924, 764, 303, 133)\n" % (
            sP.simName,
            sP.redshift,
            haloID,
        )
        out += "radius [pkpc]"
        for selection in selections:
            out += ", quantity [r_cloud=%.1f] p%d, p%d, p%d" % (selection["size"][0], percs[1], percs[0], percs[2])
        out += "\n\n"
        for i, bin_cen in enumerate(bin_cens):
            out += "%.2f" % bin_cen
            for j in range(len(selections)):
                out += " %.3f %.3f %.3f" % (yy_txt[j][i, 1], yy_txt[j][i, 0], yy_txt[j][i, 2])
            out += "\n"

        with open(filename, "w") as f:
            f.write(out)


def paperPlots():
    """Produce all papers for the LRG-MgII (small-scale CGM structure) TNG50 paper."""
    haloMassBins = [[12.3, 12.7], [12.8, 13.2], [13.2, 14.0]]
    redshift = 0.5  # default for analysis

    TNG100 = simParams(res=1820, run="tng", redshift=redshift)
    TNG50 = simParams(res=2160, run="tng", redshift=redshift)
    TNG50_2 = simParams(res=1080, run="tng", redshift=redshift)
    TNG50_3 = simParams(res=540, run="tng", redshift=redshift)
    TNG50_4 = simParams(res=270, run="tng", redshift=redshift)

    def _get_halo_ids(sP_loc, bin_inds=None, subhaloIDs=False):
        """Load and return the halo IDs in each haloMassBin. If subhaloIDs, then return these rather than haloIDs."""
        mhalo = sP_loc.groupCat(fieldsSubhalos=["mhalo_200_log"])

        if subhaloIDs:
            grnr = np.arange(sP_loc.numHalos)  # identity
        else:
            grnr = sP_loc.groupCat(fieldsSubhalos=["SubhaloGrNr"])

        haloIDs = []

        if bin_inds is None:
            # return a list, one entry per haloMassBin
            for haloMassBin in haloMassBins:
                with np.errstate(invalid="ignore"):
                    inds = np.where((mhalo > haloMassBin[0]) & (mhalo < haloMassBin[1]))[0]

                haloIDs.append(grnr[inds])
        else:
            # return a single array, all haloIDs in the specified haloMassBin(s)
            for haloMassBin in [haloMassBins[i] for i in bin_inds]:
                with np.errstate(invalid="ignore"):
                    inds = np.where((mhalo > haloMassBin[0]) & (mhalo < haloMassBin[1]))[0]
                haloIDs += list(grnr[inds])
            haloIDs = np.array(haloIDs, dtype="int32")

        return haloIDs

    # --- halo scale ---

    # figure 1 - cgm resolution
    if 0:
        sPs = [TNG50, TNG50_2, TNG50_3]
        cenSatSelect = "cen"

        simNames = "_".join([sP.simName for sP in sPs])

        for radRelToVirRad in [True]:  # ,False]:
            saveName = "resolution_profiles_%s_z%02d_%s_rvir%d.pdf" % (
                simNames,
                redshift * 10,
                cenSatSelect,
                radRelToVirRad,
            )

            radialResolutionProfiles(
                sPs,
                saveName,
                redshift=redshift,
                radRelToVirRad=radRelToVirRad,
                cenSatSelect="cen",
                haloMassBins=haloMassBins,
            )

    # figs 2, 3, 4 - vis (single halo/gallery, MgII/HI/tcool_tff/etc)
    if 0:
        sP = TNG50
        haloIDs = _get_halo_ids(sP)[2]

        # gas metallicity, N_MgII, N_HI, stellar light
        for conf in [8, 9, 10, 11]:
            lrgHaloVisualization(sP, [1], conf=conf, gallery=False, globalDepth=False)
        for conf in [1, 2, 3, 4]:
            lrgHaloVisualization(sP, haloIDs, conf=conf, gallery=True)
            lrgHaloVisualization(sP, haloIDs, conf=conf, gallery=False)

    # fig 5a: bound ion mass as a function of halo mass
    if 0:
        TNG50_z0 = simParams(run="tng50-1", redshift=0.0)
        TNG50_z1 = simParams(run="tng50-1", redshift=1.0)
        sPs = [TNG50, TNG50_z0, TNG50_z1]
        css = "cen"
        ions = ["HI_GK", "AllGas_Metal", "AllGas_Mg", "MgII"]

        for vsHaloMass in [True]:  # [True,False]:
            massStr = "%smass" % ["stellar", "halo"][vsHaloMass]

            saveName = "ions_masses_vs_%s_%s_%d_%s.pdf" % (
                massStr,
                css,
                sPs[0].snap,
                "_".join([sP.simName for sP in sPs]),
            )
            totalIonMassVsHaloMass(sPs, saveName, ions=ions, cenSatSelect=css, vsHaloMass=vsHaloMass, colorOff=0)

    # fig 5b: radial profiles
    if 0:
        sPs = [TNG50]
        ions = ["MgII"]  # 'HI_GK'
        projSpecs = ["2Dz_6Mpc", "3D"]

        simNames = "_".join([sP.simName for sP in sPs])

        for massDensity in [True, False]:
            for radRelToVirRad in [True, False]:
                for projDim in projSpecs:
                    saveName = "radprofiles_%s_%s_%s_%d_rho%d_rvir%d.pdf" % (
                        projDim,
                        "-".join(ions),
                        simNames,
                        sPs[0].snap,
                        massDensity,
                        radRelToVirRad,
                    )
                    stackedRadialProfiles(
                        sPs,
                        saveName,
                        redshift=sPs[0].redshift,
                        ions=ions,
                        massDensity=massDensity,
                        radRelToVirRad=radRelToVirRad,
                        cenSatSelect="cen",
                        projDim=projDim,
                        haloMassBins=[[11.4, 11.6], [11.9, 12.1], [12.4, 12.6], [12.8, 13.2], [13.2, 13.8]],
                        combine2Halo=True,
                        median=True,
                    )

    # figure 6a - cgm gas density/temp/pressure 1D PDFs
    if 0:
        sP = TNG50
        qRestrictions = [["rad_rvir", 0.1, 1.0]]  # 0.15<r/rvir<1

        if 0:
            ptProperty = "nh"
            xlim = [-6.0, 0.0]
        if 1:
            ptProperty = "temp_sfcold"
            xlim = [3.8, 7.6]
        if 0:
            ptProperty = "gas_pres"
            qRestrictions.append(["sfr", 0.0, 0.0])  # non-eEOS
            xlim = [0.0, 5.0]

        # all halos in two most massive bins
        subhaloIDs = _get_halo_ids(sP, bin_inds=[1, 2], subhaloIDs=True)

        # create density PDF of gas in radRangeRvir
        snapshot.histogram1d(
            [sP],
            ptType="gas",
            ptProperty=ptProperty,
            xlim=xlim,
            nBins=200,
            medianPDF=True,
            qRestrictions=qRestrictions,
            subhaloIDs=[subhaloIDs],
            ctName="plasma",
            ctProp="mhalo_200_log",
            legend=False,
            colorbar=True,
        )

    # fig 6b - cgm gas (n,T) 2D phase diagrams
    if 0:
        sP = TNG50
        qRestrictions = [["rad_rvir", 0.1, 1.0]]

        xQuant = "nh"
        yQuant = "temp"
        xlim = [-5.5, -1.0]
        ylim = [3.8, 7.8]

        haloIDs = [8]  # single example
        # haloIDs = _get_halo_ids(sP)[1] # all the halos in a mass bin

        pdf = PdfPages("phase2d_%s_%d_h%d_%s_%s.pdf" % (sP.simName, sP.snap, haloIDs[0], xQuant, yQuant))

        snapshot.phaseSpace2d(
            sP,
            partType="gas",
            xQuant=xQuant,
            yQuant=yQuant,
            weights=["mass"],
            meancolors=None,  # 'tcool_tff'
            haloIDs=haloIDs,
            pdf=pdf,
            xlim=xlim,
            ylim=ylim,
            clim=[-4.0, 0.0],
            nBins=None,
            contours=[2.5, 3.0, 3.5],  # log K/cm^3
            contourQuant="gas_pres",
            qRestrictions=qRestrictions,
        )
        pdf.close()

    # fig 11a: radial profiles of tcool, tff, tcool/tff (hot gas only)
    if 0:
        sP = TNG50
        subIDs = _get_halo_ids(sP, bin_inds=[2, 1], subhaloIDs=True)

        snapshot.stackedlProfiles1d(
            [sP],
            ptType="gas",
            ptProperty="tcool_tff",
            op="median",
            subhaloIDs=[subIDs],
            xlim=[1.0, 3.0],
            ylim=[-0.3, 2.5],
            plotIndiv=True,
            ptRestrictions={"temp_log": ["gt", 5.5]},
            ctName="plasma",
            ctProp="mhalo_200_log",
            colorbar=True,
        )

        snapshot.stackedProfiles1d(
            [sP],
            ptType="gas",
            ptProperty="entr",
            op="median",
            subhaloIDs=[subIDs],
            xlim=[1.0, 3.0],
            ylim=[7.2, 9.05],
            plotIndiv=True,
            ptRestrictions={"temp_log": ["gt", 5.5]},
            ctName="plasma",
            ctProp="mhalo_200_log",
            figsize=[7, 5],
        )  # inset

    # fig 11b - 'radial profile' of tcool (2D)
    if 0:
        sP = TNG50

        xQuant = "rad_kpc"
        yQuant = "tcool_tff"  #'tff' #'tcool'
        xlim = [1.0, 3.0]  # 2.7]
        ylim = [-3.3, 3.0]  # [-5.6, 2.9]
        contours = [-2.0, -1.0, -0.3]

        haloID = 8  # 8 # single example, 0, 1, 2, 3, 4, 5, 6, 7, 11, 13

        for i in [0, 1, 2]:
            if i == 0:
                weights = ["mass"]
                meancolors = None
                clim = [-4.0, 0.0]
                normColMax = True
                ctName = "viridis"
            if i == 1:
                weights = None
                meancolors = ["temp"]
                clim = [4.0, 7.0]
                normColMax = False
                ctName = "haline"
            if i == 2:
                weights = None
                meancolors = ["vrad"]
                clim = [-300, 300]
                normColMax = False
                ctName = "curl"

            snapshot.phaseSpace2d(
                sP,
                partType="gas",
                xQuant=xQuant,
                yQuant=yQuant,
                weights=weights,
                meancolors=meancolors,
                haloIDs=[haloID],
                xlim=xlim,
                ylim=ylim,
                clim=clim,
                nBins=200,
                ctName=ctName,
                contours=contours,
                contourQuant="mass",
                normColMax=normColMax,
                normContourQuantColMax=True,
                smoothSigma=1.0,
                colorEmpty=False,
            )

    # explore: 2pcf (todo: tpcf of single halo (0.15 < r/rvir < 1.0)
    if 0:
        sPs = [TNG50]  # [TNG100, TNG300]
        ions = ["MgII"]  # ,'Mg','gas']

        for order in [0, 1, 2]:
            saveName = "tpcf_order%d_%s_%s_z%02d.pdf" % (
                order,
                "-".join(ions),
                "_".join([sP.simName for sP in sPs]),
                redshift,
            )

            ionTwoPointCorrelation(sPs, saveName, ions=ions, redshift=redshift, order=order, colorOff=2)

    # explore: resolution convergence of total cold gas mass in halo
    if 0:
        sPs = [TNG50, TNG50_2, TNG50_3, TNG50_4]
        # sPs = []
        # for res in [1820,910,455]:
        #    sPs.append( simParams(run='tng',res=res,redshift=redshift) )
        yQuants = ["mass_halogas_sfcold"]
        xQuant = "mhalo_200_log"

        subhalos.median(sPs, yQuants, xQuant, cenSatSelect="cen", markersize=4.0, xlim=[10.5, 13.0])

    # --- clump analysis ---

    # fig 7: zoomed in (high res) visualizations of multiple properties of clumps
    if 0:
        sP = TNG50
        haloIDs = _get_halo_ids(sP)[2]

        for conf in [5, 6, 7]:
            lrgHaloVisualization(sP, None, conf=conf, gallery=False)

        # vis test - verify a ~1.5kpc clump by weighting those indices to zero in vis()
        # haloIDs = [0]
        # lrgHaloVisualization(sP, haloIDs, conf=3, gallery=False, globalDepth=False, testClumpRemoval=True)

    # fig 8: N_clumps vs halo mass, to show they are significantly more abundant towards larger halo masses
    if 0:
        clumpPropertiesVsHaloMass([TNG50, TNG50_2, TNG50_3, TNG50_4])

    # fig 9: clump demographics: size distribution, total mass, average numdens, etc
    if 0:
        haloID = 0  # 0, 8, 19
        clumpDemographics([TNG50], haloID=haloID)  # , stackHaloIDs=stackHaloIDs)

        # convergence of N_clump histogram with resolution (inset)
        sPs = [TNG50, TNG50_2, TNG50_3, TNG50_4]
        clumpDemographics(sPs, haloID=None, stackHaloIDs=[0])  # halo 0 for all res levels

        # explore tracer origins:
        # clumpDemographics([TNG50], haloID=haloID, stackHaloIDs=None, trAnalysis=True)

    # fig 10: individual (or stacked) clump radial profiles, including pressure/cellsize
    if 0:
        haloID = 8
        norm = False
        selections = [{"size": [0.5, 0.55]}, {"size": [1.0, 1.1]}, {"size": [1.5, 1.6]}, {"size": [2.0, 2.1]}]

        clumpRadialProfiles(TNG50, haloID, selections, norm=norm)

    # fig 12: time tracks via tracers (subbox)
    if 0:
        # pick a single clump (from np.where(objs['lengths'] == 100))
        haloID = 0  # only >10^13 halo which intersects with subboxes
        # sbNum = None # if None, then use fullbox time range/spacing
        sbNum = 2  # if integer, use subbox time range/spacing
        clumpID = 1592  # 3416, 3851

        clumpTracerTracks(TNG50, haloID=haloID, clumpID=clumpID, sbNum=sbNum)

        # helper: loop above over many clumps
        if 0:
            # load segmentation and select
            objs, props = voronoiThresholdSegmentation(
                TNG50, haloID=0, propName="Mg II numdens", propThresh=1e-8, propThreshComp="gt"
            )

            clumpIDs1 = np.where((objs["lengths"] >= 400) & (objs["lengths"] < 405))[0]
            clumpIDs2 = np.where((objs["lengths"] >= 300) & (objs["lengths"] < 305))[0]
            clumpIDs3 = np.where((objs["lengths"] >= 200) & (objs["lengths"] < 205))[0]
            clumpIDs4 = np.where(objs["lengths"] == 100)[0][0:10]
            clumpIDs5 = np.where(objs["lengths"] == 40)[0][0:10]

            clumpIDs = np.hstack((clumpIDs1, clumpIDs2, clumpIDs3, clumpIDs4, clumpIDs5))
            print("Processing [%d] clumps..." % clumpIDs.size)

            # load tracer tracks, cache, make plots
            for clumpID in clumpIDs:
                clumpTracerTracks(TNG50, haloID=haloID, clumpID=clumpID, sbNum=sbNum, posOnly=True)

    # fig 13: visual frames sequence: clump evo
    if 0:
        # pick a single clump
        haloID = 0  # only >10^13 halo which intersects with subboxes
        sbNum = 2  # if None, then use fullbox time range/spacing
        clumpID = 1592  # for paper

        cloudEvoVisFigure(TNG50, haloID=haloID, clumpID=clumpID, sbNum=sbNum)

        # clumpID = 4087 # finished explorations: 430, 797, 2165, 2438, 4087
        # for sizeParam in [0,1]:
        #    cloudEvoVis(TNG50, haloID=haloID, clumpID=clumpID, sbNum=sbNum, sizeParam=sizeParam)

    # fig X: visual comparison of MHD vs noMHD zoom test run of h23 down to z=0.5
    if 0:
        # no_mhd run: snaps 63-69 (z=0.594 to z=0.506), with_mhd run: snaps up to 67 (z=0.520)
        # MHD was disabled in restart at z=0.601 roughly (500 Myr from snap 63 to 67)
        for snap in [65]:  # [63,64,65,66,67]:
            zoom_with_mhd = simParams(run="tng50_zoom", hInd=23, res=11, snap=snap)
            # zoom_no_mhd = simParams(run='tng50_zoom',hInd=23,res=11,snap=snap,variant='nob_z06') # published
            zoom_no_mhd = simParams(run="tng50_zoom", hInd=23, res=11, snap=snap, variant="nob")  # new test

            conf = 3  # 2=HI, 3=MgII, 9=delta_rho
            lrgHaloVisualization(zoom_with_mhd, [0], conf=conf, gallery=False, globalDepth=False)
            lrgHaloVisualization(zoom_no_mhd, [0], conf=conf, gallery=False, globalDepth=False)

    # fig 17: resolution convergence, visual (matched halo)
    if 0:
        haloIDs = _get_halo_ids(TNG50)[2]
        lrgHaloVisResolution(TNG50, haloIDs, [TNG50_2, TNG50_3, TNG50_4])

    # --- observational comparison ---

    # fig X: obs matched samples for COS-LRG and LRG-RDR surveys
    if 0:
        sPs = [TNG50, TNG100]

        simNames = "_".join([sP.simName for sP in sPs])
        obsSimMatchedGalaxySamples(sPs, "sample_lrg_rdr_%s.pdf" % simNames, config="LRG-RDR")
        obsSimMatchedGalaxySamples(sPs, "sample_cos_lrg_%s.pdf" % simNames, config="COS-LRG")

    # fig 14: quantitative lambda comparisons for LRG-RDR and COS-LRG, associated plots
    if 0:
        for sP in [TNG50]:  # [TNG50, TNG100]:
            obsColumnsDataPlotExtended(sP, saveName="obscomp_lrg_rdr_hi_%s_ext.pdf" % sP.simName, config="LRG-RDR")
            obsColumnsDataPlotExtended(sP, saveName="obscomp_cos_lrg_hi_%s_ext.pdf" % sP.simName, config="COS-LRG HI")
            obsColumnsDataPlotExtended(
                sP, saveName="obscomp_cos_lrg_mgii_%s_ext.pdf" % sP.simName, config="COS-LRG MgII"
            )

    # fig 14 (inset): lambda versus R
    if 0:
        sP = TNG50
        configs = ["LRG-RDR", "COS-LRG HI", "COS-LRG MgII"]
        obsColumnsLambdaVsR(sP, saveName="obscomp_composite_lambda_vs_R.pdf", configs=configs)

    # fig 15 - N_MgII or N_HI vs. b (map-derived): 2D histo of N_px/N_px_tot_annuli (normalized independently by column)
    if 0:
        sP = TNG50
        haloMassBin = [13.2, 13.8]
        radRelToVirRad = False

        for ion in ["Mg II", "MHI_GK"]:
            ionColumnsVsImpact2D(sP, haloMassBin, ion=ion, radRelToVirRad=radRelToVirRad, ycum=True, fullDepth=True)

            for ycum in [True, False]:
                for fullDepth in [True, False]:
                    ionColumnsVsImpact2D(
                        sP, haloMassBin, ion=ion, radRelToVirRad=radRelToVirRad, ycum=ycum, fullDepth=fullDepth
                    )

    # fig 16: covering fraction comparison
    if 0:
        haloMassBin = haloMassBins[2]
        ion = "Mg II"
        Nthreshs = [15.0, 15.5, 16.0]
        sPs = [TNG50, simParams(run="tng50-1", redshift=0.4), simParams(run="tng50-1", redshift=0.6)]
        sPs2 = [
            [TNG50_2, simParams(run="tng50-2", redshift=0.4), simParams(run="tng50-2", redshift=0.6)],
            [TNG50_3, simParams(run="tng50-3", redshift=0.4), simParams(run="tng50-3", redshift=0.6)],
            [TNG50_4, simParams(run="tng50-4", redshift=0.4), simParams(run="tng50-4", redshift=0.6)],
        ]

        ionCoveringFractionVsImpact2D(sPs, haloMassBin, ion, Nthreshs, sPs2=sPs2, radRelToVirRad=False, fullDepth=True)

        # helper: curve of growth for MgII
        from temet.plot.cloudy import curveOfGrowth

        curveOfGrowth(lineName="MgII2803")
