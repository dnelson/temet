"""
Nhut Truong (and Martin-Navarro).

Angular anisotropy of x-ray emission of the CGM/ICM (Troung+21, TNG50, https://arxiv.org/abs/2109.06884).
OVII and OVIII emission (Troung+22, https://arxiv.org/abs/2307.01277).
"""

import hashlib
from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic_2d

from temet.plot.config import figsize, linestyles, lw, percs
from temet.plot.util import sampleColorTable
from temet.util import simParams
from temet.util.helper import closest, dist_theta_grid, running_median
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


valMinMaxQuant = {
    "coldens": [18.5, 20.0],  # in case we render actual quantities instead of deltas
    "xray_lum_05-2kev": [33, 37],
    "temp": [6.2, 6.6],
    "xray_lum": [33, 37],
    "metal_solar": [-0.5, 0.0],
}


def check_xray():
    """TEST."""
    sP = simParams(run="tng100-1", redshift=0.0)
    subhaloInd = 472174

    rVirFracs = [0.25, 0.5]
    method = "histo"  #'sphMap'
    nPixels = [100, 100]  # [1200,1200]
    axes = [0, 1]
    labelZ = False
    labelScale = "physical"
    labelSim = False
    labelHalo = True
    relCoords = True
    rotation = None  #'edge-on-stars'
    sizeType = "kpc"
    size = 600
    depthFac = 0.1

    # get mine
    panels = [{"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]}]

    # if 'xray' not in panels[0]['partField']:
    #    # temperature cut, except for x-ray where it isn't needed
    #    ptRestrictions = {'temp_sfcold_log':['gt',6.0]}

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1200 if "xray" in panels[0]["partField"] else 840
        colorbars = True
        fontsize = 22  # if 'xray' in panels[0]['partField'] else 32

    data, config = renderSingleHalo(panels, plotConfig, locals(), skipExisting=False, returnData=True)
    data = 10.0**data

    # get nhut's
    with h5py.File("Gas_Properties_TNG100-1_99_SubNum_472174_15R500_0.0.h5", "r") as f:
        data_nhut = f["Density"][()]

    w1 = np.where(np.isfinite(data))
    w2 = np.where(np.isfinite(data_nhut))
    print("data minmax: ", np.nanmin(data), np.nanmax(data))
    print("nhut minmax: ", np.nanmin(data_nhut), np.nanmax(data_nhut))

    # do manually from scratch
    groupID = sP.subhalo(subhaloInd)["SubhaloGrNr"]

    pos = sP.snapshotSubset("gas", "pos_rel_kpc", haloID=groupID)
    xx = pos[:, 0]
    yy = pos[:, 1]
    values = sP.snapshotSubset("gas", "mass", haloID=groupID)
    values = sP.units.codeMassToMsun(values)
    bounds = [[-size / 2, size / 2], [-size / 2, size / 2]]

    w_z = np.where(np.abs(pos[:, 2]) < size / 2 * depthFac)
    xx = xx[w_z]
    yy = yy[w_z]
    values = values[w_z]

    grid, _, _, _ = binned_statistic_2d(xx, yy, values, statistic="sum", bins=nPixels[0], range=bounds)
    grid /= (size / nPixels[0]) ** 2  # msun -> msun/kpc^2
    grid[grid == 0] = np.nan  # consistent with others

    print("grid minmax: ", np.nanmin(grid), np.nanmax(grid))

    w3 = np.where(np.isfinite(grid))

    print("number of non-empty pixels: ", len(w1[0]), len(w2[0]), len(w3[0]))


def _get_panels(conf, stack2Dmaps, median, renderIndiv):
    """Common config."""
    if conf == 0:
        panels = [{"partType": "gas", "partField": "delta_rho", "valMinMax": [-0.2, 0.2]}]
    if conf == 1:
        panels = [{"partType": "gas", "partField": "delta_xray_lum_05-2kev", "valMinMax": [-0.3, 0.3]}]
        if median:
            panels[0]["valMinMax"][1] = 0.1
    if conf == 2:
        panels = [{"partType": "gas", "partField": "delta_temp", "valMinMax": [0.05, 0.5]}]
        if stack2Dmaps:
            panels[0]["valMinMax"] = [-0.1, 0.1]
    if conf == 3:
        panels = [{"partType": "gas", "partField": "delta_xray_lum", "valMinMax": [-0.3, 0.1]}]
        if not median:
            panels[0]["valMinMax"][1] = 0.2
    if conf == 4:
        panels = [{"partType": "gas", "partField": "delta_metal_solar", "valMinMax": [-0.2, 0.2]}]
    if conf == 5:
        panels = [{"partType": "gas", "partField": "delta_xray_lum_05-2kev", "valMinMax": [-0.3, 0.3]}]
        if median:
            panels[0]["valMinMax"][1] = 0.1

    # or: create normal projections of original quantities, stack, and then remove radial profile in 2D?
    if stack2Dmaps:
        panels[0]["partField"] = panels[0]["partField"].replace("delta_", "")
        if panels[0]["partField"] == "rho":
            panels[0]["partField"] = "coldens"

    if renderIndiv:
        panels[0]["valMinMax"] = valMinMaxQuant[panels[0]["partField"]]

    return panels


def stackedHaloImage(sP, mStarBin, conf=0, renderIndiv=False, median=True, rvir=False, depthFac=1.0, stack2Dmaps=False):
    """Stacked halo-scale image: delta rho/rho (for Martin Navarro+21) and x-ray SB (for Truong+21).

    Orient all galaxies edge-on, and remove average radial profile, to highlight angular variation.
    """
    # select halos
    mstar = sP.subhalos("mstar_30pkpc_log")
    cen_flag = sP.subhalos("central_flag")

    with np.errstate(invalid="ignore"):
        subhaloIDs = np.where((mstar > mStarBin[0]) & (mstar <= mStarBin[1]) & cen_flag)[0]

    # vis
    rVirFracs = [0.25, 0.5]
    method = "histo"  #'sphMap'
    nPixels = [100, 100]  # [1200,1200]
    axes = [0, 1]
    labelZ = False
    labelScale = "physical"
    labelSim = False
    labelHalo = True
    relCoords = True
    rotation = "edge-on-stars"
    sizeType = "kpc"
    size = 600

    # normal config: create projections of relative quantities, first calculated in 3D per-cell
    panels = _get_panels(conf, stack2Dmaps, median, renderIndiv)

    if "xray" not in panels[0]["partField"]:
        # temperature cut, except for x-ray where it isn't needed
        ptRestrictions = {"temp_sfcold_log": ["gt", 6.0]}

    if rvir:
        size = 3.0
        sizeType = "rVirial"
        nPixels = [800, 800]

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1200 if "xray" in panels[0]["partField"] else 840
        colorbars = True
        fontsize = 22  # if 'xray' in panels[0]['partField'] else 32

    # cache and output files
    dfStr = "_df%.1f" % depthFac if depthFac != 1.0 else ""
    mStr = "_median" if median else "_mean"
    sStr = "_stack2D" if stack2Dmaps else ""
    rStr = "_rvirunits" if rvir else ""

    indivSaveName = "./vis_%s_z%d_XX%s_%s%s.pdf" % (sP.simName, sP.redshift, dfStr, panels[0]["partField"], mStr)

    saveFilename = "stack_data_global_conf%d%s%s%s%s.hdf5" % (conf, mStr, rStr, dfStr, sStr)

    if isfile(saveFilename) and not renderIndiv:
        print("Loading [%s]." % saveFilename)

        with h5py.File(saveFilename, "r") as f:
            data_global = f["data_global"][()]
            weight_global = f["weight_global"][()]

    else:
        # allocate
        print("Stacking [%d] halos." % len(subhaloIDs))
        if median:
            data_global = np.zeros((len(subhaloIDs), nPixels[0], nPixels[1]), dtype="float64")
            weight_global = np.zeros((1), dtype="int32")
        else:
            data_global = np.zeros(nPixels, dtype="float64")
            weight_global = np.zeros(nPixels, dtype="int32")

        # loop over halos
        for i, subhaloInd in enumerate(subhaloIDs):
            # render individual images?
            if renderIndiv:
                plotConfig.saveFilename = indivSaveName.replace("XX", "sh%d" % subhaloInd)
                renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

                continue

            # accumulate data for rendering single stacked image
            data_loc, config = renderSingleHalo(panels, plotConfig, locals(), skipExisting=False, returnData=True)
            data_loc = 10.0 ** data_loc.astype("float64")  # log -> linear

            if median:
                # median stacking
                data_global[i, :, :] = data_loc
            else:
                # mean stacking
                w = np.where(np.isfinite(data_loc))

                weight_global[w] += 1  # number of halos accumulated per pixel
                data_global[w] += data_loc[w]  # accumulate

        with h5py.File(saveFilename, "w") as f:
            f["data_global"] = data_global
            f["weight_global"] = weight_global
        print("Saved: [%s]." % saveFilename)

    # plot stacked image and save data grid to hdf5
    subhaloInd = subhaloIDs[int(len(subhaloIDs) / 2)]  # used for rvir circles
    labelHalo = False
    plotConfig.saveFilename = indivSaveName.replace("XX", "stack" if not stack2Dmaps else "stack2D")

    # construct input grid: mean/median average across halos, and linear -> log
    if median:
        grid = np.nanmedian(data_global, axis=0)
    else:
        grid = data_global / weight_global

    if panels[0]["partField"].startswith("delta_"):
        # delta_rho (or delta_Q) computed in 3D: use log
        grid = np.log10(grid)

    else:
        # we have gridded an actual cell property, derive mean radial profile now and remove it
        dist, _ = dist_theta_grid(size, nPixels)

        xx, yy, _ = running_median(dist, np.log10(grid), nBins=50)

        f = interp1d(xx, yy, kind="linear", bounds_error=False, fill_value="extrapolate")

        if 0:
            # debug plot
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.set_xlabel("distance [kpc]")
            ax.set_ylabel("gas cell property [log]")
            ax.scatter(dist, np.log10(grid), s=1.0, marker=".", color="black", alpha=0.5)

            ax.plot(xx, yy, "o-", lw=lw)

            dist_uniq_vals = np.unique(dist)
            yy2 = f(dist_uniq_vals)

            ax.plot(dist_uniq_vals, yy2, "-", lw=lw)

            fig.savefig("debug_dist_fit_conf%d.png" % conf)
            plt.close(fig)

        # render stacked grid prior to subtraction
        if 1:
            plotConfig.saveFilename = plotConfig.saveFilename.replace(".pdf", "_orig.pdf")
            valMinMaxOrig = panels[0]["valMinMax"]
            panels[0]["valMinMax"] = valMinMaxQuant[panels[0]["partField"]]
            grid = np.log10(grid)
            renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)
            plotConfig.saveFilename = plotConfig.saveFilename.replace("_orig", "")
            grid = 10.0**grid
            panels[0]["valMinMax"] = valMinMaxOrig

        # we have our interpolating function for the average value at a given distance
        grid /= 10.0 ** f(dist)  # Sigma -> Sigma/<Sigma> (linear)
        grid = np.log10(grid)

    # render stacked grid
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)

    # with h5py.File(plotConfig.saveFilename.replace('.pdf','.hdf5'),'w') as f:
    #    f['grid'] = grid


def stackedPropVsTheta(sP, mStarBin, distBins, conf=0, depthFac=1.0, stack2Dmaps=True, distRvir=False, nThetaBins=45):
    """Stacked plot of quantity vs azimuthal angle."""
    # load for halo selection
    mstar = sP.subhalos("mstar_30pkpc_log")
    cen_flag = sP.subhalos("central_flag")

    # vis config
    rVirFracs = [0.25, 0.5]
    method = "sphMap"
    nPixels = [1200, 1200]
    axes = [0, 1]
    labelZ = False
    labelScale = "physical"
    labelSim = False
    labelHalo = True
    relCoords = True
    rotation = "edge-on-stars"
    sizeType = "kpc"
    size = 600

    # normal config: create projections of relative quantities, first calculated in 3D per-cell
    panels = _get_panels(conf, stack2Dmaps, False, False)

    dataField = panels[0]["partField"]

    ptRestrictions = None
    if "xray" not in panels[0]["partField"]:
        # temperature cut, except for x-ray where it isn't needed
        ptRestrictions = {"temp_sfcold_log": ["gt", 6.0]}

    dist, theta = dist_theta_grid(size, nPixels)

    labels = {
        "temp": "Relative Temperature [linear]",
        "coldens": "Relative Gas Column Density [linear]",
        "xray_lum_05-2kev": "Relative L$_{\\rm X,0.5-2 keV,XSPEC}$ [linear]",
        "metal_solar": "Relative Gas Metallicity [linear]",
    }

    # start figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel("Azimuthal Angle [deg]")
    ax.set_xlim([-2, 92])
    ax.set_ylim([0.8, 1.2])
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylabel(labels[dataField])

    ax.plot(ax.get_xlim(), [1.0, 1.0], "-", color="black", lw=lw * 4, alpha=0.1)

    # loop over mass/distance bins
    colors = sampleColorTable("plasma", len(mStarBin) * len(distBins), bounds=[0.1, 0.7])
    # colors = sampleColorTable('plasma', 5, bounds=[0.1,0.8])

    # load
    for i, massBin in enumerate(mStarBin):
        with np.errstate(invalid="ignore"):
            subhaloIDs = np.where((mstar > massBin[0]) & (mstar < massBin[1]) & cen_flag)[0]

        print("[%.2f - %.2f] Processing [%d] halos..." % (massBin[0], massBin[1], len(subhaloIDs)))

        # check for existence of cache
        hashStr = "%s-%s-%s-%s-%s-%s-%s-%s-%s" % (
            method,
            nPixels,
            axes,
            rotation,
            size,
            sizeType,
            ptRestrictions,
            sP.snap,
            subhaloIDs,
        )
        m = hashlib.sha256(hashStr.encode("utf-8")).hexdigest()[::4]
        cacheFile = sP.cachePath + "aziangle_grids_%s_%s.hdf5" % (dataField, m)

        if isfile(cacheFile):
            # load cached result
            with h5py.File(cacheFile, "r") as f:
                grid_global = f["grid_global"][()]
            print("Loaded: [%s]" % cacheFile)
        else:
            # compute now
            dtype = "float64" if "xray" in dataField else "float32"
            grid_global = np.zeros((nPixels[0] * nPixels[1], len(subhaloIDs)), dtype=dtype)

            # loop over halos
            for j, subhaloInd in enumerate(subhaloIDs):  # noqa: B007

                class plotConfig:
                    saveFilename = "dummy"

                # accumulate data for rendering single stacked image
                grid, config = renderSingleHalo(panels, plotConfig, locals(), skipExisting=False, returnData=True)
                grid = 10.0 ** grid.astype("float64")  # log -> linear

                # flatten
                grid_global[:, j] = grid.ravel()

            # flatten (ignore which halo each pixel came from)
            grid_global = grid_global.ravel()

            # save cache
            with h5py.File(cacheFile, "w") as f:
                f["grid_global"] = grid_global

            print("Saved: [%s]" % cacheFile)

        # rearrange theta and dist into same shape
        dist_global = np.zeros((nPixels[0] * nPixels[1], len(subhaloIDs)), dtype="float32")
        theta_global = np.zeros((nPixels[0] * nPixels[1], len(subhaloIDs)), dtype="float32")

        for j, subhaloInd in enumerate(subhaloIDs):
            if distRvir:
                haloID = sP.groupCatSingle(subhaloID=subhaloInd)["SubhaloGrNr"]
                haloRvir_code = sP.groupCatSingle(haloID=haloID)["Group_R_Crit200"]
                dist_global[:, j] = dist.ravel() / sP.units.codeLengthToKpc(haloRvir_code)
            else:
                dist_global[:, j] = dist.ravel()
            theta_global[:, j] = theta.ravel()

        dist_global = dist_global.ravel()
        theta_global = theta_global.ravel()

        # prevent overflow, always is made relative
        if dataField == "coldens":
            grid_global /= 1e18
        if "xray" in dataField:
            grid_global /= 1e33

        # bin on the global concatenated grids
        for j, distBin in enumerate(distBins):
            w = np.where((dist_global >= distBin[0]) & (dist_global < distBin[1]))

            # median metallicity as a function of theta, 1 degree bins
            theta_vals, hist, hist_std, hist_percs = running_median(
                theta_global[w], grid_global[w], nBins=nThetaBins, percs=percs
            )

            # make relative to average value at this distance
            hist /= hist.mean()
            for k in range(hist_percs.shape[0]):
                hist_percs[k, :] /= hist_percs[k, :].mean()

            # label and color
            distStr = "b = %d kpc" if not distRvir else "b = %.1f r$_{\\rm vir}$"
            label = distStr % np.mean(distBin) if i == 0 else ""
            # if len(distBins) == 1: label = ''

            c = colors[j]
            ls = linestyles[i]

            # plot line and shaded band
            (l,) = ax.plot(theta_vals, hist, linestyle=ls, lw=lw, label=label, color=c)
            # if j == 0:
            #    ax.fill_between(theta_vals, hist_percs[0,:], hist_percs[-1,:], color=l.get_color(), alpha=0.1)

    # finish and save plot
    ax.legend(loc="best")
    mstarStr = "Mstar=%.1f" % np.mean(mStarBin[0]) if len(mStarBin) == 1 else "Mstar=%dbins" % len(mStarBin)
    distStr = "b=%d" % np.mean(distBins[0]) if len(distBins) == 1 else "b=%dbins" % len(distBins)
    fig.savefig("%s_vs_theta_%s_%s_%s_rvir=%s.pdf" % (dataField, sP.simName, mstarStr, distStr, distRvir))
    plt.close(fig)


def singleHaloImage():
    """Quick test."""
    # select halo
    sP = simParams(run="tng100-1", redshift=0.0)
    mstar = sP.subhalos("mstar_30pkpc_log")
    cen_flag = sP.subhalos("central_flag")

    mstar[cen_flag == 0] = np.nan  # skip secondaries

    _, subhaloInd = closest(mstar, 10.95)

    # vis
    rVirFracs = [0.5, 1.0]
    method = "histo"  #'sphMap'
    nPixels = [100, 100]  # [800, 800]
    axes = [0, 1]
    labelZ = False
    labelScale = "physical"
    labelSim = False
    labelHalo = "mstar,mhalo,id"
    relCoords = True
    rotation = "edge-on-stars"
    sizeType = "rVirial"
    size = 2.0

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 800
        colorbars = True
        fontsize = 22

    # panels
    partType = "gas"

    # panels = [ {'partField':'metal_solar', 'valMinMax':[-0.5,0.2]},
    #           {'partField':'temp', 'valMinMax':[5.5,6.6]},
    #           {'partType':'stars', 'partField':'coldens_msunkpc2', 'valMinMax':[6.0,9.0]} ]

    panels = [
        {"partField": "xray_lum", "valMinMax": [33, 37]},
        {"partField": "xray_lum_05-2kev", "valMinMax": [33, 37]},
        {"partField": "xray_lum_0.5-2.0kev", "valMinMax": [33, 37]},
    ]

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def paperPlots():
    """Plots for Truong+21 x-ray emission angular dependence paper."""
    sP = simParams(run="tng100-1", redshift=0.0)
    mStarBin = [10.90, 11.10]

    if 0:
        # fig 1
        for conf in [0]:  # [0,1,2,3,4,5]:
            stackedHaloImage(
                sP,
                mStarBin,
                conf=conf,
                median=False,
                rvirUnits=False,
                depthFac=0.1,
                stack2Dmaps=True,
                renderIndiv=False,
            )

    if 0:
        # fig 2
        for conf in [1]:  # [0,1,2,4]:
            distBins = [[45, 55], [90, 110], [190, 210], [290, 310]]
            stackedPropVsTheta(sP, [mStarBin], distBins, conf=conf, depthFac=1.0, distRvir=False)

            distBins = [[0.08, 0.12], [0.23, 0.27], [0.48, 0.52], [0.73, 0.77]]
            stackedPropVsTheta(sP, [mStarBin], distBins, conf=conf, depthFac=1.0, distRvir=True)


def fullboxEmissionO8():
    """Create fullbox emission figure for Truong+22."""
    # panels
    redshift = 0.0
    panels = []

    runs = ["tng100-1", "eagle", "simba100"]
    for run in runs:
        labelZ = True if run == runs[-1] else False
        panels.append({"sP": simParams(run=run, redshift=redshift), "labelZ": labelZ})

    # config
    nPixels = 2000
    axes = [0, 1]  # x,y
    labelScale = "physical"
    labelSim = True
    plotHalos = 50
    method = "sphMap"
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)

    sliceFac = 0.15

    partType = "gas"
    partField = "sb_OVIII"
    valMinMax = [-18, -10]

    class plotConfig:
        plotStyle = "edged"  # open, edged
        rasterPx = nPixels
        colorbars = True

        saveFilename = "./boxImage_%s-%s_%s_z%.1f.pdf" % (
            partType,
            partField.replace(" ", "-"),
            "-".join(runs),
            redshift,
        )

    renderBox(panels, plotConfig, locals())
