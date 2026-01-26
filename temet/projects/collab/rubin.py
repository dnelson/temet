"""
Kate Rubin / HST MST 2024 Proposal.
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from temet.cosmo.util import subsampleRandomSubhalos
from temet.plot.config import colors, figsize, linestyles, lw, markers
from temet.util import simParams
from temet.util.helper import dist_theta_grid, logZeroNaN, running_median
from temet.vis.halo import renderSingleHalo


def hubbleMCT_gibleVis(conf=1):
    """Visualization of CGM emission from a GIBLE or TNG50 halo."""

    class plotConfig:
        plotStyle = "open"
        rasterPx = 360
        colorbars = True
        title = None

    if 1:
        res = 4096  # 8, 64, 512, 4096
        hInd = 201
        redshift = 0.15  # 0.15
        run = "gible"  # gible

        subhaloInd = 0
        method = "sphMap_global"

        # pretend snapshot is at this redshift (note: manual hacks also needed in vis.common for flux and SB)
        mock_redshift = 0.36

        plotConfig.saveFilename = "gible_h%d_RF%d_%s.pdf" % (hInd, res, conf)

    if 0:
        run = "tng"
        res = 2160
        redshift = 0.36

        subhaloInd = 473093  # 549178
        method = "sphMap"
        labelHalo = "mstar,mhalo,sfr"

        plotConfig.saveFilename = "tng50-1_h%d_%s.pdf" % (subhaloInd, conf)

    rVirFracs = [2.0]
    fracsType = "rHalfMassStars"
    nPixels = [1000, 1000]
    size = 200.0  # 0.5 #2.5
    sizeType = "codeUnits"
    axes = [0, 1]
    axesUnits = "arcsec"
    labelSim = False
    labelZ = True  # False
    rotation = "edge-on"
    labelScale = False

    vmm = [-20.0, -16.5]  # log sb
    ctName = "magma_gray"  # 'magma_gray'
    colorbarnoticks = True

    if conf == 0:
        # render OVI 1032+1038 doublet combined
        panels = [{"partType": "gas", "partField": "sb_OVI_ergs", "valMinMax": vmm}]
        grid1, config1 = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

        panels = [{"partType": "gas", "partField": "sb_O--6-1037.62A_ergs", "valMinMax": vmm}]
        grid2, config2 = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

        panels[0]["grid"] = np.log10(10.0**grid1 + 10.0**grid2)
        panels[0]["colorbarlabel"] = config1["label"].replace("OVI SB", r"OVI 1032+1038$\AA$ SB")
        panels[0]["colorbarlabel"] = r"OVI 1032+1038$\AA$ Surface Brightness"

    if conf == 1:
        # CIII
        panels = [{"partType": "gas", "partField": "sb_CIII_ergs", "valMinMax": vmm}]

    if conf == 2:
        # render CIII/OVI doublet ratio
        panels = [{"partType": "gas", "partField": "sb_CIII_ergs", "valMinMax": vmm}]
        grid_CIII, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

        panels[0]["partField"] = "sb_OVI_ergs"
        grid_OVI1032, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)
        panels[0]["partField"] = "sb_O--6-1037.62A_ergs"
        grid_OVI1038, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

        grid_ratio = np.log10(10.0**grid_CIII / (10.0**grid_OVI1032 + 10.0**grid_OVI1038))

        panels[0]["grid"] = grid_ratio
        panels[0]["valMinMax"] = [-1.0, 1.0]
        panels[0]["colorbarlabel"] = "(CIII/OVI) Surface Brightness Ratio [log]"
        panels[0]["ctName"] = "curl"

    if conf == 3:
        # He II test
        panels = [{"partType": "gas", "partField": "sb_He-2-1640.43A_ergs", "valMinMax": vmm}]

    if conf == 4:
        panels = [{"partType": "gas", "partField": "sb_MgII_ergs", "valMinMax": vmm}]

    if conf == 5:
        panels = [{"partType": "gas", "partField": "sb_O--2-3728.81A_ergs", "valMinMax": vmm}]
        panels[0]["colorbarlabel"] = r"OII 3729$\AA$ SB [log erg s$^{−1}$ cm$^{−2}$ arcsec$^{−2}$]"
        panels[0]["colorbarlabel"] = r"OII 3729$\AA$ Surface Brightness"

    if conf == 6:
        panels = [{"partType": "gas", "partField": "sb_O--3-5006.84A_ergs", "valMinMax": vmm}]
        panels[0]["colorbarlabel"] = r"OIII 5007$\AA$ SB [log erg s$^{−1}$ cm$^{−2}$ arcsec$^{−2}$]"
        panels[0]["colorbarlabel"] = r"OIII 5007$\AA$ Surface Brightness"

    if conf == 7:
        panels = [{"partType": "gas", "partField": "sb_H-alpha", "valMinMax": [-16.0, -12.0]}]

    if conf == 8:
        panels = [{"partType": "gas", "partField": "sb_H-beta", "valMinMax": [-16.0, -12.0]}]

    if conf == 9:
        panels = [{"partType": "stars", "partField": "stellarComp"}]

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def hubbleMCT_emissionTrends(simname="tng50-1", cQuant=None):
    """Hubble MST Proposal 2024 of Kate Rubin."""
    sim = simParams(simname, redshift=0.36)  # tng50-1, eagle, simba

    # grid config
    method = "sphMap"  # sphMap_global
    nPixels = [1000, 1000]
    axes = [0, 1]
    size = 180.0  # 35 arcsec @ z=0.36 (SBC field of view is 35"x31")
    sizeType = "kpc"

    sim.createCloudyCache = True if "_global" in method else False

    # config
    fields = ["sb_OVI_ergs", "sb_O--6-1037.62A_ergs", "sb_CIII_ergs"]
    percs = [25, 50, 75]
    distBins = [[20, 30], [45, 55]]  # pkpc

    # sample
    mstar_min = 9.0
    mstar_max = 11.0
    num_per_dex = 100

    subInds, mstar = subsampleRandomSubhalos(sim, num_per_dex, [mstar_min, mstar_max], cenOnly=True)

    dist, _ = dist_theta_grid(size, nPixels)

    # check for existence of cache
    grids = {}
    sb_percs = {}
    cacheFile = sim.cachePath + "hstmst_grids.hdf5"

    if isfile(cacheFile):
        # load cached result
        with h5py.File(cacheFile, "r") as f:
            for field in fields:
                grids[field] = f[field][()]
            for key in f["sb_percs"]:
                sb_percs[key] = f["sb_percs/%s" % key][()]

            assert np.array_equal(subInds, f["subInds"][()])
            assert np.array_equal(percs, f["percs"][()])
            assert np.array_equal(distBins, f["distBins"][()])
        print("Loaded: [%s]" % cacheFile)
    else:
        # compute now: allocate
        for field in fields:
            grids[field] = np.zeros((nPixels[0], nPixels[1], len(subInds)), dtype="float32")

        sb_percs["OVI"] = np.zeros((len(subInds), len(distBins), len(percs)), dtype="float32")
        sb_percs["CIII"] = np.zeros((len(subInds), len(distBins), len(percs)), dtype="float32")

        # loop over subhalos
        class plotConfig:
            saveFilename = "dummy"

        for i, subhaloInd in enumerate(subInds):
            print(f"[{i:3d}] of [{len(subInds):3d}] {subhaloInd = }", flush=True)

            for field in fields:
                # project
                sP = sim
                panels = [{"partType": "gas", "partField": field}]
                grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                # stamp
                grids[field][:, :, i] = grid

            # compute statistics
            for j, distBin in enumerate(distBins):
                # pixels in this annulus
                w = np.where((dist >= distBin[0]) & (dist < distBin[1]))

                # OVI doublet map and CIII map separately
                OVI_map = np.log10(
                    10.0 ** grids["sb_OVI_ergs"][:, :, i] + 10.0 ** grids["sb_O--6-1037.62A_ergs"][:, :, i]
                )
                CIII_map = grids["sb_CIII_ergs"][:, :, i]

                sb_percs["OVI"][i, j, :] = np.percentile(OVI_map[w], percs)
                sb_percs["CIII"][i, j, :] = np.percentile(CIII_map[w], percs)

        # save cache
        with h5py.File(cacheFile, "w") as f:
            for field in fields:
                f[field] = grids[field]
            for key in sb_percs.keys():
                f["sb_percs/%s" % key] = sb_percs[key]
            f["subInds"] = subInds
            f["percs"] = percs
            f["distBins"] = distBins

        print("Saved: [%s]" % cacheFile)

    # start figure
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_\odot$ ]")
    ax.set_ylabel(r"Surface Brightness [ log erg/s/cm$^2$/arcsec$^2$ ]")
    ax.set_xlim([mstar_min, mstar_max])
    ax.set_ylim([-24.5, -17])

    if cQuant is not None:
        # c_vals = sim.subhalos(cQuant)[subInds]
        sim_cvals, clabel, cMinMax, cLog = sim.simSubhaloQuantity(cQuant)
        sim_cvals = sim_cvals[subInds]
        if cLog:
            sim_cvals = logZeroNaN(sim_cvals)
        clim = None
        cmap = "inferno"
        cMinMax = cMinMax if clim is None else clim

    # plot
    count = 0

    for i, distBin in enumerate(distBins):
        for line, label in zip(["OVI", "CIII"], ["OVI 1032+1038", "CIII 977"]):
            y_mid = sb_percs[line][:, i, 1]
            y_err_lo = sb_percs[line][:, i, 1] - sb_percs[line][:, i, 0]
            y_err_hi = sb_percs[line][:, i, 2] - sb_percs[line][:, i, 1]

            label_loc = r"%s (%d$\pm$%d kpc)" % (label, np.mean(distBin), (distBin[1] - distBin[0]) / 2)
            if cQuant is None:
                ax.errorbar(mstar, y_mid, yerr=[y_err_lo, y_err_hi], fmt="o", label=label_loc)
            else:
                opts = {"vmin": cMinMax[0], "vmax": cMinMax[1], "c": sim_cvals, "cmap": cmap, "marker": markers[count]}
                sc = ax.scatter(mstar, y_mid, label=label_loc, **opts)
                count += 1

    # finish and save plot
    legend = ax.legend(loc="upper left", title=f"{sim.simName} z = {sim.redshift:.2f}")
    legend._legend_box.align = "left"

    if cQuant is not None:
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(sc, cax=cax)
        cb.set_alpha(1)  # fix stripes
        cb.draw_all()
        cb.ax.set_ylabel(clabel)

    fig.savefig("mst_OVI_CIII_annuli_vs_mstar_%s.pdf" % sim.simName)
    plt.close(fig)


def hubbleMCT_emissionTrendsVsSim():
    """Combine results from above into a summary plot."""
    # config
    sims = ["tng50-1", "eagle", "simba"]
    nBins = 15
    redshift = 0.36

    # load
    sb_percs = {}
    mstar = {}

    for simname in sims:
        print(simname)
        sim = simParams(simname, redshift=redshift)
        cacheFile = sim.cachePath + "hstmst_grids.hdf5"

        with h5py.File(cacheFile, "r") as f:
            for key in f["sb_percs"]:
                sb_percs[f"{simname}_{key}"] = f["sb_percs/%s" % key][()]
            distBins = f["distBins"][()]
            subInds = f["subInds"][()]

        mstar[simname] = sim.subhalos("mstar_30pkpc_log")[subInds]

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(figsize[0] * 1.5, figsize[1]))

    ax1.set_xlabel(r"Galaxy Stellar Mass [ log M$_\odot$ ]")
    ax1.set_ylabel(r"OVI 1032+1038 SB [ log erg/s/cm$^2$/arcsec$^2$ ]")
    ax1.set_xlim([9.0, 11.0])
    ax1.set_ylim([-22.5, -17.5])

    ax2.set_xlabel(r"Galaxy Stellar Mass [ log M$_\odot$ ]")
    ax2.set_ylabel(r"CIII 977 SB [ log erg/s/cm$^2$/arcsec$^2$ ]")
    ax2.set_xlim([9.0, 11.0])
    ax2.set_ylim([-22.5, -17.5])

    for i, simname in enumerate(sims):
        for j, distBin in enumerate(distBins):
            sim = simParams(simname)
            for line, ax in zip(["OVI", "CIII"], [ax1, ax2]):
                y_mid = sb_percs[simname + "_" + line][:, j, 1]
                y_lo = sb_percs[simname + "_" + line][:, j, 0]
                y_hi = sb_percs[simname + "_" + line][:, j, 2]

                # running median
                xx, yy_mid, _ = running_median(mstar[simname], y_mid, nBins=nBins)
                xx, yy_lo, _ = running_median(mstar[simname], y_lo, nBins=nBins)
                xx, yy_hi, _ = running_median(mstar[simname], y_hi, nBins=nBins)

                label = f"{sim.name} ({np.mean(distBin):.0f} kpc)"  # $\pm${(distBin[1]-distBin[0])/2:.0f} kpc)'
                ax.plot(xx, yy_mid, lw=lw, color=colors[i], ls=linestyles[j], label=label)
                if j == 0:
                    ax.fill_between(xx, yy_lo, yy_hi, color=colors[i], alpha=0.2)

    # finish and save plot
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower right")
    ax1.text(0.97, 0.03, f"z = {redshift:.2f}", transform=ax1.transAxes, ha="right", va="bottom")
    ax2.text(0.03, 0.97, f"z = {redshift:.2f}", transform=ax2.transAxes, ha="left", va="top")
    fig.savefig("mst_OVI_CIII_annuli_vs_mstar.pdf")
    plt.close(fig)
