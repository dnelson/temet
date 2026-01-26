"""
Azimuthal angle dependence of CGM properties (Peroux+, TNG50).

https://arxiv.org/abs/2009.07809
"""

import hashlib
import itertools
from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from temet.plot.config import figsize, lw
from temet.plot.util import sampleColorTable
from temet.util import simParams
from temet.util.helper import dist_theta_grid, logZeroNaN, running_median
from temet.vis.halo import renderSingleHalo


def singleHaloImage(sP, subhaloInd=440839, conf=0):
    """Metallicity distribution in CGM image."""
    rVirFracs = [0.5]
    method = "sphMap"
    nPixels = [800, 800]  # for celinemuse figure
    axes = [0, 1]
    labelZ = True
    labelScale = "physical"
    labelSim = False
    labelHalo = False
    relCoords = True
    rotation = "edge-on"
    sizeType = "kpc"
    size = 200

    panels = []

    if conf == 0:
        panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-1.4, 0.2]})
    if conf == 1:
        panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [13.0, 16.0]})
    if conf == 2:
        panels.append({"partType": "gas", "partField": "Mg II", "valMinMax": [10.0, 16.0]})
    if conf == 3:
        panels.append({"partType": "gas", "partField": "MHI_GK", "valMinMax": [16.0, 22.0]})
    if conf == 4:
        panels.append({"partType": "gas", "partField": "vrad", "valMinMax": [-180, 180]})
    if conf == 5:
        panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-1.4, 0.2]})
        panels.append({"partType": "gas", "partField": "vrad", "valMinMax": [-180, 180]})
        panels.append({"partType": "gas", "partField": "bmag_uG", "valMinMax": [-1.5, 0.5]})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels[0]
        colorbars = True
        fontsize = 18
        saveFilename = "./%s.%d.%d.%s.%dkpc.pdf" % (sP.simName, sP.snap, subhaloInd, panels[0]["partField"], size)

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def metallicityVsVradProjected(sP, shIDs=(440839,), directCells=False, distBin=None, ylim=None):
    """Plot correlation of gas metallicity and vrad (i.e. mass flow rate) in projection by using the images directly."""
    rVirFracs = [0.5]
    method = "sphMap"
    nPixels = [800, 800]
    axes = [0, 1]
    rotation = "edge-on"
    sizeType = "kpc"
    size = 200
    partType = "gas"

    x_field = "vrad"
    y_field = "metal_solar"

    class plotConfig:
        dummy = True

    # start plot
    figsize_loc = (figsize[0] * 2, figsize[1] * 2)  # (figsize[1]*2,figsize[1]*2) # square
    fig = plt.figure(figsize=figsize_loc, frameon=False, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    markersize = 3.0

    # compute
    x_data = []
    y_data = []
    dist = []
    theta = []

    if directCells:
        # use actual gas cell values directly
        for subhaloInd in shIDs:
            x_data_loc = sP.snapshotSubset(partType, x_field, subhaloID=subhaloInd)
            y_data_loc = sP.snapshotSubset(partType, y_field, subhaloID=subhaloInd)
            dist_loc = sP.snapshotSubset(partType, "rad_kpc", subhaloID=subhaloInd)

            x_data = np.hstack((x_data, x_data_loc.ravel()))
            y_data = np.hstack((y_data, y_data_loc.ravel()))
            dist = np.hstack((dist, dist_loc.ravel()))

        _, _, xlog = sP.simParticleQuantity(partType, x_field)
        _, _, ylog = sP.simParticleQuantity(partType, y_field)
        # clabel = "3D Radial Distance [kpc]"

        if xlog:
            x_data = logZeroNaN(x_data)
        if ylog:
            y_data = logZeroNaN(y_data)
    else:
        # use projected images
        for subhaloInd in shIDs:  # noqa: B007
            y_data_loc, y_conf = renderSingleHalo([{"partField": y_field}], plotConfig, locals(), returnData=True)
            x_data_loc, x_conf = renderSingleHalo([{"partField": x_field}], plotConfig, locals(), returnData=True)

            x_data = np.hstack((x_data, x_data_loc.ravel()))
            y_data = np.hstack((y_data, y_data_loc.ravel()))

            # get distances of pixels
            dist_loc, theta_loc = dist_theta_grid(size, nPixels)
            dist = np.hstack((dist, dist_loc.ravel()))
            theta = np.hstack((theta, theta_loc.ravel()))

        # xlabel = x_conf["label"]
        # ylabel = y_conf["label"]
        # clabel = "Azimuthal Angle [deg]"  #'Impact Parameter [kpc]'

    # RESTRICT DISTANCE:
    if distBin is not None:
        w = np.where((dist > distBin[0]) & (dist <= distBin[1]))
        dist = dist[w]
        theta = theta[w]
        x_data = x_data[w]
        y_data = y_data[w]

    if ylim is not None:
        ax.set_ylim(ylim)

    # scatterplot
    s = ax.scatter(x_data, y_data, s=markersize, c=theta, marker=".", zorder=0)  # dist

    hStr = shIDs[0] if len(shIDs) == 1 else ("stack-%s" % len(shIDs))
    fig.savefig("scatter_%s_%s_vs_%s_h%s.png" % (sP.simName, y_field, x_field, hStr), facecolor=fig.get_facecolor())
    plt.close(fig)


def metallicityVsTheta(
    sPs,
    dataField,
    massBins,
    distBins,
    min_NHI=(None,),
    ptRestrictions=None,
    fullbox=False,
    nThetaBins=90,
    addObs=False,
    addEagle=False,
    sizefac=1.0,
    ylim=None,
    distRvir=False,
):
    """Use some projections to create the Z_gas vs. theta plot.

    Args:
      sPs (list[:py:class:`~util.simParams`]): simulation instances.
      dataField (str): what to plot on y-axis.
      massBins (list[float,2]): one or more stellar mass bins.
      distBins (list[float,2]): one or more impact parameter bins.
      min_NHI (list[float]): one or more minimum N_HI column values to consider [log cm^-2].
      ptRestrictions: pass to gridBox(), e.g. {'NeutralHydrogenAbundance':['gt',1e-3]}
      fullbox (bool): do global projections out to larger distance, otherwise fof-local scope.
      nThetaBins (int): number of azimuthal angle bins (default: 90, i.e. 1 degree each).
      addObs (bool): add observational data points.
      addEagle (bool): add EAGLE simulation results from Freeke.
      sizefac (float): figure size scaling factor.
      ylim (list[float,2]): if not None, y-axis limits.
      distRvir: distBins are in units of rvir
    """
    labels = {
        "metal_solar": r"Gas Metallicity [log Z$_\odot$]",
        "Mg II": r"Median Mg II Column Density [log cm$^{-2}$]",
        "O VI": r"Median O VI Column Density [log cm$^{-2}$]",
        "metals_O": r"O Column Density [log M$_\odot$ / kpc$^2$]",
        "metals_Mg": r"O Column Density [log M$_\odot$ / kpc$^2$]",
        "HI": r"HI Column Density [log cm$^{-2}$]",
        "temp_sfcold": r"Gas Temperature [log K]",
    }

    assert dataField in labels.keys()
    assert isinstance(massBins, list) and len(massBins) >= 1
    assert isinstance(distBins, list) and len(distBins) >= 1
    assert isinstance(min_NHI, list) and len(min_NHI) >= 1

    # grid config (must recompute grids)
    method = "sphMap"  # sphMap_global for paper figure
    nPixels = [1000, 1000]
    axes = [0, 1]
    rotation = "edge-on"
    size = 250.0
    sizeType = "kpc"
    weightField = "mass"  #'MHI_GK' #'hi mass' # 'mass' is the default

    percs = [38, 50, 62]

    if fullbox:
        method = "sphMap_global"
        size = 500.0  # dists out to 350kpc
        nPixels = [2000, 2000]  # unchanged pxSize

    # massBins = [ [9.45, 9.55] ] # proposal v2
    # ptRestrictions = None

    # binning config (no need to recompute grids)
    # distBins = [ [90,110] ] # proposalv2
    # min_NHI = [None] # 18.0, enforce minimum HI column?
    # nThetaBins = 90

    dist, theta = dist_theta_grid(size, nPixels)

    # start figure
    figsize_loc = [figsize[0] * sizefac, figsize[1] * sizefac]
    fig = plt.figure(figsize=figsize_loc)  # np.array([15,10]) * 0.55 # proposalv2
    ax = fig.add_subplot(111)

    ax.set_xlabel("Azimuthal Angle [deg]")
    ax.set_xlim([-2, 92])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xticks([0, 15, 30, 45, 60, 75, 90])
    ax.set_ylabel(labels[dataField])

    # loop over mass/distance/sP bins
    loadHI = not all(NHI is None for NHI in min_NHI)
    ls_massbins = []
    colors = sampleColorTable("plasma", len(massBins) * len(distBins) * len(min_NHI) * len(sPs), bounds=[0.1, 0.7])
    # colors = sampleColorTable('plasma', 5, bounds=[0.1,0.8])

    for k, sP in enumerate(sPs):
        # load
        gc = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log", "central_flag", "SubhaloPos"])

        # global pre-cache of selected fields into memory
        if 0 and fullbox:
            # restrict to sub-volumes around targets
            print("Caching [Coordinates] now...", flush=True)
            pos = sP.snapshotSubsetP("gas", "pos", float32=True)

            # mask
            mask = np.zeros(pos.shape[0], dtype="bool")

            with np.errstate(invalid="ignore"):
                for massBin in massBins:
                    subInds = np.where(
                        (gc["mstar_30pkpc_log"] > massBin[0])
                        & (gc["mstar_30pkpc_log"] < massBin[1])
                        & gc["central_flag"]
                    )[0]
                    for i, subInd in enumerate(subInds):
                        print(" mask [%3d of %3d] ind = %d" % (i, len(subInds), subInd), flush=True)
                        dists = sP.periodicDistsN(gc["SubhaloPos"][subInd, :], pos, squared=True)
                        w = np.where(dists <= size**2)  # confortable padding, only need d<sqrt(2)*size/2
                        mask[w] = 1

            pInds = np.nonzero(mask)[0]
            mask = None
            dists = None
            print(" masked particle fraction = %.3f%%" % (pInds.size / pos.shape[0] * 100))

            pos = pos[pInds, :]

            # insert into cache, load other fields
            dataCache = {}
            dataCache["snap%d_gas_Coordinates" % sP.snap] = pos
            for key in ["Masses", "GFM_Metallicity", "Density", "NeutralHydrogenAbundance"]:
                print("Caching [%s] now..." % key, flush=True)
                dataCache["snap%d_gas_%s" % (sP.snap, key.replace(" ", "_"))] = sP.snapshotSubsetP(
                    "gas", key, inds=pInds
                )

            print("All caching done.", flush=True)

        for i, massBin in enumerate(massBins):
            with np.errstate(invalid="ignore"):
                w = np.where(
                    (gc["mstar_30pkpc_log"] > massBin[0]) & (gc["mstar_30pkpc_log"] < massBin[1]) & gc["central_flag"]
                )
            subInds = w[0]

            print(
                "%s z = %.1f [%.2f - %.2f] Processing [%d] halos..."
                % (sP.simName, sP.redshift, massBin[0], massBin[1], len(w[0]))
            )

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
                subInds,
            )
            if weightField != "mass":
                hashStr += weightField
            m = hashlib.sha256(hashStr.encode("utf-8")).hexdigest()[::4]
            cacheFile = sP.cachePath + "aziangle_grids_%s_%s.hdf5" % (dataField, m)
            cacheFileHI = sP.cachePath + "aziangle_grids_%s_%s.hdf5" % ("HI", m)

            if path.isfile(cacheFile) and (not loadHI or path.isfile(cacheFileHI)):
                # load cached result
                with h5py.File(cacheFile, "r") as f:
                    grid_global = f["grid_global"][()]
                if loadHI:
                    with h5py.File(cacheFileHI, "r") as f:
                        nhi_global = f["nhi_global"][()]
                print("Loaded: [%s]" % cacheFile)
            else:
                # compute now
                grid_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")
                nhi_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")

                for j, subhaloInd in enumerate(subInds):
                    haloID = sP.groupCatSingle(subhaloID=subhaloInd)["SubhaloGrNr"]

                    class plotConfig:
                        saveFilename = "dummy"

                    panels = [{"partType": "gas", "partField": dataField}]
                    grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                    # flatten
                    grid_global[:, j] = grid.ravel()

                    # enforce minimum HI column? then load now
                    if loadHI:
                        panels = [{"partType": "gas", "partField": "HI"}]
                        grid_nhi, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)
                        nhi_global[:, j] = grid_nhi.ravel()

                # flatten (ignore which halo each pixel came from)
                grid_global = grid_global.ravel()
                nhi_global = nhi_global.ravel()

                # save cache
                with h5py.File(cacheFile, "w") as f:
                    f["grid_global"] = grid_global
                if loadHI:
                    with h5py.File(cacheFileHI, "w") as f:
                        f["nhi_global"] = nhi_global

                print("Saved: [%s]" % cacheFile)

            # rearrange theta and dist into same shape
            dist_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")
            theta_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")

            for j, subhaloInd in enumerate(subInds):
                if distRvir:
                    haloID = sP.groupCatSingle(subhaloID=subhaloInd)["SubhaloGrNr"]
                    haloRvir_code = sP.groupCatSingle(haloID=haloID)["Group_R_Crit200"]
                    dist_global[:, j] = dist.ravel() / sP.units.codeLengthToKpc(haloRvir_code)
                else:
                    dist_global[:, j] = dist.ravel()
                theta_global[:, j] = theta.ravel()

            dist_global = dist_global.ravel()
            theta_global = theta_global.ravel()

            # bin on the global concatenated grids
            for j, (distBin, NHI) in enumerate(itertools.product(distBins, min_NHI)):
                if NHI is None:
                    w = np.where((dist_global >= distBin[0]) & (dist_global < distBin[1]))
                else:
                    w = np.where((dist_global >= distBin[0]) & (dist_global < distBin[1]) & (nhi_global >= NHI))

                # median metallicity as a function of theta, 1 degree bins
                theta_vals, hist, hist_std, hist_percs = running_median(
                    theta_global[w], grid_global[w], nBins=nThetaBins, percs=percs
                )

                # label and color
                label = "b = %d kpc" % np.mean(distBin) if i == 0 else ""
                if len(distBins) == 1:
                    label = ""
                if addEagle:
                    label = sP.simName
                # if len(massBins) > 1: label += '($M_\\star = %.1f$' % np.mean(massBin)
                if NHI is not None:
                    label += " N$_{\\rm HI} > 10^{%d}$ cm$^{-2}$" % NHI
                if NHI is None and len(min_NHI) > 1:
                    label += " N$_{\\rm HI}$ > 0"
                if len(sPs) > 1 and sPs[1].res != sPs[0].res:
                    label += " (%s)" % sP.simName
                if len(sPs) > 1 and sPs[1].redshift != sPs[0].redshift:
                    label += "z = %.1f" % sP.redshift

                # always vary color with each line
                c = colors[i + j + k]
                ls = ":"
                if (
                    np.mean(massBin) == 9.5
                    and np.mean(distBin) == 100
                    and np.abs(sP.redshift - 0.5) < 0.01
                    and NHI is None
                ):  # fiducial
                    ls = "-"

                ls_massbins.append(ls)

                # plot line and shaded band
                (l,) = ax.plot(theta_vals, hist, linestyle=ls, lw=lw, label=label, color=c)
                if i == 0 or k == 0:
                    ax.fill_between(theta_vals, hist_percs[0, :], hist_percs[-1, :], color=l.get_color(), alpha=0.1)

            # EAGLE: load and plot (from Freeke) (distBin == [95,105])
            if addEagle:
                if np.mean(massBin) == 9.5:
                    fname = "/u/dnelson/plots/celine.Ztheta/Z_Mhalo9p5.txt"
                if np.mean(massBin) == 10.5:
                    fname = "/u/dnelson/plots/celine.Ztheta/Z_Mhalo10p5.txt"
                with open(fname) as f:
                    lines = f.readlines()

                eagle_theta = [float(line.split()[0]) for line in lines]
                eagle_z = [np.log10(float(line.split()[1])) for line in lines]
                eagle_z_down = [np.log10(float(line.split()[2])) for line in lines]
                eagle_z_up = [np.log10(float(line.split()[3])) for line in lines]

                (l,) = ax.plot(eagle_theta, eagle_z, ls, lw=lw, label="EAGLE")

                ax.fill_between(eagle_theta, eagle_z_down, eagle_z_up, color=l.get_color(), alpha=0.1)

    # observational data from Glenn
    if addObs:
        opts = {"color": "#000000", "ecolor": "#000000", "alpha": 0.9, "capsize": 0.0, "fmt": "o"}

        theta = [85.2, 30.4, 8.2, 16.6, 5.8, 86.6]
        theta_errdown = [3.7, 0.4, 5.0, 0.1, 0.5, 1.2]
        theta_errup = [3.7, 0.3, 3.0, 0.1, 0.4, 1.5]
        metal = [-1.32, -1.33, -1.69, -1.48, -0.35, -2.18]
        metal_errdown = [0.15, 0.71, 2.0, 0.02, 0.07, 0.04]
        metal_errup = [0.15, 0.66, 0.0, 0.04, 0.03, 0.03]

        ax.errorbar(
            theta,
            metal,
            yerr=[metal_errdown, metal_errup],
            xerr=[theta_errdown, theta_errup],
            **opts,
            label="Existing Data",
        )

    # second legend?
    if len(massBins) > 1:
        sExtra = []
        lExtra = []
        for i, massBin in enumerate(massBins):
            ls = ls_massbins[i]  # if len(distBins) > 1 else '-'
            c = colors[i] if len(distBins) == 1 else "black"
            sExtra.append(plt.Line2D([0], [0], color=c, lw=lw, marker="", linestyle=ls))
            lExtra.append("M$_\\star$ = %.1f" % np.mean(massBin))
        legend2 = ax.legend(sExtra, lExtra, ncol=2, loc="best")
        ax.add_artist(legend2)

    # finish and save plot
    ax.legend(ncol=(1 if len(distBins) == 1 and len(sPs) == 1 else 2), loc="best")
    sPstr = "-".join(sP.simName for sP in sPs)
    mstarStr = "Mstar=%.1f" % np.mean(massBins[0]) if len(massBins) == 1 else "Mstar=%dbins" % len(massBins)
    distStr = "b=%d" % np.mean(distBins[0]) if len(distBins) == 1 else "b=%dbins" % len(distBins)
    nhiStr = "" if (len(min_NHI) == 1 and min_NHI[0] is None) else "_NHI=%dvals" % len(min_NHI)
    fig.savefig("%s_vs_theta_%s_%s_%s%s.pdf" % (dataField.replace(" ", ""), sPstr, mstarStr, distStr, nhiStr))
    plt.close(fig)


def stackedImageProjection():
    """Testing."""
    sP = simParams(run="tng50-1", redshift=0.5)

    dataField = "Mg II"
    label = "Median Mg II Column Density [log cm$^{-2}$]"

    massBins = [[8.48, 8.52], [8.97, 9.03], [9.45, 9.55], [9.95, 10.05], [10.4, 10.6], [10.7, 10.9]]
    distRvir = True

    # grid config (must recompute grids)
    method = "sphMap_global"
    nPixels = [800, 800]
    axes = [0, 1]  # random rotation
    size = 3.0
    sizeType = "rVirial"
    weightField = "mass"  #'MHI_GK' #'hi mass' # 'mass' is the default
    partType = "gas"
    partField = dataField

    panels = []

    # load
    gc = sP.subhalos(["mstar_30pkpc_log", "central_flag", "rhalo_200_code", "SubhaloPos"])

    # global pre-cache of selected fields into memory
    if 0:
        # restrict to sub-volumes around targets
        print("Caching [Coordinates] now...", flush=True)
        pos = sP.snapshotSubsetP("gas", "pos", float32=True)

        # mask
        mask = np.zeros(pos.shape[0], dtype="bool")

        with np.errstate(invalid="ignore"):
            for massBin in massBins:
                subInds = np.where(
                    (gc["mstar_30pkpc_log"] > massBin[0]) & (gc["mstar_30pkpc_log"] < massBin[1]) & gc["central_flag"]
                )[0]
                for i, subInd in enumerate(subInds):
                    print(" mask [%3d of %3d] ind = %d" % (i, len(subInds), subInd), flush=True)
                    dists = sP.periodicDistsN(gc["SubhaloPos"][subInd, :], pos, squared=True)
                    size_loc = size * gc["rhalo_200_code"][subInd]  # rvir -> code units
                    w = np.where(dists <= size_loc**2)  # confortable padding, only need d<sqrt(2)*size/2
                    mask[w] = 1

        pInds = np.nonzero(mask)[0]
        mask = None
        dists = None
        print(" masked particle fraction = %.3f%%" % (pInds.size / pos.shape[0] * 100))

        pos = pos[pInds, :]

        # insert into cache, load other fields
        dataCache = {}
        dataCache["snap%d_gas_Coordinates" % sP.snap] = pos
        for key in ["Masses", "Density", dataField + " mass"]:  # Density for Volume -> cellrad
            print("Caching [%s] now..." % key, flush=True)
            dataCache["snap%d_gas_%s" % (sP.snap, key.replace(" ", "_"))] = sP.snapshotSubsetP("gas", key, inds=pInds)

        print("All caching done.", flush=True)

    # loop over mass bins
    stacks = []

    for i, massBin in enumerate(massBins):
        # select subhalos
        with np.errstate(invalid="ignore"):
            w = np.where(
                (gc["mstar_30pkpc_log"] > massBin[0]) & (gc["mstar_30pkpc_log"] < massBin[1]) & gc["central_flag"]
            )
        sub_inds = w[0]

        print(
            "%s z = %.1f [%.2f - %.2f] Processing [%d] halos..."
            % (sP.simName, sP.redshift, massBin[0], massBin[1], len(w[0]))
        )

        # check for existence of cache
        hashStr = "%s-%s-%s-%s-%s-%s-%s" % (method, nPixels, axes, size, sizeType, sP.snap, sub_inds)
        m = hashlib.sha256(hashStr.encode("utf-8")).hexdigest()[::4]
        cacheFile = sP.cachePath + "stacked_proj_grids_%s_%s.hdf5" % (dataField, m)

        # plot config
        class plotConfig:
            plotStyle = "edged"
            rasterPx = nPixels[0]
            colorbars = True
            # fontsize     = 24
            saveFilename = "./stacked_%s_%d.pdf" % (dataField, i)

        if path.isfile(cacheFile):
            # load cached result
            with h5py.File(cacheFile, "r") as f:
                grid_global = f["grid_global"][()]
                sub_inds = f["sub_inds"][()]
            print("Loaded: [%s]" % cacheFile)
        else:
            # allocate for full stack
            grid_global = np.zeros((nPixels[0], nPixels[1], len(sub_inds)), dtype="float32")

            for j, subhaloInd in enumerate(sub_inds):  # noqa: B007
                # render
                grid, _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                # stamp
                grid_global[:, :, j] = grid

            # save cache
            with h5py.File(cacheFile, "w") as f:
                f["grid_global"] = grid_global
                f["sub_inds"] = sub_inds

            print("Saved: [%s]" % cacheFile)

        # create stack
        grid_stacked = np.nanmedian(grid_global, axis=2)
        stacks.append({"grid": grid_stacked, "sub_inds": sub_inds})

        # make plot of this mass bin
        # panels[0]['grid'] = grid_stacked # override
        # panels[0]['subhaloInd'] = sub_inds[int(len(sub_inds)/2)] # dummy
        # renderSingleHalo(panels, plotConfig, locals())

    # make final plot
    labelScale = "physical"
    valMinMax = [8.0, 14.5]

    contour = ["gas", dataField]
    contourLevels = [11.5, 14.0]  # [11.0, 14.0] # 1/cm^2, corresponding to EW~0.002 Ang and EW~1 Ang (from COG)
    contourOpts = {"colors": "white", "alpha": 0.8}
    contourSmooth = 3.0

    class plotConfig:
        plotStyle = "open"
        rasterPx = nPixels[0] * 2
        colorbars = True
        # fontsize     = 24
        saveFilename = "./stack_%s_z%.1f_%s.pdf" % (sP.simName, sP.redshift, dataField)

    for i, massBin in enumerate(massBins):
        if i % 2 == 0:
            continue  # only every other

        p = {
            "grid": stacks[i]["grid"],
            "labelZ": True if i == len(massBins) - 1 else False,
            "subhaloInd": stacks[i]["sub_inds"][int(len(stacks[i]["sub_inds"]) / 2)],
            "title": r"log M$_{\rm \star}$ = %.1f M$_\odot$" % np.mean(massBin),
        }

        panels.append(p)

    renderSingleHalo(panels, plotConfig, locals())


def paperPlots():
    """Driver."""
    redshift = 0.5
    TNG50 = simParams(run="tng50-1", redshift=redshift)

    ylim = [-2.0, -0.3]  # [-1.3,-0.6]
    sf = 0.8

    if 0:
        # figure 1: schematic visual
        singleHaloImage(TNG50, conf=0)
        singleHaloImage(TNG50, conf=4)

    if 0:
        # figure 2: TNG50 lines for massflow rate vs angle
        from temet.plot.gasflows import outflowRatesStacked

        mStarBins = [[10.3, 10.7]]  # [ [9.4,9.6],[10.3,10.7] ]

        config = {"radInd": 7, "vcutInd": 0, "stat": "mean", "ylim": [-15, 15], "skipZeros": False, "sterNorm": True}
        outflowRatesStacked(TNG50, quant="theta", mStarBins=mStarBins, config=config)

        config = {"radInd": 7, "vcutInd": 5, "stat": "mean", "ylim": [-15, 15], "skipZeros": False, "sterNorm": True}
        outflowRatesStacked(TNG50, quant="theta", mStarBins=mStarBins, config=config, inflow=True)

    if 0:
        # fig 3: metallicity vs rad/massflowrate
        distBin = [90, 110]
        ylim = [-4.0, 0.5]

        mstar = TNG50.subhalos("mstar_30pkpc_log")
        cen_flag = TNG50.subhalos("central_flag")
        subhaloIDs = np.where((mstar > 9.95) & (mstar < 10.05) & cen_flag)[0]
        print(subhaloIDs.size)

        metallicityVsVradProjected(TNG50, directCells=False, shIDs=subhaloIDs, distBin=distBin, ylim=ylim)

    if 0:
        # figure 4: main comparison of TNG vs EAGLE
        field = "metal_solar"
        massBins = [[9.46, 9.54]]
        distBins = [[95, 105]]

        metallicityVsTheta([TNG50], field, massBins=massBins, distBins=distBins, addEagle=True, ylim=ylim)

    if 0:
        # figure 5a: subplots for variation of (Z,theta) with M*
        field = "metal_solar"
        massBins = [[8.49, 8.51], [8.99, 9.01], [9.46, 9.54], [9.95, 10.05], [10.44, 10.56], [10.9, 11.1]]
        distBins = [[95, 105]]

        metallicityVsTheta([TNG50], field, massBins=massBins, distBins=distBins, sizefac=sf, ylim=ylim, fullbox=True)

    if 0:
        # figure 5a (variant): subplots for variation of (Z,theta) with M*, at rvir relative impact parameters
        field = "metal_solar"
        massBins = [[8.49, 8.51], [8.99, 9.01], [9.46, 9.54], [9.95, 10.05], [10.44, 10.56]]
        distBins = [[0.48, 0.52]]

        metallicityVsTheta(
            [TNG50], field, massBins=massBins, distBins=distBins, sizefac=sf, ylim=ylim, fullbox=True, distRvir=True
        )

    if 0:
        # figure 5b: subplots for variation of (Z,theta) with b
        field = "metal_solar"
        massBins = [[9.46, 9.54]]
        distBins = [[20, 30], [45, 55], [95, 105], [195, 205]]

        metallicityVsTheta([TNG50], field, massBins=massBins, distBins=distBins, sizefac=sf, ylim=ylim, fullbox=True)

    if 0:
        # figure 5c: subplots for variation of (Z,theta) with redshift
        field = "metal_solar"
        massBins = [[9.46, 9.54]]
        distBins = [[95, 105]]
        redshifts = [0.0, 0.5, 1.5, 2.5]

        sPs = []
        for redshift in redshifts:
            sPs.append(simParams(run="tng50-1", redshift=redshift))

        metallicityVsTheta(sPs, field, massBins=massBins, distBins=distBins, sizefac=sf, ylim=ylim, fullbox=True)

    if 0:
        # figure 5d: subplots for variation of (Z,theta) with minimum N_HI
        field = "metal_solar"
        massBins = [[9.46, 9.54]]
        distBins = [[70, 130]]
        min_NHI = [None, 13, 15, 17]  # 16, 18

        metallicityVsTheta(
            [TNG50],
            field,
            massBins=massBins,
            distBins=distBins,
            min_NHI=min_NHI,
            sizefac=sf,
            nThetaBins=45,
            ylim=ylim,
            fullbox=True,
        )

    if 0:
        # explore: resolution convergence
        field = "metal_solar"
        massBins = [[8.98, 9.02]]
        distBins = [[45, 55]]
        runs = ["tng50-1", "tng50-2", "tng50-3"]

        sPs = []
        for run in runs:
            sPs.append(simParams(run=run, redshift=redshift))

        metallicityVsTheta(sPs, field, massBins=massBins, distBins=distBins, sizefac=sf)

    if 0:
        # explore: metallicity vs rad/massflowrate (black background vis gallery)
        mstar = TNG50.subhalos("mstar_30pkpc_log")
        cen_flag = TNG50.subhalos("central_flag")
        subhaloIDs = np.where((mstar > 8.49) & (mstar < 8.51) & cen_flag)[0]

        for shID in subhaloIDs:
            metallicityVsVradProjected(TNG50, shIDs=[shID])

    if 0:
        TNG50 = simParams(run="tng50-1", redshift=0.0)

        # figure 1: schematic visual
        subhaloInd = 551973  # 549090, 551973, 565947, 567897, 572121, 574286, 575190, 579232
        # singleHaloImage(TNG50, subhaloInd=subhaloInd, conf=0)
        # singleHaloImage(TNG50, subhaloInd=subhaloInd, conf=4)
        singleHaloImage(TNG50, subhaloInd=subhaloInd, conf=5)
