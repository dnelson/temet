"""
James Bartlett / MAGIC mission proposal.
"""

import hashlib
from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from temet.cosmo.util import subsampleRandomSubhalos
from temet.plot import subhalos
from temet.plot.config import figsize, markers
from temet.util import simParams
from temet.util.helper import dist_theta_grid, logZeroNaN
from temet.vis.halo import renderSingleHalo


def magicCGMEmissionMaps(vis_indiv=False):
    """Emission maps (single, or in stacked M* bins) for MAGIC-II proposal.

    Args:
      vis_indiv (bool): if True, render visualizations of each subhalo.
    """
    sP = simParams(run="tng50-1", redshift=0.5)

    lines = [
        "O--6-1037.62A",
        "O--6-1031.91A",
        "Si-2-1808.01A",
        "Si-2-1526.71A",
        "Si-3-1206.50A",
        "H--1-1215.67A",
        "H--1-1025.72A",
        "C--3-977.020A",
        "Si-4-1393.75A",
        "Si-4-1402.77A",
        "C--4-1550.78A",
        "C--4-1548.19A",
        "He-2-1640.43A",
        "N--5-1238.82A",
        "N--5-1242.80A",
    ]

    # massBins = [ [8.48,8.52], [8.97,9.03], [9.45,9.55], [9.97, 10.03], [10.4,10.6], [10.8,11.2] ]
    massBins = [[9.99, 10.01]]

    # grid config (must recompute grids)
    method = "sphMap_global"
    nPixels = [800, 800]
    axes = [0, 1]  # random rotation
    size = 300.0  # 1000.0
    sizeType = "kpc"

    sP.createCloudyCache = True if "_global" in method else False

    valMinMax = [-22, -18]
    labelScale = "physical"

    # global pre-cache (to disk) of photoionization calculations
    if 0:
        for line in lines:
            lineName = line.replace("-", " ")
            print("Caching [%s] now..." % lineName, flush=True)
            x = sP.snapshotSubset("gas", "%s flux" % lineName, indRange=[0, 10])

    # load and select subhalos
    gc = sP.subhalos(["mstar_30pkpc_log", "central_flag", "rhalo_200_code", "SubhaloPos"])

    subInds = []

    for massBin in massBins:
        subInds_loc = np.where(
            (gc["mstar_30pkpc_log"] > massBin[0]) & (gc["mstar_30pkpc_log"] < massBin[1]) & gc["central_flag"]
        )[0]
        subInds.append(subInds_loc)

    # global pre-cache of selected fields into memory
    if 0:
        # restrict to sub-volumes around targets
        print("Caching [Coordinates] now...", flush=True)
        pos = sP.snapshotSubsetP("gas", "pos", float32=True)

        # mask
        mask = np.zeros(pos.shape[0], dtype="bool")

        with np.errstate(invalid="ignore"):
            for i in range(len(massBins)):
                for j, subInd in enumerate(subInds[i]):
                    print(" mask [%3d of %3d] ind = %d" % (j, len(subInds), subInd), flush=True)
                    dists = sP.periodicDistsN(gc["SubhaloPos"][subInd, :], pos, squared=True)

                    if sizeType == "kpc":
                        size_loc = sP.units.physicalKpcToCodeLength(size)  # kpc -> code units
                    elif sizeType == "rVirial":
                        size_loc = size * gc["rhalo_200_code"][subInd]  # rvir -> code units
                    else:
                        assert 0  # unhandled

                    w = np.where(dists <= size_loc**2)  # confortable padding, only need d<sqrt(2)*size/2
                    mask[w] = 1

                    print(i, j, size_loc, len(w[0]), flush=True)

        pInds = np.nonzero(mask)[0]
        mask = None
        dists = None
        print(" masked particle fraction = %.3f%%" % (pInds.size / pos.shape[0] * 100))

        pos = pos[pInds, :]

        # insert into cache, load other fields
        dataCache = {}
        dataCache["snap%d_gas_Coordinates" % sP.snap] = pos

        for key in ["Masses", "Density"]:  # Density for Volume -> cellrad
            print("Caching [%s] now..." % key, flush=True)
            dataCache["snap%d_gas_%s" % (sP.snap, key.replace(" ", "_"))] = sP.snapshotSubsetP("gas", key, inds=pInds)

        for line in lines:
            loadFieldName = "%s flux" % line
            saveKey = "snap%d_gas_%s" % (sP.snap, loadFieldName.replace(" ", "_").replace("-", "_"))
            print("Caching [%s] now..." % loadFieldName, flush=True)
            dataCache[saveKey] = sP.snapshotSubsetP("gas", loadFieldName, inds=pInds)

        sP.data = dataCache
        print("All caching done.", flush=True)

    # loop over mass bins
    stacks = []

    for i, massBin in enumerate(massBins):
        # select subhalos
        sub_inds = subInds[i]

        print(
            "%s z = %.1f [%.2f - %.2f] Processing [%d] halos..."
            % (sP.simName, sP.redshift, massBin[0], massBin[1], sub_inds.size),
            flush=True,
        )

        sub_inds = sub_inds[0:3]

        # plot config
        class plotConfig:
            plotStyle = "open"
            rasterPx = nPixels[0] * 2.3
            colorbars = True

        if vis_indiv:
            # render multiple views of each subhalo (in this mass bin)
            for sub_ind in sub_inds:
                panels = []
                print(f" vis individual {sub_ind = }")
                plotConfig.saveFilename = "./%s-s%d-sh%d.pdf" % (sP.simName, sP.snap, sub_ind)
                plotConfig.nRows = 3

                for j, line in enumerate(lines):
                    field = "sb_" + line + "_ergs"
                    panels.append(
                        {
                            "partType": "gas",
                            "partField": field,
                            "subhaloInd": sub_ind,
                            "labelHalo": (j == 0),
                            "labelSim": (j == 2),
                            "labelZ": (j == 2),
                        }
                    )

                renderSingleHalo(panels, plotConfig, locals())

        # loop over lines
        for line in lines:
            panels = [{"partType": "gas", "partField": "sb_" + line + "_ergs"}]
            print("Processing [%s]..." % line, flush=True)

            # check for existence of cache
            hashStr = "%s-%s-%s-%s-%s-%s-%s" % (method, nPixels, axes, size, sizeType, sP.snap, sub_inds)
            m = hashlib.sha256(hashStr.encode("utf-8")).hexdigest()[::4]
            cacheFile = sP.cachePath + "stacked_proj_grids_%s_%s.hdf5" % (panels[0]["partField"], m)

            if 0 and path.isfile(cacheFile):
                # load cached result
                with h5py.File(cacheFile, "r") as f:
                    grid_global = f["grid_global"][()]
                    sub_inds = f["sub_inds"][()]
                print("Loaded: [%s]" % cacheFile)
            else:
                # allocate for full stack
                grid_global = np.zeros((nPixels[0], nPixels[1], len(sub_inds)), dtype="float32")

                for j, sub_ind in enumerate(sub_inds):
                    # render and stamp
                    panels[0]["subhaloInd"] = sub_ind
                    grid_global[:, :, j], _ = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

                # save cache
                # with h5py.File(cacheFile,'w') as f:
                #    f['grid_global'] = grid_global
                #    f['sub_inds'] = sub_inds
                # print('Saved: [%s]' % cacheFile)

            # create stack
            # grid_stacked = np.nanmedian(grid_global, axis=2)
            # stacks.append({'grid':grid_stacked,'sub_inds':sub_inds})

            # make plot of this mass bin
            # plotConfig.saveFilename = './stacked_%d_%s.pdf' % (i,line)
            # panels[0]['grid'] = grid_stacked # override
            # panels[0]['subhaloInd'] = sub_inds[int(len(sub_inds)/2)] # dummy
            # renderSingleHalo(panels, plotConfig, locals())

    # make plot of stacked result over the mass bins
    labelScale = "physical"
    valMinMax = [8.0, 14.5]

    class plotConfig:
        plotStyle = "open"
        rasterPx = nPixels[0] * 2
        colorbars = True
        # fontsize     = 24
        saveFilename = "./stack_%s_z%.1f_%s.pdf" % (sP.simName, sP.redshift, panels[0]["partField"])

    panels = []
    for i, massBin in enumerate(massBins):
        # if i % 2 == 0: continue # only every other

        p = {
            "grid": stacks[i]["grid"],
            "labelZ": True if i == len(massBins) - 1 else False,
            "subhaloInd": stacks[i]["sub_inds"][int(len(stacks[i]["sub_inds"]) / 2)],
            "title": r"log M$_{\rm \star}$ = %.1f M$_\odot$" % np.mean(massBin),
        }

        panels.append(p)

    renderSingleHalo(panels, plotConfig, locals())


def magicCGMEmissionTrends():
    """Emission summary statisics (auxCat-based) as a function of galaxy properties, for MAGIC-II proposal."""
    sim = simParams(run="tng50-1", redshift=0.3)

    # fields = ['lum_civ1551_outercgm','lum_civ1551_innercgm']
    fields = ["lum_heii1640_outercgm", "lum_heii1640_innercgm"]

    # plot
    for field in fields:
        subhalos.median(
            [sim],
            yQuants=[field],
            xQuant="mstar_30pkpc",
            cenSatSelect="cen",
            xlim=[9.0, 11.5],
            ylim=[35.5, 46],
            drawMedian=False,
            markersize=40.0,
            scatterPoints=True,
            scatterColor="mass_smbh",  # 'ssfr_30pkpc'
            clim=[6.5, 8.5],  # [-9.0, -10.5] # log sSFR
            sizefac=0.8,  # for single column figure
            maxPointsPerDex=200,
            legendLoc="upper left",
            pdf=None,
        )


def hubbleMCT_emissionTrends(simname="tng50-1", cQuant=None):
    """Hubble MST Proposal 2024 of Kate Rubin, and MAGIC-2 proposal."""
    # sim = simParams(simname, redshift=0.36) # MCT
    sim = simParams(simname, redshift=0.5)  # MAGIC-2

    # grid config
    method = "sphMap"  # sphMap_global
    nPixels = [1000, 1000]
    axes = [0, 1]
    size = 300.0  # 35 arcsec @ z=0.36 (SBC field of view is 35"x31")
    sizeType = "kpc"

    sim.createCloudyCache = True if "_global" in method else False

    # config
    # fields = ['sb_OVI_ergs','sb_O--6-1037.62A_ergs','sb_CIII_ergs'] # MCT
    fields = [
        "sb_O--6-1037.62A_ergs",
        "sb_O--6-1031.91A_ergs",
        "sb_Si-2-1808.01A_ergs",
        "sb_Si-2-1526.71A_ergs",
        "sb_Si-3-1206.50A_ergs",
        "sb_H--1-1215.67A_ergs",
        "sb_H--1-1025.72A_ergs",
        "sb_C--3-977.020A_ergs",
        "sb_Si-4-1393.75A_ergs",
        "sb_Si-4-1402.77A_ergs",
        "sb_C--4-1550.78A_ergs",
        "sb_C--4-1548.19A_ergs",
        "sb_He-2-1640.43A_ergs",
        "sb_N--5-1238.82A_ergs",
        "sb_N--5-1242.80A_ergs",
    ]  # MAGIC-2
    percs = [25, 50, 75]
    distBins = [[20, 30], [45, 55]]  # pkpc

    # sample
    mstar_min = 9.0
    mstar_max = 11.5
    num_per_dex = 100

    subInds, mstar = subsampleRandomSubhalos(sim, num_per_dex, [mstar_min, mstar_max], cenOnly=True)

    dist, _ = dist_theta_grid(size, nPixels)

    # check for existence of cache
    grids = {}
    sb_percs = {}
    cacheFile = sim.cachePath + "magic2_grids_z05.hdf5"  # hstmst_grids.hdf5

    if path.isfile(cacheFile):
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
            sb_percs[field] = np.zeros((len(subInds), len(distBins), len(percs)), dtype="float32")

        # sb_percs['OVI'] = np.zeros((len(subInds),len(distBins),len(percs)), dtype='float32')
        # sb_percs['CIII'] = np.zeros((len(subInds),len(distBins),len(percs)), dtype='float32')

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

                # MCST: OVI doublet map and CIII map separately
                # OVI_map = np.log10(10.0**grids['sb_OVI_ergs'][:,:,i] + 10.0**grids['sb_O--6-1037.62A_ergs'][:,:,i])
                # CIII_map = grids['sb_CIII_ergs'][:,:,i]

                # sb_percs['OVI'][i,j,:] = np.percentile(OVI_map[w], percs)
                # sb_percs['CIII'][i,j,:] = np.percentile(CIII_map[w], percs)

                # MAGIC-2
                for field in fields:
                    loc_grid = np.squeeze(grids[field][:, :, i])
                    sb_percs[field][i, j, :] = np.percentile(loc_grid[w], percs)

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

    # write text file
    with open("magic2_SBs_annuli_vs_mstar.txt", "w") as f:
        f.write("# %s z = %.1f\n" % (sim.name, sim.redshift))
        f.write("# fields: %s\n" % ", ".join([f.replace("sb_", "").replace("_ergs", "") for f in fields]))
        f.write("# distance bins [pkpc]: %s\n" % distBins)
        f.write("# percentiles: %s\n" % percs)
        f.write("# columns: subhaloInd mstar SBs\n")
        f.write('# note: "SBs" are ordered by field, dist bin, and then percentile (6 entries in a row).\n')

        for i, subhaloInd in enumerate(subInds):
            s = "%6d %6.2f" % (subhaloInd, mstar[i])
            for field in fields:
                for j in range(len(distBins)):
                    for k in range(len(percs)):
                        s += " %.2f" % sb_percs[field][i, j, k]
            f.write(s + "\n")

    # MAGIC-2: multiple axes, grouping SBs by species
    species = list({line.replace("sb_", "").split("-", 1)[0] for line in fields})
    nrows = int(np.sqrt(len(species)))
    ncols = int(np.ceil(len(species) / nrows))

    for i, distBin in enumerate(distBins):
        # start figure
        # fig, ax = plt.subplots(figsize=figsize) # MCST
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(figsize[0] * nrows, figsize[1] * ncols))  # MAGIC-2

        for species_name, ax in zip(species, axes.flatten()):  # MAGIC-2
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

            # for i, distBin in enumerate(distBins):
            # for line, label in zip(['OVI','CIII'], ['OVI 1032+1038','CIII 977']): # MCST
            for line in sb_percs.keys():  # MAGIC-2
                if species_name + "-" not in line:  # MAGIC-2
                    continue  # MAGIC-2

                label = line.replace("sb_", "").replace("_ergs", "").replace("-", " ").replace("A", r"$\AA$")
                y_mid = sb_percs[line][:, i, 1]
                y_err_lo = sb_percs[line][:, i, 1] - sb_percs[line][:, i, 0]
                y_err_hi = sb_percs[line][:, i, 2] - sb_percs[line][:, i, 1]

                label_loc = r"%s (%d$\pm$%d kpc)" % (label, np.mean(distBin), (distBin[1] - distBin[0]) / 2)
                if cQuant is None:
                    ax.errorbar(mstar, y_mid, yerr=[y_err_lo, y_err_hi], fmt="o", label=label_loc)
                else:
                    opts = {
                        "vmin": cMinMax[0],
                        "vmax": cMinMax[1],
                        "c": sim_cvals,
                        "cmap": cmap,
                        "marker": markers[count],
                    }
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

        # 'mst_OVI_CIII_annuli_vs_mstar_%s.pdf'
        fig.savefig("magic2_SBs_annuli_vs_mstar_%s_%dkpc.pdf" % (sim.simName, np.mean(distBin)))
        plt.close(fig)
