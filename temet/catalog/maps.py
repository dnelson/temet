"""
Cosmological simulations - auxiliary (map-based) catalogs for additional derived properties.
"""

import h5py
import numpy as np

from ..util.helper import dist_theta_grid


def projections(
    sP,
    partType="gas",
    partField="coldens_msunkpc2",
    conf=0,
    method="sphMap_globalZoomOrig",
    cenSatSelect="cen",
    m200_min=14.0,
    saveImages=False,
):
    """Generate projections for a given configuration, save results into a single postprocessing file.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      partType (str): particle type to project (e.g. 'gas', 'stars').
      partField (str): field to project (e.g. 'coldens_msunkpc2').
      conf (int): projection configuration index (0-3).
      method (str): mapping method to use. If 'sphMap_globalZoomOrig', all particles from original zoom run are used
        (TNG-Cluster only). Otherwise, 'sphMap' or 'sphMap_global' are reasonable choices.
      cenSatSelect (str): 'cen', 'sat', or 'all' to select central, satellite, or all subhalos.
      m200_min (float): minimum M200c [log10(Msun)] for halos to include. If None, include all halos.
      saveImages (bool): if True, render and save actual plot images as well.
    """
    from ..vis.halo import renderSingleHalo

    axes_set = [[0, 1], [0, 2], [1, 2]]
    nPixels = 2000
    depthType = "rVirial"  # r200c

    if conf == 0:
        sizeType = "rVirial"  # r200c
        size = 4.0  # +/- 2 rvir in extent
        depth = 2.0  # +/- 1 rvir in depth
        confStr = "2r200_d=r200"
    if conf == 1:
        sizeType = "r500"
        size = 1.0  # +/- 0.5 r500 in extent
        depth = 2.0  # +/- 1 rvir in depth
        confStr = "0.5r500_d=r200"
    if conf == 2:
        sizeType = "r500"
        size = 1.0  # +/- 0.5 r500 in extent
        depth = 6.0  # +/- 3 rvir in depth
        confStr = "0.5r500_d=3r200"
    if conf == 3:
        sizeType = "r500"
        depthType = "r500"
        size = 2.0  # +/- r500 in extent
        depth = 2.0  # +/- r500 in depth
        confStr = "r500_d=r500"

    # select halos
    subhaloIDs = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

    if m200_min is not None:
        # all halos with M200c > m200_min, TNG-Cluster: all primary zoom targets
        m200c = sP.subhalos("mhalo_200_log")[subhaloIDs]

        if sP.simName == "TNG-Cluster":
            # for z>0, also include main progenitors of z==0 GroupPrimaryZoomTargets
            if sP.redshift > 0:
                subhaloIDs = list(subhaloIDs)
                sP_z0 = sP.copy()
                sP_z0.setRedshift(0.0)

                # load GroupPrimaryZoomTargets at z=0, and their SubLink MPBs
                subhaloIDs_z0 = sP_z0.cenSatSubhaloIndices(cenSatSelect="cen")
                mpbs = sP_z0.loadMPBs(subhaloIDs_z0, fields=["SnapNum", "SubfindID"])

                # record any progenitors which are not HaloID==0 at this snap, in the original zooms
                subhaloIDs_add = []
                for subhaloID_z0 in subhaloIDs_z0:
                    mpb_loc = mpbs[subhaloID_z0]
                    w = np.where(mpb_loc["SnapNum"] == sP.snap)[0]
                    if len(w) == 0:
                        print("NO PROG, SKIP!", subhaloID_z0)
                        continue
                    subhaloID_cand = mpb_loc["SubfindID"][w][0]
                    if subhaloID_cand not in subhaloIDs:
                        print("ADD: ", subhaloID_z0, subhaloID_cand)
                        subhaloIDs.append(subhaloID_cand)
                subhaloIDs = np.array(subhaloIDs, dtype="int32")
        else:
            # include low-mass progenitors at high redshift
            w = np.where(m200c > m200_min)[0]
            subhaloIDs = subhaloIDs[w]

    haloIDs = sP.subhalos("SubhaloGrNr")[subhaloIDs]

    r200c = sP.subhalos("r200")[subhaloIDs]
    r500c = sP.subhalos("r500")[subhaloIDs]

    # render images?
    if saveImages:
        # plot config
        class plotConfig:
            plotStyle = "open"  # open, edged
            rasterPx = [800, 800]
            colorbars = True
            fontsize = 14

        labelHalo = "mhalo,haloidorig"
        labelSim = True
        labelZ = True
        labelScale = True
        if partField == "Mg II":
            valMinMax = [8.0, 15.0]

        # loop over all halos
        for subhaloInd in subhaloIDs:
            # loop over all projection directions
            for axes in axes_set:
                # render and stamp
                panels = [{}]

                projAxis = ["x", "y", "z"][3 - np.sum(axes)]
                saveStr = "%s-%s_%s_%s" % (partType, partField, confStr, projAxis)
                labelCustom = [confStr.replace("_", " ") + r" ($\hat{%s}$)" % projAxis]
                plotConfig.saveFilename = "%s.%d.%08d.%s.png" % (sP.simName, sP.snap, subhaloInd, saveStr)

                renderSingleHalo(panels, plotConfig, locals())  # , skipExisting=False)

        print("Done.")
        return

    # save projections (instead of rendering images): start save file
    if " " in partField:
        partField = "coldens_" + partField.replace(" ", "")

    savePath = sP.postPath + "projections/"
    saveFilename = savePath + "%s-%s__%s.%d.hdf5" % (partType, partField, confStr, sP.snap)

    with h5py.File(saveFilename, "a") as f:
        f.attrs["axes_set"] = axes_set
        f.attrs["nPixels"] = nPixels
        f.attrs["size"] = size
        f.attrs["sizeType"] = sizeType
        f.attrs["method"] = method
        f.attrs["partType"] = partType
        f.attrs["partField"] = partField
        f.attrs["simName"] = sP.simName
        f.attrs["snap"] = sP.snap
        f.attrs["depth"] = depth
        f.attrs["depthType"] = depthType

        if "HaloIDs" not in f:
            f["HaloIDs"] = haloIDs
            f["SubhaloIDs"] = subhaloIDs
            f["r200c"] = r200c
            f["r500c"] = r500c

            dist, theta = dist_theta_grid(size, nPixels)

            f["grid_dist"] = dist
            f["grid_angle"] = theta

    # loop over all halos
    for i, haloID in enumerate(haloIDs):
        # check for existence
        subhaloInd = subhaloIDs[i]
        gName = f"Halo_{haloID}"

        print(f"[{i:03d} of {len(haloIDs):03d}] Halo ID = {haloID}")

        with h5py.File(saveFilename, "r") as f:
            if gName in f:
                print(" skip")
                continue

        # loop over all orientations
        grids = np.zeros((nPixels, nPixels, len(axes_set)), dtype="float32")

        for j, axes in enumerate(axes_set):  # noqa: B007
            # render and stamp
            panels = [{}]

            class plotConfig:
                pass

            grid_loc, config = renderSingleHalo(panels, plotConfig, locals(), returnData=True)
            grids[:, :, j] = grid_loc

        # save
        with h5py.File(saveFilename, "a") as f:
            f.create_dataset(gName, data=grids)

            f.attrs["name"] = config["label"]

            f[gName].attrs["minmax_guess"] = config["vMM_guess"]
            f[gName].attrs["box_center"] = config["boxCenter"]  # code
            # box_size in [code], multiply by grid_dist to have px dists in code units for this halo
            f[gName].attrs["box_size"] = config["boxSizeImg"]

    print("Done.")


def _find_peak(grid, method="center_of_light"):
    """Find peak location in a 2D grid using different methods. Return is in pixel coordinates."""
    x = np.arange(grid.shape[0])
    y = np.arange(grid.shape[1])

    grid_norm = grid / grid.max()

    xx, yy = np.meshgrid(x, y, indexing="xy")

    if method == "max_pixel":
        # single maximum valued pixel
        ind = np.argmax(grid)
        xy_cen = np.unravel_index(ind, grid.shape)[::-1]

    if method == "center_of_light":
        # center of light
        xy_cen = np.average(xx, weights=grid_norm), np.average(yy, weights=grid_norm)

    if method == "shrinking_circle":
        # iterative/shrinking circle
        npx = grid.shape[0] * grid.shape[1]
        rad = grid.shape[0] / 2
        xy_cen = np.array([grid.shape[0] / 2, grid.shape[1] / 2], dtype="float32")
        iter_count = 0

        circ_cens = []
        circ_rads = []

        while npx > 100 and iter_count < 500:
            dx = xx - xy_cen[0]
            dy = yy - xy_cen[1]
            rr = np.sqrt(dx**2 + dy**2)

            ind = np.where(rr < rad)
            npx = len(ind[0])

            xy_cen = np.average(xx[ind], weights=grid_norm[ind]), np.average(yy[ind], weights=grid_norm[ind])
            rad *= 0.95

            iter_count += 1

            # print(iter_count, npx, xy_cen, rad)
            circ_cens.append(xy_cen)
            circ_rads.append(rad)

        return xy_cen, circ_cens, circ_rads

    return xy_cen


def summarize_projection_2d(sim, pSplit=None, quantity="sz_yparam", projConf="2r200_d=r200", op="sum", aperture="r500"):
    """Calculate summary statistic(s) from existing projections in 2D, e.g. Y_{r500,2D} for SZ."""
    # config
    assert pSplit is None
    assert op in ["sum", "peak_offset"]

    path = sim.postPath + "projections/"
    filename = "gas-%s_%03d_%s.hdf5" % (quantity, sim.snap, projConf)

    # load list of halos
    with h5py.File(path + filename, "r") as f:
        haloIDs = f["HaloIDs"][()]
        subhaloIDs = f["SubhaloIDs"][()]
        nproj = f["Halo_%d" % haloIDs[0]].shape[2]
        name = f.attrs["name"]

    # allocate
    rr = np.zeros((len(haloIDs), nproj), dtype="float32")

    # load distance grid
    with h5py.File(path + filename, "r") as f:
        dist = f["grid_dist"][()]

    # loop over all halos
    for haloInd, haloID in enumerate(haloIDs):
        # status
        if haloInd % 10 == 0:
            print(" %4.1f%%" % (float(haloInd + 1) * 100.0 / len(haloIDs)))

        # load three projections
        with h5py.File(path + filename, "r") as f:
            proj = f["Halo_%d" % haloID][()]
            box_size = f["Halo_%d" % haloID].attrs["box_size"]

        # halo-dependent unit conversions
        assert box_size[0] == box_size[1] and proj.shape[0] == proj.shape[1]
        dists_code_loc = dist * (box_size[0] / 2)

        pxSize_code = box_size[0] / proj.shape[0]
        pxSize_Kpc = sim.units.codeLengthToKpc(pxSize_code)
        pxArea_Kpc = pxSize_Kpc**2

        if op == "sum":
            # halo-specific aperture [code units]
            if aperture == "r500":
                aperture_rad = sim.halo(haloID)["Group_R_Crit500"]
            if aperture == "r200":
                aperture_rad = sim.halo(haloID)["Group_R_Crit200"]

            assert aperture_rad < box_size[0] / 2.0 + 1e-6

            # select spatial region
            w_px_spatial = np.where(dists_code_loc < aperture_rad)

        # loop over each projection direction
        for i in range(nproj):
            # linear map [e.g. dimensionless for SZY, erg/s/kpc^2 for LX]
            proj_loc = 10.0 ** (proj[:, :, i]).astype("float64")

            if op == "sum":
                # integrate within aperture
                quant_sum = np.nansum(proj_loc[w_px_spatial])

                # multiply by total area in [pKpc^2]
                # e.g. for SZ, this is [dimensionless] -> [pKpc^2]
                # e.g. for LX, this is [erg/s/kpc^2] -> [erg/s]
                quant_sum *= pxArea_Kpc

                rr[haloInd, i] = np.log10(quant_sum)

            if op == "peak_offset":
                # mark galaxy position (SubhaloPos at center by definition)
                xy_cen0 = proj.shape[0] / 2.0, proj.shape[1] / 2.0

                xy_cen2, _, _ = _find_peak(proj_loc, method="shrinking_circle")
                offsets_px = np.sqrt((xy_cen0[0] - xy_cen2[0]) ** 2 + (xy_cen0[1] - xy_cen2[1]) ** 2)
                offsets_code = offsets_px * pxSize_code

                rr[haloInd, i] = offsets_code

    # return quantities for save, as expected by load.auxCat()
    if op == "sum":
        units = name.split("[")[-1].split("]")[0] + " pkpc^2"
        desc = "Sum of [%s] in 2D projection (%s) [%s]." % (quantity, projConf, units)
    if op == "peak_offset":
        units = "code_length"
        desc = "Spatial offset of peak [%s] in 2D projection (%s) [%s]." % (quantity, projConf, units)

    select = ""
    if subhaloIDs.size == sim.nSubhalos:
        select = "All subhalos."
    elif sim.simName == "TNG-Cluster":
        select = "TNG-Cluster primary zoom targets only."

    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "ptType": "gas".encode("ascii"),
        "subhaloIDs": subhaloIDs,
    }

    return rr, attrs
