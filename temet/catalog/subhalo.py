"""
Cosmological simulations - auxiliary (subhalo-based) catalogs for additional derived properties.
"""

from getpass import getuser
from os.path import expanduser

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

from ..catalog.common import _radialRestriction, findHalfLightRadius, pSplitBounds, userCustomFields
from ..util.helper import binned_statistic_weighted, logZeroMin, logZeroNaN, mvbe
from ..util.rotation import (
    ellipsoidfit,
    momentOfInertiaTensor,
    rotateCoordinateArray,
    rotationMatricesFromInertiaTensor,
    rotationMatrixFromVec,
)
from ..util.sphMap import sphMap
from ..util.treeSearch import buildFullTree, calcHsml, calcParticleIndices


def _process_custom_func(sP, op, ptProperty, gc, subhaloID, particles, rr, i0, i1, wValid, opts):
    """Handle custom logic for user defined functions and other non-standard 'reduction' operations.

    Operate on a given particle set (e.g. those of a single subhalo).
    """
    # ufunc: kappa rot
    if ptProperty == "Krot":
        # minimum two star particles
        if len(wValid[0]) < 2:
            return [np.nan, np.nan, np.nan, np.nan]

        stars_pos = np.squeeze(particles["Coordinates"][i0:i1, :][wValid, :])
        stars_vel = np.squeeze(particles["Velocities"][i0:i1, :][wValid, :])
        stars_mass = particles["Masses"][i0:i1][wValid].reshape((len(wValid[0]), 1))

        # velocity of stellar CoM
        sub_stellarMass = stars_mass.sum()
        sub_stellarCoM_vel = np.sum(stars_mass * stars_vel, axis=0) / sub_stellarMass

        # positions relative to most bound star, velocities relative to stellar CoM vel
        for j in range(3):
            stars_pos[:, j] -= stars_pos[0, j]
            stars_vel[:, j] -= sub_stellarCoM_vel[j]

        sP.correctPeriodicDistVecs(stars_pos)
        stars_pos = sP.units.codeLengthToKpc(stars_pos)  # kpc
        stars_vel = sP.units.particleCodeVelocityToKms(stars_vel)  # km/s
        stars_rad_sq = stars_pos[:, 0] ** 2.0 + stars_pos[:, 1] ** 2.0 + stars_pos[:, 2] ** 2.0

        # total kinetic energy
        sub_K = 0.5 * np.sum(stars_mass * stars_vel**2.0)

        # specific stellar angular momentum
        stars_J = stars_mass * np.cross(stars_pos, stars_vel, axis=1)
        sub_stellarJ = np.sum(stars_J, axis=0)
        sub_stellarJ_mag = np.linalg.norm(sub_stellarJ)
        sub_stellarJ /= sub_stellarJ_mag  # to unit vector
        stars_Jz_i = np.dot(stars_J, sub_stellarJ)

        # kinetic energy in rot (exclude first star with zero radius)
        stars_R_i = np.sqrt(stars_rad_sq - np.dot(stars_pos, sub_stellarJ) ** 2.0)

        stars_mass = stars_mass.reshape(stars_mass.size)
        sub_Krot = 0.5 * np.sum((stars_Jz_i[1:] / stars_R_i[1:]) ** 2.0 / stars_mass[1:])

        # restricted to those stars with the same rotation orientation as the mean
        w = np.where((stars_Jz_i > 0.0) & (stars_R_i > 0.0))
        sub_Krot_oriented = np.nan
        if len(w[0]):
            sub_Krot_oriented = 0.5 * np.sum((stars_Jz_i[w] / stars_R_i[w]) ** 2.0 / stars_mass[w])

        # mass fraction of stars with counter-rotation
        w = np.where(stars_Jz_i < 0.0)
        mass_frac_counter = stars_mass[w].sum() / sub_stellarMass

        r0 = sub_Krot / sub_K  # \kappa_{star, rot}
        r1 = sub_Krot_oriented / sub_K  # \kappa_{star, rot oriented}
        r2 = mass_frac_counter  # M_{star,counter} / M_{star,total}
        r3 = sub_stellarJ_mag / sub_stellarMass  # j_star [kpc km/s]

        return [r0, r1, r2, r3]

    # ufunc: radial velocity
    if ptProperty == "radvel":
        gas_pos = np.squeeze(particles["Coordinates"][i0:i1, :][wValid, :])
        gas_vel = np.squeeze(particles["Velocities"][i0:i1, :][wValid, :])
        gas_weights = np.squeeze(particles["weights"][i0:i1][wValid])

        haloPos = gc["SubhaloPos"][subhaloID, :]
        haloVel = gc["SubhaloVel"][subhaloID, :]

        vrad = sP.units.particleRadialVelInKmS(gas_pos, gas_vel, haloPos, haloVel)
        if gas_weights.ndim == 0 and vrad.ndim == 1:
            gas_weights = [gas_weights]

        vrad_avg = np.average(vrad, weights=gas_weights)
        return vrad_avg

    # shape measurement via iterative ellipsoid fitting
    if ptProperty in ["shape_ellipsoid", "shape_ellipsoid_1r"]:
        scale_rad = gc["SubhaloRhalfStars"][subhaloID]

        if scale_rad == 0:
            return np.nan

        loc_val = particles["Coordinates"][i0:i1, :][wValid]
        loc_wt = particles["weights"][i0:i1][wValid]  # mass

        # positions relative to subhalo center, and normalized by stellar half mass radius
        for j in range(3):
            loc_val[:, j] -= gc["SubhaloPos"][subhaloID, j]

        sP.correctPeriodicDistVecs(loc_val)
        loc_val /= scale_rad

        if ptProperty == "shape_ellipsoid":
            ellipsoid_rin = 1.8  # rhalfstars
            ellipsoid_rout = 2.2  # rhalfstars
        if ptProperty == "shape_ellipsoid_1r":
            ellipsoid_rin = 0.8  # rhalfstars
            ellipsoid_rout = 1.2  # rhalfstars

        # fit, and save ratios of second and third axes lengths to major axis
        q, s, _, _ = ellipsoidfit(loc_val, loc_wt, scale_rad, ellipsoid_rin, ellipsoid_rout)

        return [q, s]

    # velocity dispersion: 3d, 1d (from 3d), or 1d z-direction
    if ptProperty in ["veldisp3d", "veldisp1d", "veldisp_z"]:
        loc_vel = particles["Velocities" if "Velocities" in particles else "vel_z"][i0:i1][wValid]
        loc_wt = particles["weights"][i0:i1][wValid]
        # loc_rad = np.sqrt(rr[wValid]) # could add Hubble expansion correction

        # mean velocity in each coordinate direction
        loc_vel = sP.units.particleCodeVelocityToKms(loc_vel)  # km/s

        if ptProperty == "veldisp_z":
            vel_mean = np.average(loc_vel, weights=loc_wt)
            loc_vel -= vel_mean
            loc_vel_sq = loc_vel**2
        else:
            vel_mean = np.average(loc_vel, axis=0, weights=loc_wt)

            for j in range(3):
                loc_vel[:, j] -= vel_mean[j]

            loc_vel_sq = np.sum(loc_vel**2, axis=1)

        velvar = np.sum(loc_wt * loc_vel_sq) / np.sum(loc_wt)
        veldisp = np.sqrt(velvar)

        if op == "veldisp1d":
            veldisp /= np.sqrt(3)

        return veldisp

    # ufunc: 'half radius' (enclosing 50%) of the quantity, or 80%, etc
    if op == "halfrad":
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt(rr[wValid])

        rhalf = findHalfLightRadius(loc_rad, loc_val, mags=False)
        return rhalf

    if op == "rad80":
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt(rr[wValid])

        r80 = findHalfLightRadius(loc_rad, loc_val, frac=0.8, mags=False)
        return r80

    # ufunc: '3D concentration' C = 5*log(r80/r20), e.g. Rodriguez-Gomez+2019 Eqn 16
    if op == "concentration":
        loc_val = particles[ptProperty][i0:i1][wValid]
        loc_rad = np.sqrt(rr[wValid])

        r20 = findHalfLightRadius(loc_rad, loc_val, frac=0.2, mags=False)
        r80 = findHalfLightRadius(loc_rad, loc_val, frac=0.8, mags=False)

        c = 5 * np.log10(r80 / r20)
        return c

    # distance to 256th closest particle
    if op == "dist256":
        rr_loc = np.sort(rr[wValid])
        dist = np.sqrt(np.take(rr_loc, 256, mode="clip"))

        return dist

    # 2d gridding: deposit quantity onto a grid and derive a summary statistic
    if op.startswith("grid2d_"):
        # prepare grid
        pos = np.squeeze(particles["Coordinates"][i0:i1, :][wValid, :])
        hsml = np.squeeze(particles["hsml"][i0:i1][wValid])
        mass = np.squeeze(particles[ptProperty][i0:i1][wValid])
        quant = None

        boxCen = gc["SubhaloPos"][subhaloID, :]

        # allocate return
        result = np.zeros(len(opts["isophot_levels"]), dtype="float32")
        result.fill(np.nan)

        if mass.size == 1:
            return result  # cannot grid one cell

        # run grid
        grid = sphMap(
            pos, hsml, mass, quant, opts["axes"], opts["boxSizeImg"], sP.boxSize, boxCen, opts["nPixels"], ndims=3
        )

        # post-process grid (any/all functionality of vis.common.gridBox() that we care about)
        if opts["smoothFWHM"] is not None:
            grid = gaussian_filter(grid, opts["sigma_xy"], mode="reflect", truncate=5.0)

        if " lum" in ptProperty:  # erg/s/cm^2 -> erg/s/cm^2/arcsec^2
            grid = sP.units.fluxToSurfaceBrightness(grid, opts["pxSizesCode"], arcsec2=True)

        grid = logZeroNaN(grid)  # log [erg/s/cm^2/arcsec^2]

        # derive quantity
        for j, isoval in enumerate(opts["isophot_levels"]):
            if np.isinf(isoval):
                mask = np.ones(grid.shape, dtype="bool")
            else:
                with np.errstate(invalid="ignore"):
                    mask = (grid > isoval) & (grid < isoval + 1.0)

            ww = np.where(mask)

            if op.endswith("_shape"):
                if len(ww[0]) <= 3:
                    continue  # singular matrix for mvbe

                points = np.vstack((opts["xxyy"][ww[0]], opts["xxyy"][ww[1]])).T

                # compute minimum volume bounding ellipsoid (minimum area ellipse in 2D)
                axislengths, theta, cen = mvbe(points)

                if 0:
                    # debug plot
                    import matplotlib.pyplot as plt
                    from matplotlib.patches import Ellipse

                    figsize = np.array([14, 10]) * 0.8
                    fig = plt.figure(figsize=figsize)
                    ax = fig.add_subplot(111)

                    # plot
                    plt.imshow(mask, cmap="viridis", aspect=mask.shape[0] / mask.shape[1])
                    ax.autoscale(False)

                    pxscale = opts["nPixels"][0] / opts["gridSizeCodeUnits"]
                    minoraxis_px = 2 * axislengths.min() * pxscale
                    majoraxis_px = 2 * axislengths.max() * pxscale
                    cen_px = cen * pxscale

                    e = Ellipse(cen_px, majoraxis_px, minoraxis_px, theta, lw=2.0, fill=False, color="red")
                    ax.add_artist(e)

                    ax.scatter(points[:, 1] * pxscale, points[:, 0] * pxscale, 1.0, marker="x", color="green")

                    fig.savefig("mask_%.1f.pdf" % isoval)
                    plt.close(fig)

                result[j] = axislengths.max() / axislengths.min()  # a/b > 1

            if op.endswith("_area"):
                result[j] = len(ww[0]) * opts["pxAreaCode"]  # (ckpc/h)^2

            if op.endswith("_gini"):
                # Gini coefficient (Rodriguez-Gomez+2019 Eqn 9)
                n = len(ww[0])

                if n < 2:
                    continue  # too few pixels

                Xi = np.sort(np.abs(10.0 ** grid[ww] * 1e10))  # linear, arbitrary scaling to move closer to one
                denom = np.nanmean(Xi) * n * (n - 1)
                num = np.sum((2 * np.arange(1, n + 1) - n - 1) * Xi)
                gini = num / denom
                assert gini >= 0.0 and gini <= 1.0

                result[j] = gini

            if op.endswith("_m20"):
                # M20 coefficient (Rodriguez-Gomez+2019 Sec 4.4.2)
                I = 10.0 ** grid[ww] * 1e10  # linear, arbitrary scaling to move closer to one
                I[np.isnan(I)] = 0.0

                x = opts["xxyy"][ww[0]]
                y = opts["xxyy"][ww[1]]

                # calculate centroid
                M_00 = np.sum(I)
                M_10 = np.sum(x * I)
                M_01 = np.sum(y * I)

                xc = M_10 / M_00
                yc = M_01 / M_00

                # calculate second total central moment
                M_20 = np.sum(x**2 * I)
                M_02 = np.sum(y**2 * I)

                mu_20 = M_20 - xc * M_10
                mu_02 = M_02 - yc * M_01
                second_moment_tot = mu_20 + mu_02

                if second_moment_tot <= 0:
                    continue  # negative second moment

                # calculate threshold pixel value
                sorted_vals = np.sort(I.ravel())
                lumfrac = np.cumsum(sorted_vals) / np.nansum(sorted_vals)
                thresh = sorted_vals[np.where(lumfrac >= 0.8)[0]]

                if len(thresh) == 0:
                    continue  # too few pixels

                # calculate second moment of these brightest 20% of pixels
                I_20 = I.copy()
                I_20[I < thresh[0]] = 0.0

                M_10 = np.sum(x * I_20)
                M_01 = np.sum(y * I_20)
                M_20 = np.sum(x**2 * I_20)
                M_02 = np.sum(y**2 * I_20)

                mu_20 = M_20 - xc * M_10
                mu_02 = M_02 - yc * M_01

                second_moment_20 = mu_20 + mu_02

                if second_moment_20 <= 0:
                    continue  # negative moment

                m20 = np.log10(second_moment_20 / second_moment_tot)

                result[j] = m20

        return result

    raise Exception("Unhandled op.")


def subhaloRadialReduction(
    sP,
    pSplit,
    ptType,
    ptProperty,
    op,
    rad,
    ptRestrictions=None,
    weighting=None,
    scope="subfind",
    minStellarMass=None,
    minHaloMass=None,
    cenSatSelect=None,
):
    """Compute a reduction operation of a particle/cell property for each subhalo.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      ptType (str): particle type e.g. 'gas', 'stars', 'dm', 'bhs'.
      ptProperty (str): particle/cell quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'sum', 'mean', 'max' or custom user-defined string.
      rad (float or str): if a scalar, then [physical kpc], otherwise a string label for a given
        radial restriction specification e.g. 'rvir' or '2rhalfstars' (see :py:func:`_radialRestriction`).
      ptRestrictions (dict): apply cuts to which particles/cells are included. Each key,val pair in the dict
        specifies a particle/cell field string in key, and a [min,max] pair in value, where e.g. np.inf can be
        used as a maximum to enforce a minimum threshold only.
      weighting (str): if not None, then use this additional particle/cell property as the weight.
      scope (str): Calculation is restricted to subhalo particles only if ``scope=='subfind'`` (default),
        or FoF particles if ``scope=='fof'``. If ``scope=='global'``, currently a full non-chunked snapshot load
        and brute-force distance computations to all particles for each subhalo (can change to tree method).
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    ops_basic = ["sum", "mean", "max"]
    ops_custom = [
        "ufunc",
        "halfrad",
        "rad80",
        "dist256",
        "concentration",
        "grid2d_isophot_shape",
        "grid2d_isophot_area",
        "grid2d_isophot_gini",
        "grid2d_m20",
    ]
    assert op in ops_basic + ops_custom
    assert scope in ["subfind", "fof", "global"]
    if op == "ufunc":
        assert ptProperty in userCustomFields
    assert minStellarMass is None or minHaloMass is None  # cannot have both

    # determine ptRestriction
    if ptType == "stars":
        if ptRestrictions is None:
            ptRestrictions = {}
        ptRestrictions["GFM_StellarFormationTime"] = ["gt", 0.0]  # real stars

    # config
    ptLoadType = sP.ptNum(ptType)

    desc = "Quantity [%s] enclosed within a radius of [%s] for [%s]." % (ptProperty, rad, ptType)
    if ptRestrictions is not None:
        desc += " (restriction = %s). " % ",".join(list(ptRestrictions))
    if weighting is not None:
        desc += " (weighting = %s). " % weighting
    if scope == "subfind":
        desc += " (only subhalo particles included). "
    if scope == "fof":
        desc += " (all parent FoF particles included). "
    if scope == "global":
        desc += " (all global particles included). "
    select = "All Subhalos."
    if minStellarMass is not None:
        select += " (Only with stellar mass >= %.2f)" % minStellarMass
    if minHaloMass is not None:
        select += " (Only with halo mass >= %s)" % minHaloMass
    if cenSatSelect is not None:
        select += " (Only [%s] subhalos)" % cenSatSelect

    # load group information
    gc = sP.groupCat(fieldsSubhalos=["SubhaloPos", "SubhaloLenType"])
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]
    nSubsTot = sP.numSubhalos

    if nSubsTot == 0:
        return np.nan, {}  # e.g. snapshots so early there are no subhalos

    if scope == "fof":
        # replace 'SubhaloLenType' and 'SubhaloOffsetType' by parent FoF group values (for both cen/sat)
        GroupLenType = sP.groupCat(fieldsHalos=["GroupLenType"])
        GroupOffsetType = sP.groupCatOffsetListIntoSnap()["snapOffsetsGroup"]
        SubhaloGrNr = sP.groupCat(fieldsSubhalos=["SubhaloGrNr"])

        gc["SubhaloLenType"] = GroupLenType[SubhaloGrNr, :]
        gc["SubhaloOffsetType"] = GroupOffsetType[SubhaloGrNr, :]

    # determine radial restriction for each subhalo
    radRestrictIn2D, radSqMin, radSqMax, _ = _radialRestriction(sP, nSubsTot, rad)

    if radRestrictIn2D:
        Nside = "z-axis"
        print(" Requested: radRestrictIn2D! Using hard-coded projection direction of [%s]!" % Nside)

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange, nSubsSelected = pSplitBounds(
        sP, pSplit, minStellarMass=minStellarMass, minHaloMass=minHaloMass, cenSatSelect=cenSatSelect
    )
    nSubsDo = len(subhaloIDsTodo)

    if rad not in ["2rhalfstars_fof"]:
        # skip check if e.g. some (sub)halos have no stars, and so no stellar half mass radii, in which case
        # our return here should likely also be nan
        assert np.count_nonzero(np.isnan(radSqMin[subhaloIDsTodo])) == 0, "Radial selection requires centrals only?"
        assert np.count_nonzero(np.isnan(radSqMax[subhaloIDsTodo])) == 0, "Radial selection requires centrals only?"

    if ptType not in indRange:
        # e.g. snapshots so early there are no stars, or no SMBHs
        attrs = {
            "Description": desc.encode("ascii"),
            "Selection": "NOTE: No particles of requested type at this snapshot!".encode("ascii"),
            "subhaloIDs": subhaloIDsTodo,
        }
        r = np.zeros(nSubsDo, dtype="float32")
        r.fill(np.nan)

        return r, attrs

    indRange = indRange[ptType]  # choose index range for the requested particle type

    if scope == "global":
        # all tasks, regardless of pSplit or not, do global load (at once, not chunked)
        h = sP.snapshotHeader()
        indRange = [0, h["NumPart"][sP.ptNum(ptType)] - 1]
        i0 = 0  # never changes
        i1 = indRange[1]  # never changes

    # info
    username = getuser()
    if username != "wwwrun":
        print(" " + desc)
        print(
            " Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now..."
            % (nSubsTot, nSubsSelected, nSubsDo)
        )

    # global load of all particles of [ptType] in snapshot
    fieldsLoad = []

    if rad is not None or op in ["halfrad", "rad80", "dist256", "concentration"]:
        fieldsLoad.append("pos")

    if ptRestrictions is not None:
        for restrictionField in ptRestrictions:
            fieldsLoad.append(restrictionField)

    allocSize = None

    if ptProperty == "Krot":
        fieldsLoad.append("pos")
        fieldsLoad.append("vel")
        fieldsLoad.append("mass")
        allocSize = (nSubsDo, 4)

    if ptProperty == "radvel":
        fieldsLoad.append("pos")
        fieldsLoad.append("vel")
        gc["SubhaloVel"] = sP.groupCat(fieldsSubhalos=["SubhaloVel"])
        allocSize = (nSubsDo,)

    if ptProperty in ["shape_ellipsoid", "shape_ellipsoid_1r"]:
        gc["SubhaloRhalfStars"] = sP.groupCat(fieldsSubhalos=["SubhaloHalfmassRadType"])[:, sP.ptNum("stars")]
        fieldsLoad.append("pos")
        allocSize = (nSubsDo, 2)  # q,s

    if ptProperty in ["veldisp3d", "veldisp1d"]:
        fieldsLoad.append("vel")
        allocSize = (nSubsDo,)
    if ptProperty in ["veldisp_z"]:
        fieldsLoad.append("vel_z")
        allocSize = (nSubsDo,)

    opts = None  # todo: can move to function argument
    if "grid2d" in op:
        fieldsLoad.append("pos")
        fieldsLoad.append("hsml")

        # hard-code constant grid parameters (can generalize)
        opts = {
            "isophot_levels": [-17.5, -18.0, -18.5, -19.0, -19.5, -20.0],  # erg/s/cm^2/arcsec^2
            "axes": [0, 1],  # random orientations
            "quant": None,  # distribute e.g. mass or light
            "gridExtentKpc": 100.0,
            "smoothFWHM": None,  # disabled
            "nPixels": [250, 250],
        }

        # hard-code instrumental related grid parameters (can generalize)
        if 1:
            # MUSE UDF
            opts["pxScaleKpc"] = sP.units.arcsecToAngSizeKpcAtRedshift(0.2)  # MUSE 0.2"/px
            opts["smoothFWHM"] = sP.units.arcsecToAngSizeKpcAtRedshift(0.7)  # ~MUSE UDF (arcsec, non-AO seeing)

        if op.endswith("_gini") or op.endswith("_m20"):
            # actual pixel size/counts important
            nPixels = int(np.round(opts["gridExtentKpc"] / opts["pxScaleKpc"]))
            opts["nPixels"] = [nPixels, nPixels]

        if "isophot_" not in op:
            # quantities invariant to pixel selection, or where we don't want to explore multiple levels
            opts["isophot_levels"] = [-np.inf]

        opts["gridSizeCodeUnits"] = sP.units.physicalKpcToCodeLength(opts["gridExtentKpc"])
        opts["boxSizeImg"] = opts["gridSizeCodeUnits"] * np.array([1.0, 1.0, 1.0])
        opts["pxSizesCode"] = opts["boxSizeImg"][0:1] / opts["nPixels"]
        opts["pxAreaCode"] = np.product(opts["pxSizesCode"])

        # compute pixel coordinates
        opts["xxyy"] = np.linspace(
            opts["pxScaleKpc"] / 2, opts["gridSizeCodeUnits"] - opts["pxScaleKpc"] / 2, opts["nPixels"][0]
        )

        if opts["smoothFWHM"] is not None:
            # fwhm -> 1 sigma, and physical kpc -> pixels (can differ in x,y)
            pxScaleXY = np.array(opts["boxSizeImg"])[opts["axes"]] / opts["nPixels"]
            opts["sigma_xy"] = (opts["smoothFWHM"] / 2.3548) / pxScaleXY

        allocSize = (nSubsDo, len(opts["isophot_levels"]))
        if len(opts["isophot_levels"]) == 1:
            allocSize = nSubsDo

    fieldsLoad = list(set(fieldsLoad))  # make unique

    particles = {}
    if len(fieldsLoad):
        particles = sP.snapshotSubsetP(partType=ptType, fields=fieldsLoad, sq=False, indRange=indRange)

    if op != "ufunc":
        # todo: as soon as snapshotSubset() can handle halo-centric quantities for more than one halo, we can
        # eliminate the entire specialized ufunc logic herein
        particles[ptProperty] = sP.snapshotSubsetP(partType=ptType, fields=[ptProperty], indRange=indRange)

    if "grid2d" in op:
        # ptProperty to be gridded is a luminosity? convert lum -> flux now
        if " lum" in ptProperty:  # 1e30 erg/s -> erg/s/cm^2
            particles[ptProperty] *= 1e30
            particles[ptProperty] = sP.units.luminosityToFlux(particles[ptProperty], wavelength=None)

    if "count" not in particles:
        key = list(particles.keys())[0]
        particles["count"] = particles[key].shape[0]

    # allocate, NaN indicates not computed except for mass where 0 will do
    dtype = particles[ptProperty].dtype if ptProperty in particles.keys() else "float32"  # for custom
    assert dtype in ["float32", "float64"]  # otherwise check, when does this happen?

    if allocSize is not None:
        r = np.zeros(allocSize, dtype=dtype)
    else:
        if particles[ptProperty].ndim in [0, 1]:
            r = np.zeros(nSubsDo, dtype=dtype)
        else:
            r = np.zeros((nSubsDo, particles[ptProperty].shape[1]), dtype=dtype)

    if op not in ["sum"]:
        r.fill(np.nan)  # set NaN value for subhalos with e.g. no particles for op=mean

    # load weights
    if weighting is None:
        particles["weights"] = np.zeros(particles["count"], dtype="float32")
        particles["weights"] += 1.0  # uniform
    else:
        assert op not in ["sum"]  # meaningless

        if "bandLum" in weighting:
            # prepare sps interpolator
            from ..cosmo.stellarPop import sps

            pop = sps(sP, "padova07", "chabrier", "cf00")

            # load additional fields, snapshot wide
            fieldsLoadMag = ["initialmass", "metallicity"]
            magsLoad = sP.snapshotSubset(partType=ptType, fields=fieldsLoadMag, indRange=indRange)

            # request magnitudes in this band
            band = weighting.split("-")[1]
            mags = pop.mags_code_units(
                sP,
                band,
                particles["GFM_StellarFormationTime"],
                magsLoad["GFM_Metallicity"],
                magsLoad["GFM_InitialMass"],
                retFullSize=True,
            )

            # use the (linear) luminosity in this band as the weight
            particles["weights"] = sP.units.absMagToLuminosity(mags)

        else:
            # use a particle quantity as weights (e.g. 'mass', 'volume', 'O VI mass')
            particles["weights"] = sP.snapshotSubset(partType=ptType, fields=weighting, indRange=indRange)

    assert particles["weights"].ndim == 1 and particles["weights"].size == particles["count"]

    # loop over subhalos
    printFac = 100.0 if (sP.res > 512 or scope == "global") else 10.0

    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1, int(nSubsDo / printFac)]) == 0 and i <= nSubsDo and username != "wwwrun":
            print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo))

        # slice starting/ending indices for stars local to this FoF
        if scope != "global":
            i0 = gc["SubhaloOffsetType"][subhaloID, ptLoadType] - indRange[0]
            i1 = i0 + gc["SubhaloLenType"][subhaloID, ptLoadType]

        assert i0 >= 0 and i1 <= (indRange[1] - indRange[0] + 1)

        if i1 == i0:
            continue  # zero length of this type

        # use squared radii and sq distance function
        validMask = np.ones(i1 - i0, dtype="bool")

        rr = None
        if "Coordinates" in particles:
            if not radRestrictIn2D:
                # apply in 3D
                rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], particles["Coordinates"][i0:i1, :])
            else:
                # apply in 2D projection, limited support for now, just Nside='z-axis'
                # otherwise, for any more complex projection, need to apply it here, and anyways
                # for nProj>1, this validMask selection logic becomes projection dependent, so
                # need to move it inside the range(nProj) loop, which is definitely doable
                assert Nside == "z-axis"
                p_inds = [0, 1]  # x,y
                pt_2d = gc["SubhaloPos"][subhaloID, :]
                pt_2d = [pt_2d[p_inds[0]], pt_2d[p_inds[1]]]
                vecs_2d = np.zeros((i1 - i0, 2), dtype=particles["Coordinates"].dtype)
                vecs_2d[:, 0] = particles["Coordinates"][i0:i1, p_inds[0]]
                vecs_2d[:, 1] = particles["Coordinates"][i0:i1, p_inds[1]]

                rr = sP.periodicDistsSq(pt_2d, vecs_2d)  # handles 2D

            if rad is not None:
                if radSqMax.ndim == 1:
                    # radial / circular aperture
                    validMask &= rr <= radSqMax[subhaloID]
                    validMask &= rr >= radSqMin[subhaloID]
                else:
                    # rectangular aperture in projected (x,y), e.g. slit
                    xDist = vecs_2d[:, 0] - pt_2d[0]
                    yDist = vecs_2d[:, 1] - pt_2d[1]
                    sP.correctPeriodicDistVecs(xDist)
                    sP.correctPeriodicDistVecs(yDist)

                    validMask &= xDist <= np.sqrt(radSqMax[subhaloID, 0])
                    validMask &= yDist <= np.sqrt(radSqMax[subhaloID, 1])

        # apply particle-level restrictions
        if ptRestrictions is not None:
            for restrictionField in ptRestrictions:
                inequality, val = ptRestrictions[restrictionField]

                if inequality == "gt":
                    validMask &= particles[restrictionField][i0:i1] > val
                if inequality == "lt":
                    validMask &= particles[restrictionField][i0:i1] <= val
                if inequality == "eq":
                    validMask &= particles[restrictionField][i0:i1] == val

        wValid = np.where(validMask)

        if len(wValid[0]) == 0:
            continue  # zero length of particles satisfying radial cut and restriction

        # user function reduction operations
        if op in ops_custom:
            r[i, ...] = _process_custom_func(sP, op, ptProperty, gc, subhaloID, particles, rr, i0, i1, wValid, opts)

            # ufunc processed and value stored, skip to next subhalo
            continue

        # standard reduction operation
        if particles[ptProperty].ndim == 1:
            # scalar
            loc_val = particles[ptProperty][i0:i1][wValid]
            loc_wt = particles["weights"][i0:i1][wValid]

            if op == "sum":
                r[i] = np.sum(loc_val)
            if op == "max":
                r[i] = np.max(loc_val)
            if op == "mean":
                if loc_wt.sum() == 0.0:
                    loc_wt = np.zeros(loc_val.size, dtype="float32") + 1.0  # if all zero weights
                r[i] = np.average(loc_val, weights=loc_wt)
        else:
            # vector (e.g. pos, vel, Bfield)
            for j in range(particles[ptProperty].shape[1]):
                loc_val = particles[ptProperty][i0:i1, j][wValid]
                loc_wt = particles["weights"][i0:i1][wValid]

                if op == "sum":
                    r[i, j] = np.sum(loc_val)
                if op == "max":
                    r[i, j] = np.max(loc_val)
                if op == "mean":
                    if loc_wt.sum() == 0.0:
                        loc_wt = np.zeros(loc_val.size, dtype="float32") + 1.0  # if all zero weights

                    r[i, j] = np.average(loc_val, weights=loc_wt)

    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "ptType": ptType.encode("ascii"),
        "ptProperty": ptProperty.encode("ascii"),
        "rad": str(rad).encode("ascii"),
        "weighting": str(weighting).encode("ascii"),
        "subhaloIDs": subhaloIDsTodo,
    }

    if "grid2d" in op:
        for key in ["isophot_levels", "axes", "gridExtentKpc", "pxScaleKpc", "smoothFWHM", "nPixels"]:
            attrs[key] = opts[key]

    return r, attrs


def subhaloStellarPhot(
    sP,
    pSplit,
    iso=None,
    imf=None,
    dust=None,
    Nside=1,
    rad=None,
    modelH=True,
    bands=None,
    sizes=False,
    indivStarMags=False,
    fullSubhaloSpectra=False,
    redshifted=False,
    emlines=False,
    seeing=None,
    minStellarMass=None,
    minHaloMass=None,
):
    """Compute the total band-magnitudes (or half-light radii if ``sizes==True``), per subhalo.

    Under a given assumption of an iso(chrone) model, imf model, dust model, and radial restrction.
    If using a dust model, can include multiple projection directions per subhalo.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      iso (str): isochrone library, as in :py:class:`cosmo.stellarPop.sps` initialization.
      imf (str): stellar IMF, as in :py:class:`cosmo.stellarPop.sps` initialization.
      dust (str or None): dust model, as in :py:class:`cosmo.stellarPop.sps` initialization.
      Nside (int or str or None): if None, then no 2D projections are done, should be used if e.g. if no
        viewing angle dependent dust is requested. If an integer, then a Healpix specification, which
        defines the multiple viewing angles per subhalo. If str, one of 'efr2d' (several edge-on and face-on
        viewing angles), or 'z-axis' (single projection per subhalo along z).
      rad (float or str): if a scalar, then [physical kpc], otherwise a string label for a given
        radial restriction specification e.g. 'rvir' or '2rhalfstars' (see :py:func:`_radialRestriction`).
      modelH (bool): use our model for neutral hydrogen masses (for extinction), instead of snapshot values.
      bands (list[str] or None): over-ride default list of broadbands to compute.
      sizes (bool): instead of band-magnitudes, save half-light radii.
      indivStarMags (bool): save the magnitudes for every PT4 (wind->NaN) in all subhalos.
      fullSubhaloSpectra (bool): save a full spectrum vs wavelength for every subhalo.
      redshifted (bool): all the stellar spectra/magnitudes are computed at sP.redshift and the band filters
        are then applied, resulting in apparent magnitudes. If False (default), stars are assumed to be at
        z=0, spectra are rest-frame and magnitudes are absolute (if dust is unresolved) or apparent
        (if dust is resolved).
      emlines (bool): include nebular emission lines.
      seeing (float or None): if not None, then instead of a binary inclusion/exclusion of each star particle
        based on the ``rad`` aperture, include all stars weighted by the fraction of their light which
        enters the ``rad`` aperture, assuming it is spread by atmospheric seeing into a Gaussian with a
        sigma of seeing [units of arcseconds at sP.redshift].
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d array, value for each subhalo.
      - **attrs** (dict): metadata.
    """
    from healpy.pixelfunc import nside2npix, pix2vec

    from ..cosmo.hydrogen import hydrogenMass
    from ..cosmo.stellarPop import sps

    # mutually exclusive options, at most one can be enabled
    assert sum([sizes, indivStarMags, np.clip(fullSubhaloSpectra, 0, 1)]) in [0, 1]
    assert minStellarMass is None or minHaloMass is None  # cannot have both

    # initialize a stellar population interpolator
    pop = sps(sP, iso, imf, dust, redshifted=redshifted, emlines=emlines)

    # which bands? for now, to change, just recompute from scratch
    if bands is None:
        bands = []
        bands += ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
        # bands += ['wfcam_y','wfcam_j','wfcam_h','wfcam_k'] # UKIRT IR wide
        # bands += ['wfc_acs_f606w','wfc3_ir_f125w','wfc3_ir_f140w','wfc3_ir_f160w'] # HST IR wide
        # bands += ['jwst_f070w','jwst_f090w','jwst_f115w','jwst_f150w',
        #           'jwst_f200w','jwst_f277w','jwst_f356w','jwst_f444w'] # JWST IR (NIRCAM) wide
        # if indivStarMags: bands = ['sdss_r']

    if str(bands) == "all":
        bands = pop.bands

    nBands = len(bands)

    if fullSubhaloSpectra:
        # set nBands to size of wavelength grid within SDSS/LEGA-C spectral range
        if "sdss" in rad:
            spec_min_ang = 3000.0
            spec_max_ang = 10000.0

            output_wave = pop.wave_ang  # at intrinsic stellar library model resolution / wavelength grid

            # enforced in rest-frame if redshifted is False, otherwise in observed-frame (if redshifted is True,
            # then pop.wave_ang corresponds to observed-frame wavelengths for the spectra of pop.dust_tau_model_mags)
            ww = np.where((pop.wave_ang >= spec_min_ang) & (pop.wave_ang <= spec_max_ang))[0]
            spec_min_ind = ww.min()
            spec_max_ind = ww.max() + 1
            nBands = spec_max_ind - spec_min_ind

        if "legac" in rad:
            # load lega-c dr2 wavelength grid
            with h5py.File(expanduser("~") + "/obs/LEGAC/legac_dr2_spectra_wave.hdf5", "r") as f:
                output_wave = f["wavelength"][()]

            spec_min_ang = output_wave.min()
            spec_max_ang = output_wave.max()

            spec_min_ind = 0
            spec_max_ind = output_wave.size
            nBands = output_wave.size

        # only for resolved dust models do we currently calculate full spectra of every star particle
        assert "_res" in dust

        # if fullSubhaloSpectra == 2, we include the peculiar motions of stars (e.g. veldisp)
        # in which case the rel_vel here is overwritten on a per subhalo basis below
        rel_vel_los = None

    # which projections?
    nProj = 1
    efrDirs = False

    if "_res" in dust or sizes is True:
        if isinstance(Nside, int):
            # numeric Nside -> healpix vertices as projection vectors
            nProj = nside2npix(Nside)
            projVecs = pix2vec(Nside, range(nProj), nest=True)
            projVecs = np.transpose(np.array(projVecs, dtype="float32"))  # Nproj,3
            projDesc = "2D projections."
        else:
            # string Nside -> custom projection vectors
            if Nside == "efr2d":
                projDesc = "2D: edge-on, face-on, edge-on-smallest, edge-on-random, random."
                nProj = 5

                Nside = Nside.encode("ascii")  # for hdf5 attr save
                projVecs = np.zeros((nProj, 3), dtype="float32")  # derive per subhalo
                efrDirs = True
                assert (sizes is True) or ("_res" in dust)  # only cases where efr logic exists for now
            elif Nside == "z-axis":
                projDesc = "2D: single projection along z-axis of simulation box."
                nProj = 1
                projVecs = np.array([0, 0, 1], dtype="float32").reshape(1, 3)  # [nProj,3]
            elif Nside is None:
                projDesc = "3D projection."
                pass  # no 2D radii
            else:
                assert 0  # unhandled

    # prepare catalog metadata
    desc = "Stellar light emission (total AB magnitudes) by subhalo, multiple bands."
    if sizes:
        desc = "Stellar half light radii (code units) by subhalo, multiple bands. " + projDesc
    if indivStarMags:
        desc = "Star particle individual AB magnitudes, multiple bands."
    if fullSubhaloSpectra:
        desc = "Optical spectra by subhalo, [%d] wavelength points between [%.1f Ang] and [%.1f Ang]." % (
            nBands,
            spec_min_ang,
            spec_max_ang,
        )
    if redshifted:
        desc += " Redshifted, observed-frame bands/wavelengths, apparent magnitudes/luminosities."
    else:
        # note: if '_res' in dust, then magnitudes are actually apparent!
        desc += " Unredshifted, rest-frame bands/wavelengths, absolute magnitudes/luminosities."
    if seeing is not None:
        desc += " Weighted contributions incorporating atmospheric seeing of [%.1f arcsec]." % seeing

    select = "All Subfind subhalos"
    if minStellarMass is not None:
        select += " (Only with stellar mass >= %.2f)" % minStellarMass
    if minHaloMass is not None:
        select += " (Only with halo mass >= %s)" % minHaloMass
    if indivStarMags:
        select = "All PartType4 particles in all subhalos"
    select += " (numProjectionsPer = %d) (%s)." % (nProj, Nside)

    print(" %s\n %s" % (desc, select))

    # load group information
    gc = sP.groupCat(fieldsSubhalos=["SubhaloLenType", "SubhaloHalfmassRadType", "SubhaloPos"])
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]
    nSubsTot = sP.numSubhalos

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, indRange, nSubsSelected = pSplitBounds(
        sP,
        pSplit,
        minStellarMass=minStellarMass,
        minHaloMass=minHaloMass,
        equalSubSplit=False,
        indivStarMags=indivStarMags,
    )

    nSubsDo = len(subhaloIDsTodo)
    partInds = None

    print(
        " Total # Subhalos: %d, [%d] in selection, now processing [%d] in [%d] bands and [%d] projections..."
        % (nSubsTot, nSubsSelected, nSubsDo, nBands, nProj)
    )

    # allocate
    if indivStarMags:
        # compute number of PT4 particles we will do (cover full PT4 size)
        nPt4Tot = sP.snapshotHeader()["NumPart"][sP.ptNum("stars")]
        nPt4Do = indRange["stars"][1] - indRange["stars"][0] + 1

        if pSplit is None:
            nPt4Do = nPt4Tot
        if pSplit is not None and pSplit[0] == pSplit[1] - 1:
            nPt4Do = nPt4Tot - indRange["stars"][0]

        # allocate for individual particles
        r = np.zeros((nPt4Do, nBands, nProj), dtype="float32")

        # store global (snapshot) indices of particles we process
        partInds = np.arange(indRange["stars"][0], nPt4Do + indRange["stars"][0], dtype="int64")
        assert partInds.size == nPt4Do
        print(
            " Total # PT4 particles: %d, processing [%d] now, range [%d - %d]..."
            % (nPt4Tot, nPt4Do, indRange["stars"][0], indRange["stars"][0] + nPt4Do)
        )
    else:
        # allocate one save per subhalo
        r = np.zeros((nSubsDo, nBands, nProj), dtype="float32")

    r.fill(np.nan)

    # radial restriction
    radRestrictIn2D, radSqMin, radSqMax, radRestrict_sizeCode = _radialRestriction(sP, nSubsTot, rad)
    assert radSqMin.max() == 0.0  # not handled here

    # spread light of stars into gaussians based on atmospheric seeing?
    if seeing is not None:
        assert rad is not None  # meaningless
        assert "_res" in dust  # otherwise generalize
        assert radRestrictIn2D  # only makes sense in 2d projection
        if indivStarMags or sizes:
            raise Exception("What does it mean?")

        nint = 100  # integration accuracy parameter
        seeing_pkpc = sP.units.arcsecToAngSizeKpcAtRedshift(seeing, z=sP.redshift)  # arcsec -> pkpc
        seeing_code = sP.units.physicalKpcToCodeLength(seeing_pkpc)  # pkpc -> ckpc/h

        seeing_const1 = 1.0 / (2 * np.pi * seeing_code**2)
        seeing_const2 = -1.0 / (2 * seeing_code**2)

        def _seeing_func(x, y):
            """2D Gaussian, integrand for determining overlap with collecting aperture."""
            return seeing_const1 * np.exp((x * x + y * y) * seeing_const2)

    # global load of all stars in all groups in snapshot
    starsLoad = ["initialmass", "sftime", "metallicity"]
    if "_res" in dust or rad is not None or sizes is not None:
        starsLoad += ["pos"]
    if sizes:
        starsLoad += ["mass"]
    if fullSubhaloSpectra == 2:
        starsLoad += ["vel", "masses"]  # masses is the current weight for LOS mean vel

    stars = sP.snapshotSubsetP(partType="stars", fields=starsLoad, indRange=indRange["stars"])

    printFac = 100.0 if sP.res > 512 else 10.0

    # non-resolved dust: loop over all requested bands first
    if "_res" not in dust:
        if sizes:
            gas = sP.snapshotSubsetP("gas", fields=["pos", "mass", "sfr"], indRange=indRange["gas"])

        for bandNum, band in enumerate(bands):
            print("  %02d/%02d [%s]" % (bandNum + 1, len(bands), band))

            # request magnitudes in this band for all stars (apparent if redshifted == True, otherwise absolute)
            mags = pop.mags_code_units(
                sP,
                band,
                stars["GFM_StellarFormationTime"],
                stars["GFM_Metallicity"],
                stars["GFM_InitialMass"],
                retFullSize=True,
            )

            # loop over subhalos
            for i, subhaloID in enumerate(subhaloIDsTodo):
                if i % np.max([1, int(nSubsDo / printFac)]) == 0 and i <= nSubsDo:
                    print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo), flush=True)

                # slice starting/ending indices for stars local to this subhalo
                i0 = gc["SubhaloOffsetType"][subhaloID, sP.ptNum("stars")] - indRange["stars"][0]
                i1 = i0 + gc["SubhaloLenType"][subhaloID, sP.ptNum("stars")]

                assert i0 >= 0 and i1 <= (indRange["stars"][1] - indRange["stars"][0] + 1)

                if i1 == i0:
                    continue  # zero length of this type

                # radius restriction: use squared radii and sq distance function
                validMask = np.ones(i1 - i0, dtype="bool")
                if rad is not None:
                    assert radSqMax.ndim == 1  # otherwise generalize like below for '_res'
                    rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], stars["Coordinates"][i0:i1, :])
                    validMask &= rr <= radSqMax[subhaloID]
                wValid = np.where(validMask)
                if len(wValid[0]) == 0:
                    continue  # zero length of particles satisfying radial cut and restriction

                magsLocal = mags[i0:i1][wValid]  # wind particles still here, and have NaN

                if not sizes and not indivStarMags:
                    # convert mags to luminosities, sum together
                    totalLum = np.nansum(sP.units.absMagToLuminosity(magsLocal))

                    # convert back to a magnitude in this band
                    if totalLum > 0.0:
                        r[i, bandNum, 0] = sP.units.lumToAbsMag(totalLum)
                elif indivStarMags:
                    # save raw magnitudes per particle (wind/outside subhalo entries left at NaN)
                    saveInds = np.arange(i0, i1)
                    r[saveInds[wValid], bandNum, 0] = magsLocal
                elif sizes:
                    # require at least 2 stars for size calculation
                    if len(wValid[0]) < 2:
                        continue

                    # slice starting/ending indices for -gas- local to this subhalo
                    i0g = gc["SubhaloOffsetType"][subhaloID, sP.ptNum("gas")] - indRange["gas"][0]
                    i1g = i0g + gc["SubhaloLenType"][subhaloID, sP.ptNum("gas")]

                    assert i0g >= 0 and i1g <= (indRange["gas"][1] - indRange["gas"][0] + 1)

                    # calculate projection directions for this subhalo
                    projCen = gc["SubhaloPos"][subhaloID, :]

                    if efrDirs:
                        # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                        rHalf = gc["SubhaloHalfmassRadType"][subhaloID, sP.ptNum("stars")]
                        shPos = gc["SubhaloPos"][subhaloID, :]

                        gasLocal = {
                            "Masses": gas["Masses"][i0g:i1g],
                            "Coordinates": np.squeeze(gas["Coordinates"][i0g:i1g, :]),
                            "StarFormationRate": gas["StarFormationRate"][i0g:i1g],
                            "count": (i1g - i0g),
                        }
                        starsLocal = {
                            "Masses": stars["Masses"][i0:i1],
                            "Coordinates": np.squeeze(stars["Coordinates"][i0:i1, :]),
                            "GFM_StellarFormationTime": stars["GFM_StellarFormationTime"][i0:i1],
                            "count": (i1 - i0),
                        }

                        I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
                        rots = rotationMatricesFromInertiaTensor(I)

                        rotMatrices = [
                            rots["edge-on"],
                            rots["face-on"],
                            rots["edge-on-smallest"],
                            rots["edge-on-random"],
                            rots["identity"],
                        ]
                    else:
                        # construct rotation matrices for each specified projection vector direction
                        if Nside is not None:
                            rotMatrices = []
                            for projNum in range(nProj):
                                targetVec = projVecs[projNum, :]
                                rotMatrices.append(rotationMatrixFromVec(projVecs[projNum, :], targetVec))

                    # get interpolated 2D half light radii
                    for projNum in range(nProj):
                        # rotate coordinates
                        pos_stars = np.squeeze(stars["Coordinates"][i0:i1, :][wValid, :])

                        if Nside is not None:
                            # calculate 2D radii as rr2d
                            pos_stars_rot, _ = rotateCoordinateArray(
                                sP, pos_stars, rotMatrices[projNum], projCen, shiftBack=False
                            )

                            x_2d = pos_stars_rot[:, 0]  # realize axes=[0,1]
                            y_2d = pos_stars_rot[:, 1]  # realize axes=[0,1]
                            rr2d = np.sqrt(x_2d * x_2d + y_2d * y_2d)

                            r[i, bandNum, projNum] = findHalfLightRadius(rr2d, magsLocal)
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], pos_stars)
                            rr = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i, bandNum, projNum] = findHalfLightRadius(rr, magsLocal)

    # or, resolved dust: loop over all subhalos first
    if "_res" in dust:
        # prep: resolved dust attenuation uses simulated gas distribution in each subhalo
        loadFields = ["pos", "metal", "mass"]
        if sP.snapHasField("gas", "NeutralHydrogenAbundance"):
            loadFields.append("NeutralHydrogenAbundance")

        gas = sP.snapshotSubsetP("gas", fields=loadFields, indRange=indRange["gas"])

        if sP.snapHasField("gas", "GFM_Metals"):
            gas["metals_H"] = sP.snapshotSubsetP("gas", "metals_H", indRange=indRange["gas"])  # H only

        # prep: override 'Masses' with neutral hydrogen mass (model or snapshot value), free some memory
        if modelH:
            gas["Density"] = sP.snapshotSubsetP("gas", "dens", indRange=indRange["gas"])
            gas["Masses"] = hydrogenMass(gas, sP, totalNeutral=True)
            gas["Density"] = None
        else:
            gas["Masses"] = hydrogenMass(gas, sP, totalNeutralSnap=True)

        gas["metals_H"] = None
        gas["NeutralHydrogenAbundance"] = None
        gas["Cellsize"] = sP.snapshotSubsetP("gas", "cellsize", indRange=indRange["gas"])

        # prep: unit conversions on stars (age,mass,metallicity)
        stars["GFM_StellarFormationTime"] = sP.units.scalefacToAgeLogGyr(stars["GFM_StellarFormationTime"])
        stars["GFM_InitialMass"] = sP.units.codeMassToMsun(stars["GFM_InitialMass"])

        stars["GFM_Metallicity"] = logZeroMin(stars["GFM_Metallicity"])
        stars["GFM_Metallicity"][np.where(stars["GFM_Metallicity"] < -20.0)] = -20.0

        if sizes:
            gas["StarFormationRate"] = sP.snapshotSubsetP("gas", fields=["sfr"], indRange=indRange["gas"])

        # outer loop over all subhalos
        if not fullSubhaloSpectra:
            print(" Bands: [%s]." % ", ".join(bands))

        for i, subhaloID in enumerate(subhaloIDsTodo):
            # print('[%d] subhalo = %d' % (i,subhaloID))
            if i % np.max([1, int(nSubsDo / printFac)]) == 0 and i <= nSubsDo:
                print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo), flush=True)

            # slice starting/ending indices for stars local to this subhalo
            i0 = gc["SubhaloOffsetType"][subhaloID, sP.ptNum("stars")] - indRange["stars"][0]
            i1 = i0 + gc["SubhaloLenType"][subhaloID, sP.ptNum("stars")]

            assert i0 >= 0 and i1 <= (indRange["stars"][1] - indRange["stars"][0] + 1)

            if i1 == i0:
                continue  # zero length of this type

            # radius restriction: use squared radii and sq distance function
            validMask = np.ones(i1 - i0, dtype="bool")

            if rad is not None:
                if not radRestrictIn2D:
                    # apply in 3D
                    assert radSqMax.ndim == 1

                    rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], stars["Coordinates"][i0:i1, :])
                    validMask &= rr <= radSqMax[subhaloID]
                else:
                    # apply in 2D projection, limited support for now, just Nside='z-axis'
                    # otherwise, for any more complex projection, need to apply it here, and anyways
                    # for nProj>1, this validMask selection logic becomes projection dependent, so
                    # need to move it inside the range(nProj) loop, which is definitely doable
                    assert Nside == "z-axis" and nProj == 1 and np.array_equal(projVecs, [[0, 0, 1]])
                    p_inds = [0, 1]  # x,y
                    pt_2d = gc["SubhaloPos"][subhaloID, :]
                    pt_2d = [pt_2d[p_inds[0]], pt_2d[p_inds[1]]]
                    vecs_2d = np.zeros((i1 - i0, 2), dtype=stars["Coordinates"].dtype)
                    vecs_2d[:, 0] = stars["Coordinates"][i0:i1, p_inds[0]]
                    vecs_2d[:, 1] = stars["Coordinates"][i0:i1, p_inds[1]]

                    # if doing individual weights based on seeing-spread overlap with aperture,
                    # truncate contributions to stars at distances >= 5 sigma
                    sigmaPad = 0.0 if seeing is None else 5.0 * seeing_code

                    if radSqMax.ndim == 1:
                        # radial / circular aperture
                        rr = sP.periodicDistsSq(pt_2d, vecs_2d)  # handles 2D
                        rr = np.sqrt(rr)

                        validMask &= rr <= (np.sqrt(radSqMax[subhaloID]) + sigmaPad)
                    else:
                        # rectangular aperture in projected (x,y), e.g. slit
                        xDist = vecs_2d[:, 0] - pt_2d[0]
                        yDist = vecs_2d[:, 1] - pt_2d[1]
                        sP.correctPeriodicDistVecs(xDist)
                        sP.correctPeriodicDistVecs(yDist)

                        validMask &= (xDist <= (np.sqrt(radSqMax[subhaloID, 0]) + sigmaPad)) & (
                            yDist <= (np.sqrt(radSqMax[subhaloID, 1]) + sigmaPad)
                        )

            validMask &= np.isfinite(stars["GFM_StellarFormationTime"][i0:i1])  # remove wind

            wValid = np.where(validMask)[0]

            if len(wValid) == 0:
                continue  # zero length of particles satisfying radial cut and real stars restriction

            if len(wValid) < 2 and sizes:
                continue  # require at least 2 stars for size calculation

            ages_logGyr = stars["GFM_StellarFormationTime"][i0:i1][wValid]
            metals_log = stars["GFM_Metallicity"][i0:i1][wValid]
            masses_msun = stars["GFM_InitialMass"][i0:i1][wValid]
            pos_stars = stars["Coordinates"][i0:i1, :][wValid, :]

            assert ages_logGyr.shape == metals_log.shape == masses_msun.shape
            assert pos_stars.shape[0] == ages_logGyr.size and pos_stars.shape[1] == 3

            if seeing is not None:
                # derive seeing-overlap of aperture based weights
                assert radSqMax.ndim == 2  # otherwise generalize for circular integrals as well
                seeing_weights = np.zeros(ages_logGyr.size, dtype="float32")

                # re-use previous distance computation
                for j in range(seeing_weights.size):
                    # collecting aperture is centered at (0,0) i.e. at the subhalo center
                    # shift gaussian representing each star's seeing-distributed light to the origin
                    x_min = -xDist[j] - radRestrict_sizeCode[0] * 0.5
                    x_max = -xDist[j] + radRestrict_sizeCode[0] * 0.5
                    y_min = -yDist[j] - radRestrict_sizeCode[1] * 0.5
                    y_max = -yDist[j] + radRestrict_sizeCode[1] * 0.5

                    # by hand grid sampling of 2D gaussian within the aperture area
                    # (much faster than scipy.integrate.dblquad)
                    seeing_x, seeing_y = np.meshgrid(
                        np.linspace(x_min, x_max, nint + 1), np.linspace(y_min, y_max, nint + 1)
                    )
                    wt = np.sum(_seeing_func(seeing_x, seeing_y)) * (x_max - x_min) / nint * (y_max - y_min) / nint
                    seeing_weights[j] = wt

                assert seeing_weights.min() >= 0.0 and seeing_weights.max() <= 1.0

                # enforce weights by modulating masses of the populations
                masses_msun *= seeing_weights

            if fullSubhaloSpectra == 2:
                # derive mean stellar LOS velocity of selected stars, and LOS peculiar velocities of each
                # limited support for now, just Nside='z-axis', otherwise for any more complex projection,
                # need to apply it here, and anyways for nProj>1, this vel_stars calculation becomes
                # projection dependent, so need to move it inside the range(nProj) loop, as above
                assert Nside == "z-axis" and nProj == 1 and np.array_equal(projVecs, [[0, 0, 1]])
                p_ind = 2  # z

                vel_stars = stars["Velocities"][i0:i1, :][wValid, :]
                masses_stars = stars["Masses"][i0:i1][wValid]

                vel_stars = sP.units.particleCodeVelocityToKms(vel_stars)

                # mass weighted, this could be light weighted... anyways a change to this represents a
                # constant wavelength shift, e.g. is fit out as a residual redshift
                mean_vel_los = np.average(vel_stars[:, p_ind], weights=masses_stars)

                rel_vel_los = vel_stars[:, p_ind] - mean_vel_los

            # slice starting/ending indices for -gas- local to this subhalo
            i0g = gc["SubhaloOffsetType"][subhaloID, sP.ptNum("gas")] - indRange["gas"][0]
            i1g = i0g + gc["SubhaloLenType"][subhaloID, sP.ptNum("gas")]

            assert i0g >= 0 and i1g <= (indRange["gas"][1] - indRange["gas"][0] + 1)

            # calculate projection directions for this subhalo
            projCen = gc["SubhaloPos"][subhaloID, :]

            if efrDirs:
                # construct rotation matrices for each of 'edge-on', 'face-on', and 'random' (z-axis)
                rHalf = gc["SubhaloHalfmassRadType"][subhaloID, sP.ptNum("stars")]
                shPos = gc["SubhaloPos"][subhaloID, :]

                gasLocal = {
                    "Masses": gas["Masses"][i0g:i1g],
                    "Coordinates": np.squeeze(gas["Coordinates"][i0g:i1g, :]),
                    "StarFormationRate": gas["StarFormationRate"][i0g:i1g],
                }
                starsLocal = {
                    "Masses": stars["Masses"][i0:i1],
                    "Coordinates": np.squeeze(stars["Coordinates"][i0:i1, :]),
                    "GFM_StellarFormationTime": stars["GFM_StellarFormationTime"][i0:i1],
                }

                I = momentOfInertiaTensor(sP, gas=gasLocal, stars=starsLocal, rHalf=rHalf, shPos=shPos)
                rots = rotationMatricesFromInertiaTensor(I)

                rotMatrices = [
                    rots["edge-on"],
                    rots["face-on"],
                    rots["edge-on-smallest"],
                    rots["edge-on-random"],
                    rots["identity"],
                ]
                rotMatrices.extend(rotMatrices)  # append to itself, now has (5 2d + 5 3d) = 10 elements
            else:
                # construct rotation matrices for each specified projection vector direction
                rotMatrices = []
                for projNum in range(projVecs.shape[0]):
                    targetVec = projVecs[projNum, :]
                    rotMatrices.append(rotationMatrixFromVec(projVecs[projNum, :], targetVec))

            # loop over all different viewing directions
            for projNum in range(nProj):
                # at least 2 gas cells exist in subhalo?
                if i1g > i0g + 1:
                    # subsets
                    pos = gas["Coordinates"][i0g:i1g, :]
                    hsml = 2.5 * gas["Cellsize"][i0g:i1g]
                    mass_nh = gas["Masses"][i0g:i1g]
                    quant_z = gas["GFM_Metallicity"][i0g:i1g]

                    # compute line of sight integrated quantities (choose appropriate projection)
                    N_H, Z_g = pop.resolved_dust_mapping(
                        pos, hsml, mass_nh, quant_z, pos_stars, projCen, rotMatrix=rotMatrices[projNum]
                    )
                else:
                    # set columns to zero
                    N_H = np.zeros(len(wValid), dtype="float32")
                    Z_g = np.zeros(len(wValid), dtype="float32")

                if sizes:
                    # compute attenuated stellar luminosity for each star particle in each band
                    magsLocal = pop.dust_tau_model_mags(
                        bands, N_H, Z_g, ages_logGyr, metals_log, masses_msun, ret_indiv=True
                    )

                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        if Nside is not None:
                            # rotate coordinates
                            pos_stars_rot, _ = rotateCoordinateArray(
                                sP, pos_stars, rotMatrices[projNum], projCen, shiftBack=False
                            )

                            # calculate 2D radii as rr2d
                            x_2d = pos_stars_rot[:, 0]  # realize axes=[0,1]
                            y_2d = pos_stars_rot[:, 1]  # realize axes=[0,1]
                            rr2d = np.sqrt(x_2d * x_2d + y_2d * y_2d)

                            # get interpolated 2D half light radii
                            r[i, bandNum, projNum] = findHalfLightRadius(rr2d, magsLocal[band])
                        else:
                            # calculate radial distance of each star particle if not yet already
                            if rad is None:
                                rr = sP.periodicDistsSq(projCen, stars["Coordinates"][i0:i1, :])
                            rrLocal = np.sqrt(rr[wValid])

                            # get interpolated 3D half light radius
                            r[i, bandNum, projNum] = findHalfLightRadius(rrLocal, magsLocal[band])

                elif indivStarMags:
                    # compute attenuated stellar luminosity for each star particle in each band
                    magsLocal = pop.dust_tau_model_mags(
                        bands, N_H, Z_g, ages_logGyr, metals_log, masses_msun, ret_indiv=True
                    )

                    saveInds = np.arange(i0, i1)
                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        # save raw magnitudes per particle (wind/outside subhalos left at NaN)
                        r[saveInds[wValid], bandNum, projNum] = magsLocal[band]

                elif fullSubhaloSpectra:
                    # request stacked spectrum of all stars, optionally handle doppler velocity shifting
                    spectrum = pop.dust_tau_model_mags(
                        bands,
                        N_H,
                        Z_g,
                        ages_logGyr,
                        metals_log,
                        masses_msun,
                        ret_full_spectrum=True,
                        output_wave=output_wave,
                        rel_vel=rel_vel_los,
                    )

                    # save spectrum within valid wavelength range
                    r[i, :, projNum] = spectrum[spec_min_ind:spec_max_ind]
                else:
                    # compute total attenuated stellar luminosity in each band
                    # (apparent even if redshifted == False, which is not consistent with the unresolved case)
                    magsLocal = pop.dust_tau_model_mags(bands, N_H, Z_g, ages_logGyr, metals_log, masses_msun)

                    # loop over each requested band within this projection
                    for bandNum, band in enumerate(bands):
                        r[i, bandNum, projNum] = magsLocal[band]

    # prepare save
    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "dust": dust.encode("ascii"),
        "subhaloIDs": subhaloIDsTodo,
    }

    if partInds is not None:
        attrs["partInds"] = partInds

    if fullSubhaloSpectra:
        # save wavelength grid and details of redshifting
        attrs["wavelength"] = output_wave[spec_min_ind:spec_max_ind]  # rest-frame
        # if redshifted:
        #    attrs['wavelength'] *= (1 + sP.redshift) # save in observed-frame
        attrs["spectraLumDistMpc"] = sP.units.redshiftToLumDist(sP.redshift)
        if "sdss" in rad:
            attrs["spectraUnits"] = "10^-17 erg/cm^2/s/Ang".encode("ascii")
            # attrs['spectraFiberDiameterCode'] = fiber_diameter
        if "legac" in rad:
            r *= 1e2  # 1e-17 to 1e-19 unit prefix (just convention)
            attrs["spectraUnits"] = "10^-19 erg/cm^2/s/Ang".encode("ascii")
            attrs["slitSizeCode"] = [radSqMax[0, 0], radSqMax[0, 1]]
    else:
        attrs["bands"] = [b.encode("ascii") for b in bands]

    if "_res" in dust:
        # save projection details
        attrs["nProj"] = nProj
        attrs["Nside"] = Nside
        attrs["projVecs"] = projVecs
        attrs["modelH"] = modelH

    # remove nProj and/or nBands dimensions if unity (never remove nSubsDo dimension)
    if r.shape[2] == 1:
        r = np.squeeze(r, axis=(2))
    if r.shape[1] == 1:
        r = np.squeeze(r, axis=(1))

    return r, attrs


def subhaloCatNeighborQuant(
    sP,
    pSplit,
    quant,
    op,
    rad=None,
    proj2D=None,
    subRestrictions=None,
    subRestrictionsRel=None,
    minStellarMass=None,
    minHaloMass=None,
    cenSatSelect=None,
):
    """For every subhalo, search for and compute a reduction operation over spatially nearby neighbors.

    The search radius is globally constant and/or varies per subhalo (using tree search).
    Alternatively, perform an adaptive search such that we find at least N>=1
    neighbor, and similarly compute a reduction operation over their properties.

    Args:
      sP (:py:class:`~util.simParams`): simulation instance.
      pSplit (list[int]): standard parallelization 2-tuple of [cur_job_number, num_jobs_total].
      quant (str): subhalo quantity to apply reduction operation to.
      op (str): reduction operation to apply. 'min', 'max', 'mean', 'median', and 'sum' compute over the requested
        quant for all nearby subhalos within the search, excluding this subhalo. 'closest_rad' returns
        the distance of the closest neighbor satisfying the requested restrictions. 'd[3,5,10]_rad' returns
        the distance to the 3rd, 5th, or 10th nearest neighbor, respectively. 'closest_quant'
        returns the quant of this closest neighbor. 'count' returns the number of identified neighbors.
      rad (str, float, or list): physical kpc if float, or a string as recognized by :py:func:`_radialRestriction`,
        or if a list/tuple, then we compute radial profiles for each subhalo instead of a single value,
        in which case the tuple provides the profile parameters {radMin,radMax,radNumBins,radBinsLog,radRvirUnits} as
        in subhaloRadialProfile().
      proj2D (list or None): if not None, do 3D profiles, otherwise 2-tuple specifying (i) integer coordinate axis in
        [0,1,2] to project along or 'face-on' or 'edge-on', and (ii) depth in code units (None for full box).
      subRestrictions (list): apply cuts to which subhalos are searched over. Each item in the list is a
        3-tuple consisting of (field name, min value, max value), where e.g. np.inf can be used as a
        maximum to enforce a minimum threshold only.
        This is the only option which modifies the search target sample, as opposite to the search origin sample.
      subRestrictionsRel (list): as above, but every field is understood to be relative to the current
        subhalo value, which is the normalization, e.g. ('mstar2','gt',1.0) requires that neighbors have a
        strictly larger value, while ('mstar2','gt',0.5) requires neighbors have a value half as large or smaller.
      minStellarMass (str or float): minimum stellar mass of subhalo to compute in log msun (optional).
      minHaloMass (str or float): minimum halo mass to compute, in log msun (optional).
      cenSatSelect (str): exclusively process 'cen', 'sat', or 'all'.

    Returns:
      a 2-tuple composed of

      - **result** (:py:class:`~numpy.ndarray`): 1d or 2d array, containing result(s) for each processed subhalo.
      - **attrs** (dict): metadata.
    """
    assert op in [
        "min",
        "max",
        "mean",
        "median",
        "sum",
        "closest_rad",
        "d3_rad",
        "d5_rad",
        "d10_rad",
        "closest_quant",
        "count",
    ]
    if op == "closest_quant":
        assert quant is not None
    if op in ["closest_rad", "count"]:
        assert quant is None

    nSubsTot = sP.numSubhalos

    if nSubsTot == 0:
        return np.nan, {}  # e.g. snapshots so early there are no subhalos

    # radial profiles?
    profiles = False
    if isinstance(rad, (list, tuple)):
        assert len(rad) == 5
        radMin, radMax, radNumBins, radBinsLog, radRvirUnits = rad
        assert not radRvirUnits  # otherwise generalize
        assert radBinsLog  # otherwise generalize
        profiles = True

        # determine profile bins (radMin, radMax in log physical kpc)
        radMin_code_log = np.log10(sP.units.physicalKpcToCodeLength(10.0**radMin))
        radMax_code_log = np.log10(sP.units.physicalKpcToCodeLength(10.0**radMax))

        rad_bin_edges = np.linspace(
            radMin_code_log, radMax_code_log, radNumBins + 1
        )  # bin edges, including inner and outer boundary

        rbins_sq = (10.0**rad_bin_edges) ** 2  # we work in squared distances for speed
        rad_bins_code = 0.5 * (rad_bin_edges[1:] + rad_bin_edges[:-1])  # bin centers [log]
        rad_bins_pkpc = sP.units.codeLengthToKpc(10.0**rad_bins_code)

        radMaxCode = 10.0**radMax_code_log
        radSqMax = radMaxCode**2

        # bin (spherical shells in 3D, circular annuli in 2D) volumes/areas [code units]
        r_outer = 10.0 ** rad_bin_edges[1:]
        r_inner = 10.0 ** rad_bin_edges[:-1]

        bin_volumes_code = 4.0 / 3.0 * np.pi * (r_outer**3.0 - r_inner**3.0)
        bin_areas_code = np.pi * (r_outer**2.0 - r_inner**2.0)  # 2D annuli e.g. if proj2D is not None
    else:
        # determine radial restriction for each subhalo
        radRestrictIn2D, radSqMin, radSqMax, _ = _radialRestriction(sP, nSubsTot, rad)

        assert not radRestrictIn2D  # generalize below, currently everything in 3D

    maxSearchRad = np.sqrt(radSqMax)  # code units

    if proj2D is not None:
        assert profiles  # otherwise generalize
        axes2D = {0: [1, 2], 1: [2, 0], 2: [0, 1]}[proj2D[0]]

        if proj2D[1] is None:
            print("NOTE: override maxSearchRad = %g with full box depth for 2D projection." % maxSearchRad)
            maxSearchRad = sP.boxSize
        else:
            # max depth at the max projected distance is the radial search requirement
            proj2DHalfDepth = proj2D[1] / 2.0  # code units
            maxSearchRadReq = np.sqrt(proj2DHalfDepth**2 + rbins_sq[-1])
            if maxSearchRadReq > maxSearchRad:
                print("NOTE: override maxSearchRad = %g with %g for 2D projection." % (maxSearchRad, maxSearchRadReq))
                maxSearchRad = maxSearchRadReq

    # task parallelism (pSplit): determine subhalo and particle index range coverage of this task
    subhaloIDsTodo, _, nSubsSelected = pSplitBounds(
        sP, pSplit, minStellarMass=minStellarMass, minHaloMass=minHaloMass, cenSatSelect=cenSatSelect
    )
    nSubsDo = len(subhaloIDsTodo)

    # info
    desc = "[%s] of quantity [%s] enclosed within a radius of [%s]." % (op, quant, rad)
    if subRestrictions is not None:
        for rField, rFieldMin, rFieldMax in subRestrictions:
            desc += " (%s %s %s)" % (rField, rFieldMin, rFieldMax)
    if subRestrictionsRel is not None:
        for rField, rFieldMin, rFieldMax in subRestrictionsRel:
            desc += " (rel: %s %s %s)" % (rField, rFieldMin, rFieldMax)

    select = "All Subhalos."
    if minStellarMass is not None:
        select += " (Only with stellar mass >= %.2f)" % minStellarMass
    if minHaloMass is not None:
        select += " (Only with halo mass >= %s)" % minHaloMass
    if cenSatSelect is not None:
        select += " (Only [%s] subhalos)" % cenSatSelect

    username = getuser()
    if username != "wwwrun":
        print(" " + desc)
        print(
            " Total # Subhalos: %d, [%d] in selection, processing [%d] subhalos now..."
            % (nSubsTot, nSubsSelected, nSubsDo)
        )

    # decide fields, and load all subhalos in snapshot
    fieldsLoad = ["SubhaloPos", "id"]

    if quant is not None:
        fieldsLoad.append(quant)

    if subRestrictions is not None:
        for rField, _, _ in subRestrictions:
            fieldsLoad.append(rField)

    if subRestrictionsRel is not None:
        for rField, _, _ in subRestrictionsRel:
            fieldsLoad.append(rField)

    fieldsLoad = list(set(fieldsLoad))  # make unique

    gc = sP.subhalos(fieldsLoad)

    # start all valid mask for search targets
    validMask = np.ones(nSubsTot, dtype="bool")

    # if we will apply (locally variable) restrictions
    if subRestrictionsRel is not None:
        # then the quantities must be non-nan for the subhalos we are processing
        # (efficiency improvement only)
        mask = np.ones(nSubsDo, dtype="bool")

        for rField, _, _ in subRestrictionsRel:
            # mark invalid subhalos
            mask &= np.isfinite(gc[rField][subhaloIDsTodo])

        wTodoValid = np.where(mask)

        subhaloIDsTodo = subhaloIDsTodo[wTodoValid]
        nSubsDo = len(subhaloIDsTodo)

        print(
            " Note: to make relative cuts on [%s] leaves [%d] subhalos to be processed."
            % (", ".join([rField for rField, _, _ in subRestrictionsRel]), nSubsDo)
        )

        # similarly, we can apply a global pre-filter to the search targets, based on the
        # absolute min/max values of the subhalos to be processed
        for rField, rMin, rMax in subRestrictionsRel:
            global_min = gc[rField][subhaloIDsTodo].min() * rMin
            global_max = gc[rField][subhaloIDsTodo].max() * rMax

            ww = np.where((gc[rField] < global_min) | (gc[rField] > global_max))
            validMask[ww] = 0

        print(
            " Note: most conservative application of relative cuts leaves [%d] subhalos to search over."
            % np.count_nonzero(validMask)
        )

    # apply (globally constant) restriction to those subhalos included in searches?
    if subRestrictions is not None:
        for rField, rFieldMin, rFieldMax in subRestrictions:
            with np.errstate(invalid="ignore"):
                validMask &= (gc[rField] > rFieldMin) & (gc[rField] < rFieldMax)

    wValid = np.where(validMask)

    print(" After any subRestrictions, searching over [%d] of [%d] subhalos." % (len(wValid[0]), nSubsTot))

    # take subset
    gc_search = {}

    for key in fieldsLoad:
        gc_search[key] = gc[key][wValid]

    # create inverse mapping (subhaloID -> gc_search index)
    gc_search_index = np.zeros(nSubsTot, dtype="int32") - 1
    gc_search_index[gc_search["id"]] = np.arange(gc_search["id"].size)

    # initial guess (iterative)
    if op in ["d3_rad", "d5_rad", "d10_rad"]:
        target_ngb_num = int(op.replace("_rad", "")[1:])
        prevMaxSearchRad = 1000.0

    # allocate, NaN indicates not computed
    dtype = gc[quant].dtype if quant in gc.keys() else "float32"  # for custom

    shape = [nSubsDo]
    if quant is not None and gc[quant].ndim > 1:
        shape.append(gc[quant].shape[1])

    if profiles:
        shape.append(radNumBins)

    r = np.zeros(shape, dtype=dtype)
    r.fill(np.nan)  # set NaN value for un-processed subhalos

    # build tree
    tree = buildFullTree(gc_search["SubhaloPos"], boxSizeSim=sP.boxSize, treePrec="float64", verbose=True)

    # define all valid mask
    loc_search_mask = np.ones(gc_search["SubhaloPos"].shape[0], dtype="bool")

    # loop over subhalos
    for i, subhaloID in enumerate(subhaloIDsTodo):
        if i % np.max([1, int(nSubsDo / 100.0)]) == 0 and i <= nSubsDo and username != "wwwrun":
            print("   %4.1f%%" % (float(i + 1) * 100.0 / nSubsDo), flush=True)

        loc_search_pos = gc_search["SubhaloPos"]

        if quant is not None:
            loc_search_quant = gc_search[quant]

        # apply (locally relative) restriction to those subhalos included in searches?
        if subRestrictionsRel is not None:
            # reset mask for each subhalo
            loc_search_mask.fill(1)

            # apply each requested restriction
            for rField, rFieldMin, rFieldMax in subRestrictionsRel:
                if rField == "SubhaloOrigHaloID":
                    # TNG-Cluster: restrict to subhalos from the same original zoom sim
                    assert rFieldMin == rFieldMax == 1  # otherwise not clear
                    loc_search_mask &= gc_search[rField] == gc[rField][subhaloID]
                else:
                    # can contain nan (e.g. mstar_30pkpc_log), in which case we say this fails the restriction
                    with np.errstate(invalid="ignore"):
                        # compute the relative value we apply the restriction on
                        relative_val = gc_search[rField] / gc[rField][subhaloID]

                        loc_search_mask &= (relative_val > rFieldMin) & (relative_val < rFieldMax)

            if np.count_nonzero(loc_search_mask) == 0:
                continue  # no subhalos satisfy this relative restriction

        # closest radius search?
        if op in ["closest_rad", "closest_quant"]:
            posSearch = gc["SubhaloPos"][subhaloID, :].reshape((1, 3))

            dist, index = calcHsml(
                loc_search_pos, sP.boxSize, posSearch=posSearch, posMask=loc_search_mask, nearest=True, tree=tree
            )

            if 0:
                # debug verify
                wValid = np.where(loc_search_mask)[0]
                dists = sP.periodicDists(gc["SubhaloPos"][subhaloID, :], loc_search_pos[wValid])
                dists[dists == 0] = np.inf
                index2 = np.argmin(dists)
                dist2 = dists[index2]
                assert index == wValid[index2] and np.abs(dist - dist2) < 1e-3

            if op == "closest_rad":
                r[i] = dist[0]
            if op == "closest_quant":
                r[i] = loc_search_quant[index[0]]

            continue

        if op in ["d3_rad", "d5_rad", "d10_rad"]:
            # distance to the 3rd, 5th, 10th closest neighbor
            loc_inds = []

            if np.count_nonzero(loc_search_mask) < target_ngb_num + 1:
                continue  # not enough global satisfying subhalos to find locals

            iter_num = 0
            while len(loc_inds) < target_ngb_num + 1:
                # iterative search
                loc_inds = calcParticleIndices(
                    loc_search_pos,
                    gc["SubhaloPos"][subhaloID, :],
                    prevMaxSearchRad,
                    boxSizeSim=sP.boxSize,
                    posMask=loc_search_mask,
                    tree=tree,
                )

                # if size was too small, increase
                if loc_inds is None:
                    loc_inds = []

                if len(loc_inds) < target_ngb_num + 1:
                    prevMaxSearchRad *= 1.25

                iter_num += 1
                if iter_num > 100:
                    assert 0  # can continue, but this is catastropic

            if 0:
                # debug verify
                wValid = np.where(loc_search_mask)[0]
                loc_dists = sP.periodicDists(gc["SubhaloPos"][subhaloID, :], loc_search_pos[wValid])
                loc_inds2 = np.where(loc_dists <= prevMaxSearchRad)[0]
                assert np.array_equal(np.sort(loc_inds), np.sort(wValid[loc_inds2]))

            # if size was excessive, reduce for next time
            if len(loc_inds) > target_ngb_num * 2:
                prevMaxSearchRad /= 1.25

            dists = sP.periodicDists(gc["SubhaloPos"][subhaloID], loc_search_pos[loc_inds])
            dists = np.sort(dists)

            r[i] = dists[target_ngb_num]  # includes r=0 for ourself

            continue

        # standard reductions: tree search within given search radius
        loc_inds = calcParticleIndices(
            loc_search_pos,
            gc["SubhaloPos"][subhaloID, :],
            maxSearchRad,
            boxSizeSim=sP.boxSize,
            posMask=loc_search_mask,
            tree=tree,
        )

        if loc_inds is None:
            # no neighbors within radius
            if op == "count":
                r[i] = 0

            continue

        if 0:
            # debug verify
            wValid = np.where(loc_search_mask)[0]
            loc_dists = sP.periodicDists(gc["SubhaloPos"][subhaloID, :], loc_search_pos[wValid])
            loc_inds2 = np.where(loc_dists <= maxSearchRad)[0]
            assert np.array_equal(np.sort(loc_inds), np.sort(wValid[loc_inds2]))

        if op == "count" and not profiles:
            r[i] = len(loc_inds) - 1  # do not count self

            continue

        # do not include this subhalo in any statistic
        if quant is not None:
            loc_vals = loc_search_quant.copy()

            if gc_search_index[subhaloID] >= 0:
                loc_vals[gc_search_index[subhaloID]] = np.nan

            # take subset corresponding to identified neighbors
            loc_vals = loc_vals[loc_inds]

            if np.count_nonzero(np.isfinite(loc_vals)) == 0:
                continue
        else:
            assert op == "count"  # otherwise what are we doing?
            loc_vals = np.zeros(len(loc_inds), dtype="float32")  # dummy for profiles

        # store result
        if profiles:
            # radial profile
            if proj2D is not None:
                # create 2d versions of the position arrays
                sub_pos_2d = gc["SubhaloPos"][subhaloID][axes2D]
                loc_search_pos_2d = loc_search_pos[loc_inds][:, axes2D]

                # filter projection depth
                proj_dist = loc_search_pos[loc_inds][:, proj2D[0]] - gc["SubhaloPos"][subhaloID][proj2D[0]]
                sP.correctPeriodicDistVecs(proj_dist)
                valid_inds = np.where(np.abs(proj_dist) <= proj2DHalfDepth)

                # distances in 2d projection
                dists = sP.periodicDistsSq(sub_pos_2d, loc_search_pos_2d[valid_inds])
                loc_vals = loc_vals[valid_inds]
            else:
                # distances and profile in 3d
                dists = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID], loc_search_pos[loc_inds])

            # note: includes self (at r=0)
            result, _, _ = binned_statistic_weighted(dists, loc_vals, statistic=op, bins=rbins_sq)  # , weights=loc_wt)
            r[i] = result

        else:
            # single scalar reduction
            if op == "sum":
                r[i] = np.nansum(loc_vals)
            if op == "max":
                r[i] = np.nanmax(loc_vals)
            if op == "min":
                r[i] = np.nanmin(loc_vals)
            if op == "mean":
                r[i] = np.nanmean(loc_vals)
            if op == "median":
                r[i] = np.nanmedian(loc_vals)

    attrs = {
        "Description": desc.encode("ascii"),
        "Selection": select.encode("ascii"),
        "rad": str(rad).encode("ascii"),
        "subhaloIDs": subhaloIDsTodo,
    }

    if profiles:
        attrs["rad_bin_edges"] = rad_bin_edges
        attrs["rad_bins_code"] = rad_bins_code
        attrs["rad_bins_pkpc"] = rad_bins_pkpc
        attrs["bin_volumes_code"] = bin_volumes_code
        attrs["bin_areas_code"] = bin_areas_code

    return r, attrs
