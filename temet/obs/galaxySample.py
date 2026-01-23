"""
Observations: re-create various mock galaxy samples to match surveys/datasets.
"""

from os import mkdir
from os.path import isdir, isfile

import h5py
import numpy as np

from ..cosmo.cloudy import cloudyIon
from ..cosmo.util import redshiftToSnapNum
from ..load.data import berg2019, chen2018zahedy2019, johnson2015, werk2013
from ..util.helper import closest, logZeroNaN, reportMemory
from ..util.sphMap import sphMap
from ..vis.render import getHsmlForPartType


def obsMatchedSample(sP, datasetName="COS-Halos", numRealizations=100):
    """Get a matched sample of simulated galaxies which match an observational abs. line dataset."""
    np.random.seed(424242)

    if datasetName == "COS-Halos":
        # load
        gals, logM, redshift, sfr, sfr_err, sfr_limit, R, _, _, _ = werk2013()
        logM_err = 0.2  # dex, assumed
        R_err = 1.0  # kpc, assumed
        sfr_err[sfr_limit] = 0.0  # upper limits

        # define how we will create this sample, by matching on what quantities
        propList = ["mstar_30pkpc_log", "ssfr_30pkpc_log", "central_flag", "isolated3d,mstar30kpc,max,in_300pkpc"]

        # set up required properties and limit types
        shape = (len(gals), numRealizations)

        props = {}
        props["mstar_30pkpc_log"] = np.zeros(shape, dtype="float32")
        props["ssfr_30pkpc_log"] = np.zeros(shape, dtype="float32")

        props["central_flag"] = np.zeros(shape, dtype="int16")
        props["central_flag"].fill(1)  # realization independent, always required

        props["isolated3d,mstar30kpc,max,in_300pkpc"] = np.zeros(shape, dtype="int16")
        props["isolated3d,mstar30kpc,max,in_300pkpc"].fill(1)  # realization independent, always required

        props["impact_parameter"] = np.zeros(shape, dtype="float32")

        limits = {}
        for propName in propList:
            # -1=ignore, 0=compute in distance (def.), 1=upper limit, 2=lower limit, 3=exact match required
            limits[propName] = np.zeros(shape, dtype="int16")

        limits["ssfr_30pkpc_log"][np.where(sfr_limit), :] = 1  # realization independent, upper limit
        limits["central_flag"][:] = 3  # realization/galaxy indepedent, exact
        limits["isolated3d,mstar30kpc,max,in_300pkpc"][:] = 3  # realization/galaxy indepedent, exact

        # create realizations by adding appropriate noise to obs
        for i in range(numRealizations):
            impact_param_random_err = np.random.normal(loc=0.0, scale=R_err, size=len(gals))
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))
            sfr_rnd_err = np.random.normal(loc=0.0, scale=sfr_err, size=len(gals))
            ssfr = (sfr + sfr_rnd_err) / 10.0 ** (logM + mass_random_err_log)

            props["mstar_30pkpc_log"][:, i] = logM + mass_random_err_log
            props["ssfr_30pkpc_log"][:, i] = np.log10(ssfr)
            props["impact_parameter"][:, i] = R + impact_param_random_err

    if datasetName in ["eCGM", "eCGMfull"]:
        # load Johnson+ (2015) sample
        if datasetName == "eCGM":
            surveys = ["IMACS", "SDSS"]  # i.e. exclude the COS-Halos data
        if datasetName == "eCGMfull":
            surveys = ["IMACS", "SDSS", "COS-Halos"]

        gals, logM, redshift, sfr, sfr_err, sfr_limit, R, _, _, _ = johnson2015(surveys=surveys)
        log_ssfr = np.log10(sfr / 10.0**logM)
        env = np.array([gal["environment"] for gal in gals])  # 0=I (isolated), 1=NI (not isolated)

        logM_err = 0.25  # dex, assumed
        R_err = 2.0  # kpc, assumed
        logM_min_for_iso = 9.5  # isolation criteria only enforced above (>2 sigma of logM_err hits 9.0)

        # define how we will create this sample, by matching on what quantities
        propList = ["mstar_30pkpc_log", "ssfr_30pkpc_log", "isolated3d,mstar30kpc,max_third,in_500pkpc"]

        # set up required properties and limit types
        shape = (len(gals), numRealizations)

        props = {}
        props["mstar_30pkpc_log"] = np.zeros(shape, dtype="float32")

        props["ssfr_30pkpc_log"] = np.zeros(shape, dtype="float32")
        props["ssfr_30pkpc_log"].fill(-11.0)  # we use as either an upper or lower limit

        w_env = np.where((env == 0) & (logM > logM_min_for_iso))  # set req iso only for 'I' env gals above M
        props["isolated3d,mstar30kpc,max_third,in_500pkpc"] = np.zeros(shape, dtype="int16")
        props["isolated3d,mstar30kpc,max_third,in_500pkpc"][w_env, :].fill(1)  # realization independent

        props["impact_parameter"] = np.zeros(shape, dtype="float32")

        limits = {}
        for propName in propList:
            # note only 'mstar_30pkpc_log' used in distance computation
            limits[propName] = np.zeros(shape, dtype="int16")

        w_early = np.where(log_ssfr < -11.0)
        w_late = np.where(log_ssfr > -11.0)
        limits["ssfr_30pkpc_log"][w_early, :] = 1  # realization independent, upper limit
        limits["ssfr_30pkpc_log"][w_late, :] = 2  # realization independent, lower limit

        # isolation: galaxy dependent, exact (required either I or NI if Mstar>logM_min_for_iso)
        limits["isolated3d,mstar30kpc,max_third,in_500pkpc"][w_env, :] = 3
        w_env2 = np.where((env == 1) & (logM > logM_min_for_iso))
        limits["isolated3d,mstar30kpc,max_third,in_500pkpc"][w_env2, :] = 3
        w_env_unenforced = np.where(logM <= logM_min_for_iso)
        limits["isolated3d,mstar30kpc,max_third,in_500pkpc"][w_env_unenforced, :] = -1

        # create realizations by adding appropriate noise to obs
        for i in range(numRealizations):
            impact_param_random_err = np.random.normal(loc=0.0, scale=R_err, size=len(gals))
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))

            props["mstar_30pkpc_log"][:, i] = logM + mass_random_err_log
            props["impact_parameter"][:, i] = R + impact_param_random_err

    if datasetName == "LRG-RDR":
        # load
        gals, logM, redshift, ssfr, ssfr_err, ssfr_lim, b, _, _, _ = berg2019()  # RDR survey

        logM_err = 0.2  # dex, assumed
        R_err = 2.0  # kpc, assumed
        ssfr_err[ssfr_lim] = 0.0  # upper limits

        # define how we will create this sample, by matching on what quantities
        propList = ["mstar_30pkpc_log", "ssfr_30pkpc_log", "central_flag"]

        # set up required properties and limit types
        shape = (len(gals), numRealizations)

        props = {}
        props["mstar_30pkpc_log"] = np.zeros(shape, dtype="float32")
        props["ssfr_30pkpc_log"] = np.zeros(shape, dtype="float32")

        props["central_flag"] = np.zeros(shape, dtype="int16")
        props["central_flag"].fill(1)  # realization independent, always required

        props["impact_parameter"] = np.zeros(shape, dtype="float32")

        limits = {}
        for propName in propList:
            # -1=ignore, 0=compute in distance (def.), 1=upper limit, 2=lower limit, 3=exact match required
            limits[propName] = np.zeros(shape, dtype="int16")

        limits["ssfr_30pkpc_log"][np.where(ssfr_lim), :] = 1  # realization independent, upper limit
        limits["central_flag"][:] = 3  # realization/galaxy indepedent, exact

        # create realizations by adding appropriate noise to obs
        for i in range(numRealizations):
            impact_param_random_err = np.random.normal(loc=0.0, scale=R_err, size=len(gals))
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))
            ssfr_random_err_log = np.random.normal(loc=0.0, scale=ssfr_err, size=len(gals))

            props["mstar_30pkpc_log"][:, i] = logM + mass_random_err_log
            props["ssfr_30pkpc_log"][:, i] = ssfr + ssfr_random_err_log
            props["impact_parameter"][:, i] = b + impact_param_random_err

    if datasetName == "COS-LRG":
        # load (note: 5 systems in common with RDR)
        gals, logM, redshift, ug, ug_err, ug_lim, b, _, _, _, _ = chen2018zahedy2019()  # COS-LRG survey

        logM_err = 0.2  # dex, assumed
        R_err = 2.0  # kpc, assumed

        # define how we will create this sample, by matching on what quantities
        propList = ["mstar_30pkpc_log", "color_C-30kpc-z_ug", "central_flag"]

        # set up required properties and limit types
        shape = (len(gals), numRealizations)

        props = {}
        props["mstar_30pkpc_log"] = np.zeros(shape, dtype="float32")
        props["color_C-30kpc-z_ug"] = np.zeros(shape, dtype="float32")

        props["central_flag"] = np.zeros(shape, dtype="int16")
        props["central_flag"].fill(1)  # realization independent, always required

        props["impact_parameter"] = np.zeros(shape, dtype="float32")

        limits = {}
        for propName in propList:
            # -1=ignore, 0=compute in distance (def.), 1=upper limit, 2=lower limit, 3=exact match required
            limits[propName] = np.zeros(shape, dtype="int16")

        limits["central_flag"][:] = 3  # realization/galaxy indepedent, exact

        # create realizations by adding appropriate noise to obs
        for i in range(numRealizations):
            impact_param_random_err = np.random.normal(loc=0.0, scale=R_err, size=len(gals))
            mass_random_err_log = np.random.normal(loc=0.0, scale=logM_err, size=len(gals))
            ug_random_err = np.random.normal(loc=0.0, scale=ug_err, size=len(gals))

            props["mstar_30pkpc_log"][:, i] = logM + mass_random_err_log
            props["color_C-30kpc-z_ug"][:, i] = ug + ug_random_err
            props["impact_parameter"][:, i] = b + impact_param_random_err

    if datasetName == "SimHalos_115-125":
        # pure theory analysis: all central halos with 10^11.5 < Mhalo/Msun < 10^12.5, one realization each
        numRealizations = 1

        gc = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])

        with np.errstate(invalid="ignore"):
            w_gal = np.where((gc >= 11.5) & (gc < 12.5))

        # define how we will create this sample, by matching exact to the chosen sim quantities
        propList = ["mhalo_200_log"]

        # set up required properties and limit types
        shape = (w_gal[0].size, 1)

        props = {}
        props["mhalo_200_log"] = np.zeros(shape, dtype="float32")
        props["mhalo_200_log"][:, 0] = gc[w_gal]  # exact copy of actual halo masses

        limits = {}
        limits["mhalo_200_log"] = np.zeros(shape, dtype="int16")
        limits["mhalo_200_log"][:, 0] = 3  # all exact

        # copy other parameters of interest
        _, _, _, _, _, _, R, _, _, _ = werk2013()  # will copy impact parameters randomly from COS-Halos
        redshift = [sP.redshift]  # single z=0.0

        for propName in ["mstar_30pkpc_log", "ssfr_30pkpc_log", "impact_parameter"]:
            props[propName] = np.zeros(shape, dtype="float32")
        props["mstar_30pkpc_log"][:, 0] = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])[w_gal]
        props["ssfr_30pkpc_log"][:, 0] = sP.groupCat(fieldsSubhalos=["ssfr_30pkpc_log"])[w_gal]
        props["impact_parameter"][:, 0] = np.random.choice(R, size=shape[0], replace=True)

    # save file exists?
    saveFilename = sP.derivPath + "obsMatchedSample_%s_%d.hdf5" % (datasetName, numRealizations)

    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # no, compute now
    r = {}

    if len(redshift) > 1:
        # match each galaxy redshift to a simulation snapshot
        r["snaps"] = redshiftToSnapNum(redshift, sP)
    else:
        # match all observed galaxies to the single simulation redshift 'z'
        r["snaps"] = [redshiftToSnapNum(redshift, sP)] * shape[0]

    # save the final subfind IDs we select via matching
    r["selected_inds"] = np.zeros(shape, dtype="int32")

    for propName in propList:
        r[propName] = np.zeros(shape, dtype=props[propName].dtype)

    for propName in set(props.keys()) - set(propList):
        r[propName] = props[propName]  # straight copy, not used in selection (i.e. impact parameter)

    origSnap = sP.snap

    for snap in np.unique(r["snaps"]):
        # which galaxies correspond to this snap?
        w_loc = np.where(r["snaps"] == snap)
        print("[snap %3d] N = %d" % (snap, len(w_loc[0])))

        # load catalog properties at this redshift
        sP.setSnap(snap)
        sim_props = sP.groupCat(fieldsSubhalos=propList, sq=False)
        if isinstance(sim_props, np.ndarray):
            sim_props = {propList[0]: sim_props}  # force into dict

        # loop over observed galaxies
        for gal_num in w_loc[0]:
            # loop over requested realizations
            if numRealizations > 1:
                print(" [%2d] [" % gal_num, end="")
            for realization in range(numRealizations):
                # subset of absolutely consistent simulated systems (i.e. handle limits)
                mask = np.ones(sim_props[propList[0]].size, dtype="bool")
                if numRealizations > 1:
                    print(".", end="")

                for propName in propList:
                    loc_limit = limits[propName][gal_num, realization]
                    loc_prop_val = props[propName][gal_num, realization]

                    if loc_limit == 1:  # upper limit
                        mask &= sim_props[propName] < loc_prop_val
                    if loc_limit == 2:  # lower limit
                        mask &= sim_props[propName] > loc_prop_val
                    if loc_limit == 3:  # exact, i.e. only integer makes sense
                        mask &= sim_props[propName] == loc_prop_val

                w_mask = np.where(mask)

                if not len(w_mask[0]):
                    # no galaxies satisfy limits, copy previous
                    r["selected_inds"][gal_num, realization] = r["selected_inds"][gal_num, realization - 1]
                    continue

                # L1 norm (Manhattan distance metric) for remaining properties (i.e. handle non-limits)
                dists = np.zeros(w_mask[0].size)

                for propName in propList:
                    loc_limit = limits[propName][gal_num, realization]
                    loc_prop_val = props[propName][gal_num, realization]

                    if loc_limit == 0:
                        dists += np.abs(sim_props[propName][w_mask] - loc_prop_val)

                # select minimum distance in the space of properties to be matched
                w_nan = np.isnan(dists)
                dists[w_nan] = np.nanmax(dists) + 1.0  # filter out nan arising from simulated galaxies

                r["selected_inds"][gal_num, realization] = w_mask[0][dists.argmin()]

            # store properties as matched
            for propName in propList:
                r[propName][gal_num, :] = sim_props[propName][r["selected_inds"][gal_num, :]]

            if numRealizations > 1:
                print("]")
            print(
                " [%2d] selected_inds = %s..."
                % (gal_num, ",".join(["%s" % d for d in r["selected_inds"][gal_num, 0:3]]))
            )

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    return r


def addIonColumnPerSystem(sP, sim_sample, config="COS-Halos"):
    """Derive one column density per simulated galaxy, comparable with the observational setup.

    To do so, compute gridded column densities around a sample of simulated galaxies and attach a
    single column value to each in analogy to the observational dataset.
    """
    if config in ["COS-Halos", "SimHalos_115-125"]:
        # grid parameters
        partType = "gas"
        ionName = "O VI"
        projDepth = sP.units.codeLengthToKpc(
            2000.0
        )  # pkpc in projection direction (z=0 conversion), same as rad profiles
        gridSize = 800.0  # pkpc, box sidelength
        gridRes = 2.0  # pkpc, pixel size
        axes = [0, 1]  # x,y

    if config in ["eCGM", "eCGMfull"]:
        # grid parameters (Johnson+ 2015)
        partType = "gas"
        ionName = "O VI"
        projDepth = 4000.0  # 4 pMpc is (400km/s for H(z) @ z~0.2), i.e. still only half if J15 really +/-200
        gridSize = 2600.0  # pkpc, box sidelength
        gridRes = 2.0  # pkpc, pixel size
        axes = [0, 1]  # x,y

    if config == "HI_rudie":
        # grid parameters (zooms.II)
        partType = "gas"
        ionName = "H I"
        projDepth = 4000.0  # should be 1000 km/s to match Josh
        gridSize = 600.0  # need out to 2rvir
        gridRes = 2.0
        axes = [0, 1]  # x,y

    if config == "LRG-RDR":
        # grid parameters (Berg+ 2019)
        partType = "gas"
        ionName = "H I"  #'MHI_GK' # 'H I' with H2 removed following G&K model
        projDepth = 11190.0  # +/- 1000 km/s (i.e. dv/(sP.units.H_z*1000) is pMpc at this redshift, assumed z=0.5)
        gridSize = 1200.0  # pkpc, need out to b = 500kpc
        gridRes = 2.0
        axes = [0, 1]  # x,y

    if config in ["COS-LRG HI", "COS-LRG MgII"]:
        # grid parameters (Chen+ 2018, Zahedy+ 2018)
        partType = "gas"
        ionName = "H I" if "HI" in config else "Mg II"
        projDepth = 5600.0  # +/- 500 km/s (z=0.5)
        gridSize = 500.0  # pkpc, need out to b = 160kpc
        gridRes = 2.0
        axes = [0, 1]  # x,y

    # save file exists?
    saveFilename = sP.derivPath + "obsMatchedColumns_%s_%d.hdf5" % (config, sim_sample["selected_inds"].size)

    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        r.update(sim_sample)
        return r

    if not isdir(sP.derivPath + "grids/"):
        mkdir(sP.derivPath + "grids/")
    if not isdir(sP.derivPath + "grids/%s/" % config):
        mkdir(sP.derivPath + "grids/%s/" % config)

    # no, compute now
    r = {}
    r["column"] = np.zeros(sim_sample["selected_inds"].shape, dtype="float32")

    origSnap = sP.snap
    ion = cloudyIon(None)
    np.random.seed(4242 + sim_sample["selected_inds"].size)

    nPixels = [int(gridSize / gridRes), int(gridSize / gridRes)]  # square

    def _gridFilePath(ind):
        gridPath = sP.derivPath + "grids/%s/" % config
        gridFile = gridPath + "snap-%d_ind-%d_axes-%d%d.hdf5" % (snap, ind, axes[0], axes[1])
        return gridFile

    print("sim matches cover snapshots: ", np.unique(sim_sample["snaps"]))

    for snap in np.unique(sim_sample["snaps"]):
        # which realized galaxies (a unique set) are in this snap?
        sP.setSnap(snap)
        w_loc = np.where(sim_sample["snaps"] == snap)

        inds_all = sim_sample["selected_inds"][w_loc, :].ravel()
        inds_uniq = np.unique(inds_all)

        print("[snap %3d] N_all = %d N_unique = %d" % (snap, inds_all.size, inds_uniq.size))

        # check which grids are already made / if any are missing
        allExist = True
        inds_todo = []

        for ind in inds_uniq:
            if not isfile(_gridFilePath()):
                inds_todo.append(ind)
                allExist = False

        if allExist:
            print(" all subhalos done, skipping this snapshot.")
            continue

        # process: global load all particle data needed
        massField = "%s mass" % (ionName) if " " in ionName else ionName

        print(" loading mass [%s]..." % massField, reportMemory())
        mass = sP.snapshotSubsetP(partType, massField, float32=True)  # .astype('float32')
        print(" loading pos...", reportMemory())
        pos = sP.snapshotSubsetP(partType, "pos")
        print(" loading hsml...", reportMemory())
        hsml = getHsmlForPartType(sP, partType)

        assert mass.min() >= 0.0

        # loop over all inds
        for i, ind in enumerate(inds_todo):
            # already exists?
            if isfile(_gridFilePath()):
                print(" skipping   [%3d of %3d] ind = %d, already exists" % (i, len(inds_todo), ind))
                continue
            print(" projecting [%3d of %3d] ind = %d" % (i, len(inds_todo), ind))

            # configure grid for this galaxy
            boxSizeImg = sP.units.physicalKpcToCodeLength(np.array([gridSize, gridSize, projDepth]))
            boxCenter = sP.groupCatSingle(subhaloID=ind)["SubhaloPos"]

            boxSizeImg = boxSizeImg[[axes[0], axes[1], 3 - axes[0] - axes[1]]]
            boxCenter = boxCenter[[axes[0], axes[1], 3 - axes[0] - axes[1]]]

            # call projection
            grid_d = sphMap(
                pos=pos,
                hsml=hsml,
                mass=mass,
                quant=None,
                axes=axes,
                ndims=3,
                boxSizeSim=sP.boxSize,
                boxSizeImg=boxSizeImg,
                boxCen=boxCenter,
                nPixels=nPixels,
                colDens=True,
                multi=False,
            )

            grid_d /= ion.atomicMass(ionName.split()[0])  # [H atoms/cm^2] to [ions/cm^2]
            grid_d = logZeroNaN(sP.units.codeColDensToPhys(grid_d, cgs=True, numDens=True))  # [log 1/cm^2]

            # save
            with h5py.File(_gridFilePath(), "w") as f:
                f["grid"] = grid_d

        # TODO: temporary, exit after each snap calculation (to free mem)
        # print('Quitting now, REMOVE THIS! Re-run to continue with next snap...')
        # return

    # create 2d distance mask and in order to select correct distance 'ring'
    # note: pixel scale is constant in pkpc, variable in code units
    zz = np.indices(nPixels)
    dist_mask = np.sqrt(((np.flip(zz[1], 1) - zz[1]) / 2) ** 2 + ((np.flip(zz[0], 0) - zz[0]) / 2) ** 2)
    dist_mask_local = dist_mask * gridRes  # for now, constant per halo, pkpc

    # loop over the snapshot set again
    for snap in np.unique(sim_sample["snaps"]):
        # which realized galaxies (a unique set) are in this snap?
        sP.setSnap(snap)
        w_loc = np.where(sim_sample["snaps"] == snap)

        # all grids now exist, process them to extract single column values per galaxy
        for gal_num in w_loc[0]:
            # loop through realizations and load the grid of each
            inds = np.squeeze(sim_sample["selected_inds"][gal_num, :])
            # if inds.ndim == 0: inds = np.reshape()
            print(" [%2d] compute final column for each realization..." % gal_num)

            for realization_num in range(len(inds)):
                with h5py.File(_gridFilePath(), "r") as f:
                    grid = f["grid"][()]

                # find the unique distance in the mask closest to the requested b parameter
                target_impact_parameter = sim_sample["impact_parameter"][gal_num, realization_num]
                discrete_dist_val, _ = closest(dist_mask_local, target_impact_parameter)

                # find all the pixels that share this [float32] distance value
                w_dist = np.where(dist_mask_local == discrete_dist_val)

                # randomly choose a pixel at the correct distance and save
                valid_values = grid[w_dist]
                chosen_val = np.random.choice(valid_values)
                for _iter in range(20):
                    if np.isfinite(chosen_val):
                        break  # otherwise, choose again
                    chosen_val = np.random.choice(valid_values)

                r["column"][gal_num, realization_num] = chosen_val

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    r.update(sim_sample)
    return r


def ionCoveringFractions(sP, sim_sample, config="COS-Halos"):
    """Compute covering fraction vs. impact parameter profiles for the mock samples.

    Split into the same sub-samples as the data.
    """
    numRadBins = 100
    perc_vals = [10, 16, 38, 50, 62, 84, 90]
    colDensThresholds = [12.5, 13.0, 13.5, 14.0, 14.15, 14.5, 15.0, 15.5, 16.0]

    shape = sim_sample["selected_inds"].shape
    numRealizations = sim_sample["column"].shape[1]

    if config in ["COS-Halos", "SimHalos_115-125"]:
        # grid parameters
        gridSize = 800.0  # pkpc, box sidelength
        gridRes = 2.0  # pkpc, pixel size
        axes = [0, 1]  # x,y

        gal_subsets = ["all", "ssfr_lt_n11", "ssfr_gt_n11", "mstar_lt_105", "mstar_gt_105"]

    if config in ["eCGM", "eCGMfull"]:
        # grid parameters (Johnson+ 2015)
        gridSize = 2600.0  # pkpc, box sidelength
        gridRes = 2.0  # pkpc, pixel size
        axes = [0, 1]  # x,y

        gal_subsets = [
            "all",
            "ssfr_lt_n11",
            "ssfr_gt_n11",
            "ssfr_lt_n11_I",
            "ssfr_gt_n11_I",
            "ssfr_lt_n11_NI",
            "ssfr_gt_n11_NI",
        ]

        # load halo radii (use whatever was derived from the Kravstov AM + Bryan definition procedure)
        if config == "eCGM":
            gals, _, _, _, _, _, R, _, _, _ = johnson2015()
        if config == "eCGMfull":
            gals, _, _, _, _, _, R, _, _, _ = johnson2015(surveys=["IMACS", "SDSS", "COS-Halos"])
        assert np.array_equal(R, [gal["R"] for gal in gals])  # verify
        assert len(gals) == shape[0]  # verify

        R_Rh = np.array([gal["R_Rh"] for gal in gals])  # dimensionless ratio
        halo_Rh = R / R_Rh

        # load observed isolation flag to support 'I' and 'NI' galaxy subsets later
        env = np.array([gal["environment"] for gal in gals])  # 0=I (isolated), 1=NI (not isolated)
        sim_sample["halo_env"] = np.zeros(shape, dtype="int16")
        for i in range(numRealizations):
            sim_sample["halo_env"][:, i] = env

    if config == "HI_rudie":
        # grid parameters (zooms.II)
        gridSize = 600.0  # need out to 2rvir
        gridRes = 2.0
        axes = [0, 1]  # x,y

        gal_subsets = ["all"]  # single halo
        halo_Rh = [sP.units.codeLengthToKpc(sP.groupCatSingle(haloID=0)["Group_R_Crit200"])]
        colDensThresholds = [15.0, 17.2, 19.0, 20.3]  # usual pLLS/LLS/DLA definitions

    nPixels = [int(gridSize / gridRes), int(gridSize / gridRes)]  # square

    # grids all exist already?
    saveFilename0 = sP.derivPath + "obsMatchedColumns_%s_%d.hdf5" % (config, sim_sample["selected_inds"].size)

    if not isfile(saveFilename0):
        raise Exception("Compute addIonColumnPerSystem() first to calculate all needed grids.")

    # save file exists?
    saveFilename = sP.derivPath + "obsCoveringFracs_%s_%d_nc%d_np%d.hdf5" % (
        config,
        sim_sample["selected_inds"].size,
        len(colDensThresholds),
        len(perc_vals),
    )

    if isfile(saveFilename):
        r = {}
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # no, compute now
    r = {}
    r["colDensThresholds"] = np.array(colDensThresholds)
    r["perc_vals"] = np.array(perc_vals)

    for relStr in ["", "_rel"]:
        # iter=0 for pkpc, iter=1 for pkpc normalized by halo virial radii
        r["galaxies_indiv%s" % relStr] = np.zeros(
            (shape[0], shape[1], len(colDensThresholds), numRadBins), dtype="float32"
        )

        for gs in gal_subsets:
            r["%s_mean%s" % (gs, relStr)] = np.zeros((len(colDensThresholds), numRadBins), dtype="float32")
            r["%s_stddev%s" % (gs, relStr)] = np.zeros((len(colDensThresholds), numRadBins), dtype="float32")
            r["%s_percs%s" % (gs, relStr)] = np.zeros(
                (len(colDensThresholds), numRadBins, len(perc_vals)), dtype="float32"
            )
            r["%s_mean%s" % (gs, relStr)].fill(np.nan)
            r["%s_stddev%s" % (gs, relStr)].fill(np.nan)
            r["%s_percs%s" % (gs, relStr)].fill(np.nan)

    origSnap = sP.snap

    def _gridFilePath():
        gridPath = sP.derivPath + "grids/%s/" % config
        gridFile = gridPath + "snap-%d_ind-%d_axes-%d%d.hdf5" % (snap, ind, axes[0], axes[1])
        return gridFile

    # iter=0 bin in pkpc, iter=1 bin in pkpc normalized by halo virial radii
    for iter in [0, 1]:
        # create 2d distance mask and in order to select correct distance 'ring'
        # note: pixel scale is constant in pkpc, variable in code units
        zz = np.indices(nPixels)
        dist_mask = np.sqrt(((np.flip(zz[1], 1) - zz[1]) / 2) ** 2 + ((np.flip(zz[0], 0) - zz[0]) / 2) ** 2)
        dist_mask_local = dist_mask * gridRes  # for now, constant per halo, pkpc
        dist_mask_local = dist_mask_local.ravel()  # flatten

        if iter == 0:
            # setup radial bins
            relStr = ""
            radMax = gridSize / 2
            r["radBins"] = np.linspace(gridRes, radMax, numRadBins)

            # save index sets for each radial bin
            rad_masks = []
            num_in_rad = []

            for radBin in r["radBins"]:
                # which pixels -satisfy- the radial cut
                w_rad = dist_mask_local < radBin
                rad_masks.append(w_rad)
                num_in_rad.append(np.count_nonzero(w_rad))

        if iter == 1:
            # setup radial bins (log b/rvir)
            relStr = "_rel"
            r["radBins_rel"] = np.linspace(-2.0, 1.5, numRadBins)

        # loop over the snapshot set again
        for snap in np.unique(sim_sample["snaps"]):
            # which realized galaxies (a unique set) are in this snap?
            sP.setSnap(snap)
            w_loc = np.where(sim_sample["snaps"] == snap)

            if iter == 1 and config in ["COS-Halos", "SimHalos_115-125"]:
                # load virial radii of all halos now (we take the mean per galaxy across all
                # its realizations and use this uniformly for each realization)
                halo_Rh = sP.groupCat(fieldsSubhalos=["rhalo_200"])

            # all grids now exist, process them to extract single column values per galaxy
            for gal_num in w_loc[0]:
                # loop through realizations and load the grid of each
                inds = np.squeeze(sim_sample["selected_inds"][gal_num, :])
                print(" [%2d] compute covering fractions for each realization..." % gal_num)

                if iter == 1:
                    # galaxy dependent: save index sets for each radial bin
                    rad_masks = []
                    num_in_rad = []

                    if config in ["eCGM", "eCGMfull", "HI_rudie"]:
                        local_halo_radius = halo_Rh[gal_num]
                    if config in ["COS-Halos", "SimHalos_115-125"]:
                        local_halo_radius = np.nanmean(halo_Rh[inds])

                    dist_mask_rel_log = np.log10(dist_mask_local / local_halo_radius)

                    for radBin in r["radBins_rel"]:
                        # which pixels -satisfy- the radial cut
                        w_rad = dist_mask_rel_log < radBin
                        rad_masks.append(w_rad)
                        num_in_rad.append(np.count_nonzero(w_rad))

                if numRealizations > 1:
                    print(" [", end="")
                for realization_num, ind in enumerate(inds):
                    if numRealizations > 1:
                        print(".", end="")
                    with h5py.File(_gridFilePath(ind), "r") as f:
                        grid = f["grid"][()].ravel()  # flatten

                    # loop over thresholds
                    for i, thresh in enumerate(colDensThresholds):
                        # which pixels satisfy the threshold alone?
                        w_covered = grid >= thresh

                        # loop over radial distances
                        for j in range(numRadBins):
                            # combine masks: which [flattened] pixels satisfy both cuts?
                            num_in_rad_above_thresh = np.count_nonzero(w_covered & rad_masks[j])

                            if num_in_rad[j] == 0:
                                continue

                            # save fraction
                            frac = float(num_in_rad_above_thresh) / num_in_rad[j]
                            assert np.isfinite(frac)

                            r["galaxies_indiv%s" % relStr][gal_num, realization_num, i, j] = frac

                if numRealizations > 1:
                    print("]")

        # compute binned values across all realized galaxies
        for i in range(len(colDensThresholds)):
            for j in range(numRadBins):
                # different subsets of the galaxy sample, including 'all'
                for gs in gal_subsets:
                    local_vals = r["galaxies_indiv%s" % relStr][:, :, i, j].ravel()

                    # update local_vals for this subset
                    if gs in ["ssfr_lt_n11", "ssfr_gt_n11"]:
                        ssfr = sim_sample["ssfr_30pkpc_log"].ravel()
                        with np.errstate(invalid="ignore"):
                            if gs == "ssfr_lt_n11":
                                w = np.where((ssfr < -11.0) | np.isnan(ssfr))
                            if gs == "ssfr_gt_n11":
                                w = np.where(ssfr >= -11.0)
                        local_vals = local_vals[w]

                    if gs in ["mstar_lt_105", "mstar_gt_105"]:
                        mstar = sim_sample["mstar_30pkpc_log"].ravel()
                        if gs == "mstar_lt_105":
                            w = np.where(mstar < 10.5)
                        if gs == "mstar_gt_105":
                            w = np.where(mstar >= 10.5)
                        local_vals = local_vals[w]

                    if gs in ["ssfr_lt_n11_I", "ssfr_lt_n11_NI", "ssfr_gt_n11_I", "ssfr_gt_n11_NI"]:
                        ssfr = sim_sample["ssfr_30pkpc_log"].ravel()
                        env = sim_sample["halo_env"].ravel()

                        with np.errstate(invalid="ignore"):
                            if gs == "ssfr_lt_n11_I":
                                w = np.where(((ssfr < -11.0) | np.isnan(ssfr)) & env)
                            if gs == "ssfr_gt_n11_I":
                                w = np.where((ssfr >= -11.0) & env)
                            if gs == "ssfr_lt_n11_NI":
                                w = np.where(((ssfr < -11.0) | np.isnan(ssfr)) & ~env)
                            if gs == "ssfr_gt_n11_NI":
                                w = np.where((ssfr >= -11.0) & ~env)
                        local_vals = local_vals[w]

                    if local_vals.size == 0:
                        continue

                    r["%s_mean%s" % (gs, relStr)][i, j] = np.nanmean(local_vals)
                    r["%s_stddev%s" % (gs, relStr)][i, j] = np.nanstd(local_vals)
                    r["%s_percs%s" % (gs, relStr)][i, j, :] = np.nanpercentile(
                        local_vals, perc_vals, interpolation="linear"
                    )

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    return r
