"""
"Zooming in on accretion" paper series (II) - Suresh+ 2019 (http://arxiv.org/abs/1811.01949).
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from temet.cosmo.cloudy import cloudyIon
from temet.obs.galaxySample import addIonColumnPerSystem, ionCoveringFractions
from temet.plot import snapshot
from temet.plot.config import colors, figsize, linestyles, lw, sKn, sKo
from temet.util import simParams
from temet.util.helper import logZeroNaN, running_median
from temet.util.match import match
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def check_box(snap):
    """Visualize halo."""
    panels = []

    run = "zooms2_tng"
    res = 11
    hInd = 2
    subhaloInd = 0
    nPixels = [1000, 1000]
    zoomFac = 0.1

    sP = simParams(res=res, run=run, hInd=hInd, snap=snap)
    redshift = sP.redshift

    panels.append({"partType": "dm", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})

    class plotConfig:
        plotStyle = "open"  # open, edged
        rasterPx = nPixels if isinstance(nPixels, list) else [nPixels, nPixels]
        colorbars = True

        saveFilename = "./boxImage_%s_%d_%s-%s.png" % (sP.simName, snap, panels[0]["partType"], panels[0]["partField"])

    renderBox(panels, plotConfig, locals())


def visualize_halo(conf=1, quadrant=False, snap=None):
    """Visualize single final halo of h2_L11_12_FP (boosted, sims.zooms2) at z=2.25."""
    panels = []

    run = "zooms2_josh"  # 'zooms2'
    res = 11
    hInd = 2
    subhaloInd = 0
    variant = "FP"  # MO, PO, FP, FP1/FP2/FP3 None

    redshift = 2.25 if snap is None else None

    rVirFracs = [1.0]
    method = "sphMap_global"
    nPixels = [3840, 3840]  # 960 or 3840
    axes = [1, 0]
    labelZ = True
    labelScale = True
    labelSim = False
    labelHalo = True
    relCoords = True
    rotation = None

    size = 180.0  # 400.0
    sizeType = "pkpc"
    axesUnits = "kpc"
    depthFac = 1.0

    sP = simParams(res=res, run=run, redshift=redshift, snap=snap, variant=variant, hInd=hInd)
    if redshift is None:
        redshift = sP.redshift

    if quadrant:
        # zoom in to upper right quadrant
        halo = sP.groupCatSingle(haloID=sP.zoomSubhaloID)
        cenShift = [halo["Group_R_Crit200"] * (0.25 + 0.05), halo["Group_R_Crit200"] * (0.25 + 0.05), 0]
        size = sP.units.codeLengthToKpc(halo["Group_R_Crit200"] * 0.4)
        labelHalo = False

    if conf == 0:
        # stellar mass column density
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [4.5, 8.0]})
    if conf == 1:
        # gas column density
        panels.append({"partType": "gas", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.5]})
    if conf == 2:
        # gas OVI column
        panels.append({"partType": "gas", "partField": "O VI", "valMinMax": [12.0, 17.0]})
    if conf == 3:
        # gas MgII column
        vMM = [11.5, 16.5] if quadrant else [10.0, 16.5]
        panels.append({"partType": "gas", "partField": "Mg II", "valMinMax": vMM})
    if conf == 4:
        # temperature
        panels.append({"partType": "gas", "partField": "temp", "valMinMax": [4.2, 5.9]})
    if conf == 5:
        # radial velocity
        panels.append({"partType": "gas", "partField": "radvel", "valMinMax": [-260, 260]})
        nPixels = [600, 600]
    if conf == 6:
        # magnitude of specific angular momentum
        panels.append({"partType": "gas", "partField": "specj_mag", "valMinMax": [2.0, 4.2]})
    if conf == 7:
        # gas metallicity
        panels.append({"partType": "gas", "partField": "metal_solar", "valMinMax": [-2.0, 0.0]})
        nPixels = [600, 600]

    class plotConfig:
        plotStyle = "open"
        rasterPx = int(nPixels[0] * 1.0)
        colorbars = True
        saveFilename = "./%s_%s_%s_%d_%d_z%.2f_%d%s.pdf" % (
            sP.simName,
            panels[0]["partType"],
            panels[0]["partField"],
            res,
            sP.snap,
            redshift,
            size,
            sizeType,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


def visualize_compare_vs_normal(conf=1):
    """Visualize single final halo of h2_L11_12_FP (boosted, sims.zooms2) vs h2_L11_FP (unboosted) at z=2.25."""
    panels = []

    run = "zooms2_josh"  # 'zooms2'
    res = 11
    hInd = 2
    subhaloInd = 0
    redshift = 2.25

    rVirFracs = [0.25, 0.5]
    method = "sphMap_global"
    nPixels = [960, 960]  # 960 or 3840
    axes = [1, 0]
    # labelZ     = True
    # labelScale = 'physical'
    labelSim = True
    labelHalo = False
    relCoords = False
    rotation = None

    size = 180.0  # 400.0
    sizeType = "pkpc"
    depthFac = 1.0

    # zoom in to upper left quadrant
    sP = simParams(res=res, run=run, redshift=redshift, variant="FP", hInd=hInd)
    halo = sP.groupCatSingle(haloID=sP.zoomSubhaloID)
    cenShift = [-halo["Group_R_Crit200"] * (0.25 + 0.05), halo["Group_R_Crit200"] * (0.25 + 0.05), 0]
    size = sP.units.codeLengthToKpc(halo["Group_R_Crit200"] * 0.4)

    if conf == 1:
        # gas column density
        panels.append(
            {
                "variant": "FPorig",
                "partType": "gas",
                "partField": "coldens_msunkpc2",
                "valMinMax": [5.0, 7.6],
                "plawScale": 1.3,
            }
        )
        panels.append(
            {
                "variant": "FP",
                "partType": "gas",
                "partField": "coldens_msunkpc2",
                "valMinMax": [5.0, 7.6],
                "plawScale": 1.3,
            }
        )
    if conf == 2:
        # gas MgII column
        panels.append({"variant": "FPorig", "partType": "gas", "partField": "Mg II", "valMinMax": [10.0, 15.8]})
        panels.append({"variant": "FP", "partType": "gas", "partField": "Mg II", "valMinMax": [10.0, 15.8]})
    if conf == 3:
        # temperature
        panels.append({"variant": "FPorig", "partType": "gas", "partField": "temp", "valMinMax": [4.2, 5.9]})
        panels.append({"variant": "FP", "partType": "gas", "partField": "temp", "valMinMax": [4.2, 5.9]})

    panels[0]["labelScale"] = "physical"
    panels[1]["labelZ"] = True

    class plotConfig:
        plotStyle = "edged"
        rasterPx = int(nPixels[0] * 1.0)
        colorbars = True
        saveFilename = "./vis_compare_L11FP_vs_L11_12FP-%d.pdf" % (conf)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def phase_diagram_ovi():
    """OVI mass phase diagram."""
    sP = simParams(res=11, run="zooms2_josh", variant="FP", hInd=2, redshift=2.25)

    ptType = "gas"
    xQuant = "hdens"
    yQuant = "temp"
    weights = ["O VI mass"]  # ,'O VII mass','O VIII mass']
    xMinMax = [-6.0, 0.0]
    yMinMax = [3.5, 7.0]
    contours = [-3.0, -2.0, -1.0]
    cMinMax = [-4.0, -1.0]  # [-10.0, 0.0]
    hideBelow = True
    smoothSigma = 1.5
    haloIDs = [0]  # None for fullbox

    snapshot.phaseSpace2d(
        sP,
        ptType,
        xQuant,
        yQuant,
        weights=weights,
        haloIDs=haloIDs,
        clim=cMinMax,
        xlim=xMinMax,
        ylim=yMinMax,
        contours=contours,
        smoothSigma=smoothSigma,
        hideBelow=hideBelow,
    )


def phase_diagram_coolingtime():
    """Cooling time phase diagram."""
    sP = simParams(res=11, run="zooms2_josh", variant="FP", hInd=2, redshift=2.25)

    ptType = "gas"
    xQuant = "hdens"
    yQuant = "temp"
    weights = None
    meancolors = ["cooltime"]
    xMinMax = [-6.0, 0.0]
    yMinMax = [3.5, 7.0]
    cMinMax = [-3.0, 1.0]  # 1 Myr to 10 Gyr
    contours = [-2.0, -1.0, 0.0]
    hideBelow = False
    smoothSigma = 1.5
    haloIDs = [0]  # None for fullbox

    snapshot.phaseSpace2d(
        sP,
        ptType,
        xQuant,
        yQuant,
        weights=weights,
        meancolors=meancolors,
        haloIDs=haloIDs,
        clim=cMinMax,
        xlim=xMinMax,
        ylim=yMinMax,
        contours=contours,
        smoothSigma=smoothSigma,
        hideBelow=hideBelow,
    )


def phase_diagram_vs_L11():
    """Density-temperature phase diagram, comparing L11_FP to L11_12_FP."""
    sP1 = simParams(res=11, run="zooms2_josh", variant="FP", hInd=2, redshift=2.25)
    sP2 = simParams(res=11, run="zooms2_josh", variant="FPorig", hInd=2, redshift=2.25)

    ptType = "gas"
    xQuant = "hdens"
    yQuant = "temp"
    weights = ["mass"]
    meancolors = None
    xMinMax = [-6.0, 0.0]
    yMinMax = [3.5, 7.0]
    contours = [-2.0, -1.0, 0.0]
    cMinMax = [-3.0, 1.0]  # 1 Myr to 10 Gyr
    hideBelow = False
    smoothSigma = 1.5
    haloIDs = [0]

    snapshot.phaseSpace2d(
        sP1,
        ptType,
        xQuant,
        yQuant,
        weights=weights,
        meancolors=meancolors,
        haloIDs=haloIDs,
        clim=cMinMax,
        xlim=xMinMax,
        ylim=yMinMax,
        contours=contours,
        smoothSigma=smoothSigma,
        hideBelow=hideBelow,
    )
    snapshot.phaseSpace2d(
        sP2,
        ptType,
        xQuant,
        yQuant,
        weights=weights,
        meancolors=meancolors,
        haloIDs=haloIDs,
        clim=cMinMax,
        xlim=xMinMax,
        ylim=yMinMax,
        contours=contours,
        smoothSigma=smoothSigma,
        hideBelow=hideBelow,
    )


def phase_diagram_ovi_tng50_comparison():
    """OVI mass phase diagram, comparing to TNG50."""
    sP = simParams(res=2160, run="tng", redshift=2.25)
    haloIDs = [100]

    ptType = "gas"
    xQuant = "hdens"
    yQuant = "temp"
    weights = ["O VI mass"]  # ,'O VII mass','O VIII mass']
    xMinMax = [-6.0, 0.0]
    yMinMax = [3.5, 7.0]
    contours = [-3.0, -2.0, -1.0]
    cMinMax = [-4.0, 0.0]  # [-10.0, 0.0]
    hideBelow = True
    smoothSigma = 1.0

    snapshot.phaseSpace2d(
        sP,
        ptType,
        xQuant,
        yQuant,
        weights=weights,
        haloIDs=haloIDs,
        clim=cMinMax,
        xlim=xMinMax,
        ylim=yMinMax,
        contours=contours,
        smoothSigma=smoothSigma,
        hideBelow=hideBelow,
    )


def figure1_res_statistics(conf=0):
    """Figure 1: resolution statistics in mass/size for gas cells, comparing runs."""
    sPs = []

    if conf in [0, 3]:
        # 3 run comparison
        sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, variant="PO", redshift=2.25))
        # sPs.append( simParams(res=11,run='zooms2_josh',hInd=2,variant='MO',redshift=2.25) )
        sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, variant="FP", redshift=2.25))
        sPs.append(simParams(res=11, run="zooms2", hInd=2, redshift=2.25))
    if conf in [1, 2]:
        # just compare L11 primordial vs. L11_12 primordial
        sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, variant="PO", redshift=2.25))
        sPs.append(simParams(res=11, run="zooms2", hInd=2, redshift=2.25))

    haloIDs = np.zeros(len(sPs), dtype="int32")

    if conf == 0:
        # Figure 1, lower left panel
        snapshot.histogram1d(sPs, haloIDs=haloIDs, ptType="gas", ptProperty="mass_msun", sfreq0=True, xlim=[2.0, 4.7])
    if conf == 3:
        # unused (cellsize histograms)
        snapshot.histogram1d(sPs, haloIDs=haloIDs, ptType="gas", ptProperty="cellsize_kpc", sfreq0=True)
    if conf == 1:
        # Figure 1, lower right panel
        snapshot.profile(
            sPs, haloIDs=haloIDs, ptType="gas", ptProperty="cellsize_kpc", sfreq0=True, xlim=[-0.5, 3.0], scope="global"
        )
    if conf == 2:
        # Figure 1, upper panel
        snapshot.profile(sPs, haloIDs=haloIDs, ptType="gas", ptProperty="mass_msun", xlim=[2.0, 4.7], scope="global")


def tracer_ambient_hot_halo():
    """Check the existence of an ambient/pre-existing hot halo at r<0.25rvir, vs. if all hot gas arises from wind."""
    sP = simParams(res=11, run="zooms2_josh", redshift=2.25, variant="FP", hInd=2)

    temp_bins = [
        [4.0, 4.5],
        [4.5, 4.8],
        [4.8, 5.0],
        [5.0, 5.2],
        [5.2, 5.4],
        [5.4, 5.6],
        [5.6, 5.8],
        [5.8, 6.0],
        [6.0, 6.2],
        [6.2, 6.5],
    ]
    rad_min = 0.4
    rad_max = 0.5

    # load ParentIDs of tracer catalog
    with h5py.File(sP.postPath + "tracer_tracks/tr_all_groups_%d_meta.hdf5" % sP.snap) as f:
        ParentIDs = f["ParentIDs"][()]

    # load radius, sfr, make selection
    rad = sP.snapshotSubset("gas", "rad_rvir", subhaloID=0)
    sfr = sP.snapshotSubset("gas", "sfr", subhaloID=0)
    ids = sP.snapshotSubset("gas", "ids", subhaloID=0)

    ww = np.where((sfr == 0.0) & (rad > rad_min) & (rad < rad_max))

    print("Selected [%d] of [%d] gas cells." % (len(ww[0]), sfr.size))

    # load temperature histories
    with h5py.File(sP.postPath + "tracer_tracks/tr_all_groups_%d_temp.hdf5" % sP.snap) as f:
        redshifts = f["redshifts"][()]
        temp = f["temp"][()]

    print("Loaded temperatures.")

    # crossmatch and take selection
    ind_cat, ind_snap = match(ParentIDs, ids[ww])

    print("Crossmatched.")

    rad = rad[ind_snap]
    temp = temp[:, ind_cat]

    # in a number of temp bins, find mean temperature shift as a function of time backwards
    temp_prev = np.zeros((len(temp_bins), redshifts.size), dtype="float32")
    frac_prev = np.zeros((len(temp_bins), redshifts.size), dtype="float32")

    for i, temp_bin in enumerate(temp_bins):
        # locate at z_final
        w = np.where((temp[0, :] > temp_bin[0]) & (temp[0, :] <= temp_bin[1]))[0]
        assert len(w) > 0

        loc_temps = temp[:, w]

        temp_prev[i, :] = np.nanmean(loc_temps, axis=1)
        frac_prev[i, :] = np.sum(loc_temps >= temp_bin[0], axis=1) / float(len(w))
        print(i, len(w))

    # plot
    fig = plt.figure(figsize=[14.0, 10.0])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("redshift")
    ax.set_ylabel("temperature [ log K ]")

    ax.set_ylim([3.8, 6.5])
    ax.set_xlim([2.2, 2.6])

    for i, temp_bin in enumerate(temp_bins):
        label = r"T$_{\rm zf}$ $\in$ [%.1f, %.1f]" % (temp_bin[0], temp_bin[1])
        ax.plot(redshifts, temp_prev[i, :], "-", lw=2.0, label=label)

    ax.legend()
    fig.savefig("temp_evo_rvir=%.2f-%.2f.pdf" % (rad_min, rad_max))
    plt.close(fig)

    # plot 2
    fig = plt.figure(figsize=[14.0, 10.0])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("redshift")
    ax.set_ylabel("fraction of original bin still above bin min temp")

    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([2.2, 2.6])

    for i, temp_bin in enumerate(temp_bins):
        label = r"T$_{\rm zf}$ $\in$ [%.1f, %.1f]" % (temp_bin[0], temp_bin[1])
        ax.plot(redshifts, frac_prev[i, :], "-", lw=2.0, label=label)

    ax.legend()
    fig.savefig("tempfrac_evo_rvir=%.2f-%.2f.pdf" % (rad_min, rad_max))
    plt.close(fig)


def gas_components_time_evo():
    """Plot redshift evolution of total mass in different halo gas components (Fig 4)."""
    sPs = []
    data = []

    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, variant="FP"))
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, variant="FPorig"))

    # get data
    for sP in sPs:
        saveFilename = sP.derivPath + "components_time_evo.hdf5"

        # load existing save file
        if isfile(saveFilename):
            data_loc = {}
            with h5py.File(saveFilename, "r") as f:
                for key in f:
                    data_loc[key] = f[key][()]

            data.append(data_loc)
            print("Loaded: [%s]" % saveFilename)
            continue

        # calculate now
        snaps = np.arange(53)

        data_loc = {
            "mass_m200": np.zeros(snaps.size, dtype="float32"),
            "mass_halo": np.zeros(snaps.size, dtype="float32"),
            "mass_gal": np.zeros(snaps.size, dtype="float32"),
            "mass_cgm": np.zeros(snaps.size, dtype="float32"),
            "mass_cgm_cooldense": np.zeros(snaps.size, dtype="float32"),
            "mass_cgm_non": np.zeros(snaps.size, dtype="float32"),
            "snaps": snaps,
            "redshifts": sP.snapNumToRedshift(snaps),
        }

        for i, snap in enumerate(snaps):
            sP.setSnap(snap)
            print(i)
            gas = sP.snapshotSubset("gas", ["Masses", "temp_log", "StarFormationRate", "nh", "rad_rvir"], haloID=0)

            subhalo = sP.groupCatSingle(subhaloID=0)
            halo = sP.groupCatSingle(haloID=0)

            gal_mass = sP.units.codeMassToLogMsun(subhalo["SubhaloMassInRadType"][sP.ptNum("stars")])
            gal_radius = interp1d([6, 7, 8, 9, 10, 11], [1, 4, 6, 8, 10, 20])(gal_mass)
            gal_rad_rvir = gal_radius / halo["Group_R_Crit200"]

            # define selections
            w_halo = np.where(gas["rad_rvir"] <= 1.0)
            w_gal = np.where(
                (gas["rad_rvir"] <= 1.0) & ((gas["rad_rvir"] <= gal_rad_rvir) | (gas["StarFormationRate"] > 0.0))
            )
            w_cgm = np.where(
                (gas["rad_rvir"] <= 1.0) & (gas["rad_rvir"] > gal_rad_rvir) & (gas["StarFormationRate"] == 0.0)
            )

            w_cooldense = np.where((gas["nh"][w_cgm] > 1e-3) & (gas["temp"][w_cgm] < 5.0))
            w_non = np.where((gas["nh"][w_cgm] <= 1e-3) | (gas["temp"][w_cgm] >= 5.0))

            assert len(w_cooldense[0]) + len(w_non[0]) == len(w_cgm[0])

            # save masses
            data_loc["mass_m200"][i] = sP.units.codeMassToLogMsun(halo["Group_M_Crit200"])
            data_loc["mass_halo"][i] = sP.units.codeMassToLogMsun(gas["Masses"][w_halo].sum())
            data_loc["mass_gal"][i] = sP.units.codeMassToLogMsun(
                gas["Masses"][w_gal].sum() + halo["GroupMassType"][sP.ptNum("stars")]
            )
            data_loc["mass_cgm"][i] = sP.units.codeMassToLogMsun(gas["Masses"][w_cgm].sum())

            data_loc["mass_cgm_cooldense"][i] = sP.units.codeMassToLogMsun(gas["Masses"][w_cgm][w_cooldense].sum())
            data_loc["mass_cgm_non"][i] = sP.units.codeMassToLogMsun(gas["Masses"][w_cgm][w_non].sum())

        data.append(data_loc)

        # save
        with h5py.File(saveFilename, "w") as f:
            for key in data_loc:
                f[key] = data_loc[key]
        print("Saved: [%s]" % saveFilename)

    # plot
    fig = plt.figure(figsize=[10.0, 7.0])
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass of Component [ log M$_{\\rm sun}$ ]")

    ax.set_ylim([8.0, 12.0])
    ax.set_xlim([6.0, 2.25])

    lines = [plt.Line2D([0], [0], color="black", marker="", ls=linestyles[i], lw=lw) for i in range(len(sPs))]
    legend2 = ax.legend(lines, [sP.simName for sP in sPs], loc="upper left")
    ax.add_artist(legend2)

    for i in range(len(sPs)):
        z = data[i]["redshifts"]

        ax.plot(z, data[i]["mass_m200"], linestyles[i], color="black", lw=lw, label="Halo" if i == 0 else "")
        ax.plot(
            z,
            data[i]["mass_gal"],
            linestyles[i],
            color=colors[0],
            lw=lw,
            label="Galaxies (Stars+ISM)" if i == 0 else "",
        )

        # ax.plot(z, data[i]['mass_halo'], linestyles[i], color=colors[1], lw=lw, label='Halo Gas' if i == 0 else '')
        ax.plot(z, data[i]["mass_cgm"], linestyles[i], color=colors[2], lw=lw, label="CGM" if i == 0 else "")

        ax.plot(
            z,
            data[i]["mass_cgm_cooldense"],
            linestyles[i],
            color=colors[3],
            lw=lw,
            label="CGM (cool+dense)" if i == 0 else "",
        )
        ax.plot(
            z,
            data[i]["mass_cgm_non"],
            linestyles[i],
            color=colors[4],
            lw=lw,
            label="CGM (not cool+dense)" if i == 0 else "",
        )

    ax.legend(loc="lower right")
    fig.savefig("figure_4.pdf")
    plt.close(fig)


def gas_components_radial_profiles():
    """Compare CGM cool-dense and non-cool-dense radial profiles between runs."""
    sPs = []
    redshift = 2.25
    binSize = 0.02

    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FP"))  # L11_12_FP
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="PO"))  # L11_12_PO
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FPorig"))  # L11_FP

    # start plot
    fig = plt.figure(figsize=[15.0, 7.0])
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.set_xlabel(r"R / R$_{\rm vir}$")
    ax1.set_ylabel(r"Gas Density (median cell n$_{\rm H}$) [ log cm$^{-3}$ ]")
    ax1.set_ylim([-5.0, 0.0])
    ax1.set_xlim([0.0, 1.2])

    ax2.set_xlabel(r"R / R$_{\rm vir}$")
    ax2.set_ylabel(r"Gas Density [ log M$_{\rm sun}$ kpc$^{-3}$ ]")
    ax2.set_ylim([0.0, 6.0])
    ax2.set_xlim([0.0, 1.2])

    lines = [plt.Line2D([0], [0], color="black", marker="", ls=linestyles[i], lw=lw) for i in range(2)]
    legend2 = ax1.legend(lines, ["cool-dense", "not cool-dense"], loc="lower left")
    ax1.add_artist(legend2)
    legend3 = ax2.legend(lines, ["cool-dense", "not cool-dense"], loc="lower left")
    ax2.add_artist(legend3)

    for sP in sPs:
        # load
        gas = sP.snapshotSubset("gas", ["pos", "Masses", "temp_log", "StarFormationRate", "nh", "rad_rvir"], haloID=0)
        rvir = sP.groupCatSingle(haloID=0)["Group_R_Crit200"]

        # apply satellite exclusion technique
        mask = np.zeros(gas["Masses"].size, dtype="int16")

        gc = sP.groupCat(fieldsHalos=["GroupNsubs"], fieldsSubhalos=["SubhaloPos", "SubhaloMassInRadType"])
        nSubs = gc["halos"][0]  # of first halo
        sub_mstar = sP.units.codeMassToLogMsun(gc["subhalos"]["SubhaloMassInRadType"][:, sP.ptNum("stars")])[0:nSubs]
        sub_pos = gc["subhalos"]["SubhaloPos"][0:nSubs, :]

        w = np.where(np.isfinite(sub_mstar))
        sub_mstar = sub_mstar[w]
        sub_pos = sub_pos[w[0], :]

        sub_radius = interp1d([3, 7, 8, 9, 10, 11], [1, 4, 6, 8, 10, 20])(sub_mstar)

        for sub_nr in range(sub_mstar.size):
            dists = sP.periodicDists(sub_pos[sub_nr, :], gas["Coordinates"])
            w = np.where(dists <= sub_radius[sub_nr])
            mask[w] = 1

        # select
        w_cgm = np.where((gas["rad_rvir"] > 0.15) & (gas["StarFormationRate"] == 0.0) & (mask == 0))
        w_cooldense = np.where((gas["nh"][w_cgm] > 1e-3) & (gas["temp"][w_cgm] < 5.0))
        w_non = np.where((gas["nh"][w_cgm] <= 1e-3) | (gas["temp"][w_cgm] >= 5.0))

        # median curves of gas cell density values
        xx = gas["rad_rvir"][w_cgm]
        yy = np.log10(gas["nh"][w_cgm])  # log(1/cm^3)

        xm1, ym1, sm1 = running_median(xx[w_cooldense], yy[w_cooldense], binSize=binSize)
        xm2, ym2, sm2 = running_median(xx[w_non], yy[w_non], binSize=binSize)

        # plot (left panel)
        (l,) = ax1.plot(xm1, ym1, ls=linestyles[0], lw=lw, label=sP.simName)
        ax1.plot(xm2, ym2, ls=linestyles[1], lw=lw, color=l.get_color())

        # compute density profile instead as mass_shell/volume_shell
        yy = gas["Masses"][w_cgm]

        rad_bin_edges = np.linspace(0.0, 1.2, 26)  # binsize = 0.05
        rad_bin_centers = ((rad_bin_edges + np.roll(rad_bin_edges, 1)) * 0.5)[1:]

        dens_cooldense = np.zeros(rad_bin_centers.size, dtype="float32")
        dens_non = np.zeros(rad_bin_centers.size, dtype="float32")

        for i in range(rad_bin_edges.size - 1):
            rad_min = rad_bin_edges[i]
            rad_max = rad_bin_edges[i + 1]

            rad_min_kpc = sP.units.codeLengthToKpc(rad_min * rvir)
            rad_max_kpc = sP.units.codeLengthToKpc(rad_max * rvir)
            shell_vol = 4.0 / 3 * np.pi * (rad_max_kpc**3 - rad_min_kpc**3)  # pkpc^3

            w = np.where((xx[w_cooldense] >= rad_min) & (xx[w_cooldense] < rad_max))
            tot_mass_cooldense = sP.units.codeMassToMsun(np.sum(yy[w_cooldense][w]))  # msun

            w = np.where((xx[w_non] >= rad_min) & (xx[w_non] < rad_max))
            tot_mass_non = sP.units.codeMassToMsun(np.sum(yy[w_non][w]))  # msun

            dens_cooldense[i] = np.log10(tot_mass_cooldense / shell_vol)  # zero mass -> nan (not plotted)
            dens_non[i] = np.log10(tot_mass_non / shell_vol)

        # plot (right panel)
        (l,) = ax2.plot(rad_bin_centers, dens_cooldense, ls=linestyles[0], lw=lw, label=sP.simName)
        ax2.plot(rad_bin_centers, dens_non, ls=linestyles[1], lw=lw, color=l.get_color())

    # finish plot
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    fig.savefig("figure_5.pdf")
    plt.close(fig)


def mgii_radial_profile():
    """Compare MgII column density profiles."""
    redshift = 2.25

    sPs = []
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FP"))
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="PO"))
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FPorig"))

    ion = "MgII"
    cenSatSelect = "cen"
    projDim = "2Dz_2Mpc"
    combine2Halo = True
    radRelToVirRad = False

    # plot setup
    lw = 3.0
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    radStr = "Radius" if "3D" in projDim else "Impact Parameter"
    if radRelToVirRad:
        ax.set_xlim([-2.0, 2.0])
        ax.set_xlabel("%s / Virial Radius [ log ]" % radStr)
    else:
        ax.set_xlim([0.0, 100])
        ax.set_xlabel("%s [ pkpc ]" % radStr)

    # 2D mass/column density
    ax.set_ylim([12.0, 17.0])
    ax.set_ylabel(r"Column Number Density $N_{\rm %s}$ [ log cm$^{-2}$ ]" % ion)

    # init
    ionData = cloudyIon(None)

    # loop over each fullbox run
    for sP in sPs:
        # load halo/stellar masses and CSS
        sP.setRedshift(redshift)
        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)

        # load virial radii
        rad = sP.groupCat(fieldsSubhalos=["rhalo_200_code"])
        rad = rad[cssInds]

        fieldName = "Subhalo_RadProfile%s_GlobalFoF_%s_Mass" % (projDim, ion)
        print("[%s]: %s" % (ion, sP.simName))

        ac = sP.auxCat(fields=[fieldName])
        if ac[fieldName] is None:
            continue

        # crossmatch 'subhaloIDs' to cssInds
        ac_inds, css_inds = match(ac["subhaloIDs"], cssInds)
        ac[fieldName] = ac[fieldName][ac_inds, :]
        rad_loc = rad[css_inds]

        # unit conversions: mass per bin to (space mass density) or (space number density)
        yy = ac[fieldName]

        if "3D" in projDim:
            normField = "bin_volumes_code"
            unitConversionFunc = sP.units.codeDensToPhys
        else:
            normField = "bin_areas_code"  # 2D
            unitConversionFunc = sP.units.codeColDensToPhys

        if ac[fieldName].ndim == 2:
            yy /= ac[fieldName + "_attrs"][normField]
        else:
            for radType in range(ac[fieldName].shape[2]):
                yy[:, :, radType] /= ac[fieldName + "_attrs"][normField]

        # from e.g. [code mass / code length^3] -> [ions/cm^3]
        species = ion.replace("I", "").replace("V", "").replace("X", "")  # e.g. 'OVI' -> 'O'
        yy = unitConversionFunc(yy, cgs=True, numDens=True)
        yy /= ionData.atomicMass(species)  # [H atoms/cm^3] to [ions/cm^3]

        # select
        w = [0]

        rvir_pkpc = sP.units.codeLengthToKpc(rad_loc[w])
        radType = 0  # total only

        # sum and calculate percentiles in each radial bin
        if yy.ndim == 3:
            yy_local = np.squeeze(yy[w, :, radType])

            # combine diffuse into other-halo term, and skip separate line?
            if combine2Halo and radType == 2:
                yy_local += np.squeeze(yy[w, :, radType + 1])
            if combine2Halo and radType == 3:
                continue
        else:
            yy_local = np.squeeze(yy[w, :])

        rr = ac[fieldName + "_attrs"]["rad_bins_pkpc"]

        # for low res runs, combine the inner bins which are poorly sampled
        if sP.boxSize == 25000.0:
            nInner = int(20 / (sP.res / 256))
            rInner = np.mean(rr[0:nInner])

            for dim in range(yy_local.shape[0]):
                yy_local[dim, nInner - 1] = np.nanmedian(yy_local[dim, 0:nInner])
            yy_local = yy_local[:, nInner - 1 :]
            rr = np.hstack([rInner, rr[nInner:]])

        # replace zeros by nan so they are not included in percentiles
        yy_local[yy_local == 0.0] = np.nan

        # single profile
        yy_mean = yy_local  # single profile

        # log both axes
        yy_mean = logZeroNaN(yy_mean)
        if radRelToVirRad:
            rr = np.log10(rr)

        if rr.size > sKn:
            yy_mean = savgol_filter(yy_mean, sKn, sKo)

        # plot median line
        label = sP.simName
        (l,) = ax.plot(rr, yy_mean, lw=lw, linestyle=linestyles[radType], label=label)

        # draw rvir lines (or 300pkpc lines if x-axis is already relative to rvir)
        yrvir = ax.get_ylim()
        yrvir = np.array([yrvir[1], yrvir[1] - (yrvir[1] - yrvir[0]) * 0.1]) - 0.25

        xrvir = [rvir_pkpc, rvir_pkpc]
        textStr = r"R$_{\rm vir}$"

        ax.plot(xrvir, yrvir, lw=lw * 1.5, color=l.get_color(), alpha=0.1)
        opts = {"va": "bottom", "ha": "right", "fontsize": 20.0, "alpha": 0.1, "rotation": 90}
        ax.text(xrvir[0] - 0.02, yrvir[1], textStr, color=l.get_color(), **opts)

    # legend
    ax.legend(loc="upper right")

    fig.savefig("figure_9.pdf")
    plt.close(fig)


def hi_covering_frac():
    """Plot radial profiles of HI covering fractions."""
    redshift = 2.25

    sPs = []
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FP"))
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="PO"))
    sPs.append(simParams(res=11, run="zooms2_josh", hInd=2, redshift=redshift, variant="FPorig"))

    config = "HI_rudie"
    colDensThresholds = [15.0, 17.2, 19.0, 20.3]  # usual pLLS/LLS/DLA definitions

    # single halo 0, dummy b
    sim_sample = {
        "snaps": np.array([52]),
        "selected_inds": np.zeros((1, 1), dtype="int32"),
        "impact_parameter": np.zeros((1, 1)) + 100.0,
    }

    # plot setup
    fig = plt.figure(figsize=[figsize[0], figsize[1]])

    # loop over each column density threshold
    for j, thresh in enumerate(colDensThresholds):
        # start panel
        ax = fig.add_subplot(2, 2, j + 1)
        ax.set_title(r"N > %.1f cm$^{-2}$" % thresh)

        ax.set_xlim([0.0, 2.2])
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel(r"Impact Parameter / R$_{\rm vir}$")
        ax.set_ylabel(r"f$_{\rm cover}$ (<r)")

        # overplot obs (Rudie 2012, Table 5)
        obs_1rvir = [0.9, 0.3, 0.1, 0.0]
        obs_1rvir_err = [0.09, 0.14, 0.09, 0.1]
        obs_2rvir = [0.68, 0.28, 0.08, 0.04]
        obs_2rvir_err = [0.09, 0.09, 0.05, 0.04]

        label = "Rudie+ (2012)" if j == 1 else ""
        ax.errorbar([1.0], [obs_1rvir[j]], yerr=[obs_1rvir_err[j]], color="black", alpha=0.7, fmt="o", label=label)
        ax.errorbar([2.0], [obs_2rvir[j]], yerr=[obs_2rvir_err[j]], color="black", alpha=0.7, fmt="o")

        # loop over each fullbox run (different linestyles)
        for i, sP in enumerate(sPs):
            # load
            print("[%s]: %s" % (sP.simName, thresh))

            sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

            cf = ionCoveringFractions(sP, sim_sample, config=config)

            # which index for the requested col density threshold?
            assert thresh in cf["colDensThresholds"]
            ind = np.where(cf["colDensThresholds"] == thresh)[0]

            rvir = sP.units.codeLengthToKpc(sP.groupCatSingle(haloID=0)["Group_R_Crit200"])
            relStr = ""
            xx = cf["radBins%s" % relStr] / rvir
            yy = np.squeeze(cf["all_percs%s" % relStr][ind, :, 3])

            # plot middle line
            label = sP.simName if j == len(colDensThresholds) - 1 else ""
            ax.plot(xx, yy, lw=lw, color=colors[i], linestyle="-", label=label)

        ax.legend(loc="upper right")

    # finish
    fig.savefig("figure_8.pdf")
    plt.close(fig)
