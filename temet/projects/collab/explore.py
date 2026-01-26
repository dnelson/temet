"""
Misc exploration plots and testing, checks for others.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d

from temet.plot import subhalos
from temet.plot.config import figsize, lw
from temet.plot.util import loadColorTable
from temet.util import simParams
from temet.util.helper import closest, running_median
from temet.vis.box import renderBox
from temet.vis.halo import renderSingleHalo


def amyDIGzProfiles():
    """Use some projections to create the SB(em lines) vs z plot."""
    run = "tng"
    res = 2160
    redshift = 0.1
    method = "sphMap"
    nPixels = [100, 100]
    axes = [0, 1]
    rotation = "edge-on"

    size = 30.0
    sizeType = "kpc"

    massBin = [10.00, 10.02]  # log mstar
    maxXDistPkpc = 5.0  # select pixels within 5 kpc of disk center

    lines = ["H-alpha", "H-beta", "O--2-3728.81A", "O--3-5006.84A", "N--2-6583.45A", "S--2-6730.82A"]

    # which halos?
    sP = simParams(res=res, run=run, redshift=redshift)

    gc = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log", "central_flag"])

    with np.errstate(invalid="ignore"):
        w = np.where((gc["mstar_30pkpc_log"] > massBin[0]) & (gc["mstar_30pkpc_log"] < massBin[1]) & gc["central_flag"])
    subInds = w[0]

    print("[%.2f - %.2f] Processing [%d] halos..." % (massBin[0], massBin[1], len(w[0])))

    # start the plot
    figsize = np.array([14, 10]) * 0.9
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel("z [pkpc]")
    ax.set_xlim([0, 10])
    ax.set_ylim([1, 600])
    ax.set_yscale("log")
    ax.set_ylabel("Luminosity Surface Density [ 10$^{36}$ erg s$^{-1}$ kpc$^{-2}$ ]")

    # loop over lines
    for line in lines:
        partField_loc = "sb_%s_lum_kpc" % line  # + '_sf0' to set SFR>0 cells to zero

        x_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")
        z_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float32")
        grid_global = np.zeros((nPixels[0] * nPixels[1], len(subInds)), dtype="float64")

        for j, subhaloInd in enumerate(subInds):  # noqa: B007
            # project
            class plotConfig:
                saveFilename = "dummy"

            panels = [{"partType": "gas", "partField": partField_loc, "valMinMax": [34, 41]}]
            grid, conf = renderSingleHalo(panels, plotConfig, locals(), returnData=True)

            # compute z-distance and x-distance for each pixel
            pxSize = size / nPixels[0]  # pkpc

            xx, yy = np.mgrid[0 : nPixels[0], 0 : nPixels[1]]
            xx = xx.astype("float64") - nPixels[0] / 2  # z-axis, i.e. perpendicular to disk
            yy = yy.astype("float64") - nPixels[1] / 2  # x-axis, i.e. along the major axis

            zdist = np.abs(xx) * pxSize  # symmetric (both above and below the disk)
            xdist = np.abs(yy) * pxSize

            # debug plots
            # from ..plot.util import plot2d
            # plot2d(grid, label='sb [log erg/s/kpc^2]', filename='test_grid.pdf')
            # plot2d(xdist, label='x distance[pkpc]', filename='test_xdist.pdf')
            # plot2d(zdist, label='z distance[pkpc]', filename='test_zdist.pdf')

            # save
            x_global[:, j] = xdist.ravel()
            z_global[:, j] = zdist.ravel()
            grid_global[:, j] = grid.ravel()

        # flatten and select in [x-bounds]
        x_global = x_global.ravel()
        z_global = z_global.ravel()
        grid_global = grid_global.ravel()

        w = np.where(x_global < maxXDistPkpc)

        with np.errstate(invalid="ignore"):
            grid_global = 10.0**grid_global  # remove log

        # bin: median SB as a function of z
        nBins = int(nPixels[0] / 2)
        z_vals, hist, hist_std = running_median(z_global[w], grid_global[w], nBins=nBins)

        hist /= 1e36  # units to match y-axis label

        # plot
        label = conf["label"].split(" Luminosity")[0]
        ax.plot(z_vals, hist, "-", lw=2.5, label=label)

    # finish and save plot
    ax.legend(loc="upper right")
    fig.savefig("sb_vs_z_Mstar=%.1f.pdf" % (massBin[0]))
    plt.close(fig)


def martinSubboxProj3DGrid():
    """Compare (i) 2D histo projection, (ii) 2D sphMap projection, (iii) 3D sphMap grid then projection."""
    run = "tng"
    redshift = 8.0  # subbox snap 126
    variant = "subbox0"
    res = 1080

    axes = [0, 1]  # x,y
    labelZ = False
    labelScale = False
    labelSim = False
    plotHalos = False
    hsmlFac = 2.5  # use for all: gas, dm, stars (for whole box)
    nPixels = [128, 128]

    partType = "gas"
    partField = "coldens_msunkpc2"
    valMinMax = [5.5, 7.3]

    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)

    class plotConfig:
        plotStyle = "open"  # open, edged
        rasterPx = 1000
        colorbars = True
        saveBase = ""

    # (A)
    panels = [{}]
    method = "sphMap"
    plotConfig.saveFilename = "./boxImage_%s_%d_%s_%s.pdf" % (sP.simName, sP.snap, partType, method)

    renderBox(panels, plotConfig, locals())

    # (B)
    panels = [{}]
    method = "histo"
    plotConfig.saveFilename = "./boxImage_%s_%d_%s_%s.pdf" % (sP.simName, sP.snap, partType, method)

    renderBox(panels, plotConfig, locals())

    # (C) load data to compute grids
    pos = sP.snapshotSubset(partType, "pos")
    hsml = sP.snapshotSubset(partType, "cellsize")
    mass = sP.snapshotSubset(partType, "mass")

    # (C) get data grids and compare histograms
    panels = [{}]
    method = "sphMap"
    grid_sphmap, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    panels = [{}]
    method = "histo"
    grid_histo, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    panels = [{}]
    method = "sphMap"
    nPixels = [128, 128, 128]
    grid_sphmap3d, conf = renderBox(panels, plotConfig, locals(), returnData=True)

    sphmap_total = np.sum(10.0**grid_sphmap)
    sphmap3d_total = np.sum(10.0**grid_sphmap3d)
    histo_total = np.sum(10.0**grid_histo)
    frac = np.sum(10.0**grid_sphmap) / np.sum(10.0**grid_histo)
    frac3d = np.sum(10.0**grid_sphmap3d) / np.sum(10.0**grid_histo)

    # start plot
    vmm = [5.0, 8.0]
    nBins = 120

    figsize = np.array([14, 10]) * 0.8
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel(conf["label"])
    ax.set_xlim([5.8, 7.4])
    ax.set_yscale("log")
    ax.set_ylabel("Number of Pixels")
    ax.set_title("frac = %.4f, frac3d = %.4f " % (frac, frac3d))

    # histogram and plot
    yy, xx = np.histogram(grid_sphmap.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5 * (vmm[1] - vmm[0]) / nBins

    ax.plot(xx, yy, "-", drawstyle="steps", label="sphmap [%g]" % sphmap_total)

    yy, xx = np.histogram(grid_sphmap3d.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5 * (vmm[1] - vmm[0]) / nBins

    ax.plot(xx, yy, ":", drawstyle="steps", label="sphmap3d [%g]" % sphmap3d_total)

    yy, xx = np.histogram(grid_histo.ravel(), bins=nBins, range=vmm)
    xx = xx[:-1] + 0.5 * (vmm[1] - vmm[0]) / nBins

    ax.plot(xx, yy, "--", drawstyle="steps", label="histo [%g]" % histo_total)

    # finish and save plot
    ax.legend(loc="upper right")
    fig.savefig("px_comp.pdf")
    plt.close(fig)


def auroraVoyage2050WhitePaper():
    """Create plots for Aurora's ESA Voyage 2050 white paper."""
    from temet.projects.oxygen import stackedRadialProfiles

    redshift = 0.1

    TNG100 = simParams(res=1820, run="tng", redshift=redshift)
    Illustris1 = simParams(res=1820, run="illustris", redshift=redshift)
    Eagle = simParams(res=1504, run="eagle", redshift=redshift)

    if 1:
        # radial profiles of ionic density or emission SB
        sPs = [TNG100, Eagle, Illustris1]
        ions = ["OVII"]  # ,'OVIII']
        cenSatSelect = "cen"
        haloMassBins = [[12.4, 12.6], [11.4, 11.6]]  # [[11.9,12.1], [12.4,12.6]]
        combine2Halo = True
        median = True
        massDensity = False
        fieldTypes = ["FoF"]  # GlobalFoF for final

        # cols (redshift = 0.0)
        # emFlux = False
        # projDim = '3D'

        # fluxes (redshift = 0.1)
        emFlux = True
        projDim = "2Dz_2Mpc"

        simNames = "_".join([sP.simName for sP in sPs])

        for radRelToVirRad in [True, False]:
            saveName = "radprofiles_%s_%s_%s_z%02d_%s_rho%d_rvir%d.pdf" % (
                projDim,
                "-".join(ions),
                simNames,
                redshift,
                cenSatSelect,
                massDensity,
                radRelToVirRad,
            )

            stackedRadialProfiles(
                sPs,
                saveName,
                ions=ions,
                redshift=redshift,
                massDensity=massDensity,
                radRelToVirRad=radRelToVirRad,
                cenSatSelect="cen",
                projDim=projDim,
                haloMassBins=haloMassBins,
                combine2Halo=combine2Halo,
                fieldTypes=fieldTypes,
                emFlux=emFlux,
                median=median,
            )


def smitaXMMproposal():
    """Dependence of OVII on sSFR at fixed mass."""
    sP = simParams(res=1820, run="tng", redshift=0.0)
    # sPs.append( simParams(res=2500, run='tng', redshift=0.0) )

    yQuant = "mass_ovii"  #'ssfr'
    xQuant = "mstar_30pkpc_log"  # 'mhalo_200_log'
    cenSatSelect = "cen"  # ['cen','sat','all']

    cQuant = "delta_sfms"  #'ssfr' #,'mhalo_200_log','mstar_30pkpc_log'] #quantList(wTr=True, wMasses=True)
    clim = None  # [-2.5, -0.5] #None #[10.0,11.0]
    medianLine = True
    cNaNZeroToMin = True
    minCount = 0

    xlim = [9.0, 12.0]  # [11.0, 13.5]
    ylim = [6.0, 9.0]
    qRestrictions = None
    pdf = None

    subhalos.histogram2d(
        sP,
        yQuant=yQuant,
        xQuant=xQuant,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        minCount=minCount,
        qRestrictions=qRestrictions,
        medianLine=medianLine,
        cenSatSelect=cenSatSelect,
        cNaNZeroToMin=cNaNZeroToMin,
        cQuant=cQuant,
        pdf=pdf,
    )


def nachoAngularQuenchingDens():
    """Variation of CGM gas density with azimuthal angle (for Martin Navarro+20)."""
    from temet.catalog.gasflows import radialMassFluxes
    from temet.plot.gasflows import outflowRates2DStackedInMstar

    sP = simParams(run="tng100-1", redshift=0.0)
    # mStarBins = [[9.8,10.2],[10.4,10.6],[10.9,11.1],[11.3,11.7]] # exploration
    mStarBins = [[10.8, 11.2]]  # [[10.5,10.8]] # txt-files/1d plots

    v200norm = False
    rawMass = False
    rawDens = False

    if 0:
        clims = [[-1.8, -1.1], [-1.8, -0.9], [-2.0, -0.9], [-2.0, -0.4]]
        config = {"stat": "mean", "skipZeros": False, "vcutInd": [1, 2, 5, 5]}
    if 0:
        v200norm = True
        clims = [[-1.8, -1.1], [-1.4, -0.8], [-1.4, -0.4], [-1.4, 0.0]]
        config = {"stat": "mean", "skipZeros": False, "vcutInd": [3, 3, 3, 3]}
    if 0:
        rawMass = True
        clims = [[6, 8], [6, 8.5], [6.5, 9], [6.5, 10]]
        config = {"stat": "mean", "skipZeros": False, "vcutInd": [0, 0, 0, 0]}  # only 0 is all mass (no vcut)
    if 1:
        rawDens = True
        # clims  = [[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]
        clims = [[-0.1, 0.1]]
        # clims  = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[-0.1,0.1]]
        config = {"stat": "mean", "skipZeros": False, "vcutInd": [0, 0, 0, 0]}  # only 0 is all mass (no vcut)

    outflowRates2DStackedInMstar(
        sP,
        xAxis="rad",
        yAxis="theta",
        mStarBins=mStarBins,
        clims=clims,
        v200norm=v200norm,
        rawMass=rawMass,
        rawDens=rawDens,
        config=config,
    )

    # 1d plot and txt file output
    mdot, mstar, subids, binConfig, numBins, vcut_vals = radialMassFluxes(
        sP, scope="SubfindWithFuzz", ptType="Gas", thirdQuant="theta", fourthQuant=None, v200norm=False, rawMass=True
    )

    mdot_2d = np.squeeze(mdot[:, :, config["vcutInd"][0], :]).copy()

    # bin selection
    w = np.where((mstar > mStarBins[0][0]) & (mstar <= mStarBins[0][1]))
    mdot_local = np.squeeze(mdot_2d[w, :, :]).copy()

    # relative to azimuthal average in each radial bin: delta_rho/<rho>
    h2d = np.nanmean(mdot_local, axis=0)  # mean
    radial_means = np.nanmean(h2d, axis=1)
    h2d /= radial_means[:, np.newaxis]

    # plot
    radIndsSave = [8, 9, 10, 11]  # up to 13

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_xlabel("Galactocentric Angle [ deg ] [0 = major axis, 90 = minor axis]")
    ax.set_ylabel(r"Gas $\delta \\rho / <\\rho>$ [ linear ]")

    ax.set_ylim([0.75, 1.35])
    ax.set_xlim([0, 360])
    ax.plot(ax.get_xlim(), [1, 1], "-", lw=lw, color="black", alpha=0.5)

    xx = np.rad2deg(binConfig["theta"][:-1] + np.pi)  # [-180,180] -> [0,360]

    for radInd in radIndsSave:
        radMidPoint = 0.5 * (binConfig["rad"][radInd] + binConfig["rad"][radInd + 1])

        yy = h2d[radInd, :]

        ax.plot(xx, yy, "-", lw=lw, label="r = %d kpc" % radMidPoint)

    ax.legend(loc="best")
    fig.savefig("delta_rho_vs_theta_%.1f-%.1f.pdf" % (mStarBins[0][0], mStarBins[0][1]))
    plt.close(fig)

    # write text file
    with open("delta_rho_vs_theta_Mstar_%.1f-%.1f.txt" % (mStarBins[0][0], mStarBins[0][1]), "w") as f:
        f.write("# theta [deg]:\n")
        f.write(" ".join(["%.1f" % angle for angle in xx]))
        f.write("\n")

        for radInd in radIndsSave:
            f.write(
                "# gas delta_rho/<rho> [linear], radial bin [%d-%d kpc]\n"
                % (binConfig["rad"][radInd], binConfig["rad"][radInd + 1])
            )
            f.write(" ".join(["%.3f" % val for val in np.squeeze(h2d[radInd, :])]))
            f.write("\n")


def nachoAngularQuenchingImage():
    """Images of delta rho/rho (for Martin Navarro+20)."""
    from .truong_xrayangular import stackedHaloImage

    conf = 0
    median = True
    rvirUnits = False
    depthFac = 1.0

    sP = simParams(run="tng100-2", redshift=0.0)
    mStarBin = [11.0, 11.05]

    stackedHaloImage(sP, mStarBin, conf=conf, median=median, rvirUnits=rvirUnits, depthFac=depthFac)


def omega_metals_z(metal_mass=True, hih2=False, mstar=False, mstarZ=False, hot=False, higal=False):
    """Compute Omega_Q(z) for various components (Q). Rob Yates paper 2021."""
    from temet.cosmo.hydrogen import neutral_fraction

    sP = simParams(run="eagle")

    snaps = sP.validSnapList(onlyFull=True)
    redshifts = np.zeros(snaps.size, dtype="float32")

    if hih2:
        rho_z_HI = np.zeros(snaps.size, dtype="float32")
        rho_z_H2 = np.zeros(snaps.size, dtype="float32")
    elif mstar:
        dens_threshold = 0.05  # cm^3
        rho_z_allgas = np.zeros(snaps.size, dtype="float32")

        mstar_bins = [[0, 8], [8, 9], [9, 10], [10, 13]]
        rho_z_allgas_mstar = np.zeros((len(mstar_bins), snaps.size), dtype="float32")
    elif mstarZ:
        mstar_bins = [[6, 13], [7, 13], [8, 13], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [0, 13]]
        metal_to_hydrogen_ratio_hi_weighted = np.zeros((len(mstar_bins), snaps.size), dtype="float32")
    elif hot:
        mhalo_bins = [[7, 15], [10, 11], [11, 12], [12, 13], [13, 14], [13, 15]]
        temp_cuts = [5.0, 5.5, 6.0, 6.5]  # log K

        rho_z_hotgas = np.zeros((len(mhalo_bins), snaps.size, len(temp_cuts)), dtype="float32")
    elif higal:
        rho_z_hi_gal = np.zeros(snaps.size, dtype="float32")
        rho_z_hi_70 = np.zeros(snaps.size, dtype="float32")
        rho_z_hi_fof = np.zeros(snaps.size, dtype="float32")
    else:
        dens_cuts = [0.1, 0.05, 0.025, 0.016, 0.004]  # 10^{-1, -1.3, -1.6, -1.8, -2.4} cm^-3
        nh0_cuts = [0.5, 0.1, 0.05, 0.01]

        rho_z_allgas = np.zeros(snaps.size, dtype="float32")
        rho_z_gasdens = np.zeros((len(dens_cuts), snaps.size), dtype="float32")
        rho_z_nh0frac = np.zeros((len(nh0_cuts), snaps.size), dtype="float32")

        rho_z_smbhs = np.zeros(snaps.size, dtype="float32")
        rho_z_stars = np.zeros(snaps.size, dtype="float32")

    for i, snap in enumerate(snaps):
        sP.setSnap(snap)
        print(snap, sP.redshift)
        redshifts[i] = sP.redshift

        if hih2:
            # HI and H2 (Fig 3)
            assert not metal_mass  # makes no sense here

            mass = sP.gas("MHI_GK")  # 10^10/h msun
            rho_z_HI[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

            mass = sP.gas("MH2_GK")  # 10^10/h msun
            rho_z_H2[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

        elif mstar:
            # in stellar mass bins (Fig 5)
            mass = sP.gas("mass")  # 10^10/h msun, total mass
            if metal_mass:
                mass *= sP.gas("metallicity")  # metal mass

            # fiducial ISM cut
            dens = sP.gas("nh")  # 1/cm^3 physical
            w = np.where(dens < dens_threshold)
            mass[w] = 0.0  # skip

            # sum total, and per bin
            rho_z_allgas[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

            parent_mstar = sP.gas("parent_subhalo_mstar_30pkpc_log")

            for j, mstar_bin in enumerate(mstar_bins):
                with np.errstate(invalid="ignore"):
                    w = np.where((parent_mstar >= mstar_bin[0]) & (parent_mstar < mstar_bin[1]))
                rho_z_allgas_mstar[j, i] = np.sum(mass[w], dtype="float64") / sP.HubbleParam  # 10^10 msun

        elif mstarZ:
            # mean metallicity, in stellar mass bins (Eqn 6, Fig 4)
            # note: operating on per-subhalo quantities, unlike all other options, which are per cell
            qRestrict = "nHgt025"  # 'nHgt05' (n>0.05), 'nHgt025' (n>0.025), 'SFgas' (n>0.1)
            HI_field = "Subhalo_Mass_%s_HI" % qRestrict
            metal_field = "Subhalo_Mass_%s_Metal" % qRestrict
            H_field = "Subhalo_Mass_%s_Hydrogen" % qRestrict

            HI_mass = sP.auxCatSplit(HI_field)[HI_field]  # 10^10/h msun, total HI mass, my simple model
            metal_mass = sP.auxCatSplit(metal_field)[metal_field]  # 10^10/h msun, total metal mass
            H_mass = sP.auxCatSplit(H_field)[H_field]  # 10^10/h msun, total H mass

            sub_mstar = sP.subhalos("mstar_30pkpc_log")

            for j, mstar_bin in enumerate(mstar_bins):
                with np.errstate(invalid="ignore"):
                    w = np.where((sub_mstar >= mstar_bin[0]) & (sub_mstar < mstar_bin[1]) & (H_mass > 0))

                avg_MH = np.sum(metal_mass[w] / H_mass[w] * HI_mass[w]) / np.sum(HI_mass[w])
                metal_to_hydrogen_ratio_hi_weighted[j, i] = avg_MH

        elif hot:
            # hot gas (above some temperature threshold), in halo mass bins
            mass = sP.gas("mass")  # 10^10/h msun, total mass
            if metal_mass:
                mass *= sP.gas("metallicity")  # metal mass

            temp = sP.gas("temp_log")
            parent_mhalo = sP.gas("parent_subhalo_mhalo_subfind_log")  # SubhaloMass [log msun]

            for j, mhalo_bin in enumerate(mhalo_bins):
                for k, temp_threshold in enumerate(temp_cuts):
                    with np.errstate(invalid="ignore"):
                        w = np.where(
                            (parent_mhalo >= mhalo_bin[0]) & (parent_mhalo < mhalo_bin[1]) & (temp > temp_threshold)
                        )

                    rho_z_hotgas[j, i, k] = np.sum(mass[w], dtype="float64") / sP.HubbleParam  # 10^10 msun

        elif higal:
            # fraction of total HI mass in the box container within (i) galaxies (<2rhalfstars) and (ii) FoFs
            galField = "Subhalo_Mass_2rstars_MHI_GK"  # 'Subhalo_Mass_2rstars_HI'
            gal70Field = "Subhalo_Mass_70pkpc_MHI_GK"
            fofField = "Subhalo_Mass_FoF_MHI_GK"  # 'Subhalo_Mass_FoF_HI'

            HI_mass_gal = sP.auxCatSplit(galField)[galField]
            HI_mass_70kpc = sP.auxCatSplit(gal70Field)[gal70Field]
            HI_mass_fof = sP.auxCatSplit(fofField)[fofField]

            rho_z_hi_gal[i] = np.nansum(HI_mass_gal) / sP.HubbleParam  # 10^10 msun
            rho_z_hi_70[i] = np.nansum(HI_mass_70kpc) / sP.HubbleParam  # 10^10 msun
            rho_z_hi_fof[i] = np.nansum(HI_mass_fof) / sP.HubbleParam  # 10^10 msun
        else:
            # default (Fig 1)
            # all gas
            mass = sP.gas("mass")  # 10^10/h msun, total mass
            if metal_mass:
                mass *= sP.gas("metallicity")  # metal mass
            rho_z_allgas[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

            # gas density thresholds
            dens = sP.gas("nh")  # 1/cm^3 physical

            for j, dens_cut in enumerate(dens_cuts):
                rho_z_gasdens[j, i] = np.sum(mass[np.where(dens > dens_cut)], dtype="float64")
            rho_z_gasdens[:, i] /= sP.HubbleParam  # 10^10 msun

            # gas neutral fraction thresholds
            if 0:
                nh0frac = sP.gas("NeutralHydrogenAbundance")

                w = np.where(dens > 0.13)  # cm^-3, correct for eEOS for star-forming gas
                nh0frac[w] = neutral_fraction(dens[w], sP)

                for j, nh0_cut in enumerate(nh0_cuts):
                    rho_z_nh0frac[j, i] = np.sum(mass[np.where(nh0frac > nh0_cut)], dtype="float64")
                rho_z_nh0frac[:, i] /= sP.HubbleParam  # 10^10 msun

                # stars
                mass = sP.stars("mass")  # 10^10 msun/h, total mass
                if metal_mass:
                    mass *= sP.stars("metallicity")  # metal mass
                rho_z_stars[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

                # smbhs
                if sP.numPart[sP.ptNum("bhs")]:
                    mass = sP.bhs("mass")  # 10^10 msun/h, total mass
                    if metal_mass:
                        mass *= sP.bhs("metallicity")  # metal mass
                    rho_z_smbhs[i] = np.sum(mass, dtype="float64") / sP.HubbleParam  # 10^10 msun

    # units: [10^10 msun] -> [msun/cMpc^3]
    print(sP.simName)
    print("redshifts = ", redshifts)

    if hih2:
        rho_z_HI *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_H2 *= 1e10 / sP.boxSizeCubicComovingMpc

        print("rho_z_HI = ", rho_z_HI)
        print("rho_z_H2 = ", rho_z_H2)
    elif mstar:
        rho_z_allgas_mstar *= 1e10 / sP.boxSizeCubicComovingMpc

        print("rho_allgas_mstarbins = ", rho_z_allgas_mstar)
        print("mstar_bins = ", mstar_bins)
        print("dens_threshold = ", dens_threshold)
    elif mstarZ:
        print("metal_to_hydrogen_ratio_hi_weighted = ", np.log10(metal_to_hydrogen_ratio_hi_weighted))
        print("mstar_bins = ", mstar_bins)
    elif hot:
        rho_z_hotgas *= 1e10 / sP.boxSizeCubicComovingMpc
        print("mhalo_bins = ", mhalo_bins)
        for k, temp_cut in enumerate(temp_cuts):
            print("temp_cut = ", temp_cut)
            print("rho_z_hotgas = ", rho_z_hotgas[:, :, k])
    elif higal:
        rho_z_hi_gal *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_hi_70 *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_hi_fof *= 1e10 / sP.boxSizeCubicComovingMpc
        print("rho_z_hi_gal = ", rho_z_hi_gal)
        print("rho_z_hi_70 = ", rho_z_hi_70)
        print("rho_z_hi_fof = ", rho_z_hi_fof)
    else:
        rho_z_allgas *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_gasdens *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_nh0frac *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_stars *= 1e10 / sP.boxSizeCubicComovingMpc
        rho_z_smbhs *= 1e10 / sP.boxSizeCubicComovingMpc

        print("rho_allgas = ", rho_z_allgas)
        print("rho_stars = ", rho_z_stars)
        print("rho_gasdens = ", rho_z_gasdens)
        print("rho_nh0frac = ", rho_z_nh0frac)
        print("rho_smbhs = ", rho_z_nh0frac)


def abhijeetMgIISurfDens():
    """Test for Anand+ (2022)."""
    from temet.projects.oxygen import stackedRadialProfiles

    sPs = [simParams(run="tng100-1", redshift=0.5)]

    haloMassBins = [[12.5, 13.0], [13.0, 14.0]]
    fieldTypes = ["30Mpc_GlobalFoF"]  # 30 Mpc is roughly dz=0.01 at z=0.5

    stackedRadialProfiles(
        sPs,
        "mg2_test2.pdf",
        ions=["MgII"],
        redshift=0.5,
        cenSatSelect="cen",
        projDim="2Dz",
        radRelToR500=True,
        massDensityMsun=True,
        haloMassBins=haloMassBins,
        xlim=[1.0, 4.0],
        stellarMassBins=None,
        fieldTypes=fieldTypes,
        combine2Halo=True,
        median=False,
    )


def xenoSNevo_profiles():
    """Xeno idealized SNe runs: density profiles vs time."""
    # config
    # runName = '1_1_cluster_dt_0_cRad_0pc_density_1_nstar_1_E_1e51erg_boxSize_48_res_128'
    # runName = '1_test_8_negative_pressure_set_to_zero'
    runName = "17_IC_SN_base"
    sim = simParams("~/sims.idealized/sims.xeno/%s/" % runName)

    skip = int(sim.numSnaps / 15) if sim.numSnaps > 100 else 5

    # start plot
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(figsize[0] * 3, figsize[1]), ncols=3)
    ax1.set_xlabel("Distance [pc]")
    ax1.set_ylabel("Density [cm$^{-3}$]")

    ax2.set_xlabel("Distance [pc]")
    ax2.set_ylabel("Temperature [log K]")

    ax3.set_xlabel("Distance [pc]")
    ax3.set_ylabel("Radial Velocity [km/s]")

    # loop over snaps
    for i in range(sim.numSnaps)[::skip]:
        sim.setSnap(i)
        sim.refPos = np.array([sim.boxSize / 2, sim.boxSize / 2, sim.boxSize / 2])  # for vrad
        sim.refVel = np.array([0, 0, 0])  # for vrad

        cur_time = sim.time * (sim.units.UnitTime_in_s / sim.units.s_in_kyr)

        # load
        pos = sim.gas("pos")
        dens = sim.gas("dens")
        temp = sim.gas("temp_log")
        vrad = sim.gas("vrad")

        # convert
        boxCen = np.array([0.5 * sim.boxSize, 0.5 * sim.boxSize, 0.5 * sim.boxSize])
        dist = sim.periodicDists(boxCen, pos)
        dens_phys = sim.units.codeDensToPhys(dens, numDens=True, cgs=True)  # 1/cm^3

        # plot: dens
        rr, yy, std, percs = running_median(dist, dens_phys, nBins=80, percs=[15, 50, 84])
        ax1.plot(rr, yy, "-", lw=lw, label="t = %4.1f kyr" % cur_time)

        # plot: temp
        rr, yy, std, percs = running_median(dist, temp, nBins=80, percs=[15, 50, 84])
        ax2.plot(rr, yy, "-", lw=lw)

        # plot: radial velocity
        rr, yy, std, percs = running_median(dist, vrad, nBins=80, percs=[15, 50, 84])
        ax3.plot(rr, yy, "-", lw=lw, label="t = %4.1f kyr" % cur_time)

    # finish plot
    ax1.plot([sim.boxSize / 2, sim.boxSize / 2], ax1.get_ylim(), ":", color="#cccccc", label="Box Boundary")
    ax2.plot([sim.boxSize / 2, sim.boxSize / 2], ax2.get_ylim(), ":", color="#cccccc")
    ax3.plot([sim.boxSize / 2, sim.boxSize / 2], ax3.get_ylim(), ":", color="#cccccc")

    ax1.legend(loc="best")
    ax3.legend(loc="upper right")
    fig.savefig("dens_profiles_vs_time-%s.png" % runName)
    plt.close(fig)


def xenoSNevo_movie(conf=1):
    """Xeno idealized SNe runs: render visualization frames for movie."""
    # config
    runName = "1_1_cluster_dt_0_cRad_0pc_density_1_nstar_1_E_1e51erg_boxSize_48_res_128"
    sim = simParams("~/sims.idealized/sims.xeno/%s/" % runName)

    # movie visualization
    for i, snapNum in enumerate(range(sim.numSnaps)):
        sim.setSnap(snapNum)

        sP = sim  # pass in

        nPixels = 800
        axes = [0, 1]  # x,y
        labelZ = False
        labelScale = False
        labelSim = False
        plotHalos = False

        partType = "gas"
        labelCustom = ["t = %.3f" % sim.time, "t [kyr] = %.1f" % (sim.time * sim.units.UnitTime_in_yr / 1000)]

        if conf == 1:
            panels = [
                {"partField": "coldens", "valMinMax": [20.8, 22.0]},
                {"partField": "temp", "valMinMax": [2.0, 7.0]},
            ]
        if conf == 2:
            pass

        # render config (global)
        class plotConfig:
            plotStyle = "open"
            rasterPx = 800
            colorbars = True
            title = True

            saveFilename = "frame_%s_%03d.png" % ("-".join([p["partField"] for p in panels]), i)

        renderBox(panels, plotConfig, locals())


def arjenMasses5kpc():
    """Explore Mtot_5kpc vs M*_5kpc."""
    # config
    redshift = 0.0  # 0.8
    runs = ["tng100-1", "eagle", "simba", "illustris-1"]

    yQuants = [
        "r80_stars",
        "mstar_5pkpc",
        "mgas_5pkpc",
        "mdm_5pkpc",
        "mstar_mtot_ratio_5pkpc",
        "mstar_mtot_ratio_5pkpc_log",
    ]
    xQuant = "mhalo_200"  #'mtot_5pkpc'
    css = "cen"  # ['cen','sat','all']

    scatterColor = None  # [None, 'mstar30pkpc_mhalo200_ratio'] #quantList(wTr=False, wMasses=True)
    clim = None  # [10.0,11.0]

    xlim = [11.0, 13.5]  # [9.5, 12.0]
    ylim = None  # [8.0, 12.0] # None
    qRestrictions = [["mstar_30pkpc_log", 10.5, 12.0]]

    scatterPoints = True
    drawMedian = True
    markersize = 20.0
    maxPointsPerDex = 500

    # plot
    sPs = [simParams(run=run, redshift=redshift) for run in runs]

    pdf = PdfPages("galaxy_%s_z%.1f_x=%s_%s.pdf" % ("-".join(runs), redshift, xQuant, css))

    for yQuant in yQuants:
        subhalos.median(
            sPs,
            yQuants=[yQuant],
            xQuant=xQuant,
            cenSatSelect=css,
            qRestrictions=qRestrictions,
            xlim=xlim,
            ylim=ylim,
            clim=clim,
            drawMedian=drawMedian,
            markersize=markersize,
            scatterPoints=scatterPoints,
            scatterColor=scatterColor,
            maxPointsPerDex=maxPointsPerDex,
            pdf=pdf,
        )

    pdf.close()


def yenting_vis_sample(redshift=1.0):
    """For the raw TNG-Cluster halos (not in the virtual box), render views of RIZ stars and SFR, to identify rings."""
    from temet.cosmo.zooms import _halo_ids_run

    zoomHaloInds = _halo_ids_run(onlyDone=False)[1:]  # skip first

    rVirFracs = [1.0]
    method = "sphMap"
    nPixels = [600, 600]
    size = 1.0
    sizeType = "arcmin"
    axesUnits = "arcsec"
    labelScale = "physical"
    labelHalo = "mstar,mhalo,sfr"
    # haloMassBin = [13.5, 14.2]

    class plotConfig:
        plotStyle = "open"
        colorbars = True
        fontsize = 30.0
        title = False

    # panel config
    conf1 = {"partType": "stars", "partField": "stellarCompObsFrame-sdss_r-sdss_i-sdss_z"}
    conf2 = {"partType": "gas", "partField": "sfr_msunyrkpc2", "valMinMax": [-6.0, -1.0]}

    axesLists = [[0, 2], [1, 2], [0, 1]]
    # rotations = [ 'edge-on', 'face-on' ]

    # render halos
    for zoomHaloInd in zoomHaloInds:
        # set sP
        sP = simParams(res=13, run="tng_zoom", variant="sf3", redshift=redshift, hInd=zoomHaloInd)

        # subhaloInd is always the most massive
        subhaloInd = 0

        # set panels
        panels = []

        for axesVal in axesLists:
            panels.append({**conf1, "axes": axesVal})
        panels.append({**conf1, "axes": [0, 1], "rotation": "face-on"})

        for axesVal in axesLists:
            panels.append({**conf2, "axes": axesVal})
        panels.append({**conf2, "axes": [0, 1], "rotation": "face-on"})

        plotConfig.saveFilename = "yenting_%s_z=%.1f.pdf" % (sP.simName, redshift)

        renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)
    print("Done.")


def benedetta_vis_sample():
    """For all TNG300-1 centrals at z=1, Mhalo > 5e13, plot stellar RIZ (observed-frame) composites and SFR maps."""
    res = 1820
    redshift = 0.5
    run = "tng"
    rVirFracs = [1.0]
    method = "sphMap_subhalo"
    nPixels = [400, 400]
    size = 100.0
    axes = [0, 1]
    sizeType = "codeUnits"
    partType = "stars"

    class plotConfig:
        plotStyle = "open"
        rasterPx = 1000
        colorbars = True

    # load halos
    haloIDs = [21,22,27,28,32,41,45,46,50,55,58,60,75,76,95,104,107,126,155,157,7324,7328,7331,7332,7334,7337,
               7340,7343,7354,7363,7365,7390,7424,14595,14603,14605,14607,14608,14612,14618,]  # fmt: skip

    sP = simParams(res=res, run=run, redshift=redshift)
    GroupFirstSub = sP.groupCat(fieldsHalos=["GroupFirstSub"])
    subInds = GroupFirstSub[haloIDs]

    for i, subhaloInd in enumerate(subInds[0:1]):  # noqa: B007
        panels = []

        panels.append({"partField": "stellarBandObsFrame-sdss_r", "valMinMax": [18, 28]})
        panels.append(
            {
                "partField": "stellarBandObsFrame-sdss_r",
                "rotation": "face-on",
                "labelScale": "physical",
                "valMinMax": [18, 28],
            }
        )

        # panels.append( {'partField':'stellarCompObsFrame-sdss_g-sdss_r-sdss_i'} )
        # panels.append( {'partField':'stellarCompObsFrame-sdss_g-sdss_r-sdss_i', 'rotation':'face-on'} )

        # panels.append( {'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w'} )
        # panels.append( {'partField':'stellarComp-jwst_f200w-jwst_f115w-jwst_f070w', 'rotation':'face-on'} )

        plotConfig.saveFilename = "benedetta_haloID-%d.pdf" % (haloIDs[i])

        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def erica_tng50_sfrmaps():
    """Render some SFR surface density maps of TNG50 galaxies for Nelson, E.+2021 vs. 3D-HST paper."""
    # select halo
    sP = simParams(run="tng50-1", redshift=1.0)
    mstar = sP.subhalos("mstar_30pkpc_log")
    cen_flag = sP.subhalos("central_flag")

    mstar[cen_flag == 0] = np.nan  # skip secondaries

    # vis
    rVirFracs = [1.0]
    fracsType = "rHalfMassStars"
    method = "histo"  #'sphMap'
    nPixels = [45, 45]  # [600,600]
    axes = [0, 1]
    labelZ = True
    labelScale = "physical"
    labelSim = False
    labelHalo = "mstar,mhalo,sfr,id"
    relCoords = True
    # rotation   = 'edge-on-stars'
    sizeType = "arcsec"
    size = 2.7
    # psf = 0.14" if we want

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 800
        colorbars = True
        fontsize = 22

    # panels
    partType = "gas"

    if 0:
        # single halo, test
        class plotConfig:
            plotStyle = "edged"
            rasterPx = 800
            colorbars = True
            fontsize = 22

        _, subhaloInd = closest(mstar, 10.55)
        panels = [
            {"partField": "coldens_msunkpc2", "valMinMax": [5.0, 8.0]},
            {"partField": "temp_sfcold", "valMinMax": [4.0, 6.2]},
            {"partField": "sfr_halpha", "valMinMax": [35.5, 40.5]},
            {"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [5.0, 9.0]},
        ]

    if 1:
        # gallery
        class plotConfig:
            plotStyle = "edged"
            rasterPx = 600
            colorbars = False
            fontsize = 22

        panels = []
        with np.errstate(invalid="ignore"):
            subhaloInds = np.where((mstar > 10.5) & (mstar <= 11.0))[0]

        for ind in subhaloInds:
            panels.append({"subhaloInd": ind, "partField": "sfr_halpha", "valMinMax": [35.5, 40.5]})

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def bFieldStrengthComparison():
    """Plot histogram of B field magnitude comparing runs etc."""
    sPs = []

    haloID = None  # None for fullbox
    redshift = 0.5
    nBins = 100
    valMinMax = [-7.0, 4.0]

    sPs.append(simParams(res=1820, run="tng", redshift=redshift))
    sPs.append(simParams(res=910, run="tng", redshift=redshift))
    sPs.append(simParams(res=455, run="tng", redshift=redshift))

    # start plot
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)

    hStr = "fullbox" if haloID is None else "halo%d" % haloID
    ax.set_title("z=%.1f %s" % (redshift, hStr))
    ax.set_xlim(valMinMax)
    ax.set_xlabel(r"Magnetic Field Magnitude [ log $\mu$G ]")
    ax.set_ylabel(r"N$_{\rm cells}$ PDF $\int=1$")
    ax.set_yscale("log")

    for sP in sPs:
        # load
        b_mag = sP.snapshotSubset("gas", "bmag", haloID=haloID)
        b_mag *= 1e6  # Gauss to micro-Gauss
        b_mag = np.log10(b_mag)  # log uG

        # add to plot
        yy, xx = np.histogram(b_mag, bins=nBins, density=True, range=valMinMax)
        xx = xx[:-1] + 0.5 * (valMinMax[1] - valMinMax[0]) / nBins

        ax.plot(xx, yy, label=sP.simName)

    # finish plot
    ax.legend(loc="best")

    fig.savefig("bFieldStrengthComparison_%s.pdf" % hStr)
    plt.close(fig)


def depletionVsDynamicalTimescale():
    """Andi Burkert: check depletion vs dynamical timescale.

    t_dep = M_H2/SFR   M_H2 the cold, star-forming gas or take total gas mass instead
    t_dyn = r12 / v_rot  r12 the half mass radius of the gaseous disk, v_rot its characteristic rot. vel
    """
    # config
    figsize = (14, 9)
    sP = simParams(res=1820, run="illustris", redshift=0.0)

    gc = sP.groupCat(
        fieldsHalos=["GroupFirstSub"], fieldsSubhalos=["SubhaloHalfmassRadType", "SubhaloVmax", "SubhaloSFR"]
    )
    ac = sP.auxCat(fields=["Subhalo_Mass_SFingGas", "Subhalo_Mass_30pkpc_Stars"])

    # t_dep [Gyr]
    M_cold = sP.units.codeMassToMsun(ac["Subhalo_Mass_SFingGas"])
    SFR = gc["subhalos"]["SubhaloSFR"]  # Msun/yr
    t_dep = M_cold / SFR / 1e9

    # t_dyn [Gyr]
    r12 = sP.units.codeLengthToKpc(gc["subhalos"]["SubhaloHalfmassRadType"][:, sP.ptNum("stars")])
    v_rot = gc["subhalos"]["SubhaloVmax"] * sP.units.kmS_in_kpcGyr
    t_dyn = r12 / v_rot

    # stellar masses and central selection
    m_star = sP.units.codeMassToLogMsun(ac["Subhalo_Mass_30pkpc_Stars"])

    w_central = np.where(gc["halos"] >= 0)

    centralsMask = np.zeros(gc["subhalos"]["count"], dtype=np.int16)
    centralsMask[gc["halos"][w_central]] = 1

    centrals = np.where(centralsMask & (SFR > 0.0) & (r12 > 0.0))

    t_dep = t_dep[centrals]
    t_dyn = t_dyn[centrals]
    m_star = m_star[centrals]

    # plot config
    title = sP.simName + " z=%.1f" % sP.redshift + " [only centrals with SFR>0 and r12>0]"
    tDynMinMax = [0, 0.2]
    tDepMinMax = [0, 4]
    mStarMinMax = [9.0, 12.0]
    ratioMinMax = [0, 0.05]  # tdyn/tdep
    nBinsX = 200
    nBinsY = 150
    binSizeMed = 0.01

    # (A) 2d histogram of t_dep vs. t_dyn for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(tDynMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel("t$_{\\rm dyn}$ [Gyr]")
        ax.set_ylabel("t$_{\\rm dep}$ [Gyr]")

        # 2d histo
        zz, xc, yc = np.histogram2d(t_dyn, t_dep, bins=[nBinsX, nBinsY], range=[tDynMinMax, tDepMinMax], density=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable("viridis")
        plt.imshow(
            zz,
            extent=[tDynMinMax[0], tDynMinMax[1], tDepMinMax[0], tDepMinMax[1]],
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )

        # median
        # xm, ym, sm = running_median(t_dyn,t_dep,binSize=binSizeMed)
        # ym2 = savgol_filter(ym,3,2)
        # sm2 = savgol_filter(sm,3,2)
        # ax.plot(xm[:-1], ym2[:-1], '-', color='black', lw=2.0)
        # ax.plot(xm[:-1], ym2[:-1]+sm2[:-1], ':', color='black', lw=2.0)
        # ax.plot(xm[:-1], ym2[:-1]-sm2[:-1], ':', color='black', lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel("Number of Galaxies [ log ]")

        fig.savefig("tdyn_vs_tdep_%s_a.pdf" % sP.simName)
        plt.close(fig)

    # (B) 2d histogram of ratio (t_dep/t_dyn) vs. m_star for all centrals
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(ratioMinMax)
        ax.set_xlabel(r"M$_{\rm star}$ [ log M$_\odot$ ]")
        ax.set_ylabel(r"t$_{\rm dyn}$ / t$_{\rm dep}$")

        # 2d histo
        zz, xc, yc = np.histogram2d(
            m_star, t_dyn / t_dep, bins=[nBinsX, nBinsY], range=[mStarMinMax, ratioMinMax], density=True
        )
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable("viridis")
        plt.imshow(
            zz,
            extent=[mStarMinMax[0], mStarMinMax[1], ratioMinMax[0], ratioMinMax[1]],
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )

        # median
        xm, ym, sm = running_median(m_star, t_dyn / t_dep, binSize=binSizeMed * 10)
        ym2 = savgol_filter(ym, 3, 2)
        sm2 = savgol_filter(sm, 3, 2)
        ax.plot(xm[:-3], ym2[:-3], "-", color="black", lw=2.0)
        ax.plot(xm[:-3], ym2[:-3] + sm2[:-3], ":", color="black", lw=2.0)
        ax.plot(xm[:-3], ym2[:-3] - sm2[:-3], ":", color="black", lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel("Number of Galaxies [ log ]")

        fig.savefig("tdyn_vs_tdep_%s_b.pdf" % sP.simName)
        plt.close(fig)

    # (C) t_dep vs m_star
    if 1:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

        ax.set_title(title)
        ax.set_xlim(mStarMinMax)
        ax.set_ylim(tDepMinMax)
        ax.set_xlabel(r"M$_{\rm star}$ [ log M$_\odot$ ]")
        ax.set_ylabel(r"t$_{\rm dep}$ [ Gyr ]")

        # 2d histo
        zz, xc, yc = np.histogram2d(m_star, t_dep, bins=[nBinsX, nBinsY], range=[mStarMinMax, tDepMinMax], density=True)
        zz = np.transpose(zz)
        zz = np.log10(zz)

        cmap = loadColorTable("viridis")
        plt.imshow(
            zz,
            extent=[mStarMinMax[0], mStarMinMax[1], tDepMinMax[0], tDepMinMax[1]],
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
            aspect="auto",
        )

        # median
        xm, ym, sm = running_median(m_star, t_dep, binSize=binSizeMed * 10)
        ym2 = savgol_filter(ym, 3, 2)
        sm2 = savgol_filter(sm, 3, 2)
        ax.plot(xm[:-3], ym2[:-3], "-", color="black", lw=2.0)
        ax.plot(xm[:-3], ym2[:-3] + sm2[:-3], ":", color="black", lw=2.0)
        ax.plot(xm[:-3], ym2[:-3] - sm2[:-3], ":", color="black", lw=2.0)

        # colorbar and save
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel("Number of Galaxies [ log ]")

        fig.savefig("tdyn_vs_tdep_%s_c.pdf" % sP.simName)
        plt.close(fig)


def sanch_ovi_groups():
    """Mock OVI absorption spectra around TNG50-1 z=0.1 groups for Sanch Borthakur."""
    from temet.spectra.spectrum import generate_rays_voronoi_fullbox, integrate_along_saved_rays

    sim = simParams(run="tng50-1", redshift=0.1)

    subhaloIDs = [188893, 199226, 208563, 219842, 231369, 239843, 247945, 264620, 277688, 277688, 288932, 305020]
    nRaysPerDim = 300
    raysType = "sample_localized"
    pSplit = [0, len(subhaloIDs)]

    # load
    coldens = integrate_along_saved_rays(
        sim, "O VI numdens", nRaysPerDim=300, raysType="sample_localized", subhaloIDs=subhaloIDs, pSplit=pSplit
    )

    rays_off, rays_len, rays_dl, rays_inds, cell_inds, ray_pos, ray_dir, total_dl = generate_rays_voronoi_fullbox(
        sim, nRaysPerDim=nRaysPerDim, raysType=raysType, subhaloIDs=subhaloIDs, pSplit=pSplit
    )

    # metadata
    subhaloID = subhaloIDs[pSplit[0]]
    subhalo = sim.subhalo(subhaloID)
    halo = sim.halo(subhalo["SubhaloGrNr"])
    pos = subhalo["SubhaloPos"]
    r200 = halo["Group_R_Crit200"]

    # bin
    nbins = 300
    mm = 2.1

    x = (ray_pos[:, 0] - pos[0]) / r200
    y = (ray_pos[:, 1] - pos[1]) / r200

    h2d, _, _, _ = binned_statistic_2d(
        x, y, coldens, statistic="mean", bins=[nbins, nbins], range=[[-mm, mm], [-mm, mm]]
    )

    h2d = h2d.T
    h2d = np.log10(h2d)

    # plot
    fig, ax = plt.subplots()
    im = ax.imshow(h2d, extent=[-mm, mm, -mm, mm], origin="lower", interpolation="none", aspect="equal")

    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.1)
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_ylabel("OVI Column Density [cm$^{-2}$]")

    fig.savefig("test_%s_%d_OVI_groups_%d.pdf" % (sim.name, sim.snap, pSplit[0]))
    plt.close(fig)
