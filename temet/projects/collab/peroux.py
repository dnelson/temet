"""
Celine Peroux related.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from temet.plot import snapshot
from temet.plot.config import sKn, sKo
from temet.util import simParams
from temet.util.helper import logZeroNaN, running_median
from temet.util.rotation import momentOfInertiaTensor, rotationMatricesFromInertiaTensor
from temet.vis.halo import renderSingleHalo


def writeH2CDDFBand():
    """Use H2 CDDFs with many variations (TNG100) to derive an envelope band, f(N_H2) vs. N_H2, and write a text file.

    https://arxiv.org/abs/1909.08624 (Figures 6, 7, 9).
    """
    sP = simParams(res=1820, run="tng", redshift=0.2)  # z=0.2, z=0.8

    vars_sfr = ["nH2_GK_depth10", "nH2_GK_depth10_allSFRgt0", "nH2_GK_depth10_onlySFRgt0"]
    vars_model = ["nH2_BR_depth10", "nH2_KMT_depth10"]
    # vars_diemer = ['nH2_GD14_depth10','nH2_GK11_depth10','nH2_K13_depth10','nH2_S14_depth10']
    vars_cellsize = ["nH2_GK_depth10_cell3", "nH2_GK_depth10_cell1"]
    vars_depth = ["nH2_GK_depth5", "nH2_GK_depth20", "nH2_GK_depth1"]

    speciesList = vars_sfr + vars_model + vars_cellsize + vars_depth
    speciesList = ["nH2_GK_depth10"]  # TNG300 test

    # load
    for i, species in enumerate(speciesList):
        ac = sP.auxCat(fields=["Box_CDDF_" + species])

        n_species = logZeroNaN(ac["Box_CDDF_" + species][0, :])
        fN_species = logZeroNaN(ac["Box_CDDF_" + species][1, :])

        if i == 0:
            # save x-axis on first iter
            N_H2 = n_species.copy()
            fN_H2_low = fN_species.copy()
            fN_H2_high = fN_species.copy()
            fN_H2_low.fill(np.nan)
            fN_H2_high.fill(np.nan)
        else:
            # x-axes must match
            assert np.array_equal(N_H2, n_species)

        # take envelope
        fN_H2_low = np.nanmin(np.vstack((fN_H2_low, fN_species)), axis=0)
        fN_H2_high = np.nanmax(np.vstack((fN_H2_high, fN_species)), axis=0)

    # select reasonable range
    w = np.where(N_H2 >= 15.0)
    N_H2 = N_H2[w]
    fN_H2_low = savgol_filter(fN_H2_low[w], sKn, sKo)
    fN_H2_high = savgol_filter(fN_H2_high[w], sKn, sKo)

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel(r"N$_{\rm H2}$ [cm$^{-2}$]")
    ax.set_ylabel(r"log f(N$_{\rm H2}$) [cm$^{2}$]")
    ax.set_xlim([14, 24])
    ax.set_ylim([-30, -14])
    ax.fill_between(N_H2, fN_H2_low, fN_H2_high, alpha=0.8)

    fig.savefig("h2_CDDF_%s_band-%d.pdf" % (sP.simName, len(speciesList)))
    plt.close(fig)

    # write text file
    filename = "h2_CDDF_%s_band-%d_z=%.1f.txt" % (sP.simName, len(speciesList), sP.redshift)
    out = "# %s z=%.1f\n# N_H2 [cm^-2], f_N,lower [cm^2], f_N,upper [cm^2]\n" % (sP.simName, sP.redshift)

    for i in range(N_H2.size):
        out += "%.3f %.3f %.3f\n" % (N_H2[i], fN_H2_low[i], fN_H2_high[i])
    with open(filename, "w") as f:
        f.write(out)


def galaxyImageH2():
    """Metallicity distribution in CGM image: Klitsch+ (2019) https://arxiv.org/abs/1909.08624 (Figure 8)."""
    run = "tng"
    res = 2160
    redshift = 0.5
    rVirFracs = [0.5, 1.0]  # None
    method = "sphMap_global"
    axes = [0, 1]
    labelSim = False
    relCoords = True
    rotation = "edge-on"
    sizeType = "kpc"
    rVirFracs = None

    size = 40
    subhaloInd = 564218  # for Klitsch+ (2019) paper

    faceOnOptions = {
        "rotation": "face-on",
        "labelScale": "physical",
        "labelHalo": "mstar,sfr",
        "labelZ": True,
        "nPixels": [800, 800],
    }

    edgeOnOptions = {
        "rotation": "edge-on",
        "labelScale": False,
        "labelHalo": False,
        "labelZ": False,
        "nPixels": [800, 250],
    }

    # which halo?
    sP = simParams(res=res, run=run, redshift=redshift)
    haloID = sP.groupCatSingle(subhaloID=subhaloInd)["SubhaloGrNr"]

    panels = []
    panels.append({"partType": "gas", "partField": "MH2_GK", "valMinMax": [17.5, 22.0], **faceOnOptions})
    panels.append({"partType": "gas", "partField": "MH2_GK", "valMinMax": [17.5, 22.0], **edgeOnOptions})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = faceOnOptions["nPixels"][0]
        colorbars = True
        nCols = 1
        nRows = 2
        fontsize = 24
        saveFilename = "./%s.%d.%d.%dkpc.pdf" % (sP.simName, sP.snap, subhaloInd, size)

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def galaxyImage():
    """MgII emission image for Celine proposal 2022."""
    run = "tng"
    res = 2160
    redshift = 0.5
    rVirFracs = None
    method = "sphMap"
    axes = [0, 1]
    labelSim = False
    labelScale = "physical"
    labelZ = True
    relCoords = True
    # rotation   = 'edge-on'
    rVirFracs = None

    sizeType = "arcsec"
    size = 7.5

    subhaloInd = 564218  # for Klitsch+ (2019) paper

    nPixels = [800, 800]

    # which halo?
    sP = simParams(res=res, run=run, redshift=redshift)
    # haloID = sP.groupCatSingle(subhaloID=subhaloInd)['SubhaloGrNr']

    panels = []
    panels.append({"partType": "gas", "partField": "MH2_GK", "valMinMax": [17.5, 22.0]})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = nPixels[0]
        colorbars = True
        # fontsize     = 24
        saveFilename = "./%s.%d.%d.%dkpc.pdf" % (sP.simName, sP.snap, subhaloInd, size)

    # render
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def radialProfilesHIH2():
    """Compute stacked radial profiles of N_HI(b) and N_H2(b). https://arxiv.org/abs/2011.01935 (Figure 1)."""
    sPs = []
    sPs.append(simParams(res=1820, run="tng", redshift=2.0))

    # select subhalos
    mhalo = sPs[0].groupCat(fieldsSubhalos=["mhalo_200_log"])

    with np.errstate(invalid="ignore"):
        ww = np.where((mhalo > 11.8) & (mhalo < 11.9))

    subhaloIDs = [{"11.8 < M$_{\\rm halo}$ < 11.9": ww[0]}]

    # select properties
    fields = ["MHI_GK", "MH2_GK"]
    weighting = None
    op = "sum"
    proj2D = [2, None]  # z-axis, no depth restriction

    for field in fields:
        snapshot.profilesStacked1d(
            sPs, subhaloIDs=subhaloIDs, ptType="gas", ptProperty=field, op=op, weighting=weighting, proj2D=proj2D
        )


def numdensHIVsColumn():
    """Re-create Rahmati+ (2013) Fig 2. https://arxiv.org/abs/2011.01935 (Figure 1)."""
    sP = simParams(run="tng100-1", redshift=3.0)

    N_HI = sP.snapshotSubset("gas", "hi_column")
    M_HI = sP.snapshotSubset("gas", "MHI_GK")
    M_H2 = sP.snapshotSubset("gas", "MH2_GK")
    M_H = sP.snapshotSubset("gas", "mass") * sP.units.hydrogen_massfrac

    w = np.where(np.isfinite(N_HI))  # in grid slice

    N_HI = N_HI[w]
    M_HI = M_HI[w]
    M_H2 = M_H2[w]
    M_H = M_H[w]

    # compute num dens
    vol = sP.snapshotSubsetP("gas", "volume")[w]

    numdens_HI = sP.units.codeDensToPhys(M_HI / vol, cgs=True, numDens=True)
    numdens_H2 = sP.units.codeDensToPhys(M_H2 / vol, cgs=True, numDens=True)
    numdens_H = sP.units.codeDensToPhys(M_H / vol, cgs=True, numDens=True)

    # zero densities where n_HI == 0 (although done in running_median if we are weighting by n_HI)
    w = np.where(numdens_HI == 0)
    numdens_HI[w] = np.nan
    numdens_H2[w] = np.nan
    numdens_H[w] = np.nan

    # restrict to interesting column range
    w = np.where(N_HI > 15.0)
    N_HI = N_HI[w]
    numdens_HI = numdens_HI[w]
    numdens_H2 = numdens_H2[w]
    numdens_H = numdens_H[w]

    # median (or mean weighted by n_HI)
    nBins = 90
    percs = [16, 50, 84]

    N_HI_vals, _, _, nhi_percs = running_median(
        N_HI, numdens_HI, nBins=nBins, percs=percs, mean=True, weights=numdens_HI
    )
    N_HI_vals2, _, _, nh2_percs = running_median(
        N_HI, numdens_H2, nBins=nBins, percs=percs, mean=True, weights=numdens_HI
    )
    N_HI_vals3, _, _, nh_percs = running_median(
        N_HI, numdens_H, nBins=nBins, percs=percs, mean=True, weights=numdens_HI
    )

    assert np.array_equal(N_HI_vals, N_HI_vals2) and np.array_equal(N_HI_vals, N_HI_vals3)

    nhi_percs = logZeroNaN(nhi_percs)
    nh2_percs = logZeroNaN(nh2_percs)
    nh_percs = logZeroNaN(nh_percs)

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel(r"N$_{\rm HI}$ [log cm$^{-2}$]")
    ax.set_ylabel(r"n$_{\rm H}$ or n$_{\rm HI}$ or n$_{\rm H2}$ [log cm$^{-3}$]")
    ax.set_xlim([15, 23])
    ax.set_ylim([-6.0, 2.0])

    (l,) = ax.plot(N_HI_vals, nh_percs[1, :], "-", label=r"n$_{\rm H}$")
    ax.fill_between(N_HI_vals, nh_percs[0, :], nh_percs[-1, :], alpha=0.5, color=l.get_color())

    (l,) = ax.plot(N_HI_vals, nhi_percs[1, :], "-", label=r"n$_{\rm HI}$")
    ax.fill_between(N_HI_vals, nhi_percs[0, :], nhi_percs[-1, :], alpha=0.5, color=l.get_color())

    ax.plot(N_HI_vals, nh2_percs[1, :], "-", label=r"n$_{\rm H2}$")
    ax.fill_between(N_HI_vals, nh2_percs[0, :], nh2_percs[-1, :], alpha=0.5, color=l.get_color())

    ax.legend()
    fig.savefig("N_HI_vs_n_H_HI_H2_%s.pdf" % sP.simName)
    plt.close(fig)

    # write text file
    filename = "N_HI_vs_n_H_HI_H2_%s_z=%.1f.txt" % (sP.simName, sP.redshift)
    out = "# %s z=%.1f\n (all values in log10)" % (sP.simName, sP.redshift)
    out += "# N_HI [cm^-2], "
    out += "n_H [cm^-3], n_H_p16 [cm^-3] n_H_p84 [cm^-3], "
    out += "n_HI [cm^-3], n_HI_p16 [cm^-3] n_HI_p84 [cm^-3], "
    out += "n_H2 [cm^-3], n_H2_p16 [cm^-3] n_H2_p84 [cm^-3]\n"

    for i in range(N_HI_vals.size):
        out += "%6.3f " % N_HI_vals[i]
        out += "%7.3f %7.3f %7.3f " % (nh_percs[1, i], nh_percs[0, i], nh_percs[2, i])
        out += "%7.3f %7.3f %7.3f " % (nhi_percs[1, i], nhi_percs[0, i], nhi_percs[2, i])
        out += "%7.3f %7.3f %7.3f\n" % (nh2_percs[1, i], nh2_percs[0, i], nh2_percs[2, i])
    with open(filename, "w") as f:
        f.write(out)


def galaxyRotationAngles():
    """For Daniela filaments project."""
    sim = simParams("tng50-1", redshift=2.0)
    mstar = sim.subhalos("mstar_30kpc_log")

    subhaloIDs = np.where(mstar > 7.0)[0]

    # allocate and load
    N = subhaloIDs.size
    angles = np.zeros((N, 3), dtype="float32")
    angles.fill(np.nan)

    SubhaloLenType = sim.subhalos("SubhaloLenType")

    # loop over all subhalos
    for i, subhaloID in enumerate(subhaloIDs):
        if i % 100 == 0:
            print(i, N, flush=True)

        if SubhaloLenType[subhaloID, sim.ptNum("gas")] == 0 and SubhaloLenType[subhaloID, sim.ptNum("stars")] == 0:
            continue

        # calculate rotation matrix
        I = momentOfInertiaTensor(sim, subhaloID=subhaloID)
        rots = rotationMatricesFromInertiaTensor(I)
        R = rots["face-on"]

        # calculate angle
        angles[i, 0] = R[1, 2] - R[2, 1]
        angles[i, 1] = R[2, 0] - R[0, 2]
        angles[i, 2] = R[0, 1] - R[1, 0]

    # save
    with h5py.File("angles_%s_%d.hdf5" % (sim.simName, sim.snap), "w") as f:
        f["angles"] = angles
        f["subhaloIDs"] = subhaloIDs

    print("Saved [angles_%s_%d.hdf5]." % (sim.simName, sim.snap))
