"""
Annalisa Pillepich related.
"""

from os import path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from temet.cosmo.util import crossMatchSubhalosBetweenRuns
from temet.plot.config import colors, linestyles, sKn, sKo
from temet.util import simParams
from temet.util.helper import running_median
from temet.vis.common import savePathDefault
from temet.vis.halo import renderSingleHalo


def stellarMergerContribution(sP):
    """Analysis routine for TNG flagship paper on stellar mass content (https://arxiv.org/abs/1707.03406 Figs 12,13)."""
    # config
    haloMassBins = [[11.4, 11.6], [11.9, 12.1], [12.4, 12.6], [12.9, 13.1], [13.4, 13.6], [13.9, 14.1], [14.3, 14.7]]
    rad_pkpc = 30.0
    nHistBins = 50
    histMinMax = [8.0, 12.0]  # log msun
    threshVals = [0.1, 0.5, 0.9]
    minHaloMassIndiv = 12.0  # log msun
    pt = sP.ptNum("stars")

    # check if we saved some results already?
    r = {}
    saveFilename = "stellarMergerData_%s_%d.hdf5" % (sP.simName, sP.snap)

    if path.isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # no, start new calculation: load stellar assembly cat
    fName = sP.postPath + "StellarAssembly/stars_%03d_supp.hdf5" % sP.snap

    with h5py.File(fName, "r") as f:
        InSitu = f["InSitu"][()]
        MergerMass = f["MergerMass"][()]

    MergerMass = sP.units.codeMassToLogMsun(MergerMass)

    # load stellar particle data
    stars = sP.snapshotSubset("stars", ["mass", "pos", "sftime"])
    assert stars["Masses"].shape == InSitu.shape

    # load groupcat
    radSqMax = sP.units.physicalKpcToCodeLength(rad_pkpc) ** 2
    gc = sP.groupCat(fieldsSubhalos=["SubhaloPos", "SubhaloLenType", "SubhaloMass", "SubhaloGrNr"])
    halos = sP.groupCat(fieldsHalos=["Group_M_Crit200"])
    gc["SubhaloOffsetType"] = sP.groupCatOffsetListIntoSnap()["snapOffsetsSubhalo"]

    gc["SubhaloMass"] = sP.units.codeMassToLogMsun(gc["SubhaloMass"])
    halo_masses = sP.units.codeMassToLogMsun(halos)
    halo_masses = halo_masses[gc["SubhaloGrNr"]]  # re-index to subhalos

    # process centrals only
    sat_inds = sP.cenSatSubhaloIndices(cenSatSelect="sat")
    halo_masses[sat_inds] = np.nan

    # allocate returns
    binSizeHalf = (histMinMax[1] - histMinMax[0]) / nHistBins / 2

    r["totalStarMass"] = np.zeros(len(haloMassBins), dtype="float32")
    r["mergerMassHisto"] = np.zeros((len(haloMassBins), nHistBins), dtype="float32")

    # (A) in halo mass bins
    for i, haloMassBin in enumerate(haloMassBins):
        w_sub = np.where((halo_masses >= haloMassBin[0]) & (halo_masses < haloMassBin[1]))

        if len(w_sub[0]) == 0:
            continue  # empty mass bin for this simulation

        print(haloMassBin, len(w_sub[0]))

        for subhaloID in w_sub[0]:
            # get local indices
            i0 = gc["SubhaloOffsetType"][subhaloID, pt]
            i1 = i0 + gc["SubhaloLenType"][subhaloID, pt]

            if i1 == i0:
                continue  # zero length of this type

            # radial restrict
            rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], stars["Coordinates"][i0:i1, :])

            w_valid = np.where(
                (stars["GFM_StellarFormationTime"][i0:i1] >= 0.0) & (rr <= radSqMax) & (InSitu[i0:i1] == 0)
            )

            if len(w_valid[0]) == 0:
                continue  # zero stars

            # local properties
            loc_masses = stars["Masses"][i0:i1][w_valid]
            loc_mergermass = MergerMass[i0:i1][w_valid]

            # histogram and save
            loc_hist, hist_bins = np.histogram(loc_mergermass, bins=nHistBins, range=histMinMax, weights=loc_masses)

            r["totalStarMass"][i] += loc_masses.sum()
            r["mergerMassHisto"][i, :] += loc_hist

    # (B) individual halos: intersection with threshold values
    w_sub = np.where(halo_masses >= minHaloMassIndiv)

    nApertures = 4  # <10, <30, >100, all subhalo particles

    r["indivSubhaloIDs"] = w_sub[0]
    r["indivHaloMasses"] = halo_masses[w_sub]
    r["indivHisto"] = np.zeros((len(w_sub[0]), nApertures, len(threshVals)), dtype="float32")
    r["indivHisto"].fill(np.nan)  # nan value indicates not filled

    print("Processing individuals: ", len(r["indivSubhaloIDs"]))

    for i, subhaloID in enumerate(r["indivSubhaloIDs"]):
        # get local indices
        if i % 100 == 0:
            print(i)
        i0 = gc["SubhaloOffsetType"][subhaloID, pt]
        i1 = i0 + gc["SubhaloLenType"][subhaloID, pt]

        if i1 == i0:
            continue  # zero length of this type

        # radial restrict
        rr = sP.periodicDistsSq(gc["SubhaloPos"][subhaloID, :], stars["Coordinates"][i0:i1, :])

        for apertureIter in range(nApertures):
            # aperture selections
            if apertureIter == 0:
                # < 10 pkpc
                w_valid = np.where(
                    (stars["GFM_StellarFormationTime"][i0:i1] >= 0.0) & (rr <= 10.0) & (InSitu[i0:i1] == 0)
                )
            if apertureIter == 1:
                # < 30 pkpc
                w_valid = np.where(
                    (stars["GFM_StellarFormationTime"][i0:i1] >= 0.0) & (rr <= 30.0) & (InSitu[i0:i1] == 0)
                )
            if apertureIter == 2:
                # > 100 pkpc
                w_valid = np.where(
                    (stars["GFM_StellarFormationTime"][i0:i1] >= 0.0) & (rr >= 100.0) & (InSitu[i0:i1] == 0)
                )
            if apertureIter == 3:
                # all subhalo particles
                w_valid = np.where((stars["GFM_StellarFormationTime"][i0:i1] >= 0.0) & (InSitu[i0:i1] == 0))

            if len(w_valid[0]) == 0:
                continue  # zero stars

            # local properties
            loc_masses = stars["Masses"][i0:i1][w_valid]
            loc_mergermass = MergerMass[i0:i1][w_valid]

            # histogram
            loc_hist, hist_bins = np.histogram(loc_mergermass, bins=nHistBins, range=histMinMax, weights=loc_masses)

            # normalized cumulative sum
            yy = loc_hist / loc_hist.sum()
            yy_cum = yy[::-1].cumsum()[::-1]  # above a given subhalo mass threshold

            # loop over thresholds, find intersections
            for threshIter, thresh in enumerate(threshVals):
                mass_ind = np.where(yy_cum >= thresh)[0]
                if len(mass_ind) == 0:
                    continue  # never crosses threshold? i.e. no ex-situ stars, no stars, ...

                mass_val = hist_bins[mass_ind.max()] + binSizeHalf

                r["indivHisto"][i, apertureIter, threshIter] = mass_val

    # some extra info to save
    r["hist_bins"] = hist_bins[:-1] + binSizeHalf
    r["haloMassBins"] = haloMassBins
    r["histMinMax"] = histMinMax
    r["threshVals"] = threshVals

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]
    print("Saved: [%s]." % saveFilename)

    return r


def stellarMergerContributionPlot():
    """Driver for Pillepich+2018 (https://arxiv.org/abs/1707.03406 Figs 12,13)."""
    sPs = []

    sPs.append(simParams(res=1820, run="tng", redshift=0.0))
    sPs.append(simParams(res=2500, run="tng", redshift=0.0))

    haloBinSize = 0.1
    apertureNames = ["< 10 kpc", "< 30 kpc", "> 100 kpc", "central + ICL"]
    nApertures = 4

    # load
    md = {}
    for sP in sPs:
        md[sP.simName] = stellarMergerContribution(sP)

    histMinMax = md[sPs[0].simName]["histMinMax"]
    hist_bins = md[sPs[0].simName]["hist_bins"]

    # dump text files
    for _i, sP in enumerate(sPs):
        save = []
        save.append(hist_bins)
        for j in range(len(md[sP.simName]["haloMassBins"])):
            yy = md[sP.simName]["mergerMassHisto"][j, :]
            yy /= md[sP.simName]["mergerMassHisto"][j, :].sum()
            yy_cum = yy[::-1].cumsum()[::-1]
            save.append(yy_cum)

        massBinStrs = ["halo_%.1f_%.1f" % (mb[0], mb[1]) for mb in md[sP.simName]["haloMassBins"]]
        header = "subhalo_mass %s" % " ".join(massBinStrs)
        np.savetxt("top_panel_%s.txt" % sP.simName, np.array(save).T, fmt="%.5f", header=header)

        save = []
        for apertureIter in range(nApertures):
            for threshIter in range(3):
                x_vals = md[sP.simName]["indivHaloMasses"]
                y_vals = md[sP.simName]["indivHisto"][:, apertureIter, threshIter]

                xm, ym, _, pm = running_median(x_vals, y_vals, binSize=haloBinSize, percs=[16, 84])
                ym = savgol_filter(ym, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)

                data = np.zeros((4, len(xm)), dtype=xm.dtype)
                data[0, :] = xm
                data[1, :] = ym
                data[2, :] = pm[0, :]
                data[3, :] = pm[1, :]

                filename = "bottom_%s_aperture=%d_thresh=%.1f.txt" % (
                    sP.simName,
                    apertureIter,
                    md[sPs[0].simName]["threshVals"][threshIter],
                )
                header = "%s\nhalo_m200crit median lower16 upper84" % apertureNames[apertureIter]
                np.savetxt(filename, data.T, fmt="%.5f", header=header)

    # plot
    fig = plt.figure(figsize=(10, 16))

    # top panel
    ax = fig.add_subplot(211)
    ax.set_xlabel(r"M$_{\rm stars,progenitor}$ [ log M$_{\rm sun}$ ]")
    ax.set_ylabel("Cumulative Ex-Situ Mass Frac [>= Mass]")
    ax.set_xlim(histMinMax)
    ax.set_ylim([0.0, 1.0])

    ax.plot(histMinMax, [0.1, 0.1], ":", color="black", alpha=0.05)
    ax.plot(histMinMax, [0.5, 0.5], ":", color="black", alpha=0.05)
    ax.plot(histMinMax, [0.9, 0.9], ":", color="black", alpha=0.05)

    for j in range(len(md[sP.simName]["haloMassBins"])):
        for i, sP in enumerate(sPs):
            alpha = 1.0
            if (i == 0 and j > 3) or (i == 1 and j < 2):
                alpha = 0.0  # skip lowest two bins for TNG300, highest 3 bins for TNG100

            yy = md[sP.simName]["mergerMassHisto"][j, :]
            # if mergermass==nan (originally -1), then we will not sum all
            # the weights, such that mergerMassHisto[j,:].sum() <= totalStarMass[j]
            yy /= md[sP.simName]["mergerMassHisto"][j, :].sum()

            yy_cum = yy[::-1].cumsum()[::-1]  # above a given subhalo mass threshold
            # label = '%.1f < M$_{\\rm halo}$ < %.1f' % (haloMassBin[0],haloMassBin[1])

            ax.plot(hist_bins, yy_cum, linestyles[i], color=colors[j], label="", alpha=alpha)

    # legend
    sExtra = []
    lExtra = []

    for j, haloMassBin in enumerate(md[sP.simName]["haloMassBins"]):
        label = r"%.1f < M$_{\rm halo}$ < %.1f" % (haloMassBin[0], haloMassBin[1])
        sExtra.append(plt.Line2D([0], [0], color=colors[j], marker="", linestyle="-"))
        lExtra.append(label)
    for i, sP in enumerate(sPs):
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", linestyle=linestyles[i]))
        lExtra.append(sP.simName)

    legend1 = ax.legend(sExtra, lExtra, loc="upper right", fontsize=13)
    ax.add_artist(legend1)

    # bottom panel
    ax = fig.add_subplot(212)
    ax.set_xlabel(r"M$_{\rm halo}$ [ log M$_{\rm sun}$ ] [ M$_{\rm 200,crit}$ ]")
    ax.set_ylabel(r"Satellite Progenitor Threshold Mass [ log M$_{\rm sun}$ ]")
    ax.set_xlim([12.0, 15.0])
    ax.set_ylim([8.0, 12.0])

    threshInds = [1, 2]

    for apertureIter in range(nApertures):
        for i, sP in enumerate(sPs):
            for threshIter in threshInds:
                x_vals = md[sP.simName]["indivHaloMasses"]
                y_vals = md[sP.simName]["indivHisto"][:, apertureIter, threshIter]

                xm, ym, _, pm = running_median(x_vals, y_vals, binSize=haloBinSize, percs=[16, 84])

                ym = savgol_filter(ym, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)

                alpha = [0.0, 0.3, 1.0][threshIter]

                (l,) = ax.plot(xm[:-1], ym[:-1], linestyles[i], color=colors[apertureIter], alpha=alpha)

                if apertureIter > 0 or threshIter == 1 or i > 0:
                    continue  # show percentile scatter only for first aperture

                ax.fill_between(xm[:-1], pm[0, :-1], pm[-1, :-1], color=l.get_color(), interpolate=True, alpha=0.1)

    # legend
    sExtra = []
    lExtra = []

    for apertureIter in range(nApertures):
        sExtra.append(plt.Line2D([0], [0], color=colors[apertureIter], marker="", linestyle="-"))
        lExtra.append(apertureNames[apertureIter])
    for threshIter in threshInds:
        alpha = [0.0, 0.1, 1.0][threshIter]
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", linestyle="-", alpha=alpha))
        lExtra.append(r"f$_{\rm ex-situ}$ > %.1f" % md[sPs[0].simName]["threshVals"][threshIter])
    for i, sP in enumerate(sPs):
        sExtra.append(plt.Line2D([0], [0], color="black", marker="", linestyle=linestyles[i]))
        lExtra.append(sP.simName)

    legend1 = ax.legend(sExtra, lExtra, loc="upper left", fontsize=13)
    ax.add_artist(legend1)

    # finish plot
    fig.savefig("merger_progmass_%s_%d.pdf" % ("-".join([sP.simName for sP in sPs]), sP.snap))
    plt.close(fig)


def tngMethods2_windPatterns(conf=1, pageNum=0):
    """Plot gas streamlines (galaxy wind patterns) https://arxiv.org/abs/1703.02970 (Figure 5).

    4x2, top four from L25n512_0000 and bottom four from L25n512_0010 (Illustris model), matched.
    """
    # change to: barAreaHeight = np.max([0.035,0.14 / nRows]) if conf.colorbars else 0.0
    # change to: if sP.isPartType(partType,'gas'):   config['ctName'] = 'perula' #'magma'
    run = "tng"
    res = 512
    variant = "0000"  # TNG fiducial
    matchedToVariant = "0010"  # Illustris fiducial

    # stellar composite, 50 kpc/h on a side, include M* label per panel, and scale bar once
    redshift = 2.0
    rVirFracs = None
    method = "sphMap"
    nPixels = [700, 700]
    axes = [0, 1]
    labelZ = False
    labelSim = False
    labelHalo = "mstar"
    relCoords = True
    mpb = None
    rotation = "edge-on"

    vecOverlay = "gas_vel"  # experimental gas (x,y) velocity streamlines
    vecMinMax = [0, 450]  # range for streamlines color scaling and colorbar
    vecColorbar = True
    vecMethod = "E"  # colored streamlines, uniform thickness
    vecColorPT = "gas"
    vecColorPF = "vmag"

    size = 50.0  # [50,80,120] --> 25,40,60 ckpc/h each direction
    sizeType = "codeUnits"

    if conf == 0:
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"

    if conf == 1:
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [7.2, 8.6]

    # set font
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    # pick halos from this run and crossmatch
    sP = simParams(res=res, run=run, redshift=redshift, variant=variant)
    sP2 = simParams(res=res, run=run, redshift=redshift, variant=matchedToVariant)

    # z2_11.5_page (z=2)
    # selectHalosFromMassBin(): In massBin [11.5 11.7] have 85 halos total.
    # pages = [[3498, 3833, 3861, 4097], [4250, 4481, 4511, 4578], [4601, 4656, 4720, 4763],
    #         [4882, 4898, 4913, 4928], [4952, 4985, 5004, 5023], [5037, 5049, 5062, 5077],
    #         [5133, 5154, 5173, 5186], [5202, 5220, 5231, 5240], [5278, 5295, 5310, 5323],
    #         [5396, 5441, 5459, 5469], [5482, 5491, 5508, 5520], [5533, 5546, 5558, 5584]]

    # z2 selections from above
    pages = [[3498, 4250, 5396, 5173], [4481, 4656, 5482, 5546]]

    shIDs = pages[pageNum]

    # crossmatch to other run
    shIDs2 = crossMatchSubhalosBetweenRuns(sP, sP2, shIDs)
    assert shIDs2.min() >= 0  # if any matches failed, we should make a blank panel

    # create panels, one per galaxy
    panels = []
    for i, shID in enumerate(shIDs):
        labelScaleLoc = True if i == 0 else False
        panels.append({"subhaloInd": shID, "labelScale": labelScaleLoc, "variant": variant})
    for shID in shIDs2:
        panels.append({"subhaloInd": shID, "labelScale": labelScaleLoc, "variant": matchedToVariant})

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 700
        colorbars = True
        saveFilename = "./methods2_gasflows_z2_final_page-%d_%s-%s_%s-%s_%s_%dckpch.pdf" % (
            pageNum,
            sP.simName,
            matchedToVariant,
            partType,
            partField,
            rotation,
            size,
        )

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def annalisa_tng50_presentation(setNum=0, stars=False):
    """TNG50 presentation paper: face-on + edge-on combination, 5x5 systems."""
    panels = []

    # V-band: half-light height <= 0.1 half-light radius, changing 1- to 0-indexed
    shIDs_z2 = (np.array([1,3,4,8069,8070,8073,21556,29444,31834,34605,39746,42407,50682,53050,55107,55108,57099,
                          57100,59079,60751,62459,66744,68179,69832,75669,79351,82446,83579,90627,90628,94052,99964,
                          102286,103000,103794,106288,110544,111197,113350,115248,115583,117547,118609,121253,123608,
                          124588,125842,125844,127581,128110,129662,130666,130667,132291,132792,134388,136382,139178,
                          141751,142608,143906,145493,146307,150453,150766,152212,153726,154636,155423,156280,158465,
                          160187,165800,166214,167146,169206,171250,172012,174040,176757,180691,181897,184340,184901,
                          185468,187834,188038,189522,193523,194076,194502,197278,201315,202435,204189,205429,208429,
                          211255,213909,216100,221701,228212,229128,232259,232895,233249,234374,236920,239921,241048,
                          241387,242245,242473,242664,242837,243973,244058,244372,245190,246344,249290,249750,253072,
                          253346,256641,263034,264392,267310,272731,279217,279294,280655,288386,289422,294036,294128,
                          306443,312047,319791,328084,349390,353735,364166,])-1)  # fmt: skip

    # 39745
    shIDs_z2_final25 = [29443,79350,60750,8069,57099,68178,110543,90627,55107,102285,113349,121252,125841,115247,
                        115582,127580,132290,130665,129661,139177,145492,146306,154635,189521,246343,]  # fmt: skip

    shIDs_s67_thin = [77281,353207,402894,421627,432764,433484,448408,448785,479317,495393,497214,497214]  # fmt: skip

    res = 2160
    redshift = 2.0
    run = "tng"
    rVirFracs = None
    method = "sphMap"
    axes = [0, 1]
    sizeType = "kpc"
    size = 40.0

    faceOnOptions = {
        "rotation": "face-on",
        "labelScale": "physical",
        "labelHalo": "mstar,redshift",
        "nPixels": [400, 400],
    }

    edgeOnOptions = {"rotation": "edge-on", "labelScale": False, "labelHalo": False, "nPixels": [400, 100]}

    if stars:
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
    else:
        partType = "gas"
        partField = "sfr_halpha"
        valMinMax = [38.0, 41.0]  # 40.7

    # set font
    import matplotlib as mpl

    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]

    class plotConfig:
        plotStyle = "edged"
        rasterPx = faceOnOptions["nPixels"][0] * 4
        colorbars = True

    # select halos
    if str(setNum) == "final":
        shIDs = shIDs_z2_final25
        plotConfig.nCols = 5
        nRows = 5 * 2
    elif str(setNum) == "superthin":
        shIDs = shIDs_s67_thin
        nCols = 4
        plotConfig.nRows = 3 * 2
        redshift = 0.5
    else:
        numPer = 35
        nCols = 7
        plotConfig.nRows = 5 * 2
        shIDs = shIDs_z2[numPer * setNum : numPer * (setNum + 1)]

    # configure panels: face-on and edge-on in alternating rows
    for i in range(int(plotConfig.nRows / 2)):
        for j in range(nCols):
            panels.append({"subhaloInd": shIDs[i * nCols + j], **faceOnOptions})
        for j in range(nCols):
            panels.append({"subhaloInd": shIDs[i * nCols + j], **edgeOnOptions})

    plotConfig.saveFilename = savePathDefault + "renderHalo_test_set-%s_%s.pdf" % (setNum, partType)

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def gjoshi_clustermaps(conf=0, haloID=0):
    """Joshi et al. 2020 (Figs. 1 and 2): stellar maps of two TNG50 Virgo-mass clusters at z=0.

    In a single panel centered on a halo, show one field from the box. To run for HaloID = 0,1 at snap = 099.
    Author: A. Pillepich
    """
    panels = []

    run = "tng"  #'tng_zoom_dm'
    res = 2160  # 1820
    variant = None  #'sf2' # None
    redshift = 0.0
    # redshift   = simParams(res=2160,run='tng',snap=snap).redshift
    rVirFracs = [0.5, 1.0]  # None
    method = "sphMap"
    nPixels = [2400, 2400]  # [1200,1200] #[800,800] #[1920,1920]
    axes = [0, 1]
    labelZ = True
    labelScale = True
    labelSim = False  # True
    labelHalo = True
    relCoords = True
    rotation = None
    mpb = None

    # excludeSubhaloFlag = True

    sP = simParams(res=res, run=run, redshift=redshift, hInd=haloID, variant=variant)

    if not sP.isZoom:
        # periodic box, FoF/Halo ID
        subhaloInd = sP.groupCatSingle(haloID=haloID)["GroupFirstSub"]
    else:
        # zoom, assume input haloID specifies the zoom simulation
        subhaloInd = haloID

    if conf == 0:
        # stellar mass column density
        panels.append({"partType": "stars", "partField": "coldens_msunkpc2", "valMinMax": [3.0, 10.0]})
        size = 2.0
        sizeType = "rVirial"

    class plotConfig:
        plotStyle = "edged"
        rasterPx = 1200
        colorbars = False  # True
        saveFilename = "./gjoshi_clustermaps_%d_%s_%d_%d_ID-%d_%s.pdf" % (conf, run, res, sP.snap, haloID, method)

    # plotSubhaloIDs = [15, 26, 29, 38, 39, 44, 48, 54, 63, 67, 73] #fof0 of TNG50: disks at accretion, not disks at z=0
    # plotSubhaloIDs = [10, 24] #fof0 of TNG50: disks at accretion, still disks at z=0
    # plotSubhaloIDs = [63877, 63878, 63884, 63893, 63898, 63899, 63902, 63917]
    # plotSubhaloIDs = [ 63869, 63872, 63879,63882, 63883, 63894]

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def apillepich_TNG50MWM31s_bubbles_top30(setType="top30", partType="gas", partField="P_gas", rotation="edge-on"):
    """Pillepich et al. 2021 (Figs. 2, 3 and Appendix): mostly edge-on views of MW/M31 analogs.

    6x5 posters of random 30 bubbles, P_gas and X-ray.
    Interesting options for partType='gas': xray_lum, coldens_msunkpc2, machnum, P_gas, P_B, entropy, temperature,
      metal_solar, vrad, bmag_uG, HI_segmented, ionmassratio_OVII_OVIII, SN_IaII_ratio_Fe
    Interesting options for  partType='stars': coldens_msunkpc2

    Other setType options:
      * lowSFRs: 4x4 posters of galaxies with lowSFRs and yet bubbles: P_gas, HI_segmented ...
      * MWs: 7x3 posters of MW-like galaxies with bubbles: P_gas, X-ray, temperature, machnum, HI_segmented...
      * M31s: 3x3 posters of MW-like galaxies with bubbles: P_gas, X-ray, temperature, machnum, HI_segmented...
    Author: A. Pillepich
    """
    panels = []

    # imaging options
    plotOptions = {
        "rotation": rotation,
        "labelSim": True,
        "labelZ": False,
        "labelScale": True,
        "labelHalo": "mstar,redshift",
        "nPixels": [400, 400],
    }

    # set font
    # import matplotlib as mpl
    # mpl.rcParams['font.family'] = 'serif'
    # mpl.rcParams['font.serif'] = ['Times New Roman']

    if partField == "machnum":
        ptRestrictions = {"machnum": ["gt", 0.9]}
        hsmlFac = 0.7

    class plotConfig:
        plotStyle = "edged"
        rasterPx = plotOptions["nPixels"][0] * 4
        colorbars = True
        fontsize = 70.0

    # select sample
    res = 2160
    redshift = 0.0
    snap = 99
    run = "tng"
    rVirFracs = [1.0]
    method = "sphMap"
    axes = [0, 1]
    sizeType = "kpc"
    size = 200.0
    depthFac = 0.1
    setNum = 0

    if setType == "top30":
        ids = np.loadtxt(
            "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/bubbles/Bubbles_P_gas_VisuallyIdentified_099_SubfindIDs_Top30.txt",
            dtype="int",
        )
        numPerSet = 30
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 5
        plotConfig.nRows = 6
    elif setType == "lowSFRs":
        ids = np.loadtxt(
            "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/bubbles/Bubbles_P_gas_VisuallyIdentified_099_SubfindIDs_logSFRlowerMinus1.txt",
            dtype="int",
        )
        numPerSet = 16
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 4
        plotConfig.nRows = 4
    elif setType == "MWs":
        ids = np.loadtxt(
            "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/bubbles/Bubbles_P_gas_VisuallyIdentified_099_SubfindIDs_MWAnalogs_SFR_Mstars.txt",
            dtype="int",
        )
        numPerSet = 21
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 7
        plotConfig.nRows = 3
    elif setType == "M31s":
        ids = np.loadtxt(
            "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/bubbles/Bubbles_P_gas_VisuallyIdentified_099_SubfindIDs_M31Analogs_SFR_Mstars.txt",
            dtype="int",
        )
        numPerSet = 9
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 3
        plotConfig.nRows = 3

    # custom options
    # To reproduce paper plots, leave auto ranges of all fields but 'xray_lum' and 'xray_lum_05-2kev'
    if partField == "P_gas":
        valMinMax = [1.0, 3.0]
    if partField == "xray_lum":
        valMinMax = [33.0, 36.0]
    if partField == "xray_lum_05-2kev":
        valMinMax = [33.0, 36.0]
    if partField == "temperature":
        valMinMax = [5.0, 7.5]
    if partField == "coldens_msunkpc2":
        valMinMax = [4.0, 7.0]
    if partField == "O VI":
        valMinMax = [11.0, 15.0]
    if partField == "machnum":
        valMinMax = [0.0, 5.0]
    if partField == "metal_solar":
        valMinMax = [-1.0, 1.0]

    # configure panels: only edge-on
    for i in range(int(plotConfig.nRows)):
        for j in range(nCols):
            panels.append({"subhaloInd": shIDs[i * nCols + j], **plotOptions})

    plotConfig.saveFilename = (
        savePathDefault
        + "apillepich_%s_bubbles_%s_%s_%d_%s_Lkpc_%d_DepthPercentage_%d_%s_%s.pdf"
        % (setType, run, res, snap, rotation, size, depthFac * 100, partType, partField)
    )
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def pgalan_tng50_fornax(setType="fornax10", partType="stars", partField="coldens_msunkpc2"):
    """
    Galan et al. 2022 (Fig. 1): stellar mass maps of Fornax-like groups and clusters.

    Other interesting options for partType='gas': xray_lum, coldens_msunkpc2, machnum, P_gas, P_B, entropy,
      temperature, metal_solar, vrad, bmag_uG, HI_segmented, OVI_OVII_ionmassratio, SN_IaII_ratio_Fe
    Author: A. Pillepich
    """
    panels = []

    # imaging options
    plotOptions = {
        "rotation": None,
        "labelSim": True,
        "labelZ": False,
        "labelScale": True,
        "labelHalo": "mhalo,mstar,redshift",
        "nPixels": [400, 400],
    }

    class plotConfig:
        plotStyle = "edged"
        rasterPx = plotOptions["nPixels"][0] * 4
        colorbars = True
        fontsize = 70.0

    sP = simParams(res=2160, run="tng", redshift=0.0)
    snap = 99
    rVirFracs = [1.0]
    method = "sphMap_global"
    axes = [0, 1]
    rVirFracs = [0.5, 1.0]
    size = 4.0
    sizeType = "rVirial"  #'kpc'
    setNum = 0

    if setType == "fornax10":
        ids = [2, 3, 4, 6, 7, 8, 9, 10, 11, 13]
        numPerSet = 10
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 2
        plotConfig.nRows = 5
    elif setType == "top16":
        ids = list(range(16))
        numPerSet = 16
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 4
        plotConfig.nRows = 4

    # custom options
    if partField == "coldens_msunkpc2":
        valMinMax = [3.0, 10.0]
    if partField == "xray_lum":
        valMinMax = [33.0, 36.0]
    if partField == "xray_lum_05-2kev":
        valMinMax = [33.0, 36.0]

    # configure panels:
    for i in range(int(plotConfig.nRows * nCols)):
        local_subhaloInd = sP.groupCatSingle(haloID=shIDs[i])["GroupFirstSub"]
        panels.append({"subhaloInd": local_subhaloInd, **plotOptions})

    plotConfig.saveFilename = savePathDefault + "pgalan_clustermaps_%s_%s_%d_%d_%s_%s_%s.png" % (
        setType,
        sP.run,
        sP.res,
        snap,
        method,
        partType,
        partField,
    )
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def cengler_tng50_MWM31satellites(setNum=0, setType="MWM31satellites_selection", setMaps="stellarmass"):
    """Engler et al. 2022 (Fig. 1): stellar mass and gas maps of massive MW/M31-like galaxies.

    Selection of satellites: MW/M31-like hosts, <300 kpc, massive.
    To be run for setNum = 0, ...11 and setMaps='stellarmass',gasmass,...
    Author: A. Pillepich
    """
    panels = []

    # imaging options
    plotOptions = {
        "rotation": None,
        "labelSim": True,
        "labelZ": False,
        "labelScale": True,
        "labelHalo": "mhalo,mstar,redshift",
        "nPixels": [400, 400],
    }

    class plotConfig:
        plotStyle = "edged"
        rasterPx = plotOptions["nPixels"][0] * 4
        colorbars = True
        fontsize = 70.0

    sP = simParams(res=2160, run="tng", redshift=0.0)
    snap = 99
    rVirFracs = [1.0]
    method = "sphMap"  # _subhalo'
    axes = [0, 1]
    rVirFracs = [0.5, 1.0]
    size = 100
    sizeType = "kpc"

    if setType == "MWM31satellites":
        path = "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/mwm31s/"
        file = "L35n2160TNG_099_SubfindIDs_MassiveSats_MWM31like.txt"

        ids = np.loadtxt(path + file, dtype="int")
        numPerSet = 20
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 4
        plotConfig.nRows = 5
    elif setType == "MWM31satellites_selection":
        # ids         = [388547, 424295, 479291, 492877, 511304] # C. Engler proposals
        # ids         = [388547, 424295, 447915, 479291, 485058, 492877, 502374, 511304, 525535, 567386]
        ids = [424295, 435758, 447915, 479291, 485058, 492877, 502374, 511304, 525535, 567386]  # 435758
        numPerSet = 10
        shIDs = ids[setNum * numPerSet : (setNum + 1) * numPerSet]
        print(shIDs)
        nCols = 5
        plotConfig.nRows = 2

    if str(setMaps) == "stellarmass":
        partType = "stars"
        partField = "coldens_msunkpc2"
        valMinMax = [3.0, 9.5]
        # hsmlFac     = 1.0,
    elif str(setMaps) == "stellarlight":
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
    elif str(setMaps) == "gasmass":
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [4.0, 8.0]

    # custom options
    if partField == "xray_lum":
        valMinMax = [33.0, 36.0]
    if partField == "xray_lum_05-2kev":
        valMinMax = [33.0, 36.0]

    # configure panels:
    for i in range(int(plotConfig.nRows * nCols)):
        panels.append({"subhaloInd": shIDs[i], **plotOptions})

    plotConfig.saveFilename = savePathDefault + "cengler_satellitemaps_%s_%s_%d_%d_%s_%s_%s_set-%d.png" % (
        setType,
        sP.run,
        sP.res,
        snap,
        method,
        partType,
        partField,
        setNum,
    )
    renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)


def apillepich_TNG50MWM31s_maps(setType="TNG50MWM31s", setMaps="stellarlight"):
    """Pillepich et al. 2022 (Figs. XXX): various maps of the MW/M31-like galaxies in TNG50, random projections.

    Selection of galaxies: TNG50 MW/M31-like hosts.
    To be run for setMaps='stellarlight', 'stellarmass',gasmass,...
    Author: A. Pillepich
    """
    panels = []

    # imaging options
    plotOptions = {
        "rotation": None,
        "labelSim": True,
        "labelZ": False,
        "labelScale": True,
        "labelHalo": "mhalo,mstar,redshift",
        "nPixels": [400, 400],
    }

    class plotConfig:
        plotStyle = "edged"
        rasterPx = plotOptions["nPixels"][0] * 4
        colorbars = True
        fontsize = 20.0

    sP = simParams(res=2160, run="tng", redshift=0.0)
    snap = 99
    rVirFracs = [1.0]
    method = "sphMap"  # _subhalo'
    axes = [0, 1]
    rVirFracs = [0.5, 1.0]
    size = 100
    sizeType = "kpc"

    if setType == "TNG50MWM31s":
        fname = "/u/apillepi/sims.TNG/L35n2160TNG/appostprocessing/mwm31s/TNG_L35n2160TNG_099_MWM31likeGalaxies.txt"
        data = np.genfromtxt(fname, skip_header=4)
        ids = data[:, 0]
        ids = ids.astype(int)
        print(ids)

    if str(setMaps) == "stellarmass":
        partType = "stars"
        partField = "coldens_msunkpc2"
        valMinMax = [6.0, 12.0]
        # hsmlFac     = 1.0,
    elif str(setMaps) == "stellarlight":
        partType = "stars"
        partField = "stellarComp-jwst_f200w-jwst_f115w-jwst_f070w"
    elif str(setMaps) == "gasmass":
        partType = "gas"
        partField = "coldens_msunkpc2"
        valMinMax = [4.0, 8.0]

    # configure panels:
    for i in ids:
        panels = []
        panels.append({"subhaloInd": i, **plotOptions})
        plotConfig.saveFilename = savePathDefault + "apillepich_mwm31s_maps_%s_%s_%d_%d_%s_%s_%s_%d_proj_%d_%d.png" % (
            setType,
            sP.run,
            sP.res,
            snap,
            method,
            partType,
            partField,
            i,
            axes[0],
            axes[1],
        )
        renderSingleHalo(panels, plotConfig, locals(), skipExisting=False)
