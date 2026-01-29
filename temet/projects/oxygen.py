"""
Oxygen (OVI, OVII and OVIII) TNG paper.

http://arxiv.org/abs/1712.00016
"""

from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize, colorConverter
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.stats import binned_statistic_2d, gaussian_kde

from temet.cosmo.cloudy import cloudyIon
from temet.cosmo.clustering import twoPointAutoCorrelationParticle
from temet.cosmo.util import cenSatSubhaloIndices
from temet.load.data import berg2019, chen2018zahedy2019, johnson2015, werk2013
from temet.obs.galaxySample import addIonColumnPerSystem, ionCoveringFractions, obsMatchedSample
from temet.plot import snapshot, subhalos
from temet.plot.cloudy import ionAbundFracs2DHistos
from temet.plot.config import colors, figsize, linestyles, lw, sKn, sKo
from temet.plot.quantities import quantList
from temet.plot.util import add_resolution_lines, loadColorTable
from temet.util import simParams
from temet.util.helper import closest, logZeroNaN, reducedChiSq, running_median
from temet.util.match import match


def nOVIcddf(sPs, pdf, moment=0, simRedshift=0.2, boxDepth10=False, boxDepth125=False):
    """CDDF (column density distribution function) of O VI in the whole box at z~0.

    (Schaye Fig 17) (Suresh+ 2016 Fig 11).
    """
    from temet.load.data import danforth2008, danforth2016, thomChen2008, tripp2008

    # config
    lw = 3.5
    speciesList = ["nOVI"]  # ,'nOVI_solar','nOVI_10','nOVI_25']
    if boxDepth10:
        for i in range(len(speciesList)):
            speciesList[i] += "_depth10"
    if boxDepth125:
        speciesList = ["nOVII_solarz_depth125"]  # ,'nOVII_10_solarz_depth125']

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([12.5, 15.5])
    ax.set_xlabel(r"N$_{\rm OVI}$ [ log cm$^{-2}$ ]")

    if moment == 0:
        ax.set_ylim([-17, -11])
        ax.set_ylabel(r"log f(N$_{\rm OVI}$) [ cm$^{2}$ ]")  # 0th moment
    if moment == 1:
        ax.set_ylim([-0.5, 1.5])
        ax.set_ylabel(r"log N$_{\rm OVI}$ f(N$_{\rm OVI}$)")  # 1st moment

    # observational points
    d16 = danforth2016()
    d08 = danforth2008()
    tc08 = thomChen2008()
    t08 = tripp2008()

    if moment == 1:
        d16["log_fOVI"] = np.log10(10.0 ** d16["log_fOVI"] * 10.0 ** d16["log_NOVI"])
        d08["log_fOVI"] = np.log10(10.0 ** d08["log_fOVI"] * 10.0 ** d08["log_NOVI"])
        tc08["log_fOVI"] = np.log10(10.0 ** tc08["log_fOVI"] * 10.0 ** tc08["log_NOVI"])
        t08["log_fOVI"] = np.log10(10.0 ** t08["log_fOVI"] * 10.0 ** t08["log_NOVI"])

    opts = {"alpha": 0.9, "capsize": 0.0}
    l1, _, _ = ax.errorbar(
        d16["log_NOVI"],
        d16["log_fOVI"],
        yerr=[d16["log_fOVI_errDown"], d16["log_fOVI_errUp"]],
        xerr=d16["log_NOVI_err"],
        color="#555555",
        ecolor="#555555",
        fmt="s",
        **opts,
    )

    l2, _, _ = ax.errorbar(
        d08["log_NOVI"],
        d08["log_fOVI"],
        yerr=[d08["log_fOVI_errDown"], d08["log_fOVI_errUp"]],
        xerr=d08["log_NOVI_err"],
        color="#999999",
        ecolor="#999999",
        fmt="D",
        **opts,
    )

    l3, _, _ = ax.errorbar(
        tc08["log_NOVI"],
        tc08["log_fOVI"],
        yerr=tc08["log_fOVI_err"],
        xerr=[tc08["log_NOVI_errLeft"], tc08["log_NOVI_errRight"]],
        color="#cccccc",
        ecolor="#cccccc",
        fmt="s",
        **opts,
    )

    l4, _, _ = ax.errorbar(
        t08["log_NOVI"],
        t08["log_fOVI"],
        yerr=t08["log_fOVI_err"],
        xerr=t08["log_NOVI_err"],
        color="#aaaaaa",
        ecolor="#aaaaaa",
        fmt="o",
        **opts,
    )

    labels = [d16["label"], d08["label"], tc08["label"], t08["label"]]
    legend1 = ax.legend([l1, l2, l3, l4], labels, loc="lower left")
    ax.add_artist(legend1)

    # loop over each fullbox run
    prevName = ""
    lwMod = 0.0

    for sP in sPs:
        if sP.isZoom:
            continue

        print("CDDF OVI: " + sP.simName)
        sP.setRedshift(simRedshift)

        if sP.simName.split("-")[0] == prevName:
            # decrease line thickness, leave color unchanged
            lwMod += 1.0
        else:
            # next color
            c = ax._get_lines.get_next_color()
            prevName = sP.simName.split("-")[0]
            lwMod = 0.0

        # pre-computed CDDF: first species for sizes
        ac = sP.auxCat(fields=["Box_CDDF_" + speciesList[0]])
        n_OVI = ac["Box_CDDF_" + speciesList[0]][0, :]
        fN_OVI = ac["Box_CDDF_" + speciesList[0]][1, :]

        # pre-computed CDDF: allocate for max/min bounds of our variations
        fN_OVI_min = fN_OVI * 0.0 + np.inf
        fN_OVI_max = fN_OVI * 0.0

        for i, species in enumerate(speciesList):
            # load pre-computed CDDF
            acField = "Box_CDDF_" + species

            if sP.simName in ["Illustris-1", "TNG300-1"] and i > 0:
                continue  # skip expensive variations we won't use for the oxygen paper

            ac = sP.auxCat(fields=[acField], searchExists=True)
            if ac[acField] is None:
                print(" skip: %s %s" % (sP.simName, species))
                continue

            assert np.array_equal(ac["Box_CDDF_" + species][0, :], n_OVI)  # require same x-pts
            fN_OVI = ac[acField][1, :]

            fN_OVI_min = np.nanmin(np.vstack((fN_OVI_min, fN_OVI)), axis=0)
            fN_OVI_max = np.nanmax(np.vstack((fN_OVI_max, fN_OVI)), axis=0)

        # plot 'uncertainty' band
        xx = np.log10(n_OVI)

        if moment == 0:
            yy_min = logZeroNaN(fN_OVI_min)
            yy_max = logZeroNaN(fN_OVI_max)
            yy = logZeroNaN(0.5 * (fN_OVI_min + fN_OVI_max))
        if moment == 1:
            yy_min = logZeroNaN(fN_OVI_min * n_OVI)
            yy_max = logZeroNaN(fN_OVI_max * n_OVI)
            yy = logZeroNaN(0.5 * (fN_OVI_min * n_OVI + fN_OVI_max * n_OVI))

        ax.fill_between(xx, yy_min, yy_max, color=c, alpha=0.2, interpolate=True)

        # plot middle line
        label = sP.simName
        ax.plot(xx, yy, "-", lw=lw - lwMod, color=c, label=label)

        # calculate and print reduced (mean) chi^2
        chi2v = reducedChiSq(
            xx,
            yy,
            d16["log_NOVI"],
            d16["log_fOVI"],
            data_yerr_up=d16["log_fOVI_errUp"],
            data_yerr_down=d16["log_fOVI_errDown"],
        )
        print("[%s] vs Danforth+ (2016) reduced chi^2: %g" % (sP.simName, chi2v))

    # legend
    sExtra = []  # [plt.Line2D( (0,1),(0,0),color='black',lw=3.0,marker='',linestyle=ls) for ls in linestyles]
    lExtra = []  # [str(s) for s in speciesList]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="upper right")

    pdf.savefig()
    plt.close(fig)


def cddfRedshiftEvolution(sPs, saveName, ions, redshifts, moment=0, boxDepth10=False, colorOff=0):
    """Redshift evolution of the OVI CDDF."""
    from temet.load.data import danforth2016, muzahid2011

    # plot setup
    lw = 3.0
    sizefac = 0.7 if len(redshifts) > 1 else 1.0
    heightFac = 1.0 if ("main" in saveName or len(redshifts) > 1) else 0.95
    fig = plt.figure(figsize=[figsize[0] * sizefac, figsize[1] * sizefac * heightFac])
    ax = fig.add_subplot(111)

    ax.set_xlim([12.5, 16.0])
    ax.set_xlabel(r"N$_{\rm oxygen}$ [ log cm$^{-2}$ ]")
    if len(ions) == 1:
        ax.set_xlabel(r"N$_{\rm %s}$ [ log cm$^{-2}$ ]" % ions[0])

    if moment == 0:
        ax.set_ylim([-18, -12])
        if len(ions) == 1:
            ax.set_ylim([-19, -11])
        ax.set_ylabel(r"f(N$_{\rm oxygen}$) [ log cm$^{2}$ ]")  # 0th moment
        if len(ions) == 1:
            ax.set_ylabel(r"f(N$_{\rm %s}$) [ log cm$^{2}$ ]" % ions[0])
    if moment == 1:
        ax.set_ylim([-2.5, 1.5])
        ax.set_ylabel(r"N$_{\rm oxygen}$ $\cdot$ f(N$_{\rm oxygen}$) [ log ]")  # 1st moment
        if len(ions) == 1:
            ax.set_ylabel(r"N$_{\rm %s}$ $\cdot$ f(N$_{\rm %s}$) [ log ]" % (ions[0], ions[0]))
    if moment == 2:
        ax.set_ylim([-1.5, 2.2])
        ax.set_ylabel(r"[N$_{\rm oxygen}$$^2$ / 10$^{13}$] $\cdot$ f(N$_{\rm oxygen}$) [ log ]")  # 2nd moment
        if len(ions) == 1:
            ax.set_ylabel(r"N$_{\rm %s}$ $\cdot$ f(N$_{\rm %s}$) [ log ]" % (ions[0], ions[0]))

    # observational OVI points (not in paper)
    if 0:
        d16 = danforth2016()
        m11 = muzahid2011()

        if moment == 1:
            d16["log_fOVI"] = np.log10(10.0 ** d16["log_fOVI"] * 10.0 ** d16["log_NOVI"])
            m11["log_fOVI"] = np.log10(10.0 ** m11["log_fOVI"] * 10.0 ** m11["log_NOVI"])

        l1, _, _ = ax.errorbar(
            d16["log_NOVI"],
            d16["log_fOVI"],
            yerr=[d16["log_fOVI_errDown"], d16["log_fOVI_errUp"]],
            xerr=d16["log_NOVI_err"],
            color="#555555",
            ecolor="#555555",
            alpha=0.9,
            capsize=0.0,
            fmt="s",
        )

        yerr = [m11["log_fOVI_errDown"][2:], m11["log_fOVI_errUp"][2:]]
        xerr = [m11["log_NOVI_errLow"][2:], m11["log_NOVI_errHigh"][2:]]
        l2, _, _ = ax.errorbar(
            m11["log_NOVI"][2:],
            m11["log_fOVI"][2:],
            yerr=yerr,
            xerr=xerr,
            color="#999999",
            ecolor="#999999",
            alpha=0.9,
            capsize=0.0,
            fmt="o",
        )

        yerr = [m11["log_fOVI_errDown"][:2], m11["log_fOVI_errUp"][:2]]
        xerr = [m11["log_NOVI_errLow"][:2], m11["log_NOVI_errHigh"][:2]]
        _, _, _ = ax.errorbar(
            m11["log_NOVI"][:2],
            m11["log_fOVI"][:2],
            yerr=yerr,
            xerr=xerr,
            color="#999999",
            ecolor="#999999",
            alpha=0.3,
            capsize=0.0,
            fmt="o",
        )

        legend1 = ax.legend([l1, l2], [d16["label"], m11["label"]], loc="upper right")
        ax.add_artist(legend1)

    # loop over each fullbox run
    for sP in sPs:
        txt = []

        for _ in range(colorOff + 1):
            ax.plot([], [])  # cycle color

        for _j, ion in enumerate(ions):
            print("[%s]: %s" % (ion, sP.simName))

            for i, redshift in enumerate(redshifts):
                sP.setRedshift(redshift)

                # Omega_ion value: compute
                if len(sPs) > 8:
                    fieldName = "Box_Omega_" + ion
                    boxOmega = sP.auxCat(fields=[fieldName])[fieldName]

                # pre-computed CDDF: load at this redshift
                fieldName = "Box_CDDF_n" + ion
                if boxDepth10:
                    fieldName += "_depth10"

                ac = sP.auxCat(fields=[fieldName])
                N_ion = ac[fieldName][0, :]
                fN_ion = ac[fieldName][1, :]

                xx = np.log10(N_ion)

                if moment == 0:
                    yy = logZeroNaN(fN_ion)
                if moment == 1:
                    yy = logZeroNaN(fN_ion * N_ion)
                if moment == 2:
                    yy = logZeroNaN(fN_ion * N_ion * (N_ion / 1e13))

                txt_loc = {}
                txt_loc["N"] = xx
                txt_loc["fN"] = yy
                txt_loc["z"] = redshift
                txt.append(txt_loc)

                # plot middle line
                if len(redshifts) == 1:
                    label = "%s %s" % (sP.simName, ion)
                else:
                    label = "%s %s z=%.1f" % (sP.simName, ion, redshift)

                label = ion
                if len(ions) == 1 and not boxDepth10:
                    label = sP.simName
                if len(sPs) > 1:
                    label = "%s %s" % (ion, sP.simName)
                if len(sPs) > 8:
                    label = "%s (%.1f)" % (sP.simName, boxOmega * 1e7)

                if i > 0:
                    label = ""
                c = "black" if (len(sPs) > 5 and sP.variant == "0000") else None
                lwLoc = lw if not (len(sPs) == 12 and sP.variant == "0000") else 2 * lw

                ls = linestyles[i]
                if i == 0 and len(sPs) > 8 and "BH" in sP.simName:
                    ls = "--"

                ax.plot(xx, yy, lw=lwLoc, color=c, linestyle=ls, label=label)

    # print
    if len(ions) == 1 and len(sPs) == 1:
        filename = "fig5_cddf_%s.txt" % ion
        out = "# Nelson+ (2018) http://arxiv.org/abs/1712.00016\n"
        out += "# Figure 5 CDDFs (%s z=%.1f)\n" % (sP.simName, sP.redshift)
        if boxDepth10:
            out += "# Note: Calculated for a projection depth of 10 cMpc/h (=14.8 pMpc at z=0)\n"
        out += "# N_%s [log cm^-2]" % (ion)
        for redshift in redshifts:
            out += " f(N)_z%d" % redshift
        out += " (all [log cm^2])\n"

        for i in range(len(txt) - 1):  # make sure columns are the same at each redshift (for each column)
            assert np.array_equal(txt[i + 1]["N"], txt[0]["N"])

        for i in range(txt[0]["N"].size):
            if txt[0]["N"][i] > 20.0:
                continue
            out += "%7.2f" % txt[0]["N"][i]
            for j in range(len(txt)):  # loop over redshifts
                out += " %7.3f" % txt[j]["fN"][i]
            out += "\n"

        with open(filename, "w") as f:
            f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(redshifts) > 1:
        for i, redshift in enumerate(redshifts):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            lExtra += ["z = %3.1f" % redshift]

    handles, labels = ax.get_legend_handles_labels()

    if len(sPs) == 13:  # main variants, split into 2 legends
        legend1 = ax.legend(handles[0:4], labels[0:4], loc="upper right", prop={"size": 18})
        ax.add_artist(legend1)
        ax.legend(handles[4:] + sExtra, labels[4:] + lExtra, loc="lower left", prop={"size": 18})
    else:  # default
        loc = "upper right" if len(ions) > 1 else "lower left"
        ax.legend(handles + sExtra, labels + lExtra, loc=loc)

    fig.savefig(saveName)
    plt.close(fig)


def totalIonMassVsHaloMass(
    sPs,
    saveName,
    ions=("OVI", "OVII"),
    cenSatSelect="cen",
    redshift=None,
    vsHaloMass=True,
    secondTopAxis=False,
    toAvgColDens=False,
    colorOff=2,
    toyFacs=None,
):
    """Plot total (gravitationally bound) mass of various ions, or e.g. cold/hot CGM mass, versus halo or stellar mass.

    If toAvgColDens, then plot average column density computed geometrically as (Mtotal/pi/rvir^2).
    If secondTopAxis, add the other (halo/stellar) mass as a secondary top axis, average relation.
    """
    binSize = 0.1  # log mass
    renames = {
        "AllGas": "Total Gas / 100",
        "AllGas_Metal": "Total Metals",
        "AllGas_Oxygen": "Total Oxygen",
        "AllGas_Mg": "Total Mg",
        "HI_GK": "Neutral HI",
    }
    ionColors = {"AllGas": "#444444", "AllGas_Metal": "#777777", "AllGas_Oxygen": "#cccccc"}

    runToyModel = False

    # plot setup
    lw = 3.0
    heightFac = 1.1 if secondTopAxis else 1.0
    fig = plt.figure(figsize=[figsize[0], figsize[1] * heightFac])
    ax = fig.add_subplot(111)

    mHaloLabel = r"Halo Mass [ log M$_{\rm sun}$ ]"
    mHaloField = "mhalo_200_log"
    mStarLabel = r"M$_{\star}$ [ < 30 pkpc, log M$_{\rm sun}$ ]"
    mStarField = "mstar_30pkpc_log"

    if vsHaloMass:
        ax.set_xlim([9.8, 13.7])
        ax.set_xlabel(mHaloLabel)
        massField = mHaloField
    else:
        ax.set_xlim([7.8, 11.7])
        ax.set_xlabel(mStarLabel)
        massField = mStarField

    if toAvgColDens:
        ax.set_ylim([12.0, 16.0])
        # ax.set_ylabel(r'Average Column Density $<N_{\rm oxygen}>$ [ log cm$^{-2}$ ]')
        ax.set_ylabel(r"Avg Column Density <N> [ log cm$^{-2}$ ]")
    else:
        ax.set_ylim([5.0, 11.0])
        if "AllGas" in ions:
            ax.set_ylim([4.0, 12.0])
        ax.set_ylabel(r"Total Halo Gas Mass [ log M$_{\rm sun}$ ]")

    if secondTopAxis:
        # add the other mass value as a secondary x-axis on the top of the panel
        axTop = ax.twiny()

        if vsHaloMass:  # x=halo, top=stellar
            topMassVals = [8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]
            axTop.set_xlabel(mStarLabel)
            topMassField = mStarField
        else:  # x=stellar, top=halo
            topMassVals = [11.0, 11.5, 12.0, 13.0, 14.0, 15.0]
            axTop.set_xlabel(mHaloLabel)
            topMassField = mHaloField

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())

    # loop over each fullbox run
    txt = []
    colors = []

    for i, sP in enumerate(sPs):
        # load halo masses and CSS
        txt_sP = []
        if redshift is not None:
            sP.setRedshift(redshift)
        xx = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
        xx = xx[cssInds]

        if secondTopAxis and i == 0:
            # load mass values for top x-axis, construct median relation interpolant, assign values
            xx_top = sP.groupCat(fieldsSubhalos=[topMassField])
            xx_top = xx_top[cssInds]
            xm, ym, _ = running_median(xx_top, xx, binSize=binSize, skipZeros=True, minNumPerBin=10)
            f = interp1d(xm, ym, kind="linear", bounds_error=False, fill_value="extrapolate")

            axTickVals = f(topMassVals)  # values of bottom x-axis for each topMassVals

            axTop.set_xticks(axTickVals)
            axTop.set_xticklabels(topMassVals)

        if toAvgColDens:
            # load virial radii
            rad = sP.groupCat(fieldsSubhalos=["rhalo_200_code"])
            rad = rad[cssInds]
            ionData = cloudyIon(None)

        if not toAvgColDens and runToyModel:
            # TEST: toy model for total grav. bound mass
            ion = cloudyIon(sP, el=["Oxygen"])

            # tempFac = 0.6
            # densFac = 2.0
            # metalFac = 1.5
            tempFac, densFac, metalFac = toyFacs

            # load total grav. bound gas mass
            field = "Subhalo_Mass_AllGas"
            tot_gas_mass = sP.auxCat(fields=[field])[field][cssInds]

            field2 = "Subhalo_Mass_AllGas_Metal"  # _Metal, _Oxygen
            tot_metal_mass = sP.auxCat(fields=[field2])[field2][cssInds]

            # calculate mean metallicity
            ww = np.where(xx >= 11.0)  # min halo mass to consider for mean metal frac
            mean_metal_frac = np.nanmean(tot_metal_mass[ww] / tot_gas_mass[ww])
            virMetallicity = mean_metal_frac / ion.solar_Z

            print(" mean metal mass fraction: ", mean_metal_frac)
            print(" mean virial metallicity in solar units: ", virMetallicity)
            print(" oxygen to total mass ratio (solar): ", ion._solarMetalAbundanceMassRatio("Oxygen"))

            # assume solar abundance of oxygen, scale to virMetallicity and use to compute total oxy mass
            oxyMassFracAtFacVirMetallicity = ion._solarMetalAbundanceMassRatio("Oxygen") * metalFac * virMetallicity
            virTotOxygenMass = tot_gas_mass * oxyMassFracAtFacVirMetallicity  # code units
            yy_toy = {}
            yy_toy["AllGas_Oxygen"] = sP.units.codeMassToLogMsun(virTotOxygenMass)

            # define temp, dens per halo
            haloMassesCode = sP.units.logMsunToCodeMass(xx)
            virTemp = sP.units.codeMassToVirTemp(haloMassesCode * tempFac, log=True)  # log K
            virVolume = (4.0 / 3.0) * np.pi * sP.units.codeMassToVirRad(haloMassesCode) ** 3.0
            virDens = (tot_gas_mass * ion.solar_X) / virVolume  # hydrogen, code units
            virDensPhys = np.log10(sP.units.codeDensToPhys(virDens * densFac, cgs=True, numDens=True))  # log(1/cm^3)

            # run cloudy for each ion, predict total grav. bound ionic mass
            metal = np.zeros(virTemp.size, dtype="float32") + np.log10(virMetallicity)
            for ionNum in [6, 7, 8]:
                log_ionFrac = ion.frac("Oxygen", ionNum, virDensPhys, metal, virTemp)
                virTotIonMass = 10.0**log_ionFrac * virTotOxygenMass
                yy_toy["O" + ion.numToRoman(ionNum)] = sP.units.codeMassToLogMsun(virTotIonMass)

        for j, ion in enumerate(ions):
            print("[%s]: %s" % (ion, sP.simName))

            # load and apply CSS
            fieldName = "Subhalo_Mass_%s" % ion

            ac = sP.auxCat(fields=[fieldName])
            if ac[fieldName] is None:
                continue
            ac[fieldName] = ac[fieldName][cssInds]

            # unit conversions
            if toAvgColDens:
                # per subhalo normalization, from [code mass] -> [ions/cm^2]
                ionName = ionData.formatWithSpace(ion, name=True)

                # [code mass] -> [code mass / code length^2]
                yy = ac[fieldName] / (np.pi * rad * rad)
                # [code mass/code length^2] -> [H atoms/cm^2]
                yy = sP.units.codeColDensToPhys(yy, cgs=True, numDens=True)
                yy /= ionData.atomicMass(ionName)  # [H atoms/cm^2] to [ions/cm^2]
                yy = logZeroNaN(yy)
            else:
                yy = sP.units.codeMassToLogMsun(ac[fieldName])

            if ion == "AllGas":
                yy -= 2.0  # offset!

            # calculate median and smooth
            xm, ym, sm, pm = running_median(
                xx, yy, binSize=binSize, binSizeLg=binSize * 2, skipZeros=True, percs=[10, 25, 75, 90], minNumPerBin=3
            )

            if xm.size > sKn:
                ym = savgol_filter(ym, sKn, sKo)
                sm = savgol_filter(sm, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)  # P[10,90]

            txt_sP.append([ion, xm, ym, pm[0, :], pm[-1, :]])

            # determine color
            if i == 0:
                if ion in ionColors:  # preset color
                    c = ionColors[ion]
                else:  # cycle
                    c = None
                    for _ in range(colorOff + 1):
                        ax.plot([], [])  # cycle color
                    if colorOff > 0:
                        colorOff = 0  # only once
                colors.append(c)
            else:
                c = colors[j]

            # plot median line
            label = ion if i == 0 else ""
            if ion in renames.keys() and i == 0:
                label = renames[ion]
            ax.plot(xm, ym, lw=lw, color=c, linestyle=linestyles[i], label=label)

            if i == 0:
                # show percentile scatter only for 'all galaxies'
                ax.fill_between(xm, pm[0, :], pm[-1, :], color=c, interpolate=True, alpha=0.2)

            if runToyModel and ion in ["AllGas_Oxygen", "OVI", "OVII", "OVIII"]:
                # TOY PLOT
                xm, ym, sm, pm = running_median(
                    xx, yy_toy[ion], binSize=binSize, skipZeros=True, percs=[10, 25, 75, 90], minNumPerBin=10
                )
                ym = savgol_filter(ym, sKn, sKo)
                sm = savgol_filter(sm, sKn, sKo)
                pm = savgol_filter(pm, sKn, sKo, axis=1)  # P[10,90]
                ax.plot(xm, ym, lw=lw, color=c, linestyle=":")

        txt.append(txt_sP)  # one list per sim

    # add linear scaling line for reference
    # xx = ax.get_xlim()
    # yy = [8.0, 8.0+(xx[1]-xx[0])]
    # ax.plot(xx, yy, '-', color='black', alpha=0.8, lw=lw)

    # print
    massAxis = "mhalo" if vsHaloMass else "mstar"
    for i, txt_sP in enumerate(txt):  # loop over runs
        field = "ionmasses" if not toAvgColDens else "avgcoldens"
        filename = "figN_%s_z%.1f_%s_%s.txt" % (sPs[i].simName, sPs[i].redshift, field, massAxis)

        out = "# Nelson+ (2020) http://arxiv.org/abs/1712.00016\n"
        if toAvgColDens:
            out += "# Figure N Right Panel (Average <N_ion> = M_ion / (pi*rvir^2)) (%s z=%.1f)\n" % (
                sPs[i].simName,
                sPs[i].redshift,
            )
            out += "# columns: %s" % massAxis
            for ion in ions:
                out += ", <N_%s>, p10, p90" % ion
        else:
            out += "# Figure N Left Panel (Total bound %s masses) (%s z=%.1f)\n" % (
                ", ".join(ions),
                sPs[i].simName,
                sPs[i].redshift,
            )
            out += "# columns: %s" % massAxis
            for ion in ions:
                out += ", M_%s, p10, p90" % ion
        out += "\n# all masses [log msun], column densities [log cm^-2], p10/p90 are percentiles of previous field\n"

        for j, ionData in enumerate(txt_sP):
            ion_name, ion_x, ion_y, ion_p10, ion_p90 = ionData
            if j == 0:
                ion1_x = ion_x
            assert np.array_equal(ion1_x, ion_x)  # same mass bins for all ions

        nMassBins = ion1_x.size

        for k in range(nMassBins):
            out += "%5.2f" % ion1_x[k]
            for _, ionData in enumerate(txt_sP):
                ion_name, ion_x, ion_y, ion_p10, ion_p90 = ionData
                out += ", %5.2f, %5.2f, %5.2f" % (ion_y[k], ion_p10[k], ion_p90[k])
            out += "\n"

        with open(filename, "w") as f:
            f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        allSameRun = all(sP.simName == sPs[0].simName for sP in sPs)
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            if allSameRun:
                lExtra += ["z = %.1f" % sP.redshift]
            else:
                lExtra += ["%s" % sP.simName]
    if runToyModel:
        sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=":", marker="")]
        lExtra += [r"Toy Model, f$_{\rm T}$=%.1f, f$_{\rm \rho}$=%.1f, f$_{\rm Z}$=%.1f" % (tempFac, densFac, metalFac)]
    loc = "upper right" if toAvgColDens else "lower right"
    legend1 = ax.legend(sExtra, lExtra, loc=loc)
    ax.add_artist(legend1)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper left")

    fig.savefig(saveName)
    plt.close(fig)


def stackedRadialProfiles(
    sPs,
    saveName,
    ions,
    redshift=0.0,
    cenSatSelect="cen",
    projDim="3D",
    radRelToVirRad=False,
    radRelToR500=False,
    massDensity=False,
    massDensityMsun=False,
    haloMassBins=None,
    stellarMassBins=None,
    xlim=(0.0, 3.0),
    combine2Halo=False,
    fieldTypes=("GlobalFoF",),
    emFlux=False,
    median=False,
):
    """Plot stacked radial number/mass density profiles for a series of halo or stellar mass bins.

    One or more ions, one or more runs, at a given redshift. Specify one of haloMassBins or stellarMassBins.

    Args:
      sPs (list[:py:class:`~util.simParams`]): list of simulation instances.
      saveName (str): output figure filename.
      ions (list[str]): list of ion names, e.g. ['OVI', 'OVII'].
      redshift (float): redshift to plot.
      cenSatSelect (str): 'cen', 'sat', or 'all' to select central/satellite/all subhalos.
      projDim (str): '3D' for radial profiles, or e.g. '2Dz_2Mpc' for projected/column density profiles.
      radRelToVirRad (bool): plot in [r/rvir] instead of [pkpc].
      radRelToR500 (bool): plot in [r/r500] instead of [pkpc].
      massDensity (bool): plot y-axis as [g/cm^3] instead of [1/cm^3].
      massDensityMsun (bool): plot y-axis as [Msun/kpc^3] or [Msun/kpc^2] if 2D.
      haloMassBins (list[2-tuple]): list of 2-tuples, each a halo mass bin min max (log Msun) to use.
      stellarMassBins (list[2-tuple]): list of 2-tuples, each a stellar mass bin min max (log Msun) to use.
      xlim (2-tuple): x-axis limits.
      combine2Halo (bool): combine the other-halo and diffuse terms.
      fieldTypes (list[str]): list of field types to plot, e.g. ['GlobalFoF'] (must correspond to auxcats).
      emFlux (bool): if True, then plot [photon/s/cm^2/ster].
      median (bool): if True, plot median profiles instead of mean.
    """
    # config
    percs = [16, 50, 84]  # [10,90] for oxygen paper

    partField = "Flux" if emFlux else "Mass"

    fieldNames = []
    for fieldType in fieldTypes:
        fieldNames.append("Subhalo_RadProfile%s_" + fieldType + "_%s_%s")

    radNames = ["total", "self (1-halo)", "other (2-halo)", "diffuse"]

    # plot setup
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    radStr = "Radius" if "3D" in projDim else "Impact Parameter"
    if radRelToVirRad:
        ax.set_xlim([-2.0, 2.0])
        ax.set_xlabel(r"%s / Virial Radius [ log ]" % radStr)
    elif radRelToR500:
        ax.set_xlim([-2.0, 1.0])
        ax.set_xlabel(r"%s / r$_{\rm 500}$ [ log ]" % radStr)
    else:
        ax.set_xlim(xlim)  # 4.0 for oxygen paper
        ax.set_xlabel(r"%s [ log pkpc ]" % radStr)

    speciesStr = ions[0] if len(ions) == 1 else "oxygen"

    if emFlux:
        assert "2D" in projDim
        ax.set_ylim([-6.5, 3])  # [-9.5,2]
        ax.set_xlim([1.0, 2.8])
        ax.set_ylabel(r"%s Emission [phot s$^{-1}$ cm$^{-2}$ ster$^{-1}$]" % speciesStr)
    else:
        if "3D" in projDim:
            # 3D mass/number density
            if massDensity:
                ax.set_ylim([-37.0, -30.0])
                ax.set_ylabel(r"Mass Density $\rho_{\rm %s}$ [ log g cm$^{-3}$ ]" % speciesStr)
            elif massDensityMsun:
                ax.set_ylim([-3.0, 0.0])
                ax.set_ylabel(r"Mass Density $\Sigma_{\rm %s}$ [ log M$_{\odot}$ kpc$^{-3}$ ]" % speciesStr)
            else:
                # ax.set_ylim([-14.0, -6.0])
                ax.set_ylim([-13.0, -4.0])
                ax.set_ylabel(r"Number Density $n_{\rm %s}$ [ log cm$^{-3}$ ]" % speciesStr)
        else:
            # 2D mass/column density
            if massDensity:
                ax.set_ylim([-12.0, -6.0])
                ax.set_ylabel(r"Column Mass Density $\rho_{\rm %s}$ [ log g cm$^{-2}$ ]" % speciesStr)
            elif massDensityMsun:
                ax.set_ylim([-4.0, 6.0])
                ax.set_ylabel(r"Surface Mass Density $\Sigma_{\rm %s}$ [ log M$_{\odot}$ kpc$^{-2}$ ]" % speciesStr)
            else:
                ax.set_ylim([11.0, 18.2])  # [11.0,16.0]
                ax.set_ylabel(r"Column Number Density $N_{\rm %s}$ [ log cm$^{-2}$ ]" % speciesStr)

    # init
    ionData = cloudyIon(None)
    colors = []
    rvirs = []

    if haloMassBins is not None:
        massField = "mhalo_200_log"
        massBins = haloMassBins
    else:
        massField = "mstar_30pkpc_log"
        massBins = stellarMassBins

    # loop over each fullbox run
    txt = []

    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        sP.setRedshift(redshift)
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii
        haloRadField = "rhalo_500_code" if radRelToR500 else "rhalo_200_code"
        rad = sP.groupCat(fieldsSubhalos=[haloRadField])
        rad = rad[cssInds]

        for j, ion in enumerate(ions):
            print("[%s]: %s" % (ion, sP.simName))

            # load and apply CSS
            for fieldName in fieldNames:
                fieldName = fieldName % (projDim, ion, partField)
                ac = sP.auxCat(fields=[fieldName])
                if ac[fieldName] is None:
                    continue

                # crossmatch 'subhaloIDs' to cssInds
                ac_inds, css_inds = match(ac["subhaloIDs"], cssInds)
                ac[fieldName] = ac[fieldName][ac_inds, :]
                masses_loc = masses[css_inds]
                rad_loc = rad[css_inds]

                # unit conversions: mass per bin to (space mass density) or (space number density)
                yy = ac[fieldName]

                if "3D" in projDim:
                    normField = "bin_volumes_code"
                    unitConversionFunc = sP.units.codeDensToPhys
                else:
                    normField = "bin_areas_code"  # 2D
                    unitConversionFunc = sP.units.codeColDensToPhys

                if emFlux:
                    # accumulated line fluxes [photon/s/cm^2], convert to [photon/s/cm^2/ster^2]
                    # using the bin size of the projected annulus
                    assert ac[fieldName].ndim == 2  # otherwise handle
                    nRadTypes = 1

                    pxDimsCode = [ac[fieldName + "_attrs"][normField], 1.0]  # code length^2, unity
                    yy = sP.units.fluxToSurfaceBrightness(yy, pxDimsCode, arcsec2=False, ster=True)
                else:
                    # accumulated ionic masses [code mass] -> [code density]
                    if ac[fieldName].ndim == 2:
                        yy /= ac[fieldName + "_attrs"][normField]
                        nRadTypes = 1
                    else:
                        for radType in range(ac[fieldName].shape[2]):
                            yy[:, :, radType] /= ac[fieldName + "_attrs"][normField]
                        nRadTypes = 4

                    if massDensity:
                        # from e.g. [code mass / code length^3] -> [g/cm^3]
                        yy = unitConversionFunc(yy, cgs=True)
                    elif massDensityMsun:
                        # from e.g. [code mass / code length^2] -> [Msun/kpc^2] in 2D
                        yy = unitConversionFunc(yy, msunKpc2=True)
                    else:
                        # from e.g. [code mass / code length^3] -> [ions/cm^3]
                        ionName = ionData.formatWithSpace(ion, name=True)
                        yy = unitConversionFunc(yy, cgs=True, numDens=True)
                        yy /= ionData.atomicMass(ionName)  # [H atoms/cm^3] to [ions/cm^3]

                # loop over mass bins
                for k, massBin in enumerate(massBins):
                    txt_mb = []
                    # select
                    w = np.where((masses_loc >= massBin[0]) & (masses_loc < massBin[1]))

                    print(" %s [%d] %.1f - %.1f : %d" % (projDim, k, massBin[0], massBin[1], len(w[0])))
                    assert len(w[0])

                    # radial bins: normalize to rvir if requested
                    avg_rvir_code = np.nanmedian(rad_loc[w])
                    if i == 0 and j == 0:
                        rvirs.append(avg_rvir_code)

                    # sum and calculate percentiles in each radial bin
                    for radType in range(nRadTypes):
                        if yy.ndim == 3:
                            yy_local = np.squeeze(yy[w, :, radType])

                            # combine diffuse into other-halo term, and skip separate line?
                            if combine2Halo and radType == 2:
                                yy_local += np.squeeze(yy[w, :, radType + 1])
                            if combine2Halo and radType == 3:
                                continue
                        else:
                            yy_local = np.squeeze(yy[w, :])

                        if radRelToVirRad or radRelToR500:
                            rr = 10.0 ** ac[fieldName + "_attrs"]["rad_bins_code"] / avg_rvir_code
                        else:
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
                        # yy_local[yy_local == 0.0] = np.nan

                        # calculate mean profile and scatter
                        if yy_local.ndim > 1:
                            yy_mean = np.nansum(yy_local, axis=0) / len(w[0])
                            yp = np.nanpercentile(yy_local, percs, axis=0)
                        else:
                            yy_mean = yy_local  # single profile
                            yp = np.vstack((yy_local, yy_local))  # no scatter

                        # log both axes
                        yy_mean = logZeroNaN(yy_mean)
                        yp = logZeroNaN(yp)
                        rr = np.log10(rr)

                        if rr.size > sKn:
                            yy_mean = savgol_filter(yy_mean, sKn, sKo)
                            yp = savgol_filter(yp, sKn, sKo, axis=1)  # P[10,50,90]

                        # determine color
                        if i == 0 and radType == 0:
                            c = ax._get_lines.get_next_color()

                        linestyle = linestyles[radType]  # 1-halo, 2-halo

                        # plot median line
                        label = ""
                        if i == 0 and radType == 0:
                            label = r"$M_{\rm halo}$ = %.1f" % (0.5 * (massBin[0] + massBin[1]))

                        if median:
                            ax.plot(rr, yp[1, :], color=c, linestyle=linestyle, label=label)
                        else:
                            ax.plot(rr, yy_mean, color=c, linestyle=linestyle, label=label)

                        txt_loc = {}
                        txt_loc["bin"] = massBin
                        txt_loc["rr"] = rr
                        txt_loc["yy"] = yy_mean
                        if median:
                            txt_loc["yy"] = yp[1, :]
                        txt_loc["yy_0"] = yp[0, :]
                        txt_loc["yy_1"] = yp[-1, :]
                        txt_mb.append(txt_loc)

                        # draw rvir lines (or 300pkpc lines if x-axis is already relative to rvir)
                        yrvir = ax.get_ylim()
                        yrvir = np.array([yrvir[1], yrvir[1] - (yrvir[1] - yrvir[0]) * 0.1]) - 0.25

                        if not radRelToVirRad and not radRelToR500:
                            xrvir = np.log10([avg_rvir_code, avg_rvir_code])
                            textStr = "R$_{\\rm vir}$"
                            if "3D" in projDim:
                                yrvir[1] -= 0.4 * k
                            else:
                                yrvir[1] -= 0.1 * k
                        else:
                            rvir_300pkpc_ratio = sP.units.physicalKpcToCodeLength(300.0) / avg_rvir_code
                            xrvir = np.log10([rvir_300pkpc_ratio, rvir_300pkpc_ratio])
                            textStr = "300 kpc"
                            if "3D" in projDim:
                                yrvir[1] -= 0.4 * (len(massBins) - k)
                            else:
                                yrvir[1] -= 0.1 * (len(massBins) - k)

                        if i == 0:
                            ax.plot(xrvir, yrvir, lw=lw * 1.5, color=c, alpha=0.1)
                            opts = {"alpha": 0.1, "rotation": 90, "va": "bottom", "ha": "right", "fontsize": 20}
                            ax.text(xrvir[0] - 0.02, yrvir[1], textStr, color=c, **opts)

                        if k == 0:  # i == 0 and radType == 0:
                            # show percentile scatter only for first run
                            wf = np.where(np.isfinite(yp[0, :]) & np.isfinite(yp[-1, :]))[0]
                            ax.fill_between(rr[wf], yp[-1, wf], yp[0, wf], color=c, interpolate=True, alpha=0.2)

                    txt.append(txt_mb)

    # gray resolution band at small radius
    # add_resolution_lines(ax, sPs[0], radRelToVirRad, rvirs=rvirs)

    # print
    for k in range(len(txt)):  # loop over mass bins (separate file for each)
        filename = "figN_%sdens_rad%s_m-%.2f.txt" % (
            "num" if projDim == "3D" else "col",
            "rvir" if radRelToVirRad else "kpc",
            np.mean(txt[k][0]["bin"]),
        )
        out = "# Nelson+ (2020) http://arxiv.org/abs/xxxx.xxxxx\n"
        out += "# Figure N Right Panel N_%s [log cm^-2] (%s z=%.1f)\n" % (ions[0], sP.simName, sP.redshift)
        out += "# Halo Mass Bin [%.1f - %.1f]\n" % (txt[k][0]["bin"][0], txt[k][0]["bin"][1])
        out += "# rad_logpkpc"
        for j in range(len(txt[k])):  # loop over rad types
            radName = radNames[j].split(" ")[0]
            out += " N_%s N_%s_err0 N_%s_err1" % (radName, radName, radName)
        out += "\n"
        for i in range(1, txt[k][0]["rr"].size):  # loop over radial bins
            out += "%8.4f " % txt[k][j]["rr"][i]
            for j in range(len(txt[k])):  # loop over rad types
                out += "%8.4f %8.4f %8.4f" % (txt[k][j]["yy"][i], txt[k][j]["yy_0"][i], txt[k][j]["yy_1"][i])
            out += "\n"
        with open(filename, "w") as f:
            f.write(out)

    # legend
    sExtra = []
    lExtra = []

    # if len(sPs) > 1: # linestyle by sP
    #    for i, sP in enumerate(sPs):
    #        sExtra += [plt.Line2D( (0,1),(0,0),color='black',lw=lw,linestyle=linestyles[i],marker='')]
    #        lExtra += ['%s' % sP.simName]
    if len(sPs) > 1:  # color by sP
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D([0], [0], lw=lw, linestyle="-", color=colors[i], marker="")]
            lExtra += ["%s" % sP.simName]

    for i in range(nRadTypes - int(combine2Halo)):
        sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
        lExtra += ["%s" % radNames[i]]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="lower left")

    fig.savefig(saveName)
    plt.close(fig)


def ionTwoPointCorrelation(sPs, saveName, ions, redshift=0.0, order=0, colorOff=0):
    """Plot the real-space 3D two point correlation function of e.g. OVI mass."""
    # visual config
    lw = 3.0
    alphaFill = 0.15
    drawError = True
    symSize = 7.0
    alpha = 1.0

    # quick helper mapping from ions[] inputs to snapshotSubset() particle field names
    ionNameToPartFieldMap = {
        "OVI": "O VI mass",
        "OVII": "O VII mass",
        "OVIII": "O VIII mass",
        "MgII": "Mg II mass",
        "Mg": "metalmass_Mg",
        "O": "metalmass_O",
        "Z": "metalmass",
        "gas": "mass",
        "bhmass": "BH_Mass",
    }

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim([0.0, 4.0])
    ax.set_xlabel("Radius [ log pkpc ]")

    ylim = [-1.0, 6.0]
    if order == 1:
        ylim = [1.0, 7.0]
    if order == 2:
        ylim = [1.0, 8.0]
    if ions[0] == "MgII":
        ylim[0] += 2.0
        ylim[1] += 2.0
    if ions[0] == "bhmass" and order == 0:
        ylim = [-1.0, 5.0]

    ax.set_ylim(ylim)
    ionStr = r"_{\rm %s}" % ions[0] if len(ions) == 1 else ""
    ax.set_ylabel(r"%s$\xi%s(r)$ [ log ]" % (["", "r", "r$^2$"][order], ionStr))

    # loop over each particle type/property
    for _ in range(colorOff + 1):
        ax.plot([], [])  # cycle colors

    for ion in ions:
        if ion == "bhmass":
            partType = "bh"
        else:
            partType = "gas"
        partField = ionNameToPartFieldMap[ion]

        # loop over each fullbox run
        for i, sP in enumerate(sPs):
            if "OVI" in ion and sP.res == 2160:
                continue
            if "MgII" in ion and sP.res == 1820:
                continue
            print("[%s]: %s" % (ion, sP.simName))
            sP.setRedshift(redshift)

            # load tpcf
            rad, xi, xi_err, _ = twoPointAutoCorrelationParticle(sP, partType=partType, partField=partField)

            xx = sP.units.codeLengthToKpc(rad)
            xx = rad
            ww = np.where(xi > 0.0)

            # y-axis multiplier
            if order == 0:
                yFac = 1.0
            if order == 1:
                yFac = xx[ww]
            if order == 2:
                yFac = xx[ww] ** 2

            x_plot = logZeroNaN(xx[ww])
            y_plot = logZeroNaN(yFac * xi[ww])

            label = ion if i == 0 else ""
            if label == "O":
                label = r"O, Z$_{\rm tot}$"
            (l,) = ax.plot(x_plot, y_plot, lw=lw, linestyle=linestyles[i], label=label, alpha=alpha)

            # todo, symbols, bands, etc
            if xi_err is not None and drawError:
                nSigma = 10.0  # 5sigma up and down
                if 1:
                    yy0 = y_plot - logZeroNaN(yFac * (xi[ww] - nSigma * xi_err[ww] / 2))
                    yy1 = logZeroNaN(yFac * (xi[ww] + nSigma * xi_err[ww] / 2)) - y_plot
                    ax.errorbar(
                        x_plot,
                        y_plot,
                        yerr=[yy0, yy1],
                        markerSize=symSize,
                        color=l.get_color(),
                        ecolor=l.get_color(),
                        alpha=alphaFill * 2,
                        capsize=0.0,
                        fmt="o",
                    )

                if 0:
                    yy0 = logZeroNaN(yFac * (xi[ww] - nSigma * xi_err[ww] / 2))
                    yy1 = logZeroNaN(yFac * (xi[ww] + nSigma * xi_err[ww] / 2))

                    ax.fill_between(x_plot, yy0, yy1, color=l.get_color(), interpolate=True, alpha=alphaFill)

    # gray resolution band at small radius
    add_resolution_lines(ax, sPs[0], corrMaxBox=True)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            lExtra += ["%s" % sP.simName]

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + sExtra, labels + lExtra, loc="best")

    fig.savefig(saveName)
    plt.close(fig)


def obsSimMatchedGalaxySamples(sPs, saveName, config="COS-Halos"):
    """Plot the COS-Halos (or other observed) galaxies data, and our mock sample."""
    # config
    detLimitAlpha = 0.6  # alpha transparency for upper/lower limits

    cmap = loadColorTable("RdYlGn")
    colorMinMax = [0.1, 0.3]  # redshift
    cbarTextSize = 12
    nBinsHist = 8

    cmap2D = loadColorTable("gray_r")
    nBinsHist2D = 30

    ylim_tophist = [0, 1.5]
    ylim_righthist = [0, 1.0]

    xlim = [9.5, 11.5]  # log mstar [msun]
    ylim = [-13.0, -9.0]  # log ssfr [1/yr]
    ylabel = "Galaxy sSFR [ 1/yr ]"
    if config in ["eCGM", "eCGMfull"]:
        xlim = [9.0, 11.5]

    # load survey data
    if config == "COS-Halos":
        datafunc = werk2013
    if config == "eCGM":
        datafunc = johnson2015
    if config == "eCGMfull":
        datafunc = partial(johnson2015, surveys=["IMACS", "SDSS", "COS-Halos"])

    yval_name = "ssfr_30pkpc_log"

    if config in ["COS-Halos", "eCGM", "eCGMfull"]:
        gals, logM, z, sfr, _, yval_limit, R, _, _, _ = datafunc()
        yvals = np.log10(sfr / 10.0**logM)

    if config == "LRG-RDR":
        xlim = [10.6, 12.0]
        colorMinMax = [0.4, 0.6]  # redshift
        ylim_tophist = [0, 3.0]
        gals, logM, z, yvals, yvals_err, yval_limit, R, _, _, _ = berg2019()

    if config == "COS-LRG":
        xlim = [10.6, 12.0]
        colorMinMax = [0.2, 0.5]  # redshift
        ylim_tophist = [0, 2.0]
        ylim_righthist = [0, 9.0]
        ylim = [1.4, 2.0]  # color_ug [mag]
        ylabel = "Galaxy (u-g) Color [mag]"
        yval_name = "color_C-30kpc-z_ug"
        gals, logM, z, yvals, yvals_err, yval_limit, R, _, _, _, _ = chen2018zahedy2019()

    # load obs-matched-samples data
    sim_samples = []
    for sP in sPs:
        sim_samples.append(obsMatchedSample(sP, datasetName=config))

    # plot geometry setup
    left = 0.12
    bottom = 0.12
    width = 0.64
    height = 0.66
    hist_pad = 0.02
    cbar_pad = 0.03

    rect_scatter = [left, bottom, width, height]
    rect_hist_top = [left, bottom + height + hist_pad, width, 1.0 - height - bottom - hist_pad * 2]
    rect_hist_right = [left + width + hist_pad, bottom, 1.0 - left - width - hist_pad * 2, height]
    rect_cbar = [left + cbar_pad, bottom + cbar_pad, 0.03, height / 2]

    # plot setup
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes(rect_scatter)

    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_xlabel(r"Galaxy Stellar Mass [ log M$_{\rm sun}$ ]")

    # plot sim 2d histogram in background
    bbox = ax.get_window_extent()
    nBins2D = np.array([nBinsHist2D, int(nBinsHist2D * (bbox.height / bbox.width))])

    sim_xvals = sim_samples[0]["mstar_30pkpc_log"].ravel()
    sim_yvals = sim_samples[0][yval_name].ravel()
    sim_cvals = np.zeros(sim_xvals.size)

    cc, xBins, yBins, inds = binned_statistic_2d(
        sim_xvals, sim_yvals, sim_cvals, "count", bins=nBins2D, range=[xlim, ylim]
    )

    cc = cc.T  # imshow convention
    cc2d = cc
    if config in ["eCGM", "eCGMfull"]:
        cc2d = logZeroNaN(cc2d)

    cMinMax = [np.nanmax(cc2d) * 0.1, np.nanmax(cc2d) * 1.1]
    norm = Normalize(vmin=cMinMax[0], vmax=cMinMax[1], clip=False)
    cc2d_rgb = cmap2D(norm(cc2d))

    cc2d_rgb[cc == 0.0, :] = colorConverter.to_rgba("white")  # empty bins

    extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
    plt.imshow(cc2d_rgb, extent=extent, origin="lower", interpolation="nearest", aspect="auto")

    # plot obs scatterpoints on top
    # for limitType in [2,1,0]: # upper, lower, exact
    for limitType in [1, 0]:  # upper, exact
        w = np.where(yval_limit == limitType)

        s = ax.scatter(
            logM[w],
            yvals[w],
            s=80,
            marker=["o", "v", "^"][limitType],
            c=z[w],
            label=config if limitType == 0 else "",
            alpha=1.0 if limitType == 0 else detLimitAlpha,
            edgecolors="none",
            cmap=cmap,
            vmin=colorMinMax[0],
            vmax=colorMinMax[1],
        )

    # top histogram: setup
    ax_h1 = fig.add_axes(rect_hist_top)
    ax_h1.set_ylabel("PDF")
    ax_h1.set_xlabel("")
    ax_h1.set_xlim(xlim)
    ax_h1.set_ylim(ylim_tophist)
    ax_h1.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_h1.yaxis.set_major_formatter(ticker.NullFormatter())

    # top histogram: obs and sim
    opts = {"density": True, "histType": "bar", "alpha": 0.5}
    c_obs = "black"

    ax_h1.hist(logM, bins=nBinsHist, range=xlim, color=c_obs, orientation="vertical", label=config, **opts)

    for i, sP in enumerate(sPs):
        loc_mass = sim_samples[i]["mstar_30pkpc_log"].ravel()
        ax_h1.hist(loc_mass, bins=nBinsHist * 4, range=xlim, orientation="vertical", label=sP.simName, **opts)

    ax_h1.legend(bbox_to_anchor=(1.01, 0.96), prop={"size": 18})

    # right histogram: setup
    ax_h2 = fig.add_axes(rect_hist_right)
    ax_h2.set_xlabel("PDF")
    ax_h2.set_ylabel("")
    ax_h2.set_ylim(ylim)
    ax_h2.set_xlim(ylim_righthist)
    ax_h2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax_h2.yaxis.set_major_formatter(ticker.NullFormatter())

    ax_h2.hist(yvals, bins=nBinsHist, range=ylim, color=c_obs, orientation="horizontal", label=config, **opts)

    for i, sP in enumerate(sPs):
        loc_mass = sim_samples[i][yval_name].ravel()
        ax_h2.hist(loc_mass, bins=nBinsHist * 4, range=ylim, orientation="horizontal", label=sP.simName, **opts)

    # colorbar
    cbar_ax = fig.add_axes(rect_cbar)
    cb = fig.colorbar(s, cax=cbar_ax)
    cb.locator = ticker.MaxNLocator(nbins=1)  # nbins = Nticks+1
    cb.update_ticks()
    cb.ax.set_ylabel("Galaxy Redshift", size=cbarTextSize + 5)

    # colorbar labels
    cb.set_ticks([])
    opts = {"ha": "center", "va": "center", "size": cbarTextSize, "transform": cb.ax.transAxes}
    cb.ax.text(0.5, 0.06, "%.1f" % colorMinMax[0], **opts)
    cb.ax.text(0.5, 0.5, "%.1f" % np.mean(colorMinMax), **opts)
    cb.ax.text(0.5, 0.94, "%.1f" % colorMinMax[1], **opts)

    fig.savefig(saveName)
    plt.close(fig)


def obsColumnsDataPlot(sP, saveName, radRelToVirRad=False, config="COS-Halos"):
    """Plot COS-Halos N_OVI data (or other CGM survey data), and our corresponding mock galaxy sample analysis."""
    # load data
    if config == "COS-Halos":
        datafunc = werk2013
    if config == "eCGM":
        datafunc = johnson2015
    if config == "eCGMfull":
        datafunc = partial(johnson2015, surveys=["IMACS", "SDSS", "COS-Halos"])

    yval_label = "Galaxy sSFR [ 1/yr ]"
    yval_name = "ssfr_30pkpc_log"
    ylim = [-12.0, -10.0]

    xlabel = r"Stellar Mass [ log M$_{\rm sun}$ ]"
    xlim = [9.0, 11.2]

    collim = [12.5, 15.5]
    ssfrlim = [-13.0, -9.0]
    blim = [0, 200]

    species = "OVI"

    if config in ["eCGM", "eCGMfull"]:
        blim = [0, 1000]
        collim = [11.5, 15.5]

    if config in ["COS-Halos", "eCGM", "eCGMfull"]:
        gals, logM, z, sfr, _, _, R, col_logN, col_err, col_limit = datafunc()
        yvals = np.log10(sfr / 10.0**logM)

    if config == "LRG-RDR":
        xlim = [10.6, 12.0]
        collim = [12.0, 21.0]
        ssfrlim = [-13.0, -9.0]
        blim = [0, 550]
        species = "HI"
        gals, logM, z, yvals, _, _, R, col_logN, col_err, col_limit = berg2019()

    if config in ["COS-LRG HI", "COS-LRG MgII"]:
        xlim = [10.6, 12.0]
        ylim = [1.4, 2.0]  # color_ug [mag]
        yval_label = "Galaxy (u-g) Color [mag]"
        yval_name = "color_C-30kpc-z_ug"
        gals, logM, z, yvals, _, _, R, N_HI, N_HI_err, N_MgII, N_MgII_err = chen2018zahedy2019()

    if config == "COS-LRG HI":
        species = "HI"
        col_logN = N_HI
    if config == "COS-LRG MgII":
        species = "MgII"
        col_logN = N_MgII

    sim_sample = obsMatchedSample(sP, datasetName=config)
    sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

    for iter in [0, 1]:
        # plot setup
        fig, ax = plt.subplots()

        if iter == 0:
            # x axis = impact parameter
            if radRelToVirRad:
                assert 0  # not implemented yet
                ax.set_xlim([0, 2.0])
                ax.set_xlabel("Projected Distance / Virial Radius")
                if config in ["eCGM", "eCGMfull"]:
                    ax.set_xlim([0, 10])
            else:
                ax.set_xlim(blim)
                ax.set_xlabel("Projected Distance [ pkpc ]")

        if iter == 1:
            # x axis = sSFR
            ax.set_xlim(ssfrlim)
            ax.set_xlabel(yval_label)

        ax.set_ylim(collim)
        ax.set_ylabel(r"Column Density $N_{\rm %s}$ [ log cm$^{-2}$ ]" % species)

        # plot obs
        for limitType in [2, 1, 0]:  # upper, lower, exact
            w = np.where(col_limit == limitType)

            label = config if limitType == 0 else ""
            marker = ["o", "v", "^"][limitType]

            if iter == 0:
                x_vals = R[w]
                c_vals = yvals[w]
                c_label = yval_label
                colorMinMax = ylim
                cmap = loadColorTable("RdBu")  # 'coolwarm_r'

                # clip obs to two binary colors at the maxima of the scale to avoid
                # apparent brightness changes towards the center
                if config in ["LRG-RDR"]:
                    c_vals[c_vals <= yvals.mean()] = colorMinMax[0]
                    c_vals[c_vals > yvals.mean()] = colorMinMax[1]
            if iter == 1:
                x_vals = yvals[w]
                c_vals = logM[w]
                c_label = xlabel
                colorMinMax = xlim
                cmap = loadColorTable("RdBu")  # 'coolwarm'

                # clip obs to two binary colors as above
                if config in ["LRG-RDR"]:
                    c_vals[c_vals <= logM.mean()] = colorMinMax[0]
                    c_vals[c_vals > logM.mean()] = colorMinMax[1]

            y_vals = col_logN[w]

            s = ax.scatter(
                x_vals,
                y_vals,
                s=80,
                marker=marker,
                c=c_vals,
                label=label,
                alpha=1.0,
                edgecolors="none",
                cmap=cmap,
                vmin=colorMinMax[0],
                vmax=colorMinMax[1],
            )

        # plot sim
        if iter == 0:
            x_vals = sim_sample["impact_parameter"].ravel()
            c_vals = sim_sample[yval_name].ravel()
            c_label = yval_label
            colorMinMax = ylim
            cmap = loadColorTable("RdBu")  # 'coolwarm_r'
        if iter == 1:
            x_vals = sim_sample[yval_name].ravel()
            c_vals = sim_sample["mstar_30pkpc_log"].ravel()
            c_label = xlabel
            colorMinMax = xlim
            cmap = loadColorTable("RdBu")  # 'coolwarm'

        y_vals = sim_sample["column"].ravel()

        s = ax.scatter(
            x_vals,
            y_vals,
            s=10,
            marker="s",
            c=c_vals,
            label=sP.simName,
            alpha=0.3,
            edgecolors="none",
            cmap=cmap,
            vmin=colorMinMax[0],
            vmax=colorMinMax[1],
        )

        # legend
        ax.legend(loc="upper right")

        # colorbar
        cb = fig.colorbar(s, ax=ax, pad=0)
        cb.ax.set_ylabel(c_label)

        cb.set_alpha(1)  # fix stripes
        cb.draw_all()

        fig.savefig(saveName.split(".pdf")[0] + "_v%d.pdf" % iter)
        plt.close(fig)


def obsColumnsDataPlotExtended(sP, saveName, config="COS-Halos"):
    """Plot COS-Halos N_OVI data, and our mock COS-Halos galaxy sample analysis, with stacked offset 1d KDEs."""
    cbarTextSize = 13
    nKDE1D = 100
    kdeHeightFac = 4.0  # multiplicative horizontal size beyond individual bounds
    if config in ["eCGM", "eCGMfull"]:
        kdeHeightFac = 10.0

    xlabel = r"Stellar Mass [ log M$_{\rm sun}$ ]"
    xlim = [9.0, 11.2]

    # geometry
    left = 0.06
    bottom = 0.12
    width = 0.35
    height = 0.84
    hist_pad = 0.02
    cbar_pad = 0.015
    cbar_width = 0.02

    # load data
    if config == "COS-Halos":
        datafunc = werk2013
    if config == "eCGM":
        datafunc = johnson2015
    if config == "eCGMfull":
        datafunc = partial(johnson2015, surveys=["IMACS", "SDSS", "COS-Halos"])

    collim = [12.5, 15.5]
    if config in ["eCGM", "eCGMfull"]:
        collim = [11.5, 15.5]
    # ssfrlim = [-13.0, -9.0]
    blim = [0, 200]

    species = "OVI"
    yval_label = "Galaxy sSFR [ 1/yr ]"
    yval_name = "ssfr_30pkpc_log"
    ylim = [-14.0, -10.0]  # colorbar on ext0
    ylim2 = ylim  # x-axis on ext1

    # clip obs to two binary colors at the maxima of the scale to avoid apparent brightness changes towards the center?
    clipObsColorsToExtrema = True

    simSquaresAlpha = 0.6  # 0.3 for oxygen paper

    if config in ["COS-Halos", "eCGM", "eCGMfull"]:
        gals, logM, z, sfr, _, yvals_limit, R, col_logN, col_err, col_limit = datafunc()
        yvals = np.log10(sfr / 10.0**logM)

    if config == "LRG-RDR":
        xlim = [10.6, 12.0]
        collim = [12.0, 21.0]
        ylim = [-12.0, -10.0]
        ylim2 = [-13.0, -9.0]
        xlim = [10.6, 12.0]
        blim = [0, 580]
        species = "HI"
        cbar_width = 0.02
        cbar_pad = 0.009
        gals, logM, z, yvals, _, yvals_limit, R, col_logN, col_err, col_limit = berg2019()

    if config in ["COS-LRG HI", "COS-LRG MgII"]:
        xlim = [10.6, 12.0]
        ylim = [1.4, 2.0]  # color_ug [mag]
        ylim2 = [1.35, 1.95]  # color_ug [mag]
        yval_label = "Galaxy (u-g) Color [mag]"
        yval_name = "color_C-30kpc-z_ug"
        gals, logM, z, yvals, _, yvals_limit, R, N_HI, N_HI_err, N_MgII, N_MgII_err = chen2018zahedy2019()

    if config == "COS-LRG HI":
        species = "HI"
        collim = [12.0, 21.0]
        col_logN = N_HI
        col_err = N_HI_err

    if config in ["COS-LRG MgII"]:
        species = "MgII"
        collim = [10.0, 17.0]
        col_logN = N_MgII
        col_err = N_MgII_err

    if config in ["COS-LRG HI", "COS-LRG MgII"]:
        # generate col_limit, including '-' (missing data) cases
        col_limit = np.zeros(col_logN.size, dtype="int32")
        col_limit[col_err == "<"] = 1
        col_limit[col_err == ">"] = 2
        col_limit[col_err == "-"] = 3  # missing data
        col_err[col_limit != 0] = "0.0"
        col_err = np.array(col_err, dtype="float32")

    ylabel = r"Column Density $N_{\rm %s}$ [ log cm$^{-2}$ ]" % species

    rect_mainpanel = [left, bottom, width, height]
    rect_right = [left + width + hist_pad, bottom, 1.0 - left - width - hist_pad * 2, height]
    rect_cbar1 = [
        left + width - cbar_pad * 2 - cbar_width,
        bottom + cbar_pad * (height / width),
        cbar_width,
        height / 2,
    ]
    rect_cbar2 = [left + width / 2 - width / 4, bottom + cbar_pad * 4, width / 2, cbar_width * 2]

    sim_sample = obsMatchedSample(sP, datasetName=config.split(" ")[0])  # e.g. "COS-LRG HI" -> "COS-LRG"
    sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

    for ind in range(sim_sample["impact_parameter"].shape[0]):
        x_vals = sim_sample["impact_parameter"][ind, :].ravel()
        diff = x_vals.mean() - R[ind]
        assert np.abs(diff) < 1.0  # make sure we are matched and not mixed up

    for iter in [0, 1]:
        # plot setup
        fig = plt.figure(figsize=[figsize[0] * 2.0, figsize[1]])

        ax = fig.add_axes(rect_mainpanel)

        if iter == 0:
            # x axis = impact parameter, color=sSFR
            ax.set_xlim(blim)
            if config in ["eCGM", "eCGMfull"]:
                ax.set_xlim([0, 1000])
            ax.set_xlabel("Projected Distance [ pkpc ]")

            c_label = yval_label
            colorMinMax = ylim
            cmap = loadColorTable("RdBu")  # coolwarm_r

        if iter == 1:
            # x axis = sSFR, color=Mstar
            ax.set_xlim(ylim2)
            ax.set_xlabel(yval_label)

            c_label = xlabel
            colorMinMax = xlim
            cmap = loadColorTable("RdBu")  # coolwarm

        ax.set_ylim(collim)
        ax.set_ylabel(ylabel)

        # setup right panel
        ax_right = fig.add_axes(rect_right)

        ax_right.set_ylim(collim)
        ax_right.set_xlim([0, 1])
        ax_right.xaxis.set_major_formatter(ticker.NullFormatter())
        ax_right.yaxis.set_major_formatter(ticker.NullFormatter())

        kde_inv_horz_spacing = 1.5
        if "COS-LRG MgII" in config:
            kde_inv_horz_spacing = 4.0
        if "LRG-RDR" in config:
            kde_inv_horz_spacing = 3.0

        xticks_max = 1.0 - (1.0 / (len(gals) + 2)) * (kdeHeightFac - kde_inv_horz_spacing)
        xtick_vals = np.linspace(0.0, xticks_max, len(gals) + 2)
        ax_right.set_xticks([])

        # ax_right.spines["bottom"].set_visible(False)
        # ax_right.spines["top"].set_visible(False)

        # main panel: plot obs
        for limitType in [2, 1, 0, 3]:  # upper, lower, exact, missing data (species column density)
            w = np.where(col_limit == limitType)

            if not len(w[0]):
                continue

            label = config.split(" ")[0] if limitType == 0 else ""
            marker = ["o", "v", "^", "s"][limitType]

            if iter == 0:
                x_vals = R[w]
                c_vals = yvals[w]

                if clipObsColorsToExtrema:
                    c_vals[c_vals <= yvals.mean()] = colorMinMax[0]
                    c_vals[c_vals > yvals.mean()] = colorMinMax[1]
            if iter == 1:
                x_vals = yvals[w]
                c_vals = logM[w]

                if clipObsColorsToExtrema:
                    c_vals[c_vals <= logM.mean()] = colorMinMax[0]
                    c_vals[c_vals > logM.mean()] = colorMinMax[1]

            y_vals = col_logN[w]

            # add points to main panel
            s = ax.scatter(
                x_vals,
                y_vals,
                s=120,
                marker=marker,
                c=c_vals,
                label=label,
                alpha=1.0,
                edgecolors="none",
                cmap=cmap,
                vmin=colorMinMax[0],
                vmax=colorMinMax[1],
            )

            if iter == 1:
                # x-axis values could also be limits: replicate markers into list and adjust
                ww = np.where(yvals_limit[w])
                x_off = 0.14

                if len(ww[0]):
                    s = ax.scatter(
                        x_vals[ww] - x_off,
                        y_vals[ww],
                        s=50,
                        marker="<",
                        c=c_vals[ww],
                        alpha=1.0,
                        edgecolors="none",
                        cmap=cmap,
                        vmin=colorMinMax[0],
                        vmax=colorMinMax[1],
                    )

                    norm = Normalize(vmin=colorMinMax[0], vmax=colorMinMax[1], clip=False)
                    for i in ww[0]:
                        ax.plot(
                            [x_vals[i], x_vals[i] - x_off], [y_vals[i], y_vals[i]], "-", color=cmap(norm(c_vals[i]))
                        )
                # for i in ww[0]:
                #    print(x_vals[i],y_vals[i])
                #    ax.annotate(s='Q', xy=(x_vals[i], y_vals[i]), xytext=(x_vals[i]-0.5, y_vals[i]),
                #                arrowprops=dict(arrowstyle='->'))
                #    #ax.errorbar(x_vals, y_vals,
                #    #    xuplims=ww, #xlolims=xlolims, uplims=uplims, lolims=lolims,
                #    #    marker='o', markersize=8,linestyle='-')

        # main panel: plot simulation
        if iter == 0:
            x_vals = sim_sample["impact_parameter"].ravel()
            c_vals = sim_sample[yval_name].ravel()
        if iter == 1:
            x_vals = sim_sample[yval_name].ravel()
            c_vals = sim_sample["mstar_30pkpc_log"].ravel()

        y_vals = sim_sample["column"].ravel()

        s = ax.scatter(
            x_vals,
            y_vals,
            s=10,
            marker="s",
            c=c_vals,
            label=sP.simName,
            alpha=simSquaresAlpha,
            edgecolors="none",
            cmap=cmap,
            vmin=colorMinMax[0],
            vmax=colorMinMax[1],
            rasterized=True,
        )  # rasterize the small squares into a single image

        # right panel: sort by order along x-axis of main panel
        if iter == 0:
            sort_inds = np.argsort(R)
        if iter == 1:
            sort_inds = np.argsort(yvals)

        xx = xtick_vals[1:-1]
        xx_spacing = xx[1] - xx[0]

        # save some data for later
        pvals = np.zeros(R.size, dtype="float32")
        plims = np.zeros(R.size, dtype="int32") - 1
        pvals.fill(np.nan)

        for i, sort_ind in enumerate(sort_inds):
            # plot vertical line and obs. galaxy name
            ax_right.plot([xx[i], xx[i]], collim, "-", color="black", alpha=0.04)

            textOpts = {"ha": "center", "va": "bottom", "rotation": 90, "color": "#555555", "fontsize": 8}
            ax_right.text(xx[i], collim[0], gals[sort_ind]["name"], **textOpts)

            # plot sim 1D KDE
            sim_cols = np.squeeze(sim_sample["column"][sort_ind, :])
            sim_cols = sim_cols[np.isfinite(sim_cols)]

            if len(sim_cols) == 0:
                print(gals[sort_ind]["name"], " skipped, all NaN values...")
                continue

            # KDE is heavily skewed/flattened if there are any distant outliers
            # note: important impact on MgII (not on HI) where the majority of grid samples are effectively N_MgII==0
            if 1:
                print("WARNING: clipping sim columns to column limit range (consider implications.)")  # we label
                wColLim = np.where((sim_cols >= collim[0]) & (sim_cols < collim[1]))
                frac_above = len(wColLim[0]) / sim_cols.size * 100

                opts = {"ha": "center", "va": "top", "color": "#000000", "fontsize": 12}
                ax_right.text(
                    xx[i] + xx_spacing * 0.5, collim[1] - (collim[1] - collim[0]) / 100, "%d%%" % frac_above, **opts
                )
            else:
                wColLim = np.where(np.isfinite(sim_cols))  # all

            kde_x = np.linspace(collim[0], collim[1], nKDE1D)
            kde = gaussian_kde(sim_cols[wColLim], bw_method="scott")
            kde_y = kde(kde_x) * (1.0 / len(gals)) * kdeHeightFac

            (l,) = ax_right.plot(kde_y + xx[i], kde_x, "-", alpha=1.0, lw=lw)
            ax_right.fill_betweenx(kde_x, xx[i], kde_y + xx[i], facecolor=l.get_color(), alpha=0.05)

            # locate 'height' of observed point beyond xx[i], i.e. the KDE value at its log_N
            _, kde_ind_obs = closest(kde_x, col_logN[sort_ind])

            # if we have an observed column
            if col_limit[sort_ind] in [0, 1, 2]:
                # mark observed data point
                marker = ["o", "v", "^"][col_limit[sort_ind]]
                ax_right.plot(
                    xx[i] + kde_y[kde_ind_obs],
                    col_logN[sort_ind],
                    markersize=12,
                    marker=marker,
                    color=l.get_color(),
                    alpha=1.0,
                )

                # add observational error as vertical line
                if col_err[sort_ind] > 0.0:
                    obs_xerr = [xx[i], xx[i]] + kde_y[kde_ind_obs]
                    obs_yerr = [col_logN[sort_ind] - col_err[sort_ind], col_logN[sort_ind] + col_err[sort_ind]]
                    ax_right.plot(obs_xerr, obs_yerr, "-", color=l.get_color(), alpha=1.0)

                # calculate and print a quantitative probability number
                z1 = kde.integrate_box_1d(-np.inf, col_logN[sort_ind])
                z2 = kde.integrate_box_1d(col_logN[sort_ind], np.inf)

                if col_limit[sort_ind] == 0:
                    pvals[i] = 2 * np.min([z1, z2])  # detection, 2*PDF area more extreme
                if col_limit[sort_ind] == 1:
                    pvals[i] = z1  # upper limit, PDF area which is consistent
                if col_limit[sort_ind] == 2:
                    pvals[i] = z2  # lower limit, PDF area which is consistent
                plims[i] = col_limit[sort_ind]

                if iter == 1:
                    print(gals[sort_ind]["name"], col_logN[sort_ind], pvals[i])
            else:
                if iter == 1:
                    print(gals[sort_ind]["name"], " no obs column")
                marker = "s"
                ax_right.plot(xx[i] + kde_y[-1], collim[1], markersize=8, marker=marker, color=l.get_color(), alpha=0.5)

        # print summary of pvals statistic
        if iter == 1:
            percs = np.nanpercentile(pvals, [16, 50, 84])
            print("all percentiles: ", percs)
            print("NOTE: for MgII p-values MgII, take with the KDE clipping off (to preserve the statistical meaning)")
            for lim in [0, 1, 2]:
                w = np.where(plims == lim)
                percs = np.nanpercentile(pvals[w], [16, 50, 84])
                print("limit [%d] percs:" % lim, percs)
            print("counts: ", np.count_nonzero(pvals < 0.05), np.count_nonzero(pvals < 0.01), pvals.size)

            # print summary of sim vs. obs mean/1sigma column densities
            percs = np.nanpercentile(col_logN, [16, 50, 84])
            print("obs logN: %.2f (-%.2f +%.2f)" % (percs[1], percs[1] - percs[0], percs[2] - percs[1]))
            percs = np.nanpercentile(sim_sample["column"].ravel(), [16, 50, 84])
            print("sim logN: %.2f (-%.2f +%.2f)" % (percs[1], percs[1] - percs[0], percs[2] - percs[1]))

            if config in ["COS-Halos", "eCGM", "eCGMfull"]:
                w_sf = np.where(yvals >= -11.0)
                w_qq = np.where(yvals < -11.0)
                percs_sf = np.nanpercentile(col_logN[w_sf], [16, 50, 84])
                percs_qq = np.nanpercentile(col_logN[w_qq], [16, 50, 84])
                print(
                    "obs SF logN: %.2f (-%.2f +%.2f)"
                    % (percs_sf[1], percs_sf[1] - percs_sf[0], percs_sf[2] - percs_sf[1])
                )
                print(
                    "obs QQ logN: %.2f (-%.2f +%.2f)"
                    % (percs_qq[1], percs_qq[1] - percs_qq[0], percs_qq[2] - percs_qq[1])
                )
                percs_sf = np.nanpercentile(sim_sample["column"][w_sf, :].ravel(), [16, 50, 84])
                percs_qq = np.nanpercentile(sim_sample["column"][w_qq, :].ravel(), [16, 50, 84])
                print(
                    "sim SF logN: %.2f (-%.2f +%.2f)"
                    % (percs_sf[1], percs_sf[1] - percs_sf[0], percs_sf[2] - percs_sf[1])
                )
                print(
                    "sim QQ logN: %.2f (-%.2f +%.2f)"
                    % (percs_qq[1], percs_qq[1] - percs_qq[0], percs_qq[2] - percs_qq[1])
                )

        # main panel: legend
        loc = ["upper right", "upper left"][iter]
        legend2 = ax.legend(loc=loc, markerscale=1.8)
        legend2.legendHandles[0].set_color("#000000")
        legend2.legendHandles[1].set_color("#000000")

        # colorbar
        cbar_ax = fig.add_axes(rect_cbar1 if iter == 0 else rect_cbar2)
        cb = fig.colorbar(s, cax=cbar_ax, orientation=["vertical", "horizontal"][iter])
        cb.set_alpha(1)  # fix stripes
        cb.draw_all()

        cb.locator = ticker.MaxNLocator(nbins=1)  # nbins = Nticks+1
        cb.update_ticks()
        if iter == 0:
            cb.ax.set_ylabel(c_label, size=cbarTextSize + 3)
        if iter == 1:
            cb.ax.set_xlabel(c_label, size=cbarTextSize + 3, labelpad=4)

        # colorbar labels
        cb.set_ticks([])
        if iter == 0:
            # vertical, custom labeling
            cbx = [0.5, 0.5, 0.5]
            cby = [0.06, 0.5, 0.94]
        if iter == 1:
            # horizontal, custom labeling
            cbx = [0.06, 0.5, 0.94]
            cby = [0.5, 0.5, 0.5]

        opts = {"ha": "center", "va": "center", "size": cbarTextSize, "transform": cb.ax.transAxes}
        # cb.ax.text(0.5, 0.06, '%.1f' % colorMinMax[0], **opts)
        cb.ax.text(cbx[0], cby[0], "%.1f" % colorMinMax[0], **opts)
        cb.ax.text(cbx[1], cby[1], "%.1f" % np.mean(colorMinMax), **opts)
        cb.ax.text(cbx[2], cby[2], "%.1f" % colorMinMax[1], **opts)

        fig.savefig(saveName.split(".pdf")[0] + "_v%d.pdf" % iter)
        plt.close(fig)

        if iter == 1:
            continue

        # supplementary figure: lambda vs R
        fig = plt.figure(figsize=[figsize[0] * 0.7, figsize[1] * 0.5])
        ax = fig.add_subplot(111)

        ax.set_xlim([0, 550])
        ax.set_ylim([0.0, 1])
        ax.set_xlabel("b [ pkpc ]")
        ax.set_ylabel("$\\lambda$")

        cnum = 1
        if config == "LRG-RDR":
            cnum = 2
        if config == "COS-LRG HI":
            cnum = 3

        # ax.plot(blim, [0.0,0.0], '-', lw=lw-1, color='black', alpha=0.1)
        # ax.plot(blim, [0.5,0.5], '-', lw=lw-1, color='black', alpha=0.1)
        ax.fill_between([0, 550], [0.0, 0.0], [0.5, 0.5], color="#cccccc", alpha=0.2)

        ax.plot(R, pvals, "o", color=f"C{cnum}", label=config)

        ax.legend(loc="upper left")
        fig.savefig(saveName.replace("_ext", "_lambdaVsR"))
        plt.close(fig)


def obsColumnsLambdaVsR(sP, saveName, configs="COS-Halos"):
    """Plot statistical lambda value(s) as a function of impact parameter."""
    blim = [0, 550]

    # start plot
    fig = plt.figure(figsize=[figsize[0] * 0.7, figsize[1] * 0.5])
    ax = fig.add_subplot(111)

    ax.set_xlim(blim)
    ax.set_ylim([0, 1])
    ax.set_xlabel("b [ pkpc ]")
    ax.set_ylabel("$\\lambda$")

    # ax.plot(blim, [0.0,0.0], '-', lw=lw-1, color='black', alpha=0.1)
    # ax.plot(blim, [0.5,0.5], '-', lw=lw-1, color='black', alpha=0.1)
    ax.fill_between(blim, [0.0, 0.0], [0.5, 0.5], color="#cccccc", alpha=0.2)

    # loop over requested configs
    colors = []

    for config in configs:
        # load data
        if config == "COS-Halos":
            datafunc = werk2013
        if config == "eCGM":
            datafunc = johnson2015
        if config == "eCGMfull":
            datafunc = partial(johnson2015, surveys=["IMACS", "SDSS", "COS-Halos"])

        if config in ["COS-Halos", "eCGM", "eCGMfull"]:
            gals, logM, z, sfr, _, yvals_limit, R, col_logN, col_err, col_limit = datafunc()
            yvals = np.log10(sfr / 10.0**logM)

        if config == "LRG-RDR":
            collim = [12.0, 21.0]
            gals, logM, z, yvals, _, yvals_limit, R, col_logN, col_err, col_limit = berg2019()

        if config in ["COS-LRG HI", "COS-LRG MgII"]:
            gals, logM, z, yvals, _, yvals_limit, R, N_HI, N_HI_err, N_MgII, N_MgII_err = chen2018zahedy2019()

        if config == "COS-LRG HI":
            collim = [12.0, 21.0]
            col_logN = N_HI
            col_err = N_HI_err

        if config in ["COS-LRG MgII"]:
            collim = [10.0, 17.0]
            col_logN = N_MgII
            col_err = N_MgII_err

        if config in ["COS-LRG HI", "COS-LRG MgII"]:
            # generate col_limit, including '-' (missing data) cases
            col_limit = np.zeros(col_logN.size, dtype="int32")
            col_limit[col_err == "<"] = 1
            col_limit[col_err == ">"] = 2
            col_limit[col_err == "-"] = 3  # missing data
            col_err[col_limit != 0] = "0.0"
            col_err = np.array(col_err, dtype="float32")

        sim_sample = obsMatchedSample(sP, datasetName=config.split(" ")[0])  # e.g. "COS-LRG HI" -> "COS-LRG"
        sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)

        for ind in range(sim_sample["impact_parameter"].shape[0]):
            x_vals = sim_sample["impact_parameter"][ind, :].ravel()
            diff = x_vals.mean() - R[ind]
            assert np.abs(diff) < 1.0  # make sure we are matched and not mixed up

        # right panel: sort by order along x-axis of main panel
        sort_inds = np.argsort(R)

        # save some data for later
        pvals = np.zeros(R.size, dtype="float32")
        plims = np.zeros(R.size, dtype="int32") - 1
        pvals.fill(np.nan)

        for i, sort_ind in enumerate(sort_inds):
            # plot sim 1D KDE
            sim_cols = np.squeeze(sim_sample["column"][sort_ind, :])
            sim_cols = sim_cols[np.isfinite(sim_cols)]

            if len(sim_cols) == 0:
                print(gals[sort_ind]["name"], " skipped, all NaN values...")
                continue

            # KDE is heavily skewed/flattened if there are any distant outliers
            # note: important impact on MgII (no impact on HI) where the majority of grid samples are N_MgII~=0
            if 1:
                print("WARNING: clipping sim columns to column limit range (consider implications.)")  # we label
                wColLim = np.where((sim_cols >= collim[0]) & (sim_cols < collim[1]))
            else:
                wColLim = np.where(np.isfinite(sim_cols))  # all

            kde = gaussian_kde(sim_cols[wColLim], bw_method="scott")

            # if we have an observed column
            if col_limit[sort_ind] in [0, 1, 2]:
                # calculate and print a quantitative probability number
                z1 = kde.integrate_box_1d(-np.inf, col_logN[sort_ind])
                z2 = kde.integrate_box_1d(col_logN[sort_ind], np.inf)

                if col_limit[sort_ind] == 0:
                    pvals[i] = 2 * np.min([z1, z2])  # detection, 2*PDF area more extreme
                if col_limit[sort_ind] == 1:
                    pvals[i] = z1  # upper limit, PDF area which is consistent
                if col_limit[sort_ind] == 2:
                    pvals[i] = z2  # lower limit, PDF area which is consistent
                plims[i] = col_limit[sort_ind]

        # supplementary figure: lambda vs R
        (l,) = ax.plot(R, pvals, "o", label=config)
        colors.append(l.get_color())

    # finish
    l = ax.legend(markerscale=0, handletextpad=-2.0, loc="upper right")
    for i, text in enumerate(l.get_texts()):
        text.set_color(colors[i])

    fig.savefig(saveName)
    plt.close(fig)


def coveringFractionVsDist(sPs, saveName, ions, colDensThresholds, config="COS-Halos", conf=0):
    """Covering fraction of OVI versus impact parameter.

    Show either COS-Halos data versus mock simulated sample, or physics variations with respect to fiducial model.
    colDensThresholds is a list in [1/cm^2] to compute.
    """
    assert len(ions) == 1
    assert ions[0] == "OVI"

    gsNames = {
        "all": "All Galaxies",
        "mstar_lt_105": r"$M_\star < 10^{10.5} \,$M$_{\!\odot}$",
        "mstar_gt_105": r"$M_\star > 10^{10.5} \,$M$_{\!\odot}$",
        "ssfr_lt_n11": r"sSFR < 10$^{-11}$ yr$^{-1}$",
        "ssfr_gt_n11": r"sSFR > 10$^{-11}$ yr$^{-1}$",
        "ssfr_lt_n11_I": r"sSFR < 10$^{-11}$ yr$^{-1}$ (I)",
        "ssfr_lt_n11_NI": r"sSFR < 10$^{-11}$ yr$^{-1}$ (NI)",
        "ssfr_gt_n11_I": r"sSFR > 10$^{-11}$ yr$^{-1}$ (I)",
        "ssfr_gt_n11_NI": r"sSFR > 10$^{-11}$ yr$^{-1}$ (NI)",
    }

    if config == "COS-Halos":
        werk13 = werk2013(coveringFractions=True)

        if conf == 0:
            galaxySets = ["all"]
        if conf == 1:
            galaxySets = ["ssfr_gt_n11", "ssfr_lt_n11", "mstar_lt_105", "mstar_gt_105"]
    if config in ["eCGM", "eCGMfull"]:
        j15 = johnson2015(coveringFractions=True)

        if conf == 0:
            galaxySets = ["all"]
        if conf == 1:
            galaxySets = ["ssfr_gt_n11", "ssfr_lt_n11"]
        if conf == 2:
            galaxySets = ["ssfr_gt_n11_I", "ssfr_lt_n11_I", "ssfr_gt_n11_NI", "ssfr_lt_n11_NI"]
    if config == "SimHalos_115-125":
        galaxySets = ["all"]

    # plot setup
    lw = 3.0
    heightFac = 1.0 if ("main" in saveName or len(sPs) == 1) else 0.95
    fig = plt.figure(figsize=[figsize[0], figsize[1] * heightFac])
    ax = fig.add_subplot(111)

    if config in ["COS-Halos", "SimHalos_115-125"]:
        ax.set_xlim([0.0, 400.0])
        ax.set_xlabel("Impact Parameter [ pkpc ]")
    if config in ["eCGM", "eCGMfull"]:
        ax.set_xlim([-1.05, 1.05])
        ax.set_xlabel("Impact Parameter / Virial Radius [ log ]")

    yLabelExtra = ""
    if len(colDensThresholds) == 1:
        yLabelExtra = r" (N$_{\rm OVI}$ > 10$^{%.2f}$ cm$^{-2}$)" % colDensThresholds[0]
        if int(colDensThresholds[0] * 10) / 10.0 == colDensThresholds[0]:
            yLabelExtra = r" (N$_{\rm OVI}$ > 10$^{%.1f}$ cm$^{-2}$)" % colDensThresholds[0]
        if int(colDensThresholds[0] * 1) / 1.0 == colDensThresholds[0]:
            yLabelExtra = r" (N$_{\rm OVI}$ > 10$^{%.0f}$ cm$^{-2}$)" % colDensThresholds[0]
    ax.set_ylabel(r"Covering Fraction $\kappa_{\rm OVI}$%s" % yLabelExtra)
    if conf == 0:
        ax.set_ylim([0.0, 1.04])
    if conf == 1:
        ax.set_ylim([0.1, 1.04])
    if config in ["eCGM", "eCGMfull"]:
        ax.set_ylim([-0.1, 1.04])

    # overplot obs
    colors_loc = ["black"] if len(galaxySets) == 1 else colors

    opts = {"fmt": "o", "ms": 11, "lw": 1.6, "capthick": 1.6}

    if config == "COS-Halos":
        for j, gs in enumerate(galaxySets):
            for i in range(len(werk13["rad"])):
                x = np.mean(werk13["rad"][i]) + [0, 4, 0, 4][j]  # horizontal offset for visual clarity
                y = werk13[gs]["cf"][i]
                xerr = (x - werk13["rad"][i][0]) * 0.96  # reduce a few percent for clarity
                yerr = np.array([werk13[gs]["cf_errdown"][i], werk13[gs]["cf_errup"][i]])
                yerr = np.reshape(yerr / 100.0, (2, 1))

                print(gs, i, x, xerr, y)

                label = ""
                if gs != "all":
                    label = "Werk+ (2013) " + gsNames[gs]
                if gs == "all":
                    ax.text(124, 0.64, "Werk+ (2013)", ha="left", size=18)
                    ax.text(124, 0.59, r"N$_{\rm OVI}$ > 10$^{14.15}$ cm$^{-2}$", ha="left", size=18)
                if i > 0:
                    label = ""

                ax.errorbar(x, y / 100.0, xerr=xerr, yerr=yerr, color=colors_loc[j], label=label, **opts)

    if config in ["eCGM", "eCGMfull"]:
        for j, gs in enumerate(galaxySets):
            if len(j15[gs]) == 0:
                continue  # no data points (all)

            for i in range(len(j15[gs]["rad"])):
                x = np.log10(j15[gs]["rad"][i])
                y = j15[gs]["cf"][i]

                xerr = [x - np.log10(j15[gs]["rad_left"][i]), np.log10(j15[gs]["rad_right"][i]) - x]
                xerr = np.reshape(xerr, (2, 1))
                yerr = [y - j15[gs]["cf_down"][i], j15[gs]["cf_up"][i] - y]
                yerr = np.reshape(yerr, (2, 1))

                label = ""
                if gs != "all":
                    label = "Johnson+ (2015) " + gsNames[gs]
                if i > 0:
                    label = ""

                ax.errorbar(x, y, xerr=xerr, yerr=yerr, color=colors_loc[j], label=label, **opts)

    # loop over each column density threshold (different colors)
    for _j, thresh in enumerate(colDensThresholds):
        # loop over each fullbox run (different linestyles)
        for i, sP in enumerate(sPs):
            # load
            sim_sample = obsMatchedSample(sP, datasetName=config)
            sim_sample = addIonColumnPerSystem(sP, sim_sample, config=config)
            cf = ionCoveringFractions(sP, sim_sample, config=config)

            print("[%s]: %s" % (sP.simName, thresh))

            # which index for the requested col density threshold?
            assert thresh in cf["colDensThresholds"]
            ind = np.where(cf["colDensThresholds"] == thresh)[0]

            relStr = ""
            if config in ["eCGM", "eCGMfull"]:
                relStr = "_rel"

            # different galaxy samples?
            for k, gs in enumerate(galaxySets):
                xx = cf["radBins%s" % relStr]
                yy = np.squeeze(cf["%s_percs%s" % (gs, relStr)][ind, :, 3])
                yy_min = np.squeeze(cf["%s_percs%s" % (gs, relStr)][ind, :, 2])  # -half sigma
                yy_max = np.squeeze(cf["%s_percs%s" % (gs, relStr)][ind, :, 4])  # +half sigma
                assert list(cf["perc_vals"][[3, 2, 4]]) == [50, 38, 62]  # verify as expected

                # plot middle line
                label = "N > %.2f cm$^{-2}$" % thresh
                if int(thresh * 10) / 10.0 == thresh:
                    label = "N > %.1f cm$^{-2}$" % thresh
                if len(colDensThresholds) == 1:
                    label = sP.simName
                if gs != "all":
                    label += " (%s)" % gsNames[gs]
                if i > 0 and len(colDensThresholds) > 1:
                    label = ""

                if len(galaxySets) > 1:
                    c = colors[k]
                else:
                    c = "black" if (len(sPs) > 5 and sP.variant == "0000") else None

                ls = linestyles[i] if len(colDensThresholds) > 1 else linestyles[0]
                if len(sPs) > 8 and "BH" in sP.simName:
                    ls = "--"

                ax.plot(xx, yy, lw=lw, color=c, linestyle=ls, label=label)

                # percentiles
                if i != len(sPs) - 1:
                    continue

                ax.fill_between(xx, yy_min, yy_max, color=c, alpha=0.1, interpolate=True)

                # add TNG100-2 (i.e. TNG300-1) line to COS-Halos figure?
                if gs == "all" and sP.simName == "TNG100-1" and thresh == 14.15:
                    print("add")
                    sP_ill = simParams(res=910, run="tng", redshift=0.0)
                    sim_sample_ill = obsMatchedSample(sP_ill, datasetName=config)
                    sim_sample_ill = addIonColumnPerSystem(sP_ill, sim_sample_ill, config=config)
                    cf_ill = ionCoveringFractions(sP_ill, sim_sample_ill, config=config)
                    xx = cf_ill["radBins%s" % relStr]
                    yy = np.squeeze(cf_ill["%s_percs%s" % (gs, relStr)][ind, :, 3])
                    ax.plot(xx, yy, lw=lw, linestyle="--", color=c)
                    ax.text(245, 0.33, "TNG300", color=c, fontsize=18)
                    ax.text(345, 0.29, "TNG100", color=c, fontsize=18)

    # legend
    handles, labels = ax.get_legend_handles_labels()

    if config == "eCGMfull" and conf == 2:
        prop = {"size": 15}
        legend1 = ax.legend(handles[0:4], labels[0:4], loc="upper right", prop=prop)
        ax.legend(handles[4:], labels[4:], loc="lower left", prop=prop)
        ax.add_artist(legend1)
    else:
        if len(sPs) == 13:  # main variants, split into 2 legends
            legend1 = ax.legend(handles[0:5], labels[0:5], loc="upper right", prop={"size": 18})
            ax.add_artist(legend1)
            ax.legend(handles[5:], labels[5:], loc="lower left", prop={"size": 18})
        else:  # default
            loc = "upper right" if len(sPs) == 1 else "lower left"
            prop = {}
            if config in ["eCGM", "eCGMfull"]:
                prop["size"] = 15
            ax.legend(handles, labels, loc=loc, ncol=1, prop=prop)

    fig.savefig(saveName)
    plt.close(fig)


def test_lambda_statistic():
    """Test the behavior of the lambda statistic depending on the sim vs obs draws."""
    # config
    N_sim = 100
    loc = 15.0
    scale = 0.8
    sim_cols = np.random.normal(loc=loc, scale=scale, size=N_sim)
    kde = gaussian_kde(sim_cols, bw_method="scott")

    # obs
    N_obs = 10000
    scale_fac = 0.5

    obs_cols = np.random.normal(loc=loc, scale=scale * scale_fac, size=N_obs)  # drawn from same distribution
    # obs_cols = np.zeros( N_obs, dtype='float32' ) + loc # all exact

    # lim_types = np.zeros( N_obs, dtype='int32' ) # all detections (0=detections, 1=upper lim, 2=lower lim)
    lim_types = np.random.randint(low=0, high=3, size=N_obs)  # random assortment of limit types

    lambdas = np.zeros(obs_cols.size, dtype="float32")

    for i in range(obs_cols.size):
        z1 = kde.integrate_box_1d(-np.inf, obs_cols[i])
        z2 = kde.integrate_box_1d(obs_cols[i], np.inf)

        if lim_types[i] == 0:
            lambdas[i] = 2 * np.min([z1, z2])  # detection, 2*PDF area more extreme
        if lim_types[i] == 1:
            lambdas[i] = z1  # upper limit, PDF area which is consistent
        if lim_types[i] == 2:
            lambdas[i] = z2  # lower limit, PDF area which is consistent

    print("mean lambda: ", lambdas.mean())


# -------------------------------------------------------------------------------------------------

variants1 = ["0100", "0401", "0402", "0501", "0502", "0601", "0602", "0701", "0703", "0000"]
variants2 = ["0201", "0202", "0203", "0204", "0205", "0206", "0801", "0802", "1100", "0000"]
variants3 = ["1000", "1002", "1003", "1004", "1005", "4302", "1200", "1301", "1302", "0000"]
variants4 = ["2002", "2101", "2102", "2201", "2202", "2203", "2302", "4601", "4602", "0000"]
variants5 = ["3000", "3100", "3001", "3010", "3101", "3102", "3201", "3203", "3002", "0000"]
variants6 = ["3403", "3404", "3501", "3502", "3401", "3402", "3601", "3602", "3901", "0000"]
variants7 = ["3301", "3302", "3303", "3304", "3701", "3702", "3801", "3802", "3902", "0000"]
variants8 = ["4000", "4100", "4410", "4412", "4420", "4501", "4502", "4503", "4506", "0000"]
variantSets = [variants1, variants2, variants3, variants4, variants5, variants6, variants7, variants8]

variantsMain = ["0501", "0502", "0801", "2002", "2302", "2102", "2202", "3000", "3001", "3010", "3404", "0010", "0000"]


def paperPlots():
    """Construct all the final plots for the paper."""
    TNG100 = simParams(res=1820, run="tng", redshift=0.0)
    TNG100_2 = simParams(res=910, run="tng", redshift=0.0)
    TNG100_3 = simParams(res=455, run="tng", redshift=0.0)
    TNG300 = simParams(res=2500, run="tng", redshift=0.0)
    # TNG300_2  = simParams(res=1250,run='tng',redshift=0.0)
    # TNG300_3  = simParams(res=625,run='tng',redshift=0.0)
    Illustris = simParams(res=1820, run="illustris", redshift=0.0)

    ions = ["OVI", "OVII", "OVIII"]  # whenever we are not just doing OVI

    # figure 1, 2: full box composite image components, and full box OVI/OVIII ratio
    if 0:
        from temet.vis.boxDrivers import TNG_oxygenPaperImages

        for part in [3]:  # [0,1,2,3]:
            TNG_oxygenPaperImages(part=part)

    # figure 3a: ionization data for OVI, OVII, and OVIII
    if 0:
        element = "Oxygen"
        ionNums = [6, 7, 8]
        redshift = 0.0
        metal = -1.0  # log solar

        saveName = "abundance_fractions_%s_%s_z%d_Z%d.pdf" % (
            element,
            "-".join([str(i) for i in ionNums]),
            redshift * 100,
            10**metal * 1000,
        )

        ionAbundFracs2DHistos(saveName, element=element, ionNums=ionNums, redshift=redshift, metal=metal)

    # figure 3b: global box phase-diagrams weighted by gas mass in OVI, OVII, and OVIII
    if 0:
        snapshot.phaseSpace2d(
            TNG100,
            ptType="gas",
            xQuant="hdens",
            yQuant="temp",
            weights=["O VI mass", "O VII mass", "O VIII mass"],
            meancolors=None,
            clim=[-4.0, 0.0],  # massFracMinMax
            xlim=[-9.0, 0.0],
            ylim=[3.0, 8.0],
            contours=[-3.0, -2.0, -1.0],
            smoothSigma=1.0,
            hideBelow=True,
        )

    # figure 4, CDDF of OVI at z~0 compared to observations
    if 1:
        moment = 0
        simRedshift = 0.2
        boxDepth10 = True  # use 10 Mpc/h projection depth
        sPs = [TNG100, TNG100_2, TNG100_3, TNG300, Illustris]

        pdf = PdfPages(
            "cddf_ovi_z%02d_moment%d_%s%s.pdf"
            % (10 * simRedshift, moment, "_".join([sP.simName for sP in sPs]), "_10Mpch" if boxDepth10 else "")
        )
        nOVIcddf(sPs, pdf, moment=moment, simRedshift=simRedshift, boxDepth10=boxDepth10)
        pdf.close()

    # figure 5, CDDF redshift evolution of multiple ions (combined panel, and individual panels)
    if 0:
        moment = 0
        sPs = [TNG100]  # , Illustris]
        boxDepth10 = True
        redshifts = [0, 1, 2, 4]

        saveName = "cddf_%s_zevo-%s_moment%d_%s.pdf" % (
            "-".join(ions),
            "-".join(["%d" % z for z in redshifts]),
            moment,
            "_".join([sP.simName for sP in sPs]),
        )
        cddfRedshiftEvolution(
            sPs, saveName, moment=moment, ions=ions, redshifts=redshifts, boxDepth10=boxDepth10, colorOff=2
        )

        for i, ion in enumerate(ions):
            saveName = "cddf_%s_zevo-%s_moment%d_%s%s.pdf" % (
                ion,
                "-".join(["%d" % z for z in redshifts]),
                moment,
                "_".join([sP.simName for sP in sPs]),
                "_10Mpch" if boxDepth10 else "",
            )
            cddfRedshiftEvolution(
                sPs, saveName, moment=moment, ions=[ion], redshifts=redshifts, boxDepth10=boxDepth10, colorOff=i + 2
            )

    # figure 6, CDDF at z=0 with physics variants (L25n512)
    if 0:
        simRedshift = 0.0
        moment = 1

        sPs = []
        for variant in variantsMain:
            sPs.append(simParams(res=512, run="tng", redshift=simRedshift, variant=variant))

        saveName = "cddf_ovi_z%02d_moment%d_variants-main.pdf" % (10 * simRedshift, moment)
        cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=["OVI"], redshifts=[simRedshift])

    # figure 7: 2pcf
    if 0:
        redshift = 0.0
        sPs = [TNG100]  # [TNG100, TNG300]
        ions = ["OVI", "OVII", "OVIII", "O", "gas"]

        # compute time for one split:
        # TNG100 [days] = (1820^3/256^3)^2 * (1148/60/60/60) * (8*100/nSplits) * (16/nThreads)
        # for nSplits=200000, should finish each in 1.5 days (nThreads=32) (each has 60,000 cells)
        # for TNG300, nSplits=500000, should finish each in 4 days (nThreads=32)
        for order in [0, 1, 2]:
            saveName = "tpcf_order%d_%s_%s_z%02d.pdf" % (
                order,
                "-".join(ions),
                "_".join([sP.simName for sP in sPs]),
                redshift,
            )

            ionTwoPointCorrelation(sPs, saveName, ions=ions, redshift=redshift, order=order, colorOff=2)

    # figure 8, bound mass of O ions vs halo/stellar mass
    if 0:
        sPs = [TNG300]  # , TNG100]
        cenSatSelect = "cen"
        redshift = 0.0
        ionsLoc = ["AllGas", "AllGas_Metal", "AllGas_Oxygen"] + ions

        for vsHaloMass in [True]:  # [True,False]:
            massStr = "%smass" % ["stellar", "halo"][vsHaloMass]

            # saveName = 'ions_masses_vs_%s_%s_z%d_%s.pdf' % \
            #    (massStr,cenSatSelect,redshift,'_'.join([sP.simName for sP in sPs]))
            # totalIonMassVsHaloMass(sPs, saveName, ions=ionsLoc, cenSatSelect=cenSatSelect,
            #    redshift=redshift, vsHaloMass=vsHaloMass, secondTopAxis=True)

            saveName = "ions_avgcoldens_vs_%s_%s_z%d_%s.pdf" % (
                massStr,
                cenSatSelect,
                redshift,
                "_".join([sP.simName for sP in sPs]),
            )
            totalIonMassVsHaloMass(
                sPs,
                saveName,
                ions=ions,
                cenSatSelect=cenSatSelect,
                redshift=redshift,
                vsHaloMass=vsHaloMass,
                toAvgColDens=True,
            )  # , secondTopAxis=True)

    # figure 9: average radial profiles
    if 0:
        redshift = 0.0
        sPs = [TNG100]
        ions = ["OVI"]  # OVII, OVIII
        cenSatSelect = "cen"
        haloMassBins = [[10.9, 11.1], [11.4, 11.6], [11.9, 12.1], [12.4, 12.6]]
        projSpecs = ["3D", "2Dz_2Mpc"]
        combine2Halo = True

        simNames = "_".join([sP.simName for sP in sPs])

        for massDensity in [False]:  # [True,False]:
            for radRelToVirRad in [False]:  # [True,False]:
                for projDim in projSpecs:
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
                        ions,
                        redshift=redshift,
                        massDensity=massDensity,
                        radRelToVirRad=radRelToVirRad,
                        cenSatSelect="cen",
                        projDim=projDim,
                        haloMassBins=haloMassBins,
                        combine2Halo=combine2Halo,
                    )

    # figure 10: mock COS-Halos samples
    if 0:
        sPs = [TNG100, TNG300]

        simNames = "_".join([sP.simName for sP in sPs])
        obsSimMatchedGalaxySamples(sPs, "coshalos_sample_%s.pdf" % simNames, config="COS-Halos")

    # figure 11: COS-Halos: N_OVI vs impact parameter and vs sSFR bimodality
    if 0:
        sP = TNG100

        # obsColumnsDataPlot(sP, saveName='coshalos_ovi_%s.pdf' % sP.simName, config='COS-Halos')
        obsColumnsDataPlotExtended(sP, saveName="coshalos_ovi_%s_ext.pdf" % sP.simName, config="COS-Halos")

    # figure 12: covering fractions, OVI vs obs (all galaxies, and subsamples)
    if 0:
        sPs = [TNG100]  # , TNG100_2]

        # All Galaxies
        novi_vals = [13.5, 14.0, 14.15, 14.5, 15.0]
        saveName = "coshalos_covering_frac_%s.pdf" % "_".join([sP.simName for sP in sPs])
        coveringFractionVsDist(sPs, saveName, ions=["OVI"], colDensThresholds=novi_vals, conf=0)

        # sSFR / M* subsets
        novi_vals = [14.15]
        saveName = "coshalos_covering_frac_subsets_%s.pdf" % "_".join([sP.simName for sP in sPs])
        coveringFractionVsDist(sPs, saveName, ions=["OVI"], colDensThresholds=novi_vals, conf=1)

    # figure 13: nums 11-14 repeated for the eCGM dataset instead of COS-Halos
    if 0:
        sP = TNG100
        cf = "eCGMfull"  # eCGM

        obsSimMatchedGalaxySamples([sP], "%s_sample_%s.pdf" % (cf, sP.simName), config=cf)
        obsColumnsDataPlot(sP, saveName="%s_ovi_%s.pdf" % (cf, sP.simName), config=cf)
        obsColumnsDataPlotExtended(sP, saveName="%s_ovi_%s_ext.pdf" % (cf, sP.simName), config=cf)

        coveringFractionVsDist(
            [sP],
            "%s_covering_frac_%s.pdf" % (cf, sP.simName),
            ions=["OVI"],
            colDensThresholds=[13.5, 14.0, 14.5, 15.0],
            config=cf,
            conf=0,
        )
        for conf in [1, 2]:
            coveringFractionVsDist(
                [sP],
                "%s_covering_frac_%s_conf%d.pdf" % (cf, sP.simName, conf),
                ions=["OVI"],
                colDensThresholds=[13.5],
                config=cf,
                conf=conf,
            )

    # figure 14: covering fractions, with main physics variants (L25n512)
    if 0:
        novi_vals = [14.0]

        sPs = []
        for variant in variantsMain:
            sPs.append(simParams(res=512, run="tng", redshift=0.0, variant=variant))

        saveName = "covering_frac_ovi_variants-main.pdf"
        coveringFractionVsDist(sPs, saveName, ions=["OVI"], colDensThresholds=novi_vals, config="SimHalos_115-125")

    # figure 15, 16: OVI red/blue image samples
    if 0:
        from temet.vis.haloDrivers import tngFlagship_galaxyStellarRedBlue

        tngFlagship_galaxyStellarRedBlue(evo=False, redSample=1, conf=1)
        tngFlagship_galaxyStellarRedBlue(evo=False, blueSample=1, conf=1)

    # figure 17: OVI vs color at fixed stellar/halo mass
    if 0:
        sPs = [TNG100, TNG300]
        css = "cen"
        quant = "mass_ovi"
        xQuant = "color_C_gr"

        for iter in [0, 1]:
            if iter == 0:
                sQuant = "mstar_30pkpc_log"
                sRange = [10.4, 10.6]
            if iter == 1:
                sQuant = "mhalo_200_log"
                sRange = [12.0, 12.1]

            pdf = PdfPages(
                "slice_%s_%s_%s-%.1f-%.1f_%s.pdf"
                % ("_".join([sP.simName for sP in sPs]), xQuant, sQuant, sRange[0], sRange[1], css)
            )
            subhalos.slice(sPs, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, sRange=sRange, cenSatSelect=css, pdf=pdf)
            pdf.close()

    # figure 18, 19, 20: 2d histos
    if 0:
        sP = TNG300
        xQuants = ["mstar_30pkpc_log", "mhalo_200_log"]
        cQuant = "mass_ovi"

        yQuants1 = ["ssfr", "Z_gas", "fgas2", "size_gas", "temp_halo_volwt", "mass_z"]
        yQuants2 = [
            "surfdens1_stars",
            "Z_stars",
            "color_C_gr",
            "size_stars",
            "Krot_oriented_stars2",
            "Krot_oriented_gas2",
        ]
        yQuants3 = ["nh_halo_volwt", "fgas_r200", "pratio_halo_volwt", "BH_CumEgy_low", "BH_mass", "_dummy_"]

        yQuantSets = [yQuants1, yQuants2, yQuants3]

        for i, xQuant in enumerate(xQuants):
            yQuants3[-1] = xQuants[1 - i]  # include the other

            for yQuants in yQuantSets:
                params = {"cenSatSelect": "cen", "cStatistic": "median_nan", "cQuant": cQuant, "xQuant": xQuant}

                subhalos.histogram2d(sP, yQuant=yQuants[0], **params)
                subhalos.histogram2d(sP, yQuant=yQuants[1], **params)
                subhalos.histogram2d(sP, yQuant=yQuants[2], **params)
                subhalos.histogram2d(sP, yQuant=yQuants[3], **params)
                subhalos.histogram2d(sP, yQuant=yQuants[4], **params)
                subhalos.histogram2d(sP, yQuant=yQuants[5], **params)

    # ------------ appendix ---------------

    # figure A1, all CDDFs at z=0 with physics variants (L25n512)
    if 0:
        simRedshift = 0.0
        moment = 1

        for i, variants in enumerate(variantSets):
            sPs = []
            for variant in variants:
                sPs.append(simParams(res=512, run="tng", redshift=simRedshift, variant=variant))

            saveName = "cddf_ovi_z%02d_moment%d_variants-%d.pdf" % (10 * simRedshift, moment, i)
            cddfRedshiftEvolution(sPs, saveName, moment=moment, ions=["OVI"], redshifts=[simRedshift])

    # figure 14: all covering fractions with physics variants (L25n512)
    if 0:
        novi_vals = [14.0]

        for i, variants in enumerate(variantSets):
            sPs = []
            for variant in variants:
                sPs.append(simParams(res=512, run="tng", redshift=0.0, variant=variant))

            saveName = "covering_frac_ovi_variants-%d.pdf" % (i)
            coveringFractionVsDist(sPs, saveName, ions=["OVI"], colDensThresholds=novi_vals, config="SimHalos_115-125")

    # ------------ exploration ------------

    # exploration: OVI average column vs everything at fixed stellar/halo mass
    if 0:
        sPs = [TNG100, TNG300]
        css = "cen"
        quant = "mass_ovi"
        xQuants = quantList(wCounts=False, wTr=False, wMasses=True)

        for iter in [0, 1]:
            if iter == 0:
                sQuant = "mstar_30pkpc_log"
                sRange = [10.4, 10.6]
            if iter == 1:
                sQuant = "mhalo_200_log"
                sRange = [12.0, 12.1]

            pdf = PdfPages(
                "slices_%s_x=all_%s-%.1f-%.1f_%s.pdf"
                % ("_".join([sP.simName for sP in sPs]), sQuant, sRange[0], sRange[1], css)
            )
            for xQuant in xQuants:
                subhalos.slice(
                    sPs, xQuant=xQuant, yQuants=[quant], sQuant=sQuant, sRange=sRange, cenSatSelect=css, pdf=pdf
                )
            pdf.close()

    # exploration: median OVI column vs stellar/halo mass, split by everything else
    if 0:
        sPs = [TNG300]
        simNames = "-".join([sP.simName for sP in sPs])

        css = "cen"
        quants = quantList(wCounts=False, wTr=False, wMasses=True)
        priQuant = "mass_ovi"
        sLowerPercs = [10, 50]
        sUpperPercs = [90, 50]

        for xQuant in ["mstar_30pkpc", "mhalo_200_log"]:
            # individual plot per y-quantity:
            pdf = PdfPages("medianTrends_%s_x=%s_%s_slice=%s.pdf" % (simNames, xQuant, css, priQuant))
            for yQuant in quants:
                subhalos.median(
                    sPs,
                    yQuants=[yQuant],
                    xQuant=xQuant,
                    cenSatSelect=css,
                    sQuant=priQuant,
                    sLowerPercs=sLowerPercs,
                    sUpperPercs=sUpperPercs,
                    pdf=pdf,
                )
            pdf.close()

            # individual plot per s-quantity:
            pdf = PdfPages("medianTrends_%s_x=%s_%s_y=%s.pdf" % (simNames, xQuant, css, priQuant))
            for sQuant in quants:
                subhalos.median(
                    sPs,
                    yQuants=[priQuant],
                    xQuant=xQuant,
                    cenSatSelect=css,
                    sQuant=sQuant,
                    sLowerPercs=sLowerPercs,
                    sUpperPercs=sUpperPercs,
                    pdf=pdf,
                )

            pdf.close()

    # exploration: OVI vs everything else in the median
    if 0:
        sPs = [TNG100, TNG300]
        simNames = "-".join([sP.simName for sP in sPs])

        css = "cen"
        xQuants = quantList(wCounts=False, wTr=False, wMasses=True)
        yQuant = "mass_ovi"

        rQuant = "mhalo_200_log"  # only include systems satisfying this restriction
        rRange = [11.0, 16.0]  # restriction range

        # individual plot per y-quantity:
        pdf = PdfPages(
            "medianTrends_%s_y=%s_vs-all%d_%s_%s_in_%.1f-%.1f.pdf"
            % (simNames, yQuant, len(xQuants), css, rQuant, rRange[0], rRange[1])
        )
        for xQuant in xQuants:
            subhalos.slice(
                sPs, xQuant=xQuant, yQuants=[yQuant], sQuant=rQuant, sRange=rRange, cenSatSelect=css, pdf=pdf
            )
            # for most quantities, is dominated everywhere by low-mass halos with very small mass_ovi:
            # subhalos.median(sPs, yQuants=[yQuant], xQuant=xQuant, cenSatSelect=css, pdf=pdf)
        pdf.close()

    # exploration: cloudy ionization table
    if 0:
        from temet.cosmo.cloudy import plotIonAbundances

        plotIonAbundances(elements=["Oxygen"])

    # figure 4: testing toy model for total mass in different ions
    if 0:
        sPs = [TNG300]
        cenSatSelect = "cen"
        ionsLoc = ["AllGas", "AllGas_Metal", "AllGas_Oxygen"] + ions

        toyFacsList = [
            [1.0, 1.0, 1.0],
            [0.6, 1.5, 1.5],
            [0.8, 1.0, 1.5],
            [1.0, 1.0, 1.5],
            [1.0, 1.0, 2.0],
            [0.6, 1.0, 1.5],
            [0.6, 2.0, 1.5],
            [1.5, 1.0, 1.5],
            [2.0, 1.0, 1.5],
        ]

        for toyFacs in toyFacsList:
            toyStr = "toy=%.1f_%.1f_%.1f" % (toyFacs[0], toyFacs[1], toyFacs[2])
            saveName = "ions_masses_cen_z0_%s_%s.pdf" % ("_".join([sP.simName for sP in sPs]), toyStr)
            totalIonMassVsHaloMass(
                sPs,
                saveName,
                ions=ionsLoc,
                cenSatSelect="cen",
                redshift=0.0,
                vsHaloMass=True,
                secondTopAxis=True,
                toyFacs=toyFacs,
            )
