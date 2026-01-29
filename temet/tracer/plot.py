"""
Plotting for evolution of tracer quantities in time (for cosmo boxes/zooms).
"""

import glob
from os import mkdir
from os.path import isdir, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..cosmo.util import redshiftToSnapNum, snapNumToRedshift
from ..plot.config import colors, linestyles
from ..plot.util import loadColorTable
from ..tracer import evolution
from ..util.helper import closest, logZeroSafe
from ..util.simParams import simParams


modes = {
    None: "All Modes",
    evolution.ACCMODE_SMOOTH: "Smooth",
    evolution.ACCMODE_MERGER: "Merger",
    evolution.ACCMODE_STRIPPED: "Stripped",
}


def addRedshiftAgeImageAxes(ax, sP, snaps):
    """Add a redshift (bottom) and age (top) pair of axes for imshow plots.

    Top axis does not work when a colorbar is also added to the plot.
    """
    if sP.isZoom:
        zVals = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    else:
        zVals = np.array([0, 0.5, 1.0, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    snapSpacing = (snaps.max() - snaps.min()) / snaps.size
    zValsSnaps = snapNumToRedshift(sP, snap=snaps)

    toplace = np.zeros(zVals.size, dtype="float32")
    for i, zVal in enumerate(zVals):
        _, snapsInd = closest(zValsSnaps, zVal)
        toplace[i] = snaps.max() - (float(snapsInd) * snapSpacing + 0.5 * snapSpacing)

    ax.set_xticks(toplace)
    ax.set_xticklabels(["%g" % zVal for zVal in zVals])
    ax.set_xlabel("Redshift")

    if 0:
        axTop = ax.twiny()

        ageVals = [0.7, 1.0, 1.5, 2.0, 3.0]
        ageVals.append(sP.units.redshiftToAgeFlat([zVals.min()]).round(2))
        axTickVals = redshiftToSnapNum(sP.units.ageFlatToRedshift(np.array(ageVals)), sP)

        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(axTickVals)
        axTop.set_xticklabels(ageVals)
        axTop.set_xlabel("Age of the Universe [Gyr]")


def plotConfig(fieldName, extType=""):
    """Store some common plot configuration parameters."""
    ctName = "jet"
    loadField = fieldName

    # enumerate possible fields
    if fieldName == "tracer_maxtemp":
        label = "Tracer Temperature%s [ log K ]"
        valMinMax = [4.0, 7.5]

    if fieldName == "tracer_maxtemp_tviracc":
        label = r"log ( Tracer Temp%s / Halo T$_{\rm vir}$ at AccTime )"
        valMinMax = [-2.0, 1.5]
        loadField = "tracer_maxtemp"

    if fieldName == "tracer_maxent":
        label = "Tracer Entropy%s [ log K cm^2 ]"
        valMinMax = [5.0, 9.0]

    if fieldName == "tracer_maxent_sviracc":
        label = r"log ( Tracer Entropy%s / Halo S$_{\rm 200}$ at AccTime )"
        valMinMax = [-2.0, 1.5]
        loadField = "entr"

    if fieldName == "rad_rvir":
        label = r"R / R$_{\rm vir}$ %s"
        valMinMax = [0.0, 2.0]

    if fieldName == "vrad":
        label = "Radial Velocity%s [ km / s ]"
        valMinMax = [-600, 600]

    if fieldName == "entr":
        label = "Gas Entropy%s [ log K cm^2 ]"
        valMinMax = [5.0, 9.0]

    if fieldName == "entr_sviracc":
        label = r"log ( Gas Entropy%s / Halo S$_{\rm 200}$ at AccTime )"
        valMinMax = [-2.0, 1.5]
        loadField = "entr"

    if fieldName == "temp":
        label = "Gas Temperature%s [ log K ]"
        valMinMax = [4.0, 7.0]

    if fieldName == "temp_tviracc":
        label = r"log ( Gas Temp%s / Halo T$_{\rm vir}$ at AccTime )"
        valMinMax = [-2.0, 1.5]
        loadField = "temp"

    if fieldName == "sfr":
        label = "Gas SFR%s [ Msun / yr ]"
        valMinMax = [0, 10]

    if fieldName == "subhalo_id":
        label = "Parent Subhalo ID%s"
        valMinMax = None

    if fieldName == "parent_indextype":
        label = "Parent IndexType%s"
        valMinMax = None

    if fieldName == "angmom":
        label = "Specific Angular Momentum%s [ log kpc km/s ]"
        valMinMax = [4.0, 6.5]

    # add extStr to denote extremum selection and/or t_* type selections
    extStr = ""

    if extType in ["min", "max"]:
        extStr = " [" + extType.capitalize() + "]"
    if extType in ["min_b015", "max_b015"]:
        extStr = " [" + extType.split("_")[0].capitalize() + r" Before First 0.15 r/r$_{\rm vir}$ Crossing]"
    if extType in ["t_acc"]:
        extStr = " [ At the Accretion Time ]"
    if "__" in extType:
        # recursively call ourselves to get some info on the second field
        # not quite there, perhaps getting a little too complicated
        _, extFieldName, extFieldType = extType.split("__")
        # _, label2, _, _, _ = plotConfig(extFieldName, extType=extFieldType)
        extStr = " [ At (%s) of %s ]" % (extFieldType, extFieldName)

    label = label % extStr

    return ctName, label, valMinMax, loadField


def getEvo2D(sP, field, trIndRange=None, accTime=None, accMode=None):
    """Create and cache various permutations of the full 2D evolution tracks for all tracers."""
    # load config for this field
    r = {}

    rasterVerticalSize = 1080 * 2
    resizeInterpOrder = 1  # 0=nearest, 1=linear, 2=quadratic, 3=cubic, etc
    snapStep = 1

    # check for existence
    trIndStr = "all-%d" % rasterVerticalSize
    if trIndRange is not None:
        trIndStr = "%g-%d" % (trIndRange[0], trIndRange[1])

    if sP.isZoom:
        saveFilename = sP.derivPath + "/trValHist/shID_%d_hf%d_snap_%d-%d-%d_%s_2d_%s.hdf5" % (
            sP.zoomSubhaloID,
            True,
            sP.snap,
            redshiftToSnapNum(evolution.maxRedshift, sP),
            snapStep,
            field,
            trIndStr,
        )
    else:
        boxOrHaloStr = "box"
        if sP.haloInd is not None:
            boxOrHaloStr = "halo-%d" % sP.haloInd
        if sP.subhaloInd is not None:
            boxOrHaloStr = "subhalo-%d" % sP.subhaloInd

        saveFilename = sP.derivPath + "/trValHist/%s_2d_%s_snap_%d-%d-%d_%s.hdf5" % (
            field,
            boxOrHaloStr,
            sP.snap,
            redshiftToSnapNum(evolution.maxRedshift, sP),
            snapStep,
            trIndStr,
        )

    if not isdir(sP.derivPath + "/trValHist"):
        mkdir(sP.derivPath + "/trValHist")

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key1 in f:
                # various 2d datasets grouped by accMode
                r[key1] = {}
                for key2 in f[key1]:
                    r[key1][key2] = f[key1][key2][()]
        return r

    # load data
    if accMode is None:
        accMode = evolution.accMode(sP, snapStep)
    if accTime is None:
        accTime = evolution.accTime(sP, snapStep)

    _, _, valMinMax, loadField = plotConfig(field)

    data = evolution.tracersTimeEvo(sP, loadField, snapStep, all=False)

    # normalize?
    w0 = np.where(data[loadField] == 0.0)

    if "_tviracc" in field or "_sviracc" in field:
        assert sP.haloInd is None and sP.subhaloInd is None  # todo

        if "_tviracc" in field:
            normVal = 10.0 ** evolution.mpbValsAtAccTimes(sP, "tvir", rVirFac=1.0)
        if "_sviracc" in field:
            normVal = 10.0 ** evolution.mpbValsAtAccTimes(sP, "svir", rVirFac=1.0)

        data[loadField] = 10.0 ** data[loadField]

        for i in range(data["snaps"].size):
            data[loadField][i, :] /= normVal

        data[loadField] = logZeroSafe(data[loadField])

    if loadField in ["tracer_maxtemp", "tracer_maxent"]:
        # these fields have used 0.0 for missing values (e.g. on eEOS) in the code
        # and have possibly been transformed into valid/strange values by normalization
        # here, tag as nan for consistency with e.g. temp/entr of gas cells
        data[loadField][w0] = np.nan

    # axes ranges and place image
    x_min = int(data["snaps"].max())
    x_max = int(data["snaps"].min())

    # create 2d block by mode and store in return dict
    for modeVal, modeName in modes.items():
        if modeVal is not None:
            ww = np.where(accMode == modeVal)[0]
        else:
            ww = np.arange(accMode.size)

        r[modeName] = {}

        # axes ranges
        if trIndRange is None:
            y_min = 0
            y_max = ww.size - 1
        else:
            y_min = int(trIndRange[0] * ww.size)  # fraction along trInds axis
            y_max = y_min + trIndRange[1] - 1

        r[modeName]["extent"] = [x_min, x_max, y_min, y_max]
        r[modeName]["snaps"] = data["snaps"]

        # (A) data transform, raw
        data2d = np.transpose(data[loadField][:, ww])

        # (B) data transform, sorted by t_acc
        sort_inds_t_acc = np.argsort(accTime[ww])
        data2d_sorted = np.zeros_like(data2d)
        for i in range(data2d.shape[1]):
            data2d_sorted[:, i] = data2d[sort_inds_t_acc, i]

        # resize tracerInd axis to reasonable raster size, or take small subset
        if trIndRange is None:
            zoomFac = rasterVerticalSize / data2d.shape[0]

            data2d = ndimage.zoom(data2d, [zoomFac, 1], order=resizeInterpOrder)
            data2d_sorted = ndimage.zoom(data2d_sorted, [zoomFac, 1], order=resizeInterpOrder)
        else:
            data2d = data2d[y_min : y_max + 1, :]
            data2d_sorted = data2d_sorted[y_min : y_max + 1, :]

        r[modeName]["t2d"] = data2d
        r[modeName]["t2d_sort_t_acc"] = data2d_sorted

    # save
    with h5py.File(saveFilename, "w") as f:
        for key1 in r:
            for key2 in r[key1]:
                f[key1 + "/" + key2] = r[key1][key2]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    return r


def plotEvo2D():
    """Plot various full 2D blocks showing evolution of 'all' tracer tracks vs redshift/radius."""
    # config
    # sP = simParams(res=11, run='zooms2', redshift=2.0, hInd=2)
    # fieldNames = ["tracer_maxtemp_tviracc","temp_tviracc","tracer_maxtemp","temp",
    #              "tracer_maxent","tracer_maxent_sviracc","entr",
    #              "rad_rvir","vrad","angmom"] # dens

    sP = simParams(res=910, run="tng", redshift=0.0)
    trIndRanges = [None]
    # sP = simParams(res=455, run='tng', redshift=0.0, haloInd=0)
    fieldNames = ["rad_rvir", "angmom", "subhalo_id", "temp", "vrad", "entr", "sfr"]
    # trIndRanges = [None, [0.5,1080]]

    # load accretion times, accretion modes (can change to None after cached)
    accTime = evolution.accTime(sP)
    accMode = evolution.accMode(sP)

    for field in fieldNames:
        ctName, label, valMinMax, _ = plotConfig(field)

        for trIndRange in trIndRanges:
            # load
            evo = getEvo2D(sP, field, trIndRange=trIndRange, accTime=accTime, accMode=accMode)

            # start pdf
            trIndStr = "all"
            trSubStr = ""
            if trIndRange is not None:
                trIndStr = "%g-%d" % (trIndRange[0], trIndRange[1])
            if sP.haloInd is not None:
                trSubStr = "_halo-%d" % sP.haloInd
            if sP.subhaloInd is not None:
                trSubStr = "_subhalo-%d" % sP.subhaloInd

            pdf = PdfPages("evo2D_%s_%s%s_%s.pdf" % (field, trIndStr, trSubStr, sP.simName))

            # make following plots for each accMode separately
            for modeName in modes.values():
                # plot bounds
                extent = evo[modeName]["extent"]

                # PLOT 1: overview 2D plot of all tracker tracks
                print("1. %s %s %s %s" % (sP.simName, field, trIndStr, modeName))

                fig, ax = plt.subplots()
                ax.set_title(modeName)
                ax.set_ylabel("TracerInd")

                # color mapping
                cmap = loadColorTable(ctName)

                plt.imshow(evo[modeName]["t2d"], cmap=cmap, extent=extent, aspect="auto", origin="lower")

                if valMinMax is not None:
                    plt.clim(valMinMax)

                # colobar and axes
                cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)

                cb = plt.colorbar(cax=cax)  # , format=FormatStrFormatter('%.1f'))
                cb.ax.set_ylabel(label)

                addRedshiftAgeImageAxes(ax, sP, evo[modeName]["snaps"])

                # finish
                pdf.savefig()
                plt.close(fig)

                # PLOT 2: overview 2D plot of all tracker tracks, sort by accTime
                fig, ax = plt.subplots()
                ax.set_title(modeName)
                ax.set_ylabel("TracerInd [Sorted by Accretion Time]")

                # color mapping
                cmap = loadColorTable(ctName)

                plt.imshow(evo[modeName]["t2d_sort_t_acc"], cmap=cmap, extent=extent, aspect="auto", origin="lower")

                if valMinMax is not None:
                    plt.clim(valMinMax)

                # colobar and axes
                cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.2)

                cb = plt.colorbar(cax=cax)  # , format=FormatStrFormatter('%.1f'))
                cb.ax.set_ylabel(label)

                addRedshiftAgeImageAxes(ax, sP, evo[modeName]["snaps"])

                # finish
                pdf.savefig()
                plt.close(fig)

            pdf.close()


def plotEvo1D():
    """Plot various 1D views showing evolution of tracer tracks vs redshift/radius."""
    # config
    sP = simParams(res=9, run="zooms2", redshift=2.0, hInd=2)
    fieldNames = [
        "tracer_maxtemp",
        "temp",
        "tracer_maxent",
        "tracer_maxent",
        "entr",
        "rad_rvir",
        "vrad",
        "angmom",
    ]  # dens

    assert sP.isZoom  # todo

    # load accretion times and modes for selections
    # accTime = evolution.accTime(sP)
    accMode = evolution.accMode(sP)

    pdf = PdfPages("evo1D_%s_nF%d.pdf" % (sP.simName, len(fieldNames)))

    for fieldName in fieldNames:
        ctName, label, valMinMax, _ = plotConfig(fieldName)

        # load
        # evo = getEvo2D(sP, field, trIndRange=trIndRange, accTime=accTime, accMode=accMode)
        data = evolution.tracersTimeEvo(sP, fieldName, all=False)
        data2d = np.transpose(data[fieldName].copy())

        for modeVal, modeName in modes.items():
            print("1. %s %s" % (fieldName, modeName))

            # PLOT 1: little 1D plot of a few tracer tracks
            inds = [0, 10, 100, 1000, 10000]

            fig, ax = plt.subplots()
            ax.set_title(modeName)
            # ax.set_xlim([sP.redshift,evolution.maxRedshift])
            ax.set_xlabel("Redshift")
            ax.set_ylabel(label)

            if valMinMax is not None:
                ax.set_ylim(valMinMax)

            # make selection
            if modeVal is not None:
                ww = np.where(accMode == modeVal)[0]
                plotInds = ww[inds]
            else:
                plotInds = inds

            for ind in plotInds:
                ax.plot(data["redshifts"], data2d[ind, :])

            # ax.legend(loc='upper right')

            # finish
            pdf.savefig()
            plt.close(fig)

    pdf.close()


def getValHistos(sP, field, extType, accMode=None):
    """Calculate and cache 1D histograms of field/extType combinations from the full tracer tracks."""
    # load config for this field
    assert sP.isZoom  # todo

    r = {}

    _, _, valMinMax, loadField = plotConfig(field, extType=extType)

    nBins = int(50 * sP.zoomLevel)  # 100,150,200
    # nBins = int( 25 * 2**sP.zoomLevel ) # 100,200,400
    # nBins = int( np.sqrt( data.size ) * 0.25 )

    # check for existence
    saveFilename = sP.derivPath + "/trValHist/shID_%d_hf%d_snap_%d-%d-%d_%s-%s.hdf5" % (
        sP.zoomSubhaloID,
        True,
        sP.snap,
        redshiftToSnapNum(evolution.maxRedshift, sP),
        1,
        field,
        extType,
    )

    if not isdir(sP.derivPath + "/trValHist"):
        mkdir(sP.derivPath + "/trValHist")

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key1 in f:
                # metadata
                if key1 in ["nBins", "valMinMax"]:
                    r[key1] = f[key1]
                    continue

                # x,y pairs for each mode
                r[key1] = {}
                for key2 in f[key1]:
                    r[key1][key2] = f[key1][key2][()]
        return r

    # load data
    if accMode is None:
        accMode = evolution.accMode(sP)

    if extType in evolution.allowedExtTypes:
        # the extremum values of field, one per tracer
        data = evolution.valExtremum(sP, loadField, extType=extType)
        data = data["val"]
    elif extType in ["t_acc"]:
        # the values of field at the acc time, one per tracer
        data = evolution.trValsAtAccTimes(sP, loadField, rVirFac=1.0)
    else:
        # the values of field at the time of the extremum (min/max) of a second field
        _, extFieldName, extFieldType = extType.split("__")

        data = evolution.trValsAtExtremumTimes(sP, loadField, extFieldName, extType=extFieldType)

    # normalize?
    if "_tviracc" in field or "_sviracc" in field:
        if "_tviracc" in field:
            normVal = 10.0 ** evolution.mpbValsAtAccTimes(sP, "tvir", rVirFac=1.0)
        if "_sviracc" in field:
            normVal = 10.0 ** evolution.mpbValsAtAccTimes(sP, "svir", rVirFac=1.0)

        ww = np.where(data == 0.0)  # e.g. tracer_maxtemp has 0 for missing values
        data = logZeroSafe(10.0**data / normVal)
        data[ww] = np.nan  # re-tag as nan

    # histogram by mode and store in return dict
    for modeVal, modeName in modes.items():
        if modeVal is not None:
            ww = np.where(accMode == modeVal)[0]
        else:
            ww = np.arange(accMode.size)

        yy, xx = np.histogram(data[ww], bins=nBins, range=valMinMax, density=True)
        xx = xx[:-1] + 0.5 * (valMinMax[1] - valMinMax[0]) / nBins
        # yf = savgol_filter(yy, sKn, sKo)

        r[modeName] = {}
        r[modeName]["x"] = xx
        r[modeName]["y"] = yy

    # save
    with h5py.File(saveFilename, "w") as f:
        f["nBins"] = [nBins]
        f["valMinMax"] = valMinMax

        for key1 in r:
            for key2 in r[key1]:
                f[key1 + "/" + key2] = r[key1][key2][()]

    print("Saved: [%s]" % saveFilename.split(sP.derivPath)[1])

    return r


def plotValHistos():
    """Plot (1D) histograms of extremum values, values at t_acc, or values at the extremum time of another value."""
    # config
    sPs = []

    sPs.append(simParams(res=11, run="zooms2", redshift=2.0, hInd=2))
    sPs.append(simParams(res=10, run="zooms2", redshift=2.0, hInd=2))
    sPs.append(simParams(res=9, run="zooms2", redshift=2.0, hInd=2))

    # fieldName:compSpec pairs (not relevant: "sfr","subhalo_id")
    #  - fieldName can be any tracked field, optionally normalized
    #  - compSpec can be:
    #    * min/min_b015/max/max_b015 (an extType, for the extremum values)
    #    * t_acc (for the values of fieldName at the acc time)
    #    * t__second_field__extType (for the values of fieldName at the extType extremum of second_field)
    fieldNames = {
        "tracer_maxtemp": ["max", "max_b015"],
        "temp": ["max"],
        "tracer_maxtemp_tviracc": ["max", "max_b015"],
        "tracer_maxent": ["max"],
        "entr": ["max"],
        "entr_sviracc": ["max"],
        "vrad": ["max", "min"],
        "rad_rvir": ["min", "t_acc", "t__tracer_maxtemp__max", "t__tracer_maxtemp__max_b015"],
        "angmom": ["max", "t_acc"],
    }  # dens

    # PLOT 1: split by accretion mode, one sP per plot
    for sP in sPs:
        # load global quantities for this run (can replace with None once all cached)
        accMode = evolution.accMode(sP)

        pdf = PdfPages("valExtremumHistos_ByAccMode_" + sP.simName + ".pdf")

        # loop over fields
        for field, extTypes in fieldNames.items():
            for extType in extTypes:
                # load config for this field
                ctName, label, valMinMax, _ = plotConfig(field, extType=extType)
                print("1. %s %s %s" % (sP.simName, field, extType))

                # start figure
                fig, ax = plt.subplots()
                ax.set_xlim(valMinMax)
                ax.set_xlabel(label)
                ax.set_ylabel(r"PDF $\int=1$")

                # load data for all modes (cached histograms)
                vh = getValHistos(sP, field, extType, accMode=accMode)

                # histogram by mode
                for modeName in modes.values():
                    c = None
                    if modeName == "All Modes":
                        c = "black"

                    ax.plot(vh[modeName]["x"], vh[modeName]["y"], color=c, label=modeName)

                # finish plot
                ax.legend(loc="best")

                pdf.savefig()
                plt.close(fig)

        pdf.close()

    # PLOT 2: split by resolution/sP, one accMode per plot
    for modeName in modes.values():
        pdf = PdfPages("valExtremumHistos_ByRes_" + modeName + ".pdf")

        # loop over fields
        for field, extTypes in fieldNames.items():
            for extType in extTypes:
                # load config for this field
                ctName, label, valMinMax, _ = plotConfig(field, extType=extType)
                print("2. %s %s %s" % (modeName, field, extType))

                # start figure
                fig, ax = plt.subplots()
                ax.set_xlim(valMinMax)
                ax.set_xlabel(label)
                ax.set_ylabel(r"PDF $\int=1$")

                # histogram by res
                c = "black"

                for i, sP in enumerate(sPs):
                    # load data (cached histograms)
                    vh = getValHistos(sP, field, extType)

                    (l,) = ax.plot(
                        vh[modeName]["x"], vh[modeName]["y"], color=c, label="L" + str(sP.res), linestyle=linestyles[i]
                    )

                # finish plot
                ax.legend(loc="best")

                pdf.savefig()
                plt.close(fig)

        pdf.close()

    # PLOT 3: split by resolution/sP, all accModes on each plot
    pdf = PdfPages("valExtremumHistos_ByRes_nSP" + str(len(sPs)) + ".pdf")

    # loop over fields
    for field, extTypes in fieldNames.items():
        for extType in extTypes:
            # load config for this field
            ctName, label, valMinMax, _ = plotConfig(field, extType=extType)
            print("3. %s %s" % (field, extType))

            # start figure
            fig, ax = plt.subplots()
            ax.set_xlim(valMinMax)
            ax.set_xlabel(label)
            ax.set_ylabel(r"PDF $\int=1$")

            for i, sP in enumerate(sPs):
                # load data for run (cached histograms)
                vh = getValHistos(sP, field, extType)

                # loop over each accMode and plot
                for j, modeName in enumerate(modes.values()):
                    c = colors[j] if modeName != "All Modes" else "black"

                    # set accMode labels only once
                    label = ""
                    if i == 0:
                        label = modeName

                    ax.plot(vh[modeName]["x"], vh[modeName]["y"], color=c, label=label, ls=linestyles[i])

            # finish plot
            # ax.legend(loc='best')

            sExtra = [plt.Line2D([0], [0], color="black", marker="", linestyle=ls) for ls in linestyles]
            lExtra = [str("L" + str(sP.res)) for sP in sPs]

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles + sExtra, labels + lExtra, loc="best")

            pdf.savefig()
            plt.close(fig)

    pdf.close()

    print("Done.")


# --- old ---


def plotPosTempVsRedshift():
    """Plot trMC position (projected) and temperature evolution vs redshift."""
    # config
    axis1 = 0
    axis2 = 2
    alpha = 0.05
    boxSize = 1000.0  # ckpc/h
    sP = simParams(res=1820, run="illustris", redshift=0.0)

    shNums = [int(s[:-5].rsplit("_", 1)[1]) for s in glob.glob(sP.derivPath + "subhalo_*.hdf5")]
    shNum = shNums[0]

    # load
    with h5py.File(sP.derivPath + "subhalo_" + str(shNum) + ".hdf5") as f:
        pos = f["pos"][()]
        temp = f["temp"][()]
        sfr = f["sfr"][()]
        redshift = f["Redshift"][()]

    # plot
    if 0:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_xlim(pos[:, :, axis1].mean() + np.array([-boxSize, boxSize]))
        ax.set_ylim(pos[:, :, axis2].mean() + np.array([-boxSize, boxSize]))
        ax.set_aspect(1.0)

        ax.set_title("Evolution of tracer positions with time check")
        ax.set_xlabel("x [ckpc/h]")
        ax.set_ylabel("y [ckpc/h]")

        # make relative and periodic correct
        sP.correctPeriodicPosBoxWrap(pos)

        # for i in np.arange(pos.shape[1]):
        for i in np.arange(10000):
            ax.plot(pos[:, i, axis1], pos[:, i, axis2], "-", color="#333333", alpha=alpha, lw=1.0)

        plt.savefig("trMC_checkPos_" + sP.simName + "_sh" + str(shNum) + ".pdf")
        plt.close(fig)

    # plot 2
    if 1:
        fig, ax = plt.subplots()
        ax.set_xlim([0.0, 0.5])
        ax.set_ylim([3.5, 8.0])

        ax.set_title("Evolution of tracer temperatures with time check")
        ax.set_xlabel("Redshift")
        ax.set_ylabel("Temp [log K]")

        # for i in np.arange(temp.shape[1]):
        for i in [205]:
            # plot only snapshots with temp (in gas) and sfr=0 (not eEOS)
            # ww = np.isfinite(temp[:,i]) & (sfr[:,i] == 0.0)
            # if not np.count_nonzero(ww):
            #    continue
            # ax.plot(redshift[ww], np.squeeze(temp[ww,i]), '-', color='#333333', alpha=alpha*5, lw=2.0)

            # plot only those tracers which have been always in gas with sfr=0 their whole track
            # ww = np.isnan(temp[:,i]) | (sfr[:,i] > 0.0)
            # if np.count_nonzero(ww):
            #    continue
            # ax.plot(redshift, np.squeeze(temp[:,i]), '-', color='#333333', alpha=alpha, lw=1.0)

            # test
            ww = np.isfinite(temp[:, i]) & (sfr[:, i] == 0.0)
            ax.plot(redshift[ww], np.squeeze(temp[ww, i]), "-", alpha=alpha * 10, lw=2.0, label="gas sfr==0")

            ww = np.isfinite(temp[:, i])
            ax.plot(redshift[ww], np.squeeze(temp[ww, i]), "--", alpha=alpha * 10, lw=2.0, label="gas sfr any")

            ax.plot(redshift, np.squeeze(temp[:, i]), "o", alpha=alpha * 10, lw=2.0, label="star or gas")

            print(temp[:, i])

        # test
        ax.legend()

        plt.savefig("trMC_checkTempB_" + sP.simName + "_sh" + str(shNum) + ".pdf")
        plt.close(fig)


def plotStarFracVsRedshift():
    """Plot the fraction of tracers in stars vs. gas parents vs redshift."""
    # config
    alpha = 0.3
    sP = simParams(res=1820, run="illustris", redshift=0.0)

    shNums = [int(s[:-5].rsplit("_", 1)[1]) for s in glob.glob(sP.derivPath + "subhalo_*.hdf5")]

    # plot
    fig, ax = plt.subplots()
    ax.set_xlim([0.0, 0.5])
    # ax.set_ylim([0.0,0.4])

    ax.set_title("")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Fraction of trMC in Stellar Parents")

    for shNum in shNums:
        # load
        with h5py.File(sP.derivPath + "subhalo_" + str(shNum) + ".hdf5") as f:
            temp = f["temp"][()]
            # sfr = f["sfr"][()]
            redshift = f["Redshift"][()]

        # calculate fraction at each snapshot (using temp=nan->in star)
        fracInStars = np.zeros(temp.shape[0])
        for i in np.arange(temp.shape[0]):
            numInStars = np.count_nonzero(np.isfinite(temp[i, :]))
            fracInStars[i] = numInStars / float(temp.shape[1])

        ax.plot(redshift, fracInStars, "-", color="#333333", alpha=alpha, lw=1.0)

    plt.savefig("trMC_starFracs_" + sP.simName + ".pdf")
    plt.close(fig)
