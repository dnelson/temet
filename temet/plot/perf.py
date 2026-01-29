"""
Performance, scaling and timings analysis.
"""

from datetime import datetime, timedelta
from glob import glob
from os.path import expanduser

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d

from ..load.simtxt import loadCpuTxt, loadTimebinsTxt
from ..plot.config import figsize
from ..plot.util import getWhiteBlackColors, setAxisColors
from ..util.helper import closest
from ..util.simParams import simParams


def _redshiftAxisHelper(ax):
    """Add a redshift axis to the top of a single-panel plot, assuming bottom axis is scale factor."""
    zVals = np.array([20.0, 15.0, 10.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.0])
    axTop = ax.twiny()
    axTickVals = 1 / (1 + zVals)

    scalefac_lim = ax.get_xlim()
    w = np.where((axTickVals >= scalefac_lim[0]) & (axTickVals <= scalefac_lim[1]))

    axTop.set_xlim(scalefac_lim)
    axTop.set_xticks(axTickVals[w])
    axTop.set_xticklabels(zVals[w])
    axTop.set_xlabel("Redshift")

    return axTop


def _cpuEstimateFromOtherRunProfile(sP, cur_a, cur_cpu_mh):
    """Use the CPU_hours(a) trend from one run to extrapolation a predicted CPU time curve for a second run."""
    cpu = loadCpuTxt(sP.arepoPath, keys=["total", "time", "hatb"], hatbMin=41)

    # include only bigish timesteps
    w = np.where(cpu["hatb"] >= cpu["hatb"].max() - 6)

    xx = cpu["time"][w]
    yy = np.squeeze(np.squeeze(cpu["total"])[w, 2])
    yy = yy / (1e6 * 60.0 * 60.0) * cpu["numCPUs"]  # Mh

    # not finished? replace last entry with the a=1.0 expectation
    if xx.max() < 1.0:
        # print(' update a=%.1f [%.2f] to a=1.0 [%.2f]' % (xx[-1],yy[-1],yy[-1] / xx[-1]))
        yy[-1] = yy[-1] / xx[-1]
        xx[-1] = 1.0

    # convert to fraction, interpolate to 200 points in scalefac
    frac = yy / yy.max()
    f = interp1d(xx, frac)

    scalefac = np.linspace(0.01, 1.0, 200)
    cpu_frac = f(scalefac)

    # use:
    _, ind = closest(scalefac, cur_a)
    new_fracs = cpu_frac / cpu_frac[ind]
    predicted_cpu_mh = cur_cpu_mh * new_fracs
    estimated_total_cpu_mh = predicted_cpu_mh.max()

    return scalefac, predicted_cpu_mh, estimated_total_cpu_mh


def plotCpuTimes(sims=None, xlim=(0.0, 1.0)):
    """Plot code time usage fractions from cpu.txt."""
    # config
    plotKeys = [
        "total",
        "total_log",
        "treegrav",
        "pm_grav",
        "voronoi",
        "hydro",  # enrich
        "stellarfeed",
        "sfrcool",
        "gradients",
        "domain",
        "i_o",
        "restart",
        "subfind",
    ]
    # plotKeys = ['total']

    # multipage pdf: one plot per value
    fName1 = "cpu_times.pdf"
    fName2 = "cpu_times_all.pdf"

    pdf = PdfPages(fName1)
    print(" -- run: %s --" % datetime.now().strftime("%d %B, %Y"))

    for plotKey in plotKeys:
        fig = plt.figure(figsize=(12.5, 9))

        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)
        ax.tick_params(labeltop=False, labelright=True)

        ax.set_title("")
        ax.set_xlabel("Scale Factor")

        if plotKey in ["total", "total_log"]:
            ind = 2  # 0=diff time, 2=cum time
            ax.set_ylabel("CPU Time " + plotKey + " [Mh]")

            if plotKey == "total_log":
                ax.set_ylabel("Total CPU Time [Mh]")
                ax.set_yscale("log")
                plotKey = "total"
        else:
            ind = 3  # 1=diff perc (missing in 3col format), 3=cum perc
            ax.set_ylabel("CPU Percentage [" + plotKey + "]")

        keys = ["time", "hatb", plotKey]
        pLabels = []
        pColors = []

        for sim in sims:
            # load select datasets from cpu.hdf5
            if sim.run == "tng" and sim.res in [1024, 910, 1820, 1080, 2160, 1250, 2500]:
                hatbMin = 41
            else:
                hatbMin = 37  # may need to lower for certain runs

            cpu = loadCpuTxt(sim.arepoPath, keys=keys, hatbMin=hatbMin)

            if plotKey not in cpu.keys() or cpu[plotKey][0, -1, 2] < 0.2:
                continue  # e.g. hydro fields in DMO runs

            # include only bigish timesteps
            w = np.where(cpu["hatb"] >= cpu["hatb"].max() - 6)

            # loop over each run
            xx = cpu["time"][w]
            yy = np.squeeze(np.squeeze(cpu[plotKey])[w, ind])

            if ind in [0, 2]:
                yy = yy / (1e6 * 60.0 * 60.0) * cpu["numCPUs"]

            label = sim.simName
            if "total" in plotKey:
                if yy[-1] > 0.1:
                    label = sim.simName + " (%.2f Mh)" % yy[-1]
                else:
                    label = sim.simName + " (%.2f Kh)" % (yy[-1] * 1000)

            (l,) = ax.plot(xx, yy, label=label)

            if plotKey == "total":
                print(f"{sim.simName} [{plotKey}]: max_time = {cpu['time'].max()} total CPUh = {yy.max() * 1000:.2f}k")

            # for zooms which stop at high-z, adjust x-axis
            if xx.max() < 0.99 and sim.isZoom and sim.sP_parent is not None and sim.sP_parent.redshift > 0:
                ax.set_xlim([0.0, xx.max()])

            # total time predictions for runs which aren't yet done
            if plotKey in ["total"] and xx.max() < 0.99 and not sim.isZoom:
                if ax.get_yscale() == "log":
                    ax.set_ylim([1e-1, 200])

                fac_delta = 0.02
                xp = np.linspace(xx.max() + 0.25 * fac_delta, 1.0)

                # plot variance band
                w0 = np.where(xx >= xx.max() - fac_delta * 2)
                yp0 = np.poly1d(np.polyfit(xx[w0], yy[w0], 1))
                yPredicted0 = yp0(xp)

                w1 = np.where(xx >= xx.max() - fac_delta * 0.2)
                yp1 = np.poly1d(np.polyfit(xx[w1], yy[w1], 1))
                yPredicted1 = yp1(xp)

                ax.fill_between(xp, yPredicted0, yPredicted1, color=l.get_color(), alpha=0.1)

                # plot best line
                w = np.where(xx >= xx.max() - fac_delta)
                xx2 = xx[w]
                yy2 = yy[w]

                yp = np.poly1d(np.polyfit(xx2, yy2, 1))
                xp = np.linspace(xx.max() + 0.25 * fac_delta, 1.0)
                yPredicted = yp(xp)

                ax.plot(xp, yPredicted, linestyle=":", color=l.get_color())

                # estimate finish date
                totPredictedMHs = yPredicted.max()
                totRunMHs = yy2.max()
                remainingRunDays = (totPredictedMHs - totRunMHs) * 1e6 / (cpu["numCPUs"] * 24.0)
                predictedFinishDate = datetime.now() + timedelta(days=remainingRunDays)
                predictedFinishStr = predictedFinishDate.strftime("%d %B, %Y")

                print(" Predicted total time: %.1f million CPUhs (%s)" % (totPredictedMHs, predictedFinishStr))
                # pLabels.append( 'Predict: %3.1f MHs (Finish: %s)' % (totPredictedMHs,predictedFinishStr))
                pLabels.append("Predict: %3.1f MHs" % (totPredictedMHs))
                pColors.append(plt.Line2D([0], [0], color=l.get_color(), marker="", linestyle=":"))

                # quick estimate to a specific target redshift
                if 1:
                    targetRedshift = 0.0
                    targetA = 1 / (1 + targetRedshift)
                    _, ww = closest(xp, targetA)
                    print("  * To z = %.3f estimate %.2f Mhs" % (1.0 / xp[ww] - 1.0, yPredicted[ww]))
                    _, ww = closest(xx, targetA)
                    print("  * To z = %.3f estimate %.2f Mhs" % (targetRedshift, yp(targetA)))
                    print("  * To z = %.3f estimate %.2f Mhs" % (1.0 / xx[ww] - 1.0, yy[ww]))

            # total time prediction based on L75n1820TNG and L25n1024_4503 profiles
            # if plotKey in ['total'] and xx.max() < 0.99 and sim.variant == 'None':
            #    sPs_predict = [simParams(res=1820, run='tng')]
            #                   #simParams(res=1024, run='tng', variant='4503')]
            #    ls = ['--','-.']

            #    for j, sP_p in enumerate(sPs_predict):
            #        p_a, p_cpu, p_tot = _cpuEstimateFromOtherRunProfile(sP_p, xx.max(), yy.max())
            #        w = np.where(p_a > xx.max())

            #        # plot
            #        ax.plot(p_a[w], p_cpu[w], linestyle=ls[j], color=l.get_color())

            #        # estimate finish date
            #        remainingRunDays = (p_tot-yy.max()) * 1e6 / (cpu['numCPUs'] * 24.0)
            #        p_date = datetime.now() + timedelta(days=remainingRunDays)
            #        p_str = p_date.strftime('%d %B, %Y')
            #        print(' [w/ %s] Predicted: %.1f million CPUhs (%s)' % (sP_p.simName,p_tot,p_str))

            #        #pLabels.append( ' [w/ %s]: %3.1f MHs (%s)' % (sP_p.simName,p_tot,p_str))
            #        pLabels.append( ' [w/ %s]: %3.1f MHs' % (sP_p.simName,p_tot))
            #        pColors.append( plt.Line2D( [0], [0], color=l.get_color(), marker='', linestyle=ls[j]) )

        _redshiftAxisHelper(ax)

        # add to legend for predictions
        if len(pLabels) > 0:
            pass
            # pLabels.append( '(Last Updated: %s)' % datetime.now().strftime('%d %B, %Y'))
            # pColors.append( plt.Line2D( [0], [0], color='white', marker='', linestyle='-') )
        else:
            pLabels = []
            pColors = []

        # make legend, sim names + extra
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + pColors, labels + pLabels, loc="best")  # , prop={'size':13})

        pdf.savefig()
        plt.close(fig)

    pdf.close()

    # singlepage pdf: all values on one panel
    pdf = PdfPages(fName2)

    for sim in sims:
        fig = plt.figure(figsize=(12.5, 9))

        ax = fig.add_subplot(111)
        ax.set_xlim(xlim)

        ax.set_title("")
        ax.set_xlabel("Scale Factor")

        ind = 3  # 1=diff perc (missing in 3col format), 3=cum perc
        ax.set_ylabel("CPU Percentage")
        keys = ["time", "hatb"] + plotKeys

        # load select datasets from cpu.hdf5
        if sim.run == "tng" and sim.res in [1024, 910, 1820, 1080, 2160, 1250, 2500]:
            hatbMin = 41
        else:
            hatbMin = 0

        cpu = loadCpuTxt(sim.arepoPath, keys=keys, hatbMin=hatbMin)

        # plot each
        for plotKey in plotKeys:
            if "total" in plotKey:
                continue

            if plotKey not in cpu.keys():
                continue  # e.g. hydro fields in DMO runs

            # include only bigish timesteps
            w = np.where(cpu["hatb"] >= cpu["hatb"].max() - 6)

            # loop over each run
            xx = cpu["time"][w]
            yy = np.squeeze(np.squeeze(cpu[plotKey])[w, ind])

            (l,) = ax.plot(xx, yy, label=plotKey)

        _redshiftAxisHelper(ax)

        handles, labels = ax.get_legend_handles_labels()
        pLabels = [sim.simName]
        pColors = [plt.Line2D([0], [0], color="white", marker="", linestyle="-")]
        ax.legend(handles + pColors, labels + pLabels, loc="best")  # , prop={'size':13})

        pdf.savefig()
        plt.close(fig)

    pdf.close()


def plotTimebins():
    """Plot analysis of timebins throughout the course of a run."""
    # run config and load/parse
    saveBase = expanduser("~") + "/timebins_%s.pdf"
    numPtsAvg = 500  # average time series down to N total points

    sPs = []
    sPs.append(simParams(res=128, run="tng", variant="0000"))
    sPs.append(simParams(res=256, run="tng", variant="0000"))
    # sPs.append( simParams(res=512, run='tng', variant='0000') )
    sPs.append(simParams(res=1820, run="tng"))
    sPs.append(simParams(res=2160, run="tng"))
    sPs.append(simParams(res=2500, run="tng"))

    data = []
    for sP in sPs:
        data.append(loadTimebinsTxt(sP.arepoPath))

    # (A) actual wall-clock time of the smallest timebin ('machine weather')
    fig, ax = plt.subplots()

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([1e1, 1e5])
    ax.set_xlabel("Scale Factor")
    ax.set_ylabel("Wall-clock for Smallest Timebin [msec]")
    ax.set_yscale("log")

    # loop over each run
    for i, sP in enumerate(sPs):
        # only plot timesteps where this bin was occupied
        print(" (A) ", sP.simName)

        xx = data[i]["time"]
        yy = data[i]["avg_time"] * 1000.0  # msec

        w = np.where(yy == 0.0)
        yy[w] = np.nan
        yy = np.nanmin(yy, axis=0)  # min avg_time per timestep, across any bin

        # average down to numPtsAvg
        if 0:
            # equal in timestep, not so nice
            avgSize = int(np.floor(yy.size / float(numPtsAvg)))

            xx_avg = xx[0 : avgSize * numPtsAvg].reshape(-1, avgSize)
            xx_avg = np.nanmean(xx_avg, axis=1)
            yy_avg = yy[0 : avgSize * numPtsAvg].reshape(-1, avgSize)
            yy_avg = np.nanmean(yy_avg, axis=1)
        if 1:
            # equal in scalefactor
            da = (xx.max() - 0.0) / numPtsAvg
            xx_avg = np.zeros(numPtsAvg, dtype="float32")
            yy_avg = np.zeros(numPtsAvg, dtype="float32")

            for j in range(numPtsAvg):
                x0 = 0.0 + da * j
                x1 = x0 + da
                w = np.where((xx >= x0) & (xx < x1))
                xx_avg[j] = np.nanmean(xx[w])
                yy_avg[j] = np.nanmean(yy[w])

        # plot
        label = sP.simName
        (l,) = ax.plot(xx_avg, yy_avg, "-", label=label)

    # make redshift axis, legend and finish
    _redshiftAxisHelper(ax)
    ax.legend(loc="best")
    fig.savefig(saveBase % "smallest_msec")
    plt.close(fig)

    # (B) cpu fraction evolution by timebin, one plot per run
    for i, sP in enumerate(sPs):
        # start plot
        print(" (B) ", sP.simName)
        fig = plt.figure(figsize=[figsize[0] * 1.2, figsize[1]])
        ax = fig.add_subplot(111)

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0, 100])
        ax.set_xlabel("Scale Factor")
        ax.set_ylabel("CPU Fraction per Timebin [%]")

        # only plot timesteps where this bin was occupied
        xx = data[i]["time"]
        yy = data[i]["cpu_frac"]

        w = np.where(yy == 0.0)
        yy[w] = np.nan

        # create stack, averaging down to fewer points
        if 0:
            # equal in timestep, not so nice
            avgSize = int(np.floor(yy[0, :].size / float(numPtsAvg)))

            yy_stack = np.zeros((yy.shape[0], numPtsAvg), dtype="float32")
            for ind in range(yy.shape[0]):
                yy_avg = np.squeeze(yy[ind, 0 : avgSize * numPtsAvg]).reshape(-1, avgSize)
                yy_stack[ind, :] = np.nanmean(yy_avg, axis=1)

            # average x-axis down
            xx_avg = xx[0 : avgSize * numPtsAvg].reshape(-1, avgSize)
            xx_avg = np.nanmean(xx_avg, axis=1)

        if 1:
            # equal in scalefactor
            da = (xx.max() - 0.0) / numPtsAvg
            xx_avg = np.zeros(numPtsAvg, dtype="float32")
            yy_stack = np.zeros((yy.shape[0], numPtsAvg), dtype="float32")

            for j in range(numPtsAvg):
                x0 = 0.0 + da * j
                x1 = x0 + da
                w = np.where((xx >= x0) & (xx < x1))

                xx_avg[j] = np.nanmean(xx[w])
                for ind in range(yy.shape[0]):
                    yy_stack[ind, j] = np.nanmean(yy[ind, w])

        w = np.where(np.isnan(yy_stack))
        yy_stack[w] = 0.0

        # plot
        labels = [str(bn) for bn in data[i]["bin_num"][::-1]]  # reverse
        yy_stack = np.flip(yy_stack, axis=0)  # reverse
        ax.stackplot(xx_avg, yy_stack, baseline="zero", labels=labels)

        # make redshift axis, legend and finish
        axTop = _redshiftAxisHelper(ax)

        # shrink current axis by 12%, put a legend to the right of the current axis
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        axTop.set_position([box.x0, box.y0, box.width * 0.88, box.height])
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), prop={"size": 13})

        fig.savefig(saveBase % "cpufrac_stack_%s" % sP.simName)
        plt.close(fig)


def plotTimebinsFrame(pStyle="white", conf=0, timesteps=None):
    """Plot analysis of timebins at one timestep."""
    # run config and load/parse
    barWidth = 0.4
    lw = 4.5

    if timesteps is None:
        timesteps = [6987020]  # 4741250, 6977020

    # sP = simParams(res=256, run='tng', variant='0000')
    sP = simParams(res=2160, run="tng")

    color1, color2, color3, color4 = getWhiteBlackColors(pStyle)

    # load
    data = loadTimebinsTxt(sP.arepoPath)
    xx = data["bin_num"][::-1]

    data["n_grav"] -= data["n_hydro"]  # convert 'grav' (which includes gas) into dm/stars only
    numPart = float(data["n_grav"][:, 0].sum())

    ylim = [5e-11, 3.0]  # [0.5,data['n_hydro'].max()*2.5]

    yticks = [
        numPart / 1e0,
        numPart / 1e1,
        numPart / 1e2,
        numPart / 1e3,
        numPart / 1e4,
        numPart / 1e5,
        numPart / 1e6,
        numPart / 1e7,
        numPart / 1e8,
        numPart / 1e9,
        numPart / 1e10,
    ]
    ytickv = [val / numPart for val in yticks]

    # loop over timesteps
    for i, tsNum in enumerate(timesteps):
        # start plot
        print(tsNum)

        fig = plt.figure(figsize=[19.2, 10.8])
        ax = fig.add_subplot(111, facecolor=color1)
        setAxisColors(ax, color2)

        ax.set_xlim([xx.max() + 1, xx.min() - 1])
        ax.set_ylim(ylim)
        ax.set_xlabel("Timebin")
        ax.set_ylabel("Particle Fraction")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_xticks(xx)
        ax.set_yticks(ytickv)
        ax.set_yticklabels(["$10^{%d}$" % np.log10(val / numPart) for val in yticks])

        # config 0
        yy1 = data["n_hydro"][:, tsNum][::-1] / numPart  # reverse
        yy2 = data["n_grav"][:, tsNum][::-1] / numPart
        yy3 = data["cpu_frac"][:, tsNum][::-1]  # 0-100%

        alpha = 1.0 if conf == 0 else 0.6

        ax.bar(xx - barWidth / 2, yy1, barWidth, label="Hydrodynamical Cells", alpha=alpha)
        ax.bar(xx + barWidth / 2, yy2, barWidth, label="Collisionless DM/Stars", alpha=alpha)

        active = data["active"][:, tsNum][::-1]
        w = np.where(active)
        ax.plot(xx[w] - barWidth / 2, np.zeros(len(w[0])) + 7e-11, "o", markersize=5.0, color=color2, alpha=alpha)

        if conf == 1:
            # add particle fraction line (ax)
            w = np.where(yy1 > 0)
            ax.plot(xx[w] - barWidth / 2, yy1[w], "-", lw=lw, alpha=0.9, color=color2)

        # make top axis (timestep in dscale factor)
        axTop = ax.twiny()
        setAxisColors(axTop, color2)
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(xx)
        topLabels = ["%.1f" % logda for logda in np.log10(data["bin_dt"][::-1])]
        axTop.set_xticklabels(topLabels)
        axTop.set_xlabel(r"Timestep [ log $\Delta a$ ]", labelpad=10)
        axTop.set_xlim(ax.get_xlim())

        # make right axis (particle fraction)
        axRight = ax.twinx()
        setAxisColors(axRight, color2)

        if conf == 0:
            axRight.set_yscale("log")
            axRight.set_yticks(ytickv)
            axRight.set_yticklabels(["$10^{%d}$" % np.log10(val) for val in yticks])
            axRight.set_ylabel("Number of Cells / Particles")
            axRight.set_ylim(ylim)
            axRight.minorticks_off()

        if conf == 1:
            axRight.set_yscale("linear")
            axRight.set_ylabel("Fraction of CPU Time Used by Timebin")
            yticks2 = np.linspace(0, 100, 21)
            axRight.set_yticks(yticks2)
            axRight.set_yticklabels(["%d%%" % v for v in yticks2])
            axRight.set_ylim([0, 30])

            w = np.where(yy3 > 0)
            textOpts = {"fontsize": 22, "color": color2, "horizontalalignment": "center", "verticalalignment": "center"}
            if len(w[0]):
                axRight.plot(xx[w] - barWidth / 2, yy3[w], ":", lw=lw, alpha=0.9, color=color2)
                axRight.text(xx[w][-1] - 1.0, yy3[w][-1], "%.1f%%" % yy3[w][-1], **textOpts)

        # legend/texts
        handles, labels = ax.get_legend_handles_labels()
        sExtra = [
            plt.Line2D([0], [0], color=color2, marker="", lw=0.0),
            plt.Line2D([0], [0], color=color2, marker="", lw=0.0),
        ]
        lExtra = ["ts # %d" % data["step"][tsNum], "z = %7.3f" % (1 / data["time"][tsNum] - 1)]
        if conf == 1:
            sExtra.append(plt.Line2D([0], [0], color=color2, marker="", lw=lw, linestyle=":"))
            lExtra.append("CPU Fraction")
        legend = ax.legend(handles + sExtra, labels + lExtra, loc="upper right")
        for text in legend.get_texts():
            text.set_color(color2)

        fig.savefig("timebins_%s_%04d.png" % (sP.simName, i), facecolor=color1)
        plt.close(fig)


def scalingPlots():
    """Strong (fixed problem size) and weak (Npart scales w/ Ncores) scaling plots."""
    # config
    # seriesName = '201608_scaling_ColumnFFT' # 'scaling_Aug2016_SlabFFT'
    seriesName = "202101_scaling_Hawk"

    basePath = "/virgo/simulations/IllustrisTNG/InitialConditions/tests_%s/" % seriesName
    plotKeys = ["total", "domain", "voronoi", "treegrav", "pm_grav", "hydro"]
    dtInd = 0  # index for column which is the differential time per step
    timestep = 2  # start at the second timestep (first shows strange startup numbers)
    tsMean = 10  # number of timesteps to average over
    figsize = [10.0, 8.0]  # due to second xaxis on top

    pdf = PdfPages(seriesName + ".pdf")

    def _addTopAxisStrong(ax, nCores):
        """Add a second x-axis on top with the exact core numbers."""
        ax.xaxis.set_ticks_position("bottom")
        axTop = ax.twiny()
        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(nCores)
        axTop.set_xticklabels(nCores)
        axTop.minorticks_off()

    def _addTopAxisWeak(ax, nCores, boxSizes, nPartsCubeRoot):
        """Add a second x-axis on top with the 'problem size'."""
        ax.xaxis.set_ticks_position("bottom")
        axTop = ax.twiny()
        axTop.set_xlim(ax.get_xlim())
        axTop.set_xscale(ax.get_xscale())
        axTop.set_xticks(nCores)
        axTop.set_xticklabels(["2$\\times$" + str(nPart) + "${^3}$" for nPart in nPartsCubeRoot])
        axTop.tick_params(axis="both", which="major", labelsize=12)
        # axTop.minorticks_off() # doesn't work
        # axTop.tick_params(which='minor',length=0) # works, but corrupts PDFs somewhat
        axTop.set_xlabel("Weak Scaling: Problem Size [Number of Particles]")
        axTop.xaxis.labelpad = 35

    def _loadHelper(runs, plotKeys):
        # allocate
        nCores = np.zeros(len(runs), dtype="int32")
        data = {}
        for plotKey in plotKeys + ["total_sub"]:
            data[plotKey] = np.zeros(len(runs), dtype="float32")
        for key in ["boxSize", "nPartCubeRoot"]:
            data[key] = np.zeros(len(runs), dtype="int32")

        # loop over each run
        for i, runPath in enumerate(runs):
            # load
            cpu = loadCpuTxt(runPath + "/", skipWrite=True)
            # nSteps = cpu["step"].size

            # verify we are looking at high-z (ICs) scaling runs
            tsInd = np.where(cpu["step"] == timestep)[0]
            assert cpu["step"][0] == 1
            assert len(tsInd) == 1

            # add to save struct
            nCores[i] = cpu["numCPUs"]

            for plotKey in plotKeys:
                loc_data = np.squeeze(cpu[plotKey])
                if plotKey == "total":
                    print("  ", runPath.split("/")[-1], " total sec per timestep: ", loc_data[:, dtInd])
                data[plotKey][i] = np.mean(loc_data[tsInd[0] : tsInd[0] + tsMean, dtInd])

            # extract boxsizes and particle counts from path string
            runName = runPath.split("/")[-1].split("_")[0]
            data["boxSize"][i] = int(runName.split("L")[1].split("n")[0])
            data["nPartCubeRoot"][i] = int(runName.split("n")[1])

            # derive a 'total' which is only the sum of the plotKeys (e.g. disregard i/o scaling)
            data["total_sub"][i] = np.sum([data[plotKey][i] for plotKey in plotKeys if plotKey != "total"])

        # sort based on nCores
        inds = nCores.argsort()
        nCores = nCores[inds]
        for key in data.keys():
            assert len(nCores) == len(data[key])
            data[key] = data[key][inds]

        return nCores, data

    # strong
    # ------
    for runSeries in ["L75n910", "L75n1820"]:
        # loop over runs
        runs = glob(basePath + runSeries + "_*")
        if len(runs) == 0:
            continue
        nCores, data = _loadHelper(runs, plotKeys)

        # print some totals for latex table
        for i in range(len(nCores)):
            print("%6d & %6.1f & %6.2f" % (nCores[i], data["total"][i], data["total"][0] / data["total"][i]))

        # (A) start plot, 'timestep [sec]' vs Ncore
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        # ax.set_xlim([nCores.min()*0.8,nCores.max()*1.2])
        ax.set_xlim([1e3, 1e5])
        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_xlabel("N$_{\\rm cores}$")
        ax.set_ylabel("Time per Step [sec]")

        # add each plotKey
        for plotKey in plotKeys:
            ax.plot(nCores, data[plotKey], marker="s", label=plotKey)

            # add ideal scaling dotted line for each
            xx_max = ax.get_xlim()[1] * 0.97
            xx = [nCores.min() * 0.9, xx_max]
            yy = [data[plotKey][0] / 0.9, data[plotKey][0] / (xx_max / nCores.min())]

            ax.plot(xx, yy, ":", color="#666666", alpha=0.8)

        # legend and finish plot
        ax.text(
            0.98,
            0.97,
            "Strong Scaling [Problem: %s]" % runSeries,
            transform=ax.transAxes,
            size="x-large",
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax.legend(loc="lower left")

        _addTopAxisStrong(ax, nCores)
        pdf.savefig()
        plt.close(fig)

        # (B) start plot, 'efficiency' vs Ncore
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xscale("log")
        ax.set_xlim([1e3, 1e5])
        ax.set_ylim([0.0, 1.1])

        ax.set_xlabel("N$_{\\rm cores}$")
        ax.set_ylabel("Efficiency [t$_{\\rm 0}$ / t$_{\\rm step}$ * (N$_{\\rm 0}$ / N$_{\\rm core}$)]")

        # add each plotKey
        for plotKey in plotKeys:
            eff = data[plotKey][0] / data[plotKey] * (nCores[0] / nCores)
            ax.plot(nCores, eff, marker="s", label=plotKey)

            # add ideal scaling dotted line for each
            xx1 = [nCores.min(), xx_max]
            yy1 = [1.0, 1.0]
            ax.plot(xx1, yy1, ":", color="#666666", alpha=0.8)

        # legend and finish plot
        ax.text(
            0.98,
            0.97,
            "Strong Scaling [Problem: %s]" % runSeries,
            transform=ax.transAxes,
            size="x-large",
            horizontalalignment="right",
            verticalalignment="top",
        )
        ax.legend(loc="lower left")

        _addTopAxisStrong(ax, nCores)
        pdf.savefig()
        plt.close(fig)

    # weak
    # ----
    runSeries = "tL*"

    runs = glob(basePath + runSeries)
    nCores, data = _loadHelper(runs, plotKeys)

    # print some totals for latex table
    for i in range(len(nCores)):
        print("%6d & %6.1f & %6.2f" % (nCores[i], data["total"][i], data["total"][1] / data["total"][i]))

    # (A) start plot, 'timestep [sec]' vs Ncore
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("N$_{\\rm cores}$")
    ax.set_ylabel("Time per Step [sec]")

    # add each plotKey
    for plotKey in plotKeys:
        ax.plot(nCores, data[plotKey], marker="s", label=plotKey)

        # add ideal scaling dotted line
        xx_max = ax.get_xlim()[1] * 0.97
        xx = [nCores.min() * 1.3, xx_max]
        yy = [data[plotKey][0], data[plotKey][0]]

        ax.plot(xx, yy, "-", color="#666666", alpha=0.3)

    # add core count labels above total points, and boxsize labels under particle counts
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] * 1.38  # 1.7
        # ax.text(xx,yy,str(nCores[i]),size='large',ha='center',va='top',color='#999')

        yy = ax.get_ylim()[1] * 1.5
        label = "L%s" % data["boxSize"][i]
        ax.text(xx, yy, label, size="large", ha="center", va="top", color="#999")

    # legend and finish plot
    ax.legend(loc="lower right")
    _addTopAxisWeak(ax, nCores, data["boxSize"], data["nPartCubeRoot"])
    pdf.savefig()
    plt.close(fig)

    # (B) start plot, 'efficiency' vs Ncore
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    ax.set_ylim([0.0, 1.1])

    ax.set_xlabel("N$_{\\rm cores}$")
    ax.set_ylabel("Efficiency [t$_{\\rm 0}$ / t$_{\\rm step}$]")

    # add each plotKey
    for plotKey in plotKeys:
        eff = data[plotKey][0] / data[plotKey]
        ax.plot(nCores, eff, marker="s", label=plotKey)

    # add ideal scaling dotted line
    xx = [nCores.min() * 0.6, xx_max]
    yy = [1.0, 1.0]
    ax.plot(xx, yy, ":", color="#666666", alpha=0.8)

    # add core count labels above total points
    for i in range(len(nCores)):
        xx = nCores[i]
        yy = ax.get_ylim()[0] + 0.05  # 0.08
        # ax.text(xx,yy,str(nCores[i]),size='large',ha='center',va='top',color='#999')

        yy = ax.get_ylim()[1] + 0.10  # 0.15
        label = "L%s" % data["boxSize"][i]
        ax.text(xx, yy, label, size="large", ha="center", va="top", color="#999")

    # legend and finish plot
    ax.legend(loc="lower left")
    _addTopAxisWeak(ax, nCores, data["boxSize"], data["nPartCubeRoot"])
    pdf.savefig()
    plt.close(fig)

    pdf.close()
