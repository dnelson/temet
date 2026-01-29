"""
Diagnostic and production plots based on CLOUDY photo-ionization models.
"""

from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize
from scipy.interpolate import interp1d

from ..cosmo.cloudy import cloudyIon
from ..cosmo.cloudyGrid import loadUVB
from ..cosmo.hydrogen import photoCrossSecGray, photoRate, uvbEnergyDensity
from ..plot.config import figsize, linestyles
from ..plot.util import contourf, loadColorTable, sampleColorTable
from ..util.helper import closest, evenlySample, logZeroNaN, rootPath
from ..util.simParams import simParams


def plotUVB(uvb="FG11"):
    """Debug plots of the UVB(nu) as a function of redshift."""
    # config
    redshifts = [0.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0]
    nusRyd = [0.9, 1.1, 1.7, 1.9, 5.0, 10.0, 100.0]  # [0.9,1.1,3.9,4.1,20.0]

    freq_range = [5e-1, 4e3]
    Jnu_range = [-30, -18]
    Jnu_rangeB = [-26, -18]
    z_range = [0.0, 10.0]

    # load
    uvbs = loadUVB(uvb, redshifts)
    uvbs_all = loadUVB(uvb)

    # (A) start plot: J_nu(nu) at a few specific redshifts
    fig = plt.figure(figsize=(26, 10))

    ax = fig.add_subplot(131)
    ax.set_xlim(freq_range)
    ax.set_ylim(Jnu_range)
    ax.set_xscale("log")

    ax.set_title("")
    ax.set_xlabel(r"$\nu$ [ Ryd ]")
    ax.set_ylabel(r"log J$_{\nu}(\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]")

    for u in uvbs:
        ax.plot(u["freqRyd"], u["J_nu"], label="z = " + str(u["redshift"]))

        val, w = closest(u["freqRyd"], 8.0)
        print(uvb, u["redshift"], 8.0, val, u["J_nu"][w])

    ax.legend()

    # (B) start second plot: J_nu(z) at a few specific nu's
    ax = fig.add_subplot(132)
    ax.set_xlim(z_range)
    ax.set_ylim(Jnu_rangeB)

    ax.set_title("")
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"log J$_{\nu}(\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]")

    for nuRyd in nusRyd:
        xx = []
        yy = []

        for u in uvbs_all:
            _, ind = closest(u["freqRyd"], nuRyd)
            xx.append(u["redshift"])
            yy.append(u["J_nu"][ind])

        ax.plot(xx, yy, label=r"$\nu$ = " + str(nuRyd) + " Ryd")

    ax.legend(loc="lower left")

    # (C) start third plot: 2D colormap of the J_nu magnitude in this plane
    ax = fig.add_subplot(133)
    ax.set_xlim(freq_range)
    ax.set_ylim(z_range)
    ax.set_xscale("log")

    ax.set_title("")
    ax.set_xlabel(r"$\nu$ [ Ryd ]")
    ax.set_ylabel("Redshift")

    # collect data
    x = uvbs_all[0]["freqRyd"]
    y = np.array([uvb["redshift"] for uvb in uvbs_all])
    XX, YY = np.meshgrid(x, y, indexing="ij")

    z = np.zeros((x.size, y.size), dtype="float32")
    for i, u in enumerate(uvbs_all):
        z[:, i] = u["J_nu"]
    z = np.clip(z, Jnu_range[0], Jnu_range[1])

    # render as filled contour
    contourf(XX, YY, z, 40)

    cb = plt.colorbar()
    cb.ax.set_ylabel(r"log J$_{\nu}(\nu)$ [ 4 $\pi$ erg / s / cm$^2$ / Hz ]")

    # finish
    fig.savefig("uvb_%s.pdf" % uvb)
    plt.close(fig)


def plotIonAbundances(res="lg_c17", elements=("Magnesium",)):
    """Debug plots of the cloudy element ion abundance trends with (z,dens,Z,T)."""
    # plot config
    abund_range = [-6.0, 0.0]
    ct = "jet"

    # data config and load full table
    redshift = 0.0
    gridSize = 3  # 3x3

    ion = cloudyIon(sP=simParams(run="tng100-1"), res=res, redshiftInterp=True)

    for element in elements:
        # start pdf, one per element
        pdf = PdfPages("cloudyIons_" + element + "_" + datetime.now().strftime("%d-%m-%Y") + "_" + res + ".pdf")

        # (A): plot vs. temperature, lines for different metals, panels for different densities
        cm = sampleColorTable(ct, ion.grid["metal"].size)

        # loop over all ions of this elemnet
        for ionNum in np.arange(ion.numIons[element]) + 1:
            print(" [%s] %2d" % (element, ionNum))

            fig = plt.figure(figsize=(26, 16))

            for i, dens in enumerate(evenlySample(ion.grid["dens"], gridSize**2)):
                # panel setup
                ax = fig.add_subplot(gridSize, gridSize, i + 1)
                ax.set_title(element + str(ionNum) + " dens=" + str(np.round(dens * 100) / 100))
                ax.set_xlim(ion.range["temp"])
                ax.set_ylim(abund_range)
                ax.set_xlabel("Temp [ log K ]")
                ax.set_ylabel("Log Abundance Fraction")

                # load table slice and plot
                for j, metal in enumerate(ion.grid["metal"]):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)

                    label = "Z = " + str(metal) if np.abs(metal - round(metal)) < 0.00001 else ""
                    ax.plot(T, ionFrac, color=cm[j], label=label)

            ax.legend(loc="upper right")

            pdf.savefig()
            plt.close(fig)

        # (B): plot vs. temperature, lines for different densities, panels for different metals
        cm = sampleColorTable(ct, ion.grid["dens"].size)

        # loop over all ions of this elemnet
        for ionNum in np.arange(ion.numIons[element]) + 1:
            print(" [%s] %2d" % (element, ionNum))

            fig = plt.figure(figsize=(26, 16))

            for i, metal in enumerate(evenlySample(ion.grid["metal"], gridSize**2)):
                # panel setup
                ax = fig.add_subplot(gridSize, gridSize, i + 1)
                ax.set_title(element + str(ionNum) + " metal=" + str(metal))
                ax.set_xlim(ion.range["temp"])
                ax.set_ylim(abund_range)
                ax.set_xlabel("Temp [ log K ]")
                ax.set_ylabel("Log Abundance Fraction")

                # load table slice and plot
                for j, dens in enumerate(ion.grid["dens"]):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)

                    # dens labels only even ints
                    label = "dens = " + str(dens) if np.abs(dens - round(dens / 2) * 2) < 0.00001 else ""
                    ax.plot(T, ionFrac, color=cm[j], label=label)

            ax.legend(loc="upper right")

            pdf.savefig()
            plt.close(fig)

        # (C): 2d histograms (x=T, y=dens, color=log fraction) (different panels for ions)
        metal = 1.0
        ionGridSize = int(np.ceil(np.sqrt(ion.numIons[element])))

        for redshift in np.arange(ion.grid["redshift"].max() + 1):
            print(" [%s] 2d, z = %2d" % (element, redshift))
            fig = plt.figure(figsize=(26, 16))

            for i, ionNum in enumerate(np.arange(ion.numIons[element]) + 1):
                # panel setup
                ax = fig.add_subplot(ionGridSize, ionGridSize, i + 1)
                ax.set_title(element + str(ionNum) + " Z=" + str(metal) + " z=" + str(redshift))
                ax.set_xlim(ion.range["temp"])
                ax.set_ylim(ion.range["dens"])
                ax.set_xlabel("Temp [ log K ]")
                ax.set_ylabel("Density [ log cm$^{-3}$ ]")

                # make 2D array from slices
                x = ion.grid["temp"]
                y = ion.grid["dens"]
                XX, YY = np.meshgrid(x, y, indexing="ij")

                z = np.zeros((x.size, y.size), dtype="float32")
                for j, dens in enumerate(y):
                    _, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    z[:, j] = ionFrac

                z = np.clip(z, abund_range[0], abund_range[1])

                # contour plot
                contourf(XX, YY, z, 40, cmap=ct)
                cb = plt.colorbar()
                cb.ax.set_ylabel("log Abundance Fraction")

            pdf.savefig()
            plt.close(fig)

        # (D): compare ions on same plot: (x=T, y=log fraction) (lines=ions) (panels=dens)
        cm = sampleColorTable(ct, ion.numIons[element])

        for metal in evenlySample(ion.grid["metal"], 3):
            print(" [%s] ion comp, Z = %2d" % (element, metal))

            fig = plt.figure(figsize=(26, 16))

            # load table slice and plot
            for i, dens in enumerate(evenlySample(ion.grid["dens"], gridSize**2)):
                # panel setup
                ax = fig.add_subplot(gridSize, gridSize, i + 1)
                ax.set_title(element + " Z=" + str(metal) + " dens=" + str(np.round(dens * 100) / 100))
                ax.set_xlim(ion.range["temp"])
                ax.set_ylim(abund_range)
                ax.set_xlabel("Temp [ log K ]")
                ax.set_ylabel("Log Abundance Fraction")

                # loop over all ions of this elemnet
                for j, ionNum in enumerate(np.arange(ion.numIons[element]) + 1):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)

                    label = ion._elementNameToSymbol(element) + ion.numToRoman(ionNum)
                    ax.plot(T, ionFrac, color=cm[j], label=label)

            ax.legend(loc="upper right")

            pdf.savefig()
            plt.close(fig)

        # (E): vs redshift (x=T, y=abund) (lines=redshifts) (panels=dens)
        cm = sampleColorTable(ct, int(ion.grid["redshift"].max() + 1))
        metal = -1.0

        for ionNum in np.arange(ion.numIons[element]) + 1:
            print(" [%s] vs redshift, ion = %2d" % (element, ionNum))

            fig = plt.figure(figsize=(26, 16))

            # load table slice and plot
            for i, dens in enumerate(evenlySample(ion.grid["dens"], gridSize**2)):
                # panel setup
                ax = fig.add_subplot(gridSize, gridSize, i + 1)
                ax.set_title(element + str(ionNum) + " Z=" + str(metal) + " dens=" + str(dens))
                ax.set_xlim(ion.range["temp"])
                ax.set_ylim(abund_range)
                ax.set_xlabel("Temp [ log K ]")
                ax.set_ylabel("Log Abundance Fraction")

                # loop over all ions of this elemnet
                for j, redshift in enumerate(np.arange(ion.grid["redshift"].max() + 1)):
                    T, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
                    ax.plot(T, ionFrac, color=cm[j], label="z=%d" % redshift)

            ax.legend(loc="upper right")

            pdf.savefig()
            plt.close(fig)

        pdf.close()


def grackleTable():
    """Plot Grackle cooling table."""
    # filepath = '/u/dnelson/sims.structures/grackle/grackle_data_files/input/'
    # filename1 = 'CloudyData_UVB=FG2011.h5' # orig
    # filename2 = 'grid_cooling_UVB=FG11.hdf5' # my new version (testing)
    ##filename2 = 'CloudyData_UVB=FG2011_shielded.h5' # orig

    filepath = rootPath + "tables/cloudy/"
    # filename1 = 'grid_cooling_UVB=FG20_unshielded.hdf5'
    filename1 = "grid_cooling_UVB=FG20.hdf5"
    filename2 = "grid_cooling_UVB=FG20_ext.hdf5"

    # https://github.com/brittonsmith/cloudy_cooling_tools
    # https://github.com/aemerick/cloudy_tools/blob/master/FG_files/FG_shielded/grackle_cooling_curves.py
    # https://github.com/grackle-project/grackle_data_files/issues/7

    # load
    def _load_grackle_table(fpath):
        """Load the grackle cooling table."""
        d = {}
        with h5py.File(fpath, "r") as f:
            for k1 in f["CoolingRates"]:
                d[k1] = {}
                for k2 in f["CoolingRates"][k1]:
                    d[k1][k2] = f["CoolingRates"][k1][k2][()]

            # constant
            a = dict(f["CoolingRates"]["Metals"]["Cooling"].attrs)

        return d, a

    data1, attrs1 = _load_grackle_table(filepath + filename1)  # e.g. unshielded
    data2, attrs2 = _load_grackle_table(filepath + filename2)  # e.g. shielded

    # check all attrs/param grids are the same
    # for k in attrs:
    #    if isinstance(attrs[k], np.ndarray):
    #        assert np.array_equal(attrs[k],attrs2[k])
    #        continue
    #    assert attrs[k] == attrs2[k]

    # plot config
    lambdanet_range = [-32, -15]
    temp_range = [0.0, 9.0]
    dens_range = [-10.0, 4.0]

    metallicity = 0.01  # multiplies metal cooling rates below...

    # unshielded and shielded separately
    for filename, data, attrs in zip([filename1, filename2], [data1, data2], [attrs1, attrs2]):
        # table dimensions [29, 23, 161] = [nH, z, T]
        hdens = attrs["Parameter1"]  # log cm^-3
        redshift = attrs["Parameter2"]
        temp = np.log10(attrs["Temperature"])  # log K

        # plot book
        pdf = PdfPages("grackle_%s.pdf" % filename)

        # (A) - plot vs. temperature, lines for different dens, pages for different redshifts
        gridSize = 6  # 5*6=30 to cover 29 different densities
        gridSize = int(np.ceil(np.sqrt(hdens.size + 2)))

        for redshift_ind, z in enumerate(redshift[0:1]):  # redshift
            fig = plt.figure(figsize=(36, 22))
            print("[%2d of %2d] z = %.1f (%s)" % (redshift_ind, redshift.size, z, filename))

            # look for strange values
            for k1 in ["Metals", "Primordial"]:
                for k2 in ["Cooling", "Heating"]:
                    w1 = np.where(data[k1][k2][:, redshift_ind, :] == 0.0)
                    assert np.count_nonzero(~np.isfinite(data[k1][k2])) == 0
                    minval = data[k1][k2][:, redshift_ind, :].min()
                    maxval = data[k1][k2][:, redshift_ind, :].max()
                    meanval = data[k1][k2][:, redshift_ind, :].mean()
                    print(
                        "[%s %s] min = %.5g max = %.5g mean = %.5g numzeros = %d"
                        % (k1, k2, minval, maxval, meanval, w1[0].size)
                    )

            for j, nh in enumerate(hdens):
                ax = fig.add_subplot(gridSize, gridSize - 1, j + 1)
                title = "z=%.2f Z=%.2f n=%.1f" % (z, metallicity, nh)
                # ax.set_title(title)
                ax.set_xlim(temp_range)
                ax.set_ylim(lambdanet_range)
                ax.set_xlabel("Temperature [log K]")
                ax.set_ylabel(r"$\Lambda$")

                # derive values and net rates
                cool_z = data["Metals"]["Cooling"][j, redshift_ind, :]
                cool_prim = data["Primordial"]["Cooling"][j, redshift_ind, :]
                heat_z = data["Metals"]["Heating"][j, redshift_ind, :]
                heat_prim = data["Primordial"]["Heating"][j, redshift_ind, :]

                cool_z *= metallicity
                heat_z *= metallicity

                total_cool = cool_z + cool_prim
                total_heat = heat_z + heat_prim

                total_net_cool = np.zeros(total_cool.size, dtype="float32")
                total_net_heat = np.zeros(total_cool.size, dtype="float32")
                total_net_cool.fill(np.nan)
                total_net_heat.fill(np.nan)

                w_cooling = np.where(total_cool >= total_heat)
                w_heating = np.where(total_cool < total_heat)
                total_net_cool[w_cooling] = total_cool[w_cooling] - total_heat[w_cooling]
                total_net_heat[w_heating] = total_cool[w_heating] - total_heat[w_heating]

                # plot
                (l,) = ax.plot(temp, np.log10(total_heat), label="Total Heating")
                ax.plot(temp, np.log10(heat_z), ls="--", color=l.get_color(), label="Metal Heating")
                ax.plot(temp, np.log10(heat_prim), ls=":", color=l.get_color(), label="Prim Heating")

                (l,) = ax.plot(temp, np.log10(total_cool), label="Total Cooling")
                ax.plot(temp, np.log10(cool_z), ls="--", color=l.get_color(), label="Metal Cooling")
                ax.plot(temp, np.log10(cool_prim), ls=":", color=l.get_color(), label="Prim Cooling")

                ax.plot(temp, np.log10(total_net_cool), ls="-", color="#000", label="Net = Cool - Heat")
                ax.plot(temp, np.log10(-total_net_heat), ls=":", color="#000", label="Net (Heating)")

                handles, labels = ax.get_legend_handles_labels()
                ax.legend([plt.Line2D([0], [0], lw=0, marker="")], [title], borderpad=0, loc="upper right")

            # one extra panel for legend
            ax = fig.add_subplot(gridSize, gridSize - 1, j + 2)
            ax.set_axis_off()
            ax.legend(handles, labels, borderpad=0, ncols=2, fontsize=22, loc="best")

            pdf.savefig()
            plt.close(fig)

        pdf.close()

    # (shielded/unshielded) ratios
    prim_cool_ratio = logZeroNaN(data2["Primordial"]["Cooling"] / data1["Primordial"]["Cooling"])
    prim_heat_ratio = logZeroNaN(data2["Primordial"]["Heating"] / data1["Primordial"]["Heating"])
    metal_cool_ratio = logZeroNaN(data2["Metals"]["Cooling"] / data1["Metals"]["Cooling"])
    metal_heat_ratio = logZeroNaN(data2["Metals"]["Heating"] / data1["Metals"]["Heating"])

    prim_net_ratio = logZeroNaN(
        (data2["Primordial"]["Cooling"] - data2["Primordial"]["Heating"])
        / (data1["Primordial"]["Cooling"] - data1["Primordial"]["Heating"])
    )

    metal_net_ratio = logZeroNaN(
        (data2["Metals"]["Cooling"] - data2["Metals"]["Heating"])
        / (data1["Metals"]["Cooling"] - data1["Metals"]["Heating"])
    )

    # gallery of 1d ratio lines
    pdf = PdfPages("grackle_ratio_%s.pdf" % filename2)
    for redshift_ind, z in enumerate(redshift[0:1]):  # redshift
        fig = plt.figure(figsize=(26, 16))
        print("[%2d of %2d] z = %.1f (%s)" % (redshift_ind, redshift.size, z, filename))

        for j, nh in enumerate(hdens):
            ax = fig.add_subplot(gridSize, gridSize - 1, j + 1)
            ax.set_title("z=%.2f n=%.1f" % (z, nh))
            ax.set_xlim(temp_range)
            # ax.set_ylim(lambdanet_range)
            ax.set_xlabel("Temperature [log K]")
            ax.set_ylabel("S/UnS")

            # plot
            (l,) = ax.plot(temp, prim_cool_ratio[j, redshift_ind, :], label="Prim Cooling")
            ax.plot(temp, prim_heat_ratio[j, redshift_ind, :], ls="--", color=l.get_color(), label="Prim Heating")
            (l,) = ax.plot(temp, metal_cool_ratio[j, redshift_ind, :], ls=":", label="Metal Cooling")
            ax.plot(temp, metal_heat_ratio[j, redshift_ind, :], ls="-.", color=l.get_color(), label="Metal Heating")

            handles, labels = ax.get_legend_handles_labels()

        # one extra panel for legend
        ax = fig.add_subplot(gridSize, gridSize - 1, j + 2)
        ax.set_axis_off()
        ax.legend(handles, labels, borderpad=0, ncols=2, loc="best")

        pdf.savefig()
        plt.close(fig)

    pdf.close()

    # 2D: shielded vs unshielded
    extent = [temp_range[0], temp_range[1], dens_range[0], dens_range[1]]
    mm_ratio = [-3.0, 3.0]
    cmap = loadColorTable("balance0", valMinMax=mm_ratio)
    norm = Normalize(vmin=mm_ratio[0], vmax=mm_ratio[1], clip=False)

    mm = [-30, -17]
    norm2 = Normalize(vmin=mm[0], vmax=mm[1], clip=False)

    opts_ratio = {
        "extent": extent,
        "norm": norm,
        "origin": "lower",
        "interpolation": "nearest",
        "aspect": "auto",
        "cmap": cmap,
    }
    opts = {"extent": extent, "norm": norm2, "origin": "lower", "interpolation": "nearest", "aspect": "auto"}

    pdf = PdfPages("grackle_ratio2d_%s.pdf" % filename2)
    for redshift_ind, z in enumerate(redshift[0:1]):  # redshift
        # unshielded only
        for iter in [0, 1]:
            fig = plt.figure(figsize=(26, 16))
            print("[%2d of %2d] z = %.1f (%s)" % (redshift_ind, redshift.size, z, filename))

            if iter == 0:
                data = data2
                label0 = "Shielded"
            else:
                data = data1
                label0 = "Unshielded"

            for j in range(6):
                ax = fig.add_subplot(2, 3, j + 1)
                ax.set_xlabel("Temperature [log K]")
                ax.set_ylabel("Density [log cm$^{-3}$]")

                # select
                if j == 0:
                    d = logZeroNaN(data["Primordial"]["Cooling"][:, redshift_ind, :])
                    label = "Prim Cool (%s)" % label0
                if j == 1:
                    d = logZeroNaN(data["Primordial"]["Heating"][:, redshift_ind, :])
                    label = "Prim Heat (%s)" % label0
                if j == 2:
                    d = logZeroNaN(
                        data["Primordial"]["Cooling"][:, redshift_ind, :]
                        - data["Primordial"]["Heating"][:, redshift_ind, :]
                    )
                    label = "Prim Net (%s)" % label0
                if j == 3:
                    d = logZeroNaN(data["Metals"]["Cooling"][:, redshift_ind, :])
                    label = "Metal Cool (%s)" % label0
                if j == 4:
                    d = logZeroNaN(data["Metals"]["Heating"][:, redshift_ind, :])
                    label = "Metal Heat (%s)" % label0
                if j == 5:
                    d = logZeroNaN(
                        data["Metals"]["Cooling"][:, redshift_ind, :] - data["Metals"]["Heating"][:, redshift_ind, :]
                    )
                    label = "Metal Net (%s)" % label0

                im = ax.imshow(d, **opts)
                cb = fig.colorbar(im)
                cb.ax.set_ylabel("%s [log]" % label)

            pdf.savefig()
            plt.close(fig)

        # ratio
        fig = plt.figure(figsize=(26, 16))
        print("[%2d of %2d] z = %.1f (%s)" % (redshift_ind, redshift.size, z, filename))

        for j in range(6):
            ax = fig.add_subplot(2, 3, j + 1)
            ax.set_xlabel("Temperature [log K]")
            ax.set_ylabel("Density [log cm$^{-3}$]")

            # select
            if j == 0:
                data = prim_cool_ratio[:, redshift_ind, :]
                label = "Prim Cool"
            if j == 1:
                data = prim_heat_ratio[:, redshift_ind, :]
                label = "Prim Heat"
            if j == 2:
                data = prim_net_ratio[:, redshift_ind, :]
                label = "Prim Net"
            if j == 3:
                data = metal_cool_ratio[:, redshift_ind, :]
                label = "Metal Cool"
            if j == 4:
                data = metal_heat_ratio[:, redshift_ind, :]
                label = "Metal Heat"
            if j == 5:
                data = metal_net_ratio[:, redshift_ind, :]
                label = "Metal Net"

            # mask zero (ratio is unity), show as white color
            h2d = np.ma.masked_where(data == 0, data)
            im = ax.imshow(h2d, **opts_ratio)
            cb = fig.colorbar(im)
            cb.ax.set_ylabel("%s (S/UnS log Ratio)" % label)

        pdf.savefig()
        plt.close(fig)

    pdf.close()


def gracklePhotoCrossSec(uvb="FG11"):
    """Plot the photo-ionization cross sections from Grackle. Compare to new derivation."""
    filepath = "/u/dnelson/sims.structures/grackle/grackle_data_files/input/"

    filename = None
    if uvb == "FG11":
        filename = "CloudyData_UVB=FG2011_shielded.h5"  # orig
    if uvb == "HM12":
        filename = "CloudyData_UVB=HM2012_shielded.h5"  # orig

    # load
    if filename is not None:
        with h5py.File(filepath + filename, "r") as f:
            z = f["UVBRates"]["z"][()]
            hei_avg_crs = np.log10(f["UVBRates/CrossSections/hei_avg_crs"][()])
            heii_avg_crs = np.log10(f["UVBRates/CrossSections/heii_avg_crs"][()])
            hi_avg_crs = np.log10(f["UVBRates/CrossSections/hi_avg_crs"][()])

    # compute new cross-sections
    uvbs = loadUVB(uvb)

    z_new = np.array([u["redshift"] for u in uvbs])
    cs_new = {}
    for ion in ["H I", "He I", "He II"]:
        cs_new[ion] = np.zeros(z_new.size, dtype="float32")

    for i, u in enumerate(uvbs):
        J_loc = 10.0 ** u["J_nu"].astype("float64")  # linear

        for ion in ["H I", "He I", "He II"]:
            cs_new[ion][i] = photoCrossSecGray(u["freqRyd"], J_loc, ion=ion)

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Cross Section [ log cm$^2$ ]")
    ax.set_xlim([-0.5, 11.0])
    ax.set_ylim([-18.0, -17.1])

    if filename is not None:
        # comparison of old (grackle included) and new
        (l,) = ax.plot(z, hi_avg_crs, label="H I")
        ax.plot(z_new, np.log10(cs_new["H I"]), ls="--", color=l.get_color(), label="H I (new)")

        (l,) = ax.plot(z, hei_avg_crs, label="He I")
        ax.plot(z_new, np.log10(cs_new["He I"]), ls="--", color=l.get_color(), label="He I (new)")

        (l,) = ax.plot(z, heii_avg_crs, label="He II")
        ax.plot(z_new, np.log10(cs_new["He II"]), ls="--", color=l.get_color(), label="He II (new)")
    else:
        # just new
        ax.plot(z_new, np.log10(cs_new["H I"]), ls="--", label="H I (new)")
        ax.plot(z_new, np.log10(cs_new["He I"]), ls="--", label="He I (new)")
        ax.plot(z_new, np.log10(cs_new["He II"]), ls="--", label="He II (new)")

    ax.legend(loc="upper right")

    fig.savefig("grackle_photocs_%s.pdf" % uvb)
    plt.close(fig)


def grackleReactionRates():
    """Plot the photo-ionization cross sections from Grackle. Compare to new derivation."""
    filepath = "/u/dnelson/sims.structures/grackle/grackle_data_files/input/"

    filenames = [
        "CloudyData_UVB=FG2011_shielded.h5",  # FG11 orig
        "grid_cooling_UVB=FG11_withH2.hdf5",  # FG2011 my new version
        "CloudyData_UVB=HM2012_shielded.h5",  # HM12 orig
        "grid_cooling_UVB=FG20.hdf5",
    ]  # FG20 my new version

    filenames = [
        "CloudyData_UVB=HM2012_shielded.h5",  # HM12 orig
        "grid_cooling_UVB=FG20.hdf5",
    ]  # FG20 my new version

    # load
    data = {}

    for filename in filenames:
        with h5py.File(filepath + filename, "r") as f:
            z = f["UVBRates"]["z"][()]
            k = {}
            for knum in [24, 25, 26, 27, 28, 29, 30, 31]:
                key = f"UVBRates/Chemistry/k{knum}"
                if key not in f:
                    continue
                k[knum] = np.log10(f[key][()])

        uvb = filename.split("UVB=")[1].split("_")[0].split(".")[0]

        if uvb == "FG20":
            k[25][k[25] == -40.0] = -26.0  # just for plotting (enlarges y-range too much)
        if uvb == "FG11":
            k[24] += 0.1  # just for plotting (exactly on top of FG2011)
        if uvb == "FG11":
            k[26] += 0.1  # just for plotting (exactly on top of FG2011)

        # compute new photoreaction rates
        uvbs = loadUVB(uvb.replace("2012", "12").replace("2011", "11"))

        z_local = np.array([u["redshift"] for u in uvbs])
        k24_local = np.zeros(z_local.size, dtype="float32")
        k27_local = np.zeros(z_local.size, dtype="float32")
        k28_local = np.zeros(z_local.size, dtype="float32")
        k29_local = np.zeros(z_local.size, dtype="float32")
        k30_local = np.zeros(z_local.size, dtype="float32")
        k31_local = np.zeros(z_local.size, dtype="float32")

        for i, u in enumerate(uvbs):
            J_loc = 10.0 ** u["J_nu"].astype("float64")  # linear

            k24_local[i] = photoRate(u["freqRyd"], J_loc, ion="H I")  # [1/s]
            k27_local[i] = photoRate(u["freqRyd"], J_loc, ion="k27")
            k28_local[i] = photoRate(u["freqRyd"], J_loc, ion="k28")
            k29_local[i] = photoRate(u["freqRyd"], J_loc, ion="k29")
            k30_local[i] = photoRate(u["freqRyd"], J_loc, ion="k30")
            k31_local[i] = photoRate(u["freqRyd"], J_loc, ion="k31")

        k["k24_custom"] = k24_local
        k["k27_custom"] = k27_local
        k["k28_custom"] = k28_local
        k["k29_custom"] = k29_local
        k["k30_custom"] = k30_local
        k["k31_custom"] = k31_local

        # hack: fix that original grackle fg11 tables miss very last redshift bin
        if uvb == "FG2011":
            for key in k:
                if "_custom" in str(key):
                    k[key] = k[key][:-1]

        data[uvb] = z, k
        print(uvb, k.keys())

        # add directly to (production) grackle table
        if 0 and uvb in ["FG11", "FG20"]:
            print("Careful.", uvb)

            with h5py.File(f"/vera/ptmp/gc/MCST/arepo5_setups/grid_cooling_UVB={uvb}.hdf5", "r+") as f:
                assert f["UVBRates/z"].size == z_local.size
                assert np.array_equal(f["UVBRates/z"][()], z_local)
                f["UVBRates/Chemistry/k27"][:] = k27_local
                f["UVBRates/Chemistry/k28"][:] = k28_local
                f["UVBRates/Chemistry/k29"][:] = k29_local
                f["UVBRates/Chemistry/k30"][:] = k30_local
                f["UVBRates/Chemistry/k31"][:] = k31_local

    # plot k24-26 (primordial_chemistry <= 1)
    fig, ax = plt.subplots()

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Photoionization i.e. Reaction Rate [$s^{-1}$]")
    ax.set_xlim([-0.5, 11.0])
    # ax.set_ylim([-15.0,-11.6])

    for i, uvb in enumerate(data.keys()):
        z, k = data[uvb]

        # if 'k24_custom' in k:
        #    ax.plot(z, k['k24_custom'], ls=linestyles[i], color='black', label='k24_custom (%s)' % uvb)

        if i == 0:
            (l24,) = ax.plot(z, k[24], label="k24 (%s)" % uvb)
            (l25,) = ax.plot(z, k[25], label="k25 (%s)" % uvb)
            (l26,) = ax.plot(z, k[26], label="k26 (%s)" % uvb)
        else:
            ax.plot(z, k[24], ls=linestyles[i], color=l24.get_color(), label="k24 (%s)" % uvb)
            ax.plot(z, k[25], ls=linestyles[i], color=l25.get_color(), label="k25 (%s)" % uvb)
            ax.plot(z, k[26], ls=linestyles[i], color=l26.get_color(), label="k26 (%s)" % uvb)

    ax.legend(loc="lower left")

    fig.savefig("grackle_chemistryk24-26.pdf")
    plt.close(fig)

    # plot k27-31 (primordial_chemistry > 1)
    fig, ax = plt.subplots()

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Photoionization i.e. Reaction Rate [$s^{-1}$]")
    ax.set_xlim([-0.5, 11.0])

    for i, uvb in enumerate(data.keys()):
        z, k = data[uvb]

        # if 'k27_custom' in k:
        #    #ax.plot(z, np.log10(k['k27_custom']+0.02), ls=linestyles[i], color='black', label='k27_custom (%s)' % uvb)
        #    #ax.plot(z, np.log10(k['k28_custom']+0.02), ls=linestyles[i], color='black', label='k28_custom (%s)' % uvb)
        #    #ax.plot(z, np.log10(k['k29_custom']+0.02), ls=linestyles[i], color='black', label='k29_custom (%s)' % uvb)
        #    #ax.plot(z, np.log10(k['k30_custom']+0.02), ls=linestyles[i], color='black', label='k30_custom (%s)' % uvb)
        #    ax.plot(z, np.log10(k['k31_custom']+0.02), ls=linestyles[i], color='black', label='k31_custom (%s)' % uvb)

        if 27 not in k:
            continue

        if i == 0:
            (l27,) = ax.plot(z, k[27], label="k27 (%s)" % uvb)
            (l28,) = ax.plot(z, k[28], label="k28 (%s)" % uvb)
            (l29,) = ax.plot(z, k[29], label="k29 (%s)" % uvb)
            (l30,) = ax.plot(z, k[30], label="k30 (%s)" % uvb)
            (l31,) = ax.plot(z, k[31], label="k31 (%s)" % uvb)
        else:
            ax.plot(z, k[27], ls=linestyles[i], color=l27.get_color(), label="k27 (%s)" % uvb)
            ax.plot(z, k[28], ls=linestyles[i], color=l28.get_color(), label="k28 (%s)" % uvb)
            ax.plot(z, k[29], ls=linestyles[i], color=l29.get_color(), label="k29 (%s)" % uvb)
            ax.plot(z, k[30], ls=linestyles[i], color=l30.get_color(), label="k30 (%s)" % uvb)
            ax.plot(z, k[31], ls=linestyles[i], color=l31.get_color(), label="k31 (%s)" % uvb)

    ax.legend(loc="lower left")

    fig.savefig("grackle_chemistryk27-31.pdf")
    plt.close(fig)


def uvbEnergyDens():
    """Plot the UVB energy density as a function of redshift, integrating over some frequency ranges."""
    # config
    uvb_names = ["HM12", "P19", "FG11", "FG20"]
    eV_min = 6.0  # 11.2
    eV_max = 13.6

    # load
    uvbs_z = []
    uvbs_u = []

    for uvb in uvb_names:
        uvbs = loadUVB(uvb)

        z_local = np.array([u["redshift"] for u in uvbs])
        u_local = np.zeros(z_local.size, dtype="float32")

        for i, u in enumerate(uvbs):
            J_loc = 10.0 ** u["J_nu"].astype("float64")  # linear

            u_local[i] = uvbEnergyDensity(u["freqRyd"], J_loc, eV_min=eV_min, eV_max=eV_max)
            # u_local[i] /= np.pi
            u_local[i] = np.log10(u_local[i])

        print(uvb, z_local[0], u_local[0])

        # fix 7<z<10 oscillations in FG11
        if uvb == "FG11":
            bad_inds = [144,145,149,150,151,155,156,157,161,162,163,167,168,169,174,175,176,181,182,183,188,189,
                        190,191,196,197,198,203,204,205,206,212]  # fmt: skip
            good_mask = np.zeros(z_local.size, dtype="int32")
            good_mask[bad_inds] = 1
            good_inds = np.where(good_mask == 0)

            f_interp = interp1d(z_local[good_inds], u_local[good_inds], kind="linear")
            u_local[bad_inds] = f_interp(z_local[bad_inds])

        uvbs_z.append(z_local)
        uvbs_u.append(u_local)

    # write HDF5 file (used for MCST i.e. SFR_MCS model project)
    with h5py.File("uvb_energydens.hdf5", "w") as f:
        for i, uvb_name in enumerate(uvb_names):
            f["%s/Redshift" % uvb_name] = uvbs_z[i].astype("float32")
            f["%s/EnergyDensity" % uvb_name] = uvbs_u[i]
            f["%s" % uvb_name].attrs["eV_min"] = eV_min
            f["%s" % uvb_name].attrs["eV_max"] = eV_max
            f["%s" % uvb_name].attrs["created_by"] = "dnelson".encode("ascii")
            f["%s" % uvb_name].attrs["created_on"] = datetime.now().strftime("%d-%m-%Y").encode("ascii")

    # add directly to (production) grackle table
    if 0:
        print("Careful.")

        if eV_min == 6.0 and eV_max == 13.6:
            name = "6-13.6eV"
        elif eV_min == 11.2 and eV_max == 13.6:
            name = "11.2-13.6eV"
        else:
            assert 0

        with h5py.File("/virgotng/mpia/MCST/arepo1_setups/grid_cooling_UVB=FG20.hdf5", "r+") as f:
            i = uvb_names.index("FG20")
            f["UVBEnergyDens"].attrs["NumberRedshiftBins"] = uvbs_z[i].size
            if "UVBEnergyDens/Redshift" not in f:
                f["UVBEnergyDens/Redshift"] = uvbs_z[i].astype("float32")
            f["UVBEnergyDens/EnergyDensity_" + name] = uvbs_u[i]

    # plot
    fig, ax = plt.subplots()

    ax.set_xlabel("Redshift")
    ax.set_ylabel("UVB Energy Density %.1f-%.1f eV [ log erg cm$^{-3}$ ]" % (eV_min, eV_max))
    ax.set_xlim([-0.5, 11.0])
    ax.set_ylim([-19.0, -13.0])

    for i, uvb_name in enumerate(uvb_names):
        ax.plot(uvbs_z[i], uvbs_u[i], label=uvb_name)
        # if uvb_name == 'FG11':
        #    ax.plot(uvbs_z[i][bad_inds], uvbs_u[i][bad_inds], 's', color='black')

    val_HM12 = np.log10(1.71e-16)  # 0.00324 * 5.29e-14 # erg/cm^3
    ax.plot([0.0], val_HM12, marker="o", color="black", label="HM12 (old z=0 value)")

    ax.legend(loc="lower left")

    fig.savefig("uvb_energydens.pdf")
    plt.close(fig)


def ionAbundFracs2DHistos(saveName, element="Oxygen", ionNums=(6, 7, 8), redshift=0.0, metal=-1.0):
    """Plot 2D histograms of ion abundance fraction in (density,temperature) space at one Z,z.

    Metal is metallicity in [log Solar].
    """
    # visual config
    abund_range = [-6.0, 0.0]
    nContours = 30
    ctName = "plasma"  #'CMRmap'

    # plot setup
    fig = plt.figure(figsize=[figsize[0] * (len(ionNums) * 0.9), figsize[1]])

    # load
    ion = cloudyIon(sP=simParams(res=455, run="tng"), res="lg", redshiftInterp=True)

    for i, ionNum in enumerate(ionNums):
        # panel setup
        ax = fig.add_subplot(1, len(ionNums), i + 1)
        ax.set_ylim(ion.range["temp"])
        ax.set_xlim(ion.range["dens"])
        ax.set_ylabel(r"Gas Temperature [ log K ]")
        ax.set_xlabel(r"Gas Hydrogen Density n$_{\rm H}$ [ log cm$^{-3}$ ]")  # hydrogen number density

        # make 2D array from slices
        x = ion.grid["temp"]
        y = ion.grid["dens"]
        XX, YY = np.meshgrid(x, y, indexing="ij")

        z = np.zeros((x.size, y.size), dtype="float32")
        for j, dens in enumerate(y):
            _, ionFrac = ion.slice(element, ionNum, redshift=redshift, dens=dens, metal=metal)
            z[:, j] = ionFrac

        z = np.clip(z, abund_range[0], abund_range[1] - 0.1)

        # contour plot
        V = np.linspace(abund_range[0], abund_range[1], nContours)
        ZZ = z  # np.flip(z,axis=1)
        c = contourf(YY, XX, ZZ, V, cmap=ctName)

        labelText = ion._elementNameToSymbol(element) + ion.numToRoman(ionNum)
        ax.text(y[-1] - 0.6, x[0] + 0.3, labelText, va="bottom", ha="right", color="white", fontsize="40")

    # colorbar on last panel only
    cb = fig.colorbar(c, ax=ax, pad=0)
    cb.ax.set_ylabel("Abundance Fraction [ log ]")
    cb.set_ticks(np.linspace(abund_range[0], abund_range[1], int(np.abs(abund_range[0])) + 1))

    fig.savefig(saveName)
    plt.close(fig)
