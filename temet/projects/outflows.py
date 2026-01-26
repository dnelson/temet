"""
Outflows paper (TNG50 presentation): plots.

https://arxiv.org/abs/1902.05554
"""

from functools import partial
from os.path import expanduser, isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from temet.cosmo.mergertree import loadMPBs
from temet.cosmo.time_evo import halosTimeEvoFullbox, halosTimeEvoSubbox, subhalo_subbox_overlap
from temet.plot import gasflows, snapshot, subhalos
from temet.plot.config import linestyles, lw, markers, sKn, sKo
from temet.plot.util import add_halo_size_scales, add_resolution_lines
from temet.projects.outflows_vis import galaxyMosaic_topN, singleHaloDemonstrationImage, subboxOutflowTimeEvoPanels
from temet.util import simParams
from temet.util.helper import closest, evenlySample, logZeroNaN, nUnique, running_median
from temet.util.match import match


def sample_comparison_z2_sins_ao(sP):
    """Compare available galaxies vs. the SINS-AO sample of ~35 systems."""
    from temet.load.data import foersterSchreiber2018

    # config
    xlim = [9.0, 12.0]
    ylim = [-2.5, 4.0]

    msize = 4.0  # marker size for simulated points
    binSize = 0.2  # in M* for median line
    fullSubhaloSFR = True  # use total SFR in subhalo, otherwise within 2rhalf

    # plot setup
    fig, ax = plt.subplots()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_ylabel(r"Star Formation Rate [ log M$_{\rm sun}$ / yr ]")
    ax.set_xlabel(r"Stellar Mass [ log M$_{\rm sun}$ ] [ < 2r$_{1/2}$ ]")

    # load simulation points
    sfrField = "SubhaloSFR" if fullSubhaloSFR else "SubhaloSFRinRad"
    fieldsSubhalos = ["SubhaloMassInRadType", sfrField, "central_flag", "mhalo_200_log"]

    gc = sP.groupCat(fieldsSubhalos=fieldsSubhalos)

    xx_code = gc["SubhaloMassInRadType"][:, sP.ptNum("stars")]
    xx = sP.units.codeMassToLogMsun(xx_code)

    yy = gc[sfrField]

    # centrals only above some mass limit
    with np.errstate(invalid="ignore"):
        ww = np.where((xx > xlim[0] + 0.2) & gc["central_flag"])

    w_nonzero = np.where(yy[ww] > 0.0)
    w_zero = np.where(yy[ww] == 0.0)

    (l,) = ax.plot(xx[ww][w_nonzero], np.log10(yy[ww][w_nonzero]), "o", ms=msize, label=sP.simName)
    ax.plot(xx[ww][w_zero], np.zeros(len(w_zero[0])) + ylim[0] + 0.1, "D", ms=msize, color=l.get_color(), alpha=0.5)

    # median line and 1sigma band
    xm, ym, sm = running_median(xx[ww][w_nonzero], np.log10(yy[ww][w_nonzero]), binSize=binSize, skipZeros=True)
    (l,) = ax.plot(xm[:-1], ym[:-1], "-", alpha=0.4, color=l.get_color())

    y_down = np.array(ym[:-1]) - sm[:-1]
    y_up = np.array(ym[:-1]) + sm[:-1]
    ax.fill_between(xm[:-1], y_down, y_up, color=l.get_color(), interpolate=True, alpha=0.05)

    # observational points (put on top at the end)
    fs = foersterSchreiber2018()
    (l1,) = ax.plot(fs["Mstar"], np.log10(fs["SFR"]), "s", color="#444444", label=fs["label"])

    # find analog of each
    mhalo = np.zeros(fs["Mstar"].size, dtype="float32")

    for i in range(fs["Mstar"].size):
        _, ind = closest(xx[ww], fs["Mstar"][i])
        mhalo[i] = gc["mhalo_200_log"][ww][ind]

        print(i, fs["Mstar"][i], xx[ww][ind], mhalo[i])

    print(np.min(mhalo), np.max(mhalo), np.mean(mhalo), np.median(mhalo))

    # second legend
    ax.legend(loc="upper left")

    fig.savefig("sample_comparison_%s_sfrFullSub=%s.pdf" % (sP.simName, fullSubhaloSFR))
    plt.close(fig)


def stackedRadialProfiles(
    sPs,
    field,
    cenSatSelect="cen",
    projDim="3D",
    xaxis="log_pkpc",
    reBand="jwst_f115w",
    haloMassBins=None,
    mStarBins=None,
    saveName=None,
    pdf=None,
):
    """Plot stacked radial profiles for several stellar mass bins and/or runs i.e. at different redshifts."""
    assert xaxis in ["log_pkpc", "log_rvir", "log_rhalf", "log_re", "pkpc", "rvir", "rhalf", "re"]

    percs = [16, 84]

    # plot setup
    fig, ax = plt.subplots()

    radStr = "Radius" if "3D" in projDim else "Projected Distance"

    if xaxis == "log_rvir":
        ax.set_xlim([-2.5, 0.0])
        ax.set_xlabel(r"%s / Virial Radius [ log ]" % radStr)
    elif xaxis == "rvir":
        ax.set_xlim([0.0, 1.0])
        ax.set_xlabel(r"%s / Virial Radius" % radStr)
    elif xaxis == "log_rhalf":
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel(r"%s / Stellar Half-mass Radius [ log ]" % radStr)
    elif xaxis == "rhalf":
        ax.set_xlim([0, 10])
        ax.set_xlabel(r"%s / Stellar Half-mass Radius" % radStr)
    elif xaxis == "log_pkpc":
        ax.set_xlim([-0.5, 2.0])
        ax.set_xlabel(r"%s [ log pkpc ]" % radStr)
    elif xaxis == "pkpc":
        ax.set_xlim([0, 100])
        ax.set_xlabel(r"%s [ pkpc ]" % radStr)
    elif xaxis == "log_re":
        ax.set_xlim([-0.5, 1.0])
        ax.set_xlabel(r"%s / Stellar R$_{\rm e}$ (JWST f115w) [ log ]" % radStr)
    elif xaxis == "re":
        ax.set_xlim([0, 10])
        ax.set_xlabel(r"%s / Stellar R$_{\rm e}$ (JWST f115w)" % radStr)

    ylabels_3d = {
        "SFR": r"$\dot{\rho}_\star$ [ log M$_{\rm sun}$ yr$^{-1}$ kpc$^{-3}$ ]",
        "Gas_Mass": r"$\rho_{\rm gas}$ [ log M$_{\rm sun}$ kpc$^{-3}$ ]",
        "Stars_Mass": r"$\rho_{\rm stars}$ [ log M$_{\rm sun}$ kpc$^{-3}$ ]",
        "Gas_Fraction": r"f$_{\rm gas}$ = $\rho_{\rm gas}$ / $\rho_{\rm b}$",
        "Gas_Metal_Mass": r"$\rho_{\rm metals}$ [ log M$_{\rm sun}$ kpc$^{-3}$ ]",
        "Gas_Metallicity": r"Gas Metallicity (unweighted) [ log Z$_{\rm sun}$ ]",
        "Gas_Metallicity_sfrWt": r"Gas Metallicity (SFR weighted) [ log Z$_{\rm sun}$ ]",
        "Gas_Bmag": r"Gas Magnetic Field Strength [ log Gauss ]",
    }
    ylims_3d = {
        "SFR": [-10.0, 0.0],
        "Gas_Mass": [0.0, 9.0],
        "Stars_Mass": [0.0, 11.0],
        "Gas_Fraction": [0.0, 1.0],
        "Gas_Metal_Mass": [-4.0, 8.0],
        "Gas_Metallicity": [-2.0, 1.0],
        "Gas_Metallicity_sfrWt": [-1.5, 1.0],
        "Gas_Bmag": [-9.0, -4.0],
    }

    ylabels_2d = {
        "SFR": r"$\dot{\Sigma}_\star$ [ log M$_{\rm sun}$ yr$^{-1}$ kpc$^{-2}$ ]",
        "Gas_Mass": r"$\Sigma_{\rm gas}$ [ log M$_{\rm sun}$ kpc$^{-2}$ ]",
        "Stars_Mass": r"$\Sigma_{\rm stars}$ [ log M$_{\rm sun}$ kpc$^{-2}$ ]",
        "Gas_Fraction": r"f$_{\rm gas}$ = $\Sigma_{\rm gas}$ / $\Sigma_{\rm b}$",
        "Gas_Metal_Mass": r"$\Sigma_{\rm metals}$ [ log M$_{\rm sun}$ kpc$^{-2}$ ]",
        "Gas_Metallicity": ylabels_3d["Gas_Metallicity"],
        "Gas_Metallicity_sfrWt": ylabels_3d["Gas_Metallicity_sfrWt"],
        "Gas_Bmag": ylabels_3d["Gas_Bmag"],
        "Gas_LOSVel_sfrWt": r"Gas Velocity v$_{\rm LOS}$ (SFR weighted) [ km/s ]",
        "Gas_LOSVelSigma": r"Gas Velocity Dispersion $\sigma_{\rm LOS,1D}$ [ km/s ]",
        "Gas_LOSVelSigma_sfrWt": r"Gas Velocity Dispersion $\sigma_{\rm LOS,1D,SFRw}$ [ km/s ]",
    }
    ylims_2d = {
        "SFR": [-6.0, 0.0],
        "Gas_Mass": [3.0, 9.0],
        "Stars_Mass": [3.0, 11.0],
        "Gas_Fraction": [0.0, 1.0],
        "Gas_Metal_Mass": [0.0, 8.0],
        "Gas_Metallicity": ylims_3d["Gas_Metallicity"],
        "Gas_Metallicity_sfrWt": ylims_3d["Gas_Metallicity_sfrWt"],
        "Gas_Bmag": ylims_3d["Gas_Bmag"],
        "Gas_LOSVel_sfrWt": [0, 350],
        "Gas_LOSVelSigma": [0, 350],
        "Gas_LOSVelSigma_sfrWt": [0, 350],
    }

    # only these fields are treated as total sums, and normalized/unit converted appropriately, otherwise we
    # assume the auxCat() profiles are already e.g. mean or medians in the desired units
    totSumFields = ["SFR", "Gas_Mass", "Gas_Metal_Mass", "Stars_Mass"]

    if len(sPs) > 1:
        # multi-redshift, adjust bounds
        ylims_3d["SFR"] = [-10.0, 2.0]
        ylims_2d["SFR"] = [-7.0, 2.0]
        ylims_3d["Gas_Mass"] = [1.0, 10.0]
        ylims_2d["Gas_Mass"] = [4.0, 10.0]
        ylims_3d["Gas_Metallicity"] = [-2.5, 0.5]
        ylims_2d["Gas_Metallicity"] = [-2.5, 0.5]
        ylims_3d["Gas_Metallicity_sfrWt"] = [-2.0, 1.0]
        ylims_2d["Gas_Metallicity_sfrWt"] = [-2.0, 1.0]

    fieldName = "Subhalo_RadProfile%s_FoF_%s" % (projDim, field)

    if field == "Gas_Fraction":
        # handle stellar mass auxCat load and normalization below
        fieldName = "Subhalo_RadProfile%s_FoF_%s" % (projDim, "Gas_Mass")
        fieldName2 = "Subhalo_RadProfile%s_FoF_%s" % (projDim, "Stars_Mass")

    if "3D" in projDim:
        ax.set_ylabel(ylabels_3d[field])
        ax.set_ylim(ylims_3d[field])
    else:
        ax.set_ylabel(ylabels_2d[field])
        ax.set_ylim(ylims_2d[field])

    # init
    colors = []
    rvirs = []
    rhalfs = []
    res = []

    if haloMassBins is not None:
        massField = "mhalo_200_log"
        massBins = haloMassBins
    else:
        massField = "mstar_30pkpc_log"
        massBins = mStarBins

    labelNames = True if nUnique([sP.simName for sP in sPs]) > 1 else False
    labelRedshifts = True if nUnique([sP.redshift for sP in sPs]) > 1 else False

    # loop over each fullbox run
    txt = []

    for i, sP in enumerate(sPs):
        # load halo/stellar masses and CSS
        masses = sP.groupCat(fieldsSubhalos=[massField])

        cssInds = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
        masses = masses[cssInds]

        # load virial radii, tsellar half mass radii, and (optionally) effective optical radii
        rad_rvir = sP.groupCat(fieldsSubhalos=["rhalo_200_code"])
        rad_rhalf = sP.groupCat(fieldsSubhalos=["rhalf_stars_code"])
        rad_rvir = rad_rvir[cssInds]
        rad_rhalf = rad_rhalf[cssInds]

        rad_re = np.zeros(rad_rvir.size, dtype="float32")  # unused by default
        if xaxis in ["log_re", "re"]:
            fieldNameRe = "Subhalo_HalfLightRad_p07c_cf00dust_z_rad100pkpc"
            ac_re = sP.auxCat(fieldNameRe)
            bandInd = list(ac_re[fieldNameRe + "_attrs"]["bands"]).index(reBand)
            rad_re = ac_re[fieldNameRe][:, bandInd]  # code units

        print("[%s]: %s (z=%.1f)" % (field, sP.simName, sP.redshift))

        # load and apply CSS
        ac = sP.auxCat(fields=[fieldName])

        assert ac[fieldName].ndim == 2  # self-halo term only

        # special cases requiring multiple auxCat datasets
        if field == "Gas_Fraction":
            # gas fraction = (M_gas)/(M_gas+M_stars)
            ac2 = sP.auxCat(fields=[fieldName2])
            assert ac2[fieldName2].ndim == 2
            assert np.array_equal(ac["subhaloIDs"], ac2["subhaloIDs"])

            radbins1 = ac[fieldName + "_attrs"]["rad_bins_code"]
            radbins2 = ac2[fieldName2 + "_attrs"]["rad_bins_code"]
            assert np.array_equal(radbins1, radbins2)

            ac[fieldName] = ac[fieldName] / (ac[fieldName] + ac2[fieldName2])

        # crossmatch 'subhaloIDs' to cssInds
        ac_inds, css_inds = match(ac["subhaloIDs"], cssInds)
        ac[fieldName] = ac[fieldName][ac_inds, :]

        masses = masses[css_inds]
        rad_rvir = rad_rvir[css_inds]
        rad_rhalf = rad_rhalf[css_inds]
        rad_re = rad_re[css_inds]
        # sub_inds = cssInds[css_inds]

        yy = ac[fieldName]

        # loop over mass bins
        for k, massBin in enumerate(massBins):
            txt_mb = {}

            # select
            with np.errstate(invalid="ignore"):
                w = np.where((masses >= massBin[0]) & (masses < massBin[1]))

            print(" %s %s [%d] %4.1f - %4.1f : %d" % (field, projDim, k, massBin[0], massBin[1], len(w[0])))
            if len(w[0]) == 0:
                continue

            # radial bins: normalize to rvir, rhalf, or re if requested
            avg_rvir_code = np.nanmedian(rad_rvir[w])
            avg_rhalf_code = np.nanmedian(rad_rhalf[w])
            avg_re_code = np.nanmedian(rad_re[w])

            if (i == 0 and len(massBins) > 1) or (k == 0 and len(sPs) > 1):
                rvirs.append(avg_rvir_code)
                rhalfs.append(avg_rhalf_code)
                res.append(avg_re_code)

            # sum and calculate percentiles in each radial bin
            yy_local = np.squeeze(yy[w, :])

            if xaxis in ["log_rvir", "rvir"]:
                rr = 10.0 ** ac[fieldName + "_attrs"]["rad_bins_code"] / avg_rvir_code
            elif xaxis in ["log_rhalf", "rhalf"]:
                rr = 10.0 ** ac[fieldName + "_attrs"]["rad_bins_code"] / avg_rhalf_code
            elif xaxis in ["log_re", "re"]:
                rr = 10.0 ** ac[fieldName + "_attrs"]["rad_bins_code"] / avg_re_code
            elif xaxis in ["log_pkpc", "pkpc"]:
                rr = ac[fieldName + "_attrs"]["rad_bins_pkpc"]

            # unit conversions: sum per bin to (sum 3D spatial density) or (sum 2D surface density)
            if "3D" in projDim:
                normField = "bin_volumes_code"
                unitConversionFunc = partial(sP.units.codeDensToPhys, totKpc3=True)
            else:
                normField = "bin_areas_code"  # 2D
                unitConversionFunc = partial(sP.units.codeColDensToPhys, totKpc2=True)

            if field in totSumFields:
                yy_local /= ac[fieldName + "_attrs"][normField]  # sum -> (sum/volume) or (sum/area), in code units

            if "_Mass" in field:
                # convert the numerator, e.g. code masses -> msun (so msun/kpc^3)
                yy_local = sP.units.codeMassToMsun(yy_local)

            # resample, integral preserving, to combine poor statistics bins at large distances
            if 1:
                # construct new versions of yy, rr, and normalizations
                shape = np.array(yy_local.shape)
                start_ind = int(shape[1] * 0.4)
                yy_local_new = np.zeros(shape, dtype=yy.dtype)
                rr_new = np.zeros(shape[1], dtype=rr.dtype)
                norm_new = np.zeros(shape[1], dtype=rr.dtype)

                cur_ind = 0
                read_ind = 0
                accum_size = 1

                if field in totSumFields:
                    yy_local *= ac[fieldName + "_attrs"][normField]

                while read_ind < shape[1]:
                    # print('[%d] avg [%d - %d]' % (cur_ind,read_ind,read_ind+accum_size))
                    # copy or average
                    if field in totSumFields:
                        yy_local_new[:, cur_ind] = np.nansum(yy_local[:, read_ind : read_ind + accum_size], axis=1)
                    else:
                        yy_local_new[:, cur_ind] = np.nanmedian(yy_local[:, read_ind : read_ind + accum_size], axis=1)

                    rr_new[cur_ind] = np.nanmean(rr[read_ind : read_ind + accum_size])
                    norm_new[cur_ind] = np.nansum(ac[fieldName + "_attrs"][normField][read_ind : read_ind + accum_size])

                    # update window
                    cur_ind += 1
                    read_ind += accum_size

                    # enlarge averaging region only at large distances
                    if cur_ind >= start_ind and cur_ind % 10 == 0:
                        accum_size += 1

                # re-do normalization and reduce to new size
                yy_local = yy_local_new[:, 0:cur_ind]
                if field in totSumFields:
                    yy_local /= norm_new[0:cur_ind]
                rr = rr_new[0:cur_ind]
                # print('  Note: Resampled yy,rr from [%d] to [%d] total radial bins!' % (shape[1],rr.size))

            if field in totSumFields:
                yy_local = unitConversionFunc(yy_local)  # convert area or volume in code units to pkpc^2 or pkpc^3

            # replace zeros by nan so they are not included in percentiles
            # note: don't want the median to be dragged to zero due to bins with zero particles in individual subhalos
            # rather, accumulate across subhalos and then normalize (i.e. yy_mean), so if we set zero bins to nan
            # here the resulting yy_med (and yp) are similar
            yy_local[yy_local == 0.0] = np.nan

            # calculate totsum profile and scatter
            yy_mean = np.nansum(yy_local, axis=0) / len(w[0])
            yy_med = np.nanmedian(yy_local, axis=0)
            yp = np.nanpercentile(yy_local, percs, axis=0)

            # log both axes and smooth
            if "_LOSVel" not in field and "_Fraction" not in field:
                yy_mean = logZeroNaN(yy_mean)
                yy_med = logZeroNaN(yy_med)
                yp = logZeroNaN(yp)

            if "log_" in xaxis:
                rr = np.log10(rr)

            if rr.size > sKn:
                yy_mean = savgol_filter(yy_mean, sKn + 4, sKo)
                yy_med = savgol_filter(yy_med, sKn + 4, sKo)
                yp = savgol_filter(yp, sKn + 4, sKo, axis=1)

            # if 'Metallicity' in field:
            #    # test: remove noisy last point which is non-monotonic
            #    w = np.where(np.isfinite(yy_med))
            #    if yy_med[w][-1] > yy_med[w][-2]:
            #        yy_med[w[0][-1]] = np.nan

            # extend line to right-edge of x-axis?
            w = np.where(np.isfinite(yy_med))
            xmax = ax.get_xlim()[1]

            if rr[w][-1] < xmax:
                new_ind = w[0].max() + 1
                rr[new_ind] = xmax

                opts = {"kind": "linear", "fill_value": "extrapolate"}
                yy_mean[new_ind] = interp1d(rr[w][-3:], yy_mean[w][-3:], **opts)(xmax)
                yy_med[new_ind] = interp1d(rr[w][-3:], yy_med[w][-3:], **opts)(xmax)
                for j in range(yp.shape[0]):
                    yp[j, new_ind] = interp1d(rr[w][-3:], yp[j, w][:, -3:], **opts)(xmax)

            # plot totsum and/or median line
            if haloMassBins is not None:
                label = r"$M_{\rm halo}$ = %.1f" % (0.5 * (massBin[0] + massBin[1])) if (i == 0) else ""
            else:
                label = r"M$^\star$ = %.1f" % (0.5 * (massBin[0] + massBin[1])) if (i == 0) else ""

            ax.plot(rr, yy_med, color=colors[k], linestyle=linestyles[i], label=label)
            # ax.plot(rr, yy_mean, color=c, linestyle=':', alpha=0.5)

            txt_mb["bin"] = massBin
            txt_mb["rr"] = rr
            txt_mb["yy"] = yy_med
            txt_mb["yy_0"] = yp[0, :]
            txt_mb["yy_1"] = yp[-1, :]

            # draw rvir lines (or 100pkpc lines if x-axis is already relative to rvir)
            add_halo_size_scales(
                ax, sP, field, xaxis, massBins, i, k, avg_rvir_code, avg_rhalf_code, avg_re_code, colors[k]
            )

            # show percentile scatter only for first run
            if i == 0:
                # show percentile scatter only for first/last massbin
                if (k == 0 or k == len(massBins) - 1) or (
                    field == "Gas_LOSVelSigma" in field and k == int(len(massBins) / 2)
                ):
                    ax.fill_between(rr, yp[0, :], yp[-1, :], color=colors[k], interpolate=False, alpha=0.2)

            txt.append(txt_mb)

    # gray resolution band at small radius
    if xaxis in ["log_rvir", "log_pkpc"]:
        add_resolution_lines(ax, sPs, xaxis == "log_rvir", rvirs=rvirs)

    # print
    # for k in range(len(txt)): # loop over mass bins (separate file for each)
    #    filename = 'figX_%s_%sdens_rad%s_m-%.2f.txt' % \
    #      (field,projDim, 'rvir' if radRelToVirRad else 'kpc', np.mean(txt[k]['bin']))
    #    out = '# Nelson+ (in prep) http://arxiv.org/...\n'
    #    out += '# Figure X n_OVI [log cm^-3] (%s z=%.1f)\n' % (sP.simName, sP.redshift)
    #    out += '# Halo Mass Bin [%.1f - %.1f]\n' % (txt[k]['bin'][0], txt[k]['bin'][1])
    #    out += '# rad_logpkpc val val_err0 val_err1\n'
    #    for i in range(1,txt[k]['rr'].size): # loop over radial bins
    #        out += '%8.4f  %8.4f %8.4f %8.4f\n' % \
    #        (txt[k]['rr'][i], txt[k]['yy'][i], txt[k]['yy_0'][i], txt[k]['yy_1'][i])
    #    with open(filename, 'w') as f:
    #        f.write(out)

    # legend
    sExtra = []
    lExtra = []

    if len(sPs) > 1:
        for i, sP in enumerate(sPs):
            sExtra += [plt.Line2D([0], [0], color="black", lw=lw, linestyle=linestyles[i], marker="")]
            label = ""
            if labelNames:
                label = sP.simName
            if labelRedshifts:
                label += " z=%.1f" % sP.redshift
            lExtra += [label.strip()]

    handles, labels = ax.get_legend_handles_labels()
    legendLoc = "upper right"
    if "_Fraction" in field:
        legendLoc = "lower right"  # typically rising not falling with radius
    ax.legend(handles + sExtra, labels + lExtra, loc=legendLoc)

    if pdf is not None:
        pdf.savefig()
    else:
        fig.savefig(saveName)
    plt.close(fig)


def fit_vout():
    """For text discussion and fit equations relating to outflow velocities."""
    import pickle

    import matplotlib.pyplot as plt
    from scipy.optimize import leastsq

    # load
    filename = "vout_75"
    with open("%s_TNG50-1.pickle" % filename, "rb") as f:
        data_z = pickle.load(f)

    # gather data of vout(M*,z)
    nskip = 2  # exclude last N datapoints from each vout(M*) line, due to poor statistics
    tot_len = np.sum([dset["xm"].size - nskip for dset in data_z])

    mstar = np.zeros(tot_len, dtype="float32")
    vout = np.zeros(tot_len, dtype="float32")
    redshift = np.zeros(tot_len, dtype="float32")

    offset = 0

    for dset in data_z:
        count = dset["xm"].size - nskip
        mstar[offset : offset + count] = dset["xm"][:-nskip]
        vout[offset : offset + count] = dset["ym"][:-nskip]
        redshift[offset : offset + count] = dset["redshift"]  # constant
        offset += count

    if 0:
        # least-squares fit
        def _error_function(params, mstar, z, vout):
            """Define error function to minimize."""
            (a, b, c, d, e, f) = params  # v = a + (M*/b)^c + (1+z)^d
            # vout_fit = a * (1+z)**e + (mstar/b)**(c) * (1+z)**d
            # vout_fit = a * (1+z)**e + (mstar/9)**(c) * (1+z)**d
            vout_fit = a * (1 + f * z**2) + b * (1 + z * c) * mstar + (mstar / d) ** e
            # vout_fit = a*np.sqrt(1+z) + b*(1+z) * (mstar/10) + c*(1+z) * (mstar/10)**2
            # vout_fit = np.log10(vout_fit)
            return vout_fit - vout

        params_init = [100.0, 8.0, 1.0, 1.0, 1.0, 1.0]
        # args = (mstar,redshift,np.log10(vout))
        args = (mstar, redshift, vout)

        params_best, params_cov, info, errmsg, retcode = leastsq(
            _error_function, params_init, args=args, full_output=True
        )

        print("params best:", params_best)

        # (A) vs. redshift plot
        fig, ax = plt.subplots()

        ax.set_xlim([0.8 + 1, 6.2 + 1])
        ax.set_ylim([0, 1200])
        # ax.set_ylim([1,3]) # log
        ax.set_xlabel("(1+z)")
        ax.set_ylabel(filename + " [km/s]")

        for mass_ind in [3, 6, 10, 14, 15, 16, 17]:
            # make (x,y) datapoints
            xx = [1 + dset["redshift"] for dset in data_z]
            yy = []
            for dset in data_z:
                if mass_ind < dset["ym"].size:
                    yy.append(dset["ym"][mass_ind])
                else:
                    yy.append(np.nan)

            w = np.where(np.isfinite(yy))
            xx = np.array(xx)[w]
            yy = np.array(yy)[w]

            # plot
            x_mstar = data_z[0]["xm"][mass_ind]  # log msun
            label = r"M$_\star$ = %.2f" % x_mstar
            (l,) = ax.plot(xx, yy, "-", lw=lw, label=label)  # np.log10(yy)

            # plot fit
            x_redshift = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            vout_fit = _error_function(params_best, x_mstar, x_redshift, 0.0)
            ax.plot(1 + x_redshift, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig("%s_vs_redshift.pdf" % filename)
        plt.close(fig)

        # (B) vs M* plot
        fig, ax = plt.subplots()

        ax.set_xlim([7.0, 12.0])
        ax.set_ylim([0, 1200])
        ax.set_xlabel("Stellar Mass [ log Msun ]")
        ax.set_ylabel(filename + " [km/s]")

        for dset in data_z:
            # plot
            x_mstar = dset["xm"]  # log msun
            y_vout = dset["ym"]

            (l,) = ax.plot(x_mstar, y_vout, "-", lw=lw, label="z = %.1f" % dset["redshift"])

            # plot fit
            x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
            vout_fit = _error_function(params_best, x_mstar, dset["redshift"], 0.0)
            ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

        ax.legend()
        fig.savefig("%s_vs_mstar.pdf" % filename)
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    param_z_degree = 1

    def _error_function_loc(params, mstar, vout):
        """Define error function to minimize. Form of vout(M*) at one redshift."""
        (a, b, c) = params
        vout_fit = a + b * (mstar / 10) ** c
        return vout_fit - vout

    def _error_function_loc2(params, mstar, z, vout):
        """Define error function to minimize. Same but with polynomial dependence of (1+z) on every parameter."""
        (a_coeffs, b_coeffs, c_coeffs) = params.reshape(3, param_z_degree + 1)
        a = np.poly1d(a_coeffs)(1 + z)
        b = np.poly1d(b_coeffs)(1 + z)
        c = np.poly1d(c_coeffs)(1 + z)
        vout_fit = a + b * (mstar / 10) ** c
        return vout_fit - vout

    # plot
    fig, ax = plt.subplots()

    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0, 1200])
    ax.set_xlabel("Stellar Mass [ log Msun ]")
    ax.set_ylabel(filename + " [km/s]")

    params_z = []

    # let's fit just each redshift alone
    redshift_targets = np.array([1.0, 2.0, 3.0, 4.0, 6.0])
    scalefac = 1 / (1 + redshift_targets)
    H_z_H0 = np.zeros(scalefac.size, dtype="float64")  # if float32, fails mysteriously in fitting
    for i, z in enumerate(redshift_targets):
        sP = simParams(res=1820, run="tng", redshift=z)
        H_z_H0[i] = sP.units.H_z
    H_z_H0 /= simParams(res=1820, run="tng", redshift=0.0).units.H_z

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc = vout[w]

        params_init = [100.0, 10.0, 1.0]
        args = (mstar_loc, vout_loc)

        params_best, _, _, _, _ = leastsq(_error_function_loc, params_init, args=args, full_output=True)
        params_z.append(params_best)

        print("best fit at [z=%.1f]" % redshift_target, params_best)

        # plot data and fit
        (l,) = ax.plot(mstar_loc, vout_loc, "-", lw=lw, label="z = %.1f" % redshift_target)

        x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
        vout_fit = _error_function_loc(params_best, x_mstar, 0.0)
        ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig("%s_vs_mstar_z_indiv.pdf" % filename)
    plt.close(fig)

    if 1:
        # what is the scaling of vout with (1+z), a, and H(z)?
        def _error_function_plaw(params, xx, vout):
            """Define error function to minimize."""
            (a, b) = params
            vout_fit = a * (xx) ** b
            return vout_fit - vout

        for x_mstar in [8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]:
            vout_fit = np.zeros(len(redshift_targets))

            for i in range(len(redshift_targets)):
                vout_fit[i] = _error_function_loc(params_z[i], x_mstar, 0.0)

            for iterNum in [0, 2]:
                fig, ax = plt.subplots()

                ax.set_ylim([100, 400])
                ax.set_ylabel("fit vout")

                if iterNum == 0:
                    ax.set_xlim([1.5, 7.5])
                    ax.set_xlabel("(1 + Redshift)")
                    xx = 1 + np.array(redshift_targets)
                if iterNum == 1:
                    ax.set_xlim([0.0, 0.6])
                    ax.set_xlabel("Scale factor")
                    xx = scalefac
                if iterNum == 2:
                    ax.set_xlim([0.0, 14.0])
                    ax.set_xlabel("H(z) / H0")
                    xx = H_z_H0

                yy = vout_fit
                ax.plot(xx, yy, "-", marker="o", lw=lw)

                # fit line
                line_fit = np.polyfit(xx, yy, deg=1)
                ax.plot(xx, np.poly1d(line_fit)(xx), ":", lw=lw, marker="s")

                # fit powerlaw
                params_init = [100.0, 1.0]
                args = (xx, yy)

                params_best, _, _, _, _ = leastsq(_error_function_plaw, params_init, args=args, full_output=True)
                print(x_mstar, iterNum, " exponent = %.2f" % params_best[1])
                ax.plot(xx, _error_function_plaw(params_best, xx, 0.0), ":", lw=lw, marker="d")

                # finish
                fig.savefig("%s_vs_time_%d.pdf" % (filename, iterNum))
                plt.close(fig)

    # plot each parameter vs. redshift
    line_fits = []

    for i in range(len(params_z[0])):
        fig, ax = plt.subplots()

        ax.set_xlim([1.5, 7.5])
        # ax.set_ylim([0,1200])
        ax.set_xlabel("(1 + Redshift)")
        ax.set_ylabel("parameter [%d]" % i)

        xx = 1 + np.array(redshift_targets)
        yy = [pset[i] for pset in params_z]

        ax.plot(xx, yy, "-", marker="o", lw=lw)

        line_fit = np.polyfit(xx, yy, deg=param_z_degree)
        print(" param [%d] line: " % i, line_fit)
        ax.plot(xx, np.poly1d(line_fit)(xx), ":", lw=lw, marker="s")
        line_fits.append(line_fit)

        fig.savefig("%s_param_%d_vs_z.pdf" % (filename, i))
        plt.close(fig)

    # -----------------------------------------------------------------------------------------------------

    # re-fit with the given (1+z) dependence allowed for every parameter
    params_init = line_fits
    args = (mstar, vout, redshift)

    if 0:
        print("LOADING ACTUAL GALAXIES TEST")
        mstar_z = []
        vout_z = []
        z_z = []

        for redshift in [1.0, 2.0, 4.0, 6.0]:
            sP = simParams(res=2160, run="tng", redshift=redshift)
            # load mstar and count
            xvals = sP.groupCat(fieldsSubhalos=["mstar_30pkpc_log"])

            acField = "Subhalo_OutflowVelocity_SubfindWithFuzz"
            ac = sP.auxCat(acField)

            mstar = xvals[ac["subhaloIDs"]]
            radInd = 1  # 10kpc
            percInd = 3  # 90, percs = [25,50,75,90,95,99]
            vout = ac[acField][:, radInd, percInd]

            w = np.where(mstar >= 9.0)
            mstar_z.append(mstar[w])
            vout_z.append(vout[w])
            z = np.ones(len(w[0])) + redshift
            z_z.append(z)

        # condense
        mstar2 = np.hstack(mstar_z)
        vout2 = np.hstack(vout_z)
        redshift2 = np.hstack(z_z)

        args = (mstar2, vout2, redshift2)

    params_best, _, _, _, _ = leastsq(_error_function_loc2, params_init, args=args, full_output=True)

    (a_coeffs, b_coeffs, c_coeffs) = params_best.reshape(3, param_z_degree + 1)
    print(filename, "best z-evo fit: ", params_best)
    print(
        "%s = [%.1f + %.1f(1+z)] + [%.1f + %.1f(1+z)] * (log Mstar/10)^{%.1f + %.1f(1+z)}"
        % (filename, a_coeffs[0], a_coeffs[1], b_coeffs[0], b_coeffs[1], c_coeffs[0], c_coeffs[1])
    )

    # plot
    fig, ax = plt.subplots()

    ax.set_xlim([7.0, 12.0])
    ax.set_ylim([0, 1200])
    ax.set_xlabel("Stellar Mass [ log Msun ]")
    ax.set_ylabel(filename + " [km/s]")

    for redshift_target in redshift_targets:
        w = np.where(redshift == redshift_target)

        mstar_loc = mstar[w]
        vout_loc = vout[w]

        # plot data and fit
        (l,) = ax.plot(mstar_loc, vout_loc, "-", lw=lw, label="z = %.1f" % redshift_target)

        x_mstar = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 50)
        vout_fit = _error_function_loc2(params_best, x_mstar, redshift_target, 0.0)
        ax.plot(x_mstar, vout_fit, ":", lw=lw - 1, alpha=1.0, color=l.get_color())

    ax.legend()
    fig.savefig("%s_vs_mstar_z_result.pdf" % filename)
    plt.close(fig)


def halo_selection(sP, minM200=11.5):
    """Make a quick halo selection above some mass limit.

    Sorted based on energy injection in the low BH state between this snapshot and the previous.
    """
    snap = sP.snap
    tage = sP.tage

    r = {}

    # quick caching
    saveFilename = expanduser("~") + "/temp_haloselect_%d_%.1f.hdf5" % (sP.snap, minM200)
    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # halo selection: all centrals above 10^12 Mhalo
    m200 = sP.groupCat(fieldsSubhalos=["mhalo_200_log"])
    with np.errstate(invalid="ignore"):
        w = np.where(m200 >= minM200)
    subInds = w[0]

    print("Halo selection [%d] objects (m200 >= %.1f)." % (len(subInds), minM200))

    # load mergertree for mapping the subhalos between adjacent snapshots
    mpbs = loadMPBs(sP, subInds, fields=["SnapNum", "SubfindID"])

    prevInds = np.zeros(subInds.size, dtype="int32") - 1

    for i in range(subInds.size):
        if subInds[i] not in mpbs:
            continue
        mpb = mpbs[subInds[i]]
        w = np.where(mpb["SnapNum"] == sP.snap - 1)
        if len(w[0]) == 0:
            continue  # skipped sP.snap-1 in the MPB
        prevInds[i] = mpb["SubfindID"][w]

    # restrict to valid matches
    w = np.where(prevInds >= 0)
    print("Using [%d] of [%d] snapshot adjacent matches through the MPBs." % (len(w[0]), prevInds.size))

    prevInds = prevInds[w]
    subInds = subInds[w]

    # compute a delta(BH_CumEgyInjection_RM) between this snapshot and the last
    bh_egyLow_cur = sP.subhalos("BH_CumEgy_low")

    sP.setSnap(snap - 1)
    bh_egyLow_prev = sP.subhalos("BH_CumEgy_low")  # erg

    dt_myr = (tage - sP.tage) * 1000
    sP.setSnap(snap)

    bh_dedt_low = (bh_egyLow_cur[subInds] - bh_egyLow_prev[prevInds]) / dt_myr  # erg/myr

    w = np.where(bh_dedt_low < 0.0)
    bh_dedt_low[w] = 0.0  # bad MPB track? CumEgy counter should be monotonically increasing

    # sort halo sample based on recent BH energy injection in low-state
    sort_inds = np.argsort(bh_dedt_low)[::-1]  # highest to lowest

    r["subInds"] = subInds[sort_inds]
    r["m200"] = m200[r["subInds"]]
    r["bh_dedt_low"] = bh_dedt_low

    # get fof halo IDs
    haloInds = sP.groupCat(fieldsSubhalos=["SubhaloGrNr"])
    r["haloInds"] = haloInds[r["subInds"]]

    # save cache
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]
    print("Saved [%s]." % saveFilename)

    return r


# -------------------------------------------------------------------------------------------------


def outflowVel_obs_data(ax, p):
    """Passed to outflowVel() to add observational data to plot."""
    from temet.load.data import (
        bordoloi14,
        bordoloi16,
        chen10,
        chisholm15,
        cicone16,
        erb12,
        fiore17,
        fluetsch18,
        genzel14,
        heckman15,
        leung17,
        robertsborsani18,
        rubin14,
        rupke05,
        rupke17,
        spence18,
        toba17,
    )

    color = "#555555"
    labels = []

    if "etaM_" in p["xQuant"]:
        # usual theory scalings
        xx = np.array([0.45, 1.47])
        yy = 900 * (10.0**xx) ** (-1.0)  # 'momentum driven', 1000 is arbitrary, this is proportionality only
        if p["ylog"]:
            yy = np.log10(yy)
        ax.plot(xx, yy, "--", color="black", alpha=0.6)

        # txt_pos = [1.12,1.9] # M* min = 7.5
        txt_pos = [1.01, 1.99]  # M* min = 9.0
        ax.text(txt_pos[0], txt_pos[1], r"$\eta_{\rm M} \propto v_{\rm out}^{-1}$", color=color, rotation=-40.0)

        xx = np.array([0.3, 2.06])
        yy = 1000 * (10.0**xx) ** (-0.5)  # 'energy driven'
        if p["ylog"]:
            yy = np.log10(yy)
        ax.plot(xx, yy, "--", color="black", alpha=0.6)

        # txt_pos = [1.2,2.30] # M* min = 7.5
        txt_pos = [1.15, 2.5]  # M* min = 9.0
        ax.text(txt_pos[0], txt_pos[1], r"$\eta_{\rm M} \propto v_{\rm out}^{-2}$", color=color, rotation=-26.0)

        # obs data: v_out vs etaM
        obs = [fiore17(), heckman15(), fluetsch18(), chisholm15(), genzel14(), bordoloi16(), leung17(), rupke05()]
        for i, obs_data in enumerate(obs):
            ax.plot(obs_data["etaM"], obs_data["vout"], markers[i], color=color)
            labels.append(obs_data["label"])

    if "sfr_" in p["xQuant"]:
        # obs data: v_out vs SFR
        obs = chen10()
        for i, ref in enumerate(obs["labels"]):
            w = np.where(obs["ref"] == ref)
            ax.plot(obs["sfr"][w], obs["vout"][w], markers[i], color=color)
            labels.append(ref)

        obs = [robertsborsani18(), rubin14(), bordoloi14(), chisholm15(), cicone16(), fiore17(), erb12()]
        for j, obs_data in enumerate(obs):
            ax.plot(obs_data["sfr"], obs_data["vout"], markers[i + j + 1], color=color)
            labels.append(obs_data["label"])

    if "BolLum" in p["xQuant"]:
        # obs data: v_out vs Lbol
        for i, obs in enumerate([fiore17(), fluetsch18(), leung17(), rupke17(), spence18(), toba17()]):
            ax.plot(obs["Lbol"], obs["vout"], markers[i], color=color)
            labels.append(obs["label"])

    if "mstar" in p["xQuant"] and len(p["redshifts"]) == 1 and not p["v200norm"]:
        if 0:
            # escape velocity curves, via direct summation of the enclosed mass at r < 10 kpc
            ptTypes = ["Stars", "Gas", "DM"]
            acFields = ["Subhalo_Mass_10pkpc_%s" % ptType for ptType in ptTypes]
            ac = p["sP"].auxCat(acFields)

            mass_by_type = np.vstack([ac[acField] for acField in acFields])
            tot_mass_enc = np.sum(mass_by_type, axis=0)

            rad_code = p["sP"].units.physicalKpcToCodeLength(10.0)
            vesc = np.sqrt(2 * p["sP"].units.G * tot_mass_enc / rad_code)  # code velocity units = [km/s]

        if 0 and p["sP"].snapHasField("gas", "Potential"):
            # escape velocity curves, via mean Potential in 10kpc slice for gas cells
            acField = "Subhalo_EscapeVel_10pkpc_Gas"
            vesc = p["sP"].auxCat(acField)[acField]  # physical km/s

            vesc = vesc[p["sub_ids"]]

            # add median line
            assert vesc.shape == xx.shape
            xm, ym, sm, pm = running_median(
                xx, vesc, binSize=p["binSize"], percs=p["percs"], mean=(p["stat"] == "mean"), minNumPerBin=p["minNum"]
            )

            if xm.size > sKn:
                ym = savgol_filter(ym, sKn + 2, sKo)
                pm = savgol_filter(pm, sKn + 2, sKo, axis=1)

            (l,) = ax.plot(xm, ym, ":", alpha=0.3, color="#000000")
            # ax.fill_between(xm[:], pm[0,:], pm[-1,:], color=l.get_color(), interpolate=True, alpha=0.03)
            opts = {"color": "#000000", "alpha": 0.3, "fontsize": 18.0, "va": "bottom"}
            ax.text(xm[6], ym[6] * 1.02, r"$v_{\rm esc,10 kpc}$", rotation=15, **opts)

            # second line: delta potential relative to rvir
            pot_10pkpc = p["sP"].auxCat("Subhalo_Potential_10pkpc_Gas")["Subhalo_Potential_10pkpc_Gas"][p["sub_ids"]]
            pot_rvir = p["sP"].auxCat("Subhalo_Potential_rvir_Gas")["Subhalo_Potential_rvir_Gas"][p["sub_ids"]]

            delta_pot = pot_10pkpc - pot_rvir
            delta_vesc = p["sP"].units.codePotentialToEscapeVelKms(delta_pot)
            xm, ym, sm, pm = running_median(
                xx,
                delta_vesc,
                binSize=p["binSize"],
                percs=p["percs"],
                mean=(p["stat"] == "mean"),
                minNumPerBin=p["minNum"],
            )

            if xm.size > sKn:
                ym = savgol_filter(ym, sKn + 2, sKo)
                pm = savgol_filter(pm, sKn + 2, sKo, axis=1)

            ax.plot(xm, ym, "--", alpha=0.3, color="#000000")
            ax.text(xm[14], ym[14] * 1.10, r"$\Delta v_{\rm esc,10 kpc-rvir}$", rotation=56.0, **opts)

    if "mstar" in p["xQuant"]:
        pass
        # obs data: v_out vs M* (testing)
        # for j, obs in enumerate([chisholm15()]):
        #    ax.plot( obs['mstar'], 10.0**obs['vout'], markers[i+j+1], color='green')
        #    labels.append( obs['label'] )
        # v90 (testing)
        # for j, obs in enumerate([chisholm15()]):
        #    ax.plot( obs['mstar'], 10.0**obs['v90'], markers[i+j+1], color='red')
        #    labels.append( obs['label'] )

    # legend: obs data (add white background to legends)
    legParams = {"frameon": 1, "framealpha": 0.9, "borderpad": 0.2, "fancybox": False}

    loc3 = "upper left" if (p["config"] is None or "loc3" not in p["config"]) else p["config"]["loc3"]
    if len(labels):
        handles = [plt.Line2D([0], [0], color=color, marker=markers[i], linestyle="") for i in range(len(labels))]
        locParams = {} if (p["config"] is None or "leg3white" not in p["config"]) else legParams
        ncol = 1 if (p["config"] is None or "leg3ncol" not in p["config"]) else p["config"]["leg3ncol"]
        legend3 = ax.legend(handles, labels, ncol=ncol, columnspacing=1.0, fontsize=17.0, loc=loc3, **locParams)
        ax.add_artist(legend3)

    # save pickle of data_z for fitting elsewhere
    data = p["data_z"]
    # if len(percIndsPlot) == 1:
    #    import pickle
    #    with open('vout_%d_%s.pickle' % (percs[percIndsPlot[0]],sP.simName), 'wb') as f:
    #        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    # print text file
    filename = "fig6_vel_vs_mstar_z=%.1f.txt" % p["sP"].redshift
    out = "# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n"
    out += "# Figure 6 Outflow Velocity vs. Stellar Mass (%s z=%.1f) (r = %s)\n" % (
        p["sP"].simName,
        p["sP"].redshift,
        data[0]["rad"],
    )
    out += "# M* [log Msun]"
    for entry in data:
        out += ", v_cut=%d" % entry["vperc"]
    out += " (all [km/s])\n"

    for i in range(len(data)):  # make sure all stellar mass values are the same
        assert np.array_equal(data[i]["xm"], data[0]["xm"])

    for i in range(data[0]["xm"].size):
        out += "%7.2f" % data[0]["xm"][i]
        for j in range(len(data)):  # loop over redshifts
            out += " %6.2f" % data[j]["ym"][i]
        out += "\n"

    with open(filename, "w") as f:
        f.write(out)
    print(out)


def outflowRatesStacked_post(ax, p):
    """Passed to outflowRatesStacked() to write data to txt file."""
    txt = p["txt"]

    # Nelson+19 Figure 8: outflow rate vs outflow velocity for multiple M* bins and redshifts
    if "redshift" in txt[0] and txt[0]["redshift"] is not None:
        filename = "fig8_outflowrate_vs_vout_%dkpc.txt" % p["radMidPoint"]
        out = "# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n"
        out += "# Fig 8 Gas Outflow Rate given Outflow Velocity (%s r = %d kpc)\n" % (p["sP"].simName, p["radMidPoint"])
        out += "# Multiple stellar mass bins and redshifts for every entry\n"
        out += "# vel [km/s]"
        for entry in txt:
            out += ", M*=%.1f_z=%.1f" % (entry["mstar"], entry["redshift"])
        out += "\n# (all values after vel are gas mass outflow rate [log msun/yr]) (all masses [log msun])\n"

        for i in range(len(txt)):  # make sure all vel values are the same
            assert np.array_equal(txt[i]["vout"], txt[0]["vout"])

        for i in range(txt[0]["vout"].size):
            if txt[0]["vout"][i] < 0:
                continue
            out += "%4d" % txt[0]["vout"][i]
            for j in range(len(txt)):  # loop over M* bins and redshifts
                out += " %6.3f" % txt[j]["outflowrate"][i]
            out += "\n"

        with open(filename, "w") as f:
            f.write(out)

    # Nelson+19 Figure 11: outflow rate vs angle for multiple inflow rates
    out = "# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n"
    out += "# theta [deg], outflow rate [msun/yr/ster]\n"

    txt = txt[0]
    w = np.where((txt["vout"] >= 0) & (txt["vout"] <= np.pi / 2))
    theta = np.rad2deg(txt["vout"][w])
    mdot = txt["outflowrate"][w]

    for i in range(theta.size):
        out += "%5.2f " % theta[i]
    out += "\n"
    for i in range(mdot.size):
        out += "%5.2f " % mdot[i]
    with open("rate_inflow=%s.txt" % p["inflow"], "w") as f:
        f.write(out)


def outflowRates_post(ax, p):
    """Passed to outflowRates() to plot observational data and write data to txt file."""
    txt = p["txt"]
    sP = p["sP"]
    config = p["config"]

    # print text file
    filename = "fig5_eta_vs_mstar_z=%.1f.txt" % sP.redshift
    out = "# Nelson+ (2019) http://arxiv.org/abs/1902.05554\n"
    out += "# Figure 5 Mass Loading vs. Stellar Mass (%s z=%.1f) (r = %s)\n" % (sP.simName, sP.redshift, txt[0]["rad"])
    out += "# M* [log Msun]"
    for entry in txt:
        out += ", v_cut=%d" % entry["vcut"]
    out += " (all [log km/])\n"

    for i in range(len(txt)):  # make sure all stellar mass values are the same
        assert np.array_equal(txt[i]["mstar"], txt[0]["mstar"])

    for i in range(txt[0]["mstar"].size):
        out += "%7.2f" % txt[0]["mstar"][i]
        for j in range(len(txt)):  # loop over redshifts
            out += " %7.3f" % txt[j]["eta"][i]
        out += "\n"

    with open(filename, "w") as f:
        f.write(out)

    # plot observational data
    from temet.load.data import (
        bordoloi16,
        chisholm15,
        davies18,
        fiore17,
        fluetsch18,
        genzel14,
        heckman15,
        leung17,
        rupke05,
        rupke17,
    )

    color = "#555555"
    labels = []

    if "BolLum" in p["xQuant"]:
        # obs data: etaM vs Lbol
        for i, obs in enumerate([fiore17(), fluetsch18(), leung17(), rupke17()]):
            ax.plot(obs["Lbol"], obs["etaM"], markers[i], color=color)
            labels.append(obs["label"])

    if "sfr_" in p["xQuant"]:
        # obs data: etaM vs SFR
        for i, obs in enumerate([fiore17(), fluetsch18(), chisholm15(), genzel14(), rupke05(), bordoloi16()]):
            ax.plot(obs["sfr"], obs["etaM"], markers[i], color=color)
            labels.append(obs["label"])

    if "_surfdens" in p["xQuant"]:
        # obs data: etaM vs SFR surface density
        for i, obs in enumerate([heckman15(), chisholm15(), davies18()]):
            ax.plot(obs["sfr_surfdens"], obs["etaM"], markers[i], color=color)
            labels.append(obs["label"])

    # legend: obs data
    legParams = {"frameon": 1, "framealpha": 0.9, "fancybox": False}  # to add white background to legends

    loc3 = "upper left" if (config is None or "loc3" not in config) else config["loc3"]
    if len(labels):
        handles = [plt.Line2D([0], [0], color=color, marker=markers[i], ls="") for i in range(len(labels))]
        locParams = {} if (config is None or "leg3white" not in config) else legParams
        ncol = 1 if (config is None or "leg3ncol" not in config) else config["leg3ncol"]
        legend3 = ax.legend(handles, labels, ncol=ncol, columnspacing=1.0, fontsize=17.0, loc=loc3, **locParams)
        ax.add_artist(legend3)


# -------------------------------------------------------------------------------------------------


def run():
    """Perform all the (possibly expensive) analysis for the paper."""
    redshift = 0.73  # last snapshot, 58

    TNG50 = simParams(res=2160, run="tng", redshift=redshift)
    TNG50_3 = simParams(res=540, run="tng", redshift=redshift)

    if 0:
        # print out subbox intersections with selection
        sel = halo_selection(TNG50, minM200=12.0)
        for sbNum in [0, 1, 2]:
            _ = subhalo_subbox_overlap(TNG50, sbNum, sel["subInds"], verbose=True)

    if 0:
        # subbox: save data through time
        sel = halo_selection(TNG50, minM200=12.0)
        # halosTimeEvoSubbox(TNG50, sbNum=0, sel=sel, selInds=[0,1,2,3], minM200=11.5)
        halosTimeEvoSubbox(TNG50, sbNum=2, sel=sel, selInds=[0])

    if 0:
        # fullbox: save data through time, first 20 halos of 12.0 selection all at once
        sel = halo_selection(TNG50, minM200=12.0)
        halosTimeEvoFullbox(TNG50, haloInds=sel["haloInds"][0:20])

    if 0:
        # TNG50_3 test
        sel = halo_selection(TNG50_3, minM200=12.0)
        halosTimeEvoFullbox(TNG50_3, haloInds=sel["haloInds"][0:20])


def paperPlots(sPs=None):
    """Construct all the final plots for the paper."""
    # redshift = 0.73 # snapshot 58, where intermediate trees were constructed
    redshift = 2.0

    TNG50 = simParams(res=2160, run="tng", redshift=redshift)
    TNG100 = simParams(res=1820, run="tng", redshift=redshift)
    # TNG50_2 = simParams(res=1080,run='tng',redshift=redshift)
    # TNG50_3 = simParams(res=540,run='tng',redshift=redshift)
    # TNG50_4 = simParams(res=270,run='tng',redshift=redshift)
    # Eagle   = simParams(run='eagle',redshift=redshift)

    TNG50_z1 = simParams(res=2160, run="tng", redshift=1.0)

    mStarBins = [[7.9, 8.1], [8.9, 9.1], [9.4, 9.6], [9.9, 10.1], [10.3, 10.7], [10.8, 11.2], [11.2, 11.8]]
    mStarBinsSm = [[8.7, 9.3], [9.9, 10.1], [10.3, 10.7], [10.8, 11.2]]  # less
    mStarBinsSm2 = [[7.9, 8.1], [9.4, 9.6], [10.3, 10.7], [10.8, 11.2]]
    redshifts = [1.0, 2.0, 4.0]

    radProfileFields = [
        "SFR",
        "Gas_Mass",
        "Gas_Metal_Mass",
        "Stars_Mass",
        "Gas_Fraction",
        "Gas_Metallicity",
        "Gas_Metallicity_sfrWt",
        "Gas_LOSVelSigma",
        "Gas_LOSVelSigma_sfrWt",
    ]

    quants1 = ["ssfr", "Z_gas", "fgas2", "size_gas", "temp_halo_volwt", "mass_z"]
    quants2 = ["surfdens1_stars", "Z_stars", "color_B_gr", "size_stars", "vout_75_all", "etaM_100myr_10kpc_0kms"]
    quants3 = [
        "nh_halo_volwt",
        "fgas_r200",
        "pratio_halo_volwt",
        "Krot_oriented_stars2",
        "Krot_oriented_gas2",
        "_dummy_",
    ]
    quants4 = ["BH_BolLum", "BH_BolLum_basic", "BH_EddRatio", "BH_dEdt", "BH_CumEgy_low", "BH_mass"]
    quantSets = [quants1, quants2, quants3, quants4]

    # ----------------------------------------------------------------------------------------------------------------

    # (future todo, outside the scope of this paper):
    #  * add r/rvir, v/vir as quantities in instantaneousMassFluxes(), so we can also bin in these
    #  * 'total mass absorption spectra' (or even e.g. MgII, CIV), down the barrel, R_e aperture, need Voigt profile?
    #  * eta_E (energy), eta_p (momentum), eta_Z (metal), dE/dt and dP/dt versus BH and SF energetics (with obs)
    #  * vout vs. M* and etaM vs. M* (with observations)
    #  * plot with time on the x-axis (SFR, BH_Mdot, v_out different rad, eta_out different rad) (all from a subbox?)

    if 0:
        # fig 1: resolution/volume metadata of TNG50 vs other sims
        from temet.plot.cosmoMisc import simResolutionVolumeComparison

        simResolutionVolumeComparison()

    if 0:
        # fig 2, 3: time-series visualization of gas (vel,temp,dens,Z) through a BH outflow/quenching event (subbox)
        subboxOutflowTimeEvoPanels(conf=0, depth=10)
        subboxOutflowTimeEvoPanels(conf=1, depth=10)
        # subboxOutflowTimeEvoPanels(conf=2, depth=25)
        # subboxOutflowTimeEvoPanels(conf=3, depth=25)

    if 0:
        # fig 4: large mosaic of many galaxies (stellar light + gas density)
        galaxyMosaic_topN(numHalosInd=3, panelNum=1)
        galaxyMosaic_topN(numHalosInd=3, panelNum=4)
        galaxyMosaic_topN(numHalosInd=1, panelNum=1, redshift=1.0, hIDsPlot=[20], rotation="face-on")
        galaxyMosaic_topN(numHalosInd=1, panelNum=1, redshift=1.0, hIDsPlot=[20], rotation="edge-on")
        galaxyMosaic_topN(numHalosInd=1, panelNum=4, redshift=2.0, hIDsPlot=[9], rotation="edge-on")
        galaxyMosaic_topN(numHalosInd=1, panelNum=4, redshift=2.0, hIDsPlot=[9], rotation="face-on")

    if 0:
        # fig 5: mass loading as a function of M* at one redshift, three v_rad values with individual markers
        config = {
            "vcutInds": [0, 2, 3],
            "radInds": [1],
            "stat": "mean",
            "ylim": [-0.55, 2.05],
            "markersize": 4.0,
            "addModelTNG": True,
        }
        gasflows.outflowRates(
            TNG50, xQuant="mstar_30pkpc", ptType="total", eta=True, config=config, f_post=outflowRates_post
        )

        # mass loading as a function of M* at one redshift, few variations in both (radius,vcut)
        # config = {'vcutInds':[1,2,4], 'radInds':[1,2,5], 'stat':'mean', 'addModelTNG':True}
        # gasflows.outflowRates(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config)

        # old panel: mass loading 2D contours in (radius,vcut) plane: redshift evolution
        # contours = [-1.5]
        # gasflows.outflowRates2DStacked(TNG50, xAxis='rad', yAxis='vcut', mStarBins=mStarBinsSm, contours=contours,
        # redshifts=redshifts, eta=True)

        # new panel: mass loading vs. redshift in M* bins
        # config = {'vcutInds':[0], 'radInds':[1], 'stat':'mean', 'ylim':[-0.5,2.5]}
        # gasflows.outflowRatesVsRedshift(TNG50, ptType='total', eta=False, config=config)

    if 0:
        # fig 5 (v200norm appendix)
        config = {
            "vcutInds": [0, 5, 11, 12],
            "radInds": [1],
            "stat": "mean",
            "ylim": [-0.55, 2.05],
            "markersize": 4.0,
            "addModelTNG": True,
        }
        gasflows.outflowRates(
            TNG50,
            xQuant="mstar_30pkpc",
            ptType="total",
            eta=True,
            config=config,
            v200norm=True,
            f_post=outflowRates_post,
        )

        # config = {'vcutInds':[1,6,11], 'radInds':[1,2,5], 'stat':'mean', 'addModelTNG':True}
        # gasflows.outflowRates(TNG50, xQuant='mstar_30pkpc', ptType='total', eta=True, config=config, v200norm=True)

    if 0:
        # fig 6: outflow velocity as a function of M* at one redshift, two v_perc values with individual markers
        ylim = [0, 800]
        config = {
            "percInds": [2, 4],
            "radInds": [1],
            "ylim": ylim,
            "stat": "mean",
            "markersize": 4.0,
            "loc2": "upper left",
            "addModelTNG": True,
        }
        gasflows.outflowVel(TNG50, xQuant="mstar_30pkpc", config=config, f_post=outflowVel_obs_data)

        # outflow velocity as a function of M* at one redshift, variations in (radius,v_perc) values
        # config = {'percInds':[3,1], 'radInds':[1,2,4], 'ylim':ylim, 'xlim':[8.5,11.0], 'stat':'mean',
        #           'loc2':'upper left', 'addModelTNG':True}
        # TNG50_z05 = simParams(run='tng50-1',redshift=0.5)
        # gasflows.outflowVel(TNG50, xQuant='mstar_30pkpc', config=config, f_post=outflowVel_obs_data)

        # outflow velocity: redshift evo
        # config = {'percInds':[3], 'radInds':[1], 'ylim':[100,900], 'stat':'mean', 'addModelTNG':True}
        # redshifts_loc = [0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]
        # gasflows.outflowVel(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts_loc, config=config)

    if 0:
        # fig 6 (v200norm appendix)
        ylim = [0, 15]
        config = {
            "percInds": [1, 2, 5],
            "radInds": [2],
            "ylim": ylim,
            "stat": "mean",
            "markersize": 4.0,
            "loc2": "upper right",
        }
        gasflows.outflowVel(TNG50, xQuant="mstar_30pkpc", config=config, v200norm=True, f_post=outflowVel_obs_data)

        # config = {'percInds':[1,2,4], 'radInds':[1,2,4], 'ylim':ylim, 'stat':'mean', 'loc2':'upper left'}
        # gasflows.outflowVel(TNG50, xQuant='mstar_30pkpc', config=config, v200norm=True, f_post=outflowVel_obs_data)

        # config = {'percInds':[2], 'radInds':[1], 'ylim':[0,10], 'stat':'mean', 'loc2':'upper left'}
        # redshifts_loc = [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
        # gasflows.outflowVel(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts_loc, config=config, v200norm=True)

    if 0:
        # fig 7: vrad-rad phase diagram (gas mass weighted), single halo
        sP = simParams(res=2160, run="tng", redshift=0.7)
        haloID = 22

        nBins = 200
        yQuant = "vrad"
        ylim = [-700, 3200]
        clim = [-3.0, 0.0]
        normColMax = True

        def _f_post(ax):
            """Custom behavior."""
            if 0:
                # escape velocity curve, direct from enclosed mass profile
                ptTypes = ["stars", "gas", "dm"]
                haloLen = sP.groupCatSingle(haloID=haloID)["GroupLenType"]
                totSize = np.sum([haloLen[sP.ptNum(ptType)] for ptType in ptTypes])

                offset = 0
                mass = np.zeros(totSize, dtype="float32")
                rad = np.zeros(totSize, dtype="float32")

                for ptType in ptTypes:
                    mass[offset : offset + haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, "mass", haloID=haloID)
                    rad[offset : offset + haloLen[sP.ptNum(ptType)]] = sP.snapshotSubset(ptType, xQuant, haloID=haloID)
                    offset += haloLen[sP.ptNum(ptType)]

                sort_inds = np.argsort(rad)
                mass = mass[sort_inds]
                rad = rad[sort_inds]
                cum_mass = np.cumsum(mass)

                # sample down to N radial points
                rad_code = evenlySample(rad[1:], 100)
                tot_mass_enc = evenlySample(cum_mass[1:], 100)

                if "_kpc" in xQuant:
                    rad_code = sP.units.physicalKpcToCodeLength(rad_code)  # pkpc -> code
                vesc = np.sqrt(2 * sP.units.G * tot_mass_enc / rad_code)  # code velocity units = [km/s]

            if 1:
                # escape velocity curve, directly from potential of particles
                vesc = sP.snapshotSubset("dm", "vesc", haloID=haloID)
                rad = sP.snapshotSubset("dm", xQuant, haloID=haloID)

                sort_inds = np.argsort(rad)
                vesc = vesc[sort_inds]
                rad = rad[sort_inds]

                rad_code = evenlySample(rad, 100, logSpace=True)
                vesc = evenlySample(vesc, 100, logSpace=True)

            # if xlog:
            #    rad_code = np.log10(rad_code)
            ax.plot(rad_code[1:], vesc[1:], "-", lw=lw, color="#000000", alpha=0.5)
            # ax.text(rad_code[-17], vesc[-17]*1.02, r'$v_{\rm esc}(r)$', color='#000000', alpha=0.5,
            #         fontsize=18.0, va='bottom', rotation=-4.0)

        snapshot.phaseSpace2d(
            sP,
            partType="gas",
            xQuant="rad_kpc",
            xlim=[0.0, 2.0],
            haloIDs=[haloID],
            yQuant=yQuant,
            ylim=ylim,
            nBins=nBins,
            normColMax=normColMax,
            clim=clim,
            median=False,
            f_post=_f_post,
        )
        # snapshot.phaseSpace2d(sP, partType='gas', xQuant='rad_kpc_linear', xlim=[0,80], haloIDs=[haloID],
        #    yQuant=yQuant, ylim=ylim, nBins=nBins, normColMax=normColMax, clim=clim, f_post=f_post)

    if 0:
        # fig 8: distribution of radial velocities
        config = {"radInd": 2, "stat": "mean", "ylim": [-3.0, 1.0]}

        gasflows.outflowRatesStacked(TNG50, quant="vrad", mStarBins=mStarBins, config=config)
        # redshifts_loc = [0.2,0.5,1.0,2.0,4.0]
        # gasflows.outflowRatesStacked(TNG50, quant='vrad', mStarBins=mStarBins, config=config, redshifts=redshifts_loc)
        # gasflows.outflowRatesStacked(TNG50, quant='vrad', mStarBins=mStarBinsSm2, config=config, redshift=redshifts)

    if 0:
        # fig 9: radial velocities of outflows (2D): dependence on temperature, for one m* bin
        mStarBins = [[10.7, 11.3]]

        config = {"stat": "mean", "radInd": [3]}
        gasflows.outflowRates2DStacked(
            TNG50, xAxis="temp", yAxis="vrad", mStarBins=mStarBins, clims=[[-3.5, 0.0]], config=config
        )

    if 0:
        # fig 10: outflow rates vs a single quantity (marginalized over all others), one redshift, stacked in M* bins
        config = {"radInd": 2, "vcutInd": 2, "stat": "mean", "ylim": [-3.0, 2.0]}

        gasflows.outflowRatesStacked(TNG50, quant="temp", mStarBins=mStarBins, config=config)
        # gasflows.outflowRatesStacked(TNG50, quant='temp_sfcold', mStarBins=mStarBins, config=config)
        gasflows.outflowRatesStacked(TNG50, quant="numdens", mStarBins=mStarBins, config=config)
        # gasflows.outflowRatesStacked(TNG50, quant='z_solar', mStarBins=mStarBins, config=config)
        config["ylim"] = [-3.0, 1.0]
        gasflows.outflowRatesStacked(
            TNG50, quant="z_solar", mStarBins=mStarBinsSm2[0:3], redshifts=redshifts, config=config
        )

    if 0:
        # fig 11: angular dependence: 2D histogram of outflow rate vs theta, for 2 demonstrative stellar mass bins
        mStarBins = [[9.8, 10.2], [10.8, 11.2]]
        clims = [[-2.2, -1.0], [-1.0, -0.2]]
        config = {"stat": "mean", "vcutInd": [3, 5]}

        gasflows.outflowRates2DStacked(
            TNG50_z1, xAxis="rad", yAxis="theta", mStarBins=mStarBins, clims=clims, config=config
        )

    if 0:
        # fig 12: visualization of bipolar SN-wind driven outflow:
        # gas density LIC-convolved with vel field, overlaid with streamlines
        singleHaloDemonstrationImage(conf=2)

    if 0:
        # fig 13: (relative) outflow velocity vs. Delta_SFMS, as a function of M*
        sP = TNG50_z1

        xQuant = "mstar_30pkpc_log"
        cQuant = "vout_50_2.5kpc"
        yQuant = "delta_sfms"
        cRel = [0.65, 1.35, False]  # [cMin,cMax,cLog] #None
        ylim = [-0.75, 1.25]
        params = {
            "cenSatSelect": "cen",
            "cStatistic": "median_nan",
            "cQuant": cQuant,
            "xQuant": xQuant,
            "ylim": ylim,
            "cRel": cRel,
        }

        subhalos.histogram2d(sP, yQuant=yQuant, **params)

        # inset: trend of relative vout with delta_MS for two M* slices
        xQuant = "delta_sfms"
        sQuant = "mstar2_log"
        sRange = [[9.4, 9.6], [10.4, 10.8]]
        xlim = [-1.0, 0.5]
        yRel = [0.65, 1.25, False, r"$v_{\rm out}$ / $v_{\rm out,median}$"]  # [cMin,cMax,cLog] #None
        sizefac = 0.4
        css = "cen"
        yQuant = cQuant  #'vout_50_20kpc'

        pdf = PdfPages("slice_%s_%d_x=%s_y=%s_s=%s_%s.pdf" % (sP.simName, sP.snap, xQuant, yQuant, sQuant, css))
        subhalos.slice(
            [sP],
            xQuant=xQuant,
            yQuants=[yQuant],
            sQuant=sQuant,
            sRange=sRange,
            xlim=xlim,
            yRel=yRel,
            sizefac=sizefac,
            cenSatSelect=css,
            pdf=pdf,
        )
        pdf.close()

    if 0:
        # fig 14: fraction of 'fast outflow' galaxies, in the Delta_SFMS vs M* plane
        sP = TNG50_z1

        xQuant = "mstar_30pkpc_log"
        nBins = 50
        yQuant = "delta_sfms"

        cQuant = "vout_90_10kpc"
        cFrac = [300, np.inf, False, r"Fraction w/ Fast Outflows ($v_{\rm out}$ > 300 km/s)"]

        params = {"cenSatSelect": "cen", "cStatistic": "median_nan", "cQuant": cQuant, "cFrac": cFrac, "nBins": nBins}

        subhalos.histogram2d(sP, xQuant=xQuant, yQuant=yQuant, **params)

    if 0:
        # fig 15: observational comparisons, many panels of outflow velocity vs. galaxy/BH properties
        percs = [5, 95]  # show breath of vout scatter
        minMstar = 9.0  # log msun
        binSize = 0.25  # dex in x-axis
        radInds = [1]  # 10 kpc
        percInds = [0, 1, 2, 4, 5]  # vN pecentile indices
        stat = "mean"

        # vout vs. etaM
        config = {
            "percInds": percInds,
            "radInds": radInds,
            "ylim": [1.4, 3.7],
            "stat": stat,
            "binSize": binSize,
            "loc1": None,
            "loc2": "lower right",
            "loc3": "upper left",
            "leg3white": True,
            "leg3ncol": 3,
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
            "ylabel": r"Outflow Velocity $v_{\rm out}$ [ log km/s ]",
            "xlabel": r"Mass Loading $\eta_{\rm M}$ [ log ]",
        }
        gasflows.outflowVel(
            TNG50_z1, xQuant="etaM_100myr_10kpc_0kms", ylog=True, config=config, f_post=outflowVel_obs_data
        )

        # vout vs. SFR
        config = {
            "percInds": percInds,
            "radInds": radInds,
            "xlim": [-1.5, 2.6],
            "ylim": [1.45, 3.65],
            "stat": stat,
            "binSize": binSize,
            "loc1": None,
            "loc2": "lower right",
            "leg3ncol": 2,
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
            "ylabel": r"Outflow Velocity $v_{\rm out}$ [ log km/s ]",
            "xlabel": r"Star Formation Rate [ log M$_{\rm sun}$ yr$^{-1}$ ]",
        }
        gasflows.outflowVel(TNG50_z1, xQuant="sfr_30pkpc_100myr", ylog=True, config=config, f_post=outflowVel_obs_data)

        # vout vs. Lbol
        config = {
            "percInds": percInds,
            "radInds": radInds,
            "ylim": [1.35, 3.65],
            "xlim": [40.0, 47.5],
            "stat": stat,
            "binSize": binSize,
            "loc1": None,
            "loc2": "lower right",
            "leg3ncol": 2,
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
            "ylabel": r"Outflow Velocity $v_{\rm out}$ [ log km/s ]",
        }
        gasflows.outflowVel(TNG50_z1, xQuant="BH_BolLum", ylog=True, config=config, f_post=outflowVel_obs_data)

        # fig 15: observational comparisons, many panels of etaM vs. galaxy/BH properties
        vcutInds = [0, 2, 4]  # vrad>X cut indices
        stat = "median"

        # etaM vs. SFR
        config = {
            "vcutInds": vcutInds,
            "radInds": radInds,
            "stat": stat,
            "xlim": [-1.0, 2.6],
            "ylim": [-1.1, 2.6],
            "binSize": binSize,
            "loc1": "upper right",
            "loc2": None,
            "leg3white": True,
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
            "xlabel": r"Star Formation Rate [ log M$_{\rm sun}$ yr$^{-1}$ ]",
        }
        gasflows.outflowRates(
            TNG50_z1, ptType="total", xQuant="sfr_30pkpc_100myr", eta=True, config=config, f_post=outflowRates_post
        )

        # etaM vs. Lbol
        config = {
            "vcutInds": vcutInds,
            "radInds": radInds,
            "stat": stat,
            "ylim": [-1.1, 2.6],
            "xlim": [40.0, 47.5],
            "binSize": binSize,
            "loc1": "upper right",
            "leg1white": True,
            "loc2": None,
            "loc3": "upper left",
            "leg3ncol": 2,
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
        }
        gasflows.outflowRates(
            TNG50_z1, ptType="total", xQuant="BH_BolLum", eta=True, config=config, f_post=outflowRates_post
        )

        # etaM vs. Sigma_SFR (?)
        config = {
            "vcutInds": vcutInds,
            "radInds": radInds,
            "stat": stat,
            "xlim": [-3.0, 2.0],
            "ylim": [-1.1, 2.6],
            "binSize": binSize,
            "loc1": "upper right",
            "leg1white": True,
            "loc2": None,
            "loc3": "upper left",
            "markersize": 0.0,
            "percs": percs,
            "minMstar": minMstar,
        }
        gasflows.outflowRates(
            TNG50_z1, ptType="total", xQuant="sfr1_surfdens", eta=True, config=config, f_post=outflowRates_post
        )

    if 0:
        # fig 16: stacked radial profiles of SFR surface density
        sP = simParams(run="tng50-1", redshift=0.0)  # TNG50_z1
        cenSatSelect = "cen"
        field = "SFR"
        projDim = "2Dz"
        xaxis = "log_pkpc"

        pdf = PdfPages("radprofiles_%s-%s-%s_%s_%d_%s.pdf" % (field, projDim, xaxis, sP.simName, sP.snap, cenSatSelect))
        stackedRadialProfiles(
            [sP], field, xaxis=xaxis, cenSatSelect=cenSatSelect, projDim=projDim, mStarBins=mStarBins, pdf=pdf
        )
        pdf.close()

    # ----------------------------------------------------------------------------------------------------------------
    if 0:
        # fig 15b: test more accurate vout vs. M* scaling plot, matched in percentile/etc to a single obs
        sP = simParams(res=2160, run="tng", redshift=0.27)  # chisholm+15 <redshift>
        ylim = [0, 800]
        massField = "SiII"  #'Masses', SiII' (chisholm+15, one of the ions used, absorption, highly variable aperture)
        proj2D = True  # line of sight, down the barrel scheme (instead of 3D vrad)

        config = {"percInds": [1, 3], "radInds": [13], "ylim": ylim, "stat": "mean", "loc2": "upper right"}
        gasflows.outflowVel(sP, xQuant="mstar_30pkpc", config=config, massField=massField, proj2D=proj2D)

    if 0:
        # explore: sample comparison against SINS-AO survey at z=2 (M*, SFR)
        TNG100.setRedshift(2.0)
        sample_comparison_z2_sins_ao(TNG100)

    if 0:
        # explore: outflow velocity as a function of M* at one redshift
        gasflows.outflowVel(TNG50, xQuant="mstar_30pkpc", ylog=True)
        # gasflows.outflowVel(TNG50, xQuant='mstar_30pkpc', redshifts=redshifts)

        # explore: outflow velocity as a function of etaM at one redshift
        gasflows.outflowVel(TNG50_z1, xQuant="etaM_100myr_10kpc_0kms", ylog=True)
        gasflows.outflowVel(TNG50_z1, xQuant="etaM_100myr_20kpc_0kms", ylog=True)
        gasflows.outflowVel(TNG50_z1, xQuant="etaM_100myr_all_0kms", ylog=True)

    if 0:
        # explore: net outflow rates (and mass loading factors), fully marginalized, as a function of stellar mass
        for ptType in ["Gas", "Wind", "total"]:
            gasflows.outflowRates(TNG50, xQuant="mstar_30pkpc", ptType=ptType, eta=False)
        gasflows.outflowRates(TNG50, xQuant="mstar_30pkpc", ptType="total", eta=True)

    if 0:
        # explore: net outflow rate distributions, vs a single quantity (marginalized over others), stacked in M* bins
        for quant in ["temp", "numdens", "z_solar", "theta", "vrad"]:
            gasflows.outflowRatesStacked(TNG50, quant=quant, mStarBins=mStarBins, f_post=outflowRatesStacked_post)

            # explore: redshift dependence
            gasflows.outflowRatesStacked(
                TNG50, quant=quant, mStarBins=mStarBinsSm2, redshifts=redshifts, f_post=outflowRatesStacked_post
            )

    if 0:
        # explore: specific ion outflow properties (e.g. MgII) at face value
        massField = "MgII"

        config = {
            "percInds": [2, 4],
            "radInds": [1],
            "ylim": [0, 800],
            "stat": "mean",
            "markersize": 4.0,
            "addModelTNG": True,
        }
        gasflows.outflowVel(TNG50, xQuant="mstar_30pkpc", config=config, massField=massField)

        config = {"vcutInds": [1, 2, 4], "radInds": [1, 2, 5], "stat": "mean", "addModelTNG": True}
        gasflows.outflowRates(TNG50, xQuant="mstar_30pkpc", ptType="Gas", eta=True, config=config, massField=massField)

    if 0:
        # explore: net 2D outflow rates/mass loadings, for several M* bins in three different configurations:
        #  (i) multi-panel, one per M* bin, at one redshift
        #  (ii) single-panel, many contours for each M* bin, at one redshift
        #  (ii) single-panel, one contour for each M* bin, at multiple redshifts
        # 2D here render binConfigs: 2,3,4, 5,6, 7,8,9,10,11 (haven't actually used #1...)
        for eta in [True, False]:
            # special case:
            z_contours = [-0.5 - 1 * eta]

            gasflows.outflowRates2DStacked(
                TNG50, xAxis="rad", yAxis="vcut", mStarBins=mStarBinsSm, clims=[[-3.0, 2.0 - 1 * eta]], eta=eta
            )
            gasflows.outflowRates2DStacked(
                TNG50, xAxis="rad", yAxis="vcut", mStarBins=mStarBinsSm, contours=[-0.5, 0.0, 0.5], eta=eta
            )
            gasflows.outflowRates2DStacked(
                TNG50,
                xAxis="rad",
                yAxis="vcut",
                mStarBins=mStarBinsSm,
                contours=z_contours,
                redshifts=redshifts,
                eta=eta,
            )

            # set contour levels
            if eta:
                contours = [-2.5, -2.0]
            else:
                contours = [-1.0, -0.5, 0.0]  # msun/yr

            optsSets = [
                {"mStarBins": mStarBinsSm, "contours": contours, "eta": eta},  # multi-contour, single redshift
                {
                    "mStarBins": mStarBinsSm,
                    "contours": z_contours,
                    "redshifts": redshifts,
                    "eta": eta,
                },  # single-contour, multi-redshift
                {"mStarBins": mStarBinsSm, "clims": [[-3.0, 0.5 - 1 * eta]], "eta": eta},
            ]  # no contours, instead many panels

            for opts in optsSets:
                for quant in ["temp", "numdens", "z_solar", "theta"]:
                    gasflows.outflowRates2DStacked(TNG50, xAxis="rad", yAxis=quant, **opts)
                    gasflows.outflowRates2DStacked(TNG50, xAxis=quant, yAxis="vcut", **opts)

                gasflows.outflowRates2DStacked(TNG50, xAxis="numdens", yAxis="temp", **opts)
                gasflows.outflowRates2DStacked(TNG50, xAxis="z_solar", yAxis="temp", **opts)
                gasflows.outflowRates2DStacked(TNG50, xAxis="temp", yAxis="theta", **opts)
                gasflows.outflowRates2DStacked(TNG50, xAxis="z_solar", yAxis="theta", **opts)

                gasflows.outflowRates2DStacked(TNG50, xAxis="temp", yAxis="vrad", **opts)
                gasflows.outflowRates2DStacked(TNG50, xAxis="rad", yAxis="vrad", **opts)

    if 0:
        # explore: radial profiles: stellar mass stacks, at one redshift
        TNG50.setRedshift(1.0)
        if sPs is None:
            sPs = [TNG50]
        cenSatSelect = "cen"

        for field in radProfileFields:
            pdf = PdfPages("radprofiles_%s_%s_%d_%s.pdf" % (field, sPs[0].simName, sPs[0].snap, cenSatSelect))

            for projDim in ["2Dz", "3D", "2Dfaceon", "2Dedgeon"]:
                if projDim == "3D" and "_LOSVel" in field:
                    continue  # undefined
                if projDim in ["2Dfaceon", "2Dedgeon"] and "LOSVel" not in field:
                    continue  # just 2Dz,3D for most

                for xaxis in ["log_pkpc", "pkpc", "log_rvir", "rvir", "log_rhalf", "rhalf", "log_re", "re"]:
                    stackedRadialProfiles(
                        sPs,
                        field,
                        xaxis=xaxis,
                        cenSatSelect=cenSatSelect,
                        projDim=projDim,
                        mStarBins=mStarBins,
                        pdf=pdf,
                    )
            pdf.close()

        return sPs

    if 0:
        # explore: radial profiles: vs redshift, separate plot for each mstar bin
        redshifts = [1.0, 2.0, 4.0, 6.0]
        css = "cen"

        if sPs is None:
            sPs = []
            for redshift in redshifts:
                sP = simParams(res=2160, run="tng", redshift=redshift)
                sPs.append(sP)

        for field in radProfileFields:
            pdf = PdfPages("radprofiles_%s_%s_zevo_%s.pdf" % (field, sPs[0].simName, css))

            for projDim in ["2Dz", "3D", "2Dfaceon", "2Dedgeon"]:
                if projDim == "3D" and "_LOSVel" in field:
                    continue  # undefined
                if projDim in ["2Dfaceon", "2Dedgeon"] and "LOSVel" not in field:
                    continue  # just 2Dz,3D for most

                for xaxis in ["log_pkpc", "log_rvir", "log_rhalf", "pkpc"]:
                    for mStarBin in mStarBins:
                        stackedRadialProfiles(
                            sPs, field, xaxis=xaxis, cenSatSelect=css, projDim=projDim, mStarBins=[mStarBin], pdf=pdf
                        )

            pdf.close()

        return sPs

    # exploration: eta_M vs stellar/halo mass, split by everything else
    if 0:
        sPs = [TNG50]

        # quants = quantList(wCounts=False, wTr=False, wMasses=True)
        quants = ["ssfr"]
        priQuant = "vout_99_all"  #'etaM_100myr_10kpc_0kms'

        opts = {"cenSatSelect": "cen", "sLowerPercs": [10, 50], "sUpperPercs": [90, 50]}

        for xQuant in ["mstar_30pkpc", "mhalo_200_log"]:
            # individual plot per y-quantity:
            pdf = PdfPages("medianTrends_%s_x=%s_%s_slice=%s.pdf" % (sPs[0].simName, xQuant, css, priQuant))
            for yQuant in quants:
                subhalos.median(sPs, yQuants=[yQuant], xQuant=xQuant, sQuant=priQuant, pdf=pdf, **opts)
            pdf.close()

            # individual plot per s-quantity:
            pdf = PdfPages("medianTrends_%s_x=%s_%s_y=%s.pdf" % (sPs[0].simName, xQuant, css, priQuant))
            for sQuant in quants:
                subhalos.median(sPs, yQuants=[priQuant], xQuant=xQuant, sQuant=sQuant, pdf=pdf, **opts)

            pdf.close()

    # exporation: 2d histos of new quantities (delta_sfms) vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160, run="tng", redshift=1.0)
        xQuants = ["mstar_30pkpc_log", "mhalo_200_log"]
        nBins = 50
        yQuant = "size_stars"  #'delta_sfms'
        cRel = None  # [0.7,1.3,False] # [cMin,cMax,cLog] #None

        # cQuant = 'vout_75_all'
        # cFrac  = [250, np.inf, False, r'Fast Outflow Fraction ($v_{\rm out}$ > 250 km/s)'] #[200, np.inf, False]

        cQuant = "delta_sfms"  #'etaM_100myr_10kpc_0kms'
        cFrac = [20.0, np.inf, False, None]

        params = {
            "cenSatSelect": "cen",
            "cStatistic": "median_nan",
            "cQuant": cQuant,
            "cRel": cRel,
            "cFrac": cFrac,
            "nBins": nBins,
        }

        for xQuant in xQuants:
            subhalos.histogram2d(sP, xQuant=xQuant, yQuant=yQuant, **params)

    # exploration: 2d histos of everything vs M*, color on e.g. eta/vout
    if 0:
        sP = simParams(res=2160, run="tng", redshift=1.0)
        xQuants = ["mstar_30pkpc_log", "mhalo_200_log"]
        nBins = 50

        cQuants = ["vout_75_all"]  # ,'etaM_100myr_10kpc_0kms']
        cRel = [0.7, 1.3, False]  # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == "_dummy_":
                quants3[-1] = xQuants[1 - i]  # include the other

            for cQuant in cQuants:
                # for each (x) quant, make a number of 6-panel figures, different y-axis (same coloring) for every panel
                params = {
                    "cenSatSelect": "cen",
                    "cStatistic": "median_nan",
                    "cQuant": cQuant,
                    "xQuant": xQuant,
                    "cRel": cRel,
                    "nBins": nBins,
                }

                for yQuants in [quants2]:  # quantSets):
                    subhalos.histogram2d(sP, yQuant=yQuants[0], **params)
                    subhalos.histogram2d(sP, yQuant=yQuants[1], **params)
                    subhalos.histogram2d(sP, yQuant=yQuants[2], **params)
                    subhalos.histogram2d(sP, yQuant=yQuants[3], **params)
                    subhalos.histogram2d(sP, yQuant=yQuants[4], **params)
                    subhalos.histogram2d(sP, yQuant=yQuants[5], **params)

    # exploration: 2d histos of new quantities (vout,eta,BH_BolLum,etc) vs M*, colored by everything else
    if 0:
        sP = simParams(res=2160, run="tng", redshift=1.0)
        xQuants = ["mstar_30pkpc_log", "mhalo_200_log"]
        nBins = 50

        yQuants = ["vout_90_all", "etaM_100myr_10kpc_0kms", "delta_sfms"]
        cRel = None  # [0.7,1.3,False] # None

        for i, xQuant in enumerate(xQuants):
            if quants3[-1] == "_dummy_":
                quants3[-1] = xQuants[1 - i].replace("_log", "")  # include the other

            for yQuant in yQuants:
                # for each (x,y) quant set, make a number of 6-panel figures, different coloring for every panel
                params = {
                    "cenSatSelect": "cen",
                    "cStatistic": "median_nan",
                    "yQuant": yQuant,
                    "xQuant": xQuant,
                    "cRel": cRel,
                    "nBins": nBins,
                }

                for cQuants in quantSets:
                    subhalos.histogram2d(sP, cQuant=cQuants[0], **params)
                    subhalos.histogram2d(sP, cQuant=cQuants[1], **params)
                    subhalos.histogram2d(sP, cQuant=cQuants[2], **params)
                    subhalos.histogram2d(sP, cQuant=cQuants[3], **params)
                    subhalos.histogram2d(sP, cQuant=cQuants[4], **params)
                    subhalos.histogram2d(sP, cQuant=cQuants[5], **params)

    # exploration: outlier check (for Fig 14 discussion/text)
    if 0:
        sP = TNG50_z1
        nBins = 50

        xQuants = ["sfr_30pkpc_100myr", "BH_BolLum", "sfr1_surfdens", "etaM_100myr_10kpc_0kms"]
        yQuants = ["vout_95_10kpc_log", "etaM_100myr_10kpc_0kms"]
        cQuants = ["mstar_30pkpc_log", "ssfr"]

        cRel = None
        cFrac = None

        params = {"cenSatSelect": "cen", "cStatistic": "median_nan", "cRel": cRel, "cFrac": cFrac, "nBins": nBins}

        for yQuant in yQuants:
            for xQuant in xQuants:
                for cQuant in cQuants:
                    subhalos.histogram2d(sP, xQuant=xQuant, yQuant=yQuant, cQuant=cQuant, **params)
