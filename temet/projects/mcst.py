"""
MCST: exploratory plots / intro paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from scipy.stats import binned_statistic_2d

from temet.load.simtxt import blackhole_details_mergers, sf_sn_details
from temet.plot import snapshot, subhalos_evo
from temet.plot.config import colors, figsize, linestyles, lw, markers
from temet.plot.cosmoMisc import simHighZComparison
from temet.plot.subhalos import addUniverseAgeAxis
from temet.plot.util import colored_line
from temet.projects.mcst_vis import (
    vis_gallery_galaxy,
    vis_highres_region,
    vis_movie_mpbsm,
    vis_movie_mpbsm_multi,
    vis_parent_box,
    vis_single_galaxy,
    vis_single_halo,
    vis_single_large,
)
from temet.util import simParams
from temet.util.helper import cache, logZeroNaN


def _get_existing_sims(variants, res, hInds, redshift, all=False, single=False):
    """Return a list of simulation objects, only for those runs which exist (and have reached redshift).

    Args:
      variants (list[str]): list of simulation variants to include.
      res (list[int]): list of resolutions to include.
      hInds (list[int]): list of halo indices to include.
      redshift (float): target redshift.
      all (bool): if False, only include sims with |dz| < 0.1 of target redshift. Otherwise all.
      single (bool): if True, only include the highest available resolution for each halo/variant combination.
    """
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"

    sims = []
    for hInd in hInds:
        for variant in variants:
            found_maxres = 0
            for r in res:
                try:
                    sim = simParams(run="structures", res=r, hInd=hInd, variant=variant, redshift=redshift)
                    if np.abs(sim.redshift - redshift) < 0.3 or all:
                        if single:
                            # if multiple resolutions exist, only keep the highest (but require that it is done)
                            higher_res_is_done = np.abs(sim.redshift - redshift) < 0.1

                            if sim.res > found_maxres and (higher_res_is_done or found_maxres == 0):
                                if len(sims) > 0 and sims[-1].hInd == hInd and sims[-1].variant == variant:
                                    assert sims[-1].res < sim.res, "Error in single highest-res selection."
                                    sims.pop()
                                    print(LINE_UP, end=LINE_CLEAR)  # remove previous line of stdout
                                    print(sim, " [OK]")
                                sims.append(sim)
                                print(sim, " [OK] -- selected")
                            found_maxres = sim.res
                        else:
                            sims.append(sim)
                            print(sim, " [OK]")
                    else:
                        raise Exception
                except Exception:
                    print(f"h{hInd}_L{r}_{variant} z={redshift:.1f}  [does not exist, skip]")

    return sims


def _zoomSubhaloIDsToPlot(sim, min_mhalo=7.5, verbose=False):
    """Define a common rule for which subhalo(s) to plot for a given zoom run."""
    subhaloIDs = [sim.zoomSubhaloID]

    # all centrals with stellar mass and low contamination
    contam_frac = sim.subhalos("contam_frac")
    # num_lowres = sim.subhalos('SubhaloLenType')[:,sim.ptNum('dmlowres')]
    cen_flag = sim.subhalos("cen_flag")
    mstar = sim.subhalos("mstar2_log")
    mhalo = sim.subhalos("mhalo_log")
    grnr = sim.subhalos("SubhaloGrNr")

    w = np.where((contam_frac < 1e-3) & (cen_flag == 1) & (mstar > 0) & (mhalo > min_mhalo))[0]

    subhaloIDs = w

    print(f"[{sim}] Showing {len(subhaloIDs)} subhalos.")

    for subid in subhaloIDs:
        # lowres_dist = sim.snapshotSubset('dmlowres', 'rad_kpc', subhaloID=subid)
        info_str = f" h[{grnr[subid]}] sub[{subid:4d}] "
        info_str += f"mhalo = {mhalo[subid]:.2f} "
        info_str += f"mstar = {mstar[subid]:.2f} "
        info_str += f"contam_frac = {contam_frac[subid]:.3g}"
        print(info_str)

    # go through first 10 halos also, just for information purposes
    firstsub = sim.halos("GroupFirstSub")
    num_lowres = sim.halos("GroupLenType")[:, sim.ptNum("dmlowres")]

    if verbose:
        print("first ten halos:")
        for i in range(10):
            subid = firstsub[i]
            info_str = f" h[{i}] sub[{subid:5d}] "
            info_str += f"mhalo = {mhalo[subid]:.2f} "
            info_str += f"mstar = {mstar[subid]:.1f} "
            info_str += f"{num_lowres[i] =:4d} "
            info_str += f"contam_frac = {contam_frac[subid]:.3g}"
            print(info_str)

    return subhaloIDs


def smhm_relation(sims):
    """Stellar mass vs halo mass including empirical constraints."""
    from temet.load.data import behrooziUM  # paquereau25

    xQuant = "mhalo_200_log"
    yQuant = "mstar2_log"
    xlim = [7.3, 10.3]
    ylim = [4.0, 8.5]  # log mstar

    def _draw_data(ax, sims):
        # Behroozi+2019 (UniverseMachine) stellar mass-halo mass relation
        b19_um = behrooziUM(sims[0])
        label = b19_um["label"] + " (z = %.1f)" % sims[0].redshift

        ax.plot(b19_um["haloMass"], b19_um["mstar_mid"], "--", color="#ccc", label=label)
        ax.fill_between(b19_um["haloMass"], b19_um["mstar_low"], b19_um["mstar_high"], color="#ccc", alpha=0.3)
        # ax.plot(b19_um["haloMass"], b19_um["mstar_low"], "--", color="#ccc", alpha=0.8)
        # ax.plot(b19_um["haloMass"], b19_um["mstar_high"], "--", color="#ccc", alpha=0.8)

        # MEGATRON (Katz+26) - Figure 6 (upper panel)
        k26_mhalo = [7.01, 7.29, 7.66, 7.85, 8.09, 8.29, 8.72, 9.28, 9.69, 10.11]  # log msun (median)
        k26_mstar = [2.32, 3.09, 3.62, 4.16, 5.11, 6.35, 7.15, 7.79, 8.27, 8.51]  # log msun
        k26_mstar_up = [3.17, 4.05, 4.57, 5.39, 6.43, 6.85, 7.35, 7.96, 8.42, 8.64]  # upper band (1 sigma scatter)
        k26_mstar_down = [0.0, 0.0, 0.0, 3.57, 4.19, 4.86, 6.84, 7.56, 8.07, 8.40]
        k26_label = "Katz+26 MEGATRON (z=10)"

        ax.plot(k26_mhalo, k26_mstar, "-.", color="#777", alpha=0.8, label=k26_label)
        ax.fill_between(k26_mhalo, k26_mstar_down, k26_mstar_up, color="#777", alpha=0.1)

        # Paquereau+2025 - COSMOS-Web HOD-based, z=0-12 (no halo mass overlap yet)
        # p25 = paquereau25(redshift=5.5, mstar="Mth")

        # ax.plot(p25["mhalo"], p25["mstar"], ":", color="#000", alpha=0.8, label=p25["label"])

        # THESAN-ZOOM (Kannan+25) - Figure 3 (z=5, is closer to z=5.5 in z=7 in tage) (individual target galaxies)
        k25_mhalo = [7.49, 7.46, 7.45, 7.98, 8.09, 8.13, 8.57, 8.64, 8.99, 9.04, 9.06, 9.28, 9.29, 9.41, 9.32, 9.41,
                     9.71, 9.80, 10.42, 10.30, 10.27, 10.54, 10.71, 10.71, 11.52, 11.63]  # log msun # fmt: skip
        k25_mstar = [3.83, 4.04, 4.85, 4.05, 4.20, 4.42, 5.11, 5.29, 5.82, 5.95, 6.14, 5.97, 6.15, 6.22, 6.42, 6.73,
                     7.10, 7.20, 7.43, 7.68, 7.87, 7.98, 8.27, 8.67, 9.31, 9.54]  # log msun # fmt: skip
        k25_label = "Kannan+25 THESAN-ZOOM (z=5)"

        ax.plot(k25_mhalo, k25_mstar, "s", color="#999", alpha=0.8, label=k25_label)

        # SIRIUS (Lin+26 - Table 4) ("end" at t=1.2 Gyr)
        l26_mhalo = [1.3e9, 6.3e7, 7.9e7, 1.3e8, 1.0e9, 7.9e7, 8.0e8, 5.0e8]
        l26_mstar = [3.2e6, 6.7e4, 1.3e5, 5.6e4, 2.4e6, 9.6e4, 1.8e6, 1.0e6]
        l26_label = "Lin+26 SIRIUS (z=5)"

        ax.plot(np.log10(l26_mhalo), np.log10(l26_mstar), "D", color="#777", alpha=0.8, label=l26_label)

        # SPICE (Bhagwat+24) - Figure 5 (bursty and smooth models)
        b24_mhalo1 = [1.3e8, 2.3e8, 4.8e8, 9.2e8, 1.9e9, 7.3e9, 1.5e10, 3.0e10, 6.0e10, 1.2e11, 2.4e11]  # log msun
        b24_mhalo2 = [2.4e8, 4.7e8, 9.4e8, 1.9e9, 7.4e9, 1.5e10, 5.8e10, 1.2e11, 2.4e11]  # log msun
        b24_ratio1 = [3.2e-3, 3.9e-3, 7.4e-3, 1.2e-2, 1.9e-2, 3.5e-2, 4.4e-2, 7.5e-2, 1.1e-1, 1.5e-1, 4.1e-1]  # ratio
        b24_ratio2 = [1.9e-3, 2.7e-3, 5.0e-3, 8.1e-3, 1.6e-2, 1.9e-2, 4.0e-2, 4.5e-2, 6.6e-2]  # ratio
        b24_mstar1 = np.array(b24_mhalo1) * sims[0].units.f_b * b24_ratio1
        b24_mstar2 = np.array(b24_mhalo2) * sims[0].units.f_b * b24_ratio2
        b24_label = "Bhagwat+24 SPICE (z=5)"

        ax.plot(np.log10(b24_mhalo1), np.log10(b24_mstar1), ":", color="#777", alpha=0.8, label=b24_label)
        ax.plot(np.log10(b24_mhalo2), np.log10(b24_mstar2), ":", color="#777", alpha=0.8)

        # COLIBRE (Chaikan+25, Fig 3, L25/L25m5, z=5)
        c25_mhalo = [3.02e9, 4.94e9, 7.43e9, 1.98e10, 2.75e10, 4.98e10, 1.25e11, 1.96e11, 3.20e11, 5.12e11]  # log msun
        c25_ratio = [4.47e-4, 5.51e-4, 7.31e-4, 1.40e-3, 2.06e-3, 2.85e-3, 6.99e-7, 7.60e-3, 9.19e-3, 1.34e-2]  # ratio
        c25_mstar = np.array(c25_mhalo) * c25_ratio
        c25_label = "Chaikan+25 COLIBRE (z=5)"

        ax.plot(np.log10(c25_mhalo), np.log10(c25_mstar), ls=linestyles[4], color="#777", alpha=0.8, label=c25_label)

        # todo: flares? (cannot find the paper)

        # constant SFE lines i.e. fractions of f_b * M_halo
        eps_star = [1.0, 0.1, 0.01, 0.001]
        opts = {"fontsize": 11, "color": "#444", "alpha": 1.0, "ha": "right", "va": "bottom", "rotation": 24.0}

        for i, eps in enumerate(eps_star):
            yy = np.log10(eps * sims[0].units.f_b * 10.0 ** np.array(xlim))
            label = r"$\epsilon_\star = %s$" % eps
            x_label = xlim[1] - (xlim[1] - xlim[0]) * [0.45, 0.8, 0.93, 0.45][i]
            y_label = np.interp(x_label, xlim, yy) - 0.3 - 0.03 * i

            ax.plot(xlim, yy, ":", color=opts["color"], lw=1, alpha=1.0)
            ax.text(x_label, y_label, label, **opts)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def sfr_vs_mstar(sims: list[simParams], yQuant: str) -> None:
    """Relation between SFR and Mstar including observational data."""
    from temet.load.data import asada26, chemerynska24, curti23, nakajima23

    xQuant = "mstar2_log"
    ylim = [-3.5, 1.5]  # log sfr
    xlim = [4.5, 9.0]  # log mstar

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        sim_parent = sims[0].sP_parent

        # constant sSFR lines
        sSFR = [1e-7, 1e-8, 1e-9, 1e-10]  # yr^-1
        opts = {"fontsize": 11, "color": "#444", "alpha": 1.0, "ha": "left", "va": "bottom", "rotation": 30.0}

        for i, s in enumerate(sSFR):
            yy = np.log10(s * 10.0 ** np.array(xlim))
            label = "sSFR = $10^{%d}$ yr$^{-1}$" % np.log10(s)
            x_label = xlim[0] + (xlim[1] - xlim[0]) * (0.1 + 0.15 * i)
            y_label = np.interp(x_label, xlim, yy) + 0.15
            if i > 0:
                label = "$10^{%d}$ yr$^{-1}$" % np.log10(s)

            ax.plot(xlim, yy, ":", color="#444", lw=1, alpha=1.0)
            ax.text(x_label, y_label, label, **opts)

        # Chemerynska+24 UNCOVER (z=6-8) https://arxiv.org/abs/2407.17110 (Table 1)
        c24 = chemerynska24()

        ax.errorbar(
            c24["mstar"],
            c24["sfr_ha"],
            xerr=[c24["mstar_err1"], c24["mstar_err2"]],
            yerr=[c24["sfr_ha_err1"], c24["sfr_ha_err2"]],
            fmt="o",
            color="#333",
            alpha=0.4,
            label=c24["label"] + r" $H\alpha$",
        )

        ax.errorbar(
            c24["mstar"],
            c24["sfr_uv"],
            xerr=[c24["mstar_err1"], c24["mstar_err2"]],
            yerr=[c24["sfr_uv_err1"], c24["sfr_uv_err2"]],
            fmt="o",
            color="#000",
            alpha=0.4,
            label=c24["label"] + r" UV",
        )

        # Curti+23 JWST JADES (z=3-10)
        c23 = curti23(sims[0])
        label = c23["label"]  # + r" $z\,\sim\,%.0f$" % sim_parent.redshift

        w = np.where(np.abs(c23["redshift"] - sim_parent.redshift < 1.0))  # e.g. z=3.5-4.5 for sim at z=4

        x = c23["mstar"][w]
        y = np.log10(c23["sfr_a"][w])
        xerr = [c23["mstar_err1"][w], c23["mstar_err2"][w]]
        yerr = [
            np.log10(c23["sfr_a"][w] + c23["sfr_a_err1"][w]) - y,
            y - np.log10(c23["sfr_a"][w] + c23["sfr_a_err2"][w]),
        ]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="s", color="#555", alpha=0.4, label=label)

        # Nakajima+23 (z=4-10) JWST CEERS
        n23 = nakajima23()
        label = n23["label"]  # + r" $z\,\sim\,%.0f$" % sim_parent.redshift

        w = np.where(np.abs(n23["redshift"] - sim_parent.redshift < 2.0))  # e.g. z=3-5 for sim at z=4

        xerr = [n23["mstar_err1"][w], n23["mstar_err2"][w]]
        yerr = [n23["sfr_err1"][w], n23["sfr_err2"][w]]
        ax.errorbar(n23["mstar"][w], n23["sfr"][w], xerr=xerr, yerr=yerr, fmt="p", color="#555", alpha=0.3, label=label)

        # Asada+26 (z~6) GLIMPSE (z=5.5-6.5)
        a26 = asada26()

        ax.errorbar(
            a26["mstar"],
            a26["sfr_ha"],
            xerr=a26["mstar_err"],
            yerr=a26["sfr_err"],
            fmt="D",
            color="#555",
            alpha=0.4,
            label=a26["label"],
        )

        # Popesso+23 model at z=3+ (Eqn. 15)
        a0 = 2.71
        a1 = -0.186
        a2 = 10.86
        a3 = -0.0729

        p23_redshifts = [sims[0].redshift]  # [3]
        for i, redshift in enumerate(p23_redshifts):
            t = sim_parent.units.redshiftToAgeFlat(redshift)
            sfr_max = 10.0 ** (a0 + a1 * t)
            M0 = 10.0 ** (a2 + a3 * t)
            sfr = sfr_max / (1 + M0 / 10.0 ** np.array(xlim))

            # label = 'Popesso+23 z=%d-%d' % (np.min(p23_redshifts),np.max(p23_redshifts)) if i == 0 else ''
            label = "Popesso+23 z=%.1f" % redshift if i == 0 else ""
            ax.plot(xlim, np.log10(sfr), "--", color="#555", lw=lw, alpha=0.7, label=label)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        tracks=True,
        legend="simple",
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def sfr_10_100_ratio(sims):
    """Plot ratio of SFR on 10 Myr vs 100 Myr timescales, vs stellar mass."""
    from temet.load.data import atek22

    xQuant = "mstar2_log"
    yQuant = "sfr_10_100_ratio"

    def _draw_data(ax, sims, **kwargs):
        # Navarro-Carrera+26 (Figure 8, right panel, median line) z=4-6
        nc26_mstar = [6.3, 7.7, 8.4]  # log Msun
        nc26_ratio = [0.2, 0.05, 0.0]  # log SFR_Ha / SFR_UV

        ax.plot(nc26_mstar, nc26_ratio, "--", color="#333", lw=lw, alpha=0.8, label="Navarro-Carrera+26")

        # Rinaldi+23 MIDIS (z=7-8) - Figure 9
        r23_x = [7.47, 7.47, 7.43, 7.71, 7.99, 8.07, 8.09, 8.54, 8.52, 8.71, 8.89, 9.01]  # log Msun
        r23_y = [-0.50, 0.31, 0.53, 1.43, 0.65, -0.20, -0.48, -0.95, -0.61, -0.61, -0.45, -0.92]  # log SFR ratio

        r23_x_left = [7.10, 7.42, 7.17, 7.66, 7.65, 7.39, 7.92, 8.12, 8.48, 8.48, 8.82, 8.96]
        r23_x_right = [7.63, 7.52, 7.62, 7.75, 8.15, 8.49, 8.38, 8.61, 8.57, 8.83, 8.95, 9.05]
        r23_y_up = [-0.17, 0.44, 0.94, 2.07, 1.55, -0.10, -0.36, -0.78, -0.54, -0.51, -0.36, -0.81]
        r23_y_down = [-0.82, 0.17, 0.12, -0.34, -0.25, -0.29, -0.61, -1.13, -0.77, -0.72, -0.57, -1.02]

        ax.errorbar(
            r23_x,
            r23_y,
            xerr=[np.array(r23_x) - r23_x_left, r23_x_right - np.array(r23_x)],
            yerr=[np.array(r23_y) - r23_y_down, r23_y_up - np.array(r23_y)],
            fmt="o",
            color="#000",
            alpha=0.4,
            label="Rinaldi+23 MIDIS",
        )

        # Atek+22 3D-HST (z~1)
        a22 = atek22()
        opts = {"marker": "s", "lw": 0, "ms": 4, "color": "#555", "alpha": 0.3}

        ax.plot(a22["mstar"], a22["sfr_Ha_UV_ratio"], label=a22["label"], **opts)

        # todo: Pirie+ 25 JELS (z~6), ~30 galaxies at 7.5 < M* < 9.5
        # https://ui.adsabs.harvard.edu/abs/2025MNRAS.541.1348P/abstract (Figure 12)

        # todo: Saldana-Lopez+25 (z~6) a few galaxies at M* ~ 1e8
        # https://ui.adsabs.harvard.edu/abs/2025MNRAS.544..132S/abstract (Table A1)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=[4.5, 9.0],
        ylim=[-1.2, 2.0],
        sizefac=0.8,
        parents=False,
        legend="simple",
        legend_locs=["upper left", "upper right"],
        f_selection=_zoomSubhaloIDsToPlot,
        f_post=_draw_data,
    )


def mbh_vs_mhalo(sims: list[simParams]) -> None:
    """SMBH mass versus halo mass."""
    from temet.load.data import zhang23

    xQuant = "mhalo_200_log"
    yQuant = "mass_smbh"  # largest BH_Mass in each subhalo
    # yQuant = 'BH_mass' # sum of all BH_Mass in each subhalo
    xlim = [8.0, 11.25]  # mhalo
    ylim = [2.8, 7.0]  # msmbh, MCST seeds at 1e3, TNG seeds at ~1e6

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        sim_parent = sims[0].sP_parent

        # constant mbh/mhalo ratios
        mbh_mhalo_ratios = [1e-3, 1e-4, 1e-5]
        opts = {"fontsize": 12, "color": "#444", "alpha": 0.3, "ha": "left", "va": "bottom", "rotation": 30.0}

        for _i, ratio in enumerate(mbh_mhalo_ratios):
            yy = np.log10(10.0 ** np.array(xlim) * ratio)
            label = r"$M_{\rm BH} = 10^{%d} M_{\rm halo}$" % np.log10(ratio)
            ax.plot(xlim, yy, ":", color=opts["color"], alpha=opts["alpha"])

            x_label = xlim[0] + (xlim[1] - xlim[0]) * 0.3  # (0.1 + 0.15 * i)
            y_label = np.interp(x_label, xlim, yy) + 0.1
            ax.text(x_label, y_label, label, **opts)

        # Zhang+23 TRINITY semi-empirical model
        z21 = zhang23(sim_parent)

        ax.plot(z21["mhalo"], z21["mbh"], "--", color="#444", alpha=0.8, label=z21["label"])
        ax.fill_between(z21["mhalo"], z21["mbh_p16"], z21["mbh_p84"], color="#444", alpha=0.4)

        # MCST seed mass from parameter file
        SeedBlackHoleMass = 6.774e-08  # 1000 Msun
        MinFoFMassForNewSeed_MCST = 6.774e-3  # 1e8 Msun
        MinFoFMassForNewSeed_TNG = 5.0  # ~5e10 Msun
        mbh_seed = sim_parent.units.codeMassToLogMsun(SeedBlackHoleMass)
        mhalo_seed = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_MCST)

        ax.plot([xlim[0], (xlim[1] + xlim[0]) / 2], [mbh_seed, mbh_seed], ":", color="#444", alpha=0.8)
        label = r"$M_{\rm BH,seed}$ (@ M$_{\rm FoF} = 10^{%.1f}$ M$_{\rm sun}$)" % mhalo_seed
        ax.text(xlim[0] + 0.8, mbh_seed + 0.1, label, fontsize=13, color="#444", alpha=0.5, ha="left", va="bottom")

        mhalo_seed_tng = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_TNG)
        ax.plot([mhalo_seed_tng, mhalo_seed_tng], [ylim[1], ylim[1] - 0.1], "-", color="#444", alpha=0.4)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        legend_locs=["lower right", "upper left"],
        legend_ncols=[1, 1],
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
        sizefac=0.8,
    )


def mbh_vs_mstar(sims: list[simParams]) -> None:
    """SMBH mass versus stellar mass."""
    from temet.load.data import zhang23

    xQuant = "mstar2_log"
    yQuant = "mass_smbh"  # largest BH_Mass in each subhalo
    xlim = [4.5, 10.0]  # mstar
    ylim = [2.8, 7.0]  # msmbh

    def _draw_data(ax, sims):
        xlim = ax.get_xlim()
        # ylim = ax.get_ylim()
        sim_parent = sims[0].sP_parent

        # MCST seed mass from parameter file
        SeedBlackHoleMass = 6.774e-08  # 1000 Msun
        MinFoFMassForNewSeed_MCST = 6.774e-3  # 1e8 Msun
        mbh_seed = sim_parent.units.codeMassToLogMsun(SeedBlackHoleMass)
        mhalo_seed = sim_parent.units.codeMassToLogMsun(MinFoFMassForNewSeed_MCST)

        ax.plot([xlim[0], (xlim[1] + xlim[0]) / 2], [mbh_seed, mbh_seed], ":", color="#444", alpha=0.8)
        label = r"$M_{\rm BH,seed}$ (@ M$_{\rm FoF} = 10^{%.1f}$ M$_{\rm sun}$)" % mhalo_seed
        ax.text(xlim[0] + 0.8, mbh_seed + 0.1, label, fontsize=13, color="#444", alpha=0.5, ha="left", va="bottom")

        # constant mbh/mstar ratios
        mbh_mstar_ratios = [0.1, 0.01, 0.001]
        opts = {"fontsize": 12, "color": "#444", "alpha": 0.3, "ha": "left", "va": "bottom", "rotation": 45.0}

        for i, ratio in enumerate(mbh_mstar_ratios):
            yy = np.log10(10.0 ** np.array(xlim) * ratio)
            label = r"$M_{\rm BH} = M_{\star}$ / %d" % (1 / ratio)
            ax.plot(xlim, yy, ":", color=opts["color"], alpha=opts["alpha"])

            x_label = xlim[0] + (xlim[1] - xlim[0]) * (0.45 - 0.1 * i)
            y_label = np.interp(x_label, xlim, yy) + 0.15
            ax.text(x_label, y_label, label, **opts)

        # Zhang+23 TRINITY semi-empirical model
        z21 = zhang23(sim_parent, mstar=True)

        ax.plot(z21["mstar"], z21["mbh"], "--", color="#444", alpha=0.7, label=z21["label"])
        ax.fill_between(z21["mstar"], z21["mbh_p16"], z21["mbh_p84"], color="#444", alpha=0.4)

        # Pacucci+23
        p23_alpha = -2.43
        p23_beta = 1.06
        # p23_sigma = 0.69

        p23_yy = p23_alpha + p23_beta * np.array(xlim)
        ls = (0, (3, 1, 1, 1, 1, 1))
        ax.plot(xlim, p23_yy, linestyle=ls, color="#444", alpha=0.6, label="Pacucci+23")
        # ax.fill_between(xlim, p23_yy - p23_sigma, p23_yy + p23_sigma, color="#444", alpha=0.05)

        # Ziparo+26 - Eqn 9 (https://arxiv.org/abs/2603.04358)
        z26_alpha = -4.06
        z26_beta = 1.17
        z26_sigma = 0.63  # intrinsic orthogonal scatter

        z26_yy = z26_alpha + z26_beta * np.array(xlim)
        ax.plot(xlim, z26_yy, "-.", color="#444", alpha=0.8, label="Ziparo+26")
        ax.fill_between(xlim, z26_yy - z26_sigma, z26_yy + z26_sigma, color="#444", alpha=0.2)

        # Brooks+25 (z=5.6 and z=5.8 stack points) (Table 1 / Fig 6)
        b25_label = "Brooks+25"  # JWST (z = 5.5-6)'
        b25_mstar = [7.88, 8.56]  # log msun
        b25_mstar_err = [0.18, 0.13]  # dex (note: 0.03 changed to 0.13)
        b25_mbh = [6.13, 5.21]
        b25_mbh_err = [0.53, 0.43]

        ax.errorbar(
            b25_mstar, b25_mbh, xerr=b25_mstar_err, yerr=b25_mbh_err, fmt="o", color="#555", alpha=0.8, label=b25_label
        )

        # Brooks+25 upper limit at z=5.3
        ax.errorbar([7.26], [4.99], xerr=[0.26], fmt="o", color="#555", alpha=0.8)
        ax.annotate(
            "",
            xy=(7.26, 4.99 - 0.4),
            xytext=(7.26, 4.99),
            arrowprops={"facecolor": "#555", "edgecolor": "#555", "arrowstyle": "simple", "alpha": 0.8},
        )

        # Geris+25 (5<z<7) points (Table 4)
        g25_label = "Geris+25"
        g25_mbh = [6.35, 6.30]
        g25_mbh_err = [0.37, 0.37]
        g25_mstar = [8.9, 8.01]
        g25_mstar_err = [0.83, 0.71]

        ax.errorbar(
            g25_mstar, g25_mbh, xerr=g25_mstar_err, yerr=g25_mbh_err, fmt="s", color="#555", alpha=0.8, label=g25_label
        )

        # todo: could add direct/uncorrected JWST samples e.g. Larson+23, Ubler+23, Maiolino+23, Harikane+23, etc
        # (see Brooks+25 Fig 6 / Ortame+26 Fig 6)

    def _add_kjaco_models(ax, sims, **kwargs):
        # add post-processing model results from Kiara
        # /bondi_evolution_live/446076_15_15_1018178170_seed1e+03/Mbh [snap_index]
        # filepath = "cache/MCST_BH_models_bondi_live.h5"  # temporary (/vera/u/kjaco/thesis_files/data_files/)
        filepath = "cache/MCST_BH_models_freefall_live.h5"  # temporary
        modelname = "freefall_live_hsml"  # "bondi_evolution_live"
        paramsets = ["hsml_seed1e+03_eff0.01", "hsml_seed1e+03_eff0.1"]  # , "hsml_seed1e+03_eff1"]

        with h5py.File(filepath, "r") as f:
            # locate dataset based on simulation
            for i, sim in enumerate(sims):
                # decide on BH ID
                subIDs = _zoomSubhaloIDsToPlot(sim)
                subID = subIDs[0]

                mstar = sim.subhalos(xQuant)[subID]
                bhIDs = sim.snapshotSubset("bh", "ids", subhaloID=subID)
                if isinstance(bhIDs, dict) and bhIDs["count"] == 0:
                    print(f" Warning: no BHs found in subhalo {subIDs[0]} of {sim}, skipping.")
                    continue

                bhID = bhIDs[0]
                if len(bhIDs) > 1:
                    print(f"Warning: multiple BHs found in subhalo {subIDs[0]} of {sim}, using first.")

                # load
                for j, paramset in enumerate(paramsets):
                    gname = f"{modelname}/{sim.simName}/{bhID}_{paramset}"
                    snap_index = -1  # last

                    if gname not in f:
                        print(f" Warning: dataset {gname} not found, skipping.")
                        continue

                    Mbh = f[gname + "/Mbh_ff"][()][snap_index]  # "Mbh"
                    Mbh = np.log10(Mbh)  # log Msun

                    print(f" Plotting: [{gname}], with M* = {mstar:.2f} and final Mbh = {Mbh:.2f} [log Msun].")

                    # sanity check (final value should be at z=5.5)
                    gname = f"{modelname}/{sim.simName}/{bhID}_snap"
                    redshift = f[gname + "/redshift_range"][()]

                    assert np.allclose(sim.redshift, redshift[snap_index])

                    # plot
                    ms_loc = (sim.res - 10) * 2.5 + 1  # similar scaling as in scatter2d()
                    ax.plot(mstar, Mbh, color=colors[i], marker=markers[j + 1], ms=ms_loc, mew=2, mfc="None")

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        legend_locs=["lower right", "upper left"],
        legend_ncols=[1, 1],
        f_pre=_draw_data,
        f_post=_add_kjaco_models,
        f_selection=_zoomSubhaloIDsToPlot,
        sizefac=0.8,
    )


def sizes_vs_mstar(sims):
    """Diagnostic plot of galaxy stellar size (half mass radius for now) versus stellar mass."""
    xQuant = "mstar2_log"
    yQuant = "rhalf_stars"
    ylim = [-2.7, 1.5]  # log pkpc
    xlim = [4.5, 9.0]  # log mstar

    def _draw_data(ax, sims):
        # Thesan-Zoom (McClymont+26 Figure 2 https://arxiv.org/abs/2503.04894)
        m26_mstar = [6.0, 7.0, 8.0, 9.0, 9.5]  # log msun
        m26_rhalf = [-0.18, 0.01, 0.20, 0.41, 0.53]  # log kpc

        ax.plot(m26_mstar, m26_rhalf, ":", color="#555", alpha=0.9, label="McClymont+26 (THESAN-ZOOM)")

        # Lin+26 SIRIUS (z ~ 5) https://arxiv.org/abs/2602.22206 (Figure 8)
        l26_label = "Lin+26 (SIRIUS)"
        l26_mstar = [5.6e4, 6.6e4, 9.5e4, 1.3e5, 1.0e6, 2.4e6, 1.8e6, 3.2e6]  # msun
        l26_rh2d = np.array([7.7e1, 5.5e1, 7.4e1, 8.5e1, 5.7e1, 1.1e2, 2.7e1, 2.9e1]) / 1000  # pc -> kpc

        # ELVES (z=0 local volume, not directly related)
        label = "Carlsten+21 ELVES (z=0)"
        xx = np.log10([5e5, 5e8])
        yy_mid = np.log10([2.26e-1, 1.73e0])
        yy_low = np.log10([1.37e-1, 1.05e0])
        yy_high = np.log10([3.72e-1, 2.84e0])

        ax.plot(xx, yy_mid, "--", color="#999", alpha=0.8, label=label)
        ax.fill_between(xx, yy_low, yy_high, color="#999", alpha=0.2)

        # Mowla+2019 HST (extrapolation below M* = 9.5)
        m19_label = r"Mowla+19 HST ($z \sim 3$)"
        xx = np.array([4.5, 9.5, 11.5])
        A = 10.0**0.51  # Table 2, z=2.75, star-forming
        A_high = 10.0 ** (0.51 + 0.09)
        A_low = 10.0 ** (0.51 - 0.09)
        alpha = 0.14
        reff = np.log10(A * (10.0**xx / 7e10) ** alpha)  # log pkpc

        # data constrained vs extrapolation
        ax.plot(xx[1:], reff[1:], ":", color="#999", alpha=1.0, label=m19_label)
        ax.plot(xx[:-1], reff[:-1], ":", color="#999", alpha=0.8)

        reff_low = np.log10(A_low * (10.0**xx / 7e10) ** alpha)  # log pkpc
        reff_high = np.log10(A_high * (10.0**xx / 7e10) ** alpha)  # log pkpc

        ax.fill_between(xx, reff_low, reff_high, color="#999", alpha=0.2)

        # Allen+25 (https://arxiv.org/abs/2410.16354v1 Table A1) (rest-optical uses F356W for the 5<z<6 redshift bin)
        a24_label = r"Allen+25 JWST ($z \sim 5$)"
        xx = np.array([4.5, 8.0, 9.0, 11.0])
        A = 10.0 ** (-0.070)
        sigma_Reff = 0.193  # intrinsic scatter
        M0 = 1e9
        alpha = 0.231
        reff = np.log10(A * (10.0**xx / M0) ** alpha)  # log pkpc
        log_reff = alpha * np.log10(10.0**xx / M0) + np.log10(A)
        print(xx)
        print(reff)
        print(10.0**log_reff)

        ax.plot(xx, reff, "--", color="#555", alpha=0.9, label=a24_label)
        ax.fill_between(xx, reff - sigma_Reff, reff + sigma_Reff, color="#555", alpha=0.2)

        # Matharu+24 FRESCO (z ~ 5.3)
        m24_label = "Matharu+24 FRESCO (z=5)"
        m24_mstar = [8.1, 8.6, 9.1, 9.6]  # log mstar
        m24_reff = [-0.56, -0.38, -0.35, -0.13]  # log kpc (stellar continuum)
        m24_reff_err = 0.1  # dex, assumed

        ax.errorbar(m24_mstar, m24_reff, yerr=m24_reff_err, fmt="D:", color="#555", alpha=0.9, label=m24_label)

        # JELS (https://arxiv.org/abs/2509.08045) (V-band as proxy for bulk of stellar population)
        # fmt: off
        s25_label = "Stephenson+25 JELS (z=6)"
        s25_mstar = [8.23, 8.21, 8.07, 8.19, 8.22, 8.24, 8.19, 8.23, 8.06, 8.17, 8.21, 8.29, 8.52, 8.61, 8.59, 8.57,
                     8.67, 9.08, 9.15, 9.19, 9.02, 9.06, 9.28]
        s25_mstar_left = [7.91, 7.97, 7.89, 8.02, 8.09, 8.11, 7.93, 8.03, 7.91, 8.02, 8.03, 8.12, 8.35, 8.44, 8.41,
                          8.35, 8.55, 8.91, 9.02, 9.01, 8.92, 8.88, 9.19]
        s25_mstar_right = [8.55, 8.44, 8.25, 8.37, 8.36, 8.37, 8.44, 8.43, 8.21, 8.32, 8.39, 8.47, 8.69, 8.79, 8.77,
                           8.79, 8.79, 9.26, 9.29, 9.36, 9.13, 9.25, 9.37]

        s25_re_V = [-1.45, -1.11, -0.73, -0.70, -0.52, -0.34, -0.29, -0.27, -0.20, -0.08, 0.03, -0.15, -0.38, -0.48,
                    -0.22, -0.08, 0.13, -0.04, 0.03, -0.04, -0.36, -0.32, -0.35]
        s25_re_V_up  = [-0.76, -0.91, -0.64, -0.60, -0.42, -0.24, -0.20, -0.18, -0.10, 0.02, 0.12, -0.05, -0.27,
                        -0.38, -0.12, 0.02, 0.23, 0.05, 0.12, 0.06, -0.26, -0.22, -0.25]
        s25_re_V_down = [-1.60, -1.30, -0.83, -0.79, -0.62, -0.43, -0.39, -0.37, -0.30, -0.18, -0.07, -0.24, -0.47,
                         -0.58, -0.33, -0.18, 0.04, -0.14, -0.07, -0.13, -0.46, -0.42, -0.44]
        # fmt: on

        s25_xerr = [np.array(s25_mstar) - np.array(s25_mstar_left), np.array(s25_mstar_right) - np.array(s25_mstar)]
        s25_yerr = [np.array(s25_re_V) - np.array(s25_re_V_down), np.array(s25_re_V_up) - np.array(s25_re_V)]

        ax.errorbar(
            s25_mstar, s25_re_V, xerr=s25_xerr, yerr=s25_yerr, fmt="s", color="#777", alpha=0.7, label=s25_label
        )

        # opts = {"marker": "s", "lw": 0, "ms": 4, "color": "#555", "alpha": 0.3}
        ax.plot(np.log10(l26_mstar), np.log10(l26_rh2d), lw=0, marker="X", color="#555", alpha=0.6, label=l26_label)

        # JWST PSF (Pirie+25 for JELS)
        psf_arcsec = 0.05  # best case e.g. F090W
        label = "JWST PSF (0.05'' @ z=5.5)"

        xx = ax.get_xlim()
        yy = np.log10(sims[0].units.arcsecToAngSizeKpcAtRedshift(psf_arcsec))

        ax.plot(xx, [yy, yy], ":", color="#444", lw=1, alpha=1.0)
        ax.text(xx[1] - 1.8, yy - 0.05, label, fontsize=11, color="#999", alpha=1.0, ha="left", va="top")

        # Nakane+25 VENUS (https://arxiv.org/abs/2511.14483) (Table 2)
        n25_label = "Nakane+25 VENUS (z=11)"
        n25_mstar = [7.57, 7.03, 7.09, 6.87, 6.92, 7.53, 7.02, 7.11, 6.98]  # log msun
        n25_mstar_err = [0.26, 0.19, 0.41, 0.24, 0.36, 0.22, 0.14, 0.35, 0.24]  # dex
        n25_reff = (
            np.array([181, 15, 22, 67, 36, 197, 17, 36, 56]) / 1000
        )  # kpc, "intrinsic" (corrected for magnification)
        n25_reff_err = np.array([18, 3, 5, 12, 8, 28, 9, 33, 21]) / 1000  # kpc

        n25_yerr1 = np.log10(n25_reff) - np.log10(np.array(n25_reff) - np.array(n25_reff_err))
        n25_yerr2 = np.log10(np.array(n25_reff) + np.array(n25_reff_err)) - np.log10(n25_reff)

        ax.errorbar(
            n25_mstar,
            np.log10(n25_reff),
            xerr=n25_mstar_err,
            yerr=[n25_yerr1, n25_yerr2],
            fmt="o",
            color="#333",
            alpha=0.7,
            label=n25_label,
        )

        # TODO: Miller+25
        # https://ui.adsabs.harvard.edu/abs/2025ApJ...988..196M/abstract

        # TODO: Ceverino+26 (FirstLight simulations)
        # https://arxiv.org/abs/2603.05045

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        legend_ncols=[2, 2],
        legend_locs=["upper left", "lower right"],
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def size_halpha_vs_mstar(sims):
    """Diagnostic plot of galaxy h-alpha (gas) size (half-light radius) versus stellar mass."""
    for sim in sims:
        sim.createCloudyCache = False

    xQuant = "mstar2_log"
    yQuant = "size_halpha_em"  # cloudy-based
    ylim = [-1.8, 1.8]  # log pkpc
    xlim = [4.5, 9.0]  # log mstar

    def _draw_data(ax, sims):
        # Matharu+24 FRESCO (z ~ 5.3)
        m24_label = "Matharu+24 FRESCO (z=5)"
        m24_mstar = [8.1, 8.6, 9.1, 9.6]  # log mstar
        m24_reff = [-0.11, -0.14, -0.13, -0.024]  # log kpc (Halpha)
        m24_reff_err = 0.1  # dex, assumed

        ax.errorbar(m24_mstar, m24_reff, yerr=m24_reff_err, fmt="D--", color="#555", alpha=0.9, label=m24_label)

        # JELS (https://arxiv.org/abs/2509.08045)
        # fmt: off
        s25_label = "Stephenson+25 JELS (z=6)"
        s25_mstar = [8.23, 8.19, 8.22, 8.21, 8.24, 8.29, 8.52, 8.59, 8.57, 8.61, 8.67, 8.06, 8.08, 8.18, 8.17, 8.23,
                     8.21, 9.02, 9.06, 9.19, 9.28, 9.15, 9.08]
        s25_mstar_left = [7.91, 8.02, 8.09, 7.97, 8.11, 8.12, 8.35, 8.40, 8.35, 8.44, 8.55, 7.91, 7.89, 7.93, 8.02,
                          8.03, 8.03, 8.91, 8.88, 9.01, 9.18, 9.02, 8.91]
        s25_mstar_right = [8.55, 8.37, 8.36, 8.44, 8.37, 8.47, 8.69, 8.77, 8.79, 8.79, 8.79, 8.21, 8.25, 8.44, 8.32,
                           8.43, 8.39, 9.13, 9.25, 9.37, 9.37, 9.29, 9.26]

        s25_re_nb = [-0.75, -0.71, -0.52, -0.49, -0.40, -0.37, -0.36, -0.39, -0.27, -0.20, 0.07, -0.47, -0.37, -0.23,
                     -0.01, -0.03, 0.12, -0.15, -0.41, -0.35, -0.14, -0.07, -0.07]
        s25_re_nb_up = [-0.66, -0.61, -0.42, -0.40, -0.31, -0.28, -0.27, -0.29, -0.18, -0.10, 0.17, -0.36, -0.27,
                        -0.13, 0.09, 0.07, 0.22, -0.05, -0.31, -0.25, -0.05, 0.03, 0.02]
        s25_re_nb_down = [-0.88, -0.84, -0.64, -0.61, -0.53, -0.50, -0.49, -0.51, -0.40, -0.32, -0.06, -0.62, -0.49,
                          -0.36, -0.13, -0.16, 0.00, -0.28, -0.53, -0.48, -0.27, -0.19, -0.19]
        # fmt: on

        s25_xerr = [np.array(s25_mstar) - np.array(s25_mstar_left), np.array(s25_mstar_right) - np.array(s25_mstar)]
        s25_yerr = [np.array(s25_re_nb) - np.array(s25_re_nb_down), np.array(s25_re_nb_up) - np.array(s25_re_nb)]

        ax.errorbar(
            s25_mstar, s25_re_nb, xerr=s25_xerr, yerr=s25_yerr, fmt="s", color="#555", alpha=0.9, label=s25_label
        )

        # Thesan-Zoom (McClymont+26 Figure 2 https://arxiv.org/abs/2503.04894)
        m26_mstar = [6.0, 7.0, 8.0, 8.5, 9.5]  # log msun
        m26_rhalf = [0.07, 0.23, 0.39, 0.47, 0.74]  # log kpc (Halpha)

        ax.plot(m26_mstar, m26_rhalf, "-", color="#555", alpha=0.7, label="McClymont+26 (THESAN-ZOOM)")

        # TODO: Danhaive+25
        # https://ui.adsabs.harvard.edu/abs/2025arXiv251006315D/abstract

    def _draw_psf(ax, sims, **kwargs):
        # JWST PSF (Pirie+25 for JELS)
        psf_arcsec = 0.17  # for the NB filters used for Halpha
        label = "JWST PSF (0.17'' @ z=5.5)"

        xx = ax.get_xlim()
        yy = np.log10(sims[0].units.arcsecToAngSizeKpcAtRedshift(psf_arcsec))

        ax.plot(xx, [yy, yy], ":", color="#444", lw=1, alpha=1.0)
        ax.text(xx[1] - 2.5, yy - 0.05, label, fontsize=11, color="#999", alpha=1.0, ha="left", va="top")

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        legend_ncols=[1, 3],
        legend_locs=["lower right", "upper left"],
        f_pre=_draw_data,
        f_post=_draw_psf,
        f_selection=_zoomSubhaloIDsToPlot,
        sizefac=0.8,
    )


def gas_mzr(sims):
    """Diagnostic plot of gas-phase mass-metallicity relation (MZR)."""
    from temet.load.data import asada26, chemerynska24, curti23, logOHp12_solar, nakajima23

    xQuant = "mstar2_log"
    yQuant = "Z_gas_sfrwt"
    ylim = [-2.6, 0.0]  # log pkpc
    xlim = [4.0, 9.0]  # log mstar

    def _draw_data(ax, sims):
        # change main y-axis label, and create second y-axis with 12 + log(O/H) units
        ax.set_ylabel(r"Gas Metallicity $\rm{Z_{gas}}$ [ $\rm{Z_\odot}$ ]")

        yy2 = np.array(ax.get_ylim()) + logOHp12_solar  # 12 + log(O/H)

        ax_y2 = ax.twinx()
        ax_y2.set_ylim(yy2)
        ax_y2.set_ylabel(r"12 + $\log(\rm{O/H})$")

        # Stanton+ (2024) - NIRVANDELS z=3.5 (SB99)
        s24_mstar = [8.5, 9.5, 10.5]  # log mstar
        s24_z_b18 = [-0.72, -0.39, -0.06]  # log Z/Zsun (B18 calibration)
        s24_z_b18_low = [-0.81, -0.42, -0.16]
        s24_z_b18_high = [-0.62, -0.36, 0.03]
        s24_z_c17 = [-0.58, -0.39, -0.21]  # alternative C17 calibration
        s24_z_s24 = [-0.59, -0.30, -0.01]  # alternative S24 calibration

        # adjust from A09 (Stanton+24) to our Zsun
        solar_asplund09 = 0.0142
        fac = solar_asplund09 / sims[0].units.Z_solar

        s24_z_b18 = np.log10(10.0 ** np.array(s24_z_b18) * fac)
        s24_z_b18_low = np.log10(10.0 ** np.array(s24_z_b18_low) * fac)
        s24_z_b18_high = np.log10(10.0 ** np.array(s24_z_b18_high) * fac)
        s24_z_c17 = np.log10(10.0 ** np.array(s24_z_c17) * fac)
        s24_z_s24 = np.log10(10.0 ** np.array(s24_z_s24) * fac)

        ax.plot(s24_mstar, s24_z_b18, "-", color="#888", alpha=0.8, label="Stanton+24 z~4 VANDELS")
        ax.fill_between(s24_mstar, s24_z_b18_low, s24_z_b18_high, color="#555", alpha=0.2)
        # ax.plot(s24_mstar, s24_z_c17, "-", color="#999", alpha=1.0, label="Stanton+24 (C17)")
        # ax.plot(s24_mstar, s24_z_s24, ":", color="#999", alpha=1.0, label="Stanton+24 (S24)")

        # Li+23 z=3 (B18 calibration)
        li23_mstar = [8.1, 9.0, 10.0]  # log mstar
        li23_z = [-0.59, -0.45, -0.29]  # log Z/Zsun
        li23_z = np.log10(10.0 ** np.array(li23_z) * fac)

        li23_mstar = [6.5, 7.5, 8.4, 9.9]
        li23_12pOH = np.array([7.85, 8.00, 8.14, 8.38])
        li23_12pOH_up = np.array([7.97, 8.07, 8.19, 8.48])
        li23_12pOH_down = np.array([7.72, 7.92, 8.10, 8.29])

        li23_z = li23_12pOH - logOHp12_solar
        li23_z_up = li23_12pOH_up - logOHp12_solar
        li23_z_down = li23_12pOH_down - logOHp12_solar

        ax.fill_between(li23_mstar, li23_z_down, li23_z_up, color="#888", alpha=0.5)
        ax.plot(li23_mstar, li23_z, "-.", color="#666", alpha=1.0, label="Li+23 z~3 NIRISS")

        # Kotiwale+25 (https://arxiv.org/abs/2510.19959) (Table 2)
        k25_label = "Kotiwale+25 z~6 EIGER/ALT"
        k25_mstar = [7.36, 7.88, 8.13, 8.35, 8.64, 8.84, 9.21, 9.71]
        k25_mstar_e1 = [0.29, 0.09, 0.08, 0.09, 0.08, 0.11, 0.18, 0.26]
        k25_mstar_e2 = [0.26, 0.09, 0.09, 0.06, 0.08, 0.06, 0.19, 0.17]
        k25_12pOH_A = [7.76, 7.75, 7.51, 7.83, 7.72, 7.72, 7.83, 7.66]  # "lower branch"
        k25_12pOH_B = [8.19, 8.21, 8.54, 8.12, 8.25, 8.24, 8.11, 8.32]  # "upper branch"

        k25_Z_A = np.array(k25_12pOH_A) - logOHp12_solar
        k25_Z_B = np.array(k25_12pOH_B) - logOHp12_solar

        xerr = [k25_mstar_e1, k25_mstar_e2]
        ax.errorbar(k25_mstar, k25_Z_A, xerr=xerr, fmt="D", color="#555", alpha=0.7, label=k25_label)
        ax.errorbar(k25_mstar, k25_Z_B, xerr=xerr, fmt="D", color="#555", alpha=0.9)

        # Chemerynska+24 UNCOVER (z=6-8) https://arxiv.org/abs/2407.17110 (Table 1)
        c24 = chemerynska24()

        x = c24["mstar"]
        y = c24["metallicity"] - logOHp12_solar
        x_err = [c24["mstar_err1"], c24["mstar_err2"]]
        y_err1 = (c24["metallicity"] + c24["metallicity_err"] - logOHp12_solar) - y
        y_err2 = y - (c24["metallicity"] - c24["metallicity_err"] - logOHp12_solar)

        ax.errorbar(x, y, xerr=x_err, yerr=[y_err2, y_err1], fmt="X", color="#555", alpha=0.6, label=c24["label"])

        # Arellano-Cordova+26 (https://arxiv.org/abs/2602.13007) (https://arxiv.org/abs/2412.10557)
        a26_label = "Arellano-Cordova+26"
        a26_mstar = [8.02, 8.05, 7.66, 7.70, 8.18, 9.30]
        a26_mstar_err = [0.06, 0.17, 0.27, 0.24, 0.06, 0.13]
        a26_12pOH = [7.99, 7.72, 7.50, 7.66, 7.42, 7.74]
        a26_12pOH_err = [0.13, 0.03, 0.05, 0.07, 0.05, 0.22]

        a26_y = np.array(a26_12pOH) - logOHp12_solar
        ax.errorbar(
            a26_mstar, a26_y, xerr=a26_mstar_err, yerr=a26_12pOH_err, fmt="8", color="#555", alpha=0.9, label=a26_label
        )

        # Nakajima+23 (z=4-10) JWST CEERS
        n23 = nakajima23()
        label = n23["label"]  # + r" $z\,\sim\,%.0f$" % sim_parent.redshift

        w = np.where(np.abs(n23["redshift"] - sims[0].redshift < 2.0))  # e.g. z=3-5 for sim at z=4

        n23_y = n23["metallicity"][w] - logOHp12_solar
        ax.errorbar(n23["mstar"][w], n23_y, fmt="P", color="#555", alpha=0.3, label=label)

        # Asada+26 (z~6) GLIMPSE (z=5.5-6.5)
        a26 = asada26()

        x = a26["mstar"]
        y = a26["metallicity"] - logOHp12_solar
        x_err = a26["mstar_err"]
        y_err1 = (a26["metallicity"] + a26["metallicity_err"] - logOHp12_solar) - y
        y_err2 = y - (a26["metallicity"] - a26["metallicity_err"] - logOHp12_solar)

        ax.errorbar(x, y, xerr=x_err, yerr=[y_err2, y_err1], fmt="h", color="#555", alpha=0.9, label=a26["label"])

        # Curti+23 JADES
        c23 = curti23(sims[0])

        w = np.where(np.abs(c23["redshift"] - sims[0].redshift < 1.0))  # e.g. z=4.5-6.6

        x = c23["mstar"][w]
        y = np.log10(c23["Z"][w])
        xerr = [c23["mstar_err1"][w], c23["mstar_err2"][w]]
        yerr = [np.log10(c23["Z_err1"][w] + c23["Z"][w]) - y, y - np.log10(c23["Z"][w] - c23["Z_err2"][w])]
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="s", color="#555", alpha=0.4, label=c23["label"])

        # Cameron+26 JADES (https://arxiv.org/abs/2601.15964) - no Z vs M* (mention)
        # Stanton+25 (https://arxiv.org/abs/2511.00705) - skip, z=2-8, but only M* > 8 (mention)
        # Lewis+25 (https://arxiv.org/abs/2512.03134) - skip, only M* > 8 (mention)
        # Sanders+25 (https://arxiv.org/abs/2508.10099) - skip (mention)
        # Nishigaki+25 (https://arxiv.org/abs/2512.12983) - skip, mostly M* > 8 (mention)

        # todo: https://arxiv.org/abs/2603.15761 (see Fig 6)
        # todo: https://arxiv.org/abs/2605.06770 (GLIMPSED, eight galaxies at M* < 8.2)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        legend="simple",
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def stellar_mzr(sims):
    """Diagnostic plot of stellar mass-metallicity relation (MZR)."""
    xQuant = "mstar2_log"
    yQuant = "Z_stars"  # Z_stars is cat/tree (<2rhalf), while Z_stars_masswt is aux (subhalo)
    ylim = [-2.6, 0.0]  # log pkpc
    xlim = [4.0, 9.0]  # log mstar

    def _draw_data(ax, sims):
        # adjust from A09 (all curves from Stanton+24) to our Zsun
        solar_asplund09 = 0.0142
        fac = solar_asplund09 / sims[0].units.Z_solar

        # Stanton+ (2024) - NIRVANDELS z=3.5 (SB99)
        s24_mstar = [8.5, 9.5, 10.5]  # log mstar
        s24_z = [-1.12, -0.82, -0.53]  # log Z/Zsun
        s24_z_low = [-1.23, -1.01, -0.86]
        s24_z_high = [-0.79, -0.66, -0.40]
        s24_z_v40 = [-1.19, -0.97, -0.75]  # "v40 models"

        s24_z = np.log10(10.0 ** np.array(s24_z) * fac)
        s24_z_low = np.log10(10.0 ** np.array(s24_z_low) * fac)
        s24_z_high = np.log10(10.0 ** np.array(s24_z_high) * fac)
        s24_z_v40 = np.log10(10.0 ** np.array(s24_z_v40) * fac)

        ax.plot(s24_mstar, s24_z, "-", color="#999", alpha=1.0, label="Stanton+24 ($z=3.5$)")
        ax.fill_between(s24_mstar, s24_z_low, s24_z_high, color="#555", alpha=0.2)
        # ax.plot(s24_mstar, s24_z_v40, "-", color="#999", alpha=1.0, label="Stanton+24 v40")

        # Cullen+ (2019)  2.5 < z < 5.0 (SB99)
        # c19_mstar = [8.5, 9.5, 10.2]  # log mstar
        # c19_z = [-1.08, -0.82, -0.63]  # log Z/Zsun
        # c19_z = np.log10(10.0 ** np.array(c19_z) * fac)

        # ax.plot(c19_mstar, c19_z, "--", color="#999", alpha=1.0, label="Cullen+19 ($2.5<z<5$)")

        # Chartab+ (2023) z=2.5, Kashino+ (2022) z=2 (BPASS)
        # k22_mstar = [8.9, 9.5, 10.0, 10.5]  # log mstar
        # k22_z = [-1.16, -0.97, -0.81, -0.65]  # log Z/Zsun
        # k22_z = np.log10(10.0 ** np.array(k22_z) * fac)

        # ax.plot(k22_mstar, k22_z, ":", color="#999", alpha=1.0, label="Kashino+22 z=2-3")

        # Calabro+ (2021), z=2-5 (UV Index)
        c21_mstar = [8.5, 9.5, 10.5]  # log mstar
        c21_z = [-1.23, -0.83, -0.45]  # log Z/Zsun
        c21_z = np.log10(10.0 ** np.array(c21_z) * fac)

        ax.plot(c21_mstar, c21_z, "-.", color="#999", alpha=1.0, label="Calabro+21 ($z=2-5$)")

        # THESAN-ZOOM z=6 (McClymont+26 Figure 3)
        m26_mstar = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # log mstar
        m26_z = [-1.83, -1.50, -1.19, -0.90, -0.63, -0.39]  # log Z/Zsun
        m26_label = "McClymont+26 (THESAN-ZOOM)"

        m26_z_solar = 0.02  # Sec 2.3 last paragraph
        fac = m26_z_solar / sims[0].units.Z_solar
        m26_z = np.log10(10.0 ** np.array(m26_z) * fac)

        ax.plot(m26_mstar, m26_z, "--", color="#555", alpha=0.9, label=m26_label)

        # FIRE z=6 (Ma+16)
        ma16_mstar = [6.0, 7.0, 8.0, 9.0, 10.0, 11.0]  # log mstar
        ma16_z = [-2.09, -1.76, -1.44, -1.12, -0.80, -0.49]  # log Z/Zsun
        ma16_label = "Ma+16 (FIRE)"

        ma16_z = np.log10(10.0 ** np.array(ma16_z) * fac)  # taken from McClymont+26, assume unified Z_solar

        ax.plot(ma16_mstar, ma16_z, "-.", color="#555", alpha=0.9, label=ma16_label)

        # note: Rey+25 gives stellar MZR (in <Fe/H>) at z~10 for Megatron
        # roughly on top of local z~0 dwarfs, could do a similar comparison

        # Nakane+25 (https://arxiv.org/abs/2503.11457) (uses Z_sun from Asplund+09)
        n25_label = "Nakane+25 (z=9-12)"
        # n25_names = ["GHZ2", "GS-z11-0", "GN-z11", "MACS0647-JD", "JADES6438", "GS-z9-0", "Gz9p3"]
        # n25_z = [12.3, 11.1, 10.6, 10.2, 9.7, 9.4, 9.3]  # redshift
        n25_mstar = [9.05, 8.3, 9.1, 7.6, 8.35, 8.2, 9.2]  # note: JADES Mstar assumed same as GS-Z11-0 (same M_UV)
        n25_mstar_err = [0.2, 0.1, 0.3, 0.1, 0.1, 0.1, 0.15]  # roughly, symmetrized
        n25_Z_stars = [-0.87, -0.09, -0.60, -1.87, -0.1, -1.23, -0.13]
        n25_Z_stars_err_up = [0.23, 0.24, 0.06, 0, 0, 0.09, 0.24]  # note: 0 values are upper lims
        n25_Z_stars_err_down = [1.87, 0.36, 0.12, 0.1, 0.1, 0.59, 2.01]
        n25_upper_limits = [False, False, False, True, True, False, False]

        fac = solar_asplund09 / sims[0].units.Z_solar
        n25_Z_stars = np.log10(10.0 ** np.array(n25_Z_stars) * fac)

        # ax.plot(n25_mstar, n25_Z_stars, "o", color="#555", alpha=0.8, label="Nakane+25 (z=9-12)")
        ax.errorbar(
            n25_mstar,
            n25_Z_stars,
            xerr=n25_mstar_err,
            yerr=[n25_Z_stars_err_down, n25_Z_stars_err_up],
            uplims=n25_upper_limits,
            fmt="o",
            color="#555",
            alpha=0.8,
            label=n25_label,
        )

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        sizefac=0.8,
        parents=False,
        legend="simple",
        legend_ncols=[1, 3],
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def phase_diagram(sim, xlim=None, yQuant="temp", cQuant=None, ext="pdf"):
    """Driver."""
    # config
    xQuant = "nh"

    if xlim is None:
        xlim = [-6.0, 7.0]  # xmax == 6 for L15, xmax == 7 for L16

    haloIDs = None  # [0]
    qRestrictions = [["rad_rvir", 0.0, 5.0]]  # within 5rvir only
    qRestrictions.append(["highres_massfrac", 0.5, 1.0])  # high-res only

    if yQuant == "temp":
        ylim = [1.0, 7.5]  # [1.0, 8.0]
    if yQuant == "csnd":
        ylim = [-0.5, 2.5]
    if yQuant == "pres":
        ylim = [2.5, 7.5]  # ymax == 6 for L15, ymax == 8 for L16

    cStr = cQuant
    ctCenter = None

    if cQuant is None:
        clim = [-5.0, 0.0]  # show mass distribution
        cStr = "mass"
        ctName = "viridis"
    if cQuant == "Z_solar":
        clim = [-2.0, 0.0]
        ctName = "plasma"
    if cQuant == "vrad":
        clim = [-60, 120]
        ctName = "blue_red_sharp"  # "blue_red_sharp"
        ctCenter = 0.0
    if cQuant == "vmag":
        clim = [150, 200]
        ctName = "solar"
    if cQuant == "csnd":
        clim = [0, 2.0]
        ctName = "magma"
    if cQuant == "rad_rvir":
        clim = [-2.0, 0.0]
        ctName = "inferno"
    if cQuant == "tff_local":
        clim = [0.0, 4.0]
        ctName = "thermal"

    saveFilename = "phase_%s_%s_%s_%s_%03d.%s" % (sim.simName, xQuant, yQuant, cStr, sim.snap, ext)

    # MCS model: star formation threshold
    def _f_post(ax, **kwargs):
        from temet.util.units import units

        xx = ax.get_xlim()
        dens = 10.0 ** np.array(xx)  # 1/cm^3
        dens *= sim.units.mass_proton  # g/cm^3

        # NOTE: 8.0 is a model parameter!
        for i, M_J in enumerate([1.0, 8.0]):
            M_jeans = M_J * sim.units.codeMassToMsun(sim.targetGasMass)[0]  # Msun
            M_jeans *= sim.units.Msun_in_g  # g

            # [g * (cm**3/g/s**2)**(3/2) * cm**(-3/2) g**1/2] = [cm^(9/2) * cm^(-3/2) / s^3] = [cm/s]^3
            csnd = (M_jeans * 6 * units.Gravity ** (3 / 2) * dens ** (1 / 2) / np.pi ** (5 / 2)) ** (
                1 / 3
            )  # Smith+ Eqn. 1 [cm/s]
            # [cm^2/s^2 g / erg * K] = [cm^2/s^2 g s^2/cm^2/g * K] = [K]
            temp = csnd**2 * units.mass_proton / units.gamma / units.boltzmann
            pres = (dens / sim.units.mass_proton) * temp  # K / cm^3

            if yQuant == "temp":
                yy = temp
            if yQuant == "csnd":
                yy = csnd
            if yQuant == "pres":
                yy = pres

            ax.plot(xx, np.log10(yy), ls=[":", "--"][i], color="black", alpha=0.7)

    snapshot.phaseSpace2d(
        sim,
        xQuant=xQuant,
        yQuant=yQuant,
        meancolors=cQuant,
        ctName=ctName,
        ctCenter=ctCenter,
        haloIDs=haloIDs,
        qRestrictions=qRestrictions,
        qRestrictionsLabel=False,
        nBins=250,
        xlim=xlim,
        ylim=ylim,
        clim=clim,
        sizefac=0.8,
        hideBelow=False,
        f_post=_f_post,
        saveFilename=saveFilename,
    )


def diagnostic_snapshot_spacing(sims):
    """Visualize snapshot time spacing for different setups."""
    fig, ax = plt.subplots()
    ax.set_ylim([0, 5])
    ax.set_xlim([20, 5.5])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    ax.set_xticks([20, 18, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5.5])
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Snapshot Spacing [Myr]")

    ax.plot(ax.get_xlim(), [1, 1], "-", color="#999", alpha=0.5)
    ax.plot(ax.get_xlim(), [2, 2], "-", color="#999", alpha=0.5)

    for sim in sims:
        snaps = sim.validSnapList()
        redshifts = sim.snapNumToRedshift(snaps)
        tage = sim.units.redshiftToAgeFlat(redshifts) * 1000  # Myr
        dt = np.diff(tage)

        ax.plot(redshifts[1:], dt, "o-", ms=6.0, label=f"{sim} saved")

    # load request
    fname1 = "/u/dnelson/sims.structures/arepo7/outputlist_10Myr_z10-3.txt"
    fname2 = "/u/dnelson/sims.structures/arepo7/outputlist_1Myr_z20-5.5.txt"

    for i, fname in enumerate([fname1, fname2]):
        with open(fname) as f:
            times = np.array([float(line.split()[0]) for line in f.readlines()[1:]])
            redshifts = 1 / times - 1
            tage = sims[0].units.redshiftToAgeFlat(redshifts) * 1000  # Myr
            dt = np.diff(tage)
            c = ["#666", "#000"][i]
            label = fname.split("/")[-1].replace("outputlist_", "").replace(".txt", "")
            ax.plot(redshifts[1:], dt, "o-", ms=4.0, color=c, label="%s request" % label)

    ax.legend()

    fig.savefig("snapshot_spacing_%s.pdf" % sims[0].simName)
    plt.close(fig)


def blackhole_properties_vs_time(sim):
    """Plot SMBH mass growth and accretion rates vs time, from the txt files."""
    # load
    smbhs = blackhole_details_mergers(sim)

    xlim = [12.1, 5.5]
    ageVals = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # handle mergers: if this ID ever appears in a merger pair, then
    # decide which of the two IDs to keep i.e. attach the earlier data from
    for smbh_id in smbhs.keys():
        if smbh_id == "mergers":
            continue

        w = np.where(smbhs["mergers"]["ids"] == smbh_id)[0]
        if len(w) > 0:
            print(" NOTE: SMBH ID [{smbh_id}] involved in mergers, TODO.")
            # import pdb; pdb.set_trace() # todo

    # make a multi-panel time series plot for each SMBH
    for smbh_id in smbhs.keys():
        if smbh_id == "mergers":
            continue

        # unit conversions
        print("plot: ", smbh_id)

        time = smbhs[smbh_id]["time"]
        mass_code = smbhs[smbh_id]["mass"]
        mass = sim.units.codeMassToLogMsun(mass_code)  # log msun
        mdot = logZeroNaN(smbhs[smbh_id]["mdot"])  # log msun/yr

        redshift = 1.0 / time - 1

        xlim[0] = np.max(redshift) + 0.2

        # plot
        step = 1

        mdot_edd = np.log10(sim.units.codeBHMassToMdotEdd(mass_code[::step]))
        mdot_limit = np.log10(10.0**mdot_edd * sim.params["BlackHoleEddingtonFactor"])

        # mass
        fig, ax = plt.subplots(nrows=2, figsize=(figsize[0] * 1.2, figsize[1] * 0.8))  # , sharex=True)

        ax[0].set_xlabel("Redshift")
        ax[0].set_xlim(xlim)
        ax[0].set_ylabel(r"SMBH Mass" + "\n" + r"[ log M$_{\rm sun}$ ]")

        ax[0].plot(redshift[::step], mass[::step], zorder=0)
        addUniverseAgeAxis(ax[0], sim, ageVals=ageVals)

        # mdot: full range
        ax[1].set_xlabel("Redshift")
        ax[1].set_xlim(xlim)
        ax[1].set_ylabel(r"$\dot{M}_{\rm SMBH}$" + "\n" + r"[ log M$_{\rm sun}$ yr$^{-1}$ ]")

        ax[1].plot(redshift[::step], mdot[::step], zorder=0)

        # overplot eddington
        ax[1].plot(redshift[::step], mdot_edd, color="black", label="Eddington")
        ax[1].plot(redshift[::step], mdot_limit, color="black", alpha=0.4, label="Limit")

        for a in ax:
            a.set_rasterization_zorder(1)  # elements below z=1 are rasterized

        fig.savefig(f"smbh_vs_time_{sim.simName}_{smbh_id}.pdf")
        plt.close(fig)


@cache
def _blackhole_position_vs_time_snap(sim):
    """Plot (relative) position of SMBHs vs time, using snapshot information."""
    # load
    r = {}

    sim = sim.copy()

    for snap in sim.validSnapList()[::-1]:
        sim.setSnap(snap)

        if sim.numPart[sim.ptNum("bhs")] == 0:
            continue

        # load all black holes IDs and positions, parent subhalos, relative positions
        ids_loc = sim.bhs("ids")
        pos_loc = sim.bhs("pos")
        hsml_loc = sim.bhs("BH_Hsml")
        sub_ids_loc = sim.bhs("subhalo_id")
        sub_pos_loc = sim.subhalos("SubhaloPos")

        print(snap, sim.redshift, ids_loc.size)

        pos_rel_loc = pos_loc - sub_pos_loc[sub_ids_loc]
        sim.correctPeriodicDistVecs(pos_rel_loc)

        ww = np.where(sub_ids_loc == -1)[0]
        if len(ww) > 0:
            pos_rel_loc[ww, :] = np.nan

        dist_loc = np.linalg.norm(pos_rel_loc, axis=1)
        dist_loc_pc = sim.units.codeLengthToPc(dist_loc)

        time = np.zeros(ids_loc.size, dtype="float32") + sim.tage
        z = np.zeros(ids_loc.size, dtype="float32") + sim.redshift

        # append
        if len(r) == 0:
            r["ids"] = ids_loc
            r["pos"] = pos_loc
            r["hsml"] = hsml_loc
            r["sub_ids"] = sub_ids_loc
            r["pos_rel"] = pos_rel_loc
            r["dist_pc"] = dist_loc_pc
            r["time"] = time
            r["z"] = z
        else:
            r["ids"] = np.hstack((r["ids"], ids_loc))
            r["pos"] = np.vstack((r["pos"], pos_loc))
            r["hsml"] = np.hstack((r["hsml"], hsml_loc))
            r["sub_ids"] = np.hstack((r["sub_ids"], sub_ids_loc))
            r["pos_rel"] = np.vstack((r["pos_rel"], pos_rel_loc))
            r["dist_pc"] = np.hstack((r["dist_pc"], dist_loc_pc))
            r["time"] = np.hstack((r["time"], time))
            r["z"] = np.hstack((r["z"], z))

    # convert to numpy arrays
    return r


# @cache
def _blackhole_position_vs_time(sim, n_pts=400):
    """Plot (relative) position of SMBHs vs time, using txt-files information."""
    # load
    r = {}

    sim = sim.copy()

    smbhs = blackhole_details_mergers(sim)  # , overwrite=True)

    # loop over each black hole
    for smbh_id in smbhs.keys():
        if smbh_id == "mergers":
            continue

        # identify parent subhalo at final snapshot
        bh_ids = sim.bhs("id")
        w = np.where(bh_ids == int(smbh_id))[0]

        if len(w) == 0:
            print("Warning: SMBH ID [%s] not found in final snapshot, skipping!" % smbh_id)
            continue

        # identify central subhalo of fof of this parent, i.e. in case the SMBH is in a satellite
        sub_id = sim.bhs("subhalo_id")[w[0]]
        sub_id = sim.halo(sim.subhalo(sub_id)["SubhaloGrNr"])["GroupFirstSub"]

        # load subhalo mpb position (smoothed?)
        mpb = sim.quantMPB(sub_id, quants=["SubhaloPos"], add_ghosts=True, smooth=True)
        parent_time = 1 / (1 + mpb["z"])

        # interpolate subhalo positions to blackhole times
        bh_time = smbhs[smbh_id]["time"]

        sub_pos = np.zeros((bh_time.size, 3), dtype="float32")
        for i in range(3):
            sub_pos[:, i] = np.interp(bh_time, parent_time[::-1], mpb["SubhaloPos"][:, i][::-1])

        # calculate relative positions and distances
        pos_smbh = np.vstack((smbhs[smbh_id]["x"], smbhs[smbh_id]["y"], smbhs[smbh_id]["z"])).T
        pos_rel = pos_smbh - sub_pos
        sim.correctPeriodicDistVecs(pos_rel)

        # reduce relative positions via running mean
        if n_pts is not None:
            bin_size = int(np.floor(bh_time.size / n_pts))
            offset = 0

            bh_time_bin = np.zeros(n_pts, dtype="float32")
            pos_rel_bin = np.zeros((n_pts, 3), dtype="float32")

            for i in range(n_pts):
                bh_time_bin[i] = np.mean(bh_time[offset : offset + bin_size])
                pos_rel_bin[i, :] = np.mean(pos_rel[offset : offset + bin_size, :], axis=0)
                offset += bin_size
        else:
            bh_time_bin = bh_time
            pos_rel_bin = pos_rel

        bh_z = 1 / bh_time_bin - 1

        dist = np.linalg.norm(pos_rel_bin, axis=1)
        dist_pc = sim.units.codeLengthToPc(dist)

        # bh ids (constant), and parent sub ids (just constant for now)
        ids = np.zeros(dist.size, dtype="int64") + int(smbh_id)
        sub_ids = np.zeros(dist.size, dtype="int64") + sub_id

        # append
        if len(r) == 0:
            r["ids"] = ids
            r["sub_ids"] = sub_ids
            r["pos"] = pos_smbh
            r["pos_rel"] = pos_rel_bin
            r["dist_pc"] = dist_pc
            r["time"] = bh_time_bin
            r["z"] = bh_z
        else:
            r["ids"] = np.hstack((r["ids"], ids))
            r["sub_ids"] = np.hstack((r["sub_ids"], sub_ids))
            r["pos"] = np.vstack((r["pos"], pos_smbh))
            r["pos_rel"] = np.vstack((r["pos_rel"], pos_rel))
            r["dist_pc"] = np.hstack((r["dist_pc"], dist_pc))
            r["time"] = np.hstack((r["time"], bh_time))
            r["z"] = np.hstack((r["z"], bh_z))

    return r


def blackhole_position_vs_time(sim, snap_based=True):
    """Plot (relative) position of SMBHs vs time."""
    if snap_based:
        data = _blackhole_position_vs_time_snap(sim)
        data_snap = data
    else:
        data = _blackhole_position_vs_time(sim, n_pts=None)  #
        data_snap = _blackhole_position_vs_time_snap(sim)  # for BH_Hsml

    # loop over unique IDs
    smbh_ids = np.unique(data["ids"])
    for smbh_id in smbh_ids:
        print("plot: ", smbh_id)

        # get data subset
        w = np.where(data["ids"] == smbh_id)[0]
        sort_inds = np.argsort(data["time"][w])
        w = w[sort_inds]

        pos = data["pos"][w]
        pos_rel = data["pos_rel"][w]
        z = data["z"][w]
        dist_pc = data["dist_pc"][w]
        sub_ids = data["sub_ids"][w]

        # plot
        if len(smbh_ids) > 1:
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 7.5))
        else:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 7.5))

        # plot (1): distance from center vs. time
        ax1.set_xlabel("Redshift")
        ax1.set_ylabel("Distance from Subhalo Center [pc]")
        # ax1.set_ylim([-1, 10])

        ax1.plot(z, dist_pc, lw=lw - 1, color="black")  # auto axes limits
        colored_line(z, dist_pc, c=z, ax=ax1, lw=lw, cmap="plasma")

        ageVals = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        addUniverseAgeAxis(ax1, sim, ageVals=ageVals)

        # plot (2): projected position in xy plane
        ax2.set_xlabel("x [ckpc/h]")
        ax2.set_ylabel("y [ckpc/h]")
        ax2.set_box_aspect(1.0)
        ax2.set_xlim([-1.1, 1.1])
        ax2.set_ylim([-1.1, 1.1])

        # draw circle at (final) subhalo size
        sub_rhalf = sim.subhalos("SubhaloHalfmassRadType")[:, 4]
        sub_id = sub_ids[-1]
        print(f" [{sim}] subhalo ID: {sub_id}, r_half: {sub_rhalf[sub_id]:.3f} ckpc/h")
        c1 = plt.Circle((0, 0), sub_rhalf[sub_id], color="black", alpha=0.3, zorder=-1)
        c2 = plt.Circle((0, 0), 2 * sub_rhalf[sub_id], color="black", alpha=0.3, zorder=-1)
        ax2.add_artist(c1)
        ax2.add_artist(c2)

        colored_line(pos_rel[:, 0], pos_rel[:, 1], c=z, ax=ax2, lw=lw, cmap="plasma")

        # plot (3) pairwise distances to all other black holes
        if len(smbh_ids) > 1:
            ax3.set_xlabel("Redshift")
            ax3.set_ylabel("Distance to other BHs [ckpc/h]")

            for other_smbh_id in smbh_ids:
                if other_smbh_id == smbh_id:
                    continue

                # get data subset
                w2 = np.where(data["ids"] == other_smbh_id)[0]
                sort_inds2 = np.argsort(data["time"][w2])
                w2 = w2[sort_inds2]

                pos2 = data["pos"][w2]
                z2 = data["z"][w2]

                # interpolate positions to common times
                pos2_interp = np.zeros((pos.shape[0], 3), dtype="float32")
                for i in range(3):
                    pos2_interp[:, i] = np.interp(z[::-1], z2[::-1], pos2[:, i][::-1])[::-1]

                pos_rel = pos - pos2_interp
                sim.correctPeriodicDistVecs(pos_rel)

                dist_rel_code = np.linalg.norm(pos_rel, axis=1)

                ax3.plot(z, dist_rel_code, lw=lw - 1, label=other_smbh_id)  # auto axes limits

            # get BH_Hsml from snapshot-based data
            w_snap = np.where(data_snap["ids"] == smbh_id)[0]
            sort_inds_snap = np.argsort(data_snap["time"][w_snap])
            w_snap = w_snap[sort_inds_snap]
            z_snap = data_snap["z"][w_snap]

            # interpolate hsml to common times
            hsml2 = data_snap["hsml"][w_snap]
            hsml2_interp = np.interp(z[::-1], z_snap[::-1], hsml2[::-1])[::-1]

            ax3.plot(z, hsml2_interp, lw=lw, color="black", linestyle="-", label="BH Hsml")

            ax3.set_yscale("log")

            addUniverseAgeAxis(ax3, sim, ageVals=ageVals)
            ax3.legend(loc="best")

        # save
        fig.savefig(f"smbh_pos_vs_time_{sim.simName}_{smbh_id}{'_snap' if snap_based else ''}.pdf")
        plt.close(fig)


def starformation_diagnostics(sims, supernovae=False, split_z=True, sizefac=1.0):
    """Plot PDFs of gas properties at the sites and moments of star formation (or supernovae)."""
    # config
    z_bins = [[5.5, 8.0], [8.0, 15.0]]  # [[5.5, 8.0], [8.0, 10.0], [10.0, 15.0]]
    if not split_z:
        z_bins = [[5.5, 15.0]]

    dens_lim = [1, 8] if not supernovae else [-6, 8]  # log cm^-3
    temp_lim = [1, 5.5] if not supernovae else [1, 9.5]  # log K
    metallicity_lim = [-5.1, 1.0] if not supernovae else [-5.1, 2]  # log Z/Z_solar

    dens_label = "Ambient Gas Density [ log cm$^{-3}$ ]"
    temp_label = "Ambient Gas Temperature [ log K ]"
    metallicity_label = "Ambient Gas Metallicity [ log Z/Z$_{\\odot}$ ]"

    for field in ["Density", "Temperature", "Metallicity"]:
        # plot
        fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

        if field == "Density":
            xlabel = dens_label
            xlim = dens_lim
        if field == "Temperature":
            xlabel = temp_label
            xlim = temp_lim
        if field == "Metallicity":
            xlabel = metallicity_label
            xlim = metallicity_lim

        ax.set_xlabel(xlabel)
        if supernovae:
            ax.set_ylabel("Number of Supernovae")
        else:
            ax.set_ylabel("Number of Stars Formed")
        ax.set_yscale("log")
        ax.set_xlim(xlim)

        # loop over simulations
        for i, sim in enumerate(sims):
            data, data_sn = sf_sn_details(sim)

            if supernovae:
                data = data_sn

            if field == "Density":
                # unit conversions: physical [log 1/cm^3]
                dens = sim.units.codeDensToPhys(data["Density"], scalefac=data["Time"], cgs=True, numDens=True)
                dens[dens <= 0] = dens[dens > 0].min()  # zeros/negatives rarely occur (including corrupted txt lines)
                dens[~np.isfinite(dens)] = dens[np.isfinite(dens)].max()  # rarely inf
                vals = np.log10(dens)

            if field == "Temperature":
                # [log K]
                temp = data["Temperature"]
                temp[temp <= 0] = temp[temp > 0].min()  # zeros rarely occur
                vals = np.log10(temp)

            if field == "Metallicity":
                # unit conversions: [log solar]
                metallicity = sim.units.metallicityInSolar(data["Metallicity"])
                metallicity[metallicity == 0] = metallicity[metallicity > 0].min()  # zeros rarely occur
                vals = np.log10(metallicity)

            z = 1 / data["Time"] - 1

            for j, z_bin in enumerate(z_bins):
                w = np.where((z >= z_bin[0]) & (z < z_bin[1]))[0]

                if len(w) == 0:
                    continue

                # plot hist
                label = f"h{sim.hInd}" if j == 0 else ""  # sim.simName
                label = sim.simName

                c = colors[i % len(colors)]

                # use numpy for histogram
                hist, bins = np.histogram(vals[w], bins=40)
                ax.step(bins[:-1], hist, where="post", color=c, linestyle=linestyles[j], label=label)

        # second legend
        if split_z:
            labels = [f"{z_bin[0]} < z < {z_bin[1]}" for z_bin in z_bins]
            handles = [
                plt.Line2D([0], [0], ls=linestyles[i], color="black", label=labels[i]) for i in range(len(z_bins))
            ]
            legend2 = ax.legend(handles, labels, loc="upper left")
            ax.add_artist(legend2)

        # finish plot
        hInds = sorted({sim.hInd for sim in sims})
        ax.legend(loc="upper right")
        fig.savefig(f"{'sn' if supernovae else 'sf'}_{field}{'_h' + str(hInds[0]) if len(hInds) == 1 else ''}.pdf")
        plt.close(fig)

    # two-dimensional density-temperature diagram at star formation sites
    for sim in sims:
        # load
        data, data_sn = sf_sn_details(sim)

        # density: physical [log 1/cm^3]
        dens = sim.units.codeDensToPhys(data["Density"], scalefac=data["Time"], cgs=True, numDens=True)
        dens[dens <= 0] = dens[dens > 0].min()  # zeros/negatives rarely occur (including corrupted txt lines)
        dens[~np.isfinite(dens)] = dens[np.isfinite(dens)].max()  # rarely inf
        dens = np.log10(dens)

        # temperature [log K]
        temp = data["Temperature"]
        temp[temp <= 0] = temp[temp > 0].min()  # zeros rarely occur
        temp = np.log10(temp)

        # restrict to redshift bins
        z = 1 / data["Time"] - 1

        for j, z_bin in enumerate(z_bins):
            w = np.where((z >= z_bin[0]) & (z < z_bin[1]))[0]

            if len(w) == 0:
                continue

            # histogram in 2d
            nBins2D = [120, 80]

            cc, _, _, _ = binned_statistic_2d(dens[w], temp[w], None, "count", bins=nBins2D, range=[dens_lim, temp_lim])

            cc = cc.T  # imshow convention
            cc = np.log10(cc)

            # start figure
            fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))
            ax.set_xlabel(dens_label)
            ax.set_ylabel(temp_label)
            ax.set_xlim(dens_lim)
            ax.set_ylim(temp_lim)

            # plot
            extent = [ax.get_xlim()[0], ax.get_xlim()[1], ax.get_ylim()[0], ax.get_ylim()[1]]
            plt.imshow(cc, extent=extent, origin="lower", interpolation="nearest", aspect="auto")

            # finish figure
            fig.savefig(f"{'sn' if supernovae else 'sf'}_z{j}_2d_{sim.simName}.pdf")
            plt.close(fig)


def select_ics():
    """Helper to select halos from TNG50 for resimulation."""
    import illustris_python as il

    sim = simParams(run="tng50-1", redshift=5.5)

    mhalo_min = 8.0
    dist_threshold = 1000.0  # code units (ckpc/h), within which no other more massive halo

    # check existence of cache file, if not, compute now
    cachefile = sim.cachePath + f"mpb_ids_{sim.simName}_{sim.snap}_{mhalo_min:.1f}_{dist_threshold:.0f}.hdf5"

    # load halo massees at target redshift (all centrals by definition)
    mhalo = sim.subhalos("mhalo_log")
    mstar = sim.subhalos("mstar2_log")
    grnr = sim.subhalos("SubhaloGrNr")

    if isfile(cachefile):
        # load
        with h5py.File(cachefile, "r") as f:
            z_target_mpb_ids = f["z_target_mpb_ids"][()]
            sub_ids = f["sub_ids"][()]
        print(f"Loaded [{cachefile}].")
    else:
        # load at target redshift (all centrals by definition)
        sub_ids = np.where(mhalo >= mhalo_min)[0]

        print(f"Found [{len(sub_ids)}] halos with Mhalo >= {mhalo_min}.")

        # env measure
        ac = "Subhalo_Env_Closest_Distance_MhaloRel_GtSelf"
        dist_closest = sim.auxCat(ac, expandPartial=True)[ac]

        w = np.where(dist_closest[sub_ids] > dist_threshold)[0]

        print(f"Found [{len(w)}] of these halos with no more massive neighbor within {dist_threshold} ckpc/h.")

        sub_ids = sub_ids[w]

        z_target_mpb_ids = np.zeros(sub_ids.size, dtype="int32") - 1

        for i, sub_id in enumerate(sub_ids):
            # load MDB to z=0, then load MPB to z_target, and save
            print(i, sub_id)
            fields = ["SnapNum", "SubfindID"]
            mdb = sim.loadMDB(sub_id, fields=fields)

            snap_z0 = 99
            w = np.where(mdb["SnapNum"] == snap_z0)[0]
            if len(w) == 0:
                continue

            z0_id = mdb["SubfindID"][w[0]]

            # then load MPB to z_target
            mpb = il.sublink.loadTree(sim.simPath, snap_z0, z0_id, fields=fields, onlyMPB=True)

            # check if same
            w = np.where(mpb["SnapNum"] == sim.snap)[0]
            if len(w) == 0:
                continue

            z_target_mpb_ids[i] = mpb["SubfindID"][w[0]]

        # save
        with h5py.File(cachefile, "w") as f:
            f["z_target_mpb_ids"] = z_target_mpb_ids
            f["sub_ids"] = sub_ids
        print(f"Saved [{cachefile}].")

    # sub-select halos that are on their own MPBs
    w = np.where(z_target_mpb_ids == sub_ids)

    sub_ids = sub_ids[w]

    # halo masses and IDs
    mhalo = mhalo[sub_ids]
    mstar = mstar[sub_ids]

    # bin in halo masses
    rng = np.random.default_rng(42424242)

    massbins = [[8.0, 8.1], [8.5, 8.6], [9.0, 9.1], [9.5, 9.6], [10.0, 10.1], [10.5, 10.6], [11.0, 11.1]]
    # massbins = [[8.1, 8.2], [8.2, 8.3], [8.3, 8.4]]

    mstar_tol = 0.2

    for massbin in massbins:
        # select in halo mass alone (after prior selections above)
        w = np.where((mhalo >= massbin[0]) & (mhalo < massbin[1]))[0]
        print(massbin, len(w), mhalo[w].mean(), np.nanmean(mstar[w]))

        # select as non-extreme outliers on the mstar-mhalo relation at z_target according to TNG50
        if np.count_nonzero(np.isfinite(mstar[w])):
            mstar_median = np.nanmedian(mstar[w])

            w = np.where(
                (mhalo >= massbin[0])
                & (mhalo < massbin[1])
                & (mstar >= mstar_median - mstar_tol)
                & (mstar < mstar_median + mstar_tol)
            )[0]

            print(" with mstar constraint: ", len(w), mhalo[w].mean(), np.nanmean(mstar[w]))
        else:
            print(" no mstar constraint (all nan)")

        sub_ids_bin = sub_ids[w]
        halo_ids = grnr[sub_ids_bin]
        rng.shuffle(halo_ids)
        print(" haloIDs: ", halo_ids[0:5])


def run_table_latex():
    """Helper to generate LaTeX tables of simulation parameters."""
    # list of sims to include
    variants = ["ST15"]
    res = [14, 15, 16]
    hInds = [1958, 5072, 15581, 23908, 31619, 73172, 219612, 311384, 446076, 539722, 844537]
    redshift = 5.5

    sims = _get_existing_sims(variants, res, hInds, redshift, all=True, single=True)

    for sim in sims:
        # load
        sfr_100myr = sim.subhalos("sfr_100myr")
        z_stars_masswt = logZeroNaN(sim.subhalos("z_stars_masswt"))  # no radial restriction
        z_gas_sfrwt = logZeroNaN(sim.subhalos("z_gas_sfrwt"))  # no radial restriction
        R_stars = sim.subhalos("size_stars") * 1000  # physical pc

        mhalo = sim.subhalos("mhalo_log")  # log msun
        mstar = sim.subhalos("mstar_log")  # log msun
        cen_flag = sim.subhalos("cen_flag")
        grnr = sim.subhalos("SubhaloGrNr")

        vrot = sim.subhalos("vrot_stars_map")
        sigma = sim.subhalos("veldisp_stars_map")

        M_BH = sim.units.codeMassToLogMsun(sim.bhs("BH_Mass"))
        M_BH = np.max(M_BH)  # simulation wide

        # show first two halos
        for haloID in [0, 1]:
            subID = sim.halo(haloID)["GroupFirstSub"]

            N_Cl = np.where((grnr == haloID) & (cen_flag == 0) & (mstar > 0))[0]

            s = f"{sim.simName:<16} z={sim.redshift:.2f} [{haloID = }] "
            s += f"Mhalo: {mhalo[subID]:.2f} "
            s += f"Mstar: {mstar[subID]:.2f} ".replace("nan", "   0")
            s += f"SFR: {sfr_100myr[subID]:.2f} "
            s += f"Z*: {z_stars_masswt[subID]:.2f} ".replace("nan", "  ---")
            s += f"Zgas: {z_gas_sfrwt[subID]:.2f}".replace("nan", "  ---")
            s += f" R*: {R_stars[subID]:5.1f} ".replace("nan", "  ---")
            s += f"Vrot: {vrot[subID]:4.1f} ".replace("nan", "---")
            s += f"Sigma: {sigma[subID]:4.1f} ".replace("nan", "---")
            s += f"(V/s): {vrot[subID] / sigma[subID]:4.1f} ".replace("nan", "---")
            s += f"N_Cl: {N_Cl.size:>3}"
            s += f" M_BH: {M_BH:.3f}"
            print(s)


# -------------------------------------------------------------------------------------------------


def paperPlots(a=False):
    """Plots for MCST intro paper. (if a == True, make all figures)."""
    # list of sims to include
    variants = ["ST15"]  # ['ST15c','ST15m','ST15s']
    res = [14, 15, 16]
    # hInds = [31619, 73172, 219612, 446076, 539722]
    hInds = [1958, 5072, 15581, 23908, 31619, 73172, 219612, 311384, 446076, 539722, 844537]
    redshift = 5.5

    # if (all == False), only dz < 0.1 matches
    # if (single == True), only the highest available res of each halo
    sims = _get_existing_sims(variants, res, hInds, redshift, all=False, single=True)

    # contamination diagnostic printout (info only)
    for sim in []:  # sims:
        subIDs = _zoomSubhaloIDsToPlot(sim)
        for subID in subIDs:
            subhalo = sim.subhalo(subID)
            s = f" h[{subhalo['SubhaloGrNr']}] sub[{subID:4d}] "
            s += f"Re = {sim.units.codeLengthToPc(subhalo['SubhaloHalfmassRadType'][4]):.2f} pc, "
            s += f"M_BH = {sim.units.codeMassToLogMsun(subhalo['SubhaloBHMass'])[0]:.2f}"
            print(s)

    # ------------

    # fig 1: equilibrium curves of new grackle tables
    if 0:
        from temet.cosmo.cooling import grackle_equil_vs_Zz

        grackle_equil_vs_Zz()

    # fig 2: simulation comparison meta-plot
    if 0:
        simHighZComparison()

    # fig 3: composite vis (i) parent box dm, (ii) halo-scale gas, (iii) galaxy-scale gas+stars
    if 0:
        sim_parent = simParams("tng50-1", redshift=6.0)  # z=5.5 is a mini snap, no DM hsml
        vis_parent_box(sim_parent)
        sims[0].haloInd = 0
        vis_single_halo(sims[0])
        vis_single_galaxy(sims[0])

    # figs 4,5: multi-sim galleries
    if 0:
        # sims_loc = sims[0:9] # limit to first N sims for layout
        sims_loc = []
        v = "ST14"
        sims_loc.append(simParams("structures", hInd=5072, res=14, variant=v, snap=346, haloInd=0))
        sims_loc.append(simParams("structures", hInd=15581, res=14, variant=v, redshift=5.8, haloInd=0))
        sims_loc.append(simParams("structures", hInd=23908, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams("structures", hInd=31619, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams("structures", hInd=31619, res=14, variant=v, redshift=5.5, haloInd=1))
        sims_loc.append(simParams("structures", hInd=73172, res=14, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams("structures", hInd=219612, res=15, variant=v, redshift=5.5, haloInd=0))
        sims_loc.append(simParams("structures", hInd=311384, res=15, variant=v, redshift=6.0, haloInd=0))
        sims_loc.append(simParams("structures", hInd=844537, res=15, variant=v, redshift=5.5, haloInd=0))

        vis_gallery_galaxy(sims_loc, conf=0)
        vis_gallery_galaxy(sims_loc, conf=1)

    # fig 6a: sfr vs mstar relation
    if 0 or a:
        sfr_vs_mstar(sims, yQuant="sfr_100myr")
        sfr_vs_mstar(sims, yQuant="sfr_10myr")

    # fig 6b: star formation history (using stellar histo) (all halos in one panel)
    if 0 or a:
        quant = "sfr2"
        opts = {"xlim": [12.5, 5.5], "ylim": [-5.5, 0.5], "sizefac": [1.3, 0.7], "legend": "simple"}

        subhalos_evo.tracks1d(sims, quant, sfh_lin=False, sfh_treebased=False, parents=False, **opts)

    # fig 6b: star formation history (using stellar histo) (one plot per halo, i.e. gallery of small panels)
    if 0 or a:
        quant = "sfr2"
        opts = {"xlim": [12.5, 5.5], "ylim": [-5.5, 0.5], "sizefac": 0.6, "legend": "simple"}
        # opts["f_selection"] = _zoomSubhaloIDsToPlot  # plot additional (uncontamined) galaxies as faint lines

        for i, sim in enumerate(sims):
            opts["color"] = colors[i]
            subhalos_evo.tracks1d([sim], quant, sfh_lin=False, sfh_treebased=False, parents=False, **opts)

    # fig 7: sfr burstyness (10/100 myr ratios) vs redshift
    if 0 or a:
        sfr_10_100_ratio(sims)

    # fig 8a: smhm relation
    if 0 or a:
        smhm_relation(sims)

    # fig 8b: stellar mass vs redshift evo (using stellar histo)
    if 0 or a:
        quant = "mstar2_log"
        xlim = [12.5, 5.5]
        ylim = [4.0, 9.0]

        opts = {
            "xlim": xlim,
            "ylim": ylim,
            "sizefac": 0.8,
            "legend": "simple",
            "legend_locs": ["lower right", "upper left"],
            "legend_ncols": [1, 4],
            "f_selection": _zoomSubhaloIDsToPlot,
        }

        subhalos_evo.tracks1d(sims, quant, sfh_treebased=False, parents=False, **opts)
        # subhalos_evo.tracks1d(sims, quant='mgas2_log', **opts)

    # fig 8c: density, temperature, and metallicity PDFs at star formation and supernovae sites
    if 0 or a:
        split_z = True
        starformation_diagnostics(sims, split_z=split_z, sizefac=0.8)
        starformation_diagnostics(sims, supernovae=True, split_z=split_z, sizefac=0.8)

    # fig 9: phase space diagrams (show one halo)
    if 0 or a:
        sims = [simParams("structures", hInd=219612, res=16, variant="ST15", snap=375)]
        for sim in sims:
            phase_diagram(sim)  # mass
            phase_diagram(sim, cQuant="Z_solar")
            phase_diagram(sim, cQuant="vrad")
            phase_diagram(sim, cQuant="rad_rvir")

    # fig 10a - gas metallicity
    if 0 or a:
        gas_mzr(sims)

    # fig 10b - stellar metallicity
    if 0 or a:
        stellar_mzr(sims)

    # fig 10c - metallicity vs time evolution
    if 0 or a:
        opts = {
            "xlim": [14.1, 5.5],
            "ylim": [-4.3, 1.0],
            "parents": False,
            "smooth": False,
            "monotonic": True,
            "legend": "simple",
            "legend_locs": ["lower right", "upper left"],
            "legend_ncols": [1, 3],
            "sizefac": 0.8,
            "f_selection": _zoomSubhaloIDsToPlot,
        }

        # subhalos_evo.tracks1d(sims, quant="Z_gas", **opts)  # cat/tree
        # subhalos_evo.tracks1d(sims, quant="Z_gas_sfrwt", **opts)  # cat/tree
        subhalos_evo.tracks1d(sims, quant="Z_stars", **opts)  # cat/tree (<2rhalf)
        # subhalos_evo.tracks1d(sims, quant="Z_stars_masswt", **opts)  # aux (subhalo)
        ## subhalos_evo.tracks1d(sims, quant='Z_stars_2rhalfstarsfof_masswt', **opts) # aux
        ## subhalos_evo.tracks1d(sims, quant='Z_stars_1kpc_masswt', **opts) # aux
        ## subhalos_evo.tracks1d(sims, quant='Z_stars_fof_masswt', **opts) # aux

    # fig 11a - stellar sizes
    if 0 or a:
        sizes_vs_mstar(sims)

    # fig 11b - stellar size evo
    if 0 or a:
        opts = {
            "xlim": [14.1, 5.5],
            "ylim": [-3.5, 0.7],
            "parents": False,
            "smooth": False,
            "smooth_custom": True,
            "legend": "simple",
            "legend_ncols": [4, 4],
            "legend_locs": ["lower right", "upper left"],
            "sizefac": 0.8,
            "f_selection": _zoomSubhaloIDsToPlot,
        }

        subhalos_evo.tracks1d(sims, quant="size_stars_log", **opts)
        # subhalos_evo.tracks1d(sims, quant='rhalf_stars_fof', **opts)

    # fig 11c - gas sizes
    if 0 or a:
        size_halpha_vs_mstar(sims)

    # fig X: ratio of (stellar/Halpha) size
    # fig X: CII-sizes (e.g. use low-T selection), compare to H-alpha sizes (Ikeda+25, CRISTAL)

    # fig 12: vis of single galaxy, gallery of fields
    if 0:
        sim = simParams("structures", hInd=23908, res=14, variant="ST14", redshift=5.5, haloInd=0)  # original
        # sim = simParams("structures", hInd=23908, res=15, variant="ST15", redshift=10.0, haloInd=0)
        vis_single_galaxy(sim, size=4.0, conf=1)
        vis_single_galaxy(sim, size=4.0, conf=2)
        vis_single_galaxy(sim, size=4.0, conf=3)

        # for i in range(1, 14):
        #    vis_single_large(sim, size=4.0, conf=i)

        # vis_single_galaxy(sim, noSats=True)
        # vis_single_halo(sim, haloID=0)
        # vis_gallery_manyfields(sim, haloID=0)

    # fig 13a - smbh vs mhalo and mstar relations
    if 0 or a:
        # mbh_vs_mhalo(sims)
        mbh_vs_mstar(sims)

    # fig 13b - black hole time evolution
    if 0 or a:
        for sim in sims:
            blackhole_properties_vs_time(sim)
            # blackhole_position_vs_time(sim, snap_based=True)
            # blackhole_position_vs_time(sim, snap_based=False)

    # diagnostic: halo mass, virial radii, mpb-based mstar, rhalf/rvir ratio, all vs redshift
    if 0 or a:
        opts = {"xlim": [12.1, 5.5], "sizefac": 0.8, "f_selection": _zoomSubhaloIDsToPlot}

        subhalos_evo.tracks1d(sims, "mstar2", ylim=[3.5, 7.5], sfh_treebased=True, **opts)
        subhalos_evo.tracks1d(sims, "mstar_fof", ylim=[3.5, 7.5], sfh_treebased=True, **opts)
        subhalos_evo.tracks1d(sims, "mhalo", ylim=[6.0, 11.0], parents=False, **opts)
        subhalos_evo.tracks1d(sims, "rvir", ylim=[0.5, 2.0], parents=False, **opts)
        subhalos_evo.tracks1d(sims, "rhalf_stars", ylim=[-2.0, 0.5], parents=False, **opts)
        subhalos_evo.tracks1d(sims, "re_rvir_ratio", ylim=[-3.5, -0.5], parents=False, **opts)

    # diagnostic: CPU times
    if 0 or a:
        from temet.plot.perf import plotCpuTimes

        plotCpuTimes(sims, xlim=[0.0, 0.25])


def makeMovies():
    """Make movie frames."""
    redshift = 5.5

    sim = simParams(run="structures", res=15, hInd=73172, variant="ST15", redshift=redshift)

    # # movie: galaxy-scale gas + stars vis (tree mpb manual search)
    # # vis_movie(sim, haloID=0)

    # movie: galaxy-scale gas + stars vis (final tree mpb smoothed)
    # vis_movie_mpbsm([sim], conf=1)

    # movie: halo-scale many fields (final tree mpb smoothed)
    # vis_single_halo(sim, movie=True)

    # # movie: galaxy-scale many fields (final tree mpb smoothed)
    # # vis_single_halo(sim, movie=True, galscale=True)

    # movie: phase space diagram
    for snap in sim.validSnapList():
        sim.setSnap(snap)
        phase_diagram(sim, ext="png")
    sim.setRedshift(redshift)

    # movie: high-res region
    for snap in sim.validSnapList():
        sim.setSnap(snap)
        vis_highres_region(sim, partType="gas")
        # vis_highres_region(sim, partType='dm')


def makeMoviesMulti():
    """Make movie frames."""
    # movie: all simulations together (multi-panel) (final tree mpb smoothed)
    sims_loc = []
    v = "ST15"
    z = 5.5
    # sims_loc.append(simParams("structures", hInd=1958, res=14, variant=v, redshift=z, haloInd=0))
    # sims_loc.append(simParams("structures", hInd=5072, res=14, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=15581, res=14, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=15581, res=14, variant=v, redshift=z, haloInd=4))
    sims_loc.append(simParams("structures", hInd=23908, res=14, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=23908, res=14, variant=v, redshift=z, haloInd=2))
    sims_loc.append(simParams("structures", hInd=31619, res=14, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=31619, res=14, variant=v, redshift=z, haloInd=1))
    sims_loc.append(simParams("structures", hInd=73172, res=14, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=219612, res=15, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=311384, res=15, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=446076, res=15, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=539722, res=15, variant=v, redshift=z, haloInd=0))
    sims_loc.append(simParams("structures", hInd=844537, res=16, variant=v, redshift=z, haloInd=0))

    vis_movie_mpbsm_multi(sims_loc, conf=1)
