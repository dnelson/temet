"""
Generalized plots following group catalog objects (i.e. subhalos) through time (i.e. for zooms).
"""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from scipy.signal import savgol_filter

from ..plot.config import colors, figsize, linestyles, lw, markers, sKn, sKo
from ..util.helper import closest, logZeroNaN, running_median
from ..util.simParams import simParams


def _add_legends(ax, hInds, res, variants, colors, lineplot=False):
    """Plot helper to add two legends: one showing hInds (color), one showing res/variants (symbols and markersizes)."""
    locs = ["upper left", "lower right"]
    if r"1/2,\star" in ax.get_ylabel():
        locs = locs[::-1]

    # legend one
    handles, labels = ax.get_legend_handles_labels()

    if len(hInds) == 1 and lineplot:
        # if we have only one halo, vary the linestyle by variant or res (for e.g. quant lines vs redshift)
        if len(variants) > 1 and len(res) == 1:
            for i, variant in enumerate(variants):
                handles.append(plt.Line2D([0], [0], color=colors[i], ls=linestyles[0]))
                labels.append("h%d_L%d_%s" % (hInds[0], res[0], variant))
        if len(res) > 1 and len(variants) == 1:
            for i, r in enumerate(res):
                handles.append(plt.Line2D([0], [0], color=colors[i], ls=linestyles[0]))
                labels.append("h%d_L%d_%s" % (hInds[0], r, variants[0]))
        if len(res) > 1 and len(variants) > 1:
            for i, r in enumerate(res):
                handles.append(plt.Line2D([0], [0], color=colors[i], ls="-"))
                labels.append("h%d_L%d" % (hInds[0], r))
            for i, variant in enumerate(variants):
                handles.append(plt.Line2D([0], [0], color="black", ls=linestyles[i]))
                labels.append("%s" % variant)
    else:
        for hInd in hInds:
            # color by hInd
            c = colors[hInds.index(hInd)]

            handles.append(plt.Line2D([0], [0], color=c, ls="-"))
            labels.append("h%d" % hInd)

    legend = ax.legend(handles, labels, loc=locs[0], ncols=np.min([3, len(variants)]))
    ax.add_artist(legend)

    if len(hInds) == 1 and lineplot:
        return

    # legend two
    handles = []
    labels = []

    for variant in variants:
        for r in res:
            # marker set by variant
            marker = markers[variants.index(variant)]
            ms = (r - 10) * 2.5 + 4

            handles.append(plt.Line2D([0], [0], color="black", lw=0, marker=marker, ms=ms))
            labels.append("L%d_%s" % (r, variant))

    legend2 = ax.legend(handles, labels, loc=locs[1], ncols=np.min([2, len(variants)]))
    ax.add_artist(legend2)


def scatter2d(
    sims,
    xQuant,
    yQuant,
    xlim=None,
    ylim=None,
    vstng100=False,
    vstng50=True,
    tracks=True,
    sizefac=1.0,
    f_selection=None,
    f_pre=None,
    f_post=None,
):
    """Scatterplot between two quantities, optionally including time evolution tracks through this plane.

    Designed for comparison between many zoom runs, including the target subhalo(s) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      xQuant (str): name of quantity to plot on the x-axis.
      yQuant (str): name of quantity to plot on the y-axis.
      xlim (list[float][2]): if not None, override default x-axis limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      vstng100 (bool): if True, plot the TNG100-1 relation for comparison.
      vstng50 (bool): if True, plot the TNG100-1 relation for comparison.
      tracks (bool): if True, plot tracks of individual galaxies. If False, only plot final redshift values.
      sizefac (float): multiplier on figure size.
      f_selection (function): if not None, this 'custom' function hook is called to determine which
        subhalo IDs to plot for each sim. It must accept a single argument: the simulation object,
        and return a list of subhalo IDs to plot. If None, defaults to sim.zoomSubhaloID only.
      f_pre (function): if not None, this 'custom' function hook is called just before plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
      f_post (function): if not None, this 'custom' function hook is called just after plotting.
        It must accept two arguments: the figure axis, and a list of simulation objects.
    """
    # currently assume all sims have the same parent
    # rng = np.random.default_rng(424242)
    sim_parent = sims[0].sP_parent.copy()

    # show relation/values from the parent box at the same redshift as selected for the zooms
    sim_parent.setRedshift(sims[0].redshift)

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, "All sims must have the same parent box."

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted({sim.hInd for sim in sims})
    res = sorted({sim.res for sim in sims})
    variants = sorted({sim.variant for sim in sims})

    # load: parent box relation (and also field metadata)
    sim_parent_relation = sim_parent

    if vstng100:
        sim_parent_relation = simParams(run="tng100-1", redshift=sim_parent.redshift)
    if vstng50:
        sim_parent_relation = simParams(run="tng50-1", redshift=sim_parent.redshift)

    parent_xvals, xlabel, xMinMax, xLog = sim_parent_relation.simSubhaloQuantity(xQuant)
    if xlim is not None:
        xMinMax = xlim
    if xLog:
        parent_xvals = logZeroNaN(parent_xvals)

    parent_yvals, ylabel, yMinMax, yLog = sim_parent_relation.simSubhaloQuantity(yQuant)
    if ylim is not None:
        yMinMax = ylim
    if yLog:
        parent_yvals = logZeroNaN(parent_yvals)

    parent_cen = sim_parent_relation.subhalos("cen_flag")
    w = np.where(parent_cen == 1)

    xm, ym, _, pm = running_median(parent_xvals[w], parent_yvals[w], binSize=0.05, percs=[5, 16, 50, 84, 95])

    # mass threshold
    if "mhalo" in xQuant:
        mhalo_min = sim_parent_relation.units.codeMassToLogMsun(sim_parent_relation.dmParticleMass * 100)
        w = np.where(xm >= mhalo_min)[0]
        xm = xm[w]
        ym = ym[w]
        pm = pm[:, w]

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    # parent box relation
    pm = savgol_filter(pm, sKn, sKo, axis=1)
    ax.fill_between(xm, pm[0, :], pm[-1, :], color="#bbb", alpha=0.4)
    ax.fill_between(xm, pm[1, :], pm[-2, :], color="#bbb", alpha=0.6)
    ax.plot(xm, ym, color="#bbb", lw=lw * 2, alpha=1.0, label=sim_parent_relation)

    if f_pre is not None:
        f_pre(ax, sims)

    # individual zoom runs
    for _i, sim in enumerate(sims):
        # load
        xvals = sim.subhalos(xQuant)
        yvals = sim.subhalos(yQuant)

        if xLog:
            xvals = logZeroNaN(xvals)
        if yLog:
            yvals = logZeroNaN(yvals)

        # which subhalo(s) to include?
        if f_selection is not None:
            subhaloIDs = f_selection(sim)
        else:
            subhaloIDs = [sim.zoomSubhaloID]

        # loop over each subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            xval = xvals[subhaloID]
            yval = yvals[subhaloID]

            if np.isnan(xval) or np.isnan(yval):
                print(f"NaN in {sim.simName} {xQuant}={xval} {yQuant}={yval}")
            if xval < xMinMax[0] or xval > xMinMax[1] or yval < yMinMax[0] or yval > yMinMax[1]:
                print(f"Out of bounds in {sim.simName} {xQuant}={xval:.3f} {yQuant}={yval:.3f}")

            marker_lim = False  # None
            if np.isnan(yval) or yval < yMinMax[0]:
                yval = yMinMax[0]  # + (yMinMax[1]-yMinMax[0])/25 * rng.uniform(0.6, 1.0)
                print(f" set [y] {yQuant}={yval:.3f} for visibility.")
                marker_lim = True  # 11 #CARETDOWNBASE #r'$\downarrow$'
            if np.isnan(xval) or xval < xMinMax[0]:
                xval = xMinMax[0]  # + (xMinMax[1]-xMinMax[0])/25 * rng.uniform(0.6, 1.0)
                print(f" set [x] {xQuant}={xval:.3f} for visibility.")
                marker_lim = True  # 8 #CARETLEFTBASE #r'$\leftarrow$'

            # color set by hInd
            c = colors[hInds.index(sim.hInd)]

            # marker set by variant
            marker = markers[variants.index(sim.variant) % len(markers)]

            # marker size set by resolution
            ms_loc = (sim.res - 10) * 2.5 + 4
            # lw_loc = sim.res - 10

            # filled for main target, open for additional halos
            style = {"color": c, "ms": ms_loc, "fillstyle": "full"}
            if j > 0:
                style["fillstyle"] = "none"
                style["markeredgewidth"] = 2

            clip = False if marker_lim else True

            (l,) = ax.plot(xval, yval, marker=marker, clip_on=clip, label="", **style)

            if tracks and sim.hasMergerTree:
                # various criterion for how far back to go
                max_z = 10.0
                dz = 0.2
                min_mstar = 4.5

                # sample at a number of discrete redshifts
                z_vals = np.arange(sim.redshift + dz, max_z, dz)

                mpb = sim.quantMPB(sim.zoomSubhaloID, quants=[xQuant, yQuant], z_vals=z_vals)

                x_track = mpb[xQuant]
                y_track = mpb[yQuant]

                if xLog:
                    x_track = logZeroNaN(x_track)
                if yLog:
                    y_track = logZeroNaN(y_track)

                if "mstar" in yQuant:
                    # for high-res runs, show only points above 100 star particles
                    w = np.where(y_track >= min_mstar)[0]
                    x_track = x_track[w]
                    y_track = y_track[w]

                if x_track.size == 0:
                    continue

                # variable alpha, decaying towards high redshift
                alpha = np.linspace(0.6, 0.2, x_track.size)

                # plot as series of markers
                r, g, b = to_rgb(l.get_color())
                xy_c = [[r, g, b, a] for a in alpha]
                ax.scatter(x_track, y_track, marker=marker, color=xy_c, alpha=alpha, zorder=10)

                # plot as line
                # points = np.vstack((x_track, y_track)).T.reshape(-1, 1, 2)
                # segments = np.hstack((points[:-1], points[1:]))
                # lc = LineCollection(segments, array=alpha, color=l.get_color(), lw=lw_loc)
                # line = ax.add_collection(lc)

    # halos from parent box: at the same redshift as the zooms?
    sim_parent = sims[0].sP_parent.copy()

    parent_GroupFirstSub = sim_parent.halos("GroupFirstSub")
    subhaloInds = parent_GroupFirstSub[hInds]

    # load quantities at display redshift
    sim_parent_load = sim_parent.copy()
    sim_parent_load.setRedshift(sims[0].redshift)

    xvals = sim_parent_load.subhalos(xQuant)
    yvals = sim_parent_load.subhalos(yQuant)

    for i, hInd in enumerate(hInds):
        # zooms at a different redshift than the parent volume?
        subhaloInd = subhaloInds[i]

        if np.abs(sims[0].redshift - sim_parent.redshift) > 0.1:
            parent_mpb = sim_parent.loadMPB(subhaloInds[i])
            _, target_ind = closest(parent_mpb["Redshift"], sims[0].redshift)
            subhaloInd = parent_mpb["SubfindID"][target_ind]

        # final redshift point
        xval = xvals[subhaloInd]
        yval = yvals[subhaloInd]

        label = "hX in %s" % (sim_parent_load.simName) if i == 0 else ""
        (l,) = ax.plot(xval, yval, markers[len(variants) % len(markers)], color="#555", label=label)
        print(f"parent {sim_parent_load.simName} h{hInd} {subhaloInd = } {xQuant}={xval:.3f} {yQuant}={yval:.3f}")

        # time evolution tracks
        if tracks and 0 and sim.hasMergerTree:
            mpb = sim_parent_load.quantMPB(subhaloInd, quants=[xQuant, yQuant])
            ax.plot(mpb[xQuant], mpb[yQuant], "-", color=l.get_color(), alpha=0.3)

    # finish and save plot
    if f_post is not None:
        f_post(ax, sims)

    _add_legends(ax, hInds, res, variants, colors)
    fig.savefig(f"mcst_{xQuant}-vs-{yQuant}.pdf")
    plt.close(fig)


def tracks1d(
    sims: list[simParams],
    quant: str,
    xlim: list[float] = None,
    ylim: list[float] = None,
    sfh_lin: bool = False,
    sfh_treebased: bool = False,
    plot_parent: bool = True,
    sizefac: float = 1.0,
    f_selection: Callable = None,
) -> None:
    """Evolution of a quantity versus redshift.

    Designed for comparison between many zoom runs, including the target subhalo (only) from each.

    Args:
      sims (list[simParams]): list of simulation objects to compare.
      quant (str): name of quantity to plot.
      xlim (list[float][2]): if not None, override default x-axis (redshift) limits.
      ylim (list[float][2]): if not None, override default y-axis limits.
      sfh_lin (bool): show SFH with linear y-axis.
      sfh_treebased (bool): if True, use merger tree-based tracks even for SFH-related quantities.
      plot_parent (bool): if True, plot halos from the parent box for comparison.
      sizefac (float): multiplier on figure size.
      f_selection (function): if not None, this 'custom' function hook is called to determine which
        subhalo IDs to plot for each sim. It must accept a single argument: the simulation object,
        and return a list of subhalo IDs to plot. If None, defaults to sim.zoomSubhaloID only.
    """
    # quantities based on stellar formation times of stars in the final snapshot, as opposed to tree MPBs
    star_zform_quants = ["mstar2_log", "mstar_log", "mstar_tot_log", "sfr", "sfr2"]
    if sfh_treebased:
        star_zform_quants = []  # use merger tree-based tracks for all quantities

    # currently assume all sims have the same parent
    sim_parent = sims[0].sP_parent

    for sim in sims:
        assert sim.sP_parent.simName == sim_parent.simName, "All sims must have the same parent box."

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted({sim.hInd for sim in sims})
    res = sorted({sim.res for sim in sims})
    variants = sorted({sim.variant for sim in sims})

    # load helper (called both for individual zooms and parent box halos)
    def _load_sfh(sim, quant, subhaloInd, maxpts=1000, nbins_sfh=500):
        """Helper to load a SFH using stellar ages, for a single subhalo."""
        # load all (initial) stellar masses and formation times to create a high time resolution SFH
        # note: no aperture applied, so this does not equal Mstar or SFR in any aperture smaller than the whole subhalo
        star_zform = sim.snapshotSubset("stars_real", "z_form", subhaloID=subhaloInd)
        star_mass = sim.snapshotSubset("stars_real", "mass_ini", subhaloID=subhaloInd)

        if quant in ["mstar2", "mstar2_log", "sfr2"]:
            # restrict to stars within twice the stellar half mass radius for consistency
            star_rad = sim.snapshotSubset("stars_real", "rad", subhaloID=subhaloInd)
            star_rad /= sim.subhalo(subhaloInd)["SubhaloHalfmassRadType"][sim.ptNum("stars")]

            if star_rad.max() > sim.boxSize * 10:
                print(f"Warning: {sim.simName} sub {subhaloInd} has stars at large radii, skip aperture restriction.")
            else:
                w = np.where(star_rad <= 2.0)
                star_mass = star_mass[w]
                star_zform = star_zform[w]

        # sort by formation time
        sort_inds = np.argsort(star_zform)[::-1]
        star_zform = star_zform[sort_inds]
        star_mass = star_mass[sort_inds]

        if "sfr" in quant:
            # sfh (sfr vs redshift)
            star_mass = sim.units.codeMassToMsun(star_mass)
            star_tform = sim.units.redshiftToAgeFlat(star_zform)

            # bin and convert to rate
            # nbins_sfh = int(np.round(sim.units.redshiftToAgeFlat(sim.redshift) * 1e3 / 10))
            tbins = np.linspace(0.0, sim.units.redshiftToAgeFlat(sim.redshift), nbins_sfh)
            dt_Myr = (tbins[1] - tbins[0]) * 1e3  # Myr
            tbins_cen = 0.5 * (tbins[1:] + tbins[:-1])
            zbins_cen = sim.units.ageFlatToRedshift(tbins_cen)

            sfr_zbin = np.zeros_like(tbins)
            mstar_check = 0.0
            for i in range(1, tbins.size):
                w = np.where((star_tform >= tbins[i - 1]) & (star_tform < tbins[i]))
                dt = (tbins[i] - tbins[i - 1]) * 1e9  # yr
                sfr_zbin[i] = np.sum(star_mass[w]) / dt  # Msun/yr
                mstar_check += sfr_zbin[i] * dt  # Msun

            return zbins_cen, dt_Myr, logZeroNaN(sfr_zbin)  # log Msun/yr
        else:
            # cumulative stellar mass at each formation time
            star_mass = sim.units.codeMassToLogMsun(np.cumsum(star_mass))

            # coarsen to e.g. ~1000 max points to reduce size
            stride = np.max([1, int(star_zform.size / maxpts)])
            star_zform = star_zform[::stride]
            star_mass = star_mass[::stride]

        return star_zform, np.nan, star_mass

    # field metadata
    _, ylabel, yMinMax, yLog = sims[0].simSubhaloQuantity(quant)
    if ylim is not None:
        yMinMax = ylim
    if sfh_lin:
        yMinMax[0] = 0.0
        yMinMax[1] = 1.0
        ylabel = ylabel.replace("log ", "")

    xMinMax = [10.0, 2.9] if xlim is None else xlim

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_xlabel("Redshift")
    ax.set_ylabel(ylabel)

    ax.set_xlim(xMinMax)
    ax.set_ylim(yMinMax)

    if quant in star_zform_quants:
        ylabel = ylabel.replace(r"<2r_{\star},", "")  # aperture restriction on SFH not yet implemented

        if np.min(xMinMax) > 5.0:
            xx = np.array([5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
            xlabels = np.array(["5.5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"])
        else:
            xx = np.array([3, 4, 5, 6, 8, 10, 12])
            xlabels = np.array(["3", "4", "5", "6", "8", "10", "12"])

        w = np.where((xx < xMinMax[0]) & (xx >= xMinMax[1]))[0]

        ax.set_xscale("log")
        ax.set_xticks(xx[w])
        ax.set_xticklabels(xlabels[w])
        ax.xaxis.minorticks_off()

    # individual zoom runs
    for _i, sim in enumerate(sims):
        # which subhalo(s) to include?
        if f_selection is not None:
            subhaloIDs = f_selection(sim)
        else:
            subhaloIDs = [sim.zoomSubhaloID]

        # load
        vals, _, _, valLog = sim.simSubhaloQuantity(quant)

        # loop over each subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            val = vals[subhaloID]
            if valLog and not sfh_lin:
                val = logZeroNaN(val)

            # color set by hInd
            c = colors[hInds.index(sim.hInd)]

            # marker and ls set by variant
            marker = markers[variants.index(sim.variant) % len(markers)]
            linestyle = linestyles[variants.index(sim.variant) % len(linestyles)]

            # marker size set by resolution
            ms_loc = (sim.res - 10) * 2.5 + 3
            lw_loc = lw  # (sim.res - 10) if len(res) > 1 else lw
            alpha_loc = 1.0

            # if only one hInd, then use color for either variant or res
            if len(hInds) == 1:
                marker = markers[0]
                linestyle = linestyles[0]

                if len(variants) > 1 and len(res) == 1:
                    c = colors[variants.index(sim.variant) % len(colors)]
                if len(res) > 1 and len(variants) == 1:
                    c = colors[res.index(sim.res) % len(colors)]
                if len(res) > 1 and len(variants) > 1:
                    c = colors[res.index(sim.res)]
                    linestyle = linestyles[variants.index(sim.variant) % len(linestyles)]

                if len(subhaloIDs) > 1 and len(variants) <= 2:
                    linestyle = linestyles[np.min([j, 1])]
            else:
                # more than one hInd, additional subhalos are faint
                if j > 0:
                    alpha_loc = 0.4
                    lw_loc = lw - 1

            # final redshift marker
            if len(hInds) == 1 or j == 0:
                (l,) = ax.plot(sim.redshift, val, marker, color=c, markersize=ms_loc, alpha=alpha_loc, label="")

            # time track
            if quant in star_zform_quants:
                # special case: stellar mass growth or SFH
                if sim.subhalo(subhaloID)["SubhaloLenType"][sim.ptNum("stars")] == 0:
                    print(f"[{sim}] no stars in {subhaloID = }, skipping [{quant}].")
                    continue

                star_zform, dt_Myr, star_mass = _load_sfh(sim, quant, subhaloID)
                ax.set_ylabel(ylabel.replace("instant", r"\Delta t = %.1f Myr" % dt_Myr))
                if sfh_lin:
                    star_mass = 10.0**star_mass

                w = np.where(star_zform < xMinMax[0])
                ax.plot(star_zform[w], star_mass[w], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)

                # extend to symbol
                if len(w[0]) > 0:
                    x = star_zform[w][-1]
                    y = star_mass[w][-1]
                    ax.plot([x, sim.redshift], [y, y], ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=0.2)
            else:
                # general case
                mpb = sim.quantMPB(subhaloID, quants=[quant])
                vals_track = mpb[quant]
                if valLog and not sfh_lin:
                    vals_track = logZeroNaN(vals_track)

                ax.plot(mpb["z"], vals_track, ls=linestyle, lw=lw_loc, color=l.get_color(), alpha=alpha_loc)

    # galaxies from parent box
    vals = sim_parent.subhalos(quant)
    parent_GroupFirstSub = sim_parent.halos("GroupFirstSub")

    for i, hInd in enumerate(hInds):
        # load
        if not plot_parent:
            continue

        subhaloInd = parent_GroupFirstSub[hInd]
        val = vals[subhaloInd]

        label = "hX in %s" % (sim_parent.simName) if i == 0 else ""
        if len(hInds) == 1:
            label = "h%d in %s" % (hInds[0], sim_parent.simName)

        # final redshift marker
        if sim_parent.redshift >= xMinMax[1]:
            (l,) = ax.plot(sim_parent.redshift, val, markers[3], color="#555", label=label)

        # time track
        if quant in star_zform_quants:
            # special case: stellar mass growth or SFH
            if sim_parent.subhalo(subhaloInd)["SubhaloLenType"][sim.ptNum("stars")] == 0:
                print(f"[{sim}] no stars in {subhaloInd = }, skipping [{quant}].")
                continue

            star_zform, _, star_mass = _load_sfh(sim_parent, quant, subhaloInd)

            w = np.where((star_zform >= 0.0) & (star_zform < xMinMax[0]))
            ax.plot(star_zform[w], star_mass[w], "-", color="#555", alpha=1.0, label=label)
        else:
            # general case
            pass
            # mpb = sim_parent.quantMPB(subhaloInd, quants=[quant])
            # ax.plot(mpb['z'], mpb[quant], ls=linestyle, color='#555', alpha=1.0, label=label)

    # finish and save plot
    _add_legends(ax, hInds, res, variants, colors, lineplot=True)
    hStr = "" if len(set(hInds)) > 1 else "_h%d" % hInds[0]
    tStr = "_tree" if sfh_treebased else ""
    fig.savefig(f"mcst_{quant}-vs-redshift{hStr}{tStr}.pdf")
    plt.close(fig)
