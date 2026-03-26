"""
MCST: stellar clusters paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from temet.plot import snapshot, subhalos_evo
from temet.plot.config import colors, figsize, markers
from temet.plot.util import tableau10_colors
from temet.util import simParams
from temet.util.helper import logZeroNaN

from .mcst import _get_existing_sims, _zoomSubhaloIDsToPlot, phase_diagram
from .mcst_vis import vis_gallery_clusters


mass_label = r"Star Cluster Mass [ log M$_{\odot}$ ]"
size_label = r"Star Cluster Size $R_{1/2}$ [ pc ]"  # always show tick labels in linear

mass_lim = [1.5, 4.0]  # log msun
size_lim = [-1.8, 0.8]  # log pc [-0.5, 1.8] for L15
sigma_lim = [0.5, 6.0]  # log msun/pc^2

sizeticks_lin = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  # pc


def _starClusterSubhaloIDs(sim):
    """Define a common rule for which subhalo(s) are identified as star clusters."""
    # all satellites of the first halo with non-zero stellar mass
    halo_id = sim.subhalos("halo_id")
    cen_flag = sim.subhalos("cen_flag")
    mstar = sim.subhalos("mstar_tot")
    # mdm = sim.subhalos("mdm_tot")

    subhaloIDs = np.where((halo_id == 0) & (cen_flag == 0) & (mstar > 0))[0]  # mdm == 0

    print(f"[{sim}] Showing {len(subhaloIDs)} subhalos.")

    return subhaloIDs


def diagnostic_sfr_jeans_mass(sims, haloID=0):
    """Check that the per-cell Jeans mass is being calculated correctly during the simulation.

    Load all gas properties, convert to proper, calculate the jeans mass and
    cell diameter yourself, calculate SFR yourself, plot against what the code is reporting
    (what is in the snap), should be 1-to-1, if not may be a factor of a or h missing.
    """
    # AREPO/SFR_MCS calculation:
    # dens = SphP[i].Density;
    # Sfr = 0.0;

    # /* Used for only SF when local Jeans mass < All.SfrCritJeansMassN * mcell */
    # All.SfrCritFactor  = pow(GAMMA, 1.5) * pow(M_PI, 2.5) / (6.0 * pow(All.G, 1.5) * All.SfrCritJeansMassN);

    # if((P[i].Mass * dens * dens * sqrt(All.cf_a3inv) / pow(SphP[i].Pressure, 1.5)) < All.SfrCritFactor)
    #   continue;

    # All.cf_a3inv    = 1 / (All.Time * All.Time * All.Time);
    # All.G = GRAVITY / pow(All.UnitLength_in_cm, 3) * All.UnitMass_in_g * pow(All.UnitTime_in_s, 2);
    # so All.G is G in code units (no cosmo factors)
    # t_ff = sqrt(3.0 * M_PI / (32.0 * All.G * dens * All.cf_a3inv)); # code time

    # Sfr = All.SfrEfficiency * P[i].Mass / t_ff; # [code mass/code time]
    # SphP[i].Sfr *= (All.UnitMass_in_g / SOLAR_MASS) / (All.UnitTime_in_s / SEC_PER_YEAR); # msun/yr units

    for sim in sims:
        # parameters
        print(sim)

        eps_sf = sim.params["SfrEfficiency"]
        N_J = sim.params["SfrCritJeansMassN"]
        N_J_crit = sim.params["SfrForceJeansMassN"]  # all these runs have SFR_MCS_FORCE==0

        # load
        M_J = sim.snapshotSubset("gas", "mjeans", haloID=haloID)  # msun

        mass = sim.snapshotSubset("gas", "mass_msun", haloID=haloID)  # msun
        dens = sim.snapshotSubset("gas", "dens", haloID=haloID)  # code
        dens = sim.units.codeDensToPhys(dens, cgs=True, numDens=True)  # physical [1/cm^3]

        rad_rvir = sim.snapshotSubset("gas", "rad_rvir", haloID=haloID)

        tff = sim.snapshotSubset("gas", "tff_local", haloID=haloID) / 1e6  # yr

        # calculate SFR that we would expect
        sfr_calc = np.zeros(mass.size, dtype="float32")

        w = np.where(M_J < N_J * mass)[0]
        sfr_calc[w] = eps_sf * (mass[w] / tff[w])  # msun/yr

        if "SFR_MCS_FORCE" in sim.config:
            assert sim.config["SFR_MCS_FORCE"] == 0  # set efficiency to 1.0
            w = np.where(M_J < N_J_crit * mass)[0]
            sfr_calc[w] = 1.0 * (mass[w] / tff[w])

        # sfr_calc = eps_sf * (mass / tff) # msun/yr

        # if M_J > N_J * m_cell, then SFR = 0 (handled above)
        # ww = np.where(M_J > N_J * mass)[0]
        # frac = len(ww) / len(mass)
        # print('Have [%d/%d] cells (%.2f%%) with M_J > N_J*m_cell (not star-forming).' % (len(ww),len(mass),frac*100))
        # sfr_calc[ww] = 0.0

        # compare to SFR in snapshot
        sfr_snap = sim.snapshotSubset("gas", "sfr", haloID=haloID)  # msun/yr

        print("Number of SFRs>0: snap = [%d], calc = [%d]" % (np.count_nonzero(sfr_snap), np.count_nonzero(sfr_calc)))

        w1 = np.where(sfr_calc == 0)[0]
        w2 = np.where(sfr_snap == 0)[0]
        print("Entries that are zero agree: ", np.array_equal(w1, w2))

        w3 = np.where(sfr_calc > 0)[0]

        if len(w3) > 0:
            diff = sfr_calc[w3] - sfr_snap[w3]
            ratio = sfr_calc[w3] / sfr_snap[w3]
            print("SFR diff calc vs snap: min = %g, max = %g, mean = %g" % (diff.min(), diff.max(), diff.mean()))
            print("SFR ratio calc vs snap: min = %g, max = %g, mean = %g" % (ratio.min(), ratio.max(), ratio.mean()))
        print("All close: ", np.allclose(sfr_calc, sfr_snap))
        print("All non-zero close: ", np.allclose(sfr_calc[w3], sfr_snap[w3]))

        # plot
        fig, ax = plt.subplots()
        ax.set_yscale("log")
        ax.set_xlabel("log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )")
        ax.set_ylabel("N")

        for rad_cut in [1.0, 0.1, 0.02]:
            # select
            w_rad = np.where(rad_rvir < rad_cut)
            M_J_loc = M_J[w_rad]
            mass_loc = mass[w_rad]

            # calc
            N_J_realized = M_J_loc / mass_loc

            w_N1 = np.where(N_J_realized < 1.0)
            w_N8 = np.where(N_J_realized < 8.0)
            frac_N1 = np.sum(mass_loc[w_N1]) / np.sum(mass_loc)
            frac_N8 = np.sum(mass_loc[w_N8]) / np.sum(mass_loc)

            N_J_realized = np.log10(N_J_realized)

            # plot hist
            label = "(r/r200 < %.2f) frac$_{<1}$: %.3f, frac$_{<8}$: %.3f" % (rad_cut, frac_N1, frac_N8)
            ax.hist(N_J_realized, bins=100, histtype="step", label=label)

        # select dens
        if 1:
            dens_cut = 1.0
            w_dens = np.where(dens > dens_cut)
            M_J_loc = M_J[w_dens]
            mass_loc = mass[w_dens]

            # calc
            N_J_realized = M_J_loc / mass_loc

            w_N1 = np.where(N_J_realized < 1.0)
            w_N8 = np.where(N_J_realized < 8.0)
            frac_N1 = np.sum(mass_loc[w_N1]) / np.sum(mass_loc)
            frac_N8 = np.sum(mass_loc[w_N8]) / np.sum(mass_loc)

            N_J_realized = np.log10(N_J_realized)

            # plot hist
            label = "(dens > %.1f) frac$_{<1}$: %.3f, frac$_{<8}$: %.3f" % (dens_cut, frac_N1, frac_N8)
            ax.hist(N_J_realized, bins=100, histtype="step", label=label)

        ax.plot(np.log10([1.0, 1.0]), [0, np.max(ax.get_ylim()) * 0.6], "-", color="black", alpha=0.3)
        ax.plot(np.log10([8.0, 8.0]), [0, np.max(ax.get_ylim()) * 0.6], "-", color="black", alpha=0.3)

        ax.legend(loc="best")
        fig.savefig("mjeans_%s.pdf" % sim)
        plt.close(fig)

    # plot cumulative fraction of mass with N_J > x
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 7))
    ylim = [1e-6, 1e-0]
    ax1.set_ylim(ylim)
    ax1.set_xlim([-2, 2])
    ax1.set_yscale("log")
    ax1.set_xlabel("log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )")
    ax1.set_ylabel("Fraction of Mass with N$_{\\rm J}$ < x-axis")

    ax1.plot(np.log10([1.0, 1.0]), ylim, "-", color="black", alpha=0.3)
    ax1.plot(np.log10([8.0, 8.0]), ylim, "-", color="black", alpha=0.3)

    for sim in sims:
        # load
        M_J = sim.snapshotSubset("gas", "mjeans", haloID=haloID)  # msun
        mass = sim.snapshotSubset("gas", "mass_msun", haloID=haloID)  # msun

        # rad_rvir = sim.snapshotSubset('gas', 'rad_rvir', haloID=haloID) # little impact
        # w = np.where(rad_rvir < 1.0)
        # M_J = M_J[w]
        # mass = mass[w]

        # calc
        N_J_realized = M_J / mass

        inds = np.argsort(N_J_realized)
        N_J_realized = N_J_realized[inds]
        mass = mass[inds]

        cum_mass = np.cumsum(mass)
        cum_mass /= np.sum(mass)

        ax1.plot(np.log10(N_J_realized), cum_mass, "-", label=sim)

    ax1.legend(loc="lower right")

    # plot cumulative mass
    ylim = [1e2, 1e8]
    ax2.set_ylim(ylim)
    ax2.set_yscale("log")
    ax2.set_xlim([-2, 2])
    ax2.set_xlabel("log( M$_{\\rm Jeans}$ / M$_{\\rm cell}$ )")
    ax2.set_ylabel("Total Gas Mass with N$_{\\rm J}$ < x-axis [M$_{\\rm sun}$]")

    ax2.plot(np.log10([1.0, 1.0]), ylim, "-", color="black", alpha=0.3)
    ax2.plot(np.log10([8.0, 8.0]), ylim, "-", color="black", alpha=0.3)

    for sim in sims:
        # load
        M_J = sim.snapshotSubset("gas", "mjeans", haloID=haloID)  # msun
        mass = sim.snapshotSubset("gas", "mass_msun", haloID=haloID)  # msun

        # rad_rvir = sim.snapshotSubset('gas', 'rad_rvir', haloID=haloID)
        # w = np.where(rad_rvir < 1.0)
        # M_J = M_J[w]
        # mass = mass[w]

        # calc
        N_J_realized = M_J / mass

        inds = np.argsort(N_J_realized)
        N_J_realized = N_J_realized[inds]
        mass = mass[inds]

        cum_mass = np.cumsum(mass)

        ax2.plot(np.log10(N_J_realized), cum_mass, "-", label=sim)

    ax2.legend(loc="lower right")

    fig.savefig("mjeans_cumsum_n%d_z%d.pdf" % (len(sims), sims[0].redshift))
    plt.close(fig)


def stellar_remnants(sims, haloID=0, sizefac=0.8):
    """Plot the mass distribution of different stellar remnant types (WD, NS, BH)."""
    # load
    remnant_type = []
    remnant_mass = []

    for sim in sims:
        if sim.star != 3:
            print(f" note: [{sim}] does not have solo stars, skipping.")
            continue

        loc_type = sim.stars("remnant_type", haloID=haloID)
        loc_mass = sim.stars("remnant_mass", haloID=haloID)
        remnant_type = np.hstack((remnant_type, loc_type))
        remnant_mass = np.hstack((remnant_mass, loc_mass))

    # type_names = {0: "none", 1: "white dwarf", 2: "neutron star", 3: "black hole"}
    type_names = {1: "white dwarf", 2: "neutron star", 3: "black hole"}

    # config
    x_split = 2.0  # where x-axis is split
    xmin = 0.5
    xmax = 100

    assert np.nanmax(remnant_mass) < xmax, "Need to increase xmax to show all remnants (BHs)."

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))
    ax.set_ylabel("Number")
    ax.set_yscale("log")

    # double x-axis: left and right sides of the plot have different ranges (and could be log/linear)
    ax.set_xlim([xmin, x_split])  # to show WDs and NSs
    ax.set_xscale("linear")
    ax.spines["right"].set_visible(False)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("right", size=5.8, pad=0, sharey=ax)
    ax2.set_xlim([x_split, xmax])  # to show BHs
    ax2.set_xscale("linear")

    ax.set_xlabel(" ")
    ax2.set_xlabel(r"Remnant Mass [ M$_{\odot}$ ]")
    ax2.xaxis.set_label_coords(0.35, -0.07)

    ax2.spines["left"].set_visible(False)
    ax2.spines["left"].set_linewidth(5)

    ax2.yaxis.set_ticks_position("right")
    ax2.tick_params(axis="y", which="both", labelright=False)

    h_max = 0.0

    for type_num, type_name in type_names.items():
        w = np.where(remnant_type == type_num)[0]
        n_frac = len(w) / len(remnant_type) * 100
        m_frac = remnant_mass[w].sum() / np.nansum(remnant_mass) * 100
        label = f"{type_name} (N = {n_frac:.1f}%, mass = {m_frac:.1f}%)"

        # right panel (for BHs)
        nbins = int((xmax - x_split) / 2.0)
        h = np.histogram(remnant_mass[w], bins=nbins, range=[0, xmax])  # [0.5, 2.0] to show WDs and NSs
        ax2.stairs(*h, fill=True, label=label)

        h_max = np.max([h_max, h[0].max()])

        # print(type_name, len(w), remnant_mass[w].min(), remnant_mass[w].max())

        # left panel (for WDs and NSs)
        nbins = int((x_split - xmin) / 0.1)
        h = np.histogram(remnant_mass[w], bins=nbins, range=[xmin, x_split])
        ax.stairs(*h, fill=True)

        h_max = np.max([h_max, h[0].max()])

    # fix ylim
    ylim = [2, h_max * 1.2]
    ax.set_ylim(ylim)
    ax2.set_ylim(ylim)

    # finish plot
    ax2.plot([x_split, x_split], ax.get_ylim(), ":", color="black", alpha=0.3, zorder=1)

    ax2.legend(loc="best")

    if len(sims) == 1:
        saveFilename = "stellar_remnants_{sim.simName}_{sim.snap}.pdf"
    else:
        saveFilename = "stellar_remnants_n%d.pdf" % (len(sims))
    fig.savefig(saveFilename)
    plt.close(fig)


def star_cluster_histogram(sims, quant, sizefac=1.0):
    """Plot the mass function, or size distribution, of star clusters."""
    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_ylabel("Number of Star Clusters")
    ax.set_yscale("log")

    if quant == "mass":
        xlim = mass_lim
        xlabel = mass_label

    if quant == "size":
        xlim = size_lim
        xlabel = size_label

    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)

    for sim in sims:
        # make selection
        subIDs = _starClusterSubhaloIDs(sim)

        # histogram quantity
        if quant == "mass":
            mstar = sim.subhalos("mstar_tot")
            h_quant = np.log10(mstar[subIDs])
        if quant == "size":
            rhalf = sim.subhalos("size_stars_pc")  # pc
            h_quant = np.log10(rhalf[subIDs])

        label = f"h{sim.hInd}"  # f"{sim.simName} (N={len(subIDs)}/{sim.numSubhalos})"
        label = sim.simName  # while we are studying the res dependence

        h = np.histogram(h_quant, bins=30, range=xlim)  # range=[min,max]
        ax.stairs(*h, fill=True, alpha=0.8, label=label)

        if quant == "mass":
            min_mass = np.log10(20 * sim.units.codeMassToMsun(sim.targetGasMass))
            ax.plot([min_mass, min_mass], ax.get_ylim(), color="black", linestyle=":", alpha=0.5)

        if quant == "size":
            grav_soft_stars = {13: 0.0244, 14: 0.0122, 15: 0.0061, 16: 0.003}
            grav_soft_code = grav_soft_stars[sim.res]
            grav_soft_logpc = np.log10(sim.units.codeLengthToPc(grav_soft_code))
            ax.plot([grav_soft_logpc, grav_soft_logpc], ax.get_ylim(), color="black", linestyle=":", alpha=0.5)

    if quant == "size":
        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_xticks(sizeticks_log)
        ax.set_xticklabels(sizeticks_lin)

        # todo: overplot Brown+21 data

    ax.legend(loc="upper right")

    fig.savefig(f"star_cluster_histo_{quant}.pdf")
    plt.close(fig)


def size_vs_mass(sims: list[simParams]) -> None:
    """Size-mass relation for star clusters."""
    xQuant = "mstar_tot"
    yQuant = "rhalf_stars_pc"
    xlim = mass_lim
    ylim = size_lim

    def _f_pre(ax, sims):
        # set axis label for mass
        ax.set_xlabel(mass_label)

        # set custom tick labels for size
        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_yticks(sizeticks_log)
        ax.set_yticklabels(sizeticks_lin)

    def _draw_fitline(ax, sims, **kwargs):
        # fit line to x,y using np.polyfit
        x_all = np.array(kwargs["x"])
        y_all = np.array(kwargs["y"])
        w = np.where((x_all > xlim[0]) & (x_all < xlim[1]) & (y_all > ylim[0]) & (y_all < ylim[1]))[0]

        coeffs = np.polyfit(x_all[w], y_all[w], deg=1)

        # overplot best-fit line
        x_fit = np.array(xlim) + [0.2, -0.2]
        y_fit = np.polyval(coeffs, x_fit)

        ax.plot(x_fit, y_fit, "-", color="#555", lw=3, alpha=0.5)

        # todo: Brown+21 data (https://arxiv.org/abs/2106.12420)
        # https://gillenbrown.com/LEGUS-sizes/

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        tracks=False,
        sizefac=0.8,
        markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        legend="simple",
        legend_locs=["lower right", "upper left"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_draw_fitline,
        f_selection=_starClusterSubhaloIDs,
    )


def sigma_star_galaxies(sims: list[simParams]) -> None:
    """The stellar mass surface density (Sigma_*) as a function of galaxy mass."""
    yQuant = "surfdens_stars"
    xQuant = "mstar2_log"

    ylim = [5.5, 12.0]  # log msun/kpc^2
    xlim = [4.5, 9.0]  # log mstar

    def _draw_data(ax, sims):
        from temet.load.data import claeyssens23
        # no: https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.343G/abstract (only M* > 9)

        # Claeyssens+23 (JWST): https://arxiv.org/abs/2208.10450 ("clumps")
        c23 = claeyssens23()
        w = np.where(c23["z"] >= 5.0)

        ax.errorbar(
            c23["mstar"][w],
            c23["sigma"][w],
            xerr=[c23["mstar_err1"][w], c23["mstar_err2"][w]],
            yerr=[c23["sigma_err1"][w], c23["sigma_err2"][w]],
            fmt="o",
            color="#555",
            alpha=0.7,
            label=c23["label"] + " ($z > 5$)",
        )

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        tracks=True,
        legend="simple",
        legend_locs=["upper left", "upper right"],
        legend_ncols=[1, 1],
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def sigma_star_clusters(sims: list[simParams], vs_size=False) -> None:
    """The stellar mass surface density (Sigma_*) as a function of galaxy mass."""
    yQuant = "surfdens_stars_pc"
    xQuant = "mstar_tot_log"

    ylim = sigma_lim  # log msun/pc^2
    xlim = mass_lim  # log mstar

    if vs_size:
        xQuant = "rhalf_stars_pc"
        xlim = size_lim  # log pc

    def _f_pre(ax, sims):
        # set custom tick labels for size
        if vs_size:
            sizeticks_log = np.log10(sizeticks_lin)
            ax.set_xticks(sizeticks_log)
            ax.set_xticklabels(sizeticks_lin)

        # draw observational data
        # TODO: https://arxiv.org/abs/2401.03224 (Fig 2, Sigma* vs size)

        # todo: Brown+21 data (https://arxiv.org/abs/2106.12420)

        # draw other sim data
        # TODO: (van Donkelaar+26 Fig 4)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        tracks=False,
        markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        legend="simple",
        legend_locs=["upper left", "upper right"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_selection=_starClusterSubhaloIDs,
    )


def gas_phase_fractions(sims, frac_type="Mass"):
    """Create a bar chart showing the fractions of gas in different phases (multi-phase ISM)."""
    # config
    subhaloID = 0

    ylim = [1e-4, 5.0]

    acFieldBase = f"Subhalo_{frac_type}_r015rvir_Gas_"  # r < 0.15rvir is the spatial definition for the 'galaxy'/'ISM'

    phases = ["Total", "Hot", "Warm_Ionized", "Warm_Neutral", "Cool", "Cold"]  # "Warm"
    colors = [tableau10_colors[n] for n in ["gray", "red", "orange", "orange_dark", "lightblue", "blue"]]

    # start plot
    fig, ax = plt.subplots(figsize=(figsize[0] * 1.2, figsize[1] * 0.5))
    ax.set_ylabel(f"ISM {frac_type} Fraction")
    ax.set_yscale("log")
    ax.set_ylim(ylim)

    ax.set_xlim([-0.5, len(sims) + 0.5])
    ax.set_xticks(range(len(sims) + 1))
    ax.set_xticklabels(["stack"] + [f"h{sim.hInd}" for sim in sims])

    # build dataset
    assert phases[0] == "Total"
    data = {}

    for phase in phases:
        data[phase] = np.zeros(len(sims) + 1, dtype="float32")  # +1 for stack

    # loop over simulations, load fractions (auxCat)
    for i, sim in enumerate(sims):
        for phase in phases:
            phase_fracs = sim.auxCat(f"{acFieldBase}{phase}")[f"{acFieldBase}{phase}"]
            frac = np.clip(phase_fracs[subhaloID], 1e-10, np.inf)  # avoid zero if absolutely no gas at all

            # normalize by total to get fractions
            if phase != "Total":
                frac /= data["Total"][i + 1]

            data[phase][i + 1] = frac

    # calculate stack
    for phase in phases:
        data[phase][0] = np.mean(data[phase][1:])

        # clip to make all visible
        data[phase] = np.clip(data[phase], ylim[0] * 1.2, np.inf)

    # plot barchart
    width = 1 / len(phases)  # * 1.0

    x = np.arange(len(sims) + 1) - width * (len(phases) / 2)  # center label locations

    for i, (phase_name, phase_fracs) in enumerate(data.items()):
        if i == 0:
            continue  # skip Total

        label = phase_name.replace("_", " ")
        ax.bar(x + (width * i), phase_fracs, width, color=colors[i], label=label)

    # finish plot
    ax.plot([0.5, 0.5], ax.get_ylim(), ":", color="black", alpha=0.3)
    ax.legend(loc="upper right", ncols=len(phases))

    if len(sims) == 1:
        saveFilename = f"gas_phase_fracs_{frac_type}_{sim.simName}_{sim.snap}.pdf"
    else:
        saveFilename = f"gas_phase_fracs_{frac_type}_n{len(sims)}.pdf"

    fig.savefig(saveFilename)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------


def paperPlots(a=False):
    """Plots for MCST intro paper. (if a == True, make all figures)."""
    # list of sims to include
    variants = ["ST15"]  # ['ST15c','ST15m','ST15s']
    res = [14, 15, 16]
    hInds = [1958, 5072, 15581, 23908, 31619, 73172, 219612, 311384, 446076, 539722, 844537]
    redshift = 5.5

    # if (all == False), only dz < 0.1 matches
    # if (single == True), only the highest available res of each halo
    # sims = _get_existing_sims(variants, res, hInds, redshift, all=False, single=True)

    sims = []
    # sims.append(simParams("structures", hInd=1958, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=5072, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=15581, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=23908, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=31619, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=73172, res=14, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=446076, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=539722, res=16, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=844537, res=16, variant="ST15", redshift=5.5))

    # ------------

    # fig 1a: clusters: mass and size distributions
    if 1 or a:
        # star_cluster_histogram(sims, quant="mass")
        star_cluster_histogram(sims, quant="size", sizefac=0.8)

    # fig 1b: clusters: size-mass relation
    if 0 or a:
        size_vs_mass(sims)

    # fig 2: gallery of clusters (~10pc stamps?)
    if 0 or a:
        # exploration to find nice clusters:
        sims = []

        if 0:
            sim = simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5)

            mstar = sim.subhalos("mstar_tot")
            halo_id = sim.subhalos("halo_id")
            rhalf = sim.subhalos("size_stars_pc")
            # subIDs = np.where((mstar > 1e3) & (rhalf < 0.9) & (rhalf > 0.1) & (halo_id == 0))[0]
            subIDs = np.where((mstar > 1e3) & (rhalf > 1.1) & (halo_id == 0))[0]
            print(subIDs)

            for i, subID in enumerate(subIDs[:12]):  # limit to twelve clusters
                sim_loc = sim.copy()
                sim_loc.subhaloInd = subID
                sims.append(sim_loc)
                print(i, subID, mstar[subID], rhalf[subID])

        else:
            # final set for paper
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5, subhaloInd=29))
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5, subhaloInd=115))
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5, subhaloInd=172))
            sims.append(simParams("structures", hInd=23908, res=16, variant="ST15", redshift=15.0, subhaloInd=3))
            sims.append(simParams("structures", hInd=446076, res=16, variant="ST15", redshift=5.5, subhaloInd=1))
            sims.append(simParams("structures", hInd=446076, res=16, variant="ST15", redshift=5.5, subhaloInd=13))
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5, subhaloInd=200))
            sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=6.0, subhaloInd=113))
            sims.append(simParams("structures", hInd=539722, res=16, variant="ST15", redshift=5.8, subhaloInd=45))
            sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=6.0, subhaloInd=12))
            sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=6.5, subhaloInd=29))
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5, subhaloInd=90))

        vis_gallery_clusters(sims)

    # fig X: galaxies: stellar surface density (Sigma_*)
    if 0 or a:
        sigma_star_galaxies(sims)

    # fig: stellar surface density of clusters (Sigma_*)
    if 0 or a:
        sigma_star_clusters(sims)
        sigma_star_clusters(sims, vs_size=True)

    # fig 3: pressure vs. rho phase space diagram (see Schaye+26 Colibre Fig 8)
    if 0 or a:
        xlim = [-6.0, 5.0]
        sim = simParams("structures", hInd=31619, res=15, variant="ST15", redshift=5.5)
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant=None, ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="rad_rvir", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="vrad", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="csnd", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="Z_solar", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="tff_local", ext="pdf")

    # fig 4: kennicut-schmidt relation (global)
    # fig todo: Sigma_SFR (e.g. Ceverino+26 shows JWST data comparisons)

    # fig 5: kennicut-schmidt relation (local/spatially resolved)

    # fig 6: gas mass fraction of ISM gas in different phases, at e.g. z=10 and z=6 (bar plots?)
    if 0 or a:
        gas_phase_fractions(sims, frac_type="Mass")
        gas_phase_fractions(sims, frac_type="Vol")

    # fig todo: radius and radial velocity at formation time (use birth values of member stars?)

    # fig todo: gas fraction vs M* (van Donkelaar+26 Fig 3)

    # vis todo: time evolution from pre-birth to post-birth (gas dens, vrad, tff_local, Q, stars, ...)

    # fig todo: quantitative assessment of reason for formation (e.g. self-grav instability, compression, ...)

    # fig todo: any cluster population stat, e.g. mass func slope, size-mass slope, vs. halo mass (color by redshift)
    #  "universality" or not?

    # fig todo: histogram of cluster formation redshifts (with respect to important events/starbursts/mergers)

    # fig todo: ages, i.e. histograms of member star ages (matched to vis gallery), or age vs. X scaling relations

    # fig: stellar remnants: mass distribution
    if 0 or a:
        stellar_remnants(sims)

    # radial profiles - halo comparisons
    if 0 or a:
        haloIDs = [0] * len(sims)  # assume first
        opts = {"haloIDs": haloIDs, "xlog": True, "xlim": [-2.0, 1.5], "ylog": True}

        snapshot.profile(sims, ptType="gas", ptProperty="numdens", ylim=[-4.5, 4.0], scope="global", **opts)

        snapshot.profile(sims, ptType="stars", ptProperty="dens", ylim=[2.5, 11.0], scope="global", **opts)

        snapshot.profile(sims, ptType="gas", ptProperty="cellsize_kpc", ylim=[-3.5, -0.5], scope="global", **opts)

    # radial profiles: 2d vs time
    if 0 or a:
        # evo
        opts = {"haloID": 0, "max_z": 10.0, "rlog": True, "rlim": [-2.0, 1.0], "smooth_sizes": True}

        for sim in sims:
            snapshot.profilesStacked2d(
                sim,
                ptType="gas",
                ptProperty="numdens",
                clim=[-2.0, 3.0],
                clog=True,
                scope="global",
                ctName="magma",
                **opts,
            )

            snapshot.profilesStacked2d(
                sim,
                ptType="gas",
                ptProperty="vrad",
                clim=[-50.0, 50.0],
                clog=False,
                scope="global",
                ctName="curl",
                **opts,
            )

            snapshot.profilesStacked2d(
                sim,
                ptType="gas",
                ptProperty="tff_local",
                clim=[-2.0, 2.0],
                clog=True,
                scope="global",
                ctName="inferno",
                **opts,
            )

            snapshot.profilesStacked2d(
                sim,
                ptType="stars",
                ptProperty="dens",
                clim=[3.0, 10.0],
                clog=True,
                scope="global",
                ctName="magma",
                **opts,
            )

            snapshot.profilesStacked2d(
                sim,
                ptType="gas",
                ptProperty="temp",
                clim=[3.0, 6.0],
                clog=True,
                scope="global",
                ctName="thermal",
                **opts,
            )

            snapshot.profilesStacked2d(
                sim,
                ptType="gas",
                ptProperty="menc_vesc",
                clim=[0.0, 1.7],
                clog=True,
                scope="fof",
                ctName="afmhot",
                **opts,
            )

    # diagnostic: mass fraction of PartType4 in stellar remnants
    if 0 or a:
        # note: only L15/L16 since require solo stars for remnant_type calculation
        subhalos_evo.scatter2d(
            sims,
            xQuant="mhalo_200_log",
            yQuant="remnant_massfrac",
            xlim=[7.0, 10.5],
            ylim=[0, 0.5],
            vs_sim=None,
            tracks=False,
            parents=False,
            legend="simple",
            legend_ncols=[1, 1],
            legend_locs=["upper right", "upper right"],
            f_selection=_zoomSubhaloIDsToPlot,
            sizefac=0.8,
        )

    # diagnostic: SFR debug
    if 0:
        diagnostic_sfr_jeans_mass(sims, haloID=0)
