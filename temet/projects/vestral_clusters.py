"""
VESTRAL: stellar clusters paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import make_interp_spline

from temet.load.groupcat import catalog_field
from temet.plot import snapshot, subhalos_evo
from temet.plot.config import colors, figsize, linestyles
from temet.plot.util import colored_line, tableau10_colors
from temet.util import simParams
from temet.util.helper import running_median
from temet.vis.halo import renderSingleHalo

from .vestral import _zoomSubhaloIDsToPlot, phase_diagram
from .vestral_vis import vis_gallery_clusters, vis_single_galaxy


mass_label = r"Star Cluster Mass [ log M$_{\odot}$ ]"
size_label = r"Star Cluster Size $R_{1/2}$ [ pc ]"  # always show tick labels in linear
age_label = r"Star Cluster Age [ Myr ]"

mass_lim = [1.5, 4.0]  # log msun
size_lim = [-1.8, 1.0]  # log pc, [-0.5, 1.8] for L15
sigma_lim = [0.0, 6.0]  # log msun/pc^2, [0.5, 4.0] for L15
age_lim = [-0.6, 3.0]  # log Myr

sizeticks_lin = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]  # pc
ageticks_lin = [0.25, 1, 2.5, 10, 25, 50, 100, 250, 500, 1000]  # Myr


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


def star_cluster_histogram(sims, quant, nbins=30, sizefac=1.0):
    """Plot the mass function, or size distribution, of star clusters."""
    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_ylabel("Number of Star Clusters")
    ax.set_yscale("log")

    if quant == "mass":
        xlim = mass_lim
        xlabel = mass_label

    if quant == "size":
        xlim = [size_lim[0], size_lim[1] + 0.4]
        xlabel = size_label

    if quant == "age":
        xlim = age_lim
        xlabel = age_label

    ax.set_xlabel(xlabel)
    ax.set_xlim(xlim)

    ymax = 0
    vals = []

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
        if quant == "age":
            age = sim.subhalos("stellarage_myr")
            h_quant = np.log10(age[subIDs])

        label = f"h{sim.hInd}"  # f"{sim.simName} (N={len(subIDs)}/{sim.numSubhalos})"
        # label = sim.simName  # while we are studying the res dependence

        h = np.histogram(h_quant, bins=nbins, range=xlim)
        ax.stairs(*h, fill=True, alpha=0.8, label=label)

        ymax = max(ymax, h[0].max())
        vals = np.hstack((vals, h_quant))

    # overplot stack
    h = np.histogram(vals, bins=nbins, range=xlim)
    h = [h[0] / len(sims), h[1]]  # mean
    lw = 3.0 if sizefac == 1 else 2.5
    ax.stairs(*h, fill=False, edgecolor="#000", lw=lw, alpha=0.8, label="Stack")

    # overplot evolutionary model applied to stack
    if quant in ["mass", "size"]:
        masses_ini = []
        sizes_ini = []
        masses_evo = []
        sizes_evo = []

        for sim in sims:
            # load
            subIDs = _starClusterSubhaloIDs(sim)
            mstar = sim.subhalos("mstar_tot")[subIDs]  # Msun
            rhalf = sim.subhalos("size_stars_pc")[subIDs]  # pc
            age = sim.subhalos("stellarage_myr")[subIDs]

            mstar_ini = sim.subhalos("mstar_max")[subIDs]  # Msun
            rhalf_ini = sim.subhalos("size_stars_max")[subIDs]  # pc

            masses_ini = np.hstack((masses_ini, mstar_ini))
            sizes_ini = np.hstack((sizes_ini, rhalf_ini))

            # evolve and store
            masses_evo_loc, sizes_evo_loc, _ = _cluster_evo_model_1(mstar_ini, rhalf_ini, t=age)

            masses_evo = np.hstack((masses_evo, masses_evo_loc))
            sizes_evo = np.hstack((sizes_evo, sizes_evo_loc))

        # plot histogram of initial stack
        if quant == "mass":
            h = np.histogram(np.log10(masses_ini), bins=nbins, range=xlim)
        if quant == "size":
            h = np.histogram(np.log10(sizes_ini), bins=nbins, range=xlim)

        h = [h[0] / len(sims), h[1]]  # mean
        ax.stairs(*h, fill=False, edgecolor="#000", lw=lw, linestyle="--", alpha=0.3, label="Initial")

        # plot histogram of evolved stack
        if quant == "mass":
            h = np.histogram(np.log10(masses_evo), bins=nbins, range=xlim)
        if quant == "size":
            h = np.histogram(np.log10(sizes_evo), bins=nbins, range=xlim)

        h = [h[0] / len(sims), h[1]]  # mean
        ax.stairs(*h, fill=False, edgecolor="#000", lw=lw + 0.5, linestyle=":", alpha=0.8, label="Evolved")

    # overplot resolution lines
    legend_loc = "upper right"
    ax.set_ylim([0.8, ymax * 1.2])
    ylim = ax.get_ylim()
    yy = [ylim[1] / 1.5, ylim[1]]

    for sim in sims:
        if quant == "mass":
            min_mass = np.log10(20 * sim.units.codeMassToMsun(sim.targetGasMass))
            ax.plot([min_mass, min_mass], yy, color="#444", linestyle="-", lw=lw, alpha=0.8)

        if quant == "size":
            grav_soft_stars = {13: 0.0244, 14: 0.0122, 15: 0.0061, 16: 0.003}
            grav_soft_code = grav_soft_stars[sim.res]
            grav_soft_logpc = np.log10(sim.units.codeLengthToPc(grav_soft_code))
            ax.plot([grav_soft_logpc, grav_soft_logpc], yy, color="#444", linestyle="-", lw=lw, alpha=0.8)

            # add another higher redshift (earliest where clusters really form)
            scalefac = 1 / (1 + 14.0)
            grav_soft_logpc = np.log10(sim.units.codeLengthToComovingKpc(grav_soft_code) * 1000 * scalefac)
            ax.plot([grav_soft_logpc, grav_soft_logpc], yy, color="#666", linestyle="-", lw=lw, alpha=0.8)

    # custom behavior
    if quant == "mass":
        # add mass function slope of -2
        x = np.array([2.7, 3.3])  # np.array(xlim) + [0.1, -0.1]
        y = 10 ** (x * -2.0)

        y *= ymax / y[0]  # * 1.2  # shift to match the top of the stacked histogram
        ax.plot(x, y, "--", color="#000", alpha=0.8, label=r"$dN/dM \propto M^{-2}$")

    if quant == "age":
        # vertical lines at lookback times to specific redshifts
        redshifts = [5.7, 6.0, 8.0, 14]
        for z in redshifts:
            age_z = np.log10((sim.tage - sim.units.redshiftToAgeFlat(z)) * 1000)
            yy = [ylim[0], ylim[0] * 10]
            ax.plot([age_z, age_z], yy, color="#444", linestyle=":", alpha=1.0)
            ax.text(age_z - 0.02, yy[1], f"z={z:.0f}", rotation=90, va="top", ha="right", color="#444", alpha=1.0)

        # set custom ticks
        ageticks_log = np.log10(ageticks_lin)
        ax.set_xticks(ageticks_log)
        ax.set_xticklabels(ageticks_lin)
        legend_loc = "upper left"

    if quant == "size":
        # set custom ticks
        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_xticks(sizeticks_log)
        ax.set_xticklabels(sizeticks_lin)

        # todo: overplot Brown+21 data

    ax.legend(loc=legend_loc)

    fig.savefig(f"star_cluster_histo_{quant}.pdf")
    plt.close(fig)


def _plot_evolved_properties(ax, sims, **kwargs):
    """Helper function that can be called as f_post(), to overplot the evolved properties of the clusters."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # loop over simulations
    for sim in sims:
        # load
        subIDs = _starClusterSubhaloIDs(sim)

        if subIDs.size == 0:
            continue

        mstar_ini = sim.subhalos("mstar_max")[subIDs]  # Msun
        rhalf_ini = sim.subhalos("size_stars_max")[subIDs]  # pc
        age = sim.subhalos("stellarage_myr")[subIDs]

        # evolve and store
        masses_evo, sizes_evo, _ = _cluster_evo_model_1(mstar_ini, rhalf_ini, t=age)

        if "Mass" in ax.get_xlabel():
            xval = np.log10(masses_evo)
        if "Size" in ax.get_xlabel():
            xval = np.log10(sizes_evo)

        if "Size" in ax.get_ylabel():
            yval = np.log10(sizes_evo)
        if "Surface Density" in ax.get_ylabel():
            sigma_evo = masses_evo / (np.pi * sizes_evo**2)
            yval = np.log10(sigma_evo)

        # plot with custom style
        marker_loc = {"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11}

        (l,) = ax.plot(xval, yval, ls="none", marker="o", **marker_loc)

        # fit line to x,y using np.polyfit
        if "Mass" in ax.get_xlabel() and "Size" in ax.get_ylabel():
            w = np.where((xval > xlim[0]) & (xval < xlim[1]) & (yval > ylim[0]) & (yval < ylim[1]))[0]

            coeffs = np.polyfit(xval[w], yval[w], deg=1)

            # overplot best-fit line
            x_fit = np.array(xlim) + [0.2, -0.2]
            y_fit = np.polyval(coeffs, x_fit)

            ax.plot(x_fit, y_fit, "-", color=l.get_color(), alpha=0.9)


def size_vs_mass(sims: list[simParams]) -> None:
    """Size-mass relation for star clusters."""
    xQuant = "mstar_tot"
    yQuant = "rhalf_stars_pc"
    xlim = mass_lim
    ylim = [size_lim[0], 1.3]

    def _f_pre(ax, sims):
        # set custom axis and tick labels
        ax.set_xlabel(mass_label)
        ax.set_ylabel(size_label)

        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_yticks(sizeticks_log)
        ax.set_yticklabels(sizeticks_lin)

    def _draw_fitline(ax, sims, **kwargs):
        from temet.load.data import brown21

        # fit line to x,y using np.polyfit
        x_all = np.array(kwargs["x"])
        y_all = np.array(kwargs["y"])
        w = np.where((x_all > xlim[0]) & (x_all < xlim[1]) & (y_all > ylim[0]) & (y_all < ylim[1]))[0]

        if len(w) > 0:
            coeffs = np.polyfit(x_all[w], y_all[w], deg=1)

            # overplot best-fit line
            x_fit = np.array(xlim) + [0.2, -0.2]
            y_fit = np.polyval(coeffs, x_fit)

            ax.plot(x_fit, y_fit, "-", color="#000", alpha=0.9)

        # Brown+21 LEGUS (https://arxiv.org/abs/2106.12420)
        b21 = brown21()

        w = np.where(b21["reliable"])
        x = np.log10(b21["mass"][w])
        y = np.log10(b21["r_eff"][w])

        ax.plot(x, y, marker="x", mew=1, ms=4, ls="None", color="#000", alpha=0.3, zorder=-12, label=b21["label"])

        # Brown+21 fit (Fig 13, including the local open clusters at low mass)
        # b21_betas = [0.242, 0.180, 0.279]  # full sample, 1-10Myr, and 10-100Myr age bins
        # b21_R4s = [2.548, 2.365, 2.506]
        b21_betas = [0.29]
        b21_R4s = [2.57]

        x_fit = np.array(xlim)  # + [0.1, -0.1]

        for beta, R4 in zip(b21_betas, b21_R4s):
            y_fit = np.log10(R4 * (10.0**x_fit / 1e4) ** beta)

            label = ""  # rf"Brown+21 ($\beta={beta:.3f}$, $R_4={R4:.1f}$ pc)"
            ax.plot(x_fit, y_fit, "-", color="#000", alpha=0.5, label=label)

        # Grudic+23 fit to STARFORGE sims
        g23_beta = 0.25
        g23_R4 = 1.4

        y_fit = np.log10(g23_R4 * (10.0**x_fit / 1e4) ** g23_beta)

        ax.plot(x_fit, y_fit, "-.", color="#000", alpha=0.5, label=r"Grudi${\'c}$+23")

        # Marks+12 model fit
        m12_beta = 0.13
        m12_R1 = 0.1

        y_fit = np.log10(m12_R1 * (10.0**x_fit / 1.0) ** m12_beta)

        ax.plot(x_fit, y_fit, ls="--", color="#000", alpha=0.3, label="Marks+12")

        # TODO: if cluster masses start to exceed 4.5, then we can/should consider the JWST "cluster" datasets:
        # see https://arxiv.org/abs/2603.24550 Figure 1
        # Vanzella+22,23, Mowla+24, Adamo+24, Messa+25, also SFing clumps in high-z (Fujimoto+25, Nakane+25)

    def _f_post(ax, sims, **kwargs):
        _draw_fitline(ax, sims, **kwargs)
        _plot_evolved_properties(ax, sims)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        tracks=False,
        sizefac=1.0,
        # markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.5, "zorder": -11},  # rasterize for zorder<-10
        markerstyle={"ms": 5.0, "fillstyle": "none", "mew": 1.5, "alpha": 0.3, "zorder": -11},
        legend="simple",
        legend_locs=["lower right", "upper left"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_f_post,
        f_selection=_starClusterSubhaloIDs,
    )


def sigma_star_galaxies(sims: list[simParams]) -> None:
    """The stellar mass surface density (Sigma_*) as a function of galaxy mass."""
    yQuant = "surfdens_stars"
    xQuant = "mstar2_log"

    ylim = [5.5, 13.5]  # log msun/kpc^2
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
            label=c23["label"],
        )

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        parents=False,
        tracks=True,
        sizefac=0.8,
        legend="simple",
        legend_locs=["upper right", "upper left"],
        legend_ncols=[1, 2],
        f_pre=_draw_data,
        f_selection=_zoomSubhaloIDsToPlot,
    )


@catalog_field(aliases=["cfr_10myr", "cfr_100myr"])
def cluster_formation_rate(sim, field):
    """Star cluster formation rate."""
    # mass in clusters is a sum over satellite subhalos for each central subhalo, do on the fly
    age_cl = sim.subhalos("stellarage_myr")  # "Subhalo_StellarAge_NoRadCut_MassWt"
    mstar_cl = sim.subhalos("mstar_tot")  # msun
    cen_flag = sim.subhalos("cen_flag")

    dt = 100 if "100myr" in field else 10  # Myr

    if dt == 10:
        mstar_cl[age_cl > 10] = 0  # age threshold of <10 Myr
    if dt == 100:
        mstar_cl[(age_cl < 10) | (age_cl > 100)] = 0  # age threshold of >10 and <100 Myr

    mstar_cl[cen_flag == 1] = 0  # only include satellites, not centrals

    # for each halo, sum (weighted histogram) its subhalos (only satellites by construction)
    grnr = sim.subhalos("SubhaloGrNr")
    first_sub = sim.halos("GroupFirstSub")

    mstar_cl_halo, _ = np.histogram(grnr, weights=mstar_cl, bins=np.arange(sim.numHalos + 1))

    # assign values to central subhalos
    mstar_cl_sub = np.zeros(sim.numSubhalos, dtype="float32")
    mstar_cl_sub.fill(np.nan)

    w = np.where(first_sub >= 0)
    mstar_cl_sub[first_sub[w]] = mstar_cl_halo[w]  # assign to central subhalo of each halo

    # calculate CFR = mass in clusters / time
    with np.errstate(invalid="ignore", divide="ignore"):
        cfr = mstar_cl_sub / (dt * 1e6)  # convert Myr to yr

    return cfr


cluster_formation_rate.label = (
    lambda sim, f: r"Cluster Formation Rate ($\rm{CFR_{100}}$)"
    if "100myr" in f
    else r"Cluster Formation Rate ($\rm{CFR_{10}}$)"
)
cluster_formation_rate.units = r"$\rm{M_{sun}\, yr^{-1}}$"
cluster_formation_rate.limits = [-4.0, 0.0]
cluster_formation_rate.log = True


@catalog_field(aliases=["cfe_10myr", "cfe_100myr"])
def cluster_formation_efficiency(sim, field):
    """Star cluster formation efficiency (definition of Bastian+08)."""
    dt = 100 if "100myr" in field else 10  # Myr

    # load cluster formation rate
    cfr = sim.subhalos(f"cfr_{dt}myr")  # msun/yr

    # galaxy mass is the fof sum (not central subhalo)
    mstar_tot = sim.subhalos(f"mass_stars_halo_{dt}myr")  # msun

    # calculate CFE = mass in clusters / total stellar mass
    with np.errstate(invalid="ignore", divide="ignore"):
        sfr = mstar_tot / (dt * 1e6)  # convert Myr to yr, so that CFE is dimensionless and comparable to CFR
        cfe = cfr / sfr

    if np.count_nonzero(np.isfinite(cfe)):
        if np.nanmax(cfe) > 1.0:
            print(" Warning: CFE should be <= 1.0, but max is %.4f" % np.nanmax(cfe))
        cfe = np.clip(cfe, 0, 1)  # can exceed unity due to mismatch between particle and mean subhalo ages

    return cfe


cluster_formation_efficiency.label = (
    lambda sim, f: r"Cluster Formation Efficiency ($\Gamma_{100}$)"
    if "100myr" in f
    else r"Cluster Formation Efficiency ($\Gamma_{10}$)"
)
cluster_formation_efficiency.units = r""  # dimensionless
cluster_formation_efficiency.limits = [0, 1]  # [-2.0, 0.0]
cluster_formation_efficiency.log = False  # True


def cfe_galaxies(sims: list[simParams], dt=10, xQuant="mstar2_log", cfr=False) -> None:
    """The cluster formation efficiency (Gamma) as a function of galaxy mass, or galaxy SFR."""
    xlim = None  # auto

    yQuant = f"cfe_{dt}myr"
    ylim = [0, 1]  # linear

    if cfr:
        yQuant = f"cfr_{dt}myr"
        ylim = [-5.5, 0.0]  # log msun/yr

    if xQuant in ["mstar2_log", "mstar_log"]:
        xlim = [4.5, 8.5]  # log mstar
    if xQuant in ["sfr_10myr", "sfr_100myr"]:
        xlim = [-5.5, -1.0]
    if "surfdens" in xQuant:
        xlim = [-4, 6]  # log msun/yr/kpc^2
        ylim = [4e-3, 1.0]

    def _f_pre(ax, sims):
        if not cfr and "surfdens" in xQuant:
            # custom y-axis setup
            ax.set_yscale("log")

        # overplot theoretical models
        if not cfr and "surfdens" in xQuant:
            # Dinnbier+22 (arXiv 2201.06582 Figure 7 all blue lines)
            d22_sigma = [-4.0, -3.0, -2.0, -1.0, 0.0]  # log msun/yr/kpc^2
            d22_cfe_low = 10.0 ** np.array([-1.15, -1.07, -1.01, -0.98, -0.96])  # log CFE
            d22_cfe_high = 10.0 ** np.array([-0.62, -0.50, -0.40, -0.36, -0.31])

            ax.fill_between(d22_sigma, d22_cfe_low, d22_cfe_high, color="#000", alpha=0.2, label="Dinnbier+22")

            # Kruijssen+12 (https://arxiv.org/abs/1208.2963 Eqn. 45, Fig 6)
            k12_sigma = np.linspace(-4.0, 0.0, 100)
            k12_sigma_lin = 10.0**k12_sigma
            k12_cfe = 1 / (1.15 + 0.6 * k12_sigma_lin ** (-0.4) + 0.05 * k12_sigma_lin ** (-1.0))

            ax.plot(k12_sigma, k12_cfe, "--", color="#000", alpha=0.5, label="Kruijssen+12")

    def _f_post(ax, sims, **kwargs):
        # print some statistics
        mean_yval = np.nanmean(kwargs["y"])
        print(f"Mean {yQuant} across all galaxies: {mean_yval:.3f}")

        # Cook+23 (https://arxiv.org/pdf/2212.07519 Fig 12, Table 6, binned data points for Ha and 1-10 Myr)
        c23_label = r"Cook+23"
        c23_sigma = [-3.06, -2.76, -2.35, -1.81, -1.24]
        c23_sigma_err = [0.25, 0.12, 1.08, 0.58, 0.26]
        c23_gamma = [np.nan, 0.307, 0.173, 0.241, 0.326]
        c23_gamma_err = [np.nan, 0.05, 0.105, 0.114, 0.3]  # last err lowered for visual clarity

        ax.errorbar(
            c23_sigma,
            c23_gamma,
            xerr=c23_sigma_err,
            yerr=c23_gamma_err,
            fmt="o",
            color="#555",
            alpha=0.5,
            label=c23_label,
        )

        c23_sigma = [-3.17, -2.72, -2.25, -1.51, -1.17]
        c23_sigma_err = [0.11, 0.08, 0.04, 0.01, 0.25]
        c23_gamma = [0.79, 0.346, 0.622, 0.68, 0.16]
        c23_gamma_err = [0.113, 0.163, 0.278, 0.66, 0.061]  # second to last err lowered for clarity

        ax.errorbar(c23_sigma, c23_gamma, xerr=c23_sigma_err, yerr=c23_gamma_err, fmt="s", color="#555", alpha=0.7)

        # running median
        # xm, ym, _, pm = running_median(kwargs["x"], kwargs["y"], binSize=0.5, percs=[16, 50, 84])
        # ax.plot(xm, ym, "-", color="#000", alpha=0.8)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        # cQuant="redshift",
        xlim=xlim,
        ylim=ylim,
        # clim=[5.5, 10],
        vs_sim=None,
        parents=False,
        tracks=0.1,  # 'all'
        sizefac=0.8,
        legend="simple",
        legend_locs=["lower left", "upper right"] if not cfr else ["lower right", "upper left"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_f_post,
        f_selection=_zoomSubhaloIDsToPlot,
    )


def sigma_star_vs_mass_clusters(sims: list[simParams]) -> None:
    """The stellar mass surface density (Sigma_*) of star clusters, as a function of cluster mass."""
    yQuant = "surfdens_stars_pc"
    xQuant = "mstar_tot_log"

    ylim = sigma_lim  # log msun/pc^2
    xlim = mass_lim  # log mstar

    def _f_pre(ax, sims):
        from temet.load.data import brown21

        # set custom ticks and tick labels
        ax.set_xlabel(mass_label)

        # draw lines of constant size
        xx = np.linspace(xlim[0], xlim[1], 100)
        opts = {
            "fontsize": 11,
            "color": "#444",
            "alpha": 1.0,
            "ha": "left",
            "va": "bottom",
            "rotation": 27.0,
            "bbox": {"facecolor": "white", "alpha": 0.5, "pad": 2},
        }

        sizes = [0.02, 0.05, 0.1, 0.5, 1.0]  # pc

        for i, size in enumerate(sizes):
            yy = np.log10(10.0**xx / (np.pi * (size) ** 2))
            label = f"{size:.1f} pc" if size >= 0.1 else f"{size:.2f} pc"

            ax.plot(xx, yy, ls=":", lw=1, color="#444", alpha=1.0)
            ind = 3 if i < 3 else 20 + 15 * (i - 3)
            ax.text(xx[ind], yy[ind] + 0.05, label, **opts)

        # Brown+21 LEGUS (https://arxiv.org/abs/2106.12420)
        b21 = brown21()

        w = np.where(b21["reliable"])
        y = b21["surfdens"][w]
        x = np.log10(b21["mass"][w])

        ax.plot(x, y, marker="x", mew=1, ms=4, ls="None", color="#000", alpha=0.5, zorder=-12, label=b21["label"])

        # TODO: if cluster masses start to exceed 4.5, then we can/should consider the JWST "cluster" datasets:
        # see https://arxiv.org/abs/2603.24550 Figure 1
        # Vanzella+22,23, Mowla+24, Adamo+24, Messa+25, also SFing clumps in high-z (Fujimoto+25, Nakane+25)

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
        # markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        markerstyle={"ms": 5.0, "fillstyle": "none", "mew": 1.5, "alpha": 0.3, "zorder": -11},
        legend="simple",
        legend_locs=["upper left", "upper right"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_plot_evolved_properties,
        f_selection=_starClusterSubhaloIDs,
    )


def sigma_star_vs_size_clusters(sims: list[simParams]) -> None:
    """The stellar mass surface density (Sigma_*) of stars clusters, as a function of cluster size."""
    yQuant = "surfdens_stars_pc"
    xQuant = "rhalf_stars_pc"

    ylim = sigma_lim  # log msun/pc^2
    xlim = size_lim  # log pc

    def _f_pre(ax, sims):
        from temet.load.data import adamo24, brown21, mowla24

        # set custom ticks and tick labels
        ax.set_xlabel(size_label)
        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_xticks(sizeticks_log)
        ax.set_xticklabels(sizeticks_lin)

        # draw lines of constant mass
        xx = np.linspace(xlim[0], xlim[1], 100)
        opts = {
            "fontsize": 11,
            "color": "#444",
            "alpha": 1.0,
            "ha": "left",
            "va": "bottom",
            "rotation": -41.0,
            "bbox": {"facecolor": "white", "alpha": 0.15, "pad": 2},
        }

        masses = [1e1, 1e2, 1e3, 1e4, 1e5]  # msun

        for i, mass in enumerate(masses):
            yy = np.log10(mass / (np.pi * (10.0**xx) ** 2))
            label = r"$10^{" + f"{np.log10(mass):.0f}" + r"}$ M$_{\odot}$"

            ax.plot(xx, yy, ls=":", lw=1, color="#444", alpha=1.0)
            ax.text(xx[30 + 10 * i], yy[30 + 10 * i] - 0.1, label, **opts)

        # draw observational data
        a24 = adamo24()

        ax.errorbar(
            a24["r_eff"],
            a24["surfdens"],
            xerr=[a24["r_eff_err1"], a24["r_eff_err2"]],
            yerr=[a24["surfdens_err2"], a24["surfdens_err1"]],
            fmt="D",
            color="#555",
            alpha=0.7,
            label=a24["label"],
        )

        m24 = mowla24()

        ax.errorbar(
            m24["r_eff"],
            m24["surfdens"],
            xerr=[m24["r_eff_err1"], m24["r_eff_err2"]],
            yerr=[m24["surfdens_err2"], m24["surfdens_err1"]],
            fmt="s",
            color="#555",
            alpha=0.7,
            label=m24["label"],
        )

        # Brown+21 LEGUS (https://arxiv.org/abs/2106.12420)
        b21 = brown21()

        w = np.where(b21["reliable"])
        y = b21["surfdens"][w]
        x = np.log10(b21["r_eff"][w])

        ax.plot(x, y, marker="x", mew=1, ms=4, ls="None", color="#000", alpha=0.5, zorder=-12, label=b21["label"])

        # sim: van Donkelaar+26 Fig 4
        vd26_label = "van Donkelaar+26"
        vd26_rhalf = np.log10([2.97,2.53,2.97,3.22,3.56,2.69,2.92,3.12,2.57,2.79,2.71,3.02,2.97,2.97,2.97,3.13,2.92,
                      2.54,2.57,3.03,3.46,3.81,3.86,3.65,3.76,3.8,3.77,3.95,4.3,4.35,3.51,3.71,3.85,4.1,4.35,4.48,4.57,
                      4.63,4.67,4.76,4.53,4.33,4.58,4.96,4.77,5.71,6.58,7.3,7.6,7.74,9.11,9.98,12.0])  # fmt: skip
        vd26_sigma = np.log10([9.01e+05, 3.35e+05, 2.59e+05, 3.45e+05, 3.23e+05, 1.91e+05,
                     1.64e+05, 1.22e+05, 6.84e+04, 6.48e+04, 5.81e+04, 4.93e+04,
                     3.40e+04, 3.29e+04, 2.08e+04, 1.26e+04, 4.92e+03, 3.07e+03,
                     2.64e+03, 2.17e+03, 3.43e+03, 4.31e+03, 5.92e+03, 1.40e+04,
                     1.73e+04, 1.95e+04, 4.47e+04, 3.63e+04, 3.75e+04, 5.26e+04,
                     7.97e+04, 1.36e+05, 1.28e+05, 2.54e+05, 2.80e+05, 1.55e+05,
                     2.40e+05, 2.65e+05, 2.83e+05, 1.31e+04, 5.42e+03, 4.65e+03,
                     3.62e+03, 1.46e+03, 6.23e+02, 3.81e+02, 2.17e+03, 2.52e+03,
                     1.04e+04, 1.68e+03, 2.11e+02, 2.23e+02, 6.37e+02])  # fmt: skip

        ax.plot(
            vd26_rhalf, vd26_sigma, marker="o", mew=2, mfc="None", ls="None", color="#000", alpha=0.5, label=vd26_label
        )

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        tracks=False,
        sizefac=1.0,
        # markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        markerstyle={"ms": 5.0, "fillstyle": "none", "mew": 1.5, "alpha": 0.3, "zorder": -11},
        legend="simple",
        legend_locs=["lower left", "upper left"],
        legend_ncols=[1, 2],
        f_pre=_f_pre,
        f_post=_plot_evolved_properties,
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


def gasfrac_vs_age_clusters(sims: list[simParams]) -> None:
    """The gas fraction (M_gas / (M_gas+M_stars)) of star clusters, as a function of cluster age."""
    yQuant = "fgas2_fof"  # auxcat-based recomputation within 2rhalf, but full fof-scope
    xQuant = "stellarage_myr"  # cluster age

    ylim = [-3.1, 0.0]  # [0, 1]
    xlim = age_lim  # log Myr

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        ax.set_xlabel(age_label)
        ageticks_log = np.log10(ageticks_lin)
        ax.set_xticks(ageticks_log)
        ax.set_xticklabels(ageticks_lin)

        ax.set_ylabel("Gas Fraction [ log ]")

        # note: (van Donkelaar+26 Fig 3)

    def _f_post(ax, sims, x, y):
        # overplot line where fgas = 1% and 10%
        ax.plot(xlim, [-2, -2], ls=":", color="#444", alpha=0.2)
        ax.plot(xlim, [-1, -1], ls=":", color="#444", alpha=0.2)

        # plot running median including fgas == 0 values
        y[np.isnan(y)] = ylim[0] + 0.1
        w = np.where(x < np.log10(10))[0]  # filter out points that are moved for visibility
        xm, ym, _, pm = running_median(x[w], y[w], binSize=0.1, percs=[16, 50, 84])

        ax.fill_between(xm, pm[0, :], pm[-1, :], color="#555", alpha=0.1)
        ax.plot(xm, ym, "--", color="#555", alpha=0.8)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        tracks=False,
        median=True,
        sizefac=0.8,
        markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        legend="simple",
        legend_locs=["upper left", "upper right"],
        legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_f_post,
        f_selection=_starClusterSubhaloIDs,
    )


def age_vs_mass_clusters(sims: list[simParams]) -> None:
    """Cluster age as a function of cluster mass."""
    yQuant = "stellarage_myr"  # cluster age
    xQuant = "mstar_tot_log"  # subhalo mass

    ylim = age_lim  # log Myr
    xlim = mass_lim  # log mstar

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        ax.set_ylabel(age_label)
        ageticks_log = np.log10(ageticks_lin)
        ax.set_yticks(ageticks_log)
        ax.set_yticklabels(ageticks_lin)

        ax.set_xlabel(mass_label)

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
        # legend_locs=["upper left", "upper right"],
        # legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_selection=_starClusterSubhaloIDs,
    )


def age_spread_clusters(sims: list[simParams]) -> None:
    """Cluster age spread (of member stars) as a function of cluster mass."""
    yQuant = "stellarage_spread_myr"  # cluster age spread
    xQuant = "mstar_tot_log"  # subhalo mass

    ylim = [-2.25, 1.5]  # age_lim  # log Myr
    xlim = mass_lim  # log mstar

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        # ax.set_ylabel(age_label)
        # ageticks_log = np.log10(ageticks_lin)
        # ax.set_yticks(ageticks_log)
        # ax.set_yticklabels(ageticks_lin)

        ax.set_xlabel(mass_label)

    def draw_data(ax, sims, **kwargs):
        # Longmore+2014 review quotes several numbers (Sec 3.2.3)
        # Orion Nebula Cluster (ONC)
        xerr = np.reshape([np.log10(3700) - np.log10(2000), np.log10(2000) - np.log10(900)], (2, 1))
        yerr = np.reshape([np.log10(3.0) - np.log10(2.0), np.log10(2.0) - np.log10(1.0)], (2, 1))
        ax.errorbar(np.log10(2000), np.log10(2.0), xerr=xerr, yerr=yerr, marker="o", color="#444", label="ONC")

        # Westerlund 1
        ax.plot(xlim, [np.log10(3), np.log10(3)], ls=":", color="#444", alpha=0.5)
        ax.text(xlim[1] - 0.1, np.log10(3) + 0.05, "Westerlund 1", color="#444", ha="right", va="bottom", alpha=0.5)

        # W3-main
        xerr = np.reshape([np.log10(5000) - np.log10(4000), np.log10(4000) - np.log10(3000)], (2, 1))
        yerr = np.reshape([np.log10(3.0) - np.log10(2.5), np.log10(2.5) - np.log10(2.0)], (2, 1))
        ax.errorbar(np.log10(4e3), np.log10(2.5), xerr=xerr, yerr=yerr, marker="D", color="#444", label="W3-main")

        # NGC 3603YC (core) (Kudryavtseva+12) (upper limit)
        ax.plot(xlim, [np.log10(0.1), np.log10(0.1)], ls=":", color="#444", alpha=0.5)
        ax.text(
            xlim[1] - 0.1,
            np.log10(0.1) + 0.05,
            "NGC 3603YC (upper limit)",
            color="#444",
            ha="right",
            va="bottom",
            alpha=0.5,
        )

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
        # legend_locs=["upper left", "upper right"],
        # legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=draw_data,
        f_selection=_starClusterSubhaloIDs,
    )


def age_vs_tcross_clusters(sims: list[simParams]) -> None:
    """Cluster age as a function of cluster mass."""
    xQuant = "stellarage_myr"
    yQuant = "crossing_time"

    xlim = age_lim  # log Myr
    ylim = age_lim  # log Myr

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        ax.set_xlabel(age_label)
        # ageticks_log = np.log10(ageticks_lin)
        # ax.set_xticks(ageticks_log)
        # ax.set_xticklabels(ageticks_lin)
        pass

        # overplot line where age = tcross
        ax.plot(xlim, xlim, ls="-", color="#444", alpha=0.7)

    def _f_post(ax, sims, **kwargs):
        t_pos = 0.8
        t_pad = 0.07
        opts = {"rotation": 38, "color": "#444", "ha": "center", "va": "center", "alpha": 1.0}
        ax.text(xlim[0] + t_pos - t_pad, xlim[0] + t_pos + t_pad, "Unbound", **opts)
        ax.text(xlim[0] + t_pos + t_pad, xlim[0] + t_pos - t_pad, "Bound", **opts)

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
        # legend_locs=["upper left", "upper right"],
        # legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=_f_post,
        f_selection=_starClusterSubhaloIDs,
    )


def most_massive_stars(sims: list[simParams]) -> None:
    """Most massive (actual) star in a cluster, as a function of other cluster properties."""
    xQuant = "mstar_tot"
    yQuant = "most_massive_starini"
    xlim = mass_lim
    ylim = [0.4, 2.177]  # log msun

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        massticks_lin = [3, 8, 14, 20, 40, 80, 150]  # msun
        massicks_log = np.log10(massticks_lin)
        ax.set_yticks(massicks_log)
        ax.set_yticklabels(massticks_lin)

        ax.set_xlabel(mass_label)

    def draw_data(ax, sims, **kwargs):
        from temet.load.data import weidner13

        # Larson+03 observed relation (https://arxiv.org/abs/astro-ph/0306595 Section 7.3)
        l03_mstar = np.log10([10, 1000])  # cluster mass
        l03_maxmass = np.log10([3.6, 26.9])  # most massive star

        l03_mstar = np.array(xlim)
        l03_maxmass = np.log10(1.2 * 10**l03_mstar**0.45)

        ax.plot(l03_mstar, l03_maxmass, ls="--", color="#000", alpha=0.5, label="Larson+03")

        # Elmegreen+00 theoretical model
        # e00_mstar = np.log10([10, 1000])
        # e00_maxmass = np.log10([1.6, 43.3])

        # ax.plot(e00_mstar, e00_maxmass, ls="-.", color="#000", alpha=0.5, label="Elmegreen+00")

        # Weidner+13 individual observations (young clusters with age < 3 Myr)
        w13 = weidner13()

        _, _, bars = ax.errorbar(
            w13["mass"],
            w13["maxstarmass"],
            xerr=[w13["mass_err1"], w13["mass_err2"]],
            yerr=[w13["maxstarmass_err1"], w13["maxstarmass_err2"]],
            fmt="s",
            color="#222",
            alpha=0.7,
            ms=6,
            label=w13["label"],
        )

        for bar in bars:
            bar.set_alpha(0.1)

    subhalos_evo.scatter2d(
        sims,
        xQuant=xQuant,
        yQuant=yQuant,
        xlim=xlim,
        ylim=ylim,
        vs_sim=None,
        parents=False,
        median=True,
        tracks=False,
        sizefac=0.8,
        markerstyle={"ms": 6.0, "fillstyle": "full", "alpha": 0.8, "zorder": -11},  # rasterize for zorder<-10
        legend="simple",
        # legend_locs=["lower right", "upper left"],
        # legend_ncols=[1, 1],
        f_pre=_f_pre,
        f_post=draw_data,
        f_selection=_starClusterSubhaloIDs,
    )


def _cluster_evo_model_1(Mcl0, rcl0, t=None):
    """Evolution model for star cluster mass and size (based on DRAGON-II paper).

    Args:
        Mcl0 (float): Initial cluster mass [Msun].
        rcl0 (float): Initial cluster size (half-mass radius) [pc].
        t (float or ndarray): Time to evaluate at (i.e. cluster age), or if None compute from t=0 to t=1Gyr.
    """
    if t is None:
        t = np.linspace(1, 1000, 100)  # Myr

    # fixed parameters (only for testing)
    # alpha_M = 0.3
    # beta_M = 0.3
    # alpha_R = 0.7
    # beta_R = 0.6

    # linear interpolation on Table 2 values
    table2_rhm = [0.47, 0.80, 1.75]
    table2_alpha_M = [0.15, 0.18, 0.31]
    table2_beta_M = [0.375, 0.34, 0.295]
    table2_alpha_R = [0.95, 0.7, 0.73]
    table2_beta_R = [0.50, 0.57, 0.59]

    alpha_M = np.interp(rcl0, table2_rhm, table2_alpha_M)
    beta_M = np.interp(rcl0, table2_rhm, table2_beta_M)
    alpha_R = np.interp(rcl0, table2_rhm, table2_alpha_R)
    beta_R = np.interp(rcl0, table2_rhm, table2_beta_R)

    # print(f"{alpha_M = :.2f}, {beta_M = :.2f}, {alpha_R = :.2f}, {beta_R = :.2f}")

    # from scipy.interpolate import interp1d

    # alpha_M = interp1d(table2_rhm, table2_alpha_M, kind="linear", fill_value="extrapolate")(rcl0)
    # beta_M = interp1d(table2_rhm, table2_beta_M, kind="linear", fill_value="extrapolate")(rcl0)
    # alpha_R = interp1d(table2_rhm, table2_alpha_R, kind="linear", fill_value="extrapolate")(rcl0)
    # beta_R = interp1d(table2_rhm, table2_beta_R, kind="linear", fill_value="extrapolate")(rcl0)

    # alpha_M = np.clip(alpha_M, 0.1, 0.4)
    # beta_M = np.clip(beta_M, 0.25, 0.40)
    # alpha_R = np.clip(alpha_R, 0.6, 1.2)
    # beta_R = np.clip(beta_R, 0.45, 0.65)

    # compute relaxation time
    mstar = 0.5  # mean stellar mass in Msun (assumed)
    N = Mcl0 / mstar  # number of stars, assuming mean mass
    gamma_n = 0.02  # depends on mass spectrum, 0.11-0.4 for monochromatic, as low as 0.02 for multi-mass spectrum
    lnLambda = np.log(gamma_n * N)  # Coulomb logarithm

    # # protect against negative log for very low N (scalar and array cases)
    if isinstance(lnLambda, (float, np.floating)):
        lnLambda = np.clip(lnLambda, np.log(2), np.inf)
    else:
        lnLambda[lnLambda <= 0] = np.log(2)

    t_relax = 282 / (mstar * lnLambda) * np.sqrt(Mcl0 / 1e5) * (rcl0 / 1.0) ** (3 / 2)  # Eqn 5

    # compute mass and size evolution: DRAGON-II model
    Mcl = Mcl0 * (1 + alpha_M * (t / t_relax)) ** (-beta_M)  # note: typo in arxiv version
    rcl = rcl0 * (1 + t / (alpha_R * t_relax)) ** beta_R

    return Mcl, rcl, t


def _cluster_evo_model_2(Mcl0, rcl0, t=None):
    """Evolution model for star cluster mass and size (based on B-POP paper).

    Args:
        Mcl0 (float): Initial cluster mass [Msun].
        rcl0 (float): Initial cluster size (half-mass radius) [pc].
        t (float or ndarray): Time to evaluate at (i.e. cluster age), or if None compute from t=0 to t=1Gyr.
    """
    if t is None:
        t = np.linspace(1, 1000, 100)  # Myr

    # fixed parameters
    t_st = 10.0  # Myr
    k1 = -0.1
    alpha_bnd = 100  # for YCs
    beta_bnd = 0.6  # how rapidly relaxation takes over in cluster dispersal

    f = alpha_bnd * (Mcl0 / 1e6) ** beta_bnd

    # compute relaxation time
    mstar = 0.5  # mean stellar mass in Msun (assumed)
    N = Mcl0 / mstar  # number of stars, assuming mean mass
    gamma_n = 0.02  # depends on mass spectrum, 0.11-0.4 for monochromatic, as low as 0.02 for multi-mass spectrum
    lnLambda = np.log(gamma_n * N)  # Coulomb logarithm

    t_relax = 282 / (mstar * lnLambda) * np.sqrt(Mcl0 / 1e5) * (rcl0 / 1.0) ** (3 / 2)  # Eqn 5
    print(f"Initial relaxation time: {t_relax:.2f} Myr")

    alpha = 0.2  # "selected in the range 0.15-0.25 at any timestep" (randomly changing?)
    tcc = 0.2 * t_relax  # highly uncertain, adopt this from Gurkan+04
    t0 = 5 * tcc * (1 + (rcl0 / 1.0))

    print(f"{tcc = }, {t0 = }")

    Mcl = Mcl0 * (1 + t / t_st) ** k1 * (1 - t / (f * t_relax))  # Eqn A4, A5
    rcl = rcl0 * (1 + t / t0) ** alpha

    assert np.min(Mcl) > 0  # not true for t > f*t_relax and this is quite short...
    assert np.min(rcl) > 0

    return Mcl, rcl, t


def star_cluster_evo_model(model=1):
    """Model the evolution of the masses and sizes of star clusters using results from collisional simulations."""
    # based on: https://arxiv.org/abs/2307.04805 (DRAGON-II, Eqns 6 and 7 and Table 2)
    # based on: https://arxiv.org/abs/2603.20430 (B-POP, Appendix A, Eqns A4-A8)

    # plot results
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 10))

    ax1.set_xlabel("Time [ Myr ]")
    ax1.set_ylabel("Mass [ log Msun ]")
    ax2.set_xlabel("Time [ Myr ]")
    ax2.set_ylabel("Size [ pc ]")

    for i, Mcl0 in enumerate([1e2, 1e3, 1e4]):
        for j, rcl0 in enumerate([0.05, 0.2, 0.5]):
            # run model
            if model == 1:
                Mcl, rcl, t = _cluster_evo_model_1(Mcl0, rcl0)
            if model == 2:
                Mcl, rcl, t = _cluster_evo_model_2(Mcl0, rcl0)

            # add to plot
            label = f"$M_0 = {Mcl0:.0e}$ , $r_0 = {rcl0:.1f}$ pc"
            ax1.plot(t, np.log10(Mcl), color=colors[i], linestyle=linestyles[j], label=label)
            ax2.plot(t, rcl, color=colors[i], linestyle=linestyles[j], label=label)

    ax1.legend()
    ax2.legend()

    fig.savefig(f"star_cluster_evolution_model{model}.pdf")
    plt.close(fig)


def size_mass_evo_clusters(color=None):
    """Time evolution of star clusters on the size-mass plane, compared to model tracks."""
    sim = simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5)

    clim = None
    if color == "redshift":
        clim = [5.5, 11.0]

    if 0:
        # discovery
        subIDs = _starClusterSubhaloIDs(sim)
        mpbs = sim.loadMPBs(subIDs, fields=["SubfindID", "SnapNum"], treeName="SubLink_gal")

        mstar = sim.subhalos("mstar")
        subIDs = list(mpbs.keys())[95:100]  # random selections
        print(subIDs)
        print(np.log10(mstar[subIDs]))
    else:
        subIDs = [90, 91, 283, 364, 433, 472, 505, 560, 563, 661, 685]

    def _f_pre(ax, sims):
        # set custom ticks and tick labels
        ax.set_ylabel(size_label)
        sizeticks_log = np.log10(sizeticks_lin)
        ww = np.where((sizeticks_log > ax.get_ylim()[0]) & (sizeticks_log < ax.get_ylim()[1]))[0]
        ax.set_yticks(sizeticks_log[ww])
        ax.set_yticklabels(np.array(sizeticks_lin)[ww])

        ax.set_xlabel(mass_label)

    def _model_tracks(ax, sims, **kwargs):
        mstar_ini = sim.subhalos("mstar_max")[subIDs]  # Msun
        rhalf_ini = sim.subhalos("size_stars_max")[subIDs]  # pc
        age = sim.subhalos("stellarage_myr")[subIDs]

        # evolve and store
        for i, subID in enumerate(subIDs):
            tt = np.linspace(0, age[i], 50)  # Myr
            # tt = np.logspace(-1.0, np.log10(age[i]), 100)  # Myr
            masses_evo, sizes_evo, _ = _cluster_evo_model_1(mstar_ini[i], rhalf_ini[i], t=tt)

            # debugging
            if 0:
                mpb = sim.loadMPB(subID, treeName="SubLink_gal")
                print(f"{subID = }, {mstar_ini[i] = :.2f} Msun, {rhalf_ini[i] = :.2f} pc, {age[i] = :.2f} Myr")

                mstar = sim.units.codeMassToMsun(mpb["SubhaloMassType"][:, 4])
                mass_check = np.max(mstar)
                ind_max = np.nanargmax(mstar)
                size_code = mpb["SubhaloHalfmassRadType"][ind_max, 4]

                scalefac = 1 / (1 + mpb["Redshift"][ind_max])
                size_check = sim.units.codeLengthToComovingKpc(size_code) * scalefac * 1000

                print(f"{mass_check = :.2f} Msun, {size_check = :.2f} pc")

                # acField = "Subhalo_Stars_Mass_SubLink_gal_Max"
                # ac = sim.auxCat(acField)

                print(f"{masses_evo[0] = :.2f} Msun, {sizes_evo[0] = :.2f} pc")

            # plot
            xx = np.log10(masses_evo)
            yy = np.log10(sizes_evo)

            if color is None:
                ax.plot(xx, yy, ls=":", color=colors[i], alpha=1.0)
                ax.plot(xx[0], yy[0], "D", color=colors[i], alpha=1.0, mew=2, mfc="None")  # ms=6
                ax.plot(xx[-1], yy[-1], "s", color=colors[i], alpha=1.0, mew=2, mfc="None")  # ms=6
            else:
                # colors vary along the lines
                if color == "redshift":
                    cvals = sim.units.ageFlatToRedshift(sim.tage - age[i] / 1e3 + tt / 1e3)
                else:
                    assert 0, "todo check (generalize)"  # need to interpolate MPB vals to model times!
                    mpb = sim.quantMPB(subID, color, treeName="SubLink_gal")
                    cvals = mpb[color]

                c_track_norm = (cvals - clim[0]) / (clim[1] - clim[0])
                c_track_norm = np.clip(c_track_norm, 0, 1)
                cmap = plt.get_cmap("turbo")
                c = cmap(c_track_norm)

                colored_line(xx, yy, c, ax=ax, ls=":", alpha=1.0)

                ax.plot(xx[0], yy[0], "D", color=c[0], alpha=1.0, mew=2, mfc="None")  # ms=6
                ax.plot(xx[-1], yy[-1], "s", color=c[-1], alpha=1.0, mew=2, mfc="None")  # ms=6

    opts = {
        "xquant": "mstar",
        "yquant": "size_stars_pc",
        "xlim": [2.25, 4.25],  # [2.5, 4.0],
        "ylim": [-1.7, 0.5],  # [-1.8, -0.5],
        "color": color,
        "clim": clim,
        "parents": False,
        "smooth": False,
        "smooth_custom": True,  # NOTE: if any smoothing, then evo models do not necessarily start on tracks
        "treeName": "SubLink_gal",
        "sizefac": 0.8,
        "f_pre": _f_pre,
        "f_post": _model_tracks,
        "f_selection": lambda sim: subIDs,
    }

    subhalos_evo.tracks2d([sim], **opts)

    # also 1D evo tracks: mass
    redshift_lim = [10.5, 5.3]
    opts_mass = {
        "xlim": redshift_lim,
        "ylim": opts["xlim"],
        "parents": False,
        "smooth": False,
        "f_pre": lambda ax, sims: ax.set_ylabel(mass_label),
        "treeName": "SubLink_gal",
        "f_selection": lambda sim: subIDs,  # _starClusterSubhaloIDs,
    }

    subhalos_evo.tracks1d([sim], quant=opts["xquant"], **opts_mass)

    # also 1D evo tracks: size
    def _f_pre_size(ax, sims):
        ax.set_ylabel(size_label)
        sizeticks_log = np.log10(sizeticks_lin)
        ww = np.where((sizeticks_log > ax.get_ylim()[0]) & (sizeticks_log < ax.get_ylim()[1]))[0]
        ax.set_yticks(sizeticks_log[ww])
        ax.set_yticklabels(np.array(sizeticks_lin)[ww])

    opts_size = {
        "xlim": redshift_lim,
        "ylim": opts["ylim"],
        "parents": False,
        "smooth": False,
        "f_pre": _f_pre_size,
        "treeName": "SubLink_gal",
        "f_selection": lambda sim: subIDs,  # _starClusterSubhaloIDs,
    }

    subhalos_evo.tracks1d([sim], quant=opts["yquant"], **opts_size)

    # also 1D evo tracks: surface density
    opts_surfdens = {
        "xlim": redshift_lim,
        "ylim": [3.9, 6.5],
        "parents": False,
        "smooth": False,
        "treeName": "SubLink_gal",
        "f_selection": lambda sim: subIDs,  # _starClusterSubhaloIDs,
    }

    subhalos_evo.tracks1d([sim], quant="surfdens_stars_pc", **opts_surfdens)


def vis_evo_clusters(partField="radvel"):
    """Time evolution of star clusters on the size-mass plane, compared to model tracks."""
    # vis config
    partType = "gas"
    # partField = "temp"  # "density"  # "coldens_msunkpc2"

    if partField == "dens":
        valMinMax = [0.0, 4.5]  # None
    if partField == "temp":
        valMinMax = [2.0, 5.0]
    if partField == "metal_solar":
        valMinMax = [-3.0, -1.5]

    # todo: radVel, confirm refPos and refVel working
    # todo: "rad_FUV", "rad_LW", ionizing rad, t_freefall based on gasdens

    # subhalo config
    sim = simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5)

    subIDs = [90, 91, 283, 364, 433, 472, 505, 560, 563, 661, 685]
    subID = subIDs[0]

    # plot config
    labelSim = False
    labelHalo = False
    labelZ = "2digits"
    labelScale = "physical"
    method = "voronoi_slice"
    nPixels = [400, 400]

    plotStars = "all"  # individual markers

    size = 10.0
    sizeType = "pc"

    # load MPB of a cluster (subhalo)
    mpb = sim.loadMPB(subID, treeName="SubLink_gal")

    stars_aform = sim.snapshotSubset("stars", "sftime", subhaloID=subID)  # member star (at z=5.5) mean age
    stars_zform = 1 / stars_aform - 1
    form_stars0 = sim.units.redshiftToAgeFlat(stars_zform.max())
    form_stars1 = sim.units.redshiftToAgeFlat(stars_zform.min())

    form_tage = sim.units.redshiftToAgeFlat(mpb["Redshift"][-1])  # first snap where cluster exists in tree

    # identify some (snap,id) pairs to visualize, near the beginning of the tree
    N_prior = 2
    N = 3

    snaps = mpb["SnapNum"][-N:][::-1]  # forward in time
    sub_ids = mpb["SubfindID"][-N:][::-1]
    redshifts = mpb["Redshift"][-N:][::-1]

    # setup panels
    panels = []

    # N_prior panels prior to MPB existing (before star cluster forms)
    pos_t = []
    times = []
    times_prior = []
    snaps_prior = np.arange(snaps[0] - N_prior, snaps[0])

    for snap, sub_ind in zip(snaps, sub_ids):
        sim_loc = sim.copy()
        sim_loc.setSnap(snap)
        sub = sim_loc.subhalo(sub_ind)

        pos_t.append(sub["SubhaloPos"])
        times.append(sim_loc.tage)

    pos_t = np.array(pos_t)

    for i, snap in enumerate(snaps_prior):
        sim_loc = sim.copy()
        sim_loc.setSnap(snap)

        # fit cubic spline and extrapolate to this time for pre-birth position
        pos_prior = [make_interp_spline(times, pos_t[:, i], k=2)(sim_loc.tage) for i in range(3)]

        dt0 = (sim_loc.tage - form_stars0) * 1000.0  # Myr
        dt1 = (sim_loc.tage - form_stars1) * 1000.0  # Myr

        label = r"$\Delta t_0 = %.1f$ Myr, $\Delta t_{\rm f} = %.1f$ Myr" % (dt0, dt1)
        print(sim_loc.redshift, snap, sub_ind, pos_prior)

        panels.append({"sP": sim_loc, "subhaloInd": sub_ind, "boxCenter": pos_prior, "labelCustom": label})

    # N panels after MPB exists
    for snap, sub_ind, redshift in zip(snaps, sub_ids, redshifts):
        sim_loc = sim.copy()
        sim_loc.setSnap(snap)

        dt0 = (sim_loc.tage - form_stars0) * 1000.0  # Myr
        dt1 = (sim_loc.tage - form_stars1) * 1000.0  # Myr
        label = r"$\Delta t_0 = %.1f$ Myr, $\Delta t_{\rm f} = %.1f$ Myr" % (dt0, dt1)
        print(redshift, snap, sub_ind, sim_loc.subhalo(sub_ind)["SubhaloPos"])

        panels.append({"sP": sim_loc, "subhaloInd": sub_ind, "labelCustom": label})

    # request render
    class plotConfig:
        plotStyle = "edged"
        colorbars = True
        rasterPx = nPixels
        # fontsize = 36
        nRows = 1
        saveFilename = f"star_cluster_evo_{sim.simName}_{sim.snap}_{subID}_{partField}.pdf"

    renderSingleHalo(panels, plotConfig, locals(), skipExisting=True)


# -------------------------------------------------------------------------------------------------


def paperPlots(a=False):
    """Plots for VESTRAL star clusters paper. (if a == True, make all figures)."""
    # list of sims to include
    # variants = ["ST15"]  # ['ST15c','ST15m','ST15s']
    # res = [14, 15, 16]
    # hInds = [1958, 5072, 15581, 23908, 31619, 73172, 219612, 311384, 446076, 539722, 844537]
    # redshift = 5.5

    # if (all == False), only dz < 0.1 matches
    # if (single == True), only the highest available res of each halo
    # sims = _get_existing_sims(variants, res, hInds, redshift, all=False, single=True)

    sims = []
    # sims.append(simParams("structures", hInd=1958, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=5072, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=15581, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=23908, res=14, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=31619, res=15, variant="ST15", redshift=5.5))
    # sims.append(simParams("structures", hInd=73172, res=15, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=446076, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=539722, res=16, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=844537, res=16, variant="ST15", redshift=5.5))

    # --- cluster demographics & properties ---

    # fig 1: single large galaxy image
    if 0:
        # sim = simParams("structures", hInd=23908, res=14, variant="ST14", redshift=5.5, haloInd=0)  # original
        sim = simParams("structures", hInd=219612, res=16, variant="ST15", redshift=6.1, haloInd=0)
        vis_single_galaxy(sim)

    # fig 2: clusters: mass function
    if 0 or a:
        star_cluster_histogram(sims, quant="mass")
        star_cluster_histogram(sims, quant="size", sizefac=0.8)

    # fig 3: clusters: size-mass relation, size distribution
    if 0 or a:
        # TODO: add Hunter+03 (cluster masses 1e1-1e3) and Gatto+21 (1e2-1e4) (both Magellanic clouds)
        # https://ui.adsabs.harvard.edu/abs/2009ApJ...691..946M/abstract
        size_vs_mass(sims)

    # fig 4: gallery of clusters (~10pc stamps?)
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

    # fig 5a: stellar surface density of clusters (Sigma_*)
    if 0 or a:
        sigma_star_vs_size_clusters(sims)
        sigma_star_vs_mass_clusters(sims)

    # fig 5b: galaxies: stellar surface density (Sigma_*)
    if 0 or a:
        sigma_star_galaxies(sims)

    # fig 6: gas fraction vs age (not interesting vs M*)
    if 0 or a:
        z_extra = [6.5, 6.3, 6.0, 5.9, 5.8, 5.7, 5.6]
        for z in z_extra:
            sims.append(simParams("structures", hInd=539722, res=16, variant="ST15", redshift=z))
            sims.append(simParams("structures", hInd=311384, res=16, variant="ST15", redshift=z))
            sims.append(simParams("structures", hInd=219612, res=16, variant="ST15", redshift=z))

        gasfrac_vs_age_clusters(sims)

    # fig 7a: stellar remnants: mass distribution
    if 0 or a:
        stellar_remnants(sims)

    # fig 7b: most massive star in a cluster, vs the cluster mass (see Lahen+20 fig X, Lahen+23 Fig 14)
    if 0 or a:
        most_massive_stars(sims)

    # fig 8: ages, i.e. histograms of member star ages (matched to vis gallery), or age vs. X scaling relations
    if 0 or a:
        # i.e. histogram of cluster formation redshifts (with respect to important events/starbursts/mergers)
        star_cluster_histogram(sims, quant="age", nbins=100, sizefac=[1.4, 0.8])
        ## age_vs_mass_clusters(sims)

        # crossing time (Brown+21 eqn 21, Fig 14, Fig 15) (Claeyssens+22 Fig 10) (also: fraction of bound clusters)
        age_vs_tcross_clusters(sims)

    # fig 8b: age spread (1sigma)
    if 0 or a:
        age_spread_clusters(sims)

    # --- formation ---

    # fig 9: 2d spacetime radial profiles
    if 0 or a:
        # evo
        # opts = {"haloID": 0, "max_z": 10.0, "rlog": True, "rlim": [-2.0, 1.0], "smooth_sizes": True}
        opts = {"haloID": 0, "max_z": 10.0, "rlog": True, "rlim": [-3.0, 0.0], "smooth_sizes": True}

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

    # fig 10: gas mass fraction of ISM gas in different phases, at e.g. z=10 and z=6 (bar plots?)
    if 0 or a:
        gas_phase_fractions(sims, frac_type="Mass")
        gas_phase_fractions(sims, frac_type="Vol")

    # fig 11: cluster formation efficiency (Gamma), vs galaxy M* and SFR
    if 0 or a:
        cfe_galaxies(sims, dt=10, xQuant="mstar2_log")
        cfe_galaxies(sims, dt=100, xQuant="mstar2_log")
        cfe_galaxies(sims, dt=10, xQuant="sfr_10myr")
        cfe_galaxies(sims, dt=10, xQuant="sfr_surfdens_10myr")

        # cfe_galaxies(sims, dt=10, xQuant="mstar2_log", cfr=True)
        # cfe_galaxies(sims, dt=10, xQuant="sfr_10myr", cfr=True)

    # fig 12: evolution tracks on size-mass plane (color by age, redshift, or fgas)
    # TODO: alternate model approach ~RAPSTER (https://github.com/Kkritos/Rapster) (https://arxiv.org/abs/2210.10055)
    # see also https://arxiv.org/abs/2202.06961 (Section 3.2)
    # see also https://arxiv.org/abs/2606.14852 (incl. Appendix B)
    if 0 or a:
        size_mass_evo_clusters()
        size_mass_evo_clusters(color="redshift")

    # fig 13: time evolution vis sequence, from pre-birth to post-birth (gas dens, vrad, tff_local, Q, stars, ...)
    if 1 or a:
        # todo: see Taylor+ Nature fig, and Ma+ figs (and movies) for ideas
        vis_evo_clusters()

    # fig todo: radius and radial velocity at formation time (use birth values of member stars?) (or just tree?)

    # fig todo: quantitative assessment of reason for formation (e.g. self-grav instability, compression, ...)

    # fig todo: any cluster population stat, e.g. mass func slope, size-mass slope, vs. halo mass (color by redshift)
    #  "universality" or not?
    # see https://arxiv.org/abs/2605.27509 (Fig 5, Eqn 11)

    # any mergers i.e. hierarchical assembly?

    # any disruption?
    # tidal tails: https://arxiv.org/abs/2606.06248

    # ---

    # pressure vs. rho phase space diagrams
    if 0 or a:
        xlim = [0.0, 5.0]
        sim = simParams("structures", hInd=31619, res=15, variant="ST15", redshift=5.5)
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant=None, ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="vrad", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="csnd", ext="pdf")
        phase_diagram(sim, yQuant="pres", xlim=xlim, cQuant="tff_local", ext="pdf")

    # radial profiles - halo comparisons
    if 0 or a:
        haloIDs = [0] * len(sims)  # assume first
        opts = {"haloIDs": haloIDs, "xlog": True, "xlim": [-2.0, 1.5], "ylog": True}

        snapshot.profile(sims, ptType="gas", ptProperty="numdens", ylim=[-4.5, 4.0], scope="global", **opts)

        snapshot.profile(sims, ptType="stars", ptProperty="dens", ylim=[2.5, 11.0], scope="global", **opts)

        snapshot.profile(sims, ptType="gas", ptProperty="cellsize_kpc", ylim=[-3.5, -0.5], scope="global", **opts)

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
