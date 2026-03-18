"""
MCST: stellar clusters paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

import matplotlib.pyplot as plt
import numpy as np

from temet.plot import snapshot, subhalos, subhalos_evo
from temet.plot.config import colors, figsize, linestyles, lw, markers
from temet.plot.subhalos import addUniverseAgeAxis
from temet.util import simParams
from temet.util.helper import cache, logZeroNaN

from .mcst import _get_existing_sims, _zoomSubhaloIDsToPlot


mass_label = r"Star Cluster Mass [ log M$_{\odot}$ ]"
size_label = r"Star Cluster Size $R_{1/2}$ [ pc ]"  # always show tick labels in linear

mass_lim = [1.5, 5.0]  # log msun
size_lim = [-0.5, 0.8]  # log pc


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

        tff = sim.snapshotSubset("gas", "tff_local", haloID=haloID)  # yr

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


def stellar_remnants(sim, haloID=0):
    """Plot the mass distribution of different stellar remnant types (WD, NS, BH)."""
    # start plot
    fig, ax = plt.subplots()
    ax.set_xlabel(r"Remnant Mass [ M$_{\odot}$ ]")
    ax.set_ylabel("Number")
    ax.set_yscale("log")

    remnant_type = sim.stars("remnant_type", haloID=haloID)
    remnant_mass = sim.stars("remnant_mass", haloID=haloID)

    type_names = {0: "none", 1: "white dwarf", 2: "neutron star", 3: "black hole"}

    for type_num, type_name in type_names.items():
        w = np.where(remnant_type == type_num)[0]
        label = f"{type_name} (N={len(w)})"

        h = np.histogram(remnant_mass[w], bins=50, range=[0, 100])  # [0.5, 2.0] to show WDs and NSs
        ax.stairs(*h, fill=True, label=label)

        print(type_name, len(w), remnant_mass[w].min(), remnant_mass[w].max())

    ax.legend(loc="best")

    fig.savefig(f"stellar_remnants_{sim.simName}_{sim.snap}.pdf")
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
            rhalf = sim.subhalos("rhalf_stars") * 1000  # pc
            h_quant = np.log10(rhalf[subIDs])

        label = f"h{sim.hInd}"  # f"{sim.simName} (N={len(subIDs)}/{sim.numSubhalos})"

        h = np.histogram(h_quant, bins=30, range=xlim)  # range=[min,max]
        ax.stairs(*h, fill=True, alpha=0.8, label=label)

    if quant == "mass":
        min_mass = np.log10(20 * sim.units.codeMassToMsun(sim.targetGasMass))
        ax.plot([min_mass, min_mass], ax.get_ylim(), color="black", linestyle=":", alpha=0.5)

    if quant == "size":
        sizeticks_lin = [0.5, 1.0, 2.0, 5.0]  # pc
        sizeticks_log = np.log10(sizeticks_lin)
        ax.set_xticks(sizeticks_log)
        ax.set_xticklabels(sizeticks_lin)

    ax.legend(loc="upper right")

    fig.savefig(f"star_cluster_histo_{quant}.pdf")
    plt.close(fig)


def size_vs_mass(sims, sizefac=0.8):
    """Size-mass relation for star clusters."""
    from ..plot.util import _finish_plot

    xQuant = "mstar_tot"
    yQuant = "rhalf_stars_pc"
    xlim = [1.5, 5.0]
    ylim = [-0.5, 0.8]

    # subhalos.median(
    #    sims,
    #    xQuant=xQuant,
    #    yQuants=[yQuant],
    #    cenSatSelect="sat",
    #    qRestrictions=[["halo_id", 0, 0]],  # select only subhalos in haloID=0
    #    xlim=xlim,
    #    ylim=ylim,
    #    scatterPoints=True,
    #    # f_selection=_zoomSubhaloIDsToPlot,
    # )

    # subhalos_evo.scatter2d(
    #    sims,
    #    xQuant=xQuant,
    #    yQuant=yQuant,
    #    xlim=xlim,
    #    ylim=ylim,
    #    tracks=False,
    #    parents=False,
    #    legend="simple",
    #    f_selection=_starClusterSubhaloIDs,
    # )

    # unique list of included halo IDs, resolutions, and variants
    hInds = sorted({sim.hInd for sim in sims})
    # res = sorted({sim.res for sim in sims})
    variants = sorted({sim.variant for sim in sims})

    _, xlabel, xMinMax, xLog = sims[0].simSubhaloQuantity(xQuant)
    _, ylabel, yMinMax, yLog = sims[0].simSubhaloQuantity(yQuant)

    # start plot
    fig, ax = plt.subplots(figsize=figsize * np.array(sizefac))

    ax.set_xlabel(mass_label)
    ax.set_ylabel(size_label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    yticks_lin = [0.5, 1.0, 2.0, 5.0]  # pc
    yticks_log = np.log10(yticks_lin)
    ax.set_yticks(yticks_log)
    ax.set_yticklabels(yticks_lin)

    # allocate for stack
    xx = []
    yy = []

    # individual zoom runs
    for _i, sim in enumerate(sims):
        # load
        xvals = sim.subhalos(xQuant)
        yvals = sim.subhalos(yQuant)

        xvals = logZeroNaN(xvals)
        yvals = logZeroNaN(yvals)

        # which subhalo(s) to include?
        subhaloIDs = _starClusterSubhaloIDs(sim)

        xvals = xvals[subhaloIDs]
        yvals = yvals[subhaloIDs]

        xx.append(xvals)
        yy.append(yvals)

        # color set by hInd
        c = colors[hInds.index(sim.hInd)]

        # marker set by variant
        marker = markers[variants.index(sim.variant) % len(markers)]

        # marker size set by resolution
        ms_loc = 6.0

        style = {"color": c, "ms": ms_loc, "fillstyle": "full", "linestyle": "none", "alpha": 0.8}

        label = f"h{sim.hInd}"  # f"{sim.simName} (N={len(subhaloIDs)}/{sim.numSubhalos})"
        (l,) = ax.plot(xvals, yvals, marker=marker, label=label, **style)

    # fit line to xx,yy using np.polyfit
    x_all = np.concatenate(xx)
    y_all = np.concatenate(yy)
    w = np.where((x_all > xlim[0]) & (x_all < xlim[1]) & (y_all > ylim[0]) & (y_all < ylim[1]))[0]
    coeffs = np.polyfit(x_all[w], y_all[w], deg=1)
    x_fit = np.array(xlim) + [0.2, -0.2]
    y_fit = np.polyval(coeffs, x_fit)
    ax.plot(x_fit, y_fit, "-", color="#555", lw=3, alpha=0.5)

    ax.legend(loc="upper left")

    _finish_plot(fig, saveFilename=f"scatter2d_evo_{xQuant}-vs-{yQuant}.pdf")


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
    sims.append(simParams("structures", hInd=31619, res=14, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=219612, res=15, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=311384, res=15, variant="ST15", redshift=5.5))
    sims.append(simParams("structures", hInd=844537, res=16, variant="ST15", redshift=5.5))

    # ------------

    # fig TODO: pressure vs. rho phase space diagram (see Schaye+26 Colibre Fig 8)
    # fig TODO: gas mass fraction of ISM gas in different phases, at e.g. z=10 and z=6 (bar plots?)
    # fig TODO: Kennicut-Schmidt relation, global or spatially resolved

    # star clusters: mass and size distributions
    if 0 or a:
        star_cluster_histogram(sims, quant="mass")
        star_cluster_histogram(sims, quant="size", sizefac=0.8)

    # clusters: size-mass relation
    if 0 or a:
        size_vs_mass(sims)

    # fig: stellar remnants: mass distribution
    if 0 or a:
        # config
        sim = simParams("structures", hInd=31619, res=15, variant="ST15", redshift=7.0)
        stellar_remnants(sim)

    # fig: mass fraction of PartType4 in stellar remnants
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

    # radial profiles - halo comparisons
    if 0 or a:
        haloIDs = [0] * len(sims)  # assume first
        opts = {"haloIDs": haloIDs, "xlog": True, "xlim": [-2.0, 1.5], "ylog": True}

        snapshot.profile(sims, ptType="gas", ptProperty="numdens", ylim=[-4.5, 4.0], scope="global", **opts)

        snapshot.profile(sims, ptType="stars", ptProperty="dens", ylim=[2.5, 11.0], scope="global", **opts)

        snapshot.profile(sims, ptType="gas", ptProperty="temp", ylim=[3.0, 6.0], scope="global", **opts)

        snapshot.profile(sims, ptType="gas", ptProperty="menc_vesc", ylim=[0.0, 1.7], scope="fof", **opts)

        snapshot.profile(sims, ptType="gas", ptProperty="cellsize_kpc", ylim=[-3.5, -0.5], scope="global", **opts)

    # radial profiles: 2d vs time
    if 0 or a:
        # evo
        opts = {"haloID": 0, "max_z": 10.0, "rlog": True, "rlim": [-2.0, 1.5]}

        for sim in sims:
            snapshot.profileEvo2d(
                sim,
                ptType="gas",
                ptProperty="numdens",
                clim=[-2.0, 3.0],
                clog=True,
                scope="global",
                ctName="magma",
                **opts,
            )

            snapshot.profileEvo2d(
                sim,
                ptType="stars",
                ptProperty="dens",
                clim=[3.0, 10.0],
                clog=True,
                scope="global",
                ctName="magma",
                **opts,
            )

            snapshot.profileEvo2d(
                sim,
                ptType="gas",
                ptProperty="temp",
                clim=[3.0, 6.0],
                clog=True,
                scope="global",
                ctName="thermal",
                **opts,
            )

            snapshot.profileEvo2d(
                sim,
                ptType="gas",
                ptProperty="vrad",
                clim=[-50.0, 50.0],
                clog=False,
                scope="global",
                ctName="curl",
                **opts,
            )

            snapshot.profileEvo2d(
                sim,
                ptType="gas",
                ptProperty="menc_vesc",
                clim=[0.0, 1.7],
                clog=True,
                scope="fof",
                ctName="afmhot",
                **opts,
            )

    # diagnostic: SFR debug
    if 0:
        diagnostic_sfr_jeans_mass(sims, haloID=0)
