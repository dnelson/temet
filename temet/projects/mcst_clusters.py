"""
MCST: stellar clusters paper.

https://arxiv.org/abs/xxxx.xxxxx
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

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
)
from temet.util import simParams
from temet.util.helper import cache, logZeroNaN
from .mcst import _get_existing_sims, _zoomSubhaloIDsToPlot


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


def star_cluster_histogram(sims, quant, haloID=0):
    """Plot the mass function, or size distribution, of star clusters."""
    # start plot
    fig, ax = plt.subplots()
    if quant == "mass":
        ax.set_xlabel(r"Star Cluster Mass [ log M$_{\odot}$ ]")
    if quant == "size":
        ax.set_xlabel(r"Star Cluster Size $R_{1/2}$ [ log kpc ]")
    ax.set_ylabel("Number of Star Clusters")
    ax.set_yscale("log")

    for sim in sims:
        haloID = 0

        sub_haloIDs = sim.subhalos("SubhaloGrNr")
        sub_ids = np.where(sub_haloIDs == haloID)[0][1:]

        mstar = sim.subhalos("mstar_tot")[sub_ids]
        rhalf = sim.subhalos("rhalf_stars")[sub_ids]
        mdm = sim.subhalos("mdm_tot")[sub_ids]

        # make selection
        w = np.where((mstar > 0) & (mdm == 0))[0]

        print(
            f"{sim.simName}: Found {len(w)} star clusters (mstar > 0, mdm=0) in halo {haloID} of {len(sub_ids)} subhalos."
        )

        w_total = np.where(mstar > 0)[0]
        print(f"{sim.simName}: Found {len(w_total)} subhalos with mstar > 0 (including those with mdm > 0).")
        print(f"--> {len(w_total) - len(w)} subhalos with mstar > 0 but mdm > 0 (sats?).")
        print(f"--> {len(sub_ids) - len(w_total)} subhalos with mstar = 0 (dark).")

        # histogram quantity
        if quant == "mass":
            h_quant = np.log10(mstar[w])
        if quant == "size":
            h_quant = np.log10(rhalf[w])

        label = f"{sim.simName} (N={len(w)}/{mstar.size})"
        # ax.hist(h_quant, bins=30, histtype="step", label=label)

        h = np.histogram(h_quant, bins=30)  # range=[min,max]
        ax.stairs(*h, fill=True, label=label)

    if quant == "mass":
        min_mass = np.log10(20 * sim.units.codeMassToMsun(sim.targetGasMass))
        ax.plot([min_mass, min_mass], ax.get_ylim(), color="black", linestyle="--", label="20x targetGasMass")
    if quant == "size":
        pass

    ax.legend(loc="upper right")

    fig.savefig(f"star_cluster_histo_{quant}.pdf")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------


def paperPlots(a=False):
    """Plots for MCST intro paper. (if a == True, make all figures)."""
    # list of sims to include
    variants = ["ST15"]  # ['ST15c','ST15m','ST15s']
    res = [14, 15, 16]
    hInds = [1958, 5072, 15581, 23908, 31619, 73172, 219612, 311384, 446076, 539722, 844537]
    # hInds = [31619, 73172]
    redshift = 5.5

    # if (all == False), only dz < 0.1 matches
    # if (single == True), only the highest available res of each halo
    sims = _get_existing_sims(variants, res, hInds, redshift, all=False, single=True)

    # ------------

    # fig TODO: pressure vs. rho phase space diagram (see Schaye+26 Colibre Fig 8)
    # fig TODO: gas mass fraction of ISM gas in different phases, at e.g. z=10 and z=6 (bar plots?)
    # fig TODO: Kennicut-Schmidt relation, global or spatially resolved

    # star cluster histogram test
    if 1 or a:
        sim = simParams("structures", hInd=31619, res=14, variant="ST14", redshift=5.5)
        star_cluster_histogram([sim], quant="mass")

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
