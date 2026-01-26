"""
Analysis and plotting related to Grackle-based cooling functions/tables.
"""

from os.path import isfile

import h5py
import matplotlib.pyplot as plt
import numpy as np

from ..plot.config import figsize, linestyles, lw
from ..util.helper import xypairs_to_np
from ..util.units import units


def grackle_cooling(densities, metallicity, uvb="fg20_shielded", redshift=0.0, ssm=0, PE=False, plot=False):
    """Run a single-cell cooling model (vs time) with Grackle.

    This will initialize a single cell at a given temperature, iterate the cooling solver for a fixed time,
    and output the temperature vs. time.
    """
    from pygrackle import FluidContainer, chemistry_data, evolve_constant_density
    from pygrackle.utilities.physical_constants import cm_per_mpc, mass_hydrogen_cgs, sec_per_Myr

    tiny_number = 1e-20

    # Set initial values
    # redshift = 0.0
    # density     = 0.1 # linear 1/cm^3
    # metallicity = 1.0 # linear solar
    initial_temperature = 1.0e6  # K
    final_time = 100.0  # Myr

    # Set solver parameters
    my_chemistry = chemistry_data()
    my_chemistry.use_grackle = 1
    my_chemistry.with_radiative_cooling = 1
    my_chemistry.primordial_chemistry = 1  # MCST value = 1
    my_chemistry.metal_cooling = 1
    my_chemistry.UVbackground = 1
    my_chemistry.self_shielding_method = ssm  # MCST value = 3, but this leads to long-term cooling -> 0 K behavior (!)
    my_chemistry.H2_self_shielding = 0

    # set UV background
    basepath = "/u/dnelson/sims.structures/"
    if uvb == "hm12_unshielded":
        # old GRACKLE tables
        assert my_chemistry.self_shielding_method == 0, "Cannot use grackle self-shielding with unshielded HM12 table."
        grackle_data_file = bytearray(basepath + "grackle/grackle_data_files/input/CloudyData_UVB=HM2012.h5", "utf-8")
    elif uvb == "hm12_shielded":
        # old GRACKLE tables
        grackle_data_file = bytearray(
            basepath + "grackle/grackle_data_files/input/CloudyData_UVB=HM2012_shielded.h5", "utf-8"
        )
    elif uvb == "fg20_shielded":
        # new MCST tables
        grackle_data_file = bytearray(basepath + "arepo1_setups/grid_cooling_UVB=FG20.hdf5", "utf-8")
    elif uvb == "fg20_unshielded":
        # new MCST tables
        grackle_data_file = bytearray(basepath + "arepo1_setups/grid_cooling_UVB=FG20_unshielded.hdf5", "utf-8")

    # print(grackle_data_file)
    my_chemistry.grackle_data_file = grackle_data_file

    # my_chemistry.photoelectric_heating # yes if GRACKLE_PHOTOELECTRIC

    if PE:
        # my_chemistry.use_volumetric_heating_rate = 1 # yes for PE_MCS
        my_chemistry.photoelectric_heating = 1

        # volumetric_heating_rate = calculate_pe_heating_rate(sph_idx);
        # #define HABING_UNIT 5.29e-14 /*in erg cm^-3*/ // energy density of the average ISRF UV field

        u_draine = 8.93e-14  # Draine (1978) has u = 8.93e-14 # erg/cm^3 between 6-13.6 eV (G=1.68)
        G_0 = u_draine / 5.29e-14  # G0 = u_{6-13.6eV} / 5.29e-14 erg/cm^3

        G_0 *= 10.0  # i.e. chi_0 of Kim+23

        D = 0.001  # todo: replace with DGR() function
        n = 10.0  # cm^-3 # todo: inconsistent with densities array!
        rho = n * units.mass_proton  # g/cm^3
        temp = 1e2  # K
        cs = np.sqrt(units.gamma * units.boltzmann * temp / units.mass_proton)  # cm/s
        lambda_jeans = cs / (units.Gravity * rho) ** 0.5  # cm
        G_eff = G_0 * np.exp(-1.33e-21 * D * n * lambda_jeans)

        eps = np.min([0.041, 8.71e-3 * n**0.235])

        gamma_pe = 1.3e-24 * eps * D * G_eff * n  # erg s^-1 cm^-3

        print(G_0, G_eff, gamma_pe)

        my_chemistry.photoelectric_heating_rate = gamma_pe

    # Set units
    my_chemistry.comoving_coordinates = 0  # proper units
    my_chemistry.a_units = 1.0
    my_chemistry.a_value = 1 / (1 + redshift) / my_chemistry.a_units
    my_chemistry.density_units = mass_hydrogen_cgs  # rho = 1.0 is 1.67e-24 g
    my_chemistry.length_units = cm_per_mpc  # 1 Mpc in cm
    my_chemistry.time_units = sec_per_Myr  # 1 Myr in s
    my_chemistry.set_velocity_units()

    my_chemistry.initialize()

    # Set up array of 1-zone models
    fc = FluidContainer(my_chemistry, len(densities))

    fc["density"][:] = densities

    if my_chemistry.primordial_chemistry > 0:
        fc["HI"][:] = 0.76 * fc["density"]
        fc["HII"][:] = tiny_number * fc["density"]
        fc["HeI"][:] = (1.0 - 0.76) * fc["density"]
        fc["HeII"][:] = tiny_number * fc["density"]
        fc["HeIII"][:] = tiny_number * fc["density"]

    if my_chemistry.primordial_chemistry > 1:
        fc["H2I"][:] = tiny_number * fc["density"]
        fc["H2II"][:] = tiny_number * fc["density"]
        fc["HM"][:] = tiny_number * fc["density"]
        fc["de"][:] = tiny_number * fc["density"]

    if my_chemistry.primordial_chemistry > 2:
        fc["DI"][:] = 2.0 * 3.4e-5 * fc["density"]
        fc["DII"][:] = tiny_number * fc["density"]
        fc["HDI"][:] = tiny_number * fc["density"]

    if my_chemistry.metal_cooling == 1:
        fc["metal"][:] = metallicity * fc["density"] * my_chemistry.SolarMetalFractionByMass

    fc["x-velocity"][:] = 0.0
    fc["y-velocity"][:] = 0.0
    fc["z-velocity"][:] = 0.0

    fc["energy"][:] = initial_temperature / fc.chemistry_data.temperature_units
    fc.calculate_temperature()
    fc["energy"][:] *= initial_temperature / fc["temperature"]

    # timestepping safety factor
    safety_factor = 0.05

    # let gas cool at constant density
    data = evolve_constant_density(fc, final_time=final_time, safety_factor=safety_factor)

    # plot
    if plot:
        fig, (ax, ax2) = plt.subplots(nrows=2)

        ax.set_xlabel("Time [Myr]")
        ax.set_ylabel("T [K]")
        ax2.set_ylabel("$\\mu$")

        # time
        xx = data["time"].to("Myr")
        xx[0] += (xx[1] - xx[0]) / 10  # avoid t=0 disappearing in log

        # densities
        for i, dens in enumerate(densities):
            label = "n = %.1f [log cm$^{-3}$]" % np.log10(dens)
            ax.loglog(xx, data["temperature"][:, i], label=label)
            ax2.semilogx(xx, data["mu"][:, i])

        ax.legend(loc="upper right")

        labels = [
            r"Z = %.1f [log Z$_\odot]$" % np.log10(metallicity),
            "z = %s" % redshift,
            "(UVB = %s)" % uvb,
            "(self_shielding_method = %d)" % ssm,
        ]
        if PE:
            labels.append("(w/ PE heating)")
        handles = [plt.Line2D([0], [0], color="black", lw=0) for _ in range(len(labels))]
        # ax.add_artist(ax.legend(handles, labels, loc='lower left', handlelength=0))
        ax2.add_artist(ax2.legend(handles, labels, loc="upper left", handlelength=0))

        # fig.savefig(f'cooling_cell_dens{np.log10(density):.1f}_Z{np.log10(metallicity):.1f}.pdf')
        fig.savefig("cooling_cell.pdf")
        plt.close(fig)

    # look at final value, check convergence
    for i, dens in enumerate(densities):
        dT = np.abs(data["temperature"][-1, i] - data["temperature"][-2, i]) / data["temperature"][-1, i]
        dP = np.abs(data["pressure"][-1, i] - data["pressure"][-2, i]) / data["pressure"][-1, i]

        if dT > 1e-2 or dP > 1e-2:
            print(f"Warning: [i={i} dens={dens}] not converged: dT={dT:.2e}, dP={dP:.2e}")

    T_final = data["temperature"][-1, :].value  # K
    P_final = data["pressure"][-1, :].value  # dyn/cm^2

    return (T_final, P_final)


def _add_temp_curves(ax):
    """Overplot literature reference relations for T(n)."""
    # T(n) at Z=Zsun
    kim23_T = """9.41e-3, 1.01e+4
                6.99e-1, 6.69e+3
                1.78e+0, 2.93e+3
                3.63e+0, 9.76e+2
                8.48e+0, 3.23e+2
                2.85e+1, 1.21e+2
                1.07e+2, 5.99e+1
                6.54e+2, 3.67e+1"""

    Tn, T = xypairs_to_np(kim23_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls="-", label="Kim+23")  # (Z$_{\\rm sun}$)

    smith17_T = """2.71e-2, 9.49e+3
                 1.08e-1, 6.97e+3
                 4.46e-1, 4.51e+3
                 1.08e+0, 1.70e+3
                 2.49e+0, 4.97e+2
                 1.01e+1, 2.65e+2
                 4.27e+1, 1.69e+2
                 1.67e+2, 1.15e+2
                 5.71e+2, 7.98e+1"""

    Tn, T = xypairs_to_np(smith17_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls="--", label="Smith+17")  # (Z$_{\\rm sun}$)

    wolfire03_T = """1.81e-2, 9.28e+3
                4.50e-1, 7.12e+3
                1.11e+0, 3.50e+3
                2.16e+0, 1.14e+3
                4.55e+0, 4.16e+2
                1.42e+1, 1.44e+2
                5.46e+1, 6.88e+1
                2.18e+2, 4.34e+1
                6.03e+2, 3.40e+1"""

    Tn, T = xypairs_to_np(wolfire03_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls="-.", label="Wolfire+03")  # (Z$_{\\rm sun}$)

    bialy19_T = """1.75e-2, 1.03e+4
                1.20e+0, 7.62e+3
                2.47e+0, 4.26e+3
                3.63e+0, 1.86e+3
                9.44e+0, 3.62e+2
                6.49e+1, 8.18e+1
                5.19e+2, 3.72e+1
                9.02e+2, 3.28e+1"""

    Tn, T = xypairs_to_np(bialy19_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls=":", label="Bialy+19")  # (Z$_{\\rm sun}$)

    # T(n) at Z=0.1Zsun
    smith17_T = """2.06e-2, 1.08e+4
                1.03e-1, 9.41e+3
                3.61e-1, 8.67e+3
                3.86e+0, 6.27e+3
                1.42e+1, 3.62e+3
                3.60e+1, 1.17e+3
                1.03e+2, 4.18e+2
                4.03e+2, 2.20e+2"""

    Tn, T = xypairs_to_np(smith17_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#888", ls="--", label="Smith+17")  # (0.1Z$_{\\rm sun}$)

    bialy19_T = """2.47e-2, 1.02e+4
                8.53e-2, 9.35e+3
                4.69e+0, 4.86e+3
                8.21e+0, 1.55e+3
                1.35e+1, 5.27e+2
                3.76e+1, 1.59e+2
                1.32e+2, 7.04e+1
                3.77e+2, 4.53e+1"""

    Tn, T = xypairs_to_np(bialy19_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#888", ls=":", label="Bialy+19")  # (0.1Z$_{\\rm sun}$)

    kim23_T = """2.31e-2, 9.21e+3
            9.51e-2, 8.52e+3
            3.44e-1, 7.94e+3
            3.09e+0, 5.86e+3
            5.77e+0, 2.41e+3
            9.55e+0, 7.99e+2
            1.96e+1, 2.61e+2
            6.43e+1, 9.55e+1
            2.71e+2, 4.71e+1
            7.14e+2, 3.47e+1"""

    Tn, T = xypairs_to_np(kim23_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#888", ls="-", label="Kim+23")  # (0.1Z$_{\\rm sun}$)

    # including UVB (Kim+23 Fig 19)
    kim23_T = """1.04e-2, 8.41e3,
                2.72e-2, 7.34e+3
                8.99e-2, 1.90e+3
                1.96e-1, 6.36e+2
                5.85e-1, 1.56e+2
                2.95e+0, 5.62e+1
                1.65e+1, 3.34e+1
                9.21e+1, 3.10e+1
                4.06e+2, 3.10e+1"""

    Tn, T = xypairs_to_np(kim23_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls=(0, (1, 1)), label="Kim+23 (PI+PE)")

    kim23_T = """3.57e-2, 9.31e+3
                1.00e+0, 8.38e+3
                4.82e+0, 5.80e+3
                1.39e+1, 1.23e+3
                3.35e+1, 3.20e+2
                1.33e+2, 9.42e+1
                4.99e+2, 3.55e+1"""

    Tn, T = xypairs_to_np(kim23_T)
    ax.plot(np.log10(Tn), np.log10(T), color="#000", ls=(0, (5, 5)), label="Kim+23 (PI only)")


def _add_pres_curves(ax):
    """Overplot literature reference relations for P(n)."""
    # P(n) at Z=Zsun
    smith17_P = """2.69e-2, 2.55e+2
                5.77e-1, 2.40e+3
                7.84e-1, 2.47e+3
                1.11e+0, 1.77e+3
                1.43e+0, 1.03e+3
                2.45e+0, 1.21e+3
                3.65e+0, 1.47e+3
                4.87e+0, 1.71e+3
                2.32e+1, 4.76e+3
                3.48e+1, 6.34e+3
                1.68e+2, 1.98e+4
                2.96e+2, 2.94e+4"""

    Pn, P = xypairs_to_np(smith17_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls="--")  # , label='Smith+17 (Z$_{\\rm sun}$)'

    wolfire03_P = """1.04e-2, 1.25e+2
                    9.81e-2, 8.63e+2
                    5.88e-1, 4.23e+3
                    1.28e+0, 4.14e+3
                    2.50e+0, 2.41e+3
                    5.81e+0, 2.07e+3
                    1.08e+1, 2.13e+3
                    2.25e+1, 2.58e+3
                    4.72e+1, 3.78e+3
                    9.41e+1, 5.73e+3
                    1.75e+2, 8.60e+3
                    9.65e+2, 3.55e+4"""

    Pn, P = xypairs_to_np(wolfire03_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls="-.")  # , label='Wolfire+03 (Z$_{\\rm sun}$)'

    kim23_P = """2.10e-2, 2.63e+2
            4.24e-2, 5.06e+2
            8.58e-2, 9.67e+2
            1.73e-1, 1.86e+3
            3.42e-1, 3.44e+3
            8.47e-1, 6.41e+3
            1.98e+0, 5.79e+3
            4.38e+0, 3.72e+3
            9.69e+0, 3.03e+3
            2.07e+1, 3.28e+3
            9.37e+1, 6.99e+3
            2.12e+2, 1.17e+4
            4.46e+2, 2.02e+4"""

    Pn, P = xypairs_to_np(kim23_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls="-")  # , label='Kim+23 (Z$_{\\rm sun}$)'

    bialy19_P = """1.04e-2, 1.25e+2
                6.78e-2, 6.50e+2
                1.92e+0, 1.24e+4
                4.24e+0, 5.65e+3
                1.10e+1, 3.40e+3
                3.24e+1, 4.02e+3
                9.65e+2, 3.10e+4"""

    Pn, P = xypairs_to_np(bialy19_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls=":")  # , label='Bialy+19 (Z$_{\\rm sun}$)'

    smith17_P = """1.88e-2, 2.02e+2
                9.20e+0, 4.54e+4
                1.74e+1, 5.84e+4
                5.09e+1, 3.36e+4
                1.02e+2, 4.21e+4
                7.33e+2, 1.28e+5"""

    Pn, P = xypairs_to_np(smith17_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#888", ls="--")  # , label='Smith+17 (0.1Z$_{\\rm sun}$)'

    bialy19_P = """2.29e-2, 2.18e+2
                9.09e-1, 6.77e+3
                4.29e+0, 2.55e+4
                6.10e+0, 1.87e+4
                2.45e+1, 5.24e+3
                6.02e+1, 6.04e+3
                1.47e+2, 8.95e+3
                6.14e+2, 2.21e+4"""

    Pn, P = xypairs_to_np(bialy19_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#888", ls=":")  # , label='Bialy+19 (0.1Z$_{\\rm sun}$)'

    kim23_P = """2.21e-2, 2.68e+2
                1.81e+0, 1.52e+4
                3.59e+0, 2.33e+4
                1.45e+1, 6.66e+3
                4.07e+1, 6.15e+3
                1.30e+2, 9.54e+3
                7.21e+2, 2.91e+4"""

    Pn, P = xypairs_to_np(kim23_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#888", ls="-")  # , label='Kim+23 (0.1Z$_{\\rm sun}$)'

    # including UVB (Kim+23 Fig 19)
    kim23_P = """1.00e-2, 1.05e+2
                3.72e-2, 2.87e+2
                6.02e-2, 2.50e+2
                1.81e-1, 1.13e+2
                4.03e-1, 9.66e+1
                1.37e+0, 1.32e+2
                8.86e+0, 3.59e+2
                9.82e+2, 3.32e+4"""

    Pn, P = xypairs_to_np(kim23_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls=(0, (1, 1)))  # , label='Kim+23 (PI+PE)'

    kim23_P = """9.82e-3, 2.10e+2
            2.79e+0, 2.84e+4
            4.84e+0, 3.69e+4
            8.25e+0, 3.05e+4
            2.40e+1, 1.31e+4
            5.33e+1, 1.10e+4
            2.64e+2, 1.82e+4
            1.00e+3, 3.88e+4"""

    Pn, P = xypairs_to_np(kim23_P)
    ax.plot(np.log10(Pn), np.log10(P), color="#000", ls=(0, (5, 5)))  # , label='Kim+23 (PI only)'


def _load_equil_curves(metallicity, redshift, ssm):
    """Helper. Load and/or compute equilibrium temperature and pressure vs. density curves."""
    savefile = f"cache/equil_vs_dens_Z{np.log10(metallicity):.0f}_z{redshift:.0f}_ssm{ssm}.hdf5"

    densities = np.linspace(-3.0, 3.0, 40)  # log(1/cm^3)

    if metallicity == 1.0:
        densities = np.linspace(-3.0, 2.0, 20)  # log(1/cm^3)

    uvbs = ["hm12_shielded", "hm12_unshielded", "fg20_shielded", "fg20_unshielded"]
    if ssm > 0:
        uvbs = ["hm12_shielded", "fg20_unshielded", "fg20_shielded"]

    if not isfile(savefile):
        temp = {}
        pres = {}

        # loop over UVBs (and metallicites), compute all densities at once
        for uvb in uvbs:
            temp[uvb], pres[uvb] = grackle_cooling(10**densities, metallicity, uvb=uvb, redshift=redshift, ssm=ssm)

        print(metallicity, uvb, temp[uvb].min(), temp[uvb].max())

        # save
        with h5py.File(savefile, "w") as f:
            for uvb in uvbs:
                f["T_" + uvb] = temp[uvb]
                f["P_" + uvb] = pres[uvb]
            f["densities"] = densities
            f["metallicity"] = np.array(metallicity)
        print(f"Saved: [{savefile}].")
    else:
        with h5py.File(savefile, "r") as f:
            temp = {}
            pres = {}
            for key in f:
                if key.startswith("T_"):
                    temp[key[2:]] = f[key][()]
                if key.startswith("P_"):
                    pres[key[2:]] = f[key][()]
            densities = f["densities"][()]
            metallicity = f["metallicity"][()]
        print(f"Loaded: [{savefile}].")

    return temp, pres, densities, uvbs


def grackle_equil(ssm=3):
    """Plot equilibrium temperature curve as a function of density (varying UVBs/SSMs), at fixed Z and z."""
    # https://arxiv.org/pdf/2411.07282 (Fig 1)
    # "Photochemistry and Heating/Cooling of the Multiphase Interstellar Medium with UV
    #  Radiative Transfer for Magnetohydrodynamic Simulations" (Fig 17, 19)
    metallicity = 0.1  # linear solar
    redshift = 6.0

    # ssm = 1 # grackle.self_shielding_method (MCST value = 3)

    # load
    temp, pres, densities, uvbs = _load_equil_curves(metallicity, redshift, ssm)

    # plot
    fig, (ax1, ax2) = plt.subplots(figsize=(figsize[0] * 1.4, figsize[1] * 0.9), ncols=2)
    ax1.set_xlabel(r"Density [log cm$^{-3}$]")
    ax1.set_ylabel(r"Temperature [log K]")
    ax1.set_ylim([0.5, 5.0])

    ax2.set_xlabel(r"Density [log cm$^{-3}$]")
    ax2.set_ylabel(r"P / k$_{\rm B}$ [log K cm$^{-3}$]")
    ax2.set_ylim([1.0, 5.5])

    _add_temp_curves(ax1)
    _add_pres_curves(ax2)

    # plot our results, as a function of UVB
    for uvb in uvbs:
        T = np.log10(temp[uvb])
        P = np.log10(pres[uvb] / units.boltzmann)  # dyn/cm^2 / (dyn*cm/K) = K/cm^3

        ax1.plot(densities, T, ls="-", lw=lw + 1)
        ax2.plot(densities, P, ls="-", lw=lw + 1, label=uvb)

    ax2.legend(loc="upper left")

    labels = [
        r"Z = %.1f [log Z$_\odot]$" % np.log10(metallicity),
        "z = %s" % redshift,
        "self_shielding_method = %d" % ssm,
    ]
    handles = [plt.Line2D([0], [0], color="black", lw=0) for _ in range(len(labels))]
    ax1.add_artist(ax1.legend(handles, labels, loc="upper right", handlelength=0))

    ax1.legend(loc="lower left")

    fig.savefig(f"equil_vs_dens_Z{np.log10(metallicity):.0f}_z{redshift:.0f}_ssm{ssm}.pdf")
    plt.close(fig)


def grackle_equil_vs_Zz():
    """Plot equilibrium temperature curve as a function of density (varying Z, z) at fixed UVB/SSM."""
    redshifts = [0.0, 6.0]
    metallicities = [1.0, 0.1, 0.001]  # linear solar

    ssm = 3  # grackle.self_shielding_method (MCST value = 3)
    uvb = "fg20_unshielded"

    # plot
    fig, (ax1, ax2) = plt.subplots(figsize=(figsize[0] * 0.75, figsize[1] * 1.5), nrows=2)
    ax1.set_xlabel(r"Density [log cm$^{-3}$]")
    ax1.set_ylabel(r"Temperature [log K]")
    ax1.set_ylim([0.4, 4.1])
    ax1.set_xlim([-3.0, 3.0])

    ax2.set_xlabel(r"Density [log cm$^{-3}$]")
    ax2.set_ylabel(r"P / k$_{\rm B}$ [log K cm$^{-3}$]")
    ax2.set_ylim([0.7, 5.5])
    ax2.set_xlim([-3.0, 3.0])

    _add_temp_curves(ax1)
    _add_pres_curves(ax2)

    # loop over metallicities and redshifts
    for i, redshift in enumerate(redshifts):
        ax1.set_prop_cycle(None)
        ax2.set_prop_cycle(None)

        for _j, metallicity in enumerate(metallicities):
            temp, pres, densities, _ = _load_equil_curves(metallicity, redshift, ssm)

            T = np.log10(temp[uvb])
            P = np.log10(pres[uvb] / units.boltzmann)  # dyn/cm^2 / (dyn*cm/K) = K/cm^3

            label = r"log(Z/Z$_{\rm sun}$) = %.0f" % (np.log10(metallicity)) if i == 0 else ""
            ls = linestyles[i * 2]
            ax1.plot(densities, T, ls=ls, lw=lw + 1)
            ax2.plot(densities, P, ls=ls, lw=lw + 1, label=label)

    handles, labels = ax2.get_legend_handles_labels()
    handles += [plt.Line2D([0], [0], color="black", ls=linestyles[i * 2]) for i in range(len(redshifts))]
    labels += ["z = %s" % z for z in redshifts]

    ax2.legend(handles, labels, loc="upper left")
    ax1.legend(loc="lower left")

    fig.savefig(f"equil_vs_dens_{uvb}_ssm{ssm}.pdf")
    plt.close(fig)
