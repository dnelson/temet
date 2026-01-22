"""Idealized ICs: analytical solution of the 3D Sedov blast wave at some time t. Inspired by the version from SWIFT."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma


def calc_a(g, nu=3):
    """Exponents of the polynomials of the sedov solution.

    Args:
      g (float): the polytropic gamma
      nu (int): the dimension
    """
    a = [0] * 8

    a[0] = 2.0 / (nu + 2)
    a[2] = (1 - g) / (2 * (g - 1) + nu)
    a[3] = nu / (2 * (g - 1) + nu)
    a[5] = 2 / (g - 2)
    a[6] = g / (2 * (g - 1) + nu)

    a[1] = (((nu + 2) * g) / (2.0 + nu * (g - 1.0))) * ((2.0 * nu * (2.0 - g)) / (g * (nu + 2.0) ** 2) - a[2])
    a[4] = a[1] * (nu + 2) / (2 - g)
    a[7] = (2 + nu * (g - 1)) * a[1] / (nu * (2 - g))
    return a


def calc_beta(v, g, nu=3):
    """Beta values for the sedov solution (coefficients of the polynomials of the similarity variables).

    Args:
      v (float): the similarity variable
      g (float): the polytropic gamma
      nu (int): the dimension
    """
    x = np.array(
        (
            0.25,
            (g / (g - 1)) * 0.5,
            -(2 + nu * (g - 1)) / 2.0 / ((nu + 2) * (g + 1) - 2 * (2 + nu * (g - 1))),
            -0.5 / (g - 1),
        ),
        dtype=np.float64,
    )

    beta = (nu + 2) * (g + 1) * x

    beta = np.outer(beta, v)

    y = np.array(
        (0.0, -1.0 / (g - 1), (nu + 2) / ((nu + 2) * (g + 1) - 2.0 * (2 + nu * (g - 1))), 1.0 / (g - 1)),
        dtype=np.float64,
    ).reshape((4, 1))

    beta += (g + 1) * y

    return beta


def sedov(t, E0, rho0, g=5 / 3, n=1000, nu=3):
    """Solve the sedov problem.

    Args:
      t (float): time
      E0 (float): initial energy
      rho0 (float): initial density
      g (float): polytropic gas gamma
      n (int): number of points to compute solution on
      nu (int): dimensionality
    """
    # the similarity variable
    v_min = 2.0 / ((nu + 2) * g)
    v_max = 4.0 / ((nu + 2) * (g + 1))

    v = v_min + np.arange(n) * (v_max - v_min) / (n - 1.0)

    a = calc_a(g, nu)
    beta = calc_beta(v, g=g, nu=nu)
    lbeta = np.log(beta)

    r = np.exp(-a[0] * lbeta[0] - a[2] * lbeta[1] - a[1] * lbeta[2])
    rho = ((g + 1.0) / (g - 1.0)) * np.exp(a[3] * lbeta[1] + a[5] * lbeta[3] + a[4] * lbeta[2])
    p = np.exp(nu * a[0] * lbeta[0] + (a[5] + 1) * lbeta[3] + (a[4] - 2 * a[1]) * lbeta[2])
    u = beta[0] * r * 4.0 / ((g + 1) * (nu + 2))
    p *= 8.0 / ((g + 1) * (nu + 2) * (nu + 2))

    # we have to take extra care at v=v_min, since this can be a special point.
    # It is not a singularity, however, the gradients of our variables (wrt v) are.
    # r -> 0, u -> 0, rho -> 0, p-> constant
    u[0] = 0.0
    rho[0] = 0.0
    r[0] = 0.0
    p[0] = p[1]

    # volume of an n-sphere
    vol = (np.pi ** (nu / 2.0) / gamma(nu / 2.0 + 1)) * np.power(r, nu)

    # note we choose to evaluate the integral in this way because the
    # volumes of the first few elements (i.e near v=vmin) are shrinking
    # very slowly, so we dramatically improve the error convergence by
    # finding the volumes exactly. This is most important for the
    # pressure integral, as this is on the order of the volume.

    # (dimensionless) energy of the model solution
    de = rho * u * u * 0.5 + p / (g - 1)
    # integrate (trapezium rule)
    q = np.inner(de[1:] + de[:-1], np.diff(vol)) * 0.5

    # the factor to convert to this particular problem (units: velocity)
    # e.g. [s^3 g/cm^3 erg^-1 = s^3 g/cm^3 s^2/g/cm^2 = s^5/cm^5], to the (-1/5) power --> [cm/s]
    fac = (q * (t**nu) * rho0 / E0) ** (-1.0 / (nu + 2))

    # shock speed
    shock_speed = fac * (2.0 / (nu + 2))
    # rho_s = ((g + 1) / (g - 1)) * rho0
    r_s = shock_speed * t * (nu + 2) / 2.0
    # p_s = (2.0 * rho0 * shock_speed * shock_speed) / (g + 1)
    # u_s = (2.0 * shock_speed) / (g + 1)

    r *= fac * t
    u *= fac
    p *= fac * fac * rho0
    rho *= rho0

    # re-compute volume with units, and convert to mass
    # vol_spheres = (np.pi ** (nu / 2.0) / gamma(nu / 2.0 + 1)) * np.power(r, nu) # cm^3
    # vol_shells = np.hstack( (0,vol_spheres[1:] - vol_spheres[:-1]))
    # mass = vol_shells*rho # g
    mass = 0

    return r, p, rho, u, r_s, vol, mass  # p_s, rho_s, u_s, shock_speed


def solution_cgs(time_kyr=1.0):
    """Compute the Sedov solution."""
    # config
    ga = 5 / 3

    if 0:
        rho_0 = 1.0  # Background Density
        P_0 = 1.0e-6  # Background Pressure (not used in solution)
        E_0 = 1.0  # Energy of the explosion

        time = 0.1

    # xeno:
    if 1:
        n0 = 1.0  # cm^-3
        P0 = 0.1  # code
        E0 = 1.0  # erg

        # time_kyr = 0.5

        # unit system
        mass_proton_g = 1.672622e-24
        s_in_yr = 3.155693e7
        pc_in_cm = 3.085680e18
        # boltzmann_cgs = 1.380650e-16  # cm^2 g s^-2 K^-1

        UnitLength_in_cm = 3.085678e18  # 1 pc
        UnitMass_in_g = 1.989e31  # 0.01 msun
        UnitVelocity_in_cm_per_s = 1.0e5  # 1 km/s

        UnitTime_in_s = UnitLength_in_cm / UnitVelocity_in_cm_per_s
        UnitPressure_in_cgs = UnitMass_in_g / UnitLength_in_cm / UnitTime_in_s**2.0

        # cgs units
        rho_0 = n0 * mass_proton_g  # g/cm^3
        P_0 = P0 * UnitPressure_in_cgs  # ba = g cm^-1 s^-2
        E_0 = E0 * 1e51  # erg = cm^2 g s^-2

        time = time_kyr * 1000 * s_in_yr  # s

    print(f"{n0 = }, {P0 = }, {E0 = }, {time_kyr = }")
    print(f" {rho_0 = }, {P_0 = }, {E_0 = }, {time = }")

    # The main properties of the solution
    r_s, P_s, rho_s, v_s, r_shock, vol, mass = sedov(time, E_0, rho_0, g=ga)

    # Append points for after the shock (for display only)
    r_s = np.insert(r_s, np.size(r_s), [r_shock, r_shock * 1.5])
    rho_s = np.insert(rho_s, np.size(rho_s), [rho_0, rho_0])
    P_s = np.insert(P_s, np.size(P_s), [1e-10, 1e-10])  # [P_0, P_0]
    v_s = np.insert(v_s, np.size(v_s), [1, 1])  # [0, 0]
    mass = np.insert(mass, np.size(mass), [vol[-1] * rho_0, vol[-1] * rho_0])

    # Additional arrays
    rho_s[0] = rho_s[rho_s > 0].min()  # remove central zero

    u_s = P_s / (rho_s * (ga - 1.0))  # internal energy [cm^2/s^2]
    # s_s = P_s / rho_s**ga  # entropic function

    # units: cgs -> more useful
    r_s /= pc_in_cm  # pc
    r_shock /= pc_in_cm  # pc
    v_s /= 1e5  # km/s
    rho_s /= mass_proton_g  # 1/cm^3
    P_s *= 1e8  # 1e8 Ba

    return r_s, v_s, rho_s, P_s, u_s, r_shock


def plot_timeseries():
    """Plot the Sedov solution as a function of time."""
    fig, axes = plt.subplots(figsize=(18, 12), nrows=2, ncols=2)

    times_kyr = [0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 8.5, 10, 12.2, 14.7]

    for i, time_kyr in enumerate(times_kyr):
        # calculate
        r_s, v_s, rho_s, P_s, u_s, r_shock = solution_cgs(time_kyr)

        # velocity profile
        label = "t = %.2f kyr" % time_kyr
        axes[0, 0].plot(r_s, v_s, "-", alpha=0.8, lw=2, label=label)
        axes[0, 0].set_xlabel("Radius [pc]")
        axes[0, 0].set_ylabel("Radial Velocity [km/s]")
        axes[0, 0].set_xlim(0, 1.3 * r_shock)
        if i == 0:
            axes[0, 0].set_ylim(0.5, v_s.max() * 1.1)

        # density profile
        axes[0, 1].plot(r_s, rho_s, "-", alpha=0.8, lw=2, label=label)
        axes[0, 1].set_xlabel("Radius [pc]")
        axes[0, 1].set_ylabel("Density [$\\rm{cm^{-3}}$]")
        axes[0, 1].set_xlim(0, 1.3 * r_shock)
        axes[0, 1].set_ylim(-0.2, 5.2)

        # pressure profile
        axes[1, 0].plot(r_s, P_s, "-", alpha=0.8, lw=2)
        axes[1, 0].set_xlabel("Radius [pc]")
        axes[1, 0].set_ylabel("Pressure [$10^8$ Ba]")
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_xlim(0, 1.3 * r_shock)
        if i == 0:
            axes[1, 0].set_ylim(1e-1, P_s.max() * 1.1)

        # internal energy profile
        axes[1, 1].plot(r_s, u_s, "-", alpha=0.8, lw=2)
        axes[1, 1].set_xlabel("Radius [pc]")
        axes[1, 1].set_ylabel("Internal Energy [cm$^2$/s$^2$]")
        axes[1, 1].set_yscale("log")
        axes[1, 1].set_xlim(0, 1.3 * r_shock)
        if i == 0:
            axes[1, 1].set_ylim(1e14, u_s.max() * 1.1)

    axes[0, 0].legend(loc="upper right")
    axes[0, 1].legend(loc="upper left")

    fig.savefig("sedov.pdf")
    plt.close(fig)


def plot_rshock_vs_time():
    """Plot the shock radius as a function of time."""
    # calculate
    N = 50
    times_kyr = np.linspace(0, 10, N)
    r_shock = np.zeros(N, dtype="float32")

    for i, t_kyr in enumerate(times_kyr):
        r_s, v_s, rho_s, P_s, u_s, r_shock_t = solution_cgs(t_kyr)

        r_shock[i] = r_shock_t

    # plot
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(times_kyr, r_shock, lw=2.0)
    ax.set_xlabel("Time [kyr]")
    ax.set_ylabel("Shock Radius [pc]")

    fig.savefig("sedov_rshock.pdf")
    plt.close(fig)
