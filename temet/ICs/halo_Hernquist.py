"""Idealized ICs: hydrostatic equilibrium gas sphere in a Hernquist potential (live DM optional)."""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..ICs.utilities import write_ic_file


def create_ics(filename="ics.hdf5"):
    """Create idealized ICs."""
    # input parameters
    add_halo = False  # add halo particles?
    add_gas = True  # add gas particles?

    N_gas = 200000
    N_halo = 200000
    gas_frac = 0.1
    halo_fac = 1.0 / (1.0 - gas_frac)  # increase halo particles by that factor (for runs with NO_GAS_SELFGRAVITY)

    HQ_M = 1e3  # Hernquist halo mass [10^10 Msun units?]
    HQ_c = 7.0  # Hernquist concentration

    gas_R0 = 0.01  # gas core softening [HQ_a]
    R_min_halo = 1e-5  # minimum halo sampling radius [HQ_a]
    R_max_halo = 100.0  # maximum halo sampling radius [HQ_a]
    R_min_gas = 1e-5  # minimum gas sampling radius [HQ_a]
    R_max_gas = 5.0  # maximum gas sampling radius [HQ_a]
    R_bins = 1000  # number of interpolation points for function evaluation/inversion

    # derived
    gas_mass = HQ_M * gas_frac / N_gas
    halo_mass = halo_fac * HQ_M * (1.0 - gas_frac) / N_halo
    G = np.nan  # TODO, replace with G in appropriate units
    HQ_a = (G * HQ_M / (100 * 0.1 * 0.1)) ** (1.0 / 3.0) / HQ_c * np.sqrt(2 * (np.log(1 + HQ_c) - HQ_c / (1 + HQ_c)))

    # Interpolation parameters
    INTERPOL_BINS = R_bins
    INTERPOL_R_MIN_HALO = HQ_a * R_min_halo  # minimum halo sampling radius (halo cut below)
    INTERPOL_R_MAX_HALO = HQ_a * R_max_halo  # maximum halo sampling radius (halo cut above)
    INTERPOL_R_MIN_GAS = HQ_a * R_min_gas  # minimum gas sampling radius (gas cut below)
    INTERPOL_R_MAX_GAS = HQ_a * R_max_gas  # maximum gas sampling radius (gas cut above)

    np.random.seed(424242)

    # helper functions
    def GasRho(r):
        x0 = gas_R0
        x = r / HQ_a
        return HQ_M / (2 * np.pi * HQ_a**3) * 1 / (x + x0) * 1 / (x + 1) ** 3

    def HaloRho(r):
        x = r / HQ_a
        return HQ_M / (2 * np.pi * HQ_a**3) * 1 / x * 1 / (x + 1) ** 3

    def Rho(r):
        return gas_frac * GasRho(r) + (1.0 - gas_frac) * HaloRho(r)

    def GasMass(r):
        x0 = gas_R0
        x = r / HQ_a
        return (
            HQ_M
            * ((1 - x0) * x * (x0 * (2 + 3 * x) - x) / (1 + x) ** 2 + 2 * x0**2.0 * np.log(x0 * (1 + x) / (x0 + x)))
            / (x0 - 1) ** 3.0
        )

    def HaloMass(r):
        x = r / HQ_a
        return HQ_M * x**2.0 / (1 + x) ** 2.0

    def Mass(r):
        return gas_frac * GasMass(r) + (1.0 - gas_frac) * HaloMass(r)

    def Sigma_Integrand(r):
        return G * Mass(r) * Rho(r) / r**2.0

    def Sigma(r):
        return np.sqrt(quad(Sigma_Integrand, r, INTERPOL_R_MAX_GAS, epsrel=0.1)[0] / Rho(r))

    # vectorize functions
    vecSigma = np.vectorize(Sigma)
    # vecRho = np.vectorize(Rho)
    vecGasMass = np.vectorize(GasMass)
    vecHaloMass = np.vectorize(HaloMass)
    # vecMass = np.vectorize(Mass)

    # invert function: GasMass^-1 = GasRadius
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX_GAS / INTERPOL_R_MIN_GAS) / INTERPOL_BINS
        + np.log(INTERPOL_R_MIN_GAS)
    )
    mass_bins_gas = vecGasMass(radial_bins)
    GasRadius = interp1d(mass_bins_gas, radial_bins)

    # invert function: HaloMass^-1 = HaloRadius
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX_HALO / INTERPOL_R_MIN_HALO) / INTERPOL_BINS
        + np.log(INTERPOL_R_MIN_HALO)
    )
    mass_bins_halo = vecHaloMass(radial_bins)
    HaloRadius = interp1d(mass_bins_halo, radial_bins)

    # interpolate sigma
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX_GAS / INTERPOL_R_MIN_GAS) / INTERPOL_BINS
        + np.log(INTERPOL_R_MIN_GAS)
    )
    sigma_bins = vecSigma(radial_bins)
    InterpolSigma = interp1d(radial_bins, sigma_bins)

    # generate random positions
    radius_gas = GasRadius(np.random.uniform(N_gas) * mass_bins_gas.max())
    phi_gas = 2.0 * np.pi * np.random.random_sample(N_gas)
    theta_gas = np.arcsin(2.0 * np.random.random_sample(N_gas) - 1.0)
    x_gas = radius_gas * np.cos(theta_gas) * np.cos(phi_gas)
    y_gas = radius_gas * np.cos(theta_gas) * np.sin(phi_gas)
    z_gas = radius_gas * np.sin(theta_gas)

    radius_halo = HaloRadius(np.random.uniform(N_halo) * mass_bins_halo.max())
    phi_halo = 2.0 * np.pi * np.random.random_sample(N_halo)
    theta_halo = np.arcsin(2.0 * np.random.random_sample(N_halo) - 1.0)
    x_halo = radius_halo * np.cos(theta_halo) * np.cos(phi_halo)
    y_halo = radius_halo * np.cos(theta_halo) * np.sin(phi_halo)
    z_halo = radius_halo * np.sin(theta_halo)

    utherm = 1.5 * InterpolSigma(radius_gas) ** 2.0

    # write
    massarr = np.array([0, 0, 0, 0, 0, 0], dtype="float64")
    particles = {}

    if add_gas:
        pos_gas = np.array([x_gas, y_gas, z_gas]).T
        vel_gas = np.zeros([N_gas, 3])
        ids_gas = np.arange(1, N_gas + 1)

        pt0 = {"Coordinates": pos_gas, "Velocities": vel_gas, "InternalEnergy": utherm, "ParticleIDs": ids_gas}

        massarr[0] = gas_mass
        particles["PartType0"] = pt0

    if add_halo:
        pos_halo = np.array([x_halo, y_halo, z_halo]).T
        vel_halo = np.zeros([N_halo, 3])
        ids_halo = np.arange(1 + N_gas, N_gas + N_halo + 2)

        pt1 = {"Coordinates": pos_halo, "Velocities": vel_halo, "ParticleIDs": ids_halo}

        massarr[1] = halo_mass
        particles["PartType1"] = pt1

    boxSize = 2.1 * radius_gas.max()  # todo check

    write_ic_file(filename, particles, MassTable=massarr, boxSize=boxSize)
