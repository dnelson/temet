"""Idealized ICs: hydrostatic gas in a NFW potential (with Monte Carlo tracers; no live halo)."""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d

from ..ICs.utilities import write_ic_file


def create_ics(filename="ics.hdf5"):
    """Create idealized ICs."""
    # parameters
    N_gas = 5000
    N_halo = 1000000
    N_tr = 5000
    gas_frac = 0.1

    NFW_M200 = 1e2
    NFW_c = 9.0

    gas_R0 = 0.1
    Lambda = 0.1  # 0.0 for no rotation
    S_omega = 1.0  # 0.0 for rigid body rotation
    R_min = 1e-5  # minimum sampling radius [r200]
    R_max = 1.0  # maximum sampling radius [r200]
    R_bins = 1000  # number of interpolation points for function evaluation/inversion
    Rcut = 5.0  # cut the halo within that radius

    G = np.nan  # TODO: replace with G in appropriate units
    Hubble = np.nan  # TODO
    RhoCrit = np.nan  # TODO

    np.random.seed(424242)

    # NFW parameters
    NFW_r200 = ((NFW_M200 * G) / (100.0 * Hubble**2.0)) ** (1.0 / 3.0)
    NFW_rs = NFW_r200 / NFW_c
    NFW_delta = (200.0 / 3.0) * NFW_c**3.0 / (np.log(1.0 + NFW_c) - (NFW_c / (1.0 + NFW_c)))
    Rcut *= NFW_r200

    # derived
    gas_mass = NFW_M200 * gas_frac / N_gas
    halo_mass = NFW_M200 * (1.0 - gas_frac) / N_halo
    MassCorrection = 1.0
    MassScale = 4 * np.pi * RhoCrit * NFW_delta * NFW_rs**3.0

    # Interpolation parameters
    INTERPOL_BINS = R_bins
    INTERPOL_R_MIN = NFW_r200 * R_min  # minimum gas sampling radius (gas/halo cut below)
    INTERPOL_R_MAX = NFW_r200 * R_max  # maximum gas sampling radius (gas/halo cut above)

    # helper functions
    def GasRho(r):
        x0 = gas_R0 / NFW_rs
        x = r / NFW_rs
        return MassCorrection * RhoCrit * NFW_delta / ((x + x0) * (1 + x) ** 2.0)

    def HaloRho(r):
        x = r / NFW_rs
        return RhoCrit * NFW_delta / (x * (1 + x) ** 2.0)

    def Rho(r):
        return gas_frac * GasRho(r) + (1.0 - gas_frac) * HaloRho(r)

    def GasMass(r):
        if r > NFW_r200:
            return NFW_M200
        x0 = gas_R0 / NFW_rs
        x = r / NFW_rs
        return (
            MassCorrection
            * MassScale
            * (x * (x0 - 1) / (1 + x) + x0 * x0 * np.log(1 + x / x0) + (1 - 2 * x0) * np.log(1 + x))
            / (1 - x0) ** 2.0
        )

    def HaloMass(r):
        x = r / NFW_rs
        return MassScale * (1.0 / (1.0 + x) + np.log(1.0 + x) - 1.0)

    def Mass(r):
        return gas_frac * GasMass(r) + (1.0 - gas_frac) * HaloMass(r)

    def Omega(r):
        if S_omega == 0:
            return 1.0
        else:
            return (Mass(r) / NFW_M200) ** S_omega / r**2.0

    def Sigma_Integrand(r):
        return G * Mass(r) * Rho(r) / r**2.0

    def Sigma(r):
        if r > NFW_r200:
            return 0.0
        return np.sqrt(quad(Sigma_Integrand, r, NFW_r200, epsrel=0.1)[0] / Rho(r))

    def Potential(r):
        return G * ((Mass(r) / r) + (MassScale * ((1.0 / (NFW_rs + r)) - (1.0 / (NFW_rs + NFW_r200)))))

    # vectorize functions
    vecSigma = np.vectorize(Sigma)
    # vecRho = np.vectorize(Rho)
    vecGasMass = np.vectorize(GasMass)
    vecHaloMass = np.vectorize(HaloMass)
    # vecMass = np.vectorize(Mass)
    vecOmega = np.vectorize(Omega)
    vecPotential = np.vectorize(Potential)

    # mass correction due to gas sofetning
    MassCorrection = NFW_M200 / GasMass(NFW_r200)

    # angular momentum (MMW 22, 23)
    FC = (2.0 / 3.0) + (NFW_c / 21.5) ** 0.7
    HaloEnergy = -(G * NFW_M200**2.0 * FC) / (2 * NFW_r200)
    rJ = Lambda * (G * NFW_M200**2.5) / np.sqrt(np.abs(HaloEnergy))

    # invert function: GasMass^-1 = GasRadius
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX / INTERPOL_R_MIN) / INTERPOL_BINS + np.log(INTERPOL_R_MIN)
    )
    mass_bins_gas = vecGasMass(radial_bins)
    GasRadius = interp1d(mass_bins_gas, radial_bins)

    # invert function: HaloMass^-1 = HaloRadius
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX / INTERPOL_R_MIN) / INTERPOL_BINS + np.log(INTERPOL_R_MIN)
    )
    mass_bins_halo = vecHaloMass(radial_bins)
    HaloRadius = interp1d(mass_bins_halo, radial_bins)

    # interpolate sigma
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX / INTERPOL_R_MIN) / INTERPOL_BINS + np.log(INTERPOL_R_MIN)
    )
    sigma_bins = vecSigma(radial_bins)
    InterpolSigma = interp1d(radial_bins, sigma_bins)

    # interpolate Omega
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX / INTERPOL_R_MIN) / INTERPOL_BINS + np.log(INTERPOL_R_MIN)
    )
    sigma_bins = vecOmega(radial_bins)
    InterpolOmega = interp1d(radial_bins, sigma_bins)

    # interpolate Potential
    radial_bins = np.exp(
        np.arange(INTERPOL_BINS) * np.log(INTERPOL_R_MAX / INTERPOL_R_MIN) / INTERPOL_BINS + np.log(INTERPOL_R_MIN)
    )
    sigma_bins = vecPotential(radial_bins)
    InterpolPotential = interp1d(radial_bins, sigma_bins)

    # generate random positions gas
    radius_gas = GasRadius(np.random.random_sample(N_gas) * mass_bins_gas.max())
    phi_gas = 2.0 * np.pi * np.random.random_sample(N_gas)
    theta_gas = np.arcsin(2.0 * np.random.random_sample(N_gas) - 1.0)
    x_gas = radius_gas * np.cos(theta_gas) * np.cos(phi_gas)
    y_gas = radius_gas * np.cos(theta_gas) * np.sin(phi_gas)
    z_gas = radius_gas * np.sin(theta_gas)

    radius_halo = HaloRadius(np.random.random_sample(N_halo) * mass_bins_halo.max())
    # phi_halo = 2.0 * np.pi * np.random.random_sample(N_halo)
    theta_halo = np.arcsin(2.0 * np.random.random_sample(N_halo) - 1.0)
    # x_halo = radius_halo * np.cos(theta_halo) * np.cos(phi_halo)
    # y_halo = radius_halo * np.cos(theta_halo) * np.sin(phi_halo)
    # z_halo = radius_halo * np.sin(theta_halo)

    # generate random tracer positions with gas rho(r)
    radius_tr = GasRadius(np.random.random_sample(N_tr) * mass_bins_gas.max())
    phi_tr = 2.0 * np.pi * np.random.random_sample(N_tr)
    theta_tr = np.arcsin(2.0 * np.random.random_sample(N_tr) - 1.0)
    x_tr = radius_tr * np.cos(theta_tr) * np.cos(phi_tr)
    y_tr = radius_tr * np.cos(theta_tr) * np.sin(phi_tr)
    z_tr = radius_tr * np.sin(theta_tr)

    # momentum
    AxisDistance_gas = radius_gas * np.cos(theta_gas)
    MomentumSum_gas = np.sum(gas_mass * InterpolOmega(radius_gas) * AxisDistance_gas * AxisDistance_gas)

    AxisDistance_halo = radius_halo * np.cos(theta_halo)
    MomentumSum_halo = np.sum(halo_mass * InterpolOmega(radius_halo) * AxisDistance_halo * AxisDistance_halo)

    MomentumSum = MomentumSum_gas + MomentumSum_halo
    MomentumScale = rJ / MomentumSum  # momentum scale factor

    for _iter1 in range(0, 100):
        VelocityR_gas = np.zeros(N_gas)
        VelocityPhi_gas = InterpolOmega(radius_gas) * AxisDistance_gas * MomentumScale
        VelocityZ_gas = np.zeros(N_gas)

        VelocityR_halo = np.zeros(N_halo)
        VelocityPhi_halo = InterpolOmega(radius_halo) * AxisDistance_halo * MomentumScale
        VelocityZ_halo = np.zeros(N_halo)

        ind = None  # where a1 > z2

        for iter2 in range(0, 100):  # Von Neumann cycles
            if iter2 == 0:
                sigma = InterpolSigma(radius_halo)
                VelocityScatterR_halo = sigma * np.random.randn(N_halo)
                VelocityScatterZ_halo = sigma * np.random.randn(N_halo)
                VelocityScatterPhi_halo = sigma * np.random.randn(N_halo)
            else:
                sigma = InterpolSigma(radius_halo[ind])
                VelocityScatterR_halo[ind] = sigma * np.random.randn(radius_halo[ind].shape[0])
                VelocityScatterZ_halo[ind] = sigma * np.random.randn(radius_halo[ind].shape[0])
                VelocityScatterPhi_halo[ind] = sigma * np.random.randn(radius_halo[ind].shape[0])

            a1 = (
                (VelocityR_halo + VelocityScatterR_halo) ** 2.0
                + (VelocityPhi_halo + VelocityScatterPhi_halo) ** 2.0
                + (VelocityZ_halo + VelocityScatterZ_halo)
            )
            a2 = 2.0 * InterpolPotential(radius_halo)

            check = a1 < a2
            if check.all():
                break

            ind = a1 > a2

        VelocityR_halo += VelocityScatterR_halo
        VelocityPhi_halo += VelocityScatterPhi_halo
        VelocityZ_halo += VelocityScatterZ_halo

        MomentumSum_gas = np.sum(gas_mass * VelocityPhi_gas * AxisDistance_gas)
        MomentumSum_halo = np.sum(halo_mass * VelocityPhi_halo * AxisDistance_halo)
        MomentumSum = MomentumSum_gas + MomentumSum_halo

        if rJ != 0.0:
            print(
                "desired momentum:%e current momentum:%e desired error:%e current error:%e"
                % (rJ, MomentumSum, 0.001, np.abs(1.0 - rJ / MomentumSum))
            )
            MomentumScale *= np.sqrt(rJ / MomentumSum)

        if (np.abs(1.0 - rJ / MomentumSum) < 0.001) | (rJ == 0.0):
            break

    utherm = 1.5 * InterpolSigma(radius_gas) ** 2.0
    vx_gas = VelocityR_gas * np.cos(phi_gas) - VelocityPhi_gas * np.sin(phi_gas)
    vy_gas = VelocityR_gas * np.sin(phi_gas) + VelocityPhi_gas * np.cos(phi_gas)
    vz_gas = VelocityZ_gas

    # write
    rtmp = np.sqrt(x_gas * x_gas + y_gas * y_gas + z_gas * z_gas)
    ind = rtmp < Rcut
    N_gas = rtmp[ind].shape[0]
    print("gas cut=", rtmp.max(), rtmp[ind].max(), Rcut)

    rtmp = np.sqrt(x_tr * x_tr + y_tr * y_tr + z_tr * z_tr)
    ind_tr = rtmp < Rcut
    N_tr = rtmp[ind_tr].shape[0]
    print("tr cut=", rtmp.max(), rtmp[ind_tr].max(), Rcut)

    pos_gas = np.array([x_gas[ind], y_gas[ind], z_gas[ind]]).T
    vel_gas = np.array([vx_gas[ind], vy_gas[ind], vz_gas[ind]]).T
    u_gas = utherm[ind]
    ids_gas = np.arange(1, N_gas + 1)

    pt0 = {"Coordinates": pos_gas, "Velocities": vel_gas, "InternalEnergy": u_gas, "ParticleIDs": ids_gas}

    pos_tr = np.array([x_tr[ind_tr], y_tr[ind_tr], z_tr[ind_tr]]).T
    vel_tr = np.zeros([N_tr, 3])
    mass_tr = np.zeros([N_tr])
    ids_tr = np.arange(1 + N_gas, N_gas + N_tr + 2)

    pt3 = {"Coordinates": pos_tr, "Velocities": vel_tr, "Masses": mass_tr, "ParticleIDs": ids_tr}

    massarr = np.array([gas_mass, 0, 0, 0, 0, 0], dtype="float64")
    particles = {"PartType0": pt0, "PartType3": pt3}
    boxSize = INTERPOL_R_MAX * 2.1  # todo check

    write_ic_file(filename, particles, MassTable=massarr, boxSize=boxSize)
