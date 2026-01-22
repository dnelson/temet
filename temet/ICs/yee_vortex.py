"""
Idealized initial conditions: Yee Vortex.
"""

import numpy as np

from ..ICs.utilities import write_ic_file


def create_ics(filename="ics.hdf5"):
    """Create idealized ICs."""
    # set unit system (assumed to be dimensionless)
    # UnitMass_in_g            = 1.0
    # UnitVelocity_in_cm_per_s = 1.0
    # UnitLength_in_cm         = 1.0
    # G                        = 0
    GAMMA = 1.4

    # input parameters
    Nx = 20
    Ny = 20
    beta = 5.0
    Tinf = 1.0
    Lx = 10.0
    Ly = 10.0

    # derived
    N_gas = Nx * Ny
    delta_x = Lx / Nx
    delta_y = Ly / Ny

    # positions
    x, y = np.mgrid[0.0 + 0.5 * delta_x : Lx : delta_x, 0.0 + 0.5 * delta_y : Ly : delta_y]
    z = np.zeros([Nx, Ny])

    # velocities
    radius = np.sqrt((x - 0.5 * Lx) ** 2 + (y - 0.5 * Ly) ** 2)
    phi = np.arctan2((y - 0.5 * Ly), (x - 0.5 * Lx))
    vphi = radius * beta / 2.0 / np.pi * np.exp(0.5 * (1.0 - radius * radius))
    vphi[radius > 5.0] = 0.0

    vx = -vphi * np.sin(phi)
    vy = vphi * np.cos(phi)
    vz = np.zeros([Nx, Ny])

    # thermodynamic quantities
    T = Tinf - (GAMMA - 1.0) * beta * beta / 8.0 / GAMMA / np.pi / np.pi * np.exp(1.0 - radius * radius)

    dens = T ** (1.0 / (GAMMA - 1.0))
    utherm = T / (GAMMA - 1)

    # write
    ids = np.arange(1, N_gas + 1)
    pos = np.array([x, y, z]).T
    vel = np.array([vx, vy, vz]).T

    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": dens, "InternalEnergy": utherm, "ParticleIDs": ids}

    write_ic_file(filename, {"PartType0": pt0}, boxSize=Lx)
