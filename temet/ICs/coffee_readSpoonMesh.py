"""Read mesh file for idealized ICs of "3D spoon" problem."""

import h5py
import numpy as np

from ..ICs.utilities import write_ic_file


def create_ics(filename="spoon-new_ics.hdf5"):
    """Create idealized ICs."""
    # open snapshot
    # filename="spoon_new-with-grid"
    # filename="spoon_newest-with-grid"
    # filename="simple_spoon-with-grid"
    filename = "simple_spoon_new-with-grid"

    with h5py.File(filename + ".hdf5", "r") as f:
        pos = f["PartType0"]["Coordinates"][:]
        ids = f["PartType0"]["ParticleIDs"][:]

        BoxSize = f["Header"].attrs["BoxSize"]

    # parameters
    Rho0 = 1.0
    Rho1 = 2.0
    P0 = 0.5
    GAMMA = 5.0 / 3.0

    x_gas = pos[:, 0]
    y_gas = pos[:, 1]
    z_gas = pos[:, 2]

    # fluid is at rest
    vx_gas = np.zeros(x_gas.shape[0])
    vy_gas = np.zeros(x_gas.shape[0])
    vz_gas = np.zeros(x_gas.shape[0])

    dens_gas = np.zeros(x_gas.shape[0])
    press_gas = np.repeat(P0, x_gas.shape[0])
    dens_gas[(ids > -2)] = Rho1
    dens_gas[(ids == -2)] = 200.0
    dens_gas[(ids > -2) & (y_gas > 0.6 * BoxSize)] = Rho0

    utherm = press_gas / dens_gas / (GAMMA - 1)

    # write
    pos = np.array([x_gas, y_gas, z_gas]).T
    vel = np.array([vx_gas, vy_gas, vz_gas]).T
    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": dens_gas, "InternalEnergy": utherm, "ParticleIDs": ids}

    write_ic_file(filename, {"PartType0": pt0}, boxSize=BoxSize)
