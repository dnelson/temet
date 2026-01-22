"""
Idealized initial conditions: kelvin-helmholtz instability.
"""

import numpy as np

from ..ICs.utilities import write_ic_file


def create_ics(numPartPerDim=64, filename="ics.hdf5"):
    """Create idealized ICs for KH instability in 2D."""
    # hard-coded config (can change)
    angle = 0.0  # degrees

    stripeWidth = 0.5
    boxSize = 1.0

    gamma = 1.4
    P0 = 2.5
    rho1 = 1.0
    rho2 = 2.0
    v = 0.5
    sigma = 0.05 / np.sqrt(2.0)
    w0 = 0.1

    # derived properties
    Lx = boxSize
    Ly = boxSize
    Nx = numPartPerDim
    Ny = numPartPerDim
    dx = Lx / Nx
    dy = Ly / Ny
    angle_rad = np.deg2rad(angle)

    # allocate
    pos = np.zeros((Nx * Ny, 3), dtype="float32")
    vel = np.zeros((Nx * Ny, 3), dtype="float32")
    dens = np.zeros(Nx * Ny, dtype="float32")
    u = np.zeros(Nx * Ny, dtype="float32")
    id = np.arange(Nx * Ny, dtype="int32") + 1

    # angle calculation
    x1 = 0.0
    x2 = Lx
    y1 = Ly / 2.0 - np.tan(angle_rad) * Lx / 2.0
    y2 = Ly / 2.0 + np.tan(angle_rad) * Lx / 2.0
    assert angle == 0.0  # note: discontinuities across box boundaries for periodic for !=0

    # assign gas cell properties
    for i in range(Nx):
        for j in range(Ny):
            index = i + j * Nx

            pos[index, 0] = i * dx + dx / 2.0
            pos[index, 1] = j * dy + dy / 2.0
            pos[index, 2] = 0.0

            x = pos[index, 0]
            y = pos[index, 1]

            d = np.abs((x2 - x1) * (y1 - y) - (x1 - x) * (y2 - y1)) / np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0)

            # top/bottom
            if d >= stripeWidth / 2.0:
                dens[index] = rho1
                u[index] = P0 / rho1 / (gamma - 1.0)
                vel[index, 0] = -v * np.cos(angle_rad)
                vel[index, 1] = -v * np.sin(angle_rad)

            # middle
            if d < stripeWidth / 2.0:
                dens[index] = rho2
                u[index] = P0 / rho2 / (gamma - 1.0)
                vel[index, 0] = +v * np.cos(angle_rad)
                vel[index, 1] = +v * np.sin(angle_rad)

            # initial eigen perturbation
            vel[index, 0] += (
                w0
                * np.sin(4.0 * np.pi * x * np.cos(angle_rad))
                * (
                    np.exp(-((d + stripeWidth / 2.0) ** 2.0) / (2.0 * sigma**2.0))
                    + np.exp(-((d - stripeWidth / 2.0) ** 2.0) / (2.0 * sigma**2.0))
                )
                * np.sin(angle_rad)
            )
            vel[index, 1] += (
                w0
                * np.sin(4.0 * np.pi * x * np.cos(angle_rad))
                * (
                    np.exp(-((d + stripeWidth / 2.0) ** 2.0) / (2.0 * sigma**2.0))
                    + np.exp(-((d - stripeWidth / 2.0) ** 2.0) / (2.0 * sigma**2.0))
                )
                * np.cos(angle_rad)
            )

    # density -> mass
    cell_vol = (Lx * Ly) / (Nx * Ny)
    mass = dens * cell_vol

    # write
    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": mass, "InternalEnergy": u, "ParticleIDs": id}

    write_ic_file(filename, {"PartType0": pt0}, boxSize=boxSize)


def create_ics2(numPartPerDim=64, filename="ics.hdf5"):
    """Different strategy, without angle, but with a higher density center (optional)."""
    middleAtTwiceRes = False

    boxSize = 1.0
    P = 2.5
    omega_0 = 0.1
    sigma = 0.05 / np.sqrt(2)
    GAMMA = 1.4

    # derived
    L_x = boxSize
    L_y = boxSize
    N_x = numPartPerDim
    N_y = numPartPerDim
    dx = L_x / N_x
    dy = L_y / N_y

    # mesh
    mesh = np.meshgrid(np.arange(N_x), np.arange(N_y))

    posx = (mesh[0] * dx).reshape(N_x * N_y) + 0.5 * dx
    posy = (mesh[1] * dy).reshape(N_x * N_y) + 0.5 * dy

    mesh = np.meshgrid(np.arange(N_x * 2), np.arange(N_y * 2))
    posx2 = (mesh[0] * dx / 2.0).reshape(N_x * N_y * 4) + 0.5 * dx / 2.0
    posy2 = (mesh[1] * dy / 2.0).reshape(N_x * N_y * 4) + 0.5 * dy / 2.0

    # allocate
    Ntot = N_x * N_y
    if middleAtTwiceRes:
        Ntot = N_x * N_y / 2 + N_x * N_y * 4 / 2

    pos = np.zeros((Ntot, 3), dtype="float32")
    vel = np.zeros((Ntot, 3), dtype="float32")
    mass = np.zeros(Ntot, dtype="float32")
    u = np.zeros(Ntot, dtype="float32")
    id = np.arange(Ntot, dtype="int32") + 1

    # bottom?
    pos[: N_x * N_y / 4, 0] = posx[: N_x * N_y / 4]
    pos[: N_x * N_y / 4, 1] = posy[: N_x * N_y / 4]
    mass[: N_x * N_y / 4] = 1.0 * dx * dy
    u[: N_x * N_y / 4] = P / ((GAMMA - 1) * 1.0)
    vel[: N_x * N_y / 4, 0] = -0.5

    # middle
    if middleAtTwiceRes:
        pos[N_x * N_y / 4 : N_x * N_y / 4 + N_x * N_y / 4, 0] = posx2[N_x * N_y : N_x * N_y * 3]
        pos[N_x * N_y / 4 : N_x * N_y / 4 + N_x * N_y * 2, 1] = posy2[N_x * N_y : N_x * N_y * 3]
        mass[N_x * N_y / 4 : N_x * N_y / 4 + N_x * N_y * 2] = 2.0 * dx * dy / 4.0
        u[N_x * N_y / 4 : N_x * N_y / 4 + N_x * N_y * 2] = P / ((GAMMA - 1) * 2.0)
        vel[N_x * N_y / 4 : N_x * N_y / 4 + N_x * N_y * 2, 0] = +0.5
    else:
        pos[N_x * N_y / 4 : 3 * N_x * N_y / 4, 0] = posx[N_x * N_y / 4 : 3 * N_x * N_y / 4]
        pos[N_x * N_y / 4 : 3 * N_x * N_y / 4, 1] = posy[N_x * N_y / 4 : 3 * N_x * N_y / 4]
        mass[N_x * N_y / 4 : 3 * N_x * N_y / 4] = 2.0 * dx * dy
        u[N_x * N_y / 4 : 3 * N_x * N_y / 4] = P / ((GAMMA - 1) * 2.0)
        vel[N_x * N_y / 4 : 3 * N_x * N_y / 4, 0] = +0.5

    # top?
    pos[-N_x * N_y / 4 :, 0] = posx[-N_x * N_y / 4 :]
    pos[-N_x * N_y / 4 :, 1] = posy[-N_x * N_y / 4 :]
    mass[-N_x * N_y / 4 :] = 1.0 * dx * dy
    u[-N_x * N_y / 4 :] = P / ((GAMMA - 1) * 1.0)
    vel[-N_x * N_y / 4 :, 0] = -0.5

    pos[:, 2] = 0.0

    vel[:, 1] = (
        omega_0
        * np.sin(4 * np.pi * pos[:, 0])
        * (
            np.exp(-((pos[:, 1] - 0.25) ** 2) * 0.5 / (sigma**2))
            + np.exp(-((pos[:, 1] - 0.75) ** 2) * 0.5 / (sigma**2))
        )
    )
    vel[:, 2] = 0.0

    # write
    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": mass, "InternalEnergy": u, "ParticleIDs": id}

    write_ic_file(filename, {"PartType0": pt0}, boxSize=boxSize)
