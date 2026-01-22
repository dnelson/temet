"""
Idealized initial conditions: shocks/implosion/discontinuity/"2D Riemann" tests in 2D.

Following Schulz-Rinne (1993), Kurganov & Tadmor (2002), Liska & Wendroff (2003).
http://www-troja.fjfi.cvut.cz/%7Eliska/CompareEuler/compare8/
"""

import numpy as np

from ..ICs.utilities import write_ic_file


def create_ics(numPartPerDim=200, config=1, filename="ics.hdf5"):
    """Create idealized ICs of Rosswog+ (2019) tests."""
    if config == 1:  # SR1, KT3
        rho = [0.5323, 1.5, 0.1380, 0.5323]
        vx = [1.2060, 0.0, 1.2060, 0.0]
        vy = [0.0, 0.0, 1.2060, 1.2060]
        P = [0.3, 1.5, 0.0290, 0.3]
        xc = 0.3 + 0.5
        yc = 0.3 + 0.5

    if config == 2:  # SR2, KT4
        rho = [0.5065, 1.1, 1.1000, 0.5065]
        vx = [0.8939, 0.0, 0.8939, 0.0]
        vy = [0.0, 0.0, 0.8939, 0.8939]
        P = [0.35, 1.1, 1.1, 0.35]
        xc = -0.15 + 0.5
        yc = -0.15 + 0.5

    if config == 3:  # SR3, KT5
        rho = [2.00, 1.00, 1.00, 3.00]
        vx = [-0.75, -0.75, 0.75, 0.75]
        vy = [0.500, -0.50, 0.50, -0.50]
        P = [1.0, 1.0, 1.0, 1.0]
        xc = 0.888888  # config 3b (centered widescreen) #0.0 + 0.5
        yc = 0.0 + 0.5

    if config == 4:  # SR4, KT6
        rho = [2.00, 1.00, 1.00, 3.00]
        vx = [0.75, 0.75, -0.75, -0.75]
        vy = [0.50, -0.50, 0.50, -0.50]
        P = [1.0, 1.0, 1.0, 1.0]
        xc = 0.0 + 0.5
        yc = 0.0 + 0.5

    if config == 5:  # SR5, KT11
        rho = [0.5313, 1.0, 0.8, 0.5313]
        vx = [0.8276, 0.1, 0.1, 0.1]
        vy = [0.0, 0.0, 0.0, 0.7276]
        P = [0.4, 1.0, 0.4, 0.4]
        xc = 0.0 + 0.5
        yc = 0.0 + 0.5

    if config == 6:  # SR6, KT12
        rho = [1.0, 0.5313, 0.8, 1.0]
        vx = [0.7276, 0.0, 0.0, 0.0]
        vy = [0.0, 0.0, 0.0, 0.7262]
        P = [1.0, 0.4, 1.0, 1.0]
        xc = 0.0 + 0.5
        yc = 0.0 + 0.5

    boxSize = 1.0
    gamma = 1.4
    aspect = 16 / 9  # 1.0 for real tests, 16/9 for widescreen vis

    # derived properties
    Lx = boxSize * aspect
    Ly = boxSize
    Nx = int(numPartPerDim * aspect)
    Ny = numPartPerDim
    dx = Lx / Nx
    dy = Ly / Ny

    # allocate
    pos = np.zeros((Nx * Ny, 3), dtype="float32")
    vel = np.zeros((Nx * Ny, 3), dtype="float32")
    dens = np.zeros(Nx * Ny, dtype="float32")
    u = np.zeros(Nx * Ny, dtype="float32")
    ids = np.arange(Nx * Ny, dtype="int32") + 1

    # assign gas cell properties
    for i in range(Nx):
        for j in range(Ny):
            index = i + j * Nx

            pos[index, 0] = i * dx + dx / 2.0
            pos[index, 1] = j * dy + dy / 2.0
            pos[index, 2] = 0.0

            x = pos[index, 0]
            y = pos[index, 1]

            if config <= 6:
                # lower left (SW)
                if x < xc and y < yc:
                    k = 2
                # upper left (NW)
                if x < xc and y > yc:
                    k = 0
                # upper right (NE)
                if x > xc and y > yc:
                    k = 1
                # lower right (SE)
                if x > xc and y < yc:
                    k = 3

                # assign properties
                dens[index] = rho[k]
                u[index] = P[k] / rho[k] / (gamma - 1.0)
                vel[index, 0] = vx[k]
                vel[index, 1] = vy[k]

            if config == 7:
                # Sijacki+12 S3.1.2 (implosion test of Hui+ 1999)
                if x + y > boxSize / 2:  # originally 0.15, with Lx = Ly = 0.3
                    P = 1.0
                    rho = 1.0
                else:
                    P = 0.14
                    rho = 0.125

                # assign properties (vx = vy = 0)
                dens[index] = rho
                u[index] = P / rho / (gamma - 1.0)

    # density -> mass
    cell_vol = (Lx * Ly) / (Nx * Ny)
    mass = dens * cell_vol

    # write
    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": mass, "InternalEnergy": u, "ParticleIDs": ids}

    write_ic_file(filename, {"PartType0": pt0}, boxSize=boxSize)


def uniform_ics_3d(N=64):
    """Create completely uniform ICs of gas (in 3D)."""
    # config
    rho = 1.0
    vx = 0.0
    vy = 0.0
    vz = 0.0
    P = 1.0

    L = 1.0
    gamma = 1.4

    # derived properties
    dx = dy = dz = L / N

    # allocate
    pos = np.zeros((N**3, 3), dtype="float32")
    vel = np.zeros((N**3, 3), dtype="float32")
    dens = np.zeros(N**3, dtype="float32")
    u = np.zeros(N**3, dtype="float32")
    ids = np.arange(N**3, dtype="int32") + 1

    # assign gas cell positions
    for i in range(N):
        for j in range(N):
            for k in range(N):
                index = i + j * N + k * N**2

                pos[index, 0] = i * dx + dx / 2.0
                pos[index, 1] = j * dy + dy / 2.0
                pos[index, 2] = k * dz + dz / 2.0

    # assign properties
    dens[:] = rho
    u[:] = P / rho / (gamma - 1.0)
    vel[:, 0] = vx
    vel[:, 1] = vy
    vel[:, 2] = vz

    # density -> mass
    cell_vol = L**3 / N**3
    mass = dens * cell_vol

    # write
    pt0 = {"Coordinates": pos, "Velocities": vel, "Masses": mass, "InternalEnergy": u, "ParticleIDs": ids}

    filename = "ics_%d.hdf5" % N
    write_ic_file(filename, {"PartType0": pt0}, boxSize=L)


def vis_test(conf=0):
    """Quick vis test."""
    import glob

    import h5py
    import matplotlib.pyplot as plt

    from ..util.treeSearch import calcHsml

    path = "/u/dnelson/sims.idealized/sims.turbbox/gamma53_64/output/"

    cmap = "viridis"

    num_snaps = len(glob.glob(path + "snap_*.hdf5"))

    for snap in np.arange(num_snaps):
        outfile = "frame%d_%03d.png" % (conf, snap)
        print(outfile)
        # if isfile(outfile):
        #    print(' skip')
        #    continue

        # load
        with h5py.File(path + "snap_%03d.hdf5" % snap, "r") as f:
            boxsize = f["Header"].attrs["BoxSize"]
            pos = f["PartType0"]["Coordinates"][()]
            vel = f["PartType0"]["Velocities"][()]
            dens = f["PartType0"]["Density"][()]

        vmag = np.sqrt(vel[:, 0] ** 2 + vel[:, 1] ** 2 + vel[:, 2] ** 2)
        print(" dens: ", dens.mean(), dens.min(), dens.max())
        # print(' vmag: ',vmag.mean(),vmag.min(),vmag.max())

        # shuffle
        rng = np.random.default_rng(424242)
        indices = np.arange(dens.size)
        rng.shuffle(indices)
        pos = pos[indices]
        vel = vel[indices]
        dens = dens[indices]
        vmag = vmag[indices]

        # start plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        ax.set_xlim(0, boxsize)
        ax.set_ylim(0, boxsize)
        ax.set_aspect("equal")

        # scatter, quiver, or voronoi slice?
        if conf == 1:
            ax.scatter(pos[:, 0], pos[:, 1], c=vmag, s=1, cmap=cmap)

        if conf == 2:
            cmap = plt.get_cmap(cmap)
            norm = plt.Normalize(vmin=0.95, vmax=1.05)  # vmin=0.0, vmax=0.5 for vmag

            # select slice
            n = dens.size ** (1 / 3)
            slice_depth = boxsize / n * 1.5
            boxcenter = boxsize / 2

            w = np.where((pos[:, 2] > boxcenter - slice_depth / 2) & (pos[:, 2] < boxcenter + slice_depth / 2))[0]

            # colors?
            colors = cmap(norm(dens[w]))  # always 1.0 with gamma=1.0?
            # colors = cmap(norm(vmag[w]))

            ax.quiver(pos[w, 0], pos[w, 1], vel[w, 0], vel[w, 1], scale=4.0, color=colors, cmap=cmap)

        if conf == 3:
            # define (x,y) pixel centers
            nPixels = 512
            pxSize = boxsize / nPixels
            boxcenter = boxsize / 2

            xypts = np.linspace(boxcenter - boxsize / 2, boxcenter + boxsize / 2 - pxSize, nPixels) + pxSize / 2
            xpts, ypts = np.meshgrid(xypts, xypts, indexing="ij")

            # construct [N,3] list of search positions
            search_pos = np.zeros((nPixels * nPixels, 3), dtype=pos.dtype)

            search_pos[:, 0] = xpts.ravel()
            search_pos[:, 1] = ypts.ravel()
            search_pos[:, 2] = boxcenter  # slice location along line-of-sight

            # construct tree, find nearest gas cell (parent Voronoi cell) to each pixel center
            _, index = calcHsml(pos, boxsize, posSearch=search_pos, nearest=True)

            assert index.min() >= 0 and index.max() < pos.shape[0]

            # sample values from cells onto grid pixels
            grid = vmag[index].reshape([nPixels, nPixels]).T

            extent = [0, boxsize, 0, boxsize]
            plt.imshow(grid, extent=extent, cmap=cmap, aspect=1.0)

        # finish plot
        plt.savefig(outfile)
        plt.close()
