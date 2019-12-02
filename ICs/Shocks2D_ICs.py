"""
ics/Shocks2D_ICs.py
  Idealized initial conditions: shocks/implosion/discontinuity/"2D Riemann" tests in 2D.
  Following Schulz-Rinne (1993), Kurganov & Tadmor (2002), Liska & Wendroff (2003).
  http://www-troja.fjfi.cvut.cz/%7Eliska/CompareEuler/compare8/
"""
import numpy as np
import h5py
from ICs.utilities import write_ic_file

def create_ics(numPartPerDim=200, config=1, filename='ics.hdf5'):
    """ Create idealized ICs of Rosswog+ (2019) tests. """

    if config == 1: # SR1, KT3
        rho = [0.5323, 1.5, 0.1380, 0.5323]
        vx  = [1.2060, 0.0, 1.2060, 0.0]
        vy  = [0.0,    0.0, 1.2060, 1.2060]
        P   = [0.3,    1.5, 0.0290, 0.3]
        xc  = 0.3 + 0.5
        yc  = 0.3 + 0.5

    if config == 2: # SR2, KT4
        rho = [0.5065, 1.1, 1.1000, 0.5065]
        vx  = [0.8939, 0.0, 0.8939, 0.0]
        vy  = [0.0,    0.0, 0.8939, 0.8939]
        P   = [0.35,   1.1, 1.1,    0.35]
        xc  = -0.15 + 0.5
        yc  = -0.15 + 0.5

    if config == 3: # SR3, KT5
        rho = [2.00,  1.00,  1.00, 3.00]
        vx  = [-0.75, -0.75, 0.75, 0.75]
        vy  = [0.500, -0.50, 0.50, -0.50]
        P   = [1.0,   1.0,   1.0,  1.0]
        xc  = 0.888888 # config 3b (centered widescreen) #0.0 + 0.5
        yc  = 0.0 + 0.5

    if config == 4: # SR4, KT6
        rho = [2.00, 1.00,  1.00,  3.00]
        vx  = [0.75, 0.75,  -0.75, -0.75]
        vy  = [0.50, -0.50, 0.50,  -0.50]
        P   = [1.0,  1.0,   1.0,    1.0]
        xc  = 0.0 + 0.5
        yc  = 0.0 + 0.5

    if config == 5: # SR5, KT11
        rho = [0.5313, 1.0, 0.8, 0.5313]
        vx  = [0.8276, 0.1, 0.1, 0.1]
        vy  = [0.0,    0.0, 0.0, 0.7276]
        P   = [0.4,    1.0, 0.4, 0.4]
        xc  = 0.0 + 0.5
        yc  = 0.0 + 0.5

    if config == 6: # SR6, KT12
        rho = [1.0, 0.5313, 0.8, 1.0]
        vx  = [0.7276, 0.0, 0.0, 0.0]
        vy  = [0.0, 0.0, 0.0, 0.7262]
        P   = [1.0, 0.4, 1.0, 1.0]
        xc  = 0.0 + 0.5
        yc  = 0.0 + 0.5

    boxSize = 1.0
    gamma   = 1.4
    aspect  = 16/9 # 1.0 for real tests, 16/9 for widescreen vis

    # derived properties
    Lx = boxSize * aspect
    Ly = boxSize
    Nx = int(numPartPerDim * aspect)
    Ny = numPartPerDim
    dx = Lx / Nx
    dy = Ly / Ny

    # allocate
    pos  = np.zeros( (Nx*Ny,3), dtype='float32' )
    vel  = np.zeros( (Nx*Ny,3), dtype='float32' )
    dens = np.zeros( Nx*Ny, dtype='float32' )
    u    = np.zeros( Nx*Ny, dtype='float32' )
    ids  = np.arange( Nx*Ny, dtype='int32' ) + 1

    # assign gas cell properties
    for i in range(Nx):
        for j in range(Ny):
            index = i + j*Nx

            pos[index,0] = i * dx + dx/2.0
            pos[index,1] = j * dy + dy/2.0
            pos[index,2] = 0.0

            x = pos[index,0]
            y = pos[index,1]

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
                dens[index]  = rho[k]
                u[index]     = P[k]/rho[k]/(gamma-1.0)
                vel[index,0] = vx[k]
                vel[index,1] = vy[k]

            if config == 7:
                # Sijacki+12 S3.1.2 (implosion test of Hui+ 1999)
                if x+y > boxSize/2: # originally 0.15, with Lx = Ly = 0.3
                    P = 1.0
                    rho = 1.0
                else:
                    P = 0.14
                    rho = 0.125

                # assign properties (vx = vy = 0)
                dens[index]  = rho
                u[index]     = P/rho/(gamma-1.0)

    # density -> mass
    cell_vol = (Lx*Ly) / (Nx*Ny)
    mass = dens * cell_vol

    # write
    pt0 = {'Coordinates':pos, 'Velocities':vel, 'Masses':mass, 'InternalEnergy':u, 'ParticleIDs':ids}

    write_ic_file(filename, {'PartType0':pt0}, boxSize=boxSize)
