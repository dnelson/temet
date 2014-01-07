import snapHDF5 as ws
import numpy as np
import matplotlib.pyplot as plt

#import  matplotlib.axes as maxes

#set unit system
UnitMass_in_g            = 1.0
UnitVelocity_in_cm_per_s = 1.0
UnitLength_in_cm         = 1.0
G                        = 0
GAMMA  = 1.4

##################
#INPUT PARAMETERS#
##################
Nx    = 20
Ny    = 20
beta = 5.0
Tinf = 1.0

###################
print "\n-STARTING-\n"
Lx    = 10.0
Ly    = 10.0
N_gas = Nx * Ny
delta_x = Lx/Nx
delta_y = Ly/Ny

x,y = np.mgrid[ 0.0 + 0.5 * delta_x: Lx: delta_x, 0.0 + 0.5 * delta_y: Ly: delta_y]
z = np.zeros([Nx,Ny])

radius = np.sqrt((x - 0.5* Lx)**2 + (y- 0.5* Ly)**2)
phi = np.arctan2((y- 0.5* Ly),(x - 0.5* Lx))
vphi  = radius * beta/2.0/np.pi * np.exp(0.5*(1.0-radius*radius))
vphi[radius > 5.0] = 0.0
#print vphi.min()

vx  = -vphi * np.sin(phi)
vy  =  vphi * np.cos(phi)
vz = np.zeros([Nx,Ny])

T = Tinf - (GAMMA - 1.0) * beta * beta/8.0/ GAMMA /np.pi/np.pi * np.exp(1.0 - radius * radius)

dens = T**(1.0/(GAMMA - 1.0))
utherm = T/(GAMMA - 1)
ids = np.arange(1,N_gas+1)
##########################################################
print "Writing snapshot..."
f=ws.openfile("ics.dat.hdf5")

ws.write_block(f, "POS ", 0, np.array([x,y,z]).T)
ws.write_block(f, "VEL ", 0, np.array([vx,vy,vz]).T)
ws.write_block(f, "U   ", 0, utherm)
ws.write_block(f, "MASS", 0, dens)
ws.write_block(f, "ID  ", 0, ids)

massarr=np.array([0,0,0,0,0,0], dtype="float64")
npart=np.array([N_gas,0,0,0,0,0], dtype="uint32")
header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, double = np.array([1], dtype="int32"))
ws.writeheader(f, header)
ws.closefile(f)
print "done."

print "\n-FINISHED-\n"






