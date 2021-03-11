import numpy as np
import matplotlib.pyplot as plt

#GAS PARAMETERS
NgX = 960
NgY = 540
rho0 = 1.0
rho1 = 0.5
BoxX = 1.778
BoxY = 1.0
InterfY = 0.6
P0 = 1.0
GAMMA = 5.0/3

#SPOON PARAMETERS
angle1 = 60.0
rad = 0.2
thick = 0.025
N = 500 # 75 * (960*540)/(100*100)



##########################################
l1 =  np.pi * angle1/180.0 * rad
l2 =  np.pi * (thick/2)
l3 =  np.pi * angle1/180.0 * (rad-thick)
l4 =  np.pi * (thick/2)
L = l1+l2+l3+l4

off = L/N
n1,n2 = int(l1/off),int(l2/off)
n3,n4 = N - n1 - 2*n2,n2 

x_surface, y_surface = np.zeros([2*N]), np.zeros([2*N])
id_surface = np.zeros([2*N])
id_surface[0:N],id_surface[N:2*N] = -1,-2 
#1
alpha = np.arange(- ( 90 - angle1*0.5),- ( 90 + angle1*0.5), -angle1/n1)
x_surface[0:n1] =  (rad+off/2)*np.cos(alpha * np.pi/180)
y_surface[0:n1] =  (rad+off/2)*np.sin(alpha * np.pi/180)
x_surface[N:n1+N] =  (rad-off/2)*np.cos(alpha * np.pi/180)
y_surface[N:n1+N] =  (rad-off/2)*np.sin(alpha * np.pi/180)
#2
alpha = np.arange(- angle1/2 - 90,- angle1/2 - 90-180,-180.0/n2)
x_surface[n1:n1+n2] = ((thick+off)/2)*np.cos(alpha * np.pi/180) + (rad+off/2-(thick+off)/2)*np.cos((-angle1 - 90 + angle1/2) * np.pi/180)
y_surface[n1:n1+n2] = ((thick+off)/2)*np.sin(alpha * np.pi/180) + (rad+off/2-(thick+off)/2)*np.sin((-angle1 - 90 + angle1/2) * np.pi/180)
x_surface[n1+N:n1+n2+N] = ((thick-off)/2)*np.cos(alpha * np.pi/180) + (rad-off/2-(thick-off)/2)*np.cos((-angle1 - 90 + angle1/2) * np.pi/180)
y_surface[n1+N:n1+n2+N] = ((thick-off)/2)*np.sin(alpha * np.pi/180) + (rad-off/2-(thick-off)/2)*np.sin((-angle1 - 90 + angle1/2) * np.pi/180)
#3
alpha = np.arange(- ( 90 + angle1*0.5), - ( 90 - angle1*0.5), angle1/n3)
x_surface[n1+n2:n1+n2+n3] = (rad+off/2-(thick+off))*np.cos(alpha * np.pi/180)
y_surface[n1+n2:n1+n2+n3]  = (rad+off/2-(thick+off))*np.sin(alpha * np.pi/180)
x_surface[n1+n2+N:n1+n2+n3+N]  = (rad-off/2-(thick-off))*np.cos(alpha * np.pi/180)
y_surface[n1+n2+N:n1+n2+n3+N] = (rad-off/2-(thick-off))*np.sin(alpha * np.pi/180)
#4
alpha = np.arange(angle1/2 + 90,+ angle1/2 + 90-180,-180.0/n4)
x_surface[n1+n2+n3:n1+n2+n3+n4] = ((thick+off)/2)*np.cos(alpha * np.pi/180) +  (rad+off/2-(thick+off)/2)*np.cos(( - 90 + angle1/2) * np.pi/180)
y_surface[n1+n2+n3:n1+n2+n3+n4] = ((thick+off)/2)*np.sin(alpha * np.pi/180) +  (rad+off/2-(thick+off)/2)*np.sin(( - 90 + angle1/2) * np.pi/180)
x_surface[n1+n2+n3+N:n1+n2+n3+n4+N] = ((thick-off)/2)*np.cos(alpha * np.pi/180) +  (rad-off/2-(thick-off)/2)*np.cos(( - 90 + angle1/2) * np.pi/180)
y_surface[n1+n2+n3+N:n1+n2+n3+n4+N] = ((thick-off)/2)*np.sin(alpha * np.pi/180) +  (rad-off/2-(thick-off)/2)*np.sin(( - 90 + angle1/2) * np.pi/180)

xc = x_surface.sum()/x_surface.shape[0]
yc = y_surface.sum()/y_surface.shape[0]

#dx = 0.5+0.25 - xc
#dy = 0.5 - yc
dx = 0.75*BoxX - xc
dy = 0.5*BoxY - yc

x_surface, y_surface = x_surface + dx, y_surface+dy

##########################################

Ntot = NgX*NgY
delta_x, delta_y = BoxX/NgX, BoxY/NgY
x_gas,y_gas = np.mgrid[ 0.0 + 0.5 * delta_x: BoxX: delta_x, 0.0 + 0.5 * delta_y: BoxY: delta_y]
radius = np.sqrt((x_gas - dx)**2 + (y_gas- dy)**2)
phi = np.arctan2((y_gas- dy),(x_gas - dx))

x1,y1 = (rad+off/2-(thick+off)/2)*np.cos((-angle1 - 90 + angle1/2) * np.pi/180),\
        (rad+off/2-(thick+off)/2)*np.sin((-angle1 - 90 + angle1/2) * np.pi/180)
x2,y2 = (rad+off/2-(thick+off)/2)*np.cos(( - 90 + angle1/2) * np.pi/180), \
        (rad+off/2-(thick+off)/2)*np.sin(( - 90 + angle1/2) * np.pi/180)

ind = (np.sqrt((x_gas-dx-x1)**2 + (y_gas-dy-y1)**2) > thick/2+1.5*off) & \
      (np.sqrt((x_gas-dx-x2)**2 + (y_gas-dy-y2)**2) > thick/2+1.5*off) & \
      ( (radius < rad-thick-1.5*off) | (radius > rad+1.5*off) | \
        (phi < -np.pi + (90-angle1/2)/180.0*np.pi) | (phi > -(90-angle1/2)/180.0*np.pi))
x_gas, y_gas= x_gas[ind], y_gas[ind]
id_gas = np.arange(1,x_gas.shape[0]+1)


##########################################

x = np.append(x_surface,x_gas)
y = np.append(y_surface,y_gas) 
z = np.zeros(x.shape[0])
ids = np.append(id_surface,id_gas)
vx,vy,vz = np.zeros(x.shape[0]),np.zeros(x.shape[0]),np.zeros(x.shape[0])
dens = np.zeros(x.shape[0])
dens[y < InterfY],dens[y >= InterfY] = rho0, rho1
dens[ids == -2] = 10.0*rho0
press = np.repeat(P0,x.shape[0])
tracer = np.zeros(x.shape[0])
tracer[y < InterfY],tracer[y >= InterfY] = 0.0, 1.0
utherm = press/dens/(GAMMA - 1)
N_gas = x.shape[0]

#plt.plot(x_gas,y_gas,'bo',ms=1.2,mew=0.0)
#plt.plot(x_surface,y_surface,'ro',ms=1.0,mew=0.0)
#plt.show()
##########################################################
print("Writing snapshot...")
if 0:
    f=ws.openfile("ics.big.dat.hdf5")

    ws.write_block(f, "POS ", 0, np.array([x,y,z]).T)
    ws.write_block(f, "VEL ", 0, np.array([vx,vy,vz]).T)
    ws.write_block(f, "U   ", 0, utherm)
    ws.write_block(f, "MASS", 0, dens)
    ws.write_block(f, "ID  ", 0, ids)
    ws.write_block(f, "TRCE", 0, tracer)

    massarr=np.array([0,0,0,0,0,0], dtype="float64")
    npart=np.array([N_gas,0,0,0,0,0], dtype="uint32")
    header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, double = np.array([0], dtype="int32"))
    ws.writeheader(f, header)
    ws.closefile(f)
print("Done.")
