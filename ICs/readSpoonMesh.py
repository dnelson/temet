if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import snapHDF5 as ws
    import readsnapHDF5 as rs

    
    #open snapshot
    #filename="spoon_new-with-grid"
    #filename="spoon_newest-with-grid"
    #filename="simple_spoon-with-grid"
    filename="simple_spoon_new-with-grid"
    
    header = rs.snapshot_header(filename)
    BoxX,BoxY = header.boxsize, header.boxsize

    pos = rs.read_block(filename,"POS ",parttype=0)
    ids  = rs.read_block(filename,"ID  ",parttype=0)


    BoxSize = 1.0
    Rho0 = 1.0
    Rho1 = 2.0
    P0   = 0.5
    GAMMA = 5.0/3.0


    x_gas = pos[:,0]
    y_gas = pos[:,1]
    z_gas = pos[:,2]
    
    #fluid is at rest
    vx_gas = np.zeros(x_gas.shape[0])
    vy_gas = np.zeros(x_gas.shape[0])
    vz_gas = np.zeros(x_gas.shape[0])

    dens_gas = np.zeros(x_gas.shape[0])
    press_gas = np.repeat(P0,x_gas.shape[0])
    dens_gas[(ids > -2)] = Rho1
    dens_gas[(ids == -2)] = 200.0
    dens_gas[(ids > -2) &(y_gas > 0.6*BoxSize)] = Rho0

    utherm = press_gas/dens_gas/(GAMMA-1)
    
    N_gas = x_gas.shape[0]

    plt.plot(y_gas[ids == -2],z_gas[ids == -2],'ro',ms=1.0,mew=0.0)
    plt.plot(y_gas[ids == -1],z_gas[ids == -1],'go',ms=1.0,mew=0.0)
    plt.plot(y_gas[ids > -1],z_gas[ids > -1],'bo',ms=0.5,mew=0.0)
    plt.plot(y_gas[dens_gas == Rho1],z_gas[dens_gas == Rho1],'mo',ms=0.5,mew=0.0)
    plt.show()


    
    ##########################################################
    
    print("Writing snapshot...")
    double_precision = 1
    f=ws.openfile("spoon-new_ics.hdf5")
    
    ws.write_block(f, "POS ", 0, np.array([x_gas,y_gas,z_gas]).T)
    ws.write_block(f, "VEL ", 0, np.array([vx_gas,vy_gas,vz_gas]).T)
    ws.write_block(f, "U   ", 0, utherm)
    ws.write_block(f, "MASS", 0, dens_gas)
    ws.write_block(f, "ID  ", 0, ids)

    npart=np.array([N_gas,0,0,0,0,0], dtype="uint32")
    
    massarr=np.array([0,0,0,0,0,0], dtype="float64")
    header=ws.snapshot_header(npart=npart, nall=npart, massarr=massarr, double = np.array([double_precision], dtype="int32"))
                                                                        
    ws.writeheader(f, header)
    ws.closefile(f)
    print("done.")
