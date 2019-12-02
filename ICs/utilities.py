"""
ics/utilities.py
  Idealized initial conditions: utility (common) functions.
"""
import numpy as np
import h5py

import struct
import glob
import matplotlib.pyplot as plt
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable

def write_ic_file(fileName, partTypes, boxSize, massTable=None, headerExtra=None):
    """ Helper to write a HDF5 IC file. partTypes is a dictionary with keys of the form 
    PartTypeX, each of which is its own dictionary of particle fields and ndarrays. 
    boxSize is a scalar float, and massTable a 6-element float array, if specified. """
    nPartTypes = 6

    with h5py.File(fileName,'w') as f:
        # write each PartTypeX group and datasets
        for ptName in partTypes.keys():
            g = f.create_group(ptName)
            for field in partTypes[ptName]:
                g[field] = partTypes[ptName][field]

        # set particle counts
        NumPart = np.zeros( nPartTypes, dtype='int32' )
        for ptName in partTypes.keys():
            ptNum = int(ptName[-1])
            NumPart[ptNum] = partTypes[ptName]['ParticleIDs'].size

        # create standard header
        h = f.create_group('Header')
        h.attrs['BoxSize'] = boxSize
        h.attrs['NumFilesPerSnapshot'] = 1
        h.attrs['NumPart_ThisFile'] = NumPart
        h.attrs['NumPart_Total'] = NumPart
        h.attrs['NumPart_Total_HighWord'] = np.zeros( nPartTypes, dtype='int32' )

        if headerExtra is not None:
            for key in headerExtra.keys():
                h.attrs[key] = headerExtra[key]

        for k in ['Time','Redshift','Omega0','OmegaLambda','HubbleParam']:
            if headerExtra is not None and k in headerExtra: continue
            h.attrs[k] = 0.0
        for k in ['Sfr','Cooling','StellarAge','Metals','Feedback','DoublePrecision']:
            if headerExtra is not None and k in headerExtra: continue
            h.attrs['Flag_%s' % k] = 0

        if massTable is not None:
            h.attrs['MassTable'] = massTable
        else:
            h.attrs['MassTable'] = np.zeros( nPartTypes, dtype='float64' )

def visualize_result_2d(basePath):
    """ Helper function to load density_field_NNN projection files and plot a series of PNG frames. """
    path = basePath + '/output/'

    #vMM = None # automatic
    #figsize = (16,14)

    figsize = None # automatic
    clean = True

    # config3
    vMM = [0.2, 5.0]
    cmap = 'viridis'

    # config4
    #vMM = [0.6, 4.0]
    #cmap = 'twilight_shifted'

    # config7
    #vMM = [0.4, 1.2]
    #cmap = 'magma'


    # loop over snapshots
    nSnaps = len(glob.glob(path + 'density_field_*'))

    for i in range(nSnaps):
        print(i)

        # load
        with open(path + 'density_field_%03d' % i, mode='rb') as f:
            data = f.read()

        # unpack
        nPixelsX = struct.unpack('i', data[0:4])[0]
        nPixelsY = struct.unpack('i', data[4:8])[0]

        nGridFloats = int((len(data)-8) / 4)
        grid = struct.unpack('f' * nGridFloats, data[8:])
        grid = np.array(grid).reshape( (nPixelsX,nPixelsY) )

        # fix border artifacts
        if 1:
            w = np.where( grid[0,:]>(grid[0,:].mean()+np.std(grid[0,:])) )
            grid[0,w] = grid[1,w]

            w = np.where( grid[-1,:]>(grid[-1,:].mean()+np.std(grid[-1,:])) )
            grid[-1,w] = grid[-2,w]

            w = np.where( grid[:,0]>(grid[:,0].mean()+np.std(grid[:,0])) )
            grid[w,0] = grid[w,1]

            w = np.where( grid[:,-1]>(grid[:,-1].mean()+np.std(grid[:,-1])) )
            grid[w,-1] = grid[w,-2]

            fac = 1.3
            w = np.where( grid[:,0] > grid[:,1]*fac )
            grid[w,0] = grid[w,1]
            w = np.where( grid[:,-1] > grid[:,-2]*fac )
            grid[w,-1] = grid[w,-2]
            w = np.where( grid[0,:] > grid[1,:]*fac )
            grid[w,0] = grid[w,1]
            w = np.where( grid[-1,:] > grid[-2,:]*fac )
            grid[w,-1] = grid[w,-2]

        if vMM is None: vMM = [grid.min(), grid.max()] # set on first snap

        # get time
        with h5py.File(path + 'snap_%03d.hdf5' % i,'r') as f:
            time = f['Header'].attrs['Time']
            boxSize = f['Header'].attrs['BoxSize']

        # start plot
        if figsize is None: figsize = (nPixelsX/100,nPixelsY/100) # exact

        fig = plt.figure(figsize=figsize)

        # plot (clean)
        if clean:
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()

            plt.imshow(grid.T, cmap=cmap, aspect='equal')
            ax.autoscale(False)
            plt.clim(vMM)

            ax.text(nPixelsX-10, nPixelsY-10, "t = %5.3f" % time, color='white', alpha=0.6, ha='right', va='top')

        # plot (with axes, colorbar)
        if not clean:
            ax = fig.add_subplot(111)            

            plt.imshow(grid.T, extent=[0,boxSize,0,boxSize], cmap=cmap, aspect='equal')
            ax.autoscale(False)
            plt.clim(vMM)

            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_title('snapshot %03d' % i)

            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel('Density')

            fig.tight_layout()

        fig.savefig('density_%d.png' % i)
        plt.close(fig)

    # if ffmpeg exists, make a movie
    cmd = ['ffmpeg','-framerate','30','-i','density_%d.png','-pix_fmt','yuv420p','movie.mp4','-y']
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd='%s/vis/' % path)
    except:
        pass
