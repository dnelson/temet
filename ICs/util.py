"""
ics/util.py
  Idealized initial conditions: utility (common) functions.
"""
import numpy as np
import h5py

import struct
import glob
import matplotlib.pyplot as plt
import subprocess
from mpl_toolkits.axes_grid1 import make_axes_locatable

def write_ic_file(fileName, partTypes, boxSize, massTable=None):
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

        for k in ['Time','Redshift','Omega0','OmegaLambda','HubbleParam']:
            h.attrs[k] = 0.0
        for k in ['Sfr','Cooling','StellarAge','Metals','Feedback','DoublePrecision']:
            h.attrs['Flag_%s' % k] = 0

        if massTable is not None:
            h.attrs['MassTable'] = massTable
        else:
            h.attrs['MassTable'] = np.zeros( nPartTypes, dtype='float64' )

def visualize_result_2d(basePath):
    """ Helper function to load density_field_NNN projection files and plot a series of PNG frames. """
    boxSize = 1.0
    path = basePath + '/output/density_field_'

    vMM = None
    figsize = (16,14)

    # loop over snapshots
    nSnaps = len(glob.glob(path+'*'))

    for i in range(nSnaps):
        print(i)

        # load
        with open(path + '%03d' % i, mode='rb') as f:
            data = f.read()

        # unpack
        nPixelsX = struct.unpack('i', data[0:4])[0]
        nPixelsY = struct.unpack('i', data[4:8])[0]

        nGridFloats = (len(data)-8) / 4
        grid = struct.unpack('f' * nGridFloats, data[8:])
        grid = np.array(grid).reshape( (nPixelsX,nPixelsY) )

        if vMM is None: vMM = [grid.min(), grid.max()] # set on first snap

        # plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('snapshot %03d' % i)

        plt.imshow(grid.T, extent=[0,boxSize,0,boxSize], cmap='magma', aspect=1.0)
        ax.autoscale(False)
        plt.clim(vMM)

        # colorbar
        cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
        cb = plt.colorbar(cax=cax)
        cb.ax.set_ylabel('Density')

        fig.tight_layout()
        fig.savefig('density_%d.png' % i)
        plt.close(fig)

    # if ffmpeg exists, make a movie
    cmd = ['ffmpeg','-framerate','10','-i','density_%d.png','-pix_fmt','yuv420p','movie.mp4','-y']
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd='%s/vis/' % path)
    except:
        pass