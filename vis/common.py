"""
common.py
  Visualizations: common routines.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import hashlib
import h5py
from os.path import isfile, isdir
from os import mkdir

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from util.sphMap import sphMap
from illustris_python.util import partTypeNum
from cosmo.load import snapshotSubset, snapshotHeader, groupCat

def getHsmlForPartType(sP, partType, indRange=None):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """
    # dark matter
    if partTypeNum(partType) == partTypeNum('dm'):
        raise Exception('Not implemented, use CalcHSML (with caching).')

    # gas
    if partTypeNum(partType) == partTypeNum('gas'):
        hsml = snapshotSubset(sP, partType, 'cellrad', indRange=indRange)

        # TODO: check for existence of volume, if not, use Mass/Density
        return hsml

    # stars
    if partTypeNum(partType) == partTypeNum('stars'):
        raise Exception('Not implemented, use CalcHSML (with caching).')

    raise Exception('Unimplemented partType.')

def loadMassAndQuantity(sP, partType, partField, indRange=None):
    """ desc """
    # mass/weights
    if partType in ['gas','stars']:
        mass = snapshotSubset(sP, partType, 'mass', indRange=indRange)
    elif partType == 'dm':
        h = snapshotHeader(sP)
        mass = h.massTable[cosmo.util.partTypeNum('dm')]

    if partField == 'hi_test':
        nh0_frac = snapshotSubset(sP, partType, 'NeutralHydrogenAbundance', indRange=indRange)
        mass *= sP.units.hydrogen_massfrac * nh0_frac

    # quantity
    if partField in ['coldens','coldens_cgs','hi_test']:
        # distribute mass and calculate column density grid
        quant = None
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        quant = snapshotSubset(sP, partType, partField, indRange=indRange)

    # unit pre-processing (only need to remove log for means)
    if partField == 'temp':
        quant = 10.0**quant

    return mass, quant

def gridOutputProcess(sP, grid, partField):
    """ desc """
    config = {}

    if partField == 'coldens':
        grid  = np.log10( sP.units.codeColDensToPhys( grid, cgs=False, numDens=False ) )
        config['label'] = 'Column Density [log 10$^{10}$ Msun / kpc$^2$]'

    if partField == 'coldens_cgs':
        grid  = np.log10( sP.units.codeColDensToPhys( grid, cgs=True, numDens=True ) )
        config['label'] = 'Column Density [log cm$^{-2}$]'

    if partField == 'hi_test':
        grid = sP.units.codeColDensToPhys(grid, cgs=True, numDens=True)
        grid = np.log10(grid)
        config['label'] = 'N$_{\\rm HI}$ [log cm$^{-2}$]'

    if partField == 'temp':
        grid  = np.log10( grid )
        config['label'] = 'Temperature [log K]'

    if partField == 'velmag':
        config['label'] = 'Velocity Magnitude [km/s]'

    config['ctName'] = 'todo'
    return grid, config

def gridBox(sP, method, partType, partField, nPixels, axes, boxCenter, boxSizeImg, hsmlFac, **kwargs):
    """ Caching gridding/imaging of a simulation box. """
    m = hashlib.sha256('nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1])).hexdigest()[::4]

    saveFilename = sP.derivPath + 'grids/%s.%d.%s.%s.%s.hdf5' % (method, sP.snap, partType, partField, m)

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')

    # map
    if isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid = f['grid'][...]
        print('Loaded: [%s]' % saveFilename)
    else:
        # load: 3D positions
        pos = snapshotSubset(sP, partType, 'pos')

        # load: mass/weights and quantity
        mass, quant = loadMassAndQuantity(sP, partType, partField)

        if method == 'sphMap':
            # particle by particle orthographic splat using standard SPH cubic spline kernel
            hsml = getHsmlForPartType(sP, partType) * hsmlFac

            grid = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                           boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, boxCen=boxCenter, nPixels=nPixels, 
                           colDens=(quant is None) )
        else:
            raise Exception('Not implemented.')

        # save
        with h5py.File(saveFilename,'w') as f:
            f['grid'] = grid
        print('Saved: [%s]' % saveFilename)

    # handle units and come up with units label
    grid, config = gridOutputProcess(sP, grid, partField)

    return grid, config

def renderMultiPanel(panels, plotStyle, rasterPx, saveFilename):
    """ Generalized plotting function which produces a single multi-panel plot with one panel for 
        each of panels, all of which can vary in their configuration. 
      plotStyle    : open, edged
      rasterPx     : each panel will have this number of pixels if making a raster (png) output, 
                     but note also it controls the relative size balance of raster/vector (e.g. fonts)
      saveFilename : output file name (extension determines type e.g. pdf or png)

    Each panel in panels must be a dictionary containing the following keys:
      sP         : simParams() object specifying the box, e.g. run, res, and redshift
      partType   : dm, gas, stars, tracerMC, ...
      partField  : coldens, coldens_cgs, temp, velmag, entr, ...
      method     : sphMap, voronoi_const, voronoi_grads, ...
      nPixels    : number of pixels per dimension of images when projecting
      axes       : e.g. [0,1] is x,y
      boxSizeImg : (x,y,z) extent of the imaging box in simulation units
      boxCenter  : (x,y,z) coordinates of the imaging box center in simulation units
      extent     : (axis0_min,axis0_max,axis1_min,axis1_max) in simulation units
    """

    if plotStyle == 'open':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig = plt.figure(figsize=(1.167*sizeFac*nRows/aspect,sizeFac*nRows))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = p['sP']

            # grid projection for image
            print(sP.run,sP.res,sP.redshift,p['partType'],p['partField'])

            grid, config = gridBox(**p)

            # set axes and place image
            ax = fig.add_subplot(nRows,nCols,i+1)
            ax.set_title('%s %d z=%3.1f %s %s' % (sP.simName,sP.res,sP.redshift,p['partType'],p['partField']))
            ax.set_xlabel( ['x','y','z'][p['axes'][0]] + ' [ ckpc/h ]')
            ax.set_ylabel( ['x','y','z'][p['axes'][1]] + ' [ ckpc/h ]')

            plt.imshow(grid, extent=p['extent'], aspect=1.0)

            if 1:
                # debug plotting 10 most massive halos
                gc = groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

                for j in range(20):
                    xPos = gc['halos']['GroupPos'][j,p['axes'][0]]
                    yPos = gc['halos']['GroupPos'][j,p['axes'][1]]
                    rad  = gc['halos']['Group_R_Crit200'][j] * 1.0

                    c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                    ax.add_artist(c)

            # colobar
            cax = make_axes_locatable(ax).append_axes('right', size='4%', pad=0.2)
            cb = plt.colorbar(cax=cax)
            cb.ax.set_ylabel(config['label'])

        fig.tight_layout()
        fig.savefig(saveFilename)

    if plotStyle == 'edged':
        # start plot
        nRows  = np.floor(np.sqrt(len(panels)))
        nCols  = len(panels) / nRows
        aspect = nRows/nCols

        fig = plt.figure(frameon=False, tight_layout=False)
        barAreaHeight = 0.1
        
        sizeFac = rasterPx / mpl.rcParams['savefig.dpi']
        fig.set_size_inches(sizeFac*nCols, sizeFac*nRows*(1/(1.0-barAreaHeight)))

        # for each panel: paths and render setup
        for i, p in enumerate(panels):
            sP = p['sP']

            # grid projection for image
            print(sP.run,sP.res,sP.redshift,p['partType'],p['partField'])

            grid, config = gridBox(**p)

            # set axes and place image
            curRow = np.floor(i / nCols)
            curCol = i % nCols

            rowHeight  = (1.0 - barAreaHeight) / nRows
            colWidth   = 1.0 / nCols
            bottomNorm  = 1.0 - rowHeight * (curRow+1)
            leftNorm = colWidth * curCol

            pos = [leftNorm, bottomNorm, colWidth, rowHeight]
            print(curRow,curCol,pos)
            ax = fig.add_axes(pos)
            ax.set_axis_off()

            plt.imshow(grid, extent=p['extent'], aspect=1.0)

            if 1:
                # debug plotting 10 most massive halos
                gc = groupCat(sP, fieldsHalos=['GroupPos','Group_R_Crit200'], skipIDs=True)

                for j in range(20):
                    xPos = gc['halos']['GroupPos'][j,p['axes'][0]]
                    yPos = gc['halos']['GroupPos'][j,p['axes'][1]]
                    rad  = gc['halos']['Group_R_Crit200'][j] * 1.0

                    c = plt.Circle( (xPos,yPos), rad, color='#ffffff', linewidth=1.5, fill=False)
                    ax.add_artist(c)

            # colobar
            factor  = 0.95 # bar length, fraction of column width, 1.0=whole
            height  = 0.04 # colorbar height, fraction of entire figure
            hOffset = 0.4  # padding between image and start of bar (fraction of height)

            leftNormBar   = leftNorm + 0.5*colWidth*(1-factor)
            bottomNormBar = barAreaHeight - height*hOffset - height

            posBar = [leftNormBar, bottomNormBar, colWidth*factor, height]

            cax = fig.add_axes(posBar)
            cax.set_axis_off()
            cb = plt.colorbar(cax=cax, orientation='horizontal')
            cb.ax.set_ylabel(config['label'])

        fig.savefig(saveFilename)

    plt.close(fig)
