"""
common.py
  Visualizations: common routines.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import hashlib
from os.path import isfile, isdir
from os import mkdir

from util.sphMap import sphMap
from illustris_python.util import partTypeNum
import cosmo

def getHsmlForPartType(sP, partType, indRange=None):
    """ Calculate an approximate HSML (smoothing length, i.e. spatial size) for particles of a given 
    type, for the full snapshot, optionally restricted to an input indRange. """
    # dark matter
    if partTypeNum(partType) == partTypeNum('dm'):
        raise Exception('Not implemented, use CalcHSML (with caching).')

    # gas
    if partTypeNum(partType) == partTypeNum('gas'):
        hsml = cosmo.load.snapshotSubset(sP, partType, 'cellrad', indRange=indRange)

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
        mass = cosmo.load.snapshotSubset(sP, partType, 'mass', indRange=indRange)
    elif partType == 'dm':
        h = cosmo.load.snapshotHeader(sP)
        mass = h.massTable[cosmo.util.partTypeNum('dm')]

    if partField == 'hi_test':
        nh0_frac = cosmo.load.snapshotSubset(sP, partType, 'NeutralHydrogenAbundance', indRange=indRange)
        mass *= sP.units.hydrogen_massfrac * nh0_frac

    # quantity
    if partField in ['coldens','coldens_cgs','hi_test']:
        # distribute mass and calculate column density grid
        quant = None
    else:
        # distribute a mass-weighted quantity and calculate mean value grid
        quant = cosmo.load.snapshotSubset(sP, partType, partField, indRange=indRange)

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

def gridBox(sP, method, partType, partField, nPixels, axes, boxCenter, boxSizeImg):
    """ Caching gridding/imaging of a simulation box. """
    m = hashlib.sha256('nPx-%d-%d.cen-%g-%g-%g.size-%g-%g-%g.axes=%d%d' % \
        (nPixels[0], nPixels[1], boxCenter[0], boxCenter[1], boxCenter[2], 
         boxSizeImg[0], boxSizeImg[1], boxSizeImg[2], axes[0], axes[1])).hexdigest()[::4]

    saveFilename = sP.derivPath + 'grids/%s.%d.%s.%s.%s.hdf5' % \
                   (method, sP.snap, partType, partField, m )

    if not isdir(sP.derivPath + 'grids/'):
        mkdir(sP.derivPath + 'grids/')

    # map
    if isfile(saveFilename):
        # load if already made
        with h5py.File(saveFilename,'r') as f:
            grid = f['grid'][...]
    else:
        # load: 3D positions
        pos = cosmo.load.snapshotSubset(sP, partType, 'pos')

        # load: mass/weights and quantity
        mass, quant = loadMassAndQuantity(sP, partType, partField)

        if method == 'sphMap':
            # particle by particle orthographic splat using standard SPH cubic spline kernel
            hsml = getHsmlForPartType(sP, partType)

            grid = sphMap( pos=pos, hsml=hsml, mass=mass, quant=quant, axes=axes, ndims=3, 
                           boxSizeSim=sP.boxSize, boxSizeImg=boxSizeImg, boxCen=boxCenter, nPixels=nPixels, 
                           colDens=(quant is None) )
        else:
            raise Exception('Not implemented.')

        # save
        #with h5py.File(saveFilename,'w') as f:
        #    f['grid'] = grid

    # handle units and come up with units label
    grid, config = gridOutputProcess(sP, grid, partField)

    return grid, config
