"""
clustering.py
  Calculations for TNG clustering paper.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import time
from os.path import isfile, isdir, expanduser
from os import mkdir

from cosmo.load import groupCat, groupCatHeader, auxCat
from plot.general import simSubhaloQuantity
from cosmo.util import cenSatSubhaloIndices
from util.tpcf import tpcf

def twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect='all', minRad=10.0, numRadBins=40, 
      colorBounds=None, cType=None, mstarBounds=None, mType=None):
    """ Calculate the two-point auto-correlation function in a periodic cube geometry. 
    If colorBounds or mstarBounds is not None, then a (min,max) tuple. Then we require the corresponding 
    cType=[bands,simColorsModel] or mType='appropriate cosmo.general.loadSimSubhaloQuant string'.
    Optional inputs: cenSatSelect, minRad (in code units), numRadBins. """
    assert cenSatSelect in ['all','cen','sat']
    savePath = sP.derivPath + "/clustering/"

    saveStr = 'rad-%d-%.1f' % (numRadBins,minRad)
    if colorBounds is not None:
        assert cType is not None
        bands, simColorsModel = cType
        saveStr += '_color-%s-%.1f-%.1f-%s' % (colorBounds[0],colorBounds[1],''.join(bands),simColorsModel)
    if mstarBounds is not None:
        assert mType is not None
        saveStr += '_mass-%.1f-%.1f-%s' % (mstarBounds[0],mstarBounds[1],mType)

    saveFilename = savePath + "tpcf_%d_%s_%s.hdf5" % (sP.snap,cenSatSelect,saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            rad = f['rad'][()]
            xi = f['xi'][()]

        return rad, xi

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    # get cenSatSelect indices, load and restrict if requested
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    pos = groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']
    pos = pos[wSelect,:]

    if colorBounds is not None:
        # load simulation colors
        gc_colors, gc_ids = loadSimGalColors(sP, simColorsModel, bands=bands)
        assert np.array_equal(gc_ids, np.arange(sP.numSubhalos))

        ww = np.where( (gc_colors >= colorBounds[0]) & (gc_colors < colorBounds[1]) )
        pos = pos[w,:]

        import pdb; pdb.set_trace() # verify

    if mstarBounds is not None:
        # load stellar masses
        gc_masses, _, _, _ = simSubhaloQuantity(sP, mType)

        ww = np.where( (gc_masses >= mstarBounds[0]) & (gc_masses < mstarBounds[1]) )
        pos = pos[w,:]

        import pdb; pdb.set_trace() # verify

    # radial bins
    maxRad = sP.boxSize/2
    radialBins = np.logspace( np.log10(minRad), np.log10(maxRad), numRadBins)

    rrBinSizeLog = (np.log10(maxRad) - np.log10(minRad)) / numRadBins
    rad = 10.0**(np.log10(radialBins) + rrBinSizeLog/2)[:-1]

    # quick time estimate
    nPts = pos.shape[0]
    calc_time_sec = float(pos.shape[0])/1e5 * 600.0 / 16.0
    print(' nPts = %d, estimated time = %.1f sec (%.1f min) (%.2f hours) (%.2f days)' % \
        (nPts,calc_time_sec,calc_time_sec/60.0,calc_time_sec/3600.0,calc_time_sec/3600.0/24.0))

    # calculate two-point correlation function
    xi = tpcf(pos, radialBins, sP.boxSize)

    with h5py.File(saveFilename,'w') as f:
        f['rad'] = rad
        f['xi'] = xi
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    return rad, xi
