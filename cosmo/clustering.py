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
from cosmo.color import loadSimGalColors
from util.tpcf import tpcf

def twoPointAutoCorrelationPeriodicCube(sP, cenSatSelect='all', minRad=10.0, numRadBins=40, 
      colorBin=None, cType=None, mstarBin=None, mType=None, jackKnifeNumSub=4):
    """ Calculate the two-point auto-correlation function in a periodic cube geometry. 
    If colorBin or mstarBin is not None, then a (min,max) tuple. Then we require the corresponding 
    cType=[bands,simColorsModel] or mType='appropriate cosmo.general.loadSimSubhaloQuant string'.
    Optional inputs: cenSatSelect, minRad (in code units), numRadBins. """
    assert cenSatSelect in ['all','cen','sat']
    savePath = sP.derivPath + "/clustering/"

    saveStr = 'rad-%d-%.1f' % (numRadBins,minRad)
    if colorBin is not None:
        assert cType is not None
        bands, simColorsModel = cType
        saveStr += '_color-%s-%.1f-%.1f-%s' % (''.join(bands),colorBin[0],colorBin[1],simColorsModel)
    if mstarBin is not None:
        assert mType is not None
        saveStr += '_mass-%.1f-%.1f-%s' % (mstarBin[0],mstarBin[1],mType)
    if jackKnifeNumSub is not None:
        saveStr += '_err-%d' % jackKnifeNumSub

    saveFilename = savePath + "tpcf_%d_%s_%s.hdf5" % (sP.snap,cenSatSelect,saveStr)

    if not isdir(savePath):
        mkdir(savePath)

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            rad = f['rad'][()]
            xi  = f['xi'][()]

            xi_err = f['xi_err'][()] if 'xi_err' in f else None
            covar  = f['covar'][()] if 'covar' in f else None

        return rad, xi, xi_err, covar

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    # get cenSatSelect indices, load and restrict if requested
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)

    pos = groupCat(sP, fieldsSubhalos=['SubhaloPos'])['subhalos']
    pos = np.squeeze(pos[wSelect,:])

    if colorBin is not None:
        # load simulation colors
        gc_colors, gc_ids = loadSimGalColors(sP, simColorsModel, bands=bands)
        assert np.array_equal(gc_ids, np.arange(sP.numSubhalos))
        gc_colors = gc_colors[wSelect]

        with np.errstate(invalid='ignore'):
            wColor = np.where( (gc_colors >= colorBin[0]) & (gc_colors < colorBin[1]) )

        pos = np.squeeze(pos[wColor,:])

    if mstarBin is not None:
        # load stellar masses
        gc_masses, _, _, _ = simSubhaloQuantity(sP, mType)
        gc_masses = gc_masses[wSelect]

        if colorBin is not None:
            # apply existing color restriction if applicable
            gc_masses = np.squeeze(gc_masses[wColor])

        with np.errstate(invalid='ignore'):
            wMass = np.where( (gc_masses >= mstarBin[0]) & (gc_masses < mstarBin[1]) )
        pos = np.squeeze(pos[wMass,:])

    # radial bins
    maxRad = sP.boxSize/2
    radialBins = np.logspace( np.log10(minRad), np.log10(maxRad), numRadBins)

    rrBinSizeLog = (np.log10(maxRad) - np.log10(minRad)) / numRadBins
    rad = 10.0**(np.log10(radialBins) + rrBinSizeLog/2)[:-1]

    # quick time estimate
    nPts = pos.shape[0]
    calc_time_sec = (float(pos.shape[0])/1e5)**2 * 600.0 / 16.0
    print(' nPts = %d, estimated time = %.1f sec (%.1f min) (%.2f hours) (%.2f days)' % \
        (nPts,calc_time_sec,calc_time_sec/60.0,calc_time_sec/3600.0,calc_time_sec/3600.0/24.0))

    # calculate two-point correlation function
    xi = tpcf(pos, radialBins, sP.boxSize)

    # if requested, calculate NSub^3 additional tpcf for jackknife error estimation
    xi_err = None
    covar = None

    if jackKnifeNumSub is not None:
        nSubs = jackKnifeNumSub**3
        subSize = sP.boxSize / jackKnifeNumSub

        xi_sub = np.zeros( (xi.size, nSubs), dtype='float32' )
        count = 0

        for i in range(jackKnifeNumSub):
            for j in range(jackKnifeNumSub):
                for k in range(jackKnifeNumSub):
                    # define spatial region
                    x0 = i * subSize
                    x1 = (i+1) * subSize
                    y0 = j * subSize
                    y1 = (j+1) * subSize
                    z0 = k * subSize
                    z1 = (k+1) * subSize

                    # exclude this sub-region and create reduced point set
                    w = np.where( ((pos[:,0] <= x0) | (pos[:,0] > x1)) | \
                                  ((pos[:,1] <= y0) | (pos[:,1] > y1)) | \
                                  ((pos[:,2] <= z0) | (pos[:,2] > z1)) )

                    print(' [%2d] (%d %d %d) x=[%7d,%7d] y=[%7d,%7d] z=[%7d,%7d] keep %6d points...' % \
                      (count,i,j,k,x0,x1,y0,y1,z0,z1,len(w[0])))

                    # calculcate and save tpcf
                    pos_loc = np.squeeze(pos[w,:])
                    xi_sub[:,count] = tpcf(pos_loc, radialBins, sP.boxSize)
                    count += 1

        # calculate covariance matrix
        covar = np.zeros( (xi.size, xi.size), dtype='float32' )
        for j in range(xi.size):
            for k in range(xi.size):
                covar[j,k] = (nSubs-1.0)/nSubs * np.sum( (xi_sub[k,:]-xi[k])*(xi_sub[j,:]-xi[j]) )

        # normalize and take standard deivations from the diagonal
        covar /= xi.size
        xi_err = np.sqrt( np.diag(covar) )

    with h5py.File(saveFilename,'w') as f:
        f['rad'] = rad
        f['xi'] = xi
        if xi_err is not None:
            f['xi_err'] = xi_err
            f['covar'] = covar
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    return rad, xi, xi_err, covar
