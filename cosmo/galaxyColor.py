"""
galaxyColor.py
  Calculations for TNG flagship paper: galaxy colors, color bimodality.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
from os.path import isfile, isdir, expanduser
from os import mkdir
from glob import glob
from scipy.optimize import leastsq

from util.loadExtern import loadSDSSData
from cosmo.kCorr import kCorrections, coeff
from cosmo.load import groupCat, groupCatHeader, auxCat
from cosmo.util import correctPeriodicDistVecs, cenSatSubhaloIndices
from cosmo.mergertree import loadMPBs
from plot.config import defSimColorModel
from plot.general import bandMagRange

# currently same for all sims, otherwise move into sP:
gfmBands = {'U':0, 'B':1, 'V':2, 'K':3,
            'g':4, 'r':5, 'i':6, 'z':7}

def loadSimGalColors(sP, simColorsModel, colorData=None, bands=None, projs=None):
    """ Load band-magnitudes either from snapshot photometrics or from auxCat SPS modeling, 
    and convert to a color if bands is passed in, otherwise return loaded data. If loaded 
    data is passed in with bands, do then magnitude computation without re-loading."""
    acKey = 'Subhalo_StellarPhot_' + simColorsModel

    if colorData is None:
        # load
        if simColorsModel == 'snap':
            colorData = groupCat(sP, fieldsSubhalos=['SubhaloStellarPhotometrics'])
        else:
            colorData = auxCat(sP, fields=[acKey])

    # early exit with full data?
    if bands is None:
        return colorData

    subhaloIDs = None

    # compute colors
    if simColorsModel == 'snap':
        gc_colors = stellarPhotToSDSSColor( colorData['subhalos'], bands )
    else:
        # which subhaloIDs do these colors correspond to?
        if 'subhaloIDs' in colorData:
            subhaloIDs = colorData['subhaloIDs']
        else:
            # could generate with a range() if this came up
            print(' warning: subhaloIDs not in [%s] auxCat.' % acKey)

        # band indices
        acBands = list(colorData[acKey+'_attrs']['bands'])
        i0 = acBands.index('sdss_'+bands[0])
        i1 = acBands.index('sdss_'+bands[1])

        # multiple projections per subhalo?
        if colorData[acKey].ndim == 3:
            if projs is None:
                print(' Warning: loadSimGalColors() projs unspecified, returning [random] by default.')
                projs = 'random'
            
            if projs == 'all':
                # return all
                gc_colors = colorData[acKey][:,i0,:] - colorData[acKey][:,i1,:]
            elif projs == 'random':
                # return one per subhalo, randomly chosen
                np.random.seed(42424242)
                nums = np.random.randint(0,high=colorData[acKey].shape[2],size=colorData[acKey].shape[0])
                all_inds = range(colorData[acKey].shape[0])
                gc_colors = colorData[acKey][all_inds,i0,nums] - colorData[acKey][all_inds,i1,nums]
            else:
                # otherwise, projs had better be an integer or a tuple
                assert isinstance(projs, (int,long,list,tuple))
                gc_colors = colorData[acKey][:,i0,projs] - colorData[acKey][:,i1,projs]
        else:
            # just one projection per subhalo
            gc_colors = colorData[acKey][:,i0] - colorData[acKey][:,i1]

    return gc_colors, subhaloIDs

def stellarPhotToSDSSColor(photVector, bands):
    """ Convert the GFM_StellarPhotometrics[] or SubhaloStellarPhotometrics[] vector into a 
    specified color, by choosing the right elements and handling any necessary conversions. """
    colorName = ''.join(bands)

    # dictionary of band name -> SubhaloStellarPhotometrics[:,i] index i
    ii = gfmBands

    if colorName == 'ui':
        # UBVK are in Vega, i is in AB, and U_AB = U_Vega + 0.79, V_AB = V_Vega + 0.02
        # http://www.astronomy.ohio-state.edu/~martini/usefuldata.html
        # assume Buser V = Johnson V filter (close-ish), and use Lupton2005 transformation
        # http://classic.sdss.org/dr7/algorithms/sdssUBVRITransform.html
        u_sdss_AB = photVector[:,ii['g']] + (-1.0/0.2906) * \
                    (photVector[:,ii['V']]+0.02 - photVector[:,ii['g']] - 0.0885)
        return u_sdss_AB - photVector[:,ii['z']]

    if colorName == 'gr':
        return photVector[:,ii['g']] - photVector[:,ii['r']] + 0.0 # g,r in sdss AB magnitudes

    if colorName == 'ri':
        return photVector[:,ii['r']] - photVector[:,ii['i']] + 0.0 # r,i in sdss AB magnitudes

    if colorName == 'iz':
        return photVector[:,ii['i']] - photVector[:,ii['z']] + 0.0 # i,z in sdss AB magnitudes

    if colorName == 'gz':
        return photVector[:,ii['g']] - photVector[:,ii['z']] + 0.0 # g,z in sdss AB magnitudes

    raise Exception('Band combination not implemented.')

def calcSDSSColors(bands, redshiftRange=None, eCorrect=False, kCorrect=False):
    """ Load the SDSS data files and compute a requested color, optionally restricting to a given 
    galaxy redshift range, correcting for extinction, and/or doing a K-correction. """
    assert redshiftRange is None, 'Not implemented.'

    sdss = loadSDSSData()

    # extinction correction
    if not eCorrect:
        for key in sdss.keys():
            if 'extinction_' in key:
                sdss[key] *= 0.0

    sdss_color = (sdss[bands[0]]-sdss['extinction_'+bands[0]]) - \
                 (sdss[bands[1]]-sdss['extinction_'+bands[1]])
    sdss_Mstar = sdss['logMass_gran1']

    # K-correction (absolute_M = apparent_m - C - K) (color A-B = m_A-C-K_A-m_B+C+K_B=m_A-m_B+K_B-K_A)
    if kCorrect:
        kCorrs = {}

        for band in bands:
            availCorrections = [key.split('_')[1] for key in coeff.keys() if band+'_' in key]
            useCor = availCorrections[0]
            #print('Calculating K-corr for [%s] band using [%s-%s] color.' % (band,useCor[0],useCor[1]))

            cor_color = (sdss[useCor[0]]-sdss['extinction_'+useCor[0]]) - \
                        (sdss[useCor[1]]-sdss['extinction_'+useCor[1]])

            kCorrs[band] = kCorrections(band, sdss['redshift'], useCor, cor_color)

        sdss_color += (kCorrs[bands[1]] - kCorrs[bands[0]])

    return sdss_color, sdss_Mstar

def calcMstarColor2dKDE(bands, gal_Mstar, gal_color, Mstar_range, mag_range, sP=None, simColorsModel=None):
    """ Quick caching of (slow) 2D KDE calculation of (Mstar,color) plane for SDSS z<0.1 points 
    if sP is None, otherwise for simulation (Mstar,color) points if sP is specified. """
    if sP is None:
        saveFilename = expanduser("~") + "/obs/sdss_2dkde_%s_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10)
        dName = 'kde_obs'
    else:
        assert simColorsModel is not None
        savePath = sP.derivPath + "/galMstarColor/"

        if not isdir(savePath):
            mkdir(savePath)

        saveFilename = savePath + "galMstarColor_2dkde_%s_%s_%d_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),simColorsModel,sP.snap,
            Mstar_range[0]*10,Mstar_range[1]*10,mag_range[0]*10,mag_range[1]*10)
        dName = 'kde_sim'

    # check existence
    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            xx = f['xx'][()]
            yy = f['yy'][()]
            kde_obs = f[dName][()]

        return xx, yy, kde_obs

    # calculate
    print('Calculating new: [%s]...' % saveFilename)

    vv = np.vstack( [gal_Mstar, gal_color] )
    kde = gaussian_kde(vv)

    xx, yy = np.mgrid[Mstar_range[0]:Mstar_range[1]:200j, mag_range[0]:mag_range[1]:400j]
    xy = np.vstack( [xx.ravel(), yy.ravel()] )
    kde2d = np.reshape( np.transpose(kde(xy)), xx.shape)

    # save
    with h5py.File(saveFilename,'w') as f:
        f['xx'] = xx
        f['yy'] = yy
        f[dName] = kde2d
    print('Saved: [%s]' % saveFilename)

    return xx, yy, kde2d

def calcColorEvoTracks(sP, bands=['g','r'], simColorsModel=defSimColorModel):
    """ Using already computed StellarPhot auxCat's at several snapshots, load the MPBs and 
    re-organize the band magnitudes into tracks in time for each galaxy. Do for one band 
    combination, saving only the color, while keeping all viewing angles. """
    import warnings

    savePath = sP.derivPath + "/auxCat/"

    # how many computed StellarPhot auxCats already exist?
    savedPaths = glob(savePath + "Subhalo_StellarPhot_%s_???.hdf5" % simColorsModel)
    savedSnaps = sorted([int(path[-8:-5]) for path in savedPaths], reverse=True)
    numSnaps = len( savedSnaps )

    # check existence
    saveFilename = savePath + "Subhalo_StellarPhotEvo_%s_%d-%d_%s.hdf5" % \
      (''.join(bands),sP.snap,numSnaps,simColorsModel)

    if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            colorEvo = f['colorEvo'][()]
            shIDEvo = f['shIDEvo'][()]
            subhaloIDs = f['subhaloIDs'][()]
            savedSnaps = f['savedSnaps'][()]

        return colorEvo, shIDEvo, subhaloIDs, savedSnaps

    # compute new:
    print('Found [%d] saved StellarPhot at snaps: %s.' % (numSnaps,', '.join([str(s) for s in savedSnaps])))

    # load z=0 colors and identify subset of SubhaloIDs we will track
    colors0, subhaloIDs = loadSimGalColors(sP, simColorsModel, bands=bands, projs='all')
    gcH = groupCatHeader(sP)

    if colors0.ndim > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            colors0_mean = np.nanmean(colors0, axis=1)
    else:
        colors0_mean = colors0.copy()

    assert colors0.shape[0] == gcH['Nsubgroups_Total'] == subhaloIDs.size
    assert colors0_mean.ndim == 1

    w = np.where( np.isfinite(colors0_mean) ) # keep subhalos with non-NaN colors

    print(' calcColorEvoTracks: keep [%d] of [%d] subhalos.' % (len(w[0]),subhaloIDs.size))
    subhaloIDs = subhaloIDs[w]

    # allocate
    numProjs = colors0.shape[1] if colors0.ndim > 1 else 1
    colorEvo = np.zeros( (subhaloIDs.size,numProjs,numSnaps), dtype='float32' )
    shIDEvo  = np.zeros( (subhaloIDs.size,numSnaps), dtype='int32' )

    colorEvo.fill(np.nan)
    shIDEvo.fill(-1)

    # load MPBs of the subhalo selection
    mpbs = loadMPBs(sP, subhaloIDs, fields=['SnapNum','SubfindID'])

    # walk backwards through snapshots where we have computed StellarPhot data
    origSnap = sP.snap

    for i, snap in enumerate(savedSnaps):
        sP.setSnap(snap)
        print(' [%d] snap = %d' % (i,snap))

        # load local colors
        colors_loc, subhaloIDs_loc = loadSimGalColors(sP, simColorsModel, bands=bands, projs='all')

        # loop over each z=0 subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            # get progenitor SubfindID at this snapshot
            if subhaloID not in mpbs:
                continue # not tracked anywhere

            snapInd = np.where( mpbs[subhaloID]['SnapNum'] == snap )[0]

            if len(snapInd) == 0:
                continue # not tracked to this snapshot

            # map progenitor to its color at this snapshot, and save
            subhaloID_loc = mpbs[subhaloID]['SubfindID'][snapInd]

            colorEvo[j,:,i] = colors_loc[subhaloID_loc]
            shIDEvo[j,i] = subhaloID_loc

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename,'w') as f:
        f['colorEvo'] = colorEvo
        f['shIDEvo'] = shIDEvo
        f['subhaloIDs'] = subhaloIDs
        f['savedSnaps'] = savedSnaps
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    return colorEvo, shIDEvo, subhaloIDs, savedSnaps

def _fitCMPlaneDoubleGaussian(masses, colors, xMinMax, mag_range):
    """ Return the parameters of a full fit to objects in the (color-mass) plane. A double gaussian 
    is fit to each mass bin, histogramming in color, both bin sizes fixed herein. The centers and 
    widths of the Gaussians may be optionally constrained as inputs (todo). """

    def double_gaussian(x, params):
        """ Additive double gaussian function used for fitting. """
        (A1, mu1, sigma1, A2, mu2, sigma2) = params

        y = A1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
          + A2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
        return y

    def double_gaussian_error(params, x, y):
        y_fit = double_gaussian(x, params)
        return y_fit - y

    # config
    binSizeMass = 0.15
    binSizeColor = 0.05

    nBinsMass = int(np.ceil((xMinMax[1]-xMinMax[0])/binSizeMass))

    nParams = 6
    p = np.zeros( (nParams,nBinsMass), dtype='float32' )
    m = np.zeros( nBinsMass, dtype='float32' )

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i+1)

        wMassBin = np.where( (masses > minMass) & (masses <= maxMass) )

        if len(wMassBin[0]) == 0:
            print('Warning: empty mass bin.')

        colors_data = np.ravel( colors[wMassBin,:] ) # flatten into 1D

        # have to histogram (or 1D-KDE) them, to get a (x_data,y_data) point set
        nBins = int( (mag_range[1]-mag_range[0]) / binSizeColor )

        y_data, x_data = np.histogram(colors_data, range=mag_range, bins=nBins, density=True)
        x_data = x_data[:-1] + binSizeColor/2.0

        if 1:
            print('Debug, fake test data!')
            x_data = np.linspace( mag_range[0], mag_range[1], 100 )
            p0 = [0.5, 0.3+0.02*i, 0.2, 2.0, 0.8, 0.1]
            y_data = double_gaussian(x_data, p0)
            y_data += np.random.normal(loc=0.0, scale=p0[0]*0.05, size=x_data.size)

        # initial guess for (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        params_guess = [1.0, 0.25, 0.5, 1.0, 0.75, 0.5]

        # run fit
        params_best, params_cov, info, errmsg, retcode = \
          leastsq(double_gaussian_error, params_guess, args=(x_data,y_data), full_output=True)

        # sigma_i can fit negative since this is symmetric in the fit function...
        params_best[2] = np.abs(params_best[2])
        params_best[5] = np.abs(params_best[5])

        # estimate errors (unused)
        params_stddev = np.zeros( len(params_best), dtype='float32' )

        if params_cov is not None:
            # reduced chi^2 (i.e. residual variance)
            chi2 = np.sum(double_gaussian_error(params_best, x_data, y_data)**2.0)
            reduced_chi2 = chi2 / (y_data.size - len(params_best))

            # incorporate into fractional covariance matrix
            params_cov *= reduced_chi2

            # square root of diagonal elements estimates stddev of each parameter
            for j in range(len(params_best)):
                params_stddev[j] = np.abs(np.sqrt( params_cov[j][j] ))

        # save
        m[i] = minMass
        p[:,i] = params_best

        #print(minMass,params_best)

        # the integral of y_data is unity, so we probably have normalized Gaussians such that
        # A_i = 1.0/(sigma_i*np.sqrt(2*np.pi)) is constrained and not a free parameter?

    return p, m, binSizeMass, binSizeColor

def characterizeColorMassPlane(sP, bands=['g','r'], cenSatSelect='all', simColorsModel=defSimColorModel):
    """ Do double gaussian and other methods to characterize the red and blue populations, e.g. their 
    location, extent, relative numbers, for sP at sP.snap. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    # let us also achieve all the analysis necessary for figures 12-15 here at the same time
    mag_range = bandMagRange(bands, tight=False)
    xMinMax = [9.0, 12.0]

    # check existence
    r = {}

    savePath = sP.derivPath + "/galMstarColor/"
    saveFilename = savePath + "colorMassPlaneFits_%s_%d_%s_%s.hdf5" % \
      (''.join(bands),sP.snap,cenSatSelect,simColorsModel)

    if 0 and isfile(saveFilename): # load currently disabled
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        return r

    # load colors
    gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands, projs='all')
    #gc_colors = np.reshape( gc_colors, gc_colors.shape[0]*gc_colors.shape[1] )

    # load stellar masses (<2rhalf definition)
    gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
    mstar2_log = sP.units.codeMassToLogMsun( gc['subhalos'][:,sP.ptNum('stars')] )

    # cen/sat selection
    wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
    gc_colors = gc_colors[wSelect,:]
    mstar2_log = mstar2_log[wSelect]

    # (A) double gaussian fits in 0.1 dex mstar bins, unconstrained (unrelated)
    # Levenberg-Marquadrt non-linear least squares minimization method
    r['A_params'], r['A_mstar'], r['binSizeMass'], r['binSizeColor'] = \
      _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, xMinMax, mag_range)

    # (B) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
    # T() function as in Baldry+ 2003, iterative fit (LM-LSF for each step)
    def T(x, params):
        (p0, p1, q0, q1, q2) = params
        y = p0 + p1 * x + q0 * np.tanh( (x-q1)/q2 )
        return y

    for iter in range(4):
        # fit double gaussians, all mass bins
        p, m, _, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, xMinMax, mag_range)

        # force edge bins to extrapolation (only blue params at high mass end needed?)

        # fit T(sigma)

        # re-fit double gaussians with fixed sigma

        # fit T(mu)

        # calculate change of both T() versus previous iteration

    # re-fit double gaussians with fixed sigma and mu from final T() functions

    # (C) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
    # T() function as in Baldry+ 2003, global MCMC fit to all the parameters simultaneously
    # five for each T() function, plus 2 amplitudes per mass bin (=30x2+10=70!)


    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    return r
