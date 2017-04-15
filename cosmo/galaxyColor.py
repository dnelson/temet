"""
galaxyColor.py
  Calculations for TNG flagship paper: galaxy colors, color bimodality.
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import time
from os.path import isfile, isdir, expanduser
from os import mkdir
from glob import glob
from scipy.interpolate import interp1d

from util.loadExtern import loadSDSSData
from util.helper import leastsq_fit, least_squares_fit
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

def _T(x, params, fixed=None):
    """ T() linear-tanh function of Baldry+ (2003). Note: fixed argument unused. """
    (p0, p1, q0, q1, q2) = params
    y = p0 + p1 * x + q0 * np.tanh( (x-q1)/q2 )
    return y

def _double_gaussian(x, params, fixed=None):
    """ Additive double gaussian function used for fitting. """
    if fixed is not None:
        assert len(fixed) == len(params)
        for i in range(len(fixed)):
            if fixed[i] is not None: params[i] = fixed[i]

    # pull out the 6 params of the 2 gaussians
    (A1, mu1, sigma1, A2, mu2, sigma2) = params

    y = A1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
      + A2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    return y

def _double_gaussian_rel(x, params, fixed=None):
    """ Additive double gaussian test function, each normalized with a relative amplitude parameter. """
    if fixed is not None:
        assert len(fixed) == len(params)
        for i in range(len(fixed)):
            if fixed[i] is not None: params[i] = fixed[i]

    # pull out the 5 params of the 2 gaussians
    (mu1, sigma1, mu2, sigma2, Afrac) = params

    A1 = Afrac / np.sqrt(2*np.pi) / sigma1
    A2 = (1.0-Afrac) / np.sqrt(2*np.pi) / sigma2
    y = A1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
      + A2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )

    return y

def _schechter_function(x, params, fixed=None):
    """ Schecter phi function (x=log Mstar) for params=[phinorm,alpha,M']. Note: fixed unused. """
    (phi_norm, alpha, M_characteristic) = params
    x = x.astype('float64')

    y = phi_norm * (x/M_characteristic)**(-alpha) * np.exp( -x/M_characteristic )
    return y

def _fitCMPlaneDoubleGaussian(masses, colors, xMinMax, mag_range, binSizeMass, binSizeColor, 
                              paramInds, paramIndsRel, nBinsMass, nBinsColor, 
                              relAmp=False, fixed=None, **kwargs):
    """ Return the parameters of a full fit to objects in the (color-mass) plane. A double gaussian 
    is fit to each mass bin, histogramming in color, both bin sizes fixed herein. The centers and 
    widths of the Gaussians may be optionally constrained as inputs with fixed. """

    if relAmp:
        # only one relative amplitude parameter is fit
        fit_func = _double_gaussian_rel
        # initial guess for (mu1, sigma1, mu2, sigma2, Arel) (1=blue, 2=red)
        params_guess = [0.35, 0.1, 0.75, 0.1, 0.5]

        params_bounds = (0.0, 1.0-1e-6) # none of mu,sigma,A_rel can be negative or bigger than 1

        pInds = paramIndsRel
        nParams = len(pInds)
        assert nParams == 5
    else:
        # two free amplitude parameters are fit
        fit_func = _double_gaussian
        # initial guess for (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        params_guess = [1.0, 0.35, 0.1, 1.0, 0.75, 0.1]

        params_bounds = (0.0, (np.inf,1.0,1.0,np.inf,1.0,1.0)) # none of mu,sigma <0 or >1, A_i >0 only
        
        pInds = paramInds
        nParams = len(pInds)
        assert nParams == 6

    # enforce reasonable values of Tparams
    if fixed is not None:
        for key in fixed:
            w = np.where( (fixed[key] <= 0.0) | (fixed[key] > 1.0) )
            fixed[key][w] = 0.01 # set to a value inside bounds

    # allocate
    p = np.zeros( (nParams,nBinsMass), dtype='float32' ) # parameters per mass bin
    m = np.zeros( nBinsMass, dtype='float32' ) # mass bin centers
    y = np.zeros( (nBinsMass,nBinsColor), dtype='float32' ) # 2d histogram
    n = np.zeros( nBinsMass, dtype='int32' ) # counts per mass bin

    p.fill(np.nan)

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i+1)
        m[i] = minMass + binSizeMass/2.0

        wMassBin = np.where( (masses > minMass) & (masses <= maxMass) )

        if len(wMassBin[0]) == 0:
            print('Warning: empty mass bin.')

        if colors.ndim == 2:
            colors_data = np.ravel( colors[wMassBin,:] ) # flatten into 1D
        else:
            colors_data = colors[wMassBin] # obs/single projection

        # have to histogram (or 1D-KDE) them, to get a (x_data,y_data) point set
        n[i] = colors_data.size
        y_data, x_data = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        x_data = x_data[:-1] + binSizeColor/2.0

        y[i,:] = y_data

        # any fixed (non-varying) parameters?
        fixed_loc = None

        if fixed is not None:
            # nParams for this mass bin, start with none fixed
            fixed_loc = [None for _ in range(nParams)]

            # which field(s) are fixed? pull out value at this mass bin
            for paramName in pInds.keys():
                if paramName in fixed:
                    assert fixed[paramName].size == nBinsMass and fixed[paramName].ndim == 1
                    fixed_loc[pInds[paramName]] = fixed[paramName][i]

            # any fixed fields nan? then e.g. extrapolated _T() values of a sigma or mu were invalid, 
            # so we skip this bin and don't have any gaussian parameter fits here
            #skipBin = False
            #for f in fixed_loc:
            #    if f is not None and np.isnan(f): skipBin = True
            #if skipBin:
            #    print(' nan %s skip massBin %d' % (paramName,i))
            #    continue

        # run fit
        #params_best, _ = leastsq_fit(fit_func, params_guess, args=(x_data,y_data,fixed_loc))
        params_best = least_squares_fit(fit_func, params_guess, params_bounds, args=(x_data,y_data,fixed_loc))

        # sigma_i can fit negative since this is symmetric in the fit function...
        for pName in ['sigma_blue','sigma_red']:
            params_best[ pInds[pName] ] = np.abs(params_best[ pInds[pName] ])

        # blue/red choice for each gaussian is arbitrary, enforce that red = the one with redder center
        if params_best[ pInds['mu_blue'] ] > params_best[ pInds['mu_red'] ]:
            if not relAmp:
                params_best = np.roll(params_best,3) # swap first 3 and last 3
            else:
                # swap mu,sigma and switch relative amplitude
                params_best = params_best[ [2,3,0,1,4] ]
                params_best[4] = 1.0 - params_best[4]

        # re-normalize noisy amplitudes such that total integral is 1
        if not relAmp:
            int_blue = params_best[pInds['A_blue']] * \
              params_best[pInds['sigma_blue']] * np.sqrt(2*np.pi)
            int_red = params_best[pInds['A_red']] * \
              params_best[pInds['sigma_red']] * np.sqrt(2*np.pi)
            int_fac = 1.0 / (int_blue + int_red)
            params_best[pInds['A_blue']] *= int_fac
            params_best[pInds['A_red']] *= int_fac
        else:
            if params_best[pInds['A_rel']] >= 1.0 - 1e-4:
                params_best[pInds['A_rel']] = 1.0 - 1e-4 # for stability

        # save
        p[:,i] = params_best

    assert p.max() < 1.0

    # do a schechter function fit to (mass,counts) for red and blue separately
    params_guess = [1e4, 1.0, 10.0]
    p_sch = np.zeros(6, dtype='float32')

    if 0:
        # work in progress
        for i, c in enumerate(['red','blue']):
            # number of either red or blue galaxies in this mass bin:
            if not relAmp:
                y_amps = p[pInds['A_'+c],:]
            else:
                if c == 'blue': y_amps = p[pInds['A_rel']]
                if c == 'red': y_amps = 1.0 - p[pInds['A_rel']]

            N_gal = y_amps * p[pInds['sigma_'+c],:] * n * np.sqrt(2*np.pi)

            print(n.sum(), N_gal.sum())
            import pdb; pdb.set_trace()

            p_sch[i*3:(i+1)*3], _ = leastsq_fit(_schechter_function, params_guess, args=(m,N_gal))

    return p, p_sch, m, n, y, x_data

def _fitCMPlaneMCMC(masses, colors, chain_start, xMinMax, mag_range, skipNBinsRed, skipNBinsBlue, 
                    binSizeMass, binSizeColor, nBinsMass, nBinsColor, relAmp=False, sP_snap=0, **kwargs):
    """ MCMC based fit in the color-mass plane of the full double gaussian model. """
    import emcee

    # config
    nWalkers = 200
    nBurnIn = 400 # 200, 1000
    nProdSteps = 100 # 50, 100
    fracNoiseInit = 2e-3 # 1e-3
    percentiles = [1,10,50,90,99] # middle is used to derive the best-fit for each parameter (e.g. median)
    assert percentiles[int(len(percentiles)/2)] == 50

    # global MCMC fit to all the parameters simultaneously, five for each T() function, per color, 
    # plus 1 or 2 amplitudes per mass bin (=20x2 + 10*2=60)
    ampsPerMassBin = 1 if relAmp else 2
    nDim = ampsPerMassBin*nBinsMass + 2*5*2 # 20+20 or 40+20

    p0 = np.zeros( nDim, dtype='float32' )
    p0[00:05] = chain_start['sigma_blue']
    p0[05:10] = chain_start['sigma_red']
    p0[10:15] = chain_start['mu_blue']
    p0[15:20] = chain_start['mu_red']

    if relAmp:
        p0[20:40] = chain_start['A_rel'] # A_rel, 20 of them
        fit_func = _double_gaussian_rel
    else:
        p0[20:40] = chain_start['A_blue'] # blue A, 20 of them
        p0[40:60] = chain_start['A_red'] # red A, 20 of them
        fit_func = _double_gaussian

    assert np.all(np.isfinite(p0))

    def mcmc_lnprob_binned(theta, x, y, m, y_is2):
        # run four T() functions, get sigma_r,b(Mass) and mu_r,b(Mass)
        # compute all 20 double gaussians, in each mass bin have its 6 parameters
        lp = 0.0
        nMassBins = len(y)
        lnlike_y = np.zeros( nMassBins, dtype='float32' )

        # reconstruct sigma and mu values at each mass bin from the T() functions
        sigma1 = _T(m, theta[0:5])
        sigma2 = _T(m, theta[5:10])
        mu1 = _T(m, theta[10:15])
        mu2 = _T(m, theta[15:20])

        # 'absolute' priors, so just return early
        mu_mm = [0.0,1.0] # min max tophat prior
        sigma_mm = [0.0,1.0] # min max tophat prior

        #import pdb; pdb.set_trace()

        if (mu1 > mu2).sum() > 0:
            return -np.inf
        if mu1.min() <= mu_mm[0] or mu1.max() >= mu_mm[1] or \
           mu2.min() <= mu_mm[0] or mu2.max() >= mu_mm[1]:
           return -np.inf
        if sigma1.min() <= sigma_mm[0] or sigma1.max() >= sigma_mm[1] or \
           sigma2.min() <= sigma_mm[0] or sigma2.max() >= sigma_mm[1]:
           return -np.inf
        if relAmp:
            if theta[20:].min() < 0.0 or theta[20:].max() > 1.0:
                return -np.inf  # A_rel in [0,1]
        else:
            if theta[20:].min() < 0.0:
                return -np.inf # A_i >= 0

        # compute independent loglikelihood in each mass bin
        for i in range(nMassBins): #range(skipNBinsRed,nMassBins-skipNBinsBlue):
            # note: i never enters mass bins where we leave contribution at zero

            # pull out the remaining 2 parameters (A_blue, A_red) for this mass bin
            if relAmp:
                A_rel = theta[20+i]
                params_double_gaussian = (mu1[i], sigma1[i], mu2[i], sigma2[i], A_rel)
            else:
                A1, A2 = theta[20+i], theta[40+i]
                params_double_gaussian = (A1, mu1[i], sigma1[i], A2, mu2[i], sigma2[i])

            assert np.all( np.isfinite(params_double_gaussian) )

            #y_err = 0.05 # soften
            #inv_sigma2 = 1.0/y_err**2.0
            inv_sigma2 = y_is2[i]

            # compare histogramed data
            y_fit = fit_func(x, params_double_gaussian, fixed=None)
            chi2 = np.sum( (y_fit - y[i])**2.0 * inv_sigma2 - np.log(inv_sigma2) )
            lnlike_y[i] = -0.5 * chi2

        # all mass bin likelihoods multiplied -> added in the log
        return lp + np.sum(lnlike_y)

    # make x_data
    m = np.zeros(nBinsMass)
    y_data = []
    y_is2 = []

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i+1)

        wMassBin = np.where( (masses > minMass) & (masses <= maxMass) )

        if colors.ndim == 2:
            colors_data = np.ravel( colors[wMassBin,:] ) # flatten into 1D
        else:
            colors_data = colors[wMassBin] # obs/single projection

        yy, xx = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        xx = xx[:-1] + binSizeColor/2.0

        m[i] = minMass + binSizeMass/2.0
        y_data.append(yy)

        # error estimate
        y_err = np.zeros( yy.size, dtype='float32' )
        w = np.where(yy > 0.0)
        y_err[w] = 1.0 / np.sqrt(yy[w]*binSizeColor*yy.sum())
        w2 = np.where(y_err == 0.0)
        y_err[w2] = y_err[w].max() * 10.0
        inv_sigma2 = 1.0/y_err**2.0
        y_is2.append(inv_sigma2)

    # binned method: setup initial parameter guesses (theta0) for all walkers
    p0_walkers = np.zeros( (nWalkers,nDim), dtype='float32' )
    np.random.seed(42424242L)
    for i in range(nWalkers):
        p0_walkers[i,:] = p0 + np.abs(p0) * np.random.normal(loc=0.0, scale=fracNoiseInit, size=nDim)

    if relAmp:
        p0_walkers[:,20:40] = np.clip(p0_walkers[:,20:40], 0.0, 1.0-1e-3)

    # setup sampler and run a burn-in
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_binned, args=(xx, y_data, m, y_is2))

    pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
    sampler.reset()

    # run production chain
    sampler.run_mcmc(pos, nProdSteps)

    # ideally between 0.2 and 0.5:
    mean_acc = np.mean(sampler.acceptance_fraction)
    print('done sampling in [%.1f sec] mean acceptance frac: %.2f (binned)' % (time.time() - tstart,mean_acc))

    # calculate medians of production chains as answer
    samples = sampler.chain.reshape( (-1,nDim) )

    # record median as the answer, and reconstruct all the parameters as a function of Mstar
    # and sample the percentiles (e.g. in {mu,sigma,A}-space not in Tparam-space)
    if relAmp:
        nParams = 5
    else:
        nParams = 6

    percs = np.percentile(samples, percentiles, axis=0)
    best_params = percs[int(len(percentiles)/2),:]
    assert best_params.size == nDim

    p_error_accum = np.zeros( (nParams,nBinsMass,nWalkers*nProdSteps), dtype='float32' )

    for i in range(samples.shape[0]):
        sample = samples[i,:]

        if relAmp:
            # (mu1, sigma1, mu2, sigma2, A_rel) (1=blue, 2=red)
            p_error_accum[0,:,i] = _T(m, sample[10:15]) # mu blue
            p_error_accum[1,:,i] = _T(m, sample[0:5]) # sigma blue
            p_error_accum[2,:,i] = _T(m, sample[15:20]) # mu red
            p_error_accum[3,:,i] = _T(m, sample[5:10]) # sigma red
            p_error_accum[4,:,i] = sample[20:40] # A_rel
        else:
            # (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
            p_error_accum[1,:,i] = _T(m, sample[10:15]) # mu blue
            p_error_accum[2,:,i] = _T(m, sample[0:5]) # sigma blue
            p_error_accum[4,:,i] = _T(m, sample[15:20]) # mu red
            p_error_accum[5,:,i] = _T(m, sample[5:10]) # sigma red
            p_error_accum[0,:,i] = sample[20:40] # A_blue
            p_error_accum[3,:,i] = sample[40:60] # A_red

    # shape is [Npercs,Nparams,Nmassbins]
    p_errors = np.percentile(p_error_accum, percentiles, axis=2)

    # create return parameters, reconstructing the mu_i and sigma_i from the best T() parameters
    p = np.zeros( (nParams,nBinsMass), dtype='float32' ) 
    
    if relAmp:
        # (mu1, sigma1, mu2, sigma2, A_rel) (1=blue, 2=red)
        p[4,:] = best_params[20:40] # A_rel
        p[0,:] = _T(m, best_params[10:15]) # mu blue
        p[2,:] = _T(m, best_params[15:20]) # mu red
        p[1,:] = _T(m, best_params[0:5]) # sigma blue
        p[3,:] = _T(m, best_params[5:10]) # sigma red
    else:
        # (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        p[0,:] = best_params[20:40] # blue A
        p[3,:] = best_params[40:60] # red A
        p[1,:] = _T(m, best_params[10:15]) # mu blue
        p[4,:] = _T(m, best_params[15:20]) # mu red
        p[2,:] = _T(m, best_params[0:5]) # sigma blue
        p[5,:] = _T(m, best_params[5:10]) # sigma red

    # debug plots
    if 1:
        print(' making debug plots...')
        import matplotlib.pyplot as plt
        import corner

        saveStr = '%s_snap%d_%d_%d_%d_%e' % (relAmp,sP_snap,nWalkers,nBurnIn,nProdSteps,fracNoiseInit)

        # (A) sigma vs. chain #
        fig = plt.figure(figsize=(18,12))

        for plotInd, i in enumerate(range(0,5)):
            ax = fig.add_subplot(5,2,plotInd+1)
            ax.set_xlabel('chain step')
            ax.set_ylabel('$\sigma_{\\rm blue}$ T[%d]' % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')
        for plotInd, i in enumerate(range(5,10)):
            ax = fig.add_subplot(5,2,plotInd+1+5)
            ax.set_ylabel('$\sigma_{\\rm red}$ T[%d]' % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')

        fig.tight_layout()
        fig.savefig('debug_methodC_%s_sigma.pdf' % saveStr)
        plt.close(fig)

        # (B) mu vs. chain #
        fig = plt.figure(figsize=(18,12))

        for plotInd, i in enumerate(range(10,15)):
            ax = fig.add_subplot(5,2,plotInd+1)
            ax.set_xlabel('chain step')
            ax.set_ylabel('$\mu_{\\rm blue}$ T[%d]' % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')
        for plotInd, i in enumerate(range(15,20)):
            ax = fig.add_subplot(5,2,plotInd+1+5)
            ax.set_ylabel('$\mu_{\\rm red}$ T[%d]' % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')

        fig.tight_layout()
        fig.savefig('debug_methodC_%s_mu.pdf' % saveStr)
        plt.close(fig)

        # (C) A blue
        for iterNum in [0,1]:
            fig = plt.figure(figsize=(18,12))

            for plotInd, i in enumerate(range(20+10*iterNum,30+10*iterNum)):
                ax = fig.add_subplot(5,2,plotInd+1)
                ax.set_xlabel('chain step')
                ax.set_ylabel('A$_{\\rm blue}$ T[%d]' % i)
                for walkerInd in range(nWalkers):
                    ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')

            fig.tight_layout()
            fig.savefig('debug_methodC_%s_Aset%d_blue.pdf' % (saveStr,iterNum))
            plt.close(fig)

        # (D) A red
        if not relAmp:
            for iterNum in [0,1]:
                fig = plt.figure(figsize=(18,12))

                for plotInd, i in enumerate(range(40+10*iterNum,50+10*iterNum)):
                    ax = fig.add_subplot(5,2,plotInd+1)
                    ax.set_xlabel('chain step')
                    ax.set_ylabel('A$_{\\rm red}$ T[%d]' % i)
                    for walkerInd in range(nWalkers):
                        ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')

                fig.tight_layout()
                fig.savefig('debug_methodC_%s_Aset%d_red.pdf' % (saveStr,iterNum))
                plt.close(fig)

        # (E) corners
        fig = corner.corner(samples[:,0:5])
        fig.savefig('debug_methodC_%s_corner_sigma_blue.pdf' % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:,5:10])
        fig.savefig('debug_methodC_%s_corner_sigma_red.pdf' % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:,10:15])
        fig.savefig('debug_methodC_%s_corner_mu_blue.pdf' % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:,15:20])
        fig.savefig('debug_methodC_%s_corner_mu_red.pdf' % saveStr)
        plt.close(fig)

    return p, p_errors, best_params

def characterizeColorMassPlane(sP, bands=['g','r'], cenSatSelect='all', simColorsModel=defSimColorModel, 
    remakeFlag=True):
    """ Do double gaussian and other methods to characterize the red and blue populations, e.g. their 
    location, extent, relative numbers, for sP at sP.snap, and save the results. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    # use previous (in redshift) MCMC result as starting point for z>0 method C calculations?
    startMCMCAtPreviousResult = True
    startMCMC_snaps = [99,91,84,78,72,67,59,50] # what snapshot sequence?

    # let us also achieve all the analysis necessary for figures 12-15 here at the same time
    mag_range = bandMagRange(bands, tight=False)
    xMinMax = [9.0, 12.0]

    binSizeMass = 0.15
    binSizeColor = 0.04 #0.05=20 0.04=25 0.03125=32 0.025=40

    skipNBinsBlue = 9 #9# skip N most-massive mass bins when fitting blue population (methods A,B only)
    skipNBinsRed = 4 # skip N least-massive mass bins when fitting red population (methods A,B only)
    assert skipNBinsBlue >= 1 # otherwise logic failure below

    paramInds = {'A_blue':0, 'mu_blue':1, 'sigma_blue':2,
                 'A_red':3,  'mu_red':4,  'sigma_red':5}

    paramIndsRel = {'mu_blue':0, 'sigma_blue':1, 'mu_red':2, 'sigma_red':3, 'A_rel':4}

    # derived
    nBinsMass = int(np.ceil((xMinMax[1]-xMinMax[0])/binSizeMass))
    nBinsColor = int((mag_range[1]-mag_range[0]) / binSizeColor)

    conf = locals() # store configuration variables into a dict for passing

    # check existence
    r = {}

    startMCMC_fromSnap = None

    if sP is not None:
        # sim
        pStr = ''
        if startMCMCAtPreviousResult:
            assert sP.snap in startMCMC_snaps
            startMCMC_ind = startMCMC_snaps.index(sP.snap) - 1
            if startMCMC_ind >= 0:
                # set snapshot we will load previous modelC final chain state from
                startMCMC_fromSnap = startMCMC_snaps[startMCMC_ind]
                pStr = '_chainf=%d' % startMCMC_snaps[startMCMC_ind]

        savePath = sP.derivPath + "/galMstarColor/"
        saveFilename = savePath + "colorMassPlaneFits_%s_%d_%s_%s%s.hdf5" % \
          (''.join(bands),sP.snap,cenSatSelect,simColorsModel,pStr)
    else:
        # obs
        assert cenSatSelect == 'all'
        assert simColorsModel == defSimColorModel

        savePath = expanduser("~") + "/obs/"
        saveFilename = savePath + "sdss_colorMassPlaneFits_%s_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),xMinMax[0]*10,xMinMax[1]*10,mag_range[0]*10,mag_range[1]*10)

    if not remakeFlag and isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        for k in conf: r[k] = conf[k]
        return r

    print('Note: characterizeColorMassPlane() load currently disabled, always remaking.')

    if sP is not None:
        # load colors
        if 'gc_colors' in sP.data and 'mstar2_log' in sP.data:
            print('sP load')
            gc_colors, mstar2_log = sP.data['gc_colors'], sP.data['mstar2_log']
            assert sP.data['cenSatSelect'] == cenSatSelect
        else:
            gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands, projs='random')
            #gc_colors = np.reshape( gc_colors, gc_colors.shape[0]*gc_colors.shape[1] )

            # load stellar masses (<2rhalf definition)
            gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
            mstar2_log = sP.units.codeMassToLogMsun( gc['subhalos'][:,sP.ptNum('stars')] )

            # cen/sat selection
            wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
            gc_colors = gc_colors[wSelect] #[wSelect,:]
            mstar2_log = mstar2_log[wSelect]

            # store in sP for temporary testing
            sP.data['gc_colors'], sP.data['mstar2_log'], sP.data['cenSatSelect'] = \
              gc_colors, mstar2_log, cenSatSelect
    else:
        # load observational points, restrict colors to mag_range as done for sims (for correct norm.)
        sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=True, kCorrect=True)

        w = np.where( (sdss_color >= mag_range[0]) & (sdss_color <= mag_range[1]) )
        gc_colors = sdss_color[w]
        mstar2_log = sdss_Mstar[w]

    # (A) double gaussian fits in 0.1 dex mstar bins, unconstrained (unrelated)
    # Levenberg-Marquadrt non-linear least squares minimization method
    for relAmp in [True]: #[True,False]:
        print('relAmp: ',relAmp,' sP.snap: ',sP.snap,' sP.redshift:',sP.redshift)
        relAmpStr = 'rel' if relAmp else ''
        pInds = paramIndsRel if relAmp else paramInds

        r['A%s_params' % relAmpStr], r['A%s_schechter' % relAmpStr], r['mStar'], \
        r['mStarCounts'], r['mHists'], r['mColorBins'] = \
          _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, relAmp=relAmp, **conf) 

        for i in range(nBinsMass):
            params = r['A%s_params' % relAmpStr][:,i]
            print(' A %2d'%i,' '.join(['%6.3f'%p for p in params]))

        # (B) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
        # T() function as in Baldry+ 2003, iterative fit (LM-LSF for each step)
        Tparams_prev = None

        # (B1) choose estimates for initial T() function parameters (only use for iterNum == 0)
        params_guess = [0.05, 0.1, 0.1, 10.0, 1.0]

        for iterNum in range(15):
            Tparams = {}

            # (B2) fit double gaussians, all mass bins
            p, _, m, _, _, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, relAmp=relAmp, **conf)

            # (B3) fit T(sigma), ignoring most-massive N bins for blue, least-massive N bin for red
            x_data = m[:-skipNBinsBlue]
            y_data = p[pInds['sigma_blue'],:-skipNBinsBlue]
            params_init = params_guess if Tparams_prev is None else Tparams_prev['sigma_blue']
            Tparams['sigma_blue'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

            x_data = m[skipNBinsRed:]
            y_data = p[pInds['sigma_red'],skipNBinsRed:]
            params_init = params_guess if Tparams_prev is None else Tparams_prev['sigma_red']
            Tparams['sigma_red'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

            # (B4) re-fit double gaussians with fixed sigma
            fixed = {}
            for key in Tparams:
                fixed[key] = _T(m, Tparams[key])

            p, _, m, _, _, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, fixed=fixed, relAmp=relAmp, **conf)

            # (B5) fit T(mu), ignoring same bins as for T(sigma)
            x_data = m[:-skipNBinsBlue]
            y_data = p[pInds['mu_blue'],:-skipNBinsBlue]
            params_init = params_guess if Tparams_prev is None else Tparams_prev['mu_blue']
            Tparams['mu_blue'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

            x_data = m[skipNBinsRed:]
            y_data = p[pInds['mu_red'],skipNBinsRed:]
            params_init = params_guess if Tparams_prev is None else Tparams_prev['mu_red']
            Tparams['mu_red'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

            # (B6) calculate change of both sets of T() parameters versus previous iteration
            if Tparams_prev is not None:
                diff_sum = 0.0

                for key in Tparams:
                    diff_local = (Tparams[key]-Tparams_prev[key])**2.0
                    diff_local = np.sum(np.sqrt( diff_local ))
                    diff_sum += diff_local
                    print(' [iter %2d]'%iterNum,key,diff_local,diff_sum)

                if diff_sum < 1e-4:
                    break

            Tparams_prev = Tparams

        assert diff_sum < 0.01 # otherwise we failed to converge

        # (B7) re-fit double gaussians with fixed sigma and mu from final T() functions
        fixed = {}
        for key in Tparams:
            fixed[key] = _T(m, Tparams[key])

        w = np.where( fixed['mu_blue'] >= fixed['mu_red'] )
        fixed['mu_blue'][w] = fixed['mu_red'][w] - 0.01 # enforce mu_blue<mu_red

        r['B%s_params' % relAmpStr], r['B%s_schechter' % relAmpStr], _, _, _, _ = \
          _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, fixed=fixed, relAmp=relAmp, **conf)

        for i in range(nBinsMass):
            params = r['B%s_params' % relAmpStr][:,i]
            print(' B %2d'%i,' '.join(['%6.3f'%param for param in params]))

        # (C) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
        # T() function as in Baldry+ 2003, use the previous result as the starting point for a full 
        # simultaneous MCMC fit of all [40/60 of] the parameters
        if startMCMC_fromSnap is None:
            # make sure p0 guess for MCMC is reasonable (within priors)
            Tparams['mu_blue'], _ = leastsq_fit(_T, Tparams['mu_blue'], args=(m,p[pInds['mu_blue'],:]))
            ###Tparams['sigma_blue'], _ = leastsq_fit(_T, Tparams['sigma_blue'], args=(m,p[pInds['sigma_blue']]))

            for key in Tparams:
                param_vals = _T(m,Tparams[key])
                if param_vals.max() > 1.0:
                    print(key,' '.join( ['%.3f'%t for t in Tparams[key]]))
                    Tparams[key][2] = 0.0
                    print(key,' '.join( ['%.3f'%t for t in Tparams[key]]))
                    param_vals = _T(m,Tparams[key])
                if param_vals.max() > 1.0:
                    Tparams[key][0] = 1.0
                    print(key,' '.join( ['%.3f'%t for t in Tparams[key]]))
                    param_vals = _T(m,Tparams[key])
                assert param_vals.min() > 0.0 and param_vals.max() < 1.0

            val_mu_blue = _T(m, Tparams['mu_blue'])
            val_mu_red = _T(m, Tparams['mu_red'])
            assert (val_mu_blue > val_mu_red).sum() == 0

            chain_start = Tparams # contains: mu_blue, sigma_blue, mu_red, sigma_red

            if relAmp:
                chain_start['A_rel'] = r['B%s_params'%relAmpStr][4,:]
            else:
                chain_start['A_blue'] = r['B%s_params'%relAmpStr][0,:]
                chain_start['A_red'] = r['B%s_params'%relAmpStr][3,:]
        else:
            # load previous results from startMCMC_fromSnap
            print('loading chain final state from snap [%d] for C' % startMCMC_fromSnap)
            curSnap = sP.snap
            sP.setSnap(startMCMC_fromSnap)
            fits_prev = characterizeColorMassPlane(sP, bands=bands, cenSatSelect=cenSatSelect, 
                                                  simColorsModel=simColorsModel, remakeFlag=False)
            sP.setSnap(curSnap)

            # reconstruct chain_start dictionary
            chain_start = {}
            chain_start['sigma_blue'] = fits_prev['C%s_fstate'%relAmpStr][00:05]
            chain_start['sigma_red'] = fits_prev['C%s_fstate'%relAmpStr][05:10]
            chain_start['mu_blue'] = fits_prev['C%s_fstate'%relAmpStr][10:15]
            chain_start['mu_red'] = fits_prev['C%s_fstate'%relAmpStr][15:20]

            if relAmp:
                chain_start['A_rel'] = fits_prev['C%s_fstate'%relAmpStr][20:40]
            else:
                chain_start['A_blue'] = fits_prev['C%s_fstate'%relAmpStr][20:40]
                chain_start['A_red'] = fits_prev['C%s_fstate'%relAmpStr][40:60]

        # run mcmc fit
        r['C%s_params' % relAmpStr], r['C%s_errors' % relAmpStr], r['C%s_fstate' % relAmpStr] = \
          _fitCMPlaneMCMC(mstar2_log, gc_colors, chain_start, relAmp=relAmp, sP_snap=sP.snap, **conf)

        for i in range(nBinsMass):
            params = r['C%s_params' % relAmpStr][:,i]
            print(' C %2d'%i,' '.join(['%6.3f'%p for p in params]))

        # (D) simultaneous MCMC fit, where we additionally require that the amplitudes of each of the 
        # red and blue follow a double-schechter function in log(Mstar), in which case there is only 
        # one global 'relative fraction' instead of one A_rel per mass bin
        # todo

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    for k in conf: r[k] = conf[k]
    return r
