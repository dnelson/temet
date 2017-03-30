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
from util.helper import leastsq_fit
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

def _double_gaussian(x, params, fixed):
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

def _schechter_function(x, params, fixed=None):
    """ Schecter phi function (x=log Mstar) for params=[phinorm,alpha,M']. Note: fixed unused. """
    (phi_norm, alpha, M_characteristic) = params
    x = x.astype('float64')

    y = phi_norm * (x/M_characteristic)**(-alpha) * np.exp( -x/M_characteristic )
    return y

def _fitCMPlaneDoubleGaussian(masses, colors, xMinMax, mag_range, binSizeMass, binSizeColor, 
                              paramInds, nParams, nBinsMass, nBinsColor, fixed=None, **kwargs):
    """ Return the parameters of a full fit to objects in the (color-mass) plane. A double gaussian 
    is fit to each mass bin, histogramming in color, both bin sizes fixed herein. The centers and 
    widths of the Gaussians may be optionally constrained as inputs with fixed. """
    assert nParams == 6 # fixed

    # initial guess for (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
    params_guess = [1.0, 0.25, 0.5, 1.0, 0.75, 0.5]

    # allocate
    p = np.zeros( (nParams,nBinsMass), dtype='float32' )
    m = np.zeros( nBinsMass, dtype='float32' )
    n = np.zeros( nBinsMass, dtype='int32' )

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i+1)

        wMassBin = np.where( (masses > minMass) & (masses <= maxMass) )

        if len(wMassBin[0]) == 0:
            print('Warning: empty mass bin.')

        if colors.ndim == 2:
            colors_data = np.ravel( colors[wMassBin,:] ) # flatten into 1D
        else:
            colors_data = colors[wMassBin] # obs/single projection

        # have to histogram (or 1D-KDE) them, to get a (x_data,y_data) point set
        y_data, x_data = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        x_data = x_data[:-1] + binSizeColor/2.0

        # any fixed (non-varying) parameters?
        fixed_loc = None

        if fixed is not None:
            # nParams for this mass bin, start with none fixed
            fixed_loc = [None for _ in range(nParams)]

            # which field(s) are fixed? pull out value at this mass bin
            for paramName in paramInds.keys():
                if paramName in fixed:
                    assert fixed[paramName].size == nBinsMass and fixed[paramName].ndim == 1
                    fixed_loc[paramInds[paramName]] = fixed[paramName][i]

        # run fit
        params_best, _ = leastsq_fit(_double_gaussian, params_guess, args=(x_data,y_data,fixed_loc))

        # sigma_i can fit negative since this is symmetric in the fit function...
        for pName in ['sigma_blue','sigma_red']:
            params_best[ paramInds[pName] ] = np.abs(params_best[ paramInds[pName] ])

        # blue/red choice for each gaussian is arbitrary, enforce that red = the one with redder center
        if params_best[ paramInds['mu_blue'] ] > params_best[ paramInds['mu_red'] ]:
            params_best = np.roll(params_best,3) # swap first 3 and last 3

        # re-normalize noisy amplitudes such that total integral is 1
        int_blue = params_best[paramInds['A_blue']] * params_best[paramInds['sigma_blue']] * np.sqrt(2*np.pi)
        int_red = params_best[paramInds['A_red']] * params_best[paramInds['sigma_red']] * np.sqrt(2*np.pi)
        int_fac = 1.0 / (int_blue + int_red)
        params_best[paramInds['A_blue']] *= int_fac
        params_best[paramInds['A_red']] *= int_fac

        # save
        m[i] = minMass + binSizeMass/2.0
        n[i] = colors_data.size
        p[:,i] = params_best

    # do a schechter function fit to (mass,counts) for red and blue separately
    params_guess = [1e4, 1.0, 10.0]
    p_sch = np.zeros(6, dtype='float32')

    for i, c in enumerate(['red','blue']):
        # number of either red or blue galaxies in this mass bin:
        y_data = p[paramInds['A_'+c],:] * p[paramInds['sigma_'+c],:] * n * np.sqrt(2*np.pi)
        p_sch[i*3:(i+1)*3], _ = leastsq_fit(_schechter_function, params_guess, args=(m,y_data))

    return p, p_sch, m, n

def _fitCMPlaneMCMC(masses, colors, Tparams, B_params, 
                    xMinMax, mag_range, binSizeMass, binSizeColor, nBinsMass, nBinsColor, **kwargs):
    """ Testing MCMC based fit in the color-mass plane of the full double gaussian model. """
    import emcee

    # config
    nWalkers = 200
    nBurnIn = 100
    nProdSteps = 50
    fracNoiseInit = 1e-4
    percentiles = [16,50,84] # middle is used to derive the best-fit for each parameter (e.g. median)

    # global MCMC fit to all the parameters simultaneously, five for each T() function, per color, 
    # plus 2 amplitudes per mass bin, plus a noise per mass bin (=20x2 + 10*2=60 + 20!)
    nDim = 2*nBinsMass + 2*5*2 #+ nBinsMass # 60
    p0 = np.zeros( nDim, dtype='float32' )
    p0[00:05] = Tparams['sigma_blue']
    p0[05:10] = Tparams['sigma_red']
    p0[10:15] = Tparams['mu_blue']
    p0[15:20] = Tparams['mu_red']
    p0[20:40] = B_params[0,:] # blue A, 20 of them
    p0[40:60] = B_params[3,:] # red A, 20 of them
    #p0[60:80] = 0.01 # noise nuisance parameters, one per mass bin

    def mcmc_lnprob_binned(theta, x, y, m):
        # run four T() functions, get sigma_r,b(Mass) and mu_r,b(Mass)
        # compute all 20 double gaussians, in each mass bin have its 6 parameters
        lp = 0.0
        nMassBins = len(x)
        lnlike_y = np.zeros( nMassBins, dtype='float32' )

        # reconstruct sigma and mu values at each mass bin from the T() functions
        sigma1 = _T(m, theta[0:5])
        sigma2 = _T(m, theta[5:10])
        mu1 = _T(m, theta[10:15])
        mu2 = _T(m, theta[15:20])

        # prior, 0.0 (no change to likelihood) in general, unless we violate a prior we wish to 
        # impose, then -np.inf, such that the point is rejected absolutely
        if (mu1 > mu2).sum() > 0:
            # if any blue center is larger than the red center, reject this whole theta
            #lp = -np.inf
            return -np.inf # this prior is absolute so just return early

        # compute independent loglikelihood in each mass bin
        for i in range(nMassBins):
            # pull out the remaining 2 parameters (A_blue, A_red) for this mass bin
            A1, A2 = theta[20+i], theta[40+i]
            y_err = 0.05 #theta[60+i]

            # histogramed data
            params_double_gaussian = (A1, mu1[i], sigma1[i], A2, mu2[i], sigma2[i])
            inv_sigma2 = 1.0/y_err**2.0

            y_fit = _double_gaussian(x[i], params_double_gaussian, fixed=None)
            chi2 = np.sum( (y_fit - y)**2.0 * inv_sigma2 - np.log(inv_sigma2) )
            lnlike_y[i] = -0.5 * chi2 # like_y[i] = np.exp(-chi2)

        # all mass bin likelihoods multiplied -> added in the log
        return lp + np.sum(lnlike_y)

    def mcmc_lnprob_nobin(theta, x, m):
        # run four T() functions, get sigma_r,b(Mass) and mu_r,b(Mass)
        # compute all 20 double gaussians, in each mass bin have its 6 parameters
        lp = 0.0
        nMassBins = len(x)
        lnlike_y = np.zeros( nMassBins, dtype='float32' )

        # reconstruct sigma and mu values at each mass bin from the T() functions
        sigma1 = _T(m, theta[0:5])
        sigma2 = _T(m, theta[5:10])
        mu1 = _T(m, theta[10:15])
        mu2 = _T(m, theta[15:20])

        if (mu1 > mu2).sum() > 0:
            return -np.inf # this prior is absolute so just return early

        # compute independent loglikelihood in each mass bin
        for i in range(nMassBins):
            # pull out the remaining parameter for this mass bin
            rel_amp_fac = 1.0 #theta[20+i]

            # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
            # non-histogrammed, multiplying the likelihoods of the two gaussians
            n = x[i].size

            lnlike_y1 = -rel_amp_fac*0.5*n*np.log(2*np.pi*sigma1[i]**2) - \
              1.0/(2*sigma1[i]**2) * np.sum( (x[i] - mu1[i])** 2 )
            lnlike_y2 = -0.5*n*np.log(2*np.pi*sigma2[i]**2) - \
              1.0/(2*sigma2[i]**2) * np.sum( (x[i] - mu2[i])** 2 )
            lnlike_y[i] = lnlike_y1 + lnlike_y2

            import pdb; pdb.set_trace()

        # all mass bin likelihoods multiplied -> added in the log
        return lp + np.sum(lnlike_y)

    # make x_data
    m = np.zeros(nBinsMass)
    x_data = []
    x_data2 = []
    y_data2 = []

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i+1)

        wMassBin = np.where( (masses > minMass) & (masses <= maxMass) )

        if colors.ndim == 2:
            colors_data = np.ravel( colors[wMassBin,:] ) # flatten into 1D
        else:
            colors_data = colors[wMassBin] # obs/single projection

        x_data.append( colors_data )
        m[i] = minMass + binSizeMass/2.0

        # (C) histogram
        yy, xx = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        xx = xx[:-1] + binSizeColor/2.0

        x_data2.append(xx)
        y_data2.append(yy)

    # bin method:
    if 1:
        # setup initial parameter guesses (theta0) for all walkers
        p0_walkers = np.zeros( (nWalkers,nDim), dtype='float32' )
        np.random.seed(42424242L)
        for i in range(nWalkers):
            p0_walkers[i,:] = p0 + np.abs(p0) * np.random.normal(loc=0.0, scale=fracNoiseInit)

        # setup sampler and run a burn-in
        tstart = time.time()
        sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_binned, args=(x_data2, y_data2, m))

        pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
        sampler.reset()

        # run production chain
        sampler.run_mcmc(pos, nProdSteps)

        # ideally between 0.2 and 0.5:
        mean_acc = np.mean(sampler.acceptance_fraction)

        print('done sampling in [%.1f sec] mean acceptance frac: %.2f' % (time.time() - tstart,mean_acc))

        # calculate medians of production chains as answer
        samples = sampler.chain.reshape( (-1,nDim) )

        # print median as the answer, as well as standard percentiles
        percs = np.percentile(samples, percentiles, axis=0)
        #for i in range(nDim):
        #    print(i, percs[:,i], samples[:,i].min(), samples[:,i].max())

        best_params = percs[1,:]
        assert best_params.size == nDim

        # create return parameters, reconstructing the mu_i and sigma_i from the T() parameters
        p = np.zeros( (6,nBinsMass), dtype='float32' ) # (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        
        p[0,:] = best_params[20:40]
        p[3,:] = best_params[40:60]
        p[1,:] = _T(m, best_params[10:15])
        p[4,:] = _T(m, best_params[15:20])
        p[2,:] = _T(m, best_params[0:5])
        p[5,:] = _T(m, best_params[5:10])

    # non-binned method:
    if 1:
        print('method on x_raw:')
        # setup initial parameter guesses (theta0) for all walkers
        nDim = 20 # only sigma and mu T params for red/blue (5*4)
        p0_walkers = np.zeros( (nWalkers,nDim), dtype='float32' )
        np.random.seed(42424242L)
        for i in range(nWalkers):
            p0_walkers[i,:] = p0[0:nDim] + np.abs(p0[0:nDim]) * np.random.normal(loc=0.0, scale=fracNoiseInit)

        import pdb; pdb.set_trace()

        # setup sampler and run a burn-in
        tstart = time.time()
        sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_nobin, args=(x_data, m))

        pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
        sampler.reset()

        # run production chain
        sampler.run_mcmc(pos, nProdSteps)

        # ideally between 0.2 and 0.5:
        mean_acc = np.mean(sampler.acceptance_fraction)

        print('done sampling in [%.1f sec] mean acceptance frac: %.2f' % (time.time() - tstart,mean_acc))

        # calculate medians of production chains as answer
        samples = sampler.chain.reshape( (-1,nDim) )

        # print median as the answer, as well as standard percentiles
        percs = np.percentile(samples, percentiles, axis=0)
        for i in range(nDim):
            print(i, percs[:,i], samples[:,i].min(), samples[:,i].max())

        best_params = percs[1,:]
        assert best_params.size == nDim

        # create return parameters, reconstructing the mu_i and sigma_i from the T() parameters
        p2 = np.zeros( (4,nBinsMass), dtype='float32' ) # (mu1, sigma1, mu2, sigma2) (1=blue, 2=red)
        
        p2[0,:] = _T(m, best_params[10:15])
        p2[2,:] = _T(m, best_params[15:20])
        p2[1,:] = _T(m, best_params[0:5])
        p2[3,:] = _T(m, best_params[5:10])

    # debug plots
    if 0:
        import matplotlib.pyplot as plt
        import corner

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
        fig.savefig('debug_methodC_sigma.pdf')
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
        fig.savefig('debug_methodC_mu.pdf')
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
            fig.savefig('debug_methodC_Aset%d_blue.pdf' % iterNum)
            plt.close(fig)

        # (D) A red
        for iterNum in [0,1]:
            fig = plt.figure(figsize=(18,12))

            for plotInd, i in enumerate(range(40+10*iterNum,50+10*iterNum)):
                ax = fig.add_subplot(5,2,plotInd+1)
                ax.set_xlabel('chain step')
                ax.set_ylabel('A$_{\\rm red}$ T[%d]' % i)
                for walkerInd in range(nWalkers):
                    ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd,:,i],lw=0.8,alpha=0.5,color='black')

            fig.tight_layout()
            fig.savefig('debug_methodC_Aset%d_red.pdf' % iterNum)
            plt.close(fig)

        # (E) corners
        fig = corner.corner(samples[:,0:5])
        fig.savefig('debug_methodC_corner_sigma_blue.pdf')
        plt.close(fig)

        fig = corner.corner(samples[:,10:15])
        fig.savefig('debug_methodC_corner_mu_blue.pdf')
        plt.close(fig)

    return p, p2

def characterizeColorMassPlane(sP, bands=['g','r'], cenSatSelect='all', simColorsModel=defSimColorModel):
    """ Do double gaussian and other methods to characterize the red and blue populations, e.g. their 
    location, extent, relative numbers, for sP at sP.snap, and save the results. """
    assert cenSatSelect in ['all', 'cen', 'sat']

    # let us also achieve all the analysis necessary for figures 12-15 here at the same time
    mag_range = bandMagRange(bands, tight=False)
    xMinMax = [9.0, 12.0]

    binSizeMass = 0.15
    binSizeColor = 0.05

    paramInds = {'A_blue':0, 'mu_blue':1, 'sigma_blue':2,
                 'A_red':3,  'mu_red':4,  'sigma_red':5}

    # derived
    nBinsMass = int(np.ceil((xMinMax[1]-xMinMax[0])/binSizeMass))
    nBinsColor = int((mag_range[1]-mag_range[0]) / binSizeColor)
    nParams = len(paramInds.keys())

    conf = locals() # store configuration variables into a dict for passing

    # check existence
    r = {}

    if sP is not None:
        # sim
        savePath = sP.derivPath + "/galMstarColor/"
        saveFilename = savePath + "colorMassPlaneFits_%s_%d_%s_%s.hdf5" % \
          (''.join(bands),sP.snap,cenSatSelect,simColorsModel)
    else:
        # obs
        assert cenSatSelect == 'all'
        assert simColorsModel == defSimColorModel

        savePath = expanduser("~") + "/obs/"
        saveFilename = savePath + "sdss_colorMassPlaneFits_%s_%d-%d_%d-%d.hdf5" % \
          (''.join(bands),xMinMax[0]*10,xMinMax[1]*10,mag_range[0]*10,mag_range[1]*10)

    print('Note: characterizeColorMassPlane() load currently disabled, always remaking.')
    if 0 and isfile(saveFilename):
    #if isfile(saveFilename):
        with h5py.File(saveFilename,'r') as f:
            for key in f:
                r[key] = f[key][()]
        r['paramInds'] = paramInds
        for k in conf: r[k] = conf[k]
        return r

    if sP is not None:
        # load colors
        if 'gc_colors' in sP.data and 'mstar2_log' in sP.data:
            print('sP load')
            gc_colors, mstar2_log = sP.data['gc_colors'], sP.data['mstar2_log']
            assert sP.data['cenSatSelect'] == cenSatSelect
        else:
            gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands, projs='all')
            #gc_colors = np.reshape( gc_colors, gc_colors.shape[0]*gc_colors.shape[1] )

            # load stellar masses (<2rhalf definition)
            gc = groupCat(sP, fieldsSubhalos=['SubhaloMassInRadType'])
            mstar2_log = sP.units.codeMassToLogMsun( gc['subhalos'][:,sP.ptNum('stars')] )

            # cen/sat selection
            wSelect = cenSatSubhaloIndices(sP, cenSatSelect=cenSatSelect)
            gc_colors = gc_colors[wSelect,:]
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
    r['A_params'], r['A_schechter'], r['mStar'], r['mStarCounts'] = \
      _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, **conf) 

    # (B) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
    # T() function as in Baldry+ 2003, iterative fit (LM-LSF for each step)
    Tparams_prev = None

    # (B1) choose estimates for initial T() function parameters (only use for iterNum == 0)
    params_guess = [0.05, 0.1, 0.1, 10.0, 1.0]

    for iterNum in range(15):
        Tparams = {}

        # (B2) fit double gaussians, all mass bins
        p, _, m, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, **conf)

        # (B3) fit T(sigma), ignoring most-massive 3 bins for blue, least-massive 1 bin for red
        x_data = m[:-3]
        y_data = p[paramInds['sigma_blue'],:-3]
        params_init = params_guess if Tparams_prev is None else Tparams_prev['sigma_blue']
        Tparams['sigma_blue'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

        x_data = m[1:]
        y_data = p[paramInds['sigma_red'],1:]
        params_init = params_guess if Tparams_prev is None else Tparams_prev['sigma_red']
        Tparams['sigma_red'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

        # (B4) re-fit double gaussians with fixed sigma
        fixed = {}
        for key in Tparams:
            fixed[key] = _T(m, Tparams[key])

        p, _, m, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, fixed=fixed, **conf)

        # (B5) fit T(mu), ignoring same bins as for T(sigma)
        x_data = m[:-3]
        y_data = p[paramInds['mu_blue'],:-3]
        params_init = params_guess if Tparams_prev is None else Tparams_prev['mu_blue']
        Tparams['mu_blue'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

        x_data = m[1:]
        y_data = p[paramInds['mu_red'],1:]
        params_init = params_guess if Tparams_prev is None else Tparams_prev['mu_red']
        Tparams['mu_red'], _ = leastsq_fit(_T, params_init, args=(x_data,y_data))

        # (B6) calculate change of both sets of T() parameters versus previous iteration
        if Tparams_prev is not None:
            diff_sum = 0.0

            for key in Tparams:
                diff_local = (Tparams[key]-Tparams_prev[key])**2.0
                diff_local = np.sum(np.sqrt( diff_local ))
                diff_sum += diff_local
                print(iterNum,key,diff_local,diff_sum)

            if diff_sum < 0.01:#1e-10:
                break

        Tparams_prev = Tparams

    assert diff_sum < 0.01 # otherwise we failed to converge

    # (B7) re-fit double gaussians with fixed sigma and mu from final T() functions
    fixed = {}
    for key in Tparams:
        fixed[key] = _T(m, Tparams[key])

    r['B_params'], r['B_schechter'], _, _ = \
      _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, fixed=fixed, **conf)

    # (C) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the 
    # T() function as in Baldry+ 2003, use the previous result as the starting point for a full 
    # simultaneous MCMC fit of all [60 of] the parameters
    r['C_params'], r['D_params'] = _fitCMPlaneMCMC(mstar2_log, gc_colors, Tparams, r['B_params'], **conf)

    # save
    with h5py.File(saveFilename,'w') as f:
        for key in r:
            f[key] = r[key]
    print('Saved: [%s]' % saveFilename.split(savePath)[1])

    r['paramInds'] = paramInds
    for k in conf: r[k] = conf[k]

    return r
