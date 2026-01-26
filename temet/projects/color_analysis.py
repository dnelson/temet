"""
TNG galaxy color paper: analysis.

https://arxiv.org/abs/1707.03395
"""

import time
import warnings
from glob import glob
from os.path import expanduser, isfile

import h5py
import numpy as np

from temet.cosmo.color import calcSDSSColors, loadSimGalColors
from temet.plot.quantities import bandMagRange
from temet.util import simParams
from temet.util.helper import least_squares_fit, leastsq_fit


# the dust model used by default for all colors
defSimColorModel = "p07c_cf00dust_res_conv_ns1_rad30pkpc"


def calcColorEvoTracks(sP, bands=("g", "r"), simColorsModel=defSimColorModel):
    """Calculate the evolution of galaxy colors in time using their MPBs.

    Using already computed StellarPhot auxCat's at several snapshots, load the MPBs and
    re-organize the band magnitudes into tracks in time for each galaxy. Do for one band
    combination, saving only the color, while keeping all viewing angles.
    """
    savePath = sP.derivPath + "/auxCat/"

    # how many computed StellarPhot auxCats already exist?
    savedPaths = glob(savePath + "Subhalo_StellarPhot_%s_???.hdf5" % simColorsModel)
    savedSnaps = sorted([int(path[-8:-5]) for path in savedPaths], reverse=True)
    numSnaps = len(savedSnaps)

    # check existence
    saveFilename = savePath + "Subhalo_StellarPhotEvo_%s_%d-%d_%s.hdf5" % (
        "".join(bands),
        sP.snap,
        numSnaps,
        simColorsModel,
    )

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            colorEvo = f["colorEvo"][()]
            shIDEvo = f["shIDEvo"][()]
            subhaloIDs = f["subhaloIDs"][()]
            savedSnaps = f["savedSnaps"][()]

        return colorEvo, shIDEvo, subhaloIDs, savedSnaps

    # compute new:
    print("Found [%d] saved StellarPhot at snaps: %s." % (numSnaps, ", ".join([str(s) for s in savedSnaps])))

    # load z=0 colors and identify subset of SubhaloIDs we will track
    colors0, subhaloIDs = loadSimGalColors(sP, simColorsModel, bands=bands, projs="all")
    gcH = sP.groupCatHeader()

    if colors0.ndim > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            colors0_mean = np.nanmean(colors0, axis=1)
    else:
        colors0_mean = colors0.copy()

    assert colors0.shape[0] == gcH["Nsubgroups_Total"] == subhaloIDs.size
    assert np.array_equal(subhaloIDs, np.arange(subhaloIDs.size))
    assert colors0_mean.ndim == 1

    w = np.where(np.isfinite(colors0_mean))  # keep subhalos with non-NaN colors

    print(" calcColorEvoTracks: keep [%d] of [%d] subhalos." % (len(w[0]), subhaloIDs.size))
    subhaloIDs = subhaloIDs[w]

    # allocate
    numProjs = colors0.shape[1] if colors0.ndim > 1 else 1
    colorEvo = np.zeros((subhaloIDs.size, numProjs, numSnaps), dtype="float32")
    shIDEvo = np.zeros((subhaloIDs.size, numSnaps), dtype="int32")

    colorEvo.fill(np.nan)
    shIDEvo.fill(-1)

    # load MPBs of the subhalo selection
    mpbs = sP.loadMPBs(subhaloIDs, fields=["SnapNum", "SubfindID"])

    # walk backwards through snapshots where we have computed StellarPhot data
    origSnap = sP.snap

    for i, snap in enumerate(savedSnaps):
        sP.setSnap(snap)
        print(" [%d] snap = %d" % (i, snap))

        # load local colors
        colors_loc, subhaloIDs_loc = loadSimGalColors(sP, simColorsModel, bands=bands, projs="all")

        # loop over each z=0 subhalo
        for j, subhaloID in enumerate(subhaloIDs):
            # get progenitor SubfindID at this snapshot
            if subhaloID not in mpbs:
                continue  # not tracked anywhere

            snapInd = np.where(mpbs[subhaloID]["SnapNum"] == snap)[0]

            if len(snapInd) == 0:
                continue  # not tracked to this snapshot

            # map progenitor to its color at this snapshot, and save
            subhaloID_loc = mpbs[subhaloID]["SubfindID"][snapInd]
            assert subhaloIDs_loc[subhaloID_loc] == subhaloID_loc  # otherwise, need to index elsewhere
            # in colors_loc to get subhaloID_loc

            colorEvo[j, :, i] = colors_loc[subhaloID_loc]
            shIDEvo[j, i] = subhaloID_loc

    sP.setSnap(origSnap)

    # save
    with h5py.File(saveFilename, "w") as f:
        f["colorEvo"] = colorEvo
        f["shIDEvo"] = shIDEvo
        f["subhaloIDs"] = subhaloIDs
        f["savedSnaps"] = savedSnaps
    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return colorEvo, shIDEvo, subhaloIDs, savedSnaps


def _T(x, params, fixed=None):
    """T() linear-tanh function of Baldry+ (2003). Note: fixed argument unused."""
    (p0, p1, q0, q1, q2) = params
    y = p0 + p1 * x + q0 * np.tanh((x - q1) / q2)
    return y


def _double_gaussian(x, params, fixed=None):
    """Additive double gaussian function used for fitting."""
    if fixed is not None:
        assert len(fixed) == len(params)
        for i in range(len(fixed)):
            if fixed[i] is not None:
                params[i] = fixed[i]

    # pull out the 6 params of the 2 gaussians
    (A1, mu1, sigma1, A2, mu2, sigma2) = params

    y = A1 * np.exp(-((x - mu1) ** 2.0) / (2.0 * sigma1**2.0)) + A2 * np.exp(-((x - mu2) ** 2.0) / (2.0 * sigma2**2.0))
    return y


def _double_gaussian_rel(x, params, fixed=None):
    """Additive double gaussian test function, each normalized with a relative amplitude parameter."""
    if fixed is not None:
        assert len(fixed) == len(params)
        for i in range(len(fixed)):
            if fixed[i] is not None:
                params[i] = fixed[i]

    # pull out the 5 params of the 2 gaussians
    (mu1, sigma1, mu2, sigma2, Afrac) = params

    A1 = Afrac / np.sqrt(2 * np.pi) / sigma1
    A2 = (1.0 - Afrac) / np.sqrt(2 * np.pi) / sigma2
    y = A1 * np.exp(-((x - mu1) ** 2.0) / (2.0 * sigma1**2.0)) + A2 * np.exp(-((x - mu2) ** 2.0) / (2.0 * sigma2**2.0))

    return y


def _schechter_function(x, params, fixed=None):
    """Schecter phi function (x=log Mstar) for params=[phinorm,alpha,M']. Note: fixed unused."""
    (phi_norm, alpha, M_characteristic) = params
    x = x.astype("float64")

    y = phi_norm * (x / M_characteristic) ** (-alpha) * np.exp(-x / M_characteristic)
    return y


def _fitCMPlaneDoubleGaussian(
    masses,
    colors,
    xMinMax,
    mag_range,
    binSizeMass,
    binSizeColor,
    paramInds,
    paramIndsRel,
    nBinsMass,
    nBinsColor,
    relAmp=False,
    fixed=None,
    **kwargs,
):
    """Return the parameters of a full fit to objects in the (color-mass) plane.

    A double gaussian is fit to each mass bin, histogramming in color, both bin sizes fixed herein.
    The centers and widths of the Gaussians may be optionally constrained as inputs with fixed.
    """
    if relAmp:
        # only one relative amplitude parameter is fit
        fit_func = _double_gaussian_rel
        # initial guess for (mu1, sigma1, mu2, sigma2, Arel) (1=blue, 2=red)
        params_guess = [0.35, 0.1, 0.75, 0.1, 0.5]

        params_bounds = (0.0, 1.0 - 1e-6)  # none of mu,sigma,A_rel can be negative or bigger than 1

        pInds = paramIndsRel
        nParams = len(pInds)
        assert nParams == 5
    else:
        # two free amplitude parameters are fit
        fit_func = _double_gaussian
        # initial guess for (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        params_guess = [1.0, 0.35, 0.1, 1.0, 0.75, 0.1]

        params_bounds = (0.0, (np.inf, 1.0, 1.0, np.inf, 1.0, 1.0))  # none of mu,sigma <0 or >1, A_i >0 only

        pInds = paramInds
        nParams = len(pInds)
        assert nParams == 6

    # enforce reasonable values of Tparams
    if fixed is not None:
        for key in fixed:
            w = np.where((fixed[key] <= 0.0) | (fixed[key] > 1.0))
            fixed[key][w] = 0.01  # set to a value inside bounds

    # allocate
    p = np.zeros((nParams, nBinsMass), dtype="float32")  # parameters per mass bin
    m = np.zeros(nBinsMass, dtype="float32")  # mass bin centers
    y = np.zeros((nBinsMass, nBinsColor), dtype="float32")  # 2d histogram
    n = np.zeros(nBinsMass, dtype="int32")  # counts per mass bin

    p.fill(np.nan)

    for i in range(nBinsMass):
        # select in this mass bin
        minMass = xMinMax[0] + binSizeMass * i
        maxMass = xMinMax[0] + binSizeMass * (i + 1)
        m[i] = minMass + binSizeMass / 2.0

        wMassBin = np.where((masses > minMass) & (masses <= maxMass))

        if len(wMassBin[0]) == 0:
            print("Warning: empty mass bin.")

        if colors.ndim == 2:
            colors_data = np.ravel(colors[wMassBin, :])  # flatten into 1D
        else:
            colors_data = colors[wMassBin]  # obs/single projection

        # have to histogram (or 1D-KDE) them, to get a (x_data,y_data) point set
        n[i] = colors_data.size
        y_data, x_data = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        x_data = x_data[:-1] + binSizeColor / 2.0

        y[i, :] = y_data

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

        # run fit
        # params_best, _ = leastsq_fit(fit_func, params_guess, args=(x_data,y_data,fixed_loc))
        params_best = least_squares_fit(fit_func, params_guess, params_bounds, args=(x_data, y_data, fixed_loc))

        # sigma_i can fit negative since this is symmetric in the fit function...
        for pName in ["sigma_blue", "sigma_red"]:
            params_best[pInds[pName]] = np.abs(params_best[pInds[pName]])

        # blue/red choice for each gaussian is arbitrary, enforce that red = the one with redder center
        if params_best[pInds["mu_blue"]] > params_best[pInds["mu_red"]]:
            if not relAmp:
                params_best = np.roll(params_best, 3)  # swap first 3 and last 3
            else:
                # swap mu,sigma and switch relative amplitude
                params_best = params_best[[2, 3, 0, 1, 4]]
                params_best[4] = 1.0 - params_best[4]

        # re-normalize noisy amplitudes such that total integral is 1
        if not relAmp:
            int_blue = params_best[pInds["A_blue"]] * params_best[pInds["sigma_blue"]] * np.sqrt(2 * np.pi)
            int_red = params_best[pInds["A_red"]] * params_best[pInds["sigma_red"]] * np.sqrt(2 * np.pi)
            int_fac = 1.0 / (int_blue + int_red)
            params_best[pInds["A_blue"]] *= int_fac
            params_best[pInds["A_red"]] *= int_fac
        else:
            if params_best[pInds["A_rel"]] >= 1.0 - 1e-4:
                params_best[pInds["A_rel"]] = 1.0 - 1e-4  # for stability

        # save
        p[:, i] = params_best

    assert p.max() < 1.0

    # do a schechter function fit to (mass,counts) for red and blue separately
    params_guess = [1e4, 1.0, 10.0]
    p_sch = np.zeros(6, dtype="float32")

    if 0:
        # work in progress
        for i, c in enumerate(["red", "blue"]):
            # number of either red or blue galaxies in this mass bin:
            if not relAmp:
                y_amps = p[pInds["A_" + c], :]
            else:
                if c == "blue":
                    y_amps = p[pInds["A_rel"]]
                if c == "red":
                    y_amps = 1.0 - p[pInds["A_rel"]]

            N_gal = y_amps * p[pInds["sigma_" + c], :] * n * np.sqrt(2 * np.pi)

            print(n.sum(), N_gal.sum())
            assert 0  # todo: finish

            p_sch[i * 3 : (i + 1) * 3], _ = leastsq_fit(_schechter_function, params_guess, args=(m, N_gal))

    return p, p_sch, m, n, y, x_data


def _fitCMPlaneMCMC(
    masses,
    colors,
    chain_start,
    xMinMax,
    mag_range,
    skipNBinsRed,
    skipNBinsBlue,
    binSizeMass,
    binSizeColor,
    nBinsMass,
    nBinsColor,
    nWalkers,
    nBurnIn,
    nProdSteps,
    fracNoiseInit,
    percentiles,
    relAmp=False,
    sP_snap=0,
    newErrors=False,
    **kwargs,
):
    """MCMC based fit in the color-mass plane of the full double gaussian model."""
    import emcee

    # global MCMC fit to all the parameters simultaneously, five for each T() function, per color,
    # plus 1 or 2 amplitudes per mass bin (=20x2 + 10*2=60)
    ampsPerMassBin = 1 if relAmp else 2
    nDim = ampsPerMassBin * nBinsMass + 2 * 5 * 2  # 20+20 or 40+20

    p0 = np.zeros(nDim, dtype="float32")
    p0[0:5] = chain_start["sigma_blue"]
    p0[5:10] = chain_start["sigma_red"]
    p0[10:15] = chain_start["mu_blue"]
    p0[15:20] = chain_start["mu_red"]

    if relAmp:
        p0[20:40] = chain_start["A_rel"]  # A_rel, 20 of them
        fit_func = _double_gaussian_rel
    else:
        p0[20:40] = chain_start["A_blue"]  # blue A, 20 of them
        p0[40:60] = chain_start["A_red"]  # red A, 20 of them
        fit_func = _double_gaussian

    assert np.all(np.isfinite(p0))

    def mcmc_lnprob_binned(theta, x, y, m, y_is2):
        # run four T() functions, get sigma_r,b(Mass) and mu_r,b(Mass)
        # compute all 20 double gaussians, in each mass bin have its 6 parameters
        lp = 0.0
        nMassBins = len(y)
        lnlike_y = np.zeros(nMassBins, dtype="float32")

        # reconstruct sigma and mu values at each mass bin from the T() functions
        sigma1 = _T(m, theta[0:5])
        sigma2 = _T(m, theta[5:10])
        mu1 = _T(m, theta[10:15])
        mu2 = _T(m, theta[15:20])

        # 'absolute' priors, so just return early
        mu_mm = [0.0, 1.0]  # min max tophat prior
        sigma_mm = [0.0, 1.0]  # min max tophat prior

        if (mu1 > mu2).sum() > 0:
            return -np.inf
        if mu1.min() <= mu_mm[0] or mu1.max() >= mu_mm[1] or mu2.min() <= mu_mm[0] or mu2.max() >= mu_mm[1]:
            return -np.inf
        if (
            sigma1.min() <= sigma_mm[0]
            or sigma1.max() >= sigma_mm[1]
            or sigma2.min() <= sigma_mm[0]
            or sigma2.max() >= sigma_mm[1]
        ):
            return -np.inf
        if relAmp:
            if theta[20:].min() < 0.0 or theta[20:].max() > 1.0:
                return -np.inf  # A_rel in [0,1]
        else:
            if theta[20:].min() < 0.0:
                return -np.inf  # A_i >= 0

        # compute independent loglikelihood in each mass bin
        for i in range(nMassBins):  # range(skipNBinsRed,nMassBins-skipNBinsBlue):
            # note: i never enters mass bins where we leave contribution at zero

            # pull out the remaining 2 parameters (A_blue, A_red) for this mass bin
            if relAmp:
                A_rel = theta[20 + i]
                params_double_gaussian = (mu1[i], sigma1[i], mu2[i], sigma2[i], A_rel)
            else:
                A1, A2 = theta[20 + i], theta[40 + i]
                params_double_gaussian = (A1, mu1[i], sigma1[i], A2, mu2[i], sigma2[i])

            assert np.all(np.isfinite(params_double_gaussian))

            # y_err = 0.05 # soften
            # inv_sigma2 = 1.0/y_err**2.0
            inv_sigma2 = y_is2[i]

            # compare histogramed data
            y_fit = fit_func(x, params_double_gaussian, fixed=None)
            chi2 = np.sum((y_fit - y[i]) ** 2.0 * inv_sigma2 - np.log(inv_sigma2))
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
        maxMass = xMinMax[0] + binSizeMass * (i + 1)

        wMassBin = np.where((masses > minMass) & (masses <= maxMass))

        if colors.ndim == 2:
            colors_data = np.ravel(colors[wMassBin, :])  # flatten into 1D
        else:
            colors_data = colors[wMassBin]  # obs/single projection

        yy, xx = np.histogram(colors_data, range=mag_range, bins=nBinsColor, density=True)
        xx = xx[:-1] + binSizeColor / 2.0

        m[i] = minMass + binSizeMass / 2.0
        y_data.append(yy)

        # error estimate
        y_err = np.zeros(yy.size, dtype="float32")

        w = np.where(yy > 0.0)

        if newErrors:
            # err1: Poisson
            y_err[w] = np.sqrt(yy[w])
            # err2: constant
            # y_err += 0.1
        else:
            # previous method
            y_err[w] = 1.0 / np.sqrt(yy[w] * binSizeColor * yy.sum())

        w2 = np.where(y_err == 0.0)
        y_err[w2] = y_err[w].max() * 10.0

        inv_sigma2 = 1.0 / y_err**2.0
        y_is2.append(inv_sigma2)

    # binned method: setup initial parameter guesses (theta0) for all walkers
    p0_walkers = np.zeros((nWalkers, nDim), dtype="float32")
    np.random.seed(42424242)
    for i in range(nWalkers):
        p0_walkers[i, :] = p0 + np.abs(p0) * np.random.normal(loc=0.0, scale=fracNoiseInit, size=nDim)

    if relAmp:
        p0_walkers[:, 20:40] = np.clip(p0_walkers[:, 20:40], 0.0, 1.0 - 1e-3)

    # setup sampler and run a burn-in
    tstart = time.time()
    sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_binned, args=(xx, y_data, m, y_is2))

    pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
    sampler.reset()

    # run production chain
    sampler.run_mcmc(pos, nProdSteps)

    # ideally between 0.2 and 0.5:
    mean_acc = np.mean(sampler.acceptance_fraction)
    print("done sampling in [%.1f sec] mean acceptance frac: %.2f (binned)" % (time.time() - tstart, mean_acc))

    # calculate medians of production chains as answer
    samples = sampler.chain.reshape((-1, nDim))

    # record median as the answer, and reconstruct all the parameters as a function of Mstar
    # and sample the percentiles (e.g. in {mu,sigma,A}-space not in Tparam-space)
    if relAmp:
        nParams = 5
    else:
        nParams = 6

    percs = np.percentile(samples, percentiles, axis=0)
    best_params = percs[int(len(percentiles) / 2), :]
    assert best_params.size == nDim

    p_error_accum = np.zeros((nParams, nBinsMass, nWalkers * nProdSteps), dtype="float32")

    for i in range(samples.shape[0]):
        sample = samples[i, :]

        if relAmp:
            # (mu1, sigma1, mu2, sigma2, A_rel) (1=blue, 2=red)
            p_error_accum[0, :, i] = _T(m, sample[10:15])  # mu blue
            p_error_accum[1, :, i] = _T(m, sample[0:5])  # sigma blue
            p_error_accum[2, :, i] = _T(m, sample[15:20])  # mu red
            p_error_accum[3, :, i] = _T(m, sample[5:10])  # sigma red
            p_error_accum[4, :, i] = sample[20:40]  # A_rel
        else:
            # (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
            p_error_accum[1, :, i] = _T(m, sample[10:15])  # mu blue
            p_error_accum[2, :, i] = _T(m, sample[0:5])  # sigma blue
            p_error_accum[4, :, i] = _T(m, sample[15:20])  # mu red
            p_error_accum[5, :, i] = _T(m, sample[5:10])  # sigma red
            p_error_accum[0, :, i] = sample[20:40]  # A_blue
            p_error_accum[3, :, i] = sample[40:60]  # A_red

    # shape is [Npercs,Nparams,Nmassbins]
    p_errors = np.percentile(p_error_accum, percentiles, axis=2)

    # create return parameters, reconstructing the mu_i and sigma_i from the best T() parameters
    # NOTE: these actually are biased since T() is nonlinear, use middle p_errors instead
    p = np.zeros((nParams, nBinsMass), dtype="float32")

    if relAmp:
        # (mu1, sigma1, mu2, sigma2, A_rel) (1=blue, 2=red)
        p[4, :] = best_params[20:40]  # A_rel
        p[0, :] = _T(m, best_params[10:15])  # mu blue
        p[2, :] = _T(m, best_params[15:20])  # mu red
        p[1, :] = _T(m, best_params[0:5])  # sigma blue
        p[3, :] = _T(m, best_params[5:10])  # sigma red
    else:
        # (A1, mu1, sigma1, A2, mu2, sigma2) (1=blue, 2=red)
        p[0, :] = best_params[20:40]  # blue A
        p[3, :] = best_params[40:60]  # red A
        p[1, :] = _T(m, best_params[10:15])  # mu blue
        p[4, :] = _T(m, best_params[15:20])  # mu red
        p[2, :] = _T(m, best_params[0:5])  # sigma blue
        p[5, :] = _T(m, best_params[5:10])  # sigma red

    # debug plots
    if 0:
        print(" making debug plots...")
        import corner
        import matplotlib.pyplot as plt

        saveStr = "%s_snap%d_%d_%d_%d_%e" % (relAmp, sP_snap, nWalkers, nBurnIn, nProdSteps, fracNoiseInit)

        # (A) sigma vs. chain #
        fig = plt.figure(figsize=(18, 12))

        for plotInd, i in enumerate(range(0, 5)):
            ax = fig.add_subplot(5, 2, plotInd + 1)
            ax.set_xlabel("chain step")
            ax.set_ylabel(r"$\sigma_{\rm blue}$ T[%d]" % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")
        for plotInd, i in enumerate(range(5, 10)):
            ax = fig.add_subplot(5, 2, plotInd + 1 + 5)
            ax.set_ylabel(r"$\sigma_{\rm red}$ T[%d]" % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")

        fig.savefig("debug_methodC_%s_sigma.pdf" % saveStr)
        plt.close(fig)

        # (B) mu vs. chain #
        fig = plt.figure(figsize=(18, 12))

        for plotInd, i in enumerate(range(10, 15)):
            ax = fig.add_subplot(5, 2, plotInd + 1)
            ax.set_xlabel("chain step")
            ax.set_ylabel(r"$\mu_{\rm blue}$ T[%d]" % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")
        for plotInd, i in enumerate(range(15, 20)):
            ax = fig.add_subplot(5, 2, plotInd + 1 + 5)
            ax.set_ylabel(r"$\mu_{\rm red}$ T[%d]" % i)
            for walkerInd in range(nWalkers):
                ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")

        fig.savefig("debug_methodC_%s_mu.pdf" % saveStr)
        plt.close(fig)

        # (C) A blue
        for iterNum in [0, 1]:
            fig = plt.figure(figsize=(18, 12))

            for plotInd, i in enumerate(range(20 + 10 * iterNum, 30 + 10 * iterNum)):
                ax = fig.add_subplot(5, 2, plotInd + 1)
                ax.set_xlabel("chain step")
                ax.set_ylabel(r"A$_{\rm blue}$ T[%d]" % i)
                for walkerInd in range(nWalkers):
                    ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")

            fig.savefig("debug_methodC_%s_Aset%d_blue.pdf" % (saveStr, iterNum))
            plt.close(fig)

        # (D) A red
        if not relAmp:
            for iterNum in [0, 1]:
                fig = plt.figure(figsize=(18, 12))

                for plotInd, i in enumerate(range(40 + 10 * iterNum, 50 + 10 * iterNum)):
                    ax = fig.add_subplot(5, 2, plotInd + 1)
                    ax.set_xlabel("chain step")
                    ax.set_ylabel(r"A$_{\rm red}$ T[%d]" % i)
                    for walkerInd in range(nWalkers):
                        ax.plot(np.arange(nProdSteps), sampler.chain[walkerInd, :, i], lw=0.8, alpha=0.5, color="black")

                fig.savefig("debug_methodC_%s_Aset%d_red.pdf" % (saveStr, iterNum))
                plt.close(fig)

        # (E) corners
        fig = corner.corner(samples[:, 0:5])
        fig.savefig("debug_methodC_%s_corner_sigma_blue.pdf" % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:, 5:10])
        fig.savefig("debug_methodC_%s_corner_sigma_red.pdf" % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:, 10:15])
        fig.savefig("debug_methodC_%s_corner_mu_blue.pdf" % saveStr)
        plt.close(fig)

        fig = corner.corner(samples[:, 15:20])
        fig.savefig("debug_methodC_%s_corner_mu_red.pdf" % saveStr)
        plt.close(fig)

    return p, p_errors, best_params


def characterizeColorMassPlane(
    sP,
    bands=("g", "r"),
    cenSatSelect="all",
    simColorsModel=defSimColorModel,
    nBurnIn=10000,
    remakeFlag=True,
    newErrors=False,
):
    """Characterize the red and blue populations, e.g. their location, extent, and relative numbers."""
    assert cenSatSelect in ["all", "cen", "sat"]

    # global analysis config
    mag_range = bandMagRange(bands)
    xMinMax = [9.0, 12.0]

    binSizeMass = 0.15
    binSizeColor = 0.04  # 0.05=20 0.04=25 0.03125=32 0.025=40

    # method A,B config only
    skipNBinsBlue = 9  # skip N most-massive mass bins when fitting blue population
    skipNBinsRed = 4  # skip N least-massive mass bins when fitting red population

    # method C (mcmc) config
    nWalkers = 200
    # nBurnIn = 10000 # [400], 2000, 10000
    if nBurnIn == 400:
        nProdSteps = 100
    if nBurnIn == 2000:
        nProdSteps = 200
    if nBurnIn == 10000:
        nProdSteps = 1000
    assert nBurnIn in [400, 2000, 10000]  # otherwise generalize nProdSteps

    fracNoiseInit = 2e-3
    percentiles = [1, 10, 50, 90, 99]  # middle is used to derive the best-fit for each parameter (e.g. median)

    startMCMCAtPreviousResult = True  # use previous snap MCMC result as starting point for z>0?
    startMCMC_snaps = [99, 91, 84, 78, 72, 67, 59, 50]  # what snapshot sequence?
    startObsMCMCAtSimResult = True  # use z=0.1 simulation result as a starting guess for obs fit?

    # model config
    paramInds = {"A_blue": 0, "mu_blue": 1, "sigma_blue": 2, "A_red": 3, "mu_red": 4, "sigma_red": 5}

    paramIndsRel = {"mu_blue": 0, "sigma_blue": 1, "mu_red": 2, "sigma_red": 3, "A_rel": 4}

    # derived
    nBinsMass = int(np.ceil((xMinMax[1] - xMinMax[0]) / binSizeMass))
    nBinsColor = int((mag_range[1] - mag_range[0]) / binSizeColor)

    conf = locals()  # store configuration variables into a dict for passing

    assert skipNBinsBlue >= 1  # otherwise logic failure below
    assert percentiles[int(len(percentiles) / 2)] == 50

    # check existence
    r = {}

    startMCMC_fromSnap = None

    eStr = "_err1" if newErrors else ""

    if sP is not None:
        # sim
        pStr = ""
        if startMCMCAtPreviousResult:
            assert sP.snap in startMCMC_snaps
            startMCMC_ind = startMCMC_snaps.index(sP.snap) - 1
            if startMCMC_ind >= 0:
                # set snapshot we will load previous modelC final chain state from
                startMCMC_fromSnap = startMCMC_snaps[startMCMC_ind]
                pStr = "_chainf=%d" % startMCMC_snaps[startMCMC_ind]

        savePath = sP.derivPath + "/galMstarColor/"
        saveFilename = savePath + "colorMassPlaneFits_%s_%d_%s_%s%s_mcmc%d-%d%s.hdf5" % (
            "".join(bands),
            sP.snap,
            cenSatSelect,
            simColorsModel,
            pStr,
            nBurnIn,
            nProdSteps,
            eStr,
        )
    else:
        # obs
        assert cenSatSelect == "all"
        assert simColorsModel == defSimColorModel

        sStr = "_fsr" if startObsMCMCAtSimResult else ""

        savePath = expanduser("~") + "/obs/"
        saveFilename = savePath + "sdss_colorMassPlaneFits_%s_%d-%d_%d-%d_mcmc%d-%d%s%s.hdf5" % (
            "".join(bands),
            xMinMax[0] * 10,
            xMinMax[1] * 10,
            mag_range[0] * 10,
            mag_range[1] * 10,
            nBurnIn,
            nProdSteps,
            sStr,
            eStr,
        )

    if not remakeFlag and isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                r[key] = f[key][()]
        for k in conf:
            r[k] = conf[k]
        return r

    if sP is not None:
        # load colors
        if "gc_colors" in sP.data and "mstar2_log" in sP.data:
            print("sP load")
            gc_colors, mstar2_log = sP.data["gc_colors"], sP.data["mstar2_log"]
            assert sP.data["cenSatSelect"] == cenSatSelect
        else:
            gc_colors, _ = loadSimGalColors(sP, simColorsModel, bands=bands, projs="random")
            # gc_colors = np.reshape( gc_colors, gc_colors.shape[0]*gc_colors.shape[1] )

            # load stellar masses (<2rhalf definition)
            gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
            mstar2_log = sP.units.codeMassToLogMsun(gc[:, sP.ptNum("stars")])

            # cen/sat selection
            wSelect = sP.cenSatSubhaloIndices(cenSatSelect=cenSatSelect)
            gc_colors = gc_colors[wSelect]  # [wSelect,:]
            mstar2_log = mstar2_log[wSelect]

            # store in sP for temporary testing
            sP.data["gc_colors"], sP.data["mstar2_log"], sP.data["cenSatSelect"] = gc_colors, mstar2_log, cenSatSelect
    else:
        # load observational points, restrict colors to mag_range as done for sims (for correct norm.)
        sdss_color, sdss_Mstar = calcSDSSColors(bands, eCorrect=True, kCorrect=True)

        w = np.where((sdss_color >= mag_range[0]) & (sdss_color <= mag_range[1]))
        gc_colors = sdss_color[w]
        mstar2_log = sdss_Mstar[w]

    sP_snap = sP.snap if sP is not None else 0

    def _paramDictFromModelCResult(cm_fit, relAmpStr):
        """Helper function used below."""
        r = {}
        r["sigma_blue"] = cm_fit["C%s_fstate" % relAmpStr][0:5]
        r["sigma_red"] = cm_fit["C%s_fstate" % relAmpStr][5:10]
        r["mu_blue"] = cm_fit["C%s_fstate" % relAmpStr][10:15]
        r["mu_red"] = cm_fit["C%s_fstate" % relAmpStr][15:20]

        if relAmp:
            r["A_rel"] = cm_fit["C%s_fstate" % relAmpStr][20:40]
        else:
            r["A_blue"] = cm_fit["C%s_fstate" % relAmpStr][20:40]
            r["A_red"] = cm_fit["C%s_fstate" % relAmpStr][40:60]

        return r

    # (A) double gaussian fits in 0.1 dex mstar bins, unconstrained (unrelated)
    # Levenberg-Marquadrt non-linear least squares minimization method
    for relAmp in [True]:  # [True,False]:
        print("relAmp: ", relAmp, " sP.snap: ", sP_snap)
        relAmpStr = "rel" if relAmp else ""
        pInds = paramIndsRel if relAmp else paramInds

        (
            r["A%s_params" % relAmpStr],
            r["A%s_schechter" % relAmpStr],
            r["mStar"],
            r["mStarCounts"],
            r["mHists"],
            r["mColorBins"],
        ) = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, relAmp=relAmp, **conf)

        for i in range(nBinsMass):
            params = r["A%s_params" % relAmpStr][:, i]
            print(" A %2d" % i, " ".join(["%6.3f" % p for p in params]))

        # (B) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the
        # T() function as in Baldry+ 2003, iterative fit (LM-LSF for each step)
        Tparams_prev = None

        # (B1) choose estimates for initial T() function parameters (only use for iterNum == 0)
        params_guess = [0.05, 0.1, 0.1, 10.0, 1.0]

        for iterNum in range(20):
            Tparams = {}

            # (B2) fit double gaussians, all mass bins
            p, _, m, _, _, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, relAmp=relAmp, **conf)

            # (B3) fit T(sigma), ignoring most-massive N bins for blue, least-massive N bin for red
            x_data = m[:-skipNBinsBlue]
            y_data = p[pInds["sigma_blue"], :-skipNBinsBlue]
            params_init = params_guess if Tparams_prev is None else Tparams_prev["sigma_blue"]
            Tparams["sigma_blue"], _ = leastsq_fit(_T, params_init, args=(x_data, y_data))

            x_data = m[skipNBinsRed:]
            y_data = p[pInds["sigma_red"], skipNBinsRed:]
            params_init = params_guess if Tparams_prev is None else Tparams_prev["sigma_red"]
            Tparams["sigma_red"], _ = leastsq_fit(_T, params_init, args=(x_data, y_data))

            # (B4) re-fit double gaussians with fixed sigma
            fixed = {}
            for key in Tparams:
                fixed[key] = _T(m, Tparams[key])

            p, _, m, _, _, _ = _fitCMPlaneDoubleGaussian(mstar2_log, gc_colors, fixed=fixed, relAmp=relAmp, **conf)

            # (B5) fit T(mu), ignoring same bins as for T(sigma)
            x_data = m[:-skipNBinsBlue]
            y_data = p[pInds["mu_blue"], :-skipNBinsBlue]
            params_init = params_guess if Tparams_prev is None else Tparams_prev["mu_blue"]
            Tparams["mu_blue"], _ = leastsq_fit(_T, params_init, args=(x_data, y_data))

            x_data = m[skipNBinsRed:]
            y_data = p[pInds["mu_red"], skipNBinsRed:]
            params_init = params_guess if Tparams_prev is None else Tparams_prev["mu_red"]
            Tparams["mu_red"], _ = leastsq_fit(_T, params_init, args=(x_data, y_data))

            # (B6) calculate change of both sets of T() parameters versus previous iteration
            if Tparams_prev is not None:
                diff_sum = 0.0

                for key in Tparams:
                    diff_local = (Tparams[key] - Tparams_prev[key]) ** 2.0
                    diff_local = np.sum(np.sqrt(diff_local))
                    diff_sum += diff_local
                    print(" [iter %2d]" % iterNum, key, diff_local, diff_sum)

                if diff_sum < 1e-4:
                    break

            Tparams_prev = Tparams

        assert diff_sum < 0.01  # otherwise we failed to converge

        # (B7) re-fit double gaussians with fixed sigma and mu from final T() functions
        fixed = {}
        for key in Tparams:
            fixed[key] = _T(m, Tparams[key])

        w = np.where(fixed["mu_blue"] >= fixed["mu_red"])
        fixed["mu_blue"][w] = fixed["mu_red"][w] - 0.01  # enforce mu_blue<mu_red

        r["B%s_params" % relAmpStr], r["B%s_schechter" % relAmpStr], _, _, _, _ = _fitCMPlaneDoubleGaussian(
            mstar2_log, gc_colors, fixed=fixed, relAmp=relAmp, **conf
        )

        for i in range(nBinsMass):
            params = r["B%s_params" % relAmpStr][:, i]
            print(" B %2d" % i, " ".join(["%6.3f" % param for param in params]))

        # (C) double gaussian fits in 0.1 dex mstar bins, with widths and centers constrained by the
        # T() function as in Baldry+ 2003, use the previous result as the starting point for a full
        # simultaneous MCMC fit of all [40/60 of] the parameters
        if startMCMC_fromSnap is None:
            # make sure p0 guess for MCMC is reasonable (within priors)
            Tparams["mu_blue"], _ = leastsq_fit(_T, Tparams["mu_blue"], args=(m, p[pInds["mu_blue"], :]))
            ###Tparams['sigma_blue'], _ = leastsq_fit(_T, Tparams['sigma_blue'], args=(m,p[pInds['sigma_blue']]))

            for key in Tparams:
                # check for invalid chain starting points and give some heuristic corrections
                param_vals = _T(m, Tparams[key])
                if param_vals.max() > 1.0:
                    Tparams[key][2] = 0.0
                if param_vals.max() > 1.0:
                    Tparams[key][0] = 1.0
                if param_vals.min() < 0.0:
                    Tparams[key][1] = -0.212
                param_vals = _T(m, Tparams[key])
                assert param_vals.min() > 0.0 and param_vals.max() < 1.0

            val_mu_blue = _T(m, Tparams["mu_blue"])
            val_mu_red = _T(m, Tparams["mu_red"])
            assert (val_mu_blue > val_mu_red).sum() == 0

            chain_start = Tparams  # contains: mu_blue, sigma_blue, mu_red, sigma_red

            if relAmp:
                chain_start["A_rel"] = r["B%s_params" % relAmpStr][4, :]
            else:
                chain_start["A_blue"] = r["B%s_params" % relAmpStr][0, :]
                chain_start["A_red"] = r["B%s_params" % relAmpStr][3, :]
        else:
            # load previous results from startMCMC_fromSnap
            print("loading chain final state from snap [%d] for C" % startMCMC_fromSnap)
            curSnap = sP.snap
            sP.setSnap(startMCMC_fromSnap)
            fits_prev = characterizeColorMassPlane(
                sP,
                bands=bands,
                cenSatSelect=cenSatSelect,
                simColorsModel=simColorsModel,
                remakeFlag=False,
                newErrors=newErrors,
            )
            sP.setSnap(curSnap)

            # reconstruct chain_start dictionary
            chain_start = _paramDictFromModelCResult(fits_prev, relAmpStr)

        if sP is None and startObsMCMCAtSimResult:
            # load simulation result as a reasonable starting point for the observational fit
            print("loading sim result from L75n1820TNG z=0.0 for obs C")
            sP = simParams(res=1820, run="tng", redshift=0.0)
            fits_sim = characterizeColorMassPlane(
                sP,
                bands=bands,
                cenSatSelect=cenSatSelect,
                simColorsModel=simColorsModel,
                remakeFlag=False,
                newErrors=newErrors,
            )

            # reconstruct chain_start dictionary
            chain_start = _paramDictFromModelCResult(fits_sim, relAmpStr)

        # run mcmc fit
        r["C%s_params" % relAmpStr], r["C%s_errors" % relAmpStr], r["C%s_fstate" % relAmpStr] = _fitCMPlaneMCMC(
            mstar2_log, gc_colors, chain_start, relAmp=relAmp, sP_snap=sP_snap, **conf
        )

        for i in range(nBinsMass):
            params = r["C%s_params" % relAmpStr][:, i]
            print(" C %2d" % i, " ".join(["%6.3f" % p for p in params]))

        # (D) simultaneous MCMC fit, where we additionally require that the amplitudes of each of the
        # red and blue follow a double-schechter function in log(Mstar), in which case there is only
        # one global 'relative fraction' instead of one A_rel per mass bin
        # (E) simultaneous MCMC fit, where we additionally require (1+z)^a evolution across redshift
        # bins of e.g. the mu_i(Mstar), sigma_i(Mstar), and A_rel(Mstar)

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            f[key] = r[key]
    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    for k in conf:
        r[k] = conf[k]
    return r


def colorTransitionTimes(sP, f_red, f_blue, maxRedshift, nBurnIn, bands=("g", "r"), simColorsModel=defSimColorModel):
    """Measure the various color boundary crossing times for all galaxies."""
    import json

    import matplotlib.pyplot as plt  # for debug plots

    # analysis config
    cenSatSelect = "cen"  # actually UNUSED! delete (all galaxies are handled)
    fit_method = "Crel"  # method/model used to characterize red vs blue populations in the C-M plane
    cmPlaneCSS = "all"  # which galaxies to use for color-mass plane fits
    cmPlaneRedshifts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]  # when to load color-mass plane fits
    cmPlaneColorModel = defSimColorModel  # which simColorsModel to use for color-mass plane fits
    redBlueMinSepMag = 0.02  # minimum separation between C_blue and C_red boundaries for stability

    def _plaw1plusz(x, params, fixed=None):
        """f(z) = A*(1+z)^b powerlaw function for fitting. Note: fixed argument unused."""
        (a, b) = params
        y = a * (1.0 + x) ** b
        return y

    # check existence
    r = {}

    savePath = sP.derivPath + "/galMstarColor/"
    saveFilename = savePath + "colorTransitionTimes_%s_%d_%s_%s_maxz=%.1f_fred=%.1f_fblue=%.1f.hdf5" % (
        "".join(bands),
        sP.snap,
        cenSatSelect,
        simColorsModel,
        maxRedshift,
        f_red,
        f_blue,
    )

    if isfile(saveFilename):
        with h5py.File(saveFilename, "r") as f:
            for key in f:
                if f[key].dtype == "O":
                    # json encoded string of a dict (convert string subhaloIDs keys back to int keys)
                    dd = json.loads(f[key][()])
                    r[key] = {int(key): val for key, val in dd.items()}
                else:
                    # normal ndarray
                    r[key] = f[key][()]
        return r

    # [1] load/calculate evolution of simulation colors, cached in sP.data
    if "sim_colors_evo" in sP.data:
        sim_colors_evo, shID_evo, subhalo_ids, evo_snaps = (
            sP.data["sim_colors_evo"],
            sP.data["shID_evo"],
            sP.data["subhalo_ids"],
            sP.data["evo_snaps"],
        )
    else:
        sim_colors_evo, shID_evo, subhalo_ids, evo_snaps = calcColorEvoTracks(
            sP, bands=bands, simColorsModel=simColorsModel
        )
        sP.data["sim_colors_evo"], sP.data["shID_evo"], sP.data["subhalo_ids"], sP.data["evo_snaps"] = (
            sim_colors_evo,
            shID_evo,
            subhalo_ids,
            evo_snaps,
        )

    redshifts = sP.snapNumToRedshift(evo_snaps)
    ages = sP.units.redshiftToAgeFlat(redshifts)

    w = np.where(redshifts <= maxRedshift)

    redshifts = redshifts[w]
    ages = ages[w]
    snaps = evo_snaps[w]

    # [2] load color-mass plane characterizations at every snapshot
    cmPlaneRedshifts = np.array(cmPlaneRedshifts)
    cm_fits = []

    for redshift in cmPlaneRedshifts:
        sP.setRedshift(redshift)
        print("load cm-plane fits: snap [%d] z = %.1f" % (sP.snap, redshift))
        fits_local = characterizeColorMassPlane(
            sP,
            bands=bands,
            cenSatSelect=cmPlaneCSS,
            simColorsModel=cmPlaneColorModel,
            nBurnIn=nBurnIn,
            remakeFlag=False,
        )
        cm_fits.append(fits_local)

    sP.setRedshift(redshifts[0])

    # [3] fit red and blue mu and sigma vs redshift
    pInds = cm_fits[0]["paramIndsRel"] if "rel" in fit_method else cm_fits[0]["paramInds"]
    masses = cm_fits[0]["mStar"]

    medianBin = int(cm_fits[0][fit_method + "_errors"].shape[0] / 2)
    pVals = {}

    for pName in pInds.keys():
        print(pName)

        pVals[pName] = np.zeros((cm_fits[0]["nBinsMass"], cmPlaneRedshifts.size), dtype="float32")
        pVals[pName + "_fit"] = np.zeros_like(pVals[pName])
        pVals[pName + "_fitp"] = np.zeros((2, cm_fits[0]["nBinsMass"]), dtype="float32")

        for i in range(len(cmPlaneRedshifts)):
            pVals[pName][:, i] = cm_fits[i][fit_method + "_errors"][medianBin, pInds[pName], :]

        # fit each mass bin separately in redshift as A*(1+z)^B
        params_init = [pVals[pName][0, 0], 1.0]
        for i in range(cm_fits[0]["nBinsMass"]):
            x_data = cmPlaneRedshifts
            y_data = pVals[pName][i, :]

            p_fit, p_err = leastsq_fit(_plaw1plusz, params_init, args=(x_data, y_data))
            params_init = p_fit  # use previous for initial guess of next mass bin

            # print(' ',pName,i,masses[i],p_fit)

            pVals[pName + "_fit"][i, :] = _plaw1plusz(x_data, p_fit)
            pVals[pName + "_fitp"][:, i] = p_fit

        # start plots for the redshift evolution of this parameter and its fit
        if 0:
            fig, ax = plt.subplots()

            ax.set_xlim([9.0, 12.0])
            ax.set_xlabel(r"M$_{\star}$ [ log M$_{\rm sun}$ ]")
            ax.set_ylabel(pName)

            for i in range(len(cmPlaneRedshifts)):
                yy = pVals[pName][:, i]
                yy_fit1 = pVals[pName + "_fit"][:, i]

                (l,) = ax.plot(masses, yy, "o:", markerfacecolor="none")
                ax.plot(masses, yy_fit1, "o-", label="z = %.1f" % redshift, color=l.get_color())

            ax.legend(loc="best")

            # finish plot and save
            fig.savefig("cmPlaneParamEvoWithFit_%s_%s_%s.pdf" % (sP.simName, fit_method, pName))
            plt.close(fig)

    # [4] define C_blue and C_red functions
    def C_redblue(Mstar, redshift):
        """Find mu_{red,blue} and sigma_{red,blue} at this (Mstar,z) and derive the boundaries C_red and C_blue."""
        w = np.where(Mstar > masses)[0]
        if len(w):
            mstar_ind0 = w.max()
        else:
            mstar_ind0 = 0  # extrapolation to lower mass than fit

        mstar_ind1 = mstar_ind0 + 1

        if mstar_ind1 == masses.size:
            # extrapolation to higher mass than fit
            mstar_ind0 -= 1
            mstar_ind1 -= 1

        assert mstar_ind0 >= 0 and mstar_ind1 < masses.size

        # linear interpolation in mass
        C_val = {}

        x0 = masses[mstar_ind0]
        x1 = masses[mstar_ind1]

        for whichColor in ["red", "blue"]:
            mu_0 = _plaw1plusz(redshift, pVals["mu_%s_fitp" % whichColor][:, mstar_ind0])
            mu_1 = _plaw1plusz(redshift, pVals["mu_%s_fitp" % whichColor][:, mstar_ind1])

            mu = mu_0 + (Mstar - x0) * (mu_1 - mu_0) / (x1 - x0)

            sigma_0 = _plaw1plusz(redshift, pVals["sigma_%s_fitp" % whichColor][:, mstar_ind0])
            sigma_1 = _plaw1plusz(redshift, pVals["sigma_%s_fitp" % whichColor][:, mstar_ind1])

            sigma = sigma_0 + (Mstar - x0) * (sigma_1 - sigma_0) / (x1 - x0)

            # boundary definition
            if whichColor == "red":
                C_val["red"] = mu - f_red * sigma
            elif whichColor == "blue":
                C_val["blue"] = mu + f_blue * sigma

                if C_val["blue"] >= C_val["red"] - redBlueMinSepMag:
                    C_val["blue"] = C_val["red"] - redBlueMinSepMag

        return C_val["red"], C_val["blue"]

    # plot redshift evolution of C_blue and C_red color boundaries
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    ax.set_xlim([9.0, 12.0])
    ax.set_xlabel(r"M$_{\star}$ [ log M$_{\rm sun}$ ]")
    ax.set_ylabel(r"C$_{\rm red}$ (dotted) or C$_{\rm blue}$ (solid) boundaries")
    ax.set_ylim([0.3, 0.8])

    redshifts_plot = np.linspace(0.0, 1.0, 5)
    masses_plot = np.linspace(9.0, 12.0, 100)

    for _i, redshift in enumerate(redshifts_plot):
        C_blue = np.zeros(masses_plot.size, dtype="float32")
        C_red = np.zeros(masses_plot.size, dtype="float32")

        # derive boundaries at this redshift
        for j, mass in enumerate(masses_plot):
            C_red[j], C_blue[j] = C_redblue(mass, redshift)

        (l,) = ax.plot(masses_plot, C_blue, "-", lw=2.0, label="z = %.1f" % redshift)
        ax.plot(masses_plot, C_red, ":", lw=2.0, color=l.get_color())

    ax.legend(loc="best")

    # finish plot and save
    fig.savefig("cmPlaneBoundariesEvo_%s_%s_mcmc%d.pdf" % (sP.simName, fit_method, nBurnIn))
    plt.close(fig)

    # [5] load stellar masses at all snapshots
    print("loading stellar masses...")
    mstar_evo = np.zeros((subhalo_ids.size, snaps.size), dtype="float32")
    mstar_evo.fill(np.nan)

    for i, snap in enumerate(snaps):
        snapInd = tuple(evo_snaps).index(snap)
        print(" [%d] snapInd = %d (snapshot = %d z = %.2f): " % (i, snapInd, snap, redshifts[i]), end="")

        # load the corresponding mstar values at this snapshot
        sP.setSnap(snap)

        gc = sP.groupCat(fieldsSubhalos=["SubhaloMassInRadType"])
        mstar_log = sP.units.codeMassToLogMsun(gc[:, sP.ptNum("stars")])

        # subhaloIDs in the color evo tracks (index subhalos in groupcat at this snap)
        colorSHIDs_thisSnap = np.squeeze(shID_evo[:, snapInd])

        w = np.where(colorSHIDs_thisSnap >= 0)[0]  # otherwise untracked to this snap

        frac_tracked = float(len(w)) / subhalo_ids.size * 100.0
        print("total = %d tracked_here = %d (%.2f%%)" % (subhalo_ids.size, len(w), frac_tracked))

        mstar_evo[w, i] = mstar_log[colorSHIDs_thisSnap[w]]

    sP.setRedshift(redshifts[0])

    # boundary helpers
    def _locateBoundaryCrossing(M0, C0, z0, M1, C1, z1, boundaryType):
        """Locate the crossing of either the red or blue boundary between two points in (M,C,z) space.

        Use an iterative bracketing to locate the exact crossing point (M',C',z') of
        mass, color, and redshift where the line connecting (M0,C0,z0) to (M1,C1,z1)
        intersects either C_red(M',C',z') or C_blue(M',C',z') according to boundaryType.
        """
        assert np.all(np.isfinite([M0, C0, z0, M1, C1, z1]))

        # config
        errTol = 1e-5
        maxIter = 50

        # initial bounds
        z_lower = z0
        z_upper = z1

        dM_invDeltaZ = (M1 - M0) / (z1 - z0)
        dC_invDeltaZ = (C1 - C0) / (z1 - z0)

        for i in range(maxIter):
            # current guess
            z_guess = 0.5 * (z_lower + z_upper)

            # interpolate stellar mass and color to this redshift
            Mstar_local = M0 + (z_guess - z0) * dM_invDeltaZ
            color_local = C0 + (z_guess - z0) * dC_invDeltaZ

            # calculate boundary at this redshift
            C_red, C_blue = C_redblue(Mstar_local, z_guess)
            C_threshold = C_red if boundaryType == "red" else C_blue

            # assert C0 < C_threshold and C1 > C_threshold # right? well just bounded

            # note: z1 > z0, but could have either C0>C1 or C1>C0
            if color_local < C_threshold:
                # we are (above), move lower bound up
                z_upper = z_guess
            else:
                # we are (below), move upper bound down
                z_lower = z_guess

            curError = np.abs(color_local - C_threshold)
            # print(i,z_guess,Mstar_local,color_local,C_threshold,curError)
            if curError <= errTol:
                # print(' break')
                break

            if i >= maxIter:
                print(" warning! nIter == maxIter == %d, final error = %.2f" % (i, curError))

        return z_guess, Mstar_local

    def _locateCrossingWrap(ind, snapInd, viewInd, boundaryType):
        """Wrapper around _locateBoundaryCrossing() to get needed values."""
        C0 = sim_colors_evo[ind, viewInd, snapInd]
        C1 = sim_colors_evo[ind, viewInd, snapInd + 1]
        M0 = mstar_evo[ind, snapInd]
        M1 = mstar_evo[ind, snapInd + 1]
        z0 = redshifts[snapInd]
        z1 = redshifts[snapInd + 1]
        return _locateBoundaryCrossing(M0, C0, z0, M1, C1, z1, boundaryType)

    # allocate
    r = {}
    r["subhalo_ids"] = subhalo_ids
    r["snaps"] = snaps

    for key in ["z_blue", "z_redini", "M_blue", "M_redini"]:
        r[key] = np.zeros(subhalo_ids.size, dtype="float32")
        r[key].fill(np.nan)

    r["N_rejuv"] = np.zeros(subhalo_ids.size, dtype="int16")
    r["z_rejuv_start"] = {}  # dictionaries, keys are subhalo ids
    r["z_rejuv_stop"] = {}  # elements are [] which are appended to
    r["M_rejuv_start"] = {}
    r["M_rejuv_stop"] = {}

    # loop over all (todo) viewing directions, since colors are projection-dependent
    # for viewInd in range(sim_colors_evo.shape[1]):
    viewInd = 0
    print("viewInd = 0 (todo)")

    # [6] loop over all analysis snapshots, moving forward in time from z=maxRedshift to z=0
    for i, snap in enumerate(snaps[::-1]):
        # get properties at this anapshot
        snapInd = tuple(snaps).index(snap)
        redshift = redshifts[snapInd]
        print("[%d] snapInd = %d snapshot = %d z = %.2f" % (i, snapInd, snap, redshift))

        loc_color = np.squeeze(sim_colors_evo[:, viewInd, snapInd])
        loc_mass = np.squeeze(mstar_evo[:, snapInd])

        w_valid = np.where(np.isfinite(loc_mass) & np.isfinite(loc_color) & (loc_mass > 0.0))
        n_valid = len(w_valid[0])

        loc_color = loc_color[w_valid]
        loc_mass = loc_mass[w_valid]

        C_blue = np.zeros(n_valid, dtype="float32")
        C_red = np.zeros(n_valid, dtype="float32")

        # compute current red/blue boundaries for the (mass,z) of every galaxy (not easily vectorized)
        for j in range(n_valid):
            if j % 20000 == 0:
                print(" %.1f%%" % (float(j) / n_valid * 100.0), end="")
            assert ~np.isnan(loc_mass[j])  # should have been filtered out

            C_red[j], C_blue[j] = C_redblue(loc_mass[j], redshift)
        print(" 100%")

        # flag every galaxy as either C<C_blue, C_blue<C<C_red, or C>C_red
        assert np.count_nonzero(C_blue > C_red) == 0
        mask = np.zeros(n_valid, dtype="int32")

        w1 = np.where(loc_color <= C_blue)
        w2 = np.where((loc_color > C_blue) & (loc_color <= C_red))
        w3 = np.where(loc_color > C_red)

        print(
            " num blue = %d green = %d red = %d [total %d of %d]"
            % (len(w1[0]), len(w2[0]), len(w3[0]), n_valid, subhalo_ids.size)
        )

        # verify is a full, and disjoint, subset
        mask[w1] += 1
        mask[w2] += 1
        mask[w3] += 1
        assert mask.min() == 1 and mask.max() == 1

        # save indices of galaxy classification at this snapshot
        cur_blue_inds = w_valid[0][w1]
        cur_green_inds = w_valid[0][w2]
        cur_red_inds = w_valid[0][w3]

        # if we are on the first snapshot, we are done
        if i == 0:
            prev_blue_inds = cur_blue_inds
            prev_green_inds = cur_green_inds
            prev_red_inds = cur_red_inds
            continue

        nFailed = {"red": 0, "blue": 0, "green": 0}

        # currently red systems
        tstart = time.time()
        print(" red...", end="")

        for ind in cur_red_inds:
            found = 0
            # previously red: no action
            if ind in prev_red_inds:
                found += 1

            # previously green or blue:
            for prev_ind_set in [prev_green_inds, prev_blue_inds]:
                if ind in prev_ind_set:
                    found += 1
                    if np.isnan(r["z_redini"][ind]):
                        # record first entrance into the red population
                        z_cross, M_cross = _locateCrossingWrap(ind, snapInd, viewInd, "red")

                        r["z_redini"][ind] = z_cross
                        r["M_redini"][ind] = M_cross
                    else:
                        # finished rejuvenation event
                        if ind not in r["z_rejuv_start"] or ind not in r["M_rejuv_start"]:
                            print(" WARNING! ind %d finished rejuv but we missed the start." % ind)
                            continue
                        # assert ind in r['z_rejuv_start'] and ind in r['M_rejuv_start']

                        r["N_rejuv"][ind] += 1

                        if ind not in r["z_rejuv_stop"]:
                            r["z_rejuv_stop"][ind] = []
                        if ind not in r["M_rejuv_stop"]:
                            r["M_rejuv_stop"][ind] = []

                        z_cross, M_cross = _locateCrossingWrap(ind, snapInd, viewInd, "red")

                        r["z_rejuv_stop"][ind].append(z_cross)
                        r["M_rejuv_stop"][ind].append(M_cross)

            if found == 0:
                nFailed["red"] += 1
            assert found <= 1

        print(" %.1f sec" % (time.time() - tstart))

        # currently green systems
        tstart = time.time()
        print(" green...", end="")

        for ind in cur_green_inds:
            found = 0
            # previously red: record start of rejuvenation event
            if ind in prev_red_inds:
                found += 1
                if ind not in r["z_rejuv_start"]:
                    r["z_rejuv_start"][ind] = []
                if ind not in r["M_rejuv_start"]:
                    r["M_rejuv_start"][ind] = []

                z_cross, M_cross = _locateCrossingWrap(ind, snapInd, viewInd, "red")

                r["z_rejuv_start"][ind].append(z_cross)
                r["M_rejuv_start"][ind].append(M_cross)

            # previously green: no action
            if ind in prev_green_inds:
                found += 1

            # previously blue:
            if ind in prev_blue_inds:
                found += 1
                if np.isnan(r["z_blue"][ind]):
                    # record first exit from blue population
                    z_cross, M_cross = _locateCrossingWrap(ind, snapInd, viewInd, "blue")

                    r["z_blue"][ind] = z_cross
                    r["M_blue"][ind] = M_cross
                else:
                    # e.g. part of a rejuvenation event, no action
                    pass

            if found == 0:
                nFailed["green"] += 1
            assert found <= 1

        print(" %.1f sec" % (time.time() - tstart))

        # currently blue systems
        tstart = time.time()
        print(" blue...", end="")

        for ind in cur_blue_inds:
            found = 0
            # previously red: record start of rejuvenation event (note this happens either here or
            # as above for 'currently green systems', but can only happen in one place)
            if ind in prev_red_inds:
                found += 1
                if ind not in r["z_rejuv_start"]:
                    r["z_rejuv_start"][ind] = []
                if ind not in r["M_rejuv_start"]:
                    r["M_rejuv_start"][ind] = []

                z_cross, M_cross = _locateCrossingWrap(ind, snapInd, viewInd, "red")

                r["z_rejuv_start"][ind].append(z_cross)
                r["M_rejuv_start"][ind].append(M_cross)

            # previously green: no action
            if ind in prev_green_inds:
                found += 1

            # previously blue: no action
            if ind in prev_blue_inds:
                found += 1

            if found == 0:
                nFailed["blue"] += 1
            assert found <= 1

        # save indices of galaxy classification at this snapshot
        prev_blue_inds = cur_blue_inds
        prev_green_inds = cur_green_inds
        prev_red_inds = cur_red_inds

        print(" %.1f sec" % (time.time() - tstart))
        print(
            " failed to locate in previous inds, [red %d] [green %d] [blue %d]"
            % (nFailed["red"], nFailed["green"], nFailed["blue"])
        )

    # save
    with h5py.File(saveFilename, "w") as f:
        for key in r:
            # json encode dicts to strings, otherwise normal ndarray
            if isinstance(r[key], dict):
                f[key] = json.dumps(r[key])
            else:
                f[key] = r[key]
    print("Saved: [%s]" % saveFilename.split(savePath)[1])

    return r
