"""
Temporary code for MCMC and multi-Gaussian (GMM) fitting.
"""
import numpy as np
import time
import emcee

from scipy.special import erf
from ..projects.color_analysis import _double_gaussian, _double_gaussian_rel

def _single_gaussian(x, params, fixed=None):
    """ single gaussian test function """
    (A1, mu1, sigma1) = params
    y = A1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) )
    return y

def _single_gaussian_normed(x, params, fixed=None):
    """ single gaussian test function, normalized """
    (mu1, sigma1) = params
    A1 = 1.0 / np.sqrt(2*np.pi) / sigma1
    assert np.isfinite(A1)
    y = A1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) )
    return y

def testDoubleGauss():
    """ test function"""

    # config
    if 0:
        params_true = [0.88, 0.71, 0.07] # A, mu, sigma
        p0_guess = [0.5, 0.5, 0.2]
        func = _single_gaussian
        print('using single gaussian')
    if 1:
        params_true = [0.71, 0.07] # mu, sigma
        p0_guess = [0.5, 0.2]
        func = _single_gaussian_normed
        print('using single normed gaussian')
    if 0:
        params_true = [0.44, 0.30, 0.03, 0.88, 0.71, 0.07] # A_blue, mu_blue, s_blue, A_red, mu_red, s_red
        p0_guess = [1.5, 0.4, 0.2, 2.5, 0.6, 0.2]
        func = _double_gaussian
        print('using double gaussian')
    if 0:
        params_true = [0.30, 0.04, 0.71, 0.04, 0.75] # mu_blue, s_blue, mu_red, s_red, A_frac (blue fraction)
        p0_guess = [0.4, 0.06, 0.6, 0.06, 0.5]
        func = _double_gaussian_rel
        print('using double relative normed gaussian')

    x_bounds = [0.0, 1.0] # e.g. color
    y_max = 10.0 # e.g. height of gaussians
    N_samples = 500

    nBinsColor = 40
    binSizeColor = (x_bounds[1]-x_bounds[0])/nBinsColor

    # mcmc config
    nBurnIn = 1000
    nProdSteps = 200
    nWalkers = 40
    fracNoiseInit = 0.1

    nDim = len(params_true)

    # first use rejection-sampling to generate N samples from a Gaussian
    count = 0
    niter = 0
    x_data = np.zeros( N_samples, dtype='float32' )

    np.random.seed(42424242)

    while count < N_samples:
        niter += 1
        random_x = np.random.uniform(low=x_bounds[0], high=x_bounds[1], size=1)
        random_y = np.random.uniform(low=0.0, high=y_max, size=1)

        if random_y > func(random_x, params_true, fixed=None):
            continue

        x_data[count] = random_x
        count += 1

    print('rejection sampling: ',niter,count)

    # run binned MCMC
    # ---------------
    yy, xx = np.histogram(x_data, range=x_bounds, bins=nBinsColor, density=True)
    #yy = yy.astype('float32') / float(x_data.size) # sum is 1.0
    xx = xx[:-1] + binSizeColor/2.0

    if 1:
        def mcmc_lnprob_binned(theta, x, y):
            if len(theta) == 5:
                if theta[0] > theta[2]:
                    return -np.inf # prior that red mu is larger than blue mu
                if theta[4] >= 1.0:
                    return -np.inf # relative amplitude in [0,1]
            if theta.min() <= 0.0:
                return -np.inf # no mu or sigma or Afrac allowed to be negative

            y_err = 0.1
            inv_sigma2 = 1.0/y_err**2.0
            #inv_sigma2 = 1.0

            y_fit = func(x, theta, fixed=None)
            chi2 = np.sum( (y_fit - y)**2.0 * inv_sigma2 - np.log(inv_sigma2) )
            lnlike = -0.5 * chi2

            return lnlike

        p0_walkers = np.zeros( (nWalkers,nDim), dtype='float32' )
        np.random.seed(42424242)
        for i in range(nWalkers):
            p0_walkers[i,:] = p0_guess + np.abs(p0_guess) * np.random.normal(loc=0.0, scale=fracNoiseInit)

        tstart = time.time()
        sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_binned, args=(xx, yy))

        pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
        sampler.reset()

        sampler.run_mcmc(pos, nProdSteps)

        mean_acc = np.mean(sampler.acceptance_fraction)
        print('done sampling in [%.1f sec] mean acceptance frac: %.2f (binned)' % (time.time() - tstart,mean_acc))

        samples = sampler.chain.reshape( (-1,nDim) )
        percs = np.percentile(samples, [16,50,84], axis=0)
        for i in range(nDim):
            print(i, params_true[i], percs[:,i], samples[:,i].min(), samples[:,i].max())
        params_best_binned = percs[1,:]

    # run non-binned MCMC
    # -------------------
    if 1:
        def mcmc_lnprob_raw(theta, x_raw, dummy):
            if len(theta) == 5:
                if theta[0] > theta[2]:
                    return -np.inf # prior that red mu is larger than blue mu
                if theta[4] >= 1.0:
                    return -np.inf # relative amplitude in [0,1]
            if theta.min() <= 0.0:
                return -np.inf # no mu or sigma or Afrac allowed to be negative or zero

            n = x_raw.size

            if len(theta) == 2:
                mu, sigma = theta
                
                lnlike = -0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sigma**2) - \
                  1.0/(2*sigma**2) * np.sum( (x_raw - mu)** 2 )

                # this works, s.t. adding lnlike_amp to lnlike fits A to target_amplitude
                #target_amplitude = N_samples
                #int_max_bound = 1.0 # lower bound is zero
                #g_int = A * np.sqrt(0.5*np.pi) * sigma * \
                #        ( erf(mu/(np.sqrt(2)*sigma)) - erf( (-int_max_bound+mu)/(np.sqrt(2)*sigma) ) )
                #chi_amp = g_int / np.sqrt(2*np.pi*sigma**2) - target_amplitude
                #lnlike_amp = -0.5 * chi_amp**2
                #lnlike += lnlike_amp

            if len(theta) == 5:
                # https://www.statlect.com/fundamentals-of-statistics/normal-distribution-maximum-likelihood
                mu1, sigma1, mu2, sigma2, Arel = theta

                lnlike1 = n*np.log(Arel)-0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sigma1**2) - \
                  1.0/(2*sigma1**2) * np.sum( (x_raw - mu1)**2 )
                lnlike2 = n*np.log(1.0-Arel)-0.5*n*np.log(2*np.pi) - 0.5*n*np.log(sigma2**2) - \
                  1.0/(2*sigma2**2) * np.sum( (x_raw - mu2)**2 )

                assert np.isfinite(lnlike1) and np.isfinite(lnlike2)

                lnlike = lnlike1 + lnlike2 # likelihoods multiplied -> added in the log

            return lnlike

        p0_walkers = np.zeros( (nWalkers,nDim), dtype='float32' )
        np.random.seed(42424242)
        for i in range(nWalkers):
            #p0_walkers[i,:] = p0_guess + np.abs(p0_guess) * np.random.normal(loc=0.0, scale=fracNoiseInit)
            p0_walkers[i,:] = params_best_binned + \
              np.abs(params_best_binned) * np.random.normal(loc=0.0, scale=fracNoiseInit)
            
        for i in range(nDim):
            print('p0 min max ',i,p0_walkers[:,i].min(),p0_walkers[:,i].max())

        # do not fit amplitude
        #p0_walkers = p0_walkers[:,0:2]
        #nDim = 2

        tstart = time.time()
        sampler = emcee.EnsembleSampler(nWalkers, nDim, mcmc_lnprob_raw, args=(x_data,None))

        pos, prob, state = sampler.run_mcmc(p0_walkers, nBurnIn)
        sampler.reset()

        sampler.run_mcmc(pos, nProdSteps)

        mean_acc = np.mean(sampler.acceptance_fraction)
        print('done sampling in [%.1f sec] mean acceptance frac: %.2f (raw)' % (time.time() - tstart,mean_acc))

        samples = sampler.chain.reshape( (-1,nDim) )
        percs = np.percentile(samples, [16,50,84], axis=0)
        for i in range(nDim):
            print(i, params_true[i], percs[:,i], samples[:,i].min(), samples[:,i].max())
        #params_best_raw = np.zeros( nDim+1, dtype='float32' )
        #params_best_raw[1:] = percs[1,:]
        params_best_raw = percs[1,:]

        # set amplitude by requiring discrete integral equal the histogram (bracketing)
        if 0:
            amp_lower = p0_guess[0] * 1e-2
            amp_upper = p0_guess[0] * 1e2

            for niter in range(50):
                params_best_raw[0] = 0.5 * (amp_lower + amp_upper)

                yy_guess = func(xx, params_best_raw)
                
                if yy_guess.sum() >= yy.sum(): 
                    amp_upper = params_best_raw[0]
                if yy_guess.sum() < yy.sum(): 
                    amp_lower = params_best_raw[0]

                yy_diff = np.abs(yy_guess.sum() - yy.sum())
                #print(niter,yy_guess.sum(),yy.sum(),params_best_raw[0],yy_diff)

                if yy_diff < 1e-6:
                    break

            print(0,params_true[0],params_best_raw[0])
            for i in range(nDim):
                print(i+1, params_true[i+1], percs[:,i], samples[:,i].min(), samples[:,i].max())

    # towards luminosity function
    yy_counts_per_bin = yy*binSizeColor*N_samples
    yy_best_binned_counts = func(xx, params_best_binned)*binSizeColor*N_samples

    # make a plot
    # -----------
    if 1:
        import matplotlib.pyplot as plt
        #import corner

        # (A)
        fig = plt.figure(figsize=(18,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('color')
        ax.set_xlim(x_bounds)
        ax.set_ylabel('N normalized')

        x_plot = np.linspace(x_bounds[0],x_bounds[1],1000)
        ax.plot( x_plot, func(x_plot, params_true), '-', lw=2.5, label='True')
        ax.plot( xx, yy, 'o-', label='Histogrammed')
        
        ax.plot( x_plot, func(x_plot, params_best_binned), '-', lw=2.5, label='Bestfit Binned')
        ax.plot( x_plot, func(x_plot, params_best_raw), '-', lw=2.5, label='Bestfit Raw')
        #ax.plot( x_plot, func(x_plot, params_best_rawamp), '-', lw=2.5, label='Bestfit Raw w/ Amp')
        ax.legend()

        fig.savefig('test_A.pdf')
        plt.close(fig)

    import pdb; pdb.set_trace()
