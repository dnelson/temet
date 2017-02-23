"""
obs/sdss.py
  Observational data processing, reduction, and analysis (SDSS).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import cosmo.load
import h5py
import time
import requests

import matplotlib.pyplot as plt
from os.path import expanduser

def sdss_decompose_specobjid(id):
    """ Convert 64-bit SpecObjID into its parts, returning a dict. DR13 convention. """
    r = {}
    bin = np.binary_repr( id, width=64 )
    r['plate']   = int(bin[0:14], 2) # bits 50-63
    r['fiberid'] = int(bin[14:14+12], 2) # bits 38-49
    r['mjd']     = int(bin[14+12:14+12+14], 2) + 50000 # bits 24-37 minus 50000
    r['run2d']   = int(bin[14+12+14:14+12+14+14], 2) # bits 10-23
    return r

def sdss_decompose_objid(id):
    """ Convert 64-bit ObjID into its parts, returning a dict. DR13 convention. """
    r = {}
    bin = np.binary_repr( id, width=64 )
    r['rerun']  = int(bin[5:5+11], 2) # bits 48-58
    r['run']    = int(bin[5+11:5+11+16], 2) # bits 32-47
    r['camcol'] = int(bin[5+11+16:5+11+16+3], 2) # bits 29-31
    r['field']  = int(bin[5+11+16+3+1:5+11+16+3+1+12], 2) # bits 16-27
    r['id']     = int(bin[5+11+16+3+1+12:], 2) # bits 0-15

    return r

def loadSDSSSpectrum(ind):
    """ Remotely acquire (via http) a single SDSS galaxy spectrum, according to the input index 
    which corresponds to the sdss_z0.0-0.1.hdf5 datafile (z<0.1 targets)."""
    f1 = expanduser('~') + '/obs/sdss_z0.0-0.1.hdf5'
    f2 = expanduser('~') + '/obs/sdss_objid_specobjid_z0.0-0.5.hdf5'

    # get target objid
    with h5py.File(f1,'r') as f:
        objid = f['objid'][ind]
        logMass = f['logMass_gran1'][ind]
        redshift = f['redshift'][ind]

    # get matching specobjid
    with h5py.File(f2,'r') as f:
        objids = f['objid'][()]
        w = np.where(objids == objid)[0]
        specobjid = f['specobjid'][w[0]]

    print('[%d] objid: %d  specobjid: %d logMass: %.2f' % (ind,objid,specobjid,logMass))

    # construct url
    p = sdss_decompose_specobjid(specobjid)
    url_base = 'https://dr13.sdss.org/optical/spectrum/view/data/format%3Dcsv/spec%3Dlite?'
    url = url_base + 'mjd=%d&fiberid=%d&plateid=%d' % (p['mjd'],p['fiberid'],p['plate'])

    # acquire
    print(' ' + url)
    req = requests.get(url, headers={})

    if req.status_code != 200:
        print(' WARNING! Response code = %d, skipping!' % req.status_code)
        return None

    # parse
    rows = req.text.split('\n')

    r = {'ind':ind, 'objid':objid, 'specobjid':specobjid, 'logMass':logMass, 'redshift':redshift}
    cols = ['wavelength','flux','bestfit','skyflux']
    for col in cols:
        r[col] = np.zeros( len(rows)-2, dtype='float32' )

    # first line is header, last line is empty
    for i, row in enumerate(rows[1:-1]):
        # wavelength = angstroms
        # flux = coadded calibrated flux [10^-17 erg/s/cm2/angstrom]
        # skyflux = subtracted sky flux [same units]
        # bestfit = pipeline best model fit used for classification and redshift (?)
        # ivar = inverse variance (one over simga-squared), is 0.0 for bad pixels that should be ignored
        r['wavelength'][i], r['flux'][i], r['bestfit'][i], r['skyflux'][i] = row.split(',')

    return r

def load_obs(ind):
    """ Construct observational object with a SDSS spectrum, ready for fitting. """
    spec = loadSDSSSpectrum(ind=ind)

    # convert [10^-17 erg/cm^2/s/Ang] -> [10^-23 erg/cm^2/s/Hz] since flux_nu = (lambda^2/c) * flux_lambda
    # http://coolwiki.ipac.caltech.edu/index.php/Units, https://en.wikipedia.org/wiki/AB_magnitude
    # 3.34e4 = 1/(c * 1e3 * Jy_mks) = 1/(c * 1e3 * 1e-26) = 1/(3e18 ang/s * 1e3 * 1e-26)
    flux_Jy = 1e-17 * spec['flux'] * 3.34e4 * (spec['wavelength'])**2.0
    flux_maggies = flux_Jy / 3631.0
    flux_nMgy = flux_maggies * 1e9

    # define observation
    obs = {}
    obs['wavelength'] = spec['wavelength'] # vacuum Angstroms
    obs['spectrum'] = flux_maggies # units of maggies
    obs['unc'] = spec['flux'] * 0.05 # todo! use sqrt(1/ivar)
    obs['mask'] = (obs['unc'] != 0.0) # optional spec (bool array, set False for ivar==0)

    obs['maggies'] = None # no broadband filter magnitudes
    obs['maggies_unc'] = None # no associated uncertanties
    obs['filters'] = None # no associated filters
    obs['phot_mask'] = None  # optional, no associated masks

    # auxiliary information
    for k in ['ind','objid','specobjid','logMass','redshift']:
        obs[k] = spec[k]

    return obs

def load_model_params(redshift=None):
    """ Return the set of model parameters, including which are fixed and which are free, 
    the associated priors, and initial guesses. """
    from prospect.models import priors

    model_params = []

    if redshift is None:
        # no known redshift, fit freely
        model_params.append({'name': 'zred', 'N': 1,
                                    'isfree': True,
                                    'init': 0.1,
                                    'units': '',
                                    'prior_function':priors.tophat,
                                    'prior_args': {'mini':0.0, 'maxi':0.1}})
    else:
        # redshift: fixed to sdss spectroscopic value (assumed to be 10pc if z=0, e.g. for absolute mags)
        model_params.append({'name': 'zred', 'N': 1,
                                'isfree': False,
                                'init': redshift})

    # --- SFH ---
    # FSPS parameter.  sfh=1 is a exponentially declining tau SFH, sfh=4 is a delayed-tau SFH
    model_params.append({'name': 'sfh', 'N': 1,
                            'isfree': False,
                            'init': 1,
                            'units': 'type'
                        })

    # Normalization of the SFH.  If the ``mass_units`` parameter is not supplied,
    # this will be in surviving stellar mass.  Otherwise it is in the total stellar
    # mass formed.
    model_params.append({'name': 'mass', 'N': 1,
                            'isfree': True,
                            'init': 1e10,
                            'init_disp': 1e9,
                            'units': r'M_\odot',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':1e6, 'maxi':1e12}})

    # Since we have zcontinuous=1 above, the metallicity is controlled by the
    # ``logzsol`` parameter.
    model_params.append({'name': 'logzsol', 'N': 1,
                            'isfree': True,
                            'init': 0,
                            'init_disp': 0.1,
                            'units': r'$\log (Z/Z_\odot)$',
                            'prior_function': priors.tophat,
                            'prior_args': {'mini':-3.0, 'maxi':1.0}})

    # FSPS parameter
    model_params.append({'name': 'tau', 'N': 1,
                            'isfree': True,
                            'init': 1.0,
                            'init_disp':0.1,
                            'units': 'Gyr',
                            'prior_function':priors.tophat,#priors.logarithmic,
                            'prior_args': {'mini':0.1, 'maxi':10}})

    # FSPS parameter (could change max to == t_age at this redshift)
    model_params.append({'name': 'tage', 'N': 1,
                            'isfree': True,
                            'init': 5.0,
                            'init_disp': 1.0,
                            'units': 'Gyr',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.101, 'maxi':14.0}})

    # FSPS parameter
    model_params.append({'name': 'sfstart', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'Gyr',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.1, 'maxi':14.0}})

    # FSPS parameter
    model_params.append({'name': 'tburst', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':1.3}})

    # FSPS parameter
    model_params.append({'name': 'fburst', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':0.5}})

    # --- Dust ---------
    # FSPS parameter
    model_params.append({'name': 'dust1', 'N': 1,
                            'isfree': True,
                            'init': 0.7,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':2.0}})

    # FSPS parameter (could couple to dust1, e.g. some constant fraction, through depends_on)
    model_params.append({'name': 'dust2', 'N': 1,
                            'isfree': False,
                            'init': 0.3,
                            'reinit': True,
                            #'depends_on' : dust2_func,
                            'init_disp': 0.3,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':2.0}})

    # FSPS parameter
    model_params.append({'name': 'dust_index', 'N': 1,
                            'isfree': False,
                            'init': -0.7,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':-1.5, 'maxi':-0.5}})

    # FSPS parameter
    model_params.append({'name': 'dust1_index', 'N': 1,
                            'isfree': False,
                            'init': -0.7,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':-1.5, 'maxi':-0.5}})

    # FSPS parameter
    model_params.append({'name': 'dust_tesc', 'N': 1,
                            'isfree': False,
                            'init': 7.0,
                            'units': 'log(Gyr)',
                            'prior_function_name': None,
                            'prior_args': None})

    # FSPS parameter
    model_params.append({'name': 'dust_type', 'N': 1,
                            'isfree': False,
                            'init': 0,
                            'units': 'index'})

    # FSPS parameter
    model_params.append({'name': 'add_dust_emission', 'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': 'index'})

    # FSPS parameter
    model_params.append({'name': 'duste_umin', 'N': 1,
                            'isfree': False,
                            'init': 1.0,
                            'units': 'MMP83 local MW intensity'})


    # --- Stellar Pops ------------
    # FSPS parameter
    model_params.append({'name': 'tpagb_norm_type', 'N': 1,
                            'isfree': False,
                            'init': 2,
                            'units': 'index'})

    # FSPS parameter
    model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': 'index'})

    # FSPS parameter
    model_params.append({'name': 'agb_dust', 'N': 1,
                            'isfree': False,
                            'init': 1,
                            'units': 'index'})

    # --- Nebular Emission ------

    # Here is a really simple function that takes a **dict argument, picks out the
    # `logzsol` key, and returns the value.  This way, we can have gas_logz find
    # the value of logzsol and use it, if we uncomment the 'depends_on' line in the
    # `gas_logz` parameter definition.
    #
    # One can use this kind of thing to transform parameters as well (like making
    # them linear instead of log, or divide everything by 10, or whatever.) You can
    # have one parameter depend on several others (or vice versa).  Just remember
    # that a parameter with `depends_on` must always be fixed.
    #def stellar_logzsol(logzsol=0.0, **extras):
    #    return logzsol

    # FSPS parameter
    model_params.append({'name': 'add_neb_emission', 'N': 1,
                         'isfree': False,
                         'init': True})

    # FSPS parameter
    model_params.append({'name': 'gas_logz', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': r'log Z/Z_\odot',
    #                        'depends_on': stellar_logzsol,
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':-2.0, 'maxi':0.5}})

    # FSPS parameter
    model_params.append({'name': 'gas_logu', 'N': 1,
                            'isfree': False,
                            'init': -2.0,
                            'units': '',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':-4, 'maxi':-1}})

    # --- Calibration ---------
    model_params.append({'name': 'phot_jitter', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'mags',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':0.2}})

    return model_params

def fitSDSSSpectrum(ind=12345):
    """ Run MCMC fit against a particular SDSS spectrum. """
    from prospect.models import sedmodel
    from prospect.io import write_results
    from prospect.sources import CSPBasis
    from prospect import fitting

    # load observational spectrum
    obs = load_obs(ind)

    # configuration
    run_params = {'verbose':True,
                  'debug':False,
                  # Fitter parameters
                  'nwalkers':128, # should be N*(nproc-1) where N is an even integer, Nproc=MPI tasks
                  'nburn':[32, 32, 64],
                  'niter':512,
                  #'do_powell': False,
                  #'ftol':0.5e-5, 'maxfev':5000,
                  'initial_disp':0.1,
                  # Data manipulation parameters
                  'logify_spectrum':False,
                  'normalize_spectrum':False,
                  'wlo':3750.0,
                  'whi':7200.0,
                  # SPS parameters
                  'zcontinuous': 1, # 1=continuous metallicity, 0=discretized zmet integers only
                  }


    # model
    model_params = load_model_params(redshift=obs['redshift'])
    model = sedmodel.SedModel(model_params)

    # SPS Model
    sps = CSPBasis(zcontinuous=run_params['zcontinuous'],compute_vega_mags=False)

    # setup
    initial_theta = model.rectify_theta(model.initial_theta)

    outFileBase = 'out_%d' % ind
    hf = h5py.File(outFileBase + '.hdf5','w')
    write_results.write_h5_header(hf, run_params, model)
    write_results.write_obs_to_h5(hf, obs)

    # no initial Powell guess
    postkwargs = {}
    pool = None # MPI

    powell_guesses = None
    pdur = 0.0
    initial_center = initial_theta.copy()
    initial_prob = None

    # mcmc sample
    tstart = time.time()
    out = fitting.run_emcee_sampler(lnprobfn, initial_center, model,
                                    postargs=[model,obs,sps],
                                    postkwargs=postkwargs, initial_prob=initial_prob,
                                    pool=pool, hdf5=hf, **run_params)
    esampler, burn_p0, burn_prob0 = out

    edur = time.time() - tstart
    print('done sampling in %.1f sec' % edur)

    # write pickles and hdf5
    write_results.write_pickles(run_params, model, obs, esampler, powell_guesses,
                                outroot=outFileBase, toptimize=pdur, tsample=edur,
                                sampling_initial_center=initial_center,
                                post_burnin_center=burn_p0,
                                post_burnin_prob=burn_prob0)

    write_results.write_hdf5(hf, run_params, model, obs, esampler, powell_guesses,
                             toptimize=pdur, tsample=edur,
                             sampling_initial_center=initial_center,
                             post_burnin_center=burn_p0,
                             post_burnin_prob=burn_prob0)

def debug_plot(ind=12345):
    # debug plot
    import corner
    from prospect.sources import CSPBasis
    import prospect.io.read_results as bread

    res, pr, mod = bread.results_from("out_12345_mcmc")
    #cornerfig = bread.subtriangle(res, start=0, thin=5)

    # load chains
    with h5py.File('out_%d.hdf5' % ind,'r') as f:
        chain = f['sampling']['chain'][()]
        wave = f['obs']['wavelength'][()]
        spec = f['obs']['spectrum'][()]

    # flatten chain into a flat list of samples
    ndim = chain.shape[2]
    samples = chain.reshape( (-1,ndim) )

    # recreate model
    sps = CSPBasis(zcontinuous=True,compute_vega_mags=False)
    #model = sedmodel.SedModel(model_params)

    # print percentiles 'answer'
    percs = np.percentile(samples, [16,50,84], axis=0)
    for i, label in enumerate(res['theta_labels']):
        print(label, percs[:,i])

    # corner plot
    fig = corner.corner(samples, labels=res['theta_labels'])
    fig.savefig('out_%d_triangle.pdf' % ind)
    plt.close(fig)

    # plot some chain models on top of data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda$ [Angstroms]')
    ax.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
    ax.set_ylim([1e-4,1e0])
    ax.set_yscale('log')

    ax.plot(wave, spec*1e6, '-', label='obs')

    for i in range(20):
        local_result = samples[i,:]
        spec, phot, mfrac = mod.mean_model(local_result, obs=res['obs'], sps=sps)

        ax.plot(wave, spec*1e6, '-', color='black', alpha=0.2, label='model %d' % i)

    fig.tight_layout()
    fig.savefig('out_%d_samples.pdf' % ind)
    plt.close(fig)

    import pdb; pdb.set_trace()

def debug_plot2(ind=12345):
    # debug plot raw obs vs initial guess model
    from prospect.models import sedmodel
    from prospect.sources import CSPBasis

    obs = load_obs(ind)

    # model
    model_params = load_model_params(redshift=obs['redshift'])
    model = sedmodel.SedModel(model_params)
    sps = CSPBasis(zcontinuous=True,compute_vega_mags=False)

    initial_theta = model.initial_theta
    print(initial_theta)
    initial_theta = model.rectify_theta(model.initial_theta)

    # start plot
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$\lambda$ [Angstroms]')
    ax.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
    #ax.set_ylim([1e-1,1e3])
    #ax.set_yscale('log')

    # obs
    ax.plot(obs['wavelength'], obs['spectrum']*1e6, '-', color='black', label='obs')

    # model 1
    spec, phot, mfrac = model.mean_model(initial_theta, obs=obs, sps=sps)
    ax.plot(obs['wavelength'], spec*1e6, '-', alpha=0.8, label='model initial')

    # model 2
    initial_theta[0] = 8e9
    spec, phot, mfrac = model.mean_model(initial_theta, obs=obs, sps=sps)
    ax.plot(obs['wavelength'], spec*1e6, '-', alpha=0.8, label='model mass=8e9')

    # model 3
    initial_theta[0] = 5e9
    spec, phot, mfrac = model.mean_model(initial_theta, obs=obs, sps=sps)
    ax.plot(obs['wavelength'], spec*1e6, '-', alpha=0.8, label='model mass=5e9')

    # model 4
    initial_theta[0] = 1e9
    spec, phot, mfrac = model.mean_model(initial_theta, obs=obs, sps=sps)
    ax.plot(obs['wavelength'], spec*1e6, '-', alpha=0.8, label='model mass=1e9')

    ax.legend()
    fig.tight_layout()
    fig.savefig('check_%d.pdf' % ind)
    plt.close(fig)

from prospect.likelihood import lnlike_spec, lnlike_phot, write_log

def lnprobfn(theta, model=None, obs=None, sps=None, verbose=True):
    """ Given a parameter vector theta, return the ln of the posterior. """
    lnp_prior = model.prior_product(theta)

    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    t1 = time.time()
    try:
        mu, phot, x = model.mean_model(theta, obs, sps=sps)
    except(ValueError):
        return -np.infty
    d1 = time.time() - t1

    #print('mean_model done [%.1f sec]' % d1)

    # Noise modeling
    #if spec_noise is not None:
    #    spec_noise.update(**model.params)
    #if phot_noise is not None:
    #    phot_noise.update(**model.params)
    vectors = {'spec': mu, 'unc': obs['unc'],
               'sed': model._spec, 'cal': model._speccal,
               'phot': phot, 'maggies_unc': obs['maggies_unc']}

    # Calculate likelihoods
    t2 = time.time()
    lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=None, **vectors) # spec_noise=spec_noise
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=None, **vectors) # phot_noise=phot_noise
    d2 = time.time() - t2

    if verbose:
        write_log(theta, lnp_prior, lnp_spec, lnp_phot, d1, d2)

    return lnp_prior + lnp_phot + lnp_spec
