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

from os.path import expanduser
from util.helper import sdss_decompose_specobjid

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

def fitSDSSSpectrum(ind=12345):
    """ Run MCMC fit against a particular SDSS spectrum. """
    from prospect.models import priors, sedmodel, model_setup
    from prospect.io import write_results
    from prospect.sources import CSPBasis
    tophat = priors.tophat
    from sedpy.observate import load_filters

    from prospect.io import write_results
    from prospect import fitting

    # load observational spectrum
    spec = loadSDSSSpectrum(ind=ind)

    # convert [10^-17 erg/cm^2/s/Ang] -> [10^-23 erg/cm^2/s/Hz] since flux_nu = (lambda^2/c) * flux_lambda
    # http://coolwiki.ipac.caltech.edu/index.php/Units, https://en.wikipedia.org/wiki/AB_magnitude
    flux_Jy = 1e-17 * spec['flux'] * 3.34e4 * (spec['wavelength']/1.0)**2.0
    flux_maggies = flux_Jy / 3631.0
    flux_nMgy = flux_maggies * 1e9

    # define observation
    obs = {}
    obs['wavelength'] = spec['wavelength'] # vacuum Angstroms
    obs['spectrum'] = flux_maggies # units of maggies
    obs['unc'] = spec['flux'] * 0.07 # todo! use sqrt(1/ivar)
    obs['mask'] = (obs['unc'] != 0.0) # optional spec (bool array, set False for ivar==0)

    obs['maggies'] = None # no broadband filter magnitudes
    obs['maggies_unc'] = None # no associated uncertanties
    obs['filters'] = None # no associated filters
    obs['phot_mask'] = None  # optional, no associated masks

    # configuration
    run_params = {'verbose':True,
                  'debug':False,
                  #'outfile':'output/demo_mock',
                  # Fitter parameters
                  'nwalkers':16,#128, # should be N*(nproc-1) where N is an even integer, Nproc=MPI tasks
                  'nburn':[8,8,16],#[32, 32, 64],
                  'niter':32,#512,
                  #'do_powell': False,
                  #'ftol':0.5e-5, 'maxfev':5000,
                  'initial_disp':0.1,
                  # Mock data parameters
                  #'snr': 20.0,
                  #'add_noise': False,
                  # Mock model parameters
                  #'mass': 1e9,
                  #'logzsol': -0.5,
                  #'tage': 12.,
                  #'tau': 3.,
                  #'dust2': 0.3,
                  # Data manipulation parameters
                  'logify_spectrum':False,
                  'normalize_spectrum':False,
                  'wlo':3750., 'whi':7200.,
                  # SPS parameters
                  'zcontinuous': 1,
                  }

    model_params = []

    # redshift: fixed to sdss spectroscopic value (assumed to be 10pc if z=0, e.g. for absolute mags)
    model_params.append({'name': 'zred', 'N': 1,
                            'isfree': False,
                            'init': spec['redshift']})#,
                            #'units': '',
                            #'prior_function':tophat,
                            #'prior_args': {'mini':0.0, 'maxi':4.0}})

    # --- SFH ---
    # FSPS parameter.  sfh=4 is a delayed-tau SFH
    model_params.append({'name': 'sfh', 'N': 1,
                            'isfree': False,
                            'init': 4,
                            'units': 'type'
                        })

    # Normalization of the SFH.  If the ``mass_units`` parameter is not supplied,
    # this will be in surviving stellar mass.  Otherwise it is in the total stellar
    # mass formed.
    model_params.append({'name': 'mass', 'N': 1,
                            'isfree': True,
                            'init': 1e10,
                            'init_disp': 1e8,
                            'units': r'M_\odot',
                            'prior_function':tophat,
                            'prior_args': {'mini':1e8, 'maxi':1e12}})

    # Since we have zcontinuous=1 above, the metallicity is controlled by the
    # ``logzsol`` parameter.
    model_params.append({'name': 'logzsol', 'N': 1,
                            'isfree': True,
                            'init': 0,
                            'init_disp': 0.1,
                            'units': r'$\log (Z/Z_\odot)$',
                            'prior_function': tophat,
                            'prior_args': {'mini':-1, 'maxi':0.19}})

    # FSPS parameter
    model_params.append({'name': 'tau', 'N': 1,
                            'isfree': False,
                            'init': 1.0,
                            'units': 'Gyr',
                            'prior_function':priors.logarithmic,
                            'prior_args': {'mini':0.1, 'maxi':100}})

    # FSPS parameter
    model_params.append({'name': 'tage', 'N': 1,
                            'isfree': True,
                            'init': 5.0,
                            'init_disp': 3.0,
                            'units': 'Gyr',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.101, 'maxi':14.0}})

    # FSPS parameter
    model_params.append({'name': 'sfstart', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'Gyr',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.1, 'maxi':14.0}})

    # FSPS parameter
    model_params.append({'name': 'tburst', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.0, 'maxi':1.3}})

    # FSPS parameter
    model_params.append({'name': 'fburst', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.0, 'maxi':0.5}})

    # --- Dust ---------
    # FSPS parameter
    model_params.append({'name': 'dust1', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.1, 'maxi':2.0}})

    # FSPS parameter
    model_params.append({'name': 'dust2', 'N': 1,
                            'isfree': False,
                            'init': 0.35,
                            'reinit': True,
                            'init_disp': 0.3,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.0, 'maxi':2.0}})

    # FSPS parameter
    model_params.append({'name': 'dust_index', 'N': 1,
                            'isfree': False,
                            'init': -0.7,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':-1.5, 'maxi':-0.5}})

    # FSPS parameter
    model_params.append({'name': 'dust1_index', 'N': 1,
                            'isfree': False,
                            'init': -1.0,
                            'units': '',
                            'prior_function':tophat,
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
                            'prior_function':tophat,
                            'prior_args': {'mini':-2.0, 'maxi':0.5}})

    # FSPS parameter
    model_params.append({'name': 'gas_logu', 'N': 1,
                            'isfree': False,
                            'init': -2.0,
                            'units': '',
                            'prior_function':tophat,
                            'prior_args': {'mini':-4, 'maxi':-1}})

    # --- Calibration ---------
    model_params.append({'name': 'phot_jitter', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            'units': 'mags',
                            'prior_function':tophat,
                            'prior_args': {'mini':0.0, 'maxi':0.2}})


    # SPS Model instance as global
    print('init sps')
    sps = CSPBasis(zcontinuous=run_params['zcontinuous'],compute_vega_mags=False)
    # GP instances as global
    spec_noise, phot_noise = None, None #model_setup.load_gp(**run_params)
    # Model as global
    print('init sed model')
    model = sedmodel.SedModel(model_params)
    # Obs as global
    #obs = model_setup.load_obs(**run_params)

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
    print('sampling...')
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

    hf.close()

    import pdb; pdb.set_trace()


def lnprobfn(theta, model=None, obs=None, sps=None, verbose=True):
    """ Given a parameter vector theta, return the ln of the posterior. """
    from prospect.likelihood import lnlike_spec, lnlike_phot, write_log

    #if model is None:
    #    model = global_model
    #if obs is None:
    #    obs = global_obs

    print('lnprobfn()')

    lnp_prior = model.prior_product(theta)

    print('prior_product done')

    if not np.isfinite(lnp_prior):
        return -np.infty

    # Generate mean model
    t1 = time.time()
    try:
        mu, phot, x = model.mean_model(theta, obs, sps=sps)
    except(ValueError):
        return -np.infty
    d1 = time.time() - t1

    print('mean_model done [%.1f sec]' % d1)
    import pdb; pdb.set_trace()

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

#def chisqfn(theta, model, obs):
#    """ Negative of lnprobfn for minimization, and also handles passing in keyword 
#    arguments which can only be postional arguments when using scipy minimize. """
#    return -lnprobfn(theta, model=model, obs=obs)
