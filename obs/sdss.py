"""
obs/sdss.py
  Observational data processing, reduction, and analysis (SDSS).
"""
from __future__ import (absolute_import,division,print_function,unicode_literals)
from builtins import *

import numpy as np
import h5py
import time
import requests
import os

import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from util.helper import pSplitRange

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

def loadSDSSSpectrum(ind, fits=False):
    """ Remotely acquire (via http) a single SDSS galaxy spectrum, according to the input index 
    which corresponds to the sdss_z0.0-0.1.hdf5 datafile (z<0.1 targets)."""
    basePath = os.path.expanduser('~') + '/obs/'
    savePath = basePath + 'spectra/'

    if not os.path.isdir(savePath): os.mkdir(savePath)

    f1 = basePath + 'sdss_z0.0-0.1.hdf5'
    f2 = basePath + 'sdss_objid_specobjid_z0.0-0.5.hdf5'

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

    # calculate mjd, fiberid, plateid
    p = sdss_decompose_specobjid(specobjid)
    r = {'ind':ind, 'objid':objid, 'specobjid':specobjid, 'logMass':logMass, 'redshift':redshift}

    # construct url
    fmt = 'fits' if fits else 'csv'
    #saveFilename = savePath + '/%d/spec-%04d-%d-%04d.%s' % (p['plate'],p['plate'],p['mjd'],p['fiberid'],fmt)

    url_base = 'https://dr13.sdss.org/optical/spectrum/view/data/format=%s/spec=lite?' % fmt
    url = url_base + 'mjd=%d&fiberid=%d&plateid=%d' % (p['mjd'],p['fiberid'],p['plate'])

    # acquire
    print(' ' + url)
    req = requests.get(url, headers={})

    if req.status_code != 200:
        print(' WARNING! Response code = %d, skipping!' % req.status_code)
        return None

    if not fits:
        # parse csv
        rows = req.text.split('\n')
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
    else:
        # fits, save binary
        assert 'content-disposition' in req.headers
        filename = req.headers['content-disposition'].split("filename=")[1]

        subSavePath = savePath + str(p['plate']) + '/'
        if not os.path.isdir(subSavePath): os.mkdir(subSavePath)

        with open(subSavePath + filename, 'wb') as f:
            f.write(req.content)

        # reload
        with pyfits.open(subSavePath + filename) as hdus:
            spec = np.array( hdus[1].data )

        # return in same format/units as csv, except we now also have ivar and wdisp
        r['wavelength'] = 10.0**spec['loglam']
        r['flux'] = spec['flux']
        r['ivar'] = spec['ivar']
        r['wdisp'] = spec['wdisp']
        r['skyflux'] = spec['sky']
        r['bestfit'] = spec['model']

    return r

def _writeLsfFile(spec, miles_fwhm_aa=2.54):
    """This method takes a spec file and returns the quadrature difference
    between the instrumental dispersion and the MILES dispersion, in km/s, as a
    function of wavelength
    """
    lightspeed = 2.998e5  # km/s
    # Get the SDSS instrumental resolution for this plate/mjd/fiber
    wave = spec['wavelength']
    dlam = np.gradient(wave)
    sigma_aa = spec['wdisp'] * dlam
    sigma_v = lightspeed * (sigma_aa / wave)

    # filter out some places where sdss reports zero dispersion
    good = sigma_v > 0
    wave, sigma_v = wave[good], sigma_v[good]

    # Get the miles velocity resolution function
    sigma_v_miles = lightspeed * miles_fwhm_aa / 2.355 / wave

    # Get the quadrature difference (zero and negative values are skipped by FSPS)
    dsv = np.sqrt(np.clip(sigma_v**2 - sigma_v_miles**2, 0, np.inf))

    # Restrict to regions where MILES is used
    good = (wave > 3525.0) & (wave < 7500)

    # Get the quadrature difference between the instrumental and MILES resolution
    wave, delta_v = wave[good], dsv[good]

    # Write the file
    lname = os.path.join(os.environ['SPS_HOME'],'data','lsf.dat')
    with open(lname, 'w') as out:
        for w, vel in zip(wave, delta_v):
            out.write('{:4.2f}   {:4.2f}\n'.format(w, vel))
    out.close()

    print(' WROTE [%s] careful of overlap.' % lname)

def load_obs(ind, run_params):
    """ Construct observational object with a SDSS spectrum, ready for fitting. """
    spec = loadSDSSSpectrum(ind=ind, fits=True)

    # convert [10^-17 erg/cm^2/s/Ang] -> [10^-23 erg/cm^2/s/Hz] since flux_nu = (lambda^2/c) * flux_lambda
    # http://coolwiki.ipac.caltech.edu/index.php/Units, https://en.wikipedia.org/wiki/AB_magnitude
    # 3.34e4 = 1/(c * 1e3 * Jy_mks) = 1/(c * 1e3 * 1e-26) = 1/(3e18 ang/s * 1e3 * 1e-26)
    fac = 1e-17 * 3.34e4 * (spec['wavelength'])**2.0 / 3631.0
    flux_maggies = spec['flux'] * fac
    #flux_Jy = flux_maggies * 3631.0
    #flux_nMgy = flux_maggies * 1e9

    # define observation
    obs = {}
    obs['wavelength'] = spec['wavelength'] # vacuum Angstroms
    obs['spectrum'] = flux_maggies # units of maggies

    if 'ivar' in spec:
        obs['unc'] = 1.0/np.sqrt(spec['ivar']) * fac
    else:
        print(' Warning: No actual variance.')
        obs['unc'] = spec['flux'] * 0.05

    obs['maggies'] = None # no broadband filter magnitudes
    obs['maggies_unc'] = None # no associated uncertanties
    obs['filters'] = None # no associated filters
    obs['phot_mask'] = None  # optional, no associated masks

    # lsf: such that we convolve the theoretical spectra with the instrument resolution (wave-dependent)
    _writeLsfFile(spec)

    # deredshift (fit in rest-frame below)
    obs['wavelength'] /= (1 + spec['redshift'])
    obs['spectrum'] /= (1 + spec['redshift']) # assuming spectrum is now f_nu

    # mask: bad variance, wavelength range, and lines
    obs['mask'] = (spec['ivar'] != 0.0)

    wavelength_mask = (obs['wavelength'] > run_params['wlo']) & (obs['wavelength'] < run_params['whi'])

    lines = """3715.0  3735.0  0.0  *[OII]      3726.0
               3720.0  3742.0  0.0  *[OII]      3728.8
               4065.0  4135.0  0.0  *Hdelta     4101.7
               4315.0  4370.0  0.0  *Hgamma     4340.5
               4840.0  4885.0  0.0  *Hbeta      4861.3
               4935.0  5040.0  0.0  *[OIII]     4958.9
               6285.0  6315.0  0.0  *[OI]       6300.3
               6510.0  6625.0  0.0  *Halpha+NII 6562.8
               6685.0  6770.0  0.0  *[SII]      6716.4
               7125.0  7142.0  0.0  [ArIII]     7135.8
               7315.0  7329.0  0.0  [OII]       7319.5
               7320.0  7335.0  0.0  [OII]       7330.2
               5185.0  5215.0  0.0  NI5199      5199.0
               5872.0  5916.0  0.0  NaD         5890.0"""

    emissionline_mask = np.zeros( len(obs['wavelength']), dtype=bool )

    for line in lines.split('\n'):
        wave_min = float( line.split()[0] )
        wave_max = float( line.split()[1] )
        emissionline_mask |= ((obs['wavelength'] > wave_min) & (obs['wavelength'] < wave_max ))

    emissionline_mask = ~emissionline_mask # such that 0=bad

    obs['mask'] &= wavelength_mask
    obs['mask'] &= emissionline_mask

    # auxiliary information
    for k in ['ind','objid','specobjid','logMass','redshift']:
        obs[k] = spec[k]

    return obs

def _dust2_from_dust1(dust1=None, **extras):
    """ Coupling function between dust1 and dust2 parameters. """
    return dust1 * 0.5

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
        # note we deredshift observed spectra, so this is only a residual redshift (and always left free)
        model_params.append({'name': 'zred', 'N': 1,
                                'isfree': True,
                                'init': 0.0,
                                'init_disp': 1e-4,
                                'disp_floor': 1e-4,
                                'units': 'residual redshift',
                                'prior_function':priors.tophat,
                                'prior_args': {'mini':-1e-3, 'maxi':1e-3}})

        # set the lumdist parameter to get the spectral units (and thus masses) correct. note that, by 
        # having both `lumdist` and `zred` we decouple the redshift from the distance (necessary since 
        # zred represents only the residual redshift in the fitting)
        from astropy.cosmology import Planck15 as cosmo
        lumDist = cosmo.luminosity_distance(redshift).value
        #print(' z=%.4f lumdist=%.4f Mpc' % (redshift,lumDist))

        model_params.append({'name': 'lumdist', 'N': 1,
                             'isfree': False,
                             'init': lumDist,
                             'units': 'Mpc'})


    # --- SFH ---
    # FSPS parameter.  sfh=1 is a exponentially declining tau SFH, sfh=4 is a delayed-tau SFH
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
                            'init': 4e10,
                            'init_disp': 1e10,
                            'units': r'M_\odot',
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':1e6, 'maxi':1e12}})

    model_params.append({'name': 'mass_units', 'N': 1,
                            'isfree': False,
                            'init': 'mformed',
                            })

    # Since we have zcontinuous=1 above, the metallicity is controlled by the
    # ``logzsol`` parameter.
    model_params.append({'name': 'logzsol', 'N': 1,
                            'isfree': True,
                            'init': -0.4,
                            'init_disp': 0.2,
                            'units': r'$\log (Z/Z_\odot)$',
                            'prior_function': priors.tophat,
                            'prior_args': {'mini':-1.5, 'maxi':0.5}})

    # FSPS parameter
    model_params.append({'name': 'tau', 'N': 1,
                            'isfree': True,
                            'init': 1.0,
                            'init_disp':0.1,
                            'units': 'Gyr',
                            'prior_function':priors.logarithmic,
                            'prior_args': {'mini':0.1, 'maxi':100}})

    # FSPS parameter (could change max to == t_age at this redshift)
    model_params.append({'name': 'tage', 'N': 1,
                            'isfree': True,
                            'init': 8.0,
                            'init_disp': 2.0,
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
    # FSPS parameter (0=cf00 plaw, 1=CCM, 4=Kreik and Conroy)
    model_params.append({'name': 'dust_type', 'N': 1,
                            'isfree': False,
                            'init': 0,
                            'units': 'index'})

    # FSPS parameter
    model_params.append({'name': 'dust1', 'N': 1,
                            'isfree': True,
                            'init': 0.7,
                            'reinit': True,
                            'units': '',
                            'init_disp': 0.2,
                            'disp_floor': 0.1,
                            'prior_function':priors.tophat,
                            'prior_args': {'mini':0.0, 'maxi':2.0}})

    # FSPS parameter (could couple to dust1, e.g. some constant fraction, through depends_on)
    model_params.append({'name': 'dust2', 'N': 1,
                            'isfree': False,
                            'init': 0.0,
                            #'reinit': True,
                            'depends_on' : _dust2_from_dust1,
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
                         'init': False})

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

    # --- Kinematics --------
    model_params.append({'name': 'smoothtype', 'N': 1,
                            'isfree': False,
                            'init': 'vel',
                            'units': 'do velocity smoothing',
                            })

    model_params.append({'name': 'sigma_smooth', 'N': 1,
                            'isfree': True,
                            'init': 200.0,
                            'init_disp': 100.0,
                            'disp_floor': 50.0,
                            'units': 'km/s',
                            'prior_function': priors.logarithmic,
                            'prior_args': {'mini': 50, 'maxi': 400}})

    model_params.append({'name': 'fftsmooth', 'N': 1,
                            'isfree': False,
                            'init': True,
                            'units': 'use fft for smoothing',
                            })

    # --- Photometric Calibration ---------
    #model_params.append({'name': 'phot_jitter', 'N': 1,
    #                        'isfree': False,
    #                        'init': 0.0,
    #                        'units': 'mags',
    #                        'prior_function':priors.tophat,
    #                        'prior_args': {'mini':0.0, 'maxi':0.2}})

    # --- Spectroscopic Calibration --------
    #polyeqn = 'ln(f_tru/f_obs)_j=\sum_{i=1}^N poly_coeffs_{i-1} * ((lambda_j - lambda_min)/lambda_range)^i'
    # Set the order of the polynomial.  The highest order will be \lambda^npoly
    npoly = 2
    # for setting min/max on the polynomial coefficients.
    polymax = 0.1 / (np.arange(npoly) + 1)

    model_params.append({'name': 'poly_coeffs', 'N': npoly,
                            'isfree': False,
                            'init': np.zeros(npoly),
                            'init_disp': polymax / 5.0,
                            'units': '', #polyeqn,
                            'prior_function':priors.tophat,
                            'prior_args': {'mini': 0 - polymax, 'maxi': polymax}})

    model_params.append({'name': 'cal_type', 'N': 1,
                            'isfree': False,
                            'init': 'exp_poly',
                            'units': 'switch for whether to use exponential of polynomial for calibration'
                             })

    return model_params

def fitSDSSSpectrum(ind=12345, savePath=None):
    """ Run MCMC fit against a particular SDSS spectrum. """

    # test: NGC3937 z=0.02221 objid=1237667916491325445 ind=280692
    # plate=2515  mjd=54180 fiber=377 specobjid=2831741964808906752
    # test: NGC5227 z=0.01745 objid=1237651735230545966 ind=49455
    # plate=528   mjd=52022 fiber=137 specobjid=594512843009714176

    from prospect.models import sedmodel
    from prospect.io import write_results
    from prospect.sources import CSPSpecBasis
    from prospect import fitting

    # configuration
    run_params = {'verbose':True,
                  'debug':False,
                  # Fitter parameters
                  'nwalkers':128, # should be N*(nproc-1) where N is an even integer, Nproc=MPI tasks
                  'nburn':[32, 32, 64, 128],
                  'niter':128, # note: total number of retained samples = niter*nwalkers
                  #'do_powell': False,
                  #'ftol':0.5e-5, 'maxfev':5000,
                  'initial_disp':0.1,
                  # Data manipulation parameters
                  'logify_spectrum':False,
                  'normalize_spectrum':False,
                  'wlo':3750.0,
                  'whi':7000.0, # spectral libraries are too low resolution at higher wavelengths
                  # SPS parameters
                  'zcontinuous': 1, # 1=continuous metallicity, 0=discretized zmet integers only
                  }

    # load observational spectrum
    obs = load_obs(ind, run_params)

    # model
    # default: 7 free parameters: z_red, mass, logzsol, tau, tage, dust1, sigma_smooth
    # or: 8 free parameters: z_red, mass, logzsol, tau, tage, dust1, dust2, sigma_smooth
    # (cannot force any dust>0, leads to issues with dust-free early-types? maybe better to 
    # couple them with a constant ratio instead of letting them float free)
    model_params = load_model_params(redshift=obs['redshift'])
    model = sedmodel.SedModel(model_params)

    # SPS Model
    sps = CSPSpecBasis(zcontinuous=run_params['zcontinuous'],compute_vega_mags=False)

    #assert sps.ssp.libraries[1] == 'miles'
    #sps.ssp.params['smooth_lsf'] = True
    #sps.ssp.set_lsf(wave, delta_v) # future dreams

    # setup
    initial_theta = model.rectify_theta(model.initial_theta)

    basePath = savePath if savePath is not None else ''
    outFileBase = basePath + 'chains_%d' % ind
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

def fitSDSSSpectra(pSplit):
    """ Fit a pSplit work divided segment of the entire z<0.1 SDSS selection. Results are saved 
    individually, one file per galaxy, which can be concatenated later. """
    zBin = 'z0.0-0.1'
    f1 = os.path.expanduser('~') + '/obs/sdss_%s.hdf5' % zBin

    basePath = os.path.expanduser('~') + '/obs/mcmc_fits_%s/' % zBin
    if not os.path.isdir(basePath): os.mkdir(basePath)

    # get global list of all object IDs of interest
    with h5py.File(f1,'r') as f:
        nObjIDs = f['objid'].size
        objIDRange = [0, nObjIDs-1]

    # divide range, decide indRange local to this task
    indRange = pSplitRange( objIDRange, pSplit[1], pSplit[0] )

    print('Total # galaxies: %d, processing [%d] now, range [%d - %d]...' % \
        (nObjIDs,indRange[1]-indRange[0]+1,indRange[0],indRange[1]))

    # process in a serial loop
    for index in np.arange( indRange[0], indRange[1] ):
        savePath = basePath + str(index % 100) + '/'
        if not os.path.isdir(savePath): os.mkdir(savePath)

        import pdb; pdb.set_trace()

        fitSDSSSpectrum(ind=index, savePath=savePath)

def plotSingleResult(ind=12345, sps=None):
    """ Load the results of a single MCMC fit, print the answer, render a corner plot of the joint 
    PDFs of all the parameters, and show the original spectrum as well as some ending model spectra. """
    import corner
    from prospect.sources import CSPSpecBasis
    import prospect.io.read_results as bread

    fileBase = "chains_%d" % ind #"ngc3937_1489154818"
    percs = [16,50,84]

    # load pickle
    res, pr, mod = bread.results_from(fileBase + "_mcmc")

    # load chains from hdf5
    with h5py.File(fileBase + ".hdf5",'r') as f:
        chain = f['sampling']['chain'][()]
        wave = f['obs']['wavelength'][()]
        spec = f['obs']['spectrum'][()]

    # flatten chain into a linear list of samples
    ndim = chain.shape[2]
    samples = chain.reshape( (-1,ndim) )

    # print median as the answer, as well as standard percentiles
    percs = np.percentile(samples, percs, axis=0)
    for i, label in enumerate(res['theta_labels']):
        print(label, percs[:,i])
    print('Number of (samples,free_params): ',samples.shape)

    # (A) corner plot
    if 1:
        #quantiles = np.array(percs)/100.0
        fig = corner.corner(samples, labels=res['theta_labels'])
        fig.savefig('out_%d_triangle.pdf' % ind)
        plt.close(fig)

    # (B) plot some of the final chain models on top of data
    if 1:
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.set_xlabel('$\lambda$ [Angstroms]')
        ax.set_ylabel('$F_\lambda$ [$\mu$Mgy]')
        ax.set_title('ind=%d logMass=%.1f z=%.3f specobjid=%d' % \
            (ind,res['obs']['logMass'],res['obs']['redshift'],res['obs']['specobjid']))
        ax.set_ylim([5e-2,5e0])
        ax.set_yscale('log')

        # recreate model
        if sps is None:
            sps = CSPSpecBasis(zcontinuous=True,compute_vega_mags=False)
            #model = sedmodel.SedModel(model_params)

        ax.plot(wave, spec*1e6, '-', label='obs')

        for i in range(20):
            local_result = samples[i,:]
            spec, phot, mfrac = mod.mean_model(local_result, obs=res['obs'], sps=sps)

            ax.plot(wave, spec*1e6, '-', color='black', alpha=0.2, label='model %d' % i)

        fig.tight_layout()
        fig.savefig('out_%d_samples.pdf' % ind)
        plt.close(fig)

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
